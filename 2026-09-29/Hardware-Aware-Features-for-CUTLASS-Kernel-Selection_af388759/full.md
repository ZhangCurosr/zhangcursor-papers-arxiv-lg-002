# Hardware-Aware Features for CUTLASS Kernel Selection

Shriram Chandran<sup>∗†</sup> ETH Zurich Zurich, Switzerland

Alexandru Calotoiu ETH Zurich Zurich, Switzerland

Dominic Rinderer<sup>†</sup> ETH Zurich Zurich, Switzerland

Marcin Copik ETH Zurich Zurich, Switzerland

Yakup Budanaz ETH Zurich Zurich, Switzerland

Torsten Hoefler ETH Zurich Zurich, Switzerland

## Abstract

GPU libraries such as CUTLASS expose tens of thousands of semantically equivalent kernels for a single operation, making exhaustive autotuning expensive and execution-free selection dificult. Existing analytical selectors require hand-designed performance rules, while learned selectors operate on raw configuration parameters and must infer hardware consequences from data. We introduce a hardware-aware representation for CUTLASS kernel selection that augments candidate configurations with statically computable estimates of induced hardware behavior. We construct a dataset of 4.9 million CUTLASS kernels and train gradient-boosted and neural learning-to-rank models to rank candidates within each problem. On held-out exhaustive evaluation problems, hardware-aware representations reduce selection regret by up to 40% relative to structural baselines and 64.2% relative to NVIDIA’s matrix-multiply heuristics. We further evaluate data-eficient cross-precision and epilogue-fusion transfer within CUTLASS GEMM, showing that explicitly representing candidate-induced hardware behavior provides a useful inductive bias for learned kernel selection.

## 1 Introduction

Modern GPU libraries expose increasingly large spaces of implementations for a single operation. CUTLASS [21] is a particularly expressive instance of this problem – a CUTLASS GEMM may be implemented in over sixty thousand ways using diferent tile shapes, instruction shapes, pipeline depths, schedules, cluster configura tions, tile schedulers, and epilogue implementations, interacting non-trivially with the problem dimensions, layouts, data types, accumulation types, and target architecture [20]. The best configuration is determined by the interaction of these efects rather than by any one parameter in isolation. This flexibility is valuable to kernel developers, but it also creates a very large configuration space. Selecting an eficient implementation is therefore a central requirement for achieving near-peak performance on contemporary accelerators [4]. Empirical autotuning, by compiling and bench marking candidate kernels, is expensive [1, 3, 19]. Compilation and measurement must be repeated for each target GPU, datatype, epilogue, and problem shape. Empirical selection becomes impossible when a library requires several diferent GEMM shapes or must make selections dynamically. This motivates selectors that can rank candidates without executing them [3].

Existing execution-free selectors follow two main approaches. Analytical methods encode manually designed performance rules, such as preferences for tile sizes, occupancy limits, or wave quantization based on the problem instance [19, 26]. These methods are fast but brittle, require substantial architecture-specific knowledge for development, and must be updated as hardware and kernel libraries evolve. Learned autotuners [16, 25] reduce the need for manual rules, but typically represent candidates using raw problem dimensions and template parameters, leaving the model to then infer how those parameters afect hardware behaviour. Thus, raw parameters are a poor description of these efects.

We propose hardware-aware representations for kernel selection. Each candidate is represented using its structural configuration together with inexpensive estimates of its induced hardware behaviour. We construct a 4.9M-kernel training corpus, train ranking models, score the entire valid catalogue at inference, and obtain 6.2% mean selection regret, compared with 17% for nvMMH (nvMatmulHeuristics, NVIDIA’s matrix multiply heuristics library), and 71.5% for random selection – a 64.2% and 91.3% relative reduction in selection regret, respectively.

Our approach combines the strengths of analytical selectors and measurement-driven learning: hardware behavior is exposed explicitly through static, mechanistic features, while the model learns how these efects interact from measured data rather than relying on a hand-written performance model. This gives the selector the best of both worlds: strong architectural inductive bias without brittle decision rules, and data-driven adaptability without forcing the model to rediscover hardware behavior from raw parameters. As a result, the same representation supports accurate selection, improved data eficiency, and transfer across related kernel spaces, establishing hardware-aware feature design as a practical foundation for learned performance modeling on modern accelerators.

## 2 Background

## 2.1 GPU Execution Model

An NVIDIA GPU consists ofmany streaming multiprocessors (SMs), each containing the compute units and on-chip storage used for execution. A kernel launches a grid of thread blocks, and each block is assigned to an SM independently. Threads within a block execute in groups of 32 called warps. Warps can cooperate through on-chip shared memory and synchronization, while registers are private to individual threads. The output of a kernel is therefore determined not only by the arithmetic performed, but also by how work, data, and synchronization are distributed across threads, warps, blocks, and the memory hierarchy.

For GEMM, the main data movement path is from global memory through shared memory to registers. A thread block typically loads tiles of the input matrices into shared memory, computes on those tiles using registers as accumulators, and eventually stores its output tile back to global memory. Shared memory provides lower-latency reuse within a block, while the L2 cache and device memory serve data shared across blocks and data that does not fit on chip. The amount of work and data assigned to a block determines its arithmetic intensity, memory trafic, shared-memory footprint, register usage, and ability to keep the SM occupied.

Only a limited number of blocks can reside on an SM at once. This limit is determined by the block’s register and shared-memory requirements, its thread count, and architectural limits on blocks and warps. Higher residency can help hide memory and instruction latency, but increasing the amount of work assigned to a block often increases resource usage and can reduce the number of simultaneously resident blocks. Consequently, the best GEMM configuration must balance reuse and computational eficiency against resource pressure.

## 2.2 Hopper Execution Mechanisms

Our primary evaluation targets NVIDIA Hopper GPUs, whose SM90a architecture introduces several mechanisms that substantially afect GEMM performance [12]. Hopper tensor-core operations are issued using warpgroup matrix multiply-accumulate (WGMMA) instructions, comprising four consecutive warps, or 128 threads, cooperating to issue an asynchronous matrix operation. The instruction shape and warpgroup organization impose further constraints on the tile shapes that a kernel can use. Hopper also provides the Tensor Memory Accelerator (TMA), which asynchronously transfers multidimensional tensor regions between global memory and shared memory, saving several individual thread address computations and data loads.

TMA and WGMMA enable warp-specialized pipelines [18], where diferent warps assume diferent duties. Producer warps issue asynchronous TMA transfers, while consumer warpgroups perform WGMMA on previously loaded tiles. The shared-memory bufer is divided into a multi-stage pipeline, allowing memory movement to run ahead of computation. Increasing the number of stages can hide transfer latency but increase shared-memory consumption.

Hopper additionally supports thread-block clusters, which can cooperate through distributed shared memory, or through TMA multicast which can deliver the same data to multiple blocks. Clusters can reduce memory trafic but impose scheduling and residency constraints. Whether clustering is beneficial therefore depends on the problem shape, tile decomposition, and resource requirements of the candidate kernel.

These mechanisms make the relationship between a CUTLASS configuration and hardware performance highly indirect. A configuration parameter such as pipeline depth, schedule, or cluster shape changes several hardware-level properties simultaneously. The resulting efects cannot generally be inferred from the parameter value alone.

## 2.3 GEMM Tiling and CUTLASS

We consider the general matrix multiplication $C = \alpha A B + \beta C$ , where $A \in \mathbb { R } ^ { M \times K } , B \in \mathbf { \bar { \mathbb { R } } } ^ { K \times N }$ , and $C \in \mathbb { R } ^ { \hat { M } \times N }$ . The operation performs (2���) floating-point operations. A GPU implementation partitions � into output tiles. A thread block computes one or more such tiles by iterating over the reduction dimension in chunks.

![](images/84c9cd48f3a84406ab6d3a301f6fcebe7886ba50c12c3f7047c11c263d048c1f.jpg)  
Figure 1: H100 Programming Model

A candidate configuration is characterized primarily by a tile shape $( T _ { M } , T _ { N } , T _ { K } )$ . The first two dimensions determine the tile computed by a block, while $T _ { K }$ determines the amount of the reduction processed in one mainloop iteration. Each iteration loads an $T _ { M } \times T _ { K }$ tile of � and a $T _ { K } \times T _ { N }$ tile of $B ,$ performs a matrix multiply-accumulate, and advances to the next �-tile. The number of iterations is thus $\lceil K / T _ { K } \rceil$ . The tile shape therefore afects the number of output tiles, boundary tile waste, efective arithmetic intensity, memory trafic, shared-memory and register usage, and the number ofmainloop iterations. For one shape, our BF16 enumerator exposes 534,600 raw combinations per layout before validity filtering, and 2,138,400 kernels throughout.

CUTLASS builds GEMM kernels by composing reusable mainloop and epilogue components. The mainloop loads input tiles and performs the matrix multiplication, while the epilogue applies operations such as scaling, bias addition, activation, residual addition, type conversion, and the final store. CUTLASS exposes all aforementioned configurations as template parameters, and these choices are coupled – for e.g., a larger tile may increase data reuse but increase register and shared-memory usage. More pipeline stages can make the pipeline more eficient, but consume additional resources and reduce occupancy. A schedule may require a particular tile divisibility or warpgroup arrangement. Clusters may reduce redundant loads while limiting which blocks can be resident together. Epilogues add their own computation, memory trafic, and resource requirements, and can change which mainloop configuration is best. Not every combination of choices produces a valid kernel. CUTLASS implicitly imposes architectural and implementation constraints on several of the parameters. The valid candidate set is therefore a tiny constrained subset of the cartesian product of the parameters. Static CUTLASS legality and shared-memory checks retain around 11.36%, leaving a catalogue of around 61,000 per layout (243,000 throughout), on the basis of which the selector must therefore rank candidates.

The configuration does not directly specify performance – making selecting an eficient candidate from this valid set highly nontrivial. On a GH200, measured CUTLASS throughputs on BF16 GEMMs in our sweep span 0.01–763 Tflop/s depending on problem size and configuration. For one given GEMM shape and data layout, the empirical best and worst valid kernels can difer by a factor of 1,070× in throughput (median across 68 evaluated groups; 3.9 million measured pairs), a uniform random pick achieves only 28.5% of the in-group oracle on average, and choosing a static best configuration only 71%. Less than 2.4% of configurations get performance within 5% of the best kernel on average. Mis-selection is therefore not a rounding error; it can waste most of the hard ware budget on every GEMM in a compiled program. Exhaustively compiling and benchmarking the full catalogue is impractical at scale: our evaluation set contains only 17 shapes, yet requires a benchmarking sweep over the 3.9 million distinct (kernel, shape) GPU measurements (68 unique groups, up to 61,000 candidates per group). Compiling the heavily templated configuration catalogue once alone costs on the order of hours of CPU time even before any GPU benchmarking begins.

## 2.4 A Generated Kernel

<table><tr><td>1</td><td>11</td><td>unrelated generated boilerplate omitted (includes, CUDA/</td><td></td></tr><tr><td></td><td colspan="3">CUTLASS checks)</td></tr><tr><td>2</td><td colspan="3"></td></tr><tr><td>3</td><td colspan="3">// Config: cutlass3x_sm90_tensorop_bf16_bf16_f32_bf16_bf16_</td></tr><tr><td>4</td><td colspan="3">11 128x128x64_1x1x1_4_tnn_align64_warpspecialized_epi_tma</td></tr><tr><td>5</td><td colspan="3"></td></tr><tr><td>6</td><td colspan="3">namespace Kernel_0 {</td></tr><tr><td>7</td><td colspan="3"></td></tr><tr><td>8</td><td>using</td><td>ElementA</td><td>= cutlass::bfloat16_t;</td></tr><tr><td>9</td><td>using</td><td>LayoutATag</td><td>= cutlass::layout::RowMajor;</td></tr><tr><td>10</td><td>constexpr int</td><td>AlignmentA = 64;</td><td></td></tr><tr><td>11</td><td colspan="3"></td></tr><tr><td>12</td><td>using</td><td>ElementB</td><td>= cutlass::bfloat16_t;</td></tr><tr><td>13</td><td>using</td><td>LayoutBTag</td><td>= cutlass::layout::ColumnMajor;</td></tr><tr><td>14</td><td>constexpr int</td><td>AlignmentB</td><td>= 64;</td></tr><tr><td>15</td><td colspan="3"></td></tr><tr><td>16</td><td>using</td><td>ElementD</td><td>= cutlass::bfloat16_t;</td></tr><tr><td>17</td><td>using</td><td>ElementC</td><td>= cutlass::bfloat16_t;</td></tr><tr><td>18</td><td>using</td><td>LayoutCTag</td><td>= cutlass::layout::ColumnMajor;</td></tr><tr><td>19</td><td>using</td><td>LayoutDTag</td><td>= cutlass::layout::ColumnMajor;</td></tr><tr><td>20</td><td>constexpr int</td><td>AlignmentD = 64;</td><td></td></tr><tr><td>21</td><td>constexpr int</td><td>AlignmentC = 64;</td><td></td></tr><tr><td>22</td><td colspan="3"></td></tr><tr><td>23</td><td colspan="2">using ElementAccumulator</td><td>= float;</td></tr><tr><td>24</td><td colspan="2">using ElementCompute</td><td>= float;</td></tr><tr><td>25</td><td colspan="2">using ArchTag</td><td>= cutlass::arch::Sm90;</td></tr><tr><td>26</td><td colspan="2">using OperatorClass</td><td>= cutlass::arch::OpClassTensorOp;</td></tr><tr><td>27</td><td colspan="3"></td></tr></table>

```haskell
28 using MmaTileShape = Shape < _128 , _128 , _64 >;
29 using ClusterShape = Shape <_1 , _1 , _1 >;
30 using StageCount = cutlass :: gemm :: collective ::
StageCount <4 >;
31 using TileSchedulerType = cutlass :: gemm ::
PersistentScheduler ;
32
33 using MainloopScheduleType = cutlass :: gemm ::
KernelTmaWarpSpecialized ;
34 using EpilogueScheduleType = cutlass :: epilogue ::
TmaWarpSpecialized ;
35 using EpilogueTile = cutlass :: epilogue :: collective ::
EpilogueTileAuto ;
36
37 using CollectiveEpilogue = typename cutlass :: epilogue ::
collective :: CollectiveBuilder <
38 ArchTag , OperatorClass ,
39 MmaTileShape , ClusterShape ,
40 EpilogueTile ,
41 ElementAccumulator , ElementAccumulator ,
42 ElementC , LayoutCTag , AlignmentC ,
43 ElementD , LayoutDTag , AlignmentD ,
44 EpilogueScheduleType
45 >:: CollectiveOp ;
46
47 using CollectiveMainloop = typename cutlass :: gemm :: collective
:: CollectiveBuilder <
48 ArchTag , OperatorClass ,
49 ElementA , LayoutATag , AlignmentA ,
50 ElementB , LayoutBTag , AlignmentB ,
51 ElementAccumulator
52 MmaTileShape , ClusterShape ,
53 StageCount ,
54 MainloopScheduleType
55 >:: CollectiveOp ;
56
57 using GemmKernel = cutlass :: gemm :: kernel :: GemmUniversal <
58 Shape <int ,int ,int ,int > ,
59 CollectiveMainloop ,
60 CollectiveEpilogue ,
61 TileSchedulerType >;
62
63 using Gemm = cutlass :: gemm :: device :: GemmUniversalAdapter <
GemmKernel >;
64 }
```

The snippet above is a representative CUTLASS kernel: BF16 operands with FP32 accumulation, tile 128 × 128 × 64, a 1 × 1 cluster, four mainloop pipeline stages, the plain TMA warp-specialized mainloop, a TMA warp-specialized epilogue, persistent tile scheduling, and the TN operand layout (A row-major, B column-major). The configuration name is recorded in the generated comment header.

Reading top to bottom: ElementA/B/C and ElementAccumulator fix the operand and accumulation types; LayoutA/B are the searched operand layouts while LayoutC/D are always column-major in our generator (rowmajor � is obtained ofline via the transpose–swap identity $C ^ { \top } = B ^ { \top } A ^ { \top } )$ . AlignmentA/B/C are derived as 128/sizeof(element) bytes of vectorization. ArchTag and OperatorClass pin the kernel to SM90 tensor-core WGMMA. MmaTileShape and ClusterShape are the thread-block tile $T _ { M } \times T _ { N } \times T _ { K }$ and cluster $C _ { M } \times C _ { N } \times C _ { K } ;$ StageCount sets the software-pipelined mainloop depth. MainloopScheduleType, EpilogueScheduleType, and TileSchedulerType select the Hopper TMA warp-specialized mainloop variant, epilogue implementation, and persistent vs. Stream-K tile scheduling. The two CollectiveBuilder aliases materialize epilogue and mainloop collectives; GemmKernel and Gemm wrap them into a universal device GEMM. The setup excerpt shows the runtime path our benchmark uses: packed CuTe strides, GemmUniversalMode::kGemm, alpha/beta epilogue scalars, a hardware-info struct, then can\_implement and initialize before timed run. Every generated source has essentially the same structure; the search space is formed by varying a small set of template/configuration choices while leaving the GEMM semantics unchanged. Table 1 lists the configuration space.

Table 1: Configuration knobs exposed by the BF16 SM90 generator (generate\_search\_space).
<table><tr><td>Knob</td><td>Values / domain</td><td>Meaning</td></tr><tr><td>tile_m</td><td>{64, 128, 192, 256}</td><td>Thread-block M tile extent</td></tr><tr><td>tile_n</td><td>{16, 32, 48, 64, 80, 96, 128, 192, 256}</td><td>Thread-block N tile extent</td></tr><tr><td>tile_k</td><td>{32, 64, 128, 256, 512}</td><td>Thread-block K tile / pipeline atom</td></tr><tr><td>stages</td><td>{2, . . ., 12}</td><td>Mainloop pipeline stage count</td></tr><tr><td>cluster_m, cluster_n</td><td>{1, 2, 4, 8, 16},  $C _ { M } C _ { N } \leq 1 6$ </td><td>CTA cluster shape in M and N</td></tr><tr><td>cluster_k</td><td>fixed 1</td><td>K-clustering not searched</td></tr><tr><td>kernel_schedule</td><td>WS / Pingpong / Cooperative</td><td>TMA warp-specialized mainloop policy</td></tr><tr><td>epilogue_schedule</td><td>NoSmem / TMA / TMA-Cooperative</td><td>Epilogue store path</td></tr><tr><td>scheduler</td><td>Persistent / Stream-K</td><td>Tile scheduling across the output grid</td></tr><tr><td>layout</td><td>TN, TT, NN, NT</td><td>Operand layouts for A and B</td></tr><tr><td colspan="3">Fixed (not searched independently)</td></tr><tr><td>Operand / acc. types</td><td>BF16 in, FP32 acc. (BF16 sweep)</td><td>CUTLASS element aliases</td></tr><tr><td>alignment_* C/D layout</td><td>128 bit / element size ColumnMajor</td><td>TMA vectorization requirement</td></tr><tr><td>ArchTag / OpClass</td><td>Sm90 / TensorOp</td><td>Row-major output via transpose trick Target architecture</td></tr><tr><td>EpilogueTile</td><td>EpilogueTileAuto</td><td></td></tr><tr><td></td><td></td><td>Sub-tile shape chosen by CUTLASS</td></tr></table>

Table 2: Static validity constraints for BF16 SM90 GEMM configurations.
<table><tr><td>Constraint</td><td>Reason</td></tr><tr><td>Stream-K ⇒ cooperative mainloop</td><td>CUTLASS Stream-K scheduler requires the cooperative TMA mainloop.</td></tr><tr><td>Cooperative mainloop ⇒  $T _ { M } \equiv 0$  (mod 128)</td><td>Two consumer warpgroups split the M tile; each half must stay 64-row WGMMA aligned.</td></tr><tr><td> $S _ { \mathrm { m a i n l o o p } }$  + Sepi +  $2 \mathrm { K i B } \leq 2 2 7 \mathrm { K i B }$ </td><td>Hopper shared-memory capacity per CTA; Sepi depends on epilogue schedule and tile via estimate_epilogue_smem_bytes.</td></tr><tr><td>If A is column-major:  $T _ { K } \le 2 5 6 C _ { N }$ </td><td>TMA smem box K extent is multicast-split across the N cluster dimension.</td></tr><tr><td>If B is row-major:  $T _ { K } \leq 2 5 6 C _ { M }$ </td><td>Symmetric K-box limit along the M cluster dimension for the transposed operand.</td></tr></table>

## 2.5 Configuration Space

For a fixed GEMM problem shape, BF16 precision, and operandlayout tag, the generator varies tile geometry, cluster shape, pipeline depth, mainloop and epilogue schedules, and the tile scheduler. Operand element types, accumulator type, architecture tag, operator class, �/� layout, alignments, and epilogue-tile sizing are not independent search axes: they are fixed by the precision tuple or left to CUTLASS defaults (EpilogueTileAuto). Rasterization order, swizzle patterns, and classic split-� are likewise not exposed by our API; Stream-K is the only searched alternative to persis tent scheduling, and CUTLASS resolves remaining lowering details internally.

The raw Cartesian product over the searched axes alone contains 2,138,400 combinations. Static validity filtering leaves 242,971 BF16 configurations.

## 2.6 Validity Constraints

The Cartesian product contains many combinations that cannot instantiate as valid SM90 kernels. Enumeration is therefore followed by our set of static predicates; we also statically remove wasteful

candidates with $T _ { N } > 2 M$ or $T _ { N } > 2 N .$ . Table 2 lists the constraints active for BF16.

## 3 From CUTLASS Candidates to Hardware-Aware Features

For each candidate configuration, we compute a feature map entirely from the problem description, configuration parameters, and hardware constants. This constraint is essential for an executionfree selector. To decide which execution efects are worth representing, we first profile kernels.

## 3.1 Profiling Analysis for Feature Construction

The purpose of profiling is to identify execution efects that distinguish fast and slow CUTLASS configurations and can subsequently be approximated statically.

3.1.1 Profiling setup. We collected 408 Nsight Compute profiles of BF16 CUTLASS GEMM kernels on GH200 across 17 problem shapes. Profiles are associated with fixed (�, � , �, layout) groups, within which only the CUTLASS candidate configuration changes. For every profiled candidate, we pair the collected hardware counters with the throughput measured by our benchmark harness.

Because many hardware counters scale strongly with the amount of work in the GEMM, correlations computed over all profiles can be misleading. For example, larger problems can simultaneously execute more tensor-core instructions, generate more memory traffic, and achieve higher sustained throughput. To separate these problem-size efects from candidate quality, for each metric � we compute

![](images/d5fabd32732f3f1fc44ce1b35e0b7f5548c66aa3b578aa24c6e93c5a4ce2b1fb.jpg)  
Figure 2: Overview of our execution-free CUTLASS kernel selector, which maps each problem and candidate catalogue to hardware-aware features and ranks kernels without benchmarking.

$$
\rho _ { g } ( x ) = \mathrm { S p e a r m a n } \Big ( \{ x ( c ) \} _ { c \in C _ { g } } , \{ T ( c ) \} _ { c \in C _ { g } } \Big ) ,
$$

where � is a fixed shape–layout group, $C _ { g }$ is the set of profiled candidates in that group, and � (�) is measured benchmark throughput. Table 3 reports the complete set of counters used in this analysis.

3.1.2 Global correlation is often a problem-size efect. Several metrics appear to be excellent performance predictors when all profiles are pooled, yet lose almost all ranking power once the problem is fixed. For example, sm\_active\_pct has global $\rho = + 0 . 9 1 0$ but mean within-group $\rho _ { g } = - 0 . 0 0 1$ . Likewise, l2\_read\_sectors drops from +0.942 globally to −0.102 within groups, and dram\_read\_bytes drops from +0.903 to −0.055. These quantities primarily distinguish diferent amounts of work rather than good candidates from bad candidates for the same GEMM.

Two execution quantities remain substantially correlated with candidate quality after fixing the problem. dram\_throughput\_pct has mean within-group $\rho _ { g } ~ = ~ + 0 . 4 0 8$ , and tensor\_active\_pct has mean $\rho _ { g } = + 0 . 3 9 2$ ; both have positive correlation in 83.7% of computable groups. These measurements suggest that successful candidates simultaneously sustain the memory system and keep the tensor-core pipeline active. Since both quantities require executing the kernel, we instead construct static proxies for the mechanisms that produce them: data reuse, trafic, cache-relative working sets, feed–compute balance, pipeline fill, and pipeline amortization.

3.1.3 Measured symptoms versus controllable causes. Several counters are useful diagnostically but unsuitable as direct selector features. Barrier stalls, for example, have global $\rho ~ = ~ + 0 . 8 9 4$ but only mean within-group ${ \rho _ { g } } = \mathrm { { + 0 } } \mathrm { { . 1 5 6 } }$ . Long-scoreboard stalls even change apparent interpretation: their global correlation is −0.679, while the mean within-group correlation is +0.193. Such counters describe the state of an asynchronous execution pipeline after scheduling, resource allocation, and overlap have already taken efect.

This distinction is also visible when relating configuration parameters to measured counters. Pipeline depth has mean within-group correlation −0.357 with long-scoreboard stalls (median −0.439), consistent with additional stages being associated with fewer exposed memory dependencies. However, stage count itself has only weak correlation with throughput. We therefore represent the quantities that determine the consequences of staging, such as bytes per stage, total shared-memory demand, residency, pipeline fill, and stage amortization, rather than treating the stall counter or stage count as a complete performance signal.

Table 3: Spearman correlation between profiled execution metrics and measured benchmark throughput.
<table><tr><td>Metric</td><td>Global ρ</td><td>Mean  $\rho _ { g }$ </td><td>Median  $\rho _ { g }$ </td></tr><tr><td>Throughput and pipeline activity</td><td></td><td></td><td></td></tr><tr><td>tensor_active_pct</td><td>+0.992</td><td>+0.392</td><td>+0.492</td></tr><tr><td>dram_throughput_pct</td><td>+0.817</td><td>+0.408</td><td>+0.500</td></tr><tr><td>compute_throughput_pct</td><td>+0.977</td><td>+0.235</td><td>+0.382</td></tr><tr><td>tma_active_pct</td><td>+0.792</td><td>+0.181</td><td>+0.289</td></tr><tr><td>sm_active_pct</td><td>+0.910</td><td>-0.001</td><td>+0.024</td></tr><tr><td>Memory traffic and hierarchy</td><td></td><td></td><td></td></tr><tr><td>dram_read_bytes</td><td>+0.903</td><td>-0.055</td><td>-0.093</td></tr><tr><td>dram_write_bytes</td><td>+0.817</td><td>+0.000</td><td>+0.086</td></tr><tr><td>12_read_sectors</td><td>+0.942</td><td>-0.102</td><td>-0.238</td></tr><tr><td>12_write_sectors</td><td>+0.908</td><td>-0.151</td><td>-0.184</td></tr><tr><td>12_throughput_pct</td><td>+0.868</td><td>+0.142</td><td>+0.314</td></tr><tr><td>12_hit_rate</td><td>-0.393</td><td>-0.058</td><td>-0.087</td></tr><tr><td>12_read_hit_rate</td><td>+0.003</td><td>-0.086</td><td>-0.194</td></tr><tr><td>12_write_hit_rate</td><td>+0.243</td><td>+0.110</td><td>+0.000</td></tr><tr><td>Occupancy and resource use</td><td></td><td></td><td></td></tr><tr><td>achieved_occupancy_pct</td><td>+0.388</td><td>+0.140</td><td>+0.128</td></tr><tr><td>registers_per_thread</td><td>+0.736</td><td>-0.151</td><td>-0.315</td></tr><tr><td>warps_active</td><td>+0.388</td><td>+0.112</td><td>+0.145</td></tr><tr><td>eligible_warps_per_cycle</td><td>+0.337</td><td>-0.148</td><td>-0.191</td></tr><tr><td>Warp stalls</td><td></td><td></td><td></td></tr><tr><td>stall_barrier_pct</td><td>+0.894</td><td>+0.156</td><td>+0.143</td></tr><tr><td>stall_long_scoreboard_pct</td><td>-0.679</td><td>+0.193</td><td>+0.267</td></tr><tr><td>stall_short_scoreboard_pct</td><td>-0.863</td><td>+0.029</td><td>+0.029</td></tr><tr><td>stall_gmma_pct</td><td>+0.540</td><td>-0.158</td><td>-0.217</td></tr><tr><td>stall_mio_pct</td><td>-0.100</td><td>+0.217</td><td>+0.227</td></tr><tr><td>stall_wait_pct</td><td>-0.519</td><td>-0.113</td><td>-0.143</td></tr><tr><td>stall_not_selected_pct</td><td>+0.054</td><td>-0.224</td><td>-0.238</td></tr></table>

3.1.4 Static Configuration Efects. We additionally apply the same within-group analysis to candidate parameters and statically derived quantities. Table 4 summarizes the strongest efects.

Table 4: Correlation between selected static candidate quantities and measured throughput.
<table><tr><td>Feature</td><td>Global ρ</td><td>Mean  $\rho _ { g }$ </td><td>Median  $\rho _ { g }$ </td></tr><tr><td>tile_m</td><td>+0.626</td><td>-0.261</td><td>-0.409</td></tr><tr><td>tile_n</td><td>+0.827</td><td>+0.080</td><td>+0.091</td></tr><tr><td>tile_k</td><td>-0.415</td><td>+0.257</td><td>+0.272</td></tr><tr><td>k_iters</td><td>+0.826</td><td>-0.260</td><td>-0.289</td></tr><tr><td>reg_pressure_proxy</td><td>+0.827</td><td>-0.222</td><td>-0.319</td></tr><tr><td>cluster_size</td><td>-0.337</td><td>-0.247</td><td>-0.223</td></tr><tr><td>stages</td><td>-0.001</td><td>-0.111</td><td>-0.145</td></tr><tr><td>bytes_per_stage</td><td>+0.396</td><td>+0.054</td><td>+0.092</td></tr><tr><td>pipeline_fill_frac</td><td>+0.721</td><td>+0.233</td><td>+0.171</td></tr><tr><td>restream_factor</td><td>+0.399</td><td>+0.115</td><td>+0.258</td></tr></table>

These results illustrate why raw template parameters alone are insuficient. For example, stages has almost zero global correlation and only weak negative within-group correlation with throughput, even though stage depth directly afects the asynchronous mainloop. Its performance efect depends on how much storage each stage requires, how many �-iterations are available to amortize pipeline startup, and whether the resulting resource footprint reduces residency. The representation therefore includes both the raw stage count and derived quantities describing these interactions.

Similarly, $T _ { K }$ has positive mean within-group correlation with throughput (+0.257), while the corresponding number of reduction iterations has correlation −0.260. However, $T _ { K }$ is not itself an arithmetic-intensity proxy: in the tile-level ratio

$$
I _ { \mathrm { t i l e } } = { \frac { 2 T _ { M } T _ { N } T _ { K } } { b _ { A } T _ { M } T _ { K } + b _ { B } T _ { N } T _ { K } } } = { \frac { 2 T _ { M } T _ { N } } { b _ { A } T _ { M } + b _ { B } T _ { N } } } ,
$$

$T _ { K }$ cancels. Instead, changing $T _ { K }$ modifies reduction granularity, the number of mainloop iterations, per-stage storage, and the amount of computation available between pipeline synchronization points.

3.1.5 Regime Dependence. The efect of a configuration parameter need not be uniform over the shape space. To test this, we repeat the within-group analysis after partitioning the held-out problems by size, geometry, and arithmetic-intensity regime.

The strongest example is $T _ { K }$ . Its efect remains positive across coarse size regimes, with mean within-group correlations of +0.217, +0.307, and +0.245 for small, medium, and large problems respectively, and +0.265 for compute-heavy problems. The magnitude changes substantially with geometry: the mean correlation is +0.549 for wide GEMMs, +0.116 for square GEMMs, and +0.021 for tall GEMMs. Skinny-� problems yield +0.268.

Table 5: Within-group correlation between $T _ { K }$ and throughput under selected problem regimes.
<table><tr><td>Regime</td><td>Mean  $\rho _ { g }$ </td><td>Median  $\rho _ { g }$ </td></tr><tr><td>Small</td><td>+0.217</td><td>+0.262</td></tr><tr><td>Medium</td><td>+0.307</td><td>+0.412</td></tr><tr><td>Large</td><td>+0.245</td><td>+0.273</td></tr><tr><td>Compute-heavy</td><td>+0.265</td><td>+0.289</td></tr><tr><td>Memory-heavy</td><td>+0.238</td><td>+0.262</td></tr><tr><td>Wide</td><td>+0.549</td><td>+0.577</td></tr><tr><td>Square</td><td>+0.116</td><td>+0.237</td></tr><tr><td>Tall</td><td>+0.021</td><td>-0.010</td></tr><tr><td>Skinny-K</td><td>+0.268</td><td>+0.268</td></tr></table>

These results show that the strength of its efect depends strongly on problem geometry and execution regime. More generally, this motivates exposing both candidate choices and the problem quantities that determine how those choices should be interpreted.

The same principle guides the complete feature map: a feature is retained when it is available before execution, distinguishes configurations or identifies the regime in which they operate, and has a plausible connection to an execution mechanism.

## 3.2 Feature Map

Table 6 summarizes our chosen feature set. The structural core identifies the problem and candidate. The remaining groups estimate the hardware behavior. For each problem ${ \boldsymbol { p } } ,$ candidate �, and target $h ,$ this procedure produces the feature vector $x = \phi ( p , c , h ) ;$ the feature matrix of the candidate catalogue is the sole input to the selector described in Section 4.

3.2.1 Structural Core. The structural core contains the quantities required to identify a candidate: problem dimensions, operand layouts and types, threadblock and instruction tile shapes, stage count, mainloop and epilogue schedules, cluster shape, tile scheduler, and operator and compute-engine identifiers. Architecture descriptors provide the capacities and rates used by the derived features, including the number of SMs, shared-memory and register-file capacity per SM, LLC capacity, HBM bandwidth, and the peak throughput of the selected compute engine.

Table 6: Feature groups.
<table><tr><td>Group</td><td>Features and purpose</td></tr><tr><td>Structural core</td><td>log2 M, log2 N, log2 K; operand layouts; input, accumulation, and output types; tile and in- struction shapes; pipeline stages; mainloop and epilogue schedules; cluster shape; tile sched-</td></tr><tr><td>Work decomposition</td><td>uler; and architecture identifiers. Output-tile counts, reduction iterations, edge-tile waste, SM subscription, final-wave efficiency, Stream-K applicability, and cluster fit.</td></tr><tr><td>Memory behavior</td><td>Problem and tile arithmetic intensity, operand re-streaming, working-set and resident-panel size relative to L2, and input, data/epilogue/conversion traffic.</td></tr><tr><td>Resource pressure</td><td>Bytes per pipeline stage, total mainloop and epilogue shared-memory storage, register- pressure proxy, and shared-memory- and register-limited occupancy.</td></tr><tr><td>Pipeline behavior</td><td>Estimated TMA load time, WGMMA compute time, producer-consumer ratio, pipeline-fill fraction, stage amortization, and interactions between tile depth, stage count, and occupancy.</td></tr></table>

These features are necessary but insuficient. A stage count of four does not itself state whether four stages are enough to overlap TMA and WGMMA, whether the associated bufers fit in shared memory, or whether the problem has enough �-iterations to amortize pipeline fill. The derived groups make these consequences explicit.

3.2.2 Work Decomposition and Wave Quantization. Let $( T _ { M } , T _ { N } , T _ { K } )$ be the candidate tile shape. The output grid and number of mainloop iterations are

$$
n _ { M } = \Biggl \lceil \frac { M } { T _ { M } } \Biggr \rceil , \qquad n _ { N } = \Biggl \lceil \frac { N } { T _ { N } } \Biggr \rceil , \qquad n _ { K } = \Biggl \lceil \frac { K } { T _ { K } } \Biggr \rceil .
$$

The candidate computes �<sub>�</sub>�<sub>�</sub> output tiles. The edge tiles may execute masked work that is discarded; we encode this separately along each output dimension:

$$
w _ { M } = \frac { n _ { M } T _ { M } - M } { M } , \qquad w _ { N } = \frac { n _ { N } T _ { N } - N } { N } .
$$

To estimate wave quantization, we divide the number of output tiles by the number of concurrently resident blocks across the GPU. If $\mathrm { \partial } S _ { S M }$ is the SM count and $b _ { \mathrm { r e s } }$ the estimated resident blocks per SM, then

$$
q _ { \mathrm { S M } } = \left\lceil \frac { n _ { M } n _ { N } } { S _ { \mathrm { S M } } b _ { \mathrm { r e s } } } \right\rceil
$$

is the number of passes that the resident blocks make over the tile list. The corresponding wave eficiency is

$$
\eta _ { \mathrm { l a s t } } = \frac { n _ { M } n _ { N } } { q _ { \mathrm { S M } } S _ { \mathrm { S M } } b _ { \mathrm { r e s } } } .
$$

A value near one indicates that there are several waves and/or the final wave fills the GPU, while a small value indicates a ragged tail. This directly motivates the Stream-K proxy:

$$
\mathsf { s t r e a m K \_ a p p l i c a b i l i t y } = \mathsf { m a x } ( 0 , 1 - \eta _ { \mathrm { l a s t } } ) ,
$$

together with a feature indicating whether the candidate actually uses Stream-K. Cluster features similarly encode the cluster

dimensions, number of tiles available to fill each cluster dimension, and whether the cluster overshoots the grid.

3.2.3 Memory Behavior. We estimate problem arithmetic intensity using the bytes required by the input matrices and the complete epilogue:

$$
I _ { \mathrm { p r o b l e m } } = \frac { 2 M N K } { b _ { A } M K + b _ { B } K N + Q _ { \mathrm { e p i l o g u e } } } ,
$$

where $b _ { A }$ and $b _ { B }$ are input element sizes and $Q _ { \mathrm { e p i l o g u e } }$ includes output stores and any reads or writes induced by the output type, $\beta C$ , residual, bias, activation, or conversion operations.

A complementary tile-level feature captures the amount of tensor-core work obtained from a loaded operand footprint:

$$
I _ { \mathrm { m a i n l o o p } } = { \frac { 2 T _ { M } T _ { N } } { T _ { M } b _ { A } + T _ { N } b _ { B } } } .
$$

Here $T _ { K }$ cancels. This is useful: $T _ { K }$ afects per-stage storage and pipeline depth, whereas $T _ { M }$ and $T _ { N }$ determine how much reuse the tile geometry obtains from each loaded operand row or column.

We estimate the re-streaming incurred if operand panels cannot remain resident in LLC:

$$
r _ { \mathrm { s t r e a m } } = { \frac { n _ { N } b _ { A } M K + n _ { M } b _ { B } K N } { b _ { A } M K + b _ { B } K N } } .
$$

The remaining cache features compare the complete operand working set and a single resident operand panel with LLC capacity. Together, these features distinguish a problem whose repeated operand reads are absorbed by LLC from one that must repeatedly fetch them from HBM.

3.2.4 Resource Pressure and Occupancy. The shared-memory bufer required for one mainloop stage is

$$
B _ { \mathrm { s t a g e } } = T _ { M } T _ { K } b _ { A } + T _ { N } T _ { K } b _ { B } .
$$

For � pipeline stages, the mainloop requires $S B _ { \mathrm { s t a g e } }$ bytes. The epilogue additionally requires schedule-dependent storage $B _ { \mathrm { e p i } }$ including output fragments and any fused epilogue state. For a warp-specialized schedule, mainloop and epilogue storage may overlap; cooperative and ping-pong schedules may require both simultaneously. We therefore compute the corresponding schedulespecific total shared-memory requirement $B _ { \mathrm { s m e m } }$ and its fraction of per-SM capacity:

$$
f _ { \mathrm { s m e m } } = \frac { B _ { \mathrm { s m e m } } } { C _ { \mathrm { s m e m } } } .
$$

This bounds shared-memory-limited residency:

$$
b _ { \mathrm { s m e m } } = \left\lfloor \frac { C _ { \mathrm { s m e m } } } { B _ { \mathrm { s m e m } } } \right\rfloor .
$$

We approximate register pressure from the output elements accumulated per thread,

$$
r _ { \mathrm { p r o x y } } = \frac { T _ { M } T _ { N } } { T _ { \mathrm { b l o c k } } } ,
$$

and use the register-file capacity $C _ { \mathrm { r e g } }$ to derive an independent residency bound $b _ { \mathrm { r e g } }$ . The estimated resident-block count is

$$
b _ { \mathrm { r e s } } = \operatorname* { m i n } ( b _ { \mathrm { s m e m } } , b _ { \mathrm { r e g } } , b _ { \mathrm { a r c h } } ) .
$$

Keeping the separate bounds matters, because increasing the stage count tightens the shared-memory bound but does not directly alter accumulator pressure. The model can therefore distinguish a kernel limited by registers from one that falls of a shared-memory size limitation.

3.2.5 Pipeline Balance. Hopper warp-specialized kernels overlap TMA loads from HBM to shared memory with WGMMA execution from shared memory. We estimate the producer and consumer times for one stage as

$$
t _ { \mathrm { p r o d u c e r } } = \frac { n _ { S M } B _ { \mathrm { s t a g e } } } { \widehat { B W } _ { \mathrm { T M A } } } \qquad t _ { \mathrm { c o n s u m e r } } = \frac { 2 T _ { M } T _ { N } T _ { K } } { n _ { \mathrm { c o n s u m e r } } \widehat { P } _ { \mathrm { W G M M A } } }
$$

where $n _ { \mathrm { c o n s u m e r } }$ is the number of consumer warpgroups actively executing the mainloop for the selected schedule, $\widehat { B W } _ { \mathrm { T M A } }$ is the bandwidth of the TMA operation, and $\widehat { P } _ { \mathrm { W G M M A } }$ is the throughput of the WGMMA operation. The producer-consumer ratio

$$
r _ { \mathrm { p c } } = \frac { t _ { \mathrm { c o n s u m e r } } } { t _ { \mathrm { p r o d u c e r } } }
$$

is a static proxy for whether TMA can supply WGMMA at the required rate. Values far below one indicate that consumers are likely to wait for data; values far above one indicate that loads are completed faster than arithmetic consumes them.

Two additional features describe whether the pipeline depth is useful for the problem:

$$
f _ { \mathrm { f l l } } = \frac { \operatorname* { m i n } ( S , n _ { K } ) } { S } , \qquad a _ { \mathrm { s t a g e } } = \frac { n _ { K } } { S } .
$$

The first falls below one when the contraction is too shallow to prime a deep pipeline. The second measures how often the pipeline fill cost is amortized. Finally, $S \cdot f _ { \mathrm { s m e m } }$ exposes the interaction between stage count and shared-memory pressure. This interaction captures both the stage-count clif and the regime in which a larger $T _ { K }$ becomes harmful because its enlarged stage bufers force a shallower pipeline.

## 4 Training the Selector

Given an input configuration, we list the valid CUTLASS kernel catalogue, and use a machine learning model to rank the catalogue and choose the best kernel. In this section, we explain the data collection process for training and the model design.

## 4.1 Constructing The Training Dataset

The CUTLASS configuration space is too large to measure exhaustively for every problem shape. Even after validity filtering, a single BF16 GEMM layout exposes over sixty thousand candidate configurations. Exhaustively sweeping this space over thousands of shapes would require thousands of compiles and hundreds of millions of benchmarks.

We therefore separate the collection into two tasks. Our training dataset covers a broad range of GEMM shapes while also sampling candidates within each shape-layout group. The held-out evaluation dataset covers fewer shapes, but sweeps their valid candidates exhaustively to establish a trustworthy in-space oracle. This section describes the training GEMM collection; the same collection design is used when extending the dataset to additional types, epilogues, and operator families.

4.1.1 Shape Selection. A GEMM problem shape is a triple (�, �, �). We define our structured anchor grid in 3 dimensions as [32,16384]<sup>3</sup> with each dimension snapped to a multiple of 32 – roughly 134 million lattice points. The preferred CUTLASS kernel changes with problem geometry (square, tall, wide, skinny-�), reduction depth $\left( \lceil K / T _ { K } \rceil \right)$ , and cache residency; a single random draw over this box would leave boundary (extreme large/small values of one/more dimensions) and hardware transition points severely underrepresented. We therefore use two complementary (but non-disjoint) construction axes: a geometry class (interior versus boundary/extreme) and, where needed, targeted supplements (anchor neighbours and transition points). The final dataset contains 593 base shapes (× 4 layouts = 2372 unique groups):

• Primary random coverage (469 shapes). 297 interior shapes: log-uniform draws from the anchor grid. 172 boundary shapes: deliberate hull extremes; very tall, wide, skinny-$K ,$ and corner cases up to 16384 on an axis.

• Targeted supplements (124 shapes). 92 anchor shapes: Neighbours formed by ablating shapes. For every data point, the training set includes shapes that vary one dimension at a time where the domain permits it. This helps the model learn how changing diferent dimensions of the input shape changes performance and behaviour. This adds 47 interior and 45 boundary shapes. 32 transition shapes: points near the roofline ridge (BF16 arithmetic intensity ≈ 247.3 flop/byte on GH200) and the 50 MiB LLC-residency boundary. This adds 14 interior and 18 boundary shapes.

4.1.2 Candidate Sampling. For a shape-layout group, we measure only a budgeted subset of the full catalogue during training. The default budget is 2000 candidates per group. It is reduced to 1000 for the largest shapes, where each benchmark is substantially more expensive, and increased to 3000 for atypical or transition shapes, where the configuration ranking changes more sharply.

Table 7: Fraction of valid kernels close to the exhaustive oracle.
<table><tr><td>Threshold</td><td>Mean fraction</td><td>Median fraction</td><td>Minimum fraction</td></tr><tr><td>Within 1%</td><td>0.43%</td><td>0.17%</td><td>0.17%</td></tr><tr><td>Within 5%</td><td>2.41%</td><td>1.00%</td><td>0.33%</td></tr><tr><td>Within 10%</td><td>10.91%</td><td>4.42%</td><td>1.17%</td></tr></table>

Exhaustive measurement is impractical for training. Therefore we study how the measurement budget should be allocated across candidates. The goal of this experiment is to quantify how efec tively a sampling policy exposes high-performing kernels within a fixed measurement budget.

We use the exhaustively measured BF16 evaluation groups as ground truth. For each group ${ \mathit { g } } ,$ let $C _ { g }$ denote the complete valid candidate catalogue, � (�) the measured throughput of candidate $c ,$ and

$$
T _ { g } ^ { \star } = \operatorname* { m a x } _ { c \in C _ { g } } T ( c )
$$

the exhaustive oracle throughput. Given a sampling policy that selects a subset $S _ { B } ( g ) \subseteq C _ { g }$ of � candidates, we define the sampling regret

$$
R _ { B } ( g ) = 1 - \frac { \operatorname* { m a x } _ { c \in S _ { B } ( g ) } T ( c ) } { T _ { g } ^ { \star } } .
$$

This measures the quality lost purely because the measurement subset failed to contain the best available kernels. For stochastic policies, we repeat each sampling experiment 100 times per group and report mean results.

4.1.3 Sparsity ofthe High-Performance Tail. We first quantify how many kernels in the valid catalogue are actually competitive. For each group, we compute the fraction of candidates whose through put lies within � of the exhaustive oracle:

$$
q _ { \epsilon } ( g ) = \frac { \left| \left\{ c \in C _ { g } : T ( c ) \geq ( 1 - \epsilon ) T _ { g } ^ { \star } \right\} \right| } { | C _ { g } | } .
$$

Across the evaluation groups, only 0.43% ofcandidates are within 1% ofthe oracle on average, 2.4% are within 5%, and 10.9% are within 10%. The corresponding median fractions are 0.17%, 1.0%, and 4.4%, respectively. In the most selective groups, fewer than 0.33% of valid kernels lie within 5% of the optimum.

These results show that high-performing kernels occupy a sparse tail of the catalogue. A uniformly sampled training subset can therefore contain thousands of measured candidates while still failing to observe any configuration close to the true optimum.

4.1.4 Sampling Policies. We compare the following candidateselection policies.

Uniform. Candidates are sampled uniformly at random from the complete valid catalogue.

Static top-�. Candidates are ranked by the final static sampling score from Section 4.1.8, and the highest-ranked � candidates are selected deterministically.

Figure 3: Sampling regret versus candidate measurement budget on the exhaustive BF16 evaluation groups.  
![](images/fa682dc2f7b9a844389cb9eef28c0517c117c0dd487f527c0422a825431e78ad.jpg)

Biased-diverse sampling. This is the policy used to construct the training set. A fraction 75% of the budget is sampled from the high-ranked portion of the static score, while the remaining 25% is sampled uniformly from the rest of the catalogue. The exploratory quarter is not overhead for oracle discovery alone: the throughput predictor must also observe poorly ranked configurations during training, so a purely heuristic-biased corpus would under-represent the long tail of bad kernels.

We evaluate budgets

$$
B \in \{ 1 0 0 , 2 5 0 , 5 0 0 , 1 0 0 0 , 2 0 0 0 , 3 0 0 0 , 5 0 0 0 , 1 0 0 0 0 \} .
$$

The values 1000, 2000, and 3000 correspond directly to the candidate budgets used in our training collection.

4.1.5 Sampling Eficiency. Figure 3 reports sampling regret as a function of measurement budget. Uniform sampling improves steadily with larger budgets, but converges slowly because the near-optimal tail is sparse. At a budget of 1000 candidates, uniform sampling leaves 11.3% mean regret; at 2000 candidates the regret is 8.5%, and at 3000 candidates it remains 7.1%.

The static top-� policy improves over uniform sampling at small budgets, reaching 0.6% regret at � = 2000. However, its deterministic bias can exclude competitive kernels that receive a poor heuristic score. The biased-diverse policy combines both efects and achieves 1.5%, 0.9%, and 0.5% mean regret at budgets of 1000, 2000, and 3000, respectively.

At the default training budget of 2000 candidates, biased-diverse sampling reduces sampling regret by 89.4% relative to uniform sampling (from 8.5% to 0.9%). Static top-� attains marginally lower regret (0.6%) at this budget, but it is unsuitable as a training policy because it never deliberately measures configurations the heuristic ranks poorly. Oracle-discovery metrics and training-corpus construction therefore pull in opposite directions under a fixed budget.

4.1.6 Probability of Observing a Near-Optimal Kernel. Sampling regret summarizes the quality of the best observed configuration, but it is also useful to ask how often a sampling policy discovers at least one near-optimal kernel. We therefore measure

Figure 4: Probability that the sampled candidate pool contains a near-optimal kernel.  
![](images/1e3ba8c7a7b04f081ae7914245f215e8f59c9557dff41a22fd4716a504324b4b.jpg)

$$
P _ { \epsilon } ( B ) = \mathrm { P r } \left[ \operatorname* { m a x } _ { c \in S _ { B } ( g ) } T ( c ) \geq ( 1 - \epsilon ) T _ { g } ^ { \star } \right]
$$

for $\epsilon \in \{ 0 . 0 1 , 0 . 0 5 , 0 . 1 0 \}$

Figure 4 reports $P _ { 0 . 0 5 } ( B )$ as a function of measurement budget. $\mathrm { A t } \ : B = 2 0 0 0$ , uniform sampling observes at least one kernel within 5% of the oracle in 25.6% of groups, compared with 97.1% for static top-� and 91.3% for biased-diverse sampling. At $B = 3 0 0 0$ , the corresponding fractions are 35.3%, 97.1%, and 97.3%. Uniform sampling improves with budget but remains far below the heuristic-biased policies until $B \geq 3 0 0 0 ,$ , where biased-diverse sampling essentially matches static top-�.

Table 8: Ablation of the exploitation fraction at $B = 2 0 0 0$
<table><tr><td>Heuristic-biased fraction α</td><td>Mean sampling regret</td><td>Groups within 5% of oracle</td></tr><tr><td>0.00</td><td>8.5%</td><td>25.6%</td></tr><tr><td>0.25</td><td>1.4%</td><td>87.0%</td></tr><tr><td>0.50</td><td>1.1%</td><td>88.9%</td></tr><tr><td>0.75</td><td>0.9%</td><td>91.3%</td></tr><tr><td>1.00</td><td>0.6%</td><td>97.1%</td></tr></table>

4.1.7 Efect ofthe Exploration Fraction. The biased-diverse policy trades of exploitation ofthe static score against broader exploration of the remaining catalogue. We ablate this mixture at a fixed budget of $B = 2 0 0 0$ . Let � denote the fraction of the budget drawn from the heuristic-preferred region. We test

$$
\alpha \in \{ 0 , 0 . 2 5 , 0 . 5 0 , 0 . 7 5 , 1 . 0 \} .
$$

Pure exploration $( \alpha = 0 )$ reduces to broad random sampling and yields 8.5% mean regret. Pure exploitation $( \alpha = 1 )$ achieves 0.6% regret and 97.1% within-5% coverage, the best values for oracle discovery on this evaluation set. Increasing � monotonically improves both metrics, yet $\alpha = 1$ is not a viable training policy: the model would rarely see measured examples of low-throughput kernels and could not learn to discriminate bad configurations from good ones.

This retrospective analysis validates the $\alpha = 0 . 7 5$ policy used for corpus construction. At $B = 2 0 0 0$ this yields 0.9% mean regret and 91.3% within-5% coverage—a modest cost relative to $\alpha = 1 -$ while guaranteeing that one quarter of every measurement budget samples outside the heuristic-preferred region.

4.1.8 Summary. The exhaustive study confirms that the candidateselection problem is dominated by a sparse high-performance tail: only 2.4% of valid kernels lie within 5% of the oracle on average. Consequently, uniform sampling requires substantially larger measurement budgets to expose competitive configurations. At the default budget of 2000 candidates per shape–layout group, our biased-diverse policy reduces sampling regret from 8.5% to 0.9% and increases the probability of observing a kernel within 5% of the oracle from 25.6% to 91.3%.

These results retroactively validate the candidate-sampling strategy used for the training corpus. Most measurements are directed toward configurations with favorable static eficiency estimates, while the fixed 25% exploratory fraction ensures the predictor is trained on bad as well as good kernels—a requirement that pure top-� or $\alpha = 1$ sampling cannot satisfy under the same budget. Instead, we rank the valid candidates using a cheap, static eficiency score that ranks kernels likely to give better performance. Let �<sub>�</sub>, �<sub>�</sub> be the thead-block cluster shape, $S _ { S M }$ the number of SM clusters, � the pipeline stage count, and

$$
n _ { c } = \bigg \lceil \frac { M } { c _ { M } T _ { M } } \bigg \rceil \bigg \lceil \frac { N } { c _ { N } T _ { N } } \bigg \rceil
$$

be the number of clusters required by the launch. The number of cluster waves is thus $\begin{array} { r } { W = \lceil \frac { \hat { n } _ { c } } { S _ { \mathrm { S M } } } \rceil } \end{array}$ . We use the following sampling score to rank kernels:

$$
s _ { \mathrm { s a m p l e } } = { \frac { n _ { c } } { W S _ { \mathrm { S M } } c _ { M } c _ { N } } } \cdot S \cdot { \sqrt { T _ { M } T _ { N } T _ { K } } }
$$

The first term favours eficient wave filling, penalizing oversized clusters that are too wasteful. The remaining terms favour configurations with deeper pipelines and suficiently large tiles, which are likely to have higher reuse and provide better throughput. For a group budget, three quarters of the sampled configurations are drawn from the top of this ranking, while the remaining quarter is sampled across the rest of the ranked space. This preserves examples of poor and mediocre configurations. If the number of valid candidates is below the assigned budget, we collect the entire valid pool. The final training collection contains approximately 4.9 million measured kernels.

4.1.9 Collection Pipeline. We perform data collection on a GH200 node of the Daint supercomputer on CSCS Alps, equipped with 4 GH200 GPUs. We compile CUTLASS SM90a kernels from the commit 3476ddb7 using CUDA 13.1. Each measurement records the behavior of one candidate on one shape-layout group. Each successful benchmark consists of five warm-up iterations followed by five rounds of three timed iterations. Every attempted measurement is recorded, including successful executions, compilation failures, launch failures, crashes, and timeouts. To make the dataset practical to collect, we implement the following optimizations:

(1) Batched compilation. Kernels that share a subset of configurations are compiled together into one shared library. A single exported entry point dispatches by kernel index inside the batch, amortizing template instantiation and nvcc startup cost across many configurations.

(2) Executable caching. Each batch library is stored under a name derived from a hash of its member configurations, and kernels are reused instead of recompilation.

(3) Bufer caching. Each GPU worker pre-allocates a fixed device memory pool and serves all kernels from slices of that pool rather than allocating fresh tensors per kernel. We replicate the matrices enough times from this pool until the working set exceeds L2 capacity.

(4) Node-local database storage. Benchmark results and compile metadata are written to a registry database kept in node local memory for the duration of the job, and checkpointed periodically to scratch and to durable home storage.

(5) Parallel compilation and benchmarking. Compilation and GPU benchmarking run concurrently: a pool of CPU workers builds shared libraries while one persistent process per GPU consumes a queue of newly compiled kernels. A monitor recovers from worker crashes and hung kernels by attributing the failure to the active measurement, recording it, and respawning the worker to continue with the remainder of the queue.

## 4.2 Evaluation Dataset

Unlike the training corpus, where each problem is measured on a proxy shortlist of ≈ 2,000 configurations, the evaluation corpus is exhaustively enumerated: for every held-out shape $( M , N , K )$ in Table 9 and every layout tag ℓ ∈ {TN, TT, NN, NT}, the candidate set C(�, �, �, ℓ) is the full catalogue after static validity filtering (Section 2) and shape-aware plan guards. No further proxy pruning is applied. The 17 shapes are listed exhaustively; together with four layouts they define 68 evaluation groups and 3,922,067 distinct (kernel, shape) benchmark tasks.

## 4.3 Learning to Rank Candidates

We study two model families for mapping the hardware-aware representation �(�, �, ℎ) to a scalar candidate score. In both cases, candidates are grouped by GEMM problem � and supervision is derived from measured kernel throughput within that group.

4.3.1 Problem and Candidate Sets. A GEMM instance is a tuple $\boldsymbol { p } = \left( M , N , K , \ell \right)$ , where ℓ denotes the operand layout tag. ℎ be a hardware descriptor. For each � we maintain a finite candidate set $C ( p ) \subset { \mathcal { K } } $ , where K is the statically valid CUTLASS configuration catalogue (Section 2). Every $c \in C ( p )$ is described by a feature vector $\mathbf { x } ( p , c ) = \phi ( p , c ) \in \mathbb { R } ^ { d }$ computed from problem geometry, tile and cluster parameters, schedule choices, and the analytical hardware proxies of Section $3 ;$ no feature depends on measured runtime counters. A learned selector is a scoring function $f _ { \theta } : \mathbb { R } ^ { d } $ R. The deployed decision rule is always

$$
\hat { c } ( p ) \ = \ \underset { c \in C ( p ) } { \arg \ \operatorname* { m a x } } \ f _ { \theta } \big ( \mathbf { x } ( p , c , h ) \big ) ,\tag{1}
$$

i.e. we reduce selection to within-problem ranking and do not use a predicted throughput value directly at inference time.

Table 9: Held-out evaluation shapes and valid kernel counts per layout.
<table><tr><td colspan="3">Shape</td><td colspan="5">Kernel count</td></tr><tr><td>M</td><td>N</td><td>K</td><td>TN</td><td>TT</td><td>NN</td><td>NT</td><td>Total</td></tr><tr><td>2048</td><td>2048</td><td>2048</td><td>60,795</td><td>60,740</td><td>60,740</td><td>60,696</td><td>242,971</td></tr><tr><td>4096</td><td>4096</td><td>4096</td><td>60,795</td><td>60,740</td><td>60,740</td><td>60,696</td><td>242,971</td></tr><tr><td>64</td><td>64</td><td>64</td><td>32,190</td><td>32,135</td><td>32,135</td><td>32,091</td><td>128,551</td></tr><tr><td>128</td><td>128</td><td>128</td><td>60,795</td><td>60,740</td><td>60,740</td><td>60,696</td><td>242,971</td></tr><tr><td>256</td><td>256</td><td>256</td><td>60,795</td><td>60,740</td><td>60,740</td><td>60,696</td><td>242,971</td></tr><tr><td>512</td><td>512</td><td>512</td><td>60,795</td><td>60,740</td><td>60,740</td><td>60,696</td><td>242,971</td></tr><tr><td>32</td><td>128</td><td>4096</td><td>60,795</td><td>60,740</td><td>60,740</td><td>60,696</td><td>242,971</td></tr><tr><td>2048</td><td>128</td><td>4096</td><td>60,795</td><td>60,740</td><td>60,740</td><td>60,696</td><td>242,971</td></tr><tr><td>4096</td><td>128</td><td>4096</td><td>60,795</td><td>60,740</td><td>60,740</td><td>60,696</td><td>242,971</td></tr><tr><td>12,288</td><td>128</td><td>4096</td><td>60,795</td><td>60,740</td><td>60,740</td><td>60,696</td><td>242,971</td></tr><tr><td>256</td><td>4096</td><td>4096</td><td>60,795</td><td>60,740</td><td>60,740</td><td>60,696</td><td>242,971</td></tr><tr><td>256</td><td>12,288</td><td>4096</td><td>60,795</td><td>60,740</td><td>60,740</td><td>60,696</td><td>242,971</td></tr><tr><td>64</td><td>12,288</td><td>4096</td><td>37,290</td><td>37,235</td><td>37,235</td><td>37,191</td><td>148,951</td></tr><tr><td>2048</td><td>2048</td><td>128</td><td>60,795</td><td>60,740</td><td>60,740</td><td>60,696</td><td>242,971</td></tr><tr><td>4096</td><td>11,008</td><td>4096</td><td>60,795</td><td>60,740</td><td>60,740</td><td>60,696</td><td>242,971</td></tr><tr><td>256</td><td>256</td><td>8192</td><td>60,795</td><td>60,740</td><td>60,740</td><td>60,696</td><td>242,971</td></tr><tr><td>512</td><td>3072</td><td>768</td><td>60,795</td><td>60,740</td><td>60,740</td><td>60,696</td><td>242,971</td></tr><tr><td colspan="2">Sum (68 groups)</td><td></td><td>981,405</td><td>980,470</td><td>980,470</td><td>979,722</td><td>3,922,067</td></tr></table>

4.3.2 Measured Labels. For each successfully benchmarked pair $( p , c )$ the autotuner records a mean throughput estimate ${ \widehat { T } } ( p , c )$ and an empirical standard deviation $\widehat { \sigma } ( p , c )$ over repeated timed rounds. Group all rows that share the same $\boldsymbol { p }$ and index such a group by �. The empirical group maximum is

$$
T _ { g } ^ { \star } \ = \ \operatorname * { m a x } _ { c \in C ( g ) } { \widehat T } ( g , c ) , \qquad y ( g , c ) \ = \ \frac { \widehat T ( g , c ) } { T _ { g } ^ { \star } } \in ( 0 , 1 ] ,\tag{2}
$$

so that $y ( g , c ) = 1$ for at least one measured candidate in the group and $y ( g , c ) < 1$ otherwise. On training problems $C ( g )$ is a proxy shortlist (Section 4.1); on evaluation problems it is a near-exhaustive sweep over the catalogue, so $T _ { q } ^ { \star }$ approximates the in-space empirical oracle for that shape–layout.

4.3.3 Relevance grades (ranking supervision). Ranking objectives do not use raw $\widehat { T }$ directly; they use discrete grades derived from it. Within group ${ \mathit { g } } ,$ sort candidates by decreasing $\widehat { T } ( g , c )$ and partition the sorted list into contiguous bands by merging adjacent entries � and �+1 whenever their throughput gap is smaller than twice the pooled measurement uncertainty,

$$
\widehat { T } _ { i } - \widehat { T } _ { i + 1 } < 2 \sqrt { \widehat { \sigma } _ { i } ^ { 2 } + \widehat { \sigma } _ { i + 1 } ^ { 2 } } .\tag{3}
$$

Let $b ( c ) \in \{ 0 , 1 , . . . \}$ be the band index of � (0 for the top band). We assign integer grades $\gamma ( g , c ) = G _ { \mathrm { m a x } } - b ( c )$ with $G _ { \mathrm { m a x } } = 3 1$ . For listwise tree training we further compress grades to a top-focused relevance scale with � bands (�=2 by default):

$$
\mathrm { r e l } ( g , c ) = \mathrm { c l i p } \big ( \gamma ( g , c ) - ( G _ { \mathrm { m a x } } - B ) , 0 , B \big ) .\tag{4}
$$

Thus only the highest-throughput band(s) carry non-zero relevance, concentrating learning on near-optimal orderings.

4.3.4 Model A (gradient-boosted ranker). Model A is an additive ensemble of regression trees $\begin{array} { r } { f _ { \theta } = \sum _ { m } h _ { m } } \end{array}$ fitted with XGBoost. Numeric features enter splits directly; categorical schedule and layout fields use native categorical handling with a fixed level ordering shared across train, validation, and deployment. Let $\mathcal { G } _ { \mathrm { t r } }$ denote the set of training groups and list all rows in a group-contiguous order with group-id vector g. We train with one of three objectives:

• Listwise ranking (default): minimise a surrogate for NDCG at cut-of $\left| C ( g ) \right|$ using targets rel $( g , c )$ (<sub>�</sub>, �) and the standard rank:ndcg objective;

• Pairwise ranking: rank:pairwise on the same relevance labels;

• Pointwise regression (ablation): minimise $\begin{array} { r } { \sum _ { i } \left( f _ { \theta } ( \mathbf x _ { i } ) - y _ { i } \right) ^ { 2 } } \end{array}$ with reg : squarederror, treating �<sub>�</sub> as a continuous target independent of group structure.

Only the first two objectives align the training loss with the argmax decision rule in (1); the regression ablation tests whether accurate throughput prediction implies good selection.

4.3.5 Model B (feed-forward ranker). Model B is a multilayer perceptron $f _ { \theta } : \mathbb { R } ^ { d ^ { \prime } }  \mathbb { R }$ applied to a transformed feature matrix. Let $\mathbf { x } ^ { \mathrm { { n u m } } }$ denote numeric columns and ${ \bf x } ^ { \mathrm { c a t } }$ categorical indicators obtained by one-hot encoding with the same frozen level sets as in Model A. Training applies per-column standardisation to $\mathbf { x } ^ { \mathrm { { n u m } } }$ (zero mean, unit variance, with NaN/±∞ mapped to zero before scaling); categoricals are left unscaled. We consider three training losses:

$$
\mathcal { L } _ { \mathrm { M S E } } ( \theta ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \bigl ( f _ { \theta } ( \mathbf { x } _ { i } ^ { \prime } ) - y _ { i } \bigr ) ^ { 2 } ,\tag{5}
$$

$$
\mathcal { L } _ { \mathrm { R a n k N e t } } ( \theta ) = \frac { 1 } { | \mathcal { P } | } \sum _ { ( i , j ) \in \mathcal { P } } \log \bigl ( 1 + \exp \bigl ( f _ { \theta } ( \mathbf { x } _ { j } ^ { \prime } ) - f _ { \theta } ( \mathbf { x } _ { i } ^ { \prime } ) \bigr ) \bigr ) ,\tag{6}
$$

$$
\mathcal { L } _ { \mathrm { L a m b d a R a n k } } ( \theta ) = \frac { 1 } { | \mathcal { P } | } \sum _ { ( i , j ) \in \mathcal { P } } \Delta \mathrm { N D C } G _ { i j } \log \bigl ( 1 + \exp \bigl ( f _ { \theta } ( \mathbf { x } _ { j } ^ { \prime } ) - f _ { \theta } ( \mathbf { x } _ { i } ^ { \prime } ) \bigr ) \bigr ) ,\tag{7}
$$

where $\mathbf { x } _ { i } ^ { \prime }$ is the scaled/encoded feature vector, and $\mathcal { P }$ is a set of ordered pairs within a group such that $y _ { i } > y _ { j }$ (higher-throughput index $i ,$ lower $j )$ . For LambdaRank we use gain values $g _ { i } ~ = ~ y _ { i }$ (linear mode) or $g _ { i } = 2 ^ { y _ { i } } - 1$ (exponential mode), ideal discounted cumulative gain $\begin{array} { r } { \mathrm { I D C G } = \sum _ { r = 1 } ^ { n } g _ { \pi ( r ) } / \mathrm { l o g } _ { 2 } ( 1 + r ) } \end{array}$ for the throughputsorted permutation �, and pair weights

$$
\Delta \mathrm { N D C G } _ { i j } \ = \ { \frac { \left| g _ { i } - g _ { j } \right| \left| D ( r _ { i } ) - D ( r _ { j } ) \right| } { \mathrm { I D C G } } } , \qquad D ( r ) = { \frac { 1 } { \log _ { 2 } ( 1 + r ) } } ,\tag{8}
$$

with ranks $r _ { i } , r _ { j }$ computed from the current model scores (detached from the gradient graph when evaluating �). Pairs are sampled per group with a hard cap; pairs touching any of the top eight �-values are always retained (needle guarantee) before random subsampling of the remainder. At deployment, numeric standardisation is folded into the exported inference graph so that (1) can be evaluated on raw feature rows.

4.3.6 Linear Ridge Baseline. The ridge baseline tests how much of the learned selectors’ advantage comes from the hardware-aware representation and globally linear feature combinations, without tree splits or hidden layers. It uses the same candidate feature matrix as Models A and B: numeric columns from $\phi ( p , c , h )$ are standardised with training-set mean and variance (with non-finite values mapped to zero before scaling), and categorical schedule and layout fields are one-hot encoded with the same frozen level sets. Let x<sup>′</sup> denote the resulting vector for row �.

The scorer is ordinary least squares with optional $\ell _ { 2 }$ shrinkage:

$$
f _ { \mathbf { w } , b } ( \mathbf { x } ^ { \prime } ) \ = \ \mathbf { w } ^ { \top } \mathbf { x } ^ { \prime } + b , \qquad \operatorname* { m i n } _ { \mathbf { w } , b } \ \sum _ { i \in \mathcal { P } _ { \mathrm { t r } } } \left( y _ { i } - f _ { \mathbf { w } , b } ( \mathbf { x } _ { i } ^ { \prime } ) \right) ^ { 2 } + \alpha \ \lVert \mathbf { w } \rVert _ { 2 } ^ { 2 } .\tag{9}
$$

We solve (9) in closed form after centring $\mathbf { x } ^ { \prime }$ and � (intercept handled separately); at inference the deployed rule is still the argmax in (1). Unlike the ranking objectives above, training optimises squared error on � directly; validation and test nonetheless score the selected kernel through regret (11).

4.3.7 Selection Regret (The Target Metric). For any scorer $f _ { \theta }$ the regret of group � is

$$
R ( g ; f _ { \theta } ) ~ = ~ 1 - y \big ( g , \widehat { c } ( g ) \big ) ~ = ~ 1 - { \frac { \widehat { T } \big ( g , \widehat { c } ( g ) \big ) } { T _ { g } ^ { \star } } } ,\tag{10}
$$

$$
{ \hat { c } } ( g ) = \arg \operatorname* { m a x } _ { c \in C ( g ) } f _ { \theta } { \bigl ( } \mathbf { x } ( g , c ) { \bigr ) } .\tag{11}
$$

$R ( g ) ~ \in ~ [ 0 , 1 )$ with $R ( g ) ~ = ~ 0$ if the selector picks a measured maximiser in $C ( g )$ . Aggregate reports include the mean $\bar { R } \ =$ $\begin{array} { r } { { \frac { 1 } { | \mathcal { G } | } } \sum _ { g } R ( g ) } \end{array}$ , median, 95th percentile, and maximum across groups, as well as

$$
\mathrm { T o p } 1 ( f _ { \theta } ) = \frac { 1 } { | \mathcal { G } | } \sum _ { g } \mathbf { 1 } \big [ \mathrm { r a n k } ( \hat { c } ( g ) ) = 1 \big ] ,\tag{12}
$$

$$
\mathrm { T o p } 5 ( f _ { \theta } ) = \frac { 1 } { | \mathcal { G } | } \sum _ { g } \mathbf { 1 } \big [ \mathrm { r a n k } ( \hat { c } ( g ) ) \leq 5 \big ] ,\tag{13}
$$

$$
\mathrm { W i t h i n } 5 \% ( f _ { \theta } ) = \frac { 1 } { | \mathcal { G } | } \sum _ { g } 1 \big [ R ( g ; f _ { \theta } ) \leq 0 . 0 5 \big ] ,\tag{14}
$$

where rank(·) is the rank by measured $\widehat { T }$ within the group (minimum-rank tie breaking). Regret is the quantity we minimise during hyperparameter search and of which we maximise interpretability.

4.3.8 Hyperparameter Search. We tune � with Optuna’s Treestructured Parzen Estimator (TPE), minimising CV in (15). Unless an objective is pinned for an ablation arm, each trial also selects the training loss (ranking versus pointwise regression) as a categorical hyperparameter so TPE can allocate budget to the better-surrogate family. Trials are executed in parallel on multiple GPUs against a shared SQLite study so workers contribute to one global search. Table 10 and Table 11 list the searched ranges; conditional knobs apply only when the trial’s loss is a ranking objective.

For Model B the search budget is deliberately lighter per trial (3- fold × two seeds); the winning configuration is re-validated ofline at 5-fold cross-validation with three independent initialisation seeds before the final production fit. After search completes, the best trial’s hyperparameters are locked and the model is refit once on all of $\mathcal { P } _ { \mathrm { t r } }$

4.3.9 Shape-Grouped Cross-Validation. Hyperparameters are chosen by �-fold cross-validation on $\mathcal { P } _ { \mathrm { t r } }$ only. Folds partition base shapes $( M , N , K ) ;$ : all four layouts of a shape belong to the same fold, so validation never sees a training shape under any layout. Let $s$ be the set of distinct base shapes in $\mathcal { P } _ { \mathrm { t r } }$ and $\{ S ^ { ( k ) } \} _ { k = 1 } ^ { K }$ a partition of S. For fold $k ,$ train $f _ { \theta } ^ { ( k ) }$ on all groups whose base shape lies in $s \backslash s ^ { ( k ) }$ using the chosen training objective $( \mathcal { L } _ { \mathrm { M S E } } , \mathcal { L } _ { \mathrm { R a n k N e t } } .$

Table 10: Hyperparameter search space for Model A (gradientboosted trees).
<table><tr><td>Knob</td><td>Range / values</td><td>Notes</td></tr><tr><td>η (learning rate)</td><td> $\left[ 5 \mathrm { ~  ~ { ~ \times ~ } ~ } 1 0 ^ { - 3 } , 0 . 3 \right]$  uniform</td><td></td></tr><tr><td>max_depth</td><td> $\{ 4 , . . . , 1 2 \}$ </td><td rowspan="5">row subsampling</td></tr><tr><td>min_child_weight</td><td>[1, 30] log-uniform</td></tr><tr><td>subsample</td><td>[0.5, 1.0]</td></tr><tr><td>colsample_bytree</td><td>[0.5, 1.0]</td></tr><tr><td>λ (L2)</td><td>[0.01, 20] log-uniform</td></tr><tr><td>α (L1)</td><td>[10−4, 10] log-uniform</td><td></td></tr><tr><td>γ (min split loss)</td><td>[0,5]</td><td></td></tr><tr><td>nestimators</td><td>{300, . . ., 1200}</td><td>boosting rounds</td></tr><tr><td>loss</td><td>{ndcg, pairwise, mse}</td><td>optional categorical</td></tr><tr><td>rel_top_bands</td><td>{2, ...,5}</td><td>ranking losses only</td></tr><tr><td>CV folds K</td><td>5</td><td>shape-grouped</td></tr></table>

Table 11: Hyperparameter search space for Model B (MLP).
<table><tr><td>Knob</td><td>Range / values</td><td>Notes</td></tr><tr><td>loss</td><td>{LambdaRank, RankNet, optional categorical mse}</td><td></td></tr><tr><td>learning rate</td><td> $[ 1 0 ^ { - 4 } , 5 ~ \times ~ 1 0 ^ { - 3 } ]$  uniform</td><td>log- Adam</td></tr><tr><td>hidden topology</td><td>10 preset tuples</td><td>depths 2-4, widths 32– 1024</td></tr><tr><td>dropout</td><td>[0, 0.4]</td><td>after each ReLU</td></tr><tr><td>weight decay</td><td> $\bar { \{ 0 , 1 0 ^ { - 7 } , 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 1 0 ^ { - 4 } } $  10−³}</td><td></td></tr><tr><td>epochs</td><td> $\{ 2 0 , \ldots , 1 5 0 \}$ </td><td></td></tr><tr><td>batch size</td><td>{1024, 2048, 4096, 8192} MSE only</td><td></td></tr><tr><td>groups per batch</td><td>{32, 64, 128, 256}</td><td>ranking only</td></tr><tr><td>gradient clip (max norm)</td><td> $\{ 0 , 0 . 5 , 1 , 5 , 1 0 \}$ </td><td>ranking only</td></tr><tr><td>max pairs per group</td><td> $\{ 2 \mathbf { k } , \hdots , 6 4 \mathbf { k } \}$ </td><td>ranking only</td></tr><tr><td>gain transform</td><td>{linear, exp}</td><td>LambdaRank only</td></tr><tr><td>CV folds K</td><td>3 during search</td><td>shape-grouped</td></tr><tr><td>train seeds</td><td>2 per fold</td><td>averaged before reporting CV</td></tr><tr><td>pruner</td><td>median, 10 startup trials</td><td>early-stops weak MLP tri- als</td></tr></table>

$\mathcal { L } _ { \mathrm { I } }$ , or the tree ranker analogue), then evaluate on validation groups $\boldsymbol { \mathcal { G } } ^ { ( k ) }$ by computing (11) via (1). The CV score is

$$
\mathrm { C V } ( f _ { \theta } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { | { \cal G } ^ { ( k ) } | } \sum _ { g \in { \cal G } ^ { ( k ) } } R \big ( g ; f _ { \theta } ^ { ( k ) } \big ) .\tag{15}
$$

Bayesian optimisation (Optuna TPE) minimises CV over tree depth, learning rate, regularisation, ensemble size, hidden widths, dropout, pair caps, and—when not fixed—the choice among ranking versus regression objectives. $\mathcal { P } _ { \mathrm { e v } }$ is never used in (15).

4.3.10 Final Fit and Test Protocol. After hyperparameters are fixed, we refit $f _ { \theta }$ once on all training groups in $\mathcal { P } _ { \mathrm { t r } }$ (no held-out fold). The reported generalisation result is the evaluation of (11)–(1) on the 68 oracle groups in $\mathcal { P } _ { \mathrm { e v } }$ , with breakdowns by problem regime (square, tall, wide, skinny-�, large). Baselines computed on the same measured data include uniform random selection, whose expected regret is $\mathbb { E } _ { g } [ 1 - \bar { y } ( g ) ]$ with $\begin{array} { r } { \bar { y } ( g ) = \frac { 1 } { | C ( g ) | } \sum _ { c } y ( g , c ) } \end{array}$ , and a single fixed configuration chosen by best mean � over groups where it appears in at least 80% of evaluation groups.

![](images/d19ba1806aa6b3d40f39ade040fa63e1165b5fdfaf4067b63a72a6b5abaf369a.jpg)  
Figure 5: MLP validation NDCG@10.

4.3.11 Relation between Training Loss and Validation Metric. Measured throughput always defines the supervision signal (� and �), but the default training objectives are ranking losses that penalise incorrect orderings within C(�) rather than absolute error in ${ \widehat { T } } .$ Cross-validation and final testing, by contrast, always score the deployed rule (1) through regret (11). This deliberate mismatch— optimise surrogate ranking loss, validate selection regret—matches the compiler use case: only the identity of the argmax matters, and mis-ordering among top candidates is far more costly than mis-predicting throughput on clearly suboptimal tiles.

## 5 Evaluation

We run our experiments on a gh200 node of the Daint supercomputer on CSCS Alps, equipped with 4 GH200 GPUs running SUSE Linux Enterprise Server 15 SP6. We compile CUTLASS from the commit 3476ddb7 using CUDA 13.1. The environment runs Python 3.12.3 with nvidia-matmul-heuristics-0.1.0.27 as the baseline; we are unable to compare against other related baselines [19, 24] since they are not implemented for the Hopper architecture.

## 5.1 Training

We train two model families, an MLP and XGBoost, under both the full hardware-aware representation and a structural-only ablation. For the MLP, we compare MSE, RankNet, and LambdaRank objectives; for XGBoost, we compare squared-error regression, pairwise ranking, and NDCG listwise ranking. We monitor validation NDCG@10 during training.

Figure 5 and Figure 6 show the validation trajectories. All objectives converge stably. With the full feature set, MSE achieves the highest validation NDCG@10 for both MLP (0.968) and XGBoost (0.960). On structural features, MSE again performs best, reaching 0.969 for MLP and 0.929 for XGBoost.

Validation NDCG is not perfectly aligned with held-out selection regret. In particular, MLP RankNet attains slightly lower validation NDCG than MSE but achieves the lowest held-out regret among the MLP variants. We therefore report the best held-out configuration for each model family and retain the objective-level comparison as a training analysis rather than treating validation NDCG as a direct proxy for final selection quality.

![](images/d619a75c74fc3e0ee10af2f59e67a74ad586540fa896a840950b7a84a6fd55ec.jpg)  
Figure 6: XGBoost validation NDCG@10.

![](images/16787538e714eaea07321e2c312a26d4b7e1b1bd505b162b6992a01d4a37edf8.jpg)

![](images/10ccff5d309b760880fad27c0a90a4a253599f332f03edb385bf64e87f3b7adc.jpg)  
Figure 7: Full and structural features across training objectives.

![](images/2c382458f6a49239cf27b37415237e2fd1265d5c28dad98f43652d0f5c6665f1.jpg)  
Figure 8: Mean selection regret.

Hardware-aware features improve held-out selection consistently across objectives. The gain is modest for the MLP and larger for XGBoost, suggesting that tree models benefit more strongly from exposing the derived hardware behavior explicitly.

Figure 7 compares full and structural features across all objectives.

![](images/a6a76151b0c07255271bb79a1541a502ef672ced45a50378df8810b52af12ef8.jpg)  
Figure 9: CDF of per-group selection regret.

Table 12: Selection quality on 68 exhaustively measured shape–layout groups.
<table><tr><td>Method</td><td>Mean</td><td>Median</td><td>Within 1%</td><td>Within 5%</td><td>Top-1</td></tr><tr><td>MLP (full)</td><td>6.2%</td><td>2.6%</td><td>36.8%</td><td>63.2%</td><td>27.9%</td></tr><tr><td>MLP (structural)</td><td>7.4%</td><td>3.1%</td><td>32.4%</td><td>55.9%</td><td>26.5%</td></tr><tr><td>XGBoost (full)</td><td>6.4%</td><td>4.9%</td><td>25.0%</td><td>52.9%</td><td>19.1%</td></tr><tr><td>XGBoost (structural)</td><td>10.7%</td><td>8.3%</td><td>17.6%</td><td>39.7%</td><td>13.2%</td></tr><tr><td>nvMMH (best-of-K)</td><td>12.0%</td><td>10.2%</td><td>0.0%</td><td>25.0%</td><td>0.0%</td></tr><tr><td>nvMMH (top-1)</td><td>17.3%</td><td>14.6%</td><td>0.0%</td><td>10.3%</td><td>0.0%</td></tr><tr><td>Ridge (full)</td><td>23.7%</td><td>20.0%</td><td>4.4%</td><td>16.2%</td><td>1.5%</td></tr><tr><td>Ridge (structural)</td><td>60.8%</td><td>63.6%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>Analytical score</td><td>63.7%</td><td>63.8%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>Random</td><td>71.5%</td><td>一</td><td>一</td><td>一</td><td></td></tr></table>

## 5.2 Evaluation on the Exhaustive Dataset

We consider an MLP with 1.76 M trainable parameters and an XG-Boost ensemble of 916 trees with maximum depth 7.

Table 12 summarizes the main results. On the held out evaluation dataset (Figure 8), the best hardware-aware MLP reduces mean regret to 6.2%, compared with 17.3% for the top-1 nvMMH recommendation and 71.5% for a random valid candidate. Hardware-aware XGBoost MSE follows closely at 6.4%. Hardware-aware features reduce mean regret on the best MLP from 7.4% to 6.2%; for XGBoost, they reduce it from 10.7% to 6.4%. The larger XGBoost gain indicates that the derived features substantially simplify the mapping from configuration parameters to performance.

The learned selectors are also substantially more likely to return near-oracle kernels. MLP RankNet selects a kernel within 5% of the oracle on 63.2% of groups, and XGBoost MSE does so on 52.9%. In comparison, nvMMH top-1 reaches this threshold on only 10.3% of groups. Even an oracle choice among nvMMH’s top-� recommendations reaches it on only 25.0%, showing that much of the remaining gap arises from the candidate ranking itself rather than from the final choice within the shortlist.

We next ask whether the hardware-aware representation could support strong kernel selection without a nonlinear learned model. We compare against two simpler alternatives: a validation-selected analytical score that combines a small set of mechanistic hardware terms, and a ridge regressor that linearly scores candidates using the same feature representation as our learned selectors (Table 12).

![](images/95c3f7c7c029eeedf990c4753e497f781b3e0824ee8787a58d9d07aa7b5331c1.jpg)  
Figure 10: Fraction of evaluation groups selected within 5% of oracle throughput.

The analytical score performs poorly, reaching 63.7% mean regret and never selecting a configuration within 5% of the oracle across the 68 exhaustive evaluation groups. Individual hardware features correlate with throughput in training, yet this compact composition of them fails to rank the performance tail: knowing which mechanisms matter is not the same as knowing how they interact. Under the same linear model family, structural features alone yield 60.8% regret, while adding the hardware-aware representation reduces this to 23.7%. Thus the proposed features expose substantial selection signal even without a large nonlinear model. However, this global linear combination remains worse than nvMMH (17.3%) and far behind the nonlinear MLP and XGBoost selectors (6.2% and 6.4%).

These results separate representation from model expressivity. Hardware-aware features make the problem considerably easier to learn, but a single global linear mapping is insuficient for strong performance in this setting. The nonlinear learned models are therefore important for translating mechanistic features into accurate kernel rankings.

Figure 9 shows that MLP and XGBoost place most of their mass at low regret: hardware-aware MLP and XGBoost have medians of 2.6% and 4.9%, compared with 14.6% for nvMMH, and hardwareaware models sit above their structural counterparts throughout the curve. Figure 10 reports the fraction of groups within 5% of oracle throughput.

Regime breakdown. Selection quality varies by problem geometry. MLP RankNet achieves mean regret of 0.9% on wide GEMMs and 3.3% on tall GEMMs, while square and skinny-� problems remain more dificult. The same trend appears for XGBoost and the structural ablations. nvMMH performs worse across all regimes, with mean regret between approximately 12% and 19%.

![](images/065e256672ccbbc4c911eb8380d3c354b4515cb5591e8c78fcad752f5268ebc6.jpg)

Figure 11: Mean regret by GEMM shape regime.  
![](images/4e707f0ef7ccb864b42005b20a163db84d0a1ccb32a640ab765296398b269bbd.jpg)

![](images/ac790827346beed19612b167ae877ecebd1fc209eef1544e6043a19b5833d388.jpg)  
Figure 12: Regret and within-5% rate by regime for the best full and structural variants.

![](images/6f06c607b38d89df32b198699bf4705444ea1b8c32ee9550713c3bf519aa68a1.jpg)  
Figure 13: MLP of diferent parameter counts.

## 5.3 Model Size Ablations

We next ask whether the benefit ofhardware-aware features persists as model capacity changes. We fix the MSE objective and sweep MLP widths from 16×8 to 1024×1024×512×256, and XGBoost max\_- depth over {2, 3, 4, 6, 8, 11, 14}, using three seeds per configuration and both feature sets. Figure 13 and Figure 14 report held-out mean selection regret.

Hardware-aware features substantially reduce the model capacity required for accurate kernel selection. At the smallest MLP sizes, the structural model incurs up to 14% higher regret, whereas the hardware-aware representation already captures much of the performance structure of the search space. As capacity increases, structural models gradually recover this gap, and only the largest MLP (∼1.7 M parameters) matches the full representation. This effect is even more pronounced for tree models. XGBoost benefits from the hardware-aware representation throughout the entire depth sweep and retains an advantage even at the largest tested depth. In this sense, the hardware-aware features do not merely improve final accuracy; they make kernel-selection itself easier to learn.

![](images/c9a4f5f6a4ec75d2c4970552817c49df38b4b44c2234882354dfa3e1b693f71f.jpg)  
Figure 14: XGB of diferent depths.

## 5.4 Generalization and Broad GEMM Evaluation

The exhaustive evaluation measures selection quality against a true in-space oracle, but necessarily covers only a small number of problems. We therefore evaluate whether the same selectors generalize at scale to 8,000 previously unseen BF16 GEMMs. Half of the shapes are 32-aligned, while the remainder are only 8-aligned, introducing of-grid dimensions that are sparsely represented in the training sweep. The broad evaluation uses 2,000 (�, �, �) triples held out from the production BF16 autotuning databases and evaluates each under all four operand layouts (TN, TT, NN, NT), yielding 8,000 problems. Half of the shapes are fully 32-aligned; the remainder are 8-aligned but not 32-aligned. The latter set tests generalization to of-grid dimensions that are largely absent from the regular training sweep. The performance results are in Table 13.

We compare four learned selectors—MLP and XGBoost MSE, each with the full hardware-aware representation and the structuralonly ablation—against nvMMH. Each learned model returns a sin gle rank-1 CUTLASS configuration. nvMMH emits a rank-1 tile and cluster recommendation but does not completely specify the CUTLASS mainloop, epilogue, and scheduler configuration. We therefore materialize all valid variants consistent with the nvMMH recommendation and report the best measured variant. This gives nvMMH an optimistic comparison relative to the learned selectors, which each receive only one selected kernel.

Table 13: Broad GEMM evaluation on 8,000 held-out problems.
<table><tr><td>Model</td><td>Features</td><td>Geo-mean % roof</td><td>vs. nvMMH</td><td>Win %</td></tr><tr><td>MLP MSE</td><td>Full</td><td>11.9%</td><td>1.14×</td><td>79.6%</td></tr><tr><td>MLP MSE</td><td>Structural only</td><td>11.5%</td><td>1.10×</td><td>73.9%</td></tr><tr><td>XGBoost MSE</td><td>Full</td><td>11.8%</td><td>1.13×</td><td>77.4%</td></tr><tr><td>XGBoost MSE</td><td>Structural only</td><td>11.4%</td><td>1.09×</td><td>70.1%</td></tr><tr><td>nvMMH</td><td>一</td><td>37.6</td><td>10.5%</td><td>一</td></tr></table>

All candidates are compiled and benchmarked on the same GH200 node using the same warmup, iteration-count, and operandrotation protocol as the training measurements. To isolate kernel ranking from tile-scheduler choices, every method uses the same rank-1 rasterization, swizzle, and split-� settings during benchmarking. nvMMH retains its recommended tile geometry, while the learned selectors retain their independently selected tile, schedule, cluster, and pipeline configuration.

Figure 15a places the selected kernels against the GH200 roofline. For the roofline analysis, we use a GH200 peak BF16 tensor-core throughput of 989.5 Tflop/s and memory bandwidth of 4 TB/s. Arithmetic intensity is computed from the mathematical GEMM work and minimum operand trafic. Each light point corresponds to one measured GEMM, while the bold curves show binned median throughput. The hardware-aware MLP and XGBoost selectors produce nearly identical performance envelopes despite their very diferent model architectures, and both consistently outperform nvMMH across the full arithmetic-intensity range. The improvement persists from memory-bound GEMMs through the transition region and into compute-bound problems, indicating that the learned representation captures performance efects beyond any single bottleneck regime.

All methods achieve 100% execution coverage on the 8,000 problems. The hardware-aware MLP reaches 42.9 Tflop/s geometricmean throughput compared with 37.6 Tflop/s for nvMMH, corresponding to a 1.14× improvement. Hardware-aware XGBoost reaches 42.4 Tflop/s, or 1.13× nvMMH. The structural variants remain competitive at 41.2 and 41.0 Tflop/s, but both are consistently below their hardware-aware counterparts.

Measured per problem, the full-feature MLP outperforms nvMMH on 79.6% of cases and full-feature XGBoost on 77.4%. Even the structural variants win on 73.9% and 70.1%, respectively. Together with the exhaustive-oracle evaluation, this shows that the learned selectors’ lower selection regret translates directly into higher realized throughput over a substantially broader problem distribution.

## 5.5 Transfer Learning Experiments

We evaluate whether a selector trained on BF16 can be adapted eficiently to new precisions. We transfer the BF16 models to FP32 and FP8 E4M3 GEMMs using nested subsets of the corresponding target-dtype training corpus, and evaluate on held-out problems against nvMMH.

Figures 15b and 15c show that only modest target-dtype supervision is required to recover strong performance. On FP32, hardware-aware selectors already reach approximately 1.20×– 1.22× geometric-mean speedup over nvMMH using only 5% of the available target shapes, and improve to approximately 1.24× with the full target corpus. The corresponding structural models generally trail the hardware-aware representation, particularly in the low-data regime. FP8 E4M3 presents a noisier, more dificult transfer setting, but the same trend persists. Hardware-aware MLPs reach 1.12× speedup with 5% of the target shapes and 1.19× with the full corpus, while XGBoost remains around 1.15×–1.17× across the sweep. Together, these results show that the selectors retain useful structure across changes in numerical precision, and that ex posing candidate-induced hardware behavior improves adaptation when target-domain measurements are limited.

![](images/050aa0fb6c9853595a1bcfe420eb7334daed318a2996e41df4efbc15c03f6251.jpg)  
(a) GEMM evaluation against GH200 roofline.

![](images/0f177270378f9a40d71e322e75db1b7f72cc9f705ef3fde3059202051c60edba.jpg)  
(b) Cross-precision transfer to FP32.

![](images/f7583d71adbf7992df073a924a254e9a93c6b2a6175f4e7363a0ba6c2c3f4f49.jpg)  
(c) Cross-precision transfer to FP8.

![](images/7738e65dd90f53414f64db849f3e3b545c773f7014b7d0e03c9f5dac44fe3ab3.jpg)  
(d) Cross-epilogue transfer.  
Figure 15: Model Generalization and Transfer experiments.

We repeat the same finetuning protocol for FP16 epilogue-fusion GEMMs, fusing ReLU, GeLU, and/or bias, warm-starting from the unfused BF16 checkpoints. Figure 15d summarizes the results on held-out fusion GEMMs. With only 10% of the fusion corpus, hardware-aware MLP and XGBoost already match nvMMH (1.00× and 1.00× geometric-mean speedup, respectively) and outperform the structural ablation (0.93×).

5.5.1 Transfer Learning Details. For each target dtype, FP32 TN and FP8 E4M3 TN, we construct nested subsets containing {1, 5, 10, 25, 50, 100}% of the 593 target-dtype base training shapes. The same (�, �, �) subsets are used for FP32 and FP8, while the measured candidate kernels difer according to dtype-specific validity and compilation behavior. The fractions therefore measure target-domain shape coverage rather than equal numbers of kernel measurements.

We fix the MSE objective and reuse the model architectures selected for the BF16 study. MLP transfer initializes from the corresponding BF16 checkpoint, while XGBoost is warm-started from the BF16 model by adding 200 trees. We evaluate both the complete hardware-aware representation and the structural ablation. Evaluation uses a disjoint held-out TN set and the same benchmarking protocol as Section 5.4. Learned selectors emit one rank-1 kernel, while nvMMH is evaluated using its best valid schedule realization.

Table 14: Cross-precision transfer at 100% of the target TN training shapes.
<table><tr><td>Dtype</td><td>Method</td><td>Coverage</td><td>Speedup</td><td>Win %</td></tr><tr><td>FP32</td><td>MLP (full)</td><td>89.5%</td><td>1.24×</td><td>74.6%</td></tr><tr><td>FP32</td><td>XGBoost (full)</td><td>96.7%</td><td>1.24×</td><td>78.4%</td></tr><tr><td>FP32</td><td>MLP (structural)</td><td>96.1%</td><td>1.18×</td><td>71.8%</td></tr><tr><td>FP32</td><td>XGBoost (structural)</td><td>91.7%</td><td>1.20×</td><td>68.6%</td></tr><tr><td>FP8 E4M3</td><td>MLP (full)</td><td>65.4%</td><td>1.19×</td><td>47.4%</td></tr><tr><td>FP8 E4M3</td><td>XGBoost (full)</td><td>73.0%</td><td>1.17×</td><td>51.2%</td></tr><tr><td>FP8E4M3</td><td>MLP (structural)</td><td>87.5%</td><td>1.18×</td><td>64.6%</td></tr><tr><td>FP8 E4M3</td><td>XGBoost (structural)</td><td>66.0%</td><td>1.15×</td><td>43.6%</td></tr></table>

At full target-dtype coverage, FP32 transfer reaches 1.24× geometric-mean speedup for both hardware-aware MLP and XG-Boost, with win rates of 74.6% and 78.4%, respectively. Structural variants remain strong but consistently lower, reaching 1.18× and 1.20×. FP8 remains more challenging, particularly in execution coverage, but the hardware-aware MLP and XGBoost still achieve 1.19× and 1.17× geometric-mean speedup over nvMMH.

The low-data regime is particularly informative. At 5% of the target shape pool, the hardware-aware FP32 models already exceed nvMMH by roughly 20%–22%, despite observing only 30 targetdtype GEMM geometries. The corresponding FP8 models also ex ceed nvMMH, demonstrating that much of the useful structure learned from BF16 survives the precision change and can be adapted with comparatively little target-domain data.

We apply the same finetuning protocol to the FP16 epiloguefusion GEMMs. Selectors are warm-started from the unfused BF16 checkpoints and finetuned on nested subsets ofthe 593-shape fusion training corpus, using the same {1, 5, 10, 25, 50, 100}% fractions as the precision-transfer study. We evaluate 1,000 held-out problems with in-vocabulary fusion kinds assigned at evaluation time.

At full fusion training coverage, hardware-aware XGBoost matches nvMMH (1.00× geometric-mean speedup), while the hardware-aware MLP reaches 0.97×. Structural variants remain below at 0.94×–0.94×. All configurations achieve 100% execution coverage.

The low-data fusion regime mirrors the precision-transfer trend. With only 10% of the fusion corpus (59 shapes), hardware-aware MLP and XGBoost already reach 1.00× geometric-mean speedup over nvMMH, while structural models remain lower. The hardwareaware MLP peaks at 1.01× with 25% coverage before settling near parity.

## 5.6 DeepBench Case Study

To test whether the same selectors generalize to an externally defined workload, we evaluate on dense GEMMs from DeepBench [2], a public catalogue of speech-recognition and language-modeling kernels compiled for 2016-era training and inference stacks. Our catalogue contains 151 unique (�, � , �, layout) problems after deduplication. The evaluation set contains 107 training, 39 inferenceserver, and 5 inference-device problems.

![](images/999fac90ee210af4742ecf257f164021727f300e7ffd8c19e569a272cf1bebc2.jpg)  
Figure 16: Speedup vs. nvMMH.

Table 15: DeepBench throughput generalization on 151 problems. Speedup is geometric-mean throughput relative to nvMMH.
<table><tr><td>Model</td><td>Features</td><td>vs. nvMMH</td><td>Win %</td></tr><tr><td>MLP MSE</td><td>Full</td><td>1.12×</td><td>96.0%</td></tr><tr><td>MLP MSE</td><td>Structural only</td><td>1.13×</td><td>92.7%</td></tr><tr><td>XGBoost MSE</td><td>Full</td><td>1.12×</td><td>84.8%</td></tr><tr><td>XGBoost MSE</td><td>Structural only</td><td>1.05×</td><td>59.6%</td></tr><tr><td>Ridge</td><td>Full</td><td>0.83×</td><td>40.4%</td></tr><tr><td>Ridge</td><td>Structural only</td><td>0.27×</td><td>0.0%</td></tr><tr><td>Random</td><td></td><td>0.22×</td><td>0.0%</td></tr><tr><td>nvMMH</td><td></td><td></td><td></td></tr></table>

Table 16: DeepBench speedup vs. nvMMH by catalogue split.
<table><tr><td>Split</td><td>n</td><td>MLP (full)</td><td>MLP (struct.)</td><td>XGBoost (full)</td></tr><tr><td>Training</td><td>107</td><td>1.09×</td><td>1.09×</td><td>1.07×</td></tr><tr><td>Inference server</td><td>39</td><td>1.25×</td><td>1.28×</td><td>1.29×</td></tr><tr><td>Inference device</td><td>5</td><td>1.08×</td><td>1.00×</td><td>1.04×</td></tr></table>

Table 15 and Figure 16 summarize the results. All learned selectors achieve 100% execution coverage on the retained problems. Hardware-aware MLP and XGBoost improve geometric-mean throughput by 12%–13% over nvMMH and win on 85%–96% ofproblems. The structural MLP matches the full model (1.13×), consistent with the observation that structural features extrapolate more robustly when problem geometry departs from the training grid. Ridge and random baselines remain well below nvMMH, confirming that the gains require learned nonlinear ranking rather than a linear surrogate or chance.

The improvement is largest on inference-server GEMMs (1.24×– 1.29× for MLP and XGBoost over nvMMH), which contain the batched speech and language-modeling layers DeepBench was designed to stress. Training-split problems show a smaller but still positive margin (≈ 1.07×–1.09×).

## 6 Related Work

Learned cost models guide iterative search over tensor-program spaces [1, 6, 10, 26, 28] in frameworks like AutoTVM [5, 23] and Ansor [28]. While corpora like TenSet [29] and optimization methods like AdaTune [10] or TLP [25] improve search eficiency, our model scores the entire valid CUTLASS configuration catalogue in a single batch to directly select the optimal implementation. While devicetransfer frameworks [1, 16, 25] rely on complex learned program representations, instead we show that explicit, mechanism-level hardware features provide a strong inductive bias that improves transfer accuracy with fewer target-domain measurements.

While predictive models like ISAAC [22], CUTLASS-tailor [24], and related low-bit selectors [7, 8] show that GEMM execution cost is learnable from configuration descriptors [11], our model uses analytical hardware-behavior features to directly rank and select optimal configurations. Manual analytical models [9, 15] and heuristics [3, 13, 19, 27] avoid training data by hardcoding hardware-mapping rules, but we learn these complex interactions from behavioral features instead.

Architectural innovations like Stream-K [14] and FlashAttention-3 [17] demonstrate that GPU eficiency relies on work partitioning and pipeline coordination rather than just raw computation. We model these underlying mechanisms as compile-time features to select optimal configurations from the vast spaces exposed by CUT LASS.

## 7 Conclusion

We presented a hardware-aware approach to learned CUTLASS kernel selection. Rather than asking a model to infer GPU behavior from raw configuration parameters, we augment each candidate with statically computable estimates of induced execution behavior. Using 4.9 million measured CUTLASS kernels, the resulting representation achieves mean selection regret of 6.2% with an MLP and 6.4% with XGBoost, compared with 17.3% for the top-1 nvMMH recommendation. Its benefit across both model families indicates that the improvement comes from the representation rather than a particular predictor architecture.

More broadly, our results suggest that learned performance models benefit from representing what a configuration does to the hardware, rather than only what the configuration is. Raw param eters often hide efects whose meaning changes with workload and execution regime, while fully analytical models require brittle architecture-specific rules. Hardware-aware representations pro vide an intermediate abstraction: architectural knowledge exposes the relevant mechanisms, while data learns their interactions. As accelerator configuration spaces grow, such behavioral representations may provide a reusable basis for performance models across precisions, operators, and architectures. The central implication is that better performance models may come less from larger models than from better representations aligned with hardware execution.

## AI Acknowledgment

In this work, we used generative AI tools to assist with the implementation and debugging of benchmarking and analysis code, interpret intermediate experimental results, and assist with language editing of the manuscript. We did not use generative AI tools to generate synthetic experimental data, formulate or prove mathematical claims, or fabricate experimental measurements or results. Tasks involving surveys, interviews, transcription of research material, and qualitative or thematic data analysis were not applicable to this work.

All AI-assisted research ideas, methodological suggestions, code, analyses, and manuscript text were reviewed by the authors. AIassisted code was inspected and validated through compilation, testing, and comparison against measured data before being used in the experimental pipeline. Literature suggestions and technical claims were checked against the corresponding papers or documentation. Experimental results were obtained from the authors’ own measurement infrastructure, and their interpretation and the final scientific conclusions were determined by the authors. We take responsibility for the final content of this work, including all text, claims, code, and artifacts produced with the aid of generative AI.

## Code

Our code is available online at https://github.com/spcl/cutlassselector.

## Acknowledgments

This project received support by the ERC PSAP project (Grant Agreement No. 101002047). The research was conducted as part of the FastTrackAI project at the Singapore-ETH Centre, which was established collaboratively between ETH Zurich and the National Research Foundation, Singapore. Additionally, this research is supported by the National Research Foundation, Singapore (NRF), and the Ministry of Digital Development and Information (MDDI) under the AI Visiting Professorship (Award No. AIVP-2025-005). We also thank the Swiss National Supercomputing Center (CSCS) for providing the computational resources used in this work.

## References

[1] Yang Bai, Mingjun Li, Wendong Xu, and Bei Yu. 2025. A learned performance model with transfer learning across GPUs on tensorized instructions. IEEE Transactions on Parallel and Distributed Systems 36, 9 (2025), 1904–1919.

[2] Baidu. 2016. DeepBench: Benchmarking deep learning operations on diferent hardware. https://github.com/baidu-research/DeepBench.

[3] Harrison Barclay, Ian Tramble, Michał Kukuła, Michel Migdal, Nicholai Tukanov, and Leopold Cambier. 2025. Improving GEMM Kernel Auto-Tuning Eficiency on NVIDIA GPUs with Heuristics and CUTLASS 4.2 | NVIDIA Technical Blog. https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuningeficiency-on-nvidia-gpus-wi\th-heuristics-and-cutlass-4-2/

[4] Ganesh Bikshandi and Jay Shah. 2023. Developing CUDA Kernels for Accelerated Matrix Multiplication on NVIDIA Hopper Architecture using the CUTLASS Library. Colfax Research (2023).

[5] Tianqi Chen, Lianmin Zheng, Eddie Yan, Ziheng Jiang, Thierry Moreau, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy. 2018. Learning to optimize tensor programs. In Proceedings of the 32nd International Conference on Neural Information Processing Systems (Montréal, Canada) (NIPS’18). Curran Associates Inc., Red Hook, NY, USA, 3393–3404.

[6] Junkyeong Choi, Hyucksung Kwon, Woongkyu Lee, Jungwook Choi, and Jieun Lim. 2022. Learning from distinctive candidates to optimize reduced-precision convolution program on tensor cores. arXiv preprint arXiv:2202.06819 (2022).

[7] Hong Guo, Nianhui Guo, Christoph Meinel, and Haojin Yang. 2024. Low-bit CUTLASS GEMM Template Auto-tuning using Neural Network. In 2024 IEEE International Symposium on Parallel and Distributed Processing with Applications (ISPA). 394–401. doi:10.1109/ISPA63168.2024.00057

[8] Hong Guo, Nianhui Guo, Christoph Meinel, and Haojin Yang. 2026. Leveraging Large-Scale Data for Eficient Low-Bit CUTLASS GEMM Optimization via Neural Networks. Big Data Mining and Analytics 9, 2 (2026), 632–652. doi:10.26599/ BDMA.2025.9020065

[9] Aaron Jarmusch and Sunita Chandrasekaran. 2026. Microbenchmark Driven Analytical Performance Modeling Across Modern GPU Architectures. arXiv:2605.04178 [cs.DC] https://arxiv.org/abs/2605.04178

[10] Menghao Li, Minjia Zhang, Chi Wang, and Mingqin Li. 2020. AdaTune: adaptive tensor program compilation made eficient. In Proceedings of the 34th International Conference on Neural Information Processing Systems (Vancouver, BC, Canada) (NIPS ’20). Curran Associates Inc., Red Hook, NY, USA, Article 1241, 13 pages.

[11] Xiaoteng Liu and Pavly Halim. 2024. Understanding GEMM Performance and Energy on NVIDIA Ada Lovelace: A Machine Learning-Based Analytical Approach. arXiv:2411.16954 [cs.DC] https://arxiv.org/abs/2411.16954

[12] Weile Luo, Ruibo Fan, Zeyu Li, Dayou Du, Qiang Wang, and Xiaowen Chu. 2024. Benchmarking and Dissecting the Nvidia Hopper GPU Architecture. In 2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS). 656–667. doi:10.1109/IPDPS57955.2024.00064

[13] NVIDIA. 2025. NVIDIA Matmul Heuristics — nvMatmulHeuristics. https: //docs.nvidia.com/cuda/nvidia-matmul-heuristics

[14] Muhammad Osama, Duane Merrill, Cris Cecka, Michael Garland, and John D. Owens. 2023. Stream-K: Work-Centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU. In Proceedings ofthe 28th ACM SIGPLAN Annual Symposium on Principles and Practice ofParallel Programming (Montreal, QC, Canada) (PPoPP ’23). Association for Computing Machinery, New York, NY, USA, 429–431. doi:10.1145/3572848.3577479

[15] Dezhi Ran, Shuxiao Xie, Mingfang Ji, Anmin Liu, Mengzhou Wu, Yuan Cao, Yuzhe Guo, Hao Yu, Linyi Li, Yitao Hu, Wei Yang, and Tao Xie. 2026. KernelBand: Steering LLM-based Kernel Optimization via Hardware-Aware Multi-Armed Bandits. arXiv:2511.18868 [cs.LG] https://arxiv.org/abs/2511.18868

[16] Jaehun Ryu and Hyojin Sung. 2021. Metatune: Meta-learning based cost model for fast and eficient auto-tuning frameworks. arXiv preprint arXiv:2102.04199 (2021).

[17] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. 2024. FlashAttention-3: fast and accurate attention with asynchrony and low-precision. In Proceedings ofthe 38th International Conference on Neural Information Processing Systems (Vancouver, BC, Canada) (NIPS ’24). Curran Associates Inc., Red Hook, NY, USA, Article 2193, 28 pages.

[18] Rupanshu Soi, Rohan Yadav, Fredrik Kjolstad, Alex Aiken, Maryam Mehri Dehnavi, Michael Garland, and Michael Bauer. 2025. Optimal Software Pipelin ing and Warp Specialization for Tensor Core GPUs. arXiv:2512.18134 [cs.PL] https://arxiv.org/abs/2512.18134

[19] Ryan Swann, Muhammad Osama, Xiaohu Guo, Bryant Nelson, Lixun Zhang, Alex Brown, Yen Ong, Ali Yazdani, Sean Siddens, Ganesh Dasika, et al. 2025. tritonblas: Triton-based analytical approach for gemm kernel parameter selection. arXiv preprint arXiv:2512.04226 (2025).

[20] Vijay Thakkar, Cris Cecka, Tejash Shah, Jay Shah, Paul VanKoughnett, and Ryo Asai. 2025. CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design | NVIDIA Technical Blog. https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-andcomposable-abstractions-\for-gemm-kernel-design/

[21] Vijay Thakkar, Pradeep Ramani, Cris Cecka, Aniket Shivam, Honghao Lu, Ethan Yan, Jack Kosaian, Mark Hoemmen, Haicheng Wu, Andrew Kerr, Matt Nicely, Duane Merrill, Dustyn Blasig, Aditya Atluri, Fengqi Qiao, Piotr Majcher, Paul Springer, Markus Hohnerbach, Jin Wang, and Manish Gupta. 2023. CUTLASS. https://github.com/NVIDIA/cutlass/tree/v3.0.0

[22] Philippe Tillet and David Cox. 2017. Input-aware auto-tuning of compute-bound HPC kernels. In Proceedings ofthe International Conference for High Performance Computing, Networking, Storage and Analysis (Denver, Colorado) (SC ’17). Association for Computing Machinery, New York, NY, USA, Article 43, 12 pages. doi:10.1145/3126908.3126939

[23] Xingfu Wu, Praveen Paramasivam, and Valerie Taylor. 2023. Autotuning Apache TVM-based Scientific Applications Using Bayesian Optimization. In Proceedings ofthe SC ’23 Workshops ofthe International Conference on High Performance Computing, Network, Storage, and Analysis (Denver, CO, USA) (SC-W ’23). Association for Computing Machinery, New York, NY, USA, 29–35. doi:10.1145/3624062.3626079

[24] Yongseung Yu, Donghyun Son, Younghyun Lee, Sunghyun Park, Giha Ryu, Myeongjin Cho, Jiwon Seo, and Yongjun Park. 2023. Tailoring CUTLASS GEMM using Supervised Learning. In 2023 IEEE 41st International Conference on Computer Design (ICCD). 465–474. doi:10.1109/ICCD58817.2023.00077

[25] Yi Zhai, Yu Zhang, Shuo Liu, Xiaomeng Chu, Jie Peng, Jianmin Ji, and Yanyong Zhang. 2023. TLP: A Deep Learning-Based Cost Model for Tensor Program Tuning. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (Vancouver, BC, Canada) (ASPLOS 2023). Association for Computing Machinery, New York, NY, USA, 833–845. doi:10.1145/3575693.3575737

[26] Kaixuan Zhang, Chutong Ding, Shiyou Qian, Luping Wang, Jian Cao, Guangtao Xue, Cheng Huang, Guodong Yang, and Liping Zhang. 2026. WaveTune: Waveaware Bilinear Modeling for Eficient GPU Kernel Auto-tuning. arXiv preprint arXiv:2604.10187 (2026).

[27] Xiao Zhang, Yaoyao Ding, Bolin Sun, Yang Hu, Tatiana Shpeisman, and Gennady Pekhimenko. 2026. Hexcute: A Compiler Framework for Automating Layout Synthesis in GPU Programs. arXiv:2504.16214 [cs.LG] https://arxiv.org/abs/2504. 16214

[28] Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, Ameer Haj-Ali, Yida Wang, Jun Yang, Danyang Zhuo, Koushik Sen, Joseph E. Gonzalez, and Ion Stoica. 2020. Ansor: generating high-performance tensor programs for deep learning. In Proceedings of the 14th USENIX Conference on Operating Systems Design and Implementation (OSDI’20). USENIX Association, USA, Article 49, 17 pages.

[29] Lianmin Zheng, Ruochen Liu, Junru Shao, Tianqi Chen, Joseph E Gonzalez, Ion Stoica, and Ameer Haj Ali. 2021. Tenset: A large-scale program performance dataset for learned tensor compilers. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1).