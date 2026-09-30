# Mira: Memory-Efficient MoE Inference Using Adaptive Caching and Predictive Expert Staging

Sanjali Yadav, Bahar Asgari

University of Maryland, College Park

{sanjali7, bahar}@umd.edu

Abstract—Mixture-of-Experts (MoE) models are a compelling architecture for scaling model capacity, making them especially attractive for deployment on resource-constrained, single-GPU systems. However, this benefit is difficult to realize because expert parameters dominate memory, and token-level routing is dynamic, unpredictable, and skewed. Prior work using offloading and caching remains fundamentally reactive as systems wait for router outputs before moving experts, leading to inefficient cache utilization, and an inability to overlap transfers with compute under tight VRAM budgets.

To address these challenges, we propose Mira, an algorithmsystem co-design that enables high-capacity MoE inference on a single GPU. Mira shifts from a reactive to a proactive stance by coupling predictive expert management with a tailored quantization format. It introduces lightweight per-layer predictors that anticipate expert usage two layers ahead, enabling proactive prefetching. These predictions feed a two-tier HOT+STAGE GPU cache managed by token-level routing telemetry to retain frequently used experts while staging predicted ones. To minimize transfer overhead, Mira implements a custom compression for expert parameters, which reduces metadata and improves packing efficiency, while minimally degrading accuracy.

Mira is implemented as a fully integrated runtime that coordinates predictors, caching policies, and quantized transfers to maximize overlap between communication and compute. Our experiments show that Mira reduces expert-induced stalls. Compared against state-of-the-art baselines, Mira achieves a 5.71× speedup in average throughput on a memory-constrained GPU. It accelerates Time-to-First-Token by 11.71× and achieves a 3.84× average speedup in beam search inference, demonstrating its effectiveness across diverse inference scenarios.

## I. INTRODUCTION

Large Language Models (LLMs) deployed as chat-style interfaces [4] have become a standard part of modern software stacks, powering search, productivity tools, programming assistants, and consumer applications. The dominant deployment model involves large-scale distributed inference systems, which are the focus of substantial research and capital investment aimed at serving millions of users [29], [40]. Concurrently, a compelling need for local, on-device LLM inference is emerging [1], [14], [37]. This trend is driven by critical requirements that cloud-centric systems cannot easily satisfy: data privacy and security, particularly when handling sensitive healthcare or government data [3]; the ability to personalize models through fine-tuning on confidential datasets [26]; and democratizing access to LLMs, especially for users without access to high-end GPUs or large-scale compute clusters.

However, executing local inference presents a significant system optimization challenge driven by the vast disparity between the memory footprint of the state-of-the-art models, which often contain billions of parameters, and the limited VRAM in typical consumer-grade single-GPU setups. The local inference environments lack the terabytes of highbandwidth memory and massive parallelism found in industryscale accelerator clusters, making the task to optimize them non-trivial, as every available gigabyte of VRAM must be optimally utilized to accommodate the model’s parameters and activation cache.

Additionally, the quest for higher generation quality has driven a massive increase in model scale, which in turn has escalated computational requirements. To address these high compute demands, Mixture of Experts (MoE) models have been proposed as an alternative to traditional dense LLMs [7], [10], [35]. The primary advantage of the MoE architecture is its ability to decouple model capacity from computational cost; unlike dense models that activate all parameters for every token, MoE models employ sparse activation where a trained router network contextually selects a small subset of experts to process each token. This allows the model to scale its parameter count significantly while keeping the per-token compute cost manageable.

However, the strengths of MoE at the algorithm level translate into acute system-level pressure on memory and I/O. The aggregate parameter count of the experts dominates the model’s memory footprint, and the token-dependent, sparse activation pattern causes highly skewed and dynamic expert usage. On resource-constrained, single-GPU setups, this combination creates a challenge: experts cannot all reside on GPU in high precision, yet loading them just-in-time from CPU DRAM risks adding PCIe latency on the critical path.

Prior work largely addresses this memory challenge through two orthogonal techniques: expert offloading [8], [17], [19], [35] and quantization [11], [21], [22]. Offloading keeps expert parameters in CPU memory and migrates only the selected experts to GPU when the router activates them. Quantization reduces the bit-precision of parameters to shrink their storage footprint and accelerate transfers over the PCIe bus. Recent systems combine both ideas [9], [31], [38], [47], aggressively quantizing experts while offloading them to CPU, and then using general-purpose caching or eviction policies to decide which experts to keep resident on GPU.

Yet, this hybrid approach leaves an optimization gap. First, existing systems typically treat quantization as a passive compression step and ignore the rich, token-level routing signals available in MoE, resorting to coarse-grained policies such as recency-based eviction [9], [34], [43], or static popularity heuristics [19], [43], [44]. Second, expert offloading is often reactive: the system learns which experts are needed only after the gate has already made its decision, forcing the runtime into an unfavorable trade-off. Fetching experts late incurs stalls on CPU to GPU transfers, while speculative prefetching without high accuracy wastes VRAM on experts that may never be used. The alternative relies on ultra-low-bit quantization to keep all experts on GPU, trading I/O stalls for potential degradation in model quality.

To address these limitations, we introduce Mira<sup>1</sup>, an algorithm-system co-design for MoE inference on memoryconstrained GPUs. Mira couples modest, accuracy-preserving post-training quantization with an adaptive, prediction-driven expert cache that is explicitly aware of token-level routing behavior. At the algorithm level, Mira trains lightweight perlayer predictors that learn to anticipate which experts will be activated in future MoE layers. At the system level, Mira uses these predictions to pre-stage just the right experts into GPU memory, managing VRAM as a two-tier HOT+STAGE cache backed by a CPU-resident, INT8-compressed expert store.

Concretely, Mira makes three key design choices:

• Quantization-aware expert store: Mira applies a custom INT8 post-training quantization exclusively to MoE expert parameters, while keeping non-expert layers in higher precision. All experts are stored as CPU-pinned compressed tensors, enabling high-throughput CPU to GPU transfers and a unified representation across offloaded, staged, and hot-resident experts. This design avoids ultra-low-bit quantization that risks quality loss, while still reducing the memory and bandwidth cost per expert.

• Predictive expert staging: Mira augments each MoE layer with a small predictor that uses features derived from the current hidden states, router logits, and recent routing history to forecast the top experts required in the next MoE layers (e.g., an i → i+2 pattern). These predictions drive a speculative prefetcher that stages only the most likely experts into GPU memory, overlapping CPU to GPU transfers with ongoing computation and reducing PCIe-induced stalls.

• Adaptive HOT+STAGE caching under VRAM budgets: Mira partitions GPU memory into a stable HOT region for long-lived, frequently used experts and a smaller STAGE window for predicted experts. At execution time, Mira maintains lightweight per-expert metrics (e.g., routed volume and cache hit/miss behavior) and uses them to continuously rebalance the HOT set. This metric-driven accounting allows Mira to evict underutilized experts from GPU, promote frequently used experts from CPU to GPU, and adapt to changing workloads without manual tuning.

By tightly integrating predictive expert selection with a quantization-aware caching and staging runtime, Mira enables high-capacity MoE models to run efficiently on a single, memory-constrained GPU. Our evaluation shows that Mira reduces expert-migration stalls and improves end-to-end throughput over prior offloading schemes, while maintaining model quality and supporting larger MoE configurations within the same VRAM budget. Our evaluation shows that Mira improves end-to-end throughput by 10.46× over Deep-Speed and 2.86× over MoE-Infinity on a 48GB GPU, and by 5.71× over MoE-Infinity on a 24GB GPU, while reducing time-to-first-token by up to 18.38× and achieving up to a 3.84× speedup over Fiddler in beam-search decoding, all with accuracy close to an FP16 baseline on downstream tasks.

## II. BACKGROUND

## A. Mixture of Experts

Dense LLMs adopt a uniform computation pattern in which every token is processed by all layers and all parameters of the model. In a standard Transformer-based LLM [39], each layer consists of self-attention, normalization and a feed-forward network (FFN); for each input token, all the components are executed regardless of the type of token. As a result, the per-token compute cost scales linearly with the number of parameters: increasing model capacity (e.g., wider FFNs or more layers) directly increases both training and inference FLOPs. This tight coupling between capacity and compute has driven interest in architectures that can increase the number of parameters without proportionally increasing the cost of serving each token.

![](images/5246a3a9c9a4b7a140868304c00a65984f28a5b8b38a3cd618dedb839b40a553.jpg)  
Fig. 1. The architecture of dense transformer block compared to sparse MoE block.

MoE models provide such an architecture by introducing conditional computation in place of the dense FFN blocks. In a typical MoE layer, illustrated in Figure 1, the single FFN is replaced by a pool of E experts, each an FFN with its own parameters. The router network computes a score W over experts for each token and activates only the topk experts. In Figure 1, top-k is set to 2. The token is then processed by the selected subset of experts, and their outputs are combined. Because only $k \ll E$ experts are active per token, the per-token compute and memory bandwidth cost can be kept close to that of a dense layer, while the total parameter count of the model grows roughly with E. This conditional execution decouples capacity from per-token compute. MoE models can achieve effective model sizes in the hundreds of billions or trillions of parameters while maintaining inferencetime FLOPs comparable to much smaller dense models [7], [10].

Prior work shows that this additional capacity can improve perplexity and downstream task performance at fixed or modestly increased compute budgets, and can shift the empirical scaling laws of LLMs in a favorable direction [7], [35]. From an algorithmic perspective, the experts can specialize on different regions of the input space, languages, or tasks, and the router learns to allocate computation accordingly. At the same time, MoE models introduce new challenges that do not arise in dense LLMs. First, while only a few experts are active per token, all experts must be stored somewhere in the system, and their aggregate parameter count dominates the memory footprint of the model.

In large MoEs models, the experts often account for the majority of parameters in the network, far exceeding the attention and non-expert layers. In large-scale datacenter training, this motivates distributing experts across many devices and using all-to-all communication to route tokens to remote experts [24]. In single-node inference scenarios, it creates pressure on GPU memory capacity and PCIe bandwidth, since most experts are idle for any given token but must remain available for routing decisions. Second, the token-dependent routing behavior introduces load imbalance and dynamism that are absent in dense models.

Dense LLMs have a fixed and predictable compute graph: every token incurs the same sequence of operations, enabling straightforward static scheduling and batching. In contrast, the MoE router produces a highly skewed and evolving distribution of expert usage, where some experts may be hot and frequently selected, while others are rarely used except on specific inputs or languages. This skew impacts both algorithmic considerations (e.g., load-balancing losses during training) and systems concerns (e.g., which experts should be kept resident on GPU during inference, and which can be offloaded to CPU or disk without incurring stalls).

## B. LLM Quantization

Quantization is a standard technique for reducing the memory and compute cost of neural network inference by representing weights and activations in low-precision integer formats (e.g., INT8, INT4) instead of high-precision floating point [5], [6], [25], [33], [41]. In most LLM deployments this is implemented as uniform affine quantization, where each tensor or channel is mapped to a small integer range using a calibrated scale factor.

At inference time, the common design is weight-only quantization where weights are stored in low precision and dequantized on the fly into higher-precision accumulators, or consumed directly by integer GEMM kernels with dequantization fused into the epilogue [38], [47]. This shrinks the footprint of model parameters and the bandwidth required to move them between CPU and GPU, which is particularly important for single-GPU setups where VRAM and PCIe bandwidth are the dominant bottlenecks.

Formally, post-training quantization (PTQ) typically uses uniform affine quantization. Given a full-precision tensor $w \in \mathbb { R } ^ { n }$ and a bit-width b, the integer range is $[ q _ { \mathrm { m i n } } , q _ { \mathrm { m a x } } ] =$ $[ - 2 ^ { b - 1 } , 2 ^ { b - 1 } - 1 ]$ for signed quantization. PTQ chooses a scale $s > 0$ and zero-point $z \in \mathbb { Z }$ , and maps each element as

$$
q _ { i } = \mathrm { c l i p } \big ( \mathrm { r o u n d } ( w _ { i } / s ) + z , q _ { \mathrm { m i n } } , q _ { \mathrm { m a x } } \big ) , \qquad \hat { w } _ { i } = s \cdot ( q _ { i } - z ) ,\tag{1}
$$

where $q _ { i }$ is the stored integer and $\hat { w } _ { i }$ is the dequantized approximation.

PTQ methods typically use per-channel or block-wise granularity for weights to better track local statistics, while leaving especially sensitive components such as embeddings, layer norms, and final output layers in FP16/BF16. MoE architectures amplify the benefits of this heterogeneous treatment: expert MLPs dominate the parameter count and are accessed sparsely, making them attractive targets for aggressive compression, whereas the shared backbone and router networks are relatively small but latency-critical and thus often kept at higher precision to preserve routing behavior and overall quality [12].

## III. MOTIVATION

## A. Quantization Tradeoff

Table I illustrates the core challenge in quantizing MoE experts for offloaded inference by reporting the measured perplexity on a subset of WikiText-2 under different quantization schemes. As the bit-width of expert weights is reduced from FP16 to INT8 and then to INT4, the perplexity increases, degrading generation quality. This degradation, however, comes with the benefit of substantially lower CPU to GPU transfer time, almost a $2 \times$ reduction as quantization becomes more aggressive, so more aggressive quantization directly translates into faster expert migrations. In practice, though, ultra-low-bit formats can spend a larger fraction of time in dequantization and related bookkeeping, especially in local inference settings with single-batch decoding, where there is insufficient compute to hide the dequantization overhead.

TABLE I  
PERPLEXITY ON WIKITEXT-2 FOR VARYING QUANTIZATION BIT WIDTHS.
<table><tr><td></td><td>FP16</td><td>INT8</td><td>INT8 (Mira)</td><td>INT4</td><td>INT3</td><td>INT2</td></tr><tr><td>Perplexity ↓</td><td>3.9324</td><td>3.9695</td><td>3.9554</td><td>4.1048</td><td>5.2653</td><td>1147.5236</td></tr></table>

Importantly, the transfer volume is not solely determined by the nominal bit-width of the weights. Modern low-bit schemes rely on increasingly fine-grained scaling (e.g., per-channel or per-group scales, sometimes with additional outlier paths) to recover accuracy at 4 bits and below. As a result, an expert migration over PCIe consists not only of the quantized weight matrix but also of its associated scales and metadata, which may be stored at higher precision. Under aggressive groupwise quantization, the number of scale values grows with the number of groups, thus this metadata can become a non-trivial fraction of the overall transfer volume and an additional source of memory traffic and kernel complexity during dequantization [42].

![](images/6ce5973bcfc51016917ed1ab1cbb739ef50a500044693ad4112c3ddf246e75b0.jpg)

![](images/282f8e34902c2633fa8ac1489644d499503aa9d092640d2fb34d74f77ad5cedf.jpg)

![](images/67b08009f0b636995dd09d6c2a435464fb9d93828c0be2e1740fe0300e01e600.jpg)  
MoE Layers

![](images/9a3f6373ef6076614113110d49793b437fe03e8e0daeedad3b88b03d2be358f0.jpg)  
Fig. 2. Aggregated expert activation patterns for Mixtral-8x7B [15] on four distinct MMLU tasks . Each heatmap shows the activation frequency (color) for each of the 8 experts (y-axis) across the 32 MoE layers (x-axis).

Many recent systems sidestep this tradeoff by co-designing quantization schemes with specialized accelerator support. Integer-only methods such as I-BERT [20] and I-LLM [16] quantize the entire transformer computation graph to low-bit integers and execute all operators using integer arithmetic on hardware INT8/INT4 units, explicitly avoiding any dequantization back to floating point during inference. Likewise, hardware-oriented PTQ frameworks such as ZeroQuant [45] and SmoothQuant [41] target W8A8 or mixed INT4/INT8 formats that are executed directly by low-precision GEMM kernels on tensor-core GPUs, while recent FP8 recipes for LLMs rely on native FP8 Tensor Cores in Hopper-class accelerators to run transformer matmuls in FP8 end-to-end rather than treating FP8 as a mere storage format [28], [30].

While recent accelerators provide native INT8/INT4/FP8 tensor units and many quantization schemes are explicitly codesigned for such hardware, this makes the inference stack contingent on access to specific GPU generations and highend datacenter GPUs, which runs counter to our goal of democratizing LLM inference. In contrast, our goal is to design a modest quantization regime that remains effective across commodity single-GPU environments without assuming specialized low-precision compute support, while still reducing expert transfer cost (including metadata overhead) and preserving model quality on standard language modeling benchmarks. This design target motivates the quantization strategy we introduce in Mira (details in Section IV-B).

## B. Sparse Expert Activation

MoE models rely on sparse expert activation. For each token, a small subset of experts (e.g., top-k) is selected by the router, while the remaining experts remain idle. This sparsity is highly skewed and input-dependent as a few experts may be frequently activated for common workloads, while a long tail of experts fires rarely but is essential for maintaining model quality on specific topics or user queries. Figure 2 illustrates the expert activation behavior of Mixtral-8x7B [18] on four representative workloads: dark-humor detection, movie recommendation, vitamin fact verification, and physics questions from the MMLU dataset [15]. Each panel visualizes a single task: the x-axis enumerates the 32 MoE layers in Mixtral, the y-axis indexes the eight experts within each layer, and the color encodes the fraction of routed tokens in that task assigned to a given expert at that layer (i.e., per-layer activation frequency). Bright cells correspond to hot experts that receive a large share of tokens in that layer, while darker cells correspond to rarely selected experts.

These heatmaps highlight several salient patterns. First, within a task, expert selection is skewed across layers. Most layers are dominated by a few experts, which appear as bright horizontal bands, while the remaining experts in that layer are only weakly activated for certain tokens. Thus, a small subset of experts per layer handles the bulk of traffic, and the others are only sporadically used at a given time during decode phase in local inference. Second, expert activation patterns vary substantially across tasks. For the vitamin-facts dataset, the activation is comparatively more uniform than for the other workloads. Moreover, the set of the hot experts per layer change with the task. These observations underscore that expert popularity is both layer and workload dependent, which in turn motivates adaptive cache management that can respond to evolving expert activation patterns rather than relying on fixed popularity statistics.

Consequently, maintaining a static GPU-resident cache of popular experts, for example, pinning the top experts per layer based on offline profiling, as in static caching schemes such as Fiddler [19], is insufficient. Static caches assume a stable routing distribution; they cannot adapt when the query mix shifts over time, when the model processes long-context inputs where topics drift within a single sequence, or when a user’s local workload deviates from the distribution used to build the cache. As a result, they either waste VRAM on experts that are no longer hot or incur frequent cache misses for newly important experts, leading to PCIe stalls and degraded throughput.

A natural alternative is to react to routing decisions by prefetching experts for the next layer, e.g., an i → i+1 prefetching policy that stages experts predicted to be used in the following MoE layer [17]. While such next-layer prefetching can hide some latency in large-batch, high-throughput settings, it is often inadequate for the single-batch, interactive inference regime that dominates local LLM usage. In the single-batch case, there is limited opportunity to overlap PCIe transfers with useful computation, so any misprediction or late prefetch directly translates into additional end-to-end latency.

![](images/8cfd5a5499d063c32f9192096e303f28273756526e749ea961548cc8b4aee2e2.jpg)  
Top 15 Expert IDs  
Fig. 3. Top-15 expert activations across decode tokens on the movie recommendation dataset using Mixtral-8x7B. Each column tracks the activation frequency of one expert over a decode sequence.

When experts are stored in higher-precision formats (e.g., INT8 instead of ultra-low-bit formats), each migration is more expensive, making i+1 prefetching even less effective, as the system must wait for the top-k experts to arrive to finish computation. Further, inaccurate fetching wastes both bandwidth and scarce VRAM. These limitations motivate a design in which a HOT cache and a STAGE prefetch window work together, with both regions explicitly aware of sparse and dynamic activation patterns. The HOT region holds a dynamically maintained set of experts that are consistently popular over recent routing history, providing low-latency access for the majority of tokens. The STAGE region, in contrast, is driven by a predictive prefetcher that targets experts that are not currently hot but are likely to be needed soon.

Crucially, the HOT set must be continuously rebalanced. In a model like Mixtral-8x7B, which has 8 experts per layer across 32 layers (i.e., 256 experts in total), keeping all experts resident would exceed the capacity of a single commodity GPU, so only a subset of experts can live in GPU memory at any given time, and which experts deserve those slots must change as the workload evolves. Figure 3 provides a zoomedin view of Figure 2; we focus on the 15 most frequently activated experts on the movie recommendation dataset and track how the hot expert set evolves over decode tokens. Across decode steps, no single expert remains dominant. For example, expert 59 is heavily activated for early tokens but becomes inactive later on; beyond that point, keeping it in the HOT cache provides little benefit. If it is needed again sporadically, it can be brought in via the prefetcher. In contrast, expert 28 is not frequently activated at the beginning but becomes active for a long contiguous span of later tokens, making it a strong candidate to be promoted into the HOT cache during that phase. This figure is a small snapshot of what happens during decoding. Over longer decode phases, expert activations exhibit even richer temporal variation. By combining predictive staging for out-of-the-blue experts with lightweight, online rebalancing of the HOT set, the system can track the evolving sparse activation structure of the MoE, reducing both cold misses and unnecessary transfers while respecting tight VRAM budgets in local inference environments.

## IV. MIRA: AN ALGORITHM-SYSTEM CO-DESIGN

This section presents Mira, our algorithm-system codesigned runtime for MoE transformers. Mira jointly optimizes three components: a custom INT8 quantization layout tailored to expert feed-forward blocks, a two-level GPU cache for expert weights, and a predictive prefetcher with adaptive rebalancing driven by MoE routing signals. Mira is designed to be compatible with any transformer MoE that exposes experts and routing decisions via standard interfaces.

## A. Design Objectives and Overview

Mira is designed around three primary goals: (1) Model Generality: The runtime must support a variety of MoE models, avoiding hardcoded assumptions about expert layout, routing scheme, or training procedure. Its interface requires only the ability to enumerate experts, access their weights, and observe router assignments. (2) Single-GPU Efficiency: The system must be VRAM-aware and deliver high performance on a single accelerator, including commodity devices with modest memory capacity, without relying on multi-GPU expert sharding or specialized tensor-core formats. (3) Adaptive Self-Management: Mira must operate as a self-sufficient system that dynamically drives its own caching, prefetching, and memory allocation policies. Our adaptive cache and predictive expert staging are the core mechanisms designed to achieve these goals.

Figure 4 conceptually illustrates Mira’s runtime components. During sequential layer-by-layer inference, Mira’s runtime mainly impacts how the expert FFN is computed 1 . All expert weights are first converted offline into a unified, custom INT8 representation 2 . At runtime, Mira maintains the GPU memory as a two-level expert cache 3 . The HOT region holds resident experts that demonstrate frequent reuse, which can be invoked with no host-to-device traffic. The STAGE region acts as a prefetch window, holding experts expected to be needed in the near future based on predictor output.

During computation, experts are fetched from this cache. Mira employs a custom kernel that fuses the dequantization of INT8 weights with the FFN computation itself 4 . To facilitate this prefetching, Mira attaches a small predictor head to each applicable MoE layer (i). This head consumes hidden states and router statistics from layer i to predict which experts will be required two layers ahead (i + 2) 5 . These predictions initiate asynchronous data transfers from the CPU-pinned store into the STAGE region. Concurrently, Mira continuously gathers token-level telemetry, quantifying expert assignments, data source tiers (HOT, STAGE, or CPU), and cache miss rates. This data feeds a rebalancer that adaptively optimizes the HOT set composition over time 6 .

![](images/4c5e57179fb7d788f83606d14e44b035636df97c28c55b8e18c141ca2ae015a8.jpg)  
Fig. 4. Mira’s Runtime Components: At runtime, Mira maintains a two-level GPU expert cache with HOT and STAGE regions. During inference, experts are fetched from this cache and processed using a custom kernel that fuses dequantization with FFN computation. A lightweight predictor forecasts which experts will be needed shortly and triggers asynchronous prefetching into the STAGE region. Meanwhile, token-level telemetry informs a rebalancer that periodically adjusts the HOT expert set for optimal reuse. Offline, on CPU side, all expert FFN weights are converted into a unified INT8 format.

The design is intentionally non-invasive relative to the base transformer model. Components like embedding layers, selfattention sublayers, layer norms, and the final LM head remain in BF16 on the GPU and execute unchanged. Only the expert FFNs are wrapped by Mira’s runtime: when the model reaches an MoE layer and calls its experts, that call is redirected into Mira’s execution pipeline, which handles quantized storage, cache lookup, and dequantization. As a result, the interface seen by the rest of the model, particularly the shapes and semantics of hidden states and output logits, remains identical to the baseline.

## B. Quantization-Aware Expert Store

Mira’s custom quantization representation targets MoE architecture where each expert is a three-matrix feed-forward block with an up projection $W _ { 1 }$ , a down projection $W _ { 2 } ,$ and a gate modulation matrix $W _ { 3 }$ with the up and gate matrices $W _ { 1 }$ and $W _ { 3 }$ have shape $R \times C$ , while the down matrix $W _ { 2 }$ has shape $C \times R ,$ where R is the expert hidden dimension and C is the model hidden dimension.

Mira converts each expert into a compact INT8 format. The up and gate matrices are stored as INT8 tensors of shape $R \times$ $C ,$ and the down matrix is stored as an INT8 tensor of shape $C \times R .$ A single BF16 vector $s \in \mathbb { R } ^ { R }$ stores per-row scales that are shared across all three matrices. The quantization is constructed as follows. For each row index r, Mira computes

$$
\alpha _ { r } = \operatorname* { m a x } \left( \| W _ { 1 } [ r , : ] \| _ { \infty } , \| W _ { 3 } [ r , : ] \| _ { \infty } , \| W _ { 2 } [ : , r ] \| _ { \infty } \right) ,
$$

and defines a scale

$$
s _ { r } = \frac { \operatorname* { m a x } ( \alpha _ { r } , \epsilon ) } { 1 2 7 } ,
$$

where ϵ is a small constant to avoid division by zero. The three matrices are then quantized using this shared per-row scale:

$$
\hat { W } _ { 1 } [ r , c ] = \mathrm { r o u n d } \left( \frac { W _ { 1 } [ r , c ] } { s _ { r } } \right) ,\tag{2}
$$

$$
\hat { W } _ { 3 } [ r , c ] = \mathrm { r o u n d } \left( \frac { W _ { 3 } [ r , c ] } { s _ { r } } \right) ,\tag{3}
$$

$$
\hat { W } _ { 2 } [ c , r ] = \mathrm { r o u n d } \left( \frac { W _ { 2 } [ c , r ] } { s _ { r } } \right) ,\tag{4}
$$

and each value is clamped to the range [−127, 127] so it can be stored in a signed 8-bit integer. Our custom INT8 scheme employs joint compression, utilizing a single scale factor shared across all three matrices, unlike standard permatrix INT8 quantization. As shown in Table I and Section VI-B, this approach incurs a negligible impact on model quality compared to the standard INT8 baseline.

The primary benefit of our shared-scale INT8 format is reducing per-expert staging overhead in MoE inference, where experts are repeatedly migrated and prefetched under limited GPU memory. Standard per-row INT8 stores one scale vector per matrix, requiring multiple scale transfers and additional copy operations each time an expert is staged. In contrast, our scheme uses a single shared scale per expert, reducing scale metadata movement and the number of CPU → GPU copy operations during staging. Although end-to-end staging time is typically dominated by transferring the INT8 weight matrices, these metadata and copy-call savings accumulate when staging multiple experts (e.g., during prefetch) and reduce transferside overhead. In a scale-only microbenchmark, shared scaling reduces CPU → GPU scale transfer latency by 1.64× on Mixtral-8x7B, by 1.62× on Mixtral-8x22B, and by 1.68× on DeepSeek-V2-Lite.

This INT8 representation is deliberately moderate. We avoid ultra-low-bit schemes such as INT3 or INT2, which require elaborate packing, additional dequantization metadata, and often incur noticeable accuracy loss. INT8 offers a pragmatic compromise: it reduces the per-expert footprint sufficiently to make large-scale caching and offloading attractive, yet it remains conceptually simple, hardware-portable, and amenable to efficient dequantization strategies.

From an implementation perspective, our joint quantization scheme’s only firm prerequisite is that expert FFNs are expressible as a three-matrix (up, down, gate) transformation. This pattern is characteristic of a broad class of contemporary MoE architectures, making our technique widely applicable. Furthermore, Mira’s runtime is modular: if an MoE model deviates from this structure, it can employ a standard quantization technique while still leveraging Mira’s core inference components.

![](images/b009cab522661fb4fff1fa5b729b79874d978445c56ee9ac85b4b863fa735039.jpg)  
Fig. 5. Mira’s memory layout. Compressed expert weights W<sub>1</sub>, $W _ { 2 } ,$ and $W _ { 3 }$ are combined (denoted by +) on the CPU. The weights and a shared scale are packed (indicated by the dashed line) and transferred to the GPU.

## C. HOT+STAGE Expert Cache in GPU Memory

At runtime, Mira manages GPU memory dedicated to experts according to an explicit byte-level budget. Mira estimates how much capacity must be reserved for non-expert model layers and BF16 working buffers (workspace reserve) used during. The remainder is available for caching expert weights. This remaining budget is expressed in units of expert slots. Since the compressed expert size is known, Mira can compute how many experts can be resident simultaneously. It then partitions the budget into two regions dedicated to HOT and STAGE, respectively. The partitioning is dynamic and depends on whether the runtime is operating under tight memory constraints. Figure 5 illustrates this memory layout.

The HOT partition in Mira contains experts that are fully resident as compressed INT8 tensors in GPU memory and are expected to exhibit long-term reuse. They remain in the cache until evicted by the rebalancing policy. The STAGE partition acts as a prefetch window. It holds experts that have been copied to GPU memory in anticipation of near-future use, because the predictor has identified them as likely to be activated in upcoming layers. STAGE entries are shortlived; they are dropped when their target layer falls sufficiently behind the layer currently being processed.

Mira adapts its cache behavior to the available VRAM budget, operating in either a small-VRAM or large-VRAM mode. This distinction primarily governs the provisioning and eviction policies for the STAGE partition. In large-VRAM mode, the STAGE partition is provisioned generously. Experts are granted longer residency, and eviction policies are less aggressive. This larger capacity provides a longer observation window, allowing the runtime to more accurately identify frequently demanded experts for eventual promotion to the HOT cache. Conversely, the small-VRAM mode operates under an extremely tight budget where only a small percentage of experts can be GPU-resident, making the STAGE partition’s capacity minimal and not intended for long-term residency.

In this small-VRAM mode, the STAGE partition is provisioned to hold only a handful of predicted experts for immediate future layers (e.g., layer i+2 while processing layer i), and its memory is aggressively reclaimed post-computation. If the STAGE partition is full when a new expert is predicted, the runtime declines to stage that expert, defaulting to the

CPU path when that expert is eventually required. This design strictly conserves VRAM while still allowing low-capacity GPUs to exploit predictive staging and overlap whenever the STAGE partition can accommodate it.

This aggressive eviction makes efficient HOT set management paramount, and we found that Mira’s rebalancer provides significant benefits over standard cache policies such as Least Recently Used (LRU) or Least Frequently Used (LFU). A conventional LRU policy is suboptimal, as it promotes staged experts based on recency of use, not frequency, which pollutes the HOT cache with potential cold experts that may not be subsequently reused. A pure LFU policy is also not optimal, as it suffers from cache pollution (where old, popular experts never get evicted). Mira’s rebalancer, in contrast, acts as a history-based gatekeeper. It uses the STAGE partition as a probationary area, maintaining comprehensive usage statistics for all experts in both STAGE and HOT partitions. An expert is only promoted from STAGE after the rebalancer observes a persistent history of frequent access. This check ensures that only experts truly worthy of the high-value HOT cache capacity are promoted, avoiding issues with standard caching policies.

## D. Adaptive HOT Cache Rebalancing

To effectively determine HOT residency, Mira maintains fine-grained, token-level accounting. For each expert, the runtime tracks two key statistics: (1) the total number of tokens routed to it by the MoE router, and (2) the source memory tier for those tokens (i.e., served from HOT, STAGE, or CPU). These counters are maintained over a sliding window of tokens and are reset after each rebalancing interval.

From these raw counts, the runtime computes higher-level metrics by categorizing the service path for each token. A HOT-served token is one processed by an expert already resident in the high-performance HOT cache, which we use to measure cache utilization. A STAGE-served token is one processed by an expert that was successfully pre-fetched into the STAGE partition, which measures prediction effectiveness. Finally, a CPU-served token represents a complete cache miss where the expert had to be fetched from CPU memory upon demand, which directly quantifies the cost-of-miss.

This accounting provides a concrete link between abstract MoE routing decisions and their physical system costs. For example, an expert with a high routed volume but also a high CPU-served count incurs significant PCIe and dequantization overhead, making it a strong promotion candidate. Conversely, a HOT-resident expert that serves few tokens (i.e., low HOTserved count relative to its peers) represents poor VRAM utilization and becomes a candidate for eviction.

After processing a predetermined number of tokens, Mira triggers its rebalancing pass, which operates in three phases.

• Victim Scoring: The rebalancer examines each HOTresident expert and assigns a victim score. Experts with low routed volume in the current window receive high scores, marking them for eviction. This score also incorporate cache-level metadata, such as recency and frequency

• Candidate Scoring: Concurrently, it considers all non-HOT experts with recent traffic and assigns a candidate score. This score prioritizes experts with high routed volume and a high number of CPU-served tokens, indicating both high demand and a high cost-of-miss

• Bounded Swapping: Finally, the rebalancer performs a bounded number of swaps, evicting the worst-scoring HOT residents and promoting the best-scoring candidates. The number of swaps per interval is intentionally limited to prevent cache thrashing. The token threshold to trigger balancing and the number of swaps are offline-tuned hyperparameters, derived from a performance sweep to find the optimal trade-off between rebalancing overhead and cache adaptability for a given MoE architecture.

This rebalancing procedure is fundamentally MoE-aware in a way that generic caching policies (e.g., LRU/LFU) are not. It does not treat experts as anonymous, uniformly-sized cache lines. Instead, it leverages the semantic signal of MoE routing decisions and the provenance of served tokens (i.e., HOT, STAGE, or CPU) to make cost-aware decisions about which experts have earned residency.

## E. Predictive Expert Staging

Adaptive caching alone is insufficient to hide expert offload latency on a single GPU. To move beyond this limitation, Mira uses per-layer predictor heads that attempt to forecast expert usage a few layers ahead. For each MoE layer i, except the last two, we attach a small neural module that produces a probability distribution over experts in layer i + 2.

The predictor consumes a compact feature vector constructed from the state at layer i. Specifically, Mira mean-pools the hidden activations across tokens to obtain a single vector summarizing the current semantic context, averages the router logits to capture the marginal tendency of the router to favor each expert, and builds a histogram over the experts actually selected at that layer. In addition, Mira identifies the top-k experts who receive the most tokens at layer i. These features together encode both the content being processed and the router’s current behavior in an inexpensive-to-compute form.

The predictor head itself, which we call a MixtureHead, is implemented as a dynamic mixture of two sub-models. The first branch, an ActHead, processes the semantic context and router logits. The second branch, a HistHead, processes the historical expert selection features. These two branches produce separate logit predictions, which are then combined using a contextual gate. This small gating network learns to output a dynamic weight, α, that adaptively balances the influence of the activation-based and history-based branches for each prediction.

Training of the predictor heads is entirely offline and does not modify the base model. The runtime collects traces in which it records the feature vector at layer i and the actual expert activations at layer i + 2 for a representative workload. Each predictor is trained with a standard cross-entropy loss to align its output distribution with the empirical distribution of routed tokens across experts in the target layer. During inference, the trained predictors are loaded to the GPU and run in evaluation mode, adding minimal computational overhead relative to the main expert FFNs.

![](images/466c10bf350b861fba8ffd186bb6b3b76c9055d98f29b27e9fe51e04575afd9b.jpg)  
Fig. 6. Timeline illustration of expert computation under varying hardware setups.

During inference, for each expert predicted for layer i + 2, Mira checks if the expert is already present in HOT or STAGE partition, the runtime records a prefetch skipped event and moves on. If the expert is not yet cached, the prefetch controller attempts to admit a new STAGE entry. Admission triggers an asynchronous copy of the compressed expert from the CPU-pinned store to GPU memory on a high-priority CUDA stream. The controller also records that expert e is staged for target layer i + 2 so that repeated staging requests for the same layer do not issue redundant copies.

Our choice of a two-layer lookahead (predicting for layer i + 2 at layer i) is dictated by the system’s latency profile as illustrated in figure 6. Even with INT8 compression, the end-to-end latency of fetching an expert, including the PCIe transfer, on-device dequantization to BF16, and the predictor’s own inference overhead, is non-trivial. The i+2 target provides a sufficiently large window to overlap this data transfer with the active computation of layers i and i + 1. This i + 2 window represents a critical sweet spot. We experimentally implemented and evaluated a one-layer lookahead (i + 1) predictor, but found it was insufficient to fully hide this latency, especially in single-token inference. Conversely, looking further ahead would increase prediction uncertainty and risk wasting bandwidth. Empirically, our i + 2 predictor provided the necessary slack time while, importantly, achieving the same prediction accuracy as the i + 1 model, indicating that sufficient predictive signals exist two layers deep.

## F. Algorithm-System Co-Design

Mira derives its effectiveness from the tight coupling between algorithmic insight and systems design. On the algorithmic side, it introduces per-layer predictors trained on real routing traces and a custom quantization scheme tuned to the expert FFN structure. These components exploit MoEspecific patterns while leaving the base model and router unmodified. On the systems side, Mira provides a two-level (HOT+STAGE) GPU cache with explicit VRAM budgeting, a predictive prefetcher that acts on the predictor’s signals, and utilization-aware policy for HOT rebalancing.

This co-design is exemplified by the expert execution pipeline, which is carefully structured to manage the overhead of our custom quantization. To keep this overhead manageable, Mira pre-allocates BF16 working buffers on the GPU for the dequantized weights and scales. These buffers are reused across experts and decoding steps to eliminate allocator churn on the critical path. The runtime then orchestrates execution using two CUDA streams: a dequantization stream copies INT8 data and BF16 scales (initiating H2D transfers from pinned memory if needed), and a compute stream performs the FFN computation. The expert’s FFN computation is fused with weight dequantization, avoiding the need to materialize the full-precision weights in VRAM.

Neither the algorithmic nor the systems components would be as effective in isolation. Predictors are inert without a cache system to act on their signals, and a naive cache, blind to MoE routing structure, would struggle to anticipate expert demands under tight VRAM budgets. By designing both pieces together, Mira turns the statistical regularities of expert usage into concrete reductions in PCIe stalls and improved effective throughput, all while maintaining model quality. In Section VI, we quantify these benefits and show that Mira enables larger MoE models to be served within constrained VRAM budgets.

## V. EXPERIMENTAL SETUP

Our experiments are designed to achieve several key objectives. First, we benchmark Mira’s inference performance against state-of-the-art baselines that specialize in expert offloading, compression, or hybrid approaches. Second, we evaluate Mira’s task accuracy. Third, we demonstrate Mira’s robustness by assessing its performance across different hardware setups with varying VRAM capacities.

Baselines: We employ following state-of-the-art baselines for single-GPU inference on memory-constrained systems: DeepSpeed-MII [27], an inference system that utilizes ZeRO-Infinity [32] to offload expert parameters to CPU and NVMe; Mixtral-Offloading [9], an MoE framework that integrates expert prediction and caching mechanisms with 4-bit quantization; MoE-Infinity [43], an offloading system that uses sequence-level activation tracing to perform sparsity-aware expert prefetching and caching; Fiddler [19], a CPU-GPU coexecution system that minimizes data transfer overhead by dynamically offloading portions of the expert computation to the CPU; and HQQ, a 2-bit weight-only quantization with 128 group size [2].

Datasets: To evaluate inference performance, we use ShareGPT [46], a dataset of human-chatbot conversations chosen to model realistic expert selection patterns, randomly sampling conversations with varying input and output lengths; this setup is similar to the Fiddler [19] and FloE [47].

Our comparative analysis focuses mainly on Mixtral-8x7B [18], as it is the only model supported across the open-sourced codebases of key baselines Fiddler [19] and Mixtral Offloading [9], but also includes DeepSeek-V2-Lite-Chat to demonstrate Mira’s generalizability.

Metrics: For a comprehensive performance view, we report two key metrics: Time-To-First-Token (TTFT) to measure initial latency, which is especially important for interactive uses cases and end to end tokens-per-second to measure throughput of the system. We also report accuracy on downstream tasks performance using lm-eval-harness [13] by sampling from various datasets.

System configurations: We evaluated Mira on three distinct hardware environments to test performance across different VRAM capacities and memory bandwidths. Environment 1 featured an NVIDIA RTX 6000 Ada (48GB VRAM, 960 GB/s bandwidth) paired with an Intel(R) Xeon(R) w5-2565X. Environment 2 used an NVIDIA RTX A5000 (24GB VRAM, 768 GB/s bandwidth) with an AMD EPYC 7352. Environment 3 consists of NVIDIA RTX 3070 (8GB VRAM, 448 GB/s bandwidth) with an Intel(R) Xeon(R) w5-2565X. The number of processor cores is restricted to 8 to mirror resourceconstrained environments.

## VI. EVALUATION

## A. Performance

Figure 7 illustrates the performance of Mira compared to the baselines in Environment 1, which features 48GB of VRAM. In terms of end-to-end throughput (TPS) averaged across different input and output lengths, Mira+P (with prefetching enabled) achieves an average speedup of 10.46× over Deep-Speed, 1.25× over HQQ, 2.86× over MoE-Infinity and 1.55× over Fiddler. These end-to-end time reports for Mira include the overhead of our predictor and rebalancer, and we observe that these additional runtime components do not have a significant performance impact. The speedup over Mira (prefetching disabled) is 1.07×. This modest gain is expected, as the 48GB VRAM allows the HOT cache to hold a large number of experts. Consequently, the prefetcher has a limited role during decoding because most predicted experts are already resident in the cache, and the few misses do not contend for I/O. This memory advantage also benefits other baselines; Fiddler sees better performance as fewer computations are offloaded to the CPU, and The HQQ baseline also fits fully on-device, but its performance is bound by the computational overhead of dequantizing its highly aggressive quantization scheme.

The performance advantages are more pronounced in the Time-to-First-Token (TTFT) metric, which is critical for interactive sessions. Here, Mira+P achieves an average speedup of 18.38× over DeepSpeed, 11.71× over Fiddler, 7.49× over MoE-Infinity, and 3.62× over Mixtral Offloading. The HQQ baseline also has low prefill overhead, because the entire model is fully resident in GPU VRAM, eliminating any I/O latency. Notably, Mira+P demonstrates a 1.85× speedup over Mira for TTFT. This significant gain is because the prefill stage processes multiple tokens per layer, activating a far greater number of experts than in the single-token decode stage. The prefetcher excels in this scenario, correctly anticipating the diverse expert demand and ensuring those weights are staged on the GPU in time for computation, thus effectively hiding I/O latency.

![](images/c6e871b88a744acb18e36502c61efc9bef140f091a5742eb8399246b295bca08.jpg)  
Fig. 7. Inference performance comparison of Mira+P (with prefetcher) and Mira (without prefetcher against baselines in Environment 1 (48GB VRAM), measured by Time-to-First-Token (TTFT) and throughput (Tokens/s) for varying input and output token lengths. Higher is better for Tokens/s; lower is better for TTFT.

![](images/f1cb3214c894955deb55ca7b8606191fce87cce8c57f82d229b70d851e578f1e.jpg)  
Fig. 8. Inference performance comparison of Mira+P (with prefetcher) and Mira (without prefetcher against baselines in Environment 2 (24GB VRAM), measured by Time-to-First-Token (TTFT) and throughput (Tokens/s) for varying input and output token lengths. Higher is better for Tokens/s; lower is better for TTFT.

Figure 8 presents the performance results for Environment 2, which is significantly more memory-constrained with 24GB of VRAM (half of Environment 1). In this environment, we had to omit two baselines. DeepSpeed was excluded as it required an excessive amount of CPU RAM for offloading, which was unavailable in this environment and it was slow in general. This observation aligns with prior work [19], [47], which also noted DeepSpeed sub-optimal for such memoryconstrained inference. We also omitted Fiddler, as its performance is critically dependent on the AVX512 instruction set, which the CPU in this environment lacks. This hardware mismatch resulted in extremely poor performance (we observed a 21.81× average speedup over it), making a direct comparison misleading. This highlights a key advantage of Mira as it imposes no such limiting hardware requirements, supporting our goal of democratizing LLM access.

In terms of end-to-end throughput, Mira+P achieves a 5.71× average speedup over MoE-Infinity, 3.86× over HQQ, and a 1.32× speedup over Mira. For the TTFT metric, Mira+P demonstrates advantages, achieving a 5.61× over MoE-Infinity, 1.23× over HQQ and 1.42× over Mira-Base. The speedup over Fiddler was 46.9×. These TTFT gains of Mira+P, particularly over Mira, are directly attributable to the prefetcher, which effectively hides I/O latency that becomes a critical bottleneck in this memory-constrained setting.

![](images/7de534d336bf994db261a4f469111f608e2d451387e5b7414bfe23abfc3944a0.jpg)  
Fig. 9. Mass-weighted Coverage@k by source layer for Mixtral-8x7B. The predictor selects the top-k experts to prefetch for a future MoE layer; Coverage@k reports the fraction of routed token mass that lands on those prefetched experts. Higher coverage indicates more effective prefetching.

## B. Accuracy Analysis

TABLE II  
ACCURACY COMPARISON ON DOWNSTREAM TASKS
<table><tr><td></td><td>Arc-E</td><td>SCIQ</td><td>Winogrande</td><td>TriviaQA</td></tr><tr><td>FP16</td><td>0.870</td><td>0.970</td><td>0.760</td><td>0.57</td></tr><tr><td>Mira</td><td>0.871</td><td>0.975</td><td>0.750</td><td>0.54</td></tr></table>

Table II compares the accuracy results of Mira with the FP16 baseline across various downstream tasks. More aggressive quantization systems, such as FloE [47], have reported notable accuracy degradation on downstream tasks, around 4–7%. In contrast, INT8 quantization [5] is a well-established and robust technique. When paired with our custom method, it avoids the significant precision loss and outlier-clipping issues that plague lower-bit formats.

We designed our custom INT8 scheme to strike a balance. It provides sufficient model compression to reduce the PCIe data transfer payload and mitigate offloading costs, but does so without incurring the substantial accuracy degradation of more aggressive methods. This sweet spot ensures high model fidelity, which is critical for a practical single-GPU inference system, while still yielding tangible performance benefits from reduced data movement.

In Figure 9, we evaluate how well the predictor supports expert prefetching in Mixtral-8x7B using mass-weighted Coverage@k. For each source layer on the x-axis, the predictor outputs a ranked list of experts for a future MoE layer (at offset ∆); we then measure what fraction of the actual routed token mass at that target layer falls within the predicted topk experts. Interpreting k as the prefetch budget, Coverage@1 corresponds to prefetching a single expert, while Coverage@2 and Coverage@3 correspond to prefetching the top two or three experts, respectively. The consistently higher values for larger k show that a small prefetch budget captures most of the routing traffic, indicating that the predictor concentrates probability mass on the experts that dominate routing and would therefore reduce expert stalls and improve cache effectiveness.

![](images/214f9cf1bfe09a8d24b768a1a0d605c2f9b3fe008bd18ffa76139e3b53da0e49.jpg)  
Fig. 10. Mira’s performance gain over Fiddler with beam width scaling.

TABLE III  
MIRA AND MIRA+P THROUGHPUT ACROSS VARYING VRAM BUDGETS.
<table><tr><td>VRAM (GB)</td><td>Mira (Tokens/s)</td><td>Mira+P(Tokens/s)</td><td>Speedup</td></tr><tr><td>16</td><td>1.70</td><td>2.53</td><td>1.49</td></tr><tr><td>24</td><td>2.77</td><td>3.87</td><td>1.40</td></tr><tr><td>32</td><td>3.65</td><td>4.48</td><td>1.23</td></tr><tr><td>40</td><td>5.26</td><td>7.29</td><td>1.39</td></tr><tr><td>48</td><td>8.97</td><td>9.66</td><td>1.08</td></tr></table>

## C. Additional Results

Figure 10 illustrates how Mira’s performance changes as we increase the beam width in beam search decoding. In our earlier experiments, we effectively used greedy decoding (beam width 1). Fiddler also analyzes larger beam widths to evaluate how well the system sustains high-quality generation via beam search under resource-constrained settings. In the single-batch setting, increasing the beam width increases the number of candidate sequences decoded in parallel, stressing the system similarly to a larger effective batch while aiming for better-quality outputs.

Following Fiddler’s setup, we fix the input and output lengths to 64 and 256 tokens and sweep the beam width. While Fiddler reports its performance gain over llama.cpp in terms of end-to-end tokens per second, here we report our performance gain over Fiddler. We show that both Mira and Mira+P outperform Fiddler as the beam width increases. On average Mira achieves a 1.45× speedup and Mira+P achieves a 3.84× speedup. These results demonstrate that Mira scales well in beam-search settings that target higher-quality generation.

Lastly, to isolate and quantify the prefetcher’s benefit under varying memory constraints, Table III presents an ablation study. This experiment was conducted in Environment 1, but we artificially limited the VRAM budget available to the runtime to simulate low-memory scenarios. We used a fixed sequence length (64 input, 256 output). The results support our main findings in Figures 7 and 8, demonstrating that the performance gain from prefetching increases as the VRAM budget decreases. With a smaller budget, the HOT cache shrinks, making cache misses more frequent and costly. The prefetcher’s ability to correctly anticipate and stage experts is therefore critical to mitigating this I/O bottleneck.

## VII. RELATED WORK

Recent MoE inference systems for commodity hardware, such as MoE-Infinity [43], Fiddler [19], FloE [47], and Mixtral-Offloading [9], utilize expert offloading, compression, and basic caching heuristics. Orthogonal works, like Pre-gated MoE [17], modify the MoE architecture itself to decouple selection from execution. In contrast, Mira retains the original architecture and introduces a predictive, token-level expert cache. It combines a dedicated INT8 representation with a HOT+STAGE cache, driven by a novel i → i + 2 expert predictor. Mira thus acts as a drop-in cache manager whose policies are guided by dynamic routing statistics, not static popularity or generic heuristics.

Parallel work on dense LLMs, independent of MoE routing, also addresses memory pressure. Systems like FlexGen [36] and DeepSpeed Zero-Inference [30] offload dense weights, while vLLM [23] optimizes the KV. Similarly, post-training quantization methods [2], [11], [12], [21], [25], [41], [45] uniformly compress all model weights, agnostic to MoE routing. Mira’s design is complementary. It applies a moderate, expert-only INT8 scheme and, crucially, exploits MoE-specific routing signals to drive its dynamic cache placement and prefetching policies.

## VIII. CONCLUSIONS

This work introduced Mira, an algorithm-system co-design whose effectiveness stems from its proactive, MoE-aware architecture. By introducing a lightweight predictor that forecasts expert usage two layers in advance, Mira transitions from a reactive to a predictive stance, staging experts onto the GPU before they are needed. This predictive prefetching, combined with a novel, telemetry-driven HOT+STAGE cache and an accuracy-preserving INT8 representation, allows Mira to effectively hide I/O latency without compromising model integrity.

By coupling this predictive staging with an intelligent, adaptive runtime, Mira mitigates the severe memory bottlenecks inherent to MoE inference. It enables high-throughput, highfidelity large language model execution on widely available consumer hardware, marking a significant step toward democratizing access to state-of-the-art AI.

## REFERENCES

[1] Y. Anand, Z. Nussbaum, B. Duderstadt, B. Schmidt, and A. Mulyar, “Gpt4all: Training an assistant-style chatbot with large scale data distillation from gpt-3.5-turbo,” https://github.com/nomic-ai/gpt4all, 2023.

[2] H. Badri and A. Shaji, “Half-quadratic quantization of large machine learning models,” November 2023. [Online]. Available: https://mobiusml.github.io/hqq blog/

[3] Z. by PrivateGPT, “PrivateGPT,” May 2023. [Online]. Available: https://github.com/zylon-ai/private-gpt

[4] S. K. Dam, C. S. Hong, Y. Qiao, and C. Zhang, “A complete survey on llm-based ai chatbots,” 2024. [Online]. Available: https: //arxiv.org/abs/2406.16937

[5] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, “Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale,” Advances in neural information processing systems, vol. 35, pp. 30 318–30 332, 2022.

[6] P. Dong, L. Lu, C. Wu, C. Lyu, G. Yuan, H. Tang, and Y. Wang, “Packqvit: Faster sub-8-bit vision transformers via full and packed quantization on the mobile,” Advances in Neural Information Processing Systems, vol. 36, pp. 9015–9028, 2023.

[7] N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu, M. Krikun, Y. Zhou, A. W. Yu, O. Firat, B. Zoph, L. Fedus, M. Bosma, Z. Zhou, T. Wang, Y. E. Wang, K. Webster, M. Pellat, K. Robinson, K. Meier-Hellstern, T. Duke, L. Dixon, K. Zhang, Q. V. Le, Y. Wu, Z. Chen, and C. Cui, “Glam: Efficient scaling of language models with mixture-of-experts,” 2022. [Online]. Available: https://arxiv.org/abs/2112.06905

[8] Z. Du, S. Li, Y. Wu, X. Jiang, J. Sun, Q. Zheng, Y. Wu, A. Li, H. Li, and Y. Chen, “Sida: Sparsity-inspired data-aware serving for efficient and scalable large mixture-of-experts models,” Proceedings of Machine Learning and Systems, vol. 6, pp. 224–238, 2024.

[9] A. Eliseev and D. Mazur, “Fast inference of mixture-of-experts language models with offloading,” arXiv preprint arXiv:2312.17238, 2023.

[10] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity,” 2022. [Online]. Available: https://arxiv.org/abs/2101.03961

[11] E. Frantar and D. Alistarh, “Qmoe: Practical sub-1-bit compression of trillion-parameter models,” arXiv preprint arXiv:2310.16795, 2023.

[12] Z. Fu, N. Ding, K. Han, X. Yu, X. Li, X. Chen, Y. Tang, and Y. Wang, “Eaquant: Enhancing post-training quantization for moe models via expert-aware optimization,” arXiv preprint arXiv:2506.13329, 2025.

[13] L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, A. Le Noac’h, H. Li, K. McDonell, N. Muennighoff, C. Ociepa, J. Phang, L. Reynolds, H. Schoelkopf, A. Skowron, L. Sutawika, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou, “The language model evaluation harness,” 07 2024. [Online]. Available: https://zenodo.org/records/12608602

[14] E. D. Giacinto, “Localai: The free, open source openai alternative,” https: //github.com/go-skynet/LocalAI, 2023.

[15] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, “Measuring massive multitask language understanding,” arXiv preprint arXiv:2009.03300, 2020.

[16] X. Hu, Y. Cheng, D. Yang, Z. Yuan, J. Yu, C. Xu, and S. Zhou, “Illm: Efficient integer-only inference for fully-quantized low-bit large language models,” arXiv preprint arXiv:2405.17849, 2024.

[17] R. Hwang, J. Wei, S. Cao, C. Hwang, X. Tang, T. Cao, and M. Yang, “Pre-gated moe: An algorithm-system co-design for fast and scalable mixture-of-expert inference,” in 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA). IEEE, 2024, pp. 1018–1031.

[18] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. de las Casas, E. B. Hanna, F. Bressand, G. Lengyel, G. Bour, G. Lample, L. R. Lavaud, L. Saulnier, M.-A. Lachaux, P. Stock, S. Subramanian, S. Yang, S. Antoniak, T. L. Scao, T. Gervet, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed, “Mixtral of experts,” 2024. [Online]. Available: https://arxiv.org/abs/2401.04088

[19] K. Kamahori, T. Tang, Y. Gu, K. Zhu, and B. Kasikci, “Fiddler: Cpu-gpu orchestration for fast inference of mixture-of-experts models,” in Proceedings of the International Conference on Learning Representations (ICLR 2025), 2025, arXiv preprint arXiv:2402.07033. [Online]. Available: https://arxiv.org/abs/2402.07033

[20] S. Kim, A. Gholami, Z. Yao, M. W. Mahoney, and K. Keutzer, “I-bert: Integer-only bert quantization,” in International conference on machine learning. PMLR, 2021, pp. 5506–5518.

[21] Y. J. Kim, R. Fahim, and H. H. Awadalla, “Mixture of quantized experts (moqe): Complementary effect of low-bit quantization and robustness,” 2023. [Online]. Available: https://arxiv.org/abs/2310.02410

[22] Y. J. Kim, R. Henry, R. Fahim, and H. H. Awadalla, “Who says elephants can’t run: Bringing large scale moe models into cloud scale production,” arXiv preprint arXiv:2211.10017, 2022.

[23] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for large language model serving with pagedattention,” in Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

[24] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, “Gshard: Scaling giant models with conditional computation and automatic sharding,” arXiv preprint arXiv:2006.16668, 2020.

[25] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, and S. Han, “Awq: Activation-aware weight quantization for on-device llm compression and acceleration,” Proceedings of machine learning and systems, vol. 6, pp. 87–100, 2024.

[26] H. Lyu, S. Jiang, H. Zeng, Y. Xia, Q. Wang, S. Zhang, R. Chen, C. Leung, J. Tang, and J. Luo, “Llm-rec: Personalized recommendation via prompting large language models,” in Findings of the Association for Computational Linguistics: NAACL 2024, 2024, pp. 583–612.

[27] Microsoft, “Deepspeed-mii,” https://github.com/microsoft/DeepSpeed-MII.

[28] NVIDIA Corporation, “Floating-point 8: An introduction to efficient, lower precision ai training,” https://developer.nvidia.com/blog/floatingpoint-8-an-introduction-to-efficient-lower-precision-ai-training/, 2025, accessed: 2025-11-14.

[29] D. Patel and A. Ahmad, “The inference cost of search disruption–large language model cost analysis,” Verfugbar unter https://www. semianal-¨ ysis. com/p/theinference-cost-of-search-disruption, 2023.

[30] PyTorch Team, “Accelerating Llama 3 FP8 inference with triton kernels,” https://pytorch.org/blog/accelerating-llama3/, 2024, accessed: 2025-11- 14.

[31] S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi, A. A. Awan, J. Rasley, and Y. He, “Deepspeed-moe: Advancing mixtureof-experts inference and training to power next-generation ai scale,” in International conference on machine learning. PMLR, 2022, pp. 18 332–18 346.

[32] S. Rajbhandari, O. Ruwase, J. Rasley, S. Smith, and Y. He, “Zeroinfinity: Breaking the gpu memory wall for extreme scale deep learning,” in Proceedings of the international conference for high performance computing, networking, storage and analysis, 2021, pp. 1–14.

[33] M. Rakka, M. Fournarakis, O. Krestinskaya, J. Bazzi, K. N. Salama, F. Kurdahi, A. M. Eltawil, and M. E. Fouda, “Mixed-precision quantization for language models: Techniques and prospects,” arXiv preprint arXiv:2510.16805, 2025.

[34] R. Sarkar, H. Liang, Z. Fan, Z. Wang, and C. Hao, “Edge-moe: Memory-efficient multi-task vision transformer architecture with tasklevel sparsity via mixture-of-experts,” in 2023 IEEE/ACM International Conference on Computer Aided Design (ICCAD). IEEE, 2023, pp. 01–09.

[35] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, “Outrageously large neural networks: The sparselygated mixture-of-experts layer,” 2017. [Online]. Available: https: //arxiv.org/abs/1701.06538

[36] Y. Sheng, L. Zheng, B. Yuan, Z. Li, M. Ryabinin, B. Chen, P. Liang, C. Re, I. Stoica, and C. Zhang, “Flexgen: high-throughput generative´ inference of large language models with a single gpu,” in Proceedings of the 40th International Conference on Machine Learning, ser. ICML’23. JMLR.org, 2023.

[37] Y. Song, Z. Mi, H. Xie, and H. Chen, “Powerinfer: Fast large language model serving with a consumer-grade gpu,” in Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles, 2024, pp. 590–606.

[38] P. Tang, J. Liu, X. Hou, Y. Pu, J. Wang, P.-A. Heng, C. Li, and M. Guo, “Hobbit: A mixed precision expert offloading system for fast moe inference,” arXiv preprint arXiv:2411.01433, 2024.

[39] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[40] B. Wu, Y. Zhong, Z. Zhang, S. Liu, F. Liu, Y. Sun, G. Huang, X. Liu, and X. Jin, “Fast distributed inference serving for large language models,” arXiv preprint arXiv:2305.05920, 2023.

[41] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han, “Smoothquant: Accurate and efficient post-training quantization for large language models,” in International conference on machine learning. PMLR, 2023, pp. 38 087–38 099.

[42] X. Xie, L. Wang, L. Xiao, M. Han, L. Liu, X. Xu, J. Wang, Z. Song, and X. Liao, “Amove: Accelerating llms through mitigating outliers and salient points via fine-grained grouped vectorized data type,” in Proceedings of the 58th IEEE/ACM International Symposium on Microarchitecture®, 2025, pp. 854–868.

[43] L. Xue, Y. Fu, Z. Lu, L. Mai, and M. Marina, “Moe-infinity: Efficient moe inference on personal machines with sparsity-aware expert cache,” 2025. [Online]. Available: https://arxiv.org/abs/2401.14361

[44] J. Yao, Q. Anthony, A. Shafi, H. Subramoni, and D. K. D. Panda, “Exploiting inter-layer expert affinity for accelerating mixture-of-experts

model inference,” in 2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS). IEEE, 2024, pp. 915–925.

[45] Z. Yao, R. Yazdani Aminabadi, M. Zhang, X. Wu, C. Li, and Y. He, “Zeroquant: Efficient and affordable post-training quantization for largescale transformers,” Advances in neural information processing systems, vol. 35, pp. 27 168–27 183, 2022.

[46] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica, “Judging llm-as-a-judge with mt-bench and chatbot arena,” 2023.

[47] Y. Zhou, Z. Li, J. Zhang, J. Wang, Y. Wang, Z. Xie, K. Chen, and L. Shou, “Floe: On-the-fly moe inference on memory-constrained gpu,” in Proceedings of the International Conference on Machine Learning (ICML 2025), 2025, arXiv preprint arXiv:2505.05950. [Online]. Available: https://arxiv.org/abs/2505.05950