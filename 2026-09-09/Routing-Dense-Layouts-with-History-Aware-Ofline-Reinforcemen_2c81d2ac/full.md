# Routing Dense Layouts with History-Aware Ofline Reinforcement Learning using LSTM

Afsara Khan

atk331@nyu.edu

New York University

Brooklyn, New York, USA

Austin Rovinski   
rovinski@nyu.edu   
New York University   
Brooklyn, New York, USA

## Abstract

Detailed routing remains a dominant runtime bottleneck in physical design due to increasing complexity of design rules. Modern routers can struggle to resolve persistent violations under dense operating conditions. While recent work leverages reinforcement learning (RL) to dynamically select costs for each routing iteration, we find that this technique struggles with high-density designs where routing solutions are significantly harder. To address this, we present a history-aware ofline RL policy which predicts iterative cost weights in these dense regimes to improve convergence across placement densities by utilizing readily available features from the router. Our policy uses conservative Q-learning similarly to prior work; however, our key insight is that addition of a lightweight LSTM architecture and additional features can retain sequence context and improve routing convergence across multiple densities and route guide qualities. Our policy can be integrated into any cost-based router with minimal pipeline changes, as it does not interfere with the core search algorithm. We evaluate our policy on held-out density and adjustment settings, including dificult operating points induced by dense placement and low guide quality. Our policy reduces design rule violations (DRVs) by an average of 92% over the top public baseline while simultaneously reducing runtime by 10%.

## CCS Concepts

• Hardware → Wire routing; • Computing methodologies → Partially-observable Markov decision processes.

## Keywords

Detailed routing, Reinforcement learning, Physical design, Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM)

## 1 Introduction

Detailed routing is often a very time-consuming step in the place & route flow. As technology nodes become more costly, achieving high logic density is critical to both reducing cost and improving performance. Modern routers rely on cost-driven iterative search and repair pathfinding algorithms, where the relative weighting of cost penalties can strongly influence the convergence behavior. Most public state-of-the-art work relies on fixed or manually designed cost schedules [1–3], where the algorithm continuously rips up and reroutes until it finds a set of costs that route cleanly, or it terminates with violations.

As placement density increases, so too does the routing congestion. This leaves fewer open tracks available for the detailed router to perform rerouting, which increases the likelihood of stalls and persistent violations. Fixed cost schedules and simple regressions struggle with high density placements, because the best routing solution varies dramatically based on the local congestion and the current mix of violations, both of which can change from one routing iteration to another. While tuning static costs can help improve results on a given design family or operating point, the cost map can be brittle and make other operating points worse.

In this paper, we study iteration-level control of existing routing cost weights in high utilization regimes, when the detailed router is left with very limited resources for rerouting. We demonstrate our implementation using OpenROAD as a public reference point; however, any iterative cost-based detailed router can adopt the same technique with router-specific feature extraction and lightweight software integration. The broader methodology of using a learned policy to control routing cost weights at each iteration using builtin router features can be extended to dense regimes where opensource state-of-the-art routers fail, directly resolving layouts that are dificult to route with existing routing flows.

Prior work [4] has shown that ofline conservative Q-learning (CQL) can act as an intelligent cost weight generator and accelerate convergence in a fixed operating regime with superior results over public baselines. However, varying density and operating conditions introduce additional challenges, since layout features can change substantially even with small density shifts, causing distribution shift between training and inference and pushing cost weight actions toward saturation. Alongside introducing an LSTM head for stronger sequential awareness, our method addresses these distributional issues through feature transformation, masking of density and adjustment inputs during training, and the use of ratios and logarithms for local signals whose absolute counts can vary widely. We demonstrate in Figure 1 that, while the CQL approach of prior work [4] degrades sharply as placement density rises, our model remains efective in these higher utilization regimes.

The results show consistent improvements over a wide range of densities. The largest gains appear when a test density lies between training points, which is consistent with the broader tendency for learned models to behave more reliably in interpolation than in extrapolation [5]. However, the interpolated densities remain nontrivial in our setting, since even intermediate density values can induce materially diferent layouts and routing behavior (discussed further in Section 5).

We train a conservative Q objective with a stochastic actor and add an LSTM head for the policy to better retain short routing history to link recent choices to their efects rather than react to a single snapshot. Each run is subject to strict limits: at most 65 routing iterations or 7 days of runtime. Runs that exceed either limit are not considered and we avoid extrapolating outcomes. Our work ofers the following contributions:

![](images/ff2dc1a2fa01ea70a48a8f796ea87a861601d0cc6437443196d1eb666c425353.jpg)  
Figure 1: Routing performance for aes vs. placement density. CQL denotes the approach from prior work [4] and CQL+LSTM is this work.

• Development of a detailed routing training dataset using perturbation sampling and random exploration techniques across various density and adjustment configurations, with emphasis on dense routing regimes

• Identification of critical routing state variables and reward components suitable for history-aware iterative control in detailed routing using reinforcement learning (RL)

• Implementation and evaluation of an ofline RL approach using conservative Q-learning with LSTM based soft actorcritic network to predict improved cost weights for detailed routing that outperform public baselines.

• Demonstration of improved routing convergence for di verse designs in high utilization regimes.

• The training and inference implementation are open sourced at https://github.com/realise-lab/RLDRT.

The remainder of the paper is organized as follows. Section 2 discusses related work, Section 3 discusses our methodology, Section 4 discusses our model architecture and training, Section 5 discusses the experimental results, and Section 6 provides our conclusions.

## 2 Related Work

## 2.1 Detailed Routers

The foundation of modern detailed routing traces back to Lee’s maze routing algorithm [6], a breadth-first search (BFS) that guaranteed minimum cost paths. Rapid notable improvements on this base algorithm included the A\* search using heuristics to guide the search [7, 8], along with techniques like line search [9] to speed up execution. Core strategies in modern routers include iterative rip-up and reroute [3] and sometimes leverage multicommodity flow concepts [10]. TritonRoute [3] utilizes the prior findings to employ an iterative A\*-based search with partitions, enhanced by dynamic boundary adjustments. Similarly, Dr. CU [1] uses Dijkstra’s algorithm coupled with a sparse grid-graph and partitioning to eficiently route nets. As newer technology nodes arrive, routing research has shifted focus towards techniques like gridless pin access [11], routing under complex patterning like SADP [12, 13], minimum area-sensitive path search [14], and ILP-based formulations [10].

In terms of academic routers, OpenROAD [15] provides stateof-the-art performance, achieving 0 DRVs on the ISPD ‘18 [16] benchmark and 0 DRVs on all but one test case on ISPD ‘19 [17]. OpenROAD’s router is derived from TritonRoute [2, 3], but several enhancements have been made and it is actively maintained to continue improving detailed routing results.

In the academic routing flows discussed above, the exposed cost weights that guide the iterative search are controlled by manually designed schedules rather than by the evolving routing state of the design.

## 2.2 Machine Learning Techniques in Routing

Several prior works have shown the benefits of applying traditional machine learning to global and detailed routing. Many works have focused on congestion and DRV hotspot prediction at the routing level [18, 19], while others have focused on predicting and avoiding pin access violations [20]. These predictions typically required secondary mechanisms to influence routing behavior. More direct guidance has been explored via reinforcement learning. For example, Chen et al. [21] propose an online RL framework that combines graph neural networks (GNNs) with PPO policy learning.

Prior work [4] demonstrates that ofline RL can learn efective cost weights to accelerate convergence in detailed routing. However, that work trains and evaluates at a single operating point. Density and adjustment changes shift the underlying feature distribution substantially, and a policy fit to a single density can no longer sustain convergence under that shift without deliberate feature and architectural engineering. Retaining the ofline RL formulation from prior work [4], we introduce robust feature construction and transformation, masking of density and adjustment signals during training, and an LSTM that conditions on recent routing history rather than a single state. These additions fundamentally change the nature of the problem: from accelerating convergence at one operating point to sustaining it across dense layouts, including hard cases that do not converge at all under existing flows.

While we demonstrate our formulation using OpenROAD, the technique can be adapted to any cost based router. To the best of our knowledge, no prior work has studied history aware ofline RL for iterative cost weights across varying routing regimes in general detailed routing.

## 3 Methodology

## 3.1 Routing and Policy

We study detailed routing across varying placement utilization, with special attention to the dificult, higher-density cases. Global routing adjustment controls how aggressively the global router is allowed to use the available routing tracks. At 0.0, the global router may use all tracks, which packs nets tightly and leaves the detailed router with very limited resources to resolve violations. Higher adjustment restricts a higher fraction of track capacity from the global router, leaving more usable space for the detailed router that follows [3]. Lowering the global-routing adjustment is not universally required as density increases; it becomes necessary only when, at a given density, the global router would otherwise return a congested guide. In those situations, reducing adjustment leaves the global router with more usable tracks to resolve congestion and return a routing guide for the detailed router. However, this leaves fewer reserved detailed routing resources, as mentioned above, increasing the likelihood of stalls and persistent violations.

The policy interfaces through the standard cost multipliers. At the end of iteration �, the router aggregates a global feature snapshot to calculate the feature vectors for the model. The agent then returns a shared set of cost multipliers for iteration �+1: drcCost, markerCost, fixedShapeCost, markerDecay. These multipliers control TritonRoute’s routing search by acting as penalty weights: drc-Cost and markerCost penalize routing through Design Rule Check (DRC) violating regions and near existing violation markers, fixed-ShapeCost penalizes overlap with fixed obstructions, and markerDecay sets how quickly old violation markers lose influence [3, 4]. Rather than fixing these multipliers to a schedule as seen in the baseline router, our policy resets them each iteration from the current routing state. We adopt global iteration control to keep integration lightweight and avoid additional coordination and overhead that per-partition control would require. At iteration 0, the router uses the default cost weights because no prior state history is yet available. The core search remains unchanged.

Iteration count and total runtime are related but not identical. Each iteration activates multiple partitions across workers. When violations are stubborn, each partition requires longer time to rip up and repair its local region, so the iteration takes longer; easier violations finish faster. We tested cumulative DRV as a runtime proxy by correlating violation traces with measured time across designs and operating points. We define the cumulative count in Equation 1 as

$$
S _ { k } \ = \ \sum _ { t = 0 } ^ { k } \mathrm { D R V } _ { t } ,\tag{1}
$$

where $\mathrm { D R V } _ { t }$ is the violation count at iteration �. The correlation was inconsistent: per-iteration DRV often showed only weak association with per-iteration runtime, and runs with similar total DRV sometimes had large runtime diferences because a small number of late-stage, stubborn violations dominate cost. In addition, the router resolves multiple partitions in parallel [3], so a larger number of violations does not necessarily increase runtime linearly.

## 3.2 Data Generation

As with most reinforcement learning methods, data generation is central to performance. We generate data from 8 Nangate45 designs in the OpenROAD Design Suite [22] and span 6 placement densities per design. The densities are evenly spaced and start at the default utilization for the design suite. Global-routing adjustment varies from 0.0 to 0.3. Lower adjustments yield valuable but slow traces. Higher adjustments yield many more traces and can run in parallel with others. Most data therefore come from 0.10 to 0.30, with a smaller portion at 0.00 where runs are slower but informative. In total, about 10,000 routing runs were collected across designs and densities. Harder routing cases in the dataset arise naturally from plausible density and adjustment sweeps within the same routing flow, where higher density increases routing pressure and lower adjustment is sometimes needed to obtain a routable guide.

A coarse grid over the die records a violation heatmap at each iteration. The grid also captures simple dynamics such as whether violations remain in place, migrate, or form clusters as routing proceeds. Figure 2 shows one region at iteration 0 and iteration 4. All signals come from the detailed router or from light transforms of those signals, so feature extraction overhead remains minimal relative to routing runtime, especially on complex designs.

![](images/3b682bc30d272b419f9d1c90e575589c53776ba3f811ea15c05d6c1b9a6fc140.jpg)  
(a) Iteration 0

![](images/38647eaa35709d7ed6b7bd80a510f4e3cc1a6ce0aa0287d4a245e16d0a1a9c88.jpg)  
(b) Iteration 4  
Figure 2: Example violation progress on aes in an 11-iteration routing run

Feature values are normalized to a common scale based on the training statistics. For features whose raw magnitudes can vary substantially across diferent routing settings, we prefer ratios or logarithmic transforms over raw counts. For example, current and initial DRV are log transformed, and progress is represented relative to the initial DRV from the first iteration. This practice provides a more stable signal across various densities and prevents the model from action saturation due to drastic distributional shifts in data. The action itselfis part ofthe state: we include the previous iteration weights and their change rate. We categorize features into two types: Dynamic Features:

• Current violation count (normalized)

• Initial violation count (normalized)

• Stagnant violation regions which are based on coarse-grid cells whose aggregate local violation count does not decrease across three consecutive iterations

• Change in the maximum local violation count and clustering spread for the current iteration, where clustering spread measures how unevenly violations are distributed across active coarse-grid cells

• Violation type ratios for common classes: short, metal spacing, cut spacing, end-of-line spacing

• Previous weights, i.e. drcCost, markerCost, fixedShapeCost, markerDecay

Static Design Features:

• Terminal count in log scale

• Die area in log scale

• Placement density

• Routing adjustment

• Interaction of density, adjustment, and terminal count (the goal is to signal that the same density can behave diferently by design, and that the hardest cases arise from combinations such as high density with 0.0 adjustment and high terminal count)

The underlying router parameters follow prior work [3], while the derived coarse grid and progress features are contributions introduced in this work.

After each iteration, we sample each of these values as part of the state �. A “sequence” � is formed by a series of states $\left\{ s _ { 1 } , s _ { 2 } , . . . , s _ { n } \right\}$ where � is the number of iterations and forms from one full routing run that terminates either at DRV convergence or at the iteration/runtime cap. Each sequence forms a data point which is used to train our RL model.

## 4 Model Architecture

We frame weight selection as a sequential decision process. At the end of iteration �, the routing engine emits a state vector $s _ { k }$ and the policy returns the cost multipliers for iteration �+1. The episode starts at iteration 0 and terminates when DRV reaches 0 or when iteration exceeds 65.

Formulation. We adopt ofline reinforcement learning with a conservative Q objective as described by [23]. The network is based on a stochastic actor in the soft actor-critic (SAC) family with LSTM heads, and we model state as a sequence. Both policy and value functions consume short temporal windows so the agent can condition on recent routing history. The window length is a tuned hyperparameter (Table 1). Within an episode, the hidden state propagates across iterations and resets only between episodes, so the efective context reaches beyond the window itself.

State representation. The state contains the signals described in Section 3, including the previous iteration’s weights, and is fed to sequence encoders before the multilayer heads. An episode starts at iteration 0 and ends when DRV reaches 0 or when iteration exceeds 65. This sequence-aware formulation is a key change relative to non-recurrent baselines.

Stability of standard Q-learning. We also trained a standard Q learning variant by setting the conservative weight to zero. Despite a broad hyperparameter search with Optuna over learning rates, discount, target update rate, batch size, entropy temperature, and gradient clipping, losses remained unstable. We observed sustained growth in the Bellman error, frequent gradient explosions, and pronounced overestimation on out-of-distribution actions drawn from the replay bufer. Double-Q critics with target networks slowed divergence but did not prevent it. The conservative Q objective provided the needed regularization and produced stable training curves on the same data.

Policy and critics. Both actor and critics are sequence aware. Each network uses 2 LSTM layers followed by a multilayer head. Hidden state propagates across iterations within an episode and resets at the start of the next episode. This captures short-range dependencies such as violation migration, hotspot persistence, and delayed efects of weight changes that may appear one or two iterations later.

Action space and safety. The action is a 4-dimensional vector that specifies the next iteration’s multipliers {drcCost, markerCost, fixedShapeCost, markerDecay}. Outputs pass through a bounded squashing function into action ranges matched to the training data. This mapping avoids pathological values and keeps the downstream search stable.

Learning objective and Reward Function. We validated the chosen reward function ofline on the collected dataset by applying to logged trajectories. The reward scalar clearly distinguished runs with the lowest iteration counts and the fastest average runtimes from the rest of the dataset. Motivated by the routing analysis in Section 3, we prioritize DRV reduction with an iteration penalty and hotspot term based on changes in the maximum local violation count, since violation dificulty rather than count often drives runtime. The reward aggregates the following:

• Progress: improvement rate relative to the initial DRV.

• Speed: a convergence bonus when DRV = 0 and an iteration penalty that grows with �, so earlier completion earns higher return.

• Dificulty-aware scaling: bonuses and penalties are scaled by a design or instance complexity factor built from log transformed terminal count and die area, together with a bounded density modulation, that may be expressed as:

$$
C _ { \mathrm { i n s t } } = \left( 1 + \alpha \log ( T + 1 ) + \beta \log ( A + 1 ) \right) g ( \rho ) ,\tag{2}
$$

where � is terminal count, � is die area, �ℎ� is placement density, and �(�) is a bounded modulation term. This scaling ensures that identical absolute gains receive larger credit on harder instances.

• Locality: an explicit hotspot term based on changes in the maximum local violation count on the coarse grid.

We intentionally do not include cumulative DRV in the training objective, and we do not use true runtime during training, since data generation runs in parallel across machines with variable load. Final runtime claims are based on isolated validation runs as stated in Section 3.

Training protocol. The replay bufer stores complete episodes and contiguous sequences so temporal order is preserved; within episode shufling is disabled.

We regularize density and adjustment interactions with feature masking (feature dropout) on the input layer so the policy does not overfit to a single operating point and learns to adjust to distribution shifts [24]. For selected features, we form a masked view

$$
\begin{array} { r } { \tilde { \textbf { x } } = \textbf { x } \odot \mathbf { m } , \quad m _ { i } \sim \mathrm { B e r n o u l l i } ( 1 - p _ { i } ) , } \end{array}\tag{3}
$$

where $p _ { i } \in \left[ 0 , 0 . 5 \right]$ is the masking probability tuned during training and only a small subset of inputs is eligible for masking (placement density, adjustment signals, and their interaction). Masking is applied to training episodes only. Validation and test use full features. We search $\mathbf { \nabla } \mathcal { P } i$ with Optuna over [0, 0.5] per masked feature and select the configuration that yields the best isolated runtime and iteration metrics on the validation set.

Optimization uses Double-Q critics with target networks and a soft policy update. Inputs are normalized; ratios replace raw counts when ranges are wide; logs are used when growth is steep.

![](images/e432a731612765564abde7ec05f46d68eb4a514d7a0c673cf6736d5899114b02.jpg)  
Figure 3: Training and Inference Summarized

Table 1: CQL Hyperparameter Values
<table><tr><td rowspan=1 colspan=1>Hyperparameter</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Critic and LSTM DropoutTemporal Window (iters)</td><td rowspan=1 colspan=1>1.00 × 10−18</td></tr><tr><td rowspan=1 colspan=1>Critic Hidden Units</td><td rowspan=1 colspan=1>512, 512, 256</td></tr><tr><td rowspan=1 colspan=1>Actor Learning Rate</td><td rowspan=1 colspan=1> $\overline { { 1 . 0 0 \times 1 0 ^ { - 4 } } }$ </td></tr><tr><td rowspan=1 colspan=1>Critic Learning Rate</td><td rowspan=1 colspan=1> $\overline { { 3 . 0 0 \times 1 0 ^ { - 4 } } }$ </td></tr><tr><td rowspan=1 colspan=1>Conservative Weight</td><td rowspan=1 colspan=1> $2 . 4 4 \times 1 0 ^ { 0 }$ </td></tr><tr><td rowspan=1 colspan=1>Batch Size</td><td rowspan=1 colspan=1>512</td></tr><tr><td rowspan=1 colspan=1>Initial Temperature</td><td rowspan=1 colspan=1> $\overline { { 1 . 0 0 \times 1 0 ^ { 0 } } }$ </td></tr><tr><td rowspan=1 colspan=1>Temperature LR</td><td rowspan=1 colspan=1> $\overline { { 1 . 0 0 \times 1 0 ^ { - 4 } } }$ </td></tr><tr><td rowspan=1 colspan=1>Tau (Polyak τ)</td><td rowspan=1 colspan=1> $\overline { { 1 . 0 0 \times 1 0 ^ { - 2 } } }$ </td></tr></table>

Model Parameters and Hyperparameters. Our goal in tuning was to accelerate convergence in favorable weight regimes while avoiding overestimation, out-of-distribution drift, and gradient pathologies. Guided by findings in the original CQL [23] paper and d3rlpy [25] documentation, we ran 100 Optuna trials over standard ranges and selected the configuration in Table 1.

To prevent degradation or policy collapse, we employ three broad categories of early stopping checks evaluated every epoch, plus a simple sanity check on value magnitudes. If early stopping does not trigger, training proceeds to 20 epochs, based on prior trials indicating convergence by that point, to produce a full model for routing validation discussed below.

Loss explosion or divergence detection. We monitor critic loss, actor loss, conservative loss, and TD error for sustained growth that signals failure in the Q approximation or unstable bootstrapping. We allow transient overshoot early in training but stop when any tracked loss exceeds 100× its initial value or shows a persistent upward trend across 10 consecutive epochs. This prevents unreliable actor gradients once the critic destabilizes.

Action diference monitoring. We track the d3rlpy action\_diff metric, with actions normalized to [0, 1] during training. Moderate deviation from dataset actions is expected, but large divergence indicates poor generalization. Values above 1.0 act as a warning threshold; we stop when the metric continues to increase across successive checkpoints.

![](images/738c3ba2590c115ce6d80d5c314bf1928e51f527eeb635e9518b496ea0336f67.jpg)  
Figure 4: Model Training Progress. 1 epoch = 10,000 timesteps

Figure 4 shows training metrics for the LSTM–CQL policy. Actor and critic losses drop sharply and then flatten near zero without overshoot. TD error decays rapidly and remains low. The conservative loss rises monotonically from a large negative value towards zero as the policy places more mass on actions supported by the dataset. The initial state value increases smoothly and plateaus; there is no overshoot.

Here, an epoch is a fixed budget of 10,000 policy/critic update steps sampled (with replacement) from the replay bufer, not a full pass over the dataset. All training was performed on CPU (320 cores @ 2.10 GHz) as per-epoch runtime was comparable to an H100, likely because the high core count removed the compute bottleneck for our data and sequence models.

Inference and integration. At runtime, the router calls the policy once per iteration boundary, provides the current state $s _ { k } ,$ and applies the returned multipliers in iteration �+1. Integration touches only these 4 knobs. The underlying search remains unchanged, which keeps the approach portable across designs, densities, and adjustment settings. Inference is integrated via TorchScript/LibTorch inside the C++ router.

## 5 Experimental Results

## 5.1 Methodology

We evaluate on OpenROAD design-suite circuits in the Nangate45 node so that placement utilization and routing adjustment can be varied. ISPD ‘18 and ISPD ‘19 benchmarks ship fixed LEF/DEF and do not permit density sweeps, so they are not used for the high utilization study here. We consider two test regimes:

(A) Converging cases. The baseline converges to zero DRV within the standard cap, and we test whether the policy converges faster and in fewer iterations across unseen operating points. To induce a distribution shift, densities at test time are interleaved between the training densities (e.g., midpoints rather than the exact values used for training), while routing adjustment is fixed at a moderately dificult setting. The density sweeps in Figure 7 therefore form the core evaluation for these converging cases. By holding adjustment fixed, we isolate placement density as the primary variable and can measure how routing behavior and policy performance change with it. We also include a visual comparison of the same region of a design at two densities in Figure 5 to show that even moderate density changes can meaningfully alter topology, clustering, and blockage patterns.

![](images/8286a613ff34cb3cd6b24175cc9180960d7a35a80c047c202f279b06f8562a89.jpg)  
(a) Density=0.42

![](images/952bd0e3fa7f4d5f007561f9244522cae86ff02e0d0d0840a4f548a2616ac7d2.jpg)  
(b) Density=0.65  
Figure 5: Example layout (aes) with varying density. Core area is the same in (a) and (b), but clustering is tighter in (b).

(B) Hard cases (baseline non-convergent). The baseline fails to reach zero DRV within 65 iterations. We report whether the policy (i) converges fully, or (ii) reduces terminal DRV versus the baseline when convergence is not reached under the same cap. These are deliberately harder operating points than those seen in training due to runtime constraints in data generation. In practice, many such failures occur at the lowest adjustment (e.g., 0.0) for higher densities; some designs exhibit failure at higher adjustments as well.

## 5.2 Reporting format

For Case (A), we present a density sweep as a plot of iterations versus placement density with two curves (baseline and RL agent). Runtime is summarized separately using a paired scatter of baseline versus policy per density on the same axis. Following the reporting used in prior work [4], Table 2 reports metrics for a single density per design to enable direct comparison between our model and prior work.

For Case (B), we present Table 3 with the terminal DRV count at the iteration cap; zero DRVs mean that convergence was achieved within the iteration cap. We also report runtime and wirelength. All inference overheads are included in policy runtimes.

![](images/47d3f0bc5bae05e58dd1337bdc44781b2a8dc73fb1430e92f663d34225ede3b0.jpg)  
Figure 6: DRV and Runtime in aes

Our benchmarks were run on an AMD EPYC 9275F CPU @ 4.1 GHz with 768 GB of DDR5 RAM and 48 threads. The runtime for each benchmark was averaged over 10 runs for each configuration (default and RL-guided). Runtimes for our approach include policy inference overhead, which was measured to be about 3s total per run due to the LSTM overhead.

## 5.3 Benchmark Performance

Our experimental results are summarized in Tables 2 and 3. We begin with the converging cases. Table 2 reports a single density per design across 1) the baseline OpenROAD router 2) reported values from prior work [4] 3) our model. All three are compared on identical designs at the same operating point. Figure 7 then extends this to full density sweeps. For most designs, the iteration and runtime improvements from our model persist across held-out densities. While each design in this set is included in the training data, the specific density and adjustment configurations tested here were not present during training. Visual inspection of the generated layouts (Figure 5) confirms diferences in placement topology between densities. Therefore, the results demonstrate the model’s ability to generalize across unseen routing settings which can induce dramatically diferent layout topologies on the same design rather than simply memorizing training examples.

We then present Table 3 to show results across 1) the baseline OpenROAD detailed router and 2) our RL model for the harder nonconverging cases. We omit prior work [4] here because it reports a single placement density and does not perform sweeps. Sweeping density or adjustment drastically shifts the distribution of the raw features a learned policy consumes, and a policy trained at one density on those raw features saturates under the shift. This out-of-distribution behavior is well documented by several works. Dakhmouche and Gorji [5] examine why machine learning models fail to extrapolate beyond their training distribution. Figure 1 further corroborates our claim that the prior approach [4] degrades sharply as density rises while our model with recurrent memory remains stable. The baseline, by contrast, is an expert curated cost schedule, not a learned policy, so it does not sufer such saturation. Using machine learning theory and Figure 1, we can accurately predict where the prior approach [4] degrades, but the same cannot be approximated for a human written baseline. Thus, we compare our performance with the static baseline that the prior approach [4] itself builds on.

Table 3 also reports substantially higher runtimes than Table 2 since these cases route under minimal guide quality (adjustment=0.0), where the detailed router has fewer reserved resources and must rip-up and reroute extensively, as discussed in Section 3. Figure 6 shows this runtime explosion when entering the hard regime in aes. The only exceptions we found were ibex and gcd, which maintained reasonable runtime under extreme conditions due to their simpler routing topologies.

![](images/34e18d68ed5c08e5c54b50e3aedfdf3b65449d56f1571575d2f8d3e5c59b3257.jpg)  
Figure 7: Density vs. routing iterations and runtime on dense benchmark designs. All designs finish with 0 DRVs.

Table 2: Performance on OpenROAD Design Suite (default density. Values from the prior work [4] are taken from their published results; our model is evaluated on the same test set.)
<table><tr><td rowspan=1 colspan=1>Design</td><td rowspan=1 colspan=1>Density</td><td rowspan=1 colspan=3>Iterations</td><td rowspan=1 colspan=4>Runtime (s)</td><td rowspan=1 colspan=4>Wirelength (um)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>[4]</td><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>[4]</td><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>Diff</td><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>[4]</td><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>Diff</td></tr><tr><td rowspan=1 colspan=1>aes</td><td rowspan=1 colspan=1>0.53</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=1>46</td><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>-8.70%</td><td rowspan=1 colspan=1>309750</td><td rowspan=1 colspan=1>312744</td><td rowspan=1 colspan=1>304773</td><td rowspan=1 colspan=1>-2.55%</td></tr><tr><td rowspan=1 colspan=1>ariane136</td><td rowspan=1 colspan=1>0.30</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>147</td><td rowspan=1 colspan=1>143</td><td rowspan=1 colspan=1>130</td><td rowspan=1 colspan=1>-9.09%</td><td rowspan=1 colspan=1>8017226</td><td rowspan=1 colspan=1>8038308</td><td rowspan=1 colspan=1>8038338</td><td rowspan=1 colspan=1>0.00%</td></tr><tr><td rowspan=1 colspan=1>bp_be</td><td rowspan=1 colspan=1>0.36</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>115</td><td rowspan=1 colspan=1>56</td><td rowspan=1 colspan=1>48</td><td rowspan=1 colspan=1>-14.29%</td><td rowspan=1 colspan=1>3028121</td><td rowspan=1 colspan=1>3037392</td><td rowspan=1 colspan=1>3014477</td><td rowspan=1 colspan=1>-0.75%</td></tr><tr><td rowspan=1 colspan=1>bp_fe</td><td rowspan=1 colspan=1>0.31</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1>43</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>-16.28%</td><td rowspan=1 colspan=1>2352172</td><td rowspan=1 colspan=1>2359893</td><td rowspan=1 colspan=1>2266409</td><td rowspan=1 colspan=1>-3.96%</td></tr><tr><td rowspan=1 colspan=1>bp_multi</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>132</td><td rowspan=1 colspan=1>110</td><td rowspan=1 colspan=1>109</td><td rowspan=1 colspan=1>-0.91%</td><td rowspan=1 colspan=1>4711286</td><td rowspan=1 colspan=1>4729301</td><td rowspan=1 colspan=1>4729973</td><td rowspan=1 colspan=1>0.01%</td></tr><tr><td rowspan=1 colspan=1>gcd</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>-25.00%</td><td rowspan=1 colspan=1>4786</td><td rowspan=1 colspan=1>4827</td><td rowspan=1 colspan=1>4791</td><td rowspan=1 colspan=1>-0.75%</td></tr><tr><td rowspan=1 colspan=1>ibex</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>-21.43%</td><td rowspan=1 colspan=1>330625</td><td rowspan=1 colspan=1>332567</td><td rowspan=1 colspan=1>332558</td><td rowspan=1 colspan=1>0.00%</td></tr><tr><td rowspan=1 colspan=1>jpeg</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>-18.92%</td><td rowspan=1 colspan=1>1163854</td><td rowspan=1 colspan=1>1169491</td><td rowspan=1 colspan=1>1164986</td><td rowspan=1 colspan=1>-0.39%</td></tr><tr><td rowspan=1 colspan=1>Total</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=1>43</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>598</td><td rowspan=1 colspan=1>467</td><td rowspan=1 colspan=1>420</td><td rowspan=1 colspan=1>-10.06%</td><td rowspan=1 colspan=1>19917820</td><td rowspan=1 colspan=1>19984523</td><td rowspan=1 colspan=1>19856305</td><td rowspan=1 colspan=1>-0.64%</td></tr></table>

Table 3: Performance on OpenROAD Design Suite (augmented density). Prior work [4] omitted; see Section 5.3.
<table><tr><td rowspan=1 colspan=1>Design</td><td rowspan=1 colspan=1>Density</td><td rowspan=1 colspan=3>DRVs</td><td rowspan=1 colspan=3>Runtime (s)</td><td rowspan=1 colspan=3>Wirelength (um)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>Diff</td><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>Diff</td><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>Diff</td></tr><tr><td rowspan=1 colspan=1>aes</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>5342</td><td rowspan=1 colspan=1>486</td><td rowspan=1 colspan=1>-90.90%</td><td rowspan=1 colspan=1>362000</td><td rowspan=1 colspan=1>349040</td><td rowspan=1 colspan=1>-3.58%</td><td rowspan=1 colspan=1>371455</td><td rowspan=1 colspan=1>375030</td><td rowspan=1 colspan=1>0.96%</td></tr><tr><td rowspan=1 colspan=1>ariane136</td><td rowspan=1 colspan=1>0.52</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>464</td><td rowspan=1 colspan=1>232</td><td rowspan=1 colspan=1>-50.00%</td><td rowspan=1 colspan=1>6628779</td><td rowspan=1 colspan=1>6669617</td><td rowspan=1 colspan=1>0.62%</td></tr><tr><td rowspan=1 colspan=1>bp_be</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>1856</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>-98.01%</td><td rowspan=1 colspan=1>869618</td><td rowspan=1 colspan=1>800179</td><td rowspan=1 colspan=1>-7.99%</td><td rowspan=1 colspan=1>2698034</td><td rowspan=1 colspan=1>2627586</td><td rowspan=1 colspan=1>-2.61%</td></tr><tr><td rowspan=1 colspan=1>bp_fe</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>6882</td><td rowspan=1 colspan=1>542</td><td rowspan=1 colspan=1>-92.12%</td><td rowspan=1 colspan=1>402020</td><td rowspan=1 colspan=1>285869</td><td rowspan=1 colspan=1>-28.89%</td><td rowspan=1 colspan=1>1871959</td><td rowspan=1 colspan=1>1905874</td><td rowspan=1 colspan=1>1.81%</td></tr><tr><td rowspan=1 colspan=1>bp_multi</td><td rowspan=1 colspan=1>0.52</td><td rowspan=1 colspan=1>1086</td><td rowspan=1 colspan=1>89</td><td rowspan=1 colspan=1>-91.80%</td><td rowspan=1 colspan=1>399180</td><td rowspan=1 colspan=1>379243</td><td rowspan=1 colspan=1>-4.99%</td><td rowspan=1 colspan=1>3848245</td><td rowspan=1 colspan=1>3890018</td><td rowspan=1 colspan=1>1.09%</td></tr><tr><td rowspan=1 colspan=1>gcd</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>10.00%</td><td rowspan=1 colspan=1>5326</td><td rowspan=1 colspan=1>5372</td><td rowspan=1 colspan=1>0.86%</td></tr><tr><td rowspan=1 colspan=1>ibex</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>166</td><td rowspan=1 colspan=1>146</td><td rowspan=1 colspan=1>-12.05%</td><td rowspan=1 colspan=1>333569</td><td rowspan=1 colspan=1>330897</td><td rowspan=1 colspan=1>-0.80%</td></tr><tr><td rowspan=1 colspan=1>jpeg</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>601</td><td rowspan=1 colspan=1>61</td><td rowspan=1 colspan=1>-89.85%</td><td rowspan=1 colspan=1>126000</td><td rowspan=1 colspan=1>119700</td><td rowspan=1 colspan=1>-5.00%</td><td rowspan=1 colspan=1>1046000</td><td rowspan=1 colspan=1>1052050</td><td rowspan=1 colspan=1>0.58%</td></tr><tr><td rowspan=1 colspan=1>Total</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>15767</td><td rowspan=1 colspan=1>1215</td><td rowspan=1 colspan=1>-92.29%</td><td rowspan=1 colspan=1>2159458</td><td rowspan=1 colspan=1>1934420</td><td rowspan=1 colspan=1>-10.42%</td><td rowspan=1 colspan=1>16803367</td><td rowspan=1 colspan=1>16856444</td><td rowspan=1 colspan=1>0.32%</td></tr></table>

Across both regimes, our model consistently reduces iterations and runtime in the converging cases and significantly reduces DRVs in the non-converging cases. We see a 92% decrease in DRVs from the baseline alongside a 10% runtime improvement. Three cases in Table 3 stand out. ibex and gcd were evaluated at the maximum possible density with the hardest adjustment settings but still converged. Notably, our model still achieves improvements over the baseline. For ariane136, we were unable to identify a nonconverging operating point within reasonable runtime constraints and therefore report results at the highest utilization settings we explored.

For gcd and ibex, the RL inference overhead dominates over the speedup from RL and thus total changes remain modest. However, in harder cases (e.g., higher cell count or extreme routing congestion) that motivate this work, the inference cost is well under 0.01% of total routing time and is negligible against the time this policy saves.

Overall wirelength impact remains small, with a 0.32% total increase across the augmented density set. Any increases in wirelength over 1% are balanced by reducing DRV count over the baseline by >90%. We did not include wirelength in the reward function and therefore it was not an optimization target when selecting weights.

## 6 Conclusion

This paper presented a history aware RL agent for iterative cost control under varying routing conditions for improved detailed routing convergence and quality. We introduce a novel ofline RL model with LSTM to implement our methodology that can be extended to any router with iterative search. Our model integrates design features and DRV history to infer the router’s cost weights, making it portable and adaptable to any router which uses iterative cost-based algorithms. We evaluated 8 designs with augmented densities from the OpenROAD Design Suite, demonstrating an average 92% DRV reduction while achieving an average 10% runtime improvement over the baseline. Compared to the prior CQL-only approach [4], our LSTM-enhanced architecture better captures routing dynamics and improves convergence across densities.

## Acknowledgments

We thank Saik Anam Siam for helpful guidance on model saturation from distribution shift in machine learning. This work was partially funded by a Google charitable gift.

## References

[1] G. Chen, C.-W. Pui, H. Li, J. Chen, B. Jiang, and E. F. Y. Young, “Detailed Routing by Sparse Grid Graph and Minimum-Area-Captured Path Search,” in Proc. ASP-DAC, ser. ASPDAC ’19. New York, NY, USA: Association for Computing Machinery, 2019, p. 754–760.

[2] A. B. Kahng, L. Wang, and B. Xu, “TritonRoute: An Initial Detailed Router for Advanced VLSI Technologies,” in Proc. ICCAD. IEEE, 2018, pp. 1–8.

[3] ——, “TritonRoute: The Open-Source Detailed Router,” IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol. 40, no. 3, pp. 547–559, 2020.

[4] A. Khan and A. Rovinski, “Accelerating Detailed Routing Convergence through Ofline Reinforcement Learning,” in Proc. DATE, 2026, pp. 1–7.

[5] R. Dakhmouche and H. Gorji, “Why Cannot Neural Networks Master Extrapolation? Insights from Physical Laws,” arXiv preprint https://arxiv.org/abs/2510. 04102, 2025.

[6] C. Y. Lee, “An Algorithm for Path Connections and Its Applications,” IRE Transactions on Electronic Computers, vol. EC-10, no. 3, pp. 346–365, 1961.

[7] M. H. Arnold and W. S. Scott, “An Interactive Maze Router with Hints,” in Proc. DAC, ser. DAC ’88. Washington, DC, USA: IEEE Computer Society Press, 1988, p. 672–676.

[8] H. Kaindl and G. Kainz, “Bidirectional Heuristic Search Reconsidered,” J. Artif. Int. Res., vol. 7, no. 1, p. 283–317, Dec. 1997.

[9] D. W. Hightower, “A Solution to Line-Routing Problems on the Continuous Plane,” in Proc. DAC, ser. DAC ’69. New York, NY, USA: Association for Computing Machinery, 1969, p. 1–24.

[10] K. Han, A. B. Kahng, and H. Lee, “Evaluation of BEOL Design Rule Impacts Using an Optimal ILP-Based Detailed Router,” in Proc. DAC, ser. DAC ’15. New York, NY, USA: Association for Computing Machinery, 2015.

[11] T. Nieberg, “Gridless Pin Access in Detailed Routing,” in Proc. DAC, ser. DAC ’11. New York, NY, USA: Association for Computing Machinery, 2011, p. 170–175.

[12] Y. Ding, C. Chu, and W.-K. Mak, “Self-Aligned Double Patterning Lithography Aware Detailed Routing with Color Preassignment,” IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol. 36, no. 8, pp. 1381–1394, 2017.

[13] I.-J. Liu, S.-Y. Fang, and Y.-W. Chang, “Overlay-Aware Detailed Routing for Self-Aligned Double Patterning Lithography Using the Cut Process,” in Proc. DAC, 2014, pp. 1–6.

[14] M. Ahrens, M. Gester, N. Klewinghaus, D. Muller, S. Peyer, C. Schulte, and G. Tellez, “Detailed Routing Algorithms for Advanced Technology Nodes,” IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol. 34, no. 4, pp. 563–576, Apr. 2015.

[15] T. Ajayi, D. Blaauw, T.-B. Chan, C.-K. Cheng, V. A. Chhabria, D. K. Choo, M. Coltella, S. Dobre, R. G. Dreslinski, M. Fogaça, S. Hashemi, A. Hosny, A. B. Kahng, M. Kim, J. Li, Z. Liang, U. Mallappa, P. Penzes, G. Pradipta, S. Reda, A. Rovinski, K. Samadi, S. S. Sapatnekar, L. Saul, C. Sechen, V. Srinivas, W. Swartz, D. Sylvester, D. Urquhart, L. Wang, M. Woo, and B. Xu, “OpenROAD: Toward a Self-Driving, Open-Source Digital Layout Implementation Tool Chain,” in Proceedings ofGovernment Microcircuit Applications and Critical Technology Conference, ser. GOMACTech ’19, 2019.

[16] S. Mantik, G. Posser, W.-K. Chow, Y. Ding, and W.-H. Liu, “ISPD 2018 Initial Detailed Routing Contest and Benchmarks,” in Proc. ISPD, ser. ISPD ’18. New York, NY, USA: Association for Computing Machinery, 2018, p. 140–143.

[17] W.-H. Liu, S. Mantik, W.-K. Chow, Y. Ding, A. Farshidi, and G. Posser, “ISPD 2019 Initial Detailed Routing Contest and Benchmark with Advanced Routing Rules,” in Proc. ISPD, ser. ISPD ’19. New York, NY, USA: Association for Computing Machinery, 2019, p. 147–151.

[18] W. Zeng, A. Davoodi, and R. O. Topaloglu, “Explainable DRC Hotspot Prediction with Random Forest and SHAP Tree Explainer,” in Proc. DATE, 2020, pp. 1151– 1156.

[19] H. Park, K. Baek, S. Kim, K. Choi, and T. Kim, “Pin Accessibility and Routing Congestion Aware DRC Hotspot Prediction for Designs in Advanced Technology Nodes with Consolidated Practical Applicability and Sustainability,” IEEE Transactions on Computer-Aided Design ofIntegrated Circuits and Systems, vol. 43, no. 12, pp. 4786–4799, 2024.

[20] R. Liang, H. Xiang, D. Pandey, L. Reddy, S. Ramji, G.-J. Nam, and J. Hu, “DRC Hotspot Prediction at Sub-10nm Process Nodes Using Customized Convolutional Network,” in Proc. ISPD. New York, NY, USA: Association for Computing Machinery, 2020, p. 135–142.

[21] H. Chen, K.-C. Hsu, W. J. Turner, P.-H. Wei, K. Zhu, D. Z. Pan, and H. Ren, “Reinforcement Learning Guided Detailed Routing for Custom Circuits,” in Proc. ISPD, ser. ISPD ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 26–34.

[22] A. Rovinski, T. Ajayi, M. Kim, G. Wang, and M. Saligane, “Bridging Academic Open-Source EDA to Real-World Usability,” in Proc. ICCAD, 2020, pp. 1–7.

[23] A. Kumar, A. Zhou, G. Tucker, and S. Levine, “Conservative Q-Learning for Ofline Reinforcement Learning,” in Proc. NeurIPS, vol. 33. Curran Associates, Inc., 2020, pp. 1179–1191.

[24] M. A. Haque, M. M. Kamol, I. Hossain, S. K. Amalapuram, V. Kreinovich, and M. S. Rahman, “CITADEL: A Semi-Supervised Active Learning Framework for Malware Detection under Continuous Distribution Drift,” arXiv preprint https: //arxiv.org/abs/2511.11979, 2025.

[25] T. Seno and M. Imai, “d3rlpy: An Ofline Deep Reinforcement Learning Library,” Journal ofMachine Learning Research, vol. 23, 2022.