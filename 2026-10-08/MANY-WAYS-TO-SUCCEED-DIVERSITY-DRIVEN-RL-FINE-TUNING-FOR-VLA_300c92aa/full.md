# MANY WAYS TO SUCCEED: DIVERSITY-DRIVEN RL FINE-TUNING FOR VLA GENERALIZATION

Haoru Li<sup>1∗</sup> Jinmei Liu<sup>1∗</sup> Zhiyong Wang<sup>2</sup> Xiaoming Li<sup>1</sup> Zhenhong Sun<sup>3</sup> Daoyi Dong<sup>4</sup> Chunlin Chen<sup>1</sup> Zhi Wang<sup>1</sup>

<sup>1</sup> Nanjing University <sup>2</sup> Harbin Institute of Technology (Shenzhen) <sup>3</sup> Australian National University <sup>4</sup> University of Technology Sydney {haoruli,jmliu}@smail.nju.edu.cn zhiwang@nju.edu.cn

## ABSTRACT

Reinforcement learning (RL) fine-tuning improves vision-language-action (VLA) policies through closed-loop experience, yet generalization beyond the fine-tuning distribution remains limited. Our analysis reveals a selective reshaping of exploration: RL contracts behavior globally, yet diversifies successful trajectories, elicits success with fewer rollouts, and covers more of the latent task-valid solution space than supervised fine-tuning. Broader successful-mode coverage may provide alternative strategies under distribution shifts. Inspired by this, we introduce DRIVE (Diversity-driven RL fIne-tuning for VLA gEneralization), which turns successful-behavior diversity into an explicit RL objective. DRIVE groups rollouts under matched task conditions, compares their trajectories with temporal alignment, and derives a success-conditioned intrinsic reward from relative behavioral diversity. This design encourages broader coverage of feasible solutions without rewarding diverse failures or superficial timing differences. Across LIBERO-Plus, ManiSkill3, and RoboTwin 2.0, DRIVE improves the average outof-domain (OOD) performance over vanilla RL fine-tuning by 5.3 points on π<sub>0</sub> and 2.0 points on π<sub>0.5</sub>. On a dual-arm AgileX PiPER-X platform, DRIVE further increases average OOD success from 64.1% to 73.3% (+9.2 points), demonstrating gains that persist under physical deployment.

## 1 INTRODUCTION

Recent advances in vision-language-action (VLA) models have substantially expanded the range of manipulation tasks that robots can perform from visual observations and language instructions (Kim et al., 2024). VLA models are trained primarily through supervised fine-tuning (SFT) on robot demonstrations, acquiring broad manipulation skills by imitating a fixed behavior distribution (Intelligence et al., 2025b). Reinforcement fine-tuning (RFT) complements this paradigm with self-generated experience, using outcome feedback to explore beyond demonstrations and optimize closed-loop task performance (Intelligence et al., 2025a). Recent studies have shown that RL fine-tuning, particularly with PPO, can improve VLA generalization in semantic understanding and execution robustness (Liu et al., 2025a). However, such gains do not transfer uniformly across distribution shifts, and VLA policies remain sensitive to changes in visual appearance, scene configuration, and execution conditions (Fei et al., 2025). Robust VLA generalization is essential for real-world deployment, yet remains challenging beyond the fine-tuning distribution. To understand how RFT reshapes the range of behaviors available to VLA policies, we analyze both how rollout distributions evolve and how readily their successful modes can be elicited (see Sec. 3). As training progresses, similarity increases across all rollouts but decreases among successful ones, revealing global behavioral contraction alongside successful-mode diversification. Pass@k measures how readily any successful behavior can be elicited, showing that RFT requires far fewe rollouts than SFT to obtain a success. Coverage@k instead measures the breadth of these successes, showing that, under the same rollout budget, RFT covers a substantially broader portion of the latent task-valid solution space than SFT. Together, these results suggest that RFT makes task-valid solutions more accessible even as the overall behavior distribution contracts. Broader coverage of this space may provide alternative strategies under distribution shifts, revealing an underexplored opportunity to improve generalization through successful-behavior diversity.

![](images/5ccf91e05a1d0c1432edae8475322cf756bc2f490fb678923dd232eb1fd8de41.jpg)  
Figure 1: Beyond task success, diverse successful trajectories reveal broader solution modes. This diversity provides a new opportunity for improving VLA generalization.

Turning this opportunity into a useful training objective is nontrivial. Standard binary task rewards distinguish success from failure but not among different successful solutions, providing no explicit incentive to maintain broad successful-mode coverage.

Although diversity has been studied in unsupervised skill discovery (Eysenbach et al., 2019), policy diversification (Masood & Doshi-Velez, 2019), and embodied exploration (Chen et al., 2026), directly applying these approaches to VLA fine-tuning can reward superficial differences caused by execution speed, temporal misalignment, or irrelevant motion, as well as noisy and unsuccessful behaviors. A suitable diversity objective must therefore compare trajectories under matched task conditions, account for temporal variation, and reward behavioral differences only when the trajectories remain task-valid.

Motivated by these observations, we propose DRIVE (Diversity-driven RL fIne-tuning for VLA gEneralization), a framework that turns successful-mode diversity into an explicit optimization target for promoting generalization. DRIVE groups rollouts collected under matched task conditions, quantifies their behavioral novelty using temporally aligned GAK (global alignment kernel) similarity, and assigns a relative novelty bonus only to successful trajectories. This design focuses the diversity incentive on distinct task-valid solutions rather than arbitrary variation across tasks, timing, or failures. Then, we formulate the intra-group diversity as an intrinsic reward and design a potential based reward shaping scheme to preserve optimal policy invariance. By encouraging broader coverage of successful behaviors during fine-tuning, DRIVE provides alternative task-solving strategies that may transfer under distribution shifts. Across LIBERO-Plus, ManiSkill3, and RoboTwin 2.0, DRIVE improves 11 of 12 benchmark-split results on each backbone, raising the OOD macroaverage from 48.1% to 53.4% for $\pi _ { 0 }$ and from 71.1% to 73.1% for $\pi _ { 0 . 5 }$ . On a dual-arm AgileX PiPER-X platform, DRIVE further increases average OOD success across two tasks and three distribution shifts from 64.1% to 73.3% (+9.2 points). Together, these results validate successful-mode diversity as a practical objective across simulated distribution shifts and physical deployment.

In summary, our main contributions are threefold:

• We show that global behavioral contraction coexists with successful-mode diversification, revealing successful-behavior diversity as an underexplored route to VLA generalization.

• We introduce DRIVE, which converts temporally aligned comparisons of grouped trajectories into a success-conditioned intrinsic reward for diverse task-valid exploration.

• We demonstrate consistent generalization gains across LIBERO-Plus, ManiSkill, and RoboTwin, and confirm that DRIVE’s improvements persist under physical deployment.

## 2 RELATED WORK

VLA Models and Generalization. VLA models map visual observations and language instructions to robot actions, typically combining pretrained vision-language representations with supervised learning from robot demonstrations (Kim et al., 2024; Black et al., 2025; Xie et al., 2026). However, systematic evaluations reveal sensitivity to changes in visual appearance, scene configurations, robot initial states, and execution conditions (Fei et al., 2025; Liu et al., 2025a). To improve generalization, prior work has explored data augmentation and domain randomization (Chen et al., 2025b; Xue et al., 2025), large-scale pretraining and heterogeneous co-training (Brohan et al., 2023; Intelligence et al., 2025b), and model designs that strengthen spatial representations and embodied reasoning (Qu et al., 2025; Zawalski et al., 2024). These efforts primarily focus on training data and model capabilities, with less attention to how behavior distributions shaped during RFT affect generalization.

![](images/6eace959098d35bd4fa8caa93e878056f6dbd3994a8bc760e81e98685356b3d0.jpg)

![](images/8f676f2ef2e8c8e213931a35e1268c0e7bbdc4259c941fe5118a165aad0f1553.jpg)

(c) Pass@K  
![](images/7ffd1d4ae99b2605e62165f0647a6675729e491abfe9724bf27ed8f368bde9f8.jpg)

(d) Coverage@K  
![](images/5e5c9207b50f855e3e80b374046e4f832493eb851207bba12da438b1c12bc57a.jpg)  
Figure 2: Exploration dynamics and successful-behavior accessibility in VLA reinforcement finetuning. (a–b) During RFT, similarity increases across all rollouts while decreasing among successful trajectories; SR denotes success rate. (c) Pass@K shows that the final RFT policy elicits success with fewer rollouts than SFT. (d) Coverage@ $K .$ , measured against SFT successful trajectories at $K = 2 5 6$ , shows broader reference-repertoire coverage under limited sampling.

Reinforcement Fine-Tuning for VLA. RFT complements demonstration-based SFT with selfgenerated experience and task-level feedback from closed-loop interaction, with successful deployments in LLMs (Guo et al., 2025; Yan et al., 2025; Zhan et al., 2026), VLMs (Liu et al., 2025b; Hu et al., 2026a), and VLAs (Intelligence et al., 2025a; Liu et al., 2025a). Recent PPO-based (Lu et al., 2025; Tan et al., 2025) and GRPO-based methods (Li et al., 2026) use this feedback to optimize task-completion or progress rewards. Empirical studies report that such fine-tuning improves task success over SFT and can enhance OOD generalization, particularly in semantic understanding and execution robustness (Liu et al., 2025a; Chen et al., 2025a). However, this line of work primarily examines the performance benefits of task-reward optimization, with less attention to how successful-behavior distributions evolve during RFT or how broader coverage of the successful be havior space can be explicitly promoted. Behavioral Diversity and Exploration. Unsupervised skill discovery promotes diverse behaviors without task rewards (Eysenbach et al., 2019; Sharma et al., 2020), while intrinsic-motivation methods encourage exploration through prediction-based novelty rewards (Pathak et al., 2017; Burda et al., 2019).

Task-aware methods instead combine return maximization with differences between policy-induced trajectory distributions (Masood & Doshi-Velez, 2019) or encourage diversity among near-optimal behaviors (Kumar et al., 2020; Hu et al., 2026b; Liu et al., 2026). In VLA learning, diversity has been used to guide exploration through trajectory expansion and behavior conditioning (Chen et al., 2026; Kong et al., 2026), and to enrich pretraining data with diverse successful trajectories (Yang et al., 2025). In contrast, DRIVE applies a success-conditioned diversity reward across rollouts under matched task conditions, targeting broader task-valid solution coverage rather than global behavioral variation.

## 3 EXPLORATION DYNAMICS IN VLA REINFORCEMENT FINE-TUNING

We study how RFT reshapes the diversity and accessibility of successful behaviors in $\pi _ { 0 }$ and $\pi _ { 0 . }$ .5 on ManiSkill3 (Tao et al., 2025), evaluating the SFT and intermediate/final RFT checkpoints. Using SDE sampling under matched task conditions, we measure trajectory similarity separately across all and successful rollouts. For the sampling-budget analysis, we evaluate SFT and final-RFT π<sub>0</sub> on 32 task-initial-state configurations, sampling up to 256 rollouts per policy and configuration.

Global Contraction Masks Successful-Mode Diversification. Fig. 2(a–b) reveals a contrasting trend: normalized Global Alignment Kernel (GAK) similarity increases across all rollouts but decreases among successful ones as RFT progresses. Meanwhile, success rates rise from 41% to 75% for $\pi _ { 0 }$ and from 52% to 77% for $\pi _ { 0 . 5 }$ . Thus, RFT concentrates the overall behavior distribution while improving task success and diversifying successful executions. Global contraction therefore masks successful-mode diversification rather than implying a collapse of task-valid exploration.

RFT Makes Latent Successful Behaviors More Accessible. We measure the accessibility of successful behavior using Pass@K, the probability that at least one of $K$ rollouts succeeds for a taskinitial-state configuration. Fig. 2(c) shows that RFT achieves substantially higher Pass@K at small $K ,$ while the gap narrows as SFT saturates with larger budgets. This pattern suggests that successful behavior is often latent in SFT but difficult to elicit. RFT makes such behavior accessible with far fewer rollouts, converting sparse success into reliable execution under limited sampling.

![](images/0a738ea32a30ba0cca4251d974c4a059c4074a493a39425601b376f9153c5b96.jpg)  
Figure 3: Overview of DRIVE. (a) Rollouts are encoded as per-decision feature trajectories. (b) GAK computes behavioral similarity by softly aligning feature sequences over monotonic temporal paths, handling execution-rate mismatch. (c) Pairwise similarities are converted into successconditioned diversity rewards and added to environment rewards to promote exploration diversity.

Elicitation Spans a Broad Task-Valid Solution Space. Pass@K does not distinguish between repeatedly sampling a few dominant modes and accessing a broader solution set. To measure breadth, we use SFT successes at K = 256 as a reference and compute Coverage@K through normalized GAK matching, with leave-one-out matching for SFT queries to avoid trivial self-matches and ensure a fair comparison. As shown in Fig. 2(d), RFT achieves substantially higher coverage at small K and matches SFT coverage with far fewer rollouts. Thus, RFT improves both the accessibility of success and the breadth of task-valid solutions available under limited sampling.

Taken together, RFT contracts behavior globally while expanding access to diverse task-valid solutions, revealing an underexplored opportunity: broader successful-mode coverage may provide alternative strategies under distribution shifts. Since standard rewards do not distinguish among successful solutions, the central question behind DRIVE is whether explicitly incentivizing their diversity can broaden task-valid exploration and improve VLA generalization.

## 4 MORE WAYS TO SUCCEED WITH DRIVE

In this section, we present DRIVE, a diversity-driven framework for VLA RFT. We first formulate the RFT objective and motivate behavior-aware rewards, then derive temporally aligned trajectory similarity from grouped rollouts, and finally convert relative successful-behavior diversity into a normalized intrinsic reward for success-conditioned shaping.

## 4.1 PROBLEM FORMULATION

We consider a language-conditioned embodied control problem modeled as a Markov decision process (MDP) $\mathcal { M } \overset {  } { = } ( \overset {  } { S } , \overset { \triangledown } { \mathcal { A } } , P _ { 0 } , P , R , \gamma )$ , where $P _ { 0 } , P , { \bar { R } } _ { }$ and $\gamma$ denote the initial-state distribution, environment dynamics, environment reward, and discount factor, respectively. Given a language instruction l, a VLA policy produces $a _ { t } \sim \pi _ { \theta } ( \cdot \ | \ s _ { t } , l ) \quad$ , where $s _ { t }$ includes visual and proprioceptive observations and $a _ { t }$ denotes either a single control action or an action chunk. A rollout is $\tau = ( s _ { 0 } , a _ { 0 } , \ldots , s _ { T - 1 } , a _ { T - 1 } , s _ { T } )$ , and $\boldsymbol { r } _ { t } ^ { \mathrm { e n v } }$ denotes the environment reward at step t.

In RFT of VLA models, policy-gradient methods update the policy using rewards from environment rollouts to favor higher-return behaviors (Li et al., 2026). We adopt PPO (Schulman et al., 2017), which constrains the update through a clipped importance ratio, with the surrogate objective

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { P P O } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { t } \left[ \operatorname* { m i n } \left( \rho _ { t } ( \boldsymbol { \theta } ) \hat { A } _ { t } , \operatorname { c l i p } ( \rho _ { t } ( \boldsymbol { \theta } ) , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { t } \right) \right] , } \end{array}\tag{1}
$$

where $\rho _ { t } ( \theta ) = \pi _ { \theta } ( a _ { t } \mid s _ { t } , l ) / \pi _ { \theta _ { \mathrm { o l d } } } ( a _ { t } \mid s _ { t } , l )$ is the importance ratio and $\hat { A } _ { t }$ is the estimated advantage, computed from environment rewards and value estimates. Because PPO advantages are derived from rewards, reward design determines which behaviors RFT reinforces. Standard task rewards capture accomplishment or progress but treat behaviorally distinct successes alike, motivating trajectory-level similarity as a basis for rewarding diverse task-valid solutions.

## 4.2 DERIVING TRAJECTORY-LEVEL BEHAVIORAL SIMILARITY

To distinguish different task-valid solution modes, we require a trajectory-level measure of behavioral similarity between rollouts. We compare rollouts only within matched groups that share the same task and environment condition, so that the resulting similarity reflects differences in execution rather than differences in task specification or initial conditions.

We represent each trajectory as an ordered sequence of per-decision embeddings. For rollout i with $L _ { i }$ executed action chunks, let $z _ { i , c }$ denote the feature representation of the observation at policy decision c, for $c = 0 , \ldots , L _ { i } - 1$ . In our implementation, we reuse the VLM-prefix representation already computed by the policy and mean-pool the valid prefix tokens into one embedding per policy decision. After the rollout terminates, we additionally extract the same representation from the terminal observation, yielding

$$
Z _ { i } = ( z _ { i , 0 } , \ldots , z _ { i , L _ { i } } ) ,\tag{2}
$$

where $z _ { i , L _ { i } }$ denotes the terminal-observation representation.

Behaviorally similar trajectories may progress at different rates, making step-wise comparison sensitive to temporal misalignment. We therefore use the Global Alignment Kernel (GAK) (Cuturi, 2011), which compares feature sequences while preserving temporal order and accommodating local timing differences. Let $G _ { i , j } ( t , u )$ denote the accumulated alignment similarity between the prefixes $z _ { i , 0 : t }$ and $z _ { j , 0 : u } . \mathrm { G A K }$ computes this quantity through the dynamic-programming recurrence

$$
G _ { i , j } ( t , u ) = \kappa ( z _ { i , t } , z _ { j , u } ) \left[ G _ { i , j } ( t - 1 , u ) + G _ { i , j } ( t , u - 1 ) + G _ { i , j } ( t - 1 , u - 1 ) \right] ,\tag{3}
$$

where κ denotes the local similarity between feature embeddings; in our implementation, we use a shifted cosine similarity mapped to [0, 1]. The three predecessor transitions allow either sequence, or both, to advance, accommodating local timing differences while preserving temporal order. By accumulating over these alternatives, GAK accounts for all admissible monotonic alignments path rather than selecting a single best alignment. After both trajectories have been fully traversed, the final dynamic-programming value gives their trajectory-level GAK kernel:

$$
K _ { \mathrm { G A K } } ( Z _ { i } , Z _ { j } ) = G _ { i , j } ( | Z _ { i } | , | Z _ { j } | ) .\tag{4}
$$

We normalize this kernel using the self-similarities of the two trajectories:

$$
\widetilde { K } _ { i j } = \frac { K _ { \mathrm { G A K } } ( Z _ { i } , Z _ { j } ) } { \sqrt { K _ { \mathrm { G A K } } ( Z _ { i } , Z _ { i } ) K _ { \mathrm { G A K } } ( Z _ { j } , Z _ { j } ) } } .\tag{5}
$$

We use $\widetilde { K } _ { i j }$ as the normalized trajectory-level behavioral similarity between trajectories i and $j ,$ where larger values indicate greater behavioral similarity.

## 4.3 SUCCESS-CONDITIONED DIVERSITY SHAPING

The trajectory similarities derived in Sec. 4.2 quantify behavioral differences but do not directly provide an optimization signal. We first convert them into a relative diversity score, then use this score to construct a success-conditioned potential and derive the DRIVE shaping reward.

Intra-Group Diversity. For a matched rollout group $^ { g , }$ let $\mathcal { T } _ { g } = \{ \tau _ { i } \} _ { i = 1 } ^ { N }$ contain N trajectories collected under the same task and environment condition. We define the average similarity $\mathrm { s i m } _ { i }$ and relative diversity $d _ { i }$ of trajectory $\tau _ { i }$ as

$$
\mathrm { s i m } _ { i } = \frac { 1 } { N - 1 } \sum _ { j \neq i } \widetilde { K } _ { i j } , \qquad d _ { i } = 1 - \mathrm { s i m } _ { i } .\tag{6}
$$

A larger $d _ { i }$ indicates that $\tau _ { i }$ is more behaviorally distinct from the other rollouts. We compare each trajectory against all group members because both successful and failed rollouts represent behaviors already explored by the policy.

Let $y _ { i } = \mathbf { 1 } [ \tau _ { i }$ succeeds] and $S _ { g } = \{ i \mid y _ { i } = 1 \}$ denote the success indicator and the successful set within the group g. To stabilize the reward scale, we normalize diversity within $S _ { g } \mathbf { \mathrm { . } }$

$$
\hat { d } _ { i } = \mathrm { c l i p } \left( \frac { d _ { i } - d _ { \operatorname* { m i n } } } { \operatorname* { m a x } ( d _ { \operatorname* { m a x } } - d _ { \operatorname* { m i n } } , \delta ) } , 0 , 1 \right) , \quad \mathrm { w h e r e } \ d _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { j \in S _ { g } } d _ { j } , d _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { j \in S _ { g } } d _ { j } ,\tag{7}
$$

and $\delta > 0$ prevents negligible differences from being amplified. For $i \not \in S _ { g }$ , we set $\hat { d } _ { i } = 0$

Success-Conditioned Diversity Potential. Let $s _ { i , c }$ denote the state containing the prefix up to action chunk c in trajectory $\tau _ { i } \in g$ . We define the diversity potential conditioned on task success as

$$
\Phi _ { g } ( s _ { i , c } ) = y _ { i } \cdot \hat { d } ( s _ { i , c } ) .\tag{8}
$$

Thus, behavioral diversity contributes potential only when the underlying trajectory successfully completes the task, preventing diverse failures from receiving an exploration bonus.

Potential-Based Reward Shaping. We define the intrinsic reward as the potential difference between adjacent states as

$$
\begin{array} { r } { r _ { i , c } ^ { \mathrm { d i v } } = \gamma \Phi _ { g } ( s _ { i , c + 1 } ) - \Phi _ { g } ( s _ { i , c } ) . } \end{array}\tag{9}
$$

Since the policy gradient operates at the trajectory level, the discounted return telescopes to

$$
r _ { \tau _ { i } } ^ { \mathrm { d i v } } = \sum _ { c = 0 } ^ { L _ { i } - 1 } \gamma ^ { c } r _ { i , c } ^ { \mathrm { d i v } } = \gamma ^ { L _ { i } } \Phi _ { g } ( s _ { i , L _ { i } } ) - \Phi _ { g } ( s _ { i , 0 } ) = \gamma ^ { L _ { i } } \cdot y _ { i } \cdot \hat { d } _ { i } ,\tag{10}
$$

where the diversity of an initial state is zero, i.e., $\Phi _ { g } ( s _ { i , 0 } ) = 0 \mathrm { ~ }$ . This formulation avoids the need to calculate the diversity of any intermediate prefixes within the trajectory. Finally, DRIVE implements trajectory-level diversity bonus as a success-conditioned, potential-based shaping reward as

$$
r _ { \tau _ { i } } ^ { \mathrm { D R I V E } } = r _ { \tau _ { i } } ^ { \mathrm { e n v } } + \alpha \cdot r _ { \tau _ { i } } ^ { \mathrm { d i v } } ,\tag{11}
$$

where α controls the strength of the diversity reward relative to the environment reward.

Theorem 1 (Optimal Policy Invariance). Let $M = ( S , A , P , R , \gamma )$ denote the MDPfor the VLA RFT task. $\hat { d } ( \cdot ) : S \mapsto \mathbb { R }$ is a real-valuedfunction that computes the trajectory-level diversity $\hat { d } ( s )$ of the state s within a group of rollouts. We formulate $r ^ { d i \bar { \nu } } ( \cdot ) : S \times A \times S \mapsto \mathbb { R }$ as an intrinsic reward function that is the difference between trajectory diversities of two adjacent states, such that for all $s \in S , a \in A , s ^ { \prime } \in S , r ^ { d i v } ( s , a , s ^ { \prime } ) = \gamma y \hat { d } ( s ^ { \prime } ) - y \hat { d } ( s ) $ , where $y \in \{ 0 , 1 \}$ is a binary success indicator. Then, with any constant balancing ratio α, every optimal policy in the transformed MDP $M ^ { \prime } { = } ( S , A , P , R + \alpha R ^ { \dot { d } i \nu } , \gamma )$ will also be an optimal policy in M, and vice versa.

This shows that the success-conditioned diversity reward reshapes the learning signal without changing the set of optimal policies. The proof is provided in Appendix A.

## 5 EXPERIMENTS

We evaluate DRIVE from three complementary perspectives: generalization across distribution shifts, behavioral dynamics during RFT, and transfer to real-world deployment.

Benchmarks. We first evaluate DRIVE on three manipulation benchmarks in simulation (see Sec. 5.3 for real-robot deployment): LIBERO-Plus, ManiSkill3, and RoboTwin 2.0. On LIBERO-Plus (Fei et al., 2025), we construct IND (in-domain) and OOD (out-of-domain) conditions for the four LIBERO suites: Spatial, Object, Goal, and Long (Liu et al., 2023). For each suite, the RFT/IND pool retains all ten base tasks and uses selected LIBERO-Plus perturbation values, while OOD evaluation uses held-out perturbation values over the same tasks. On ManiSkill3 (Tao et al., 2025), we follow the generalization setting of RL4VLA (Liu et al., 2025a), covering visual-language, semantic, and execution shifts. On RoboTwin 2.0 (Chen et al., 2025b), we adapt the perturbation taxonomy of RoboTwin 2.0-Plus (Zhang et al., 2026) to construct disjoint IND and OOD initialization sets. Detailed task selections and split definitions are provided in Appendix C.

Baselines and Training. We compare DRIVE with Vanilla RFT and three representative exploration or regularization baselines. Vanilla RFT follows the Flow-SDE-based RFT setup of $\pi _ { \mathrm { R L } }$ (Chen et al., 2025a) and uses only the environment reward. Higher Noise increases stochasticity during Flow-SDE sampling (Chen et al., 2025a; Zou et al., 2026). KL Regularization constrains the policy toward the SFT reference policy through a KL penalty (GX-Chen et al., 2026). Clip-Higher relaxes the upper PPO clipping bound following DAPO (Yu et al., 2025). These baselines intervene at the sampling, policy, and optimization levels, respectively, whereas DRIVE directly shapes successfulbehavior diversity. We evaluate both $\pi _ { 0 }$ (Black et al., 2025) and $\pi _ { 0 . 5 }$ (Intelligence et al., 2025b), with all methods initialized from the same SFT checkpoint and trained under a common PPO-based Flow-SDE RFT pipeline. We report IND and OOD success rates, with detailed training hyperparameters and implementation settings provided in Appendix D.

Table 1: Main results on simulation benchmarks. Values are task success rates (%). RoboTwin 2.0 averages over Click Bell and Press Stapler. Avg. is the macro-average over benchmark families. Best and second-best results are bold and underlined.
<table><tr><td rowspan="2">Method</td><td colspan="2">Spatial</td><td colspan="2">Object</td><td colspan="2">LIBERO-Plus LIBERO-Plus LIBERO-Plus LIBERO-Plus Goal</td><td colspan="2">Long</td><td colspan="2">ManiSkill3 RoboTwin 2.0</td><td colspan="2"></td><td colspan="2"> $\operatorname { A v g } .$ </td></tr><tr><td>IND OOD</td><td></td><td>IND</td><td>OOD</td><td>IND</td><td>OOD</td><td>IND</td><td>OOD</td><td>IND OOD</td><td>IND</td><td></td><td>OOD</td><td>IND OOD</td></tr><tr><td colspan="10"></td><td colspan="3"></td></tr><tr><td>SFT</td><td>22.4</td><td>14.8</td><td>42.0</td><td>27.6</td><td>18.4</td><td> $\pi _ { 0 }$  14.1</td><td>30.0 24.5</td><td>39.7</td><td>21.1</td><td>42.0</td><td></td><td>46.1</td><td>36.6</td><td>29.2</td></tr><tr><td>Vanilla RFT</td><td>78.8</td><td>48.9</td><td>70.8</td><td>53.4</td><td>55.6</td><td>37.9</td><td>66.0 46.1</td><td>66.2</td><td>35.5</td><td>65.9</td><td></td><td>62.1</td><td>66.6</td><td>48.1</td></tr><tr><td>DRIVE</td><td>79.6</td><td>50.2</td><td>74.4</td><td>53.5</td><td>54.8</td><td>39.8</td><td>66.4</td><td>50.8</td><td>68.4</td><td>40.3</td><td>70.6</td><td>71.4</td><td>69.3</td><td>53.4</td></tr><tr><td colspan="15">π0.5</td></tr><tr><td>SFT</td><td>85.6</td><td>74.8</td><td>86.4</td><td>80.9</td><td>52.8</td><td>50.7</td><td>38.4</td><td>32.6 55.3</td><td></td><td>34.2</td><td>78.7</td><td>79.7</td><td>66.657.9</td><td></td></tr><tr><td>Vanilla RFT</td><td>97.6</td><td>84.5</td><td>92.8</td><td>82.8</td><td>64.4</td><td>57.9</td><td>76.8</td><td>59.6</td><td>81.5</td><td>48.6</td><td>93.3</td><td>93.6</td><td>85.9</td><td>71.1</td></tr><tr><td>Higher Noise</td><td>93.6</td><td>84.5</td><td>94.0</td><td>86.6</td><td>60.4</td><td>56.7</td><td>74.0</td><td>58.2</td><td>81.8</td><td>45.9</td><td>93.3</td><td>93.0</td><td>85.2</td><td>70.1</td></tr><tr><td>Clip-Higher</td><td>96.4</td><td>81.7</td><td>92.8</td><td>84.3</td><td>64.8</td><td>57.3</td><td>77.6</td><td>59.9</td><td>80.0</td><td>52.9</td><td>92.9</td><td>94.0</td><td>85.3</td><td>72.6</td></tr><tr><td>KL Reg.</td><td>98.0</td><td>86.3</td><td>92.0</td><td>81.2</td><td>65.6</td><td>59.0</td><td>74.4</td><td>60.1</td><td>85.3</td><td>49.0</td><td>93.5</td><td>93.4</td><td>87.1</td><td>71.4</td></tr><tr><td>DRIVE</td><td>98.8</td><td>88.1</td><td>93.2</td><td>85.8</td><td>65.2</td><td>56.8</td><td>78.8</td><td>62.5</td><td>86.8</td><td>51.9</td><td>93.8</td><td>94.2</td><td>88.2</td><td>73.1</td></tr></table>

## 5.1 MAIN RESULTS

Table 1 summarizes the main simulation results across LIBERO-Plus, ManiSkill3, and RoboTwin 2.0. We compare Vanilla RFT and DRIVE on both $\pi _ { 0 }$ and $\pi _ { 0 . 5 }$ , with additional comparisons against Higher Noise, KL Regularization, and Clip-Higher on $\pi _ { 0 . 5 }$

DRIVE improves upon Vanilla RFT across both backbones and most benchmark splits. On $\pi _ { 0 } .$ DRIVE improves 11 of the 12 benchmark-split results, including all six OOD results, increasing the benchmark-family macro-average from 66.6% to 69.3% under IND conditions and from 48.1% to 53.4% under OOD conditions. The same pattern largely holds for $\pi _ { 0 . 5 }$ , where DRIVE again improves 11 of the 12 results and five of the six OOD results, raising the macro-average from 85.9% to 88.2% under IND conditions and from 71.1% to 73.1% under OOD conditions. Although the OOD gain is larger on $\pi _ { 0 }$ than on $\pi _ { 0 . 5 }$ , the improvements span the four LIBERO-Plus suites, ManiSkill3, and RoboTwin 2.0. This consistency across two VLA backbones and heterogeneous manipulation settings indicates that the benefit of successful-behavior diversity shaping is not confined to a particular model or task distribution, but instead reflects a broader effect that persists across different model architectures and evaluation settings.

The broader comparison on $\pi _ { 0 . 5 }$ shows that alternative exploration and regularization strategies can improve individual benchmarks, but their gains vary across conditions. Higher Noise, KL Regularization, and Clip-Higher each outperform Vanilla RFT in selected settings, yet none provides consistently stronger performance across the evaluated benchmarks, suggesting that these alternatives offer condition-specific gains rather than a uniform advantage across evaluation conditions. DRIVE achieves the highest benchmark-family macro-average under both IND (88.2%) and OOD (73.1%) evaluation while remaining competitive across individual benchmarks. Overall, these results show that explicitly shaping successful-behavior diversity provides broad performance gains across heterogeneous manipulation settings and improves generalization under distribution shift.

![](images/e96886dd686aae1bdfa77a89cda978ecec95970221626a4d894b0a4f22a26126.jpg)

![](images/99fe080a04f73ba677b9a46af2e3146f11fd97e5322183e6ceef2f0c6b097a30.jpg)

![](images/5f315423824b790af7b2f676b402b7e6c5b544e1226d576fda8cddffdb715592.jpg)

![](images/510f41b1f79b06948e5252b0066248f6c80f1f4ead68d47253174e8faabea605.jpg)  
Figure 4: Generalization and behavioral dynamics during RFT for $\pi _ { 0 . 5 }$ on ManiSkill3. (a) IND success rate across training steps. (b) Average OOD success rate across visual-language, semantic, and execution shifts at selected training steps for Vanilla RFT and DRIVE with two shaping strengths. (c) Normalized GAK similarity over all trajectories for Vanilla RFT and DRIVE. (d) Normalized GAK similarity over successful trajectories for Vanilla RFT and DRIVE. Behavioral analyses in (c–d) use DRIVE with $\alpha = 0 . 1$ . Lower similarity indicates greater behavioral diversity.

## 5.2 GENERALIZATION AND BEHAVIORAL DYNAMICS

We further examine how DRIVE’s generalization evolves during RFT and whether its performance gains are accompanied by changes in behavioral diversity. We first track IND and OOD performance across training checkpoints, then compare trajectory-level behavioral similarity over all rollouts and successful rollouts, and finally isolate the role of success-conditioned diversity shaping.

Performance evolution during RFT. Figure 4(a–b) shows the evolution of IND performance throughout RFT and OOD performance at selected evaluation steps for $\pi _ { 0 . 5 }$ on ManiSkill3, comparing Vanilla RFT with DRIVE under two shaping strengths, $\alpha = 0 . 1$ and $\alpha = 0 . 2$ . Under IND evaluation, both DRIVE variants improve rapidly and achieve stronger performance than Vanilla RFT over substantial portions of training. Under OOD evaluation, performance is less monotonic and fluctuates more across checkpoints, yet both DRIVE variants achieve higher success rates than Vanilla RFT at multiple evaluated checkpoints. The two shaping strengths exhibit broadly similar performance trends across both IND and OOD evaluation, showing that the observed improvement is consistent across the tested values of α. The non-monotonic OOD trajectories further indicate that continued optimization on the RFT distribution does not necessarily translate into steadily improving generalization. This discrepancy also suggests that IND performance alone provides an incomplete characterization of how the policy generalizes during RFT. Evaluating both distributions throughout training is therefore important for distinguishing improvements on the fine-tuning distribution from changes in robustness to distribution shift.

Successful-behavior diversity. We next examine whether the performance differences are accompanied by changes in behavioral diversity. Using the normalized GAK similarity introduced in Sec. 4.2, Fig. 4(c) compares all rollout trajectories, while Fig. 4(d) considers only successful trajectories. Across all rollouts, Vanilla RFT and DRIVE exhibit broadly similar similarity trends, with only a moderate separation emerging later in training. The difference becomes substantially clearer among successful trajectories, where DRIVE maintains lower similarity during the later stages of RFT, indicating greater diversity among tasksuccessful behaviors. This contrast shows that the behavioral effect of DRIVE is concentrated within successful trajectories rather than expressed as uniformly broader variation across the entire rollout distribution. Consequently, diversity measured over all rollouts can obscure changes that occur specifically within the task-valid solution space. Together with the stronger OOD performance in Fig. 4(b), these results suggest that DRIVE’s generalization gains are associated with greater diversity among successful behaviors, consistent with the motivation behind success-conditioned diversity shaping.

![](images/c3cc6c798bbdf8c1c2c2153043381989e022d7f99ec6f3dd12fac59d4cddd659.jpg)  
Figure 5: Effect of success conditioning on IND performance. IND evaluation success rates of DRIVE with and without success conditioning throughout RFT.

![](images/45d9c0693f3bec071f0ede192599015f2043f3dcfe5882929c959227db9e3890.jpg)

<table><tr><td>Task</td><td>Method</td><td>Clean Distractor Background Lighting OOD Avg.</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Click Bell</td><td>Vanilla RFT</td><td>80</td><td>60</td><td>70</td><td>65</td><td>65.0</td></tr><tr><td>DRIVE</td><td>85</td><td>75</td><td>75</td><td>65</td><td>71.7</td></tr><tr><td rowspan="2">Press Stapler</td><td>Vanilla RFT</td><td>75</td><td>70</td><td>65</td><td>55</td><td>63.3</td></tr><tr><td>DRIVE</td><td>80</td><td>75</td><td>80</td><td>70</td><td>75.0</td></tr></table>

Figure 6: Representative real-world evaluation conditions: Clean, Distractor, Background, and Lighting.  
Table 2: Real-world success rates (%) over 20 trials per task-condition pair. OOD Avg. denotes the mean over Distractor, Background, and Lighting.

Isolating the role of success conditioning. We finally isolate the role of success conditioning by comparing full DRIVE with a variant that applies the same diversity shaping without restricting the reward to successful trajectories. As shown in Fig. 5, the two variants exhibit comparable IND performance during the early stages of RFT, with no consistent advantage for either method. As training progresses, however, full DRIVE achieves higher success rates and the separation becomes clearer in the later stages. The delayed emergence of this gap suggests that success conditioning does not simply strengthen the optimization signal uniformly throughout training. Rather, the later separation suggests that restricting the diversity bonus to task-valid trajectories becomes more important as RFT progresses, avoiding explicit rewards for behavioral variation that does not lead to task success. These results indicate that diversity shaping is more effective when conditioned on task success than when applied indiscriminately across all trajectories.

## 5.3 REAL-WORLD EXPERIMENTS

Finally, we evaluate whether the generalization gains of DRIVE persist under physical deployment. We conduct experiments on a dual-arm platform equipped with two AgileX PiPER-X manipulators. To reduce the simulation-to-real discrepancy, we perform hand–eye and robot-coordinate calibration, reproduce the relative arm placement and manipulation workspace in simulation, align the camera viewpoints, and adjust the scene geometry and illumination to better match the physical environment. Additional details of the physical setup, simulation alignment, evaluation protocol, and representative rollout sequences are provided in Appendix E.

We evaluate two manipulation tasks, Click Bell and Press Stapler. Since the physical robot configuration differs from the original benchmark setup, we first collect expert trajectories in the aligned simulation environment to obtain a common SFT initialization, from which Vanilla RFT and DRIVE are trained separately. For real-world evaluation, each policy is tested under a clean condition and three distribution shifts involving distractor objects, tabletop-background changes, and lighting variations. Each task–condition pair is evaluated over 20 trials. Figure 6 shows representative evaluation conditions, while Table 2 reports the corresponding success rates.

DRIVE consistently improves performance under physical deployment. Across the eight task– condition combinations, DRIVE achieves higher success rates in seven settings and matches Vanilla RFT in the remaining one. Averaged across the two tasks, the clean-condition success rate increases from 77.5% to 82.5%. The improvement is larger under distribution shift, where the average OOD success rate increases from 64.1% to 73.3%, corresponding to a 9.2 percentage-point gain over Vanilla RFT. For Click Bell, the largest improvement occurs under Distractor (+15 points), while for Press Stapler the largest gains occur under Background and Lighting (both +15 points). In comparison, DRIVE improves the clean-condition success rate by 5 points on both tasks, suggesting that its advantage is more pronounced under distribution shift. These results are consistent with the simulation experiments and further show that the generalization gains of DRIVE persist when transferred from simulation to the physical robotic system.

## 6 CONCLUSIONS, LIMITATIONS, AND FUTURE WORK

We show that VLA RFT contracts behavior globally while expanding the accessibility and diversity of task-valid successful modes. Motivated by this finding, DRIVE uses potential-based, successconditioned diversity shaping to encourage broader solution coverage without rewarding diverse failures. Across three simulation benchmarks, DRIVE improves the OOD macro-average over Vanilla RFT by 5.3 points on π<sub>0</sub> and 2.0 points on π<sub>0.5</sub>, with a further 9.2-point gain on a dualarm AgileX PiPER-X platform. These results establish successful-mode coverage as both a usefu lens for understanding RFT and a practical objective for improving VLA generalization.

In DRIVE, pairwise GAK comparisons within matched rollout groups introduce additional computation and rely on VLM representations to capture meaningful behavioral differences. Future work could develop more efficient, learned diversity metrics that better distinguish task-solving strategies from incidental motion. Moreover, extending DRIVE to longer-horizon tasks, broader embodiments, and online real-world fine-tuning would further assess its scalability and generality.

## REFERENCES

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π<sub>0</sub>: A vision-language-action flow model for general robot control. In Proceedings ofRobotics: Science and Systems, 2025.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. In Proceedings of the 7th Conference on Robot Learning, volume 229 of Proceedings ofMachine Learning Research, pp. 2165– 2183. PMLR, 2023.

Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. In International Conference on Learning Representations, 2019.

Canyu Chen, Yuguang Yang, Zhewen Tan, Yizhi Wang, Ruiyi Zhan, Haiyan Liu, Xuanyao Mao, Jason Bao, Xinyue Tang, Linlin Yang, et al. Devil is in narrow policy: Unleashing exploration in driving vla models. arXiv preprint arXiv:2603.06049, 2026.

Kang Chen, Zhihao Liu, Tonghe Zhang, Zhen Guo, Si Xu, Hao Lin, Hongzhi Zang, Xiang Li, Quanlu Zhang, Zhaofei Yu, et al. π : Online RL fine-tuning for flow-based VLA models. arXiv preprint arXiv:2510.25889, 2025a.

Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025b.

Marco Cuturi. Fast global alignment kernels. In Proceedings of the International Conference on Machine Learning, pp. 929–936, 2011.

Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, and Sergey Levine. Diversity is all you need: Learning skills without a reward function. In Proceedings of International Conference on Learning Representations, 2019.

Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, et al. LIBERO-Plus: In-depth robustness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. DeepSeek-R1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025.

Anthony GX-Chen, Jatin Prakash, Jeff Guo, Rob Fergus, and Rajesh Ranganath. KL-regularized reinforcement learning for generative modelling is designed to mode collapse. In International Conference on Learning Representations, 2026.

Zican Hu, Xuyang Hu, Yiming Liu, Zuwei Long, Wei Liu, Yunzhuo Hao, Jiawei Gu, Linjie Li, Yu Cheng, Zhenhong Sun, Weibo Gu, Xing Sun, and Zhi Wang. Bridging interleaved multi modal reasoning as a unified decision process. arXiv preprint arXiv:2607.03748, 2026a.

Zican Hu, Shilin Zhang, Yafu Li, Jianhao Yan, Xuyang Hu, Leyang Cui, Xiaoye Qu, Chunlin Chen, Yu Cheng, and Zhi Wang. Diversity-incentivized exploration for versatile reasoning. In Proceedings of International Conference on Learning Representations, 2026b.

Physical Intelligence, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, et al. $\pi _ { 0 . 6 } ^ { * } \colon$ A VLA that learns from experience. arXiv preprint arXiv:2511.14759, 2025a.

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. $\pi _ { 0 . 5 } { : }$ A vision-language-action model with open-world generalization. In Proceedings of the Conference on Robot Learning, pp. 17–40, 2025b.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. In Proceedings of the Conference on Robot Learning, pp. 2679– 2713, 2024.

Yilun Kong, Yunpeng Qing, Guozheng Ma, Haoyu Wang, Li Shen, Zhi Hou, and Dacheng Tao. Extoken: Structured exploration for efficient vision-language-action reinforcement fine-tuning. arXiv preprint arXiv:2607.12931, 2026.

Saurabh Kumar, Aviral Kumar, Sergey Levine, and Chelsea Finn. One solution is not all you need: Few-shot extrapolation via structured MaxEnt RL. In Advances in Neural Information Processing Systems, volume 33, pp. 8198–8210, 2020.

Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Yang Zhaohui, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, et al. Simplevla-rl: Scaling vla training via reinforcement learning. In International Conference on Learning Representations, 2026.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. In Advances in Neural Information Processing Systems, volume 36, pp. 44776–44791, 2023.

Jijia Liu, Feng Gao, Bingwen Wei, Xinlei Chen, Qingmin Liao, Yi Wu, Chao Yu, and Yu Wang. What can RL bring to VLA generalization? an empirical study. In Advances in Neural Information Processing Systems, volume 38, pp. 97121–97151, 2025a.

Jinmei Liu, Haoru Li, Zhenhong Sun, Chaofeng Chen, Yatao Bian, Bo Wang, Daoyi Dong, and Zhi Wang. Beyond the Dirac Delta: Mitigating diversity collapse in reinforcement fine-tuning for image generation. In The Fortieth Annual Conference on Neural Information Processing Systems, 2026.

Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-RFT: Visual reinforcement fine-tuning. In 2025 IEEE/CVF International Conference on Computer Vision, pp. 2034–2044, 2025b.

Guanxing Lu, Wenkai Guo, Chubin Zhang, Yuheng Zhou, Haonan Jiang, Zifeng Gao, Yansong Tang, and Ziwei Wang. Vla-rl: Towards masterful and general robotic manipulation with scalable reinforcement learning. arXiv preprint arXiv:2505.18719, 2025.

Muhammad A. Masood and Finale Doshi-Velez. Diversity-inducing policy gradient: Using maximum mean discrepancy to find a set of diverse policies. In Proceedings of the International Joint Conference on Artificial Intelligence, pp. 5923–5929, 2019.

Andrew Y Ng, Daishi Harada, and Stuart Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In Proceedings of International Conference on Machine Learning, pp. 278–287, 1999.

Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In Proceedings of the International Conference on Machine Learning, pp. 2778–2787, 2017.

Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visuallanguage-action model. arXiv preprint arXiv:2501.15830, 2025.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Archit Sharma, Shixiang Gu, Sergey Levine, Vikash Kumar, and Karol Hausman. Dynamics-aware unsupervised discovery of skills. In International Conference on Learning Representations, 2020.

Shuhan Tan, Kairan Dou, Yue Zhao, and Philipp Krahenb ¨ uhl. Interactive post-training for vision- ¨ language-action models. arXiv preprint arXiv:2505.17016, 2025.

Stone Tao, Fanbo Xiang, Arth Shukla, Yuzhe Qin, Xander Hinrichsen, Xiaodi Yuan, Chen Bao, Xinsong Lin, Yulin Liu, Tse-kai Chan, Yuan Gao, Xuanlin Li, Tongzhou Mu, Nan Xiao, Arnav Gurha, Viswesh Nagaswamy Rajesh, Yong Woo Choi, Yen-Ru Chen, Zhiao Huang, Roberto Calandra, Rui Chen, Shan Luo, and Hao Su. Maniskill3: Gpu parallelized robotics simulation and rendering for generalizable embodied AI. In Proceedings ofRobotics: Science and Systems, 2025.

Xinyi Xie, Zican Hu, Zhanyu Liu, Yicheng Dong, Wenhao Wu, Zhenhong Sun, Haoran Li, Chunlin Chen, Zhi Wang, and Pichao Wang. Look before you leap: Distilling tree search into action evaluation for frozen vla models. arXiv preprint arXiv:2607.03751, 2026.

Yuquan Xue, Guanxing Lu, Zhenyu Wu, Chuanrui Zhang, Bofang Jia, Zhengyi Gu, and Ziwei Wang. Resample: A robust data augmentation framework via exploratory sampling for robotic manipulation. arXiv preprint arXiv:2510.17640, 2025.

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason under off-policy guidance. In Advances in Neural Information Processing Systems, 2025.

Rushuai Yang, Zhiyuan Feng, Tianxiang Zhang, Kaixin Wang, Chuheng Zhang, Li Zhao, Xiu Su, Yi Chen, and Jiang Bian. Discover, learn, and reinforce: Scaling vision-language-action pretraining with diverse rl-generated trajectories. arXiv preprint arXiv:2511.19528, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. DAPO: An open-source LLM reinforcement learning system at scale. In Advances in Neural Information Processing Systems, volume 38, pp. 113222– 113244, 2025.

Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693, 2024.

Runzhe Zhan, Yafu Li, Zhi Wang, Xiaoye Qu, Dongrui Liu, Jing Shao, Derek F Wong, and Yu Cheng. ExGRPO: Learning to reason from experience. In Proceedings of International Conference on Learning Representations, 2026.

Zhanguang Zhang, Zhiyuan Li, Behnam Rahmati, Rui Heng Yang, Yintao Ma, Amir Rasouli, Sajjad Pakdamansavoji, Yangzheng Wu, Lingfeng Zhang, Tongtong Cao, et al. Do world action models generalize better than vlas? a robustness study. arXiv preprint arXiv:2603.22078, 2026.

Jade Zou, Tao Huang, Weijie Kong, Junzhe Li, Yue Wu, Qi Tian, Jiangfeng Xiong, Jianwei Zhang, Liefeng Bo, and Zhao Zhong. Precise: Sde-consistent stochastic sampling for rl post-training of flow-matching models. arXiv preprint arXiv:2605.23522, 2026.

## APPENDIX

A Optimal Policy Invariance of DRIVE 14   
B Trajectory Representation and Global Alignment Similarity 14   
C Benchmark and Evaluation Details 15   
C.1 LIBERO-Plus 15   
C.2 ManiSkill3 16   
C.3 RoboTwin 2.0 . 17   
D Training and Implementation Details 18   
E Real-World Setup and Evaluation Details 19

## A OPTIMAL POLICY INVARIANCE OF DRIVE

Following $\mathrm { N g }$ et al. (1999), we provide the proof of Theorem 1, showing that the success-conditioned diversity signal preserves optimal policies when introduced through potential-based reward shaping.

Proof of Theorem 1. For a fixed rollout group, define the success-conditioned diversity potential

$$
\Phi ( s ) \triangleq y \hat { d } ( s ) ,
$$

where $y \in \{ 0 , 1 \}$ is the binary success indicator and ${ \hat { d } } ( s )$ is the normalized trajectory-level diversity. Since $\hat { d } ( s ) \in [ 0 , 1 ]$ , the potential $\Phi ( s )$ is bounded. The intrinsic reward in Theorem 1 can then be written as

$$
r ^ { \mathrm { d i v } } ( s , a , s ^ { \prime } ) = \gamma \Phi ( s ^ { \prime } ) - \Phi ( s ) = \gamma y \hat { d } ( s ^ { \prime } ) - y \hat { d } ( s ) .
$$

Let $Q _ { M } ^ { * } ( s , a )$ denote the optimal action-value function of the original MDP $M = ( S , A , P , R , \gamma )$ It satisfies the Bellman optimality equation

$$
Q _ { M } ^ { * } ( s , a ) = \mathbb { E } _ { s ^ { \prime } \sim P ( \cdot \mid s , a ) } \left[ r ( s , a , s ^ { \prime } ) + \gamma \operatorname* { m a x } _ { a ^ { \prime } \in A } Q _ { M } ^ { * } ( s ^ { \prime } , a ^ { \prime } ) \right] .
$$

Subtracting $\alpha \Phi ( s )$ from both sides and adding and subtracting $\alpha \gamma \Phi ( s ^ { \prime } )$ inside the expectation gives

$$
Q _ { M } ^ { * } ( s , a ) - \alpha \Phi ( s ) = \mathbb { E } _ { s ^ { \prime } } \Big [ r ( s , a , s ^ { \prime } ) + \alpha \big ( \gamma \Phi ( s ^ { \prime } ) - \Phi ( s ) \big ) + \gamma \operatorname* { m a x } _ { a ^ { \prime } \in A } \big ( Q _ { M } ^ { * } ( s ^ { \prime } , a ^ { \prime } ) - \alpha \Phi ( s ^ { \prime } ) \big ) \Big ] .
$$

The equality holds because $\Phi ( s ^ { \prime } )$ depends only on the successor state and is therefore independent of the maximization over $a ^ { \prime }$

Define $\widetilde { Q } ( s , a ) \triangleq Q _ { M } ^ { * } ( s , a ) - \alpha \Phi ( s )$ . Using $r ^ { \mathrm { d i v } } ( s , a , s ^ { \prime } ) = \gamma \Phi ( s ^ { \prime } ) - \Phi ( s )$ , we obtain

$$
\widetilde { Q } ( s , a ) = \mathbb { E } _ { s ^ { \prime } \sim P ( \cdot \vert s , a ) } \left[ r ( s , a , s ^ { \prime } ) + \alpha r ^ { \mathrm { d i v } } ( s , a , s ^ { \prime } ) + \gamma \operatorname* { m a x } _ { a ^ { \prime } \in A } \widetilde { Q } ( s ^ { \prime } , a ^ { \prime } ) \right] .
$$

This is precisely the Bellman optimality equation for the transformed MDP

$$
M ^ { \prime } = ( S , A , P , R + \alpha R ^ { \mathrm { { d i v } } } , \gamma ) .
$$

By the uniqueness of the Bellman optimality fixed point,

$$
Q _ { M ^ { \prime } } ^ { * } ( s , a ) = \widetilde { Q } ( s , a ) = Q _ { M } ^ { * } ( s , a ) - \alpha \Phi ( s ) .
$$

Because the subtracted term $\alpha \Phi ( s )$ depends only on the state and not on the action,

$$
\begin{array} { c } { { \arg \displaystyle \operatorname* { m a x } _ { a \in { \cal A } } Q _ { M ^ { \prime } } ^ { * } ( s , a ) = \arg \displaystyle \operatorname* { m a x } _ { a \in { \cal A } } \left[ Q _ { M } ^ { * } ( s , a ) - \alpha \Phi ( s ) \right] } } \\ { { = \arg \displaystyle \operatorname* { m a x } _ { a \in { \cal A } } Q _ { M } ^ { * } ( s , a ) . } } \end{array}
$$

Therefore, the sets of optimal actions are identical at every state. Every optimal policy in $M ^ { \prime }$ is consequently optimal in $M$ , and every optimal policy in M is also optimal in $M ^ { \prime }$ . Hence, DRIVE’s success-conditioned potential-based shaping preserves the set of optimal policies. □

## B TRAJECTORY REPRESENTATION AND GLOBAL ALIGNMENT SIMILARITY

This section provides additional details on the trajectory representation and temporally aligned similarity used in Sec. 4.2.

Trajectory representation. For each policy decision, we extract a feature $z _ { i , t }$ from the VLM prefix representation already computed by the policy, using mean pooling over valid prefix tokens. If trajectory $\tau _ { i }$ contains $L _ { i }$ action chunks, we additionally extract a feature from the terminal observation and represent the trajectory as

$$
Z _ { i } = \left( z _ { i , 0 } , \dots , z _ { i , L _ { i } - 1 } , z _ { i , \mathrm { t e r m } } \right) , \qquad M _ { i } : = | Z _ { i } | = L _ { i } + 1 .
$$

Thus, each trajectory is retained as an ordered variable-length feature sequence rather than collapsed into a single embedding.

Local similarity and temporal alignment. For two features, we use a shifted cosine similarity

$$
k _ { i j } ( t , u ) = \mathrm { c l i p } \left( \frac { \widehat { z } _ { i , t } ^ { \top } \widehat { z } _ { j , u } + 1 } { 2 } , \epsilon , 1 \right) ,
$$

where $\widehat { z }$ denotes the $\ell _ { 2 }$ -normalized feature and $\epsilon = 1 0 ^ { - 1 2 }$

To accommodate different execution speeds and trajectory lengths, we compute the Global Alignment Kernel (GAK) Cuturi (2011). Let $G _ { i j } ( a , b )$ denote the alignment kernel between the first a features of $Z _ { i }$ and the first b features of $Z _ { j }$ . With $G _ { i j } ( 0 , 0 ) = 1$ and the remaining boundary entries set to zero,

$$
G _ { i j } ( a , b ) = k _ { i j } ( a - 1 , b - 1 ) \left[ G _ { i j } ( a - 1 , b ) + G _ { i j } ( a , b - 1 ) + G _ { i j } ( a - 1 , b - 1 ) \right] .
$$

The three transitions allow either trajectory, or both, to advance while preserving temporal order. Unlike a single best-path alignment, this recurrence aggregates contributions from all admissible monotonic alignment paths.

Normalized similarity. The unnormalized trajectory kernel is

$$
K _ { \mathrm { G A K } } ( Z _ { i } , Z _ { j } ) = G _ { i j } ( | Z _ { i } | , | Z _ { j } | ) .
$$

and we use the normalized similarity

$$
\widetilde { K } _ { i j } = \frac { K _ { \mathrm { G A K } } ( Z _ { i } , Z _ { j } ) } { \sqrt { K _ { \mathrm { G A K } } ( Z _ { i } , Z _ { i } ) K _ { \mathrm { G A K } } ( Z _ { j } , Z _ { j } ) } } .
$$

The implementation evaluates the dynamic program in log space for numerical stability. The resulting $\widetilde { K } _ { i j } \in [ 0 , 1 ]$ is used as the trajectory-level behavioral similarity throughout the paper.

## C BENCHMARK AND EVALUATION DETAILS

## C.1 LIBERO-PLUS

We evaluate all four standard LIBERO suites—Spatial, Object, Goal, and Long—using the task variations provided by LIBERO-Plus (Fei et al., 2025; Liu et al., 2023). Our split preserves all ten original base tasks in each suite and separates IND and OOD conditions through the variation values applied to these tasks, rather than by holding out base-task identities. This keeps the underlying manipulation semantics fixed while testing generalization to changes in scene configuration, visual appearance, language, and robot/environment state.

IND construction. For each suite, the RFT/IND pool contains all ten clean LIBERO tasks together with 240 selected LIBERO-Plus variants, yielding 250 task definitions per suite. We construct the IND variation pool from four perturbation families that can be partitioned by discrete configurations or identities: Confounding Objects, Target Object Pose, Scene Theme, and Surface Appearance. The selected variants cover all ten base tasks and remain fixed throughout training. Table 3 summarizes the resulting composition.

For Spatial, Object, and Goal, the selected variants are distributed across the ten base tasks to maintain broad task coverage while exposing the policy to multiple values from each IND perturbation family. For Long, which contains longer-horizon and more fragile manipulation sequences, we retain the same overall pool size but use a more conservative allocation of layout perturbations and a larger proportion of background variations. The Spatial, Object, and Goal splits sample uniformly from their complete sets of selected task definitions. For LIBERO Long, however, the number of available definitions differs across base tasks, so we use base-task-uniform hierarchical sampling. Specifically, we first sample one of the ten base tasks uniformly, then sample uniformly from the selected clean and Plus task definitions associated with that base task, and finally sample a valid initial-state index for the selected definition. This keeps the marginal sampling frequency of the Long base tasks balanced despite differences in the number of available Plus variants.

OOD construction. OOD evaluation uses perturbation values that are excluded from the IND/RFT pool. For object-layout perturbations, we separate distractor identities and target-object configurations between IND and OOD. Similarly, scene themes and surface appearances are partitioned by their underlying identities or configurations to avoid overlap between the two splits. The remaining LIBERO-Plus dimensions—camera viewpoints, robot initial states, language instructions, lighting conditions, and sensor noise—provide additional held-out distribution shifts for OOD evaluation. Thus, IND and OOD retain the same underlying manipulation tasks while differing in the perturbation values that define the corresponding observation and execution conditions.

LIBERO-Plus organizes its variations into seven major perturbation families: Objects Layout, Camera Viewpoints, Robot Initial States, Language Instructions, Light Conditions, Background Textures, and Sensor Noise. Across the four suites, the complete LIBERO-Plus manifests contain 2,402 Spatial, 2,518 Object, 2,591 Goal, and 2,519 Long variants. Table 4 summarizes their distribution across perturbation families.

Table 3: Composition of the LIBERO-Plus IND task pools. Each suite contains the ten original clean LIBERO tasks and 240 selected Plus variants.
<table><tr><td>Suite</td><td>Clean</td><td>Confounding</td><td>Target Pose</td><td>Scene Theme</td><td>Surface</td><td>Total</td></tr><tr><td>Spatial</td><td>10</td><td>78</td><td>66</td><td>44</td><td>52</td><td>250</td></tr><tr><td>Object</td><td>10</td><td>77</td><td>72</td><td>37</td><td>54</td><td>250</td></tr><tr><td>Goal</td><td>10</td><td>81</td><td>63</td><td>42</td><td>54</td><td>250</td></tr><tr><td>Long</td><td>10</td><td>65</td><td>50</td><td>52</td><td>73</td><td>250</td></tr></table>

Table 4: LIBERO-Plus perturbation statistics for the four evaluated suites. Counts denote variants in the complete Plus manifests and are distinct from the selected IND subsets in Table 3.
<table><tr><td>Perturbation family</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td></tr><tr><td>Objects Layout</td><td>385</td><td>403</td><td>425</td><td>312</td></tr><tr><td>Camera Viewpoints</td><td>376</td><td>396</td><td>408</td><td>419</td></tr><tr><td>Robot Initial States</td><td>350</td><td>398</td><td>409</td><td>393</td></tr><tr><td>Language Instructions</td><td>390</td><td>354</td><td>410</td><td>383</td></tr><tr><td>Light Conditions</td><td>292</td><td>297</td><td>279</td><td>274</td></tr><tr><td>Background Textures</td><td>258</td><td>248</td><td>281</td><td>289</td></tr><tr><td>Sensor Noise</td><td>351</td><td>422</td><td>379</td><td>449</td></tr><tr><td>Total</td><td>2,402</td><td>2,518</td><td>2,591</td><td>2,519</td></tr></table>

Evaluation protocol. We evaluate each task definition using exactly one fixed initial state. Specif ically, we always select the first available initialization associated with that task definition. Original LIBERO tasks may provide multiple initialization states (e.g., 50 initializations), whereas many LIBERO-Plus variants provide only a single state; in both cases, only the first initialization is used for evaluation. Consequently, every evaluated task definition contributes exactly one episode to the reported metric, independent of the number of initialization states provided by the benchmark. The same fixed task–initialization pairs are used for all compared methods and VLA backbones. Performance is reported as the mean binary task success rate over the corresponding IND or OOD evaluation set.

## C.2 MANISKILL3

We evaluate ManiSkill3 (Tao et al., 2025) using the PutOnPlateInScene25Main-v3 pickand-place environment, following the generalization setting of RL4VLA (Liu et al., 2025a). Unlike LIBERO-Plus, which contains multiple semantic base tasks, the ManiSkill3 evaluation uses a common manipulation objective and constructs distribution shifts by modifying the visual observation, semantic grounding requirements, or execution conditions. The IND environment contains 16 training objects, 16 training scenes, and randomized object and plate initial states.

OOD construction. We organize the held-out evaluation conditions into three groups: Visual-Language OOD, Semantic OOD, and Execution OOD. Visual-Language OOD changes the appearance of the visual input or the language expression while preserving the underlying manipulation objective. The six settings consist of an instruction variation, Instruct-v1; an unseen background appearance, VisionImage-v1; and four texture perturbations, VisionTexture03-v1, VisionTexture05-v1, VisionWhole03-v1, and VisionWhole05-v1. The texture variants apply either local foreground or global image-level perturbations at two strengths.

Semantic OOD modifies the grounding problem while retaining the same basic pick-and-place skill. We consider four settings: MultiCarrot-v1 (train) and MultiCarrot-v1 (test), which introduce multiple candidate objects, and MultiPlate-v1 (train) and MultiPlate-v1 (test), which introduce multiple candidate receptacles. The train variants use identities from the training object/receptacle sets, whereas the test variants introduce held-out identities. These settings therefore test whether the policy can resolve the target object or receptacle from language in scenes containing semantically competing alternatives.

Execution OOD changes the state distribution or environment during task execution rather than primarily modifying appearance or semantic content. The three settings cover shifts in initial object/plate configurations and dynamic changes during an episode. For example, Position-v1 changes the initial spatial distribution, while PositionChangeTo-v1 changes the target-object position during execution, requiring the policy to react to a modified environment. Together, these cases evaluate spatial-state generalization and closed-loop recovery under execution-time distribution shifts.

Table 5: ManiSkill3 evaluation protocol. Each setting is evaluated over 320 episodes. The OOD benchmark contains 13 settings grouped into visual-language, semantic, and execution shifts.
<table><tr><td>Split / Category</td><td>Evaluation focus</td><td>Settings</td><td>Episodes</td></tr><tr><td>IND</td><td>Training objects, scenes, and state distribution</td><td>1</td><td>320</td></tr><tr><td>Visual-Language OOD</td><td>Visual appearance and language variation</td><td>6</td><td>1,920</td></tr><tr><td>Semantic OOD</td><td>Object/receptacle grounding under semantic ambiguity</td><td>4</td><td>1,280</td></tr><tr><td>Execution OOD</td><td>Initial-state shifts and execution-time changes</td><td>3</td><td>960</td></tr><tr><td>Total</td><td>IND + three OOD categories</td><td>14</td><td>4,480</td></tr></table>

Evaluation protocol. Each evaluation setting is run for 320 episodes using a fixed evaluation configuration shared by all compared methods and backbones. This yields 1,920 episodes for Visual-Language OOD, 1,280 for Semantic OOD, and 960 for Execution OOD, for a total of 4,160 OOD episodes in addition to the 320 IND episodes. Success is determined by the environment’s binary task-completion criterion. We first average success over the episodes belonging to each evaluation setting and then aggregate the settings within their corresponding OOD category. The resulting IND and OOD success rates are used for the comparisons reported in the main paper.

## C.3 ROBOTWIN 2.0

We evaluate RoboTwin 2.0 (Chen et al., 2025b) on two dual-arm manipulation tasks, Click Bell and Press Stapler. To evaluate generalization beyond the nominal task distribution, we construct task-specific IND and OOD initialization sets following the structured perturbation design of RoboTwin 2.0-Plus (Zhang et al., 2026). The underlying task objective remains unchanged across the two splits; the distribution shift is induced through controlled changes to objects, scene appearance, camera and lighting conditions, robot state, language, and visual observations.

IND/OOD construction. For each manipulation task, we construct a set of 1,500 evaluation configurations. The first 1,000 configurations, indexed by [0, 1000), define the IND set, while the remaining 500, indexed by [1000, 1500), define the OOD set. The IND set contains clean configurations together with variations in object configuration, background appearance, and robot initial state. The OOD set introduces held-out values from these dimensions and additionally includes shifts in lighting, camera viewpoint, language instruction, and sensor observations. Table 6 summarizes the composition used for each task.

Table 6: RoboTwin 2.0 IND/OOD construction for each evaluated task. The same split structure is independently instantiated for Click Bell and Press Stapler.
<table><tr><td>Perturbation family</td><td>IND</td><td>OOD</td><td>Description</td></tr><tr><td>Clean</td><td>50</td><td>0</td><td>Nominal task configurations</td></tr><tr><td>Objects</td><td>134</td><td>34</td><td>Object additions and pose variations</td></tr><tr><td>Background</td><td>750</td><td>150</td><td>Scene/background appearance variations</td></tr><tr><td>Lighting</td><td>0</td><td>50</td><td>Held-out illumination conditions</td></tr><tr><td>Camera</td><td>0</td><td>50</td><td>Held-out camera configurations</td></tr><tr><td>Robot State</td><td>66</td><td>16</td><td>Robot initial-state variations</td></tr><tr><td>Language</td><td>0</td><td>50</td><td>Held-out instruction variations</td></tr><tr><td>Sensor</td><td>0</td><td>150</td><td>Visual/sensor perturbations</td></tr><tr><td>Total</td><td>1,000</td><td>500</td><td>1,500 initializations per task</td></tr></table>

Perturbation generation. The manifest combines explicitly validated physical-state configurations with deterministically derived observation-level perturbations. For each task, 300 configurations are grounded in expert-validated scene states: 250 belong to IND, consisting of 50 clean configurations, 67 object-addition variants, 67 object-pose variants, and 66 robot-state variants; the remaining 50 are OOD anchors, consisting of 17 object-addition, 17 object-pose, and 16 robot-state variants. The other 1,200 configurations are generated deterministically from validated scene states by modifying background, lighting, camera, language, or sensor conditions. This construction allows appearance-level shifts to reuse a physically valid underlying manipulation scene rather than requiring a separate expert demonstration for every rendered variation.

The object perturbations contain two primary subtypes: object additions (O1) and target/object pose changes (O2). The remaining perturbations operate primarily on scene appearance, observation, instruction, or robot initialization. All perturbations are instantiated as single-factor variations according to the manifest definition, allowing the source of each distribution shift to be controlled during evaluation.

Evaluation protocol. For each task, the full IND pool contains 1,000 evaluation configurations. We uniformly sample a fixed subset of 350 configurations from this pool for evaluation and reuse exactly the same subset across all compared methods and VLA backbones. For OOD evaluation, we use the full set of 500 predefined OOD configurations. The same perturbation assignments are used for all compared methods and backbones.

Each selected configuration produces one rollout and is scored using the binary task-success criterion defined by the RoboTwin environment. We compute success rates separately for Click Bell and Press Stapler, and report their macro-average as the overall RoboTwin 2.0 result in the main table.

## D TRAINING AND IMPLEMENTATION DETAILS

This section provides additional details of the RFT training protocol, rollout collection, DRIVE implementation, and hyperparameters used in our experiments. Unless otherwise specified, all methods within the same benchmark–backbone setting are initialized from the same SFT checkpoint and use the same training horizon, rollout distribution, and PPO optimization configuration. Training horizons are selected per benchmark–backbone setting according to task difficulty and convergence behavior and are held fixed across the compared methods.

Common RFT setup. All experiments use PPO with Flow-SDE sampling and are initialized from the same SFT checkpoint within each benchmark–backbone setting. We use $\gamma = 0 . 9 9 , { \mathrm { G A E } } \lambda =$ 0.95, and PPO clip ratio 0.2 throughout. Benchmark-specific optimization, rollout, and action-chunk parameters are reported in Tables 7 and 8. Here, H denotes the action prediction horizon and H<sup>′</sup> the number of actions executed before replanning.

Table 7: Training hyperparameters for LIBERO-Plus. Runner epochs and DRIVE shaping coefficients correspond to the final experimental configurations. All compared methods within each column use the same runner horizon.
<table><tr><td rowspan="2">Parameter</td><td colspan="4">π0</td><td colspan="4">π0.5</td></tr><tr><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td></tr><tr><td>Runner epochs</td><td>200</td><td>300</td><td>400</td><td>200</td><td>300</td><td>250</td><td>100</td><td>300</td></tr><tr><td>Global batch size</td><td>2048</td><td>2048</td><td>2048</td><td>2048</td><td>2048</td><td>2048</td><td>2048</td><td>2048</td></tr><tr><td>PPO update epochs</td><td>4</td><td>4</td><td>4</td><td>4</td><td>3</td><td>3</td><td>3</td><td>4</td></tr><tr><td>Actor learning rate</td><td>5×10−6</td><td>5×10−6</td><td>5×10−6</td><td>5×10−6</td><td>5×10−6</td><td>5×10-6</td><td>5×10−6</td><td>5×10−6</td></tr><tr><td>Critic learning rate</td><td>1×10−4</td><td>1×10−4</td><td>1×10−4</td><td>1×10 -4</td><td>1×10⁻4</td><td>1×10-4</td><td>1×10−4</td><td>1×10−4</td></tr><tr><td>LR schedule</td><td>Const.</td><td>Const.</td><td>Const.</td><td>Const.</td><td>Const.</td><td>Const.</td><td>Const.</td><td>Cosine</td></tr><tr><td>Reward discount γ</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td></tr><tr><td>GAEλ PPO clip €</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td></tr><tr><td></td><td>0.20</td><td>0.20</td><td>0.20</td><td>0.20</td><td>0.20</td><td>0.20</td><td>0.20</td><td>0.20</td></tr><tr><td>Episode horizon</td><td>240</td><td>240</td><td>320</td><td>480</td><td>240</td><td>240</td><td>320</td><td>480</td></tr><tr><td>Sampling groups</td><td>64</td><td>64</td><td>64</td><td>64</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Rollouts / group</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Trajectories / collection</td><td>512</td><td>512</td><td>512</td><td>512</td><td>512</td><td>512</td><td>512</td><td>512</td></tr><tr><td>Prediction horizon H</td><td>50</td><td>50</td><td>50</td><td>50</td><td>10</td><td>10</td><td>10</td><td>10</td></tr><tr><td>Execution horizon H&#x27;</td><td>10</td><td>10</td><td>10</td><td>10</td><td>10</td><td>10</td><td>10</td><td>10</td></tr><tr><td>Flow-SDE denoise steps</td><td>4</td><td>4</td><td>4</td><td>4</td><td>3</td><td>5</td><td>5</td><td>5</td></tr><tr><td>Flow-SDE noise σ</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.3</td><td>0.3</td><td>0.5</td></tr><tr><td>DRIVE α</td><td>0.10</td><td>0.10</td><td>0.05</td><td>0.05</td><td>0.10</td><td>0.10</td><td>0.10</td><td>0.10</td></tr></table>

Rollout collection. We sample 64 groups with 8 stochastic rollouts per group, yielding 512 trajectories per collection. Rollouts within a group share the same task and environment condition, and DRIVE computes trajectory similarity only within each group.

DRIVE implementation. For each rollout, we construct a temporal feature sequence from the VLM-prefix representations associated with the executed action chunks. Valid token features are aggregated using mean pooling, and the terminal observation is included when constructing the trajectory representation. Within each rollout group, we compute pairwise normalized Global Alignment Kernel (GAK) similarity between trajectories as described in Sec. 4.2.

The resulting relative diversity signal is used as an intrinsic shaping term. Importantly, the shaping reward is success-conditioned: only successful trajectories receive the diversity bonus, preventing behaviorally unusual failures from being explicitly rewarded. The shaping term is applied at the terminal action chunk. The coefficient α controls the strength of this term and is held constant throughout training; we do not use an α decay schedule in the reported experiments. Benchmarkspecific values are listed in Tables 7 and 8.

Baseline implementations. The auxiliary baselines use the same underlying RFT pipeline and modify only the component associated with the corresponding exploration or regularization strategy. Vanilla RFT uses the standard Flow-SDE sampling configuration, PPO clipping, and environment reward without additional diversity shaping. Higher Noise increases the Flow-SDE sampling noise to σ = 0.7. KL Regularization adds a KL penalty toward the SFT reference policy with coefficient β = 0.5. Clip-Higher keeps the lower PPO clipping bound at 0.20 while increasing the upper bound from 0.20 to 0.28. All other benchmark-specific settings follow the corresponding Vanilla RFT configuration.

## E REAL-WORLD SETUP AND EVALUATION DETAILS

This section provides additional details of the physical platform, simulation-to-real alignment, evaluation protocol, and representative real-world rollouts used in Sec. 5.3.

Table 8: Training hyperparameters for ManiSkill3 and RoboTwin 2.0. Training horizons are shared by all compared methods within each benchmark–backbone setting.
<table><tr><td></td><td colspan="2">ManiSkill3</td><td colspan="2">Click Bell</td><td colspan="2">Press Stapler</td></tr><tr><td>Parameter</td><td>π0</td><td>π0.5</td><td>π0</td><td> $\pi _ { 0 . 5 }$ </td><td>π0</td><td>π0.5</td></tr><tr><td>Runner epochs</td><td>300</td><td>200</td><td>100</td><td>100</td><td>100</td><td>50</td></tr><tr><td>Global batch size</td><td>5120</td><td>5120</td><td>2048</td><td>2048</td><td>2048</td><td>2048</td></tr><tr><td>PPO update epochs</td><td>4</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td></tr><tr><td>Actor learning rate</td><td> $7 . 9 1 \times 1 0 ^ { \cdot }$  -6</td><td> $7 . 9 1 \times 1 0 ^ { - }$  -6</td><td>5.6×10 -6</td><td>5×10−⁶</td><td> $5 . 6 \times 1 0 ^ { - }$  -6</td><td>5×10−⁶</td></tr><tr><td>Critic learning rate</td><td> $1 . 5 5 \times 1 0 ^ { \cdot }$  -4</td><td> $1 . 5 5 \times 1 0 ^ { - }$  -4</td><td> $1 . 1 \times 1 0 ^ { - 4 }$ </td><td>1×10 -4</td><td> $1 . 1 \times 1 0 ^ { \cdot }$  -4</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>LR schedule</td><td>Const.</td><td>Const.</td><td>Const.</td><td>Const.</td><td>Const.</td><td>Const.</td></tr><tr><td>Reward discount γ</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td><td>0.99</td></tr><tr><td>GAE λ</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td><td>0.95</td></tr><tr><td>PPO clip €</td><td>0.20</td><td>0.20</td><td>0.20</td><td>0.20</td><td>0.20</td><td>0.20</td></tr><tr><td>Episode horizon</td><td>80</td><td>80</td><td>200</td><td>200</td><td>200</td><td>200</td></tr><tr><td>Sampling groups</td><td>64</td><td>64</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Rollouts / group</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Trajectories / collection</td><td>512</td><td>512</td><td>512</td><td>512</td><td>512</td><td>512</td></tr><tr><td>Prediction horizon H</td><td>8</td><td>8</td><td>50</td><td>50</td><td>50</td><td>50</td></tr><tr><td>Execution horizon H&#x27;</td><td>5</td><td>5</td><td>20</td><td>20</td><td>20</td><td>20</td></tr><tr><td>Flow-SDE denoise steps</td><td>4</td><td>4</td><td>4</td><td>5</td><td>4</td><td>5</td></tr><tr><td>Flow-SDE noise σ</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td>DRIVE α</td><td>0.10</td><td>0.10</td><td>0.02</td><td>0.05</td><td>0.05</td><td>0.05</td></tr></table>

Table 9: Method-specific modifications relative to Vanilla RFT.
<table><tr><td>Method</td><td>Modification</td></tr><tr><td>Vanilla RFT</td><td>Standard PPO + Flow-SDE</td></tr><tr><td>Higher Noise</td><td>Flow-SDE noise  $\sigma = 0 . 7$ </td></tr><tr><td>KL Regularization</td><td>KL penalty  $\beta = 0 . 5$ </td></tr><tr><td>Clip-Higher</td><td>PPO upper clip  $0 . 2 0  0 . 2 8$ </td></tr><tr><td>DRIVE</td><td>Success-conditioned GAK shaping</td></tr></table>

Physical platform. Our real-world experiments are conducted on a dual-arm manipulation platform equipped with two AgileX PiPER-X manipulators. The workspace contains the two robot arms, wrist-mounted observation cameras, a planar manipulation surface, and the task objects used for Click Bell and Press Stapler. Because this hardware configuration differs from the original RoboTwin setup, we construct a corresponding simulation environment whose robot placement, workspace geometry, and observation viewpoints are aligned with the physical system before collecting demonstrations and performing RFT.

Geometric and camera calibration. We use a metric planar ChArUco board as the common calibration frame for the dual-arm workspace. The left and right manipulators are calibrated independently using the same procedure and their respective wrist-mounted cameras. For each arm, the end effector is moved through 48 predefined poses spanning different positions and orientations in the reachable workspace. At each pose, the robot state provides the base-to-end-effector transform, while ChArUco detection followed by solvePnP estimates the board-to-camera transform from the RGB observations. Multiple valid estimates collected at the same pose are fused before calibration.

The resulting multi-pose observations are passed to OpenCV’s Shah robot-world/hand-eye solver, which jointly recovers the pose of each robot base in the common workspace frame and the rigid transform between the end effector and its wrist-mounted camera. For calibration pose i, the transformations satisfy

$$
{ } ^ { W } \mathbf { T } _ { B } { } ^ { B } \mathbf { T } _ { E } ^ { \left( i \right) E } \mathbf { T } _ { C } = { } ^ { W } \mathbf { T } _ { C } ^ { \left( i \right) } ,\tag{12}
$$

where W, B, E, and C denote the ChArUco/workspace, robot-base, end-effector, and wrist-camera coordinate frames, respectively. Calibrating both arms with respect to the same workspace frame also provides their relative placement.

Simulation alignment. We construct the corresponding RoboTwin setup to match the geometry and observation configuration of the physical dual-arm system. The calibrated physical workspace provides a common reference for the relative robot arrangement, while the RoboTwin embodiment uses the corresponding PiPER-X asset configuration and inter-arm spacing. We additionally reproduce the manipulation-plane geometry and relative task layout to match the real setup.

For camera alignment, the real-camera intrinsics are transferred to the simulated observation cameras and adjusted to the rendered image resolution. The wrist-camera extrinsics are not applied a second time from the hand-eye calibration; instead, they remain defined by the fixed PiPER-X URDF asset used by the RoboTwin embodiment. This avoids double-applying the camera-to-robot transform while preserving consistent wrist-camera viewpoints. We further match the static observation viewpoint and adjust scene illumination and visual appearance to reduce the remaining sim-to-real discrepancy.

Task initialization and training. We evaluate two manipulation tasks, Click Bell and Press Stapler. Since the physical robot configuration differs from the original benchmark configuration, we first collect expert trajectories in the aligned simulation environment and use them to obtain a common SFT initialization. Vanilla RFT and DRIVE are then trained independently from the same SFT checkpoint in the aligned simulation environment. The resulting policies are evaluated on the physical platform using the same observation and action interface.

Real-world evaluation protocol. For each task, we evaluate one clean condition together with three distribution shifts: Distractor, Background, and Lighting. The Distractor condition introduces additional objects into the workspace; the Background condition changes the tabletop appearance; and the Lighting condition changes scene illumination while preserving the underlying task. These conditions probe whether the policy remains robust to visual and contextual changes beyond the nominal physical setup.

Each task–condition pair is evaluated over 20 trials, and the same evaluation conditions are used for Vanilla RFT and DRIVE. A rollout is considered successful when the corresponding physical manipulation objective is completed. The quantitative results are reported in Table 2.

Representative physical rollouts. Figure 7 provides qualitative examples of successful physical executions under the four evaluation conditions. For clarity, we visualize Click Bell, for which the task objective remains identical while the surrounding visual context changes. Each row contains five temporally ordered frames sampled from a single rollout, showing the progression from the initial scene to successful interaction with the target.

![](images/51acbed54a7be07445a35d05a964a43bc71658b9f78fde1b2b4656ddcd64e2fb.jpg)  
Figure 7: Representative successful real-world Click Bell rollouts under Clean, Distractor, Background, and Lighting conditions (top to bottom). Each row shows a temporally ordered sequence of selected frames from one physical execution.