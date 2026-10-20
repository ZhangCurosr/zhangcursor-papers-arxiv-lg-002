# Predictive Bandit Curriculum Reinforcement Learning via Gradient-Based Sample Group Discovery

Liang Xu<sup>1,2</sup>, Shufan Shen<sup>1,2</sup>, Zhongni Hou<sup>3,†</sup>, Chao Bi<sup>1</sup>, Yufei Zhang<sup>3</sup>, Wei Lin<sup>3</sup>, Guojun Yin<sup>3</sup>, Shuhui Wang<sup>1,†</sup>

<sup>1</sup>State Key Lab. of AI Safety, Institute of Computing Technology,

Chinese Academy of Sciences

<sup>2</sup>University of Chinese Academy of Sciences <sup>3</sup>Meituan {xuliang23z}@ict.ac.cn

## Abstract

Curriculum learning, which organizes training samples according to their learning priority, has been widely adopted in reinforcement learning to improve training efficiency. However, existing methods commonly construct curriculum units using predefined data properties and estimate their utility by simply aggregating a single proxy signal from recent training history. Such designs may group samples with different optimization effects and fail to anticipate how their utility evolves in relation to the policy. We propose Predictive Gradient-based Bandit (PGB) Curriculum, which first discovers optimization-aware learning units by clustering projected sample-level gradients. PGB then employs a lightweight temporal predictor to forecast the future utility of each unit and converts the predictions into a Boltzmann sampling distribution for adaptive data allocation. Experiments across multiple model scales and reasoning benchmarks show that PGB consistently outperforms standard GRPO, difficultybased curricula, and history-based scheduling strategies, demonstrating improved data utilization and training efficiency.

## 1 Introduction

Reinforcement learning (RL) has emerged as a key paradigm for improving the reasoning abilities of large language models (LLMs) (OpenAI, 2024; Guo et al., 2025; Schulman et al., 2017; Shao et al., 2024). Despite achieving remarkable performance, most RL methods sample prompts from a fixed uniform distribution over the training dataset during training. Such a plain sampling strategy may lead to inefficient data utilization, i.e., well-learned samples offer diminishing marginal improvements, whereas overly challenging samples may yield indistinguishable rewards and thus weak learning signals (Fang et al., 2019). Moreover, as the policy evolves, the same sample may transition from being overly difficult to informative and eventually to well learned, indicating that its role in optimization changes throughout training.

![](images/b88e3829c558c100a11fef289192a051bcbb8b4faa2d7a361092df801d7e86ee.jpg)  
(a) Comparison between existing MAB-based CRL and PGB.

![](images/ea5d0270d63ccf20f82d9b966b7525083b4c025e57dd6e11c3d60d0ec485602c.jpg)  
(b) Gradient-space distribution of predefined task categories.  
Figure 1: (a) Comparison between existing curriculum methods and PGB. (b) Task groups in gradient space. Each point is a sample, colored by task category.

Recent studies have incorporated curriculum learning (Bengio et al., 2009) into RL post-training to dynamically prioritize samples that are most beneficial to the current policy along the learning procedure. A common solution first organizes training samples into predefined groups and then adaptively adjusts the sampling probability of each group throughout optimization (Lin et al., 2025). The samples are usually clustered according to predefined attributes or statistics such as difficulty (Campos, 2021), success rate (Xi et al., 2024), and task categories (Teh et al., 2017). For sampling probability allocation, these methods assign priorities to different groups based on proxy signals such as average advantage (Wang et al., 2025) or learning progress (Portelas et al., 2020), and adjust their sampling probabilities accordingly. By selectively sampling training data, curriculum-based strategies can substantially improve data utilization and training efficiency over uniform sampling.

However, existing curriculum methods primarily assess a sample’s learning utility, i.e., the contribution of a sample to model learning, using predefined data properties (e.g., difficulty (Qi et al., 2025)) or historical performance signals (e.g., absolute advantage (Chen et al., 2025)). They often overlook that the utility is policy-dependent and continuously evolves as policy changes. This issue manifests itself both in sample grouping and in group scheduling. First, existing methods usually group training samples according to predefined properties, such as task categories (Teh et al., 2017) or success rate (Xi et al., 2024), rather than the optimization effects they induce under a given policy. However, samples with similar properties do not necessarily induce similar optimization effects, as shown in Figure 1(b). Consequently, groups constructed using such metrics may contain samples with heterogeneous optimization effects, thereby weakening the consistency of each group and reducing the effectiveness of group-level scheduling. Second, the sampling probabilities at the current training step are determined by simply aggregating the historical observations of a single proxy metric (e.g., average advantages). Simple aggrega tion reduces rich historical observations to a single scalar, making it difficult to capture their evolving patterns and potentially rendering the estimate outdated as the policy evolves. Worse still, the reliance on a single proxy metric neglects the rich dynamics of variables such as policy parameters or gradients, and thus provides a deficient characterization of the current policy state. Consequently, such esti mation methods may be insufficient to capture the temporal evolution of learning signals and their de pendence on the policy state, thereby limiting the effectiveness of curriculum scheduling.

In this paper, we propose Predictive Gradientbased Bandit (PGB) Curriculum for RL-based LLM Post-training, a policy-state-informed curriculum learning framework that incorporates policydependent optimization information into both sample grouping and curriculum scheduling. Specifically, for group construction, PGB clusters samples based on their gradient representations rather than external metrics. The gradient representations provide an optimization-aware approximation of how samples influence parameter updates under the given policy, enabling PGB to identify samples with similar learning signals. For sampling probability allocation, we introduce the predictive multi-armed bandit (MAB). Instead of simply aggregating historical proxy signals to estimate learning utilities, PGB employs a lightweight learnable temporal model to predict these utilities by jointly modeling the evolution of external metrics and intrinsic policy states during optimization. Based on the predicted utilities, PGB constructs a Boltzmann sampling distribution that dynamically prioritizes high-utility groups. By incorporating the policy’s internal states during sample grouping and curriculum scheduling, PGB better captures the evolving relationship between training samples and the state of the policy, leading to more effective curriculum scheduling.

Extensive experiments demonstrate that PGB consistently improves the training efficiency and final performance of RL. Compared with the uniform sampling strategy (Shao et al., 2024), PGB achieves substantial improvements on both indomain (ranging from 8.8% to 14.3%) and out-ofdomain datasets (ranging from 14.7% to 16.3%). PGB also substantially improves the training efficiency by achieving competitive performance with fewer steps, reaching the final performance of GRPO earlier during training. Moreover, compared with existing curriculum learning methods, PGB achieves substantial performance gains (ranging from 4.9% to 9.2%), further demonstrating the effectiveness of our approach. Overall, these results indicate that PGB enables more effective data utilization during training by leveraging the intrinsic states of the model, leading to significant improvements to current RL-based post-training methods.

## 2 Related Work

## 2.1 Reinforcement Learning for LLM Post-Training

Reinforcement learning (Guo et al., 2025; Lambert et al., 2025), using algorithms such as PPO (Schulman et al., 2017) and GRPO (Shao et al., 2024), is widely used for LLM post-training. Its effectiveness depends on data allocation, as samples vary in difficulty and learning utility (Schaul et al., 2016). Curriculum learning improves training efficiency by adaptively prioritizing useful samples.

## 2.2 Curriculum Learning

Curriculum learning (Bengio et al., 2009) aims to improve training efficiency and final performance by controlling the order or distribution of training data during model optimization. In reinforcement learning, curriculum design generally involves two complementary components: sample grouping and curriculum scheduling.

According to the granularity at which curriculum decisions are made, existing methods can be broadly categorized into task-based and samplebased approaches. Task-based curriculum reinforcement learning (CRL) (Huang et al., 2022) treats tasks or task configurations as scheduling units and controls their learning order (Lin et al., 2025). Representative methods include Reverse Curriculum Generation (Florensa et al., 2017), TSCL (Matiisen et al., 2019), and ALP GMM (Portelas et al., 2020), which select or generate tasks according to difficulty or learning progress. Recent works have extended this idea to LLM agents and reasoning tasks, such as R3 (Xi et al., 2024) and WebRL (Qi et al., 2025). Com pared with task-level scheduling, sample-based approaches allow more flexible control over the training distribution by composing batches from different groups. Representative sample-based approaches include Prioritized Experience Replay (PER) (Schaul et al., 2016), Experience Re play Optimization (ERO) (Zha et al., 2019), and CHER (Fang et al., 2019), which prioritize training samples according to estimated learning value or experience utility. Several sample-based methods formulate curriculum scheduling as a multi-armed bandit problem, where predefined data groups are treated as arms and selected according to estimated learning utility. SEC (Chen et al., 2025) models curriculum selection under a non-stationary bandit framework using absolute advantage as the learning signal, while DUMP (Wang et al., 2025) employs a UCB-based scheduler with recent statistics to allo cate sampling budgets across data groups. Gradient information also supports training-data influence estimation and subset selection. Influence functions (Koh and Liang, 2017) and TracIn (Pruthi et al., 2020) estimate how training examples affect model predictions, while CRAIG (Mirzasoleiman et al., 2020) and GRAD-MATCH (Killamsetty et al., 2021) select weighted subsets that approxi mate target gradients. These methods characterize sample influence or gradient representativeness, but do not explicitly predict the future learning utility of curriculum groups during RL post-training.

Existing bandit-based curricula typically operate on predefined categories or data distributions, whose samples may induce heterogeneous optimization effects. Although their utility estimates adapt to training feedback, they generally rely on current or recent proxy statistics without explicitly modeling temporal patterns and evolving policy states. Consequently, these estimates may lag behind changes in learning utility during nonstationary RL training.

## 3 Method

In this section, we present the PGB framework, which incorporates model-internal optimization information into both data organization and curriculum scheduling during training. The framework is designed to address two key limitations of existing methods, i.e., the reliance on external properties for data grouping and the use of retrospective signals for curriculum scheduling. The overall framework consists of two components: (i) gradient-based learning unit discovery and (ii) predictive multiarmed bandit CRL.

## 3.1 Preliminaries

We begin by motivating the gradient-based representation used for learning unit discovery. Inspired by the Neural Tangent Kernel (NTK) theory (Jacot et al., 2018), we first establish the connection between sample-wise gradients and local optimization effects, which provides the foundation for grouping the training data in the first stage. For a training sample $( x _ { i } , y _ { i } )$ , we represent its local update signal using the gradient $g _ { i }$ of its training loss:

$$
g _ { i } = \nabla _ { \theta } \ell _ { i } ( \theta ) .\tag{1}
$$

Consider a one-step gradient update induced by sample i:

$$
\theta ^ { \prime } = \theta - \eta g _ { i } .\tag{2}
$$

By first-order Taylor expansion, the loss change of another sample j is approximated as

$$
\ell _ { j } ( \theta ^ { \prime } ) - \ell _ { j } ( \theta ) \approx - \eta g _ { j } ^ { \top } g _ { i } .\tag{3}
$$

Therefore, $g _ { j } ^ { \top } g _ { i }$ characterizes how an update induced by sample i locally affects the loss of sample j. Since the raw inner product combines gradient magnitude and direction, and the magnitude may reflect sample-specific scale, we normalize the gradients and focus on their directional alignment when constructing learning units.

## 3.2 Gradient-based Learning Unit Discovery

Based on the local gradient analysis above, we group training samples with similar gradient directions into learning units (Caeiro et al., 2014). We use supervised fine-tuning (SFT) gradients computed on reference trajectories under the initial policy as a proxy for sample-level update directions, enabling gradient-based grouping without additional on-policy rollouts.

![](images/8b0fdda36ce58811efa23973b4a45bf36e2896a453fb5bdc3c6e5e88d37316c8.jpg)

Stage 2: Predictive Bandit CRL  
![](images/f8ecbb67f4e111b86e5d572181c577146161bd12d62dd2661c96c01287fbbf4b.jpg)  
Figure 2: Overview of our proposed Predictive Gradient-based Bandit (PGB) Curriculum method. PGB first discovers gradient-based learning units according to optimization effects and then performs curriculum scheduling by predicting learning utility from evolving policy states within a multi-armed bandit framework.

Concretely, for each training sample $( x _ { i } , y _ { i } )$ , we compute its gradient representation using SFT loss:

$$
\mathcal { L } _ { \mathrm { S F T } } ( x _ { i } , y _ { i } ) = - \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \log p _ { \theta } ( y _ { t } \mid x _ { i } , y _ { < t } ) ,\tag{4}
$$

where $p _ { \theta }$ denotes the conditional distribution defined by the policy model, $T$ is the number of response tokens in the reference trajectory, x is the input prompt, and $y _ { i }$ denotes the corresponding reference trajectory provided by the dataset or generated by a teacher model. The gradient representation of each sample is then defined as:

$$
g _ { i } = \nabla _ { \theta } \mathcal { L } _ { \mathrm { S F T } } ( x _ { i } , y _ { i } ) ,\tag{5}
$$

which serves as an optimization-aware representation of the local update signal. Since $g _ { i }$ contains both magnitude and directional information, we apply $L _ { 2 }$ normalization to isolate the direction of the induced model update:

$$
{ \bar { g } } _ { i } = { \frac { g _ { i } } { \| g _ { i } \| _ { 2 } } }\tag{6}
$$

In principle, learning units can be constructed by directly clustering the normalized gradient representations according to their directional similarity. However, operating in the full gradient space is computationally expensive for large language models. We therefore apply a Gaussian random projection (GRP), which approximately preserves pairwise distances and similarities in a lower-dimensional space according to the Johnson– Lindenstrauss lemma (Johnson et al., 1984). Specifically, we sample

$$
\begin{array} { c } { \mathbf { R } \in \mathbb { R } ^ { d \times D } , } \\ { \mathbf { R } _ { a b } \sim \mathcal { N } \left( 0 , \frac { 1 } { d } \right) , } \end{array}\tag{7}
$$

and compute the projected representation as $z _ { i } =$ $\mathbf { R } \bar { g } _ { i }$ . According to the properties of GRP, we have

$$
\begin{array} { r } { \mathbb { E } [ \langle z _ { i } , z _ { j } \rangle ] = \mathbb { E } [ \langle \mathbf { R } \bar { g } _ { i } , \mathbf { R } \bar { g } _ { j } \rangle ] = \langle \bar { g } _ { i } , \bar { g } _ { j } \rangle , } \end{array}\tag{8}
$$

which indicates that the projected inner product is an unbiased estimator of the cosine similarity between the original gradients. A detailed theoretical justification is provided in the Appendix. Therefore, rather than clustering in the original gradient space, we apply spherical K-means in the projected space to construct the learning units:

$$
\begin{array} { c } { { c _ { i } = \mathrm { A s s i g n } ( z _ { i } ) \in \{ 1 , \ldots , K \} , } } \\ { { u _ { k } = \{ ( x _ { i } , y _ { i } ) \in \mathcal { D } \mid c _ { i } = k \} . } } \end{array}\tag{9}
$$

We compute the sample-wise gradients using the initial policy before RL training and construct the learning units only once. The fixed units provide a stable partition based on the initial optimization relationships among samples and preserve consistent arm identities throughout training. Their stagedependent learning utilities are then dynamically modeled by the predictive scheduler.

## 3.3 Predictive Bandit Curriculum Learning

After constructing the learning units, the remaining challenge is to allocate the training budget according to their evolving learning utility during optimization. Since the utility of each unit is policydependent and changes as the policy evolves, directly relying on current or historical observations may provide inaccurate estimates of future utility. Therefore, curriculum scheduling can be formulated as a sequential decision-making problem. We model each learning unit as an arm in a multi-armed bandit framework and predict its future utility to guide adaptive sampling.

Specifically, given the set of learning units U obtained by clustering,

$$
\mathcal { U } = \{ u _ { 1 } , u _ { 2 } , . . . , u _ { K } \}\tag{10}
$$

we formulate each unit $u _ { k } \in \mathcal { U }$ as an arm in the bandit. Unlike the classical single-pull bandit settings, which select a single arm at each decision step, we consider a probabilistic multi-arm allocation setting where each training iteration distributes a fixed sample budget across multiple arms. During training, each arm receives a reward signal that quantifies the contribution of its corresponding learning unit. Following prior work (Xiong et al., 2025), we refer to this contribution as the learning utility of the unit. Drawing on existing curriculum learning methods (Chen et al., 2025; Wang et al., 2025), we use observable reward signals to characterize this utility from two perspectives: (1) the average absolute advantage, serving as a proxy for the strength of the learning signal

$$
\mathcal { R } _ { k } ^ { t } = \frac { 1 } { | B _ { k } ^ { t } | } \sum _ { x _ { i } \in B _ { k } ^ { t } } \left| A _ { i } ^ { t } \right| ,\tag{11}
$$

and (2) reward progress, which captures the temporal changes in model performance

$$
\mathcal { P } _ { k } ^ { t } = { r } _ { k } ^ { t } - { r } _ { k } ^ { t - 1 }\tag{12}
$$

Here, $B _ { k } ^ { t }$ is the batch sampled from $u _ { k } , \left| A _ { i } ^ { t } \right|$ is the absolute advantage of $x _ { i }$ , and $r _ { k } ^ { t }$ denotes the average rollout reward.

Existing bandit-based curriculum methods (Chen et al., 2025; Wang et al., 2025) estimate unit utility by aggregating historical proxy signals, without explicitly modeling its evolution with the policy state. Such estimates may lag behind changing learning dynamics and yield unstable sampling decisions (Figure 4). To address this issue, we explicitly model and predict the future utility of each learning unit, enabling more informed curriculum scheduling decisions. Specifically, we represent the learning state of each unit $u _ { k }$ at step t as

$$
\mathbf { s } _ { k } ^ { t } = \left[ \mathcal { R } _ { k } ^ { t } , \mathcal { P } _ { k } ^ { t } , \mathcal { N } _ { k } ^ { t } , \mathbf { G } _ { k } ^ { t } \right]\tag{13}
$$

where $\mathcal { R } _ { k } ^ { t }$ is the average absolute advantage, $\mathcal { P } _ { k } ^ { t }$ is the reward progress, $\mathcal { N } _ { k } ^ { t }$ is the number of samples drawn at the current step, and $\mathbf { G } _ { k } ^ { t }$ is the average gradient feature during RL optimization. Unlike the gradients used for unit construction, $G _ { k } ^ { t }$ records compact last-layer gradient statistics to characterize the evolving optimization state. For units that are not sampled, we set $\mathcal { N } _ { k } ^ { t } = 0$ , and retain $\mathcal { P } _ { k } ^ { t }$ and $\mathcal { R } _ { k } ^ { t }$ from the previous step. Instead of estimating future utilities solely from historical rewards, we use the historical learning state trajectories of each unit to predict its future utility. For each learning unit $u _ { k }$ we maintain a sequence of learning states

$$
\mathcal { H } _ { k } ^ { t } = \{ \mathbf s _ { k } ^ { t - L + 1 } , \dots , \mathbf s _ { k } ^ { t } \} ,\tag{14}
$$

and the future utility $\hat { R } _ { k } ^ { t + 1 }$ is predicted by

$$
\left[ \hat { R } _ { 1 } ^ { t + 1 } , \ldots , \hat { R } _ { K } ^ { t + 1 } \right] ^ { \top } = f _ { \phi } \left( \mathcal { H } _ { 1 } ^ { t } , \ldots , \mathcal { H } _ { K } ^ { t } \right) ,\tag{15}
$$

Here, we parameterize $f _ { \phi }$ as a lightweight learnable predictor consisting of a Gated Recurrent Unit (GRU) followed by a Multi-Layer Perceptron (MLP). It jointly processes the historical states of all K learning units to capture temporal dynamics and cross-unit correlations, and predicts their utilities at the next training step. The predictor operates on compact learning-state representations rather than full model parameters or sample-level gradients. The predicted utilities $\hat { R } _ { k } ^ { t + 1 }$ are converted into allocation probabilities through a Boltzmann policy.

$$
n _ { k } ^ { t + 1 } = b \pi _ { k } ^ { t + 1 } = \frac { b \cdot \exp { \left( \hat { R } _ { k } ^ { t + 1 } / \tau \right) } } { \sum _ { j = 1 } ^ { K } \exp { \left( \hat { R } _ { j } ^ { t + 1 } / \tau \right) } }\tag{16}
$$

where $\tau$ is the temperature hyperparameter controlling the exploration–exploitation trade-off and b is the batch size.

Overall, our framework follows a two-step training procedure. Step 1: Warm up. The warm-up step serves two purposes: ensuring sufficient exploration of all learning units and collecting initial state–utility trajectories for predictor training.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td>ID</td><td colspan="7">OOD</td></tr><tr><td>DAPO-Math</td><td>AIME 24</td><td>AIME 25</td><td>AIME 26</td><td>Mean AIME</td><td>MATH-500</td><td>AMC</td><td>Mean</td></tr><tr><td rowspan="4">Qwen3-1.7B</td><td>Original</td><td>0.121</td><td>0.000</td><td>0.067</td><td>0.033</td><td>0.033</td><td>0.282</td><td>0.150</td><td>0.104</td></tr><tr><td>GRPO</td><td>0.559</td><td>0.233</td><td>0.233</td><td>0.200</td><td>0.222</td><td>0.770</td><td></td><td>0.7000.426</td></tr><tr><td>Difficulty-UCB</td><td>0.565</td><td>0.267</td><td>0.233</td><td>0.233</td><td>0.242</td><td>0.742</td><td>0.800</td><td>0.454</td></tr><tr><td>PGB (Ours)</td><td>0.608</td><td>0.367</td><td>0.300</td><td>0.300</td><td>0.322</td><td>0.710</td><td>0.800</td><td>0.495</td></tr><tr><td rowspan="4">Qwen3-4B</td><td>Original</td><td>0.245</td><td>0.100</td><td>0.067</td><td>0.100</td><td>0.089</td><td>0.692</td><td>0.425</td><td>0.277</td></tr><tr><td>GRPO</td><td>0.709</td><td>0.500</td><td>0.400</td><td>0.400</td><td>0.433</td><td>0.850</td><td></td><td>0.825 0.595</td></tr><tr><td>Difficulty-UCB</td><td>0.803</td><td>0.567</td><td>0.533</td><td>0.400</td><td>0.499</td><td>0.846</td><td>0.875</td><td>0.644</td></tr><tr><td>PGB (Ours)</td><td>0.811</td><td>0.533</td><td>0.533</td><td>0.533</td><td>0.533</td><td>0.852</td><td>0.975</td><td>0.683</td></tr><tr><td rowspan="4">Qwen3-8B</td><td>Original</td><td>0.204</td><td>0.100</td><td>0.067</td><td>0.133</td><td>0.099</td><td>0.414</td><td></td><td>0.3000.202</td></tr><tr><td>GRPO</td><td>0.682</td><td>0.467</td><td>0.400</td><td>0.467</td><td>0.445</td><td>0.870</td><td></td><td>0.8500.611</td></tr><tr><td>Difficulty-UCB</td><td>0.769</td><td>0.633</td><td>0.533</td><td>0.433</td><td>0.531</td><td>0.844</td><td>0.900</td><td>0.667</td></tr><tr><td>PGB (Ours)</td><td>0.774</td><td>0.733</td><td>0.400</td><td>0.600</td><td>0.577</td><td>0.872</td><td>0.900</td><td>0.700</td></tr></table>

Table 1: Accuracy comparison on ID and OOD benchmarks across different model scales. Our PGB consistently achieves the best or comparable results across settings compared with standard GRPO and difficulty-UCB curriculum baselines, demonstrating strong generalization and scalability.

During this step, training batches are formed by uniformly sampling from the learning units. After each reinforcement learning update, we record the learning state and observed utility of each sampled unit, thereby constructing the initial training sequences for the predictor. The predictor is initialized through supervised training on the state–utility trajectories by minimizing the mean squared error between predicted and observed utilities:

$$
\mathcal { L } _ { \mathrm { p r e d } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left( \hat { R } _ { k } ^ { t + 1 } - R _ { k } ^ { t + 1 } \right) ^ { 2 } .\tag{17}
$$

Step 2: Predictive curriculum RL. After the warm-up step, the trained predictor is used to predict the future learning utility of each unit at the next training step. Based on the predicted utilities, PGB computes the Boltzmann sampling distribution in (16) and samples data from the units to construct the training batches for reinforcement learning. During training, newly collected learning states are continuously added to the history trajectories, and the predictor is periodically updated according to (17) every $T _ { p r e d }$ training iterations to adapt to the evolving learning dynamics. The training procedure is shown in Algorithm 1.

## 4 Experiments

This section evaluates the effectiveness of PGB for RL-based LLM post-training across multiple model scales and both in-distribution and out-ofdistribution benchmarks. We further conduct ablation studies to analyze the contribution of each component, including gradient-based learning unit construction and predictive utility modeling.

## 4.1 Experimental Setup

We conduct RL-based post-training experiments on the DAPO-Math-17K dataset (Yu et al., 2025). We randomly hold out 1,000 samples for in-domain evaluation and use the remaining samples for training. We evaluate our method on three Qwen3 model scales (Yang et al., 2025), including Qwen3- 1.7B, Qwen3-4B-Thinking, and Qwen3-8B. The evaluation benchmarks include DAPO-Math for in-distribution evaluation and AIME 2024–2026, MATH-500 (Hendrycks et al., 2021), and AMC for out-of-distribution evaluation (Shao et al., 2024; Lewkowycz et al., 2022), covering a range of mathematical reasoning difficulties.

We use Group Relative Policy Optimization (GRPO) (Shao et al., 2024) as the base RL algorithm and compare PGB with the following baselines: (1) GRPO, which uniformly samples training prompts; and (2) Difficulty-UCB, which reproduces the DUMP curriculum strategy (Wang et al., 2025). Specifically, it partitions training samples into groups according to the initial policy’s response accuracy and dynamically allocates the sampling budget among these groups using a UCBbased scheduler. We reimplement this method within our training framework. All methods share the same GRPO settings, training budget, and evaluation protocol for fair comparison. Detailed settings are provided in the Appendix.

## 4.2 Main Results

Table 1 shows the main results on both indistribution (ID) and out-of-distribution (OOD) benchmarks across Qwen3-1.7B, 4B, and 8B. In general, PGB consistently outperforms standard GRPO and difficulty-UCB curriculum baselines on the ID benchmark and in mean OOD performance across all three model variants, supporting the overall effectiveness of our proposed predictive curriculum learning framework.

![](images/3f9ac4a2269eeb425aa516c2f051632abefe689a0aaf019ef71417c2fbab53f4.jpg)

![](images/f10b866765ddac616a9c3eee0cde0e193606c617d7e0b37c9c37ad45318bd386.jpg)  
Figure 3: Training curves of mean reward for PGB + GRPO and GRPO on Qwen3-4B and 8B models.

In ID evaluation (DAPO-Math), PGB achieves consistent improvements over GRPO across all model sizes, achieving relative performance gains of 8.8% to 14.3%. These results show that PGB improves the performance of in-domain reasoning across model scales, consistent with its objective of adapting data allocation to the evolving policy state. PGB also achieves substantial gains on OOD benchmarks, with relative improvements of 14.7%– 16.3%. The gains are particularly pronounced on AIME 2024–2026, where PGB substantially outperforms standard GRPO and consistently improves the mean AIME score across all model scales, indicating improved average performance on challenging mathematical reasoning problems. PGB also achieves the best or comparable performance on Math-500 and AMC, suggesting that its benefits extend across multiple mathematical reasoning benchmarks. Notably, it consistently surpasses difficulty-UCB curriculum methods by 4.9%–9.2%. Since both methods dynamically allocate the training budget across data groups, the results show that the overall PGB design performs better than the combination of difficulty-based grouping and UCB-based scheduling. The contributions of learning unit construction and predictive scheduling are examined separately in the following ablations.

Overall, these results demonstrate the effectiveness of combining optimization-aware learning unit construction with predictive curriculum scheduling, leading to consistent improvements across model scales and benchmarks.

<table><tr><td>Steps</td><td>4B-PGB</td><td>4B-GRPO</td><td>8B-PGB</td><td>8B-GRPO</td></tr><tr><td>100</td><td>0.741</td><td>0.628</td><td>0.699</td><td>0.621</td></tr><tr><td>200</td><td>0.807</td><td>0.645</td><td>0.748</td><td>0.645</td></tr><tr><td>300</td><td>0.791</td><td>0.695</td><td>0.762</td><td>0.669</td></tr><tr><td>400</td><td>0.815</td><td>0.677</td><td>0.771</td><td>0.664</td></tr><tr><td>500</td><td>0.811</td><td>0.709</td><td>0.771</td><td>0.682</td></tr></table>

Table 2: Accuracy over steps on the DAPO dataset for PGB and standard GRPO on Qwen3-4B and 8B models.
<table><tr><td>Method DAPO</td><td>AIME MATH-500 AMC</td></tr><tr><td>GRPO 0.709 0.433</td><td>0.850 0.825</td></tr><tr><td>Difficulty 0.800</td><td>0.499 0.842 0.875</td></tr><tr><td>Rubric 0.807</td><td>0.433 0.846 0.900</td></tr><tr><td>Gradient 0.811</td><td>0.533 0.852 0.975</td></tr></table>

Table 3: Comparison of different unit construction strategies across math benchmarks on Qwen3-4B.

## 4.3 Ablation Studies

## 4.3.1 Training Efficiency Comparison

To evaluate the training-step efficiency of PGB relative to standard GRPO, we evaluate PGB and standard GRPO by tracking the mean training reward over optimization steps and the accuracy of intermediate checkpoints on the DAPO test set, as shown in Figure 3 and Table 2. First, as shown in Figure 3, PGB maintains generally higher training rewards than GRPO over most of the training process for both Qwen3-4B and Qwen3-8B models. More importantly, PGB reaches the peak performance of the GRPO baseline at a much earlier training stage (approximately within 100–200 steps), suggesting faster empirical convergence. Second, from the step-wise evaluation in Table 2, PGB consistently outperforms GRPO at every training stage across both model scales. At only 100 training steps, PGB already matches or exceeds the accuracy achieved by standard GRPO at the end of training, further demonstrating improved sample efficiency. Finally, the performance gap between PGB and GRPO is maintained throughout training, suggesting that PGB not only accelerates convergence but also achieves higher performance under the same steps. Overall, these results show that PGB substantially improves training efficiency by enabling faster convergence and more effective utilization of training steps.

## 4.3.2 Effect of Learning Unit Construction

To investigate the impact of different learning unit construction methods on model performance, we conduct comparative experiments with the following methods varying only the method used to construct learning units: (1) GRPO, which performs standard RL training with uniform sampling without constructing learning units; (2) Difficulty, which partitions training samples into learning units according to the model’s response accuracy; (3) Rubric, which constructs learning units based on predefined reasoning skill rubrics; and (4) Gradient, our proposed method. As shown in Table 3, different learning unit construction strategies have substantial impacts on model performance. Both difficulty- and rubric-based construction generally outperform standard GRPO, suggesting that group-based curriculum scheduling can improve over uniform sampling. Gradient-based grouping achieves the best overall performance, including gains in both DAPO-Math and AIME, indicating that optimization-aware grouping provides a better basis for curriculum construction. These results demonstrate that gradient features provide a more informative signal for uncovering the learning structure of training data than difficulty levels or reasoning rubrics. Consequently, gradient features provide a more effective basis for constructing learning units for curriculum scheduling.

<table><tr><td>Method</td><td>DAPO AIME</td><td>MATH-500 AMC</td></tr><tr><td>Boltzmann</td><td>0.570 0.242</td><td>0.702 0.800</td></tr><tr><td>UCB</td><td>0.572 0.222</td><td>0.706 0.775</td></tr><tr><td>Predictive</td><td>0.608 0.322</td><td>0.710 0.800</td></tr></table>

Table 4: Comparison of predictive and reward-based curriculum scheduling strategies on reasoning benchmarks on Qwen3-1.7B.  
![](images/023bcd6ab6b1db13bed4423c99a99a270d17ae8a11cba6577ee98e3cbdf4929f.jpg)

(a) Sampling probability dynamics of a learning unit under predictive and UCB-based scheduling.  
![](images/c5f6f1eba3fdcdeb43a742fd81f06eeccdfc5539c3502d28a22e2b208d9a8fc8.jpg)  
(b) Comparison of different future utility prediction strategies.  
Figure 4: Analysis of predictive curriculum scheduling.

## 4.3.3 Effect of Predictive Scheduling

We compare predictive scheduling with an SECstyle scheduler (Chen et al., 2025), which applies Boltzmann sampling to EMA estimates of absolute advantage, and a UCB scheduler. All strategies use the same gradient-based learning units to isolate the effect of scheduling. As shown in Table 4, predictive scheduling consistently achieves the best performance in all the benchmarks evaluated, supporting the effectiveness of incorporating future utility estimation into curriculum scheduling. To better understand this improvement, Figure 4(a) visualizes the sampling probability trajectories under predictive and UCB-based scheduling. UCB exhibits frequent fluctuations in allocation probabilities, suggesting its sensitivity to changing reward signals, whereas predictive scheduling produces smoother trajectories. Figure 4(b) compares different utility prediction strategies. The GRU-based predictor achieves lower prediction errors than all baselines compared after the initial stage, suggesting that modeling the evolving utility of learning units provides more accurate utility estimates.

## 4.3.4 Generalization to Science Reasoning

While our main experiments focus on mathematical reasoning tasks, we further evaluate the generalization ability of PGB on SciKnowEval (Feng et al., 2025), covering Chemistry, Biology and Physics questions. We compare PGB with the original model and standard GRPO to investigate whether predictive curriculum scheduling remains effective beyond math reasoning scenarios. As shown in Table 7, PGB consistently improves the performance across different science reasoning domains. These results provide evidence that PGB can extend beyond mathematical reasoning.

## 5 Conclusion

We present PGB, a model-aware predictive curriculum framework for RL-based LLM post-training. PGB combines an optimization-aware partition constructed under the initial policy with predictive modeling of how unit utility evolves during RL training. Experiments across multiple variants of the Qwen3 model and mathematical reasoning benchmarks show that PGB achieves stronger overall ID and OOD performance over standard GRPO and Difficulty-UCB. Further analyses support gradient-based grouping and show that utility prediction improves estimation accuracy and smooths sampling allocation. These results highlight the benefits of integrating model-internal optimization signals for data organization with predictive modeling of evolving learning utility for RL post-training.

## Limitations

Our evaluation is limited to the Qwen3 model family and focuses primarily on mathematical reasoning. To partially mitigate this limitation, we evaluate three Qwen3 model scales and include science reasoning benchmarks; generalization to other model families remains untested. Due to the computational cost of RL post-training, we conduct one training run per configuration. To make these single-run comparisons more controlled, we use the same GRPO settings, training budget, and evaluation protocol across methods; variability across runs remains unmeasured.

## References

Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. 2009. Curriculum learning. In Proceedings ofthe 26th Annual International Conference on Machine Learning, ICML ’09, page 41–48, New York, NY, USA. Association for Computing Machinery.

Manuel Caeiro, Martín Llamas, and Luis Anido. 2014. Poeml: Modeling learning units through perspectives. Computer Standards & Interfaces, 36(2):380–396.

Daniel Campos. 2021. Curriculum learning for language modeling. Preprint, arXiv:2108.02170.

Xiaoyin Chen, Jiarui Lu, Minsu Kim, Dinghuai Zhang, Jian Tang, Alexandre Piché, Nicolas Gontier, Yoshua Bengio, and Ehsan Kamalloo. 2025. Selfevolving curriculum for llm reasoning. Preprint, arXiv:2505.14970.

Meng Fang, Tianyi Zhou, Yali Du, Lei Han, and Zhengyou Zhang. 2019. Curriculum-guided hindsight experience replay. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc.

Kehua Feng, Xinyi Shen, Weijie Wang, Xiang Zhuang, Yuqi Tang, Qiang Zhang, and Keyan Ding. 2025. Sciknoweval: Evaluating multi-level scientific knowledge of large language models. Preprint, arXiv:2406.09098.

Carlos Florensa, David Held, Markus Wulfmeier, and Pieter Abbeel. 2017. Reverse curriculum generation for reinforcement learning. CoRR, abs/1707.05300.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, and 175 others. 2025. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. Preprint, arXiv:2103.03874.

Hanchi Huang, Deheng Ye, Li Shen, and Wei Liu. 2022. Curriculum-based asymmetric multi-task reinforcement learning. Preprint, arXiv:2211.03352.

Arthur Jacot, Franck Gabriel, and Clément Hongler. 2018. Neural tangent kernel: Convergence and generalization in neural networks. Advances in neural information processing systems, 31.

William B Johnson, Joram Lindenstrauss, et al. 1984. Extensions of lipschitz mappings into a hilbert space. Contemporary mathematics, 26(189-206):1.

Krishnateja Killamsetty, Sivasubramanian Durga, Ganesh Ramakrishnan, Abir De, and Rishabh Iyer. 2021. Grad-match: Gradient matching based data subset selection for efficient deep model training. In International Conference on Machine Learning, pages 5464–5474. PMLR.

Pang Wei Koh and Percy Liang. 2017. Understanding black-box predictions via influence functions. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1885–1894. PMLR.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, and 4 others. 2025. Tulu 3: Pushing frontiers in open language model post-training. Preprint, arXiv:2411.15124.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. 2022. Solving quantitative reasoning problems with language models. Preprint, arXiv:2206.14858.

Siyu Lin, Qingwei Mi, and Tianhan Gao. 2025. A survey of curriculum learning in deep reinforcement learning. In 2025 IEEE 15th Annual Computing and Communication Workshop and Conference (CCWC), pages 01141–01147.

Tambet Matiisen, Avital Oliver, Taco Cohen, and John Schulman. 2019. Teacher–student curriculum learning. IEEE transactions on neural networks and learning systems, 31(9):3732–3740.

Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. 2020. Coresets for data-efficient training of machine learning models. In International Conference on Machine Learning, pages 6950–6960. PMLR.

OpenAI. 2024. Learning to reason with llms. https://openai.com/zh-Hans-CN/index/ learning-to-reason-with-llms/. Accessed: 2026-06-24.

Rémy Portelas, Cédric Colas, Katja Hofmann, and Pierre-Yves Oudeyer. 2020. Teacher algorithms for curriculum learning of deep rl in continuously parameterized environments. In Conference on Robot Learning, pages 835–853. PMLR.

Garima Pruthi, Frederick Liu, Mukund Sundararajan, and Satyen Kale. 2020. Estimating training data influence by tracing gradient descent. Preprint, arXiv:2002.08484.

Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Jiadai Sun, Xinyue Yang, Yu Yang, Shuntian Yao, Wei Xu, et al. 2025. Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning. In International Conference on Learning Representations, volume 2025, pages 79791–79821.

Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. 2016. Prioritized experience replay. Preprint, arXiv:1511.05952.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. CoRR, abs/1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Yee Whye Teh, Victor Bapst, Wojciech Marian Czarnecki, John Quan, James Kirkpatrick, Raia Hadsell, Nicolas Heess, and Razvan Pascanu. 2017. Distral: Robust multitask reinforcement learning. Preprint, arXiv:1707.04175.

Zhenting Wang, Guofeng Cui, Yu-Jhe Li, Kun Wan, and Wentian Zhao. 2025. Dump: Automated distribution-level curriculum learning for rl-based llm post-training. Preprint, arXiv:2504.09710.

Zhiheng Xi, Wenxiang Chen, Boyang Hong, Senjie Jin, Rui Zheng, Wei He, Yiwen Ding, Shichun Liu, Xin Guo, Junzhe Wang, et al. 2024. Training large language models for reasoning through reverse curriculum reinforcement learning. arXiv preprint arXiv:2402.05808.

Feng Xiong, Hongling Xu, Yifei Wang, Runxi Cheng, Yong Wang, and Xiangxiang Chu. 2025. HS-STaR: Hierarchical sampling for self-taught reasoners via difficulty estimation and budget reallocation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 5539– 5555, Suzhou, China. Association for Computational Linguistics.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, and 16 others. 2025. Dapo: An open-source llm reinforcement learning system at scale. Preprint, arXiv:2503.14476.

Daochen Zha, Kwei-Herng Lai, Kaixiong Zhou, and Xia Hu. 2019. Experience replay optimization. Preprint, arXiv:1906.08387.

## A Predictive Bandit CRL Algorithm

Algorithm 1 Predictive Multi-armed Bandit CRL   
Input: Dataset $\mathcal { D } ,$ learning units $\begin{array} { r l } { \mathbf { \overline { { \mathbf { \Lambda } } } } ^ { } \overline { { \mathbf { \Lambda } } } ^ { } } & { { } \overline { { \mathbf { \Lambda } } } ^ { } } \end{array}$   
$\{ u _ { 1 } , \dotsc , u _ { K } \}$ , policy θ, warm-up steps $T _ { \mathrm { w a r m } } , { \mathrm { p r e } } -$   
dictor update interval $T _ { p r e d }$   
Output: Trained policy parameters $\theta$   
1: Initialize predictor $f _ { \phi }$   
2: Stage 1: Warm-up   
3: for $t = 1 , \dots , T _ { \mathrm { w a r m } }$ do   
4: Uniformly sample training batches from   
$\{ u _ { i } \}$   
5: Update policy $\theta$ using RL optimization.   
6: Collect learning states $\mathcal { H } _ { k } = \mathcal { H } _ { k } \cup \{ \mathbf { s } _ { k } ^ { t } \}$   
7: end for   
8: Train predictor $f _ { \phi }$ using $\mathcal { H } _ { k } ^ { t }$   
9: Stage 2: Predictive Sampling   
10: for $t = T _ { \mathrm { w a r m } } + 1 , \ldots , T$ do   
11: Predict future utility of all learning units:   
$\{ \hat { R } _ { k } ^ { t + 1 } \} _ { k = 1 } ^ { K } = { \bf \bar { f } } _ { \phi } ( \mathcal { H } _ { 1 } ^ { t } , \dots , \mathcal { H } _ { K } ^ { \bar { t } } )$   
12: Compute sampling distribution $\pi ^ { t + 1 }$   
13: Sample training batch according to $\pi ^ { t + 1 }$   
14: Update policy θ using RL optimization.   
15: Update $f _ { \phi }$ every $T _ { p r e d }$ steps.   
16: end for   
17: return $\theta$

## B Theoretical Explanation

## B.1 First-Order Interaction between Sample Gradients

For training samples $( x _ { i } , y _ { i } )$ and $( x _ { j } , y _ { j } )$ , we

Assume that $\ell _ { j } ( \theta )$ is twice differentiable and its Hessian is bounded by $\| \nabla _ { \theta } ^ { 2 } \ell _ { j } ( \theta ) \| _ { 2 } \le H$ . For the update $\theta ^ { \prime } = \theta - \eta g _ { i }$ , let $\Delta \theta = \theta ^ { \prime } - \theta = - \eta g _ { i }$ Taylor’s theorem gives

$$
\begin{array} { r l } & { \ell _ { j } ( { \boldsymbol { \theta } } ^ { \prime } ) = \ell _ { j } ( { \boldsymbol { \theta } } ) + \nabla _ { { \boldsymbol { \theta } } } \ell _ { j } ( { \boldsymbol { \theta } } ) ^ { \top } \Delta { \boldsymbol { \theta } } + \displaystyle \frac { 1 } { 2 } \Delta { \boldsymbol { \theta } } ^ { \top } \nabla _ { { \boldsymbol { \theta } } } ^ { 2 } \ell _ { j } ( \widetilde { { \boldsymbol { \theta } } } ) \Delta { \boldsymbol { \theta } } } \\ & { \qquad = \ell _ { j } ( { \boldsymbol { \theta } } ) - \eta g _ { j } ^ { \top } g _ { i } + \displaystyle \frac { \eta ^ { 2 } } { 2 } g _ { i } ^ { \top } \nabla _ { { \boldsymbol { \theta } } } ^ { 2 } \ell _ { j } ( \widetilde { { \boldsymbol { \theta } } } ) g _ { i } , } \end{array}\tag{18}
$$

Rearranging and taking absolute values, we obtain

$$
\Big | \ell _ { j } ( { \boldsymbol { \theta } } ^ { \prime } ) - \ell _ { j } ( { \boldsymbol { \theta } } ) + \eta g _ { j } ^ { \top } g _ { i } \Big | = \frac { \eta ^ { 2 } } { 2 } \left| g _ { i } ^ { \top } \nabla _ { { \boldsymbol { \theta } } } ^ { 2 } \ell _ { j } ( \tilde { { \boldsymbol { \theta } } } ) g _ { i } \right| .\tag{19}
$$

Using the quadratic-form bound

$$
| x ^ { \top } A x | \leq \| A \| _ { 2 } \| x \| _ { 2 } ^ { 2 }\tag{20}
$$

and the Hessian bound

$$
\left\| \nabla _ { \theta } ^ { 2 } \ell _ { j } ( \tilde { \theta } ) \right\| _ { 2 } \leq H ,\tag{21}
$$

we have

$$
\left| \ell _ { j } ( \theta ^ { \prime } ) - \ell _ { j } ( \theta ) + \eta g _ { j } ^ { \top } g _ { i } \right| \leq \frac { \eta ^ { 2 } H } { 2 } \| g _ { i } \| _ { 2 } ^ { 2 } .\tag{22}
$$

## B.2 Properties of Gradient Random Projection

To reduce the computational cost of clustering high-dimensional sample gradients, we apply a random projection before clustering. According to the Johnson-Lindenstrauss Lemma (Johnson et al., 1984), given $0 ~ < ~ \varepsilon ~ < ~ 1$ , a set $X ~ \subset ~ \mathbb { R } ^ { D }$ and a number $\begin{array} { r } { d \ > \ \Omega ( \frac { \log D } { \varepsilon ^ { 2 } } ) } \end{array}$ , there is a linear map $f : \mathbb { R } ^ { D }  \mathbb { R } ^ { d }$ such that

$$
( 1 - \varepsilon ) \| u - v \| ^ { 2 } \leqslant \| f ( u ) - f ( v ) \| ^ { 2 } \leqslant ( 1 + \varepsilon ) \| u - v \| ^ { 2 }\tag{23}
$$

for all $u , v \in X$ . Specifically, given a normalized gradient representation $\bar { g } _ { i } \in \mathbb { R } ^ { \bar { D } }$ , we project it into a lower-dimensional space using a random matrix $R \in \mathbb { R } ^ { d \times D }$

$$
z _ { i } = R \bar { g } _ { i } ,\tag{24}
$$

where each entry of $R$ is independently sampled from a Gaussian distribution:

$$
R _ { m n } \sim \mathcal { N } ( 0 , \frac { 1 } { d } ) .\tag{25}
$$

The projection approximately preserves pairwise distances and inner products according to the Johnson–Lindenstrauss lemma. Here, we show that the projected inner product is an unbiased estimator of the original gradient similarity.

For two normalized gradient representations $\bar { g } _ { i }$ and $\bar { g } _ { j }$ , their projected similarity is:

$$
\langle z _ { i } , z _ { j } \rangle = \langle R \bar { g } _ { i } , R \bar { g } _ { j } \rangle = \bar { g } _ { i } ^ { \top } R ^ { \top } R \bar { g } _ { j } .\tag{26}
$$

Since each element of R follows $\textstyle { \mathcal { N } } ( 0 , { \frac { 1 } { d } } )$ , we have:

$$
\mathbb { E } [ R ^ { \top } R ] = I _ { D } .\tag{27}
$$

Therefore,

$$
\begin{array} { r l } & { \mathbb { E } [ \langle z _ { i } , z _ { j } \rangle ] = \mathbb { E } [ \bar { g } _ { i } ^ { \top } R ^ { \top } R \bar { g } _ { j } ] } \\ & { \quad \quad \quad = \bar { g } _ { i } ^ { \top } \mathbb { E } [ R ^ { \top } R ] \bar { g } _ { j } } \\ & { \quad \quad \quad = \bar { g } _ { i } ^ { \top } \bar { g } _ { j } . } \end{array}\tag{28}
$$

Thus, the expected inner product between projected gradients is equal to the original gradient similarity. With sufficiently large projection dimension $d ,$ the Johnson–Lindenstrauss lemma further guarantees that pairwise distances are preserved with high probability:

$$
( 1 - \epsilon ) \| \bar { g } _ { i } - \bar { g } _ { j } \| _ { 2 } ^ { 2 } \leq \| z _ { i } - z _ { j } \| _ { 2 } ^ { 2 } \leq ( 1 + \epsilon ) \| \bar { g } _ { i } - \bar { g } _ { j } \| _ { 2 } ^ { 2 } ,\tag{29}
$$

where ϵ controls the distortion introduced by the random projection. Therefore, random projection preserves the relative geometry of sample gradients while substantially reducing the dimensionality, allowing efficient clustering of optimization-aware learning units.

## C Implementation Details

This section provides implementation details for reproducing our experiments, including the model and dataset configurations, learning unit construction, predictor architecture, curriculum scheduling, and RL hyperparameters. Unless otherwise specified, all experiments follow the same training protocol.

We implement PGB using the VERL reinforcement learning framework with GRPO. We train Qwen3-1.7B, Qwen3-4B-Thinking, and Qwen3- 8B on DAPO-Math-17K. For evaluation, AIME 2024, 2025, and 2026 each contain 30 problems, while AMC and MATH-500 contain 40 and 500 problems, respectively. The training data comprise English mathematical reasoning problems from the English subset of DAPO-Math-17K. The maximum prompt and response lengths are 2,048 and 8,192 tokens, respectively. Each model is trained for 500 steps with a global batch size of 32. In the ablation studies, we vary only the component under investigation while keeping the remaining configurations fixed.

To construct learning units, we use Qwen3-30B to generate reference responses for the DAPO-Math-17K training prompts. We compute samplelevel SFT gradients under the initial policy, normalize them, and project them to $d = 1 0 2 4$ dimensions before partitioning the training data into K = 8 learning units using spherical K-means. We then treat the units as arms in a probabilistic bandit. Our curriculum scheduler uses a GRU followed by an MLP to predict each unit’s next-step mean absolute advantage from historical learning states. Reward progress, sample counts, and gradient statistics serve as contextual features. Training begins with a 20-step warm-up stage using uniform sampling. Subsequently, the predictor is updated every 10 training steps, and its predictions guide sampling with a Boltzmann temperature of 0.4. At each predictor update, we perform 5 optimization iterations with a learning rate of $1 \times 1 0 ^ { - 3 } .$ The GRU and MLP hidden dimensions are 64 and 32, respectively, and the history sequence length is 10.

<table><tr><td>T</td><td>DAPO AIME MATH-500 AMC</td></tr><tr><td>0.2 0.739 0.489</td><td>0.854 0.825</td></tr><tr><td>0.3 0.740</td><td>0.411 0.860 0.875</td></tr><tr><td>0.4 0.761 0.489</td><td>0.862 0.95</td></tr><tr><td>0.5 0.745</td><td>0.489 0.860 0.95</td></tr><tr><td>0.6 0.734 0.500</td><td>0.860 0.9</td></tr></table>

Table 5: Comparison of different temperature parameter τ across math reasoning benchmarks on Qwen3-4B.
<table><tr><td>Group</td><td>Grad Norm Std</td><td>Loss Mean Loss Std</td></tr><tr><td>0</td><td>0.567 0.731</td><td>0.118</td></tr><tr><td>1</td><td>0.657 0.838</td><td>0.109</td></tr><tr><td>2</td><td>0.520 0.861</td><td>0.125</td></tr><tr><td>3</td><td>0.496 0.772</td><td>0.116</td></tr><tr><td>4</td><td>0.576 0.707</td><td>0.148</td></tr><tr><td>5</td><td>0.567 0.729</td><td>0.129</td></tr><tr><td>6</td><td>0.907 0.778</td><td>0.155</td></tr><tr><td>7</td><td>1.100 0.795</td><td>0.100</td></tr></table>

Table 6: Optimization statistics of task-based groups.

For GRPO optimization, the actor learning rate is $1 \times 1 0 ^ { - 6 }$ . We use dynamic batch sizing and gradient checkpointing. Rollouts are generated with vLLM using tensor parallelism of 4 and 8 responses per prompt. All experiments use FSDP-based distributed training on a single node with 8 NVIDIA H20 GPUs (141 GB per GPU). Due to the computational cost of RL post-training, we report one independent run per configuration.

## C.1 Gradient-Based Unit Construction

Although the policy evolves during RL, we construct learning units only once using the initial policy. The resulting units serve as stable data partitions, while their stage-dependent usefulness is captured by the predictive scheduler. Re-clustering throughout training would introduce considerable overhead and change the arm identity, making temporal utility modeling difficult.

## D Experiments

## D.1 Ablation Study on the Temperature Parameter τ

We conduct an ablation study to investigate the effect of the temperature parameter τ , which controls the exploration-exploitation trade-off in the Boltzmann allocation strategy. The results in Table 5 indicate that PGB is robust to different temperature settings, while extreme values lead to slightly worse performance. A small τ makes the sampling distribution more concentrated, which may reduce exploration and limit the discovery of useful learning units. In contrast, a large τ increases exploration but weakens the exploitation of highutility units. Overall, $\tau = 0 . 4$ achieves the best performance on most benchmarks, obtaining the highest scores on DAPO-Math, MATH-500, and AMC. Therefore, we use $\tau = 0 . 4$ as the default setting.

(a) Rubric-based Groups  
![](images/0d93935602b71a937e44b0a03bb1a220cad681d1e238d6f87634a5005a102e13.jpg)

(b) Gradient-based Units  
![](images/1b898034c955c27d4c236bb5cb1878d4e9df22cb0991e735a08763d2511ac64b.jpg)  
Figure 5: Visualization of task-based groups and gradient-based learning units in gradient space. Samples from the same rubric category are widely distributed and overlap with other categories, while gradient-based clustering discovers more coherent optimization structures.

## D.2 Gradient Analysis of Predefined Task Groups

To investigate whether task-based grouping reflects the intrinsic optimization behaviors of training samples, we conduct an analysis on the DAPO-MATH dataset. Specifically, we categorize the dataset into eight task groups according to rubric annotations and uniformly sample 1,000 training examples. For each sample, we compute its gradient representation with respect to the model parameters and analyze the gradient characteristics within each task group.

As shown in Figure 5, samples from the same task are widely distributed in the gradient space and substantially overlap with samples from other categories. This indicates that task-level semantic categories do not form coherent clusters according to model-induced optimization behaviors. In contrast, clustering samples based on gradient representations produces more compact and distinguishable groups, suggesting that gradient similarity better captures the intrinsic optimization structure of training data.

![](images/309fc6d09b4f26cd5950a1107862b6909e14cadb0ef37af51793a484cc9cd2c5.jpg)  
Figure 6: Comparison of future utility prediction strategies.

<table><tr><td>Method</td><td>Chemistry Biology</td><td>Physics</td><td></td></tr><tr><td>Original</td><td>0.443</td><td>0.240</td><td>0.500</td></tr><tr><td>GRPO</td><td>0.505</td><td>0.260</td><td>0.525</td></tr><tr><td>PGB</td><td>0.524</td><td>0.300</td><td>0.538</td></tr></table>

Table 7: Performance comparison on science reasoning tasks from SciKnowEval (Feng et al., 2025) with Qwen3-1.7B.

We further quantify the gradient characteristics of task-based groups in Table 6. The standard deviation of gradient norms varies across groups, with several groups (e.g., Group 6 and 7) exhibiting large internal gradient variance (0.907 and 1.100, respectively). These statistics indicate substantial within-group variation in gradient magnitudes, complementing the gradient-space visualization.

These observations demonstrate that predefined task fail to characterize optimization-consistent sample groups. Therefore, we construct learning units based on gradient similarity to obtain optimization-aware curriculum structures.

## D.3 Ablation Study on Utility Prediction Strategies

To evaluate the effectiveness of the proposed utility prediction module, we compare four strategies for estimating future utility:

1. GRU+MLP Predictor, models the temporal evolution of utility by taking historical utility states as input and predicting the utility at the next training step.

2. Sliding Window Average, estimates future utility using the average utility over recent steps.

![](images/c3024fcfcb5860691e13659811ad4cacaf3619f245ee12d08388e0806dc84fbc.jpg)

![](images/6beeac9148c87322cc5ae086e3c5d1fce80efb8350bc8a1765822dec1b191b4e.jpg)

Sample Probablity of Unit-2 for Qwen3-8B  
![](images/6643b7555335810e2efff13d17134938b81478fef6a703d195030ae585aeacd2.jpg)

![](images/e38ef913d05c26af29d3f2f8500af58ba8bde47dc16aa27d08b8291bd73b509d.jpg)

Sample Probablity of Unit-4 for Qwen3-8B  
![](images/7ec7476724ac3e9bc91dd31e139bf5371b40101f51aa4a4039a0f28de106219d.jpg)  
Figure 7: Sampling probability dynamics of learning unit 0-4 under predictive and UCB-based curriculum scheduling.

3. Linear Regression method, fits a linear function over historical utility values to extrapolate future trends.

4. Last Step method, directly uses the most recent utility value as the prediction.

We train Qwen3-4B-Thinking on the DAPO-Math dataset using DAPO for 200 steps, including the warm-up stage, and measure prediction accuracy using the average relative error of four future utility prediction strategies throughout the training process:

$$
\mathrm { R E } ^ { t } = \frac { 1 } { b } \sum _ { k = 1 } ^ { b } \frac { \Big | \hat { R } _ { k } ^ { t } - R _ { k } ^ { t } \Big | } { \Big | R _ { k } ^ { t } \Big | + \epsilon } ,\tag{30}
$$

where ϵ is a small constant for numerical stability. The results are shown in Figure 6. The GRU+MLP Predictor initially shows higher errors due to the limited historical trajectories available at the early stage. As more training data are collected, its prediction error decreases continuously and becomes lower than all other approaches. We further summarize the prediction errors using the mean and median values. GRU+MLP Predictor achieves the lowest average and median errors among all strategies, with a mean error of 0.1698 and a median error of 0.1475. In comparison, the best-performing baseline obtains a mean error of 0.2000 and a median error of 0.1946. These results demonstrate that explicitly modeling temporal utility dynamics provides more accurate future utility estimation for curriculum scheduling.

Figure 7 further extends the sampling-dynamics visualization presented in the main text by reporting the allocation trajectories for learning units. Across units, predictive scheduling generally produces smoother probability trajectories than UCBbased scheduling, while preserving unit-specific changes in sampling priorities throughout training.

## D.4 Generalization to Science Reasoning Tasks

While our main experiments focus on mathematical reasoning tasks, we further evaluate the generalization ability of PGB on science reasoning tasks. Specifically, we conduct experiments on Sci-KnowEval (Feng et al., 2025), covering Chemistry, Biology and Physics questions. We compare PGB with the original model and standard GRPO to investigate whether predictive curriculum scheduling remains effective beyond math reasoning scenarios. As shown in Table 7, PGB consistently improves the performance across different science reasoning domains. For example, on the chemistry benchmark, PGB achieves an accuracy of 52.4%, outperforming standard GRPO by 1.9 percentage points. Similar improvements are observed on biology and physics tasks. These results provide evidence that PGB can extend beyond mathematical reasoning.