# TRAINING SPECIALIST LLMS ON YOUR PHONE: FORWARD-ONLYOPTIMIZATION VIA BLOCK-SPECTRAL COORDINATES

Liyan Tan <sup>\*</sup> <sup>1</sup> Yequan Zhao <sup>\*</sup> <sup>1</sup> Ziyue Liu <sup>2</sup> Yupeng Su <sup>2</sup> Ruijie Zhang <sup>2</sup> Zheng Zhang <sup>1</sup>

## ABSTRACT

Large language models (LLMs) on personal devices keep user data local, but limited memory restricts their general capability. Since most user-specific tasks do not require the breadth of frontier models, a practical solution is to adapt a shared on-device model into lightweight specialists. However, backpropagation is prohibitive in memory, and users rarely provide labeled data. We present MobileSpec, a system that trains specialists with forward passes alone and learns from automatically checked task outcomes, as in reinforcement learning with verifiable rewards (RLVR). Because forward-only updates are built from random perturbations, training is efficient only when these perturbations resemble the weight changes of RLVR, which largely preserve the singular values of pretrained weights and rotate their singular vectors. The core algorithm of MobileSpec, FuRA-S, produces such perturbations while training only the block-spectral coordinates of pretrained weights, namely the singular values of their column blocks. As we show theoretically and empirically, rescaling these blocks independently shifts their relative contributions to each weight matrix and thereby rotates its singular vectors. Since this update can be rewritten as a transformation of the layer input, FuRA-S also adapts INT4 models without modifying their weights. With specialists of only 1–5 MB that update 0.015–0.05% of the parameters, MobileSpec raises the success rate of a phone-use agent from 6–23% up to 89–100% and trains a personalized 8B model on an iPhone 16 Pro overnight. On RLVR benchmarks, FuRA-S is competitive with full-parameter evolution strategies (ES) and outperforms ES over LoRA adapters and quantized ES on INT4 models.

## 1 INTRODUCTION

Large language models (LLMs) are increasingly deployed as autonomous agents on personal devices, from desktop environments (Xie et al., 2024) to mobile interfaces (Wang et al., 2024). With the rapid progress of small language models (Liu et al., 2024) and the dedicated inference accelerators (NPUs) in recent smartphones, running such agents entirely on the device has become practical. These agents keep user data local and work offline.

However, the limited memory of mobile devices restricts on-device models to a few billion parameters, leaving their general capability well below that of frontier models. In practice, users rarely need such breadth, since the tasks they perform on the device are highly user-specific. This motivates a deployment paradigm in which a shared base model, referred to as the generalist, is adapted locally into lightweight specialists. Each specialist is a small set of parameters trained for one of these tasks.

![](images/90d329e7f0f2cd55348a599c004974d4ea89e6c29b81bb453cdf91f4aaab6d81.jpg)  
Figure 1. Memory needed to train on a phone. Serving the model alone occupies most of the memory available to an app, so training with backpropagation or with full-parameter ES exceeds it, whereas FuRA-S adds only 1–5 MB of trainable parameters.

Training specialists on the device, however, imposes constraints that standard post-training pipelines do not satisfy. Since labeled data are scarce, the model should learn from signals that arise during use. These signals include automatic checks of task outcomes, as used in reinforcement learning with verifiable rewards (RLVR; Lambert et al., 2025), and the records that users accumulate on the device. Moreover, mobile NPUs may not afford backpropagation, which consumes several times the memory of inference (Malladi et al., 2023) and is poorly supported by the quantized dataflow of on-device engines (Zhao et al., 2025). Forward-only methods, including zeroth-order (ZO) optimization (Malladi et al., 2023; Zhao et al., 2025) and evolution strategies (ES; Qiu et al., 2026; Sarkar et al., 2026), avoid these costs by estimating updates from the losses or rewards of randomly perturbed models. Yet they still perturb a large number of model parameters and maintain a full-precision copy of the weights, which exceeds the memory available to an app (Figure 1) and makes every specialist as large as the model itself. This raises a natural question: can a phone still train its own specialists from the signals that arise during use, with forward passes alone?

In this work, we investigate this question and present MobileSpec, an on-device training system that realizes the generalist–specialist paradigm (Figure 2). Its specialists share one INT4 generalist and differ only in files of 1–5 MB, so switching tasks leaves the generalist and its kernels unchanged. When the phone is idle, MobileSpec trains each specialist with forward passes, measuring the fitness of the perturbed models by automatic checks of task outcomes or by the likelihood of the user’s records. For long-horizon phone-use tasks, we further design a step-wise fitness function that credits each action by comparing it with the actions that other perturbed models take on the same screen.

How efficiently MobileSpec learns depends on which parameters its specialists change. Parameter-efficient finetuning (PEFT) methods such as LoRA (Hu et al., 2022) are a natural choice, but they were designed for backpropagation. Backpropagation computes the exact gradient for any parameterization, whereas forward-only optimization estimates the update as a weighted sum of random perturbations. Learning is therefore efficient only when these perturbations resemble the weight changes that RLVR makes. Recent work characterizes these changes as spectral inheritance: RLVR largely preserves the singular values of pretrained weights (Zhu et al., 2026a;b) and adapts the model mainly by rotating their singular vectors (Zhu et al., 2026a). The generic low-rank perturbations of LoRA are not designed to produce such rotations. The parameters that existing methods train are thus either too large to store for every task or not aligned with the weight changes of RLVR.

At the core of MobileSpec is FuRA-S (Full-Rank Adaptation–Spectra), an extremely parameter-efficient method for forward-only training. FuRA-S trains only the block-spectral coordinates of pretrained weights, namely the singular values of their column blocks under an untruncated singular value decomposition (SVD). Yet its ES perturbations change the weights in the way RLVR does. Theoretically, FuRA-S achieves this because rescaling the singular values of each block independently shifts the relative contributions of the blocks to the assembled matrix and thereby rotates its singular vectors rather than changing its singular values. Empirically, what FuRA-S learns comes almost entirely from these rotations: the resulting singular vectors combined with the pretrained singular values retain over 95% of the accuracy gain. The untruncated factorization also lets FuRA-S adapt the INT4 generalist without modifying its weights, since the update can be rewritten exactly as a transformation of the layer input.

## Our contributions are summarized as follows:

• We design MobileSpec, an on-device training system that realizes the generalist–specialist paradigm. Its specialists of 1–5 MB share one INT4 generalist, and its step-wise fitness function credits the individual actions of long-horizon phone-use tasks.

• We propose FuRA-S, the core algorithm of Mobile-Spec, which trains only the block-spectral coordinates of pretrained weights, about 0.03% of the parameters of Qwen2.5-1.5B, and adapts released INT4 models without modifying their weights.

• We show through a first-order analysis that FuRA-S changes pretrained weights in the way RLVR does, rotating their singular vectors while largely preserving their singular values, and confirm empirically that the weights it trains exhibit spectral inheritance.

• We conduct experiments on phone-use, personalization, reward-based, supervised, and agentic tasks. Mobile-Spec raises the success rate of a phone-use agent from 6–23% up to 89–100%. On RLVR benchmarks, FuRA-S is competitive with full-parameter ES, consistently outperforms ES over LoRA adapters (LoRA-ES), and surpasses quantized evolution strategies (QES) by up to 16.25 points.

## 2 RELATED WORK

Mobile agents and on-device training. LLM agents operate phones through their screens, either by locating interface elements with visual perception tools (Wang et al., 2024) or, as in foundation GUI agents such as MAI-UI (Zhou et al., 2025a), by acting directly on screenshots. Mobile-World (Kong et al., 2026) evaluates them on long tasks that often span several apps. Online reinforcement learning trains them in many parallel environments (Zhou et al., 2025a; Tang et al., 2026), and GiGPO (Feng et al., 2025) credits individual steps by grouping the actions that different trajectories take from the same state. To fine-tune models on the phone itself, memory-efficient backpropagation reduces the memory of the backward pass (Song & Tang, 2025; Park et al., 2026) or adapts it to device constraints through sharding and energy-aware scheduling (Geng et al., 2025), whereas PocketLLM (Peng et al., 2024) and MobiZO (Gao et al., 2025) avoid it with forward-only optimization. For personalization, OPPU (Tan et al., 2024) trains a separate LoRA adapter for each user.

![](images/bb43eececc30235afc8604c92031080625cb9fa7f10052a8c11408f22a20c994.jpg)  
Figure 2. On-device training of a specialist with MobileSpec. When the generalist agent fails at a task that the user repeats, MobileSpec trains a specialist for it on the phone overnight. Perturbed models attempt the task in sandboxed apps, and forward-only ES moves the block singular values toward those of the better perturbed models (Section 3.2). The resulting specialist is a 2 MB vector δ that the serving engine applies on top of the shared INT4 generalist.

Forward-only optimization. MeZO (Malladi et al., 2023) fine-tunes LLMs with the memory footprint of inference, and later ZO methods refine its gradient estimate with lowrank, sparse, or per-example perturbations (LOZO (Chen et al., 2025), Sparse-MeZO (Liu et al., 2025a), GRZO (Tan et al., 2026)) or lower the precision of its forward passes (QuZO (Zhou et al., 2025b)). ES at Scale (Qiu et al., 2026) applies full-parameter ES to RLVR of billion-parameter LLMs, and other work extends ES to agentic tasks (Zheng et al., 2026), quantized weights (QES (Xu et al., 2026)), and the singular values of LoRA adapters (Korotyshova et al., 2026), or scales it to large populations with low-rank perturbations (EGGROLL (Sarkar et al., 2026)).

Parameter-efficient fine-tuning. LoRA (Hu et al., 2022) learns low-rank updates, and VeRA (Kopiczko et al., 2024) and Uni-LoRA (Li et al., 2025) reduce its parameters further through shared random projections. Spectral methods initialize LoRA from the principal or minor singular components of pretrained weights (Meng et al., 2024; Wang et al., 2025), tune only the singular values (SVF (Sun et al., 2022; 2025), SVFit (Sun et al., 2024)), or learn sparse combinations of singular vectors (Lingam et al., 2024). In RLVR, SVD-informed initializations suffer from spectral collapse, and extreme parameter reduction limits reasoning capacity (Yin et al., 2026). FuRA (Zhao et al., 2026) factors each column block of a weight matrix by an untruncated SVD and trains the block singular values together with the right factor of each block, which enables full-rank updates.

## 3 THE MOBILESPEC SYSTEM

MobileSpec realizes the generalist–specialist paradigm by keeping one INT4 generalist on the device and training a specialist for each of the user’s tasks with forward passes alone while the phone is idle (Figure 2). Its training algorithm, FuRA-S, changes only the block singular values of the generalist and runs directly on its INT4 weights.

## 3.1 Workflow

The generalist comprises the INT4 weights and frozen block factors that are computed from them once on a server and distributed with the model (Section 3.3). Each specialist is a 1–5 MB file of relative changes to the block singular values.

Task and fitness. Each task needs a fitness function that the device can compute without labels. For a task that ends in a verifiable state, such as replying to a post, the perturbed models attempt it in sandboxed apps, and a checker inspects the final state of the app and returns a reward. For a task defined by the user’s own records, such as tagging movies, the fitness of a perturbed model is the likelihood that it assigns to these records.

Training. When the phone is idle, for example overnight, MobileSpec trains the specialist with FuRA-S (Section 3.2). Each iteration evaluates a population of perturbed models, called members, and moves the specialist toward the members with higher fitness, which for long-horizon tasks is the step-wise fitness of Section 3.4. These forward evaluations stream the decoder layers from flash, so the device never holds the entire model in memory.

Selection. MobileSpec periodically evaluates the specialist on held-out instances of the task, such as the user’s later records or new episodes. It retains the best-scoring checkpoint and stops training once the score no longer improves.

Serving. When the user runs a task, the device loads its specialist and applies it as a transformation of the input before each INT4 multiplication (Section 3.3). Switching to another task therefore replaces only the specialist, while the generalist and its kernels stay unchanged.

## 3.2 Training specialists with FuRA-S

FuRA-S trains each specialist by ES over the block singular values of the generalist (Figure 3a). They make up about 0.03% of the parameters of Qwen2.5-1.5B and form the search space, which is the set of parameters that ES perturbs. Although only these values change, their perturbations rotate the singular vectors of each weight matrix while barely changing its singular values, as RLVR does. Section 4 shows that this property lets ES find improving perturbations for longer and learn faster than in other search spaces.

Block-wise SVD reparameterization. Following FuRA (Zhao et al., 2026), we split $\mathbf { W } \in \mathbb { R } ^ { m \times n }$ into K column blocks of width b, $\mathbf { W } = [ \mathbf { \bar { W } } _ { 1 } \mid \dots \mid \mathbf { W } _ { K } ]$ , and factor each block by its untruncated SVD:

$$
\begin{array} { r } { { \bf W } _ { k } = { \bf L } _ { k } \mathrm { d i a g } ( { \bf s } _ { k } ) { \bf R } _ { k } ^ { \top } , } \end{array}\tag{1}
$$

where $\mathbf { L } _ { k } \in \mathbb { R } ^ { m \times b }$ has orthonormal columns, $\mathbf { R } _ { k } \in \mathbb { R } ^ { b \times b }$ is orthogonal, and $\mathbf { s } _ { k } \in \mathbb { R } ^ { b }$ holds the block singular values. Because every block keeps all of its singular values, the factorization reconstructs W exactly. The product $\mathbf { L } _ { k } \mathrm { d i a g } ( \mathbf { s } _ { k } )$ has the same size as $\mathbf { W } _ { k }$ and takes its place, so the only overhead in storage and computation comes from the $K$ factors ${ \bf R } _ { k }$ of size $b \times b \colon K b ^ { 2 } = n b$ values per matrix, or a fraction $b / m$ of W. Partitioning the columns as in FuRA (Appendix C) keeps this overhead at about 3%.

Update rule. ES updates the parameters from the fitness of perturbed models instead of gradients. At iteration $t ,$ it evaluates $N$ members $\pmb \theta _ { t } + \mu \mathbf z _ { j }$ with $\mathbf { z } _ { j } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ , obtains their fitness $F _ { j } ,$ such as a task reward or a negative loss, from forward passes, and updates $\begin{array} { r } { \pmb { \theta } _ { t + 1 } = \pmb { \theta } _ { t } + \frac { \alpha } { N } \sum _ { j } F _ { j } \mathbf { z } _ { j } } \end{array}$ where the step size α absorbs $1 / \mu$

FuRA-S changes the search space and how the perturbations are sampled and weighted. FuRA-S freezes $\mathbf { L } _ { k }$ and ${ \bf R } _ { k }$ and confines the trainable parameter set $\pmb { \theta }$ to the block singular values. The adapted matrix is

$$
\mathbf { W } ^ { \prime } = \big [ \mathbf { L } _ { 1 } \mathrm { d i a g } ( \mathbf { s } _ { 1 } ^ { \prime } ) \mathbf { R } _ { 1 } ^ { \top } ~ \big | ~ \cdots ~ \big | ~ \mathbf { L } _ { K } \mathrm { d i a g } ( \mathbf { s } _ { K } ^ { \prime } ) \mathbf { R } _ { K } ^ { \top } \big ] .\tag{2}
$$

Let $\pmb { \theta } _ { t } \in \mathbb { R } ^ { d }$ collect all block singular values at iteration $t ,$ initialized at their pretrained values $\theta _ { 0 }$ . The population consists of $N / 2$ antithetic pairs of perturbed values, ${ \pmb \theta } _ { t } \pmb \pm$ $\mathbf { D } \mathbf { z } _ { j }$ , with $\mathbf { z } _ { j } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } )$ and $\mathbf { D } ~ = ~ \mathrm { d i a g } ( \mu _ { \mathrm { r e l } } | \pmb { \theta } _ { 0 } | )$ , so that each singular value moves in proportion to its own size. As in MeZO (Malladi et al., 2023), the directions are regenerated from random seeds and never stored. After the $N$ members are scored, their fitness values are standardized as $\widetilde { F } _ { j } ^ { \pm } = ( F _ { j } ^ { \pm } - \bar { F } ) / s _ { F }$ , where $\bar { F }$ and $s _ { F }$ are the mean and estandard deviation of $\{ F _ { j } \}$ . The block singular values are updated as

$$
\pmb { \theta } _ { t + 1 } = \pmb { \theta } _ { t } + \frac { \alpha } { N } \mathbf { D } \sum _ { j = 1 } ^ { N / 2 } \big ( \widetilde { F } _ { j } ^ { + } - \widetilde { F } _ { j } ^ { - } \big ) \mathbf { z } _ { j } ,\tag{3}
$$

where α is the step size. Each direction is weighted by the fitness difference of its pair, so the update moves θ toward the perturbations with higher fitness (Figure 3b and Algorithm 1 in Appendix C). For likelihood fitness, FuRA-S can instead update the block singular values with Adam (Kingma & Ba, 2015) on the two-point gradient estimate of MeZO (Appendix E). Since neither update stores activations or gradients, training adds only about 20 MB to the memory of forward evaluation on Llama-3-8B, even with the Adam moments (Appendix D).

## 3.3 Serving specialists on the INT4 generalist

Phones run models released in quantized form, which store each weight matrix as $\mathbf { W } _ { q } = ( \mathbf { Q } - \mathbf { Z } ) \odot \mathbf { \Gamma }$ with INT4 codes $\mathbf { Q } ,$ , zero points Z, and scales Γ. Storing high-precision block-SVD factors next to these weights would offset the memory saved by quantization, and requantizing $\mathbf { W } ^ { \prime }$ after each iteration would accumulate quantization error. Mobile-Spec therefore rewrites the update of FuRA-S so that the quantized weight itself stays in the computation.

Write the adapted singular values as $\mathbf { s } _ { k } ^ { \prime } = \mathbf { s } _ { k } \odot \left( 1 + \delta _ { k } \right)$ Since $\mathbf { R } _ { k } ^ { \top } \mathbf { R } _ { k } = \mathbf { I } .$ , Eq. (1) yields $\mathbf { L } _ { k }$ di $\mathbf { \boldsymbol { \mathrm { a g } } } ( \mathbf { \boldsymbol { s } } _ { k } ) = \mathbf { \boldsymbol { \mathrm { W } } } _ { k } \mathbf { \boldsymbol { R } } _ { k }$ and substituting this product into Eq. (2) removes $\mathbf { L } _ { k } \mathbf { ; }$

$$
\mathbf { W } ^ { \prime } = \mathbf { W } ( \mathbf { I } + \mathbf { E } ) , \qquad \mathbf { E } = \mathbf { R } \mathrm { d i a g } ( \pmb { \delta } ) \mathbf { R } ^ { \top } ,\tag{4}
$$

where $\mathbf { R } = \mathrm { b l o c k d i a g } _ { k } ( \mathbf { R } _ { k } )$ and $\pmb { \delta }$ concatenates the $\delta _ { k }$ This rewrite involves no approximation, because no block singular value is truncated. For the INT4 generalist, R comes from the block SVD of its dequantized weights, and each adapted layer computes

$$
\begin{array} { r } { \mathbf { y } = \underbrace { \mathbf { W } _ { q } } _ { \mathrm { I N T 4 , f r o z e n } } \left( \mathbf { x } + \underbrace { \mathbf { R } } _ { \mathrm { f r o z e n } } \mathrm { d i a g } ( \underbrace { \delta } _ { \mathrm { t r a i n a b l e } } ) \mathbf { R } ^ { \top } \mathbf { x } \right) . } \end{array}\tag{5}
$$

![](images/4a6ad770a940c0f433f748dc7bd1e147f26a374980c146fafab75ef638eb3b57.jpg)  
Figure 3. FuRA-S. (a) Only the block singular values are trained, and the adapted block reuses the original, possibly INT4, weights. (b) FuRA-S restricts its perturbations to the region where more of them improve the model, whereas LoRA perturbs in unconstrained, less effective directions. (c) The trained singular vectors alone retain over 95% of the gain.

In Eq. (5), the large factor $\mathbf { L } _ { k }$ never appears, and the released $\mathbf { W } _ { q }$ keeps its integer codes, zero points, and scales, so it runs on the existing INT4 kernels. The frozen R, which the released checkpoint does not contain, is stored in 4 bits and shared by all specialists, so each specialist stores only its full-precision δ (Appendix C.3). Since the adaptation only adds R diag(δ)R<sup>⊤</sup>x to the input, $\delta = 0$ recovers the quantized model exactly, regardless of the precision of R.

## 3.4 Step-wise fitness for long-horizon tasks

In long-horizon tasks such as operating a phone app, one wrong action can decide the outcome of an episode, yet the agent receives only a single reward at its end. In Mobile-World (Kong et al., 2026), for example, an agent asked to reply to a Mastodon post often searches for the text of the post, which the server does not index, rather than for the account of its author, and then scrolls until it reaches the step limit. When ES scores each member by the rewards of its own episodes, a better action receives credit only if a perturbation replaces the wrong action with it and thereby alters the outcome of the member’s few episodes, in which the randomness of all other steps masks its effect.

The step-wise fitness instead scores each action against the whole population. Following GiGPO (Feng et al., 2025), each step t of an episode with H steps receives the episode reward discounted by $\gamma ^ { H - 1 - t }$ , with $\gamma \in ( 0 , 1 )$ , as its return. Steps with nearly identical screenshots form an anchor group $g ,$ in which the credit of step t is the advantage of its action, $\hat { Q } ( g _ { t } , a _ { t } ) - \hat { V } ( g _ { t } )$ . Here $\mathinner { \hat { V } \mathopen { \left( g \right) } }$ is the mean return of the steps in $^ { g , }$ and $\hat { Q } ( g , a )$ is that of the steps taking action $a .$ Unlike GiGPO, we pool the groups over all members and sum the credits of each member’s steps into its fitness,

$$
F _ { i } = \frac { 1 } { J } \sum _ { e = 1 } ^ { J } \sum _ { t \in e } \bigl [ \hat { Q } ( g _ { t } , a _ { t } ) - \hat { V } ( g _ { t } ) \bigr ] ,\tag{6}
$$

where the outer sum runs over the J episodes of member i. By the performance-difference lemma (Kakade & Langford, 2002), $F _ { i }$ approximates the return of member i relative to that of the average policy of the population, whose values $\hat { Q }$ and $\hat { V }$ estimate. The step-wise fitness thus ranks the members by return, as the episode rewards do, but assigns the credit to individual actions. This credit is nonzero only on screens where members act differently, since ${ \hat { Q } } ( g , a ) =$ $\hat { V } ( g )$ wherever all members take the same action. On these screens, $\hat { Q }$ pools the episodes of all members, so a member that takes the better action is rewarded even when its own episodes fail for other reasons. The fitness is computed entirely from the screenshots, actions, and rewards that the episodes already record, adding no forward passes or environment calls.

## 4 ANALYZING FURA-S THROUGH SPECTRAL INHERITANCE

This section explains why the search space of FuRA-S suits ES. In existing search spaces, including LoRA adapters and all parameters, ES soon stops finding perturbations that improve the model (Section 4.1). The reason lies in how RLVR changes pretrained weights: the trained weights exhibit spectral inheritance (Zhu et al., 2026a), and perturbations that respect it improve the model more often (Section 4.2). FuRA-S follows this structure with very few parameters, since rescaling the block singular values independently rotates the singular vectors of each weight matrix while barely changing its singular values (Section 4.3). The weights it trains also exhibit spectral inheritance (Section 4.4).

## 4.1 ES in existing search spaces

The search space determines both the size of a specialist and how quickly it improves. LoRA is the natural choice, since it is competitive with full fine-tuning under backpropagation (Schulman & Thinking Machines Lab, 2025). Under ES, however, its perturbations soon stop improving the model. We compare four search spaces on the arithmetic reasoning task Countdown with Qwen2.5-1.5B-Instruct, scaling the perturbations of each to the same KL divergence from the unperturbed model. From iteration 50 on, no perturbation of LoRA-ES or full-parameter ES improves on this model (Figure 4). A smaller search space does not help: Uni-LoRA (Li et al., 2025), which reduces the 4.6M trainable values of LoRA to 508,928, drifts furthest from the base policy and ends with the lowest accuracy. FuRA-S trains as many values but keeps finding improving perturbations through iteration 125 and reaches the highest accuracy. The directions of a search space therefore matter more than its size.

![](images/48a76a737af6571993126fb58a0f48265db7ab14c3848bfcc6c5cc0505ce33a8.jpg)  
Figure 4. ES on Countdown with Qwen2.5-1.5B-Instruct in four search spaces: (a) accuracy on 500 test problems, (b) fraction of the 30 perturbations per iteration that improve on the unperturbed model, and (c) KL divergence from the base policy per response token.

## 4.2 ES under spectral inheritance

The convergence rate of ES depends on the fraction of perturbations that improve on the unperturbed model. Once this fraction approaches zero, the fitness-weighted update carries little signal and learning slows. An effective search space must therefore keep this fraction positive.

Spectral inheritance, a property of the weight changes that RLVR makes (Zhu et al., 2026a), points to such a search space. Let $\mathbf { W } = \mathbf { U } \pmb { \Sigma } \mathbf { V } ^ { \top }$ be the SVD of a pretrained weight matrix $\mathbf { W } \in \mathbb { R } ^ { m \times n }$ , where $\Sigma = \operatorname { d i a g } ( \sigma _ { 1 } , \dots , \sigma _ { q } )$ holds the spectrum and $q = \operatorname* { m i n } ( m , n )$ . The left and right singular vectors, the orthonormal columns of $\mathbf { U } \in \mathbb { R } ^ { m \times q }$ and $\mathbf { V } \in \mathbb { R } ^ { n \times q }$ , form the output and input singular frames. The matrices $\mathbf { U } ^ { \prime } \pmb { \Sigma } \mathbf { V } ^ { \prime \top }$ with orthonormal $\mathbf { U } ^ { \prime }$ and $\mathbf { V } ^ { \prime }$ differ from W only in their frames and form its fixed-spectrum family (W). ISO (Zhu et al., 2026a) finds that RLVR barely moves the spectra of pretrained weights, whereas it rotates both their output and their input frames. Combining the trained frames with the pretrained spectrum retains the gain of RLVR, and optimizing only the frames at a fixed spectrum reaches the end-of-run accuracy of AdamW with 2.7 fewer training steps on Qwen3-8B-Base. The solutions that RLVR finds therefore lie near the fixed-spectrum family (W).

(a) Base model  
![](images/08d2547fd9882435795bd0b4f14375eb1a57c2961cf22923bf3090d02dd79ed2.jpg)

(b) Iteration 100  
![](images/9bfad8d52a9853682e1cf8c1fa0e74f5dbee3110a490d2f1b3eecead7f582aca.jpg)  
Figure 5. Fraction of perturbations that improve on the unperturbed model on Countdown, (a) around the base model and (b) around the FuRA-S model at iteration 100.

Perturbations that rotate the frames at a fixed spectrum should improve on the unperturbed model more often, even late in training. We test this on Countdown with Qwen2.5- 1.5B-Instruct over three seeds, comparing four kinds of perturbations matched in KL divergence, which change all weights (Full), rank-4 LoRA adapters (LoRA), only the frames at a fixed Σ (U/V), or only the global singular values in Σ (S). Around the base model, frame-only perturbations improve on it slightly more often than those of all weights, 74.0% compared with 70.8% (Figure 5a). Around the model of FuRA-S at iteration 100, 8.3% of them still do, compared with 1.0% for Full and none for LoRA (Figure 5b).

Observation 1: ES finds improving perturbations more often by rotating the frames of each weight matrix at a fixed spectrum, where the solutions of RLVR lie.

Rotating the frames through the SVD of each full weight matrix, however, is impractical on a phone. A specialist that changes both U and V would be larger than the model itself, and for INT4 weights these dense factors would also need high-precision storage. Smaller alternatives fall short as well: low-rank adapters initialized from a truncated SVD suffer from spectral collapse in RLVR (Yin et al., 2026), and perturbing only the global singular values, as in SVF (Sun et al., 2022), cannot rotate the frames. The block singular values that FuRA-S trains avoid these problems, and Section 4.3 shows that perturbing them still rotates the frames.

## 4.3 How block rescaling follows spectral inheritance

We now show that FuRA-S follows spectral inheritance: its perturbations rotate the frames of W while keeping $\mathbf { W } ^ { \prime }$ near (W). As in ISO (Zhu et al., 2026a), we quantify how much an update $\Delta \mathbf { W } = \mathbf { W } ^ { \prime } - \mathbf { W }$ changes the spectrum with two metrics (Appendix A). The spectral residual $\rho _ { \Sigma }$ is the distance from $\mathbf { W } ^ { \prime } \mathbf { \Sigma }  t o \mathcal { F } ( \mathbf { W } )$ relative to the update size, ranging from 0 for a pure rotation to 1 for a pure rescaling of singular values. The spectrum-changing energy κ is the fraction of the update energy that changes the singular values to first order, normalized to one for a random dense update. We measure frame rotation by the principal angles between the leading singular subspaces of W and $\mathbf { W } ^ { \prime }$

Frame rotation from block rescaling. By Eq. (4), rescaling each $\mathbf { s } _ { k }$ by $1 + \delta _ { k }$ gives $\mathbf { W } ^ { \prime } = \mathbf { W } ( \mathbf { I } + \mathbf { E } )$ with $\mathbf { E } = \mathbf { R } \mathrm { d i a g } ( \delta ) \mathbf { R } ^ { \top }$ . In the singular basis of W, the update decomposes as $\mathbf { U } ^ { \top } \Delta \mathbf { W } \mathbf { V } = \Sigma \mathbf { G }$ with $\mathbf { G } = \mathbf { V } ^ { \top } \mathbf { E } \mathbf { V } \colon$ the diagonal of G changes the spectrum, and its off-diagonal entries rotate both frames (Proposition 1, Appendix A). Rescaling the singular values of each block independently shifts the relative contributions of the blocks to the assembled matrix, producing off-diagonal entries in G and thereby rotating U and V. With a single block, the block SVD is the SVD of W, so R = V, G is diagonal, and the same rescaling changes only the spectrum. The rotation therefore arises because each global singular vector spans several blocks, whose singular values are rescaled differently.

Training only the block singular vectors, by contrast, is less effective at rotating the frames. Rotating ${ \bf R } _ { k }$ leaves U and Σ unchanged and rotates only V, whereas RLVR rotates both frames (Zhu et al., 2026a). Rotating $\mathbf { L } _ { k }$ within its column span limits how far U can move. Dropping the orthogonality of ${ \bf R } _ { k }$ rotates U as well but turns each ${ \bf R } _ { k }$ into a dense factor with b times as many values as $\mathbf { s } _ { k }$ , losing the parameter efficiency that on-device training requires.

Spectrum preservation. To first order, the relative change of each singular value is a weighted average of the relative changes of the block singular values (Proposition 1):

$$
\frac { \Delta \sigma _ { i } } { \sigma _ { i } } = \sum _ { j = 1 } ^ { n } w _ { i j } \delta _ { j } , \qquad w _ { i j } = ( \mathbf { r } _ { j } ^ { \top } \mathbf { v } _ { i } ) ^ { 2 } ,\tag{7}
$$

where $\mathbf { r } _ { j }$ are the columns of R and the weights $w _ { i j }$ are nonnegative and sum to one over j. Since ES perturbs the block singular values independently with zero mean, the positive and negative $\delta _ { j }$ from different blocks largely cancel in this average, and the spectrum barely moves.

Measured rotation. We verify both properties by applying random updates at the scale of one ES perturbation, $\| \Delta \mathbf { W } \| _ { F } = 0 . 0 3 \| \mathbf { W } \| _ { F }$ , to each block-SVD factor of 12 Qwen2.5-1.5B-Instruct matrices (Table A1 in Appendix B). All updates preserve the spectrum $( \rho _ { \Sigma } \leq 5 . 3 \% , \kappa \leq 7 . 5 )$ , far below the values ISO reports for supervised fine-tuning, but differ sharply in frame rotation. Rescaling $\mathbf { s } _ { k }$ rotates U and V by about $9 ^ { \circ }$ , four times as far as a dense update of W, whereas every block singular-vector update rotates them less despite $( b { - } 1 ) / 2$ to b times as many values. Rescaling the global singular values rotates neither frame (Figure 8a,b in Appendix B).

Takeaway 1: Perturbing only the block singular values preserves the spectrum ofeach weight matrix but rotates both of its frames four times as far as a dense update.

## 4.4 Spectral inheritance of the trained weights

Individual perturbations follow spectral inheritance (Section 4.3), but the accumulated update, a fitness-weighted sum of many such perturbations, need not preserve this structure. We verify that it does by decomposing the weights that FuRA-S trains on MATH (Qwen2.5-Math-7B) and Countdown (Qwen2.5-1.5B-Instruct) into rotated frames and a changed spectrum. The accuracy gain comes almost entirely from the rotated frames: combined with the pretrained spectrum, they retain 95.1% of it on MATH and 95.8% on Countdown (Figure 3c). Conversely, the trained spectrum with the pretrained frames retains only 15% of the gain on MATH and none on Countdown. The trained weights thus exhibit spectral inheritance, as RLVR-trained models do (Zhu et al., 2026a).

Takeaway 2: Accumulated over training, the updates of FuRA-S still follow spectral inheritance: the rotated frames alone retain over 95% of the accuracy gain.

## 5 ALGORITHM EVALUATION

We first evaluate FuRA-S on standard benchmarks, testing whether it matches full-parameter ES with a tiny fraction of the parameters (Section 5.1), outperforms ES over LoRA adapters and quantized weights (Sections 5.1 and 5.2), and remains effective beyond single-turn RLVR (Section 5.3).

Table 1. Countdown accuracy (%) on 2,000 test problems. The BP-based policy-gradient baselines are from ES at Scale, Full ES is our reproduction of ES at Scale (Appendix C.1), and LoRA-ES uses rank-4 adapters, and VeRA-ES uses rank-1024 VeRA adapters. Best per row in bold, and parentheses give the difference from Full ES.
<table><tr><td></td><td></td><td colspan="3">BP-based</td><td colspan="4">BP-free</td></tr><tr><td>Base model</td><td>Original</td><td>PPO</td><td>GRPO</td><td>Dr. GRPO</td><td>Full ES</td><td>LoRA-ES</td><td>VeRA-ES</td><td>FuRA-S (ours)</td></tr><tr><td>Qwen2.5-0.5B-Instruct</td><td>0.1</td><td>0.3</td><td>13.0</td><td>13.5</td><td>14.4</td><td>12.6</td><td>13.5</td><td>14.4 (0.0)</td></tr><tr><td>Llama-3.2-1B-Instruct</td><td>0.4</td><td>11.2</td><td>14.9</td><td>13.9</td><td>15.6</td><td>11.7</td><td>13.7</td><td>15.8 (+0.2)</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>0.7</td><td>14.2</td><td>27.8</td><td>31.0</td><td>36.7</td><td>34.7</td><td>28.1</td><td>37.0 (+0.3)</td></tr><tr><td>Qwen2.5-3B-Instruct</td><td>10.0</td><td>20.1</td><td>37.8</td><td>43.8</td><td>58.2</td><td>54.3</td><td>55.2</td><td>55.8 (−2.4)</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>31.2</td><td>55.1</td><td>57.0</td><td>57.5</td><td>58.9</td><td>55.5</td><td>54.2</td><td>56.2 (−2.7)</td></tr></table>

Table 2. Greedy pass@1 (%) of Qwen2.5-Math-7B at the final checkpoint, with FuRA-S minus Full ES in parentheses.
<table><tr><td>Method</td><td>AIME24</td><td>AMC</td><td>MATH500</td><td>Minerva</td><td>Olympiad</td><td>Avg-5</td><td>Avg-3</td></tr><tr><td>Full ES</td><td>26.7</td><td>57.8</td><td>77.6</td><td>26.8</td><td>37.0</td><td>45.2</td><td>47.2</td></tr><tr><td>LoRA-ES</td><td>16.7</td><td>53.0</td><td>74.0</td><td>22.8</td><td>35.4</td><td>40.4</td><td>44.1</td></tr><tr><td>FuRA-S (ours)</td><td>23.3</td><td>54.2</td><td>76.8</td><td>26.5</td><td>40.0</td><td>44.2 (-1.0)</td><td>47.8(+0.6)</td></tr></table>

## 5.1 RLVR on Countdown and MATH

Countdown and MATH are scored by automatic checks of the final answer, the first kind of fitness of Section 3.1. On both tasks, we compare FuRA-S with full-parameter ES and with LoRA-ES on rank-4 adapters, which uses the same number of members and per-matrix perturbation size as FuRA-S, so that only the search space differs. On Countdown, we also compare with the policy-gradient methods PPO (Schulman et al., 2017), GRPO (Shao et al., 2024), and Dr. GRPO (Liu et al., 2025b), which train all parameters by backpropagation (BP).

Countdown. Following ES at Scale (Qiu et al., 2026), we train five models from 0.5B to 7B parameters on 200 Countdown problems (Appendix C.1). FuRA-S is on par with full-parameter ES, matching or exceeding it on the three smaller models and staying within three points on the two larger ones (Table 1), although it trains at most 0.05% of the parameters that full-parameter ES perturbs. It also outperforms LoRA-ES on every model while training about nine times fewer parameters, which reflects the advantage of a search space that follows spectral inheritance (Section 4.3). VeRA-ES, which perturbs VeRA adapters (Kopiczko et al., 2024) with only 1.4 to 1.9 times as many trainable parameters as FuRA-S, falls below FuRA-S on every model as well, by up to 8.9 points on Qwen2.5-1.5B, so this advantage does not come from the smaller search space alone. Finally, without computing any gradient, FuRA-S surpasses all BP-based baselines on four of the five models, with the largest margin on Qwen2.5-3B.

MATH. We further train Qwen2.5-Math-7B under the protocol of ES at Scale and evaluate it on five mathematics benchmarks (Appendix C.2). The comparisons with ES carry over to this harder task (Table 2). FuRA-S slightly exceeds full-parameter ES on the average over the three larger benchmarks (Avg-3) and stays within one point on the average over all five (Avg-5), which also includes the small AIME24 and AMC sets. On both averages, it again outperforms LoRA-ES at the final checkpoint and at the best checkpoints by Avg-5 and by Avg-3 (Table A3).

## 5.2 RLVR on quantized models

Since phones run quantized models, we adapt the released GPTQ-INT4 (Frantar et al., 2023) checkpoints of Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct on Countdown while keeping their 4-bit weights unchanged. We compare FuRA-S with QES (Xu et al., 2026), which applies full-parameter ES to quantized weights, under its original protocol. Searching the block singular values instead of the quantized weights is more effective on both models. FuRA-S outperforms our reproduction of QES by 5.85 and 16.25 points after 300 iterations and exceeds the best accuracy of the entire QES run within the first 50 iterations (Figure 9 in Appendix C.3). Both ES methods also far outperform QuZO, the zeroth-order baseline of Xu et al. (2026). Because a specialist leaves the INT4 weights intact, serving it adds only about 9% memory and 11–15% decoding time per token on the existing INT4 kernels (Appendix D).

## 5.3 Supervised and long-horizon objectives

We also examine whether the search space supports supervised fine-tuning and RLVR over many turns. Supervised fine-tuning maximizes the likelihood of the data, as MobileSpec does with the user’s records, but it changes the spectrum much more than RLVR does (Figure 8c,d in Appendix B). On Commonsense-170K (Hu et al., 2023) with Llama-3-8B, FuRA-S outperforms all other parameterefficient forward-only methods under the same budget of forward passes and exceeds the strongest of them, VeRA, by about three points (Appendix E). The search space thus remains effective for an objective whose weight changes do not follow spectral inheritance. The agentic Sudoku benchmark of Agentic ESOpt (Zheng et al., 2026) tests RLVR over many turns, since the agent acts for up to 45 turns and receives a single reward at the end of the episode, as in phoneuse tasks. On this benchmark, FuRA-S matches the held-out success that Agentic ESOpt reports for full-parameter ES with a constant perturbation scale (Appendix F).

## 5.4 Ablations

We ablate the block partition and the perturbation distribution on Countdown with Qwen2.5-3B-Instruct, where FuRA-S reaches 49.25% accuracy after 200 iterations. Replacing the block SVD with the SVD of the whole matrix reduces FuRA-S to ES over the global singular values, the search space of SVF. Its perturbations change only the spectrum and rotate neither frame (Section 4.3), and accuracy drops by about four points to 45.40%. Frame rotation, which only the block structure provides, therefore contributes to the gain of FuRA-S. Replacing the Gaussian perturbations with Rademacher ones, which change every block singular value by exactly $\pm \mu _ { \mathrm { r e l } }$ times its pretrained value, reduces accuracy by almost ten points to 39.60%. Since Rademacher perturbations have the zero mean and equal variance that the first-order analysis assumes (Corollary 1 in Appendix A), the benefit of Gaussian perturbations comes from properties beyond these two moments.

Takeaway 3: FuRA-S matchesfull-parameter ES with at most 0.05% ofthe parameters, outperforms ES over LoRA adapters or quantized weights, and surpasses BPbased policy-gradient methods on most models.

## 6 ON-DEVICE TRAINING EVALUATION

We next evaluate MobileSpec in its target settings, testing whether the step-wise fitness makes long-horizon phoneuse tasks trainable (Section 6.1), whether specialists trained on the likelihood of the user’s records compete with peruser LoRA trained by backpropagation (Section 6.2), and whether training fits in the memory of a phone (Section 6.3).

Table 3. Success rate (%) of MAI-UI-2B on MobileWorld tasks, for the generalist and its specialists after up to 15 ES iterations.
<table><tr><td>Task</td><td>Base</td><td>Episode reward</td><td>Step-wise (ours)</td></tr><tr><td>MastodonReply</td><td>23.4</td><td>79.7</td><td>89.1</td></tr><tr><td>CheckEventTime</td><td>9.4</td><td>92.2</td><td>100.0</td></tr><tr><td>RequestCarpooling</td><td>6.2</td><td>14.1</td><td>98.4</td></tr><tr><td>MastodonFollow</td><td>23.4</td><td>64.1</td><td>64.1</td></tr><tr><td>MastodonPostPoll</td><td>9.4</td><td>20.3</td><td>68.8</td></tr></table>

## 6.1 Phone-use agents

Training a phone-use agent for its user’s tasks is the main use case of MobileSpec. We evaluate this use case on the GUI-only tasks of the MobileWorld benchmark (Kong et al., 2026), which involve real Android apps such as Mastodon, an email client, and a calendar. In each task, the agent has up to 30 steps to accomplish a natural-language goal on a freshly reset phone, after which a rule-based checker inspects the device and returns the reward (Appendix C.4).

MAI-UI-2B (Zhou et al., 2025a) succeeds on only 11.1% of these tasks, and online reinforcement learning raises the success rate to 17.1%, as reported by Tang et al. (2026). A user, however, needs the agent to succeed on the tasks that they perform, so MobileSpec trains one specialist for each of them. We train such specialists on five tasks on which MAI-UI-2B succeeds in fewer than a quarter of the episodes. The tasks are to reply to a Mastodon post (MastodonReply), set an alarm from the time of an event in an email (Check-EventTime), text a contact for a ride (RequestCarpooling), follow a given Mastodon account (MastodonFollow), and post a multiple-choice poll with given hashtags (Mastodon-PostPoll). Each specialist changes only the 516,096 block singular values of the model, 0.024% of its parameters, and trains for 15 iterations, or 960 episodes. We compare the step-wise fitness with episode rewards (Section 3.4).

The step-wise fitness turns tasks that the generalist rarely solves into tasks that its specialists solve reliably (Table 3). The specialists succeed in nearly every episode on three tasks and improve substantially on the other two. Episode rewards also improve on the generalist but trail the stepwise fitness on four of the five tasks and fail almost entirely on RequestCarpooling, since an episode reward cannot tell which action moved the member toward the goal.

## 6.2 Personalization

Personalization tests the second kind of fitness of Section 3.1, the likelihood of the user’s own records. We evaluate MobileSpec on six LaMP tasks (Salemi et al., 2024) with Llama-3.1-8B-Instruct in the user-based setting of LaMP and the time-based setting of OPPU (Tan et al., 2024). In the user-based setting (U), one specialist per task is trained on some users and tested on unseen users, whose prompts (a) New users (U): one shared model, trained once

![](images/c69b698575def1639eaccdd7ab67b8dff2e6ff4cbc524e3ef47278c9f402649e.jpg)

![](images/6ba6daf59a488dea283ea194ca616553bb9a1920567f672729866b872df8fd89.jpg)

![](images/89ffe20ca0a652ddce8f9df2301d3fc37ea7d925bd5c551daac521be9dfb225e.jpg)

(b) Per-user checkpoints (T): trained on each user's history  
![](images/2e20bb4fcca1ffc8f2f970f4a09d75aa613600020ff000d38e3f605218d2545d.jpg)

![](images/0df42a6b203d38f71f1e472818028c80a5c811af9a9a6e0bdd124da162e67251.jpg)  
FuRA-S + profile

![](images/ff04d4c8706a0f70827335e00f19a2548909bd4c7f3768e9c9790f5f1b86808d.jpg)

![](images/49cfb3c4794e9ddae26ffbb58b5b646105427a5b0e2e5b3a6949931d65479608.jpg)

![](images/2951d2189434c7271416d8fff6d869599d29ea9bb83e6d1b5aa12027da9594ee.jpg)  
base + profile

![](images/65eb3b79769ca5c7aab430561562566d2b1d0236a9e32efc4f2836d7dfb0732d.jpg)

![](images/9088e3fd25e987f4d210f6e08e775cc6bc0d1b32b041adf40c37ce63fce866cd.jpg)

![](images/b0448379e0cea62300d060db73e1fe947707f8bfba1959c3fbc118b70a891351.jpg)

![](images/3f9df7ca28c8dd46e065bc57943c1b87944e98fe534dc4d8bec35d53bb2095f0.jpg)  
Figure 6. LaMP with Llama-3.1-8B-Instruct. (a) User-based setting, without and with the retrieved profile. (b) Time-based setting, with per-user specialists trained by ES (ZO in the legend) and per-user LoRA trained by backpropagation (FO in the legend).

![](images/b2b54ad8ee19fb72cd5eb4eb36805a2e24a58f0f5ca74e8fc18f00a01f7e74db.jpg)

![](images/1284b8d348dacdf4675bd07ac92de241b2ec17ad6d1ee60a5f16f7b1e7459089.jpg)  
Figure 7. Training a specialist for a LaMP-2M user on an iPhone 16 Pro: (a) training and validation loss, (b) memory use.

either omit or include one record retrieved from their profile of past records. In the time-based setting (T), a separate specialist is trained for each user on their earlier records and tested on later queries, as MobileSpec does on the phone. We compare these specialists with per-user LoRA trained by backpropagation as in OPPU (Appendix C.5).

The likelihood of the user’s records proves to be an effective fitness function. In the user-based setting, the specialists improve over the generalist on all six tasks when both use the retrieved profile, with the largest gain on the title-generation task LaMP-5 (Figure 6a and Table A9). In the time-based setting, the per-user specialists nearly double the accuracy of the generalist on the movie-tagging task LaMP-2M and more than triple its ROUGE-1 on LaMP-5. With forward passes alone, they outperform per-user LoRA trained by backpropagation on LaMP-2M and the tweet-paraphrasing task LaMP-7 (Figure 6b and Table A10), while each specialist stores only its 1.25M relative changes δ, or 4.98 MB. The gains in the user-based setting hold for Qwen3.5-4B on all six tasks, and its per-user specialists also outperform per-user LoRA on LaMP-1 and LaMP-2M (Appendix H).

## 6.3 Training on an iPhone

We run MobileSpec on an iPhone 16 Pro with 8 GB of memory and train a specialist for one LaMP-2M user on top of the released GPTQ-INT4 (Frantar et al., 2023) Llama-3.1-8B-Instruct, following the time-based protocol of Section 6.2. The phone reproduces the forward computation of a GPU server, with a validation loss of 0.8505 for the generalist on the phone and 0.8524 on the server. After 20 iterations over 10.4 hours, the validation loss drops from 0.85 to 0.19 (Figure 7a), and the specialist answers 8 of 15 held-out queries correctly, compared with 5 for the generalist. The INT4 model occupies 5.5 GB, more than the 5.3 GB available to an app, but since MobileSpec streams the decoder layers from flash, training uses only 2.1 GB of resident memory and peaks at 3.5 GB (Figure 7b). Forward computation accounts for 97% of the training time, and streaming takes only 14 seconds per iteration. Training a specialist for an 8B model thus finishes overnight within the memory available to an app.

Takeaway 4: MobileSpec trains specialists that reliably solve phone-use tasks on which the generalist rarely succeeds, and it trains per-user specialists of a few megabytes within the memory of a phone.

## 7 CONCLUSION

We have presented MobileSpec, an on-device system that trains specialist LLMs with forward passes over the INT4 model that the phone already runs. At its core, FuRA-S trains only the block-spectral coordinates of pretrained weights, whose independent rescaling rotates the singular vectors of each weight matrix while largely preserving its singular values, as RLVR does. With specialists of 1–5 MB,

MobileSpec raises the success rate of a phone-use agent from 6–23% up to 89–100% and trains a personalized 8B model on an iPhone overnight. Training with forward passes only remains slow, since each iteration evaluates a population of perturbed models and a multi-step task multiplies this cost by the episodes of every member and the steps of every episode. Each task also needs a fitness that the device can compute without labels, specialists do not yet share what they learn, and our evaluation covers five MobileWorld tasks and six LaMP tasks. More broadly, training specialist LLMs on personal devices becomes practical once the search space is small enough to store for every task and aligned with how RLVR changes pretrained weights, so that a phone can adapt a model to its user while the user’s data never leaves the device.

## REFERENCES

Chen, Y., Zhang, Y., Cao, L., Yuan, K., and Wen, Z. Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures. In The Thirteenth International Conference on Learning Representations (ICLR 2025), pp. 62581–62607, 2025. URL https://proceedi ngs.iclr.cc/paper\_files/paper/2025/h ash/9ccc9d814d3dee4750debaf23061e733 -Abstract-Conference.html.

Feng, L., Xue, Z., Liu, T., and An, B. Group-in-group policy optimization for LLM agent training. In Advances in Neural Information Processing Systems 38 (NeurIPS 2025), pp. 46375–46408, 2025. doi: 10.52202/085713-1 544. URL https://proceedings.neurips.cc /paper\_files/paper/2025/hash/420c9f7 77c0b4f78d515e53cf74d58b2-Abstract-C onference.html.

Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. OPTQ: Accurate quantization for generative pre-trained transformers. In The Eleventh International Conference on Learning Representations (ICLR 2023), 2023. URL https://openreview.net/forum?id=tcbB PnfwxS. Also known as GPTQ; arXiv:2210.17323.

Gao, L., Ziashahabi, A., Niu, Y., Avestimehr, S., and Annavaram, M. MobiZO: Enabling Efficient LLM Fine-Tuning at the Edge via Inference Engines. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 20206–20223, 2025. doi: 10.18653/v1/2025.emnlp-main.1022. URL https://aclanthology.org/2025.emnlp-m ain.1022/.

Geng, J., Zhao, L., Lu, Y., and Luo, B. MobileFineTuner: A Mobile-Native Framework for On-Device LLM Fine-Tuning in Real-World Embedded AI Applications, 2025. URL https://arxiv.org/abs/2512.08211.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations (ICLR 2022), 2022. URL https://openreview.net/forum?id=nZeV KeeFYf9.

Hu, Z., Wang, L., Lan, Y., Xu, W., Lim, E.-P., Bing, L., Xu, X., Poria, S., and Lee, R. LLM-Adapters: An adapter family for parameter-efficient fine-tuning of large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023), pp. 5254–5276, 2023. doi: 10.18653/v1/2023.emnlp-main.319. URL https://ac lanthology.org/2023.emnlp-main.319/.

Kakade, S. and Langford, J. Approximately optimal approximate reinforcement learning. In Proceedings ofthe 19th International Conference on Machine Learning (ICML 2002), pp. 267–274, 2002.

Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In 3rd International Conference on Learning Representations (ICLR 2015), 2015. URL https://arxiv.org/abs/1412.6980.

Kong, Q., Zhang, X., Yang, Z., Gao, N., Liu, C., Tong, P., Cai, C., Zhou, H., Zhang, J., Chen, L., Liu, Z., Hoi, S., and Wang, Y. MobileWorld: Benchmarking autonomous mobile agents in agent-user interactive and MCP-augmented environments. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6142– 6167, 2026. doi: 10.18653/v1/2026.acl-long.278. URL https://aclanthology.org/2026.acl-lon g.278/.

Kopiczko, D. J., Blankevoort, T., and Asano, Y. M. VeRA: Vector-based Random Matrix Adaptation. In The Twelfth International Conference on Learning Representations (ICLR 2024), pp. 6815–6835, 2024. URL https:// proceedings.iclr.cc/paper\_files/pape r/2024/hash/1b53ad08de383a049e9668a9 d0b6a053-Abstract-Conference.html.

Korotyshova, D., Shaposhnikov, B., Malakhov, A., Khokhulin, A., Surnachev, N., Ovcharenko, K., Bredis, G., Gorbatovski, A., Sinii, V., and Gavrilov, D. ESSA: Evolutionary strategies for scalable alignment. In Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing, pp. 50104–50142, 2026. doi: 10.18653/2026.emnlp-main.2204. URL https://aclanthology.org/2026.emnlp-m ain.2204/.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles (SOSP 2023), pp. 611–626, 2023. doi: 10.1145/3600006.3613165. URL https://doi.org/10.1145/3600006.3613 165.

Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., Brahman, F., Miranda, L. J. V., Liu, A., Dziri, N., Lyu, X., Gu, Y., Malik, S., Graf, V., Hwang, J. D., Yang, J., Le Bras, R., Tafjord, O., Wilhelm, C., Soldaini, L., Smith, N. A., Wang, Y., Dasigi, P., and Hajishirzi, H. Tulu 3: Pushing frontiers in open language model posttraining. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum ?id=i1uGbfHHpH.

Li, K., Han, S., Su, Q., Li, W., Cai, Z., and Ji, S. Uni-LoRA: One Vector is All You Need. In Advances in Neural Information Processing Systems 38 (NeurIPS 2025), pp. 43816–43841, 2025. doi: 10.52202/085713-1461. URL https://proceedings.neurips.cc/paper \_files/paper/2025/hash/3e596a70b60dc f7c6e0ac1a2d73e7470-Abstract-Confere nce.html.

Lingam, V., Tejaswi, A., Vavre, A., Shetty, A., Gudur, G. K., Ghosh, J., Dimakis, A., Choi, E., Bojchevski, A., and Sanghavi, S. SVFT: Parameter-Efficient Fine-Tuning with Singular Vectors. In Advances in Neural Information Processing Systems 37 (NeurIPS 2024), pp. 41425–41446, 2024. doi: 10.52202/079017-1311. URL https://pr oceedings.neurips.cc/paper\_files/pap er/2024/hash/48c368f105e8145b945227b 73255635a-Abstract-Conference.html.

Liu, Y., Zhu, Z., Gong, C., Cheng, M., Hsieh, C.-J., and You, Y. Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning. In Advances in Neural Information Processing Systems 38 (NeurIPS 2025), pp. 21139–21169, 2025a. doi: 10.52202/085713-0712. URL https://proceedings.neurips.cc/p aper\_files/paper/2025/hash/1e5c2efbd dc02c1d971e2f19ccdb07d0-Abstract-Con ference.html.

Liu, Z., Zhao, C., Iandola, F., Lai, C., Tian, Y., Fedorov, I., Xiong, Y., Chang, E., Shi, Y., Krishnamoorthi, R., Lai, L., and Chandra, V. MobileLLM: Optimizing sub-billion parameter language models for on-device use cases. In Proceedings of the 41st International Conference on Machine Learning (ICML 2024), volume 235 of Proceedings of Machine Learning Research, pp. 32431–32454. PMLR, 2024. URL https://proceedings.mlr.pres s/v235/liu24ce.html.

Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., Lee, W. S., and Lin, M. Understanding R1-Zero-like training: A critical perspective. In Conference on Language Modeling (COLM 2025), 2025b. URL https: //openreview.net/forum?id=5PAF7PAY2Y.

Malladi, S., Gao, T., Nichani, E., Damian, A., Lee, J. D., Chen, D., and Arora, S. Fine-tuning language models with just forward passes. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023), pp. 53038–53075, 2023. doi: 10.52202/075280-2308. URL https://proceedings.neurips.cc/paper \_files/paper/2023/hash/a627810151be4 d13f907ac898ff7e948-Abstract-Confere nce.html.

Meng, F., Wang, Z., and Zhang, M. PiSSA: Principal singular values and singular vectors adaptation of large lan-

guage models. In Advances in Neural Information Processing Systems 37 (NeurIPS 2024), pp. 121038–121072, 2024. doi: 10.52202/079017-3846. URL https://pr oceedings.neurips.cc/paper\_files/pap er/2024/hash/db36f4d603cc9e3a2a5e10b 93e6428f2-Abstract-Conference.html.

Park, J., Hong, Y., Kim, S., and Lee, J. Memory-Efficient Structured Backpropagation for On-Device LLM Fine-Tuning. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 6: Industry Track), pp. 906–916, 2026. doi: 10.18653/v1/20 26.acl-industry.62. URL https://aclanthology .org/2026.acl-industry.62/.

Peng, D., Fu, Z., and Wang, J. PocketLLM: Enabling On-Device Fine-Tuning for Personalized LLMs. In Proceedings of the Fifth Workshop on Privacy in Natural Language Processing, pp. 91–96, 2024. URL https://ac lanthology.org/2024.privatenlp-1.10/.

Qiu, X., Gan, Y., Hayes, C. F., Liang, Q., Xu, Y., Dailey, R., Meyerson, E., Hodjat, B., and Miikkulainen, R. Evolution strategies at scale: LLM fine-tuning beyond reinforcement learning. In Proceedings of the 43rd International Conference on Machine Learning (ICML 2026), volume 306 of Proceedings of Machine Learning Research, pp. 101987–102015. PMLR, 2026. URL https://proc eedings.mlr.press/v306/qiu26f.html.

Salemi, A., Mysore, S., Bendersky, M., and Zamani, H. LaMP: When Large Language Models Meet Personalization. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7370–7392, 2024. doi: 10.18653/v1/2024.acl- long.399. URL https: //aclanthology.org/2024.acl-long.399/.

Sarkar, B., Fellows, M., Duque, J. A., Letcher, A., Villares, A. L., Sims, A., Wibault, C., Samsonov, D., Cope, D., Liesen, J. L., Li, K., Seier, L., Wolf, T., Berdica, U., Mohl, V., Goldie, A. D., Courville, A., Sevegnani, K., Whiteson, S., and Foerster, J. N. Evolution strategies at the hyperscale. In Proceedings of the 43rd International Conference on Machine Learning (ICML 2026), volume 306 of Proceedings of Machine Learning Research, pp. 107670–107741. PMLR, 2026. URL https://proc eedings.mlr.press/v306/sarkar26a.html.

Schulman, J. and Thinking Machines Lab. LoRA without regret. Thinking Machines Lab: Connectionism, 2025. URL https://thinkingmachines.ai/blog /lora/.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms,

2017. URL https://arxiv.org/abs/1707.0 6347.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., and Guo, D. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Song, C. and Tang, X. Memory-Efficient Backpropagation for Fine-Tuning LLMs on Resource-Constrained Mobile Devices. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 766–774, 2025. doi: 10.18653/v1/2025.e mnlp-industry.52. URL https://aclanthology .org/2025.emnlp-industry.52/.

Sun, C., Wei, J., Wu, Y., Shi, Y., He, S., Ma, Z., Xie, N., and Yang, Y. SVFit: Parameter-Efficient Fine-Tuning of Large Pre-Trained Models Using Singular Values, 2024. URL https://arxiv.org/abs/2409.05926.

Sun, Q., Cetin, E., and Tang, Y. Transformer-Squared: Self-adaptive LLMs. In The Thirteenth International Conference on Learning Representations (ICLR 2025), 2025. URL https://openreview.net/forum ?id=dh4t9qmcvK.

Sun, Y., Chen, Q., He, X., Wang, J., Feng, H., Han, J., Ding, E., Cheng, J., Li, Z., and Wang, J. Singular Value Finetuning: Few-shot Segmentation requires Few-parameters Fine-tuning. In Advances in Neural Information Processing Systems 35 (NeurIPS 2022), pp. 37484–37496, 2022. doi: 10.52202/068431-2717. URL https: //papers.neurips.cc/paper\_files/pape r/2022/hash/f3bfbd65743e60c685a3845b d61ce15f-Abstract-Conference.html.

Tan, L., Zhao, Y., Yang, Y., Zhang, R., Yu, X., and Zhang, Z. GRZO: Group-Relative Zeroth-Order Optimization for Large Language Model Fine-Tuning. In Findings of the Associationfor Computational Linguistics: EMNLP 2026, pp. 36551–36581, 2026. URL https://acla nthology.org/2026.findings-emnlp.171 1/.

Tan, Z., Zeng, Q., Tian, Y., Liu, Z., Yin, B., and Jiang, M. Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 6476–6491, 2024. doi: 10.18653/v1/2024.emnlp-main.372. URL https://ac lanthology.org/2024.emnlp-main.372/.

Tang, F., Lu, Z., Zhang, B., Lu, W., Xiao, J., Zhuang, Y., and Shen, Y. ClawGUI: A unified framework for training, evaluating, and deploying GUI agents, 2026. URL http s://arxiv.org/abs/2604.11784.

Wang, H., Li, Y., Wang, S., Chen, G., and Chen, Y. MiLoRA: Harnessing minor singular components for parameterefficient LLM finetuning. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 4823–4836, 2025. doi: 10.18653/v1/2025.naacl-long.248. URL https://aclanthology.org/2025.na acl-long.248.

Wang, J., Xu, H., Ye, J., Yan, M., Shen, W., Zhang, J., Huang, F., and Sang, J. Mobile-Agent: Autonomous multi-modal mobile device agent with visual perception, 2024. URL https://arxiv.org/abs/2401.1 6158.

Xie, T., Zhang, D., Chen, J., Li, X., Zhao, S., Cao, R., Hua, T. J., Cheng, Z., Shin, D., Lei, F., Liu, Y., Xu, Y., Zhou, S., Savarese, S., Xiong, C., Zhong, V., and Yu, T. OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track, pp. 52040–52094, 2024. doi: 10.52202/079017-1650. URL https://proceedings.neurips.cc/paper \_files/paper/2024/hash/5d413e48f84dc 61244b6be550f1cd8f5-Abstract-Dataset s\_and\_Benchmarks\_Track.html.

Xu, Y., Schweighofer, K., Miikkulainen, R., and Qiu, X. Quantized evolution strategies: High-precision finetuning of quantized LLMs at low-precision cost, 2026. URL https://arxiv.org/abs/2602.03120.

Yin, Q., Wu, Y., Shen, Z., Li, S., Wang, Z., Li, Y., Leong, C. T., Kang, J., and Gu, J. Evaluating parameter efficient methods for RLVR. In Forty-third International Conference on Machine Learning, 2026. URL https: //openreview.net/forum?id=76CIL1O0bz.

Zhao, Y., Li, H., Young, I., and Zhang, Z. Poor man’s training on MCUs: A memory-efficient quantized backpropagation-free approach. ACM Transactions on Design Automation of Electronic Systems, 30(5):1–33, 2025. doi: 10.1145/3745772. URL https://doi.org/10.1 145/3745772.

Zhao, Y., Zhang, R., Tan, L., Moran, N., Qin, T., and Zhang, Z. FuRA: Full-rank parameter-efficient fine-tuning with spectral preconditioning. In Advances in Neural Information Processing Systems 39 (NeurIPS 2026), 2026. URL https://openreview.net/forum?id=zf1v Yk0XnQ. To appear.

Zheng, Z., Chen, R., Ba, Y., Wang, Z., Teh, Y. W., and Lee, W. S. Agentic ESOpt: Fine-tuning long-horizon LLM

agents with minimal GPU requirements, 2026. URL https://arxiv.org/abs/2608.17310.

Zhou, H., Zhang, X., Tong, P., Zhang, J., Chen, L., Kong, Q., Cai, C., Liu, C., Wang, Y., Zhou, J., and Hoi, S. MAI-UI technical report: Real-world centric foundation GUI agents, 2025a. URL https://arxiv.org/abs/ 2512.22047.

Zhou, J., Yang, Y., Zhen, K., Liu, Z., Zhao, Y., Banijamali, E., Mouchtaris, A., Wong, N., and Zhang, Z. QuZO: Quantized Zeroth-Order Fine-Tuning for Large Language Models. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pp. 5341– 5359, 2025b. doi: 10.18653/v1/2025.emnlp-main.271. URL https://aclanthology.org/2025.em nlp-main.271/.

Zhu, H., Cong, W., Sha, Z., Mukherjee, S., Song, X., Gonzalez-Mart ´ ´ınez, D., Wu, X., Tian, Y., Liu, S., Pan, D. Z., and Wang, Z. ISO: An RLVR-native optimization stack, 2026a. URL https://arxiv.org/abs/26 07.19331v1.

Zhu, H., Zhang, Z., Huang, H., Su, D., Liu, Z., Zhao, J., Fedorov, I., Pirsiavash, H., Lee, J., Pan, D. Z., Wang, Z., Tian, Y., and Tai, K. S. The path not taken: RLVR learns off the principals. In Advances in Neural Information Processing Systems 39 (NeurIPS 2026), 2026b. URL https://openreview.net/forum?id=7ob9 PpTAMB. To appear.

## A FIRST-ORDER EFFECT OF BLOCK RESCALING

Section 4.3 uses the following two results.

Proposition 1 (Block rescaling). Suppose W has distinct positive singular values. Tofirst order in δ, the relative change ofeach singular value ofW is the weighted average ofthe relative changes ofthe block singular values in Eq. (7), with nonnegative weights $w _ { i j }$ that sum to one over j. Each off-diagonal entry $G _ { i k }$ rotates the singular vectors $\mathbf { u } _ { k }$ and $\mathbf { v } _ { k }$ toward $\mathbf { u } _ { i }$ and $\mathbf { v } _ { i } ,$ with a magnitude that grows as $\sigma _ { i }$ approaches $\sigma _ { k }$

Corollary 1 (Source of the rotation). (i) Ifall δ equal c, then $\mathbf { W } ^ { \prime } = ( 1 + c ) \mathbf { W }$ , which scales the spectrum and leaves the singular vectors unchanged. (ii) IfW forms a single block $( K = 1 )$ , then $\mathbf { R } = \mathbf { V }$ and $\mathbf { G } = \mathrm { d i a g } ( \delta )$ , so the update changes only the spectrum. (iii) $I f m \geq n$ and the entries of δ are independent with zero mean and equal variance, the diagonal of G accountsfor an expectedfraction $\begin{array} { r } { \frac { 1 } { n } \sum _ { i , j } w _ { i j } ^ { 2 } \mathop { o f { \mathbb E } } \| \mathbf { G } \| _ { F } ^ { 2 } , } \end{array}$ , and the off-diagonal entries accountfor the rest.

We prove both below. Let $\mathbf { W } = \mathbf { U } \pmb { \Sigma } \mathbf { V } ^ { \top }$ be the reduced SVD of $\mathbf { W } \in \mathbb { R } ^ { m \times n }$ with distinct positive singular values, and let $\mathbf { W } ^ { \prime } = \mathbf { W } ( \mathbf { I } + \mathbf { E } )$ with $\mathbf { E } = \mathbf { R } \mathrm { d i a g } ( \delta ) \mathbf { R } ^ { \top }$ as in Eq. (4). The update is $\Delta \mathbf { W } = \mathbf { W } ^ { \prime } - \mathbf { W } = \mathbf { U } \boldsymbol { \Sigma } \mathbf { V } ^ { \top } \mathbf { E }$ , so that $\mathbf { U } ^ { \top } \Delta \mathbf { W } \mathbf { V } = \Sigma \mathbf { G }$ with $\mathbf { G } = \mathbf { V } ^ { \top } \mathbf { E } \mathbf { V }$ . We write $\begin{array} { r } { \mathbf { E } = \sum _ { j } \delta _ { j } \mathbf { r } _ { j } \mathbf { r } _ { j } ^ { \top } } \end{array}$ , where $\mathbf { r } _ { j }$ is the j-th column of the orthogonal matrix R.

Block identity. Since ${ \bf R } _ { k }$ is orthogonal, $\mathbf { L } _ { k }$ diag $( \mathbf { s } _ { k } \odot ( 1 + \delta _ { k } ) ) \mathbf { R } _ { k } ^ { \top } = \mathbf { L } _ { k } \operatorname { d i a g } ( \mathbf { s } _ { k } ) \mathbf { R } _ { k } ^ { \top } \mathbf { R } _ { k } \operatorname { d i a g } ( 1 + \delta _ { k } ) \mathbf { R } _ { k } ^ { \top } = \mathbf { W } _ { k } \left( \mathbf { I } + \mathbf { R } _ { k } \right) ,$ $\mathbf R _ { k } \mathrm { d i a g } ( \delta _ { k } ) \mathbf R _ { k } ^ { \top } )$ . Stacking the blocks gives Eq. (4).

Singular values. For a small update, first-order perturbation theory gives $\Delta \sigma _ { i } \approx \mathbf { u } _ { i } ^ { \top } \Delta \mathbf { W } \mathbf { v } _ { i }$ for each simple singular value. Since $\mathbf { u } _ { i } ^ { \top } \mathbf { U } \mathbf { \Sigma } \mathbf { V } ^ { \top } = \sigma _ { i } \mathbf { v } _ { i } ^ { \top }$ , this equals $\sigma _ { i } \mathbf { v } _ { i } ^ { \top } \mathbf { E } \mathbf { v } _ { i } = \sigma _ { i } G _ { i i }$ , and

$$
G _ { i i } = \sum _ { j = 1 } ^ { n } \delta _ { j } ( \mathbf { r } _ { j } ^ { \top } \mathbf { v } _ { i } ) ^ { 2 } = \sum _ { j = 1 } ^ { n } w _ { i j } \delta _ { j } .\tag{8}
$$

The weights $w _ { i j }$ are nonnegative and sum to $\| \mathbf { R } ^ { \top } \mathbf { v } _ { i } \| _ { 2 } ^ { 2 } = \| \mathbf { v } _ { i } \| _ { 2 } ^ { 2 } = 1$ , since R is orthogonal. This part holds for any shape of W.

Singular vectors. The same theory gives the first-order changes of the singular vectors:

$$
\mathbf { v } _ { k } ^ { \prime } - \mathbf { v } _ { k } \approx \sum _ { i \neq k } { \frac { \sigma _ { i } ^ { 2 } + \sigma _ { k } ^ { 2 } } { \sigma _ { k } ^ { 2 } - \sigma _ { i } ^ { 2 } } } G _ { i k } \mathbf { v } _ { i } ,
$$

$$
\mathbf { u } _ { k } ^ { \prime } - \mathbf { u } _ { k } \approx \sum _ { i \neq k } { \frac { 2 \sigma _ { i } \sigma _ { k } } { \sigma _ { k } ^ { 2 } - \sigma _ { i } ^ { 2 } } } G _ { i k } \mathbf { u } _ { i } .\tag{9}
$$

The off-diagonal entries of G therefore rotate the singular vectors, with coefficients that grow as $\sigma _ { i }$ approaches $\sigma _ { k }$ . When W has more columns than rows and full row rank, the right singular vectors additionally move outside the original row space, by

$$
( \mathbf { I } - \mathbf { V } \mathbf { V } ^ { \top } ) \mathbf { E } \mathbf { v } _ { k }\tag{10}
$$

to first order. These expressions assume distinct singular values. For repeated singular values, the perturbation must instead be characterized at the level of singular subspaces.

Corollary 1. (i) If $\delta _ { j } = c$ for all j, then $\mathbf { E } = c \mathbf { R } \mathbf { R } ^ { \top } = c \mathbf { I } .$ , and $\mathbf { W } ^ { \prime } = ( 1 + c ) \mathbf { W }$ has the singular vectors of W and the spectrum $( 1 + c ) \Sigma$ . (ii) If $K = 1$ , the block SVD is the SVD of W, so R = V, with the reduced factors when $m < n$ , and $\mathbf { G } = \mathrm { d i a g } ( \delta )$ . All off-diagonal entries of G then vanish, and so does the term in Eq. (10), since $\mathbf { E v } _ { k } = \delta _ { k } \mathbf { v } _ { k }$ lies in the row space. (iii) Let $m \geq n ,$ , so that V is square and orthogonal, and let the entries of δ be independent with zero mean and variance $\nu ^ { 2 }$ . Then $\mathbb { E } \| \mathbf { G } \| _ { F } ^ { 2 } = \mathbb { E } \| \mathbf { E } \| _ { F } ^ { 2 } = \bar { \mathbb { E } } \| \pmb { \delta } \| _ { 2 } ^ { 2 } = n \nu ^ { 2 }$ , while

$$
\mathbb { E } G _ { i i } ^ { 2 } = \sum _ { j , l } w _ { i j } w _ { i l } \mathbb { E } [ \delta _ { j } \delta _ { l } ] = \nu ^ { 2 } \sum _ { j } w _ { i j } ^ { 2 } .\tag{11}
$$

The diagonal therefore holds the expected fraction $\textstyle \frac { 1 } { n } \sum _ { i , j } w _ { i j } ^ { 2 }$ of $\mathbf { \Delta E } \Vert \mathbf { G } \Vert _ { F } ^ { 2 }$ . Since the weights of each $\mathbf { v } _ { i }$ sum to one, $\textstyle \sum _ { j } w _ { i j } ^ { 2 }$ lies between $1 / n ,$ , when $\mathbf { v } _ { i }$ spreads evenly over all columns of $\mathbf { R } ,$ and 1, when $\mathbf { v } _ { i }$ is a column of R.

Spectral metrics. Section 4.3 and Appendix B use the metrics of ISO (Zhu et al., 2026a),

$$
\begin{array} { l } { \displaystyle \rho _ { \Sigma } = \frac { \| \boldsymbol { \sigma } ( \mathbf { W } ^ { \prime } ) - \boldsymbol { \sigma } ( \mathbf { W } ) \| _ { 2 } } { \| \Delta \mathbf { W } \| _ { F } } , } \\ { \displaystyle \kappa = \frac { m n } { q } \frac { \| \mathrm { d i a g } ( \mathbf { U } ^ { \top } \Delta \mathbf { W } \mathbf { V } ) \| _ { 2 } ^ { 2 } } { \| \Delta \mathbf { W } \| _ { F } ^ { 2 } } , } \end{array}\tag{12}
$$

where $\sigma ( \cdot )$ denotes the sorted singular values. Since $\mathbf { W } \in \mathcal { F } ( \mathbf { W } )$ , the distance from $\mathbf { W } ^ { \prime }$ to $\mathcal { F } ( \mathbf { W } )$ is at most $\| \Delta \mathbf { W } \| _ { F } ,$ so $0 \leq \rho _ { \Sigma } \leq 1$ . The matrices $\mathbf { u } _ { i } \mathbf { v } _ { i } ^ { \top }$ form an orthonormal basis of the q-dimensional subspace of first-order spectrum-changing directions, and an isotropic random update of fixed Frobenius norm places an expected fraction $q / ( m n )$ of its squared norm in this subspace, so κ has expectation one. Since $m n / q = \operatorname* { m a x } ( m , n )$ , κ equals $\operatorname* { m a x } ( m , n )$ times the share of $\| \Delta \mathbf { W } \| _ { F } ^ { 2 }$ on the diagonal of $\mathbf { U } ^ { \top } \Delta \mathbf { W } \mathbf { V }$

## B MEASURED ROTATION AND SPECTRAL METRICS

Table A1 lists the random updates of the block-SVD factors that Section 4.3 compares. Figure 8(a) measures the rotation of the leading singular vectors of eight Qwen2.5-1.5B-Instruct matrices under four updates with the norm of the update learned on Countdown, and panel (b) compares it with its first-order prediction at small update sizes. At the size of the update learned on Countdown, a random block rescaling rotates the leading singular vectors by $3 3 ^ { \circ }$ and the learned one by $3 7 ^ { \circ }$ , compared with $1 0 ^ { \circ }$ for a dense update. Panels (c) and (d) give the per-layer $\rho _ { \Sigma }$ and κ of Qwen2.5-Math-7B after training on MATH, together with the values that ISO (Zhu et al., 2026a) reports for supervised fine-tuning.

![](images/5e0773ae23708d6aa4800909edecba38ddafdc3f9517a81cc90b1a081a7d10e6.jpg)

![](images/db49406b5c91c3ff8481d4dd1c915168bab4a6641c4985fd0e99fb4e36db8721.jpg)

![](images/ec2b7d913640950d64d7b70ad83898fead3404dba931b77c21534a7dbbe9dce2.jpg)

![](images/69c65d882e5e3a38e051d6fc69677ebff3b6460680d651865ef4c5f344774e34.jpg)  
Figure 8. (a) Rotation of the leading singular vectors of eight Qwen2.5-1.5B-Instruct matrices under four updates with the norm of the update learned on Countdown, and (b) its first-order prediction at small sizes. (c, d) Per-layer $\rho _ { \Sigma }$ and κ of Qwen2.5-Math-7B after MATH training, with the values reported by ISO for supervised fine-tuning in gray.

Table A1. Random updates of different block-SVD factors (medians). $\angle \mathbf { U } , \mathbf { \Xi } \angle \mathbf { V } \colon$ largest principal angles between the leading 16- dimensional singular subspaces of W and $\mathbf { W } ^ { \prime }$
<table><tr><td>Updated factor</td><td>Values</td><td> $\rho _ { \Sigma }$ </td><td>κ ∠U</td><td> $\angle \mathbf { V }$ </td></tr><tr><td>Block  $\mathbf { s } _ { k }$  (FuRA-S)</td><td></td><td>1.5k 5.3% 7.5</td><td> $8 . 6 ^ { \circ }$ </td><td>8.8°</td></tr><tr><td> $\mathbf { R } _ { k } ,$  additive</td><td>135k 3.0%</td><td></td><td>1.2  $6 . 1 ^ { \circ }$ </td><td> $6 . 4 ^ { \circ }$ </td></tr><tr><td> $\mathbf { R } _ { k } .$  rotation</td><td>67k</td><td>0.0%</td><td>1.2  $0 . 0 ^ { \circ }$ </td><td> $1 . 9 ^ { \circ }$ </td></tr><tr><td> ${ \bf L } _ { k } ,$  rotation in span</td><td>67k</td><td>0.6%</td><td>1.4  $1 . 8 ^ { \circ }$ </td><td> $2 . 4 ^ { \circ }$ </td></tr><tr><td>Dense W</td><td>8M</td><td>2.3%</td><td>1.02.1°</td><td> $2 . 1 ^ { \circ }$ </td></tr></table>

## C EXPERIMENTAL DETAILS

Table A2 summarizes the hyperparameters of the experiments in Sections 5 and $^ { 6 , }$ including the supervised and agentic experiments of Appendices E and F. Unless otherwise specified, FuRA-S adapts all linear projections in the decoder layers using the block partition of Section 3.2, in which $n = K b$ and $K \leq b$ are the factor pair of n closest to ${ \sqrt { n } } ,$ as in FuRA, and attention and gated-DeltaNet output projections are split into blocks by head. For $m < b ,$ the reduced SVD of a block contains m singular values. The block SVD of Qwen2.5-1.5B takes about 32 seconds on one A100 GPU. Perturbations are scaled by the pretrained singular values, and reward-based ES standardizes fitness across each population without momentum.

Algorithm 1 summarizes the ES loop of FuRA-S (Section 3.2).

Algorithm 1 FuRA-S: forward-only block-spectral ES   
Require: Pretrained weights {W}, perturbation scale $\mu _ { \mathrm { r e l } } ,$ , step size α, population   
$N ,$ iterations T   
1: for each $\mathbf { W } \in \{ \mathbf { W } \}$ do   
2: $\mathbf { W } = [ \mathbf { W } _ { 1 } \mid \cdots \mid \mathbf { W } _ { K } ] , \quad \mathbf { W } _ { k } = \mathbf { L } _ { k } \mathrm { d i a g } ( \mathbf { s } _ { k } ) \mathbf { R } _ { k } ^ { \top }$   
3: end for   
4: θ<sub>0</sub> ← concat({s<sub>k</sub>}), D ← diag(µ<sub>rel</sub>|θ<sub>0</sub>|)   
5: Freeze $\{ \mathbf { L } _ { k } , \mathbf { R } _ { k } \}$   
6: for $t = \mathrm { \bar { 0 } } , \dots , \hat { T } - 1$ do   
7: for $j = 1 , \ldots , N / 2$ do   
8: $\mathbf { z } _ { j _ { . } } \sim \mathcal { N } ( 0 , \mathbf { I } _ { d } )$   
9: $f _ { j } ^ { \pm } \gets F ( \pmb { \theta } _ { t } \pm \mathbf { D } \mathbf { z } _ { j } )$   
10: end for   
11: { <sup>˜</sup>f<sup>±</sup>} ← Standardize $\{ f _ { j } ^ { \pm } \} )$   
N/2   
12: $\pmb { \theta } _ { t + 1 }  \pmb { \theta } _ { t } + \frac { \alpha } { N } \mathbf { D } \sum _ { j = 1 } ^ { \llangle , - } ( \tilde { f } _ { j } ^ { + } - \tilde { f } _ { j } ^ { - } ) \mathbf { z } _ { j }$   
13: end for   
14: return ${ \pmb \theta } _ { T }$

Implementation. Experiments run on NVIDIA A100 40GB GPUs. For generation-intensive tasks, we use vLLM (Kwon et al., 2023) to evaluate perturbed models. Population members are evaluated independently and in parallel. FuRA-S keeps an FP32 master copy of its block singular values and reconstructs perturbed dense weights for generation, so generation uses standard inference kernels. Supervised fine-tuning evaluates perturbations directly in the factored representation.

Table A2. FuRA-S settings across experiments. Pairs denotes the number of antithetic perturbation pairs per update, Batch denotes the number of examples, problems, or puzzles evaluated per population member, and $\mathbf { L } _ { k }$ indicates the storage precision of the frozen left singular factors. All ES experiments use $\mu _ { \mathrm { r e l } } = 3 \times 1 0 ^ { - 2 } .$ , with $\alpha = 1$ except for per-user LaMP adaptation.
<table><tr><td>Experiment</td><td>Model</td><td>Update</td><td>μrel</td><td>Step</td><td>Pairs</td><td>Batch</td><td>Updates</td><td> $\mathbf { L } _ { k }$ </td></tr><tr><td>Countdown</td><td>Qwen2.5 0.5B–7B, Llama-3.2-1B</td><td>ES</td><td> $3 \times 1 0 ^ { - 2 }$ </td><td>α = 1</td><td>15</td><td>200</td><td>500</td><td>FP32</td></tr><tr><td>MATH</td><td>Qwen2.5-Math-7B</td><td>ES</td><td> $3 \times 1 0 ^ { - 2 }$ </td><td>α = 1</td><td>15</td><td>1,024</td><td>200</td><td>FP32</td></tr><tr><td>Countdown (INT4)</td><td>Qwen2.5 1.5B, 3B</td><td>ES</td><td> $3 \times 1 0 ^ { - 2 }$ </td><td>α = 1</td><td>25</td><td>200</td><td>300</td><td>FP32</td></tr><tr><td>SFT</td><td>Llama-3-8B</td><td>ZO-Adam</td><td>10⁻2</td><td>η = 0.5</td><td>8</td><td>16</td><td>5,987</td><td>BF16/NF4</td></tr><tr><td>Sudoku</td><td>Qwen3.5-4B</td><td>ES</td><td> $3 \times 1 0 ^ { - 2 }$ </td><td>α = 1</td><td>16</td><td>32</td><td>100</td><td>FP32</td></tr><tr><td>LaMP (U)</td><td>Llama-3.1-8B, Qwen3.5-4B</td><td>ES</td><td> $3 \times 1 0 ^ { - 2 }$ </td><td>α = 1</td><td>15</td><td>128</td><td>200</td><td>BF16</td></tr><tr><td>LaMP (T)</td><td>Llama-3.1-8B, Qwen3.5-4B</td><td>ES</td><td> $3 \times 1 0 ^ { - 2 }$ </td><td>α = 3</td><td>15</td><td>≤ 64</td><td>≤ 20</td><td>BF16</td></tr></table>

## C.1 RLVR on Countdown

Experimental protocol. We follow the Countdown benchmark of ES at Scale (Qiu et al., 2026), which contains 2,200 arithmetic reasoning problems. The first 200 problems are used for training and the remaining 2,000 for evaluation. At each iteration, all population members are evaluated on the same 200 training problems.

The prompt follows the plain-text format of ES at Scale without a chat template, requesting reasoning within <think> tags and a final expression within <answer> tags. The reward combines answer correctness and format compliance:

$$
r = r _ { \mathrm { a n s w e r } } + 0 . 1 r _ { \mathrm { f o r m a t } } .\tag{13}
$$

The answer reward equals one when the final expression uses exactly the given numbers and evaluates to the target, and zero otherwise. The format reward equals one for the required reasoning-and-answer structure. Otherwise, a reasoning block receives 0.1 and an answer block receives 0.5.

Training and evaluation use greedy decoding with at most 1,024 generated tokens. Each population member receives the mean reward across the 200 training problems. Test accuracy is the fraction of problems with a correct final answer.

FuRA-S and full-parameter ES. FuRA-S trains 245,760 block singular values on Qwen2.5-0.5B, 327,680 on Llama-3.2- 1B, 508,928 on Qwen2.5-1.5B, 838,656 on Qwen2.5-3B, and 1,132,544 on Qwen2.5-7B. All five models use 15 antithetic pairs per iteration, $\mu _ { \mathrm { r e l } } = 3 \times 1 0 ^ { - 2 }$ , and $\alpha = 1$ for 500 iterations. We select these hyperparameters on Qwen2.5-1.5B and apply them unchanged to the other models.

The frozen spectral factors are stored in FP32, and the reconstructed weights are evaluated in BF16. For full-parameter ES, we reproduce ES at Scale using 30 independent Gaussian perturbations per iteration, with $\sigma = 1 0 ^ { - 3 }$ and $\alpha = 5 \times 1 0 ^ { - 4 }$ Full-parameter ES maintains an FP32 master copy of the model weights. Both methods evaluate 30 perturbed models per iteration, so each run generates 3,000,000 training responses.

LoRA-ES and VeRA-ES comparison. To compare the optimization spaces under the same ES procedure, we additionally optimize rank-4 LoRA adapters on the seven linear projections of each decoder layer. The adapters contain between 2.2M and 10.1M trainable parameters across the evaluated models, with the LoRA B factors initialized to zero.

LoRA-ES uses 15 antithetic pairs per iteration, $\mu _ { \mathrm { r e l } } = 3 \times 1 0 ^ { - 2 }$ , and α = 1, with the perturbation magnitude matched to FuRA-S for each weight matrix. Both methods use the same evaluation budget. The Qwen2.5-1.5B comparison is averaged over two runs, with seeds 42 and 1. The remaining model configurations use seed 42.

VeRA-ES applies VeRA to the same projections. Frozen random matrices of rank 1024 are shared across all layers, and each projection trains a scaling vector d initialized to 0.1 and a scaling vector b initialized to zero, which gives between 0.48M and 1.59M trainable parameters across the evaluated models. VeRA-ES uses the settings of LoRA-ES, with the perturbation of b matched to FuRA-S for each weight matrix and that of d set to $\mu _ { \mathrm { r e l } }$ times its initial value, and every model uses seed 42.

## C.2 RLVR on MATH

Table A3. Greedy pass@1 (%) of Qwen2.5-Math-7B on five math benchmarks under three checkpoint selections, with FuRA-S minus Full ES in parentheses. Avg-5 averages all five benchmarks, and Avg-3 averages MATH500, Minerva, and OlympiadBench.
<table><tr><td>Selection</td><td>Method</td><td>Iter.</td><td>AIME24</td><td>AMC</td><td>MATH500</td><td>Minerva</td><td>Olympiad</td><td>Avg-5</td><td>Avg-3</td></tr><tr><td rowspan="3">Final</td><td>Full ES</td><td>200</td><td>26.7</td><td>57.8</td><td>77.6</td><td>26.8</td><td>37.0</td><td>45.2</td><td>47.2</td></tr><tr><td>LoRA-ES</td><td>200</td><td>16.7</td><td>53.0</td><td>74.0</td><td>22.8</td><td>35.4</td><td>40.4</td><td>44.1</td></tr><tr><td>FuRA-S</td><td>200</td><td>23.3</td><td>54.2</td><td>76.8</td><td>26.5</td><td>40.0</td><td>44.2 (-1.0)</td><td>47.8(+0.6)</td></tr><tr><td rowspan="3">Best Avg-5</td><td>Full ES</td><td>130</td><td>33.3</td><td>57.8</td><td>76.8</td><td>28.7</td><td>39.7</td><td>47.3</td><td>48.4</td></tr><tr><td>LoRA-ES</td><td>120</td><td>30.0</td><td>53.0</td><td>73.2</td><td>23.2</td><td>35.3</td><td>42.9</td><td>43.9</td></tr><tr><td>FuRA-S</td><td>185</td><td>23.3</td><td>56.6</td><td>77.2</td><td>29.4</td><td>40.1</td><td>45.3(-1.9)</td><td>48.9(+0.5)</td></tr><tr><td rowspan="3">Best Avg-3</td><td>Full ES</td><td>170</td><td>16.7</td><td>59.0</td><td>78.2</td><td>30.5</td><td>39.0</td><td>44.7</td><td>49.2</td></tr><tr><td>LoRA-ES</td><td>190</td><td>13.3</td><td>51.8</td><td>74.6</td><td>22.8</td><td>36.9</td><td>39.9</td><td>44.8</td></tr><tr><td>FuRA-S</td><td>185</td><td>23.3</td><td>56.6</td><td>77.2</td><td>29.4</td><td>40.1</td><td>45.3(+0.7)</td><td>48.9 (−0.3)</td></tr></table>

Experimental protocol. We follow the MATH training protocol of ES at Scale (Qiu et al., 2026), using 8,523 problems of difficulty levels 3–5 with Qwen2.5-Math-7B. Prompts use the model’s chat template and instruct it to reason step by step and place the final answer in . A response receives a binary reward of one if its final boxed answer is correct under the verifier of ES at Scale, and zero otherwise.

Each iteration evaluates 30 population members on the same 1,024 training prompts, selected sequentially from a fixed random permutation of the training set. Generation uses greedy decoding with a maximum of 3,000 new tokens.

We evaluate greedy pass@1 on AIME24 (30 problems), AMC (83), MATH500 (500), Minerva (272), and OlympiadBench (675), using the same answer verifier and generation limit. We evaluate at initialization and every five iterations. Table A3 reports the final checkpoint at iteration 200 and the best checkpoints selected according to the five-benchmark average (Avg-5) or the average over MATH500, Minerva, and OlympiadBench (Avg-3). Avg-3 covers 1,447 problems and leaves out the two smallest sets, AIME24 and AMC.

Additional output-side coordinates. For the MATH experiment, FuRA-S adds a small set of trainable LM-head offsets to its block-spectral coordinates. These coordinates are constructed from the model’s pretrained output weights and the dominant directions of the hidden states associated with answer generation.

We first generate responses to 512 randomly selected training problems using the pretrained model. From the resulting answers, we identify the 2,048 most frequent answer tokens, accounting for 98.5% of observed answer-token occurrences.

We also compute the leading 64 eigenvectors of the hidden-state second-moment matrix

$$
\sum _ { h } \mathbf { h } \mathbf { h } ^ { \top } ,\tag{14}
$$

where $\mathbf { h } \in \mathbb { R } ^ { d _ { h } }$ denotes a hidden state entering the language-model head. These directions account for 83% of the measured hidden-state energy.

Let $\mathbf { V } \in \mathbb { R } ^ { 6 4 \times d _ { h } }$ contain the selected eigenvectors and let $\mathbf { H } \in \mathbb { R } ^ { 2 0 4 8 \times d _ { h } }$ denote the LM-head weight rows corresponding to the selected answer tokens. We introduce a trainable matrix $\mathbf { C } \in \mathbb { R } ^ { 2 0 4 8 \times 6 4 }$ , initialized to zero, and parameterize the selected output weights as

$$
\mathbf { H } ^ { \prime } = \mathbf { H } + \mathbf { C } \mathbf { V } .\tag{15}
$$

The ES procedure jointly optimizes C and the block singular values. Perturbations of C are scaled elementwise by $\mu _ { \mathrm { r e l } } | \mathbf { H } \mathbf { V } ^ { \top } |$ , following the relative scaling of the block singular values.

The resulting trainable state contains 131,072 output-side coordinates and 1,132,544 block singular values, for a total of 1,263,616 parameters.

Optimization and computational cost. FuRA-S, LoRA-ES, and full-parameter ES are trained for 200 iterations using the same sequence of training prompts. FuRA-S uses 15 antithetic pairs, $\mu _ { \mathrm { r e l } } = 3 \times 1 0 ^ { - 2 }$ , and $\alpha = 1$ , with seed 42. Its spectra factors are stored in FP32 and reconstructed into BF16 weights for generation with vLLM.

Our full-parameter ES reproduction uses 30 independent perturbations with $\sigma = 1 0 ^ { - 3 }$ and $\alpha = 5 \times 1 0 ^ { - 4 }$ , also with seed 42. It maintains an FP32 master copy of the model weights and evaluates perturbed models in BF16. LoRA-ES uses the rank-4 adapter configuration and perturbation procedure described in Appendix C.1, with 10.1M trainable parameters.

On six NVIDIA A100 40GB GPUs, FuRA-S requires 732 seconds per iteration, compared with 739 seconds for fullparameter ES, as generation dominates both. The trainable state of FuRA-S occupies 5.1 MB, whereas full-parameter ES maintains a 30.5 GB FP32 master copy.

## C.3 RLVR on Quantized Models

Experimental protocol. We follow the released Countdown protocol of QES (Xu et al., 2026), using the same 200 training problems and 2,000 test problems as the full-precision experiments (Appendix C.1). Both methods use identical Qwen2.5 checkpoints, prompt templates, reward definitions, and greedy decoding with a maximum of 1,024 generated tokens. The reward combines answer correctness with a weighted format reward, following the QES implementation.

Each iteration evaluates 50 population members on all 200 training problems. The unperturbed model is evaluated on the 2,000 test problems every 25 iterations, and results are reported after 300 training iterations. Evaluating the released INT4 checkpoints before adaptation with vLLM’s GPTQ kernels yields initial accuracies of 3.45% and 2.65% for Qwen2.5-1.5B and Qwen2.5-3B, against the 3.50% and 2.80% reported by QES.

FuRA-S on released INT4 checkpoints. We adapt the released GPTQ-INT4 checkpoints of Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct, using the same quantized models as QES. Decoder projections use groupwise 4-bit quantization with FP16 scales and a group size of 128 input channels. Embeddings, normalization layers, and biases remain in FP16.

FuRA-S constructs its block-spectral coordinates from the dequantized weights, using the partitioning rule of Section 3.2 with block widths ranging from 48 to 128. The decomposition reconstructs the dequantized weights with relative error below $4 \times 1 0 ^ { - 5 }$ . The original INT4 weights and quantization parameters remain unchanged. The frozen R is stored in 4 bits, by symmetric round-to-nearest with one FP16 scale per column of each block.

The two models contain 508,928 and 838,656 trainable singular values, respectively. FuRA-S uses 25 antithetic pairs per iteration, which match the 50 members of QES, and the $\mu _ { \mathrm { r e l } } = 3 \times 1 0 ^ { - 2 }$ and $\alpha = 1$ of the other reward-based experiments.

During training, perturbed weights are reconstructed from their spectral factors and evaluated in BF16 using vLLM. Section 5.2 and Table A4 report accuracy under this evaluation. The INT4 deployment formulation of Section 3.3 instead applies the learned adaptation while retaining the released quantized weights and GPTQ kernels. Its memory and inferencetime costs are reported in Appendix D. On the first 200 test problems, the deployed checkpoints at iteration 300 solve 23.5% and 46.0%, against 22.5% and 46.0% in the BF16 evaluation, with the same outcome on 198 and 188 problems.

![](images/9ce673a872fcee3c857d556a0b29995102e0a6618b180dedf1babfc66dfe50cc.jpg)  
Figure 9. Countdown accuracy of the INT4 checkpoints over iterations, with the best accuracy of QES dotted.

QES baseline. We reproduce QES using its released implementation and original optimization settings. Full-parameter QES is evaluated on two A100 40GB GPUs for Qwen2.5-1.5B and six for Qwen2.5-3B.

At iteration 300, the reproduced QES checkpoints achieve 16.00% and 34.15% accuracy, compared with 16.00% and 31.85% in the original QES report. QES reaches its best observed accuracy of 16.75% after 201 updates on Qwen2.5-1.5B and 34.15% after 300 updates on Qwen2.5-3B.

FuRA-S surpasses both values by iteration 50, reaching 17.70% and 39.20%, respectively (Table A4). QES evaluates the model after 1, $2 6 , \ldots , 2 7 6 ,$ and 300 updates, and Figure 9 plots its points at these update counts.

Table A4. Countdown accuracy (%) of FuRA-S on the GPTQ-INT4 checkpoints over training iterations, evaluated using reconstructed BF16 weights under the QES protocol. FuRA-S exceeds the best accuracy of our QES reproduction, 16.75% and 34.15%, within 50 iterations.
<table><tr><td>Iteration</td><td>0</td><td>25</td><td>50</td><td>100</td><td>150</td><td>200</td><td>250</td><td>300</td></tr><tr><td>Qwen2.5-1.5B</td><td>3.55</td><td>15.95</td><td>17.70</td><td>18.90</td><td>19.50</td><td>20.10</td><td>21.70</td><td>21.85</td></tr><tr><td>Qwen2.5-3B</td><td>3.25</td><td>27.15</td><td>39.20</td><td>43.75</td><td>46.65</td><td>48.00</td><td>49.85</td><td>50.40</td></tr></table>

## C.4 Mobile GUI Agents

Environment and model. The task runs in a MobileWorld Android emulator in Docker (Kong et al., 2026), reset before every episode. The agent is MAI-UI-2B with its released system prompt, which asks for a thought in <thinking> tags and one mobile use tool call per step, with click, long press, type, swipe, open, drag, system button, wait, terminate, answer, and ask-user actions. Each step sees the task, the action history, and the recent screenshots. An episode ends on terminate or answer, or after 30 steps, and the task’s checker scores the final device state with a partial credit between 0 and 1. An episode succeeds when its credit is 1. Generation runs in vLLM.

Step-wise fitness. After each episode, every step is reduced to a screenshot hash and an action bucket. The hash is the 256-bit difference hash of the screenshot the agent acted on (grayscale, resized to 17  16, signs of the horizontal gradient). The bucket is the action type with the 108-pixel grid cell for click-like actions, the direction for swipes, and the lower-cased argument for text, open-app, and key actions. Anchor groups are formed greedily over the pooled steps of the iteration: a step joins the first group whose representative lies within a Hamming distance of 10, and steps in a group visited by fewer than two distinct episodes receive zero advantage. We use $\gamma = 0 . 9 5$ . An iteration pools about 1,260 steps into 136 groups on average, 76 of them shared by at least two episodes, and 52% of steps receive a nonzero advantage.

Optimization. FuRA-S trains the block singular values of every linear projection of the language model, 516,096 values, with 8 antithetic pairs (N = 16), J = 4 greedy episodes per member, fitness standardized over the population, and α = 1, for 15 iterations. Greedy members still produce different episodes, because the emulator’s loading times split their trajectories: in the first iteration, members succeed on 0 to 3 of their 4 episodes. Two vLLM servers share one GPU and drive 12 emulators. Malformed tool calls stay at or below 1.3% of steps. Each task trains on its own GPU with the same settings, and an iteration takes 15 to 51 minutes, almost all of it rollouts. Episode-reward ES uses the same settings, with members sampled at temperature 0.7 under common random numbers per episode index, and scores each member by its mean partial credit over its 4 episodes.

Evaluation. Each trained model is rebuilt from its perturbation seeds and update coefficients and runs 64 new episodes. Base and trained models are evaluated back to back on the same 12 emulators. The trained model completes MastodonReply in 11.5 steps on average. During training, the current model is also evaluated every 5 iterations on 16 episodes.

## C.5 Personalization on LaMP

Experimental protocol. We evaluate personalization on six LaMP tasks (Salemi et al., 2024) under two splits. The user-based split (U) follows the official LaMP partition. FuRA-S trains on the training users, selects its checkpoint on the development users, and is tested on 2,500, 870, 2,500, 2,376, 2,500, and 1,496 queries for LaMP-1, 2, 3, 4, 5, and 7, respectively. Prompts follow the LaMP templates and include the top BM25 item from the user’s own profile within a budget of 512 tokens, whereas the model without the profile receives the input alone.

The time-based split (T) follows the release of OPPU (Tan et al., 2024). For each task, we take the 100 users with the longest histories (101 for LaMP-3 and LaMP-5), which yields 123, 3,302, 112, 6,725, 107, and 109 test queries for LaMP-1, 2M, 3 4, 5, and 7, respectively. Prompts follow the OPPU templates, with each field truncated to 768 words. Both splits apply the chat template of each model, with thinking disabled for Qwen3.5-4B, and limit sequences to 2,048 tokens.

All methods sample one answer per query with temperature 0.1, top-k 10, and top-p 0.9, generating at most 200 new tokens. We report accuracy and macro-F1 for LaMP-1, 2, and 2M, mean absolute error (MAE) and root mean squared error (RMSE) for LaMP-3, and ROUGE-1 and ROUGE-L for LaMP-4, 5, and 7.

Optimization settings. FuRA-S uses 15 antithetic pairs and $\mu _ { \mathrm { r e l } } = 3 \times 1 0 ^ { - 2 }$ in both splits, with the mean log-likelihood of the target tokens as fitness. In the user-based split, each task trains one model for 200 iterations with α = 1, and each iteration evaluates 128 random training examples. The unperturbed model is scored on 256 fixed development examples at initialization and every 20 iterations, and the best state is retained.

In the time-based split, each user’s checkpoint starts from the base model and runs for at most 20 iterations with α = 3, each on up to 64 examples from the user’s history. The latest 10% of the history, with at least 8 records and at most half of the history, serves as validation. Training stops after three iterations without improvement, and the best state answers the test queries. The frozen $\mathbf { L } _ { k }$ factors are stored in BF16, while s<sub>k</sub> and ${ \bf R } _ { k }$ are maintained in FP32.

Per-user LoRA baseline. We reimplement the per-user stage of OPPU. Each user receives a new rank-8 LoRA adapter on the query and value projections, trained from the base model for two epochs over the user’s history. Training uses AdamW with a learning rate of 10<sup>−4</sup>, weight decay of 0.01, batch size 16, LoRA scale 8, dropout 0.05, 10% warm-up followed by linear decay, and gradient clipping at 0.3.

Computational cost. Each setting runs once with seed 42 on nodes with four NVIDIA A100 40GB GPUs. On Llama-3.1- 8B-Instruct, a user-based model takes approximately 0.9 node-hours per task and up to 2.0 node-hours for LaMP-3.

## D SYSTEM COST

Measurement protocol. We measure training memory and runtime on Llama-3-8B under the Commonsense-170K setting (Appendix E), using an NVIDIA A100 40GB GPU. Peak memory is measured over training steps using the same 128 examples and 16-example mini-batches for all methods. Runtime includes perturbation, forward evaluation, and parameter updates, measured over 384 training examples. We evaluate both length-grouped batches and conventionally padded batches to distinguish optimization costs from sequence-padding overhead. GRZO evaluates its loss per example and therefore runs only with padded batches.

Table A5. Peak training memory and time per 16-example forward evaluation on Llama-3-8B using one A100 40GB GPU. Runtime includes perturbation and parameter updates, with length-grouped or padded batches.
<table><tr><td>Method</td><td>Peak memory (GB)</td><td>Grouped (s)</td><td>Padded (s)</td></tr><tr><td>FuRA-S</td><td>17.17</td><td>0.341</td><td>0.570</td></tr><tr><td>FuRA-S (4-bit L)</td><td>7.52</td><td>0.358</td><td>0.579</td></tr><tr><td>MeZO</td><td>16.26</td><td>0.318</td><td>0.481</td></tr><tr><td>LOZO</td><td>16.26</td><td>0.265</td><td>0.427</td></tr><tr><td>GRZO</td><td>20.90</td><td></td><td>1.028</td></tr><tr><td>Sparse-MeZO</td><td>30.90</td><td>0.408</td><td>0.570</td></tr><tr><td>VeRA</td><td>16.93</td><td>0.379</td><td>0.656</td></tr><tr><td>Uni-LoRA</td><td>16.49</td><td>0.261</td><td>0.436</td></tr><tr><td>QuZO (4-bit)</td><td>6.71</td><td>0.325</td><td>0.493</td></tr></table>

Training memory. Forward-only training stores no activations for backpropagation. FuRA-S further limits its trainable state to the block singular values and their optimizer states, requiring approximately 20 MB beyond forward evaluation. With BF16 singular factors, peak training memory reaches 17.17 GB, compared with 17.15 GB for forward evaluation alone. Quantizing the frozen left singular factors to NF4 reduces training memory to 7.52 GB, approximately 46% of the memory required by full-parameter MeZO (Table A5).

Training time. FuRA-S has forward-evaluation costs comparable to other zeroth-order methods. On length-grouped mini-batches, a forward evaluation including perturbation and update takes 0.341 seconds, increasing to 0.358 seconds with NF4 left singular factors. On MATH, an iteration takes 732 seconds for FuRA-S and 739 seconds for full-parameter ES on six A100 40GB GPUs, because generation dominates its cost.

Adaptation storage. For full-precision models, the adapted blocks can be merged into dense weights without introducing additional inference operations. Alternatively, multiple adaptations can share the same frozen factors and store only their optimized block singular values. A FuRA-S checkpoint for Llama-3-8B occupies 4.98 MB in FP32, compared with 16.1 GB for a BF16 copy of the entire model. For the INT4 checkpoints of Section 5.2, the released 4-bit weights remain unchanged, and the 4-bit R and FP32 block singular values raise the storage of the projections from 4.16 to 4.29–4.31 bits per weight.

INT4 inference overhead. We evaluate the released GPTQ-INT4 Qwen2.5-1.5B and Qwen2.5-3B checkpoints from Section 5.2 on one NVIDIA A100 40GB GPU. All configurations use the same GPTQ kernels in vLLM (Kwon et al., 2023), with 4-bit weights and FP16 activations. Decoding is measured over 256 generated tokens following a 128-token prompt at batch size one, using a key-value cache and captured CUDA graphs. Prefill is measured using one or eight prompts of 512 tokens.

We compare the original INT4 models, the deployed FuRA-S formulation, and two training-time representations with FP16 or FP32 right singular factors. The deployed formulation folds the learned blockwise correction into a single batched operation preceding each original INT4 weight multiplication. During training, the right singular factors and block singular values remain separate, requiring two batched operations.

Relative to the INT4 baseline, the deployed formulation increases GPU memory by approximately 9% and decoding latency by 11–15%, while retaining a memory footprint 2.4–2.8 times smaller than FP16. Prefill incurs a larger overhead, approximately 50% for a single prompt and 18–21% for eight prompts (Table A6).

Table A6. INT4 inference cost on one A100 40GB GPU. Memory is reported in GB, decoding latency per token in milliseconds, and prefill latency in milliseconds for one or eight prompts. The deployed FuRA-S formulation retains the original GPTQ weights and introduces approximately 9% additional memory relative to INT4.
<table><tr><td rowspan="2">Model form</td><td colspan="4">Qwen2.5-1.5B</td><td colspan="4">Qwen2.5-3B</td></tr><tr><td>Memory</td><td>Decode</td><td>Prefill 1</td><td>Prefill 8</td><td>Memory</td><td>Decode</td><td>Prefill 1</td><td>Prefill 8</td></tr><tr><td>FP16</td><td>2.88</td><td>6.93</td><td>23.9</td><td>80.5</td><td>5.85</td><td>10.58</td><td>31.9</td><td>156.9</td></tr><tr><td>INT4</td><td>1.09</td><td>5.74</td><td>31.2</td><td>87.4</td><td>1.94</td><td>7.95</td><td>42.2</td><td>160.8</td></tr><tr><td>INT4 + FuRA-S, deployed</td><td>1.19</td><td>6.58</td><td>46.6</td><td>105.6</td><td>2.11</td><td>8.83</td><td>63.5</td><td>189.1</td></tr><tr><td>INT4 + FuRA-S, training, FP16 Rk</td><td>1.20</td><td>7.94</td><td>56.7</td><td>125.2</td><td>2.12</td><td>10.34</td><td>74.4</td><td>220.0</td></tr><tr><td>INT4 + FuRA-S, training, FP32 Rk</td><td>1.30</td><td>8.92</td><td>62.8</td><td>209.0</td><td>2.28</td><td>11.85</td><td>81.7</td><td>355.7</td></tr></table>

## E SUPERVISED FINE-TUNING

For supervised fine-tuning on Commonsense-170K, FuRA-S keeps the perturbations of Section 3.2 and replaces the ES update with the two-point gradient estimate of MeZO over P antithetic pairs, each evaluated on its own mini-batch $B _ { j }$

$$
\hat { \bf g } _ { t } = \frac { 1 } { P } \sum _ { j = 1 } ^ { P } \frac { \mathcal { L } ( \pmb { \theta } _ { t } + \mathbf { D } \mathbf { z } _ { j } ; \mathcal { B } _ { j } ) - \mathcal { L } ( \pmb { \theta } _ { t } - \mathbf { D } \mathbf { z } _ { j } ; \mathcal { B } _ { j } ) } { 2 } \mathbf { z } _ { j } ,\tag{16}
$$

to which it applies Adam (Kingma & Ba, 2015) before scaling the update by D.

We compare it with full-parameter ZO methods, parameter-efficient baselines, and 4-bit QuZO (Zhou et al., 2025b) under the same budget of forward evaluations.

Experimental protocol. We fine-tune Llama-3-8B on Commonsense-170K from LLM-Adapters (Hu et al., 2023), following the prompt and answer format of FuRA (Zhao et al., 2026). We reserve 120 training examples for validation and optimize the mean negative log-likelihood of response tokens, with a maximum sequence length of 2,048 tokens. All methods use the same sequence of 16-example mini-batches drawn from a fixed random permutation of the training set.

Evaluation follows FuRA, using beam search with four beams and a maximum of 256 generated tokens. The first label word in the generated response determines the predicted answer. We evaluate the final checkpoint on PIQA (1,838 examples), SIQA (1,954), HellaSwag (10,042), WinoGrande (1,267), ARC-e (2,376), ARC-c (1,172), and OBQA (500). Each method is evaluated in one run with seed 43 on a single A100 40GB GPU.

Training budget. We match the computational budget of first-order FuRA, which performs three training epochs with 31,932 updates on 16-example mini-batches. Approximating one forward-backward update as three forward evaluations yields a budget of 95,796 forward evaluations, equivalent to 47,898 antithetic perturbation pairs.

All ZO methods in Table A8 receive this budget. Baselines using one perturbation pair per update perform 47,898 updates, while FuRA-S evaluates eight pairs per update for 5,987 updates. In the full-precision setting, each FuRA-S pair uses a separate mini-batch, resulting in the same 4.5 effective data passes as the ZO baselines. In the 4-bit frozen-factor setting, al eight pairs within an update share the same mini-batch.

Optimization settings. FuRA-S uses a relative perturbation scale of $\mu _ { \mathrm { r e l } } ~ = ~ 1 0 ^ { - 2 }$ and ZO-Adam with $\beta _ { 1 } ~ = ~ 0 . 9$ $\beta _ { 2 } = 0 . 9 9 9$ , and $\epsilon = 1 0 ^ { - 8 }$ . The peak learning rate is 0.5, with linear warm-up over the first 3% of updates followed by linear decay to zero.

For full-precision adaptation, the frozen $\mathbf { L } _ { k }$ factors are stored in BF16, while $\mathbf { s } _ { k }$ and ${ \bf R } _ { k }$ are maintained in FP32. In the 4-bit frozen-factor variant, $\mathbf { L } _ { k }$ is stored using NF4 with double quantization and a block size of 64. The block singular values and right singular factors remain in FP32.

Table A7 provides the optimization settings for all baselines. VeRA and Uni-LoRA have no zeroth-order recipe of their own, so we train them with the ZO-SGD procedure of MeZO.

Table A7. Optimization settings for Commonsense-170K. LR denotes the learning rate, with peak values for FuRA-S, the initial value for QuZO, and constant values otherwise. All methods use the same budget of 47,898 antithetic perturbation pairs.
<table><tr><td>Method</td><td>Update</td><td>LR</td><td>Perturbation</td><td>Other settings</td></tr><tr><td>FuRA-S</td><td>ZO-Adam</td><td>0.5</td><td> $\mu _ { \mathrm { r e l } } = 1 0 ^ { - 2 }$ </td><td>per-pair batches, BF16 Lk</td></tr><tr><td>FuRA-S (4-bit L)</td><td>ZO-Adam</td><td>0.5</td><td> $\mu _ { \mathrm { r e l } } = 1 0 ^ { - 2 }$ </td><td>shared batch, NF4 Lk</td></tr><tr><td>GRZO</td><td>GRZO</td><td> $5 \times 1 0 ^ { - 8 }$ </td><td> $\epsilon = { 1 0 } ^ { - 3 }$ </td><td>per-example perturbations, FP16</td></tr><tr><td>LOZO</td><td>ZO-SGD</td><td> $1 0 ^ { - 7 }$ </td><td> $\epsilon = 1 0 ^ { - 3 }$ </td><td>rank  $2 , \nu = 5 0 , { \mathrm { F P } } 1 6$ </td></tr><tr><td>MeZO</td><td>ZO-SGD</td><td> $1 0 ^ { - 7 }$ </td><td> $\epsilon = { 1 0 } ^ { - 3 }$ </td><td>FP16</td></tr><tr><td>VeRA</td><td>ZO-SGD</td><td> $5 \times 1 0 ^ { - 3 }$ </td><td> $\epsilon = 1 0 ^ { - 2 }$ </td><td> $r = 1 0 2 4 , d _ { 0 } = 0 . 1 , \mathrm { F P 1 6 }$ </td></tr><tr><td>Uni-LoRA</td><td>ZO-SGD</td><td> $1 0 ^ { - 4 }$ </td><td> $\epsilon = 1 0 ^ { - 2 }$ </td><td> $d = 5 2 4 , 2 8 8 .$  LoRA rank 4, FP16</td></tr><tr><td>Sparse-MeZO</td><td>ZO-SGD</td><td> $1 0 ^ { - 6 }$ </td><td> $\epsilon = { 1 0 } ^ { - 3 }$ </td><td>sparsity 0.9, FP16</td></tr><tr><td>QuZO</td><td>ZO-SGD</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td> $\epsilon = 2 \times 1 0 ^ { - 3 }$ </td><td>LoRA rank 8 on q, v, NF4 base</td></tr></table>

Results. FuRA-S achieves 64.64% average accuracy while training only 1.25M parameters (0.0155% of the model), outperforming all parameter-efficient baselines (Table A8). Across the seven benchmarks, it ranks second only to GRZO on six tasks and achieves the highest accuracy on WinoGrande. It exceeds VeRA by 2.98 points and full-parameter MeZO by 9.93 points. With its frozen left singular factors stored in 4 bits, FuRA-S retains 59.97% accuracy, outperforming QuZO by 16.97 points.  
Table A8. Accuracy (%) on seven tasks after SFT of Llama-3-8B on Commonsense-170K with the same forward-pass budget, best per group in bold. FuRA-S is the most accurate PEFT method and leads QuZO by 16.97 points in 4 bits.
<table><tr><td>Method</td><td>Trainable</td><td>Trainable %</td><td>PIQA</td><td>SIQA</td><td>HellaS.</td><td>WinoG.</td><td>ARC-e</td><td>ARC-c</td><td>OBQA</td><td>Avg</td></tr><tr><td colspan="9">Full-parameter updates</td></tr><tr><td>GRZO</td><td>8.03B</td><td>100</td><td>78.7</td><td>64.4</td><td>74.3</td><td>53.0</td><td>84.2</td><td>65.7</td><td>60.6</td><td>68.70</td></tr><tr><td>LOZO</td><td>8.03B</td><td>100</td><td>70.8</td><td>53.2</td><td>45.0</td><td>49.3</td><td>68.8</td><td>53.0</td><td>52.0</td><td>56.01</td></tr><tr><td>MeZO</td><td>8.03B</td><td>100</td><td>76.2</td><td>55.2</td><td>40.5</td><td>51.5</td><td>67.0</td><td>51.8</td><td>40.8</td><td>54.71</td></tr><tr><td colspan="9">PEFT updates</td></tr><tr><td>FuRA-S</td><td>1.25M</td><td>0.0155</td><td>78.0</td><td>56.2</td><td>66.3</td><td>53.4</td><td>79.7</td><td>61.3</td><td>57.6</td><td>64.64</td></tr><tr><td>VeRA</td><td>1.61M</td><td>0.0200</td><td>76.3</td><td>54.0</td><td>60.1</td><td>50.7</td><td>77.2</td><td>60.7</td><td>52.6</td><td>61.66</td></tr><tr><td>Uni-LoRA</td><td>0.52M</td><td>0.0065</td><td>76.3</td><td>52.1</td><td>45.1</td><td>52.2</td><td>74.8</td><td>56.1</td><td>54.8</td><td>58.77</td></tr><tr><td>Sparse-MeZO</td><td>0.80B</td><td>10</td><td>74.0</td><td>47.4</td><td>38.6</td><td>51.2</td><td>71.0</td><td>57.8</td><td>53.8</td><td>56.26</td></tr><tr><td colspan="9">4-bit comparison</td></tr><tr><td>FuRA-S (4-bit L)</td><td>1.25M</td><td>0.0155</td><td>73.7</td><td>54.0</td><td>53.8</td><td>52.9</td><td>72.5</td><td>58.5</td><td>54.4</td><td>59.97</td></tr><tr><td>QuZO (4-bit)</td><td>3.41M</td><td>0.0424</td><td>53.1</td><td>45.3</td><td>31.7</td><td>50.1</td><td>48.9</td><td>40.3</td><td>31.6</td><td>43.00</td></tr></table>

## F LONG-HORIZON AGENTIC OPTIMIZATION

We evaluate FuRA-S on a long-horizon agentic task, where the agent acts over up to 45 turns and receives a single reward only when it completes the task.

Experimental protocol. We follow the multi-turn Sudoku benchmark of Agentic ESOpt (Zheng et al., 2026), using its 15-cell completion setting. Each agent must fill 15 missing cells of a 9 9 board over at most 45 turns, with a maximum of 64 generated tokens per turn. A reward is provided only when the completed board is valid. The benchmark contains 32 training puzzles and 32 held-out puzzles.

We evaluate Qwen3.5-4B in BF16 with thinking disabled. The model has 32 decoder layers comprising attention and gated-DeltaNet components. FuRA-S adapts the linear projections in both components and the MLPs, covering 248 weight matrices. Each population member is scored by its success rate over the 32 training puzzles, and fitness is standardized across the population. Generation uses temperature 0.7, top-p 0.8, top-k 20, and presence penalty 1.5. Held-out performance is evaluated every 10 iterations with three repeated evaluations.

Optimization settings and comparison. FuRA-S trains 835,584 block singular values using 16 antithetic pairs per iteration, $\mu _ { \mathrm { r e l } } = 3 \times 1 0 ^ { - 2 }$ , and α = 1. Training runs for 100 iterations. The full-parameter Vanilla ES baseline perturbs all 4.2B model parameters using 32 one-sided Gaussian perturbations per iteration, with $\sigma = \alpha = 5 \times 1 0 ^ { - 4 }$ . Both settings use the same benchmark and number of population evaluations. Agentic ESOpt additionally decays the perturbation scale with a cosine schedule. FuRA-S and the Vanilla ES baseline both use a constant scale, so the comparison isolates the effect of the search space.

Results. FuRA-S matches the performance of full-parameter ES while optimizing only 835,584 block singular values, approximately 0.02% of Qwen3.5-4B’s parameters (Figure 10). Both methods achieve approximately 38% held-out success averaged over the final three evaluations. The trainable state of FuRA-S occupies 3.3 MB, against the 8.4 GB of weights tha full-parameter ES updates.

![](images/6452f6432f65c2c9bc1f7fbaaaea97eb4a60caa79ff4a6d9b89e13d57ed92cfe.jpg)

![](images/ee860db960d89183a8e304fd195f4a9f6af7188ea1baa48c47db5df9620824e6.jpg)  
Figure 10. Agentic Sudoku with Qwen3.5-4B: (a) held-out success, final and over the last three evaluations, with standard deviations; (b) trained parameters. FuRA-S is on par with full-parameter ES while training 5,000 times fewer parameters. Results for full-parameter ES are from the released logs of Zheng et al. (2026).

## G LAMP SCORES WITH LLAMA-3.1-8B-INSTRUCT

Tables A9 and A10 report the scores behind Figure 6, together with a secondary metric for each task. In the user-based split, FuRA-S with the retrieved profile improves over the base model with the same profile on all six tasks, with the largest gain on LaMP-5, where ROUGE-1 rises from 0.132 to 0.458. In the time-based split, the per-user FuRA-S checkpoints lead per-user LoRA on LaMP-2M and LaMP-7, whereas per-user LoRA leads on LaMP-1, LaMP-3, LaMP-4, and LaMP-5.

Table A9. LaMP user-based split with Llama-3.1-8B-Instruct, with or without one retrieved item from the user’s profile in the prompt, for FuRA-S in both training and testing. With the profile, FuRA-S improves over the base model on all six tasks.
<table><tr><td>Task</td><td>Metric</td><td>Base</td><td>Base + profile</td><td>FuRA-S</td><td>FuRA-S + profile</td></tr><tr><td>LaMP-1 citation</td><td>accuracy / F1</td><td>0.528 / 0.528</td><td>0.665 / 0.666</td><td>0.515 / 0.515</td><td>0.681 / 0.681</td></tr><tr><td>LaMP-2 movie tag</td><td>accuracy / F1</td><td>0.285 / 0.267</td><td>0.347 / 0.302</td><td>0.446 / 0.404</td><td>0.452 / 0.404</td></tr><tr><td>LaMP-3 rating</td><td>MAE / RMSE↓</td><td>0.383 / 0.697</td><td>0.369 / 0.748</td><td>0.272 / 0.577</td><td>0.254 / 0.564</td></tr><tr><td>LaMP-4 news headline</td><td>ROUGE-1 / ROUGE-L</td><td>0.093 / 0.081</td><td>0.123 / 0.109</td><td>0.165 / 0.150</td><td>0.182 / 0.166</td></tr><tr><td>LaMP-5 paper title</td><td>ROUGE-1 / ROUGE-L</td><td>0.118 / 0.109</td><td>0.132 /0.120</td><td>0.438 / 0.398</td><td>0.458 / 0.418</td></tr><tr><td>LaMP-7 tweet paraphrase</td><td>ROUGE-1 /ROUGE-L</td><td>0.335 / 0.279</td><td>0.335 / 0.284</td><td>0.476 / 0.418</td><td>0.470 / 0.414</td></tr></table>

Table A10. LaMP time-based split with Llama-3.1-8B-Instruct: the base model and per-user LoRA and FuRA-S checkpoints. FuRA-S leads per-user LoRA on LaMP-2M and LaMP-7.
<table><tr><td>Task</td><td>Metric</td><td>Queries (users)</td><td>Base</td><td>Per-user LoRA</td><td>FuRA-S</td></tr><tr><td>LaMP-1 citation</td><td>accuracy / F1</td><td>123 (100)</td><td>0.480 / 0.478</td><td>0.488 / 0.486</td><td>0.480 / 0.470</td></tr><tr><td>LaMP-2M movie tag</td><td>accuracy / F1</td><td>3302 (100)</td><td>0.296 / 0.256</td><td>0.491 / 0.357</td><td>0.541 / 0.422</td></tr><tr><td>LaMP-3 rating</td><td>MAE / RMSE↓</td><td>112 (101)</td><td>0.402 / 0.840</td><td>0.179 / 0.463</td><td>0.259 / 0.701</td></tr><tr><td>LaMP-4 news headline</td><td>ROUGE-1 / ROUGE-L</td><td>6725 (100)</td><td>0.090 / 0.078</td><td>0.164 / 0.147</td><td>0.141 / 0.125</td></tr><tr><td>LaMP-5 paper title</td><td>ROUGE-1 / ROUGE-L</td><td>107 (101)</td><td>0.145 / 0.131</td><td>0.493 / 0.445</td><td>0.476 / 0.415</td></tr><tr><td>LaMP-7 tweet paraphrase</td><td>ROUGE-1 / ROUGE-L</td><td>109 (100)</td><td>0.296 / 0.246</td><td>0.320 / 0.266</td><td>0.338 / 0.290</td></tr></table>

## H LAMP SCORES WITH QWEN3.5-4B

Qwen3.5-4B adds gated-DeltaNet layers to attention layers. The per-user LoRA baseline adapts the query and value projections of attention layers and the fused query-key-value projections of gated-DeltaNet layers.

In the user-based split, FuRA-S with the retrieved profile improves over the base model with the same profile on all six tasks (Figure 11 and Table A11). In the time-based split, FuRA-S leads per-user LoRA on LaMP-1 and LaMP-2M, and per-user LoRA leads on the other four tasks (Table A12). On LaMP-7, the base model, per-user LoRA, and FuRA-S score within 0.004 ROUGE-1 of one another.

## (a) New users (U): one shared model, trained once

![](images/e8c2b54dd0a97b1d03ca1e1fe748c9e295d847fe2d11e17150f72f5fdae41f78.jpg)

![](images/70dd54847323b43bc9f19d7c92f1b1cb1cdc0b2e584d6de9751f17825613e9fe.jpg)

![](images/0883b0d9e3eb5ec83ecf20a7255e67b2308e928bc2146a1a8b86d25cf271a6af.jpg)

![](images/74d36a241be6fc3c122cc01eb0fc07db0da4de14a469f8d3f8f4a48fab46e6ad.jpg)  
base + profile

![](images/9fb459886cd51a2e30fbf9f35cb75e34ca2156b3537c1aeaea867f81f86e8a53.jpg)  
FuRA-S + profile

![](images/0c70e0566495222e5ef960e1f724dbcc4a929e7d41760a76a216d4b98330ad56.jpg)

(b) Per-user checkpoints (T): trained on each user's history  
![](images/83817b8fc0f08c97ca63e457bc2195d31edd94fef5a4a4386fae1dec6c96fdf2.jpg)

![](images/42c1c42d44a5eba7931dac5fcdcc9bd1457afefb99d7ea2996298bd34983962f.jpg)

![](images/b2a86041ca41dd1cda04cf6636620e7b5dfd7a48f4c74930a5c386d480718ae9.jpg)

![](images/342394a02a56dd8c3da2737005ebf817128ed8b3089a88bfd802d951bdc9ac39.jpg)

![](images/f20addaba38b28f142bc82af294f83b16af950ab235652fa4639281b5c00b8f4.jpg)

![](images/54c1e64b836d459781cba13d8c7caba5a523773798a740570779c9ada703fc71.jpg)  
Figure 11. LaMP with Qwen3.5-4B, with the panels of Figure 6. With the profile, FuRA-S again improves all six tasks for new users, and its per-user checkpoints lead per-user LoRA on LaMP-1 and LaMP-2M.

Table A11. LaMP user-based split with Qwen3.5-4B, with or without one retrieved item from the user’s profile in the prompt, for FuRA-S in both training and testing. With the profile, FuRA-S improves over the base model on all six tasks.
<table><tr><td>Task</td><td>Metric</td><td>Base</td><td>Base + profile</td><td>FuRA-S</td><td>FuRA-S + profile</td></tr><tr><td>LaMP-1 citation</td><td>accuracy / F1</td><td>0.500 / 0.511</td><td>0.646 / 0.642</td><td>0.509 / 0.498</td><td>0.671 / 0.670</td></tr><tr><td>LaMP-2 movie tag</td><td>accuracy / F1</td><td>0.372 / 0.304</td><td>0.308 / 0.257</td><td>0.414 / 0.350</td><td>0.454 / 0.388</td></tr><tr><td>LaMP-3 rating</td><td>MAE / RMSE↓</td><td>0.326 / 0.675</td><td>0.381 / 0.826</td><td>0.276 / 0.607</td><td>0.251 / 0.578</td></tr><tr><td>LaMP-4 news headline</td><td>ROUGE-1/ROUGE-L</td><td>0.122 / 0.107</td><td>0.117 / 0.103</td><td>0.153 / 0.139</td><td>0.154 / 0.141</td></tr><tr><td>LaMP-5 paper title</td><td>ROUGE-1/ROUGE-L</td><td>0.110 / 0.099</td><td>0.324 / 0.281</td><td>0.449 / 0.410</td><td>0.460 / 0.420</td></tr><tr><td>LaMP-7 tweet paraphrase</td><td>ROUGE-1/ROUGE-L</td><td>0.452 / 0.396</td><td>0.415 / 0.363</td><td>0.489 / 0.432</td><td>0.472 / 0.419</td></tr></table>

Table A12. LaMP time-based split with Qwen3.5-4B: the base model and per-user LoRA and FuRA-S checkpoints. FuRA-S leads per-user LoRA on LaMP-1 and LaMP-2M.
<table><tr><td>Task</td><td>Metric</td><td>Queries (users)</td><td>Base</td><td>Per-user LoRA</td><td>FuRA-S</td></tr><tr><td>LaMP-1 citation</td><td>accuracy / F1</td><td>123 (100)</td><td>0.431 / 0.446</td><td>0.423 / 0.425</td><td>0.455 / 0.458</td></tr><tr><td>LaMP-2M movie tag</td><td>accuracy / F1</td><td>3302 (100)</td><td>0.319 / 0.248</td><td>0.486 / 0.334</td><td>0.512 / 0.385</td></tr><tr><td>LaMP-3 rating</td><td>MAE / RMSE↓</td><td>112 (101)</td><td>0.411 / 0.732</td><td>0.196 / 0.567</td><td>0.241 / 0.620</td></tr><tr><td>LaMP-4 news headline</td><td>ROUGE-1 / ROUGE-L</td><td>6725 (100)</td><td>0.117 / 0.104</td><td>0.144 / 0.129</td><td>0.128 / 0.115</td></tr><tr><td>LaMP-5 paper title</td><td>ROUGE-1 / ROUGE-L</td><td>107 (101)</td><td>0.502 / 0.422</td><td>0.514 / 0.457</td><td>0.489 / 0.432</td></tr><tr><td>LaMP-7 tweet paraphrase</td><td>ROUGE-1 / ROUGE-L</td><td>109 (100)</td><td>0.471 / 0.421</td><td>0.468 / 0.421</td><td>0.467 / 0.419</td></tr></table>