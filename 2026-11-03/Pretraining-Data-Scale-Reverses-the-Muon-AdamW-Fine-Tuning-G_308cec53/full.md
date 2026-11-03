# Pretraining Data Scale Reverses the Muon–AdamW Fine-Tuning Gap

Catherine Li<sup>∗</sup> Jingchu Gai<sup>∗</sup> Andrej Risteski Jacob Mitchell Springer<sup>†</sup> Aditi Raghunathan<sup>†</sup> Carnegie Mellon University

\# {catheri4, jgai}@andrew.cmu.edu

## Abstract

Muon has recently emerged as a promising optimizer for pretraining large language models, ofering a speedup over AdamW in minimizing pretraining loss. However, it remains unclear whether Muon’s advantage persists after fine-tuning. In this paper, we study the downstream behavior of Muon and AdamW and uncover a surprising trend: beyond a suficiently large pretraining data budget, AdamW outperforms Muon after fine-tuning. Moreover, as model size increases, AdamW overtakes Muon at progressively smaller pretraining token–parameter ratios. To explain this reversal, we view fine-tuning as a perturbation of the outputs and introduce the margin-to-spread ratio, which measures how far the correct logit lies above the wrong ones relative to how widely the wrong logits are spread. In language models, Muon has the larger margin-to-spread ratio at small data budgets and AdamW at large ones, mirroring the fine-tuning trend. In a linear associative memory, we prove that after one update the margin-to-spread ratio of gradient descent gains on that of Muon as the data budget grows, and simulations show that this trend persists after extended training.

## 1 Introduction

The Muon optimizer has recently shown strong empirical performance for pretraining LLMs, challenging AdamW’s status as the standard optimizer for minimizing pretraining loss (Jordan et al., 2024). Prior work has shown that the pretraining optimization choices that minimize pretraining loss do not necessarily yield the best model after fine-tuning (Watts et al., 2026). What ultimately matters downstream is the learning–forgetting tradeof: how well a fine-tuned model learns the target task while retaining its pretrained knowledge. Motivated by this, we ask:

## Does Muon’s pretraining advantage over AdamWextend to the learning–forgetting tradeof?

To answer this question, we measure the Muon–AdamW fine-tuning gap, which compares the two optimizers on this tradeof; a negative gap indicates that the Muon pretrained model dominates the AdamW pretrained model after fine-tuning, and a positive gap indicates the reverse. We show that the answer depends on the pretraining data budget, and find that pretraining data scale reverses the Muon–AdamW fine-tuning gap. At small budgets, Muon achieves lower fine-tuning loss with less forgetting than AdamW and the fine-tuning gap is negative. At larger budgets, this trend reverses: AdamW achieves a better learning–forgetting tradeof than Muon, and the fine-tuning gap becomes positive.

In Section 3, we pretrain OLMo-style 30M–600M models (Groeneveld et al., 2024; Team OLMo et al., 2025) on 150M to 154B DCLM tokens (Li et al., 2024), comparing AdamW and Muon, and fine-tune each base model on four datasets while sweeping the fine-tuning optimizer and learning rate, for nearly 600 pretraining runs and 29,000 fine-tuning runs in total. We show that when tuning pretraining hyperparameters for the lowest pretraining loss, Muon consistently achieves lower loss than AdamW. However, with suficient pretraining data, AdamW outperforms Muon after fine-tuning, reversing the Muon–AdamW fine-tuning gap. This reversal also holds when we disregard pretraining loss, instead sweeping pretraining hyperparameters to obtain the best learning–forgetting tradeof after fine-tuning. We additionally show that as model size scales, AdamW surpasses Muon at progressively smaller data budgets.

![](images/a0e911bcabb6d947b7a3e82ff0682055e224b601bbeb3cae8f66fd5b7a506eba.jpg)

![](images/7bc03fe3b54fd0aff7a6327b1a225c53d1ffb7f722540b1a12db0918bff5ca92.jpg)

![](images/6c9311c8e6a350054dd32fc59dc40d67ab6978ebaca62a9b52921c0bb70fd1e4.jpg)

![](images/07855fe7038b3c13176426f054f255570c2a7d8c8a5c85fea5258312a9958c7b.jpg)  
Figure 1: Beyond a suficiently large data budget, AdamW outperforms Muon. We pretrain OLMo-60M models with Muon and AdamW and fine-tune on StackMathQA. For each pretraining optimizer, we take the pretrained model with the lowest pretraining loss, sweep across all fine-tuning configurations, and plot the resulting learning–forgetting frontier, from 2.4B to 38.4B pretraining tokens. At 2.4B tokens, Muon’s frontier dominates AdamW’s; at larger budgets the trend reverses and AdamW surpasses Muon. This reversal happens despite Muon achieving lower pretraining loss across all data budgets.

In Section 4, we investigate why the Muon–AdamW fine-tuning gap reverses as the pretraining data budget grows. Fine-tuning changes the output distribution, so we generalize fine-tuning updates with perturbations to the output distribution and study which optimizer produces base models that are more robust to such perturbations. To quantify sensitivity to output perturbations, we introduce the margin-to-spread ratio (MSR), a simple metric derived from the model’s output logits that characterizes robustness. For each prediction, the MSR is the diference between the correct logit and the average of the incorrect logits, normalized by the spread of the incorrect logits (Definition 4.1): a prediction is more robust when the MSR is larger.

To understand where this trend comes from, we study a linear associative memory trained with gradient descent (GD) and Muon (Section 4.2). Analyzing a single update from zero initialization, we prove that the ratio of GD’s MSR to Muon’s increases with the data budget (Theorem 4.2). Simulations of many training steps in the linear associative memory setting confirm the same trend: Muon has the larger MSR at small data budgets, while GD has the larger MSR at large data budgets (Section 4.2). Measuring the MSR of our pretrained language models, we find that Muon has the larger MSR at small data budgets, while AdamW has the larger MSR at large data budgets, mirroring the reversal of the fine-tuning gap (Section 4.3).

## 2 Preliminaries

Our goal is to compare AdamW- and Muon-pretrained models afterfine-tuning. Since fine-tuning can induce forgetting of the pretraining distribution, performance on the fine-tuning task alone is not suficient: one base model may learn more but also forget more. We therefore compare base models by the tradeof between the two, and define metrics to track how this comparison changes with model size and pretraining data budget. Let $\theta _ { \mathrm { P T } }$ denote a pretrained model.

Learning–forgetting tradeof. Fine-tuning improves downstream performance but may degrade pretrained knowl edge (Goodfellow et al., 2014; Kirkpatrick et al., 2017; Springer et al., 2025). When $\theta _ { \mathrm { P T } }$ is fine-tuned to obtain $\theta _ { \mathrm { F T } }$ we measure learning by the fine-tuning loss ${ \mathcal { L } } _ { \mathrm { F T } } ( \theta _ { \mathrm { F T } } )$ and forgetting by the pretraining loss ${ \mathcal { L } } _ { \mathrm { P T } } ( \theta _ { \mathrm { F T } } )$ . Since both depend on the fine-tuning hyperparameters, a single fine-tuned checkpoint does not capture how well $\theta _ { \mathrm { P T } }$ can be adapted. We therefore summarize each base model by the outcomes reachable by fine-tuning it,

$$
\mathcal { T } ( \theta _ { \mathrm { P T } } ) : = \left\{ \left( \mathcal { L } _ { \mathrm { F T } } ( \theta _ { \mathrm { F T } } ) , \mathcal { L } _ { \mathrm { P T } } ( \theta _ { \mathrm { F T } } ) \right) \big | \theta _ { \mathrm { F T } } \in \Theta _ { \mathrm { F T } } ( \theta _ { \mathrm { P T } } ) \right\} ,
$$

where $\Theta _ { \mathrm { F T } } ( \theta _ { \mathrm { P T } } )$ is the set of models obtained by fine-tuning $\theta _ { \mathrm { P T } }$ , and compare base models through the lower-left Pareto frontier of this set, which we call the learning–forgettingfrontier. For a collection of base models B, we compare the corresponding sets

$$
\mathcal { T } ( B ) : = \{ \mathcal { T } ( \theta _ { \mathrm { P T } } ) \ | \ \theta _ { \mathrm { P T } } \in B \} .
$$

Fine-tuning gap. We compare the two optimizers on this tradeof with a single metric, fine-tuning gap $G _ { \mathrm { F T } }$ , which we define as the signed mean minimum Euclidean distance between the AdamW and Muon frontier. The fine-tuning gap is positive when the AdamW frontier dominates the Muon frontier, and the reverse else wise. Thus $G _ { \mathrm { F T } } > \bar { 0 }$ when AdamW dominates and $G _ { \mathrm { F T } } < 0$ when Muon dominates (Section C.5 gives the details).

![](images/9fd62d677c1592494a89b2cd9d390c495204a0468931aab0ff6e550852b39835.jpg)

![](images/827905a3a23920e670e541661ae4c48815f45342f7d9d96f7275331566415713.jpg)

![](images/b3fb97f90e7e678973f695d9263cd32f906a54cdc4c246d691ca6e6a13c62f80.jpg)

![](images/386f582193d2f6c558d549dd8c6ccb73f0b66290c2980c273543f4eb6b05a5d8.jpg)  
Figure 2: Crossover TPR decreases with model size. In the lowest-pretraining-loss setting, where each pretraining optimizer’s lowest-loss pretrained model is swept across all fine-tuning hyperparameters, Muon’s advantage reverses at substantially smaller pretraining data budgets as model size increases. The exact crossover TPR varies depending on the fine-tuning dataset, but the high-level trend is consistent.

Crossover token–parameter ratio (TPR). We quantify pretraining data budget using token–parameter ratio (TPR), the number of pretraining tokens per model parameter. For a fixed model size and fine-tuning dataset, the crossover TPR is the minimum TPR required for the AdamW frontier to overtake the Muon frontier.

## 3 Experiments

We now compare how the learning–forgetting tradeofs of Muon and AdamW scale with pretraining data budget and model size. In Section 3.2, we show that as the pretraining data budget increases, Muon’s advantage over AdamW shrinks and eventually reverses in favor of AdamW. This reversal happens at progressively smaller data budgets as model size grows.

## 3.1 Experimental setup

Pretraining. We pretrain OLMo-style 30M–600M models on 150M to 154B DCLM tokens, comparing AdamW and Muon under a warmup–stable–decay (WSD) schedule. We sweep hyperparameters broadly and report the complete pretraining methodology in Section C.

Fine-tuning. For each base model, we evaluate the learning–forgetting tradeof on four standard datasets: GSM8K (Cobbe et al., 2021), StackMathQA (Zhang, 2024), Alpaca (Taori et al., 2023), and MusicPile (Yuan et al., 2024), sweeping AdamW learning rates from $1 0 ^ { - 4 } \mathrm { t o } 1 0 ^ { - 2 }$ and Muon learning rates from $6 \times 1 0 ^ { - 4 } \dot { \mathrm { ~ t o ~ } } 2 \times 1 0 ^ { - 2 }$ (Section C.2). We fine-tune AdamW-pretrained models with AdamW and Muon-pretrained models with both AdamW and Muon: AdamW fine-tuning consistently outperforms Muon fine-tuning on AdamW-pretrained models, whereas neither optimizer is consistently better on Muon-pretrained models (Section D.3). We then plot the learning–forgetting frontier over the resulting runs. We choose a broad learning-rate range to ensure we capture the complete learning– forgetting tradeof spectrum: higher learning rates favor learning at the expense of forgetting, while lower rates preserve pretraining knowledge but learn less.

Evaluating the learning–forgetting tradeof. For each fine-tuned checkpoint, we compute the fine-tuning loss and pretraining loss on held-out fine-tuning and pretraining (DCLM) data, and use these checkpoints to approximate the learning–forgetting tradeof. Although this approximation is constructed from a finite set of fine-tuning runs, we conduct a comprehensive sweep over four datasets and across a wide range of fine-tuning learning rates to obtain a close estimate. We evaluate the learning–forgetting tradeof in two settings: lowest-pretraining-loss setting, and the best-frontier setting.

1. Lowest-pretraining-loss setting: Pretraining hyperparameters are typically selected to minimize pretraining loss. Therefore, for each optimizer, we tune the pretraining hyperparameters to minimize pretraining loss for a given pretraining data budget and model size. We then sweep across all fine-tuning configurations for each of the lowest-loss pretrained models, and evaluate the resulting frontier.

2. Best-frontier setting: However, prior studies find that the lowest-loss pretrained model is not always the best model after fine-tuning: models pretrained with other hyperparameters can ofer a better learning– forgetting tradeof (Watts et al., 2026). Therefore, for each pretraining optimizer, we additionally sweep across all pretraining and fine-tuning configurations to achieve the best possible learning–forgetting frontier for a given pretraining data budget and model size (Section C.1.3).

(a)  
![](images/e65e711eb3b46f842ad88f43c6f72732dc3e7a31f21c3372ac65e069cd71fa4c.jpg)

(b)  
![](images/bde96b07fe63e449cd01f2cf6c42216ef5ee65ea08b8dcaa893688d35cda6ff8.jpg)

(c)  
![](images/99de9166d7a48fd438235fd2c2663204326c4b9b74c52b6420201db403e64622.jpg)  
Figure 3: The Muon–AdamW fine-tuning gap widens with increasing pretraining data budget, and Muon’s advantage shrinks with model size. (a) The fine-tuning gap after fine-tuning OLMo-60M models on StackMathQA, in the lowest-pretraining-loss setting (each pretraining optimizer’s lowest-loss pretrained model swept across all fine-tuning configurations). As the pretraining data budget increases, the fine-tuning gap transitions from negative to positive, with AdamW outperforming Muon beyond the crossover TPR. (b) Pretraining loss at 40 TPR. Muon’s advantage over AdamW shrinks as model size grows. (c) Degradation at 40 TPR, when we accept a StackMathQA loss 10% above the best in exchange for less forgetting. Muon degrades increasingly more than AdamW as model size grows.

## 3.2 Main result

We now investigate how the Muon–AdamW fine-tuning gap scales with pretraining data budget and model size.

Main result: Pretraining data scale reverses the Muon–AdamW fine-tuning gap. We compare the finetuning gap for AdamW and Muon at diferent pretraining data budgets in the lowest-pretraining-loss setting, using StackMathQA as the fine-tuning dataset (Figure 3a). At small pretraining data budgets (2.4B tokens), Muon Paretodominates AdamW (Figure 1) and the fine-tuning gap is negative. At larger pretraining data budgets, this trend reverses: beyond 2.4B tokens, AdamW achieves lower fine-tuning loss with less forgetting than Muon. The trend also holds in the best-frontier setting (Figure 12), and for additional fine-tuning datasets in Section D.

We next investigate how the reversal of the Muon–AdamW fine-tuning gap scales with model size. Specifically, we ask how the amount of pretraining data needed for AdamW to overtake Muon changes as model size grows.

As model size scales, AdamW surpasses Muon at progressively smaller data budgets. We find that the crossover TPR decreases with model size. On StackMathQA, the crossover is 320 TPR for 30M models but only 10 TPR for 600M models (Figure 2). The specific crossover TPR depends on the fine-tuning dataset, but the broader trend persists; as model size grows, the crossover TPR decreases, and Muon’s advantage reverses with substantially smaller data budgets.

We hypothesize that in the lowest-pretraining-loss setting, this scaling trend is driven by two factors: Muon’s shrinking pretraining loss advantage and increased degradation relative to AdamW (Figure 3b,c). For each optimizer and model size, we compare its base pretraining loss and its degradation $\mathcal { L } _ { \mathrm { P T } } ( \theta _ { \mathrm { F T } } ) \stackrel {  } { - } \mathcal { L } _ { \mathrm { P T } } ( \theta _ { \mathrm { P T } } )$ , after fine-tuning on StackMathQA. First, for a fixed TPR, the gap in pretraining loss between Muon and AdamW shrinks as model size increases, reducing Muon’s advantage as a lower-loss base model. Additionally, for a fixed TPR and fine-tuning loss, Muon forgets increasingly more than AdamW as model size grows. Consequently, as model size scales, Muon’s pretraining advantage over AdamW narrows, while its forgetting relative to AdamW increases, enabling AdamW to overtake Muon at progressively smaller data budgets.

## 4 Analyzing the Muon–AdamW Forgetting Gap

In Section 3, we show that Muon ofers a better learning–forgetting tradeof at small token budgets, but AdamW overtakes it beyond a crossover budget. This reversal occurs even though Muon achieves lower pretraining loss at every budget we consider. In this section, we develop a theoretical explanation for this reversal. We compare Muon with exact orthogonalization to full-batch gradient descent (GD), a tractable comparison that captures the qualitative budget trend of Muon and AdamW in language models (Section 4.3).

To formalize the relationship between GD and Muon, we study a linear associative memory model of fact learning. In this setting, GD and Muon learn the facts to a similar degree, but their checkpoints can difer in their sensitivity to perturbations of the output distribution. Since fine-tuning changes the output distribution, we expect checkpoints that preserve factual accuracy under a wider range of perturbations to be more resistant to forgetting during fine-tuning. Our central goal is therefore to characterize this sensitivity across pretraining budgets and use it to explain why the forgetting gap reverses.

To quantify sensitivity to output perturbations, we introduce the margin-to-spread ratio (MSR): the diference between the correct answer’s logit and the average incorrect logit, normalized by the spread of the incorrect logits. The MSR captures a checkpoint’s robustness to output perturbations and provides an analytically tractable basis for comparing GD and Muon.

Roadmap. We introduce the linear associative memory setting, define the MSR, and establish its connection to sensitivity (Section 4.1). We then prove that, after one update step, the ratio of Muon’s MSR to GD’s decreases as the data budget grows, indicating that Muon becomes more sensitive to output perturbations relative to GD (Section 4.2). Finally, we show that the MSR tracks the robustness of AdamW and Muon in real language models (Section 4.3).

## 4.1 Setup and the margin-to-spread ratio

We begin by introducing our linear associative memory setup and defining the margin-to-spread ratio.

Linear associative memory. To capture the behavior of Muon and GD in a tractable setting, we consider a linear associative memory, in which a single weight matrix is trained to associate each of N keys, represented by a vector, with its corresponding value, also represented by a vector. Formally, we represent each fact by a key–value pair $( u _ { i } , v _ { i } ) \in \mathbb { R } ^ { d } \times \mathbf { \dot { \mathbb { R } } } ^ { d }$ , with all keys and values drawn independently from $\mathcal { N } ( 0 , \hat { \cal I } _ { d } )$ . The memory is parameterized by a matrix $W \in \mathbb { R } ^ { d \times d }$ . Given a key $u _ { i }$ , it scores each candidate value v<sub>j</sub> as $s _ { i j } \mathbf { \dot { ( } } W ) : = v _ { i } ^ { \top } W u _ { i }$ , which we treat as logits, and retrieves the highest-scoring value. Fact i is recovered when $s _ { i i } ( W ) > s _ { i j } ( W )$ for all $j \neq i$

We train with the softmax cross-entropy loss. Given a query distribution $q = ( q _ { 1 } , \ldots , q _ { N } )$ over the facts, the predicted probabilities and cross-entropy loss are

$$
\begin{array} { r } { \hat { p } _ { W } ( \cdot \vert i ) : = \mathrm { s o f t m a x } \big ( s _ { i 1 } ( W ) , \dots , s _ { i N } ( W ) \big ) , \mathcal { L } ( W ) : = - \sum _ { i = 1 } ^ { N } q _ { i } \log \hat { p } _ { W } ( i \mid i ) . } \end{array}\tag{1}
$$

Data budget. We denote the memory dimension by d and the number of facts by N, and grow the two together, keeping their ratio $\gamma = N / d > 1$ fixed, so that $N = \gamma d$ as $d \to \infty$ . This mirrors the typical scaling behavior of language models, which store many more facts than they have dimensions. We call γ the data budget: it measures facts per dimension and plays the role of token-parameter ratio in Section 2. We compare data budgets by taking this limit at diferent fixed values of $\gamma .$ Since $\gamma > 1$ , we have $N - 1 \geq d .$ and log $N / \bar { d } \to 0$ , written log $N = o ( d )$ For random variables X and a deterministic $a > 0 , X = O _ { p } ( a )$ means that for every $\varepsilon > 0$ there is a C with $\operatorname* { P r } [ | X | > C a ] \leq \varepsilon$ eventually, and $X = o _ { p } ( a )$ means that $\operatorname* { P r } [ | \dot { \boldsymbol { X } } | > \varepsilon a ]  0$ for every $\varepsilon > 0$

Optimization. For simplicity, we consider the exact orthogonalization variant of Muon. For a matrix A with singular value decomposition $\mathbf { \bar { \boldsymbol { A } } } = \boldsymbol { P } \boldsymbol { \Sigma } \boldsymbol { Q } ^ { \intercal }$ , let polar $( A ) : = { \dot { P Q } } ^ { \top }$ . Muon updates $W _ { t + 1 } = W _ { t } - \eta$ polar $\left( \nabla \mathcal { L } ( W _ { t } ) \right)$ We compare Muon to full-batch gradient descent (GD), whose updates descend along the gradient itself, $W _ { t + 1 } =$ $W _ { t } - \eta \dot { \nabla } \mathcal { L } ( W _ { t } )$

Margin-to-spread ratio. In a binary classification problem, a prediction is robust if the correct logit is far from the incorrect logit: the larger the margin, the more robust the prediction. In a multiclass classification problem, such as our linear associative memory setting, there are many incorrect logits. The robustness of a correct output to perturbations of the logits thus depends on its separation from all of the incorrect logits. As it turns out, the degree of separation can be usefully summarized by a single statistic, which we call the margin-to-spread ratio (MSR). MSR captures the average separation between the correct logit and all other logits, normalized by the spread of the incorrect logits. A smaller spread, or a larger average separation, would both place the incorrect logits further below the correct logit relative to their spread, which can increase the robustness of the prediction to perturbations. To capture this intuition, we formally define the margin-to-spread ratio as follows:

Definition 4.1 (Margin-to-spread ratio). For fact $i ,$ define the pairwise margins $m _ { i j } : = s _ { i i } ( W ) - s _ { i j } ( W )$ for $j \neq i ,$ with mean and variance

$$
h _ { i } : = \operatorname* { m e a n } _ { j \neq i } m _ { i j } , \qquad \kappa _ { i } ^ { 2 } : = \operatorname* { m e a n } _ { j \neq i } ( m _ { i j } - h _ { i } ) ^ { 2 } .\tag{2}
$$

The margin-to-spread ratio (MSR) for fact i is $\rho _ { i } : = h _ { i } / \kappa _ { i } $ for $\kappa _ { i } > 0$ . At the model level, we average the mean margins and variances uniformly over facts before taking the ratio:

$$
\bar { h } : = \operatorname * { m e a n } _ { i } h _ { i } , \qquad \bar { \kappa } ^ { 2 } : = \operatorname * { m e a n } _ { i } \kappa _ { i } ^ { 2 } , \qquad \rho : = \frac { \bar { h } } { \bar { \kappa } } ,\tag{3}
$$

where $\rho$ is defined for $\bar { \kappa } > 0 .$

Since $s _ { i i } ( W )$ is the same for every competitor $j , \kappa _ { i }$ is also the standard deviation of the incorrect logits $s _ { i j } ( W )$ , so the MSR measures the average margin in units of their spread. The definition applies to any model that assigns a logit to each candidate answer; for a language model, $s _ { i j } ( W )$ are the next-token logits (Section 4.3).

Under isotropic Gaussian noise added to the logits, the MSR gives a lower bound on the fraction of incorrect answers that stay below the correct one: with noise of standard deviation σ on each logit, this fraction is at least $\rho _ { i } ^ { 2 } / ( \rho _ { i } ^ { 2 } + 1 + 2 \bar { \sigma ^ { 2 } } / \kappa _ { i } ^ { 2 } )$ in expectation. For a fixed noise strength relative to the incorrect-logit spread, this bound improves as the MSR increases (Corollary B.5).

## 4.2 One-step analysis of the margin-to-spread ratio

We have so far introduced our linear associative memory setup and built intuition for how the MSR captures the robustness of a model to perturbations of its logits. Here, we argue that the MSR of GD is greater than that of Muon for suficiently large data budgets. We precisely track the one-step dynamics of the MSR ratio $\mathcal { R } : = \rho ^ { \mathrm { G D } } / \rho ^ { \mathrm { M u o n } }$ between GD and Muon in the linear associative memory.

We analyze a single update from zero initialization, using uniform query weights $q _ { i } = 1 / N$ . The weights after many optimization steps depend on the whole training trajectory and are hard to characterize precisely, so theoretical studies of how an optimizer shapes the learned weights often analyze its first step (Ba et al., 2022; Dandi et al., 2024; Moniri et al., 2024). Kim et al. (2026) take the same approach to compare a single step of Muon, SGD, and Newton’s method in a linear associative memory.

Theorem 4.2 (Informal). Consider one GD step and one Muon step from zero in the Gaussian associative memory model, with data $b u d g e t \gamma = N / d .$ . For suficiently large γ:

• GD has a larger MSR than Muon.

• Increasing the data budget $\gamma$ by any fixed percentage increases the ratio $\mathcal { R } = \rho ^ { \mathrm { G D } } / \rho ^ { \mathrm { M u o n } }$ , with high probability as the dimension grows.

• The ratio approaches 3π $\cdot / 8 \approx 1 . 1 8$ from below at a rate of $\operatorname { 1 } / \gamma \colon$

$$
\begin{array} { r } { - \frac { C } { \gamma } \leq \mathcal { R } - \frac { 3 \pi } { 8 } \leq - \frac { c } { \gamma } , } \end{array}
$$

where $0 < c < C$ are absolute constants.

All claims hold with probability tending to one as $d \to \infty$ , for suficiently large fixed $\gamma .$

Proof sketch. Our goal is to compare one update step for Muon and for GD.

1. Since the MSR does not change when W is multiplied by a positive constant, the one-step weights can be written as

$$
W ^ { \mathrm { G D } } = G , \qquad W ^ { \mathrm { M u o n } } = O : = \operatorname { p o l a r } ( G )
$$

(Appendix, Lemma F.1). We now want to compute $\rho = \bar { h } / \bar { \kappa }$ for these matrices. This requires the mean margins and their standard deviations; we begin with the means.

2. Compute the mean margins. Recall that h<sup>¯</sup> averages the diference between the correct and incorrect logits over facts and incorrect answers. For uniformly weighted facts,

$$
\begin{array} { r } { G = \sum _ { k = 1 } ^ { N } ( v _ { k } - \bar { v } ) u _ { k } ^ { \top } , \qquad \bar { v } = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } v _ { k } . } \end{array}
$$

Substituting into this average gives $\bar { h } ( W ) = \langle W , G \rangle / ( N - 1 )$ , and therefore

$$
\begin{array} { r } { \bar { h } ^ { \mathrm { G D } } = \frac { \| G \| _ { F } ^ { 2 } } { N - 1 } , \qquad \bar { h } ^ { \mathrm { M u o n } } = \frac { \| G \| _ { * } } { N - 1 } , } \end{array}
$$

where $\left\| \cdot \right\| _ { F }$ is the Frobenius norm and $\lVert \cdot \rVert _ { * }$ <sub>∗</sub> is the sum of the singular values.

3. Compute $G D ' s$ variance. For each query $i ,$ we take the variance across incorrect candidates $j ,$ holding the dataset and weights fixed:

$$
\kappa _ { i } ^ { 2 } = \mathrm { V a r } _ { j \neq i } ( v _ { j } ^ { \top } G u _ { i } ) .
$$

To calculate its typical size, expand the incorrect logit:

$$
\begin{array} { r } { v _ { j } ^ { \top } G u _ { i } \approx \underbrace { \sum _ { k \notin \{ i , j \} } ( v _ { j } ^ { \top } v _ { k } ) ( u _ { k } ^ { \top } u _ { i } ) } _ { \mathrm { o t h e r ~ f a c t s } } + \underbrace { \| u _ { i } \| ^ { 2 } ( v _ { j } ^ { \top } v _ { i } ) + \| v _ { j } \| ^ { 2 } ( u _ { j } ^ { \top } u _ { i } ) } _ { \mathrm { q u e r y ~ a n d ~ c a n d i d a t e ~ f a c t s } } . } \end{array}
$$

Here ≈ omits the mean-subtraction term in $G ,$ which only afects lower-order terms in the variance. Each other fact contributes a product of two independent, mean-zero inner products, each with variance $d ,$ giving variance $d ^ { 2 }$ In the query and candidate terms, we approximate the squared norms by $d ,$ dropping their fluctuations. This gives variance $d ^ { 3 }$ each, with $O ( d ^ { 2 } )$ corrections. Replacing $\bar { N _ { - 2 } }$ by N also discards only $O ( d ^ { 2 } )$ terms.

These contributions are uncorrelated over the Gaussian training data. Including the omitted corrections and concentration of the empirical average, we calculate GD’s variance (Appendix, Lemma F.15) to be

$$
( \bar { \kappa } ^ { \mathrm { G D } } ) ^ { 2 } = \big ( \underbrace { N d ^ { 2 } } _ { \mathrm { o t h e r f a c t s } } + \underbrace { 2 d ^ { 3 } } _ { \mathrm { q u e r y ~ a n d ~ c a n d i d a t e } } \big ) ( 1 + o _ { p } ( 1 ) ) = d ^ { 3 } ( \gamma + 2 ) ( 1 + o _ { p } ( 1 ) ) .
$$

4. Compute Muon’s variance. Again, the variance is over incorrect candidates for each fixed query. Gaussian isotropy and orthogonality give the leading calculation

$$
\begin{array} { r } { ( \bar { \kappa } ^ { \mathrm { M u o n } } ) ^ { 2 } \approx \frac { 1 } { N ( N - 1 ) } \sum _ { i } \sum _ { j \neq i } ( v _ { j } ^ { \top } O u _ { i } ) ^ { 2 } \approx \frac { 1 } { N } \sum _ { i } \| O u _ { i } \| ^ { 2 } = \frac { 1 } { N } \sum _ { i } \| u _ { i } \| ^ { 2 } \approx d . } \end{array}
$$

The first approximation drops the squared mean of the incorrect logits for each query. The second neglects departures of the sampled incorrect values from perfect isotropy, and the last drops fluctuations in the average squared key length. We calculate the variance including the omitted corrections and the dependence of O on the training data (Appendix, Lemma F.31) to be

$$
( \bar { \kappa } ^ { \mathrm { M u o n } } ) ^ { 2 } = d \left[ 1 + { \cal O } ( \gamma ^ { - 3 / 2 } ) + o _ { p } ( 1 ) \right] .
$$

5. Put the means and standard deviations together. We calculate the norms in the mean margins using the Gaussian singular-value distribution (Appendix, Lemma F.4). Dividing by the standard deviations gives, to first order in $1 / \gamma ,$

$$
\begin{array} { r l r } { \rho ^ { \mathrm { G D } } \approx \sqrt { \frac { d } { \gamma } } \left( 1 - \frac { 1 } { \gamma } \right) , } & { { } } & { \rho ^ { \mathrm { M u o n } } \approx \frac { 8 } { 3 \pi } \sqrt { \frac { d } { \gamma } } \left( 1 - \frac { 1 } { 1 0 \gamma } \right) . } \end{array}
$$

These approximations omit higher-order budget corrections and relative fluctuations that vanish as $d \to \infty$ Taking their ratio cancels the common factor $\sqrt { d / \gamma }$ . Keeping track of the errors gives

$$
\begin{array} { r } { \mathcal { R } = \frac { 3 \pi } { 8 } \left( 1 - \frac { 9 } { 1 0 \gamma } \right) + O ( \gamma ^ { - 3 / 2 } ) + o _ { p } ( 1 ) . } \end{array}
$$

Since $3 \pi / 8 > 1 , \mathrm { G D }$ has the larger MSR at suficiently large budgets. The negative $1 / \gamma$ term in $\mathcal { R } - 3 \pi / 8$ dominates the smaller remainder, giving the claimed rate. A fixed percentage increase in γ raises the leading expression by order $1 / \gamma .$ which also dominates the remainder, establishing the increase with high probability. See Section F for the quantitative bound and full proof, and Figure 4 and Section E.1 for numerical checks.

This theorem demonstrates that in the linear associative memory, when the data budget is suficiently large, checkpoints learned with GD have a larger MSR than checkpoints learned with Muon after a single step of training. Combined with the robustness bound of Section $4 . 1$ , GD’s checkpoints therefore carry the stronger guarantee of robustness to perturbations of the output distribution in our theoretical setting.

Remark (Choice of data distribution). Independent standard Gaussian keys and values provide a canonical isotropic model with no preferred direction. Other data distributions can change both the gradient spectrum and the competing-score variances, so the optimizer ordering may difer. The analogous budget trend in the language-model experiments (Section 4.3) suggests that the behavior captured by this simple model is relevant beyond the Gaussian setting.

(a)  
![](images/dd13f2851e7da506aff6162435b5eaabbe2277e9400a0bac3c6c7278a93b42cb.jpg)

(b)  
![](images/f2f5d34488f0f3414d62c6bdac608003e94c835e620522d7fe8ca3539291f362.jpg)

(c)  
![](images/fa0226c2caa08942082663f3c67e6c3e644ff8b2a95828d3b64a1aaffc3c0cc4.jpg)

(d)  
![](images/920391ad62c692d2a4de893bd87c2af627747ee31b98261933617aeabfeddfb5.jpg)  
Figure 4: The margin-to-spread ratio of GD gains on Muon’s as the data budget grows. (a, b) After $t = 1 0 ^ { 4 }$ steps with model dimension d = 256, GD’s MSR ρ overtakes Muon’s as γ grows, and the ordering of the perturbed robustness $\sigma _ { 5 0 }$ turns over with it. (c, d) After one step with d = 512, GD’s normalized MSR $\rho / \sqrt { d / \gamma }$ overtakes Muon’s (c), so the MSR ratio $\mathcal { R } = \rho ^ { \mathrm { G D } } / \rho ^ { \mathrm { M u o n } }$ rises with γ (d), following the theory of Theorem 4.2.

Beyond one step. We next check whether the trend persists after extended training. We train the linear memory for $T \equiv 1 0 ^ { 4 }$ full-batch steps from $W _ { 0 } = 0$ at constant learning rates across three seeds. We summarize the saturated MSR by the per-seed median over $t \geq 1 0 ^ { 3 }$ and measure robustness by $\sigma _ { 5 0 } ,$ , the relative perturbation size at which half of the initially recovered facts are lost. We exclude data budgets past the memory’s capacity where the two optimizers recover diferent item sets. The full setup and results for every dimension d appear in Section E.1.

The results in Figure 4 show that R increases with the data budget. Muon has the larger MSR at small $\gamma ,$ while GD has the larger MSR at large γ. The robustness comparison shifts in the same direction: Muon tolerates stronger perturbations at small $\gamma ,$ and GD does at large γ. Thus, the budget trend persists in the linear setting with GD as the baseline.

## 4.3 Connection to practice

We test whether the budget trend predicted by Theorem 4.2 also appears in real language models, beyond the assumptions used in our theoretical analysis.

Setup. We evaluate the AdamW and Muon models trained in Section 3, spanning five model sizes, on held-out DCLM text. We compute the MSR at each token position and report the median over correctly predicted positions. At each position, h and κ are the mean and standard deviation of the margins between the correct next-token logit and all other vocabulary logits. We use the median to reduce sensitivity to outliers; Definition 4.1 instead uses the mean margin divided by the square root of the mean variance.

Experimental Result. Figure 5 plots the MSR for the three model sizes with the widest budget ranges, from 30M to 100M. Muon has a larger MSR at small data budgets, while AdamW has a larger MSR at suficiently large budgets. As the budget grows, the diference between AdamW’s and Muon’s MSRs increases, and the ratio of AdamW’s MSR to Muon’s crosses one. This shift toward AdamW is consistent with the theoretical budget trend and parallels the reversal in relative robustness to fine-tuning observed in Section 3.

(a)  
![](images/3fc5db42ecfdbee9cd0a28b03997c4fbe70aa8aba47392104ad65f5b717f9634.jpg)

![](images/e96c7eeea5358934b550b8d49827ce8ad8ac862a7e8d2fa86cc95d42c2dcae2a.jpg)

![](images/db858951edb9bb3fb226eb4475802562e6f4d358d462077a5690ed1acb0a0b5d.jpg)

(b)  
![](images/951fcb681bb5b89f4a5d44602ba249b075e640e84adcac39141eb0dafb554c8c.jpg)  
Figure 5: In real language models, the margin-to-spread ratio of AdamW gains on Muon’s as the token budget grows. (a) MSR ρ of AdamW and Muon against pretraining TPR for 30M, 60M, and 100M models: Muon has the larger MSR at small budgets and AdamW at large ones. (b) The MSR ratio $\rho ^ { \mathrm { A d a m W } } / \rho ^ { \mathrm { M u o n } }$ crosses one as the budget grows, at every model size.

The MSR and fine-tuning crossovers do not necessarily appear at exactly the same budget, since the learning– forgetting tradeof also depends on pretraining loss. Nevertheless, the MSR captures the direction of the observed shift and connects our theoretical analysis to the behavior of trained language models.

Remark. We summarize the analysis of this section and relate MSR to other metrics used to predict post-training.

• Takeaway. In a linear associative memory, we prove that after one update the ratio $\rho ^ { \mathrm { G D } } / \rho ^ { \mathrm { M u o n } }$ of the two optimizers’ MSRs increases with the data budget (Theorem 4.2). Simulations of many training steps show that the same trend persists after extended training (Section 4.2).

Extension: Relation of MSR to Other Metrics Predicting Post-Training. Other work predicts forgetting after post-training with diferent metrics. For example, Springer et al. (2025) show that the loss increase after adding Gaussian noise to the pretrained weights tracks how much a model degrades under fine-tuning. In Section B, we show that MSR is connected to this robustness to Gaussian perturbations: in the linear memory, a larger MSR provably guarantees that more competitors stay ranked below the correct answer after the perturbation (Theorem B.3), and in both language models and the linear memory, MSR is strongly correlated with perturbation robustness (Section B.2.2).

## 5 Related Work

Muon. Muon, an optimizer introduced by Jordan et al. (2024), has recently shown strong empirical performance for pretraining LLMs, challenging AdamW’s dominance as the status-quo optimizer. Recent work has sought to explain Muon’s pretraining advantage through analyses of simplified theoretical models, associative memory, convergence behavior, and loss landscape geometry (Kim et al., 2026; Li et al., 2026; Sato et al., 2025; Su, 2025; Shen et al., 2025; Wang et al., 2026a). Another line of work investigates the training dynamics of Muon, finding that it learns data components more uniformly than AdamW (Vasudeva et al., 2026; Wang et al., 2026b; Dragutinović et al., 2026), while also characterizing the implicit biases of Muon (Fan et al., 2025; Dragutinović et al., 2026). Yet fewer works have examined the downstream consequences of Muon. Ruan et al. (2026) compare Muon and AdamW after fine-tuning; they find that for small data regimes, Muon outperforms AdamW, consistent with our findings. We show that this advantage does not generalize across pretraining data scale; after a suficiently large pretraining data budget, the trend reverses, and AdamW outperforms Muon.

Muon versus AdamW on pretraining loss. Muon has been reported to reach lower pretraining loss than AdamW for the same compute across a wide range of settings (Liu et al., 2025; Essential AI et al., 2025; Mehta et al., 2025), and frontier models have since been pretrained with Muon or its variants (Kimi Team, 2025; GLM-4.5 Team et al., 2025). The size of this advantage depends on scale: under controlled tuning, Muon’s speedup over AdamW shrinks as model size and data grow, and Muon is overtaken by other matrix-based optimizers at large data-to-model ratios (Wen et al., 2026b;a). All of these works measure pretraining loss. Optimizers can also reach qualitatively diferent solutions rather than faster or slower routes to the same minimum (Pascanu et al., 2025); we ask whether these diferences matter after fine-tuning.

Catastrophic forgetting. Catastrophic forgetting, the degradation of previously learned knowledge during subsequent training, remains a central challenge in deep learning (Goodfellow et al., 2014). Numerous approaches have been proposed to mitigate catastrophic forgetting, including regularization-based methods (Kirkpatrick et al., 2017; Zenke et al., 2017; Aljundi et al., 2018), replay techniques (Shin et al., 2017), gradient projection methods (Lopez-Paz & Ranzato, 2017; Chaudhry et al., 2019), and optimization-based approaches (Watts et al., 2026). Our learning–forgetting tradeof is an instance of the stability–plasticity dilemma, with fine-tuning performance capturing plasticity and knowledge retention capturing stability (Lopez-Paz & Ranzato, 2017). We study forgetting as an intrinsic property of the pretrained checkpoint, rather than as a consequence of the fine-tuning algorithm. Specifically, we investigate how optimization choices during pretraining shape downstream forgetting, showing that Muon and AdamW induce diferent learning–forgetting tradeofs.

Pretraining and downstream adaptation. Better pretraining loss does not always yield a better model after fine-tuning. Springer et al. (2025) show that extending pretraining can make language models harder to fine-tune, as the pretrained parameters become increasingly sensitive to modifications, which they measure in part with Gaussian perturbations. Watts et al. (2026) show that sharpness-aware pretraining mitigates forgetting during fine-tuning, and Liu et al. (2026) find that fine-tuning with the same optimizer as pretraining forgets less. We study how the choice of pretraining optimizer interacts with the pretraining data budget, and connect the resulting fine-tuning gap to robustness to weight perturbations.

## 6 Conclusion

We asked whether the quality of Muon base models persists through fine-tuning, and found that the answer depends on the pretraining budget: beyond a crossover budget, AdamW outperforms Muon after fine-tuning, even though Muon reaches lower pretraining loss. We do not necessarily expect this for every fine-tuning distribution, since fine-tuning on data close to the pretraining data is simply continued pretraining. Still, the trend holds broadly on all datasets we study, including math, instruction following, and music. Furthermore, the crossover budget decreases with model size, which suggests that the problem persists as we scale.

We explain why AdamW outperforms Muon at large budgets by tracking how robust each model is to small isotropic perturbations of its output distribution, which we characterize with the margin-to-spread ratio (MSR). AdamW becomes increasingly more robust than Muon as the data budget grows, and this robustness is a property of the output distribution that the training loss does not capture. Since the MSR measures robustness to perturbations of the output distribution in general, our theory suggests that this robustness is fundamental to the checkpoint learned by each optimizer, rather than specific to fine-tuning, to the perturbations we study, or to a particular architecture or parameterization. It should therefore apply to any update that changes the output distribution.

Although outside the scope ofthis work, our analysis suggests that pretraining should explicitly optimize for properties beyond the pretraining loss. Optimizer selection requires more than comparing pretraining loss, and we can design objectives that explicitly target adaptability, for which the MSR is a potential candidate.

More broadly, we expect this lesson to extend beyond Muon. Many pretraining choices, such as hyperparameters, data, and architecture, are selected by pretraining loss. Yet, our results show that pretraining loss can be a poor proxy for adaptability, increasingly so as the data budget grows. This suggests that for any of these choices, a lower pretraining loss does not by itself yield a model that can better adapt.

## Acknowledgements

We gratefully acknowledge support from Apple, Google, Jane Street, the National Science Foundation and the FLAME cluster at Carnegie Mellon University.

This material is based upon work supported by the National Science Foundation Graduate Research Fellowship under Grant No. DGE2140739. Any opinions, findings, conclusions, or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

The authors thank Kaiyue Wen, Alex Damian, Ishaan Watts, Lawrence Feng, and Gene Yang for helpful discussions.

## References

Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, and Tinne Tuytelaars. Memory aware synapses: Learning what (not) to forget. In Proceedings of the European conference on computer vision (ECCV), pp. 139–154, 2018.

Jimmy Ba, Murat A. Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu, and Greg Yang. High-dimensional asymptotics of feature learning: How one gradient step improves the representation. In Advances in Neural Information Processing Systems, 2022. URL https://arxiv.org/abs/2205.01445.

Arslan Chaudhry, Marc’Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny. Eficient lifelong learning with A-GEM. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?i d=Hkf2\_sC5FX.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/abs/2110.14168.

Yatin Dandi, Florent Krzakala, Bruno Loureiro, Luca Pesce, and Ludovic Stephan. How two-layer neural networks learn, one (giant) step at a time. Journal ofMachine Learning Research, 25(349):1–65, 2024. URL https://jmlr.o rg/papers/v25/23-1543.html.

Sara Dragutinović, Yedi Zhang, and Rajesh Ranganath. To use or not to use Muon: How simplicity bias in optimizers matters, 2026. URL https://arxiv.org/abs/2603.00742.

Essential AI, Ishaan Shah, Anthony M. Polloreno, Karl Stratos, Philip Monk, Adarsh Chaluvaraju, et al. Practical eficiency of Muon for pretraining, 2025. URL https://arxiv.org/abs/2505.02222.

Chen Fan, Mark Schmidt, and Christos Thrampoulidis. Implicit bias of spectral descent and Muon on multiclass separable data. In Advances in Neural Information Processing Systems, 2025. URL https://arxiv.org/abs/2502 .04664.

GLM-4.5 Team, Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, et al. GLM-4.5: Agentic, reasoning, and coding (ARC) foundation models, 2025. URL https://arxiv.org/abs/2508.06471.

Ian J. Goodfellow, Mehdi Mirza, Da Xiao, Aaron Courville, and Yoshua Bengio. An empirical investigation of catastrophic forgetting in gradient-based neural networks. In International Conference on Learning Representations, 2014. URL https://arxiv.org/abs/1312.6211.

Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Raghavi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Morrison, Niklas Muennighof, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, Will Smith, Emma Strubell, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Luke Zettlemoyer, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah A. Smith, and Hannaneh Hajishirzi. OLMo: Accelerating the science of language models. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pp. 15789–15809, 2024. URL https://aclanthology.org/2024.acl-long.841.

Jordan Hofmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://arxiv.org/abs/22 03.15556.

Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks. Blog post, 2024. URL https://kellerjordan.github.io/p osts/muon/.

Juno Kim, Eshaan Nichani, Denny Wu, Alberto Bietti, and Jason D. Lee. Sharp capacity scaling of spectral optimizers in learning associative memory, 2026. URL https://arxiv.org/abs/2603.26554.

Kimi Team. Kimi K2: Open agentic intelligence, 2025. URL https://arxiv.org/abs/2507.20534.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015. URL https://arxiv.org/abs/1412.6980.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. Proceedings ofthe National Academy ofSciences, 114(13):3521–3526, March 2017. ISSN 1091-6490. doi: 10.1073/pnas.1611835114. URL http://dx.doi.org/10.1073/pnas.1611835114.

Binghui Li, Kaifei Wang, Han Zhong, Pinyan Lu, and Liwei Wang. Muon in associative memory learning: Training dynamics and scaling laws. In International Conference on Machine Learning, 2026. URL https://arxiv.org/abs/ 2602.05725.

Jefrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Yitzhak Gadre, Hritik Bansal, Etash Kumar Guha, Sedrick Keh, Kushal Arora, Saurabh Garg, Rui Xin, Niklas Muennighof, Reinhard Heckel, Jean Mercat, Mayee F Chen, Suchin Gururangan, Mitchell Wortsman, Alon Albalak, Yonatan Bitton, Marianna Nezhurina, Amro Kamal Mohamed Abbas, Cheng-Yu Hsieh, Dhruba Ghosh, Joshua P Gardner, Maciej Kilian, Hanlin Zhang, Rulin Shao, Sarah M Pratt, Sunny Sanyal, Gabriel Ilharco, Giannis Daras, Kalyani Marathe, Aaron Gokaslan, Jieyu Zhang, Khyathi Chandu, Thao Nguyen, Igor Vasiljevic, Sham M. Kakade, Shuran Song, Sujay Sanghavi, Fartash Faghri, Sewoong Oh, Luke Zettlemoyer, Kyle Lo, Alaaeldin El-Nouby, Hadi Pouransari, Alexander T Toshev, Stephanie Wang, Dirk Groeneveld, Luca Soldaini, Pang Wei Koh, Jenia Jitsev, Thomas Kollar, Alex Dimakis, Yair Carmon, Achal Dave, Ludwig Schmidt, and Vaishaal Shankar. Datacomp-LM: In search of the next generation of training sets for language models. In The Thirty-eighth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum?id=CNWdWn47IE.

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, and Zhilin Yang. Muon is scalable for LLM training, 2025. URL https://arxiv.org/abs/2502.16982.

Yuxing Liu, Jianyu Wang, and Tong Zhang. Optimizer-model consistency: Full finetuning with the same optimizer as pretraining forgets less, 2026. URL https://arxiv.org/abs/2605.06654.

David Lopez-Paz and Marc’Aurelio Ranzato. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30, 2017.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019.

Sushant Mehta, Raj Dandekar, Rajat Dandekar, and Sreedath Panat. Muon: Training and trade-ofs with latent attention and MoE, 2025. URL https://arxiv.org/abs/2509.24406.

Behrad Moniri, Donghwan Lee, Hamed Hassani, and Edgar Dobriban. A theory of non-linear feature learning with one gradient step in two-layer neural networks. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, 2024. URL https://proceedings.mlr.press/ v235/moniri24a.html.

Razvan Pascanu, Clare Lyle, Ionut-Vlad Modoranu, Naima Elosegui Borras, Dan Alistarh, Petar Velickovic, Sarath Chandar, Soham De, and James Martens. Optimizers qualitatively alter solutions and we should leverage this, 2025. URL https://arxiv.org/abs/2507.12224.

Tianyu Ruan, Fengzhuo Zhang, Shuche Wang, and Shihua Zhang. Muon learns more robust and transferable features than Adam, 2026. URL https://arxiv.org/abs/2606.09658.

Naoki Sato, Hiroki Naganuma, and Hideaki Iiduka. Convergence bound and critical batch size of Muon optimizer, 2025. URL https://arxiv.org/abs/2507.01598.

Wei Shen, Ruichuan Huang, Minhui Huang, Cong Shen, and Jiawei Zhang. On the convergence analysis of Muon, 2025. URL https://arxiv.org/abs/2505.23737.

Hanul Shin, Jung Kwon Lee, Jaehong Kim, and Jiwon Kim. Continual learning with deep generative replay. In Advances in Neural Information Processing Systems, volume 30, 2017. URL https://arxiv.org/abs/1705.08690.

Jacob Mitchell Springer, Sachin Goyal, Kaiyue Wen, Tanishq Kumar, Xiang Yue, Sadhika Malladi, Graham Neubig, and Aditi Raghunathan. Overtrained language models are harder to fine-tune. In International Conference on Machine Learning, 2025. URL https://arxiv.org/abs/2503.19206.

Weijie Su. Isotropic curvature model for understanding deep learning optimization: Is gradient orthogonalization optimal?, 2025. URL https://arxiv.org/abs/2511.00674.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford Alpaca: An instruction-following LLaMA model. https://github.com/tatsu-lab/stanf ord\_alpaca, 2023.

Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, Nathan Lambert, Dustin Schwenk, Oyvind Tafjord, Taira Anderson, David Atkinson, Faeze Brahman, Christopher Clark, Pradeep Dasigi, Nouha Dziri, Allyson Ettinger, Michal Guerquin, David Heineman, Hamish Ivison, Pang Wei Koh, Jiacheng Liu, Saumya Malik, William Merrill, Lester James V. Miranda, Jacob Morrison, Tyler Murray, Crystal Nam, Jake Poznanski, Valentina Pyatkin, Aman Rangapur, Michael Schmitz, Sam Skjonsberg, David Wadden, Christopher Wilhelm, Michael Wilson, Luke Zettlemoyer, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi. 2 OLMo 2 furious. In Conference on Language Modeling, 2025. URL https: //arxiv.org/abs/2501.00656.

Bhavya Vasudeva, Puneesh Deora, Yize Zhao, Vatsal Sharan, and Christos Thrampoulidis. How Muon’s spectral design benefits generalization: A study on imbalanced data. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=YzjS4jcfmS.

Shuche Wang, Fengzhuo Zhang, Jiaxiang Li, Dirk Bergemann, and Zhuoran Yang. Why Muon outperforms Adam: A curvature perspective, 2026a. URL https://arxiv.org/abs/2606.04662.

Shuche Wang, Fengzhuo Zhang, Jiaxiang Li, Cunxiao Du, Chao Du, Tianyu Pang, Zhuoran Yang, Mingyi Hong, and Vincent Y. F. Tan. Muon outperforms Adam in tail-end associative memory learning. In The Fourteenth International Conference on Learning Representations, 2026b. URL https://arxiv.org/abs/2509.26030.

Ishaan Watts, Catherine Li, Sachin Goyal, Jacob Mitchell Springer, and Aditi Raghunathan. Sharpness-aware pretraining mitigates catastrophic forgetting. In Proceedings of the 43rd International Conference on Machine Learning, 2026. URL https://arxiv.org/abs/2605.02105.

Kaiyue Wen, Xingyu Dang, Kaifeng Lyu, Tengyu Ma, and Percy Liang. Fantastic pretraining optimizers and where to find them II: Hyperball optimization, 2026a. URL https://arxiv.org/abs/2606.16899.

Kaiyue Wen, David Hall, Tengyu Ma, and Percy Liang. Fantastic pretraining optimizers and where to find them. In The Fourteenth International Conference on Learning Representations, 2026b. URL https://openreview.net/for um?id=2J51qUZ0iG.

Ruibin Yuan, Hanfeng Lin, Yi Wang, Zeyue Tian, Shangda Wu, Tianhao Shen, Ge Zhang, Yuhang Wu, Cong Liu, Ziya Zhou, Ziyang Ma, Liumeng Xue, Ziyu Wang, Qin Liu, Tianyu Zheng, Yizhi Li, Yinghao Ma, Yiming Liang, Xiaowei Chi, Ruibo Liu, Zili Wang, Pengfei Li, Jingcheng Wu, Chenghua Lin, Qifeng Liu, Tao Jiang, Wenhao Huang, Wenhu Chen, Emmanouil Benetos, Jie Fu, Gus Xia, Roger Dannenberg, Wei Xue, Shiyin Kang, and Yike Guo. ChatMusician: Understanding and generating music intrinsically with LLM. In Findings ofthe Association for Computational Linguistics: ACL 2024, pp. 6252–6271, 2024. URL https://aclanthology.org/2024.findings-acl.373/.

Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. In International conference on machine learning, pp. 3987–3995. PMLR, 2017.

Yifan Zhang. StackMathQA: A Curated Collection of 2 Million Mathematical Questions and Answers Sourced from Stack Exchange. Technical report, Math-AI, 2024.

## Appendix Contents

A Definitions 15   
A.1 Optimizers 15   
A.2 Learning-rate schedules 15   
A.3 Token–parameter ratio 16   
B Other Metrics for Analyzing Post-Training 16   
B.1 Gaussian perturbation . 16   
B.2 Gaussian perturbation and the margin-to-spread ratio . 17   
C Experimental Details 19   
C.1 Pretraining . 19   
C.2 Fine-tuning . 21   
C.3 Fine-tuning regularization 21   
C.4 Gaussian perturbations 21   
C.5 Evaluation 21   
D Additional Results for Section 3 22   
D.1 Lowest-pretraining-loss setting 22   
D.2 Best-frontier setting . 23   
D.3 Choice of fine-tuning optimizer 25   
D.4 Fine-tuning regularization 25   
E Additional Results for Section 4 and Appendix B 29   
E.1 Margin-to-spread ratio across data budgets . 29   
E.2 Perturbation robustness . 31   
E.3 Margin-to-spread ratio and robustness 32   
F Proof of Theorem 4.2 34   
F.1 The two one-step iterates 34   
F.2 Concentration estimates for the data 35   
F.3 The GD step 38   
F.4 The Muon step 41   
F.5 Combining the GD and Muon estimates . 47   
F.6 Monotonicity in the data budget . 48

## A Definitions

## A.1 Optimizers

Optimizers update a model’s parameters during training so as to decrease a loss. Given the gradient of the loss with respect to the parameters, an optimizer determines the direction and the magnitude of each update, and in doing so shapes training stability, convergence speed and the solution that is finally reached. This section recalls the two optimizers compared in the paper.

## A.1.1 AdamW

Adam (Kingma & Ba, 2015) combines momentum with a per-coordinate step size. With $g _ { t } = \nabla _ { \theta } \mathcal { L } ( \theta _ { t } )$ the gradient at step t, it maintains exponential moving averages of the first and second moments of the gradient,

$$
\begin{array} { r } { m _ { t } = \beta _ { 1 } m _ { t - 1 } + ( 1 - \beta _ { 1 } ) g _ { t } , \qquad v _ { t } = \beta _ { 2 } v _ { t - 1 } + ( 1 - \beta _ { 2 } ) g _ { t } ^ { 2 } , } \end{array}
$$

with the square taken coordinatewise, corrects their bias towards the zero initialization,

$$
\begin{array} { r } { \hat { m } _ { t } = \frac { m _ { t } } { 1 - \beta _ { 1 } ^ { t } } , \qquad \hat { v } _ { t } = \frac { v _ { t } } { 1 - \beta _ { 2 } ^ { t } } , } \end{array}
$$

and updates

$$
\begin{array} { r } { \theta _ { t + 1 } = \theta _ { t } - \alpha \frac { \hat { m } _ { t } } { \sqrt { \hat { v } _ { t } } + \epsilon } , } \end{array}
$$

where $\alpha$ is the learning rate, $\beta _ { 1 } , \beta _ { 2 } \in [ 0 , 1 )$ are the decay rates of the two averages, $\epsilon > 0$ guards the division, and the division and the square root are coordinatewise. In efect each coordinate is normalized by the scale of its own recent gradients.

Regularizing Adam with an $\ell _ { 2 }$ penalty $\frac { \lambda } { 2 } \lVert \boldsymbol { \theta } \rVert ^ { 2 }$ adds $\lambda \theta _ { t }$ to g<sub>t</sub>, so the penalty is itself divided by $\sqrt { \hat { v } _ { t } } + \epsilon$ and coordinates with large past gradients are regularized less. AdamW (Loshchilov & Hutter, 2019) instead decouples the weight decay from the adaptive step,

$$
\begin{array} { r } { \theta _ { t + 1 } = \theta _ { t } - \alpha \frac { \hat { m } _ { t } } { \sqrt { \hat { v } _ { t } } + \epsilon } - \alpha \lambda \theta _ { t } , } \end{array}
$$

with $g _ { t }$ now the gradient of the unregularized loss and λ the weight decay coeficient, so that every coordinate decays at the same rate. AdamW is the standard optimizer for language-model pretraining and the baseline throughout the paper.

## A.1.2 Muon

Muon (Jordan et al., 2024) treats each weight matrix as a whole rather than coordinate by coordinate. For a matrixvalued parameter θ with gradient $G _ { t } ,$ , it accumulates momentum $M _ { t } = \mu M _ { t - 1 } + G _ { t }$ and updates along the orthogonalized momentum,

$$
\theta _ { t + 1 } = \theta _ { t } - \eta \mathrm { O r t h } ( M _ { t } ) ,\tag{4}
$$

where η is the learning rate and, for a matrix X with singular value decomposition $X = U \Sigma V ^ { \top } , \operatorname { O r t h } ( X ) = U V ^ { \top }$ is its polar factor: the matrix with the singular vectors of X and all singular values set to one. The update therefore has the same size along every singular direction, and it can be read as the steepest descent direction when the size of a step is measured in the spectral norm (Jordan et al., 2024). An exact singular value decomposition at every step is expensive, so Orth is approximated by a few steps of a Newton–Schulz iteration,

$$
\begin{array} { r } { X _ { 0 } = \frac { M _ { t } } { \| M _ { t } \| _ { F } } , \qquad X _ { k + 1 } = a X _ { k } + b ( X _ { k } X _ { k } ^ { \top } ) X _ { k } + c ( X _ { k } X _ { k } ^ { \top } ) ^ { 2 } X _ { k } , } \end{array}
$$

whose odd matrix polynomial pushes every singular value of $X _ { 0 }$ towards one; the reference implementation uses $( a , b , c ) = ( 3 . 4 4 4 5 , - \dot { 4 } . 7 7 5 0 , \dot { 2 . } 0 3 1 5 )$ and five steps. Muon is meant for the two-dimensional weight matrices of the hidden layers; embeddings, the output layer and one-dimensional parameters such as normalization gains are updated by AdamW. Large-scale implementations further add decoupled weight decay and rescale the update to the root-mean-square size of AdamW’s (Liu et al., 2025). In our experiments the hidden matrices are updated by Muon and all other parameters by AdamW, each with its own learning rate and weight decay (Section 3.1).

## A.2 Learning-rate schedules

## A.2.1 Warmup–stable–decay

All pretraining runs use a warmup–stable–decay (WSD) schedule: the learning rate increases linearly from zero to its peak value over the warmup steps, stays at the peak for most of training, and is annealed to zero over the final decay phase.

## A.2.2 Cosine

All fine-tuning runs use a cosine schedule: after a linear warmup of 20 steps, the learning rate follows $\eta _ { t } ~ =$ $\begin{array} { r } { \frac { 1 } { 2 } \eta _ { \mathrm { m a x } } \big ( 1 + \cos ( \pi t / T ) \big ) } \end{array}$ over the remaining T steps.

## A.3 Token–parameter ratio

We measure the pretraining data budget of a model with N parameters trained on D tokens by its token–parameter ratio, $\mathrm { T P R } = \hat { D ^ { \prime } } \hat { N }$ . The compute-optimal budget of Hofmann et al. (2022) corresponds to $\mathrm { T P R } = 2 0 ;$ our budgets range from 5 to 2,560 TPR.

## B Other Metrics for Analyzing Post-Training

Besides the margin-to-spread ratio, the robustness of a base model to post-training can be probed by perturbing its weights directly. This section reports Gaussian perturbation results in language models and in the linear memory (Section B.1), and then relates them to the MSR (Section B.2).

## B.1 Gaussian perturbation

We first check whether Gaussian perturbations of the weights reproduce the budget trend of the learning–forgetting tradeof, in language models and in the linear memory.

## B.1.1 Language models

We test whether the perturbed loss of the pretrained language models shifts from Muon toward AdamW as the token budget grows, as the learning–forgetting tradeof does.

Setup. Springer et al. (2025) show that adding isotropic Gaussian noise to pretrained weights reproduces key empirical trends in forgetting after fine-tuning. Motivated by this connection, we study Gaussian weight perturbations as a simpler setting in which to investigate why the learning–forgetting gap between Muon and AdamW changes with pretraining budget. At the same checkpoints as in Section 3, we add isotropic Gaussian noise to each weight matrix, scaled to a fraction δ of the matrix’s norm, and measure the loss increase $\Delta _ { \mathrm { n o i s e } } ( \theta _ { \mathrm { P T } } ; \delta ) : = \mathcal { L } _ { \mathrm { P T } } ( \widetilde { \theta } _ { \delta } ) - \mathcal { L } _ { \mathrm { P T } } ( \theta _ { \mathrm { P T } } )$ where $\widetilde { \theta } _ { \delta }$ denotes the perturbed weights; a smaller increase indicates greater perturbation robustness. See Section E.2 for full experimental details.

Result. Figure 6 plots the gap in loss degradation, $\Delta _ { \mathrm { n o i s e } } ( \mathrm { M u o n } ; \delta ) - \Delta _ { \mathrm { n o i s e } } ( \mathrm { A d a m W } ; \delta )$ . This gap is positive across the token budgets tested and grows with the pretraining budget: AdamW incurs a smaller loss increase throughout the sweep, and its advantage grows with budget.

To connect this gap to the learning–forgetting tradeof, we also account for pretraining loss: loss after perturbation equals pretraining loss plus $\Delta _ { \mathrm { n o i s e } } .$ . Muon starts with lower pretraining loss, so it can retain lower loss after perturbation at small budgets despite its larger loss increase. As its pretraining-loss advantage shrinks and the degradation gap grows, AdamW achieves lower loss after perturbation beyond a crossover budget. For suficiently strong perturbations, this crossover occurs at smaller token budgets as model size increases, paralleling the crossover observed after fine-tuning (Table 2 in Section E.2).

![](images/ddb58f681b5b7b7d7b1d7e8ea9b16ffb6be566fe5b78962c26174f5797eda3aa.jpg)  
Figure 6: The perturbation degradation gap grows with pretraining data budget. Under a perturbation of magnitude $\delta = 0 . 0 2$ , Muon degrades increasingly more than AdamW as the pretraining data budget grows. Points show the mean over three noise draws.

## B.1.2 Linear associative memory

We repeat the perturbation experiment in the linear memory, where the efect of a perturbation can be measured fact by fact.

Setup. We train the memory of Section 4.1 with GD and Muon to minimize cross-entropy, varying the data budget $\gamma = N / d$ and the dimension $d ,$ as in Section 4.2. The trained weights are perturbed to $\mathbf { \hat { \boldsymbol { W } } } ^ { \prime } = \mathbf { \dot { \boldsymbol { W } } } + \sigma \| \boldsymbol { W } \| _ { F } \boldsymbol { \xi }$ with $\| \xi \| _ { F } \overset { \cdot } { = } 1$ , over 16 perturbation directions shared by the two optimizers. Since the memory retrieves facts, we measure degradation by the fraction of initially recovered facts that are lost. At a single perturbation magnitude, fact loss can be close to zero or one, making memories with diferent noise tolerances dificult to distinguish. We therefore compare the perturbation strengths needed to reach a common intermediate level of degradation: $\sigma _ { 5 0 }$ is the relative perturbation strength at which half of the initially recovered facts are lost. A larger $\sigma _ { 5 0 }$ means that more noise is needed to cause the same fraction of facts to be lost. Data budgets past the memory’s capacity, where the two optimizers recover diferent sets of facts, are excluded. Full details appear in Sections E.1 and E.3.

Result. The robustness ordering of the two optimizers turns over with the data budget: Muon tolerates stronger perturbations at small $\gamma ,$ and GD does at large γ (Figure 4b and the bottom row of Figure 20). The $\sigma _ { 5 0 }$ ordering flips at γ ≈ 0.5, 1.3 and 11 at $d = 1 2 8$ , 256 and 512, and at $d = 6 4 \mathrm { G D }$ is already the more robust at the smallest data budget we ran.

## B.2 Gaussian perturbation and the margin-to-spread ratio

We next relate robustness to Gaussian perturbations to the MSR, first in theory and then in experiments.

## B.2.1 Theoretical analysis

We use the scores $s _ { i j } ( W ) = v _ { i } ^ { \top } W u _ { i }$ of Section 4.1 and the margins $m _ { i j } , h _ { i } , \kappa _ { i }$ and $\rho _ { i }$ of Definition 4.1, for arbitrary keys, values and W. The weights are perturbed to $W + \sigma E$ , where the entries of E satisfy

$$
\mathbb { E } [ E _ { k l } ] = 0 , \qquad \mathbb { E } [ E _ { k l } E _ { k ^ { \prime } l ^ { \prime } } ] = { \bf 1 } \{ ( k , l ) = ( k ^ { \prime } , l ^ { \prime } ) \} .\tag{5}
$$

Scores are linear in $W$ , so exactly

$$
s _ { i i } ( W + \sigma E ) - s _ { i j } ( W + \sigma E ) = m _ { i j } + \sigma ( v _ { i } - v _ { j } ) ^ { \top } E u _ { i } .
$$

We measure robustness by the fraction of competitors still ranked below the correct value,

$$
\mathrm { A U C } _ { i } ( \sigma ) = \operatorname* { m e a n } _ { j \neq i } \operatorname* { P r } _ { E } \left[ s _ { i i } ( W + \sigma E ) > s _ { i j } ( W + \sigma E ) \right] .
$$

Let $J$ be uniform on $\{ 1 , \ldots , N \} \setminus \{ i \}$ , independent of $E .$ . Then

$$
\mathrm { A U C } _ { i } ( \sigma ) = \operatorname* { P r } \big [ s _ { i i } ( W + \sigma E ) - s _ { i J } ( W + \sigma E ) > 0 \big ] = \operatorname* { P r } \big [ m _ { i J } + \sigma ( v _ { i } - v _ { J } ) ^ { \top } E u _ { i } > 0 \big ] ,
$$

so it sufices to bound the probability that one random variable is positive, from its mean and variance.

Lemma B.1 (Cantelli). $I f \mathbb { E } X > 0$ and Var $X < \infty$ , then

$$
\operatorname* { P r } [ X \leq 0 ] \ \leq \ { \frac { \operatorname { V a r } X } { \operatorname { V a r } X + ( \mathbb { E } X ) ^ { 2 } } } .
$$

Proof. For $t \geq 0$ , Markov’s inequality gives $\operatorname* { P r } [ X \leq 0 ] \leq \mathbb { E } ( \mathbb { E } X - X + t ) ^ { 2 } / ( \mathbb { E } X + t ) ^ { 2 } = ( \operatorname { V a r } X + t ^ { 2 } ) / ( \mathbb { E } X + t ) ^ { 2 } ;$ take $t = \operatorname { V a r } X / \mathbb { E } X$ □

Lemma B.2 (Noise adds variance, not bias). Let E satisfy Equation (5). Then

$$
\begin{array} { r l } & { \quad \mathbb { E } \big [ m _ { i J } + \sigma ( v _ { i } - v _ { J } ) ^ { \top } E u _ { i } \big ] = h _ { i } , } \\ & { \quad \mathrm { V a r } \big [ m _ { i J } + \sigma ( v _ { i } - v _ { J } ) ^ { \top } E u _ { i } \big ] = \kappa _ { i } ^ { 2 } + \sigma ^ { 2 } \| u _ { i } \| ^ { 2 } \underset { j \neq i } { \mathrm { m e a n } } \| v _ { i } - v _ { j } \| ^ { 2 } . } \end{array}
$$

Proof. For fixed $\begin{array} { r } { j , ( v _ { i } - v _ { j } ) ^ { \top } E u _ { i } = \sum _ { k , l } ( v _ { i } - v _ { j } ) _ { k } E _ { k l } ( u _ { i } ) _ { l } } \end{array}$ has, by Equation (5), mean 0 and second moment $\| v _ { i } - v _ { j } \| ^ { 2 } \| u _ { i } \| ^ { 2 }$ . Average over E first, then over J. □

Theorem B.3 (Perturbed AUC and the margin-to-spread ratio). Let E satisfy Equation (5). For every W, all data, every $\sigma \geq 0$ and every i with $h _ { i } > 0$

$$
\begin{array} { r } { \mathrm { A U C } _ { i } ( \sigma ) \ \geq \ \frac { h _ { i } ^ { 2 } } { h _ { i } ^ { 2 } + \kappa _ { i } ^ { 2 } + \sigma ^ { 2 } \| u _ { i } \| ^ { 2 } \operatorname* { m e a n } _ { j \neq i } \| v _ { i } - v _ { j } \| ^ { 2 } } . } \end{array}
$$

Proof. $1 - \mathrm { A U C } _ { i } ( \sigma ) = \mathrm { P r } \left[ m _ { i J } + \sigma ( v _ { i } - v _ { J } ) ^ { \top } E u _ { i } \le 0 \right]$ . Apply Lemma B.1 with the mean and variance of Lemma B.2. □

No assumption is made on $W$ , on the shape of the margin distribution, or on d and N.

Corollary B.4 (Gaussian data). If moreover $u _ { 1 } , \dotsc , u _ { N } , v _ { 1 } , \dotsc , v _ { N }$ are i.i.d. $\mathcal { N } ( 0 , I _ { d } )$ , then

$$
\mathrm { A U C } _ { i } ( \sigma ) \geq \frac { h _ { i } ^ { 2 } } { h _ { i } ^ { 2 } + \kappa _ { i } ^ { 2 } + 2 \sigma ^ { 2 } d ^ { 2 } \big ( 1 + O _ { p } ( d ^ { - 1 / 2 } ) \big ) } = \frac { \rho _ { i } ^ { 2 } } { \rho _ { i } ^ { 2 } + 1 + 2 ( \sigma d / \kappa _ { i } ) ^ { 2 } \big ( 1 + O _ { p } ( d ^ { - 1 / 2 } ) \big ) } .
$$

Proof. $\| u _ { i } \| ^ { 2 } = d \big ( 1 + O _ { p } ( d ^ { - 1 / 2 } ) \big )$ and mean $_ { j \neq i } \| v _ { i } - v _ { j } \| ^ { 2 } = 2 d \big ( 1 + O _ { p } ( d ^ { - 1 / 2 } ) \big )$

At a fixed noise level in units of $\kappa _ { i } ,$ , a larger $\rho _ { i }$ therefore guarantees a larger AUC. For the relative perturbation $\boldsymbol { W } + \delta \lVert \boldsymbol { W } \rVert _ { F } \boldsymbol { E } / \lVert \boldsymbol { E } \rVert _ { F }$ used in the experiments, with i.i.d. standard Gaussian entries of E, the bounds hold with σ replaced by $\delta \lVert \dot { W } \rVert _ { F } \ddot { / } d .$

Corollary B.5 (Noise on the logits). Suppose instead that each logit is perturbed directly, to $s _ { i j } ( W ) + \sigma \xi _ { i j } ,$ where the $\xi _ { i j }$ have mean 0 and variance 1 and are uncorrelated (for example, i.i.d. $\mathcal { N } ( 0 , 1 ) )$ . Then for every W, every $\sigma \geq 0$ and every i with $h _ { i } > 0$

$$
\operatorname* { m e a n } _ { j \neq i } \operatorname* { P r } _ { \xi } \left[ s _ { i i } ( W ) + \sigma \xi _ { i i } > s _ { i j } ( W ) + \sigma \xi _ { i j } \right] \geq \frac { h _ { i } ^ { 2 } } { h _ { i } ^ { 2 } + \kappa _ { i } ^ { 2 } + 2 \sigma ^ { 2 } } = \frac { \rho _ { i } ^ { 2 } } { \rho _ { i } ^ { 2 } + 1 + 2 \sigma ^ { 2 } / \kappa _ { i } ^ { 2 } } .
$$

Proof. With J uniform on $\{ 1 , \ldots , N \} \setminus \{ i \}$ and independent of $\displaystyle { \dot { \xi } } ,$ the variable $m _ { i J } + \sigma ( \xi _ { i i } - \xi _ { i J } )$ has mean $h _ { i }$ and variance $\kappa _ { i } ^ { 2 } + 2 \sigma ^ { 2 }$ , since $\xi _ { i i } - \xi _ { i j }$ has mean 0 and variance 2 for every $j \neq i . \mathrm { A p p l y }$ Lemma B.1. □

A fact is mislabeled only if some incorrect answer overtakes the correct one, so by a union bound the probability of mislabeling fact i is at most $N - 1$ times one minus the left-hand side. This bound is informative only when $\dot { \rho _ { i } ^ { 2 } }$ is much larger than $N _ { \cdot }$

## B.2.2 Empirical validation

We test whether the MSR predicts how much perturbation a checkpoint can tolerate, first in language models and then in the linear memory.

Language models. We measure MSR as in Section 4.3 and degradation by the held-out loss increase of Section B.1.1. The robustness threshold $\delta ^ { * }$ is the perturbation magnitude at which this increase reaches $\Delta L$ nats. Across checkpoints, $\rho$ and $\delta ^ { * }$ are strongly correlated for both $\Delta L = 0 . 0 5$ and 0.1 (Figure 7b): the Spearman correlation over the 72 checkpoints is $+ 0 . { \bar { 8 3 } }$ at both thresholds, and within each optimizer it i $\mathsf { s } + 0 . 8 7 \mathsf { t o } ^ { - } + 0 . 8 9$ for AdamW and +0.92 to +0.93 for Muon. Full experimental details appear in Section E.3.

Linear associative memory. With $\sigma _ { 5 0 }$ as in Section B.1.2, $\sigma _ { 5 0 }$ is strongly correlated with MSR for both optimizers (Figure 7a). Within each optimizer, σ and $\rho$ move together at every model dimension (Figure 23).

(a)  
GD  
![](images/0119b64400842fea58f1c3aeccac3304afe4653a18d11cf23f2d887de49f458c.jpg)

![](images/073d7f1add4264ed5fbfe55c2493b49332c9d039e8231a64f573c779905f48aa.jpg)

![](images/9d03724946c2422f9c832519140ddc3cd5fc350e9382442fb3e538be36f18495.jpg)  
Language models

![](images/0c13fb6fb89ef61555f8f1a3b34feb59dd48e8a7c7dc17410299d69ae564c26c.jpg)  
Figure 7: The margin-to-spread ratio tracks robustness to weight perturbations. (a) For the linear models trained under GD and Muon, we plot the perturbation $\sigma _ { 5 0 }$ required to lose half of the recovered facts against its MSR, $\bar { h } / \bar { \kappa }$ (Definition 4.1). A higher MSR requires more noise to cause the same loss. (b) We measure the MSR of 72 pretrained language models (30M–600M) as the median of $h / \kappa$ over correctly predicted tokens, and plot the perturbation $\delta ^ { * }$ at which held-out loss rises by $\Delta L .$ The same relationship holds.

## C Experimental Details

## C.1 Pretraining

Models and data. All language models are OLMo-style decoder-only transformers (Groeneveld et al., 2024; Team OLMo et al., 2025) with 30M, 60M, 100M, 300M and 600M parameters, trained on DCLM (Li et al., 2024) with the OLMo 2 tokenizer. Budgets span 5–1,280 TPR at 30M, 5–2,560 TPR at 60M, 5–640 TPR at 100M, 5–160 TPR at 300M and 5–40 TPR at 600M.

Optimization. Every model is pretrained with sequence length 4096, a global batch of $2 ^ { 2 0 }$ tokens, a warmup– stable–decay (WSD) schedule, weight decay 0.1 and gradient clipping at 1.0. AdamW uses $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 \bar { 8 } )$ Muon updates the hidden weight matrices; the remaining parameters (embeddings, output layer and norms) are updated by AdamW.

## C.1.1 Lowest-pretraining-loss seting

For each model size, budget and optimizer, we sweep the peak learning rate and call the run with the lowest held-out DCLM loss the loss-optimal base model. For Muon we sweep the Muon learning rate and fix the learning rate of the AdamW-updated parameters to the tuned AdamW value of the same cell (Section C.1.2). Table 1 lists the resulting learning rates. In the lowest-pretraining-loss setting, used throughout the main text, each optimizer is represented by its loss-optimal base model alone: we fine-tune it over the full sweep of Section C.2 and compute the learning–forgetting frontier from its fine-tuning runs.

## C.1.2 Learning rate of Muon’s AdamW-updated parameters

Muon has a second learning rate for the parameters it leaves to AdamW. Rather than tuning it in every cell, we set it to the tuned AdamW learning rate of the same cell. To check this choice, we pretrain 60M Muon models at 20–160

Table 1: Tuned peak pretraining learning rates per model size and budget. For Muon, the Muon learning rate; the AdamWparameter learning rate is fixed to the tuned AdamW value.
<table><tr><td rowspan="2">Size</td><td rowspan="2">Optimizer</td><td colspan="10">TPR</td></tr><tr><td>5</td><td>10</td><td>20</td><td>40</td><td>80</td><td>160</td><td>320</td><td>640</td><td>1,280</td><td>2,560</td></tr><tr><td rowspan="2">30M</td><td>AdamW</td><td>0.08</td><td>0.04</td><td>0.04</td><td>0.014</td><td>0.02</td><td>0.014</td><td>0.02</td><td>0.014</td><td>0.01</td><td>一</td></tr><tr><td>Muon</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.014</td><td>0.02</td><td>0.01</td><td>0.014</td><td>0.01</td><td>0.014</td><td>一</td></tr><tr><td rowspan="2">60M</td><td>AdamW</td><td>0.056</td><td>0.04</td><td>0.02</td><td>0.02</td><td>0.01</td><td>0.014</td><td>0.01</td><td>0.01</td><td>0.007</td><td>0.01</td></tr><tr><td>Muon</td><td>0.014</td><td>0.01</td><td>0.014</td><td>0.01</td><td>0.014</td><td>0.01</td><td>0.014</td><td>0.007</td><td>0.005</td><td>0.005</td></tr><tr><td rowspan="2">100M</td><td>AdamW</td><td>0.04</td><td>0.02</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.007</td><td>0.007</td><td>0.007</td><td></td><td>一</td></tr><tr><td>Muon</td><td>0.014</td><td>0.014</td><td>0.014</td><td>0.014</td><td>0.01</td><td>0.014</td><td>0.01</td><td>0.005</td><td>一 一</td><td>1</td></tr><tr><td rowspan="2">300M</td><td>AdamW</td><td>0.001</td><td>0.0025</td><td>0.0035</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Muon</td><td>0.02</td><td>0.014</td><td>0.02</td><td>0.0025 0.014</td><td>0.0025 0.014</td><td>0.0025 0.02</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td rowspan="2">600M</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1</td><td>一</td><td>一</td><td>一</td></tr><tr><td>AdamW Muon</td><td>0.0025 0.014</td><td>0.0025 0.014</td><td>0.0025 0.014</td><td>0.0025 0.014</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td></tr></table>

TPR over a grid of Muon learning rates and three to four AdamW learning rates between 0.5× and 2.8× the tuned value. Figure 8 shows the best loss over Muon learning rates for each AdamW learning rate. It varies by at most 0.021 nats, and the tuned value is within 0.016 nats of the best, well below the 0.03–0.20 nats by which Muon beats AdamW at these budgets.

![](images/87c703068c7d697e781f14df74e5ef1f856d0fdc0f26e335bc35b5b921174f71.jpg)  
Figure 8: Muon is insensitive to the learning rate of its AdamW-updated parameters. Best pretraining loss over Muon learning rates for each AdamW learning rate, relative to the tuned AdamW value (dashed), for 60M models.

## C.1.3 Best-frontier seting

The loss-optimal base model is not always the best starting point for fine-tuning: models pretrained with a larger learning rate can ofer a better learning–forgetting tradeof (Watts et al., 2026). We therefore also compare the two opti mizers in the best-frontier setting, which sweeps over both pretraining and fine-tuning configurations. For each model size, budget and pretraining optimizer, we fine-tune the loss-optimal base model and the base models pretrained at the next five larger learning rates of the sweep grid {0.005, 0.007, 0.01, 0.014, 0.02, 0.028, 0.04, 0.056, 0.08, 0.12, 0.16}, pool all of their fine-tuning runs, and compute a single learning–forgetting frontier from the pooled runs. This frontier is at least as good as the one of the lowest-pretraining-loss setting, since it contains the loss-optimal base model’s runs. Section D.2 repeats the main results in this setting.

## C.2 Fine-tuning

Each base model is fine-tuned for 20M tokens on each of Alpaca (Taori et al., 2023), GSM8K (Cobbe et al., 2021), MusicPile (Yuan et al., 2024) and StackMathQA (Zhang, 2024), with sequence length 1024, a global batch of 65,536 tokens (about 305 steps), a cosine schedule with 20 warmup steps, weight decay 0.1, $( \beta _ { 1 } , \bar { \beta } _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$ and gradient clipping at 1.0. AdamW-pretrained models are fine-tuned with AdamW, and Muon-pretrained models with both AdamW and Muon (Section D.3 motivates this choice). We sweep AdamW learning rates from $1 0 ^ { - 4 } \mathrm { t o } 1 0 ^ { - 2 }$ and Muon learning rates from $6 \times 1 0 ^ { - 4 } \mathrm { t o } 2 \times 1 0 ^ { - 2 }$ with the learning rate of the AdamW-updated parameters set to $0 . 2 5 \times$ the Muon learning rate. The sweep ranges from runs that barely move the base model to runs that reach the lowest fine-tuning loss at the cost of more than a nat of forgetting.

## C.3 Fine-tuning regularization

A natural question is whether the reversal survives when fine-tuning itself is regularized to limit forgetting. We test two standard methods for mitigating catastrophic forgetting, data replay (Shin et al., 2017) and elastic weight consolidation (EWC; Kirkpatrick et al., 2017), on the loss-optimal base models of every size. Both keep the fine-tuning recipe and learning-rate sweep of Section C.2, and each frontier pools all strengths of the regularizer. Section D.4 reports the results.

Replay. We replace a fraction $r \in \{ 0 . 1 , 0 . 2 , 0 . 3 \}$ of the fine-tuning tokens with DCLM data drawn from a shard that is disjoint from both the pretraining data and the held-out evaluation shard.

EWC. EWC penalizes movement away from the pretrained weights in the directions that matter for pretraining. It minimizes

$$
\mathcal { L } _ { \mathrm { F T } } ( \boldsymbol { \theta } ) + \frac { \lambda } { 2 } \sum _ { k } F _ { k } \big ( \theta _ { k } - \theta _ { \mathrm { P T } , k } \big ) ^ { 2 } ,\tag{6}
$$

where F is the diagonal empirical Fisher information of the pretraining loss at $\theta _ { \mathrm { P T } } ,$ , estimated from squared gradients over 100 batches of the disjoint DCLM shard used for replay. Following Watts et al. (2026), F is not normalized, so we sweep $\lambda \in \{ 1 0 ^ { 3 } , 4 \cdot 1 0 ^ { 5 } , 1 0 ^ { 4 } , 4 \cdot 1 0 ^ { 4 } \}$

## C.4 Gaussian perturbations

We perturb each base model by adding independent isotropic Gaussian noise to every parameter tensor $W _ { i }$ , scaled so that $\| \Delta W \| _ { F } = \delta \| W \| _ { F }$ , and measure the perturbation degradation $\Delta _ { \mathrm { n o i s e } } ( \theta _ { \mathrm { P T } } ; \delta )$ of Section B.1.1. We average over three noise draws per checkpoint and magnitude, use the loss-optimal checkpoint of each cell, and consider magnitudes from $\delta = 0 . { \bar { 0 0 2 } }$ to 0.56.

## C.5 Evaluation

Losses. Forgetting is the loss on a held-out DCLM shard of about 4.19M tokens that no model is trained on, and learning is the loss on the validation split of the fine-tuning dataset. Both are mean next-token cross-entropy in nats.

Frontiers. Each fine-tuned checkpoint $\theta _ { \mathrm { F T } }$ of a base model gives a point $\left( \mathcal L _ { \mathrm { P T } } ( \theta _ { \mathrm { F T } } ) , \mathcal L _ { \mathrm { F T } } ( \theta _ { \mathrm { F T } } ) \right)$ . We compute the learning–forgetting frontier as the lower-left arc of the convex hull of these points, treated as a polyline.

Fine-tuning gap and crossover. We estimate the fine-tuning gap $G _ { \mathrm { F T } }$ of Section 2 from 100 points sampled uniformly by arc length along the AdamW frontier, each assigned its Euclidean distance to the nearest point of the Muon frontier, negative when that point lies closer to the origin. The crossover TPR is the smallest budget at which $G _ { \mathrm { F T } } > 0$ . In both settings, the gap and crossover compare the two optimizers’ frontiers as defined above: one base model per optimizer in the lowest-pretraining-loss setting, pooled base models in the best-frontier setting.

Degradation. For Figure 3c, the degradation of a base model is the increase in held-out DCLM loss caused by fine-tuning, $\mathcal { L } _ { \mathrm { P T } } ( \theta _ { \mathrm { F T } } ) - \mathcal { L } _ { \mathrm { P T } } ( \theta _ { \mathrm { P T } } )$ . We read ${ \mathcal { L } } _ { \mathrm { P T } } ( \theta _ { \mathrm { F T } } )$ of the base model’s frontier at a target fine-tuning loss of 1.1× the lowest fine-tuning loss reached by either optimizer at that model size, so that both optimizers are compared at the same amount of learning.

## D Additional Results for Section 3

This section extends the fine-tuning results of Section 3. Unless noted otherwise, figures show 60M models, and frontiers are restricted to the region of Section C.5.

## D.1 Lowest-pretraining-loss setting

Figure 9 shows that the loss-optimal Muon model reaches a lower pretraining loss than its AdamW counterpart at every budget and model size, so the reversal is not inherited from pretraining loss. Figures 10 and 11 extend the frontiers and gaps of the lowest-pretraining-loss setting (Figures 1 and 3) to all four datasets, at the budgets around each crossover. On every dataset Muon is better than or level with AdamW below the crossover and AdamW is better beyond it.

AdamW Muon  
![](images/7aae99a48d5b431d61b530f4cc35a9666aba6cde0ffbc3d8e47e554adba40ec3.jpg)

![](images/10bc0f5fb3c1f978f1e4c99e5df2ec41e499c18f57e522bea3dea64a658bdc01.jpg)

![](images/86d00b71a179b0093f855f02419f1be4985b324f3b618b0c78facd7fdd587609.jpg)

![](images/d978eecda7e7ac7784ddf81dd09db42a4b90e52ca24d8d0bd3bf739f47d33754.jpg)

![](images/b0591d77844c21a6bde23fa00584a603ec16b5f2fc71893cd93262e15cd51b92.jpg)

Figure 9: Muon reaches lower pretraining loss than AdamW. DCLM held-out loss of the loss-optimal base models across pretraining data budgets and model sizes.  
![](images/62e433f5ccc94ab5b29ab9c6c82bbbfbee4f5fca5bcf7d3eedc0d141c5b0eb04.jpg)

![](images/dabca25de793244e2ebb8e9975b4b58efb2965e97e39e075e8989b9fd9d10c80.jpg)

![](images/b0cb4f0578d27320019e06d6abb9c115402f9247104e21122cf521bf72da909c.jpg)  
Figure 10: Fine-tuning gap on every fine-tuning dataset, loss-optimal base models (60M). Negative values mean the Muon frontier dominates.

![](images/a6825fcab00c625c1f351a1a2b1741dc6b817d2d7532a16eddb8fbba2b7643de.jpg)

![](images/55af4f8f6ec7cfdfa7f3d9875a37f6b44ee43a371779ab88e6f021076dab4736.jpg)  
Figure 11: Learning–forgetting frontiers on every fine-tuning dataset, loss-optimal base models (60M). Each row shows the budgets around that dataset’s crossover TPR.

## D.2 Best-frontier setting

We repeat the analyses of Section 3 in the best-frontier setting (Section C.1.3), where each pretraining optimizer is swept across all pretraining and fine-tuning configurations. Figures 12 and 13 show the learning–forgetting frontiers and fine-tuning gaps of 60M models on all four fine-tuning datasets, at the budgets around each crossover. On every dataset, the Muon frontier is better below the crossover; at the crossover AdamW has caught up with or overtaken Muon, and beyond it the AdamW frontier dominates. The fine-tuning gap turns positive at the crossover and stays positive, broadly growing with the data budget. Figure 14 shows that, as in the lowest-pretraining-loss setting (Figure 2), the crossover TPR decreases with model size.

![](images/4de08ccad7b1128bc6e876059a0b47b817eb368924be0cbaa21aeba288ce57eb.jpg)

![](images/b693e1491c87524f7f398bb1caa7a035bb663eeb1f0e06e3ffb3e38af19b6b8c.jpg)

![](images/40669e8dd4beaa06718c07d60ec8c3875623f332fc3a7c78f00caa07a7d17c9d.jpg)

![](images/18a3dd61a67b51a745ba797c3cafa28e04075a781f98cdca9515e05185ac2f24.jpg)  
Figure 12: Fine-tuning gap on every fine-tuning dataset, best-frontier setting (60M). Negative values mean the Muon frontier dominates.

![](images/4ba2b4a651b8e3d53b3d770a33e928c54b573449b95081280d855169677917b8.jpg)  
Figure 13: Learning–forgetting frontiers on every fine-tuning dataset, best-frontier setting (60M). Each row shows the budgets around that dataset’s crossover TPR.

Forgetting  
Forgetting  
Forgetting  
![](images/c57ec97d3d2f8f53050dc126d66dba854f6429d21e2bfe389bc27a0065f5f9b6.jpg)

![](images/da2bb7747b5f19b3fc8f87680c38b110eaacf0368546296f1402b88b7dc8c3e1.jpg)

![](images/87a401f7f32c363a4c61e291cf5793e8fd8f924d7c5b9b9d50030a293411a17d.jpg)  
Model size (M)

![](images/7ff44ce10153cfc7d2657dfc8b6312a363f237e7df9d3854c35aa7d8da7e3464.jpg)  
Figure 14: Crossover TPR decreases with model size, in the best-frontier setting. As model size increases, Muon’s advantage reverses with substantially smaller pretraining data budgets.

## D.3 Choice of fine-tuning optimizer

Our frontiers fine-tune AdamW-pretrained models with AdamW and Muon-pretrained models with both AdamW and Muon. Figure 15 motivates this choice, and shows that Muon’s greater forgetting at large budgets is a property of the pretrained model rather than of the fine-tuning recipe. For each pretraining optimizer and budget, we take the best 60M base model among those fine-tuned with both optimizers, and plot its AdamW- and Muon-fine-tuned frontiers separately. We use GSM8K, the dataset on which AdamW-pretrained models were fine-tuned with both optimizers across the full range of budgets.

Two observations follow. First, the fine-tuning optimizer matters for AdamW-pretrained models but not for Muonpretrained ones: on AdamW-pretrained models, AdamW fine-tuning dominates Muon fine-tuning at every budget, whereas the two frontiers of a Muon-pretrained model lie close together with no consistent winner. Second, the reversal does not depend on how the Muon-pretrained model is fine-tuned: at 1.2B tokens both ofits frontiers lie below those of the AdamW-pretrained model, and from 19.2B tokens onward the AdamW-pretrained, AdamW-fine-tuned frontier dominates both.

![](images/06436210574fc16e82c42446c3e33298014003396baa9fab406b216925a433f3.jpg)

![](images/95bf3cf65b6f83d576d71a22141c6a966ed9abaa0b15803b51825f9370233170.jpg)  
AdamW pretrain, AdamW fine-tune

![](images/0262dba46676aef8747e17be32b55b10eabd580b052531e55f6b148c599263fe.jpg)

![](images/3d92f18d0d5ef30ef74fcd4ddd83b9a63a4ddd53b93c2a93612963e5772f85d8.jpg)  
Muon pretrain, AdamW fine-tune  
AdamW pretrain, Muon fine-tune  
Muon pretrain, Muon fine-tune  
Figure 15: Muon’s greater forgetting comes from the pretrained model, not the fine-tuning optimizer. Learning–forgetting frontiers of the best 60M base models on GSM8K, split by fine-tuning optimizer (solid: AdamW; dashed: Muon). AdamW fine-tuning is always better for AdamW-pretrained models, while the two fine-tuning optimizers give nearly the same frontier for Muon-pretrained models. At large budgets, the AdamW-pretrained model dominates under either choice.

## D.4 Fine-tuning regularization

We repeat the comparison when fine-tuning is regularized with replay or EWC (setup in Section C.3), using the loss-optimal 60M base models. Figures 16 and 17 show the frontiers on all four datasets; each row shows the budgets around that dataset’s crossover under the regularizer, a quarter and half of the crossover TPR, the crossover itself, and four times it. The crossover is the first budget at which the fine-tuning gap turns positive (Section C.5). Both regularizers reduce forgetting and can move the crossover to larger budgets (from 80 to 320 TPR on Alpaca and StackMathQA), but AdamW’s frontier is better at large budgets on every dataset under both (Figure 18), with one exception: under EWC on StackMathQA the gap turns negative again at the largest budget, 2,560 TPR. With replay,

Muon’s frontier is better than or level with AdamW’s below the crossover on every dataset. With EWC the picture below the crossover is mixed: Muon is clearly better at small budgets on GSM8K, while on the other datasets the two frontiers are close.

![](images/af9d916def7f40e76c83ee4868d21d57a41ae0655e5eb7579cc00adcf67dddfc.jpg)

![](images/2f6e9af0ff28556fcd0bb33ef8f548e59a606c641aa16b2f0cf8dbd7bc87cf1b.jpg)

![](images/5b8b9cd080b346de8079d826cfa7226d76f8d2fdeb44a71c3608dd7770546048.jpg)

![](images/8c741a7db13056a049837f9f81214c0011c27fc6b84b5ea10598ae44d5b03b44.jpg)

![](images/ceaffe580d5cd5d1296ad58b99694e6a0ca5ec8aa177491c1eab3894b3abc7bd.jpg)

![](images/3c7df0907abe2b92de2ce008a791f805a1fbe4fb1dbd04ba8bff9b2c9d4befe2.jpg)

![](images/67155f109f5c778be15b9d39c762d403444156e183c0c1e4809e113ed0a69515.jpg)

![](images/e6986bb7a1856323ccd16bfd74f7fe3af479bea87a340a1089ce01e77f5a6eac.jpg)

![](images/822e78ed7a5131a744b30462701d03a6bbf0b36e67b4f17a6acafd6bca2aeb71.jpg)

![](images/65a495a598259857198639d5da56629eaaf899e4fd65523ac6b70d1628638088.jpg)

![](images/343eaeaa74f70acffd4fd0cd3a956fde4b3a18bca73d100ea56bef01dcc674cc.jpg)

![](images/f55e775b7b6e9a27177d8ddcda712683828b8a6f9e9dd917d529a8e4b77f9080.jpg)

![](images/a58b025845bd0d441efd587f89d9ee10bbec3ca36611585b80a9c718da62b29f.jpg)

![](images/3e97148c220c748bdbb7989f040c291348e575e74f086131ef15e80ea8b9136a.jpg)

![](images/43c6cf45f684a3f9c11107c01ff095af824cd7a4781b98d0f20f9002827c44f7.jpg)

![](images/df31fb8c988acdd774c3a75447a77f6b6c39cc1331e7b59fbef094385b65c1bc.jpg)  
Figure 16: Learning–forgetting frontiers with replay of 10–30% DCLM data, loss-optimal 60M base models. Each row shows the budgets around that dataset’s crossover under replay.

![](images/91f08f77708d3731d4c4ff65081dd9f2b2ed25f82a5374aacf3196f15c41635f.jpg)

![](images/66fc0693ec6aeadde0077b06cc2f3eba7fc2eeeba9837598ccc690f23e5f5235.jpg)

![](images/a86e6c15bced1373cd6bd3f2e7faf1dbda4b83840bd56211ac78021acefc45a7.jpg)

![](images/18061d0444805e406ae215b4b6791a46b7b930e37ccab0df5677deb0a7dfc9ad.jpg)

![](images/61d95b5aa3ae4e94e98ab1c469c207ead6afca285ff6fa321323e68e909461a6.jpg)

![](images/036c003bf2bcf4ce7d01ab69227621efeeedd366d9c369245583c94e9f35cef8.jpg)

![](images/28f59bf18b93210abf2ade9dafdb557f20db1ea9e772380ecea9df921d888668.jpg)

![](images/6fd8ff01dce67a8422125207adb9b2d88d3078d665e4a9dbeea566f202d7793e.jpg)

![](images/d9f01164e844418973bce955e04f1351ccc9720d9886e2e3ef575c83a29b30e2.jpg)

![](images/9ee93233068388e67c00b4a5344f24050c98d3c6d645460d52dafcdae1cc1871.jpg)

![](images/9a168b8d207bc1f14352a18034ca8ca9f98576db42c6379bb8be263962d05dfe.jpg)

![](images/2385df04d966967feb8989dd1fa10643b288cff5b32e3953f973df39cb9ebc72.jpg)

![](images/08d39c88b25f21f5861d3a5c3bee8fa43969042ede8711f2fed50b3e2fe0fe63.jpg)

![](images/bee111303b5907ff1f3c39211c6c7506790208566abdd8cf5972739b6048fe2f.jpg)

![](images/b284ef06e0e4948195552592be9892f1e8c6a62821bbb668d5617f072d61e3d1.jpg)

![](images/9154547410b9ca40e29687d9fd0815eff46766db95e61ec0d4f46a5126b3ccca.jpg)  
Figure 17: Learning–forgetting frontiers with EWC, $\lambda \in \{ 1 0 ^ { 3 } , 4 \cdot 1 0 ^ { 3 } , 1 0 ^ { 4 } , 4 \cdot 1 0 ^ { 4 } \}$ , loss-optimal 60M base models. Each row shows the budgets around that dataset’s crossover under EWC.

![](images/2d419ce504f5e0d76fc510a754c64ae1d93e9948bd0530a55325570e4303e70f.jpg)

![](images/2a6261b183a558c8e0155aa6402385a21d646b34bdc58d466d2e9cc0e2b8b3b6.jpg)

![](images/3f97e8b1ec779ca9c5389c82a1ae7fd431b9672d4f750235c42a76d24cecf07d.jpg)

![](images/161631bf08873998b33105776c34c33c58d7a8562cd60578f783ec1b5af7bbd5.jpg)  
Figure 18: Fine-tuning gap under fine-tuning regularization, with replay (solid) and EWC (dashed), loss-optimal 60M base models. Negative values mean the Muon frontier dominates.

## E Additional Results for Section 4 and Appendix B

This section collects details and additional results for the analysis of Section 4 and for the perturbation experiments of Section B: the margin-to-spread ratio across data budgets in the linear memory (Section E.1, for Section 4.2), perturbation robustness at every magnitude (Section E.2, for Section B.1), and the margin-to-spread ratio and robustness in the linear memory and in language models (Section E.3, for Sections B.2 and 4.3).

## E.1 Margin-to-spread ratio across data budgets

## E.1.1 One step

This is the regime Theorem 4.2 is about: the two solutions are the exact first step from $W _ { 0 } = 0 ,$ , so every quantity below can be compared against a closed form.

Setup Data. Keys and values are drawn as in Section 4.1 at $\begin{array} { r l r } { d } & { { } = } & { 5 1 2 } \end{array}$ , with $\begin{array} { r l r } { N } & { { } = } & { \gamma d } \end{array}$ and $\gamma \in$ 1 1 3   
{ <sub>4</sub> , <sub>2</sub> , <sub>4</sub> , 1, <sub>4</sub> , <sub>2</sub> ,   
quires $\bar { N } - \bar { 1 } \bar { \geq } d ,$ , so the points with $\gamma \leq 1$ lie outside the range in which Theorem 4.2 is stated and are drawn only for context.

One step. The gradient of the loss of Section 4.1 at $W _ { 0 } = 0$ is computed exactly, with equal item weights. GD takes it as $W ^ { \mathrm { G D } }$ ; Muon takes its polar factor $W ^ { \mathrm { M u o n } } = \mathrm { O r t h } ( W ^ { \mathrm { G D } } )$ , computed exactly by singular value decomposition rather than by the Newton–Schulz iteration of Section A.1.2. As a baseline we also record the first AdamW step, which with zero initialization and the usual ϵ is the entrywise sign $( W ^ { \mathrm { G D } } )$ ; Theorem 4.2 says nothing about it.

Observables. Every $h _ { i }$ and $\kappa _ { i }$ is evaluated in closed form for all i as in Definition 4.1, giving $\bar { h } , \bar { \kappa } , \rho$ and R.

Plotting. Solid lines are means over the five seeds and bands are ± one standard deviation; dashed lines are the closed forms behind Theorem 4.2, with $\Phi ( \gamma )$ replaced by its limit $\frac { 8 } { 3 \pi }$ . In the h<sup>¯</sup> and κ¯ panels each solution is normalized to $\| W \| _ { F } = 1$ , which is needed because the two steps have diferent scales while $\rho$ does not see them; the $\bar { h }$ and $\rho$ panels are divided by $\sqrt { d / \gamma }$ , the unit in which Theorem 4.2 is stated.

Result Figure 19 (top) shows $\mathcal { R }$ rising monotonically from 0.53 at $\begin{array} { r } { \gamma = \frac { 1 } { 4 } } \end{array}$ to 1.16 at $\gamma = 6 4$ and crossing 1 near $\gamma = 5$ , with the closed form tracking it once $\gamma \gtrsim 2$ . The remaining panels separate the two factors. Normalized, GD’s margin is constant in $\gamma ,$ which is the identity $\bar { h } ^ { \mathrm { G D } } = \| W ^ { \mathrm { G D } } \| _ { F } ^ { 2 } / ( \bar { N } - 1 )$ read at $\| W \| _ { F } = 1$ , while Muon’s climbs to about $\frac { 8 } { 3 \pi }$ ; both variances approach the same limit, GD’s from above and Muon’s from below. The limiting ratio $\frac { 3 \pi } { 8 }$ is therefore carried by the margin. Its approach to that limit comes from both factors: Muon’s margin contributes the $+ \frac { 1 } { 1 0 \gamma }$ correction, while the $- \frac { \daleth } { \gamma }$ part of the $\frac { 9 } { 1 0 \gamma }$ term comes from GD’s variance, the factor $\gamma + 2$ in $( \breve { \bar { \kappa } } ^ { \mathrm { G D } } ) ^ { 2 }$ (Section 4.2). The bottom row runs the same comparison against the AdamW surrogate: its $\mathcal { R }$ has the same shape but reaches only 0.93 at $\gamma = 6 4$ , so over the data budgets we ran the one-step reversal is a statement about GD, not about AdamW.

## E.1.2 Many steps

The theory covers one step. This experiment runs the same memory to convergence and asks whether the ordering it predicts survives training.

Setup Data. Keys and values are drawn as in Section 4.1, with item i queried with probability $q _ { i } \propto 1 / i ( \mathrm { Z i p f } )$ . We use $\bar { d } \in \{ 6 4 , 1 2 8 , 2 5 6 , 5 1 2 \}$ , γ log-spaced in [ <sup>1</sup> , 128], and three seeds per $( d , \gamma )$ shared by the two optimizers.

Training. From $W _ { 0 } ~ = ~ 0$ we take $T \ : = \ : 1 0 ^ { 4 }$ full-batch steps of $W _ { t + 1 } = W _ { t } + 0 . 0 2 G _ { t }$ for GD and $W _ { t + 1 } ~ =$ $W _ { t } + 0 . { \bar { 0 0 5 } } \operatorname { O r t h } ( G _ { t } )$ for Muon, where $\begin{array} { r } { G _ { t } = \sum _ { i } q _ { i } \big ( 1 - \hat { p } _ { W _ { t } } \big ( i \mid i \big ) \big ) u _ { i } v _ { i } ^ { \top } } \end{array}$ is the Hebbian part of the gradient of the loss of Section 4.1, at constant learning rate, without momentum or weight decay, and with the polar factor again taken by singular value decomposition.

Observables. $h _ { i }$ and $\kappa _ { i }$ are as in Definition 4.1, with equal item weights rather than the $q _ { i }$ used in training. The saturated value of an observable is the per-seed median over $t \geq 1 0 ^ { 3 } ; \breve { \mathcal { R } }$ is formed per seed and only then averaged Curves are means over seeds and bands are ± one standard deviation.

Perturbed robustness. We perturb the trained weights as in Section B.1.2, $W ^ { \prime } = W _ { T } + \sigma \| W _ { T } \| _ { F } \xi$ with $\| \xi \| _ { F } = 1$ over 16 perturbation directions shared by the two optimizers, and read of $\sigma _ { 5 0 } ,$ , the σ at which half of the items correct

Data budget (γ)

GD vs Muon

Data budget (γ)  
Data budget (γ)  
Data budget (γ)  
![](images/3b734b58915d1078252b5ccd35f503b492e8a64c123367d85b30b02811ffe73f.jpg)

![](images/fbd850ad38b2d5033682722c910675d3e72b55a9cb3a441b85bbdfdfcc6a9194.jpg)

![](images/8418213ccd8260f805b1545a210ae752db29318f1290425aa4baba1710935a64.jpg)  
Data budget (γ)

![](images/386f917c6db733047d123b8f1bacc0f5efc5e96908f85c415fabf527adf03725.jpg)

AdamW vs Muon

![](images/7ee72294438b30560eaed8488730332d8d7448a8823a55482766c11fa175b6bc.jpg)

![](images/3f13322bde4bc11224cb2f34ec1edba4b28b7efc75763fb13147f9a16df07a83.jpg)

![](images/c2d1383bbd1e864920cbffc9294eacce67450079c5f3920bd12242b815496d3b.jpg)

![](images/bdcca6396e765b649e0a774cb3840cb22aa460e6c33584231f9690d6f80774bc.jpg)  
GD / AdamW Muon Measured Theory  
Figure 19: One step at $d = 5 1 2 ,$ , GD against Muon (top) and the AdamW surrogate against Muon (bottom); solid is experiment, dashed the closed form of Theorem 4.2.

at $\sigma = 0$ are lost. For $N < d$ the Muon solution is rescaled to its component in the span of the keys, since outside that span it is an arbitrary orthogonal completion that carries norm but no information. Data budgets at which either optimizer is already past the capacity of the memory, meaning accuracy below 0.99 at $\sigma = 0 _ { : }$ are dropped from the $\sigma _ { 5 0 }$ panels: there the two optimizers recover diferent sets of items and the two $\sigma _ { 5 0 }$ are not comparable.

![](images/5019e74a52fccb5948aa6d42f32b965024bfc0e46e073dd16f2fe26d905c1967.jpg)

![](images/b66d5e190a47d7c67d00322a4bb23907caadbf4415ed19e48b5e29ad41c7882d.jpg)

![](images/7e638958bb00a1c9cb9b1974bee67cabcfccfc6bc734a0614d3c773c652c0fb5.jpg)

![](images/4369176bc6fc628ce7445725897c5d69d2a2d89b66677ae2b92322688bf600b3.jpg)

![](images/ecf878ed96402b1b48bf6da67eb7ea3faefe149d4d21fd748a88d2f0acc57e08.jpg)

![](images/cd89646cbfcccc1c22b9b89248a729298c4cc4f9bd38ada78285d43409d2569b.jpg)

![](images/738429513655a158b3373491f23961ea7d19cc70d7678e90a3e8da574721d4e6.jpg)

![](images/59a634954656e9030ceebdd4057e210d7520099132b36ad2f2c2245f92e180fc.jpg)  
GD Muon  
Figure 20: Top: saturated $\rho$ against the data budget, the two curves behind Figure 21. Bottom: perturbed robustness $\sigma _ { 5 0 }$ against the data budget, below capacity only. One column per $d .$

Result Training does not undo the one-step picture. Figure 21 shows R increasing with $\gamma$ at every $d ,$ over a range from about 0.4 to about 1.2, and the crossing $\bar { \mathcal { R } } = 1$ moving to larger data budgets as d grows: $\gamma ^ { * }$ ≈ 4.0, 4.4, 5.4 and 9.9 at d = 64, 128, 256 and 512. Figure 20 shows the two $\rho$ separately: both fall with the data budget, Muon’s from roughly twice GD’s at $\begin{array} { r } { \gamma = \frac { 1 } { 4 } } \end{array}$ to below it past $\gamma ^ { * }$ . The bottom row shows that the robustness ordering turns over as well, GD’s $\sigma _ { 5 0 }$ gaining on Muon’s at every $\cdot d .$ The two turnovers need not coincide, and they do not: the $\sigma _ { 5 0 }$ ordering flips at $\gamma \approx 0 . 5 , 1 . 3$ and 11 at $d = 1 2 8 ,$ , 256 and 512, and at $d = 6 4 \mathrm { G D }$ is already the more robust at the smallest data budget we ran.

![](images/e44f8546ac2cc7998fabef3cb57838d291ea501e5d96efd9aaae63fd604c6c83.jpg)

![](images/4b30f1a94aebc979124ca77c3099af4aa45091e775b757378b144a10b698a61e.jpg)  
Data budget (γ)

![](images/fe2bbac55afea2b00c00f21ff42510b656cf2a47dab960bab7d04af545116025.jpg)  
Data budget (γ)

![](images/bfe0f631fd981347f30016da05f313ce3586bc5001296704aa40742d7924ab86.jpg)  
Figure 21: Saturated R against the data budget, one panel per d.

## E.2 Perturbation robustness

## E.2.1 Setup

For every (model size, data budget) cell and each optimizer we take the checkpoint attaining the lowest held-out DCLM loss over the full pretraining learning-rate grid. A Gaussian perturbation is added independently to each parameter tensor of that checkpoint, scaled so that $\lVert \Delta W \rVert _ { F } \approx \delta \lVert \dot { W } \rVert _ { F } .$ and we report the increase in held-out DCLM loss over the checkpoint without perturbation, averaged over three perturbation draws. The grid spans three model sizes, 30M to 100M, token–parameter ratios from 5 to 2,560 TPR, and perturbation magnitudes δ from 0.005 to 0.28. Magnitudes below $\delta \approx 0 . { \bar { 0 2 } }$ are omitted from the figures because the increase in loss is then at the level of the seed-to-seed variation for both optimizers.

Table 2 and the robustness counts below use the same protocol on the loss-optimal checkpoints of all five model sizes, 30M to 600M: 36 (model size, data budget) cells and perturbation magnitudes δ from 0.002 to 0.56.

## E.2.2 Result

Figure 6 fixes the perturbation magnitude; Figure 22 reports the same diference of degradations at every magnitude we sweep. The left three panels fix δ and vary the model size, the right panel fixes the model size at 100M and varies δ. The diference is positive at every budget, model size and magnitude we tested, and grows with the budget at every δ. Magnitudes are not comparable across panels, since the degradation of both optimizers grows steeply with $\delta ;$ what the panels share is the shape of the curve.

Over the 36 cells, the perturbation magnitude $\delta ^ { * }$ at which the loss has risen by 0.05, 0.1 or 0.2 nats is larger for AdamW in every cell and at every threshold, by a median factor of 1.4, and Muon’s increase in loss exceeds AdamW’s in every cell at every δ from 0.005 to 0.2. The ordering does not depend on which checkpoint of a cell is used: giving each optimizer its most perturbation-robust checkpoint within 0.02 nats of the cell’s best base loss still leaves Muon degrading 20–400% more.

Loss under perturbation. Table 2 compares the loss under perturbation itself, rather than its increase, and places the result next to the fine-tuning crossover of Figure 2. At every model size, Muon’s loss under perturbation is the lower one at small budgets and AdamW’s at large ones, apart from the three marked entries. The crossover moves to smaller budgets as δ grows, so its value is not comparable to the fine-tuning crossover; its trend with model size is. From $\delta = 0 . { \overset { } { 1 } }$ on, the crossover budget does not increase with model size, as for every fine-tuning dataset; at $\delta = 0 . 0 5$ it is not monotone.

Table 2: Crossover budgets, in TPR (tokens per parameter). Left: the smallest budget at which AdamW’s loss under a perturbation of magnitude δ is below Muon’s (loss-optimal checkpoints, mean over three perturbation draws); > 20: Muon’s is lower up to the largest 600M budget, 20 TPR; ≤ 5: AdamW’s is lower already at the smallest budget; <sup>†</sup>: Muon’s is lower again at 160 TPR, by 0.003 nats. Right: the fine-tuning crossover of Figure 2.
<table><tr><td rowspan="2">Size</td><td colspan="4">Perturbation, δ</td><td colspan="4">Fine-tuning</td></tr><tr><td>0.05</td><td>0.1</td><td>0.13</td><td>0.2</td><td>Alpaca</td><td>GSM8K</td><td>MusicPile</td><td>StackMathQA</td></tr><tr><td>30M</td><td>160</td><td>40</td><td>40</td><td>20</td><td>160</td><td>80</td><td>640</td><td>320</td></tr><tr><td>60M</td><td>80</td><td>40</td><td>40</td><td>20</td><td>80</td><td>80</td><td>80</td><td>80</td></tr><tr><td>100M</td><td>160</td><td>40</td><td>20</td><td>10</td><td>80</td><td>40</td><td>80</td><td>40</td></tr><tr><td>300M</td><td> $8 0 ^ { \dagger }$ </td><td>40</td><td>20</td><td>10</td><td>40</td><td>10</td><td>40</td><td>20</td></tr><tr><td>600M</td><td>&gt; 20</td><td>20</td><td>10</td><td>≤5</td><td>20</td><td>5</td><td>10</td><td>10</td></tr></table>

![](images/2883904b30e7f8b381ffb8e14db33e8c8ba01e44f4aae5208fcdf53660d3406a.jpg)  
Figure 22: The perturbation degradation gap is positive at every perturbation magnitude. Muon minus AdamW increase in held-out DCLM loss. Left three panels: one $\delta$ each; right: 100M across $\delta .$

## E.3 Margin-to-spread ratio and robustness

## E.3.1 Setup

Linear model. Each cell is a data budget $\gamma = N / d$ and a model dimension $d ,$ run with three seeds. We train W from $W _ { 0 } = 0$ for $T = 1 0 ^ { 4 }$ full-batch steps on Zipf-distributed item frequencies, with GD at a learning rate of 0.02 and Muon at 0.005, and read $\rho$ of the trained $\dot { W }$ . The same W is then perturbed, $W ^ { \prime } = W + \sigma \| W \| _ { F } \xi$ with $\| \xi \| _ { F } = 1$ over 16 perturbation directions shared by the two optimizers, and $\sigma _ { 5 0 }$ is the perturbation level at which half of the items recovered at $\sigma = 0$ are lost, interpolated between grid points on a log scale.

Real language models. The analogue of a stored fact is a next-token prediction, and the analogue of the competitors of Definition 4.1 is the rest of the vocabulary. We evaluate the checkpoint on held-out DCLM text (128 sequences of 4096 tokens, never trained on). At a position with gold token t and logit vector z, the margins are

$$
m _ { j } \ = \ z _ { t } - z _ { j } , \qquad j \neq t ,\tag{7}
$$

over the whole vocabulary, and h and $\kappa ^ { 2 }$ are their mean and variance over those $| V | - 1$ competitors, exactly as in Definition 4.1. Two choices make this well defined. First, only the positions at which the gold token is already the argmax, min $_ { j \neq t } m _ { j } > 0 _ { \mathrm { { \ell } } }$ , are used: elsewhere the prediction is wrong before any perturbation is added and the margin is not a robustness margin. Second, the position-level values $h / \kappa$ are summarised by their median over positions rather than their mean, since the distribution over positions is heavy-tailed. The reduction over the vocabulary axis is carried out in double precision and chunked over positions, so the margin vector is never materialised. Both quantities are measured on the same checkpoints: for every (model size, token budget) cell, the checkpoint with the lowest held-out DCLM loss over the pretraining learning-rate grid, which gives $7 2$ checkpoints over five model sizes. For robustness we take $\delta ^ { * }$ , the relative weight-perturbation level at which the held-out loss rises by $\Delta L$ nats, interpolated on a log–log scale and averaged over three perturbation draws.

Learning-rate ablation. At two data budgets, each optimizer is run at its learning rate and at that rate scaled by $\textstyle { \frac { 1 } { 4 } }$ and by 4, all other settings unchanged, at $d = 1 2 8$ and $T = 1 0 ^ { 4 }$ over several seeds.

<table><tr><td>γ</td><td>optimizer</td><td>lr</td><td>accuracy</td><td> $\mathrm { f i n a l } \rho$ </td><td>saturated  $\rho$ </td><td>R</td></tr><tr><td rowspan="6">16</td><td>GD</td><td> $0 . 0 0 5 \ : ( \times \frac { 1 } { 4 } )$ </td><td> $0 . 2 0 5 7 \pm 0 . 0 0 3 1$ </td><td> $2 . 4 6 1 0 \pm 0 . 0 1 4 2$ </td><td> $2 . 4 5 5 0 \pm 0 . 0 1 3 6$ </td><td>1.1354</td></tr><tr><td>Muon</td><td> $0 . 0 0 1 2 5 \bigl ( \times \frac { 1 } { 4 } \bigr )$ </td><td> $0 . 1 5 8 0 \pm 0 . 0 0 3 1$ </td><td> $2 . 1 4 7 0 \pm 0 . 0 1 2 5$ </td><td> $2 . 1 6 2 4 \pm 0 . 0 1 3 3$ </td><td></td></tr><tr><td>GD</td><td>0.02</td><td> $0 . 1 9 7 8 \pm 0 . 0 0 1 4$ </td><td> $2 . 4 6 0 9 \pm 0 . 0 1 4 5$ </td><td> $2 . 4 6 1 0 \pm 0 . 0 1 4 4$ </td><td>1.1468</td></tr><tr><td>Muon</td><td>0.005</td><td> $0 . 1 5 6 6 \pm 0 . 0 0 2 4$ </td><td> $2 . 1 4 1 8 \pm 0 . 0 1 2 1$ </td><td> $2 . 1 4 6 1 \pm 0 . 0 1 2 3$ </td><td></td></tr><tr><td>GD</td><td>0.08 (×4)</td><td> $0 . 1 9 6 9 \pm 0 . 0 0 1 7$ </td><td> $2 . 4 6 0 7 \pm 0 . 0 1 4 6$ </td><td> $2 . 4 6 0 8 \pm 0 . 0 1 4 5$ </td><td>1.1445</td></tr><tr><td>Muon</td><td> $0 . 0 2 \ : ( \times 4 )$ </td><td> $0 . 1 5 7 7 \pm 0 . 0 0 4 2$ </td><td> $2 . 1 4 9 2 \pm 0 . 0 1 1 1$ </td><td> $2 . 1 5 0 2 \pm 0 . 0 1 1 2$ </td><td></td></tr><tr><td rowspan="6">64</td><td>GD</td><td> $0 . 0 0 5 \ : ( \times \frac { 1 } { 4 } )$ </td><td> $0 . 0 3 5 6 \pm 0 . 0 0 1 3$ </td><td> $0 . 9 4 1 3 \pm 0 . 0 1 2 9$ </td><td> $0 . 9 3 8 4 \pm 0 . 0 1 2 8$ </td><td>1.1992</td></tr><tr><td>Muon</td><td> $0 . 0 0 1 2 5 \ : ( \times \frac { 1 } { 4 } )$ </td><td> $0 . 0 2 7 2 \pm 0 . 0 0 1 3$ </td><td> $0 . 7 7 5 5 \pm 0 . 0 0 7 0$ </td><td> $0 . 7 8 2 5 \pm 0 . 0 0 7 3$ </td><td></td></tr><tr><td>GD</td><td>0.02</td><td> $0 . 0 3 4 4 \pm 0 . 0 0 1 2$ </td><td> $0 . 9 4 1 1 \pm 0 . 0 1 2 8$ </td><td> $0 . 9 4 1 2 \pm 0 . 0 1 2 9$ </td><td>1.2145</td></tr><tr><td>Muon</td><td>0.005</td><td> $0 . 0 2 6 9 \pm 0 . 0 0 1 5$ </td><td> $0 . 7 7 3 1 \pm 0 . 0 0 7 0$ </td><td> $0 . 7 7 5 0 \pm 0 . 0 0 7 0$ </td><td></td></tr><tr><td>GD</td><td>0.08 (×4)</td><td> $0 . 0 3 4 1 \pm 0 . 0 0 1 1$ </td><td> $0 . 9 4 1 0 \pm 0 . 0 1 2 8$ </td><td> $0 . 9 4 1 0 \pm 0 . 0 1 2 8$ </td><td>1.2138</td></tr><tr><td>Muon</td><td>0.02 (×4)</td><td> $0 . 0 2 7 1 \pm 0 . 0 0 1 4$ </td><td> $0 . 7 7 4 8 \pm 0 . 0 0 7 4$ </td><td> $0 . 7 7 5 3 \pm 0 . 0 0 7 3$ </td><td></td></tr></table>

Table 3. Learning-rate ablation in the linear model at $d = 1 2 8 , T = 1 0 ^ { 4 }$ , mean ± sd over seeds; “final” is ρ at the last step and “saturated” its median over $t \geq 1 0 ^ { 3 }$ . Each R pairs the two rows beside it. Accuracy and $\rho$ move by less than 1% over the 16× range, and so does R.

## E.3.2 Result

Within each optimizer $\sigma _ { 5 0 }$ and $\rho$ move together at every model dimension, over more than an order of magnitude in $\sigma _ { 5 0 }$ , and the association is monotone in $\rho$ rather than in d or γ separately (Figure 23). In the language models, the Spearman correlation between $\rho$ and $\delta ^ { * }$ over the 72 checkpoints is +0.83 at $\bar { \Delta } L = 0 . 0 5$ and 0.1 nats and +0.84 at 0.2 nats; within each optimizer it is +0.87 to +0.89 for AdamW and $+ 0 . 9 2 \ : \mathrm { t o  0 . 9 3 }$ for Muon (Figure 23). Across the 16× learning-rate range, accuracy moves by less than one percentage point and $\rho$ by less than 1%, far below the gap between the two optimizers, and the ratio R between them by at most 1.3% (Table 3); the comparison in Section 4.2 is therefore not an artefact of the learning rates.

(a)  
![](images/b681e2a480937a4842479d41ac22875ca878576ac3a1c949d694b27a2938a763.jpg)  
(b)

![](images/708f1c7d8ed1e4943ac375a4d02708fcd52d2d9470f6cf9c47368c9c39aa7cf3.jpg)

![](images/1f7b7e628c3258d39dbb7272e661ef01f0496c8571ce3f71a299d549efb645cc.jpg)

![](images/15a57445e19689b3da79aadb7dfc40fce2f1355e99db4ed7d9e3dc749198f0b5.jpg)

![](images/c526dddf30824c27e84ee4523bf0277cdfa2651ed1a769a7d99b857678323eff.jpg)  
MNR (ρ)

![](images/6fd3d189cb1520a09e47faa024bc78d65d19cef6819b720a49311a615df47c0e.jpg)  
MNR (ρ)

ΔL = 0.2 nats  
![](images/54c6ebde0dafc0a1f18dffcbfd6ba4d5e89632ba88b3c9015e68a495d3d4482e.jpg)  
GD (a) / AdamW (b) Muon  
MNR (ρ)  
Figure 23: Top: $\sigma _ { 5 0 }$ against $\rho$ in the linear model, one panel per model dimension, one point per γ cell, below capacity only. Bottom: real language models, $\delta ^ { * }$ against $\rho$ at every threshold; marker shape is the model size.

## F Proof of Theorem 4.2

This section proves the quantitative bound underlying the informal Theorem 4.2. For every fixed data budget $\gamma = N / d \geq 1 \bar { 6 }$ , we establish

$$
\Big | \mathcal { R } - \frac { 3 \pi } { 8 } \Big ( 1 - \frac { 9 } { 1 0 \gamma } \Big ) \Big | \le C _ { 1 } \gamma ^ { - 3 / 2 } + o _ { p } ( 1 )\tag{8}
$$

as $d \to \infty ,$ , where $C _ { 1 }$ is an absolute constant. The fixed-percentage increase is established in Lemma F.33; the other conclusions of Theorem 4.2 follow from the negative $1 / \dot { \gamma }$ term in $\mathcal { R } - 3 \pi / 8 .$ , as detailed below.

We first compute the one-step GD and Muon iterates and collect concentration estimates for the Gaussian data. We then evaluate the mean margin h<sup>¯</sup> and the margin spread κ¯ of Definition 4.1 separately for the two iterates, and finally take the ratio of the two margin-to-spread ratios. The GD estimates hold for every $\gamma > 1$ ; the Muon estimates, and hence the theorem, require $\gamma \geq 1 6$

Setup. We work in the setting of Section 4.1 with uniform query weights, $q _ { i } = 1 / N$ for every fact i; this is the setting of Theorem 4.2 and is assumed throughout. The keys and values $u _ { 1 } , \dotsc , u _ { N } , v _ { 1 } , \dotsc , v _ { N }$ are independent $\sqrt { ( 0 , I _ { d } ) }$ vectors, $N = \gamma d$ with the data budget $\gamma > 1$ held fixed, $N - 1 \geq d ,$ and $d \to \infty ; O _ { p }$ and $o _ { p }$ are as in Section 4.1.

Notation. We collect the data in $U = [ u _ { 1 } \cdot \cdot \cdot u _ { N } ] \in \mathbb { R } ^ { d \times N }$ and $V = [ v _ { 1 } \cdot \cdot \cdot v _ { N } ] \in \mathbb { R } ^ { d \times N }$ , and write $\mathbf { 1 } = ( 1 , \ldots , 1 ) ^ { \top }$ $e _ { i }$ for the i-th standard basis vector and $\langle X , Y \rangle \stackrel { \cdot } { = } \operatorname { t r } ( X ^ { \top } Y )$ ). The operator, Frobenius and nuclear norms $\left\| \cdot \right\| _ { \mathrm { o p } } , \left\| \cdot \right\| _ { F }$ and $\left\| \cdot \right\| .$ <sub>∗</sub> satisfy

$$
\begin{array} { r l } & { \| X \| _ { \infty } \leq \| X \| _ { F } \leq \| X \| _ { * } \leq \mathrm { r a n k } ( X ) \| X \| _ { \mathrm { o p } } , \qquad \| X \| _ { F } \leq \sqrt { \mathrm { r a n k } ( X ) } \| X \| _ { \mathrm { o p } } , } \\ & { | \langle X , Y \rangle | \leq \| X \| _ { * } \| Y \| _ { \mathrm { o p } } . } \end{array}\tag{9}
$$

For $X \in \mathbb { R } ^ { d \times d }$ with compact SVD $X = P \Sigma Q ^ { \top }$ we write $\operatorname { O r t h } ( X ) = P Q ^ { \top }$ for its polar factor, so that $\operatorname { O r t h } ( c X ) =$ $\operatorname { O r t h } ( X )$ for $c > 0$ and $\| \mathrm { O r t h } ( X ) \| _ { \mathrm { o p } } \leq 1$

Scores and observables. By Definition 4.1, for any $W _ { i }$

$$
\bar { h } = \operatorname* { m e a n } _ { i } u _ { i } ^ { \top } W v _ { i } - \operatorname* { m e a n } _ { i } \operatorname* { m e a n } _ { j \ne i } u _ { j } ^ { \top } W v _ { i } ,
$$

$$
\bar { \kappa } ^ { 2 } = \frac { 1 } { N ( N - 1 ) } \sum _ { i = 1 } ^ { N } \sum _ { j \neq i } \left( u _ { j } ^ { \top } W v _ { i } \right) ^ { 2 } - \operatorname* { m e a n } _ { i } \left( \operatorname* { m e a n } _ { j \neq i } u _ { j } ^ { \top } W v _ { i } \right) ^ { 2 } ,\tag{10}
$$

where the scores are written as $u _ { j } ^ { \top } W v _ { i }$ rather than $v _ { j } ^ { \top } W u _ { i }$ as in Section 4.1. The two conventions are exchanged by swapping the roles of $U$ and V: reading Section 4.1 with $( U , V )$ replaced by $( V , U )$ turns every score $v _ { j } ^ { \top } W u _ { i }$ into $u _ { i } ^ { \top } W v _ { i }$ and its one-step iterate into Equation (11), leaving the loss and every $h _ { i }$ and $\kappa _ { i }$ unchanged. Since $U$ and $V$ are i.i.d., the law of $\dot { } \rho$ is the same under either convention.

## F.1 The two one-step iterates

Lemma F.1 (One step). Up to a positive multiple, one step of GD and one step of Muon from $W _ { 0 } = 0$ give

$$
W ^ { \mathrm { G D } } = U V ^ { \top } - \frac { 1 } { N } ( U { \bf 1 } ) ( V { \bf 1 } ) ^ { \top } = U \Big ( I _ { N } - \frac { { \bf 1 1 } ^ { \top } } { N } \Big ) V ^ { \top } , \qquad W ^ { \mathrm { M u o n } } = \mathrm { O r t h } \big ( W ^ { \mathrm { G D } } \big ) .\tag{11}
$$

The positive multiple is immaterial for ρ.

Proof. Gradient. With $\partial _ { W } ( u _ { j } ^ { \top } W v _ { i } ) = u _ { j } v _ { i } ^ { \top }$ , the gradient of the loss of Section 4.1 with uniform weights $q _ { i } = 1 / N$ is

$$
- \nabla _ { W } \mathcal { L } ( W ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \Big ( u _ { i } - \sum _ { j = 1 } ^ { N } \hat { p } _ { W } ( j \mid i ) u _ { j } \Big ) v _ { i } ^ { \top } .\tag{12}
$$

At $W = 0$ all scores vanish, so $\hat { p } _ { 0 } ( j \mid i ) = 1 / N ;$ moreover $\begin{array} { r } { \sum _ { j } u _ { j } = U \mathbf { 1 } , \sum _ { i } u _ { i } v _ { i } ^ { \top } = U V ^ { \top } } \end{array}$ and $\begin{array} { r } { \sum _ { i } v _ { i } ^ { \top } = ( V \mathbf { 1 } ) ^ { \top } } \end{array}$

$$
- \nabla _ { W } \mathcal { L } ( 0 ) = \frac { 1 } { N } \Big ( U V ^ { \top } - \frac { 1 } { N } ( U { \bf 1 } ) ( V { \bf 1 } ) ^ { \top } \Big ) = \frac { 1 } { N } U \Big ( I _ { N } - \frac { { \bf 1 1 } ^ { \top } } { N } \Big ) V ^ { \top } .\tag{13}
$$

One step. With step size $\eta ~ > ~ 0$ , GD gives $- \eta \nabla _ { W } { \mathcal { L } } ( 0 ) ~ = ~ \textstyle { \frac { \eta } { N } } W ^ { \mathrm { G D } }$ and Muon gives $\eta \mathrm { O r t h } ( - \nabla _ { \cal W } \mathcal { L } ( 0 ) ) ~ =$ $\eta \operatorname { O r t h } ( W ^ { \mathrm { G D } } )$ , using $\operatorname { O r t h } ( c X ) = \operatorname { O r t h } ( X )$ for $c > 0$ . Normalization. Replacing W by cW with $c > 0$ multiplies every score u<sub>j</sub> ${ \overline { { W v _ { i } \ b y \ c } } } ,$ , hence $h _ { i } \mapsto c h _ { i }$ and $\kappa _ { i } \mapsto c \kappa _ { i } ,$ so $\bar { h } \mapsto c \bar { h } , \bar { \kappa } \mapsto$ cκ¯ and ρ is unchanged. □

Lemma F.2 (Full rank and orthogonality). Almost surely rank $W ^ { \mathrm { G D } } = d .$ Consequently $W ^ { \mathrm { M u o n } }$ is an orthogonal matrix: $\| W ^ { \mathrm { M u o n } } x \| = \| x \| , \| W ^ { \mathrm { M u o n } } \| _ { \mathrm { o p } } = 1$ and $\| W ^ { \mathrm { M u o \bar { n } } } \| _ { F } ^ { 2 } = d .$

Proof. det $W ^ { \mathrm { G D } }$ is a polynomial in the entries of $( U , V )$ . Taking $U = \left[ I _ { d } \ 0 \right]$ and $\begin{array} { r } { V = \big [ I _ { d } } & { { } - \sum _ { k < d } e _ { k } ~ 0 \big ] } \end{array}$ , which is possible because $N \geq d + 1$ , gives $V \mathbf { 1 } = 0$ and $W ^ { \mathrm { G D } } = U V ^ { \top } = I _ { d } .$ , so the polynomial is not identically zero; since $\bar { ( \cal U , \ / V ) }$ has a density, Pr[det $\check { W } ^ { \mathrm { G D } } = 0 ] = 0$ . At full rank the factors P, Q of the compact SVD are $d \times \dot { d }$ orthogonal matrices, so $W ^ { \mathrm { M u o n } } = \dot { P } Q ^ { \top }$ is orthogonal and $\| W ^ { \mathrm { M u o n } } \| _ { F } ^ { 2 } =$ rank $W ^ { \mathrm { { \scriptsize { G D } } } } = d .$ □

## F.2 Concentration estimates for the data

Lemma F.3 (Concentration for $U , V )$ . As $d \to \infty$ along the sequence above:

(i) mea $\mathsf { \iota } _ { k } \| u _ { k } \| ^ { 2 } = d \bigl ( 1 + O _ { p } ( ( N d ) ^ { - 1 / 2 } ) \bigr )$ , and likewise for V.

(ii) mean<sub>k</sub> $\left( \| u _ { k } \| \| v _ { k } \| - d \right) ^ { 2 } = O _ { p } ( d )$

(iii) $\| U \mathbf { 1 } \| ^ { 2 } / ( N d ) \to 1$ in probability; in particular $\| U \mathbf { 1 } \| ^ { 2 } = O _ { p } ( N d )$ . Likewise for V .

(iv) $\| U \| _ { \mathrm { o p } } / \sqrt { N }  1 + \gamma ^ { - 1 / 2 }$ almost surely, hence $\| U \| _ { \mathrm { o p } } ^ { 2 } \le 4 N ( 1 + o _ { p } ( 1 ) )$ ; and for every m $\geq 1$ there is $C _ { m }$ with $\mathbb { E } \Vert U \Vert _ { \mathrm { o p } } ^ { 2 m } \leq C _ { m } N ^ { m }$ . Likewise for V.

(v) $\| U U ^ { \top } - N I _ { d } \| _ { \mathrm { o p } } = O _ { p } ( \sqrt { N d } ) . \ I f \gamma \geq 1 6$ , then with probability tending to one both $\begin{array} { r } { S \ = \ \frac { 1 } { N } V V ^ { \top } } \end{array}$ and $\begin{array} { r } { S = \frac { 1 } { N } ( V V ^ { \top } - \frac { 1 } { N } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } ) } \end{array}$ satisfy $\| S - I _ { d } \| _ { \mathrm { o p } } \leq 3 \gamma ^ { - 1 / 2 }$ and $\begin{array} { r } { \lambda _ { \operatorname* { m i n } } ( S ) \ge \frac { 1 } { 8 } } \end{array}$

(vi) $\| V V ^ { \top } \| _ { F } ^ { 2 } = O _ { p } ( N ^ { 2 } d ) , a n d \mathbb { E } \| V V ^ { \top } \| _ { F } ^ { 2 m } \leq C _ { 2 m } ( N ^ { 2 } d ) ^ { m }$ . Likewise for $U U ^ { \top }$

(vii) $\lVert U V ^ { \top } \rVert _ { F } ^ { 2 } = N d ^ { 2 } ( 1 + O _ { p } ( d ^ { - 1 } ) )$ and $\begin{array} { r } { \| { U V ^ { \top } } \| _ { F } ^ { 2 } - d \sum _ { k } \| v _ { k } \| ^ { 2 } = O _ { p } ( \sqrt { N } d ^ { 3 / 2 } + N d ) . } \end{array}$

(viii) $U V ^ { \top } \triangleq ( U U ^ { \top } ) ^ { 1 / 2 } \ b { \Xi } \ast \boldsymbol { w } i t h \ b { \Xi } \in \mathbb { R } ^ { d \times d }$ having i.i.d. $\mathcal { N } ( 0 , 1 )$ entries independent of U; with probability tending to one, $\| U V ^ { \top } \| _ { \mathrm { o p } } \leq 5 \sqrt { N d } .$

Proof. $\begin{array} { r } { \mathrm { ( i ) } \sum _ { k } \| u _ { k } \| ^ { 2 } \sim \chi _ { N d } ^ { 2 } } \end{array}$ has mean Nd and variance $2 N d ,$ so Chebyshev gives $\begin{array} { r } { \sum _ { k } \| u _ { k } \| ^ { 2 } = N d + O _ { p } ( \sqrt { N d } ) } \end{array}$ divide by N.

(ii) For $x \geq 0$ one has $\begin{array} { r } { \sqrt { x } \ge 1 + \frac { x - 1 } { 2 } - \frac { { ( x - 1 ) } ^ { 2 } } { 2 } } \end{array}$ : the diference g of the two sides satisfies $g ( 0 ) = g ( 1 ) = g ^ { \prime } ( 1 ) = 0$ and $\begin{array} { r } { g ^ { \prime \prime } ( x ) = 1 - \frac { 1 } { 4 } x ^ { - 3 / 2 } } \end{array}$ changes sign once, at $x _ { 0 } = 4 ^ { - 2 / 3 }$ , so g is concave on $[ 0 , x _ { 0 } ]$ and convex on $[ x _ { 0 } , \infty )$ , and the two boundary values force $g \geq 0$ . With $u \sim \mathcal { N } ( 0 , I _ { d } ) , \mathbb { E } \| u \| ^ { 2 } / d = 1$ and var $( \| u \| ^ { 2 } / d ) = 2 / d \ t$ , so $\begin{array} { r } { \mathbb { E } \| u \| \geq \sqrt { d } ( 1 - \frac { 1 } { d } ) } \end{array}$ and $( \mathbb { E } \| u \| ) ^ { 2 } \geq d - 2$ . Hence for independent $u , v ,$

$$
\begin{array} { r } { \mathbb E \big ( \| u \| \| v \| - d \big ) ^ { 2 } = \mathbb E \| u \| ^ { 2 } \mathbb E \| v \| ^ { 2 } - 2 d \mathbb E \| u \| \mathbb E \| v \| + d ^ { 2 } \leq 2 d ^ { 2 } - 2 d ( d - 2 ) = 4 d , } \end{array}\tag{14}
$$

and Markov’s inequality gives the claim.

(iii) $\begin{array} { r } { U \mathbf { 1 } = \sum _ { k } u _ { k } \sim \mathcal { N } ( 0 , N I _ { d } ) , \mathbf { s o } \parallel U \mathbf { 1 } \parallel ^ { 2 } / N \sim \chi _ { d } ^ { 2 } } \end{array}$ has mean d and variance 2d; Chebyshev gives $\| U { \bf 1 } \| ^ { 2 } / ( N d ) =$ $1 + O _ { p } ( d ^ { - 1 / 2 } )$

(iv) $\| U \| _ { \mathrm { o p } } ^ { 2 } = \lambda _ { \mathrm { m a x } } ( U U ^ { \top } )$ , and by the Bai–Yin law $\lambda _ { \operatorname* { m a x } } ( U U ^ { \top } / N )  ( 1 + \gamma ^ { - 1 / 2 } ) ^ { 2 } \leq$ 4 almost surely, since $\gamma \geq 1$ For the moments, $U \mapsto \| U \| _ { \mathrm { o p } }$ is 1-Lipschitz for ∥·∥<sub>F</sub> and $\mathbb { E } \| U \| _ { \mathrm { o p } } \le \sqrt { N } + \sqrt { d } \le 2 \sqrt { N }$ by Gordon’s theorem, so Gaussian concentration gives $\operatorname* { P r } [ \| U \| _ { \mathrm { o p } } \geq 2 \sqrt { N } + t ] \leq e ^ { - t ^ { 2 } / 2 } ;$ integrating the tail bounds all moments.

(v) By Bai–Yin, $\lambda _ { \operatorname* { m a x } } ( U U ^ { \top } / N )  ( 1 + \gamma ^ { - 1 / 2 } ) ^ { 2 }$ and $\lambda _ { \operatorname* { m i n } } ( U U ^ { \top } / N )  ( 1 - \gamma ^ { - 1 / 2 } ) ^ { 2 }$ almost surely, so $\lVert U U ^ { \top } / N -$ $I _ { d } \| _ { \mathrm { o p } }  \operatorname* { m a x } \{ ( 1 + \gamma ^ { - 1 / 2 } ) ^ { 2 } - 1 , 1 - ( 1 - \gamma ^ { - 1 / 2 } ) ^ { 2 } \} = 2 \gamma ^ { - 1 / 2 } + \gamma ^ { - 1 }$ , which is at most $3 \gamma ^ { - 1 / 2 } < 4 \dot { \gamma } ^ { - 1 / 2 }$ ; multiplying by N gives the first claim. For $\gamma \geq 1 6$ we have $\gamma ^ { - 1 / 2 } \leq \frac { 1 } { 4 }$ , so in probability $\| V V ^ { \top } / N - I _ { d } \| _ { \mathrm { o p } } \to 2 \gamma ^ { - 1 / 2 } + \gamma ^ { - 1 } \leq$ $\frac { 9 } { 4 } \gamma ^ { - 1 / 2 } < 3 \gamma ^ { - 1 / 2 }$ and $\lambda _ { \operatorname* { m i n } } ( V V ^ { \top } / N )  ( 1 - \gamma ^ { - 1 / 2 } ) ^ { 2 } \bar { \geq } \frac { 9 } { 1 6 } > \frac { 1 } { 8 }$ , and the inequalities being strict they hold with probability tending to one. For the centred version, write $\begin{array} { r } { \frac { 1 } { N } \big ( V V ^ { \top } - \frac { 1 } { N } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } \big ) = \frac { V V ^ { \top } } { N } - \frac { 1 } { N ^ { 2 } } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } } \end{array}$ where $\begin{array} { r } { \frac { 1 } { N ^ { 2 } } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } \succeq 0 } \end{array}$ has operator norm $\textstyle \| V \mathbf { 1 } \| ^ { 2 } / N ^ { 2 } = \frac { \| V \mathbf { 1 } \| ^ { 2 } } { N d } \cdot \frac { d } { N } \to \gamma ^ { - 1 }$ in probability by (iii). By Weyl’s inequality,

$$
\lambda _ { \operatorname* { m i n } } \Big ( \frac { 1 } { N } \big ( V V ^ { \top } - \frac { 1 } { N } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } \big ) \Big ) \geq \lambda _ { \operatorname* { m i n } } \Big ( \frac { V V ^ { \top } } { N } \Big ) - \frac { \| V \mathbf { 1 } \| ^ { 2 } } { N ^ { 2 } } ,\tag{15}
$$

$$
\Big \| \frac { 1 } { N } \big ( V V ^ { \top } - \frac { 1 } { N } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } \big ) - I _ { d } \Big \| _ { \mathrm { o p } } \leq \Big \| \frac { V V ^ { \top } } { N } - I _ { d } \Big \| _ { \mathrm { o p } } + \frac { \| V \mathbf { 1 } \| ^ { 2 } } { N ^ { 2 } } ,
$$

and in probability the two right-hand sides converge to

$$
( 1 - \gamma ^ { - 1 / 2 } ) ^ { 2 } - \gamma ^ { - 1 } \ge \frac { 9 } { 1 6 } - \frac { 1 } { 1 6 } = \frac { 1 } { 2 } > \frac { 1 } { 8 } , \qquad 2 \gamma ^ { - 1 / 2 } + 2 \gamma ^ { - 1 } \le \frac { 5 } { 2 } \gamma ^ { - 1 / 2 } < 3 \gamma ^ { - 1 / 2 } ,\tag{16}
$$

using $\gamma ^ { - 1 / 2 } \leq \frac { 1 } { 4 } ;$ ; the inequalities being strict, they hold with probability tending to one.

(vi) $V V ^ { \top }$ has rank at most d, so $\| V V ^ { \top } \| _ { F } \leq \sqrt { d } \| V V ^ { \top } \| _ { \mathrm { o p } } = \sqrt { d } \| V \| _ { \mathrm { o p } } ^ { 2 }$ by Equation (9); now use (iv).

(vii) Expanding, $\begin{array} { r } { \| \boldsymbol { U } \boldsymbol { V } ^ { \top } \| _ { F } ^ { 2 } = \sum _ { \boldsymbol { k } , \boldsymbol { l } } ( \boldsymbol { u } _ { \boldsymbol { k } } ^ { \top } \boldsymbol { u } _ { \boldsymbol { l } } ) ( \boldsymbol { v } _ { \boldsymbol { k } } ^ { \top } \boldsymbol { v } _ { \boldsymbol { l } } ) = \sum _ { \boldsymbol { k } } \| \boldsymbol { u } _ { \boldsymbol { k } } \| ^ { 2 } \| \boldsymbol { v } _ { \boldsymbol { k } } \| ^ { 2 } + \sum _ { \boldsymbol { k } \neq \boldsymbol { l } } ( \boldsymbol { u } _ { \boldsymbol { k } } ^ { \top } \boldsymbol { u } _ { \boldsymbol { l } } ) ( \boldsymbol { v } _ { \boldsymbol { k } } ^ { \top } \boldsymbol { v } _ { \boldsymbol { l } } ) . } \end{array}$

Mean. $\mathbb { E } \| u _ { k } \| ^ { 2 } \| v _ { k } \| ^ { 2 } = d ^ { 2 }$ and, for $k \neq l , \mathbb { E } ( u _ { k } ^ { \top } u _ { l } ) ( v _ { k } ^ { \top } v _ { l } ) = 0 , \thinspace \thinspace \mathrm { s o } \mathbb { E } \Vert U V ^ { \top } \Vert _ { F } ^ { 2 } = N d ^ { 2 } .$

Variance. The variables $\| u _ { k } \| ^ { 2 } \| v _ { k } \| ^ { 2 } , k \in [ N ] ,$ , are independent with var $\cdot ( \| u _ { k } \| ^ { 2 } \| v _ { k } \| ^ { 2 } ) = ( d ^ { 2 } + 2 d ) ^ { 2 } - d ^ { 4 } \leq 8 d ^ { 3 }$ Next, $\begin{array} { r } { \sum _ { k \ne l } ( u _ { k } ^ { \top } u _ { l } ) ( v _ { k } ^ { \top } v _ { l } ) = 2 \ddot { \sum } _ { k < l } ( u _ { k } ^ { \top } u _ { l } ) ( v _ { k } ^ { \top } } \end{array}$ v<sub>l</sub>); for unordered pairs $\{ \boldsymbol { k } , \dot { \boldsymbol { l } } \} \stackrel { . . } { \neq } \ddot { \{ \boldsymbol { k } ^ { \prime } , \dot { \boldsymbol { l } } ^ { \prime } \} }$ one of the indices $k , l , k ^ { \prime } , l ^ { \prime }$ occurs exactly once, and integrating over the u with that index gives $\mathbb { E } \big [ ( u _ { k } ^ { \top } u _ { l } ) ( v _ { k } ^ { \top } v _ { l } ) ( u _ { k ^ { \prime } } ^ { \top } u _ { l ^ { \prime } } ) ( v _ { k ^ { \prime } } ^ { \top } v _ { l ^ { \prime } } ) \big ] = 0 ;$ since $\mathbb { E } ( u _ { k } ^ { \top } u _ { l } ) ^ { 2 } ( v _ { k } ^ { \top } v _ { l } ) ^ { 2 } = d ^ { 2 }$ , this gives var $\begin{array} { r } { \sum _ { k \neq l } ( u _ { k } ^ { \top } u _ { l } ) ( v _ { k } ^ { \top } v _ { l } ) = 4 \cdot \frac { N ( N - 1 ) } { 2 } \cdot d ^ { 2 } \leq 2 N ^ { 2 } d ^ { 2 } } \end{array}$ . For $k ^ { \prime } \in [ N ]$ and $k \neq l ,$ one of k, l difers from $k ^ { \prime }$ , and integrating over the u with that index gives E $\begin{array} { r } { \natural \big [ ( \| u _ { k ^ { \prime } } \| ^ { 2 } \| v _ { k ^ { \prime } } \| ^ { 2 } - d ^ { 2 } ) ( u _ { k } ^ { \top } u _ { l } ) ( v _ { k } ^ { \top } v _ { l } ) \big ] = 0 , } \end{array}$ so the two sums are uncorrelated. Therefore

$$
\operatorname { v a r } \lVert U V ^ { \top } \rVert _ { F } ^ { 2 } \le 8 N d ^ { 3 } + 2 N ^ { 2 } d ^ { 2 } , \qquad \frac { \big ( \operatorname { v a r } \lVert U V ^ { \top } \rVert _ { F } ^ { 2 } \big ) ^ { 1 / 2 } } { \mathbb { E } \lVert U V ^ { \top } \rVert _ { F } ^ { 2 } } \le \sqrt { \frac { 8 } { N d } + \frac { 2 } { d ^ { 2 } } } \le \frac { 4 } { d } ,\tag{17}
$$

and Chebyshev gives the first claim.

Centred version. $\begin{array} { r } { \| U V ^ { \top } \| _ { F } ^ { 2 } - d \sum _ { k } \| v _ { k } \| ^ { 2 } = \sum _ { k } ( \| u _ { k } \| ^ { 2 } - d ) \| v _ { k } \| ^ { 2 } + \sum _ { k \neq l } ( u _ { k } ^ { \top } u _ { l } ) ( v _ { k } ^ { \top } v _ { l } ) } \end{array}$ has mean zero. The variables $( \| u _ { k } \| ^ { 2 } - d ) \| v _ { k } \| ^ { 2 }$ are independent with variance $\mathbb { E } ( \| u \| ^ { 2 } - d ) ^ { 2 } \mathbb { E } \| v \| ^ { \dot { 4 } } = 2 d ( d ^ { 2 } + 2 d ) \le 6 d ^ { 3 } \mathrm { . }$ ; the second sum has variance at most $2 N ^ { 2 } d ^ { 2 }$ as above; and for $k ^ { \prime } \in [ N ]$ and $\dot { k } \neq l ,$ integrating over the u whose index among k, l difers from k<sup>′</sup> gives $\mathbb { E } \big [ ( \| u _ { k ^ { \prime } } \| ^ { 2 } - d ) \| v _ { k ^ { \prime } } \| ^ { 2 } ( u _ { k } ^ { \top } u _ { l } ) ( v _ { k } ^ { \top } v _ { l } ) \big ] = 0$ . Hence the variance is at most $6 N d ^ { 3 } + 2 N ^ { 2 } d ^ { 2 }$ , its square root is at most $\sqrt { 6 } \sqrt { N } d ^ { 3 / 2 } + \sqrt { 2 } N d ,$ and Chebyshev gives the second claim.

(viii) Conditionally on U, the columns of $U V ^ { \top }$ are $\mathrm { i . i . d . } \mathcal { N } ( 0 , U U ^ { \top } )$ , i.e. of the form $( U U ^ { \top } ) ^ { 1 / 2 } g$ with $g \sim \mathcal { N } ( 0 , I _ { d } ) ;$ stacking the d vectors g as the columns of Ξ gives the stated equality in distribution, and the conditional law of Ξ does not depend on U, so Ξ is independent of U. For the norm, $\begin{array} { r } { \bar { \| ( U U ^ { \top } ) ^ { 1 / 2 } \Xi \| _ { \mathrm { o p } } } \le \| U \| _ { \mathrm { o p } } \| \Xi \| _ { \mathrm { o p } } } \end{array}$ , and almost surely $\| U \| _ { \mathrm { o p } } / \sqrt { N }  1 + \gamma ^ { - 1 / 2 } \leq 2$ by (iv) while $\| \Xi \| _ { \mathrm { o p } } / \sqrt { d } \to 2$ by Bai–Yin with aspect ratio 1. □

Lemma F.4 (Norms of $W ^ { \mathrm { G D } } )$ .

$$
\begin{array} { r } { ( i ) \ \| W ^ { \mathrm { G D } } \| _ { F } ^ { 2 } = ( N - 1 ) d ^ { 2 } ( 1 + O _ { p } ( d ^ { - 1 } ) ) , a n d \left\| W ^ { \mathrm { G D } } \right\| _ { F } ^ { 2 } - d \mathrm { t r } \left( V V ^ { \top } - \frac { 1 } { N } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } \right) = O _ { p } ( \sqrt { N } d ^ { 3 / 2 } + N d ) . } \end{array}
$$

(ii) $\begin{array} { r } { \| W ^ { \mathrm { G D } } \| _ { \mathrm { o p } } \leq \| U \| _ { \mathrm { o p } } \| V \| _ { \mathrm { o p } } , } \end{array}$ , and with probability tending to one $\| W ^ { \mathrm { G D } } \| _ { \mathrm { o p } } \leq 7 \sqrt { N d } .$

(iii) With probability tending to one, $\begin{array} { r } { \frac 1 8 d \sqrt { N d } \leq \| W ^ { \mathrm { G D } } \| _ { * } \leq 7 d \sqrt { N d } . } \end{array}$

(iv) In probability, $\| W ^ { \mathrm { G D } } \| _ { * } / ( N d ) \to \Phi ( \gamma ) / \sqrt { \gamma }$ , where

$$
\Phi ( \gamma ) : = \frac { 1 } { \pi } \int _ { 0 } ^ { 1 } \bigg [ \sqrt { 1 - \frac { p } { \gamma } } \frac { 1 + p } { \sqrt { p } } + \frac { \sqrt { p } ( 1 - p ) } { \gamma \sqrt { 1 - p / \gamma } } \bigg ] \mathrm { d } p ,\tag{18}
$$

and $f o r \gamma \geq 1 6$

$$
\Phi ( \gamma ) = \frac { 8 } { 3 \pi } \Bigl ( 1 - \frac { 1 } { 1 0 \gamma } + r ( \gamma ) \Bigr ) , \qquad | r ( \gamma ) | \le \gamma ^ { - 2 } .\tag{19}
$$

Part (iv) is where the constant 3π $^ { \cdot / 8 }$ enters: Muon replaces $W ^ { \mathrm { G D } }$ by its polar factor, which turns the diagonal score into $\lVert W ^ { \mathrm { G D } } \rVert _ { * } ,$ and $8 / ( 3 \pi )$ is the limiting Marčenko–Pastur integral at $\gamma = \infty$

Proof. (i) By Equation (11) and $\| X - Y \| _ { F } ^ { 2 } = \| X \| _ { F } ^ { 2 } - 2 \langle X , Y \rangle + \| Y \| _ { F } ^ { 2 }$

$$
\| \boldsymbol { W } ^ { \mathrm { G D } } \| _ { F } ^ { 2 } = \| \boldsymbol { U } \boldsymbol { V } ^ { \top } \| _ { F } ^ { 2 } - \frac { 2 } { N } ( \boldsymbol { U } \mathbf { 1 } ) ^ { \top } \boldsymbol { U } \boldsymbol { V } ^ { \top } ( \boldsymbol { V } \mathbf { 1 } ) + \frac { 1 } { N ^ { 2 } } \| \boldsymbol { U } \mathbf { 1 } \| ^ { 2 } \| \boldsymbol { V } \mathbf { 1 } \| ^ { 2 } ,\tag{20}
$$

where the cross term is $O _ { p } ( N ^ { 2 } d ) / N = O _ { p } ( N d )$ and the last term is $O _ { p } ( d ^ { 2 } )$ by Theorem F.3(iii),(iv),(viii). Substituting $\| U V ^ { \top } \| _ { F } ^ { 2 } = N d ^ { 2 } ( 1 + \hat { O _ { p } } ( d ^ { - 1 } ) )$ from Theorem F.3(vii) and using $d ^ { 2 } \leq N d$ gives the first claim. For the second, $\begin{array} { r } { \mathrm { t r } ( V V ^ { \top } ) = \sum _ { k } \| v _ { k } \| ^ { 2 } } \end{array}$ and tr $\mathbf { \nabla } \cdot ( ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } ) = \| V \mathbf { 1 } \| ^ { 2 }$ , so the same expansion gives

$$
\begin{array} { r l r } {  { \| { \boldsymbol W } ^ { \mathrm { G D } } \| _ { F } ^ { 2 } - d \mathrm { t r } ( V { \boldsymbol V } ^ { \top } - \frac { 1 } { N } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } ) = ( \| { \boldsymbol U } { \boldsymbol V } ^ { \top } \| _ { F } ^ { 2 } - d \sum _ { k } \| v _ { k } \| ^ { 2 } ) + \frac { d \| { \boldsymbol V } \mathbf { 1 } \| ^ { 2 } } { N } } } \\ & { } & { - \ \frac { 2 } { N } ( { \boldsymbol U } \mathbf { 1 } ) ^ { \top } { \boldsymbol U } { \boldsymbol V } ^ { \top } ( { \boldsymbol V } \mathbf { 1 } ) + \frac { \| { \boldsymbol U } \mathbf { 1 } \| ^ { 2 } \| { \boldsymbol V } \mathbf { 1 } \| ^ { 2 } } { N ^ { 2 } } , } \end{array}\tag{21}
$$

whose four terms are $O _ { p } ( \sqrt { N } d ^ { 3 / 2 } + N d )$ by the second half of Theorem F.3(vii), $O _ { p } ( d ^ { 2 } ) , O _ { p } ( N d )$ and $O _ { p } ( d ^ { 2 } )$ respectively; since $d ^ { 2 } \leq N d ,$ , the claim follows.

(ii) $\begin{array} { r } { \| I _ { N } - \frac { { \bf 1 1 } ^ { \top } } { N } \| _ { \mathrm { o p } } = 1 } \end{array}$ gives the first inequality. For the second, the triangle inequality and the operator norm of a rank-one matrix give $\begin{array} { r } { \| \boldsymbol { W } ^ { \mathrm { G D } } \| _ { \mathrm { o p } } \leq \| \boldsymbol { U } \boldsymbol { V } ^ { \top } \| _ { \mathrm { o p } } + \| \boldsymbol { U } \mathbf { 1 } \| \| \boldsymbol { V } \mathbf { 1 } \| / N \leq 5 \sqrt { N d } + \frac 3 2 d \leq 7 \sqrt { N d } } \end{array}$ with probability tending to one, by Theorem F.3(iii),(viii) and $d \leq { \sqrt { N d } }$

(iii) For any X, $\begin{array} { r } { \| \boldsymbol { X } \| _ { F } ^ { 2 } = \sum _ { a } s _ { a } ^ { 2 } \leq s _ { 1 } \sum _ { a } s _ { a } = \| \boldsymbol { X } \| _ { \mathrm { o p } } \| \boldsymbol { X } \| , } \end{array}$ <sub>∗</sub>, and $\| X \| _ { * } \leq \mathrm { r a n k } ( X ) \| X \| _ { \mathrm { o p } } .$ Combining with (i) and (ii),

$$
\begin{array} { r l } & { \| W ^ { \mathrm { G D } } \| _ { * } \ge \displaystyle \frac { \| W ^ { \mathrm { G D } } \| _ { F } ^ { 2 } } { \| W ^ { \mathrm { G D } } \| _ { \mathrm { o p } } } \ge \displaystyle \frac { ( N - 1 ) d ^ { 2 } ( 1 - o _ { p } ( 1 ) ) } { 7 \sqrt { N d } } \ge \frac 1 8 d \sqrt { N d } , } \\ & { \| W ^ { \mathrm { G D } } \| _ { * } \le d \| W ^ { \mathrm { G D } } \| _ { \mathrm { o p } } \le 7 d \sqrt { N d } . } \end{array}\tag{22}
$$

(iv) Removing the centring. By the triangle inequality for the nuclear norm and the nuclear norm of a rank-one matrix, $| \| U V ^ { \top } \| _ { * } - \| W ^ { \mathrm { G D } } \| _ { * } | \le \| \dot { U } \mathbf { 1 } \| \| V \mathbf { 1 } \| \overset { \smile } { / } N = \dot { O } _ { p } ( d )$ , which is $o ( N d )$ ; so it sufices to prove the limit for $\| U V ^ { \top } \| _ { * }$

Limit. By Theorem F.3(viii), $\{ s _ { a } ( U V ^ { \top } ) ^ { 2 } / ( N d ) \} _ { a \leq d }$ has the same law as the eigenvalues of $\begin{array} { r } { \frac { 1 } { d } \big ( \frac { U U ^ { \top } } { N } \big ) ^ { 1 / 2 } { \Xi } { \Xi } ^ { \top } \big ( \frac { U U ^ { \top } } { N } \big ) ^ { 1 / 2 } } \end{array}$ with Ξ independent of U: conditionally on $U$ this is a sample covariance matrix with population covariance $\bar { \frac { 1 } { N } } U U ^ { \top }$ and aspect ratio 1, and the spectral distribution of ${ } _ { \frac { 1 } { N } U U } ^ { \top }$ converges almost surely to the Marčenko–Pastur law H with ratio $1 / \gamma$ and mean 1. By the Marčenko–Pastur–Silverstein theorem the empirical distribution of $\{ s _ { a } ( U V ^ { \top } ) ^ { 2 } / ( N d ) \}$ converges weakly in probability to a law ν whose Stieltjes transform m satisfies

$$
z = - \frac { 1 } { m } + \int \frac { t } { 1 + t m } \mathrm { d } H ( t ) .\tag{23}
$$

By Theorem F.3(viii) all $s _ { a } ^ { 2 } / ( N d ) \leq 2 5$ with probability tending to one, so both the empirical law and ν are supported in [0, 25]; taking the limit for the bounded continuous function $\lambda \mapsto \operatorname* { m i n } ( \sqrt { \lambda } , 5 )$ gives $\textstyle { \frac { 1 } { d } } \sum _ { a } s _ { a } / { \sqrt { N d } } \to { \bar { \int { \sqrt { \lambda } } } } \mathrm { d } \nu$ in probability, that is

$$
\frac { \| U V ^ { \top } \| _ { * } } { N d } = \sqrt { \frac { d } { N } } \cdot \frac { 1 } { d } \sum _ { a } \frac { s _ { a } } { \sqrt { N d } } \longrightarrow \frac { 1 } { \sqrt { \gamma } } \int \sqrt { \lambda } \mathrm { d } \nu ( \lambda ) .\tag{24}
$$

Closed form. Let $\begin{array} { r } { \psi ( x ) : = \int \frac { \lambda x } { 1 - \lambda x } \ d x } \end{array}$ dν. From $\begin{array} { r } { \frac { \lambda x } { 1 - \lambda x } = - 1 + \frac { 1 / x } { 1 / x - \lambda } } \end{array}$ we get $\begin{array} { r } { \psi ( x ) = - 1 - \frac { 1 } { x } m ( 1 / x ) } \end{array}$ , i.e. $m ( 1 / x ) =$ $- x ( 1 + \psi ( x ) )$ ). Taking $z = 1 / x$ in Equation (23) and multiplying by $x ( 1 + \psi ( x ) )$ ),

$$
\psi ( x ) = \int { \frac { x ( 1 + \psi ( x ) ) t } { 1 - x ( 1 + \psi ( x ) ) t } } \mathrm { d } H ( t ) .\tag{25}
$$

On the other hand the Stieltjes transform of the Marčenko–Pastur law with ratio $1 / \gamma$ and mean 1 satisfies $\begin{array} { c } { { \frac { 1 } { \gamma } z m _ { H } ( z ) ^ { 2 } - } } \end{array}$ $\begin{array} { r } { ( 1 - \frac { 1 } { \gamma } - z ) m _ { H } ( z ) + 1 = 0 } \end{array}$ , and $\begin{array} { r } { \frac { u t } { 1 - u t } = - 1 + \frac { 1 / u } { 1 / u - t } } \end{array}$ gives m $\begin{array} { r } { \mathbf { \Omega } _ { T } ( 1 / u ) = - u \big ( 1 + \int \frac { u t } { 1 - u t } \mathrm { d } H ( t ) \big ) } \end{array}$ ; substituting $z = 1 / u$ into the quadratic yields, for all $u ,$

$$
\frac { u } { \gamma } \Big ( 1 + \int \frac { u t } { 1 - u t } \mathrm { d } H ( t ) \Big ) ^ { 2 } + \Big ( 1 - \frac { 1 } { \gamma } \Big ) u \Big ( 1 + \int \frac { u t } { 1 - u t } \mathrm { d } H ( t ) \Big ) - \int \frac { u t } { 1 - u t } \mathrm { d } H ( t ) = 0 .\tag{26}
$$

Take $u = x ( 1 + \psi ( x ) )$ . By Equation (25) the integral then equals $\psi ( x )$ , so

$$
\frac { 1 } { \gamma } x ( 1 + \psi ) ^ { 3 } + \Big ( 1 - \frac { 1 } { \gamma } \Big ) x ( 1 + \psi ) ^ { 2 } - \psi = 0 , \qquad \mathrm { i . e . } \qquad x = \frac { \psi ( x ) } { ( 1 + \psi ( x ) ) ^ { 2 } ( 1 + \psi ( x ) / \gamma ) } .\tag{27}
$$

Using $\begin{array} { r } { \pi \sqrt { \lambda } = \int _ { 0 } ^ { \infty } \frac { \lambda } { \lambda + w } \frac { \mathrm { d } w } { \sqrt { w } } } \end{array}$ and Tonelli’s theorem, and then substituting $p = - \psi ( - 1 / w ) \in ( 0 , 1 )$ ,

$$
\begin{array} { r l r } {  { \int \sqrt { \lambda } \mathrm { d } \nu = \frac { 1 } { \pi } \int _ { 0 } ^ { \infty } \frac { - \psi ( - 1 / w ) } { \sqrt { w } } \mathrm { d } w , } } \\ & { } & { w = \frac { ( 1 - p ) ^ { 2 } ( 1 - p / \gamma ) } { p } , ~ - \frac { \mathrm { d } w } { \mathrm { d } p } = w \Big [ \frac { 2 } { 1 - p } + \frac { 1 / \gamma } { 1 - p / \gamma } + \frac { 1 } { p } \Big ] , } \\ & { } & { \int \sqrt { \lambda } \mathrm { d } \nu = \frac { 1 } { \pi } \int _ { 0 } ^ { 1 } \sqrt { 1 - \frac { p } { \gamma } } \Big [ 2 \sqrt { p } + \frac { \sqrt { p } ( 1 - p ) / \gamma } { 1 - p / \gamma } + \frac { 1 - p } { \sqrt { p } } \Big ] \mathrm { d } p ; ~ } \end{array}\tag{28}
$$

then $\textstyle 2 { \sqrt { p } } + { \frac { 1 - p } { \sqrt { p } } } = { \frac { 1 + p } { \sqrt { p } } }$ and $\begin{array} { r } { \sqrt { 1 - p / \gamma } \cdot \frac { \sqrt { p } ( 1 - p ) / \gamma } { 1 - p / \gamma } = \frac { \sqrt { p } ( 1 - p ) } { \gamma \sqrt { 1 - p / \gamma } } } \end{array}$ give Equation (18).

Expansion. For $\begin{array} { r } { 0 \leq s \leq \frac { 1 } { 1 6 } } \end{array}$ one has

$$
\left| { \sqrt { 1 - s } } - 1 + { \frac { s } { 2 } } \right| \leq { \frac { s ^ { 2 } } { 5 } } , \qquad 0 \leq ( 1 - s ) ^ { - 1 / 2 } - 1 \leq { \frac { 2 s } { 3 } } .\tag{29}
$$

For the first, $g ( s ) : = 1 - { \textstyle \frac { s } { 2 } } - { \textstyle \sqrt { 1 - s } }$ satisfies $g ( 0 ) = 0 , g ^ { \prime } ( s ) = \frac { 1 } { 2 } ( ( 1 - s ) ^ { - 1 / 2 } - 1 ) \geq 0$ and $\begin{array} { r } { g ^ { \prime \prime } ( s ) = \frac { 1 } { 4 } ( 1 - s ) ^ { - 3 / 2 } \leq } \end{array}$ $\textstyle { \frac { 1 } { 4 } } { \big ( } 1 - { \frac { 1 } { 1 6 } } { \big ) } ^ { - 3 / 2 } < { \frac { 2 } { 5 } }$ on $[ 0 , \frac { 1 } { 1 6 } ]$ , so $\begin{array} { r } { g ( s ) \le \frac { s ^ { 2 } } { 2 } } \end{array}$ max $\begin{array} { r } { g ^ { \prime \prime } < \frac { s ^ { 2 } } { 5 } } \end{array}$ . For the second, $\begin{array} { r } { f ( s ) : = ( 1 - s ) ^ { - 1 / 2 } - 1 - \frac { 2 s } { 3 } } \end{array}$ satisfies $f ( 0 ) = 0$ and $f ^ { \prime } ( s ) = { \textstyle \frac { 1 } { 2 } } ( 1 - s ) ^ { - 3 / 2 } - { \textstyle \frac { 2 } { 3 } } < 0$ for $s < 1 - ( \frac { 3 } { 4 } ) ^ { 2 / 3 } \approx 0 . 1 7 5$ , hence on $[ 0 , { \frac { 1 } { 1 6 } } ] ,$ so $f \leq 0$ there. Take $s = p / \gamma$ with $\gamma \geq 1 6$ . From

$$
\frac { 1 } { \pi } \int _ { 0 } ^ { 1 } \frac { 1 + p } { \sqrt { p } } \mathrm { d } p = \frac { 8 } { 3 \pi } , \qquad - \frac { 1 } { 2 \gamma \pi } \int _ { 0 } ^ { 1 } \sqrt { p } ( 1 + p ) \mathrm { d } p + \frac { 1 } { \gamma \pi } \int _ { 0 } ^ { 1 } \sqrt { p } ( 1 - p ) \mathrm { d } p = \frac { 8 } { 3 \pi } \Big ( - \frac { 1 } { 1 0 \gamma } \Big ) ,\tag{30}
$$

and Equation (18), the remainder is bounded by

$$
\begin{array} { l } { \displaystyle { \left| \Phi ( \gamma ) - \frac { 8 } { 3 \pi } \Big ( 1 - \frac { 1 } { 1 0 \gamma } \Big ) \right| \le \frac { 1 } { \pi \gamma ^ { 2 } } \Big [ \frac { 1 } { 5 } \int _ { 0 } ^ { 1 } p ^ { 3 / 2 } ( 1 + p ) \mathrm { d } p + \frac { 2 } { 3 } \int _ { 0 } ^ { 1 } p ^ { 3 / 2 } ( 1 - p ) \mathrm { d } p \Big ] } } \\ { \displaystyle { \qquad = \frac { 1 } { \pi \gamma ^ { 2 } } \Big [ \frac { 2 4 } { 1 7 5 } + \frac { 8 } { 1 0 5 } \Big ] < \frac { 8 } { 3 \pi } \gamma ^ { - 2 } , } } \end{array}\tag{31}
$$

which is Equation (19).

## F.3 The GD step

By Equation (10) with $W = W ^ { \mathrm { G D } }$

$$
\bar { h } ^ { \mathrm { G D } } = \underbrace { \mathrm { m e a n } u _ { i } ^ { \top } W ^ { \mathrm { G D } } v _ { i } } _ { \mathrm { d i a g o n a l } } - \underbrace { \mathrm { m e a n } \operatorname* { m e a n } _ { i \ne i } u _ { j } ^ { \top } W ^ { \mathrm { G D } } v _ { i } } _ { \mathrm { o f f - d i a g o n a l } } .\tag{32}
$$

Lemma F.5 (Diagonal-sum identity). The diagonal scores $o f W ^ { \mathrm { G D } }$ sum to

$$
\sum _ { i = 1 } ^ { N } u _ { i } ^ { \top } W ^ { \mathrm { G D } } v _ { i } = \| W ^ { \mathrm { G D } } \| _ { F } ^ { 2 } + \frac { 1 } { N } ( U { \bf 1 } ) ^ { \top } W ^ { \mathrm { G D } } ( V { \bf 1 } ) .
$$

Proof. A scalar equals its own trace and $\begin{array} { r } { \sum _ { i } u _ { i } v _ { i } ^ { \top } \ = \ U V ^ { \top } } \end{array}$ , so $\begin{array} { r l } { \sum _ { i } u _ { i } ^ { \top } W ^ { \mathrm { G D } } v _ { i } \ = \ \mathrm { t r } \left( ( W ^ { \mathrm { G D } } ) ^ { \top } \sum _ { i } u _ { i } v _ { i } ^ { \top } \right) \ = } \end{array}$ $\langle W ^ { \mathrm { G D } } , U V ^ { \top } \rangle$ ⟩. Rearranging Equation (11) gives $\begin{array} { r } { U V ^ { \top } = W ^ { \mathrm { G D } } + \frac { 1 } { N } ( U \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } } \end{array}$ , and bilinearity together with $\langle Z , a b ^ { \top } \rangle = a ^ { \top } Z b$ gives the claim. □

Lemma F.6 (Rank-one cross term). $\begin{array} { r } { \frac { 1 } { N ^ { 2 } } | ( U \mathbf { 1 } ) ^ { \top } W ^ { \mathrm { G D } } ( V \mathbf { 1 } ) | = O _ { p } ( d ) } \end{array}$

口

Proof. By Lemma F.4(ii) and Theorem F.3(iii),(iv),

$$
\frac { | ( U \mathbf { 1 } ) ^ { \top } W ^ { \mathrm { G D } } ( V \mathbf { 1 } ) | } { N ^ { 2 } } \leq \frac { \| U \mathbf { 1 } \| \| W ^ { \mathrm { G D } } \| _ { \mathrm { o p } } \| V \mathbf { 1 } \| } { N ^ { 2 } } = \frac { O _ { p } ( \sqrt { N d } ) \cdot O _ { p } ( N ) \cdot O _ { p } ( \sqrt { N d } ) } { N ^ { 2 } } = O _ { p } ( d ) .\tag{33}
$$

Lemma F.7. mean<sub>i</sub> $u _ { i } ^ { \top } W ^ { \mathrm { G D } } v _ { i } = d ^ { 2 } ( 1 + O _ { p } ( d ^ { - 1 } ) )$ ).

Proof. Divide Theorem F.5 by N. By Lemma $\begin{array} { r l } & { \mathrm { F . 4 ( i ) } , \| W ^ { \mathrm { G D } } \| _ { F } ^ { 2 } / N = \frac { N - 1 } { N } d ^ { 2 } ( 1 + O _ { p } ( d ^ { - 1 } ) ) = d ^ { 2 } ( 1 + O _ { p } ( d ^ { - 1 } ) ) } \end{array}$ , using $\textstyle { \frac { 1 } { N } } \leq { \frac { 1 } { d } } ;$ by Theorem F.6 the second term is $O _ { p } ( d ) = d ^ { 2 } \cdot O _ { p } ( d ^ { - 1 } )$ □

Lemma F.8 (Column-sum identity). For every $W \in \mathbb { R } ^ { d \times d }$ and every i, $\begin{array} { r } { ( N - 1 ) \mathrm { m e a n } _ { j \neq i } u _ { j } ^ { \top } W v _ { i } = ( U \mathbf { 1 } ) ^ { \top } W v _ { i } - } \end{array}$ $u _ { i } ^ { \top } W v _ { i }$

$$
\begin{array} { r } { P r o o f . \ ( N - 1 ) \operatorname* { m e a n } _ { j \neq i } u _ { j } ^ { \top } W v _ { i } = \sum _ { j } u _ { j } ^ { \top } W v _ { i } - u _ { i } ^ { \top } W v _ { i } \ \mathrm { a n d } \sum _ { j } u _ { j } = U \mathbf { 1 } . } \end{array}
$$

Lemma F.9 (Two sums of squares). $\begin{array} { r } { \sum _ { i } \left( ( U 1 ) ^ { \top } W ^ { \mathrm { G D } } v _ { i } \right) ^ { 2 } = O _ { p } ( N ^ { 4 } d ) \ a n d \sum _ { i } \left( u _ { i } ^ { \top } W ^ { \mathrm { G D } } v _ { i } \right) ^ { 2 } = O _ { p } ( N ^ { 4 } d ) . } \end{array}$

Proof. The numbers $( U \mathbf { 1 } ) ^ { \top } W ^ { \mathrm { G D } } v _ { i }$ are the entries of $V ^ { \top } ( W ^ { \mathrm { G D } } ) ^ { \top } U { \bf 1 }$ , so by Lemma F.4(ii) and Theorem F.3(iii),(iv),

$$
\begin{array} { r } { \left\| { V ^ { \top } ( W ^ { \mathrm { G D } } ) ^ { \top } U { \mathbf { 1 } } } \right\| ^ { 2 } \leq \| { V } \| _ { \mathrm { o p } } ^ { 2 } \| W ^ { \mathrm { G D } } \| _ { \mathrm { o p } } ^ { 2 } \| U { \mathbf { 1 } } \| ^ { 2 } = O _ { p } ( N \cdot N ^ { 2 } \cdot N d ) = O _ { p } ( N ^ { 4 } d ) . } \end{array}\tag{34}
$$

For the second sum, $u _ { i } ^ { \top } W ^ { \mathrm { G D } } v _ { i } ~ = ~ [ U ^ { \top } W ^ { \mathrm { G D } } V ] _ { i i }$ and $\begin{array} { r } { \sum _ { i } X _ { i i } ^ { 2 } \ \leq \ \| X \| _ { F } ^ { 2 } } \end{array}$ ; by Equation (11), $U ^ { \top } W ^ { \mathrm { G D } } V ~ =$ $( U ^ { \top } U ) ( I _ { N } - \frac { \mathbf { 1 1 } ^ { \top } } { N } ) ( V ^ { \top } V )$ , and since the nonzero eigenvalues of $V ^ { \top } V$ and $V V ^ { \top }$ coincide, $\| V ^ { \top } V \| _ { F } = \| V V ^ { \top } \| _ { F }$ so by Theorem $\mathrm { \ddot { F } } . 3 ( \mathrm { i v } ) , ( \mathrm { v i } )$

$$
\sum _ { i } \big ( { u } _ { i } ^ { \top } { W } ^ { \mathrm { G D } } { v } _ { i } \big ) ^ { 2 } \leq \| { U } ^ { \top } { W } ^ { \mathrm { G D } } { V } \| _ { F } ^ { 2 } \leq \| { U } \| _ { \mathrm { o p } } ^ { 4 } \| { V } { V } ^ { \top } \| _ { F } ^ { 2 } = O _ { p } ( { N } ^ { 2 } \cdot { N } ^ { 2 } d ) = O _ { p } ( { N } ^ { 4 } d ) .\tag{35}
$$

Lemma F.10. mean<sub>i</sub> $\begin{array} { r } { \big ( \mathrm { m e a n } _ { j \neq i } u _ { j } ^ { \top } W ^ { \mathrm { G D } } v _ { i } \big ) ^ { 2 } = O _ { p } ( \gamma ^ { 2 } d ^ { 2 } ) } \end{array}$ , hence |mean<sub>i</sub> mean $j \neq i \mathcal { u } _ { j } ^ { \top } W ^ { \mathrm { G D } } v _ { i } | = O _ { p } ( \gamma d )$

Proof. Square Theorem F.8 with $W = W ^ { \mathrm { G D } }$ , use $( x - y ) ^ { 2 } \leq 2 x ^ { 2 } + 2 y ^ { 2 }$ , sum over i and apply Theorem F.9:

$$
\sum _ { i } \left( ( N - 1 ) \operatorname* { m e a n } _ { j \neq i } u _ { j } ^ { \top } W ^ { \mathrm { G D } } v _ { i } \right) ^ { 2 } = O _ { p } ( N ^ { 4 } d ) \leq O _ { p } ( N ^ { 5 } ) ,\tag{36}
$$

using $d \leq N$ . Dividing by $\begin{array} { r } { N ( N - 1 ) ^ { 2 } \geq \frac { 1 } { 4 } N ^ { 3 } } \end{array}$ and substituting $N = \gamma d$ gives the first claim; the second follows from Cauchy–Schwarz, |mean $_ i x _ { i } | \leq ( \mathrm { m e a n } _ { i } \bar { x _ { i } ^ { 2 } } ) ^ { 1 / 2 }$ □

Lemma F.11 (h<sup>¯</sup> for GD). $\bar { h } ^ { \mathrm { G D } } = d ^ { 2 } \big ( 1 + O _ { p } ( \gamma / d ) \big )$

Proof. By Equation (32), Theorem F.7 and Theorem F.10, $\bar { h } ^ { \mathrm { G D } } = d ^ { 2 } ( 1 + O _ { p } ( d ^ { - 1 } ) ) - O _ { p } ( \gamma d ) = d ^ { 2 } ( 1 + O _ { p } ( d ^ { - 1 } ) +$ $O _ { p } ( \dot { \gamma } / d ) \dot { ) }$ , and $\gamma \geq 1$ absorbs the first error term into the second. □

We turn to $\bar { \kappa } ^ { \mathrm { G D } }$ . By Equation (10),

$$
( \bar { \kappa } ^ { \mathrm { G D } } ) ^ { 2 } = \underbrace { \frac { 1 } { N ( N - 1 ) } \sum _ { i = 1 } ^ { N } \sum _ { j \neq i } \left( u _ { j } ^ { \top } W ^ { \mathrm { G D } } v _ { i } \right) ^ { 2 } } _ { \mathrm { s e c o n d m o m e n t } } - \underbrace { \operatorname* { m e a n } \left( \operatorname* { m e a n } _ { j \neq i } u _ { j } ^ { \top } W ^ { \mathrm { G D } } v _ { i } \right) ^ { 2 } } _ { \mathrm { s q u a r e d m e a n } } .\tag{37}
$$

Lemma F.12 (Expectation of the second moment).

$$
\begin{array} { r l r } {  { \mathbb E \sum _ { i = 1 } ^ { N } \sum _ { j \neq i } ( u _ { j } ^ { \top } W ^ { \mathrm { G D } } v _ { i } ) ^ { 2 } = N ( N - 1 ) d ^ { 2 } ( N + 2 d + 2 ) } } \\ & { } & { \ - d ^ { 2 } \Big [ \Big ( 1 + \frac { 1 } { N } \Big ) ( N + d + 1 ) ^ { 2 } - 2 ( d + 2 ) ^ { 2 } - 2 ( N - 1 ) \Big ] } \\ & { } & { = N ( N - 1 ) d ^ { 2 } ( N + 2 d + 2 ) \big ( 1 + O ( N ^ { - 1 } ) \big ) . } \end{array}\tag{38}
$$

Proof. Write $u _ { i } ^ { \top } W ^ { \mathrm { G D } } v _ { i } ~ = ~ [ U ^ { \top } W ^ { \mathrm { G D } } V ] _ { i i } ;$ by Equation (11), $\begin{array} { r } { U ^ { \top } W ^ { \mathrm { G D } } V ~ = ~ ( U ^ { \top } U ) ( I _ { N } ~ - ~ \frac { \mathbf { 1 1 } ^ { \top } } { N } ) ( V ^ { \top } V ) } \end{array}$ , i.e. $\begin{array} { r } { u _ { j } ^ { \top } W ^ { \mathrm { G D } } v _ { i } = \bar { \sum } _ { k , l } ( u _ { j } ^ { \top } u _ { k } ) ( \delta _ { k l } - \frac { 1 } { N } ) ( v _ { l } ^ { \top } \bar { v _ { i } } ) } \end{array}$ , with $U$ and V independent. Two second moments, identical for u and v, are used:

$$
\mathbb { E } \big [ ( u _ { k } ^ { \top } u _ { l } ) ( u _ { l } ^ { \top } u _ { m } ) \big ] = \left\{ \begin{array} { l l } { d ^ { 2 } + 2 d , } & { k = l = m } \\ { d , } & { k = m \ne l } \\ { 0 , } & { \mathrm { o t h e r w i s e , } } \end{array} \right. \quad \mathbb { E } \big [ ( u _ { i } ^ { \top } u _ { k } ) ( u _ { i } ^ { \top } u _ { k ^ { \prime } } ) \big ] = \delta _ { k k ^ { \prime } } d \big [ 1 + ( d + 1 ) \delta _ { k i } \big ] .\tag{39}
$$

The first of these gives $\begin{array} { r } { \left[ \mathbb { E } ( U ^ { \top } U ) ^ { 2 } \right] _ { k m } = \sum _ { l } \mathbb { E } \left[ ( u _ { k } ^ { \top } u _ { l } ) ( u _ { l } ^ { \top } u _ { m } ) \right] = \delta _ { k m } \left[ ( d ^ { 2 } + 2 d ) + ( N - 1 ) d \right] = \delta _ { k m } d ( d + N + 1 ) = 1 - \frac { d } { d } \sum _ { l = 0 } ^ { m } \left[ ( d ^ { 2 } + 2 d ) ^ { 2 } \right] _ { l = 0 } ^ { m } \delta _ { l m } . } \end{array}$ i.e. $\mathbb { E } ( U ^ { \top } U ) ^ { 2 } = d ( d + N + 1 ) I _ { N }$ , and likewise for V . All pairs (i, j). Since $\| X \| _ { F } ^ { 2 } = \operatorname { t r } ( X ^ { \top } X )$ and by cyclicity,

$$
\sum _ { i , j } \big ( u _ { j } ^ { \top } W ^ { \mathrm { G D } } v _ { i } \big ) ^ { 2 } = \| U ^ { \top } W ^ { \mathrm { G D } } V \| _ { F } ^ { 2 } = \mathrm { t r } \left[ ( U ^ { \top } U ) ^ { 2 } \Big ( I _ { N } - \frac { { \bf 1 1 } ^ { \top } } { N } \Big ) ( V ^ { \top } V ) ^ { 2 } \Big ( I _ { N } - \frac { { \bf 1 1 } ^ { \top } } { N } \Big ) \right] ,\tag{40}
$$

and by independence and $\begin{array} { r } { \mathrm { t r } [ ( I _ { N } - \frac { { \mathbf { 1 1 } } ^ { \top } } { N } ) ^ { 2 } ] = N - 1 } \end{array}$

$$
\mathbb { E } \sum _ { i , j } \big ( u _ { j } ^ { \top } W ^ { \mathrm { G D } } v _ { i } \big ) ^ { 2 } = \big [ d ( d + N + 1 ) \big ] ^ { 2 } ( N - 1 ) .\tag{41}
$$

Diagonal $j = i .$ Using the independence of U and V , the second display above for both factors, and then $\begin{array} { r l r } { ( \delta _ { k l } - \frac { 1 } { N } ) ^ { 2 } = } & { { } } & { } \end{array}$ $\delta _ { k l } \big ( 1 - \textstyle \frac { 2 } { N } \big ) + \frac { 1 } { N ^ { 2 } }$ together with $\begin{array} { r } { \sum _ { k } [ 1 + ( d + 1 ) \delta _ { k i } ] = N + d + 1 } \end{array}$ and $\textstyle \sum _ { k } [ 1 + ( d + 1 ) \delta _ { k i } ] ^ { 2 } = N - 1 + ( d + 2 ) ^ { 2 }$

$$
\begin{array} { l } { \displaystyle \mathbb { E } \big ( u _ { i } ^ { \top } W ^ { \mathrm { G D } } v _ { i } \big ) ^ { 2 } = \sum _ { k , l , k ^ { \prime } , l ^ { \prime } } \mathbb { E } \big [ ( u _ { i } ^ { \top } u _ { k } ) ( u _ { i } ^ { \top } u _ { k ^ { \prime } } ) \big ] \mathbb { E } \big [ ( v _ { l } ^ { \top } v _ { i } ) ( v _ { l ^ { \prime } } ^ { \top } v _ { i } ) \big ] \Big ( \delta _ { k l } - \frac { 1 } { N } \Big ) \Big ( \delta _ { k ^ { \prime } l ^ { \prime } } - \frac { 1 } { N } \Big ) } \\ { \displaystyle \qquad = d ^ { 2 } \sum _ { k , l } \big [ 1 + ( d + 1 ) \delta _ { k i } \big ] \big [ 1 + ( d + 1 ) \delta _ { l i } \big ] \Big ( \delta _ { k l } - \frac { 1 } { N } \Big ) ^ { 2 } } \\ { \displaystyle \qquad = d ^ { 2 } \Big [ \Big ( 1 - \frac { 2 } { N } \Big ) \big ( N - 1 + ( d + 2 ) ^ { 2 } \big ) + \frac { ( N + d + 1 ) ^ { 2 } } { N ^ { 2 } } \Big ] . } \end{array}\tag{42}
$$

Subtracting. $\begin{array} { r } { \mathbb { E } \sum _ { i } \sum _ { j \neq i } = \mathbb { E } \sum _ { i , j } - N \mathbb { E } ( u _ { i } ^ { \top } W ^ { \mathrm { G D } } v _ { i } ) ^ { 2 } } \end{array}$ ; rearranging with $( N + d + 1 ) ^ { 2 } - ( d + 2 ) ^ { 2 } - ( N - 1 ) =$ $( N - 1 ) ( N + 2 d + \bar { 2 } )$ gives the first equality. Since $d \leq N$ , the discarded terms are bounded by $3 8 N ^ { 2 } d ^ { 2 } \ \leq$ $\begin{array} { r } { \frac { 3 8 } { N - 1 } \cdot \dot { N } ( N - 1 ) d ^ { 2 } ( N + 2 d + 2 ) } \end{array}$ , which is the stated relative error. □

Lemma F.13 (Variance of the second moment). There is an absolute constant C with var $\begin{array} { r } { \big [ \sum _ { i } \sum _ { j \neq i } ( u _ { j } ^ { \top } W ^ { \mathrm { G D } } v _ { i } ) ^ { 2 } \big ] \leq } \end{array}$ $C N ^ { 7 } d$

Proof. By Equation (11), $\begin{array} { r } { \sum _ { i } \sum _ { i \neq i } ( u _ { i } ^ { \top } W ^ { \mathrm { G D } } v _ { i } ) ^ { 2 } = \| U ^ { \top } W ^ { \mathrm { G D } } V \| _ { F } ^ { 2 } - \sum _ { i } [ U ^ { \top } W ^ { \mathrm { G D } } V ] _ { i i } ^ { 2 } . } \end{array}$ , a smooth function of the $2 N d$ independent standard Gaussian entries of $( U , V )$ . By the Gaussian Poincaré inequality the variance is at most $\mathbb { E } \lVert \nabla _ { U } \cdot \rVert _ { F } ^ { 2 ^ { \bullet } } + \mathbb { E } \lVert \nabla _ { V } \cdot \rVert _ { F } ^ { 2 }$ .

The gradient of $M \mapsto \textstyle \| M \| _ { F } ^ { 2 } - \sum _ { i } M _ { i i } ^ { 2 }$ is $2 M - 2 \mathrm { d i a g } ( M )$ , of Frobenius norm at most $2 \Vert M \Vert _ { F } ;$ the directional derivative of $\begin{array} { r } { U \mapsto ( U ^ { \top } U ) ( I _ { N } - \frac { \mathbf { 1 1 } ^ { \top } } { N } ) ( V ^ { \top } V ) } \end{array}$ ) along H has Frobenius norm at most $2 \| H \| _ { F } \| U \| _ { \mathrm { o p } } \| V \| _ { \mathrm { o p } } ^ { 2 }$ . With $\begin{array} { r } { \| \boldsymbol { U } ^ { \top } \boldsymbol { W } ^ { \mathrm { G D } } \boldsymbol { V } \| _ { F } \leq \| \boldsymbol { U } \| _ { \mathrm { o p } } ^ { 2 } \| \boldsymbol { V } \boldsymbol { V } ^ { \top } \| _ { F } , } \end{array}$ , the chain rule gives

$$
\Big \| \nabla _ { U } \sum _ { i } \sum _ { j \neq i } ( u _ { j } ^ { \top } W ^ { \mathrm { G D } } v _ { i } ) ^ { 2 } \Big \| _ { F } \leq 4 \| U \| _ { \mathrm { o p } } ^ { 3 } \| V \| _ { \mathrm { o p } } ^ { 2 } \| V V ^ { \top } \| _ { F } .\tag{43}
$$

Squaring, taking expectations and applying Hölder together with Theorem $\mathrm { F } . 3 ( \mathrm { i v } ) , ( \mathrm { v i } )$ gives $\begin{array} { r l } { \mathbb { E } \| \nabla _ { U } \cdot \| _ { F } ^ { 2 } } & { { } \leq } \end{array}$ $1 \hat { 6 } \big ( \mathbb { E } \| U \| _ { \mathrm { o p } } ^ { 1 8 } ) ^ { 1 / 3 } \big ( \mathbb { \bar { E } } \| V \| _ { \mathrm { o p } } ^ { 1 2 } ) ^ { 1 / 3 } \big ( \mathbb { E } \| V V ^ { \top } \| _ { F } ^ { \hat { 6 } } \big ) ^ { \hat { 1 } / 3 } \leq 1 6 \big ( C _ { 9 } C _ { 6 } ^ { 2 } \big ) ^ { \top 3 } N ^ { 7 } d .$ The expression is invariant under transposition combined with swapping $U$ and V, so the same bound holds for $\nabla _ { V } ;$ adding the two gives the claim with $C = 3 2 ( C _ { 9 } C _ { 6 } ^ { 2 } ) ^ { 1 / 3 }$ □

Lemma F.14 (Second moment). The second moment ofthe of-diagonal GD scores satisfies

$$
\frac { 1 } { N ( N - 1 ) } \sum _ { i = 1 } ^ { N } \sum _ { j \neq i } \left( u _ { j } ^ { \top } W ^ { \mathrm { G D } } v _ { i } \right) ^ { 2 } = d ^ { 3 } ( 2 + \gamma ) \left( 1 + O _ { p } ( \sqrt { \gamma } / d ) \right) .
$$

Proof. By Equation (38) and $N ( N - 1 ) d ^ { 2 } ( N + 2 d + 2 ) \geq N ^ { 2 } ( N - 1 ) d ^ { 2 } \geq \frac { 1 } { 2 } N ^ { 3 } d ^ { 2 }$ for $N \geq 2$ , the expectation is at least $\scriptstyle { \frac { 1 } { 4 } } N ^ { 3 } d ^ { 2 }$ for N large. Combined with Theorem F.13, the relative standard deviation is at most $\scriptstyle { \sqrt { C N ^ { 7 } d } } / ( { \frac { 1 } { 4 } } N ^ { 3 } d ^ { 2 } ) =$ $\bar { 4 } \sqrt { C } \sqrt { N / d ^ { 3 } } = 4 \sqrt { C } \sqrt { \gamma } / d ,$ , so Chebyshev gives

$$
\sum _ { i } \sum _ { j \neq i } \big ( u _ { j } ^ { \top } W ^ { \mathrm { G D } } v _ { i } \big ) ^ { 2 } = \Big ( \mathbb { E } \sum _ { i } \sum _ { j \neq i } \big ( u _ { j } ^ { \top } W ^ { \mathrm { G D } } v _ { i } \big ) ^ { 2 } \Big ) \big ( 1 + O _ { p } ( \sqrt { \gamma } / d ) \big ) .\tag{44}
$$

Since $N ^ { - 1 } = \gamma ^ { - 1 } d ^ { - 1 } \leq \sqrt { \gamma } / d ,$ , the relative error $O ( N ^ { - 1 } )$ of Theorem F.12 is absorbed; dividing by $N ( N - 1 )$ leaves $\begin{array} { r } { d ^ { 2 } ( N + 2 d + 2 ) = d ^ { 3 } ( \gamma + 2 + \frac { 2 } { d } ) = d ^ { 3 } ( 2 + \gamma ) ( 1 + \frac { 2 } { d ( 2 + \gamma ) } ) \mathrm { ~ a n d ~ } \frac { 2 } { d ( 2 + \gamma ) } \le \frac { 1 } { d } \le \sqrt { \gamma } / d . } \end{array}$ □

Lemma F.15 (κ¯ for GD). $( \bar { \kappa } ^ { \mathrm { G D } } ) ^ { 2 } = d ^ { 3 } ( 2 + \gamma ) \big ( 1 + O _ { p } ( \gamma / d ) \big )$

Proof. $\mathrm { B y }$ Theorem F.10 the squared mean in Equation (37) is $\begin{array} { r } { O _ { p } ( \gamma ^ { 2 } d ^ { 2 } ) = d ^ { 3 } ( 2 + \gamma ) \cdot O _ { p } \left( \frac { \gamma ^ { 2 } } { d ( 2 + \gamma ) } \right) = d ^ { 3 } ( 2 + \gamma ) \cdot O _ { p } ( \gamma / d ) } \end{array}$ Subtracting it from Theorem F.14 and using $\sqrt { \gamma } \leq \gamma \_$ gives the claim. □

Lemma F.16 $( \rho$ for GD). The margin-to-spread ratio of the GD iterate is

$$
\rho ^ { \mathrm { G D } } = \sqrt { \frac { d } { 2 + \gamma } } \big ( 1 + O _ { p } ( \gamma / d ) \big ) = \sqrt { \frac { d } { \gamma } } \Big ( 1 - \frac { 1 } { \gamma } + O ( \gamma ^ { - 2 } ) + o _ { p } ( 1 ) \Big ) .
$$

Both $\bar { h } ^ { \mathrm { G D } }$ and $\bar { \kappa } ^ { \mathrm { G D } }$ are a power of d times a function of $\gamma$ alone, so $\rho ^ { \mathrm { G D } }$ depends on d only through the prefactor ${ \sqrt { d } } .$

Proof. By Theorem F.11, Theorem F.15 and $\sqrt { 1 + O _ { p } ( \gamma / d ) } = 1 + O _ { p } ( \gamma / d )$

$$
\rho ^ { \mathrm { G D } } = \frac { \bar { h } ^ { \mathrm { G D } } } { \bar { \kappa } ^ { \mathrm { G D } } } = \frac { d ^ { 2 } \big ( 1 + O _ { p } ( \gamma / d ) \big ) } { d ^ { 3 / 2 } \sqrt { 2 + \gamma } \big ( 1 + O _ { p } ( \gamma / d ) \big ) } = \sqrt { \frac { d } { 2 + \gamma } } \big ( 1 + O _ { p } ( \gamma / d ) \big ) .\tag{45}
$$

With $\gamma$ fixed and $d \to \infty , O _ { p } ( \gamma / d ) = o _ { p } ( 1 )$ ; moreover $\sqrt { \gamma / ( 2 + \gamma ) } = ( 1 + 2 / \gamma ) ^ { - 1 / 2 } = 1 - \gamma ^ { - 1 } + O ( \gamma ^ { - 2 } )$

## F.4 The Muon step

Throughout this subsection $\gamma \geq 1 6$ . By Equation (10) with $W = W ^ { \mathrm { M u o n } }$ the same two decompositions hold as in Equation (32) and Equation (37), with $W ^ { \mathrm { G D } }$ replaced by $W ^ { \mathrm { M u o n } }$ . We use throughout that $W ^ { \mathrm { \tiny { M u o n } } }$ is orthogonal (Theorem F.2).

Lemma F.17 (Diagonal-sum identity). The diagonal scores $o f W ^ { \mathrm { M u o n } }$ sum to

$$
\sum _ { i = 1 } ^ { N } u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } = \| W ^ { \mathrm { G D } } \| _ { * } + \frac { 1 } { N } ( U \mathbf { 1 } ) ^ { \top } W ^ { \mathrm { M u o n } } ( V \mathbf { 1 } ) .
$$

Proof. As in Theorem $\begin{array} { r } { \mathrm { F . 5 , } \sum _ { i } u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } = \langle W ^ { \mathrm { M u o n } } , U V ^ { \top } \rangle = \langle W ^ { \mathrm { M u o n } } , W ^ { \mathrm { G D } } \rangle + \frac { 1 } { N } ( U \mathbf { 1 } ) ^ { \top } W ^ { \mathrm { M u o n } } ( V \mathbf { 1 } ) } \end{array}$ ). Writing the compact SVD $W ^ { \mathrm { G D } } = P \Sigma Q ^ { \top }$ , so that $W ^ { \mathrm { M u o n } } = P Q ^ { \top }$ , and using $P ^ { \top } P = Q ^ { \top } Q = I ,$

$$
\langle W ^ { \mathrm { M u o n } } , W ^ { \mathrm { G D } } \rangle = \operatorname { t r } \left( Q P ^ { \top } P \Sigma Q ^ { \top } \right) = \operatorname { t r } \left( \Sigma Q ^ { \top } Q \right) = \operatorname { t r } \Sigma = \| W ^ { \mathrm { G D } } \| _ { * } .\tag{46}
$$

Lemma F.18 (Rank-one cross term). $\begin{array} { r } { \frac { 1 } { N ^ { 2 } } | ( U \mathbf { 1 } ) ^ { \top } W ^ { \mathrm { M u o n } } ( V \mathbf { 1 } ) | = O _ { p } ( d / N ) } \end{array}$

Proof. $\begin{array} { r } { \mathrm { B y } \parallel W ^ { \mathrm { M u o n } } \parallel _ { \mathrm { o p } } \leq 1 } \end{array}$ and Theorem F.3(iii), $| ( U \mathbf { 1 } ) ^ { \top } W ^ { \mathrm { M u o n } } ( V \mathbf { 1 } ) | / N ^ { 2 } \leq \| U \mathbf { 1 } \| \| V \mathbf { 1 } \| / N ^ { 2 } = O _ { p } ( N d ) / N ^ { 2 } \leq$ $O _ { p } ( \bar { d } / N )$ □

Lemma F.19. mea $\begin{array} { r } { _ { 1 i } u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } = \frac { \| W ^ { \mathrm { G D } } \| _ { * } } { N } \big ( 1 + O _ { p } ( ( N d ) ^ { - 1 / 2 } ) \big ) } \end{array}$

Proof. Divide Theorem F.17 by N and use Theorem F.18. By Lemma F.4(iii), with probability tending to one $\lVert W ^ { \mathrm { G D } } \rVert _ { * } / N \ge d \sqrt { N d } / ( 8 N ) = d ^ { 3 / 2 } / ( 8 \sqrt { N } )$ , so the error term satisfies $\begin{array} { r } { O _ { p } ( d / N ) \ = \ \frac { \| W ^ { \mathrm { G D } } \| _ { * } } { N } O _ { p } \big ( \frac { d } { N } \frac { 8 \sqrt { N } } { d ^ { 3 / 2 } } \big ) \ = \ } \end{array}$ $\frac { \| W ^ { \mathrm { G D } } \| _ { * } } { N } O _ { p } ( ( N d ) ^ { - 1 / 2 } )$ □

Lemma F.20 (Two sums of squares). $\begin{array} { r } { \sum _ { i } \left( ( U 1 ) ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } \right) ^ { 2 } = O _ { p } ( N ^ { 2 } d ) \ a n d \sum _ { i } \left( u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } \right) ^ { 2 } = O _ { p } ( N d ^ { 2 } ) } \end{array}$

Proof. For the first, $W ^ { \mathrm { M u o n } }$ is orthogonal, so $\| ( W ^ { \mathrm { M u o n } } ) ^ { \top } U \mathbf { 1 } \| = \| U \mathbf { 1 } \|$ , and $\begin{array} { r } { \sum _ { i } v _ { i } v _ { i } ^ { \top } = V V ^ { \top } } \end{array}$ , whence by Theorem F.3(iii),(iv)

$$
\sum _ { i } \left( ( U \mathbf { 1 } ) ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } \right) ^ { 2 } = ( U \mathbf { 1 } ) ^ { \top } W ^ { \mathrm { M u o n } } V V ^ { \top } ( W ^ { \mathrm { M u o n } } ) ^ { \top } ( U \mathbf { 1 } ) \leq \| V \| _ { \mathrm { o p } } ^ { 2 } \| U \mathbf { 1 } \| ^ { 2 } = O _ { p } ( N \cdot N d ) .\tag{47}
$$

For the second, Cauchy–Schwarz and $\| W ^ { \mathrm { M u o n } } v _ { i } \| = \| v _ { i } \|$ give $( u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } ) ^ { 2 } \leq \| u _ { i } \| ^ { 2 } \| v _ { i } \| ^ { 2 }$ , and $x ^ { 2 } \leq 2 ( x - d ) ^ { 2 } +$ $2 d ^ { 2 }$ with Theorem F.3(ii) gives $\begin{array} { r } { \sum _ { i } \| u _ { i } \| ^ { 2 } \| v _ { i } \| ^ { 2 } \le 2 \sum _ { i } ( \| u _ { i } ^ { * } \| \| v _ { i } \| - d ) ^ { 2 } + 2 N d ^ { 2 } = N \cdot \dot { O } _ { p } ( \ddot { d } ) + 2 N d ^ { 2 } = \overline { { O } } _ { p } ( \dot { N } d ^ { 2 } ) . } \end{array}$ □

Lemma F.21. mea $\begin{array} { r } { \mathrm { n } _ { i } \left( \mathrm { m e a n } _ { j \neq i } u _ { j } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } \right) ^ { 2 } = O _ { p } ( d / N ) } \end{array}$ , hence |mean<sub>i</sub> mean $\mathsf { \iota } _ { { j \neq i } } \boldsymbol { u } _ { j } ^ { \top } \boldsymbol { W } ^ { \mathrm { M u o n } } \boldsymbol { v } _ { i } \vert = O _ { p } ( \sqrt { d / N } )$

Proof. Square Theorem F.8 with $W = W ^ { \mathrm { M u o n } }$ , use $( x - y ) ^ { 2 } \leq 2 x ^ { 2 } + 2 y ^ { 2 }$ , sum over i and apply Theorem F.20:

$$
\sum _ { i } \left( ( N - 1 ) \operatorname* { m e a n } _ { j \neq i } u _ { j } ^ { \top } W ^ { \operatorname { M u o n } } v _ { i } \right) ^ { 2 } \leq 2 O _ { p } ( N ^ { 2 } d ) + 2 O _ { p } ( N d ^ { 2 } ) = O _ { p } ( N ^ { 2 } d ) ,\tag{48}
$$

using $d \leq N$ . Dividing by $N ( N - 1 ) ^ { 2 } \geq \frac { 1 } { 4 } N ^ { 3 }$ gives the first claim, and Cauchy–Schwarz the second.

Lemma F.22 $( \bar { h }$ for Muon). The mean margin ofthe Muon iterate is

$$
\bar { h } ^ { \mathrm { M u o n } } = \frac { \| W ^ { \mathrm { G D } } \| _ { * } } { N } \big ( 1 + O _ { p } ( d ^ { - 1 } ) \big ) = \frac { d \Phi ( \gamma ) } { \sqrt { \gamma } } \big ( 1 + o _ { p } ( 1 ) \big ) .
$$

Proof. By Theorem F.19 and Theorem F.21, $\begin{array} { r } { \bar { h } ^ { \mathrm { M u o n } } = \frac { \| W ^ { \mathrm { G D } } \| _ { * } } { N } ( 1 + O _ { p } ( ( N d ) ^ { - 1 / 2 } ) ) - O _ { p } ( \sqrt { d / N } ) } \end{array}$ ). By Lemma F.4(iii), with probability tending to one $\textstyle \frac { \| W ^ { \mathrm { G D } } \| _ { * } } { N } \geq \frac { d ^ { 3 / 2 } } { 8 \sqrt { N } }$ , so

$$
O _ { p } \big ( \sqrt { d / N } \big ) = \frac { \| { \cal W } ^ { \mathrm { G D } } \| _ { * } } { N } \cdot O _ { p } \Big ( \sqrt { \frac { d } { N } } \cdot \frac { 8 \sqrt { N } } { d ^ { 3 / 2 } } \Big ) = \frac { \| { \cal W } ^ { \mathrm { G D } } \| _ { * } } { N } \cdot O _ { p } ( d ^ { - 1 } ) ,\tag{49}
$$

and $( N d ) ^ { - 1 / 2 } \leq d ^ { - 1 }$ , which gives the first equality. For the second, Lemma F.4(iv) gives $\begin{array} { r } { \frac { \| W ^ { \mathrm { G D } } \| _ { * } } { N d } = \frac { \Phi ( \gamma ) } { \sqrt { \gamma } } + o _ { p } ( 1 ) } \end{array}$ and $\frac { \Phi ( \gamma ) } { \sqrt { \gamma } } \geq \frac { 1 } { 8 \sqrt { \gamma } } > 0$ by Lemma $\mathrm { F . 4 ( i i i ) }$ , so the additive $o _ { p } ( 1 )$ may be written multiplicatively. □

We now turn to $\bar { \kappa } ^ { \mathrm { M u o n } }$ . The orthogonality of $W ^ { \mathrm { M u o n } }$ turns the second moment into three terms.

Lemma F.23 (Three-term identity). Almost surely

$$
\begin{array} { r l r } {  { \frac { 1 } { N ( N - 1 ) } \sum _ { i = 1 } ^ { N } \sum _ { j \neq i } \big ( u _ { j } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } \big ) ^ { 2 } = \frac { N } { N - 1 } \operatorname* { m e a n } _ { i } \lVert v _ { i } \rVert ^ { 2 } - \frac { \operatorname* { m e a n } _ { i } \big ( u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } \big ) ^ { 2 } } { N - 1 } } } \\ & { } & { \mathrm + \frac { \mathrm { t r } [ ( U U ^ { \top } - N I _ { d } ) W ^ { \mathrm { M u o n } } V V ^ { \top } ( W ^ { \mathrm { M u o n } } ) ^ { \top } ] } { N ( N - 1 ) } . } \end{array}\tag{50}
$$

Proof. Since $( U ^ { \top } x ) _ { j } = u _ { j } ^ { \top }$ x and $\| W ^ { \mathrm { M u o n } } v _ { i } \| = \| v _ { i } \|$

$$
\sum _ { j \neq i } \left( u _ { j } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } \right) ^ { 2 } = \| U ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } \| ^ { 2 } - \left( u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } \right) ^ { 2 } ,\tag{51}
$$

$$
\| \boldsymbol { U } ^ { \top } \boldsymbol { W } ^ { \mathrm { M u o n } } \boldsymbol { v } _ { i } \| ^ { 2 } = N \| \boldsymbol { v } _ { i } \| ^ { 2 } + \left( \boldsymbol { W } ^ { \mathrm { M u o n } } \boldsymbol { v } _ { i } \right) ^ { \top } \left( \boldsymbol { U } \boldsymbol { U } ^ { \top } - N \boldsymbol { I } _ { d } \right) \left( \boldsymbol { W } ^ { \mathrm { M u o n } } \boldsymbol { v } _ { i } \right) .
$$

Summing over i and using $x ^ { \top } Z x = \mathrm { t r } ( Z x x ^ { \top } )$ with $\begin{array} { r } { \sum _ { i } v _ { i } v _ { i } ^ { \top } = V V ^ { \top } } \end{array}$ turns the middle term into the stated trace; dividing by $N ( N - 1 )$ and using $\begin{array} { r } { \sum _ { i } \lVert v _ { i } \rVert ^ { 2 } = N } \end{array}$ mean $_ { i } \| v _ { i } \| ^ { 2 }$ gives the claim. □

Lemma F.24 (Norm term). $\begin{array} { r } { \frac { N } { N - 1 } \operatorname { m e a n } _ { i } \| v _ { i } \| ^ { 2 } = d \bigl ( 1 + O ( N ^ { - 1 } ) + O _ { p } ( ( N d ) ^ { - 1 / 2 } ) \bigr ) . } \end{array}$

Proof. Theorem F.3(i) gives mean $\begin{array} { r } { \sharp \| v _ { i } \| ^ { 2 } = d ( 1 + O _ { p } ( ( N d ) ^ { - 1 / 2 } ) ) , \mathrm { a n d } \frac { N } { N - 1 } = 1 + \frac { 1 } { N - 1 } = 1 + O ( N ^ { - 1 } ) . } \end{array}$

The trace term requires two auxiliary facts.

Lemma F.25 (Decomposition of $U U ^ { \top } )$ . Almost surely $V V ^ { \top } - { \frac { 1 } { N } } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top }$ is invertible and there is $\mathbf { \Psi } \in \mathbb { R } ^ { ( N - 1 - d ) \times d }$ with

$$
U U ^ { \top } - \frac { 1 } { N } ( U \mathbf { 1 } ) ( U \mathbf { 1 } ) ^ { \top } = W ^ { \mathrm { G D } } \Big ( V V ^ { \top } - \frac { 1 } { N } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } \Big ) ^ { - 1 } ( W ^ { \mathrm { G D } } ) ^ { \top } + \Gamma ^ { \top } \Gamma ,\tag{52}
$$

where, conditionally on $V _ { i }$ , the entries of Γ are $i . i . d . \mathcal { N } ( 0 , 1 )$ and independent $o f W ^ { \mathrm { G D } }$ . Consequently, for every symmetric $Y \in \bar { \mathbb { R } } ^ { d \times d }$ that is a function $o f ( V , W ^ { \mathrm { { \tilde { G } D } } } )$ ,

$$
\begin{array} { r l } & { \quad \mathbb { E } \big [ \operatorname { t r } ( \Gamma ^ { \top } \Gamma Y ) \mid V , W ^ { \mathrm { G D } } \big ] = ( N - 1 - d ) \operatorname { t r } Y , } \\ & { \mathrm { v a r } \big [ \operatorname { t r } ( \Gamma ^ { \top } \Gamma Y ) \mid V , W ^ { \mathrm { G D } } \big ] = 2 ( N - 1 - d ) \| Y \| _ { F } ^ { 2 } . } \end{array}\tag{53}
$$

Proof. Invertibility. $\begin{array} { r } { V V ^ { \top } - \frac { 1 } { N } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } = V ( I _ { N } - \frac { \mathbf { 1 1 } ^ { \top } } { N } ) V ^ { \top } \succeq 0 } \end{array}$ and its determinant is a polynomial in $V$ which is nonzero at $\begin{array} { r } { V = \big [ I _ { d } } & { { } - \sum _ { k \le d } e _ { k } \ 0 \big ] } \end{array}$ , where it equals det $\begin{array} { r } { ( I _ { d } + ( \sum _ { k } e _ { k } ) ( \sum _ { k } e _ { k } ) ^ { \top } ) = 1 + d \ne 0 ; } \end{array}$ so it is nonzero almost surely.

Basis. If $\mathbf { \bar { \Psi } } c _ { 0 } \mathbf { 1 } ^ { \top } + c ^ { \top } V = 0$ for some $c _ { 0 } ~ \in ~ \mathbb { R }$ and $c \in \mathbb { R } ^ { d }$ , then multiplying on the right by $\begin{array} { r } { ( I _ { N } - \frac { { \bf 1 1 } ^ { \top } } { N } ) V ^ { \top } c , } \end{array}$ and using $\begin{array} { r } { \mathbf { 1 } ^ { \top } ( I _ { N } - \frac { \mathbf { 1 1 } ^ { \top } } { N } ) = 0 } \end{array}$ , gives $c ^ { \top } ( V V ^ { \top } - { \textstyle \frac { 1 } { N } } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } ) c = 0 , \ s \circ \ c = 0$ and then $c _ { 0 } ~ = ~ 0$ . Hence $\{ x \in \mathbb { R } ^ { N } : \mathbf { 1 } ^ { \top } x = \dot { 0 _ { , } } V x = 0 \}$ has dimension $N - 1 - d$ almost surely; stack an orthonormal basis of it as the columns of $R \in \mathbb { R } ^ { N \times ( N - 1 - d ) }$ , so that $R ^ { \top } R = I , \mathbf { 1 } ^ { \top } R = 0$ and $V R = 0 ;$ and set $\Gamma : = R ^ { \top } U ^ { \top }$

Projection identity. Because $\mathbb { R } ^ { N } = \operatorname { s p a n } { \mathbf { 1 } } \oplus \operatorname { c o l } \left( ( I _ { N } - \frac { \mathbf { 1 1 } ^ { \top } } { N } ) V ^ { \top } \right)$ ⊕col R with the three subspaces pairwise orthogonal and of dimensions $1 + d + ( N - 1 - d ) = N$

$$
I _ { N } - \frac { \mathbf { 1 1 } ^ { \top } } { N } = \Big ( I _ { N } - \frac { \mathbf { 1 1 } ^ { \top } } { N } \Big ) V ^ { \top } \Big ( V V ^ { \top } - \frac { 1 } { N } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } \Big ) ^ { - 1 } V \Big ( I _ { N } - \frac { \mathbf { 1 1 } ^ { \top } } { N } \Big ) + R R ^ { \top } .\tag{54}
$$

Both sides are symmetric, so it sufices to check them on the three subspaces. Write $\begin{array} { r } { P : = I _ { N } - \frac { { \mathbf { 1 1 } } ^ { \top } } { N } } \end{array}$ and $G : =$ $V V ^ { \top } - { \textstyle \frac { 1 } { N } } ( V \mathbf { 1 } ) ( \mathbf { \dot { \Gamma } } V \mathbf { 1 } ) ^ { \top }$ , and use $P ^ { 2 } = P , V P V ^ { \top } = G$ and $P R = R - { \textstyle \frac { 1 } { N } } { \bf 1 } ( { \bf 1 } ^ { \top } R ) = R$ . For $x = { \bf 1 } ; P x = 0 , V P x = 0$ and $R ^ { \top } x = 0$ , so both sides vanish. For $x = P V ^ { \top } y \colon P x = x ,$ while $\begin{array} { r } { \dot { V } P x = V P V ^ { \top } y = G y , } \end{array}$ , so the first term of the right-hand side is $P V ^ { \top } G ^ { - 1 } G y = x ,$ and $R ^ { \top } x = ( V R ) ^ { \top } y = 0$ , so the second vanishes. For $x = R z \colon P x = R z = x ,$ while $V P x = V R z = 0$ kills the first term, and $R R ^ { \top } R z = R z = x$ . In all three cases both sides agree. Multiplying Equation (54) on the left by U and on the right by $U ^ { \top }$ , and using $\begin{array} { r } { U ( I _ { N } - \frac { \mathbf { 1 1 } ^ { \top } } { N } ) U ^ { \top } = U U ^ { \top } - \frac { 1 } { N } ( U \mathbf { 1 } ) ( U \mathbf { 1 } ) ^ { \top } } \end{array}$ and $\begin{array} { r } { U ( I _ { N } - \frac { \mathbf { 1 1 } ^ { \top } } { N } ) V ^ { \top } = W ^ { \mathrm { G D } } } \end{array}$ , gives the decomposition with $U R R ^ { \top } U ^ { \top } = \Gamma ^ { \top } \Gamma$

Distribution. Conditionally on V, the rows of U are i.i.d. $\mathcal { N } ( 0 , I _ { N } ) ;$ the a-th column ofΓ is $R ^ { \top } U ^ { \top } e _ { a } \sim { \mathcal N } ( 0 , R ^ { \top } R ) =$ $\mathscr { N } ( 0 , I _ { N - 1 - d } )$ with independent columns, so the entries of Γ are i.i.d. standard Gaussian. The a-th row of $W ^ { \mathrm { G D } }$ is $\begin{array} { r } { \left( V ( I _ { N } - \frac { \mathbf { 1 1 } ^ { \top } } { N } ) U ^ { \top } e _ { a } \right) ^ { \top } } \end{array}$ and Cov $\begin{array} { r } { \left( V ( I _ { N } - \frac { \mathbf { 1 1 } ^ { \top } } { N } ) U ^ { \top } e _ { a } , R ^ { \top } U ^ { \top } e _ { a } \right) = V ( I _ { N } - \frac { \mathbf { 1 1 } ^ { \top } } { N } ) R = 0 , } \end{array}$ ; being jointly Gaussian conditionally on $V , \Gamma$ and $W ^ { \mathrm { G D } }$ are conditionally independent.

Moments. The rows of Γ are i.i.d. $\mathcal { N } ( 0 , I _ { d } )$ and $\begin{array} { r } { \mathrm { t r } ( \Gamma ^ { \top } \Gamma Y ) = \sum _ { m } ( \Gamma ^ { \top } e _ { m } ) ^ { \top } Y ( \Gamma ^ { \top } e _ { m } ) } \end{array}$ ; for $g \sim \mathcal { N } ( 0 , I _ { d } )$ and fixed symmetric $Y , \mathbb { E } [ g ^ { \top } Y g ] = \operatorname { t r } Y$ and $\operatorname { v a r } [ g ^ { \top } Y g ] = 2 \| Y \| _ { F } ^ { 2 }$ ; summing over the $N - 1 - d$ rows gives the claim.

Lemma F.26 (Trace of a commutator). Almost surely, with $\begin{array} { r } { S : = \frac { 1 } { N } \big ( V V ^ { \top } - \frac { 1 } { N } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } \big ) } \end{array}$

$$
\Big | \operatorname { t r } \Big [ W ^ { \mathrm { G D } } ( N S ) ^ { - 1 } ( W ^ { \mathrm { G D } } ) ^ { \top } W ^ { \mathrm { M u o n } } ( N S ) ( W ^ { \mathrm { M u o n } } ) ^ { \top } \Big ] - \| W ^ { \mathrm { G D } } \| _ { F } ^ { 2 } \Big | \leq \frac { 2 \| S - I _ { d } \| _ { \mathrm { o p } } ^ { 2 } } { \lambda _ { \operatorname* { m i n } } ( S ) } \| W ^ { \mathrm { G D } } \| _ { F } ^ { 2 } .\tag{55}
$$

Proof. A matrix inequality. For symmetric $S \succ 0$ and symmetric $T ,$

$$
\big | \operatorname { t r } [ S ^ { - 1 } T S T ] - \| T \| _ { F } ^ { 2 } \big | \leq \frac { 2 \| S - I _ { d } \| _ { \mathrm { o p } } ^ { 2 } \| T \| _ { F } ^ { 2 } } { \lambda _ { \operatorname* { m i n } } ( S ) } .\tag{56}
$$

Indeed, cyclicity gives t $\cdot [ S ^ { - 1 } S T T ] = \| T \| _ { F } ^ { 2 }$ and $\mathrm { t r } [ ( T S - S T ) T ] = 0$ , so with $S ^ { - 1 } = I _ { d } - S ^ { - 1 } ( S - I _ { d } )$

$$
\mathrm { t r } [ S ^ { - 1 } T S T ] - \| T \| _ { F } ^ { 2 } = \mathrm { t r } \left[ S ^ { - 1 } ( T S - S T ) T \right] = - \mathrm { t r } \left[ S ^ { - 1 } ( S - I _ { d } ) ( T S - S T ) T \right] ,\tag{57}
$$

and $| \mathrm { t r } ( A B C ) | \le \| A \| _ { \mathrm { o p } } \| B \| _ { F } \| C \| _ { F } , \| S ^ { - 1 } ( S - I _ { d } ) \| _ { \mathrm { o p } } \le \| S - I _ { d } \| _ { \mathrm { o p } } / \lambda _ { \mathrm { m i n } } ( S )$ and $\Vert T S - S T \Vert _ { F } \leq 2 \Vert S -$ $I _ { d } \| _ { \mathrm { o p } } \| T \| _ { F }$ give Equation (56).

Polar decomposition. With $W ^ { \mathrm { G D } } ~ = ~ P \Sigma Q ^ { \top }$ and $W ^ { \mathrm { M u o n } } ~ = ~ P Q ^ { \top }$ we have $( W ^ { \mathrm { G D } } ) ^ { \top } W ^ { \mathrm { M u o n } } \ = \ Q \Sigma Q ^ { \top } \ =$ $( W ^ { \mathrm { M u o n } } ) ^ { \top } W ^ { \dot { \mathrm { G D } } }$ and $\| Q \Sigma Q ^ { \top } \| _ { F } = \| \Sigma \| _ { F } = \| W ^ { \mathrm { G D } } \| _ { F }$

Substitution. Moving the leading $W ^ { \mathrm { G D } }$ to the end by cyclicity,

$$
\mathrm { t r } \left[ W ^ { \mathrm { G D } } ( N S ) ^ { - 1 } ( W ^ { \mathrm { G D } } ) ^ { \top } W ^ { \mathrm { M u o n } } ( N S ) ( W ^ { \mathrm { M u o n } } ) ^ { \top } \right] = \mathrm { t r } \left[ ( N S ) ^ { - 1 } \big ( Q \Sigma Q ^ { \top } \big ) ( N S ) \big ( Q \Sigma Q ^ { \top } \big ) \right] ,\tag{58}
$$

and applying Equation (56) with $T = Q \Sigma Q ^ { \top }$ (the factors N cancel) gives the claim.

Lemma F.27 (Trace term). $I f \gamma \geq 1 6 ,$ , then with probability tending to one

$$
\frac { 1 } { N ( N - 1 ) d } \Big | \mathrm { t r } \left[ \left( U U ^ { \top } - N I _ { d } \right) W ^ { \mathrm { M u o n } } V V ^ { \top } ( W ^ { \mathrm { M u o n } } ) ^ { \top } \right] \Big | \leq 1 4 5 \gamma ^ { - 2 } .\tag{59}
$$

The assumption $\gamma \geq 1 6$ is tailored to this estimate: through Theorem F.3(v) it controls how far $\scriptstyle { \frac { 1 } { N } } V V ^ { \top }$ is from the identity, so that the fluctuation of $U U ^ { \top }$ around $N I _ { d }$ , seen through the Muon iterate, contributes only $O ( \gamma ^ { - 2 } )$ to the second moment.

Proof. Write $B : = W ^ { \mathrm { M u o n } } \bigl ( V V ^ { \top } - \textstyle \frac { 1 } { N } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } \bigr ) ( W ^ { \mathrm { M u o n } } ) ^ { \top }$ . By Theorem F.2 orthogonal conjugation preserves $\| \cdot \| _ { F } , \| \cdot \| _ { \mathrm { o p } }$ and the trace; moreover $\begin{array} { r } { V V ^ { \top } - \frac { 1 } { N } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } = V ( I _ { N } - \frac { \mathbf { 1 1 } ^ { \top } } { N } ) V ^ { \top } } \end{array}$ satisfies $0 \preceq V V ^ { \top } - \textstyle { \frac { 1 } { N } } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } \preceq$ $V V ^ { \top }$ . Hence, by Theorem F.3(iii),(iv),(vi)

$$
\begin{array} { l } { \displaystyle \| B \| _ { F } \leq \| V V ^ { \top } \| _ { F } + \frac { \| V \mathbf { 1 } \| ^ { 2 } } { N } = O _ { p } ( N \sqrt { d } ) , } \\ { \displaystyle \| B \| _ { \mathrm { o p } } \leq \| V \| _ { \mathrm { o p } } ^ { 2 } = O _ { p } ( N ) , } \\ { \displaystyle ~ \mathrm { t r } B = \sum _ { k } \| v _ { k } \| ^ { 2 } - \frac { \| V \mathbf { 1 } \| ^ { 2 } } { N } = O _ { p } ( N d ) . } \end{array}\tag{60}
$$

Rank-one replacement. $\begin{array} { r } { V V ^ { \top } - ( V V ^ { \top } - \frac { 1 } { N } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } ) = \frac { 1 } { N } ( V \mathbf { 1 } ) ( V \mathbf { 1 } ) ^ { \top } } \end{array}$ , so by Theorem F.3(iii),(v),

$$
\begin{array} { r l } & { \Big | \mathrm { t r } \left[ ( U U ^ { \top } - N I _ { d } ) W ^ { \mathrm { M u o n } } V V ^ { \top } ( W ^ { \mathrm { M u o n } } ) ^ { \top } \right] - \mathrm { t r } \left[ ( U U ^ { \top } - N I _ { d } ) B \right] \Big | } \\ & { \qquad \leq \displaystyle \frac { 1 } { N } \| U U ^ { \top } - N I _ { d } \| _ { \mathrm { o p } } \| V \mathbf { 1 } \| ^ { 2 } = N ( N - 1 ) d \cdot O _ { p } \big ( \sqrt { d } N ^ { - 3 / 2 } \big ) . } \end{array}\tag{61}
$$

Splitting. By Theorem F.25 and $d + ( N - 1 - d ) + 1 = N$

$$
U U ^ { \top } - N I _ { d } = \left[ W ^ { \mathrm { G D } } \big ( N S \big ) ^ { - 1 } ( W ^ { \mathrm { G D } } ) ^ { \top } - d I _ { d } \right] + \Big [ \Gamma ^ { \top } \Gamma - ( N - 1 - d ) I _ { d } \Big ] + \Big [ \frac { 1 } { N } ( U \mathbf { 1 } ) ( U \mathbf { 1 } ) ^ { \top } - I _ { d } \Big ] ,\tag{62}
$$

and we bound the trace of each bracket against B separately.

First bracket. Since tr B appears, write

$$
\begin{array} { r l } & { \mathrm { t r } \left[ ( W ^ { \mathrm { G D } } ( N S ) ^ { - 1 } ( W ^ { \mathrm { G D } } ) ^ { \top } - d I _ { d } ) B \right] = \left( \mathrm { t r } [ W ^ { \mathrm { G D } } ( N S ) ^ { - 1 } ( W ^ { \mathrm { G D } } ) ^ { \top } B ] - \| W ^ { \mathrm { G D } } \| _ { F } ^ { 2 } \right) } \\ & { \qquad + \left( \| W ^ { \mathrm { G D } } \| _ { F } ^ { 2 } - d \mathrm { t r } B \right) . } \end{array}\tag{63}
$$

By Theorem F.26, Theorem $\mathrm { F } . 3 ( \mathrm { v } ) ( \gamma \geq 1 6 )$ and Lemma $\mathrm { F . 4 ( i ) }$ , with probability tending to one the first diference is at most

$$
\frac { 2 ( 3 \gamma ^ { - 1 / 2 } ) ^ { 2 } } { 1 / 8 } \| { \cal W } ^ { \mathrm { G D } } \| _ { F } ^ { 2 } = 1 4 4 \gamma ^ { - 1 } ( N - 1 ) d ^ { 2 } \big ( 1 + O _ { p } ( d ^ { - 1 } ) \big ) = N ( N - 1 ) d \cdot 1 4 4 \gamma ^ { - 2 } \big ( 1 + O _ { p } ( d ^ { - 1 } ) \big ) ,\tag{64}
$$

while the second is $O _ { p } ( \sqrt { N } d ^ { 3 / 2 } + N d ) = N ( N - 1 ) d \cdot O _ { p } ( \sqrt { d } N ^ { - 3 / 2 } + N ^ { - 1 } )$ by Lemma F.4(i).

Second bracket. B is symmetric and a function of $( V , W ^ { \mathrm { G D } } )$ , so Theorem F.25 gives $\mathbb { E } \big [ \mathrm { t r } [ ( \Gamma ^ { \top } \Gamma - ( N - 1 - d ) I _ { d } ) B ] \ |$ $V , W ^ { \mathrm { G D } } ] = 0$ and, by Equation (60), conditional variance $2 ( N - 1 - d ) \| B \| _ { F } ^ { 2 } = \dot { O } ( N ) \cdot O _ { p } ( N ^ { 2 } d ) = O _ { p } ( N ^ { 3 } d )$ The conditional Chebyshev inequality then gives $\mathrm { t r } [ ( \Gamma ^ { \top } \Gamma - ( N - 1 - d ) I _ { d } ) B ] = O _ { p } ( N ^ { 3 / 2 } d ^ { 1 / 2 } ) = N ( N - 1 ) d$ $O _ { p } ( ( N d ) ^ { - 1 / 2 } )$

Third bracket. By $\mathrm { t r } ( a a ^ { \top } Y ) = a ^ { \top } Y a \leq \| a \| ^ { 2 } \| Y \| _ { \mathrm { o p } }$ , Equation (60) and Theorem F.3(iii),

$$
\begin{array} { l } { \Big | \operatorname { t r } \Big [ \big ( \frac { 1 } { N } ( U \mathbf { 1 } ) ( U \mathbf { 1 } ) ^ { \top } - I _ { d } \big ) B \Big ] \Big | \leq \frac { \| U \mathbf { 1 } \| ^ { 2 } } { N } \| B \| _ { \mathrm { o p } } + | \operatorname { t r } B | } \\ { = O _ { p } ( d ) \cdot O _ { p } ( N ) + O _ { p } ( N d ) = N ( N - 1 ) d \cdot O _ { p } ( N ^ { - 1 } ) . } \end{array}\tag{65}
$$

Total. Adding the four contributions and dividing by $N ( N - 1 ) d ,$ with probability tending to one the left-hand side is at most $1 4 4 \breve { \gamma } ^ { - 2 } ( 1 + O _ { p } ( d ^ { - 1 } ) ) + o _ { p } ( 1 ) = 1 4 4 \breve { \gamma } ^ { - 2 } + o _ { p } ( 1 )$ , which is at most $1 4 \dot { 5 } \gamma ^ { - 2 }$ since $\gamma$ is fixed. □

The remaining term of Theorem F.23, the diagonal one, needs the following sandwich.

Lemma F.28 (Nuclear-norm sandwich). For all $a , b \in \mathbb { R } ^ { d }$

$$
a ^ { \top } \operatorname { O r t h } \big ( W ^ { \mathrm { G D } } - a b ^ { \top } \big ) b \leq \| W ^ { \mathrm { G D } } \| _ { * } - \| W ^ { \mathrm { G D } } - a b ^ { \top } \| _ { * } \leq a ^ { \top } W ^ { \mathrm { M u o n } } b .\tag{66}
$$

Proof. For all $X , Y , \langle X , \mathrm { O r t h } ( Y ) \rangle \leq \| X \| _ { * } \| \mathrm { O r t h } ( Y ) \| _ { \mathrm { o p } } \leq \| X \|$ <sub>∗</sub> by Equation (9), and writing the compact SVD $Y = P \Sigma Q ^ { \intercal }$ gives $\langle Y , \operatorname { O r t h } ( Y ) \rangle = \mathrm { t r } ( Q \Sigma P ^ { \top } P Q ^ { \top } ) = \| Y \| _ { * } ; \mathrm { a l s o } \langle a b ^ { \top } , Z \rangle = a ^ { \top } Z b$ . Hence

$$
\begin{array} { r l } & { a ^ { \top } \operatorname { O r t h } \big ( W ^ { \mathrm { G D } } - a b ^ { \top } \big ) b = \big \langle W ^ { \mathrm { G D } } , \mathrm { O r t h } ( W ^ { \mathrm { G D } } - a b ^ { \top } ) \big \rangle - \big \langle W ^ { \mathrm { G D } } - a b ^ { \top } , \mathrm { O r t h } ( W ^ { \mathrm { G D } } - a b ^ { \top } ) \big \rangle } \\ & { \qquad \leq \| W ^ { \mathrm { G D } } \| _ { * } - \| W ^ { \mathrm { G D } } - a b ^ { \top } \| _ { * } , } \end{array}
$$

and symmetrically

$$
\begin{array} { r l } & { \| W ^ { \mathrm { G D } } \| _ { * } - \| W ^ { \mathrm { G D } } - a b ^ { \top } \| _ { * } = \langle W ^ { \mathrm { G D } } , W ^ { \mathrm { M u o n } } \rangle - \| W ^ { \mathrm { G D } } - a b ^ { \top } \| _ { * } } \\ & { \qquad \leq \langle W ^ { \mathrm { G D } } , W ^ { \mathrm { M u o n } } \rangle - \langle W ^ { \mathrm { G D } } - a b ^ { \top } , W ^ { \mathrm { M u o n } } \rangle = a ^ { \top } W ^ { \mathrm { M u o n } } b . } \end{array}
$$

Lemma F.29 (Self term). $I f \gamma \geq 1 6$ , then with probability tending to one

$$
0 \leq \frac { 1 } { ( N - 1 ) d } \operatorname* { m e a n } _ { i } \left( u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } \right) ^ { 2 } \leq 1 5 \gamma ^ { - 3 / 2 } .\tag{67}
$$

This bounds the diagonal term of Theorem F.23; it is what forces the $\gamma ^ { - 3 / 2 }$ error in $\bar { \kappa } ^ { \mathrm { M u o n } }$ , and the only place where Theorem F.28 is used.

Proof. Nonnegativity is clear.

Step 1 (reduction to a first absolute moment). By $\begin{array} { r } { \| W ^ { \mathrm { M u o n } } \| _ { \mathrm { o p } } \leq 1 , | u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } | \leq \| u _ { i } \| \| v _ { i } \| , } \end{array}$ so

$$
\begin{array} { r l } & { \operatorname* { m e a n } _ { i } \left( u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } \right) ^ { 2 } \leq d \operatorname* { m e a n } _ { i } \left| u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } \right| } \\ & { \qquad + \left( \operatorname* { m e a n } _ { i } \left( u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } \right) ^ { 2 } \right) ^ { 1 / 2 } \Big ( \operatorname* { m e a n } _ { i } \left( \| u _ { i } \| \| v _ { i } \| - d \right) ^ { 2 } \Big ) ^ { 1 / 2 } . } \end{array}\tag{68}
$$

For $x , c _ { 0 } , c _ { 1 } \geq 0 , x ^ { 2 } \leq c _ { 0 } + c _ { 1 } x$ forces $\begin{array} { r } { x \leq \frac { c _ { 1 } + \sqrt { c _ { 1 } ^ { 2 } + 4 c _ { 0 } } } { 2 } \leq c _ { 1 } + \sqrt { c _ { 0 } } } \end{array}$ and hence $x ^ { 2 } \leq 2 c _ { 1 } ^ { 2 } + 2 c _ { 0 } ;$ taking x = (mean<sub>i</sub>(u<sup>⊤</sup>W <sup>Muon</sup>v<sub>i</sub>)<sup>2</sup>)<sup>1/2</sup>, c<sub>0</sub> = d mean<sub>i</sub>|u<sup>⊤</sup>W <sup>Muon</sup>v<sub>i</sub>| and $c _ { 1 } = ( \mathrm { m e a n } _ { i } ( \| u _ { i } \| \| v _ { i } \| - d ) ^ { 2 } ) ^ { 1 / 2 }$ , and using Theorem F.3(ii),

$$
\operatorname* { m e a n } _ { i } \left( u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } \right) ^ { 2 } \leq O _ { p } ( d ) + 2 d \operatorname* { m e a n } _ { i } \left| u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } \right| .\tag{69}
$$

Step 2 (a signed mean). By Theorem F.17 and Theorem F.18, mean $\begin{array} { r } { _ { i } u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } = \frac { \| W ^ { \mathrm { G D } } \| _ { * } } { N } + O _ { p } ( d / N ) } \end{array}$ . For real $x \overset { \bar { } } { \underset { } { \geq } } \ell$ one has

$$
| x | \leq x + 2 | \ell | :\tag{70}
$$

this is clear for $x \ge 0$ , and for $x \ : < 0 \ :$ it follows from $| x | = - x \leq - \ell \leq | \ell | \leq x + 2 | \ell |$ , the last step because $x + 2 | \ell | \geq \ell + 2 | \ell | \geq | \ell |$

Step 3 (conditioning). Fix i and set

$$
a : = \frac { u _ { i } - \frac { 1 } { N } U { \bf 1 } } { \sqrt { 1 - 1 / N } } , \qquad b : = \frac { v _ { i } - \frac { 1 } { N } V { \bf 1 } } { \sqrt { 1 - 1 / N } } .
$$

Since $\begin{array} { r } { \mathbf { 1 } ^ { \top } ( e _ { i } - \frac { 1 } { N } \mathbf { 1 } ) = 0 } \end{array}$ and $\begin{array} { r } { \| e _ { i } - \frac { 1 } { N } \mathbf { 1 } \| ^ { 2 } = 1 - \frac { 1 } { N } } \end{array}$ , Equation (11) gives

$$
W ^ { \mathrm { G D } } - a b ^ { \top } = U \Big [ \Big ( I _ { N } - \frac { { \bf 1 1 } ^ { \top } } { N } \Big ) - \frac { ( e _ { i } - \frac { 1 } { N } { \bf 1 } ) ( e _ { i } - \frac { 1 } { N } { \bf 1 } ) ^ { \top } } { 1 - 1 / N } \Big ] V ^ { \top } ,\tag{71}
$$

where the bracket, call it $M ,$ is the orthogonal projection onto $\{ x \in \mathbb { R } ^ { N } : { \mathbf { 1 } } ^ { \top } x = 0 , ( e _ { i } - \frac { 1 } { N } { \mathbf { 1 } } ) ^ { \top } x = 0 \}$ . Indeed $M = P - \Pi$ with $\begin{array} { r } { P : = I _ { N } - \frac { { \mathbf { 1 1 } } ^ { \top } } { N } } \end{array}$ and $\begin{array} { r } { \Pi : = ( e _ { i } - \frac { 1 } { N } \mathbf { 1 } ) ( e _ { i } - \frac { 1 } { N } \mathbf { 1 } ) ^ { \top } / \| e _ { i } - \frac { 1 } { N } \mathbf { 1 } \| ^ { 2 } } \end{array}$ the orthogonal projections onto $\mathbf { 1 } ^ { \perp }$ and onto span $\begin{array} { r } { ( e _ { i } - \frac { 1 } { N } \mathbf { 1 } ) } \end{array}$ ; since $\mathbf { 1 } ^ { \top } ( e _ { i } - \frac { 1 } { N } \mathbf { 1 } ) \stackrel {  } { = } 0$ the range of Π lies in that of $P ,$ so $P \Pi = \Pi P = \Pi$ and hence $M ^ { \top } = M , M ^ { 2 } = \ddot { M }$ and $\begin{array} { r } { M ( e _ { i } - \frac { 1 } { N } { \bf 1 } ) = 0 } \end{array}$ . Consequently a and b are standard Gaussian and independent of $W ^ { \mathrm { G D } } - a b ^ { \intercal }$ . For $a \mathrm { : }$ its k-th coordinate is $\begin{array} { r } { ( e _ { i } - \frac { 1 } { N } \mathbf { 1 } ) ^ { \top } U ^ { \top } e _ { k } / \sqrt { 1 - 1 / N } } \end{array}$ and the k-th row of UM is $M U ^ { \top } e _ { k } ;$ conditionally on $V$ the vectors $U ^ { \top } e _ { k }$ are i.i.d. $\mathcal { N } ( 0 , I _ { N } )$ , and

$$
\mathrm { C o v } \left( ( e _ { i } - \frac { 1 } { N } { \bf 1 } ) ^ { \top } U ^ { \top } e _ { k } , ~ M U ^ { \top } e _ { k } \right) = M ( e _ { i } - \frac { 1 } { N } { \bf 1 } ) = 0 ,\tag{72}
$$

so, being jointly Gaussian, a is independent of UM, and $\| ( e _ { i } - \frac { 1 } { N } \mathbf { 1 } ) / \sqrt { 1 - 1 / N } \| = 1$ gives $a \sim \mathcal { N } ( 0 , I _ { d } )$ . The same argument applied to the rows of ${ \bf \nabla } \cdot \dot { V }$ gives $b \sim \mathcal { N } ( 0 , I _ { d } )$ independent of $V M ,$ and $U \perp { \bf \bar { V } }$ makes $a , b$ independent of each other; since $W ^ { \mathrm { G D } } - a b ^ { \top } = ( U M ) ( V M ) ^ { \top }$ by idempotence of $M ,$ the pair $( a , b )$ is independent of $\bar { W } ^ { \mathrm { G D } } - a b ^ { \top }$ Step 4 (two mean-square bounds). For independent $a , b \sim \mathcal { N } ( 0 , I _ { d } )$ and $Z$ independent of them, $\mathbb { E } [ ( a ^ { \top } Z b ) ^ { 2 } \mid Z ] =$ $\| Z \| _ { F } ^ { 2 }$ ; taking $Z = \dot { \mathrm { O r t h } } ( W ^ { \mathrm { G D } } - a b ^ { \top } )$ ), whose squared Frobenius norm is its rank, at most $d ,$ and averaging over $i ,$

$$
\left( \operatorname* { m e a n } _ { i } \left( a ^ { \top } \operatorname { O r t h } ( W ^ { \mathrm { G D } } - a b ^ { \top } ) b \right) ^ { 2 } \right) ^ { 1 / 2 } = O _ { p } ( \sqrt { d } ) .\tag{73}
$$

Substituting $\begin{array} { r } { u _ { i } = \sqrt { 1 - \frac { 1 } { N } } a + \frac { 1 } { N } U \mathbf { 1 } } \end{array}$ and $\begin{array} { r } { v _ { i } = \sqrt { 1 - \frac { 1 } { N } } b + \frac { 1 } { N } V { \bf 1 } } \end{array}$

$$
u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } - \Big ( 1 - \frac { 1 } { N } \Big ) a ^ { \top } W ^ { \mathrm { M u o n } } b = \sqrt { 1 - \frac { 1 } { N } } a ^ { \top } W ^ { \mathrm { M u o n } } \frac { V { \bf 1 } } { N } + \sqrt { 1 - \frac { 1 } { N } } \frac { ( U { \bf 1 } ) ^ { \top } } { N } W ^ { \mathrm { M u o n } } b
$$

$$
+ \frac { ( U { \bf 1 } ) ^ { \top } } { N } W ^ { \mathrm { M u o n } } \frac { V { \bf 1 } } { N } ,\tag{74}
$$

so by $( x + y + z ) ^ { 2 } \leq 3 ( x ^ { 2 } + y ^ { 2 } + z ^ { 2 } ) , \| W ^ { \mathrm { M u o n } } \| _ { \mathrm { o p } } \leq 1 , \mathbb { E } \| a \| ^ { 2 } = \mathbb { E } \| b \| ^ { 2 } = d$ (so that mean $_ i \| a \| ^ { 2 } = O _ { p } ( d )$ by Markov) and Theorem F.3(iii),

$$
\begin{array} { r l r } {  { \operatorname* { m e a n } _ { i } ( u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } - \big ( 1 - \frac { 1 } { N } \big ) a ^ { \top } W ^ { \mathrm { M u o n } } b ) ^ { 2 } } } \\ & { } & { \leq 3 \Big [ \operatorname* { m e a n } _ { i } \| a \| ^ { 2 } \frac { \| V { \bf 1 } \| ^ { 2 } } { N ^ { 2 } } + \frac { \| U { \bf 1 } \| ^ { 2 } } { N ^ { 2 } } \operatorname* { m e a n } _ { i } \| b \| ^ { 2 } + \frac { \| U { \bf 1 } \| ^ { 2 } \| V { \bf 1 } \| ^ { 2 } } { N ^ { 4 } } \Big ] = O _ { p } \Big ( \frac { d ^ { 2 } } { N } \Big ) , } \end{array}\tag{75}
$$

that is

$$
\left( \operatorname* { m e a n } _ { i } \Big ( u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } - \big ( 1 - \frac { 1 } { N } \big ) a ^ { \top } W ^ { \mathrm { M u o n } } b \Big ) ^ { 2 } \right) ^ { 1 / 2 } = O _ { p } \big ( d / \sqrt { N } \big ) .\tag{76}
$$

Step 5 (combining). By Theorem F.28 and $\begin{array} { r } { 1 - \frac { 1 } { N } > 0 , u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } \geq ( 1 - \frac { 1 } { N } ) a ^ { \top } \mathrm { O r t h } ( W ^ { \mathrm { G D } } - a b ^ { \top } ) b - | u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } - \frac { 1 } { N } | a ^ { \top } \mathrm { O r t h } ( W ^ { \mathrm { G D } } - a b ^ { \top } ) | , } \end{array}$ $( 1 - \frac { 1 } { N } ) a ^ { \top } W ^ { \mathrm { M u o n } } b |$ . Using Equation (70) of Step 2 with this lower bound, averaging over $i ,$ and applying Equation (73), Equation (76) and $\| W ^ { \mathrm { G D } } \| _ { * } \le 7 d \sqrt { N d }$ from Lemma F.4(iii),

$$
\operatorname* { m e a n } _ { i } \big | u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } \big | \leq \frac { \| W ^ { \mathrm { G D } } \| _ { * } } { N } + O _ { p } ( d / N ) + O _ { p } ( \sqrt { d } ) + O _ { p } ( d / \sqrt { N } ) \leq \frac { 7 d ^ { 3 / 2 } } { \sqrt { N } } + O _ { p } ( \sqrt { d } ) .\tag{77}
$$

Substituting into Equation (69) and dividing by $( N - 1 ) d ,$

$$
\begin{array} { r l r } {  { \frac { 1 } { ( N - 1 ) d } \operatorname* { m e a n } _ { i } ( u _ { i } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } ) ^ { 2 } \leq \frac { O _ { p } ( d ) } { ( N - 1 ) d } + \frac { 2 } { N - 1 } \Big ( \frac { 7 d ^ { 3 / 2 } } { \sqrt { N } } + O _ { p } ( \sqrt { d } ) \Big ) } } \\ & { } & { = 1 4 \gamma ^ { - 3 / 2 } \Big ( 1 + \frac { 1 } { N - 1 } \Big ) + o _ { p } ( 1 ) , } \end{array}\tag{78}
$$

which is at most $1 5 \gamma ^ { - 3 / 2 }$ with probability tending to one.

Lemma F.30 (Second moment for Muon). $I f \gamma \geq 1 6 $ , then $\begin{array} { r } { \frac { 1 } { N ( N - 1 ) d } \sum _ { i = 1 } ^ { N } \sum _ { j \neq i } \left( u _ { j } ^ { \top } W ^ { \mathrm { M u o n } } v _ { i } \right) ^ { 2 } = 1 + O ( \gamma ^ { - 3 / 2 } ) + } \end{array}$ $o _ { p } ( 1 )$ , where with probability tending to one the $O ( \gamma ^ { - 3 / 2 } )$ term has absolute value at most $5 2 \gamma ^ { - 3 / 2 }$

Proof. Divide Theorem F.23 by d and subtract 1. By Theorem F.24 the norm term contributes $O ( N ^ { - 1 } ) +$ $O _ { p } ( ( N d ) ^ { - 1 / 2 } ) = o _ { p } ( 1 )$ ; by Lemma F.27 the trace term is at most $\begin{array} { r } { 1 4 5 \gamma ^ { - 2 } \leq \frac { 1 4 5 } { 4 } \gamma ^ { - 3 / 2 } } \end{array}$ in absolute value for $\gamma \geq 1 6 ;$ by Lemma F.29 the diagonal term lies in $[ - 1 5 \gamma ^ { - 3 / 2 } , 0 ]$ . Adding, $\textstyle { \frac { 1 4 5 } { 4 } } + 1 5 \leq 5 2$ □

Lemma F.31 (κ¯ for Muon). $I f \gamma \geq 1 6$ , then $( \bar { \kappa } ^ { \mathrm { M u o n } } ) ^ { 2 } = d \big ( 1 + O ( \gamma ^ { - 3 / 2 } ) + o _ { p } ( 1 ) \big )$ with the same constant as in Theorem F.30.

Proof. By Equation (37) with $W = W ^ { \mathrm { M u o n } }$ , Theorem F.30 and Theorem F.21, $( \bar { \kappa } ^ { \mathrm { M u o n } } ) ^ { 2 } = d ( 1 + O ( \gamma ^ { - 3 / 2 } ) + o _ { p } ( 1 ) ) -$ $O _ { p } ( d / N ) = d ( 1 + O ( \gamma ^ { - 3 / 2 } ) + o _ { p } ( 1 ) )$ □

Lemma F.32 (ρ for Muon). $I f \gamma \geq 1 6$ , then

$$
\rho ^ { \mathrm { M u o n } } = \sqrt { \frac { d } { \gamma } } \Big [ \frac { 8 } { 3 \pi } \Big ( 1 - \frac { 1 } { 1 0 \gamma } \Big ) + { \cal O } ( \gamma ^ { - 3 / 2 } ) + o _ { p } ( 1 ) \Big ] .\tag{79}
$$

Unlike for GD, the data budget enters $\rho ^ { \mathrm { M u o n } }$ a second time, through $\Phi ( \gamma )$

Proof. By Theorem F.22, $\bar { h } ^ { \mathrm { M u o n } } / \sqrt { d } = \sqrt { d / \gamma } \Phi ( \gamma ) ( 1 + o _ { p } ( 1 ) )$ . Set $y : = ( \bar { \kappa } ^ { \mathrm { M u o n } } ) ^ { 2 } / d .$ By Theorem F.31 and $\gamma \geq 1 6$ with probability tending to one

$$
| y - 1 | \leq 5 2 \gamma ^ { - 3 / 2 } + o _ { p } ( 1 ) \leq \frac { 5 2 } { 6 4 } + o _ { p } ( 1 ) \leq \frac { 9 } { 1 0 } ,
$$

so $\begin{array} { r } { y \ge \frac { 1 } { 1 0 } } \end{array}$ . For $\begin{array} { r } { y \geq \frac { 1 } { 1 0 } . } \end{array}$

$$
| y ^ { - 1 / 2 } - 1 | = \frac { | 1 - y | } { \sqrt { y } \left( 1 + \sqrt { y } \right) } \leq \sqrt { 1 0 } | y - 1 | ,
$$

which gives $\sqrt { d } / \bar { \kappa } ^ { \mathrm { M u o n } } = 1 + O ( \gamma ^ { - 3 / 2 } ) + o _ { p } ( 1 )$ . Multiplying the two,

$$
\rho ^ { \mathrm { M u o n } } = \frac { \bar { h } ^ { \mathrm { M u o n } } } { \sqrt { d } } \cdot \frac { \sqrt { d } } { \bar { \kappa } ^ { \mathrm { M u o n } } } = \sqrt { \frac { d } { \gamma } } \Phi ( \gamma ) \bigl ( 1 + { \cal O } ( \gamma ^ { - 3 / 2 } ) + o _ { p } ( 1 ) \bigr ) ,\tag{80}
$$

and Lemma F.4(iv) gives $\begin{array} { r } { \Phi ( \gamma ) = \frac { 8 } { 3 \pi } ( 1 - \frac { 1 } { 1 0 \gamma } + r ( \gamma ) ) } \end{array}$ ) with $\begin{array} { r } { | r ( \gamma ) | \leq \gamma ^ { - 2 } \leq \frac { 1 } { 4 } \gamma ^ { - 3 / 2 } } \end{array}$ , so $\begin{array} { r } { 0 < \Phi ( \gamma ) \le \frac { 8 } { 3 \pi } ( 1 + \gamma ^ { - 2 } ) < 1 } \end{array}$ and $\begin{array} { r } { \Phi ( \gamma ) ( 1 + O ( \gamma ^ { - 3 / 2 } ) + o _ { p } ( 1 ) ) = \frac { 8 } { 3 \pi } ( 1 - \frac { 1 } { 1 0 \gamma } ) + O ( \gamma ^ { - 3 / 2 } ) + o _ { p } ( 1 ) . } \end{array}$

Two by-products of this computation are used in the proof of Theorem 4.2. First, the constant implied by the $O ( \gamma ^ { - 3 / 2 } )$ in the displayed multiplicative form is absolute, so there is an absolute $\gamma _ { 1 } \geq 1 6$ such that for $\gamma \geq \gamma _ { 1 }$ that factor lies in $[ \frac { 1 } { 2 } , \frac { 3 } { 2 } ]$ with probability tending to one. Second, for every $\gamma \geq$ 16 the same display bounds $\rho ^ { \mathrm { M u o n } }$ away from zero without any expansion: $y \leq { \frac { 1 9 } { 1 0 } }$ gives $\begin{array} { r } { \sqrt { d } / \bar { \kappa } ^ { \mathrm { M u o n } } \ge ( 1 9 / 1 0 ) ^ { - 1 / 2 } > \frac { 7 } { 1 0 } } \end{array}$ , and $\begin{array} { r } { \Phi ( \gamma ) \ge \frac { 1 } { 8 } } \end{array}$ by Lemma F.4(iii),(iv), so with probability tending to one

$$
\rho ^ { \mathrm { M u o n } } \geq \frac { 7 } { 8 0 } \sqrt { d / \gamma } \left( 1 - o _ { p } ( 1 ) \right) \geq \frac { 1 } { 1 6 } \sqrt { d / \gamma } .\tag{81}
$$

## F.5 Combining the GD and Muon estimates

ProofofEquation (8). Reduction to large γ. By Lemma F.16 and $2 + \gamma \geq \gamma , \rho ^ { \mathrm { G D } } \leq \sqrt { d / \gamma } \left( 1 + o _ { p } ( 1 ) \right)$ ), so Equation (81) gives $\mathcal { R } \leq 1 7$ with probability tending to one, for every $\gamma \geq 1 6$ ; together with $\begin{array} { r } { \dot { 0 } \le \frac { 3 \pi } { 8 } \bigl ( 1 - \frac { 9 } { 1 0 \gamma } \bigr ) \le \frac { 3 \pi } { 8 } } \end{array}$ this makes the left-hand side of Equation (8) at most 19 there. On a bounded range $1 6 \le \gamma \le \gamma _ { 1 }$ we have $9 \leq 1 9 \gamma _ { 1 } ^ { 3 / 2 } \gamma ^ { - 3 / 2 }$ so Equation (8) holds on that range as soon as $C _ { 1 } \geq 1 9 \gamma _ { 1 } ^ { 3 / 2 }$ . It therefore sufices to prove it for $\gamma \geq \gamma _ { 1 }$ , with $\gamma _ { 1 }$ the absolute constant of Lemma F.32; on that range the error factor of Lemma F.32 lies in $[ \frac { 1 } { 2 } , \frac { 3 } { 2 } ]$ with probability tending to one, so its reciprocal is again $1 + O ( \gamma ^ { - 3 / 2 } ) + o _ { p } ( 1 )$

Expansion. By Lemma F.16 and Lemma F.32, and since $\Phi ( \gamma ) > 0$

$$
\begin{array} { r l } {  { \mathcal { R } = \frac { \rho ^ { \mathrm { G D } } } { \rho ^ { \mathrm { M u o n } } } = \sqrt { \frac { d } { 2 + \gamma } } \big ( 1 + o _ { p } ( 1 ) \big ) \cdot \sqrt { \frac { \gamma } { d } } \frac { 1 } { \Phi ( \gamma ) \big ( 1 + O \big ( \gamma ^ { - 3 / 2 } \big ) + o _ { p } ( 1 ) \big ) } } } \\ & { \quad \quad \quad = \frac { \sqrt { \gamma / ( 2 + \gamma ) } } { \Phi ( \gamma ) } \big ( 1 + O ( \gamma ^ { - 3 / 2 } ) + o _ { p } ( 1 ) \big ) . } \end{array}\tag{82}
$$

The deterministic prefactor is expanded in $1 / \gamma \colon \sqrt { \gamma / ( 2 + \gamma ) } = ( 1 + 2 / \gamma ) ^ { - 1 / 2 } = 1 - \gamma ^ { - 1 } + O ( \gamma ^ { - 2 } )$ , while Lemma F.4(iv) gives $\begin{array} { r } { \Phi ( \gamma ) ^ { - 1 } = \frac { 3 \pi } { 8 } \bigl ( 1 + \frac { 1 } { 1 0 \gamma } + { O } ( \gamma ^ { - 2 } ) \bigr ) } \end{array}$ , so

$$
\frac { \sqrt { \gamma / ( 2 + \gamma ) } } { \Phi ( \gamma ) } = \frac { 3 \pi } { 8 } \Big ( 1 - \frac { 1 } { \gamma } + { \cal O } ( \gamma ^ { - 2 } ) \Big ) \Big ( 1 + \frac { 1 } { 1 0 \gamma } + { \cal O } ( \gamma ^ { - 2 } ) \Big ) = \frac { 3 \pi } { 8 } \Big ( 1 - \frac { 9 } { 1 0 \gamma } \Big ) + { \cal O } ( \gamma ^ { - 2 } ) .\tag{83}
$$

For $\gamma \geq 1 6$ the prefactor is bounded, $\begin{array} { r } { 0 < \sqrt { \gamma / ( 2 + \gamma ) } / \Phi ( \gamma ) \le \frac { 3 \pi } { 8 } ( 1 - \frac { 1 } { 1 6 0 } - \frac { 1 } { 2 5 6 } ) ^ { - 1 } } \end{array}$ , so multiplying by $1 +$ $O ( \gamma ^ { - 3 / 2 } ) + o _ { p } ( 1 )$ leaves the error of the same order; and $\gamma ^ { - 2 } \leq \mathring { \frac { 1 } { 4 } } \gamma ^ { - 3 / 2 }$ in that range absorbs the $O ( \gamma ^ { - 2 } )$ term.

$$
\mathcal { R } = \frac { 3 \pi } { 8 } \Big ( 1 - \frac { 9 } { 1 0 \gamma } \Big ) + O ( \gamma ^ { - 3 / 2 } ) + o _ { p } ( 1 ) ,\tag{84}
$$

where, with probability tending to one, the $O ( \gamma ^ { - 3 / 2 } )$ term has absolute value at most $C _ { 1 } \gamma ^ { - 3 / 2 }$ with $C _ { 1 }$ independent of $\gamma , d , N$ . This is Equation (8). □

## F.6 Monotonicity in the data budget

The expansion Equation (8) has two consequences for how R changes with $\gamma .$ The first compares a fixed pair of data budgets; the second concerns the limit $d \to \infty$

Lemma F.33 (Fixed ratio). Let $\lambda > 0$ and

$$
\gamma > \gamma _ { 0 } ( \lambda ) : = \operatorname* { m a x } \Big \{ 1 6 , \Big ( \frac { 1 6 0 C _ { 1 } ( 1 + \lambda ) } { 2 7 \pi \lambda } \Big ) ^ { 2 } \Big \} .
$$

For the same d, let $N _ { 1 } = \gamma d$ and $N _ { 2 } = ( 1 + \lambda ) \gamma d ,$ , and let $\mathcal { R } _ { N _ { 1 } , d }$ and $\mathcal { R } _ { N _ { 2 } , d }$ be the corresponding values of R. Then, with probability tending to one, $\mathcal { R } _ { N _ { 2 } , d } > \mathcal { R } _ { N _ { 1 } , d }$

Proof. Apply Equation (8) once at $\gamma$ and once at $( 1 + \lambda ) \gamma$ , both of which are at least 16: for every $\varepsilon > 0$ , with probability tending to one,

$$
\begin{array} { r l r } {  { \mathcal { R } _ { N _ { 2 } , d } - \mathcal { R } _ { N _ { 1 } , d } \geq \frac { 3 \pi } { 8 } \Big ( 1 - \frac { 9 } { 1 0 ( 1 + \lambda ) \gamma } \Big ) - \frac { 3 \pi } { 8 } \Big ( 1 - \frac { 9 } { 1 0 \gamma } \Big ) } } \\ & { } & { - C _ { 1 } \big ( ( 1 + \lambda ) \gamma \big ) ^ { - 3 / 2 } - C _ { 1 } \gamma ^ { - 3 / 2 } - 2 \varepsilon } \\ & { } & { \geq \frac { 2 7 \pi } { 8 0 } \cdot \frac { \lambda } { ( 1 + \lambda ) \gamma } - 2 C _ { 1 } \gamma ^ { - 3 / 2 } - 2 \varepsilon . } \end{array}\tag{85}
$$

Moreover $\frac { 2 7 \pi } { 8 0 } ~ \cdot ~ \frac { \lambda } { ( 1 { + } \lambda ) \gamma } ~ > ~ 2 C _ { 1 } \gamma ^ { - 3 / 2 }$ is equivalent to $\begin{array} { r } { \gamma ^ { 1 / 2 } > \frac { 1 6 0 C _ { 1 } ( 1 + \lambda ) } { 2 7 \pi \lambda } } \end{array}$ , which holds by assumption; taking $\begin{array} { r } { \varepsilon = \frac 1 4 \big ( \frac { 2 7 \pi } { 8 0 } \cdot \frac \lambda { ( 1 + \lambda ) \gamma } - 2 C _ { 1 } \gamma ^ { - 3 / 2 } \big ) } \end{array}$ gives the claim. □

Deduction of the informal theorem. Write $A = 3 \pi / 8$ and $B = 2 7 \pi / 8 0$ . By Equation $( 8 ) , \mathcal { R } - A = - B / \gamma +$ $O ( \gamma ^ { - 3 / 2 } ) + o _ { p } ( 1 )$ . For suficiently large fixed $\gamma ,$ the $O ( \gamma ^ { - 3 / 2 } )$ term has absolute value at most $B / ( 4 \gamma )$ with probability tending to one, and the $o _ { p } ( 1 )$ term has absolute value at most $B / ( 4 \gamma )$ with probability tending to one as $d \to \infty$ Consequently,

$$
- \frac { 3 B } { 2 \gamma } \le \mathcal { R } - A \le - \frac { B } { 2 \gamma }
$$

with probability tending to one. This gives the claimed two-sided $1 / \gamma$ rate, and $\mathcal { R } > 1$ once $\gamma$ is suficiently large because $A > 1$ . Together with Lemma F.33, this proves all three conclusions of Theorem 4.2.

Lemma F.34 (Pointwise monotonicity). Suppose that for every suficiently large γ the limit in probability $\mathcal { R } _ { \infty } ( \gamma ) : =$ $\operatorname* { l i m } _ { d \to \infty } \mathcal { R }$ exists, and that $\mathcal { R } _ { \infty }$ is analytic as a function $o f 1 / \gamma$ in a neighbourhood of0, i.e. $\begin{array} { r } { \dot { \mathcal { R } _ { \infty } } ( \gamma ) = \dot { \sum } _ { k \ge 0 } \dot { c } _ { k } \gamma ^ { - k } } \end{array}$ for all suficiently large γ. Then there $i s \gamma _ { 2 }$ such that $\mathcal { R } _ { \infty } ^ { \prime } ( \gamma ) > 0 f o r \gamma > \gamma _ { 2 }$

Proof. By Equation (8) and convergence in probability, $\begin{array} { r } { | \mathcal { R } _ { \infty } ( \gamma ) - \frac { 3 \pi } { 8 } ( 1 - \frac { 9 } { 1 0 \gamma } ) | \leq C _ { 1 } \gamma ^ { - 3 / 2 } } \end{array}$ , that is

$$
\Big | \sum _ { k \geq 0 } c _ { k } \gamma ^ { - k } - \frac { 3 \pi } { 8 } + \frac { 2 7 \pi } { 8 0 } \gamma ^ { - 1 } \Big | \leq C _ { 1 } \gamma ^ { - 3 / 2 } .\tag{86}
$$

Letting $\gamma \to \infty$ gives $\begin{array} { r } { c _ { 0 } = \frac { 3 \pi } { 8 } } \end{array}$ ; dividing both sides by $\gamma ^ { - 1 }$ and letting $\gamma \to \infty$ gives $\begin{array} { r } { c _ { 1 } = - \frac { 2 7 \pi } { 8 0 } } \end{array}$ . Hence

$$
\mathcal { R } _ { \infty } ^ { \prime } ( \gamma ) = - \sum _ { k \geq 1 } k c _ { k } \gamma ^ { - k - 1 } = \gamma ^ { - 2 } \Big ( \frac { 2 7 \pi } { 8 0 } - \sum _ { k \geq 2 } k c _ { k } \gamma ^ { - ( k - 1 ) } \Big ) ,\tag{87}
$$

and the series in the parentheses tends to 0 as $\gamma \to \infty$ , so the parenthesized quantity is positive for all suficiently large γ. □

Remark F.35. Neither hypothesis of Theorem F.34 is proved here. The estimates above pin $( \bar { \kappa } ^ { \mathrm { M u o n } } ) ^ { 2 } / d$ down only to within $O ( \gamma ^ { - 3 / 2 } )$ (Theorem F.31), so they do not by themselves give the limit in probability; $\rho ^ { \mathrm { G D } } / \sqrt { d }$ and $\bar { h } ^ { \mathrm { M u o n } } / d$ by contrast, do converge, by Lemma F.16 and Theorem F.22. Analyticity does not follow from Equation (8) alone either: with $x = 1 / \gamma$ , the function $1 - x + x ^ { 3 / 2 } \sin ( 1 / x )$ admits an expansion of the same form and is not monotone near $x = 0$

Numerical consequences. Since $\textstyle { \frac { 3 \pi } { 8 } } \approx 1 . 1 7 8 > 1$ , Equation (8) puts R above one for all large $\gamma \colon$ past some data budget the GD iterate has the larger margin-to-spread ratio. The leading term $\begin{array} { r } { \frac { 3 \pi } { 8 } ( 1 - \frac { 9 } { 1 0 \gamma } ) } \end{array}$ is increasing in $\gamma ;$ Theorem F.33 turns this into a statement about R itself, and Theorem F.34 into one about its limit, under the hypotheses discussed in Theorem F.35. The leading term crosses one at $\gamma \approx 5 . 9 5$ , below the range $\gamma \geq 1 6$ in which Lemma F.32 is proved, so the location of the crossing is established empirically rather than by Theorem 4.2.