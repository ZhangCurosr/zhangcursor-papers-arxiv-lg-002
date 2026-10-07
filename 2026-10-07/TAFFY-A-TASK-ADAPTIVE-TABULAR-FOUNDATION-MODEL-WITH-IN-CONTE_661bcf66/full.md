![](images/c8ed0dd61bd71d28922543ca21923c0bbb638aac923c5172a413f6a72a1eb284.jpg)

# TAFFY: A TASK-ADAPTIVE TABULAR FOUNDATION MODEL WITH IN-CONTEXT DIVERSITY

Zijian Li<sup>1,2</sup>, Xiangchen Song<sup>1</sup>, Gongxu Luo<sup>2</sup>, Jie Qiao<sup>3</sup>, Ruichu Cai<sup>3</sup>,

Zhenhao Chen<sup>2</sup>, Xinshuai Dong<sup>1</sup>, Fan Feng<sup>4</sup>, Guangyi Chen<sup>1,2</sup>, Kun Zhang<sup>1,2</sup>

<sup>1</sup>Carnegie Mellon University <sup>2</sup>Mohamed bin Zayed University of Artificial Intelligence

<sup>3</sup>Guangdong University of Technology <sup>4</sup>University of California, San Diego

## ABSTRACT

Recent progress in tabular foundation models suggests that training on synthetic tasks can substantially improve in-context learning capabilities, with overall performance largely depending on how well models can infer task-specific predictive relationships from the available context during inference. In this paper, we introduce TAFFY, a tabular foundation model with an In-Context Diversity Prior and a Task-Conditioned Looped Transformer that strengthen this ability. Specifically, to construct each synthetic pretraining context, the In-Context Diversity Prior samples from multiple related environments derived via controlled interventions and distribution shifts on a shared causal process. This in-context diversity encourages the model to learn a more comprehensive and task-specific representation. Moreover, the Task-Conditioned Looped Transformer iteratively and selectively applies a shared group of Transformer blocks to refine contextual representations, with a task-conditioned gate modulating the final hidden-state update. This enables taskadaptive iterative refinement. Together, these components encourage the model to identify predictive relationships from contextual contrasts during pretraining and dynamically modulate context integration for each task. Across six classification and five regression benchmark datasets, TAFFY attains the lowest average rank.

## 1 INTRODUCTION

Tabular foundation models leverage in-context learning (ICL) to predict query labels from labeled support examples without explicit parameter updates. Prior-data fitted networks formulate this paradigm as approximating Bayesian inference under a task prior: synthetic tasks provide support– query pairs, and pretraining optimizes query predictions conditioned on the support set (Müller et al., 2022). Recent architectures such as TabPFN, TabICL, LimiX, and Mitra have advanced tabular ICL through developments in synthetic data generation and prediction architectures (Hollmann et al., 2023; 2025; Qu et al., 2025; 2026; Zhang et al., 2025a; Wang et al., 2026; Zhang et al., 2025b). Because semantic meanings and feature dependencies shift across datasets, a model cannot rely on fixed feature semantics; instead, it must infer task-specific functional dependencies directly from the support context—an intuition formalized by recent kernel and nearest-neighbor views of tabular ICL (Koshil et al., 2025; Miftachov et al., 2026). Consequently, generalization to unseen tasks depends on how effectively the model learns task-specific predictive relationships from context.

To strengthen this ability, we introduce TAFFY, which combines an In-Context Diversity Prior for synthetic task construction with a Task-Conditioned Looped Transformer for task-dependent context processing. First, the In-Context Diversity Prior incorporates observations from multiple related environments into a single synthetic pretraining context. Starting from a randomly sampled causal prototype, we first generate a set of environments through controllable distribution shifts (Zhang et al., 2015) and interventions (Pearl, 2009), then combine their support data and query data to form one prediction task. This design is inspired by causal discovery under changing distributions (Huang et al., 2020a) and latent-variable learning with auxiliary information (Hyvärinen et al., 2019), which allow the model to explore a sufficient amount of variation across different data generation condi tions. Although the model is not provided with environment identifiers or access to the generating process, the mixing prior enables observations from related conditions which provides contrasting evidence about predictive relationships. Training on these contrasts encourages TAFFY to infer taskspecific predictive relationships from the available context.

(a) OpenML  
![](images/89218e0cd1a67c5bea84bef86abbf04ac47ca768af83825c1c264f02ae8ad23f.jpg)  
(b) BCCO  
(c) TALENT

![](images/d1a96b4d44f815fa53bb4774cd1bf1231fd331a975ec9660724a9a8dba92837f.jpg)  
(d) Training-data efficiency  
Figure 1: Predictive performance and pretraining-data efficiency. (a–c) Average accuracy ranks on OpenML-CC18, BCCO, and TALENT using the ranking pools of Table 1 (lower is better; rank axes start at 2). (d) Accuracies on TALENT classification datasets versus cumulative maximum feature elements. Markers denote measured checkpoints connected by lines; the grey line marks TabICLv2’s best measured accuracy.

Second, the Task-Conditioned Looped Transformer combines a looped transformer stack with a task-specific gate to strengthen the model’s ability to learn task-specific predictive relationships from context. Building on representations produced by the column and row encoders, our model repeatedly applies a shared ICL Transformer stack (Dehghani et al., 2019; Yang et al., 2024) to further integrate information from the labeled context. Later passes recompute attention using representations already informed by support features and labels, allowing them to build on information integrated earlier. Since the usefulness of further refinement may vary across tasks, we introduce a task-conditioning module that maps support-dataset statistics to a task-dependent scalar gate modulating the final hidden-state update. The shared loop enables iterative context processing, while the gate adjusts how strongly the final refinement contributes to prediction with the number of passes held fixed. Since these feature and label statistics are computed solely from the labeled support and unlabeled query dataset, this task-dependent refinement is also applicable when inferring predictive relationships for unseen real-world tasks. Together, the in-context diversity prior provides relational evidence, while the task-conditioned looped Transformer refines the tabular representations.

We evaluate TAFFY across six classification and five regression benchmark suites. Using Taffy-4L throughout the classification evaluation, TAFFY obtains the lowest average rank on all six classification suites; the regression model ranks first on all five regression suites. Figure 1(a–c) shows selected foundation models on OpenML-CC18, BCCO, and TALENT, with the full comparison in Table 1. Figure 1(d) compares TAFFY with reproduced TabICLv2 (Qu et al., 2026) across measured pretraining budgets, providing an exploratory view of pretraining data efficiency. Section 5.4 defines the cumulative maximum-feature-element budget and reports the measured checkpoints.

## Our contributions are summarized as follows:

• We introduce a synthetic prior that places causally related environments within the same task.

• We develop a looped Transformer whose final refinement is modulated by task-specific statistics.

• TAFFY obtains the lowest average rank on all classification and regression benchmarks.

## 2 RELATED WORK

## 2.1 TABULAR FOUNDATION MODELS

Tabular foundation models learn from synthetic tasks to make predictions on real-world datasets through in-context learning (Qu et al., 2025; Zhang et al., 2025b). Building on prior-data fitted networks (Müller et al., 2022), the TabPFN family extends this approach across classification, regression, and larger tables (Hollmann et al., 2023; 2025; Grinsztajn et al., 2025). Generator design includes tree-based tasks, mixtures of priors, and distributional perturbations (den Breejen et al., 2024; Zhang et al., 2025b; Bouadi et al., 2026), while the TabICL and LimiX families develop table representations and contextual interactions (Qu et al., 2025; 2026; Zhang et al., 2025a; Wang et al., 2026). Other approaches use real-data pretraining (Ma et al., 2025; Garg et al., 2025) or self-supervised representation learning (Wu et al., 2024). Closely related to our task construction, Drift-Resilient TabPFN generates evolving SCM mechanisms and uses ordered domain indices for temporal extrapolation (Helli et al., 2024). TAFFY instead constructs unordered mixtures of related environments through distribution shifts and interventions, without supplying environment identifiers. Its focus is on placing observations from these conditions within the same support–query task, making their joint use part of the prediction problem practiced during pretraining.

## 2.2 LOOPED TRANSFORMERS

Universal Transformers and ALBERT share parameters across processing depth (Dehghani et al., 2019; Lan et al., 2020), while Relaxed Recursive Transformers add depth-specific low-rank adapters (Bae et al., 2025a). In ICL, looped Transformers have been studied as models of iterative algorithms (Yang et al., 2024), including connections to multi-step preconditioned gradient descent for linear regression (Gatmiry et al., 2024). Work on latent reasoning explores repeated computation in hidden states (Saunshi et al., 2025; Geiping et al., 2025), and CoTFormer permits attention to intermedi ate representations from earlier iterations (Mohtashami et al., 2025). For tabular prediction, Balef et al. (2026) investigate layerwise inference dynamics and a parameter-efficient single-layer looped model. A separate line of work learns halting or token-level computation allocation (Graves, 2016; Banino et al., 2021; Raposo et al., 2024; Chen et al., 2025; Bae et al., 2025b). In this paper, TAFFY uses shared multi-block recurrence to refine representations already conditioned on each task. A gate based on task statistics modulates only the final update, making its contribution task-dependent without token routing or changing the number of passes.

## 2.3 CAUSAL LEARNING

Our task prior is related to causal learning through its use of structured changes across environments (Schölkopf et al., 2021). Causal discovery studies changing mechanisms (Huang et al., 2020a) and cross-environment invariance (Peters et al., 2016), while multi-source domain adaptation relates distributional variation to transferable mechanisms (Zhang et al., 2015). Nonlinear ICA and iVAE establish latent identifiability using auxiliary information under specific assumptions (Hyvärinen et al., 2019; Khemakhem et al., 2020). Related work considers latent causal identification from in terventions (Varici et al., 2024) and content identification from paired augmentations (von Kügelgen et al., 2021). These results motivate studying structured variation, but do not establish identifiability for our mixtures without environment identifiers. In TAFFY, causal prototypes and interventions serve as tools for constructing related environments within the same pretraining task. The goal is to train prediction from their pooled observations, rather than to recover the generating process.

## 3 PROBLEM FORMULATION

Tabular Foundation Model with in-context learning. We consider classification and regression tasks on tabular data. For a given task, each sample has a feature vector $x \in \mathbb { R } ^ { d }$ and the corresponding label $y \in \mathcal { D }$ , where $\mathcal { V } = \{ 1 , \ldots , K \}$ for classification and $\mathcal { V } = \mathbb { R }$ for regression. The available data consist of a labeled support dataset $S = \{ ( x _ { i } ^ { s } , y _ { i } ^ { s } ) \} _ { i = 1 } ^ { n _ { s } }$ and unlabeled query dataset $X _ { Q } = ( x _ { j } ^ { q } ) _ { j = 1 } ^ { n _ { q } }$ whose corresponding labels $y _ { Q } = ( y _ { j } ^ { q } ) _ { j = 1 } ^ { n _ { q } }$ are unknown to the model. A tabular foundation model with parameters θ predicts the query labels by conditioning on the support dataset:

$$
( \hat { y } _ { 1 } , . . . , \hat { y } _ { n _ { q } } ) = f _ { \theta } ( X _ { Q } ; S ) ,\tag{1}
$$

where $\hat { y } _ { j }$ denotes the predictive output for query $x _ { i } ^ { q }$ . The output is a class-probability vector for classification or a vector of conditional quantile predictions for regression.

Pretraining objective. Following the prior-data fitted framework (Müller et al., 2022; Qu et al., 2026), we learn θ from synthetic prediction tasks. Let $\mathcal { E } = ( S , X _ { Q } , y _ { Q } )$ ) denote a synthetic pretraining task, and let Π denote the distribution over such tasks induced by the synthetic prior. Pretraining minimizes the expected query prediction loss as follows:

$$
\mathcal { L } _ { \mathrm { p r e } } ( \theta ) = \mathbb { E } _ { \mathcal { E } \sim \Pi } \left[ \frac { 1 } { n _ { q } } \sum _ { j = 1 } ^ { n _ { q } } \ell _ { \mathcal { E } } \left( [ f _ { \theta } ( X _ { Q } ; S ) ] _ { j } , y _ { j } ^ { q } \right) \right] ,\tag{2}
$$

![](images/6013a942431c074ac03438b94d981abcbf8ddaa59961c447e9180fc7b8596a2d.jpg)

(a) In-Context Diversity Prior.  
![](images/62cd54ed0e51333de5f86bf339b28d7c5bfa6503e43ea730fc7096b60ff05619.jpg)  
(b) Illustration of Taffy Architecture.  
Figure 2: The pretraining pipeline of TAFFY combines the in-context diversity prior with the taskconditioned looped transformer. (a) In-Context Diversity Prior proceeds through three steps: basetable generation, multiple-environment construction, and context assembly. (b) Column and row encoders produce representations for a looped Transformer, whose final update is modulated by a support-derived task gate before query prediction.

where $\ell \varepsilon$ is defined separately for classification and regression, following TabICLv2 (Qu et al., 2026). For classification, we use cross-entropy loss. For regression, we use pinball loss averaged over 999 predicted quantiles. At inference, point predictions are obtained by averaging these quantiles. Query labels supervise the pretraining objective but are excluded from the model inputs. The prior Π specifies the joint construction of support and query data.

## 4 TAFFY

## 4.1 OVERALL FRAMEWORK

TAFFY is designed to learn task-specific predictive relationships from a new support context. Figure 2 shows how the training-task construction and prediction network serve this goal. Specifically, the In-Context Diversity Prior defines the contextual prediction problems used during pretraining, including tasks that combine observations from multiple causally related environments. The Task-Conditioned Looped Transformer refines support-conditioned representations through a reusable update of Transformers, with later passes building on information integrated by earlier ones. The query-label objective in Eq. (2) trains the network to predict from each sampled task’s support data.

Given a labeled support dataset and query features, column and row encoders produce row representations that incorporate support-label information. The looped Transformer applies a shared ICL stack repeatedly to these representations. A separate branch summarizes support features and labels statistically and computes a scalar gate, which modulates the final recurrent update before query prediction. At downstream inference, only the prediction network and its task-conditioning branch are used; the synthetic generator and environment identifiers of synthetic data are not required.

## 4.2 IN-CONTEXT DIVERSITY PRIOR

Design principle. The synthetic prior guides how the model learns feature–label predictive relationships from context. Our In-Context Diversity Prior places data from multiple causally related environments within the same task. Starting from a shared causal prototype, we construct these environments through different types of distribution shifts and interventions. Because the environments share the same causal prototype but slightly different mechanisms, their joint context provides contrasting evidence about how feature–label relationships behave under changing conditions. In causal discovery, such mechanism diversity can help identify causal structures among variables that are difficult to identify from a single observational distribution (Zhang et al., 2015; Pearl, 2009; Huang et al., 2020a). We adapt this insight to synthetic pretraining by using multi-environment diversity to help the model learn predictive relationships from context, rather than to recover the underlying causal graph. Figure 2(a) summarizes the construction in three steps: base-table generation, multiple-environment construction, and context assembly, which are illustrated as follows.

Step 1: Base-table generation. We first sample a causal process (Hollmann et al., 2023) as the causal prototype that defines a joint distribution over features and labels, then draw a base table from this distribution, serving as the starting point for constructing the causally related environments.

Step 2: Multiple-environment construction. Starting from the shared causal prototype, we construct multiple environments through distribution shifts in data sampling and interventions on structural mechanisms. Let D denote the number of environments, with $( S _ { e } , Q _ { e } )$ denoting the support and query datasets for environment e. We construct datasets for different environments as follows.

Distribution shifts. Keeping the sampled causal graph and its structural mechanisms fixed, we first generate a fresh candidate table for each environment. We then construct environment-specific datasets through three sampling schemes (Zhang et al., 2015) corresponding to three types of distribution shifts. For label shift, we change $p ( y )$ relative to the base table while keeping $p ( x \mid y )$ unchanged. We do so by assigning different sampling probabilities to the labels and sampling data points based on their labels. For covariate shift, we change $p ( x )$ relative to the base table while keeping $p ( y \mid x )$ unchanged. We achieve it by assigning sampling probabilities based on feature values and sampling data points accordingly. For conditional $s h i f t .$ , we change $p ( x \mid y )$ relative to the base table while keeping $p ( y )$ unchanged. Specifically, we implement it by assigning featuredependent sampling probabilities separately within each label and sampling data points accordingly. Definitions of these shifts and implementation details are provided in Appendix $\bar { \mathbf { A } } . 2$

Interventions. Starting from the same sampled causal graph in step 1, we construct interventional environments by modifying the mechanisms of selected non-label variables (Pearl, 2009), without directly intervening on the label node. For a hard intervention, we replace each selected variable’s structural equation with an externally specified value that does not depend on its parents. This re moves its incoming edges, while its outgoing effects remain. For a soft intervention, we keep the causal graph fixed but replace the selected variable’s structural mechanism with an environmentspecific, potentially nonlinear mechanism over the same parents, thereby changing how the parents determine the variable without removing their influence (Brouillard et al., 2020). We then generate a new table from the intervened causal process, propagating intervention effects to descendant variables. Definitions and implementation details are provided in Appendix ${ \bf A } . 3 .$

Step 3: Context assembly. We combine environment-specific support and query data as follows:

$$
S = \bigcup _ { e = 1 } ^ { D } S _ { e } , \qquad Q = \bigcup _ { e = 1 } ^ { D } Q _ { e } ,\tag{3}
$$

where U denotes row-wise concatenation with repeated examples retained. The resulting S and $Q = ( { \bar { X } } _ { Q } , y _ { Q } )$ form the assembled support and query datasets. For multi-environment tasks, we first sample D (up to 4 distinct environments, including the unchanged base-table environment), then sample the changes used to construct the remaining environments from the candidate distribution shifts and interventions in Step 2. The model receives the labeled support dataset and query features, while query labels provide supervision through the pretraining objective. To retain exposure to single-environment tasks that may arise during downstream inference, we leave 50% of pretraining tasks as single-environment tasks and apply the multi-environment construction to the remainder.

It is noted that the environment indices are used only to assemble synthetic tasks and are not supplied to the model. The predictor learns from the pooled support and query data through query-label supervision, with no environment-prediction or causal-identification objective.

## 4.3 COLUMN AND ROW ENCODERS

We follow the column-then-row encoding design of TabICLv2 (Qu et al., 2026) to obtain fixeddimensional representations from tables with varying feature counts. Circular shifts form overlapping groups of three feature values, each projected to a 128-dimensional token by a shared linear layer. Support-label embeddings are added to the corresponding support tokens to incorporate feature–label associations. Three induced-attention blocks then use 128 inducing vectors to summarize support information for each feature group and propagate it to support and query tokens.

Next, the row encoder models interactions among the feature-group tokens within each example using three Transformer blocks and four learned [CLS] tokens. Concatenating the four [CLS] outputs yields a 512-dimensional row representation. A separate label embedding is then added to each support-row representation before the looped ICL learner. These augmented support representations and the query-row representations form the initial hidden states $h ^ { ( 0 ) }$ . Query labels are never used as inputs. Both encoders run once per forward pass; only the ICL learner is repeated.

## 4.4 TASK-CONDITIONED LOOPED TRANSFORMER

Design principle. The In-Context Diversity Prior exposes the model to varied predictive relationships by placing observations from multiple environments within the same context. These relationships differ in complexity: some can be identified from the initial context representations, whereas others benefit from further integration across examples and features. Looped computation with taskconditioned gating provides a way to accommodate this variation. Reapplying a shared Transformer allows each pass to build on contextual information integrated by earlier passes (Dehghani et al., 2019; Yang et al., 2024), while a task-specific gate regulates how much the final refinement contributes to prediction. The module therefore adapts the influence of iterative computation to the current task, while keeping the number of passes fixed and sharing parameters across passes.

Looped Transformer. After column and row encoding in Section 4.3, we obtain the initial representations $h ^ { ( 0 ) }$ . We then use a looped ICL Transformer to progressively integrate information across the labeled context. Let $F _ { \theta }$ denote the twelve-block ICL Transformer with hidden width 512. Following shared recurrent computation of Transformers (Dehghani et al., 2019; Yang et al., 2024; Balef et al., 2026), we apply the same stack to successive hidden states:

$$
h ^ { ( \ell ) } = F _ { \boldsymbol { \theta } } \big ( h ^ { ( \ell - 1 ) } ; M _ { S } \big ) , \qquad \ell = 1 , \dots , L ,\tag{4}
$$

where L is the number of passes and $M _ { S }$ denotes a support-only attention pattern. In each block, all $N = n _ { s } + n _ { q }$ positions serve as queries, whereas only the first $n _ { s }$ support positions serve as keys and values. We implement this pattern by placing support representations before query representations and slicing the first $n _ { s }$ hidden states for the key/value sequence. Thus, both support and query representations are updated, while query examples do not attend to one another.

Task-specific gate. The task-specific gate makes the final recurrent update adaptively depend on a statistical description of the current task. Specifically, we summarize the support dataset using statistics such as feature means, standard deviations, and class proportions, and combine them with input-shape information into a parameter-free descriptor $c _ { S } \in \mathbf { \mathbb { R } } ^ { 5 \hat { 1 } }$ to guide contextual refinement. The complete list of statistics is provided in Appendix B.1. Since query labels are unavailable in realworld tasks, we compute the label summaries solely from the labeled support dataset. For regression, the four class-specific entries are set to zero. To translate this task descriptor into a gate that regulates the final refinement, we use a scalar-output multilayer perceptron (MLP) parameterized by $W _ { 2 }$ to compute a bounded task-dependent correction to a global scalar a:

$$
b _ { S } = W _ { 1 } \big ( 2 \sigma ( \mathrm { M L P } _ { W _ { 2 } } ( c _ { S } ) ) - 1 \big ) , \qquad \alpha _ { S } = \operatorname { t a n h } ( a + b _ { S } ) ,\tag{5}
$$

where $\sigma$ is the logistic sigmoid, $W _ { 1 }$ is a scalar scaling coefficient, and $W _ { 2 }$ denotes the parameters of the MLP. The gate combines a global update scale with a task-specific correction.

Conditioned final update. After L passes, the looped Transformer produces hidden states ${ { h } ^ { ( 1 ) } } , \dots , { { h } ^ { ( L ) } }$ , each containing the support and query representations. We use the task-specific gate to scale the refinement from the final pass and add it to the preceding state:

$$
h ^ { \mathrm { o u t } } = h ^ { ( L - 1 ) } + \alpha _ { S } \big ( h ^ { ( L ) } - h ^ { ( L - 1 ) } \big ) .\tag{6}
$$

Here, $h ^ { ( L ) } - h ^ { ( L - 1 ) }$ is the update contributed by the final pass. Setting $\alpha _ { S } = 0$ retains the preceding state; a positive gate preserves a fraction of this update, while a negative gate reverses its direction. All L passes are executed, so task conditioning changes the final representation rather than the number of computation steps. Finally, the prediction module produces $\widehat { Y } _ { Q } = g _ { \theta } ( h _ { Q } ^ { \mathrm { o u t } } )$ , where $g _ { \theta }$ maps query representations to class probabilities or conditional quantile predictions for regression.

Table 1: Average ranks across all evaluated classification and regression datasets (lower is better). The classification row uses Taffy-4L on every suite; regression uses the separately trained model.
<table><tr><td rowspan="2">Model</td><td colspan="6">Classification</td><td colspan="5">Regression</td></tr><tr><td>BCCO</td><td>OpenML</td><td>PFN</td><td>TALENT</td><td>TabArena</td><td>TabZilla</td><td>BCCO</td><td>CTR23</td><td>PFN</td><td>TALENT</td><td>TabArena</td></tr><tr><td colspan="10">Ensemble methods</td></tr><tr><td>AutoGluon</td><td>10.62</td><td>8.29</td><td>9.88</td><td>9.96</td><td>9.98</td><td>8.52</td><td>5.96</td><td>5.85</td><td>5.96</td><td>5.89</td><td>6.15</td></tr><tr><td colspan="10">Tree-based methods</td></tr><tr><td>CatBoost</td><td>10.29</td><td>9.88</td><td>9.30</td><td>10.70</td><td>10.78</td><td>11.02</td><td>7.20</td><td>7.39</td><td>7.25</td><td>7.02</td><td>7.77</td></tr><tr><td>XGBoost</td><td>11.66</td><td>10.52</td><td>9.95</td><td>11.81</td><td>11.03</td><td>11.07</td><td>11.30</td><td>10.55</td><td>10.93</td><td>10.39</td><td>10.92</td></tr><tr><td>ET</td><td>12.23</td><td>11.55</td><td>10.96</td><td>13.34</td><td>14.77</td><td>11.83</td><td>10.15</td><td>10.03</td><td>10.29</td><td>9.76</td><td>9.54</td></tr><tr><td>RF</td><td>10.66</td><td>13.45</td><td>11.38</td><td>13.06</td><td>10.48</td><td>13.15</td><td>10.61</td><td>11.48</td><td>10.32</td><td>10.06</td><td>10.54</td></tr><tr><td colspan="10">Deep tabular methods</td></tr><tr><td>SwitchTab</td><td>16.73</td><td>16.46</td><td>16.16</td><td>17.45</td><td>18.98</td><td>16.00</td><td>16.85</td><td>16.36</td><td>16.39</td><td>16.47</td><td>17.23</td></tr><tr><td>T2G-Former</td><td>16.43</td><td>17.70</td><td>19.00</td><td>16.85</td><td>14.95</td><td>17.06</td><td>13.24</td><td>13.88</td><td>14.43</td><td>13.42</td><td>14.23</td></tr><tr><td>TANGOS</td><td>14.25</td><td>16.16</td><td>16.93</td><td>15.78</td><td>14.39</td><td>14.93</td><td>12.04</td><td>11.18</td><td>12.43</td><td>12.74</td><td>13.00</td></tr><tr><td>TabCaps</td><td>14.73</td><td>14.02</td><td>14.45</td><td>13.90</td><td>14.86</td><td>13.48</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TabM</td><td>12.20</td><td>11.68</td><td>12.30</td><td>11.56</td><td>13.00</td><td>11.37</td><td>16.11</td><td>15.52</td><td>15.18</td><td>15.55</td><td>17.08</td></tr><tr><td>TabNet</td><td>18.90</td><td>18.34</td><td>19.34</td><td>18.63</td><td>18.88</td><td>18.93</td><td>16.15</td><td>15.58</td><td>16.96</td><td>16.35</td><td>16.38</td></tr><tr><td>TabR</td><td>11.91</td><td>11.11</td><td>11.14</td><td>11.12</td><td>12.12</td><td>10.54</td><td>10.74</td><td>10.67</td><td>12.93</td><td>11.14</td><td>12.31</td></tr><tr><td>TabTransformer</td><td>16.60</td><td>13.67</td><td>13.34</td><td>15.66</td><td>16.14</td><td>14.09</td><td>18.52</td><td>17.55</td><td>16.57</td><td>17.47</td><td>17.92</td></tr><tr><td colspan="10">Tabular foundation models</td></tr><tr><td>TabICLv1</td><td>7.73</td><td>8.16</td><td>7.93</td><td>7.58</td><td>7.06</td><td>8.65</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LimiX-2M</td><td>7.02</td><td>6.56</td><td>6.14</td><td>6.85</td><td>5.72</td><td>6.20</td><td>9.76</td><td>8.88</td><td>7.75</td><td>9.17</td><td>7.15</td></tr><tr><td>TabPFN2</td><td>7.35</td><td>7.87</td><td>6.68</td><td>6.40</td><td>6.44</td><td>7.33</td><td>5.09</td><td>6.30</td><td>5.50</td><td>6.16</td><td>6.15</td></tr><tr><td>Mitra</td><td>9.72</td><td>12.60</td><td>12.20</td><td>11.34</td><td>10.31</td><td>12.09</td><td>9.30</td><td>10.39</td><td>8.93</td><td>10.35</td><td>9.92</td></tr><tr><td>LimiX-16M</td><td>6.15</td><td>6.02</td><td>7.00</td><td>6.06</td><td>6.03</td><td>6.67</td><td>8.09</td><td>7.55</td><td>7.57</td><td>8.00</td><td>6.46</td></tr><tr><td>TabICLv2</td><td>5.52</td><td>6.20</td><td>5.57</td><td>4.59</td><td>5.88</td><td>6.41</td><td>3.72</td><td>3.73</td><td>3.39</td><td>4.15</td><td>3.23</td></tr><tr><td>TabPFN3</td><td>5.83</td><td>6.10</td><td>6.46</td><td>4.62</td><td>4.73</td><td>6.48</td><td>3.11</td><td>4.18</td><td>3.93</td><td>3.48</td><td>2.69</td></tr><tr><td>TAFFY</td><td>4.47</td><td>4.67</td><td>4.89</td><td>3.73</td><td>4.45</td><td>5.19</td><td>2.07</td><td>2.94</td><td>3.29</td><td>2.45</td><td>1.31</td></tr></table>

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Benchmarks. We evaluate classification and regression on TALENT (Liu et al., 2025), BCCO (Zhang et al., 2025a), the TabPFN benchmark (Hollmann et al., 2025), and TabArena (Erickson et al., 2025), plus OpenML-CC18 (Bischl et al., 2021) and TabZilla (McElfresh et al., 2023) for classification and CTR23 (Fischer et al., 2023) for regression. Preprocessing is detailed in Appendix C.

Baselines and inference. Our baseline roster covers four groups: (i) tree-based methods, including CatBoost (Prokhorenkova et al., 2018), XGBoost (Chen & Guestrin, 2016), random forests (RF) (Breiman, 2001), and extremely randomized trees (ET) (Geurts et al., 2006); (ii) deep tabular and related predictors, including TabM (Gorishniy et al., 2025), TabR (Gorishniy et al., 2024), and SwitchTab (Wu et al., 2024); (iii) tabular foundation models, including TabPFN2 (Hollmann et al., 2025), TabPFN3 (Grinsztajn et al., 2026), TabICLv1/v2 (Qu et al., 2025; 2026), LimiX-2M/16M (Wang et al., 2026; Zhang et al., 2025a), and Mitra (Zhang et al., 2025b); and (iv) AutoGluon (Erickson et al., 2020). Please refer to Appendix C.1 for more details of the baselines.

Implementation and evaluation. We conduct pretraining on a cluster with 192 AMD MI210 GPUs. We train for 25,000 steps with the Muon optimizer. Each pretraining context contains up to four environments, with equal numbers of rows per environment subject to a fixed total row budget. We denote by Taffy-XL the model whose shared ICL stack is applied X times. For the primary compari son, classification uses Taffy-4L on all suites; regression uses a separately trained model. Pretraining Taffy-2L, Taffy-3L, and Taffy-4L takes approximately 2.54, 3.07, and 3.65 days, respectively, on 64 GPUs (Appendix C.4). Our primary metric is average rank within each benchmark: methods are ranked on every dataset using classification accuracy or regression RMSE, and the resulting dataset-level ranks are averaged. We further consider the Elo metric (Elo, 1978) in Appendix C.10.

## 5.2 OVERALL PERFORMANCE

Main Results. Following TabICLv2 (Qu et al., 2026), we exclude datasets overlapping with its development set from the primary evaluation and evaluate the 12 TALENT tasks with more than ten classes separately. For these many-class tasks, we use TabICL’s hierarchical classification (Qu et al., 2025) with mixed-radix ensembling; their rank and Elo results are reported in Appendix C.7. We further report rank and Elo under the LimiX-2M dataset-selection criteria (Wang et al., 2026) in Appendix C.8. Table 1 summarizes average ranks across six classification and five regression benchmarks. TAFFY obtains the lowest average rank on all six classification and five regression benchmarks. On TALENT, it obtains ranks of 3.73 for classification and 2.45 for regression, compared with 4.62 and 3.48 for TabPFN3 and 4.59 and 4.15 for TabICLv2.

![](images/e77f699879344ef89f8b611534b770625619aa764ba4e44266151884e68f80de.jpg)  
(a) In-context diversity

![](images/da0e092aa516abd06ad4ef8d6d9f5cc6e3fae5b01f5257b1017165b5271ef9bd.jpg)  
(b) Task-conditioned gate

![](images/07cefa4ee631091849ebc82b64a662cbdde8aa16b93685da69ef7080002d2e05.jpg)  
(c) Matched loops

![](images/d2ab40b915838466f446e2c144d9e10446f8f3090b30aaec7aa6db90caa4aed9.jpg)  
(d) Fixed training at 2  
Figure 3: Component analyses: (a) in-context versus cross-context diversity; (b) task-specific versus global gating; (c) matched training/inference depth; (d) extra inference loops after two-loop training. In (a,b), each variant is ranked by accuracy against four fixed baselines (TabICLv1, TabICLv2, LimiX-2M, and LimiX-16M) on each dataset, with average ranks for ties before averaging across datasets. In (c), ranks are computed on all datasets of each benchmark. Lower ranks are better; thin lines show checkpoints, thick lines show smoothed trends.

Ablation Study. To verify that each proposed component contributes to TAFFY, we compare three configurations in Figure 4: Baseline, which removes both the In-Context Diversity Prior and the Task-Conditioned Looped Transformer; + Diversity, which adds only the diversity prior; and Full Taffy, which includes both components and uses four loops during training and inference. Lower average rank indicates better performance. According to the ablation studies, adding in-context diversity improves the rank from 2.40 to 2.17, and incorporating the task-conditioned loop further reduces it to 2.07. This monotonic improvement shows that both additions contribute to the full model, while the final comparison evaluates the loop and task gate jointly rather than isolating them.

![](images/572b7f4ca05fe5b4616f6fae881790519010d575b38ede95865e07fa1253fef5.jpg)  
Figure 4: Average-rank ablation on TALENT classification datasets. Each variant is ranked separately against TabICLv1, TabICLv2, LimiX-2M, and LimiX-16M.

## 5.3 ANALYSIS OF CONTEXT CONSTRUCTION AND REFINEMENT

We further analyze context organization, task-specific gating, the loop depth during training and inference, and investigate which information the task-specific gate can capture.

In-Context Diversity vs. Cross-Context Diversity. To assess whether multiple environments are more useful when presented jointly within a prediction task, we compare in-context diversity with cross-context diversity. The cross-context variant retains environments generated through distribution shifts and interventions but shuffles the environment-specific data across training tasks, ensuring that each task contains data from only one environment. The in-context variant instead combines multiple related environments within half of its training tasks. Both variants use the same singleloop backbone and are evaluated every 50 steps on the TALENT classification datasets. Figure 3(a) shows the smoothed mean-rank trajectories. In-context diversity attains a lower mean rank at step 25,000, with its advantage becoming increasingly pronounced as training progresses. These results show that environmental diversity is more effective within prediction tasks than across training tasks.

Task-Specific vs. Global Gating. To evaluate the contribution of task-specific statistics, we compare a task-specific gate with a global gate. Specifically, both variants are trained and evaluated with two loops on the TALENT classification datasets; the global-gate configuration retains the same model parameterization but does not receive task-specific support statistics, making α invariant across datasets. Figure 3(b) shows that the task-specific gate achieves a consistently lower mean rank than the global gate toward the end of training, with the gap becoming more pronounced as training progresses. This result demonstrates that support-derived task statistics provide useful signals for adapting the strength of contextual refinement across datasets. Appendix C.12 provides a frozen-checkpoint intervention against a support-independent coefficient and α = 1.

Further Analysis of the Task-Conditioned Looped Transformer. We first vary the number of loops jointly during training and inference. Figure 3(c) shows that Taffy-4L obtains the lowest average ranks on TALENT and OpenML, whereas Taffy-3L performs best on PFN and TabArena. Increasing the matched depth from two to four loops improves the rank from 4.70 to 3.95 on TALENT and from 6.18 to 5.62 on OpenML, while PFN and TabArena favor three loops. Thus, additional trained loops can improve prediction, although the preferred depth varies across benchmark suites.

We next fix training at two loops and vary only the number of inference loops. Figure 3(d) shows that a third inference loop improves the OpenML rank from 4.22 to 3.97 and the PFN rank from 4.20 to 4.09, but does not improve TALENT or TabArena. Across the 428 benchmark entries shared by all three inference configurations, two and three loops both obtain aggregate ranks that round to 4.00, whereas four loops worsen the rank to 4.23. Additional inference-time repetition therefore does not reliably reproduce the benefit of training with greater loop depth.

What Does the Task-Specific Gate Capture? To understand what the task-specific gate captures, we compute correlations between α and task properties, including feature composition, numerical distribution shape, label structure, and missingness, across TALENT tasks.

Feature composition. Across all three depths, a higher fraction of categorical input features is associated with lower α (Table 2), controlling for support rows, effective feature columns, class count, and support-label entropy. The relationship remains consistent across preprocessing views and within datasets containing both numerical and categorical features.

Table 2: Spearman correlations between α and task attributes. <sup>∗</sup> indicates corrected $q < 0 . 0 5 .$
<table><tr><td>Property</td><td>2L</td><td>3L</td><td>4L</td></tr><tr><td>Categorical fraction</td><td> $- . 5 3 3 ^ { * }$ </td><td> $- . 4 5 3 ^ { * }$ </td><td> $- . 4 5 2 ^ { * }$ </td></tr><tr><td>Excess kurtosis</td><td> $+ . 2 3 0 ^ { * }$ </td><td> $+ . 3 2 6 ^ { * }$ </td><td> $+ . 3 2 3 ^ { * }$ </td></tr><tr><td>Label entropy</td><td>+.084</td><td> $+ . 0 1 0$ </td><td> $- . 0 1 1$ </td></tr><tr><td>Missing ratio</td><td>+.021</td><td> $+ . 0 4 2$ </td><td>+.031</td></tr></table>

Numerical distributions. Controlling also for categorical fraction, median excess kurtosis across numerical features correlates positively with α. Both survive multiple-testing correction, suggesting the gate responds to feature composition and numerical distribution shape.

Uncaptured task properties. The support-label entropy and the raw weighted missing-value ratio show no stable association with α. Specifically, the correlations with support-label entropy remain close to zero across all three depths. A more complex conditioning architecture is needed to exploit such dataset-specific information more fully, which is an interesting future direction.

## 5.4 PRETRAINING DATA EFFICIENCY

Figure 1(d) compares pretraining data efficiency on TALENT classification datasets. The horizontal axis measures cumulative maximum feature elements, $\begin{array} { r } { B ( s ) = \sum _ { t = 1 } ^ { s } B _ { t } R _ { t } ^ { \operatorname* { m a x } } d _ { t } ^ { \operatorname* { m a x } } } \end{array}$ , where $B _ { t }$ is the global task batch size, $R _ { t } ^ { \mathrm { m a x } }$ is the maximum number of support and query rows per task, and $d _ { t } ^ { \operatorname* { m a x } }$ is the maximum number of feature columns, excluding labels. Budgets accumulate across pretraining stages without multiplying by loop count (Appendix C.9). These results reveal two patterns. First, over the shared budget range, Taffy-3L achieves data efficiency comparable to TabICLv2 and continues to improve after TabICLv2 reaches its best result, while Taffy-4L attains higher accuracy at comparable budgets. This suggests that Taffy can extract additional predictive gains from more pretraining data after the baseline has largely saturated. Second, Taffy-4L > Taffy-3L > Taffy-2L holds at checkpoints at the same training steps. This consistent ordering shows that increasing the trained loop depth improves the predictive performance obtained from a fixed data budget.

## 6 CONCLUSION

We presented TAFFY to improve how tabular foundation models learn and refine predictive relationships from context. Rather than treating task construction and context processing separately, TAFFY coordinates them: the In-Context Diversity Prior places related environments within a pretraining task to expose informative variation, while the Task-Conditioned Looped Transformer repeatedly integrates this evidence and uses support statistics to regulate the final refinement. Controlled comparisons favor in-context diversity, task-specific gating, and greater trained loop depth. Across six classification and five regression suites, the reported classification and regression configurations attain the lowest average rank on all eleven benchmarks under the stated protocols. Overall, the results identify context composition and task-adaptive iterative refinement as complementary design axes for tabular foundation models.

## REFERENCES

Sercan Ö. Arik and Tomas Pfister. TabNet: Attentive interpretable tabular learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 6679–6687, 2021. doi: 10. 1609/aaai.v35i8.16826. URL https://ojs.aaai.org/index.php/AAAI/article/ view/16826.

Sangmin Bae, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Seungyeon Kim, and Tal Schuster. Relaxed recursive transformers: Effective parameter sharing with layer-wise LoRA. In International Conference on Learning Representations, pp. 34282–34327, 2025a. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 54d6a55225cebbdc16fbb0e45c5bdf2b-Abstract-Conference.html.

Sangmin Bae, Yujin Kim, Reza Bayat, Sungnyun Kim, Jiyoun Ha, Tal Schuster, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Aaron C. Courville, and Se-Young Yun. Mixture-of-recursions: Learning dynamic recursive depths for adaptive token-level computation. In Advances in Neural Information Processing Systems, volume 38, pp. 96572–96617, 2025b. doi: 10.52202/ 085713-3229. URL https://papers.nips.cc/paper\_files/paper/2025/hash/ 8b08bbf8b420faa6eeb4020720582ec7-Abstract-Conference.html.

Amir Rezaei Balef, Mykhailo Koshil, and Katharina Eggensperger. Is one layer enough? understanding inference dynamics in tabular foundation models. In Proceedings of the 43rd International Conference on Machine Learning, 2026. doi: 10.48550/arXiv.2605.06510. URL https://arxiv.org/abs/2605.06510.

Andrea Banino, Jan Balaguer, and Charles Blundell. PonderNet: Learning to ponder. In 8th ICML Workshop on Automated Machine Learning, 2021. URL https://openreview.net/ forum?id=1EuxRTe0WN.

Bernd Bischl, Giuseppe Casalicchio, Matthias Feurer, Pieter Gijsbers, Frank Hutter, Michel Lang, Rafael Gomes Mantovani, Jan van Rijn, and Joaquin Vanschoren. OpenML benchmarking suites. In Proceedings of the Neural Information Process ing Systems Track on Datasets and Benchmarks, volume 1, 2021. URL https: //datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/ c7e1249ffc03eb9ded908c236bd1996d-Abstract-round2.html.

Mohamed Bouadi, Nassim Bouarour, Varun Kulkarni, Shivam Dubey, Aditya Tanna, and Vinay Kumar Sankarapu. Shaping the prior: How synthetic task distributions determine tabular foundation model quality, 2026. URL https://arxiv.org/abs/2605.18971.

Leo Breiman. Random forests. Machine Learning, 45:5–32, 2001. doi: 10.1023/A:1010933404324. URL https://doi.org/10.1023/A:1010933404324.

Philippe Brouillard, Sébastien Lachapelle, Alexandre Lacoste, Simon Lacoste-Julien, and Alexandre Drouin. Differentiable causal discovery from interventional data. In Advances in Neural Information Processing Systems, volume 33, pp. 21865–21877, 2020.

Jintai Chen, Kuanlun Liao, Yanwen Fang, Danny Z. Chen, and Jian Wu. TabCaps: A capsule neural network for tabular data classification with BoW routing. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=OgbtSLESnI.

Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794, 2016. doi: 10.1145/2939672.2939785. URL https://doi.org/10.1145/ 2939672.2939785.

Yilong Chen, Junyuan Shang, Zhenyu Zhang, Yanxi Xie, Jiawei Sheng, Tingwen Liu, Shuohuan Wang, Yu Sun, Hua Wu, and Haifeng Wang. Inner thinking transformer: Leveraging dynamic depth scaling to foster adaptive internal thinking. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 28241–28259, 2025.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. In International Conference on Learning Representations, 2019.

Felix den Breejen, Sangmin Bae, Stephen Cha, and Se-Young Yun. Fine-tuned in-context learning transformers are excellent tabular data classifiers, 2024. URL https://arxiv.org/abs/ 2405.13396.

Arpad E. Elo. The Rating of Chessplayers, Past and Present. Arco Publishing, 1978.

Nick Erickson, Jonas Mueller, Alexander Shirkov, Hang Zhang, Pedro Larroy, Mu Li, and Alexander Smola. AutoGluon-Tabular: Robust and accurate AutoML for structured data. arXiv preprint arXiv:2003.06505, 2020. URL https://arxiv.org/abs/2003.06505.

Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Desai, David Salinas, and Frank Hutter. TabArena: A living benchmark for machine learning on tabular data. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-0519. URL https://proceedings.neurips.cc/paper\_files/paper/2025/ hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets\_and\_ Benchmarks\_Track.html.

Sebastian Felix Fischer, Liana Harutyunyan, Matthias Feurer, and Bernd Bischl. OpenML-CTR23: A curated tabular regression benchmarking suite. In AutoML Conference 2023 (Workshop), 2023. URL https://openreview.net/pdf?id=HebAOoMm94.

Anurag Garg, Muhammad Ali, Noah Hollmann, Lennart Purucker, Samuel Müller, and Frank Hutter. Real-TabPFN: Improving tabular foundation models via continued pre-training with real-world data. arXiv preprint arXiv:2507.03971, 2025.

Khashayar Gatmiry, Nikunj Saunshi, Sashank J. Reddi, Stefanie Jegelka, and Sanjiv Kumar. Can looped transformers learn to implement multi-step gradient descent for in-context learning? In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 15130–15152. PMLR, 2024. URL https://proceedings.mlr.press/v235/gatmiry24b.html.

Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. In Advances in Neural Information Processing Systems, volume 38, pp. 41340–41391, 2025. doi: 10.52202/085713-1380. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 3b01972cf31e6fa0fe29e4b8b5c2a0a1-Abstract-Conference.html.

Pierre Geurts, Damien Ernst, and Louis Wehenkel. Extremely randomized trees. Machine Learning, 63:3–42, 2006. doi: 10.1007/s10994-006-6226-1. URL https://doi.org/10.1007/ s10994-006-6226-1.

Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov, and Artem Babenko. TabR: Tabular deep learning meets nearest neighbors. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ 4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html.

Yury Gorishniy, Akim Kotelnikov, and Artem Babenko. TabM: Advancing tabular deep learning with parameter-efficient ensembling. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html.

Alex Graves. Adaptive computation time for recurrent neural networks. arXiv preprint arXiv:1603.08983, 2016.

Léo Grinsztajn, Klemens Flöge, Oscar Key, Felix Birkel, Philipp Jund, Brendan Roof, Benjamin Jäger, Dominik Safaric, Simone Alessi, Adrian Hayler, Mihir Manium, Rosen Yu, Felix Jablonski, Shi Bin Hoo, Anurag Garg, Jake Robertson, Magnus Bühler, Vladyslav Moroshan, Lennart Purucker, Clara Cornu, Lilly Charlotte Wehrhahn, Alessandro Bonetto, Bernhard Schölkopf, Sauraj Gambhir, Noah Hollmann, and Frank Hutter. TabPFN-2.5: Advancing the state of the art in tabular foundation models, 2025. URL https://arxiv.org/abs/2511.08667.

Léo Grinsztajn, Klemens Flöge, Oscar Key, Felix Birkel, Philipp Jund, Brendan Roof, Mihir Manium, Shi Bin Hoo, Magnus Bühler, Anurag Garg, Dominik Safaric, Jake Robertson, Benjamin Jäger, Simone Alessi, Adrian Hayler, Vladyslav Moroshan, Lennart Purucker, Philipp Singer, Alan Arazi, Julien Siems, Jan Hendrik Metzen, Georg Grab, Nick Erickson, Siyuan Guo, Eliott Kalfon, Simon Bing, David Salinas, Clara Cornu, Lilly Charlotte Wehrhahn, Diana Kriuchkova, Kursat Kaya, Lydia Sidhoum, Marie Salmon, Jerry Chen, Madelon Hulsebos, Yann LeCun, Samuel Müller, Bernhard Schölkopf, Sauraj Gambhir, Noah Hollmann, and Frank Hutter. TabPFN-3: Technical report, 2026. URL https://arxiv.org/abs/2605.13986.

Kai Helli, David Schnurr, Noah Hollmann, Samuel Müller, and Frank Hutter. Drift-resilient TabPFN: In-context learning temporal distribution shifts on tabular data. In Advances in Neural Information Processing Systems, volume 37, 2024.

Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. TabPFN: A transformer that solves small tabular classification problems in a second. In International Conference on Learning Representations, 2023.

Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 637(8045):319–326, 2025. doi: 10.1038/s41586-024-08328-6.

Biwei Huang, Kun Zhang, Jiji Zhang, Joseph Ramsey, Ruben Sanchez-Romero, Clark Glymour, and Bernhard Schölkopf. Causal discovery from heterogeneous/nonstationary data. Journal of Machine Learning Research, 21(89):1–53, 2020a. URL https://www.jmlr.org/papers/ v21/19-232.html.

Xin Huang, Ashish Khetan, Milan Cvitkovic, and Zohar Karnin. TabTransformer: Tabular data modeling using contextual embeddings. arXiv preprint arXiv:2012.06678, 2020b. URL https: //arxiv.org/abs/2012.06678.

Aapo Hyvärinen, Hiroaki Sasaki, and Richard E. Turner. Nonlinear ICA using auxiliary variables and generalized contrastive learning. In Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pp. 859–868. PMLR, 2019. URL https://proceedings.mlr.press/v89/ hyvarinen19a.html.

Alan Jeffares, Tennison Liu, Jonathan Crabbé, Fergus Imrie, and Mihaela van der Schaar. TANGOS: Regularizing tabular neural networks through gradient orthogonalization and specialization. In International Conference on Learning Representations, 2023. URL https://openreview. net/forum?id=n6H86gW8u0d.

Ilyes Khemakhem, Diederik P. Kingma, Ricardo Pio Monti, and Aapo Hyvärinen. Variational autoencoders and nonlinear ICA: A unifying framework. In Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings of Machine Learning Research, pp. 2207–2217. PMLR, 2020. URL https://proceedings. mlr.press/v108/khemakhem20a.html.

Mykhailo Koshil, Matthias Feurer, and Katharina Eggensperger. In-context learning of soft nearest neighbor classifiers for intelligible tabular machine learning. In Proceedings of the 4th Table Representation Learning Workshop, pp. 182–191, 2025.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. ALBERT: A lite BERT for self-supervised learning of language representations. In International Conference on Learning Representations, 2020. URL https://openreview.net/ forum?id=H1eA7AEtvS.

Si-Yang Liu, Hao-Run Cai, Qi-Le Zhou, Huai-Hong Yin, Tao Zhou, Jun-Peng Jiang, and Han-Jia Ye. TALENT: A tabular analytics and learning toolbox. Journal of Machine Learning Research, 26(226):1–16, 2025. URL http://jmlr.org/papers/v26/25-0512.html.

Junwei Ma, Valentin Thomas, Rasa Hosseinzadeh, Alex Labach, Jesse Cresswell, Keyvan Golestan, Guangwei Yu, Anthony L Caterini, and Maks Volkovs. TabDPT: Scaling tabular

foundation models on real data. In Advances in Neural Information Processing Systems, volume 38, pp. 172692–172722. Curran Associates, Inc., 2025. doi: 10.52202/085713-5748. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html.

Duncan McElfresh, Sujay Khandagale, Jonathan Valverde, Vishak Prasad C, Ganesh Ramakrishnan, Micah Goldblum, and Colin White. When do neural nets outperform boosted trees on tabular data? In Advances in Neural Information Processing Systems, volume 36, 2023. doi: 10.52202/ 075280-3337. URL https://proceedings.neurips.cc/paper\_files/paper/ 2023/hash/f06d5ebd4ff40b40dd97e30cee632123-Abstract-Datasets\_ and\_Benchmarks.html.

Ratmir Miftachov, Bruno Charron, and Simon Valentin. Interpretable tabular foundation models via in-context kernel regression. arXiv preprint arXiv:2602.02162, 2026.

Amirkeivan Mohtashami, Matteo Pagliardini, and Martin Jaggi. CoTFormer: A chain of thought driven architecture with budget-adaptive computation cost at inference. In International Conference on Learning Representations, pp. 11503–11520, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 1eaa5146756be028ad6fff1efcc8e6bd-Abstract-Conference.html.

Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. Transformers can do Bayesian inference. In International Conference on Learning Representations, 2022.

Judea Pearl. Causality: Models, Reasoning, and Inference. Cambridge University Press, 2 edition, 2009. doi: 10.1017/CBO9780511803161. URL https://www.cambridge.org/core/ books/causality/B0046844FAE10CBF274D4ACBDAEB5F5B.

Jonas Peters, Peter Bühlmann, and Nicolai Meinshausen. Causal inference by using invariant prediction: Identification and confidence intervals. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 78(5):947–1012, 2016. doi: 10.1111/rssb.12167. URL https://doi.org/10.1111/rssb.12167.

Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin. CatBoost: Unbiased boosting with categorical features. In Advances in Neural Information Processing Systems, volume 31, 2018. URL https://proceedings.neurips.cc/ paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html.

Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. TabICL: A tabular foundation model for in-context learning on large data. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 50817–50847. PMLR, 2025.

Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. TabICLv2: A better, faster, scalable, and open tabular foundation model. In Proceedings of the 43rd International Conference on Machine Learning, 2026. URL https://arxiv.org/abs/2602.11139.

David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro. Mixture-of-depths: Dynamically allocating compute in transformer-based language models, 2024. URL https://arxiv.org/abs/2404.02258.

Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J. Reddi. Reasoning with latent thoughts: On the power of looped transformers. In International Conference on Learning Representations, pp. 14855–14881, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 2676109d49d1eb26d6bc584a8f556305-Abstract-Conference.html.

Bernhard Schölkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, and Yoshua Bengio. Toward causal representation learning. Proceedings of the IEEE, 109(5):612–634, 2021.

Burak Varici, Emre Acartürk, Karthikeyan Shanmugam, and Ali Tajer. General identifiability and achievability for causal representation learning. In Proceedings of the 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pp. 2314–2322. PMLR, 2024. URL https://proceedings.mlr.press/ v238/varici24a.html.

Julius von Kügelgen, Yash Sharma, Luigi Gresele, Wieland Brendel, Bernhard Schölkopf, Michel Besserve, and Francesco Locatello. Self-supervised learning with data augmentations provably isolates content from style. In Advances in Neural Information Processing Systems, volume 34, pp. 16451–16467. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 8929c70f8d710e412d38da624b21c3c8-Abstract.html.

Yuanrui Wang, Xingxuan Zhang, Han Yu, Mingchao Hao, Gang Ren, Hao Yuan, Li Mao, Yunjia Zhang, Chun Yuan, and Peng Cui. LimiX-2M: Mitigating low-rank collapse and attention bottlenecks in tabular foundation models. In International Conference on Machine Learning, 2026. doi: 10.48550/arXiv.2606.04485. URL https://arxiv.org/abs/2606.04485.

Jing Wu, Suiyao Chen, Qi Zhao, Renat Sergazinov, Chen Li, Shengjie Liu, Chongchao Zhao, Tianpei Xie, Hanqing Guo, Cheng Ji, Daniel Cociorva, and Hakan Brunzell. SwitchTab: Switched autoencoders are effective tabular learners. In Proceedings of the AAAI Conference on Artifi cial Intelligence, volume 38, pp. 15924–15933, 2024. doi: 10.1609/aaai.v38i14.29523. URL https://ojs.aaai.org/index.php/AAAI/article/view/29523.

Jiahuan Yan, Jintai Chen, Yixuan Wu, Danny Z. Chen, and Jian Wu. T2G-Former: Organizing tabular features into relation graphs promotes heterogeneous feature interaction. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 10720–10728, 2023. URL https://ojs.aaai.org/index.php/AAAI/article/view/26272.

Liu Yang, Kangwook Lee, Robert Nowak, and Dimitris Papailiopoulos. Looped transformers are better at learning learning algorithms. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ b8402301e7f06bdc97a31bfaa653dc32-Abstract-Conference.html.

Kun Zhang, Mingming Gong, and Bernhard Schölkopf. Multi-source domain adaptation: A causal view. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, pp. 3150– 3157. AAAI Press, 2015. doi: 10.1609/aaai.v29i1.9542. URL https://ojs.aaai.org/ index.php/AAAI/article/view/9542.

Xingxuan Zhang, Gang Ren, Han Yu, Hao Yuan, Hui Wang, Jiansheng Li, Jiayun Wu, Lang Mo, Li Mao, Mingchao Hao, Ningbo Dai, Renzhe Xu, Shuyang Li, Tianyang Zhang, Yue He, Yuanrui Wang, Yunjia Zhang, Zijing Xu, Dongzhe Li, Fang Gao, Hao Zou, Jiandong Liu, Jiashuo Liu, Jiawei Xu, Kaijie Cheng, Kehan Li, Linjun Zhou, Qing Li, Shaohua Fan, Xiaoyu Lin, Xinyan Han, Xuanyue Li, Yan Lu, Yuan Xue, Yuanyuan Jiang, Zimu Wang, Zhenlei Wang, and Peng Cui. LimiX: Unleashing structured-data modeling capability for generalist intelligence. arXiv preprint arXiv:2509.03505, 2025a. doi: 10.48550/arXiv.2509.03505. URL https://arxiv. org/abs/2509.03505.

Xiyuan Zhang, Danielle Maddix Robinson, Junming Yin, Nick Erickson, Abdul Fatir Ansari, Boran Han, Shuai Zhang, Leman Akoglu, Christos Faloutsos, Michael W. Mahoney, Tony Hu, Huzefa Rangwala, George Karypis, and Yuyang (Bernie) Wang. Mitra: Mixed synthetic priors for enhancing tabular foundation models. In Advances in Neural Information Processing Systems, volume 38, 2025b. doi: 10.52202/085713-0535. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 177d68f4adef163b7b123b5c5adb3c60-Abstract-Conference.html.

## A SYNTHETIC TASK GENERATION

This section details the In-Context Diversity Prior of Section 4.2. All environments of a multienvironment task are derived from one base structural causal model. Three distribution shifts change how rows are sampled from this model while keeping its graph and mechanisms fixed; two interventions change the model itself before data are generated. Table 3 summarizes the five operators. Appendix A.4 describes how the resulting environments are assembled into a single support–query task.

Table 3: Environment operators of the In-Context Diversity Prior. The three distribution shifts reweight rows sampled from the unchanged base model, with strength fixed by $\mathrm { K L } ( p _ { e } \| p _ { 0 } ) = 0 . 1$ The two interventions modify the mechanisms of 5% of the non-label nodes and regenerate the data.
<table><tr><td>Operator</td><td>Implementation</td><td>Preserved</td><td>Changed</td><td>Strength</td></tr><tr><td colspan="5">Distribution shifts (row sampling from  $\mathcal { M } _ { \mathrm { 0 } } )$ </td></tr><tr><td>Label shift</td><td>Reweight rows by  $a _ { e } ( y ) \overset { \cdot } { = } \exp \bigl ( \tau _ { e } \overset { \cdot } { u } _ { e } ( y ) \bigr )$ </td><td> $p ( x \mid y )$ </td><td> $p ( y )$  , hence  $p ( y \mid x )$ </td><td> $\mathrm { K L } = 0 . 1$ </td></tr><tr><td>Covariate shift</td><td>Reweight rows by  $b _ { e } ( x ) \stackrel { - } { = } \mathrm { e x p } \big ( - \stackrel { \cdot } { \tau _ { e } } \| x - \mu _ { e } \| _ { 2 } \big )$ </td><td> $p ( y \mid x )$ </td><td> $p ( x )$ </td><td> $\mathrm { K L } = 0 . 1$ </td></tr><tr><td></td><td>Conditional shift Reweight rows within each class by  $c _ { e } ( x , y ) = \exp \big ( - \tau _ { e } \| x - \mu _ { e , y } \| _ { 2 } \big )$ </td><td> $p ( y )$ </td><td> $p ( x \mid y ) ,$  hence  $p ( y \mid x )$ </td><td> $\mathrm { K L } = 0 . 1$ </td></tr><tr><td colspan="5">Interventions (mechanism changes in  $\mathcal { M } _ { \mathrm { 0 } } )$ </td></tr><tr><td>Hard intervention Set</td><td> $z _ { j } : = v _ { e , j }$  with  $v _ { e , j } \sim \mathcal { U } [ \ell _ { j } , u _ { j } ]$   $j \in I _ { e }$ </td><td>for Other mechanisms</td><td> $z _ { j }$  and its descendants</td><td>5% of non-label nodes</td></tr><tr><td>Soft intervention Replace</td><td> $g _ { j }$  by  $g _ { e , j }$  from the same function family with resampled parameters, for  $j \in I _ { e }$ </td><td>Graph G, other mechanisms</td><td>Mechanisms in  $I _ { e }$  and their descendants</td><td>5% of non-label nodes</td></tr></table>

## A.1 BASE CAUSAL PROTOTYPE

Let $\mathcal { M } _ { 0 }$ denote a structural causal model sampled from the causal prior of Hollmann et al. (2023), with an acyclic graph G over nodes $V = \{ 1 , \ldots , m \}$ and structural equations

$$
z _ { j } = g _ { j } \left( z _ { \mathrm { p a } _ { G } ( j ) } , \epsilon _ { j } \right) , \qquad j = 1 , \ldots , m ,\tag{7}
$$

where $\epsilon = ( \epsilon _ { 1 } , \dots , \epsilon _ { m } )$ follows the sampled exogenous-noise distribution. A readout selects the feature nodes and the label node $y \in V ;$ ; nodes that are not selected remain unobserved. Evaluating Equation (7) in topological order and applying the readout induces the base joint distribution $p _ { 0 } ( x , y )$ , from which the base table is drawn. We use $p$ for a probability density or mass function, as appropriate. All environments of a task share $G ,$ the node ordering, and the readout; they differ only in their sampling distributions or in selected mechanisms.

## A.2 DISTRIBUTION SHIFTS

Candidate pool and weights. For a distribution-shift environment e, we draw a fresh candidate pool $\mathcal { C } _ { e } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ from $\mathcal { M } _ { 0 }$ without modifying its graph, mechanisms, noise distribution, or readout. The empirical base distribution $\hat { p } _ { 0 }$ places mass $1 / n$ on each row. For label and covariate shift, each row receives a normalized sampling weight of the exponential-tilting form

$$
w _ { e , i } ( \tau ) = \frac { \exp \bigl ( \tau s _ { e } ( x _ { i } , y _ { i } ) \bigr ) } { \sum _ { k = 1 } ^ { n } \exp \bigl ( \tau s _ { e } ( x _ { k } , y _ { k } ) \bigr ) } , \qquad \tau \geq 0 ,\tag{8}
$$

where the score $s _ { e }$ depends on the shift type and is specified below. Conditional shift applies the same form separately within each class (Appendix A.2.3). The rows of environment e are then drawn with replacement from $\mathcal { C } _ { e }$ according to $w _ { e } ( \tau _ { e } )$ , which defines the shifted distribution $p _ { e }$ Setting $\tau = 0$ gives uniform weights and recovers the base distribution.

Shift strength. We control the strength of each distribution shift through its Kullback–Leibler divergence from the base distribution. For a reweighted distribution $p _ { e } = r _ { e } p _ { 0 }$ with density ratio $r _ { e } , \bar { \mathrm { K L } } ( p _ { e } \| p _ { 0 } ) = \mathbb { E } _ { p _ { e } } [ \log r _ { e } ]$ . On the candidate pool this becomes

$$
\widehat { \mathrm { K L } } _ { e } ( \tau ) = \sum _ { i = 1 } ^ { n } w _ { e , i } ( \tau ) \log \bigl ( n w _ { e , i } ( \tau ) \bigr ) ,\tag{9}
$$

computed with the natural logarithm. We set a common target $\kappa = 0 . 1$ nats for all three shift types and choose $\tau _ { e }$ by bisection so that $\widehat { \mathrm { K L } } _ { e } ( \tau _ { e } ) = \kappa$ up to a numerical tolerance. The bisection is well posed because $\widehat { \mathrm { K L } } _ { e } ( 0 ) = 0 , \widehat { \mathrm { K L } } _ { e }$ is continuous, and its derivative

$$
\frac { \mathrm { d } } { \mathrm { d } \tau } \widehat { \mathrm { K L } } _ { e } ( \tau ) = \tau \mathrm { V a r } _ { w _ { e } ( \tau ) } \big [ s _ { e } ( X , Y ) \big ] \geq 0\tag{10}
$$

makes it nondecreasing in $\tau .$ . For conditional shift, the same argument applies within each class, so the class-averaged divergence is also nondecreasing. $\mathrm { A s } ~ \tau \to \infty , \widehat { \mathrm { K L } } _ { e } ( \tau )$ approaches $\log ( n / n _ { e } ^ { \star } )$ where $n _ { e } ^ { \star }$ is the number of rows attaining the maximal score, so the target is attainable whenever $\log ( n / n _ { e } ^ { \star } ) > \kappa$ . If the target is not attainable, we discard the environment and sample a new one.

Because each shift multiplies $p _ { 0 }$ by a factor that depends only on the modified component, the joint divergence equals the divergence of that component. For label shift, $\mathrm { K L } ( p _ { e } \| { \bf \bar { \it p } } _ { 0 } ) =$ $\mathrm { K L } \big ( p _ { e } ( y ) | | p _ { 0 } ( y ) \big )$ ; for covariate shift, $\mathrm { K L } ( p _ { e } \parallel p _ { 0 } ) = \mathrm { K L } \bigl ( p _ { e } ( x ) \parallel p _ { 0 } ( x ) \bigr )$ ; and for conditional shift, $\mathrm { K L } ( p _ { e } \parallel p _ { 0 } ) = \mathbb { E } _ { p _ { 0 } ( y ) } \bigl [ \mathrm { K L } \bigl ( p _ { e } ( x \mid y ) \parallel p _ { 0 } ( x \mid y ) \bigr ) \bigr ]$ . A common κ therefore gives the three shift types comparable magnitudes, measured on the component each one changes. The value $\kappa = 0 . 1$ corresponds to a mild shift; for example, it moves a balanced binary label distribution from $5 0 / 5 0$ to approximately $7 2 / 2 8$

## A.2.1 LABEL SHIFT

Label shift uses a score that depends only on the label, $s _ { e } ( x , y ) = u _ { e } ( y )$ , where $u _ { e }$ assigns each class k an independent score $u _ { e , k }$ drawn from a uniform distribution. Because $\tau _ { e }$ is calibrated to the KL target, the range of this uniform distribution does not affect the resulting weights. For regression, the continuous label is first discretized into 10 bins, each bin is treated as a class, and label shift and conditional shift are then constructed exactly as for classification. The resulting label weight $a _ { e } ( y ) = \exp \bigl ( \tau _ { e } u _ { e } ( y ) \bigr )$ defines

$$
p _ { e } ^ { \mathrm { l a b } } ( x , y ) = \frac { a _ { e } ( y ) p _ { 0 } ( x , y ) } { Z _ { e } ^ { \mathrm { l a b } } } , \qquad p _ { e } ^ { \mathrm { l a b } } ( y ) = \frac { a _ { e } ( y ) p _ { 0 } ( y ) } { Z _ { e } ^ { \mathrm { l a b } } } , \qquad Z _ { e } ^ { \mathrm { l a b } } = \mathbb { E } _ { p _ { 0 } } [ a _ { e } ( Y ) ] .\tag{11}
$$

Since the weight depends only on the label, $p _ { e } ^ { \mathrm { l a b } } ( x \mid y ) = p _ { 0 } ( x \mid y )$ wherever $p _ { e } ^ { \mathrm { l a b } } ( y ) > 0$ . The class proportions change, and $p ( y \mid x ) \propto a _ { e } ( y ) p _ { 0 } ( y \mid x )$ changes accordingly. Classes with larger $u _ { e } ( y )$ are over-represented, and $\tau _ { e }$ is calibrated so that KL $\big ( p _ { e } ^ { \mathrm { l a b } } ( y ) \| \hat { p } _ { 0 } ( y ) \big ) \bumpeq 0 . 1$

## A.2.2 COVARIATE SHIFT

Covariate shift uses a score that depends only on the features. Features are standardized before distances are computed. We draw a center $\mu _ { e }$ uniformly at random within the range of the standardized features and set $s _ { e } ( x , y ) = - \| x - \mu _ { e } \| _ { 2 }$ , so that $b _ { e } ( x ) = \exp \big ( - \tau _ { e } \| x - \mu _ { e } \| _ { 2 } \big )$ and

$$
p _ { e } ^ { \mathrm { { c o v } } } ( x , y ) = \frac { b _ { e } ( x ) p _ { 0 } ( x , y ) } { Z _ { e } ^ { \mathrm { { c o v } } } } , \qquad p _ { e } ^ { \mathrm { { c o v } } } ( x ) = \frac { b _ { e } ( x ) p _ { 0 } ( x ) } { Z _ { e } ^ { \mathrm { { c o v } } } } , \qquad Z _ { e } ^ { \mathrm { { c o v } } } = \mathbb { E } _ { p _ { 0 } } [ b _ { e } ( X ) ] .\tag{12}
$$

Rows closer to $\mu _ { e }$ receive larger weights, so the environment concentrates on a random region of the feature space. The conditional label distribution satisfies $p _ { e } ^ { \mathrm { c o v } } ( y \mid x ) = p _ { 0 } ( y \mid x )$ on the support of the shifted feature distribution, because selection depends on the features alone.

## A.2.3 CONDITIONAL SHIFT

Conditional shift changes the feature distribution within each class while keeping the class proportions fixed (Zhang et al., 2015). For regression, the label bins of Appendix A.2.1 serve as classes, so the distribution over bins is preserved. For each class k, we draw a separate center $\mu _ { e , k }$ in the same

way as for covariate shift and set $s _ { e } ( x , y ) = - \| x - \mu _ { e , y } \| _ { 2 }$ , giving $c _ { e } ( x , y ) = \exp \left( - \tau _ { e } \| x - \mu _ { e , y } \| _ { 2 } \right)$ The weights are normalized separately within each class,

$$
q _ { e } ( x \mid y ) = { \frac { c _ { e } ( x , y ) p _ { 0 } ( x \mid y ) } { Z _ { e } ^ { \mathrm { c o n d } } ( y ) } } , \qquad Z _ { e } ^ { \mathrm { c o n d } } ( y ) = \mathbb { E } _ { p _ { 0 } ( X \mid y ) } [ c _ { e } ( X , y ) ] ,\tag{13}
$$

and the environment distribution is $p _ { e } ^ { \mathrm { c o n d } } ( x , y ) = p _ { 0 } ( y ) q _ { e } ( x \mid y )$ On the candidate pool, row i receives weight $\begin{array} { r } { \hat { p } _ { 0 } ( y _ { i } ) c _ { e } ( x _ { i } , y _ { i } ) / \sum _ { l : \ y _ { l } = y _ { i } } c _ { e } ( x _ { l } , y _ { l } ) } \end{array}$ Within-class normalization is essential: a single global normalizer would generally change the label marginal as well. Consequently, $p _ { e } ^ { \mathrm { c o n d } } \bar { ( } y ) \bar { = } p _ { 0 } ( y )$ , while $p ( x \mid y )$ and hence $p ( y \mid x )$ change. The temperature $\tau _ { e }$ is shared across classes and calibrated so that the class-averaged divergence $\begin{array} { r } { \sum _ { k } \hat { p } _ { 0 } ( k ) \mathrm { K L } \big ( q _ { e } ( \cdot  { \textbf { \ i } } k ) \| \hat { p } _ { 0 } ( \cdot  { \textbf { \ i } } k ) \big ) } \end{array}$ equals 0.1. The invariances in this subsection hold for the environment distributions; finite sampled datasets need not satisfy their empirical counterparts exactly.

## A.3 INTERVENTIONS

Intervention targets. Interventions are applied directly to the causal graph and mechanisms of $\mathcal { M } _ { 0 } ,$ rather than to a completed table (Pearl, 2009). For each interventional environment $e ,$ we sample the target set $I _ { e }$ uniformly at random without replacement from the non-label nodes $V \backslash \{ y \}$ with

$$
| I _ { e } | = { \left\lceil 0 . 0 5 \left( m - 1 \right) \right\rceil } ,\tag{14}
$$

so that at least one node is intervened on. Targets may be observed feature nodes or unobserved nodes. The label node is never intervened on directly, although the label can change through intervened ancestors. After the mechanisms in $I _ { e }$ are modified, we draw fresh exogenous noise, evaluate the modified model in topological order, and apply the base readout. Intervention effects therefore propagate to all descendants of $I _ { e } ,$ , including the label when an intervened node is one of its ancestors. Unlike the distribution shifts, interventions are not calibrated to a KL target; their effect is determined by the 5% budget and by the positions of the targets in $G .$

## A.3.1 HARD INTERVENTIONS

A hard intervention replaces the structural equation of each target with a constant:

$$
z _ { j } = \left\{ \begin{array} { l l } { v _ { e , j } , } & { j \in I _ { e } , } \\ { g _ { j } ( z _ { \mathrm { p a } _ { G } ( j ) } , \epsilon _ { j } ) , } & { j \notin I _ { e } , } \end{array} \right. \quad v _ { e , j } \sim \mathcal { U } [ \ell _ { j } , u _ { j } ] ,\tag{15}
$$

where $\ell _ { j }$ and $u _ { j }$ are the minimum and maximum values of node $j$ in the base table. The value $v _ { e , j }$ is drawn once per environment and shared by all of its rows. Each target no longer depends on its parents, so its incoming edges are removed in the intervened graph. Its outgoing effects remain: descendants are generated from their original structural equations using the intervened value. The exogenous-noise distribution and all non-target mechanisms are unchanged. When a target is a feature node, that feature is constant within the environment, which introduces a point mass into the pooled context.

## A.3.2 SOFT INTERVENTIONS

A soft intervention replaces the mechanism of each target while retaining its parental dependence:

$$
z _ { j } = \left\{ \begin{array} { l l } { g _ { e , j } ( z _ { \mathrm { p a } _ { G } ( j ) } , \epsilon _ { j } ) , } & { j \in I _ { e } , } \\ { g _ { j } ( z _ { \mathrm { p a } _ { G } ( j ) } , \epsilon _ { j } ) , } & { j \notin I _ { e } . } \end{array} \right.\tag{16}
$$

The new mechanism $g _ { e , j }$ belongs to the same function family as $g _ { j }$ in the causal prior, with its parameters resampled independently for environment e. The graph ${ \check { G } } ,$ the parent set of each target, and the noise distribution are unchanged, so a soft intervention changes how the parents determine the target without removing their influence (Brouillard et al., 2020). All other mechanisms retain their base definitions.

## A.4 CONTEXT ASSEMBLY AND MODEL INPUTS

For a multi-environment task, we first sample the number of environments D uniformly from {2, 3, 4}. The environments consist of the unmodified base environment and $D - 1$ further environments, each constructed with one operator sampled uniformly from the five candidates in Table 3.

Given the support and query budgets $n _ { s }$ and $n _ { q } .$ , every environment contributes an equal number of rows, $| S _ { e } | \approx n _ { s } / D$ and $| Q _ { e } | \approx n _ { q } / D$ , so that $\textstyle \sum _ { e = 1 } ^ { D } | S _ { e } | = n _ { s }$ and $\begin{array} { r } { \sum _ { e = 1 } ^ { D } | Q _ { e } | = n _ { q } } \end{array}$ . These datasets are combined as in Equation (3). Because each environment contributes to the support and query sets in the same proportion, both follow the same environment mixture; the construction introduces heterogeneity within a task rather than a shift between its support and query data. Half of the pretraining tasks use the unmodified single-environment process, and the remainder use this multi-environment construction.

The synthetic generator may use labels, including query labels, to construct a task. The prediction network receives only support features, support labels, and query features; query labels provide supervision. The task-conditioning branch computes its feature and label summaries from support data only. Environment identifiers, sampling weights, intervention targets, the causal graph, and its mechanisms are not provided to the prediction network.

The complete procedure for one multi-environment task is summarized below.

1. Sample $\mathcal { M } _ { 0 }$ from the causal prior and draw the base table.

2. Sample $D$ uniformly from $\{ 2 , 3 , 4 \}$ and, for each of the $D - 1$ non-base environments, one operator uniformly from Table 3.

3. For a distribution shift, draw a candidate pool from $\mathcal { M } _ { 0 }$ , compute scores $s _ { e } .$ , find $\tau _ { e }$ by bisection so that $\widehat { \mathrm { K L } } _ { e } ( \tau _ { e } ) = 0 . 1$ 1, and sample rows with replacement according to $w _ { e } ( \tau _ { e } )$ ; if the target is not attainable, discard the environment and sample a new one.

4. For an intervention, sample $I _ { e }$ with $| I _ { e } | = \lceil 0 . 0 5 ( m - 1 ) \rceil$ , modify the mechanisms in $I _ { e }$ according to Equation (15) or (16), and regenerate data in topological order.

5. Split each environment’s rows into $S _ { e }$ and $Q _ { e }$ with equal per-environment budgets, and concatenate them as in Equation (3).

## B MODEL DETAILS

## B.1 SUPPORT STATISTICS

Table 4 summarizes the 51-dimensional descriptor used by the task-specific support gate. The scalar H denotes the input feature width before grouping, including padding, while $d \leq H$ counts valid features. Each of the ten per-feature summaries is aggregated across valid features by its mean, standard deviation, minimum, and maximum. Both standard deviations use the population definition. The near-zero, near-integer, and near-binary tests are $| x | < 1 0 ^ { - 6 } , | x - \mathrm { r o u n d } ( x ) | < 1 0 ^ { - 4 }$ , and min $( | x | , | x - 1 | ) < 1 0 ^ { - 4 }$ , respectively. Summaries use finite support values; a valid feature with none receives zero summaries. Padded features are excluded, and nonfinite descriptor entries are replaced by zero.

Table 4: Statistics used by the task-specific support gate. The descriptor contains 51 entries. Here, H is the input feature width before grouping, including padding, and $d \leq H$ is the number of valid features. Feature summaries are aggregated across valid features by mean, standard deviation, minimum, and maximum.
<table><tr><td>Category</td><td>Statistics</td><td>Dim.</td></tr><tr><td>Size and validity</td><td> $\begin{array} { r } { \log ( 1 + d ) , \log ( 1 + n _ { s } ) , \log ( 1 + N ) , n _ { s } / N , d / H , 1 - d / H , } \end{array}$  and the fraction of finite support values over valid features.</td><td>7</td></tr><tr><td>Support labels</td><td>Observed class count divided by  $K _ { \mathrm { m a x } } ;$  label entropy divided by log  $K _ { \mathrm { m a x } } ;$  largest and smallest proportions among observed classes, with  $K _ { \mathrm { m a x } } = 2 0$ </td><td>4</td></tr><tr><td rowspan="2">Support features</td><td>Per-feature mean, standard deviation, mean absolute value, minimum, maximum, and range.</td><td> $6 \times 4$ </td></tr><tr><td>Per-feature fractions of zero, near-zero, near-integer, and near-binary values.</td><td> $4 \times 4$ </td></tr><tr><td>Total</td><td></td><td>51</td></tr></table>

## C EXPERIMENTAL DETAILS AND ADDITIONAL RESULTS

## C.1 BASELINE ROSTER

The 20 baselines in Table 1 are organized into four groups. Only available results under the matching evaluation protocol are reported; inclusion in this roster does not indicate that every method has a completed result on every suite.

Tree-based methods. These comprise CatBoost (Prokhorenkova et al., 2018), XGBoost (Chen & Guestrin, 2016), random forests (RF) (Breiman, 2001), and extremely randomized trees (ET) (Geurts et al., 2006).

Deep tabular methods. We include SwitchTab (Wu et al., 2024), T2G-Former (Yan et al., 2023), TANGOS (Jeffares et al., 2023), TabCaps (Chen et al., 2023), TabM (Gorishniy et al., 2025), TabNet (Arik & Pfister, 2021), TabR (Gorishniy et al., 2024), and TabTransformer (Huang et al., 2020b). TabCaps is evaluated only for classification.

Tabular foundation models. We include TabICLv1 (Qu et al., 2025), LimiX-2M (Wang et al., 2026), TabPFN2 (Hollmann et al., 2025), Mitra (Zhang et al., 2025b), LimiX-16M (Zhang et al., 2025a), TabICLv2 (Qu et al., 2026), and TabPFN3 (Grinsztajn et al., 2026). TabICLv1 is evaluated only for classification.

Ensemble methods. AutoGluon (Erickson et al., 2020) represents automated ensembling in the comparison roster.

## C.2 DATA SPLITS AND PREPROCESSING

We retain the provided train–test splits. TALENT training and validation datasets are combined as support, BCCO uses its published splits, and OpenML-derived tasks use repeat 0 and fold 0. Numerical missing values are imputed with support-dataset medians. Categorical encodings are fitted on support examples, assigning unseen query categories a value of −1. Regression-label transformations are also fitted on support data, and predictions are mapped back to the original label units before evaluation. Benchmark suites may overlap.

## C.3 TRAINING AND INFERENCE SETTINGS

Classification pretraining. Classification pretraining uses 64 AMD MI210 GPUs, a global batch size of 1,024 synthetic tasks, and a maximum context length of 4,096 rows, including support and query examples. The configured schedule comprises 25,000 steps with the Muon optimizer, a peak learning rate of $6 \times 1 0 ^ { - 4 }$ , weight decay of 0.01, gradient clipping at 10, and label smoothing of 0.02. The learning rate warms up for 500 steps and then follows cosine decay.

Regression model and pretraining. Following TabICLv2 (Qu et al., 2026), we train a separate TAFFY model for regression; it shares no parameters with the classification model. The regression model retains the column and row encoders and the task-conditioned looped Transformer of Sections 4.3 and 4.4, with the regression-specific components of TabICLv2. Continuous support labels are embedded with linear layers instead of class lookup tables, both in the column encoder and before the looped ICL learner. The prediction head outputs 999 quantiles at probability lev els $\{ 0 . 0 0 1 , 0 . 0 0 2 , \ldots , 0 . 9 9 9 \}$ instead of class probabilities, is trained with the pinball loss of Section 3, and produces a point prediction by averaging the predicted quantiles. In the task-conditioning branch, the four class-specific support-label statistics are set to zero (Section 4.4). Regression pretraining uses the In-Context Diversity Prior of Appendix A and the same hardware, global batch size, maximum context length, number of steps, and optimizer settings as classification pretraining, with approximately the same training time (Section 5.1); label smoothing applies only to classification.

Primary-comparison model. In Table 1, classification uses Taffy-4L on all suites; regression uses the separately trained regression model. No per-dataset model selection is performed.

Inference. In-context foundation models predict query labels from the labeled support dataset without updating pretrained parameters. The recorded cross-suite classification protocol uses 32 estimators and FP32 inference for the TabICL-based classifiers, with both identity and powertransformed feature representations. For regression, we use 8 estimators, and the ensemble construction and the feature and target preprocessing follow the TabICLv2 regression configuration: ensemble members differ in random column permutations and feature preprocessing, target trans formations are fitted on the support data (Appendix C.2), and the predictions of the ensemble members are averaged. TabPFN and LimiX retain their model-specific inference configurations, includ ing ensemble and retrieval settings. Thus, frozen-parameter inference does not imply an identical inference-compute budget across models.

## C.4 PRETRAINING COST

Table 5 reports the pretraining cost of the single-pass backbone and the three TAFFY variants on 64 AMD MI210 GPUs. The single-pass backbone uses the TabICLv2 architecture with one pass of the ICL stack. We measure the wall-clock time per training step and project the total time of the 25,000- step schedule from it; GPU-hours multiply this projected time by the 64 GPUs. Pretraining cost increases with loop depth: Taffy-4L requires 1.80× the backbone time per step, while regression has comparable cost under the same setup (Appendix C.3).

Table 5: Pretraining cost on 64 AMD MI210 GPUs. Time per step is measured; total time and GPU-hours are projected for the 25,000-step schedule. Relative is the time per step divided by that of the single-pass backbone.
<table><tr><td>Model</td><td>Time/step (s)</td><td>Relative</td><td>25k steps (h)</td><td>GPU-hours</td></tr><tr><td>Single-pass backbone</td><td>7.01</td><td>1.00×</td><td>48.66 (2.03 days)</td><td>3,114</td></tr><tr><td>Taffy-2L</td><td>8.79</td><td>1.25×</td><td>61.03 (2.54 days)</td><td>3,906</td></tr><tr><td>Taffy-3L</td><td>10.62</td><td>1.51×</td><td>73.74 (3.07 days)</td><td>4,719</td></tr><tr><td>Taffy-4L</td><td>12.62</td><td>1.80×</td><td>87.67 (3.65 days)</td><td>5,611</td></tr></table>

## C.5 METRICS AND AGGREGATION

We report test accuracy for classification and RMSE for regression. Methods are ranked on each dataset, with ties assigned their average rank, and these ranks are averaged within each benchmark suite; lower average rank is better. Missing results are neither imputed nor treated as losses. Elo ratings are computed from observed dataset-level pairwise outcomes as detailed in Appendix C.10.

Table 6: Results on TALENT classification datasets with more than ten classes. Only methods with observed results on all such datasets are included.
<table><tr><td>Method</td><td>Avg. rank ↓</td><td>Elo ↑</td></tr><tr><td>RF</td><td>8.42</td><td>1000.00</td></tr><tr><td>TabICLv1</td><td>5.17</td><td>1595.23</td></tr><tr><td>LimiX-2M</td><td>5.50</td><td>1552.44</td></tr><tr><td>TabPFN2</td><td>5.17</td><td>1595.23</td></tr><tr><td>Mitra</td><td>8.25</td><td>1044.05</td></tr><tr><td>LimiX-16M</td><td>5.12</td><td>1600.49</td></tr><tr><td>TabICLv2</td><td>2.62</td><td>1923.15</td></tr><tr><td>TabPFN3</td><td>2.62</td><td>1923.15</td></tr><tr><td>Taffy</td><td>2.12</td><td>2000.81</td></tr></table>

## C.6 PRIMARY EVALUATION POOL

TabICLv2 (Qu et al., 2026) reports excluding development tasks but does not provide a machinereadable list. We therefore use an operational reconstruction from public materials: classification tasks identified in TabICL Appendix F (Qu et al., 2025) and regression tasks matched between the TabPFNv2 supplementary development roster (Hollmann et al., 2025) and TALENT by name, row count, and feature count. Matching dataset identities are removed from other suites as well; name matches with different source identities and table structures are retained. No missing result is interpolated.

The reconstructed classification exclusions are allbp, baseball, delta\_ailerons, eye\_movements, eye\_movements\_bin, heloc, hill-valley, JapaneseVowels, led24, national-longitudinal-survey-binary, page-blocks, ringnorm, rl, thyroid-ann, and waveform-5000. The regression exclusions are analcatdata\_supreme, cpu\_act, cpu\_small, debutanizer, delta\_elevators, mauna-loa-atmospheric-co2, puma32H, stock\_fardamento02, treasury, weather\_izmir, and wind.

After these exclusions, the primary evaluation uses the remaining memberships in each suite. Classification additionally removes TALENT tasks with more than ten classes from Table 1 and reports them separately below.

## C.7 MANY-CLASS TALENT TASKS

For TALENT classification datasets with more than ten classes, we apply hierarchical classification following TabICL (Qu et al., 2025) together with mixed-radix ensembling (Qu et al., 2026). To avoid imputation, Table 6 includes only the nine methods with observed results on all such datasets. Taffy has the best average rank and Elo in this matched comparison.

## C.8 LIMIX-COMPATIBLE EVALUATION POOL

Following the dataset-eligibility criteria of LimiX-2M (Wang et al., 2026), we retain datasets with at most 50,000 training examples and 10,000 features, and for classification at most 10 classes. We further remove eucalyptus memberships with known evaluation-sample mismatches. Table 7 recomputes accuracy- and RMSE-based ranks using the complete applicable-method subset in each suite. This analysis adopts the LimiX-2M dataset filter, but not its multi-metric aggregation. Taffy has the lowest overall average rank in this pool.

## C.9 PRETRAINING DATA BUDGET

We measure the configured input-data budget by cumulative maximum feature elements, $B ( s ) =$ $\textstyle \sum _ { t = 1 } ^ { s } B _ { t } R _ { t } ^ { \operatorname* { m a x } } d _ { t } ^ { \operatorname* { m a x } }$ , where $B _ { t }$ is the global number of training tasks per step, $R _ { t } ^ { \mathrm { m a x } }$ is the maximum number of support and query rows, and $d _ { t } ^ { \operatorname* { m a x } }$ is the maximum number of input-feature columns, excluding labels. Multistage budgets include the initializing checkpoint’s accumulated budget. Loop count does not multiply this input-data proxy, which is not a measure of computational cost or

Table 7: Average ranks on the LimiX-2M-eligible, sample-compatible evaluation pool (lower is better). Rank aggregation uses the complete applicable-method subset within each suite; dashes denote incomplete task-type coverage.
<table><tr><td rowspan="2">Model</td><td colspan="6">Classification</td><td colspan="5">Regression</td><td rowspan="2">Average</td></tr><tr><td>|BCCO OpenML PFN TALENT TabArena TabZilla|BCCO CTR23</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>PFN TALENT TabArena|</td><td></td></tr><tr><td colspan="10">Ensemble methods</td><td></td><td></td><td></td></tr><tr><td>AutoGluon</td><td>10.53</td><td>8.29</td><td>9.88</td><td>10.00</td><td>9.98</td><td>8.52 5.86</td><td>5.85</td><td>5.96</td><td>6.00</td><td>6.15</td><td>7.91</td></tr><tr><td colspan="10">Tree-based methods</td><td></td><td></td></tr><tr><td>CatBoost</td><td>10.47</td><td>9.88</td><td>9.30</td><td>10.78</td><td>11.02</td><td>7.12</td><td>7.39</td><td>7.25</td><td>6.99</td><td>7.77</td><td>8.97</td></tr><tr><td>XGBoost</td><td>11.68</td><td>10.52</td><td>9.96</td><td>10.72 11.71</td><td>11.03 11.09</td><td>11.28</td><td>10.55</td><td>10.93</td><td>10.29</td><td>10.92</td><td>10.91</td></tr><tr><td>ET</td><td>12.03</td><td>11.56</td><td>10.93</td><td>13.23</td><td>14.77</td><td>11.80 10.14</td><td>10.03</td><td>10.29</td><td>9.70</td><td>9.54</td><td>11.27</td></tr><tr><td>RF</td><td>10.80</td><td>13.46</td><td>11.39</td><td>13.06</td><td>10.48 13.15</td><td>10.60</td><td>11.48</td><td>10.32</td><td>10.06</td><td>10.54</td><td>11.39</td></tr><tr><td colspan="10">Deep tabular methods</td><td></td><td></td></tr><tr><td colspan="10">SwitchTab 16.02 16.90</td><td></td><td>17.23</td><td>16.84</td></tr><tr><td>T2G-Former</td><td>16.71 16.49</td><td>16.45 17.70</td><td>16.16 19.00</td><td>17.45 16.94</td><td>19.00 14.95</td><td>17.06 13.18</td><td>16.36 13.88</td><td>16.39 14.43</td><td>16.56 13.41</td><td></td><td>15.57</td></tr><tr><td>TANGOS</td><td>14.26</td><td>16.16</td><td>16.93</td><td>15.82</td><td>14.39 14.93</td><td>12.10</td><td>11.18</td><td>12.43</td><td>12.80</td><td>14.23 13.00</td><td>14.00</td></tr><tr><td>TabCaps</td><td>14.78</td><td>14.02</td><td>14.46</td><td>14.04</td><td>14.84</td><td>13.48</td><td></td><td></td><td></td><td></td><td>14.27</td></tr><tr><td>TabM</td><td>12.31</td><td>11.67</td><td>12.30</td><td>11.30</td><td>13.00</td><td>11.37 16.00</td><td>15.52</td><td>15.18</td><td>15.61</td><td>17.08</td><td>13.76</td></tr><tr><td>TabNet</td><td>18.92</td><td>18.34</td><td>19.34</td><td>18.70</td><td>18.88</td><td>18.93 16.20</td><td>15.58</td><td>16.96</td><td>16.41</td><td>16.38</td><td>17.69</td></tr><tr><td>TabR</td><td>11.99</td><td>11.09</td><td>11.14</td><td>11.00 15.83</td><td>12.12 10.54</td><td>10.66</td><td>10.67</td><td>12.93</td><td>11.20</td><td>12.31</td><td>11.42</td></tr><tr><td>TabTransformer</td><td>16.54</td><td>13.67</td><td>13.34</td><td>16.14</td><td>14.09</td><td>18.56</td><td>17.55</td><td>16.57</td><td>17.57</td><td>17.92</td><td>16.16</td></tr><tr><td colspan="10">Tabular foundation models</td></tr><tr><td>TabICLv1</td><td>7.74</td><td>8.16</td><td>7.93</td><td>7.63</td><td>7.06</td><td>8.65</td><td></td><td></td><td></td><td></td><td>7.86</td></tr><tr><td>LimiX-2M</td><td>6.88</td><td>6.56</td><td>6.14</td><td>6.65</td><td>5.72</td><td>6.20 10.02</td><td>8.88</td><td>7.75</td><td>9.24</td><td>7.15</td><td>7.38</td></tr><tr><td>TabPFN2</td><td>7.30</td><td>7.87</td><td>6.68</td><td>6.40</td><td>6.44 7.33</td><td>5.14</td><td>6.30</td><td>5.50</td><td>6.05</td><td>6.15</td><td>6.47</td></tr><tr><td>Mitra LimiX-16M</td><td>9.71</td><td>12.60</td><td>12.20</td><td>11.34</td><td>10.31 12.09</td><td>9.26</td><td>10.39</td><td>8.93</td><td>10.13</td><td>9.92</td><td>10.63</td></tr><tr><td>TabICLv2</td><td>6.09</td><td>6.03</td><td>7.00</td><td>5.94</td><td>6.03 6.67</td><td>8.16</td><td>7.55</td><td>7.57</td><td>7.99</td><td>6.46</td><td>6.86</td></tr><tr><td>TabPFN3</td><td>5.50 5.83</td><td>6.20</td><td>5.57</td><td>4.70</td><td>5.88 6.41</td><td>3.72 3.12</td><td>3.73 4.18</td><td>3.39</td><td>4.12 3.39</td><td>3.23</td><td>4.77 4.69</td></tr><tr><td>TAFFY</td><td>4.44</td><td>6.10</td><td>6.45</td><td>4.68</td><td>4.73 6.48</td><td>1.98</td><td>2.94</td><td>3.93 3.29</td><td>2.47</td><td>2.69 1.31</td><td></td></tr><tr><td></td><td></td><td>4.68</td><td>4.89</td><td>3.85</td><td>4.45</td><td>5.19</td><td></td><td></td><td></td><td></td><td>3.59</td></tr></table>

wall-clock throughput. Figure 1(d) compares measured checkpoints at matched or lower input-data budgets; inference settings are not fully controlled across models.

Table 8: Corresponding observed-only Elo ratings under the operational development-set reconstruction (higher is better).
<table><tr><td rowspan="2">Method</td><td colspan="5">Classification</td><td rowspan="2">Regression</td><td colspan="5"></td><td rowspan="2">Mean</td></tr><tr><td>TALENT</td><td>BCCO</td><td>OpenML</td><td>PFN</td><td>TabArena</td><td>TabZilla TALENT</td><td>BCCO</td><td>CTR23</td><td>PFN</td><td>TabArena</td></tr><tr><td>AutoGluon</td><td>1139.13</td><td>1001.45</td><td>1217.39</td><td>1064.14</td><td>1022.49</td><td>1187.80</td><td>1241.40</td><td>1295.80</td><td>1310.03</td><td>1257.72</td><td>1332.51</td><td>1188.17</td></tr><tr><td>CatBoost</td><td>1105.99</td><td>1015.31</td><td>1150.20</td><td>1088.56</td><td>986.67</td><td>1086.68</td><td>1171.15</td><td>1210.85</td><td>1219.76</td><td>1178.34</td><td>1204.78</td><td>1128.94</td></tr><tr><td>XGBoost</td><td>1056.39</td><td>957.92</td><td>1123.70</td><td>1061.09</td><td>975.45</td><td>1084.46</td><td>981.73</td><td>956.97</td><td>1050.70</td><td>964.64</td><td>970.83</td><td>1016.72</td></tr><tr><td>ET</td><td>986.81</td><td>934.06</td><td>1080.91</td><td>1017.65</td><td>800.16</td><td>1053.96</td><td>1016.33</td><td>1027.80</td><td>1078.09</td><td>1002.07</td><td>1074.19</td><td>1006.55</td></tr><tr><td>RF</td><td>1000.00</td><td>1000.00</td><td>1000.00</td><td>1000.00</td><td>1000.00</td><td>1000.00</td><td>1000.00</td><td>1000.00</td><td>1000.00</td><td>1000.00</td><td>1000.00</td><td>1000.00</td></tr><tr><td>SwitchTab</td><td>769.70</td><td>718.50</td><td>854.24</td><td>767.09</td><td>522.28</td><td>869.73</td><td>573.18</td><td>490.49</td><td>677.30</td><td>590.18</td><td>322.86</td><td>650.50</td></tr><tr><td>T2G-Former</td><td>806.69</td><td>735.87</td><td>779.84</td><td>555.29</td><td>790.55</td><td>812.06</td><td>803.46</td><td>827.86</td><td>860.26</td><td>743.84</td><td>682.11</td><td>763.44</td></tr><tr><td>TANGOS</td><td>866.91</td><td>844.88</td><td>870.70</td><td>719.58</td><td>819.03</td><td>921.98</td><td>846.05</td><td>909.75</td><td>1016.50</td><td>874.66</td><td>800.31</td><td>862.76</td></tr><tr><td>TabCaps</td><td>960.88</td><td>822.62</td><td>974.46</td><td>859.48</td><td>795.37</td><td>985.93</td><td></td><td></td><td></td><td></td><td></td><td>899.79</td></tr><tr><td>TabM</td><td>1067.57</td><td>935.32</td><td>1075.44</td><td>959.47</td><td>885.79</td><td>1072.59</td><td>652.53</td><td>576.99</td><td>746.94</td><td>689.38</td><td>344.10</td><td>818.74</td></tr><tr><td>TabNet</td><td>683.62</td><td>567.24</td><td>734.24</td><td>519.59</td><td>532.55</td><td>680.37</td><td>583.67</td><td>572.26</td><td>742.28</td><td>535.86</td><td>434.53</td><td>598.75</td></tr><tr><td>TabR</td><td>1087.05</td><td>947.49</td><td>1098.94</td><td>1009.99</td><td>926.06</td><td>1105.94</td><td>939.75</td><td>992.01</td><td>1044.23</td><td>843.41</td><td>860.32</td><td>986.84</td></tr><tr><td>TabTransformer</td><td>873.54</td><td>725.87</td><td>990.23</td><td>912.61</td><td>726.10</td><td>959.55</td><td>467.88</td><td>201.40</td><td>555.86</td><td>573.90</td><td>218.03</td><td>655.00</td></tr><tr><td>TabICLv1</td><td>1250.62</td><td>1125.39</td><td>1223.07</td><td>1148.49</td><td>1159.11</td><td>1182.42</td><td></td><td></td><td></td><td></td><td></td><td>1181.52</td></tr><tr><td>LimiX-2M</td><td>1287.21</td><td>1157.56</td><td>1295.61</td><td>1232.14</td><td>1229.12</td><td>1290.29</td><td>1048.98</td><td>1051.50</td><td>1139.16</td><td>1148.70</td><td>1251.78</td><td>1193.82</td></tr><tr><td>TabPFN2</td><td>1310.96</td><td>1142.44</td><td>1235.63</td><td>1206.00</td><td>1190.78</td><td>1238.43</td><td>1223.98</td><td>1361.42</td><td>1282.47</td><td>1288.05</td><td>1332.51</td><td>1255.70</td></tr><tr><td>Mitra</td><td>1077.65</td><td>1039.40</td><td>1036.81</td><td>964.20</td><td>1007.72</td><td>1043.47</td><td>983.63</td><td>1079.13</td><td>1058.78</td><td>1080.25</td><td>1045.86</td><td>1037.90</td></tr><tr><td>LimiX-16M</td><td>1329.06</td><td>1199.91</td><td>1321.67</td><td>1190.80</td><td>1212.16</td><td>1268.48</td><td>1114.45</td><td>1153.87</td><td>1211.35</td><td>1159.22</td><td>1306.92</td><td>1224.35</td></tr><tr><td>TabICLv2</td><td>1416.24</td><td>1232.42</td><td>1313.12</td><td>1261.44</td><td>1220.58</td><td>1280.59</td><td>1365.45</td><td>1480.73</td><td>1458.84</td><td>1447.78</td><td>1647.59</td><td>1374.98</td></tr><tr><td>TabPFN3</td><td>1414.45</td><td>1216.29</td><td>1317.99</td><td>1216.32</td><td>1286.42</td><td>1277.10</td><td>1422.65</td><td>1543.54</td><td>1423.03</td><td>1402.27</td><td>1731.77</td><td>1386.53</td></tr><tr><td>Taffy</td><td>1476.85</td><td>1292.81</td><td>1394.88</td><td>1298.76</td><td>1304.18</td><td>1341.96</td><td>1529.52</td><td>1678.27</td><td>1529.66</td><td>1457.50</td><td>2047.85</td><td>1486.57</td></tr></table>

## C.10 ELO COMPARISON

We report Elo ratings under both the primary evaluation pool and the LimiX-2M eligibility rule. The former follows the development-set and many-class controls used for Table 1; the latter uses the stricter LimiX-2M dataset filter. Taffy has the highest mean Elo under both evaluation pools.

Rating protocol. Each dataset supplies one comparison per method pair. Higher accuracy wins for classification and lower RMSE wins for regression; ties count as half a win. Accuracy differences within $1 0 ^ { - 1 2 }$ are ties; RMSE ties require equality of stored values. We jointly fit an unpenalized Bradley–Terry model per benchmark, with equal dataset weights. The fitted strengths $\theta _ { i }$ give orderindependent Elo ratings,

$$
E _ { i } = 1 0 0 0 + { \frac { 4 0 0 } { \ln 1 0 } } ( \theta _ { i } - \theta _ { \mathrm { R F } } ) , \qquad p _ { i j } = { \frac { 1 } { 1 + 1 0 ^ { ( E _ { j } - E _ { i } ) / 4 0 0 } } } .\tag{17}
$$

RF is anchored at 1000, following TabArena’s calibration convention (Erickson et al., 2025). Average is the mean of all eleven unrounded ratings, not a pooled Elo fit; it is omitted for methods without regression results. The evaluation settings are those used for the reported results; this is not a compute-matched comparison or a test of statistical significance.

Primary evaluation. Table 8 applies the same operational development-set exclusions as Table 1;   
classification additionally excludes TALENT tasks with more than ten classes.

LimiX-compatible evaluation. Table 9 uses the same LimiX-compatible evaluation pool as Table 7. Missing scores are neither imputed nor treated as losses.

## C.11 GATE ANALYSIS DETAILS

Protocol and cohorts. We analyze one late-training checkpoint of each of Taffy-2L, Taffy-3L, and Taffy-4L. We reconstruct α on CPU from the saved FP32 gate parameters and the implementation’s statistics of the full processed training-plus-validation support dataset. Input-shape metadata retain the full query count. For each dataset, we average the coefficients from the none and power preprocessing views. These are reconstructed coefficients, not recorded GPU traces. The analysis uses TALENT datasets with at most 20 classes and at most one million support cells. Categorical fraction requires matching raw metadata, while the kurtosis analysis additionally requires numerical features and finite metadata.

Descriptors and adjustment. Categorical fraction is the fraction of original input columns that are categorical. Numerical excess kurtosis is the median across raw numerical columns of mean $[ ( x -$ $\bar { x } ) ^ { 4 } ] / \sigma ^ { 4 } - 3 ,$ , using the population standard deviation and assigning zero to constant columns. The raw numerical summaries use finite values from a deterministic sample of at most 10,000 original training rows. Datasets without numerical columns are excluded from the kurtosis analysis. Raw metadata and processed support statistics therefore describe different stages of the data pipeline. We compute partial Spearman correlations by residualizing the midranks of each descriptor and α against an intercept and the midranks of log support rows, log effective columns, class count, and support-label entropy $H ( y ) /$ log 20. Kurtosis fits also control for categorical fraction.

Table 9: Elo ratings on the LimiX-2M-eligible, sample-compatible evaluation pool (higher is better; RF = 1000 per benchmark). Only observed pairwise outcomes are used; missing scores are not imputed, and dashes denote incomplete task-type coverage.
<table><tr><td rowspan="2">Model</td><td colspan="6">Classification</td><td colspan="5">Regression</td><td rowspan="2">Average</td></tr><tr><td>BCCO OpenML</td><td></td><td>PFN</td><td></td><td>TALENT TabArena TabZilla| BCCO</td><td></td><td>CTR23</td><td>PFN</td><td></td><td>TALENT TabArena|</td></tr><tr><td colspan="10">Ensemble methods</td></tr><tr><td>AutoGluon</td><td>|1011.381217.39</td><td></td><td>1064.141138.25</td><td></td><td>1021.541187.80|1303.991310.031257.72</td><td></td><td></td><td></td><td>1236.25</td><td></td><td>1332.51 |1189.18</td></tr><tr><td colspan="10"></td></tr><tr><td>CatBoost</td><td>1013.58</td><td>1150.20</td><td>1088.56</td><td>1106.17</td><td>Tree-based methods 973.79</td><td>1086.68</td><td></td><td>1216.331219.761178.34</td><td>1174.86</td><td>1204.78</td><td></td></tr><tr><td>XGBoost</td><td>962.97</td><td>1123.70</td><td>1061.09</td><td>1061.68</td><td>976.48</td><td>1084.46</td><td></td><td>957.85 1050.70 964.64</td><td>986.98</td><td>970.83</td><td>1128.46 1018.31</td></tr><tr><td>ET</td><td>948.06</td><td>1080.91</td><td>1017.65 1000.00</td><td>991.99 1000.00</td><td>809.15 1000.00 1000.00</td><td>1053.96</td><td></td><td>1028.09 1078.091002.07 1000.001000.001000.00</td><td>1020.32 1000.00</td><td>1074.19 1000.00</td><td>1009.50 1000.00</td></tr><tr><td colspan="10">RF 1000.00 1000.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Deep tabular methods</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SwitchTab T2G-Former</td><td>724.60 737.05</td><td>854.24 779.84</td><td>767.09 555.29</td><td>768.92 799.91</td><td>521.75 869.73 800.02 812.06</td><td>480.01 831.16</td><td>677.30 860.26</td><td>590.18 743.84</td><td>556.05 800.85</td><td>322.86 682.11</td><td>648.43 763.85</td></tr><tr><td>TANGOS</td><td>849.72</td><td>870.70</td><td>719.58</td><td>863.75</td><td>827.08 921.98</td><td>905.28</td><td>1016.50</td><td>874.66</td><td>840.19</td><td>800.31</td><td>862.70</td></tr><tr><td>TabCaps</td><td>825.54</td><td>974.46</td><td>859.48</td><td>954.11</td><td>792.33</td><td>985.93</td><td></td><td></td><td></td><td></td><td>898.64</td></tr><tr><td>TabM</td><td>935.27</td><td>1075.44</td><td>959.47</td><td>1079.86</td><td>880.95</td><td>1072.59 586.18</td><td>746.94</td><td>689.38</td><td>640.91</td><td>344.10</td><td>819.19</td></tr><tr><td>TabNet</td><td>570.05</td><td>734.24</td><td>519.59 675.82</td><td></td><td>528.89 680.37</td><td>564.30</td><td>742.28</td><td>535.86</td><td>569.60</td><td>434.53</td><td>595.96</td></tr><tr><td>TabR</td><td>949.47</td><td>1098.94</td><td>1009.99 1093.77</td><td></td><td>922.90 1105.94</td><td>996.32</td><td>1044.23</td><td>843.41</td><td>935.41</td><td>860.32</td><td>987.34</td></tr><tr><td>TabTransformer</td><td>734.37</td><td>990.23</td><td>912.61 863.60</td><td></td><td>723.74 959.55</td><td>182.08</td><td>555.86</td><td>573.90</td><td>444.82</td><td>218.03</td><td>650.80</td></tr><tr><td colspan="10">Tabular foundation models</td></tr><tr><td>TabICLv1</td><td>1131.13</td><td>1223.07</td><td>1148.49</td><td>1249.15</td><td>1152.41</td><td>1182.42</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LimiX-2M</td><td>1170.90</td><td>1295.61</td><td>1232.14</td><td>1298.69 1198.36</td><td>1290.29</td><td></td><td></td><td>1035.391139.161148.70</td><td>1045.70</td><td>1251.78</td><td>1181.11 1191.52</td></tr><tr><td>TabPFN2</td><td>1151.09</td><td>1235.63</td><td>1206.00</td><td>1311.45</td><td>1182.70 1238.43</td><td></td><td></td><td>1358.901282.471288.05</td><td>1233.01</td><td>1332.51</td><td>1256.39</td></tr><tr><td>Mitra</td><td>1045.63</td><td>1036.81</td><td>964.20</td><td>1078.11</td><td>1007.40</td><td>1043.47</td><td></td><td>1081.58 1058.781080.25</td><td>996.04</td><td>1045.86</td><td>1039.83</td></tr><tr><td>LimiX-16M</td><td>1210.05</td><td>1321.671190.80</td><td></td><td>1336.53</td><td>1203.14</td><td>1268.48</td><td></td><td>1149.401211.351159.22</td><td>1116.35</td><td>1306.92</td><td>1224.90</td></tr><tr><td>TabICLv2</td><td>1240.08</td><td>1313.12 1261.44</td><td></td><td>1409.58</td><td>1211.18 1273.94</td><td>1280.59</td><td></td><td>1483.451458.841447.78</td><td>1370.48</td><td>1647.59</td><td>1374.92</td></tr><tr><td>TabPFN3</td><td>1222.74</td><td>1317.99</td><td>1216.32</td><td>1410.83</td><td></td><td>1277.10</td><td></td><td>1546.031423.031402.27</td><td>1433.48</td><td>1731.77</td><td>1386.86</td></tr><tr><td>TAFFY</td><td></td><td></td><td>1301.291394.881298.761467.72</td><td></td><td>1290.81</td><td>1341.96</td><td></td><td>1697.50 1529.66 1457.501530.52</td><td></td><td>2047.85</td><td>1487.13</td></tr></table>

Uncertainty and multiple testing. Table 10 reports pointwise 95% percentile intervals from 2,000 dataset bootstrap resamples, recomputing ranks and residuals each time. Categorical-fraction qvalues use 9,999 residual permutations and Benjamini–Hochberg correction over 15 tests: three descriptors across five late-training checkpoints of the three variants. Kurtosis q-values use approximate two-sided t tests and a separate 18-test correction covering three additional descriptors, each tested against three gate coefficients and three fixed-checkpoint inference-depth gains. Across the three model depths, Table 10 shows negative associations with categorical fraction and positive associations with numerical excess kurtosis.

Table 10: Adjusted associations between α and TALENT dataset properties. Categorical fraction and numerical excess kurtosis use separate multiple-testing corrections.
<table><tr><td>Descriptor</td><td>Model</td><td colspan="2">Partial ρ [95% CI]</td><td>q</td></tr><tr><td>Categorical fraction</td><td>Taffy-2L</td><td>-0.533</td><td>[−0.652, -0.394]</td><td>0.0003</td></tr><tr><td rowspan="5">Numerical excess kurtosis</td><td>Taffy-3L</td><td>-0.453</td><td>[−0.580, −0.299]</td><td>0.0003</td></tr><tr><td>Taffy-4L</td><td>-0.452</td><td>[-0.579, -0.296]</td><td>0.0003</td></tr><tr><td>Taffy-2L</td><td></td><td>0.230 [0.057, 0.383]</td><td>0.0338</td></tr><tr><td>Taffy-3L</td><td></td><td>0.326 [0.167, 0.466]</td><td>0.000703</td></tr><tr><td>Taffy-4L</td><td></td><td>0.323 [0.167, 0.463]</td><td>0.000703</td></tr></table>

## C.11.1 SUPPORT ROWS AND EFFECTIVE COLUMNS

Across TALENT datasets, α is positively associated with support-row count but shows no consistent association with effective column count (Table 11).

Table 11: Unadjusted associations between α and support size on TALENT datasets. All q-values use Benjamini–Hochberg correction over the original 216 tests.
<table><tr><td rowspan="2">Model</td><td colspan="2">Support rows</td><td colspan="2">Effective columns</td></tr><tr><td>ρ</td><td>q</td><td>ρ</td><td>q</td></tr><tr><td>Taffy-2L</td><td>0.192</td><td>0.0374</td><td>0.138</td><td>0.199</td></tr><tr><td>Taffy-3L</td><td>0.297</td><td> $2 . 3 1 \times 1 0 ^ { - 4 }$ </td><td>0.053</td><td>0.660</td></tr><tr><td>Taffy-4L</td><td>0.397</td><td> $3 . 4 4 \times 1 0 ^ { - 7 }$ </td><td>0.022</td><td>0.881</td></tr></table>

## C.12 FROZEN-CHECKPOINT GATE INTERVENTION

We examine task-specific gating with separately trained two-loop models in Figure 3(b) and a frozen-checkpoint intervention that holds all model parameters fixed.

Starting from a Taffy-4L checkpoint, we rerun the complete inference path under three conditions: the native task-specific coefficient $\alpha _ { S }$ , the checkpoint’s support-independent coefficient obtained by setting $b _ { S } = 0$ , and $\alpha = 1$ , which bypasses the learned modulation and retains the final recurrent update at full strength. All conditions use identical parameters, four passes, preprocessing, ensemble construction, and query batches; only the update coefficient changes. Complete outputs for all three conditions are available on matched TALENT classification datasets, so we compute the comparison without imputing missing conditions. On each dataset, the three conditions are ranked by classification performance, with average ranks assigned to ties, before averaging across datasets.

Table 12: Frozen-checkpoint gate intervention on matched TALENT classification datasets. Mean rank among the three inference conditions; lower is better.
<table><tr><td>Inference condition</td><td>Update coefficient</td><td>Average rank ↓</td></tr><tr><td>Task-specific gate</td><td>αs</td><td>1.57</td></tr><tr><td>Support-independent gate</td><td>tanh(a)</td><td>1.88</td></tr><tr><td>Gate bypass (full update)</td><td>1</td><td>2.55</td></tr></table>

On the matched TALENT datasets, the task-specific gate has the lowest average rank among the three inference conditions (Table 12).

## D LIMITATIONS

Task conditioning in TAFFY uses a single task-level scalar gate applied to the final refinement. Although this keeps the mechanism lightweight, it cannot assign different refinement strengths across examples, features, layers, or earlier passes. Comparing this design with finer-grained gating under matched parameter and compute budgets is an important extension.