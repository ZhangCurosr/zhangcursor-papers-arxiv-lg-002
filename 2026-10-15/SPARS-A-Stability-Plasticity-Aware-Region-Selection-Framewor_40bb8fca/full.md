# SPARS: A Stability-Plasticity-Aware Region Selection Framework for Continual Learning

Teif Aldaajani<sup>1\*</sup>, Huda Hakami<sup>1\*</sup> and Maha A. Thafar<sup>1</sup>

<sup>1\*</sup> Department of Computer Science, College of Computers and Information Technology, Taif University, Taif, 21944, Saudi Arabia .

\*Corresponding author(s). E-mail(s): s44680101@students.tu.edu.sa; hahakami@tu.edu.sa; Contributing authors: m.thafar@tu.edu.sa;

## Abstract

Continual learning aims to acquire new knowledge while retaining previously learned knowledge. However, updating shared parameters when learning new tasks often overwrites existing knowledge, causing catastrophic forgetting. Memory replay methods mitigate catastrophic forgetting but rely on previous samples, raising privacy, memory, and scalability concerns. On the other hand, parameter isolation-based methods preserve learned knowledge but may limit knowledge transfer, whereas regularization promotes adaptability but remains susceptible to interference between old and new tasks. To maintain the stability-plasticity tradeof, this paper proposes Stability-Plasticity-Aware Region Selection (SPARS), a replay-free hybrid framework combining global parameter isolation with localized parameter regularization. SPARS estimates parameter importance using either the Fisher Information Matrix (FIM) or a novel Confidence Score (CS) estimator that leverages logit sensitivity and achieves comparable performance to FIM without requiring labelled data. Based on parameter importance, SPARS partitions parameters into three importance-aware regions namely, winning, neighboring, and non-important parameter regions. Each region receives a different adaptation strategy: selective freezing, soft regularization, or unrestricted learning based on its role in the stability-plasticity trade-of. To mitigate classifier bias in class-incremental learning, we employ Quadratic Discriminant Analysis to improve performance on previously learned classes .Experiments on CIFAR-10 and TinyImageNet demonstrate consistent improvement in average incremental accuracy over existing isolation- and regularization-based methods across taskand class-incremental learning. On TinyImageNet, SPARS achieves the best performance with 6% improvement for class-incremental learning while retaining over 60% of parameters for future tasks.

## 1 Introduction

As the real world is inherently dynamic, data are continuously generated and evolve over time, violating the conventional independent and identically distributed (i.i.d.) assumption on which most deep learning models rely. Although deep neural networks (DNNs) have achieved remarkable success across a wide range of applications, they struggle to adapt to continuously evolving data distributions after deployment. Continual Learning (CL) aims to enable models to acquire new knowledge sequentially while preserving previously learned knowledge. A fundamental challenge in CL is catastrophic forgetting (CF) [1], where learning new tasks causes the model to forget previously acquired knowledge. Catastrophic forgetting arises when optimization on newly arriving data modifies parameters that are important for previously learned tasks, thereby disrupting established representations and degrading performance on earlier tasks.

Mitigating catastrophic forgetting is essential for enabling models to adapt to changing data distributions while retaining knowledge acquired from previous tasks. Existing CL methods can be broadly categorized into memory replay, regularizationbased, and parameter-isolation approaches. Memory replay methods retain a subset of samples from previous tasks and replay them when learning new tasks [2–5]. Although achieving State-of-the-art performance, these methods, storing previous data introduces privacy and regulatory concerns, particularly in restricted environments, while also increasing memory and computational costs. Memory-free approaches avoid storing previous samples but face important trade-ofs. Regularization-based methods constrain updates to parameters considered important for previous tasks, but such constraints may be insuficient to preserve learned representations over long task sequences[6–8]. In contrast, parameter isolation methods protect previous knowledge by restricting diferent parameters to diferent tasks, but progressively consume available model capacity and can eventually lead to model saturation as the number of tasks increases [9–12].

Despite the individual limitations of diferent CL approaches, their performance is also influenced by the incremental learning setting in which they are evaluated. Incremental learning is generally classified into two settings, namely task-incremental learning (TIL) and class-incremental learning (CIL)[13]. Among these settings, CIL is particularly challenging because task identity is unavailable during inference. Therefore, in addition to CF, it sufers from classifier bias, whereby the classifier favors recently learned classes because samples from previous tasks are imbalanced in memory replay methods or unavailable in memory-free methods. Some prior work has addressed this issue through model-output calibration [14–16], while other approaches require additional training procedures or optimization steps [17, 18]. However, these approaches may not be suitable for real-world applications due to either their additional computational cost or the limited reliability of the calibration process.

Building on these limitations, we propose Stability-Plasticity-Aware Region Selection (SPARS), a novel replay-free continual learning framework that explores the underexamined combination of importance-based global parameter isolation and localized parameter regularization. Unlike existing approaches that either constrain all important parameters or isolate model components without accounting for their functional importance, SPARS selectively preserves critical regions while allowing less important regions to adapt to new knowledge. In particular, we partition the network into three importance-aware regions (winning, neighboring and non-important) and enables each region to adapt according to its role in the stability–plasticity tradeof. Moreover, to estimate parameter importance in privacy-sensitive environments where access to data is restricted, we propose a novel estimator, termed Confidence Score (CS), which leverages model confidence without requiring labeled data. To address classifier bias in CIL, we employ Quadratic Discriminant Analysis (QDA) as an inference-time discriminative classifier, eliminating the need for model-output calibration or storing samples from previous tasks. The framework is designed as a plug-and-play CL method that can be applied to any pretrained model to support additional tasks. It also avoids model saturation, making it more scalable than previous methods. Extensive experiments are conducted to demonstrate the efectiveness of the proposed framework against various baselines, along with exploratory studies of its diferent components. Comparison experiment results show that our proposed SPARS achieves competitive performance on CIFAR-10 and superior performance on TinyImageNet, with a 6% improvement over the strongest baselines. .

The key contributions of our work are summarized as follows:

• Propose SPARS, a replay-free and hybrid CL framework that combines global parameter isolation with local parameter regularization through a region-aware adaptation strategy.

• Introduce a novel Confidence Score (CS)-based parameter importance estimator that leverages logit sensitivity of the model output and provides a label-free alternative to the Fisher Information Matrix (FIM) without requiring access to original training data.

• Mitigate classifier bias in CIL by adopting Quadratic Discriminant Analysis (QDA) as an inference-time classifier, and compare its efectiveness with Linear Discriminant Analysis (LDA) across two benchmark datasets.

## 2 Related Work

## 2.1 Continual Learning Approaches

Existing continual learning approaches can generally be categorized into replay-based, parameter regularization-based, parameter isolation-based methods, and hybrid methods. Each category addresses catastrophic forgetting from a diferent perspective, depending on how previously learned knowledge is preserved while adapting to new tasks.

Replay-Based Methods Memory replay represents one of the most efective stateof-the-art strategies to mitigate catastrophic forgetting by retaining [2, 19–21] or generating past data [22–24] to rehearse during new task training. Early approaches such as Experience Replay [25] maintain a memory bufer of previously observed sam ples, while Incremental Classifier and Representation Learning (iCaRL) [2] employs exemplar selection and rehearsal together with a distillation objective to preserve knowledge of previously learned classes. Other methods such as Gradient Episodic Memory (GEM) [19, 20] store a subset of examples from previously learned tasks and, before updating the model, compare the gradient of the current task with gradients computed from the episodic memory. The update is constrained when the gradients conflict, thereby reducing interference with previously acquired knowledge. Recent replay-based methods, such as Dark Experience Replay (DER) [26], modify traditional experience replay by focusing on soft probabilities (logits) a model produced in the past, thereby providing additional supervision for preserving earlier representations. Instead of storing samples from previous tasks, which results in memory overhead, privacy concerns, or data-access restrictions, pseudo-memory replay methods seek to approximate the benefits of rehearsal without maintaining the original training samples. Generative replay methods, such as Deep Generative Replay (DGR) [22] and Memory replay GANs (MeRGAN) [23], train a generative model to produce synthetic samples resembling earlier tasks, allowing the model to rehearse previous knowledge without directly storing the original examples.

Although replay-based methods efectively reduce forgetting, their dependence on stored or generated samples limits their applicability in scenarios with strict memory or privacy constraints. In contrast, parameter-based approaches (regularization or isolation as discussed next) seek to preserve previous knowledge directly within the model parameters, eliminating the need to retain or reconstruct previous data. This motivates our proposed SPARS framework, which falls into the parameter-based category.

Parameter Regularization-Based Methods These methods identify parameters that are important for previously learned tasks and penalize changes to those parameters during subsequent training [6–8, 27]. Elastic Weight Consolidation (EWC) [6] estimates parameter importance using the Fisher Information Matrix and imposes stronger penalties on highly important parameters, while Synaptic Intelligence (SI) [7] estimates parameter importance online based on their contribution to learning throughout training. Memory Aware Synapses (MAS) [8], in contrast, estimates importance from the sensitivity of the network outputs and can therefore operate without labeled data. Subsequent approaches have further improved parameter importance estimation, such as RWalk [28], which combines multiple importance measures, and online EWC [29], which maintains a cumulative importance estimate across tasks. Variational Continual Learning (VCL) [30] adopts a Bayesian alternative by using the posterior learned from previous tasks as the prior for the current task, thus constraining the evolution of the model parameters while maintaining uncertainty over their values.

Although these methods can efectively preserve previously acquired knowledge without replay, conventional parameter regularization generally treats parameter importance independently, allowing important parameters to adapt under penalties determined by their estimated importance.

Recent research has therefore explored structured regularization strategies that preserve groups of related parameters to better retain learned representations while maintaining adaptability to new tasks [31–33]. However, these approaches typically focus on preserving structured parameter groups rather than explicitly diferentiating the degree of adaptation according to the importance of individual parameters and their local relationship to important parameters. The proposed SPARS addresses this gap by adopting an importance-aware, localized regularization strategy that diferentiates the treatment of parameters according to their importance and proximity to important parameters.

Parameter Isolation-Based Methods This category of CL assumes the independence between tasks so that each task has dedicated parameters [12, 34, 35], subnetworks [36, 37], masks [11, 38], or experts [39, 40] rather than relying on parameter regularization. In general, there is no significant forgetting with this type of method. However, they are not aligned with other desiderata of continual learning, such as knowledge transfer and the trade-of between stability and plasticity. A representative approach is Hard Attention to the Task (HAT) [11], which learns task-specific masks to identify and protect parameters that are important for previously learned tasks. Building on the idea of allocating diferent subsets of parameters to diferent tasks, PackNet [9] progressively prunes and freezes parameters after learning each task, thereby reserving the remaining capacity for subsequent tasks. SupSup [36] takes parameter isolation further by representing each task as a sparse subnetwork within a shared network, allowing multiple tasks to reuse parameters while preventing updates to previously selected subnetworks. More recent approaches such as Soft-Winning Sub-Networks WSN [38] further explore parameter sharing and task-specific isolation to improve the utilization of the available model capacity.

Despite these advances, isolation-based methods generally impose a hard separation between parameters that are preserved and those available to learn new tasks, which limits their capacity for a long sequence of tasks. SPARS builds upon the parameter-isolation paradigm by relaxing the hard separation through applying isolation globally across tasks and regularization locally to important winning parameters.

Hybrid Methods Given that excessive emphasis on stability may hinder the acquisition of new knowledge, several continual learning methods integrate mechanisms that preserve previously learned knowledge while simultaneously promoting model plasticity. These methods combine two or more approaches to address catastrophic forgetting. Since each category has its own limitations, such combinations can help alleviate the limitations of individual approaches. Nevertheless, such combinations may introduce new limitations, including additional computational complexity and sensitivity to hyperparameters. A common approach is to combine regularization-based methods with memory replay to mitigate representation drift while learning new tasks and leveraging stored samples [41–46]. In addition, regularization has been combined with isolation-based approaches to isolate learned knowledge and integrate it again through knowledge distillation, thereby achieving a balance between acquiring new knowledge and preserving previously learned knowledge [47]. Isolation-based approaches have also been combined with model-growth approaches, providing a direct means of addressing model saturation. While isolation can lead to model saturation as the number of tasks increases, model growth helps overcome this limitation and enables the network to scale. However, model growth introduces additional computational costs, and this combination may limit knowledge sharing across tasks [48]. Furthermore, modelgrowth-based approaches have been combined with memory replay and orthogonal regularization to promote orthogonality between tasks while leveraging memory replay to mitigate bias [49, 50].

Despite their potential, these hybrid approaches still inherit some of the limitations of their constituent categories. Memory replay-based methods require memory bufers, which are limited by available memory capacity and may raise privacy concerns. Isolation-based approaches can limit knowledge sharing between tasks by keeping them separate. Regularization-based approaches, meanwhile, typically apply constraints across model parameters regardless of their individual contributions to model stability or plasticity. Our method aims to address these limitations by handling parameters according to their contribution to model stability and plasticity. Specifically, it incorporates global isolation with local regularization across tasks while avoiding the need for replayed examples, reducing the risk of model saturation, and eliminating the need for model expansion.

## 2.2 Bias in Class-Incremental Learning

Class-incremental learning is a challenging CL setting in which task identifiers are unavailable during inference, requiring the model to distinguish among all learned classes without explicit task information. Consequently, methods that perform well in TIL can sufer substantial performance degradation under CIL, largely due to bias at the representation or classifier level [51]. This bias may favor either old or newly introduced classes [52–54], with a bias toward new classes being more commonly observed.

Several approaches have been proposed to address the bias issue in CIL, including correction-based, out-of-distribution (OOD)-based, and inference-based methods. These approaches difer in where and how bias is addressed: some operate primarily during inference [2, 15], whereas others introduce additional training procedures or optimization objectives [14, 17, 18]. Correction-based methods employ explicit bias correction mechanisms, such as lightweight, learnable layers [14] or calibration of model outputs and class weight norms [17, 18]. However, such methods may have limited efectiveness because they primarily compensate for bias at the output level, rather than addressing its underlying causes [14, 18].

OOD-based methods [15, 55] formulate CIL as two complementary subproblems: within-task prediction, which discriminates among classes belonging to the same task, and task identification, which is treated as an OOD detection problem. Although this strategy avoids explicit output calibration and its associated limitations, it introduces additional inference complexity and latency because prediction efectively involves two stages. More importantly, it implicitly attempts to recover task identity at inference time, which is unavailable by definition in the standard CIL setting.

Inference-time solutions can be attractive in practice because they mitigate bias without requiring additional model training or modification of the learned parameters [56–58] One representative approach is presented in [2], which replaces the conventional Softmax classifier with a nearest-mean-of-exemplars (NME) classifier. However, relying solely on class means may be insuficient for deep neural networks, as their learned feature representations can exhibit complex distributions that are not adequately captured by a single class centroid. Moreover, in continual learning, the feature distributions of previously learned classes may shift as new classes are introduced, making their original class means less reliable.

Overall, existing bias-mitigation approaches either rely on output-level calibration, require additional statistical information or replay, or introduce inference complexity through implicit task identification. To address these limitations, our proposed SPARS framework employs a simple inference-time method that replaces the original classifier with a Quadratic Discriminant Analysis classifier during inference. Unlike the NME method, QDA models both class means and covariance matrices, enabling it to capture richer class distributions and construct more flexible decision boundaries. The deep network is therefore used primarily as a feature extractor, while the original classifier is discarded during inference and replaced by QDA. This design provides a practical inference-time solution capable of modeling complex feature distributions without requiring explicit task identification or output-level calibration.

## 3 Methodology

## 3.1 Preliminary

Continual learning is commonly studied under two settings: task-incremental learning and class-incremental learning. In both settings, the model learns a sequence of tasks, $T = [ t _ { 1 } , t _ { 2 } , t _ { 3 } , \ldots , t _ { n } ]$ , where each task t contains a set of classes, $C _ { t } .$ and their corresponding labels, $Y _ { t }$ . Label sets are assumed to be disjoint across tasks. The model consists of a backbone network and one or more classification heads. In TIL, the model employs a set of task-specific heads, $H = [ h _ { 1 } , h _ { 2 } , h _ { 3 } , \ldots , h _ { n } ]$ , where each head $h _ { i }$ corresponds to a specific task and contains output nodes representing the classes associated with that task. In CIL, the model employs a single shared classification head that contains all classes learned up to the current task and consequently expands as new classes are introduced.

In TIL, the task identity is available at inference time, enabling the model to select the corresponding task-specific head for prediction. However, the task identity is often unavailable in real-world applications, making TIL less practical in such scenarios. In contrast, CIL is task-agnostic, requiring the model to classify each input sample into one of the classes learned across all tasks. Both TIL and CIL are afected by catastrophic forgetting, where knowledge acquired from previous tasks is degraded as the model learns new tasks. In addition, CIL is susceptible to classifier bias as discussed in subsection 2.2, whereby the classification layer tends to favor recently learned classes over those encountered in earlier tasks.

## 3.2 Stability-Plasticity-Aware Region Selection Framework

Our proposed framework, SPARS, mitigates CF by combining local parameter regularization with global parameter isolation. In contrast to existing isolation-based methods that allocate dedicated parameter regions to individual tasks, SPARS performs isolation at the global level by identifying a single shared region containing the most important parameters across all tasks. This design enables the model to preserve previously acquired shared knowledge more efectively while reducing the risk of network saturation caused by continuously allocating separate parameter regions to new tasks. Unlike conventional regularization-based methods, which generally impose constraints on a broad set of network parameters, SPARS applies regularization locally to specific parameter regions according to their spatial proximity to important parameters. This localized regularization provides a soft constraint on these regions while retaining sufficient flexibility for adaptation to new tasks. As illustrated in Figure 1, the proposed framework consists of two main stages: (1) parameter importance estimation (Section 3.2.1), which assigns an importance score to each parameter, and (2) importance-based region partitioning (Section 3.2.2), which uses these importance scores to divide the network into three importance-aware regions. The proposed QDA approach for mitigating bias at inference time in the CIL setting is discussed in Section 3.2.3. Our code is available at https://github.com/teifaldaajani1999-sys/SPARS.git

![](images/918b1417184e5cf6e648d4397b4897b69df28a33b838810c6abd23d46f9fbd33.jpg)  
Fig. 1: Overview of the SPARS Framework. The framework consists of parameter importance estimation, Top-k parameter selection, partitioning model parameters into winners (W), neighboring (N), and non-important regions, followed a region-aware parameter adaptation strategy.

## 3.2.1 Parameter Importance Estimation

Parameter importance estimation quantifies the sensitivity of a neural network’s output to perturbations of its parameters, thereby identifying parameters that are critical for preserving previously acquired knowledge. Specifically, the importance of a parameter reflects the extent to which a small change in that parameter alters the model’s output behavior. Parameters with high importance scores are considered more critical to the current solution, whereas parameters with low scores are less influential and can therefore be adapted more freely. This distinction is particularly useful in continual learning, where preserving parameters that encode previously acquired knowledge must be balanced against the need to adapt the model to new tasks. Moreover, the overparameterization of deep neural networks provides multiple parameter configurations that can achieve comparable performance. SPARS exploits this flexibility by selectively preserving highly important parameters while allowing less important parameters to adapt to subsequent tasks, thereby improving the stability–plasticity trade-of without requiring access to data from previously learned tasks.

One common approach to estimating parameter importance in CL is the Fisher Information Matrix (FIM) [6]. From a Bayesian perspective, the model parameters after training encode the knowledge captured by the posterior distribution. To estimate how much information each parameter retains, this distribution is analyzed. However, because computing the posterior distribution directly is computationally expensive, it is commonly approximated by a Gaussian distribution with mean and variance, where the FIM provides an estimate of the inverse parameter variance. This approximation helps to evaluate the importance of each parameter for the task. The gradient of the loss function of task T, after training, is computed with respect to each parameter. A higher magnitude of the gradient indicates greater significance of the parameters, as denoted by the following equation:

$$
F _ { i } \approx \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \left( \frac { \partial \log p ( y _ { n } | x _ { n } , \theta ) } { \partial \theta _ { i } } \right) ^ { 2 }\tag{1}
$$

where $F _ { i }$ denotes the estimated importance of parameter $\theta _ { i }$ , N is the number of samples, $x _ { n }$ and $y _ { n }$ are the n-th input and its corresponding label, respectively, $p ( y _ { n } \mid x _ { n } , \theta )$ represents the model’s predicted probability for the correct class, and ∂ log p( $y _ { n } | x _ { n } , \dot { \theta } )$ measures the sensitivity of the log-likelihood with respect to $\theta _ { i }$

However, empirical FIM-based importance estimation typically requires access to labeled training examples. In real-world scenarios, access to the original training set may be infeasible due to privacy concerns, storage limitations, or online learning settings where past data are no longer available. This creates a bottleneck for our framework. We therefore propose a confidence-based importance score that estimates parameter importance without requiring ground-truth labels. Instead of measuring the sensitivity of the loss with respect to each parameter, this score measures the sensitivity of the model’s confidence in its highest probability predicted class to small parameter perturbations. Parameters that strongly afect the model’s confidence are considered more important, as they have a larger influence on the model’s prediction behavior. This formulation is illustrated in the equation below.

$$
C S _ { i } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \left( \frac { \partial F ( x _ { n } ; \theta ) } { \partial \theta _ { i } } \right) ^ { 2 }\tag{2}
$$

where $C S _ { i }$ denotes the estimated importance of parameter $\theta _ { i } , ~ N$ is the number of unseen samples used for evaluation, $x _ { n }$ is the n-th input sample, $F ( x _ { n } ; \theta )$ represents the logit corresponding to the model’s most confident class, and $\frac { \partial F ( x _ { n } ; \theta ) } { \partial \theta _ { i } }$ measures the sensitivity of this confidence score with respect to $\theta _ { i }$

## 3.2.2 Importance-Based Region Partitioning

To mitigate catastrophic forgetting while maintaining the stability–plasticity trade-of, the importance scores are utilized to partition the network into three distinct regions. Unlike existing continual learning methods that treat the entire network uniformly, our approach divides the network into three shared regions across tasks: The winning parameter region, the neighboring parameter region, and the non-important parameter region. Each region is then handled diferently according to its role in preserving prior knowledge and adapting to new tasks.

The motivation behind this design is based on the observed distribution of Fisher values, or importance scores, shown in Figure 2, which exhibits a long-tailed distribution in which only a small fraction of parameters receives high importance scores. This observation supports the hypothesis that model parameters do not contribute equally to preserving learned knowledge. Accordingly, we define the important region by selecting the top-k parameters with the highest importance scores across the model. To preserve the spatial structure of the learned representation, we define a neighboring region that consists of parameters adjacent to those in the important region, while all remaining parameters are assigned to the non-important region. Thus, the first region contains the most important parameters from previous tasks, the second region includes parameters spatially adjacent to these important parameters, and the third region consists of parameters considered non-important.

![](images/c0e86aa71e1a551d16d5a48b902e849a87fb9e41d7001156479e810bd859241a.jpg)  
Fig. 2: Distribution of Fisher importance scores across all model parameters, showing a long-tailed distribution in which only a small subset of parameters has high importance.

Let $I _ { i }$ denote the importance score of parameter $\theta _ { i } ,$ and let $M _ { i }$ denote its update mask coeficient. We define the winning-parameter set $\mathcal { W } _ { k }$ as the set of parameters with the top-k highest importance scores. We also define $\mathcal { N } ( R _ { 1 } )$ as the set of parameters neighboring the important region $R _ { 1 }$ . The parameters are partitioned into three regions as follows:

$$
\begin{array} { r } { \mathcal { R } ( \theta _ { i } ) = \left\{ \begin{array} { l l } { R _ { 1 } , \theta _ { i } \in \mathcal { W } _ { k } , } \\ { R _ { 2 } , \theta _ { i } \in \mathcal { N } ( R _ { 1 } ) , } \\ { R _ { 3 } , \mathrm { o t h e r w i s e } } \end{array} \right. } \end{array}
$$

where $R _ { 1 }$ represents the important region, $R _ { 2 }$ represents the neighboring region, and $R _ { 3 }$ represents the non-important region. The update mask coeficient $M _ { i }$ is then assigned according to the region of each parameter:

$$
M _ { i } = \left\{ \begin{array} { l l } { 0 , } & { \theta _ { i } \in R _ { 1 } , } \\ { \alpha _ { i } , } & { \theta _ { i } \in R _ { 2 } , \quad \mathrm { ~ w h e r e ~ } 0 < \alpha _ { i } < 1 . } \\ { 1 , } & { \theta _ { i } \in R _ { 3 } } \end{array} \right.
$$

A value of $M _ { i } = 0$ indicates that the parameter is frozen, whereas $M _ { i } = 1$ indicates that the parameter can be freely updated. Intermediate values, $0 < \alpha _ { i } < 1$ , represent partial update freedom, where the degree of regularization varies according to the parameter importance.

Important Parameter Region (Winning Parameters) This region contains the parameters that receive the highest importance scores for previously learned tasks and must remain stable during the training of new tasks. To identify this region, we define top-k as a hyperparameter that selects the most important weights, i.e., those with the highest importance values, which strongly influence the model’s behavior on previous tasks. This region is frozen because it encodes essential knowledge that the model must preserve; therefore, the parameters within this region are not updated during the learning of a new task. This prevents a significant performance drop on previous tasks after learning subsequent tasks. Overall, this region follows the principle of isolation-based methods; however, it operates globally rather than locally. Specifically, the important parameters accumulated across tasks are consolidated into this region and isolated from further updates.

Neighboring Parameter Region (Latently Influential Parameters) This region contains the neighboring parameters of the winning ones (i.e., the highly important parameters). For each winning parameter $\theta _ { o , i , h , w }$ , the neighboring parameters are selected from the same convolutional kernel by keeping the output and input channel indices (o and i) fixed, while allowing the kernel coordinates (h and w) to vary by at most one position, resulting in the eight adjacent parameters. Although these weights may not have high importance scores individually, they are closely connected to and influenced by the winning weights due to their shared input context and joint contribution to the activation of the same neuron. Therefore, regardless of their individual importance values, they still contribute to neuron activation when connected to the winning parameters. This region is subject to a regularization penalty that prevents the parameters from deviating excessively from their previous values, thereby keeping them close to the optimal configuration learned for prior tasks. Moreover, this constraint enables the neighboring parameters to preserve the influence of both the winning parameters and previously learned tasks, while still allowing limited flexibility for adaptation to new tasks. Consequently, these parameters act as a bridge that facilitates knowledge transfer in both backward and forward directions. This region follows the principles of regularization-based approaches, but applies them selectively to specific parameters rather than uniformly across the entire network.

Non-Important Parameter Region (Plastic Parameters) This region includes parameters with low importance scores, indicating that changes to these parameters have minimal impact on previously learned tasks. Consequently, this region consists of all remaining parameters and provides high plasticity, while the previous regions contribute to maintaining stability. These parameters can be updated freely during learning, allowing the model to adapt to new tasks without compromising the stability preserved by the first two regions.

The three regions described above collectively define the proposed partitioning strategy. Algorithm 1 summarizes the complete framework, including the region partitioning stage after learning task $t _ { i }$ and the region-aware adaptation (fine-tuning) stage for task $t _ { i } + 1$

Algorithm 1 Importance-Based Region Partitioning and Region-Aware   
Adaptation   
Require: Trained model after task $t _ { i } \in T$ with parameters θ; K: Top-K hyperparameter;   
regularization weight λ.   
1: Phase 1: Region Partitioning After Task $t _ { i }$   
2: for all $p \in \theta$ do   
3: Compute $F ( p )$ // The importance function   
4: end for   
5: L ← Sorted parameters in descending order of $F ( p )$   
6: $W  L [ 1 . . K ]$ // Winning Parameters   
7: $N \gets \emptyset$ // Neighboring Parameters   
8: for all $w _ { i j } \in W$ do   
9: if w<sub>ij</sub> in a convolutional layer then   
10: $N \gets N \cup \{ w _ { i \pm 1 , j } , w _ { i , j \pm 1 } , w _ { i \pm 1 , j \pm 1 } \} / /$ Set of eight spatially adjacent parameters.   
11: end if   
12: end for   
13: Phase 2: Region-Aware Adaptation   
14: for all $p \in \theta$ do   
15: if $p \in W$ then   
16: p.grad $ 0$ // Freeze winning parameters   
17: else if $p \in N$ then   
18: p.grad $ \operatorname { g r a d } + ( \lambda \cdot F ( p ) \cdot ( p _ { \mathrm { o l d } } - p _ { \mathrm { n e w } } ) ^ { 2 } )$ // Soft regularization   
19: else   
20: p.grad ← grad // Full plasticity   
21: end if   
22: end for

Region Accumulation Across Tasks The regions are updated cumulatively across tasks. Specifically, the winner region is defined as the union of winner parameters identified from all previously learned tasks and remains frozen during training on the current task. Likewise, the neighboring region is constructed as the union of neighboring parameters identified across previous tasks. If a parameter is identified as a neighbor in multiple tasks, it is stored only once, and its associated weight is set to the maximum value observed across those tasks. Parameters are assigned exclusively to either the winning region or the neighboring region. Therefore, if a parameter belongs to both sets, it is assigned to the winner region. This design maintains a clear separation between parameter regions throughout incremental learning.

## 3.2.3 Mitigating Classifier Bias in CIL

Classifier bias is a well-known issue in class-incremental learning, where the model tends to favor recently learned classes over previously learned ones. This bias primarily arises from the imbalance between old and new class samples in replay-based methods. In memory-free methods, where samples from previous tasks are unavailable, the absence of old data makes the problem even more challenging.

To alleviate classifier bias within the proposed memory-free framework, we replace the conventional fully connected classifier during inference with a discriminant classi fier based on Mahalanobis distance. Specifically, the backbone network is used solely as a feature extractor to produce discriminative feature representations, while the final FC classification layer is discarded. Classification is then performed using the extracted features together with the estimated class means and covariance matrices. We investigate both LDA and QDA as the discriminant classifiers [59]. LDA assumes a shared covariance matrix across classes, whereas QDA estimates an individual covariance matrix for each class, allowing more flexible decision boundaries. The LDA discriminant function is defined as follows:

$$
\delta _ { k } ( x ) = x ^ { T } \Sigma ^ { - 1 } \mu _ { k } - { \frac { 1 } { 2 } } \mu _ { k } ^ { T } \Sigma ^ { - 1 } \mu _ { k } + \log P ( y = k )\tag{3}
$$

where x denotes the input feature vector, $\mu _ { k }$ is the mean feature vector of class $k , \Sigma$ is the pooled covariance matrix shared across all classes, and $P ( y = k )$ represents the prior probability of class k. Unlike LDA, QDA estimates a separate covariance matrix for each class, leading to the following discriminant function.

$$
\delta _ { k } ( x ) = - \frac { 1 } { 2 } \log | \Sigma _ { k } | - \frac { 1 } { 2 } ( x - \mu _ { k } ) ^ { T } \Sigma _ { k } ^ { - 1 } ( x - \mu _ { k } ) + \log \pi _ { k }\tag{4}
$$

where x denotes the input feature vector, $\mu _ { k }$ is the mean feature vector of class $k , \Sigma _ { k }$ is the covariance matrix of class $k , \ | \Sigma _ { k } |$ denotes the determinant of the covariance matrix $\Sigma _ { k }$ , and $\pi _ { k }$ represents the prior probability of class k. The comparative performance of LDA and QDA is evaluated in section 4.2.2

## 4 Experiments

## 4.1 Experimental Setup

Datasets Following recent work on continual learning, we use two widely adopted datasets, CIFAR-10 [60] and TinyImageNet [61], for both CIL and TIL settings. We use the standard protocol, in which each dataset is partitioned into multiple tasks, each containing a distinct, non-overlapping subset of classes. Table 1 summarizes the dataset statistics and the specific splits used in our experiments. CIFAR-10 consists of 60,000 RGB images of size 32×32 pixels, split into 50,000 training samples and 10,000 test samples. Each of the 10 classes contains 5,000 training images and 1,000 test images, with no overlap between classes. Following the standard continual learning setting, CIFAR-10 is divided into five tasks, each containing two classes, for both TIL and CIL scenarios [60]. TinyImageNet is a subset of the larger ImageNet Large Scale Visual Recognition Challenge (ILSVRC) dataset [61]. It includes 200 classes, each with 500 training samples and 100 validation/test samples (50 each). Each image is 64×64 pixels. It is typically divided into 10 tasks, each containing 20 classes for use in both TIL and CIL scenarios.

Table 1: Dataset statistics and continual learning task splits used in our experiments.
<table><tr><td>Dataset</td><td>Train</td><td>Test</td><td>TIL and CIL Splits</td></tr><tr><td>CIFAR-10</td><td>50,000</td><td>10,000</td><td>5 tasks, each with 2 classes</td></tr><tr><td>TinyImageNet</td><td>100,000</td><td>10,000</td><td>10 tasks, each with 20 classes</td></tr></table>

Evaluation Metrics The performance is evaluated using two standard metrics that capture complementary aspects of CL namely, Average Incremental Accuracy (AIA) and Average Forgetting (AF). AIA measures the model’s average classification accuracy across all incremental learning steps. At each step, accuracy is computed over all tasks encountered up to that point, and AIA is obtained by averaging these accuracies across all steps. Equation 5 defines the AIA metric.

$$
\mathrm { A I A } = \frac { 1 } { T } \sum _ { i = 1 } ^ { T } A A _ { i }\tag{5}
$$

where $T$ denotes the total number of tasks, and $A A _ { i }$ denotes the average accuracy on task i.

Average forgetting is a direct measure of catastrophic forgetting that measures how much performance on previously learned tasks decreases after the model learns subsequent tasks. For each task i, the forgetting at the final step is defined as the diference between its maximum previously achieved accuracy, ma $\mathsf { \bar { t } } \in \{ i , . . . , T - 1 \} ^ { a _ { t , i } }$ , and its accuracy after the final task $_ { a _ { T } , i }$ as follows:

$$
f _ { i } = \operatorname* { m a x } _ { t \in \{ i , . . . , T - 1 \} } a _ { t , i } - a _ { T , i }\tag{6}
$$

The AF is defined across all previously learned tasks as in Equation 7, where the last task is excluded because there is no subsequent learning step after it from which forgetting can be measured.

$$
A F = { \frac { 1 } { T - 1 } } \sum _ { i = 1 } ^ { T - 1 } f _ { i }\tag{7}
$$

Implementation Details Following previous work [4, 26, 62, 63], we use ResNet18 [64] as the backbone architecture for all experiments, and it is trained from scratch without pre-training. For TIL, we adopt a multi-head setup with five heads for CIFAR-10 and ten heads for TinyImageNet, where each head corresponds to a specific task. For CIL, we use a single-head classifier that starts with the initial number of classes and expands incrementally as new tasks are introduced. Following [26], we apply standard data augmentation techniques, including random cropping and random horizontal flipping, to the training data for both CIFAR-10 and TinyImageNet. The framework is trained using the cross-entropy loss function and optimized with SGD optimization. The learning rate is set to $1 0 ^ { - 5 }$ for the backbone, while the classifier head is trained with a learning rate of 0.9 to allow faster adaptation to newly introduced tasks. The weight decay is set to $5 \times 1 0 ^ { - 9 }$ , with a batch size of 32 in all experiments. For the CL configuration, the Fisher regularization coeficient is fixed at λ = 2. The top-k value is set to 200K in the TIL setting. In the CIL setting, the first task uses 900K, while 300K is used for all subsequent tasks. Experiments are conducted on a single NVIDIA A100 GPU.

Baselines To assess the efectiveness of SPARS, we compare SPARS against diferent categories of continual learning methods: regularization-based methods (oEWC [29], SI [7], and MAS [8]), isolation-based methods (SupSup [36], WSN [38], SPG [65], and IBM [37]), and hybrid methods (EDD [50] and HAL [41]). Even though our framework is data-free, we also compare it with memory replay-based methods (iCaRL [2], DER [26], BiC [14], WA [18], TPCIL [66], Mnemonics [67], MER [68], GEM [19], and a-GEM [20], ) and pseudo-replay methods (DGM [24], DGR [22], and MeRGAN [23]). Among the replay-based methods, iCaRL, BiC, and WA are baselines that address the bias issue in the CIL setting. iCaRL relies on the nearest mean of exemplars to classify samples, BiC adopts a lightweight layer to correct the bias through learnable parameters, and WA aligns the weights of the classifier by aligning the norms of the weight vectors of new classes with those of old classes. We also compare SPARS against an upper-bound JOINT baseline obtained by jointly training on all tasks simultaneously. Brief descriptions of these baselines are provided in Appendix Appendix A.

## 4.2 Main Results and Analysis

Table 2 presents the average incremental accuracy of the proposed method and the baseline methods under both TIL and CIL settings. Replay-based methods achieve the highest performance on CIFAR-10 and rank second on TinyImageNet. However, their efectiveness is constrained by the fixed memory-bufer capacity, which limits the number and diversity of stored samples as the number of classes increases. Although various memory management strategies can reduce this limitation, replaybased methods remain dificult to scale to long task sequences. Under these conditions, pseudo-replay methods can perform comparatively well by generating samples from previously learned classes. In contrast, regularization-based methods show lower performance on both benchmarks, falling behind the other continual learning categories by a considerable margin. This performance gap is more pronounced as the number of tasks and classes increases, because estimating and preserving parameter importance becomes increasingly dificult. The proposed SPARS method achieves AIA values of 89.35% and 61.61% on CIFAR-10 and TinyImageNet, respectively, under TIL. On CIFAR-10, SPARS remains competitive with the leading replay-based methods despite requiring no stored samples from previous tasks. On TinyImageNet, SPARS outperforms all compared categories and achieves approximately 76% of the JOINT upper bound, demonstrating strong performance under class-scarce conditions. Furthermore, SPARS outperforms pseudo-replay methods without relying on synthetic data generation, thereby avoiding potential noise and visual artifacts that associated with generated samples and degrade representation quality during incremental learning.

Under the CIL setting, all methods exhibit a noticeable performance decline when moving from CIFAR-10 to TinyImageNet, reflecting the greater dificulty of the latter benchmark. TinyImageNet contains 20 times more classes than CIFAR-10, making the classification problem substantially more diverse. Moreover, it provides only 500 training examples per class, compared with 5,000 examples per class in CIFAR-10. Consequently, the performance diferences among continual learning categories become smaller on TinyImageNet. Among the compared methods, DGM achieves the highest performance on CIFAR-10,while EDD achieves the second best performance on TinyImageNet. The DGM performance can be attributed to its pseudo-replay mechanism, which helps preserve knowledge of previously learned classes through generated samples. Nevertheless, SPARS achieves 59.28% and 36.13% AIA on CIFAR-10 and TinyImageNet, respectively, and surpasses DGM on TinyImageNet by a substantial margin without relying on synthetic replay samples.

Overall, SPARS demonstrates strong performance under both TIL and CIL on TinyImageNet, indicating that it can acquire new knowledge while retaining a substantial portion of previously learned knowledge. It also maintains a favorable stability–plasticity trade-of under long task sequences and challenging benchmark conditions. These results further indicate that methods performing well on CIFAR-10 do not necessarily maintain the same performance on TinyImageNet, where the larger number of classes, the smaller number of samples per class, and the longer task sequence may contribute to a more challenging continual learning setting. Therefore, SPARS appears particularly suitable for continual learning scenarios in which data are limited, and tasks arrive sequentially over time.

## 4.2.1 Efect of the Confidence-Based Importance Score

We evaluate the efectiveness of the proposed CS importance estimation method on CIFAR-10 and TinyImageNet under both TIL and CIL settings.

As shown in Table 2, SPARS-CS achieves performance competitive with SPARS-FIM on both CIFAR-10 and TinyImageNet under TIL and CIL, despite not requiring access to training data or class labels for importance estimation. These results indicate that efective parameter importance estimation can be achieved without labelled data, thereby improving the robustness and applicability of SPARS in settings where data access is limited. Moreover, the Confidence Score is not specific to SPARS and could potentially be integrated into other continual learning methods that rely on parameter importance estimation.

## 4.2.2 CIL Discriminant Classifier Analysis

To investigate the bias observed in the CIL setting, alternative inference-time classification methods are evaluated using feature representations extracted from the trained backbone network. Unlike the conventional classifier head, these classifiers utilize the network solely as a feature extractor. Consequently, they provide an opportunity to address bias without requiring norm or logit calibration and without modifying the training dynamics. Therefore, Mahalanobis distance-based classifiers, namely LDA and QDA, are investigated in this section as inference-time alternatives. Their performance is first evaluated on CIFAR-10 and TinyImageNet, followed by an analysis of their strengths and limitations in continual learning scenarios. Figure 3 shows the AIA results on CIFAR-10 for LDA and QDA. Both LDA and QDA begin with similar AA on the first task. However, starting from the second task, a gap of approximately 30 percentage points appears in favor of QDA, and this gap remains consistent across the subsequent tasks. Overall, the AIA of QDA is higher than that of LDA by 17.75 percentage points. On TinyImageNet, QDA and LDA achieve similar AA during the first two tasks. Afterward, QDA consistently outperforms LDA by approximately 8 percentage points. In addition, the results show greater stability for QDA on both benchmarks. This is because QDA can create more flexible decision boundaries by considering individual covariance matrices for each class, which helps when feature distributions drift over time. Since these boundaries are curved rather than rigid, unlike LDA, they remain more adaptable to new feature space locations. Especially at Task 10, the nonlinear overlap between classes makes LDA struggle to separate them efectively. Figure 4 illustrates the feature space behavior of LDA and QDA under the CIL setting after diferent sequential tasks. The visualizations were generated using PCAprojected feature representations, with each color representing a diferent class and the shaded regions indicating the corresponding decision boundaries. The figure shows the decision boundaries of LDA and QDA for three classes: the first class from Task 2, and the second and third classes from Tasks 6 and 10, respectively. The arrows indicate feature drift between earlier and later task states. While noticeable drift occurs for previously learned classes, the class clusters remain partially separable in the learned feature space. LDA fails to capture this evolving geometry, whereas QDA adapts more efectively to nonlinear distributional changes.

Table 2: Results of the proposed method compared with diferent baselines in terms of AIA. Bold values indicate the best performance, while “-” denotes experiments that were not reported in the reference papers. All baseline results are reported from [62], while for the baselines marked with “\*”, their results are taken from the cited papers indicated alongside them.
<table><tr><td rowspan="2">Category</td><td rowspan="2">Method</td><td colspan="2">TIL</td><td colspan="2">CIL</td></tr><tr><td>CIFAR-10</td><td>TinyImageNet</td><td>CIFAR-10</td><td>TinyImageNet</td></tr><tr><td>JOINT</td><td>Upper bound</td><td>98.45</td><td>82.07</td><td>92.02</td><td>59.45</td></tr><tr><td rowspan="3">Regularization based</td><td>oEWC</td><td>68.29</td><td>19.20</td><td>19.49</td><td>7.58</td></tr><tr><td>SI</td><td>68.05</td><td>36.32</td><td>19.48</td><td>6.58</td></tr><tr><td>MAS* [69]</td><td>一</td><td>45.08</td><td>一</td><td>27.79</td></tr><tr><td rowspan="4">Isolation based</td><td>Supsup* [37]</td><td>一</td><td>40.27</td><td>一</td><td>=</td></tr><tr><td>SPG* [37]</td><td></td><td>21.14</td><td></td><td>一</td></tr><tr><td>WSN* [37]</td><td></td><td>49.36</td><td></td><td>=</td></tr><tr><td>IBM* [37]</td><td></td><td>52.38</td><td>一</td><td>–</td></tr><tr><td rowspan="8">Memory replay</td><td>MER</td><td>93.61</td><td>48.64</td><td>57.74</td><td>9.99</td></tr><tr><td>GEM</td><td>92.16</td><td></td><td>26.20</td><td></td></tr><tr><td>a-GEM</td><td>89.48</td><td>25.33</td><td>22.67</td><td>8.06</td></tr><tr><td>DER</td><td>93.40</td><td>51.78</td><td>70.51</td><td>17.75</td></tr><tr><td>TPCIL</td><td>96.67</td><td>48.41</td><td>58.72</td><td>25.03</td></tr><tr><td>iCaRL</td><td>88.22</td><td>31.55</td><td>47.55</td><td>9.38</td></tr><tr><td>BiC WA</td><td>89.91</td><td>32.94</td><td>49.18</td><td>11.09</td></tr><tr><td>UER* [21]</td><td>94.21 93.89</td><td>37.12</td><td>53.76</td><td>15.42</td></tr><tr><td rowspan="3">Pseudo memory replay</td><td></td><td></td><td>44.81 61.32</td><td>68.87</td><td>13.61</td></tr><tr><td>DGM</td><td>89.91</td><td></td><td>71.94</td><td>28.45</td></tr><tr><td>DGR</td><td>79.86</td><td>38.41</td><td>49.69</td><td>17.38</td></tr><tr><td rowspan="3">Hybrid approach</td><td>MeRGAN</td><td>84.76</td><td>58.32</td><td>66.76</td><td>23.86</td></tr><tr><td>EDD* [50]</td><td></td><td></td><td>55.13</td><td>30.11</td></tr><tr><td>HAL* [50]</td><td></td><td></td><td>19.99</td><td>8.89</td></tr><tr><td rowspan="3"></td><td>SPARS-FIM (Our)</td><td>89.35</td><td>61.61</td><td>59.28</td><td>36.13</td></tr><tr><td>SPARS-CS (Our)</td><td>89.70</td><td>60.29</td><td>59.37</td><td>36.12</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/eb73943d1247b6fd9a16bd42a0c567194461440dc81dd9a0b8492dfd5d460ecf.jpg)  
(a) CIFAR-10

![](images/4b9e65c9cdac0007d50b71ea2fe3ba4de5a1ada0443fd9554ef602c32826108f.jpg)  
(b) TinyImageNet  
Fig. 3: Per-task Average Accuracy of QDA and LDA across the tasks for CIL on (a) CIFAR-10 and (b) TinyImageNet.

Overall, these observations, with the amount of feature drift, suggest that the issue in CIL is not solely caused by feature drift, as the learned representations retain partial robustness. Instead, the results indicate that a significant portion of the performance degradation originates from the classifier method itself, which is consistent with the behavior shown in the figures. The improved separation obtained with QDA suggests that part of the degradation in CIL arises from limitations of the classifier head rather than from complete degradation of the learned representations.

![](images/b4e618b605d61e6d4d1c25b582e5297537dbe42172e047beeb6d4fb77052ddda.jpg)  
Fig. 4: PCA visualization of feature embeddings extracted from the backbone network after tasks 2, 6, and 10 under the CIL setting. Light-colored samples denote earlier feature locations, while darker samples represent their updated positions after subsequent tasks. Arrows visualize the direction and magnitude of feature drift over time.

## 4.3 Ablation Studies

Efect of Top-k Values To investigate the efect of the top-k parameters, we conducted extensive experiments using the following top-k values: [100k, 300k, 500k, 700k]. Note that the top-k value for the first task remains fixed at 900k, as used in the main results. Preserving this setting allows for a fair comparison among the diferent top-k values. Table 3 shows that the 500K configuration achieves the highest AIA among all evaluated configurations. It also produces the lowest AF, indicating improved resistance to catastrophic forgetting. Overall, these results suggest a favorable trade-of between stability and plasticity. Although 500k achieves the best overall performance, the variation in AIA across the evaluated top-k values remains relatively small. This suggests that the proposed framework is generally robust to moderate changes in the top-k setting while maintaining a favorable stability–plasticity trade-of. As shown in Figure 5, increasing the top-k value also increases the overlap of top-k parameters across tasks. Therefore, the number of unique top-k parameters remains lower than the total accumulated top-k parameters across all tasks, indicating the existence of shared winner (important) parameters among tasks. This observation directly suggests that these frozen parameters contribute to performance across task sequences, indicating a degree of forward knowledge transfer. In addition, only a subspace of the model parameters is preserved through freezing the top-k (winner) parameters and regularizing neighboring parameters. Furthermore, the percentage of this protected parameter subspace does not exceed 27% of the total model parameters. As a result, a large portion of the network remains available for adaptation, indicating that the model still retains suficient capacity to learn additional tasks without saturation.

Table 3: AIA and AF across diferent top-k sizes for CIL on TinyImageNet. Higher AIA and lower AF indicate better performance across all 10 tasks. The best result is achieved at top-k = 500k.
<table><tr><td>Top-k</td><td>100K</td><td>300K</td><td>500K</td><td>700K</td></tr><tr><td>AIA</td><td>36.02</td><td>36.13</td><td>36.14</td><td>36.11</td></tr><tr><td>AF</td><td>15.86</td><td>15.48</td><td>14.60</td><td>14.77</td></tr></table>

![](images/f698481bca611da3b1f182b65797a615b9fd3d1deac57313784817489020678a.jpg)  
Fig. 5: Comparison between the number of unique top-(k) parameters across 10 tasks on TinyImageNet under CIL setting, the number of unique neighboring parameters accumulated across tasks, and the overlap among the top-(k) parameters identified across tasks. The blue curve shows the ratio of the combined top-(k) and neighboring parameters to the total number of model parameters.

Ablation on Neighboring Regions Since neighboring parameters represent one of the key components of SPARS, this experiment investigates their contribution to knowledge preservation under diferent regularization strengths. Specifically, we further explored the efect of the neighbors and the strength of regularization by investigating the following values of the regularization coeficient λ: [0, 2, 100, 1000] under the CIL setting on TinyImageNet, with top-k set to 100k. Figure 6 presents the AIA and AF across diferent λ values. AF decreases as λ increases, indicating that stronger regularization leads to better preservation of previous knowledge. At the same time, AIA remains stable and slightly improves as λ increases, suggesting that stronger regularization does not negatively afect plasticity. Although adding neighboring parameters results in limited changes in AIA, diferences are observed in AF, indicating their contribution to knowledge retention. To better understand this behavior, the distributions of the top-k winner parameters and their neighboring parameters were further analyzed.

Distribution of importance Scores Figure 7 shows that the importance values of the top-k parameters exhibit a highly skewed distribution with significantly larger importance values. Especially in the tail of the distribution, only a small number of parameters possess very high importance values, indicating that the learning process across all 10 tasks is sparse. On the other hand, the neighbors exhibit lower importance values and a denser distribution. Nevertheless, the two distributions show considerable overlap, suggesting that neighboring parameters are also important to some extent, which supports our hypothesis that neighbors contain meaningful information and contribute to knowledge. This means that even after removing the exact overlap between the neighbor list and the top-k list, the importance values of the remaining neighbors still remain close to many of the selected top-k parameters. This observation indicates that the importance estimation method can capture the importance of neighboring parameters and suggests that knowledge in CNNs is structurally distributed, where neighboring parameters share importance and collaboratively support learning. Consequently, learning in convolutional layers may not be entirely sparse at the kernel level, particularly when using relatively small convolutional kernels (e.g., 3×3). We further explored the ratio of the exact overlap and found that more than 80% of the top-k list intersects with the neighbor set. This partially explains why the efect of neighboring parameters remains limited in AIA. However, it still afects AF, which suggests that neighbor regularization mainly improves stability by supporting the top-k parameters. In this context, the ratio between the regularization gradient and the cross-entropy gradient was further analyzed. We found that the regularization gradient is approximately 0.00138% of the cross-entropy gradient, making it too weak to efectively dominate the optimization process in favor of preserving neighbors. This behavior is partially influenced by λ; therefore, increasing λ improves AF. On the other hand, the small importance values of the neighbors also contribute to weak regularization strength.

![](images/e72849cf48ee4b713d3bd56a0d204ff15748ece9bc9ef13966d8b042ed1c974c.jpg)  
Fig. 6: AIA and AF under diferent values of λ on TinyImageNet in the CIL setting: λ = 0 (no neighbors),λ = 2, λ = 100, and λ = 1000.

Layer-Wise Parameter Distribution Figure 8 shows that the number of neighbors in the early layers is smaller than the number of top-k parameters, which indicates a high exact overlap between them before overlap removal. In these layers, the kernels contain a higher proportion of top-k parameters compared to neighbors, suggesting that a large fraction of the kernel parameters contribute significantly to learning lowlevel image representations and therefore receive high Fisher importance scores. In contrast, the deeper layers contain a larger neighboring set even after removing the overlap, which may indicate that the overlap ratio is lower in these layers and that learning becomes more sparse. In other words, not all parameters contribute equally in the deeper layers. Since these layers are no longer responsible for capturing fine image details, they instead focus on learning more abstract representations.

![](images/4dc1afa598eacccfce92bda1642d24b01fda7fbfdc0c0d1d7ecff62c264ea7de.jpg)

![](images/6ca99b90d34e38c32bbe7208def50bee57a96b27bdb022cdcf481fa3743da453.jpg)

![](images/43bbecf809016cca1a916702f20c9e12f159c729a1d77e0bf2167bccfa192841.jpg)

![](images/898a2eacbaebfffaed6c588d944201870935f2a3cfb36a16bb519ec01a5e6f48.jpg)  
Fig. 7: Distributions of FIM and CS (log10 scale) for the top-k winner parameters and their neighboring parameters on CIFAR-10 in the CIL setting.

Overall, these observations help explain why the impact of neighboring parameters is less visible. In addition, the relatively small kernel size (3×3) may further reduce the observable efect of neighboring parameters. The high overlap ratio between the winner and neighboring parameters further supports the importance of neighboring parameters and provides additional insight into the behavior of convolutional layers, which is consistent with our hypothesis. On the other hand, our decision to handle the overlap by removing it from the neighbor list may further reduce its impact in terms of AIA. In this context, we conducted exploratory experiments aimed at avoiding the overlap by performing local top-k selection, where any kernel containing a selected winner parameter is excluded from further top-k selection. As a result, the overlap ratio becomes zero, but at the expense of AIA, which slightly decreases because this strategy ignores the importance values during the selection process. Moreover, no remarkable improvements were observed in the ablation study.

Layer-wise Top-k vs Neighbor Distribution  
![](images/b3459e318c2ca66a04d36d4081a8cee03494e456f14396f590f52615b9c77941.jpg)  
Fig. 8: Layer-wise distribution of top-k winner and neighboring parameters across convolutional layers.

## 5 Conclusion

This work introduces a hybrid continual learning framework that mitigates catastrophic forgetting while maintaining a balance between stability and plasticity without using a memory bufer, model expansion, or requiring training data from previous tasks. The proposed SPARS framework bridges parameter isolation and parameter regularization by partitioning the network into three importance-aware regions in which each region is assigned a dedicated adaptation strategy. This region-aware adaptation attempts to obtain a better stability-plasticity trade-of than existing methods. Experimental results demonstrate superior performance on TinyImageNet under both task-incremental learning and class-incremental learning settings, as well as competitive performance on CIFAR-10. The findings also indicate that performance degradation in CIL is influenced not only by feature drift but also by classifier behavior, as evidenced by the superior performance of statistical classifiers. Moreover, the Confidence Score performs competitively with the Fisher Information Matrix, highlighting its efectiveness in privacy-sensitive and data-restricted scenarios. These findings demonstrate that importance-aware region selection provides an efective and scalable direction for memory-free continual learning in real-world applications.

## Limitations

The framework heavily relies on the importance estimation strategy and method, and it also depends on a convolutional based architecture. In this work, neighborhood relationships are investigated at the kernel parameter level using a fixed, predefined number of neighbors. Therefore, investigating diferent types of relationships, such as those at the filter or channel level, and exploring more flexible selection approaches may improve performance. In addition, the regularization term regularizes the neighbors individually based on their importance scores. Therefore, global regularization that preserves the selected neighborhood area and mitigates the overall shift may help preserve the relationships among neighboring parameters and further support the knowledge represented in these areas. Furthermore, the substantial overlap between the neighbors and the top-k parameters may indicate that the selection of the top-k parameters should follow specific criteria, such as satisfying a predefined threshold, to avoid including too many parameters that may not contribute to stability.

## Appendix A Baseline descriptions

This appendix provides brief descriptions of the baseline methods considered in our evaluation. For clarity, the baselines are organized according to their primary continual learning strategy, including regularization-based, isolation-based, hybrid, memory replay-based, and pseudo-replay methods. For each baseline, we summarize its main mechanism for preserving previously acquired knowledge and mitigating catastrophic forgetting.

## Regularization-Based Method:

• oEWC [29], Online Elastic Weight Consolidation, estimates parameter importance using the FIM in an online manner and regularizes important parameters against changes.

• SI [7], Synaptic Intelligence, estimates parameter importance by monitoring the influence of each parameter on the loss function throughout the training phase in an online manner, and accumulate these importance scores along the learning path to obtain a more robust estimate of parameter importance. The changes to important parameters are then penalized.

• MAS [8], Memory Aware Synapses, estimates the importance of model parameters in an unsupervised manner by measuring the sensitivity of the predicted output function to changes in each parameter. Changes to important parameters are then penalized.

Isolation-Based Methods:

• SupSup [36], Supermasks in Superposition, learns a task-specific subnetwork on top of a fixed, randomly initialized network. When the task identity is not available, the framework formulates task inference as an optimization problem, where the correct subnetwork is expected to produce a low-entropy output distribution.

• WSN [38], Winning SubNetworks, Finds subnetworks based on binary masks by selecting the top-p parameters that are most important for each task to construct the corresponding subnetwork. Parameter importance is determined by a learnable importance weight associated with each regular model weight.

• SPG [65], Soft-masking of Parameter-level Gradient flow, Creates subnetworks in a soft manner, where parameter importance is determined based on gradients. The gradient flow of these parameters is directly constrained during the backward pass, restricting updates to each parameter individually according to its importance score.

• IBM [37], Information Bottleneck Masked Sub-network, accumulates important information into weights based on the information bottleneck principle to create redundancy-free subnetworks. It mitigates catastrophic forgetting by freezing these weights via task-specific binary masks.

## Hybrid Methods:

• EDD [50], Expandable and Diferentiable Dual Memory, consists of two memories: one learns common features that can be shared across all tasks, while the other is task-specific and learns discriminative features. Orthogonal regularization is applied to the task-specific memory.

• HAL [41], Hindsight Anchor Learning, Uses stored samples to approximate forgetting through fine-tuning and optimize the anchor points computed for each task. These anchor points are then preserved through regularization.

## Memory Replay-Based Methods:

• iCaRL [2], Incremental Classifier and Representation Learning, utilizes KD and prototype rehearsal to preserve previously learned knowledge and classifies samples using the nearest-mean-of-exemplars rule.

• DER [26], Dark Experience Replay, Stores past samples along with their corresponding logits produced during training. Knowledge is preserved by aligning the current model outputs with the stored logits using an L<sup>2</sup> matching loss.

• BiC [14], Bias Correction, addresses the bias issue in CIL by training a correction layer with two learnable parameters on a validation set to reduce the bias toward new classes.

• WA [18], Weight Aligning, Uses stored samples and KD to preserve the discrimination of old classes, and then aligns the classifier weights as a post-processing step after training.

• TPCIL [66], Topology-Preserving Class-Incremental Learning, Uses an Elastic Hebbian Graph to model the topology of the feature space and proposes a topology-preserving loss to penalize changes in feature relationships.

• MER [68], Meta-Experience Replay, Combines experience replay with optimizationbased meta-learning. It uses the Reptile algorithm in the inner and outer loops to update the parameters. The inner loop maintains alignment within each batch, while the outer loop maintains alignment across batches, ensuring that the updates do not interfere with previous tasks while maximizing the transfer of previously learned knowledge.

• GEM [19], Gradient Episodic Memory, Uses an episodic memory of previous tasks and computes the gradients of both the current and previous tasks. It uses the gradients from previous tasks to define constraints on the current update, ensuring that it does not increase their losses. When the constraints are violated, GEM modifies the current gradient by solving a quadratic programming problem.

• a-GEM [20], Averaged GEM, simplifies GEM by replacing its multiple task-specific constraints with a single constraint based on the average loss over the episodic memory, significantly reducing computational and memory costs.

• UER [21], Uncertainty-Aware Dark Experience Replay, improves upon DER by leveraging model and data uncertainty to select representative samples through an Uncertainty-Aware Sampling strategy. To estimate uncertainty, it employs a Bayesian Neural Network. In addition, it introduces an uncertainty loss into DER to preserve the uncertainty information associated with previous tasks, thereby further mitigating catastrophic forgetting.

## Pseudo-Replay Methods:

• DGM[24], Dynamic Generative Memory, employs a generative memory module that synthesizes samples from previous classes. The method learns task-specific binary masks to control plasticity in the generator. Additionally, DGM dynamically expands the network capacity to maintain a consistent number of free parameters for each task.

• DGR [22], Deep Generative Replay, employs a generative replay strategy in which a ”scholar” model, composed of a generator and a solver, is jointly trained. The generator synthesizes samples of previously learned tasks, while the solver learns from a mixture of real data from the current task and replayed synthetic data. At each new task, the generator reconstructs the cumulative input distribution, and the solver aligns its predictions with both real labels and past solver outputs.

• MeRGANs [23], Memory Replay GANs, Uses a conditional GAN to generate samples from previous tasks and mitigates forgetting through two main mechanisms: joint training with replay and replay alignment.

Funding No funding was received for conducting this study.

Data availability The data that support the findings of this study are openly available through the cited sources and our code repository.

## Declarations

Conflict of interest The authors declare no competing interests that are relevant to the content of this article.

## References

[1] McCloskey, M., Cohen, N.J.: Catastrophic interference in connectionist networks:

The sequential learning problem. In: Psychology of Learning and Motivation vol. 24, pp. 109–165. Elsevier, ??? (1989)

[2] Rebufi, S.-A., Kolesnikov, A., Sperl, G., Lampert, C.H.: icarl: Incremental classifier and representation learning. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2001–2010 (2017)

[3] Titsias, M.K., Schwarz, J., Matthews, A.G.d.G., Pascanu, R., Teh, Y.W.: Functional regularisation for continual learning with gaussian processes. In: Proceedings of the International Conference on Learning Representations (ICLR) (2020). https://arxiv.org/abs/1901.11356

[4] Wang, A.I., Zhang, X., Yang, K., Yu, L., Li, C., Hong, L., Zhang, S., Li, Z., Zhong, Y., Zhu, J.: Memory replay with data compression for continual learning. arXiv preprint arXiv:2202.06592 (2022)

[5] Yan, Q., Gong, D., Liu, Y., Van Den Hengel, A., Shi, J.Q.: Learning bayesian sparse networks with full experience replay for continual learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 109–118 (2022)

[6] Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A.A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al.: Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences 114(13), 3521–3526 (2017)

[7] Zenke, F., Poole, B., Ganguli, S.: Continual learning through synaptic intelligence. In: Precup, D., Teh, Y.W. (eds.) Proceedings of the 34th International Conference on Machine Learning (ICML). Proceedings of Machine Learning Research, vol. 70, pp. 3987–3995. PMLR, Sydney, Australia (2017). https://doi.org/10.5555/ 3305890.3306093

[8] Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M., Tuytelaars, T.: Memory aware synapses: Learning what (not) to forget. In: Proceedings of the European Conference on Computer Vision (ECCV), pp. 139–154 (2018)

[9] Mallya, A., Lazebnik, S.: PackNet: Adding multiple tasks to a single network by iterative pruning. In: Proceedings of the European Conference on Computer Vision (ECCV) (2018)

[10] Mallya, A., Davis, D., Lazebnik, S.: Piggyback: Adapting a single network to multiple tasks by learning to mask weights. arXiv preprint arXiv:1801.06519 (2018)

[11] Serr\`a, J., Sur´ıs, D., Miron, M., Karatzoglou, A.: Overcoming catastrophic forgetting with hard attention to the task. In: Proceedings of the 35th International Conference on Machine Learning (ICML) (2018). arXiv preprint

[12] Gurbuz, M.B., Dovrolis, C.: Nispa: Neuro-inspired stability-plasticity adaptation for continual learning in sparse networks. arXiv preprint arXiv:2206.09117 (2022) https://doi.org/10.48550/arXiv.2206.09117

[13] Ven, G.M., Tuytelaars, T., Tolias, A.S.: Three types of incremental learning. Nature Machine Intelligence 4(12), 1185–1197 (2022)

[14] Wu, Y., Chen, Y., Wang, L., Ye, Y., Liu, Z., Guo, Y., Fu, Y.: Large scale incremental learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 374–382 (2019). https://doi.org/10.48550/ arXiv.1905.13260

[15] Lin, H., Shao, Y., Qian, W., Pan, N., Guo, Y., Liu, B.: Class incremental learning via likelihood ratio based task prediction. In: International Conference on Learning Representations (ICLR) (2024). https://doi.org/10.48550/arXiv.2309. 15048

[16] Chen, X., Wang, Z.: Uni-dtr: A unified framework for balancing knowledge preservation and adaptation in exemplar-free class-incremental learning. Neurocomputing 686, 133722 (2026) https://doi.org/10.1016/j.neucom.2026.133722

[17] Belouadah, E., Popescu, A.: Il2m: Class incremental learning with dual memory. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 583–592 (2019). https://doi.org/10.1109/ICCV.2019.00067

[18] Zhao, B., Xiao, X., Gan, G., Zhang, B., Xia, S.: Maintaining discrimination and fairness in class incremental learning. arXiv preprint arXiv:1911.07053 (2019) https://doi.org/10.48550/arXiv.1911.07053

[19] Lopez-Paz, D., Ranzato, M.: Gradient episodic memory for continual learning. arXiv preprint arXiv:1706.08840 (2017) https://doi.org/10.48550/arXiv.1706. 08840

[20] Chaudhry, A., Ranzato, M., Rohrbach, M., Elhoseiny, M.: Eficient lifelong learning with a-gem. In: International Conference on Learning Representations (ICLR) (2019). https://doi.org/10.48550/arXiv.1812.00420

[21] Wang, Q., Ji, Z., Pang, Y., Zhang, Z.: Uncertainty-aware enhanced dark experience replay for continual learning. Applied Intelligence 54(13–14), 7135–7150 (2024) https://doi.org/10.1007/s10489-024-05488-w

[22] Shin, H., Lee, J.K., Kim, J., Kim, J.: Continual learning with deep generative replay. arXiv preprint arXiv:1705.08690 (2017) https://doi.org/10.48550/arXiv. 1705.08690

[23] Wu, C., Herranz, L., Liu, X., Wang, Y., Weijer, J., Raducanu, B.: Memory replay GANs: learning to generate images from new categories without forgetting. arXiv preprint arXiv:1809.02058 (2018) https://doi.org/10.48550/arXiv.1809.02058

[24] Ostapenko, O., Puscas, M., Klein, T., Jahnichen, P., Nabi, M.: Learning to remember: A synaptic plasticity driven framework for continual learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11321–11329 (2019). https://doi.org/10.1109/CVPR. 2019.01158 . https://doi.org/10.1109/CVPR.2019.01158

[25] Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., Wayne, G.: Experience replay for continual learning. Advances in neural information processing systems 32 (2019)

[26] Buzzega, P., Boschini, M., Porrello, A., Abati, D., Calderara, S.: Dark experience for general continual learning: A strong, simple baseline. arXiv preprint arXiv:2004.07211 (2020) https://doi.org/10.48550/arXiv.2004.07211

[27] Ebrahimi, S., Elhoseiny, M., Darrell, T., Rohrbach, M.: Uncertainty-guided continual learning with bayesian neural networks. arXiv preprint arXiv:1906.02425 (2019) https://doi.org/10.48550/arXiv.1906.02425

[28] Chaudhry, A., Dokania, P.K., Ajanthan, T., Torr, P.H.: Riemannian walk for incremental learning: Understanding forgetting and intransigence. In: Proceedings of the European Conference on Computer Vision (ECCV), pp. 532–547 (2018)

[29] Schwarz, J., Czarnecki, W., Luketina, J., Grabska-Barwinska, A., Teh, Y.W., Pascanu, R., Hadsell, R.: Progress & compress: A scalable framework for continual learning. In: International Conference on Machine Learning, pp. 4528–4537 (2018). PMLR

[30] Nguyen, C.V., Li, Y., Bui, T.D., Turner, R.E.: Variational continual learning. arXiv preprint arXiv:1710.10628 (2017)

[31] Jung, S., Ahn, H., Cha, S., Moon, T.: Continual learning with node-importance based adaptive group sparse regularization. In: Advances in Neural Information Processing Systems, vol. 33 (2020)

[32] Tousside, B., Mohr, J., Frochte, J.: Group and exclusive sparse regularizationbased continual learning of cnns. In: The 35th Canadian Conference on Artificial Intelligence (2022)

[33] Chen, R., Chen, G., Liao, X., Xiong, W.: Class-incremental learning via prototype similarity replay and similarity-adjusted regularization. Applied Intelligence 54, 9971–9986 (2024) https://doi.org/10.1007/s10489-024-05695-5

[34] Hurtado, J., Raymond-Saez, A., Soto, A.: Optimizing reusable knowledge for <sup>´</sup> continual learning via metalearning. arXiv preprint arXiv:2106.05390 (2021)

[35] Jin, H., Kim, G.-H., Ahn, C., Kim, E.-K.: Growing a brain with sparsity-inducing generation for continual learning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 18915–18924 (2023). https://doi. org/10.1109/ICCV51070.2023.01738

[36] Wortsman, M., Ramanujan, V., et al.: Supermasks in superposition. arXiv preprint arXiv:2006.14769 (2020) https://doi.org/10.48550/arXiv.2006.14769

[37] Chen, C., Song, J., Gao, L., Shen, H.T.: Towards redundancy-free sub-networks in continual learning. arXiv preprint arXiv:2312.00840 (2023) https://doi.org/10. 48550/arXiv.2312.00840

[38] Kang, H., Yoon, J., Madjid, S.R., Hwang, S.J., Yoo, C.D.: Forget-free continual learning with soft-winning subnetworks. arXiv preprint arXiv:2303.14962 (2023)

[39] Zhu, H., Majzoubi, M., Jain, A., Choromanska, A.: Tame: Task agnostic continual learning using multiple experts. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 4139–4148 (2024)

[40] Li, H., Lin, S., Duan, L., Liang, Y., Shrof, N.B.: Theory on mixture-of-experts in continual learning. In: International Conference on Learning Representations (2025)

[41] Chaudhry, A., Gordo, A., Dokania, P.K., Torr, P., Lopez-Paz, D.: Using hindsight to anchor past knowledge in continual learning. arXiv preprint arXiv:2002.08165 (2020) https://doi.org/10.48550/arXiv.2002.08165

[42] Han, X., Guo, Y.: Continual learning with dual regularizations. In: Machine Learning and Knowledge Discovery in Databases. Research Track, pp. 619–634 (2021). https://doi.org/10.1007/978-3-030-86486-6 38

[43] Bhat, P., Zonooz, B., Arani, E.: Consistency is the key to further mitigating catastrophic forgetting in continual learning. arXiv preprint arXiv:2207.04998 (2022) https://doi.org/10.48550/arXiv.2207.04998

[44] Vahedifar, M.A., Zhang, Q.: No forgetting learning: Bufer-free continual learning classification. arXiv preprint arXiv:2503.04638 (2025) https://doi.org/10.48550/ arXiv.2503.04638

[45] Alssum, L., Hammoud, H.A.A.K., Alfarra, M., Leon Alcazar, J.C., Ghanem, B.: Forget less, retain more: A lightweight regularizer for rehearsal-based continual learning. arXiv preprint arXiv:2512.01818 (2025) https://doi.org/10.48550/ arXiv.2512.01818

[46] Manore, J., Jamkar, S., Chavan, A., Wakode, M.S.: Mitigating catastrophic forgetting using elastic weight consolidation and hybrid replay frameworks. In: 2026 International Conference on Emerging Smart Computing and Informatics (ESCI)

[47] Wu, Z., Yao, H., Jiang, H., Ma, J.: An incremental learning approach for sustainable regional isolation and integration. Computers and Electrical Engineering 111, 108911 (2023) https://doi.org/10.1016/j.compeleceng.2023.108911

[48] Siddika, F., Hossen, M.A., Mallick, T., Jannesari, A.: Sparse subspace-to-expert sharing for task-agnostic continual learning. arXiv preprint arXiv:2606.07500 (2026) https://doi.org/10.48550/arXiv.2606.07500

[49] Ebrahimi, S., Meier, F., Calandra, R., Darrell, T., Rohrbach, M.: Adversarial continual learning. In: Computer Vision – ECCV 2020, pp. 386–402. Springer, ??? (2020). https://doi.org/10.1007/978-3-030-58621-8 23

[50] Moon, H.-J., Cho, S.-B.: Expandable and diferentiable dual memories with orthogonal regularization for exemplar-free continual learning. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, pp. 24441–24448 (2026). https://doi.org/10.1609/aaai.v40i29.39626

[51] Zhu, F., Zhang, X.-Y., Cheng, Z., Liu, C.-L.: Pass++: A dual bias reduction framework for non-exemplar class-incremental learning. arXiv preprint arXiv:2407.14029 (2024) https://doi.org/10.48550/arXiv.2407.14029

[52] Shi, W., Ye, M.: Prospective representation learning for non-exemplar classincremental learning. In: Advances in Neural Information Processing Systems 37 (NeurIPS 2024), pp. 995–1018 (2024). https://doi.org/10.52202/079017-0030

[53] Zhang, J., Zhang, J., Ghosh, S., Li, D., Tasci, S., Heck, L., Zhang, H., Kuo, C.- C.J.: Class-incremental learning via deep model consolidation. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 1131–1140 (2020). https://doi.org/10.48550/arXiv.1903.07864

[54] He, C., Wang, R., Chen, X.: A tale of two cils: The connections between class incremental learning and class imbalanced learning, and beyond. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 3554–3564 (2021). https://doi.org/10. 1109/CVPRW53098.2021.00395

[55] Kim, G., Xiao, C., Konishi, T., Ke, Z., Liu, B.: A theoretical study on solving continual learning. In: Advances in Neural Information Processing Systems (NeurIPS) (2022). https://doi.org/10.48550/arXiv.2211.02633

[56] Mai, Z., Li, R., Kim, H., Sanner, S.: Supervised contrastive replay: Revisiting the nearest class mean classifier in online class-incremental continual learning. arXiv preprint arXiv:2103.13885 (2021) https://doi.org/10.48550/arXiv.2103.13885

[57] Yang, Y., Cui, Z., Xu, J., Zhong, C., Zheng, W.-S., Wang, R.: Continual learning

with bayesian model based on a fixed pre-trained feature extractor. arXiv preprint arXiv:2204.13349 (2022) https://doi.org/10.48550/arXiv.2204.13349

[58] Momeni, S., Mazumder, S., Liu, B.: Continual learning using a kernel-based method over foundation models. arXiv preprint arXiv:2412.15571 (2024)

[59] Hastie, T., Tibshirani, R., Friedman, J.: The Elements of Statistical Learning: Data Mining, Inference, and Prediction, 2nd edn. Springer Series in Statistics. Springer, New York, NY (2009). https://doi.org/10.1007/978-0-387-84858-7

[60] Krizhevsky, A., Hinton, G., et al.: Learning multiple layers of features from tiny images (2009)

[61] Le, Y., Yang, X.: Tiny imagenet visual recognition challenge. CS 231N 7(7), 3 (2015)

[62] PourKeshavarzi, M., Zhao, G., Sabokrou, M.: Looking back on learned experiences for class/task incremental learning. In: International Conference on Learning Representations (2022)

[63] Li, D., Wang, T., Chen, J., Ren, Q., Kawaguchi, K., Zeng, Z.: Towards continual learning desiderata via hsic-bottleneck orthogonalization and equiangular embedding. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, pp. 13464–13473 (2024)

[64] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778 (2016)

[65] Konishi, T., Kurokawa, M., Ono, C., Ke, Z., Kim, G., Liu, B.: Parameter-level soft-masking for continual learning. arXiv preprint arXiv:2306.14775 (2023) https: //doi.org/10.48550/arXiv.2306.14775

[66] Tao, X., Chang, X., Hong, X., Wei, X., Gong, Y.: Topology-preserving classincremental learning. In: Vedaldi, A., Bischof, H., Brox, T., Frahm, J.-M. (eds.) Computer Vision – ECCV 2020. Lecture Notes in Computer Science, vol. 12364, pp. 254–270. Springer, Cham (2020). https://doi.org/10.1007/978-3-030-58529-7 16

[67] Liu, Y., Su, Y., Liu, A.-A., Schiele, B., Sun, Q.: Mnemonics training: Multiclass incremental learning without forgetting. In: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12242–12251 (2020). https://doi.org/10.1109/CVPR42600.2020.01226

[68] Riemer, M., Cases, I., Ajemian, R., Liu, M., Rish, I., Tu, Y., Tesauro, G.: Learning to learn without forgetting by maximizing transfer and minimizing interference. arXiv preprint arXiv:1810.11910 (2018) https://doi.org/10.48550/arXiv.

[69] Carta, A., Cossu, A., Lomonaco, V., Bacciu, D., Weijer, J.: Projected latent distillation for data-agnostic consolidation in distributed continual learning. arXiv preprint arXiv:2303.15888 (2023)