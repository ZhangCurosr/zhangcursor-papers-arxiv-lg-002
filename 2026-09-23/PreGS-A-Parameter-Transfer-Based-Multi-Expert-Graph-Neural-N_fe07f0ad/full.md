# PreGS: A Parameter-Transfer-Based Multi-Expert Graph Neural Network for Node Classification

Zhicong Cai, Yinglong Zhang, Xiaoying Hong, Xuewen Xia, and Xing Xu

<sup>1</sup>College of Physics and Information Engineering, Minnan Normal University, Zhangzhou 363000, China Corresponding author: Yinglong Zhang (e-mail: zhang yinglong@126.com).

ABSTRACT Graph neural networks have achieved strong performance in node classification by aggregating information from graph neighborhoods. However, a single aggregation mechanism may be insufficient to capture diverse structural patterns across graph datasets. Moreover, independently training multiple structural branches can introduce substantial overhead without necessarily producing stable node representations. To address these issues, this paper proposes PreGS, a parameter-transfer-based multiexpert graph neural network framework. PreGS first pretrains a multi-head graph attention network (GAT) and transfers the linear transformation weights of its first-layer attention heads to multiple GraphSAGE experts. The transferred experts are frozen and used as complementary structural branches. The fused raw node features, GAT head representations, and GraphSAGE expert representations are fed into a multilayer perceptron (MLP), whose output is further fused with the pretrained GAT logits. Based on PreGS, we further develop PreGSv2, which introduces source-level weighting and a structural gating mechanism for adaptive multi-source feature integration. Experiments on eight public graph datasets show that PreGS and PreGSv2 achieve competitive performance against representative graph neural network baselines. Ablation, parameter-transfer, sensitivity, aggregator, visualization, and training-time analyses further validate the effectiveness and stability of the proposed framework. The code and datasets are available at https://github.com/LH-Czc/PreGS.

INDEX TERMS Graph neural networks, multi-expert fusion, node classification, parameter transfer.

## I. INTRODUCTION

ODE classification is a fundamental task in graph machine learning and has been widely applied to citation analysis, social networks, recommendation systems, and knowledge discovery. Graph neural networks (GNNs) have become a dominant paradigm for this task by learning node representations through neighborhood aggregation and message passing [1]. Representative models such as the graph convolutional network (GCN) [2], GraphSAGE [3], and the graph attention network (GAT) [4] capture graph structural information through normalized propagation, fixed neighborhood aggregation, and attention-based neighborhood weighting, respectively.

Recent studies have further improved GNNs from multiple perspectives, including attention mechanisms, simplified propagation, expressive aggregation functions, multi-scale representation learning, and multi-channel fusion [5]–[11]. Meanwhile, graph Transformers and other global modeling methods have attracted increasing attention because of their ability to capture long-range dependencies beyond local message passing [12]–[14]. However, recent re-evaluations suggest that classical GNNs remain highly competitive when properly tuned and structurally enhanced. In several nodelevel and graph-level settings, they can even match or outperform more complex graph Transformer models while retaining better computational efficiency [15], [16]. These observations indicate that the representational potential of classical message-passing architectures has not yet been fully exploited.

Despite these advances, existing methods still face two limitations. First, most GNNs rely on a fixed neighborhood modeling bias within a single propagation branch. For instance, the mean aggregator in GraphSAGE is simple and efficient, but it cannot explicitly distinguish the relative importance of neighboring nodes. GAT introduces attentionbased weights to model neighbor importance, but its propagation process is still constrained by a single attention-driven aggregation form. As a result, a single GNN branch may not fully capture the complementary advantages of different neighborhood aggregation mechanisms.

Second, although ensemble learning and mixture-ofexperts (MoE) models can increase model capacity, general

MoE methods usually depend on expert selection and sparse gating mechanisms [17], [18]. Graph-based MoE methods have also been studied for node classification, decoupled message passing, weak-and-strong expert collaboration, and out-of-distribution graph learning [19]–[22]. Nevertheless, these methods often require independently trained experts or additional routing modules, which increases computational cost and makes it difficult to ensure explicit parameter correspondence and representation-space consistency among experts.

Related to this direction, graph-neural-network-tomultilayer-perceptron (GNN-to-MLP) distillation methods attempt to transfer structural knowledge from GNNs to lightweight MLPs or expert models, thereby reducing training or inference costs [23]–[26]. Recent adaptive hierarchical distillation methods further align GNN and MLP representations across different layers and dimensions to reduce information loss caused by representation mismatch [27]. However, these approaches mainly focus on teacher-student knowledge transfer and make limited use of the internal parameter structures of trained GNNs to construct graph experts with aligned representations.

Motivated by these observations, we propose PreGS, a parameter-transfer-based multi-expert graph neural network framework. The key idea is to revisit GNNs from a unified neighborhood aggregation perspective and interpret the difference between GAT and GraphSAGE as different strategies for constructing neighborhood aggregation weights. Since the two models share compatible linear feature transformation structures, the parameters of different attention heads in the first layer of a pretrained multi-head GAT can be transferred to multiple GraphSAGE experts. In this way, PreGS constructs a structurally homologous multi-expert system with explicit parameter correspondence and aligned representation spaces.

During training, the pretrained GAT and the transferred GraphSAGE experts are frozen, while only the fusion module and the MLP classifier are optimized. This design decouples graph structural feature extraction from task-specific adaptation, thereby reducing optimization complexity and improving training stability. The base model, PreGS, adopts a grouped weighted fusion strategy to separately integrate GAT multi-head features and GraphSAGE expert features. The enhanced model, PreGSv2, further introduces sourcelevel weighting and a structural gating mechanism to achieve finer-grained adaptive modeling of multi-source features.

The main contributions of this paper are summarized as follows:

• We present a unified neighborhood aggregation perspective and analyze the transferable relationship between GAT and GraphSAGE in terms of their linear feature transformation structures.

• We propose a GraphSAGE multi-expert construction mechanism based on pretrained GAT parameter transfer, forming an expert system with structural homology and aligned representation spaces.

• We introduce a decoupled training paradigm with frozen experts, where only the fusion module and the MLP classifier are trained, thereby reducing optimization complexity and improving training stability.

• We design two fusion models, PreGS and PreGSv2, and conduct a comprehensive evaluation through node classification, ablation, parameter-transfer, sensitivity, aggregator, visualization, and training-time experiments.

## II. RELATED WORK

Existing GNNs usually differ in how neighborhood information is weighted, aggregated, and fused. GCN [2] adopts normalized graph convolution, GraphSAGE [3] uses predefined neighborhood aggregators, and GAT [4] assigns adaptive attention weights to neighboring nodes. Later variants such as GATv2 [5], TANGNN [6], SGC [7], GIN [8], JK-Net [9], and PNA [10] further improve attention expressiveness, scalable attention-based neighborhood selection, propagation efficiency, aggregation capacity, cross-layer representation fusion, and multi-aggregator modeling. However, these methods are usually designed as independent architectures, and the transferable relationship between the internal parameters of different GNN branches has received less attention. This motivates our investigation of whether the first-layer attention-head parameters of a pretrained GAT can be reused to construct structurally aligned GraphSAGE experts.

Beyond conventional local message passing, recent studies have explored global interaction and multi-branch modeling to enhance graph representation learning. Graph Transformers introduce global node interactions to capture longrange dependencies [12]–[14], while multi-channel, multiscale, multi-aggregator, and graph mixture-of-experts methods improve representation learning through complementary branches, aggregation functions, or expert routing mechanisms [10], [11], [19]–[22]. Although these methods demonstrate the value of combining multiple structural views, they mainly rely on additional architectures or routing modules. In contrast, PreGS constructs aligned GraphSAGE experts by explicitly transferring parameters from pretrained GAT heads, which provides a direct parameter-level connection between different GNN branches.

Another related line of work is decoupled graph learning and GNN-to-MLP distillation, which separate structural representation learning from downstream prediction through structural constraints, knowledge transfer, or hybrid expert modeling [21], [23]–[25], [27]. These methods mainly transfer knowledge through predictions, intermediate representations, or structural regularization. PreGS follows the spirit of decoupled learning, but focuses on a different transfer target: it reuses the internal feature-transformation parameters of pretrained GAT heads to construct structurally aligned GraphSAGE experts, and then performs lightweight task adaptation through trainable fusion modules.

## III. PRELIMINARIES

To describe the proposed framework, this section introduces the graph notation and briefly reviews the basic components used in this work, including GAT, GraphSAGE, MLP, and the node classification loss.

## A. Notation

Let $G = ( V , E )$ denote an undirected graph, where $V =$ $\{ v _ { 1 } , v _ { 2 } , \ldots , v _ { N } \}$ is the node set, $N = | V |$ is the number of nodes, and $E \subseteq V \times V$ is the edge set. The node feature matrix is denoted by $\mathbf { X } \in \mathbb { R } ^ { N \times d }$ , where d is the feature dimension. For node $v _ { i } ,$ , its original feature vector is denoted by $\mathbf { x } _ { i } \in \mathbb { R } ^ { 1 \times d }$ . The label of node $v _ { i }$ is denoted by $y _ { i }$ , and the label set of all nodes is denoted by $\mathbf { Y } = \{ y _ { i } \} _ { i = 1 } ^ { N }$ , where $y _ { i } \in \{ 1 , 2 , \ldots , C \}$ and C is the number of classes.

For node $v _ { i }$ , its one-hop neighborhood is denoted by $\mathcal { N } ( i ) = \{ v _ { j } \mid ( v _ { i } , v _ { j } ) \in E \}$ . When the aggregation process includes the target node itself, the self-loop neighborhood is written as $\overset { \sim } { \mathcal { N } } ( i ) = \mathcal { N } ( i ) \cup \{ v _ { i } \}$

In this paper, σ(·) denotes a nonlinear activation function, ∥ denotes feature concatenation, softmax(·) denotes a normalization function, and AGG(·) denotes a replaceable neighborhood aggregation operator. Common aggregation operators include mean, sum, and max, corresponding to neighborhood averaging, neighborhood summation, and neighborhood maximum aggregation, respectively.

For consistency, the hidden representation of node $v _ { i }$ is denoted by $\mathbf { h } _ { i } ,$ and a linear transformation parameter is denoted by W. If a model contains K attention heads or $K$ expert branches, the parameter and output representation of the kth branch are denoted by $\mathbf { W } ^ { ( k ) }$ and ${ \bf h } _ { i } ^ { ( k ) }$ , respectively, where $k = 1 , 2 , \dots , K$

Unless otherwise specified, all node feature vectors in this paper are represented as row vectors. Accordingly, linear transformations, attention computation, and feature concatenation are written in row-vector form.

## B. Graph Attention Network

Graph Attention Network (GAT) assigns adaptive weights to neighboring nodes through an attention mechanism, thereby modeling their different contributions to the target node representation [4]. For node $v _ { i }$ and its neighbor $v _ { j } \in \tilde { \mathcal { N } } ( i )$ let h<sub>i</sub> and $\mathbf { h } _ { j }$ denote their input representations in the current GAT layer. In the first GAT layer, $\mathbf { h } _ { i } = \mathbf { x } _ { i }$ and $\mathbf { h } _ { j } = \mathbf { x } _ { j }$

With attention coefficients $\alpha _ { i j }$ normalized over $\widetilde { \mathcal { N } } ( i )$ , the GAT update can be written as

$$
\mathbf { h } _ { i } ^ { \prime } = \sigma \left( \sum _ { v _ { j } \in \widetilde { \mathcal { N } } ( i ) } \alpha _ { i j } \mathbf { h } _ { j } \mathbf { W } \right) .\tag{1}
$$

For a multi-head GAT with K attention heads, the output of the kth head is

$$
\mathbf { h } _ { i } ^ { ( k ) } = \sigma \left( \sum _ { v _ { j } \in \widetilde { N } ( i ) } \alpha _ { i j } ^ { ( k ) } \mathbf { h } _ { j } \mathbf { W } ^ { ( k ) } \right) ,\tag{2}
$$

where $k = 1 , 2 , \dots , K$

In this paper, GAT serves as the pretrained source model. The independent first-layer head parameters $\{ \mathbf { W } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ provide the parameter sources for constructing the subsequent GraphSAGE expert branches.

## C. GraphSAGE

GraphSAGE is a neighborhood-aggregation-based graph neural network for learning node representations in an inductive manner [3]. Unlike GAT, which learns adaptive attention weights for neighboring nodes, GraphSAGE usually adopts a predefined aggregation function to summarize neighborhood features.

When the mean aggregator is used and the target node itself is included in the aggregation set, the GraphSAGE update can be written as

$$
\mathbf { h } _ { i } ^ { \prime } = \sigma \left( \frac { 1 } { | \widetilde { \mathcal { N } } ( i ) | } \sum _ { v _ { j } \in \widetilde { \mathcal { N } } ( i ) } \mathbf { h } _ { j } \mathbf { W } \right) .\tag{3}
$$

This formulation shows that the mean aggregation in Graph-SAGE can be interpreted as a neighborhood weighted-sum process with fixed uniform weights. It therefore shares a common aggregation form with the attention-based weighted aggregation in GAT, providing a structural basis for transferring parameters from GAT attention heads to GraphSAGE expert branches.

## D. Multilayer Perceptron

The Multilayer Perceptron (MLP) is used as the classifier after feature fusion. Unlike GNN layers, it does not explicitly use graph topology. Given the fused representation $\mathbf { f } _ { i }$ of node $v _ { i } .$ , the two-layer MLP classifier is defined as

$$
\begin{array} { r } { \mathbf { z } _ { i } ^ { \mathrm { M L P } } = \mathrm { M L P } ( \mathbf { f } _ { i } ) = \sigma ( \mathbf { f } _ { i } \mathbf { W } _ { 1 } + \mathbf { b } _ { 1 } ) \mathbf { W } _ { 2 } + \mathbf { b } _ { 2 } , } \end{array}\tag{4}
$$

where $\mathbf { z } _ { i } ^ { \mathrm { { M L P } } }$ denotes the MLP logits of node $v _ { i }$ . In the proposed framework, the MLP performs task-specific prediction based on the fused representations.

## E. Node Classification Loss

For node classification, this paper uses the cross-entropy loss as the training objective. Let $y _ { i }$ denote the ground-truth label of node $v _ { i } ,$ and let $\mathbf { z } _ { i }$ denote the final logits produced by the model. For a training node set $\mathcal { V } _ { \mathrm { t r } }$ , the cross-entropy loss is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C E } } = - \displaystyle \frac { 1 } { | \mathcal { V } _ { \mathrm { t r } } | } \sum _ { v _ { i } \in \mathcal { V } _ { \mathrm { t r } } } \sum _ { c = 1 } ^ { C } \mathbf { 1 } ( y _ { i } = c ) } \\ { \times \log \left( \frac { \exp \left( z _ { i c } \right) } { \sum _ { r = 1 } ^ { C } \exp \left( z _ { i r } \right) } \right) , } \end{array}\tag{5}
$$

where $C$ is the number of classes, $\mathbf { 1 } ( \cdot )$ is the indicator function, and $z _ { i c }$ denotes the final logit of node $v _ { i }$ for class $c .$ The predicted label is obtained by $\hat { y } _ { i } = \arg \operatorname* { m a x } _ { c } z _ { i c } .$

In PreGS and PreGSv2, this loss is used to optimize the fusion module and the MLP classifier, while the pretrained GAT and the transferred GraphSAGE experts are kept fixed.

## IV. PROPOSED METHOD

This section presents the proposed PreGS family framework. We first analyze the structural relationship between GAT and GraphSAGE from a unified neighborhood aggregation perspective. Then, we describe the three-stage procedure of PreGS, including GAT pretraining, GraphSAGE expert construction, and decoupled fusion training. Finally, we introduce the base model PreGS, the enhanced model PreGSv2, and the corresponding algorithmic procedure.

## A. Theoretical Basis

To explain the rationale behind the proposed framework, we start from a unified view of neighborhood aggregation. Although existing graph neural networks adopt different mechanisms for neighborhood modeling, their common objective is to update the representation of a target node by aggregating information from its neighbors. For example, GAT learns adaptive neighborhood weights through attention, while GraphSAGE summarizes neighborhood features using a predefined aggregation function. Despite their architectural differences, both models can be interpreted as neighborhood feature aggregation processes.

## 1) Unified Neighborhood Aggregation Perspective

To characterize the relationship between GAT and Graph-SAGE, we formulate their aggregation processes from a unified neighborhood aggregation perspective.

Proposition 1. For graph neural networks that adopt linear feature transformation and neighborhood information aggregation, the node representation update process of weightedsum-based aggregators can be written in the following unified form:

$$
\mathbf { h } _ { i } ^ { \prime } = \sigma \left( \sum _ { v _ { j } \in \widetilde { \mathcal { N } } ( i ) } \beta _ { i j } \mathbf { h } _ { j } \mathbf { W } \right) ,\tag{6}
$$

where W is the feature transformation matrix, and $\beta _ { i j }$ denotes the aggregation weight assigned by node $v _ { i }$ to node $v _ { j } .$ . Different aggregation mechanisms mainly differ in the construction of $\beta _ { i j }$

## Proof:

For GAT, by setting $\beta _ { i j } = \alpha _ { i j }$ , the GAT update rule in (1) can be written in the unified form of (6). In this case, $\alpha _ { i j }$ is normalized over the neighborhood of node $v _ { i } .$

For GraphSAGE with the mean and sum aggregators, by setting $\beta _ { i j } = 1 / | \widetilde { N } ( i ) |$ and $\beta _ { i j } ~ = ~ 1$ , respectively, the corresponding aggregation rules can also be written in the unified form of (6).

Therefore, the above aggregation rules can all be expressed under the unified weighted-sum aggregation form.

This unified neighborhood aggregation view indicates that GAT, GraphSAGE-mean, and GraphSAGE-sum share compatible linear feature transformation structures and differ mainly in their neighborhood summarization strategies. Therefore, the parameters learned by a pretrained GAT have the potential to be transferred to GraphSAGE expert structures, and the multi-head attention architecture of GAT provides a natural basis for constructing multiple experts. In the proposed framework, different GraphSAGE experts can further adopt different aggregation operators to increase representation diversity, as described in the experimental configuration.

## B. PreGS Family Framework

The central problem addressed in this paper is how to use the naturally parallel multi-head structure of GAT to construct multiple GraphSAGE expert branches with explicit parameter correspondence and aligned representation spaces. To this end, we propose the PreGS family, a parametertransfer-based multi-expert graph neural network framework, including the base model PreGS and the enhanced model PreGSv2. Instead of training independent experts or using complex routing mechanisms commonly adopted in MoEbased graph models [19]–[21], PreGS transfers the parameters of the first-layer attention heads of a pretrained GAT to multiple GraphSAGE experts, making the experts structurally homologous and explicitly parameter-related.

In the subsequent training stage, both the pretrained GAT and the transferred GraphSAGE experts are frozen. Only the fusion module and the MLP classifier are optimized. This decoupled design preserves the representation ability of the pretrained model while reducing the optimization complexity of later training.

## C. Overall Procedure

The proposed method consists of three stages: GAT pretraining, multi-expert construction with parameter transfer, and decoupled fusion training.

1) GAT pretraining. In the first stage, a multi-head GAT is pretrained on the target graph to obtain stable attentionhead parameters and graph structural representations. Let ${ \bf h } _ { i } ^ { ( k ) }$ denote the output representation of node $v _ { i }$ from the kth attention head in the first GAT layer, where $k = 1 , 2 , \ldots , K$ Meanwhile, the final output logits of the pretrained GAT are retained and denoted by $\bar { \mathbf { z } _ { i } ^ { \mathrm { G A T } } }$ . These logits contain high-level structural information obtained after multi-layer attention propagation and are used as an important supplement in the final prediction fusion. After pretraining, the GAT parameters are fixed and no longer updated during the subsequent fusion training stage.

2) Multi-expert construction and parameter transfer. In the second stage, the linear transformation parameters of the kth attention head in the first GAT layer are transferred to the kth GraphSAGE expert:

$$
\mathbf { W } _ { \mathrm { G S } } ^ { ( k ) }  \mathbf { W } _ { \mathrm { G A T } } ^ { ( k ) } ,\tag{7}
$$

where $k = 1 , 2 , \ldots , K$ . This operation constructs K singlelayer GraphSAGE expert branches. Each expert corresponds to one GAT attention head, so the experts are not randomly initialized or independently trained from scratch. Instead, they are derived from the multi-head structure of the pretrained GAT. The output representation of node $v _ { i }$ produced by the kth single-layer GraphSAGE expert is denoted by ${ \bf e } _ { i } ^ { ( k ) }$ . After expert construction, all GraphSAGE expert parameters are frozen.

3) Decoupled fusion training. In the third stage, the pretrained GAT and the transferred GraphSAGE experts act as fixed feature extractors and do not participate in parameter updates. The model only trains the fusion module and the MLP classifier. Based on this design, this paper develops two fusion strategies. PreGS performs grouped weighted fusion over GAT multi-head features and GraphSAGE expert features, while PreGSv2 further introduces source-level weighting and structural gating for more adaptive multisource feature modeling.

Because the parameters of the pretrained GAT and the GraphSAGE experts are frozen, the training process only needs to optimize the branch-fusion weights, logit-fusion weights, and MLP classifier parameters, together with the source-level weighting and gating parameters in PreGSv2. This design transforms the highly coupled end-to-end optimization problem in conventional GNNs into a lightweight task-adaptation problem, thereby reducing training complexity and improving training stability.

Figure 1 further presents the detailed architecture of the PreGS family. It shows how the pretrained GAT branch, the transferred GraphSAGE expert branch, and the raw feature branch are integrated before the final fusion-and-prediction stage. The detailed formulations of PreGS and PreGSv2 are given in the following subsections.

## D. PreGS: Grouped Weighted Fusion

For each node $v _ { i } ,$ , PreGS uses three types of feature sources: the raw node feature $\mathbf { x } _ { i } .$ , the first-layer GAT head representations $\{ \mathbf { h } _ { i } ^ { ( k ) } \} _ { k = 1 } ^ { K } ,$ and the transferred GraphSAGE expert representations $\{ \bar { \mathbf { e } } _ { i } ^ { ( k ) } \} _ { k = 1 } ^ { K }$

To model the importance of different branches, learnable weights are introduced for the GAT heads and the GraphSAGE experts, respectively. For the GAT branch, the normalized weight of the kth head is defined as

$$
\omega _ { k } ^ { \mathrm { G A T } } = \frac { \exp ( \widetilde { \omega } _ { k } ^ { \mathrm { G A T } } ) } { \sum _ { t = 1 } ^ { K } \exp ( \widetilde { \omega } _ { t } ^ { \mathrm { G A T } } ) } ,\tag{8}
$$

where $\{ \widetilde { \omega } _ { k } ^ { \mathrm { G A T } } \} _ { k = 1 } ^ { K }$ are learnable parameters. For the Graph-SAGE expert branch, the normalized expert weight is defined as

$$
\omega _ { k } ^ { \mathrm { G S } } = \frac { \exp ( \widetilde { \omega } _ { k } ^ { \mathrm { G S } } ) } { \sum _ { t = 1 } ^ { K } \exp ( \widetilde { \omega } _ { t } ^ { \mathrm { G S } } ) } ,\tag{9}
$$

where $\{ \widetilde \omega _ { k } ^ { \mathrm { G S } } \} _ { k = 1 } ^ { K }$ are also learnable parameters.

The two groups of features are then fused separately by weighted summation:

$$
\mathbf { f } _ { i } ^ { \mathrm { G A T } } = \sum _ { k = 1 } ^ { K } \omega _ { k } ^ { \mathrm { G A T } } \mathbf { h } _ { i } ^ { ( k ) } ,\tag{10}
$$

$$
\mathbf { f } _ { i } ^ { \mathrm { G S } } = \sum _ { k = 1 } ^ { K } \omega _ { k } ^ { \mathrm { G S } } \mathbf { e } _ { i } ^ { ( k ) } .\tag{11}
$$

After that, the raw node feature and the two fused representations are concatenated to obtain the final fusion representation:

$$
\mathbf { f } _ { i } = \mathbf { x } _ { i } \lVert \mathbf { f } _ { i } ^ { \mathrm { G A T } } \rVert \mathbf { f } _ { i } ^ { \mathrm { G S } } .\tag{12}
$$

The fused representation is then fed into an MLP classifier:

$$
\begin{array} { r } { { \bf z } _ { i } ^ { \mathrm { M L P } } = \mathrm { M L P } ( { \bf f } _ { i } ) . } \end{array}\tag{13}
$$

Since the final output of the pretrained GAT contains high-level structural information, PreGS further combines the MLP logits with the final GAT logits through learnable logit-level fusion:

$$
\mathbf { z } _ { i } = \frac { \exp ( \lambda _ { 1 } ) } { \exp ( \lambda _ { 1 } ) + \exp ( \lambda _ { 2 } ) } \mathbf { z } _ { i } ^ { \mathrm { M L P } } + \frac { \exp ( \lambda _ { 2 } ) } { \exp ( \lambda _ { 1 } ) + \exp ( \lambda _ { 2 } ) } \mathbf { z } _ { i } ^ { \mathrm { G A T } } ,\tag{14}
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are learnable fusion parameters, and $\mathbf { z } _ { i }$ denotes the final logits of node $v _ { i }$

During training, the pretrained GAT and the transferred GraphSAGE experts are frozen. Only the fusion weights and the MLP classifier parameters are optimized. The training objective follows the cross-entropy loss defined in (5).

## E. PreGSv2: Source-Weighted Gated Fusion

Although PreGS can effectively combine the structural information from the pretrained GAT and the GraphSAGE experts, its fusion weights are shared across all nodes after training. Therefore, it cannot dynamically adjust the importance of different feature sources according to the structural characteristics of individual nodes. To improve the adaptive modeling ability of the framework, we further propose PreGSv2, which introduces source-level weighting and a structural gating mechanism on the basis of PreGS.

## 1) Source-Level Weighted Fusion

PreGSv2 introduces learnable source-level weights for the raw feature, the fused GAT representation, and the fused GraphSAGE representation. The normalized source weight is defined as

$$
\rho _ { s } = \frac { \exp ( \eta _ { s } ) } { \sum _ { t = 1 } ^ { 3 } \exp ( \eta _ { t } ) } ,\tag{15}
$$

where $\{ \eta _ { s } \} _ { s = 1 } ^ { 3 }$ are learnable parameters. The sourceweighted representation is then constructed by weighted concatenation:

$$
\begin{array} { r } { \mathbf { f } _ { i } ^ { \mathrm { s r c } } = \rho _ { 1 } \mathbf { x } _ { i } \| \rho _ { 2 } \mathbf { f } _ { i } ^ { \mathrm { G A T } } \| \rho _ { 3 } \mathbf { f } _ { i } ^ { \mathrm { G S } } , } \end{array}\tag{16}
$$

where $\mathbf { f } _ { i } ^ { \mathrm { G A T } }$ and $\mathbf { f } _ { i } ^ { \mathrm { G S } }$ are obtained from (10) and (11), respectively.

![](images/a000828454e4f48e6d80ab34e84596e43296a762eb3b4c1b60c2037b1d98c5e2.jpg)  
FIGURE 1. Detailed architecture of the proposed PreGS and PreGSv2 framework. The figure illustrates GAT pretraining, GraphSAGE expert construction through parameter transfer, and the fusion-and-prediction stage used by the two model variants.

## 2) Structural Gating Mechanism

To further enhance node-level adaptive modeling, PreGSv2 introduces a structural gating mechanism. The gate vector is generated from the fused GAT representation:

$$
\mathbf { g } _ { i } = \sigma \left( \mathbf { f } _ { i } ^ { \mathrm { G A T } } \mathbf { W } _ { g } + \mathbf { b } _ { g } \right) ,\tag{17}
$$

where $\mathbf { W } _ { g }$ and ${ \bf b } _ { g }$ are learnable parameters. The gate vector is then used to modulate the source-weighted representation dimension by dimension:

$$
\mathbf { f } _ { i } ^ { \mathrm { g a t e } } = \mathbf { g } _ { i } \odot \mathbf { f } _ { i } ^ { \mathrm { s r c } } ,\tag{18}
$$

where ⊙ denotes the Hadamard product.

The gated representation is fed into the MLP classifier:

$$
{ \bf z } _ { i } ^ { \mathrm { M L P } } = \mathrm { M L P } \left( { \bf f } _ { i } ^ { \mathrm { g a t e } } \right) .\tag{19}
$$

As in PreGS, the MLP logits are further combined with the final GAT logits:

$$
\mathbf { z } _ { i } = \frac { \exp ( \lambda _ { 1 } ) } { \exp ( \lambda _ { 1 } ) + \exp ( \lambda _ { 2 } ) } \mathbf { z } _ { i } ^ { \mathrm { M L P } } + \frac { \exp ( \lambda _ { 2 } ) } { \exp ( \lambda _ { 1 } ) + \exp ( \lambda _ { 2 } ) } \mathbf { z } _ { i } ^ { \mathrm { G A T } } .\tag{20}
$$

Compared with PreGS, PreGSv2 keeps the parametertransfer mechanism and the multi-expert structure unchanged, but introduces source-level weighting and structural gating to dynamically adjust different feature sources. This improves the ability of the model to represent complex graph structures. PreGSv2 is trained with the same cross-entropy loss defined in (5).

## F. Further Theoretical Analysis

The previous subsections construct a parameter-transferbased multi-expert graph neural network framework from the unified neighborhood aggregation view. This subsection further analyzes the framework from three aspects: the feasibility of parameter transfer, representation-space consistency after transfer, and the multi-expert interpretation of the transferred GraphSAGE branches.

Proposition 2. Assume that a pretrained GAT attention head and a GraphSAGE expert have the same input and output feature dimensions. Then, the feature transformation parameter W learned by the GAT attention head can be transferred to the GraphSAGE expert as its initialization parameter, regardless ofthe specific neighborhood aggregation operator used by the expert.

## Proof:

For a fixed input and output feature space, the matrix W represents the feature mapping between node representations. The neighborhood aggregation operator determines how transformed neighborhood features are summarized, but it does not change the dimensional compatibility of W. Therefore, when the input and output dimensions are consistent, the feature transformation parameter learned by a GAT attention head can be transferred to a GraphSAGE expert. ■

Proposition 2 shows that the feasibility of parameter transfer comes from the shared feature transformation structure of the two models, rather than from identical neighborhood aggregation mechanisms. Therefore, the pretrained GAT can provide effective initialization for GraphSAGE experts and avoid constructing each expert from random initialization.

Corollary 1. Under the parameter-transfer condition, the GAT attention heads and the GraphSAGE experts share the same linear transformation parameters. Therefore, their output representations lie in aligned feature transformation spaces and can be directly fused at the feature level.

## Proof:

Let the transformation matrix of the kth GAT attention head be $\mathbf { W } _ { \mathrm { G A T } } ^ { ( k ) }$ . According to the parameter-transfer operation, the corresponding GraphSAGE expert is initialized with the same transformation matrix, namely $\mathbf { W } _ { \mathrm { G S } } ^ { ( k ) } = \mathbf { W } _ { \mathrm { G A T } } ^ { ( k ) } .$ Thus, both branches map the input features into the same transformed feature dimension. Although their neighborhood aggregation weights are different, their feature mapping spaces remain aligned. Therefore, the GAT head representation ${ \bf h } _ { i } ^ { ( k ) }$ and the GraphSAGE expert representation ${ \bf e } _ { i } ^ { ( k ) }$ have representation-space consistency.

Representation-space consistency ensures that $\mathbf { f } _ { i } ^ { \mathrm { G A T } }$ and $\mathbf { f } _ { i } ^ { \mathrm { G S } }$ can be weighted and concatenated directly without introducing an additional projection layer. This provides the theoretical basis for the fusion strategy in PreGS.

Proposition 3. Suppose the first layer of a pretrained GAT contains K attention heads, whose feature transformation matrices are denoted by $\{ \mathbf { W } _ { \mathrm { G A T } } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ . If the parameter of each attention head is transferred to one GraphSAGE expert, the resulting GraphSAGE expert set can be regarded as a structurally homologous reconstruction and extension of the GAT multi-head representation in the GraphSAGE architecture.

## Proof:

Each GraphSAGE expert inherits the feature transformation parameter learned by a different GAT attention head. Therefore, different experts preserve the feature projection ability of different attention heads. Meanwhile, because the experts may use different aggregation operators or neighborhood weighting patterns, they produce diverse expert representations. Thus, the transferred GraphSAGE experts form a parameter-related and structurally corresponding expert set derived from the pretrained GAT multi-head structure.

Algorithm 1 Training Procedure of PreGS and PreGSv2   
Require: Graph $\overline { { G \ = \ ( V , E ) } }$ , features X, labels Y, heads/experts $K ,$   
variant m   
Ensure: Node predictions $\{ \hat { y } _ { i } \} _ { v _ { i } \in V }$   
1: Train a K-head GAT and freeze it after pretraining.   
2: Store first-layer GAT-head representations $\{ \mathbf { h } _ { i } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ and final GAT   
logits $\mathbf { z } _ { i } ^ { \mathrm { { G A T } } } .$   
3: for $k = 1$ to K do   
4: Build a single-layer ${ \mathrm { G S } } _ { k }$ by parameter transfer in (7), assign its   
aggregator, freeze it, and extract ${ \bf e } _ { i } ^ { ( k ) }$   
5: end for   
6: repeat   
7: Compute $\mathbf { f } _ { i } ^ { \mathrm { G A T } }$ and ${ \bf f } _ { i } ^ { \mathrm { G S } }$ by (8)–(11).   
8: if m = PreGS then   
9: Construct f<sub>i</sub> by concatenation in (12).   
10: Obtain $\mathbf { z } _ { i } ^ { \mathrm { M L P } }$ and z by (13) and (14).   
11: else   
12: Construct $\mathbf { f } _ { i } ^ { \mathrm { g a t e } }$ by source weighting and gating in (15)–(18).   
13: Obtain $\mathbf { z } _ { i } ^ { \mathrm { M L P } }$ and $\mathbf { z } _ { i }$ by (19) and (20).   
14: end if   
15: Update trainable parameters by minimizing (5).   
16: until early stopping criterion is satisfied   
17: Predict labels from the final logits.   
18: return $\{ \hat { y } _ { i } \} _ { v _ { i } \in V }$

Proposition 3 indicates that multiple GraphSAGE experts can be interpreted as a reconstruction and extension of the pretrained GAT multi-head knowledge under the Graph-SAGE architecture. Therefore, the proposed framework does not simply combine independent models; instead, it builds a multi-expert system with explicit parameter origins and structural correspondence.

In summary, the proposed framework is supported by parameter-transfer feasibility, representation-space consistency, and multi-expert construction. First, the unified neighborhood aggregation view explains why GAT parameters can be transferred to GraphSAGE structures. Second, shared initialization parameters keep different experts in aligned representation spaces. Finally, the structural correspondence between GAT multi-head attention and GraphSAGE experts provides an interpretation for multi-expert construction. These analyses jointly provide the theoretical basis for the proposed multi-expert graph neural network framework.

## G. Algorithmic Procedure

Algorithm 1 summarizes the training procedure of PreGS and PreGSv2.

## V. EXPERIMENTS

## A. Experimental Setup

## 1) Datasets

Experiments are conducted on eight public graph datasets: ACM, AMAC, AMAP, DBLP, EAT, FILM, PubMed, and Texas. These datasets cover different graph domains, including academic networks, citation networks, Amazon co-purchasing networks, air-traffic networks, actor cooccurrence networks, and webpage networks [28]–[31]. They vary in graph scale, feature dimensionality, and number of classes, providing a diverse benchmark for node classification. The dataset statistics are summarized in Table 1.

TABLE 1. Statistics of the Datasets
<table><tr><td>Dataset</td><td>Nodes (N)</td><td>Feature Dim. (d)</td><td>Classes (C)</td></tr><tr><td>ACM</td><td>3025</td><td>1870</td><td>3</td></tr><tr><td>AMAC</td><td>2405</td><td>128</td><td>4</td></tr><tr><td>AMAP</td><td>1043</td><td>128</td><td>6</td></tr><tr><td>DBLP</td><td>4057</td><td>334</td><td>4</td></tr><tr><td>EAT</td><td>1575</td><td>64</td><td>5</td></tr><tr><td>FILM</td><td>778</td><td>932</td><td>5</td></tr><tr><td>PubMed</td><td>19717</td><td>500</td><td>3</td></tr><tr><td>Texas</td><td>183</td><td>1703</td><td>5</td></tr></table>

## 2) Experimental Configuration

To ensure fair comparison, all models follow the same data splitting and training protocol, and the GNN baselines are implemented with a two-layer architecture where applicable. For PreGS and PreGSv2, a two-layer GAT with eight attention heads is first pretrained as the source model, and the output dimension of each head is set to 8. The first-layer attention-head parameters are then transferred to eight singlelayer GraphSAGE experts. For the default eight-head setting, PreGS and PreGSv2 use the expert aggregation configuration (mean, mean, max, max, max, sum, sum, sum). This mixed configuration is adopted to introduce complementary neighborhood summarization behaviors among the transferred experts. Specifically, mean and sum aggregators follow the weighted-sum aggregation form discussed in Section A, while the max aggregator is included as an additional summarization operator to enrich expert diversity.

All models are trained with the Adam optimizer. The learning rate is set to 0.005, the weight decay coefficient is set to $5 \times 1 0 ^ { - 4 }$ , and the dropout rate is set to 0.6. The maximum number of training epochs is 2000, and early stopping with a patience of 100 is used. Unless otherwise specified, each quantitative experiment is repeated 30 times, and the average result is reported.

For data splitting, the training ratios are set to 20%, 40%, and 60%, respectively. The validation ratio is fixed at 10%, and the remaining nodes are used for testing. All compared models are optimized using the same cross-entropy loss defined in (5).

## 3) Baselines

We compare the proposed PreGS and PreGSv2 with representative node classification baselines, including GCN [2],

SGC [7], GIN [8], GAT [4], GATv2 [5], GraphSAGE [3], GraphSAGE++ [32], GNNMoE [20], JK-Net [9], and MLP. These baselines cover normalized propagation, simplified propagation, expressive aggregation, attention-based message passing, fixed neighborhood aggregation, multiscale aggregation, mixture-of-experts-based adaptive message passing, cross-layer fusion, and feature-only classification. For consistency, GraphSAGE and GraphSAGE++ are trained without neighbor sampling, using the same input graph for message passing as the other GNN baselines. All compared models follow the same data splitting, validation, and early-stopping protocol.

## B. Accuracy Results

Table 2 reports the node classification accuracy under different training ratios. Overall, the proposed models show strong and consistent performance across the eight datasets. Among the 24 dataset–training-ratio settings, either PreGS or PreGSv2 achieves the best result in 20 settings, and at least one of the two models ranks among the top two in 21 settings. Specifically, PreGS obtains 11 best and 7 secondbest results, while PreGSv2 obtains 9 best and 6 secondbest results. These results indicate that parameter transfer and multi-source feature fusion can provide substantial improvements over a single attention-based branch.

More specifically, PreGS achieves the best performance under all three training ratios on ACM, DBLP, and Texas, and also ranks first on AMAP under the 20% and 40% training ratios. PreGSv2 achieves the best results under all three training ratios on AMAC and FILM, ranks first on AMAP under the 60% training ratio, and obtains the best performance on PubMed under the 20% and 40% training ratios. GNNMoE is a strong competing baseline, particularly on ACM, AMAP, and PubMed, and achieves the best result on PubMed under the 60% training ratio. The proposed models do not dominate EAT, where JK-Net and GIN consistently perform better. Overall, PreGS provides stronger performance on ACM, DBLP, and Texas, whereas PreGSv2 shows clearer advantages on AMAC, FILM, and PubMed, demonstrating that the two fusion strategies are complementary across different graph structures.

## C. Effectiveness ofParameter Transfer

To examine whether pretrained GAT parameters can provide useful initialization for GraphSAGE experts, we transfer the first-layer attention-head parameters of a two-layer eighthead GAT to eight single-layer GraphSAGE experts. An MLP classifier is then attached after each transferred expert. The experiment is conducted under the 20% training ratio, and the results are reported in Table 3.

As shown in Table 3, a single transferred GraphSAGE expert generally performs worse than the complete GAT and GraphSAGE models, but most experts still retain nontrivial classification ability. This indicates that the linear transformation parameters learned by GAT attention heads can be reused by GraphSAGE experts as meaningful feature mappings. The results also show that different aggregators adapt to the transferred parameters differently. For example, on AMAC, mean-based experts are more stable than maxbased experts. This supports the use of multiple aggregation operators to provide diverse expert representations.

## D. Ablation Study

TABLE 2. Node Classification Accuracy Under Different Training Ratios (%)
<table><tr><td>Dataset</td><td>Train</td><td>GCN</td><td>SGC</td><td>GIN</td><td>GAT</td><td>GATv2</td><td>GraphSAGE</td><td>GraphSAGE++ GNNMoE</td><td></td><td>JK-Net</td><td>MLP</td><td>PreGS</td><td>PreGSv2</td></tr><tr><td rowspan="3">ACM</td><td>20%</td><td>91.23</td><td>86.29</td><td>89.16</td><td>90.92</td><td>91.18</td><td>90.80</td><td>90.65</td><td> $9 2 . 4 2 _ { ( 2 ) }$ </td><td>91.37</td><td>88.26</td><td> $9 2 . 6 3 _ { ( 1 ) }$ </td><td> $9 2 . 3 5 _ { ( 3 ) }$ </td></tr><tr><td>40%</td><td>91.94</td><td>87.22</td><td>90.28</td><td>91.71</td><td>91.75</td><td>91.61</td><td>91.18</td><td> $9 3 . 4 1 _ { ( 2 ) }$ </td><td>92.12</td><td>89.85</td><td> $9 3 . 5 5 _ { ( 1 ) }$ </td><td> $9 3 . 3 6 _ { ( 3 ) }$ </td></tr><tr><td>60%</td><td>92.42</td><td>88.52</td><td>91.49</td><td>92.33</td><td>92.38</td><td>92.15</td><td>91.86</td><td> $9 3 . 8 3 _ { ( 3 ) }$ </td><td>92.50</td><td>90.37</td><td> $9 3 . 9 6 _ { ( 1 ) }$ </td><td> $9 3 . 8 5 _ { ( 2 ) }$ </td></tr><tr><td rowspan="3">AMAC</td><td>20%</td><td>78.59</td><td>76.11</td><td>74.73</td><td>76.40</td><td>82.10</td><td>76.98</td><td>79.48</td><td> $\mathbf { 8 3 . 7 9 } _ { ( 3 ) }$ </td><td>81.75</td><td>81.92</td><td> $\mathbf { 8 6 . 0 0 } _ { ( 2 ) }$ </td><td> ${ \bf 8 6 . 2 9 } _ { ( 1 ) }$ </td></tr><tr><td>40%</td><td>79.11</td><td>77.03</td><td>76.09</td><td>76.91</td><td>83.23</td><td>77.29</td><td>81.58</td><td> $\mathbf { 8 5 . 9 0 } _ { ( 3 ) }$ </td><td>82.27</td><td>83.52</td><td> $\mathbf { 8 7 . 1 8 } _ { ( 2 ) }$ </td><td> $\mathbf { 8 7 . 3 1 } _ { ( 1 ) }$ </td></tr><tr><td>60%</td><td>78.96</td><td>77.51</td><td>77.14</td><td>77.11</td><td>83.40</td><td>77.26</td><td>82.45</td><td> $8 6 . 7 2 _ { ( 3 ) }$ </td><td>82.45</td><td>84.19</td><td> $\mathbf { 8 7 . 4 4 } _ { ( 2 ) }$ </td><td> $\mathbf { 8 7 . 7 5 } _ { ( 1 ) }$ </td></tr><tr><td rowspan="3">AMAP</td><td>20%</td><td>93.37</td><td>92.82</td><td>67.80</td><td>93.28</td><td>93.76</td><td>92.94</td><td>93.17</td><td> $9 4 . 0 3 _ { ( 3 ) }$ </td><td>93.61</td><td>89.46</td><td> $9 4 . 8 7 _ { ( 1 ) }$ </td><td> $\mathbf { 9 4 . 8 5 } _ { ( 2 ) }$ </td></tr><tr><td>40%</td><td>93.67</td><td>93.16</td><td>72.84</td><td>93.54</td><td>94.03</td><td>93.12</td><td>93.97</td><td> $\mathbf { 9 4 . 8 1 } _ { ( 3 ) }$ </td><td>93.96</td><td>91.01</td><td> $9 5 . 3 7 _ { ( 1 ) }$ </td><td> $9 5 . 3 6 _ { ( 2 ) }$ </td></tr><tr><td>60%</td><td>93.66</td><td>93.24</td><td>66.92</td><td>93.83</td><td>94.26</td><td>93.28</td><td>94.31</td><td> $9 5 . 2 1 _ { ( 3 ) }$ </td><td>94.09</td><td>91.60</td><td> $9 5 . 5 2 _ { ( 2 ) }$ </td><td> $9 5 . 5 5 _ { ( 1 ) }$ </td></tr><tr><td rowspan="3">DBLP</td><td>20%</td><td>81.93</td><td>71.38</td><td>79.72</td><td>81.61</td><td>81.65</td><td>81.38</td><td>79.89</td><td>81.40</td><td> $\mathbf { 8 2 . 4 0 } _ { ( 3 ) }$ </td><td>79.46</td><td> $\mathbf { 8 3 . 0 8 } _ { ( 1 ) }$ </td><td> $\mathbf { 8 2 . 6 6 } _ { ( 2 ) }$ </td></tr><tr><td>40%</td><td>83.12</td><td>73.17</td><td>81.87</td><td>82.79</td><td>82.68</td><td>82.77</td><td>81.54</td><td>83.07</td><td> ${ \bf 8 3 . 4 6 } _ { ( 3 ) }$ </td><td>81.23</td><td> $\mathbf { 8 4 . 0 4 } _ { ( 1 ) }$ </td><td> $\mathbf { 8 3 . 8 0 } _ { ( 2 ) }$ </td></tr><tr><td>60%</td><td>84.06</td><td>74.09</td><td>83.03</td><td>83.32</td><td>83.34</td><td>83.63</td><td>82.64</td><td>83.89</td><td> $\mathbf { 8 4 . 2 4 } _ { ( 2 ) }$ </td><td>81.63</td><td> $\mathbf { 8 4 . 5 0 } _ { ( 1 ) }$ </td><td> $\mathbf { 8 4 . 1 0 } _ { ( 3 ) }$ </td></tr><tr><td rowspan="3">EAT</td><td>20%</td><td> $4 7 . 0 9$ </td><td>41.15</td><td> $5 1 . 8 5 _ { ( 2 ) }$ </td><td>34.38</td><td>34.45</td><td>36.30</td><td> $4 7 . 7 1 _ { ( 3 ) }$ </td><td>46.30</td><td> $5 4 . 5 3 _ { ( 1 ) }$ </td><td>39.31</td><td>47.00</td><td>47.52</td></tr><tr><td>40%</td><td> ${ \mathfrak { s o . 4 0 } } _ { ( 3 ) }$ </td><td>41.26</td><td> $5 4 . 6 3 _ { ( 2 ) }$ </td><td>39.24</td><td>38.29</td><td>37.91</td><td>48.18</td><td>49.05</td><td> $5 5 . 1 4 _ { ( 1 ) }$ </td><td>43.57</td><td>49.15</td><td>49.29</td></tr><tr><td>60%</td><td> $5 1 . 0 7 _ { ( 3 ) }$ </td><td>40.99</td><td> $5 4 . 5 2 _ { ( 2 ) }$ </td><td>38.87</td><td>41.52</td><td>38.04</td><td>44.77</td><td>50.14</td><td> ${ \bar { \mathbf { 5 8 . 3 5 } } } _ { ( 1 ) }$ </td><td>45.73</td><td>49.23</td><td>48.84</td></tr><tr><td rowspan="3">FILM</td><td>20%</td><td>28.03</td><td>25.97</td><td>26.89</td><td>27.89</td><td>27.98</td><td>28.80</td><td>28.48</td><td>35.26</td><td>28.16</td><td> $3 5 . 9 0 _ { ( 3 ) }$ </td><td> $3 6 . 1 2 _ { ( 2 ) }$ </td><td> $3 6 . 5 6 _ { ( 1 ) }$ </td></tr><tr><td>40%</td><td>28.73</td><td>26.36</td><td>28.06</td><td>28.35</td><td>28.30</td><td>29.41</td><td>29.87</td><td>35.93</td><td>29.30</td><td> $3 6 . 9 8 _ { ( 3 ) }$ </td><td> $3 6 . 9 9 _ { ( 2 ) }$ </td><td> $3 7 . 8 9 _ { ( 1 ) }$ </td></tr><tr><td>60%</td><td>28.97</td><td>26.65</td><td>28.64</td><td>28.61</td><td>28.60</td><td>29.63</td><td>30.69</td><td>36.99</td><td>29.70</td><td> $3 7 . 6 6 _ { ( 3 ) }$ </td><td> $3 7 . 7 4 _ { ( 2 ) }$ </td><td> $3 8 . 3 0 _ { ( 1 ) }$ </td></tr><tr><td rowspan="3">PubMed40%</td><td>20%</td><td>68.55</td><td>57.66</td><td>79.13</td><td>67.35</td><td>70.72</td><td>68.98</td><td>77.15</td><td>84.94</td><td>73.84</td><td> $\mathbf { 8 5 . 9 9 } _ { ( 2 ) }$ </td><td> $\mathbf { 8 5 . 5 3 } _ { ( 3 ) }$ </td><td> $\mathbf { 8 6 . 6 7 } _ { ( 1 ) }$ </td></tr><tr><td></td><td>68.66</td><td>58.01</td><td>82.13</td><td>67.34</td><td>71.14</td><td>69.11</td><td>77.92</td><td> $\mathbf { 8 6 . 6 1 } _ { ( 3 ) }$ </td><td>74.32</td><td> $\mathbf { 8 6 . 7 0 } _ { ( 2 ) }$ </td><td> $8 6 . 4 8 $ </td><td> ${ \bf 8 7 . 1 2 } _ { ( 1 ) }$ </td></tr><tr><td>60%</td><td>68.82</td><td>58.16</td><td>83.55</td><td>67.43</td><td>71.42</td><td>69.25</td><td>78.32</td><td> $\mathbf { 8 7 . 6 8 } _ { ( 1 ) }$ </td><td>74.58</td><td> ${ \bf 8 7 . 2 1 } _ { ( 3 ) }$ </td><td>86.91</td><td> ${ \mathbf { 8 7 . 3 7 } } _ { ( 2 ) }$ </td></tr><tr><td rowspan="3">Texas</td><td>20%</td><td>54.83</td><td>52.87</td><td>52.51</td><td>54.34</td><td>54.44</td><td>53.51</td><td> $5 5 . 3 7 _ { ( 3 ) }$ </td><td>54.39</td><td>54.83</td><td> ${ \bar { \mathbf { 5 9 . 1 5 } } } _ { ( 2 ) }$ </td><td> $6 2 . 3 5 _ { ( 1 ) }$ </td><td>54.68</td></tr><tr><td>40%</td><td>54.71</td><td>53.01</td><td>52.21</td><td>53.80</td><td>54.09</td><td>54.28</td><td> ${ \bar { \bf 5 9 . 2 4 } } _ { ( 3 ) }$ </td><td>55.07</td><td>56.78</td><td> $6 2 . 7 2 _ { ( 2 ) }$ </td><td> $\mathbf { 6 6 . 9 6 } _ { ( 1 ) }$ </td><td>54.09</td></tr><tr><td>60%</td><td>56.19</td><td>54.35</td><td>53.10</td><td>54.40</td><td>54.40</td><td>52.50</td><td> ${ \mathfrak { s o } } . 8 8 _ { ( 3 ) }$ </td><td>54.35</td><td>58.45</td><td> $\mathbf { 6 9 . 9 4 } _ { ( 2 ) }$ </td><td> $7 5 . 0 6 _ { ( 1 ) }$ </td><td>54.40</td></tr></table>

Bes $\mathrm { t } _ { ( 1 ) } ,$ second-best<sub>(2)</sub>, and third-best<sub>(3)</sub> mark the top three results in each row.

To analyze the contribution of each component, ablation studies are conducted for PreGS and PreGSv2 under the 20% training ratio. The results are shown in Table 4. The Full Model rows report the accuracy of the complete models, while the remaining rows report the accuracy change relative to the corresponding full model.

For PreGS, using only raw features leads to clear performance drops on most datasets, confirming the importance of graph structural information. Removing the final GAT logits also causes declines on several datasets, indicating that the high-level structural information from the pretrained GAT is useful for final prediction. By contrast, removing the GAT1-fused features or GS-fused features has a more dataset-dependent effect. In particular, removing GS-fused features causes a large drop on EAT, showing that the transferred GraphSAGE experts provide useful complementary neighborhood information on this dataset.

For PreGSv2, removing weighted concatenation or gating causes only small changes on most datasets, but clear drops are observed on PubMed. This suggests that source-level weighting and gating are helpful when different feature sources contribute unevenly. However, the positive changes on Texas also indicate that these adaptive modules are not universally beneficial for every graph. Overall, the ablation results show that the main components have meaningful but dataset-dependent contributions.

## E. Sensitivity to the Number ofGATHeads

To examine the sensitivity of the proposed framework to the number of GAT heads, we vary the number of attention heads under the 20% training ratio and compare GAT, PreGS, and PreGSv2. In this experiment, all transferred GraphSAGE experts use the mean aggregator to isolate the effect of the number of attention heads. The results are shown in Table 5.

Across all selected datasets and head settings, PreGS and PreGSv2 consistently outperform the original GAT. PreGS achieves the best performance on ACM and DBLP, while PreGSv2 performs best on AMAC and FILM. This indicates that the proposed framework can obtain stable gains under different head settings, and that the two variants show different advantages across datasets.

TABLE 3. Effectiveness of Head-Wise Parameter Transfer Under the 20% Training Ratio
<table><tr><td>Dataset</td><td>GAT</td><td>GraphSAGE</td><td>Mean-1</td><td>Mean-2</td><td>Max-1</td><td>Max-2</td><td>Max-3</td><td>Sum-1</td><td>Sum-2</td><td>Sum-3</td></tr><tr><td>ACM</td><td>90.91</td><td>90.80</td><td>88.76</td><td>88.24</td><td>88.08</td><td>87.42</td><td>82.91</td><td>87.90</td><td>88.50</td><td>86.56</td></tr><tr><td>AMAC</td><td>76.48</td><td>76.91</td><td>70.19</td><td>66.35</td><td>47.11</td><td>43.99</td><td>40.42</td><td>54.28</td><td>67.64</td><td>64.45</td></tr></table>

All values denote node classification accuracy (%).

TABLE 4. Ablation Study of PreGS and PreGSv2 Under the 20% Training Ratio
<table><tr><td>Model</td><td>Variant</td><td>ACM</td><td>AMAC</td><td>AMAP</td><td>DBLP</td><td>EAT</td><td>FILM</td><td>PubMed</td><td>Texas</td></tr><tr><td rowspan="6">PreGS</td><td>Full Model</td><td>92.64</td><td>85.91</td><td>94.90</td><td>83.08</td><td>46.94</td><td>36.06</td><td>85.47</td><td>62.38</td></tr><tr><td>Raw Features Only</td><td>-4.32</td><td>-2.23</td><td>-4.15</td><td>-2.98</td><td>-5.78</td><td>-0.36</td><td>0.53</td><td>1.65</td></tr><tr><td>w/o GAT Final</td><td>-1.38</td><td>-0.38</td><td>-1.28</td><td>-1.51</td><td>0.43</td><td>-0.37</td><td>0.06</td><td>1.55</td></tr><tr><td>w/o GAT1-Fused</td><td>-0.27</td><td>0.30</td><td>-0.05</td><td>-0.12</td><td>0.31</td><td>0.03</td><td>0.34</td><td>0.44</td></tr><tr><td>w/o GS-Fused</td><td>-0.16</td><td>-0.09</td><td>0.15</td><td>-0.10</td><td>-6.86</td><td>0.17</td><td>0.21</td><td>-0.18</td></tr><tr><td>w/o Raw Features</td><td>-1.81</td><td>-4.34</td><td>-1.43</td><td>-1.97</td><td>2.33</td><td>-7.64</td><td>-9.36</td><td>-7.55</td></tr><tr><td rowspan="3">PreGSv2</td><td>Full Model</td><td>92.35</td><td>86.21</td><td>94.92</td><td>82.66</td><td>47.46</td><td>36.57</td><td>86.65</td><td>54.68</td></tr><tr><td>w/o Weighted Concat</td><td>0.01</td><td>0.08</td><td>0.00</td><td>0.21</td><td>-0.50</td><td>-0.17</td><td>-1.24</td><td>1.44</td></tr><tr><td>w/o Gating</td><td>0.11</td><td>-0.04</td><td>-0.02</td><td>0.20</td><td>0.16</td><td>-0.01</td><td>-0.57</td><td>1.21</td></tr></table>

Full Model rows report accuracy values. Other rows report accuracy changes relative to the corresponding full model.

TABLE 5. Sensitivity to the Number of GAT Heads Under the 20% Training Ratio
<table><tr><td>Dataset</td><td>Heads</td><td>GAT</td><td>PreGS</td><td>PreGSv2</td></tr><tr><td rowspan="3">ACM</td><td>2</td><td> $9 0 . 9 1 _ { ( 3 ) }$ </td><td> $9 2 . 4 2 _ { ( 1 ) }$ </td><td> $9 2 . 2 7 _ { ( 2 ) }$ </td></tr><tr><td>4</td><td> $\mathbf { 9 0 . 9 3 } _ { ( 3 ) }$ </td><td> $\mathbf { 9 2 . 3 9 } _ { ( 1 ) }$ </td><td> $9 2 . 1 8 _ { ( 2 ) }$ </td></tr><tr><td>8</td><td> $9 0 . 9 2 _ { ( 3 ) }$ </td><td> $9 2 . 6 2 _ { ( 1 ) }$ </td><td> $9 2 . 2 \mathbf { 0 } _ { ( 2 ) }$ </td></tr><tr><td rowspan="3">AMAC</td><td>2</td><td> ${ \bf 6 9 . 9 5 } _ { ( 3 ) }$ </td><td> $8 5 . 5 9 _ { ( 2 ) }$ </td><td> ${ \bf 8 5 . 9 0 } _ { ( 1 ) }$ </td></tr><tr><td>4</td><td> $7 5 . 6 3 ( _ { ( 3 ) }$ </td><td> $\mathbf { 8 6 . 0 0 } _ { ( 2 ) }$ </td><td> $\mathbf { 8 6 . 1 0 } _ { ( 1 ) }$ </td></tr><tr><td>8</td><td> $7 6 . 2 0 _ { ( 3 ) }$ </td><td> $8 5 . 6 3 _ { ( 2 ) }$ </td><td> $8 6 . 0 2 _ { ( 1 ) }$ </td></tr><tr><td rowspan="3">DBLP</td><td>2</td><td> $8 0 . 6 6 _ { ( 3 ) }$ </td><td> $8 2 . 8 4 _ { ( 1 ) }$ </td><td> $8 2 . 4 2 _ { ( 2 ) }$ </td></tr><tr><td>4</td><td> $8 1 . 0 4 _ { ( 3 ) }$ </td><td> $8 2 . 8 7 _ { ( 1 ) }$ </td><td> ${ \mathbf { 8 2 . 3 6 } } _ { ( 2 ) }$ </td></tr><tr><td>8</td><td> $\mathbf { 8 1 . 6 1 } _ { ( 3 ) }$ </td><td> $8 3 . 0 4 _ { ( 1 ) }$ </td><td> $\mathbf { 8 2 . 7 1 } _ { ( 2 ) }$ </td></tr><tr><td rowspan="3">FILM</td><td>2</td><td> $2 7 . 3 4 _ { ( 3 ) }$ </td><td> $3 5 . 9 3 _ { ( 2 ) }$ </td><td> $3 6 . 8 4 _ { ( 1 ) }$ </td></tr><tr><td>4</td><td> $2 7 . 5 3 _ { ( 3 ) }$ </td><td> $3 5 . 8 6 _ { ( 2 ) }$ </td><td> $3 6 . 7 8 _ { ( 1 ) }$ </td></tr><tr><td>8</td><td> $2 7 . 8 9 _ { ( 3 ) }$ </td><td> $3 6 . 1 0 _ { ( 2 ) }$ </td><td> $3 6 . 6 8 _ { ( 1 ) }$ </td></tr></table>

$\mathbf { B e s t } _ { ( 1 ) } .$ , second-bes $_ ( 2 ) ,$ , and third-best mark the top three results in each row.

Although the performance of GAT may improve when more heads are used on some datasets, the proposed models maintain clear improvements over GAT for 2, 4, and 8 heads. Therefore, the performance gain does not simply come from changing the number of attention heads, but mainly from parameter transfer, multi-expert aggregation, and multi-source feature fusion.

## F. Aggregator Combination Analysis

To examine the effect of expert aggregation strategies, we compare representative single-aggregator and mixedaggregator configurations under the 20% training ratio. Table 6 reports the best result and its corresponding configuration for each dataset.

TABLE 6. Best Aggregator Configurations Under the 20% Training Ratio
<table><tr><td rowspan="2">Dataset</td><td colspan="2">PreGS</td><td colspan="2">PreGSv2</td></tr><tr><td>Acc. (%)</td><td>Config.</td><td>Acc. (%)</td><td>Config.</td></tr><tr><td>ACM</td><td>92.65</td><td>Mixed</td><td>92.40</td><td>Max-Sum</td></tr><tr><td>AMAC</td><td>85.96</td><td>Mean-Sum</td><td>86.23</td><td>All-Sum</td></tr><tr><td>AMAP</td><td>95.06</td><td>All-Max</td><td>95.09</td><td>All-Max</td></tr><tr><td>DBLP</td><td>83.09</td><td>All-Max</td><td>82.73</td><td>Mean-Sum</td></tr><tr><td>EAT</td><td>47.43</td><td>All-Sum</td><td>48.22</td><td>Max-Sum</td></tr><tr><td>FILM</td><td>36.13</td><td>All-Mean</td><td>36.68</td><td>All-Mean</td></tr><tr><td>PubMed</td><td>85.59</td><td>All-Mean</td><td>86.66</td><td>Mean-Sum</td></tr><tr><td>Texas</td><td>63.28</td><td>All-Max</td><td>54.68</td><td>All-Mean</td></tr></table>

Mixed denotes (mean, mean, max, max, max, sum, sum, sum).

The optimal configuration varies across datasets and model variants, indicating that no single aggregation strategy is universally optimal. This result supports the use of heterogeneous experts to capture complementary neighborhood information.

## G. Embedding Visualization Analysis

To qualitatively compare the learned node representations, we visualize the embeddings of twelve models on AMAC under the 20% training ratio using t-distributed stochastic neighbor embedding (t-SNE). As shown in Figures 2 and 3, several baselines still exhibit noticeable class overlap. In comparison, PreGS forms more organized local clusters, while PreGSv2 shows clearer separation in several regions, consistent with its higher accuracy on AMAC.

![](images/00995e40fe27527f72e938fdba1a330e6c4c93c1444c8f6de4dee0bcc637d58d.jpg)  
FIGURE 2. t-SNE visualization on AMAC: GCN, SGC, GIN, GAT, GATv2, and GraphSAGE.

![](images/4a3cb93530b3eabc11990fe27478aa46a2f4f6a2112a0ba22db1ee7b4581b173.jpg)  
FIGURE 3. t-SNE visualization on AMAC: GraphSAGE++, GNNMoE, JK-Net, MLP, PreGS, and PreGSv2.

TABLE 7. Training Time Comparison on ACM and AMAC
<table><tr><td>Dataset</td><td>MLP</td><td>GAT</td><td>GNNMoE</td><td>PreGS</td><td>PreGSv2</td></tr><tr><td>ACM</td><td>0.50</td><td>2.19</td><td>5.08</td><td>3.33</td><td>3.81</td></tr><tr><td>AMAC</td><td>1.08</td><td>8.49</td><td>13.58</td><td>10.76</td><td>11.03</td></tr></table>

All values are reported in seconds.

## H. Training Time Analysis

To evaluate the computational overhead of the proposed models, we compare the end-to-end training convergence time of MLP, GAT, GNNMoE, PreGS, and PreGSv2 on ACM and AMAC under the 20% training ratio. For PreGS and PreGSv2, the reported time includes GAT pretraining and the subsequent fusion-training stage. The results are reported in Table 7.

As shown in Table 7, MLP requires the least training time because it does not involve graph-based message passing. PreGS and PreGSv2 incur additional computational cost compared with GAT because of the parameter-transfer and fusion-training stages. Nevertheless, both proposed models are consistently more efficient than GNNMoE on the two datasets. This result indicates that the proposed framework introduces moderate additional overhead while maintaining a more favorable training efficiency than the competing graph mixture-of-experts model.

## VI. CONCLUSION

This paper proposed PreGS, a parameter-transfer-based multi-expert graph neural network framework for node classification. PreGS transfers the first-layer attention-head parameters of a pretrained GAT to multiple GraphSAGE experts and freezes both the GAT and the transferred experts during subsequent training. Based on PreGS, PreGSv2 further introduces source-level weighting and structural gating to improve adaptive multi-source feature fusion.

Experiments on eight public graph datasets show that PreGS and PreGSv2 achieve competitive performance against representative GNN baselines. The parametertransfer, ablation, sensitivity, aggregator, visualization, and training-time analyses further support the effectiveness of the proposed framework. Future work will extend the framework to larger-scale graphs and other graph learning tasks.

## REFERENCES

[1] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl, “Neural message passing for quantum chemistry,” in Proc. 34th Int. Conf. Mach. Learn., Sydney, NSW, Australia, 2017, pp. 1263–1272.

[2] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in Proc. 5th Int. Conf. Learn. Represent., Toulon, France, Apr. 24–26, 2017.

[3] W. L. Hamilton, R. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” in Proc. 31st Int. Conf. Neural Inf. Process. Syst., Long Beach, CA, USA, Dec. 4–9, 2017, pp. 1024–1034.

[4] P. Velickoviˇ c, G. Cucurull, A. Casanova, A. Romero, P. Li´ o, and Y.\` Bengio, “Graph attention networks,” in Proc. 6th Int. Conf. Learn. Represent., Vancouver, BC, Canada, Apr. 30–May 3, 2018.

[5] S. Brody, U. Alon, and E. Yahav, “How attentive are graph attention networks?,” in Proc. 10th Int. Conf. Learn. Represent., Virtual Event, Apr. 25–29, 2022.

[6] J. E, Y. Zhang, X. Xia, and X. Xu, “TANGNN: A concise, scalable and effective graph neural networks with top-m attention mechanism for graph representation learning,” Expert Syst. Appl., vol. 271, May 2025, Art. no. 126599, doi: 10.1016/j.eswa.2025.126599.

[7] F. Wu, A. Souza, T. Zhang, C. Fifty, T. Yu, and K. Weinberger, “Simplifying graph convolutional networks,” in Proc. 36th Int. Conf. Mach. Learn., Long Beach, CA, USA, Jun. 9–15, 2019, pp. 6861– 6871.

[8] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How powerful are graph neural networks?,” in Proc. 7th Int. Conf. Learn. Represent., New Orleans, LA, USA, May 6–9, 2019.

[9] K. Xu, C. Li, Y. Tian, T. Sonobe, K.-I. Kawarabayashi, and S. Jegelka, “Representation learning on graphs with jumping knowledge networks,” in Proc. 35th Int. Conf. Mach. Learn., Stockholm, Sweden, Jul. 10–15, 2018, pp. 5453–5462.

[10] G. Corso, L. Cavalleri, D. Beaini, P. Lio, and P. Veli \` ckovi ˇ c, “Principal´ neighbourhood aggregation for graph nets,” in Proc. 34th Conf. Neural Inf. Process. Syst., Virtual Event, Dec. 6–12, 2020, pp. 13260–13271.

[11] X. Wang, M. Zhu, D. Bo, P. Cui, C. Shi, and J. Pei, “AM-GCN: Adaptive multi-channel graph convolutional networks,” in Proc. 26th ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., Virtual Event, CA, USA, Aug. 23–27, 2020, pp. 1243–1253, doi: 10.1145/3394486.3403177.

[12] C. Deng, Z. Yue, and Z. Zhang, “Polynormer: Polynomial-expressive graph transformer in linear time,” in Proc. 12th Int. Conf. Learn. Represent., Vienna, Austria, May 7–11, 2024.

[13] D. Fu, Z. Hua, Y. Xie, J. Fang, S. Zhang, K. Sancak, H. Wu, A. Malevich, J. He, and B. Long, “VCR-Graphormer: A mini-batch graph transformer via virtual connections,” in Proc. 12th Int. Conf. Learn. Represent., Vienna, Austria, May 7–11, 2024.

[14] J. Zhou, J. Xia, S. Li, Y. Liu, W. Wang, Y. Huang, C. Chi, M. Hong, Z. Ouyang, S. Wang, Z. Wang, X. Wu, C. Yu, and S. Z. Li, “VecFormer: Towards efficient and generalizable graph transformer with graph token attention,” in Proc. ACM Web Conf., Dubai, United Arab Emirates, Jun. 29–Jul. 3, 2026, pp. 1115–1126, doi: 10.1145/3774904.3792453.

[15] Y. Luo, L. Shi, and X.-M. Wu, “Classic GNNs are strong baselines: Reassessing GNNs for node classification,” in Adv. Neural Inf. Process. Syst., vol. 37, Vancouver, BC, Canada, Dec. 10–15, 2024, pp. 97650– 97669, doi: 10.52202/079017-3098.

[16] Y. Luo, L. Shi, and X.-M. Wu, “Can classic GNNs be strong baselines for graph-level tasks? Simple architectures meet excellence,” in Proc. 42nd Int. Conf. Mach. Learn., Vancouver, BC, Canada, Jul. 13–19, 2025, pp. 41290–41310.

[17] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. V. Le, G. E. Hinton, and J. Dean, “Outrageously large neural networks: The sparsely-gated mixture-of-experts layer,” in Proc. 5th Int. Conf. Learn. Represent., Toulon, France, Apr. 24–26, 2017.

[18] W. Fedus, B. Zoph, and N. Shazeer, “Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity,” J. Mach. Learn. Res., vol. 23, no. 120, pp. 1–39, 2022.

[19] Y. Shi, Y. Wang, W. Liang, J. Zhang, P. Dong, and A. Li, “Mixture of experts for node classification,” in Proc. Int. Conf. Multimedia Retrieval, Chicago, IL, USA, Jun. 30–Jul. 3, 2025, pp. 1154–1162, doi: 10.1145/3731715.3733392.

[20] X. Chen, J. Zhou, S. Yu, and Q. Xuan, “Mixture of experts meets decoupled message passing: Towards general and adaptive node classification,” in Companion Proc. ACM Web Conf., Sydney, NSW, Australia, Apr. 28–May 2, 2025, pp. 907–910, doi: 10.1145/3701716.3715462.

[21] H. Zeng, H. Lyu, D. Hu, Y. Xia, and J. Luo, “Mixture of weak and strong experts on graphs,” in Proc. 12th Int. Conf. Learn. Represent., Vienna, Austria, May 7–11, 2024.

[22] J. Sun, M. A. Hassan, Y. Zhang, W. Zhang, and C.-G. Lee, “Diverse and sparse mixture-of-experts for causal subgraph-based out-ofdistribution graph learning,” in Proc. 14th Int. Conf. Learn. Represent., Rio de Janeiro, Brazil, Apr. 23–27, 2026.

[23] S. Zhang, Y. Liu, Y. Sun, and N. Shah, “Graph-less neural networks: Teaching old MLPs new tricks via distillation,” in Proc. 10th Int. Conf. Learn. Represent., Virtual Event, Apr. 25–29, 2022.

[24] Y. Tian, S. Xu, and M. Li, “Decoupled graph knowledge distillation: A general logits-based method for learning MLPs on graphs,” Neural Netw., vol. 179, Nov. 2024, Art. no. 106567, doi: 10.1016/j.neunet.2024.106567.

[25] P. Rumiantsev and M. Coates, “Graph knowledge distillation to mixture of experts,” Trans. Mach. Learn. Res., Oct. 2024.

[26] A. Eskandari, A. Anand, E. Rashno, and F. Zulkernine, “InfGraND: An influence-guided GNN-to-MLP knowledge distillation,” Trans. Mach. Learn. Res., Jan. 2026.

[27] J. Zhang, C. Xie, B. Yu, and R. Yang, “Adaptive hierarchical knowledge distillation from GNNs to MLPs,” Knowl. Inf. Syst., vol. 67, no. 9, pp. 7619–7639, Sep. 2025, doi: 10.1007/s10115-025-02447-w.

[28] X. Wang, H. Ji, C. Shi, B. Wang, P. Cui, P. S. Yu, and Y. Ye, “Heterogeneous graph attention network,” in Proc. World Wide Web Conf., San Francisco, CA, USA, May 13–17, 2019, pp. 2022–2032, doi: 10.1145/3308558.3313562.

[29] Z. Yang, W. W. Cohen, and R. Salakhutdinov, “Revisiting semisupervised learning with graph embeddings,” in Proc. 33rd Int. Conf. Mach. Learn., New York, NY, USA, Jun. 19–24, 2016, pp. 40–48.

[30] L. F. R. Ribeiro, P. H. P. Savarese, and D. R. Figueiredo, “struc2vec: Learning node representations from structural identity,” in Proc. 23rd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., Halifax, NS, Canada, Aug. 13–17, 2017, pp. 385–394, doi: 10.1145/3097983.3098061.

[31] H. Pei, B. Wei, K. C.-C. Chang, Y. Lei, and B. Yang, “Geom-GCN: Geometric graph convolutional networks,” in Proc. 8th Int. Conf. Learn. Represent., Virtual Conf., Apr. 26–30, 2020.

[32] J. E, Y. Zhang, S. Yang, H. Wang, X. Xia, and X. Xu, “GraphSAGE++: Weighted multi-scale GNN for graph representation learning,” Neural Process. Lett., vol. 56, no. 1, Feb. 2024, Art. no. 24, doi: 10.1007/s11063-024-11496-1.