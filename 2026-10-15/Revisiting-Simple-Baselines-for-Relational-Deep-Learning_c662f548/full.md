# Revisiting Simple Baselines for Relational Deep Learning

Viet The Nguyen<sup>1</sup>, Vijay Prakash Dwivedi<sup>2,3</sup>, Gunther Gust<sup>1</sup> <sup>1</sup>University of Würzburg, <sup>2</sup>Seamless Labs, Inc., <sup>3</sup>Stanford University {the-viet.nguyen,gunther.gust}@uni-wuerzburg.de dwi@seamlesslabs.ai

## Abstract

Relational deep learning has emerged as a powerful paradigm for learning end-toend predictive models over relational databases by representing them as temporal heterogeneous graphs. Recent advances based on graph neural networks and Transformers have achieved strong performance across diverse domains. However, the effectiveness of simpler relational baselines remains underexplored, making it difficult to assess how much architectural complexity is needed to achieve strong predictive performance. Motivated by the observation that predictive signals arise from target attributes, relational attributes, and historical labels, we revisit how effectively these signals can be exploited using simple propagation methods. We adapt fixed low-pass feature propagation to relational entity graphs and train an MLP for prediction. The resulting predictions are further refined using Correct-and-Smooth, a graph-based label propagation algorithm that incorporates historical labels over task-specific meta-graphs connecting target entities. Across classification and regression tasks on RelBench, an MLP trained on neighborhood-filtered features outperforms one using only target-entity features on most tasks and recovers a substantial portion of the performance of heterogeneous GNNs. Correct-and-Smooth provides further gains from historical labels, although applying it requires a separately constructed meta-graph because connections among target entities are not directly specified by the relational entity graph. Together, our studies revisit the effectiveness of established propagation methods as simple RDL baselines while identifying canonical meta-graph construction as a key challenge for effective label propagation.

## 1 Introduction

Data in domains such as e-commerce, healthcare, and social media is predominantly stored in relational databases, where tables are linked through primary– foreign key relationships. To reduce feature engineering efforts, relational deep learning (RDL) provides a framework for predictive modeling directly over such databases by representing each table row as a node in a temporal heterogeneous graph, whose edges encode primary–foreign key relationships [1, 2]. This representation enables graph neural networks (GNNs) and transformer-based models to aggregate information across related entities, and has established RDL as an effective paradigm for learning on large-scale relational databases [3–8].

![](images/f4341b20c9966ddf920a31e7802fc63264dd6af3d2023dae6abf65fcd4e009bb.jpg)  
Figure 1: Example of graph conversion. (Left) A relational entity graph, where each node is an entity, and edges are primary– foreign key links. (Right) All nodes are collapsed into a single type, and feature sets are merged and normalized.

Early approaches primarily adapted heterogeneous GNNs with relation-aware message passing to relational databases [2–4]. Recent advances adopted transformer-based architectures with the aim of building relational foundation models, which achieved strong performance on large-scale benchmarks [6, 8]. In parallel, closed-source relational foundation models demonstrated that largescale pretraining produces representations that transfer across databases and tasks [5, 7]. Across these models, the predictive signal on an entity-level task can be grouped into three sources: the attributes of the target entity, the attributes of entities related to it through foreign keys, and the historical labels of the target and of related entities. These models utilize target and relational attributes by sampling a temporal neighborhood around each target entity and aggregating its features through GNN message passing [2, 4] or Transformer attention [6, 8]. Recent work further shows that historical labels can add substantial predictive signals to improve zero-shot predictions [6].

Simple counterparts for exploiting these three sources are well established in graph learning. Specif ically, an $\mathrm { M L P }$ operates directly on node features, fixed neighborhood propagation incorporates information from neighboring nodes without learned message passing [9], and Correct-and-Smooth (C&S) propagates label information over a graph [10]. However, these methods are typically studied on homogeneous static graphs and cannot be applied to relational entity graphs directly. RDL graphs are heterogeneous, with nodes from different tables carrying different feature spaces. Additionally, they are temporal, requiring propagation to respect the time of each prediction, and labels are available only for the target entities of a task, rather than for all nodes in the graph. In particular, the graph induced by the database schema is not necessarily an appropriate graph for propagating task labels.

In this work, we make three contributions. First, we adapt established propagation methods to a simple RDL baseline that separates the three sources of predictive signal. We train an MLP on target attributes, then train the same MLP architecture on features enriched by fixed relational propagation [9], before refining its predictions with historical labels through the post-processing step with C&S [10]. Second, we evaluate how much performance these stages recover on RelBench [11, 12]: feature propagation improves over the unfiltered MLP on 20 of 21 tasks and brings average classification AUROC within two points of the supervised heterogeneous GNN baseline. Finally, we study task-specific meta-graphs for label propagation and highlight canonical, temporally valid graph construction as a key challenge for exploiting historical labels in RDL.

## 2 Simple Propagation for Relational Deep Learning

We describe how feature and label propagation incorporate the three sources of predictive signal into the baseline. Appendices B and C give the feature construction, filter definition, and C&S updates.

## 2.1 Relational Entity Graphs

A relational database induces an entity graph $G = ( \nu , \mathcal { E } )$ , where nodes are table rows and edges represent primary–foreign key links [2]. Each row carries attributes and, when applicable, a timestamp. An entity-level task provides labeled examples (e, t), where e is the target entity and t is the seed time; predictions may use only information available by t. While RDL models typically learn relationspecific aggregation on this graph [2, 4], we use it for fixed feature propagation before training an MLP.

## 2.2 Feature Propagation

Homogeneous relational graph. We combine all primary–foreign key links into one adjacency matrix and place each table’s encoded attributes in a separate block of feature columns. This gives a homogeneous propagation graph while keeping attributes from different tables distinct (Figure 1). Appendix B describes the encoding and normalization.

Low-pass propagation with temporal visibility. For each seed time t, we remove rows with later timestamps, retain static rows, and apply a fixed low-pass filter [9] to the visible graph. Each propagation step combines a node’s features with those of its neighbors using symmetric degree normalization. We then train an MLP on the resulting representations, using $K = 2$ steps in the main experiments. The $K = 0$ baseline uses target attributes alone, with structural features disabled. More details on the filtering process and feature definitions are also provided in Appendix B.

## 2.3 Label Propagation with Correct-and-Smooth

After training on the filtered features, we incorporate historical labels through C&S [10] to refine the MLP predictions. In particular, the Correct stage propagates residuals from labeled nodes to adjust the MLP predictions, and the Smooth stage propagates these predictions with observed labels

Table 1: Entity classification on RelBench-v1 and RelBench-v2 (AUROC ↑).
<table><tr><td>Task</td><td colspan="4">MLP (target attributes) MLP (+relational attributes) C&amp;S (+historical labels) HeteroGNN</td></tr><tr><td colspan="6">RelBench-v1</td></tr><tr><td>amazon/user-churn</td><td> $5 3 . 6 5 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $6 8 . 1 2 { \scriptstyle \pm 0 . 0 5 }$ </td><td> ${ \bf 6 8 . 6 9 } { \scriptstyle \pm 0 . 0 4 }$ </td><td> $\underline { { 7     0 . 4 2 \pm 0 . 0 5 } }$ </td></tr><tr><td>amazon/item-churn</td><td> $6 6 . 6 0 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $8 0 . 5 8 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $\mathbf { 8 1 . 0 3 } 2 0 . 0 7$ </td><td> $\underline { { \mathbf { 8 2 . 8 1 } \pm 0 . 0 3 } }$ </td></tr><tr><td>avito/user-clicks</td><td> $5 1 . 7 4 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $6 5 . 0 8 { \scriptstyle \pm 1 . 0 6 }$ </td><td> $\mathbf { 6 5 . 1 0 { \scriptstyle \pm 1 . 0 6 } }$ </td><td> $\underline { { { \bf 6 5 . 9 0 \pm 1 . 9 5 } } }$ </td></tr><tr><td>avito/user-visits</td><td> $4 9 . 6 4 { \scriptstyle \pm 0 . 0 1 }$ </td><td> ${ \bf 6 6 . 2 9 } { \scriptstyle \pm 0 . 2 0 }$ </td><td> $\underline { { { \bf 6 6 . 4 1 \pm 0 . 1 8 } } }$ </td><td> $6 6 . 2 0 { \scriptstyle \pm 0 . 1 0 }$ </td></tr><tr><td>hm/user-churn</td><td> $5 5 . 2 5 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $6 8 . 2 1 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $\mathbf { 6 9 . 1 6 { \scriptstyle \pm 0 . 0 5 } }$ </td><td> $\underline { { { \bf 6 9 . 8 8 \pm 0 . 2 1 } } }$ </td></tr><tr><td>stack/user-badge</td><td> $5 1 . 9 3 { \scriptstyle \pm 1 . 8 7 }$ </td><td> $8 6 . 0 4 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $\mathbf { 8 6 . 1 6 { \scriptstyle \pm 0 . 0 9 } }$ </td><td> $\overline { { 8 8 . 8 6 \pm 0 . 0 8 } }$ </td></tr><tr><td>stack/user-engagement trial/study-outcome</td><td> $5 3 . 0 4 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $8 8 . 5 3 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $\mathbf { 8 9 . 0 2 } { \scriptstyle \pm 0 . 0 9 }$ </td><td> $\underline { { \mathbf { 7 0 . 5 9 2 0 . 0 9 } } }$ </td></tr><tr><td></td><td> $6 5 . 7 7 { \scriptstyle \pm 0 . 3 6 }$  55.95</td><td> $6 6 . 3 7 { \scriptstyle \pm 0 . 3 4 }$  73.65</td><td> ${ \bf 6 7 . 6 8 } { \scriptstyle \pm 0 . 3 9 }$  74.16</td><td> $\underline { { 6 8 . 6 0 \pm 1 . 0 1 } }$  75.41</td></tr><tr><td colspan="5">Average</td></tr><tr><td>RelBench-v2 ratebeer/user-churn</td><td> $6 3 . 9 8 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $\mathbf { 8 9 . 7 3 { \scriptstyle \pm 0 . 1 2 } }$ </td><td> $\mathbf { 8 9 . 7 3 { \scriptstyle \pm 0 . 1 1 } }$ </td><td> $\underline { { \ 9 4 . 2 7 \pm 0 . 2 2 } }$ </td></tr><tr><td>ratebeer/beer-churn</td><td> $6 2 . 4 0 { \scriptstyle \pm 0 . 8 4 }$ </td><td> $\underline { { 7 8 . 9 3 \pm 0 . 0 7 } }$ </td><td> $\mathbf { 7 8 . 9 0 { \scriptstyle \pm 0 . 0 8 } }$ </td><td> $7 8 . 6 7 { \scriptstyle \pm 0 . 6 0 }$ </td></tr><tr><td>ratebeer/brewer-dormant</td><td> $6 3 . 6 7 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $\overline { { 7 9 . 7 3 2 0 . 0 4 } }$ </td><td> $\mathbf { 7 9 . 7 8 { \scriptstyle \pm 0 . 0 4 } }$ </td><td> $\mathbf { 8 0 . 5 1 \pm 0 . 1 7 }$ </td></tr><tr><td>arxiv/paper-citation</td><td> $7 3 . 6 5 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $8 1 . 1 1 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $\mathbf { 8 1 . 1 7 \pm 0 . 0 5 }$ </td><td> $\overline { { 8 2 . 5 0 \pm 0 . 0 4 } }$ </td></tr><tr><td>Average</td><td>65.93</td><td>82.38</td><td>82.40</td><td>83.99</td></tr></table>

Table 2: Entity regression on RelBench-v1 and RelBench-v2 (MAE ↓).
<table><tr><td>Task</td><td colspan="3">MLP (target attributes) MLP (+relational attributes) C&amp;S (+historical labels)</td><td>HeteroGNN</td></tr><tr><td>RelBench-v1</td><td colspan="3"></td><td></td></tr><tr><td>amazon/user-ltv</td><td> $1 6 . 7 5 7 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $\mathbf { 1 4 . 6 5 6 { \scriptstyle \pm 0 . 0 1 3 } }$ </td><td> $\mathbf { 1 4 . 6 5 6 { \scriptstyle \pm 0 . 0 1 3 } }$ </td><td>14.313±0.013</td></tr><tr><td>amazon/item-ltv</td><td>59.958±0.150</td><td> $\mathbf { 5 0 . 9 9 4 } \mathrm { \pm 0 . 1 6 8 }$ </td><td>50.994±0.168</td><td>50.053±0.163</td></tr><tr><td>avito/ad-ctr</td><td>0.043±0.000</td><td> $\underline { { \mathbf { 0 . 0 4 0 } 2 0 . 0 0 3 } }$ </td><td> $\underline { { \mathbf { 0 . 0 4 0 } 2 0 . 0 0 3 } }$ </td><td>0.041±0.001</td></tr><tr><td>hm/item-sales</td><td>0.076±0.000</td><td> $\overline { { { \bf 0 . 0 6 0 } _ { \pm 0 . 0 0 0 } } }$ </td><td>0.060±0.000</td><td>0.056±0.000</td></tr><tr><td>stack/post-votes</td><td>0.068±0.000</td><td> $\mathbf { 0 . 0 6 6 } { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.066±0.000</td><td>0.065±0.000</td></tr><tr><td>trial/site-success</td><td>0.426±0.002</td><td> $0 . 4 3 7 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td>0.443±0.006</td><td>0.400±0.020</td></tr><tr><td>trial/study-adverse</td><td>48.203±0.676</td><td> $4 4 . 8 6 7 { \scriptstyle \pm 0 . 1 5 1 }$ </td><td> $\mathbf { 4 4 . 8 1 7 { \scriptstyle \pm 0 . 1 5 0 } }$ </td><td>44.473±0.209</td></tr><tr><td>RelBench-v2</td><td colspan="3"></td><td></td></tr><tr><td>ratebeer/user-count</td><td>14.453±0.006</td><td> $\mathbf { 9 . 1 1 5 { \scriptstyle \pm 0 . 0 7 5 } }$ </td><td> $\mathbf { 9 . 1 1 5 { \scriptstyle \pm 0 . 0 7 5 } }$ </td><td> $\underline { { 7 . 3 7 4 2 0 . 1 0 2 } }$ </td></tr><tr><td>arxiv/author-publication</td><td> $0 . 5 7 7 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $\underline { { \mathbf { 0 . 4 8 9 } } } \pm \mathbf { 0 . 0 0 2 }$ </td><td> $\underline { { \mathbf { 0 . 4 8 9 } \pm 0 . 0 0 2 } }$ </td><td> $\overline { { \mathbf { 0 . 5 1 3 } \pm \mathbf { 0 . 0 0 8 } } }$ </td></tr></table>

as anchors. Since task labels belong to target entities that may be connected only indirectly in the relational graph, we construct a separate meta-graph over their prediction rows and historical labels.

We compare meta-graphs that attach the target’s own history and related entities’ labels directly or through temporal chains. Each construction admits only labels observed strictly before the seed time (Appendix C.1) with propagation settings selected on validation before refining the MLP predictions.

## 3 Experiments

We evaluate whether the three simple propagation stages can provide a competitive relational baseline across the entity-level classification and regression tasks in RelBench-v1 [11] and RelBench-v2 [12]. We exclude rel-f1 and rel-event, for which temporal leakage has been reported [6, 13]. Following RelBench, classification is evaluated using AUROC [14] and regression using MAE.

Our experiments follow the three stages introduced in Section 2. The unfiltered MLP uses only the attributes of the target entity and corresponds to $K = 0$ . The filtered MLP uses $K = 2$ steps of fixed feature propagation and therefore incorporates attributes from related entities. Finally, filtered MLP + C&S applies label propagation to the filtered predictions using the graph variant that incorporates both historical labels and neighboring labels (CROSS+SELF), which is described in Appendix C.1. We select the C&S configuration on validation using the task metric (Appendix C). All reported results are averaged over ten random seeds. Task statistics and training details are given in Appendices A and B. We compare our method against HeteroGNN, the supervised RelBench baselines [11, 12].

## 3.1 Task Performance

Tables 1 and 2 report the performance on the entity classification and entity regression tasks, respectively. The best result per task is bold and underlined; the second best is bold. For classification, feature propagation improves all twelve tasks, raising average AUROC from 55.95 to 73.65 on

![](images/68235d3a215fe61a670cccf5a99f23c8fa4fbd59aef05d92ff3f660e2490a09e.jpg)

<table><tr><td>Database</td><td>Reduction</td></tr><tr><td>rel-amazon</td><td>65.9%</td></tr><tr><td>rel-avito</td><td>95.5%</td></tr><tr><td>rel-hm</td><td>64.8%</td></tr><tr><td>rel-stack</td><td>72.3%</td></tr><tr><td>rel-trial</td><td>58.7%</td></tr><tr><td>rel-ratebeer</td><td>61.8%</td></tr><tr><td>rel-arxiv</td><td>80.4%</td></tr></table>

Figure 2: Trainable parameters (left) and MLP parameter reduction relative to HeteroGNN (right).

RelBench-v1 and from 65.93 to 82.38 on RelBench-v2 (Table 1). As a result, the filtered MLP comes within 1.8 and 1.6 points of HeteroGNN, respectively, without learned relational aggregation. Further refinement with C&S adds about 0.51 and 0.02 points on average, with the largest gains on trial/study-outcome (+1.31) and hm/user-churn (+0.95). For regression, feature propagation lowers MAE on eight of nine tasks and matches or exceeds HeteroGNN on avito/ad-ctr and arxiv/author-publication (Table 2). The only exception to these feature-propagation gains is trial/site-success, where MAE rises from 0.426 to 0.437. By contrast, post-processing with C&S provides almost no regression improvement, which may reflect its original design for node classification [10]. Consequently, validation tuning allows the propagation effect to be very small, and the selected configurations can largely preserve the filtered MLP’s performance.

## 3.2 Parameter Reduction

In addition, our method significantly reduces the number of trainable parameters by training only the MLPs, compared with HeteroGNN (Figure 2), since feature propagation and C&S use only fixed operators. To examine whether these effects depend on different graph constructions, we perform label propagation on further variants in Appendix D.

## 4 Discussion

In conclusion, we evaluate an MLP on target attributes, enrich its inputs through fixed feature propagation, and refine its predictions with historical labels through C&S. Our results show that fixed feature propagation recovers substantial relational signal without learned aggregation, providing a simple baseline for assessing the gains from more complex RDL models. However, historical labels pose a different challenge: the schema connects related entities but does not specify which connections support label propagation. The variation across meta-graphs therefore motivates task-aware constructions that capture useful label dependencies while respecting temporal visibility alongside more efficient preprocessing to reduce the cost of filtering each seed-time snapshot (Appendix B.2).

## References

[1] Joshua Robinson, Rishabh Ranjan, Weihua Hu, Kexin Huang, Jiaqi Han, Alejandro Dobles, Matthias Fey, Jan Eric Lenssen, Yiwen Yuan, Zecheng Zhang, et al. Relational deep learning: Graph representation learning on relational databases. In NeurIPS 2024 third table representation learning workshop, 2024. 1

[2] Matthias Fey, Weihua Hu, Kexin Huang, Jan Eric Lenssen, Rishabh Ranjan, Joshua Robinson, Rex Ying, Jiaxuan You, and Jure Leskovec. Position: Relational deep learning-graph representation learning on relational databases. In Forty-first International Conference on Machine Learning, 2024. 1, 2

[3] Yanbo Wang, Xiyuan Wang, Quan Gan, Minjie Wang, Qibin Yang, David Wipf, and Muhan Zhang. Griffin: Towards a graph-centric relational database foundation model. arXiv preprint arXiv:2505.05568, 2025. 1

[4] Tianlang Chen, Charilaos Kanatsoulis, and Jure Leskovec. RelGNN: Composite message passing for relational deep learning. In Forty-second International Conference on Machine Learning, 2025. 1, 2, 14

[5] Matthias Fey, Vid Kocijan, Federico Lopez, J Lenssen, and Jure Leskovec. Kumorfm: A foundation model for in-context learning on relational data. Technical whitepaper, Kumo AI, 2025. 1

[6] Rishabh Ranjan, Valter Hudovernik, Mark Znidar, Charilaos Kanatsoulis, Roshan Upendra, Mahmoud Mohammadi, Joe Meyer, Tom Palczewski, Carlos Guestrin, and Jure Leskovec. Relational transformer: Toward zero-shot foundation models for relational data. In International Conference on Learning Representations, volume 2026, pages 69149–69174, 2026. 1, 2, 3

[7] Valter Hudovernik, Federico López, Vid Kocijan, Akihiro Nitta, Jan Eric Lenssen, Jure Leskovec, and Matthias Fey. Kumorfm-2: Scaling foundation models for relational learning. arXiv preprint arXiv:2604.12596, 2026. 1

[8] Vijay Prakash Dwivedi, Sri Jaladi, Yangyi Shen, Federico López, Charilaos Kanatsoulis, Rishi Puri, Matthias Fey, and Jure Leskovec. Relational graph transformer. In International Conference on Learning Representations, volume 2026, pages 154348–154369, 2026. 1, 2

[9] Felix Wu, Amauri Souza, Tianyi Zhang, Christopher Fifty, Tao Yu, and Kilian Weinberger. Simplifying graph convolutional networks. In International conference on machine learning, pages 6861–6871. Pmlr, 2019. 2

[10] Qian Huang, Horace He, Abhay Singh, Ser-Nam Lim, and Austin Benson. Combining label propagation and simple models out-performs graph neural networks. In International Conference on Learning Representations, 2021. 2, 4, 8, 9

[11] Joshua Robinson, Rishabh Ranjan, Weihua Hu, Kexin Huang, Jiaqi Han, Alejandro Dobles, Matthias Fey, Jan E Lenssen, Yiwen Yuan, Zecheng Zhang, et al. Relbench: A benchmark for deep learning on relational databases. Advances in Neural Information Processing Systems, 37: 21330–21341, 2024. 2, 3, 6, 8, 9

[12] Justin Gu, Rishabh Ranjan, Charilaos Kanatsoulis, Haiming Tang, Martin Jurkovic, Valter Hudovernik, Mark Znidar, Pranshu Chaturvedi, Parth Shroff, Fengyu Li, et al. Relbench v2: A large-scale benchmark and repository for relational data. arXiv preprint arXiv:2602.12606, 2026. 2, 3, 6, 8, 9

[13] Linjie Xu, Yanlin Zhang, Quan Gan, Minjie Wang, and David Wipf. No need to train your rdb foundation model. arXiv preprint arXiv:2602.13697, 2026. 3, 6, 14

[14] James A Hanley and Barbara J McNeil. A method of comparing the areas under receiver operating characteristic curves derived from the same cases. Radiology, 148(3):839–843, 1983. 3

[15] Mark EJ Newman. Scientific collaboration networks. ii. shortest paths, weighted networks, and centrality. Physical review E, 64(1):016132, 2001. 9

[16] Linjie Xu and David Wipf. Parameter-free encoders remain viable for RDB foundation models. arXiv preprint arXiv:2607.05476, 2026. 14

## A Benchmark Tasks and Constructed Graphs

This section describes how we build the homogeneous graph for feature filtering based on the database schema, and how the tasks in the datasets are defined with labels.

Databases. Each database is converted once into the collapsed graph and stacked feature matrix described in Section 2, independently of the task evaluated on that database. Table 4 summarizes the resulting representations, reporting the number of graph rows, the dimensionality of the feature matrix, the distribution of this dimensionality across text, numerical, and other column types, and the number of parameters in the resulting MLP. In particular, feature dimensions and model sizes vary substantially across databases. For example, rel-avito has 622 input dimensions, while rel-trial has 15,145, mainly from text, and an MLP roughly 24× larger. This feature dimensionality, rather than the number of labeled examples, is the primary factor driving precomputation costs for some databases (Appendix B.2).

Tasks. Within each database, each task provides labeled examples, which RelBench splits into train, validation, and test sets according to the chronological order [11]. Table 3 reports the sizes of these splits together with the number of distinct seed times, which determines the number of filtered snapshots computed for each task. Tasks defined on the same database share the underlying graph but require separate snapshots. Consequently, tasks with more seed times incur higher precomputation costs. For example, hm/user-churn, with 54 seed times, requires substantially more snapshot computation than the avito tasks, which have 5 seed times. We evaluate all entity-level classification and regression tasks except those from rel-event, for which temporal leakage has been reported, and rel-f1, which [13] identify as likely compromised. For schema diagrams and task semantics, we refer to the benchmark papers [11, 12].

Table 3: Task sizes by official split and number of distinct seed times.
<table><tr><td>Task</td><td>Train</td><td>Val</td><td>Test</td><td>Seed times</td></tr><tr><td>RelBench-v1</td><td></td><td></td><td></td><td></td></tr><tr><td>classification</td><td></td><td></td><td></td><td></td></tr><tr><td>amazon/user-churn</td><td>4,708,383</td><td>409,792</td><td>351,885</td><td>33</td></tr><tr><td>amazon/item-churn</td><td>2,536,014</td><td>177,689</td><td>166,842</td><td>33</td></tr><tr><td>avito/user-clicks</td><td>59,454</td><td>21,183</td><td>47,996</td><td>5</td></tr><tr><td>avito/user-visits</td><td>86,619</td><td>29,979</td><td>36,129</td><td>5</td></tr><tr><td>hm/user-churn</td><td>3,832,692</td><td>76,556</td><td>74,575</td><td>54</td></tr><tr><td>stack/user-badge</td><td>3,386,276</td><td>247,398</td><td>255,360</td><td>42</td></tr><tr><td>stack/user-engagement</td><td>1,360,850</td><td>85,838</td><td>88,137</td><td>48</td></tr><tr><td>trial/study-outcome</td><td>11,994</td><td>960</td><td>825</td><td>21</td></tr><tr><td>regression</td><td></td><td></td><td></td><td></td></tr><tr><td>amazon/user-ltv</td><td>4,708,383</td><td>409,792</td><td>351,885</td><td>33</td></tr><tr><td>amazon/item-ltv</td><td>2,707,679</td><td>166,978</td><td>178,334</td><td>33</td></tr><tr><td>avito/ad-ctr</td><td>5,100</td><td>1,766</td><td>1,816</td><td>5</td></tr><tr><td>hm/item-sales</td><td>5,488,184</td><td>105,542</td><td>105,542</td><td>54</td></tr><tr><td>stack/post-votes</td><td>2,453,921</td><td>156,216</td><td>160,903</td><td>48</td></tr><tr><td>trial/site-success</td><td>151,407</td><td>19,740</td><td>22,617</td><td>21</td></tr><tr><td>trial/study-adverse</td><td>43,335</td><td>3,596</td><td>3,098</td><td>21</td></tr><tr><td>RelBench-v2</td><td></td><td></td><td></td><td></td></tr><tr><td>classification ratebeer/user-churn</td><td></td><td></td><td></td><td></td></tr><tr><td>ratebeer/beer-churn</td><td>373,709</td><td>19,908</td><td>9,392</td><td>76</td></tr><tr><td></td><td>2,470,686</td><td>92,367</td><td>79,927</td><td>76</td></tr><tr><td>ratebeer/brewer-dormant</td><td>98,697</td><td>15,840</td><td>16,366</td><td>20</td></tr><tr><td>arxiv/paper-citation</td><td>534,233</td><td>155,845</td><td>193,696</td><td>10</td></tr><tr><td>regression</td><td></td><td></td><td></td><td></td></tr><tr><td>ratebeer/user-count</td><td>373,709</td><td>19,908</td><td>9,392</td><td>76</td></tr><tr><td>arxiv/author-publication</td><td>210,769</td><td>39,015</td><td>39,655</td><td>10</td></tr></table>

Table 4: Graph and feature statistics by database. Input dimensions include degree and age, used only for K > 0.
<table><tr><td>Database</td><td>Tables</td><td>Nodes</td><td>Input dim</td><td>Text cols</td><td>Num. cols</td><td>Other</td><td>MLP params</td></tr><tr><td>RelBench-v1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>rel-amazon</td><td>3</td><td>15,000,713</td><td>1,814</td><td>6</td><td>7</td><td>2</td><td>464,897</td></tr><tr><td>rel-avito</td><td>8</td><td>20,679,117</td><td>622</td><td>2</td><td>40</td><td>0</td><td>159,745</td></tr><tr><td>rel-hm</td><td>3</td><td>16,664,809</td><td>3,092</td><td>10</td><td>13</td><td>14</td><td>792,065</td></tr><tr><td>rel-stack</td><td>7</td><td>4,247,264</td><td>4,529</td><td>15</td><td>29</td><td>6</td><td>1,159,937</td></tr><tr><td>rel-trial</td><td>15</td><td>5,434,924</td><td>15,145</td><td>50</td><td>53</td><td>30</td><td>3,877,633</td></tr><tr><td>RelBench-v2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>rel-ratebeer</td><td>13</td><td>13,787,005</td><td>16,992</td><td>56</td><td>66</td><td>50</td><td>4,350,465</td></tr><tr><td>rel-arxiv</td><td>6</td><td>2,146,112</td><td>1,803</td><td>6</td><td>15</td><td>0</td><td>462,081</td></tr></table>

## B Feature Propagation and Training Details

In this section, we describe in detail how to encode and normalize attributes once per database, then append structural features for each visible snapshot.

Step 1: Column encoding. We encode each table $T _ { i }$ without learned parameters. Numerical columns use one scalar; text columns use 300-dimensional GloVe sentence embeddings, as in RelBench. Categorical columns use indicators for the six most frequent values and the log-frequency of the row’s value, counted through the validation timestamp. Other timestamp columns encode years relative to the row’s time, or since 1990 for tables without a time column. Concatenating the encodings of all non-key columns of $T _ { i }$ yields $X _ { i } \in \mathbb { R } ^ { n _ { i } \times d _ { i } }$ , where $n _ { i } = \left| T _ { i } \right|$ and $d _ { i }$ denotes the encoded dimensionality of table $T _ { i }$ . Table 4 reports the resulting dimensions aggregated per database. Primary and foreign keys are excluded from the feature representation and are used solely to define the graph structure.

Step 2: Stacking. For m tables, we place the feature matrices in disjoint row and column blocks,

$$
X = \operatorname { d i a g } ( X _ { 1 } , \ldots , X _ { m } ) \in \mathbb { R } ^ { n \times d } , \qquad n = \sum _ { i = 1 } ^ { m } n _ { i } , \qquad d = \sum _ { i = 1 } ^ { m } d _ { i } ,\tag{1}
$$

so each row contains its entity’s encoded attributes in the block for its source table and zeros elsewhere.

Step 3: Row normalization. SGC divides each feature vector by its signed row sum, which suits non-negative bag-of-words features but can become unstable for signed GloVe coordinates. For row $x _ { v }$ of $\bar { X }$ , we instead use

$$
\tilde { x } _ { v } = \left\{ \begin{array} { l l } { x _ { v } / \| x _ { v } \| _ { 1 } , } & { \| x _ { v } \| _ { 1 } > 0 , } \\ { 0 , } & { \| x _ { v } \| _ { 1 } = 0 , } \end{array} \right. \qquad \| x _ { v } \| _ { 1 } = \sum _ { c = 1 } ^ { d } | X _ { v , c } | .\tag{2}
$$

For nonzero, non-negative features, this agrees with signed-sum normalization; zero rows remain zero, and the empty column blocks do not affect the norm. Consequently, omitting this normalization and the symmetric adjacency reduced test AUROC by up to 41 percentage points.

Step 4: Structural columns. After normalization, we append two structural features,

$$
\bar { \boldsymbol { x } } _ { v } ( t ) = \left[ ~ \tilde { \boldsymbol { x } } _ { v } ~ \big | ~ \log \bigl ( 1 + \deg _ { A _ { t } } ( v ) \bigr ) ~ \big | ~ a _ { v } ( t ) ~ \right] \in \mathbb { R } ^ { d + 2 } ,\tag{3}
$$

where $\pmb { A } _ { t }$ contains only rows visible at t and $\deg _ { t } ( v )$ excludes self-loops. The age $a _ { v } ( t ) \ =$ min $\mathsf { \Omega } _ { 1 } ( t - t _ { v } , 5 0 )$ is measured in years and set to zero for rows without timestamps. We recompute both features for each snapshot and append them after row normalization to preserve their absolute scale. Stacking $\bar { x } _ { v } ( t )$ gives $\bar { X } _ { t }$ , which the filter propagates together with the encoded attributes.

## B.1 Filter Definition

For the entity graph $G = ( \nu , \mathcal { E } )$ of Section 2, let $\pmb { A } _ { t }$ be the symmetric adjacency induced by the visible rows:

$$
\mathcal { V } _ { t } = \{ v \in \mathcal { V } : v \mathrm { ~ h a s ~ n o ~ t i m e s t a m p ~ o r ~ } t _ { v } \leq t \} .\tag{4}
$$

Adding self-loops gives $\tilde { \pmb { A } } _ { t } = \pmb { A } _ { t } + \pmb { I }$ , with degree matrix $\tilde { D } _ { t }$ . The normalized filter and its output are defined as:

$$
\begin{array} { r } { { \cal S } _ { t } = \tilde { { \cal D } } _ { t } ^ { - 1 / 2 } \tilde { { \cal A } } _ { t } \tilde { { \cal D } } _ { t } ^ { - 1 / 2 } , } \\ { { \cal H } _ { t } = { \cal S } _ { t } ^ { \cal K } \bar { { \cal X } } _ { t } , \qquad } \end{array}\tag{5}
$$

where $\bar { X } _ { t }$ contains the normalized attributes and snapshot-specific structural features from Eq. (3). The MLP then takes $h _ { e , t } = ( \pmb { H } _ { t } ) _ { e }$ for prediction row $( e , t )$ . Setting $K = 0$ and both structural columns to zero gives the target-attribute baseline.

## B.2 Graph Filtering Cost

The filter is precomputed per task: the collapsed graph and the stacked features are shared by all tasks of a database, but one snapshot is computed per distinct seed time of the task and read out at the task’s entity rows, so cost grows with the seed-time count of Table 3. We measured the filtering time, ranging from 5.5 minutes (rel-avito) to 47 minutes (rel-stack) for one representative task per database; on rel-avito, deepening the filter to K = 10 grows this to 113 minutes. Dense design matrices dominate memory, requiring roughly (rows) × (d + 2) × 4 bytes per snapshot. Serial peak memory ranged from 44 GB (rel-avito) to 214 GB (rel-amazon). This process can be further sped up by utilizing GPUs for filtering, which remains as future work.

Parallel filtering and timing. To reduce the filtering time, we process independent snapshots and feature columns across 32 parallel jobs, producing the same features as serial filtering. Table 5 reports filtering wall time with all column types at K = 2, MLP training time, and HeteroGNN training time of one run [11, 12]. We note that the timings exclude column encoding cached for both methods and include HeteroGNN’s neighborhood sampling during training.

Table 5: Filtering and training time per task (minutes); timing protocol in Appendix B.2.
<table><tr><td>Task</td><td>Filtering</td><td>Training MLP</td><td>HeteroGNN</td></tr><tr><td>RelBench-v1 classification</td><td></td><td></td><td></td></tr><tr><td>amazon/user-churn</td><td>8.5</td><td>16.2</td><td>28.8</td></tr><tr><td>amazon/item-churn</td><td>9.2</td><td>10.2</td><td>15.3</td></tr><tr><td>avito/user-clicks</td><td>1.1</td><td>0.3</td><td>9.6</td></tr><tr><td>avito/user-visits</td><td>1.8</td><td>0.3</td><td>3.9</td></tr><tr><td>hm/user-churn</td><td>1.1</td><td>25.3</td><td>16.3</td></tr><tr><td>stack/user-badge</td><td>1.4</td><td>18.7</td><td>21.9</td></tr><tr><td>stack/user-engagement</td><td>1.4</td><td>14.2</td><td>30.3</td></tr><tr><td>trial/study-outcome</td><td>2.0</td><td>1.0</td><td>4.9</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>regression</td><td>11.5</td><td>23.0</td><td>16.4</td></tr><tr><td>amazon/user-ltv</td><td></td><td></td><td></td></tr><tr><td>amazon/item-ltv</td><td>13.6</td><td>18.8</td><td>13.6</td></tr><tr><td>avito/ad-ctr</td><td>3.1</td><td>0.05</td><td>1.9</td></tr><tr><td>hm/item-sales</td><td>1.2</td><td>35.3</td><td>16.4</td></tr><tr><td>stack/post-votes</td><td>1.6</td><td>26.7</td><td>25.4</td></tr><tr><td>trial/site-success</td><td>2.0</td><td>5.8</td><td>4.1</td></tr><tr><td>trial/study-adverse</td><td>1.8</td><td>3.4</td><td>1.8</td></tr><tr><td>RelBench-v2 classification</td><td></td><td></td><td></td></tr><tr><td>ratebeer/user-churn</td><td>11.2</td><td>13.4</td><td>15.7</td></tr><tr><td>ratebeer/beer-churn</td><td>31.2</td><td>89.9</td><td>46.2</td></tr><tr><td>ratebeer/brewer-dormant</td><td>25.7</td><td>5.6</td><td>11.1</td></tr><tr><td>arxiv/paper-citation</td><td>1.5</td><td>3.0</td><td>13.8</td></tr><tr><td>regression</td><td></td><td></td><td></td></tr><tr><td>ratebeer/user-count</td><td>4.5</td><td>12.6</td><td>16.3</td></tr><tr><td>arxiv/author-publication</td><td>0.9</td><td>1.2</td><td>3.4</td></tr></table>

## B.3 Training Details

The prediction head follows the reference MLP of [10], with batch normalization removed. It consists of a single hidden layer of width 256, followed by ReLU activation and dropout with rate 0.5. The same architecture is used across all datasets and is trained with Adam using a batch size of 512, following RelBench configurations [11]. We remove batch normalization because its running statistics are estimated from the training period and can degrade substantially under RelBench’s temporal test shift. On one task, for example, removing batch normalization left validation AUROC unchanged but prevented test AUROC from collapsing from 0.886 to 0.224. Then, we train with binary cross-entropy for classification and $\ell _ { 1 }$ loss for regression, clipping regression predictions to the 2nd–98th percentile range of training targets [11]. Evaluation uses the official RelBench AUROC and MAE implementations, with MAE reported in the target’s original units. We select the learning rate on validation from $\{ 1 0 ^ { - 3 } , 3 { \cdot } 1 0 ^ { - 3 } , 1 0 ^ { - 2 } \}$ and stop after $5$ epochs without improvement in the validation metric.

We use the official RelBench splits [11, 12] without modification. The benchmark fixes validation and test timestamps per database, generates labeled examples at a calendar stride, and masks test labels; we do not perform any additional re-splitting. All results are averaged over ten fixed seeds. To reproduce our results, readers can select the following seeds: {0, 7, 13, 21, 42, 77, 123, 555, 1234, 2026}, which control MLP initialization and batch shuffling. The filter is deterministic on the relational entity graph. The baselines additionally employ per-edge-type message functions and column encoders on top of comparable input encoders, whereas our model uses a single prediction head after feature propagation. Thus, our prediction head has fewer trainable parameters by construction.

Comparison to the RDL training pipeline. RelBench-style models such as RelGNN and RelGT sample a temporal neighborhood for each labeled row, restrict it to information visible at the seed time, and jointly train the relational model and prediction head. Our pipeline instead filters the full visible graph once per seed time, then trains an MLP on the resulting rows. This moves relational computation into preprocessing, allowing the filtered features to be computed once and reused.

Reproducibility. Our code is available at https://github.com/vietngth/revisit-baselines-rdl.

## C Label Propagation with Correct-and-Smooth

Correct-and-Smooth (C&S) [10] post-processes the predictions of a base model with two labelpropagation stages over a graph concerning the task entities. Let $Z$ denote the base predictions, Y the observed labels, and L the labeled nodes of the graph. For weighted label-graph adjacency A and degree matrix D, the two stages use normalizations $S _ { 1 }$ and $S _ { 2 }$ selected from $\{ \bar { D } ^ { - 1 / 2 } \bar { A } D ^ { - 1 / 2 } , D ^ { \bar { - } 1 } A , A D ^ { - 1 } \}$ . Correct initializes residuals as $E _ { L } = Y _ { L } - Z _ { L }$ , with zeros elsewhere, and computes the following:

$$
\begin{array} { c } { { E ^ { ( 0 ) } = E , } } \\ { { E ^ { ( k + 1 ) } = \left( 1 - \alpha _ { 1 } \right) E + \alpha _ { 1 } S _ { 1 } E ^ { ( k ) } , } } \\ { { Z ^ { ( r ) } = Z + s \odot E ^ { ( 5 0 ) } , } } \end{array}\tag{6}
$$

with three scaling options: none $( s = 1 )$ , FDiff-scale, a constant $s \in [ 1 , 5 0 ]$ with labeled residuals reset to $E _ { L }$ each iteration, or Autoscale, for classification only, with $s _ { i } = \sigma / | E _ { i } ^ { ( 5 0 ) } |$ , where $\sigma$ is the mean absolute labeled residual in the prediction row’s graph. Next, Smooth sets $G _ { L } = Y _ { L }$ and $G = Z ^ { ( r ) }$ elsewhere, then computes

$$
\begin{array} { c } { { G ^ { ( 0 ) } = G , } } \\ { { G ^ { ( k + 1 ) } = \left( 1 - \alpha _ { 2 } \right) G + \alpha _ { 2 } S _ { 2 } G ^ { ( k ) } . } } \end{array}\tag{7}
$$

Here, the algorithm is fixed with $k = 5 0$ iterations, clipping to the training-label range after each iteration. Finally, we read out the seed prediction; seeds without labeled neighbors keep their base predictions. Both stages assume positive correlation of errors and labels along edges. We tune $\bar { ( \alpha _ { 1 } , \alpha _ { 2 } ) } \in [ 0 , 1 ] ^ { 2 }$ , the normalizations $S _ { 1 }$ and $S _ { 2 }$ , the residual scaling, and stage selection on validation using the task metric. We use a tree-structured Parzen estimator with 60 trials for RelBench-v1 classification and 300 otherwise, and apply the selected configuration to all ten seeds. Setting both coefficients to zero recovers the base exactly, which the search may select.

## C.1 Graph Constructions

C&S requires a graph whose nodes correspond to the task’s prediction rows and whose edges carry label information, but the relational schema does not provide such a graph directly. We therefore construct one for each prediction row $( e , t )$ using two sources of label information observable at time t: the entity’s own historical labels and the historical labels of related entities. Specifically, we identify related entities by composing the schema’s foreign-key relations along walks that eventually return to the task’s entity table. Two entities are connected with a weight proportional to the number of records they share, where each shared record contributes inversely to the number of entities that it connects, following the co-occurrence weighting scheme used in collaboration networks [15]. For each seed time $t , \mathcal { N } ( e , t )$ contains up to 100 related entities with the largest weights from shared records observed before t. Table 6 specifies the label sources and history limits, Figure 3 shows their connections, and Algorithm 1 gives the shared construction steps.

Algorithm 1 Build a label graph for prediction row $( e , t )$   
Require: histories H; related entities $\mathcal { N } ( e , t )$ and weights $w _ { u } ;$ base predictions $Z ;$ variant from   
Table 6   
1: Initialize an unlabeled seed s with $Z _ { s } = Z _ { e , t } ;$ set $w _ { e } = 1$   
2: for all $\iota \in \{ e \} \cup \mathcal { N } ( e , t )$ do   
3: Read attachment type and cap c from Table 6 $( c = 0$ for “None”)   
4: $L _ { u } \gets \{ ( y , t _ { y } ) \in \dot { \mathcal { H } } ( u ) : t _ { y } < t \}$ , ordered newest first   
5: Skip u if its attachment type is “None” or $L _ { u }$ is empty   
6: $T _ { u } \gets$ the first min $( c , | L _ { u } ^ { \bar { \mathbf { \alpha } } } | )$ labels of $L _ { u }$   
7: if labels attach directly then   
8: Connect s to each $y _ { i } \in T _ { u }$ with weight $0 . 7 ^ { i - 1 }$ for SELF, otherwise $w _ { u }$   
9: else if labels form a chain then   
10: Add the path $s - y _ { 1 } - \cdot \cdot \cdot - y _ { | T _ { u } | } ;$ first edge weight $w _ { u } ,$ remaining weights 1   
11: else ▷ relay   
12: Create one relay per label in $L _ { u }$ if $u = e ;$ otherwise one for the newest label   
13: Initialize each relay with its associated past label’s base prediction   
14: Connect s to each relay with weight $w _ { u } ;$ attach a separate copy of $T _ { u }$ with unit edges   
15: end if   
16: end for   
17: Initialize history nodes with their associated base predictions and observed labels   
18: return the weighted undirected graph and seed s

Table 6: Meta-graph variants in Figure 3. A star attaches labels directly; a chain orders them newest first; a relay is an unlabeled intermediate node between the seed and historical labels.
<table><tr><td>Variant</td><td>Panel</td><td>Own history</td><td>Related history (per entity)</td></tr><tr><td>SELF</td><td>(a)</td><td>Star, all labels</td><td>None</td></tr><tr><td>CROSS</td><td>(b)</td><td>None</td><td>Most recent label, directly</td></tr><tr><td>CROSS+SELF</td><td>(c)</td><td>Star, all labels</td><td>Most recent label, directly</td></tr><tr><td>CROSS+HIST</td><td>(d)</td><td>None</td><td>Relay, up to 3 labels</td></tr><tr><td>ALL</td><td>(e)</td><td>One relay per past label; up to 3 label copies each</td><td>Relay, up to 3 labels</td></tr><tr><td>SELF-CHAIN</td><td>(f)</td><td>Chain, all labels</td><td>None</td></tr><tr><td>CHAIN+CROSS</td><td>(g)</td><td>Chain, all labels</td><td>Most recent label, directly</td></tr><tr><td>CHAIN+HIST-3</td><td>(h)</td><td>Chain, up to 3</td><td>Chain, up to 3 labels</td></tr><tr><td>CHAIN+HIST</td><td>(i)</td><td>Chain, all labels</td><td>Chain, all labels</td></tr></table>

All variants exclude labels observed at or after the seed time; boundary audits verify that adding such labels leaves the admitted history unchanged. With this common temporal rule, the variants differ in which histories they use and how they connect them to the prediction row. The following descriptions follow panels (a)–(i) in Figure 3 and complement Table 6.

(a) SELF. We connect the prediction row directly to all of the target entity’s past labels. Each label is one hop away, but its edge weight decreases with age rank. For example, the newest label receives weight 1, the next 0.7, and so on. This construction uses the entity’s own past outcomes as context, with greater edge weight on recent observations.

(b) CROSS. We instead use the most recent label of each selected related entity, connecting it directly to the prediction row with its co-occurrence weight. The target’s own labels are omitted. This construction asks whether related entities provide useful label information, including when the target has no label history of its own.

(c) CROSS+SELF. We attach both the target’s full label history and the newest label of each selected related entity directly to the prediction row. Both sources are therefore one hop away. Unlike

![](images/55b374934e81d2e3f6df843b1e31e93ee86b3f5397faeba268a3a90da301c03c.jpg)

prediction row

![](images/d03a8b90621e0227e0fcfcca620f34e9cce5d005074399c670e6e2a5ee2f7cdb.jpg)

own label

![](images/d365b92c57d363e2ba863bc3f312ed66fb3cf35232b394e673259b50996cae95.jpg)

related label

![](images/7acbb0af2ecc10670494c4576bdda2edbf18cbad5e4525ad0c49f0fdcccbcc13.jpg)

unlabeled relay

Direct and relay connections

Temporal chains

![](images/de1ba2ceebf029c499c3dfff84920aa9075a61b7c639fccfd5a38facf66f6b93.jpg)

![](images/bfa949b6a6b4cb4ce1a29b0b0b2c0f2cd017976806ec45777e57558eab11dc03.jpg)

(a) SELF

![](images/078bb9f53d71a3156aa0cd908ad6084ecafb1f00a26f07f0322a083162a0182f.jpg)

(f) SELF-CHAIN

![](images/8a9b0d9d274b2fad1c342b9ceb9690c2dba927072017a30b98eaafe574321dc3.jpg)

(b) CROSS

![](images/1fb264e03cc363efb91f52072d3403aa978e46691c0d4f02103f3e15f40cb82c.jpg)

(g) CHAIN+CROSS

![](images/a30573585eeb582212c591f2ec4ae039a0e06f94cebee31c8bf1e5aaceb7f7fa.jpg)

(c) CROSS+SELF

![](images/f9e5006cbb0a33df7dd5f1550591e307f314ff323e86f54d01d315f4208ef770.jpg)

(h) CHAIN+HIST-3

![](images/c9a232a8e2fdb687a671556f03ae2b97a636a8a72d6d878d9971ad389318554a.jpg)

(d) CROSS+HIST

![](images/da6497708e31e54215af99e40a75daa94afcaa0375e651de75bc4d408c3e03f9.jpg)

(i) CHAIN+HIST

(e) ALL

Figure 3: Meta-graphs for $( e , t )$ with undirected edges and labels observed before t. Own histories lie left of e, related histories right; indices increase from newest to oldest, with $y _ { j , k }$ denoting related entity $v _ { j } \mathrm { ' s }$ kth label. Dashed relays are unlabeled intermediate nodes that carry base predictions and connect historical labels to the seed in two hops. In (e), $r _ { 1 } , \ldots , r _ { m }$ represent all m own past labels, each with separate copies of up to three newest own labels. Table 6 gives the history limits.

SELF, own-label edges here have weight 1 throughout, while related-label edges retain their co-occurrence weights. This is the construction used in the main results.

(d) CROSS+HIST. For each selected related entity, we insert an unlabeled relay between the prediction row and up to three of that entity’s newest labels. The relay includes the base prediction associated with its newest label, while its attached history nodes carry the observed labels. This gives each related entity a local history around an intermediate prediction node; label information reaches the target through two hops. The target’s own history is omitted.

(e) ALL. We extend CROSS+HIST with the target’s own history, also through relays. Each own past label creates a separate unlabeled relay carrying the base prediction associated with that label. Every own relay receives fresh copies of the same up to three newest own labels, and all edges in these own-history branches have weight 1. Thus, the number of own relays follows the full history length, while each relay’s labeled neighborhood remains limited to three observations.

(f) SELF-CHAIN. We arrange the target’s full history along a path: the prediction row connects to the newest label, that label to the next newest, and so on. All edges have weight 1, and related entities are omitted. Older observations lie progressively more hops from the prediction row, so the topology encodes recency instead of assigning a different edge weight to each past label.

(g) CHAIN+CROSS. We retain the own-history path of SELF-CHAIN and add a direct edge to the newest label of each selected related entity, weighted by co-occurrence. Own and related histories therefore enter through different routes: older own labels lie along the temporal path, while each related entity contributes one label at one hop.

(h) CHAIN+HIST-3. We build a separate temporal path for the target and for each selected related entity, keeping up to three newest labels in every path, including the target’s. The first edge of a related path has its co-occurrence weight; all remaining edges, including the own-history path, have weight 1. This construction represents recent histories from both sources through the same temporal ordering, without unlabeled relays.

(i) CHAIN+HIST. We remove the three-label limit from CHAIN+HIST-3 and retain all visible labels in both the own and related histories. Each history still forms a newest-first path with the same edge-weight rules, placing older observations farther from the prediction row. Comparing these two variants therefore examines the effect of retaining longer histories under the same attachment rules.

## D Experimental Results

To assess the effect of graph construction, Table 7 reports all variants evaluated on the same filtered base, and the column names follow the constructions defined in Table 6. There are three emerging patterns in the experiments. First, improvements are generally small and construction-dependent. Thus, no single construction consistently outperforms the others, and some constructions are detrimental. For example, the self-label star reduces AUROC by 2.7 points on avito/user-clicks. Second, the exceptions are informative based on a specific database. For instance, on trial/study-outcome, incorporating related-entity labels improves AUROC by up to +1.67 points, suggesting that this task contains a stronger cross-entity label signal. In contrast, stack/user-engagement is best served by the pure temporal chain, whose one-hop structure effectively imposes a recency constraint. Third, the same constructions produce substantially larger gains on the unfiltered base, ranging from +2.8 to +4.1 points on the tasks above. Nevertheless, these models still underperform the filtered base without label propagation on almost all tasks, suggesting that filtered features already capture much of the label signal. Table 9 summarizes regression results.

To summarize the gains across constructions, Table 8 compares the best variant per task with the two MLP stages. The best construction varies across tasks, with no single construction being best on more than two tasks. The resulting mean improvement therefore reflects per-task selection of the best variant and should be interpreted as an oracle-style upper bound. Any single fixed variant achieves a smaller improvement. For regression, selecting meta-graphs on validation lowers MAE on six of seven tasks (Table 9), with five tasks selecting SELF or SELF-CHAIN. In particular, these own-history graphs give the largest gains, from 0.060 to 0.044 on hm/item-sales and 50.99 to 46.93 on amazon/item-ltv, while the two trial tasks select related-entity histories with slight improvements.

Table 7: Label-graph variants with Optuna tuning on the trained filtered MLP over 10 seeds.
<table><tr><td></td><td>SELF</td><td>CROSS</td><td>CROSS+SELF</td><td>CROSS+HIST</td><td>ALL</td><td>SELF-CHAIN</td><td>CHAIN+CROSS</td><td>CHAIN+HIST-3</td><td>CHAIN+HIST</td></tr><tr><td colspan="10">RelBench-v1</td></tr><tr><td>classification (AUROC×100 ↑) amazon/user-churn</td><td>68.42±0.05</td><td>68.47±0.04</td><td>68.69±0.04</td><td>68.44±0.04</td><td>68.61±0.04</td><td> $6 8 . 4 7 { \scriptstyle \pm 0 . 0 5 }$ </td><td>68.59±0.04</td><td>68.77±0.05</td><td>68.76±0.05</td></tr><tr><td>amazon/item-churn</td><td>80.93±0.06</td><td>81.02±0.06</td><td>81.03±0.07</td><td>81.07±0.06</td><td>81.09±0.06</td><td>80.76±0.06</td><td>81.07±0.06</td><td>81.18±0.05</td><td>81.18±0.05</td></tr><tr><td>avito/user-clicks</td><td> $6 2 . 3 4 { \pm } 0 . 8 5 $ </td><td> $6 5 . 0 8 { \scriptstyle \pm 1 . 0 6 }$ </td><td>65.10±1.06</td><td> $\underline { { 6 5 . 4 4 \pm 1 . 0 4 } }$ </td><td> $6 4 . 8 6 { \pm } 1 . 0 2 $ </td><td> $6 1 . 9 1 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $6 4 . 6 2 { \scriptstyle \pm 1 . 0 9 }$ </td><td> $\overline { { 6 5 . 0 9 2 1 . 0 6 } }$ </td><td>65.09±1.06</td></tr><tr><td>avito/user-visits</td><td> $6 6 . 4 3 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $6 6 . 4 0 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $6 6 . 4 1 { \scriptstyle \pm 0 . 1 8 }$ </td><td>66.36±0.18</td><td> $\mathbf { 6 6 . 7 2 \pm 0 . 1 1 }$ </td><td></td><td> $\mathbf { 6 6 . 4 5 { \scriptstyle \pm 0 . 1 4 } }$ </td><td>66.37±0.18</td><td>66.37±0.18</td></tr><tr><td>hm/user-churn</td><td> $\mathbf { 6 8 . 9 4 } \pm \mathbf { 0 . 0 6 }$ </td><td> $6 8 . 2 3 { \scriptstyle \pm 0 . 0 7 }$ </td><td></td><td> $6 8 . 3 7 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $\overline { { 6 8 . 5 4 \pm 0 . 0 6 } }$ </td><td> $_ { \mathrm { ~ 6 6 . 4 0 2 0 . 1 7 } } ^ { \mathrm { 6 6 . 4 0 2 0 . 1 7 } }$   $6 8 . 4 6 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $6 8 . 3 5 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $6 8 . 3 5 { \scriptstyle \pm 0 . 0 7 }$ </td><td>68.35±0.07</td></tr><tr><td>stack/user-badge</td><td> $\mathbf { 8 6 . 1 4 } \pm \mathbf { 0 . 1 0 }$ </td><td> $8 6 . 1 0 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $\mathbf { 6 9 . 1 6 { \scriptstyle \pm 0 . 0 5 } }$  86.16±0.09</td><td> $8 6 . 0 9 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $8 6 . 1 0 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $8 6 . 0 7 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $8 6 . 1 0 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $8 6 . 1 1 { \scriptstyle \pm 0 . 0 9 }$ </td><td>86.11±0.09</td></tr><tr><td>stack/user-engagement</td><td> $\mathbf { 8 9 . 8 2 } \pm \mathbf { 0 . 1 0 }$ </td><td>88.60±0.11</td><td>89.02±0.09</td><td> $8 8 . 6 5 { \scriptstyle \pm 0 . 0 8 }$ </td><td>89.04±0.17</td><td> $\underline { { \mathbf { 9 0 . 0 0 } \pm 0 . 0 3 } }$ </td><td> $8 8 . 6 2 { \scriptstyle \pm 0 . 1 1 }$ </td><td>88.64±0.10</td><td>88.64±0.10</td></tr><tr><td>trial/study-outcome</td><td> $6 6 . 3 7 { \scriptstyle \pm 0 . 3 4 }$ </td><td>67.49±0.38</td><td>67.68±0.39</td><td>68.03±0.42</td><td>68.03±0.42</td><td> $\overline { { 6 6 . 3 7 \pm 0 . 3 4 } }$ </td><td>67.68±0.39</td><td>67.49±0.38</td><td>67.68±0.39</td></tr><tr><td colspan="10">regression (MAE ↓)</td></tr><tr><td>amazon/user-ltv</td><td></td><td></td><td></td><td></td><td></td><td></td><td> $1 4 . 6 5 6 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td>14.656±0.013</td><td>14.656±0.013</td></tr><tr><td>amazon/item-ltv</td><td>14.594±0.009</td><td>14.656±0.013</td><td>14.656±0.013</td><td>14.656±0.013 50.994±0.168</td><td>14.625±0.011 50.994±0.168</td><td>14.587±0.009 46.932±0.111</td><td>50.995±0.168</td><td>50.995±0.168</td><td>50.997±0.167</td></tr><tr><td>avito/ad-ctr</td><td>48.029±0.177</td><td>50.994±0.168</td><td>50.994±0.168</td><td></td><td></td><td> $\underline { { \mathbf { 0 . 0 3 8 } \pm 0 . 0 0 1 } }$ </td><td>0.040±0.003</td><td>0.040±0.003</td><td>0.040±0.003</td></tr><tr><td>hm/item-sales</td><td>0.039±0.003</td><td>0.040±0.003</td><td>0.040±0.003</td><td>0.040±0.003</td><td>0.040±0.003</td><td></td><td></td><td></td><td></td></tr><tr><td>stack/post-votes</td><td>0.049±0.000 0.066±0.000</td><td>0.060±0.000</td><td>0.060±0.000</td><td>0.060±0.000</td><td>0.060±0.000</td><td> $\overline { { 0 . 0 4 4 \pm 0 . 0 0 0 } }$ </td><td> $0 . 0 5 7 { \scriptstyle \pm 0 . 0 0 0 }$  0.066±0.000</td><td> $0 . 0 6 0 { \scriptstyle \pm 0 . 0 0 0 }$  0.066±0.000</td><td>0.060±0.000 0.066±0.000</td></tr><tr><td>trial/site-success</td><td> $\overline { { 0 . 4 3 4 { \pm } 0 . 0 0 5 } }$ </td><td>0.066±0.000</td><td>0.066±0.000</td><td>0.066±0.000</td><td> $\underline { { \mathbf { 0 . 0 6 6 } } } { \pm } 0 . 0 0 0$ </td><td> $\overline { { { \bf 0 . 0 6 6 } _ { \pm 0 . 0 0 0 } } }$ </td><td> $\overline { { 0 . 4 5 4 \pm 0 . 0 0 4 } }$ </td><td> $\underline { { \mathbf { 0 . 4 2 5 \pm 0 . 0 0 6 } } }$ </td><td></td></tr><tr><td>trial/study-adverse</td><td>44.867±0.151</td><td> $\overline { { 0 . 4 4 5 { \scriptstyle \pm 0 . 0 0 6 } } }$  44.822±0.150</td><td>0.443±0.006 44.817±0.150</td><td>0.437±0.005 44.616±0.153</td><td> $\overline { { 0 . 4 3 7 { \scriptstyle \pm 0 . 0 0 5 } } }$  44.616±0.153</td><td> $\overline { { { \bf 0 . 4 3 2 } _ { \pm 0 . 0 0 4 } } }$   $4 4 . 8 6 7 { \scriptstyle \pm 0 . 1 5 1 }$ </td><td> $4 4 . 8 2 2 { \scriptstyle \pm 0 . 1 5 0 }$ </td><td>44.822±0.150</td><td>0.425±0.006 44.822±0.150</td></tr><tr><td colspan="10">RelBench-v2</td></tr><tr><td colspan="10">classification (AUROC× 100 ↑)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ratebeer/user-churn</td><td>89.83±0.10 78.97±0.06</td><td>89.73±0.11</td><td>89.73±0.11</td><td> $8 9 . 7 2 { \scriptstyle \pm 0 . 1 2 }$  78.98±0.06</td><td>89.73±0.13</td><td> $\mathbf { 8 9 . 8 1 } 2 0 . 1 1$ </td><td> $8 9 . 7 3 { \scriptstyle \pm 0 . 1 1 }$  78.90±0.07</td><td>89.74±0.11</td><td>89.74±0.11 78.88±0.08</td></tr><tr><td>ratebeer/beer-churn ratebeer/brewer-dormant</td><td>79.82±0.04</td><td>78.91±0.07</td><td>78.90±0.08 79.78±0.04</td><td> $7 9 . 7 1 { \scriptstyle \pm 0 . 0 4 }$ </td><td>78.98±0.06 79.83±0.04</td><td> $\underline { { 7 9 . 0 0 \pm 0 . 0 5 } }$   ${ \bf 7 9 . 8 2 \pm 0 . 0 4 }$ </td><td> $7 9 . 8 1 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $7 8 . 9 0 { \scriptstyle \pm 0 . 0 8 }$   $7 9 . 8 1 { \scriptstyle \pm 0 . 0 4 }$ </td><td>79.81±0.04</td></tr><tr><td>arxiv/paper-citation</td><td>81.68±0.04</td><td> $7 9 . 7 3 { \scriptstyle \pm 0 . 0 4 }$   $8 1 . 1 3 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $8 1 . 1 7 { \scriptstyle \pm 0 . 0 5 }$ </td><td>81.20±0.05</td><td>81.18±0.05</td><td> $\mathbf { 8 1 . 3 6 } \pm \mathbf { 0 . 0 6 }$ </td><td> $8 1 . 1 5 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $8 1 . 1 5 { \scriptstyle \pm 0 . 0 5 }$ </td><td>81.15±0.05</td></tr><tr><td colspan="10">regression (MAE ↓)</td></tr><tr><td>ratebeer/user-count</td><td>9.900±0.058</td><td>9.115±0.075</td><td>9.115±0.075</td><td>9.244±0.093</td><td>9.294±0.098</td><td> $9 . 5 8 2 { \scriptstyle \pm 0 . 0 3 5 }$ </td><td>9.134±0.078</td><td> $9 . 1 4 5 { \scriptstyle \pm 0 . 0 8 0 }$ </td><td>9.145±0.080</td></tr><tr><td>arxiv/author-publication</td><td>0.489±0.001</td><td>0.489±0.002</td><td>0.489±0.002</td><td>0.489±0.001</td><td>0.489±0.002</td><td>0.489±0.002</td><td>0.489±0.002</td><td> $\underline { { \mathbf { 0 . 4 8 9 } \pm 0 . 0 0 2 } }$ </td><td>0.489±0.002</td></tr></table>

On stack/post-votes and arxiv/author-publication, all nine variants remain within 0.14% of the filtered MLP, as validation favors configurations that largely preserve its predictions. Historical labels are available for 90–98% and 61–71% of validation rows, respectively, and the target’s own past labels correlate with its current value (Pearson 0.61 and 0.59). However, this historical signal does not translate into lower MAE. The targets are concentrated at low values, with 95.2% of post-votes targets equal to zero and a median of one on author-publication. Since MAE favors conditional medians, mean-based corrections can increase error even when past labels correlate with the target. For example, adding the mean residual from each row’s own label history raises validation MAE from 0.060 to 0.100 on post-votes and from 0.428 to 0.494 on author-publication.

Table 8: Best meta-graph per task (test AUROC×100); reporting as in Table 1.
<table><tr><td>Task</td><td>Best variant</td><td>MLP (target attributes)</td><td>MLP (+relational attributes)</td><td>C&amp;S (+historical labels)</td></tr><tr><td>amazon/user-churn</td><td>CHAIN+HIST-3</td><td>53.65±0.18</td><td>68.12±0.05</td><td>68.77±0.05</td></tr><tr><td>amazon/item-churn</td><td>CHAIN+HIST-3</td><td>66.60±0.44</td><td>80.58±0.08</td><td>81.18±0.05</td></tr><tr><td>avito/user-clicks</td><td>CROSS+HIST</td><td>51.74±0.51</td><td>65.08±1.06</td><td>65.44±1.04</td></tr><tr><td>avito/user-visits</td><td>ALL</td><td>49.64±0.01</td><td>66.29±0.20</td><td>66.72±0.11</td></tr><tr><td>hm/user-churn</td><td>CROSS+SELF</td><td>55.25±0.05</td><td>68.21±0.06</td><td>69.16±0.05</td></tr><tr><td>stack/user-badge</td><td>CROSS+SELF</td><td>51.93±1.87</td><td>86.04±0.08</td><td>86.16±0.09</td></tr><tr><td>stack/user-engagement</td><td>SELF-CHAIN</td><td>53.04±0.04</td><td>88.53±0.12</td><td>90.00±0.03</td></tr><tr><td>trial/study-outcome</td><td>CROSS+HIST</td><td>65.77±0.36</td><td>66.37±0.34</td><td>68.03±0.42</td></tr><tr><td>Average</td><td></td><td>55.95</td><td>73.65</td><td>74.43</td></tr></table>

Table 9: Regression with validation-selected meta-graphs.

<table><tr><td>Task</td><td>Selected construction</td><td>MLP (target attributes)</td><td>MLP (+relational attributes)</td><td>C&amp;S (+historical labels)</td></tr><tr><td>amazon/user-ltv</td><td>SELF-CHAIN</td><td>16.757±0.007</td><td>14.656±0.013</td><td>14.587±0.009</td></tr><tr><td>amazon/item-ltv</td><td>SELF-CHAIN</td><td>59.958±0.150</td><td>50.994±0.168</td><td>46.932±0.111</td></tr><tr><td>avito/ad-ctr</td><td>SELF-CHAIN</td><td>0.043±0.000</td><td>0.040±0.003</td><td>0.038±0.001</td></tr><tr><td>hm/item-sales</td><td>SELF-CHAIN</td><td>0.076±0.000</td><td>0.060±0.000</td><td>0.044±0.000</td></tr><tr><td>stack/post-votes</td><td>SELF</td><td>0.068±0.000</td><td>0.066±0.000</td><td>0.066±0.000</td></tr><tr><td>trial/site-success</td><td>CHAIN+HIST-3</td><td>0.426±0.002</td><td>0.437±0.006</td><td>0.425±0.006</td></tr><tr><td>trial/study-adverse</td><td>CROSS+HIST</td><td>48.203±0.676</td><td>44.867±0.151</td><td>44.616±0.153</td></tr></table>

## E Extended Results

In this section, we compare against additional RDL models for completeness, and examine structural features, filter depth, runtime, and historical labels as MLP inputs. Table 12 uses an earlier filtered base with only numerical and text columns. All other tables in this appendix use all column types and the same runs as the main tables.

## E.1 Comparisons with More Models

Table 10 compares our pipeline with RelGNN [4] and RDBLearn [13]. RelGNN results are obtained from [4], and RDBLearn results come from [13] for RelBench-v1 and [16] for RelBench-v2. Our

Table 10: Extended results with more advanced models.
<table><tr><td>Task</td><td>MLP (target attributes)</td><td>MLP (+relational attributes)</td><td>C&amp;S (+historical labels)</td><td>HeteroGNN</td><td></td><td>RelGNN RDBLearn</td></tr><tr><td>RelBench-v1</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>classification (AUROC× 100 ↑)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>amazon/user-churn</td><td>53.65±0.18</td><td> $6 8 . 1 2 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $6 8 . 6 9 { \scriptstyle \pm 0 . 0 4 }$ </td><td> ${ \bf 7 0 . 4 2 { \scriptstyle \pm 0 . 0 5 } }$ </td><td>70.99</td><td>67.57</td></tr><tr><td>amazon/item-churn</td><td>66.60±0.44</td><td> $8 0 . 5 8 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $8 1 . 0 3 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $\underline { { { \mathbf { 8 2 . 8 1 } \pm 0 . 0 3 } } }$ </td><td>82.64</td><td>82.07</td></tr><tr><td>avito/user-clicks</td><td>51.74±0.51</td><td> $6 5 . 0 8 { \scriptstyle \pm 1 . 0 6 }$ </td><td> $6 5 . 1 0 { \scriptstyle \pm 1 . 0 6 }$ </td><td> $\overline { { 6 5 . 9 0 { \pm 1 . 9 } 5 } }$ </td><td>68.23</td><td>69.04</td></tr><tr><td>avito/user-visits</td><td>49.64±0.01</td><td> $\mathbf { 6 6 . 2 9 } \pm \mathbf { 0 . 2 0 }$ </td><td> $\underline { { { \bf 6 6 . 4 1 } } } \pm \bf { 0 . 1 8 }$ </td><td> $6 6 . 2 0 { \scriptstyle \pm 0 . 1 0 }$ </td><td>66.18</td><td>65.49</td></tr><tr><td>hm/user-churn</td><td> $5 5 . 2 5 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $6 8 . 2 1 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $\overline { { 6 9 . 1 6 { \pm 0 . 0 5 } } }$ </td><td> $\mathbf { 6 9 . 8 8 \pm 0 . 2 1 }$ </td><td>70.93</td><td>68.05</td></tr><tr><td>stack/user-badge</td><td> $5 1 . 9 3 { \scriptstyle \pm 1 . 8 7 }$ </td><td> $8 6 . 0 4 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $8 6 . 1 6 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $\mathbf { 8 8 . 8 6 { \scriptstyle \pm 0 . 0 8 } }$ </td><td>88.98</td><td>85.26</td></tr><tr><td>stack/user-engagement</td><td> $5 3 . 0 4 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $8 8 . 5 3 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $8 9 . 0 2 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $\mathbf { 9 0 . 5 9 } { \scriptstyle \pm 0 . 0 9 }$ </td><td>90.75</td><td>89.39</td></tr><tr><td>trial/study-outcome</td><td>65.77±0.36</td><td> $6 6 . 3 7 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $6 7 . 6 8 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $6 8 . 6 0 { \scriptstyle \pm 1 . 0 1 }$ </td><td>71.24</td><td>71.58</td></tr><tr><td>Average</td><td>55.95</td><td>73.65</td><td>74.16</td><td>75.41</td><td>76.24</td><td>74.81</td></tr><tr><td>regression (MAE ↓)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>amazon/user-ltv</td><td>16.757±0.007</td><td>14.656±0.013</td><td>14.656±0.013</td><td>14.313±0.013</td><td>14.230</td><td>14.540</td></tr><tr><td>amazon/item-ltv</td><td>59.958±0.150</td><td>50.994±0.168</td><td>50.994±0.168</td><td>50.053±0.163</td><td>48.767</td><td>48.559</td></tr><tr><td>avito/ad-ctr</td><td>0.043±0.000</td><td> $0 . 0 4 0 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 0 4 0 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td>0.041±0.001</td><td>0.037</td><td>0.034</td></tr><tr><td>hm/item-sales</td><td> $0 . 0 7 6 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 0 6 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 0 6 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.056±0.000</td><td>0.054</td><td>0.064</td></tr><tr><td>stack/post-votes</td><td> $0 . 0 6 8 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 0 6 6 } { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 0 6 6 } { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.065±0.000</td><td>0.065</td><td>0.068</td></tr><tr><td>trial/site-success</td><td> $0 . 4 2 6 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 4 3 7 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 4 4 3 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td>0.400±0.020</td><td>0.301</td><td>0.424</td></tr><tr><td>trial/study-adverse</td><td> $4 8 . 2 0 3 { \scriptstyle \pm 0 . 6 7 6 }$ </td><td> $4 4 . 8 6 7 { \scriptstyle \pm 0 . 1 5 1 }$ </td><td> $4 4 . 8 1 7 { \scriptstyle \pm 0 . 1 5 0 }$ </td><td> $4 4 . 4 7 3 { \scriptstyle \pm 0 . 2 0 9 }$ </td><td>44.461</td><td>43.913</td></tr><tr><td>RelBench-v2</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>classification (AUROC×100 ↑) ratebeer/user-churn</td><td>63.98±0.00</td><td></td><td> $8 9 . 7 3 { \scriptstyle \pm 0 . 1 1 }$ </td><td></td><td></td><td></td></tr><tr><td>ratebeer/beer-churn</td><td>62.40±0.84</td><td> $8 9 . 7 3 { \scriptstyle \pm 0 . 1 2 }$  78.93±0.07</td><td> $7 8 . 9 0 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $\mathbf { 9 4 . 2 7 { \scriptstyle \pm 0 . 2 2 } }$  78.67±0.60</td><td></td><td>97.84 83.56</td></tr><tr><td>ratebeer/brewer-dormant</td><td>63.67±0.32</td><td>79.73±0.04</td><td> $7 9 . 7 8 { \scriptstyle \pm 0 . 0 4 }$ </td><td>80.51±0.17</td><td></td><td>81.18</td></tr><tr><td>arxiv/paper-citation</td><td> $7 3 . 6 5 { \scriptstyle \pm 0 . 2 2 }$ </td><td>81.11±0.05</td><td> $8 1 . 1 7 { \scriptstyle \pm 0 . 0 5 }$ </td><td>82.50±0.04</td><td></td><td>82.37</td></tr><tr><td></td><td>65.93</td><td></td><td>82.40</td><td></td><td></td><td></td></tr><tr><td>Average</td><td></td><td>82.38</td><td></td><td>83.99</td><td></td><td>86.24</td></tr><tr><td>regression (MAE ↓) ratebeer/user-count</td><td> $1 4 . 4 5 3 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td></td><td> $9 . 1 1 5 { \scriptstyle \pm 0 . 0 7 5 }$ </td><td> $\mathbf { 7 . 3 7 4 { \scriptstyle \pm 0 . 1 0 2 } }$ </td><td></td><td>6.870</td></tr><tr><td></td><td></td><td> $9 . 1 1 5 { \scriptstyle \pm 0 . 0 7 5 }$ </td><td></td><td></td><td></td><td></td></tr><tr><td>arxiv/author-publication</td><td> $0 . 5 7 7 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 4 8 9 { \scriptstyle \pm 0 . 0 0 2 } }$ </td><td> $\mathbf { 0 . 4 8 9 { \scriptstyle \pm 0 . 0 0 2 } }$ </td><td> $0 . 5 1 3 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td></td><td>0.459</td></tr></table>

goal is to measure how much predictive signal a simple baseline captures without learned relational aggregation. While these models outperform our pipeline on several tasks, we believe the comparison helps contextualize the performance of our simple baseline and its role in assessing the need for architectural complexity in RDL. Both RelGNN and RDBLearn report only average performance across multiple random seeds. RelGNN [4] predates RelBench-v2 and reports no results on it.

## E.2 Results without Structural Columns

To isolate the contribution of structural information, we compare filtered representations with and without the degree and age features of Eq. (3). Both features remain disabled in the $K = 0$ baseline. The two rightmost columns of Table 11 repeat the main results. These results show that our core methodology remains unchanged: relational attributes provide substantial improvements over the $K = 0$ baselines, and C&S can still improve the performance by exploiting label propagation.

## E.3 Filtering Hyperparameters and Runtime

In this section, we provide further ablation studies on the choice of filtering K and the runtime of our experiments.

## E.3.1 Filtering Choices

We use $K \ = \ 2$ to match the reference HeteroGNN’s two-hop aggregation, then test $K \in$ $\{ 2 , 4 , 6 , 8 , 1 0 \}$ on rel-avito. Figure 4 shows that $K = 2$ performs best on most tasks, while deeper propagation increases filtering cost without consistent gains. Similarly, on the ten classification tasks from other databases with results at all tested depths (Figure 4), $K = 2$ is best or tied on five. Deeper filters gain at most 0.21 AUROC points (amazon/user-churn) but lose up to 7.1 (trial/study-outcome). Meanwhile, from $K = 2$ to $K = 8 ,$ , filtering takes about 4× longer on rel-arxiv and 630× on rel-trial.

![](images/326a0e0f2d4740602c282f5bb650daba19f03424b1e77fb526627c9054ac6df3.jpg)  
Figure 4: Effect of the filter depth per task.

Table 11: Ablation study on structural columns (degree and age).
<table><tr><td rowspan="2">Task</td><td rowspan="2">MLP</td><td colspan="2">Without structural columns</td><td colspan="2">With structural columns</td></tr><tr><td>MLP (target attributes) (+relational attributes)</td><td>C&amp;S (+historical labels)</td><td>MLP (+relational attributes)</td><td>C&amp;S (+historical labels)</td></tr><tr><td colspan="6">RelBench-v1</td></tr><tr><td>classification (AUROC×100 ↑)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>amazon/user-churn</td><td>53.65±0.18</td><td>65.82±0.10</td><td>67.35±0.04</td><td>68.12±0.05 80.58±0.08</td><td>68.69±0.04</td></tr><tr><td>amazon/item-churn</td><td>66.60±0.44</td><td>75.91±0.07</td><td>78.07±0.03</td><td></td><td>81.03±0.07</td></tr><tr><td>avito/user-clicks</td><td>51.74±0.51</td><td>66.26±0.50</td><td>66.26±0.50</td><td>65.08±1.06</td><td>65.10±1.06</td></tr><tr><td>avito/user-visits</td><td>49.64±0.01</td><td>66.30±0.07</td><td>66.52±0.06</td><td>66.29±0.20</td><td>66.41±0.18</td></tr><tr><td>hm/user-churn</td><td>55.25±0.05</td><td>68.05±0.02</td><td>69.08±0.06</td><td>68.21±0.06</td><td>69.16±0.05</td></tr><tr><td>stack/user-badge</td><td>51.93±1.87</td><td>84.03±0.05</td><td>84.94±0.09</td><td>86.04±0.08</td><td>86.16±0.09</td></tr><tr><td>stack/user-engagement trial/study-outcome</td><td>53.04±0.04 65.77±0.36</td><td>86.56±0.12 68.25±0.28</td><td>87.97±0.10 68.96±0.24</td><td>88.53±0.12 66.37±0.34</td><td>89.02±0.09 67.68±0.39</td></tr><tr><td></td><td>55.95</td><td>72.65</td><td>73.64</td><td>73.65</td><td>74.16</td></tr><tr><td colspan="6">Average</td></tr><tr><td>regression (MAE ↓) amazon/user-ltv</td><td>16.757±0.007</td><td>15.372±0.043</td><td>15.372±0.043</td><td>14.656±0.013</td><td></td></tr><tr><td>amazon/item-ltv</td><td>59.958±0.150</td><td>55.827±0.314</td><td>55.827±0.314</td><td>50.994±0.168</td><td>14.656±0.013 50.994±0.168</td></tr><tr><td>avito/ad-ctr</td><td>0.043±0.000</td><td>0.040±0.001</td><td>0.039±0.000</td><td>0.040±0.003</td><td>0.040±0.003</td></tr><tr><td>hm/item-sales</td><td>0.076±0.000</td><td>0.076±0.000</td><td>0.076±0.000</td><td>0.060±0.000</td><td>0.060±0.000</td></tr><tr><td>stack/post-votes</td><td>0.068±0.000</td><td>0.066±0.000</td><td>0.066±0.000</td><td>0.066±0.000</td><td>0.066±0.000</td></tr><tr><td>trial/site-success</td><td>0.426±0.002</td><td>0.419±0.001</td><td>0.427±0.001</td><td>0.437±0.006</td><td>0.443±0.006</td></tr><tr><td>trial/study-adverse</td><td> $4 8 . 2 0 3 { \scriptstyle \pm 0 . 6 7 6 }$ </td><td>43.781±0.239</td><td>43.738±0.238</td><td>44.867±0.151</td><td>44.817±0.150</td></tr><tr><td colspan="6">RelBench-v2</td></tr><tr><td>classification (AUROC× 100 ↑)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ratebeer/user-churn</td><td>63.98±0.00 62.40±0.84</td><td>89.42±0.15 76.84±0.17</td><td>89.49±0.13 76.81±0.17</td><td>89.73±0.12 78.93±0.07</td><td>89.73±0.11</td></tr><tr><td>ratebeer/beer-churn ratebeer/brewer-dormant</td><td>63.67±0.32</td><td>77.96±0.12</td><td>77.95±0.13</td><td>79.73±0.04</td><td>78.90±0.08</td></tr><tr><td>arxiv/paper-citation</td><td>73.65±0.22</td><td>79.84±0.06</td><td>80.64±0.02</td><td>81.11±0.05</td><td>79.78±0.04 81.17±0.05</td></tr><tr><td></td><td>65.93</td><td>81.02</td><td>81.22</td><td>82.38</td><td>82.40</td></tr><tr><td colspan="6">Average</td></tr><tr><td>regression (MAE ↓) ratebeer/user-count</td><td>14.453±0.006</td><td>13.270±0.323</td><td>13.270±0.323</td><td>9.115±0.075</td><td>9.115±0.075</td></tr><tr><td>arxiv/author-publication</td><td> $0 . 5 7 7 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td>0.532±0.001</td><td>0.526±0.001</td><td>0.489±0.002</td><td>0.489±0.002</td></tr></table>

## E.3.2 Runtime

Table 5 compares filtering and training times. Filtering remains a major computational cost even though the prediction head is small. In particular, for target-row indices T, we need only $( H _ { t } ) _ { T } =$ $( S _ { t } ^ { K } \bar { \bar { \mathbf { X } } } _ { t } ) _ { T } . \mathrm { \bar { A t } } \ K = 2$ , associativity gives

$$
( { \pmb H } _ { t } ) _ { T } = \big ( ( { \pmb S } _ { t } ) _ { T } { \pmb S } _ { t } \big ) \bar { \pmb X } _ { t } = ( { \pmb S } _ { t } ) _ { T } \big ( { \pmb S } _ { t } \bar { \pmb X } _ { t } \big ) .\tag{8}
$$

The final output contains only target rows, but other visible nodes still contribute through intermediate propagation. As K increases, expanding neighborhoods can make the row-restricted computation more expensive than full propagation, so the implementation switches when the latter is cheaper. The larger neighborhoods can therefore cause a sharp increase in filtering time. For example, on rel-avito, filtering rises from about 5.5 minutes at K = 2 to 60–77 minutes at K = 4, then grows roughly linearly with K (Figure 4).

## E.4 Ablation Study with Labels as Lagged Features

To test whether historical labels can replace C&S when used as MLP inputs, we summarize each entity’s visible history with twelve features. For prediction row (v, t), let

$$
H _ { v } ( t ) = \{ ( t _ { i } , y _ { i } ) \} _ { i = 1 } ^ { m } , \qquad t _ { 1 } < \cdot \cdot \cdot < t _ { m } < t ,\tag{9}
$$

where $t _ { i }$ is the label observation time. For $m > 0$ , five features summarize history availability, size, recency, and label values:

$$
\begin{array} { l } { f _ { 1 } = { \displaystyle \mathbf { 1 } [ m > 0 ] } , \ } \\ { f _ { 2 } = \log ( 1 + m ) , \ ~ } \\ { f _ { 3 } = \log ( 1 + \Delta _ { ( 0 ) } ) , \ ~ } \\ { f _ { 4 } = y _ { ( 0 ) } , \ ~ } \\ { f _ { 5 } = \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } y _ { i } , \ ~ } \end{array}\tag{10}
$$

where $y _ { ( 0 ) }$ is the most recent label and $\Delta _ { ( 0 ) }$ its age in days. To capture recency at different scales, we add five averages weighted by recency rank,

$$
\mathrm { E W M A } _ { \alpha } = \frac { \sum _ { r = 0 } ^ { m - 1 } ( 1 - \alpha ) ^ { r } y _ { ( r ) } } { \sum _ { r = 0 } ^ { m - 1 } ( 1 - \alpha ) ^ { r } } ,\tag{11}
$$

where $\alpha \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$ , and rank $r = 0$ denotes the newest label and larger α emphasizes recent observations. Two further averages instead use elapsed time:

$$
\mathrm { H L } _ { \tau } = \frac { \sum _ { i = 1 } ^ { m } e ^ { - \Delta _ { i } / \tau } y _ { i } } { \sum _ { i = 1 } ^ { m } e ^ { - \Delta _ { i } / \tau } } ,\tag{12}
$$

where $\Delta _ { i } = t - t _ { i }$ , and $\tau \in \{ 3 0 , 1 8 0 \}$ days. Rows without visible history use an absence indicator and default summary values. We apply the same summaries to related entities’ historical labels to form the cross-entity feature block, admitting only labels observed strictly before t. Using these summaries, we compare three variants on the same filtered representation: the base MLP, the MLP augmented with self-history features, and the MLP augmented with both self-history and cross-entity label features. We then apply C&S to the latter model using the same cross-entity graph construction described in Appendix C.1. Table 12 presents the results on RelBench-v1. Notably, we observe that

Table 12: Historical labels as features and with C&S.
<table><tr><td>Task</td><td>MLP (+relational attributes)</td><td></td><td>+ self hist. + (self+cross) hist. + C&amp;S</td><td></td></tr><tr><td>amazon/user-churn</td><td>68.09</td><td>68.51</td><td>69.05</td><td>69.30</td></tr><tr><td>amazon/item-churn</td><td>80.58</td><td>80.91</td><td>81.12</td><td>81.41</td></tr><tr><td>avito/user-clicks</td><td>65.35</td><td>65.05</td><td>65.46</td><td>65.03</td></tr><tr><td>avito/user-visits</td><td>66.11</td><td>66.55</td><td>66.68</td><td>66.68</td></tr><tr><td>hm/user-churn</td><td>67.84</td><td>69.65</td><td>69.76</td><td>69.76</td></tr><tr><td>stack/user-badge</td><td>86.06</td><td>86.26</td><td>86.29</td><td>86.34</td></tr><tr><td>stack/user-engagement</td><td>88.52</td><td>90.03</td><td>90.27</td><td>90.46</td></tr><tr><td>trial/study-outcome</td><td>64.36</td><td>n/a</td><td>64.71</td><td>66.47</td></tr></table>

self-history features improve performance on tasks where an entity’s own past labels are informative, with gains of +1.81 on hm/user-churn and +1.51 on stack/user-engagement. Beyond these feature-based gains, applying C&S to the label-augmented model provides additional improvements on five of eight tasks, including +1.76 AUROC on trial/study-outcome, whose target entities have no self-history by construction. Overall, these results show that label propagation can add information beyond fixed history summaries, although the improvements vary across tasks and avito/user-clicks shows a small degradation.

## E.5 Discussion and Open Questions

Overall, our studies show that simple propagation methods allow lightweight models to recover a substantial portion of relational predictive power without learned aggregation. Integrating historical label signals further improves performance on tasks where target entities carry informative past outcomes. Nevertheless, the relational database schema itself does not specify which entity connections support effective label propagation. Since shared records create structural links without guaranteeing correlated labels, establishing a canonical method of graph construction for label propagation remains essential for advancing relational deep learning under realistic temporal constraints.