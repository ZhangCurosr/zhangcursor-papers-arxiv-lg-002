# Tram-FL: Reducing Communication and Computation Costs through Sequential Model Circulation in Decentralized Federated Learning

Kota Maejima, Takayuki Nishio, Asato Yamazaki, and Yuko Hara-Azumi

Abstract— Conventional decentralized federated learning (DFL) often focuses on clients, with each client maintaining a model copy, performing updates individually, and undertaking model exchange and integration. While fully leveraging computational resources can shorten training times, it can also lead to significant computational and communication waste. This is especially pronounced with non-independent and identically distributed (non-IID) data, where achieving high model accuracy demands extra resources. This research shifts focus to the model itself, aiming to realize DFL with minimal computation and communication costs. To this end, we propose Tram-FL (Traveling Model Training Mechanism for Decentralized Federated Learning), a mechanism designed to efficiently address these challenges. It sequentially trains a single model by circulating it among nodes. We address the training scheduling problem in model circulation-based training, specifically determining which nodes should update the model and the number of updates to perform. This is approached by considering the model’s circulation route and update iteration allocation, for which we propose simple yet effective methods. Additionally, with quantized momentum, Tram-FL achieves high accuracy with fewer model circulations while controlling communication load per transmission. Experimental results show that the proposed algorithm, even with non-IID data, converges to a global model with reduced communication and computation.

Index Terms—Decentralized Federated Learning, Communication Efficiency, Distributed Machine Learning.

## I. INTRODUCTION

tivates the adoption of federated learning (FL) [2], where participants can collaboratively learn a model by synchronizing only the locally trained model parameters without revealing their raw data. In conventional federated learning, a central parameter server coordinates a large consortium of participating clients, aggregating gradients or model parameters from clients to collaboratively advance the learning of a global model [3]. While this FL system is promising, they face numerous challenges. In some instances, large organizations can serve the role of a central server in internet of things (IoT) applications, yet in many FL environments, finding a reliable and robust central server is challenging. Furthermore, the failure of the central server introduces a single point of failure for the entire network [4].

Hence, a decentralized federated learning (DFL) framework has been proposed. This framework eliminates the need for a central server and synchronizes FL updates and aggregation among data nodes. Gossip averaging [5]–[7] is a well-known method in diverse, decentralized algorithms, allowing different nodes within a network to exchange information in a peerto-peer manner without the aid of a central server. [8]– [12] shows that gossip averaging, combined with stochastic gradient descent (SGD), is used to train deep learning models in a decentralized manner, exhibiting superior convergence properties. Building on these studies, gossip averaging was subsequently applied to the DFL scheme, where model updates and averaging are alternately conducted at local nodes [13]– [16]. This averaging consensus technique is employed in DFL and is commonly used in Consensus-Based distributed SGD (CDSGD) [17], decentralized parallel SGD (D-PSGD) [18], and related algorithm [19].

The previously mentioned DFL algorithms surmount several challenges inherent to conventional FL systems that necessitate a central server, yet they are not invariably efficient. Fig. 1 illustrates the training processes of the conventional DFL methodology alongside our proposed approach. As depicted in Fig. 1 (a), in conventional DFL, a multiplicity of learning models exist within the network, advancing in proficiency through frequent inter-nodal communication and updates. Consequently, the communication costs between clients and the computational costs of the clients often become significant bottlenecks. Furthermore, since different nodes possess distinct training datasets, usually non-overlapping and with diverse distributions (known as non-IID data), the optimal points of each node’s loss function can be substantially divergent, potentially far from the global optimum [20]. Consequently, the aggregated model may deviate substantially from the global optimum, necessitating additional communication and computation to achieve a high-precision global model.

Numerous studies have been conducted to address this issue. Methods have been proposed that impose constraints on model updates [21], apply techniques such as mutual knowledge transfer (MKT), quadratic programming (QP), and knowledge distillation [22]–[24], utilize the similarity of label distributions [25], [26], and effectively control updates and communications [27], [28]. However, while these methods improve the convergence of models in non-IID settings, they still involve frequent updates and communications across multiple models, thus resulting in substantial computational and communication costs. Furthermore, although their effectiveness in non-IID settings with minor data skewness has been demonstrated in scenarios with significant data skewness, achieving a highaccuracy global model still necessitates additional communication and computation.

![](images/e47258b96e9bcbd424baafe593925d7c9c059a1c0a7690121a320558b924feed.jpg)

![](images/b5abdb4be6615eb62790455e020cd6f6f8af89ed6b6692d548d069e6bc6f8f45.jpg)  
(a) Conventional DFL  
(b) Ours: Tram-FL  
Fig. 1. Training processes of the conventional DFL methodology alongside our proposed approach Tram-FL.

To address the aforementioned challenges, we introduce a novel DFL framework named Tram-FL. Unlike existing methodologies, Tram-FL enhances learning by circulating a single global model among nodes as Fig. 1 (b). In other words, instead of utilizing local models or model aggregation, each node directly updates the global model. By carefully scheduling the circulation-based model training, specifically determining the nodes through which the model circulates and the number of update iterations, it is feasible to mitigate the accuracy degradation caused by non-IID data. For example, if certain nodes contain data primarily from Classes 1-3, 4- 6, and 7-9, respectively, circulating the model sequentially through these nodes helps prevent the model from becoming biased toward any specific class. Furthermore, compared to traditional DFL, Tram-FL achieves quicker model convergence and requires only a single model to be learned within the network, thereby significantly reducing both computational and communication costs.

The scheduling problem is complex, as it must consider network topology, distribution of data held by each node, and their computational capabilities. This paper proposes an approach where, under a simplified scenario, the number of update iterations and the determination of routes are addressed disjointedly, offering a streamlined solution to this complex problem. We define the method of determining the update iterations number as ‘mini-batch number allocation’ and the decision of a suitable circulation route as ‘route determination. The proposed mini-batch number allocation determines the number of update iterations by minimizing the distribution bias in the data used for previous model training. Furthermore, the proposed route determination selects a route that avoids shortterm data biases while imposing constraints of consecutive training on the last node of the cycle.

We further augment the aforementioned methodology by adding the quantized momentum. Specifically, during model training, momentum is computed and transmitted to the next node along with the model parameters. However, transmitting momentum as-is would increase communication costs, leading to a trade-off between model accuracy and communication expenses. Therefore, we adopt a technique of compressing and transmitting the momentum under the assumption that the precision of momentum does not significantly impact training.

To evaluate the efficacy of Tram-FL, we conducted experiments utilizing multiple datasets. These experiments involved a comparative analysis of model accuracy per communication traffic and model update with existing methodologies. Furthermore, we assessed the impact of momentum compression on model accuracy. The results demonstrated that Tram-FL achieved stable convergence with significantly reduced communication traffic and model updates. Additionally, it was evidenced that compression of momentum up to a certain threshold offers greater benefits in reducing communication traffic than the corresponding deterioration in accuracy.

The contributions of this paper are summarized as follows:

• We introduce Tram-FL, a novel approach that enhances learning by circulating a single global model among nodes. Unlike traditional DFL methods, which frequently employ local models and model aggregation, Tram-FL operates by allowing each node to directly update the global model. This methodology significantly reduces communication and computational costs by eliminating the need for frequent updates and transmissions of multiple models, a prevalent bottleneck in conventional DFL techniques.

• We introduce mini-batch number allocation to determine the optimal number of update iterations for each node, route determination to decide the most efficient circulation path for the model, and the momentum compression and transmission. These enhancements augment the learning efficiency of Tram-FL and achieve further reductions in communication and computational expenses.

This paper extends our previous work [29], which provided a preliminary demonstration of a model circulation-based DFL mechanism. In this paper, we further advance the initial framework by developing a method for adjusting the number of mini-batches used in each node’s local training, along with the quantized momentum, leading to significant alterations in the method of determining the model’s circulation route to ensure compatibility with the proposed mini-batch number allocation. Additionally, we performed extensive evaluations to thoroughly assess the effectiveness of these improvements.

The remainder of this manuscript is structured as follows: In the Section II, we describe conventional DFL and research aimed at improving model accuracy in non-IID settings. In the Section III, we elaborate on the system model, problem formulation, the mini-batch number allocation, the method for determining cyclical routes, and the approach for momentum compression. In the Section IV, we assess the proposed method through experiments utilizing an image classification dataset. In the Section V, we articulate the conclusions of this manuscript and propose directions for future research.

## II. RELATED WORKS

As delineated in the Section I, within non-IID settings, DFL encounters formidable challenges in achieving a highaccuracy global model. Numerous studies have been conducted to address this issue.

PDMM-SGD algorithm [21] integrates the PDMM algorithm [30], [31], originally designed for convex model optimization on P2P networks, into deep neural network (DNN) cost minimization problems, thereby reformulating the constrained minimization problem. The reconstructed problem incorporates linear consensus constraints, which advocate for uniformity of model variables across nodes in the network. This constraint ensures that the optimal point of each node’s loss function coincides with the global optimum. [22] effectively incorporates the benefits of MKT into the DFL framework. In each round of the proposed Def-KT algorithm, local models are transmitted to a randomly selected node. In contrast to averaging, each recipient node fuses the local model with the received model using MKT. Through MKT, two models with distinct knowledge learn from each other, resulting in a model that indirectly acquires knowledge from both sources. [23] introduces a cross-gradient aggregation method to address the challenges posed by non-IID data. This technique entails collecting derivatives of a model with respect to the datasets of neighboring nodes. The gathered gradients are then aggregated and utilized to update the model through projected gradients, which are optimized via quadratic programming (QP). [24] is a methodology derived from applying knowledge distillation to DFL, introducing two additional contrastive loss terms related to cross-features in each node. Cross-features encompass feature quantities obtained by inputting local data into the model of adjacent nodes, as well as those acquired by feeding data from neighboring nodes into the local model. Minimizing the contrastive loss function enhances the similarity among local models, thereby improving model performance and coherence.

There are attempts to improve the convergence of the dataheterogeneous DFL by controlling the topology of peer-topeer networks. This involves selectively choosing peers for the exchange of models and gradients. [25] enables nodes with similar data distributions to detect each other, collaboratively evaluate training losses on each other’s data, and learn models suited to their local data distributions. [26] proposes a novel topology for DFL, grouping nodes into sparsely connected cliques. This strategy ensures that the label distribution within cliques represents the global label distribution, thereby reducing gradient bias. [27] integrates adaptive control of both local update frequency and network topology. It establishes a theoretical relationship between local update frequency and network topology in terms of model training performance and seeks an upper bound on convergence. Based on this, it proposes an optimization algorithm that adaptively determines local update frequency and constructs network topology. [28] facilitates the reflection of data from all nodes on all models in the network, even in settings with large network topology bias, by efficiently using both swap, which exchanges models, and skip, which acts like a hub and sends one node’s model to another node.

TABLE I  
ANALYTICAL COMPARISON OF COMMUNICATION AND COMPUTATION PER TRAINING ROUND, WHERE ONE ROUND LETS EVERY NODE UPDATE ONCE (FOR TR $_ \mathrm { A M - F L }$ , ONE CYCLE). n: NODES, d: AVERAGE DEGREE, M: MODEL SIZE, $M _ { \mathrm { m o m } } ;$ : COMPRESSED MOMENTUM SIZE.
<table><tr><td></td><td>Transmissions per round</td><td>Models trained</td></tr><tr><td>Gossip-SGD</td><td> $\overline { { O ( n d ) \times M } }$ </td><td>n</td></tr><tr><td>PDMM-SGD</td><td> $O ( n d ) \times M$ </td><td>n</td></tr><tr><td>Tram-FL</td><td> $n \times ( M + M _ { \mathrm { m o m } } )$ </td><td>1</td></tr></table>

These methodologies have addressed several challenges faced by DFL in non-IID settings. However, all of them retain the premise of maintaining multiple models in parallel across the network. Under this premise, two costs remain structural: the training computation, which is proportional to the number of models maintained in the network, and the recurrent model exchanges required for aggregation or consensus. Table I summarizes this point analytically: with n nodes of average degree d and model size M, gossip- and consensus-based methods transmit $O ( n d )$ model copies per round and train n models in parallel, whereas circulating a single model reduces the transmissions to n transfers of a single model (plus compressed momentum $M _ { \mathrm { m o m } } )$ per cycle—i.e., from $O ( n d )$ to $O ( n )$ model transmissions per round—and the number of maintained models to one. Reducing the number of models in the network to one is the design point that eliminates both structural costs by construction, which partial refinements of parallel DFL cannot reach. This observation motivates the model-circulation approach proposed in this paper, targeting accurate global models in highly skewed non-IID settings with minimal communication and computational costs.

## III. PROPOSED METHOD

This section presents the proposed Tram-FL. In a nutshell, Tram-FL trains a single global model by circulating it among nodes: at the beginning of each cycle, the number of training mini-batches for every node is allocated so as to balance the cumulative label usage (Step 1), the circulation route is determined accordingly (Step 2), and the model and momentum are then updated at each node (Step 3) and forwarded with quantized momentum to the next node (Step 4), as illustrated in Fig. 2 and summarized in Algorithm 1. In the following, Section III-A describes the system model, Section III-B details the training mechanism, Section III-C and Section III-D present the mini-batch number allocation and the route determination, respectively, Section III-E describes the momentum compression, and Section III-F summarizes the overall procedure.

## A. System Model

FL can be categorized into two distinct types: cross-device FL and cross-silo FL [32]. In cross-device FL, nodes are typically small-scale distributed entities, such as smartphones, wearables, and edge devices. These nodes possess only a modest amount of local data. Participation in each training round is restricted to a mere fraction of these nodes. Hence, the success of cross-device FL typically necessitates the involvement of a considerable number of edge devices, potentially reaching up to several million. Conversely, in cross-silo FL, nodes are customarily entities such as corporations or organizations (like hospitals or banks). The number of participants is relatively limited (ranging from 2 to 100 entities), and each node is expected to participate in the entire training process. This paper presupposes that the data servers involved in cross-silo FL (hereinafter referred to as ’nodes’) collaboratively learn a DNN model for a specific shared application.

In this study, we assume that corporate data servers in cross-silo FL act as nodes. Each node possesses sufficient computational power to train a model on its local data within a reasonable time frame, and the nodes communicate with each other over the Internet with acceptable latency; we assume reliable logical links with no network disconnections during training. The communication traffic generated when exchanging models between nodes is treated as the communication overhead of the training process, and our proposed method aims to improve the trade-off between this communication traffic and the accuracy of the trained model. Issues related to reducing communication and computation latency for greater real-time performance, which require considering heterogeneous communication bandwidths and computation power among nodes, are beyond the scope of this paper and will be explored in future work.

We note, however, that the circulation mechanism itself offers a natural recovery path against node failures: because the entire training state is carried by the circulating model and momentum, a failure is detected by a transmission timeout; since the preceding node retains its copy of the model and momentum until the transfer is acknowledged, it can re-route the model to the next reachable node in the order of the allocated batch counts, and the failed node’s allocation can be re-distributed in the next cycle by re-solving the allocation with the node excluded. The cost of such a failure is thus the loss of at most one node’s worth of updates, not the loss of the training state. A quantitative evaluation of robustness under node churn is left as future work.

Because the model visits nodes sequentially, the wall-clock time of one cycle grows linearly with the number of nodes, and a single circulating model cannot exploit parallel computation across nodes. Tram-FL is therefore designed for cross-silo FL, where the number of participants typically ranges from several to around one hundred [32], rather than for cross-device FL with thousands of devices. For larger networks, a natural extension is to partition nodes into clusters and circulate one model per cluster with periodic inter-cluster synchronization, trading a part of the communication savings for parallelism; we leave this hierarchical extension to future work.

This paper assumes each node holds data that has been subjected to proper preprocessing steps, such as labeling and imputation of missing values, and it is postulated that the nodes are capable of training models using their own data. For the purpose of model training, nodes construct mini-batches from their data through random sampling. This assumption is ubiquitous in the field of FL. Particularly in a cross-silo setting, where each node represents an enterprise or organization, this assumption is commonplace due to the high reliability of the

data.

Furthermore, it is anticipated that statistical information about the datasets owned by the nodes, such as label distributions, can be shared in advance. Specifically, the distribution of data labels across all nodes participating in the training is shared beforehand. This information is crucial in determining the number of training batches and the route of the model, and it is a common assumption in non-IID settings of FL setups [33], [34].

We also note the privacy implication of this assumption. What is shared is only the per-label sample counts $L _ { i } ,$ i.e., a $| C |$ -dimensional histogram, rather than raw data, features, or gradients; the risk of reconstructing individual samples from such coarse statistics is considerably lower than that of gradient-sharing schemes, which are known to be vulnerable to gradient inversion attacks [35]. Nevertheless, label histograms may still reveal business-sensitive information (e.g., the case mix of a hospital). In such cases, the histograms can be perturbed by differential-privacy mechanisms or aggregated through secure multi-party computation before being shared; since the proposed allocation depends on $L _ { i }$ only through the expected label counts, it remains compatible with noisy statistics at the cost of a looser approximation of the uniformdistribution objective. A quantitative study of this privacy– utility trade-off is left for future work.

## B. Model Training Mechanism of Tram-FL

Tram-FL aims to achieve model convergence with minimal communication and computational costs in highly skewed non-IID settings. A pivotal concept in realizing this goal is the circulation of a single model across the network with an optimized number of batches and routes during the learning process. Fig. 2 illustrates the sequence of model updates and transfers among the nodes. These procedures are delineated into four distinct steps: Step 1 is the mini-batch number allocation step, Step 2 is the route determination step, Step 3 is the local model and momentum update step, and Step 4 is the model and momentum transmission step. Here, one cycle is defined as the global model traversing all nodes participating in the training. At the onset of each cycle, the node that first acquires the global model initiates Step 1: mini-batch number allocation step to determine the number of batches for model updating to be utilized by each node throughout the cycle. Given that the batch size is fixed, this procedure enables the regulation of the update frequency for each node. Subsequently, Step 2: route determination step determines the circulation route of the model for the cycle. Afterward, in Step 3: local model and momentum update step, the model and momentum are updated based on the predetermined batch count. Upon completion of the updates, the model and momentum are transmitted to the subsequent node following the prescribed circulation route in Step 4: model and momentum transmission step. This procedure is iteratively executed until the model validation loss or accuracy reaches a predetermined threshold or until a specified number of communication cycles are completed.

The implementation of Tram-FL is depicted in Algorithm 1. Let $w ^ { t }$ denote the DNN model variables, $v ^ { t }$ the momentum variables, and $v _ { q } ^ { t }$ the compressed momentum at model update iteration t. The DNN cost function to be minimized is represented as $F ( \cdot )$ , with momentum compression and decompression denoted by $C ( \cdot )$ and $D ( \cdot )$ , respectively. V represents the set of nodes, α the momentum coefficient and η the learning rate. During K cycles of iteration, Steps 1 through 4 are iteratively executed. At the beginning of each cycle, the mini-batch number allocation step determines the batch count for each node to update the model. Following this, the route determination step establishes the model’s circulation route for the cycle. If not the initial model update, decompression of the received compressed momentum from the previous node takes place. The nodes follow the order determined in the route determination step to execute the local model and momentum update step and the model and momentum transmission step. In the local model and momentum update step, the node holding the model updates w and v by extracting the batch count determined in the mini-batch number allocation step from its own training data. During the model and momentum transmission step, the momentum is compressed and then transmitted along with the model to the next node via the determined route in the route determination step. The methods for determining the training batch count for each node and the model’s circulation route in the mini-batch number allocation and route determination steps, as well as the method for compressing the momentum, are elaborated in Section III-C to Section III-E.

![](images/10a7b63ebe822e6e4e7a20b8eb3fdd7a1fcbbfacd9caf93f99cb7b70c074f8b0.jpg)  
Fig. 2. Sequence diagram of the proposed Tram-FL model update procedure. In Step 1, the number of training batches for each node is determined. Subsequently, in Step 2, the model’s circulatory route is established, followed by Step 3, where the node updates the model parameters and momentum using mini-batches. Finally, in Step 4, the model and momentum are transmitted to the next node.

By transmitting a single model, our proposed framework can significantly reduce overall network traffic, offering an advantage over traditional approaches that involve transmitting multiple models. An inherent drawback of this approach is the serialization of model updates, which can potentially prolong the learning duration. However, the serialization also yields substantial benefits when handling non-IID data, and the reason can be explained as follows. In aggregation-based DFL, each node locally descends toward the minimizer of its own empirical risk, whose optimum can be far from the global optimum under label skew; averaging the resulting divergent parameters does not, in general, descend the global objective, a phenomenon known as client drift [36]. In contrast, Tram-FL applies all updates to a single model in sequence. From the model’s perspective, training is equivalent to stochastic gradient descent on a single virtual data stream formed by concatenating the mini-batches drawn along the circulation route. Our mini-batch number allocation explicitly shapes this stream: by minimizing the variance of the cumulative label histogram, the empirical label distribution of the stream converges to the uniform distribution as cycles proceed. Consequently, in the medium to long term, the update sequence approximates SGD on label-balanced (IID-like) data, avoiding the parameter divergence and cancellation inherent to parallel aggregation. This view is consistent with recent theoretical results showing that sequential FL enjoys better convergence guarantees than parallel FL on heterogeneous data [37]. Two sources of degradation remain: the short-term label bias within a cycle, and the catastrophic forgetting between consecutive visits to a node; the route determination in Section III-D is designed to suppress both. In addition, reducing the training batch count increases the frequency of model transmission and network traffic. Thus, optimizing both the training batch count and the model’s routing path is of paramount importance, and this study primarily focuses on these two elements.

Algorithm 1 Tram-FL   
1: Initialization of $w ^ { ( 0 ) }$ and $v ^ { ( 0 ) }$   
2: $t \gets 0$   
3: Locate $w ^ { ( 0 ) }$ to randomly selected node i   
4: for each cycle $k = 1 , 2 , 3 , \ldots , K$ do   
5: ▷ Step 1: Decide number of training batch x   
6: ▷ Step 2: Decide model’s circulation route   
7: for each node $i \in \nu$ do   
8: if t is not 0 then   
9: Decompression momentum   
10: $v ^ { t } \gets \bar { D } ( v _ { q } ^ { t } )$   
11: end if   
12: for training batch $x _ { i }$ do   
13: ▷ Step 3: Update model parameters and momentum   
14: $\boldsymbol { v } ^ { t + 1 }  \alpha \boldsymbol { v } ^ { t } - \eta \nabla F ( \boldsymbol { w } ^ { t } )$   
15: $w ^ { t + 1 }  w ^ { t } + v ^ { t + 1 }$   
16: $t \gets t + 1$   
17: end for   
18: Compression momentum   
19: $\tilde { v _ { q } ^ { t + 1 } } \bar {  } C ( v ^ { t + 1 } )$   
20: ▷ Step 4: Transmit model and momentum to next   
node   
21: Transmit $v _ { q } ^ { t + 1 }$ and $w ^ { t + 1 }$ to node j   
22: end for   
23: end for

## C. Mini-batch Number Allocation

In the mini-batch number allocation step, the number of mini-batches each node uses to update the model is determined. In standard machine learning, a fixed number of mini-batches—typically covering all samples in the training dataset—are used to update the model across multiple epochs. However, when client data are non-IID, using a set number of batches and multiple epochs without accounting for data distribution can lead to a model biased towards certain classes. To address this, our proposed method restricts the number of mini-batches each node uses, allowing the model to be updated in a balanced way as it circulates across nodes with varied data, thereby reducing bias.

We define a cycle as the period of time during which the global model travels through all nodes participating in the training. Let’s denote $L _ { i } ~ = ~ \{ l _ { i , c } ~ | ~ c ~ \in ~ C \}$ as the set of the number of training samples stored by node i, where C signifies a set of labels and $l _ { i , c } \geq 0$ indicates the number of samples with label c present at node i. The total number of samples possessed by node i is represented by $N _ { i }$ , hence $\begin{array} { r } { N _ { i } = \sum _ { c \in C } l _ { i , c } . } \end{array}$ Moreover, let $L _ { i } ^ { k }$ represent the number of samples utilized by node i for training the received model during cycle k, thus $L _ { i } ^ { k } \ = \ \{ l _ { i , c } ^ { k } \ | \ c \ \in \ C \}$ , with $l _ { i , c } ^ { k } , ( 0 \leq l _ { i , c } ^ { k } \leq l _ { i , c } )$ being the count of samples with label c used by node i in cycle k.

We posit that the cumulative number of samples used for model training up to cycle k is expressed as $L ^ { k }$ , yielding

$$
L ^ { k } = \Big \{ \sum _ { \kappa = 1 } ^ { k } \sum _ { i \in \mathcal { V } } l _ { i , c } ^ { \kappa } | c \in C \Big \} ,\tag{1}
$$

where $\kappa$ is the cycle index running over the past cycles. For simplified notation, we use $l _ { c } ^ { k }$ to represent $\textstyle \sum _ { \kappa = 1 } ^ { k } \sum _ { i \in \mathcal { V } } l _ { i , c } ^ { \kappa } .$

Our proposed mini-batch number allocation determines the number of training batches for each node such that $L ^ { k }$ adheres to a pattern closely approximating a uniform distribution, specifically aiming to achieve

$$
l _ { c } ^ { k } = l _ { c ^ { \prime } } ^ { k } ( c , c ^ { \prime } \in C ) .\tag{2}
$$

To accomplish this, the proposed algorithm identifies the number of training batches for each node that is expected to minimize the variance of $L ^ { k }$ . The selection policy can be formulated as follows:

$$
\quad x ^ { * } \quad = \quad \underset { x } { \mathrm { a r g m i n } } \sigma ( L ^ { k } + \sum _ { i \in \nu } \frac { B x _ { i } } { N _ { i } } L _ { i } )\tag{3}
$$

$$
x ~ = ~ \{ x _ { i } \mid i \in \mathcal { V } , x _ { i } \in \mathbb { Z } , 0 \leq x _ { i } \leq \frac { N _ { i } } { B } \}\tag{4}
$$

$$
\sigma ( L ^ { k } ) = \frac { 1 } { | C | } \sum _ { c \in C } ( l _ { c } ^ { k } - \frac { 1 } { | C | } \sum _ { c \in C } l _ { c } ^ { k } ) ^ { 2 }\tag{5}
$$

$$
\sum _ { i \in \mathcal { V } } x _ { i } > \beta ,\tag{6}
$$

where B is the batch size and $x _ { i }$ is the number of training batches for node i. In this study, we assume that the node receiving the model randomly samples its non-IID data to construct mini-batches of size B, which are then used for training in that round. Consequently, $\begin{array} { r } { \frac { B x _ { i } } { N _ { i } } L _ { i } } \end{array}$ reflects the expected number of samples labeled c that will be utilized by node i for training the model. Therefore, by determining the node according to (3), the expected value of the variance of $L ^ { k + 1 }$ can be minimized. When the variance becomes

0, our objective (2) is fulfilled. That is, the distribution of $L ^ { k + 1 }$ will be a uniform distribution, which suggests that all labels have been used equally for model training up to cycle $k + 1$ . This uniform distribution is desirable as it implies unbiased learning based on all available labels, leading to a model with balanced performance across all classes. $\beta$ denotes the minimum aggregate number of training batches, and a constraint is imposed such that the cumulative count of training batches per cycle surpasses $\beta ,$ , thereby averting the cessation of the training process. Note that since there are no constraints on the number of training batches for each node, scenarios may arise where $x _ { i } = 0$ , indicating nodes that do not participate in training.

We remark that the allocation naturally accommodates extreme label skew. If node i holds no samples of class c $( l _ { i , c } = 0 )$ , the expected contribution $\textstyle { \frac { B x _ { i } } { N _ { i } } } L _ { i }$ simply contains no mass on class c, and the optimizer compensates by allocating more batches to the nodes that do hold class c. In the extreme case where including a node only increases the variance objective, the solution sets $x _ { i } = 0$ and the node is skipped in that cycle, while the constraint $\textstyle \sum _ { i } x _ { i } > \beta$ guarantees that training never stalls. Our experimental setting, in which each node holds only 2–5 of the 10 labels, already covers the case where every node entirely lacks half or more of the classes.

The main objective of this paper is to demonstrate the effectiveness of the model circulation-based FL in improving the trade-off between model accuracy and communication overhead. Therefore, while efficient solutions for the optimization problem are not the focus of this study, we discuss here that the problem is manageable within reasonable constraints. Since this optimization problem is an integer programming problem due to the integer constraint on $x _ { i }$ , it is classified as NP-hard. This makes it challenging to obtain an exact solution when the number of nodes or the amount of training data per node is large. However, obtaining an exact solution is not essential for this optimization problem. If there is a significant class imbalance in the training data, it could impact model performance, but otherwise, the effect is expected to be minimal. Thus, it is sufficient to design a heuristic that provides an approximate solution, even with a loose approximation ratio.

Moreover, the main application scenario of this circulationbased distributed FL is cross-silo FL, where the number of participants is typically limited to a few dozen, and the dataset sizes are relatively small. (If nodes possessed large datasets, they would likely train models individually without participating in FL.) Therefore, exploratory metaheuristics such as random search are also valid options. Specifically, if the value of $\sigma ( L ^ { k } )$ is close to zero in cycle 0, exploring the neighborhood of the solution obtained in cycle 0 for subsequent cycles $k \geq 1$ may lead to efficient discovery of good solutions. In the experimental evaluation of this study, since both the number of nodes and the data volume per node are small, we determine the mini-batch size through exhaustive search.

## D. Route Determination

In the route determination step, the circulating path of the model is determined for nodes to which the number of batches has been allocated in the mini-batch number allocation step. Considering that the data on clients is non-IID, following a statically determined route without accounting for data distribution can lead to a model biased towards certain classes of data in the short term, significantly deviating from the global optimum. To address this, our proposed methodology optimizes the circulating route of the model. This prevents the model from being updated in an imbalanced manner across different classes during its circulation among nodes, thereby averting short-term biases in the model.

As a premise for determining the routing path, nodes with fewer training batches tend to exhibit a greater bias in data distribution (stronger non-IID). This is because our proposed aim is to minimize the bias in the label distribution of data used for training. Consequently, nodes with less bias in their data label distribution end up having a higher number of training batches. Conversely, nodes with a more significant bias in their data label distribution have fewer training batches. Therefore, training nodes with fewer batches early in the cycle is anticipated to skew the data distribution used for model updates in the short term. To counteract this, we propose a routing path that trains nodes with a higher number of batches first, thus avoiding short-term training data bias.

Furthermore, this route determination is expected to prevent catastrophic forgetting. Our proposed Tram-FL learns by circulating a single model throughout the network. Therefore, once a node has trained the model, its data is not used for training again until the model returns to that node. This interim period could lead to catastrophic forgetting. The training sequence determined by the route determination implies that nodes trained earlier are more likely to receive and train the model later, increasing the risk of catastrophic forgetting. However, by prioritizing training at nodes with larger batch numbers, our route determination potentially reduces this risk, as nodes with higher catastrophic forgetting risk undergo more extensive batch training.

We impose a constraint in our route determination where the last node of a cycle continues training consecutively. Specifically, if node ’i’ is the last to train in cycle t-1, it is also the first to train in cycle t. This approach is expected to reduce communication frequency by one cycle per round, leading to a maximum communication cost reduction of 1/n, considering ’n’ as the total number of nodes. Given our focus on a crosssilo setting with fewer nodes, this constraint is anticipated to reduce communication traffic significantly.

Our route determination assumes that any node can forward the model to any other node, which holds for cross-silo FL over the Internet, where connectivity is logical rather than physical. If the overlay is sparse, the route must be chosen from the feasible successors only, and a strongly constrained topology (e.g., a line graph) may force label-imbalanced orderings, degrading the short-term balance that our route rule provides. Because Steps 1 and 2 are executed at the beginning of every cycle, the formulation permits per-cycle reoptimization under the current connectivity, which is a natural basis for dynamic route adaptation.

Finally, we note the scope of the route determination in this paper. It adopts a deterministic rule that orders nodes by their allocated batch counts, which suppresses short-term label bias and mitigates the forgetting risk as described above. A full treatment of route optimization, however, should jointly account for per-node resource constraints such as heterogeneous computation capabilities, time-varying communication bandwidths, and node availability, all of which interact with the batch allocation and constitute a training-scheduling problem in its own right. We therefore leave the joint optimization of the circulation route under resource constraints to future work. Indeed, our follow-up study on load-aware training scheduling for model-circulation-based DFL [38] addresses a part of this problem by scheduling training under computation and communication load variations, and experimentally confirms that the node selection and ordering strategy materially affects convergence.

## E. Momentum Compression

In Tram-FL, momentum is computed and transmitted to the subsequent node during training. This approach is predicated on the findings indicating that incorporating the concept of momentum into FL, as demonstrated in [36], [39], [40], enhances model accuracy. However, direct transmission of momentum increases communication costs, thereby creating a trade-off between model accuracy and communication overhead. Consequently, we adopted a methodology wherein the momentum is compressed prior to transmission based on the premise that the precision of momentum does not significantly impact training.

For momentum compression, we employed quantization, a technique commonly used for compressing DNN models. Quantization reduces the bit-width of model parameters, which are typically represented as 32-bit floating-point numbers, to 8-bit or even lower integer values. Various quantization methods for DNNs have been proposed [41]. We adopt dynamic affine quantization because it is stateless, incurs negligible computational overhead, and bounds the element-wise approximation error, which suits the momentum whose precision requirement is loose. Alternative compression schemes are complementary rather than competing: sparsification (e.g., topk) can achieve higher compression ratios but requires errorfeedback memory at each node, which conflicts with our design in which nodes are stateless between visits; knowledgedistillation-based compression transfers knowledge rather than parameters and presupposes an auxiliary dataset at each node. Combining such techniques with model circulation is left as future work, since they apply equally to all DFL baselines and do not affect the relative comparison in this paper.

The conversion of a real value $v \in [ p , q ]$ into a signed b-bit integer value $v _ { q } \in \{ - 2 ^ { b - 1 } , - 2 ^ { b - 1 } + \bar { 1 } , \dots , 2 ^ { b - 1 } - \bar { 1 } \}$ can be expressed using the following formula:

$$
\begin{array} { r c l } { v _ { q } } & { = } & { \mathrm { c l i p } ( \mathrm { r o u n d } ( s \cdot v + z ) , - 2 ^ { b - 1 } , 2 ^ { b - 1 } , - 1 \ j 7 ) } \end{array}
$$

$$
\begin{array} { c c c } { \mathrm { c l i p } ( v , l , u ) } & { = } & { \left\{ \begin{array} { c c } { l , { \ v } < l } \\ { v , \ l \le v \le u } \\ { u , \ v > u } \end{array} \right. } \end{array}\tag{8}
$$

$$
s = \frac { 2 ^ { b } - 1 } { q - p }\tag{9}
$$

$$
z ^ { \mathrm { ~ ~ } } = - \mathrm { r o u n d } ( p \cdot s ) - 2 ^ { b - 1 } .\tag{10}
$$

Here, s represents the scale factor, and z denotes the zeropoint, which is determined by the range $[ p , q ]$ of the quantization target, as well as by the bit length after quantization. ”round” means to convert a real number to the nearest integer value. Conversely, the formula for dequantization is represented as follows:

$$
\hat { v } \quad = \quad { \frac { 1 } { s } } ( v _ { q } - z ) .\tag{11}
$$

It is important to note that obtaining an exact replica of the original input v through dequantization is challenging due to the rounding of values during quantization. Therefore, the above formula yields only an approximate value $\hat { v } \approx v .$

In a manner akin to momentum, models can also be quantized, a known technique for reducing communication costs during model transmission. Previous research indicates that even when models are quantized to 8 bits, the accuracy degradation is limited to approximately 1%. Consequently, applying model quantization to the proposed methodology is likely to yield a reduction in communication costs with minimal impact on model accuracy. However, since model quantization is applicable to most existing DFL methods, applying it to our proposed approach could lead to a lack of uniqueness and compromise fairness in comparative analysis. Consequently, model compression is not considered within the scope of this paper.

## F. Summary

In summary, each cycle of Tram-FL proceeds as follows. The node initiating the cycle solves the mini-batch number allocation to assign the number of training batches $x _ { i }$ to every node (Step 1) and orders the nodes with allocated batches in descending order of $x _ { i }$ , keeping the last node of the previous cycle first (Step 2). The model and the momentum then circulate along this route: each node performs its allocated momentum-SGD updates (Step 3) and forwards the model together with the quantized momentum to the next node (Step 4). The cycles repeat until the validation accuracy reaches the target or the communication budget is exhausted.

## IV. EVALUATION

## A. Experimental setup

In this study, we benchmarked Tram-FL against baseline methodologies, including PDMM-SGD, Gossip-SGD, and Centralized machine learning. We adopted image classification tasks using the MNIST [42] and CIFAR-10 [43] datasets. MNIST comprises images of handwritten digits across 10 classes, while CIFAR-10 consists of color images of objects belonging to 10 classes. The sample sizes for the training and test sets were 60,000, 10,000 for MNIST and 50,000, 10,000 for CIFAR-10, respectively. We emulated a mesh network with five nodes, randomly assigning 2 to 5 data labels to each node, ensuring that all labels were allocated across the nodes. Samples were distributed to each node according to the label allocation, with each node receiving all samples associated with its distributed data labels. Consequently, the number of samples increased proportionally to the number of assigned labels.

TABLE II  
DNN STRUCTURE FOR MNIST
<table><tr><td>32 ch Conv  ${ \overline { { ( 3 \times 3 , } } }$  valid), ReLU</td></tr><tr><td>64 ch Conv  ${ \overline { { ( 3 \times 3 , } } }$  valid), ReLU</td></tr><tr><td>64 ch Max pooling (2 × 2), Dropout (0.25), Flatten</td></tr><tr><td>Dense (128 neurons), ReLU, Dropout (0.5), Dense (10 neurons), Softmax</td></tr></table>

TABLE III  
DNN STRUCTURE FOR CIFAR-10
<table><tr><td rowspan=1 colspan=1>32 ch, Conv (3 × 3, same size), Group Norm (32 ch × 1group), ReLU</td></tr><tr><td rowspan=1 colspan=1>64 ch, Conv ${ \overline { { ( 3 \times 3 , } } }$ same size), Group Norm (32 ch × 2groups), ReLU</td></tr><tr><td rowspan=1 colspan=1>64 ch, Max pooling ${ \overline { { ( 2 \times 2 ) } } } ,$ Dropout (0.25)</td></tr><tr><td rowspan=1 colspan=1>64 ch, Conv ${ \overline { { ( 3 \times 3 , } } }$ same size), Group Norm (32 ch × 2groups), ReLU</td></tr><tr><td rowspan=1 colspan=1>64 ch, Conv ${ \overline { { ( 3 \times 3 , } } }$ same size), Group Norm (32 ch × 2groups), ReLU</td></tr><tr><td rowspan=1 colspan=1>64 ch, Max pooling (2 × 2), Dropout (0.25), Flatten</td></tr><tr><td rowspan=1 colspan=1>Dense (512 neurons), ReLU, Dropout (0.5), Dense (10 neurons), Softmax</td></tr></table>

The DNN architectures employed for training on the MNIST and CIFAR-10 datasets are presented in Table II and Table III. In such scenarios, batch normalization [44] is customarily inserted before each convolution layer as the de facto standard. However, when utilizing non-IID datasets, the statistics of mini-batch sets are skewed, making group normalization [45], layer normalization [46], and instance normalization [47] more appropriate choices. For the CIFAR-10 DNN structure, we employed group normalization in our normalization process, which utilizes the mean/variance of channel groups. The initial weights, $w ^ { ( 0 ) }$ , were initialized using He’s method [48]. In PDMM-SGD, a common seed value was utilized for initialization across all nodes. Momentum was compressed using 8-bit quantization. Refer to Table IV for additional parameters.

Furthermore, we investigated the relationship between communication cost and model accuracy due to momentum compression. This analysis was conducted using a 5-node mesh network emulated with CIFAR-10. Momentum was originally 32-bit and was compressed to 16-bit, 8-bit, 4-bit, and 2-bit through quantization. The other settings remained consistent with those previously described.

TABLE IV  
EXPERIMENTAL SETUP
<table><tr><td>Dataset</td><td>MNIST</td><td>CIFAR-10</td></tr><tr><td>Number of nodes</td><td>3, 5, 10</td><td>3, 5, 10</td></tr><tr><td>Model size</td><td>1,199,882</td><td>2,168,746</td></tr><tr><td>Input size</td><td>784</td><td>1024</td></tr><tr><td>Mini-batch size</td><td>100</td><td>100</td></tr><tr><td>Learning-rate</td><td>0.005</td><td>0.005</td></tr><tr><td>α</td><td>0.9</td><td>0.9</td></tr><tr><td>β</td><td>15, 25, 50</td><td>15, 25, 50</td></tr></table>

## B. Experimental result

1) Performance comparison with benchmarks: Fig. 3 illustrates the test accuracy relative to communication traffic. In the Gossip-SGD and PDMM-SGD algorithms, the accuracy of the global model was determined by averaging the accuracies of all models within the network. As communication traffic escalates, the model undergoes updates and advances in learning, culminating in enhanced test accuracy across all tasks. The experimental outcomes for MNIST and CIFAR-10 are respectively delineated in Fig. 3 (a) and (b). The centralized IID (red solid line) represents the accuracy achieved when all datasets are consolidated at a single location for training. In every setting, Tram-FL converged with less communication traffic compared to Gossip-SGD and PDMM-SGD. An increase in the number of nodes tends to deteriorate overall convergence. It is particularly noteworthy that in Gossip-SGD and PDMM-SGD, the impact of varying node counts is pronounced, significantly diminishing accuracy improvement when the node count reaches 10. Conversely, Tram-FL exhibits relatively minor effects from changes in node count, consistently achieving the accuracy of centralized IID even with 10 nodes.

It may appear counterintuitive that a decentralized method matches, and in some runs slightly exceeds, the centralized-IID reference. Two factors explain this behavior. First, the mini-batch number allocation makes the label usage exactly balanced over cycles, whereas centralized training samples mini-batches uniformly from the pooled dataset, whose label proportions fluctuate batch by batch; balanced label exposure can yield marginally better generalization on the classbalanced test set. Second, the restricted number of updates per cycle with circulated momentum acts as an implicit regularization, reducing overfitting to any particular subset of the data. We emphasize that the observed differences are marginal and should be interpreted as Tram-FL attaining centralized-level accuracy rather than as evidence of systematic superiority over centralized training.

Fig. 4 demonstrates the test accuracy in relation to the total number of model updates. Fig. 4 (a) and (b) respectively present the experimental results for MNIST and CIFAR-10. In every scenario, Tram-FL achieved convergence with fewer model updates than Gossip-SGD and PDMM-SGD, signifying a reduction in computational costs. Additionally, akin to the results in Fig. 3, Tram-FL demonstrated minimal impact from node count variations and exhibited stable convergence characteristics.

Table V presents the communication traffic and the number of updates required to achieve a specified level of accuracy. For the MNIST dataset, thresholds were set at 99%, 90%, and 80%, while for the CIFAR-10 dataset, thresholds were established at 80%, 60%, and 40%. In the Gossip-SGD and PDMM-SGD algorithms, the accuracy of the global model was determined by averaging the accuracies of all models within the network. The ”-” category represents scenarios in which the thresholds were not attained. In the experiment utilizing the MNIST dataset with Tram-FL, the interval for testing was set after every 500 model updates. It is noteworthy that the thresholds of 80% and 90% were achieved prior to the initial 500 model updates. It was observed that the Gossip-SGD and PDMM-SGD algorithms failed to surpass certain thresholds in some scenarios. In contrast, Tram-FL exceeded all the designated thresholds with lower communication traffic and fewer updates. When comparing only scenarios where thresholds were exceeded with PDMM-SGD, the communication traffic for MNIST was reduced by an average of 84.7%, and the number of model updates by 96.8%. For CIFAR-10, the reductions were 83.6% in communication traffic and 96.1% in the number of model updates, respectively. Considering that scenarios not surpassing the thresholds would require additional communication and computation to do so, these figures represent minimum values, suggesting that actual communication and computation costs could be further reduced. These measured reductions are also consistent with the analytical comparison in Table I: the savings stem from transmitting and training a single model rather than n models in parallel.

2) Impact of Compression Rates on Model Accuracy: Fig. 5 illustrates the relationship between the cumulative number of model updates within the network and the corresponding test accuracy for the initial scenario. This data represents the mean of 15 trials conducted with varying seed values. It is observed that as the total number of model updates between nodes increases, the model undergoes continuous updates and learning, resulting in enhanced test accuracy across all tasks. Remarkably, there is a negligible difference in the accuracy per model update among the 32-bit non-quantized momentum, 16-bit quantized momentum, and 8-bit quantized momentum. Furthermore, although the 4-bit and 2-bit quantized momentum requires a greater number of updates to converge, the final test accuracy is virtually identical to that of the non-quantized momentum. The communication traffic per transmission is significantly reduced by quantizing the momentum, while the model accuracy remains largely unaffected. Consequently, this approach holds promise for substantial reductions in overall communication costs.

Fig. 6 presents a box-and-whisker plot depicting the communication traffic (in GB) at the point where test accuracy reaches 80%. This data is based on an average of 15 trials conducted with varying seed values. All methods employing momentum quantization demonstrated a reduction in communication traffic compared to the approach without momentum quantization. Notably, the 8-bit quantized momentum method achieved an average reduction of 40.09% in communication traffic compared to the non-quantized approach.

## V. CONCLUSION

In this paper, we propose a novel DFL methodology based on model circulation, termed Tram-FL, which incorporates the transmission of compressed momentum. We also present an example of the mini-batch number allocation and a compatible route determination. Tram-FL operates by sharing a single model across the entire network. The learning process is advanced by circulating this model with an optimal number of training batches and a designated circulation route, thereby realizing a global model. Our experiments demonstrate that while the compression of momentum predominantly contributes to a substantial reduction in communication traffic, it does not significantly compromise accuracy. Furthermore, Tram-FL successfully acquires an accurate global model with minimal communication and computational costs in various scenarios, even when non-IID datasets are distributed among nodes. This suggests that efficient learning on non-IID datasets is feasible if an optimal mini-batch number allocation is established, enabling the selection of the ideal training batch size, and a route determination is implemented for choosing the most effective circulation route. We conclude by summarizing the limitations of this work. First, Tram-FL targets crosssilo FL; the sequential circulation limits parallelism, and the wall-clock time of a cycle grows linearly with the number of nodes. Second, the mini-batch number allocation relies on pre-shared label histograms, whose privacy implications were discussed but not quantitatively evaluated. Third, the evaluation assumes reliable links and no node churn. Fourth, the route determination adopts a deterministic rule; its joint optimization with per-node resource constraints (heterogeneous computation capabilities, time-varying bandwidths, and node availability) is a training-scheduling problem in its own right, a part of which is addressed by our follow-up work [38].

![](images/f446766ed9e6522d1ca3646a12635e5263edb9186e7c7b301895296cf9ea9941.jpg)  
(a) MNIST

![](images/f6a7f17c73a75b258ae9a6d0974a6e3b5258d7265732d87c3ac56890b71b534e.jpg)  
(b) CIFAR-10  
Tram-FL 3nodes Tram-FL 5nodes Tram-FL 10nodes PDMM 3nodes PDMM 5nodes PDMM 10nodes Gossip 3nodes Gossip 5nodes Gossip 10nodes Centralized IID

communication traffic (GB)Fig. 3. Test accuracy as a function of the communication traffic, utilizing non-IID data sets. Evaluation includes (a) MNIST for handwritten digit classification and (b) CIFAR-10 for object recognition. 8-bit quantization was used for momentum compression.  
![](images/b86ffd2c9e90b5916ff4d2eb0f3ed815eacb4593820fdfc50b62c0c03bf85e90.jpg)  
(a) MNIST

![](images/2aa686e1479f75c03d4fb17e27b34ad3f4ed91a01885ba585f78424a65916174.jpg)  
(b) CIFAR-10  
Tram-FL 3nodes Tram-FL 5nodes Tram-FL 10nodes PDMM 3nodes PDMM 5nodes PDMM 10nodes Gossip 3nodes Gossip 5nodes Gossip 10nodes Centralized IID

communication traffic (GB)Fig. 4. Test accuracy as a function of the total number of model updates, utilizing non-IID data sets. Evaluation includes (a) MNIST for handwritten digit classification and (b) CIFAR-10 for object recognition. 8-bit quantization was used for momentum compression.  
![](images/f99398d03965655c5b02a0af0517434144b08b36ebc456bb648a74f15d14f40f.jpg)  
Fig. 5. Test accuracy as a function of the total number of model updates across different bits of model quantization. 5-node mesh network with CIFAR-10.

![](images/d9926d9d0523b3fb79f93f0b07480ffa8fade1db22cfcf3383a6f2e1c7929925.jpg)  
Fig. 6. Comparison of the number of model transmissions required to reach an 80% accuracy level, illustrated across different bits of model quantization.

TABLE V  
COMMUNICATION TRAFFIC AND NUMBER OF UPDATES TO REACH A CERTAIN ACCURACY.  
“-“ MEANS THE THRESHOLD WAS NOT ATTAINED.
<table><tr><td rowspan="2"></td><td rowspan="2"></td><td colspan="3">Tram-FL</td><td colspan="3">PDMM-SGD</td><td colspan="3">Gossip-SGD</td></tr><tr><td>Acc.</td><td>3nodes</td><td>5nodes</td><td>10nodes</td><td>3nodes</td><td>5nodes</td><td>10nodes</td><td>3nodes 5nodes</td><td></td><td>10nodes</td></tr><tr><td rowspan="6">MNIST</td><td rowspan="3">Communication traffic (GB)</td><td>99%</td><td>9.31</td><td>9.03</td><td>18.19</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>90%</td><td>1.86</td><td>2.24</td><td>2.36</td><td>3.20</td><td>16.09</td><td></td><td>3.22</td><td>32.18</td><td></td></tr><tr><td>80%</td><td>1.86</td><td>2.24</td><td>2.36</td><td>1.59</td><td>5.36</td><td>42.91</td><td>1.61</td><td>5.36</td><td></td></tr><tr><td rowspan="3">Model update (times)</td><td>99%</td><td>2500</td><td>2000</td><td>3800</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>90%</td><td>500</td><td>500</td><td>500</td><td>3600</td><td>18000</td><td></td><td>3600</td><td>36000</td><td></td></tr><tr><td>80%</td><td>500</td><td>500</td><td>500</td><td>1800</td><td>6000</td><td>48800</td><td>1800</td><td>6000</td><td></td></tr><tr><td rowspan="6">CIFAR-10</td><td rowspan="3">Communication traffic (GB)</td><td>80%</td><td>43.09</td><td>75.69</td><td>184.04</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>60%</td><td>14.14</td><td>23.68</td><td>31.29</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>40%</td><td>8.75</td><td>15.05</td><td>15.98</td><td>53.32</td><td></td><td></td><td>58.17</td><td></td><td></td></tr><tr><td rowspan="3">Model update (times)</td><td>80%</td><td>6400</td><td>9600</td><td>20300</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>60%</td><td>2100</td><td>3000</td><td>3500</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>40%</td><td>1300</td><td>1900</td><td>1800</td><td>33000</td><td></td><td></td><td>36000</td><td></td><td></td></tr></table>

Our future research agenda is oriented towards adapting the proposed method to more realistic scenarios. Specifically, we aim to establish methods for sharing label distributions in scenarios where data at each node increases during training, and techniques to ensure that all nodes contribute sufficiently to the training process.

## REFERENCES

[1] H. Ren, H. Li, Y. Dai, K. Yang, and X. Lin, “Querying in internet of things with privacy preserving: Challenges, solutions and opportunities,” IEEE Netw., vol. 32, no. 6, pp. 144–151, Mar. 2018.

[2] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Proc. Artif. Intell. Statist., Florida, USA, Apr. 2017, pp. 1273– 1282.

[3] T. Li, A. K. Sahu, A. Talwalkar, and V. Smith, “Federated learning: Challenges, methods, and future directions,” IEEE Signal Process. Mag., vol. 37, no. 3, pp. 50–60, 2020.

[4] P. Kairouz, H. B. McMahan, B. Avent, A. Bellet, M. Bennis, A. N. Bhagoji, K. Bonawitz, Z. Charles, G. Cormode, R. Cummings et al., “Advances and open problems in federated learning,” Foundations Trends® Machine Learn., vol. 14, no. 1-2, pp. 1–210, 2021.

[5] S. Boyd, A. Ghosh, B. Prabhakar, and D. Shah, “Randomized gossip algorithms,” IEEE Trans. Inf. Theory, vol. 52, no. 6, pp. 2508–2530, 2006.

[6] A. D. Dimakis, A. D. Sarwate, and M. J. Wainwright, “Geographic gossip: Efficient averaging for sensor networks,” IEEE Trans. Signal Process., vol. 56, no. 3, pp. 1205–1216, 2008.

[7] A. Khosravi and Y. S. Kavian, “Broadcast gossip ratio consensus: Asynchronous distributed averaging in strongly connected networks,” IEEE Trans. Signal Process., vol. 65, no. 1, pp. 119–129, 2016.

[8] M. Blot, D. Picard, M. Cord, and N. Thome, “Gossip training for deep learning,” arXiv:1611.09726, 2016.

[9] P. H. Jin, Q. Yuan, F. N. Iandola, and K. Keutzer, “How to scale distributed deep learning?” ArXiv, vol. 1611.04581, 2016.

[10] R. Ormandi, I. Heged´ us, and M. Jelasity, “Gossip learning with linear˝ models on fully distributed data,” Concurrency Comput.: Pract. Experience, vol. 25, no. 4, pp. 556–571, 2013.

[11] A. Koloskova, S. Stich, and M. Jaggi, “Decentralized stochastic optimization and gossip algorithms with compressed communication,” in Proc. Int. Conf. Machine Learn., 2019, pp. 3478–3487.

[12] A. Koloskova, N. Loizou, S. Boreiri, M. Jaggi, and S. Stich, “A unified theory of decentralized sgd with changing topology and local updates,” in Proc. Int. Conf. Machine Learn., 2020, pp. 5381–5393.

[13] I. Hegedus, G. Danner, and M. Jelasity, “Gossip learning as a decen-˝ tralized alternative to federated learning,” in Distrib. Appl. Interoperable Syst., 2019, pp. 74–90.

[14] A. G. Roy, S. Siddiqui, S. Polsterl, N. Navab, and C. Wachinger,¨ “Braintorrent: A peer-to-peer environment for decentralized federated learning,” arXiv:1905.06731, 2019.

[15] S. Savazzi, M. Nicoli, V. Rampa, and S. Kianoush, “Federated learning with mutually cooperating devices: A consensus approach towards server-less model optimization,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process., 2020, pp. 3937–3941.

[16] C. Hu, J. Jiang, and Z. Wang, “Decentralized federated learning: A segmented gossip approach,” arXiv:1908.07782, 2019.

[17] Z. Jiang, A. Balu, C. Hegde, and S. Sarkar, “Collaborative deep learning in fixed topology networks,” Advances Neural Inf. Process. Syst., vol. 30, 2017.

[18] X. Lian, C. Zhang, H. Zhang, C.-J. Hsieh, W. Zhang, and J. Liu, “Can decentralized algorithms outperform centralized algorithms? a case study for decentralized parallel stochastic gradient descent,” Advances Neural Inf. Process. Syst., vol. 30, 2017.

[19] H. Tang, X. Lian, M. Yan, C. Zhang, and J. Liu, “D2: Decentralized training over decentralized data,” in Proc. Int. Conf. Machine Learn., 2018.

[20] H. Zhu, J. Xu, S. Liu, and Y. Jin, “Federated learning on non-iid data: A survey,” Neurocomputing, vol. 465, pp. 371–390, Nov. 2021.

[21] K. Niwa, N. Harada, G. Zhang, and W. B. Kleijn, “Edge-consensus learning: Deep learning on p2p networks with nonhomogeneous data,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, 2020, pp. 668–678.

[22] C. He, M. Annavaram, and S. Avestimehr, “Group knowledge transfer: Federated learning of large cnns at the edge,” Advances Neural Inf. Process. Syst., vol. 33, pp. 14 068–14 080, 2020.

[23] Y. Esfandiari, S. Y. Tan, Z. Jiang, A. Balu, E. Herron, C. Hegde, and S. Sarkar, “Cross-gradient aggregation for decentralized learning from non-iid data,” in Proc. Int. Conf. Machine Learn., 2021, pp. 3036–3046.

[24] S. A. Aketi and K. Roy, “Cross-feature contrastive loss for decentralized deep learning on heterogeneous data,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vision, 2024, pp. 12–21.

[25] N. Onoszko, G. Karlsson, O. Mogren, and E. L. Zec, “Decentralized federated learning of deep neural networks on non-iid data,” arXiv:2107.08517, 2021.

[26] A. Bellet, A.-M. Kermarrec, and E. Lavoie, “D-cliques: Compensating for data heterogeneity with topology in decentralized federated learning,” in 2022 41st Int. Symp. Reliable Distrib. Syst., 2022, pp. 1–11.

[27] Y. Liao, Y. Xu, H. Xu, L. Wang, and C. Qian, “Adaptive configuration for heterogeneous participants in decentralized federated learning,” in Proc. IEEE Int. Conf. Comput. Commun., 2023, pp. 1–10.

[28] A. Yamazaki, T. Nishio, and Y. Hara-Azumi, “Convergence improvement by parameters exchange in asynchronous decentralized federated learning for non-iid data,” in Proc. Euromicro Conf. Softw. Eng. Adv., 2023, pp. 36–43.

[29] K. Maejima, T. Nishio, A. Yamazaki, and Y. Hara–Azumi, “Tram-fl: Routing-based model training for decentralized federated learning,” in Proc. IEEE Consum. Commun. Netw. Conf., 2024, pp. 1038–1039.

[30] T. W. Sherson, R. Heusdens, and W. B. Kleijn, “Derivation and analysis of the primal-dual method of multipliers based on monotone operator theory,” IEEE Trans. Signal Inf. Process. Netw., vol. 5, no. 2, pp. 334– 347, 2018.

[31] G. Zhang and R. Heusdens, “Distributed optimization using the primaldual method of multipliers,” IEEE Trans. Signal Inf. Process. Netw., vol. 4, no. 1, pp. 173–187, 2017.

[32] I. Kholod, E. Yanaki, D. Fomichev, E. Shalugin, E. Novikova, E. Filippov, and M. Nordlund, “Open-source federated learning frameworks for iot: A comparative review and analysis,” Sensors, vol. 21, no. 1, p. 167, Dec. 2020.

[33] M. Duan, D. Liu, X. Chen, Y. Tan, J. Ren, L. Qiao, and L. Liang, “Astraea: Self-balancing federated learning for improving classification accuracy of mobile deep learning applications,” in Proc. IEEE Int. Conf. Comput. Des., Abu Dhabi, UAE, Nov. 2019, pp. 246–254.

[34] J. Luo and S. Wu, “Fedsld: Federated learning with shared label distribution for medical image classification,” in 2022 IEEE 19th Int. Symp. Biomed. Imag., 2022, pp. 1–5.

[35] L. Zhu, Z. Liu, and S. Han, “Deep leakage from gradients,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019.

[36] S. P. Karimireddy, S. Kale, M. Mohri, S. Reddi, S. Stich, and A. T. Suresh, “Scaffold: Stochastic controlled averaging for federated learning,” in Proc. Int. Conf. Machine Learn., 2020, pp. 5132–5143.

[37] Y. Li and X. Lyu, “Convergence analysis of sequential federated learning on heterogeneous data,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023.

[38] R. Kainuma and T. Nishio, “Load-aware training scheduling for model circulation-based decentralized federated learning,” arXiv:2506.09769, 2025.

[39] A. Balu, Z. Jiang, S. Y. Tan, C. Hedge, Y. M. Lee, and S. Sarkar, “Decentralized deep learning using momentum-accelerated consensus,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process., 2021, pp. 3675–3679.

[40] H. Yu, R. Jin, and S. Yang, “On the linear speedup analysis of communication efficient momentum sgd for distributed non-convex optimization,” in Proc. Int. Conf. Machine Learn., 2019, pp. 7184–7193.

[41] H. Wu, P. Judd, X. Zhang, M. Isaev, and P. Micikevicius, “Integer quantization for deep learning inference: Principles and empirical evaluation,” ArXiv, vol. 2004.09602, 2020.

[42] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proc. IEEE, vol. 86, no. 11, pp. 2278– 2324, Nov. 1998.

[43] A. Krizhevsky, G. Hinton et al., “Learning multiple layers of features from tiny images,” 2009.

[44] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in Proc. Int. Conf. Machine learn., 2015, pp. 448–456.

[45] Y. Wu and K. He, “Group normalization,” in Proc. Eur. Conf. Comput. vision, 2018, pp. 3–19.

[46] J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer normalization,” arXiv:1607.06450, 2016.

[47] D. Ulyanov, A. Vedaldi, and V. Lempitsky, “Instance normalization: The missing ingredient for fast stylization,” arXiv:1607.08022, 2016.

[48] K. He, X. Zhang, S. Ren, and J. Sun, “Delving deep into rectifiers: Surpassing human-level performance on imagenet classification,” in Proc. IEEE Int. Conf. Comput. Vision, Santiago, Chile, Dec. 2015, pp. 1026–1034.