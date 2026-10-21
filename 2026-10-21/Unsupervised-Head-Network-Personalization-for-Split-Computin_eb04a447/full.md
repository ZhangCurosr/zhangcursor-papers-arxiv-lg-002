# Unsupervised Head Network Personalization for Split Computing via Federated Learning

Yutaro Horikawa and Takayuki Nishio

Abstract—Split Computing (SC) is a distributed inference method designed for balancing computational load and reducing latency by partitioning a neural network into head and tail networks and deploying them across both a mobile device and a server, respectively. Due to the limited bandwidth and computational power of mobile devices, the head network is often replaced with a neural network that requires lower computational and communication resources. However, it is challenging to prepare a sufficient quantity and quality of labeled data on the mobile device side for training the replacement head network. Furthermore, the data prepared by mobile users often contains sensitive privacy information, making data aggregation difficult. In this paper, we propose a data-efficient head network training method for SC. This method utilizes federated learning and knowledge distillation to train the head network using a small amount of non-IID and unlabeled data from mobile devices, ensuring that the training datasets are not exposed. Moreover, by introducing personalization, we enhance the accuracy to levels comparable to or exceeding those of existing methods. The experimental evaluations demonstrate that, despite data constraints imposed only on our method, it achieves accuracy comparable to the baselines. Furthermore, it significantly reduces the communication requirements by 95% on mobile devices, with only a 3-5 percentage point drop in accuracy from the original model.

Index Terms—Distributed inference, split computing, knowledge distillation, non-IID data, federated learning

## I. INTRODUCTION

Deep learning and deep neural networks (DNNs) are critical in a wide range of applications [1]. Furthermore, advancements in foundational models have enabled sophisticated tasks such as high-quality image generation [2], machine translation [3], and robotic control [4]. However, these advanced DNNs demand significant computational and memory resources, which challenges local computing. At the same time, the growing necessity to localize complex tasks such as speech recognition, personal assistants, and healthcare support [5] on mobile and edge devices for latency and privacy reasons makes it essential to adapt DNNs for these platforms. Yet, the inherently limited computational and memory capacities of mobile devices pose challenges for their real-time operation [6].

Split computing (SC) has been proposed as an intermediate solution between cloud/edge server execution and local execution [7]. SC enables local devices to offload heavy computations of the DNN models to the cloud or edge servers. SC involves splitting the DNN into two parts: the first half processes on the device, which then sends the output of the middle layer to the server, and the server executes the remaining half. The first half of the split model is referred to as the Head Network, and the second half as the Tail Network. SC aims to reduce overall inference time and offers the added benefit of enhancing data privacy, as raw data is not transmitted to the server [8].

In SC, it is often necessary to train a suitable dedicated model based on the constraints on computational performance and communication bandwidth of the local device, and replace it as the Head Network [9]. This is because general DNNs frequently map features to high-dimensional spaces near the input layer, resulting in very large intermediate representations. Using these as the Head Network leads to high memory consumption and communication overhead. Therefore, in SC, a layer with a smaller dimensionality for intermediate outputs, known as a bottleneck layer, is often used. By training a neural network with such a bottleneck layer, where the number of units and filters is limited, and replacing the original Head Network, the computational load and communication volume on the local device can be significantly reduced.

Various methods for training Head Networks have been explored [10]–[15]. However, these methods often assume the availability of sufficiently large datasets, particularly labeled training sets for supervised learning. In models using data related to images, audio, or healthcare, training data is not always accessible. Consequently, while local devices could potentially train their dedicated Head Networks using their own datasets, these datasets are often unlabeled, small in volume, and biased in distribution, making it difficult to train high-accuracy models.

To address the challenges arising from dataset constraints on local devices, this study proposes a new training method for SC which incorporates federated learning (FL) [16] and head network distillation [10]. This method enables cooperative training of the head network across multiple local devices using a small amount of unlabeled and biased data from mobile devices, ensuring that the training datasets are not exposed. FL is a distributed model training framework that allows learning while keeping private data local. Additionally, the proposed method employs head network distillation, a knowledge distillation-based technique that can train on unlabeled data. By combining FL with personalization, it can train under severe data constraints without significantly degrading model performance. This approach is capable of training smaller models that decrease the overall inference time on SC, with only a minimal loss in accuracy, while preserving privacy under the stringent data constraints of the involved devices. The contributions of this paper are summarized as follows:

A head network training method using small, unlabeled, and biased local datasets while preserving data privacy: This contribution outlines the development of a novel training method for the head network in SC that utilizes federated learning and head network distillation. The method ensures data privacy by keeping training datasets local and addresses the challenge of using small, unlabeled, and biased data from mobile devices.

Performance evaluation using benchmark datasets: We conduct a thorough performance evaluation of the proposed method using CIFAR-10 and CIFAR-100 under a non-IID setting, comparing it with baseline methods. The method reduces communication traffic by 95% on local devices while maintaining accuracy within 3-5 percentage points of the original model. The personalization we introduce significantly enhances accuracy, achieving results comparable to the original model despite stringent data constraints.

The rest of the paper is organized as follows. We discuss related works in Section II. Section III introduces the building blocks of our method: split computing, knowledge distillation, and federated learning. Section IV describes the proposed training method. Section V provides the experimental evaluation. Finally, Section VI concludes this paper.

## II. RELATED WORKS

This section reviews split computing and existing approaches to training models suitable for split computing, and clarifies the position of our contribution.

## A. Split Computing

SC is a technique that divides the DNN into two parts and distributes the inference process between mobile devices and servers [7]. A trained model is divided into a head network deployed on a mobile client and a tail network deployed on a server; the client transmits the intermediate output of the head network to the server, which completes the inference. The formal notation of this procedure, used throughout this paper, is introduced in Section III-A.

SC can reduce computational latency by distributing the neural network’s processing between the local device and the server. However, care must be taken regarding the communication latency associated with transmitting the intermediate output. Many neural networks map input features to a higherdimensional space in layers close to the input layer. Thus, simply splitting an existing model can increase communication latency, potentially raising the total inference latency. Therefore, many existing studies propose replacing the original model with models designed to be more suitable for split computing. Specifically, these models are designed to map to a low-dimensional space, referred to as a bottleneck, at the split point [9], [10].

## B. Training Methods for Split Computing Models

The methods for efficiently training models suitable for split computing are a critical issue in this field, and various approaches have been proposed. Two particularly important challenges are as follows:

Designing model architectures appropriate for split computing: The first challenge is optimizing the model architecture, especially the head network and the split point, considering the limitations of the mobile device’s computational power and communication bandwidth. Neural network architectures have nearly infinite combinations of layers, types, units, and filters, making it essential to determine an optimal architecture. This task is part of the domain known as Neural Architecture Search (NAS). Traditional NAS focuses on metrics such as accuracy, computational cost, and latency, with methods aimed at maximizing accuracy or improving the accuracy-computation trade-off while minimizing computation for searching the architecture and training the model parameters [20], [21]. However, for split computing, it is crucial to also consider communication cost and latency.

Kang et al. focus on identifying optimal split points that balance computation latency between the server and local devices, as well as communication latency. They conducted a comprehensive investigation on well-known benchmark models [7]. Drawing from these insights, they design a protocol to optimize the split point. Bakhtiarnia et al. propose dynamic split computing, which adjusts the splitting point based on varying communication bandwidth to minimize total latency [22]. These studies do not alter the structure of pre-trained models but only adjust the split points.

To improve the trade-off between accuracy and latency, particularly by reducing communication traffic, a widely used approach is to insert a layer with a bottleneck structure, which has a lower dimensionality than the original structure, at the split point. This approach is akin to using an incomplete autoencoder [13]–[15]. However, in many studies, the inserted bottleneck structures are designed manually or empirically. On the other hand, methods based on the NAS framework have also been proposed to explore structures suitable for split computing [18], [19]. By incorporating model accuracy and total latency, including both computation and communication, into the objective function or constraints of NAS, it is possible to search for the optimal architecture from among the candidates. In particular, our previous research [18] propose a method to simultaneously search for model structures and parameters using the Adaptive Stochastic Natural Gradient method. However, these studies require substantial computational resources and training data for structure search, model parameter training, and fine-tuning.

Efficient Training Methods for Modified Models: Efficiently training the parameters of models updated to have structures suitable for split computing is a significant practical challenge. In environments where split computing is applied, access to abundant computational resources and training data is not always guaranteed. Therefore, it is essential to train models that achieve accuracy comparable to the original models with minimal computational effort, memory consumption, and training data.

TABLE I: Summary of related works on head network training.  
A: model accuracy, C: computation load, D: amount of data transmitted, L: latency, CE: cross-entropy, KD: knowledge distillation, reconst.: reconstruction loss
<table><tr><td colspan="4"></td><td colspan="3">Data constraints</td></tr><tr><td>Research</td><td>Metrics</td><td>Loss</td><td>Unlabeled</td><td>Decentralized</td><td></td><td>Non-IID</td></tr><tr><td>[9]</td><td>A, D, L</td><td>CE</td><td></td><td></td><td></td><td></td></tr><tr><td>[17]</td><td> $\mathrm { A } , \mathrm { D } , \mathrm { L }$ </td><td>CE</td><td></td><td>一</td><td></td><td></td></tr><tr><td>[11]</td><td> $\mathbf { A } , \mathbf { C } , \mathbf { D } , \mathbf { L }$ </td><td></td><td>KD</td><td>√</td><td></td><td></td></tr><tr><td>[12]</td><td> $\mathbf { A } , \mathbf { C } , \mathbf { D } , \mathbf { L }$ </td><td>KD</td><td></td><td>√</td><td>一</td><td></td></tr><tr><td>[13]</td><td>A, D</td><td>reconst.</td><td></td><td>√</td><td></td><td></td></tr><tr><td>[14]</td><td> $\mathbf { A } , \mathbf { C } , \mathbf { D }$ </td><td> $\mathrm { r e c o n s t . + K D }$ </td><td></td><td>√</td><td>一</td><td></td></tr><tr><td>[15]</td><td> $\mathrm { A } , \mathrm { D } , \mathrm { L }$ </td><td> $\mathrm { r e c o n s t . + K D }$ </td><td></td><td></td><td>一</td><td></td></tr><tr><td>[18]</td><td> $\mathrm { A } , \mathrm { L }$ </td><td> $\mathrm { C E } + \mathrm { L a t e n c y }$ </td><td>loss</td><td></td><td>一</td><td></td></tr><tr><td>[19]</td><td> $\mathrm { A } , \mathrm { L }$ </td><td> $\mathrm { C E } + \mathrm { L a t e n c y }$ </td><td>loss</td><td>一</td><td></td><td></td></tr><tr><td>Ours</td><td> $\mathbf { A } , \mathbf { C } , \mathbf { D }$ </td><td>KD</td><td></td><td>√</td><td>√</td><td>√</td></tr></table>

When ample labeled training data and computational resources are available, the most straightforward training method is to perform end-to-end training of the entire model using backpropagation based on loss functions such as cross-entropy (CE), similar to standard model training practices [9], [17]. However, sufficient labeled data is not always accessible. Particularly when leveraging publicly available open models, the data used for their training may not be available.

In cases where a bottleneck layer resembling an incomplete autoencoder is inserted, training methods that utilize reconstruction errors have been proposed. These methods do not require labeled data for training [13]–[15]. By training the newly inserted autoencoder section to match its input and output using unlabeled data, the signal changes before and after the bottleneck are minimized, maintaining inference accuracy. However, this approach is limited to autoencoders, and adding new layers increases total computation. Thus, it is challenging to apply this method when reducing computational load on the local device or server is prioritized, even if it means sacrificing some accuracy.

To address this, a training method using knowledge distillation (KD), so called Head network distillation (HND), has been proposed [11], [12]. KD [23] is introduced to efficiently train smaller models (called student models) to achieve comparable accuracy to larger pre-trained models (called teacher models). Matsubara et al. adapt KD for training models in split computing, particularly for the head network. They propose a method that efficiently trains only the head network (specifically, the head network and decoder part) using a portion of the original model [11]. This method achieves higher accuracy using only unlabeled data compared to end-to-end training with cross-entropy loss on labeled datasets and conventional KD-based training that involves entire models. However, it still requires a substantial amount of centralized unlabeled dataset.

The integration of FL and KD itself has been explored since the early days of FL. Federated distillation, pioneered by Jeong et al. [24], exchanges model outputs (logits) instead of model parameters to drastically reduce the training-time communication of FL under non-IID data, and Itahara et al. [25] extended this line to semi-supervised settings in which an unlabeled shared dataset is exploited through distillation for communication-efficient collaborative training. Subsequent works follow this direction: FedKD [26] mutually distills knowledge between a local teacher and a globally shared student model, and Li et al. [27] combine KD with transfer learning for communication-efficient FL in UAV networks. These studies employ KD to reduce the training-time communication of FL itself, and their goal is to obtain a full inference model on each client. In contrast, our method employs distillation for a fundamentally different purpose: it transfers the intermediate representations of a pre-trained model to a lightweight head network so that the inferencetime communication and computation of SC are reduced, and the distillation target itself eliminates the need for labels on the client side. Recent studies on learned feature compression for SC, such as FrankenSplit [28], also reduce inferencetime communication, but assume that training data can be centralized. To the best of our knowledge, no prior work trains SC head networks under the combination of unlabeled, decentralized, and non-IID data constraints, as summarized in Table I.

Inspired by HND, this study proposes an efficient head network training method under data constraints, particularly when the data is unlabeled and distributed across local devices, with each device having access to a limited amount of data. Table I summarizes existing research and our work, highlighting the differences among them. The proposed method overcomes data constraint challenges by integrating federated learning and HND. Additionally, the method mitigates accuracy degradation caused by non-IID data introduced by federated learning through a model personalization mechanism. This approach achieves accuracy comparable to original and centrally trained models, even when the data distribution varies significantly across local devices.

## III. PRELIMINARIES

This section introduces the building blocks of the proposed method: the formulation of split computing, knowledge distillation and head network distillation, and a brief overview of federated learning.

![](images/6162a3f1e93b458e1a9fba882a092dd9a3eac52e4caecfe055e06504959beed2.jpg)  
(a) Teacher and student models

![](images/ec1ae32757528ebe28d42704df83b848bd239f56f68c0832e0267bbe3ece384b.jpg)  
(b) Training procedure  
Fig. 1: Head Network Distillation [11]: (a) teacher and student models extracted from the original pre-trained model, and (b) the training procedure minimizing the error between the intermediate outputs.

## A. Split Computing Formulation

We consider a neural network model $f ( x ; w )$ parameterized by w. For example, in the case of a 3-layer neural network, the model can be represented as

$$
f ( x ; w ) = l _ { 3 } ( l _ { 2 } ( l _ { 1 } ( x ) ) ) ,\tag{1}
$$

where x is the input and $l _ { i }$ is a mapping of the i-th layer parameterized by $w _ { i }$ . In this manner, the computation of a neural network can be regarded as an operation where the input sequentially passes through multiple functions. Thus, we can divide the model into two composite functions, such as $l _ { 1 }$ and $l _ { 3 } \circ l _ { 2 }$ . The first part is called the head network, $f _ { \mathrm { h e a d } } = l _ { 1 }$ and is deployed to a mobile client, while the second part is called the tail network, $f _ { \mathrm { t a i l } } = l _ { 3 } \circ l _ { 2 } .$ , and is deployed to a server.

In the distributed inference procedure, the mobile client feeds input x and obtains the intermediate output of the head network,

$$
z = f _ { \mathrm { h e a d } } ( x ) = l _ { 1 } ( x ) .\tag{2}
$$

The intermediate output z is transferred to the server and used as the input for the tail network, resulting in the inference output

$$
y = f _ { \mathrm { t a i l } } ( z ) = l _ { 3 } ( l _ { 2 } ( z ) ) .\tag{3}
$$

## B. Knowledge Distillation and Head Network Distillation

HND [11] is a training method for the head network based on knowledge distillation [23]. Knowledge distillation leverages a well-trained large model, called the teacher model, to train an untrained small model, called the student model. In knowledge distillation, the student model is trained using the output of the teacher model, referred to as a soft target, as well as the labels of data samples, referred to as hard targets. This allows the student model to achieve accuracy comparable to the teacher model while maintaining a smaller model size. HND applies knowledge distillation to the intermediate outputs of the model, enabling the training of only a part of the model using unlabeled data.

Fig. 1a shows an example of HND’s teacher and student models extracted from the original pre-trained model. Unlike conventional knowledge distillation, which uses the entire model, HND uses only several layers from the input side of the model as the teacher model. The number of layers used is a hyperparameter. A student model, with the same input and output layers as the teacher model but with fewer parameters and a bottleneck structure, is prepared. This bottleneck part, indicated by the dashed line in Fig. 1a, divides the model. The input-side portion is used as the head network, while the remaining part, combined with the rest of the original model excluding the teacher model, forms the tail network used in split computing. In HND, the student model is trained to minimize the errors between the outputs of the teacher and student models, ensuring consistency, as shown in Fig. 1b. Specifically, the loss function L is defined as follows:

$$
L = \sum _ { x \in X } \| f _ { \mathrm { T } } ( x ; w _ { \mathrm { T } } ) - f _ { \mathrm { S } } ( x ; w _ { \mathrm { S } } ) \| ^ { 2 } ,\tag{4}
$$

where X is a set of training data, x is a data sample, and $f _ { \mathrm { T } } ( x ; w _ { \mathrm { T } } )$ and $f _ { \mathrm { S } } ( x ; w _ { \mathrm { S } } )$ are the outputs of the teacher model parameterized by w<sub>T</sub> and student model parameterized by w<sub>S</sub>, respectively, with x as input. By training the student model, its output becomes identical or very close to the output of the teacher model. Consequently, the processing results in the subsequent layers also yield the same outcomes as when using the teacher model. This ensures that the inference results remain consistent whether the original model is used or the teacher model is replaced with the student model.

## C. Federated Learning

FL [16] is a distributed training framework in which multiple clients collaboratively train a shared model while keeping their data local. In the representative FedAvg algorithm [16], a server distributes a global model to the clients; each selected client updates the model locally on its own dataset for several epochs; and the server aggregates the returned models, typically by weight averaging, to update the global model. This procedure is iterated over multiple rounds until convergence. Since only model parameters are exchanged, FL preserves the privacy of the clients’ raw data. A well-known challenge of FL is the accuracy degradation under non-IID client data, which has motivated a large body of work on personalized federated learning [29]–[31].

## IV. FEDERATED HEAD NETWORK PERSONALIZATION

This section presents the proposed method, FedHNP. We first describe the system model and assumptions, then provide

![](images/08cb822a79b683653a3c9e2daad41935c3cf9e380180d6e2b87cfb7905b51406.jpg)  
Fig. 2: System model

an overview and the detailed two-stage procedure, and finally illustrate the model architecture with a concrete example.

## A. System Model

Fig. 2 shows the system model. The system consists of a server and N edge devices such as Raspberry Pi and Jetson Nanos. Following the terminology of federated learning, these edge devices are referred to as clients in this paper.

First, we outline the assumptions regarding communication. Each client communicates only with the server during training with FL and distributed inference with SC, but not directly with other clients. There are bandwidth limitations on communication between the client and the server, and it is necessary to reduce the amount of data sent to the server during machine learning inferences to minimize execution time. On the other hand, we assume that model training is performed independently of inference processing, during times with available bandwidth, such as overnight, making the reduction of communication volume during training outside the scope of this study.

Next, we outline the assumptions regarding computational capability. The clients lack the computational capability to efficiently process the server’s high-performance pre-trained models within the desired latency for real-time tasks. In contrast, these devices can swiftly handle smaller models. Additionally, the urgency for rapid processing is less critical during the training phase than during inference, as real-time performance is not required for model training, making speed less of a priority.

Then, we outline the assumptions regarding training data. Each client collects unlabeled data samples, which are used for training and inference. These clients solve common tasks, such as object detection or anomaly detection, but the data sample distributions observed in each client’s environment differ, resulting in a non-IID setting when the clients’ data is aggregated. Additionally, the amount of data available to each client is limited, making it difficult to train a high-accuracy model independently, and this data cannot be collected or aggregated on the server due to privacy concerns.

Finally, we outline the assumptions regarding the machine learning model. The server has a pre-trained model, assumed to be well-trained with labeled data, but the labeled training dataset is unavailable. This is not an unusual assumption, as it often occurs when using publicly available models. For this study, we focus on the training process, assuming that the structures of the original model and the head and tail networks are predetermined and known to the server and each client. The head network is designed to be small enough to be trained and processed by the client, while the optimization of the model’s structure is outside the scope of this study.

Algorithm 1 Federated Head Network Personalization   
Stage 1. FL-based global model training   
1. Initialization   
Obtain teacher model from pre-trained model,   
and distribute it to each client   
Initialize the parameter of all client models $w _ { k }$ and   
the global model $w _ { \mathrm { g } }$   
2. Local update   
Select C clients out of N   
for Each selected client k in parallel do   
for $e = 1 , \ldots , E _ { \mathrm { F L } }$ do   
Update each model $w _ { k }$ locally   
$w _ { k } \gets w _ { k } - \eta \nabla L _ { k }$   
end for   
end for   
3. Aggregation   
C clients upload their models $\hat { w } _ { k }$ to the server   
Aggregate the models on the server and update   
the global model   
$w _ { g } \gets \frac { 1 } { C } \sum _ { i \in I _ { C } } \hat { w } _ { i }$   
4. Distribution   
Distribute the global model to all clients

Stage 2. Client personalization   
for Each client k in parallel do   
for $e = 1 , \ldots , E _ { \mathrm { p } }$ do   
Update each model $w _ { k }$ locally   
$w _ { k } \gets w _ { k } - \eta \nabla L _ { k }$   
end for   
end for

## B. Overview of Proposed Method

This section provides an overview of our proposed FedHNP method, as shown in Fig. 3. FedHNP enables the training of models more suitable for SC, with reduced computation and communication costs, while maintaining the accuracy of the original model. The algorithm of the proposed FedHNP is shown in Algorithm 1. The training method is divided into two stages: FL-based global model training and client personalization.

The FL-based global model training stage aims to train a base common model, referred to as the global model, using the FL procedure. By introducing FL, each client can contribute to training the global model without aggregating data, thus preserving the privacy of the clients’ data, even when each client has only a small amount of data.

![](images/82582ddf5e229bca2153697f2c040425d90193ac509f2fffa5e406b39baef3fe.jpg)  
Fig. 3: Proposed method: Federated Head Network Personalization

The training flow follows the well-known FedAvg algorithm [16]. In the original FedAvg, an initialized global model is shared with all the clients, who then train the model locally using conventional gradient descent methods with output loss. The locally trained models, called client models, are then gathered and aggregated to update the global model. In contrast, in the proposed FedHNP, to enable training with unlabeled data, the local training on the client side is replaced with Head Network Distillation (HND) [11]. The details of HND are described in Section III-B. HND allows for unsupervised training using unlabeled data by ensuring that the intermediate outputs of the trained model and the target model are consistent.

The second stage is the personalization of the well-trained models to fit the clients’ local non-IID data. In this paper, we employ a fine-tuning method for personalization, which is simple and straightforward yet effective for many tasks [29]. This personalization can significantly improve the performance of client models against each client’s local data.

## C. Details of FedHNP

This section describes our proposed FedHNP in detail. FedHNP integrates distillation-based model training based on HND and cooperative training based on FL and incorporates a personalization method. This is the key difference between FedHNP and original HND. As briefly explained in the previous section, FedHNP consists of the following two stages.

1) FL-based Global Model Training: A global model is trained using the distributed clients’ local dataset based on the FL algorithm. Algorithm 1 describes the detailed procedures. 1. Initialization. First, the server determines which layers of the original model will constitute the teacher model and extracts the teacher model from the pre-trained original model for HND. The server also determines the architecture of a student model considering the computation and communication capabilities of the local devices, initializes the parameters of the student models $w _ { k }$ (where $k = 1 , 2 , \ldots , N$ , indicating the identification of a client), and distributes the teacher model and a student model to each client. The student models, which are stored and updated by the clients, are referred to as client models. The architectures of the teacher and student models are hyperparameters. Although the methods to optimize them are beyond the scope of this paper, we discuss the model architecture design with specific examples in Section IV-C4.

## 2. Local update.

From the N clients, C clients are randomly selected. These selected clients update their client models and upload them to the server, thereby reducing communication traffic and preventing congestion in the server networks.

Each selected client k trains its client model $w _ { k }$ by HND’s algorithm using only its dataset $X _ { k }$ . Redefining the loss function $L _ { k }$ for client k using eq. (4), we obtain

$$
L _ { k } = \sum _ { x \in X _ { k } } \| f _ { \mathrm { T } } ( x ; w _ { \mathrm { T } } ) - f _ { \mathrm { S } } ( x ; w _ { k } ) \| ^ { 2 } ,\tag{5}
$$

and the model parameter is updated via the stochastic gradient descent method as follows:

$$
w _ { k } \gets w _ { k } - \eta \nabla L _ { k } ,\tag{6}
$$

where $\eta$ is the learning rate. Clients update their client models for a predetermined number of epochs $E _ { \mathrm { F I } }$ . The updated client models $\hat { w } _ { k }$ are sent to the server.

3. Aggregation. The server aggregates the updated client models to create an updated global model with the parameter $w _ { \mathrm { g } } .$ Although there are various methods for aggregation, this paper employs a simple arithmetic mean, which is widely used in many FL algorithms, as represented below:

$$
w _ { g } \gets \frac { 1 } { C } \sum _ { i \in I _ { C } } \hat { w } _ { i } ,\tag{7}
$$

where $I _ { C }$ is the index set of the selected clients.

4. Distribution. The global model $w _ { \mathrm { g } }$ is distributed to all clients. Clients then replace their client models with $w _ { k }$ as follows:

$$
w _ { k }  w _ { g } .\tag{8}
$$

The procedures from ”2. local update” to ”4. distribution” are repeated for a predefined number of iterations R or until certain training and validation losses are achieved.

2) Client Personalization: The clients update their client models to better fit their local data distribution, thereby improving model accuracy for their specific data domains. The model personalization mechanism in federated learning, known as Personalized Federated Learning (PFL), has been widely studied to address the accuracy degradation caused by non-IID data [29]. Representative approaches include metalearning-based methods that optimize the global model as an initialization for fast local adaptation [30] and hypernetworkbased methods that generate client-specific parameters [31]. This paper adopts local fine-tuning for three reasons. First, a recent comprehensive benchmark reports that fine-tuning consistently yields stable accuracy across diverse tasks and often matches or outperforms more sophisticated PFL methods [29]. Second, fine-tuning is executed entirely on the client and requires no additional communication or serverside coordination, which suits the bandwidth-limited system model in Section IV-A. Third, unlike meta-learning or hypernetwork approaches, it introduces no additional models or hyperparameters to be maintained on resource-constrained clients. Fine-tuning directly addresses the non-IID challenge because the global model trained in Stage 1 provides a generalized initialization, while the subsequent local updates specialize the model to each client’s own data distribution, which is consistent with the distribution the client encounters at inference time.

For personalization using fine-tuning, each client trains its client model, which is the well-trained global model, using only its own data through the same local update method as in Step 1. However, unlike during FL-based global model training, it is necessary to sufficiently fit the client model to the client’s local data. Therefore, the number of fine-tuning epochs $E _ { p }$ is set to be greater than $E _ { \mathrm { F L } }$

It is worth noting the overhead of the personalization stage. Since fine-tuning is performed locally, it incurs no communication cost; the total training-time communication of FedHNP is determined solely by the FL stage (per round, C clientmodel uploads and the distribution of the global model, where the transferred student model is far smaller than the original model). The computational overhead of personalization is $E _ { p }$ local epochs (10 for CIFAR-10 and 20 for CIFAR-100 in our experiments), which is marginal compared with the epochs consumed over the R rounds of the FL stage. As shown later in Fig. 7 (Section V), most of the accuracy gain is obtained within the first few epochs, indicating that the accuracy benefit of personalization is obtained at a small and controllable computational cost.

3) Distributed Inference after Training: During the inference phase, the client executes the head network, while the server executes the tail network. Therefore, it is necessary to transfer the remaining portion of the student model, after the head network, from the client to the server. As will be explained in more detail in the next section, this can be understood clearly by referring to the right side of Fig. 4. The portion of the student model (shown in yellow) up to the splitting point, remains as the head network on the client, while the remaining part of the network is transferred to the server. The server combines this transferred network with the blue-colored portion, which was not used during training, to form the tail network.

Since the fine-tuned student model parameters differ for each client, the tail network must be constructed individually for each client. Specifically, for client k, the model is defined

![](images/de99c9418f058df4ee7289f47a9011de9e9406af977c3e7395e659a50708d371.jpg)  
Fig. 4: Architecture of the original and bottleneck injected models. Training uses only the teacher and student models.

as:

$$
f _ { \mathrm { S } } ( \cdot ; w _ { k } ) = f _ { \mathrm { D } } ( f _ { \mathrm { H } } ( \cdot ; w _ { k } ^ { \mathrm { H } } ) ; w _ { k } ^ { \mathrm { D } } ) ,\tag{9}
$$

where $f _ { \mathrm { H } } ( \cdot ; w _ { k } ^ { \mathrm { H } } )$ represents the head network of client k parameterized by $w _ { k } ^ { \mathrm { H } }$ , and $f _ { \mathrm { D } } ( \cdot ; w _ { k } ^ { \mathrm { D } } )$ represents the remaining part of the student model, parameterized by $w _ { k } ^ { \mathrm { D } }$ , which is similar in structure to a decoder. Let x be the input data, $O _ { \mathrm { h e a d } }$ the output of the head model, and $O _ { \mathrm { t a i l } }$ the output of the tail model. These outputs are represented as follows:

$$
o _ { \mathrm { h e a d } } = f _ { \mathrm { H } } ( x ; w _ { k } ^ { \mathrm { h } } ) ,\tag{10}
$$

$$
o _ { \mathrm { t a i l } } = f _ { \mathrm { R } } ( f _ { \mathrm { D } } ( o _ { \mathrm { h e a d } } ; w _ { k } ^ { \mathrm { D } } ) ; w _ { \mathrm { R } } ) ,\tag{11}
$$

where $f _ { \mathrm { R } } ( \cdot ; w _ { \mathrm { R } } )$ is the remainder of the pre-trained model, excluding the teacher model, and is parameterized by the pretrained weights w . In distributed inference, the client first computes $O _ { \mathrm { h e a d } }$ and sends it to the server. The server then computes $o _ { \mathrm { t a i l } }$ and that is the inference result.

4) Model Architecture: Example Study: This section aims to clarify the relationships among the various models and sub-models discussed in this paper—namely, the Original model, Teacher model, Student model, head network, and tail network—using specific examples. Additionally, we will discuss the design guidelines for the student model.

Fig. 4 illustrates the model architecture used in our experiments as an example. We used ResNet-18 [32] as our original model. The left side of the figure presents a pre-trained model hosted on the server, while the right side shows a model that has been newly trained. During the model training phase, the red segment is used as the teacher model $f _ { \mathrm { T } } ( \cdot ; w _ { \mathrm { T } } )$ , and the yellow segment is updated as the student model $f _ { \mathrm { S } } ( x ; w _ { \mathrm { S } } )$ The blue segment is $f _ { \mathrm { R } } ( \cdot ; w _ { \mathrm { R } } )$ , and its parameters w<sub>R</sub>, sourced from the original model, remain unchanged throughout the entire process.

In the inference phase, the trained model on the right side is used. The initial segment of the yellow-colored head network constitutes the head network $f _ { \mathrm { H } } ( \cdot ; w _ { k } ^ { \mathrm { h } } )$ , computed by client k. The subsequent portion is used as the tail network, $l _ { \mathrm { R } } \circ l _ { \mathrm { D } } ,$ , and is processed by the server. Therefore, the output of the third convolution layer of the model on the right side corresponds to $O _ { \mathrm { h e a d } } .$ , and is transmitted via networks. The server receives $O _ { \mathrm { h e a d } }$ from the client and processes it through the tail network $l _ { \mathrm { R } } \circ l _ { \mathrm { D } }$

Although the model shown in Fig. 4 is just an example, we will discuss key considerations when designing the architecture of the student model. Two particularly important aspects are:

• Model size: This refers to the number of model parameters. The computational time, particularly the size of the head network, significantly impacts processing time on the client.

• Bottleneck size: This corresponds to the intermediate tensor size at the split point, i.e., the size of the output tensor from the head network. It greatly affects the amount of data transmitted during inference and, consequently, the communication latency.

While reducing both model size and bottleneck size can decrease computation and communication latency in SC, there is a trade-off with model accuracy. A model that is too small in size or has an excessively reduced bottleneck may significantly degrade inference accuracy, potentially failing to meet performance requirements.

In the model used for this study, we designed a head network with fewer parameters by reducing the filter size while largely preserving the structure of the original model. Specifically, the head network size was set to 5,080 parameters by reducing the number of filters from 64 to 12. This is significantly smaller than the 38,848 parameters that the original model would have had if split at the same point, resulting in a substantial reduction in computation latency on the client.

As for bottleneck size, the smaller it is, the smaller the intermediate outputs sent from the client to the server during each inference, which helps reduce communication latency. A simple strategy is to reduce the number of filters or units corresponding to the output layer of the head network, thus reducing the bottleneck size while keeping the original model’s structure. However, this strategy may not be applicable in certain cases, particularly in deep neural networks (DNNs) that employ skip connections, which add the input of one layer to the output of another. When the split point is set across a skip connection, the total tensor size increases because both the outputs and the input to the skip connection need to be transmitted. Therefore, it is important to avoid splitting at skip connections, though this may limit the available splitting points and make it more difficult to reduce the size of the head network.

To address these challenges, we designed a student model without skip connections and implemented a bottleneck with small filter sizes in the early layers. This design preserves much of the architecture of the original ResNet-18 model, except for modifications around the bottleneck and skip connections.

In our design, the bottleneck dimensionality is selected such that the transmitted tensor, (12, 9, 9), is smaller than the raw input (3, 32, 32), reducing the per-inference traffic to 1.0 KB with 8-bit quantization—one third of cloud offloading and $1 / 8 – 1 / 1 6$ of splitting the original model (Table IV in Section V)—while keeping the client-side computation comparable to the shallowest viable split point of the original model. In general, the bottleneck should be dimensioned as the smallest tensor for which the accuracy degradation remains within the application’s tolerance; exploring this trade-off automatically, e.g., with NAS [18], [19], is orthogonal to our training method.

It is worth reiterating that the student model presented here may not be optimal, as the ideal model configuration depends on factors such as communication speed, the computational resources available on the device, and the specific task being executed. Although some studies [18], [19] have introduced NAS as a strategy for designing models tailored to SC, our study does not focus on optimizing the model architecture. Instead, our focus is on training methods for the head network under data constraints.

## V. EVALUATION

This section evaluates FedHNP on CIFAR-10 and CIFAR-100 under non-IID settings, in terms of computation and communication costs, accuracy against baselines, and the effect of personalization.

## A. Setup

We use the CIFAR-10 and CIFAR-100 datasets [33], which are image datasets containing 10 and 100 different labels, respectively. Both datasets are pre-divided into training and test sets. We use 60% of the training data to prepare a pretrained original model, while the remaining 40% is allocated to the client datasets. The dataset used for training the original model is assumed to be IID.

In contrast, we adopt a Non-IID setting for the client datasets in this study. Specifically, each client is assigned data for only 2 out of 10 labels in the CIFAR-10 dataset, and 10 out of 100 labels in the CIFAR-100 dataset. This type of Non-IID setting is also commonly used in related work [31].

Specifically, we generate Non-IID client data in the following procedure. First, we allocate two or ten classes (for CIFAR-10/CIFAR-100) to each client. Then, for each client i and its class c, we sample $\alpha _ { i , c } \sim U ( 0 . 4 , 0 . 6 )$ and assign $\frac { \alpha _ { i , c } } { \sum _ { j } { \alpha _ { j , c } } }$ of the class c data. We do the same for N clients and their classes. Of the allocated training data for each client, 20 percent is reserved for validation.

On the other hand, the test dataset is divided into N portions, matching the number of clients. The distribution of the test dataset for each client is made consistent with the distribution of the client’s training set, by using $c , \alpha _ { i , c }$ assigned to the clients during the splitting of the training set.

Fig. 4 illustrates the structure of the teacher and student models. We use ResNet-18 [32] as the original model, which is pre-trained and owned by the server. The first half of the model is distilled and personalized to be more suitable for SC using the proposed method. During training, each client holds copies of the teacher model (which is pre-trained and frozen) and the student model, and the student model is updated locally. In the inference phase, each personalized head network, extracted from the student model, is deployed on the client, while the corresponding tail networks are deployed on the server. We assume stable communication links, where the student model parameters and intermediate representations are ideally transferred from the clients to the server without any loss.

Table II shows the settings. The number of epochs for pretraining is the same for both methods to align the condition. For CIFAR-100, the number of epochs for local HND is higher than for CIFAR-10 because the task is more complicated and takes longer to learn. The number of training epochs increases as the clients’ data volume decreases to equalize the number of gradient updates when the data volume changes. We use Adam [34] to optimize the parameters by minimizing the loss function shown by eq. (4) in training client models.

Compared methods. Table III shows compared methods. Both use HND, but the scope of data use and training methods differ. The proposed method, referred to as FedHNP, is compared with the conventional HND under different data constraints.

Centralized HND: Centralized HND assumes an ideal scenario where client data can be aggregated, meaning there are no data constraints. In this baseline, the clients’ datasets are collected, combined, and used for model training with HND on the server. The goal of the proposed method is to achieve accuracy comparable to this centralized approach.

Individual HND: Similar to the proposed method, this approach assumes a scenario where client data cannot be aggregated, and each client performs HND individually using only its own data. Due to the reduced amount of available data during HND, accuracy is expected to deteriorate, particularly when the number of client data samples is small.

## B. Results

1) Reduction of Computational and Communication Costs via Model Replacement: We begin by demonstrating the impact of replacing the original model with the trained Head and Tail networks on both computational and communication loads. Table IV outlines the computational load on the client and server, along with the communication traffic generated during inference when using cloud computing, SC with the original model, and SC with the model trained by the proposed method. The split point refers to the layer at which the model is divided, starting from the input layer. Computational load is measured by the number of floating-point operations (additions and multiplications) performed by the models. Data volume corresponds to the size of the intermediate representation transmitted from the client during inference, presented in both the commonly used 32-bit floating-point format and an example of 8-bit quantization-based compression. For cloud computing, communication traffic is calculated under the assumption that input data (e.g., images) are directly transmitted.

![](images/08e39a56248d0cb06f1a3331bfe74744ef35f24930fb21e596335ee14a63befb.jpg)  
Fig. 5: Comparison of training curves with CIFAR-10, 20000 client training samples. For FedHNP, we plot the stage 1 accuracy for each of the ten epochs and the accuracy after stage 2 in the “personalized” position.

As shown in Table IV, replacing the original model with the proposed Head and Tail networks significantly improves the trade-off between computational load and communication traffic. When splitting the original ResNet-18 model, the viable split points, considering client computational load and avoiding skip connections, are at the 2nd and 8th layers. However, splitting at the 2nd layer results in high communication traffic, while splitting at the 8th layer imposes excessive computational load on the client, rendering both points unsuitable for split computing in their original form.

In contrast, by replacing the original model with the one trained using the proposed method, the overall computational load is reduced by 24% compared to the original model, with client-side computation kept at a similar level to the 2ndlayer split. Furthermore, communication traffic is drastically reduced—down to one-third of that in cloud computing and 1/8 to 1/16 of that in split computing with the original model. These results highlight the effectiveness of replacing the Head and Tail networks when implementing split computing.

2) Comparison of Training Curves Across Different Training Methods: As part of the evaluation of model accuracy for each method, we first present the training curves versus training epochs for the proposed method and the baseline methods. For the proposed method, FedHNP, since training occurs in rounds of ten epochs, we report the test accuracy after each round during the FL stage, as well as the final accuracy after personalization.

Fig. 5 illustrates the test accuracy at each training epoch using CIFAR-10 with 20,000 client training samples. The dashed line represents the accuracy of the original model trained with 30,000 pre-training samples. For both FedHNP and Individual HND, the models trained on each client are evaluated using the test dataset assigned to that client, with the average and standard deviation of the results depicted. In contrast, for the Original model and Centralized HND, evaluation is performed using the aggregated test data from all clients.

In Fig. 5, Centralized HND, which trains the student model under ideal conditions, shows the fastest increase in accuracy, with final accuracy approaching that of the original model. In contrast, Individual HND demonstrates a slower accuracy increase, ultimately converging at a lower level.

TABLE II: Training parameters
<table><tr><td>Parameters</td><td>Values for CIFAR-10</td><td>Values for CIFAR-100</td></tr><tr><td>Sum of clients&#x27; data</td><td>1000 to 20000</td><td>4000 to 20000</td></tr><tr><td>Number of pre-training epochs</td><td colspan="2">200</td></tr><tr><td>Number of clients to choose from in one round of FL step in FedHNP C</td><td colspan="2">10</td></tr><tr><td>Number of client epochs within one round of FL in FedHNP EFL</td><td colspan="2"> $\lfloor 2 \times 1 0 ^ { 5 } / \mathrm { s u m }$  of clients&#x27; data|</td></tr><tr><td>Number of rounds of FL for FedHNP R</td><td>20 10</td><td>100</td></tr><tr><td>Number of epochs of personalization in FedHNP  $E _ { p }$ </td><td></td><td>20</td></tr><tr><td>Number of HND training epochs for method 2 and 3</td><td> $\lfloor 4 \times 1 0 ^ { 6 } / \mathrm { s u m }$  of clients&#x27; data]</td><td> $\lfloor 2 \times 1 0 ^ { 7 } \rfloor$  /sum of clients&#x27; data]</td></tr></table>

TABLE III: Compared methods
<table><tr><td></td><td></td><td>Is the client data label needed?</td><td>Is the labeled data for pre-train needed?</td><td>Is communication during training needed?</td><td>Is the privacy of the client data maintained?</td></tr><tr><td>Method1</td><td>FedHNP (Proposed)</td><td>No</td><td>Yes</td><td>Yes</td><td>Yes</td></tr><tr><td>Method2</td><td>Centralized HND</td><td>No</td><td>Yes</td><td>No</td><td>No</td></tr><tr><td>Method3</td><td>Individual HND</td><td>No</td><td>Yes</td><td>No</td><td>Yes</td></tr></table>

TABLE IV: Computation and Communication Cost Reduction via Split Computing vs. Cloud Computing.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Split point</td><td colspan="2">Computation load (FLOPs)</td><td colspan="3">Communication traffic</td></tr><tr><td>Client</td><td>Server</td><td>Data shape</td><td>Data volume (8bit)</td><td>Data volume (32bit)</td></tr><tr><td>Cloud computing using original model</td><td></td><td>0</td><td> $1 . 4 0 \times 1 0 ^ { 8 }$ </td><td>(3, 32, 32)</td><td>3.1 KB</td><td>12.3 KB</td></tr><tr><td>Split computing using original model</td><td>2</td><td> $1 . 7 7 \times 1 0 ^ { 6 }$ </td><td> $1 . 3 8 \times 1 0 ^ { 8 }$ </td><td>(64, 16, 16)</td><td>16.4 KB</td><td>65.5 KB</td></tr><tr><td>(two split points are listed)</td><td>8</td><td> $5 . 4 2 \times 1 0 ^ { 7 }$ </td><td> $8 . 6 0 \times 1 0 ^ { 7 }$ </td><td>(128, 8, 8)</td><td>8.20 KB</td><td>32.8 KB</td></tr><tr><td>Split computing using replacement model</td><td>3</td><td> $2 . 0 2 \times 1 0 ^ { 6 }$ </td><td> $1 . 0 4 \times 1 0 ^ { 8 }$ </td><td>(12, 9, 9)</td><td>1.0 KB</td><td>3.9 KB</td></tr></table>

Meanwhile, FedHNP exhibits the slowest accuracy increase during the FL-based training stage, achieving significantly lower accuracy after 200 epochs. However, client personalization significantly boosts model performance, with the mean accuracy becoming comparable to or even exceeding that of the original model while still preserving client data privacy. This improvement is likely because, while Centralized HND trains a single model capable of recognizing all classes, FedHNP leverages personalization to tailor the model to each client’s training dataset, which is consistent with the distribution of that client’s test dataset. On the other hand, although Individual HND is trained on a dataset with a distribution consistent with the test dataset, it does not achieve the same high performance as FedHNP.

These results suggest that while the Federated Learning stage in FedHNP does not directly yield high accuracy on each client’s test data, it serves as an effective pre-training mechanism for the student model. This pre-training significantly enhances the personalization stage, enabling efficient fine-tuning of the model.

3) Accuracy Evaluation Under Different Dataset Configurations: Fig. 6 shows the test accuracy for each method using the CIFAR-10 and CIFAR-100 datasets, with varying numbers of client training samples. For CIFAR-10, we use between 1,000 and 20,000 total samples, corresponding to 20 to 400 samples per client. For CIFAR-100, due to the difficulty of training with too few samples, we use between 4,000 and 20,000 total samples, corresponding to 80 to 400 samples per client. The dashed line indicates the accuracy of the original model, which is pre-trained and thus independent of the total client data. Additionally, for the proposed method, the accuracy after personalization is shown. The results represent the average of 10 trials, along with their standard deviation, each conducted with a different random seed.

![](images/99dbbdc09c96022d2556a6d8af9b236c4cd9889b44f3861318758891ddd131ad.jpg)

![](images/edc263915df356f074498d34c0df12789e348842b80442ebeef1a28b7490f626.jpg)  
(a) CIFAR-10  
(b) CIFAR-100  
Fig. 6: CIFAR-10 / CIFAR-100 test accuracy in different total number of clients data

As the number of client data samples increases, the accuracy of all methods improves, though none surpass the accuracy of the original model. This is due to the fact that, as shown in Section V-B1, the head network replaces parts of the original model with a smaller number of model parameters and a bottleneck to reduce computational and communication costs. Despite these challenges, the proposed method still achieved model accuracy comparable to that of Centralized HND, which operates under ideal conditions. Even in cases where client datasets were small, although the proposed method lagged behind the original model by a dozen points and fell a few points short of Centralized HND, it still outperformed Individual HND by several dozen points. This indicates that the application of the Federated Learning framework significantly mitigates the accuracy degradation caused by insufficient client data.

We note that the non-IID setting used in this evaluation, in which each client holds only 2 out of 10 classes (CIFAR-10) or 10 out of 100 classes (CIFAR-100), corresponds to an extreme label-distribution skew, often referred to as a pathological non-IID setting [16]. In addition, the sampling ratio $\alpha _ { i , c } ~ \sim ~ U ( 0 . 4 , 0 . 6 )$ introduces quantity skew across clients. The reported results therefore characterize the behavior of FedHNP under one of the severest heterogeneity conditions. Qualitatively, as the heterogeneity decreases toward IID, the Stage-1 global model approaches the behavior of Centralized HND and the additional gain of personalization diminishes; that is, FedHNP is expected to degrade gracefully to its FL backbone under mild heterogeneity. A systematic sweep of intermediate heterogeneity levels, e.g., using Dirichlet-based partitions, requires re-running the full training pipeline and is left as future work.

4) Evaluation of Personalization Effectiveness in FedHNP: In the results presented so far, the proposed FedHNP, by incorporating HND, enabled local model training with unlabeled data. The use of the FL framework primarily acted as an effective pre-training mechanism for the student model. Moreover, personalization significantly improved accuracy to levels comparable to the original model. Here, we delve deeper into the impact of personalization in FedHNP.

Fig. 7 illustrates the training and validation loss, as well as accuracy, plotted against training epochs during the client personalization stage. The dataset is CIFAR-10 with 20,000 client samples. The loss and accuracy at epoch 0 represent the values from the global model trained in stage 1. As shown in Fig. 7, both training and validation losses drop sharply in the first few epochs, followed by a gradual decline. Similarly, accuracy improves substantially during the initial fine-tuning epochs, then levels off with a slower rate of improvement. This indicates that even the relatively small number of finetuning epochs (10 epochs) used in the previous experiments was sufficient to significantly enhance accuracy for this task. Furthermore, it suggests that dedicating additional computational resources to personalization could further enhance the model’s accuracy.

Fig. 8 presents box-and-whisker plots of client model test accuracy before and after personalization, using 1,000 and 20,000 client data samples. The orange line represents the median, while the green triangle denotes the mean accuracy across clients. The box reflects the interquartile range (IQR), capturing the middle 50% of the data, while the whiskers extend to 1.5 times the IQR, showing the overall data spread.

Personalization not only significantly boosts accuracy but also reduces the variation in model accuracy between clients. This effect is particularly pronounced when the number of client data samples is small, with the median accuracy increasing by 40 points. Additionally, in cases where clients have 20,000 data samples, more than 50% of clients achieve accuracy levels of 90% or higher, comparable to the original model.

![](images/5532a719488e55fc65e52609ae6cf82257a0b3fa36f72e1b798344d092c749cb.jpg)  
(a) Loss

![](images/6499b6973b152a2020588d5ff30b3a8b00d8b2f6f53e3371446c6c00e5e0efa8.jpg)  
(b) Accuracy

Fig. 7: Stage 2 training curve with CIFAR-10, 20000 data. Personalizing for 100 epochs.  
![](images/7ba40be24f30471e30e56e911581e0ac048dfa5941de9bf64d6bd2bbb3dd9a40.jpg)

![](images/c693b6fad0211d1839e5bb10965ebb8d2d2d2f225227cb6cd1b930e83a0f6586.jpg)  
(a) Case1: 1000 data for clients (b) Case2: 20000 data for clients  
Fig. 8: Test accuracy before / after personalization in FedHNP. The triangle represents the mean, and the line in the box represents the median.

Thus, personalization is a crucial component in the proposed FedHNP. The integration of HND, FL, and personalization proves to be highly effective, allowing for the training of accurate Head Networks using only small amounts of unlabeled data.

5) Discussion: Reliance on the Pre-trained Model: Finally, we discuss the reliance on the pre-trained model. Because HND trains the student to reproduce the intermediate representations of the teacher extracted from the pre-trained model, the accuracy of the pre-trained model acts as an effective upper bound for the student: the dashed lines in Fig. 6 represent this empirical ceiling, and none of the distillation-based methods exceed it. Conversely, if the pre-trained model is poorly trained for the target task, FedHNP inherits its deficiency. We consider the availability of a well-trained model a mild assumption in practice, since high-quality pre-trained models for common tasks are widely available in public repositories, whereas their original training datasets are typically not released—which is precisely the situation our method targets. When no suitable pre-trained model exists, supervised training of the full model is unavoidable and our method is not applicable.

## VI. CONCLUSION

In this study, we proposed an unsupervised training method for Head Networks that reduces both computational load on mobile devices and communication with the server, even under highly data-constrained conditions. The method integrates HND, a distillation-based training approach, with the FL framework, a collaborative model training method, to address challenges posed by unlabeled and limited training data. Additionally, by introducing a personalization technique, the drop in inference accuracy resulting from model replacement with the Head Network is effectively mitigated. Experimental evaluations demonstrate that, despite the stringent data constraints unique to our proposed method, it achieves accuracy comparable to baselines with fewer or no data constraints. The Head Network trained by our method reduces communication requirements on mobile devices by 95%, with only a 3-5 point drop in accuracy compared to the original model. Additionally, the results show that executing personalization—specifically fine-tuning—on the client side can increase inference accuracy by more than 30 points, further highlighting the significant role personalization plays in mitigating accuracy degradation when using compact Head Networks.

Future work will focus on developing methods to determine the optimal model architecture for each device. In this study, we assumed a fixed model structure across all clients. However, in real-world applications, computational power and communication delays are likely to vary between devices and over time. Therefore, it is essential to explore approaches that tailor model architectures to individual devices or prepare multiple models and select the most appropriate one for each device.

## REFERENCES

[1] G. Hinton, Y. LeCun, and Y. Bengio, “Deep learning,” nature, vol. 521, no. 7553, pp. 436–444, 2015.

[2] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” Advances neural inf. process. syst., vol. 34, pp. 8780–8794, 2021.

[3] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” arXiv preprint arXiv:1810.04805, 2018.

[4] S. H. Vemprala, R. Bonatti, A. Bucker, and A. Kapoor, “Chatgpt for robotics: Design principles and model abilities,” IEEE Access, vol. 12, pp. 55 682–55 696, 2024.

[5] S. S. Biswas, “Role of chat gpt in public health,” Annals biomed. eng., vol. 51, no. 5, pp. 868–869, 2023.

[6] T. Guo, “Cloud-based or on-device: An empirical study of mobile deep inference,” in Proc. IEEE Int. Conf. Cloud Eng., 2018, pp. 184–190.

[7] Y. Kang et al., “Neurosurgeon: Collaborative intelligence between the cloud and mobile edge,” SIGARCH Comput. Archit. News, vol. 45, no. 1, pp. 615–629, 2017.

[8] Y. Matsubara, M. Levorato, and F. Restuccia, “Split computing and early exiting for deep learning applications: Survey and research challenges,” ACM Comput. Surv., vol. 55, no. 5, pp. 1–30, 2022.

[9] A. E. Eshratifar, A. Esmaili, and M. Pedram, “Bottlenet: A deep learning architecture for intelligent mobile cloud computing services,” in Proc. IEEE/ACM Int. Symp. Low Power Electron. Des., Lausanne, Switzerland, 2019, pp. 1–6.

[10] Y. Matsubara, S. Baidya, D. Callegaro, M. Levorato, and S. Singh, “Distilled split deep neural networks for edge-assisted real-time systems,” in Proc. Workshop Hot Top. Video Analyt. Intell. Edges, Los Cabos, Mexico, 2019, pp. 21–26.

[11] Y. Matsubara, D. Callegaro, S. Baidya, M. Levorato, and S. Singh, “Head network distillation: Splitting distilled deep neural networks for resource-constrained edge computing systems,” IEEE Access, vol. 8, pp. 212 177–212 193, 2020.

[12] Y. Matsubara and M. Levorato, “Neural compression and filtering for edge-assisted real-time object detection in challenged networks,” in Proc. 2020 25th Int. Conf. Pattern Recognit., 2021, pp. 2272–2279.

[13] H. Choi, R. A. Cohen, and I. V. Bajic, “Back-and-forth prediction for´ deep tensor compression,” in Proc. IEEE Int. Conf. Acoust. Speech Signal Process., 2020, pp. 4467–4471.

[14] M. Sbai, M. R. U. Saputra, N. Trigoni, and A. Markham, “Cut, distil and encode (cde): Split cloud-edge deep inference,” in Proc. 18th Annu. IEEE Int. Conf. Sens. Commun. Netw., 2021, pp. 1–9.

[15] S. Yao, J. Li, D. Liu, T. Wang, S. Liu, H. Shao, and T. Abdelzaher, “Deep compressive offloading: speeding up neural network inference by trading edge computation for network latency,” in Proc. 18th Conf. Embedded Netw. Sens. Syst., 2020, pp. 476–488.

[16] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y. Arcas, “Communication-Efficient Learning of Deep Networks from Decentralized Data,” in Proc. 20th Int. Conf. Artif. Intell. Statist., vol. 54, 2017, pp. 1273–1282.

[17] D. Hu and B. Krishnamachari, “Fast and accurate streaming cnn inference via communication compression on the edge,” in Proc. IEEE/ACM 5th Int. Conf. Internet-of-Things Des. Implement., 2020, pp. 157–163.

[18] S. Shimizu, T. Nishio, S. Saito, Y. Hirose, C. Yen-Hsiu, and S. Shirakawa, “Neural architecture search for improving latency-accuracy trade-off in split computing,” in IEEE Globecom Workshops, 2022, pp. 1864–1870.

[19] X. Dong, B. De Salvo, M. Li, C. Liu, Z. Qu, H. Kung, and Z. Li, “Splitnets: Designing neural architectures for efficient distributed computing on head-mounted systems,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 12 559–12 569.

[20] Y. Akimoto, S. Shirakawa, N. Yoshinari, K. Uchida, S. Saito, and K. Nishida, “Adaptive stochastic natural gradient method for one-shot neural architecture search,” in Proc. 36th Int. Conf. Mach. Learn., vol. 97, 2019, pp. 171–180.

[21] C. Li, Z. Yu, Y. Fu, Y. Zhang, Y. Zhao, H. You, Q. Yu, Y. Wang, and Y. C. Lin, “HW-NAS-bench: Hardware-aware neural architecture search benchmark,” in Int. Conf. Learn. Represent., 2021.

[22] A. Bakhtiarnia, N. Milosevi ˇ c, Q. Zhang, D. Bajovi ´ c, and A. Iosifidis,´ “Dynamic split computing for efficient deep edge intelligence,” in Proc. IEEE Int. Conf. Acoust. Speech Signal Process., 2023, pp. 1–5.

[23] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv: 1503.02531, 2015.

[24] E. Jeong, S. Oh, H. Kim, J. Park, M. Bennis, and S.-L. Kim, “Communication-efficient on-device machine learning: Federated distillation and augmentation under non-IID private data,” in Proc. Conf. Neural Inf. Process. Syst., 2nd Workshop Mach. Learn. Phone Other Consum. Devices, Nov. 2018, pp. 1–6.

[25] S. Itahara, T. Nishio, Y. Koda, M. Morikura, and K. Yamamoto, “Distillation-based semi-supervised federated learning for communication-efficient collaborative training with non-IID private data,” IEEE Transactions on Mobile Computing, vol. 22, no. 1, pp. 191– 205, 2023.

[26] C. Wu, F. Wu, L. Lyu, Y. Huang, and X. Xie, “Communication-efficient federated learning via knowledge distillation,” Nature Communications, vol. 13, no. 2032, 2022.

[27] Y. Li, C. Wu, Z. Du, L. Zhong, and T. Yoshinaga, “Communicationefficient federated learning for UAV networks with knowledge distillation and transfer learning,” in Proc. IEEE Global Communications Conference (GLOBECOM), Kuala Lumpur, Malaysia, 2023, pp. 5739– 5744.

[28] A. Furutanpey, P. Raith, and S. Dustdar, “FrankenSplit: Efficient neural feature compression with shallow variational bottleneck injection for mobile edge computing,” IEEE Transactions on Mobile Computing, vol. 23, no. 12, pp. 10 770–10 786, 2024.

[29] K. Matsuda, Y. Sasaki, C. Xiao, and M. Onizuka, “Benchmark for personalized federated learning,” IEEE Open J. Comput. Soc., vol. 5, pp. 2–13, 2024.

[30] A. Fallah, A. Mokhtari, and A. Ozdaglar, “Personalized federated learning with theoretical guarantees: A model-agnostic meta-learning approach,” in Proc. Adv. Neural Inf. Process. Syst., vol. 33. Curran Associates, Inc., 2020, pp. 3557–3568.

[31] A. Shamsian, A. Navon, E. Fetaya, and G. Chechik, “Personalized federated learning using hypernetworks,” in Proc. 38th Int. Conf. Mach Learn., vol. 139, 2021, pp. 9489–9502.

[32] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., Las Vegas, NV, USA, 2016, pp. 770–778.

[33] A. Krizhevsky, G. Hinton et al., “Learning multiple layers of features from tiny images,” 2009.

[34] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Proc. 3rd Int. Conf. Learn. Represent., 2015.