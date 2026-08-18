# Transfer Learning of Keystroke Dynamics for Cross-Device User Authentication

Nuwan Kaluarachchi <sup>†</sup>, Sevvandi Kandanaarachchi <sup>‡</sup>, Kristen Moore <sup>‡</sup>, Arathi Arakala <sup>†</sup>, Conrad Sanderson <sup>‡⋄</sup>

<sup>†</sup> RMIT University, Australia; <sup>‡</sup> CSIRO, Australia; <sup>⋄</sup> Griffith University, Australia

Abstract—Keystroke dynamics (typing patterns) can be used as a behavioural biometric modality for user authentication, with applications such as fraud prevention. While the modality has been shown to work well for single device authentication, its application to cross-device scenarios is more challenging. Dynamics learned on one device (eg., phone) may not be directly applicable to authentication on a secondary device with a different form factor (eg., tablet) due to changes in typing patterns that can lead to distribution drifts. To address this, we propose a crossdevice user authentication system based on inductive transfer learning, where keystroke dynamics learned on one device are adapted to a secondary device. The adapted data is then combined with necessarily limited training data for the secondary device, which is used to robustly train a binary classifier. Furthermore, an extended set of keystroke features is used to better capture discriminative dynamics. Experiments on the BBMAS dataset show that proposed system achieves an equal error rate of 14.2% for the cross-device scenario, surpassing state-of-the-art methods. Index Terms—transfer learning, biometrics, user authentication, Index Terms—transfer learning, biometrics, user authentication,

cross-device, keystroke dynamics, human-computer interaction.

## I. Introduction

With the increasing use of passkeys for one-time password free authentication, user fingerprint and face features are commonly used to ensure device security. However, when continuous device security is required (eg., for fraud prevention in financial applications [1]), authentication via keystroke dynamics is an alternative user-friendly approach [2], [3], [4].

In a biometric authentication scenario, a necessarily limited number of samples of a user’s biometric features is used to build a user model at registration. When a user is required to authenticate their identity, they present their biometric sample to the device, which compares it with the registered model and either accepts or rejects the user’s identity claim.

In our increasingly connected world, users have multiple devices with varying form factors (eg., phones, tablets, laptops) and authenticate on each device separately in a manner unique to that device. Variations in keyboard size, style and surface can cause changes in users typing patterns. If a user were to employ continuous authentication on all their devices, each device would need to be trained separately. This repetition and tediousness can deter users from embracing continuous authentication, thereby reducing overall security.

Despite the variation in keystroke dynamics due to users moving across devices, consistency in typing styles can be exploited to transfer authentication models trained on one device to authenticate users on a secondary device [5]. If enough keystroke data for each device is available, then standard machine learning models can be trained on multi-device data for cross-device authentication [6]. However, challenges arise when there is limited training data for the secondary device [7].

Transfer learning problems can be categorised into three types based on their target labels [8], [9], [10]: (i) transduct ive, (ii) inductive, (iii) unsupervised. In transductive transfer learning, the labels come from only the source domain, and the labels of the target domain are predicted using the model. In inductive transfer learning, labels are available in both the source and the target domains. In unsupervised transfer learning, no labelled data is available for both source and target.

Transfer learning can also be categorised into two categories based on the employed feature space: (i) homogeneous and (ii) heterogeneous [11]. Homogeneous transfer learning indicates that the source and target feature space and label space are similar. Heterogeneous transfer learning indicates that the source features, target features, and label spaces are different.

Existing work on keystroke based authentication across devices has notable limitations. For example, Lin et al. [5] use a temporally aware learning mechanism to authenticate users across sessions. Model parameters are adapted via ad-hoc fine-tuning and retraining to take into account cross-device settings, rather than using explicit transfer learning. Sun et al. [7] propose stratified transfer learning for cross-device user identification, where the feature spaces are aligned into a common representation for both devices, without explicitly learning the transformations between them. Monaco and Vindiola [12] conduct a cross-domain comparison using an inductive transfer encoder to learn a general transformation between domains. However, each domain uses the same device with unique settings (such as left- and right-hand usage), rather than an explicitly different device.

It must also be noted that the above studies used a small number of features (4 to 6) to represent keystroke dynamics, potentially limiting their ability to discriminate between users. Table I shows salient characteristics of recent literature on keystroke dynamics using cross-device learning and/or limited target data.

In this work, we propose a novel user-specific cross-device authentication system based on a homogeneous inductive transfer learning approach, which employs sufficient source device data and a limited amount of target device data to train a transfer encoder. The proposed system is named as Transfer Encoder Data-fusion cross-device Binary Classification, shortened to TEDxBC. The system can reduce the training burden for a secondary device by taking a well-trained keystroke model on one device and adapting it using limited data to a keystroke model specific for the secondary device. Furthermore, we use an extended set of features in order to better capture discriminative keystroke dynamics.

We continue the paper as follows. The proposed system is detailed in Section II. The employed keystroke features are summarised in Section III. Performance evaluation and an ablation study are given in Section IV. The main findings and future avenues of research are given in Section V.

![](images/4fadf915d7b6ef5ce79359a20e8326cbe34e8f10e92e085f826d3bdf0f1e87cf.jpg)  
Figure 1: Overview of the proposed TEDxBC system for cross-device user authentication. The top section shows the training phase, comprised of 3 components: Transfer Encoder (TE), Data Fusion (D), and Binary Classifier (BC). All source data and limited target data are used fo training and validation. The bottom section shows the testing phase, which evaluates the system’s performance using unseen target data.

Table I: Salient characteristics of recent literature on keystroke based authentication using cross-device learning (XDL) and/or limited target data (LTD). The text for authentication can be either restricted to fixed words/phrases, or unrestricted (free).
<table><tr><td>Year [Ref] Device</td><td></td><td>Text type Features XDL LTD</td><td></td><td></td><td></td></tr><tr><td>2016 [12]</td><td>desktop</td><td>fixed</td><td>4</td><td>x</td><td>√</td></tr><tr><td>2017 [13]</td><td>desktop</td><td>free</td><td>2</td><td>x</td><td>√</td></tr><tr><td rowspan="3">2020 [6]</td><td>desktop to phone</td><td rowspan="3">free</td><td>6</td><td>√</td><td>x</td></tr><tr><td>desktop to tablet</td><td></td><td>√</td><td>x</td></tr><tr><td>tablet to phone</td><td></td><td>√</td><td>x</td></tr><tr><td rowspan="3">2022 [7] 2022 [5]</td><td>phone to tablet</td><td>fixed, free</td><td>5</td><td>√</td><td>√</td></tr><tr><td>desktop1 to desktop2 free</td><td></td><td>6</td><td>x</td><td>√</td></tr><tr><td>phone to tablet</td><td>free</td><td>6</td><td>√</td><td>√</td></tr><tr><td rowspan="2">2023 [14]</td><td>desktop</td><td>free</td><td>4</td><td>x</td><td>√</td></tr><tr><td>phone</td><td>free</td><td>4</td><td>x</td><td>√</td></tr><tr><td>2024 [15]</td><td>desktop to phone</td><td>free</td><td>16</td><td>√</td><td>x</td></tr><tr><td>this paper</td><td>phone to tablet</td><td>fixed, free</td><td>24</td><td>√</td><td>√</td></tr></table>

## II. Proposed Approach

We assume the following authentication scenario. A given user has a large amount of keystroke data from the source device (eg., phone) and a limited amount of data from the target device (eg., tablet). The target device needs to be trained so that the user can be authenticated accurately on it. However, the training process is constrained by the lack of sufficient keystroke data from the target device. To address this challenge, we propose the TEDxBC system, summarised in Fig. 1.

The training phase is comprised of 3 components: Transfer Encoder (TE), Data Fusion (D), and Binary Classifier (BC). The transfer encoder is a multi-layer neural network which is used as a function that maps the features from the source domain to the target domain. The data fusion component applies this function to convert the keystroke data from the source device into the target domain, and combines it with the existing target domain data to form an enlarged training dataset for the target device. The binary classifier then uses the augmented training dataset to build a user-specific model for the target device. During the test phase, the trained classifier is used for determining whether a presented typing pattern belongs to the claimed user. Each of the components is summarised in the following subsections.

## A. Transfer Encoder

As labelled data is available in both the source and target domains, an inductive transfer encoder is employed [10]. Let $X = \{ x _ { i } \} _ { i = 1 } ^ { N }$ be a set of a user’s labelled samples, where each sample �<sub>�</sub> is a vector of keystroke features. The labels represent classes, where class 1 indicates samples from a given user, while class 0 indicates impostor samples (ie., samples that do not belong to the given user). Furthermore, let $Z = \{ z _ { i } \} _ { i = 1 } ^ { N }$ be the user’s set of labelled samples (vectors) in the target domain.

The source and target data for a user are paired using a bipartite sampling strategy [12] to create non-overlapping sets of cross-device source-target pairs for training and validation of the transfer encoder.

In brief, the encoding function transforms the input source data into a latent space and uses a single hidden layer with the hyperbolic tangent activation function:

$$
\pmb { h } _ { i } = \operatorname { t a n h } \left( \pmb { W } \pmb { x } _ { i } + \pmb { b } _ { 1 } \right)\tag{1}
$$

where � is a weight parameter matrix, and $\pmb { b } _ { 1 }$ is a bias parameter vector. The output latent vector $\pmb { h } _ { i }$ has values between −1 and +1. Similarly, the decoding function maps $\pmb { h } _ { i }$ to the target domain using:

$$
\widehat { z } _ { i } = \operatorname { t a n h } \left( W ^ { T } \pmb { h } _ { i } + \pmb { b } _ { 2 } \right)\tag{2}
$$

where b�� is the estimated mapping for the given source data $x _ { i }$ in the target domain, $W ^ { T }$ is known as the tied weight parameter matrix (transpose of the earlier � matrix), and ${ \pmb b } _ { 2 }$ is a bias parameter vector. We add a drop-out layer between the encoder and decoder layers, with parameter � indicating the drop rate. Overall, the parameters of the transfer encoder are $\{ W , b _ { 1 } , b _ { 2 } , \gamma \}$

In encoder-decoder architectures, it is common for the decoder’s weight matrix to be chosen as the transpose of the encoder’s weight matrix for several reasons [16]: (i) to minimise the number of parameters to be learned during training, (ii) to allow the model to capture genuine patterns using the same basis vectors used by �, and (iii) to generalise better to unseen data.

The original samples from the target domain (�<sub>�</sub>) are compared with the estimated samples $\widehat { z _ { i } }$ via the loss function:

$$
L \left( z _ { i } , { \widehat { z _ { i } } } \right) = \parallel z _ { i } - { \widehat { z _ { i } } } \parallel _ { 2 }\tag{3}
$$

where ∥ · ∥<sub>2</sub> denotes the $l ^ { 2 } .$ -norm and $z _ { i } \in Z , \ { \widehat { z } } _ { i } \in { \widehat { Z } } , \ i \in [ 1 , N ]$

During the training process, parameters �, $\pmb { b } _ { 1 }$ and ${ \pmb b } _ { 2 }$ are initialised using an optimal initialiser selected from a range of options, including Glorot Uniform, Glorot Normal, He Normal, He Uniform, and Random Uniform [17]. To mitigate the risk of overfitting, we employ an early-stopping strategy. This technique monitors the performance on the validation set specific to each user, halting training when the validation performance ceases to improve. This approach also ensures the model generalises well to unseen data, thereby enhancing the reliability and robustness of the user authentication system.

The loss function for each user is by minimised separately, meaning that for each user we determine the optimal initialiser, the optimal parameters, and the optimal number of hidden units for the encoder layers.

## B. Data Fusion

As the target domain has minimal data to train a classifier, the trained and validated transfer encoder is used to transform the source data into the target domain to increase the number of samples available in the target domain. The Data Fusion step combines this transformed source data with the limited target domain training data for a user to form an enlarged training dataset for the binary classifier.

## C. Binary Classifier

The binary classifier is used to decide if a given keystroke sample belongs to the user (class 1) or not (class 0). Random Forest was chosen as the machine learning model since it is generally considered to be robust to overfitting [18], [19] and has been previously used in keystroke-based authentication [7].

Each user has their own classifier, with the hyperparameters tuned to achieve optimal performance through a straightforward grid search strategy. The search considers various parameters: the complexity of individual trees (max\_tree\_depth), minimum number of samples required at a leaf node (min\_samples\_leaf) and to split an internal node (min\_samples\_split). The ensemble size (n\_estimators) is varied, and two split quality criteria are used: Gini impurity (gini) and Information gain based on entropy (entropy). The list of hyperparameters and their search ranges are listed in Table II.

## III. Keystroke Features

During typing on a keyboard (physical or touchscreen), a raw keystroke sequence comprises the sequence of pressed keys, the direction of pressure, and the timestamps of each key press and release. Consequently we extract 24 distinct features from such a sequence, which are placed into three categories: 6 common temporal features such as hold times and flight times [6], 16 Distance-Enhanced Flight Times features [20], and 2 so-called non-conventional features [21]. The features are described in the subsections below.

## A. Temporal Features

Conventional temporal features are based on press and release timing instances of single keys or adjacent key pairs in a keystroke sequence [6]. The hold time is a feature for a single key press, calculated as the difference between the time of release and time of press of a key. The flight time is a feature for adjacent key pairs and based on press and release instances; it can be further delineated into four variants [22] as demonstrated in Fig. 2 and summarised below:

• Flight 1 (F1): up-down time or key interval latency. Time between press of the second key and release of the first key.

• Flight 2 (F2): up-up time or key release latency. Time between release of the second key and release of the first key.

• Flight 3 (F3): down-down time digraph time or key press latency. Time difference between the presses of the first and second keys.

• Flight 4 (F4): down-up time. Time between release of the second key and press of the first key.

We compute the median F1, F2, F3 and F4 values of all keypairs in a user’s keystroke sequence to form the first 4 features. We also use the median hold time of all keys in a sequence and the median hold time for a tri-graph as features. This gives us 6 common temporal features.

## B. Distance-Enhanced Flight Times Features

Distance-Enhanced Flight Times (DEFT) are temporal features defined for key pairs that are at a fixed distance from each other on a keyboard [20]. Fig. 3 shows the QWERTY keyboard and examples of key pairs and their distances. For example, key pairs (A-S) and (X-D) are both at a distance of 1 and are on the left side of the keyboard. The features are defined as the median flight times of all distance {1, 2, 3, 4} key pairs on either the left or right side of a keyboard, typed in a keystroke sequence. There are a total of 32 possible DEFT features (4 flight times × 4 distances × 2 keyboard sides), out of which we select the 16 most discriminative features (determined via preliminary experiments).

## C. Non-Conventional Features

So-called non-conventional features are keystroke dynamic features that, compared to the features in preceding subsections, focus on other typing characteristics such as semi-timing and editing features as defined in [21]. Based on preliminary experiments, we use two non-conventional features: the median error rate percentage and the median negative up-down feature.

![](images/54e9d3de044eb6589607a5c5bb6ae778dfe30d23266967b3cde93c4215160e67.jpg)  
Figure 2: Press (P) and release (R) events for two keys (H and E).

![](images/65302153705fd55ca284b8ad8be86ac98a977c2cce310423252d47963c5abf11.jpg)  
Figure 3: Calculation of distances between key pairs. Pair (A-S) is a distance 1 digraph, pair (T-H) is a distance 2 digraph, and pair (N-L) is a distance 3 digraph. The longest distance bet ween a key pair is distance 9. The blue line separates the keyboard into left and right sides, which helps identify keys typed by the left or right hand.

Table II: List of hyperparameters tuned via grid search for the Random Forest classifier.
<table><tr><td>Parameter</td><td>Possible Values</td></tr><tr><td>max_tree_depth</td><td>10, 30, 50, 70, 90, 100, None</td></tr><tr><td>min_samples_leaf</td><td>1,2, 4</td></tr><tr><td>min_samples_split</td><td>2, 5, 10</td></tr><tr><td>n_estimators</td><td>100, 200, 400, 600, 800, 1000</td></tr><tr><td>criterion</td><td>&#x27;gini&#x27;, ‘entropy&#x27;</td></tr></table>

## IV. Experiments

We evaluate the performance of the proposed system on the publicly available Behavioral Biometrics Multi-device and multi-Activity datafrom Same users (BBMAS) dataset [23], [24]. We use data from 114 participants for which keystroke sequences are available from phones and tablets (ie., a crossdevice scenario). Two types of performance evaluations are performed: (i) an ablation study, and (ii) comparison against two existing keystroke methods that can be applied to crossdevice authentication.

For the ablation study, 3 variants of the proposed TEDxBC system are evaluated: (a) full TEDxBC (all components); (b) TExBC, which is TEDxBC without the data fusion component (target data is not used); (c) DxBC, which is TEDxBC without the transfer encoder component.

For comparison against existing methods, we use DoubleType (DT) [6] and Stratified Transfer Learning (STL) [7], which were proposed for the cross-device paradigm and used for the phone to tablet scenario on the BBMAS dataset.

Compared to the proposed TEDxBC approach, the DT and STL methods considerably differ in their handling of cross-device scenarios. DT relies solely on device feature relationships without employing transfer learning techniques for cross-device adaptation. STL aligns the source and target domain data at the class level to create a common feature space, rather than explicitly learning the transformation between the source and target spaces.

The same experiment setup is used for all methods. The drop-out layer between the encoder and decoder layers uses an empirically determined drop rate of $\gamma = 0 . 3$ . The keystroke data from each user on each device is divided into samples. Similar to [6], [7], we define one sample as consisting of 150 keystrokes. This sample size is chosen based on the typical message size limit on common social media platforms [25]. With this sample size, each user has an average of 45 samples per device. The 45 samples for each user are partitioned into three non-overlapping subsets using a 40%-20%-40% split for both the source device (phone) and the target device (tablet). In each set, an equal number of observations that do not represent the user (feature vectors from other users) are added to create a balanced 2-class set.

In the source domain, the first two subsets are used as the training and validation sets for the transfer encoder, while the remaining subset is reserved for transformation into the target domain. The transformed samples are subsequently used to augment the training data for the binary classifier. In the target domain, the first two subsets are similarly used for training and validating the transfer encoder. The final subset is held out and used exclusively as the test set for evaluating the binary classifier.

The following performance metrics are used: accuracy, precision, recall, F1 score and equal error rate (EER) [26], [27]. For all metrics except the EER, higher values indicate better performance. For the EER, lower values indicate better performance. Examples of variations in the distribution of a feature across users and devices in a cross-device setting are shown in Fig. 4. The results are shown in Table III, with the corresponding ROC curves given in Fig. 5.

Overall, the proposed TEDxBC method achieves the highest performance across all metrics, surpassing the DT and STL methods. The ablation study indicates that both the transfer encoder and data fusion are critical components, with the former having the most effect. The results reveal that the transfer encoder transforms the source data well into the target domain, and the binary classifier works effectively when it is combined with a limited amount of target data.

![](images/8c404ac84fdc74ccbba55192e4fff78afbcd86d23af975c6a3b01cd20173d487.jpg)

![](images/6cf1f637e3bdcfb00c47e0a9ba1f351c3cc70093df45bf2f8fb9b0993434411f.jpg)  
Figure 4: Examples of variations in the distribution of a feature across users and devices in a cross-device setting (phone and tablet). Panel (a) shows an almost ideal situation: well separated distributions between the users, with each user having very similar distributions across two devices. Panel (b) shows a challenging situation: overlapping distributions between the users, with each user exhibiting significant drift in the distributions across two devices.

## V. Main Findings

In this work we have proposed a method for cross-device keystroke authentication, called TEDxBC, suited to scenarios where the amount of training data for a secondary device is limited. For example, a well-trained authentication model is available on a user’s phone, but authentication is required on the user’s tablet, which has a significantly different form factor that can lead to changes in keystroke dynamics.

The proposed method has three main components: inductive transfer learning, data fusion, and user-specific binary classification. Furthermore, an extended set of keystroke features is used to better capture discriminative keystroke dynamics of individuals. Specifically, it addition to traditional temporal features, we use Distance-Enhanced Flight Time features [20] which include median flight times between key pairs at specific distances, as well as non-conventional features [21] such as text editing events.

The transfer encoder is a multi-layer neural network which is used as a function that maps features from the source domain to the target domain. The data fusion component applies this function to convert the keystroke data from the source device into the target domain, and combines it with the existing target domain data to form an enlarged training dataset for the target device. The binary classifier then uses the augmented training dataset to build a user-specific model for the target device.

Table III: Performance obtained on the BBMAS dataset for: proposed TEDxBC method; DxBC and TExBC (ablated versions of TEDxBC); DoubleType (DT) [6]; Stratified Transfer Learning (STL) [7].
<table><tr><td>Method</td><td>EER</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>TEDxBC</td><td>14.2%</td><td>84.1%</td><td>88.6%</td><td>78.9%</td><td>82.8%</td></tr><tr><td>TExBC</td><td>22.1%</td><td>78.3%</td><td>78.9%</td><td>69.2%</td><td>72.1%</td></tr><tr><td>DxBC</td><td>27.0%</td><td>68.8%</td><td>66.2%</td><td>60.4%</td><td>61.3%</td></tr><tr><td>DT [6]</td><td>24.7%</td><td>78.3%</td><td>80.8%</td><td>67.0%</td><td>73.9%</td></tr><tr><td>STL [7]</td><td>35.0%</td><td>66.8%</td><td>55.9%</td><td>50.1%</td><td>51.0%</td></tr></table>

![](images/6a6448dfa13334346a4f14bf09c1013a33452cd2a1d95ee5e3b0c374ec811c64.jpg)  
Figure 5: Performance obtained on the BBMAS dataset for: proposed TEDxBC method; DxBC and TExBC (ablated versions of TEDxBC); DoubleType (DT) [6]; Stratified Transfer Learning (STL) [7]. The closer a curve is to the to the top-left corner, the better the performance.

The transfer encoder learns the relationship between keystroke dynamics on various devices by mapping feature distributions from one device to another. Instead of relying solely on model fine-tuning or feature space alignment, the proposed method dynamically captures the behavioural variations across devices, enabling more effective and personalised cross-device authentication. This approach ensures that user-specific typing patterns remain distinguishable even in the presence of deviceinduced variations, providing a more robust and adaptable authentication mechanism.

Comparative evaluations on the BBMAS dataset show that the proposed method achieves an equal error rate of 14.2% for the cross-device scenario (phone to tablet), surpassing the performance of DoubleType [6] and Stratified Transfer Learning [7] methods.

A limitation of the present work is that the evaluation uses a single dataset, which may limit generalisability in realworld scenarios. As such, in future work we aim to assess the general applicability of the proposed method by validating its effectiveness on multiple other datasets that capture keystroke dynamics of the same individuals across various devices, including the recent KVC-onGoing benchmark [28]. It may also be useful to evaluate the performance in more detail according to the notions of Doddington’s zoo (biometric menagerie) [29], [30], where authentication performance can differ based on the category of users.

[1] J. H. Huh, S. Kwag, I. Kim et al., “On the long-term effects of continuous keystroke authentication: Keeping user frustration low through behavior adaptation,” Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, vol. 7, no. 2, pp. 1–32, 2023.

[2] R. Shadman, A. A. Wahab, M. Manno et al., “Keystroke dynamics: Con cepts, techniques, and applications,” ACM Computing Surveys, vol. 57, no. 11, pp. 1–35, 2025.

[3] J. J. Jeong, Y. Zolotavkin, and R. Doss, “Examining the current status and emerging trends in continuous authentication technologies through citation network analysis,” ACM Computing Surveys, vol. 55, no. 6, pp. 1–31, 2022.

[4] F. Monrose and A. D. Rubin, “Keystroke dynamics as a biometric for authentication,” Future Generation Computer Systems, vol. 16, no. 4, pp. 351–359, 2000.

[5] C. Lin, J. He, C. Shen, Q. Li, and Q. Wang, “CrossBehaAuth: Crossscenario behavioral biometrics authentication using keystroke dynamics,” IEEE Transactions on Dependable and Secure Computing, vol. 20, no. 3, pp. 2314–2327, 2022.

[6] A. K. Belman and V. V. Phoha, “DoubleType: Authentication using relationship between typing behavior on multiple devices,” in International Conference on Artificial Intelligence and Signal Processing (AISP), 2020.

[7] H. Sun, G. Xu, X. Zhang, Z. Wu, and B. Gao, “Stratified transfer learning of touchscreen behavior on cross-device for user identification,” in Communications in Computer and Information Science (CCIS), vol. 1588, 2022, pp. 618–631.

[8] Z. T. Pritee, M. H. Anik, S. B. Alam et al., “Machine learning and deep learning for user authentication and authorization in cybersecurity: A state-of-the-art review,” Computers & Security, vol. 140, p. 103747, 2024.

[9] P. Yan, A. Abdulkadir, P.-P. Luley et al., “A comprehensive survey of deep transfer learning for anomaly detection in industrial time series: Methods, applications, and directions,” IEEE Access, vol. 12, pp. 3768–3789, 2024.

[10] S. J. Pan and Q. Yang, “A survey on transfer learning,” IEEE Transactions on Knowledge and Data Engineering, vol. 22, no. 10, pp. 1345–1359, 2010.

[11] W. Guo, Y. Dong, and G.-F. Hao, “Transfer learning empowers accurate pharmacokinetics prediction of small samples,” Drug Discovery Today, vol. 29, no. 4, p. 103946, 2024.

[12] J. V. Monaco and M. M. Vindiola, “Crossing domains with the inductive transfer encoder: Case study in keystroke biometrics,” in IEEE International Conference on Biometrics Theory, Applications and Systems (BTAS), 2016.

[13] H. Çeker and S. Upadhyaya, “Transfer learning in long-text keystroke dynamics,” in IEEE International Conference on Identity, Security and Behavior Analysis (ISBA), 2017.

[14] T. Neacsu, T. Poncu, S. Ruseti, and M. Dascalu, “DoubleStrokeNet: Bigram-level keystroke authentication,” Electronics, vol. 12, no. 20, p. 4309, 2023.

[15] Y. Yang, B. Guo, Y. Liang, K. Zhao, and Z. Yu, “Cross-device free-text keystroke dynamics authentication using federated learning,” Personal and Ubiquitous Computing, vol. 28, no. 3, pp. 491–505, 2024.

[16] P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, and P.-A. Manzagol, “Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion,” Journal of Machine Learning Research, vol. 11, no. 110, pp. 3371–3408, 2010.

[17] X. Glorot and Y. Bengio, “Understanding the difficulty of training deep feedforward neural networks,” in International Conference on Artificial Intelligence and Statistics (AISTATS), 2010, pp. 249–256.

[18] L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.

[19] E. Scornet and G. Hooker, “Theory of random forests,” Annual Review of Statistics and Its Application, vol. 13, no. 1, pp. 99–121, 2026.

[20] N. Kaluarachchi, S. Kandanaarachchi, K. Moore, and A. Arakala, “DEFT: A new distance-based feature set for keystroke dynamics,” in International Conference of the Biometrics Special Interest Group (BIOSIG), 2023, pp. 79–89.

[21] A. Alsultan, K. Warwick, and H. Wei, “Non-conventional keystroke dynamics for user authentication,” Pattern Recognition Letters, vol. 89, pp. 53–59, 2017.

[22] A. K. Belman and V. V. Phoha, “Discriminative power of typing features on desktops, tablets, and phones for user identification,” ACM Transactions on Privacy and Security, vol. 23, no. 1, pp. 1–36, 2020.

[23] A. K. Belman, L. Wang, S. S. Iyengar et al., “SU-AIS BB-MAS (Syracuse University and Assured Information Security - Behavioral Biometrics Multi-device and multi-Activity data from Same users) Dataset,” https: //doi.org/10.21227/rpaz-0h66, IEEE Dataport, 2019.

[24] A. K. Belman, L. Wang, S. S. Iyengar et al., “Insights from BB-MAS – a large dataset for typing, gait and swipes of the same person on desktop, tablet and phone,” arXiv:1912.02736, 2019.

[25] S. McLachlan, “Ideal social media post length for every platform,” https: //blog.hootsuite.com/ideal-social-media-post-length/, 2026, accessed 2026-07-30.

[26] A. Tharwat, “Classification assessment methods,” Applied Computing and Informatics, vol. 17, no. 1, pp. 168–192, 2020.

[27] F. Cardinaux, C. Sanderson, and S. Bengio, “User authentication via adapted statistical models of face images,” IEEE Transactions on Signal Processing, vol. 54, no. 1, pp. 361–373, 2006.

[28] G. Stragapede, R. Vera-Rodriguez, R. Tolosana et al., “KVC-onGoing: Keystroke verification challenge,” Pattern Recognition, vol. 161, p. 111287, 2025.

[29] G. R. Doddington, W. Liggett, A. F. Martin et al., “Sheep, goats, lambs and wolves: A statistical analysis of speaker performance in the NIST 1998 speaker recognition evaluation,” in International Conference on Spoken Language Processing (ICSLP), 1998.

[30] A. Mhenni, E. Cherrier, C. Rosenberger, and N. Essoukri Ben Amara, “Analysis of Doddington zoo classification for user dependent template update: Application to keystroke dynamics recognition,” Future Genera tion Computer Systems, vol. 97, pp. 210–218, 2019.