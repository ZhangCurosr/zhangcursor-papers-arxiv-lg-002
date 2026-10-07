# Multi-Label Perceptual Bug Detection in Video Games using Deep Learning on Gameplay Footage

Nahian Rifaat

Felix Morosov

Loutfouz Zaman

Ontario Tech University

Universitat Osnabr¨ uck¨

Ontario Tech University

Oshawa, Ontario, Canada

Osnabruck, Niedersachsen, Germany¨

Oshawa, Ontario, Canada

nahian.rifaat@ontariotechu.net

fmorosov@uni-osnabrueck.de

loutfouz.zaman@ontariotechu.ca

Abstract—Traditional approaches for automated bug detection in video games, such as manual testing, can be beneficial for the improvement of quality assurance, but they can be expensive and time-consuming. The scarce number of tools available to detect multiple perceptual bugs in the same video frame introduces detection challenges for automated bug detection tools in realworld scenarios. We propose a deep learning model for multilabel perceptual bug detection and compare it against video classification models such as Inflated 3D ConvNet and 3D ResNet. Our proposed model, ResNet-BiLSTM, achieved an F1 score of 85.78% on the benchmark dataset. Our results demonstrated that temporal dependency modelling is beneficial for accurate video-based bug detection. We believe this work with multilabel perceptual bug detection on gameplay videos will help save resources spent on manual testing workloads in video games. Furthermore, we introduce a new dataset with multilabel perceptual bugs in this work. The dataset contains 77,969 video clips across different genres of games with approximately 1.2 million frames, containing combinations from 5 classes of bugs in the same video frame.

Index Terms—Automated bug detection, bidirectional long short-term memory (Bi-LSTM), multi-label classification.

## I. INTRODUCTION

While playing video games, encountering a bug can reduce the quality of immersion a player experiences [1]. There are numerous effects of these bugs causing minor to major negative impacts on player experience, the reputation of the developers of the game, and the loss of potential revenue that can prevent players from reaching their intended goal [2].

Although games are thoroughly tested with massive amounts of resources directed towards quality assurance before launch [3], [4], even AAA game developers face the reality of bugs creeping into their games [5]. The Last of Us Part 1 faced such issues with character models missing details, jittery camera movement, and distorted hair, among others [6]. These bugs damaged the reputation of the developers and the publishing company. The scale and intricacies of such games present a real challenge to ensure players have a bug-free experience before release. Furthermore, manual testing and other effective methods of detecting bugs can be time-consuming and expensive [3]. Coppola et al. [7] proposed a taxonomy of 26 distinct issues and combined them into five macrocategories. They reported that available research focuses a lot on bugs detectable through the multimedia appearance of a game, such as animation errors, HUD failures, and glitches in rendering. The authors noted that automated detection requires complex and resource-heavy instrumentation of the game engine. This underscores the need for detection methods that can identify bugs without requiring deep integration into the game engine. The vastness of the variety of bugs can make it almost impossible and impractical to search for bugs at every level for complex and expansive video games [8].

Among low-level, engine, application, perceptual, and behavioural bugs, the latter two cannot be detected with logical rules [9]. Perceptual bugs in games are issues that affect the display or appearance of game elements [9]. On the contrary, behavioural bugs have a negative impact on game mechanics or game logic. In the era of Artificial Intelligence (AI), game developers can use deep learning to automate and effectively enhance the bug detection process [10], [11].

Perceptual bugs present unique challenges for bug detection. These classes of bugs, such as geometry corruption, texture corruption, and Z-clipping, require an understanding of visual characteristics. Furthermore, perceptual bugs such as Z-fighting bugs cause objects to flicker, which requires an understanding of the bug’s temporal characteristics in addition to its visual ones over multiple frames [12]. Additionally, video clips might contain multiple bugs occurring at the same time due to cascading defects, introducing the challenge of identifying multiple simultaneous perceptual bugs, which is more difficult than multiclass classification.

To address the mentioned difficulties, we framed perceptual bug detection as a supervised multi-label classification problem. In this work, we introduce a new architecture combining spatial and temporal feature extraction to create an automated test oracle to solve the multi-label bug classification problem. Our key contributions in this work are as follows:

• a new model architecture combining spatial and temporal feature extraction that learns a multi-task objective for multi-label classification;

• a benchmark dataset [13] for multi-label perceptual bug detection with 77,969 video clips prepared from recorded gameplay sessions, containing clips with a combination of multiple perceptual bugs (up to a combination of 3 classes of bugs from a set of 5 classes of perceptual bugs).

## II. RELATED WORK

## A. Traditional Game Testing

In traditional game testing, human testers participate in sessions where they conduct manual assessments and observations to recognize bugs. Human testers actively look for potential bugs while they execute test cases. Usability testing evaluations are widely used in these manual approaches.

For example, a case study was conducted [14] where Choi was able to identify that usability expert evaluation provides valuable and novel feedback for testing games. Korhonen and Koivisto [15] proposed a model including three main modules, namely, usability, mobility, and gameplay, for playability heuristics for mobile games. In this study, the model’s effectiveness in enhancing the game evaluation and playability issue detection was tested and verified. Moreno-Ger et al. [16] proposed a new methodology based on analysis of recorded gameplay that provided a systematic and targeted approach to suggest improvements that benefited playtesting in serious games. Diah et al. [17] conducted a study to verify the effectiveness of a game called Jelajah within their proposed usability testing method with children. Their data proved that their proposed evaluation method can be effectively adopted for usability testing. Duarte et al. [18] introduced a systematic framework for exploratory testing. The framework consisted of nine tailored strategies. This framework successfully helped uncover 80 bugs in the study. However, the authors confirmed that such methods are heavily reliant on human testers, which makes evaluation costly and time-consuming. Additionally, all these manual testing strategies could be used to detect perceptual bugs, but would lead to the same outcome in resource expenses.

## B. Automated Bug Detection

Automated Bug Detection (ABD) primarily involves software agent creation with the ability of game engagement and exploration to uncover and identify bugs. Although there is no definitive answer to the superior effectiveness of ABD techniques compared to traditional methods such as manual testing, earlier researchers have conducted comparative studies in this regard. Dobles et al. [19] compared manual and automated testing and found that automated tests require more initial effort compared to manual ones. However, their findings suggest that automated testing is superior for repetitive testing and performs better in defect detection.

Varvaressos et al. [20] reported an approach where instrumenting the game loop could enable game developers to monitor the temporal properties in the runtime and detection violations of formal specifications during gameplay. Such methods require access to the game’s state and manual definition of rules.

Iftikhar et al. [21] introduced an automated functional testing approach using a model-based methodology for games and evaluated it on two platformer games. Their proposed approach employed unified modelling language profiles, class diagrams, and state machines to automatically generate and execute test cases. The authors reported that the effectiveness of their approach might be less effective in complex games because of its inability to cover all paths and functionalities. This approach requires users with previous software engineering experience to create models prior to testing.

Lovreto et al. [22] manually crafted some functional test scripts for their proposed automated test script method and evaluated the method on 16 mobile games. However, the authors reported that their approach had several limitations, including the inability to address the unpredictability inherent in certain aspects of the games effectively.

Guglielmi et al. [23] proposed GamEpLay Issue Detector (GELID) for anomaly identification through the analysis of gameplay videos to offer insights to game developers to improve their games. To evaluate GELID, they performed a study on 604 video segments derived from 80 hours of gameplay footage across three video games. The authors reported that the segmentation process and issue-based clustering were effective. However, the steps involving categorization and contextbased clustering segments are reported to require refinement before being implemented in practical applications reliably.

Prasetya et al. [24] proposed an agent-based automated navigation framework for playtesting in video games. The agents perform graph-based pathfinding. The framework uses search-based and rule-based techniques. However, the authors reported that the framework lacks generalization.

Guerrero-Romero et al. [25] proposed an automated testing approach using a team of AI agents to detect behavioural anomalies and validate game design during development.

Albaghajati and Ahmed [26] leveraged two agent-based genetic algorithms for playtesting on video games. One of the agents generates buggy states within the game, while the other agent analyzes the game state using a representation based on Petri nets modelling. The approach relies on rule-based logic and struggles to identify a bug from which other related bugs emerge. The authors acknowledge that their algorithm struggles to effectively pinpoint the primary bug that could potentially create additional issues in the game.

Zhao et al. [27] introduced an approach that derived playing tactics from actual gameplay and uses them to automatically test games for potential bugs. This approach incorporates a library of rule-based inferences and is able to respond to randomly generated game scenarios. But the applicability of this framework may be limited in more complex games.

## C. AI and Machine Learning in Game Testing

AI and Machine Learning (ML) has proven to be a powerful tool for bug detection backed by years of previous literature. With modern video games increasing in challenges and growing complexity, and a requirement for quicker release cycles and Downloadable Content (DLC) updates, manual playtesting, along with other traditional methods of bug detection, continues to remain resource-intensive and expensive. Approaches that incorporate AI are a promising alternative for ABD with more efficiency and effective results.

Bug identification and automated game exploration are the two main components of ABD. Wuji is one of the most recent frameworks that incorporates ABD and game exploration with the use of deep reinforcement learning (DRL) [28]. Wuji detects four categories of bugs, namely, crash, stuck, logical, and gaming balance, by combining evolutionary multi-object optimization and DRL. Rani et al. [29] demonstrated that DRL can be used to find the presence of bugs by analyzing game screens effectively even in low-quality or blurred frames. Xue used Q-learning to explore and test graphical user interfaces (GUIs) for a collection of mobile apps to create behaviour models and train two optimization objectives, namely, activity coverage and the number of crashes [30]. However, the methods proposed by Zheng et al. [28] and Xue [30] were both rulebased approaches that could not detect high-level perceptual and behavioural bugs. In this work, our focus is to design models capable of detecting these challenging bugs faced by players. Furthermore, a self-learning game testing framework that can navigate and take advantage of game mechanics through reinforcement learning (RL) guided by a predefined reward signal was introduced by Bergdahl et al. [10]. Pfau et al. [31] focused on developing an autonomous framework, where games are played, and bugs are reported. Pfau et al. used discrete RL, where they used short-term and long-term memory in pairs as a solving mechanic. The mentioned literature mostly works on game environment exploration rather than bug detection. On the contrary, our work detects bugs on a benchmark dataset created from human-played sessions and works towards maximizing detecting multiple bugs at the same time in gameplay videos.

Wilkins et al. [32] introduced an open platform for ABD testing for 3D game environments, focusing more on perceptual bugs. The platform in this study called World of Bugs (WOB), employs learning-based methods by leveraging the rendered scenes. An AI agent explores, captures, and detects bugs from the captured video frames from the game environment, similar to a real player. The authors identified ten types of perceptual bugs and varying detection accuracy across categories, probably stemming from instances in which the object’s rear geometry did not render properly, tricking the model into not classifying the event as a bug.

Paduraru et al. [33] introduced a hierarchical detection framework using real-time game state information to identify visual anomalies, addressing a range of glitches, namely, mesh distortions, missing textures, and camera clipping. In this work, they used a two-stage hierarchical architecture. The first stage used a ShuffleNetV2 model to identify regions of interest, and in the second stage, another backbone is triggered for verification of the bug if an anomaly is suspected. Their system is designed to stream screenshots to external inference servers to avoid impacting the game’s runtime performance. Earlier, we framed bug detection as an anomaly detection problem, making the system learn the normal state of a game and identify deviations as bugs [9], [34]. We integrated an RL agent to autonomously explore game environments and gather data. We used a Convolutional Long Short-Term Memory (ConvLSTM) model to process sequences of gameplay frames and captured temporal dependencies by a sliding window approach that helped detect bugs such as flickering or inconsistent rendering. The framework is trained on bug-free gameplay, and once an anomaly is detected, the framework employs the bug classification stage and uses the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm to cluster predicted buggy frames into distinct categories of perceptual bugs, such as geometry corruption, boundary hole, Z-clipping, and so on. In summary, our previous work first detected anomalies and then used unsupervised clustering to identify the type of perceptual bug. Lin et al. [35] introduced a methodology for video-based bug detection where they used video metadata and the random forest classifier to rank gameplay videos based on the likelihood of bug presence by framing bug detection as a binary classification problem.

Ling et al. identified rendering defects such as stretched geometry and missing textures with a ShuffleNetV2 architecture on static individual frames [36], but only dealt with a single visual glitch each frame. Abdelfattah et al. detected multi-frame bugs such as Level of Detail (LOD) pops and culling pops in their work using a Hybrid Co-Fine Tuning (CFT) framework to detect visual bugs when there is a scarcity of labelled data [37]. They defined the task as an object detection objective to both localize using bounding boxes and classify visual bugs within a frame. The model was trained simultaneously on labelled data from the ”target” game and a mixture of other games. To utilize large amounts of unlabelled data, they used a Joint Embedding Predictive Architecture (JEPA), which involves masking image patches and tasking the model with reconstructing them in the embedding space. Their framework utilizes a faster Region-based Convolutional Neural Network (RCNN) object detection architecture with a Vision Transformer (ViT) backbone. Savran et al. [38] used YOLO v10 to detect visual and logical bugs to perform realtime localization of bugs, treating bug detection as an object detection problem. The dataset used in this work has images for each bug class out of five classes of bugs.

TABLE I: Comparison of automated bug detection approaches
<table><tr><td>Study</td><td>Approach</td><td>Multi-label</td><td>Data</td></tr><tr><td>Varvaressos et al. [20]</td><td>Runtime Monitoring</td><td>No</td><td>Engine Hooks</td></tr><tr><td>Lin et al. [35]</td><td>Ranking</td><td>No</td><td>Video metadata</td></tr><tr><td>Abdelfattah et al. [37]</td><td>Hybrid co-finetuning</td><td>No</td><td>Multiframe</td></tr><tr><td>Rani et al. [29]</td><td>DRL (DQN)</td><td>No</td><td>Image</td></tr><tr><td>Savran et al. [38]</td><td>YOLO v10</td><td>No</td><td>Image</td></tr><tr><td>Paduraru et al. [33]</td><td>ShuffleNetV2</td><td>No</td><td>Image+State</td></tr><tr><td>Our previous study [9], [34]</td><td>Anomaly detection</td><td>No</td><td>Video</td></tr><tr><td>This study</td><td>ResNet-BiLSTM</td><td>Yes</td><td>Video</td></tr></table>

As illustrated in Table I, each literature approached bug detection in games differently. However, none of the mentioned literature tackled the problem of detecting multiple perceptual bugs at the same time in the gameplay footage. To the best of our knowledge, such a dataset is not available. Therefore, we acknowledged this research gap and developed a benchmark dataset with large amounts of labelled data, which consists of human-played gameplay footage with multiple perceptual bugs occurring simultaneously. Furthermore, we developed a new model architecture to test our proposed model’s performance on the mentioned dataset.

## III. MATERIALS AND METHODS

This section covers the details of the benchmark datasets used in this study. The dataset contains data collected from gameplay footage. We have also provided the details of the proposed neural network architecture used in this study for multi-label perceptual bug detection on the mentioned benchmark dataset.

![](images/fc2821d4afdb6f2809fc8c67bd160540aa9e28614c98e246f5a23424f5ac49c2.jpg)  
Fig. 1: Dataset preparation pipeline.

## A. Dataset Preparation

The benchmark dataset was prepared by recording gameplay sessions from three different open-source games, namely, Open Nights [39], Unity3D Mario Kart Racing Game [40], and FPS Microgame [41]. The data collection procedure for the study was approved by the institutional Research Ethics Board (Reference File No.: 20215). The games selected for data collection are described below:

1) Open Nights: Open Nights is a first-person horror game focused on survival. This game is characterized by low-light environments and static camera angles. Open Nights was chosen for the dataset to test the model’s ability to detect bugs that are harder to detect in low light, such as texture corruption and Z-fighting.

2) Unity3D Mario Kart Racing Game: The Unity3D Mario Kart Racing Game is an open-source implementation of the Mario Kart game. It is a third-person racing game with highspeed movements on colourful racing tracks. We selected this game to test models on bugs in complex 3D environments, where bugs occur with high-speed moving assets.

3) FPS Microgame: The FPS Microgame is a fast-paced first-person shooter game characterized by rapid camera movements. This game was selected for the dataset to test the ability of models to distinguish motion blur from bugs such as geometry corruption and Z-fighting.

The above titles were selected so that our dataset contained footage from diverse genres of games with different graphical demands and bugs to test our models on.

11 participants volunteered to be part of the gameplay recording sessions and were recruited through convenience sampling. The videos were recorded at 30 frames per second (fps). To introduce bugs, we deployed a bug injector system and a gameplay recorder within the Unity projects of the mentioned games to record the gameplay sessions of the participants. During the gameplay sessions, the bug injector algorithm introduced perceptual bugs at random moments or when the game character came in close range of a predetermined game object. We chose to work with artificially introduced bugs because the chosen games were open-source and did not have perceptual bugs. Up to 3 perceptual bugs were injected at the same time from a list of 5 classes of perceptual bugs, namely, geometry corruption, texture corruption, boundary hole, Z-fighting, and Z-clipping. These bug types have distinct features, which allows the creation of models that can classify them properly. Furthermore, all of the classes of bugs are injected into the gameplay sessions of each game. After the recording of a gameplay session was finished, our clip miner system resized the gameplay footage to a resolution of 224 × 224 and separated the recorded video into 2-second clips at 8 fps. After all of the gameplay sessions had been recorded, we froze the dataset by splitting it into train (70%), validation (15 %), and test (15 %) sets, maintaining the same proportions of the number of bug combinations (up to three bugs) in each split. The dataset was finalized by preparing a JavaScript Object Notation (JSON) file that contained the data of the clips, the dataset splits, and the data about the bugs within those video clips. The dataset preparation pipeline is shown in Fig. 1.

![](images/30964ac5518c27b76f04440cb6ed4e5aa98f9ea84d58f4411a727001d4f2bdfb.jpg)

![](images/305b664c589b7cb7510a2d616a743bc0ec627c542598727ac070dc301b13e8c6.jpg)  
(b) Z-clipping in the Open Nights game: after

(a) Z-clipping in the Open Nights game: before  
![](images/88bea5cf1b7d1db0662ea865be889f9163024035f7bfbdb2d7f0d74641f65722.jpg)  
(c) Texture corruption in Mario Kart: before.

![](images/d3cc76b4eb62fe33a3d219535d8541ffe9b398821a61a479847e7ff68f9d7e8b.jpg)  
(d) Texture corruption in Mario Kart: after.

![](images/704ef3d90a65b9fa0f0cc26c75e81578ad4ffd59f36771d5822d4bda78df54e9.jpg)  
(e) Geometry corruption in the FPS Microgame: before.

![](images/315d9a7ff739ed82a0c7281938bae5ca4e313410bf646e0f6076a8d2b918800f.jpg)  
(f) Geometry corruption in the FPS Microgame: after.  
Fig. 2: Examples of perceptual bugs from the dataset.

## B. Dataset Description

The multi-label bug detection dataset we prepared consisted of five classes of perceptual bugs (bugs that affect the appearance of game elements), up to a combination of three concurrent bugs per clip taken from the gameplay footage. We set the bug combinations to a maximum of three concurrent perceptual bug classes per frame to prevent overlapping bugs from occluding each other, allowing models to accurately recognize the visual features of each class of bug. Some examples of the bugs available in the dataset are shown in Fig. 2. The five classes of bugs are, namely, boundary hole, texture corruption, geometry corruption, Z-clipping, and Zfighting. Each bug type occurs at most once in a video clip. Gameplay sessions from all 3 games are injected with each class of the mentioned bugs.

![](images/96a47185e01eaa0f0ff26a1c3e81e4caa10f0bb4b8c48425a43fd4ad01c14366.jpg)  
Fig. 3: ResNet-BiLSTM architecture.

The dataset consists of 77,969 video clips (2-second clips at 8 fps) prepared from the gameplay sessions. The dataset has a total of 1,247,504 frames and includes all the video clips. This dataset was split into train (70%), validation (15%), and test (15%) sets. Each split contains all 5 classes of perceptual bugs previously mentioned and approximately the same proportion of the number of bug combinations (up to three bugs per video clip). Further details about the dataset are shown in Table II and Table III.

TABLE II: Distribution of bug count combinations across splits.
<table><tr><td>Bug Count</td><td>Test</td><td>Train</td><td>Validation</td></tr><tr><td>0 bug(s)</td><td>7,669 (65.55%)</td><td>35,785 (65.57%)</td><td>7,668 (65.58%)</td></tr><tr><td>1 bug(s)</td><td>2,103 (17.97%)</td><td>9,807 (17.97%)</td><td>2,101 (17.97%)</td></tr><tr><td>2 bug(s)</td><td>1,070 (9.15%)</td><td>4,986 (9.14%)</td><td>1,068 (9.13%)</td></tr><tr><td>3 bug(s)</td><td>858 (7.33%)</td><td>3,998 (7.33%)</td><td>856 (7.32%)</td></tr><tr><td>Total</td><td>11,700</td><td>54,576</td><td>11,693</td></tr></table>

TABLE III: Individual bug class distribution in videos across splits.
<table><tr><td>Bug Type</td><td>Test</td><td>Train</td><td>Validation</td></tr><tr><td>Boundary Hole</td><td>1,313 (11.22%)</td><td>6,138 (11.25%)</td><td>1,363 (11.66%)</td></tr><tr><td>Texture Corruption</td><td>1,434 (12.26%)</td><td>6,616 (12.12%)</td><td>1,352 (11.56%)</td></tr><tr><td>Geometry Corruption</td><td>1,287 (11.00%)</td><td>6,205 (11.37%)</td><td>1,326 (11.34%)</td></tr><tr><td>Z-Clipping</td><td>1,450 (12.39%)</td><td>6,620 (12.13%)</td><td>1,413 (12.08%)</td></tr><tr><td>Z-Fighting</td><td>1,333 (11.39%)</td><td>6,194 (11.35%)</td><td>1,351 (11.55%)</td></tr></table>

## C. Model Architecture

In this study, we introduce a model architecture, ResNet and Bidirectional Long Short-Term Memory (ResNet-BiLSTM), for the multi-label bug detection task. The model first extracts visual features from the frames of the video using a pretrained ResNet-18 [42] and then feeds the extracted features into a Bidirectional Recurrent Neural Network (BRNN) [43]. In our case, we are using a Long Short-Term Memory Network (LSTM) [44] to use within the BRNN. This Bidirectional LSTM (BiLSTM) network is used to extract temporal information from the video frames. Next, the output of the BiLSTM is fed into the Temporal Attention layer to extract the most essential features for the multi-label classification task, followed by a dropout regularization layer. The output of the dropout layer is then fed into three separate classification heads. The first classification head has a Fully Connected (FC) layer followed by a sigmoid activation function to classify the multiple bugs present in the video clip. Furthermore, the second classification head consists of an FC layer followed by a softmax classifier to predict if the video contains the number of bugs between 0, 1, 2, or 3 bugs. Finally, the third classification head contains an FC layer followed by a sigmoid classifier to detect the presence of bugs. The model architecture with output shapes after each layer is shown in Fig. 3.

## IV. EXPERIMENTAL SETUP

This section covers the setup used in the experiments, including the environment, hyperparameters and evaluation metrics.

## A. Hardware, Environment and Hyperparameters

The experiments ran on servers with an NVIDIA H100 GPU. The server ran on the Fedora Linux operating system. We used PyTorch to build the models and experiments. Furthermore, we used Weights & Biases to track the experiments. Training hyperparameters for our model are given below:

• Epochs: 60

• Batch size: 32

• Dropout rate: 0.3

• Learning rate: 0.0007

• Optimizer: ADAM [45]

## B. Loss function

For the model architecture shown in Fig. 3, we used a combination of different loss functions. The multi-task objective

![](images/f5caefac8e45599bc683b249a2e1148f99abefc1282a8a40e538b2effa55f609.jpg)  
Fig. 4: Evaluation metrics per bug combination for the proposed ResNet-BiLSTM model on the test set.

used for our multi-label bug classification task, which includes a combination of three loss terms, is as follows:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { t y p e s } } + \lambda _ { \mathrm { c n t } } \mathcal { L } _ { \mathrm { c o u n t } } + \lambda _ { \mathrm { a n y } } \mathcal { L } _ { \mathrm { a n y } }\tag{1}
$$

where $\lambda _ { \mathrm { c n t } } , \lambda _ { \mathrm { a n y } }$ are scalar weights controlling the contribution of each auxiliary term. $\mathcal { L } _ { \mathrm { t y p e s } }$ is the Asymmetric Loss (ASL) [46] being used for multi-label bug classification. ${ \mathcal { L } } _ { \mathrm { c o u n t } }$ is the Focal Cross-Entropy [47] loss being used for bug count detection, and $\mathcal { L } _ { \mathrm { a n y } }$ indicates the Binary Cross-Entropy loss for bug presence detection.

## C. Evaluation metrics

We use the micro-averaged F1 score as the evaluation metric for multi-label bug type classification because micro-averaging gives equal weight to each prediction rather than each class, making it suitable for imbalance among classes in the dataset and also because there is a dominance of examples with no bugs. The equation for the micro-averaged F1 score is as follows:

$$
F _ { 1 } = \frac { 2 \times P \times R } { P + R }\tag{2}
$$

where Precision, $\begin{array} { r } { P = \frac { T P } { T P + F P } } \end{array}$ and Recall $\begin{array} { r } { R = \frac { T P } { T P + F N } . } \end{array}$ with $T P , F P ,$ and FN denoting true positives, false positives, and false negatives, respectively.

## V. RESULTS

This section summarizes the outcomes of the experiments performed utilizing the proposed architecture from our study on the proposed dataset. We compared the proposed model with video-based classification models.

As shown in Table IV, the proposed model architecture shown in Fig. 3 outperforms other video classification models such as Inflated 3D ConvNet (I3D) [48] and 3D ResNet (R3D-18) [49], which were fine-tuned with the same hyperparameters as our proposed model for the comparison. Our proposed model achieved an F1 score of 85.78% on the test dataset. The model also performed worse when replacing the BiLSTM layer with Gated Recurrent Unit (GRU) [50], where ResNet-18 combined with GRU achieved an F1 score of 19.34%. It might be because BiLSTM provides more contextual information. From Fig. 4, we can see the proposed model’s performance on each bug combination from no bugs and individual bugs to up to bug combinations of 2 and 3 bug classes in a single moment in the video clips of the test dataset. Our model achieved an F1 score of 88.5% as shown in Fig. 4 when the video clip had a combination of the boundary hole, geometry corruption, and Z-clipping taking place simultaneously. It performed successfully when the clip had no bugs, with an F1 score of 95.6%, and achieved an F1 score of 55.1% on the clips with only the geometry corruption bug.

TABLE IV: Comparison of different model architectures on the test dataset.
<table><tr><td>Model</td><td>F1 Score (%)</td></tr><tr><td>I3D [48]</td><td>19.54</td></tr><tr><td>R3D-18 [49]</td><td>18.01</td></tr><tr><td> $\mathrm { R e s N e t - } 1 8 + \mathrm { G R U }$ </td><td>19.34</td></tr><tr><td> $\mathbf { R e s N e t { - } 1 8 } + \mathbf { B i L S T M } \ ( \mathbf { O u r s } )$ </td><td>85.78</td></tr></table>

## VI. DISCUSSION

From the results in Table IV and Fig. 4, we gained useful insights about the strengths and limitations of our proposed model architecture (ResNet-BiLSTM) on the proposed benchmark dataset. When comparing the use of BiLSTM and GRU for the temporal feature extraction in the models, we saw that the model performed better when using BiLSTM with an F1 score gain of 76.44%, while also outperforming I3D [48] and R3D-18 [49]. The likely reason is the ability of BiLSTM to capture more contextual information while extracting temporal dependencies in both directions across frames, while other models might not leverage the temporal context as effectively.

Additionally, Fig. 4 shows noticeable patterns where the model performs better with videos having bug combinations consisting of the boundary hole bug and moderately well with the Z-clipping bug. We also noticed that geometry corruption and Z-fighting bugs had the worst performance with the proposed model, even in a combination with other perceptual bugs when not including Z-clipping or boundary hole within the combination. This could be because geometry corruption and Z-fighting have subtler visual artifacts that are harder to distinguish than boundary hole or Z-clipping due to more pronounced visual anomalies.

Furthermore, we found that bug combinations with three classes of bugs performed better than with fewer bug combinations, which is counterintuitive. Although more bugs should be harder to detect, our model achieves better performance with more bugs, as more bugs in a video have more pronounced features that can make them easier to detect.

Moreover, the dataset has 65.55% of video clips without any perceptual bugs to simulate real games with fewer instances of perceptual bugs. This could be the reason that the model performs well with no bugs present in the video. The bug occurrence frequency can be much lower depending on the video game. But we chose the mentioned distribution of bugs because of our limitation of needing a significant amount of generated buggy examples with limited computational resources.

On another note, our proposed dataset consists of only three games and five classes of perceptual bugs, which may limit generalization. Also, synthetic bugs injected into the gameplay session of our dataset may lack the nuance of real-world visual bugs, which are caused by broken underlying game logic, specific player interaction sequences, and environmental triggers in the game. Moreover, the perceptual bug combinations begin and end at the same time in the video. Varying durations of each class of buggy events would be beneficial to include for improving the dataset and testing our proposed model further.

## VII. CONCLUSION AND FUTURE WORK

In this work, we introduced a new dataset for multi-label bug detection from perceptual bugs using gameplay videos. The dataset consisted of gameplay videos from three different games and included a combination of 1-3 types of bugs from 5 different classes of perceptual bugs in each frame in the gameplay videos. We compared different models on this dataset and found that our proposed model with ResNet and Bidirectional LSTM achieved an F1 score of 85.78%. Our model demonstrated that temporal context is necessary for capturing features needed to perform bug detection on videos.

Although our dataset includes thousands of video clips from three different games for multi-label bug detection, it is imperative to include more videos from different genres of games in future versions of this dataset. Future research should include an ablation study, real-time inference and temporal localization of these perceptual bug combinations for fulllength gameplay. We hope that ABD for multi-label perceptual bugs through our proposed approach can benefit both game developers and players by reducing quality assurance costs, accelerating development cycles, and improving game quality.

## REFERENCES

[1] S. Ariyurek, A. Betin-Can, and E. Surer, “Automated video game testing using synthetic and humanlike agents,” IEEE Transactions on Games, vol. 13, no. 1, pp. 50–67, 2019.

[2] A. Rollings and E. Adams, Andrew Rollings and Ernest Adams on game design, 1st ed. New Riders, 2003.

[3] A. Albaghajati and M. Ahmed, “Video game automated testing approaches: An assessment framework,” IEEE transactions on games, vol. 15, no. 1, pp. 81–94, 2020.

[4] P. Gutierrez-S´ anchez, M. A. G´ omez-Martın, P. A. Gonz´ alez-Calero, and´ P. P. Gomez-Martın, “A proposal for combining reinforcement learning´ and behavior trees for regression testing over gameplay metrics,” in CEUR Workshop Proceedings, vol. 3082, 2021, pp. 116–127.

[5] C. Politowski, F. Petrillo, and Y.-G. Gueh´ eneuc, “A survey of video game´ testing,” in 2021 IEEE/ACM International Conference on Automation of Software Test (AST), 2021, pp. 90–99.

[6] R. Sinha, “The last of us part 1 pc errors and fixes: Crashes, graphical glitches, and more,” https://gamingbolt.com/the-last-of-us-part-1-pc-e rrors-and-fixes-crashes-graphical-glitches-and-more, 4 2023.

[7] R. Coppola, T. Fulcini, and F. Strada, “Know your bugs: A survey of issues in automated game testing literature,” in GEM ’24, 2024, pp. 1–6.

[8] A. Roque, J. Sotomayor, D. Santiago, and P. Clarke, “A literature review of software testing practices and frameworks in the video gaming industry,” Software Testing Verification and Reliability, vol. 35, p. e70001, 3 2025.

[9] E. Azizi and L. Zaman, “Astrobug: Automatic game bug detection using deep learning,” IEEE Transactions on Games, vol. 16, no. 4, pp. 793– 806, 2024.

[10] J. Bergdahl, C. Gordillo, K. Tollmar, and L. Gisslen, “Augmenting´ automated game testing with deep reinforcement learning,” in 2020 IEEE Conference on Games (CoG), 2020, pp. 600–603.

[11] S. F. Gudmundsson, P. Eisen, E. Poromaa, A. Nodet, S. Purmonen, B. Kozakowski, R. Meurling, and L. Cao, “Human-like playtesting with deep learning,” in CIG ’18, 2018, pp. 1–8.

[12] B. Wilkins and K. Stathis, “Learning to identify perceptual bugs in 3d video games,” arXiv preprint arXiv:2202.12884, 2022.

[13] N. Rifaat, “Code and dataset for multi-label perceptual bug detection,” https://github.com/NahianAlindo/multilabel-perceptual-bug-detection, 2026.

[14] Y. J. Choi, “Providing novel and useful data for game development using usability expert evaluation and testing,” in 2009 Sixth International Conference on Computer Graphics, Imaging and Visualization, 2009, pp. 129–132.

[15] H. Korhonen and E. M. I. Koivisto, “Playability heuristics for mobile games,” in MobileHCI ’06. New York, NY, USA: Association for Computing Machinery, 2006, p. 9–16.

[16] P. Moreno-Ger, J. Torrente, Y. G. Hsieh, and W. T. Lester, “Usability testing for serious games: Making informed design decisions with user data,” Advances in Human-Computer Interaction, vol. 2012, no. 1, p. 369637, 2012.

[17] N. M. Diah, M. Ismail, S. Ahmad, and M. K. M. Dahari, “Usability testing for educational computer game using observation method,” in 2010 international conference on information retrieval & knowledge management (CAMP). IEEE, 2010, pp. 157–161.

[18] Y. Duarte, C. Politowski, and A. T. Endo, “Towards a framework for exploratory testing in video games,” in 2025 IEEE/ACM 9th International Workshop on Games and Software Engineering (GAS), 2025, pp. 25–32.

[19] I. Dobles, A. Mart´ınez, and C. Quesada-Lopez, “Comparing the effort´ and effectiveness of automated and manual tests,” in CISTI ’19, 2019, pp. 1–6.

[20] S. Varvaressos, K. Lavoie, A. B. Masse, S. Gaboury, and S. Hall´ e, “Auto-´ mated bug finding in video games: A case study for runtime monitoring,” in 2014 IEEE Seventh International Conference on Software Testing, Verification and Validation, 2014, pp. 143–152.

[21] S. Iftikhar, M. Z. Iqbal, M. U. Khan, and W. Mahmood, “An automated model based testing approach for platform games,” in MODELS ’15, 2015, pp. 426–435.

[22] G. Lovreto, A. T. Endo, P. Nardi, and V. H. S. Durelli, “Automated tests for mobile games: An experience report,” in SBGames ’18, 2018, pp. 48–488.

[23] E. Guglielmi, S. Scalabrino, G. Bavota, and R. Oliveto, “Using gameplay videos for detecting issues in video games,” Empirical Software Engineering 2023 28:6, vol. 28, pp. 136–, 10 2023.

[24] I. S. W. B. Prasetya, M. Voshol, T. Tanis, A. Smits, B. Smit, J. v. Mourik, M. Klunder, F. Hoogmoed, S. Hinlopen, A. v. Casteren, J. v. d. Berg, N. G. Prasetya, S. Shirzadehhajimahmood, and S. G. Ansari, “Navigation and exploration in 3d-game automated play testing,” in Proceedings of the 11th ACM SIGSOFT International Workshop on Automating TEST Case Design, Selection, and Evaluation, ser. A-TEST 2020. New York, NY, USA: Association for Computing Machinery, 2020, p. 3–9.

[25] C. Guerrero-Romero, S. M. Lucas, and D. Perez-Liebana, “Using a team of general ai algorithms to assist game design and testing,” in CIG ’18, 2018, pp. 1–8.

[26] A. Albaghajati and M. Ahmed, “A co-evolutionary genetic algorithms approach to detect video game bugs,” Journal of Systems and Software, vol. 188, p. 111261, 2022.

[27] Y. Zhao, E. Tang, H. Cai, X. Guo, X. Wang, and N. Meng, “A lightweight approach of human-like playtest for android apps,” in SANER ’22, 2022, pp. 309–320.

[28] Y. Zheng, X. Xie, T. Su, L. Ma, J. Hao, Z. Meng, Y. Liu, R. Shen, Y. Chen, and C. Fan, “Wuji: Automatic online combat game testing using evolutionary deep reinforcement learning,” in ASE ’19, 2019, pp. 772–784.

[29] G. Rani, U. Pandey, A. A. Wagde, and V. S. Dhaka, “A deep reinforcement learning technique for bug detection in video games,” International Journal of Information Technology 2022 15:1, vol. 15, pp. 355–367, 8 2022.

[30] F. Xue, “Automated mobile apps testing from visual perspective,” in Proceedings of the 29th ACM SIGSOFT International Symposium on Software Testing and Analysis, 2020, pp. 577–581.

[31] J. Pfau, J. D. Smeddinck, and R. Malaka, “Automated game testing with icarus: Intelligent completion of adventure riddles via unsupervised solving,” in Extended Abstracts Publication of the Annual Symposium on Computer-Human Interaction in Play, ser. CHI PLAY ’17 Extended Abstracts. New York, NY, USA: Association for Computing Machinery, 2017, p. 153–164.

[32] B. Wilkins and K. Stathis, “World of bugs: A platform for automated bug detection in 3d video games,” in 2022 IEEE Conference on Games (CoG), 2022, pp. 520–523.

[33] C. Paduraru, “A state-aware, hierarchical deep learning framework for automated visual glitch detection in games,” Engineering Applications of Artificial Intelligence, vol. 166, p. 113497, 2026.

[34] E. Azizi and L. Zaman, “Automatic bug detection in games using lstm networks,” in 2023 IEEE Conference on Games (CoG), 2023, pp. 1–4.

[35] D. Lin, C.-P. Bezemer, and A. E. Hassan, “Identifying gameplay videos that exhibit bugs in computer games,” Empirical Software Engineering, vol. 24, no. 6, pp. 4006–4033, 2019.

[36] C. Ling, K. Tollmar, and L. Gisslen, “Using deep convolutional neural´ networks to detect rendered glitches in video games,” in Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, vol. 16, no. 1, 2020, pp. 66–73.

[37] F. Yi, S. Abdelfattah, W. Huang, and A. Brown, “A hybrid co-finetuning approach for visual bug detection in video games,” in Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, vol. 21, no. 1, 2025, pp. 163–173.

[38] M. Savran and H. Bulut, “Real-time error detection in digital games based on the yolo v10 model,” in IDAP ’24, 2024, pp. 1–5.

[39] Bottom Hat, “Open Nights by Bottom Hat - Game Jolt.” [Online]. Available: https://gamejolt.com/games/OpenNights/907131

[40] “Unity3D-Mario-Kart-Racing-Game.” [Online]. Available: https://gith ub.com/Ishaan35/Unity3D-Mario-Kart-Racing-Game

[41] “FPS Microgame.” [Online]. Available: https://learn.unity.com/project/ fps-template

[42] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[43] M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” IEEE Transactions on Signal Processing, vol. 45, pp. 2673– 2681, 1997.

[44] L. Tran, T. Hoang, T. Nguyen, H. Kim, and D. Choi, “Multi-model long short-term memory network for gait recognition using window-based data segment,” IEEE Access, vol. 9, pp. 23 826–23 839, 2021.

[45] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.

[46] T. Ridnik, E. Ben-Baruch, N. Zamir, A. Noy, I. Friedman, M. Protter, and L. Zelnik-Manor, “Asymmetric loss for multi-label classification,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 82–91.

[47] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss´ for dense object detection,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2980–2988.

[48] J. Carreira and A. Zisserman, “Quo vadis, action recognition? a new model and the kinetics dataset,” in proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 6299–6308.

[49] D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri, “A closer look at spatiotemporal convolutions for action recognition,” in CVPR ’18, 2018, pp. 6450–6459.

[50] K. Cho, B. Van Merrienboer, C¸ . Gulc¸ehre, D. Bahdanau, F. Bougares,¨ H. Schwenk, and Y. Bengio, “Learning phrase representations using rnn encoder–decoder for statistical machine translation,” in Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), 2014, pp. 1724–1734.