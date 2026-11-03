# Improving Financial Cybersecurity Through Fraud Transaction with Deep Learning-Based Advanced Models

Harish Kant Pathak

Independent Researcher

harish.pathak@gmail.com

ORCID-0009-0004-4390-4170

Rahul Kumar

Independent Researcher

logtorahul@gmail.com

ORCID-0009-0004-4390-4170

Abstract—The quick digitalization of financial services has greatly exposed financial operations to cyber threats. Conventional rule-based cybersecurity controls do not have the strength to protect against the changing attack schemes, such as fraud, identity theft, adversarial manipulation, and advanced persistent threats. The research presents a deep learning-based approach to detect fraudulent transactions by using the Fraud Detection Transactions Dataset. The proposed solution includes a large amount of data preparation, such as outlier detection, feature engineering, label encoding, feature scaling, and class imbalance control with SMOTE. There are two types of categorization models: artificial neural networks (ANNs) and wide-and-deep neural networks. Experimental results show that the Wide & Deep model outperforms the ANN in acc (0.9846), prec (0.9634), rec (0.9895), F1-score (0.9763), and ROC-AUC (0.99). Fraud patterns of complex nature can be detected by the linear and DL elements. In addition, an ablation study is applied to show the effects of the various optimizers, and their comparison with the current models proves the power of the developed approach. The results demonstrate that cybersecurity systems for financial transactions may greatly benefit from using the Wide and Deep architecture to enhance the precision and resilience of fraud detection.

Keywords—artificial intelligence, cybersecurity, financial transactions, fraud detection, deep learning.

## I. INTRODUCTION

The financial ecosystem of the world has shifted to digital platforms, such as online banking, mobile payment systems, e-wallets, and cryptocurrencies. Although digital transformation brings benefits such as greater efficiency and accessibility to operations, it exposes a company to more cyber threats [1]. There is a wide variety of illegal behaviors that go under the umbrella phrase "banking fraud," including phishing, account takeover, credit card fraud, check fraud, and identity theft. Financial cyber fraud has become a major menace in the financial sector in the digital age, affecting individuals, companies, and governments around the world. Cybercriminals utilize sophisticated techniques to take advantage of flaws in financial systems, which may lead to significant financial losses, harm to one's reputation, and even legal consequences [2]. A growth in online transactions and the digitalization of financial services has brought attention to the need for reliable fraud detection systems.

Financial fraud involves a variety of strategies that defraud customers of their money, such as credit card fraud, money laundering, and insurance fraud [3]. Despite the significant breakthroughs that artificial intelligence (AI)-based fraud detection has shown, it has certain inherent limits [4]. Model performance may be adversely affected by a lack of generalizability, bias resulting from unbalanced datasets, and reliance on data quality. Adversarial attacks also make it more difficult to detect financial fraud as they imply that criminals manipulate the transaction data to avoid detection systems [5][6]. Overestimated false positive rates and long reaction times are consequences of conventional fraud detection systems' failure to adapt to evolving fraud tactics. These systems depend on rule-based algorithms and human investigations [7]. One possible solution could lie in evolving methods of detection, notably with the help of DL(Deep Learning) and ML (Machine Learning) [8]. These techniques can enhance the effectiveness of fraud protection systems since they can process large amounts of data at any given moment, detect intricate patterns, and evolve to new approaches of fraud [9]. DL is a branch of machine learning, and it is able to identify complex fraud patterns by applying multi-layered neural networks to derive abstract models using data. Models based on Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTM) networks, and Natural Language Processing (NLP) have shown promise for real-time fraud detection. This hybrid structure fuses a wide (linear) aspect of feature interaction memorization and a deep (nonlinear) aspect of generalization. Incorporating these factors can go a long way in enhancing the ability of the model to detect both simple and sophisticated trends in the data of financial transactions. The study's most significant contributions include the following:

Developed a universal fraud detection model that included Synthetic Minority, feature engineering, and preprocessing. Using the SMOTE Over-sampling Technique to Fix Class Inequality.

Designed and trained Artificial Neural Network (ANN) and Models for fraud classification using Wide and Deep Neural Networks.

• To determine how various optimizers impacted the model's performance, an optimizer ablation study was conducted.

The model's performance was assessed using F1-score, ROC-AUC, recall, accuracy, precision, and other measures.

## A. Motivation of the study

A novel level of risk has been introduced into the financial sector by the rapid expansion of digital payment systems such as internet banking, mobile payments, and shopping online, and existing traditional rule-based methods of detecting fraudsters are not sufficient because they are too slow to adjust to the constantly changing fraud behavior and too powerless to work with heavy data volumes and imbalanced data. Thus, intelligent and scalable solutions that can correctly identify complex and dynamic fraud behaviors in real-time are highly needed, hence the desire to apply advanced deep learning methods in this study.

## B. Organization of the Paper

An overview of the project is as follows: In Section II provides a synopsis of the relevant literature and current methods for fraud in financial transactions. The suggested approach is outlined in Section III. The findings, analysis, comparison with existing models, and recommendations for future study are all included in Section IV. Section V concludes the investigation.

## II. LITERATURE REVIEW

Recent research has examined the effectiveness of ML and DL approaches in identifying instances of financial fraud. S. Mallikarjunaiah et al. (2025) suggested method would combine CNN and LSTM networks to find intricate fraud patterns in transaction data, while SVM will be effective when applied to structured data. The system has an accuracy of 96, which is very high compared to the traditional techniques like random forest and SVM, thus it is very strong in detecting real-time fraud [10]. Ebnezer et al. (2025) proposed DL method utilizes Bi-LSTMs to identify fraudulent financial transactions in real-time. The model recorded high performance with an acc of 91.8, F1score 90.5 and PR-AUC of 0.94 compared to the traditional ML algorithms like the LR, RF, and XGBoost [11]. Additionally, K. P. Kumar et al. (2025) proposed a DL model that employs an Attention Mechanism and Bi-LSTM to increase precision of identifying fraud in Ethereum transactions. On a Kaggle Ethereum dataset, the model achieved 97% acc, 97% prec, 97% rec, and a 97% F1score, much higher than existing works [12].

R. Sharma and S. Gupta (2024) explored the potential of improving fraud detection in online banking with the help of DL algorithms. The LSTM model outperformed others in terms of prec, acc, F1score, recognition, and AUC-ROC of 0.936, 95.8 and 82.7, respectively. The CNN model demonstrated the following results: 95.2% accuracy, 84.1% precision, and 77.8% recall, in terms of accuracy, precision, recall, F1score, and AUC, respectively [13]. In the study conducted by R. Sharma and A. Sharma (2024), provided significant volume of transactional data in their research, and a combination of supervised and unsupervised learning approaches was employed to determine problematic ones. This method also automates data cleansing. As shown by its 0.972 AUC, 93.7% sensitivity, 97.5% specificity, and 95.8% accuracy, the RNN architecture outperforms the CNN model [14]. N. R. Shanbhog et al. (2024) compared advanced tools of analysis in the effective separation of legitimate and deceptive patterns in financial data. The three models - XGBoost, ANN, and R-GCN demonstrated good results, with accuracies 97.0%, 94.0%, 98.5% respectively [15].

P. Y. Prasad et al.'s (2023) main objective is to develop a ML-based fraud detection solution. The basis of algorithmic frameworks like XGBoost, RF, SVM, DT, and LR is machine learning. The algorithms' output is impacted by the F1score, rec, acc, and prec. Values of 99.9%, 85.71%, 93%, and 98% have been used to modify the Area Under Curves (AUC) [16]. Current fraud detection techniques based on CNNs, LSTMs, and other sophisticated models tend to target more sophisticated patterns however they do not merge the linear and non-linear relations well. The problem of class imbalance is not well addressed in much research, and as a consequence, the findings are skewed since they are not examined using a variety of criteria. In addition, there has been a lack of research on optimizer analysis and model comparison, which highlights the need for a stronger and more versatile model to improve fraud detection overall.

## III. METHODOLOGY

Methodology and process for the suggested fraud detection system are shown in Fig. 1. Preprocessing the data includes tasks such as identifying outliers, engineering features, encoding labels, and scaling features. The procedure begins with a dataset of fraud detection transactions. To correct the class imbalance, the data is processed and then split into a training set and a test set using SMOTE. Then ANN and Wide and Deep models are applied to classification. The models tested according to recall, precision, accuracy, F1score, and ROC-AUC, and ablation study conducted, which result in the final results.

![](images/29313e5f39058bf83b827abb325a97098fae4f29fac32bd342de106ea1667bfa.jpg)  
Fig. 1. Proposed Methodology for fraud detection system.

The proposed methodology steps and implementation strategy are elaborate below:

## A. Data Collection and Visualization

The dataset that utilized in this research is Fraud Detection Transactions Dataset available on Kaggle and includes 50,000 transaction records that have 21 features. The dataset shows a substantial disparity between genuine and fraudulent transactions, with 76% of the transactions being legitimate and 24% being fraudulent. There are numerical and nominal elements that indicate the user's behavior and transaction pattern, such as the amount of the transaction, the kind of the transaction, the account balance, the location, the merchant category, and the risk score. There is some visualization like heatmap, distribution graph etc. discuss below:

Fig. 2 shows the distribution of various significant variables in the fraud detection data, such as TransactionAmount, AccountBalance, IP AddressFlag, PreviousFraudulentActivity, DailyTransactionCount, AverageTransactionAmount (7 days), FailedTransaction count, CardAge, TransactionDistance, RiskScore, weekend indicator and fraud label. The plots display that a majority of the numerical characteristics including transaction amount and transaction distance are right skewed meaning that there are high-value outliers. Binary characteristics such as IPaddressFlag, PreviousFraudulentActivity, Is\_Weekend and FraudLabel have a high rate of class imbalance with more normal cases than the fraudulent ones.

![](images/462bf93ac0bd263f1033cf917662f441cab80aad85f13f879bff3cea1554f126.jpg)  
Fig. 2. Distribution of Key Transaction Features.

![](images/8d1e5f1ff2900489d87ca68d6e2b601f3b12f39657dd60af06f67ce0bf70747e.jpg)  
Fig. 3. Heatmap of Features and Values Correlation.

Fig. 3 heatmap gives correlation between all data features. Correlation is low between most features and this means that multicollinearity is low. There are increased correlations observed among some derived features such as Transaction Amount, Amount Riskratio and High amount. Most features are weakly correlated with the target variable Fraud\_Label, which leads to the importance of complex models to ensure an ability to capture complex patterns.

## B. Data Preprocessing

Data is pre-processed to make sure it is prepared for model training. This comprised data cleaning, feature engineering and encoding of categorical variables as well as feature scaling. Also, outlier detection is done to detect extreme values of numerical features especially in the variables of Transaction\_Amount and other associated variables.

## C. Feature Engineering

The feature engineering is done to add new meaningful features to the data based on the existing attributes. The amount of risk ratio (Transaction Amount)/(Risk Score + 1) is a continuous feature Amount Risk Ratio which is formed to provide the relative intensity of risk in a transaction without dividing by zero. Also, a binary attribute High\_Amount is added, to reveal the high-value transactions, with the transactions beyond the 90th percentile of the Transaction\_Amount called 1 and the rest 0. In addition to temporal features that have been previously extracted (i.e., Hour, Day, Month, DayOfWeek, and Is\_Night), they are engineered features that make the model more effective in detecting latent patterns of fraud and lead to making the model better in terms of classification.

## D. Label Encoding for Categorical Columns

The Label Encoding is used to transform categorical features into numerical values. Under this technique, a different integer is given to each distinct category under a feature. The data incorporate features such as Transaction Type, Type of device, Location, Merchant Category, Card Type and method for authentication and verification and verify that data is structured properly for model training.

## E. Feature Scaling

Feature scaling is carried out using the StandardScaler approach to standardize the numerical data such that the model represents each variable equally. The model's overall performance and convergence are enhanced by standardization, which comprises changing the data to a mean of 0 and a StandardDeviation of 1. The transformation can be defined as Equation (1):

$$
Z = { \frac { X - \mu } { \sigma } }\tag{}
$$

Where, � is original FeatureValue, � is mean of the feature and � is StandardDeviation of the feature.

## F. Data Splitting

The dataset is stratified-sampled and split 75:25 across the training and testing sets to keep the class distribution constant.

There are 37,500 samples with 24 features in the training set and 12,500 samples with 24 characteristics in the testing set.

## G. Handling Class Imbalance

The significant imbalance in the classes included in the dataset is addressed by the SMOTE. SMOTE is a method for balancing data that improves models' ability to handle imbalanced classification issues by creating synthetic minority examples via interpolation between existing instances.

![](images/4dc6adc828fd066808eabfd1af51f6182d17ff50a0ad2f3231e4c26e3144ba67.jpg)  
Fig. 4. Distribution of classes before and after SMOTE.

Fig. 4 the left plot, the initial imbalanced data is presented, in which the non-fraud group (0) has 25,450 samples and the fraud group (1) has 12,050 samples. The right plot shows the balanced dataset, which is balanced after the use of SMOTE in both classes (0 and 1) 25,450 samples of each of the classes are equally represented, which means that the model is better trained, and there is less bias with the majority class.

## H. Proposed Models for fraud detection

The DL models include the ones listed below.

## 1) Artificial Neural Network (ANN)

A sequential architecture with layers of fully connected architecture is created to generate an ANN to detect fraud [17]. After an input layer to identify complex patterns in the data, the model consists of three hidden layers with ReLU activation, each having 128, 64, and 32 neurons. There are dropout layers (0.1 and 0.2) that followed the first two hidden layers to avoid overfitting. Binary classification uses the sigmoid activation function on the output layer. A BatchSize of 32 and 30 epochs is applied with a validation split of 20% to train.

## 2) Wide and Deep Neural Network

The Wide and Deep Neural Network uses memory and generalizability by modeling a deep neural network and a long linear framework as a fraud detection model. The wide component uses linear activation and a dense layer of 64 neurons to learn basic feature interactions. In contrast, the deep component employs ReLU activation in multiple dense layers of 128, 128, and 64 neurons to learn complex patterns. To further stabilize and regularize the model, it also makes use of BatchNormalization and dropout (0.1). The two parts are joined through concatenation and binary classification through a sigmoid output layer is used. They used the AdamOptimizer to train their model, using an initial learning rate of 0.001, BinaryCross-entropy loss as their evaluation measure, a BatchSize of 64, 50 epochs, and a valid test size of 50%.

## 3) Experimental setup and Model Evaluation

The following components make up the computational ecosystem: Powerful hardware: Includes a 32GB RAM stick, an Intel Core i9 CPU, and a 16GB NVIDIA GPU. The required software packages are TensorFlow 2.0, Scikit-Learn 0.24, Pandas, NumPy, and at least Python 3.8%. A wide range of evaluation metrics is utilized to evaluate how well fraud detection systems perform. When comparing all anticipated frauds to the proportion of accurately expected frauds, it gets the forecast's accuracy; when comparing this percentage to all projected frauds, it gets the forecast's precision. The sensitivity or recall of a model is a measure of its ability to identify real instances of fraudulent transactions. Rec and prec are both taken into account by the F1score. To evaluate a model's capabilities in class distinction across different thresholds, the ROC-AUC is a helpful statistic. All of them are computed using Equations (2) through (5):

$$
\begin{array} { r } { A c c = \frac { T P + T N } { T P + F P + T N + F N } } \end{array}\tag{2}
$$

$$
\begin{array} { r } { P r e c = \frac { T P } { T P + F P } } \end{array}\tag{3}
$$

$$
\begin{array} { r } { R e c = \frac { T P } { T P + F N } } \end{array}\tag{4}
$$

$$
\begin{array} { r } { F 1 - S c o r e = 2 * \frac { ( P r e c i s i o n * R e c a l l ) } { P r e c i s i o n + R e c a l l } } \end{array}\tag{5}
$$

Here, TP indicates that fraud cases have been appropriately identified, TN indicates that genuine transactions have been accurately classified, FP indicates that fraud instances have been reported incorrectly, and FN indicates that fraud cases have not been detected.

## IV. RESULT ANALYSIS AND DISCUSSION

Table I displays the results of comparing the suggested ANN and Wide & Deep Neural Network models' performance on the fraud detection dataset. The results demonstrate that when comparing the two models, the Wide & Deep Neural Network model outperforms the ANN in terms of the majority of assessment criteria, including acc (0.9846), prec (0.9634), rec (0.9895), and the F1score (0.9763). Both of them are good at classification having the same ROC-AUC at 0.9990. The ANN may have a little higher recall (0.9983), but the Wide and Deep model outperforms it in all three metrics, making it the superior fraud detector.

TABLE I. PERFORMANCE OF PROPOSED MODELS FOR FRAUD TRANSACTIONS USING FRAUD DETECTION TRANSACTIONS DATASET
<table><tr><td rowspan=1 colspan=1>Matrix</td><td rowspan=1 colspan=1>ANN</td><td rowspan=1 colspan=1>Wide &amp; Deep Neural Network</td></tr><tr><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>0.9667</td><td rowspan=1 colspan=1>0.9846</td></tr><tr><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>0.9074</td><td rowspan=1 colspan=1>0.9634</td></tr><tr><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>0.9983</td><td rowspan=1 colspan=1>0.9895</td></tr><tr><td rowspan=1 colspan=1>F1 Score</td><td rowspan=1 colspan=1>0.9507</td><td rowspan=1 colspan=1>0.9763</td></tr><tr><td rowspan=1 colspan=1>ROC-AUC</td><td rowspan=1 colspan=1>0.9990</td><td rowspan=1 colspan=1>0.9990</td></tr></table>

![](images/4f4115508f67791a5545e0de551ce66392c00610f3b45ff262576873c9a61f32.jpg)

![](images/174b1726062094cd38058d60ed2a94d701b40521fb8967c8238236a17cbdcba7.jpg)  
Fig. 5. Accuracy and loss graph for ANN model.

Fig. 5 displays the ANN model's accuracy and loss throughout epochs during training and validation. The accuracy plot shows that there is a gradual increase in training accuracy with constant changes in validation accuracy which is high and indicates good learning. The loss plot depicts steady reduction in training and slow reduction in validation loss following small fluctuations, which suggests successful training with little overfitting.

![](images/0b86566ce265b0df619d4601d919ab6ef1719fb5b2168f797a77bb869e10b3d3.jpg)

![](images/f786fec749d6fb8de06a0905e55704d1a73c2cff00554b682da175ce11e1e642.jpg)  
Fig. 6. ANN Model's Confidence Matrix and ROC Curv.

Fig. 6 displays the ANN model's ROC and confusion. The Confusion Matrix presents 8223 TN, 4017 TP, 0 FP, and 260 FN, which means that the classifier has a good performance with minimal mistakes. According to the ROC curve, the model has an impressive AUC of 0.999, which means it can tell the difference between good and bad transactions.

![](images/70acba7010012afa99660d185cb3f5b1a1558974319ee089d7d305dd87c0acd9.jpg)

![](images/4eda6be604bd90f025f29fde1a312686a54c2e2813c90a3949c87c728849790b.jpg)  
Fig. 7. Accuracy and loss graph for Wide & Deep Neural Network model.

Fig. 7 displays the Wide & Deep model's training and validation accuracies as well as its losses relative to epochs. The training accuracy rapidly attains high levels whereas the validation accuracy steadily attains low variations. The training loss is low, but validation loss is also not constant, which means that it is a high-learning model with minimal overfitting but with an overall good performance.

![](images/38b555970fd104f73d15858d0833e6cc64f4db04b113441e37bb303a8abf6970.jpg)

![](images/12bd9d73692d00234686636d971ac3390f178a7f3aabc6f1b84a1e4c9622a6ae.jpg)  
Fig. 8. Confusion matrix and ROC of Wide & Deep Neural Network model.

The Confusion Matrix and ROC curve for the Deep and Wide models are shown in Fig. 8. According to the Confusion Matrix, the low-error classification accuracy is represented by 8448 TN, 3879 TP, 138 FP, and 35 FN. The AUC of 0.999 for the ROC indicates that it is very good at differentiating between legitimate and fraudulent transactions.

## A. Ablation Study

Table II shows the results of Wide and Deep model involving the comparative performance of various optimizers using the optimizer ablation study. The findings show that SGD (lr = 0.01, momentum = 0.9) gives the most overall performance with highest acc (0.9649), prec (0.9895), F1score (0.9737), MCC (0.9220) and Kappa (0.9210). RMSprop is also a competitor to balanced precision and recall, with Adam also showing strong performance, Adam with slightly reduced performance. Adamax presents relatively weaker outcomes of all optimizers. SGD, in general, is a much better optimizer of this model, and it is more stable and generalizes better. Additionally, by using this Wide and Deep design, it may take advantage of stable learning dynamics, which allows for the identification of both linear and nonlinear correlations without overfitting the neural network model's training data..In practical terms, this makes the model more reliable and effective since the implementation of SGD provides greater robustness in its operation.

TABLE II. OPTIMIZER ABLATION STUDY FOR WIDE AND DEEP MODEL PERFORMANCE
<table><tr><td rowspan=1 colspan=1>Optimizer</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>Prec</td><td rowspan=1 colspan=1>Rec</td><td rowspan=1 colspan=1>F1</td><td rowspan=1 colspan=1>MCC</td><td rowspan=1 colspan=1>Kappa</td></tr><tr><td rowspan=1 colspan=1>Adam (lr =0.001)</td><td rowspan=1 colspan=1>0.9567</td><td rowspan=1 colspan=1>0.9738</td><td rowspan=1 colspan=1>0.9620</td><td rowspan=1 colspan=1>0.9679</td><td rowspan=1 colspan=1>0.9018</td><td rowspan=1 colspan=1>0.9017</td></tr><tr><td rowspan=1 colspan=1>SGD (lr =0.01,momentum= 0.9)</td><td rowspan=1 colspan=1>0.9649</td><td rowspan=1 colspan=1>0.9895</td><td rowspan=1 colspan=1>0.9584</td><td rowspan=1 colspan=1>0.9737</td><td rowspan=1 colspan=1>0.9220</td><td rowspan=1 colspan=1>0.9210</td></tr><tr><td rowspan=1 colspan=1>RMSprop(lr = 0.001)</td><td rowspan=1 colspan=1>0.9641</td><td rowspan=1 colspan=1>0.9731</td><td rowspan=1 colspan=1>0.9740</td><td rowspan=1 colspan=1>0.9735</td><td rowspan=1 colspan=1>0.9179</td><td rowspan=1 colspan=1>0.9179</td></tr><tr><td rowspan=1 colspan=1>Adamax (lr= 0.002)</td><td rowspan=1 colspan=1>0.9592</td><td rowspan=1 colspan=1>0.9820</td><td rowspan=1 colspan=1>0.9573</td><td rowspan=1 colspan=1>0.9695</td><td rowspan=1 colspan=1>0.9085</td><td rowspan=1 colspan=1>0.9079</td></tr></table>

## B. Comparative Study and Discussion

Table III presents a comparison of the proposed and existing models' performance across various datasets. The results show that the suggested Wide and Deep Neural Network performs best overall in the dataset of fraud detection transactions, with greatest acc (0.9846), prec (0.9634), rec (0.9895), and F1score (0.9763), followed by the ANN model. The ANN model also performs well, achieving high recall (0.9983) and a balanced F1-score of 0.9507.

TABLE III. FRAUD TRANSACTION MODEL PERFORMANCE COMPARISON OF PROPOSED AND EXISTING MODELS
<table><tr><td rowspan=1 colspan=1>Models</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>Prec</td><td rowspan=1 colspan=1>Rec</td><td rowspan=1 colspan=1>F1</td></tr><tr><td rowspan=1 colspan=1>ANN</td><td rowspan=1 colspan=1>0.9667</td><td rowspan=1 colspan=1>0.9074</td><td rowspan=1 colspan=1>0.9983</td><td rowspan=1 colspan=1>0.9507</td></tr><tr><td rowspan=1 colspan=1>Wide &amp; DNN</td><td rowspan=1 colspan=1>0.9846</td><td rowspan=1 colspan=1>0.9634</td><td rowspan=1 colspan=1>0.9895</td><td rowspan=1 colspan=1>0.9763</td></tr><tr><td rowspan=1 colspan=1>NB[18]</td><td rowspan=1 colspan=1>0.9778</td><td rowspan=1 colspan=1>0.0419</td><td rowspan=1 colspan=1>0.8469</td><td rowspan=1 colspan=1>0.1163</td></tr><tr><td rowspan=1 colspan=1>DT [19]</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.909</td><td rowspan=1 colspan=1>0.919</td><td rowspan=1 colspan=1>0.928</td></tr><tr><td rowspan=1 colspan=1>Logistics regression[20]</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.08</td></tr><tr><td rowspan=1 colspan=1>Random Forest[20]</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.17</td></tr><tr><td rowspan=1 colspan=1>XGBoost[21]</td><td rowspan=1 colspan=1>0.926</td><td rowspan=1 colspan=1>0.958</td><td rowspan=1 colspan=1>0.902</td><td rowspan=1 colspan=1>0.929</td></tr><tr><td rowspan=1 colspan=1>SVM[22]</td><td rowspan=1 colspan=1>0.957</td><td rowspan=1 colspan=1>0.669</td><td rowspan=1 colspan=1>0.814</td><td rowspan=1 colspan=1>0.734</td></tr></table>

When it comes to metrics, the decision tree and XGBoost models have performed around average, but the Support Vector Machine model has done really well, with respectable recall and accuracy. Unfortunately, although having respectable recalls and accuracies, the random forest, logistic regression, and Naive Bayes models have all failed miserably when it comes to precision.

## C. Areas for Improvement

Although the suggested models have shown good results, there are some aspects that can be enhanced in future studies. The present research has been based on one data set, which might not be generalizable to other real-life financial systems. It is also possible to add real-time streaming data to the models to have live fraud detection. Also, to be useful in financial institutions, it is necessary to enhance the model interpretability. Ensemble or hybrid models would also increase performance and robustness. Lastly, looking into the complex methods of dealing with extreme imbalance classes and adversarial fraud patterns may enhance the stability of the system setup in changing conditions.

## V. CONCLUSION AND FUTURE WORK

This research provided a thorough framework for fraud detection that uses Wide & Deep Neural Network architecture. It aims to overcome the constraints of standard deep learning methods that only use one route. The implementation of the proposed solution showed that the framework is capable of detecting transactions that are fraudulent with great accuracy (98.46%) and high F1-score (97.63%). Where the ANN was the gold standard, the work outperformed it. Improvements in system quality have been achieved via feature engineering, SMOTE (a method for dealing with class imbalance), and optimization algorithm analysis. As for future research, there is a great opportunity to continue work on creating the system for real-time detection of frauds using stream processing methods. In addition, the application of explainable artificial intelligence techniques may be considered, which can help in ensuring regulatory compliance. The creation of adaptive learning strategies will also enable the system to remain in par with the changes in the way fraudsters operate. The other significant activity would be to engage multi-source data (behavioral data, graph analytics) in the process.

## REFERENCES

[1] S. B. Shah, “Advancing Financial Security with Scalable AI: Explainable Machine Learning Models for Transaction Fraud Detection,” in 2025 4th International Conference on Distributed Computing and Electrical Circuits and Electronics (ICDCECE), 2025, pp. 1–7. doi: 10.1109/ICDCECE65353.2025.11034838.

[2] V. Pal and S. K. Chintagunta, “Transformer-Based Graph Neural Networks for RealTime Fraud Detection in Blockchain Networks,” Int. J. Adv. Res. Sci. Commun. Technol., vol. 3, no. 3, pp. 1401–1411, Jul. 2023, doi: 10.48175/IJARSCT-11978Y.

[3] K. Ramkumar, S. Kilaru, P. Preethi, A. Nerella, S. Kilaru, and G. G. Battu, “A Temporal Graph Neural Network Approach for Deep Fraud Detection in Real-Time Financial Transactions,” in The 16th International IEEE Conference on Computing, Communication and Networking Technologies (ICCCNT), 2025.

[4] S. Kakkar, “Explainable AI Models for Credit Risk Scoring in Banking: Balancing Accuracy and Regulatory Transparency,” Int. J. Financ. Data Sci., vol. 3, no. 2, pp. 1–6, Aug. 2025, doi: 10.34218/IJFDS\_03\_02\_001.

[5] D. Yeligandla, “Enhancing Financial Cybersecurity: A Review of Secure Web Applications for Fraud Prevention and Data Protection,” J. Comput. Anal. Appl., vol. 33, no. 8, pp. 2653–2671, 2024.

[6] J. W. Sajja, G. B. Komarina, and N. K. R. Choppa, “The Convergence of Financial Efficiency and Sustainability in Enterprise Cloud Management,” J. Comput. Sci. Technol. Stud., vol. 7, no. 4, pp. 964– 992, May 2025, doi: 10.32996/jcsts.2025.7.4.110.

[7] Y. Chen, C. Zhao, Y. Xu, C. Nie, and Y. Zhang, “Deep Learning in Financial Fraud Detection: Innovations, Challenges, and Applications,” Data Sci. Manag., Aug. 2025, doi: 10.1016/j.dsm.2025.08.002.

[8] N. Prajapati, “The Role of Machine Learning in Big Data Analytics: Tools, Techniques, and Applications,” ESP J. Eng. Technol. Adv., vol. 5, no. 2, pp. 16–22, 2025, doi: 10.56472/25832646/JETA-V5I2P103.

[9] S. Singamsetty, “Efficacy of Data Governance a Cutting Edge Approach to Ensuring Data Quality in Machine Learning for Banking

Industry,” in 2024 2nd International Conference on Signal Processing, Communication, Power and Embedded System (SCOPES), IEEE, Dec. 2024, pp. 1–7. doi: 10.1109/SCOPES64467.2024.10991944.

[10] S. Mallikarjunaiah, E. A. Rao, S. Girinath, S. H, V. S. P. Kandi, and M. K. Chakravarthi, “Explainable AI-Powered Fraud Detection Device for Secure Financial Transactions,” in 2025 2nd International Conference on Artificial Intelligence and Knowledge Discovery in Concurrent Engineering (ICECONF), IEEE, Oct. 2025, pp. 1–6. doi: 10.1109/ICECONF65644.2025.11379627.

[11] Ebnezer, S. Mohan, S. Subalya, and T. Jansi Rani, “AI-Powered Fraud Detection in UPI and Digital Wallet Transactions Using Deep Learning Models,” in 2025 IEEE 5th International Conference on ICT in Business Industry &amp; Government (ICTBIG), IEEE, Dec. 2025, pp. 1–6. doi: 10.1109/ICTBIG68706.2025.11323826.

[12] K. P. Kumar, S. Lubna, and P. T. Kumar, “Fraud Detection in Ethereum Transactions Using Bi-Directional LSTM and Attention Mechanism,” in 2025 5th International Conference on Intelligent Technologies (CONIT), IEEE, Jun. 2025, pp. 1–5. doi: 10.1109/CONIT65521.2025.11167814.

[13] R. Sharma and S. Gupta, “Strategic Deployment of Deep Learning Algorithms to Mitigate Fraud in Online Finance,” in Proceedings of International Conference on Circuit Power and Computing Technologies, ICCPCT 2024, 2024. doi: 10.1109/ICCPCT61902.2024.10673115.

[14] R. Sharma and A. Sharma, “Combatting Digital Financial Fraud through Strategic Deep Learning Approaches,” in 2024 2nd International Conference on Sustainable Computing and Smart Systems (ICSCSS), 2024, pp. 824–828. doi: 10.1109/ICSCSS60660.2024.10625249.

[15] N. R. Shanbhog, K. S. Totad, A. R. Hanchinal, and A. P. Bidargaddi, “Fraud Detection in Financial Transactions Using Deep Learning Approach: A Comparative Study,” in 2024 5th International Conference for Emerging Technology (INCET), IEEE, May 2024, pp. 1–7. doi: 10.1109/INCET61516.2024.10593486.

[16] P. Y. Prasad, A. S. Chowdarv, C. Bavitha, E. Mounisha, and C. Reethika, “A Comparison Study of Fraud Detection in Usage of Credit Cards using Machine Learning,” in 7th International Conference on Trends in Electronics and Informatics, ICOEI 2023 - Proceedings, 2023. doi: 10.1109/ICOEI56765.2023.10125838.

[17] S. A. Pushkala, “Financial Fraud Identification Using Graph Neural Network And LSTM With Autoencoder-Based Data Refinement,” J. Int. Cris. RISK Commun. Res., vol. 9, no. 1, pp. 198–214, 2026.

[18] B. O. O., K. J. A., and A. A. A., “Credit Card Fraud Detection Using Machine Learning Algorithms,” Br. J. Comput. Netw. Inf. Technol., vol. 7, no. 3, pp. 1–35, Jul. 2024, doi: 10.52589/BJCNIT-YDIJNXG2.

[19] S. B. Shah, “Improving Financial Fraud Detection System with Advanced Machine Learning for Predictive Analysis and Prevention,” Int. J. Sci. Res. Comput. Sci. Eng. Inf. Technol., vol. 10, no. 6, pp. 2451–2463, Nov. 2024, doi: 10.32628/CSEIT24861147.

[20] J. K. Afriyie et al., “A supervised machine learning algorithm for detecting and predicting fraud in credit card transactions,” Decis. Anal. J., vol. 6, p. 100163, Mar. 2023, doi: 10.1016/j.dajour.2023.100163.

[21] P. Gupta, A. Varshney, M. R. Khan, R. Ahmed, M. Shuaib, and S. Alam, “Unbalanced Credit Card Fraud Detection Data: A Machine Learning-Oriented Comparative Study of Balancing Techniques,” Procedia Comput. Sci., vol. 218, pp. 2575–2584, 2022, doi: 10.1016/j.procs.2023.01.231.

[22] D. Shah and L. K. Sharma, “Benchmarking Machine Learning Techniques for Credit Card Fraud Detection,” Indian J. Sci. Technol., vol. 18, no. 21, pp. 1681–1695, Jun. 2025, doi: 10.17485/IJST/v18i21.903.