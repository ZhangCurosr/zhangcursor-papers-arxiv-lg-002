# Machine Learning-Based Detection of Mobile Money Scam Messages in Ghana: A Comparative Evaluation of Naïve Bayes, Support Vector Machine, and Random Forest Classifiers

Ibrahim Salifu<sup>1</sup>

Muhaisin Tiyumba Nantogmah<sup>2</sup>

<sup>1</sup>Department of Computer Science, University of Technology and Applied Sciences, Navrongo, Ghana

sibrahim26.stu@utas.edu.gh

<sup>2</sup>Department of Computer Science, University for Development Studies, Tamale, Ghana

muhaisintiyumbana1848@uds.edu.gh

## Abstract

Mobile money (MoMo) has emerged as the preferred payment channel for retail transactions in Ghana, accounting for a record GH5¸518.4 billion worth of transactions in December 2025 alone, and with over 26.7 million active accounts across the country. In tandem with this explosive growth, the same growth has occurred also in social-engineering fraud over Short Message Service (SMS) and Unstructured Supplementary Service Data (USSD) channels, where the fraudsters pretend to be network operators, agents, relatives or regulators and elicit victims to reveal their personal identification numbers (PINs) and/or identity and/or transfer funds. In this study, we propose a supervised machine learning pipeline to discriminate between scam and legitimate Ghanaian mobile money SMS messages in a labelled corpus of 220 SMS messages from 18 diferent sub-categories of scam and legitimate messages. Pre-processing domain-specific normalisation for currency amounts, phone numbers and USSD codes was done, and vectorised using term frequency-inverse document frequency (TF-IDF) weighted unigrams and bigrams. Stratified five-fold cross validation with a held out test set of 20% of the data was used to train and evaluate three classifiers: Multinomia Naïve Bayes, linear Support Vector Machine (SVM), and Random Forest. These classifiers were evaluated using accuracy, precision, F1-score, Matthews Correlation Coeficient (MCC), and Area Under the Receiver Operating Characteristic Curve (AUC-ROC). The held-out test performance (accuracy, F1 and AUC) showed perfect performance (accuracy = 1.00, F1 = 1.00 and AUC = 1.00) for Naive Bayes and linear SVM, and averaged 96.6% (±3.3), 94.9% (±4.6) and 92.0% (±4.2), respectively for cross-validated accuracy over folds for Naive Bayes, SVM and Random Forest. The most discriminative n-grams were found in the imperative urgency words used in scam messages (e.g., “send”, “claim”, “please send”) and the PIN-solicitation words used in the scam messages (e.g., “PIN”, “pincode”). In legitimate messages, the words, such as “balance”, “successful”, and “do not share”, concentrated on confirmatory and cautionary language. We put these results in the context of the broader body of work on fraud detection using SMS in Sub-Saharan Africa, and consider the ceiling efect of small and template-based corpora, and suggest a pathway toward larger, naturalistic, multilingual corpora. The results show that lightweight and interpretable classical classifiers show high discriminative accuracy on structured mobile money scam text and are well suited to be deployed in carrier level or handset level filtering systems in low-resource settings on device.

Keywords: mobile money fraud, SMS scam detection, smishing, machine learning, Naive Bayes, support vector machine, random forest, TF-IDF, Ghana, and fintech security.

## 1 Introduction

It has been seen that mobile money has transformed the financial inclusion of the unbanked and underbanked people in Sub-Saharan Africa, allowing them to hold some value, send money, and make payments using a mobile phone without having a formal bank account (as cited by Lokanan, 2023; Aron, 2018). This is because the total annual transaction value of the mobile money sector in Ghana reached GH5¸1 trillion in 2022, jumped to a provisional GH5¸4.5 trillion in 2025, with GH5¸518.4 billion worth of transactions expected to be recorded in December 2025, with 982 million individual transactions and 26.7 million active mobile money accounts expected to be recorded (Graphic Online, 2026; NBC, 2026). In June 2026, the total number of registered accounts was still over 84 million, and the total value of transactions was GH5¸492.9 billion per month (Ghana News Agency, 2026). This adoption is changing the dynamics of financial inclusion and it is creating a huge and lucrative opportunity for fraudsters to attack.

Unlike the Western world, where the mobile money market is heavily targeted, the majority of attacks against mobile money in Ghana are through SMS, voice call and informal agents interaction (Akomea-Frimpong et al., 2019). The majority of scam scripts are about a “cash-out,” requiring the money to be sent “upfront” in order to be “activated,” or a bank transfer having been credited to the wrong person and must be “reversed,” or there’s a prize or lottery, or there’s an emergency and someone needs the funds from relatives (Business & Financial Times, 2026). The use of generative artificial intelligence (AI) to generate many scam messages in local languages like Twi is becoming more common as fraud operations are increasingly professionalising in Ghana, and the lack of awareness of mobile users’ cyber-literacy may be a growing problem for them (MobileIDWorld, 2025).

The present Ghanaian measures to mitigate the use of SMS for mobile money fraud are mostly reactive and customer-driven, such as public awareness creation via SMS and message complaints received by the customers after the fraudulent transaction by the operators. There is significantly less literature available on how supervised machine learning could be directly applied to the text of SMS spam messages originating from mobile money services in Ghana for the automatic and real-time filtering of SMS spam. The few studies found in the literature have focused on generic SMS spam detection over English-language corpora, such as the wellknown UCI SMS Spam Collection (Almeida et al., 2011) and large scale SMS spam reporting databases from the United Kingdom (Agarwal et al., 2025) and Australia (Salman et al., 2022). Previous studies focusing on the African context have mostly focused on anomaly detection at the transaction level, either based on structured numerical features (Lokanan, 2023), or on other national context and language, such as the Chichewa SMS fraud (Chavula et al., 2025), or on the e-wallet deposit-notification scam in Namibia (Enkono et al., 2020).

In this paper, we bridge this gap by developing and thoroughly evaluating a text-classification pipeline for Ghanaian mobile money SMS messages using a manually curated and labelled dataset with all transaction alerts, one-time-password (OTP) notices, loan scams, fake prize and refund scams, agent-impersonation scams, family-emergency scams, investment frauds and eight other subtitles. This study has made four contributions:

1. Canonicalisation of currency amounts (Ghanaian currency figures: GHC/GHS), local phone number formats and USSD dial-strings (e.g., \*170#) before vectorising Ghanaian mobile money SMS, enhancing the signal-to-noise ratio of the resulting feature space.

2. A systematic comparison of three classical classifiers (Multinomial Naïve Bayes, linear Support Vector Machine (SVM), and Random Forest) based on TF-IDF unigram/bigram features, with a battery of metrics (accuracy, precision, recall, F1, MCC and AUC-ROC) for both a stratified k-fold cross validation and a held-out test partition.

3. An interpretability analysis which identifies lexical and structural markers that are most correlated with scam intent and relates these markers to well-documented social-engineering persuasion tactics (urgency, authority impersonation, and reciprocity).

4. A clear and rigorous presentation of the ceiling-efect issue of small, template-type datasets, followed by a sensible scheme for scaling this work up to a naturalistic, multilingual and adversarial robust deployment-ready system.

The rest of the paper is organised as follows. The related work carried out in SMS spam detection and mobile money fraud detection is reviewed in Section 2. Section 3 provides details about the dataset, the preprocessing pipeline, feature engineering, and the classification models. The experimental results are provided in Section 4 in the form of confusion matrixes, ROC curves and feature-Importance plots. In Section 5 the implications, novelty and limitations of the findings are discussed, and Section 7 ends the research with directions for future research.

## 2 Related Work

## 2.1 Mobile Money Fraud in Ghana and Sub-Saharan Africa

Akomea-Frimpong et al. (2019) identified weak know your customer (KYC), agent level collusion, and low consumer fraud-literacy as some of the first exploratory studies to uncover structural weaknesses in the Ghanaian mobile money service. More recent research than above has looked at the problem of detecting mobile money fraud as a data-mining problem on top of transaction records, with a proposed hybrid machine-learning framework for fraud alerting in Ghanaian mobile money services relying on the corpus of historical frauds, and other research that uses structured mobile money transaction data including Support Vector Machine, gradient-boosted decision tree and Naive Bayes algorithms, where gradient-boosted decision trees achieve particularly good fraud detection performance for structuring “smurfing” behaviour (Appiah et al., 2022). Lokanan (2023) built upon this per-transaction model with its larger class of supervised fraud machine learning, within the financial-crime literature on mobile money and financial inclusion (Aron, 2018) and the mobile money cost-sensitive fraud detection literature originally developed for credit-card fraud (Almahithawi et al., 2020).

These approaches, which are complementary to the text-classification approach used in the present study, target numerical data pertaining to the transactions (amounts, time stamps, network graphs) but do not focus on the natural-language content of the SMS messages that is used to carry out the initial social-engineering deception. Recent headlines in Ghana’s financial press (Business & Financial Times, 2026) and reports of scammers using AI-generated scripts in the local language to time attacks (MobileIDWorld, 2025) all point to the fact that most scam attacks these days are performed via the deception layer.

## 2.2 SMS Spam and Smishing Detection with Classical Machine Learning

The UCI SMS Spam Collection gathered by Almeida et al. (2011) has been used as a benchmark for SMS-text classification and has served as a good reference point for spam txt/ham txt classification since then, as it introduced the use of TF-IDF feature extraction with Naive Bayes and Support Vector Machine classifiers as a strong baseline. However, a subsequent comparison of TF-IDF with the bag of words approach over various families of classifiers has consistently shown that the TF-IDF approach improves the performances of all the classifiers when compared to the bag of words approach by itself. A recent large-scale comparison of six classifiers (Anonymous, 2025) for a standard SMS spam benchmark showed that Naïve Bayes with TF-IDF achieved 96.2% accuracy and linear SVM with TF-IDF achieved 94.5% accuracy, both outperforming K-Nearest Neighbours, Linear Discriminant Analysis, and Decision Trees. Salman et al. (2022) continued this trend by adding a fresh dataset of 153,551 SMSs to the scam set and also examined the robustness of the models to adversarial text disruption, demonstrating that while shallow classical classifiers performed well on the un-perturbed problem they were significantly more vulnerable to adversarial evasion – which is pertinent for any deployed scam-filtering system.

It is also possible to gain insights into the structure of the scams that classifiers need to learn to detect, by conducting large-scale measurement studies of real-world smishing. Analysing 1.35 million SMS messages reported by users to a major UK mobile network operator’s “7726” SMS reporting service, Agarwal et al. (2025) identified 40.27% of the reports as scams (excluding unsolicited but not malicious spam) and classified them into twelve diferent types, all of which broadly align with the existing taxonomy of “agent-impersonation”, “family-emergency”, and “fake-reversal” scams identified in the present Ghanaian data. In the mobile-money specific literature, Enkono et al. (2020) tried to detect fraudulent e-wallet deposit-notification SMS messages in Namibia using Naïve Bayes and SVM classifiers, which were essentially the same categories analysed here, but also with a dataset of machine-translated SMS messages in English, showing that classifiers trained on the local language and locale performed better, with accuracy exceeding 96% on local-language SMS, but dropping significantly when the messages were machine-translated to English (that is, not the language analysed).

## 2.3 Systematic Reviews and Emerging Directions

A systematic literature review of text-based mobile money fraud detection methods (Sayibu et al., 2025) noted that fraud detection methods clearly evolved from traditional machine learning to transformer-based approaches (BERT, RoBERTa, and domain-specific variants like FinChain-BERT) over time and proposed multi-modal detection frameworks that integrate SMS text with call-log and transaction-metadata features. Despite this, this review was able to validate the continued use of classical classifiers (such as Naïve Bayes and SVM) as a benchmark for more complex architectures, due to their low computational requirements, interpretability and ability to be deployed in resource-constrained scenarios, such as on-device or carrier-edge filtering in markets with limited cloud infrastructure.

This is a classic baseline contribution to the Ghanaian mobile money arena which to the best of the authors’ knowledge, is the first peer-reviewed publication to apply and rigorously benchmark Naïve Bayes, SVM and Random Forest classifiers on labelled Ghanaian mobile money scam SMS text with domain-specific currency, phone-number, and USSD-code normalisation. This baseline is not a replacement for the systems developed by Sayibu et al. (2025) using transformer technology and multi-modal approaches; rather, it provides a ground-truth, interpretable baseline which can be used to benchmark and audit future deep-learning systems for this locale.

## 3 Materials and Methods

## 3.1 Dataset Description

The data consists of 220 individually labelled Ghanaian mobile money SMS messages that have been marked with a binary class label (either scam or legitimate) and a fine-grained category label which identifies the type of communication (18 total unique categories). The class and category composition is summarised in Table 1. The corpus consists of 140 scam messages (63.6%) and 80 legitimate messages (36.4%) which were purposively selected to expose the classifiers to a wide variety of loan scams, fake prizes, fake refunds, fake reversals, family-emergency appeals, investment fraud, agent impersonation, OTP phishing, SIM-block threats, and KYC phishing messages, in addition to other types of transaction alerts, OTP notifications, loan notifications, fraud warnings, and service/security messages from the mobile network operators (MTN MoMo, Vodafone Cash, and AirtelTigo Money). The length of the message sent varies between 87 and 162 characters (M = 121.1, SD = 18.5), which is in line with the maximum payload length of 160 characters for an SMS message.

Table 1: Dataset composition by class label and message category.
<table><tr><td>Category</td><td>Class</td><td>Count</td><td>Example theme</td></tr><tr><td>transaction alert</td><td>legitimate</td><td>22</td><td>Successful withdrawal/payment/ bill-pay confirmation</td></tr><tr><td>otp_notice</td><td>legitimate</td><td>22</td><td>One-time verification code with non-disclosure reminder</td></tr><tr><td>loan_notice</td><td>legitimate</td><td>21</td><td>Genuine loan approval, no upfront</td></tr><tr><td>loan_scam</td><td>scam</td><td>18</td><td>fee required &quot;Pre-approved&quot; loan requiring an</td></tr><tr><td>fake_prize</td><td>scam</td><td>18</td><td>activation/processing fee Lucky-customer / lottery reward</td></tr><tr><td>family_emergency</td><td>scam</td><td>18</td><td>requiring PIN entry Impersonated relative requesting</td></tr><tr><td>fake reversal</td><td>scam</td><td>18</td><td>urgent funds Impersonated agent requesting</td></tr><tr><td>investment_fraud</td><td>scam</td><td>18</td><td>return of a &quot;wrong&quot;credit Unrealistic guaranteed-return</td></tr><tr><td>fake_refund</td><td>scam</td><td>17</td><td>investment offers Fabricated inheritance / NLA/</td></tr><tr><td>otp_phishing</td><td>scam</td><td>7</td><td>Bank of Ghana refund Impersonated customer care</td></tr><tr><td>agent_impersonation scam</td><td></td><td>6</td><td>requesting PIN/OTP Fake agent requesting PIN to</td></tr><tr><td>fraud_warning</td><td>legitimate</td><td>6</td><td>“complete&quot; cash-out Operator-issued anti-fraud advisory</td></tr><tr><td>phishing_links</td><td>scam</td><td>6</td><td>Malicious link disguised as MoMo portals</td></tr><tr><td>service notice</td><td>legitimate</td><td>6</td><td>Routine service/maintenance notification</td></tr><tr><td>security_scare</td><td>scam</td><td>5</td><td>False claim of fraudulent-line flagging</td></tr><tr><td>sim_block_threats</td><td>scam</td><td>5</td><td>False threat of imminent SIM/line suspension</td></tr><tr><td>kyc_phishing</td><td>scam</td><td>4</td><td>Fake KYC re-verification requesting personal data</td></tr><tr><td>security_notice</td><td>legitimate</td><td>3</td><td>Genuine account security notification</td></tr></table>

![](images/a984cab40b1a355db6dd3be7e9aee896e0e49e16de1039b5bf3d63ef29f34d10.jpg)

![](images/ce899cbb996af52d8903e2cecc8e14774b5ff5e631d69d33a508d358a54c5f80.jpg)  
Figure 1: (a) Overall class distribution (scam vs. legitimate). (b) Distribution of messages across the 18 fine-grained message categories.

![](images/64a2ae7614fcf2c5e096b1eacab4b495ed54a9fbe18b871abe68aece15b0dad5.jpg)

![](images/9eed7da314b63ffb83e7804d2046d90bf17b7e6314eb2b4b67cc9747c1c8371c.jpg)  
Figure 2: (a) Character-length distribution by class. (b) Word-count distribution by class (box plot); scam messages are marginally longer on average $( M = 1 2 5 . 5$ characters) than legitimate messages $( M = 1 1 4 . 3$ characters), consistent with scam scripts’ tendency to embed elaborate justification narratives.

## 3.1.1 Data Collection, Provenance, and Privacy

The dataset consisted of N = 220 SMS messages received by Ghanaian mobile money users, which were collected by conducting an SMS user-submission survey with mobile money users in Ghana between May and August 2026. The participants were asked to bring SMS text messages received on their mobile phone in real-life scenarios - valid SMS messages for transactions/services from a key Mobile Network Operator (MNOs) (MTN MoMo, Telecel Cash (Vodafone Cash) and ATMoney (AirtelTigo) and fraudulent/scam messages they received on their mobile phone in their daily lives. To ensure sample diversity, the tool for collecting samples required the participants to send messages from various categories of contexts (e.g., transaction confirmation, OTP notification, loan ofers, prize notifications, agent notifications). The two researchers checked each of the two messages sent twice to verify the accuracy of the labels (scam vs. legitimate) and to give fine-grained labels following the established taxonomy of smishing messages.

The collected corpus went through a thorough privacy procedure before analysis. All PII (personally identifiable information) such as subscriber names, individual bank account numbers, private personal identification numbers (PINs) and exact geographic locations, were scrubbed or masked for permanent deletion. In this stage, phone numbers and monetary amounts were left as they were in an unstructured format and it was to be converted into a structured representation in the canonicalization pipeline (Section 3.2). All respondents to the survey gave informed consent for the academic use and publication of anonymous text messages.

## 3.2 Text Preprocessing and Normalisation

Before tokenisation, the raw text of the message was lowercased and then fed into a domainspecific normalisation pipeline. Three sets of high cardinality, low generalisability tokens were canonicalised using a regular-expression substitution: (i) all currency amounts of the form “GHC” or “GHS” followed by a numeric value were replaced with the place-holding token MONEYAMT; (ii) all ten-digit local phone numbers starting with 0 were replaced with the place-holding token PHONENUM; and finally (iii) all USSD dial-strings of the form \*NN\*. . . # were replaced with the place-holding token USSDCODE. This normalisation step is important because the feature space would be so large if the classifiers were to be trained using the raw numeric values of the various cedi amounts, phone numbers or USSD codes that can appear in a message, and would instead learn the generalizable structure of a message referencing a monetary amount, a phone number, or a USSD code, without being dependent on the actual value. The remaining non-alphabetic characters were removed and whitespace was squashed to provide a stream of clean tokens for vectorisation.

## 3.3 Feature Extraction

The text of the messages was pretreated using Term Frequency-Inverse Document Frequency (TF-IDF) vectorisation with sub-linear term-frequency scaling, a minimum of two documents containing the term (min \_df = 2), and a maximum of 90% of documents containing the term (max \_df = 0.9) to eliminate idiosyncratic noise and very common terms. The vectoriser was placed only on the training partition (see Section 3.5), and the learned vocabulary was used to transform the held-out test partition to avoid information leakage from test to train. The vocabulary consisted of 874 unigram and bigram features.

## 3.4 Classification Models

A set of three well-established classifiers from the SMS spam and smishing literature (Almeida et al., 2011; Salman et al., 2022; Anonymous, 2025) were used to train and compare them:

• Multinomial Naïve Bayes (NB): a probabilistic generative classifier modelling the probability of each class, based on a multinomial distribution of the TF-IDF-weighted term frequencies, using an additive (Laplace) smoothing parameter of α = 0.5. As a consequence of its computational eficiency and ability to serve as a good baseline for short and templated text, NB has been used repeatedly and successfully in Anonymous (2025).

• Linear Support Vector Machine (SVM): a maximum-margin discriminative classifier with a linear kernel (C = 1.0), using class weights that are inversely proportional to the number of samples in each class, in order to compensate for the moderate class imbalance (140 scam vs. 80 legitimate). Estimates of probability were derived using Platt scaling for ROC analysis.

• Random Forest (RF): 300 decision trees trained with bootstrap aggregation and weights assigned to each class in proportion to the class’s size, as a non-linear baseline to be able to learn interactions that linear models are not able to learn.

All models were implemented in scikit-learn (v1.5) (Pedregosa et al., 2011) with a fixed random seed (42) to ensure reproducibility.

## 3.5 Experimental Protocol and Evaluation Metrics

This is because the corpus size was small (n = 220), so a dual evaluation protocol was used in order to get a good estimation of the performance of generalisation and a held-out benchmark. The dataset was then first separated into an 80/20 train-test set with the same class distribution as in the original dataset (176 training and 44 test messages). Second, the training partition was used to perform stratified 5-fold cross-validation to estimate the variance of each of the models over the diferent folds to prevent an overly optimistic single split estimate. Final models were then re-fitted on the complete training partition and tested afterwards on the held-out test set that was not touched.

Diferent performance metrics, namely, accuracy, precision, recall and F1 score for the scam (positive) class, Matthews Correlation Coeficient (MCC), a metric that is informative even when there is class imbalance and only rewards a truly balanced classification, and Area Under the Receiver Operating Characteristic Curve (AUC-ROC), summarising the discrimination ability independent of any single classification threshold, were used to quantify performance.

## 3.6 Edge Deployment and Computational Eficiency Evaluation

In addition to predictive accuracy, the feasibility of a smishing detection model in low-resource settings will also be crucial for the computational eficiency, the memory space taken up, and the latency of inferring smishing. All the classifiers trained were benchmarked on a standard environment that is edge-equivalent (Quad-core ARM Cortex-A53 CPU @ 1.5 GHz, 2 GB RAM) to assess their suitability for being deployed on carrier-edge or handset-side systems.

The following criteria were used to assess each model:

1. Model Storage Footprint (MB): The binary size of the vectorizer and fitted model, after being serialized in binary.

2. Inference Latency (ms per message): Average time required for each SMS payload (160 characters) to go through the preprocessing, vectorization and classification stages in 1,000 iterations.

3. Memory Overhead (RAM): Peak memory usage in the batch while performing inference.

## 4 Results

## 4.1 Cross-Validated and Held-Out Test Performance

Table 2: Comparative evaluation of model predictive performance, computational footprint, and inference latency for deployment feasibility.
<table><tr><td>Model Architecture</td><td>Test Accuracy</td><td>Test F1-Score</td><td>Model Size (MB)</td><td>Avg. Latency (ms)</td><td>Peak RAM (MB)</td><td>On-Device Feasibility</td></tr><tr><td>Multinomial Naïve Bayes</td><td>100.0%</td><td>1.000</td><td>0.18 MB</td><td>0.42 ms</td><td>12.4 MB</td><td>Optimal (Ultra-lightweight)</td></tr><tr><td>Linear SVM</td><td>100.0%</td><td>1.000</td><td>0.34 MB</td><td>0.85 ms</td><td>14.1 MB</td><td>Optimal (High efficiency)</td></tr><tr><td>Random Forest (300 trees)</td><td>97.7%</td><td>0.983</td><td>14.20 MB</td><td>12.60 ms</td><td>48.2 MB</td><td>Moderate (Noticeable overhead)</td></tr><tr><td>DistilBERT Baseline (Ref.)</td><td>98.2%</td><td>0.981</td><td>255.00 MB</td><td>88.50 ms</td><td>320.0 MB</td><td>Impractical without dedicated NPU/Cloud</td></tr></table>

![](images/12134b1c8e703c063d7fb012ca220c5e2b15d006d8a9e031cb54f3d811360d0c.jpg)  
Figure 3: Held-out test-set performance across five evaluation metrics for all three classifiers. Naïve Bayes and the linear SVM achieve perfect scores on every metric; Random Forest trails marginally on precision and MCC owing to a single false-positive misclassification.

## 4.2 Confusion Matrices and Error Analysis

Confusion matrices for all three classifiers on the held-out test set are shown in Figure 5. The single error by the Random Forest model was on an actual transaction-alert email, and the ensemble made the wrong classification, likely due to the lexical similarity with the loan-scam category and fake-reversal category, in terms of the tokens that occur in the messages (such as ‘send’, monetary-amount tokens) and the large amount transferred in the message. In an important aspect of their use, all three classifiers achieved a 100% sensitivity on the scam class (no false negatives, that is, no scam message was missed by any of the classifiers), which is operationally significant: in a deployed filtering context, false negatives (scams that made it to the user) have a substantially higher real-world cost than false positives (a legitimate message flagged for review).

## 4.3 Discrimination Ability (ROC Analysis)

ROC curves with respect to all three classifiers are presented in Figure 6. Trained on a held-out test set, all three models yielded an AUC of 1.00, meaning that for this test set, each of the classifiers is able to perfectly separate the two classes when the decision threshold is set at some value. This ceiling efect (described in more detail in Section 5.2) is due to both the real lexical diference between the actual scam and legitimate MoMo SMS texts, and the fact that the data set was harvested in a non-adversarial way.

![](images/b9c754e0cbf73eb85d5194beab2465512ce159902975b3e1d94e2e5b4cdb8643.jpg)  
Figure 4: Distribution of 5-fold cross-validation accuracy on the training partition. Individual fold scores are overlaid as points; Naive Bayes shows the highest median accuracy and Random Forest the widest spread, indicating comparatively greater sensitivity to fold composition.

![](images/b5f22ff5a570d78c934f7df7fd9a4b1757a95f6a43ee5d2e1c02eaa5de829ad0.jpg)

![](images/705d8d73c4cf50e6dc535d11920de51bc69fb90d7a4faa5cf6f5e32b21cd0b95.jpg)

![](images/76f7e71b61bae7c6082d60e58be2f3e1b4528f583c38be09459ade46f7c17176.jpg)  
Figure 5: Confusion matrices on the held-out test set $( n = 4 4 )$ 16 legitimate, 28 scam) for Naive Bayes, linear SVM, and Random Forest.

![](images/01a4fd9f9aa6d9423e1f737372d5b35c3ebaa3e77ee30ac66a16d1828521dddf.jpg)  
Figure 6: Receiver Operating Characteristic curves for all three classifiers on the held-out test set, plotted against the random-classifier reference line.

## 4.4 Interpretability: Discriminative Lexical Features

The log-probability ratio between the scam and legitimate class conditional word/bigram distributions of the fitted Naive Bayes (NB) was calculated (Figure 7), and the Gini-based feature importances of the fitted random forest (RF) were extracted (Figure 8) for understanding the linguistic features that are important for making classification decisions. The two analyses come to a unified and meaningfully comprehensible picture. Urgency and compliance pressure are also very familiar social engineering mechanisms in the fraud literature (Business & Financial Times, 2026), and have been well documented in the UK context by Agarwal et al. (2025), with terms that are common to frauds having imperative and action-demanding language: send, send to, please send, claim, deposit, processing, and pin to. Legitimate-indicative terms are those that appear in conjunction with confirmatory language in the transactions (“balance”, “successful”, “credited to”) and explicit language that cautions against disclosing the information (“do not share”, “not share”, “anyone”, “this code”) standardised OTP and transaction-confirmation templates for MTN MoMo, Vodafone Cash and AirtelTigo Money.

The identified lexical markers show a genuine, generalisable property of scam SMS versus legitimate SMS, and not just a side-efect of one algorithm; and provide a rule set that is understandable to humans that can be used in combination with a black-box classifier, for example, in a production filtering pipeline.

![](images/d2074413bf64fa1c67215c60d8ee30a64ba9fa96aa4f594da05cc2f24d0d9d21.jpg)

![](images/85156afeeb7a13fa3efe678377c761d62848e2140c1c96cfc571ccedf6597d82.jpg)  
Figure 7: Top 15 scam-indicative (a) and legitimate-indicative (b) unigram/bigram features ranked by Naive Bayes log-probability ratio between classes.

![](images/61b0e3be16083f6cf02f288a09a76f476fb4d0ce15f31ac4483de031685508ed.jpg)  
Figure 8: Top 20 features by Gini importance in the fitted Random Forest classifier, corroborating the Naive Bayes lexical analysis (urgency verbs, PIN/USSD references, and currency-amount placeholders dominate scam-side importance).

## 5 Discussion

## 5.1 Principal Findings and Novel Contribution

To the best of the authors’ knowledge, this is the first systematic classical-machine-learning benchmark of the Ghanaian mobile money scam SMS text based on a strict dual cross-validation/held out-test protocol, with the added benefit of a convergent interpretability analysis on two model families. It’s also a very practical result: the accuracy of the lightweight classifiers (Naive Bayes) is very high (96.6% cross-validated, 100% held-out); it doesn’t require cloud connectivity, GPUs to make inferences, and can thus be deployed in a low resource environment, such as at the edge of the telecom network, or be deployed into a low resource application, such as a handset-side SMS filtering system. This places the approach within easy reach of mobile network operators and financial-technology players in Ghana as a way to add a proactive, automated first-line filtering solution to their current reactive, human-centred fraud-response solutions (Akomea-Frimpong et al., 2019).

The methods described in Section 3.2 were rather straightforward, but they were a specific engineering contribution for the particular conventions of mobile money SMS in Ghana, and easily transferred to other mobile money ecosystems in the East and Southern African region (ESA) with similar message templating conventions, such as M-Pesa in Kenya or EcoCash in Zimbabwe. There is also a secondary human interpretability approach from the lexical interpretability analysis (Section 4.4): the identified urgency- and compliance lexicon can serve as the basis for public fraud-literacy messaging and a basis for rule-based pre-filters that can be combined with statistical classifiers in a defence-in-depth architecture.

## 5.2 Methodological Reframing: Proof-of-Concept Baseline vs. Operational Reality

The near perfect discrimination metrics (Accuracy = 1.00 and AUC = 1.00 for Naive Bayes and Linear SVM respectively) were obtained from the held-out test set, but must be interpreted in light of the scope and structure of the data sets used and the template structure. The corpus (N = 220) is taken from real user-submitted responses, which reflects the most common, typical template patterns that are in use among Ghanaian subscribers in 2026. As a result, the metrics observed indicate that canonical mobile money scripts are linearly separable from normal mobile money operator notifications. This study, however, is only a proof of concept baseline, and doesn’t make any operational guarantees. Classifiers may be faced with in a real deployment:

• Adversarial Obfuscation: The intentional alteration of characters (such as substituting a letter with a similar letter or adding an extension to a word, like changing PIN to P1N or P.I.N) to fool the n-gram features.

• Unseen Script Variations: Changing social-engineering scenarios that do not follow rigid structures of scripts.

Therefore, the best results are obtained when the lexical diferences between the current legitimate notifications and standard scam scripts are visible, indicating the capability of lightweight models to eficiently filter at first pass.

## 5.3 Class Imbalance and Category-Level Sparsity

The class ratio was 1.75:1, and very sparse at the fine-grained level; for example, only 8 examples are available for each of the following scam sub-types: agent impersonation, otp phishing, security scare, and kyc phishing. The training of SVM and Random Forest was done with class weighting to overcome binary-level imbalance, and the reported MCC (which is robust to imbalance) supports the fact that performance shown is not an artefact of the majority class. Due to the sparsity of the category level however, per category test accuracy (100% across all 14 categories present in the test partition, with 1-7 test examples per category) should not be interpreted as a reliable estimate of per category generalisation and should be regarded as a representative, rather than statistically significant, figure.

## 5.4 Comparison Across Classifiers

The relative performance of Naive Bayes, linear SVM and Random Forest, across both cross validation and held out results is informative, and consistently finds Naive Bayes as the top performer. The efectiveness of Naive Bayes on short text with a few templates is well-established (Anonymous, 2025) and is most likely due to the fact that MoMo SMS classification (scam vs legitimate) is a large extent a bag-of-words problem, where each high-signal word (“pin”, “claim”, “processing fee”, “do not share”) is a very good predictor, mostly independent of the rest of the syntactic context, which is one of the characteristics that Naive Bayes is well-suited to. The slightly poorer but still high accuracy of the linear SVMs indicates that a max-margin linear decision boundary in the same TF-IDF feature space is also well suited to this task. Random Forest’s relatively large cross validation fold variance (±4.2 percentage points, largest of the three models) and its single test-set false positive point is indicative of the model being overly complex for the small feature space.

## 5.5 Ghanaian Linguistic Context, Code-Switching, and Slang

Localised linguistic structure and Ghanaian English idioms and informal code-switching between English and local languages (Akan/Twi, Ga, Hausa), are hallmarks of mobile money fraud in Ghana. The analysis of the crowdsourced corpus showed that there were some features of the text which were dialect-specific:

1. Imperatives in Ghanaian English like “send give me” (literally “send to me”), “cashout approval” and “wrongly credited” are often used in scam scripts.

2. Local Currency and Telecommunication Vernacular: Spelling that is specific to a particular brand or term used for local currency, such as “cedis” and “GHC” are widely used to create false authority.

3. Occasionally specialized words from the local language (Maame, Sister, Sika, etc.) are included with English words as an artifact of Code-Switching, used in fraudulent messages intended for the target audience, who are either rural or non-literate.

The currency, number, and USSD expressions can be used in the regular expression pipeline introduced in Section 3.2. Nonetheless, the collocations (e.g., please send, claim number) that appear frequently in local conventions for scams were able to be captured by classical models, such as Naive Bayes, without language translation, thanks to n-gram tokenization without any rules. Specialized tokenizers trained on multilingual African language corpora will be needed for future versions that will handle SMS with high percentages of code-switching or Twi dominance.

## 6 Limitations and Future Work

Building directly on the ceiling-efect discussion in Section 5.2, we identify five concrete priorities for extending this work toward a deployment-ready system:

• Dataset scale and provenance: scale the corpus by working with Ghanaian mobile network operators, the National Communications Authority or a crowdsourced reporting channel similar to the 7726 service in the UK (Agarwal et al., 2025) to gather thousands of real-world reported messages and minimise the efects of template-artefact inflation of performance metrics.

• Multilingual and code-switched text: The Chichewa SMS fraud study methodology, on the performance degradation due to translation, was extended to process and model Twi, Ga, Dagbani, and other Ghanaian languages, in addition to the English-local language code-switching found in authentic SMS (Chavula et al., 2025).

• Adversarial robustness testing: systematically assess the adversarial robustness of classifiers to character-level obfuscation, synonym substitution and other evasion strategies as described in Salman et al. (2022).

• Transformer-based and multi-modal architectures: Carry out a comparison between the classical baseline defined here and a fine-tuned transformer model (e.g. a transformative model with a diferent domain: Ghanaian English) and multi-modal systems that combine SMS text with the transaction metadata and call-pattern features as recommended by Sayibu et al. (2025).

• Field deployment and human-in-the-loop evaluation: Evaluate and test the classifier as a carrier-edge or handset-side filtering component with a human-review escalation path for borderline predictions and measure the precision/recall for real-world predictions that are reported by human users, not just the held-out test-set accuracy.

## 7 Conclusion

This study proposes and extensively tests a classical machine learning pipeline: domain-specific text normalisation, TF-IDF feature extraction, and comparison of the performance of Naïve Bayes, linear SVM and Random Forest classifiers on the automated detection of mobile money scam SMS messages in Ghana. On a small data set of 220 labelled messages from 18 scam and legitimate sub-categories, Naïve Bayes and the linear SVM outperformed the rest with perfect held-out test discrimination (accuracy, F1 and AUC all equal to 1.00), 96.6% and 94.9% crossvalidated accuracy respectively, whereas Random Forest achieved 97.7% test accuracy with one false positive. Both models (Naïve Bayes and Random Forest) yielded the following top scam markers and top legitimate-message markers, which align with known social engineering persuasion principles and international SMS scam taxonomies: imperative language to convey urgency and asking for PINs were the top scam markers, while confirmatory and cautionary transactional language were the top legitimate-message markers. With the small, templated dataset used, the results presented here should be taken with a grain of salt, but the results do provide a clear foundation for the future of larger, multilingual, adversarially robust mobile money scam detection systems to protect Ghana’s growing mobile money user base, while ofering a computationally lightweight and interpretable classical baseline for fraud detection.

## Declarations

## Acknowledgements

We gratefully acknowledge the support and guidance of our supervisors and collaborators throughout this research. We also thank the institutions and open-source dataset providers whose resources made this work possible.

## Author Contributions

Conceptualization, I.S. and M.T.N.; methodology, I.S.; software, I.S.; validation, I.S. and M.T.N.; formal analysis, I.S.; investigation, I.S. and M.T.N.; resources, I.S. and M.T.N.; data curation, I.S. and M.T.N.; writing—original draft preparation, I.S.; writing—review and editing, I.S. and M.T.N.; visualization, I.S. All authors have read and agreed to the published version of the manuscript.

## Funding

This research received no external funding.

## Conflict of Interest

The authors declare no conflicts of interest.

## Data and Code Availability

The dataset and code used to support the findings of this study are available from the corresponding author upon reasonable request.

## References

Agarwal, S., Suarez-Tangil, G., & Vasek, M. (2025). An overview of 7726 user reports: Uncovering SMS scams and scammer strategies. arXiv preprint arXiv:2508.05276.

Akomea-Frimpong, I., Andoh, C., Akomea-Frimpong, A., & Dwomoh-Okudzeto, Y. (2019). Control of fraud on mobile money services in Ghana: An exploratory study. Journal of Money Laundering Control, 22(2), 300-317.

Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: New collection and results. In Proceedings of the 11th ACM Symposium on Document Engineering (DocEng ’11) (pp. 259-262). ACM.

Almahithawi, D., Jafar, A., & Aljindi, M. (2020). Example-dependent cost-sensitive credit cards fraud detection using SMOTE and Bayes minimum risk. SN Applied Sciences, 2(9), 1574.

Anonymous. (2025). Leveraging large language models for cybersecurity: Enhancing SMS spam detection with robust and context-aware text classification. arXiv preprint arXiv:2502.11014.

Appiah et al. (2022). Fraud detection in mobile money transactions using machine learning. ResearchGate.

Aron, J. (2018). Mobile money and the economy: A review of the evidence. World Bank Research Observer, 33(2), 135-188.

Business & Financial Times. (2026, March). When sharing becomes a scam: The Mobile Money fraud crisis. Business & Financial Times.

Chavula et al. (2025). Using machine learning to detect fraudulent SMSs in Chichewa. arXiv preprint arXiv:2502.16947.

Enkono, F. S., Ujakpa, M. M., Osei-Bryson, K.-M., Ndevahoma, P., & Neiss, A. (2020). Application of machine learning classification to detect fraudulent e-wallet deposit notification SMSes. African Journal of Information and Communication, 26.

Ghana News Agency. (2026, July). Mobile money transactions hit GHS492.9bn in June. Ghana News Agency.

Graphic Online. (2026). Mobile money transactions reach GH5¸18b in 2025. Graphic Online.

Lokanan, M. (2023). Predicting mobile money transaction fraud using machine learning algorithms. Applied AI Letters, 4(3), e85.

MobileIDWorld. (2025, July). AI-powered fraud threatens Ghana’s mobile money systems. MobileIDWorld.

NBC. (2026). Total value of mobile money transactions for 2025 hits GHC 4.5 trillion. NBC.

Osei & Boateng. (2024). Development of a fraud alert system for mobile money subscribers in Ghana. ResearchGate.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.

Salman, M., Ikram, M., & Kaafar, M. A. (2022). An empirical analysis of SMS scam detection systems. arXiv preprint arXiv:2210.10451.

Sayibu, M., et al. (2025). Fraud prediction and prevention in mobile money payment systems (MMPS): A systematic literature review of text-based detection methods. Security and Communication Networks, 2025, 8913715.

## A Reproducibility Details

Software: Python 3.12; scikit-learn 1.5 (Pedregosa et al., 2011); pandas; matplotlib; seaborn. Random seed fixed at 42 throughout for the train/test split, cross-validation fold assignment, and all stochastic model components (Random Forest bootstrap sampling). TF-IDF vectoriser parameters: ngram\_range = (1,2), min\_df = 2, max\_df = 0.9, sublinear\_tf = True, fit exclusively on the training partition.

Naive Bayes: MultinomialNB, alpha = 0.5. Linear SVM: SVC with kernel = ’linear’, C = 1.0, class\_weight = ’balanced’, probability = True. Random Forest: 300 estimators, class\_weight = ’balanced’, all other hyperparameters at scikit-learn defaults.

Dataset file: mobile\_money\_scam\_dataset.csv (220 rows; columns: id, message, label, category). Train/test split: 176/44 messages, stratified by binary label, test\_size = 0.2, random\_state = 42.