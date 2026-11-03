# Multidimensional Suicide Risk Assessment with Machine Learning using Electronic Health Records

Nathan Scollar<sup>1</sup>, Kemal Davaslioglu<sup>1</sup>, Nathan Conger<sup>1</sup>, Sastry Kompella<sup>1</sup>, Yalin E. Sagduyu<sup>1</sup>, and Nathaniel D. Bastian<sup>2</sup>

<sup>1</sup>Nexcepta, Gaithersburg, MD, USA

<sup>2</sup>United States Military Academy, West Point, NY, USA

Abstract—Suicide prevention remains a critical public health challenge, and electronic health records (EHRs) provide an opportunity to characterize suicide-related risk indicators using documented clinical, behavioral, medical, and social-contextual information. This paper presents a multidimensional machine learning framework to model and estimate suicide-related outcomes using MIMIC-IV EHR data. We construct clinically motivated diagnosis categories and define a composite outcome from documented codes corresponding to suicidal ideation, suicide attempt, prior suicidal behavior, and intentional selfharm. To address severe class imbalance and reduce age-related confounding, we apply age-matched resampling that retains all positive patients and samples comparison-cohort patients from the same age group. Logistic regression, random forest, and XG-Boost classifiers are evaluated before and after resampling with isotonic calibration. The calibrated XGBoost model achieves the strongest matched-cohort performance, outperforming the other evaluated classifiers. SHAP-based interpretation shows that diagnostic burden, depression-related features, age, personalitydisorder history, social determinants related to violence, and medical comorbidity contribute to elevated predicted probabilities. Our results demonstrate that combining broad EHRderived features, age-matched cohort construction, calibration, and feature-level interpretation provides a practical framework for retrospective suicide-related risk modeling.

Index Terms—Electronic health records, suicide-related risk modeling, age-matched resampling, class imbalance, explainable machine learning

## I. INTRODUCTION

Electronic health records (EHRs) provide a valuable foundation for health-risk modeling by capturing clinical, behavioral, and contextual information across patient encounters. Advances in machine learning make it possible to analyze these complex and heterogeneous data to support predictive healthcare analytics and earlier clinical assessment. Suicide prevention is a particularly important application: in 2023, nearly 50,000 Americans died by suicide, while 12.8 million seriously considered suicide, 3.7 million made a plan, and 1.5 million attempted suicide [1]. U.S. suicide rates increased by 37% between 2000 and 2018, briefly declined between 2018 and 2020, and returned to their peak by 2022 [1]. These trends highlight the need for data-driven methods that can characterize suicide-related risk indicators and support more informed clinical assessment.

This paper investigates retrospective suicide-related risk modeling using MIMIC-IV, a large-scale EHR dataset containing emergency department and intensive care unit records from over 200,000 patients [2]. Building on prior machine learning approaches for suicide risk assessment that use structured EHR variables, temporal information, and ensemble learning [3], [4], our objective is to develop a retrospective modeling framework that integrates psychiatric, behavioral, medical, and social-determinant-related features associated with documented suicide-related outcomes. We first perform odds-ratio analysis to identify diagnosis categories with strong univariate associations with the composite outcome, then train and calibrate logistic regression, random forest, and XGBoost classifiers under both imbalanced and age-matched cohort settings. To support interpretability, we use SHAP-based feature attribution to examine which EHR-derived factors contribute most strongly to elevated predicted probabilities [5]. Although mental health diagnoses provide essential information, this framework is designed to capture broader patterns across the documented health profilse that may not be represented by psychiatric variables alone.

The novel contributions of this paper are as follows:

1) Multidimensional EHR Feature Framework: We construct 53 clinically motivated diagnosis categories spanning psychiatric conditions, substance abuse, cognitive and behavioral factors, physical health conditions, and social determinants of health to support retrospective suicide-related risk modeling.

2) Age-Matched Cohort Construction for Class Imbalance: We introduce an age-matched resampling strategy that retains all positive patients while selecting three negative patients from the same age group for each positive patient, ensuring a consistent 25% positive rate across age groups and reducing confounding associated with age-dependent outcome prevalence.

3) Calibrated Model Evaluation and Interpretation: We develop an evaluation framework that compares logistic regression, random forest, and XGBoost classifiers before and after age-matched resampling, applies isotonic calibration, and assesses robustness across 20 random seeds. The calibrated XGBoost model achieves the strongest matched-cohort performance, with an F1 score of 0.81 and an Area Under the Precision-Recall Curve (AUPRC) of 0.89. Across the 20-seed robustness analysis, XGBoost maintains stable performance, with an average F1 score of 0.81 and AUPRC of 0.89. We further use SHAP-based analysis to interpret predicted probabilities and identify the EHR-derived features that contribute most strongly to model predictions.

The remainder of this paper is organized as follows. Section II reviews related work. Sections III–V describe data preprocessing, exploratory analysis, and age-matched resampling. Section VI presents the modeling framework, Section VII reports the experimental results, and Section VIII concludes the paper.

## II. RELATED WORK

Prior work has applied machine learning methods to suicide-related risk modeling, including a balanced random forest approach that embeds temporal information within each tree to support suicide risk assessment [3], ensemble models trained on Swedish registry data that combine penalized logistic regression, random forests, gradient boosting, and neural networks [4], and interpretable CART models for suicide-attempt and self-harm risk after medical hospitalization [6]. Clinical narratives have also been analyzed to identify warning signs preceding suicide attempts among adults with substance use problems [7]. These studies demonstrate the value of structured EHR variables, temporal information, ensemble learning, interpretable models, and clinical narratives.

A related body of work focuses on veterans and military populations, including discrete-time survival models for longhorizon suicide prediction [8], STARRS-LS models based on baseline and post-service surveys [9], age-stratified logistic regression for preventive intervention [10], and REACH VET evaluations for legal-involved veterans [11]. These efforts highlight the operational importance of suicide-risk modeling in mission-relevant populations and motivate generalizable methods that can be validated on large-scale EHR data.

Recent network-analysis work further supports a multidimensional view of suicide-related risk by jointly examining sociodemographic, clinical, and psychopharmacological factors in psychiatric cohorts with multi-affected families [12]. Although focused on association modeling rather than largescale EHR prediction, this work reinforces the importance of jointly modeling clinical, behavioral, and contextual factors. Our complementary work in BEACON-SP [13] investigates ontology-grounded GraphRAG for clinician-facing suicideprevention decision support, whereas the present study focuses on retrospective EHR-based risk modeling.

This paper differs from prior work by combining broad diagnosis-category engineering, age-matched cohort construction, probability calibration, robustness testing across random resampling seeds, and SHAP-based feature interpretation within a single MIMIC-IV EHR modeling framework. Rather than focusing only on a specific population, a single model family, temporal structure, or narrative text, our approach evaluates multiple calibrated classifiers on multidimensional EHR-derived features and explicitly examines how agematched resampling affects performance under severe class imbalance.

![](images/bd227ec4da1782127a51790812101dd10b930d149977f7ad08c83279b7dda9ea.jpg)  
Fig. 1. The 53 health-related categories utilized for creating features.

## III. DATA PRE-PROCESSING

The MIMIC-IV dataset contains data from the emergency room and ICU visits of over 200,000 patients, including over 6.3 million recorded diagnoses. Out of these diagnoses, there are over 27,000 unique clinical diagnoses. The 53 clinical categories are constructed using keyword-based mappings that group diagnoses with related clinical meanings, includ ing diagnoses expressed using slightly different terminology. These categories span psychiatric conditions, substance abuse, cognitive and behavioral factors, social determinants of health (SDoH), and other physical health conditions. The complete category set is shown in Fig 1. For each clinical category, we generate two predictor features: (i) a binary indicator representing whether the patient has at least one diagnosis in that category and (ii) a count representing the number of recorded diagnoses within the category. We also include onehot encoded age-group and sex indicators, together with each patient’s total number of recorded diagnoses. The age groups are 18–25, 26–35, 36–45, 46–55, 56–65, 66–75, 76–85, and 86+. The resulting feature vector contains 117 predictors: 106 category-derived features, 8 age-group indicators, 2 sex indicators, and total diagnostic count.

The binary outcome is constructed from documented suicide-related diagnosis and external-cause codes. ICD-9- CM codes E950–E959 identify suicide and self-inflicted injury events, while ICD-9-CM code V62.84 identifies suicidal ideation. ICD-10-CM codes X71–X83 identify intentional self-harm by specified mechanisms other than poisoning and asphyxiation; ICD-10-CM code T14.91 identifies suicide attempt; ICD-10-CM code R45.851 identifies suicidal ideation; and ICD-10-CM code Z91.51 identifies a personal history of suicidal behavior. We also include intentional self-harm poisoning codes within ICD-10-CM ranges T36–T50 when the intent character indicates intentional self-harm.

Patients with at least one diagnosis or external-cause code satisfying this composite suicide-related outcome definition are assigned to the positive, or high-risk, cohort. Patients without a documented code satisfying this outcome definition are assigned to the negative, or comparison, cohort. The terms high-risk cohort and control cohort are used as operational

TABLE I  
CATEGORIES WITH THE TEN LARGEST ODDS RATIOS FOR THE COMPOSITESUICIDE-RELATED OUTCOME.
<table><tr><td>Feature</td><td>High-Risk Cohort Rate</td><td>Control Cohort Rate</td><td>Odds Ratio</td></tr><tr><td>SDOH Violence</td><td>29.6%</td><td>1.2%</td><td>34.62</td></tr><tr><td>Mood Disorders</td><td>18.8%</td><td>0.9%</td><td>26.13</td></tr><tr><td>PTSD</td><td>21.3%</td><td>1.6%</td><td>17.03</td></tr><tr><td>SDOH Housing</td><td>17.6%</td><td>1.3%</td><td>16.62</td></tr><tr><td>Depression</td><td>75.6%</td><td>16.6%</td><td>15.53</td></tr><tr><td>Personality Disorders</td><td>26.7%</td><td>2.8%</td><td>12.60</td></tr><tr><td>Schizoaffective Disorder</td><td>5.9%</td><td>0.6%</td><td>10.89</td></tr><tr><td>Depressant Abuse</td><td>13.9%</td><td>1.7%</td><td>9.49</td></tr><tr><td>ADHD</td><td>8.7%</td><td>1.3%</td><td>7.15</td></tr><tr><td>Schizophrenia</td><td>5.8%</td><td>0.9%</td><td>6.66</td></tr></table>

TABLE II

CATEGORIES WITH THE TEN SMALLEST ODDS RATIOS FOR THECOMPOSITE SUICIDE-RELATED OUTCOME.
<table><tr><td>Feature</td><td>High-Risk Cohort Rate</td><td>Control Cohort Rate</td><td>Odds Ratio</td></tr><tr><td>Cancer</td><td>7.9%</td><td>22.2%</td><td>0.30</td></tr><tr><td>Vascular</td><td>13.1%</td><td>31.3%</td><td>0.33</td></tr><tr><td>Cardiovascular</td><td>35.9%</td><td>61.9%</td><td>0.34</td></tr><tr><td>Hematologic Malignancy</td><td>0.8%</td><td>2.3%</td><td>0.35</td></tr><tr><td>Endocrine</td><td>29.1%</td><td>52.8%</td><td>0.37</td></tr><tr><td>Procedural Status</td><td>7.7%</td><td>16.6%</td><td>0.42</td></tr><tr><td>Kidney Disease</td><td>15.0%</td><td>28.5%</td><td>0.44</td></tr><tr><td>Cancer Specific</td><td>1.7%</td><td>3.5%</td><td>0.47</td></tr><tr><td>Hematologic</td><td>24.6%</td><td>40.1%</td><td>0.49</td></tr><tr><td>Diabetes</td><td>13.0%</td><td>22.9%</td><td>0.50</td></tr></table>

EHR-derived study labels and should not be interpreted as prospective clinical risk determinations. Outcome-defining codes were used only for label construction and were excluded from the predictor feature set.

## IV. EXPLORATORY DATA ANALYSIS

Initial exploratory analysis was performed using odds ratios to identify categories associated with the composite suiciderelated outcome. For a binary category feature, the odds ratio quantifies the odds of that category being present in the highrisk cohort relative to the control cohort. Odds ratios greater than 1 indicate that the category is more prevalent in the highrisk cohort, whereas odds ratios less than 1 indicate that the category is more prevalent in the control cohort.

Table I reports the ten categories with the largest odds ratios. Among the SDoH-related categories, SDOH Housing primarily represents diagnoses related to homelessness or housing instability, whereas SDOH Violence captures diagnoses related to abuse or violence exposure, including physical or sexual abuse. The High-Risk Cohort Rate column reports the percentage of patients satisfying the composite suicide-related outcome definition who have at least one diagnosis in the corresponding category. The Control Cohort Rate column reports the corresponding percentage among patients without documented positive outcome codes.

Table II shows the 10 features with the lowest odds ratios. It is worth noting that the difference between Cancer and Cancer Specific is that Cancer encompasses more common and general cancer-related diagnoses, whereas Cancer Specific contains rarer and more specific cancer-related diagnoses.

## V. AGE-MATCHED RESAMPLING

The original dataset is highly imbalanced, containing 11,197 patients in the high-risk cohort and 212,255 patients in the control cohort, corresponding to a positive outcome rate of 5.0%. To construct an age-balanced analytical cohort, we retain all patients in the high-risk cohort and sample three comparison-cohort patients from the same age group for each high-risk-cohort patient. This procedure produces a 3:1 comparison-to-high-risk ratio and a 25% positive outcome rate within each age group and across the matched cohort overall. Because this resampled cohort does not preserve the original outcome prevalence, results obtained on this cohort characterize model performance under age-matched evaluation conditions rather than expected clinical deployment performance.

We also conducted preliminary experiments with Synthetic Minority Over-Sampling Technique (SMOTE) [14], Adaptive Synthetic Sampling Approach for Imbalanced Learning (ADASYN) [15], and decision-threshold tuning. These approaches were not retained for the primary analysis because age-matched cohort construction provided the strongest validation performance while explicitly controlling the age distribution of the control cohort. In addition, class weighting was applied during model training: logistic regression and random forest used balanced class weight, and XGBoost used the balance of positive and negative weights based on the training-cohort class ratio.

## VI. SUICIDE-RELATED RISK MODELING

We evaluate three supervised classifiers: logistic regression, random forest, and XGBoost. Each model is hyperparameter tuned and calibrated using an age-group-stratified 70/30 train/test split, with hyperparameters selected by average F1 score from five-fold cross-validation on the training set.

We organize the analysis into three experiments. Experiment 1 compares the three classifiers before and after agematched resampling using a single seed, and further characterizes the best-performing model through feature importance and age-group performance. Experiment 2 quantifies the stability of the best-performing model across 20 random seeds. The third experiment complements the matched-cohort evaluation with a held-out comparison-cohort analysis, examining which documented features drive elevated predicted probabilities. The matched-cohort experiments quantify performance under controlled age-group representation, whereas the heldout analysis examines model behavior among comparisoncohort patients excluded from matched training.

## A. Hyperparameter Tuning

Each model is hyperparameter tuned using average F1 score across five-fold cross-validation on the training set as the selection metric. The full set of tuned hyperparameters and the corresponding values tested are presented in Table III.

TABLE III  
HYPERPARAMETERS TUNED AND THE VALUES TESTED FOR EACH MODEL.
<table><tr><td>Model</td><td>Feature</td><td>Values Tested</td></tr><tr><td>Logistic Regression</td><td>C</td><td>0.01, 0.1, 1, 10</td></tr><tr><td>Logistic Regression</td><td>Penalty</td><td>L1, L2</td></tr><tr><td>Random Forest</td><td>num estimators</td><td>200,400</td></tr><tr><td>Random Forest</td><td>max depth</td><td>None, 20, 30</td></tr><tr><td>Random Forest</td><td>min samples split</td><td>2,5</td></tr><tr><td>XGBoost</td><td>num estimators</td><td>100, 200, 300</td></tr><tr><td>XGBoost</td><td>max depth</td><td>5,7,9</td></tr><tr><td>XGBoost</td><td>learning rate</td><td>0.01, 0.05, 0.1, 0.2</td></tr></table>

![](images/0f6b42a9701b0d602e503ba0e1617b6c9fed96caf62369d3c53460f303977d1a.jpg)  
Fig. 2. The impact of isotonic model calibration.

## B. Model Calibration

Each classifier is calibrated using CalibratedClassifierCV from the scikit-learn library. Calibration aligns predicted probabilities with observed outcome frequencies in the evaluation cohort. We use isotonic calibration as it does not impose a fixed sigmoid mapping between classifier scores and calibrated probabilities.

Fig. 2 compares the calibrated and uncalibrated XGBoost models. Prior to calibration, the XGBoost calibration curves fall below the ideal calibration line, indicating that predicted probabilities exceed observed positive outcome rates over much of the probability range. Isotonic calibration substantially improves agreement between predicted probabilities and observed rates within both evaluated cohorts. Importantly, calibration on the age-matched cohort reflects its constructed 25% positive outcome prevalence and should not be interpreted as calibration under real-world deployment prevalence.

## VII. EXPERIMENTS AND RESULTS

## A. Experiment 1: Classifier Comparison Results

The first experiment compares logistic regression, random forest, and XGBoost using a single age-group-stratified 70/30 split. Each model is first evaluated on the original imbalanced cohort and then re-evaluated after age-matched resampling, isolating the effect of cohort construction on classification performance.

TABLE IV  
CLASSIFICATION RESULTS FOR THE COMPOSITE SUICIDE-RELATEDOUTCOME BEFORE AND AFTER AGE-MATCHED RESAMPLING.
<table><tr><td>Model</td><td>Precision</td><td>Recall</td><td>F1</td><td>AUPRC</td><td>AUROC</td></tr><tr><td>Logistic Regression (Before)</td><td>0.7282</td><td>0.4475</td><td>0.5543</td><td>0.6567</td><td>0.9543</td></tr><tr><td>Logistic Regression (After)</td><td>0.7781</td><td>0.7782</td><td>0.7816</td><td>0.8377</td><td>0.9343</td></tr><tr><td>Random Forest (Before)</td><td>0.7429</td><td>0.5350</td><td>0.6220</td><td>0.6896</td><td>0.9614</td></tr><tr><td>Random Forest (After)</td><td>0.8157</td><td>0.7812</td><td>0.7981</td><td>0.8701</td><td>0.9501</td></tr><tr><td>XGBoost (Before)</td><td>0.8175</td><td>0.5799</td><td>0.6785</td><td>0.7573</td><td>0.9603</td></tr><tr><td>XGBoost (After)</td><td>0.8434</td><td>0.7806</td><td>0.8108</td><td>0.8891</td><td>0.9533</td></tr></table>

![](images/f79b1d7d6d7dd4468daff2bf200ed0918c59b3db95abc2fb121b685313720065.jpg)  
Fig. 3. Precision-recall curves of all three models after age-matched resampling.

The full classifier results are shown in Table IV. The XGBoost model had the strongest results both before and after age-matched resampling, with a 0.6785 F1 score and a 0.7573 Area Under the Precision-Recall Curve (AUPRC) prior to age-matched resampling, compared to a 0.8108 F1 score and a 0.8891 AUPRC after age-matched resampling. All three models achieve higher F1 and AUPRC under the age-matched 25% prevalence setting.

Fig. 3 presents the precision-recall curves for the three classifiers following age-matched resampling. XGBoost achieves the strongest overall precision-recall performance, with an average precision of 0.889, outperforming logistic regression and random forest across much of the recall range. This result indicates that XGBoost provides a better trade-off between precision and recall for identifying patients in the positive outcome cohort under the age-matched evaluation setting.

1) XGBoost Feature Importance: Model-derived feature importance values from XGBoost were examined to identify the factors contributing most strongly to classification decisions in the age-matched cohort. These values describe relative contributions within the trained classifier and should not be interpreted as causal effects or independent clinical risk factors. The most influential feature was the presence of depression, followed by personality disorders, social determinants of health related to violence, mood disorders, and schizophrenia. The top 15 features for the calibrated XGBoost model after age-matched resampling are shown in Fig. 4.

![](images/12e75086cecf9367e28a24d9e2f71dd53b160049e04b2869add5e7f3d2ac9c4d.jpg)  
Fig. 4. Top 15 XGBoost feature importance values for classification of the composite suicide-related outcome after age-matched resampling.

TABLE V  
AGE-GROUP-SPECIFIC RESULTS FOR THE CALIBRATED XGBOOST CLASSIFIER WITHIN THE AGE-MATCHED EVALUATION COHORT.
<table><tr><td>Age Group</td><td>Test Size</td><td>Precision</td><td>Recall</td><td>F1 Score</td></tr><tr><td>18-25</td><td>4601</td><td>0.8418</td><td>0.8330</td><td>0.8374</td></tr><tr><td>26-35</td><td>2696</td><td>0.8547</td><td>0.8027</td><td>0.8279</td></tr><tr><td>36-45</td><td>2167</td><td>0.8466</td><td>0.7841</td><td>0.8142</td></tr><tr><td>46-55</td><td>1993</td><td>0.8620</td><td>0.7651</td><td>0.8106</td></tr><tr><td>56-65</td><td>1207</td><td>0.8171</td><td>0.6953</td><td>0.7513</td></tr><tr><td>66-75</td><td>463</td><td>0.7917</td><td>0.6552</td><td>0.7170</td></tr><tr><td>76-85</td><td>241</td><td>0.7272</td><td>0.4000</td><td>0.5161</td></tr><tr><td>86+</td><td>69</td><td>0.8750</td><td>0.4118</td><td>0.5600</td></tr></table>

2) Performance by Age Group: Table V reports the test size, precision, recall, and F1 score for the calibrated XG-Boost classifier within each age group. Because age-matched resampling enforces a 25% positive outcome rate within every age group, differences in these metrics reflect differences in classification difficulty across age groups under controlled class representation rather than differences in outcome prevalence within the evaluated cohort.

Performance is strongest for patients between 18 and 55 years of age, with F1 scores exceeding 0.81 across the four youngest age groups. Performance decreases for older age groups, with the largest declines occurring among patients aged 76 years or older. This decrease is driven primarily by lower recall where the model identifies a smaller fraction of positive outcome cases in older age groups even under matched evaluation conditions. These findings should be interpreted cautiously because the oldest age groups have substantially smaller evaluation sample sizes. Additional evaluation using larger older-adult cohorts and untouched outcome prevalence is needed before drawing conclusions regarding age-specific clinical utility.

## B. Experiment 2: Robustness Results

To evaluate sensitivity to the random selection of comparison-cohort patients, we repeat age-matched resampling and evaluation of the calibrated XGBoost classifier with optimal hyperparameters across 20 randomly selected seeds.

TABLE VI  
PERFORMANCE OF THE CALIBRATED XGBOOST CLASSIFIER WITH AGE-MATCHED RESAMPLING ACROSS 20 RANDOM SEEDS.
<table><tr><td>Metric</td><td>Average Value</td><td>95% CI Lower Bound</td><td>95% CI Upper Bound</td></tr><tr><td>Precision</td><td>0.8368</td><td>0.8335</td><td>0.8401</td></tr><tr><td>Recall</td><td>0.7779</td><td>0.7744</td><td>0.7814</td></tr><tr><td>F1 Score</td><td>0.8062</td><td>0.8042</td><td>0.8083</td></tr><tr><td>AUPRC</td><td>0.8904</td><td>0.8888</td><td>0.8921</td></tr><tr><td>AUROC</td><td>0.9545</td><td>0.9539</td><td>0.9552</td></tr></table>

TABLE VII

TOP 10 FEATURES BY MEAN ABSOLUTE SHAP VALUE AMONG HIGH-SCORING PATIENTS IN THE HELD-OUT CONTROL COHORT.
<table><tr><td>Feature</td><td>Mean Abs. SHAP</td><td>SD</td><td>CV</td></tr><tr><td>Total diagnoses</td><td>0.2176</td><td>0.0115</td><td>0.0528</td></tr><tr><td>Has depression</td><td>0.0900</td><td>0.0077</td><td>0.0858</td></tr><tr><td>Count depression</td><td>0.0818</td><td>0.0055</td><td>0.0674</td></tr><tr><td>Age 76–85</td><td>0.0593</td><td>0.0128</td><td>0.2164</td></tr><tr><td>Has Personality Disorders</td><td>0.0570</td><td>0.0070</td><td>0.1220</td></tr><tr><td>Has SDOH Violence</td><td>0.0496</td><td>0.0093</td><td>0.1869</td></tr><tr><td>Age 66–75</td><td>0.0437</td><td>0.0077</td><td>0.1754</td></tr><tr><td>Count Cardiovascular</td><td>0.0435</td><td>0.0054</td><td>0.1237</td></tr><tr><td>Count GI Disease</td><td>0.0384</td><td>0.0037</td><td>0.0968</td></tr><tr><td>Has Mood Disorders</td><td>0.0335</td><td>0.0067</td><td>0.1991</td></tr></table>

The seeds are drawn without replacement from the integers between 1 and 100. For each seed, the full age-matched resampling process is performed, with the seed determining both which comparison-cohort patients are retained and the train/test split. For every seed we record accuracy, precision, recall, F1 score, AUPRC, Area Under the Receiver Operating Characteristic (AUROC), and we report the average and 95% confidence interval of each metric.

Table VI reports the average performance and corresponding 95% confidence intervals. The calibrated XGBoost classifier achieved an average F1 score of 0.8062 with a 95% confidence interval of [0.8042, 0.8083], and an average AUPRC of 0.8904 with a 95% confidence interval of [0.8888, 0.8921]. The narrow confidence intervals across evaluation metrics indicate that classifier performance within the age-matched cohort is stable across the evaluated random resampling seeds.

## C. Experiment 3: Held-Out Comparison-Cohort Results

The third experiment complements matched-cohort evaluation with a held-out comparison cohort excluded from training. For each seed, the training cohort contains all highrisk patients and three age-matched comparison patients per positive case; the remaining comparison patients, none with documented positive outcome codes, form the held-out set. Calibrated XGBoost is trained on the matched cohort and applied to the held-out cohort. We then examine the top 5% of predicted probabilities to identify features associated with elevated scores among patients without documented positive outcomes.

The primary interpretability metric is the mean absolute SHAP value averaged across the 20 experimental seeds, which summarizes the magnitude of each feature’s contribution to the model output independent of direction. We also report the standard deviation across seeds and the coefficient of variation (CV), where smaller CV values indicate more consistent feature contribution estimates.

![](images/b0f5b3173ecb017a5dfe4352af8b2ec57dce5bffa89a4dcff42f938a33e5f726.jpg)  
Fig. 5. Distribution of SHAP values for total diagnoses among high-scoring patients in the held-out control cohort.

Table VII shows the ten features with the largest mean absolute SHAP values among high-scoring patients in the held-out comparison cohort. The feature total diagnoses is the dominant contributor to elevated predicted probabilities, with a mean absolute SHAP value of 0.2176 and a low CV of 0.0528. Several SHAP-identified contributors, including depression-related features, personality-disorder history, SDoH-related violence, and mood disorders, are consistent with the strongest associations observed in the odds-ratio analysis. This agreement suggests that key psychiatric and social-contextual categories remain important across both univariate association analysis and multivariate model interpretation. At the same time, SHAP also highlights diagnostic burden, older age indicators, and medical comorbidities such as cardiovascular and gastrointestinal disease, indicating that the model integrates broader clinical context beyond the categories with the largest individual odds ratios. Fig. 5 shows the distribution of SHAP values for total diagnoses among these high-scoring patients. Note that these results should not be interpreted as demonstrating confirmed false positives. A patient assigned to the control cohort may lack a documented positive outcome code in the available EHR record while still exhibiting unobserved, undocumented, or future suicide-related risk. Instead, this analysis characterizes which documented features cause the model to assign elevated scores among patients without recorded positive outcomes.

## VIII. CONCLUSION

This paper presented a multidimensional machine learning framework for retrospective suicide-related risk modeling using MIMIC-IV EHR data. We constructed clinically motivated diagnosis categories spanning psychiatric, medical, behavioral, and social-determinant-related factors and defined a composite outcome from documented codes for suicidal ideation, suicide attempt, prior suicidal behavior, and intentional self-harm. To address class imbalance and reduce agerelated confounding, we applied age-matched cohort construction and evaluated logistic regression, random forest, and XG-Boost with isotonic calibration. All classifiers achieved higher F1 and AUPRC under age-matched evaluation conditions, with calibrated XGBoost achieving the strongest matchedcohort results. Robustness testing across 20 random seeds showed stable performance, and SHAP-based interpretation identified diagnostic burden, depression-related features, age, personality-disorder history, SDoH-related violence, and medical comorbidity as major contributors to elevated predicted probabilities. The evaluation results show that broad EHRderived features, age-matched cohort construction, calibration, and feature-level interpretation provide a practical framework for suicide-related risk modeling. Future work will extend this approach toward patient-level encounter-sequence and timeto-event modeling to capture how suicide-related risk evolves over a patient’s documented history.

## REFERENCES

[1] Center for Disease Control and Prevention, “Suicide data and statistics,” 2025.

[2] A. E. Johnson et al., “MIMIC-IV, a freely accessible electronic health record dataset,” Scientific data, vol. 10, no. 1, p. 1, 2023.

[3] I. Bayramli et al., “Temporally informed random forests for suicide risk prediction,” Journal of the American Medical Informatics Association, vol. 29, pp. 62–71, 2022.

[4] Q. Chen et al., “Predicting suicide attempt or suicide death following a visit to psychiatric specialty care: A machine learning study using Swedish National Registry Data,” PLOS Medicine, vol. 17, 2020.

[5] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” Advances in neural information processing systems, vol. 30, 2017.

[6] J. B. Edgcomb, R. Thiruvalluru, J. Pathak, and J. O. Brooks, “Machine learning to differentiate risk of suicide attempt and self-harm after general medical hospitalization of women with mental illness,” Medical Care, vol. 59, 2021.

[7] K. R. Conner, J. C. Kearns, and L. M. Denneson, “Qualitative analysis of hospital patient narratives of warning signs on the day of their suicide attempt,” General Hospital Psychiatry, vol. 79, pp. 146–151, 2022.

[8] C. J. Kennedy, J. C. Kearns, and J. C. Geraci, “Predicting suicides among U.S. Army soldiers after leaving active service,” JAMA Psychiatry, vol. 12, pp. 1215–1224, 2024.

[9] J. C. Kearns et al., “A practical risk calculator for suicidal behavior among transitioning U.S. Army soldiers: Results from the Study to Assess Risk and Resilience in Servicemembers-Longitudinal Study (STARRS-LS),” Psychological Medicine, vol. 53, pp. 7096–7105, 2023.

[10] J. F. McCarthy et al., “Predictive modeling and concentration of the risk of suicide: Implications for preventive interventions in the U.S. Department of Veterans Affairs,” American Journal of Public Health, vol. 105, pp. 1935–1942, 2015.

[11] A. H. S. Harris, A. K. Finlay, and E. L. Meerwijk, “Evaluating the accuracy of the Veterans Health Administration’s REACH VET suicide prediction model for legal involved veterans,” npj Mental Health Research, vol. 53, 2025.

[12] G. Torino et al., “Suicide attempt risk factors in individuals with psychiatric disorders and multi-affected families: a network analysis,” npj Mental Health Research, 2026.

[13] K. Davaslioglu, N. Conger, S. Kompella, Y. E. Sagduyu, and N. D. Bastian, “Beacon-sp: Ontology-grounded graphrag framework for clinical suicide risk assessment,” in 2026 IEEE Healthcom, pp. 1–6, 2026.

[14] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: Synthetic minority over-sampling technique,” Journal of Artificial Intelligence Research, vol. 16, pp. 321–357, 2002.

[15] H. He, Y. Bai, E. A. Garcia, and S. Li, “ADASYN: Adaptive synthetic sampling approach for imbalanced learning,” in IEEE International Joint Conference on Neural Networks, pp. 1322–1328, 2008.