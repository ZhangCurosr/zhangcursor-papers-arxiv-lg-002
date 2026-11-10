# Replication of Thermal Comfort Cohort Models: No Support for Improved Thermal Preference Prediction

Thomas Hirsch<sup>a,b,∗</sup>, Katja Ceplak<sup>a</sup>, Veronika Windisch<sup>a</sup>, Samina Kadkhoda Masoumali<sup>a</sup>, Matias Quintana<sup>c,d</sup>, Clayton Miller<sup>c</sup>, Stefano Schiavon<sup>e</sup> and Gerald Schweiger<sup>a,b</sup>

<sup>a</sup>TU Wien, Vienna, Austria

<sup>b</sup>Complexity Science Hub, Vienna, Austria

<sup>c</sup>Singapore Management University, Singapore, Singapore

<sup>d</sup>Singapore-ETH Centre, Singapore, Singapore

<sup>e</sup>University ofCalifornia, Berkeley, Berkeley, USA

## A R T I C L E I N F O

Keywords:   
cohort comfort models   
personal thermal comfort model   
supervised machine learning   
classification   
replication study

## A BS T R AC T

Cohort comfort models (CCMs) aim to overcome the extensive individual data requirements of personal comfort models (PCMs) by grouping occupants based on demographic and psychometric characteristics (cold start) or similar thermal preferences (warm start). We evaluated the validity of the reported CCM performance gains [1] through (i) a methodological assessment of the original study, (ii) a replication using one of the two original datasets (DORN: 20 participants, 6 months) under a revised experimental protocol, and (iii) a replication using an independently collected dataset (PersonAI: 72 participants, 4 weeks). Classification performance was evaluated using F1-micro, as in the original study, and F1-macro to account for class imbalance. Under F1-micro, a trivial baseline that always outputs the majority class performed on par with cohort models. Under F1-macro, all cohort models achieved lower median scores than a simple baseline that makes random predictions based on the estimated distribution of thermal preferences. In our methodological assessment we identified the following key issues: Limited data per participant and per minority class, reliance on a metric insensitive to class imbalance, missing trivial baselines, inadequate cross validation strategy, and multiple occurrences of information leakage. Our findings challenge the original study’s conclusion that CCMs improve thermal preference prediction. CCMs did not outperform simple class-distribution baselines. Future studies should improve experimental designs, including controlled thermal conditions, to obtain suficient observations across thermal preference classes.

## 1. Introduction.

Buildings face a dual and often conflicting challenge. They play a central role in global eforts to reduce greenhouse gas emissions, accounting for approximately 40% of worldwide energy demand [2]. At the same time, they must provide indoor conditions that support health, comfort, and performance, as people spend the vast majority of their lives indoors [3]. Over the past decades, researchers and standardization bodies have developed methods to predict indoor environmental quality and occupant comfort that are built on aggregated representations of how a typical occupant perceives the thermal environment [4]. These aggregated representations are inherently limited because they estimate group averages and therefore often predict poorly for individuals. As a result, designing and operating buildings<sub>a</sub> according to such models can fail to improve real occupant satisfaction and may instead encourage conservative HVAC sizing and scheduling, increasing energy use without reliably reducing discomfort [4]. To address the limitations of aggregate models, data-driven Personal Comfort Models (PCMs) have been proposed. PCMs are a new approach to thermal comfort modeling that predicts individuals’ thermal comfort responses, instead of the average response of a large population and it can be applied to any HVAC system [5]. The increasing availability of of-the-shelf wearable devices has strengthened this approach by enabling the collection of physiological measurements, such as skin temperature and heart rate, which have been shown to improve prediction accuracy. Yet collecting longitudinal field data remains costly and burdensome, as it requires repeated Right-Here-Right-Now (RHRN) surveys during daily activities and high-resolution environmental monitoring that is dificult to sustain at scale [6]. To address the scalability limitations of personalized models, Cohort Comfort Models (CCMs) have been proposed, exploiting similarities between occupants to estimate the thermal preferences of new individuals with little or no historical data [1]. By estimating preferences based on similarities between occupants, CCMs combine the personalization of individua models with the data eficiency of aggregate approaches.

If CCMs are to inform real design and operational decisions at scale, their empirical claims must be suficiently robust. At present, no independent replication has evaluated these claims, and the statistical methods and machine learning procedures used in the original study have not yet been independently re-examined for reproducibility, robustness, or potential sources of bias.

## 1.1. Main Contribution

To evaluate the credibility of the CCMs, we pursue a multi-level validation:

• Methodological assessment. We first reviewed the original experimental setup, including the data splitting strategy, model training procedure, and evaluation protocol. Based on this assessment, we revised the implementation to conform to established Machine Learning (ML) best practices and to eliminate methodological shortcomings that could compromise the validity of the reported findings. The resulting protocol, which preserves the original research objective while correcting the identified implementation shortcomings, is hereafter referred to as the revised experimental setup.

• Replication using the original data. We examined whether the original conclusions hold when the study is repeated using the revised experimental setup and one of the original datasets. We also requested access to the second dataset used in the original study and contacted the original authors; however, this dataset was unavailable

• Replication using independently collected data, generating the PersonAI dataset. We examined whether the original conclusions generalize to an independent dataset collected through a field experiment involving 80 participants, using the revised experimental setup.

The two replication studies preserve the original research hypothesis and experimental design while introducing deliberate changes to the implementation. Under the framework of Desai et al. [7], they are classified as direct replications, whereas under that of Gundersen [8], they are classified as interpretation reproducibility studies.

## 2. Background

Academic success depends heavily on publication records, which influence funding [9], promotion [10], and tenure [11]. However, scientific publishing favors novel, positive, and unambiguous findings over negative results and replication studies [12]. Together, these factors create structural incentives for researchers to prioritize novel and positive findings over replication and rigorous methodological scrutiny. This current practice is defensible only under the implicit assumption that the large majority of published discoveries are correct [13]. If science were to produce numerous major discoveries, most of which are true and accurate, allocating resources to replication would indeed appear to be of low priority. However, empirical evidence across diverse scientific fields suggests that this narrative is untenable [14, 15, 16]. Science is a dificult endeavor, and numerous biases afect diferent study designs and disciplines [17]; even with the best intentions, findings can easily be false, exaggerated, or inflated. In scientific systems where genuine discoveries are not abundant and the proportion of false findings is appreciable, replication may have greater scientific value than an additional original discovery claim. Moreover, if false findings carry substantial consequences (such as wasted resources, misdirected research eforts, or harm to end users), original discovery work may even have negative value [13]. Beyond the structural disincentives that discourage replication eforts, a second and independent challenge concerns the practical feasibility of reproducibility itself. Even when researchers attempt to verify prior findings, reproduction may fail because methods are not described in suficient detail, analyses are incompletely or selectively reported, or the underlying data are unavailable [18]. In AI research, these general problems are compounded by additional barriers, including under-specified model architectures and training procedures, missing information about data preprocessing, selective reporting of results, and the absence of code and data [8, 19]. Some contemporary AI models also require substantial computational infrastructure for training and evaluation; although reproducible in principle, their cost and scale restrict independent verification and thereby limit reproducibility in practice.

As in many other applied engineering disciplines, research on buildings has closely followed the recent wave of enthusiasm for AI [20]. AI has been applied to system control [21], fault detection [22], and energy prediction [23], as well as to the modelling of IEQ, including personalised thermal preference modelling [24]. While replication studies in engineering remain relatively rare, systematic investigations into the reproducibility of machine learning research have begun to emerge [25, 26].

## 2.1. From Personal to Cohort Comfort Models

The Predicted Mean Vote (PMV) model, introduced by Fanger [27], has long served as the standard for thermal comfort assessment in buildings. Designed to predict the thermal sensation of a group of persons, the PMV acknowledges individual variation but sets it aside. It is also used to predict the comfort of a single person, for example someone occupying a private ofice. A related population-level approach, the Adaptive model [28], similarly averages individual responses by linking comfort to prevailing outdoor temperatures. Both models have proven useful at the aggregate level, yet studies using the ASHRAE Global Thermal Comfort Database II [29] reveal that the PMV achieves only 33% accuracy when predicting the thermal sensation of a specific individual [30]; an unsatisfying result for occupant-centric applications. The proliferation of IoT sensors and wearable devices opened the door to a more granular alternative: PCMs [5]. By pairing direct occupant feedback with environmental and/or physiological and/or behavioural data collected longitudinally in the field, PCMs can be tailored to an individual’s specific preferences and have been shown to substantially outperform the PMV at the personal level [31]. Despite their promise, PCMs carry a significant practical burden: collecting the 60–200 labeled data points per occupant required for stable rediction [31, 32, 33, 34] is logistically demanding, disruptive to daily activities, and cost-prohibitive at scale, particularly in flexible workspaces where occupancy turnover is high. CCMs [1] are introduced as a middle path between these two extremes. Rather than averaging across an entire population or waiting to accumulate suficient individual data, CCMs identify groups of occupants who share underlying thermal preference patterns and leverage their pooled historical data to generate predictions for a new arrival. By drawing on onboarding surveys, physiological and environmental sensor measurements, and as few as a single labeled comfort response, CCMs seek to capture the personalization benefits of PCMs without the associated data collection burden.

## 2.2. Cohort Comfort Models

Building on the personal comfort modeling framework established by Kim et al. [31, 5], Quintana et al. [1] addressed a fundamental limitation of PCMs: the requirement to collect a large number of labeled data points from every individual occupant before a useful model can be trained. Where Kim et al. demonstrated that roughly 60 survey responses per occupant are needed for stable prediction, Quintana et al. asked whether historical data from similar occupants could substitute for (or significantly reduce) that individual data collection burden. To this end, they introduced CCMs, a framework that clusters occupants into groups based on shared thermal preference characteristics and uses the resulting group model to predict the comfort of a new arrival with little or no personal history. The framework was validated on two longitudinal datasets totalling 55 participants and more than 6000 RHRN thermal preference surveys: the DORN dataset (20 occupants, Singapore, 6 months) and the SMC dataset (37 occupants, Redwood City CA, 12 weeks). The cohort creation followed two strategies. In the cold start scenario (directly analogous to the onboarding challenge Kim et al. identified) cohorts were formed using only one-time survey responses (demographic data, the Satisfaction With Life Scale, the Highly Sensitive Person Scale, and the Big Five Personality inventory), requiring no longitudinal comfort data from the new occupant whatsoever. In the warm start scenario, cohorts were constructed from cross-model performance evaluation and response distribution similarity between occupants’ PCMs, and a new occupant was assigned to a cohort using as few as a single labeled RHRN data point. Random Forest classifiers, the same algorithm identified as top-performing by Kim et al. [31], were used throughout, and model performance was evaluated with the F1-micro score. They found that warm start approaches consistently outperformed both the general-purpose aggregate model and the cold start approaches, achieving median F1-micro scores of 0.61–0.72 across the two datasets. At the occupant level, they found that roughly half of participants in the DORN dataset and one third in the SMC dataset experienced an average improvement of 8% and 5% respectively over the general-purpose baseline, with gains as large as 36% and 46% for the most responsive individuals. Among cold start approaches, they found that cohorts based on the Satisfaction With Life Scale and the Agreeableness personality trait provided modest but consistent improvements over the general-purpose model, suggesting that even purely demographic and psychometric information can meaningfully stratify occupant thermal preferences. They conclude that CCMs can match or exceed the predictive accuracy of general-purpose data-driven models while requiring as little as a single labeled observation from the new occupant, substantially reducing the data collection burden that Kim et al. [31] had identified as the primary practical obstacle to personalized thermal comfort modeling at scale.

![](images/e55942542712af56fb8ab9d2de319ac4565fa15354b04d09bac973fab59c4eb4.jpg)  
Figure 1: Overview of the study methodology.

## 3. Method

## 3.1. Study Design and Data Collection

Over a four-week period, we collected subjective, physiological, and environmental data to investigate thermal comfort under real-life conditions. Diferent wearable devices and sensors were used for this purpose (Figure 1). The data were then used to train the CCMs, which established separate groups based on diferent thermal preferences. This study was approved by the ethical committee of the Graz University of Technology, Austria (GZ: EK-35/2024). As a participation compensation, the participants kept the Apple Watches used in the study.

## 3.1.1. Participants

A total of 83 individuals volunteered to participate in the study. Participants were excluded from the final data set if they did not complete the four-week data collection, if their environmental or physiological data were missing, or if the timestamps from diferent devices did not match. Therefore, the total sample consisted of 72 participants, of whom 40 were women (age: M = 32.3, SD = 9.0, range = 18 - 64). The sample was highly educated: nine participants (12.5%) held a PhD, 28 (38.9%) a master’s degree, and 20 (27.8%) a bachelor’s degree, while 14 (19.4%) had completed upper secondary education and one (1.4%) compulsory schooling only. Based on the Body-Mass-Index and self-report, participants were healthy (BMI; M = 23.4, SD = 3.4 ) and the majority reported participating in regular exercise (92%). Living arrangements were skewed towards living in an apartment (75%), followed by a single-family home (19%) and, lastly, a townhouse (6%) The sample was recruited through social media and word of mouth. The inclusion criteria for participation were 18 to 65 years of age, working in an ofice setting and owning an iPhone with an IOS 17 operating system or later, which supported an Apple Watch Series 9. The exclusion criteria consisted of having a chronic condition and taking medication that might interfere with perception of air quality or thermoregulation. After potential participants disclosed their chronic conditions and medications in the screening survey, the evaluation was done on a case-by-case basis.

## 3.1.2. Wearable sensors and devices

Each participant was equipped with an Apple Watch, an Oura Ring, and two environmental sensors.

Apple Watch. Apple Watches Series 9 were used in the study. The watches were coupled with the participants’ own iPhones. The Cozie app [35] was used to present RHRN surveys directly on the Apple Watch screen and to obtain heart rate (HR) data collected by the Apple Watch. Participants were instructed to wear their watch during the scheduled RHRN surveys, from 08:00 to 20:00 every day.

Oura Ring. In addition to smart watches, Oura Rings Gen4 were also used in the study. In order for the smart ring to collect data, the Oura app needed to be installed on the participants’ own phone. Participants were instructed to wear the ring as much as possible, however, they were free to remove it at night if discomfort occurred. No restrictions were made on the visibility and accessibility of the app features. HR data from the Oura Ring were used as the primary physiological input for the comfort models, as they were available at a higher temporal resolution and more consistently throughout the day, including around RHRN prompts. The smart ring measures daytime HR for one minute every five minutes [36] whereas the Apple Watch records background HR at irregular intervals depending on the person’s activity [37]. Oura-derived HR has also shown high correspondence with a research-grade reference for resting HR under real-life conditions [38].

AirCo2ntrol 5000. Each participant was equipped with two indoor air quality sensors. The AirCo2ntrol 5000 device [39], was used to measure temperature, relative humidity, and $\mathrm { C O } _ { 2 } \mathrm { - c o n c e n t r a t i o n }$ in 5 second intervals. Participants were instructed to place one sensor on their work desk and one in their home, specifically in the room where they spent most of their leisure time. They were instructed to not change the location of the sensor for the duration of the study and to not place them in direct sunlight. The sensor did not need manual calibration. It was chosen because of its user-friendliness that allowed the participants to configure the sensor on their own. Due to the participants placing the sensor at their place of work, we opted out of wireless data-transfer for data protection reasons. Data were stored on a SD-card.

## 3.1.3. Questionnaires

Self-report data were collected via two sources: RHRN and an onboarding surveys. The RHRN surveys provided subjective data on the experience of current thermal conditions and contextual information. The onboarding survey provided demographic data, as well as heat and cold tolerance, interoception, mental health and physiological symptoms, personality, and satisfaction with life assessments.

Right-Here-Right-Now. The RHRN survey consisted of five questions that were presented seven times per day, every day between 08:00 and 20:00 in a 105-min interval. This totals to 196 surveys over the span of four weeks. Participants answered the surveys on their Apple Watch screen through the Cozie app [35], allowing them to answer quickly, without much efort or disruption.

The RHRN survey consisted of:

1. How would you prefer to be?

Answers: cooler, no changes, warmer

2. How do you feel right now?

A: good, neutral, bad

3. In this moment, I can feel my heart beat:

A: very strongly, strongly, medium, weak, not at all

4. Where are you?

A: at home, at work, on the go

5. What did you do in the last 10 min?

A: physical activity, desk work, relaxation, other

The questions were used to assess momentary thermal preference (1), current mood (2), body awareness, specifically heartbeat perception (3, [40]), location and proximity to an environmental sensor (4) and activity level in the last 10 min (5). The questions were always presented in the same order. If an inquiry was missed, an RHRN could be manually triggered anytime by opening the Cozie app on the Apple Watch. Participants were instructed to complete a minimum of 147 RHRN surveys (75%) to meet the participation compensation requirement. A question about clothing was not included, as previous work found that adding self-reported clothing information did not consistently improve the predictive performance of models [6].

Heat and cold tolerance. Participants’ self-reported heat and cold tolerance was accessed through a six item questionnaire. The self-constructed questionnaire consisted of three items dedicated to each temperature tolerance. Examples were "I find high temperatures very unpleasant" for heat tolerance and "I tend to freeze faster than others"

for cold tolerance. Participants answered on a five-point Likert scale ranging from "1 - not applicable at all" to "5 - fully applicable". The internal consistency was acceptable for the cold, but questionable for the heat scale (cold: McDonald’s omega = .76, heat: McDonald’s omega =.64).

Interoception. The Multidimensional Assessment of Interoceptive Awareness (MAIA-2 [41]) was used to assess self-reported interoceptive awareness by trait. The questionnaire consisted of eight scales: Noticing, Not-Distracting, Not-Worrying, Attention Regulation, Emotion Awareness, Self-Regulation, Body Listening, and Trust. The items were answered on a six-point Likert scale, ranging from never (0) to always (5). An example of the Not Worrying scale is: “I can notice an unpleasant body sensation without worrying about it”. The internal consistency was satisfactory (McDonalds’s omega = .80 - .93).

Symptoms. The Brief Symptom Inventory (BSI [42] [43]) assessed mental health and psychological symptoms through 53 items that were merged into nine scales: Somatization, Obsession-Compulsion, Interpersonal Sensitivity, Depression, Anxiety, Hostility, Phobic anxiety, Paranoid ideation, and Psychoticism. Items were answered on a fivepoint Likert scale, ranging from not at all (0) to extremely (4). The scales showed adequate internal consistency (McDonald’s omega = .77 - .88).

Personality. The Big Five Personality Inventory (BFI-2-S [44] [45]) examined five personality traits Extraversion, Agreeableness, Conscientiousness, Negative Emotionality and Open-Mindedness with 30 items, six per dimension. Items were answered on a five-point Likert scale, ranging from strongly disagree (1) to strongly agree (5). “I am someone who has dificulty getting started on tasks” is one example of an item. Internal consistency was acceptable (McDonald’s omega = .74 - .89).

Satisfaction with life. Satisfaction with Life Scale (SWLS [46] [47]) consisted of 5 items, which were answered on a seven-point Likert scale, ranging from strongly disagree (1) to strongly agree (7). The instrument assesses cognitiveevaluative and afective components of life satisfaction. An example of an item is: “If I could live my life over, I would change almost nothing”. Internal consistency was suficient (McDonald’s omega = .89).

## 3.1.4. Procedure

The data collection procedure took place in four separate steps. People who were interested in participating in the study first completed the screening survey in which the inclusion and exclusion criteria were checked. Suitable participants were invited to an onboarding event where consent forms were signed, wearables and sensors were distributed and set-up, and tasks were explained. Afterwards, the four-week data collection started in which the participants filled out seven RHRN surveys per day. Participants were also asked to complete the onboarding survey shortly after the start of the study.

## 3.2. Machine Learning: Revised Experimental Setup

As this work is a replication study, we follow the machine learning methodology proposed by Quintana et al. [1]. The original experimental design, model architecture, and cohorting framework are therefore not described again here. Instead, this section focuses exclusively on the methodological and implementation changes introduced in our revised experimental setup.

The revisions were motivated by three objectives:

• correcting methodological shortcomings,

• more robust assessment of classification performance in regards to class imbalances,

• fixing implementation errors (e.g., information leakage and dataset loading issues), and

• reducing unnecessary computational overhead.

Train/Test splitting. Instead of 100 repetitions of randomized 80/20 train test splits we utilized a leave-one-out cross-validation process. The reason for this is two fold: First, the original approach is statistically weak and such obtained results for such small datasets are hard to interpret. Second, the 100 repetitions of random splits of datasets containing less than 50 persons does not increase confidence in the results, especially given the high variance of the obtained results, and does not warrant the computational overhead. To summarize, results obtained from a leave-oneout cross-validation process are easier to interpret while in this special case causing less computational overhead.

Performance metrics. The original publication evaluates classification performance using micro averaged F1, which is sensitive to class imbalance. On imbalanced datasets such as DORN and PersonAI, it can therefore yield inflated and potentially misleading results. We therefore als include macro averaged F1 to provide a more robust evaluation.

Hyperparameter tuning. We omitted hyperparameter optimization and evaluated all models using the default parameters. In the original experimental setup, hyperparameters were optimized independently for each of the 100 randomized train–test splits, resulting in a diferent model specification for each repetition. This simplification also substantially reduced the computational cost of the experiments.

Dataset handling. The DORN dataset in its provided form<sup>1</sup> could not be read with the with the provided CCM code<sup>2</sup>. Adaptions to the dataset loading code were required in order to be able to load the dataset and run the experiments. As the original paper include a dataset version in the form of a repository release, tag, or commit hash, it was not possible to identify the exact software state used to produce the original results.

Feature selection logic. In the original codebase, model inputs were defined by excluding features rather than explicitly specifying the features to include. This approach is prone to errors because newly introduced variables can unintentionally become part of the model input. During our review, we identified a bug that caused information leakage by inadvertently including a feature that was not intended for the corresponding experiment. We therefore replaced the exclusion-based logic with explicit feature selection.

## 3.3. Baselines

We introduce a number of baselines to contextualize and compare the measured performance scores. The Majority classifier does not look at any input features during training, but instead simply determine the majority class found in the training set. On inference, this class is predicted always. For example, the class distribution in the training set is 70% "no change", 20% "prefer cooler" and 10% "prefer warmer", the classifier will always return "no change". The PCM Majority is such a classifier as described above and was "trained" and evaluated alongside the PCM models. The Cohort Majority is such a classifier as described above and was "trained" and evaluated alongside the cohort models. The ApproxDistribution classifier does not look at any input features during training, but instead estimates the target distribution by sampling 10 points from the training set. On inference, a random class value is produced adhering to the learned distribution. For example, 10 points are randomly sampled from the trainig set, there are 7 "no change", 2 "prefer cooler" and 1 "prefer warmer", the classifier will return a random class, with the probability of 0.7 for for "no change", 0.2 for "prefer cooler" and 0.1 for "prefer warmer. The Random F1 scorefor a balanced 3-class problem, being the F1 score produced by a uniformly random choices of classes (uniform distribution) used as ground truth and its predictor, being 0.333 F1 macro/micro.

## 4. Results

## 4.1. Correlation Analysis

Correlation analysis on the PersonAI dataset (see Figure 2) showed that temperature had the strongest linear association with thermal preference (Pearson’s $r = 0 . 4 5 )$ . The remaining environmental and physiological variables were more weakly associated with the outcome: relative humidity $( r = - 0 . 2 0 ) , \mathrm { C O } _ { 2 }$ concentration $( r = - 0 . 0 9 7 )$ , its log-transformed counterpart $( r = - 0 . 1 1 )$ , and heart rate $( r = 0 . 0 2 4 )$

Correlation analysis of the onboarding variables (see Table 1) showed generally weak linear associations with thermal preference. The strongest positive associations were observed for Openness (BFI2S\_Ofenheit; Pearson’s $r = 0 . 2 3 )$ , its short-form counterpart (BFI2XS\_Ofenheit; � = 0.21), and related openness facets, including aesthetic sensitivity $( r = 0 . 1 8 )$ , intellectual curiosity $( r = 0 . 1 8 )$ , and creative imagination (� = 0.16). The strongest negative associations were observed for the anxiety facet (BFI2S\_Ängstlichkeit; � = −0.20), MAIA Trust $( r = - 0 . 1 7 )$ , and the short-form Negative Emotionality scale (� = −0.17). The Satisfaction With Life Scale (SWLS), one of the onboarding variables used for cold-start cohort assignment in the original CCM framework, showed only a weak linear association with thermal preference $( r = - 0 . 0 5 7 )$ ).

![](images/01a9c88145ef0159c785d996f3e581d98f5cd71830542466b3138269bf76a3a6.jpg)  
Figure 2: Pearson correlation matrix between numeric environmental/biological features and the thermal comfort target.

<table><tr><td colspan="2">target_num</td></tr><tr><td> $B F I 2 S _ { O } p e n - M i n d e d n e s s$ </td><td>0.23</td></tr><tr><td> $B F I 2 X S _ { O } p e n - M i n d e d n e s s$ </td><td>0.21</td></tr><tr><td> $B F I 2 S _ { A } e s t h e t i c S e n s i t i v i t y$ </td><td>0.18</td></tr><tr><td> $B F I 2 S _ { I } n t e l l e c t u a l C u r i o s i t y$ </td><td>0.18</td></tr><tr><td> $B F I 2 S _ { C } r e a t i v e I m a g i n a t i o n$ </td><td>0.16</td></tr><tr><td> $B F I 2 S _ { E } n e r g y L e v e l$ </td><td>0.16</td></tr><tr><td> $M A I A _ { N } o t - D i s t r a c t i n g$ </td><td>0.14</td></tr><tr><td> $B F I 2 S _ { A } g r e e a b l e n e s s$ </td><td>0.13</td></tr><tr><td> $B F I 2 S _ { C } o m p a s s i o n$ </td><td>0.12</td></tr><tr><td> $B F I 2 S _ { T } r u s t$ </td><td>0.11</td></tr><tr><td> $B F I 2 X S _ { A } g r e e a b l e n e s s$   $B F I 2 S _ { R } e s p e c t f u l n e s s$ </td><td>0.07</td></tr><tr><td> $B F I 2 S _ { E } x t r a v e r s i o n$ </td><td>0.07</td></tr><tr><td> $B F I 2 S _ { A } s s e r t i v e n e s s$ </td><td>0.06</td></tr><tr><td> $M A I A _ { N } o t - W o r r y i n g$ </td><td>0.06</td></tr><tr><td> $B S I _ { H } o s t i l i t y$ </td><td>0.04</td></tr><tr><td> $B S I _ { A } n x i e t y$ </td><td>0.04</td></tr><tr><td> $B S I _ { S } o m a t i z a t i o n$ </td><td>0.02</td></tr><tr><td> $M A I A _ { E } m o t i o n a l A w a r e n e s s$ </td><td>0.02</td></tr><tr><td> $M A I A _ { A } t t e n t i o n R e g u l a t i o n$ </td><td>0.02</td></tr><tr><td> $B S I _ { P } S T$ </td><td>0.01</td></tr><tr><td> $M A I A _ { S } e l f - R e g u l a t i o n$ </td><td>0.00 -0.04</td></tr><tr><td> $B F I 2 S _ { D } e p r e s s i o n$ </td><td>-0.05</td></tr><tr><td> $M A I A _ { B } o d y L i s t e n i n g$ </td><td>-0.05</td></tr><tr><td> $B F I 2 X S _ { E } x t r a v e r s i o n$ </td><td>-0.06</td></tr><tr><td> $S W L S$ </td><td>-0.06</td></tr><tr><td> $B F I 2 S _ { R } e s p o n s i b i l i t y$ </td><td>-0.08</td></tr><tr><td> $B F I 2 S _ { s } o c i a b i l i t y$ </td><td>-0.09</td></tr><tr><td> $B S I _ { P } D S I _ { t } e m p$ </td><td>-0.10</td></tr><tr><td> $B S I _ { G } S I$ </td><td>-0.10</td></tr><tr><td> $B S I _ { I } n t e r p e r s o n a l S e n s i t i v i t y$ </td><td>-0.10</td></tr><tr><td> $B S I _ { P } a r a n o i d I d e a t i o n$ </td><td>-0.11</td></tr><tr><td> $B F I 2 S _ { P } r o d u c t i v e n e s s$ </td><td>-0.11</td></tr><tr><td> $B F I 2 S _ { E } m o t i o n a l V o l a t i l i t y$ </td><td>-0.12</td></tr><tr><td> $B S I _ { P } h o b i c A n x i e t y$ </td><td>-0.12</td></tr><tr><td> $B S I _ { P } D S I$ </td><td>-0.12</td></tr><tr><td> $M A I A _ { N } o t i c i n g$ </td><td>-0.13</td></tr><tr><td> $B S I _ { o } b s e s s i o n - C o m p u l s i o n$ </td><td>-0.13</td></tr><tr><td> $B F I 2 X S _ { C } o n s c i e n t i o u s n e s s$ </td><td>-0.13</td></tr><tr><td> $B F I 2 S _ { o } r g a n i z a t i o n$ </td><td>-0.13</td></tr><tr><td> $B F I 2 S _ { N } e g a t i v e E m o t i o n a l i t y$ </td><td>-0.14</td></tr><tr><td> $B F I 2 S _ { C } o n s c i e n t i o u s n e s s$ </td><td>-0.14</td></tr><tr><td> $B S I _ { P } s y c h o t i c i s m$ </td><td>-0.14</td></tr><tr><td> $B S I _ { D } e p r e s s i o n$ </td><td>-0.15</td></tr><tr><td> $B F I 2 X S _ { N } e g a t i v e E m o t i o n a l i t y$ </td><td>-0.17</td></tr><tr><td></td><td>-0.17</td></tr><tr><td> $M A I A _ { T } r u s t$   $B F I 2 S _ { A } n x i e t y$ </td><td>-0.20</td></tr></table>

Table 1  
Pearson correlation (r) between aggregated one-time onboarding survey scores and the thermal comfort target.

## 4.2. Influence of metric selection

Figures 3a and 3b show the distributions of per-user F1-micro and F1-macro scores for the DORN and PersonAI datasets, respectively. The boxplots compare the PCM, the PCM Majority baseline, the ApproxDistribution baseline, the best-performing warm-start cohort model, the best-performing cold-start cohort model, cohort baseline (generalpurpose model), and the cohort Majority baseline. The PCM is a Random Forest classifier, the warm start cohort model uses cluster predictions from the combined response-distribution and cross-model performance approach (Cross\_Dist\_survey).

For the DORN dataset, the median F1-micro scores ranked as follows: PCM (≈ 0.74) > PCM Majority $( \approx 0 . 7 0 )$ > Warm Start (≈ 0.69) > Cohort Majority (≈ 0.64) > ApproxDistribution (≈ 0.61) > Cold Start $( \approx 0 . 5 8 ) > \mathrm { C o h o r t }$ Baseline $( \approx 0 . 5 2 )$ . The median F1-macro scores ranked as follows: PCM (≈ 0.51 )> ApproxDistribution (≈ 0.39) > Majority (≈ 0.33) > Warm Start (≈ 0.31) ≈ Cold Start (≈ 0.29) > Cohort Baseline (≈ 0.28) > Cohort Majority $( \approx 0 . 2 7 )$ . The cohort-based models, median scores ranging from approximately 0.29 (cold) to 0.31 (warm) and the cohort baseline with 0.28, stay below the F1 macro score that is achieved by a uniformly random predictor on uniformly random data (0.33).

A similar pattern was observed for the PersonAI dataset (Figure 3b). The median F1-micro scores were closely clustered between approximately 0.71 and 0.78 across all models, including the Cohort and PCM Majority baselines. Using F1-macro, the Warm Start and Cold Start models perform worse than the Cohort basline (approximately 0.30– 0.31 vs. 0.32), all of them staying below the F1 macro score and staying below the F1 score of a uniformly random predictor on uniformly random data (0.33), while the PCM achieved the highest median score (approximately 0.41).

All methods, including the stoachastic or rule based baseline, achieve substantially higher F1-micro than F1-macro scores.

![](images/5f6c037e8882b2bc0250de2218be4ef911d8e68390d4ea68cecea62766bc1b4d.jpg)

(a) DORN  
![](images/9f1d80dc305fec53d88f6e8201bee7ce215a46b976ebc60c31edb18b7da6d42c.jpg)  
(b) PersonAI  
Figure 3: Per-user distribution of F1-micro (blue) and F1-macro (orange) scores for the PCM, PCM Majority Baseline, PCM ApproxDist Baseline, best warm start cohort model, best cold start cohort model, cohort baseline, and the cohort majority baseline. The dashed red line signifies the Random 3-class F1 score.

## 4.3. Classification performance

The following methods were evaluated: the Personal Comfort Model (PCM), the General Purpose model, the Predicted Mean Vote (PMV) model, the Majority Class baselines, the per person random from approximate distribution model (PCM ApproxDistribution), the Cold Start cohort models, the Warm Start cohort models, and the Wrong Cluster controls. Wrong Cluster controls assign participants to incorrect cohorts while keeping the underlying cohort models unchanged. Warm Start models use the ground-truth warm-start cohort assignments (PersonAI\_warm and DORN\_warm). Figures 4 and 5 show the F1-micro scores, computed per fold across all experiments. Each boxplot shows the score distribution across folds.

For the DORN dataset (Figure 4, top), the median F1-micro scores were 0.52 for the General Purpose model, 0.74 for the PCM, 0.51 for PMV, and 0.64 for the Majority Class baseline. Among the cold start approaches the following cohorting methods contradict the expected direction of the ablation: swls (0.41 correct assigned vs. 0.59 incorrect assigned), extraversion (0.40 correct vs. 0.55 incorrect). Conversely, survey (0.57 vs. 0.50), emotional\_stability (0.58 vs. 0.45), conscientiousness (0.57 vs. 0.43), and agreeableness (0.49 vs. 0.38) all show the correct assignment outperforming incorrect assignment, while sex (0.52 vs. 0.53) and openness\_to\_experiences (0.49 vs. 0.50) show essentially no diference between correct and incorrect cluster assignment. The warm start approaches show the largest correct-vs-incorrect gaps of any approach: Cross reaches 0.69 (correct) vs. 0.26 (∼Cross), and Dist\_Cross reaches 0.69 vs. 0.26.

For the PersonAI dataset (Figure 4, bottom), the median F1-micro scores were 0.74 for the General Purpose model, 0.78 for the PCM, and 0.75 for the Majority Class baseline. The Majority Class baseline therefore performed similarly to both the General Purpose model and the PCM. The Cold Start models showed little diference between correct and incorrect cluster assignment. For example, sex achieved median F1-micro scores of 0.71 and 0.74, while agreeableness showed the opposite pattern, with incorrect assignment outperforming correct assignment (0.75 vs. 0.69). In contrast, the Warm Start models showed much larger diferences: Cross achieved 0.75 versus 0.39, and Dist\_Cross achieved 0.76 versus 0.60.

When switching from F1-micro to F1-macro, many of the diferences observed under F1-micro become much smaller. For DORN (Figure 5, top), General Purpose drops to 0.28, PCM to 0.51, PMV to 0.33, and Majority Class to 0.27. All Cold Start models, regardless of whether cluster assignment was correct or incorrect, achieved median F1-macro scores between 0.24 and 0.29. These values were similar to those of both the General Purpose model (0.28) and the Majority Class baseline (0.27). The Warm Start models also outperformed the incorrectly assigned models under F1-macro, although the diferences were smaller than under F1-micro. Dist\_survey achieved median F1- macro scores of 0.32 and 0.16 for the correct and incorrect assignments, respectively. The corresponding values for Cross\_Dist\_survey were 0.31 and 0.18. All of the cohort models are performing below the F1 score produced by a uniformly random classifier on uniformly random data for a 3-class classification problem. The PCM remained the best-performing model under both metrics. Under F1-micro, the median scores were 0.74 for PCM and 0.52 for the General Purpose model; under F1-macro, the corresponding scores were 0.51 and 0.28.

For the PersonAI dataset (Figure 5, bottom), the General Purpose (0.30), PMV (0.31), and Majority Class (0.29) models achieved similar median F1-macro scores. All warm- and cold-start cohort models, with both correct and incorrect cluster assignments, also fell within a narrow range of 0.29 to 0.31. None exceeded the General Purpose model by more than 0.01 F1. All of the cohort models are performing below the F1 score produced by a uniformly random classifier on uniformly random data for a 3-class classification problem. The PCM was the only model that clearly outperformed the others, with a median F1-macro score of 0.41. PCM models outperformed all other models in terms of F1 macro, followed by the PCM ApproxDistribution model on both datasets.

![](images/2200cf6648df2a12c97016a1045a11fb0f1aa4aa760dc99579a90866f023b053.jpg)  
Figure 4: F1-micro performance comparison across all cohort approaches for the DORN (top) and PersonAI (bottom) datasets. Tilde-prefixed entries denote the ablation condition in which test users are purposely assigned to an incorrect cluster. The red band marks the range between the General Purpose and PCM medians. Cold and warm experiments are shown in blue and orange, respectively.

![](images/bedf314afc20b3f607830a300f8b85c73410461580fc161b72100f901e3f6a2f.jpg)  
Figure 5: F1-macro performance comparison across all cohort approaches for the DORN (top) and PersonAI (bottom) datasets. Tilde-prefixed entries denote the ablation condition in which test users are purposely assigned to an incorrect cluster. The red band marks the range between the General Purpose and PCM medians.

## 4.4. Class distribution

The diferences between F1-micro and F1-macro reflect the class imbalance in the two datasets. As shown in Figure 6, the “no change” class dominates both DORN and PersonAI, whereas the “prefer cooler” and “prefer warmer” classes are much less frequent. The class distribution on a participant basis is shown in Figure 7a for the DORN dataset and in Figure 7b for the PersonAI dataset. This asymmetry is precisely the condition under which F1-micro, being equivalent to overall accuracy in a single-label multi-class setting, rewards models for defaulting to the majority class and penalizes minority-class errors only marginally.

These imbalances lead to a classification problem that is highly skewed, as shown in the confusion matrix of the general-purpose / baseline model on the DORN dataset (Figure 8a). Aggregated across all users, the baseline model achieves an overall accuracy (equivalent to F1-micro) of $2 1 6 7 / 4 6 2 0 \approx 4 6 . 9 \%$ , but this figure masks a strongly classdependent behavior: recall for “no change” is $1 6 5 4 / 2 6 2 0 \approx 6 3 . 1 \%$ , recall for “prefer warmer” is $5 0 9 / 1 7 3 6 \approx 2 9 . 3 \%$ and recall for “prefer cooler” collapses to just $4 / 2 6 4 \approx 1 . 5 \%$ . The model is heavily biased toward predicting “no change” and “prefer warmer” and almost never predicts “prefer cooler” correctly, even though that class accounts for a non-trivial share of the ground truth, illustrating the class-dependent failure mode hidden by an aggregate F1- micro score. This class-wise disparity is not well captured by F1-micro, but is reflected by F1-macro, which averages performance equally across classes.

However, this problem is even more pronounced on the PCM level. Many users have highly imbalanced thermal preference distributions, while making their responses easy to predict and inflating the overall performance scores. Figure 8b shows an example PCM for such a user in the DORN dataset (user dorn14): of the 231 labeled instances for this user, 225 are “no change,” with only 2 “prefer warmer” and 4 “prefer cooler” instances. The PCM predicts almost exclusively “no change,” achieving an accuracy of $2 2 4 / 2 3 1 \approx 9 7 . 0 \%$ (F1-micro) while failing to correctly predict any instance of either minority class (0% recall for both “prefer warmer” and “prefer cooler”). This user alone would contribute a near-perfect F1-micro score, despite the model failing completely on the minority classes.

![](images/05fd052441f986adb6f7e6fcbd9d6fc05a8e51d941bb599be226b47fde1f3861.jpg)  
Figure 6: Class distribution of thermal preference labels for the DORN and PersonAI datasets. Both datasets are dominated by the “no change” class.

![](images/d06272114d9b4389c6cf8b007a9f35588e28f916e341e2468e183939f2de0737.jpg)  
(a) DORN

![](images/fc4161e7bb999ba5e81f41316c13eac802d679f5896ad901030c80db577690ae.jpg)  
(b) PersonAI  
Figure 7: Class distribution of thermal preference labels per participant.

![](images/da37fbc31bdc540a11fbbbf8721d3db94a202ec9675dbd3cfc75175066bb06ab.jpg)  
(a) Confusion matrix of the general-purpose (cohort baseline) model on the full DORN dataset.

![](images/21da09c1feff019b61561332b0c0a7465ee1b25bffcc5bc27c646c7d73f79e98.jpg)  
(b) Confusion matrix of the PCM for user dorn14 in the DORN dataset, an example of an extremely imbalanced individual user.

## 4.5. Participant-level performance changes

To examine the drivers behind the observed performance increases of specific cohort modes, we replicated the peruser percentage change in performance from the General Purpose model to the cohort models using the methodology of the original study and applying F1 micro as metric. Figure 9a shows the per-user percentage change in F1- micro,Δ = 100 × (coh − base)∕base for the Agreeableness, SWLS, Dist, and Dist-Cross approaches on both datasets. On the DORN dataset, a small number of users showed percentage improvements exceeding 500%, with the largest values reaching approximately 3500–3600% for the Dist and Dist-Cross approaches. The PersonAI dataset showed the same pattern, although the largest percentage improvements were substantially smaller (up to approximately 190%). These extreme percentage improvements arose because some users had very low baseline F1-micro scores. As illustrated in Figure 8b, these users typically had highly imbalanced thermal preference distributions. Consequently, even small absolute improvements produced very large percentage changes. Overall, the reported improvement in cohort model performance was driven by a small number of users with very large relative gains.

Figure 9b shows the same comparison after restricting the percentage change to values below 200%, making the distribution for most users easier to inspect. For the DORN dataset, the median improvement over the General Purpose model was approximately 8% for both the Dist-Cross and Dist approaches. The corresponding interquartile ranges were approximately 0–50% and 0–30%, respectively. The SWLS and Agreeableness approaches were centred close to 0%, and several users experienced decreases of 75–90% relative to the General Purpose model. For the PersonAI dataset, all four approaches were centred close to 0%, with most users showing changes within approximately ±20% and only a few reaching 150–190%. Overall, these results indicate that the reported micro averaged F1 improvement of the cohort framework is driven by a small number of users with very large relative gains, whereas the typical user experiences little or no improvement over the General Purpose model.

![](images/b3b3e61fe7a43691263dcc00888ccf756c1c664af5198f6d1e14df2b3cee28ba.jpg)

(a) Full range.  
![](images/8db3862d2df6d205906ce0d32a20f460c0698dcc0605881cc9b9f62362d23d7d.jpg)  
(b) Restricted to gaps under 200%.  
Figure 9: Per-user percentage change in F1-micro from the general-purpose model to each cohort/PCM approach, for DORN and PersonAI.

## 4.6. Dataset Size

The learning curves of the PCM models for both the DORN and PersonAI datasets show no evidence of convergence or a consistent performance trend. Figures 10a and 10b show the learning curves for the participants with the largest number of observations in each dataset.

The average number of available datapoints per person in the DORN dataset is 567, for the PersonAI dataset this number is lower at 110. The average size of the minority class per person in DORN and PersonAI dataset is 34.3 and respectively 6.8. There are 3 participants in the DORN dataset that have no datapoints in at least one class, the same is true for 6 participants in the PersonAI dataset. There are 5 participants in DORN dataset and 38 participants in PersonAI dataset for which there are less than 5 datapoints available for the minority class.

![](images/2146d18138051e4caacfab911e232a1ce63c6f70548264bf5a6d617736c7c0b8.jpg)  
(a) User 001 from DORN Dataset

![](images/c70569c23150bdaf8ebb5e89fd78d7f09bdc7ca870a2f3dd6b5b5c4f2ccf3388.jpg)  
(b) User 003 from PersonAI dataset  
Figure 10: Learning curves for the participants with the most available data for PCM models in each dataset

## 5. Discussion

The authors acknowledge that many of the methodological issues identified in this study could, in principle, have been detected through a careful methodological review of the original work. However, these issues only became apparent during the process of reproducing the study and analysing the PersonAI own dataset. This highlights one of the strengths of independent replication studies bringing a broader range of expertise to the subject. Beyond assessing reproducibility, they can reveal methodological limitations and practices that may not be apparent from the published work alone, while also providing insight into the research context in which the original study was conducted.

## 5.1. Too little data

Quintana et al. [1] justify the amount of training data required for personal thermal comfort modelling by referring to several prior studies [31, 32, 34]. Our own results (Section 4.6, Figs. 10a and 10b) show no clear convergence even for the participants with the most available data, which calls into question the widely repeated assumption that around 60 data points per person is suficient for stable thermal preference prediction. The other two studies commonly cited alongside it also rely on small samples. Li et al. [32] base their conclusions on data from only three participants (271 observations in total), yet conclude that “with a relatively small dataset (approximately 50 samples), the comfort model can achieve an acceptable classification accuracy.” Liu et al. [34] study 14 participants with an average of 275 data points each, follow a very similar experimental and evaluation style to Kim et al., and attribute some of their own models’ low predictive power to participants not "answering carefully" or being inherently less predictable. Notably, using standard interpretation guidelines for Cohen’s kappa (0.01–0.20 as none to slight, 0.21–0.40 as fair, 0.41–0.60 as moderate, 0.61–0.80 as substantial, and 0.81–1.00 as almost perfect agreement), their reported best-case kappa of 0.25 would itself be classified as only "fair" agreement, which is a further indication that the underlying models are not capturing the target well even in the most favorable case reported.

Since Kim et al. [31] is the primary source of the widely cited estimate of 60 data points per person, we examine the basis for this estimate in more detail. The paper introduces the concept of “first convergence,” obtained by training the same model on increasingly large training set sizes and producing a learning curve for each participant (their Fig. A1). “First convergence” is defined as the point at which the derivative of the learning curve plateaus (within ±0.001) for two successive runs. The notion of “first convergence” is therefore a heuristic criterion that is in no way related to the mathematical concept of ”convergence”, or the ML concept of “model convergence“, and is unsuitable to serve as estimate of how much training data is required. The “60 data points” figure is obtained by averaging the training set size at which this first plateau occurs across all participants. The following statement from Kim et al. [31] is particularly noteworthy: “Note that some models go through more than one convergence. This is because the mode has to relearn new patterns in the data with the addition of new data points.” Given the pronounced class imbalance in the dataset, the observed behaviour is also consistent with a model learning the marginal class distribution rather than the relationship between the predictors and an individual’s thermal preference. Small-sample regimes are known to carry a substantial risk of overfitting and optimistic performance estimates [48]. The sample size required to train and evaluate a classifier depends on many factors, including the strength of the predictive signal, class balance, model complexity, feature dimensionality, and the dependence between observations. To provide an estimate of dataset sizes required we consider:

Class availability In order to apply subervised ML classification labeled training data has to be available. In order to train a classifier to distinguish 3 classes, all 3 classes must occur in the traing data. This is not the case for 15% of the participants of the DORN dataset (3 out of 20), and 9% of the participants in the PersonAI dataset (8 out of 66) (see Section4.6). In order to measure and evaluate the performance of a ML based classification approach, the test set also must contain all classes. Given that 5-fold-CV is used to train the PCM models, this gives us a theoretical limit of at least 5 datapoints per class to be available for each participant. This limit is not met for 25% of participants in DORN dataset, and 52% of participants in the PersonAI dataset. Per class Recall, and in consequence per class F1, may be undefined for test data that is missing this class. In practice, ML software libraries use a class F1 of 0 in this case. In order to properly assess classification performance of a model, there must be suficient test data available for every class to establish a desired resolution of performance metrics and expressiveness of the experiment. For example, given 20 data points for a single class in a test set, allows the calculation of the classes Recall in a resolution of 0.05 increments. For more than the half of DORN participants this resolution is not achievable.

EPV Events per variable (EPV) originate from logistic regression and since has been used in a more general ML context for estimating a lower bounds for dataset sizes. EPV is calculated by dividing the number of datapoints of the minority class by the number of predictors. The corresponding rule of thumb is at least "10 events per variable"[49]. The feature set used for training PCM on the DORN dataset consists of 5 features (Air temperature, Relative humidity, Near-body temperature, Heart rate, Clothing level). Performing this calculation, shows us that at least 50 data points per class are required to achieve an ��� <= 10, the mean per person minority class is 34.3 in the DORN dataset, and 6.8 in the PersonAI dataset. Only 25% of participants in the DORN dataset would exceed this limit. PersonAI uses 6 features in the PCM experiments, providing a lower bound of 60 datapoints per class to satisfy the "10 events per variable rule". None of the participants in the PersonAI dataset satistfy this condition. While some researchers critique the "10 events per variable rule" as an oversimplification that under certain circumstances lower EPV can be mitigated [50], other reseachers have found that for approaches other than logistic regression EPV should be greater than 100 [51], putting the number of datapoints per minority class around 500-600 for the PCM models.

Related work on dataset sizes Previous research into data set size requirements for ML classification of medical and psycological data show, depending on the task at hand, a minimum of few hundreds of datapoints per class for a workable solution, and thousands of points per class to achieve good classification performance[52, 53, 54, 55, 56]. To emphasize, the lowest starting point in any of the above cited analysis is a dataset size of 100 items.

Regarding the utilization of existing research datasets, Riley et al. [57] summarized: "Many AI research studies rely on existing datasets that are readily available rather than recruiting new participants, as done in randomised trials or prospective cohort studies. Using an existing dataset provides a convenience sample, but researchers should not proceed before considering whether the sample size is adequate. When the existing dataset is considerably small, abstaining from using it, and instead seeking additional data is often a more appropriate course of action."

Oversampling There are well researched methods for generating additional minority class datapoints, such as SMOTE [58] and ADASYN [59] These methods however are data-driven methods that require a certain amount of data to start with, and must not be applied to the test data. There is too little available data in both the PersonAI and DORN datasets to fulfil those criterea. We did not correct this issue in our reproduction, and instead used the DORN dataset with the same per-person sample sizes (261) as was done by Quintana et al. [1]’s experiments.

## 5.2. Distribution of predictions

While the class imbalance present in the original datasets is documented in both Quintana et al. [1] and Kim et al. [31], the distribution of predictions actually produced by their models is not. These predictions form the basis of every reported performance score, yet neither paper reports or visualizes them. One established way of examining classifier behaviour beyond a single aggregate score is the confusion matrix. Producing confusion matrices would have revealed that the models are heavily biased toward the majority class, and that for many individuals the resulting PCMs efectively reduce to a two-class, or even one-class, classification problem rather than the three-class problem the framework sets out to solve. We addressed this gap by producing confusion matrices for both datasets and including them in this paper to make the models’ actual predictive behaviour transparent (Figs. 8a and 8b).

## 5.3. Use of improper metrics

The use of micro-averaged F1 on imbalanced data sufers from the same well-documented issues as accuracy: in a single-label multi-class setting, micro-F1 is mathematically equivalent to overall accuracy, and both are known to be misleading measures of classifier performance under class imbalance [60, 61, 62]. The resulting metric shows how well a model fits the label distribution of a test set, not how well it can discriminate between classes. Small improvements in micro-F1 may therefore reflect better majority-class prediction rather than any meaningful gain in detecting minority preferences, exactly the pattern we observe in Section 4.3 and Section 4.4, where the trivial majorityclass baseline is matches or outperforms most cohort approaches under F1-micro (Fig. 4), and where per-class recall for the minority “prefer cooler” class collapses to under 2% even for the general-purpose model (Fig. 8a). Complementary, class-balanced metrics are therefore necessary to fully assess model performance. Macro-averaged F1 addresses this to some degree [60, 61, 62, 63] by weighting all classes equally regardless of their support, and is the metric under which the apparent gains of the cohort and warm-start approaches largely disappear in our reanalysis (Section 4.3, Fig. 5). However, for data as imbalanced as ours, where individual users may contribute few or even no instances of a given class within a fold (Section 4.4, Fig. 8b), the interpretation of macro-F1 also requires caution because precision and recall are undefined for absent classes. This impacts the experiments and evaluation of the PCM models, as well as the evaluation of the cohorting models in a leave one out manner on user basis - generating per class F1 scores of 0 for the unavailable classes. We added macro-averaged F1 scores in our reproduction study to provide a more complete picture of model performance alongside the originally reported micro-F1 scores. However, as discussed above, macro-F1 also has important limitations and should not be considered a standalone solution for evaluating classifiers on imbalanced data [63, 62]. Rather, it should be viewed as one of several complementary metrics for assessing per-user thermal preference models. The macro ROC-AUC metric utilized by Kim et al. [31], being macro averaged, is another suitable metric for such imbalanced classification problems.

## 5.4. Missing trivial baselines

Baselines are an essential safeguard against illusory progress. A new method is only convincing if it clearly outperforms simple baselines, such as majority-class prediction and a random noise baseline for classification tasks as the one at hand. Without these comparisons, apparent improvements may reflect properties of the data or the evaluation metric rather than a genuine improvement. Given the task of multiclass classification, it is important to understand the used metrics, their limitations, and the performance envelope one aims to operate in. Important considerations that form this performance envelope in our experiment are the F1 score of a uniformly random predictor applied to uniformly random data. While such a random baseline is provided in the work of Kim et al. [31], it is missing in Quintana et al. [1]. Further, given such imbalanced data, a majority class predictor and a random predictor adhereing to the target distribution may be considered. Without such baselines, it is dificult to determine whether the reported improvements reflect genuine personalization or simply the strong class imbalance present in the data. Further, such baselines can help finding errors and problems in the experiments in an early stage. Our reproduction addressed this by including a majority-class baseline, a uniformly random predictor on uniformly random target, and a random predictor adhering to the target distribution, to provide context for the reported performance gains. In our results we show that the F1 micro performance of the majority-class baseline outperforms all cohorting model approaches on the DORN dataset and performs about equally well to them in the PersonAI dataset (see Section 4.3, Figure 4). Which is a strong indicator that the utilized metric is not a good indicator for measuring the capabilities of a classifier to distinguish the three classes. We show that the F1 macro performance of all cohorting models in both datasets is below the performance of the uniformly random predictor with 0.33 (see Section 4.3, Figure 3a and Figure 3b). This is a result of the trained models overfitting to the majority class and either never predicting at least one class, or at least one class often being missing from the test set. We show that the F1 macro performance of all cohorting models in both datasets is below the performance of the target distribution adhereing random classifier (see Section 4.3, Figure 3a and Figure 3b). This demonstrates that the features used in both the models themselves, as well as the features used for cohort creation and model selection, do not infuse more information that the models can leverage to produce better predictions - since a completely feature oblivous random actor outperforms them.

## 5.5. Methodological limitations of the experimental setup

Quintana et al. [1] perform 100 randomized repetitions of 80/20 train-test splits, while perform 150 randomized repetitions of 2-fold cross-validation. This so-called “repeated cross-validation” or “Monte Carlo cross-validation” has been shown to be statistically weaker than regular �-fold CV while being very computationally expensive [64, 65, 66, 67, 68, 69], with some authors calling it “a waste of resource” [66]. There is a general consensus that leave-one-out cross-validation is to be preferred where computationally feasible [69, 64, 65], and it has even been referred to as the “gold standard” [64] among cross-validation methods. Such resampling methodologies (the various flavors of cross-validation, bootstrapping, etc.) provide more stable and robust results and insights, and have therefore become commonplace in ML. Furthermore, in the case of Quintana et al. [1] and Kim et al. [31], applying Monte Carlo cross-validation to datasets that are both small and highly imbalanced produces results that are opaque and hard to interpret, analyze, or troubleshoot. The wide range of performance scores across repeated random splits, in some cases spanning roughly 0.25 to 0.75, is consistent with unstable performance estimates. This evaluation strategy also obscures the outlier efect we identify in Section 4.5 (Figs. 9a and 9b, and Figure 5 in Quintana et al.[1]): a small number of highly imbalanced, trivially-predictable individuals who drive a disproportionate share of the reported gains remain hidden, since in each random split they are always lumped together with a handful of other participants’ data in the test set, and their extreme contribution is smoothed out across 100–150 repetitions rather than exposed on a per-person basis. Leave-one-out cross-validation avoids this problem by producing participant-level results, making it easier to identify the observations driving overall performance and we encourage using it in future personal and cohort comfort modeling.

## 5.6. Hypothesis and conclusion

Quintana et al. [1] formulate their central hypothesis as follows: “our research hypothesis is that the thermal comfort preferences of building occupants can be estimated by analyzing their psychological and behavioral traits and comparing them to other occupants from which subjective feedback was previously obtained.” The original publication does not report a correlation analysis between these traits and thermal preference before proceeding to predictive modelling. We addressed this gap in our analysis (Section 4.1) and found only weak linear associations between the measured psychological, demographic, and biological features and thermal preference. While the absence of linear associations does not preclude predictive relationships, it provides little initial evidence towards the proposed hypothesis. The original authors designed “cold start” experiments including an ablation analysis, in which participants are assigned to both the correct cohort model and to an incorrect/worst cohort, in order to “corroborate the usefulness of each cohort approach.” Their original figures (Fig. 4 in [1]) show that of the six feature sets used for cohorting (sex, surveys, sensitivity, life satisfaction, Big Five personalities, agreeableness), only a single cohorting approach (life satisfaction) shows a marginally higher performance for the correct cohort assignment (0.55 vs. 0.53). Every other approach performs better when participants are assigned to the wrong cohort; the opposite of what the hypothesis predicts. Our own experiments on the DORN and PersonAI datasets show a similar pattern (Section 4.3, Figs. 4 and 5). In the original publication, the authors discuss the result that supports their hypothesis as follows: “The Life Satisfaction is the only survey that by itself has an above General-purpose performance. [...] Since both of these cohort approaches are related to satisfaction and optimism, we hypothesize occupants within the resulting cohorts are less prone to preference changes due to small variations, meaning they have a wider acceptability range.” In contrast, the more common cases in which incorrect cohort assignment outperformed correct assignment receive little discussion. The interpretation therefore focuses on evidence consistent with the proposed hypothesis, while the contradictory findings are largely left unconsidered.

Given the small sample size, the small efect sizes, the large number of cohorting strategies, and the absence of formal statistical testing including correction for multiple compoarisons, the study does not establish that this isolated positive finding reflects a genuine efect rather than random variation. Under these circumstances, a small number of positive results would be expected by chance alone. The finding should therefore be regarded as exploratory and interpreted cautiously until confirmed by independent replication. A further result reported by the authors also tempers the interpretation of the Life Satisfaction cohort. Table A.2 shows that this approach improved prediction performance for only four participants, while prediction performance decreased for six. Thus, even the only cold-start cohorting strategy reported to outperform the general-purpose model on average did not improve prediction performance for most occupants to whom it was applied. The percentage improvements reported over the General Purpose model for the two cohort approaches with positive average performance are also not examined at the participant level. In both cases, the average improvement is driven by two to four participants with very large gains, while the remaining participants show little change or a decrease in performance. These large gains occur for participants with highly imbalanced thermal preference distributions (e.g., dorn10, who responded “prefer cooler” in 97% of observations, and dorn14, who responded “no change” in 97% of observations; Figs. A.1 and A.4 in [1]), with a similar pattern observed for the SMC dataset (smc30 and smc46). Our analysis reproduced the same pattern in both the DORN and PersonAI datasets (Section 4.5).

## 5.7. Redefinition of metrics

Kim et al. [31] state that they use ROC-AUC as their evaluation metric, yet throughout the paper the reported performance scores are referred to interchangeably as “AUC” and “accuracy” due to the lack of familiarity of the term “AUC” in the building science community at the time of publication. While we can see the reasons why authors try to avoid word-repetition in prose text, renaming a well known and well established metric reduces clarity, while introducing the risk to accidentally overlap with another established name of convention ("accuracy"). The resulting terminological imprecisions undermine the reproducibility and comparability of reported results. However, Kim et al. [31] provides a source and description of the metric and importantly also of the averaging process applied in the multi-class scenario at hand.

## 5.8. Information leakage

Inspection of the original codebase revealed an information leakage issue arising from the feature selection logic. Rather than explicitly selecting the intended input variables, the implementation excluded a set of named columns, allowing the dataframe index (Unnamed: 0) to be unintentionally included as a predictor. Because both the DORN and SMC datasets span multiple seasons, this index efectively encodes temporal information that should not be available to the model, leading to optimistic performance estimates. Although we removed this unintended feature in our reproduction, we cannot determine with certainty whether it afected the originally reported results. The publicly available code and datasets no longer correspond to a uniquely identifiable version of the original experiments, making it impossible to reconstruct the exact computational environment. This illustrates a broader challenge as applied disciplines increasingly adopt computational methods: making code and data available is necessary but not suficient for reproducibility. Versioned releases, archived datasets, and commit hashes are also needed to enable independent verification and exact reproduction of reported results [53]. [70].

## 5.9. Shared temporal and spatial context between train and test

Although Quintana et al. [1] perform train-test splits on a participant basis, there is no temporal separation, and only very limited spatial separation, between participants. The SMC dataset [31] is collected from participants located in the same ofice building, and the DORN dataset [6] from participants located in the same city. Given this spatial proximity, participants assigned to the training and test sets nonetheless share the same weather conditions and seasonal patterns over the same time span, so a model can partly succeed by picking up on shared environmental context rather than on individual-specific comfort preferences. A temporal train-test split (e.g., training on an earlier period and testing on a later one) including a deliberate gap in time between the two would better isolate genuine person-level generalization from this shared confounding. We did not correct this issue in our own experiments.

## 5.10. Potential overfitting and further information leakage

Quintana et al. [1] perform hyperparameter tuning on the same data used to evaluate performance, which leaks information from the test set into model selection. Tartarini et al. [6] perform hyperparameter tuning once on the entire dataset and then apply the resulting best parameters across all downstream experiments, which leaks information in a similar way. We addressed this issue in our reproduction by removing hyperparameter tuning entirely and relying on the default parameters of scikit-learn’s<sup>3</sup> Random Forest implementation.

## 6. Conclusion

We assessed the original methodology of cohort comfort models and conducted two replications using the original dataset and independently collected data, both under a revised experimental setup. The reported predictive advantages of CCMs were not supported when performance was evaluated using class-imbalance-aware metrics and simple baselines. More broadly, the limited availability of data remains a fundamental challenge for ML approaches in thermal comfort modeling. This study highlights the importance of independent replication and rigorous evaluation in assessing the credibility of scientific claims. Future work should develop guidelines for experimental design and ML analysis in thermal comfort modeling, assess existing methods for personal comfort modeling, and replicate findings from previous personal comfort studies.

## Data Availability

All data and code used in the experiment is made publicly availabe on Github<sup>4</sup> and Zenodo<sup>5</sup>

## Acknowledgements

The research leading to these results was conducted within the projects PersonAI (project number FO999901784), funded under the programme “Technologien und Innovationen für die Klimaneutrale Stadt 2022”, and Wellfit (project number FO999920134), funded under the programme “Digitale Lösungen für Mensch und Gesellschaft”. Both projects were funded by the Austrian Research Promotion Agency (FFG). The authors acknowledge TU Wien Bibliothek for financial support through its Open Access Funding Programme.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

The manuscript structure, scientific arguments, and interpretation of results were developed entirely by the authors. ChatGPT (OpenAI) and Claude (Anthropic) was used for grammatical proofreading and suggestions to improve sentence structure and readability.

## A. My Appendix

## References

[1] Matias Quintana, Stefano Schiavon, Federico Tartarini, Joyce Kim, and Clayton Miller. Cohort comfort models — Using occupant’s similarity to predict personal thermal preference with less data. Building and Environment, 227:109685, January 2023.

[2] Gerald Schweiger, Lisa V Eckerstorfer, Irene Hafner, Andreas Fleischhacker, Johannes Radl, Barbara Glock, Matthias Wastian, Matthias Rößler, Georg Lettner, Niki Popper, and others. Active consumer participation in smart energy systems. Energy and Buildings, 227:110359, 2020. Publisher: Elsevier.

[3] Neil E Klepeis, William C Nelson, Wayne R Ott, John P Robinson, Andy M Tsang, Paul Switzer, Joseph V Behar, Stephen C Hern, and William H Engelmann. The National Human Activity Pattern Survey (NHAPS): a resource for assessing exposure to environmental pollutants. Journal ofexposure science & environmental epidemiology, 11(3):231–252, 2001.

[4] Alex Donkers, Dujuan Yang, Bauke de Vries, and Nico Baken. Personal indoor comfort models through knowledge discovery in cross-domain semantic digital twins. Building and Environment, 269:112433, 2025.

[5] Joyce Kim, Stefano Schiavon, and Gail Brager. Personal comfort models – A new paradigm in thermal comfort for occupant-centric environmental control. Building and Environment, 132:114–124, March 2018.

[6] Federico Tartarini, Stefano Schiavon, Matias Quintana, and Clayton Miller. Personal comfort models based on a 6-month experiment using environmental parameters and data from wearables. Indoor Air, 32(11):e13160, 2022. e13160 INA-22-07-334.R1.

[7] Abhyuday Desai, Mohamed Abdelhamid, and Nakul R Padalkar. What is reproducibility in artificial intelligence and machine learning research? AI Magazine, 46(2):e70004, 2025.

[8] Odd Erik Gundersen. The fundamental principles of reproducibility. Philosophical Transactions ofthe Royal Society A, 379(2197):20200210, 2021.

[9] Thijs Bol, Mathijs De Vaan, and Arnout Van De Rijt. The Matthew efect in science funding. Proceedings of the National Academy of Sciences, 115(19):4887–4890, 2018. Publisher: National Academy of Sciences.

[10] Hugo Horta and João M Santos. The impact of publishing during PhD studies on career research publication, visibility, and collaborations. Research in Higher Education, 57(1):28–50, 2016. Publisher: Springer.

[11] Giorgio Tripodi, Xiang Zheng, Yifan Qian, Dakota Murray, Benjamin F Jones, Chaoqun Ni, and Dashun Wang. Tenure and research trajectories. Proceedings ofthe National Academy ofSciences, 122(30):e2500322122, 2025. Publisher: National Academy of Sciences.

[12] Brian A Nosek, Jefrey R Spies, and Matt Motyl. Scientific utopia: II. Restructuring incentives and practices to promote truth over publishability. Perspectives on Psychological Science, 7(6):615–631, 2012. Publisher: Sage Publications Sage CA: Los Angeles, CA.

[13] John PA Ioannidis. Why replication has more scientific value than original discovery. Behavioral and Brain Sciences, 41, 2018. Publisher: Cambridge University Press.

[14] Colin F Camerer. Anna Dreber. Eskil Forsell. Teck-Hua Ho. Jürgen Huber. Magnus Johannesson, Michael Kirchler. Johan Almenberg, Adam Altmejd, Taizan Chan, and others. Evaluating replicability of laboratory experiments in economics. Science, 351(6280):1433–1436, 2016. Publisher: American Association for the Advancement of Science.

[15] John PA Ioannidis. Why most published research findings are false. PLoS medicine, 2(8):e124, 2005. Publisher: Public Library of Science.

[16] Timothy M Errington, Maya Mathur, Courtney K Soderberg, Alexandria Denis, Nicole Perfito, Elizabeth Iorns, and Brian A Nosek. Investigating the replicability of preclinical cancer biology. elife, 10:e71601, 2021. Publisher: eLife Sciences Publications, Ltd.

[17] Daniele Fanelli, Rodrigo Costas, and John PA Ioannidis. Meta-assessment of bias in science. Proceedings ofthe National Academy ofSciences, 114(14):3714–3719, 2017. Publisher: National Academy of Sciences.

[18] Marcus R Munafò, Brian A Nosek, Dorothy VM Bishop, Katherine S Button, Christopher D Chambers, Nathalie Percie du Sert, Uri Simonsohn, Eric-Jan Wagenmakers, Jennifer J Ware, and John PA Ioannidis. A manifesto for reproducible science. Nature human behaviour, 1(1):0021, 2017.

[19] Harald Semmelrock, Tony Ross-Hellauer, Simone Kopeinik, Dieter Theiler, Armin Haberl, Stefan Thalmann, and Dominik Kowald. Reproducibility in machine-learning-based research: Overview, barriers, and drivers. AI Magazine, 46(2):e70002, 2025.

[20] Daoyang Li, Zhenzhen Qi, Yiming Zhou, and Mohamed Elchalakani. Machine learning applications in building energy systems: review and prospects. Buildings, 15(4):648, 2025.

[21] Shiyu Yang, Man Pun Wan, Wanyu Chen, Bing Feng Ng, and Swapnil Dubey. Model predictive control with adaptive machine-learning-based model for building energy eficiency and comfort optimization. Applied Energy, 271:115147, 2020.

[22] Adil Mukhtar, Thomas Hirsch, and Gerald Schweiger. One-class classification and cluster ensembles for anomaly detection and diagnosis in multivariate time series data. In 35th international conference on principles ofdiagnosis and resilient systems (DX 2024), pages 14–1. Schloss Dagstuhl–Leibniz-Zentrum für Informatik, 2024.

[23] Thomas Schranz, Johannes Exenberger, Christian Møldrup Legaard, Jan Drgona, and Gerald Schweiger. Energy prediction under changed demand conditions: Robust machine learning models and input feature combinations. In Building simulation 2021, volume 17, pages 3268– 3275. IBPSA, 2021.

[24] Prageeth Jayathissa, Matias Quintana, Mahmoud Abdelrahman, and Clayton Miller. Humans-as-a-sensor for buildings—intensive longitudinal indoor comfort models. Buildings, 10(10):174, 2020.

[25] Tianzhen Hong and Han Li. Good practices for documenting ai-based studies on energy and buildings. Energy and Buildings, page 117043, 2026.

[26] Adil Mukhtar, Michael Hadwiger, Franz Wotawa, and Gerald Schweiger. Reproducibility of machine learning-based fault detection and diagnosis for hvac systems in buildings: An empirical study. Energy and AI, page 100658, 2025.

[27] P. O. Fanger. Thermal comfort. Analysis and applications in environmental engineering. Copenhagen: Danish Technical Press., 1970.

[28] Richard de Dear and G. S. Brager. Developing an adaptive model of thermal comfort and preference. ASHRAE Transactions, 104:145, 1998.

[29] Veronika Földváry Ličina, Toby Cheung, Hui Zhang, Richard De Dear, Thomas Parkinson, Edward Arens, Chungyoon Chun, Stefano Schiavon, Maohui Luo, Gail Brager, et al. Development of the ashrae global thermal comfort database ii. Building and Environment, 142:502–512, 2018.

[30] Toby Cheung, Stefano Schiavon, Thomas Parkinson, Peixian Li, and Gail Brager. Analysis of the accuracy on PMV – PPD model using the ASHRAE Global Thermal Comfort Database II. Building and Environment, 153:205–217, April 2019.

[31] Joyce Kim, Yuxun Zhou, Stefano Schiavon, Paul Raftery, and Gail Brager. Personal comfort models: Predicting individuals’ thermal preference using occupant heating and cooling behavior and machine learning. Building and Environment, 129:96–106, February 2018.

[32] Da Li, Carol C. Menassa, and Vineet R. Kamat. Personalized human comfort in indoor building environments under diverse conditioning modes. Building and Environment, 126:304–317, December 2017.

[33] David Daum, Frédéric Haldi, and Nicolas Morel. A personalized measure of thermal comfort for building controls. Building and Environment, 46(1):3–11, January 2011.

[34] Shichao Liu, Stefano Schiavon, Hari Prasanna Das, Ming Jin, and Costas J. Spanos. Personal thermal comfort models with wearable sensors. Building and Environment, 162:106281, September 2019.

[35] F. Tartarini, M. Frei, S. Schiavon, Y. X. Chua, and C. Miller. Cozie apple: An ios mobile and smartwatch application for environmental quality satisfaction and physiological data collection. Journal ofPhysics: Conference Series, 2600(14):142003, 2023.

[36] Oura Health. Heart rate graph. Oura Member Care, October 2026.

[37] Apple Inc. Monitor your heart rate with apple watch. Apple Support, September 2026.

[38] A. Henriksen, F. Svartdal, S. Grimsgaard, G. Hartvigsen, and L. Hopstock. Polar vantage and oura physical activity and sleep trackers: Validation and comparison study. JMIR Formative Research, 6(5):e27248, 2022.

[39] TFA Dostmann. Co2 monitor with data logger airco2ntrol 5000 31.5008. TFA Dostmann, n.d. Retrieved October 13, 2026.

[40] M. Walentynowicz, Y. Erbaş, J. Joran, V. V. Leonie, and O. Luminet. Interoception in daily life: Development and validation of momentary body awareness scale. In SAA, page 127, Amsterdam, Netherlands, 2023.

[41] W. E. Mehling, M. Acree, A. Stewart, J. Silas, and A. Jones. The multidimensional assessment of interoceptive awareness, version 2 (maia-2). PLOS ONE, 13(12):e0208034, 2018.

[42] L. R. Derogatis. Brief Symptom Inventory. Clinical Psychometric Research, Baltimore, MD, 1975.

[43] G. H. Franke. Brief Symptom Inventory von L. R. Derogatis (Kurzform der SCL-90-R) – Deutsche Version. Manual. Beltz Test, Göttingen, 2000.

[44] C. J. Soto and O. P. John. Short and extra-short forms of the big five inventory–2: The bfi-2-s and bfi-2-xs. Journal ofResearch in Personality, 68:69–81, 2017.

[45] B. Rammstedt, D. Danner, C. J. Soto, and O. P. John. Validation of the short and extra-short forms of the big five inventory-2 (bfi-2) and their german adaptations. European Journal ofPsychological Assessment, 36(1):149–161, 2020.

[46] E. Diener, R. A. Emmons, R. J. Larsen, and S. Grifin. The satisfaction with life scale. Journal ofPersonality Assessment, 49(1):71–75, 1985.

[47] S. Janke and A. Glöckner-Rist. Deutsche version der satisfaction with life scale (swls). Zusammenstellung sozialwissenschaftlicher Items und Skalen (ZIS), 2012.

[48] Andrius Vabalas, Emma Gowen, Ellen Poliakof, and Alexander J. Casson. Machine learning algorithm validation with a limited sample size. PLOS ONE, 14(11):e0224365, November 2019.

[49] Peter Peduzzi, John Concato, Elizabeth Kemper, Theodore R. Holford, and Alvan R. Feinstein. A simulation study of the number of events per variable in logistic regression analysis. Journal ofClinical Epidemiology, 49(12):1373–1379, December 1996.

[50] Maarten van Smeden, Joris A. H. de Groot, Karel G. M. Moons, Gary S. Collins, Douglas G. Altman, Marinus J. C. Eijkemans, and Johannes B. Reitsma. No rationale for 1 variable per 10 events criterion for binary logistic regression analysis. BMC Medical Research Methodology, 16(1):163, November 2016.

[51] Tjeerd van der Ploeg, Peter C. Austin, and Ewout W. Steyerberg. Modern modelling techniques are data hungry: A simulation study for predicting dichotomous endpoints. BMC Medical Research Methodology, 14(1):137, December 2014.

[52] Alhanoof Althnian, Duaa AlSaeed, Heyam Al-Baity, Amani Samha, Alanoud Bin Dris, Najla Alzakari, Afnan Abou Elwafa, and Heba Kurdi. Impact of Dataset Size on Classification Performance: An Empirical Evaluation in the Medical Domain. 11(2):796.

[53] Daniyal Rajput, Wei-Jen Wang, and Chun-Chuan Chen. Evaluation of a decided sample size in machine learning applications. BMC Bioinformatics, 24(1):48, February 2023.

[54] Margarita Sordo and Qing Zeng. On Sample Size and Classification Accuracy: A Performance Comparison. In José Luís Oliveira, Víctor Maojo, Fernando Martín-Sánchez, and António Sousa Pereira, editors, Biological and Medical Data Analysis, pages 193–201, Berlin, Heidelberg, 2005. Springer.

[55] Anita Rácz, Dávid Bajusz, and Károly Héberger. Efect of Dataset Size and Train/Test Split Ratios in QSAR/QSPR Multiclass Classification. Molecules, 26(4):1111, 2021.

[56] Claudia Beleites, Ute Neugebauer, Thomas Bocklitz, Christoph Kraft, and Jürgen Popp. Sample size planning for classification models. Analytica Chimica Acta, 760:25–33, January 2013.

[57] Richard D. Riley, Joie Ensor, Kym I. E. Snell, Lucinda Archer, Rebecca Whittle, Paula Dhiman, Joseph Alderman, Xiaoxuan Liu, Laura Kirton, Jay Manson-Whitton, Maarten van Smeden, Karel G. Moons, Krishnarajah Nirantharakumar, Jean-Baptiste Cazier, Alastair K. Denniston, Ben Van Calster, and Gary S. Collins. Importance of sample size on the quality and utility of AI-based prediction models for healthcare. The Lancet Digital Health, 7(6), June 2025.

[58] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer. SMOTE: Synthetic Minority Over-sampling Technique. Journal ofArtificia Intelligence Research, 16:321–357, June 2002.

[59] Haibo He, Yang Bai, Edwardo A. Garcia, and Shutao Li. ADASYN: Adaptive synthetic sampling approach for imbalanced learning. In 2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence), pages 1322–1328, June 2008.

[60] Margherita Grandini, Enrico Bagli, and Giorgio Visani. Metrics for Multi-Class Classification: An Overview, August 2020.

[61] Juri Opitz. A Closer Look at Classification Evaluation Metrics and a Critical Reflection of Common Evaluation Practice. Transactions ofthe Associationfor Computational Linguistics, 12:820–836, 2024.

[62] Nathalie Japkowicz and Mohak Shah. Evaluating Learning Algorithms: A Classification Perspective. Cambridge University Press, 2011.

[63] Haibo He and Edwardo A. Garcia. Learning from Imbalanced Data. IEEE Transactions on Knowledge and Data Engineering, 21(9):1263– 1284, 2009.

[64] Luke A. Yates, Zach Aandahl, Shane A. Richards, and Barry W. Brook. Cross validation for model selection: A review with examples from ecology. Ecological Monographs, 93(1):e1557, 2023. \_eprint: https://esajournals.onlinelibrary.wiley.com/doi/pdf/10.1002/ecm.1557.

[65] Gavin C. Cawley and Nicola L.C. Talbot. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. J. Mach. Learn. Res., 11:2079–2107, August 2010.

[66] Gitte Vanwinckelen, Hendrik Blockeel, Bernard De Baets, Bernard Manderick, Michaël Rademaker, and Willem Waegeman. On estimating model accuracy with repeated cross-validation, January 2012. ISBN: 978-94-6197-044-2.

[67] Umberto Michelucci. Fundamental Mathematical Conceptsfor Machine Learning in Science. Springer International Publishing, Cham, 2024.

[68] Robbie T. Nakatsu. Validation of machine learning ridge regression models using Monte Carlo, bootstrap, and variations in cross-validation. Journal ofIntelligent Systems, 32(1), January 2023.

[69] Prabir Burman. A Comparative Study of Ordinary Cross-Validation, v-Fold Cross-Validation and the Repeated Learning-Testing Methods. Biometrika, 76(3):503–514, 1989.

[70] Mark D. Wilkinson, Michel Dumontier, IJsbrand Jan Aalbersberg, Gabrielle Appleton, Myles Axton, Arie Baak, Niklas Blomberg, Jan-Willem Boiten, Luiz Bonino da Silva Santos, Philip E. Bourne, Jildau Bouwman, Anthony J. Brookes, Tim Clark, Mercè Crosas, Ingrid Dillo, Olivier Dumon, Scott Edmunds, Chris T. Evelo, Richard Finkers, Alejandra Gonzalez-Beltran, Alasdair J.G. Gray, Paul Groth, Carole Goble, Jefrey S. Grethe, Jaap Heringa, Peter A.C ’t Hoen, Rob Hooft, Tobias Kuhn, Ruben Kok, Joost Kok, Scott J. Lusher, Maryann E. Martone, Albert Mons, Abel L. Packer, Bengt Persson, Philippe Rocca-Serra, Marco Roos, Rene van Schaik, Susanna-Assunta Sansone, Erik Schultes, Thierry Sengstag, Ted Slater, George Strawn, Morris A. Swertz, Mark Thompson, Johan van der Lei, Erik van Mulligen, Jan Velterop, Andra Waagmeester, Peter Wittenburg, Katherine Wolstencroft, Jun Zhao, and Barend Mons. The FAIR Guiding Principles for scientific data management and stewardship. Scientific Data, 3(1):160018, 2016.