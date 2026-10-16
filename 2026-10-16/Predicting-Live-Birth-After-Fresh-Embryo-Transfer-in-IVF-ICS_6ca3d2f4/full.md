# Predicting Live Birth After Fresh Embryo Transfer in IVF/ICSI: Differential ICM/Trophectoderm Morphology Encoding and a Hybrid Attention–Logistic Regression Ensemble

Abdul Raheem Shahzad<sup>†</sup>, Jinxin Chen<sup>†</sup>, Ming Zhang<sup>∗</sup>, Juan Deng<sup>∗</sup>

Abstract— Objective: Predicting live birth before fresh embryo transfer could support shared decision-making in IVF/ICSI, but routine clinical data are small, heterogeneous, and contain pathway-dependent variables prone to outcome leakage. We aimed to construct a curated singlecentre IVF/ICSI dataset and develop a prediction model suited to small tabular clinical data. Materials and Methods: We assembled 616 fresh embryo-transfer cycles (251 live birth, 365 non-live birth) with clinical, laboratory, and embryological predictors, and embryo morphology encoded with a novel differential weighting of the inner cell mass and trophectoderm. After KNN imputation, robust scaling, and selection of 50 features, we developed a Hybrid Attention Network (HAN), a learnable feature-attention gate followed by a compact multilayer perceptron, and a HAN + logisticregression (LR) ensemble with a validation-tuned weight. Models were benchmarked against classical machine learning, TabPFN, and text-serialized language models on a held-out test set; AUC-ROC was the primary metric. Results: The HAN + LR ensemble achieved the highest test AUC-ROC (0.710; accuracy 0.710, F1 0.630), ahead of the HAN backbone alone (0.700), logistic regression (0.667), the TabPFN + LR hybrid (0.666), and TabPFN (0.650). Textserialized language models performed near chance (AUC ≤ 0.61). The differentially weighted trophectoderm feature ranked highest by consensus importance across models. Discussion: A compact attention network combined with a linear model outperformed both components and strong foundation-model baselines on a small clinical cohort. Conclusion: The curated dataset and HAN + LR ensemble offer a transparent, reproducible basis for IVF/ICSI live-birth prediction; external validation is the next step.

Index Terms— Blastocyst grading, clinical prediction model, fresh embryo transfer, in vitro fertilization, live birth prediction, machine learning

![](images/a2e0d1919550882b1d0caae5475252d720e447579c865bc9f08733c97e3974ac.jpg)  
Graphical abstract. Across 616 fresh IVF/ICSI embryo-transfer cycles, a compact hybrid attention network combined with logistic regression (HAN + LR) achieved the highest test AUC-ROC (0.710), outperforming its components, TabPFN, and text-serialized language models. Weighting the trophectoderm above the inner cell mass in blastocyst grading yielded the top-ranked predictor.

## I. BACKGROUND AND SIGNIFICANCE

NFERTILITY affects roughly one in six adults worldwide, I and assisted reproductive technology (ART), in particular in vitro fertilization (IVF) and intracytoplasmic sperm injection (ICSI), is now mainstream [1], [2]. Despite continued improvement in laboratory and clinical practice, the live-birth rate after a fresh embryo-transfer cycle remains modest, on the order of one in four, and varies substantially between patients [2], [3].

Because each cycle carries a considerable physical, emotional, and financial burden, an individualized estimate of livebirth probability before transfer would be clinically valuable, supporting counselling and shared decisions such as the number of embryos to transfer or whether to pursue a freeze-all strategy [4]–[6].

These decisions are made just before transfer, when the clinician already knows the patient’s baseline characteristics, stimulation and laboratory results, and embryo morphology. A prediction delivered then, from information available up to that point, is directly actionable. This contrasts with tools estimating cumulative live birth over multiple cycles [7], which serve a longer-horizon purpose; our work targets the pertransfer decision point, matched to single-cycle structured

data.

Many prognostic models have been proposed in ART, but systematic reviews repeatedly identify two limitations: most are not externally validated, and many are at risk of bias under formal appraisal [7]–[9]. Reporting frameworks such as TRIPOD+AI and PROBAST+AI address these gaps [10], [11]. Much recent methodological energy has gone into imagebased deep learning [12]–[14], whereas models built on the structured clinical data routinely captured during a cycle remain underexplored, even though such data exist in essentially every clinic.

Tabular clinical data pose their own difficulties: cohorts are small relative to the appetite of modern neural networks, predictors are heterogeneous and partially missing, and several informative variables are pathway-dependent, being themselves consequences of earlier steps in the cycle (for example the number of usable embryos or their grade). Recent advances for tabular learning, including attention-based architectures such as TabNet and FT-Transformer, tabular foundation models such as TabPFN, and serialization approaches such as TabLLM [15]–[18], are rarely evaluated head-to-head for IVF live-birth prediction on a single, transparently curated cohort.

This motivates three contributions: (1) a curated singlecentre IVF/ICSI dataset for live-birth prediction; (2) a novel numerical encoding of embryo morphology that assigns differential weights to the inner cell mass and trophectoderm of the blastocyst, in place of the equal weighting implicit in conventional Gardner grading; and (3) a prediction model purpose-built for small tabular clinical data and benchmarked against contemporary baselines.

## II. RELATED STUDIES

## A. Datasets for IVF Outcome Prediction

Research on IVF outcome prediction draws on two data sources. National registries such as the Human Fertilisation and Embryology Authority (United Kingdom) and SART CORS (United States) offer large samples but capture only a standardized variable set, lacking clinic-level laboratory and embryo detail [2], [19], [20], whereas single-centre datasets are rich but small, heterogeneous, and affected by missing values. A parallel line of work has released image and timelapse embryo datasets [13], [21], [22], which target embryo grading rather than cycle-level prediction from structured data. Registry-scale studies have predicted live birth from such databases [23], [24], and large image corpora have been released for benchmarking [25], [26]. Our dataset sits in the second group: a curated single-centre collection in which morphology grades are numerically encoded and combined with clinical and laboratory predictors.

## B. Traditional Machine-Learning Approaches

Classical machine learning has been applied repeatedly to IVF/ICSI outcome prediction, where logistic regression remains a strong baseline [27], [28], and tree-based methods (decision trees, random forests, gradient boosting) have modelled pregnancy and live-birth outcomes [29]–[31]. Reported discrimination varies with cohort and outcome definition, large-cohort studies often reporting AUCs of 0.70 to 0.80 [24], [28] and single-centre live-birth studies lower and more variable. Broader reviews survey the algorithms used across ART [32]–[34]. Systematic reviews caution that many models use modest samples, are rarely externally validated, and omit calibration [7]–[9], [35].

## C. Deep Learning, Foundation Models, and Language Models

Deep learning has entered along two strands. The dominant, image-based strand trains convolutional models on embryo images for grading and selection [13], [14], [24], [36], including foundation models [26] and pipelines localizing informative time points [37], [38], with reviews charting laboratory adoption [39]–[41] and work on male-factor assessment [42] and IVF-laboratory automation [43], [44]. The second strand, to which this work belongs, applies deep and transformer models to tabular data: TabNet and FT-Transformer brought attention to tabular learning [15], [16], tabular foundation models such as TabPFN give in-context prediction for small datasets [17], [45], and language models have been adapted by serializing records into text [18]. Evidence is mixed: recent benchmarks find well-tuned classical models still match or exceed foundation models on clinical tabular tasks [46], underscoring the value of head-to-head evaluation on a single curated cohort.

Two gaps follow. First, structured-data, cycle-level live-birth prediction is under-studied relative to image-based embryo selection. Second, modern tabular model families are rarely benchmarked against one another on the same IVF cohort under a consistent endpoint and transparent pipeline. This study addresses both.

## D. Objective

We set out to (1) construct and characterize a curated dataset of fresh embryo-transfer IVF/ICSI cycles with a binary live-birth label; (2) introduce a differential weighting of the inner cell mass and trophectoderm for numerically encoding blastocyst morphology; and (3) develop and evaluate a Hybrid Attention Network plus logistic-regression (HAN + LR) ensemble for live-birth prediction, benchmarking it against classical machine learning, a tabular foundation model, and text-serialized language models.

## III. MATERIALS AND METHODS

## A. Data Source and Study Cohort

We retrospectively assembled clinical data from patients who underwent fresh IVF/ICSI embryo-transfer cycles at the Reproductive Medicine Center of Zhongnan Hospital, Wuhan University, between February 2018 and November 2025. Each treatment cycle constitutes one sample. The curated dataset comprises 616 cycles. The study was approved by the Ethics Committee of Zhongnan Hospital of Wuhan University (Ethical Approval No. 2025346K); given the retrospective design, the requirement for informed consent was waived. Table I reports the baseline characteristics of both outcome groups together with the modelling status of every collected variable.

TABLE I  
BASELINE CHARACTERISTICS OF THE STUDY COHORT BY OUTCOME GROUP, WITH THE MODELLING STATUS OF EVERY COLLECTED VARIABLE
<table><tr><td>Variable</td><td>Live birth (n = 251)</td><td>Non-live birth (n = 365)</td><td>p</td><td>SMD</td><td>Selected</td></tr><tr><td>Female factors</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Female age (years)</td><td> $3 1 . 8 \pm 3 . 7$ </td><td> $3 3 . 1 \pm 4 . 0$ </td><td>&lt;0.001</td><td>0.34</td><td>√</td></tr><tr><td>Female smoking history, n (%)</td><td></td><td></td><td>0.722</td><td></td><td></td></tr><tr><td>Have</td><td>4 (1.6%)</td><td> $4 \ : ( 1 . 1 \% )$ </td><td></td><td></td><td></td></tr><tr><td>None</td><td>247 (98.4%)</td><td>361 (98.9%)</td><td></td><td></td><td></td></tr><tr><td>Female height (cm)</td><td> $1 6 0 . 8 \pm 4 . 9$ </td><td> $1 6 1 . 2 \pm 4 . 5$ </td><td>0.269</td><td>0.09</td><td>√</td></tr><tr><td>Female weight (kg)</td><td> $5 9 . 2 \pm 8 . 6$ </td><td> $5 8 . 0 \pm 8 . 0$ </td><td>0.062</td><td>0.15</td><td>– (collinear with BMI)</td></tr><tr><td>Female BMI (kg/m²)</td><td> $2 3 . 0 \pm 3 . 2$ </td><td> $2 2 . 3 \pm 2 . 9$ </td><td>0.008</td><td>0.22</td><td>√</td></tr><tr><td>Female basal FSH (mIU/mL)</td><td> $7 . 2 \pm 1 . 9$ </td><td> $7 . 5 \pm 2 . 2$ </td><td>0.103</td><td>0.13</td><td>√</td></tr><tr><td>Female basal LH (mIU/mL)</td><td> $6 . 0 \pm 4 . 3$ </td><td> $5 . 6 \pm 3 . 5$ </td><td>0.228</td><td>0.10</td><td>V</td></tr><tr><td>Female basal E2 (pg/mL)</td><td> $3 8 . 5 \pm 4 0 . 1$ </td><td> $4 1 . 1 \pm 5 9 . 4$ </td><td>0.528</td><td>0.05</td><td></td></tr><tr><td>Female basal PRL (ng/mL)</td><td> $1 8 . 9 \pm 8 . 8$ </td><td> $1 9 . 6 \pm 1 1 . 2$ </td><td>0.380</td><td>0.07</td><td></td></tr><tr><td>Female basal T (ng/mL)</td><td> $0 . 3 \pm 1 . 0$ </td><td> $0 . 3 \pm 0 . 2$ </td><td>0.451</td><td>0.07</td><td></td></tr><tr><td>Female basal P (ng/mL)</td><td> $0 . 4 \pm 1 . 3$ </td><td> $0 . 4 \pm 1 . 5$ </td><td>0.847</td><td>0.02</td><td></td></tr><tr><td>Female AMH (ng/mL)</td><td> $3 . 0 \pm 1 . 7$ </td><td> $2 . 7 \pm 1 . 6$ </td><td>0.035</td><td>0.18</td><td></td></tr><tr><td>Female TSH (mIU/L)</td><td> $2 . 1 \pm 1 . 0$ </td><td> $2 . 1 \pm 1 . 1$ </td><td>0.561</td><td>0.05</td><td></td></tr><tr><td>Duration of infertility (years)</td><td> $2 . 9 \pm 2 . 4$ </td><td> $2 . 9 \pm 2 . 2$ </td><td>0.759</td><td>0.03</td><td></td></tr><tr><td>Female Gn-start-day FSH (mIU/mL)</td><td> $6 . 8 \pm 2 . 1$ </td><td> $7 . 0 \pm 2 . 4$ </td><td>0.402</td><td>0.07</td><td></td></tr><tr><td>Female Gn-start-day LH (mIU/mL)</td><td> $5 . 0 \pm 2 . 6$ </td><td> $4 . 9 \pm 2 . 7$ </td><td>0.573</td><td>0.05</td><td>√</td></tr><tr><td>Female Gn-start-day E2 (pg/mL)</td><td> $3 0 . 7 \pm 1 5 . 6$ </td><td> $3 3 . 6 \pm 2 4 . 1$ </td><td>0.079</td><td>0.14</td><td>V</td></tr><tr><td>Female Gn-start-day P (ng/mL)</td><td> $0 . 3 \pm 0 . 2$ </td><td> $0 . 3 \pm 0 . 3$ </td><td>0.013</td><td>0.21</td><td>√</td></tr><tr><td>Female trigger-day LH (mIU/mL)</td><td> $3 . 1 \pm 2 . 0$ </td><td> $3 . 2 \pm 2 . 9$ </td><td>0.607</td><td>0.04</td><td>V</td></tr><tr><td>Female trigger-day E2 (pg/mL)</td><td> $2 5 9 4 . 6 \pm 1 3 3 6 . 3$ </td><td> $2 4 3 3 . 3 \pm 1 3 8 8 . 2$ </td><td>0.148</td><td>0.12</td><td>√</td></tr><tr><td>Female trigger-day P (ng/mL)</td><td> $0 . 6 \pm 0 . 3$ </td><td> $0 . 7 \pm 0 . 3$ </td><td>0.322</td><td>0.08</td><td></td></tr><tr><td>Trigger-day endometrial thickness (mm)</td><td> $1 1 . 0 \pm 2 . 4$ </td><td> $1 0 . 7 \pm 2 . 4$ </td><td>0.126</td><td>0.13</td><td>√</td></tr><tr><td>Male factors</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Male age (years)</td><td> $3 3 . 3 \pm 4 . 4$ </td><td> $3 4 . 5 \pm 4 . 7$ </td><td>0.002</td><td>0.26</td><td>V</td></tr><tr><td>Male smoking history, n (%)</td><td></td><td></td><td>0.883</td><td></td><td></td></tr><tr><td>Have</td><td>91 (36.3%)</td><td>129 (35.3%)</td><td></td><td></td><td></td></tr><tr><td>None</td><td>160 (63.7%)</td><td> $2 3 6 \ ( 6 4 . 7 \% )$ </td><td></td><td></td><td></td></tr><tr><td>Male height (cm)</td><td> $1 7 3 . 4 \pm 5 . 6$ </td><td> $1 7 3 . 3 \pm 5 . 4$ </td><td>0.857</td><td>0.01</td><td></td></tr><tr><td>Male weight (kg)</td><td> $7 4 . 7 \pm 1 2 . 0$ </td><td> $7 5 . 5 \pm 1 2 . 5$ </td><td>0.398</td><td>0.07</td><td>√</td></tr><tr><td>Male BMI (kg/m²)</td><td> $2 4 . 8 \pm 3 . 4$ </td><td> $2 5 . 1 \pm 3 . 7$ </td><td>0.295</td><td>0.09</td><td>√</td></tr><tr><td>Male serum FSH (mIU/mL)</td><td> $4 . 8 \pm 4 . 1$ </td><td> $4 . 8 \pm 3 . 6$ </td><td>0.891</td><td>0.01</td><td></td></tr><tr><td>Male serum LH (mIU/mL)</td><td> $4 . 6 \pm 2 . 9$ </td><td> $4 . 7 \pm 2 . 9$ </td><td>0.514</td><td>0.05</td><td>√</td></tr><tr><td>Male serum PRL (ng/mL)</td><td> $1 3 . 1 \pm 1 4 . 7$ </td><td> $1 3 . 2 \pm { 7 . 5 }$ </td><td>0.923</td><td>0.01</td><td></td></tr><tr><td>Male serum T (ng/mL)</td><td> $4 . 5 \pm 2 . 9$ </td><td> $5 . 3 \pm 1 6 . 7$ </td><td>0.362</td><td>0.07</td><td></td></tr><tr><td>Sperm concentration  $( 1 0 ^ { 6 } / \mathrm { m L } )$ </td><td> $4 5 . 5 \pm 3 6 . 5$ </td><td> $4 5 . 7 \pm 3 3 . 6$ </td><td>0.954</td><td>0.00</td><td></td></tr><tr><td>Progressive motility (PR, %)</td><td> $3 7 . 3 \pm 1 8 . 0$ </td><td> $3 9 . 0 \pm 1 8 . 1$ </td><td>0.274</td><td>0.09</td><td>√</td></tr><tr><td>Non-progressive motility (NP, %)</td><td> $4 . 3 \pm 4 . 2$ </td><td> $4 . 5 \pm 6 . 7$ </td><td>0.716</td><td>0.03</td><td></td></tr><tr><td>Immotility (IM, %)</td><td> $5 7 . 6 \pm 1 9 . 6$ </td><td> $5 6 . 4 \pm 1 9 . 4$ </td><td>0.445</td><td>0.06</td><td>V</td></tr><tr><td>Normal sperm morphology (%)</td><td> $6 . 6 \pm 5 . 8$ </td><td> $6 . 2 \pm 4 . 3$ </td><td>0.414</td><td>0.07</td><td></td></tr><tr><td>Sperm DFI (%)</td><td> $1 1 . 3 \pm 5 . 6$ </td><td> $1 2 . 1 \pm 5 . 8$ </td><td>0.188</td><td>0.13</td><td>√</td></tr></table>

Continuous variables are mean ± SD; categorical variables are n (%). p values are from Welch t tests for continuous variables and $\chi ^ { 2 }$ or Fisher exact tests for categorical variables, reported without correction for multiple comparisons; SMD is the absolute standardised mean difference. The final column marks variables retained by feature selection (✓). Selection operated on encoded features, in which each category of a categorical variable becomes a separate binary feature, so a variable is marked if any of its categories was selected. The 50 selected features therefore correspond to 47 variables, because three categorical variables (ovarian stimulation protocol, method of fertilization, and number of cleavage-stage embryos transferred) each had two categories selected. Variables excluded for a documented reason are annotated.

## B. Outcome Definition

The prediction target was live birth, defined as the delivery of one or more living infants after at least 24 weeks of gestation, in line with standard fertility-care glossaries [47], versus non-live birth; the non-live birth group comprised cycles with no pregnancy (n = 293), a biochemical pregnancy (n = 27), or a miscarriage (n = 45). Cycles were accordingly assigned to the live birth group or the non-live birth group, and this binary outcome served as the label for all models.

## C. Predictors and Embryo-Morphology Encoding

Candidate predictors were organized into the four clinical domains used throughout this paper and in Table I. Female factors comprised age, height, weight, body-mass index, smoking history, duration of infertility, the basal, Gnstart-day and trigger-day hormonal profile (follicle-stimulating and luteinizing hormone, estradiol, progesterone, prolactin, testosterone, anti-Mullerian and thyroid-stimulating hormone), and trigger-day endometrial thickness. Male factors comprised age, height, weight, body-mass index, smoking history, serum follicle-stimulating and luteinizing hormone, prolactin and testosterone, and semen parameters (concentration, progressive and non-progressive motility, immotility, normal morphology, DNA fragmentation index). Stimulation and laboratory variables comprised the ovarian-stimulation protocol, broadly consistent with guideline practice [48], the method of fertilization, the duration of gonadotropin stimulation, the total gonadotropin, follicle-stimulating hormone, menopausal gonadotropin and luteinizing hormone doses, and the trigger type and dose. Oocyte-retrieval and embryo-development variables comprised the numbers of punctured follicles, retrieved and MII oocytes, 2PN and multipronuclear zygotes, high-quality embryos, embryos available at day 3, blastocyst-cultured embryos and blastocysts formed, with the type and number of embryos transferred and the encoded morphology. Embryo morphology, recorded as qualitative grades, was converted into numerical features using the Peter system for cleavagestage embryos and the Gardner system for blastocysts, so that morphology could be used directly irrespective of transfer stage, an issue of ongoing debate [49].

TABLE I(CONTINUED)
<table><tr><td>Variable</td><td>Live birth (n = 251)</td><td>Non-live birth (n = 365)</td><td>p</td><td>SMD</td><td>Selected</td></tr><tr><td>Stimulation and laboratory</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ovarian stimulation protocol, n (%)</td><td></td><td></td><td>0.795</td><td></td><td>V</td></tr><tr><td>Antagonist regimen</td><td>199 (79.3%)</td><td>296 (81.1%)</td><td></td><td></td><td></td></tr><tr><td>Extra Long Plan</td><td>3 (1.2%)</td><td>7 (1.9%)</td><td></td><td></td><td></td></tr><tr><td>Long protocol for early follicular phase</td><td>21 (8.4%)</td><td>29 (7.9%)</td><td></td><td></td><td></td></tr><tr><td>Luteal phase extension protocol</td><td>26 (10.4%)</td><td>29 (7.9%)</td><td></td><td></td><td></td></tr><tr><td>Microstimulation Program</td><td>2 (0.8%)</td><td>4 (1.1%)</td><td></td><td></td><td></td></tr><tr><td>Method of fertilization, n (%)</td><td></td><td></td><td>0.315</td><td></td><td></td></tr><tr><td>ICSI</td><td>20 (8.0%)</td><td>20 (5.5%)</td><td></td><td></td><td></td></tr><tr><td>IVF</td><td>224 (89.2%)</td><td>338 (92.6%)</td><td></td><td></td><td></td></tr><tr><td>IVF+RICSI</td><td>3 (1.2%)</td><td>1 (0.3%)</td><td></td><td></td><td></td></tr><tr><td>RICSI</td><td>4 (1.6%)</td><td>6 (1.6%)</td><td></td><td></td><td></td></tr><tr><td>Duration of Gn stimulation (days)</td><td> $1 0 . 0 \pm 1 . 6$ </td><td> $9 . 7 \pm 1 . 6$ </td><td>0.045</td><td>0.17</td><td>√</td></tr><tr><td>Total Gn dose (IU)</td><td> $2 5 2 2 . 0 \pm 7 3 6 . 7$ </td><td> $2 5 2 7 . 2 \pm 6 6 7 . 3$ </td><td>0.928</td><td>0.01</td><td>√</td></tr><tr><td>Total FSH dose (IU)</td><td> $2 1 0 2 . 0 \pm 7 5 5 . 1$ </td><td> $2 1 5 8 . 3 \pm 6 0 6 . 4$ </td><td>0.326</td><td>0.08</td><td>V</td></tr><tr><td>Total HMG dose (IU)</td><td> $3 7 4 . 0 \pm 5 1 4 . 2$ </td><td> $3 6 3 . 1 \pm 4 3 3 . 1$ </td><td>0.784</td><td>0.02</td><td></td></tr><tr><td>Total LH dose (IU)</td><td> $4 3 . 0 \pm 1 1 3 . 8$ </td><td> $2 3 . 8 \pm 9 7 . 7$ </td><td>0.030</td><td>0.18</td><td></td></tr><tr><td>Urinary hCG trigger dose (IU)</td><td> $6 3 7 4 . 5 \pm 4 7 9 7 . 0$ </td><td> $6 6 0 8 . 2 \pm 5 1 2 8 . 3$ </td><td>0.564</td><td>0.05</td><td></td></tr><tr><td>Recombinant hCG trigger dose (µg)</td><td> $1 2 . 9 \pm 5 5 . 5$ </td><td> $7 . 5 \pm 4 2 . 8$ </td><td>0.194</td><td>0.11</td><td>√</td></tr><tr><td>GnRH agonist trigger dose (mg)</td><td> $0 . 1 \pm 0 . 1$ </td><td> $0 . 1 \pm 0 . 1$ </td><td>0.633</td><td>0.04</td><td></td></tr><tr><td>Trigger type, n (%)</td><td></td><td></td><td>0.704</td><td></td><td></td></tr><tr><td>Double trigger</td><td>106 (42.2%)</td><td>161 (44.1%)</td><td></td><td></td><td></td></tr><tr><td>Single trigger</td><td>145 (57.8%)</td><td>204 (55.9%)</td><td></td><td></td><td></td></tr><tr><td>Oocyte retrieval and embryo development</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Number of follicles punctured</td><td> $1 2 . 6 \pm 5 . 2$ </td><td> $1 1 . 6 \pm 5 . 5$ </td><td>0.019</td><td>0.19</td><td>√</td></tr><tr><td>Number of oocytes retrieved</td><td> $1 0 . 2 \pm 4 . 3$ </td><td> $9 . 2 \pm 4 . 5$ </td><td>0.003</td><td>0.24</td><td>√</td></tr><tr><td>Number of MII oocytes</td><td> $9 . 9 \pm 4 . 2$ </td><td> $8 . 9 \pm 4 . 5$ </td><td>0.006</td><td>0.23</td><td>√</td></tr><tr><td>Number of 2PN zygotes</td><td> $7 . 2 \pm 3 . 4$ </td><td> $6 . 1 \pm 3 . 3$ </td><td>&lt;0.001</td><td>0.31</td><td>√</td></tr><tr><td>Number of multipronuclear zygotes</td><td> $0 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td><td></td><td></td><td></td></tr><tr><td>Number of high-quality embryos</td><td> $4 . 9 \pm 2 . 7$ </td><td> $4 . 2 \pm 2 . 6$ </td><td>0.002</td><td>0.26</td><td>√</td></tr><tr><td>Number of embryos available at D3</td><td> $6 . 6 \pm 3 . 2$ </td><td> $5 . 7 \pm 3 . 1$ </td><td>&lt;0.001</td><td>0.30</td><td>√</td></tr><tr><td>Number of blastocyst-cultured embryos</td><td> $6 . 5 \pm 3 . 2$ </td><td> $5 . 7 \pm 3 . 2$ </td><td>0.002</td><td>0.27</td><td>√</td></tr><tr><td>Number of blastocysts formed</td><td> $5 . 2 \pm 2 . 8$ </td><td> $4 . 6 \pm 2 . 9$ </td><td>0.006</td><td>0.24</td><td>√</td></tr><tr><td>Types of embryos transferred, n (%)</td><td></td><td></td><td>0.001</td><td></td><td>L</td></tr><tr><td>Blastocyst stage embryo</td><td>172 (68.5%)</td><td>201 (55.1%)</td><td></td><td></td><td></td></tr><tr><td>Cleavage stage embryo</td><td>79 (31.5%)</td><td>164 (44.9%)</td><td></td><td></td><td></td></tr><tr><td>Embryo morphology (52 distinct values)</td><td></td><td></td><td>0.009</td><td></td><td>(encoded as 10 columns)</td></tr><tr><td>Number of cleavage-stage embryos transferred</td><td> $1 . 5 \pm 0 . 5$ </td><td> $1 . 3 \pm 0 . 4$ </td><td>&lt;0.001</td><td>0.56</td><td>√</td></tr><tr><td>Number of blastocysts transferred</td><td> $1 . 0 \pm 0 . 2$ </td><td> $1 . 0 \pm 0 . 2$ </td><td>0.420</td><td>0.08</td><td>√</td></tr><tr><td>Blastocyst 1 expansion quality</td><td> $1 . 4 \pm 0 . 9$ </td><td> $1 . 1 \pm 1 . 0$ </td><td>&lt;0.001</td><td>0.28</td><td>√</td></tr><tr><td>Blastocyst 1 ICM (weighted)</td><td> $0 . 8 \pm 0 . 6$ </td><td> $0 . 7 \pm 0 . 6$ </td><td>0.002</td><td>0.25</td><td>√</td></tr><tr><td>Blastocyst 1 TE (weighted)</td><td> $1 . 3 \pm 1 . 0$ </td><td> $1 . 0 \pm 1 . 0$ </td><td>&lt;0.001</td><td>0.32</td><td></td></tr><tr><td>Blastocyst 2 expansion quality</td><td> $0 . 0 \pm 0 . 3$ </td><td></td><td>0.775</td><td></td><td></td></tr><tr><td>Blastocyst 2 ICM (weighted)</td><td> $0 . 0 \pm 0 . 1$ </td><td> $0 . 0 \pm 0 . 3$ </td><td></td><td>0.02</td><td></td></tr><tr><td>Blastocyst 2 TE (weighted)</td><td> $0 . 0 \pm 0 . 2$ </td><td> $0 . 0 \pm 0 . 2$ </td><td>0.626</td><td>0.04</td><td></td></tr><tr><td>Cleavage embryo 1 grade</td><td> $0 . 9 \pm 1 . 4$ </td><td> $0 . 0 \pm 0 . 3$ </td><td>0.575</td><td>0.05</td><td></td></tr><tr><td>Cleavage 1 cell-count quality</td><td> $0 . 8 \pm 1 . 2$ </td><td> $1 . 3 \pm 1 . 5$ </td><td>0.001</td><td>0.26</td><td>√</td></tr><tr><td>Cleavage embryo 2 grade</td><td> $0 . 5 \pm 1 . 1$ </td><td> $1 . 1 \pm 1 . 3$   $0 . 3 \pm 0 . 9$ </td><td>0.002 0.059</td><td>0.26</td><td>√ √</td></tr><tr><td></td><td></td><td></td><td>0.084</td><td>0.16</td><td>√</td></tr><tr><td>Cleavage 2 cell-count quality</td><td> $0 . 4 \pm 1 . 0$ </td><td> $0 . 3 \pm 0 . 8$ </td><td></td><td>0.14</td><td></td></tr></table>

## D. A Differential ICM/TE Weighting for Blastocyst Grading

A specific methodological contribution of this work con cerns how embryo morphology is encoded. The raw morphology text, describing one or two transferred embryos in either the Gardner blastocyst system (expansion, inner cell mass [ICM], trophectoderm [TE]) or the cleavage-stage system (grade and cell number), was converted into ten numeric columns, with separate slots for up to two embryos and for the two systems. Blastocyst ICM and TE letters were mapped A/B/C to 3/2/1 and cleavage Roman grades I-IV to 4-1. Rather than treat ICM and TE as equally informative, as conventional Gardner grading implicitly does, we weight them differentially: informed by a correlation analysis in this

Embryo Morphology Encoding Scheme (as implemented)

![](images/e17243bd2110486c323db21acc5d766a8fef12d6e1d135552228ccdfa75a62f3.jpg)

![](images/78603fbefbb3e9c61a727c75cf5cd828b82045d06fc715ed79b25300b56ff8d4.jpg)  
Fig. 1. Diagram of the embryo morphology encoding scheme.

cohort, in which TE was more strongly associated with live birth than ICM, the composite stores ICM at half weight and TE at full weight $( \mathrm { I C M } \ \times \ 0 . 5 , \ \mathrm { T E } \ \times \ 1 . 0 )$ . This resolves the ambiguity between blastocysts sharing an expansion stage but with mirror-image grades, ranking the TE-superior 4BA above 4AB. Two further non-monotonic facts are captured by inverted-U optimality scores: cleavage cell count (best around eight cells) and blastocyst expansion (best at intermediate stages). The weighting is applied during dataset construction, giving models a more expressive morphology feature than an equal-weight grade sum; the TE-weighted feature was among the 50 selected and, in cross-checks, among the most influential. Fig. 1 summarizes the encoding scheme.

## E. Preprocessing and Feature Selection

Records were cleaned to coerce stray text entries to missing values. To prevent leakage, all fitting was confined to the training split: numerical variables were imputed with a k-nearestneighbour imputer $( k = 5 )$ and scaled with a robust scaler (median and interquartile range). Female weight was dropped as collinear with body-mass index, and categorical variables were one-hot encoded with the first level as reference. Of the 71 collected predictors listed in Table I, two were removed before modelling: the raw embryo-morphology text, which is represented by the ten numeric columns described above, and female weight. The remaining 69 variables comprised 61 numerical variables (including those ten embryo columns) and 8 categorical variables, and one-hot encoding the latter produced 15 dummies, giving 76 candidate features. From these, the 50 most informative were retained using univariate selection (SelectKBest with the ANOVA F-statistic), fit on the training split only, a deliberately simple, transparent filter suited to a small cohort. The same preprocessed feature set and the same stratified partition were used for every model, so that performance differences reflect the models rather than differences in data handling. Because selection operates on encoded features, a categorical variable can contribute more than one: three variables (ovarian stimulation protocol, method of fertilization, and number of cleavage-stage embryos transferred) each had two categories selected, so the 50 selected features correspond to 47 of the 71 collected variables; Table I records the modelling status of every collected variable alongside its distribution in each outcome group.

TABLE II  
HAN + LR ARCHITECTURE AND TRAINING CONFIGURATION
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Feature Attention</td><td>Learnable per-feature gate, sigmoid,</td></tr><tr><td>Module</td><td>element-wise multiplicative</td></tr><tr><td>MLP layers</td><td>50 → 24 (LayerNorm, ReLU, dropout 0.25); 24 → 12 (ReLU, dropout 0.15); 12 → 1 (sigmoid)</td></tr><tr><td>HAN ensemble</td><td>3 seeds (42, 123, 456), probabilities averaged</td></tr><tr><td>Parameters</td><td>≈1,635 per HAN instance; 4,957 total</td></tr><tr><td>Ensemble rule</td><td> $p = w p _ { \mathrm { H A N } } + \left( 1 - w \right) p _ { \mathrm { L R } } ; w = 0 . 7 0 \mathrm { ~ b y }$  validation grid search</td></tr><tr><td>Optimizer / loss</td><td>AdamW (lr 0.002, weight decay 0.001); binary cross-entropy</td></tr><tr><td>Schedule</td><td>≤80 epochs, batch 32, early stopping on validation AUC</td></tr><tr><td>Stability</td><td>Xavier initialization; gradient clipping at max-norm 1.0</td></tr></table>

## F. Model: Hybrid Attention Network and the HAN + LR Ensemble

The HAN backbone has two stages. A Feature Attention Module first applies a global, dataset-level learnable importance gate: a weight vector is passed through a sigmoid and multiplied element-wise with the input features, allowing the network to up- or down-weight predictors. Unlike per-sample attention in large transformers, this gate is shared across the cohort, keeping the parameter count low and suiting a small clinical dataset. The gated features then pass through a compact multilayer perceptron (50→24 with layer normalization, ReLU, and dropout 0.25; 24→12 with ReLU and dropout 0.15; 12→1 with a sigmoid), yielding a live-birth probability. Each HAN instance contains approximately 1,635 trainable parameters. To reduce variance from random initialization, the backbone is realized as an ensemble of three independently seeded instances whose probabilities are averaged.

The proposed model combines the HAN backbone with an L2-regularized logistic-regression (LR) model trained on the same features. The final probability is a convex combination $p ~ = ~ w \cdot p _ { \mathrm { H A N } } + ( 1 - w ) \cdot p _ { \mathrm { L R } }$ , with the mixing weight selected by a grid search that maximized validation AUC. The search selected $w = 0 . 7 0$ , weighting the attention backbone more heavily than the linear model, reflecting that the HAN backbone was the stronger of the two components on this cohort while logistic regression still contributed complementary structure. Counting all components, the ensemble comprises 4,957 parameters (three HAN instances, the LR model, and the mixing weight).

## G. Training Configuration

HAN instances were trained by minimizing binary crossentropy with the AdamW optimizer (learning rate 0.002, weight decay 0.001), for up to 80 epochs with a batch size of 32, using Xavier initialization, gradient clipping at a maximum norm of 1.0, and early stopping on the best validation AUC. Key settings are summarized in Table II.

## H. Comparator Models

We compared the HAN + LR ensemble with: classical machine-learning models (logistic regression, random forest, gradient boosting, support-vector machine, k-nearest neighbours, naive Bayes, and XGBoost); a tabular foundation model (TabPFN) and a TabPFN + LR hybrid; and a family of pretrained language models adapted to tabular data by text serialization, in which each record is rendered as a natural-language string of feature–value pairs and embedded (MiniLM, Distil-BERT, BERT, ELECTRA, and RoBERTa) [17], [18]. Classical models used standard, lightly tuned configurations; the TabPFN + LR hybrid used the same validation-tuned weighting as the proposed ensemble, so that only the neural component differs. The language-model pipeline used one fixed serialization template across all encoders.

## I. Evaluation

Data were partitioned by stratified sampling into 523 training and 93 held-out test cycles (approximately 85% and 15%), preserving the class ratio. Cross-validation within the training set was used for model selection and tuning, including the ensemble mixing weight; final performance was reported on the held-out test set. The primary metric was the area under the receiver-operating-characteristic curve (AUC-ROC), which is threshold-independent and appropriate for a moderately imbalanced binary outcome; accuracy, precision, recall, and F1 were reported as secondary metrics at the default 0.5 decision threshold. Bootstrap resampling of the test set was used to characterize the stability of the estimates. Reporting follows the TRIPOD+AI checklist [10].

## IV. RESULTS

## A. Cohort

The curated cohort comprised 616 fresh embryo-transfer cycles, partitioned into 523 training and 93 test cycles (of which 38 were live births). The live birth group was the minority (251 cycles, 40.7%) and the non-live birth group the majority (365 cycles, 59.3%); we preserved this ratio by stratified sampling rather than applying explicit resampling. Baseline characteristics by outcome group are given in Table I.

The 50 selected features spanned all four predictor domains (Table I) but were dominated by oocyte-retrieval and embryodevelopment variables and by the female hormonal profile. This composition is clinically coherent, since embryo quality and yield are among the strongest correlates of transfer success, but it also signals the pathway-dependence discussed below.

## B. Primary Model Comparison

Across the eleven models evaluated on the held-out test set, the HAN + LR ensemble achieved the highest AUC-ROC (0.710), with an accuracy of 0.710 and F1 of 0.630 (Table III). It outperformed the HAN backbone evaluated alone (0.700), the strongest classical baseline logistic regression (0.667), the TabPFN + LR hybrid (0.666), and standalone TabPFN (0.650). The remaining models (support-vector machine, gradient boosting, XGBoost, k-nearest neighbours, naive Bayes, and random forest) trailed with AUC-ROC between 0.56 and 0.63; receiver-operating-characteristic curves for the models are shown in Fig. 2.

TABLE III  
TEST-SET PERFORMANCE OF THE ELEVEN EVALUATED MODELS, RANKED BY AUC-ROC
<table><tr><td>Model</td><td>AUC</td><td>Acc</td><td>Prec</td><td>Rec</td><td>F1</td></tr><tr><td>HAN + LR (proposed)</td><td>0.710</td><td>0.710</td><td>0.657</td><td>0.605</td><td>0.630</td></tr><tr><td>HAN backbone (alone)</td><td>0.700</td><td>0.667</td><td>0.590</td><td>0.605</td><td>0.597</td></tr><tr><td>Logistic regression</td><td>0.667</td><td>0.645</td><td>0.558</td><td>0.632</td><td>0.593</td></tr><tr><td>TabPFN +LR</td><td>0.666</td><td>0.656</td><td>0.615</td><td>0.421</td><td>0.500</td></tr><tr><td>TabPFN</td><td>0.650</td><td>0.613</td><td>0.545</td><td>0.316</td><td>0.400</td></tr><tr><td>Support-vector machine</td><td>0.632</td><td>0.624</td><td>0.588</td><td>0.263</td><td>0.364</td></tr><tr><td>Gradient boosting</td><td>0.627</td><td>0.591</td><td>0.500</td><td>0.395</td><td>0.441</td></tr><tr><td>XGBoost</td><td>0.624</td><td>0.613</td><td>0.533</td><td>0.421</td><td>0.471</td></tr><tr><td>k-nearest neighbours</td><td>0.613</td><td>0.645</td><td>0.586</td><td>0.447</td><td>0.507</td></tr><tr><td>Naïve Bayes</td><td>0.581</td><td>0.570</td><td>0.471</td><td>0.421</td><td>0.444</td></tr><tr><td>Random forest</td><td>0.556</td><td>0.591</td><td>0.500</td><td>0.263</td><td>0.345</td></tr></table>

Higher is better for all metrics.

![](images/6b4d730751553ae94841edf2d1d9f53ac90758b460bb339476b0364544cecdb9.jpg)  
Fig. 2. Receiver-operating-characteristic (AUC-ROC) curves for the evaluated models on the held-out test set.

Two observations are notable. First, the HAN + LR ensemble improved on its own components by +0.010 AUC over the HAN backbone (0.700) and +0.043 over logistic regression (0.667); the gain is modest but consistent, and the HAN backbone alone again outperformed logistic regression and every other single model, ranking second only to the hybrid. Second, unlike the HAN backbone, the tabular foundation model did not help the linear model: standalone TabPFN reached only 0.650 and the TabPFN + LR hybrid 0.666, neither exceeding logistic regression (0.667), so the complementary signal came specifically from pairing the linear model with the attention backbone rather than a foundation model.

![](images/3e558d52cf1cf80f9f9c6bd7b7f1fee39a0d485842df95f3922cb26ea4bfe908.jpg)  
Fig. 3. Top 15 predictors by consensus importance, averaged across the five models that expose an importance measure and coloured by the clinical domains defined in Table I.

The threshold-dependent secondary metrics highlight different operating behaviours that AUC-ROC alone conceals. TabPFN, the support-vector machine, and random forest favoured precision at the cost of recall (0.263 to 0.316), flagging few cycles as live births, whereas logistic regression achieved the highest recall (0.632) at lower precision (0.558). The HAN + LR ensemble led on four of the five metrics, AUC-ROC, accuracy (0.710), precision (0.657), and F1 (0.630), while maintaining a recall of 0.605. Because these metrics use a fixed 0.5 threshold, they are sensitive to the operating point and to sampling variation; AUC-ROC is therefore the more reliable basis for ranking, and the bootstrap intervals around them were wide.

## C. Feature Importance Across Models

To characterize which predictors each model relied on, we extracted each model’s native importance measure: Gini importance (random forest), Friedman MSE improvement (gradient boosting), gain-based weight (XGBoost), absolute coefficient magnitude (logistic regression), and the learned attention weights (HAN). Because these live on different scales, we min-max scaled them within each model and averaged the normalized values into a consensus ranking (Table IV, Figs. 3 and 4).

Three patterns emerged. First, the differentially weighted blastocyst trophectoderm feature (Blastocyst 1 TE) was the highest-ranked predictor overall (4.53% of consensus importance) and carried by far the largest logistic-regression coefficient, which supports the encoding scheme introduced above; the cleavage embryo grades and the blastocyst expansionquality score also appeared among the top 15, with the weighted Blastocyst 1 ICM feature ranked 17th. Second, the models disagreed markedly about which predictors matter (Fig. 4). The tree ensembles concentrated importance on continuous physiological variables, chiefly female BMI, sperm DFI, female Gn-start-day progesterone, and male serum prolactin, while assigning almost no importance to the one-hot categorical indicators; logistic regression showed the opposite emphasis, ranking the categorical stimulation and laboratory indicators and the embryo-morphology features highest. This divergence is expected: axis-aligned trees split poorly on sparse binary dummies whereas a linear model can assign them large coefficients, so a single-model ranking would mislead. Third, the HAN attention weights spanned only about 0.50 to 0.56 across the 30 highest-ranked features, indicating mild global re-weighting rather than sharp feature selection; the ordering it induces should not be over-interpreted.

![](images/9e3c614693bc5bb6ee8b8918811d110cf60119842804fd54dbd40e8a988e9bd3.jpg)  
Fig. 4. Relative importance of each of the top 15 predictors within each model, min-max scaled per model so that differently scaled measures can be compared. Scaling amplifies the narrow spread of the HAN attention weights and should be read as ordering, not magnitude.

![](images/190668512644799c9005a26f9495038c779e212f0b54e781e0d429d02e7c927c.jpg)  
Fig. 5. Share of total consensus importance contributed by each of the four clinical domains defined in Table I, with the number of selected features in each domain.

Aggregated by the same four clinical domains used in Table I, oocyte-retrieval and embryo-development variables contributed the largest share of consensus importance, with the remainder distributed across female factors, stimulation and laboratory variables, and male factors (Fig. 5).

TABLE IV  
TOP 15 PREDICTORS BY CONSENSUS IMPORTANCE, WITH THE NATIVE IMPORTANCE REPORTED BY EACH MODEL
<table><tr><td>Feature</td><td>Domain</td><td>Consensus (%)</td><td>RF</td><td>GB</td><td>XGB</td><td>LR</td><td>HAN</td></tr><tr><td>Blastocyst 1 TE (weighted)</td><td>Oocyte retrieval and embryo development</td><td>4.53</td><td>0.008</td><td>0.019</td><td>0.014</td><td>1.208</td><td>0.537</td></tr><tr><td>Ovarian stimulation protocol (Extra Long Plan)</td><td>Stimulation and laboratory</td><td>3.35</td><td>0.001</td><td>0.000</td><td>0.000</td><td>0.759</td><td>0.560</td></tr><tr><td>Method of fertilization (IVF+RICSI)</td><td>Stimulation and laboratory</td><td>2.93</td><td>0.001</td><td>0.000</td><td>0.000</td><td>0.599</td><td>0.555</td></tr><tr><td>Cleavage embryo 1 grade</td><td>Oocyte retrieval and embryo development</td><td>2.66</td><td>0.003</td><td>0.000</td><td>0.001</td><td>0.523</td><td>0.520</td></tr><tr><td>Number of follicles punctured</td><td>Oocyte retrieval and embryo development</td><td>2.61</td><td>0.024</td><td>0.007</td><td>0.009</td><td>0.464</td><td>0.522</td></tr><tr><td>Number of oocytes retrieved</td><td>Oocyte retrieval and embryo</td><td>2.57</td><td>0.020</td><td>0.019</td><td>0.015</td><td>0.427</td><td>0.532</td></tr><tr><td>Female age (years)</td><td>development Female factors</td><td>2.50</td><td>0.025</td><td>0.020</td><td>0.022</td><td>0.359</td><td>0.558</td></tr><tr><td>Female Gn-start-day P (ng/mL)</td><td>Female factors</td><td>2.48</td><td>0.036</td><td>0.040</td><td>0.041</td><td>0.308</td><td>0.553</td></tr><tr><td>Sperm DFI (%)</td><td>Male factors</td><td>2.33</td><td>0.038</td><td>0.034</td><td>0.041</td><td>0.252</td><td>0.551</td></tr><tr><td>Cleavage embryo 2 grade</td><td>Oocyte retrieval and embryo development</td><td>2.31</td><td>0.005</td><td>0.018</td><td>0.011</td><td>0.355</td><td>0.520</td></tr><tr><td>Female BMI (kg/m²)</td><td>Female factors</td><td>2.30</td><td>0.040</td><td>0.050</td><td>0.041</td><td>0.233</td><td>0.542</td></tr><tr><td>Blastocyst 1 expansion quality</td><td>Oocyte retrieval and embryo development</td><td>2.28</td><td>0.006</td><td>0.000</td><td>0.000</td><td>0.377</td><td>0.517</td></tr><tr><td>Number of 2PN zygotes</td><td>Oocyte retrieval and embryo development</td><td>2.19</td><td>0.020</td><td>0.005</td><td>0.013</td><td>0.316</td><td>0.510</td></tr><tr><td>Number of blastocysts formed</td><td>Oocyte retrieval and embryo development</td><td>2.19</td><td>0.022</td><td>0.008</td><td>0.013</td><td>0.317</td><td>0.502</td></tr><tr><td>Male serum PRL (ng/mL)</td><td>Male factors</td><td>2.18</td><td>0.036</td><td>0.029</td><td>0.035</td><td>0.253</td><td>0.505</td></tr></table>

Variable names and clinical domains follow Table I. Values are Gini importance (random forest), Friedman MSE improvement (gradient boosting), gain-based weight (XGBoost), absolute coefficient magnitude (logistic regression), and the learned attention weight (HAN). Note the differing native scales, and in particular the narrow range of the HAN attention weights.

## D. Ablation

Module-level ablations on the validation folds were consistent with the design choices. Removing the Feature Attention Module reduced validation AUC relative to the full backbone, supporting the contribution of the learned importance gate; the chosen dropout configuration (0.25 and 0.15) outperformed both no-dropout and heavier-dropout settings; and the grid search over mixing weights favoured a HAN-weighted combination (w = 0.7) over more balanced settings (w = 0.5 and w = 0.3 were lower on validation).

## E. Cross-Validation Stability

The test-set ranking was consistent with the cross-validation used for model selection: the HAN + LR ensemble and the HAN backbone led under both, and the tree- and distancebased models trailed in both. This agreement, with threeseed averaging, supports the stability of the ordering, though absolute values carry wide confidence intervals at this cohort size.

## F. Language-Model Baselines

Adapting pre-trained language models by text serialization was not competitive: the best reached an AUC of about 0.61, well below the classical and ensemble models, indicating that at this cohort size general-purpose text encoders do not capture the structured clinical signal as effectively as models operating on the tabular features directly.

## V. DISCUSSION

On a small, real-world IVF/ICSI cohort, a compact attention network combined with logistic regression predicted live birth more accurately than either component alone and than foundation-model and language-model baselines. The attention backbone was the stronger single component, the linear model contributed complementary structure, and a HAN-weighted average (w = 0.7) gave the best and most stable prediction. That combining logistic regression with the HAN backbone helped while combining it with TabPFN did not suggests the benefit comes from genuine diversity between the two models rather than from ensembling per se.

These findings match a broader literature reporting that, on small clinical tabular datasets, well-regularized classical models remain competitive with, and often superior to, elaborate deep architectures [45], [46], and that careful feature engineering matters more than raw capacity. They also complement the predominantly image-based ART literature [12]–[14] by showing what is achievable from routinely captured structured data alone, aligning with work on advanced machine-learning paradigms for IVF prognostication [50] and with calls to weigh practical implementation considerations alongside raw accuracy [51].

The absolute discrimination achieved (AUC-ROC around 0.67 to 0.71) is in the range that is plausible, and arguably honest, for a strict live-birth endpoint predicted before transfer from a single-centre cohort. Studies built on large registries sometimes report higher figures [24], [28], but they often predict easier endpoints, benefit from larger samples, or omit external validation; conversely, optimistic results in small studies frequently reflect leakage or the absence of a heldout test set. We aimed for a realistic rather than a maximal estimate of performance.

The choice of a hybrid was deliberate. A single large neural network overfits on a few hundred samples, while a purely linear model cannot capture interactions. Combining a compact attention network with logistic regression lets the convex combination act as a regularizer, which proved more effective than either component alone and than pairing the linear model with a foundation model. For small clinical cohorts, combining complementary simple models appears a more dependable route to robust performance than scaling up a single architecture.

The differential ICM/TE weighting is, to our knowledge, a novel encoding choice for this prediction task. Conceptually it resolves the tie between discordant blastocysts (for example 4AB versus 4BA) that equal-weight Gardner scoring cannot, and the TE-dominant weighting we found is consistent with evidence that trophectoderm quality is a strong correlate of implantation and live birth. We stress, however, that the association of the composite score with live birth was weak in absolute terms and that the weights were tuned on a single cohort; the scheme should therefore be regarded as a transparent, promising encoding to be validated externally rather than as a settled clinical scoring rule.

Within these limits, the study contributes a transparently described dataset and a reproducible model competitive with modern baselines while small enough (under 5,000 parameters) to train and audit easily [10], [11].

The compactness of the model is itself a practical advantage: the ensemble trains in minutes on a single workstation and needs no specialized hardware at inference, lowering the barrier to local re-training and audit as regulators move toward lifecycle oversight of AI in reproductive medicine [52], [53].

Before any clinical use, two steps are essential beyond discrimination. First, the probabilities should be formally calibrated, since a well-discriminating model can still be poorly calibrated and calibrated probabilities are what a counselling clinician needs. Second, the decision threshold should reflect the clinical costs of false positives and negatives rather than defaulting to 0.5. Both, with prospective evaluation, form part of the planned external-validation work.

## A. Clinical Interpretation of the Leading Predictors

The predictors carrying the most weight are clinically coherent, supporting the model’s face validity even though the analysis is associative. The leading role of the weighted trophectoderm grade agrees with evidence that trophectoderm quality tracks implantation and live birth at least as closely as the inner cell mass, and is the empirical counterpart of the encoding decision described above. The prominence of Gnstart-day progesterone matches the established observation that premature progesterone elevation compromises endometrial receptivity and depresses fresh-transfer success, motivating freeze-all strategies in affected cycles. Female age and bodymass index are firmly established determinants of oocyte competence, and the counts of punctured follicles, retrieved oocytes, and blastocysts index ovarian response and the size of the embryo cohort available for selection. Notably, two male factors, sperm DFI and serum prolactin, ranked highly in the tree ensembles, echoing a growing literature that paternal factors contribute measurably to live birth yet are underrepresented in prediction models.

Two cautions attach. First, several of these variables are pathway-dependent: embryo counts and morphology are products of earlier steps in the same cycle, so their predictive value does not imply that intervening on them would change the outcome. Second, the disagreement between model families means no single ranking is definitive; we report the consensus and per-model detail so this uncertainty stays visible.

## VI. LIMITATIONS

Several limitations should be acknowledged. First, the dataset is single-centre and of modest size, which bounds generalizability and inflates the variance of subgroup estimates; external, multi-centre validation is required before any clinical use. Second, the design is retrospective. Third, the endpoint is binary live birth, which does not capture the full outcome pathway (e.g. biochemical pregnancy, clinical pregnancy, miscarriage). Fourth, the language-model baselines were evaluated with a single serialization scheme and may improve with more elaborate prompting or fine-tuning. Finally, our reliance on univariate feature selection and a single centre means the selected feature set may not transfer unchanged to other populations, and the importance analysis is associative and model-dependent rather than causal.

## VII. CONCLUSION

We developed a curated single-centre IVF/ICSI dataset, a differential ICM/TE encoding of blastocyst morphology that separates discordant grades which conventional equal-weight Gardner scoring cannot, and a HAN + LR ensemble for livebirth prediction after fresh embryo transfer. The ensemble achieved the best discrimination among eleven models, exceeding both its own components and contemporary tabular foundation-model and language-model baselines, while remaining compact and reproducible. The dataset, the encoding scheme, and the model provide a foundation for the next step, prospective, multi-centre external validation, which is necessary before such a tool can responsibly inform patient care.

## FUNDING

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

## DATA AND CODE AVAILABILITY

Data. The clinical dataset analysed in this study is not publicly available. It consists of retrospective patient records from the Reproductive Medicine Center of Zhongnan Hospital, Wuhan University, and its release is restricted by patient-confidentiality requirements and by the conditions of the institutional ethics approval. A description of the dataset, including the variable list, the encoding scheme, live-birth-prediction/tree/main/data. Deidentified data may be made available to qualified researchers by the corresponding authors on reasonable request, subject to approval by the Ethics Committee of Zhongnan Hospital of Wuhan University and to a data-use agreement.

and the conditions for access, is provided at https: //github.com/abdul-raheem-shahzad/ivf-

Code. All code required to reproduce the preprocessing, the embryo-morphology encoding, the models, and every table and figure in this paper is openly available at https://github.com/abdul-raheem-shahzad/ ivf-live-birth-prediction.

## ETHICS APPROVAL AND CONSENT TO PARTICIPATE

This study was approved by the Ethics Committee of Zhongnan Hospital of Wuhan University (Ethical Approval No. 2025346K). Due to the retrospective nature of the study, the requirement for informed consent was waived by the Ethics Committee.

## CONFLICT OF INTEREST

The authors report no conflicts of interest.

[1] World Health Organization, “1 in 6 people globally affected by infertility,” 2023. [Online]. Available: https://www.who.int/ news/item/04-04-2023-1-in-6-people-globallyaffected-by-infertility

[2] Human Fertilisation and Embryology Authority, “Fertility treatment 2023: trends and figures,” 2025.

[3] S. Zhu, H. Xu, R. Li et al., “Development and validation of a machine learning-based predictive model for live birth outcomes following fresh embryo transfer in patients with endometriosis,” J. Assist. Reprod. Genet., vol. 42, no. 11, pp. 3853–3867, 2025.

[4] S. Stormlund, N. Sopa, A. Zedeler et al., “Freeze-all versus fresh blastocyst transfer strategy during in vitro fertilisation in women with regular menstrual cycles: multicentre randomised controlled trial,” BMJ, vol. 370, p. m2519, 2020.

[5] ESHRE Guideline Group on the Number of Embryos to Transfer, “ESHRE guideline: number of embryos to transfer during IVF/ICSI,” Hum. Reprod., vol. 39, no. 4, pp. 647–657, 2024.

[6] Practice Committee of the American Society for Reproductive Medicine and the Society for Assisted Reproductive Technology, “Guidance on the limits to the number of embryos to transfer: a committee opinion,” Fertil. Steril., vol. 116, no. 3, pp. 651–654, 2021.

[7] M. B. Ratna, S. Bhattacharya, and D. J. McLernon, “External validation of models for predicting cumulative live birth over multiple complete cycles of IVF treatment,” Hum. Reprod., vol. 38, no. 10, pp. 1998–2010, 2023.

[8] I. Henderson, M. Rabilloud, L. Bujan et al., “Predicting the outcomes of assisted reproductive technology treatments: a systematic review and quality assessment of prediction models,” F&S Rev., vol. 2, no. 1, pp. 1– 10, 2021.

[9] C. L. Andaur Navarro, J. A. A. Damen, T. Takada et al., “Risk of bias in studies on prediction models developed using supervised machine learning techniques: systematic review,” BMJ, vol. 375, p. n2281, 2021.

[10] G. S. Collins, K. G. M. Moons, P. Dhiman et al., “TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods,” BMJ, vol. 385, p. e078378, 2024.

[11] K. G. M. Moons, J. A. A. Damen, T. Kaul et al., “PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods,” BMJ, vol. 388, p. e082505, 2025.

[12] A. Mina, A. Favre Inhofer, G. Le Conte et al., “Predicting pregnancy outcomes in IVF cycles: a systematic review and diagnostic metaanalysis of artificial intelligence in embryo assessment,” Contracept. Reprod. Med., vol. 10, no. 1, p. 59, 2025.

[13] J. Theilgaard Lassen, M. F. Kragh, J. Rimestad et al., “Development and validation of deep learning based embryo selection across multiple days of transfer,” Sci. Rep., vol. 13, no. 1, p. 4235, 2023.

[14] C. M. Louis, N. Handayani, T. Aprilliana et al., “Review of computer vision application in in vitro fertilization: the application of deep learning-based computer vision technology in the world of IVF,” J. Assist. Reprod. Genet., vol. 38, no. 7, pp. 1627–1639, 2021.

[15] S. O. Arık and T. Pfister, “TabNet: attentive interpretable tabular<sup>¨</sup> learning,” Proc. AAAI Conf. Artif. Intell., vol. 35, no. 8, pp. 6679–6687, 2021.

[16] Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, “Revisiting deep learning models for tabular data,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2021.

[17] N. Hollmann, S. Muller, K. Eggensperger, and F. Hutter, “TabPFN:¨ a transformer that solves small tabular classification problems in a second,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2023.

[18] S. Hegselmann, A. Buendia, H. Lang et al., “TabLLM: few-shot classification of tabular data with large language models,” in Proc. Int. Conf. Artif. Intell. Statist. (AISTATS), 2023.

[19] “SART CORS IVF registry: looking to the past to shape future perspectives,” Fertil. Steril., 2022.

[20] M. S. Kupka, G. M. Chambers, S. Dyer et al., “International Committee for Monitoring Assisted Reproductive Technology world report: assisted reproductive technology, 2015 and 2016,” Fertil. Steril., vol. 122, no. 5, pp. 875–893, 2024.

[21] T. Gomez, M. Feyeux, J. Boulant et al., “A time-lapse embryo dataset for morphokinetic parameter prediction,” Data Brief, vol. 42, p. 108258, 2022.

[22] “An annotated human blastocyst dataset to benchmark deep learning architectures for in vitro fertilization,” Sci. Data, vol. 10, p. 271, 2023.

[23] “Live-birth prediction of natural-cycle in vitro fertilization using 57,558 linked cycle records: a machine learning perspective,” Front. Endocrinol., vol. 13, p. 838087, 2022.

[24] B. Huang, W. Tan, Z. Li, and L. Jin, “Using deep learning to predict the outcome of live birth from more than 10,000 embryo data,” BMC Pregnancy Childbirth, vol. 22, no. 1, p. 36, 2022.

[25] “Assisted reproductive technology dataset of embryo time-lapse images and clinical data,” medRxiv, 2024.

[26] “FEMI: a foundational model for in vitro fertilization trained on 18 million time-lapse images,” Nat. Commun., vol. 16, 2025.

[27] X. Liu, Z. Chen, and Y. Ji, “Construction of the machine learning-based live birth prediction models for the first in vitro fertilization pregnant women,” BMC Pregnancy Childbirth, vol. 23, no. 1, p. 476, 2023.

[28] A. Goyal, M. Kuchhal, R. Gupta et al., “Machine learning predicts live-birth occurrence before in-vitro fertilization treatment,” Sci. Rep., vol. 10, no. 1, p. 20925, 2020.

[29] K. Fu, Y. Li, H. Lv et al., “Development of a model predicting the outcome of in vitro fertilization cycles by a robust decision tree method,” Front. Endocrinol., vol. 13, p. 877518, 2022.

[30] R. Bai, Y. Wang, Z. Liu et al., “Predictive modeling of pregnancy outcomes utilizing multiple machine learning techniques for in vitro fertilization-embryo transfer,” BMC Pregnancy Childbirth, vol. 25, no. 1, p. 316, 2025.

[31] M. R. Hassan, S. Al-Insaif, M. I. Hossain, and J. Kamruzzaman, “A machine learning approach for prediction of pregnancy outcome following in vitro fertilization treatment,” Neural Comput. Appl., vol. 32, pp. 2283–2297, 2020.

[32] K. Ranjini, A. Suruliandi, and S. P. Raja, “Machine learning techniques for assisted reproductive technology: a review,” J. Circuits Syst. Comput., vol. 29, no. 11, p. 2030010, 2020.

[33] B. Raef and R. Ferdousi, “A review of machine learning approaches in assisted reproductive technologies,” Acta Inform. Med., vol. 27, no. 3, pp. 205–211, 2019.

[34] J. H. Kim, “Current state of artificial intelligence applications in assisted reproductive technology: a narrative review,” Clin. Exp. Reprod. Med., vol. 52, no. 1, pp. 1–12, 2025.

[35] Y. Wu, H. Huang, L. Zhang et al., “Artificial intelligence and assisted reproductive technology: a comprehensive systematic review,” Taiwan. J. Obstet. Gynecol., vol. 64, no. 1, pp. 11–26, 2025.

[36] F. Delestro, J. Chambost, J. Berntsen et al., “A new artificial intelligence system to assist embryo selection across time-lapse incubators,” Hum Reprod., vol. 37, no. Suppl 1, 2022.

[37] “Deep learning pipeline reveals key moments in human embryonic development predictive of live birth after in vitro fertilization,” Biol. Methods Protoc., vol. 9, no. 1, p. bpae052, 2024.

[38] “Deep-learning model for embryo selection using time-lapse imaging of matched high-quality embryos,” Sci. Rep., vol. 15, p. 28013, 2025.

[39] M. A. Riegler, M. H. Stensen, O. Witczak et al., “Artificial intelligence in the fertility clinic: status, pitfalls and possibilities,” Hum. Reprod., vol. 36, no. 9, pp. 2429–2442, 2021.

[40] Y. Hew, B. Kutuka, U. Sarpel et al., “Artificial intelligence in IVF laboratories: elevating outcomes through precision and efficiency,” Biology (Basel), vol. 13, no. 12, p. 988, 2024.

[41] M. Salih, C. Austin, R. R. Warty et al., “Embryo selection through artificial intelligence versus embryologists: a systematic review,” Hum. Reprod. Open, vol. 2023, no. 3, p. hoad031, 2023.

[42] K. Qaderi, L. Shahmoradi, A. Farzaneh Nejad et al., “Artificial intelligence (AI) approaches to male infertility in IVF: a mapping review,” Eur. J. Med. Res., vol. 30, p. 1, 2025.

[43] E. I. Fernandez, A. S. Ferreira, M. H. M. Cecilio et al., “Artificial intelligence in the IVF laboratory: overview through the application of different types of algorithms for the classification of reproductive data,” J. Assist. Reprod. Genet., vol. 37, no. 10, pp. 2359–2376, 2020.

[44] C. Siristatidis, A. Pouliakis, C. Chrelias, and D. Kassanos, “Artificial intelligence in IVF: a need,” Syst. Biol. Reprod. Med., vol. 57, no. 4, pp. 179–185, 2011.

[45] N. Hollmann, S. Muller, L. Purucker¨ et al., “Accurate predictions on small data with a tabular foundation model,” Nature, vol. 637, no. 8045, pp. 319–326, 2025.

[46] “Established machine learning matches tabular foundation models in clinical predictions,” medRxiv, 2026.

[47] F. Zegers-Hochschild, G. D. Adamson, S. Dyer et al., “The International Glossary on Infertility and Fertility Care, 2017,” Hum. Reprod., vol. 32, no. 9, pp. 1786–1801, 2017.

[48] ESHRE Guideline Group on Ovarian Stimulation, “ESHRE guideline: ovarian stimulation for IVF/ICSI: an update in 2025,” Hum. Reprod., 2026.

[49] D. Glujovsky, A. M. Quinteiro Retamar, C. R. Alvarez Sedo et al., “Blastocyst-stage versus cleavage-stage embryo transfer in assisted reproductive technology,” Cochrane Database Syst. Rev., vol. 1, no. 1, p. CD002118, 2026.

[50] “Catalyzing IVF outcome prediction: exploring advanced machine learning paradigms for enhanced success rate prognostication,” Front. Artif. Intell., vol. 7, p. 1392611, 2024.

[51] J. Swain, M. T. VerMilyea, M. Meseguer, and D. Ezcurra, “AI in the treatment of fertility: key considerations,” J. Assist. Reprod. Genet., vol. 37, no. 11, pp. 2817–2824, 2020.

[52] S. Medenica, D. Zivanovic, L. Batkoska et al., “The future is coming: artificial intelligence in the treatment of infertility could improve assisted reproduction outcomes,” Diagnostics (Basel), vol. 12, no. 12, p. 2979, 2022.

[53] E. Guell, “Criteria for implementing artificial intelligence systems in reproductive medicine,” Clin. Exp. Reprod. Med., vol. 50, no. 1, pp. 1– 5, 2023.