# Less is more: error–distance scaling relation for data-efficient kilometer-scale downscaling of extreme heat

Ahmed Marey<sup>1,2</sup>, Henry Lu<sup>2</sup>, Abhishek Gaur<sup>2</sup>, Liangzhu Leon Wang<sup>1,\*</sup>, Sherif Goubran<sup>3</sup>, Malek Aloui<sup>1</sup>, Theodore Potsis<sup>1</sup>, Alex Hernandez-Garcia<sup>4,5</sup>, David Rolnick<sup>4,6</sup>

<sup>1</sup> Centre for Zero Energy Building Studies, Department of Building, Civil and Environmental Engineering, Concordia University, Montreal, H3G 1M8 Canada

<sup>2</sup> Building and Climate Interface, Construction Research Centre, National Research Council Canada, Ottawa, ON, K1A 0R6, Canada

<sup>3</sup> Department of Architecture, School of Sciences and Engineering, The American University in Cairo, New Cairo 11835, Egypt

<sup>4</sup> Mila – Quebec Artificial Intelligence Institute, Montreal, QC, H2S 3H1, Canada

<sup>5</sup> Department of Computer Science and Operations Research, Université de Montréal, Montreal, QC, H3C 3J7, Canada

<sup>6</sup> School of Computer Science, McGill University, Montreal, QC, H3A 0G4, Canada

Author to whom correspondence should be addressed: leon.wang@concordia.ca

## Abstract

Extreme heat is where urban adaptation needs kilometer-scale data the most, but the simulations training a downscaler can cost more than they save, and how much is needed has not been identified. We measured it with CASPER, a U-Net with a structure-preserving loss downscaling 32 km reanalysis to 1 km temperature, humidity and wind, across 24 configurations of one to eight months. Held-out error grows linearly with climatological distance to the training data, RMSE = 0.83 + 2.95 d, explaining 90% of its variance against 7% for volume and predicting unseen months in advance. On held-out extreme summer weeks CASPER preserves the fine-scale structure and cross-variable physics that matched-budget baselines degrade, and matches station observations during documented heat waves to within 1.8 K. Transfer to a new region degrades geographically; 11 days of local simulation cuts Vancouver's held-out error from 3.8 to 1.3 K. Training periods should span the target climate: the same accuracy for four times less simulation, putting kilometer-scale downscaling of extreme heat within reach of groups without large computing facilities.

## Introduction

Kilometer-scale atmospheric data is indispensable for urban climate adaptation [1], hydrological forecasting [2] and predicting extreme weather events [3]. The demand is highest for extreme heat, whose impacts fall on particular hours and locations within a city and are invisible in coarse-resolution fields; heat waves are the case this work is built around, with typical summer conditions kept as the control. Climate-resilient infrastructure must also be evaluated against detailed long-term projections spanning divergent warming pathways [4]. Dynamical downscaling with regional climate models such as the Weather Research and Forecasting (WRF) model resolves the interactions among topography, the land surface and atmospheric dynamics by solving the governing equations of atmospheric motion at high resolution [5], but the computational overhead is prohibitive [6]: producing a single month of training data for this study's domain required 5 days of computation on a supercomputer with 480 CPU cores, which puts large-scale scenario analysis and ensemble projections for uncertainty quantification out of reach [7].

Statistical downscaling infers empirical relationships between coarse-scale atmospheric patterns and local weather conditions [8]. Quantile mapping [9] and empirical regression [10] cannot capture the nonlinear dynamics and complex spatial dependencies of high-resolution atmospheric phenomena [11], and fail during extreme events, when linear assumptions break down and spatial structure dominates local impacts [12]. Machine learning has narrowed that gap [13], [14]: random forests [15], convolutional networks [16] and, in particular, U-Net architectures learn nonlinear mappings with a much better accuracy [17], while generative adversarial networks [18], [19] and diffusion models [20], [21] produce sharp, realistic finescale variability. Each family carries a cost. Adversarial training is prone to mode collapse, which often limits accurate representation of the distributional tails that define extreme events [22], [23].

Diffusion models train more stably but consume more data, and they treat bias correction as a separate stage: CorrDiff [24] adds a diffusion head to a deterministic regression backbone, and TAUDiff [25] applies quantile mapping as input preprocessing, rather than enforcing distributional agreement during training. Both generative families need large, diverse training sets – a conditional Wasserstein GAN for operational wind downscaling over Canada required a full year of paired forecasts [26]. The binding constraint is therefore not architecture but the generation of high-resolution training data, and it is amplified at kilometer scale. Published models are trained on continuous records of three to fifty years [6], [24], [27], and no public pre-computed paired dataset exists at 1 km, unlike coarser resolutions (10–25 km) at which reanalysis or global model output can serve as both input and target. Every group must generate its own simulations from scratch – a substantial barrier for meteorological services, especially in developing nations [28], for municipal governments and for groups without highperformance computing. Table 1 surveys what is reported: records from one year to nearly three decades, and no study reporting how skill would change if the record were shortened, which periods carry the information, or how much simulation a new deployment would need.

Table 1: Training-data records reported in deep-learning downscaling. High-resolution training record used by representative deep-learning downscaling studies, ordered by target resolution; the ratio is the linear resolution enhancement from the coarse input to the target grid. The final column records whether the study reports a data-volume sensitivity analysis – an explicit measurement of how skill depends on the amount, or the choice, of training data. None of the surveyed studies reports one: the length of the training record is stated but not justified.
<table><tr><td>Study</td><td>Resolutio n (ratio)</td><td>Domain</td><td>High-resolution training record</td><td>Model family</td><td>Data requireme nt measured</td></tr><tr><td>Mardani al. [24]</td><td>et 25 → 2 km (12×)</td><td>Taiwan</td><td>3 years hourly (24,154 fields, 2018-2020)</td><td>U-Net corrective diffusion</td><td>十 No</td></tr><tr><td>Tomasi al. [42]</td><td>et 16 → 2.2 Italy km (8×)</td><td></td><td>15 years hourly (of a 2000-2020 archive)</td><td>Latent diffusion</td><td>No</td></tr></table>

<table><tr><td>al. [26]</td><td>Guevara et 20 → 2.5 Canada km (8×)</td><td></td><td>12 months of paired Conditional forecasts forecast hours)</td><td>(7,150 WGAN-GP</td><td>No</td></tr><tr><td>Pérez et al. 25 → 5.5 Europe [52]</td><td>km (4.5×)</td><td></td><td>29 years 3-hourly Swin (1985-2013)</td><td>transformer</td><td>No</td></tr><tr><td>Jha et al. [27]</td><td>250 → 25 km (10×)</td><td>South Asia</td><td>31 years </td><td>Residual CNN</td><td>No</td></tr><tr><td>Singh et al. [30]</td><td>10 km → Austin, 300 m (33×)</td><td>USA</td><td>9 years daily (2001- Iterative 2009)</td><td>SRCNN</td><td>No</td></tr><tr><td>Chajaei and Bagheri [29]</td><td>100 → 5 Amsterda m (20×)</td><td>m</td><td>1 year (2017)</td><td>Gradient boosting</td><td>No</td></tr><tr><td></td><td>This work 32 → 1 (CASPER) km (32×)</td><td>Montreal- Ottawa, +3 transfer</td><td>1-8 months; 1 U-Net month + 11 days structure- for a deployed preserving regional model</td><td>loss</td><td>+ Yes</td></tr></table>

Four questions follow, and they are the questions any group planning a kilometer-scale simulation campaign for extreme heat has to answer before it starts. How much simulation is enough? How should the simulation periods be selected? How well the resulting model will generalize to conditions and places it has not seen, and can that be known before the computing time is spent? And how much additional simulation does it cost to move the model to a new region? Answering them requires treating the training set as a design variable rather than an inheritance, and it requires a downscaler that trains stably on months rather than years, so that the training set can actually be varied.

We therefore build on the U-Net, the backbone of deep-learning downscaling, in a deterministic configuration [29], and address its main weakness – spatially smoothed output – with a structure-preserving loss. We call the resulting framework CASPER (Context-Aware Structural Prior Enhanced Resolution). It takes high-resolution static geographic features – terrain elevation and land use – as primary input channels: the surface boundary conditions that modulate local atmospheric response through orographic forcing, differential heating and turbulence induced by surface roughness [30], [31], [32]. Combined with the multi-component loss, this design lets the model learn location-specific responses, capturing terrain-forced circulations [33] and urban heat island effects [34] that the 32 km NARR forcing leaves unresolved (Figure 1), without hard physical constraints [35].

![](images/a37a464ddff6fa7c20998208eda6b5fab422f39ca6eba3b4d7bff96bddc52d22.jpg)  
a) How is a model built

![](images/d40315f6284cc641526100dbc7d4726e0dfd733e793f6de8d79b1c604dc9b5a2.jpg)

![](images/b1cc9552c6bb47d1c864acf165dc75c0b9e6de8fea5d31c40df2681c72645dcc.jpg)

$$
\begin{array} { c } { c = \pmb { c } _ { \mathrm { M A E } } + \pmb { c } _ { \mathrm { 9 r a d } } + \pmb { c } _ { \mathrm { M S } } } \\ { + \pmb { c } _ { \mathrm { p a t c h } } + \pmb { c } _ { \mathrm { M B C } } } \end{array}
$$

![](images/706c960c613a8d3c2857c6c765ba9d30d786331b5ff7c0fcefbd654dc18904d5.jpg)

![](images/15af1e0b244382f8b2462ab401bf16d6729ca983e1fe640fa5ff80d77285d471.jpg)

![](images/b4a830754cd926b39c2479603a658a344c3a34b07f8c3dfa737575a5fac596cf.jpg)  
c) What does the framework deliver

![](images/ed956ddc9d1c733fa3cc3fdc7776a8804e4cbc386163ac3e644706dd8835d50a.jpg)

![](images/6e249c485fb050bc958de8fb18aa0ea25559b20f21f37455ad9596efc259e652.jpg)  
Figure 1: The CASPER framework. Panels outlined in maroon are this work's contribution; grey panels are standard components. (a) Candidate simulation periods are placed in climate space and the training months are chosen to span it, before any simulation is run. The network takes the 32 km reanalysis state as 53 coarse channels together with 1 km terrain elevation and land-use class, and returns the 1 km column: temperature, wind and humidity on 10 pressure levels plus 2 m and skin temperature. (b) The backbone is a conventional U-Net; the structurepreserving loss is what is added to it, acting during training rather than in a correction stage afterwards. Its terms hold the prediction to the target in three respects, shown for 2 m temperature: sharp gradients along a transect, variance across spatial scales, and the joint distribution with a second variable. A U-Net trained with an L1 loss alone, shown for contrast, smooths the gradients and loses fine-scale variance. (c) Left, held-out error rises with the climatological distance between an evaluation month and the training distribution, so error is governed by which months were simulated rather than by how many. Right, transfer to an unseen region – the 1 km target, the model applied with no local simulation, and the model after a short local window – with the error against the length of that window directly below them; the open marker is the no-simulation level.

Two design choices carry the framework, and Figure 1 marks them. First, where probabilistic approaches preserve spatial gradients and statistical properties by sampling [36], a structurepreserving loss reaches the same objectives deterministically: alongside the topographic forcing of the static input channels [37], CASPER is trained with a composite of gradient penalties [38], multiscale structural similarity [39], patch coherence [40] and multivariate bias correction [41], so sharp features and the joint distribution are enforced during training rather than restored by a separate post-processing stage, recovering at small budgets the distributional fidelity usually sought through sampling [42]. Second, most downscalers are univariate, which neglects the physical relationships between variables [43], [44] – thermal wind balance [45], Clausius-Clapeyron moisture-temperature coupling [46] – and risks meteorologically inconsistent fields. CASPER downscales the atmospheric column jointly: temperature, both wind components and relative humidity across 10 pressure levels (970-105 hPa), with 2 m and skin temperature, shared representations preserving consistency across variables [47]. Together these let a deterministic model trained on a few months retain the structure and cross-variable physics that make a measurement of the data requirement meaningful.

With the framework fixed, we use CASPER as an instrument rather than a product: because the architecture, loss and normalization never change across configurations, varying the training months alone isolates what each additional month buys. We train on budgets of one to eight months drawn from a 40-year climatology to span the joint distribution of near-surface temperature and relative humidity, and evaluate each configuration on months it never saw; the deployed model is then tested on held-out summer weeks that include the hottest of the 40-year record, and against station observations during documented heat waves. Where previous studies train on multi-year continuous records [24], [48], we ask which months carry the information and how far a model can be pushed before it fails – geographically as well as temporally, applying the Montreal-Ottawa model to Vancouver, Calgary and Toronto.

We answer all four. The first is the central contribution; the others follow from it.

How much data is enough – a distance, not a number of months. Held-out error is set by the climatological distance between an evaluation target and the training distribution rather than by the volume of training data, and adding months that do not move the training distribution toward the conditions of interest leaves the error where it was. At a matched budget CASPER retains the fine-scale structure and cross-variable physics that interpolation, random-forest, U-Net and adversarial baselines degrade – the property that makes the measurement meaningful.

How data should be selected – by statistical coverage, not by convention. Candidate simulation periods can be placed in a climate space computed from coarse reanalysis alone, ranked, and chosen to span the conditions of interest before any high-resolution simulation is committed. Temperature errors follow a thermodynamic distance and the winds a nearly orthogonal circulation distance, which turns the choice of months into a measurable design decision rather than a convention.

How generalization can be measured and predicted. The relation between distance and error is tight enough, and stable enough on months the relation never saw, to predict the error of an unseen climate state within a stated uncertainty. The expected accuracy of a planned simulation campaign can therefore be quoted, with an interval, before the first hour of computing time is spent.

How regional adaptation can be achieved efficiently. Zero-shot transfer to an unseen region degrades along a geographic axis that the climatological distance does not measure, and a short window of local simulation recovers most of that loss. The total simulation behind a deployed model can therefore be several times smaller than current practice assumes.

These results should be read in two parts. The data-selection framework is architectureindependent: placing candidate simulation periods in a climate space computed from coarse reanalysis, choosing them to span the target conditions, measuring held-out error against climatological distance, and re-measuring the coefficients for a new domain require only coarse input fields and a held-out score. Nothing in the procedure depends on the downscaler being CASPER, and it applies equally to a generative model or to a different pair of resolutions. What is specific to CASPER is the deterministic design and the structure-preserving loss: they set how low the baseline requirement can go, because a deterministic backbone trains stably at budgets where adversarial and score-based objectives do not and enforcing distributional agreement inside the loss removes the post-hoc correction stage generative pipelines need, but they do not change how the requirement is measured. The coefficients are specific too: measured for one domain, one reanalysis and one configuration, so what transfers is the procedure, not the constants.

## Results

## Error scaling with climatological distance

We selected candidate training periods from 40 years (1980–2020) of the North American Regional Reanalysis (NARR) [49], computing the domain-mean 2 m temperature and relative humidity of every three-hourly field over the exact model domain. These two variables define the climate space in which we place every training set and every evaluation target. From that space we drew eight candidate months, two per season, spanning the joint distribution over the whole year (Supplementary Figure S1). Holding architecture, loss function and normalization fixed across every configuration ensures that differences in skill reflect the training data alone. We built the model set on four principles. A budget ladder from one to eight months varies training volume. Stratified compositions at a fixed budget vary which months enter the training set while holding volume constant. Randomly drawn subsets, fixed before we examined any result, guard against post-hoc selection. A targeted ladder withholds both January months while the budget grows from four to six, populating the high-budget, high-distance regime that separates the effect of volume from the effect of climatological distance. In total the set comprises 24 configurations, of which the 23 that each withhold at least one month provide the held-out (model, month) pairs behind the scaling relation; Supplementary Tables S1 and S2 list every configuration with its design arm, month composition and evaluation periods.

Across the model set, held-out error depends far more on which months a model saw than on how many it saw (Figure 2). The clearest case needs no statistics: adding a second July to a July-trained model doubles its training data and leaves its error on January unchanged, at 26.7 and 26.6 K, because the second July widens the training distribution without moving it toward the target. Adding two May months to the two Julys instead reaches 14.3 K on the same target – a 12 K improvement for the same doubling of data.

![](images/5d7e56390d27b4687b0f915941f386a14f348180842d0fa435c8b1c54bd9c143.jpg)  
Figure 2: Held-out error is set by climatological distance, not by training budget. Each point is one (model, held-out month) pair (n = 112), placed at the model's training budget in months and coloured by the climatological distance between that month and the model's training distribution. Within each budget the points are sorted by distance and spread evenly across the column, so horizontal position encodes distance ordering only and carries no units; the dark tick marks the median at that budget. The same low-to-high distance spread recurs at every budget, so distance, not volume, organizes the error: distance explains 90% of the variance in held-out error against 7% for training budget. Medians at the sparsest budgets (6, 2 and 2 pairs at five, six and seven months) reflect which months were withheld rather than the budget itself. The full variance-explained comparison is given in Supplementary Fig. S2.

We designed the targeted ladder to test this directly. Holding out both January months while the budget grows from four to six months keeps the target far from the training distribution as volume increases. Held-out January error stays near 12 to 13 K across the ladder, 13.2 K at four months, 11.5 to 12.7 K at five and 12.2 K at six, while the distance to the training climate stays near or above four throughout. Adding months that do not move the training distribution toward the target does not reduce the error.

Held-out 2 m temperature RMSE grows linearly with the distance d between the target month's climate and the centre of the model's training distribution (Figure 3a), following RMSE = 0.83 + 2.95 d across 112 model–month pairs from the 23 configurations that each withheld at least one month, with a coefficient of determination of 0.90; a bootstrap over whole evaluation months gives a 90% confidence interval of 2.48 to 3.09 K per unit distance for the slope. Here d is measured in the same two-variable climate space used to select the months: it is the separation between the centre of the evaluation month's temperature–humidity cloud and the centre of the training cloud, with each axis scaled by the standard deviation of the training cloud, so a distance of one is one training-distribution standard deviation. Three cases fix the scale. A four-month model evaluated on the withheld May 1984 sits at ${ \mathrm { d } } = 0 . 1 1$ and scores 2.8 K; a six-month model evaluated on a January withheld from it sits at d = 4.7 and scores 13.0 K; a model trained on July 2008 alone and evaluated on that same January sits at ${ \mathrm { d } } = 1 1 . 0$ and scores 29.1 K. The larger budget is the worse model whenever the target lies further away. Bilinear interpolation of the coarse field degrades far more gently with distance (median 3.1

K, 5.2 K for the farthest January), so an out-of-coverage neural downscaler is worse than interpolation beyond a distance of about 0.8 and a coverage-selected model is better below it (Supplementary Figure S3): the large errors along the relation are extrapolation collapse, not the intrinsic difficulty of cold months.

![](images/d823b2a2af6dcaceb282bdcf32d6ab28c1fc9abba286612229a5ae3dfd58bc86.jpg)

![](images/dc3ea1779277f16c89f2e9a69fcdf0fd536524330e0d9db865b6a8f95b296a49.jpg)  
Figure 3: The error–distance scaling. a) Held-out month error against distance to the training climate, with the fitted relation and its 90% prediction interval; points are coloured by season. b) Leave-one-month-out validation: the relation is refitted with an entire month withheld and used to predict it, with each point carrying the prediction interval the relation itself quotes.

We then tested whether the relationship predicts as well as it fits. Because the 112 pairs come from only eight distinct climate states – an effective sample size of eight, with about half the held-out-error variance lying between months – we refit the relationship with an entire month withheld and use it to predict that month, repeating for all eight (Figure 3b). It retains a leaveone-month-out coefficient of determination of 0.89 on months it never saw, its 90% prediction interval attains 90% coverage overall, and the slope holds within every season. Its one weakness is the tail: the interval under-covers the January cold-extrapolation regime (78% versus 95% elsewhere, worst held-out error 8 K), so the relation is reliable for interpolation and shoulder-season extrapolation and should be read with an explicit cold-extrapolation caveat. It therefore predicts the error of an unseen climate state and reports how uncertain that prediction is – the property that makes it usable for planning a campaign rather than describing one.

Fitting the same relationship to every output variable separates them into two groups (Supplementary Fig. S4). Temperature errors track the thermodynamic distance closely: 2 m temperature gains 2.95 K per unit distance (coefficient of determination 0.90), skin temperature 2.83 K (0.89) and three-dimensional temperature 1.79 K (0.88). Humidity and wind errors are almost independent of it, with coefficients of 0.07, 0.38 and 0.22. Wind error is not unpredictable, however, but organized by a second axis nearly orthogonal to the first (r = 0.18): adding a circulation distance in domain-mean 544 hPa wind raises the zonal-wind coefficient from 0.38 to 0.72 in sample and from 0.09 to 0.63 under leave-one-month-out cross-validation, with meridional wind improving similarly, while humidity is predicted by neither (Supplementary Figure S5). The coordinate is therefore two-axis – thermodynamic for temperature, circulation for the winds – and the paper's headline relationship is its thermodynamic component.

## Matched-budget validation of the framework

We evaluate CASPER against multiple baselines on the extreme test set – four held-out warmseason weeks: the hottest–driest and hottest–wettest of the 40-year record, together with the two coolest weeks of the warm season that bracket the summer range (Supplementary Table S2) – with typical summer weeks as the control. Importantly, all baseline models were trained on the identical eight-month dataset, isolating the effect of architecture and loss from that of data selection; it is also where the case for a deterministic backbone is tested, since generative models are expected to struggle at such a small budget. Table 2 reports quadratic interpolation, random forest, a U-Net with L1 loss only, a conditional GAN and CASPER across all atmospheric variables on both test sets.

Table 2: Matched-budget performance comparison. All methods were trained on the identical eight-month dataset and scored on the same test splits, so differences reflect method rather than training data. The U-Net L1 baseline is architecturally identical to CASPER but trained with an L1 loss only and without the static geographic features. Values are root-meansquare error in physical units on the extreme and typical test sets, for 2 m temperature (T2, K), skin temperature (TSK, K), three-dimensional atmospheric temperature (T, K), relative humidity (RH, %), and the zonal and meridional wind components (U, V, m/s). Lowest error in each column is best; the lowest value per variable and test split is set in bold.
<table><tr><td>Model</td><td>Test</td><td> $T _ { 2 } ( K )$ </td><td> $T _ { s k } ( K )$ </td><td> $T \left( K \right)$ </td><td>RH (%)</td><td> $U ( m / s )$ </td><td> $V ( m / s )$ </td></tr><tr><td>Quadratic Interpolation</td><td>Extreme</td><td>2.58</td><td>4.70</td><td>2.66</td><td>8.15</td><td>4.02</td><td>4.75</td></tr><tr><td></td><td>Typical</td><td>2.25</td><td>4.00</td><td>2.20</td><td>7.52</td><td>3.42</td><td>3.45</td></tr><tr><td>Random Forest</td><td>Extreme</td><td>2.69</td><td>4.26</td><td>2.58</td><td>7.18</td><td>4.04</td><td>4.80</td></tr><tr><td></td><td>Typical</td><td>2.05</td><td>3.85</td><td>2.14</td><td>6.63</td><td>3.55</td><td>3.60</td></tr><tr><td rowspan="3">U-Net L1</td><td>Extreme</td><td>2.66</td><td>4.23</td><td>2.53</td><td>6.79</td><td>3.90</td><td>4.74</td></tr><tr><td>Typical</td><td>2.05</td><td>3.75</td><td>2.03</td><td>6.91</td><td>3.32</td><td>3.52</td></tr><tr><td>Extreme</td><td>2.41</td><td>4.11</td><td>2.74</td><td>6.82</td><td>4.04</td><td>4.74</td></tr><tr><td>GAN</td><td>Typical</td><td>1.92</td><td>3.63</td><td>2.41</td><td>7.02</td><td>3.54</td><td>3.60</td></tr><tr><td rowspan="3">CASPER</td><td>Extreme</td><td>2.30</td><td>3.96</td><td>2.47</td><td>6.91</td><td>3.90</td><td>4.71</td></tr><tr><td>Typical</td><td>1.86</td><td>3.51</td><td>2.16</td><td>6.76</td><td>3.35</td><td>3.46</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

The published downscaling models of Table 1 are typically trained on continuous records of three to fifty years [21], [24], [48], [50], [51]; none of them was retrained on our domain and budget, so we make no parity claim against them. Every baseline in Table 2, by contrast, we trained ourselves on the same eight months as CASPER: quadratic interpolation, random forest, an L1-only U-Net and a conditional GAN; a conditional diffusion baseline trained on those same months did not converge to physically valid fields and is discussed below rather than tabulated. The GAN converged but carries the signature of adversarial training at a small budget – low mean errors alongside spurious high-frequency spectral energy – consistent with the large-data appetite of generative downscalers. Since U-Net L1 optimizes mean errors alone, its RMSE is expected to match or better CASPER's, so spatial accuracy is assessed instead from the temporal mean across all six output variables over the 363×390 km domain (Figure 4 for the extreme test set; the typical test set is in Supplementary Fig. S6 and a mid-tropospheric level of the extreme set in Supplementary Fig. S7). Our results, along with Supplementary Note 3, cover both extreme and typical conditions, and our conclusions are drawn from both.

![](images/98a407630969022276b23994b336b55a12b0989e3b41ae49954ab0f5358e2245.jpg)  
Figure 4: Spatial prediction maps for the extreme test set (temporal mean over its 670 warm-season hours). Columns correspond to NARR input, WRF training target, U-Net with L1 loss only, the conditional GAN, and CASPER predictions (left to right). Rows show the six predicted variables – temperature, zonal and meridional wind, and relative humidity at 970 hPa, then 2 m temperature and skin temperature (top to bottom) – with the root-mean-square error and bias of each model against WRF annotated. The 363x390 km domain covers the Montreal-Ottawa region at 1 km resolution. The same comparison for the typical test set is Supplementary Figure S6, and a mid-tropospheric level of the extreme test set is Supplementary Figure S7.

CASPER reproduces fine-scale features including the urban heat islands over Montreal and Ottawa that the L1 U-Net smooths away (a 1-2 K enhancement on the typical test set, Supplementary Figure S6), sharp temperature gradients along the St. Lawrence River Valley, and terrain-driven wind channeling. The L1 baseline in Table 2 omits both the multicomponent loss and the static geographic inputs, so we separated the two: the surfacetemperature gain is carried mostly by the static features (0.17 of the 0.19 K reduction on the typical set), whereas the fine-scale structural advantage is the loss – an L1 U-Net with identical static inputs still has a mean spectral error of 0.45 against CASPER's 0.27 (Supplementary Note 5.8). Spatial error patterns show a slight negative bias in extreme conditions and a slight positive bias in typical ones, both addressable at post-processing.

The eight-month budget does not cost the model fine-scale structure. On the extreme test set, power spectral analysis shows CASPER recovering variance across the resolved scales with a mean spectral error of 0.32, against 1.05 for an L1-trained U-Net, 0.44 for random forest and 1.07 for quadratic interpolation (Figure 5; the typical test set is shown in Supplementary Figure S8, where the corresponding values are 0.27, 0.98, 0.44 and 1.00). The conditional GAN reaches a lower mean, 0.29 on both test sets, and stays closer to the target than CASPER at the finest scales of 2 m and skin temperature, where CASPER loses variance; its spectra carry instead the spurious high-frequency energy that adversarial training introduces, visible as the upturn at the highest wavenumbers in every panel. The decisive range lies between 1 and 100 km, where the model must generate variability absent from the 32 km input; there CASPER reproduces the atmospheric energy cascade with spectral slopes that match WRF. Distributional agreement follows the same pattern, holding into the tails that represent heat waves and the coolest summer weeks (Supplementary Figures S9 and S10).

![](images/c9175aa7406543301e83b910c1522579121fd7619753e4b70dd29f7d98c443cf.jpg)  
Figure 5: Spatial structure validation via power spectral analysis. Two-dimensional power spectral density S(k) as a function of wavenumber k for all six predicted variables (temperature, U-wind, V-wind, relative humidity, 2 m temperature, and skin temperature), as the mean over the extreme test set; the typical test set is shown in Supplementary Figure S8. Spectra are plotted premultiplied and variance-normalized, k S(k) $/ \sigma ^ { 2 } .$ , where $\sigma ^ { 2 }$ is the resolved variance of the WRF target for that variable; the same constant scales every curve in a panel, so the six panels share one dimensionless axis, the ${ \bf k } ^ { - 5 / 3 }$ slope is flattened enough for the curves to separate, and the annotated spectral errors, computed in log space, are unchanged by the normalization. Each panel compares CASPER (red) against the WRF target (black), quadratic interpolation (blue), Random Forest (orange), U-Net with L1-only loss (green) and the conditional GAN (purple). Lower normalized spectral error, annotated per panel, indicates

better preservation of the atmospheric energy cascade across the mesoscale (10–100 km) and fine (1–10 km) ranges that downscaling must reconstruct.

The predicted variables retain the physical relationships between them. The 2 m temperature– humidity distribution follows the training target rather than admitting spurious combinations such as cold, saturated air, and the wind components retain the near-circular joint distribution expected of isotropic flow statistics (Supplementary Figures S11 and S12). Through the atmospheric column the model reproduces boundary-layer structure, including the 850–700 hPa jet and the mid-tropospheric moisture minimum, with the smallest errors in the midtroposphere where reanalysis constraints are strongest (Supplementary Figures S13 and S14). These relationships are not imposed as constraints; they follow from predicting the whole column in one network under a loss whose multivariate term acts on the joint distribution. We did not train a separate network per variable, so the single network and the multivariate loss term are not separated experimentally here; the loss term itself is varied from zero to dominant across the ablation configurations of Supplementary Table S6.

## Case studies and model application

Spatial and temporal means can conceal errors that cancel over time, so we examined two events end to end, pairing a single hour with the mean over the whole event (Figure 6). In the climatologically median hour of the typical event the 32 km input carries a single smooth warm anomaly, while the model resolves the St Lawrence valley, the Laurentian uplands and the urban areas of Montreal and Ottawa as distinct thermal structures, reproducing the WRF field to 1.52 K. During the hottest hour of the extreme test set the input reaches 33.5 °C as a nearuniform slab and the model recovers a field agreeing with WRF to 1.48 K. Averaged over the 119 hours of the typical event and the 168 of the extreme, the errors fall to 0.84 and 1.15 K, so hour-to-hour errors partly cancel rather than accumulate. The residuals concentrate along shorelines and steep terrain, where a 1 km field changes fastest.

Two heat events from the extreme test set are examined this way, each by the technique it suits: 12–18 August 2002 spatially, field by field against the 1 km WRF target, and 4–10 August 2003 temporally, hour by hour against station observations independent of the WRF fields the model was trained on. Both belong to the pre-specified test set – the hottest–driest and hottest– wettest weeks of the 40-year record. At the four in-domain stations it tracks most closely the model follows the observed diurnal cycle to 1.47–1.65 K with biases within ±0.6 K, and the median across all eleven stations in that window is 1.73 K. Agreement with WRF shows the model reproduces the simulation it was trained to emulate; agreement with instruments tests model and simulation together against the atmosphere itself.

![](images/26c973e910b18debcc1e704bf1d51f6b3533d208e6832d7ab6fd134a1787310b.jpg)  
Case-study validation against independent weather stations: CASPER vs observed 2 m temperature during the extreme heat events

![](images/2cc94358d125b0eae773cc4b84b8722b2418d0cc030a885642ee5323b8d822df.jpg)

![](images/2b434f7e8b434d2200ba4d0889b4324e7deff529c9390addeb280665d90f9433.jpg)

![](images/34290beeb2198c623602cc2f7b510b3237b3d7ae6820758d3952eec877d3852c.jpg)  
ECCC observationsCASPER

![](images/ce4ef3f6f72fea36e56716aaeaffe412333f90f9fec60880438ceca274fc7d70.jpg)  
Figure 6: Case studies - spatial fields and validation against independent weather stations. Top: 2 m temperature for a typical event (first two rows, centred on 28 July 2014) and an extreme event (last two rows, centred on 14 August 2002); for each event the upper row shows a single hour and the lower row the mean over the whole event. Columns show the 32 km NARR input, the 1 km WRF target, the CASPER prediction and their difference, with the domain root-mean-square error annotated. Bottom: hourly 2 m temperature from CASPER (blue) against Environment and Climate Change Canada station observations (black) at four in-domain stations during the August 2003 heat event, an independent window outside the training months; the per-station root-mean-square error and bias are annotated and CASPER reproduces the observed diurnal cycle.

The scaling relation is built on held-out months within one domain, so transferring the model to a new region tests it against a second axis of dissimilarity: geography. We applied the eightmonth model without retraining to Vancouver, Calgary and Toronto, three Canadian cities spanning Pacific coastal mountains, a prairie–mountain transition and a Great Lakes continental climate (Figure 7). Because these domains differ from the training region in terrain and land cover as well as climate, they probe the limit of what a climatological distance can anticipate.

(a) Vancouver · Jul 2016 simulation · held-out sample  
![](images/b8fe91cda577e40cce1ed5767b8c18ec3ad96f800d17e54b2de52e97e90461f2.jpg)

![](images/93866e4f7d4430b22a882020445ecf75920640a778f789ce0af261a5dd99d5b4.jpg)

(b) Calgary · May 1983 simulation · held-out sample  
![](images/edae1f237b014fb5c75a53534057f2831ea828ba76dccb05ab8c4445e38173dc.jpg)

![](images/9c0b90d89fe306b2027be9e8d4ae16dc789fb4002cabdd7ff70fa29223331220.jpg)

![](images/8403ea5a62a19d6bbf0da5702e5a44ea8b16079132566a7ca0f2072624c4192e.jpg)  
(c) Toronto · Jul 2016 · held-out tail (48-sample in-regime adaptation)

![](images/be08b4c783e170c0c74b4ea3094336a61faab63c5b2abfc4df0f4034635b93fa.jpg)

(d) Held-out error against the guided fine-tuning budget  
![](images/064ee93f2bf8bfa0f6443d81072a73fba4ea2f1a249bfa2347d341478de8829d.jpg)  
fine-tuning samples, coverage-selected (hourly)

Figure 7: Transfer by decoder adaptation to three out-of-domain cities. Each panel is a held-out sample of that city's own 1 km simulation: (a) Vancouver (July 2016), (b) Calgary (May 1983) and (c) Toronto (July 2016). Within each panel the rows are 2 m temperature and skin temperature and the columns are the 32 km NARR input, the WRF 1 km ground truth, the eight-month model applied zero-shot, and the same model after decoder-only fine-tuning on 32 samples and on the full window (one month for Vancouver and Calgary; few-shot), with per-panel root-mean-square error and bias in degrees Celsius. Zero-shot prediction is strongly cold-biased over unseen terrain; local adaptation – one month for Vancouver and Calgary, the 48-sample window for Toronto – recovers the fine-scale structure, reducing 2 m temperature error from 6.6 to 1.4 K in Vancouver, 6.1 to 1.6 K in Calgary and 3.1 to 1.8 K in Toronto. Under each city's maps, hourly 2 m temperature from the zero-shot and few-shot models is compared against independent Environment and Climate Change Canada station observations at the stations each adapted model tracks best – during the June 2021 western heat dome for Vancouver and Calgary, and over the July 2016 simulation window for Toronto; adaptation removes most of the zero-shot cold bias and follows the observed diurnal heat peaks. Panel (d) shows, for Vancouver and Calgary, the held-out 2 m temperature error, averaged over all heldout samples, against the fine-tuning budget when samples are chosen by guided coverage selection (farthest-point sampling over the target pool's climate features); the random-prefix ladder is drawn as a light reference, zero-shot as dashed lines, and stars mark N\*, the smallest budget within 0.25 K of the full-window result, beyond which additional samples buy little; Toronto's 48-sample window is too short to resolve a plateau. The remaining 970 hPa fields and the June 2021 heat-dome maps are given in the Supplementary Information (Supplementary Figs. S15 and S16).

The three domains differ in how far they sit from the training region on the geographic axis, and that ordering decides how much adaptation buys. Measured on the static fields the model ingests, Toronto is closest – 0.72 times the training elevation range, two unseen land-use classes – then Calgary and Vancouver at 2.68 and 3.78 times the range with five and seven unseen classes. One month of local adaptation accordingly helps least where the domain is closest: Toronto's 2 m temperature error falls only from 3.1 to 1.8 K, against 6.1 to 1.6 K for Calgary and 6.6 to 1.4 K for Vancouver. The ordering is geographic rather than climatological – Vancouver is the closest of the three in climate space yet transfers worst – the same dissociation the station comparison shows, and part of the residual traces to unseen land-use classes such as wooded tundra. That argues for selecting training domains to span land cover and topography as we select months to span climate; removing terrain and land use degrades finescale reconstruction sharply, so explicit topographic encoding is what lets the model adapt at all. Degradation of this kind, recovered by local adaptation, is consistent with reports for neural downscalers evaluated outside their training domains [52].

We tested the model against independent station observations rather than against WRF, using two documented heat events across three city domains: the July 2010 Quebec heat wave, which killed 280 people, and the June 2021 western heat dome, which killed 619 in British Columbia and 66 in Alberta [53]. The 2021 event postdates every training month; the 2010 event lies outside the training months but within their span. Driving the model with public reanalysis alone, we compared 2 m temperature against Environment and Climate Change Canada records. Within the training domain no adaptation is needed: Montreal station temperatures are reproduced to 1.77 K mean absolute error across 11 stations. Applied unchanged to Vancouver and Calgary it degrades to 5.30 and 5.35 K (Supplementary Figure S16). That degradation is not ordered by climatological distance – the model lies at d = 0.33 from Vancouver and d = 1.75 from Calgary, yet the two errors are indistinguishable – so the transfer gap lies on a geographic axis the climate distance does not measure. Toronto, which the heat dome did not reach, was compared over its July 2016 window, where the adapted model tracks the four nearest stations to 1.0–1.3 K (Figure 7c).

A model deployed in a geographically distant region should therefore be adapted, and adaptation is cheap: 11 days of target-region simulation is enough. Its station mean absolute error falls to 3.23 K in Calgary, a 40 % reduction on the un-adapted model, and to 4.40 K in Vancouver, a 17 % reduction (Supplementary Figure S16). The gain concentrates where the zero-shot cold bias is largest: at the four Calgary stations of Figure 7b the error falls from 5.0– 6.6 K to 2.9–3.6 K, whereas the Vancouver stations were already among the best tracked without adaptation and change little. Across four one-month fine-tuning seeds the recovery is stable, 0.98 ± 0.04 K held-out for both cities against 3.8 and 3.6 K zero-shot, so it is not a favourable-seed artifact. The budget behind the 11-day figure can also be planned: selecting samples by coverage of the target pool's climate features, mean held-out error falls from 2.1 K at 8 samples to 1.3 K at 256 in Vancouver and from 2.0 to 1.1 K in Calgary – within a quarter kelvin of the one-month result, beyond which additional samples buy little (Figure 7d). Those

256 samples are 11 days' worth of hourly fields, so converting the budget into simulation time assumes they are run as short contiguous blocks. Toronto, already the closest domain, sits at that point from the start: no budget below its full 48-sample window improves materially on zero-shot (2.5 to 2.2 K). What adaptation cannot supply is the extreme itself: the residual error concentrates in the hottest hours, which a fine-tuning month of ordinary conditions does not contain.

The distance that governs training does not govern fine-tuning. When samples are added to an already-trained model, selecting them to minimise distance to the target – the choice the relation appears to recommend – produced the worst overall error of any strategy we tested, in every configuration and across three random seeds. The reason is geometric. A training distribution that already holds hundreds of samples cannot be recentred by a few dozen more unless those few are extreme, so minimising the distance pushes the selection past the target and trains the model on its tail. Selecting samples that span the target distribution instead gave the lowest overall error in every configuration; at the smallest budgets the distance-minimizing selection was better on the extremes while remaining worse overall. The practical rule is therefore split in two: use distance to decide which periods to simulate before training, and coverage of the target distribution to decide which samples to add afterwards. The relation predicts what a trained model will do; it does not prescribe what to add to one.

Because a model can be adapted after the fact, the quantity a group pays is not the size of the training set but the total simulation behind the deployed model. Measured that way, a small model with targeted adaptation beats a larger one without it. Adapting the one-month model to the held-out January with 32 selected samples reaches $3 . 9 9 \pm 0 . 4 0 \mathrm { K }$ , against 14.55 K for the four-month model applied directly – a lower error from 1.04 months of simulation than from four. The same holds spatially: one month plus 32 selected Vancouver samples reaches 2.37 K where the eight-month model without adaptation gives 4.00 K, at roughly an eighth of the simulation.

This is possible because the samples are chosen from reanalysis alone: placing candidate hours in the climate space needs only the coarse input fields, so the selection is made before any highresolution simulation is committed. That accounting assumes those hours are produced as short contiguous blocks, since high-resolution fields come from continuous integrations with spinup rather than isolated time steps. The advantage is over an un-adapted model, not over adaptation in general: given the same 32 samples the larger base remains ahead, at 3.16 ± 0.08 K in-domain and 1.87 K at Vancouver. A wider training distribution still sets a lower floor, and where that floor matters the simulation has to be spent up front; where it does not, it can be spent far more sparingly than current practice assumes.

## Discussion

The bottleneck in deploying machine-learning downscaling is the cost of generating highresolution training data, and the field has addressed it by assuming that more simulation is better. We find instead that error is set by how far the target climate lies from the training distribution. This formalizes, for kilometre-scale downscaling, the domain-shift principle that a model's error grows with distributional distance from its training data; what is new is that the distance is computable from reanalysis before any simulation and is calibrated to predict an unseen month's error with a stated interval. Extremes are valuable because they extend that distribution, not because hard examples teach more [54].

The relation is applied as follows. Compute the joint temperature–humidity climatology of the target domain and place the candidate simulation periods and the conditions of interest in that space. Ranking candidate periods by distance costs nothing and needs no simulation because the coordinates come from reanalysis alone; converting a distance into an expected error in kelvin is a further step that requires the local slope and intercept. Where that error is unacceptable, the remedy is to simulate months that span the target, not to simulate more of them. The assessment therefore costs nothing to run before committing supercomputer time, which is exactly when the decision must be made.

The multivariate loss term is what makes this regime workable: where conventional bias correction calibrates region-specific transfer functions after the fact [8], it enforces distributional agreement during training, so a group can train on a short, targeted simulation campaign without building a separate post-processing workflow. Static geographic features are what make transfer possible at all: removing them degrades performance sharply, and the model adapts to a new region by passing that region's terrain and land cover through the same learned response functions. This is why transfer quality tracks geographic resemblance to the training domain, and why selecting training domains for geographic and climatological diversity matters more than extending them in time. Predicting the full column jointly, rather than one surface variable at a time, preserves the relationships between variables: moisture and temperature follow the Clausius–Clapeyron relation [46], and the vertical structure retains realistic lapse rates that single-level methods cannot guarantee.

Spectral energy tracks WRF through the mesoscale, temperature fronts co-locate with moisture discontinuities, and wind shear preserves rotational structure, consistent with quasi-geostrophic theory [55] and frontogenesis [56] – all of it obtained from the training objective alone, without hard physical constraints. Convolutional U-Nets remain well suited to this problem. Transformers [50] and state-space architectures [51] model long-range dependencies more flexibly, but they scale quadratically with sequence length [57] and require more data – a poor trade in a regime where the binding constraint is simulation cost rather than model capacity.

Generative models, including GANs [26] and diffusion models [21], [24], remain the natural route to ensembles and calibrated uncertainty, but they are poorly matched to the small-data regime this paper targets. We tested this directly at matched budget: a conditional diffusion downscaler trained on the identical eight-month dataset failed to converge to a usable model. Its per-step denoising loss decreased, but the reverse-diffusion sampler diverged to temperatures far outside any physical range, so it could not be scored against WRF and is excluded from the matched-budget comparison – the behaviour expected of score-based training, which requires large and diverse datasets for stable sampling. We therefore chose a deterministic model because the mapping is largely deterministic once high-resolution boundary conditions are supplied, and structure-preserving losses recovered the distributional properties usually attributed to sampling, without the separate post-hoc correction stage that current diffusion pipelines require [24], [25].

Several limitations bound these results. The framework is deterministic and produces point predictions, which suits operational use [29] but leaves uncertainty unquantified for individual fields; ensemble extensions would address this but might sacrifice data efficiency [7]. The relation itself rests on eight climate states in a single domain, driven by one reanalysis and one model configuration, so its coefficients should be re-measured rather than transported; what transfers is the procedure, not the constants. Transfer also degrades where climate or terrain fall outside the training range, which meta-learning [58] and few-shot adaptation [59] could mitigate.

This changes what a downscaling project must commit to in advance. Rather than assembling a multi-year archive before training begins, a group can simulate a handful of months chosen to span the conditions it cares about, train in at most a few days on a single consumer GPU, and know beforehand where the resulting model will be reliable – lowering the barrier for services without large computing facilities [28]. Measuring the relationship between training data and error, rather than assuming it, tells a group planning a simulation campaign how much to simulate, which periods to choose, and where the resulting model will fail, before the first hour of computing time is spent.

## Methods

## Problem Formulation and Study Domain

We formulate statistical downscaling as a supervised learning problem that maps coarseresolution atmospheric fields from the North American Regional Reanalysis (NARR) at 32 km horizontal resolution to high-resolution Weather Research and Forecasting (WRF) model simulations at 1 km. The primary study domain encompasses the Montreal and Ottawa regions of Eastern Canada, characterized by complex topography, including the St. Lawrence River Valley, extensive urban development, and the Laurentian Mountains (Supplementary Figure S17a). The downscaled output comprises a 363×390 pixel grid at 1 km resolution. Our model predicts three-dimensional multivariate atmospheric fields including temperature (T), horizontal wind components (U and V), and relative humidity (RH) across 10 vertical pressure levels spanning the troposphere from 970 hPa to 105 hPa, along with two critical surface variables: 2-meter temperature (T2) and skin temperature (TSK). This vertical discretization captures the essential atmospheric structure from the planetary boundary layer to the upper troposphere, with higher vertical resolution near the surface, where gradients are most substantial and meteorological impacts are most significant.

To assess the spatial transferability of the learned downscaling relationships, we evaluated model performance on three additional Canadian regions exhibiting distinct climatological regimes and topographic characteristics: Vancouver on the Pacific coast with complex mountainous terrain and maritime influence (Supplementary Figure S17b), Calgary at the prairie-mountain transition zone subject to dramatic chinook wind events (Supplementary Figure S17c), and Toronto influenced by Great Lakes dynamics and urban heat island effects (Supplementary Figure S17d). These test domains contributed no training data to the zero-shot evaluation, providing a rigorous assessment of zero-shot spatial generalization.

## Data Acquisition and Preprocessing

NARR provides three-hourly atmospheric state estimates on a 32 km Lambert conformal grid covering North America. From this dataset, we extract five three-dimensional variables: zonal wind �, meridional wind �, temperature �, relative humidity ��, and pressure � at 10 selected pressure levels, plus surface pressure $P _ { s f c }$ , yielding 51 dynamic atmospheric channels. Vertical level selection prioritized meteorologically significant pressure surfaces while reducing computational cost, with higher sampling density near the surface (970, 953, 909, 826, 711 hPa) where boundary layer processes dominate, and coarser spacing in the free atmosphere (580, 435, 288, 175, 105 hPa) sufficient for capturing synoptic-scale flow patterns. Two static geographic features derived from WRF preprocessing system geogrid files (terrain elevation H and land use category index LU) are normalized and concatenated as additional input channels, bringing the total input dimensionality to 53. These static features encode crucial surface boundary conditions that modulate local weather through orographic forcing, surface roughness, and thermal properties.

WRF model version 4.3 simulations at 1 km resolution provide the high-resolution training target for training and evaluation (see Supplementary Note 5.2 and Supplementary Table S3 for the complete WRF configuration). From these simulations, we extract four threedimensional variables at the same 10 pressure levels used for input, plus surface variables, yielding 42 output channels. This configuration captures the complete three-dimensional atmospheric state while maintaining computational tractability for the deep learning framework.

The preprocessing pipeline applies bilinear spatial interpolation to NARR variables to upscale from 32 km to 1 km grid spacing using the WRF Preprocessing System (WPS), and nearestneighbor interpolation to the categorical land-use field to preserve discrete classification boundaries. All continuous variables undergo standardization (Equation 1) using statistics computed exclusively from the training split:

$$
\tilde { x } = ( x - \mu _ { t r a i n } ) / \sigma _ { t r a i n }\tag{1}
$$

## Model Architecture

Our architecture employs a U-Net design with multi-head self-attention mechanisms, processing the full $3 6 3 \times 3 9 0$ spatial domain without patching to preserve long-range correlations essential for coherent atmospheric features (Supplementary Figure S18). The network comprises a contracting encoder path that progressively downsamples spatial resolution while increasing feature dimensionality, a bottleneck that incorporates global context via attention, and an expanding decoder path that reconstructs high-resolution predictions via learned upsampling and skip connections from corresponding encoder levels (see Supplementary Note 5.5 and Supplementary Table S4 for complete architectural specifications).

The encoder consists of six downsampling stages. Each encoder stage contains a ResidualBlock that implements two sequential $3 { \times } 3$ convolutions with GroupNorm, SiLU activations, and dropout (rate 0.1), followed by 3×3 strided convolution for spatial downsampling. Residual connections employ 1×1 convolutions when channel dimensions change, enabling gradient flow through the deep architecture. GroupNorm provides batch-size-independent normalization, critical for stable training with our small batch size, with the group count fixed at 32, ensuring 8-32 channels per group across all feature dimensions.

Multi-head self-attention blocks are placed at the four deepest encoder levels and at the bottleneck, but each is gated to act only where the spatial resolution has fallen to 32 pixels or below, so for the $3 6 3 \times 3 9 0$ input attention is active at the two deepest encoder levels and the bottleneck, where quadratic attention is computationally tractable. At these coarse scales, attention mechanisms aggregate global context, which is crucial for representing synoptic-scale atmospheric patterns such as jet stream position, frontal system orientation, and large-scale pressure gradients. The attention computation follows a standard scaled dot-product formulation with query, key, and value projections split across multiple heads to capture diverse long-range dependencies.

The decoder mirrors the encoder structure, with six upsampling stages: each consists of $2 \times$ bilinear upsampling followed by a 3×3 convolution for feature refinement, concatenation of skip connections from the corresponding encoder level, and processing through ResidualBlocks. Skip connections transfer fine-scale spatial information lost during encoding directly to the decoder, enabling reconstruction of sharp gradients and detailed mesoscale features. A final 1×1 convolution projects the 256-channel decoder output to the required 42 output channels representing all predicted atmospheric variables and vertical levels.

To encode absolute spatial position, which is essential for learning location-dependent phenomena such as coastal effects, lake breezes, and orographic precipitation, we concatenate an 8-channel sinusoidal positional encoding to the input. This encoding employs four spatial frequencies $2 ^ { 0 }$ through $\bar { 2 } ^ { 3 }$ , with paired sine and cosine components computed on normalized spatial coordinates $y , x \in [ - 1 , 1 ]$ according to ��� $( 2 ^ { i } \pi y )$ and ��� $( 2 ^ { i } \pi x )$ for frequency index �. The multi-frequency representation captures spatial patterns across scales from domain-wide gradients to local variations.

All convolutional layers use Kaiming initialization which promotes stable gradient magnitudes during early training. The complete architecture contains about 625 million trainable parameters.

## Multi-Component Loss Function

Standard pixel-wise regression losses such as L1 or L2 produce spatially blurred predictions with systematic misalignment of sharp features – a fundamental limitation for meteorological applications where steep gradients define frontal boundaries, convergence zones, and orographic wind acceleration. To address this, we developed a multi-component loss function (Equation 2) balancing point-wise accuracy with explicit preservation of spatial structure:

$$
L _ { t o t a l } = w _ { I } L _ { M A E } + w _ { 2 } L _ { g r a d } + w _ { 3 } L _ { M S } + w _ { 4 } L _ { p a t c h } + w _ { 5 } L _ { M B C }\tag{2}
$$

The mean absolute error term provides baseline point-wise accuracy (Equation 3):

$$
L _ { M A E } = \frac { I } { N } \Sigma \mid \hat { y } - y \mid\tag{3}
$$

where $\hat { y }$ and $y$ denote predicted and target values, respectively, � is the total number of elements across all variables and atmospheric pressure levels. We employ L1 (absolute error) rather than L2 (squared error) to be more robust to outliers inherent in extreme weather events, as the absolute error metric penalizes large deviations linearly rather than quadratically, reducing sensitivity to rare but physically important extremes.

The gradient loss penalizes differences in spatial derivatives to preserve sharp features. We compute spatial gradients in x and y directions using Sobel operators, yielding gradient components. $\partial \hat { y } / \partial x , \partial \hat { y } / \partial y$ and gradient magnitude $\mid \nabla \hat { y } \mid = \sqrt { ( \partial \hat { y } / \partial x ) ^ { 2 } + ( \partial \hat { y } / \partial y ) ^ { 2 } }$ . The gradient loss employs smooth L1 loss (Huber loss) for robustness (Equation 4):

$$
L _ { g r a d } = \frac { I } { 3 } \bigg ( L _ { H u b e r } \bigg ( \frac { \partial \widehat { y } } { \partial x } , \frac { \partial y } { \partial x } \bigg ) + L _ { H u b e r } \bigg ( \frac { \partial \widehat { y } } { \partial y } , \frac { \partial y } { \partial y } \bigg ) + L _ { H u b e r } ( \mid \nabla \widehat { y } \mid , \mid \nabla y \mid ) \bigg )\tag{4}
$$

This formulation preserves both directional gradient information and edge strength, which are critical for weather downscaling, where atmospheric phenomena manifest as spatial gradients: temperature fronts, convergence-divergence patterns, and topographically induced wind acceleration.

The multi-scale loss ensures pattern consistency across spatial scales by computing structural similarity at four resolutions: the native 1 km grid and the half-, quarter- and eighth-resolution fields obtained by $2 \times , 4 \times$ and $8 \times$ average pooling. The four scales carry weights $w _ { \mathrm { s } }$ of 0.4, 0.3, 0.2 and 0.1, so the native resolution carries the largest weight. At each scale, we compute the structural similarity index (Equation 5):

$$
S S I M _ { s } ( \hat { y } , y ) = \frac { \left( 2 \mu _ { \hat { y } } \mu _ { y } + C _ { I } \right) ( 2 \sigma _ { \hat { y } y } + C _ { 2 } ) } { \left( \mu _ { \hat { y } } ^ { 2 } + \mu _ { y } ^ { 2 } + C _ { I } \right) ( \sigma _ { \hat { y } } ^ { 2 } + \sigma _ { y } ^ { 2 } + C _ { 2 } ) }\tag{5}
$$

where $\mu , \sigma ,$ and $\sigma _ { \hat { y } y }$ denote local means, standard deviations, and covariance computed over 11×11 Gaussian windows, with stability constants $C _ { I } = 0 . 0 I ^ { 2 } { \mathrm { a n d } } C _ { 2 } = 0 . 0 3 ^ { 2 }$ . The multi-scale loss aggregates across scales (Equation 6):

$$
L _ { M S } = I - \sum _ { s = I } ^ { 4 } w _ { s } S S I M _ { s }\tag{6}
$$

This multi-resolution approach ensures both large-scale synoptic patterns (100+ km) and finescale mesoscale features (1-10 km) are accurately reproduced.

The patch loss addresses global spatial misalignment by enforcing local coherence. We randomly sample 32 patches of $1 6 \times 1 6$ pixels per training sample and compute normalized correlation within each patch, averaged across all patches. This local coherence constraint prevents the model from producing globally shifted predictions while allowing regional flexibility, a common failure mode in dense spatial regression tasks.

$K = 1 2 0 0 N = 4 2 Y _ { p r e d } , Y _ { t r u e } \in R { ^ { K \times N } \mathrm { T h } } { }$ multivariate bias correction term preserves the physical relationships among variables by combining energy distance and quantile mapping. We randomly sample 1,200 spatial locations uniformly across the domain and extract, separately for each of the ten pressure levels and for the two surface variables, the corresponding channels at those points, forming sample matrices. The energy distance component measures multivariate distributional similarity through pairwise sample distances (Equation 7):

$$
\begin{array} { c } { { D _ { E } ^ { 2 } = \displaystyle \frac { 2 } { K ^ { 2 } } \sum _ { i , j } \left\| y _ { p r e d , i } - y _ { t r u e , j } \right\| - \displaystyle \frac { 1 } { K ^ { 2 } } \sum _ { i , j } \left\| y _ { p r e d , i } - y _ { p r e d , j } \right\| } } \\ { { - \displaystyle \frac { 1 } { K ^ { 2 } } \sum _ { i , j } \left\| y _ { t r u e , i } - y _ { t r u e , j } \right\| } } \end{array}\tag{7}
$$

where ∥⋅∥ denotes Euclidean distance in the �-dimensional variable space. This metric captures multivariate correlation structure while being computationally tractable.

The quantile mapping component ensures distributional matching for each variable independently. For each output channel, we compute the prediction and target quantiles at 99 evenly spaced levels from 0.01 to 0.99, then minimize the L1 distance between them, averaged across channels and quantile levels. The complete MBC loss combines these components (Equation 8):

$$
L _ { M B C } = w _ { e n e r g y } D _ { E } ^ { 2 } + w _ { q m a p } L _ { q m a p }\tag{8}
$$

An ablation study across 15 loss configurations identified optimal weights (see Supplementary Note 5.7 and Supplementary Table S5 for the loss-component weights and complementary implementations, and Supplementary Note 5.8, Supplementary Table S6 and Supplementary Figures S19–S22 for the detailed ablation results). The ablation was carried out with the earlier four-month warm-season training configuration; the selected weights were retained unchanged for all models trained in this work. MAE and multi-scale terms receive moderate weights to balance spatial quality with point-wise accuracy, while patch loss and MBC use smaller weights to avoid over-constraining local predictions and unstable training. The gradient loss was the highest to preserve sharp gradients in the atmospheric fields. These weights remain fixed throughout training rather than employing progressive schedules, simplifying the training procedure.

## Training Procedure and Evaluation

We trained the model using the AdamW optimizer with an initial learning rate of $1 0 ^ { - 4 } ;$ , weight decay of $1 0 ^ { - 4 }$ , and a cosine annealing schedule with a minimum learning rate of $1 0 ^ { - 6 }$ (see Supplementary Note 5.9 and Supplementary Table S7 for all hyperparameters). Limited by GPU memory constraints, we used a batch size of 2 with gradient accumulation over 2 steps to achieve an effective batch size of 4. Gradient clipping with a maximum norm of 1.0 prevented training instabilities caused by extreme weather outliers in the data.

$I { \boldsymbol { 0 } } ^ { - 4 }$ The model was trained for 100 epochs, requiring approximately 72 hours on a single NVIDIA RTX 3090 GPU with 24 GB of memory. Early stopping monitored validation loss with a patience of 15 epochs and a minimum improvement threshold of $1 0 ^ { - 4 }$ . Training and validation loss remain close at every training budget (Supplementary Figure S23). We selected the best model based on the minimum validation loss, with checkpoints saved every 5 epochs to protect against hardware failures. We did not use mixed-precision training due to numerical stability concerns with the gradient-based loss components at our small batch size.

We evaluated model quality using multiple complementary metrics targeting different aspects of downscaling performance. We quantified point-wise accuracy through mean absolute error and root mean square error per variable (Equations 9 and 10):

$$
M A E = \frac { I } { N } \sum \mid \hat { y } - y \mid , R M S E = \sqrt { \frac { I } { N } \Sigma ( \hat { y } - y ) ^ { 2 } }\tag{9, 10}
$$

�(�, �)Spatial structure was assessed through a two-dimensional power spectral density computed via Fast Fourier Transform. For a 2D field y(x, y), the power spectrum quantifies energy distribution across spatial wavenumbers (Equation 11):

$$
P ( k ) = \mid { \cal F } \{ y \} ( k ) \mid ^ { 2 }\tag{11}
$$

where � denotes the 2D Fourier transform, and � is the wavenumber magnitude. We quantified spectral fidelity between predicted and target spectra through normalized root-mean-square deviation in logarithmic space. This metric captures pattern similarity at different spatial scales corresponding to synoptic versus mesoscale features, with lower values indicating better preservation of multi-scale atmospheric structure. Physical consistency was verified through temperature-humidity and U-V winds joint distributions and gradients, confirming thermodynamic relationships and validating the dynamical balance, along with vertical wind shear profiles assessing a realistic boundary-layer structure.

We compared CASPER against four baseline methods: quadratic interpolation of 32 km NARR fields to 1 km resolution, Random Forest regression with 100 trees trained independently per output channel and per pixel, a standard U-Net trained with L1 loss only, and a conditional GAN of comparable capacity. All baselines used identical training data and evaluation protocols, ensuring fair comparison. We additionally attempted a conditional diffusion baseline, a one-stage denoising-diffusion model with a denoiser of comparable capacity, trained on the identical eight-month dataset; its reverse-diffusion sampling did not converge to physically valid fields at this data budget, so it is documented in Supplementary Note 5.10 rather than included in Table 1. To assess spatial generalization, we applied the Montreal-Ottawa-trained model without modification to the Vancouver, Calgary, and Toronto domains – a stringent zero-shot test across diverse geographic regions and climatic zones.

## Training data selection and the climate feature space

We characterized the domain climate from 40 years of NARR (1980–2020) [60]. For every three-hourly field we computed the domain-mean 2 m temperature and relative humidity over the exact 1 km model grid, taking the mean across all grid points of the model domain rather than over a bounding box, so that the statistic matches the quantity the network actually receives. These two variables define the feature space used throughout: every training set is a cloud of points in it, and every evaluation target is another cloud. We chose temperature and humidity because they are available in any reanalysis product, which keeps the procedure reproducible for any region.

The eight months selected on this criterion are January 1981, May 1984, January 1999, September 1999, July 2002, May 2005, July 2008 and September 2016. After quality control these yielded 5,902 hourly training samples (Supplementary Table S2). During training we monitored a validation set formed by a random 10% split with a fixed seed, used only for early stopping. Because temporally adjacent hours are correlated, this split is not an independent test; the independent tests are the held-out months, which no configuration saw during training, and the transfer cities. Overfitting could not produce the held-out results: a memorized training set cannot yield an error that falls on a leave-one-month-out-validated scaling relation across 23 independently trained models, nor generalize to a region absent from training. Regularization during training comprised dropout of 0.1, weight decay of 10⁻⁴, and early stopping.

Distance in this space is the standardized-Euclidean distance between the centre of the evaluation target's cloud and the centre of the training cloud, with each axis scaled by the standard deviation of the training cloud. Scaling by the training cloud rather than by the climatology is deliberate: it expresses distance in units of what the model has actually seen, so a wide training distribution is correctly treated as closer to any given target than a narrow one. We compared this metric by cross-validation against alternatives, including Mahalanobis distance, nearest-neighbour distances and distances measured beyond the edge of the training support under several definitions of that support (Supplementary Note 5.3).

## Design of the model set

The model set was built to separate the effect of training volume from the effect of climatological distance, which are otherwise confounded because larger training sets are usually also closer to any given target. Four principles govern it. A budget ladder of nested subsets, in which each model's months are a superset of the previous model's, varies volume from one to eight months. Stratified compositions at fixed budget vary which months enter while holding volume constant. A group of randomly drawn subsets, generated from a fixed seed before any results were examined, guards against post-hoc selection of favourable compositions. Finally, a targeted ladder withholds both January months while the budget grows from four to six, populating the high-budget, high-distance region of the design that is otherwise empty and that alone distinguishes a volume effect from a distance effect. Every configuration shares the same architecture, loss function, optimizer, schedule and normalization statistics, so differences in skill are attributable to the training data alone.

## Fitting and validating the error–distance scaling

For each trained model we evaluated every month absent from its training set, giving one (model, month) pair per evaluation, and regressed the resulting root-mean-square error on distance by ordinary least squares. Because the pairs are drawn from only eight distinct climate states, they are not independent: several models share an evaluation month, so a conventional coefficient of determination overstates how well the relationship predicts a new climate state. We therefore validate by blocking on month, refitting the relation with all pairs from one month withheld and using it to predict them, repeating for each of the eight months in turn. We report the resulting out-of-sample coefficient of determination alongside the in-sample value. We also report calibration: for each withheld month we compute the 90% prediction interval implied by the month-held-out fit and record the fraction of withheld months whose observed error falls inside it. A relationship that predicts accurately but understates its own uncertainty would be unsafe to use for planning, so both quantities are necessary.

## Observational validation

We validated the model against station observations rather than against WRF for two documented heat events spanning three city domains: 4–10 July 2010 in Quebec, and 25–30

June 2021 in British Columbia and Alberta. The 2021 event postdates every training month; the 2010 event falls outside the training months but within their span. For each event we downloaded public NARR fields, generated boundary conditions with the WRF Preprocessing System, and ran the model in inference mode without any WRF target. Hourly 2 m temperature records came from the Environment and Climate Change Canada historical climate archive. We retained only stations lying within the model domain and within 3 km of a grid cell centre, deduplicated records across overlapping city queries, aligned observation and model times in UTC, and sampled the predicted field at each station's coordinates. This left 11 stations in Montreal, 17 in Vancouver and 20 in Calgary. We report mean absolute error, bias and Pearson correlation per station and average across stations. Station measurements represent a point while a model value represents a 1 km cell mean, so a residual representativeness error is unavoidable and is not removed; and because the model emulates WRF rather than the atmosphere, the station comparison additionally reflects WRF's own near-surface bias, making agreement with observations a stricter test than agreement with WRF. We also quantified how far each target region lies from the training domain on the geographic axis, using only the static fields the network ingests. For each domain we compared the spread of normalized terrain elevation with that of the training domain and counted the land-use classes present in the target but absent from training. This measure is independent of the climatological distance used for the error–distance scaling and of the station records used to evaluate skill, so the three lines of evidence are not circular.

## Few-shot adaptation and selection experiments

To test how cheaply a trained model can be adapted to a new target, we fine-tuned it on a small number of samples drawn from the target region or month. For the transfer figure we additionally fine-tuned the eight-month model along two budget ladders: random prefixes of the seed-0 permutation (8-128 samples), and a guided ladder whose samples are chosen by farthest-point (coverage) sampling over the pool's domain-mean temperature and humidity, at budgets of 8-256 samples. Every ladder draws from the same pool as the one-month fine-tune, so all budgets remain disjoint from the held-out evaluation samples; the plateau budget N\* is the smallest whose held-out error lies within 0.25 K of the full-window result. The encoder path was frozen and only the decoder path was updated, which limits the number of free parameters and suits the small sample counts involved; optimization used Adam at a learning rate of 1e-4 for at most 30 epochs, with early stopping on a validation split disjoint from both the fine-tuning samples and the test split.

To ask which samples should be selected, we compared four strategies for choosing N finetuning samples from a candidate pool: random selection; coverage, which spans the pool by farthest-point sampling; nearest, which takes the samples closest to the target distribution centre; and a distance-minimizing strategy that greedily selects whichever sample most reduces the climatological distance between the augmented training distribution and the target. Each configuration was repeated for three random seeds, and every configuration shared one fixed test split, evaluated once at the end. We report error over the whole test split and over its most distant fifth, the part of the target the model is extrapolating into. Full results are given in Supplementary Note 5.4 and Supplementary Table S8.

## Data Availability

NARR reanalysis data are publicly available from NOAA/NCEP through the NCAR Research Data Archive at https://rda.ucar.edu/datasets/ds608.0/. WRF simulation data and preprocessed training/test datasets will be made available upon request. The eight training months and the 40-year climatological analysis used to select them are documented in the paper. Station observations are from the Environment and Climate Change Canada historical climate archive.

## Code Availability

Source code for model architecture, training procedures, multi-component loss implementation, and evaluation scripts is available at https://github.com/UMBE-LAB/CASPER-Context-Aware-Structural-Prior-Enhanced-Resolution. Trained model weights and inference code will be released upon publication. The documentation includes instructions for reproducing training across the eight selected months and for applying the model to new geographic regions.

## Competing interests

The authors declare no competing interests.

## Acknowledgements

This research was supported by the Natural Sciences and Engineering Research Council of Canada (NSERC) Discovery Grants Program [RGPIN-2024-06297], the Fonds de recherche du Québec – Nature et technologies (FRQNT) Doctoral Research Scholarships Program, the Canada First Research Excellence Fund (CFREF) IMPACT Project on “Transforming Built and Urban microclimates: Advancing Resilience Science for Vulnerable Populations in a Decarbonized and Electrified Canada,” and the Canada First Research Excellence Fund (Volt-Age) SEED project on “Creating Electrified and Decarbonized Healthy Urban Microclimate around Building Clusters through Climate-Resilient Solutions.” We thank the Digital Research Alliance of Canada for providing the computational resources to run the Weather Research and Forecasting (WRF) simulations.

## Authors Contributions

All authors contributed to the writing of the manuscript. A.M. proposed the idea with L.W., led the project, managed experiments and prepared the codebase. H.L. and A.G. generated the training data, prepared the preprocessed data as well as provided weather and climate domain expertise. L.W. secured funding, facilitated connections with partners in addition to providing guidance and supervision on the scientific implementation. S.G. led the editing and revisions and ensured scientific validity. M.A. and T.P analyzed the results, led the literature review process and assisted on revisions. A.HG. and D.R. assisted in the experiments, ablation studies and scaling as well as supervised the models training and evaluation, providing machine learning and artificial intelligence domain expertise.

## References

[1] A. G. De Lima Moraes and S. Khoshnood Motlagh, “The Climate Data for Adaptation and Vulnerability Assessments and the Spatial Interactions Downscaling Method,” Sci Data, vol. 11, no. 1, p. 1157, Oct. 2024, doi: 10.1038/s41597-024-03995-6.

[2] R. O. Imhoff, J. Buitink, W. J. Van Verseveld, and A. H. Weerts, “A fast high resolution distributed hydrological model for forecasting, climate scenarios and digital twin applications using wflow\_sbm,” Environmental Modelling & Software, vol. 179, p. 106099, Aug. 2024, doi: 10.1016/j.envsoft.2024.106099.

[3] A. Marey, L. L. Wang, A. Gaur, H. Lu, S. Leroyer, and S. Belair, “Urban climate simulation for extreme heat events – A comparison between WRF and GEM,” Urban Climate, vol. 63, p. 102570, 2025, doi: https://doi.org/10.1016/j.uclim.2025.102570.

[4] Intergovernmental Panel On Climate Change (Ipcc), Climate Change 2021 – The Physical Science Basis: Working Group I Contribution to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change, 1st ed. Cambridge University Press, 2023. doi: 10.1017/9781009157896.

[5] W. C. Skamarock et al., “A Description of the Advanced Research WRF Version 4,” 2021.

[6] N. Rampal et al., “Enhancing Regional Climate Downscaling through Advances in Machine Learning,” Artificial Intelligence for the Earth Systems, vol. 3, no. 2, p. 230066, Apr. 2024, doi: 10.1175/AIES-D-23-0066.1.

[7] T. Palmer, “The ECMWF ensemble prediction system: Looking back (more than) 25 years and projecting forward 25 years,” Quart J Royal Meteoro Soc, vol. 145, no. S1, pp. 12– 24, Sep. 2019, doi: 10.1002/qj.3383.

[8] D. Maraun and M. Widmann, Statistical Downscaling and Bias Correction for Climate Research. Cambridge University Press, 2018.

[9] A. J. Cannon, S. R. Sobie, and T. Q. Murdock, “Bias Correction of GCM Precipitation by Quantile Mapping,” Journal of Climate, vol. 28, no. 17, pp. 6938–6959, 2015.

[10] A. Busuioc, “Empirical-Statistical Downscaling: Nonlinear Statistical Downscaling.” Oxford University Press, Aug. 2021. doi: 10.1093/acrefore/9780190228620.013.770.

[11] G. Camps-Valls et al., “Artificial intelligence for modeling and understanding extreme weather and climate events,” Nat Commun, vol. 16, no. 1, p. 1919, Feb. 2025, doi: 10.1038/s41467-025-56573-8.

[12] V. Thandlam, A. Rutgersson, and E. Sahlée, “Structural uncertainty in mapping Euro-Atlantic atmospheric rivers obscures understanding of associated meteorological extremes,” Sci Rep, vol. 15, no. 1, p. 33325, Sep. 2025, doi: 10.1038/s41598-025-19685-1.

[13] S. Rasp, M. S. Pritchard, and P. Gentine, “Deep learning to represent subgrid processes in climate models,” Proc. Natl. Acad. Sci. U.S.A., vol. 115, no. 39, pp. 9684–9689, Sep. 2018, doi: 10.1073/pnas.1810286115.

[14] M. Reichstein et al., “Deep learning and process understanding for data-driven Earth system science,” Nature, vol. 566, no. 7743, pp. 195–204, Feb. 2019, doi: 10.1038/s41586- 019-0912-1.

[15] C. Hutengs and M. Vohland, “Downscaling land surface temperatures at regional scales with random forest regression,” Remote Sensing of Environment, vol. 178, pp. 127–141, Jun. 2016, doi: 10.1016/j.rse.2016.03.006.

[16] A. Prasad et al., “Evaluating the transferability potential of deep learning models for climate downscaling,” Jul. 17, 2024, arXiv: arXiv:2407.12517. doi: 10.48550/arXiv.2407.12517.

[17] G. Ascenso, A. Ficchì, M. Giuliani, E. Scoccimarro, and A. Castelletti, “Downscaling, bias correction, and spatial adjustment of extreme tropical cyclone rainfall in ERA5 using deep

learning,” Weather and Climate Extremes, vol. 46, p. 100724, Dec. 2024, doi: 10.1016/j.wace.2024.100724.

[18] B. Kumar et al., “On the modern deep learning approaches for precipitation downscaling,” Earth Sci Inform, vol. 16, no. 2, pp. 1459–1472, Jun. 2023, doi: 10.1007/s12145-023-00970-4.

[19] X. Hong, L. Hu, and A. Kareem, “A tropical cyclone intensity prediction model using conditional generative adversarial network,” Journal of Wind Engineering and Industrial Aerodynamics, vol. 240, p. 105515, Sep. 2023, doi: 10.1016/j.jweia.2023.105515.

[20] J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” Dec. 16, 2020, arXiv: arXiv:2006.11239. doi: 10.48550/arXiv.2006.11239.

[21] R. A. Watt and L. A. Mansfield, “Generative Diffusion-based Downscaling for Climate,” Apr. 27, 2024, arXiv: arXiv:2404.17752. doi: 10.48550/arXiv.2404.17752.

[22] M. Arjovsky and L. Bottou, “Towards Principled Methods for Training Generative Adversarial Networks,” Jan. 17, 2017, arXiv: arXiv:1701.04862. doi: 10.48550/arXiv.1701.04862.

[23] T. Salimans et al., “Improved Techniques for Training GANs”.

[24] M. Mardani et al., “Residual corrective diffusion modeling for km-scale atmospheric downscaling,” Commun Earth Environ, vol. 6, no. 1, p. 124, Feb. 2025, doi: 10.1038/s43247- 025-02042-5.

[25] R. Sundar, Y. Hu, N. Parashar, A. Blanchard, and B. Dodov, “TAUDiff: Highly efficient kilometer-scale downscaling using generative diffusion models,” Mar. 13, 2025, arXiv: arXiv:2412.13627. doi: 10.48550/arXiv.2412.13627.

[26] J. Guevara et al., “Enhancing operational wind downscaling capabilities over Canada: Application of a Conditional Wasserstein GAN methodology,” Feb. 26, 2025, arXiv: arXiv:2412.06958. doi: 10.48550/arXiv.2412.06958.

[27] S. K. Jha, V. Gupta, P. J. Sharma, A. Mishra, and S. Joshi, “Deep learning superresolution for temperature data downscaling: a comprehensive study using residual networks,” Front. Clim., vol. 7, p. 1572428, May 2025, doi: 10.3389/fclim.2025.1572428.

[28] B. Lamptey et al., “Challenges and ways forward for sustainable weather and climate services in Africa,” Nat Commun, vol. 15, no. 1, p. 2664, Mar. 2024, doi: 10.1038/s41467- 024-46742-6.

[29] F. Chajaei and H. Bagheri, “Machine learning framework for high-resolution air temperature downscaling using LiDAR-derived urban morphological features,” Urban Climate, vol. 57, p. 102102, Sep. 2024, doi: 10.1016/j.uclim.2024.102102.

[30] M. Singh et al., “Urban precipitation downscaling using deep learning: a smart city application over Austin, Texas, USA,” Aug. 15, 2022, arXiv: arXiv:2209.06848. doi: 10.48550/arXiv.2209.06848.

[31] G. E. Forsythe, A.C., “A Generalization of the Thermal Wind Equation to Arbitrary Horizontal Flow,” Bulletin of the American Meteorological Society, vol. 26, no. 9, pp. 371– 375, Nov. 1945, doi: 10.1175/1520-0477-26.9.371.

[32] N. J. Lutsko, “The Relative Contributions of Temperature and Moisture to Heat Stress Changes under Warming,” Journal of Climate, vol. 34, no. 3, pp. 901–917, Feb. 2021, doi: 10.1175/JCLI-D-20-0262.1.

[33] K. Chen et al., “The operational medium-range deterministic weather forecasting can be extended beyond a 10-day lead time,” Commun Earth Environ, vol. 6, no. 1, p. 518, Jul. 2025, doi: 10.1038/s43247-025-02502-y.

[34] A. Marey, L. L. Wang, and S. Goubran, “Developing accurate land cover projection to accelerate the realization of SDG 11 in urbanized cities: a comparative study,” Clean Technologies and Environmental Policy, Aug. 2025, doi: 10.1007/s10098-025-03297-4.

[35] R. B. Smith, “The Influence of Mountains on the Atmosphere,” vol. 21, B. Saltzman, Ed., in Advances in Geophysics, vol. 21. , Elsevier, 1979, pp. 87–230. doi: https://doi.org/10.1016/S0065-2687(08)60262-9.

[36] A. Marey et al., “Forecasting Urban Land Use Dynamics Through Patch-Generating Land Use Simulation and Markov Chain Integration: A Multi-Scenario Predictive Framework,” Sustainability, vol. 16, no. 23, p. 10255, Nov. 2024, doi: 10.3390/su162310255.

[37] D. Hao, G. Bisht, L. Li, and L. R. Leung, “Representing fine-scale topographic effects on surface radiation balance in hyper-resolution land surface models,” Feb. 04, 2025, Preprints. doi: 10.22541/essoar.173869500.04138737/v1.

[38] I. A. Assenova, L. L. Vitanova, and D. Petrova-Antonova, “Urban heat islands from multiple perspectives: Trends across disciplines and interrelationships,” Urban Climate, vol. 56, p. 102075, Jul. 2024, doi: 10.1016/j.uclim.2024.102075.

[39] J. González-Abad, Á. Hernández-García, P. Harder, D. Rolnick, and J. M. Gutiérrez, “Multi-Variable Hard Physical Constraints for Climate Model Downscaling,” AAAI-SS, vol. 2, no. 1, pp. 62–67, Jan. 2024, doi: 10.1609/aaaiss.v2i1.27650.

[40] C. Finn, P. Abbeel, and S. Levine, “Model-agnostic meta-learning for fast adaptation of deep networks,” in Proceedings of the 34th International Conference on Machine Learning - Volume 70, in ICML’17. Sydney, NSW, Australia: JMLR.org, 2017, pp. 1126–1135.

[41] W.-Y. Chen, Y.-C. Liu, Z. Kira, Y.-C. F. Wang, and J.-B. Huang, “A Closer Look at Few-shot Classification,” Jan. 12, 2020, arXiv: arXiv:1904.04232. doi: 10.48550/arXiv.1904.04232.

[42] E. Tomasi, G. Franch, and M. Cristoforetti, “Can AI be enabled to perform dynamical downscaling? A latent diffusion model to mimic kilometer-scale COSMO5.0\_CLM9 simulations,” Geosci. Model Dev., vol. 18, no. 6, pp. 2051–2078, Apr. 2025, doi: 10.5194/gmd-18-2051-2025.

[43] J. Chen, T. Janke, F. Steinke, and S. Lerch, “Generative machine learning methods for multivariate ensemble post-processing,” Ann. Appl. Stat., vol. 18, no. 1, Mar. 2024, doi: 10.1214/23-AOAS1784.

[44] A. Marey, J. Zou, S. Goubran, L. L. Wang, and A. Gaur, “Urban morphology impacts on urban microclimate using artificial intelligence – a review,” City and Environment Interactions, vol. 28, p. 100221, Dec. 2025, doi: 10.1016/j.cacint.2025.100221.

[45] L. Ge and L. Dou, “G-Loss: A loss function with gradient information for superresolution,” Optik, vol. 280, p. 170750, Jun. 2023, doi: 10.1016/j.ijleo.2023.170750.

[46] A. Hamplová, T. Novák, M. Žáček, and J. Brožek, “Effects of Normalised SSIM Loss on Super-Resolution Tasks,” CMES, vol. 143, no. 3, pp. 3329–3349, 2025, doi: 10.32604/cmes.2025.066025.

[47] T. An, B. Mao, B. Xue, C. Huo, S. Xiang, and C. Pan, “Patch loss: A generic multiscale perceptual loss for single image super-resolution,” Pattern Recognition, vol. 139, p. 109510, Jul. 2023, doi: 10.1016/j.patcog.2023.109510.

[48] A. J. Cannon, “Multivariate quantile mapping bias correction: an N-dimensional probability density function transform for climate model simulations of multiple variables,” Clim Dyn, vol. 50, no. 1–2, pp. 31–49, Jan. 2018, doi: 10.1007/s00382-017-3580-6.

[49] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-Based Generative Modeling through Stochastic Differential Equations,” Feb. 10, 2021, arXiv: arXiv:2011.13456. doi: 10.48550/arXiv.2011.13456.

[50] A. Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” Feb. 26, 2021, arXiv: arXiv:2103.00020. doi: 10.48550/arXiv.2103.00020.

[51] National Centers for Environmental Prediction, National Weather Service, NOAA, U.S. Department of Commerce, “NCEP North American Regional Reanalysis (NARR).” Research Data Archive at the National Center for Atmospheric Research, Computational and

Information Systems Laboratory, Boulder CO, 2005. [Online]. Available: https://rda.ucar.edu/datasets/ds608.0/

[52] A. Pérez, M. S. Cruz, D. S. Martín, and J. M. Gutiérrez, “Transformer based superresolution downscaling for regional reanalysis: Full domain vs tiling approaches,” Oct. 16, 2024, arXiv: arXiv:2410.12728. doi: 10.48550/arXiv.2410.12728.

[53] Z. Liu et al., “MambaDS: Near-Surface Meteorological Field Downscaling with Topography Constrained Selective State Space Modeling,” Aug. 20, 2024, arXiv: arXiv:2408.10854. doi: 10.48550/arXiv.2408.10854.

[54] A. El-Kabid, L. Benabbou, R. Lguensat, and A. Hernández-García, “Multi-scale Neural PDE Surrogates for Prediction and Downscaling: Application to Ocean Currents,” Oct. 20, 2025, arXiv: arXiv:2507.18067. doi: 10.48550/arXiv.2507.18067.

[55] K. E. Trenberth, J. T. Fasullo, and T. G. Shepherd, “Attribution of climate extreme events,” Nature Clim Change, vol. 5, no. 8, pp. 725–730, Aug. 2015, doi: 10.1038/nclimate2657.

[56] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, “Curriculum learning,” in Proceedings of the 26th Annual International Conference on Machine Learning, Montreal Quebec Canada: ACM, Jun. 2009, pp. 41–48. doi: 10.1145/1553374.1553380.

[57] B. J. Hoskins, I. Draghici, and H. C. Davies, “A new look at the ω-equation,” Quarterly Journal of the Royal Meteorological Society, vol. 104, no. 439, pp. 31–38, 1978, doi: https://doi.org/10.1002/qj.49710443903.

[58] B. J. Hoskins and F. P. Bretherton, “Atmospheric Frontogenesis Models: Mathematical Formulation and Solution,” Journal of Atmospheric Sciences, vol. 29, no. 1, pp. 11–37, 1972, doi: 10.1175/1520-0469(1972)029%3C0011:AFMMFA%3E2.0.CO;2.

[59] Y. Tay, M. Dehghani, D. Bahri, and D. Metzler, “Efficient Transformers: A Survey,” Mar. 14, 2022, arXiv: arXiv:2009.06732. doi: 10.48550/arXiv.2009.06732.

[60] G. Papacharalampous et al., “Global-scale massive feature extraction from monthly hydroclimatic time series: Statistical characterizations, spatial patterns and hydrological similarity,” Science of The Total Environment, vol. 767, p. 144612, May 2021, doi: 10.1016/j.scitotenv.2020.144612.

# Supplementary Information for: Less is more: error–distance scaling relation for data-efficient kilometer-scale downscaling of extreme heat

Ahmed Marey<sup>1,2</sup>, Henry Lu<sup>2</sup>, Abhishek Gaur<sup>2</sup>, Liangzhu Leon Wang<sup>1,\*</sup>, Sherif Goubran<sup>3</sup>, Malek Aloui<sup>1</sup>, Theodore Potsis<sup>1</sup>, Alex Hernandez-Garcia<sup>4,5</sup>, David Rolnick<sup>4,6</sup>

<sup>1</sup> Centre for Zero Energy Building Studies, Department of Building, Civil and Environmental Engineering, Concordia University, Montreal, H3G 1M8 Canada

<sup>2</sup> Building and Climate Interface, Construction Research Centre, National Research Council Canada, Ottawa, ON, K1A 0R6, Canada

<sup>3</sup> Department of Architecture, School of Sciences and Engineering, The American University in Cairo, New Cairo 11835, Egypt

<sup>4</sup> Mila – Quebec Artificial Intelligence Institute, Montreal, QC, H2S 3H1, Canada

<sup>5</sup> Department of Computer Science and Operations Research, Université de Montréal, Montreal, QC, H3C 3J7, Canada

<sup>6</sup> School of Computer Science, McGill University, Montreal, QC, H3A 0G4, Canada

\* Author to whom correspondence should be addressed.

## Supplementary Note 1: Related Work and Context

Statistical downscaling has emerged as a cost-effective alternative to dynamical downscaling for generating high-resolution atmospheric predictions from coarse global model outputs [1]. Early efforts focused on relatively simple architectures such as convolutional neural networks for precipitation downscaling [2] and random forests for temperature interpolation [3]. More recent work has explored deeper architectures, including U-Nets [4], which have proven particularly effective for image-to-image translation tasks in computer vision [5].

Recent studies demonstrate the extent of data requirements for achieving strong performance at large resolution differences. Jha et al. [6] downscale ERA5 2-m temperature using 31 years of training data, with deeper residual models achieving SSIM of 0.96 and PSNR of 34 dB. Rampal et al. [7] downscale daily precipitation over New Zealand from ERA5 to VCSN using 33 years of training, improving explained variance from 0.35 to 0.52. The challenge of data efficiency becomes particularly acute for very high resolution ratios. DeepSD demonstrated 8× spatial-resolution enhancement in precipitation using convolutional networks trained on 25 years of data [8]. Baño-Medina et al. [9] showed similar results for temperature and relative humidity downscaling using 30 years of reanalysis data. Recent work on super-resolution climate downscaling [10] achieved 12× resolution enhancement but required training on continuous 22-year climate reanalysis data. These substantial data requirements present practical limitations for operational weather services and research groups with limited access to high-resolution numerical weather prediction outputs. Similar data efficiency has been demonstrated in urban wind and temperature prediction, where localized training strategies with geometric features enabled accurate 3D predictions [11], [12].

Generative models have recently attracted attention for atmospheric downscaling due to their ability to capture uncertainty and generate realistic fine-scale variability [13]. Generative adversarial networks (GANs) have been applied to precipitation downscaling [14], [15], [16], demonstrating improved spatial structure compared to deterministic models. Diffusion models [17] offer improved training stability and have shown promise for probabilistic weather downscaling [18], [19], [20]. However, both types of models have their limitations related to training instability and/or practical limitations for operational purposes.

Several architectural innovations have emerged to improve the preservation of spatial structure in downscaling applications. Attention mechanisms [21] enable models to capture long-range spatial dependencies relevant for synoptic-scale weather patterns. Stochastic weight averaging [22] and ensemble methods [29] improve the quantification of prediction uncertainty. Multiscale loss functions [23] encourage models to preserve atmospheric variability across the hierarchy of spatial scales from synoptic systems to mesoscale features. However, these techniques have primarily been evaluated on problems with moderate resolution ratios and extensive training datasets.

The present work addresses several gaps in existing literature. First, we measure how a downscaler's held-out error depends on its training data, and show that error follows a predictable error–distance scaling in a two-variable climate space, so which months are simulated governs skill more than how many. This data efficiency stems from recognizing that atmospheric downscaling relationships are largely deterministic, with most variance in highresolution fields explainable by coarse-scale inputs and static terrain features [24], [25], [26]. Second, we introduce a composite loss function that combines spatial gradient preservation, multi-scale structural consistency, local patch coherence, and multivariate distribution, specifically designed for atmospheric applications where sharp meteorological features coexist with smooth synoptic-scale patterns. Third, we validate model performance on a challenging 32:1 resolution ratio (32 km → 1 km) across multiple atmospheric variables and vertical levels, demonstrating robust generalization to both typical conditions and extreme weather events. Finally, we provide a comprehensive physical consistency analysis examining multivariate relationships, vertical structure, and spatial-scale-dependent behavior to ensure meteorologically realistic predictions beyond simple point-wise accuracy metrics.

## Supplementary Note 2: Training-Data Design and the Error–Distance Scaling

## 2.1 Training Data Selection Strategy

Supplementary Figure S1 places the hourly training states within the 40-year climatology of the model domain; Supplementary Table S1 lists the 24 training configurations and their exact month compositions, and Supplementary Table S2 the training and held-out evaluation periods.

Training coverage in 40-year climatology  
![](images/0859027fb0580d67b7a91718abbd82bd1afc5cbcf44c56116fd3003d848b1115.jpg)

![](images/c7a8a65217d706f53ea120f84cc848c9b66d3cadb5f436ca81112fee5b968382.jpg)

![](images/cea173b5ab35c97b73e3b60dc6e0d34e5942c67b13f5eb1d7f67ff6ce0595210.jpg)

Supplementary Figure S1: Training coverage within the 40-year climatology. Hourly training states, coloured by season, and the eight training-month means are shown against the 1980–2020 three-hourly climatology of the model domain (blue density), in the climate space of domain-mean 2 m temperature and relative humidity. Marginal distributions compare the training states with the full climatology on each axis. The eight months span essentially the full 40-year temperature range, while the climatology extends to lower humidity than the training data.

Supplementary Table S1: The 24 training configurations and their exact month compositions. Each row is one independently trained model. The eight candidate months (two per season, spanning the 40-year joint temperature-humidity distribution) are January 1981,

May 1984, January 1999, September 1999, July 2002, May 2005, July 2008 and September 2016.
<table><tr><td>Configuration</td><td>Budget (months)</td><td>Training months</td></tr><tr><td>b1 f</td><td>1</td><td>Sep 1999</td></tr><tr><td>b1_sp</td><td>1</td><td>May 1984</td></tr><tr><td>b1_su</td><td>1</td><td>Jul 2002</td></tr><tr><td>b1_w</td><td>1</td><td>Jan 1981</td></tr><tr><td>sw_k1_a</td><td>1</td><td>Jul 2008</td></tr><tr><td>sw_k1_b</td><td>1</td><td>Sep 2016</td></tr><tr><td>sw_k2_a</td><td>2</td><td>Sep 1999, Sep 2016</td></tr><tr><td>sw_k2_b</td><td>2</td><td>May 1984, May 2005</td></tr><tr><td>b2 cold</td><td>2</td><td>Jan 1981, Jan 1999</td></tr><tr><td>b2_mixed</td><td>2</td><td>Jan 1981, Jul 2002</td></tr><tr><td>b2_warm</td><td>2</td><td>Jul 2002, Jul 2008</td></tr><tr><td>sw_k3_a</td><td>3</td><td>May 1984, Jul 2002, Sep 2016</td></tr><tr><td>sw_k3_b</td><td>3</td><td>Jan 1999, May 2005, Jul 2008</td></tr><tr><td>sw_k4_a</td><td>4</td><td>Jan 1981, May 2005, Jul 2008, Sep 2016</td></tr><tr><td>sw_k4_b</td><td>4</td><td>Jan 1999, Jul 2002, Jul 2008, Sep 2016</td></tr><tr><td>b4_balanced</td><td>4</td><td>Jan 1981, May 1984, Sep 1999, Jul 2002</td></tr><tr><td>b4_cold</td><td>4</td><td>Jan 1981, Jan 1999, Sep 1999, Sep 2016</td></tr><tr><td>b4_warm</td><td>4</td><td>May 1984, Jul 2002, May 2005, Jul 2008</td></tr><tr><td>tc_k5_a</td><td>5</td><td>May 1984, Jul 2002, May 2005, Jul 2008, Sep 2016</td></tr><tr><td>tc_k5_b</td><td>5</td><td>May 1984, Sep 1999, Jul 2002, May 2005, Jul 2008</td></tr><tr><td>tc_k6</td><td>6</td><td>May 1984, Sep 1999, Jul 2002, May 2005, Jul 2008, Sep 2016</td></tr><tr><td>sw_k7_a</td><td>7</td><td>Jan 1981, Jan 1999, Sep 1999, Jul 2002, May 2005, Jul 2008, Sep 2016</td></tr><tr><td>sw_k7_b</td><td>7</td><td>Jan 1981, May 1984, Jan 1999, Jul 2002, May 2005, Jul 2008, Sep 2016</td></tr><tr><td>b8_all</td><td>8</td><td>Jan 1981, May 1984, Jan 1999, Sep 1999, Jul 2002, May 2005, Jul 2008, Sep 2016</td></tr></table>

Supplementary Table S2: Training and testing sets from 40-year climatology.
<table><tr><td>Split</td><td>Category</td><td>Start Date</td><td>End Date</td><td>Type</td></tr><tr><td rowspan="5">Tiai D DTaata Budtng</td><td></td><td>1/1/1981</td><td>1/31/1981</td><td>Winter</td></tr><tr><td></td><td>5/1/1984</td><td>5/31/1984</td><td>Spring</td></tr><tr><td>mo ts pr</td><td>1/1/1999</td><td>1/31/1999</td><td>Winter</td></tr><tr><td>seon</td><td>9/1/1999</td><td>9/30/1999</td><td>Fall</td></tr><tr><td></td><td>7/1/2002</td><td>7/31/2002</td><td>Summer</td></tr><tr><td rowspan="5"></td><td>8</td><td>5/1/2005</td><td>5/31/2005</td><td>Spring</td></tr><tr><td></td><td>7/1/2008</td><td>7/31/2008</td><td>Summer</td></tr><tr><td></td><td>9/1/2016</td><td>9/30/2016</td><td>Fall</td></tr><tr><td></td><td>8/12/2002</td><td>8/18/2002</td><td>Hottest Driest</td></tr><tr><td></td><td>8/4/2003</td><td>8/10/2003</td><td>Hottest Wettest</td></tr><tr><td rowspan="3">Testng</td><td>Exttemme</td><td>6/6/1988</td><td>6/12/1988</td><td>Coldest Driest</td></tr><tr><td></td><td>5/26/2003</td><td>6/1/2003</td><td>Coldest Wettest</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

<table><tr><td rowspan="5">Tycal</td><td>7/28/2014</td><td>8/3/2014</td><td>Typical Summer</td></tr><tr><td>6/26/2000</td><td>7/2/2000</td><td>Typical Summer</td></tr><tr><td>8/18/2008</td><td>8/24/2008</td><td>Typical Summer</td></tr><tr><td>8/20/2001</td><td>8/26/2001</td><td>Typical Summer</td></tr></table>

## 2.2 Predictors of Held-Out Error and the Interpolation Baseline

![](images/c96d2be62977662d6371ed7a7aa23ad3a04fe7bd3c53d1e018deaf6ce1ddbdd1.jpg)

![](images/a21194902c88669f8081e6de7ca1587c653e67a3d6740132ee751de0c8a8c87c.jpg)  
Supplementary Figure S2: Variance of held-out error explained by candidate predictors. (a) For the held-out 2 m temperature error, the fraction of variance explained by training budget, log-budget, and climatological distance. (b) For each of the six output variables, the fraction of error variance explained by T2-RH climatological distance. Distance dominates for temperature and is weak for humidity and wind.

![](images/a75ee057fce8d7101516bced91b5ac27d99d9ae9ec69886fda07042f6784a9a3.jpg)

Supplementary Figure S3: The neural downscaler's large held-out errors are extrapolation collapse, not intrinsic difficulty. CASPER held-out 2 m temperature RMSE per configuration and held-out month (circles) against climatological distance, with the fitted relation and its 90% prediction interval, overlaid with a trivial bilinear interpolation of each target month's 32 km field to 1 km (diamonds, one per month). Interpolation degrades only gently with distance (median 3.1 K, rising to 5.2 K for the farthest January), so an out-ofcoverage neural downscaler exceeds interpolation beyond a distance of about 0.8, whereas a coverage-selected model is more accurate below it. The order-of-magnitude error growth along the relation therefore reflects neural extrapolation out of the training regime rather than the intrinsic difficulty of cold months, and motivates selecting training months to span the target.

## 2.3 Variable-Resolved Error–Distance Scaling

![](images/90e34fbf2404dae957ab720b27a1fc88b5db6d805849d1d92b206f4c498b2ecb.jpg)  
Supplementary Figure S4: The error–distance scaling resolved by output variable. Error added per unit of climatological distance for each of the six predicted variables, in physical units and grouped by unit, with 95% confidence intervals from bootstrap resampling blocked by evaluation month; the coefficient of determination (variance explained by distance) is annotated on each bar. All three temperature variables scale strongly with distance (1.8-3.0 K per unit distance, $\mathrm { R } ^ { 2 } = 0 . 8 8 – 0 . 9 0 )$ , while relative humidity and the wind components are nearly flat $\left( \mathrm { R } ^ { 2 } \leq 0 . 3 8 \right)$ , so the scaling is fundamentally a temperature relation. Thermodynamic and circulation distances as predictors of held-out error

![](images/ec8bc55edb3b68ef8c849378a0b03507fbac29faac028164ba9e27dfff1b5b36.jpg)

![](images/765555dcae411a6b80334d30ed84366de3aa78aa585201b8181170a54a453451.jpg)  
Supplementary Figure S5: A two-axis distance coordinate. Coefficient of determination for held-out error regressed on a thermodynamic distance (standardized-Euclidean separation in

domain-mean 2 m temperature and relative humidity), a circulation distance (the same reduction in domain-mean mid-tropospheric \~544 hPa zonal and meridional wind, nearorthogonal to the thermodynamic distance, $\mathrm { ~ r ~ } = \ 0 . 1 8 )$ and both combined, for the four continuous predicted variables, in-sample (left) and under leave-one-month-out crossvalidation (right). Surface-temperature error follows the thermodynamic axis, wind error the circulation axis (zonal-wind cross-validated R² rises from 0.09 to 0.63 when the circulation axis is added), and humidity error is predicted by neither.

## Supplementary Note 3: Matched-Budget Evaluation

## 3.1 Spatial Performance on Typical and Extreme Conditions

The comparisons in this note cover the deterministic family - quadratic interpolation, random forest, an L1-only U-Net and CASPER - because their purpose is to isolate what the structurepreserving loss and the static geographic inputs add to a U-Net backbone. The conditional GAN is evaluated in the main text (Table 2 and Figures 4 and 5) on the extreme test set, where the relevant question is not architectural enhancement but whether adversarial training is stable at this data budget; Supplementary Figures S6 and S8 repeat those two main-text comparisons, GAN included, for the typical test set. Supplementary Figure S6 presents the mean spatial fields of all six output variables over the typical test set at 970 hPa and the surface, comparing CASPER and the matched-budget baselines against the WRF target.

![](images/0e0b8497285ceff29925f73f47dcf3219606fc7a582c989268af66e322efa424.jpg)  
Supplementary Figure S6: Spatial comparison over the typical test set (temporal mean). Columns show the 32 km NARR input, the 1 km WRF target, the U-Net with L1 loss only, the conditional GAN and CASPER; rows show temperature, zonal and meridional wind and relative humidity at 970 hPa, then 2 m and skin temperature, with the root-mean-square error and bias of each model against WRF annotated. The layout is that of main-text Figure 4, which shows the extreme test set.

Model robustness under extreme atmospheric conditions is assessed using the extreme dataset containing samples representing climatological extremes in temperature and relative humidity. These cases challenge the model’s ability to simulate rare conditions that other models usually struggle to reproduce, highlighting the advantage of the curated training selection (Supplementary Figure S7).

![](images/a1f4a35847538d6753580611bcb9506ee53155e2306fbdb293ba7b2fda3fb0e2.jpg)  
Supplementary Figure S7: Spatial comparison over the extreme test set at a midtropospheric level. Mean fields at output level 6 of 10 (580 hPa) for the three-dimensional variables, together with 2 m and skin temperature. Columns show the 32 km NARR input, the 1 km WRF ground truth, and four of the matched-budget methods (quadratic interpolation, random forest, U-Net L1, CASPER), with RMSE and bias against WRF annotated on each method panel; the three-dimensional temperature is WRF potential temperature.

## 3.2 Power Spectra and Spatial Structure Preservation for Typical Conditions

Preservation of atmospheric variability across spatial scales from synoptic systems (>1000 km) through mesoscale (10-100 km) to fine scales (<10 km) is essential for capturing realistic weather patterns. Supplementary Figure S8 presents power spectra for all six predicted variables over the typical test set, on the same axes as main-text Figure 5.

![](images/e7ff787a54e87e53733a1d5afc6dbdc9e0d7f253ce55b6d87f120d74b2e937ca.jpg)  
Supplementary Figure S8: Power spectra for the typical test set. As in main-text Figure 5, for the typical test set: two-dimensional power spectral density S(k) against wavenumber k for the six output variables, as the mean over the full typical test set, plotted premultiplied and variance-normalized (k S(k) $/ \sigma ^ { 2 } .$ , with $\sigma ^ { 2 }$ the resolved variance of the WRF target). Each panel compares CASPER against the WRF target, quadratic interpolation, random forest, the U-Net with L1-only loss and the conditional GAN, with the normalized spectral error of each method annotated.

## 3.3 Statistical Distribution Fidelity

Supplementary Figures S9, S11 and S13 provide the distributional, multivariate and vertical validation for the typical test set; each is paired below with its extreme-test-set counterpart.

![](images/31f53d4b6a152ac49c6f439ca1ce658e87a845cf3f2f98f372a63933c15db43f.jpg)

![](images/67927e1eb371de6ce3962c5614d757f9852567a16e4fb479ee5eca2b8a41242a.jpg)

![](images/9087cacce25f1b09cc55af1f3e9a00530abcf13de8b4b9085ec07e3adbde1fd5.jpg)

![](images/420089e569c56552709665bdcf83cdcc146bec79bdd72a0a59c9eef49d90ca46.jpg)

![](images/bf7be28e360a8511760ba13c5813cc5faf1608b811e6e4a46fdc119bba6a824e.jpg)

![](images/b98f2e82d9aed6e0318d0ab55c2d39a86dcb5f9fb1892047a2a4d705cc344598.jpg)  
Supplementary Figure S9: Statistical fidelity for the typical test set. Probability density functions (logarithmic scale) for the six output variables, comparing the WRF ground truth with quadratic interpolation, random forest, U-Net L1 and CASPER; the

Kolmogorov–Smirnov statistic of each method is annotated per panel. Distributions here are computed on a 50-sample subset at a single output level, so the annotated statistics are not directly comparable with the full-test-set means quoted in the main text.

Accurate representation of probability distributions across the full range is essential for risk assessment and climate adaptation applications during extreme events. Supplementary Figure S10 examines distributional fidelity through probability density functions.

![](images/b6d1569303ba45eedce12ab0d503ae01a449b8d2206bd8a3a3636caeb82aa265.jpg)

![](images/423c2c43c528226719adbbea28a5b5e8782eb662c60d4da4776db8e51a752528.jpg)

![](images/5f8a09066180f6a65a4e39d65ea29d673e279c326be859aa1e0176a8aafd5d5c.jpg)

![](images/a97309cc16eb9dfc00198650de1995597c77f7725387145c843fda00e90767e1.jpg)

![](images/9b6ad34a158c1e07362fe582c6fe8b2143ff5ab9c797a98429640403b1b262b3.jpg)

![](images/57a2fd44df04bb182f360f6b9c884c1cd754cf82d4725438e44c575889a9dfd4.jpg)  
Supplementary Figure S10: Statistical fidelity for the extreme test set. As in Supplementary Figure S9, for the extreme test set.

Atmospheric variables exhibit coupled relationships, constrained by thermodynamic conditions and dynamical balances that trained models must preserve. Supplementary Figure S11 examines multivariate relationships to validate that the model learns fundamental atmospheric coupling.

## 3.4 Physical Consistency Across Variables and Vertical Levels

![](images/9f7215963f6f7475bc02428ee6902c2500d065a5ff5b65ca7aa4ea3f1db1df5e.jpg)  
Supplementary Figure S11: Multivariate physical consistency for the typical test set. Cross-variable spatial gradients (columns 1 and 3) and joint distributions (columns 2 and 4) for the temperature–humidity and wind–wind pairs: the upper rows compare the WRF ground truth with each matched-budget method, and the lower rows show each method's difference from the WRF density, with the log-density RMSE annotated on every difference panel and the

change in cross-variable correlation on the gradient panels. L5 in the panel titles denotes that same mid-tropospheric level.

![](images/9cdd6c0787bbeb22882e001b12648ee11d3d60d67d74fbfda8489ae92569836e.jpg)  
Supplementary Figure S12: Multivariate physical consistency for the extreme test set. As in Supplementary Figure S11, for the extreme test set.

![](images/e8ff995efd09a1c5408538b2743bc1478eb2d7cd7af114a840588b53680699bf.jpg)  
Supplementary Figure S13: Vertical profile validation for the typical test set. Domainmean profiles of temperature (WRF potential temperature), zonal and meridional wind, and relative humidity across the output pressure levels, for the WRF ground truth and the four matched-budget methods; horizontal bars show the spread across samples and the profile RMSE of each method is annotated.

Atmospheric predictions must also maintain realistic vertical structure across all pressure levels to ensure physical consistency for downstream applications such as radiation calculations and aviation forecasting, especially during extreme events. Supplementary Figure S13 validates predictions across the tropospheric column from surface to upper levels.

![](images/e4eb47e555f3345afa80717cad6cf81f385c3b59407df545f0d3716592625159.jpg)  
Supplementary Figure S14: Vertical profile validation for the extreme test set. As in Supplementary Figure S13, for the extreme test set.

## Supplementary Note 4: Case Studies and Transfer

Supplementary Figures S15 and S16 extend the main-text transfer analysis: Supplementary Figure S15 shows the remaining 970 hPa variables for the three transfer cities, and Supplementary Figure S16 the zero-shot versus few-shot comparison on the June 2021 western heat dome.

(a) Vancouver · Jul 2016 simulation · held-out sample  
![](images/036ed71541d4513646e0de5ae599e01715c4218340b36ebfba8afeabb182eb5b.jpg)

(b) Calgary ·May 1983 simulation · held-out sample  
![](images/db3e05745f6bb7b01d32c042b33dcd6537983af559f0759e229d57d00f2ddc6e.jpg)

(c) Toronto ·Jul 2016 · held-out tail (48-sample in-regime adaptation)  
![](images/df340a0878c5d971a25bf066b5acf690e551c6660c9042e1aeb967370ee1c488.jpg)  
Supplementary Figure S15: Transfer by decoder adaptation, remaining variables. As in the main-text transfer figure but for the 970 hPa temperature, zonal and meridional wind, and relative humidity, for (a) Vancouver, (b) Calgary and (c) Toronto, each on a held-out sample of its own 1 km simulation. Columns are the 32 km NARR input, the WRF 1 km ground truth, CASPER zero-shot, and the same model after decoder-only fine-tuning on 32 samples and on the full adaptation window; per-panel root-mean-square error and bias are annotated.

![](images/e9bf3089cd113fc2c612d3d07b1cb88de8a4d36938ca23a8c80025240289183c.jpg)  
Supplementary Figure S16: Extreme-event transfer - CASPER zero-shot versus few-shot on the June 2021 western heat dome, for Vancouver and Calgary. Columns show the zeroshot 2 m temperature, the few-shot 2 m temperature, and their difference at the peak heat-dome hour; the mean absolute error against independent ECCC station observations is annotated. Adaptation warms the cold-biased zero-shot fields, reducing station error from 5.30 to 4.40 K in Vancouver and 5.35 to 3.23 K in Calgary.

## Supplementary Note 5: Methods

## 5.1 Study Domains

![](images/3ff5ab4a9f65ee2602efca26e103731d820dcb35acf2b2fb3712365ef240dd0c.jpg)

![](images/8433756f7c6969a1d56abff10136d690d200a5f6e9f8ddabb6a74ebe32e7fb8b.jpg)

![](images/2e9e7e0f75e0d34a950d1722bb1fd2929ec6bfac4a1b929f46e1833e54bd0f07.jpg)

![](images/5009ed42c83ed5f77aad3ba77d1c09849117f25b3cc8d720f61bd0fe09b2c995.jpg)  
Supplementary Figure S17: Training and testing domains with validation weather stations. Land use of the four 1 km domains: (a) the Montreal–Ottawa training domain and the (b) Vancouver, (c) Calgary and (d) Toronto transfer domains, with the ECCC stations used for validation marked as triangles — the July 2010 heat wave for the training domain (11 stations), the June 2021 western heat dome for Vancouver and Calgary (stations with resolvable coordinates: 16 of 17 and all 20), and the July 2016 window for Toronto (the 12 stations inside the 3 km co-location gate). The grey line is the 500 m elevation contour; the Toronto grid is the shoreline-georeferenced reconstruction described in Methods.

## 5.2 WRF Model Configuration

High-resolution target fields for model training and evaluation were generated using the Weather Research and Forecasting (WRF) model version 4.3 [27]. WRF simulations were initialized and forced with North American Regional Reanalysis (NARR) data at 32 km horizontal resolution and 3-hourly temporal resolution [28]. The model domain covers the Montreal-Ottawa region with 363×390 grid points at 1 km horizontal spacing. The vertical coordinate system employs 40 pressure levels, which were then reduced to 10 for computational efficiency, spanning approximately 970 hPa near the surface to 105 hPa in the upper troposphere. The simulation follows validated setup in the same study domain [29]. Supplementary Table S3 summarizes the complete physics configuration, and Supplementary Figure S17 shows the training and testing domains of our study area.

Supplementary Table S3: WRF model physics configuration.
<table><tr><td>Component</td><td>Configuration</td></tr><tr><td>Boundary conditions</td><td>NARR 32 km, 3-hourly</td></tr><tr><td>Microphysics</td><td>WRF Single – Moment 3</td></tr><tr><td>Cumulus parameterization</td><td>Kain-Fritsch</td></tr><tr><td>Planetary boundary layer</td><td>BouLac</td></tr><tr><td>Surface layer</td><td>Eta similarity</td></tr><tr><td>Land surface model</td><td>Unified Noah</td></tr><tr><td>Longwave radiation</td><td>RRTM</td></tr><tr><td>Shortwave radiation</td><td>Dudhia</td></tr></table>

## 5.3 Choice of Distance Metric

The error–distance scaling requires a distance between a model's training distribution and an evaluation target in the two-variable climate space. Several definitions are defensible, so we selected one by cross-validation rather than by assumption. Each candidate was used to refit the relation, which was then scored by leave-one-month-out cross-validation, blocking on evaluation month because the model–month pairs derive from only eight distinct climate states. This comparison was run on a pilot set of 45 model-month pairs from eight configurations; the 0.89 quoted in the main text is the same statistic over the full set of 112 pairs. Distance to the centre of the training cloud, standardized per axis by the training cloud's standard deviation, achieved an out-of-sample coefficient of determination of 0.902 on the pilot set. Distances measured beyond the edge of the training support scored higher (0.934 for the mean over target samples, 0.938 for the 90th percentile), and nearest-neighbour distances scored slightly lower (0.887 for the single nearest training sample, 0.892 for the mean of the two nearest). The fraction of the target lying outside the training support, which discards magnitude information, performed markedly worse at 0.657, and a Mahalanobis distance using the training covariance performed worst of all at 0.731, because the sign of the temperature–humidity correlation flips across the eight months, making the pooled covariance unstable to invert.

We retain the centre distance for the reported relation. Although edge-based distances fit marginally better, their fitted intercept implies an error floor at zero distance that is not physically credible, and the centre distance is the quantity a practitioner can compute directly from monthly climatological statistics. Adding a second predictor did not help: combining centre and edge distances improved the in-sample fit but reduced out-of-sample performance, so the single-term relation is retained.

The edge-based definitions also require a definition of the training support itself. We compared a per-axis bounding box, the convex hull, a union of balls whose radius was set from the training cloud's own nearest-neighbour distances, and the highest-density region of a kernel density estimate. The simple bounding box performed best. Tighter definitions classify almost no evaluation target as lying wholly inside the support, because a few of its samples always fall outside, so the distance never reaches zero and degenerates towards a nearest-neighbour distance.

## 5.4 Selection-Strategy Experiment

Held-out January 1981 was the target, with geography held fixed so the only gap between model and target is climatological. Two base models (one and four training months) were finetuned on N samples chosen by each strategy, for three random seeds (Supplementary Table S8). Values are mean ± s.d. over seeds of the 2 m temperature RMSE (K) on the fixed test split, and on its most distant fifth.

## 5.5 Network Architecture

The fundamental architecture follows the U-Net encoder-decoder paradigm [5], with several domain-specific modifications for atmospheric downscaling. The encoder comprises six hierarchical levels with progressively increasing feature dimensionality as detailed in Supplementary Table S4.

Supplementary Table S4: Model architecture configuration parameters. Feature dimensions represent channel counts at each of the six encoder-decoder levels.
<table><tr><td>Component</td><td>Specification</td><td>Value</td></tr><tr><td>Encoder levels</td><td>Number of downsampling stages</td><td>6</td></tr><tr><td>Decoder levels</td><td>Number of upsampling stages</td><td>6</td></tr><tr><td>Base channels</td><td>Initial feature dimensionality</td><td>256</td></tr><tr><td>Channel multipliers</td><td>Per-level expansion factors</td><td>[1, 2, 3, 4, 5, 6]</td></tr><tr><td>Feature dimensions</td><td>Channel counts by level</td><td>[256, 512, 768,</td></tr><tr><td>Attention heads</td><td>Multi-head attention configuration</td><td>1024, 1280, 1536] 8</td></tr><tr><td>Attention levels</td><td>Encoder-decoder stages with attention</td><td>[4, 5, 6]</td></tr><tr><td>Attention threshold</td><td>Maximum spatial dimension for attention</td><td>32×32</td></tr><tr><td>GroupNorm groups</td><td>Normalization group count</td><td>32</td></tr><tr><td>Dropout rate</td><td>Spatial dropout probability</td><td>0.1</td></tr><tr><td>Positional encoding</td><td>Number of frequency octaves</td><td>4</td></tr><tr><td>Total parameters</td><td>Learnable weights (millions)</td><td>625.7</td></tr></table>

Beginning from the 53 atmospheric input channels, the architecture incorporates eight additional channels of sinusoidal positional encoding, yielding an initial 61-channel tensor that enters the first encoder block operating at full 363×390 spatial resolution. The encoder expands features via convolutional operations, with channel counts following the sequence [256, 512, 768, 1024, 1280, 1536], as specified in Supplementary Table S4. Each downsampling operation, a $3 { \times } 3$ convolution with stride 2, halves the spatial resolution while the channel count increases with depth, enabling the network to capture increasingly coarse-scale atmospheric patterns across the spatial hierarchy.

Positional encoding provides explicit spatial information essential for learning positiondependent atmospheric processes such as terrain-forced flows and land-sea contrasts. The implementation employs sinusoidal functions at four frequency octaves. For each octave � ∈ {0,1,2,3}, the model computes frequency $f _ { i } = 2 ^ { i }$ and generates two encoding channels via sin $( f _ { i } \pi y _ { \mathrm { n o r m } } )$ and cos $( f _ { i } \pi x _ { \mathrm { n o r m } } )$ , where $y _ { \mathrm { n o r m } }$ and $x _ { \mathrm { { n o r m } } }$ represent normalized pixel coordinates spanning the interval [-1, 1]. This multi-frequency representation enables learning both largescale geographic patterns captured by low-frequency components and fine-scale positiondependent features represented by high-frequency components, which are crucial for modeling topographic influences and circulation patterns that vary systematically across the domain.

Each encoder and decoder level contains ResidualBlocks [30] that preserve gradient flow through the deep network architecture. A ResidualBlock implements the following computational sequence: (i) 3×3 convolution with padding to maintain spatial dimensions, configured without bias terms since subsequent normalization layers include affine transformations; (ii) GroupNorm with 32 groups providing batch-independent normalization suitable for small batch training [31]; (iii) SiLU activation function [32] defined as $f ( x ) = x$ $\sigma ( x )$ where � represents the sigmoid function; (iv) Dropout2d [33] with rate 0.1 providing spatial regularization by randomly zeroing entire feature maps during training; (v) second 3×3 convolution followed by GroupNorm; and (vi) residual addition with 1×1 projection convolution applied when input and output channel dimensions differ. This design enables training of very deep networks exceeding 100 layers while maintaining stable gradient magnitudes throughout the optimization process.

Attention $( Q , K , V ) = \operatorname { s o f t m a x } ( Q K ^ { T } / \sqrt { d _ { k } } ) V d _ { k }$ Self-attention mechanisms operate at the four deepest encoder and decoder levels and at the bottleneck, where computational costs remain tractable due to reduced spatial dimensions. The attention module activates only when feature map spatial dimensions fall below 32×32 pixels; larger feature maps bypass attention operations to manage GPU memory requirements during training. With eight attention heads, the mechanism computes scaled dot-product attention following where represents the perhead dimension, calculated as the total number of channels divided by 8 [21]. This architectural choice allows the network to aggregate global atmospheric context at coarse spatial scales where long-range teleconnections and large-scale circulation patterns dominate, while relying on convolutional operations for efficient local feature extraction at finer spatial scales where computational costs would otherwise become prohibitive.

The decoder implements a symmetric structure mirroring the encoder through six upsampling stages. Each decoder level begins with 2× bilinear upsampling of features from the previous coarser level, followed by channel-wise concatenation with skip connections originating from the corresponding encoder level. These skip connections represent a fundamental component of U-Net performance, providing the decoder with high-resolution spatial information inevitably lost during the encoding bottleneck compression and enabling sharp reconstruction of fine-scale atmospheric features. The concatenated feature tensors pass through ResidualBlocks structurally identical to encoder blocks but parameterized with independently learned weights. The final decoder level produces 256-channel features at full spatial resolution (363×390 pixels), which a subsequent 1×1 convolution projects to the 42 output channels representing the predicted atmospheric state across all variables and vertical levels.

![](images/dfa397d5da8d431edc4674370470c848cb5fe16187eb6a9984ac8aa315c4b576.jpg)  
Supplementary Figure S18: CASPER model architecture and training design. The sixlevel U-Net encoder-decoder with skip connections (black arrows) maps a 53-channel input to 42 output channels. Orange blocks (ResidualBlocks, detailed at bottom-left) use two sequential 3×3 convolutions with GroupNorm and SiLU activation; purple blocks (multi-head self-

attention, detailed at bottom-right) operate at the three deepest encoder levels and the bottleneck, providing global context for synoptic-scale patterns.

## 5.6 Parameter Budget

The network contains 625.7 million trainable parameters, distributed as follows: decoder path 173.0 M (27.6%), bottleneck 169.9 M (27.2%), encoder path 100.2 M (16.0%), learned upsampling 83.8 M (13.4%), strided-convolution downsampling 53.7 M (8.6%), and encoder and decoder attention 22.6 M each (3.6% each). The bottleneck is a single block operating on 3,072 channels with 8 attention heads, which is why so large a share of the parameters sits at the coarsest resolution.

## 5.7 Loss Function Design

The training objective combines multiple loss components designed to preserve different aspects of atmospheric structure and spatial fidelity. Unlike conventional single-objective losses that optimize exclusively for point-wise accuracy, the composite loss function ensures the model simultaneously learns accurate field values and realistic spatial patterns characteristic of atmospheric flows. The complete loss function specification, including weights and mathematical formulations, appears in Supplementary Table S5.

Supplementary Table S5: Composite loss function weights and component descriptions.
<table><tr><td>Loss Component</td><td>Weight</td><td>Purpose</td></tr><tr><td>L1 Reconstruction</td><td>0.5</td><td>Point-wise accuracy</td></tr><tr><td>Gradient Preservation</td><td>2.0</td><td>Sharp meteorological features</td></tr><tr><td>Multi-scale Structural</td><td>0.5</td><td>Hierarchical spatial consistency</td></tr><tr><td>Patch Coherence</td><td>0.1</td><td>Local smoothness</td></tr><tr><td>Multivariate Bias Correction (MBC)</td><td>0.1</td><td>Multivariate distributional fidelity</td></tr></table>

The L1 formulation provides robustness to outliers while maintaining sensitivity to systematic biases, with a weighting of 0.5 in the composite objective function, as shown in Supplementary Table S5.

Sharp meteorological features such as temperature fronts, wind shear zones, and moisture gradients require explicit penalties on spatial derivatives. The gradient loss component computes spatial derivatives via normalized Sobel operators, implementing 3×3 convolutional kernels defined as:

$$
S _ { x } = \frac { 1 } { 8 } \left[ \begin{array} { c c c } { - 1 } & { 0 } & { 1 } \\ { - 2 } & { 0 } & { 2 } \\ { - 1 } & { 0 } & { 1 } \end{array} \right] , \quad S _ { y } = \frac { 1 } { 8 } \left[ \begin{array} { c c c } { - 1 } & { - 2 } & { - 1 } \\ { 0 } & { 0 } & { 0 } \\ { 1 } & { 2 } & { 1 } \end{array} \right]
$$

For each atmospheric field channel, the model computes x-direction and y-direction spatial derivatives.

To manage GPU memory constraints when processing 42 output channels across large spatial domains, the implementation processes atmospheric variables in chunks of 32 channels, computing gradients independently for each chunk before aggregating results. This chunked processing strategy enables training on consumer-grade GPUs with 24GB memory. The gradient loss receives a weight of 2.0 in the composite objective function.

Atmospheric processes inherently span multiple spatial scales, from synoptic weather systems extending thousands of kilometres to mesoscale features on the order of tens of kilometres. The multi-scale structural loss enforces consistency across this spatial hierarchy by computing structural similarity at four progressively coarser resolutions. Starting from native resolution (363x390 pixels), the model generates three additional scales via 2x2 average pooling: half resolution (181x195), quarter resolution (90x97), and eighth resolution (45x48). The multiscale loss aggregates these scale-specific components with weights [0.4, 0.3, 0.2, 0.1] that progressively de-emphasize coarser scales. This weighting scheme ensures the model preserves both large-scale synoptic circulation patterns and fine-scale terrain-forced features that emerge at native resolution. The multi-scale loss receives a weight of 0.5 in the composite objective function.

Local spatial coherence prevents checkerboard artifacts and ensures smooth transitions between neighboring atmospheric features. The patch loss randomly extracts 32 patches of size 16×16 pixels from each prediction-target pair during training. For each patch, the loss computes normalized correlation between predicted and target fields. When the spatial dimensions fall below 16×16 pixels, the implementation computes global correlation rather than patch-based statistics. This patch loss component, weighted at 0.1 in the composite objective, provides regularization encouraging local spatial consistency without imposing the excessive smoothness characteristic of global correlation penalties.

The Multivariate Bias Correction (MBC) loss [34] enforces statistical fidelity across coupled atmospheric variables at each vertical level. Unlike point-wise losses that treat variables independently, MBC loss preserves the multivariate distributional structure, which is essential for maintaining physical relationships among temperature, wind, and humidity. The MBC loss operates independently at each of the ten pressure levels, computing two complementary metrics that, together, ensure distributional consistency.

The first component employs energy distance [35], a metric for comparing multivariate distributions. This metric quantifies the discrepancy between joint distributions of all atmospheric variables at a given level, capturing correlations and dependencies that univariate metrics cannot detect. To manage computational costs, energy-distance calculations subsample 1,200 spatial points per level.

The second component uses quantile-mapping distance, which measures how well the predicted quantiles align with the target quantiles across the full distribution for each variable. For 99 evenly spaced quantile levels from 0.01 to 0.99, the loss computes the mean absolute difference between the predicted and target quantile values. This ensures the model reproduces not just the mean and variance but the complete distributional shape, including tails representing extreme events, which are critical for weather applications where rare extremes often matter most.

The total training objective combines these five components as:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = w _ { 1 } \mathcal { L } _ { \mathrm { L 1 } } + w _ { 2 } \mathcal { L } _ { \mathrm { g r a d } } + w _ { 3 } \mathcal { L } _ { \mathrm { m u l t i - s c a l e } } + w _ { 4 } \mathcal { L } _ { \mathrm { p a t c h } } + w _ { 5 } \mathcal { L } _ { \mathrm { M B C } }
$$

with weights [w1, w2, w3, w4, w5] = [0.5, 2.0, 0.5, 0.1, 0.1] determined through systematic experimentation.

## 5.8 Ablation Study

To understand the contribution of each loss component and identify optimal weighting configurations, we conducted systematic ablation experiments that varied the weights of the L1 (w\_L1), gradient (w\_Grad), multi-scale structural (w\_Struct), patch (w\_Patch), and MBC (w\_MBC) loss terms. Supplementary Table S6 summarizes 15 model configurations designed to test specific hypotheses about loss component interactions, internal MBC metric balancing (quantile mapping Q vs. energy distance E), and the necessity of static geographic features. Supplementary Figures S19–S22 present quantitative comparisons across spatial accuracy, spectral fidelity, and distributional preservation metrics on the typical test set. This ablation was carried out with the earlier four-month warm-season training configuration; the selected weights were retained unchanged for every model trained in this work.

Supplementary Table S6: Ablation study configurations testing loss component contributions and optimal weighting.

<table><tr><td rowspan=1 colspan=1>ModD</td><td rowspan=1 colspan=1>LW</td><td rowspan=1 colspan=1>GradW</td><td rowspan=1 colspan=1>StrutW</td><td rowspan=1 colspan=1>tch</td><td rowspan=1 colspan=1>WBC</td><td rowspan=1 colspan=1>InternnalMBCSpit</td><td rowspan=1 colspan=1>Purose</td></tr><tr><td rowspan=1 colspan=1>M0 (U-NetL1)</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>Baseline without static features: U-Net L1model</td></tr><tr><td rowspan=1 colspan=1>M1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>Pure L1 reconstruction: establishes systematicover-smoothing baseline</td></tr><tr><td rowspan=1 colspan=1>M2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>Structure-preserving losses only: tests spatialsharpness without distributional correction</td></tr><tr><td rowspan=1 colspan=1>M3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>Q=0.8,E=0.2</td><td rowspan=1 colspan=1>Minimal distributional correction: tests if lightMBC improves extreme value tails</td></tr><tr><td rowspan=1 colspan=1>M4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>Q=0.8,E=0.2</td><td rowspan=1 colspan=1>Moderate distributional correction: balancesspatial structure and statistical fidelity</td></tr><tr><td rowspan=1 colspan=1>M5</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Q=0.8,E=0.2</td><td rowspan=1 colspan=1>Maximum distributional fidelity: tests ifextreme MBC  weight  over-constrainspredictions</td></tr><tr><td rowspan=1 colspan=1>M6</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.2</td><td rowspan=1 colspan=1>Q=0.2,E=0.8</td><td rowspan=1 colspan=1>Energy distance emphasis: tests multivariatecoupling over marginal distribution matching</td></tr><tr><td rowspan=1 colspan=1>M7(CASPER)</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>Q=0.8,E=0.2</td><td rowspan=1 colspan=1>Final CASPER configuration: emphasizessharpmeteorological features  throughenhanced gradient preservation</td></tr><tr><td rowspan=1 colspan=1>M8</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Q=0.8,E=0.2</td><td rowspan=1 colspan=1>Physics-prioritized: tests if reducing point-wise accuracy improves physical consistency</td></tr><tr><td rowspan=1 colspan=1>M9</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>Q=0.5,E=0.5</td><td rowspan=1 colspan=1>Balanced    MBC     components    withconservative     weight:     tests     equalenergy/quantile importance</td></tr><tr><td rowspan=1 colspan=1>M10</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>Q=0.5,E=0.5</td><td rowspan=1 colspan=1>Balanced MBC components with aggressiveweight: strong  distributional correctionequally weighted</td></tr><tr><td rowspan=1 colspan=1>M11</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>Sequential ablation step 1: isolates gradientloss contribution to boundary sharpness</td></tr><tr><td rowspan=1 colspan=1>M12</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>Sequential ablation step 2: adds multi-scalestructural consistency across atmospherichierarchy</td></tr><tr><td rowspan=1 colspan=1>M13</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>Sequential ablation step 3: adds patchcoherence for local spatial smoothness</td></tr><tr><td rowspan=1 colspan=1>M14</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Q=0.5,E=0.5</td><td rowspan=1 colspan=1>Sequential ablation step 4: complete loss withall components for comprehensive evaluation</td></tr></table>

![](images/cba69047ba061615f97a6319caded0b68f7e05a220941aa3eb5aab86d849f6fe.jpg)  
Supplementary Figure S19: Spatial predictions from Level 970 hPa for 3D variables and surface variables. Maps showing the predictions of 15 models with different weights for the loss function components compared to the WRF target for six atmospheric variables – temperature, U-wind, V-wind, relative humidity, 2 m temperature, and skin temperature –

(from top to bottom). Domain: 363×390 pixels at 1 km resolution covering the Montreal-Ottawa region.

![](images/58e4933609e55524b89df3bf39637251b443317aa096a4998a7e17eec12a070c.jpg)

![](images/8352ab4cbd0b42ad165981d25289e3540c075b0e4c0ebc105f25892804964787.jpg)

![](images/482b29adc7e441ade6ef969854eb5c32b633921b2382f6227aaa7c0f02d7004a.jpg)

![](images/e83036da415c3c398d981ca41dae85584ecf75942e6531ca831d4b4254214886.jpg)

![](images/e1bebcc16f70e20d264e0456490b047950112d229308449cdaed326c7871b36b.jpg)

![](images/b0d41072a7fd70592ec9f279b8ca0281bdde56b34f713d954f8eb10d0445eac0.jpg)  
Supplementary Figure S20: Spatial structure validation. 2D power spectra (power spectral density vs. wavenumber) for temperature, U-wind, V-wind, relative humidity, 2-m temperature, and skin temperature. Comparison across 15 models with different weights for the loss function components with WRF target.

![](images/6d4283d7cf477efe5273dfd08fd16a5183785f5ef6ceb826573312987440d475.jpg)

![](images/992d8538b21f17ca9b1233832b536948f6fee8109dbb7c729a96692e4aa54ca3.jpg)

![](images/7c82690b9ba6809a6897a7cfa1239916ae66517802679746607528f8d9a2f31c.jpg)

![](images/5b2ba77b9ea795d3105cb93794e2efdc43574685a543107545dfc010b07e351a.jpg)

![](images/092832f4e722c67409d3c31e33a886c80e5a0b0f586b9cebc7e05e7e5195df24.jpg)

![](images/88e8639da70daa3a2a9ec4f61a34fe97eb3d786fa46031bee42a208c64f663d9.jpg)  
Supplementary Figure S21: Statistical fidelity validation. Probability density functions (log scale) for temperature, U-wind, V-wind, relative humidity, 2-m temperature, and skin temperature comparing 15 models with different weights for the loss function components with WRF target. The logarithmic y-axis emphasizes the tails of the distribution, which represent extreme events. Kolmogorov-Smirnov (KS) statistics quantify distributional agreement, with lower values indicating better fidelity.

![](images/e3212a2a445cd9ae4686e10cb33f8653ee665a1a148d7d727bd6d57e89eebb4c.jpg)  
Supplementary Figure S22: Multivariate consistency and spatial coupling validation. Columns 1-2: T2-RH cross-variable gradients and joint distributions. Columns 3-4: U-V cross-

variable gradients and joint distributions. Comparison across 15 models with different weights for the loss function components with WRF target.

## 5.9 Training Protocol

Model training employs the AdamW optimizer [36] with initial learning rate $1 0 ^ { - 4 }$ and weight decay regularization 10⁻⁴. The complete training hyperparameter configuration appears in Supplementary Table S7.

Supplementary Table S7: Training hyperparameters and optimization configuration.
<table><tr><td>Parameter</td><td>Value</td><td>Description</td></tr><tr><td>Optimizer</td><td> $\mathrm { A d a m W }$ </td><td>Adam with decoupled weight decay</td></tr><tr><td>Initial learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>Maximum rate</td></tr><tr><td>Minimum learning rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td><td>Asymptotic rate from cosine annealing</td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>L2 regularization strength</td></tr><tr><td>Batch size (per GPU)</td><td>2</td><td>Samples per gradient computation</td></tr><tr><td>Gradient accumulation</td><td>2 steps</td><td>Effective batch size multiplier</td></tr><tr><td>Effective batch size</td><td>4</td><td>Total samples per parameter update</td></tr><tr><td>Gradient clip norm</td><td>1.0</td><td>Maximum gradient magnitude</td></tr><tr><td>Number of epochs</td><td>100</td><td>Maximum training duration</td></tr><tr><td>Early stopping patience</td><td>15 epochs</td><td>Tolerance before termination</td></tr><tr><td>Early stopping threshold</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>Minimum improvement required</td></tr><tr><td>Checkpoint frequency</td><td>5 epochs</td><td>Save interval</td></tr><tr><td>Data workers</td><td>2</td><td>Parallel data loading processes</td></tr></table>

The AdamW formulation decouples weight decay from gradient-based parameter updates, improving generalization performance compared to standard Adam optimization. The relatively small per-GPU batch size of 2 samples reflects memory constraints imposed by processing 363×390 spatial grids across 53 input and 42 output channels with FP32. To improve gradient estimate stability without exceeding available GPU memory, training employs gradient accumulation over 2 steps as shown in Supplementary Table S7, yielding an effective batch size of 4 samples before parameter updates. Gradients are clipped to a maximum norm of 1.0 to prevent training instabilities that can arise when optimizing deep networks with attention mechanisms.

Learning rate scheduling follows a cosine annealing strategy defined as $\begin{array} { r } { \eta _ { t } = \eta _ { \mathrm { m i n } } + \frac { 1 } { 2 } ( \eta _ { \mathrm { m a x } } - } \end{array}$ $\eta _ { \mathrm { m i n } } ) ( 1 + \cos ( \pi t / T ) )$ where � = 100 epochs represents the maximum training duration, $\eta _ { \mathrm { m a x } } = 1 0 ^ { - 4 }$ denotes the initial learning rate, and $\eta _ { \mathrm { m i n } } = 1 0 ^ { - 6 }$ specifies the minimum rate approached asymptotically. This schedule provides rapid initial convergence with high learning rates during early epochs, while enabling fine-scale parameter refinement in later epochs through gradual annealing toward the minimum learning rate.

Data loading operations employ 2 worker processes with memory pinning enabled to overlap CPU-GPU data transfer with GPU computation. Workers persist across training epochs, eliminating the overhead of subprocess initialization. Despite these optimizations, data loading occasionally creates bottlenecks in the training pipeline, suggesting that future work could benefit from faster storage systems or increased preprocessing parallelism to fully saturate GPU utilization.

Supplementary Figure S23 reports the training and validation loss of the nested one-, two-, four- and eight-month configurations, together with their in-sample and out-of-sample error.

![](images/64d49b0d8bcac7b784a55cdb7622c103f1b8497f1c0bf716110f69a4a0d60b3f.jpg)  
Supplementary Figure S23: Training and validation loss across data budgets. (a) Training (solid) and validation (dashed) loss over 100 epochs for the nested configurations of one, two, four and eight months (b1\_su, b2\_warm, b4\_warm, b8\_all); the two curves track each other throughout at every budget, and at the final epoch the validation loss exceeds the training loss by 1.8, 8.3, 0.8 and 2.2 % respectively. (b) The same models scored on their own validation split (in-sample) and on the four months withheld from every one-, two- and four-month configuration (out-of-sample), as all-channel normalized root-mean-square error, with the gap annotated; the eight-month configuration was trained on all eight months and so has no month in that common held-out set. In-sample error is flat across budgets while out-of-sample error falls from 0.51 to 0.33, so the larger error of the small-budget models on unseen months reflects climatological distance rather than a failure to fit the training data.

## 5.10 Baseline Model Specifications

Quadratic interpolation provides the simplest baseline by fitting a quadratic function between the input and its equivalent output variable. While computationally trivial and requiring no training, this approach provides only $C ^ { 0 }$ continuity (continuous but not differentiable) and severely smooths all spatial features, making it a natural lower bound on expected downscaling performance. We also trained a random forest model with 100 trees as another baseline model for basic machine learning applications

An architecturally identical U-Net using exclusively the L1 reconstruction loss was also used as a baseline. This model maintains identical network capacity (architecture specified in Supplementary Table S4), training procedures, learning rate schedules (Supplementary Table S7), and optimization hyperparameters, differing only in the loss-function specification, with weights [w1, w2, w3, w4, w5] = [1, 0, 0, 0, 0], and in omitting the static geographic inputs, to show the improvement their integration brings.

We also attempted a generative diffusion baseline to gauge whether a score-based model could match CASPER at the same data budget. Following the one-stage conditional formulation of denoising diffusion, we trained a denoiser of comparable capacity to predict the noise added to the 42-channel high-resolution field, conditioned on the same 53-channel coarse input, on the identical eight-month training set and normalization used for CASPER, and sampled with the standard 1000-step ancestral reverse process. The per-step denoising loss decreased steadily over training, but running the full reverse-diffusion chain did not yield physically valid fields: the generated 2 m temperature drifted far outside any realizable range, giving errors of order

100 K against the WRF target, so the model could not be scored on the test splits. This is consistent with the well-documented dependence of generative diffusion models on large and diverse training corpora for stable sampling, and with our central finding that score-based training is ill-suited to the small-data regime this study targets. We therefore report the diffusion attempt qualitatively and exclude it from the matched-budget comparison in Table 2. An ablation study presented in the extended results shows the performance of models with different weights for loss components. Comparison between this L1-only model and the full CASPER system reveals the specific performance gains attributable to gradient preservation, multi-scale structural, and statistical losses.

Supplementary Table S8: Selection-strategy experiment. Held-out 2 m temperature RMSE $( \mathrm { K } ; \mathrm { m e a n } \pm \mathrm { s . d . }$ . over three seeds) when fine-tuning the one- and four-month base models on N samples chosen by each selection strategy (Supplementary Note 5.4).
<table><tr><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>Strategy</td><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>Bulk RMSE (K)</td><td rowspan=1 colspan=1>Tail RMSE (K)</td></tr><tr><td rowspan=1 colspan=1>1 month</td><td rowspan=1 colspan=1>coverage</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1> $4 . 8 8 \pm 0 . 0 9$ </td><td rowspan=1 colspan=1> $4 . 5 8 \pm 0 . 6 5$ </td></tr><tr><td rowspan=1 colspan=1>1 month</td><td rowspan=1 colspan=1>coverage</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1> $3 . 9 9 \pm 0 . 4 0$ </td><td rowspan=1 colspan=1> $3 . 1 8 \pm 0 . 0 6$ </td></tr><tr><td rowspan=1 colspan=1>1 month</td><td rowspan=1 colspan=1>mindist</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1> $8 . 4 7 \pm 0 . 4 2$ </td><td rowspan=1 colspan=1> $3 . 3 3 \pm 0 . 1 0$ </td></tr><tr><td rowspan=1 colspan=1>1 month</td><td rowspan=1 colspan=1>mindist</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1> $5 . 9 8 \pm 0 . 4 1$ </td><td rowspan=1 colspan=1> $3 . 3 5 \pm 0 . 3 2$ </td></tr><tr><td rowspan=1 colspan=1>1 month</td><td rowspan=1 colspan=1>nearest</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1> $5 . 2 4 \pm 0 . 1 6$ </td><td rowspan=1 colspan=1> $7 . 5 8 \pm 0 . 6 4$ </td></tr><tr><td rowspan=1 colspan=1>1 month</td><td rowspan=1 colspan=1>nearest</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1> $5 . 0 5 \pm 0 . 0 5$ </td><td rowspan=1 colspan=1> $5 . 8 6 \pm 0 . 1 5$ </td></tr><tr><td rowspan=1 colspan=1>1 month</td><td rowspan=1 colspan=1>random</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1> $5 . 8 2 \pm 0 . 7 3$ </td><td rowspan=1 colspan=1> $5 . 5 7 \pm 1 . 0 7$ </td></tr><tr><td rowspan=1 colspan=1>1 month</td><td rowspan=1 colspan=1>random</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1> $4 . 2 6 \pm 0 . 2 2$ </td><td rowspan=1 colspan=1> $3 . 7 6 \pm 0 . 3 8$ </td></tr><tr><td rowspan=1 colspan=1>4 months</td><td rowspan=1 colspan=1>coverage</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1> $3 . 6 8 \pm 0 . 0 7$ </td><td rowspan=1 colspan=1> $4 . 5 4 \pm 0 . 1 3$ </td></tr><tr><td rowspan=1 colspan=1>4 months</td><td rowspan=1 colspan=1>coverage</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1> $3 . 1 6 \pm 0 . 0 8$ </td><td rowspan=1 colspan=1> $2 . 5 1 \pm 0 . 1 0$ </td></tr><tr><td rowspan=1 colspan=1>4 months</td><td rowspan=1 colspan=1>mindist</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1> $4 . 4 5 \pm 0 . 0 8$ </td><td rowspan=1 colspan=1> $3 . 3 0 \pm 0 . 1 9$ </td></tr><tr><td rowspan=1 colspan=1>4 months</td><td rowspan=1 colspan=1>mindist</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1> $4 . 4 3 \pm 0 . 1 7$ </td><td rowspan=1 colspan=1> $2 . 9 1 \pm 0 . 1 3$ </td></tr><tr><td rowspan=1 colspan=1>4 months</td><td rowspan=1 colspan=1>nearest</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1> $4 . 3 4 \pm 0 . 1 1$ </td><td rowspan=1 colspan=1> $6 . 0 4 \pm 0 . 2 0$ </td></tr><tr><td rowspan=1 colspan=1>4 months</td><td rowspan=1 colspan=1>nearest</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1> $3 . 5 3 \pm 0 . 1 5$ </td><td rowspan=1 colspan=1> $3 . 9 2 \pm 0 . 4 2$ </td></tr><tr><td rowspan=1 colspan=1>4 months</td><td rowspan=1 colspan=1>random</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1> $4 . 0 3 \pm 0 . 2 7$ </td><td rowspan=1 colspan=1> $3 . 5 5 \pm 0 . 7 8$ </td></tr><tr><td rowspan=1 colspan=1>4 months</td><td rowspan=1 colspan=1>random</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1> $3 . 2 9 \pm 0 . 0 9$ </td><td rowspan=1 colspan=1> $3 . 1 0 \pm 0 . 1 6$ </td></tr></table>

## References

[1] D. Maraun and M. Widmann, Statistical Downscaling and Bias Correction for Climate Research. Cambridge University Press, 2018.

[2] S. Agrawal, L. Barrington, C. Bromberg, J. Burge, C. Gazen, and J. Hickey, “Machine Learning for Precipitation Nowcasting from Radar Images,” Dec. 11, 2019, arXiv: arXiv:1912.12132. doi: 10.48550/arXiv.1912.12132.

[3] C. Hutengs and M. Vohland, “Downscaling land surface temperatures at regional scales with random forest regression,” Remote Sensing of Environment, vol. 178, pp. 127–141, Jun. 2016, doi: 10.1016/j.rse.2016.03.006.

[4] Y. Sha, D. J. Gagne Ii, G. West, and R. Stull, “Deep-Learning-Based Gridded Downscaling of Surface Meteorological Variables in Complex Terrain. Part II: Daily Precipitation,” Journal of Applied Meteorology and Climatology, vol. 59, no. 12, pp. 2075–2092, Dec. 2020, doi: 10.1175/JAMC-D-20-0058.1.

[5] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, vol. 9351, N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi, Eds., in Lecture Notes in Computer Science, vol. 9351. , Cham: Springer International Publishing, 2015, pp. 234–241. doi: 10.1007/978-3-319-24574-4\_28.

[6] S. K. Jha, V. Gupta, P. J. Sharma, A. Mishra, and S. Joshi, “Deep learning superresolution for temperature data downscaling: a comprehensive study using residual networks,” Front. Clim., vol. 7, p. 1572428, May 2025, doi: 10.3389/fclim.2025.1572428.

[7] N. Rampal et al., “High-resolution downscaling with interpretable deep learning: Rainfall extremes over New Zealand,” Weather and Climate Extremes, vol. 38, p. 100525, Dec. 2022, doi: 10.1016/j.wace.2022.100525.

[8] T. Vandal, E. Kodra, S. Ganguly, A. Michaelis, R. Nemani, and A. R. Ganguly, “DeepSD: Generating High Resolution Climate Change Projections through Single Image Super-Resolution,” Mar. 09, 2017, arXiv: arXiv:1703.03126. doi: 10.48550/arXiv.1703.03126.

[9] J. Baño-Medina, R. Manzanas, and J. M. Gutiérrez, “Configuration and intercomparison of deep learning neural models for statistical downscaling,” Geosci. Model Dev., vol. 13, no. 4, pp. 2109–2124, Apr. 2020, doi: 10.5194/gmd-13-2109-2020.

[10] L. Glawion, J. Polz, H. Kunstmann, B. Fersch, and C. Chwala, “Global spatio-temporal ERA5 precipitation downscaling to km and sub-hourly scale using generative AI,” npj Clim Atmos Sci, vol. 8, no. 1, p. 219, Jun. 2025, doi: 10.1038/s41612-025-01103-y.

[11] S. Qin, D. Zhan, A. Marey, D. Geng, T. Potsis, and L. L. Wang, “Data-efficient rapid prediction of urban airflow and temperature fields for complex building geometries,” Mar. 25, 2025, arXiv: arXiv:2503.19708. doi: 10.48550/arXiv.2503.19708.

[12] S. Qin, D. Zhan, A. Marey, and T. Potsis, “Generalizable Deep Learning for Rapid Urban Wind Field Prediction Trained on Only 24 CFD Simulations”.

[13] E. Tomasi, G. Franch, and M. Cristoforetti, “Can AI be enabled to dynamical downscaling? A Latent Diffusion Model to mimic km-scale COSMO5.0\_CLM9 simulations,” Geosci. Model Dev., vol. 18, no. 6, pp. 2051–2078, Apr. 2025, doi: 10.5194/gmd-18-2051-2025.

[14] X. Hong, L. Hu, and A. Kareem, “A tropical cyclone intensity prediction model using conditional generative adversarial network,” Journal of Wind Engineering and Industrial Aerodynamics, vol. 240, p. 105515, Sep. 2023, doi: 10.1016/j.jweia.2023.105515.

[15] B. Kumar et al., “On the modern deep learning approaches for precipitation downscaling,” Earth Sci Inform, vol. 16, no. 2, pp. 1459–1472, Jun. 2023, doi: 10.1007/s12145-023-00970-4.

[16] N. Rampal et al., “Enhancing Regional Climate Downscaling through Advances in Machine Learning,” Artificial Intelligence for the Earth Systems, vol. 3, no. 2, p. 230066, Apr. 2024, doi: 10.1175/AIES-D-23-0066.1.

[17] J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” Dec. 16, 2020, arXiv: arXiv:2006.11239. doi: 10.48550/arXiv.2006.11239.

[18] S. Sinha, B. Benton, and P. Emami, “On the Effectiveness of Neural Operators at Zero-Shot Weather Downscaling,” Environ. Data Science, vol. 4, p. e21, 2025, doi: 10.1017/eds.2025.11.

[19] R. A. Watt and L. A. Mansfield, “Generative Diffusion-based Downscaling for Climate,” Apr. 27, 2024, arXiv: arXiv:2404.17752. doi: 10.48550/arXiv.2404.17752.

[20] S. Tahmasebi, G. Tian, S. Qin, A. Marey, L. (Leon) Wang, and S. Rayegan, “Using diffusion models for reducing spatiotemporal errors of deep learning based urban microclimate predictions at post-processing stage,” Physics of Fluids, vol. 37, no. 3, p. 035173, Mar. 2025, doi: 10.1063/5.0256658.

[21] A. Vaswani et al., “Attention Is All You Need,” Aug. 02, 2023, arXiv: arXiv:1706.03762. doi: 10.48550/arXiv.1706.03762.

[22] P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, and A. G. Wilson, “Averaging Weights Leads to Wider Optima and Better Generalization”.

[23] H. Zhao, O. Gallo, I. Frosio, and J. Kautz, “Loss Functions for Image Restoration With Neural Networks,” IEEE Trans. Comput. Imaging, vol. 3, no. 1, pp. 47–57, Mar. 2017, doi: 10.1109/TCI.2016.2644865.

[24] A. Marey, J. Zou, S. Goubran, L. L. Wang, and A. Gaur, “Urban morphology impacts on urban microclimate using artificial intelligence – a review,” City and Environment Interactions, vol. 28, p. 100221, Dec. 2025, doi: 10.1016/j.cacint.2025.100221.

[25] A. Marey et al., “Forecasting Urban Land Use Dynamics Through Patch-Generating Land Use Simulation and Markov Chain Integration: A Multi-Scenario Predictive Framework,” Sustainability, vol. 16, no. 23, p. 10255, Nov. 2024, doi: 10.3390/su162310255.

[26] A. Marey, L. L. Wang, and S. Goubran, “Developing accurate land cover projection to accelerate the realization of SDG 11 in urbanized cities: a comparative study,” Clean Technologies and Environmental Policy, Aug. 2025, doi: 10.1007/s10098-025-03297-4.

[27] W. C. Skamarock et al., “A Description of the Advanced Research WRF Version 4,” 2021.

[28] National Centers for Environmental Prediction, National Weather Service, NOAA, U.S. Department of Commerce, “NCEP North American Regional Reanalysis (NARR).” Research Data Archive at the National Center for Atmospheric Research, Computational and Information Systems Laboratory, Boulder CO, 2005. [Online]. Available: https://rda.ucar.edu/datasets/ds608.0/

[29] A. Marey, L. L. Wang, A. Gaur, H. Lu, S. Leroyer, and S. Belair, “Urban climate simulation for extreme heat events – A comparison between WRF and GEM,” Urban Climate, vol. 63, p. 102570, 2025, doi: https://doi.org/10.1016/j.uclim.2025.102570.

[30] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” Dec. 10, 2015, arXiv: arXiv:1512.03385. doi: 10.48550/arXiv.1512.03385.

[31] Y. Wu and K. He, “Group Normalization,” Jun. 11, 2018, arXiv: arXiv:1803.08494. doi: 10.48550/arXiv.1803.08494.

[32] S. Zhang, J. Lu, and H. Zhao, “Deep Network Approximation: Beyond ReLU to Diverse Activation Functions,” Jan. 31, 2024, arXiv: arXiv:2307.06555. doi: 10.48550/arXiv.2307.06555.

[33] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: A Simple Way to Prevent Neural Networks from Overfitting”.

[34] A. J. Cannon, “Multivariate quantile mapping bias correction: an N-dimensional probability density function transform for climate model simulations of multiple variables,” Clim Dyn, vol. 50, no. 1–2, pp. 31–49, Jan. 2018, doi: 10.1007/s00382-017- 3580-6.

[35] G. J. Székely and M. L. Rizzo, “Energy statistics: A class of statistics based on distances,” Journal of Statistical Planning and Inference, vol. 143, no. 8, pp. 1249–1272, Aug. 2013, doi: 10.1016/j.jspi.2013.03.018.

[36] I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” Jan. 04, 2019, arXiv: arXiv:1711.05101. doi: 10.48550/arXiv.1711.05101.