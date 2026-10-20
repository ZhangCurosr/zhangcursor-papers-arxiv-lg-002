# OceanTensorCube and ST-MMT: A Multimodal Data and Transformer System for Seven-Day Forecasting of Pyropia Bleaching

Gwangjin Kim<sup>\*1</sup>, Cheolyoung Jang<sup>2</sup>, and Yosep Jo<sup>3</sup>

<sup>1</sup>NCS Instructor, Smart Human Resources Development Institute, Republic of Korea <sup>2</sup>NH Networks LLC, Republic of Korea <sup>3</sup>DreamTech Co., Ltd., Republic of Korea

## Abstract

Bleaching (discoloration) of cultured laver (Pyropia spp.) destroys the market value of a crop that is harvested from floating nets over a few months each winter, and the damage is usually noticed only after the blades have lost their pigment. We present ST-MMT, a spatiotemporal multimodal transformer that forecasts bleaching seven days ahead for laver-farming areas on the south-western coast of Korea. The model reads a 14-day stack of 32 co-registered channels on a 64 × 64 grid at 30 m resolution and combines two paths: a joint encoder over all channels, and three modality-specific encoders (in-situ ocean, atmosphere, satellite) that are fused by two rounds of multi-head cross-attention with the in-situ ocean stream as the query hub. The cross-attention output enters the joint representation through a learnable residual gate. Inputs are drawn from the OceanTensorCube, a layered (Bronze/Silver/Gold) pipeline that aligns in-situ surveys, weather stations, ocean-model fields and satellite reflectance on a common grid. We also describe the Bleaching Label Calibrator (BLC), a procedure that converts point-in-time damage reports into daily soft labels and confidences; the model evaluated here is trained on binary labels. On a validation set of 2,792 cluster-day samples (176 positives, base rate 6.30%), which was also used to select checkpoints and the alert threshold, a six-seed ensemble reaches ROC-AUC 0.9940 and PR-AUC 0.9097. At the operating threshold chosen to detect every reported positive (0.0940), precision is 65.7% and the false-alarm rate is at most 3.5% (92 of 2,616 negatives, all in 2024). Because positive labels occur only in October–November 2025, a calendar rule alone separates the labels. In a separate cross-validated training of the same architecture designed to remove this shortcut, the AUC among clusters on the same November dates was 0.976–0.986 (bootstrap lower bounds ≥ 0.94); the calendar independence of the deployed ensemble itself has not been measured. We also quantify the main operational obstacle: dissolved inorganic nitrogen (DIN) is surveyed only quarterly, and freezing the nutrient channels 30, 60 and 129 days before the input window lowers the AUC to 0.941, 0.833 and 0.527. We report the negative results that shaped the design, including a rejected proxy for nutrient drift, and list what remains unmeasured.

## 1 Introduction

Laver, the red alga known as gim in Korea and nori in Japan, is farmed on nets suspended in shallow coastal water and is one of the most valuable marine crops of East Asia [19, 27, 34]. The cultivated species, formerly placed in Porphyra and now largely assigned to Pyropia [55], are tolerant of strong intertidal stress [6], yet a recurring failure mode remains: the blades lose their dark pigment and turn yellow-brown or pale. This discoloration, which we call bleaching following the Korean term hwangbaekhwa, lowers product grade sharply. Long-term studies in the Seto Inland Sea and the Ariake Sea link it to the depletion of dissolved inorganic nitrogen (DIN), often after blooms of large diatoms such as Eucampia zodiacus that compete with the farmed blades for nutrients [26, 28, 29, 47]. Temperature also matters: low temperature has been shown to cause discoloration by repressing growth and nitrogen-transporter expression [56], and nutrient-ratio limitation has been documented in Korean farming grounds [36].

![](images/c10789e036869730dccafd2d31708c5f45ee843a50eee8393474bdd1857efbcf.jpg)  
(a) Laver on culture nets

![](images/7bca5682497a006c5eac0ac132b8357dae6e5b19cf1102e6347df86b724e1572.jpg)  
(b) Net seen from above  
Figure 1: Laver-farming grounds photographed in connection with the 2022 Haenam bleaching case. Photographs courtesy of the National Institute of Fisheries Science (NIFS).

The 2022 event in Haenam, South Jeolla, shows the scale of a single episode (Figure 1). Bleaching began in early January 2022 at farms around Songji-myeon. By 4 February, a provisional count by the National Institute of Fisheries Science (NIFS) put 351 of the 418 laver-farming households in Haenam as affected, with 31,170 of the county’s 70,208 net sets (one set measures 2.2 m by 40 m) [44]. On the same day a newspaper reported 2,236 ha affected in the Songji, Hwasan and Hyeonsan townships, spreading toward Hwangsan, and losses of about 12.2 billion won [51]. Samples from a joint survey on 18 January contained a bloom of the diatom Cylindrotheca closterium at 1,653–9,958 cells/mL, which was presumed to have consumed nitrogen and other nutrients in the water [44]; NIFS later described the event as a case of nutrient depletion caused by mass growth of phytoplankton [45].

Farmers can respond to an expected nutrient shortage, for example by applying nutrients or by advancing the harvest, but only if they are warned before the pigment loss is visible. That makes bleaching a forecasting problem with three unusual properties.

1. Heterogeneous and asynchronous inputs. The drivers are observed by different systems at very different cadences: hourly weather stations, daily ocean-model fields, weekly or 8-day satellite composites, and nutrient surveys that are repeated only a few times per year. A Korean national monitoring network measures dissolved inorganic nutrients four times a year [22].

2. Rare, report-based labels. Bleaching is recorded when an agency or a farmer reports damage. The report date marks when the damage was noticed rather than when the environment turned unfavourable, and unreported case (for example farms that recovered after timely fertilisation) are absent. The negatives are therefore unlabeled rather than confirmed healthy, which is the positive–unlabeled setting [4, 16].

3. Asymmetric costs. A missed event can mean a lost harvest, whereas a false alarm costs an unnecessary intervention. Operational evaluation should therefore fix the miss rate first and report the false alarms that remain.

Machine learning has been applied to related coastal problems such as harmful algal bloom and red-tide prediction [3, 12, 13, 23, 37, 40] and sea-surface temperature forecasting on the same coast [35]. Transformer architectures [58] have become the standard tool for sequence and spatiotemporal data [1, 5, 15, 20], and cross-attention is the usual mechanism for fusing modalities [8, 30, 43, 57]. We are not aware of a published deep-learning model that forecasts laver bleaching from multimodal ocean, weather and satellite data.

Contributions. This paper describes a working system and is explicit about what has and has not been measured.

• OceanTensorCube (Section 3): a Bronze/Silver/Gold data pipeline that aligns in-situ nutrient surveys, weather stations, ocean-model currents and mixed-layer depth, and satellite near-infrared reflectance onto a common 30 m grid, together with the rules we use to keep the inputs causal and symmetric between years.

• BLC, the Bleaching Label Calibrator (Section 4): a procedure that turns point-in-time damage reports into daily soft labels, severity grades and confidences.

• ST-MMT (Section 5): a residual architecture in which a joint spatiotemporal self-attention encoder is refined by

hub-style multi-head cross-attention between modality groups.

• An evaluation protocol for report-based labels (Section 6): a recall-first operating threshold, sample-level counting, a same-date control against calendar shortcuts, and pre-registered acceptance criteria.

• An operational analysis of the nutrient gap (Section 8): how fast forecast skill decays when the nutrient survey ages, how persistence and climatology compare as stand-ins, and why a daily proxy for DIN drift was rejected.

## 2 Related work

Laver discoloration and nutrients. Field studies in Japan established that nori discoloration follows declines in DIN and that winter diatom blooms are a frequent cause [26, 29, 47]. The broader context is a shift from eutrophic to oligotrophic conditions in enclosed seas [28], which mechanistic ecosystem models try to reproduce [54]. Physiological work identifies low temperature and nutrient-ratio limitation as further factors [36, 56], and disease must be distinguished from nutrient bleaching when yield falls [2]. Our work is data-driven and complementary: it does not model the nitrogen budget but learns which combinations of observed conditions precede reported damage. The stoichiometric reasoning behind the nitrogen-to-phosphorus ratio channel follows Redfield [48].

Machine learning for coastal hazards. Reviews of harmful-algal-bloom forecasting describe a progression from regression and tree ensembles to deep networks that combine in-situ series and remote sensing [13, 23]. In Korea, convolutional networks driven by weather and ocean-model fields have predicted Cochlodinium blooms in the South Sea [12], ocean-colour imagery from the Geostationary Ocean Color Imager (GOCI) [11, 49] has been used to detect red tides [37], imbalanced red-tide data have been addressed with oversampling [3], and recurrent networks have forecast blooms in rivers [40] and high water temperature near Goheung and Yeosu [35]. Most of these studies treat events as reliably labeled and evaluate at a default threshold; our labels are sparser and noisier, which motivates the protocol in Section 6.

Attention for spatiotemporal and multimodal data. Multi-head attention [58] applied to image patches [15] and factorized over space and time [1, 5] is the basis of our encoders. For weather and climate fields, Earthformer applies cuboid attention to Earth-system data [20]. Time-series transformers such as Informer, the Temporal Fusion Transformer and PatchTST handle long horizons, mixed covariates and patching [41, 46, 59]. Cross-attention fuses streams in vision–language models [43, 57], in Perceiver-style latent bottlenecks [30, 31] and between token streams of different scale [8]. Convolutional recurrent models such as ConvLSTM [52] and LSTMs [25] remain strong baselines for gridded sequences.

Weak labels, imbalance and leakage. When negatives are only unlabeled, false-alarm rates measured against them are upper bounds and recall is measured only on reported events [4, 16]. Under heavy class imbalance, precision–recall analysis is more informative than ROC analysis [14, 50], and the Matthews correlation coefficient summarizes a confusion matrix more faithfully than accuracy or F1 [10]. Leakage, information in the inputs that would not be available at prediction time or that encodes the label indirectly, is a common cause of inflated results in applied machine learning [32, 33]; Section 6 describes the specific leakage paths we found and how we controlled them. We use focal loss [42] for imbalance and seed ensembles [39] for stability.

## 3 Data: the OceanTensorCube

## 3.1 Study area and spatial units

The unit of prediction is a tile cluster. The national register of licensed laver-farming zones contains 1,194 polygons, concentrated on the coasts of South Jeolla (Jeonnam) and North Jeolla (Jeonbuk); we group them into 105 clusters. Each cluster is represented by a 512 × 512 tile at 30 m resolution in the Korean projected coordinate system EPSG:5179, which is 15.36 km on a side. The model sees the central 64 × 64 window (1.92 km on a side, pixel indices 224–287 in both directions). Ten clusters without in-situ observations were excluded; patches exist for 81 clusters in 2024 and 95 in 2025. Forecasts are mapped back from clusters to farms; the current forecast pack covers 1,109 of the 1,194 farms, and the remaining 85 belong to clusters without 2024 or 2025 patches and are reported as “no forecast” rather than filled.

Table 1: Data sources in the OceanTensorCube. Cadence is the native update interval of the variables as used.
<table><tr><td>Source</td><td>Variables (channels)</td><td>Type</td><td>Cadence</td></tr><tr><td>KOEM monitoring network</td><td>SST, DIN, DIP, SiO2, salinity, chlorophyll, DO, pH,  $\mathrm { N O _ { 3 } , N H _ { 4 } , }$  suspended solids (11)</td><td>point stations</td><td>quarterly surveys</td></tr><tr><td>KMA ASOS</td><td>precipitation, solar radiation, wind speed, wind direction (sin, cos), air temperature (6)</td><td>point stations</td><td>hourly → daily</td></tr><tr><td>HYCOM / KHOA; ocean model</td><td>surface current u, v; mixed-layer depth (3)</td><td>model grid / stations</td><td>sub-daily → daily</td></tr><tr><td>Sentinel-2 L2A Derived</td><td>near-infrared index (1) N:P ratio, SST anomaly, 7- and 30-day means, SST</td><td>30 m imagery computed</td><td>8-day composite</td></tr><tr><td></td><td>gradient, current speed, days since rain, NIR daily change (9)</td><td></td><td>as parents</td></tr><tr><td>Calendar</td><td>month sine and cosine (2)</td><td>static</td><td>monthly</td></tr><tr><td>NIFS reports</td><td>bleaching damage reports (labels)</td><td>events</td><td>irregular</td></tr></table>

## 3.2 Sources

Table 1 lists the sources. They differ by more than two orders of magnitude in update frequency, which is the central difficulty of the problem. The marine environment monitoring network of the Korea Marine Environment Management Corporation (KOEM) provides nutrients and related water-quality variables at fixed stations; its nutrient survey campaigns are quarterly (February, May, August and November), consistent with the four-per-year schedule reported for the national network [22]. After interpolating several stations onto a tile, the KOEM channels in our patches change value on 6–8 days per year, and one of our internal audits describes the cadence as six surveys per year; we use “quarterly” for the campaign schedule. Weather comes from the Korea Meteorological Administration (KMA) Automated Synoptic Observing System (ASOS) [38]. Currents come from the HYCOM global ocean forecast system [7] for 2024 and from the Korea Hydrographic and Oceanographic Agency (KHOA) from 2025. Mixed-layer depth comes from an ocean-model product: our data audit traces it to the HYCOM mixed-layer thickness, while the latest build scripts read a daily CMEMS table [17]; we have not reconciled the two records. The near-infrared index used in the final patches comes from Sentinel-2 Level-2A 8-day composites extracted over farm polygons; earlier development versions used GOCI-II [11, 49]. Daily biogeochemistry from the Copernicus Marine Service [18] is ingested but is not used by the model reported here (Section 9 describes a replacement experiment).

## 3.3 Layered pipeline

The pipeline follows a Bronze/Silver/Gold layering (Figure 2).

• Bronze stores raw records from each provider in a relational database, without transformation.

• Silver converts units, interpolates point data onto each tile grid and assembles a cube $\mathbf { V } \in \mathbb { R } ^ { T \times 3 2 \times 5 1 2 \times 5 1 2 }$ per tile and calendar year (T = 365 or 366). Nutrient concentrations reported in µg/L are converted to µmol/L by dividing by the atomic weight of the element (14.007 for nitrogen species, 30.974 for phosphate, 28.086 for silicate). Point observations are interpolated by inverse-distance weighting (power 2, nearest four stations for KOEM and nearest stations for KMA). Currents are interpolated with a thin-plate radial basis function and then projected onto their divergence-free part by a Helmholtz decomposition (1,500 Jacobi iterations), because a scalar interpolation of a vector field creates spurious convergence. Cubes are stored as Zarr arrays with one chunk per day (1, 32, 512, 512) and zstd compression; a raw tile-year of 12.25 GB is stored in about 2.5 GB.

• Gold forward-fills the sparse channels, extracts the central 64 × 64 patch, and writes one array of shape (T, 32, 64, 64) per cluster and year (tens of megabytes). Gold also stores the sample index and labels.

We use the name OceanTensorCube for the resulting family of aligned tensors. The design target is a five-dimensional array indexed by region, month, channel and grid, of shape [8, 72, 32, 512, 512] (eight regions, 72 months). What exists today are the per-tile, per-year four-dimensional slices described above.

## 3.4 The 32 channels and their real information content

Table 2 lists the channels. The count overstates the number of independent observations. The eleven KOEM channels come from a single water sample per survey, so they change on the same days: in the 2025 training patches the DIN channel has 7–9 distinct values per year and changes on only $_ { 6 - 8 }$ days, and SST, salinity, chlorophyll, DO, pH, $\mathrm { N O _ { 3 } }$ and $\mathrm { N H _ { 4 } }$ change on exactly the same days. Counted by information source, the 32 channels comprise 6 daily weather variables, 3 daily ocean-model or current variables, 1 satellite variable, 11 low-frequency point variables, 9 derived variables and 2 calendar variables.

![](images/32ec4434699c51c58eb3ecc3f3534a32c1b667d35fb145ec3c383832e56f9bd5.jpg)  
Figure 2: The OceanTensorCube pipeline and serving path. Raw records (Bronze) are aligned to a 30 m tile grid (Silver), forwardfilled and cropped to 64 × 64 training patches (Gold). The six-seed ensemble writes a forecast pack that the web service reads through a three-stage output gate (Section 8.6); the service never runs the network itself.

Two derived channels needed correction during development. The N:P ratio is computed as

$$
\mathrm { N { : } P = \mathrm { c l i p \big ( D I N / \operatorname* { m a x } ( D I P , 0 . 0 1 ) , 0 , 2 0 0 \big ) ; } }
$$

an earlier form divided by $\mathrm { D I P } + 1 0 ^ { - 6 }$ and produced values up to $2 . 6 5 \times 1 0 ^ { 6 }$ in one tile, which dominated the gradient saliency until it was fixed. The NIR channel 29 was first a 7-day moving average, which correlated with channel 24 at 0.78 and lowered AUC from 0.903 to 0.850 in a development run; the day-to-day change (correlation 0.26) restored it to 0.917, and the definition was changed accordingly.

Figure 3 shows the discrimination of each channel on its own: the area under the ROC curve of the 14-day window mean, computed on the validation set and taken in whichever direction is larger. The strongest single channels are wind speed (0.942), month sine (0.917), N:P ratio (0.908) and the cosine of wind direction (0.906). For reference, the year alone scores 0.800 and the month alone 0.858. High single-channel scores for slowly varying, season-correlated variables are a warning sign rather than an achievement: they show how much of the label can be explained by when a sample was taken (Section 4.2).

## 3.5 Missing values, causality and year symmetry

Three rules govern how gaps are filled. Each was introduced after a specific failure, and one known violation remains in the training data.

Forward fill. Twelve sparse channels (the KOEM channels other than suspended solids, plus the N:P ratio and SST gradient) are forward-filled within each calendar year: an unobserved day carries the most recent observation. Forward filling is causal, but it encodes the assumption that the variable has not changed since the last survey (Section 8).

Table 2: Channel definitions. Source is the source actually used by the extraction code; where it differs from the original channel master (turbidity, solar-radiation units, currents, mixed-layer depth), the code is authoritative.
<table><tr><td>ch</td><td>Name</td><td>Source</td><td>Unit</td><td>ch</td><td>Name</td><td>Source</td><td>Unit</td></tr><tr><td>0</td><td>SST</td><td>KOEM</td><td> $^ \circ _ { \mathbf { C } }$ </td><td>16</td><td>Wind speed</td><td>KMA</td><td>m/s</td></tr><tr><td>1</td><td>DIN</td><td>KOEM</td><td>µmol/L</td><td>17</td><td>Wind direction (sin)</td><td>KMA</td><td></td></tr><tr><td>2</td><td>DIP</td><td>KOEM</td><td>µmol/L</td><td>18</td><td>Wind direction (cos)</td><td>KMA</td><td></td></tr><tr><td>3</td><td>SiO2</td><td>KOEM</td><td>µmol/L</td><td>19</td><td>Air temperature</td><td>KMA</td><td> $^ \circ \mathrm { C }$ </td></tr><tr><td>4</td><td>DIN/DIP ratio</td><td>derived</td><td>ratio</td><td>20</td><td>pH</td><td>KOEM</td><td></td></tr><tr><td>5</td><td>Salinity</td><td>KOEM</td><td>psu</td><td>21</td><td> $\mathrm { \bar { N O } _ { 3 } }$ </td><td>KOEM</td><td>µmol/L</td></tr><tr><td>6</td><td>Precipitation</td><td>KMA</td><td>mm/day</td><td>22</td><td> $\mathrm { N H _ { 4 } }$ </td><td>KOEM</td><td>µmol/L</td></tr><tr><td>7</td><td>Solar radiation</td><td>KMA</td><td> $ { \mathrm { M J / m ^ { 2 } / d a y } }$ </td><td>23</td><td>SST gradient</td><td>derived</td><td>°C/km</td></tr><tr><td>8</td><td>Chlorophyll</td><td>KOEM</td><td> $\mathrm { m g / m ^ { 3 } }$ </td><td>24</td><td>NIR index</td><td>Sentinel-2</td><td></td></tr><tr><td>9</td><td>DO</td><td>KOEM</td><td> $\mathrm { m g / L }$ </td><td>25</td><td>Month (sin)</td><td>calendar</td><td></td></tr><tr><td>10</td><td>Current u</td><td>HYCOM/KHOA m/s</td><td></td><td>26</td><td>Month (cos)</td><td>calendar</td><td></td></tr><tr><td>11</td><td>Current v</td><td>HYCOM/KHOA</td><td>m/s</td><td>27</td><td>Current speed</td><td>derived</td><td>m/s</td></tr><tr><td>12</td><td>SST anomaly</td><td>derived</td><td> $^ \circ _ { \mathbf { C } }$ </td><td>28</td><td>Chlorophyll 7-day mean</td><td>derived</td><td> $\mathrm { m g / m ^ { 3 } }$ </td></tr><tr><td>13</td><td>SST 7-day mean</td><td>derived</td><td> $^ \circ _ { \mathbf { C } }$ </td><td>29</td><td>NIR daily change</td><td>derived</td><td></td></tr><tr><td>14</td><td>Days since rain</td><td>derived</td><td>days</td><td>30</td><td>SST 30-day mean</td><td>derived</td><td> $^ \circ \mathrm { C }$ </td></tr><tr><td>15</td><td>Suspended solids</td><td>KOEM</td><td>mg/L</td><td>31</td><td>Mixed-layer depth</td><td>HYCOM</td><td>m</td></tr></table>

Missing as zero. At patch extraction, remaining gaps are set to zero. Zero is a physically valid value for several channels (no rain, no sunlight), so the network cannot distinguish “zero” from “unknown”. Section 7.5 measures what happens when channels that should carry values are zero at inference time.

Year symmetry. Training contrasts a year without reported bleaching (2024) with a year with bleaching (2025), so any channel that exists in only one of the two years can identify the year and, through it, the label. The pipeline therefore writes values only for dates covered in both years and leaves the rest at zero in both. For example, the current record begins on 15 March, so the three current channels are zero from 1 January to 14 March in both years; in the training patches their zero fraction is exactly 0.200 (73 of 365 days), which confirms that the rule was applied. Daily biogeochemistry from CMEMS is available for both years but was not used.

A known violation: backward fill. The Gold step also contains a second pass that fills the days before a year’s first survey with that survey’s value, which uses information from the future. This pass affects samples whose targets fall in January and February: 6,828 of the 30,576 samples (22.3%). All positive labels fall in October and November, so the affected samples are negatives, but the inputs are not reproducible in real time. The operational path uses forward fill only; retraining without the backward pass has not yet been done.

## 4 Labels

## 4.1 The Bleaching Label Calibrator

Official bleaching records from the National Institute of Fisheries Science (NIFS) consist of a report date, a place name and a severity grade. We used 31 such reports from 2010 to 2025, obtained with the cooperation of the NIFS seaweed research division. Point-in-time reports are a poor training target for three reasons: the report date marks when a person noticed the damage, which can lag the onset by one to two weeks; a binary label cannot express gradual progression; and reports are too rare to train a model on their own. BLC converts each report into a spatiotemporal field of soft labels, severity grades and confidences. For a report e at time $t _ { e }$ and location $g _ { e }$ it proceeds as follows.

1. Window. Consider days $t \in [ t _ { e } - 1 4 , t _ { e } + 1 4 ]$ and pixels $p$ in a buffer around $g _ { e }$ . Place names are converted to coordinates, projected to EPSG:5179 and matched to the nearest grid cell; we recommend averaging over a $3 \times 3$ neighbourhood rather than using a single pixel.

2. Six-factor score. Compute a weighted environmental anomaly score

$$
s ( t , p ) = \sum _ { k = 1 } ^ { 6 } w _ { k } \nVdash \bigl [ \phi _ { k } ( t , p ) \bigr ] , \qquad \sum _ { k } w _ { k } = 1 0 0 ,\tag{1}
$$

single-channel AUC (window mean, direction-agnostic)  
![](images/d75986fd05247f2993ffe2c13397901fca65e9dae83e9a759c64901268d3c243.jpg)  
Figure 3: Single-channel AUC of the 14-day window mean on the validation set $( 2 , 7 9 2$ samples, 176 positives), coloured by information source. The calendar and season-correlated channels rank near the top, which motivates the same-date control in Section 7.3.

where the conditions $\phi _ { k }$ and weights $w _ { k }$ are listed in Table 3.

3. Peak normalization and time decay. Normalize by the maximum score in the window and multiply by a linear decay with a floor:

$$
r ( t , p ) = \frac { s ( t , p ) } { \operatorname* { m a x } _ { t ^ { \prime } , p ^ { \prime } } s ( t ^ { \prime } , p ^ { \prime } ) } , \qquad \delta ( t ) = \operatorname* { m a x } \Bigl ( 0 . 3 , 1 - \frac { | t - t _ { e } | } { 1 4 } \Bigr ) , \qquad y ( t , p ) = r ( t , p ) \delta ( t ) \in [ 0 , 1 ] .\tag{2}
$$

4. Severity. Grade 0 (normal) if $y < 0 . 3 , 1$ (caution) if $0 . 3 \leq y < 0 . 5 , 2$ (warning) if $0 . 5 \leq y < 0 . 8$ , and 3 (severe) if $y \geq 0 . 8$

5. Confidence. $c ( t ) = 0 . 9 5$ for $| t - t _ { e } | \leq 3 , 0 . 8 5 \mathrm { f o r } \leq 7 , 0 . 7 0 \mathrm { f o r } \leq 1 4 $ , and 0.50 outside the window.

6. Storage. Write (y, c, grade) into a label cube aligned with the OceanTensorCube. Outside all event windows, $y = 0$ with low confidence.

The BLC design motivates the ±14-day window by three time scales: the one- to two-week lag between onset and report, an assumed one- to two-week physiological response of the blades, and the roughly two-week spring–neap tidal cycle. These are design assumptions rather than measured quantities. Figure 4 shows the decay and confidence schedules. BLC is designed to supervise a model through a confidence-weighted regression loss, $\mathcal { L } _ { \mathrm { m a i n } } = \mathrm { m e a n } \left( c \cdot ( \hat { y } - y ) ^ { 2 } \right)$ with an optional masked-reconstruction auxiliary term weighted 0.3.

Applied to the 2024 Namhae coast as a first test (environmental scoring with event confidences), BLC produced 122 labels: 85 normal, 30 caution, 7 warning and none severe, with soft labels mostly between 0.1 and 0.4 and confidences of 0.70, 0.85 and 0.95. In development experiments a confidence-weighted loss was used with confidences assigned by evidence type (0.95 for official reports, 0.80 for field confirmation, 0.50 for environmental estimates) so that “not reported” is not learned as “did not occur”.

Table 3: BLC six-factor weak-label score. Thresholds are relative to the regional normal; the weights are initial values intended to be tuned by ablation.
<table><tr><td>Factor</td><td>Channel</td><td>Condition  $\phi _ { k }$ </td><td>Wk</td></tr><tr><td>DIN deficit</td><td>ch1</td><td>below the regional lower quartile, exponentially weighted over 10 days</td><td>30</td></tr><tr><td>DIP deficit</td><td>ch2</td><td>below the regional lower quartile, exponentially weighted over 10 days</td><td>15</td></tr><tr><td>N:P imbalance</td><td>ch4</td><td>&lt; 10 (nitrogen-limited) or &gt; 30 (phosphorus-limited)</td><td>15</td></tr><tr><td>SST anomaly</td><td>ch12</td><td> $\mathrm { o u t s i d e \pm 2 ^ { \circ } C }$  of normal during the growing season</td><td>15</td></tr><tr><td>Short-term chlorophyll drop</td><td>ch28</td><td>7-day mean at least 30% below the 14-day mean</td><td>15</td></tr><tr><td>Rain deficit</td><td>ch14</td><td>more than 14 days since rain</td><td>10</td></tr></table>

![](images/e15fea593c162b04a6a900c641e39fbcda7cf5e3107327a6f7793c05937caee0.jpg)  
Figure 4: BLC time rules around a reported event: the time-decay factor $\delta ( t )$ that multiplies the normalized six-factor score inside the ±14-day window, the confidence schedule c(t), and the cut-offs that map the soft label y to severity grades 0–3.

## 4.2 Labels used in this study

The classifier evaluated in Section 7 is trained and evaluated on binary cluster-day labels (version v7 of our label set), not on the BLC soft-label cube, and the training records do not state whether the final checkpoints used confidence weights. The positive days are recorded per cluster and date; our records do not trace each of them back to an individual report window. BLC is therefore presented as the labeling methodology of the platform, and its effect on forecast accuracy has not been isolated.

A sample is a pair (cluster, target date) with target dates from 21 January 2024 to 17 November 2025: 30,576 samples, of which 380 are positive (Table 4). Every positive lies in October or November 2025, and no negative does. The validation set consists of 8 clusters over both years, 2,792 samples with 176 positives (base rate 6.30%); all 92 false alarms reported below come from 2024.

This label structure has two consequences that shape the whole evaluation.

1. Calendar shortcut. The single rule “year is 2025 and month is October or November” separates all 380 positives from all 30,196 negatives (AUC 1.000). A model can score highly by learning the calendar, either directly from channels 25–26 or indirectly from any seasonal variable. Section 7.3 tests discrimination with the calendar held fixed.

2. Positive–unlabeled negatives. NIFS seaweed researchers told us that farms showing early discoloration often recover after timely fertilisation, and such cases do not enter official statistics. Negatives are therefore “not reported”, and the false-alarm rates we report are upper bounds; recall is measured only on reported events [4, 16].

Table 4: Composition of the labeled samples (left) and of the validation set (right).
<table><tr><td colspan="3">All samples</td><td colspan="4">Validation (8 clusters  $\times 2$  years)</td></tr><tr><td>Period</td><td>Positive</td><td>Negative</td><td>Year</td><td>Total</td><td>Positive</td><td>Negative</td></tr><tr><td>2024, Jan–Dec</td><td>0</td><td>18,144</td><td>2024</td><td>1,568</td><td>0</td><td>1,568</td></tr><tr><td>2025, Jan–May</td><td>0</td><td>12,052</td><td>2025</td><td>1,224</td><td>176</td><td>1,048</td></tr><tr><td>2025, 1–20 Oct (2 clusters)</td><td>40</td><td>0</td><td></td><td></td><td></td><td></td></tr><tr><td>2025, 1–17 Nov (20 clusters)</td><td>340</td><td>0</td><td></td><td></td><td></td><td></td></tr><tr><td>Total</td><td>380</td><td>30,196</td><td>Total</td><td>2,792</td><td>176</td><td>2,616</td></tr></table>

## 5 Model: ST-MMT

## 5.1 Task and window arithmetic

For a cluster k and target date τ , the model receives the 14 daily frames ending seven days before the target,

$$
{ \bf X } _ { k , \tau } = { \bf V } _ { k } [ \tau - 2 0 : \tau - 7 ] \in \mathbb { R } ^ { T \times C \times H \times W } , \qquad T = 1 4 , C = 3 2 , H = W = 6 4 ,\tag{3}
$$

where frame 0 is 1 January of the year, and outputs the probability $p _ { k , \tau }$ that the cluster is in a bleaching state on day τ . The forecast is issued on $\tau - 7 ,$ the last day of the input. We verified this arithmetic on every one of the 2,792 validation samples (input start = τ − 20, length $^ { 1 4 , }$ horizon 7, no exceptions). Getting it right was not trivial: in earlier development versions the target was set to the day after the input window, so those models were next-day predictors although they were described as seven-day predictors (Section 9).

## 5.2 Architecture

ST-MMT has two paths that see the same input (Figure 5). The joint path processes all 32 channels together and is the backbone. The group path splits the channels by physical origin, encodes each group separately, and fuses the groups with cross-attention. The group result refines the backbone through a gated residual connection. The design was chosen after a cross-attention-only variant underperformed a joint self-attention model at the recall-100% operating point (Section 9): separating channels into groups before any joint processing prevents the encoder from capturing interactions across groups at the first layer, while a purely joint model has no explicit mechanism for one modality to query another.

Patch embedding. Each frame is embedded by a strided convolution with kernel and stride 4,

$$
\mathbf { Z } _ { t } = \operatorname { C o n v 2 d } _ { C  d } ^ { 4 \times 4 , s = 4 } ( \mathbf { X } _ { t } ) \in \mathbb { R } ^ { N \times d } , \qquad N = ( 6 4 / 4 ) ^ { 2 } = 2 5 6 , d = 2 5 6 ,\tag{4}
$$

so each token covers $4 \times 4$ pixels $( 1 2 0 \mathrm { m } \times 1 2 0 \mathrm { m } )$ . Group encoders use the same embedding on their channel subsets.

Spatiotemporal blocks. An ST block applies spatial self-attention among the N tokens of each frame, then temporal self-attention along the T frames for each token position, then a position-wise feed-forward network, in the factorized manner of video transformers [1, 5]. Each attention is the multi-head scaled dot-product attention of Vaswani et al. [58] with h = 8 heads:

$$
\mathrm { A t t n } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = \mathrm { s o f t m a x } \Big ( \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d _ { h } } } \Big ) \mathbf { V } , \qquad d _ { h } = d / h = 3 2 ,\tag{5}
$$

$$
\mathrm { M H A } ( \mathbf { U } , \mathbf { W } ) = \left[ \mathrm { A t t n } ( \mathbf { U } \mathbf { W } _ { i } ^ { Q } , \mathbf { W } \mathbf { W } _ { i } ^ { K } , \mathbf { W } \mathbf { W } _ { i } ^ { V } ) \right] _ { i = 1 } ^ { h } \mathbf { W } ^ { O } .\tag{6}
$$

Self-attention is MHA(U, U). The joint encoder stacks four ST blocks; each group encoder stacks two.

![](images/045e96d32e39385d70b65c8f5c853935b14d80b2a1ee264bb977a483773a36aa.jpg)  
Figure 5: ST-MMT architecture (STMMTCrossAttnV3). A joint spatiotemporal encoder over all channels is refined by modalityspecific encoders fused through two rounds of multi-head cross-attention with the in-situ ocean group as the query hub. The refinement enters through a learnable gate α. The head outputs ${ \bf a ~ 1 6 } \times { \bf 1 6 }$ map of class probabilities; the forecast pack stores its spatial mean and maximum.

Modality groups. The 32 channels are divided as follows. Group A (in-situ ocean, 20 channels) contains the KOEM channels except suspended solids, the two current components and eight ocean-derived channels (N:P ratio, SST anomaly, SST 7-day and 30-day means, SST gradient, current speed, chlorophyll 7-day mean, mixed-layer depth). Group B (atmosphere, 7 channels) contains the six KMA channels and days since rain. Group C (satellite, 3 channels) contains suspended solids, the NIR index and its daily change; suspended solids sits here because the channel master originally assigned it a satellite source, although the code reads KOEM data (Table 2). The two calendar channels are spatially averaged and added to the group representation, in the manner of a positional encoding.

Hub cross-attention. Let $\mathbf { F } _ { A } , \mathbf { F } _ { B } , \mathbf { F } _ { C } \in \mathbb { R } ^ { T \times N \times d }$ be the group encodings. The group path applies two crossattention blocks with Group A as the query hub,

$$
\mathbf { H } _ { 1 } = \mathrm { C A } ( \mathbf { F } _ { A } ; \mathbf { F } _ { B } ) , \qquad \mathbf { R } = \mathrm { C A } ( \mathbf { H } _ { 1 } ; \mathbf { F } _ { C } ) , \qquad \mathrm { C A } ( \mathbf { U } ; \mathbf { W } ) = \mathrm { M H A } ( \mathbf { U } , \mathbf { W } ) ,\tag{7}
$$

so the queries come from the in-situ ocean representation and the keys and values from the atmosphere and then from the satellite stream. In the implementation the attention weights have shape $( B \cdot T , N , N )$ : at each time step, every ocean token attends over all spatial tokens of the other modality. Group A is the hub because in a gradient-saliency analysis of an earlier self-attention model most of the highest-ranked channels (precipitation, SST anomaly, N:P ratio, solar radiation, SST) belonged to Group A or to the closely related Group B. Fully pairwise bidirectional cross-attention among the three groups was tried and performed worse (Section 9).

Gated residual fusion and head. The two paths are combined as

$$
\mathbf { F } = \mathrm { L a y e r N o r m } \big ( \mathbf { F } _ { \mathrm { j o i n t } } + \alpha \mathbf { R } \big ) ,\tag{8}
$$

with a learnable scalar α initialized at 0.1. If the refinement is unhelpful, training can drive α toward zero and the model reduces to the joint backbone. The representation at the last time step is passed to a classification head with two output per token, giving a $1 6 \times 1 6$ map of logits<sup>1</sup>; the positive-class probability is $p = \operatorname { s o f t m a x } ( \cdot ) .$ . The forecast pack stores the spatial mean of p over the cluster window as the risk score and the spatial maximum as the local maximum risk.

Table 5: ST-MMT configuration as deployed $( \mathtt { x a t t n \_ v 3 \_ f o c a l \_ l d a y } )$
<table><tr><td>Input</td><td> $1 4 \times 3 2 \times 6 4 \times 6 4$ </td><td>Hidden size d</td><td>256</td></tr><tr><td>Horizon</td><td> $7 ~ \mathrm { d a y s }$ </td><td>Attention heads</td><td>8 4/2</td></tr><tr><td>Patch size</td><td> $4 \times 4 ( 2 5 6 \mathrm { t o k e n s } )$ </td><td>Joint / group ST blocks</td><td>A←B, then· ←C</td></tr><tr><td>Groups (A/B/C)</td><td> $2 0 / 7 / 3$  channels (+2 calendar)</td><td>Cross-attention Fusion</td><td> $\mathrm { L N } ( \mathbf { F } _ { \mathrm { j o i n t } } + \alpha \mathbf { R } )$ </td></tr><tr><td>Loss</td><td>focal,  $\gamma = 2 ,$  class weights 0.5/1.5</td><td></td><td></td></tr><tr><td>Ensemble</td><td>6 seeds, mean probability</td><td>Parameters</td><td> $9 , 2 1 8 , 1 7 9$ </td></tr></table>

Size. With $d = 2 5 6 , h = 8$ , four joint and two group layers per group, and patch size 4, the model has 9,218,179 parameters (a 37 MB checkpoint per seed). For comparison, the self-attention-only predecessor had 3,819,650 parameters and the first cross-attention-only variant 5,922,690.

## 5.3 Training

The loss is the focal loss [42]

$$
\mathcal { L } _ { \mathrm { F L } } = - \alpha _ { y } ( 1 - p _ { y } ) ^ { \gamma } \log p _ { y } , \qquad \gamma = 2 , \quad ( \alpha _ { 0 } , \alpha _ { 1 } ) = ( 0 . 5 , 1 . 5 ) ,\tag{9}
$$

where $p _ { y }$ is the predicted probability of the true class, applied per output token, plus a spatial smoothness term on the output map; the class weights act as the focal α term. Gradients are clipped to norm 1.0. Inputs are standardized per channel with means and standard deviations computed on the training patches (standard deviations below $1 0 ^ { - 6 }$ are set to 1), and the statistics are stored with each checkpoint. The checkpoint is selected by F1 at the recall-100% threshold on the validation set, with early stopping (patience 10 for the adopted seven-day configuration). The optimizer, learning-rate schedule, batch size and the weight of the smoothness term are not recorded in the documents from which this paper was prepared, and we do not report them.

Six models are trained from different initializations (seeds 42, 1, 2, 3, 5 and 7), and the ensemble probability is the mean of their positive-class probabilities [39]. The seeds differ substantially at the operating point: individual recall-100% thresholds range from 0.0741 to 0.1247 and the false-alarm rate in 2024 from 5.5% to 18.0% among the seeds for which it was recorded, against 5.9% for the ensemble. Averaging is therefore part of the method, not a refinement.

## 6 Evaluation protocol

Sample-level counting. Every metric counts samples (cluster-days), not pixels. Several of our early development reports counted pixels in the output map, which multiplied each sample by up to 4,096 and made small validation sets look large; those numbers are not comparable with the ones reported here.

Recall-first operating threshold. Because a missed event is the costlier error, the operating threshold is the highest threshold that still detects every reported positive in the validation set,

$$
\theta ^ { \star } = \operatorname* { m a x } \{ \theta : \mathrm { r e c a l l } ( \theta ) = 1 \} = \operatorname* { m i n } _ { i : y _ { i } = 1 } \bar { p } _ { i } - \epsilon .\tag{10}
$$

For the ensemble the smallest positive probability is 0.0941 and $\theta ^ { \star } = 0 . 0 9 4 0 . \mathrm { A t } \theta ^ { \star }$ we report precision, false-alarm rate (false positives over negatives), F1 and the Matthews correlation coefficient [10]. We also report threshold-free ROC-AUC and PR-AUC (average precision), reading PR-AUC against the base rate [50], and the maximum-F1 point for reference. Since $\theta ^ { \star }$ is fixed on the validation set, the 100% recall is an in-sample property of the threshold, not an out-of-sample guarantee.

Same-date control. To test whether the model separates clusters for reasons other than the calendar, we compute the AUC among clusters on the same date. Within one date every cluster shares the same calendar features, so the calendar contributes no information and the within-date AUC measures spatial discrimination only. In the labels used for the main model, no negative exists on any October or November 2025 date, so the control requires its own design. We built a separate label set in which, on the 37 dates with positive clusters, the other 75 clusters are labeled negative (3,515 samples: 380 positive, 3,135 negative), and trained the same architecture with five-fold cross-validation over clusters and three seeds (15 models). The metric is the mean within-date AUC of out-of-fold predictions over 17 November dates, with a cluster bootstrap (95 clusters, 2,000 resamples). The pass criterion (lower bound above 0.5) was fixed before measurement. This control tests the architecture and training procedure, not the deployed checkpoints.

Table 6: Validation accuracy of the six-seed ST-MMT ensemble (left) and the confusion matrix at the operating threshold $\theta ^ { \star } = 0 . 0 9 4 0$ (right).
<table><tr><td>Metric</td><td>Value</td><td></td><td>Predicted +</td><td>Predicted —</td></tr><tr><td>ROC-AUC</td><td>0.9940</td><td>Reported positive</td><td>176</td><td>0</td></tr><tr><td>PR-AUC (average precision)</td><td>0.9097</td><td>Negative (2024)</td><td>92</td><td>1,476</td></tr><tr><td>Recall at  $\theta ^ { \star }$ </td><td>100.0%</td><td>Negative (2025)</td><td>0</td><td>1,048</td></tr><tr><td>Precision at  $\theta ^ { \star }$ </td><td>65.7%</td><td>Negative, total</td><td>92</td><td>2,524</td></tr><tr><td>False-alarm rate at  $\theta ^ { \star \mathrm { a } }$ </td><td>3.5%</td><td></td><td></td><td></td></tr><tr><td>F1 at  $\theta ^ { \star }$ </td><td>0.7928</td><td></td><td></td><td></td></tr><tr><td>MCC at  $\theta ^ { \star \ b }$ </td><td>0.7960</td><td></td><td></td><td></td></tr><tr><td>Maximum F1 (recall 85.2%)</td><td>0.8721</td><td></td><td></td><td></td></tr></table>

<sup>a</sup> Upper bound: negatives are unreported days, not confirmed healthy (Section 4.2).  
<sup>b</sup> Computed from the confusion matrix.

Pre-registration. For the robustness experiments in Sections 7.5–8, the acceptance criterion was written down before the measurement and not changed afterwards: a degraded-input condition is “usable” only if detection stays at 100% and AUC drops by less than 0.02. When a criterion later proved to be badly posed, we recorded it as a criterion error rather than reinterpreting the result.

Baselines. A separate benchmark compared ST-MMT with three baselines trained on the same samples, normalization, loss (focal loss with class weights 0.5/1.5 and confidence weighting) and optimizer settings (Adam, learning rate $1 0 ^ { - 4 }$ with cosine annealing, gradient clipping 1.0, batch 16, at most 30 epochs, patience 8): a two-layer LSTM [25] with hidden size 256 applied to the spatial mean of each channel, giving a 14 × 32 sequence; a VGG16-configuration network [53], without pretrained weights, applied to the temporal mean of the input; and XGBoost [9] with 400 trees of depth 6 on 160 hand-crafted features (mean, standard deviation, trend, last value and spatial standard deviation for each of the 32 channels). Each baseline was trained with seeds 42, 1 and 7, and each model’s threshold was set by the same recall-100% rule on its own predictions. No model received a hyperparameter search.

## 7 Results

Except for the same-date control (Section 7.3), which comes from a separate cross-validated training, all results in this section are for the six-seed ensemble of xattn\_v3\_focal\_7day on the validation set of 2,792 samples (8 clusters, 2024–2025; 176 positives, base rate 6.30%). The same samples were used to select checkpoints and to set the operating threshold, so the figures describe fit to the validation set rather than performance on unseen data. Measurement dates are listed in Appendix B.

## 7.1 Overall accuracy

Table 6 summarizes accuracy. The ensemble ranks positives above negatives almost perfectly (ROC-AUC 0.9940), and its PR-AUC of 0.9097 is 14.4 times the base rate. At the operating threshold $\theta ^ { \star } = 0 . 0 9 4 0$ every reported positive is detected, 92 of 2,616 negatives are flagged (3.5%), and precision is 65.7%. Choosing the maximum-F1 threshold instead would raise F1 from 0.7928 to 0.8721 but miss 26 of the 176 positives (recall 85.2%), which is why we do not use it operationally.

All false alarms occur in 2024, the year without reported damage, and none among the 1,048 negatives of 2025. The most likely explanation is the season: every 2025 negative has a target date between January and May, whereas the 2024 negatives cover the whole year, including the autumn months in which the model raises its probabilities. A second possibility is that some 2024 alarms reflect real but unreported stress, which the positive–unlabeled structure cannot rule out. With two years of data we cannot separate these explanations.

![](images/512f0b29bd81c3e672f4ee117bf83b92c64f950736884fa55ab4c618b0e1b07e.jpg)  
Figure 6: Validation samples by ensemble probability band. Boundaries are the recall-100% threshold (0.0940) and the median of positive probabilities (0.1938); the early-warning and warning grades share the middle band (split at 0.1120).

Table 7: Same-date control: within-date AUC among 95 clusters on November 2025 dates (out-of-fold predictions of the cross-validated control training), where calendar features are identical for all clusters.
<table><tr><td>Replicateª</td><td>Within-date AUC</td><td>Bootstrap lower bound</td></tr><tr><td>1</td><td>0.9778</td><td>0.9466</td></tr><tr><td>2</td><td>0.9763</td><td>0.9429</td></tr><tr><td>3</td><td>0.9857</td><td>0.9621</td></tr></table>

<sup>a</sup> Our records label the three rows differently (seeds 1–3; a design that fixed seeds 42, 1 and 7; or three November date groups). The values are the same in all records.

## 7.2 Alert grades

The service shows four grades. Their boundaries are fixed by the distribution of validation positives rather than chosen by hand: $\theta ^ { \star } = 0 . 0 9 4 0$ separates normal from early warning, the lower quartile of positive probabilities (0.1120) separates early warning from warning, and the median (0.1938) separates warning from severe. (One internal report describes 0.1120 and 0.1938 as the median and upper quartile; the band counts below, with 88 of the 176 positives at or above 0.1938, confirm that 0.1938 is the median.) Figure 6 shows how the validation samples fall into the three resulting bands. No positive falls in the normal band, which holds 2,524 negatives. The severe band holds 88 positives and 13 negatives, and the middle band holds 88 positives and 79 negatives. The early band is only 0.018 wide, and in October 2025 a third of the forecasts fall into it; small changes of the threshold would therefore move many farms between grades, which is why the boundaries are pre-registered rather than tuned.

## 7.3 Same-date control

The calendar shortcut in Section 4.2 means that a high AUC alone could reflect the season rather than the ocean. Table 7 reports the within-date AUC of the separate cross-validated training described in Section 6: on November 2025 dates, across 95 clusters, out-of-fold predictions rank the clusters reported as bleached above the others with AUC 0.976–0.986, and all bootstrap lower bounds exceed 0.94, well above the pre-registered criterion of 0.5. With the calendar held fixed, the architecture trained under this design still separates clusters.

Three limits apply. First, the control was run on a separately trained set of models, not on the deployed ensemble, whose own calendar independence has not been measured; the deployed forecast pack still assigns the early-warning grade to about 40% of farms in September of both years (36.8% in 2024 and 44.3% in 2025), a residual seasonal baseline. Second, the control removes the calendar but not every confounder: on a fixed date, clusters still differ in persistent characteristics such as location and typical nutrient levels, and the 20 positive clusters may share such traits. Third, the clusters labeled negative on November dates are unreported, not confirmed healthy. The control therefore shows spatial discrimination, not that the model has learned the causal pathway to bleaching. It was introduced after we found that an earlier forecast pack behaved as a calendar switch (Section 7.6).

Table 8: Benchmark at each model’s own recall-100% threshold. Single-model rows are means over seeds; the ensemble row is computed from averaged probabilities.
<table><tr><td>Model</td><td>Parameters</td><td>ROC-AUC</td><td>PR-AUC</td><td>Precision</td><td>False alarm</td><td>MCC</td></tr><tr><td>XGBoost (3 seeds)</td><td>_a</td><td>0.9150</td><td>0.6288</td><td>14.62%</td><td>39.67%</td><td>0.2970</td></tr><tr><td>VGG16 (3 seeds)b</td><td>39,913,602</td><td>0.9810</td><td>0.7704</td><td>41.49%</td><td>36.44%</td><td>0.7505</td></tr><tr><td>LSTM (3 seeds)</td><td>823,810</td><td>0.9953</td><td>0.9487</td><td>60.15%</td><td>4.74%</td><td>0.7550</td></tr><tr><td>ST-MMT (6 single seeds)</td><td>9,218,179</td><td>0.9907</td><td>0.8583</td><td>48.71%</td><td>7.67%</td><td>0.6681</td></tr><tr><td>ST-MMT (6-seed ensemble)c</td><td>6×9,218,179</td><td>0.9940</td><td>0.9096</td><td>65.67%</td><td>3.52%</td><td>0.7960</td></tr></table>

<sup>a</sup> About 1,000 tree leaves per model.  
<sup>b</sup> One seed assigned a near-zero probability to one positive, so its recall-100% threshold fell to 0 and every sample was flagged (MCC undefined); the MCC shown averages the other two seeds  
<sup>c</sup> PR-AUC 0.9096 here against 0.9097 in Table 6, from a separate evaluation of the same checkpoints.

![](images/dfb1ab148446b79fcc32f2347502c9e238b36fdd4127f3180ffdcf7ad30a742c.jpg)  
Figure 7: Matthews correlation coefficient at each model’s recall-100% threshold (benchmark of 7 October 2026). Single-model bars are means over seeds; the VGG16 bar excludes one seed whose threshold collapsed to zero.

## 7.4 Comparison with baselines

Table 8 reports the benchmark run of 7 October 2026 on the same 2,792 validation samples (training: 27,784 samples, 204 positives, 87 clusters). Each baseline was trained with three seeds; its row is the mean of the per-seed values, each computed at that seed’s own recall-100% threshold. ST-MMT is shown both as the mean of its six single-seed checkpoints and as the six-seed ensemble.

Figure 7 summarizes the operating-point comparison. The result does not favour ST-MMT as a single model. Model for model, the LSTM on spatially averaged channels is better on every metric: higher ranking accuracy (ROC-AUC 0.9953 against 0.9907, PR-AUC 0.9487 against 0.8583) and, at the operating point, higher precision (60.15% against 48.71%), fewer false alarms (4.74% against 7.67%) and a higher MCC (0.7550 against 0.6681). The ST-MMT ensemble has the best operating-point figures in the table (precision 65.67%, false alarms 3.52%, MCC 0.7960), but it is compared with the average of single LSTMs, which is not a like-for-like comparison, and the best single LSTM (seed 1: precision 73.6%, false alarms 2.4%, MCC 0.848) exceeds it. The variation between seeds is large for both families: single-seed ST-MMT false-alarm rates range from 3.7% to 10.8%, and single-seed LSTM rates from 2.4% to 6.6%.

One feature of the LSTM runs calls for caution before reading the comparison as evidence that space does not matter. In all three seeds the training loss fell to about zero within the first two epochs, and two seeds reached their best validation score in the first epoch. The LSTM receives the spatial means of all channels, including the month encodings, and the labels can be separated perfectly by year and month (Section 4.2). A model that learns the season will look excellent on this validation set. We have not tested whether the LSTM relies on the calendar. A pre-registered follow-up trains the baselines as six-seed ensembles, compares ensembles with bootstrap intervals, and submits the baselines to the same-date control of Section 7.3; it will be reported whatever its outcome. Until then, the defensible reading is that the current labels do not demonstrate an advantage of the spatiotemporal transformer over a temporal model of cluster averages, and that the transformer’s practical additions are the within-cluster probability map and an architecture in which modalities can interact explicitly.

Table 9: Experiment A: effect of degrading nine channels (ch10–13, 24, 27–30). Pre-registered criterion: detection 100% and AUC $\mathrm { d r o p } < 0 . 0 2$
<table><tr><td>Condition</td><td>ROC-AUC</td><td>Detection</td><td>False alarm</td><td>False alarm 2024</td><td>Verdict</td></tr><tr><td>Original</td><td>0.9940</td><td>100.0%</td><td>3.5%</td><td>5.9%</td><td>reference</td></tr><tr><td>Nine channels flattened in time</td><td>0.9943</td><td>100.0%</td><td>3.9%</td><td>6.4%</td><td>usable</td></tr><tr><td>Nine channels set to zero</td><td>0.9778</td><td>88.6%</td><td>3.1%</td><td>5.2%</td><td>not usable</td></tr></table>

## 7.5 Robustness to missing channels

In operation some channels arrive late or not at all. Experiment A simulated this for nine channels that depend on separate collection steps (currents u, v and speed; SST anomaly; SST 7- and 30-day means; chlorophyll 7-day mean; NIR index and its daily change). Two failure modes were compared: values frozen in time (flattened) and values set to zero (Table 9).

Frozen values are harmless, but zeros are not: 20 of the 176 positives are missed. The lower false-alarm rate in the zero condition is not an improvement; the model simply outputs lower probabilities overall. This matters directly for operation, because the 2026 inference patches checked on 7 October 2026 had exactly ten channels at zero (the nine above plus mixed-layer depth). The cause was a build script that skipped several processing steps, not missing data, and the failure would have recurred on any machine (Section 10). A follow-up recorded in our nutrient-estimation log reports AUC 0.9768 and detection 88.6% with all ten channels at zero, and AUC 0.9850 with 100% detection after restoring the moving averages, currents and mixed-layer depth; our training report still lists these conditions as unmeasured, so we cite them only as preliminary.

## 7.6 Inference port and forecast pack

The service does not run the network. A stand-alone inference module produces probabilities that are written into a forecast pack, and the web service reads the pack. We compared the stand-alone module with the training code on all 2,792 validation samples: the maximum absolute difference was $2 . 3 \times 1 0 ^ { - 6 }$ and no sample differed by more than $1 0 ^ { - 4 }$

Auditing the forecast pack that the service used before regeneration, which carried the same model name but was produced by a different generator, revealed a calendar switch (Table 10). Within a month, almost every farm received the same grade: 100% severe in September 2024 and 92.5% in October 2024, although 2024 was a year without reported damage, and on 1 November 2025 the risk scores of eight clusters spanned a range of only 0.013. The regenerated pack from the ensemble has no severe grades in any month of 2024, concentrates severe grades in October–December 2025 (15.0%, 43.8% and 24.7% of farms), and spans a range of 0.509 on the same date. Regeneration also fixed three defects in the original generator: dates were collapsed because only the last row per tile was kept, the “severe” field was a fixed multiple (1.2×) of the mean probability instead of a model output, and grade boundaries were arbitrary cut-offs on ten times the probability. The pack now covers 469 dates (21 January 2024 to 31 December 2025) and 1,109 farms; the grade distribution is 82.9% normal, 9.9% early, 1.6% warning and 5.6% severe, and 136 farms are severe on 14 October 2025.

One check from the regenerated pack remains open: 24.7% of farms are graded severe in December 2025, and we have not yet compared this with damage records for that month.

Table 10: Forecast pack before and after regeneration with the ensemble.
<table><tr><td>Check</td><td>Previous pack</td><td>Regenerated pack</td></tr><tr><td>Share severe, September 2024</td><td>100%</td><td>0.0%</td></tr><tr><td>Share severe, October 2024</td><td>92.5%</td><td>0.0%</td></tr><tr><td>Share severe, November 2025</td><td>10.9%</td><td>43.8%</td></tr><tr><td>Share of “early” grade</td><td>0.0% (all 17 months)</td><td>9.9%</td></tr><tr><td>Range across clusters, 1 Nov 2025</td><td>0.013</td><td>0.509</td></tr></table>

Table 11: Most recent nutrient observation per source on 7 October 2026.
<table><tr><td>Source</td><td>Latest nutrient value</td><td>Age (days)</td><td>Note</td></tr><tr><td>KOEM monitoring network</td><td>28May 2026</td><td>132</td><td>August survey not yet published</td></tr><tr><td>NIFS serial oceanographic stations</td><td>16 Dec 2025</td><td>295</td><td>no 2026 nutrient records</td></tr><tr><td>NIFS fishing-ground stations</td><td>5 Nov 2025</td><td>336</td><td>sparse responses</td></tr><tr><td>NIFS real-time stations</td><td></td><td></td><td>temperature only</td></tr><tr><td>CMEMS biogeochemistry (model)</td><td>5 Oct 2026</td><td>2</td><td>daily; not used in training</td></tr></table>

## 8 The nutrient gap in operation

DIN is the nutrient most closely associated with bleaching in the literature [26, 47], and it is the input that is hardest to obtain in real time. DIN is not measured directly: it is the sum of three separately analysed species,

$$
\mathrm { D I N } = \mathrm { [ N O _ { 3 } ^ { - } - N ] } + \mathrm { [ N O _ { 2 } ^ { - } - N ] } + \mathrm { [ N H _ { 4 } ^ { + } - N ] } ,\tag{11}
$$

so a DIN value exists only for days on which water was sampled and analysed. This section quantifies the gap and evaluates the obvious ways of bridging it.

## 8.1 How old is the newest nutrient observation?

Table 11 lists, for each source we ingest, the most recent nutrient observation available on 7 October 2026. KOEM surveys are quarterly (February, May, August and November). The late-August surveys of 2024 and 2025 are in our database; the 2026 August survey was not available, and manual queries for four sea areas and two periods returned no records and no errors, which points to delayed publication rather than a failure of our collector. The freshest usable nutrient value was therefore 132 days old. The NIFS station networks had no 2026 nutrient records, and the NIFS real-time stations report temperature only. CMEMS daily biogeochemistry is current but is a model product without ammonium, and it was not used to train the model.

## 8.2 Cost of stale nutrients

Experiment B froze the eleven KOEM channels at their values N days before each input window and re-evaluated the ensemble (Figure 8 and Table 12). Two reading rules apply. First, the training data already contain a median natural staleness of 17 days, so the effective age is about N + 17. Second, the number of evaluable samples falls from 2,537 to 820 as N grows, and the composition changes; detection rates at 129 and 180 days are therefore not interpretable, and only AUC and false-alarm rate are reported for those rows.

Under the pre-registered criterion not even the 30-day condition is acceptable: AUC drops by 0.0535 and detection falls to 97.7%. At 129 days the AUC (0.5267) is indistinguishable from chance, and at 180 days it is below 0.5. Two practical conclusions follow. The 60-day freshness limit used by our output gate (Section 8.6) has no empirical basis, because every measured condition fails; a defensible limit requires measurements at 7, 14 and 21 days, which we have not yet made. And during the months when the newest survey is three to four months old, the model should not issue nutrient-dependent alerts at all.

![](images/79c83a6236828fc3ff9ecf4d1dea75d3b0ddcfdd8d70ce589396c6d3bc54150f.jpg)

![](images/0f3c5b9765f34263b924d1860dd28bc2e30d8b3b6c741f3b3441248408578a76.jpg)  
Figure 8: Experiment B: ROC-AUC (left) and false-alarm rate (right) when the eleven KOEM channels are frozen N days before the input window. At 129 days, close to the 132-day age of the freshest nutrient value on 7 October 2026, discrimination is close to chance.

Table 12: Experiment B in detail. Detection rates for 129 and 180 days are omitted because the evaluable sample changes composition.
<table><tr><td>N (days)</td><td>Samples</td><td>ROC-AUC</td><td>Detection</td><td>False alarm</td><td>False alarm 2024</td></tr><tr><td>0</td><td>2,792</td><td>0.9940</td><td>100.0%</td><td>3.5%</td><td>5.9%</td></tr><tr><td>30</td><td>2,537</td><td>0.9405</td><td>97.7%</td><td>12.1%</td><td>19.7%</td></tr><tr><td>60</td><td>2,087</td><td>0.8333</td><td>67.6%</td><td>24.4%</td><td>37.6%</td></tr><tr><td>90</td><td>1,637</td><td>0.6778</td><td>87.5%</td><td>42.7%</td><td>60.6%</td></tr><tr><td>129</td><td>1,052</td><td>0.5267</td><td>一</td><td>67.4%</td><td>78.0%</td></tr><tr><td>180</td><td>820</td><td>0.3964</td><td></td><td>84.9%</td><td>84.9%</td></tr></table>

## 8.3 Decomposing DIN for buoy measurements

A buoy with nitrate and ammonium sensors, but no nitrite sensor, is planned for a partner farm in Goheung. To estimate the bias of $\mathrm { N O _ { 3 } + N H _ { 4 } }$ as a DIN measurement, we analysed 24,513 NIFS fishing-ground samples in which all three species were reported (Table 13). The sum of the three species matches the reported DIN (median relative error 0.0000; 5.2% of samples differ by more than 10%), so the nitrite share can be trusted. Nitrite contributes a median 5.26% of DIN (mean 8.07%, 95th percentile 23.84%). If nitrite is a fraction $f$ of DIN, then $\mathrm { D I N } = \mathrm { ( N O _ { 3 } + N H _ { 4 } ) } / ( 1 - f )$ Once the buoy is deployed, we plan to correct its readings with the median factor $1 / ( 1 - 0 . 0 5 2 6 ) = 1 . 0 5 5$ and to use the 95th-percentile factor $1 / ( 1 - 0 . 2 3 8 4 ) = 1 . 3 1 3$ for decisions near alert thresholds, where under-estimation would suppress alerts. (Our internal report rounds the conservative factor to 1.24, i.e. 1 + 0.2384, which under-corrects.) For scale, KOEM DIN values $( n = 2 3 , 5 9 7 )$ have mean 138.41, median 76.57 and 90th percentile 314.02 µg/L.

## 8.4 Persistence versus climatology

Forward filling a 132-day-old May value into October assumes that DIN has not changed, an assumption that had never been tested. We compared it with a station climatology using pairs of consecutive KOEM observations at 425 stations from 2011 to 2026 (22,736 pairs). For a pair $\left( { t _ { k } , t _ { k + 1 } } \right)$ at station s, persistence predicts $D ( s , t _ { k + 1 } ) = D ( s , t _ { k } )$ , and climatology predicts

$$
C ( s , d ) = \frac { 1 } { | \mathscr { y } | } \sum _ { y \in \mathscr { y } } \operatorname * { m e a n } \big \{ D ( s , t ) : \operatorname { y e a r } ( t ) = \mathscr { y } , | \mathrm { d o y } ( t ) - d | _ { \circ } \leq 1 5 \big \} ,\tag{12}
$$

where d is the day of year of $t _ { k + 1 } , \mid \cdot \mid _ { \circ }$ is the circular day-of-year distance, and Y excludes the evaluation year (leave-one-year-out) and must contain at least three years. The pre-registered criterion was at least 100 pairs and an MAE improvement of at least 20%.

Table 13: Share of nitrite in DIN (NIFS fishing-ground samples, $n = 2 4 { , } 5 1 3 )$
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>NO2 share, median / mean / 95th percentile</td><td>5.26% / 8.07% / 23.84%</td></tr><tr><td>Median relative error, sum of species vs. reported DIN</td><td>0.0000</td></tr><tr><td>Samples with relative error  $> 1 0 \%$ </td><td>1,278 (5.2%)</td></tr><tr><td>Correction factor  $1 / ( 1 - f )$  for  $\mathrm { N O _ { 3 } + N H _ { 4 } }$  (median / 95th percentile)</td><td>1.055 / 1.313</td></tr></table>

![](images/edb5b8df7dc4330e6fe737658c475a90933b0ad5301f3e06eaaca27d9d01c353.jpg)  
Figure 9: DIN prediction error of persistence (forward fill) and leave-one-year-out climatology for consecutive KOEM observations, grouped by the gap between observations. Labels give the relative reduction in MAE achieved by climatology.

Climatology won in every gap bucket, by 26.1% to 35.7% (Figure 9); at the current gap of 105–135 days the MAE fell from 86.03 to $5 7 . 2 6 \mu \mathrm { g / L }$ (33.4%). The persistence error at that gap is larger than the median DIN value itself $( 7 6 . 5 7 \mu \mathrm { g } / \mathrm { L } )$ , which is consistent with the near-chance AUC of Experiment B. This is, however, an error in DIN, not in bleaching detection. Climatology varies less than forward-filled values and may lie outside the distribution the model saw in training, and it does not change from day to day. We have not yet measured detection accuracy with climatological inputs, and we have not changed the operational input.

## 8.5 A rejected proxy: nutrient drift from daily variables

The Nutrient Drift Estimator (NDE) tried to fill the gap with change rather than level: find a channel that is observed daily and whose changes track changes in DIN, and propagate that change,

$$
\widehat { D } ( t ) = D ( t _ { k } ) + \beta \big [ P ( t ) - P ( t _ { k } ) \big ] ,\tag{13}
$$

where $t _ { k }$ is the last survey and P is the proxy. We first separated candidate channels by whether they are measured in the same survey as DIN; same-survey channels such as nitrate, phosphate and silicate track DIN closely (change correlations 0.71–0.95) but go stale together with it, so they cannot fill the gap. Among daily channels, the pre-registered criterion required an absolute change correlation of at least 0.30 and a top-three rank in both 2024 and 2025.

The three best daily candidates in 2024 were air temperature and the two month encodings (Table 14), that is, the season: DIN is high in winter and low in summer. In 2025, the year with bleaching, their correlations collapsed to −0.089, 0.168 and 0.046. Without the requirement that a proxy rank highly in both years, we would have adopted air temperature on the strength of 2024 alone, and it would have failed in the year that mattered. Two follow-up variants were also rejected under their pre-registered criteria, according to our nutrient-estimation log: summary statistics over the gap interval (for every one of 103 combinations, at least one of the two directions increased the error; the best combination, the interval maximum of the wind-direction cosine, improved one direction by 0.7% and worsened the other by 0.8%), and ridge regression of log DIN on same-day ocean-model and calendar fields, which was worse than climatology by 244.6% (2024→2025) and 55.0% (2025→2024). Our training report of the same date still lists the interval-summary measurement as pending, so these two results have not been cross-checked. One observation from this search is worth keeping: level and change correlations can differ sharply (ammonium: 0.126 for level and 0.596 for change; dissolved oxygen: 0.019 and −0.585), so external sources should not be dismissed on level correlation alone.

Table 14: Change correlation with DIN for the leading candidates. The daily candidates that ranked highest in 2024 are seasonal variables whose correlation collapses in 2025.
<table><tr><td>Group</td><td>Channel</td><td>2024</td><td>2025</td></tr><tr><td rowspan="3">Same survey (cannot fill the gap)</td><td>NO3 (ch21)</td><td>0.945</td><td>0.894</td></tr><tr><td>DIP (ch2)</td><td>0.800</td><td>0.809</td></tr><tr><td>SiO2 (ch3)</td><td>0.710</td><td>0.842</td></tr><tr><td rowspan="3">Daily (candidates)</td><td>Air temperature (ch19)</td><td>-0.501</td><td>-0.089</td></tr><tr><td>Month cosine (ch26)</td><td>0.461</td><td>0.168</td></tr><tr><td>Month sine (ch25)</td><td>0.345</td><td>0.046</td></tr></table>

Table 15: Harness Gate applied to every forecast before display.
<table><tr><td>Stage</td><td>Rule</td><td>Default</td></tr><tr><td>1. Freshness</td><td>block if the newest nutrient observation is older than the limit</td><td>60 days</td></tr><tr><td>2. Data quality</td><td>require declared valid channels; otherwise mark the forecast “unknown&quot;</td><td></td></tr><tr><td>3. Persistence</td><td>raise a grade after consecutive exceedances, lower it after consecutive clear days</td><td>2 / 5 / 7 days</td></tr></table>

## 8.6 The output gate

Forecasts pass through a three-stage gate before they are shown, and each stage fails closed, meaning that when the system cannot decide, it does not alert (Table 15).

With the newest nutrient value 132 days old, stage 1 blocks every farm. Given Experiment B, this is the correct behaviour: at that age the false-alarm rate is 43–67%, and repeated false alarms would teach users to ignore the system. The service response carries the age of the nutrient data, and the pack declares its own alert threshold (0.0940), which overrides the gate’s default persistence threshold of 0.30. The service also refuses to serve a pack whose model name does not match the deployed model (HTTP 503) rather than fall back silently to an older model.

## 9 Development history and lessons

The deployed model is the result of more than twenty training configurations. Their metrics are not comparable with the results above: early runs used a single 2021 tile (Wando) and later two tiles (Wando and Mokpo), counted pixels rather than samples, and, until a code review found the error, predicted the next day rather than seven days ahead. We summarize the lessons rather than the numbers; Appendix A lists the key configurations with these caveats.

Diagnose suspiciously good results. A model trained on the Mokpo tile alone reached ROC-AUC and PR-AUC of exactly 1.0000. We treated this as an artifact rather than a success; it persisted after removing a suspect channel and was not used. The same attitude found the N:P ratio bug (values up to 2.65 × 10<sup>6</sup> that dominated the saliency map) and a suspended-solids channel whose apparent importance came from having only four to seven distinct values per year.

The 2021 South Jeolla experiments. Development began with a single 2021 tile at Wando, where an official record placed a bleaching event from 6 September to 26 October 2021 (51 days); with the remaining 314 days labeled normal, one tile-year gave 345 sliding-window samples (300 for training, 45 for validation). Mokpo was added as a second tile, and Jindo and then Goheung (with a December 2016 event) as third and fourth. Performance fell as tiles were added: ROC-AUC 0.9071, 0.8378 and 0.7790 and PR-AUC 0.8522, 0.8091 and 0.7212 for two, three and four tiles (validation sets of 90, 135 and 180 samples). Tracing the drop exposed data problems rather than a modelling limit. The tile centres for Jindo and Goheung lay 19.65 km and 23.78 km from the actual farms, outside a tile that is 15.36 km wide, against 4.53 km for Wando and 6.15 km for Mokpo; the Goheung tile had only 28 of 32 channels, with no satellite data before

Table 16: Retraining against the seasonal confound (8 external clusters, three seeds, mean ± s.d.). The ratio is the mean probability in November 2025 divided by that in June–August; alarm days count 2024 days above an alert threshold, which was 0.30 for the second row and 0.50 for the third (at 0.30 the third configuration has 30 alarm days).
<table><tr><td>Configuration</td><td>ROC-AUC</td><td>F1 (recall 100%)</td><td>Nov./summer ratio</td><td>Alarm days 2024</td></tr><tr><td>Original patches (22 valid channels)</td><td>0.9825</td><td>0.7756</td><td></td><td></td></tr><tr><td>Moving averages and currents restored (29)</td><td> $0 . 9 9 2 2 \pm 0 . 0 0 5 3$ </td><td> $0 . 8 6 5 6 \pm 0 . 0 8 0 6$ </td><td>1.41</td><td>132</td></tr><tr><td>Empty clusters excluded, SST anomaly and MLD</td><td> $0 . 9 8 2 0 \pm 0 . 0 0 6 7$ </td><td> $0 . 7 6 4 7 \pm 0 . 0 5 6 0$ </td><td>9.42</td><td>0</td></tr><tr><td>KOEM nutrients replaced by CMEMS</td><td> $0 . 9 3 5 5 \pm 0 . 0 0 9 2$ </td><td> $0 . 6 9 2 9 \pm 0 . 0 1 3 6$ </td><td></td><td></td></tr></table>

2020, most solar-radiation values missing and no ocean-model currents for January–April 2016; and the Jindo tile lacked mixed-layer depth. The project subsequently moved to the national register of licensed farm polygons, cluster tiles built around those farms, and the 2024–2025 two-year design used in the deployed model. The 2021 numbers were counted in pixels and, as noted above, predicted the next day, so they are not comparable with Section 7; Appendix A lists them.

Cross-attention alone was not enough. The first cross-attention model, with the three groups encoded separately and fused through the Group A hub, improved ROC-AUC over its self-attention predecessor (0.9191 against 0.8716 on the two-tile, next-day task) but lowered F1 at the recall-100% threshold (0.8013 against 0.8953) and precision (75.36% against 84.75%). Adding a fourth nutrient group did not help (F1 0.7786), and neither did fully pairwise bidirectional cross-attention (0.7343), forcing the residual gate to start at 1.0 (0.8507) or hard-negative mining (0.8059). Hard-negative mining on the four-group model reached F1 0.8985 but lowered its probability for an external event to 8.2%, so it was discarded; the residual design with focal loss was the first configuration that exceeded the self-attention model at the operating point and was retained (F1 0.9271, precision 88.74%). In the residual variant trained before focal loss was introduced, the learned gate ended at $\alpha = 0$ .105 from an initial 0.1, indicating that the cross-attention path contributed little in that setting. We have not recorded the gate value of the deployed checkpoints, and we have not run an ablation that removes the group path from the final configuration. The contribution of cross-attention to the results in Section 7 is therefore not isolated, and we regard measuring it as the most important open experiment for the architecture.

The seasonal confound and the retraction of an external validation. An external check on the Seocheon coast in 2025 initially appeared to catch a large event on 14 October. An annual inference curve later showed that the model of that period assigned probabilities of 0.70–0.80 in July and August but 0.11 in November and December, and that its output correlated most strongly with the 30-day SST mean (0.843), the month sine (−0.812) and air temperature (0.802). It had learned the season. We withdrew the external validation and retrained with a design that contrasts the same months of a year without reported damage (2024) with the event year (2025). Table 16 summarizes the retraining on eight external clusters. Restoring channels improved AUC but left a strong summer response (November/summer ratio 1.41 and 132 alarm days in 2024); excluding clusters without observations and restoring SST anomaly and mixed-layer depth gave a ratio of 9.42 and no alarm days in 2024 at a threshold of 0.50 (30 days at 0.30); replacing KOEM nutrients with CMEMS model nutrients lowered AUC to 0.9355. Since then an annual inference curve has been a required check before any model is adopted, because validation metrics cannot reveal seasonal dependence when no labels exist for the summer months.

Separate ranking metrics from operating metrics. F1 at the recall-100% threshold depends on the single lowestscoring positive and therefore varies widely between seeds (0.8087 to 0.9578 for one configuration). We use AUC to compare models and report precision, false alarms and F1 at the operating point to describe how a model will behave in service.

## 10 Limitations

• Two years and a small validation set. Training contrasts one year without reported bleaching with one year with bleaching. The validation set covers 8 clusters and was also used to select checkpoints and the threshold; the false-alarm rate of 3.5% is measured on those clusters only and need not hold for all 105. There is no separate test set.

• Report-based, positive–unlabeled labels. All positives fall in October and November 2025, so the year and month alone separate the labels. The same-date control was run on a separately trained set of models; the calendar independence of the deployed ensemble is not measured, and its pack still shows a raised early-warning share in September of both years. False-alarm rates are upper bounds.

• Non-causal backward fill. 22.3% of the training samples (January–February targets) contain values back-filled from the first survey of the year. The operational path uses forward fill only, and retraining without the backward pass is pending.

• Cross-attention not isolated. The contribution of the cross-attention path in the deployed configuration has not been measured by ablation, and the BLC soft labels were not the training target of the evaluated model.

• Baselines. A single LSTM on cluster-average series outperforms a single ST-MMT on every benchmark metric, and only the ST-MMT ensemble leads at the operating point, against single-seed baselines. Ensemble-to-ensemble comparison and a same-date control for the baselines are pre-registered but not yet run.

• Nutrient staleness. Skill decays steeply with the age of the nutrient survey (Section 8.2), and the 2026 August survey had not been published at the time of writing. The gate’s 60-day limit is not yet grounded in measurement.

• 2026 inputs. The 2026 inference patches checked on 7 October had ten channels at zero because the build skipped several processing steps. The moving averages, currents and mixed-layer depth can be restored from stored data; the SST anomaly needs its multi-year baseline, whose location must still be confirmed; the satellite NIR channels require new collection. Preliminary measurements of the restored conditions (Section 7.5) have not been cross-checked.

• Unmeasured items. Inference speed on CPU or edge hardware, the accuracy of a distilled model, detection accuracy with climatological nutrient inputs, freshness limits shorter than 30 days, and the December 2025 severe grades against damage records.

## 11 Conclusion and future work

We described ST-MMT, a residual spatiotemporal transformer with hub-style multi-head cross-attention that forecasts laver bleaching seven days ahead from 32 aligned channels, together with the data pipeline (OceanTensorCube), the labeling procedure (BLC) and the evaluation protocol around it. On the validation set the six-seed ensemble detects every reported positive at a false-alarm rate of at most 3.5%, and a same-date control on a separately trained set of models shows that the architecture can discriminate between clusters when the calendar is held fixed. The same evaluation shows where the system is fragile: performance collapses when the quarterly nutrient survey is a few months old, and zeros in place of missing channels cost 20 of the 176 detections.

Three lines of work follow directly. First, measure what is unmeasured: an ablation of the cross-attention path, operating-point comparisons with the baselines, and freshness limits at 7–21 days. Second, close the nutrient gap at the source: the planned Goheung buoy is to provide nitrate and ammonium continuously, corrected for nitrite as in Section 8, and the climatology result suggests a fallback whose effect on detection must be tested before use. Third, edge deployment: the platform plan uses the 9.2-million-parameter model as the teacher for a 3.8-million-parameter student [21, 24], followed by reduced numerical precision for on-site hardware; this is planned and none of it has been measured yet.

## Data and service availability

The forecasting service is operated at https://oceantensor.ai.kr. Input data come from public Korean and international providers (KOEM, KMA, KHOA, NIFS, HYCOM, Copernicus Sentinel-2 and Copernicus Marine) under their respective terms.

## Author contributions

G.K. developed the backbone model and the evaluation, and leads the model-compression work. C.J. developed the forecasting service and dashboard. Y.J. carried out exploratory data analysis and the IoT sensor work.

## Acknowledgments

We thank the seaweed research division of the National Institute of Fisheries Science for cooperation on bleaching case records and for advice on how recovered cases are recorded. This work was carried out within the 2026 AI Champion competition program in Korea.

## References

[1] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Luciˇ c, and Cordelia Schmid. ViViT: A video´ vision transformer. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 6816–6826, 2021. doi: 10.1109/ICCV48922.2021.00676.

[2] Hyeon-Jeong Bae and Tae-Yong Jeong. Major diseases in edible red algae Pyropia under aquaculture: Infectious agents and procedure, detection, influencing factors, prevention, and treatment. Ocean and Coastal Research, 73: e25022, 2025. doi: 10.1590/2675-2824073.24044.

[3] S. H. Bak, D. H. Hwang, U. Enkhjargal, and H. J. Yoon. Prediction and mapping of Cochlodinium polykrikoides red tide using machine learning under imbalanced data. In The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, volume XLII-4/W16, pages 143–145, 2019. doi: 10.5194/ isprs-archives-XLII-4-W16-143-2019.

[4] Jessa Bekker and Jesse Davis. Learning from positive and unlabeled data: a survey. Machine Learning, 109: 719–760, 2020. doi: 10.1007/s10994-020-05877-5.

[5] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 813–824, 2021. URL https://proceedings.mlr.press/v139/ bertasius21a.html.

[6] Nicolas A. Blouin, Juliet A. Brodie, Arthur C. Grossman, Pu Xu, and Susan H. Brawley. Porphyra: a marine crop shaped by stress. Trends in Plant Science, 16(1):29–37, 2011. doi: 10.1016/j.tplants.2010.10.004.

[7] Eric P. Chassignet, Harley E. Hurlburt, Ole Martin Smedstad, George R. Halliwell, Patrick J. Hogan, Alan J. Wallcraft, Remy Baraille, and Rainer Bleck. The HYCOM (HYbrid Coordinate Ocean Model) data assimilative system. Journal ofMarine Systems, 65(1–4):60–83, 2007. doi: 10.1016/j.jmarsys.2005.09.016.

[8] Chun-Fu Richard Chen, Quanfu Fan, and Rameswar Panda. CrossViT: Cross-attention multi-scale vision transformer for image classification. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 347–356, 2021. doi: 10.1109/ICCV48922.2021.00041.

[9] Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 785–794, 2016. doi: 10.1145/2939672.2939785.

[10] Davide Chicco and Giuseppe Jurman. The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. BMC Genomics, 21:6, 2020. doi: 10.1186/s12864-019-6413-7.

[11] Jong-kuk Choi, Myung-sook Park, Kyung-soo Han, Hyun-cheol Kim, and Jungho Im. One year of GOCI-II launch present and future. Korean Journal ofRemote Sensing, 37(5-2):1229–1234, 2021. doi: 10.7780/kjrs.2021.37.5.2.1.

[12] Youngjin Choi, Youngmin Park, Weol-Ae Lim, Seung-Hwan Min, and Joon-Soo Lee. Convolution neural network for the prediction of Cochlodinium polykrikoides bloom in the South Sea of Korea. Journal ofMarine Science and Engineering, 10(1):31, 2022. doi: 10.3390/jmse10010031.

[13] Rafaela C. Cruz, Pedro Reis Costa, Susana Vinga, Ludwig Krippahl, and Marta B. Lopes. A review of recent machine learning advances for forecasting harmful algal blooms and shellfish contamination. Journal ofMarine Science and Engineering, 9(3):283, 2021. doi: 10.3390/jmse9030283.

[14] Jesse Davis and Mark Goadrich. The relationship between precision-recall and ROC curves. In Proceedings of the 23rd International Conference on Machine Learning (ICML), pages 233–240, 2006. doi: 10.1145/1143844. 1143874.

[15] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021.

[16] Charles Elkan and Keith Noto. Learning classifiers from only positive and unlabeled data. In Proceedings ofthe 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 213–220, 2008. doi: 10.1145/1401890.1401920.

[17] E.U. Copernicus Marine Service Information (CMEMS). Global ocean physics analysis and forecast (GLOBAL\_ANALYSISFORECAST\_PHY\_001\_024). Marine Data Store (MDS); Product User Manual CMEMS-GLO-PUM-001-024. URL https://documentation.marine.copernicus.eu/PUM CMEMS-GLO-PUM-001-024.pdf.

[18] E.U. Copernicus Marine Service Information (CMEMS). Global ocean biogeochemistry analysis and forecast (GLOBAL\_ANALYSISFORECAST\_BGC\_001\_028). Marine Data Store (MDS); Product User Manual CMEMS-GLO-PUM-001-028, Issue 1.2, November 2023, 2023. URL https://data.marine.copernicus.eu/ product/GLOBAL\_ANALYSISFORECAST\_BGC\_001\_028/description.

[19] FAO. The State of World Fisheries and Aquaculture 2024: Blue Transformation in Action. Food and Agriculture Organization of the United Nations, Rome, 2024. ISBN 978-92-5-138763-4. doi: 10.4060/cd0683en.

[20] Zhihan Gao, Xingjian Shi, Hao Wang, Yi Zhu, Yuyang Wang, Mu Li, and Dit-Yan Yeung. Earthformer: Exploring space-time transformers for earth system forecasting. In Advances in Neural Information Processing Systems, volume 35, pages 25390–25403, 2022.

[21] Jianping Gou, Baosheng Yu, Stephen J. Maybank, and Dacheng Tao. Knowledge distillation: A survey. International Journal ofComputer Vision, 129(6):1789–1819, 2021. doi: 10.1007/s11263-021-01453-z.

[22] Seung Heo, Jung-Ro Kweon, and Jong-Soo Park. The variation of the dissolved inorganic nutrients in the costal area of Gunsan, Yellow Sea from 2001 to 2010. Journal ofthe Korean Society ofMarine Environment and Safety, 17(4):357–365, 2011. URL https://db.koreascholar.com/Article/Detail/263400. In Korean with English abstract.

[23] Paul R. Hill, Anurag Kumar, Marouane Temimi, and David R. Bull. HABNet: Machine learning, remote sensingbased detection of harmful algal blooms. IEEE Journal ofSelected Topics in Applied Earth Observations and Remote Sensing, 13:3229–3239, 2020. doi: 10.1109/JSTARS.2020.3001445.

[24] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network, 2015.

[25] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural Computation, 9:1735–1780, 1997. doi: 10.1162/neco.1997.9.8.1735.

[26] Yutaka Hori, Shouju Mochizuki, and Nobuo Shimamoto. Relationship between the discoloration of cultivated Porphyra thalii and long-term changes of the environmental factors in the northern part of Harima-Nada, eastern Seto Inland Sea, Japan. Bulletin ofthe Japanese Society ofFisheries Oceanography, 72(2):107–112, 2008. doi: 10.34423/jsfo.72.2\_107.

[27] Eun Kyoung Hwang and Chan Sun Park. Seaweed cultivation and utilization of Korea. Algae, 35(2):107–121, 2020. doi: 10.4490/algae.2020.35.5.15.

[28] Ichiro Imai, Mineo Yamaguchi, and Yutaka Hori. Eutrophication and occurrences of harmful algal blooms in the Seto Inland Sea, Japan. Plankton and Benthos Research, 1(2):71–84, 2006. doi: 10.3800/pbr.1.71.

[29] Yuji Ito, Toshiya Katano, Naoki Fujii, Masumi Koriyama, Kenji Yoshino, and Yuichi Hayami. Decreases in turbidity during neap tides initiate late winter blooms of Eucampia zodiacus in a macrotidal embayment. Journal ofOceanography, 69(4):467–479, 2013. doi: 10.1007/s10872-013-0187-3.

[30] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 4651–4664, 2021. URL https: //proceedings.mlr.press/v139/jaegle21a.html.

[31] Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier Hénaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, and João Carreira. Perceiver IO: A general architecture for structured inputs & outputs. In International Conference on Learning Representations (ICLR), 2022.

[32] Sayash Kapoor and Arvind Narayanan. Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 4(9):100804, 2023. doi: 10.1016/j.patter.2023.100804.

[33] Shachar Kaufman, Saharon Rosset, Claudia Perlich, and Ori Stitelman. Leakage in data mining: Formulation, detection, and avoidance. ACM Transactions on Knowledge Discoveryfrom Data, 6(4):15, 2012. doi: 10.1145/ 2382577.2382579.

[34] Jang K. Kim, Charles Yarish, Eun Kyoung Hwang, Miseon Park, and Youngdae Kim. Seaweed aquaculture: cultivation technologies, challenges and its ecosystem services. Algae, 32(1):1–13, 2017. doi: 10.4490/algae. 2017.32.3.3.

[35] Minkyu Kim, Hyun Yang, and Jonghwa Kim. Sea surface temperature and high water temperature occurrence prediction using a long short-term memory model. Remote Sensing, 12(21):3654, 2020. doi: 10.3390/rs12213654.

[36] Sangil Kim, Sang Chol Yoon, Man Ho Yoo, Kyung Woo Park, Sang Rul Park, and Seok-Hyun Youn. Physiological responses of cultured seaweed Pyropia yezoensis to phosphorous limitation in the Nakdong river estuary, Korea. Ocean Science Journal, 54(1):129–139, 2019. doi: 10.1007/s12601-018-0065-4.

[37] Soo Mee Kim, Jisun Shin, Seungjae Baek, and Joo-Hyung Ryu. U-Net convolutional neural network model for deep red tide learning using GOCI. Journal ofCoastal Research, 90(sp1):302–309, 2019. doi: 10.2112/SI90-038.1.

[38] Korea Meteorological Administration. Open MET data portal: Marine meteorological buoy and surface observation data. https://data.kma.go.kr. Accessed 2026-10-11.

[39] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems, volume 30, pages 6402–6413, 2017.

[40] Sangmok Lee and Donghyun Lee. Improved prediction of harmful algal blooms in four major South Korea’s rivers using deep learning models. International Journal of Environmental Research and Public Health, 15(7): 1322, 2018. doi: 10.3390/ijerph15071322.

[41] Bryan Lim, Sercan Ö. Arık, Nicolas Loeff, and Tomas Pfister. Temporal fusion transformers for interpretable multi-horizon time series forecasting. International Journal of Forecasting, 37(4):1748–1764, 2021. doi: 10.1016/j.ijforecast.2021.03.012.

[42] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 2999–3007, 2017. doi: 10.1109/ICCV. 2017.324.

[43] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. ViLBERT: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In Advances in Neural Information Processing Systems, volume 32, pages 13–23, 2019.

[44] National Institute of Fisheries Science. [NIFS director visits laver farms damaged by bleaching in Haenam] (in Korean). Press release, National Institute of Fisheries Science, Busan, Korea, February 2022. URL https:// www.nifs.go.kr/news/actionNewsView.do?MENU\_ID=M0000307&NEWS\_SEQ=3991. Released 4 February 2022.

[45] National Institute of Fisheries Science. [NIFS president holds on-site meeting on current issues in laver farming] (in Korean). Press release, National Institute of Fisheries Science, Busan, Korea, September 2022. URL https: //nifs.go.kr/news/actionNewsView.do?MENU\_ID=M0000307&NEWS\_SEQ=4101. Released 1 September 2022.

[46] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations (ICLR), 2023.

[47] Tetsuya Nishikawa, Yutaka Hori, Satoshi Nagai, Kazutaka Miyahara, Yukinobu Nakamura, Kazuhiro Harada, Kuninao Tada, and Ichiro Imai. Long time-series observations in population dynamics of the harmful diatom Eucampia zodiacus and environmental factors in Harima-Nada, eastern Seto Inland Sea, Japan during 1974–2008. Plankton and Benthos Research, 6(1):26–34, 2011. doi: 10.3800/pbr.6.26.

[48] Alfred C. Redfield. The biological control of chemical factors in the environment. American Scientist, 46(3): 205–221, 1958.

[49] Joo-Hyung Ryu, Hee-Jeong Han, Seongick Cho, Young-Je Park, and Yu-Hwan Ahn. Overview of geostationary ocean color imager (GOCI) and GOCI data processing system (GDPS). Ocean Science Journal, 47(3):223–233, 2012. doi: 10.1007/s12601-012-0024-4.

[50] Takaya Saito and Marc Rehmsmeier. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3):e0118432, 2015. doi: 10.1371/journal. pone.0118432.

[51] Seoul Shinmun. [bleaching accelerates the devastation of laver farms on the south coast] (in Korean). Seoul Shinmun, news article, February 2022. URL https://go.seoul.co.kr/news/newsView.php?id= 20220204500113. Published 4 February 2022.

[52] Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong, and Wang-chun Woo. Convolutional LSTM network: A machine learning approach for precipitation nowcasting. In Advances in Neural Information Processing Systems, volume 28, 2015.

[53] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In International Conference on Learning Representations (ICLR), 2015.

[54] Akio Sohma, Riku Imada, Tetsuya Nishikawa, and Hisashi Shibuki. Modeling the life cycle of four types of phytoplankton and their bloom mechanisms in a benthic-pelagic coupled ecosystem. Ecological Modelling, 467, 2022. doi: 10.1016/j.ecolmodel.2022.109882.

[55] Judith E. Sutherland et al. A new look at an ancient order: generic revision of the Bangiales. Journal of Phycology, 47(5):1131–1151, 2011. doi: 10.1111/j.1529-8817.2011.01052.x.

[56] Megumu Takahashi, Puja Kumari, Chengze Li, and Koji Mikami. Low temperature causes discoloration by repressing growth and nitrogen transporter gene expression in the edible red alga Pyropia yezoensis. Marine Environmental Research, 159:105004, 2020. doi: 10.1016/j.marenvres.2020.105004.

[57] Hao Tan and Mohit Bansal. LXMERT: Learning cross-modality encoder representations from transformers. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5100–5111, 2019. doi: 10.18653/v1/D19-1514.

[58] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017. URL https://papers.nips.cc/paper/2017/hash/ 3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.

[59] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. Proceedings ofthe AAAI Conference on Artificial Intelligence, 35(12):11106–11115, 2021. doi: 10.1609/aaai.v35i12.17325.

## A Development configurations

Table 17 lists the configurations discussed in Section 9. These runs used 2021 data from one or two tiles, small validation windows around a single event, and pixel-level confusion matrices; all but the last row predicted the day after the input window. They document the design path and are not comparable with Section 7.

Table 17: Key development configurations (two-tile Wando+Mokpo task unless noted; F1 and precision at the recall-100% threshold, where some source tables label the F1 as macro-F1; pixel-level counting).
<table><tr><td>Configuration</td><td>Change</td><td>Params</td><td>ROC-AUC</td><td>PR-AUC</td><td>F1</td><td>Precision</td></tr><tr><td>v7 (Wando only, 30 ch)</td><td>recall-100% threshold introduced</td><td></td><td>0.903-0.927</td><td></td><td>0.907–0.926</td><td>83.3-86.2%</td></tr><tr><td>v13</td><td>two tiles</td><td></td><td>0.9071</td><td>0.8522</td><td>0.9193</td><td>87.72%</td></tr><tr><td>v14</td><td>+ Jindo (three tiles)</td><td></td><td>0.8378</td><td>0.8091</td><td>0.7408</td><td>70.80%</td></tr><tr><td>v16</td><td>+ Goheung, 2016 event (four tiles)</td><td></td><td>0.7790</td><td>0.7212</td><td>0.7580†</td><td></td></tr><tr><td>v19</td><td>N:P ratio bug fixed; self-attention</td><td>3,819,650</td><td>0.8716</td><td>0.8516</td><td>0.8953</td><td>84.75%</td></tr><tr><td>xattn_v1</td><td>3-group hub cross-attention</td><td>5,922,690</td><td>0.9191</td><td>0.9110</td><td>0.8013</td><td>75.36%</td></tr><tr><td>xattn_v2</td><td>4 groups (nutrients separate)</td><td></td><td>0.9414</td><td></td><td>0.7786</td><td></td></tr><tr><td>xattn_v3</td><td>residual joint + cross-attention</td><td></td><td>0.8917</td><td></td><td>0.8953</td><td>84.75%</td></tr><tr><td>xattn_v3_focal</td><td>+ focal loss (γ = 2)</td><td></td><td>0.9034</td><td></td><td>0.9271</td><td>88.74%</td></tr><tr><td>xattn_v3_focal_7day</td><td>target moved to +7 days</td><td></td><td>0.9627</td><td>0.9818</td><td>0.8370</td><td>84.93%</td></tr></table>

<sup>†</sup> Macro-F1 at the default threshold; the recall-100% value was not reported for this configuration.

The v13 result was later judged unreliable because it was trained before the N:P ratio bug was fixed. The first seven-day result included an external Seocheon check that was explicitly withdrawn; the next-day Seocheon check of the focal variant rests on the same seasonal pattern and should not be cited either (Section 9).

## B Measurement provenance

Table 18: Source and date of each reported measurement.
<table><tr><td>Measurement</td><td>Procedure</td><td>Date</td></tr><tr><td>Overall metrics and confusion matrix (Table 6)</td><td>validation set, 6-seed ensemble, θ* = 0.0940</td><td>2026-10-04/05</td></tr><tr><td>Same-date control (Table 7)</td><td>separate 5-fold CV training, within-date AUC, cluster bootstrap</td><td>2026-10-04/05</td></tr><tr><td>Inference port check</td><td>element-wise comparison with training path</td><td>2026-10-04</td></tr><tr><td>Experiments A and B (Tables 9, 12)</td><td>ablation script with pre-registered criteria</td><td>2026-10-06</td></tr><tr><td>2026 patch channel audit</td><td>zero-fraction check across machines</td><td>2026-10-07</td></tr><tr><td>Nutrient freshness (Table 11)</td><td>per-source latest record</td><td>2026-10-07</td></tr><tr><td>Nitrite share (Table 13)</td><td>NIFS fishing-ground samples, n = 24,513</td><td>2026-10-07</td></tr><tr><td>Proxy search (Table 14)</td><td>change correlations, per-tile standardized</td><td>2026-10-07</td></tr><tr><td>Persistence vs. climatology (Figure 9)</td><td>leave-one-year-out, n = 22,736 pairs</td><td>2026-10-07</td></tr></table>

## C Reproduction steps

1. Build patches: tile cube → central 64 × 64 patch per cluster and year; check the zero fraction of all 32 channels.

2. Build labels and the split; verify the window arithmetic (τ − 20, length 14, horizon 7) on every sample.

3. Train six seeds (42, 1, 2, 3, 5, 7); store normalization statistics with each checkpoint.

4. Verify the stand-alone inference module element-wise against the training path (tolerance $1 0 ^ { - 4 } )$

5. Compute the overall metrics, the same-date control and the robustness experiments; compare with Tables 6–9 and 12.

6. Run ensemble inference over all clusters and dates, build the forecast pack, and check its grade distribution against Table 10.

7. Deploy; the service refuses a pack whose model name does not match.