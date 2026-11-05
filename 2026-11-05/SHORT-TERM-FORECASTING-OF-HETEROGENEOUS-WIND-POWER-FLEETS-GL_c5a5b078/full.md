# SHORT-TERM FORECASTING OF HETEROGENEOUS WIND POWER FLEETS: GLOBAL TEMPORAL FUSION TRANSFORMER LEVERAGING STATIC COVARIATES

Viktor Walter<sup>1</sup>, Andreas Wagner<sup>1,2</sup>

<sup>1</sup> Faculty of Management Science and Engineering, Karlsruhe University of Applied Sciences, Karlsruhe, Germany

<sup>2</sup> Fraunhofer Institute for Industrial Mathematics, Kaiserslautern, Germany

E-mail: viktor.walter@h-ka.de

## ABSTRACT

With the increasing integration of renewable energy sources, accurate short-term wind power forecasting has be come critical for grid stability and efficient electricity trading. While deep learning, particularly the transformer architecture, has emerged as the state-of-the-art approach, current industry practices predominantly rely on specialized local models trained for individual assets. This study investigates the efficacy of a global temporal fusion transformer (TFT) trained on a heterogeneous portfolio of 100 wind farms across Germany. The primary objective is to determine whether a global model trained jointly on multiple wind farms can outperform local models trained exclusively on single sites in 48-hour-ahead forecasting. Combining numerical weather prediction (NWP) data from the regional ICON-D2 and the global ECMWF HRES models, we compare the performance of a global TFT against distinct local models as well as a linear-regression baseline. The global model leverages static context variables, such as farm age, altitude above sea level, turbine hub height, rotor diameter and power curve parameters to learn shared representations across diverse sites. We evaluate on 100 synthetic wind farm profiles and, additionally, on a real operational portfolio of 11 German wind farms, across two seasonal test periods with intermediate retraining. The empirical results demonstrate that the global TFT outperforms the local baselines for the majority of sites (87% of the synthetic and 77% of the real wind farms) while clearly surpassing the linear-regression baseline. Statistical analysis using the Wilcoxon signed-rank test confirms the significance of these performance gains, which hold across both seasons and on the real portfolio. Crucially, we show that this superiority is strictly conditional on the inclusion of static covariates. A global model lacking asset-specific context fails to generalize effectively. Furthermore, the interpretability of the TFT’s variable selection network (VSN) yields interpretable insights into the learned feature selection. The analysis identifies altitude above sea level as a relevant static discriminator on both synthetic and real data, while hub height emerges as a decisive feature on the real portfolio. Furthermore, it reveals a dual modeling strategy for dynamic inputs: while the encoder chooses historical power as the most relevant observed feature, the decoder draws on a combination of upper-level wind speeds from both NWP models rather than a single atmospheric level or source, consistent with averaging out model-specific NWP errors. These findings support a shift from single-site to scalable multi-site forecasting architectures, offering a pathway to reduce operational complexity while enhancing predictive accuracy in large-scale renewable energy portfolios.

Keywords: Renewable Energy, Wind Power, Time Series Forecasting, Temporal Fusion Transformer

## INTRODUCTION

Renewable energy (RE) has established itself as a cornerstone of global energy infrastructure. In 2024, RE accounted for 17% of global electricity production, with wind energy contributing 25% of this share (Energy Institute, 2024). Wind power contributed 33% to Germany’s net electricity production (Fraunhofer Institute for Solar Energy Systems ISE, 2025). However, the inherent stochasticity of wind introduces significant fluctuations into the grid. Consequently, market participants in short-term electricity trading, particularly on short-term markets such as day-ahead and intraday markets, rely heavily on accurate forecasts to mitigate volume risk.

Historically, forecasting methods have shifted from statistical approaches like ARIMA (Kavasseri and Seetharaman, 2009) to data-driven machine learning. Hybrid neural networks have become the standard, effectively learning non-linear relationships between weather conditions and power generation (Guo et al., 2021). Yet, current industry practice typically relies on site-specific models calibrated for individual wind farms. These models often generalize poorly when data is scarce, a limitation stemming from the heterogeneous nature of the domain: wind farms differ significantly in both local meteorology and technical turbine parameters. A global model can only generalize effectively if it learns to recognize these underlying distinctions.

A seminal architecture for such multi-horizon problems is the temporal fusion transformer (TFT) (Lim et al., 2021). The TFT is uniquely advantageous as it incorporates static context features, allowing the network to differentiate be tween heterogeneous time series. While TFTs represent the state of the art, existing literature on their application to wind power remains limited. This study addresses open questions regarding how a global model must be configured to outperform specialized local models and clarifies which features are prioritized in this process. The main contributions of this paper are summarized as follows:

• We design a wind-speed–focused feature set that fuses two complementary NWP models—the high-resolution regional ICON-D2 by the German weather service (DWD) and the global European Centre for Medium-Range Weather Forecasts (ECMWF) highest resolution model (HRES)—across several atmospheric levels, supplying the model with complementary, height-resolved wind information.

• We show that the advantage of a global TFT over site-specific models is conditional on encoding physical static covariates: through a no-static ablation, a global model without asset-specific context fails to beat local baselines, whereas one with it does. We validate this finding on both synthetic profiles and a real operational portfolio of German wind farms, benchmarked against a linear-regression baseline.

• We leverage the explainable AI capabilities of the TFT’s variable selection network (VSN) to elucidate the importance of various features that drive the heterogeneity between wind turbines, contrasting the feature attributions obtained on synthetic and real data.

The remainder of this paper is structured as follows: The Related Work section reviews relevant literature on wind power forecasting and TFT. The Methodology section details the feature selection and the model architecture. The data, the hyperparameter optimization (HPO) and the evaluation is described in the Experimental Setup section. The Results section presents and discusses the results of the experiments. The last section concludes the paper and outlines directions for future work.

## RELATED WORK

This section reviews existing literature pertinent to the present study. Table 1 categorizes recent developments and positions the present study within this landscape.

Operational day-ahead forecasting fundamentally relies on NWP integration. While approaches based solely on historical measurements (Kisvari et al., 2021; Liu et al., 2024; Mi et al., 2025) show promise for very short-term horizons (<6 hours), they are unsuitable for day-ahead scenarios where actual wind measurements are unavailable at the time of inference (Wang et al., 2017; Tsai et al., 2023). Consequently, recent advances predominantly focus on hybrid architectures fusing NWP data with deep learning (Xu et al., 2024; Jachuła and Wydra, 2025).

A critical distinction in current literature lies in the modeling scope. Despite the shift toward transformers, the vast majority of research remains tethered to specialized, site-specific models (Verdone et al., 2025). Only recently have global models emerged: Cohen et al. (2025) demonstrated that a global transformer trained on 74 sites could reduce errors by 33% compared to local baselines, though this was limited to a 12-hour horizon and framed as post-calibration of NWP forecast errors on a power-capacity metric rather than direct power forecasting.

The TFT (Lim et al., 2021) has become the architecture of choice for addressing heterogeneity in time series. Niu et al. (2024) and Michalakopoulos et al. (2025) validated the TFT’s ability to utilize static features to differentiate between sites. However, a critical limitation in these studies is the scarcity of spatial diversity and the nature of the static context used. While Niu et al. (2024) incorporate static variables, their approach relies on categorical site identifiers (e.g., wind farm ID or season) rather than physical asset parameters. Consequently, their model is conditioned to memorize specific site behaviors rather than learning transferable physical laws governed by turbine attributes. Furthermore, their evaluation is restricted to only three wind farms. Such a small sample size is likely insufficient for the model to learn a generalized mapping, as the variance across three samples prevents the disentanglement of site-specific noise from genuine physical characteristics. Conversely, studies that achieve a higher site count, such as Zhang et al. (2025) with 115 sites, typically ignore static covariates entirely (Table 1). Consequently, existing research either lacks the necessary scale to exploit static features effectively or neglects them altogether in favor of purely autoregressive or NWP-based patterns.

As summarized in Table 1, a distinct research gap remains. While components such as global training (Cohen et al., 2025), static feature integration (Wang et al., 2024), and day-ahead horizons (Cheng et al., 2024) exist in isolation, no study has yet systematically combined a large-scale global TFT approach with NWP and static context for 48-hour power forecasting. This work bridges this gap by experimentally comparing global versus local models on a dataset of 100 wind farms, ensuring sufficient heterogeneity to robustly quantify the impact of static features on forecast accuracy.

## METHODOLOGY

This section details the methodology, starting with the model architecture, followed by the feature selection.

## Temporal Fusion Transformer

The temporal fusion transformer (TFT) is an attention-based architecture for multi-horizon time series forecasting that captures multi-scale temporal dynamics while preserving interpretability through its variable selection network (VSN) (Lim et al., 2021). The following description highlights the components most relevant to the wind energy application; we refer to the original publication for the complete framework.

Table 1. Literature Overview of Transformer-based Wind Power Forecasting
<table><tr><td>Author</td><td>Model</td><td>NWP</td><td>Horizon</td><td>Scope</td><td>Sites</td><td>Static Cov.</td></tr><tr><td>Mi et al. (2025)</td><td>TCN-TFT</td><td></td><td>1h</td><td>Local</td><td>1</td><td></td></tr><tr><td>Liu et al. (2024)</td><td>TFT</td><td></td><td>1h</td><td>Local</td><td>1</td><td></td></tr><tr><td>Hou and Liu (2026)</td><td>TFT</td><td></td><td>1h</td><td>Local</td><td>1</td><td></td></tr><tr><td>Mo et al. (2024)</td><td>Transformer</td><td></td><td>4 h</td><td>Local</td><td>1</td><td></td></tr><tr><td>van Heerden et al. (2023)</td><td>TFT</td><td></td><td>24 h</td><td>Local</td><td>1</td><td></td></tr><tr><td>Michalakopoulos et al. (2025)</td><td>TFT</td><td></td><td>24 h</td><td>Local</td><td>2</td><td>√</td></tr><tr><td>Wang et ai. (2024)</td><td>TFT</td><td>√</td><td>4h</td><td>Local</td><td>1</td><td>√</td></tr><tr><td>Jachuła and Wydra (2025)</td><td>TFT</td><td>√</td><td>60 h</td><td>Local</td><td>1</td><td></td></tr><tr><td>Xu et al. (2024)</td><td>Transformer</td><td>√</td><td>72 h</td><td>Local</td><td>1</td><td></td></tr><tr><td>Zhang et al. (2025)</td><td>TFT</td><td>√</td><td>4 h</td><td>Global</td><td>115</td><td></td></tr><tr><td>Cohen et al. (2025)</td><td>Transformer</td><td>√</td><td>12 h</td><td>Global</td><td>74</td><td></td></tr><tr><td>Niu et al. (2024)</td><td>TFT</td><td>√</td><td>24 h</td><td>Global</td><td>3</td><td>√</td></tr><tr><td>Cheng et àl. (2024)</td><td>Transformer</td><td>√</td><td>48 h</td><td>Global</td><td>7</td><td>-</td></tr><tr><td>This Work</td><td>TFT</td><td>√</td><td>48 h</td><td>Global</td><td>100</td><td>√</td></tr></table>

## Input Processing and Feature Differentiation

A distinct advantage of the TFT is the separate handling of diverse input types relevant to wind power generation. The model partitions inputs into static covariates, past-observed inputs, and known future inputs. Static covariates encompass time-invariant attributes to learn from context. These parameters condition the model to specific site characteristics. Past-observed inputs utilize historical time series including active power generation. Known future inputs integrate external information available at the time of prediction. This specifically includes NWP data. Known inputs span the entire sequence, covering both the observation window and the forecast horizon.

The experimental setup utilizes a 48-hour forecast horizon and a corresponding 48-hour lookback window, the horizon being limited to 48 hours by the NWP model. Let t denote the relative time index, with $t = 0$ the moment of prediction. The past-observed inputs $\mathbf { x } ^ { ( o b s ) }$ are defined over the lookback window $( - 4 7 \ \leq \ t \ \leq \ 0 )$ , yielding $| \mathbf { x } ^ { ( o b s ) } | = 4 8$ , whereas the known future inputs $\mathbf { x } ^ { ( k n o w n ) }$ span both the lookback window and the forecast horizon $( - 4 7 \leq t \leq 4 8 )$ , yielding $| \mathbf { x } ^ { ( k n o w n ) } | = 9 6$ . Finally, static covariates $\mathbf { x } ^ { ( s t a t i c ) }$ possess no temporal dimension and remain constant across all time steps.

## Variable Selection Network

Wind power forecasting involves high-dimensional datasets due to characteristics of NWP models. NWP models provide meteorological variables at multiple height levels and for many gridpoints close to the wind farm. Turbine hub heights also vary significantly across different installation sites. The TFT handles this complexity through its variable selection network. The VSN assigns specific learnable weights to each input feature independently, which creates interpretability of the feature importances. This weighting process occurs strictly before the data enters the main recurrent network components. The mechanism effectively filters out irrelevant height levels or redundant sensor data, which can be highly site-specific. This suppression drastically reduces the effective feature space.

Mathematically, this suppression represents a projection from a multi-feature space into a single latent representation via a convex combination. Initially, the m input features are transformed into distinct hidden state vectors of size $d _ { m o d e l } ,$ creating an intermediate feature space in $\mathbb { R } ^ { m \times d _ { m o d e l } }$ . The VSN aggregates these embedding vectors $\tilde { \xi } _ { t } ^ { ( j ) }$ representing the j-th feature at time step t, using learnable selection weights $\nu _ { t } ^ { ( j ) }$ to filter out irrelevant information through a weighted sum:

$$
\tilde { \xi } _ { t } = \sum _ { j = 1 } ^ { m } \nu _ { t } ^ { ( j ) } \tilde { \xi } _ { t } ^ { ( j ) }\tag{1}
$$

Consequently, the effective feature dimension is drastically reduced, resulting in a single compact representation vector for the time step t in $\mathbb { R } ^ { d _ { m o d e l } }$

## Static Covariate Encoders

Wind farms comprise turbines with varying physical properties. The static covariate encoders mechanism integrates these time-invariant attributes into the forecasting logic. Primary static inputs include hub height, altitude above sea level, rotor diameter, farm age and power curve parameters (cut-in, cut-out, rated wind speed), as these factors define the physical identity of a site. The TFT processes these features using dedicated encoders to produce abstract context vectors in the hidden space. These vectors condition the entire network by initializing the recurrent layers and influencing variable selection. This architecture allows a single global model to distinguish between different turbine types and site characteristics. It effectively adapts predictions to the specific physical constraints of each asset without retraining.

## Implementation and Loss Function

The implementation adheres to the original TFT framework. The model is trained with the quantile (pinball) loss evaluated at the 0.5 quantile, which yields a single median point forecast; full probabilistic multi-quantile forecasting is left for future work.

The source code is publicly available<sup>1</sup>.

## Feature Selection

This section details the feature set. A central design choice is to restrict the inputs to wind speed—the physically dominant driver of power (Eq. 2)—rather than the full meteorological state, and to draw it from two complementary NWP models at several atmospheric levels. As outlined in the methodology, input variables are categorized into past-observed, known future, and static features. Table 2 provides a comprehensive list of all utilized variables.

Table 2. Features used for model training
<table><tr><td>Past Observed</td><td>Known Future (NWP)</td><td>Static</td></tr><tr><td rowspan="5">Power</td><td>ICON-D2 wind speed at 78m</td><td>Power curve (cut-in, cut-out, rated wind speed)</td></tr><tr><td>ICON-D2 wind speed at 127m</td><td>Hub height</td></tr><tr><td>ICON-D2 wind speed at 184m</td><td>Rotor diameter</td></tr><tr><td>HRES wind speed at 100m</td><td>Farm age at inference time</td></tr><tr><td>HRES wind speed at 200m</td><td>Surface altitude above sea level</td></tr></table>

The dominant role of wind speed follows directly from the physical definition of wind power:

$$
P = { \frac { 1 } { 2 } } \cdot \rho \cdot A \cdot C _ { p } \cdot v ^ { 3 }\tag{2}
$$

where $\rho$ denotes the air density, $A$ the rotor swept area, $C _ { p }$ the power coefficient, and v the wind speed. Because generation scales with the cube of wind speed, accurate wind-speed information at the rotor-relevant heights is by far the primary driver of forecast skill.

To this end, we fuse wind-speed forecasts from two complementary NWP models: the high-resolution, convectionpermitting DWD’s regional ICON-D2 model (2 km) and the global ECMWF HRES model (9 km). Combining a regional and a global model exposes the network to complementary forecast-error structures, which the VSN can exploit when weighting the individual inputs. Wind speed is provided at several atmospheric levels spanning the rotor-relevant height range (78–184 m for ICON-D2 and 100–200 m for HRES), since hub heights differ markedly across the heterogeneous portfolio. Rather than aggregating these layers a priori, we expose the individual levels to the model and leave the selection of the relevant ones to the variable selection network (Section ), preserving the full vertical information of the wind profile.

## EXPERIMENTAL SETUP

The first subsection introduces the utilized dataset. Subsequently, the hyperparameter optimization (HPO) is detailed. Finally, the performance metrics and statistical tests used to evaluate the results are defined.

## The Data

This study utilizes a public dataset (Walter and Wagner) combining synthetic German wind farm profiles with real NWP data. Wind-speed forecasts are drawn from DWD’s ICON-D2 and ECMWF HRES. The dataset comprises hourly power generation profiles for 160 distinct locations across Germany. Each site includes time series for six different turbine types and meteorological features consisting of both station measurements and NWP data. The forecast data incorporates the 06:00, 09:00, 12:00, and 15:00 UTC ICON-D2 runs, each providing a 48-hour prediction horizon. The data spans the period from July 2023 to Jan 2026 and is publicly available on Mendeley Data (Walter, 2026).

For this study, 100 randomly distributed sites comprising wind turbines across Germany were selected. To quantify the effect of portfolio size, we additionally define a fixed 50-farm subset (a random half of the 100 sites): a global model trained only on these 50 farms (Global-50) is compared against the 100-farm model evaluated on the same subset, with both optimizing their hyperparameters. Six synthetic turbine models recur throughout the sample. Table 3 lists their technical parameters and the share of wind farms featuring these turbines. Figure 1 illustrates the distribution of farm age and altitude above sea level. The portfolio comprises both young and mature turbines. Most wind farms are situated at low altitudes, though the range extends to 900 meters. Here, altitude serves mainly as a proxy for more complex, elevated terrain, which affects forecasting difficulty.

Table 3. Synthetic wind power turbines occurring in this study
<table><tr><td>Turbine</td><td>Rated Power</td><td>Hub</td><td>Diameter</td><td>Cut-in</td><td>Cut-out</td><td>Rated Speed</td><td>Frequency</td></tr><tr><td>Vestas V112</td><td>3.45 MW</td><td>119 m</td><td>112 m</td><td>3 m/s</td><td>25 m/s</td><td>13 m/s</td><td>21</td></tr><tr><td>Enercon E-82 E2</td><td>2.0 MW</td><td>138 m</td><td>82 m</td><td>2 m/s</td><td>34 m/s</td><td>13 m/s</td><td>17</td></tr><tr><td>Enercon E-115</td><td>2.5 MW</td><td>149 m</td><td>115 m</td><td>3 m/s</td><td>34 m/s</td><td>12 m/s</td><td>16</td></tr><tr><td>Vestas V80</td><td>1.8 MW</td><td>78 m</td><td>80 m</td><td>4 m/s</td><td>25 m/s</td><td>15 m/s</td><td>16</td></tr><tr><td>Enercon E-70 E4</td><td>2.3 MW</td><td>57 m</td><td>71 m</td><td>3 m/s</td><td>34 m/s</td><td>15 m/s</td><td>15</td></tr><tr><td>Vestas V90</td><td>2.0 MW</td><td>95 m</td><td>90 m</td><td>4 m/s</td><td>25 m/s</td><td>13 m/s</td><td>15</td></tr></table>

![](images/9b11590a05b33da600a5db70919e399083f946beb8888d159d04c97923996f61.jpg)

![](images/fe41e1d354d7879d252b047dc9c0526f4329217a0a5b5f462656dedd79e2d016.jpg)  
Figure 1. Distribution of farm age and surface altitude above sea level

To validate the findings beyond synthetic profiles, we additionally evaluate the models on a real operational portfolio of 11 German wind farms distributed across the country. The portfolio comprises 53 turbines with a total installed capacity of 120.45 MW and covers approximately one year of measured power generation. In contrast to the synthetic profiles, this data is used without any prior cleaning and therefore reflects real operating conditions, including curtailment, maintenance, outages, and grid constraints. This deliberately conservative setting allows us to assess whether the conclusions drawn from the synthetic experiments transfer to real-world SCADA behavior. For the real portfolio, we reuse the hyperparameters obtained from the synthetic HPO without any re-tuning. This is a deliberate choice: it avoids information leakage from the real test periods (which precede the synthetic HPO validation window) and additionally tests the transferability of the configuration.

## Hyperparameter Optimization

Hyperparameter optimization (HPO) utilizes a 12-fold rolling time series cross-validation. The minimum training period (July 2023–July 2024) is followed by a validation period from August 2024 to July 2025.

In each iteration k, the model trains on all historical data up to the k-th validation month, progressively expanding the training set. Optuna’s tree-structured parzen estimator (TPE) is employed to minimize the average RMSE across all folds.

To accelerate the search, median pruning terminates unpromising trials early if intermediate performance (checked after the second fold) exceeds the median of prior trials.

## Evaluation

Following HPO, the models are retrained using the best-found hyperparameter set and evaluated on consecutive held-out test windows. To probe seasonal generalization, the synthetic portfolio is assessed on two test periods: a first window covering August to October 2025 (late summer and early autumn) and a second window covering November 2025 to January 2026 (winter), with the models retrained on all data preceding each window. The real world portfolio is evaluated analogously on two consecutive three-month windows, the first under winter and the second under spring conditions, again with intermediate retraining. This rolling protocol mirrors operational practice and tests whether the performance gains persist across seasons rather than being specific to a single period.

In addition to the local TFT baselines, we benchmark all configurations against a simple reference model: a linear regression on the NWP wind-speed inputs (LinReg). This contextualizes the marginal value of the TFT architecture itself relative to a classical statistical predictor.

The predictive performance is assessed using the coefficient of determination (R<sup>2</sup>), root mean squared error (RMSE), and mean absolute error (MAE). To ensure comparability across sites with varying capacities, the error metrics are normalized by the rated power and reported as percentages. Given the ubiquity of these metrics in regression analysis, their formal mathematical definitions are omitted for the sake of brevity.

To statistically evaluate the performance differences between models, we employ the two-sided Wilcoxon signedrank test on the paired per-site errors. As a non-parametric paired test, it makes no assumption of normality for the differences, which makes it robust for the limited and potentially skewed sample sizes encountered here. We treat a difference as significant at the 5% level (p < 0.05); to guard against false positives from multiple comparisons, all reported significant results additionally survive Holm–Bonferroni correction across the five tests.

## RESULTS

This section presents the empirical results of the study. First, Figure 2 compares the global and the locally trained models on both the synthetic and the real portfolio, with and without static covariates. The corresponding Wilcoxon signed-rank tests are summarized in Table 4. Subsequently, the feature importance for future, past, and static covariates is depicted in Figure 3. Tables 5 and 6 summarize the performance metrics for the synthetic and the real portfolio across all model configurations and test periods. Finally, Table 7 lists the optimal hyperparameters derived from the validation data.

Global vs. Local Models

![](images/952b94975f375272ea032881f667326e6681e038b46512689c1ddd4a6d0e06c1.jpg)

![](images/f741ef75bbad6b2d063ec89c8a73571e787df485176bb9265da2ae5282bef70b.jpg)

![](images/9928a8cad18c0100a5e7e108282c68566be293b5ca3bc433998867c4f2dcec98.jpg)

![](images/b5a9eda21d3eca03234190b3d6fd1fbeebe9dbcb9a4bfc0c75321ee0a86e0574.jpg)  
Figure 2. Per-farm $\mathrm { R } ^ { 2 }$ of the global model against the local baseline, for the synthetic (100 farms) and the real (11 farms) portfolios, each with and without static covariates. Points above the diagonal (green) favor the global model.

Figure 2 shows four scatterplots of the per-farm test-set $\mathrm { R ^ { 2 } }$ of the global model (y-axis) against the local models (x-axis); points above the diagonal (green) favor the global model, those below (red) the local model.

When static covariates are included, the global model outperforms the local baseline for the clear majority of sites: 87% of the synthetic farms and 77% of the real farms lie above the diagonal. Removing the static context reverses this picture entirely. Without it, only 30% of the synthetic and 18% of the real farms still benefit from global training, i.e. the local models prevail for the majority of sites. This pattern is remarkably consistent between the synthetic and the real portfolio.

Table 4 confirms these observations statistically. Using a two-sided Wilcoxon signed-rank test, the static global model significantly improves on the local baseline on the synthetic data (both the 100- and the 50-farm configuration) and on the real portfolio. Conversely, both no-static configurations are significantly worse than the local baseline, underscoring that without static covariates global training actively degrades accuracy. All five comparisons remain significant after Holm–Bonferroni correction. This corroborates the key finding: static covariates are a prerequisite for a global model to reliably outperform site-specific training, and this requirement holds on real-world data just as on the synthetic profiles.

Table 4. Two-sided Wilcoxon signed-rank test on RMSE against the local baseline. Green/red columns denote a statistically significant improvement/degradation of the global model relative to the local baseline $( p < 0 . 0 5 )$ ; all five comparisons remain significant after Holm–Bonferroni correction. $( ^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1 . )$
<table><tr><td rowspan="2"></td><td colspan="3">Synthetic</td><td colspan="2">Real</td></tr><tr><td>100 (static)</td><td>100 (no static)</td><td>50 (static)</td><td>Real (static)</td><td>Real (no static)</td></tr><tr><td>Median rel. ∆RMSE [%]</td><td>+2.42</td><td>-3.04</td><td>+2.01</td><td>+1.64</td><td>-6.19</td></tr><tr><td>p-value</td><td> $2 . 6 \times 1 0 ^ { - 1 0 }$ </td><td> $1 . 8 \times { 1 0 } ^ { - 7 }$ </td><td> $8 . 9 \times 1 0 ^ { - 6 }$ </td><td>0.039</td><td>0.0074</td></tr><tr><td>Significance</td><td>***</td><td></td><td>***</td><td>*</td><td>**</td></tr></table>

The 50-farm configuration additionally addresses the question of scale. Training the same 50 farms as part of the larger 100-farm portfolio yields comparable, but marginally higher accuracy (Table 5), showing that scaling the portfolio does not dilute per-site performance.

## Feature Importance

The VSN inherent to the TFT architecture enables the interpretability of feature importance. Figure 3 displays the average importance weights for future, past, and static covariates, separately for the model trained on synthetic (left) and on real (right) data.

Remarkably, the three most important future features are identical on both datasets: the ICON-D2 wind speed at 184 m, the HRES wind speed at 200 m, and the ICON-D2 wind speed at 127 m. The decoder thus distributes its weight across several upper-level wind speeds drawn from both NWP models rather than relying on a single layer or a single source, which is consistent with a strategy of averaging out the systematic forecast errors of the individual models. For the past (observed) inputs, historical power output is the most important predictor on both datasets, complemented by the same upper-level wind speeds.

The static features reveal the most instructive differences between the synthetic and the real data. Altitude above sea level is assigned a consistently high weight of roughly 20% in both cases, making it the most robust static discriminator between sites. This is physically plausible, as altitude acts as a proxy for complex or elevated terrain, which complicates the interdependencies between the NWP layers. Farm age, by contrast, receives a high weight on the synthetic data but a markedly lower one on the real portfolio. This is expected: in the synthetic profiles, turbine ageing is modeled explicitly, which artificially inflates the relevance of this feature, whereas its effect is weaker and noisier in reality. Research nonetheless indicates that real turbines aged 20 years or older may experience up to a 35% decline in capacity factor (Staffell and Green, 2014). The comparatively high weight of the cut-out speed on the synthetic data is likewise unexpected, since cut-out events (25–34 m/s) are physically rare; we therefore attribute it to an artifact of the synthetic generation process rather than to a genuine physical driver.

Most notably, hub height emerges as a decisive static feature on the real data, ranking second, while it plays only a minor role on the synthetic data. This is plausible: the hub height determines which atmospheric levels are physically relevant for a given turbine and therefore directly conditions the selection of the NWP wind-speed inputs. The comparatively weak role of hub height on the synthetic data likely stems from the wind-speed extrapolation method used to generate the synthetic profiles, which does not reproduce this height dependence one-to-one. This contrast offers valuable context to the prior work of Michalakopoulos et al. (2025): on real measurements, hub height is in fact among the most informative static covariates.

![](images/d12185360911e29bb9fabb560ef66c5710db3f00bb35c60e53574b159ed8c18a.jpg)

![](images/c8efb09697b0923f5a28697879f90fc8e68887ea30ca9caa565fad6095a25c85.jpg)  
Figure 3. Feature importance weights of the VSN for the model trained on synthetic (left) and real (right) data.

## Performance on Test Data

Tables 5 and 6 report the normalized performance metrics for all configurations across both test periods, separately for the synthetic (100- and 50-farm sets) and the real portfolio. The static global model achieves the best accuracy throughout, reaching $\mathrm { R } ^ { 2 } = 8 6 . 2 \% , \mathrm { R M S E } = 8 . 0 \%$ and $\mathrm { \dot { M } A E } = 4 . 7 \%$ on the synthetic late-summer and early-autumn window, and remaining the strongest model in the more challenging winter period. Performance differences between the 100-farm and the 50-farm set are small, but the 100-farm model is marginally better, indicating that it gains persite accuracy due to the increased training sample size.

Two observations stand out. First, the linear regression on NWP inputs reaches $\mathrm { R } ^ { 2 } \approx 6 1 \mathrm { - } 7 1 \%$ , well below the TFT models, confirming that the task is non-trivial and that the marginal value of the TFT architecture itself is substantial relative to a classical statistical predictor. Second, and most importantly, the results on the real portfolio confirm the synthetic findings: the static global model again outperforms both the local baseline and the no-static variant in both seasons, despite the raw, uncleaned nature of the real SCADA data, although in the spring window the margin over the local model is small (72.7 vs. 72.1 in R<sup>2</sup>).

It should be emphasized that the real-world data was used without any prior cleaning. The signal therefore contains curtailment, maintenance shutdowns, outages, and grid-induced deviations that are physically unpredictable from NWP inputs alone and impose a hard ceiling on the attainable accuracy. Filtering such artifacts, for instance via density-based clustering such as DBSCAN, together with a longer training history and accurate turbine master data, can be expected to improve the achievable performance further. That the global model already prevails under these conservative conditions underlines the robustness of the approach.

Table 5. Normalized performance on the synthetic portfolio across both test periods, for the 100- and 50-farm evaluation sets. Best $\mathrm { R } ^ { 2 }$ per column in bold; “–” marks configurations not evaluated on the respective set.
<table><tr><td></td><td colspan="3">Test Period 1 (100 farms) Aug-Oct 2025</td><td colspan="3">Nov 2025–Jan 2026</td><td colspan="3">Aug-Oct 2025</td><td colspan="3">Test Period 2 (100 farms) Test Period 1 (50 farms) Test Period 2 (50 farms) Nov 2025–Jàn 2026</td></tr><tr><td>Model</td><td> $\mathrm { R ^ { 2 } }$ </td><td>RMSE</td><td>MAE</td><td> $\mathrm { R ^ { 2 } }$ </td><td>RMSE</td><td>MAE</td><td> $\mathrm { R ^ { 2 } }$ </td><td>RMSE</td><td>MAE</td><td> $\mathrm { R ^ { 2 } }$ </td><td>RMSE</td><td>MAE</td></tr><tr><td>Global-100 (static)</td><td>86.2</td><td>8.0</td><td>4.7</td><td>79.6</td><td>17.7</td><td>10.7</td><td>85.8</td><td>8.3</td><td>4.9</td><td>79.9</td><td>18.0</td><td>11.1</td></tr><tr><td>Global-100 (no static)</td><td>84.2</td><td>8.4</td><td>5.0</td><td>76.6</td><td>18.6</td><td>11.3</td><td>84.1</td><td>8.7</td><td>5.2</td><td>77.2</td><td>19.0</td><td>11.7</td></tr><tr><td>Global-50 (static)</td><td></td><td></td><td>一</td><td></td><td></td><td></td><td>85.5</td><td>8.3</td><td>5.1</td><td>79.8</td><td>18.1</td><td>11.2</td></tr><tr><td>Local</td><td>85.2</td><td>8.3</td><td>5.1</td><td>78.7</td><td>18.0</td><td>11.3</td><td>84.7</td><td>8.6</td><td>5.3</td><td>79.1</td><td>18.3</td><td>11.5</td></tr><tr><td>LinReg</td><td>71.2</td><td>11.4</td><td>8.5</td><td>61.4</td><td>23.7</td><td>18.1</td><td>71.2</td><td>11.7</td><td>8.8</td><td>62.5</td><td>24.3</td><td>18.7</td></tr></table>

Table 6. Normalized performance on the real portfolio (11 farms) across both test periods. Best $\mathrm { R } ^ { 2 }$ per period in bold.
<table><tr><td rowspan="2">Model</td><td colspan="3">Test Period 1 Dec 2023–Feb 2024</td><td colspan="3">Test Period 2 Mar-May 2024</td></tr><tr><td> $\mathrm { R ^ { 2 } }$ </td><td>RMSE</td><td>MAE</td><td> $\mathrm { R ^ { 2 } }$ </td><td>RMSE</td><td>MAE</td></tr><tr><td>Global (static)</td><td>76.2</td><td>15.3</td><td>10.7</td><td>72.7</td><td>11.6</td><td>7.7</td></tr><tr><td>Global (no static)</td><td>72.6</td><td>16.6</td><td>12.2</td><td>68.1</td><td>12.5</td><td>8.3</td></tr><tr><td>Local</td><td>74.2</td><td>16.2</td><td>11.5</td><td>72.1</td><td>11.8</td><td>8.0</td></tr><tr><td>LinReg</td><td>63.5</td><td>19.2</td><td>14.3</td><td>62.2</td><td>13.6</td><td>10.3</td></tr></table>

Table 7 reports the optimal hyperparameters of the best cross-validated model, while Table 8 lists the training cost of each configuration. Although the local models can be trained quickly in parallel, the global approach consolidates the entire fleet into a single model trained in a few GPU-hours, drastically reducing the number of models to maintain and deploy.

Table 7. Optimal hyperparameters of the 100-farm model.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>batch_size</td><td>46</td></tr><tr><td>learning_rate</td><td> $1 . 9 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>n_heads hidden_dim</td><td>8 40</td></tr><tr><td>dropout</td><td>0.0992</td></tr><tr><td>num_Istm_layers</td><td></td></tr><tr><td></td><td>1</td></tr><tr><td>static_embed_dim</td><td>128</td></tr><tr><td>clipnorm</td><td>6.9179</td></tr></table>

Table 8. Training cost per configuration on a single RTX A6000, split by test period (TP , TP ). Each sample comprises a known-input window of $9 6 \times 5 ,$ an observed window of $4 8 \times 1$ , 7 static features, and a 48-step target. For the local models, N is the per-site training-set size (the data is partitioned across the sites), whereas the listed duration is the total wall-clock for all sites trained concurrently on one GPU (4 parallel runs at 0.25 GPU each for the synthetic and 10 at 0.1 GPU each for the real data); all other configurations use a full GPU.
<table><tr><td>Data</td><td>Model</td><td> $\mathsf { D u r a t i o n } _ { \mathrm { T P } _ { 1 } }$  1</td><td> $\mathsf { D u r a t i o n } _ { \mathrm { T P } _ { 2 } }$ </td><td> $N _ { \mathrm { T P _ { 1 } } }$ </td><td> $N _ { \mathrm { T P _ { 2 } } }$ </td></tr><tr><td rowspan="4">Synthetic</td><td>100 static</td><td>2h12m</td><td>1 h 46 m</td><td>292k</td><td>329k</td></tr><tr><td>100 no static</td><td>2h12m</td><td>1 h 46 m</td><td>292k</td><td>329k</td></tr><tr><td>50 static</td><td>55m</td><td>1h13m</td><td>146k</td><td>164k</td></tr><tr><td>Local (×100)</td><td>42m</td><td>42m</td><td>≈2.9k</td><td>≈3.3k</td></tr><tr><td rowspan="3">Real</td><td>Global (static)</td><td>1 m 23 s</td><td>4m13s</td><td>5.3k</td><td>9.3k</td></tr><tr><td>Global (no static)</td><td>1 m 16 s</td><td>3 m 47 s</td><td>5.3k</td><td>9.3k</td></tr><tr><td>Local (×11)</td><td>1 m 24 s</td><td>1 m 29 s</td><td>≈0.5k</td><td>≈0.8k</td></tr></table>

## CONCLUSION

This study evaluated the efficacy of a global TFT for 48-hour-ahead wind power forecasting across a heterogeneous portfolio. The findings demonstrate that a global model can significantly outperform specialized local models, supporting a shift from single-asset to scalable multi-asset architectures.

The primary empirical result indicates that the global model outperforms the local baselines for the large majority of wind farms (87% on the synthetic and 77% on the real portfolio), and clearly surpasses the linear-regression baseline. These gains are statistically significant and, crucially, persist across two seasonal test periods and on a real operational portfolio of 11 German wind farms. However, this performance gain is critically dependent on the integration of static context variables. It was demonstrated that a global model lacking asset-specific metadata fails to generalize significantly better than local benchmarks. This confirms that explicitly modeling heterogeneity via parameters such as altitude and hub height is a prerequisite for successful cross-learning between sites.

Furthermore, the interpretability provided by the TFT’s VSN offers valuable physical insights. The feature importance analysis suggests the model employs a dual strategy: while the encoder utilizes historical power to physically reconstruct turbine behavior, the decoder combines upper-level wind speeds from both NWP models rather than relying on a single layer or source, consistent with averaging out model-specific NWP errors. Notably, the three most influential future features are identical on the synthetic and the real data, whereas the static attributions differ in an interpretable way, with hub height emerging as a decisive discriminator on real measurements.

In summary, this work illustrates that global transformer models not only reduce operational complexity by consolidating model architectures but also measurably enhance forecast accuracy for the renewable energy sector through the utilization of fleet-wide data. Future research should examine the transferability of these findings to probabilistic forecasting, assess the impact of systematic SCADA data cleaning, and explore the potential of integrating further NWP models and grid points to enhance forecast accuracy.

## ACKNOWLEDGEMENT

This work was supported by the German Federal Ministry of Research, Technology and Space [grant 13FH587KX1] (FederatedForecasts).

## NOMENCLATURE

A Rotor swept area, $\mathrm { m ^ { 2 } }$   
$C _ { p }$ Wind power coefficient   
$d _ { m o d e l }$ Latent (hidden) space dimension   
$j$ Feature index   
$k$ Iteration in cross validation   
MAE Mean absolute error normalized by installed capacity   
$m$ Number of features   
$P$ Power, W   
RMSE Root mean squared error normalized by installed capacity   
$\mathrm { R } ^ { 2 }$ Coefficient of determination (R<sup>2</sup>)   
t Time index   
v Wind speed, $\mathrm { m } \mathrm { s } ^ { - 1 }$   
$| \mathbf { x } ^ { ( k n o w n ) } |$ Length of known input vector   
$| { \bf x } ^ { ( o b s ) } |$ Length of observed input vector   
$\nu _ { \ast } ^ { ( j ) }$ Selection weight of feature j at t   
$\xi _ { t } ^ { ( j ) }$ Processed embedding vector of feature $j$ at t   
$\rho$ Air density, kg $\mathrm { m ^ { - 3 } }$

## REFERENCES

Junhao Cheng, Xing Luo, and Zhi Jin. Integrating domain knowledge into transformer for short-term wind power forecasting. Energy, 312:133511, 2024.

Joseph Cohen, Tianyi Xu, Youngchan Jang, Pengwei Du, Eunshin Byon, and Xun Huan. Inverted transformers for effective post-calibration in multisite wind power forecasting. Wind Energy, 28(10):e70057, 2025.

Energy Institute. Statistical review of world energy. energy institute (2024), 2024.

Fraunhofer Institute for Solar Energy Systems ISE. Public electricity generation 2024: Renewable energies cover more than 60 percent of german electricity consumption for the first time. https://www.ise.fraunhofer.de/en/pressmedia/press-releases/2025/public-electricity-generation-2024-renewable-energies-cover-more-than-60-percentof-german-electricity-consumption-for-the-first-time.html, January 2025.

Hao-Tian Guo, Lin Pan, Jian Wang, Xin-Bo Fan, Jin Li, and Zhe Liu. Short-term wind power prediction method based on tcn-gru combined model. In 2021 IEEE Sustainable Power and Energy Conference (iSPEC), pages 3764–3769. IEEE, 2021.

Guolian Hou and Pengchao Liu. Short-term prediction of wind power based on improved temporal fusion transformer model with enhanced hippopotamus optimization algorithm. Electric PowerSystems Research, 252:112464, 2026.

Weronika Jachuła and Michał Wydra. Wind power prediction in poland using temporal fusion transformers and numerical weather prediction. Bulletin of the Polish Academy of Sciences Technical Sciences, pages e155038– e155038, 2025.

Rajesh G Kavasseri and Krithika Seetharaman. Day-ahead wind speed forecasting using f-arima models. Renewable Energy, 34(5):1388–1393, 2009.

Adam Kisvari, Zi Lin, and Xiaolei Liu. Wind power forecasting–a data-driven method along with gated recurrent neural network. Renewable Energy, 163:1895–1909, 2021.

Bryan Lim, Sercan Ö Arık, Nicolas Loeff, and Tomas Pfister. Temporal fusion transformers for interpretable multihorizon time series forecasting. Internationaljournal of forecasting, 37(4):1748–1764, 2021.

Lei Liu, Xinyu Wang, Xue Dong, Kang Chen, Qiuju Chen, and Bin Li. Interpretable feature-temporal transformer for short-term wind power forecasting with multivariate time series. Applied Energy, 374:124035, 2024.

Lihua Mi, Yan Han, Lizhi Long, Hui Chen, and CS Cai. A physics-informed temporal convolutional network-temporal fusion transformer hybrid model for probabilistic wind speed predictions with quantile regression. Energy, page 136302, 2025.

Vasilis Michalakopoulos, Antonis Zakynthinos, Elissaios Sarmas, Vangelis Marinakis, and Dimitris Askounis. Hybrid short-term wind power forecasting model using theoretical power curves and temporal fusion transformers. Renewable Energy, page 124008, 2025.

Site Mo, Haoxin Wang, Bixiong Li, Zhe Xue, Songhai Fan, and Xianggen Liu. Powerformer: A temporal-based transformer model for wind power forecasting. Energy Reports, 11:736–744, 2024.

Zhewen Niu, Xiaoqing Han, Dongxia Zhang, Yuxiang Wu, and Songyan Lan. Interpretable wind power forecasting combining seasonal-trend representations learning with temporal fusion transformers architecture. Energy, 306: 132482, 2024.

Iain Staffell and Richard Green. How does wind farm performance decline with age? Renew Energy, 66:775–86, 2014. doi: 10.1016/j.renene.2013.10.041.

Wen-Chang Tsai, Chih-Ming Hong, Chia-Sheng Tu, Whei-Min Lin, and Chiung-Hsing Chen. A review of modern wind power generation forecasting technologies. Sustainability, 15(14):10757, 2023.

Liaan van Heerden, Chantelle van Staden, and HJ Vermeulen. Temporal fusion transformer for day-ahead wind power forecasting in the south african context. In 2023 IEEE International Conference on Environment and Electrical Engineering and 2023 IEEE Industrial and Commercial Power Systems Europe (EEEIC/I&CPS Europe), pages 1–5. IEEE, 2023.

Alessio Verdone, Massimo Panella, Enrico De Santis, and Antonello Rizzi. A review of solar and wind energy forecasting: From single-site to multi-site paradigm. Applied Energy, 392:126016, 2025.

Viktor Walter. Synthetic simulation data of wind power time series in Germany. https://data.mendeley.com/ datasets/xv3s82rvd2/3, 2026.

Viktor Walter and Andreas Wagner. A measurement-and era5-driven simulation framework for synthetic wind power time series accounting for turbine aging. Available at SSRN 6346310.

Huai-zhi Wang, Gang-qiang Li, Gui-bin Wang, Jian-chun Peng, Hui Jiang, and Yi-tao Liu. Deep learning based ensemble approach for probabilistic wind power forecasting. Applied Energy, 188:56–70, 2017.

Sen Wang, Yonghui Sun, Wenjie Zhang, CY Chung, and Dipti Srinivasan. Very short-term wind power forecasting considering static data: An improved transformer model. Energy, 312:133577, 2024.

Shiwei Xu, Yongjun Wang, Xinglei Xu, Guang Shi, Yingya Zheng, He Huang, and Chengqiu Hong. A multi-step wind power group forecasting seq2seq architecture with spatial–temporal feature fusion and numerical weather prediction correction. Energy, 291:130352, 2024.

Can Zhang, Xianyong Xiao, Ying Wang, Michael Z Hou, Shudong Huang, Wenxi Hu, Ming Hu, and Rui Huang. Finegrained ultra-short-term wind power forecasting based on temporal fusion transformers integrated with turbine power time series clustering. Energy, page 137995, 2025.