# SCALE-AWARE AND SPATIALLY-LOCALISED OPTIMISATION OF KILOMETRE-SCALE MLWP MODELS

James Chappell\*<sup>1</sup>, Sophie Luise Arana<sup>†2</sup>, Jack Bowyer<sup>1</sup>, Matthew Burnand<sup>1</sup>, Alejandro Coca-Castro<sup>2</sup>, Evangeline Corcoran<sup>2</sup>, Eric G. Daub<sup>2</sup>, Tom Dunstan<sup>1</sup>, Noushin Eftekhari<sup>2</sup>, Anna-Louise Ellis<sup>1</sup>, J. Scott Hosking<sup>2</sup>, Manvendra Janmaijaya<sup>2</sup>, Sam Madge<sup>1</sup>, David Salvador-Jasin<sup>2</sup>, Ben Shipway<sup>1</sup>, Nathan Simpson<sup>‡2</sup>, Oliver Strickson<sup>2</sup>, and David Walters<sup>1</sup>

<sup>1</sup>Met Office, Exeter, United Kingdom

<sup>2</sup>The Alan Turing Institute, London, United Kingdom

## ABSTRACT

Preserving fine-scale structure without degrading spatial accuracy is a central challenge for kilometre-scale machine-learning weather prediction (MLWP). We introduce a loss based on the dual-tree complex wavelet transform that constrains atmospheric fields jointly by scale, location, and orientation. We evaluate its performance using an experimental joint global–regional Fast-Net configuration, which produces 1.5 km forecasts over the UK alongside ∼ 31 km global forecasts. Relative to both mean-squared error (MSE) and an FFT-based spectral objective, the wavelet loss substantially improves spectral fidelity. Compared with MSE training, it maintains similar pointwise performance, improves neighbourhood-based verification, and produces sharper precipitation fields with more realistic intensity distributions. These results demonstrate that spatially localised wavelet constraints can preserve multiscale atmospheric structure while largely avoiding the loss of pointwise accuracy associated with globally supported spectral objectives.

## 1 Introduction

Machine-learning weather prediction (MLWP) models can produce accurate forecasts at a fraction of the computational cost of conventional numerical weather prediction [1–8]. As these models are extended to higher spatial resolutions [9, 10], a variety of regional MLWP approaches have emerged using stretched-grid [11–13], limited-area [14–17], and datasetspecific encoder–decoder architectures (as used in this study). At kilometre-scale resolutions, the training data capture local weather systems and spatially varying surface influences that are poorly represented by coarser atmospheric fields, providing information important for forecasting high-impact weather and issuing local warnings [18]. However, resolving these features in the training data does not guarantee their preservation in forecasts, regardless of model architecture.

This challenge is particularly acute for deterministic models trained using pointwise meansquared error (MSE). Kilometre-scale atmospheric fields contain intermittent and sharply localised structures, such as convective precipitation, whose precise evolution and location are difficult to predict. Because MSE strongly penalises spatial displacement errors, minimising this objective tends to favour smooth predictions over uncertain fine-scale structures, suppressing small-scale variability and reducing effective resolution [19]. While probabilistic approaches can represent uncertain fine-scale variability through ensembles of plausible realisations [20–22], deterministic models provide a computationally efficient and controlled setting for investigating how training objectives influence forecast structure.

![](images/cabbdba45dba00ac55eec23edd77902deff4ca2e1bf031333f04490ea5c2cb5e.jpg)

![](images/4435dbe1af96a0372cd65521deea6e5f2f68b6df8004fe62de60095301d4ba16.jpg)  
Figure 1: Overview of the regional MLWP architecture and wavelet training objective. (a) The model produces simultaneous forecasts on a global N320 reduced Gaussian grid and a 1.5 km regional grid centred on the UK; the example shows 6 hour accumulated precipitation valid at 2023-09-28T00Z UTC at a lead time of 24 hours. (b) The shared latent processor mesh spans the global domain and is locally refined over the regional domain. Black and orange points denote global and regional processor nodes, respectively, while curves show representative multiscale processor edges from one regional node. The edges are coloured according to their length and are represented by arcs for visual clarity. (c) During loss calculation, the prediction and target are decomposed using a J-level DTCWT. At each level, the preceding low-pass component is separated into a coarser low-pass component and six complex directional wavelet-bands. The high-pass coefficients contribute amplitude- and coherence-sensitive losses that can be weighted independently by scale, while the final low-pass residual is compared using MSE.

One approach to mitigate smoothing of deterministic forecasts is incorporating spectral information into the training objective. Subich et al. [19] introduced the adjusted mean-squared error (AMSE), which decomposes forecast errors spectrally and separates contributions associated with amplitude and coherence. Originally formulated using spherical harmonics for global models, AMSE can be adapted to regional domains using Fourier transforms, while related amplitude-based spectral losses have shown promise in probabilistic regional forecasting [12]. However, their globally supported basis functions limit their ability to constrain forecast errors locally in space. Improvements in spectral fidelity may therefore come at the expense of accurately positioning individual features, potentially degrading pointwise verification metrics such as root-mean-squared error (RMSE) [8]. This motivates objectives that preserve fine-scale variability while retaining sensitivity to the location of forecast structures.

Here, we develop a deterministic loss based on the two-dimensional dual-tree complex wavelet transform (DTCWT) [23, 24], illustrated schematically in Figure 1. By decomposing fields jointly by scale, location, and orientation, the DTCWT allows loss contributions to be calculated and weighted independently across spatial scales. The approximate shift invariance [25] and directional selectivity of the DTCWT make it well suited to localised atmospheric structures, while its complex coefficients support amplitude- and coherence-sensitive terms analogous to those used in spectral AMSE. Wavelets have previously been applied to atmospheric analysis and verification [26, 27], with recent machine-learning applications focusing primarily on precipitation, including generative downscaling [28] and a composite DTCWT–Fourier loss for global prediction [29]. Our standalone objective extends wavelet-based optimisation to multivariable, kilometre-scale forecasting, aiming to preserve fine-scale spectral variability without compromising spatial accuracy. The underlying multiscale decomposition could also provide a foundation for scale-aware probabilistic training objectives, complementing recent developments in probabilistic MLWP [30, 31].

We evaluate the proposed objective using a joint global–regional configuration developed as part of ongoing work towards a regional version of the medium-range MLWP model FastNet [7, 8]. The model simultaneously produces kilometre-scale forecasts over the UK and coarser forecasts globally, providing a controlled setting in which to compare wavelet-based training with MSE and an FFT-based spectral objective. We show that the wavelet loss preserves finescale spectral energy while maintaining strong spatial localisation of the training gradient. It achieves pointwise performance comparable to MSE, outperforms MSE under neighbourhoodbased verification, and improves spectral fidelity relative to both MSE and the purely spectral objective.

## 2 Deterministic wavelet loss

To combine scale-selective amplitude matching with greater spatial localisation, we formulate the loss in the dual-tree complex wavelet domain. Let xb and x denote predicted and target fields, respectively, on a two-dimensional input grid. Applying a J-level DTCWT to a field x produces complex directional band-pass coefficient field $, z _ { j , o } [ { \pmb x } ]$ , at each level $j = 1 , \dots , J$ and at six approximate orientations $( \pm 1 5 ^ { \circ } , \pm 4 5 ^ { \circ }$ , and $\pm 7 5 ^ { \circ } ) _ { \cdot }$ , together with a final real-valued low-pass residual, $\mathbf { \xi } _ { l , \lceil \mathbf { x } \rceil }$

$$
\mathcal { W } [ \pmb { x } ] = \left( l _ { J } [ \pmb { x } ] , \{ z _ { j , o } [ \pmb { x } ] \} _ { j = 1 , \dots , J ; o = 1 , \dots , 6 } \right) .\tag{1}
$$

We write $\widehat { \boldsymbol { z } } _ { j , o } = \boldsymbol { z } _ { j , o } [ \widehat { \pmb { x } } ]$ and $\boldsymbol { z } _ { j , o } = \boldsymbol { z } _ { j , o } [ \pmb { x } ]$ for the prediction and target coefficients, respectively, with analogous notation for the low-pass residuals.

The index $j = 1$ denotes the finest level, with increasing $j$ corresponding to progressively coarser scales. Because the transform downsamples the coefficient fields at successive levels, each level has its own coefficient grid and set of spatial indices, $S _ { j }$ . Thus, $\pmb { r } \in S _ { j }$ denotes a location on the level-j grid rather than on the original input grid.

For each high-pass scale and orientation, we calculate the band-integrated wavelet powers of the prediction and target:

$$
\widehat { P } _ { j , o } = \sum _ { \pmb { r } \in S _ { j } } \left| \widehat { z } _ { j , o } ( \pmb { r } ) \right| ^ { 2 } , \qquad P _ { j , o } = \sum _ { \pmb { r } \in S _ { j } } \left| z _ { j , o } ( \pmb { r } ) \right| ^ { 2 } .\tag{2}
$$

Following the AMSE formulation [19], the loss for each wavelet band combines an amplitude error with a coherence-dependent penalty:

$$
\mathcal { L } _ { j , o } = \left( \sqrt { \widehat { P } _ { j , o } } - \sqrt { P _ { j , o } } \right) ^ { 2 } + 2 \operatorname* { m a x } \left( \widehat { P } _ { j , o } , P _ { j , o } \right) \left( 1 - \rho _ { j , o } \right) ,\tag{3}
$$

where the normalised coherence is

$$
\rho _ { j , o } = \frac { \displaystyle \sum _ { \boldsymbol { r } \in \mathcal { S } _ { j } } \mathrm { R e } \left[ \widehat { z } _ { j , o } ( \boldsymbol { r } ) z _ { j , o } ^ { * } ( \boldsymbol { r } ) \right] } { \left( \widehat { P } _ { j , o } P _ { j , o } \right) ^ { 1 / 2 } } .\tag{4}
$$

The wavelet-band loss is summed over the six orientations and combined across scales using configurable weights, $\alpha _ { j }$ . The final low-pass residual represents the coarsest spatial structure and is compared directly using MSE. The total loss is

$$
\mathcal { L } _ { \mathrm { w a v } } = \frac { 1 } { N } \left[ \sum _ { j = 1 } ^ { J } \alpha _ { j } \sum _ { o = 1 } ^ { 6 } \mathcal { L } _ { j , o } + \left| \left| \widehat { l } _ { J } - l _ { J } \right| \right| _ { 2 } ^ { 2 } \right] ,\tag{5}
$$

where N is the number of points in the original input grid.

The two terms in Equation 3 measure distinct aspects of the error within each scale–orientation band. The amplitude term compares the band-integrated powers of the prediction and target, while the second term applies a power-weighted penalty for disagreement between their complex coefficient patterns. Prediction and target coefficients are paired at corresponding locations in the coherence numerator before their cross-power is summed, making the loss sensitive to their relative spatial arrangement as well as their band-integrated powers. The loss therefore distinguishes forecasts with similar band-integrated power but different spatial organisation.

Although Equation 5 is additive across levels and orientations, each band loss is computed from statistics aggregated over all locations in $S _ { j }$ and is therefore not a sum of independent coefficient-wise losses—the corresponding locations and gradients remain coupled within each band. Moreover, each coefficient is the response of a compactly supported wavelet filter to a neighbourhood of input values, rather than a pointwise value on the original grid. Backpropagating through these filters consequently produces more spatially localised field-space gradients than transforms with globally supported basis functions, such as the Fourier transform, as shown in Appendix A. The following subsection demonstrates how the separate level-wise terms provide scale-selective loss contributions and field-space gradients.

## 2.1 Scale selectivity

Figure 2 illustrates the scale selectivity of the wavelet loss using two contrasting constructed prediction fields designed to probe different error regimes. Case 1 is the atmospheric state 6 hours before the target time with Gaussian noise added $( \sigma = 5 \% )$ , producing errors across a broad range of spatial scales. Case 2 is obtained by applying a Gaussian blur with $\sigma = 3$ grid cells to the target, preserving its large-scale structure while selectively attenuating finer-scale variability. Uniform scale weights, $\alpha _ { j } = 1$ , are used in both cases.

The scalar loss is decomposed exactly into the wavelet-band components, $\begin{array} { r l } { { \mathcal { L } } _ { j } } & { { } = } \end{array}$ $\begin{array} { r } { N ^ { - 1 } \alpha _ { j } \sum _ { o = 1 } ^ { 6 } \mathcal { L } _ { j , o } , } \end{array}$ , and the final low-pass component, $\mathcal { L } _ { \mathrm { L P } } = N ^ { - 1 } \Vert \widehat { l } _ { J } - l _ { J } \Vert _ { 2 } ^ { 2 }$ . For each component, we also calculate its gradient with respect to the predicted field on the original input grid, $\nabla _ { \widehat { \pmb { x } } } \mathcal { L } _ { j }$ , by back-propagating through the wavelet transform. These field-space gradients represent the signed first-order sensitivity of each scalar component to changes in the predicted field, revealing the scale and spatial structure of the training signal presented to the forecasting model. The total field-space gradient is the sum of the low-pass and wavelet-band gradients, although the eventual parameter update also depends on back-propagation through the model and on the optimiser.

The two cases produce clearly different responses. For the perturbed time-offset prediction (Case 1, Fig. 2b), broad structural errors dominate the scalar objective, with the low-pass component contributing 80% of the total loss. Its share of the summed component-gradient norms is much lower, at 46%, with the remaining gradient magnitude distributed across the wavelet-band levels and increasing towards the coarser levels. The wavelet components therefore provide substantial scale-dependent training signals even when the scalar loss is dominated by its low-pass component.

By contrast, the blurred prediction (Case 2, Fig. 2c) retains the target’s large-scale structure, giving a negligible low-pass contribution to both the loss and gradient magnitude. Its response is concentrated at wavelet-bands $j = 2$ and $j = 3 ,$ , which together account for approximately 81% of the scalar loss and 85% of the summed component-gradient norms. This concentration reflects both the range of scales attenuated by the Gaussian filter and the distribution of variance in the target field: although the finest scales undergo strong fractional attenuation, the 2 m temperature field contains more variance within the affected wavelet-bands at $j = 2$ and $j = 3$ than at $j = 1$

![](images/be61d9e847350c718c24e61e071a396e48f03bba2cee06558ab9078f5cb1d109.jpg)

(a) Inputs and component summaries  
![](images/2866a5b8d2fd66814949f0786fd6110a1d372b92a5e31de10a2ef1eac6a8168f.jpg)

![](images/a54786bc23260ce7fa582a0457d87910943777d3e1a6fc0228e9d3d31e0fc2e5.jpg)  
Figure 2: Scale-wise decomposition of the wavelet loss for two constructed prediction fields. (a) Target field and component-wise summaries for a perturbed 6 hour time-offset prediction (Case 1) and a Gaussian-blurred target with $\sigma = 3$ grid cells (Case 2). The scalar-loss percentages give the exact contribution of the final low-pass term and each high-pass wavelet level to the total loss. The gradient percentages give each component-gradient $L _ { 2 }$ norm as a fraction of the sum of the component-gradient $L _ { 2 }$ norms; they therefore provide a relative measure of gradient magnitude rather than an additive decomposition of the norm of the total gradient. (b,c) The two constructed predictions and the gradients of the total loss and its low-pass and level-wise components with respect to the predicted field on the original input grid. Each map shows the signed first-order sensitivity of the corresponding scalar loss component and therefore the field-space training signal supplied by that component.

Together, these examples show that the wavelet objective produces scale-dependent scalar losses and field-space training signals. Broad structural errors activate the low-pass and coarserscale terms, whereas selectively removing finer-scale variability concentrates the response in the affected fine-scale wavelet-band components. The spatial localisation of the complete wavelet-loss gradient is compared with that produced by MSE and FFT-AMSE in Appendix A.

## 2.2 Evolution of scale-wise loss contributions during training

Having demonstrated the scale selectivity of the loss using controlled perturbations, we next examine how its scale-wise composition evolves during model training. Figure 3 shows the magnitude and scale composition of the regional wavelet validation loss during the early adaptation of the model to the regional domain for three variables. These variables were selected to span a range of characteristic spatial structures represented within kilometre-scale atmospheric fields: 500 hPa geopotential is dominated by smooth, large-scale variability; 2 m temperature contains additional structure associated with orography, surface heterogeneity, and land–sea contrasts; and precipitation is intermittent and contains substantial variability across intermediate and fine scales. The figure covers the first 25% of the regional single-step training stage (Stage 3a; see Appendix B for the complete training schedule).

For 500 hPa geopotential, the low-pass MSE contributes more than 90% of the objective throughout the displayed period, and the pointwise MSE and complete wavelet loss (Eq. 5) follow similar trajectories. Although the absolute losses decrease at every wavelet level, the low-pass term remains dominant, consistent with the concentration of geopotential variance at large spatial scales.

![](images/abca40b9186f663919f0504dbd2bc6e14601de389b060d3b2999b3acf69c0b5b.jpg)  
Figure 3: Evolution of validation losses and relative scale-wise wavelet-loss contributions over the first 25% of Stage 3a training for (a) 500 hPa geopotential, (b) 2 m temperature, and (c) 6h accumulated total precipitation. All quantities are evaluated over the regional UKV domain using validation forecasts from the wavelet-trained model; the global ERA5 component is not included. Upper panels show pointwise MSE and the complete wavelet objective, both normalised by the wavelet validation loss at the first recorded training step. Lower panels show the low-pass MSE component and individual wavelet-band contributions as fractions of the total regional wavelet validation loss. Here, j = 1 denotes the finest wavelet-band, with increasing j corresponding to progressively coarser scales. All decomposition levels use uniform scale weights, $\alpha _ { j } = 1$

For 2 m temperature and precipitation, the absolute low-pass and wavelet-band losses also decrease, but the low-pass error falls more rapidly. The wavelet-band levels therefore account for an increasing fraction of the remaining wavelet objective. For temperature, the continued separation between the wavelet validation loss and pointwise MSE indicates residual scaledependent errors that are not distinguished by the pointwise metric. These contributions are consistent with the greater regional and fine-scale variability of near-surface temperature. Precipitation contains substantial scale-dependent error from the beginning of training. Its pointwise MSE flattens comparatively early, whereas the wavelet validation loss continues to decrease, indicating continued improvement in aspects of the precipitation field represented by the multiscale objective.

The three examples demonstrate that the distribution of the wavelet loss reflects the characteristic spatial scales of each variable. Broad-scale fields remain dominated by the low-pass component, whereas variables containing greater regional and fine-scale variability receive progressively larger contributions from the wavelet bands. The increasing relative wavelet-band contributions reflect differences in the rates at which errors are reduced, rather than growth in scale-dependent error. These diagnostics characterise the model’s early adaptation to the regional domain and should not be interpreted as the complete evolution of the loss over the full training schedule.

## 3 Forecast evaluation

We evaluate three variants of a developmental joint global–regional configuration for a regional FastNet [7, 8] model, implemented using the Anemoi [6] framework. The variants share the same architecture and differ principally in their training objectives: pointwise MSE, an FFTbased variant of the AMSE loss introduced by Subich et al. [19], and the wavelet loss introduced here. For the wavelet-trained model, all decomposition levels are weighted uniformly, with $\alpha _ { j } = 1$ . Each model produces forecasts on a global N320 reduced Gaussian grid (∼ 31 km)

and a 1.5 km regional grid centred on the UK, using a separate encoder–decoder pair for each dataset. Verifying forecasts are initialised every 6 hours, with output at 6 hour intervals to a lead time of 240 hours, and are produced for the entirety of 2023. Full details of the model configuration and the training schedule are provided in Appendix B.

Forecasts are verified against surface observations using pointwise RMSE and the High-Resolution Assessment (HiRA) framework [32]. HiRA treats the model grid points surrounding each observation as a pseudo-ensemble, which we evaluate using the continuous ranked probability score (CRPS). This neighbourhood-based approach provides greater tolerance to small spatial displacement errors than pointwise verification and assesses whether forecasts contain locally relevant structure. We calculate HiRA CRPS using neighbourhoods of $3 \times 3 , 7 \times 7 ,$ and 11×11 model grid points. Details of the observation processing and forecast–observation matching are provided in Appendix C, with additional surface and upper-air verification presented in Appendix D.

## 4 Results

## 4.1 Verification against surface observations

![](images/16fa5234791bd8256015585c84ca7204abe7e49fc22328bb8c6b2ddec57256c5.jpg)  
Figure 4: Verification of the MSE- and wavelet-trained models against surface observations as a function of lead time. The top row shows HiRA CRPS for $3 \times 3 , 7 \times 7$ and $1 1 \times 1 1$ model-grid neighbourhoods; the bottom row shows pointwise RMSE. Columns show (a,e) 2 m temperature, (b,f) 10 m wind speed, (c,g) 10 m wind-vector error and (d,h) 6 hour accumulated precipitation. Curves are aggregated over forecasts initialised between 1 January and 21 December 2023. Lower values indicate better performance.

The models show similar overall skill but their relative performance depends consistently on the verification metric (Figure 4). The wavelet-trained model generally achieves lower HiRA CRPS, with the clearest improvements for both 10 m wind diagnostics across all three neighbourhood sizes. Smaller improvements are present for 2 m temperature and precipitation over parts of the forecast, although the differences become less systematic at longer leads. In contrast, the MSE-trained model generally has slightly lower pointwise RMSE, particularly for precipitation and, to a lesser extent, temperature and wind diagnostics.

This contrast is consistent with the wavelet-trained model retaining more local spatial variability without uniformly improving the exact placement of forecast features. Retaining this structure can improve neighbourhood-based scores while remaining vulnerable to the double penalty imposed by pointwise RMSE. However, lower HiRA CRPS may also result from improved local bias, intensity or variability, and should not be interpreted solely as evidence of displaced but otherwise accurate forecast features.

![](images/d6debfa002275445e0c9835c15b849d8e2d9a2a282f6b0c84945dd5174332c36.jpg)

![](images/a31e8bbb32c306b1ab9756f287162036155c411dc08f521c3a0ca6f63609aa3b.jpg)

![](images/7cc95c3f23259d78bc59dbde1435882cb1e86229efc339400f35c506800aa00a.jpg)  
Figure 5: Zonal power spectra of forecasts from the MSE- (blue), FFT-AMSE- (purple), and wavelet-trained (orange) models at a lead time of 24 hours, verified against UKV operational analyses (black, dashed). Columns show (a) 2 m temperature, (b) 10 m wind speed, and (c) 6 hour accumulated precipitation. Upper panels show the absolute spectra, while lower panels show $\log _ { 1 0 } ( P _ { f } / P _ { r } )$ , where $P _ { f }$ and $\bar { P _ { r } }$ are the forecast and reference powers, respectively. Zero indicates equal forecast and reference power, while negative values indicate a forecast power deficit. Vertical dotted lines mark the internal processor-mesh spacing, $\Delta x _ { \mathrm { m e s h } } = 4$ km, and wavelengths corresponding to $2 \Delta x _ { \mathrm { m e s h } }$ $4 \Delta x _ { \mathrm { m e s h } }$ , and $8 \Delta x _ { \mathrm { m e s h } }$

## 4.2 Spectral fidelity

The spectral differences among the models are substantially larger than their differences in pointwise and neighbourhood-based verification scores, illustrating the value of training objectives that explicitly constrain multiscale structure. Figure 5 compares the three training objectives at a lead time of 24 hours; similar results are observed across lead times and are provided in Appendix D for reference.

The MSE-trained model develops a systematic power deficit towards shorter wavelengths, consistent with the smoothing associated with pointwise optimisation. This deficit is modest for 2 m temperature but pronounced for 10 m wind speed and accumulated precipitation. FFT-AMSE recovers some of the variance lost under MSE training, but the wavelet-trained forecasts remain closer to the UKV reference over a broader range of intermediate and short wavelengths. The distinction is clearest for wind speed and precipitation, for which wavelet training produces a more accurate distribution of variance across the resolved scales.

The MSE-trained spectra also contain regions of excess power aligned with wavelengths associated with the internal processor mesh, most clearly for precipitation. These features are absent or strongly suppressed in the wavelet-trained spectra, indicating that the improvement is not simply an indiscriminate increase in short-wavelength variance. Instead, the wavelet objective both reduces spectral power deficits and limits non-physical mesh-related artefacts [8, 33].

## 4.3 Precipitation forecasts

Precipitation provides a stringent test of km-scale MLWP models because forecast quality depends on representing both the intensity and spatial organisation of highly intermittent smallscale features. The improved spectral fidelity obtained with wavelet training is reflected in both the structure and distribution of forecast precipitation (Figure 6).

The wavelet-trained model produces precipitation fields that are sharper and more intense than those generated by the MSE-trained model (Figure $\mathrm { 6 a , c ) }$ . The difference is particularly evident for instantaneous rainfall rates, where the MSE forecast is dominated by diffuse weak precipitation, while the wavelet-trained forecast contains more coherent and spatially concentrated features that more closely resemble the radar observations. Similar behaviour is evident for accumulated precipitation, with stronger local maxima and sharper gradients.

![](images/ccacf0b4ea507f58d1313d4a7a90aeb68e9fc3ce4710ea393ecf6d0c5933e1e2.jpg)  
Figure 6: Precipitation forecast characteristics. (a) Mean rainfall rate over the preceding 6 hours for the case valid at 0000 UTC on 14 October 2023, calculated by dividing the 6 hour precipitation accumulations by six. The panels show the UKV operational NWP forecast at a lead time of 6 hours and the wavelet- and MSE-trained model forecasts at a lead time of 24 hours. (b) Quantile–quantile comparison of forecast and analysis 6 hour accumulated precipitation over the 2023 verification period, comparing the MSE- (blue), FFT-AMSE- (purple), and wavelet-trained (orange) models at lead times of 24 (solid), 72 (dashed), and 120 (dotted) hours (c) Instantaneous rainfall rate for the same case and valid time, with the model forecasts at a lead time of 24 hours verified against radar-derived rainfall estimates. Panels (a) and (c) use the same colour scale in mm h<sup>−1</sup>, such that a given colour represents the same rainfall rate in both rows. (d) As in (b), but for instantaneous rainfall rate verified against radar-derived rainfall estimates. In (b) and (d), the grey dashed diagonal indicates perfect agreement.

These qualitative differences are reflected in the precipitation distributions. For both accumulated precipitation and instantaneous rainfall rate, the wavelet-trained model more closely reproduces the upper tail of the observed distribution than either the MSE- or FFT-AMSE-trained models (Figure 6b,d). FFT-AMSE partially reduces the underprediction of heavy precipitation relative to MSE, while wavelet training provides a further improvement. This greater intensity and fine-scale variability is accompanied by improved spatial skill: at a lead time of 24 hours, the wavelet-trained model achieves higher Fractions Skill Scores than both alternatives across most precipitation thresholds and neighbourhood sizes, with the clearest gains for heavier precipitation and broader neighbourhoods (Appendix D.3). These combined improvements are consistent with benefits from the spatially localised wavelet representation beyond those obtained using Fourier-based spectral constraints alone.

Despite these improvements, precipitation remains a clear weakness of all model variants. The highest quantiles are systematically underestimated, particularly for instantaneous rainfall rate, and substantial departures from one-to-one agreement persist even at short lead times. Furthermore, although wavelet training produces sharper accumulated precipitation fields, neither MLWP model fully reproduces the elongated streak-like structures evident in the operational forecast that result from precipitation features advecting during the six-hour accumulation period. The wavelet objective therefore improves the representation of heavy precipitation but does not eliminate important shortcomings in the temporal organisation, morphology and intensity of precipitation.

## 5 Conclusions

We have introduced a deterministic training objective for machine-learning weather prediction based on the two-dimensional dual-tree complex wavelet transform. The DTCWT decomposes atmospheric fields jointly by scale, location, and orientation, allowing amplitude- and coherencesensitive loss terms to be evaluated independently at each spatial scale. Its compactly supported basis functions also produce a more spatially localised training gradient than an equivalent Fourier-domain objective, while its approximate shift invariance and directional selectivity are well suited to sharply localised atmospheric structures.

Applied to an MLWP model that simultaneously produces 1.5 km forecasts over the UK and ∼ 31 km forecasts globally, the wavelet loss provides a better balance between spatial accuracy and multiscale fidelity than either pointwise MSE or FFT-AMSE. Within the regional domain, the wavelet-trained model retains pointwise performance close to the MSE baseline, generally improves neighbourhood-based verification, and outperforms FFT-AMSE. The distinction is clearest in spectral verification: across the evaluated variables and lead times, the wavelettrained forecasts remain closer to the reference spectra over a broader range of resolved scales than either baseline. Wavelet training substantially reduces the intermediate- and small-scale power deficits produced by MSE and suppresses non-physical excess power aligned with the internal processor-mesh scales. Compared with FFT-AMSE, it more accurately reproduces the reference distribution of variance across spatial scales without incurring the same degradation in pointwise performance.

These improvements are reflected in sharper, more spatially concentrated precipitation forecasts and more realistic intensity distributions than those produced by either baseline. Nevertheless, all three models underestimate the most intense rainfall, and wavelet training does not fully recover the morphology and temporal organisation of accumulated precipitation. The wavelet loss therefore reduces, but does not eliminate, the trade-off between preserving fine-scale variability and maintaining spatial accuracy. Future work should assess its sensitivity to scale weighting, evaluate its generality across additional regions and weather regimes, and investigate its use within probabilistic forecasting systems that represent both fine-scale structure and forecast uncertainty.

## Acknowledgments

The development of FastNet, including this work and its contributors, is funded by The Alan Turing Institute and the Met Office AWAKEN project. The AWAKEN project is led by the UK Met Office and is funded through the UK Government’s International Science Partnerships Fund (ISPF), with support from the Department for Business, Innovation, Science and Trade (DBIST) and the Foreign, Commonwealth and Development Office (FCDO). The authors acknowledge the use of resources provided by the Isambard-AI National AI Research Resource (AIRR). Isambard-AI is operated by the University of Bristol and is funded by the UK Government’s Department for Business, Innovation, Science and Trade (DBIST) via UK Research and Innovation; and the Science and Technology Facilities Council [ST/AIRR/I-A-I/1023]. The authors gratefully acknowledge all contributors to the Anemoi framework.

## Statement on the use of generative AI

All results, analysis and scientific conclusions in this manuscript were produced by the authors.   
Microsoft 365 Copilot was used for editorial assistance to improve clarity and readability.

## References

[1] R. Keisler. Forecasting Global Weather with Graph Neural Networks. 2022. arXiv: 2202.07575 [physics.ao-ph]. URL: https://arxiv.org/abs/2202.07575.

[2] J. Pathak, S. Subramanian, P. Harrington, S. Raja, A. Chattopadhyay, M. Mardani, T. Kurth, D. Hall, Z. Li, K. Azizzadenesheli, P. Hassanzadeh, K. Kashinath, and A. Anandkumar. FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators. 2022. arXiv: 2202.11214 [physics.ao-ph]. URL: https://arxiv.org/abs/2202.11214.

[3] K. Bi, L. Xie, H. Zhang, X. Chen, X. Gu, and Q. Tian. “Accurate medium-range global weather forecasting with 3D neural networks”. In: Nature 619.7970 (July 2023), pp. 533– 538.

[4] R. Lam, A. Sanchez-Gonzalez, M. Willson, P. Wirnsberger, M. Fortunato, F. Alet, S. Ravuri, T. Ewalds, Z. Eaton-Rosen, W. Hu, A. Merose, S. Hoyer, G. Holland, O. Vinyals, J. Stott, A. Pritzel, S. Mohamed, and P. Battaglia. “Learning skillful mediumrange global weather forecasting”. In: Science 382.6677 (2023), pp. 1416–1421. DOI: 10.1126/science.adi2336.

[5] L. Chen, X. Zhong, F. Zhang, Y. Cheng, Y. Xu, Y. Qi, and H. Li. “FuXi: a cascade machine learning forecasting system for 15-day global weather forecast”. In: npj Climate and Atmospheric Science 6.1 (Nov. 2023), p. 190.

[6] S. Lang, M. Alexe, M. Chantry, J. Dramsch, F. Pinault, B. Raoult, M. C. A. Clare, C. Lessig, M. Maier-Gerber, L. Magnusson, Z. B. Bouallègue, A. P. Nemesio, P. D. Dueben, A. Brown, F. Pappenberger, and F. Rabier. AIFS – ECMWF’s data-drivenforecasting system. 2024. arXiv: 2406.01465 [physics.ao-ph]. URL: https://arxiv.org/ abs/2406.01465.

[7] E. G. Daub, T. Dunstan, T. Bennett, M. Burnand, J. Chappell, A. Coca-Castro, N. Eftekhari, J. S. Hosking, M. Janmaijaya, J. Lillis, D. Salvador-Jasin, N. Simpson, O. T. Strickson, R. S.-Y. Chan, M. Elmasri, L. A. France, S. Madge, A. Owen, J. Robinson, A. A. Scaife, D. Walters, P. Yatsyshin, T. McCaie, L. Bokeria, H. Brown, T. Dodds, D. Llewellyn-Jones, S. Moreton, T. Potter, I. Stenson, L. van Zeeland, K. Bett-Williams, and K. I. Dale. Technical overview and architecture of the FastNet Machine Learning weather prediction model, version 1.0. 2025. arXiv: 2509.17658 [physics.ao-ph]. URL: https://arxiv.org/abs/2509.17658.

[8] T. Dunstan, O. Strickson, T. Bennett, J. Bowyer, M. Burnand, J. Chappell, A. Coca-Castro, K. I. Dale, E. G. Daub, N. Eftekhari, M. Janmaijaya, J. Lillis, D. Salvador-Jasin, N. Simpson, R. S.-Y. Chan, M. Elmasri, L. A. France, S. Madge, S. L. Arana, L. Bokeria, H. Brown, E. Corcoran, T. Dodds, A.-L. Ellis, T. Lazauskas, D. Llewellyn-Jones, T. McCaie, S. Moreton, T. Potter, J. Robinson, A. A. Scaife, I. Stenson, D. Walters, K. Bett-Williams, L. van Zeeland, P. Yatsyshin, and J. S. Hosking. “FastNet: Improving the Physical Consistency of Machine Learning Weather Prediction Models through Loss Function Design”. In: Artificial Intelligencefor the Earth Systems 5.3 (2026), p. 250090. DOI: 10.1175/AIES-D-25-0090.1.

[9] C. Bodnar, W. P. Bruinsma, A. Lucic, M. Stanley, A. Allen, J. Brandstetter, P. Garvan, M. Riechert, J. A. Weyn, H. Dong, J. K. Gupta, K. Thambiratnam, A. T. Archibald, C.-C. Wu, E. Heider, M. Welling, R. E. Turner, and P. Perdikaris. “A foundation model for the Earth system”. In: Nature 641.8065 (May 2025), pp. 1180–1187.

[10] T. Goecke, M. Jacob, F. Prill, M. Denhard, F. Fundel, J. Keller, R. Potthast, H. Reich, B. Seegebrecht, S. Ulbrich, A. Valmassoi, and S. Wahl. AICON: An operational global machine learning weatherforecasting model. 2026. arXiv: 2608.24651 [physics.ao-ph]. URL: https://arxiv.org/abs/2608.24651.

[11] T. N. Nipen, H. H. Haugen, M. S. Ingstad, E. M. Nordhagen, A. F. S. Salihi, P. Tedesco, I. A. Seierstad, J. Kristiansen, S. Lang, M. Alexe, J. Dramsch, B. Raoult, G. Mertes, and M. Chantry. “Regional Data-Driven Weather Modeling with a Global Stretched Grid”. In: Artificial Intelligence for the Earth Systems 5.2 (2026), p. 250001. DOI: 10.1175/AIES-D-25-0001.1.

[12] E. M. Nordhagen, H. H. Haugen, A. F. S. Salihi, M. S. Ingstad, T. N. Nipen, I. A. Seierstad, I.-L. Frogner, M. Clare, S. Lang, M. Chantry, P. Dueben, and J. Kristiansen. High-Resolution Probabilistic Data-Driven Weather Modeling with a Stretched-Grid. 2025. arXiv: 2511.23043 [physics.ao-ph]. URL: https://arxiv.org/abs/2511. 23043.

[13] J. S. Wijnands, M. Van Ginderachter, B. François, S. Buurman, P. Termonia, and D. Van den Bleeken. “Data-driven regional weather forecasting: A comparison of stretchedgrid and limited-area modelling”. In: Machine Learning with Applications 24 (2026), p. 100887. ISSN: 2666-8270. DOI: https://doi.org/10.1016/j.mlwa.2026. 100887.

[14] J. Oskarsson, T. Landelius, and F. Lindsten. Graph-based Neural Weather Predictionfor Limited Area Modeling. 2023. arXiv: 2309.17370 [cs.LG]. URL: https://arxiv. org/abs/2309.17370.

[15] S. Adamov, J. Oskarsson, L. Denby, T. Landelius, K. Hintz, S. Christiansen, I. Schicker, C. Osuna, F. Lindsten, O. Fuhrer, and S. Schemm. Building Machine Learning Limited Area Models: Kilometer-Scale Weather Forecasting in Realistic Settings. 2025. arXiv: 2504.09340 [physics.ao-ph]. URL: https://arxiv.org/abs/2504.09340.

[16] P. Xu, X. Zheng, T. Gao, Y. Wang, J. Yin, J. Zhang, X. Zhang, S. Luo, Z. Wang, Z. Zhang, X. Hu, and X. Chen. “An artificial intelligence-based limited area model for forecasting of surface meteorological variables”. In: Communications Earth & Environment 6.1 (May 2025), p. 372.

[17] E. Larsson, J. Oskarsson, T. Landelius, and F. Lindsten. Diffusion-LAM: Probabilistic Limited Area Weather Forecasting with Diffusion. 2025. arXiv: 2502.07532 [cs.LG]. URL: https://arxiv.org/abs/2502.07532.

[18] P. Clark, N. Roberts, H. Lean, S. P. Ballard, and C. Charlton-Perez. “Convectionpermitting models: a step-change in rainfall forecasting”. In: Meteorological Applications 23.2 (2016), pp. 165–181. DOI: https://doi.org/10.1002/met.1538.

[19] C. Subich, S. Z. Husain, L. Separovic, and J. Yang. “Fixing the Double Penalty in Data-Driven Weather Forecasting Through a Modified Spherical Harmonic Loss Function”. In: Forty-second International Conference on Machine Learning. 2025. URL: https: //openreview.net/forum?id=YNh77OLRid.

[20] F. Alet, I. Price, A. El-Kadi, D. Masters, S. Markou, T. R. Andersson, J. Stott, R. Lam, M. Willson, A. Sanchez-Gonzalez, and P. Battaglia. Skillful joint probabilistic weatherforecastingfrom marginals. 2025. arXiv: 2506.10772 [cs.LG]. URL: https: //arxiv.org/abs/2506.10772.

[21] B. Bonev, T. Kurth, A. Mahesh, M. Bisson, J. Kossaifi, K. Kashinath, A. Anandkumar, W. D. Collins, M. S. Pritchard, and A. Keller. FourCastNet 3: A geometric approach to probabilistic machine-learning weatherforecasting at scale. 2025. arXiv: 2507.12144 [cs.LG]. URL: https://arxiv.org/abs/2507.12144.

[22] S. Lang, M. Alexe, M. C. A. Clare, C. Roberts, R. Adewoyin, Z. Ben Bouallègue, M. Chantry, J. Dramsch, P. D. Dueben, S. Hahner, P. Maciel, A. Prieto-Nemesio, C. O’Brien, F. Pinault, J. Polster, B. Raoult, S. Tietsche, and M. Leutbecher. “AIFS-CRPS: ensemble forecasting using a model trained with a loss function based on the continuous ranked probability score”. In: npj Artificial Intelligence 2.1 (Feb. 2026), p. 18. ISSN: 3005-1460. DOI: 10.1038/s44387-026-00073-7.

[23] N. Kingsbury. “The dual-tree complex wavelet transform: A new efficient tool for image restoration and enhancement”. In: 9th European Signal Processing Conference (EUSIPCO 1998). 1998, pp. 1–4.

[24] I. Selesnick, R. Baraniuk, and N. Kingsbury. “The dual-tree complex wavelet transform”. In: IEEE Signal Processing Magazine 22.6 (2005), pp. 123–151. DOI: 10.1109/MSP. 2005.1550194.

[25] N. Kingsbury. “Shift invariant properties of the dual-tree complex wavelet transform”. In: 1999 IEEE International Conference on Acoustics, Speech, and Signal Processing. Proceedings. ICASSP99 (Cat. No.99CH36258). Vol. 3. 1999, 1221–1224 vol.3. DOI: 10.1109/ICASSP.1999.756198.

[26] R. W. Scovell. “Applications of directional wavelets, Universal Multifractals and anisotropic scaling in ensemble nowcasting; a review of methods with case-studies”. In: Quarterly Journal ofthe Royal Meteorological Society 146.730 (2020), pp. 2066–2095. DOI: https://doi.org/10.1002/qj.3780.

[27] S. Buschow and P. Friederichs. “SAD: Verifying the scale, anisotropy and direction of precipitation forecasts”. In: Quarterly Journal of the Royal Meteorological Society 147.735 (2021), pp. 1150–1169. DOI: https://doi.org/10.1002/qj.3964.

[28] C. Yi, M. Yu, W. Qian, Y. Wen, and H. Yang. “Efficient Kilometer-Scale Precipitation Downscaling With Conditional Wavelet Diffusion”. In: Journal ofGeophysical Research: Machine Learning and Computation 3.2 (2026), e2025JH000941. DOI: https://doi. org/10.1029/2025JH000941.

[29] S. An, Y. Kim, D. Cho, and Y.-G. Ham. “Toward spatially sharper precipitation prediction via global-local frequency guidance”. In: npj Climate and Atmospheric Science 9.1 (June 2026), p. 200.

[30] S. Lang, M. Leutbecher, and P. Maciel. A multi-scale loss formulation for learning a probabilistic model with proper score optimisation. 2025. arXiv: 2506.10868 [physics.ao-ph]. URL: https://arxiv.org/abs/2506.10868.

[31] S. Lang, M. Leutbecher, and S. Hatfield. On the sensitivity ofmachine-learned probabilistic weatherforecast models to scale-aware scoring rules. 2026. arXiv: 2607.19161 [physics.ao-ph]. URL: https://arxiv.org/abs/2607.19161.

[32] M. P. Mittermaier. “A Strategy for Verifying Near-Convection-Resolving Model Forecasts at Observing Sites”. In: Weather and Forecasting 29.2 (2014), pp. 185–204. DOI: 10.1175/WAF-D-12-00075.1.

[33] F. Pasquini, M. Baatsen, B. François, N. Theeuwes, and M. Schmeits. Assessing the ability ofa stretched-grid deep-learning weather prediction model to capture physical balances. 2026. arXiv: 2604.01454 [physics.ao-ph]. URL: https://arxiv.org/ abs/2604.01454.

[34] S. McIntosh-Smith, S. R. Alam, and C. Woods. Isambard-AI: a leadership class supercomputer optimised specifically for Artificial Intelligence. 2024. arXiv: 2410.11199 [cs.DC]. URL: https://arxiv.org/abs/2410.11199.

[35] P. Tillet, H. T. Kung, and D. Cox. “Triton: an intermediate language and compiler for tiled neural network computations”. In: Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages. MAPL 2019. Phoenix, AZ, USA: Association for Computing Machinery, 2019, pp. 10–19. ISBN: 9781450367196. DOI: 10.1145/3315508.3329973.

[36] F. Cotter and S. McLaughlin.fbcotter/pytorch\_wavelets: Zenodo Release. Version 1.1.0. Oct. 2019. DOI: 10.5281/zenodo.3471950. URL: https://doi.org/10.5281/ zenodo.3471950.

[37] F. Cotter. “Uses of Complex Wavelets in Deep Convolutional Neural Networks”. PhD thesis. Apollo - University of Cambridge Repository, 2019. DOI: 10.17863/CAM.53748. URL: https://www.repository.cam.ac.uk/handle/1810/306661.

[38] J. M. Wallace and P. V. Hobbs. Atmospheric Science: An Introductory Survey. 2nd ed. Academic Press, 2006.

[39] N. M. Roberts and H. W. Lean. “Scale-Selective Verification of Rainfall Accumulations from High-Resolution Forecasts of Convective Events”. In: Monthly Weather Review 136.1 (2008), pp. 78–97. DOI: 10.1175/2007MWR2123.1.

## A Spatial attribution and gradient comparison

MSE  
FFT-AMSE  
Wavelet  
![](images/0fe4b16a2403365e49e300c66f25f1825d570d078e3fafac70ee697661f675a2.jpg)  
Figure A.1: Comparison of first-order spatial attribution and field-space gradients for 2 m temperature using MSE, Fourier-domain AMSE (FFT-AMSE), and the wavelet-based loss. Rows show the two constructed prediction fields introduced in Figure 2: a perturbed 6 hour time-offset prediction with added Gaussian noise (Case 1), and a Gaussian-blurred target with $\sigma = 3$ grid cells (Case 2). For each objective, the left panel shows the attribution $A ( r )$ defined by Equation A.1, and the right panel shows the field-space gradient $\partial \mathcal { L } / \partial \widehat { x } ( \pmb { r } )$ . Attribution is displayed on a logarithmic scale. Common colour scales are used across objectives for each diagnostic. Attribution combines the local prediction error with the sensitivity of the objective and is therefore a first-order diagnostic rather than an exact spatial decomposition of the scalar objective. Absolute loss and gradient magnitudes depend on the definition and normalisation of each objective, so the comparison emphasises the spatial structure of the resulting fields.

Section 2.1 showed how the wavelet objective distributes its scalar loss and field-space training signal among decomposition levels. Here, we compare the spatial structure of the complete training signals produced by pointwise MSE, FFT-AMSE, and the wavelet-based loss for the same two constructed prediction fields.

For a scalar objective ${ \mathcal { L } } ,$ prediction xb, and target x, we define the first-order spatial attribution at location r as

$$
A ( \pmb { r } ) = \left| \left[ \widehat { \boldsymbol { x } } ( \pmb { r } ) - \boldsymbol { x } ( \pmb { r } ) \right] \frac { \partial \mathcal { L } } { \partial \widehat { \boldsymbol { x } } ( \pmb { r } ) } \right| .\tag{A.1}
$$

This diagnostic combines the magnitude of the local prediction error with the sensitivity of the complete objective at the same location. It represents the magnitude of the first-order change associated with moving that prediction value towards its target. For pointwise MSE, the attribution is proportional to the pointwise squared error. For transform-domain objectives such as FFT-AMSE and the wavelet-based loss, coefficients from multiple spatial locations are aggregated and coupled within the objective. The attribution is therefore not an exact additive decomposition of the finite scalar loss, but provides a common field-space diagnostic of where the different objectives are most sensitive.

Figure A.1 shows the attribution and field-space gradient for each objective and prediction regime. Attribution is shown logarithmically because it spans several orders of magnitude. The gradient is displayed separately because the two diagnostics answer different questions: the gradient shows the signed sensitivity of the objective to changes in the predicted field, whereas the attribution also weights that sensitivity by the prediction error already present. A location can therefore have a large gradient but modest attribution when its error is small, or large attribution when both error and sensitivity are appreciable.

For the perturbed time-offset prediction (Case 1), all three objectives produce broadly similar attribution patterns, highlighting many of the same broad and finer-scale errors. Their fieldspace gradients nevertheless differ. The MSE gradient follows the pointwise prediction error, consistent with the local definition of the objective. The FFT-AMSE gradient is more broadly distributed across the domain, consistent with the global support of the Fourier basis. The wavelet gradient retains stronger location-dependent variation around the atmospheric structures contributing to the error. This behaviour is consistent with the spatial indexing and compact support of the wavelet coefficients, despite the coupling introduced by aggregation within each scale–orientation band.

The contrast is stronger for the blurred prediction (Case 2). The MSE gradient follows the pointwise residual introduced by smoothing but does not distinguish the spatial scales contributing to that residual. FFT-AMSE responds to the missing spectral variance, but its gradient is distributed throughout much of the domain. The wavelet objective is similarly sensitive to the removed variability, while producing a gradient that is more strongly organised around the temperature features altered by smoothing. The response is concentrated primarily over land and is particularly structured over regions of complex orography, including the Scottish Highlands, with a weaker response over much of the surrounding sea. This organisation is consistent with fine-scale 2 m temperature variability associated with terrain and land–sea contrasts.

Together with the scale-wise decomposition in Figure 2, this comparison shows that the wavelet objective both identifies the affected range of spatial scales and retains location-dependent sensitivity to the corresponding atmospheric structures. Absolute loss and gradient magnitudes depend on the definition and normalisation of each objective and are not directly comparable; the comparison is therefore restricted to the spatial structure and relative localisation of the attribution and gradient fields.

## B Model configuration and training schedule

The model was developed and trained using the Anemoi framework [6], and follows an encode– process–decode architecture with graph-transformer GNNs used in all three components. Input fields are encoded onto a latent mesh, processed in latent space and decoded onto the output grids. Separate encoder–decoder pairs are used for two datasets: ERA5 on its native reduced Gaussian grid, representing the global atmospheric state, and Met Office UKV operational analyses on a regional grid covering the UK at approximately 1.5 km resolution. This configuration allows the datasets to use different grids and variables while sharing a common latent representation. The model therefore produces global and high-resolution regional forecasts simultaneously in a single forward pass.

The MSE-, FFT-AMSE- and wavelet-trained models use the staged training schedule summarised in Table B.1, following the approach introduced in Ref. [11]. Stage 1 establishes the global forecasting capability using ERA5 on the O96 grid and a coarse global latent mesh. In Stage 2, the resolution of the ERA5 dataset is increased to N320 and the global latent mesh is refined, allowing the model to learn a higher-resolution global representation over the same training period.

Stage 3a introduces the UKV dataset, together with a regional encoder–decoder pair and a highresolution regional mesh embedded within the global mesh. The shared latent representation retains the global atmospheric context while adapting the model to the UK domain. Global and regional losses are assigned relative weights of 67% and 33%, respectively. Without this weighting, the regional domain would contribute less than 1% of the combined objective because of its much smaller spatial extent.

Stage 3b extends training from single-step forecasts to autoregressive rollouts of 2–12 steps, exposing the model to its own predictions and improving stability at longer lead times. The reduced learning rate in this stage allows the model to be fine-tuned over longer forecast horizons while limiting degradation of the global and regional representations learned in the preceding stages.

Table B.1: Training configuration for each stage of the model used in this study. A dash indicates that no regional dataset or regional mesh was used in that stage. All training was performed on the Isambard-AI (Phase 2) HPC system [34] using NVIDIA GH200 superchips.
<table><tr><td rowspan="2"></td><td rowspan="2">Stage 1</td><td rowspan="2">Stage 2</td><td colspan="2">Stage 3</td></tr><tr><td>a</td><td>b</td></tr><tr><td>Global dataset</td><td>ERA5</td><td>ERA5</td><td>ERA5</td><td>ERA5</td></tr><tr><td>Global data grid</td><td>096</td><td>N320</td><td>N320</td><td>N320</td></tr><tr><td>Global resolution</td><td>100 km</td><td>30 km</td><td>30 km</td><td>30 km</td></tr><tr><td>Global mesh refinement</td><td>5</td><td>6</td><td>6</td><td>6</td></tr><tr><td>Global wavelet levels, J</td><td>4</td><td>6</td><td>6</td><td>6</td></tr><tr><td>Regional dataset</td><td>一</td><td>一</td><td>UKV</td><td>UKV</td></tr><tr><td>Regional resolution</td><td>一</td><td>一</td><td>1.5 km</td><td>1.5 km</td></tr><tr><td>Regional mesh refinement</td><td>一</td><td>一</td><td>11</td><td>11</td></tr><tr><td>Regional wavelet levels, J</td><td>一</td><td>一</td><td>6</td><td>6</td></tr><tr><td>Rollout</td><td>1</td><td>1</td><td>1</td><td>2-12</td></tr><tr><td>Training period</td><td>1980-2022</td><td>1980-2022</td><td>2020–2022</td><td>2020-2022</td></tr><tr><td>Iterations</td><td>100,000</td><td>50,000</td><td>25,000</td><td>3200</td></tr><tr><td>Peak learning rate</td><td> $2 \times 1 0 ^ { - 3 }$ </td><td> $3 . 2 \times 1 0 ^ { - 4 }$ </td><td> $4 \times 1 0 ^ { - 4 }$ </td><td> $2 . 5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Warm-up iterations</td><td>1000</td><td>100</td><td>500</td><td>50</td></tr><tr><td>Batch size</td><td>128</td><td>32</td><td>32</td><td>32</td></tr><tr><td>Total number of GPUs</td><td>16</td><td>32</td><td>64</td><td>64</td></tr><tr><td>GPUs per model instance</td><td>1</td><td>1</td><td>2</td><td>2</td></tr><tr><td>Total GPU hours</td><td>540</td><td>720</td><td>800</td><td>400</td></tr></table>

Note. Mesh refinement denotes the subdivision frequency of the icosahedron used to construct the latent mesh with larger values corresponding to finer meshes. Refinement frequencies of 6 and 11 give average node spacings of approximately 110 km and 4 km, respectively. Wavelet-transform settings apply only to the wavelet-trained experiment; all other configuration settings are shared by the three model variants.

The three models differ only in their training objectives. The MSE- and wavelet-loss trained models use their respective loss functions throughout all four stages and for both the global and regional components. The FFT-AMSE model used a fast Fourier Transform (FFT) variant of the AMSE loss introduced in Ref. [19] in the regional component, replacing the spherical harmonic transform with an FFT performed over the regional domain. Its global component uses MSE and is initialised from the MSE model at the end of Stage 2.

For the wavelet experiment, fields on reduced Gaussian grids are regridded to regular latitude– longitude grids before applying the two-dimensional DTCWT: O96 fields are mapped to a 1.0° grid and N320 fields to a 0.25° grid, following Ref. [8]. UKV fields are transformed directly on the native regional grid. The number of wavelet decomposition levels used for each dataset is given in Table B.1, and is calculated via

$$
J = \lfloor \log _ { 2 } \left[ \operatorname* { m i n } ( H , W ) \right] \rfloor - 3 ,\tag{B.1}
$$

where H and W are the meridional and zonal dimensions of the input grid, respectively.

We compute the dual-tree complex wavelet transform using a custom Triton [35] implementation written by the authors, validating its forward transform and gradients against pytorch\_wavelets [36, 37] to within numerical precision. The implementation uses the ‘near\_sym\_b’ and ‘qshift\_b’ filter sets provided by pytorch\_wavelets, corresponding to a first-level biorthogonal filter bank followed by quarter-shift filters at subsequent levels [24]. With the Triton implementation, wavelet training achieved approximately 20% lower throughput than the MSE baseline in Stages 1 and 2. No difference was observed in Stage 3, where throughput was limited by data loading rather than loss computation.

## C Verification methodology

Surface verification uses quality-controlled observations from 114 stations across the UK. Six-hour precipitation accumulations are derived separately from hourly Synoptic Rainfall Europe West observations, following standard operational verification procedures. The same observation datasets and processing are used to evaluate all model configurations.

Neighbourhood-based verification is performed using the High-Resolution Assessment (HiRA) framework [32]. For each observation, a square neighbourhood centred on the nearest model grid point is interpreted as a pseudo-ensemble, and the resulting distribution is evaluated using the continuous ranked probability score (CRPS). We use neighbourhoods of $3 \times 3 , 7 \times 7$ , and 11×11 model grid points. HiRA therefore provides some tolerance to small spatial displacement errors while still rewarding appropriate local values and variability.

For variables without suitable station observations, forecasts are verified against the corresponding UKV operational analyses. For each analysis field, 1,000 grid points are sampled at fixed, reproducibly selected locations within the regional domain and treated as pseudo-observations. At each location, the surrounding forecast grid points form the HiRA pseudo-ensemble, from which CRPS is calculated for neighbourhoods of $3 \times 3 , 7 \times 7$ , and $1 1 \times 1 1$ model grid points. These scores measure agreement with the gridded analysis rather than independent observational skill.

Pointwise RMSE is calculated using variable-specific collocation. Forecast 2 m temperature is bilinearly interpolated to each observation location, and observations are adjusted to the UKV model orography using a standard environmental lapse rate of −0.0065 K m<sup>−1</sup> [38]. The same elevation-adjusted observations are used for the corresponding HiRA calculation. Six-hour precipitation accumulations are compared with the nearest model grid point to avoid smoothing intermittent precipitation fields. The 10 m zonal and meridional wind components are bilinearly interpolated before calculating wind speed and vector error.

## D Additional verification

## D.1 Comparison with the MSE-trained model

Figure D.1 extends the comparison between the MSE- and wavelet-trained models across surface and upper-air variables, verification metrics, and lead times. For verification against surface observations, wavelet training generally improves HiRA CRPS, particularly for the wind diagnostics and at larger neighbourhood sizes, while the MSE-trained model generally retains a small advantage in pointwise RMSE. Differences in upper-air RMSE are also generally small and vary with variable and lead time.

The corresponding spectral differences are larger and more systematic. For the surface variables, the reduction in intermediate- and short-wavelength power deficits under wavelet training persists across lead times of 6, 24, and 72 hours (Figure D.2). The distinction is clearest for 10 m wind speed and accumulated precipitation, while the differences for 2 m temperature are smaller. This lead-time comparison supports the conclusions drawn from the 24-hour spectra in Section 4.2, showing that the improved agreement with the reference spectra is not confined to a single forecast lead time.

Similar behaviour is found for the upper-air variables (Figure D.3). The MSE-trained model develops intermediate-scale power deficits for 850 hPa temperature and 700 hPa specific humidity, both of which are substantially reduced by wavelet training. For 500 hPa geopotential, the MSE-trained spectra instead contain narrow regions of excess power at wavelengths associated with the internal processor mesh; these features are strongly suppressed in the wavelet-trained model.

Reference model: MSE Experimental model: WVLT Validation period: 2023-01-01T00:00Z to 2023-12-21T18:00Z (6 hr freq.)  
![](images/9a44822433cafd5f61a2357407a73b405af59336031cd3ede3e7fdef8d2ae6c3.jpg)  
Figure D.1: Scorecard showing the percentage difference in verification scores between the wavelet- and MSE-trained models over lead times of up to 10 days. “OBS” and “ANL” denote verification against observations and operational analyses, respectively. Surface variables are 10 m wind speed (10ws), 10 m wind vector error (10wv), the zonal and meridional components of 10 m wind (10u and 10v), 2 m temperature (2t), total precipitation (tp), surface pressure (sp) and skin temperature (skt). Upper-air variables are temperature (t), specific humidity (q), geopotential (z), and wind speed (ws), with the corresponding pressure levels given in hPa. Columns show pointwise root-mean-squared error (RMSE) and high-resolution assessment (HiRA) continuous ranked probability score (CRPS), with HiRA CRPS calculated using neighbourhoods of $3 \times 3 , 7 \times 7 .$ , and 11 × 11 model grid points. Scores are aggregated over forecasts initialised between 1 January and 21 December 2023. Blue indicates lower errors for the wavelet-trained model, whereas red indicates lower errors for the MSE-trained model.

Together, the additional verification shows that wavelet training produces forecasts that remain closer to the reference spectra across a broader range of resolved scales, variables, and lead times, while its effect on pointwise and neighbourhood-based scores depends more strongly on the variable and verification metric.

## D.2 Comparison with the FFT-AMSE-trained model

The wavelet-trained model generally outperforms the FFT-AMSE-trained model across variables, verification metrics, and lead times (Figure D.4). Improvements are particularly widespread for upper-air RMSE and are largest at short to medium lead times. Wavelet training also generally improves HiRA CRPS for surface variables, particularly at the larger neighbourhood sizes. Differences in surface RMSE are smaller and less systematic, although they predominantly favour the wavelet-trained model.

![](images/c4999072d7e9dff2c4e4fff8e681124640e004b96b694086d5c7ae12404753f5.jpg)

![](images/4158420bc84cef09d513b92cc5b4ace2ac7b8c6b9501672f6e9d79f452de1cc3.jpg)

![](images/c5c5e95f9780f7857ef8750a9ba3dfce6b81080f25faa0828f23ef84c70a8c50.jpg)  
Figure D.2: Lead-time dependence of the zonal power spectra for the MSE- (blue/green) and wavelet-trained (orange) models, verified against UKV operational analyses (black, dashed). Columns show (a) 2 m temperature, (b) 10 m wind speed, and (c) 6 hour accumulated precipitation. Upper panels show the absolute spectra at lead times of 6, 24, and 72 hours, with colour shade indicating lead time. Lower panels show $\log _ { 1 0 } ( P _ { f } / P _ { r } )$ , where $P _ { f }$ and $P _ { r }$ are the forecast and reference powers, respectively. Vertical dotted lines mark the internal processormesh spacing, $\Delta x _ { \mathrm { m e s h } } = 4$ km, and wavelengths corresponding to $2 \Delta x _ { \mathrm { m e s h } } , 4 \Delta x _ { \mathrm { m e s h } } .$ and $8 \Delta x _ { \mathrm { m e s h } }$ . The reduction in intermediate- and short-wavelength power deficits under wavelet training persists across the evaluated lead times.

![](images/7eb0cc3c55804939ecc725992aaca0aebdca87e5ef613d44dc832c4b2bb41a68.jpg)

![](images/fd0a5664aadfdfb97f3d122512b792311833b6c854e092d6b7571fa0ead14615.jpg)

![](images/4b0821ee37a54036f982d5de8175b28f5148cc3e6f31dfbea43a04c87e8464ee.jpg)  
Figure D.3: As Figure D.2, but for upper-air variables: (a) 850 hPa temperature, (b) 700 hPa specific humidity, and (c) 500 hPa geopotential.

Spectral fine-tuning commonly degrades pointwise performance relative to MSE because preserving fine-scale variability limits the smoothing that minimises pointwise error. The results presented here show that wavelet training substantially reduces this trade-off. The wavelet-trained model retains pointwise performance close to that of the MSE-trained model and performs considerably better than FFT-AMSE across most pointwise and neighbourhoodbased metrics. At the same time, the spectral comparisons presented in Section 4.2 show that the wavelet-trained forecasts remain closer to the reference spectra over a broader range of resolved scales than either baseline. MSE training produces the largest intermediate- and small-scale power deficits, while FFT-AMSE recovers some of this missing variance but remains less consistent with the reference than the wavelet-trained model. Wavelet training therefore reproduces the reference distribution of variance across spatial scales more accurately than FFT-AMSE, while maintaining pointwise performance much closer to that of MSE. Among the three training objectives, the wavelet loss consequently provides the most favourable combination of forecast accuracy and physically realistic multiscale structure.

Reference model: FFT-AMSE Experimental model: WVLT Validation period: 2023-01-01T00:00Z to 2023-12-21T18:00Z (6 hr freq.)  
![](images/96b857662e31c2ea079d2807eaf387d71ebf7b7163e4216a69fbae3eda8259cd.jpg)  
Figure D.4: Scorecard showing the percentage difference in verification scores between the wavelet- and FFT-AMSE-trained models over lead times of up to 10 days. “OBS” and “ANL” denote verification against observations and operational analyses, respectively. Columns show pointwise RMSE and HiRA CRPS calculated using neighbourhoods of $3 \times 3 , 7 \times 7 ,$ , and $1 1 \times 1 1$ model grid points. Scores are aggregated over forecasts initialised between 1 January and 21 December 2023. Blue indicates lower errors for the wavelet-trained model, whereas red indicates lower errors for the FFT-AMSE-trained model.

## D.3 Neighbourhood precipitation verification

The spectral and intensity-distribution diagnostics show that wavelet training produces sharper precipitation fields with greater fine-scale variability. Although HiRA CRPS assesses the continuous distribution of forecast values within a neighbourhood, it does not directly isolate performance for particular precipitation intensities. We therefore evaluate the Fractions Skill Score (FSS) [39] across a range of precipitation thresholds and neighbourhood sizes to determine how spatial skill varies with both event intensity and scale. FSS compares the fractions of forecast and observed grid points exceeding a given threshold within each neighbourhood. It ranges from zero to one, with one indicating perfect agreement, and provides a displacementtolerant measure of precipitation skill that avoids the double penalty associated with strict pointwise verification.

![](images/dbc89706f02a96684dfa8885372913d8d4fa8c563fbc2af7abc47dcdd5bc55f2.jpg)  
Figure D.5: Fractions Skill Score (FSS) for 6 hour accumulated precipitation, ‘tp’, at a lead time of 24 hours, evaluated across precipitation thresholds and neighbourhood sizes over the regional domain. (a) Absolute FSS for the wavelet-trained model. (b,c) FSS difference between the wavelet-trained model and the MSE- and FFT-AMSE-trained models, respectively; positive values (green) indicate improved performance for the wavelet-trained model. Physical neighbourhood scales, calculated from the regional grid spacing, are shown along the upper axis. The same difference colour scale is used in panels (b) and (c).

Figure D.5 shows the absolute FSS of the wavelet-trained model and its differences relative to the MSE- and FFT-AMSE-trained models at a lead time of 24 hours. Wavelet training improves FSS relative to MSE across almost all precipitation thresholds and neighbourhood sizes, with the largest gains occurring for heavier precipitation and broader neighbourhoods. Improvements over FFT-AMSE are smaller but remain consistently positive across most of the matrix and follow a similar dependence on threshold and neighbourhood size.

The improvement over FFT-AMSE indicates that the benefit of wavelet training is not explained solely by greater spectral amplitude. The spatial localisation of the wavelet objective also improves the neighbourhood-scale agreement of precipitation structures. Wavelet training therefore produces sharper and more intense precipitation features without sacrificing spatial skill, with particularly clear benefits under displacement-tolerant evaluation of heavier precipitation.