# MW-Nowcast: Six-hour ensemble nowcasting of extreme precipitation

Ning Wang<sup>1,2</sup>†, Zuliang Fang<sup>1</sup>†, Weixin Jin<sup>1</sup>†, Zhongjian Lv<sup>1</sup>, Shuang Qin<sup>1</sup>, Pengcheng Zhao<sup>1</sup>, Siqi Xiang<sup>1</sup>, Jiang Bian<sup>1</sup>, Haoyi Xiong<sup>1</sup>, Nan Guan<sup>2</sup>, Bin Zhang<sup>1</sup>, Liangjie Zhang<sup>1</sup>, Denvy Deng<sup>1</sup>, Qi Zhang<sup>1</sup>, Matt Corey<sup>1</sup>, Jitu Keshri<sup>1</sup>, Sridhar Iyer<sup>1</sup>, Hongyu Sun<sup>1</sup>, Kit Thambiratnam<sup>1</sup>, Jonathan Weyn<sup>1</sup>, Richard E. Turner<sup>3</sup>, Haiyu Dong<sup>1\*</sup>

<sup>1</sup>Microsoft Corporation. <sup>2</sup>City University of Hong Kong. <sup>3</sup>University of Cambridge.

\*Corresponding author(s). E-mail(s): haiyu.dong@microsoft.com; <sup>†</sup>These authors contributed equally to this work.

## Abstract

Extending reliable nowcasting of extreme precipitation could provide critical additional time for warnings and emergency response during high-impact events such as flash floods. Radar-based generative machine-learning models have enabled skilful hyperlocal precipitation nowcasting, but accurate prediction of intense precipitation remains confined to the first few hours. Because storm-scale structure is predictable for longer than individual cells, a natural strategy is to predict that structure while generatively modelling only the uncertain local growth, decay, reorganisation and initiation of storms. Here we present Microsoft Weather Nowcast (MW-Nowcast), a six-hour ensemble radar nowcasting model that jointly learns a deterministic predictor to capture organised precipitation structure shared across ensemble members, and a generator to produce diverse local residuals around this shared prediction. Across independent test data from the United States, Europe and China, MW-Nowcast achieves higher detection skill than leading methods for heavy and extreme precipitation throughout the 6 h horizon. For the most intense rainfall, MW-Nowcast doubles the available warning time across all three regions, delivering 6 h forecasts with skill previously limited to 3 h for the leading generative baseline. A cost–loss decision analysis shows that MW-Nowcast retains substantial value for a broad range of applications even at 4–6 h, where alternative methods ofer little benefit. These additional hours can give forecasters and emergency managers the time to warn and act before extreme rainfall strikes, helping to protect lives and property.

## 1 Introduction

Precipitation nowcasting aims to provide weather forecasts for the next few hours, translating available information into guidance that is local and early enough to support immediate proactive mitigation. The World Meteorological Organisation frames nowcasting as local weather prediction from the present to up to 6 h ahead [1]. Within this window, predicting hazardous precipitation reliably can broaden the range of possible responses, including issuing warnings, positioning emergency resources, and managing drainage systems, roads, aviation and energy operations. This need is especially acute for extreme precipitation, because assessing its potential impacts requires forecasts that extend far enough to characterize the likely duration of intense rainfall over afected locations [2].

Operational nowcasting requires forecasts to be updated frequently as new observations arrive and to resolve rapidly-evolving precipitation at convective scales. Numerical weather prediction provides physically-grounded forecasts and can assimilate recent radar observations. However, its computational cost makes updates at radar cadence challenging [3, 4]. These constraints have motivated methods that infer future precipitation directly from recent radar sequences. Advection-based statistical methods such as STEPS provide fast and interpretable ensemble nowcasts by combining the extrapolation of recent radar echoes with stochastic, scale-dependent evolution [5, 6]. However, their reliance on transported recent echoes limits their ability to represent local growth, decay, reorganisation and initiation, especially as lead time extends beyond 3 h.

Over the last decade, machine-learning methods have become the gold standard for radar nowcasting, retaining the speed and radar-native character of extrapolation while learning nonlinear precipitation evolution from data. Early machine-learning approaches based on deep learning generally formulated this task as supervised sequence prediction, training deterministic networks with grid-cell losses on observed future radar sequences [7–9]. They improved the representation of nonlinear spatiotemporal evolution beyond extrapolation, but forecasts often became smooth at longer lead times, with weakened intense cores and limited uncertainty information. Deep Generative Models of Radar (DGMR) marked an important step towards probabilistic nowcasting by learning a conditional generative model for entire future precipitation fields, producing sharper and more useful ensemble forecasts [10]. NowcastNet extended this direction by conditioning the generative network on an advectioninformed deterministic evolution network [11]. It was also strongly preferred by professional meteorologists in 3 h case evaluations.

Despite these advances, DGMR and NowcastNet reported results for forecast horizons of 90 min and 3 h, respectively, while the full 0–6 h nowcasting window remains less explored for extreme precipitation [10, 11]. At longer lead times, future precipitation evolution remains partly constrained by the observed radar history, whereas its conditional distribution becomes increasingly broad. Existing generative systems typically sample complete future radar sequences, even when a deterministic prediction is supplied as conditioning information. In this formulation, the generative component must simultaneously preserve the storm-scale structure supported by recent observations and represent uncertainty in local cell evolution. Coupling these roles may make it harder to maintain coherent storm placement and organisation while generating suficient ensemble variability at longer lead times. Inspired by scale-dependent precipitation predictability and stochastic representations of unresolved weather variability, a promising direction is to make the allocation between predictable dynamics and stochastic variability itself learnable [12–14].

We show that long-lead-time ensemble nowcasting benefits from an explicit, jointly-learned separation between a deterministic forecast shared across members and member-dependent probabilistic residuals. We developed Microsoft Weather Nowcast (MW-Nowcast), a radar nowcasting model that couples two decoders, one producing the deterministic prediction and the other modelling the probabilistic residual relative to it. The two are trained jointly rather than as a fixed two-stage cascade, so that the allocation between shared storm-scale structure and local evolution adapts throughout training. Across radar networks in the United States (US), Europe (EU) and China (CN), MW-Nowcast extends skilful detection of heavy and extreme precipitation from the previously demonstrated 3 h to the full 6 h operational window while preserving the multiscale structure of the observations. In two recent high-impact events, the storm that spawned an EF4 tornado in Mississippi and the July 2025 extreme rainfall over Beijing, MW-Nowcast kept the intense rain cores over the afected areas for longer than baselines, which weakened or displaced them. A cost–loss analysis shows that this advantage translates into value at 4–6 h lead times across the full range of protective decisions. Within these hours, authorities can issue warnings, manage drainage and transport systems, and position emergency resources before flash floods develop.

## 2 MW-Nowcast

MW-Nowcast is a 6 h ensemble radar nowcasting framework that predicts the future radar sequence $\mathbf { X } _ { 1 : T } = ( \mathbf { X } _ { 1 } , \ldots , \mathbf { X } _ { T } )$ from an observed radar history H = $( \mathbf { X } _ { - T _ { 0 } + 1 } , \ldots , \mathbf { X } _ { 0 } )$ , with $T _ { 0 } ~ = ~ 9$ and $T \ = \ 3 6$ at 10 min intervals, by combining a deterministic prediction with sampled probabilistic residuals. As shown in Fig. 1, the model first encodes the radar history with a shared encoder parameterised by ψ, then branches into a deterministic decoder and a flow-matching residual decoder. We denote the full-horizon mappings by $\mathcal { S } _ { \psi , \theta }$ for deterministic prediction and $\mathcal { R } _ { \psi , \phi }$ for residual sampling. Both mappings include the shared encoder; “deterministic decoder” and “residual decoder” refer only to the trainable modules after the encoder. Stochasticity is introduced by initialising the residual flow with a Gaussian noise tensor $\mathbf { Z } ^ { ( k ) }$ for each ensemble member. The k-th ensemble member can therefore be written as

$$
\mathbf { X } _ { 1 : T } ^ { ( k ) } = \mathbf { S } _ { 1 : T } + \mathbf { R } _ { 1 : T } ^ { ( k ) } ,\tag{1}
$$

where $\mathbf { S } _ { 1 : T } : = S _ { \psi , \theta } ( \mathbf { H } )$ is the full-horizon deterministic prediction, and ${ \bf R } _ { 1 : T } ^ { ( k ) } : = \qquad $ $\mathcal { R } _ { \psi , \phi } ( \mathbf { Z } ^ { ( k ) }$ , H) is the probabilistic residual generated for the k-th ensemble member. The noise tensor satisfies $\mathbf { Z } ^ { ( k ) } \in \mathbb { R } ^ { T \times S \times S }$ with $Z _ { t , i , j } ^ { ( k ) } \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , 1 )$ , independently across ensemble member $k ,$ , physical lead-time index t and spatial indices $( i , j )$ . The residual decoder is conditioned on the shared radar-history representation.

![](images/00c37c216d76102c3585ea2d9adb1ef9160cf890a8eb29c71db4ebc194db6c2d.jpg)  
Fig. 1 MW-Nowcast for 6 h ensemble precipitation nowcasting. A history encoder provides the deterministic and residual decoders with a shared representation of the observed radar sequence H. The deterministic decoder produces the common full-horizon forecast $\mathbf { S } _ { 1 : T } ,$ while the flow-matching residual decoder transforms Gaussian noise $\mathbf { Z } ^ { ( k ) }$ into sampled residuals through an ODE solver. At each lead time t, the sampled residual is added to $\mathbf { S } _ { t }$ to obtain $\mathbf { X } _ { t } ^ { ( k ) }$ ; diferent noise draws yield the forecast ensemble.

During training, the shared encoder, deterministic decoder and residual decoder are optimised as one model. The encoder is trained to provide a radar-history representation that supports both the deterministic decoder and the residual decoder. Through joint optimisation, the deterministic decoder and residual decoder co-adapt to allocate future radar evolution between a common predictable component and conditional residuals designed to represent plausible local variability over the 6 h horizon. In the Supplementary Information, we show that joint optimisation yields stronger long-lead-time high-threshold skill than training the deterministic decoder before the residual decoder, a cascade-style strategy analogous to residual correction in generative downscaling [15].

Training combines two objectives. The first penalises deviations between the deterministic prediction and the observed future sequence. The second is a probabilisticresidual flow-matching loss, with the target dynamically recomputed as

$$
\begin{array} { r } { \mathbf { R } _ { 1 : T } ^ { \star } = \mathbf { X } _ { 1 : T } - \mathbf { S } _ { 1 : T } , } \end{array}\tag{2}
$$

so the distribution of learned probabilistic residuals changes as the deterministic prediction is updated. The residual decoder is implemented with flow matching. Unlike GAN-based nowcasting, which encourages realism through a discriminator, flow matching trains the residual decoder to predict a velocity field at intermediate states along a probability path from Gaussian noise to the target residual distribution [16]. Recent difusion and flow-matching nowcasting models have improved sample diversity and uncertainty quantification compared to GAN-based systems [17–19].

During sampling, we integrate the learned probability-flow ODE to transform each independent noise tensor into a plausible residual sequence conditioned on the observed radar history. Adding this residual sequence to the deterministic prediction yields an ensemble member with detailed local structure, rather than an average over possible futures.

## 3 Experimental Setup

We train and evaluate MW-Nowcast on three regional radar datasets covering the US, EU and CN domains. All datasets are converted to a common 6 h nowcasting task defined above over a 256 km 256 km domain at 2 km resolution. The US dataset pairs NOAA/NCEI Storm Events reports with MRMS Seamless Hybrid Scan Reflectivity radar composites [20, 21]; the EU dataset pairs European Severe Weather Database records with OPERA radar mosaics [22, 23]; and the CN dataset is constructed from China Meteorological Administration composite reflectivity data using radar-native, peak-anchored strong-precipitation sampling. Training sets contain 130,000–175,000 training sequences per region. Each region is evaluated on 4,000 test sequences drawn from a strictly later observation period, selected under matching criteria to focus the benchmark on strong-precipitation and severe convective regimes (Methodology).

We compare MW-Nowcast against three representative baselines spanning operational extrapolation, deterministic deep prediction and probabilistic ensemble nowcasting. We use the STEPS implementation in PySTEPS as a widely used advection-based ensemble nowcasting baseline [5, 6]. SimVPv2 is a recent deterministic spatiotemporal prediction model that has served as a strong, state-of-the-art baseline across prediction benchmarks [24]. NowcastNet is a leading generative ensemble nowcasting model that showed strong performance for extreme precipitation in both US and CN evaluations [11]. We retrain SimVPv2 and NowcastNet using their oficial model implementations and evaluate all baselines under the same protocol.

Evaluation combines threshold-based, probabilistic, structural, and decisionanalytic diagnostics. Threshold-based verification uses rain-rate exceedances at 16, 32 and 64 mm h<sup>−1</sup> via the Critical Success Index within a $5 \times 5$ grid-cell neighbourhood (CSIN) [11, 25]. All ensemble methods are evaluated with four members and aggregated using Probability Matched Mean (PMM) wherever a single field is required [26].

Probabilistic forecasts are evaluated using the Continuous Ranked Probability Score (CRPS) and the Continuous Ranked Probability Skill Score (CRPSS), which measures how well the ensemble distribution agrees with the observations [27, 28]. Power Spectral Density (PSD) is used separately as a threshold-free diagnostic of spatial structure [29]. Practical decision utility is quantified using Relative Economic Value (REV) across cost–loss ratios [10, 30].

## 4 Precipitation Cases

We present two high-impact precipitation cases from contrasting storm regimes: a rapidly-evolving tornadic convective system in the United States and persistent mountainous heavy rainfall in China. We assess how well each method retains the precipitation system’s organisation and embedded intense cores as lead time increases, and how MW-Nowcast allocates the forecast between shared, predictable storm-scale structure and uncertain local evolution across ensemble members. Additional cases from all three domains are presented in Extended Data Figs. 1–9.

The convective storm over Walthall County, Mississippi, on 15 March 2025 (Fig. 2a) spawned an EF4 tornado. The event was documented by an NWS Storm Survey, with 3 fatalities, 8 injuries and approximately \$5.0 million in reported property damage. We initialise the forecast at 17:30 UTC. In the lead-time diagnostics (Fig. 2b), MW-Nowcast maintains higher CSIN than the baselines at 32 and 64 mm $\mathrm { h } ^ { - 1 }$ over much of the final three hours, with the separation clearest at 64 mm $\mathrm { h } ^ { - 1 }$ . It also better preserves multiscale spatial variability and achieves higher probabilistic skill than the ensemble baselines over most lead times.

The forecast fields show how MW-Nowcast overcomes the characteristic failure modes of existing baselines (Fig. 2c). Throughout the forecast window, a southwest– northeast–oriented organised precipitation band remains identifiable, while multiple embedded intense cores undergo local intensification, weakening, fragmentation and reorganisation. MW-Nowcast preserves rain-band continuity, concentrated heavy-rain cores and the overall spatial extent of the system at longer lead times, with memberwise scores confirming that the late-lead heavy-precipitation signal is carried by individual ensemble members (Fig. 2d). By contrast, PySTEPS advects part of the early motion, but its intensity fades progressively within the domain. SimVPv2 keeps the band continuous, but smooths the embedded convective cores into a broad, uniform region of heavy rain. NowcastNet produces sharper local structures at early and middle leads, but its later forecasts substantially underestimate heavy-core intensity while spreading light-to-moderate rain over an excessively broad area. These are precisely the modes that the design of MW-Nowcast targets. Its deterministic predictor learns the predictable storm-scale structure from data rather than transporting recent echoes, and its generator adds the small-scale detail as sampled residuals, which keeps the forecast sharp while remaining anchored to that structure rather than generating the entire field.

Comparing the deterministic prediction of MW-Nowcast with its ensemble members reveals this scale-dependent structure of forecast uncertainty (Fig. 2e). The deterministic branch captures the rain-band location, orientation and overall organisation shared across the ensemble, while the residual branch generates member-dependent variations in the locations, intensities and morphologies of the embedded convective cores. Rather than merely adding spatial detail to the deterministic forecast, the residual branch represents multiple possible pathways of local convective intensification, weakening and reorganisation while preserving the storm-scale evolution supported by the observed history.

d  
![](images/08d18bbce795443d64c9c4c816f34efe910b11f5095c534337dc452dc207bb73.jpg)  
Fig. 2 Case study of a US tornado event over the 6 h forecast horizon. The case shows a convective storm over Walthall County, Mississippi, on 15 March 2025 that spawned an EF4 tornado; the forecast is initialised at 17:30 UTC. a, Geographic and radar context, with the forecast domain outlined in orange. b, Forecast diagnostics: CSIN in a 5 × 5 grid-cell neighbourhood at rain-rate thresholds of 32 and 64 mm h<sup>−1</sup>, CRPS, CRPSS, and radially averaged PSD at lead times of 3 and 6 h. “Deterministic” denotes the output of the deterministic decoder of MW-Nowcast. c, Observed and predicted rain-rate fields at hourly lead times from 1 to 6 h. Rows show observations, PySTEPS, SimVPv2, NowcastNet and MW-Nowcast. d, CSIN and POD at thresholds of 32 and 64 mm h<sup>−1</sup> for the four MW-Nowcast ensemble members and their PMM summary. e, The deterministic prediction and four ensemble members at lead times of 2, 4 and 6 h.

MW-Nowcast likewise maintains its advantage in a contrasting convective predictability regime, where organised storm structure progressively scatters and deterministic predictability is rapidly lost (Fig. 3a). The late-July 2025 event over the northern Miyun district of Beijing and adjacent Hebei caused 30 deaths and led to the relocation of 80,332 people. For a forecast initialised at 11:00 UTC (19:00 local time)

on 26 July 2025, during the evening-to-night development of the storm, MW-Nowcast retains markedly higher skill than the baselines over the final three hours, especially at 64 mm $\mathrm { h } ^ { - 1 }$ , while achieving the highest probabilistic skill and closest spectral agreement with observations (Fig. 3b). In the forecast fields, the baselines exhibit the same error modes as in the previous case, and NowcastNet, although generative, likewise loses the intense cores at later lead times, whereas MW-Nowcast retains the sustained heavy-rainfall signal over Miyun to about 5 h (Fig. 3c).

The comparison between the deterministic prediction and the ensemble members shows how the allocation learned by MW-Nowcast shifts as predictability is lost (Fig. 3e). Over the first 4 h, the observed rainfall over the Miyun region is relatively concentrated, the deterministic prediction carries a coherent rainfall signal there, and the members largely agree on its location while difering in core detail. As the observed rainfall becomes increasingly scattered, the deterministic prediction fades to a faint patch by 6 h, whereas the members continue to produce intense cores in varied locations, often where the deterministic forecast shows almost nothing. This difers substantially from the Mississippi case, in which the deterministic branch carried most of the forecast throughout (Fig. 2e). The diagnostics quantify the shift: deterministic CSIN at 64 mm $\mathrm { h } ^ { - 1 }$ falls to zero within about 3 h, whereas the ensemble retains skill to about 5 h (Fig. 3b), with member-wise scores confirming that this signal is carried by individual ensemble members at middle lead times (Fig. 3d). By the final hour, with little storm-scale structure left to hold onto, the residual branch still supplies most of the heavy-rainfall signal as diverse scenarios. While the widening spread marks the limit of local predictability in this case, with individual members increasingly diverging from the observed cores towards 6 h, the ensemble still conveys a sustained likelihood of heavy rain over the region, an important signal that supports continued monitoring rather than de-escalation.

![](images/669c30066a1b41f72347d3af7f1686479e4c0018856eb1b2eb572404b5df1fa4.jpg)  
Fig. 3 Case study of the Beijing Miyun mountainous extreme-rainfall event over the 6 h forecast horizon. Extreme rainfall over the northern Miyun district of Beijing and adjacent Hebei during the late-July 2025 episode; forecast initialised at 11:00 UTC on 26 July 2025 (19:00 local time). Panel descriptions are as in Fig. 2.

## 5 Quantitative Evaluation

Figure 4 summarises verification over the complete US, EU and CN test sets. MW-Nowcast achieved the highest CSIN at every evaluated lead time and threshold across all three regions (Fig. 4a). Whereas baseline CSIN rapidly approached zero after approximately 3 h at a threshold of 64 mm $\mathrm { h } ^ { - 1 }$ , MW-Nowcast retained measurable skill throughout the 6 h horizon. NowcastNet was originally evaluated over a 3 h forecasting horizon, whereas we retrain it here to predict the full 6 h sequence. Using the 3 h CSIN of this retrained baseline as the reference, MW-Nowcast sustained equivalent skill for an additional 2.3 to more than 3 h across the nine region–threshold combinations, with larger gains for more extreme precipitation. These gains are supported by structured, member-dependent residual adjustments to the common prediction, illustrated by residual sampling and ensemble-spread visualisations (Supplementary Sections 3.1 and 3.2). Consistently, the PMM ensemble better recovers the observed high-intensity rain-rate tail than the deterministic prediction across all three regions (Extended Data Fig. 10). Both generating the complete future sequence without a deterministic branch and training the deterministic and residual models sequentially reduce long-lead high-threshold skill (Supplementary Section 2 and Supplementary Table 1). These results support jointly-learned residual generation as a key contributor to sustained extreme-precipitation skill at longer forecast horizons.

![](images/db0f68aea1a2cc7d43c14b53d39cb87499284d66970c6f4780a2c7145813a1d9.jpg)  
Fig. 4 Quantitative evaluation of 6 h precipitation nowcasts. Curves compare MW-Nowcast, NowcastNet, SimVPv2 and PySTEPS. a, Critical Success Index in a 5 × 5 grid-cell neighbourhood over the 6 h forecast horizon for the US, EU and CN test sets at rain-rate thresholds of 16, 32 and 64 mm $\mathrm { h } ^ { - 1 } \colon$ ; higher values are better. b, grid-point Continuous Ranked Probability Score (CRPS) of the four-member ensembles for the three regions; lower values are better. c, radially averaged power spectral density (PSD) over the US at lead times of 2, 4 and 6 h, with the observed spectrum shown for reference. d, relative economic value (REV) over the US as a function of cost–loss ratio at lead times of 2, 4 and 6 h for rain-rate exceedances of 16 and 32 mm $\mathrm { h } ^ { - 1 }$

We next evaluated the four-member ensembles directly using grid-point continuous ranked probability score (CRPS), which measures the agreement between the predictive distribution and the observation at each location (Fig. 4b). MW-Nowcast achieved lower CRPS than both NowcastNet and PySTEPS across all three regions, indicating a better balance between individual-member accuracy and ensemble spread. PySTEPS nevertheless outperformed NowcastNet in CRPS. Because domain-averaged grid-point CRPS weights every location equally and does not specifically emphasise intense precipitation, conservative forecasts can remain competitive even when they lose high-threshold skill.

We further assessed scale-dependent precipitation variability in all three regions using radially averaged PSD at lead times of 2, 4 and 6 h. Figure 4c shows the US results. SimVPv2 progressively lost spectral power at medium and short wavelengths, consistent with increasingly smooth forecasts. By contrast, MW-Nowcast remained close to the observed spectrum throughout the 6 h horizon, outperforming NowcastNet and more closely matching observations than PySTEPS at 2 and 4 h, with comparable agreement at 6 h. Similar spectral behaviour was observed in EU and CN (Supplementary Section 4.2). These results indicate better retention of precipitation variability across spatial scales.

Beyond standard verification metrics, we assessed whether improved forecast skill could better support protective decision-making using the relative economic value (REV) framework adopted in previous probabilistic forecasting studies [10, 31]. REV measures whether forecast-based action can reduce expected expense in a stylised cost–loss decision problem and is normalised such that 1 represents a perfect forecast and 0 represents no improvement over the optimal climatological strategy (Methodology). DGMR [10] evaluated REV for precipitation accumulated over a 90 min forecast window, whereas we evaluated the instantaneous value separately at lead times of 2, 4 and 6 h to examine whether forecasts retain decision value at longer lead times and could therefore support earlier protective action. At each lead time, we calculated REV across cost–loss ratios for rain-rate exceedances of 16 and 32 mm $\mathrm { h } ^ { - 1 }$ in all three regions. Across these regions, MW-Nowcast provided substantially greater REV overall than the baselines at both thresholds and all three lead times (Fig. 4d and Supplementary Information). The US results in Fig. 4d show that, at 4 and 6 h, MW-Nowcast retained positive value over a broad range of cost–loss ratios, whereas the baseline values were close to zero. These results suggest that the long-lead-time skill of MW-Nowcast can support useful protective decisions across users with different action costs, extending the time available for preparation throughout the 6 h nowcasting window.

## 6 Conclusion

The value of extreme-precipitation nowcasting depends not only on producing realistic precipitation fields, but also on retaining useful skill far enough ahead to support early warning and emergency response, yet the latter has remained a dificult frontier. We have presented MW-Nowcast, which jointly learns a shared deterministic prediction and conditional probabilistic residuals around it to provide six-hour ensemble radar nowcasting. Controlled experiments show that learning this division end-to-end is essential to sustaining skill at longer lead times. Across radar datasets from the United States, Europe and China, MW-Nowcast delivers skilful ensemble nowcasts of extreme precipitation across the full 6 h window, doubles the available warning time for the most intense rainfall from three hours to six, and retains decision value at 4–6 h, where existing methods ofer little.

Future work should extend precipitation nowcasting beyond the information available from radar alone. Combining radar with geostationary satellite imagery and numerical-weather-prediction guidance could provide earlier information about cloud development and environmental conditions, with particular potential for improving forecasts of convective initiation and rapidly evolving storms. Our evaluation spans three regions with dense radar networks, but radar observations could also serve as training targets for models driven by satellite and numerical inputs which are available more widely, bringing nowcasting to regions with sparse or no operational radar coverage. Such approaches could improve both the accuracy and geographical reach of long-lead precipitation nowcasting and extend the decision value demonstrated here to communities with the most limited early-warning capabilities.

## References

[1] Wang, Y., et al.: Guidelines for Nowcasting Techniques. World Meteorological Organization, Geneva (2017)

[2] Georgakakos, K.P., Modrick, T.M., Shamir, E., Campbell, R., Cheng, Z., Jubach, R., Sperfslage, J.A., Spencer, C.R., Banks, R.: The flash flood guidance system implementation worldwide: A successful multidecadal research-to-operations efort. Bulletin of the American Meteorological Society 103(3), 665–679 (2022) https://doi.org/10.1175/BAMS-D-20-0241.1

[3] Sun, J.: Convective-scale assimilation of radar data: progress and challenges. Quarterly Journal of the Royal Meteorological Society: A journal of the atmospheric sciences, applied meteorology and physical oceanography 131(613), 3439–3463 (2005)

[4] Sun, J., et al.: Use of nwp for nowcasting convective precipitation: recent progress and challenges. Bulletin of the American Meteorological Society 95(3), 409–426 (2014) https://doi.org/10.1175/BAMS-D-11-00263.1

[5] Bowler, N.E., Pierce, C.E., Seed, A.W.: Steps: A probabilistic precipitation

forecasting scheme which merges an extrapolation nowcast with downscaled nwp. Quarterly Journal of the Royal Meteorological Society: A journal of the atmospheric sciences, applied meteorology and physical oceanography 132(620), 2127–2155 (2006)

[6] Pulkkinen, S., Nerini, D., P´erez Hortal, A.A., Velasco-Forero, C., Seed, A., Germann, U., Foresti, L.: Pysteps: An open-source python library for probabilistic precipitation nowcasting (v1. 0). Geoscientific Model Development 12(10), 4185–4219 (2019)

[7] Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W.-K., Woo, W.-c.: Convolutional lstm network: A machine learning approach for precipitation nowcasting. In: Advances in Neural Information Processing Systems, vol. 28 (2015)

[8] Wang, Y., Long, M., Wang, J., Gao, Z., Philip, S.Y.: Predrnn: Recurrent neural networks for predictive learning using spatiotemporal lstms. In: Advances in Neural Information Processing Systems, vol. 30 (2017)

[9] Gao, Z., Shi, X., Wang, H., Zhu, Y., Wang, Y., Li, M., Yeung, D.-Y.: Earthformer: Exploring space-time transformers for earth system forecasting. In: Advances in Neural Information Processing Systems (2022). https://arxiv.org/abs/2207.05833

[10] Ravuri, S., Lenc, K., Willson, M., Kangin, D., Lam, R., Mirowski, P., Fitzsimons, M., Athanassiadou, M., Kashem, S., Madge, S., et al.: Skilful precipitation nowcasting using deep generative models of radar. Nature 597(7878), 672–677 (2021)

[11] Zhang, Y., Long, M., Chen, K., Xing, L., Jin, R., Jordan, M.I., Wang, J.: Skilful nowcasting of extreme precipitation with nowcastnet. Nature 619(7970), 526–532 (2023)

[12] Germann, U., Zawadzki, I.: Scale-dependence of the predictability of precipitation from continental radar images. part i: description of the methodology. Monthly Weather Review 130, 2859–2873 (2002)

[13] Lorenz, E.N.: The predictability of a flow which possesses many scales of motion. Tellus 21(3), 289–307 (1969) https://doi.org/10.1111/j.2153-3490.1969.tb00444. x

[14] Palmer, T.N.: Predicting uncertainty in forecasts of weather and climate. Reports on Progress in Physics 63(2), 71 (2000) https://doi.org/10.1088/0034-4885/63/ 2/201

[15] Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.-Y., Liu, C.-C., Vahdat, A., Nabian, M.A., Ge, T., Subramaniam, A., Kashinath, K., Kautz, J., Pritchard, M.: Residual Corrective Difusion Modeling for Km-scale Atmospheric Downscaling (2023). https://doi.org/10.48550/arXiv.2309.15214 . https://arxiv.

[16] Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. In: 11th International Conference on Learning Representations, ICLR 2023 (2023)

[17] Gao, Z., Shi, X., Han, B., Wang, H., Jin, X., Maddix, D., Zhu, Y., Li, M., Wang, Y.: Predif: Precipitation nowcasting with latent difusion models. In: Advances in Neural Information Processing Systems (2023). https://arxiv.org/abs/2307.10422

[18] Leinonen, J., Hamann, U., Nerini, D., Germann, U., Franch, G.: Latent difusion models for generative precipitation nowcasting with accurate uncertainty quantification. arXiv preprint arXiv:2304.12891 (2023)

[19] Ribeiro, B.P., Pucer, J.F.: Flowcast: Advancing precipitation nowcasting with conditional flow matching. In: International Conference on Learning Representations (2026). https://doi.org/10.48550/arXiv.2511.09731 . https://arxiv.org/abs/2511.09731

[20] NOAA National Centers for Environmental Information: Storm Events Database. Accessed March 18, 2026 (2026). https://www.ncei.noaa.gov/stormevents/

[21] Zhang, J., Howard, K., Langston, C., Kaney, B., Qi, Y., Tang, L., Grams, H., Wang, Y., Cocks, S., Martinaitis, S., Arthur, A., Cooper, K., Brogden, J., Kitzmiller, D.: Multi-radar multi-sensor (mrms) quantitative precipitation estimation: Initial operating capabilities. Bulletin of the American Meteorological Society 97(4), 621–637 (2016) https://doi.org/10.1175/BAMS-D-14-00174.1

[22] Dotzek, N., Groenemeijer, P., Feuerstein, B., Holzer, A.M.: Overview of ESSL’s severe convective storms research using the European Severe Weather Database ESWD. Atmospheric Research 93(1–3), 575–586 (2009) https://doi.org/10.1016/ j.atmosres.2008.10.020

[23] K¨ock, K., Leltne, T., Randeu, W., Divjak, M., Schrelber, K.-J.: Opera: Operational programme for the exchange of weather radar information. first results and outlook for the future. Physics and Chemistry of the Earth, Part B: Hydrology, Oceans and Atmosphere 25(10-12), 1147–1151 (2000)

[24] Tan, C., Gao, Z., Li, S., Li, S.Z.: Simvpv2: Towards simple yet powerful spatiotemporal predictive learning. IEEE Transactions on Multimedia (2025)

[25] Schaefer, J.T.: The critical success index as an indicator of warning skill. Weather and forecasting 5(4), 570–575 (1990)

[26] Ebert, E.E.: Ability of a poor man’s ensemble to predict the probability and distribution of precipitation. Monthly Weather Review 129(10), 2461–2480 (2001) https://doi.org/10.1175/1520-0493(2001)129 2461:AOAPMS 2.0.CO;2

[27] Gneiting, T., Raftery, A.E.: Strictly proper scoring rules, prediction, and estimation. Journal of the American statistical Association 102(477), 359–378 (2007)

[28] M¨uller, W., Appenzeller, C., Doblas-Reyes, F., Liniger, M.: A debiased ranked probability skill score to evaluate probabilistic ensemble forecasts with small ensemble sizes. Journal of Climate 18(10), 1513–1523 (2005)

[29] Harris, D., Foufoula-Georgiou, E., Droegemeier, K.K., Levit, J.J.: Multiscale statistical properties of a high-resolution precipitation forecast. Journal of Hydrometeorology 2(4), 406–418 (2001)

[30] Richardson, D.S.: Skill and relative economic value of the ECMWF ensemble prediction system. Quarterly Journal of the Royal Meteorological Society 126(563), 649–667 (2000) https://doi.org/10.1002/qj.49712656313

[31] Price, I., Sanchez-Gonzalez, A., Alet, F., Andersson, T.R., El-Kadi, A., Masters, D., Ewalds, T., Stott, J., Mohamed, S., Battaglia, P., Lam, R., Willson, M.: Probabilistic weather forecasting with machine learning. Nature 637(8044), 84–90 (2025) https://doi.org/10.1038/s41586-024-08252-9

[32] Fulton, R.A., Breidenbach, J.P., Seo, D.-J., Miller, D.A., O’Bannon, T.: The WSR-88D rainfall algorithm. Weather and Forecasting 13(2), 377–395 (1998) https://doi.org/10.1175/1520-0434(1998)013 0377:TWRA 2.0.CO;2

[33] Ruzanski, E., Chandrasekar, V., Wang, Y.: The casa nowcasting system. Journal of Atmospheric and Oceanic Technology 28(5), 640–655 (2011)

[34] Sun, H., Yang, Y., Han, W., Huang, W., Chen, H., Gao, Z., Li, Z., Huo, Z., Niu, Z.: Stormdit: A generative ai model bridges the 2–6 hour gray zone in precipitation nowcasting. arXiv preprint arXiv:2601.20342 (2026)

[35] Gong, J., Bai, L., Ye, P., Xu, W., Liu, N., Dai, J., Yang, X., Ouyang, W.: Cascast: Skillful high-resolution precipitation nowcasting via cascaded modelling. In: Proceedings of the International Conference on Machine Learning (2024). https://arxiv.org/abs/2402.04290

[36] Yu, D., Li, X., Ye, Y., Zhang, B., Luo, C., Dai, K., Wang, R., Chen, X.: Difcast: A unified framework via residual difusion for precipitation nowcasting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024). https://arxiv.org/abs/2312.06734

[37] Roberts, N.M., Lean, H.W.: Scale-selective verification of rainfall accumulations from high-resolution forecasts of convective events. Monthly Weather Review 136(1), 78–97 (2008) https://doi.org/10.1175/2007MWR2123.1

## 7 Methodology

We use this section to define the architecture, training objective and sampling procedure with the same notation as the main text. A temporal stack $\mathbf { X } _ { ( a , b ] }$ denotes the left-open, right-closed sequence $( \mathbf { X } _ { a + 1 } , \ldots , \mathbf { X } _ { b } )$ . The observed history is ${ \textbf { H } } =$ $( \mathbf { X } _ { - T _ { 0 } + 1 } , \ldots , \mathbf { X } _ { 0 } )$ , equivalently $\mathbf { H } = \mathbf { X } _ { ( - T _ { 0 } , 0 ] } ,$ , and therefore contains exactly $T _ { 0 } = 9$ frames, including the issue-time frame $\mathbf { X } _ { 0 } .$ The future target sequence is ${ \bf X } _ { 1 : T } \ =$ $( \mathbf { X } _ { 1 } , \ldots , \mathbf { X } _ { T } )$ , equivalently $\mathbf { X } _ { ( 0 , T ] }$ , with $T ~ = ~ 3 6$ . The parameterised full-horizon mappings for deterministic prediction and residual sampling are $S _ { \psi , \theta }$ and ${ \mathcal { R } } _ { \psi , \phi } ,$ , respectively. Both mappings include the shared encoder, and neither symbol denotes a decoder module alone. Their outputs are $\mathbf { S } _ { 1 : T } : = S _ { \psi , \theta } ( \mathbf { H } )$ and $\mathbf { R } _ { 1 : T } ^ { ( k ) } : = \mathcal { R } _ { \psi , \phi } ( \mathbf { Z } ^ { ( k ) } , \mathbf { H } )$ Throughout, calligraphic symbols denote mappings, whereas bold symbols denote tensor-valued inputs and outputs. Thus, Eq. (1) is the single top-level definition of the model used in both the main text and Methods. For physical lead time $t ,$ we write $\mathbf { X } _ { t } ^ { ( k ) } : = [ \mathbf { X } _ { 1 : T } ^ { ( k ) } ] _ { t } , \mathbf { S } _ { t } : = [ \mathbf { S } _ { 1 : T } ] _ { t }$ and $\mathbf { R } _ { t } ^ { ( k ) } : = [ \mathbf { R } _ { 1 : T } ^ { ( k ) } ] { t }$ <sub>t</sub>. We omit the member index k when referring to a generic probabilistic residual.

## 7.1 Deterministic-probabilistic formulation

For the discrete forecasting task, $\mathbf { X } _ { t } \in \mathbb { R } ^ { S \times S }$ denotes the reflectivity field at physical lead index $t ,$ and bold uppercase symbols such as $\mathbf { X } _ { ( a , b ] }$ denote sequences stacked along the time dimension. The deterministic prediction mapping $\boldsymbol { S } _ { \boldsymbol { \psi } , \boldsymbol { \theta } }$ produces the full T-frame output $\mathbf { S } _ { 1 : T }$ in parallel from H, avoiding autoregressive error accumulation. During training, this deterministic prediction defines the probabilistic residual target in Eq. (2). This target is not an observed physical forcing; it is precisely the part of the future sequence not represented by the current deterministic prediction. Because the model components underlying $\mathcal { S } _ { \psi , \theta }$ and $\mathcal { R } _ { \psi , \phi }$ are trained jointly, the probabilistic residual target evolves as the deterministic decoder improves.

The residual sampling map $\mathcal { R } _ { \psi , \phi }$ is defined by a residual decoder trained with conditional flow matching and an ODE solver used at inference. Physical lead time $t \in \{ 1 , \ldots , T \}$ indexes forecast frames, whereas generative time $\tau \in [ 0 , 1 ]$ indexes an internal trajectory from a Gaussian noise tensor to a probabilistic residual. To keep Eqs. (3)–(6) compact, we write $\mathbf { r } ^ { \star } : = \mathbf { R } _ { 1 : T } ^ { \star }$ for the full-sequence probabilistic residual target and $\mathbf { r } ( \tau )$ for the full sequence-valued flow state; the physical lead dimension is therefore implicit in both symbols. For a noise tensor Z drawn as specified in Eq. (1) and $\tau \sim \mathcal { U } [ 0 , 1 ]$ , the linear probability path is

$$
\mathbf { r } ( \tau ) = ( 1 - \tau ) \mathbf { Z } + \tau \mathbf { r } ^ { \star } .\tag{3}
$$

Its target velocity is

$$
\mathbf { u } ^ { \star } = \frac { d \mathbf { r } ( \tau ) } { d \tau } = \mathbf { r } ^ { \star } - \mathbf { Z } .\tag{4}
$$

The residual decoder, parameterised by $\phi$ and conditioned on the history representation produced by the shared encoder parameterised by $\psi _ { : }$ predicts this velocity from

the current flow state, generative time and observed history:

$$
\mathbf { v } _ { \psi , \phi } ( \mathbf { r } ( \tau ) , \tau , \mathbf { H } ) \approx \mathbf { u } ^ { \star } .\tag{5}
$$

At inference, a probabilistic residual is obtained by solving

$$
\frac { d \mathbf { r } ( \tau ) } { d \tau } = \mathbf { v } _ { \psi , \phi } ( \mathbf { r } ( \tau ) , \tau , \mathbf { H } ) , \qquad \mathbf { r } ( 0 ) = \mathbf { Z } , \qquad \mathbf { R } _ { 1 : T } : = \mathcal { R } _ { \psi , \phi } ( \mathbf { Z } , \mathbf { H } ) = \mathbf { r } ( 1 ) .\tag{6}
$$

This last identity connects the terminal ODE state directly to the probabilistic residual in Eq. (1). For numerical inference, we discretise generative time as $\tau _ { n } = n / N , n =$ $0 , \ldots , N$ , with $N = 2 0$ and step size $\Delta \tau = 1 / N$ , and write $\mathbf { r } _ { n } : = \mathbf { r } ( \tau _ { n } )$ . At a discrete solver state, the residual decoder predicts $\mathbf { v } _ { n } : = \mathbf { v } _ { \psi , \phi } ( \mathbf { r } _ { n } , \tau _ { n } , \mathbf { H } )$ . The fixed-step Heun integrator advances the flow by

$$
\tilde { \mathbf { r } } _ { n + 1 } = \mathbf { r } _ { n } + \Delta \tau \mathbf { v } _ { n } , \qquad \mathbf { v } _ { n + 1 } ^ { \mathrm { p } } = \mathbf { v } _ { \psi , \phi } ( \tilde { \mathbf { r } } _ { n + 1 } , \tau _ { n + 1 } , \mathbf { H } ) , \qquad \mathbf { r } _ { n + 1 } = \mathbf { r } _ { n } + { \frac { \Delta \tau } { 2 } } \bigl ( \mathbf { v } _ { n } + \mathbf { v } _ { n + 1 } ^ { \mathrm { p } } \bigr ) .\tag{7}
$$

For ensemble member k, $\mathbf { r } _ { 0 } ^ { ( k ) } \ = \ \mathbf { Z } ^ { ( k ) }$ and $\mathbf { R } _ { 1 : T } ^ { ( k ) } ~ = ~ \mathbf { r } _ { N } ^ { ( k ) }$ , while the deterministic prediction remains shared across members.

## 7.2 Model details

For all experiments, each sample covers a 256 km 256 km physical domain, while the model input is represented on a square radar grid with side length $S = 1 2 8$ . The input history has shape $( T _ { 0 } , S , S , 1 ) = ( 9 , 1 2 8 , 1 2 8 , 1 )$ and the target future sequence has shape $( T , S , S , 1 ) = ( 3 6 , 1 2 8 , 1 2 8 , 1 )$ , corresponding to 1.5 h of history and 6 h of prediction at 10 min resolution. A stride-4 patch projection gives feature-grid side length $s = S / 4 = 3 2$ , and the token feature dimension is $c = 2 5 6$ . The Gaussian noise tensor Z has mathematical shape $( T , S , S ) = ( 3 6 , 1 2 8 , 1 2 8 )$ and implementation shape (36, 128, 128, 1). Its 36 temporal slices are sampled independently.

MW-Nowcast contains three trainable modules: a shared history encoder $\mathcal { E } _ { \psi } , \mathrm { ~ a ~ }$ deterministic decoder parameterised by θ, and a residual decoder parameterised by $\phi .$ The latter two are the post-encoder modules, whereas $\mathcal { S } _ { \psi , \theta }$ and $\mathcal { R } _ { \psi , \phi }$ remain reserved for the corresponding end-to-end mappings. The encoder supplies the flattened history context $\mathbf { C } _ { \mathrm { { h i s t } } }$ to both decoders. The deterministic decoder produces $\mathbf { S } _ { 1 : T }$ , while each residual-decoder evaluation maps $( \mathbf { r } _ { n } , \tau _ { n } , \mathbf { C } _ { \mathrm { h i s t } } )$ to the probability-flow velocity $\mathbf { v } _ { n } .$ Table 1 summarises the symbols, tensor shapes and repeated-module counts used in Fig. 5.

The history encoder patchifies each observed frame with a strided $4 \times 4$ convolution, reducing the spatial grid from $S \times S$ to $s \times s$ and projecting the single input channel to width $c \ : = \ : 2 5 6$ . The flattened history-token sequence has shape $( T _ { 0 } , s ^ { 2 } , c ) = ( 9 , 1 0 2 4 , 2 5 6 )$ . The encoder applies $M _ { E } = 3$ stacked Temporal–Spatial– Frequency (TSF) blocks. Each block uses temporal attention, spatial mixing, AFNO frequency mixing and a second temporal-attention layer. Temporal attention uses 4 heads at width 256; the spatial modules alternate shifted-window and global-token attention, and AFNO uses $g = 8$ channel groups. All attention and MLP sublayers use pre-normalisation, SwiGLU activations and residual connections. The output is $\mathbf { C } _ { \mathrm { h i s t } } \doteq \mathbb { R } ^ { T _ { 0 } \times s ^ { 2 } \times c }$ .

Table 1 Architecture and discrete-inference notation.
<table><tr><td>Symbol</td><td>Meaning</td><td>Value</td></tr><tr><td> $T _ { 0 } , T$ </td><td>Numbers of history and future frames</td><td>9,36</td></tr><tr><td> $K$ </td><td>Number of ensemble members</td><td>4</td></tr><tr><td> $N$ </td><td>Number of Heun intervals</td><td>20</td></tr><tr><td> $S , s , c$ </td><td>Radar-grid side, feature-grid side and feature width</td><td>128, 32, 256</td></tr><tr><td> $g$ </td><td>Number of AFNO channel groups</td><td>8</td></tr><tr><td> $M _ { E }$ </td><td>Number of repeated TSF blocks in each context backbone 3</td><td></td></tr><tr><td> $M _ { T }$ </td><td>Number of temporal-upsample stages in the deterministic 2 decoder</td><td></td></tr><tr><td>H</td><td>Observed radar history</td><td>(9, 128, 128, 1)</td></tr><tr><td> $\mathbf { C } _ { \mathrm { h i s t } }$ </td><td>Flattened history-context tokens</td><td>(9, 1024, 256)</td></tr><tr><td> $\mathbf { S } _ { 1 : T }$ </td><td>Deterministic prediction</td><td>(36, 128, 128, 1)</td></tr><tr><td> $\mathbf { Z } ^ { ( k ) }$ </td><td>Initial noise for ensemble member k</td><td>(36, 128, 128, 1)</td></tr><tr><td> $\tau _ { n }$ </td><td>Discrete generative time at solver state n</td><td>0, 0.05, . . . , 1</td></tr><tr><td> $\mathbf { r } _ { n }$ </td><td>Probabilistic-residual flow state at  $\tau _ { n }$ </td><td>(36, 128, 128, 1)</td></tr><tr><td> ${ \bf v } _ { n }$ </td><td>Velocity predicted from  $\mathbf { r } _ { n }$  at  $\tau _ { n }$ </td><td>(36, 128, 128, 1)</td></tr><tr><td> $\mathbf { R } _ { 1 : T } ^ { ( k ) }$ </td><td>Terminal probabilistic residual for ensemble member k</td><td>(36, 128, 128, 1)</td></tr></table>

The deterministic decoder predicts all $T = 3 6$ future frames in parallel. Four 1D temporal convolution blocks with kernel size 3 and dilation rates 1, 2, 4, 8 aggregate the history tokens at width c. The temporal decoder uses $M _ { T } = 2$ transposed 1D convolution stages to expand the temporal axis from $T _ { 0 } = 9$ to $T = 3 6$ . Two PixelShufle stages then increase the feature grid from $s = 3 2$ to $S = 1 2 8$ , followed by the convolutional output head. The result is $\mathbf { S } _ { 1 : T } \in \mathbb { R } ^ { T \times S \times S \times 1 }$

At solver state $n ,$ the residual decoder receives $\mathbf { r } _ { n } \in \mathbb { R } ^ { T \times S \times S \times 1 } , \ \tau _ { n }$ and $\mathbf { C } _ { \mathrm { { h i s t } } }$ A stride-4 projection produces $( T , s ^ { 2 } , c )$ future tokens, and the sinusoidal embedding of $\tau _ { n }$ is projected to width c. The residual pathway applies $M _ { E } ~ = ~ 3$ TSF blocks with cross-attention to the history context. Four dilated temporal-convolution blocks then mix information across the $T ~ = ~ 3 6$ future positions without changing their number. Because the residual decoder has equal input and output lengths, its temporal-upsample operator is the identity; the same two-stage spatial upsampler reconstructs the radar grid from $s \ = \ 3 2$ to $S \ = \ 1 2 8$ . The decoder output is $\mathbf { \bar { v } } _ { n } : = \mathbf { v } _ { \psi , \phi } ( \mathbf { r } _ { n } , \tau _ { n } , \mathbf { H } ) \in \mathbb { R } ^ { T \times { \bar { S } } \times { S } \times 1 }$

The shared encoder, deterministic decoder and residual decoder contain approximately 18.6M, 11.2M and 23.4M trainable parameters, respectively. Diversity is introduced exclusively through the frame-wise independent noise tensor Z in the residual decoder. Consequently, ensemble spread represents conditional uncertainty in local growth, decay and structural reorganisation around a shared deterministic prediction, rather than stochasticity injected into the deterministic decoder.

![](images/226621242349ae45d94b5d416dc973beda6bc40881341c3483347a8da89c72c7.jpg)  
Fig. 5 Network architecture of MW-Nowcast. The history encoder maps H to $\mathbf { C } _ { \mathrm { h i s t } }$ , and the deterministic decoder produces $\mathbf { S } _ { 1 : T }$ . At discrete ODE state $( \mathbf { r } _ { n } , \tau _ { n } )$ , the residual decoder uses the shared history context to predict $\mathbf { v } _ { n } .$ . In panel c, $T _ { \mathrm { i n } } = T _ { 0 }$ for the deterministic decoder and $T _ { \mathrm { i n } } = T$ for the residual decoder. The dashed temporal-upsample block is applied only in the deterministic decoder and is bypassed as an identity operation in the residual decoder. The Heun solver iterates from $\mathbf { r } _ { 0 } ^ { ( k ) } = \mathbf { Z } ^ { ( k ) } \mathrm { \ddot { t o } } \mathbf { R } _ { 1 : T } ^ { ( k ) } = \mathbf { r } _ { N } ^ { ( k ) }$ . Symbols, shapes and repeated-module counts are defined in Table 1.

## 7.3 Training objective function

The three modules are optimised jointly. The deterministic decoder is supervised by an intensity-weighted reconstruction loss, while the residual decoder is supervised by the conditional flow-matching objective. The total loss is

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { \mathrm { d e t } } \mathcal { L } _ { \mathrm { d e t } } + \mathcal { L } _ { \mathrm { F M } } , } \end{array}\tag{8}
$$

where $\lambda _ { \mathrm { d e t } } = 0 . 2$ in all experiments.

## 7.3.1 Deterministic loss

Let $\begin{array} { r } { \mathbf { S } _ { t } ~ = ~ [ \mathbf { S } _ { 1 : T } ] _ { t } } \end{array}$ <sub>t</sub> be the deterministic prediction at physical lead time t. The deterministic loss is

$$
\mathcal { L } _ { \mathrm { d e t } } = \frac { s _ { \mathrm { d e t } } } { T } \sum _ { t = 1 } ^ { T } \ell _ { \mathrm { d e t } } ( \mathbf { S } _ { t } , \mathbf { X } _ { t } ) ,\tag{9}
$$

where $s _ { \mathrm { d e t } } = 0 . 5$ . The per-frame loss combines weighted MSE and MAE terms:

$$
\ell _ { \mathrm { d e t } } ( \mathbf { S } , \mathbf { X } ) = \frac { 1 } { S ^ { 2 } } \sum _ { i = 1 } ^ { S } \sum _ { j = 1 } ^ { S } W ( \mathbf { X } ) _ { i , j } \left[ \lambda _ { \mathrm { m s e } } ( \mathbf { S } _ { i , j } - \mathbf { X } _ { i , j } ) ^ { 2 } + \lambda _ { \mathrm { m a e } } | \mathbf { S } _ { i , j } - \mathbf { X } _ { i , j } | \right] ,\tag{10}
$$

with $\lambda _ { \mathrm { m s e } } = \lambda _ { \mathrm { m a e } } = 1$ . The intensity weight is

$$
[ W ( { \bf X } ) ] _ { i , j } = q _ { k _ { i , j } } , \qquad k _ { i , j } = \sum _ { m = 1 } ^ { 6 } { \bf 1 } \{ { \bf X } _ { i , j } > \theta _ { m } \} ,\tag{11}
$$

where $( i , j )$ indexes a grid cell, $\pmb { \theta } ~ = ~ ( - 0 . 4 , 0 , 0 . 2 6 6 7 , 0 . 4 6 6 7 , 0 . 6 6 6 7 , 0 . 9 )$ and ${ \textbf { q } } =$ (1, 1, 2, 4, 6, 8, 10) for normalised reflectivity values in [ 1, 1].

## 7.3.2 Probabilistic-residual flow-matching loss

Using the probabilistic residual target in Eq. (2), the interpolation in Eq. (3) and the target velocity in Eq. (4), we train the residual decoder with

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { ( \mathbf { H } , \mathbf { X } _ { 1 : T } ) \sim p _ { \mathrm { d a t a } } , \tau , \mathbf { Z } } \left[ \frac { 1 } { T S ^ { 2 } } \left. \mathbf { v } _ { \psi , \phi } ( \mathbf { r } ( \tau ) , \tau , \mathbf { H } ) - \mathbf { u } ^ { \star } \right. _ { 2 } ^ { 2 } \right] .\tag{12}
$$

We train for 200,000 optimisation steps with a batch size of 8 using AdamW $( \beta _ { 1 } =$ $0 . 9 , \beta _ { 2 } = 0 . 9 9 9 , \epsilon = 1 0 ^ { - 8 }$ and weight decay 0.01). The learning rate is linearly increased from $1 0 ^ { - 5 }$ to a peak of $2 \times 1 0 ^ { - 4 }$ over the first 10% of training and then cosine-decayed to $1 0 ^ { - 5 }$ over the remaining steps. Training uses mixed precision and DeepSpeed gradient clipping with a global-norm threshold of 1.0. No data augmentation beyond random sample shufling is used.

## 7.4 Datasets

We evaluate all methods on three regional radar datasets from the US, EU and CN domains. Although the three datasets are constructed from diferent event sources and radar archives, they are converted to a common nowcasting protocol. Each sample contains 9 historical radar frames and 36 future target frames at 10 min cadence, corresponding to a 1.5 h input window and a 6 h forecast horizon. Radar fields are represented on a common spatial grid and extracted over a fixed 256 km $\times ~ 2 5 6$ km physical domain. Models use preprocessed radar reflectivity as input and predict future reflectivity. Reflectivity is then converted to rain rate for verification and precipitation visualisation.

The US dataset is constructed by pairing NOAA/NCEI Storm Events reports with MRMS Seamless Hybrid Scan Reflectivity radar composites [20, 21]. The NCEI records provide the occurrence time, location, event type and impact information for severe weather events, while MRMS provides the corresponding high-resolution precipitation structure within the event-centred radar window. We extract radar sequences using the reported event time and location as anchors, and construct the training and evaluation samples after quality control and de-duplication. The training period spans January 2021 to September 2024 and contains 152,408 samples. The evaluation period spans October 2024 to February 2026. For the main evaluation, we use a compact test set of 4,000 samples constructed from the future holdout period; this set preserves spatiotemporal diversity while prioritising high-impact events.

The EU dataset follows a similar event-record-driven strategy, but uses the European Severe Weather Database (ESWD) as the event source and OPERA radar mosaics as the radar source [22, 23]. ESWD provides event-level records for severe convective weather, including wind, hail, heavy precipitation, tornadoes and lightning. We retain events that fall within the OPERA radar coverage and have a complete nowcasting time window, and then extract the corresponding radar sequences. Because the reported event time and location do not necessarily coincide with a pronounced precipitation echo in the radar field, we apply radar-signal quality control to remove samples with little or no precipitation signal. The training period spans January 2020 to December 2023 and contains 131,761 samples. The evaluation period spans January 2024 to 23 July 2024. The main evaluation uses a compact 4,000-sample test set that balances event diversity and high-impact event coverage.

The CN dataset is rebuilt from China Meteorological Administration composite reflectivity data using a radar-native, peak-anchored sampling strategy. The sampler first identifies spatially organised future strong-echo peaks in the radar sequence by requiring both local window-mean and window-maximum reflectivity to satisfy strongprecipitation criteria, reducing the chance that isolated noisy pixels trigger candidate samples. For each retained peak, the forecast initialisation time is generated by shifting backward from the peak time using sparse lead-to-peak buckets of 60, 120, 180 and 240 min. This peak-anchored design samples near-peak, mid-growth and early-growth stages, allowing the evaluation to probe 6 h forecasts of heavy-precipitation growth, maintenance and reorganisation.

To reduce near-duplicate samples from the same storm system, we apply frameand peak-level non-maximum suppression and limit both the daily number of samples and the number of issue times retained for each peak. Training samples are additionally required to retain late-label strong-echo activity, ensuring that the 6 h target window contains meaningful heavy-precipitation evolution. The CN training period spans November 2021 to June 2025 and contains 174,753 samples. The evaluation period spans July 2025 to June 2026, and the main evaluation uses a temporally held-out strict compact test set of 4,000 samples. This compact test set requires strong-echo activity in every forecast hour and future centre-region reflectivity maxima of at least 45 dBZ. It is stratified by month, convective-growth stage and lead-to-peak bucket, and uses a 2 h / 200 km time-space diversity rule to reduce near-duplicate samples.

Together, the three regional datasets support a common evaluation goal: testing long-lead radar nowcasting skill for heavy precipitation and high-impact convective weather. The US and EU datasets provide event-record-centred evaluations, while the CN dataset provides a complementary radar-native evaluation of strong-precipitation processes. Despite these diferences in event definition, all three datasets are converted to the same input-output protocol, allowing models and baselines to be compared under a consistent technical setting. This design also determines the focus of our evaluation: high rain-rate thresholds, long forecast lead times and the preservation of organised strong-echo structure, rather than average performance over all precipitation intensities.

Table 2 Regional radar datasets used in this study. All datasets are converted to a common 9-frame input and 36-frame forecast protocol at 10 min cadence.
<table><tr><td>Region</td><td>Construction</td><td>Training set</td><td>Evaluation set</td></tr><tr><td>US</td><td>NOAA/NCEI Storm Events + MRMS SeamlessHSR</td><td>Jan. 2021–Sep. 2024 152,408 samples</td><td>Oct. 2024–Feb. 2026 4,000 samples</td></tr><tr><td>EU</td><td>ESWD severe-weather records + OPERA radar mosaics</td><td>Jan. 2020–Dec. 2023 131,761 samples</td><td>Jan. 2024–23 Jul. 2024 4,000 samples</td></tr><tr><td>CN</td><td>Radar-native strong-precipitation sampling + CMA reflectivity</td><td>Nov. 2021–Jun. 2025 174,753 samples</td><td>Jul. 2025–Jun. 2026 4,000 samples</td></tr></table>

## 7.5 Baselines

We compare against three representative baselines spanning classical extrapolation, deterministic deep prediction and physics-guided generative nowcasting. All methods use the same 9-frame history length, 36-frame forecast horizon, reflectivity preprocessing and 128 128 verification targets over a 256 km 256 km domain. The learned models receive the corresponding 128 128 input histories. As described below, PyS-TEPS is intentionally supplied with a larger 256 256 spatial context before its forecasts are centre-cropped to the common verification domain.

## 7.5.1 PySTEPS

PySTEPS [6] is used as the classical transport-based ensemble baseline. We apply the STEPS nowcasting method [5] through the oficial PySTEPS implementation, using Lucas–Kanade motion estimation and a second-order autoregressive model. For each sample, PySTEPS receives the 9-frame, 256 256 radar history and predicts the subsequent 36 frames. The forecasts are then centre-cropped to the common 128 128 verification domain. This setting provides PySTEPS with broader spatial context than the learned models, including upstream precipitation that may subsequently advect into the verification domain. We generate four ensemble members using independent random seeds for probabilistic evaluation.

## 7.5.2 SimVPv2

SimVPv2 [24] is used as the deterministic deep-learning baseline. We train it on the same training split as MW-Nowcast using the oficial model implementation, with the input and output lengths adjusted to 9 and 36 frames, respectively, and the input tensor shape fixed at 128 128. To provide a strong deterministic baseline for the high-intensity precipitation regime considered here, SimVPv2 is trained with the same per-frame intensity-weighted MSE-plus-MAE reconstruction loss as the deterministic pixel head of MW-Nowcast, including identical intensity-bin weights and equal coeficients for the MSE and MAE terms. This setting places the same emphasis on intense precipitation in both deterministic predictors, allowing their comparison to focus more directly on diferences in model architecture. All other hyperparameters follow the default optimiser and scheduler settings of the released code unless adjustment is required by the modified input–output configuration.

## 7.5.3 NowcastNet

NowcastNet [11] is used as the physics-guided generative baseline. We retrain it on our dataset using the oficial released code, with the input/output length adapted to the 9-frame history and 36-frame forecast setting used in this paper. We then evaluate the retrained model on the corresponding chronological held-out split under the same preprocessing and reflectivity-space evaluation protocol as the other methods.

## 7.6 Evaluation settings and metrics

The primary forecast object in this paper is the ensemble itself. For selected figures and thresholded summary comparisons that require a single field, we also report Probability Matched Mean (PMM) [26]. For a grid with $S ^ { 2 }$ cells and K ensemble members, let $\begin{array} { r } { m _ { i } = K ^ { - 1 } \sum _ { k = 1 } ^ { K } x _ { i } ^ { ( k ) } } \end{array}$ be the ensemble mean at cell i, and let $r _ { i }$ be the rank of $m _ { i }$ among the $S ^ { 2 }$ mean-field values. If $Q _ { \mathrm { e n s } } ( p )$ denotes the empirical quantile of the pooled ensemble values $\{ x _ { j } ^ { ( k ) } : j = 1 , \ldots , S ^ { 2 } ; k = 1 , \ldots , K \}$ , the PMM field is computed as

$$
\tilde { x } _ { i } = Q _ { \mathrm { e n s } } \left( \frac { r _ { i } - 1 / 2 } { S ^ { 2 } } \right) .\tag{13}
$$

Equivalently, PMM ranks the grid cells of the ensemble mean and replaces them with values drawn from the pooled ensemble-value distribution according to the same rank order. This retains the spatial organisation of the ensemble mean while avoiding some intensity damping caused by arithmetic averaging.

All models operate in reflectivity-space (dBZ). When rain-rate-based thresholds are needed for evaluation, we convert reflectivity to rain rate using the WSR-88D $Z { - } R$ relation [32]

$$
Z = 3 0 0 R ^ { 1 . 4 } , \qquad Z = 1 0 ^ { \mathrm { d B Z / 1 0 } } ,\tag{14}
$$

which yields

$$
R = \left( \frac { 1 0 ^ { \mathrm { d B Z / 1 0 } } } { 3 0 0 } \right) ^ { 1 / 1 . 4 } .\tag{15}
$$

This conversion is applied only in evaluation and figure labelling, not in model training.

## 7.6.1 Grid metrics

Critical Success Index Neighbourhood (CSIN). Following the neighbourhood verification used by NowcastNet [11], CSIN allows limited spatial displacement when assessing threshold exceedances. We first convert the forecast and observation to rain rate and form binary exceedance masks $1 \{ R \ge \rho \}$ . Each mask is then dilated by taking the maximum over a centred $w \times w$ grid-cell neighbourhood. Hits $( \mathrm { T P } _ { w } )$ , false alarms $\left( \mathrm { F P } _ { w } \right)$ and misses $\left( \mathrm { F N } _ { w } \right)$ are accumulated from the resulting neighbourhood masks over all grid cells in the test set, giving

$$
\mathrm { C S I N } ( \rho , w ) = \frac { \mathrm { T P } _ { w } } { \mathrm { T P } _ { w } + \mathrm { F P } _ { w } + \mathrm { F N } _ { w } } .\tag{16}
$$

We use the $5 \times 5$ neighbourhood adopted by NowcastNet unless a diferent window is specified. For the regional quantitative comparison in Figure 4, we report CSIN at $\rho \in \{ 1 6 , 3 2 , 6 4 \}$ mm $\mathrm { h } ^ { - 1 }$ ; other case and ensemble figures use $\rho \in \{ 3 2 , 6 4 \}$ mm $\mathrm { h } ^ { - 1 }$ When a method produces an ensemble, CSIN is reported on the PMM summary field unless otherwise specified in a figure caption. Curves by forecast lead are computed first and, where a single score is needed, averaged over the reported lead times.

Probability of Detection (POD). POD measures the fraction of observed threshold exceedances that are detected by the forecast. Using the original pointwise threshold masks, it is defined as

$$
\mathrm { P O D } ( \rho ) = \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } } .\tag{17}
$$

## 7.6.2 Probabilistic metrics

Continuous Ranked Probability Score (CRPS). CRPS [27] is computed directly from the empirical ensemble distribution rather than from a fitted Gaussian. It measures the distance between the forecast distribution and the observation, with lower values indicating better probabilistic forecasts. For an ensemble forecast $\{ x ^ { ( k ) } \} _ { k = 1 } ^ { K }$ and observation $y ,$ the grid-cell CRPS is

$$
\mathrm { C R P S } ( \{ x ^ { ( k ) } \} , y ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \Big | x ^ { ( k ) } - y \Big | - \frac { 1 } { 2 K ^ { 2 } } \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { K } \Big | x ^ { ( k ) } - x ^ { ( l ) } \Big | .\tag{18}
$$

We use $K = 4$ members for MW-Nowcast, PySTEPS, and NowcastNet when probabilistic evaluation is reported. CRPS is computed in precipitation-space at the original grid resolution, then averaged over all grid cells and lead times. Deterministic baselines are excluded from CRPS tables unless explicitly converted to a one-member degenerate forecast.

Continuous Ranked Probability Skill Score (CRPSS). CRPSS reports the relative improvement in CRPS over a reference forecast. We use a persistence forecast, obtained by holding the latest observed radar field constant over the forecast horizon, as the reference forecast when computing CRPSS. For a method with score $\mathrm { C R P S } _ { \mathrm { m o d e l } }$ and a reference score $\mathrm { C R P S } _ { \mathrm { r e f } }$ , we compute

$$
\mathrm { C R P S S } = 1 - \frac { \mathrm { C R P S } _ { \mathrm { m o d e l } } } { \mathrm { C R P S } _ { \mathrm { r e f } } } .\tag{19}
$$

Higher CRPSS values indicate better probabilistic skill relative to the reference forecast; a value of zero indicates equal skill to the reference, and negative values indicate worse skill.

## 7.6.3 Relative economic value

Following the cost–loss decision framework used by DGMR [10] and standard ensemble verification [30], we evaluate whether probabilistic forecasts can reduce the expense of protective decisions. At a fixed region, forecast lead and rain-rate threshold $\rho ,$ each

grid cell and issue time defines a binary event. For a K-member forecast, the predicted event probability is

$$
{ \widehat { p } } = { \frac { 1 } { K } } \sum _ { k = 1 } ^ { K } \mathbf { 1 } \Big \{ r ^ { ( k ) } \geq \rho \Big \} ,\tag{20}
$$

where $r ^ { ( k ) }$ is the rain rate of member $k ;$ the verifying event is $y = \mathbf { 1 } \{ r ^ { \mathrm { o b s } } \geq \rho \}$ . A user takes protective action when $\widehat { p } \geq \gamma$ , where $\gamma \in [ 0 , 1 ]$ is a probability decision threshold. Taking action incurs cost $C ,$ whereas failing to act when the observed event occurs incurs loss L. If $N _ { \mathrm { T P } } , \ N _ { \mathrm { F P } }$ , N<sub>TN</sub> and $N _ { \mathrm { F N } }$ are the resulting contingency counts, the mean forecast expense is

$$
E _ { \mathrm { f } } ( \gamma ) = \frac { ( N _ { \mathrm { T P } } + N _ { \mathrm { F P } } ) C + N _ { \mathrm { F N } } L } { N _ { \mathrm { T P } } + N _ { \mathrm { F P } } + N _ { \mathrm { T N } } + N _ { \mathrm { F N } } } .\tag{21}
$$

Let $p _ { \mathrm { c } }$ be the observed climatological event frequency. The minimum expense of the optimal climatological strategy and the expense of a perfect forecast are, respectively,

$$
E _ { \mathrm { c } } = \mathrm { m i n } ( C , p _ { \mathrm { c } } L ) , \qquad E _ { \mathrm { p } } = p _ { \mathrm { c } } C .\tag{22}
$$

Because the value depends on $C$ and L only through the cost–loss ratio $\alpha = C / L$ we set $L = 1$ and $C = \alpha$ . For each $\alpha \in ( 0 , 1 )$ , we sweep the attainable probability thresholds $\gamma$ and report the maximum relative economic value

$$
\mathrm { R E V } ( \alpha ) = \operatorname* { m a x } _ { \gamma } \frac { E _ { \mathrm { c } } - E _ { \mathrm { f } } ( \gamma ) } { E _ { \mathrm { c } } - E _ { \mathrm { p } } } .\tag{23}
$$

This normalisation is algebraically equivalent to that used by DGMR: REV = 1 for a perfect forecast, $\mathrm { R E V } = 0$ for the optimal climatological strategy, and negative values indicate worse expense than climatology. We use $K = 4$ members and evaluate rainrate exceedances at $\rho \in \{ 1 6 , 3 2 \}$ mm $\mathrm { h } ^ { - 1 }$ for lead times of $2 ,$ 4 and 6 h in each region. Counts and climatological frequencies are pooled over all grid cells and samples in the corresponding chronologically held-out compact test set. Deterministic predictions are treated as degenerate one-member probabilities.

## 7.6.4 Structural diagnostic

Power Spectral Density (PSD). PSD [29] is used as a diagnostic of spatial structure. For each lead time and field $\mathbf { \Psi } _ { X } \in \mathbb { R } ^ { \bar { S } \times \bar { S } }$ , we compute the two-dimensional discrete Fourier transform ${ \mathcal { F } } ( X )$ and its power spectrum

$$
P ( k _ { x } , k _ { y } ) = \frac { 1 } { S ^ { 2 } } \left| \mathcal { F } ( X ) ( k _ { x } , k _ { y } ) \right| ^ { 2 } .\tag{24}
$$

The PSD curve is obtained by radially averaging $P ( k _ { x } , k _ { y } )$ over spatial wave number $k = \sqrt { k _ { x } ^ { 2 } + k _ { y } ^ { 2 } }$ . In figures, we plot forecast and observation PSD curves together at selected forecast lead times. PSD is used as a structural diagnostic curve rather than collapsed into a single scalar score.

## 7.7 Motivation for the residual formulation

All model inputs and outputs are radar reflectivity composites in dBZ on a fixed Cartesian grid after preprocessing. Observed radar sequences combine relatively predictable storm-scale displacement with less predictable local growth, decay, merging, splitting and convective initiation. For the following conceptual continuous-time statement, X(t) denotes the continuously indexed reflectivity field corresponding to $\mathbf { X } _ { t } .$ and $\mathbf { v } ( t )$ denotes the standard advection velocity. We express the physical motivation as

$$
\frac { \partial { \bf X } ( t ) } { \partial t } = \underbrace { - { \bf v } ( t ) \cdot \nabla { \bf X } ( t ) } _ { { \bf S } _ { t } \mathrm { ( t r a n s p o r t - d o m i n a t e d ) } } + \underbrace { { \bf R } _ { t } ^ { ( k ) } } _ { \mathrm { s o u r c e - l i k e ~ r e s i d u a l } } ,\tag{25}
$$

where the underbraces indicate the modelling correspondence to the previously defined deterministic prediction $\mathbf { S } _ { t }$ and probabilistic residual $\mathbf { R } _ { t } ^ { ( k ) }$ . Equation (25) is used only as physical motivation: the model does not explicitly estimate $\mathbf { v } ( t )$ or impose the transport equation as a training constraint.

Radar reflectivity evolution over several hours contains components with diferent predictability. Storm-scale displacement and broad precipitation organisation are often partly constrained by the recent radar history, whereas local growth, decay, merging, splitting and convective initiation become increasingly uncertain with lead time. This motivates the decomposition in Eq. (1): the deterministic prediction $\mathbf { S } _ { 1 : T }$ represents the shared forecast structure supported by the observed history, while the probabilistic residual $\mathbf { R } _ { 1 : T } ^ { ( k ) }$ represents member-dependent local departures from that shared structure. This decomposition is not imposed by an explicit advection equation; instead, the allocation between deterministic structure and probabilistic residual variability is learned jointly from data.

## 8 Data availability

The US severe-weather event records used in this study were obtained from the NOAA/NCEI Storm Events Database through the oficial NCEI bulk-data archive (https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/). The corresponding radar observations were obtained from the NOAA Multi-Radar/Multi-Sensor (MRMS) system, for which oficial data-access information is available at https:// www.nssl.noaa.gov/projects/mrms/MRMS data.php. The European severe-weather event records were obtained from the European Severe Weather Database (ESWD; https://www.eswd. $\mathrm { . e u / } )$ , operated by the European Severe Storms Laboratory, and were used in accordance with the ESWD User’s Agreement. The European maximumreflectivity radar mosaics were obtained from EUMETNET OPERA under licence; access to these data requires application to and approval by EUMETNET. The China dataset was constructed from composite reflectivity data provided by the China Meteorological Administration (CMA); access to these data requires application to and approval by CMA.

## 9 Code availability

The model code and trained checkpoints for MW-Nowcast will be made publicly available upon publication at https://github.com/microsoft/MW-Nowcast. The baseline implementations are based on PySTEPS (https://github.com/pySTEPS/pysteps), NowcastNet (https://doi.org/10.24433/CO.0832447.v1) and OpenSTL (https:// github.com/chengtan9907/OpenSTL).

## 10 Acknowledgements

We acknowledge the National Centers for Environmental Information (NCEI) for the NOAA Storm Events Database, the National Oceanic and Atmospheric Administration (NOAA) for the Multi-Radar Multi-Sensor (MRMS) products, the European Severe Storms Laboratory (ESSL) for the European Severe Weather Database (ESWD), EUMETNET OPERA for the European radar mosaics, and the China Meteorological Administration (CMA) for the composite reflectivity data used in this study.

## 11 Contributions

H.D., N.W. and Z.F. conceived the study and designed the overall methodology. Z.F. and Z.L. curated and processed the datasets. Z.F. and N.W. implemented the experimental pipeline, performed the experiments and analysed the results. S.Q., P.Z. and S.X. maintained the operational data pipelines. N.W. drafted the initial manuscript. W.J. substantially revised the manuscript and led its scientific framing, including the central story, claims and interpretation of the results. J.W., R.E.T., K.T., N.W., Z.F. and H.D. contributed to the scientific framing, interpretation of the results and revision of the manuscript. J.B. and H.X. reviewed the manuscript and provided feedback. M.C., H.S., J.K. and K.T. contributed to product requirements and operational context. N.G., B.Z., L.Z., D.D., Q.Z., H.S., K.T. and S.I. provided scientific and strategic guidance and project oversight. H.D. supervised and coordinated the study. All authors reviewed and approved the final manuscript.

## 12 Competing interests

The authors declare no competing interests.

![](images/d19bb249d0b8a1579afa05d2bc8cfd7e2add57c5514950d1b09ac631a2564c33.jpg)

![](images/e96970d897c3644ada10fe0c927df6f714cd8dc127657fb124b5e1d5a41828a2.jpg)

![](images/dd1b9e9a42e486433c5398e90f561e4f7f7840cfafc2ec6d8b6e43157df97ca7.jpg)

b  
![](images/ec5714eef4b850daf5ccacf186bfb2fe839cba5579cebee9e5213f526d870580.jpg)

d  
![](images/9f2fe0ac6c91c719615d2c587101452ff219cd4be8026debfc2e96c0b6176cc9.jpg)

![](images/4efb0d903a8c333f6075700a0d1c38a273f72da032259a24c4b982163f9436cb.jpg)  
Extended Data Fig. 1 US flash-flood precipitation forecast. The case is located in Tipton County, Tennessee, on 19 June 2025, and the forecast is initialised at 09:10 UTC. The observed sequence shows an organised heavy-precipitation band moving from the northwestern part of the forecast patch towards the southeast. The band remains relatively continuous during the first half of the 6 h window, before the main precipitation shifts towards the southern to southeastern part of the patch and locally weakens at later lead times. Embedded intense cores repeatedly reorganise during this evolution, making the case sensitive to rain-band displacement, core retention and late-lead structural evolution. All Extended Data case figures use the same five-panel layout: a, Geographic and radar context, with the forecast domain outlined in orange. b, Forecast diagnostics: CSIN in $\mathrm { ~ a ~ } 5 \times 5$ grid-cell neighbourhood at rain-rate thresholds of 32 and 64 mm $\mathrm { h } ^ { - 1 }$ , CRPS, CRPSS, and radially averaged PSD at lead times of 3 and 6 h. “Deterministic” denotes the output of the deterministic decoder of MW-Nowcast. c, Observed and predicted rain-rate fields at hourly lead times from 1 to 6 h. Rows show observations, PySTEPS, SimVPv2, NowcastNet and MW-Nowcast. d, CSIN and POD at thresholds of 32 and 64 mm $\mathrm { h } ^ { - 1 }$ for the four MW-Nowcast ensemble members and their PMM summary. e, The deterministic prediction and four ensemble members at lead times of 2, 4 and 6 h.

![](images/d226caaadf32fbc2b561626f99a1af4547174a3fbfb9d16254b469c57b0d1be5.jpg)

![](images/01b845f1df01b2a7e2ca58b8c38725af8e2d17c9a4287196391d10cc70875ede.jpg)

![](images/5483ddc8f5a4884f9b6488d903d11709c1dde80b48aa31e85adbfa0e7d03007d.jpg)

b  
![](images/84814a5cc99c7426861937747c68fe743b5494d944ab01d6955bd65b1dc9014d.jpg)

![](images/674a6203e8344b726e8fe7fe066903181b673bdc15bf90d22daba28c5d932887.jpg)

![](images/ab00af8e847080be9f7a1c950c67d39f34cedfd97f40104fa9f9fd9c2ae290c0.jpg)

d  
![](images/27bf32da8a21fca989eb89fcec145d3dff14267e59cdf35f4f28762561de02c7.jpg)

![](images/4ac2e37478dbffb56a165c3abf5bb8122c1764bcc22d48d1ac97de818133ede2.jpg)

![](images/429c05ea476dab907cbc08b59bd7aba385622892ebc5ae87474584911ab3925d.jpg)

![](images/9f5dc18af9f15969b6ff176ecf9645794fca264e284d751047d230a5a0ab39b0.jpg)

![](images/9e430f9c2f737b7eea7a36759e34d3c2ea478cd68dfce6515fe6fa4de75009e3.jpg)

![](images/26fad19752ff87e9aa3db0eec039b6030d82553d85779b5bab9f9bf0ee90cf89.jpg)  
Extended Data Fig. 2 US thunderstorm-wind convective precipitation forecast. The case is located in Rockland County, New York, on 14 July 2025, and the forecast is initialised at 19:30 UTC. The observed sequence shows convective precipitation moving generally eastward and becoming more organised into a coherent rain band during the first half of the forecast window. At later lead times, the precipitation coverage contracts, and the embedded intense cores weaken and become more fragmented. This case tests whether a model can retain rain-band structure during convective organisation while representing the subsequent contraction and decay of high-intensity cores. Panel descriptions are as in Extended Data Fig. 1.

![](images/9bf5e4928ae87f1795ae61ce836b4bb69257ce0a64c0a7aa24b0c535b8af6d58.jpg)

![](images/1c18019578435c41b8b3d39e57a5e6b50e26f392d2ee58d2f36a5aa1f1df007b.jpg)

![](images/ab2cb836e9ac19c5547ba2e355e2847c665c97efc075c6014ca237829a7dd209.jpg)

![](images/325485d2ff7905a130136b2399e35d2f95cb896f11a25032c12007721be64038.jpg)

a  
![](images/3ab932e4e6d9f0bee578e387b988c123ecd0c19ed7d9712a4ad5725fb5fba917.jpg)  
b

![](images/7d17cf57ec466a94af8de599090af149dd517ddcb73d91eb9aafb81ed6fe8340.jpg)  
d

![](images/28254465d434e164b2675c614ac7e907a9c2991870f9c3aa947e569958130b71.jpg)

![](images/df47c806c45981f07ffd534a36a407f293e1b3d24f3af0d0789f9cb5a859b639.jpg)  
Extended Data Fig. 3 US hail-associated convective precipitation forecast. The case is located in Douglas County, Nebraska, on 24 April 2025, and the forecast is initialised at 21:20 UTC. The observed sequence shows a convective echo field moving generally eastward and expanding across the forecast patch, with intense precipitation organised as several compact local cores rather than a single continuous rain band. These cores repeatedly develop, split and reorganise during the 6 h window, making the case sensitive to convective-core initiation, displacement and multi-core structural retention. Panel descriptions are as in Extended Data Fig. 1.

![](images/b094bc1c37a49c70186ea3584fb1e9b17a442c0cb022dffcc4b21643ee0e2bbb.jpg)

![](images/c1c25f7947502d612076f4525f4a87128522bfd3a75b3c25b593f01da93d0fe0.jpg)  
b

![](images/5952a195581022e81e2bc80c4e1ce3fa4b1c3efddaa50bf310a38231c5e08e47.jpg)

![](images/c91f6a3697d81be239dcb56c39a17e7849ed98ffaabc355f708db041daea5eca.jpg)

![](images/2476bdc348c0f72dfe0493d9a518c0e2457fd1db4e48f487e0861d89a6f3643f.jpg)

![](images/6da05998ee3c64d2059956950e34772e922dd6d1b661deea6618538db4e6fcca.jpg)

![](images/84f910bdc20f9e581e8daf3a872a94d471439bebdad4b4afac649f3d978e16fa.jpg)

d  
![](images/f650b6691550d3bcf1cd1005fdb3851ba452675e4dc6bddcff6f2031839b2b4b.jpg)

![](images/a1fd29c7e0129117921a3c91b4ead2510ca0ad1c3039c5e9cc2a5eeaa3421759.jpg)

![](images/6f1b7553a90673d7d76462943145439764c987f953098d827c64f64ec77a6747.jpg)

![](images/683c8a6206f43530ba105a47885dc2d2d3db10ad0c0c212b5718a8ed105466a9.jpg)

![](images/e5b0898ac48fae61893a28b1550dd4e7eb1065499ddbc0d13392aed3a8c0e98a.jpg)  
Extended Data Fig. 4 Mountainous heavy-rainfall forecast over western Hebei. The forecast patch is centred near Yi County, Baoding, and is initialised at 14:50 UTC on 24 July 2025. The observed sequence shows persistent heavy precipitation that remains active within the forecast patch and gradually expands, while the main rain area retains broad spatial continuity. Peak intensity weakens at later lead times, but embedded local intense cores continue to reorganise. This case tests whether a model can retain rain-area extent and structure in a persistent mountainous rainfall setting while representing the weakening and reorganisation of local intense cores. Panel descriptions are as in Extended Data Fig. 1.

![](images/67e8a59c9f06ed2b36211f4886301d31126a47819d1295ce4634c1c05590c30c.jpg)

![](images/790831b694bce8667f3ad660fc54fc59a88ad25df0ffdc2b46a29362cd7e255a.jpg)  
b

![](images/b2cd725c9b7c69fe7ba2a2b13d05e27e792a45a11aef7000f532453aeb0059f4.jpg)

![](images/9b1f2817896e70b84fceae1e3639813ab164138ee10466324f39c5bee88a1e36.jpg)

![](images/1f15a7ca3b32e9f8df57bef4a16362eb7e327812874ea8ae9273aa4b5a080a96.jpg)

![](images/f34522d0993213163c00a2781427e7e12d0977b9d888e18b8f1455d55a53516b.jpg)

![](images/6b762759b6db0ced48a49f7bef3d8b3d91a010a4c634c1835c6c1f454bc9097f.jpg)

d  
![](images/ba793cc6692304494f075beddbd8576c19858e5863129787acbf9235d8548300.jpg)

![](images/5cb84f49306a0f0c881aaca36f2ce6dafa78c7f5eb0327017ec03e56c934ffab.jpg)

![](images/c1d438bb3d18c9c5ee5e9b81482e8960dce77cc1b0dc4a7caaaa64a9a9962de4.jpg)

![](images/479eb08805674739a86d736d56296035ef2cc760f91c9934406b9aec9b082ecf.jpg)

![](images/ce6f3903dc7fea37e7306042a9ae9705166275462de1a16b6e32c93c27c182d7.jpg)  
Extended Data Fig. 5 Hunan Shimen mountainous heavy-rainfall forecast. This forecast window forms part of the May 2026 heavy-rainfall episode in Shimen County, Hunan, during which public reports documented six deaths and ten missing people. The forecast is initialised at 08:20 UTC on 17 May 2026. The observed sequence shows a persistent heavy-precipitation system moving generally eastward and expanding across the forecast patch. Embedded intense cores remain strong through the 6 h window and locally reorganise as the rain area expands. This case tests whether a model can retain high-intensity cores and coherent rain-area structure in a moving and expanding mountainous rainfall system. Panel descriptions are as in Extended Data Fig. 1.

b  
d  
![](images/97d96e1d5395a0f1426a5abf1aac33df88697b2b1ae8aa52c20084fd00fea63a.jpg)  
Extended Data Fig. 6 Guizhou Guiding–Majiang heavy-rainfall forecast. This forecast window forms part of the 15–20 May 2026 rainstorm episode in Guizhou; an oficial disaster summary reported 19 people dead or missing across Guiding, Majiang and several other counties. The forecast is initialised at 22:10 UTC on 14 May 2026. The observed sequence shows a heavy-precipitation system moving generally eastward and gradually contracting after the middle of the forecast window. Although precipitation coverage decreases, embedded intense cores persist and locally reorganise. This case tests whether a model can retain the placement, intense cores and late-lead organisation of a moving heavy-rainfall system during contraction and structural adjustment. Panel descriptions are as in Extended Data Fig. 1.

![](images/c72eadda4e66c6e42297842e1b7be6898487c2a7f6a6b5871055c27972c99e7c.jpg)

![](images/d1071476a8b29c880fe049d1b886ee9c93069007c43a0d65ff45e9b1fd04f7c8.jpg)

![](images/56ae5341a85a22eb18a964d2d56893598350b604edb4027502d3a89ae378e5b9.jpg)

a  
![](images/4d3c95b0ad41540177ac8d708c833dc690d5476b8acbe855d6e1f0c53fd5214b.jpg)  
b

![](images/ea8c159f5f3484700b6a65bc1f83b32897e14e575c7ec22b67ead747f41f6d41.jpg)

![](images/ebf271c8ceba0f2571afca81e6d4f5e4720419cc14246dd8db7a2141239aef5a.jpg)

![](images/0e1d035b710cd344bb5461646a058ca3675895829ff5c6c31d66d1cb54e2b973.jpg)  
d

![](images/5dffcc85aaea0f217ed3500aa9e98a5e400b98cd5b0af78b4a9642af91f6d4a4.jpg)

![](images/eb9a4a2ffbaf5daf9ed61a2b8d0f30d33c9ae80f4308cdd5fe3f01e731be9aa2.jpg)

![](images/517dbd22744154ce7212cccc0eca3f680f2e3ed58cd402dd451e201957b8eb8a.jpg)

![](images/7b596e54c01db07fe54c9bcffcb2d8fc23b7a023cc9353938762f8ec8fb617a5.jpg)

![](images/0f41ce1a3188a6e9737f797ec54d156e46434498fb99e283c08aa5c9fcabd99f.jpg)  
Extended Data Fig. 7 European lightning-associated convective precipitation forecast. The case is located in southern Germany on 23 May 2024, and the forecast is initialised at 12:44 UTC. The observed sequence shows organised convective precipitation moving generally eastward and expanding through the first two-thirds of the forecast window. The precipitation area then contracts modestly, while embedded intense cores persist and locally reorganise along the main rain area. This case tests whether a model can retain rain-area continuity and intense-core structure in a moving and expanding convective system while representing the later modest contraction. Panel descriptions are as in Extended Data Fig. 1.

![](images/d85a0df3084e031a9ddd0c9862c7a122a96ea81cc7933c8672ce204dbebed2bd.jpg)

![](images/fe01a70840fac3c015d62ff649e31c2488d5f7db110337c2adeee181ada840d6.jpg)

![](images/f1acaf0ee9451c955c3b97ee732763b6efb1f2c2302deb99e24f97692d809646.jpg)

![](images/65312ef62ceebb1d3e3195dbef7dfdee256edb419d4d57bb856968eb35137eee.jpg)  
b

![](images/bc408c0a86607f4be1d431ee77da380ba6c45df41a10bfc5706ea578f65be097.jpg)

d  
![](images/46c1af6ab22ea2b36bfc45c054dce40de36aa88f371de55ed1882c55730a07ff.jpg)  
Extended Data Fig. 8 European tornado-associated convective precipitation forecast. The case is located in southern France on 3 March 2024, and the forecast is initialised at 00:30 UTC. The observed sequence shows a slowly moving precipitation system that expands through the 6 h window. Intense precipitation is organised mainly as multiple compact cores that repeatedly develop, split and reorganise, rather than as a stable single rain band. This case tests whether a model can represent local core initiation, displacement and morphology changes within a slowly moving but expanding convective system. Panel descriptions are as in Extended Data Fig. 1.

![](images/ca9722f8c6fa9b042fb1604b78fcfa8f01230ca1a8f03b0a5a5f25f8bd34c3e9.jpg)

![](images/b699809e98fe48965746291630e4dd73f584e8f528b8a9090ef811ad01929749.jpg)

![](images/bd8f985e272f4801ca28badb402bfaae547b0c0ebd957fef61407d34d2844d5b.jpg)

![](images/c413ca0e3ab1d27789542fbb857cfc0dca33c59eb42d9e751529624b503f0ecd.jpg)

![](images/85bff0343eb16053a5c3d07e21989f96b16930b56588bc5a9e2f3b74fe5a4e0b.jpg)

![](images/9f8994dba983b00fda0c55c43cc6ae03a4f7cd0e463efb1254a60a0326396d37.jpg)

![](images/c5a9bc07945dd2e6184528c493f3f32b3ae0edd90fbe379bc2edb15ba0d0a947.jpg)  
Extended Data Fig. 9 European heavy-precipitation forecast over southwestern France. The case is located in southwestern France on 8 June 2024, and the forecast is initialised at 18:00 UTC. The observed sequence shows a slowly moving heavy-precipitation area that expands substantially through the 6 h window, gradually forming a broader and more continuous rain region. Embedded intense cores continue to reorganise within the main rain area rather than rapidly moving out of the forecast patch. This case tests whether a model can retain rain-area extent, spatial continuity and local intense-core evolution in a slowly moving and steadily expanding heavy-precipitation system. Panel descriptions are as in Extended Data Fig. 1.

![](images/47fce9a9f25d4ac6b23b4e0dfc4f4df4f895546b54ef139260a7b11a7566fc0e.jpg)  
Extended Data Fig. 10 Regional rain-rate distributions and high-intensity-tail recovery. a–c, Conditional rain-rate densities pooled over 4,000 held-out events in the United States (US), Europe (EU) and China (CN), respectively. Densities use efective-rain pixels with $0 . 1 \leq R <$ 128 mm $\mathrm { h } ^ { - 1 }$ . Blue solid curves show the deterministic branch, orange solid curves show the Probability Matched Mean (PMM) ensemble summary, and black dashed curves show ground truth. The shaded region marks the high-intensity tail, $R \geq 1 6$ mm $\mathrm { h } ^ { - 1 }$ , corresponding to the lower high-rain threshold used in the regional quantitative evaluation. Insets report pooled pixel–time exceedance probability conditional on the displayed rain-rate range. Across all three regions, PMM recovers substantially more high-intensity probability mass than the deterministic branch and closely approaches the observed tail.

## Supplementary Information

## 1 Related works

## 1.1 Historical perspective

Precipitation nowcasting has developed along two complementary lines: extrapolating radar-echo motion and learning nonlinear evolution from radar sequences. Classical radar nowcasting starts from the observation that short-lead precipitation fields often contain a strong advective component. Echo motion is estimated from recent radar frames and then extrapolated into the future. STEPS and its PySTEPS implementation extend this idea by combining scale-dependent extrapolation with stochastic perturbations, yielding fast and interpretable ensemble forecasts [5, 6]. Related optical-flow and radar-tracking methods also exploit motion information in recent radar observations and remain valuable in rapidly updated operational settings [33]. Their limitation is that severe convective precipitation is not a passive tracer. As lead time extends towards 3–6 h, local growth, decay, merging, splitting and initiation become increasingly important, making pure extrapolation insuficient for maintaining intense cores and organised precipitation structure [1, 3].

Deep learning broadened the modelling capacity of radar nowcasting by learning nonlinear spatiotemporal evolution from large radar archives. ConvLSTM formulated precipitation nowcasting as convolutional recurrent sequence prediction, and later models such as PredRNN, EarthFormer and SimVPv2 further improved the representation of spatiotemporal dependencies [7–9, 24]. These deterministic models can capture morphology beyond linear extrapolation, but point-wise training losses tend to average over multiple plausible futures, weakening intense echoes at longer lead times. Generative radar nowcasting models address this limitation by sampling multiple future fields. DGMR showed that deep generative models can produce useful probabilistic precipitation nowcasts, and NowcastNet further demonstrated the value of incorporating physical structure for extreme-precipitation nowcasting [10, 11]. Together, these studies show that long-lead extreme-precipitation nowcasting requires both the retention of predictable structure supported by the recent radar history and the representation of uncertain local evolution.

## 1.2 Difusion-based and flow-based nowcasting

Recent difusion and flow-based nowcasting studies extend generative radar forecasting along complementary directions. PreDif and LDCast use latent difusion to make probabilistic precipitation nowcasting tractable while representing forecast uncertainty [17, 18]. StormDiT brings latent rectified-flow transformers into the 2–6 h precipitation nowcasting grey zone, and FlowCast explores conditional flow matching for precipitation nowcasting [19, 34]. CasCast and DifCast use cascaded or residual formulations to connect deterministic evolution with stochastic refinement [35, 36]. Together, these studies suggest a useful principle for longer-range nowcasting: forecast uncertainty should be represented explicitly, while the more predictable component of storm evolution should remain structurally anchored.

MW-Nowcast follows this principle but makes a diferent set of modelling choices. Several existing difusion nowcasting systems operate in latent space, which improves tractability but can make the recovery of compact high-intensity radar structures more dependent on the decoder. Other cascaded or residual systems separate deterministic prediction from stochastic refinement, so the residual model is trained relative to a fixed or separately trained forecast. In contrast, MW-Nowcast is a pixel-level, endto-end flow-matching ensemble nowcaster. It pairs a deterministic decoder, which provides the shared predictable forecast structure, with a probabilistic residual decoder, which samples local growth, decay, reorganisation and initiation around that shared forecast. Because the residual target is recomputed as the deterministic prediction changes during joint training, the two branches learn the decomposition together rather than forming a fixed post-processing cascade.

This distinction motivates the ablation experiments reported below. We compare pixel- and latent-space configurations, and use controlled comparisons to test an explicit deterministic-prediction– probabilistic-residual decomposition against a single generative head, joint end-to-end training against a two-stage residual cascade, and flow matching against an EDM difusion objective.

## 2 Ablation study

The ablation study evaluates which modelling choices contribute to the skill of MW-Nowcast. All comparison variants are evaluated on the held-out US test set described in the main text, using the same 9-frame

input, 36-frame output and 6 h nowcasting protocol as the main quantitative comparison. We compare MW-Nowcast with five variants: Latent, OneStage, TwoStage, EDM-ODE and EDM-SDE. These comparisons examine generation space, decoder decomposition, training strategy and probabilistic objective.

## 2.1 Decoder coupling and training strategy

The OneStage and TwoStage variants examine the design hypotheses underlying the deterministicprediction–probabilistic-residual formulation and the way this decomposition is learned. OneStage removes the explicit deterministic decoder and uses a single flow-matching generative head to produce the complete future sequence directly. Its lower $\mathrm { C S I N _ { 5 } }$ across all reported thresholds and lead times suggests that directly generating the full forecast sequence provides a less efective basis for detecting intense precipitation. One possible explanation is that a single generator must represent both the largescale, relatively predictable evolution and the smaller-scale uncertain departures, whereas the explicit decomposition allows these components to be handled by specialised decoders.

TwoStage retains the deterministic decoder and residual decoder but trains them sequentially: the deterministic decoder is trained first, and the residual decoder is then trained against a fixed residual target. Compared with OneStage, this configuration recovers part of the skill at the 32 and 64 mm $\mathrm { h } ^ { - 1 }$ thresholds, supporting the value of an explicit deterministic forecast anchor. However, it remains weaker than MW-Nowcast at all representative lead times and in the mean over the full forecast horizon reported in Supplementary Table 1. This remaining gap may arise because the residual decoder is learned from predictions produced by a fixed deterministic model. Joint optimisation in MW-Nowcast instead allows the shared representation, deterministic prediction and probabilistic residual target to adapt together during training.

## 2.2 Generation space and difusion objective

The Latent variant represents an alternative generative design that moves forecasting from radar-pixel space to a compact VAE latent space. Its consistently lower $\mathrm { C S I N _ { 5 } }$ suggests that this configuration retains less information relevant to intense-precipitation detection. A plausible contributing factor is the compression and reconstruction process, which may smooth compact precipitation cores or attenuate local extremes, causing fewer locations to exceed the high rain-rate thresholds after decoding. This efect is particularly relevant at high thresholds, where even modest attenuation can change whether an intense precipitation feature is detected.

The EDM-ODE and EDM-SDE variants retain the two-branch architecture and joint training setting, but replace the flow-matching residual objective used in MW-Nowcast with an EDM difusion objective. They difer only in the sampling procedure: EDM-ODE uses deterministic ODE sampling, whereas EDM-SDE uses stochastic sampling with added noise. MW-Nowcast achieves higher $\mathrm { C S I N _ { 5 } }$ than both EDM variants in every reported column, indicating a consistent advantage for flow matching under the evaluated setting. The absolute diferences are nevertheless modest, particularly at the highest threshold and longest lead times, and are consistent with flow matching being well suited to learning the residual transformation rather than identifying the probabilistic objective as the dominant source of improvement.

Taken together, the ablation results provide consistent empirical support for the main design hypotheses of MW-Nowcast. The clearest gains are associated with combining an explicit deterministic forecast anchor with probabilistic residual generation and optimising the two components jointly. The full pixelspace configuration performs best among the evaluated generation designs, while flow matching provides a further, more modest improvement over the EDM alternatives.

Supplementary Table 1 summarises the quantitative comparison using $\mathrm { C S I N _ { 5 } }$ at representative lead times and as a mean over the complete 6 h forecast horizon.

Supplementary Fig. 1 complements these aggregate results with a qualitative comparison for a representative strong-precipitation case. The example illustrates how the alternative configurations difer in their ability to retain compact intense precipitation cores and organised storm structure as lead time increases.

Supplementary Table 1 $\mathbf { C S I N _ { 5 } }$ comparison of MW-Nowcast ablation variants on the heldout US test set. Neighbourhood Critical Success Index with a $5 \times 5$ neighbourhood is reported at rain-rate thresholds of 16, 32 and 64 mm $\mathrm { h } ^ { - 1 }$ for representative lead times T+2 h, T+4 h and T+6 h, together with the arithmetic mean of the unsmoothed per-lead scores over all 36 lead times from $\mathrm { T } + 1 0$ min to T+6 h. Higher is better. The best displayed value in each column is shown in bold.
<table><tr><td>Model</td><td colspan="3">T+2 h</td><td colspan="3">T+4 h</td><td colspan="3">T+6 h</td><td colspan="3">Mean</td></tr><tr><td></td><td>16</td><td>32</td><td>64</td><td>16</td><td>32</td><td>64</td><td>16</td><td>32</td><td>64</td><td>16</td><td>32</td><td>64</td></tr><tr><td>MW-Nowcast</td><td>0.284</td><td>0.214</td><td>0.160</td><td>0.197</td><td>0.129</td><td>0.085</td><td>0.095</td><td>0.055</td><td>0.032</td><td>0.256</td><td>0.194</td><td>0.147</td></tr><tr><td>Latent</td><td>0.216</td><td>0.147</td><td>0.099</td><td>0.155</td><td>0.100</td><td>0.062</td><td>0.072</td><td>0.038</td><td>0.021</td><td>0.204</td><td>0.144</td><td>0.103</td></tr><tr><td>OneStage</td><td>0.239</td><td>0.168</td><td>0.118</td><td>0.164</td><td>0.096</td><td>0.058</td><td>0.078</td><td>0.039</td><td>0.021</td><td>0.225</td><td>0.164</td><td>0.121</td></tr><tr><td>TwoStage</td><td>0.233</td><td>0.174</td><td>0.129</td><td>0.163</td><td>0.114</td><td>0.079</td><td>0.073</td><td>0.044</td><td>0.027</td><td>0.218</td><td>0.169</td><td>0.130</td></tr><tr><td>EDM-ODE</td><td>0.247</td><td>0.181</td><td>0.129</td><td>0.187</td><td>0.126</td><td>0.085</td><td>0.075</td><td>0.046</td><td>0.027</td><td>0.235</td><td>0.178</td><td>0.134</td></tr><tr><td>EDM-SDE</td><td>0.244</td><td>0.178</td><td>0.127</td><td>0.186</td><td>0.126</td><td>0.085</td><td>0.068</td><td>0.042</td><td>0.025</td><td>0.234</td><td>0.177</td><td>0.133</td></tr></table>

![](images/a0b0dc06fac92a87b0d123c3329c6dc2c8dc8482deca1935218a9db3f6852e70.jpg)  
Supplementary Fig. 1 Visual comparison of MW-Nowcast ablation variants for a held-out US strong-precipitation case. The forecast is initialised at 09:00 UTC on 13 February 2025 for a patch centred near $\mathrm { 2 9 . 4 ^ { \circ } N , ~ 8 7 . 5 ^ { \circ } W }$ . The figure includes the input context and the subsequent 6 h forecast evolution. Rows compare the observed fields, MW-Nowcast and the comparison variants Latent, OneStage, TwoStage, EDM-ODE and EDM-SDE. The case illustrates how the ablated design choices afect the retention of compact intense precipitation cores and organised storm structure over the forecast horizon.

## 3 Flow-matching visualisation

This section visualises how the residual decoder in MW-Nowcast constructs ensemble members. Supplementary Figs. 2 and 3 show single-member residual trajectories, illustrating how Gaussian noise is transformed into a structured probabilistic residual and then added to the deterministic prediction. Supplementary Figs. 4 and 5 show how multiple members diverge along the same learned flow, revealing where ensemble uncertainty is concentrated.

## 3.1 Single-member residual construction

For a fixed forecast lead time, the residual flow evolves from a Gaussian sample at $\tau = 0$ to a probabilistic residual at $\tau = 1$ , using the same Heun ODE solver as in deployment. Adding this probabilistic residual to the deterministic prediction yields the final prediction for that ensemble member. Supplementary Figs. 2 and 3 show this construction for a US hail case and a European thunderstorm case, each at lead times T+4 h, T+5 h and T+6 h.

In every panel, the top row traces the probabilistic residual as it evolves from noise at $\tau = 0 . 0 0$ to a structured field at $\tau = 1 . 0 0$ , and the bottom row shows the corresponding prediction obtained by adding the evolving probabilistic residual to the deterministic prediction. The right-hand column reports the deterministic prediction and the ground-truth observation for reference. The trajectories show that the probabilistic residual is not unstructured noise, but a spatially organised correction that modifies precipitation cores and fine-scale structure around the deterministic forecast.

![](images/19582d1826f6582f986945e9d5d1c862715380497f45ac820f4b19a776f4e310.jpg)  
(a) Lead time T+4 h.

![](images/67764c1ea0fb80ab84678a3255a471504ea403041e986fb804cdcdd7269f3c01.jpg)  
(b) Lead time T+5 h.

![](images/8b9ac55b3fb34469029099c44b5f6273a70d6cc72eff86efe7885fe949a53474.jpg)  
(c) Lead time T+6 h.  
Supplementary Fig. 2 Flow-matching construction of the probabilistic residual for a US hail case. The case was reported at 23:07 UTC on 19 October 2024 near $3 3 . 8 7 ^ { \circ } \mathrm { N } .$ $1 0 4 . 5 9 ^ { \circ } \mathrm { W }$ , with the radar clip spanning 20:07 UTC on 19 October to 05:07 UTC on 20 October over a patch of approximately 31.29–36.45<sup>◦</sup>N and 107.17– $1 0 2 . 0 1 ^ { \circ } \mathrm { W }$ . For each lead time, the top row shows the probabilistic residual evolving along the flow-matching trajectory from a Gaussian sample at $\tau = 0 . 0 0$ to its terminal state at $\tau = 1 . 0 0$ , and the bottom row shows the prediction formed by adding the evolving probabilistic residual to the deterministic prediction. The right-hand column gives the deterministic prediction (top) and the ground-truth observation (bottom).

![](images/abaf623f91d805854bb3430aea2ef4216966236c1baa1a4be08f30108674b2ff.jpg)  
(a) Lead time T+4 h.

![](images/8d69f4a7f73e9d293a0991b96d8cb5b405855243cab510d9dc1fef42ce37dcb2.jpg)  
(b) Lead time T+5 h.

![](images/b5157b99f417066e28052835d38e2cf3eaf981cc0705037911aeae2698d31e6a.jpg)  
(c) Lead time T+6 h.  
Supplementary Fig. 3 Flow-matching construction of the probabilistic residual for a European thunderstorm case. The radar clip spans 07:50–15:50 UTC on 15 May 2024 over the Alpine region, with the patch centred near $4 6 . 0 4 ^ { \circ } \mathrm { N } , 8 . 6 7 ^ { \circ } \mathrm { E }$ and covering approximately $4 3 . 3 8 { - } 4 8 . 5 0 ^ { \circ } \mathrm { N }$ and $6 . 3 1 { \mathrm { - } } 1 1 . 4 3 ^ { \circ } \mathrm { E } .$ Layout follows Supplementary Fig. 2: the top row of each panel traces the probabilistic residual along the flow-matching trajectory and the bottom row shows the resulting prediction, with the deterministic prediction and ground-truth observation in the right-hand column.

## 3.2 Ensemble divergence along the flow

The single-member trajectories above show how one realisation of the probabilistic residual is constructed, but they do not show how ensemble members separate from one another. Because all members share the same deterministic prediction and difer only in the Gaussian sample drawn at $\tau = 0$ , member-to-member spread is a direct visualisation of the conditional uncertainty represented by the residual decoder.

Supplementary Figs. 4 and 5 show four independently sampled members for the same two cases and lead times. Section A shows residual fields at $\tau = 0 . 7 5$ and $\tau = 1 . 0 0$ , when the residual has developed structured precipitation-related features. Section B summarises the converged residual ensemble using the per-pixel ensemble mean and standard deviation at $\tau = 1 . 0 0$ . The standard deviation concentrates near precipitation cores and edges rather than spreading uniformly across the domain, indicating that the residual decoder expresses uncertainty around meteorologically active structures.

![](images/fc725b3cf45ddd9a63746a5a9bf5d25177c9c57c479f7ebfa1db057c1ba3b418.jpg)  
(c) Lead time T+6 h.  
Supplementary Fig. 4 Ensemble divergence of the probabilistic residual for a US hail case. The case was reported at 23:07 UTC on 19 October 2024 near $3 3 . 8 7 ^ { \circ } \mathrm { N } ,$ 104.59<sup>◦</sup>W. Four members (seeds 1–4) are integrated along the flow from independent Gaussian samples. Section A: the probabilistic residual for each member at flow positions $\tau = 0 . 7 5$ (top row) and $\tau = 1 . 0 0$ (bottom row). Section B: the per-pixel ensemble mean (top) and standard deviation (bottom) of the probabilistic residual at $\tau = 1 . 0 0$ . The standard deviation localises along precipitation cores and edges, showing where the probabilistic residual contributes the most uncertainty.

![](images/6b1cb7a3f84c37becfab5c6d58061c1957865f16c66e507f0f5b0eed25623472.jpg)  
(c) Lead time T+6 h.  
Supplementary Fig. 5 Ensemble divergence of the probabilistic residual for a European thunderstorm case. The case occurred on 15 May 2024 over the Alpine region, with the patch centred near $4 6 . 0 4 ^ { \circ } \mathrm { N } ,$ $8 . 6 7 ^ { \circ } \mathrm { E }$ . Layout follows Supplementary Fig. 4: Section A shows four members at $\tau = 0 . 7 5$ and $\tau = 1 . 0 0$ , and Section B shows the ensemble mean and standard deviation of the probabilistic residua $\mathrm { ~ a t ~ } \tau = 1 . 0 0 .$

## 4 Additional quantitative results

## 4.1 Supplementary verification metrics

The main text defines the core metrics used in its figures and quantitative results, including CSIN, POD, CRPS, CRPSS, REV and PSD. Here we give additional details for the supplementary diagnostic metrics reported below.

False Alarm Ratio (FAR). FAR measures the fraction of forecast threshold exceedances that are not observed. Using the pooled threshold-contingency counts, it is defined as

$$
\mathrm { F A R } ( \rho ) = \frac { \mathrm { F P } } { \mathrm { T P } + \mathrm { F P } } ,\tag{S1}
$$

where lower values are better.

Neighbourhood Critical Success Index (CSIN). CSIN allows a spatial tolerance around threshold exceedances. Let $O _ { \rho } ( p ) = { \bf 1 } \{ y ( p ) \geq \rho \}$ and $F _ { \rho } ( p ) = \mathbf { 1 } \{ x ( p ) \geq \rho \}$ be the observed and forecast binary fields at grid cell $p ,$ and let $\textstyle { \mathcal { N } } _ { n } ( p )$ denote the centred n n neighbourhood of $p .$ We dilate both binary fields using

$$
O _ { \rho , n } ^ { \operatorname* { m a x } } ( p ) = \operatorname* { m a x } _ { q \in { \mathcal { N } } _ { n } ( p ) } O _ { \rho } ( q ) , \qquad F _ { \rho , n } ^ { \operatorname* { m a x } } ( p ) = \operatorname* { m a x } _ { q \in { \mathcal { N } } _ { n } ( p ) } F _ { \rho } ( q ) .\tag{S2}
$$

After pooling the contingency counts $\mathrm { T P } _ { n } , \mathrm { F P } _ { n }$ and $\mathrm { F N } _ { n }$ from these dilated fields over the test set, we compute

$$
\mathrm { C S I N } _ { n } ( \rho ) = \frac { \mathrm { T P } _ { n } } { \mathrm { T P } _ { n } + \mathrm { F P } _ { n } + \mathrm { F N } _ { n } } .\tag{S3}
$$

Values outside the domain are treated as non-events. In the tables and figures below, the subscript denotes the centred neighbourhood width in grid cells.

Fractions Skill Score (FSS). FSS [37] compares neighbourhood event fractions. For the same binary fields, define

$$
o _ { \rho , n } ( p ) = \frac { 1 } { n ^ { 2 } } \sum _ { q \in \mathcal { N } _ { n } ( p ) } O _ { \rho } ( q ) , \qquad f _ { \rho , n } ( p ) = \frac { 1 } { n ^ { 2 } } \sum _ { q \in \mathcal { N } _ { n } ( p ) } F _ { \rho } ( q ) .\tag{S4}
$$

The score for one forecast field is

$$
\mathrm { F S S } _ { n } ( \rho ) = 1 - \frac { \sum _ { p } \left[ f _ { \rho , n } ( p ) - o _ { \rho , n } ( p ) \right] ^ { 2 } } { \sum _ { p } f _ { \rho , n } ( p ) ^ { 2 } + \sum _ { p } o _ { \rho , n } ( p ) ^ { 2 } } .\tag{S5}
$$

Higher values are better and a perfect forecast has value 1. We compute FSS per sample and lead time and then average the valid scores for each reported threshold and neighbourhood window.

## 4.2 Regional quantitative diagnostics

To complement the lead-time curves in the main text, Supplementary Table 2 reports neighbourhood Critical Success Index with a $5 \times 5$ neighbourhood (CSIN ) at the three high rain-rate thresholds (16, 32 and 64 mm $\mathrm { h } ^ { - 1 } )$ . The table gives both representative lead-time values at T+2 h, T+4 h and T+6 h and the mean over all 36 lead times from T+10 min to T+6 h. These values are computed from the same regional evaluation data as the main-text high-threshold comparison and provide a compact numerical reading of the regional $\mathrm { C S I N _ { 5 } }$ curves. Across all regions and thresholds, MW-Nowcast retains the highest $\mathrm { C S I N _ { 5 } } ,$ with the advantage remaining clear at later lead times and heavier precipitation thresholds.

Supplementary Figs. 6–8 provide diagnostic views that complement the main-text CSIN<sub>5</sub> and CRPS results. Supplementary Fig. 6 shows that MW-Nowcast maintains substantially higher POD across regions and thresholds, while its FAR is generally lower than that of PySTEPS and comparable to that of NowcastNet. Although SimVPv2 often produces a lower FAR, this is accompanied by markedly lower POD, indicating more frequent missed events.

Supplementary Fig. 7 shows that the advantage of MW-Nowcast persists as the FSS neighbourhood broadens from $5 \times 5$ to $2 5 \times 2 5$ grid cells. From T+1 h onwards, it achieves the highest FSS for every evaluated region–threshold–neighbourhood combination, indicating that the improvement is not specific to the neighbourhood used in the main comparison.

Supplementary Fig. 8 extends the PSD and REV diagnostics to EU and CN. MW-Nowcast remains closer to the observed spectra at T+2 h, T+4 h and T+6 h, while retaining positive economic value over a broader range of cost–loss ratios than the baselines, particularly at the longer lead times.

![](images/59b430b53c000bd872c63faaed39d4c8c01b9de295321e8364d16d643cd088b7.jpg)  
Supplementary Fig. 6 Pointwise detection and false-alarm characteristics across regions. Rows show the US, EU and CN test sets, and columns show rain-rate thresholds of 16, 32 and 64 mm ${ \mathrm { h } } ^ { - 1 } . { \mathrm { a } } ,$ Probability of detection (POD). b, False alarm ratio (FAR). Both metrics are computed from pointwise threshold exceedances over lead times from T+10 min to T+6 h. Curves compare MW-Nowcast with NowcastNet, SimVPv2 and PySTEPS. Higher values are better for POD, whereas lower values are better for FAR.

![](images/22e27792b3d8ca66cc5d3d369632456a239bdc15bd997c3a573cfa3282ba16fe.jpg)  
Supplementary Fig. 7 FSS sensitivity to neighbourhood size across regions. a, US; b, EU; c, CN. Within each regional block, the upper row reports FSS for 5×5 and 11×11 grid-cell neighbourhoods, and the lower row reports FSS for $1 7 \times 1 7$ and 25 × 25 neighbourhoods. For each neighbourhood size, columns show rain-rate thresholds of 16, 32 and 64 mm $\mathrm { h } ^ { - 1 }$ . Curves compare MW-Nowcast with NowcastNet, SimVPv2 and PySTEPS over lead times from T+10 min to T+6 h. Higher values indicate better neighbourhood-scale forecast skill.

![](images/dd686643c5fa4f829a651c36f0b437e981f7c49ff180e7aebe42dc7ed48d22b6.jpg)  
Supplementary Fig. 8 Structural and relative economic value diagnostics for the EU and CN regions. The upper row shows the EU test set and the lower row shows the CN test set. a, Radially averaged power spectral density (PSD) at T+2 h, T+4 h and T+6 h. The observed spectrum is shown together with forecast spectra from MW-Nowcast, NowcastNet, SimVPv2 and PySTEPS; closer agreement with the observed spectrum indicates better preservation of multiscale precipitation structure. b, Standard pixel-wise relative economic value (REV) at the same lead times as a function of cost–loss ratio. Solid and dashed curves denote rain-rate thresholds of 16 and 32 mm $\mathrm { h } ^ { - 1 }$ , respectively. Positive REV indicates lower expected expense than the optimal climatological strategy, and higher values indicate greater economic value under the corresponding cost–loss setting.

## 5 Additional extended precipitation cases

The main text and appendix show representative examples from the US, EU and CN regions. Here we add two additional cases for each region and two additional US ablation examples. The regional cases broaden the qualitative coverage across storm types and geographic settings, whereas the ablation cases complement Supplementary Fig. 1 by showing how the comparison variants behave in additional US strong-precipitation settings.

## 5.1 Additional regional forecast examples

Supplementary Figs. 9–14 provide additional 6 h forecast examples. Each case follows the same five-panel layout as the case figures in the main text. a, Geographic and radar context, with the 256 km 256 km forecast domain outlined. b, Lead-time diagnostics comprising $\mathrm { C S I N _ { 5 } }$ at rain-rate thresholds of 32 and 64 mm $\mathrm { h } ^ { - 1 }$ , CRPS, CRPSS and radially averaged PSD at lead times of 3 and 6 h. c, Observed and predicted rain-rate fields from PySTEPS, SimVPv2, NowcastNet and MW-Nowcast at hourly lead times from 1 to 6 h. d, $\mathrm { C S I N _ { 5 } }$ and POD at thresholds of 32 and 64 mm $\mathrm { h } ^ { - 1 }$ for the four MW-Nowcast ensemble members and their PMM summary. e, The MW-Nowcast deterministic prediction and four ensemble members at lead times of 2, 4 and 6 h.

Supplementary Table 2 Regional $\mathbf { C S I N _ { 5 } }$ comparison of baseline models. Neighbourhood Critical Success Index with a $5 \times 5$ neighbourhood is reported at rain-rate thresholds of 16, 32 and 64 mm $\mathrm { h } ^ { - 1 }$ for representative lead times T+2 h, T+4 h and T+6 h, together with the mean over all 36 lead times from $\mathrm { T } + 1 0$ min to T+6 h. Values are computed from the same regional evaluation data as the main-text quantitative comparison. Higher is better; the best value within each region and column is shown in bold.
<table><tr><td rowspan="2">Region Model</td><td rowspan="2"></td><td colspan="3">T+2 h</td><td colspan="3"> $\mathrm { T } { + } 4 \ \mathrm { h }$ </td><td colspan="3"> $\mathrm { T } { + } 6 \ \mathrm { h }$ </td><td colspan="3">Mean</td></tr><tr><td>16</td><td>32</td><td>64</td><td>16</td><td>32</td><td>64</td><td>16</td><td>32</td><td>64</td><td>16</td><td>32</td><td>64</td></tr><tr><td rowspan="3">US</td><td>NowcastNet</td><td>0.192</td><td>0.144</td><td>0.070</td><td>0.075</td><td>0.037</td><td>0.004</td><td>0.038</td><td>0.016</td><td>0.002</td><td>0.171</td><td>0.129</td><td>0.064</td></tr><tr><td>SimVPv2</td><td>0.043</td><td>0.023</td><td>0.006</td><td>0.005</td><td>0.001</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.065</td><td>0.047</td><td>0.025</td></tr><tr><td>PySTEPS MW-Nowcast</td><td>0.089 0.284</td><td>0.043 0.214</td><td>0.022</td><td>0.037</td><td>0.017</td><td>0.009</td><td>0.026</td><td>0.014</td><td>0.007</td><td>0.099</td><td>0.064</td><td>0.045</td></tr><tr><td rowspan="3">EU</td><td></td><td></td><td></td><td>0.160</td><td>0.197</td><td>0.129</td><td>0.085</td><td>0.095</td><td>0.055</td><td>0.032</td><td>0.256</td><td>0.194</td><td>0.147</td></tr><tr><td>NowcastNet</td><td>0.171</td><td>0.123</td><td>0.087</td><td>0.081</td><td>0.051</td><td>0.028</td><td>0.036</td><td>0.022</td><td>0.011</td><td>0.158</td><td>0.118</td><td>0.086</td></tr><tr><td>SimVPv2</td><td>0.147</td><td>0.083</td><td>0.070</td><td>0.046</td><td>0.039</td><td>0.043</td><td>0.034</td><td>0.038</td><td>0.045</td><td>0.122</td><td>0.086</td><td>0.075</td></tr><tr><td rowspan="3"></td><td>PySTEPS MW-Nowcast</td><td>0.122 0.288</td><td>0.072 0.213</td><td>0.039 0.150</td><td>0.069 0.188</td><td>0.037 0.132</td><td>0.020 0.095</td><td>0.048</td><td>0.026</td><td>0.014 0.063</td><td>0.130</td><td>0.085</td><td>0.055</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.111</td><td>0.080</td><td></td><td>0.254</td><td>0.192</td><td>0.144</td></tr><tr><td>NowcastNet</td><td>0.184</td><td>0.128</td><td>0.067</td><td>0.074</td><td>0.034</td><td>0.004</td><td>0.034</td><td>0.012</td><td>0.002</td><td>0.166</td><td>0.118</td><td>0.067</td></tr><tr><td rowspan="3">CN</td><td>SimVPv2</td><td>0.109</td><td>0.022</td><td>0.002</td><td>0.014</td><td>0.001</td><td>0.000</td><td>0.001</td><td>0.000</td><td>0.000</td><td>0.095</td><td>0.042</td><td>0.020</td></tr><tr><td>PySTEPS</td><td>0.134</td><td>0.067</td><td>0.028</td><td>0.071</td><td>0.034</td><td>0.014</td><td>0.052</td><td>0.027</td><td>0.013</td><td>0.140</td><td>0.086</td><td>0.053</td></tr><tr><td>MW-Nowcast 0.293</td><td></td><td>0.197</td><td>0.129</td><td>0.195</td><td>0.123</td><td>0.077</td><td>0.112</td><td>0.063</td><td>0.033</td><td>0.264</td><td></td><td>0.184 0.130</td></tr></table>

a  
![](images/fccb18f5f9bf23100434b59896b218464577d6be3a74614fde45101704889269.jpg)

b  
![](images/bb645529b8393bce95c288384c1fe8caf86e4c3ce22c14e45beb9830469345de.jpg)

d  
![](images/9baa58fe238da760b55c23962c03cae15400edb5985725365a5fcff856881240.jpg)

![](images/ce7c1dc2629d26b729ec04264c3ad306370a629f4478342e3e003006293ce1fa.jpg)

![](images/924a19e58bf1836919b506b619498fd496fb8d4e331bd8140546742da1035ff1.jpg)

![](images/d5bf99b0cbffc74fc46943603704c5702ae3116ae6606e78eeb212458c226e36.jpg)

![](images/50a6b62e0f48cd61438c95a1fb4224ac45df982fc69e98b5b44871cea3d925d8.jpg)  
Supplementary Fig. 9 Convective heavy-precipitation forecast over the US Gulf Coast. The forecast is initialised at 05:30 UTC on 27 October 2025 for a patch centred near $3 0 . 5 0 ^ { \circ } \mathrm { N } , 8 6 . 3 3 ^ { \circ } \mathrm { W }$ . The observed sequence shows convective precipitation developing over the Gulf Coast region and generally expanding eastward to southeastward across the forecast patch. Rainfall intensity increases during the first half of the 6 h window, while the strongest precipitation is organised as multiple compact, discrete cores that merge, split and locally reorganise during the evolution.

![](images/2eb7bb88c118076adf18a50b47185336873960887abd419b82d1e04e6c0b9f99.jpg)

a  
![](images/d9378d80fe92702f4e8993592d0534f906262b3959b70ac199bd945da88a89a8.jpg)  
b

d  
![](images/3641809ff38a7f2c06b3ba23246ab1e6951f89f28ec73ec7ef712355c520bb46.jpg)

![](images/daeadca65a8fc0d2b9569ed3b81e8a3e9328c2ddf21fb8560282a6d22594e22a.jpg)

![](images/97816c7ae0cc7871622c28af91bd37ad282c5ace85b71f5c78a2004e5410f02a.jpg)

![](images/89610ec6a5c55d57b72c3b86cdca9e0f94f0069a70ac24c191e619156be55e8f.jpg)

![](images/4568ef25126bbe193f6532fa39b330347396ea850f1f9e9b515dca09df64582c.jpg)  
Supplementary Fig. 10 Convective heavy-precipitation forecast over the central US. The forecast is initialised at 06:00 UTC on 17 July 2025 for a patch centred near $3 8 . 1 7 ^ { \circ } \mathrm { N } , 9 6 . 9 3 ^ { \circ } \mathrm { W }$ . The observed sequence shows organised convective precipitation over the central US, with the rain area first advancing eastward to southeastward and then extending mainly towards the southern part of the patch. The high-intensity area broadens through the forecast window, but its embedded cores become increasingly fragmented, producing a multi-core structure rather than a single continuous rain band.

![](images/5c81cc6c205e96a51f18e9b2cb01bc6d1028d7dc46f570679fc3592e368a651b.jpg)

![](images/0bf157b0a066a077726b2fd39fa91a232f7a5fd8b4335e8dd17a883419f7e430.jpg)

![](images/f4ead2d8251476a7d01ec2f830d38337abbe998acf6dec531df62a48855face7.jpg)  
b

![](images/e8ccdbbbee01e9f4bd69c70057d472a8b1bd1900d11da6b0f2e4ff5d107e5f5c.jpg)

![](images/7cb34241c044a1b4b685d3e284749eea0edf1fd839315449cbf7f54a308e5bdb.jpg)

![](images/af9fcc133b315f3d1c1cd9e69f2cc6e1bbbda0658f9f27bbff4f0b29b550925e.jpg)

![](images/4a082d2dc653e24fc257e3c1f9f8f61d3044ff7c93b85bfb24a6bd6068bc2e52.jpg)

d  
![](images/2ffda185dbaeabff1ecfda7645728aa56f0773b62b3112278f9f1f49f827327d.jpg)

![](images/87af051616bbf54ae304870683a6e6a9d52bc7cb1748f865ec257d3c382b8d91.jpg)

![](images/faaea702cdb9317cce16e1f69da1bd7dcbb9704dfc3fa043440329163aa539dd.jpg)

![](images/e4988e0f778ebf10fa414b365a887003d2c1a252155997e4cb70a9df60ef0b4e.jpg)

![](images/e55f8661b8cf81c113e1bb9c17bd9b3714bf759e16bd9051d7deceac2013191b.jpg)  
Supplementary Fig. 11 Heavy-precipitation forecast over northern Italy. The forecast is initialised at 02:15 UTC on 7 July 2024 and is associated with a precipitation report near 45.95<sup>◦</sup>N, 9.09<sup>◦</sup>E in northern Italy. The observed sequence shows Alpine convective precipitation that remains concentrated near the main rain area rather than translating as a single coherent band. Localised intense echoes are repeatedly renewed and reorganised during the 6 h window, giving the case a compact but intermittently multi-core structure.

![](images/3d15bc433eecc222b0272de65d8ab58ad84637d55d23857cc013adab0bd8f481.jpg)

a  
![](images/c8c6f1d243d3ecd5147761a5157f2157a6ebc205480b1b7701f9560e974ada42.jpg)  
b

d  
![](images/916596f76d213c0214e85b73fdeb9f58cfcc9daafc52ede8ec497726bcf97d30.jpg)

![](images/a71db45e7b862fc30c8c67edfc90bab83ac04e581b476ca5f8c188c0525c1929.jpg)

![](images/4890d67ca6a6677a7394c75e4bdd2e580f6fc96fe61052cf935257de75f0e496.jpg)

![](images/984164413376eed9bcb5a041a87adc99b553d054b1e3b54ffff2521bd366ef2c.jpg)

![](images/f98b486609b7a3cfffab345a91015dc72e96a087bba26943fab2109a3fd4d183.jpg)  
Supplementary Fig. 12 Heavy-precipitation forecast over southern Germany. The forecast is initialised at 12:00 UTC on 3 June 2024 and is associated with a precipitation report near $4 7 . 6 2 ^ { \circ } \mathrm { N } ,$ , 11 $. 2 2 ^ { \circ } \mathrm { E }$ in southern Germany. The observed sequence shows organised heavy precipitation along the Alpine region, with the main precipitation area remaining relatively coherent through the forecast window. Intense echoes persist within this broader system, while local cores are renewed and rearranged along the rain area instead of separating into fully isolated cells.

![](images/9115ebc7525836d32167688e4f8879616f2520f468b967044611c46ef8e3c1a0.jpg)

a  
![](images/18bb9d65cf509227d991fb5ea8bfa1725fe6d635d3caa17114c32aacf0bc6180.jpg)

b  
![](images/50438be05fc78b3fbd02034184a921239aece39418d9a6a4658fd806362c9fba.jpg)

![](images/1418dc9bbe29c2417b923ef13b330e8bd89fa2729bb7cfaeb137a0e5c114c99b.jpg)

![](images/7c72c6e0d43e0d95d78d04604d59503aeab05db9fcad7c1c27ead09f76291f00.jpg)

![](images/3220ead0456e562337c741b2e067d209d82659fa9b2afd3499af0e9abfde49ea.jpg)

![](images/10161ea21bcc5650907c4960606f03292ec3a005221c3285a605cf3ceb432a91.jpg)

d  
![](images/a5139fa03fc0328d3d841b705c8aa9654b71689c023cbc276e4bc81b15dc25bc.jpg)

![](images/1374485ad6f3662485d4378e03c168b6bd53987fbb90369c044cfb0f201f62c0.jpg)

![](images/937c9ae703a3e4944013ac8c8e52c338fc6b96aa784fdf47538c918e7b20811a.jpg)

![](images/66b12f15c21a8df1a94b6e6fa306fbf398642504b6193e2f158a8b97749ba867.jpg)  
Supplementary Fig. 13 Heavy-precipitation forecast over southern China. The forecast is initialised at 18:00 UTC on 17 June 2026 for a patch centred near 21.93<sup>◦</sup>N, 112.05<sup>◦</sup>E. The observed sequence shows developing heavy precipitation over southern China, with the rain area expanding mainly towards the eastern to northeastern part of the patch. The intense precipitation area strengthens through the first half of the forecast window and then forms a broader, semi-continuous core region with smaller embedded maxima that reorganise locally.

a  
![](images/484b8ccef38b81e9846299d82458acad21a38d8512680d3727b8a34388ddea48.jpg)

b  
![](images/79737201da097892f4a510b0828fd160f01f948b67faf1f66ba2bb9b9d0115ad.jpg)

![](images/900e71039b81b3d85707a3413adb4d156f9853ea62b41cfbe5636fcd1c9eeee9.jpg)

![](images/70ab91433617980f47456bc63e596ec2f9942303d3733c5b0b14fa8580038994.jpg)

![](images/dc9f3b63d3bc0866a6561aac477cbb412274680fea59c718fc341d9561f6a090.jpg)  
d

![](images/9893e4cd02f3afdcd4d9f76a30aa444254bb02cb7fda874a1d2b8170e201bc92.jpg)

![](images/f30432728d22fa5d547f3eda32c2fb8476e3532a9acc9e45ab5851feac08f4c8.jpg)

![](images/49a47f65dfe88231a228375cc8ae279936440ee33dd46a895989dac38b62ab12.jpg)

![](images/c80e6ab056022f1e8ea3edb07ee113d967ae30efa6f402a2c406f194cea2fba0.jpg)

![](images/7330b5b4a3b3c866b6105fece343c90080206c7aea4628d39ab046c11c7b0ec0.jpg)

![](images/c61ba8609260644fafce8162fa7c9c2214ebcf0d06d32b7f81073619c982f4c6.jpg)  
Supplementary Fig. 14 Heavy-precipitation forecast over eastern China. The forecast is initialised at 11:10 UTC on 25 May 2026 for a patch centred near 31.77<sup>◦</sup>N, 117.87<sup>◦</sup>E. The observed sequence shows an organised precipitation system over eastern China that progresses eastward to southeastward across the forecast patch. The high-intensity precipitation expands and becomes more connected during the first half of the window, before the strongest region shifts downstream and breaks into several embedded cores at later lead times.

## 5.2 Additional ablation examples

To extend the single ablation example in Supplementary Fig. 1, Supplementary Figs. 15 and 16 show the same two US cases under the ablation variants. Each grid follows the same visualisation style as Supplementary Fig. 1: rows give the observation and the ablation variants, and columns give the input context and forecast lead times through T+6 h.

![](images/52fc29bcc3ce399a05b1b256381b16132eb74005b3a5ed0c24fc6724edd67697.jpg)  
Supplementary Fig. 15 Ablation forecast grid for the US strong-precipitation case initialised at 05:30 UTC on 27 October 2025. The patch is centred near 30.50 N, 86.33 W. Rows show the observation and the MW-Nowcast, Latent, OneStage, TwoStage, EDM-ODE and EDM-SDE variants; columns show the input context and forecast lead times through T+6 h.

![](images/48775e8ef54f3681986150b491f8bf71ecb971d516e81cfbff44f6b495b151a1.jpg)  
Supplementary Fig. 16 Ablation forecast grid for the US strong-precipitation case initialised at 06:00 UTC on 17 July 2025. The patch is centred near 38.17<sup>◦</sup>N, 96.93<sup>◦</sup>W. Layout follows Supplementary Fig. 15: rows give the observation and the MW-Nowcast, Latent, OneStage, TwoStage, EDM-ODE and EDM-SDE variants, and columns give the input context and forecast lead times through T+6 h.