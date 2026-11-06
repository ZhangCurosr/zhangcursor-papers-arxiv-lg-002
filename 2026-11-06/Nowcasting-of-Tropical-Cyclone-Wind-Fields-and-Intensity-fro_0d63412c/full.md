# Nowcasting of Tropical Cyclone Wind Fields and Intensity from Geostationary Imagery

Simon Donike<sup>1,2</sup> Pritthijit Nath<sup>1,3</sup> Cristina Radin<sup>1,4</sup>

Arthur Avenas<sup>1,5</sup> Ruben Cartuyvels<sup>1,5</sup> Emiliano Diaz<sup>1,2</sup> William K. Jones<sup>1,5</sup> Anna Jungbluth<sup>1,6</sup>

<sup>1</sup> Earth Systems Lab <sup>2</sup> University of Valencia <sup>3</sup> University of Cambridge <sup>4</sup> Max Planck Institute for Marine Microbiology, Bremen, Germany <sup>5</sup> European Space Agency <sup>6</sup> Imperial College London

## Abstract

High-resolution tropical cyclone surface wind speeds derived from synthetic aperture radar (SAR) observations can support storm structure and intensity modelling, but remain spatially and temporally sparse and unavailable near-real-time. Geostationary satellites capture cloud tops continuously in space and time, but do not directly resolve surface winds. Here, we present simple neural networks to predict 2D wind fields and storm-scale intensity metrics from geostationary imagery. Through a series of ablation studies, we find that joint prediction of 2D fields and scalar metrics from the model latent space improves the retrieval of the (radius of) maximum wind, especially for rapid intensification events.

## 1 Introduction

Tropical cyclones (TCs) are destructive natural hazards that form over the warm tropical oceans. Accurately predicting their intensity and evolution remains a challenge, especially during rapid intensification events (RI, at least 30 kt increase in 24 hours), which leave limited time for evacuation and disaster response [1]. While numerical weather prediction and data-driven forecasting have made progress in predicting TC trajectories [2], intensity estimates still need to be improved. Moreover, existing models largely focus on scalar outputs, such as maximum sustained wind speed $\left( V _ { \mathrm { m a x } } \right)$ , even though destructive potential is linked to the full wind field through integrated kinetic energy [3].

Spaceborne synthetic aperture radar (SAR) signal can penetrate cloud cover to provide surface wind speed estimates at kilometre resolution [4], but overpasses are infrequent, roughly twice a day across the TC lifetime [5], making it unsuitable for continuous monitoring. Geostationary (GEO) satellite observations, in contrast, provide near-continuous imagery that primarily captures the cloud-top structure, also at a high spatial resolution. GEO has long served as the predominant observational input for data-driven TC intensity estimation, from the original Dvorak technique [6, 7] to convolutional neural networks (CNNs) trained to predict maximum wind speed [8–10], category [11], or radius [12, 13], as well as RI forecasting [14–16]. Modelling of high-resolution spatial fields is less common, with previous approaches predicting aircraft-level winds [17] or downscaling coarse wind products with SAR-derived observations [18]. These methods, however, either target scalar intensity, reconstruct wind fields at flight level rather than the surface, or refine existing data sets.

To the best of our knowledge, no prior work has used only satellite-based observations as input to predict high-resolution SAR wind fields and storm-scale intensity metrics. Here, we show – in a nowcasting sertup – that estimating $V _ { \mathrm { m a x } }$ and wind-field structure during RI events benefits from jointly learning to reconstruct SAR-derived surface wind fields rather than predicting scalar quantities alone. We further show, through a series of ablation studies, that observation-only models are not meaningfully improved by adding coarse reanalysis products as input. With this, our work yields an observation-only workflow suitable for near-real-time downstream applications.

## 2 Data

TC tracks: IBTrACS [19] compiles historical best-track estimates, including location, intensity, and extent, for TCs. In this work, we use the following metrics: $V _ { \mathrm { m a x } }$ (maximum sustained wind speed), $R _ { \mathrm { m a x } }$ (radius of maximum wind), and $R _ { 3 4 } , R _ { 5 0 }$ , and $R _ { 6 4 }$ (radii of 34, 50, and 64 kt winds in each geographical quadrant).

SAR wind fields: Surface wind speed estimates from SAR provide target data for spatially resolved fields. These wind retrievals are based on variations in the SAR radar backscatter caused for instance by wind-driven capillary waves on the ocean surface. SAR data are sourced from the CyclObs database [5], which provides TC surface wind retrievals for a wide variety of SAR instruments at 3 km resolution.

Geostationary satellite imagery: Imagery from GOES-16 and -18 as well as Himawari-8 and -9 (see Appendix Figure A.1) provide infrared (IR) brightness temperature measurements across ten shared thermal IR channels at high spatial (2 km at nadir) and temporal (10 minute) resolution [20, 21].

Reanalysis data: As optional inputs, we use seven ERA5 [22] variables (precipitable water, seasurface temperature, mean sea-level pressure, 2 m air and dew-point temperature, and 10 m u- and v-wind) and derive 10 m wind speed and relative vorticity.

Data processing: We pair CyclObs 2D wind fields with spatially aligned GEO imagery and temporally interpolated IBTrACS metrics, using the closest GEO observation within ±90 minutes of each SAR acquisition to maximize the number of samples. In most cases, this matching results in GEO imagery within ±15 mins from the corresponding SAR observation.

GEO imagery is cropped to 196 × 196 pixels while varying the TC-centre position to reduce spatial bias. Storms are classified as no RI or RI based on 24-hour wind. We perform a stratified 60/20/20 train/validation/test split, keeping observations from the same storm together to prevent data leakage. This yields 841/232/212 samples from 115/34/38 storms, including 134 RI samples from 62 storms.

## 3 Models

![](images/155bafd44d66298c8bedccf4d651094a98fc03d4957b9783b216bed4f4139239.jpg)  
Figure 1: Overview of our modelling pipeline. GEO imagery, spatial context channels, and optionally ERA5 are translated into SAR-derived wind fields via a U-Net. An MLP can be added to the latent space, or after the 2D field output, to predict the maximum and radius of maximum wind $( V _ { \mathrm { m a x } } ,$ $R _ { \operatorname* { m a x } } )$ , as well as the radii of 34, 50, and 64 kt winds $( R _ { 3 4 } , R _ { 5 0 } , R _ { 6 4 } )$

Figure 1 shows the modelling pipeline. The input contains ten GEO channels and four spatial context channels: local-solar-time sine/cosine, solar zenith angle, and distance to storm centre. Experiments optionally add nine ERA5 channels, and a validity mask is appended to the resulting 14- or 23-channel input. Here, $I _ { \mathrm { G E O } }$ denotes the complete input tensor, with evaluation against $V _ { \mathrm { m a x } }$ and $R _ { \mathrm { m a x } }$

As a baseline, a CNN encoder computes $z = \mathrm { e n c } _ { \theta } ( I _ { \mathrm { G E O } } )$ without training against the 2D wind field. Here, an MLP maps global mean- and maximum-pooled features to $( \hat { V } _ { \operatorname* { m a x } } , \hat { R } _ { \operatorname* { m a x } } ) = \mathbf { M } \mathbf { L } \mathbf { P } _ { \phi } ( z )$ . We compare the baseline with three wind field models:

1. U-Net: A U-Net [23] predicts $\begin{array} { r } { \hat { I } _ { \mathrm { W i n d } } = \mathbf { d e c } _ { \eta } ( \mathbf { e n c } _ { \theta } ( I _ { \mathrm { G E O } } ) ) } \end{array}$ . With ERA5 input, the model predicts a residual added to the ERA5 field, otherwise, it predicts absolute wind. From these predicted wind fields, we derive $\hat { V } _ { \mathrm { m a x } }$ and $\hat { R } _ { \operatorname* { m a x } }$ and compare them with the IBTrACS metrics.

2. Post-hoc MLP: A two-stage model uses a frozen U-Net and predicts $( \hat { V } _ { \operatorname* { m a x } } , \hat { R } _ { \operatorname* { m a x } } ) =$ ML $\mathbf { P } _ { \zeta } ( \mathbf { U } { - } \mathbf { N e t } _ { \psi } ( \hat { I } _ { \mathrm { W i n d } } ) , m )$ , where the U-Net encodes the field, validity mask, and stormcentred distance, and m contains current-time and location metadata.

3. Latent MLP: A joint model shares the features $z = \mathrm { e n c } _ { \theta } ( I _ { \mathrm { G E O } } )$ between $\hat { I } _ { \mathrm { W i n d } } = \deg _ { \eta } ( z )$ and $( \hat { V } _ { \operatorname* { m a x } } , \hat { R } _ { \operatorname* { m a x } } ) = \mathbf { M } \mathbf { L } \mathbf { P } _ { \phi } ( z )$

All U-Nets are trained with loss ${ \mathcal { L } } _ { \mathrm { W i n d } } .$ , which upweights winds between 25 and $\mathrm { 5 0 m s ^ { - 1 } }$ on a scale from 1× to 8× and includes a 0.05-weighted peak of the top 0.5 percentile values. All MLPs are trained with $\mathcal { L } _ { V _ { \mathrm { m a x } } } + 0 . 2 5 ~ \mathcal { L } _ { R _ { \mathrm { m a x } } }$ , using storm- and intensity-balanced weights. We also compare scalar supervision of only $V _ { \mathrm { m a x } }$ and $R _ { \mathrm { m a x } }$ , with models additionally trained against $R _ { 3 4 } , R _ { 5 0 } , R _ { 6 4 }$ for more spatial nuance. SAR-derived wind fields are training targets only and are therefore not required at inference time. All objectives use Huber losses in physical units. The Huber transitions are $2 \mathrm { { \bar { m } s ^ { - 1 } } }$ $\mathrm { 5 m s ^ { - 1 } }$ , and 20 km for the wind field, $V _ { \mathrm { m a x } }$ , and $R _ { \mathrm { m a x } }$ , respectively. Further hyperparameter settings are provided in Appendix B.

Table 1: (a) $V _ { \mathrm { m a x } } ~ \mathrm { M A E } ~ ( \mathrm { m } \mathrm { s } ^ { - 1 } )$ on the held-out test set. The U-Net only predicts the wind field, all other models additionally predict $V _ { \mathrm { m a x } } , R _ { m a x } , R _ { 3 4 } , R _ { 5 0 } , R _ { 6 4 }$ . RI denotes rapid-intensification observations, bold values are best within each training regime. (b) MAE for the joint U-Net and latent MLP supervision ablations, trained to predict the wind fields and either just $V _ { \mathrm { m a x } }$ or also $R _ { m a x } , R _ { 3 4 }$ $R _ { 5 0 } , R _ { 6 4 } ^ { - } . \ V _ { \mathrm { m a x } }$ is reported in $\mathrm { m } \mathrm { s } ^ { - 1 }$ and $R _ { \mathrm { m a x } }$ in km. Note that these metrics were calculated over a reduced test set where the TC centre falls on a valid SAR pixel, as required for radii calculations, lowering the number of storms from 38 to 33. Bold values are the best in each column.

(a)
<table><tr><td>Model</td><td>ERA5</td><td>All</td><td>RI</td></tr><tr><td>U-Net</td><td>x</td><td>7.518</td><td>17.859</td></tr><tr><td>Post-hoc MLP</td><td>x</td><td>6.715</td><td>12.507</td></tr><tr><td>Latent MLP</td><td>x</td><td>5.885</td><td>8.934</td></tr><tr><td>U-Net</td><td>√</td><td>6.446</td><td>14.475</td></tr><tr><td>Post-hoc MLP</td><td>√</td><td>5.529</td><td>10.512</td></tr><tr><td>Latent MLP</td><td>√</td><td>7.302</td><td>7.716</td></tr></table>

(b)
<table><tr><td colspan="3"></td><td rowspan="2"> $V _ { \mathrm { m a x } }$ </td><td colspan="2"> $R _ { \mathrm { m a x } }$ </td></tr><tr><td></td><td>SAR Radius</td><td>All</td><td>RI</td><td>All RI</td></tr><tr><td>√</td><td>x</td><td>6.694</td><td>6.867</td><td></td><td></td></tr><tr><td>x</td><td>√</td><td>6.2977.198</td><td></td><td>19.4413.14</td><td></td></tr><tr><td>x</td><td>x</td><td>6.8286.645</td><td></td><td></td><td></td></tr><tr><td>√</td><td>√</td><td></td><td></td><td>6.344 5.46918.24 11.18</td><td></td></tr></table>

## 4 Results

We report $V _ { \mathrm { m a x } }$ and $R _ { \mathrm { m a x } }$ , both over all observations and RI.

Architecture and ERA5 ablations: Table 1a shows the comparison across different training strategies and model inputs. Without ERA5, the U-Net and latent MLP performs best, with a $V _ { \mathrm { m a x } } \mathbf { \bar { M } } \mathbf { A } \mathbf { E }$ of $5 . 8 8 5 \mathrm { m } \mathrm { { \dot { s } } ^ { - 1 } }$ , compared with 6.715 for the post-hoc MLP and $7 . 5 1 8 \mathrm { { m s } ^ { - 1 } }$ for the simple U-Net without a scalar-guided loss. It also performs best during RI (8.934 vs. 12.507 and $1 7 . { \overset { - } { 8 5 9 } } \operatorname { m } \operatorname { s } ^ { - 1 } )$ . ERA5 improves the U-Net and post-hoc MLP, and also reduces the error for RI with the latent MLP, although the error is not reduced when considering all storms.

Scalar supervision ablations: For the GEO-only latent MLP model (see Table 1b), combining SAR wind field reconstruction with radius supervision reduces RI $V _ { \mathrm { m a x } }$ MAE from 7.198 to 5.469 m s<sup>−1</sup> and RI $R _ { \mathrm { m a x } } \mathrm { M A E }$ from 13.14 to 11.18 km. Without radius supervision, SAR alone does not improve RI $V _ { \mathrm { m a x } } .$ , indicating that the spatial and scalar objectives are complementary.

Complete-storm case studies: Figure 2 compares the GEO-only U-Net and latent MLP models over the complete life cycles of Hurricane Humberto (North Atlantic, 2019), Kiko (East Pacific, 2025), and Otis (East Pacific, 2023). Across the 3,266 observations of these storms, the latent MLP reduces $V _ { \mathrm { m a x } }$ MAE from 9.197 to $7 . 4 3 4 \mathrm { { m s } ^ { - 1 } }$ relative to the simple U-Net, and from 12.203 to $5 . 4 4 1 \mathrm { m } \mathrm { s } ^ { - 1 }$ during RI.

Figure 3 shows a wind field reconstruction example at category 5 intensity. While the output is smoothed compared with the SAR-derived reference field, we correctly predict the extent of winds $\geq 3 0 \mathrm { m s ^ { - 1 } }$ and can fill in beyond the edge of the SAR swath.

![](images/a8125ba607b618abef1db5847408406f54f9cf4e78133d5fc266e8599fb61078.jpg)  
Figure 2: Complete-storm $V _ { \mathrm { m a x } }$ nowcasts from the latent MLP (model 3) and field-only U-Net (model 1), both without ERA5 input. Shading denotes RI periods. Curves use a centred five-hour rolling mean, while metrics use native predictions.

![](images/1d3c2882b0e053da2217b64b6eb0597370030f84819a18d7b1e7b535a4cc1911.jpg)  
Longitude (°)

(b) SAR wind field  
![](images/ce8023a4f2558de322fbe387dfbde9dd848246e6fee75ca417c5fa53a62c251a.jpg)  
Longitude (°)

![](images/e5080ccc5c2f29ce50057c498685567c95d5968c9ecb364b726b4a9a140b9c1d.jpg)  
Longitude (°)  
Figure 3: Wind field reconstruction of a category 5 storm. (a) Geostationary imagery (infrared channel 13), (b) target SAR-derived wind field, and (c) wind field prediction from the latent MLP model. Red crosses mark the IBTrACS storm centre.

## 5 Conclusion

In this work, we derive 2D wind fields and scalar metrics of TCs through their life cycles, including RI. Our best model jointly optimises a U-Net for field prediction with a latent MLP to predict storm-scale quantities such as the (radius of) maximum sustained wind. Jointly supervising the latent representation with SAR wind fields and storm-scale quantities is particularly beneficial during RI, when accurately representing fine-scale wind structure is most important. We show that both the wind fields and metrics can be derived from geostationary imagery alone.

While ERA5 conditioning provides useful environmental context and improves performance in some configurations, removing it moves the workflow away from reanalysis-based, non-observational inputs. Our resulting model requires only near-real-time GEO imagery and deterministic spatial context at inference, improving its availability and usability for operational monitoring and other downstream applications. Initial exploration of more complex, higher-parameter models [24–27] did not outperform our approach, largely due to the limited amount, quality, and balance of training data.

We therefore opted for simple ML approaches that allow for quick experimentation and easy ablation studies. Future work will explore alternative architectures, for instance based on vision transformers, which have shown great promise for geospatial analysis. Beyond limitations arising from the training data and the need for paired examples, we acknowledge several areas for future work. We have not yet conducted extensive ablation studies of the ERA5 inputs to determine whether alternative combinations of input channels could further improve performance, nor have we explored additional hyperparameter tuning, including the choice of loss function. Additional geophysical variables, e.g., describing the upper ocean conditions along the forecasted track, may further improve our estimates. Nevertheless, the ability to recover both spatial wind structure and storm-scale intensity metrics from GEO imagery alone provides a promising foundation for observation-driven TC monitoring and potential future integration into near-real-time monitoring workflows.

## Acknowledgements and Disclosure of Funding

This work has been enabled by Frontier Development Lab Earth Systems Lab (https://eslab.ai/) a public-private partnership between the European Space Agency (ESA), Trillium Technologies and University of Oxford and leaders in commercial AI supported by Scan Computers, ESA Space HPC, UK AI Research Resource (AIRR), JASMIN, Google Cloud and NVIDIA Corporation.

## References

[1] Kaplan J, DeMaria M. Large-scale characteristics of rapidly intensifying tropical cyclones in the North Atlantic basin. Weather and Forecasting. 2003;18(6):1093-108. Available from: https://journals. ametsoc.org/view/journals/wefo/18/6/1520-0434\_2003\_018\_1093\_lcorit\_2\_0\_co\_2.xml.

[2] Cangialosi JP, Blake E, DeMaria M, Penny A, Latto A, Rappaport E, et al. Recent progress in tropical cyclone intensity forecasting at the National Hurricane Center. Weather and Forecasting. 2020;35(5):1913- 22. Available from: https://doi.org/10.1175/WAF-D-20-0059.1.

[3] Powell MD, Reinhold TA. Tropical cyclone destructive potential by integrated kinetic energy. Bulletin of the American Meteorological Society. 2007;88(4):513-26. Available from: https://doi.org/10. 1175/BAMS-88-4-513.

[4] Mouche A, Chapron B, Knaff J, Zhao Y, Zhang B, Combot C. Copolarized and cross-polarized SAR measurements for high-resolution description of major hurricane wind structures: Application to Irma category 5 hurricane. Journal of Geophysical Research: Oceans. 2019;124(6):3905-22. Available from: https://doi.org/10.1029/2019JC015056.

[5] Avenas A, Tsukada T, Cévaër T, Knaff J, Chapron B, Mouche A. The Global Kilometric Tropical Cyclone Wind Database (CyclObs) Enabling High-Resolution Data-Driven Studies. ESS Open Archive. 2025;2025(1226). Available from: https://doi.org/10.22541/essoar.176676860.01253431/v1.

[6] Dvorak VF. Tropical Cyclone Intensity Analysis and Forecasting from Satellite Imagery. Monthly Weather Review. 1975;103(5):420-30. Available from: https://journals.ametsoc.org/view/journals/ mwre/103/5/1520-0493\_1975\_103\_0420\_tciaaf\_2\_0\_co\_2.xml.

[7] Dvorak VF. Tropical cyclone intensity analysis using satellite data. United States, National Environmental Satellite, Data, and Information Service; 1984. Available from: https://repository.library.noaa. gov/view/noaa/19322.

[8] Chen B, Chen BF, Lin HT. Rotation-blended CNNs on a New Open Dataset for Tropical Cyclone Image-to-intensity Regression. In: Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. Association for Computing Machinery; 2018. p. 90-9. Available from: https://doi.org/10.1145/3219819.3219926.

[9] Maskey M, Ramachandran R, Ramasubramanian M, Gurung I, Freitag B, Kaulfus A, et al. Deepti: Deep-Learning-Based Tropical Cyclone Intensity Estimation System. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing. 2020;13:4271-81. Available from: https://doi.org/ 10.1109/JSTARS.2020.3011907.

[10] Wang C, Zheng G, Li X, Xu Q, Liu B, Zhang J. Tropical Cyclone Intensity Estimation From Geostationary Satellite Imagery Using Deep Convolutional Neural Networks. IEEE Transactions on Geoscience and Remote Sensing. 2022;60:1-16. Available from: https://doi.org/10.1109/TGRS.2021.3066299.

[11] Pradhan R, Aygun RS, Maskey M, Ramachandran R, Cecil DJ. Tropical Cyclone Intensity Estimation Using a Deep Convolutional Neural Network. IEEE Transactions on Image Processing. 2018;27(2):692-702. Available from: https://doi.org/10.1109/TIP.2017.2766358.

[12] Zhuo JY, Tan ZM. Physics-Augmented Deep Learning to Improve Tropical Cyclone Intensity and Size Estimation from Satellite Imagery. Monthly Weather Review. 2021;149(7):2097-113. Available from: https://doi.org/10.1175/MWR-D-20-0333.1.

[13] Wang C, Li X. Physically Consistent Tropical Cyclone Monitoring From Infrared Satellite Imagery via Knowledge-Guided Deep Learning. Journal of Geophysical Research: Machine Learning and Computation. 2026;3(1):e2025JH000875. Available from: https://doi.org/10.1029/2025JH000875.

[14] Griffin SM, Wimmers A, Velden CS. Predicting Rapid Intensification in North Atlantic and Eastern North Pacific Tropical Cyclones Using a Convolutional Neural Network. Weather and Forecasting. 2022;37(8):1333-55. Available from: https://doi.org/10.1175/WAF-D-21-0194.1.

[15] Chen BF, Kuo YT, Huang TS. A deep learning ensemble approach for predicting tropical cyclone rapid intensification. Atmospheric Science Letters. 2023;24(5):e1151. Available from: https://doi.org/10. 1002/asl.1151.

[16] Wang C, Yang N, Li X. Advancing forecasting capabilities: A contrastive learning model for forecasting tropical cyclone rapid intensification. Proceedings of the National Academy of Sciences. 2025;122(4):e2415501122. Available from: https://doi.org/10.1073/pnas.2415501122.

[17] Wimmers AJ, Griffin S, Velden C. A U-Net Retrieval of Tropical Cyclone Inner-Core Wind Fields from Microwave and Infrared Satellite Imagery. Artificial Intelligence for the Earth Systems. 2024;3(4):e230084. Available from: https://doi.org/10.1175/AIES-D-23-0084.1.

[18] Zhang E, Su H, Chan PW, Zhai C, Zhou W, Xu W, et al. Deep Learning-Based MultiSource Satellite Data Fusion and Downscaling for Tropical Cyclone Wind Fields. Journal of Geophysical Research: Machine Learning and Computation. 2025;2(4):e2025JH000792. Available from: https://doi.org/10.1029/ 2025JH000792.

[19] Knapp KR, Kruk MC, Levinson DH, Diamond HJ, Neumann CJ. The International Best Track Archive for Climate Stewardship (IBTrACS): Unifying Tropical Cyclone Data. Bulletin of the American Meteorological Society. 2010;91(3):363 376. Available from: https://doi.org/10.1175/2009BAMS2755.1.

[20] Schmit TJ, Griffith P, Gunshor MM, Daniels JM, Goodman SJ, Lebair WJ. A closer look at the ABI on th GOES-R series. Bulletin of the American Meteorological Society. 2017;98(4):681-98.

[21] Bessho K, Date K, Hayashi M, Ikeda A, Imai T, Inoue H, et al. An introduction to Himawari-8/9—Japan’s new-generation geostationary meteorological satellites. Journal of the Meteorological Society of Japan Ser II. 2016;94(2):151-83.

[22] Hersbach H, Bell B, Berrisford P, Hirahara S, Horányi A, Muñoz-Sabater J, et al. The ERA5 Global Reanalysis. Quarterly Journal of the Royal Meteorological Society. 2020;146(730):1999-2049. Available from: https://doi.org/10.1002/qj.3803.

[23] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer; 2015. p. 234-41. Available from: https://doi.org/10.1007/978-3-319-24574-4\_28.

[24] Chen LC, Papandreou G, Kokkinos I, Murphy K, Yuille AL. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. IEEE Transactions on Pattern Analysis and Machine Intelligence. 2018;40(4):834-48. Available from: https://doi.org/10. 1109/TPAMI.2017.2699184.

[25] Zhou Z, Siddiquee MMR, Tajbakhsh N, Liang J. UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation. IEEE Transactions on Medical Imaging. 2020;39(6):1856-67. Available from: https://doi.org/10.1109/TMI.2019.2959609.

[26] Xie E, Wang W, Yu Z, Anandkumar A, Alvarez JM, Luo P. SegFormer: simple and efficient design for semantic segmentation with transformers. In: Proceedings of the 35th International Conference on Neural Information Processing Systems. NIPS ’21. Red Hook, NY, USA: Curran Associates Inc.; 2021. Available from: https://proceedings.neurips.cc/paper/2021/hash/ 64f1f27bf1b4ec22924fd0acb550c235-Abstract.html.

[27] Dauvilliers C, Monteleoni C. MoTiF: a self-supervised model for multi-source forecasting with application to tropical cyclones. Environmental Data Science. 2025;4. Available from: https://doi.org/10.1017/ eds.2025.10014.

## Appendix A Geostationary Field-of-Views

![](images/74542d4f0bc12038016f53133512f95aaaa88eb35ef32a41f30a02633d367ca9.jpg)  
Figure A.1: Field-of-view of the geostationary satellites used in this work. Future work will expand our approach to also include coverage from the Meteosat Second Generation (MSG) satellite to complete coverage over the Indian Ocean.

## Appendix B Hyperparameters

We use four-level residual U-Nets with channel widths (32, 64, 128, 256), GroupNorm, SiLU activations, and skip connections, operating on 192 × 192 centre crops. The MLP has hidden widths 128 and 64 with dropout 0.1, and no decoder is used. All models use AdamW with weight decay $1 0 ^ { - 4 }$ U-Net and encoder-only models use batch size 4 and learning rate $2 \times 1 0 ^ { - 4 }$ , while the two-stage model uses batch size 16 and learning rate $3 \times 1 0 ^ { - 4 }$ . We reduce the learning rate on validation plateaus and use early stopping with patience 50.