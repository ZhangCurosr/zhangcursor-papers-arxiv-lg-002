# Observation Assimilation Transformer Probabilistic Weather Analysis from Sparse Station Observations

Christian Donner<sup>⋆,1</sup>, Michele Cattaneo<sup>⋆,1</sup>, Maxim Samarin<sup>1</sup>, Francesco Zanetta<sup>2</sup>, Ic´ıar Llor´ens Jover<sup>2</sup>, Gabriela Aznar Siguan<sup>2</sup>, Michele Volpi<sup>1</sup>

<sup>1</sup> Swiss Data Science Center, Eidgen¨ossische Technische Hochschule Z¨urich (ETHZ) <sup>2</sup> Federal Ofice of Meteorology and Climatology MeteoSwiss, Switzerland <sup>⋆</sup> contributed equally

October 21, 2026

## Abstract

We present the Observation Assimilation Transformer (OAT), a geometry-aware neural process that corrects and refines a prior numerical weather prediction (NWP) field into a highresolution posterior field by assimilating sparse observations. OAT is an encoder-decoder transformer that considers a set of observed locations and their NWP prior to condition predictions at a set of arbitrary unobserved query locations. Its sparse attention mechanism makes it scalable to thousands of station observations and depends on relative positions between locations, never on the absolute coordinates. The model allows for partially missing observations, predictions at arbitrary continuous locations without grid projection, and outputs are parametric distributions. A station-randomized training algorithm is developed with continuous ranked probability score (CRPS) loss. We test OAT on two datasets – one at regional scale with hundreds of stations, and one at continental scale with thousands of stations. On these datasets OAT outperforms several baselines and shows well calibrated predictions. In an illustrative case study, OAT’s downscaling capabilities are demonstrated by downscaling ERA5 (∼ 31km) to COSMO (1km), where OAT achieves lower mean absolute error (MAE) and CRPS for temperature and precipitation on test stations compared to COSMO.

## 1 Introduction

Numerical weather prediction (NWP) models solve the governing equations of the atmospheric motions on high-performance computing (HPC) systems [Bauer et al., 2015], but their utility is fundamentally limited by the spatial resolution at which these equations can be practically solved. Their spatial and temporal discretization introduces errors and systematic biases through imperfect modelling of processes occurring at scales smaller than the model grid. Additionally, in the presence of complex topography, the spatial resolution of the NWP grid may be insuficient to accurately represent the underlying terrain over which the equations are solved. While increasing the horizontal spatial resolution can reduce some of these errors, doing so comes at a substantial increase of computational cost. Fully data-driven alternatives to NWP models are currently achieving remarkable results at a fraction of the inference cost, often matching or surpassing NWP models on specific metrics. However their trustworthiness and ability to model extreme events is still, at the time of writing, an open question [Bonavita, 2024]. A natural approach to address these limitations, is to reconcile NWP output with observations, which, despite being afected by measurement errors, are the best available estimate of the true atmospheric state.

Statistical postprocessing [Rasp and Lerch, 2018] addresses the issue by learning a distribution that corrects systematic errors of raw NWP ensemble forecasts at station locations, conditioned on an NWP ensemble, geographic location and a set of predictors features derived from topography. Observations serve as training target, and if the learned model does not rely on station-specific embeddings, it can be transferred to arbitrary locations, under the assumption that topographic features alone are suficient to describe the local processes. However, statistical postprocessing treats prediction locations independently, where neighboring stations are not used to inform predictions at nearby or related locations. Furthermore, observations are only used as training targets, meaning the model can be applied before any observation is available but fails to assimilate the actual realization of the target process at inference time. Statistical downscaling [Rampal et al., 2024, Mardani et al., 2025] addresses a complementary limitation; rather than correcting errors at station locations, it learns a mapping from a coarse resolution field, to fine-scale predictions by targeting high-resolution gridded references. This may or may not include coupled past observations [Vaughan et al., 2022, Bruinsma et al., 2023, Sousa et al., 2026]. However, like statistical postprocessing, downscaling typically does not assimilate observations at inference time. In contrast, data assimilation [Carrassi et al., 2018] explicitly combines model forecasts with newly available observations to refine the estimated atmospheric state at any given time, and is an integral part of the NWP pipeline. These approaches are complementary; statistical postprocessing focuses on correcting biases, statistical downscaling primarily enhances spatial resolution and data assimilation incorporates observations at inference time.

We aim to combine elements from each paradigm into a data-driven framework that can correct NWP biases, generate higher resolution fields, and condition probabilistic predictions on observations. To this end, we present the Observation Assimilation Transformer (OAT), a geometry-aware neural process combining:

• NWP & Topography informed predictions, embedding both sources into a (learned) latent vector.

• Observation assimilated predictions via a dedicated transformer architecture.

• No absolute positions; attention only depends on relative geometry similar to Ashman et al. [2024a].

• Scalability to thousands of stations via kNN-sparsified attention.

• Embedding partial observations via a neural gating mechanism.

• A randomized training algorithm using station observations as target.

After discussing related work in Section 2, we introduce the architecture and training procedure of OAT in Section 3. Then we introduce two datasets, one focusing on Switzerland and one whole Europe in Section 4. We show OAT’s capabilities on these datasets and we compare it against several data-driven baselines, analyze the performance of OAT under diferent factors, show a downscaling example and the calibration of OAT to provide predictive distribution. Section 5 concludes discussing limitations and future work. All code is made available at the Swiss Data Science Center Github.

## 2 Related Work

## 2.1 Gaussian Processes and Data Assimilation Approaches

For decades, researchers attempt to incorporate sparse observations into NWP fields to correct and downscale them. Traditionally, statistical methods such as linear correction models [B¨urger, 1996] and kriging methods [Biau et al., 1999] are used to estimate a correction function of the field by regression. Today machine learning methods are on the forefront of data-driven downscaling and data-assimilation. Zanetta et al. [2026] use Gaussian Processes (GPs) for downscaling NWP wind fields, assimilating station observations. The GP prior’s mean is parametrized by a neural network, parsing NWP products and topography features. The authors then propose to use deep kernel learning [Wilson et al., 2016] to condition the GP posterior on station observation.

## 2.2 ML models for combining NWP products with spatial observations

There have been various machine learning methods proposed for combining observational station data with gridded products. Miralles et al. [2026] use graph neural networks (GNN) [Deferrard et al., 2016, Kipf and Welling, 2016] for improving nowcasting. Yang et al. [2024] use transformer and GNN architectures for forecasting at station level. While our work focuses more on spatial aggregation, it combines the transformer and GNN perspective of both works. We will rely on the transformer architecture, but can also view the proposed aggregation as belonging to the message-passing framework [Gilmer et al., 2017] in GNNs. MetNet-3 [Andrychowicz et al., 2023] combines gridded numerical, gridded observational, and sparse observational station inputs in a U-Net architecture trained to predict three types of targets, among which the sparse station input data densified onto a 4 km × 4 km grid. Statistical postprocessing methods such as the classic EMOS [Gneiting et al., 2005] or more recent deep learning approaches [Rasp and Lerch, 2018, Zanetta et al., 2023] correct NWP predictions at station locations using relationships learned from historical observational data, but do not condition on real-time observations at inference time.

## 2.3 Neural processes

An interesting model class that tries to address the scalability limitations of GPs are Neural Processes [Garnelo et al., 2018] (NPs), but they come with arguably less consistent conditioning properties. For eficient processing of spatial data, convolutional neural processes [Gordon et al., 2019] have been proposed and have also been applied to weather forecasting [Vaughan et al., 2022]. In this work they also combine gridded NWP products with station data to predict temperature and precipitation at unseen locations. However, despite the elegant model construction that enables prediction anywhere, the main computation efectively happens at the resolution of the gridded data.

To assimilate sparse observations without gridded data, Kim et al. [2019] developed the attentive neural processes. This model uses cross-attention to aggregate observations into predictions at the locations of interest. A problem with attention is that computation and memory scale quadratically with the number of observations. Hence, Bao et al. [2024] extended the model to spatial multi-attention conditional neural processes (SMACNP) to specifically update spatial information and uncertainty via diferent cross-attention channels. Dumitrescu [2026] have adopted the model for assimilation of daily weather variables, aggregating station observations and topography for Romania.

However, only relying on cross-attention, i.e. each query location aggregates over all observations, neglects certain information that the set of observations can provide. For example, if two observations contradict each other, the model should weigh them accordingly. This can be handled by the transformer neural process (TNP) [Nguyen and Grover, 2022], exploiting self-attention between the observed points to update the information flow, before incorporating it into the crossattention step. Yet, in the standard formulation, the attention uses the absolute positions to learn how locations are related. This is undesirable in weather prediction, because it could lead to memorizing specific frequent location-dependent patterns, without learning broader patterns that generalize across locations. To address this, researchers developed translation equivariant TNP (TE-TNP) [Ashman et al., 2024a] and its extensions [Ashman et al., 2024b]. Furthermore, Ashman et al. [2025] addresses the scalability issue with a gridded TNP (although not translation equivariant) by encoding all observations (gridded and unstructured) on a regular grid, which then allows using Swin Transformers [Liu et al., 2021] to learn eficiently global relations.

In conclusion, while significant efort has been made to develop ML models for data aggregation, there is no method that covers the full spectrum: a model that (i) aggregates information from NWP products, topography, and sparse observations, (ii) is scalable to thousands of station observations, (iii) and does not rely on absolute position (see Table 1). Our work closes this gap.

<table><tr><td></td><td>NWP prior</td><td>Geo-attn</td><td>Station SA</td><td>Pred. anywhere</td><td>Scalable</td></tr><tr><td>(Deep Kernel) GP [Zanetta et al., 2026]</td><td>√</td><td>~</td><td>√</td><td>√</td><td>2</td></tr><tr><td>ConvCNP [Gordon et al., 2019]</td><td>√</td><td>x</td><td>x</td><td>√</td><td>√</td></tr><tr><td>TE-TNP [Ashman et al., 2024a,b]</td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td></tr><tr><td>Gridded TNP [Ashman et al., 2025]</td><td>X</td><td>x</td><td>2</td><td>√</td><td>√</td></tr><tr><td>SMACNP [Bao et al., 2024]</td><td>x</td><td>~</td><td>x</td><td>√</td><td>√</td></tr><tr><td>OAT (ours)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 1: Structural comparison of OAT with closely related models. ✓ = present; ✗ = absent; $\sim =$ partial.

## 3 Methodology

## 3.1 Problem Formulation

At a given time t we have a low resolution NWP field $U ^ { \mathrm { N W P } } ~ \in ~ \mathbb { R } ^ { N _ { U } \times C _ { U } }$ , where $C _ { U }$ are the variables of the field and $N _ { U }$ is the number of coordinates, where the field is given. Note that we explicitly allow for irregular non-gridded fields. Positions and observations for the N stations in the dataset are denoted by $\mathbf { x } ^ { c } \in \mathbb { R } ^ { N \times 2 }$ and $\mathbf { y } ^ { c } \in \mathbb { R } ^ { N \times C _ { y } }$ , respectively. We allow for missing values in the observations, either at the channel (variable) level, or at the station level, where the entire observation vector is unavailable. We define a binary mask $\mathbf { m } ^ { y } \in \{ 0 , 1 \} ^ { N \times C _ { y } }$ , where 1 and 0 denote a valid and invalid observation, respectively. Additionally, we have high-resolution topographic data $T \in \mathbb { R } ^ { N _ { T } \times C _ { T } }$ , which as for the NWP can be an irregular field. We define all the given information in the set $\mathcal { D } _ { t }$ at a given time t. The aim is to predict $p _ { \phi } ( \mathbf { y } _ { t } ^ { \star } | \mathbf { x } ^ { \star } , \mathcal { D } _ { t } )$ , an informed distribution over the target variables $\mathbf { y } ^ { \star }$ given any set of points $\mathbf { x } ^ { \star }$ on the considered domain. The distribution is parametrized by $\phi .$ An example snapshot of the data can be seen in Figure 1 in the left panel, while the right panel illustrates the general architecture of OAT.

## 3.2 Observation Assimilation Transformer

Predictions without observations First, we construct a model, akin to classical postprocessing [Rasp and Lerch, 2018], that allows predicting anywhere without observations, i.e. $p ( \mathbf { y } _ { t } ^ { \star } | \mathbf { x } ^ { \star } , \mathcal { D } _ { t } \backslash \{ \mathbf { y } _ { t } ^ { c } , \mathbf { x } ^ { c } \} )$

![](images/6ecac19aa874d6606b32588a9e4651bfcd28ae2263841ceaca76bae9b6fd2eb1.jpg)  
Figure 1: Schematic of data and observation assimilation transformer (OAT). Left: Data snapshot with high-resolution topography, coarse NWP field, station locations (orange circles) and illustrative query locations (blue diamonds). Query locations can lie on a grid or at arbitrary locations. Right: Schematic of the OAT architecture. Topography and NWP fields are first embedded at observed and query locations in the Pre-OAT stage. Then observed embeddings are updated with observation values resulting in the informed station tokens. The OAT block comprises a selfattention encoder on the observed embeddings, and a cross-attention decoder between the observed embeddings and query embeddings. The query embeddings, informed by the observational context, are then producing a probabilistic prediction via an output head to produce the final output distributions. The same shared output head can provide predictions on query tokens before going through the OAT block.

(see Pre-OAT block in Figure 1). We do that for any point of interest i at position $\mathbf { x } _ { i } ^ { \star }$ . We construct a joint non-linear embedding $\mathbf { z } _ { t , i } = e ( U _ { t , i } ^ { \mathrm { { N W P } } } , T _ { i } , t )$ containing all the information we have at any location i and time t. $U _ { t , i } ^ { \mathrm { { N W P } } }$ and $T _ { i }$ are obtained by interpolation at the points of interest given the field information. Both are embedded with separate multi-layer perceptrons (MLP), and then conditionally fused to a joint embedding $\mathbf { z } _ { t , i } \in \mathbb { R } ^ { D _ { \mathrm { e m b } } }$ . To this end, we use adaptive layer norm, where the shift- and scale-parameters are conditioned on the features extracted from time t, i.e. sine and cosine embeddings of day of the year and time of the day (see Appendix B). $D _ { \mathrm { e m b } }$ is a crucial hyperparameter and indicates the dimensionality of the latent space. Note that we do not provide the absolute coordinates $\mathbf { x } ^ { \star }$ themselves such that the model can also be used outside of the training domain and does not overfit to coordinate particularities. Finally, we can use an output head $\phi _ { t , i } = o ( \mathbf { z } _ { t , i } )$ mapping to the parameters of the distributions of the target variables, allowing us to make predictions without station observations $\mathbf { y } _ { t }$ . We call this distribution without any information from station observations pre-OAT distribution.

Observation assimilation at station locations With OAT, we aim at creating an observationinformed embedding at the station locations ${ \mathbf x } ^ { c } ,$ where we have observations $\mathbf { y } ^ { c }$ . To this end, we simply fuse the latent embedding z with the observations $\mathbf { y } ^ { c }$ via another adaptive layer norm.

The fusion layer $\tilde { \mathbf { z } } _ { y } = \mathrm { F u s e } ( \mathbf { z } , \mathbf { y } , t )$ , conditioned on time t, incorporates the observation into a preliminary informed embedding. If an observation is missing, the observation is set to 0. To control the information flow based on which values are observed, we design a learnable gate that depends on the mask of observed values $\begin{array} { r } { \mathbf { m } _ { t } ^ { y } ; } \end{array}$ entries are set to 1 for each location and variable that was observed and 0 otherwise. This leads to

$$
\mathbf { z } _ { y } = ( 1 - g ( \mathbf { m } ^ { y } ) ) \mathbf { z } + g ( \mathbf { m } ^ { y } ) \tilde { \mathbf { z } } _ { y } ,\tag{1}
$$

where $g : \{ 0 , 1 \} ^ { N \times C _ { y } }  [ 0 , 1 ] ^ { N \times D }$ emb is a learnable MLP. This way, the gate is able to funnel the information flow per dimension of the embeddings.

Observation assimilation via Transformer Next, we want to propagate the information obtained at the station locations $\mathbf { x } ^ { c }$ to our prediction locations $\mathbf { x } ^ { \star }$ . At the former, we have the informed embedding $\mathbf { z } _ { y }$ , while at the latter the uninformed ones $\mathbf { z } ^ { \star }$ . We use a customized transformer, where we have queries $\mathbf { z } ^ { \star }$ and keys and values $\mathbf { z } _ { y }$ as input. Compared to the standard encoder-decoder transformer [Vaswani et al., 2017], we remove the self-attention among queries in the decoder blocks to avoid that other queries influence the prediction at a given position. Also, positional encoding is discarded. Instead, in the spirit of the translation equivariant transformer, the attention logits between two locations $i , j$ are given by

$$
\alpha _ { i , j } = \left\{ \begin{array} { l l } { Q _ { i } ^ { \top } K _ { j } + b _ { \mathrm { a t t n } } ( \mathbf x _ { i } ^ { \mathrm { e l e v } } - \mathbf x _ { j } ^ { \mathrm { e l e v } } ) } & { \mathrm { ~ i f ~ } j \in \mathcal { N } _ { i } , } \\ { - \infty } & { \mathrm { ~ o t h e r w i s e ~ } , } \end{array} \right.\tag{2}
$$

where $Q = W _ { Q } \mathbf { z } ^ { \star }$ are the queries, $K = W _ { K } \mathbf { z } _ { y }$ are the keys, ${ \mathcal { N } } _ { i }$ is a neighborhood set for i. $\mathbf { x } ^ { \mathrm { e l e v } }$ denotes the location concatenated with the elevation at this position. This attention mechanism difers from standard attention in two ways: First, the attention bias depends on the signed distance between locations, allowing the model to incorporate distance and direction information in the attention. Second, sparse attention allows us to define a graph between stations in the transformer encoder, and stations as well as prediction points in the decoder, respectively. This allows for scalability as the number of stations grows, and the receptive field can be enlarged by adding layers. How to define the neighborhood sets and therefore the graph is a design choice. Simple options include k-nearest neighbor construction or a radius graph, though more sophisticated approaches such as terrain-informed graphs are also possible. For details and alternative attention mechanisms see Appendix B.

Informed predictions anywhere Since OAT allows us to have informed embeddings $\mathbf { z } _ { y } ^ { \star }$ anywhere, predictions at arbitrary locations, such as on a high-resolution grid, become straightforward. We use the same output head that we used already for the pre-OAT output to have the final distribution parameters $\boldsymbol { \phi } _ { y } ^ { \star } = \boldsymbol { o } ( \mathbf { z } _ { y } ^ { \star } )$ , which yields the desired distribution $p _ { \phi ( \mathbf { z } _ { u } ^ { \star } ) } ( \mathbf { y } _ { t } ^ { \star } | \mathbf { x } ^ { \star } , \mathcal { D } _ { t } )$ .

Model training At each training step, the observed locations $\mathbf { x } ^ { c }$ are randomly partitioned into a conditioning set, whose observations are provided to the model, and a holdout set, whose observations are withheld and used only as prediction targets to supervise the model. We use the continuous ranked probability score (CRPS) on each observed variable as the loss function. To encourage best possible predictions even without observational data, we also add a loss term for the pre-OAT predictions. The total loss is given by

$$
\ell ( \mathbf { y } ^ { \star } ) = ( 1 - w _ { \ell } ) \ell _ { \phi _ { y } ^ { \star } } ( \mathbf { y } ^ { \star } ) + w _ { \ell } \ell _ { \phi ^ { \star } } ( \mathbf { y } ^ { \star } ) ,\tag{3}
$$

which is diferentiable with respect to all model parameters, where $\mathbf { y } ^ { \star }$ is given during training. We set the weight $w _ { \ell } = 0 . 1$ throughout this work. Pseudo-code of the training procedure is provided in Algorithm 1. For further implementation details, refer to Appendix B and Table 4.

![](images/e7f8db62a4b73e90dc192cc2418ad4b9c19e9bfb96972ab09bb6373d8582eeee.jpg)  
Figure 2: Dataset extent and station distribution The dashed line shows the extent for which the NWP product is available. The square, triangle, and circle markers are train, validation, and test stations, respectively. Left: The Switzerland dataset. Center: Europe dataset. Right: Same data as the Europe dataset, but with a regional hold-out split, which coincides with small dashed box on the right representing the extent Switzerland dataset.

Algorithm 1 OAT Training Step   
Input: NWP fields $\overline { { U _ { t } ^ { \mathrm { N W P } } } }$ , stations $\left( \mathbf { x } , \mathbf { y } _ { t } \right)$ , topography T   
1 $( \mathbf { x } ^ { c } , \mathbf { y } ^ { c } ) , ~ ( \mathbf { x } ^ { * } , ~ \mathbf { y } ^ { * } ) \gets$ RandomSplit((x, y)) // Random 80/20 partition of stations   
2 for each location i in $\mathbf { x } ^ { c } \cup \mathbf { x } ^ { * }$ do   
3 $\mathbf { z } _ { i } ~ \gets ~ e ( U _ { t , i } ^ { \mathrm { { N W P } } } , ~ T _ { i } , ~ t )$ // Embed and fuse NWP prediction and topography cond. on   
time   
4 $\phi ^ { * }  o ( \mathbf { z } ^ { * } )$ // Get pre-oat predictive distribution   
5 for each station i in $\mathbf { x } ^ { c }$ do   
6 $\tilde { \mathbf { z } } _ { i } \gets$ Fuse $\left( { \bf z } _ { i } , { \bf y } _ { i } , t \right)$ // Incorporate observations   
7 $\mathbf { z } _ { y , i } \gets \left( 1 - g ( \mathbf { m } _ { i } ^ { y } ) \right) \mathbf { z } _ { i } + g ( \mathbf { m } _ { i } ^ { y } ) \tilde { \mathbf { z } } _ { i } / /$ Gate for (partial) observation   
8 $\mathbf { z } _ { y } ^ { \ast } \gets$ Transformer $( \mathbf { z } ^ { * } , \ \mathbf { z } _ { y } ) \ / / $ Propagate via transformer   
9 $\phi _ { y } ^ { * }  o ( \mathbf { z } _ { y } ^ { * } )$ // Get oat predictive distribution   
10 $\ell \gets ( 1 - w \ell ) \ell _ { \phi _ { y } ^ { * } } ( \mathbf { y } ^ { * } ) + w \ell \ \ell _ { \phi ^ { * } } ( \mathbf { y } ^ { * } )$ // Compute loss (and gradients)

## 4 Experiments

## 4.1 Data

We train and evaluate our model on two datasets that difer in spatial scale, NWP source, and station network, allowing us to assess generalization in diferent settings. Both datasets span ∼5-6 years at hourly resolution and share the same three-part structure: an NWP deterministic prior, high resolution topography with additional features derived from it, and ground truth sparse observations. We perform experiments at two spatial scales, covering a regional scale around Switzerland and a broader European scale (see Figure 2).

Switzerland dataset For the first dataset we consider Switzerland, which poses a dificult challenge due to its complex topography. We take the COSMO-1 operational analysis archive of MeteoSwiss (see Zanetta et al. [2026] for details) as NWP input field, covering Switzerland at a horizontal grid spacing of ∼1.1 km (0.01°). It is a regional forecasting system operated by MeteoSwiss, and its hourly analysis is obtained via an ensemble Kalman filter data assimilation scheme. We use four physical fields that are considered to be directly relevant for the later targets (temperature and precipitation): 2 m temperature $( T _ { 2 \mathrm { m } } )$ , ground temperature $( T _ { G } )$ , hourly total precipitation, and total cloud cover fraction.

For each station and query location, terrain features are sampled from a high-resolution digital elevation model (DEM) at ∼50 m resolution. Besides the elevation itself, four terrain descriptors are used: topographic position index (TPI) at 500 m and 2000 m scales [De Reu et al., 2013], and west– east and south–north terrain slope derivatives at 2000 m scale. These descriptors are precomputed on the DEM grid, so multi-scale terrain context enters through the descriptors themselves rather than through a spatial patch.

Surface observations are sourced from the PeakWeather dataset [Zambon et al., 2025], which provides access to measurements from the SwissMetNet, the automatic station network of MeteoSwiss. The measurements are available at 10-minute resolution and are aggregated to hourly for this work. For temperature the aggregation function is the average, and for precipitation it is the sum. The network comprises 302 stations across Switzerland. Two target variables are considered: 2-m temperature (<sup>◦</sup>C) and hourly precipitation accumulation (mm h<sup>−1</sup>). Station coordinates are expressed in the Swiss national reference frame (CH1903+/LV95). The data span 2019-07-03 to 2024-09-04.

Europe dataset The second dataset has a continental scale, where we consider the spatial extent of the European continent and surroundings $( 3 4 \mathrm { - 7 1 . 7 5 ^ { \circ } N , 2 4 . 7 5 ^ { \circ } W \mathrm { - } 4 5 ^ { \circ } E } )$ . The NWP field consists of ERA5 reanalysis fields [Hersbach et al., 2020] obtained from ECMWF via the MARS archive on the native N320 reduced Gaussian grid, whose spacing is ∼31 km. It is a global atmospheric reanalysis produced by ECMWF, generated by assimilating observational data into the Integrated Forecast System (IFS) using 4D-Var [Dimet and Talagrand, 1986]. Again, three fields are used that are directly relevant for the target: 2-m temperature, total precipitation, and total cloud cover fraction.

Two static surface descriptors are provided per station: terrain elevation is taken from the Terrain Tiles [Mapzen, 2017] at 3 arc-seconds (∼90 m) and is resampled to $0 . 0 1 ^ { \circ }$ . Additionally, we compute distance to the nearest coastline at the same resolution, which serves as a proxy for the maritime influence on local climate.

Hourly surface observations are acquired from the Meteostat CLI and its data providers (Deutscher Wetterdienst for Germany, GeoSphere Austria for Austria, and NOAA ISD, License: CC BY 4.0), a consolidated open-access database of global surface stations. After quality filtering, the network comprises 3554 stations distributed across Europe. Two target variables are considered: 2-m temperature (<sup>◦</sup>C) and hourly precipitation accumulation (mm h<sup>−1</sup>). In Figure 2 we see that some regions are more densely sampled than others (e.g. Germany has a much denser sampling rate). This is due to station availability in the Meteostat interface. The archive spans 2019-01-01 to 2024-12-31.

Europe dataset with region hold-out The third dataset is the same as the second one, except that an entire region encompassing Switzerland and its neighboring areas is held out during training. This configuration enables an assessment of the spatial generalization ability of the tested models to unobserved regions. At test time, 80% of the hold-out stations are selected as conditioning stations, while the remaining 20% are reserved as final test stations.

NWP inperpolation and local readout Both NWP archives are delivered as unstructured lists of grid points. The subsequent methodology requires NWP fields at arbitrary locations. Hence, we interpolate the NWP fields with Delaunay triangulation, since NWP grids are irregular in both datasets. The topographic variables are provided at relatively high resolution and are dificult to interpolate meaningfully (e.g. TPI). Hence, we resort to just take the value at the nearest available location.

While the Swiss coordinate system allows the use of Euclidean distance metrics with negligible distortion within the Swiss domain, this does not hold for the Europe dataset. Distances in planar coordinates become increasingly distorted toward higher latitudes. We therefore project coordinates onto the sphere and compute Euclidean distances in the three-dimensional space. Consequently, $\mathbf { x } ^ { \mathrm { e l e v } }$ is four dimensional with coordinates in 3D space and elevation. Since we only consider a nearest-neighbor network, this is a fair approximation.

For a valid training procedure, we use an identical, dataset-agnostic splitting procedure for both datasets, applied separately along the temporal and station (spatial) axes (Figure 2 left and center). Stations are randomly permuted and partitioned into disjoint train/validation/test sets with a nominal 60%/20%/20% split, never shared across splits. This yields 182/60/60 stations (of 302) for the Swiss dataset, and 2134/710/710 stations (of 3554) for the Europe dataset. The splits are the same across all models. A single contiguous temporal cutof reserves the last 365 days of each dataset as the test period; the remaining, earlier period is shared by train and validation alike. It should be noted that train and validation are therefore not separated in time, only test is temporally held out – validation is a purely spatial (station) hold-out to make use of most of the data. For the Switzerland dataset, train/val covers 2019-07-03 to 2023-09-05, and test covers 2023-09-06 to 2024-09-04; in the Europe dataset, train/val covers 2019-01-01 to 2024-01-01, and test covers 2024-01-02 to 2024-12-31 (365 days). Although training and validation periods should ideally be disjoint, we allowed them to overlap in order to make most eficient use of the data. This choice does not compromise the validity of the reported results, as all evaluations are conducted on a separate, non-overlapping test period.

Since OAT does not rely on absolute coordinates, it could, in principle, also make predictions in areas where it was not trained. To test this, we use the Europe dataset to construct a spatial hold-out experiment, where all stations in the Swiss domain are excluded from the training set (Figure 2 right). This yields 2471 training stations and 617 validation stations. Then we define 80% of the stations (372) within the test window as context, to predict the remaining 20% (94). This split also allows us to directly benchmark OAT against COSMO fields. The training period is 2019-01-01 to 2024-01-01. The test period is 2024-01-02 to 2024-09-04, as COSMO is not available for the full 2024 year.

## 4.2 Baselines

We consider five diferent baselines. The first is the NWP interpolator, which does not involve learning parameters but uses interpolation of the NWP field of the target variable, and takes that as prediction. The second is the uninformed baseline, which uses only the first part of OAT, using the uninformed embedding for predictions, i.e. there is no observation assimilation transformer block. This makes the uninformed baseline a feedforward neural network, trained on weather stations, that tries to predict at test locations based on the interpolated NWP field, the topography, and the time-embedding, but without the ability of data assimilation. As a third baseline, we use the Gaussian process methodology by Zanetta et al. [2026] in two variants. In both the prior mean is predicted by an MLP, as in the uninformed baseline. In the GP baseline, we use an automatic relevance determination (ARD) kernel as the covariance function. In the DK-GP (deep kernel GP), the covariance function is an ARD kernel multiplied by a deep RBF kernel, following Zanetta et al. [2026] (see Appendix C for details). Since the GP framework requires a Gaussian likelihood for exact inference, we train this model on the log-transformed (with 10<sup>−3</sup> ofset) precipitation values. However, we track the CRPS in the original data space on the validation set for model selection. Furthermore, while it is possible in principle, no multivariate code for GP is provided, hence we train every variable separately. As an additional baseline, we consider the gridded TNP [Ashman et al., 2025], encoding NWP and observations on a latent pseudo-token grid (80 × 120 for Switzerland, 40 × 72 for Europe; see Appendix D), where then Swin attention is used to encode the global information. The decoder then provides the predictive distribution at the points of interest. For fairness, while diferent from the original model, we provide the decoder with the high-resolution topographic data. Like the OAT, all models are trained with the closed form CRPS loss of each predictive distribution (see Appendix A). We use the AdamW optimizer [Loshchilov and Hutter, 2019] with cosine annealing learning rate scheduling for all models except the Gridded TNP. For the latter, we used Adam and no scheduler. The hyperparameters used for OAT are given in Table 3 in the Appendix.

![](images/a820da3b90f84fd8bf40dde26f1671a07e2585198fd638bb9b9c1d9d1c13b2f3.jpg)  
Figure 3: Example of predicted fields From left to right: The interpolated NWP, the pre-OAT predicted, and the OAT field with assimilated and test stations. The color code of the stations are the measurement values. Rightmost is the diference between the pre-OAT and OAT field, where the station color code is between pre-OAT and measurement at that station. Top row: Temperature fields. Bottom row: Precipitation fields. The prediction fields are sampled on a regular 250 m grid.

## 4.3 Predictive performance of OAT

In Figure 3 we show illustrative predictive (mean) fields of temperature and precipitation in the Switzerland dataset queried at 250m resolution. To see the efect of the observation assimilation, we make use of OAT’s design, and compare the predictive field of its pre-OAT distribution (no observation assimilated) with its final prediction. The temperature fields of both the prior and final OAT are more finely resolved compared to the NWP, demonstrating that both successfully incorporate the high-resolution topography information. The diference between the two fields indicates that the observation assimilation transformer has a correcting efect on the fields by moving the fields closer to the observed station temperatures. While topography seems to play a less important role for precipitation, the observation assimilation efect is even stronger. While the pre-OAT field follows closely the NWP field, the OAT predicts a wide precipitation band along the north of the Alps, closely matching the observed station values at assimilated and test locations.

We would like to emphasize that the figures show the mean of the predictive distributions; these are therefore expected to appear smoother than physical precipitation patterns.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="3">Temperature</td><td colspan="3">Precipitation</td></tr><tr><td>MAE↓</td><td>CRPS↓</td><td>Skill ↑</td><td>MAE↓</td><td>CRPS ↓</td><td>Skill ↑</td></tr><tr><td rowspan="6">Switzerland</td><td>COSMO Interp.</td><td>1.287</td><td>1.287</td><td>-0.637</td><td>0.146</td><td>0.146</td><td>-0.587</td></tr><tr><td>Uninf. baseline</td><td>1.089</td><td>0.786</td><td>0.000</td><td>0.122</td><td>0.092</td><td>0.000</td></tr><tr><td>GP</td><td>1.109</td><td>0.815</td><td>-0.037</td><td>0.130</td><td>0.111</td><td>-0.206</td></tr><tr><td>DK-GP</td><td>1.125</td><td>0.827</td><td>-0.052</td><td>0.119</td><td>0.101</td><td>-0.094</td></tr><tr><td>Gridded TNP</td><td>1.429</td><td>1.115</td><td>-0.418</td><td>0.097</td><td>0.073</td><td>0.207</td></tr><tr><td>OAT</td><td>1.046</td><td>0.751</td><td>0.045</td><td>0.086</td><td>0.064</td><td>0.306</td></tr><tr><td rowspan="6">Europe</td><td>ERA5 Interp.</td><td>1.305</td><td>1.305</td><td>-0.445</td><td>0.106</td><td>0.106</td><td>-0.623</td></tr><tr><td>Uninf. baseline</td><td>1.247</td><td>0.903</td><td>0.000</td><td>0.082</td><td>0.065</td><td>0.000</td></tr><tr><td>GP</td><td>1.189</td><td>0.934</td><td>-0.036</td><td>0.086</td><td>0.076</td><td>-0.170</td></tr><tr><td>DK-GP</td><td>1.145</td><td>0.874</td><td>0.031</td><td>0.083</td><td>0.074</td><td>-0.134</td></tr><tr><td>Gridded TNP</td><td>1.178</td><td>0.882</td><td>0.023</td><td>0.077</td><td>0.060</td><td>0.072</td></tr><tr><td>OAT</td><td>0.992</td><td>0.714</td><td>0.209</td><td>0.067</td><td>0.052</td><td>0.203</td></tr><tr><td rowspan="4">Europe (region hold-out)</td><td>ERA5 Interp.</td><td>1.702</td><td>1.702</td><td>-0.376</td><td>0.160</td><td>0.160</td><td>-0.668</td></tr><tr><td>COSMO Interp.</td><td>1.168</td><td>1.168</td><td>0.056</td><td>0.141</td><td>0.141</td><td>-0.471</td></tr><tr><td>Uninf. baseline</td><td>1.701</td><td>1.237</td><td>0.000</td><td>0.118</td><td>0.096</td><td>0.000</td></tr><tr><td>OAT</td><td>1.086</td><td>0.786</td><td>0.365</td><td>0.090</td><td>0.069</td><td>0.277</td></tr></table>

Table 2: Model performance on the test set. Skill is computed with CRPS and the comparison is to the uninformed baseline.

Table 2 shows the aggregate performance of diferent models across the test-sets (see Table 6 for confidence intervals). We compute the mean absolute error (MAE) between the median prediction and the observations, the CRPS, and the skill in CRPS compared to the uninformed baseline on the test stations. In terms of skill, OAT significantly outperforms the uninformed baseline on both datasets and for both variables, except for temperature on the Switzerland dataset, where the diference is not statistically significant. OAT achieves the lowest MAE for temperature, reducing it close to ${ \sim } 1 ^ { \circ } \mathrm { C }$ . Surprisingly, on the Swiss domain, the remaining baselines sufer performance degradation compared to the uninformed baseline for temperature. A plausible explanation is that, in such complex topography, most of the relevant information is already contained in the NWP temperature fields and elevation. Hence, the baselines seem to struggle to efectively assimilate station observations in this setting. In particular, the Gridded TNP’s performance sufers on the Switzerland dataset. We hypothesize that some information is lost by passing it through the latent 80×120 grid, which is relatively coarse and cannot resolve the fine grained topography of Switzerland. For precipitation, by contrast, the Gridded TNP generally improves upon the uninformed baseline, although it is still outperformed by OAT. Moreover, OAT’s predictive distributions achieve a lower CRPS. Table 5 compares alternative attention mechanisms for OAT on the Swiss dataset. The proposed mechanism (Equation (2)) performs comparably, with only marginal diferences between the variants.

We assess probabilistic calibration with the sample-based probability integral transform (PIT) histogram. For the zero-inflated distribution of precipitation, we split the analysis into a reliability diagram for the dry events and a PIT histogram for the wet events (see Appendix E). The calibration

![](images/de681bfc1285f4256baeb91d6531e03136ba931d0a16c59a72177c8e036e1865.jpg)

![](images/87af4f36b570d31680d5fd3afdfafa2160f0bb737593eb6b46a8c9b4917b8b69.jpg)  
(a) Calibration on Switzerland dataset.

![](images/526cdbbda346ccbbb6e81903c0fbb2f00f9e2954a7ea98f1f565888cb92e9fc1.jpg)

![](images/4cd827547c0af4e331f8d8e0e395df04f0ac8c45797680988a94aac86d54b17d.jpg)

![](images/15a5be8cd45e6701d81a7cc6e541ab7334a89eabac60a81ad5023b0eefe7ae22.jpg)  
(b) Calibration on Europe dataset

![](images/82befabc567ec21bc2adda385192f29d8d0582c184b6174b51050034dc91d8f9.jpg)

Figure 4: Calibration for the OAT on the test set. In each panel, from left to right: the PIT histogram for temperature, the reliability diagram for the probability of zero precipitation, and the precipitation PIT histogram for wet events.

of the uninformed baseline and the OAT model for both datasets is presented in Figure 4. For Switzerland, we see that both models sufer from overdispersion for temperature, i.e. their predictive distributions are too conservative, although OAT shows slight improvement over the uninformed baseline. For precipitation, the uninformed baseline is even marginally better calibrated. We hypothesize that this is due to the comparatively small number of stations in the Switzerland dataset, as the picture changes for the Europe dataset, where OAT is better calibrated for both temperature and precipitation.

![](images/34474836af2100aab2cc8fab1a979a57d364c3277124de7819d0a9e9c168a1c3.jpg)  
(a) Switzerland dataset

![](images/3973b5644b304e9ad43f661f75dc3e8784458a9d03bfa37fe97df13c18e165bb.jpg)  
(b) Europe dataset  
Figure 5: Stratification analysis of CRPS From left to right, the CRPS on the test set as function of season, elevation, and distance to closest observed station. The white bar denotes the median, and the boxes are the first and third quartile. Since the uninformed baseline is not conditioned on any observation, it is not displayed in the distance panels.

In order to understand the performance of OAT better, we stratify the CRPS in Figure 5a and 5b. For seasonality, the temperature CRPS stays relatively constant, while precipitation during the summer months shows less variable CRPS. For precipitation OAT performs best across all seasons. With respect to elevation, the CRPS increases with altitude for both variables and all models, with OAT performing best across the entire range. On the Switzerland dataset, this increase is particularly pronounced for precipitation at high altitudes (>2 km). We further examine how the distance to the nearest observed station afects the CRPS. This is of particular interest for the Europe dataset, where station density varies considerably. As shown in Figure 5b, the CRPS increases slightly with distance for temperature. For precipitation, it remains relatively stable up to a distance of ∼100 km, beyond which it rises noticeably. This suggests a critical distance beyond

![](images/2fb892177e4966d13362911bb86407acb99ce2629c1b896bca18110bbbf92782.jpg)  
Figure 6: Region hold-out and downscaling temperature from the ERA5 grid to the COSMO grid (∼1 million points) on 2024-06-21 12:00 UTC for Swiss domain. In the first row, the input ERA5 field, the uninformed mean prediction, and the OAT mean prediction. In the second row, the corresponding COSMO analysis field as well as the diferences between the informed and uninformed models against that analysis. The third row shows the test stations and the corresponding prediction error compared to the observed temperature for COSMO, the uninformed baseline, and OAT.

## 4.4 Region hold-out and downscaling from ERA5 to COSMO

One important design choice of OAT was to not provide the model with the absolute positions of stations or query points, but only with distance information. In principle, this would allow for spatial transferability, i.e. making predictions in a region on which OAT was not trained. To investigate whether OAT’s good predictive performance persists in such a scenario, we train on the Europe dataset excluding all stations within and around Switzerland (Figure 2 right). This setup also allows assessment of OAT’s capabilities against the high resolution analysis product COSMO (∼1 km), while only providing OAT the NWP information from the ERA5 N320 reduced Gaussian grid (∼31 km). We provide 80% of the stations in the hold-out domain as condition and predict on the remaining 20%. The results are provided in the bottom of Table 2. OAT slightly outperforms COSMO for temperature in MAE, even though COSMO might have had access to some of the test stations. Furthermore, OAT improves the ERA5 interpolation, establishing transfer capabilities in this setting. For a particular example of downscaling the temperature field and comparison to COSMO see Figure 6. OAT also outperforms COSMO interpolation for precipitation.

![](images/c2c019924890a2188a82331d17eeacd128833723d513617e62df2f804376e355.jpg)  
Figure 7: OAT’s predictive performance for exemplary extreme events. Top: Extreme precipitation event causing flooding in the Maggia valley end of June 2024. Left: Cumulated precipitation map over 24h for NWP, OAT and their diference, respectively. Right: Time-series of precipitation at two test-stations. Bottom: Heat wave in July 2024 in south eastern Europe. Left: Temperature maps of NWP (COSMO/ERA5) and OAT, with diferences. Right: Two test-stations for one week.

## 4.5 Case study: Predictions on extreme events

A crucial capability for data-driven models is to perform well during extreme events. Hence, we investigate OAT’s performance on two events, one for the Switzerland dataset and one for the Europe dataset (without hold-out). Both events are in the corresponding test sets. The results are shown in Figure 7. The first case is the Maggia flood on 2024-06-29/30, triggered by an extreme precipitation event. For the accumulated precipitation over this period, OAT achieves a substantially lower MAE than COSMO at the test stations in the afected region (5.5 mm vs. 14.1 mm). The field plot shows that COSMO predicts considerably more rainfall than OAT. Given the extreme nature of the precipitation event, accurate predictions at specific stations are inherently dificult. OAT accounts for this by drastically increasing its predictive uncertainty during the event, which is a desirable behavior of a probabilistic model. The second extreme event is the heatwave in southeastern Europe in July 2024. Here, OAT outperforms the ERA5 interpolation in terms of MAE at the test stations $( 1 . 0 ^ { \circ } \mathrm { C } \ \mathrm { v s . } \ 1 . 6 ^ { \circ } \mathrm { C } )$ for the timestep shown in the lower right panel in Figure 7. Moreover, OAT’s predictive distribution closely tracks the observed temperature throughout the week of the event (Figure 7, lower right).

## 5 Discussion

Inspired by classical statistical post-processing and recent advances in neural processes [Ashman et al., 2024a,b], we introduced OAT, a novel architecture that refines NWP fields by leveraging high-resolution topographic information and sparse station observations. Benchmarked against several strong baselines, OAT achieves competitive predictive performance on both a small and a large scale dataset, with the latter demonstrating the scalability of our approach (see Appendix F for detailed analysis). The model is highly flexible with respect to the choice of output distributions and handles missing observations in a straightforward manner. A particularly useful feature of OAT is its ability to inspect predictions before and after observations are assimilated, making it possible to quantify the contribution of the observational input. We further showed that OAT generalizes to regions not seen during training, enabling spatial transferability, while its performance is limited at the tails of the training distribution (e.g., precipitation at high altitudes). Finally, in case studies of extreme events, OAT appropriately increases its predictive uncertainty, indicating robust uncertainty quantification in these scenarios.

Despite these findings, OAT has some limitations. First, the model only captures the marginal predictive distribution at each location, rather than the correlations and joint distribution across locations. Consequently, samples at diferent locations are conditionally independent: although the marginals are well calibrated, a field composed of independent samples at each location does not represent a physically meaningful precipitation scenario. Second, OAT does not currently incorporate dense radar observation products, which, from a meteorological perspective, are essential for obtaining spatially coherent precipitation fields. Third, OAT provides no straightforward way to distinguish measurement noise from model uncertainty, a separation that arises naturally in the Bayesian GP framework.

Several directions could address these limitations. Modeling correlated noise with neural processes, as proposed by Bruinsma et al. [2021], would be a natural extension of OAT, as would the inclusion of additional dense observational inputs. Additionally, spatial coherence could be encouraged through a multivariate scoring rule such as an energy score, a multivariate generalization of the CRPS, evaluated jointly across locations. Alternatively, OAT’s architecture could be embedded within conditional generative denoising difusion [Ho et al., 2020] or flow matching frameworks [Lipman et al., 2023].

In this work, we focused on the instantaneous refinement of NWP fields. Incorporating past and future observations is a promising direction for further improving predictions. In OAT, this can be achieved naturally by treating time as an additional coordinate alongside location and elevation. OAT could then even be used for forecasting, trained to predict future observations conditioned on past ones, which would make it directly comparable to the models by Miralles et al. [2026] and Yang et al. [2024].

Beyond the setting considered here, we believe OAT is applicable to a broader range of problems. One example is the refinement of sea surface temperature or ocean current fields using observations from floating surface drifters [Botvynko et al., 2025], a challenging task due to the continuously

changing positions of the measurements.

In conclusion, OAT is a versatile model that demonstrated strong performance on the NWP refinement task considered here, ofering multiple avenues for future extensions.

## Acknowledgements

This project was conducted as part of the MeteoSwiss-Swiss Data Science Center collaborative framework agreement. Michele Cattaneo was supported by the EUMETNET Artificial Intelligence and Machine Learning for Weather, Climate and Environmental Applications (E-AI) Programme through the 2025 Research Cooperation Grant.

## References

Marcin Andrychowicz, Lasse Espeholt, Di Li, Samier Merchant, Alexander Merose, Fred Zyda, Shreya Agrawal, and Nal Kalchbrenner. Deep learning for day forecasts from sparse observations. arXiv preprint arXiv:2306.06079, 2023.

Matthew Ashman, Cristiana Diaconu, Junhyuck Kim, Lakee Sivaraya, Stratis Markou, James Requeima, Wessel P Bruinsma, and Richard E Turner. Translation equivariant transformer neural processes. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of PMLR, 2024a.

Matthew Ashman, Cristiana Diaconu, Adrian Weller, Wessel Bruinsma, and Richard E. Turner. Approximately equivariant neural processes. In Advances in Neural Information Processing Systems, 2024b.

Matthew Ashman, Cristiana Diaconu, Eric Langezaal, Adrian Weller, and Richard E Turner. Gridded transformer neural processes for spatio-temporal data. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of PMLR, 2025.

Li-Li Bao, Jiang-She Zhang, and Chun-Xia Zhang. Spatial multi-attention conditional neural processes. Neural Networks, 173:106201, 2024.

Peter Bauer, Alan Thorpe, and Gilbert Brunet. The quiet revolution of numerical weather prediction. Nature, 525(7567):47–55, 2015.

G´erard Biau, Eduardo Zorita, Hans von Storch, and Hans Wackernagel. Estimation of precipitation by kriging in the eof space of thesea level pressure field. Journal of climate, 12(4):1070–1085, 1999.

Massimo Bonavita. On some limitations of current machine learning weather prediction models. Geophysical Research Letters, 51(12):e2023GL107377, 2024.

Daria Botvynko, Carlos Granero-Belinchon, Simon Van Gennip, Abdesslam Benzinou, and Ronan Fablet. Neural prediction of lagrangian drift trajectories on the sea surface. Artificial Intelligence for the Earth Systems, 4(3):e240052, 2025.

Wessel P. Bruinsma, James Requeima, Andrew Y. K. Foong, Jonathan Gordon, and Richard E. Turner. The gaussian neural process, 2021.

Wessel P. Bruinsma, Stratis Markou, James Requiema, Andrew Y. K. Foong, Tom R. Andersson, Anna Vaughan, Anthony Buonomo, J. Scott Hosking, and Richard E. Turner. Autoregressive conditional neural processes, 2023.

Gerd B¨urger. Expanded downscaling for generating local weather scenarios. Climate Research, 7: 111–128, 1996.

Alberto Carrassi, Marc Bocquet, Laurent Bertino, and Geir Evensen. Data assimilation in the geosciences: An overview of methods, issues, and perspectives. Wiley Interdisciplinary Reviews: Climate Change, 9(5):e535, 2018.

Jeroen De Reu, Jean Bourgeois, Machteld Bats, Ann Zwertvaegher, Vanessa Gelorini, Philippe De Smedt, Wei Chu, Marc Antrop, Philippe De Maeyer, Peter Finke, Marc Van Meirvenne, Jacques Verniers, and Philippe Cromb´e. Application of the topographic position index to heterogeneous landscapes. Geomorphology, 186:39–49, 2013. ISSN 0169-555X. doi: https://doi.org/10.1016/j. geomorph.2012.12.015.

Micha¨el Deferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. Advances in neural information processing systems, 29, 2016.

Francois-Xavier Le Dimet and Olivier Talagrand. Variational algorithms for analysis and assimilation of meteorological observations: theoretical aspects. Tellus A, 38A(2):97–110, 1986. doi: https://doi.org/10.1111/j.1600-0870.1986.tb00459.x.

Alexandru Dumitrescu. A deep learning framework for gridding daily climate variables from a sparse station network. EGUsphere preprint, 2026. doi: 10.5194/egusphere-2026-223.

Jacob R Gardner, Geof Pleiss, David Bindel, Kilian Q Weinberger, and Andrew Gordon Wilson. GPyTorch: Blackbox matrix-matrix gaussian process inference with gpu acceleration. In Advances in Neural Information Processing Systems, 2018.

Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, Danilo J Rezende, SM Ali Eslami, and Yee Whye Teh. Neural processes. arXiv preprint arXiv:1807.01622, 2018.

Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In International conference on machine learning, pages 1263–1272. Pmlr, 2017.

Tilmann Gneiting, Adrian E Raftery, Anton H Westveld III, and Tom Goldman. Calibrated probabilistic forecasting using ensemble model output statistics and minimum crps estimation. Monthly weather review, 133(5):1098–1118, 2005.

Jonathan Gordon, Wessel P Bruinsma, Andrew YK Foong, James Requeima, Yann Dubois, and Richard E Turner. Convolutional conditional neural processes. arXiv preprint arXiv:1910.13556, 2019.

Hans Hersbach, Bill Bell, Paul Berrisford, Shoji Hirahara, Andr´as Hor´anyi, Joaqu´ın Mu˜noz-Sabater, Julien Nicolas, Carole Peubey, Raluca Radu, Dinand Schepers, Adrian Simmons, Cornel Soci, Saleh Abdalla, Xavier Abellan, Gianpaolo Balsamo, Peter Bechtold, Gionata Biavati, Jean Bidlot, Massimo Bonavita, Giovanna De Chiara, Per Dahlgren, Dick Dee, Michail Diamantakis, Rossana Dragani, Johannes Flemming, Richard Forbes, Manuel Fuentes, Alan Geer, Leo Haimberger,

Sean Healy, Robin J. Hogan, El´ıas H´olm, Marta Janiskov´a, Sarah Keeley, Patrick Laloyaux, Philippe Lopez, Cristina Lupu, Gabor Radnoti, Patricia de Rosnay, Iryna Rozum, Freja Vamborg, Sebastien Villaume, and Jean-No¨el Th´epaut. The era5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730):1999–2049, 2020. doi: https://doi.org/10.1002/qj.3803.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Hyunjik Kim, Andriy Mnih, Jonathan Schwarz, Marta Garnelo, Ali Eslami, Dan Rosenbaum, Oriol Vinyals, and Yee Whye Teh. Attentive neural processes. arXiv preprint arXiv:1901.05761, 2019.

Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixiang Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10012–10022, 2021.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id= Bkg6RiCqY7.

Hengzhao Ma and Jianzhong Li. A sub-linear time algorithm for approximating k-nearest-neighbor with full quality guarantee. Theoretical Computer Science, 857:59–70, 2021.

Mapzen. Terrain tiles. Registry of Open Data on AWS, https://registry.opendata. aws/terrain-tiles, 2017. Accessed 2026-05-26. Source attribution: https://github.com/ tilezen/joerd/blob/master/docs/attribution.md.

Morteza Mardani, Noah Brenowitz, Yair Cohen, Jaideep Pathak, Chieh-Yu Chen, Cheng-Chin Liu, Arash Vahdat, Mohammad Amin Nabian, Tao Ge, Akshay Subramaniam, et al. Residual corrective difusion modeling for km-scale atmospheric downscaling. Communications Earth & Environment, 6(1):124, 2025.

Oph´elia Miralles, Daniele Nerini, Jonas Bhend, Baudouin Raoult, and Christoph Spirig. Observation-guided interpolation using graph neural networks for high-resolution operational nowcasting in switzerland. Artificial Intelligence for the Earth Systems, 5(2):250081, 2026. doi: 10.1175/AIES-D-25-0081.1.

Tung Nguyen and Aditya Grover. Transformer neural processes: Uncertainty-aware meta learning via sequence modeling. arXiv preprint arXiv:2207.04179, 2022.

Neelesh Rampal, Sanaa Hobeichi, Peter B Gibson, Jorge Ba˜no-Medina, Gab Abramowitz, Tom Beucler, Jose Gonz´alez-Abad, William Chapman, Paula Harder, and Jos´e Manuel Guti´errez. Enhancing regional climate downscaling through advances in machine learning. Artificial Intelligence for the Earth Systems, 3(2):230066, 2024.

Stephan Rasp and Sebastian Lerch. Neural networks for postprocessing ensemble weather forecasts. Monthly Weather Review, 146(11):3885–3900, 2018.

Pedro Sousa, Will Tebbutt, Sadiq Jafer, Robin Young, Anil Madhavapeddy, and Richard E. Turner. Earth observation embeddings are efective sub-grid descriptors for probabilistic weather downscaling, 2026.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.

Anna Vaughan, Will Tebbutt, J Scott Hosking, and Richard E Turner. Convolutional conditiona neural processes for local climate downscaling. Geoscientific Model Development, 15(1):251–268, 2022.

Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, and Eric P Xing. Deep kernel learning. In Artificial intelligence and statistics, pages 370–378. PMLR, 2016.

Qidong Yang, Jonathan Giezendanner, Daniel Salles Civitarese, Johannes Jakubik, Eric Schmitt, Anirban Chandra, Jeremy Vila, Detlef Hohl, Chris Hill, Campbell Watson, and Sherrie Wang. Local of-grid weather forecasting with multi-modal earth observation data. arXiv preprint arXiv:2410.12938, 2024.

Daniele Zambon, Michele Cattaneo, Ivan Marisca, Jonas Bhend, Daniele Nerini, and Cesare Alippi. Peakweather: Meteoswiss weather station measurements for spatiotemporal deep learning, 2025.

Francesco Zanetta, Daniele Nerini, Tom Beucler, and Mark A Liniger. Physics-constrained deep learning postprocessing of temperature and humidity. Artificial Intelligence for the Earth Systems, 2(4):e220089, 2023.

Francesco Zanetta, Daniele Nerini, Matteo Buzzi, and Henry Moss. Eficient modeling of subkilometer surface wind with gaussian processes and neural networks. Artificial Intelligence for the Earth Systems, 5(2):240047, 2026.

## A Output distributions and loss Functions

Models are trained with the continuous ranked probability score (CRPS), a proper scoring rule which measures how well a predicted probability distribution matches an observed outcome in terms of calibration and sharpness. Many parametric distributions have a closed form definition. For the Normal distribution ${ \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ and target y:

$$
\operatorname { C R P S } _ { \mathrm { N } } \big ( ( \mu , \sigma ^ { 2 } ) , y \big ) = \sigma \left( z ( 2 \Phi ( z ) - 1 ) + 2 \phi ( z ) - \frac { 1 } { \sqrt { \pi } } \right) ,\tag{4}
$$

where $z = { \frac { y - \mu } { \sigma } }$ and $\Phi , \phi$ are the standard normal CDF and PDF respectively.

For Log-Normal distribution with location parameter $\mu _ { \mathrm { L N } }$ and scale $\sigma _ { \mathrm { L N } }$ the density function is

$$
p _ { \mathrm { L N } } ( y | \mu _ { \mathrm { L N } } , \sigma _ { \mathrm { L N } } ) = \frac { 1 } { y \sigma _ { \mathrm { L N } } \sqrt { 2 \pi } } \exp \left( - \frac { \left( \ln y - \mu _ { \mathrm { L N } } \right) ^ { 2 } } { 2 \sigma _ { \mathrm { L N } } ^ { 2 } } \right) .\tag{5}
$$

The CRPS is

$$
\mathrm { C R P S } _ { \mathrm { L N } } \big ( \big ( \mu _ { \mathrm { L N } } , \sigma _ { \mathrm { L N } } \big ) , y \big ) = y \big ( 2 \Phi \big ( z _ { \mathrm { L N } } \big ) - 1 \big ) - 2 m \left( \Phi \big ( z _ { \mathrm { L N } } - \sigma _ { \mathrm { L N } } \big ) - \Phi \big ( - \sigma _ { \mathrm { L N } } / \sqrt { 2 } \big ) \right) ,\tag{6}
$$

with $z _ { \mathrm { L N } } = ( \ln y - \mu _ { \mathrm { L N } } ) / \sigma _ { \mathrm { L N } }$ and $m = \exp ( \mu _ { \mathrm { L N } } + \sigma _ { \mathrm { L N } } ^ { 2 } / 2 )$ . While we do not use the log-normal distribution it is an important one to e.g. model wind speed. We provide this in the code.

For zero-inflated Log-Normal distribution with $\pi _ { \mathrm { Z I L N } }$ the probability of a random variable being non-zero, and µ and scale σ being location and scale parameter of the non-zero density, the density is given by

$$
\begin{array} { r } { p _ { \mathrm { Z I L N } } ( y | \pi _ { Z I L N } , \mu _ { \mathrm { Z I L N } } , \sigma _ { \mathrm { Z I L N } } ) = ( 1 - \pi _ { \mathrm { Z I L N } } ) \delta ( y ) + \pi _ { \mathrm { Z I L N } } p _ { \mathrm { L N } } ( y | \mu _ { \mathrm { Z I L N } } , \sigma _ { \mathrm { Z I L N } } ) , } \end{array}\tag{7}
$$

where $\delta ( y )$ is a Dirac delta function. The CRPS is

$$
\begin{array} { r l } & { \mathrm { C R P S } _ { \mathrm { Z I L N } } ( ( \pi _ { \mathrm { Z I L N } } , \mu _ { \mathrm { Z I L N } } , \sigma _ { \mathrm { Z I L N } } ) , y ) = ( 1 - \pi _ { \mathrm { Z I L N } } ) y + \pi _ { \mathrm { Z I L N } } \mathrm { C R P S } _ { \mathrm { L N } } ( ( \mu _ { \mathrm { Z I L N } } , \sigma _ { \mathrm { Z I L N } } ) , y ) } \\ & { \qquad + \ : 2 ( 1 - \pi _ { \mathrm { Z I L N } } ) \pi _ { \mathrm { Z I L N } } m ( \Phi ( \sigma _ { \mathrm { Z I L N } } / \sqrt { 2 } ) - 1 ) . } \end{array}\tag{8}
$$

Alternatively, a Monte Carlo estimation via predicted samples $\{ \hat { y _ { i } } \} _ { i = 1 } ^ { i = M }$ from a generic distribution exists in the form:

$$
\mathrm { C R P S } \left( \{ \hat { y _ { i } } \} _ { i = 1 } ^ { M } , y \right) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mid \hat { y _ { i } } - y \mid - \frac { 1 } { 2 M ( M - 1 ) } \sum _ { i = 1 } ^ { M } \sum _ { k = 1 } ^ { M } \mid \hat { y _ { i } } - \hat { y } _ { k } \mid\tag{9}
$$

We use the closed form losses in normalized space for training and the MC estimate for validation metrics using denormalized samples.

## B OAT Architectural Details

## Fusion layer:

The fusion layer $\mathrm { F u s e } ( a , b , c )$ combines information from a and b into a single vector conditioned on c. First, a and b are concatenated as $x = [ a \rVert b ] \in \mathbb { R } ^ { d }$ and then passed through a series of blocks as follows:

$$
\begin{array} { r } { \tilde { \boldsymbol { x } } = \mathrm { A d a L N } ( x ^ { l } , c ) } \\ { x ^ { l + 1 } = x ^ { l } + \mathrm { D r o p o u t } _ { p } \left( \mathrm { G L U } ( W \tilde { \boldsymbol { x } } ) \right) } \end{array}\tag{10}
$$

where $\mathrm { G L U } ( w ) = u \otimes \sigma ( v )$ is a gated linear unit function, where $w = [ u \rVert v ]$ , i.e u and v split w in two halves. AdaLN is an adaptive layer norm, allowing the fusion to be conditioned on inputs c. A linear projection $W \in \mathbb { R } ^ { 2 d \times d }$ (and optionally, a bias) is necessary to ensure that ${ \tilde { x } } ,$ composed of a and b concatenated in x scaled and shifted by the adaptive layer norm, is first mixed before being split into input and gate to the GLU, respectively.

In the OAT model, the fusion layer is used on two occasions; when combining the NWP data with the topography, and when combining the uninformed tokens with the observations. a and b are those values respectively. In both cases, the conditioning value c is an embedding of the day-of-year and hour-of-day informing the model about the time.

## Geometry-aware attention:

We consider three diferent attention mechanisms in this work. The first is very close to the standard dot product attention [Vaswani et al., 2017]. It is given by Equation (2), but without attention bias and difers from the standard implementation, that only a set of neighbors ${ \mathcal { N } } _ { i }$ for each station i is considered. The choice of neighborhood allows to introduce specific relational inductive biases in the architecture, similar to GNNs. We call this no-geo attention, since the attention mechanism is not provided with geographic information (apart from the topography that is embedded in the latent vectors ${ \bf z } )$ . The second and main mechanism that is used throughout the paper if not specified otherwise we call additive and is given by Equation (2). The third attention mechanism we call TE-TNP since it is closest to the mechanism proposed by Ashman et al. [2024a], where the attention logits are given by

$$
\alpha _ { i , j } = \{ \begin{array} { l l } { f ( [ Q _ { i } ^ { \top } K _ { j } \| ( \mathbf { x } _ { i } ^ { \mathrm { e l e v } } - \mathbf { x } _ { j } ^ { \mathrm { e l e v } } ) ] ) } & { \mathrm { ~ i f ~ } j \in \mathcal { N } _ { i } , } \\ { - \infty } & { \mathrm { ~ o t h e r w i s e ~ } , } \end{array}\tag{11}
$$

where the distance in position (and elevation) get concatenated to the dot product and then attention is formed non-linearly from these two sources. This is the richest attention class from a modeling perspective, but more dificult to interpret than the additive attention mechanism due to the non-linear mixing of $f .$ . The results of an ablation study for the diferent mechanisms can be found in Table 5.

Multi-head self-attention $\mathrm { M H S A } _ { \mathrm { g e o } } ( \mathbf { z } ; \mathbf { x } _ { y } ^ { \mathrm { e l e v } } )$ projects the same embeddings z linearly into queries, keys and values. Multi-head cross-attention $\begin{array} { r } { \mathrm { M H C A } _ { \mathrm { g e o } } ( \mathbf { z } _ { q } , \mathbf { z } _ { k } , \mathbf { z } _ { v } ; \mathbf { x } ^ { \mathrm { e l e v } , \star } , \mathbf { x } _ { y } ^ { \mathrm { e l e v } } ) } \end{array}$ , instead, projects distinct sequences into queries, keys and values. Throughout this work we assume that $\mathbf { z } _ { k } = \mathbf { z } _ { v }$ for cross-attention. The computation for self- and cross-attention is identical once the projections are formed, and we distinguish the two names purely for clarity in the following discussion.

## Observation-assimilation Transformer Block

At observed locations we generate embeddings $\mathbf { z } _ { y }$ informed by the observed y, while at query location we have uninformed embeddings $\mathbf { z } ^ { \star }$ . An encoder-decoder transformer performs a token mixing of those two sets:

$$
\begin{array} { r } { \mathbf { z } _ { y } ^ { \star } = \operatorname { T r a n s f o r m e r } ( \mathbf { z } ^ { \star } , \mathbf { x } ^ { \mathrm { e l e v , \star } } , \mathbf { z } _ { y } , \mathbf { x } _ { y } ^ { \mathrm { e l e v } } ) = \operatorname { D e c } ( ( \mathbf { z } ^ { \star } , \mathbf { x } ^ { \mathrm { e l e v , \star } } ) , ( \operatorname { E n c } ( \mathbf { z } _ { y } , \mathbf { x } _ { y } ^ { \mathrm { e l e v } } ) , \mathbf { x } _ { y } ^ { \mathrm { e l e v } } ) ) } \end{array}\tag{12}
$$

The architecture is close to the one described by Vaswani et al. [2017], with residual connections, dropout, layer norm, and feed-forward block after the attention. The only diferences are in that we do not have any positional embedding and we do not consider self-attention among the queries in the decoder blocks. Hence, the encoder is only composed of $\mathrm { M H S A _ { g e o } }$ blocks, while the decoder is only composed of ${ \mathrm { M H C A } } _ { \mathrm { g e o } }$ blocks.

The output of the final decoder layer is then used in the output head, where it is mapped to the parametric distribution via an MLP.

## C Gaussian Process baseline

For the GP models, we need a mean function and a covariance function, also called kernel. For the mean function, we use the same architecture as the uninformed baseline, except for the output head. As kernel, we use an automatic relevance determination kernel for the GP model, i.e. a radial basis function with dimension specific length scales. The input to the kernel are location and elevation. For the DK-GP [Wilson et al., 2016, Zanetta et al., 2026], we use the same kernel multiplied by an RBF kernel on a 4-dimensional MLP embedding of the feature trunk output, which combines NWP, topography, and time (2 hidden layers and a linear readout, width 64, ReLU). The loss is the CRPS on training held out stations. Context stations and target stations, on which the loss is computed, are sampled in the same manner as in the OAT training. For precipitation, we model the variable in log-space with a small ofset to account for the dry events. Compared to OAT training, we have to resort to 64-bit precision to ensure numerical stability. The hyperparameters of both GP variants are given in Table 7.

## D Gridded TNP baseline

We use the gridded TNP of Ashman et al. [2025] based on the implementation of the authors. The NWP field is subsampled with a fixed stride and encoded together with the observations onto a latent pseudo-token grid. Swin attention mixes the latent grid, and a cross-attention decoder reads it out at the query locations. As described in the main text, we give the decoder the same topographic features as OAT. The hyperparameters are given in Table 8.

## E Evaluation of model calibration

We consider 500 predictive samples on all test stations over the whole test year. For each station and time step, the PIT value is the fraction of predictive samples that fall at or below the observation, and the histogram of these values across all pairs is uniform when the forecasts are calibrated. For the reliability diagram, we take the predicted probability of a dry event as the fraction of zerovalued samples as well as the actual binary outcome, bin the pairs by the predicted probability, and plot the mean predicted probability against the observed frequency of dry observations in each bin, which lies on the diagonal under perfect calibration.

## F Model Scalability

We showcase empirical scalability comparing the GP and OAT models in Figure 8. The model’s hyperparameters are taken from the Switzerland experiment with a small change for the OAT. Because the GP is univariate, for the sake of this experiment, we make OAT univariate as well.

We evaluate the models with a fixed number of 1024 query stations as well as maintaining a proportional number of query stations as the number of conditioning stations grows. In the latter setting, the 80-20 ratio is used. The deep GP model’s scalability was comparable to the standard GP, and is therefore omitted from the plots. All timings were measured on a single NVIDIA A100 GPU. We discard 3 warm-up forward passes and report the mean time over the following 10. In the proportional setting, the exact and approximate GP have the same speed before $2 ^ { 1 0 } = 1 0 2 4$ observed stations. This is because, by default, the approximate solver is only called when there are more than 800 conditioning points. For the exact GP, we manually increase this limit to be greater than any number of points in the experiment. In the setting with fixed 1024 observed stations, the approximate solver is always in play. Interestingly, the exact solver is faster for such small matrices. We notice that for large values of number of points, the computation time of the OAT is dominated by the KNN computation. Under the assumption that all observed points are known a priori, a single pre-computed KD-tree could be used. Approximate KNN algorithms could also be employed to speed up OAT’s training and inference [Ma and Li, 2021]. The main takeaway from this scalability analysis is that, in this setting, the GP model is faster for small numbers of points but scales worse as the number of points grows. We attribute its early advantage to the highly optimized low-level linear algebra routines in GPyTorch [Gardner et al., 2018] and see a clear advantage of OAT for larger datasets. Figure 9 shows that while the KNN computation is a dominant computational component, it is still beneficial over a full attention scheme.

![](images/b2bcb82024f95aee076d6d5f3eee610ae7d511c3346d72d6533e3daf782ed52e.jpg)  
(a) Training with fixed number of query points.

![](images/eae1c3aeee93e9685af61796c881bb4adc0cfddec2339343e0bdc95b693d3589.jpg)  
(b) Inference with fixed number of query points.

![](images/bb51b73a4ac2690b91ac12506de309daf9f4935d38f946373f490432b5e96b9b.jpg)

![](images/4fd932d0891312275d4936906733e3e61ee2104426fe3674d8935f023d795e7a.jpg)  
(c) Training with proportional number of query points.  
(d) Inference with proportional number of query points.

Figure 8: Scalability analysis of OAT and GP. We empirically measure the time per step under two scenarios (the rows in the figure) as the number of observed stations grows up to $2 ^ { 1 6 } \approx 6 5 ^ { \prime } 0 0 0$ . In the first scenario, there is a fixed number of 1024 query points, while the second scenario has a number of query points proportional to the observed stations to maintain a 80%-20% ratio. The columns show the results for a training step, which includes forward and backward pass, and an evaluation step. A major bottleneck of the OAT’s forward pass is the nearest neighbour computation, which we include in the plot as a dashed line. For the GP, we compare two ways of computing its predictive distribution: GP (exact) relies on a Cholesky decomposition, whose cost grows cubically with the number of points, while GP (iterative) uses GPyTorch’s default, which switches from Cholesky to iterative conjugate-gradient and Lanczos methods above 800 points.

![](images/cb79d60cddccf187c5aa8eaf7eb42feb94c9a239b6fa778e0384bc46c1d60e3d.jpg)  
(a) Training with fixed number of query points.

![](images/ffdcb09c099142adf75713998fc5530155a7dd3ebdbabc45352f52747d7ea987.jpg)  
(b) Inference with fixed number of query points.

![](images/c0214845916372f2c99e5680a30d8322ee88e0bee71590ea748226cef5071b86.jpg)  
(c) Training with proportional number of query points.

![](images/7db744c2609a45fdbe8a8ac2ab800e39929e1703f56e4e2aa8488e2e2f7daec6.jpg)  
(d) Inference with proportional number of query points.

Figure 9: Comparison between a full-attention scheme and the default sparse attention scheme used in the OAT model to perform training and inference steps. The OAT setting is the same as in Figure 8, with the only diference that the full-attention scheme is turned on.

<table><tr><td>Component</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="4">NWP encoder</td><td>Type</td><td>pointwise MLP, GELU</td></tr><tr><td>Input channels</td><td>4 (CH) / 3 (EU)</td></tr><tr><td>Hidden layers</td><td>1, width 16</td></tr><tr><td>Output channels</td><td>16</td></tr><tr><td rowspan="4">Topographic encoder</td><td>Type</td><td>pointwise MLP, GELU</td></tr><tr><td>Input channels</td><td>5 (CH) / 2 (EU)</td></tr><tr><td>Hidden layers</td><td>1, width 16</td></tr><tr><td>Output channels</td><td>16</td></tr><tr><td>Both encoders</td><td>Batch normalization</td><td>yes</td></tr><tr><td rowspan="5">Fusion layers</td><td>Output / hidden channels</td><td>32  / 32</td></tr><tr><td>Conditioning dim. (time)</td><td>4</td></tr><tr><td>Normalization</td><td>AdaLN (pre-norm)</td></tr><tr><td>Blocks</td><td>1</td></tr><tr><td>Dropout</td><td>0.1</td></tr><tr><td rowspan="9">OAT transformer</td><td>Model dim. Demb</td><td>32</td></tr><tr><td>Attention heads</td><td>4</td></tr><tr><td>Encoder / decoder layers</td><td>3/3</td></tr><tr><td>Feedforward dim.</td><td>64</td></tr><tr><td>Dropout</td><td>0.1</td></tr><tr><td>Geometric attention</td><td>additive</td></tr><tr><td>Geo. bias input dim.</td><td>3 (CH) / 4 (EU)</td></tr><tr><td>Geo. bias FFN dim.</td><td>32</td></tr><tr><td>Attention mask</td><td>KNN, k = 8</td></tr><tr><td rowspan="3">Output head</td><td>Input features</td><td>32</td></tr><tr><td>Temperature</td><td>Normal</td></tr><tr><td>Precipitation</td><td>zero-inflated log-normal</td></tr><tr><td colspan="2">Pre-OAT loss weight we</td><td>0.1</td></tr><tr><td colspan="2">Trainable parameters</td><td>≈70k</td></tr></table>

Table 3: OAT architecture hyperparameters. CH: Switzerland dataset, EU: Europe dataset. Values are identical for both datasets except where noted. The uninformed baseline has only the encoders, the NWP–topography fusion and the output head. It uses widths 32 in the encoders and 64 in the fusion layer and output head, and has ≈ 20k parameters.

<table><tr><td>Setting</td><td>OAT / Uninf.</td><td>GP / DK-GP</td><td>Gridded TNP</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>Adam</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 2 } ~ \mathrm { ( C H ) } ~ / ~ 5 \times 1 0 ^ { - 3 } ~ \mathrm { ( E U ) }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td>0.01</td><td>0.01</td><td></td></tr><tr><td>LR schedule</td><td>cosine annealing,</td><td> $T _ { \mathrm { m a x } } = 5 0 0 ~ \mathrm { e p o c h s ^ { c } }$ </td><td>none</td></tr><tr><td>Batch size (time steps)a</td><td>1024 (CH)</td><td> $/ ~ 2 5 6 ~ ( \mathrm { E U } ) ^ { \mathrm { c } }$ </td><td> $1 ^ { \mathrm { b } }$ </td></tr><tr><td>Gradient clipping (norm)</td><td>1.0</td><td>1.0</td><td>0.5</td></tr><tr><td>Max. epochs</td><td>500c</td><td>500</td><td>300</td></tr><tr><td>Early-stopping patience</td><td>40</td><td>40</td><td></td></tr><tr><td>Float precision</td><td>32</td><td>64</td><td>32</td></tr><tr><td>Training loss</td><td>CRPS</td><td>CRPS</td><td>CRPS</td></tr><tr><td>Model selection</td><td>best val. loss</td><td>best val. CRPS (in data space)</td><td>best val. loss</td></tr><tr><td>Context/target split</td><td></td><td>80% / 20% of training stations, resampled each step</td><td></td></tr><tr><td>Hardware</td><td></td><td>1 node, 4 GPUs (A100), DDP</td><td></td></tr></table>

Table 4: Training configuration. CH: Switzerland dataset, EU: Europe dataset. <sup>a</sup>Per GPU; efective batch size ×4. One sample is one hourly time step with all its stations. <sup>b</sup>The Gridded TNP generator emits one sample per step with 10 000 steps per epoch. <sup>c</sup>The region hold-out models use a batch size of 32, $T _ { \mathrm { m a x } } = 1 0 0$ and at most 100 epochs.

<table><tr><td rowspan="2">Method</td><td colspan="3">Temperature</td><td colspan="3">Precipitation</td></tr><tr><td>MAE↓</td><td>CRPS↓</td><td>Skill ↑</td><td>MAE↓</td><td>CRPS ↓</td><td>Skill ↑</td></tr><tr><td>OAT (additive)</td><td>1.046</td><td>0.751</td><td>0.045</td><td>0.086</td><td>0.064</td><td>0.306</td></tr><tr><td>OAT (vanilla)</td><td>1.076</td><td>0.771</td><td>0.019</td><td>0.088</td><td>0.066</td><td>0.287</td></tr><tr><td>OAT (TE-TNP)</td><td>1.035</td><td>0.743</td><td>0.054</td><td>0.085</td><td>0.064</td><td>0.310</td></tr></table>

Table 5: Ablation of diferent attention mechanisms. Evaluated on the Switzerland dataset.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>Temperature</td><td rowspan=2 colspan=4>PrecipitationMAE↓        CRPS↓          Skill ↑</td></tr><tr><td rowspan=1 colspan=1>Dataset            Method</td><td rowspan=1 colspan=4>MAE↓         CRPS↓          Skill ↑</td></tr><tr><td rowspan=2 colspan=1>COSMO Interp.Uninf. baseline</td><td rowspan=1 colspan=2>[1.127, 1.464]</td><td rowspan=1 colspan=1>[1.132, 1.460]</td><td rowspan=1 colspan=1>[-0.785, -0.494]</td><td rowspan=1 colspan=2>[0.138, 0.155]  [0.138, 0.155]</td><td rowspan=1 colspan=1>[-0.633, -0.548]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[0.968, 1.210]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[0.706, 0.879]</td><td rowspan=1 colspan=1>[0.000, 0.000]</td><td rowspan=1 colspan=1>[0.114, 0.131]</td><td rowspan=1 colspan=1>[0.086, 0.098]</td><td rowspan=1 colspan=1>[0.000, 0.000]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=4 colspan=1>GPSwitzerlandDK-GPGridded TNPOAT (ours)</td><td rowspan=1 colspan=1>[1.015, 1.203]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[0.751, 0.890]</td><td rowspan=1 colspan=1>[-0.108, 0.031]</td><td rowspan=1 colspan=1>[0.118, 0.143]</td><td rowspan=1 colspan=1>[0.101, 0.122]</td><td rowspan=1 colspan=1>[-0.278, -0.144]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[1.022, 1.230]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[0.760, 0.906]</td><td rowspan=1 colspan=1>[-0.136, 0.022]</td><td rowspan=1 colspan=1>[0.108, 0.134]</td><td rowspan=1 colspan=1>[0.092, 0.113]</td><td rowspan=1 colspan=1>[-0.158, -0.041]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[1.236, 1.617]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[0.959, 1.293]</td><td rowspan=1 colspan=1>[-0.678, -0.186]</td><td rowspan=1 colspan=1>[0.090, 0.106]</td><td rowspan=1 colspan=1>[0.067, 0.079]</td><td rowspan=1 colspan=1>[0.180, 0.233]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>[0.954, 1.143]  [0.684, 0.835]</td><td rowspan=1 colspan=1>[-0.013, 0.093]</td><td rowspan=1 colspan=1>[0.079, 0.094]</td><td rowspan=1 colspan=2>[0.058, 0.070]   [0.280, 0.331]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=6 colspan=1>ERA5 Interp.Uninf. baselineGPEuropeDK-GPGridded TNPOAT (ours)</td><td rowspan=1 colspan=2>[1.226, 1.379]  [</td><td rowspan=1 colspan=1>1.233, 1.385]</td><td rowspan=1 colspan=1>[-0.492, -0.407]</td><td rowspan=1 colspan=2>[0.102, 0.109]  [0.102, 0.109]</td><td rowspan=1 colspan=1>[-0.640, -0.608]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[1.190, 1.309]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[0.862, 0.945]</td><td rowspan=1 colspan=1>[-0.003, 0.003]</td><td rowspan=1 colspan=1>[0.079, 0.085]</td><td rowspan=1 colspan=1>[0.063, 0.067]</td><td rowspan=1 colspan=1>[-0.000, 0.000]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[1.148, 1.234]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[0.906, 0.968]</td><td rowspan=1 colspan=1>[-0.059, -0.012]</td><td rowspan=1 colspan=1>[0.084, 0.090]</td><td rowspan=1 colspan=1>[0.073, 0.079]</td><td rowspan=1 colspan=1>[-0.180, -0.160]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[1.101, 1.193]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>0.844, 0.906]</td><td rowspan=1 colspan=1>[0.004, 0.056]</td><td rowspan=1 colspan=1>[0.080, 0.086]</td><td rowspan=1 colspan=1>[0.071, 0.076]</td><td rowspan=1 colspan=1>[-0.153, -0.117]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[1.133, 1.223]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>0.842, 0.924]</td><td rowspan=1 colspan=1>[-0.012, 0.055]</td><td rowspan=1 colspan=1>[0.074, 0.079]</td><td rowspan=1 colspan=1>[0.058, 0.062]</td><td rowspan=1 colspan=1>[0.061, 0.083]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>[0.954, 1.031]  [</td><td rowspan=1 colspan=1>0.684, 0.750]</td><td rowspan=1 colspan=1>[0.185, 0.232]</td><td rowspan=1 colspan=1>[0.065, 0.069</td><td rowspan=1 colspan=1>]  [0.050, 0.054]</td><td rowspan=1 colspan=2>[0.190, 0.216]</td></tr><tr><td rowspan=3 colspan=1>ERA5 Interp.Europe             COSMO Interp.(region hold-out)  Uninf. baseline</td><td rowspan=1 colspan=2>[1.430, 2.035]</td><td rowspan=1 colspan=1>[1.423, 2.017]</td><td rowspan=1 colspan=1>[-0.492, -0.260]</td><td rowspan=1 colspan=2>[0.152, 0.168]  [0.152, 0.168]</td><td rowspan=1 colspan=2>[-0.719, -0.622]</td></tr><tr><td rowspan=1 colspan=1>[1.049, 1.327]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[1.042, 1.315]</td><td rowspan=1 colspan=1>[-0.066, 0.171]</td><td rowspan=1 colspan=1>[0.132, 0.150]</td><td rowspan=1 colspan=1>[0.132, 0.151]</td><td rowspan=1 colspan=1>[-0.545, -0.400]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[1.451, 1.997]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[1.054, 1.460]</td><td rowspan=1 colspan=1>[0.000, 0.000]</td><td rowspan=1 colspan=1>[0.110, 0.126]</td><td rowspan=1 colspan=1>[0.090, 0.102]</td><td rowspan=1 colspan=2>[0.000, 0.000]</td></tr><tr><td rowspan=1 colspan=1>OAT (ours)</td><td rowspan=1 colspan=2>[0.956, 1.260]</td><td rowspan=1 colspan=1>[0.686, 0.917]</td><td rowspan=1 colspan=1>[0.277, 0.449]</td><td rowspan=1 colspan=1>[0.084, 0.097]</td><td rowspan=1 colspan=1>[0.064, 0.075]</td><td rowspan=1 colspan=2>[0.240, 0.307]</td></tr><tr><td rowspan=1 colspan=9>Table 6: Bootstrapped confidence intervals for Table 2. 95 % percentile intervals from a bootstrap over test stations (1000resamples; each station&#x27;s metric is averaged over the test period). MAE and CRPS intervals are marginal per model. Skill intervalsare paired: model and uninformed baseline are evaluated on the same resampled stations in each resample. Stations are treated asindependent, so the intervals do not account for spatial correlation.</td></tr></table>

<table><tr><td>Component</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>Feature trunk</td><td>Architecture Embedding dim.</td><td>as uninformed baseline, no output head 64</td></tr><tr><td>Mean function</td><td>MLP on embedding</td><td>64 → 64 → 1, ReLU</td></tr><tr><td rowspan="2">Stationary kernel</td><td>Type Inputs (CH)</td><td>RBF with ARD easting, northing, elevation</td></tr><tr><td>Inputs (EU)</td><td>longitude, latitude, elevation</td></tr><tr><td rowspan="2">Deep kernel (DK-GP only)</td><td>Lower bounds Type</td><td>length scale  $1 0 ^ { - 2 } .$  output scale  $1 0 ^ { - 2 }$  RBF on MLP features, fixed scales</td></tr><tr><td>MLP on embedding Combination</td><td> $6 4  6 4  6 4  4 .$  ReLU product with stationary kernel</td></tr><tr><td>Likelihood</td><td>Type Noise variance</td><td>Gaussian, one model per variable init. 1.0, lower bound  $1 0 ^ { - 3 }$ </td></tr><tr><td>Targets</td><td>Transform Precipitation</td><td>standardized before standardization</td></tr><tr><td>Trainable parameters (CH)</td><td></td><td> $\log ( y + 1 0 ^ { - 3 } )$   $\mathrm { G P } \approx 2 8 \mathrm { k } ,$  DK-GP ≈ 45k</td></tr></table>

Table 7: GP and DK-GP hyperparameters. Values are identical for both GP variants and for the Switzerland (CH) and Europe (EU) datasets except where noted. Kernel inputs are standardized. For the training configuration see Table 4.

<table><tr><td>Component</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="3">Latent grid</td><td>NWP stride</td><td>10 (CH) / 4 (EU)</td></tr><tr><td>Pseudo-token grid</td><td> $8 0 \times 1 2 0 ~ \mathrm { ( C H ) } ~ / ~ 4 0 \times 7 2 ~ \mathrm { ( E U ) }$ </td></tr><tr><td>NWP input channels</td><td>4 (CH) / 3 (EU)</td></tr><tr><td rowspan="2">Position encoding</td><td>Type</td><td>spherical harmonics (absolute lon./lat.)</td></tr><tr><td>Legendre polynomials / embedding dim.</td><td>10 / 128</td></tr><tr><td rowspan="2">Token encoders</td><td>Observation / NWP MLP</td><td>2 layers, width 128 5 (CH) / 2 (EU) → 32, batch norm</td></tr><tr><td>Topographic MLP</td><td></td></tr><tr><td rowspan="5">Transformer</td><td>Embedding dim.</td><td>128</td></tr><tr><td>Heads × head dim.</td><td>8 × 16</td></tr><tr><td>Layers</td><td>5</td></tr><tr><td>Feedforward dim.</td><td>128</td></tr><tr><td>Swin window / shift</td><td> $4 \times 4 ~ / ~ 2 \times 2$ </td></tr><tr><td>Grid decoder</td><td>Nearest pseudo-tokens per query (k)</td><td>9</td></tr><tr><td rowspan="2">Output head</td><td>Decoder MLP</td><td>2 layers, width 128</td></tr><tr><td>Distributions</td><td>as OAT (Normal / zero-inflated log-normal)</td></tr><tr><td colspan="2">Trainable parameters</td><td></td></tr></table>

Table 8: Gridded TNP hyperparameters. CH: Switzerland dataset, EU: Europe dataset. For the training configuration see Table 4.