# Thermally Conditioned Time-Series Forecasting of Grain Growth in Polycrystalline Materials

Eliane Younes, Elie Hachem, Marc Bernacki Mines Paris, PSL University Centre for material forming (CEMEF), UMR CNRS 06904 Sophia Antipolis, France

## Abstract

Grain growth during heat treatment strongly afects the microstructure of polycrystalline materials and, consequently, their mechanical properties. Although numerical methods such as front-tracking or front-capturing simulations can accurately describe grain growth, their high computational cost limits their use for long or nonisothermal thermal cycles. In previous work, grain growth was successfully predicted under isothermal conditions by treating grain size distributions as time series and forecasting their evolution using recurrent neural networks, allowing complete grain growth trajectories to be generated in seconds rather than through costly PDE-based simulations. However, such models cannot represent the strong influence of thermal history on grain boundary kinetics. In the present work, this forecasting framework is reformulated to handle non-isothermal conditions by explicitly incorporating thermal information into the model architecture. Grain size distributions are encoded using a convolutional neural network, thermal descriptors are embedded through a dedicated multilayer perceptron, and temporal evolution is predicted using a recurrent network. Instantaneous temperature and thermal rate are provided at each time step, allowing the model to adapt its predictions to changing thermal conditions. The framework is trained and evaluated on datasets generated from validated front-tracking simulations across a wide range of non-isothermal thermal paths. The results show that the thermally conditioned model accurately reproduces grain growth dynamics under complex thermal histories, achieving, in k-fold cross-validation over all simulations, mean relative errors of 2.99% (one-hour cycles) and 4.51% (four-hour cycles) with coeficients of determination exceeding 0.99, and mean relative errors between 1.91% and 14.66% on ten simulations following four unseen thermal histories of increasing complexity.

Keywords: Grain Growth, Deep Learning, Time Series Forecasting, Microstructural Evolution, Mean-Field Descriptors, Non-isothermal heat treatment

## 1 Introduction

Thermal processing leads to significant changes in the microstructure of polycrystalline materials, which in turn afect their mechanical behaviour. In this context, grain growth plays an important role [1–6], as it governs grain size evolution during annealing, which strongly influences mechanical properties such as flow stress and creep resistance [7, 8]. Accurate prediction of grain growth therefore remains a key objective in materials science, particularly in the context of heat treatment design [9].

At the mesoscale, grain growth in single-phase polycrystals is commonly described as a curvature-driven process, in which grain boundary migration results from the reduction of total interfacial energy [1, 3, 10]. In such kinetic descriptions, the normal velocity of a grain boundary can be reasonably approximated at high temperature and for polycrystal scale predictions as:

$$
\mathbf { v } = - { \boldsymbol { \mu } } \gamma \kappa \mathbf { n } ,\tag{1}
$$

where $\mu$ denotes the grain boundary mobility, $\gamma$ the grain boundary energy, κ the mean curvature, and n the outward unit normal to the interface.

Although simplified kinetic formulations have demonstrated strong predictive capability at the polycrystalline scale, grain boundary mobility is known to depend strongly on temperature, most often through Arrhenius-type relations [11]. It is therefore commonly expressed as:

$$
\mu = \mu _ { 0 } \exp \left( - { \frac { Q } { R T } } \right) ,\tag{2}
$$

where $\mu _ { 0 }$ is the pre-exponential factor, $Q$ the activation energy, R the gas constant, and $T$ the absolute temperature. Consequently, the evolution of a given microstructure is governed not only by its initial state, but also by the thermal history imposed during processing.

Numerical approaches such as Monte Carlo Potts methods [12, 13], cellular automata [14, 15], front-tracking [16–20], phase-field [21, 22], and level-set methods [23–26] have been extensively developed to simulate grain growth with high physical fidelity. These methods accurately capture the topology and kinetics of evolving grain boundary networks. However, because microstructural evolution must be resolved incrementally in time, such Partial Diferential Equation (PDE)-based simulations incur substantial computational cost, particularly for large systems and long thermal histories.

To reduce these limitations, reduced or mean-field representations have been developed as an alternative to full-field descriptions by representing the evolving microstructure through statistical descriptors rather than explicitly resolving the full spatial topology [27]. Although these approaches significantly decrease data dimensionality and computational requirements, the temporal prediction of microstructural evolution over extended time scales remains challenging.

In our previous study [28], we addressed these limitations by introducing a data-driven surrogate model for grain growth prediction under isothermal conditions. Grain size distributions extracted from validated front-tracking simulations ToRealMotion (TRM) [18, 29] were represented as evolving time series and predicted using recurrent neural networks. Using this formulation, the full grain growth trajectory could be generated from a short temporal history, with prediction accuracies on the order of 90% relative to reference simulations. Once trained, the surrogate produced complete grain growth evolutions in a few seconds, compared to simulation times on the order of tens of minutes for the underlying front-tracking model, corresponding to speedups exceeding two orders of magnitude. At the same time, physically consistent trends in the predicted grain-size distributions and mean grain size evolution were preserved.

Despite these results, the earlier surrogate formulation was developed and assessed only for constant temperature conditions. In practice, industrial heat treatments rarely follow strictly isothermal paths and instead involve non-isothermal thermal schedules composed of successive heating, holding, and cooling stages, often with variable heating rates. Since grain boundary mobility depends explicitly on temperature, identical initial microstructures subjected to diferent thermal histories may evolve along distinct trajectories. Such efects cannot be captured by models trained solely on isothermal data without explicit incorporation of thermal information.

The present work extends the previously proposed time-series forecasting framework to non-isothermal grain growth prediction. The proposed extension preserves the representation based on grain size distributions developed previously [28], while additional thermal descriptors are introduced to account for temperature-dependent grain growth kinetics. Instantaneous temperature and thermal rate are provided as conditioning variables, enabling the surrogate to adapt its predictions to evolving thermo-kinetic conditions while preserving the computational eficiency observed under isothermal training.

The extended framework is trained and evaluated using datasets generated with the TRM framework, whose grain growth predictions have been validated against EBSD measurements for nickel-based superalloys and stainless steels, under a wide range of non-isothermal thermal cycles. Several recurrent architectures are examined under identical training conditions, allowing a controlled comparison of their ability to propagate grain growth dynamics over time. In addition, diferent conditioning strategies are investigated to distinguish the respective influence of temperature, thermal rate, and their combined efect. Beyond standard train–validation–test evaluation, thermal histories excluded from training are considered to assess the robustness and generalization capability of the proposed surrogate under increasingly complex conditions.

By extending data-driven grain growth prediction from isothermal to non-isothermal regimes, this work provides a practical step toward surrogate models applicable to realistic heat treatment scenarios. The proposed approach ofers a computationally eficient alternative to high-fidelity simulations while maintaining physically meaningful predictive behaviour, with direct relevance for large-scale studies and process-level applications.

The paper is organized as follows. In Section 2, we describe the dataset generation procedure under non-isothermal conditions, along with the dataset organization, preprocessing steps, and the architecture of the thermally conditioned surrogate model. In Section 3, we present the prediction results and analyze the efects of diferent conditioning strategies, recurrent formulations, and generalization capabilities. We discuss the predictive behavior of the model and the impact of thermal conditioning on grain growth forecasting in Section 4. We summarize the main findings and conclude in Section 5.

## 2 Methodology

## 2.1 Dataset Generation

The initial polycrystalline microstructure states are generated following the same procedure as in the previous isothermal study [28], using the LaVoGen (Laguerre–Vorono¨ı tessellation) algorithm [30]. These microstructures are characterized by diferent domain sizes, mean equivalent circle radii, and grain size distribution standard deviations, representative of polycrystalline microstructures.

Grain growth evolutions are then simulated using the TRM front-tracking framework [18, 29], with each initial microstructure subjected to diferent non-isothermal thermal conditions during the TRM simulations. Within this framework, grain boundary migration is driven by curvature under the assumption of isotropic grain boundary energy. The efect of temperature on grain boundary kinetics is accounted for through the Arrhenius-type mobility relation introduced in Eq. (2). The parameters $\mu _ { 0 } , \ Q$ , and γ are chosen to be representative of a 304L stainless steel, ensuring realistic grain boundary behavior over the considered temperature range [26, 31]. Their numerical values are reported in Table 1.

For each initial microstructure, two categories of thermal cycles are applied to investigate both fast and slow thermo-kinetic regimes. Short (3600 s) cycles correspond to rapidly evolving thermal histories, whereas longer $( 1 4 4 0 0 ~ \mathrm { s } )$ cycles are associated with slower thermokinetic conditions through extended heating and cooling stages under identical peak temperature conditions. The complete set of thermal cycles is illustrated in Figs. 1a and 1b.

Each thermal cycle is described by a temperature profile composed of successive heating, holding, and cooling stages, with linear variations of temperature within each phase. The thermal evolution is characterized by the instantaneous temperature $T ( t )$ and the corresponding thermal rate $\dot { T } ( t )$ , shown in Figs. 1a and 1b. The thermal rate is computed from the temperature values of the neighbouring stored time steps (central diferences, one-sided at the first and last steps):

$$
\dot { T } ^ { t } = \frac { T ^ { t + d t } - T ^ { t - d t } } { 2 d t } .\tag{3}
$$

where dt is the time interval between two stored states (60 s after the temporal downsampling described in Section 2.2).

The temporal evolution is discretized using a constant time step, and microstructural states are stored at each step. Equivalent circle radius (ECR) are extracted and converted into normalized grain size histograms with a fixed number of bins (30 bins of equal width between 0 and a maximum size chosen just above the largest grain observed in each dataset: 0.11 mm for the one-hour cycles and 0.15 mm for the four-hour cycles, so that no grain is excluded), providing a compact statistical representation suitable for time-series forecasting. The thermal descriptors $T ( t )$ and $\dot { T } ( t )$ are used as conditioning variables for the learning framework.

The full set of parameters used for both LaVoGen microstructure generation and TRM grain growth simulations is summarized in Table 1.
<table><tr><td>Parameter</td><td>Value(s)</td><td>Description</td></tr><tr><td>a (mm)</td><td> $2 , 3 , 4 , 5$ </td><td>Side length of the square domain</td></tr><tr><td>Distribution</td><td> ${ \mathcal { N } } ( { \bar { R } } , \sigma ^ { 2 } )$ </td><td>Initial ECR distribution  $\left( \mathrm { L a V o G e n } \right)$ </td></tr><tr><td> $\bar { R } \ ( \mu \mathrm { m } )$ </td><td>20</td><td>Mean equivalent circle radius (ECR)</td></tr><tr><td> $\sigma ~ ( \mu \mathrm { m } )$ </td><td>2, 8, 16, 32</td><td>Standard deviation of the ECR distribution</td></tr><tr><td> $T ~ ( \mathrm { K } )$ </td><td>293.15-1323.15</td><td>Temperature range</td></tr><tr><td> $d T / d t$  heating  $\mathrm { ( K s ^ { - 1 } ) }$ </td><td>0.215-1.0</td><td>Heating rate range</td></tr><tr><td> $| d T / d t |$  cooling  $\mathrm { ( K s ^ { - 1 } ) }$ </td><td>0.05-10</td><td>Cooling rate range</td></tr><tr><td>dt (s)</td><td>10</td><td>Simulation time step</td></tr><tr><td> $t _ { \mathrm { s i m } } ~ \mathrm { ( m i n ) }$ </td><td>60, 240</td><td>Simulation duration</td></tr><tr><td>Storage frequency</td><td>1</td><td>Microstructure storage frequency</td></tr><tr><td> $Q \ \mathrm { ( J \ m o l ^ { - 1 } ) }$ </td><td>280 000</td><td>Activation energy</td></tr><tr><td> $\gamma \ ( \mathrm { J } \mathrm { m } ^ { - 2 } )$ </td><td>0.6</td><td>Grain boundary energy</td></tr><tr><td> $\mu _ { 0 } \ \mathrm { ( m ^ { 4 } J ^ { - 1 } s ^ { - 1 } ) }$ </td><td>0.156</td><td>Pre-exponential mobility constant</td></tr></table>

Table 1: Main parameters used for $\mathrm { L a V o G }$ en microstructure generation and TRM grain growth simulations under non-isothermal thermal cycles.

![](images/35f58d0c72a443b05b5b4483333b94764c9cc03300c03f30d82fe0f55ca13da2.jpg)

![](images/3d7592c4dcecd13e2b9eac63715c230bcb08c63fc6bf5887c9df25065c21d7ba.jpg)  
(a) One-hour thermal cycles (3600 s): temperature $T ( t )$ and thermal rate $\dot { T } ( t )$ for cases A1– A6.

![](images/5aa1ac861a2821b142ce57d1d28527f4999a451a03e72474085039f35f7152d9.jpg)

![](images/6bdbbabc066bdd9ae994625adf1d9b3af48e554364ec3f70b3bc43aee91ec9c0.jpg)  
(b) Four-hour thermal cycles (14400 s): temperature $T ( t )$ and thermal rate $\dot { T } ( t )$ for cases A1–A6.

Figure 1: Non-isothermal thermal histories considered in this study. Short-duration cycles (1 h) are used to investigate fast thermo-kinetic regimes, whereas long-duration cycles (4 h) are introduced to examine slower grain growth conditions. In each case, the thermal evolution is characterized by the instantaneous temperature $T ( t )$ and the corresponding thermal rate $\dot { T } ( t )$

## 2.2 Dataset Organization and Preprocessing

Each simulation is represented as a temporal dataset stored through a pair of aligned files: an Evolution file and a Property file. The Evolution file contains the normalized grain size distributions arranged in a matrix of size $( 3 0 \times N _ { t } )$ , where $N _ { t }$ is the number of stored time steps and each column consists of 30 values defining the grain size distribution at a given time step. The sequence of columns therefore captures the temporal evolution of the grain size distribution over the course of the simulation.

Following our previous work [28], the grain size distributions are normalized at each time step by dividing the number of grains in each bin by the total number of grains. Each bin therefore represents the relative frequency of grains within the corresponding size interval, and each distribution is described by 30 relative-frequency values.

The corresponding thermal information is stored in a Property file, which contains the instantaneous temperature $T ( t )$ and the thermal rate $\dot { T } ( t )$ , forming a matrix of size $( 2 \times N _ { t } )$

Although the TRM simulations are performed with a time step of $d t = 1 0 \mathrm { \ s }$ and microstructural states are stored at each step, the dataset used for training is temporally downsampled by retaining one state every six time steps. This results in an efective temporal resolution of 1 min for all learning experiments.

In this framework, the grain size distributions correspond to the quantities to be predicted, while the thermal descriptors serve as conditioning variables that provide the physical context of the non-isothermal evolution.

The one-hour and four-hour datasets, composed of 86 and 208 simulations respectively, are treated separately, and a dedicated model is trained for each of them. Model performance is evaluated by k-fold cross-validation at the sequence level, with $k = 1 0$ for the one-hour dataset and $k = 5$ for the four-hour dataset. In this way, each simulation is predicted by a model that did not use it during training, and all simulations contribute to the reported results. Since the partitioning is performed at the sequence level, all time steps of a given simulation remain in the same subset, which prevents information leakage between sequences.

Thermal descriptors are standardized using z-score normalization based exclusively on statistics computed from the training subset in order to improve numerical conditioning and training stability while avoiding information leakage between data partitions [32]:

$$
T _ { \mathrm { n o r m } } ^ { t } = \frac { T ^ { t } - \bar { T } } { s _ { T } } , \qquad \dot { T } _ { \mathrm { n o r m } } ^ { t } = \frac { \dot { T } ^ { t } - \overline { { \dot { T } } } } { s _ { \dot { T } } } ,\tag{4}
$$

where $\bar { T }$ and $\bar { \dot { T } }$ are the mean values of temperature and thermal rate computed from the training subset, and $s _ { T }$ and $s _ { \dot { T } }$ are the corresponding standard deviations. The resulting normalization parameters are subsequently applied unchanged to the validation and test subsets in order to ensure consistency and avoid data leakage [32].

Samples are constructed using a sliding-window strategy with a fixed window size of $W = 5$ corresponding to five consecutive time steps of the downsampled evolution. Each sample consists of the grain size histograms and associated thermal descriptors contained within the temporal window.

All experiments are conducted with fixed random seeds (seed 42) to ensure reproducibility of data splitting and model training.

## 2.3 Surrogate Model Architecture

The present study extends the time-series forecasting framework introduced in our previous work, where grain growth under isothermal conditions was modeled using sequences of grain size distributions and recurrent neural networks. In this earlier framework, the temporal evolution of the microstructure was learned exclusively from histogram-based representations obtained at constant temperature. While this approach proved efective for isothermal annealing, it does not explicitly capture the changes in grain growth kinetics associated with heating and cooling stages.

To address this limitation, the forecasting architecture is modified so that the evolving microstructural state and the applied thermal history are considered jointly. At each time step t, the model receives a temporal window of length W containing the past grain size distributions $x ( t - W ) , \ldots , x ( t - 1 )$ together with the corresponding thermal descriptors $T ( t - W ) , \dots , T ( t - 1 )$ and ${ \dot { T } } ( t - W ) , \dots , { \dot { T } } ( t - 1 )$ . Rather than predicting the subsequent distribution $x ( t )$ directly, the model predicts its variation with respect to the last known state, $x ( t ) = x ( t - 1 ) + \alpha \Delta ( t )$ , where $\Delta ( t )$ is the output of the network and $\alpha = 0 . 1$ is a fixed scaling factor. This residual formulation keeps the increments small at each step and stabilizes the auto-regressive rollout. The value $\alpha = 0 . 1$ was retained after comparison with the unscaled formulation $( \alpha = 1 )$ using the same cross-validation: over the 86 one-hour simulations, the MRE of the LSTM decreases from 3.56% to 2.99% (from 7.68% to 4.52% for the GRU), and the only simulation whose error exceeded 30% at some time step with $\alpha = 1$ no longer does. In addition, the distributions are processed in square-root space $( { \sqrt { x } } )$ , which reduces the weight of the dominant bins relative to the sparsely populated bins at the edges of the distribution. The prediction is then squared back to obtain the relative frequencies. During inference, predictions are propagated in an auto-regressive manner, while the future thermal conditions are assumed known and injected at each successive prediction step.

![](images/e1bcb448bd9b2c53b16ec0fd0b5d8cdeae0d38b4545f5231fc7bd9627e124ad7.jpg)  
Figure 2: Schematic representation of the thermally conditioned surrogate model for grain size distribution forecasting. A temporal window of histograms is encoded using a CNN, while the thermal descriptors $T ( t )$ and $\dot { T } ( t )$ are processed through a dedicated MLP. The resulting feature vectors are concatenated and passed to a recurrent temporal model (RNN, GRU, or LSTM). The output projection predicts the next grain size distribution, which is reused during auto-regressive rollout with known future thermal conditions.

The microstructural pathway relies on a one-dimensional convolutional neural network (1D-CNN) [33, 34] applied to the grain size histograms. In this representation, neighboring bins are inherently correlated, since peak displacement, distribution broadening, or local redistribution of grain populations generally afect adjacent size classes. One-dimensional convolutions therefore provide an eficient mechanism to capture short-range morphological dependencies along the grain size dimension. In the present implementation, two convolutional layers with 16 filters and a kernel size of 3 are employed, each followed by a LeakyReLU activation function with slope parameter $\alpha = 0 . 1$ . The resulting convolutional features are flattened and projected through a fully connected layer into a compact 32-dimensional latent embedding. A dropout rate of 0.2 is applied to this embedding to improve robustness and prevent overfitting, yielding a representation that characterizes the microstructural state at each time step.

In parallel, the thermal conditions are encoded using a dedicated Multi-Layer Perceptron (MLP) [35]. The instantaneous temperature $T ( t )$ and thermal rate $\dot { T } ( t )$ are standardized based on statistics computed from the training set and fed to two fully connected hidden layers with 16 units each. LeakyReLU activation functions with slope parameter $\alpha = 0 . 1$ are used, and a dropout rate of 0.2 is applied. A final fully connected layer then projects these features onto a 64-dimensional embedding that represents the local thermo-kinetic regime at each time step.

The latent representations extracted from the CNN and the MLP are concatenated to form a joint feature vector that combines morphological information with the thermal context. This fused representation is then supplied to the recurrent component of the model.

To identify the most suitable temporal predictor for non-isothermal grain growth forecasting, three recurrent architectures were considered: a Simple Recurrent Neural Network (RNN), a Gated Recurrent Unit (GRU), and a Long Short-Term Memory (LSTM) network. Comprehensive tutorials on these deep models are available in [35–37]. These architectures were evaluated under identical conditions with the objective of comparing their ability to propagate microstructural evolution under auto-regressive forecasting and selecting the most appropriate formulation for subsequent analyses. For all three variants, the same hyperparameter configuration was retained to ensure a fair comparison. Each recurrent architecture is implemented using three stacked recurrent layers with a hidden dimension of 128 units. All models are trained for at most 100 epochs using the Adam optimizer with an initial learning rate of $1 \times 1 0 ^ { - 3 }$ and a batch size of 32. The learning rate is halved when the validation loss stops decreasing for 4 epochs, and training is stopped after 10 epochs without improvement. These common settings allow diferences in predictive behavior to be attributed primarily to the recurrent formulation rather than to variations in training configuration.

Finally, the output of the recurrent block is projected through a dense layer onto the 30- bin normalized grain-size distribution. A ReLU activation function is applied to enforce non-negativity of the predicted frequencies, ensuring physically admissible outputs during auto-regressive prediction.

Model training is performed in a supervised manner to predict successive grain size distributions, using the Mean Absolute Error (MAE) computed on the square roots of the relative frequencies as the loss function [38]. To reduce the mismatch between training, where the inputs are reference distributions, and auto-regressive inference, where the inputs are the model’s own predictions, a pushforward training strategy is used: in addition to the next-step prediction, the model is unrolled over three successive steps using its own predictions as inputs, and the loss is evaluated at each of these steps. All models are optimized within the TensorFlow/Keras framework [39] using the Adam optimizer [40], with identical training settings applied to all recurrent architectures in order to enable a fair comparison. Hyperparameters were selected based on preliminary tests to ensure stable training behavior.

Prediction performance is evaluated by comparing the predicted and reference normalized grain-size distributions obtained after auto-regressive rollout. The error metrics are computed directly from the corresponding 30-bin frequency vectors. Performance is evaluated using the Mean Relative Error (MRE) and the coeficient of determination $R ^ { 2 } \ [ 4 1 , 4 2 ]$ . The MRE is defined as

$$
\mathrm { M R E } = \frac { 1 0 0 } { N } \sum _ { i : y _ { i } > 0 } \frac { | y _ { i } - \hat { y } _ { i } | } { y _ { i } } ,\tag{5}
$$

where $y _ { i }$ and $\hat { y } _ { i }$ are the reference and predicted relative frequencies and N is the number of

predicted values.

A schematic overview of the resulting architecture is presented in Fig. 2.

## 2.4 Conditioning Strategies

To investigate the role of thermal information, several conditioning strategies are evaluated using the same dataset and forecasting protocol:

• C0 – No T: no thermal descriptors are supplied, prediction relies only on the sequence of grain size distributions.

$\mathrm { C 1 } - T$ only: conditioning based solely on the instantaneous temperature $T ( t )$

$\mathrm { C 2 } - \dot { T }$ only: conditioning based solely on the thermal rate $\dot { T } ( t )$

$C 3 - T + { \dot { T } } \mathrm { ; }$ : simultaneous conditioning by temperature and thermal rate, corresponding to the complete proposed framework.

This comparison makes it possible to distinguish the respective influence of absolute temperature, transient thermal efects, and their combined contribution to grain growth prediction under non-isothermal conditions.

## 2.5 Generalization cases

In addition to the standard train–validation–test evaluation, four thermal histories not included in the learning dataset were considered to examine the response of the surrogate model under non-isothermal conditions. These cases represent thermal histories of increasing complexity, ranging from a simple reference cycle to multi-stage thermal paths involving successive heating and cooling stages.

Case 1: Baseline one-hour thermal cycle The first case represents a baseline heating– holding–cooling cycle with a total duration of one hour. The temperature increases from 293.15 K to 1323.15 K during the heating stage, followed by an isothermal holding period at 1323.15 K, and a final cooling stage down to 650.15 K. This case provides a reference nonisothermal scenario within a relatively simple thermo-kinetic regime. Its thermal history is shown in Fig. 3a.

Case 2: Extended four-hour thermal cycle The second case extends the thermal treatment to a four-hour cycle. The temperature increases from 293.15 K to 1323.15 K, followed by a prolonged isothermal holding stage, and a final cooling back to 293.15 K. Compared to the baseline case, this thermal cycle is characterized by a longer duration and an extended holding stage at high temperature. Its thermal history is shown in Fig. 3b.

Case 3: Multi-stage realistic thermal cycle The third case corresponds to a one-hour multi-stage thermal path. The temperature increases from 293.15 K to 1273.15 K and is held constant, then decreases to 1223.15 K during a short cooling stage and is held again, before being increased to 1323.15 K for a final holding stage. The cycle concludes with cooling to 293.15 K. Its thermal history is shown in Fig. 3c.

Case 4: Complex multi-stage non-monotonic thermal cycle The fourth case corresponds to a multi-stage non-monotonic thermal path involving successive heating, holding, cooling, and reheating stages. The temperature first increases from 293.15 K to 1273.15 K and is maintained during an initial holding stage. It is then reduced to 1123.15 K before being increased to 1323.15 K and held constant. A subsequent cooling stage lowers the temperature to 1273.15 K, followed by a final reheating stage back to 1323.15 K and a second holding period. The cycle concludes with cooling to 293.15 K. This non-monotonic profile was introduced to represent thermal treatments involving multiple successive heating and cooling stages over a broad temperature range. Its thermal history is shown in Fig. 3d. A variant of this cycle, in which the intermediate holding stage is performed at 1190.15 K instead of 1123.15 K, is also considered.

![](images/cce8f7d55807e39e770b834028935db56d4bc57c49b6e2d53edd8d2732bbb7b9.jpg)

![](images/10e0ef4361015576fd206f1863de3bd60e3919aa8454a66c9c67d227fb6f9edf.jpg)  
(a) Case 1: baseline one-hour thermal cycle.

![](images/332764608e55714919f1e01e3d55ed72d71f9d7652271688f15326d1129cae2a.jpg)

![](images/9129d0415d8ff5ffce652f7345b4dcccc4f5e0a7ef64e187e8ffd0b9d858544b.jpg)  
(b) Case 2: extended four-hour thermal cycle with a prolonged isothermal holding stage.

![](images/074c3ce0eec59ccf8fd6bd4219b1cb72b4d684c5b665ff39cee050e20a316d1e.jpg)

![](images/6bef67d7f4c085691cac9387fa14126f61697db07826c60a12dc021453979600.jpg)

(c) Case 3: multi-stage realistic thermal cycle.  
![](images/64ea66f54ab85c3f52db16321389aeb896d991082105a12f1d8674f7e88002f7.jpg)

![](images/827e6f6246ddf547b5e7b3f94652b1cc096bc0db93b924dd6f5afd9b7ead69e1.jpg)  
(d) Case 4: complex multi-stage non-monotonic thermal cycle.  
Figure 3: Thermal histories of the four generalization cases excluded from training. For each case, the temperature evolution (left) and the corresponding thermal rate (right) are reported throughout the cycle.

## 3 Results

## 3.1 Global prediction performance on unseen test sequences

The overall prediction performance of the diferent conditioning strategies is summarized in Table 2. The reported results correspond to averages computed over all simulations of the one-hour dataset, each simulation being predicted in the k-fold cross-validation by a model that did not use it for training or validation, and evaluated over the full prediction horizon (from t = 5 to $t = 5 9$ min).

Table 2: Global prediction performance averaged over all 86 one-hour simulations (k-fold cross-validation)
<table><tr><td>Configuration</td><td>MRE (%) ↓  $R ^ { 2 } \uparrow$ </td></tr><tr><td>C0 – No thermal input</td><td>45.26 0.8055</td></tr><tr><td>C1 – Temperature only (T) 3.10</td><td>0.9969</td></tr><tr><td>C2 – Thermal rate only  $( \dot { T } )$  4.53</td><td>0.9905</td></tr><tr><td>C3 – Temperature + thermal rate  $( T + { \dot { T } } )$  2.99</td><td>0.9970</td></tr></table>

When no thermal information is provided to the model (C0), considerably larger prediction errors are obtained for all non-isothermal test sequences. The unconditioned configuration reaches an MRE of 45.26% and an $R ^ { 2 }$ of 0.8055, clearly underperforming the thermally conditioned configurations. For 66 of the 86 simulations, the error exceeds 30% at some point of the rollout.

Introducing either temperature (C1) or thermal rate (C2) substantially improves prediction accuracy compared with C0. Temperature conditioning leads to an MRE of 3.10% and an $R ^ { 2 }$ of 0.9969, while thermal-rate conditioning gives an MRE of 4.53% and an $R ^ { 2 }$ of 0.9905.

The best performance is obtained when temperature and thermal rate are provided simultaneously (C3), with an MRE of 2.99% and an $R ^ { 2 }$ of 0.9970. Over the 86 simulations, the MRE of individual simulations ranges from 0.78% to 8.17%. Among the four conditioning strategies, C3 therefore provides the closest agreement with the reference predictions.

## 3.2 Influence of the recurrent formulation within the hybrid framework

The efect of the recurrent formulation is examined in Table 3. In all cases, the recurrent block (Simple RNN, GRU, or LSTM) is embedded within the same hybrid architecture, combining a CNN for grain size distribution encoding and an MLP for thermal descriptor processing.

Table 3: Comparison of recurrent formulations within the hybrid architecture $( \mathrm { C N N + M L P }$ + recurrent).
<table><tr><td>Recurrent block</td><td>MRE (%) ↓</td><td> $R ^ { 2 } \uparrow$ </td></tr><tr><td>Simple RNN</td><td>10.33</td><td>0.9812</td></tr><tr><td>GRU</td><td>4.52</td><td>0.9935</td></tr><tr><td>LSTM</td><td>2.99</td><td>0.9970</td></tr></table>

The Simple RNN shows notably higher prediction errors than the two gated architectures, with an MRE of 10.33% and an $R ^ { 2 }$ of 0.9812. For 14 of the 86 simulations, its error exceeds 30% at some point of the rollout. In contrast, both the GRU and LSTM provide substantially lower errors and maintain a high agreement with the reference evolutions. Among the three recurrent formulations, the LSTM achieves the best overall performance, with an MRE of 2.99% and an $R ^ { 2 }$ of 0.9970, compared with 4.52% and 0.9935 for the GRU, respectively.

The temporal evolution of the errors of the three recurrent formulations is shown in Fig. 4. The error of the Simple RNN increases from the beginning of the heating stage and shows strong oscillations, whereas the GRU and the LSTM remain close during the first part of the cycle. The gap between the GRU and the LSTM then increases progressively: at the end of the cycle, the mean MRE reaches about 19% for the Simple RNN and 9.5% for the GRU, compared with 7.2% for the LSTM, and the mean $R ^ { 2 }$ is about 0.93, 0.975 and 0.984, respectively. The advantage of the LSTM therefore lies mainly in its ability to limit the accumulation of errors during the auto-regressive rollout.

## 3.3 Contribution of the hybrid architecture components

To isolate the contribution of the architectural components beyond the recurrent block itself, Table 4 compares two LSTM-based configurations under identical thermal conditioning $( T + { \dot { T } } )$

In the LSTM only configuration, grain size distributions and thermal descriptors are injected directly into the LSTM without prior processing. In the hybrid configuration, the grain size distributions are first encoded by the CNN and the thermal descriptors by the MLP, and the resulting features are combined before the LSTM.

Table 4: Impact of architectural components under identical thermal conditioning (one-hour dataset, k-fold cross-validation).
<table><tr><td>Configuration</td><td>CNN</td><td>MLP</td><td>MRE  $( \% ) \downarrow$ </td><td>Max. MRE (%) ↓</td><td> $R ^ { 2 } \uparrow$ </td></tr><tr><td>LSTM only</td><td>一</td><td>一</td><td>3.80</td><td>10.75</td><td>0.9942</td></tr><tr><td> $\mathbf { C N N } + \mathbf { M L P } + \mathbf { L S T M }$ </td><td>√</td><td>V</td><td>2.99</td><td>8.17</td><td>0.9970</td></tr></table>

The comparison shows that encoding the grain size distributions with the CNN and the thermal descriptors with the MLP before the LSTM improves the prediction performance. The MRE decreases from 3.80% to 2.99%, the largest error among the 86 simulations decreases from 10.75% to 8.17%, and the $R ^ { 2 }$ increases from 0.9942 to 0.9970. In addition, two simulations exceed an MRE of 30% at some point of the rollout with the LSTM-only configuration, compared with none for the hybrid architecture.

![](images/f56576af8142e74359ed31c89deaf8ad93011be6ea0ddaac486129fceb30e8b0.jpg)  
(a)

![](images/b6586dfb19de80468894b5cbb0f90c9a9e982a17ca4e6fdbb2f9ca326d2ffa21.jpg)  
(b)

![](images/8d38ef625752c212b2fe3ec0e35ba981ab6d482bc667b30eae4110c824ee678b.jpg)  
(c)  
Figure 4: Comparison of the recurrent formulations within the hybrid architecture (CNN + MLP + recurrent block) over the 86 one-hour simulations: mean (a) MAE, (b) MRE and (c) $R ^ { 2 }$ over time.

## 3.4 Phase-wise error analysis

To better understand how prediction errors are distributed across diferent thermo-kinetic regimes, the Mean Relative Error is evaluated separately for the heating phase, the isothermal plateau, and the cooling phase. The results, averaged over all test sequences, are reported in Table 5.

Table 5: Phase-wise Mean Relative Error (averages over the 86 one-hour simulations).
<table><tr><td>Configuration</td><td>Heating ↓</td><td>Plateau↓</td><td>Cooling ↓</td></tr><tr><td>C0 – No thermal input</td><td>41.26</td><td>39.78</td><td>56.50</td></tr><tr><td>C1 − T only</td><td>0.48</td><td>3.76</td><td>5.61</td></tr><tr><td> $\mathrm { C 2 \mathrm { ~ - ~ } } \dot { T } \ \mathrm { o n l y }$ </td><td>1.44</td><td>5.34</td><td>7.48</td></tr><tr><td> $\mathbf { C 3 } - T + \dot { T }$ </td><td>0.44</td><td>3.59</td><td>5.49</td></tr></table>

Clear diferences are observed between the three thermal stages. For the thermally conditioned configurations, the errors remain relatively low during heating and increase during the isothermal plateau and cooling. This increase is most pronounced for C2, with an MRE of 7.48% during cooling, compared with 5.61% for C1. In contrast, C3 maintains the lowest error throughout the three stages, with MRE values of 0.44%, 3.59%, and 5.49% during heating, holding, and cooling, respectively.

## 3.5 Temporal evolution and physical consistency of the selected model

Following the global and phase-wise performance analyses, the temporal behavior of the selected model is examined over the full prediction horizon. The joint conditioning strategy (C3) combined with the LSTM recurrent architecture is considered in more detail, as it provided the most accurate predictions in the previous analyses.

Figure 5(a) reports the temporal evolution of the sum of predicted relative frequencies over the 86 simulations (mean, minimum and maximum). The reference distributions remain equal to unity by construction, while the predicted values remain close to one throughout the prediction horizon. The predicted distributions therefore remain close to the expected normalization during the auto-regressive rollout.

The temporal evolution of the Mean Absolute Error and Mean Relative Error is presented in Figs. 5(b) and $5 ( \mathrm { c } )$ , respectively. Both error measures increase progressively as the prediction horizon extends, the mean MRE reaching about 7% at the end of the cycle. The maximum envelope shows occasional peaks at the end of the rollout, which remain below 30% for all simulations. These peaks occur mainly for the smallest domains with the widest initial distributions (2 mm, $\sigma = 3 2 ~ \mu \mathrm { m } )$ , which contain about 500 grains at the end of the simulation: several bins then contain only a few grains, so that a diference of a single grain produces a large relative error in that bin.

The evolution of the coeficient of determination is shown in Fig. 5(d). Although a gradual decrease in $R ^ { 2 }$ is observed as the prediction progresses, the values remain high over the full sequence, indicating that the predicted trajectories retain a strong correlation with the reference evolution, including at later stages.

To further illustrate the temporal behavior of the selected model, Fig. 6 shows the evolution of the normalized grain-size distribution for a representative sequence (the simulation with the median MRE, 2.71%). The selected snapshots correspond to the initial state, an early prediction step, the end of heating, the end of the isothermal holding stage, and the final state after cooling. The progressive displacement of the distribution toward larger grain sizes is observed throughout the thermal cycle and remains consistent with the reference evolution. From the first predicted step to the final cooling stage, the main characteristics of the distribution are maintained, including the location of the dominant peak and the overall spread of grain sizes. Small diferences become visible at later times, particularly for the largest grain classes. These diferences remain limited compared with the overall evolution of the distribution.

The errors of the individual simulations are shown in Fig. 7, where the 86 simulations are sorted by increasing MRE. The MRE remains below 5% for 80 of the 86 simulations and below 8.2% for all of them, and the $R ^ { 2 }$ is higher than 0.99 for 82 simulations. The three largest errors are obtained for F32, F36 and F74, which all correspond to the widest initial distributions $( \sigma _ { 0 } = 3 2 \mu \mathrm { m } )$ in the smallest domains (2 and 3 mm). These simulations contain few grains at the end of the cycle (about 500 to 1100, compared with a median of about 3200 over the dataset), so that several bins contain only a few grains and the relative error of these bins increases. The mean values reported in Table 2 are therefore representative of the whole dataset and are not driven by a few well-predicted simulations.

![](images/d79da613b5cf54447262c4901f6928efa5f9167bfb6768dc3a9bc5c4864395d7.jpg)  
(a)

![](images/0f58f3bedd93c047fcf0dd4e7de1d4e6f4578980c0da9a496e73353bc69863ce.jpg)  
(b)

![](images/618cce3e756877db4ff3115193694ba2d5b5e4d3960de61e6a3b596f0bf9224b.jpg)  
(c)

![](images/d2d58d74c681e7d51e4fe50f4fe014a47908fbd6f61294170096252d254500ca.jpg)  
(d)  
Figure 5: Temporal evolution of forecasting accuracy and physical consistency for the selected hybrid CNN–MLP–LSTM model (C3) over the 86 one-hour simulations: mean (line) and minimum–maximum envelope (shaded area). (a) Sum of predicted relative frequencies. (b) Mean Absolute Error (MAE) over time. (c) Mean Relative Error (MRE) over time. (d) Coeficient of determination $( R ^ { 2 } )$ over time.

## 3.6 Four-hour thermal cycles

The same model, trained and evaluated in the same way on the four-hour dataset (208 simulations, $k \ : = \ : 5 )$ , gives an MRE of 4.51% (from 2.07% to 8.20% depending on the simulation) and an $R ^ { 2 }$ of 0.9975. No simulation exceeds an MRE of 30% at any time step of the 236 predicted steps. As for the one-hour cycles, the error is lowest during heating (0.27%) and highest during cooling (9.44%), with 4.00% during the holding stages (Fig. 8). The evolution of the grain-size distribution of a representative four-hour simulation (the simulation with the median MRE) is shown in Fig. 9: the predicted distributions follow the reference ones over the 236 predicted steps.

The errors of the 208 four-hour simulations are shown individually in Fig. 10. The MRE remains below 5% for 150 simulations and below 8.2% for all of them, and the $R ^ { 2 }$ is higher than 0.992 for every simulation. In contrast with the one-hour cycles, the errors depend only weakly on the initial distribution: the mean MRE is 4.41%, 4.23%, 4.76% and 4.63% for $\sigma _ { 0 } = 2$ , 8, 16 and 32 $\mu \mathrm { m }$ , respectively. The largest MRE values are obtained for narrow initial distributions $( \sigma _ { 0 } = 2 \ \mu \mathrm { m } )$ , whose $R ^ { 2 }$ nevertheless remains above 0.998, while the lowest $R ^ { 2 }$ values correspond to the widest initial distributions $( \sigma _ { 0 } = 3 2 \ \mu \mathrm { m } )$ .

![](images/73de0ff4a00989929ff77614b7061cde063a902e311b934923cbfefd5b6b1a2a.jpg)  
(a)

![](images/fc30078e75575284ff40a9f53250626210aa262e815d9dde0c6350ae9d65f7eb.jpg)  
(b)

![](images/bb6efc8a9d9c5df2e59207ea93cdee2c8ebfe415ac9d594ded34b735195442d9.jpg)  
(c)

![](images/c806f44e44a922b27d43355794b424dcb39e59382561dac30fbca9dbd19760ea.jpg)  
(d)

![](images/8834001b77b748425a7ccca4ccfeaf79c3f681b2ff881afa8e895f4288cce3d8.jpg)  
(e)  
Figure 6: Comparison of predicted (CNN–MLP–LSTM) and TRM normalized grain-size distributions for a representative test sequence at selected prediction steps: (a) the initial state, (b) the first predicted time step, (c) an intermediate stage of the thermal cycle, (d) a later stage of grain growth, and (e) the final state of the sequence.

![](images/1cdffae261d2cd577e84785e20e5aa2ff58636e68aeb9d3cf276fd0dc3c6cd15.jpg)  
Figure 7: MRE (top) and $R ^ { 2 }$ (bottom) of each of the 86 one-hour simulations (k-fold crossvalidation), sorted by increasing MRE and coloured by the standard deviation $\sigma _ { 0 }$ of the initial distribution. The dashed lines give the mean values.

![](images/d357e2f3c62258098c05e45b81476d4d7eee3075cb7388f695faa0c0d53c7991.jpg)  
(a)

![](images/0f86fc7b91bab6857b83b745c5ffeddbbadeb53e77e53242a780166b1ff820d0.jpg)  
(b)  
Figure 8: Four-hour thermal cycles: (a) MRE and (b) $R ^ { 2 }$ over time for the 208 simulations (k-fold cross-validation): mean (line) and minimum–maximum envelope (shaded area).

![](images/d574214b6cd3104037e8a9d9c1d8a5a067c932ececd752f7fd010c50f6bf9e98.jpg)  
(a)

![](images/26a7f64a97a5ccfd35bf63b3bd871222757d4910dad2ae66a2ebdf63f52ebc8c.jpg)  
(b)

![](images/9ff69e354761bdc129a54547466986c1e2fbebbfb27c6f15a16f6547f71bd0f1.jpg)  
(c)

![](images/de9b35029ccc2d3c31656a211921308fc5e443e51248989bbc0d19f11aef3719.jpg)  
(d)

![](images/2eac54c03ddbc033b5c7e2ab3c3fba101f66a95a3ba0028f44ee83f1a2c874ec.jpg)  
(e)

![](images/60ab58f8e3720b199985cdf8089761bea4b074907facab4144041ccc6037263c.jpg)  
(f)  
Figure 9: Four-hour thermal cycles: comparison of predicted (CNN–MLP–LSTM) and TRM normalized grain-size distributions for a representative simulation at (a) the initial state, (b) the first predicted time step, (c) t = 60 min, (d) t = 120 min, (e) t = 180 min, and (f) the final state (t = 240 min).

![](images/06f9ab20d81aa20f61bcb541197b79536c1bf8d14c805714efa2a99797bf1f4f.jpg)  
simulations (sorted by MRE)  
Figure 10: MRE (top) and $R ^ { 2 }$ (bottom) of each of the 208 four-hour simulations (k-fold cross-validation), sorted by increasing MRE and coloured by the standard deviation $\sigma _ { 0 }$ of the initial distribution. The dashed lines give the mean values.

## 3.7 Generalization to unseen thermal histories

The generalization capability of the proposed CNN–MLP–LSTM framework is assessed using four thermal histories explicitly excluded from the training and validation datasets. For these tests, the model is trained once on all simulations of the corresponding dataset (one-hour dataset for Cases 1, 3 and 4, four-hour dataset for Case 2), and the first five minutes of each unseen simulation are given before the auto-regressive rollout. These cases involve temperature levels and temporal paths not previously encountered by the model and progressively introduce longer and more complex thermal cycles. For each unseen cycle, all available simulations (diferent initial distributions $\sigma _ { 0 }$ on the same thermal path) are predicted, and the results are reported in Table 6. The distributions shown in the following figures correspond to the simulations marked with an asterisk.

Table 6: Prediction performance on the ten simulations of the four unseen thermal cycles. The simulation shown in the corresponding figure is marked with an asterisk; <sup>†</sup>: intermediate holding stage at 1190.15 K instead of 1123.15 K.
<table><tr><td>Case</td><td>Model a (mm)</td><td></td><td> $\sigma _ { 0 } ~ ( \mu \mathrm { m } )$ </td><td>MRE (%) ↓</td><td> $R ^ { 2 } \uparrow$ </td></tr><tr><td>Case 1 – one-hour cycle</td><td>1 h</td><td>3</td><td> $1 6 ^ { * }$ </td><td>1.91</td><td>0.9988</td></tr><tr><td rowspan="3">Case 2 – four-hour cycle</td><td>4 h</td><td>5</td><td>8*</td><td>10.40</td><td>0.9922</td></tr><tr><td></td><td></td><td>16</td><td>14.66</td><td>0.9825</td></tr><tr><td></td><td></td><td>32</td><td>12.70</td><td>0.9705</td></tr><tr><td rowspan="4">Case 3 – multi-stage cycle</td><td>1 h</td><td>3</td><td>2</td><td>8.84</td><td>0.9919</td></tr><tr><td></td><td></td><td>8*</td><td>6.82</td><td>0.9916</td></tr><tr><td></td><td></td><td>16</td><td>7.36</td><td>0.9815</td></tr><tr><td></td><td></td><td>32</td><td>11.17</td><td>0.9404</td></tr><tr><td>Case 4 – non-monotonic cycle</td><td>1 h</td><td>3</td><td>8*</td><td>6.09</td><td>0.9930</td></tr><tr><td>Case 4 – variant†</td><td>1 h</td><td>3</td><td>8</td><td>5.73</td><td>0.9922</td></tr></table>

## 3.7.1 Case 1: One-hour unseen thermal cycle

For the first unseen thermal cycle, the predicted distributions remain in close agreement with the reference distributions at the selected prediction steps (Fig. 11). The displacement of the dominant peak toward larger grain sizes is reproduced throughout the cycle, with only small diferences appearing in the tails of the distribution at the final stage. The model reaches an MRE of 1.91% and an $R ^ { 2 }$ of 0.9988 (Table 6).

## 3.7.2 Case 2: Four-hour unseen thermal cycle

The second case extends the prediction to a four-hour thermal cycle. Despite the longer prediction horizon, the predicted distributions continue to follow the reference evolution during heating, holding, and cooling (Fig. 12). Diferences become more visible at the later stages, particularly for the largest grain-size classes. An MRE of 10.40% and an $R ^ { 2 }$ of 0.9922 are obtained with the model trained on the four-hour dataset. This cycle also involves a domain size (5 mm) that is not present in the four-hour training data. For the two other initial distributions of this cycle $( \sigma _ { 0 } = 1 6 $ and 32 µm), the MRE reaches 14.66% and 12.70%, with $R ^ { 2 }$ values of 0.9825 and 0.9705 (Table 6).

## 3.7.3 Case 3: Multi-stage realistic thermal cycle

Case 3 considers a multi-stage thermal cycle involving successive heating, holding, partial cooling, and reheating stages. The predicted distributions follow the reference distributions across these successive thermal transitions (Fig. 13). In particular, the changes in the position and shape of the distribution after cooling and subsequent reheating are reproduced by the model. An MRE of 6.82% and an $R ^ { 2 }$ of 0.9916 are obtained for this case; the three other initial distributions of the same cycle give MRE values between 7.36% and 11.17%, the largest error corresponding to the widest initial distribution $( \sigma _ { 0 } = 3 2 \ \mu \mathrm { m } , R ^ { 2 } = 0 . 9 4 0 4 )$ (Table 6).

## 3.7.4 Case 4: Multi-stage non-monotonic thermal cycle

Case 4 considers a more complex non-monotonic thermal cycle with multiple changes in the imposed thermal conditions. The predicted distributions follow the main changes observed in the reference distributions throughout the successive stages (Fig. 14). Larger diferences appear toward the end of the cycle, particularly in the largest grain-size classes. An MRE of 6.09% and an $R ^ { 2 }$ of 0.9930 are obtained. The largest errors occur shortly after the end of the first heating stage and after the last reheating, when the grain growth rate changes abruptly. The variant of this cycle with an intermediate holding stage at 1190.15 K gives an MRE of 5.73% and an $R ^ { 2 }$ of 0.9922 (Table 6).

![](images/6cb032f470be8a5e34c004006ecc999462775079ed1f6cffd7307b51f03737e8.jpg)  
(a)

![](images/a2f26c7a934614816a2d200e0011a82f2ccdec7ce886d69b06ee88f89a7003b5.jpg)  
(b)

![](images/668f24d00b720a26ab33e6484a48838bb9400272dddbeecfb9dbddc0f9e209ab.jpg)  
(c)

![](images/0d022d08f7e973829c905cb0d4e9c85cbe16e975185a8f351428cc043095584f.jpg)  
(d)

![](images/1bfcfbc8a01ac4391dfa760a474d6cae05127f5a86af71490f278b7d61ebf5b4.jpg)  
(e)  
Figure 11: Case 1: Comparison of reference (TRM) and predicted (CNN–MLP–LSTM) normalized grain-size distributions for a representative one-hour unseen thermal cycle at (a) the initial state, (b) the first predicted time step, (c) the end of heating, (d) the end of the isothermal holding stage, and (e) the final state after cooling.

![](images/d8cecd209b975d8e0986c5b31236c7fb436f42dbd8a0205fcec98480f7f03402.jpg)  
(a)

![](images/aac098a318d071d09709e58512dab24d10ee5d01bfdcceba18cfa22182198e33.jpg)  
(b)

![](images/870d431a5ab281793959e1381a2e45de610cd90717407038175225a75a51ba94.jpg)  
(c)

![](images/632ed3d7004d46e4f391b03ef0f519ccf2db9fc167cf7d9523d7936f5f85d190.jpg)  
(d)

![](images/56e4cc5f0811187c32a3a7b6cb720a962dbfa737d202e045ac5ef5e00314b149.jpg)  
(e)  
Figure 12: Case 2: Comparison of reference (TRM) and predicted (CNN–MLP–LSTM) normalized grain-size distributions for a representative four-hour unseen thermal cycle at (a) the initial state, (b) the first predicted time step, (c) the end of heating, (d) the end of the extended isothermal holding stage, and (e) the final state after cooling.

![](images/26f5f48f115e55ded0b9699e3bf73ca6394ecdc4b944951a062c3b25275c02eb.jpg)  
(a)

![](images/c37e2ba6b6d09cd809130464b2cc5b18a4e9c79834000925613fda370f25f033.jpg)  
(b)

![](images/c7fe4234a4839b64d2efb2dc6626d696a39e3c09f0d4d6a89b0dd2375cd31599.jpg)  
(c)

![](images/4472d057983abcd56a814bf26d6ed36cf28cdb2225260ca0d2cfb90ad5f59a7a.jpg)  
(d)

![](images/d7421e91b0557fca95e5fb69b2eb02fe501026f1e237d4ab8b3b0511b8aa777e.jpg)  
(e)

![](images/7312db0e29b3d501c5e9e96bbbb6566298f3b066c7a8431cd083243bf86a97a2.jpg)  
(f)  
Figure 13: Case 3: Comparison of reference (TRM) and predicted (CNN–MLP–LSTM) normalized grain-size distributions for a representative multi-stage unseen thermal cycle at (a) the initial state, (b) after the first isothermal plateau, (c) after partial cooling, (d) following reheating and holding, (e) at the end of the second heating stage, and (f) after the final plateau.

![](images/72c41d11d4d99304865c8016397bd94e46ea7ba6927b163e7456bfc7ce809f1c.jpg)  
(a)

![](images/c030ef6c422fb591ce23df73b23f85bbdd1ef93abac520b257144d83faf8fd67.jpg)  
(b)

![](images/b1ef392a3f8d6d6dfa70eecc00e0be2215958f81de3df1bdd8af4f0e11647a95.jpg)  
(c)

![](images/e84e7f68e6ea4c501911a17bbb7011bf59e0227364ed085ba88ca12186ccc964.jpg)  
(d)

![](images/563c84bcc6472f47b5427881ba19f7bc2dfce965db2b395a2a62f9cfeb140373.jpg)  
(e)

![](images/cf9fee7b71a40f132726fb3c51f5e28e92fc5ccb0d9fdd58a821c05adaa6daef.jpg)  
(f)

![](images/f002134f85527385c52e20b597012804f603d77a0a4d6a0d31ba80aa4f8b700e.jpg)  
(g)

![](images/3be7950640be0ee2fda5857ecad1291553e4d063b900f336796c6d2062733749.jpg)  
(h)  
Figure 14: Case 4: Comparison of reference (TRM) and predicted (CNN–MLP–LSTM) normalized grain-size distributions for a representative multi-stage non-monotonic thermal cycle at (a) the initial state, (b) the first predicted time step, (c) the end of the first heating stage, (d) after the first holding stage, (e) after reheating and holding at high temperature, (f) following the subsequent holding stage, (g) at the end of the final heating stage, and (h) after the final holding stage.

## 4 Discussion

This work extends a time-series forecasting framework for grain growth, previously limited to isothermal conditions, to non-isothermal regimes representative of industrial heat treatments. The results clearly demonstrate that explicitly conditioning the surrogate model with thermal information is indispensable to accurately capture thermo-kinetic efects on microstructural evolution.

The poor performance obtained without thermal conditioning shows that the grain-size history alone does not provide suficient information to describe non-isothermal grain growth (Table 2). Since grain boundary mobility depends on temperature, similar grain-size distributions may evolve diferently depending on the imposed thermal history. Thermal information is therefore needed to account for these changes in grain growth kinetics.

The improvement obtained when either temperature or thermal rate is introduced shows that both descriptors carry relevant information on the thermal history. Temperature provides information on the current thermal state, whereas the thermal rate describes how this state is changing with time. Their respective contributions therefore difer, which motivates their combined use in C3.

Combining both descriptors provides the model with information on the current temperature and its variation during the thermal cycle. Temperature directly influences grain boundary mobility through the Arrhenius relation, while the thermal rate provides information on the evolution of the imposed thermal conditions. Their combined use therefore allows the model to better account for changes in grain growth kinetics during heating, holding, and cooling, leading to improved prediction performance.

Beyond thermal conditioning, the recurrent formulation also plays an important role in the prediction of grain growth evolution. The clear improvement obtained with the gated recurrent architectures can be related to their ability to retain information over longer temporal sequences. This is particularly important in the present framework, where each prediction depends on the preceding microstructural states and the associated thermal history. The poorer behavior of the Simple RNN suggests that this information is not retained effectively over long auto-regressive sequences. In contrast, the memory mechanisms of the GRU and LSTM are better suited to follow the accumulated efects of the thermal history. The slightly better performance of the LSTM may be attributed to its dedicated cell state, which provides an additional mechanism for preserving relevant information as the prediction progresses (Table 3).

The processing of the microstructural and thermal inputs before the recurrent block also influences the prediction performance. Providing both inputs directly to the LSTM results in higher prediction errors. When the grain-size distributions are encoded by the CNN and the thermal descriptors by the MLP before their features are combined and passed to the LSTM, both the mean and the largest errors decrease (Table 4). This separate encoding of the microstructural and thermal information therefore provides a more suitable representation for temporal prediction.

The prediction accuracy also varies across the diferent stages of the thermal cycle. The relatively low errors obtained during heating indicate that the conditioned models follow the evolution of grain growth as the temperature and grain boundary mobility increase. Larger errors are observed during the isothermal plateau, where the temperature remains constant while the microstructure continues to evolve, allowing prediction errors to accumulate progressively over successive auto-regressive steps. The errors increase further during cooling, as the decrease in temperature reduces grain boundary mobility and progressively slows down the grain growth kinetics. The larger errors obtained with C2, and to a lesser extent with C1, during cooling show that using either temperature or thermal rate alone is less efective in this regime. In contrast, the lower error maintained by C3 shows the benefit of combining both descriptors when the thermal conditions change and the grain growth kinetics decelerate (Table 5).

The temporal analysis of the selected C3 model also shows how the predictions evolve during the auto-regressive rollout. The preservation of the distribution normalization and the continuous increase in the predicted mean grain size show that the expected grain growth behavior is maintained throughout the prediction. The evolution of MAE, MRE, and $R ^ { 2 }$ also shows a progressive accumulation of error over successive auto-regressive steps, without signs of instability. Despite this accumulation, the predicted distributions continue to follow the displacement of the grain-size population toward larger sizes. These results show that the model reproduces the main trends of grain growth over the full prediction horizon despite the accumulation of forecasting errors (Figs. 5 and 6).

The generalization tests consider thermal histories that were not included during training or validation. The good agreement obtained for the first case shows that the model can predict grain growth for an unseen thermal cycle, while the second case shows that good predictions are also obtained over a longer prediction period. The multi-stage cases are more dificult, since the model has to follow several changes in the thermal conditions rather than a single heating–holding–cooling sequence. The higher error obtained for the most complex case shows that the prediction becomes more dificult as the thermal history becomes more variable. However, over the ten simulations of the four unseen cycles, the MRE remains between 1.91% and 14.66% and the $R ^ { 2 }$ above 0.94 (above 0.99 for six of them), showing that the main evolution of the grain-size distributions is still reproduced. These results show that the proposed framework can be applied to thermal histories outside those used for model training, including longer and non-monotonic thermal cycles (Table 6).

## 5 Conclusion

This work extends the previously developed time-series forecasting framework for grain growth from isothermal to non-isothermal conditions. Temperature and thermal rate are introduced together with the grain-size distributions to account for the efect of the thermal history on grain growth evolution.

The results show that thermal information is needed to predict grain growth under nonisothermal conditions. When no thermal information is provided, the prediction error remains high. The use of either temperature or thermal rate considerably improves the predictions, while their combined use gives the best performance, with an MRE of 2.99% and an $R ^ { 2 }$ of 0.9970 over the 86 one-hour simulations, and 4.51% and 0.9975 over the 208 four-hour simulations (Table 2). The phase-wise analysis also shows that this configuration gives the lowest errors during heating, holding, and cooling (Table 5).

We also compare diferent recurrent formulations within the same architecture. The Simple RNN gives significantly larger prediction errors, while GRU and LSTM perform better. Among them, the LSTM gives the lowest error (Table 3). The comparison of the architectural components also shows that the best results are obtained when the CNN and MLP are both used before the LSTM (Table 4). In this configuration, the grain-size distributions and thermal descriptors are processed separately before temporal prediction.

The temporal evolution of the predictions shows that the error progressively increases during the auto-regressive rollout (Fig. 5). This is expected since each predicted distribution is used again to predict the following state, allowing prediction errors to accumulate with time. The residual formulation, the square-root representation and the pushforward training keep this accumulation limited, so that none of the 86 one-hour and 208 four-hour simulations diverges. Despite this increase, the predicted distributions continue to follow the grain growth evolution over the full prediction period (Fig. 6).

The model is finally tested on four thermal histories that were not included during training or validation. These cases include longer thermal cycles and multi-stage profiles with successive cooling and reheating stages. Over the ten simulations of these four cycles, the MRE ranges from 1.91% to 14.66% and the $R ^ { 2 }$ remains above 0.94 (Table 6). These results show that the model can predict grain growth for thermal histories diferent from those used during training.

Overall, the proposed approach extends time-series forecasting of grain growth to nonisothermal conditions while keeping the low computational cost of the surrogate model. The results obtained for diferent thermal conditions and unseen thermal histories show its potential for predicting grain-size distribution evolution under more realistic heat-treatment conditions.

## Data Availability

raw data used for the training and trained models (original data) are available at https: //github.com/mbernacki/GG\_DL\_TSM.

## Acknowledgments

The authors acknowledge the valuable support of ArcelorMittal, Aperam, Aubert & Duval, CEA, Constellium, Framatome, and Safran. This work was funded in part by the French National Research Agency (ANR) through the DIGIMU consortium and the RealIMotion Industrial Chair (Grant No. ANR-22-CHIN-0003).

## References

[1] C. S. Smith. “Introduction to Grains, Phases, and Interfaces—an Interpretation of Microstructure”. In: Transactions of the American Institute of Mining and Metallurgical Engineers 175.2 (1948), pp. 15–51.

[2] G. S. Rohrer et al. “Grain boundary migration in polycrystals”. In: Annual Review of Materials Research 53.1 (2023), pp. 347–369.

[3] J. Burke and D. Turnbull. “Recrystallization and grain growth”. In: Progress in Metal Physics 3 (1952), pp. 220–292. issn: 0502-8205. doi: https://doi.org/10.1016/ 0502 - 8205(52 ) 90009 - 9. url: https : / / www . sciencedirect . com / science / article/pii/0502820552900099.

[4] W. W. Mullins. “Two-Dimensional Motion of Idealized Grain Boundaries”. In: Journal of Applied Physics 27.8 (Aug. 1956), pp. 900–904. issn: 0021-8979. doi: 10.1063/ 1.1722511. eprint: https://pubs.aip.org/aip/jap/article- pdf/27/8/900/ 18315748/900\_1\_online.pdf. url: https://doi.org/10.1063/1.1722511.

[5] J. Von Neumann. “Metal interfaces”. In: American Society for Metals, Cleveland 108 (1952).

[6] A. Rollett, G. S. Rohrer, and J. Humphreys. Recrystallization and Related Annealing Phenomena. Amsterdam, Boston, Elsevier, 2017.

[7] C. Zhang et al. “Study on creep properties of nickel-based superalloy blades based on microstructure characteristics”. In: Journal of Alloys and Compounds 890 (2022), p. 161710.

[8] A. Lasalmonie and J. Strudel. “Influence of grain size on the mechanical behaviour of some high strength materials”. In: Journal of Materials Science 21.6 (1986), pp. 1837– 1852.

[9] P. Tep and M. Bernacki. “Predicting Grain Growth Evolution Under Complex Thermal Profiles with Deep Learning through Thermal Descriptor Modulation”. In: arXiv preprint arXiv:2603.24090 (2026).

[10] C. Herring. “Surface tension as a motivation for sintering”. In: Fundamental contributions to the continuum theory of evolving phase interfaces in solids: a collection of reprints of 14 seminal papers (1999), pp. 33–69.

[11] K. Chen, J. Han, and D. J. Srolovitz. “On the temperature dependence of grain boundary mobility”. In: Acta Materialia 194 (2020), pp. 412–421.

[12] F. Y. Wu. “The Potts model”. In: Rev. Mod. Phys. 54 (1 Jan. 1982), pp. 235–268.

[13] P. Peczak and M. Luton. “A Monte Carlo study of the influence of dynamic recovery on dynamic recrystallization”. In: Acta metallurgica et materialia 41.1 (1993), pp. 59– 71.

[14] Y. Liu, T. Baudin, and R. Penelle. “Simulation of normal grain growth by cellular automata”. In: Scripta materialia 34.11 (1996).

[15] H. W. Hesselbarth and I. G¨obel. “Simulation of recrystallization by cellular automata”. In: Acta Metallurgica et Materialia 39.9 (1991), pp. 2135–2143.

[16] H. Frost et al. “A two-dimensional computer simulation of capillarity-driven grain growth: Preliminary results”. In: Scripta Metallurgica 22.1 (1988), pp. 65–70.

[17] L. Barrales-Mora et al. “A novel implementation for the simulation of 2-D grain growth with consideration to external energetic fields”. In: Computational materials science 39.1 (2007), pp. 160–165.

[18] S. Florez et al. “A novel highly eficient Lagrangian model for massively multidomain simulation applied to microstructural evolutions”. In: Computer Methods in Applied Mechanics and Engineering 367 (2020), p. 113107.

[19] S. Florez and M. Bernacki. “Eficient and accurate simulation of the Smith–Zener pinning mechanism during grain growth using a front-tracking numerical framework”. In: Computational Materials Science 256 (2025), p. 113958. doi: https://doi.org/ 10.1016/j.commatsci.2025.113958.

[20] M. Bernacki. “Vertex and Front-Tracking Methods for the Modeling of Microstructure Evolution at the Solid State”. In: Digital Materials: Continuum Numerical Methods at the Mesoscopic Scale (2024), p. 155.

[21] L.-Q. Chen and D. Fan. “Computer simulation of grain growth using a continuum field model”. In: Acta Materialia(USA) 45.2 (1997), pp. 611–622.

[22] C. E. Krill III and L.-Q. Chen. “Computer simulation of 3-D grain growth using a phase-field model”. In: Acta materialia 50.12 (2002), pp. 3059–3075.

[23] M. Elsey, S. Esedog, P. Smereka, et al. “Difusion generated motion for grain growth in two and three dimensions”. In: Journal of Computational Physics 228.21 (2009), pp. 8015–8033.

[24] H.-K. Zhao et al. “A variational level set approach to multiphase motion”. In: Journal of computational physics 127.1 (1996), pp. 179–195.

[25] M. Bernacki et al. “Level set framework for the numerical modelling of primary recrystallization in polycrystalline materials”. In: Scripta Materialia 58.12 (2008), pp. 1129– 1132.

[26] M. Bernacki. “Kinetic equations and level-set approach for simulating solid-state microstructure evolutions at the mesoscopic scale: State of the art, limitations, and prospects”. In: Progress in Materials Science 142 (2024), p. 101224.

[27] M. Roth et al. “Comparison of Grain-Growth Mean-Field Models Regarding Predicted Grain Size Distributions”. In: Materials 16.20 (2023), p. 6761.

[28] E. Younes, E. Hachem, and M. Bernacki. “Predicting grain growth in polycrystalline materials using deep learning time series models”. In: Computational Materials Science 275 (2026), p. 115038.

[29] S. Florez et al. “Parallelization of an eficient 2D-Lagrangian model for massive multidomain simulations”. In: Modelling and Simulation in Materials Science and Engineering 29.6 (2021), p. 065005.

[30] K. Hitti et al. “Precise generation of complex statistical Representative Volume Elements (RVEs) in a finite element context”. In: Computational Materials Science 61 (2012), pp. 224–238.

[31] L. Maire et al. “Modeling of dynamic and post-dynamic recrystallization by coupling a full field approach to phenomenological laws”. In: Materials & Design 133 (2017), pp. 498–519.

[32] I. Goodfellow et al. Deep learning. Vol. 1. 2. MIT press Cambridge, 2016.

[33] Y. Chen et al. “Deep and low-level feature based attribute learning for person reidentification”. In: Image and Vision Computing 79 (2018), pp. 25–34.

[34] Y. LeCun, Y. Bengio, and G. Hinton. “Deep learning”. In: nature 521.7553 (2015), pp. 436–444.

[35] A. G´eron. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow. ” O’Reilly Media, Inc.”, 2022.

[36] C. K. Wikle and A. Zammit-Mangion. “Statistical deep learning for spatial and spatiotemporal data”. In: Annual Review of Statistics and Its Application 10.1 (2023), pp. 247–270.

[37] J. Fan, C. Ma, and Y. Zhong. “A selective overview of deep learning”. In: Statistical science: a review journal of the Institute of Mathematical Statistics 36.2 (2020), p. 264.

[38] J. Qi et al. “On mean absolute error for deep neural network based vector-to-vector regression”. In: IEEE Signal Processing Letters 27 (2020), pp. 1485–1489.

[39] M. Abadi et al. “TensorFlow: a system for large-scale machine learning”. In: 12th USENIX symposium on operating systems design and implementation (OSDI 16). 2016, pp. 265–283.

[40] D. P. Kingma and J. Ba. Adam: A Method for Stochastic Optimization. 2017. arXiv: 1412.6980 [cs.LG]. url: https://arxiv.org/abs/1412.6980.

[41] T. O. Hodson. “Root mean square error (RMSE) or mean absolute error (MAE): When to use them or not”. In: Geoscientific Model Development Discussions 2022 (2022), pp. 1–10.

[42] D. Chicco, M. J. Warrens, and G. Jurman. “The coeficient of determination R-squared is more informative than SMAPE, MAE, MAPE, MSE and RMSE in regression analysis evaluation”. In: Peerj computer science 7 (2021), e623.