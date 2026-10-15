# TWO-STAGE RESIDUAL LEARNING FOR MODELLING STRENGTH-INFORMED OPTIMISATION OF THERMAL SYSTEMS

Talha Ansar<sup>1</sup> Waqar Muhammad Ashraf<sup>2</sup> Abdulelah S. Alshehri<sup>3</sup> Omar Bayomie <sup>4,5,6†</sup>   
<sup>1</sup>Department of Mechanical Engineering, University of Engineering and Technology Lahore, Kala Shah Kaku, Pakistan   
<sup>2</sup>The Sargent Centre for Process Systems Engineering, Department of Chemical Engineering, University College London, London, UK

<sup>3</sup>Department of Chemical Engineering, College of Engineering, King Saud University, Riyadh, Saudi Arabia <sup>4</sup>The Advanced Centre for Biochemical Engineering, Department of Biochemical Engineering, University College London, London, UK

<sup>5</sup>SSPC, The Science Foundation Ireland Research Centre for Pharmaceuticals, University College Dublin, Belfield, Dublin, Ireland

<sup>6</sup>Institut des Systemes Intelligents et de Robotique, Sorbonne University, Paris 75005, France <sup>†</sup>Corresponding author: omar.bayomie.23@ucl.ac.uk

## ABSTRACT

Aggregate performance metrics do not identify local modelling inaccuracies in machine learning (ML) surrogates used for thermal-system optimisation. This study assesses local modelling strength through validation-based reliability screening and post-optimisation residual assessment. A two-stage residual learning (TSRL) framework, built on artificial neural networks (ANNs), is developed for modelling-strength-informed and domain-constrained optimisation of thermal systems. Within the TSRL framework, a base ANN predicts the process output, and a residual ANN learns the signed residuals left by the base ANN. The TSRL framework is implemented on two case studies of thermal systems, including 817 simulation observations of a crude distillation unit and operating records from a 395 MW industrial gas turbine. The TSRL framework reduces the test root mean square error (RMSE) from 10.3 g/s to 4.7 g/s for Fuel Oil in the refinery dataset and from 151 kJ/kWh to 105 kJ/kWh for turbine heat rate (THR). The base ANNs trained within TSRL are embedded in domain-constrained optimisation problems, and the modelling strength at each computed solution is assessed using the locked residual-score threshold. Each computed solution receives a good modelled region (GMR) or weak modelled region (WMR) label to inform ML model-based decision-making for safety-critical operation of thermal systems.

## Nomenclature

<table><tr><td>Symbol or acronym</td><td>Meaning</td></tr><tr><td>AGO</td><td>Atmospheric gas oil</td></tr><tr><td>AH</td><td>Ambient humidity</td></tr><tr><td>ANN</td><td>Artificial neural network</td></tr><tr><td>AP</td><td>Ambient pressure</td></tr><tr><td> $\mathbf { A T }$ </td><td>Ambient temperature</td></tr><tr><td>AUC</td><td>Area under the ROC curve</td></tr><tr><td>CDP</td><td>Compressor discharge pressure</td></tr><tr><td>CDT</td><td>Compressor discharge temperature</td></tr><tr><td> $D _ { M }$ </td><td>Mahalanobis distance</td></tr><tr><td> $b$ </td><td>Half-range used to truncate indicator-band perturbations</td></tr><tr><td> $\delta$ </td><td>Mahalanobis-distance limit</td></tr><tr><td> $e _ { i }$ </td><td>Absolute base-model error for observation i</td></tr><tr><td> $\epsilon _ { k }$ </td><td>kth sampled indicator-band perturbation</td></tr><tr><td>FGT</td><td>Fuel gas temperature at the combustion-chamber inlet</td></tr><tr><td>GFFR</td><td>Gas fuel flow rate</td></tr><tr><td>GMR</td><td>Good modelled region</td></tr><tr><td> $\mathcal { L } _ { B } , \mathcal { L } _ { R }$ </td><td>Base- and residual-ANN training losses</td></tr><tr><td> $\mathbf { M A E }$ </td><td>Mean absolute error</td></tr><tr><td> $\bf M L P$ </td><td>Multilayer perceptron</td></tr><tr><td>MSE</td><td>Mean squared error</td></tr><tr><td> $N$ </td><td>Number of observations in the relevant data split</td></tr><tr><td> $N _ { \mathrm { r e f } }$ </td><td>Number of refinery modelling observations (817)</td></tr><tr><td>PHGOT</td><td>Performance-heater gas outlet temperature</td></tr><tr><td> $r _ { i }$ </td><td>Signed base-model residual for observation i</td></tr><tr><td>RMSE</td><td>Root mean square error</td></tr><tr><td>ROC</td><td>Receiver operating characteristic</td></tr><tr><td> $s _ { i }$ </td><td>Residual score (predicted residual magnitude) for observation ¿</td></tr><tr><td> $\sigma _ { R }$ </td><td>Normal scale used for indicator-band sampling</td></tr><tr><td> $\mathrm { S i L U }$ </td><td>Sigmoid linear unit</td></tr><tr><td>TE</td><td>Thermal efficiency</td></tr><tr><td>THR</td><td>Turbine heat rate</td></tr><tr><td>TSRL</td><td>Two-stage residual learning</td></tr><tr><td> $T _ { e }$ </td><td>Observed-error threshold</td></tr><tr><td> $T _ { s }$ </td><td>Residual-score threshold</td></tr><tr><td>WMR</td><td>Weak modelled region</td></tr><tr><td> $x , { \tilde { x } }$ </td><td>Scaled and physical-scale process-input vectors</td></tr><tr><td> $w _ { j } , w _ { i j }$ </td><td>Crude blend fractions</td></tr><tr><td> $z _ { i j }$ </td><td>Raw Latin hypercube design coordinate</td></tr><tr><td> $y , { \hat { y } } , { \bar { y } }$ </td><td>Observed output, model prediction, and split mean</td></tr><tr><td></td><td> $\mathrm { C O _ { 2 } }$  ceiling in refinery optimisation</td></tr><tr><td> $y _ { \mathrm { { C O _ { 2 } } } } ^ { \mathrm { { c a p } } }$ </td><td></td></tr><tr><td> $\boldsymbol { r } ^ { \mathrm { t a r g e t } }$ </td><td>Prescribed scaled input target</td></tr><tr><td> $\hat { \boldsymbol { u } } ^ { \mathit { \Pi } }$ </td><td>Base-model prediction</td></tr><tr><td> $\stackrel { \circ } { \hat { \boldsymbol { u } } ^ { R } }$ </td><td>Residual-model prediction</td></tr><tr><td> $\overset { \circ } { \boldsymbol { \hat { y } } ^ { C } }$ </td><td>Residual-corrected prediction</td></tr><tr><td> $\theta _ { B } , \theta _ { R }$ </td><td>Trainable parameter vectors of the base and residual ANN</td></tr></table>

## 1 Introduction

Optimising the operation of thermal systems determines operating decisions that improve a defined objective while satisfying equipment, material, energy, and safety constraints [1]. Surrogate-based optimisation [2, 3] reduces the computational cost of model-based decision-making by replacing repeated calls to a detailed model with an approximation that can be evaluated quickly. In refinery-wide energy optimisation, artificial neural networks (ANNs) replace high-fidelity first-principles process models [4]. ANN-based models are also used for the data-driven design of cooling systems [5].

An ANN is a nonlinear continuous function approximator supported by universal approximation theory [6]. Its differentiable form suits surrogate-based optimisation, including thermal-power studies that use machine-intelligence models in constrained operation [7]. The explicit algebraic form of a trained ANN can be embedded in a mathematical programme. Deterministic global optimisation [8], the OMLT toolkit [9], and a recent gas-turbine simulation platform [10] evaluate trained networks inside constrained optimisation. Misener and Biegler [11] identify robustness to uncertainty and accurate extrapolation as essential conditions for surrogates used in process optimisation. Parameter estimation, model selection, and final evaluation require separate data to avoid optimistic error estimates from reusing selection data [12,13]. Root mean square error (RMSE), mean absolute error (MAE), and the coefficient of determination (R<sup>2</sup>) summarise split-level performance, but do not locate modelling errors within the learned input–output mapping.

Limited information about local modelling accuracy at given input conditions restricts the use of surrogate-based optimisation in the operational control of thermal systems. The optimiser searches for favourable predictions and may return a point with high local modelling error. Kahrs and Marquardt [14] bounded the optimisation of hybrid models by a validity domain built from the training inputs. Schweidtmann et al. [15] combined topological data analysis with one-class support vector machines and convex hulls, and encoded the data domain as optimisation constraints. Meyer and Pebesma [16] introduced an area of applicability for spatial prediction models with a cross-validation-derived threshold to identify regions where predictive errors are expected to be comparable to cross-validation errors. Recent gas-turbine studies [10, 17] use a Mahalanobis constraint to retain the multivariate input combinations represented in plant data. These methods restrict surrogate evaluation to observed input combinations, but satisfying a domain constraint does not establish local prediction accuracy.

Domain restrictions can be complemented by predictive uncertainty estimates from Monte Carlo dropout [18] and deep ensembles [19]. Gawlikowski et al. [20] reviewed their assumptions, computational cost, and performance. Cajahuaringa et al. [21] applied Monte Carlo dropout to a nonlinear autoregressive model of a 215 MW gas turbine and reported the uncertainty of the predicted rotational speed. The study addressed predictive uncertainty but did not carry the uncertainty into surrogate-based operational optimisation of the gas turbine system. Yang et al. [22] trained an ensemble of ReLU networks on resampled training sets, approximated the ensemble with a mixture density network, and solved a chance-constrained mixed-integer programme. These approaches require repeated training or stochastic forward passes. When used in optimisation, the uncertainty model enters the problem and changes its size and form. The resulting spread measures disagreement among networks or samples, rather than a decision threshold calibrated against observed errors of the deployed surrogate.

Residual learning offers a direct connection to observed errors. In stagewise function fitting [23], each new learner is fitted to the errors left by the preceding model. Computer-model calibration [24] treats model discrepancy as a term separate from the simulator response. Hybrid semi-parametric models [25, 26] use a data-driven component to correct a mechanistic model. In these studies, the residual learner improves prediction accuracy. The size of the learned correction is not used as a reliability signal. Lahlou et al. [27] trained a secondary predictor of the pointwise error of a main model as a measure of epistemic uncertainty. Their method was developed for active learning, sequential model optimisation, and exploration in reinforcement learning. It returns a continuous uncertainty estimate. It does not lock a decision threshold on held-out data, and it does not label the solutions of a constrained nonlinear programme.

This work calibrates a residual score against observed base-model errors on a validation split, then assesses solutions of domain-constrained optimisation without adding an uncertainty model to its formulation. Refinery and gas-turbine surrogates are embedded in optimisation studies [4,10,17], yet the reported operating points do not carry a point-specific measure of modelling strength before they reach the operator.

This work develops a two-stage residual learning (TSRL) framework for assessing local modelling strength in surrogate based optimisation. In the first stage, a base ANN is fitted for each process output. In the second stage, the base parameters are fixed and a residual ANN learns the signed base-model error. The signed residual gives a point correction. Its magnitude gives a residual score, which remains available at an operating point where the true output is unknown. Validation data link the score to the observed base error and determine two percentile thresholds by maximising F . The thresholds are locked before test evaluation and before any optimisation solution is scored. Only the base ANN is embedded in the optimisation problem. The residual score then labels each returned solution as lying in a good modelled region (GMR) or a weak modelled region (WMR).

The TSRL framework is examined on two thermal systems with different sources of modelling error. The first case uses 817 simulated observations of a crude distillation unit, with seven inputs that cover varied crude blends and utility conditions. The second case uses operating records of a 395 MW industrial gas turbine [7], where the measurements carry variability from operating dynamics and sensor noise. Gas-turbine studies [10, 17, 28] report ML-based prediction and optimisation of plant performance without a point-specific check of modelling strength at new input conditions. TSRL adds local reliability assessment to surrogate-based optimisation through three main contributions.

• The TSRL framework uses a residual ANN to correct the base surrogate and its predicted residual magnitude to score reliability at unseen operating points.

• Residual scores are calibrated by cross-matching percentile labels of observed base error and residual score for each validation observation. The threshold pair is selected by $F _ { 1 }$ , and both thresholds are locked before testing.

• Computed solutions are assessed after optimisation while retaining the original optimisation problem. Base ANNs are embedded in Mahalanobis-constrained nonlinear programmes, and each solution receives a GMR or WMR label from TSRL after convergence. Many solutions of the refinery case study satisfy the domain constraints and yet fall in the WMR.

The rest of the paper is organised as follows. Section 2 defines the models, calibration procedure, and optimisation formulation. Section 3 presents the refinery and gas turbine studies. Section 4 summarises the findings, limitations, and directions for further work.

## 2 Methods

The TSRL framework separates surrogate fitting, residual correction, reliability calibration, and optimisation, as illustrated in Figure 1. For each process output, a base ANN is fitted and its prediction, $\hat { y } ^ { B } ( x )$ , is used as the surrogate response in the optimisation problem. After the base-network parameters are fixed, a residual ANN receives the same input vector x and predicts the signed base-model error, $\hat { y } ^ { R } ( \check { x } )$ . The corrected response is $\hat { y } ^ { C } ( x ) = \hat { y } ^ { B } ( x ) + \hat { y } ^ { R } ( x )$ while the magnitude $| \hat { y } ^ { R } ( x ) |$ is retained separately as a residual score. This score remains available at a proposed operating point where the true process output is unknown.

Each validation observation has an observed absolute base error and a residual score. A percentile search applies an error threshold to the first quantity and a score threshold to the second quantity. The two resulting labels are cross-matched sample by sample to calculate validation precision, recall, and $F _ { 1 }$ for each threshold pair. The pair that maximises validation $F _ { 1 }$ fixes the observed-error threshold $T _ { e }$ and residual-score threshold $T _ { s }$ . Both thresholds are locked before test evaluation [12, 13]. The domain-constrained optimisation then embeds only the base ANN and returns an operating solution $x ^ { * }$ . The residual score at $x ^ { * }$ is compared with the locked $T _ { s }$ to assign a GMR or WMR label.

## 2.1 Data partition, scaling, and evaluation

Each dataset of the two case studies is divided randomly into 70% training observations, 15% validation observations, and 15% test observations. The training split is used to estimate network parameters. The validation split guides hyperparameter selection, early stopping, output selection for detailed reliability analysis, and threshold calibration. The test split is evaluated once after modelling choices have been fixed. Each input is normalised to the interval [0, 1] by min-max scaling,

$$
x _ { i j } = \frac { \tilde { x } _ { i j } - \tilde { x } _ { j } ^ { \mathrm { m i n } } } { \tilde { x } _ { j } ^ { \mathrm { m a x } } - \tilde { x } _ { j } ^ { \mathrm { m i n } } } ,\tag{1}
$$

where $i$ indexes an observation and $j$ indexes an input variable. $\tilde { x } _ { i j }$ is the physical-scale value of input $j$ at observation $i ,$ while $\tilde { x } _ { i } ^ { \mathrm { m i n } }$ and $\tilde { x } _ { j } ^ { \mathrm { m a x } }$ are the lower and upper values used to scale that input. The resulting $x _ { i j }$ is dimensionless and lies in [0, 1]. The same limits are used for all splits and to convert an optimisation solution between scaled and physical input values.

The modelling performance of each ANN is evaluated using RMSE, MAE, and $R ^ { 2 }$

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( y _ { i } - \hat { y } _ { i } \right) ^ { 2 } } ,
$$

$$
\mathrm { M A E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| y _ { i } - \hat { y } _ { i } \right| ,\tag{2}
$$

$$
R ^ { 2 } = 1 - \frac { \sum _ { i = 1 } ^ { N } \left( y _ { i } - \hat { y } _ { i } \right) ^ { 2 } } { \sum _ { i = 1 } ^ { N } \left( y _ { i } - \bar { y } \right) ^ { 2 } } .
$$

![](images/93b4ae8b27484f0a0692de6dca101c50621bd1e11a43081de2caa795152f831d.jpg)  
Figure 1: TSRL information flow. The data contain process inputs x and observed output $y ,$ while the base and residual ANNs predict $\hat { y } ^ { B }$ and $\hat { y } ^ { R }$ , respectively. The residual target is $r _ { i } = y _ { i } - \hat { y } _ { i } ^ { B }$ . Validation labels based on absolute base error and residual score fix $T _ { e }$ and $T _ { s }$ before testing. Only the base ANN enters the optimisation problem, and $| \hat { y } ^ { R } ( x ^ { * } )$ assigns a GMR or WMR label after convergence.

For the evaluated split, N is the number of observations and $\bar { y }$ is its mean output, while $\hat { y } _ { i }$ denotes the model prediction being assessed.

## 2.2 Base and residual surrogate models

The base and residual ANNs use a multilayer perceptron (MLP) architecture for nonlinear input–output modelling [6, 28, 29]. A separate multi-input single-output network is fitted for every process output. Each network has one hidden layer and a linear output. The hidden units use the sigmoid linear unit (SiLU) activation [30], with $\mathrm { S i L U } ( u ) = u \dot { / } ( 1 + \exp ( - u ) )$ . The base surrogate is written as

$$
\begin{array} { r } { \hat { y } ^ { B } = W _ { \mathrm { o u t } } \operatorname { S i L U } \left( W _ { \mathrm { i n } } x + b _ { \mathrm { i n } } \right) + b _ { \mathrm { o u t } } . } \end{array}\tag{3}
$$

$W _ { \mathrm { i n } }$ and $W _ { \mathrm { o u t } }$ contain the input-to-hidden and hidden-to-output network weights, while $b _ { \mathrm { i n } }$ and $b _ { \mathrm { o u t } }$ contain the corresponding bias vectors. This differentiable expression can be embedded in a nonlinear programme [8, 10]. The base-ANN loss combines mean squared error (MSE) with an L1 penalty,

$$
\mathcal { L } _ { B } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( y _ { i } - \hat { y } _ { i } ^ { B } \right) ^ { 2 } + \lambda _ { 1 , B } \Vert \theta _ { B } \Vert _ { 1 } .\tag{4}
$$

Regularisation is applied to the base-network parameter vector $\theta _ { B }$ . The coefficient $\lambda _ { 1 , B }$ weights the L1 norm $\| \theta _ { B } \| _ { 1 }$ The L2 coefficient $( \lambda _ { 2 , B } )$ is applied through Adam weight decay. The learning rate, L1 coefficient, L2 coefficient, and hidden-layer size are tuned in 50 trials for each network using Hyperopt [31] with the tree-structured Parzen estimator [32] over the search space in Table 1. The candidate with the lowest validation RMSE is retained.

Table 1: Hyperparameter search space used independently for the base and residual ANNs.
<table><tr><td>Hyperparameter</td><td>Search range</td><td>Sampling type</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 5 } ~ \mathrm { t o } ~ 1 0 ^ { - 1 }$ </td><td>Log continuous</td></tr><tr><td>L1 coefficient</td><td> $1 0 ^ { - 7 } ~ \mathrm { t o } ~ 1 0 ^ { - 1 }$ </td><td>Log continuous</td></tr><tr><td>L2 coefficient</td><td> $1 0 ^ { - 7 } \ \mathrm { t o } \ 1 0 ^ { - 1 }$ </td><td>Log continuous</td></tr><tr><td>Hidden-layer neurons</td><td>2 to 32</td><td>Discrete in steps of 2</td></tr></table>

Training uses the Adam optimiser [33] in $\mathrm { P y }$ Torch [34] with mini-batch gradients, a maximum of 5000 epochs, and validation-based early stopping after 200 successive epochs without improvement in RMSE [35].

After fitting the base ANN, $\theta _ { B }$ is fixed and the signed residual target for the second stage is computed as [23]

$$
r _ { i } = y _ { i } - \hat { y } _ { i } ^ { B } ,\tag{5}
$$

where a positive $r _ { i }$ indicates underprediction by the base ANN. A second network uses the same process inputs to predict $r _ { i }$ . Its hyperparameters are selected independently from the same ranges. The residual prediction and corrected response are

$$
\begin{array} { r } { \hat { y } _ { i } ^ { R } = F _ { R } ( x _ { i } ) , \qquad \hat { y } _ { i } ^ { C } = \hat { y } _ { i } ^ { B } + \hat { y } _ { i } ^ { R } . } \end{array}\tag{6}
$$

Here $F _ { R } ( \cdot )$ denotes the residual ANN mapping. The residual loss is expressed as

$$
\mathcal { L } _ { R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( r _ { i } - \hat { y } _ { i } ^ { R } \right) ^ { 2 } + \lambda _ { 1 , R } \Vert \theta _ { R } \Vert _ { 1 } .\tag{7}
$$

The residual-network parameter vector is $\theta _ { R } ,$ , and $\lambda _ { 1 , R }$ weights its L1 penalty. L2 weight decay $\lambda _ { 2 , R }$ is applied during training. The corrected prediction is evaluated without updating the base network.

For each case study, one output is carried forward for detailed residual-score assessment. The selection uses the relative validation improvement,

$$
\Delta _ { \mathrm { v a l } } = \frac { \mathrm { R M S E } _ { \mathrm { v a l } } ^ { B } - \mathrm { R M S E } _ { \mathrm { v a l } } ^ { C } } { \mathrm { R M S E } _ { \mathrm { v a l } } ^ { B } } ,\tag{8}
$$

where $\mathrm { R M S E } _ { \mathrm { v a l } } ^ { B }$ and $\mathrm { R M S E } _ { \mathrm { v a l } } ^ { C }$ are the validation RMSE values of the base and corrected predictions, respectively. A positive $\Delta _ { \mathrm { v a l } }$ means that residual correction lowers validation RMSE. The selected output has the largest positive $\dot { \Delta } _ { \mathrm { v a l } }$ among the outputs for which training RMSE also decreases. This rule is applied before any test result is examined.

## 2.3 Residual-score calibration and reliability classification

The validation split supplies the empirical link between observed base-model error and the residual score. For each validation observation,

$$
e _ { i } = \left| y _ { i } - \hat { y } _ { i } ^ { B } \right| , \qquad s _ { i } = \left| \hat { y } _ { i } ^ { R } \right| .\tag{9}
$$

The observed absolute base error $e _ { i }$ requires a known output $y _ { i } .$ , while the residual score $s _ { i }$ is also available at a new operating point. Both quantities are calculated for the same validation observation before threshold calibration. Spearman rank correlation is calculated between $e _ { i }$ and $s _ { i }$

The continuous quantities are converted to reference and predicted classes with percentile thresholds,

$$
\begin{array} { r l } & { T _ { e } ( p _ { e } ) = P _ { p _ { e } } ( e ) , \quad z _ { i } ^ { \mathrm { t r u e } } = \mathbb { I } \left[ e _ { i } \geq T _ { e } ( p _ { e } ) \right] , } \\ & { T _ { s } ( p _ { s } ) = P _ { p _ { s } } ( s ) , \quad z _ { i } ^ { \mathrm { p r e d } } = \mathbb { I } \left[ s _ { i } \geq T _ { s } ( p _ { s } ) \right] . } \end{array}\tag{10}
$$

Here, $P _ { p } ( \cdot )$ denotes the empirical pth percentile, $p _ { e }$ and $p _ { s }$ are the candidate error and score percentiles, and $T _ { e }$ and $T _ { s }$ are the resulting numerical thresholds. I[·] is an indicator function that equals one when its condition is met and zero otherwise. Thus, $z _ { i } ^ { \mathrm { t r u e } } = 1$ identifies a validation observation with high observed base error, while $z _ { i } ^ { \mathrm { p r e d } } = 1$ identifies an observation with a high residual score. The candidate grids are $p _ { e } , p _ { s } \in \{ 8 0 , 8 1 , \ldots , 9 5 \}$ . For every pair, these two labels are cross-matched for the same validation observations, and precision, recall, and $F _ { 1 }$ are computed as

$$
\mathrm { P r e c i s i o n } = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } } } , \qquad \mathrm { R e c a l l } = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } } } , \qquad F _ { 1 } = { \frac { 2 \mathrm { P r e c i s i o n ~ R e c a l l } } { \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } } } .\tag{11}
$$

In these expressions, TP is the number of observations labelled high by both $z _ { i } ^ { \mathrm { t r u e } }$ and $z _ { i } ^ { \mathrm { p r e d } }$ , FP is the number labelled high by the score but not by the observed error, and FN is the number labelled high by the observed error but not by the score. The pair with the highest validation $F _ { 1 }$ is selected. Receiver operating characteristic (ROC) curves and their area under the curve (AUC) are calculated from the validation labels and residual scores.

The selected $T _ { e }$ and $T _ { s }$ are fixed before test evaluation. A point with $s ( x ) < T _ { s }$ is assigned to GMR. A point with $s ( x ) \geq T _ { s }$ is assigned to WMR. GMR and WMR locate a point’s residual score relative to the locked validation threshold.

The high-error reference class is also defined from validation percentiles, so $F _ { 1 }$ measures agreement with a relative error label. Where an accepted process-error limit is available, it can replace the percentile definition of $T _ { e }$ within the same calibration procedure.

## 2.4 Constrained optimisation and local assessment

Pyomo [36] is used to write the fitted base networks, process constraints, and objective as one nonlinear programme. The trained surrogate is evaluated within the optimisation problem [8, 10]. The general problem is formulated as

$$
\begin{array} { r l r } { \underset { x } { \operatorname* { m a x } } } & { J ( x ) } \\ { \mathrm { s u b j e c t ~ t o } } & { y _ { q } ^ { \mathrm { L B } } \leq \hat { y } _ { q } ^ { B } ( x ) \leq y _ { q } ^ { \mathrm { U B } } } & { q \in \mathcal { P } } \\ & { D _ { M } ^ { 2 } ( \tilde { x } ) \leq \delta ^ { 2 } } \\ & { 0 \leq x _ { j } \leq 1 } & { j \in \mathcal { X } , } \end{array}\tag{12}
$$

where $J ( x )$ is the case-specific objective and $\mathcal { P }$ is the set of constrained outputs indexed by $q . ~ { \hat { y } } _ { q } ^ { B } ( x )$ is the base-ANN prediction for output $q ,$ and $y _ { q } ^ { \mathrm { L B } }$ and $y _ { q } ^ { \mathrm { U B } }$ are its lower and upper permitted values. X indexes the process inputs. Mahalanobis distance [37] accounts for covariance when measuring separation from a multivariate reference distribution. Recent gas-turbine formulations [10, 17] use this distance to restrict data-driven optimisation to correlated plant-input combinations. The squared distance is written as

$$
D _ { M } ^ { 2 } ( \tilde { x } ) = \left( \tilde { x } - \mu \right) ^ { \mathsf { T } } \Sigma ^ { + } \left( \tilde { x } - \mu \right) .\tag{13}
$$

The mean vector $\mu$ and covariance matrix Σ are calculated from the reference inputs. The Moore-Penrose pseudoinverse [38], denoted by $\Sigma ^ { + }$ , is used when the covariance matrix is singular. The empirical limit δ restricts the search to a covariance-aware input region represented by the data. This excludes input combinations that satisfy individual bounds but are poorly represented jointly [16, 39].

The nonlinear programme is solved with IPOPT through the GAMS solver interface. IPOPT implements an interiorpoint filter line-search method [40] and returns a locally optimal constrained solution. The base surrogate remains in the objective and constraints so that residual assessment does not change the optimisation problem. After a solution $x ^ { * }$ has been returned, the residual network is evaluated at $x ^ { * }$ . Its score is $\mathbf { \bar { \Psi } } ^ { s } ( x ^ { * } ) = \mathbf { \bar { \Psi } } | \hat { y } ^ { R } ( x ^ { * } ) |$ and is compared with the fixed $T _ { s }$

A residual-informed indicator band is centred on the corrected prediction. Its scale is defined as

$$
b = \frac { \left| \hat { y } ^ { R } ( x ^ { * } ) \right| } { 2 } , \qquad \sigma _ { R } = \frac { b } { 1 . 9 6 } .\tag{14}
$$

The half-range b equals half the residual score and bounds the sampled perturbations. $\sigma _ { R }$ is the standard deviation of the normal distribution before truncation. Five hundred values are sampled from the zero-centred normal distribution and restricted to $[ - b , b ]$ ,

$$
\epsilon _ { k } \sim \mathcal { N } ( 0 , \sigma _ { R } ^ { 2 } ) , \qquad - b \leq \epsilon _ { k } \leq b , \qquad \hat { y } ^ { ( k ) } = \hat { y } ^ { C } ( x ^ { * } ) + \epsilon _ { k } , \qquad k = 1 , \dots , 5 0 0 ,\tag{15}
$$

The 2.5th and 97.5th empirical percentiles of $\hat { y } ^ { ( k ) }$ form the reported band. The indicator band visualises sampled perturbations around the corrected prediction for comparison among solutions, with its width set by the residual score.

## 3 Results and Discussion

Model performance and residual-score screening are evaluated before computed optimisation solutions are labelled GMR or WMR. The refinery simulation is considered first because its controlled input variation provides a direct test of approximation error. The gas turbine study then applies the TSRL framework to sensor-based industrial records that contain measurement variation and a less uniform operating history.

## 3.1 Case Study 1. Oil Refinery

## 3.1.1 Rigorous process model

The refinery case study is built on a steady-state Aspen HYSYS model of an industrial crude distillation unit (CDU) that replicates an operating unit with a design throughput of 291.7 t/h. The flowsheet comprises the crude preheat train, the fired heater, and the atmospheric column with its side strippers and pump-around circuits, and delivers the product slate of the real unit, including fuel gas, LPG, stabilised naphtha, heavy naphtha, Kerosene, gas oil, and atmospheric residue (fuel oil). The four crude oils are characterised in the HYSYS Oil Manager from their laboratory assays (true-boiling-point curves, light-ends compositions, and gravity data), and vapour–liquid equilibrium is described using an equation of state (EOS). The model was validated against plant operating data in earlier work [41]. The flow rates and product draw temperatures match plant records to within a few percent, and the simulated ASTM D86 5–95% gaps between adjacent cuts reproduce the measured separation quality. Throughout this work, this validated model is treated as the ground-truth response function whose input–output behaviour is learned by TSRL, with local modelling strength assessed for the resulting surrogates.

## 3.1.2 Crude blend input space and sampling design

The operational degrees of freedom are the blend fractions of four commercially traded crudes, namely Basrah, Arab Heavy, Arab Medium, and Kuwait, spanning the light-to-heavy API range processed by the unit. The total feed rate is held at the design throughput, so a blend is fully specified by the fraction vector $\mathbf { w } = ( w _ { 1 } , \dots , w _ { 4 } )$ with $w _ { j } \geq 0$ and $\textstyle \sum _ { j = 1 } ^ { 4 } w _ { j } = 1$ . Design points are generated by Latin hypercube sampling (LHS), selected for its stratified marginal coverage at moderate sample sizes [42, 43]. A raw design in the four-dimensional unit hypercube was drawn with the scipy.stats.qmc implementation under a fixed random seed for reproducibility, and each point was projected onto the simplex by normalisation,

$$
w _ { i j } = \frac { z _ { i j } } { \sum _ { \ell = 1 } ^ { 4 } z _ { i \ell } } , \qquad j = 1 , \dots , 4 .\tag{16}
$$

The projection does not preserve the strict one-dimensional stratification of the hypercube design, and the normalised fractions concentrate mildly towards the simplex centroid relative to a uniform (flat Dirichlet) distribution. However, it retains the space-filling character of LHS over the feasible blend space while guaranteeing that every sample satisfies the unit-sum constraint exactly. No crude was constrained away from zero, allowing near-vertex blends (one crude dominant) and near-centroid blends.

## 3.1.3 Automated simulation campaign

The sampling campaign was executed through a programmatic Python–HYSYS interface using an object-model approach to integrate Python with the process simulator [44]. The simulator is driven over the ActiveX/COM automation layer (pywin32), and a HYSYS Spreadsheet operation embedded in the flowsheet serves as the single input–output interface. Its input cells are linked to the four blend fractions and its output cells import the monitored flowsheet variables, so the Python driver does not address individual unit operations directly. This design decouples the sampling logic from the flowsheet topology and makes the driver reusable across model revisions.

For each design point the driver executes a hold–write–release cycle. The solver is paused (Solver.CanSolve = False), the four blend fractions are written to the spreadsheet input cells, and the solver is released, after which the flowsheet re-converges from the previous solution as a warm start. The output cells are then read back before the driver proceeds to the next sample.

## 3.1.4 Data description

The refinery modelling dataset contains $N _ { \mathrm { r e f } } = 8 1 7$ simulation observations. The seven inputs are the Basrah, Arab Heavy, Arab Medium, and Kuwait crude flow rates, reboiler duty, and two condenser loads. The seven outputs are Light Naphtha, Gases, Heavy Naphtha, Kerosene, AGO, Fuel Oil, and carbon dioxide $\left( \mathbf { C O } _ { 2 } \right)$ . Table 2 reports the retained ranges. The crude rates span near-zero values to about 56–65 kg/s, which permits wide blend variation. The full reboiler-duty range, in contrast, is about 3.5% of its mean. The gas stream is zero or near zero for much of the dataset and has a small number of larger observations. This imbalance affects its regression metrics.

Table 2: Descriptive statistics for the refinery variables. Values use two decimal places except the low-flow gas stream, which uses four.
<table><tr><td>Variable</td><td>Mean</td><td>SD</td><td>Minimum</td><td>Median</td><td>Maximum</td></tr><tr><td>Basrah (kg/s)</td><td>20.24</td><td>11.02</td><td>0.01</td><td>20.10</td><td>55.71</td></tr><tr><td>Arab Heavy (kg/s)</td><td>19.87</td><td>9.92</td><td>0.12</td><td>20.31</td><td>58.18</td></tr><tr><td>Arab Medium (kg/s)</td><td>20.50</td><td>10.98</td><td>0.06</td><td>20.76</td><td>58.33</td></tr><tr><td>Kuwait (kg/s)</td><td>20.41</td><td>10.84</td><td>0.04</td><td>21.09</td><td>64.82</td></tr><tr><td>Reboiler duty (kW)</td><td>35,297.00</td><td>193.00</td><td>34,638.00</td><td>35,300.00</td><td>35,860.00</td></tr><tr><td>Condenser 1 (kW)</td><td>6,399.96</td><td>239.55</td><td>4,850.55</td><td>6,417.13</td><td>7,021.43</td></tr><tr><td>Condenser 2 (kW)</td><td>13,314.49</td><td>439.53</td><td>10,230.67</td><td>13,316.29</td><td>14,629.20</td></tr><tr><td>Light Naphtha (kg/s)</td><td>5.24</td><td>0.14</td><td>4.80</td><td>5.23</td><td>5.63</td></tr><tr><td>Gases (kg/s)</td><td>0.0031</td><td>0.0179</td><td>0.0000</td><td>0.0000</td><td>0.3245</td></tr><tr><td>Heavy Naphtha (kg/s)</td><td>2.92</td><td>0.16</td><td>2.44</td><td>2.91</td><td>3.40</td></tr><tr><td>Kerosene (kg/s)</td><td>12.25</td><td>0.30</td><td>11.51</td><td>12.25</td><td>13.26</td></tr><tr><td>AGO (kg/s)</td><td>11.70</td><td>0.15</td><td>11.26</td><td>11.71</td><td>12.13</td></tr><tr><td>Fuel Oil (kg/s)</td><td>46.16</td><td>0.58</td><td>44.20</td><td>46.16</td><td>47.65</td></tr><tr><td> $\mathrm { C O _ { 2 } \ ( k g / h ) }$ </td><td>8,354.88</td><td>45.71</td><td>8,198.81</td><td>8,355.57</td><td>8,488.15</td></tr></table>

(a)  
![](images/40aeb77b88e75c83258697d414624a7231d765f8cb16174867588b7d78a2318c.jpg)

![](images/fd32052bfd867f730071d5bee230366c1a5ce3be9f6661867efce9e6f467ef72.jpg)

![](images/2a9b1284d08f5a773153d7a8dcffb04768c199ab972148edbc923334c1f42d06.jpg)

![](images/f79e84fba26339cf3159f35601d0c87bc0fb4b7c2b71434ae4fd801f160234bd.jpg)

![](images/6f7f2179ea61e504f98b8416fd64219646838393f1f39343d157fa2541dcff06.jpg)

![](images/b82212cbc2a657c55468a497446fefbedc5602eae7ea7ba9fd03bec910182061.jpg)  
Figure 2: Data characteristics for the crude distillation unit. Panel (a) shows kerosene increasing with Arab Medium flow while Light Naphtha decreases with Arab Heavy flow. Panel (b) shows relative crude contributions to product streams. Panel (c) shows the Spearman correlation map of the input and output variables.

## 3.1.5 Surrogate performance and residual screening

Figure 3 presents the base-ANN parity plots on the training, validation, and test splits. Figure $3 ( \mathrm { a } ) { - } ( \mathrm { d } ) , ( \mathrm { f } )$ , and (g) place Light Naphtha, Heavy Naphtha, Kerosene, Fuel Oil, $\mathrm { C O _ { 2 } } .$ , and AGO close to the one-to-one line. Figure 3(b) gives a test $R ^ { 2 }$ of 0.98 for Heavy Naphtha, while Figure 3(a), (c), (d), (f), and (g) each give a value greater than 0.99. The corresponding test RMSE values are $2 1 . 2 \mathrm { g } / \mathrm { s }$ for Heavy Naphtha in Figure 3(b), 10.3 g/s for Fuel Oil in Figure 3(d), and 0.78 kg/h for $\mathrm { C O _ { 2 } }$ in Figure 3(f). The six non-gas prediction errors are small relative to the observed response ranges. In contrast, Figure 3(e) shows a poor fit for the gas stream. For the gas stream, the zero median and occasional larger flows make a small RMSE insufficient evidence of a useful fit. The test $R ^ { 2 }$ of −2.17 in Figure 3(e) confirms that the model does not explain the variation among the less frequent non-zero observations and is less informative than the test-mean baseline.

![](images/664a769287aef9547b58ddceccc77aaa50f56ca80463eaeab73de4022adbd931.jpg)

![](images/6bdc81dcbea09feb6b755dacf80a42e4110ae629fae3eccf5aceba34ed7b7de7.jpg)

![](images/1ecaa8275f8ffff99bbe5aad1bb0b30010e9c72d2053deff5871f27b75add33f.jpg)

![](images/da3aa4ebac9a909de237eb680446e8baa66c4ecbc5e78e1f5518384d7791bebf.jpg)

![](images/48f7a6aa6faa23accf94c223f3d6adae803d9ff6d93333ef41efecbb0a3bcc2c.jpg)

![](images/3090a933190e776e159cc7ad207cf314a879a94e66cd7e0be0f49f97f1cba23b.jpg)

![](images/5fd5520e9874fce9426bb30b4562845093208fadc5046c06a1c237429f79d2d3.jpg)  
Figure 3: Base-model parity plots across training, validation, and test data for (a) Light Naphtha, (b) Heavy Naphtha, (c) Kerosene, (d) Fuel Oil, (e) Gases, (f) CO , and (g) AGO. The six non-gas surrogates provide reasonable simulator approximations within the sampled range.

Table 3 shows that the effect of residual correction depends on the output. Corrections make only small changes to Heavy Naphtha, Kerosene, and AGO, but increase Light Naphtha test RMSE from 10.9 to 18.7 g/s. The Gases test $R ^ { 2 }$ remains negative at −2.02, despite residual correction. For $\mathrm { C O _ { 2 } } ,$ , validation RMSE falls from 0.91 to 0.66 kg/h, while test RMSE rises from 0.78 to 2.55 kg/h. Fuel Oil shows the largest validation gain, with RMSE decreasing from 7.4 to 4.1 g/s, and its training RMSE also falls from 10.5 to 4.9 g/s. Fuel Oil is selected for the detailed reliability analysis under the rule in Equation 8. Its test RMSE falls from 10.3 to 4.7 g/s, a reduction of about 54%.

Table 3: Base and residual-corrected performance for the crude distillation unit.
<table><tr><td rowspan="2">Target</td><td rowspan="2">Metric</td><td colspan="2">Training</td><td colspan="2">Validation</td><td colspan="2">Test</td></tr><tr><td>Base</td><td>Corrected</td><td>Base</td><td>Corrected</td><td>Base</td><td>Corrected</td></tr><tr><td rowspan="2">Light Naphtha (kg/s)</td><td>RMSE</td><td>0.0218</td><td>0.0243</td><td>0.0209</td><td>0.0205</td><td>0.0109</td><td>0.0187</td></tr><tr><td> $R ^ { 2 }$ </td><td>0.9745</td><td>0.9683</td><td>0.9734</td><td>0.9743</td><td>0.9945</td><td>0.9838</td></tr><tr><td rowspan="2">Gases (kg/s)</td><td>RMSE</td><td>0.0197</td><td>0.0195</td><td>0.0182</td><td>0.0181</td><td>0.0086</td><td>0.0083</td></tr><tr><td> $R ^ { 2 }$ </td><td>-0.0336</td><td>-0.0171</td><td>0.0702</td><td>0.0814</td><td>-2.1670</td><td>-2.0193</td></tr><tr><td rowspan="2">Heavy Naphtha (kg/s)</td><td>RMSE</td><td>0.0192</td><td>0.0193</td><td>0.0195</td><td>0.0194</td><td>0.0212</td><td>0.0214</td></tr><tr><td> $R ^ { 2 }$ </td><td>0.9860</td><td>0.9859</td><td>0.9850</td><td>0.9852</td><td>0.9819</td><td>0.9816</td></tr><tr><td rowspan="2">Kerosene (kg/s)</td><td>RMSE</td><td>0.0048</td><td>0.0048</td><td>0.0058</td><td>0.0052</td><td>0.0037</td><td>0.0038</td></tr><tr><td> $R ^ { 2 }$ </td><td>0.9997</td><td>0.9997</td><td>0.9996</td><td>0.9997</td><td>0.9998</td><td>0.9998</td></tr><tr><td rowspan="2">AGO (kg/s)</td><td>RMSE</td><td>0.0038</td><td>0.0035</td><td>0.0039</td><td>0.0036</td><td>0.0034</td><td>0.0033</td></tr><tr><td> $R ^ { 2 }$ </td><td>0.9994</td><td>0.9995</td><td>0.9993</td><td>0.9994</td><td>0.9996</td><td>0.9996</td></tr><tr><td rowspan="2">Fuel Oil (kg/s)</td><td>RMSE</td><td>0.0105</td><td>0.0049</td><td>0.0074</td><td>0.0041</td><td>0.0103</td><td>0.0047</td></tr><tr><td> $R ^ { 2 }$ </td><td>0.9997</td><td>0.9999</td><td>0.9998</td><td>0.9999</td><td>0.9997</td><td>0.9999</td></tr><tr><td rowspan="2"> $\mathrm { C O _ { 2 } \ ( k g / h ) }$ </td><td>RMSE</td><td>10.3731</td><td>6.5386</td><td>0.9125</td><td>0.6567</td><td>0.7815</td><td>2.5496</td></tr><tr><td> $R ^ { 2 }$ </td><td>0.9503</td><td>0.9802</td><td>0.9996</td><td>0.9998</td><td>0.9997</td><td>0.9966</td></tr></table>

Fuel Oil residual reliability diagnostic The selected Fuel Oil surrogate is examined for local modelling strength (Table 3). Figure 4(a) compares predicted and true signed residuals. The RMSE and MAE values are 4.9 and 3.8 g/s for training, 4.1 and 3.4 g/s for validation, and 4.7 and 3.7 g/s for testing. The similar errors across the splits indicate that the correction model retains its point-prediction performance on unseen data. Figure 4(b) places most test correction errors near zero, although a small number of observations extend beyond 10 g/s in either direction. On the validation split, the absolute base error and the residual score have a Spearman correlation of 0.59. The calibration selects the 94th percentile of observed absolute base error and the 93rd percentile of residual score, which gives the 11.92 g/s residual-score threshold drawn in Figure 4(d). Figure 4(c) gives a validation AUC of 0.99. Figure 4(d) then applies the locked thresholds to the test split and gives 110 true negatives, 10 true positives, 2 false positives, and 1 false negative. These counts correspond to a precision of 0.83, recall of 0.91, and $F _ { 1 }$ of 0.87. The score detects 10 of the 11 high-error Fuel Oil observations in the test split.

![](images/debf7065fdbfa40602a4a5b5b80cd0047d86a7a98379dbbc73fe652b50296c49.jpg)

(b)  
![](images/1986c1864edda3b97b1ea3eb418bc4f2e8244b2fa359784642de7231b8e5af48.jpg)

(c)  
![](images/283cc8f33a011b668e761bb980efcd525abc812c266a0ed8766db46c54ed17ea.jpg)

(d)  
![](images/ec56d5b1a89d5781e79408994f418d56a9eb7adcfccef32fcd4caaed316588ba.jpg)  
Figure 4: Fuel Oil residual and reliability-screening analysis showing (a) predicted and true signed residuals, (b) 50-bin test correction-error distribution, (c) validation ROC curve, and (d) residual-score classification profile on the test split with the locked threshold.

## 3.1.6 Optimisation Cases

CO<sub>2</sub>-constrained production study The refinery objective adds the normalised product predictions and subtracts the three scaled utility inputs. The optimisation problem is written as

$$
\begin{array} { r l r } { \operatorname* { m a x } \ } & { J _ { \mathrm { r e f } } ( x ) = \displaystyle \sum _ { y \in P _ { \mathrm { r e f } } } \frac { \hat { y } _ { y } ^ { D } ( x ) - y _ { \mathrm { r e f } } ^ { \mathrm { m i n } } } { y ^ { \mathrm { m a x } } - y _ { \mathrm { p e f } } ^ { \mathrm { m i n } } } - x _ { \mathrm { D u t y } } - x _ { \mathrm { G 1 } } - x _ { \mathrm { C 2 } } } \\ { \mathrm { s u b j e c t t ~ t o } } & { 8 0 \leq \displaystyle \sum _ { j \in C } \hat { x } _ { j } \leq 8 2 } \\ & { - 2 \leq \displaystyle \sum _ { j \in C } \hat { x } _ { j } - \displaystyle \sum _ { y \in P _ { \mathrm { r e f } } } \hat { y } _ { j } ^ { B } ( x ) \leq 2 } \\ & { \qquad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \displaystyle _ { y \mathrm { r o } } \ \sum _ { j } \hat { y } _ { p } ^ { B } ( x ) \leq \hat { y } _ { p } ^ { \mathrm { t m i n } } } & { \leq \hat { y } _ { p } ^ { \mathrm { t m i n } } } \\ & { \displaystyle _ { y \mathrm { C O } _ { 2 } } \ \leq \hat { y } _ { \mathrm { C O } _ { 2 } } ^ { \mathrm { t m i n } } \leq \hat { y } _ { \mathrm { C O } _ { 2 } } ^ { \mathrm { c o b } } } & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad p \in \mathcal { P } _ { \mathrm { r e f } } } \\ & { \displaystyle _ { B } ^ { 2 } ( \hat { x } ) \leq \hat { v } _ { \mathrm { r e f } } ^ { \mathrm { t m i n } } } &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \ \end{array}\tag{17}
$$

The refinery objective $J _ { \mathrm { r e f } } ( x )$ is dimensionless. $y _ { p } ^ { \mathrm { m i n } }$ and $y _ { p } ^ { \mathrm { m a x } }$ are the limits used to normalise each product prediction. The terms $x _ { \mathrm { D u t y } } , x _ { \mathrm { C 1 } }$ , and $x _ { \mathrm { { C 2 } } }$ are the scaled utility decisions. The set $\mathcal { C }$ contains the four crude feeds, and ${ \tilde { x } } _ { j }$ is the physical flow of feed $j .$ . The product set $\mathcal { P } _ { \mathrm { r e f } }$ contains Light Naphtha, Gases, Heavy Naphtha, Kerosene, $\mathrm { \bf A G O }$ , and Fuel Oil. Throughput is held between 80 and 82 kg/s. The approximate balance allows a difference of at most 2 kg/s between total crude feed and predicted product flow. $y _ { \mathrm { C O _ { 2 } } } ^ { \mathrm { L B } }$ is the lower bound on predicted $\mathrm { C O _ { 2 } } , y _ { \mathrm { C O _ { 2 } } } ^ { \mathrm { c a p } }$ is the selected $\mathrm { C O _ { 2 } }$ ceiling, and $\delta _ { \mathrm { r e f } }$ is the refinery Mahalanobis-distance limit. The objective assigns unit weights to the six scaled product predictions and three scaled utility penalties.

Figure 5 is a parallel-coordinate comparison of the grey refinery simulation observations and the coloured solutions obtained under different $\mathrm { C O _ { 2 } }$ ceilings. It compares the joint changes in crude blend, utility inputs, predicted emissions, and product rates while the input-domain constraint is maintained. The numerical solution values and residual assessments are given in Table 4.

![](images/284b9c9c6b316ed9fe670c4b3d0fd006d9a40f408f68e5e18780b326f9bd7377.jpg)  
Figure 5: Parallel-coordinate comparison of grey refinery simulation observations and coloured optimisation solutions under different $\mathrm { { C O } _ { 2 } }$ ceilings. The figure relates the crude blend and utility inputs to the predicted emissions and product rates for solutions that satisfy the imposed input-domain constraint.

All solutions in Table 4 exceed the locked Fuel Oil residual-score threshold of 11.92 g/s and are labelled WMR. The score is 13.1 g/s at the tightest $\mathrm { { C O } _ { 2 } }$ limit of 8,198.8 kg/h and rises to 101.5 g/s at 8,446.8 kg/h. The indicator half-width increases from 6.3 to 49.4 g/s over the same range. The first solution lies close to the threshold, whereas the remaining scores move progressively farther above it. These domain-compliant solutions have high Fuel Oil residual scores.

Table 4: Fuel Oil model-region assessment for the refinery $\mathrm { C O _ { 2 } }$ sweep. The band is centred on corrected Fuel Oil, and its half-width is given in g/s. $\mathrm { C O _ { 2 } }$ is rounded to one decimal place and the remaining quantities to two.
<table><tr><td> $\mathbf { C O _ { 2 } }$  limit (kg/h)</td><td>Base-model CO2 (kg/h)</td><td>Corrected Fuel Oil (kg/s)</td><td>Residual score (g/s)</td><td>Band half-width (g/s)</td><td>Model region</td></tr><tr><td>8,198.8</td><td>8,198.8</td><td>44.21</td><td>13.10</td><td>6.30</td><td>WMR</td></tr><tr><td>8,240.2</td><td>8,235.2</td><td>44.22</td><td>26.10</td><td>12.60</td><td>WMR</td></tr><tr><td>8,281.5</td><td>8,276.5</td><td>44.23</td><td>35.10</td><td>17.50</td><td>WMR</td></tr><tr><td>8,322.8</td><td>8,317.8</td><td>44.25</td><td>49.70</td><td>23.60</td><td>WMR</td></tr><tr><td>8,364.2</td><td>8,359.2</td><td>44.26</td><td>65.70</td><td>32.90</td><td>WMR</td></tr><tr><td>8,405.5</td><td>8,400.5</td><td>44.28</td><td>84.30</td><td>41.10</td><td>WMR</td></tr><tr><td>8,446.8</td><td>8,441.8</td><td>44.30</td><td>101.50</td><td>49.40</td><td>WMR</td></tr><tr><td>Unconstrained</td><td>8,198.8</td><td>44.21</td><td>13.10</td><td>6.30</td><td>WMR</td></tr></table>

Arab Heavy contract sensitivity In this case, the Arab Heavy operating level is held near a prescribed scaled target while the remaining variables are re-optimised. Mathematically, this constraint is written as

$$
- 0 . 0 1 \leq x _ { \mathrm { A r a b H e a v y } } - x _ { \mathrm { A r a b H e a v y } } ^ { \mathrm { t a r g e t } } \leq 0 . 0 1 .\tag{18}
$$

Here, $x _ { \mathrm { A r a b H e a v y } }$ is the optimised scaled Arab Heavy input and $x _ { \mathrm { A r a b H e a v y } } ^ { \mathrm { t a r g e t } }$ is its prescribed scaled target. Because the physical Arab Heavy range is 0.12 to 58.18 kg/s, a scaled tolerance of 0.01 corresponds to about 0.58 kg/s. This explains the difference between nominal and optimised physical values in Table 5.

The contract sensitivity examines changes in the refinery predictions as Arab Heavy availability varies while the total crude-flow constraint is preserved. Figure 6(a) presents the re-optimised crude blend as the Arab Heavy target decreases from 51.73 to 0.12 kg/s. In Figure 6(a), Arab Medium and Kuwait provide most of the replacement feed, while Basrah remains comparatively small. Figure 6(b) relates the resulting Kerosene and corrected Fuel Oil predictions. In Figure 6(b), Kerosene rises from about 12.2 to 13.1 kg/s, whereas Fuel Oil falls from 46.55 to 44.22 kg/s. These values convey a predicted shift in the product slate towards Kerosene and away from Fuel Oil as Arab Heavy availability is reduced.

![](images/f3a4cc05971a012b7f1566f7fab01754179aee2717888750be8ce7442e943728.jpg)

(b)  
![](images/90be9f5e245ee858e9034e4cf8b234b3f0ebacb12cfb753ddf0cac752eccab74.jpg)  
Figure 6: Arab Heavy sensitivity results showing (a) re-optimised crude blend as the Arab Heavy target falls and (b) resulting changes in kerosene and corrected Fuel Oil predictions. Both panels show a predicted shift towards kerosene as Arab Heavy availability decreases.

For Arab Heavy targets down to 19.48 kg/s, the solutions remain in GMR (Table 5). Lower targets move to WMR as Arab Medium and Kuwait replace more of the Arab Heavy feed under the same throughput constraint. The resulting change in product slate is accompanied by larger predicted Fuel Oil corrections. The score at the 13.02 kg/s target is only 0.38 g/s above $T _ { s }$

Table 5: Fuel Oil model-region assessment for the Arab Heavy sensitivity. The band is centred on corrected Fuel Oil, and its half-width is given in g/s.
<table><tr><td>Arab Heavy target (kg/s)</td><td>Optimised Arab Heavy (kg/s)</td><td>Corrected Fuel Oil (kg/s)</td><td>Residual score (g/s)</td><td>Band half-width (g/s)</td><td>Model region</td></tr><tr><td>51.73</td><td>51.15</td><td>46.55</td><td>9.20</td><td>4.50</td><td>GMR</td></tr><tr><td>45.28</td><td>44.69</td><td>45.98</td><td>11.00</td><td>5.30</td><td>GMR</td></tr><tr><td>38.83</td><td>38.24</td><td>45.68</td><td>10.40</td><td>5.20</td><td>GMR</td></tr><tr><td>32.38</td><td>31.79</td><td>45.41</td><td>7.30</td><td>3.50</td><td>GMR</td></tr><tr><td>25.93</td><td>25.34</td><td>45.11</td><td>6.00</td><td>3.00</td><td>GMR</td></tr><tr><td>19.48</td><td>18.89</td><td>44.78</td><td>8.70</td><td>4.20</td><td>GMR</td></tr><tr><td>13.02</td><td>12.44</td><td>44.43</td><td>12.30</td><td>6.00</td><td>WMR</td></tr><tr><td>6.57</td><td>7.16</td><td>44.21</td><td>14.60</td><td>7.00</td><td>WMR</td></tr><tr><td>0.12</td><td>0.71</td><td>44.22</td><td>23.70</td><td>11.90</td><td>WMR</td></tr></table>

## 3.2 Case Study 2. 395 MW Industrial Gas Turbine System

## 3.2.1 System and data description

The plant study by Ashraf and Dua [7] provides records from a 395 MW gas turbine unit for the industrial case. Ambient air is compressed before entering the combustion system. Preheated natural gas is mixed with the compressed air and undergoes combustion. The hot combustion products expand through the gas turbine to drive the electrical generator. The exhaust supplies a heat recovery steam generator for the associated steam cycle. The present analysis models the gas turbine at unit level through the variables that describe its air path, fuel path, ambient conditions, and performance.

The selection of input variables follows the source study [7], in which plant process and performance engineers identified variables used during power ramp-up and ramp-down. Compressor discharge pressure (CDP) and compressor discharge temperature (CDT) represent the compressed-air state before combustion. Gas fuel flow rate (GFFR), performanceheater gas outlet temperature (PHGOT), and fuel gas temperature at the combustion-chamber inlet (FGT) describe the fuel-side operating state. Ambient temperature (AT), ambient pressure (AP), and ambient humidity (AH) represent the air conditions of the site. The three output variables to be modelled are unit electrical power generation (Power), thermal efficiency (TE), and turbine heat rate (THR). The descriptive statistics of the dataset [7] are provided in Table 6. The input and output variables cover wide operating ranges, and Power varies from 185.7 to 395.7 MW. These ranges indicate that ramping operation of the gas turbine is represented in the data.

Table 6: Descriptive statistics for the gas turbine dataset. All values use two decimal places.
<table><tr><td>Variable</td><td>Minimum</td><td>Median</td><td>Maximum</td><td>Mean</td><td>SD</td></tr><tr><td>Compressor discharge temperature (°F)</td><td>813.00</td><td>847.80</td><td>925.80</td><td>861.00</td><td>34.20</td></tr><tr><td>Compressor discharge pressure (psi)</td><td>185.90</td><td>232.70</td><td>312.00</td><td>247.90</td><td>36.90</td></tr><tr><td>Gas fuel flow rate (lb/s)</td><td>28.56</td><td>37.81</td><td>49.83</td><td>39.20</td><td>5.67</td></tr><tr><td>Performance-heater gas outlet temperature (°F)</td><td>399.50</td><td>411.20</td><td>424.90</td><td>411.20</td><td>2.50</td></tr><tr><td>Fuel gas temperature at combustor inlet (°F)</td><td>484.00</td><td>517.60</td><td>534.90</td><td>513.50</td><td>14.80</td></tr><tr><td>Ambient temperature (°C)</td><td>20.50</td><td>25.60</td><td>34.80</td><td>26.30</td><td>3.60</td></tr><tr><td>Ambient pressure (hPa)</td><td>982.70</td><td>988.70</td><td>991.80</td><td>988.10</td><td>2.00</td></tr><tr><td>Ambient humidity (%)</td><td>34.20</td><td>67.60</td><td>98.00</td><td>66.40</td><td>14.10</td></tr><tr><td>Thermal efficiency (%)</td><td>32.44</td><td>39.02</td><td>43.05</td><td>38.96</td><td>2.38</td></tr><tr><td>Power (MW)</td><td>185.70</td><td>280.30</td><td>395.70</td><td>296.40</td><td>59.50</td></tr><tr><td>Turbine heat rate (kJ/kWh)</td><td>8,363.00</td><td>9,227.00</td><td>11,099.00</td><td>9,275.00</td><td>585.00</td></tr></table>

## 3.2.2 Base and corrected surrogate performance

Residual correction is evaluated separately for Power, TE, and THR. Figure 7(a)(i) places the Power predictions close to the one-to-one line, with a test RMSE of 1.02 MW over the observed range of 185.7 to 395.7 MW and $R ^ { 2 } > 0 . 9 9 $ Figure 7(a)(iii) gives a TE test RMSE of 0.46 percentage points and $R ^ { 2 } = 0 . 9 6$ . In contrast, Figure 7(a)(ii) shows wider scatter for THR, particularly at higher values, with a test RMSE of 151 kJ/kWh and $R ^ { 2 } = \stackrel { \smile } { 0 . 9 4 }$ . Figure 7(b)(i)–(iii) compares measured, base, and corrected THR distributions on the training, validation, and test splits. Residual correction lowers THR error across all three splits, with the largest relative validation gain among the gas turbine outputs. Test RMSE falls from 151 to 105 kJ/kWh.

(a)  
![](images/b89ee937d92c9fbea8133f8ec6bc9028e78c8b953304272c050ea864c529ad57.jpg)

![](images/80c7558dd183daecd6b13bbe2e700e8811b14fbf838f56892d51cfd838e17203.jpg)

![](images/11b992e2e4bec2f33b0e1ec38bd98f07f595f36aaff46a5ed3e66ddf77ced779.jpg)

(b)  
![](images/b7d05f5ac122c697ea9902a0d645bd52c6b6f81b9e5c33e77513b628f96db9e6.jpg)

![](images/a43f78b9f93a618410339bf09c2af9ed2e5cb2587393544564ecb39bb0ca2fb9.jpg)

![](images/a1cee57f530244aaa216134a081fa01ebdfa569907ba9517cae494dec67daba7.jpg)

![](images/b2d499fbd1d7ba9f3ff10dad6685af5a05c446793a45c8640440866579ea13c4.jpg)  
Figure 7: Gas turbine model performance showing (a) base-model parity plots for Power, THR, and TE, alongside (b) density curves of measured, base, and corrected THR on the training, validation, and test splits.

The effect of correction differs among the gas turbine outputs (Table 7). Power test RMSE increases from 1.02 to 1.08 MW, while TE test RMSE remains at 0.46 percentage points. THR validation RMSE decreases from 158 to 99 kJ/kWh, which is a 37% improvement. Its test RMSE decreases from 151 to 105 kJ/kWh and test $R ^ { 2 }$ rises from 0.94 to 0.97. THR has the largest validation improvement and is selected for the detailed reliability analysis.

Table 7: Gas turbine base and residual-corrected performance.
<table><tr><td rowspan="2">Target</td><td rowspan="2">Metric</td><td colspan="2">Training</td><td colspan="2">Validation</td><td colspan="2">Test</td></tr><tr><td>Base</td><td>Corrected</td><td>Base</td><td>Corrected</td><td>Base</td><td>Corrected</td></tr><tr><td>Power (MW)</td><td>RMSE  $R ^ { 2 }$ </td><td>0.90 0.9998</td><td>0.79 0.9998</td><td>0.95 0.9997</td><td>0.84 0.9998</td><td>1.02 0.9997</td><td>1.08 0.9997</td></tr><tr><td>THR (kJ/kWh)</td><td>RMSE  $R ^ { 2 }$ </td><td>147.27 0.9342</td><td>88.00 0.9765</td><td>158.19 0.9334</td><td>99.02 0.9739</td><td>151.08 0.9361</td><td>104.86 0.9692</td></tr><tr><td>Thermal efficiency (%)</td><td>RMSE  $R ^ { 2 }$ </td><td>0.38 0.9739</td><td>0.38 0.9738</td><td>0.39 0.9740</td><td>0.39 0.9743</td><td>0.46 0.9636</td><td>0.46 0.9638</td></tr></table>

## 3.2.3 THR residual reliability diagnostic

THR residual-score screening is evaluated against the high-error reference labels. Figure 8(a) compares predicted and true signed residuals and gives Spearman correlations of 0.80 for training, 0.75 for validation, and 0.72 for testing. Figure 8(a) also reports residual RMSE values of 88, 99, and 105 kJ/kWh and MAE values of 68, 80, and 83 kJ/kWh across the same splits. These values indicate that the model captures much of the direction and ordering of the required correction. The test correction-error distribution is centred near zero but extends from about −300 to +300 kJ/kWh (Figure 8(b)). Residual correction lowers average THR error, although some individual corrections remain inaccurate. Screening uses the residual magnitude rather than its sign. On the validation split, the Spearman correlation between absolute base error and residual score is 0.35. The calibration selects an observed-error threshold of 208.62 kJ/kWh and a residual-score threshold of 161.32 kJ/kWh, and Figure 8(c) gives a validation AUC of 0.78. Figure 8(d) applies these fixed thresholds to the test split and gives 79 true negatives, 12 true positives, 8 false positives, and 10 false negatives. These counts correspond to a precision of 0.60, recall of 0.55, and $\bar { F _ { 1 } } ^ { \bar { } }$ of 0.57, which indicates moderate discrimination.

![](images/5267ed302168a21777434563e28ddbe5bbbe05ec987be79ac935513d4b724b10.jpg)

(b)  
![](images/0bb228337b19bdee223f54fb34bca4f50b739ae6ea3239aa186f5e9068e699a5.jpg)

(c)  
![](images/143086d05e3648289f094486c0406998681df43e5411900d76d7ef01b0e11567.jpg)

(d)  
![](images/299b31f1609adc7e5b6d4d3eb9c92d65e11ea3ad4f0f820cdbc8279ae36353ae.jpg)  
Figure 8: THR residual and reliability-screening analysis showing (a) predicted and true signed residuals, (b) test correction-error distribution, (c) validation ROC curve, and (d) residual-score density and test classification with the locked threshold.

## 3.2.4 Thermal-efficiency optimisation across fuel-flow targets

The gas turbine study maximises the base-model TE prediction while keeping the three base outputs within their observed ranges. The Mahalanobis limit is set to the 95th percentile of the reference distances. A scaled GFFR target is imposed to examine the solution along the fuel-flow range. The optimisation problem for the gas turbine is written as

$$
\begin{array} { r l r } { \underset { x } { \operatorname* { m a x } } } & { \hat { y } _ { \mathrm { T E } } ^ { B } ( x ) } \\ { \mathrm { s u b j e c t ~ t o } } & { y _ { q } ^ { \mathrm { L B } } \leq \hat { y } _ { q } ^ { B } ( x ) \leq y _ { q } ^ { \mathrm { U B } } } & { q \in \{ \mathrm { T E } , \mathrm { P o w e r } , \mathrm { T H R } \} } \\ & { D _ { M } ^ { 2 } ( \hat { x } ) \leq \delta _ { 9 5 } ^ { 2 } } \\ & { 0 \leq x _ { j } \leq 1 } & { j \in \mathcal { X } } \\ & { - 0 . 0 1 \leq x _ { \mathrm { G F F R } } - x _ { \mathrm { G F F R } } ^ { \mathrm { t a r g e t } } \leq 0 . 0 1 . } \end{array}\tag{19}
$$

In Equation $1 9 , \hat { y } _ { \mathrm { T E } } ^ { B } ( x )$ is the base-ANN prediction of thermal efficiency and $q$ indexes the constrained outputs TE, Power, and THR. $\bar { y } _ { q } ^ { \mathrm { { L B } } }$ and $y _ { q } ^ { \mathrm { U B } }$ are the corresponding output bounds. $\delta _ { 9 5 }$ is the 95th-percentile Mahalanobis-distance limit, and x<sup>target</sup><sub>GFFR</sub> is the prescribed scaled GFFR target around which x<sub>GFFR</sub> may vary by 0.01. The nominal scaled targets are 0.2, 0.4, 0.6, 0.8, and 1.0. Using the observed range of 28.56 to 49.83 lb/s, the operating levels of GFFR correspond to nominal physical targets of 32.81, 37.07, 41.32, 45.58, and 49.83 lb/s. The scaled tolerance of 0.01 permits an adjustment of about 0.21 lb/s. IPOPT solves each case through the GAMS solver interface, after which the THR residual model supplies the signed correction, its magnitude as the score, and the indicator band.

Table 8: Gas turbine optimisation and post-optimisation THR assessment. The THR indicator band is centred on the corrected THR prediction.
<table><tr><td>Scaled GFFR</td><td>Nominal GFFR (lb/s)</td><td>Base-model TE (%)</td><td>Signed THR residual (kJ/kWh)</td><td>THR indicator band (kJ/kWh)</td><td>Model region</td></tr><tr><td>0.2</td><td>32.81</td><td>38.74</td><td>-462.31</td><td> $9 , 2 0 8 . 4 5 \pm 2 2 4 . 2 4$ </td><td>WMR</td></tr><tr><td>0.4</td><td>37.07</td><td>41.30</td><td>-424.97</td><td> $8 , 7 2 7 . 0 5 \pm 2 0 6 . 1 4$ </td><td>WMR</td></tr><tr><td>0.6</td><td>41.32</td><td>42.94</td><td>-229.65</td><td> $8 , 4 3 2 . 2 4 \pm 1 1 4 . 4 6$ </td><td>WMR</td></tr><tr><td>0.8</td><td>45.58</td><td>43.05</td><td>-106.29</td><td> $8 , 4 8 0 . 8 6 \pm 5 0 . 3 8$ </td><td>GMR</td></tr><tr><td>1.0</td><td>49.83</td><td>42.44</td><td>163.06</td><td> $8 , 5 2 5 . 7 3 \pm 8 1 . 5 3$ </td><td>WMR</td></tr></table>

Predicted TE reaches the observed upper bound of 43.05% at the 0.8 GFFR target (Table 6), then falls at the 1.0 target. At the 0.8 target, the TE bound is active and the THR residual score gives the only GMR label. The large negative THR corrections at the three lower GFFR targets shift to a positive correction at the highest target, whose score is only 1.74 kJ/kWh above $T _ { s }$ (Table 8).

## 4 Conclusion

ML surrogates can be accurate on their aggregated scores while their predictive accuracy varies across the local input space. The variation in local predictive accuracy limits their use in the decision-making for operation of safety-critical thermal systems. This paper developed the TSRL framework, which attaches a point-specific modelling-strength label (good or weak) to each computed optimal solution, facilitating the decision-making for the operation of thermal systems. As a case study, The TSRL framework is applied to a simulated crude distillation unit and a 395 MW industrial gas turbine and the strength of computed optimal solutions under various operating scenarios is investigated.

Residual correction improved Fuel Oil and THR predictions, although its effect depended on the output. Screening was more effective for Fuel Oil than for THR. All eight solutions of the refinery $\mathrm { C O _ { 2 } }$ sweep satisfied the Mahalanobis constraint but were labelled WMR, and the Arab Heavy sensitivity moved to WMR at the three lowest targets. For the gas turbine, the scaled GFFR target of 0.8 gave the highest predicted TE and the only GMR label among the five targets.

The domain constraints keep the solution search close to the data envelopes, while the residual score distinguishes GMR and WMR within the data-driven domain. The optimal solutions computed through TSRL ensures input-domain compliance and providing the information about the local modelling strength which facilitates the decision-making for the operation of thermal systems.

## Limitations and future work

The thresholds are calibrated for one simulated refinery dataset and one gas turbine unit, which limits their transfer across operating regimes. Transfer to another unit will require checking score calibration against its operating data and, where available, an engineering error limit. The present GMR and WMR labels remain relative to validation error percentiles. A single random split was used for each case, leaving sensitivity to alternative partitions and temporal drift unquantified. Repeated and chronological partitions could assess this sensitivity as additional industrial units and operating regimes are tested. Screened optimisation solutions also require independent checks through simulator reruns or controlled plant trials.

## Author statement

Talha Ansar: Methodology, Formal Analysis, Writing – Original Draft, Waqar Muhammad Ashraf: Conceptualization, Validation, Data Curation, Methodology, Writing – Review & Editing, Abdulelah S. Alshehri: Methodology, Writing – Review & Editing, Funding Omar Bayomie: Conceptualization, Methodology, Validation, Data Curation, Software, Writing – Review & Editing, Project Supervision.

## Acknowledgements

The authors acknowledge that they have used LLMs (Claude - Opus) for improving grammar and enhancing the readability of manuscript. They have thoroughly reviewed the article and take full responsibility of the work. Abdulelah S. Alshehri would like extend his appreciation to King Saud University for funding his research through the Ongoing Research Funding program, (ORF-2026-1455), King Saud University, Riyadh, Saudi Arabia.

## Declaration of competing interests

The authors declare no competing financial or non-financial interests.

## References

[1] Yogesh Jaluria. Design and Optimization of Thermal Systems: With MATLAB Applications. CRC Press, 3 edition, 2019.

[2] Alexander I. J. Forrester and Andy J. Keane. Recent advances in surrogate-based optimization. Progress in Aerospace Sciences, 45(1–3):50–79, 2009.

[3] Atharv Bhosekar and Marianthi Ierapetritou. Advances in surrogate based modeling, feasibility analysis, and optimization: A review. Computers & Chemical Engineering, 108:250–267, 2018.

[4] Taoufiq Gueddar and Vivek Dua. Novel model reduction techniques for refinery-wide energy optimisation. Applied Energy, 89(1):117–126, 2012.

[5] Muhammad Ahmad Jamil, Waqar Muhammad Ashraf, Nida Imtiaz, Ben Bin Xu, Syed M. Zubair, Haseeb Yaqoob, Muhammad Imran, and Muhammad Wakil Shahzad. Machine learning-based process design of a novel sustainable cooling system. Energy Conversion and Management, 319:118941, 2024.

[6] Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. Neural Networks, 2(5):359–366, 1989.

[7] Waqar Muhammad Ashraf and Vivek Dua. Driving towards net-zero from the energy sector: Leveraging machine intelligence for robust optimization of coal and combined cycle gas power stations. Energy Conversion and Management, 314:118645, 2024.

[8] Artur M. Schweidtmann and Alexander Mitsos. Deterministic global optimization with artificial neural networks embedded. Journal of Optimization Theory and Applications, 180(3):925–948, 2019.

[9] Francesco Ceccon, Jordan Jalving, Joshua Haddad, Alexander Thebelt, Calvin Tsay, Carl D. Laird, and Ruth Misener. OMLT: Optimization & machine learning toolkit. Journal ofMachine Learning Research, 23(349):1–8, 2022.

[10] Waqar Muhammad Ashraf, Abdulelah S. Alshehri, Abdulrahman bin Jumah, Ghulam Moeen Uddin, Muhammad Akhtar, and Vivek Dua. Domain-enforced and operator-in-the-loop neural simulation platform for techno-enviroeconomic performance enhancement of gas turbine system. Computers & Chemical Engineering, 206:109529, 2026.

[11] Ruth Misener and Lorenz Biegler. Formulating data-driven surrogate models for process optimization. Computer & Chemical Engineering, 179:108411, 2023.

[12] Yun Xu and Royston Goodacre. On splitting training and validation set: A comparative study of cross-validation, bootstrap and systematic sampling for estimating the generalization performance of supervised learning. Journal ofAnalysis and Testing, 2(3):249–262, 2018.

[13] Sudhir Varma and Richard Simon. Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics, 7(1):91, 2006.

[14] Olaf Kahrs and Wolfgang Marquardt. The validity domain of hybrid models and its application in process optimization. Chemical Engineering and Processing: Process Intensification, 46(11):1054–1066, 2007.

[15] Artur M. Schweidtmann, Jana M. Weber, Christian Wende, Linus Netze, and Alexander Mitsos. Obey validity limits of data-driven models through topological data analysis and one-class classification. Optimization and Engineering, 23(2):855–876, 2022.

[16] Hanna Meyer and Edzer Pebesma. Predicting into unknown space? estimating the area of applicability of spatial prediction models. Methods in Ecology and Evolution, 12(9):1620–1633, 2021.

[17] Waqar Muhammad Ashraf, Amir H. Keshavarzzadeh, Abdulelah S. Alshehri, Abdulrahman bin Jumah, Ramit Debnath, and Vivek Dua. Domain-informed operation excellence of gas turbine system with machine learning. Energy Conversion and Management, 348:120789, 2026.

[18] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In Proceedings of the 33rd International Conference on Machine Learning, volume 48 of Proceedings ofMachine Learning Research, pages 1050–1059. PMLR, 2016.

[19] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

[20] Jakob Gawlikowski, Cedrique Rovile Njieutcheu Tassi, Mohsin Ali, Jongseok Lee, Matthias Humt, Jianxiang Feng, Anna Kruspe, Rudolph Triebel, Peter Jung, Ribana Roscher, et al. A survey of uncertainty in deep neural networks. Artificial Intelligence Review, 56(Suppl 1):1513–1589, 2023.

[21] Armando Cajahuaringa, Rubén Aquize Palacios, Juan M Mauricio Villanueva, Aurelio Morales-Villanueva, José Machuca, Juan Contreras, and Kiara Rodríguez Bautista. Uncertainty evaluation of a gas turbine model based on a nonlinear autoregressive exogenous model and monte carlo dropout. Sensors, 24(2):465, 2024.

[22] Shu-Bo Yang, Zukui Li, and Wei Wu. Data-driven process optimization considering surrogate model prediction uncertainty: A mixture density network-based approach. Industrial & Engineering Chemistry Research, 60(5):2206–2222, 2021.

[23] Jerome H. Friedman. Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5):1189–1232, 2001.

[24] Marc C. Kennedy and Anthony O’Hagan. Bayesian calibration of computer models. Journal of the Royal Statistical Society. Series B, 63(3):425–464, 2001.

[25] Moritz von Stosch, Rui Oliveira, Joana Peres, and Sebastião Feyo de Azevedo. Hybrid semi-parametric modeling in process systems engineering: Past, present and future. Computers & Chemical Engineering, 60:86–101, 2014.

[26] William Bradley, Jinhyeun Kim, Zachary Kilwein, Logan Blakely, Michael Eydenberg, Jordan Jalvin, Carl Laird, and Fani Boukouvala. Perspectives on the integration between first-principles and data-driven modeling. Computers & Chemical Engineering, 166:107898, 2022.

[27] Salem Lahlou, Moksh Jain, Hadi Nekoei, Victor Ion Butoi, Paul Bertin, Jarrid Rector-Brooks, Maksym Korablyov, and Yoshua Bengio. DEUP: Direct epistemic uncertainty prediction. arXiv preprint arXiv:2102.08501, 2021.

[28] Zuming Liu and Iftekhar A Karimi. Gas turbine performance prediction via machine learning. Energy, 192:116627, 2020.

[29] Waqar Muhammad Ashraf and Vivek Dua. Data information integrated neural network (DINN) algorithm for modelling and interpretation performance analysis for energy systems. Energy and AI, 16:100363, 2024.

[30] Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural Networks, 107:3–11, 2018.

[31] James Bergstra, Brent Komer, Chris Eliasmith, Dan Yamins, and David D. Cox. Hyperopt: A Python library for model selection and hyperparameter optimization. Computational Science & Discovery, 8(1):014008, 2015.

[32] James Bergstra, Rémi Bardenet, Yoshua Bengio, and Balázs Kégl. Algorithms for hyper-parameter optimization. In Advances in Neural Information Processing Systems, volume 24, pages 2546–2554. Curran Associates, Inc., 2011.

[33] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

[34] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, volume 32, pages 8024–8035. Curran Associates, Inc., 2019.

[35] Lutz Prechelt. Early stopping—but when? In Genevieve B. Orr and Klaus-Robert Müller, editors, Neural Networks: Tricks ofthe Trade, pages 55–69. Springer, Berlin, 1998.

[36] Michael L. Bynum, Gabriel A. Hackebeil, William E. Hart, Carl D. Laird, Bethany L. Nicholson, John D. Siirola, Jean-Paul Watson, and David L. Woodruff. Pyomo—Optimization Modeling in Python. Springer, Cham, 3 edition, 2021.

[37] Hamid Ghorbani. Mahalanobis distance and its application for detecting multivariate outliers. Facta Universitatis, Series: Mathematics and Informatics, 34(3):583–595, 2019.

[38] Roger Penrose. A generalized inverse for matrices. Mathematical Proceedings ofthe Cambridge Philosophical Society, 51(3):406–413, 1955.

[39] Martin Rätz, Patrick Henkel, Phillip Stoffel, Rita Streblow, and Dirk Müller. Identifying the validity domain of machine learning models in building energy systems. Energy and AI, 15:100324, 2024.

[40] Andreas Wächter and Lorenz T. Biegler. On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming. Mathematical Programming, 106(1):25–57, 2006.

[41] Omar S. Bayomie, Omar Y. Abdelaziz, and Mamdouh A. Gadalla. Exceeding pinch limits by process configuration of an existing modern crude oil distillation unit – a case study from refining industry. Journal of Cleaner Production, 231:1050–1058, 2019.

[42] M. D. McKay, R. J. Beckman, and W. J. Conover. Comparison of three methods for selecting values of input variables in the analysis of output from a computer code. Technometrics, 21(2):239–245, 1979.

[43] D. Donovan, K. Burrage, P. Burrage, T.A. McCourt, B. Thompson, and E.¸S. Yazici. Estimates of the coverage of parameter space by latin hypercube and orthogonal array-based sampling. Applied Mathematical Modelling, 57:553–564, 2018.

[44] Anikesh Kumar, Chi Hung Vo, M Shahabuddin, Sushant S Garud, and Iftekhar A Karimi. Integrating coding platforms with process simulators for custom applications. Computers & Chemical Engineering, 201:109247, 2025.