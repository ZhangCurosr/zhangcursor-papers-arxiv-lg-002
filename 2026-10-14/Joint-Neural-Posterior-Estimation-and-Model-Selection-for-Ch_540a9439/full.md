# Joint Neural Posterior Estimation and Model Selection for Chemical Reaction Networks

J. Arturo Esquivel¹, Jorge A. Campos-Gonzalez-Angulo2,3,4, Sergio Pablo-García5,6, Alán Áspuru-Guzik5,7,8,9,10,11, and Vianey Leos-Barajas1,12

1Department of Statistical Sciences, University of Toronto, Toronto, Ontario, Canada

2Schools of Mathematics & Physics and Chemistry & Molecular Biosciences, The University of Queensland, St Lucia, Queensland, Australia

3Australian Research Council Centre of Excellence in Quantum Biotechnology, St Lucia, Queensland, Australia 4Queensland Quantum Decarbonisation Alliance, St Lucia, Queensland, Australia

5The Matter Lab, University of Toronto, Toronto, Ontario, Canada 6Institute of Micro and Nanotechnology

7Departments of Chemistry, Computer Science, Materials Science & Engineering, Chemical Engineering & Applied Chemistry, and Institute of Medical Science, University of Toronto, Toronto, Ontario, Canada 8Acceleration Consortium, Toronto, Ontario, Canada

9Vector Institute for Artificial Intelligence, Toronto, Ontario, Canada 10Canadian Institute for Advanced Research (CIFAR), Toronto, Ontario, Canada 11NVIDIA, Toronto, Ontario, Canada

12School of the Environment, University of Toronto, Toronto, Ontario, Canada

## Abstract

Simulation-based Bayesian inference for Chemical Reaction Networks (CRNs) must simultaneously solve two computationally demanding problems: discriminating between overlapping reaction pathways and estimating reaction rate constants. Classical approaches, such as Sequential Monte Carlo ABC, rely on handcrafted summary statistics and scale poorly across large parameter spaces. To overcome this, we introduce a joint amortized inference architecture. Our pipeline utilizes shared 1D-CNNs and a permutation-invariant Set Attention mechanism to learn low-dimensional summary statistics from variable-length, multi-experiment time-series data. These statistics simultaneously condition a model selection network to compute posterior model probabilities and model-specific conditional Invertible Neural Networks (cINNs) that estimate parameter posterior distributions. We demonstrate the practical utility of our approach on a real protodeboronation example, illustrating how it handles structural ambiguity and extracts kinetic parameters from competing catalyst deactivation pathways.

## 1 Introduction

Chemical reaction networks (CRNs) are at the heart of many societal and scientific concerns, such as designing efficient industrial chemical processes [Blau et al., 2021], optimizing pharmaceutical production [Dong et al., 2023], and addressing climate-related challenges [Heald and Kroll, 2020]. Chemists decompose complex chemical processes into diagrams with elementary (or intermediate) reactions that occur over time. Their dynamics are dictated by reaction rate constants—quantitative measures of the propensity that an elementary reaction occurs within the network [Gillespie, 2007]. In practice, these underlying dynamics are studied by analyzing time-varying concentration data of the chemical species within the system.

A complete CRN describes all intermediate reactions that occurred within a specified time frame. Yet, chemical concentrations are often measured sparsely across time, and it is frequently only possible to measure a few of the chemical species involved in the process [Jovanovski et al., 2025]. Consequently, different combinations of elementary reactions (CRN structures) can adequately describe similar concentration profiles (time-series). The scientific challenge lies in: (1) determining the valid graphical representation of the CRN that includes all elementary reactions; (2) robustly estimating the elementary reaction rates; and (3) inferring the changes that unmeasured species experience over time [Raue et al., 2009].

Estimating the reaction rates is challenged by intractable chemical kinetics likelihood functions. Approximate Bayesian computation (ABC), and its extensions, is the standard for this kind of likelihood-free scenarios. But, it scales poorly when evaluating large parameter spaces and relies heavily on the choice of summary statistics and rejection thresholds. To conduct model selection, ABC typically relies on estimating Bayes Factors via expensive sequential Monte Carlo sampling |Toni et al., 2008, Didelot et al., 2011], a process that scales poorly in the context of CRNs, as the networks complexities increase.

Recent breakthroughs, most notably by Burés and Larrosa [2023], have demonstrated the utility of deep learning for studying reaction mechanisms (CRNs). Their approach trains a deep neural network to map kinetic data to mechanistic categories. While groundbreaking, they approach the problem solely as a classification task, and omit the continuous parameter space and associated uncertainty quantification. We generalize and expand upon this foundational idea. Our architecture utilizes normalizing flows, set attention, and a model neural network to: (1) quantify model uncertainty, learning calibrated posterior probabilities from chemical data; (2) estimate posterior distributions for the parameters governing the reactions involved in each model (CRN) considered; and (3) amortize the cost of inference, allowing for cheap evaluation of newly observed chemical data. We note that a similar architecture (combining Set Transformers with Convolutional Neural Networks and Normalizing flows) is concurrently being explored as a software tool for inference in electrochemistry [Yan, Bing, 2026]. Our inference pipeline is tailored to accommodate time-series of variable length across multiple experimental conditions as commonly observed when studying kinetic chemical systems.

## 2 Methods

## 2.1 Chemical Reaction Networks

Consider a network representing a chemical process composed of $n _ { r }$ elementary reactions involving $n _ { s }$ distinct chemical species. The dynamic evolution of this system is parameterized by a vector of reaction rate constants k $\mathbf { \Psi } _ { \cdot } \in \mathbb { R } _ { + } ^ { n _ { r } }$ . For a well-mixed system with constant temperature, the state of the network at time t is given by the $n _ { s }$ -dimensional vector of the number of molecules per species, $\mathbf X ( t )$ The state changes of the network are encoded in the $n _ { s } \times n _ { r }$ stoichiometric matrix $\mathbf { S } _ { \mathrm { i } }$ where the entry $S _ { i j }$ represents the net change in the number of molecules of species i upon the occurrence of reaction j.

The rate at which these reactions occur is determined by an $n _ { r }$ -dimensional propensity vector, ${ \mathbf { h } } ( { \mathbf { X } } ( t ) , { \mathbf { k } } )$ [Gillespie, 2007]. Let ${ \bf S } ^ { - }$ be the $n _ { s } \times n _ { r }$ reactant stoichiometric matrix, where each entry $S _ { i j } ^ { - }$ specifies the number of molecules of species i consumed by reaction $j .$ Under mass-action kinetics, the propensity $h _ { j }$ of the j-th reaction is proportional to the product of its available reactants:

$$
h _ { j } ( \mathbf { X } ( t ) , \mathbf { k } ) = k _ { j } \prod _ { i = 1 } ^ { n _ { s } } X _ { i } ( t ) ^ { S _ { i j } ^ { - } } , \quad \mathrm { f o r } \ j = 1 , \ldots , n _ { r } .\tag{1}
$$

When the volume and the number of molecules are sufficiently large, stochastic fluctuations become marginal [Fearnhead et al., 2014] and changes in the system can be well approximated by deterministic Ordinary Differential Equations (ODEs):

$$
\frac { d { \bf X } ( t ) } { d t } = { \bf S h } ( { \bf X } ( t ) , { \bf k } ) .\tag{2}
$$

We employ this ODE formulation as our primary simulator. To perform Bayesian inference on k, evaluating the likelihood of the observed data requires numerically integrating these non-linear ODEs. Because this integration is computationally expensive, exact inference via Markov chain Monte Carlo (MCMC) quickly becomes prohibitive, as it demands sequential simulations for every proposed parameter step. Furthermore, the observable, typically consisting of multiple independent experiments $\mathbf { \bar { y } } _ { 0 } = \{ f ( \mathbf { X } ^ { ( 1 ) } ) , \dotsc , f ( \mathbf { X } ^ { ( N _ { \mathrm { e x p } } ) } ) \}$ , contains measurements from only a subset of the chemical species in the system. Because intermediate species are often not observed, the kinetic parameters k can be unidentifiable, requiring exhaustive exploration of the parameter space to accurately estimate their posterior distributions.

## 2.2 Classical Likelihood-Free Baseline

To bypass the problems of exact inference, Simulation-Based Inference (SBI) relies on generating synthetic datasets from the simulator. The most widely used traditional approach is Sequential Monte Carlo Approximate Bayesian Computation (SMC-ABC) [Sisson et al., 2007]. SMC-ABC utilizes sequential importance sampling to iteratively filter proposed parameters k through a sequence of decreasing discrepancy tolerances $\epsilon ,$ comparing summary statistics $T ( \mathbf { y } _ { \mathrm { s i m } } )$ against $T ( \mathbf { y } _ { 0 } )$ . Moreover, by estimating the marginal likelihood from the importance sampling weights, SMC-ABC can approximate the posterior probability of model $M , p ( M \mid \mathbf { y } _ { 0 } )$ , enabling selection between competing CRN structures via Bayes factors [Toni et al., 2008, Didelot et al., 2011]. An overview of the mathematical formulation, along with the algorithm used for our SMC-ABC baseline is provided in Appendix A.

While theoretically sound, SMC-ABC still suffers three operational inefficiencies: (1) the curse of dimensionality requires handcrafted, low-dimensional summary statistics $T ( \cdot )$ that are difficult to design and will likely discard relevant temporal kinetics [Fearnhead and Prangle, 2012]; (2) achieving precise inference and model selection also depends heavily on the schedule of discrepancy tolerances used [Robert et al., 2011]; and (3) much of the computational cost is incurred repeatedly for every newly observed dataset.

## 2.3 Amortized Inference and Model Selection via Deep Learning

Addressing the scaling limitations of traditional ABC, we designed a deep learning architecture leveraging the BayesFlow library [Kühmichel et al., 2026]. By framing Bayesian inference and model selection as a joint problem of representation learning, neural density estimation, and classification, we transfer the computational burden of simulating the CRN into an initial training phase. We achieve this through a unified architecture that tackles these tasks using three jointly trained components. First, a summary network [Radev et al., 2022] extracts optimal statistics from the concentration time-series. Second, this learned representation conditions inference networks [Papamakarios and Murray, 2016] to estimate parameter posteriors. Third, the same representation simultaneously conditions a model network [Elsemüller et al., 2024| to compute posterior model probabilities. Together, these three components target the factorized joint posterior:

$$
p ( M , \mathbf { k } \mid \mathbf { y } _ { 0 } ) = p ( M \mid \mathbf { y } _ { 0 } ) \cdot p ( \mathbf { k } \mid \mathbf { y } _ { 0 } , M ) ,\tag{3}
$$

where $M \in \mathcal { M }$ represents a specific model (CRN pathway), k represents the kinetic parameters, and $\mathbf { y } _ { 0 }$ denotes the observed concentration data. Figure 1 shows a high-level overview of the architecture used.

## 2.3.1 The Summary Network

Circumventing the challenges involved in designing summary statistics, we define a summary network $T _ { \phi } : \mathcal { V }  \mathbb { R } ^ { d _ { e } }$ with learnable weights φ. Because $T _ { \phi }$ is trained jointly with the downstream classifier and density estimators, it is designed to preserve the mutual information between the concentration data and the target parameters, converging to a near-sufficient statistic [Radev et al., 2022, Chen et al., 2020].

![](images/d629d4d11dbb77c41555e26c8147fdcf68b94e7629415b6024f4e0a7de4746fe.jpg)  
Figure 1: High-level architecture of the joint neural inference pipeline. Each time-series observation (4 in total) goes through a parallel, shared-weight CNN to extract kinetic features. The architecture bifurcates to evaluate model evidence and condition M independent model-specific conditional Invertible Neural Networks to estimate parameter posteriors.

Motivated by permutation-invariant set encoding [Zaheer et al., 2017], the network treats the time-series from different experiments as an unordered set, ensuring the learned representation is independent of the order in which the results are provided. Each time-series is therefore passed independently through a shared one-dimensional Convolutional Neural Network (1D-CNN) to extract temporal dynamics [e.g., Ismail Fawaz et al., 2019, Radev et al., 2022], followed by Global Average Pooling (GAP) that enforces length-invariance. This produces one fixed-length feature vector for each experimental setup. The collection of vectors is passed as an unordered set of tokens into a Set Transformer [Lee et al., 2019] utilizing a Set Attention Block (SAB). Such mechanism allows the network to model interactions between different setups and assign contextual importance weights to the most informative conditions The updated set of tokens is mean-pooled and passed through a Multi-Layer Perceptron (MLP) that yields a permutation-invariant summary embedding of fixed dimension ${ \bf e } = T _ { \phi } ( { \bf y } _ { 0 } )$ . The detailed architecture is provided in Appendix B.1.

## 2.3.2 The Inference Network (Parameter Estimation)

Because multi-modal posteriors and non-linear dependencies are prevalent in CRNs, we train an independent conditional Invertible Neural Network (cINN) for each candidate model $M \in \mathcal { M }$ [Radev et al., 2022].

Each cINN $f _ { \psi _ { M } } ( \mathbf { k } ; \mathbf { e } )$ bijectively maps the reaction rate constants k to a latent variable $\mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ Applying the change of variables theorem, the cINN defines a conditional probability density:

$$
p _ { \psi _ { M } } ( \mathbf { k } \mid \mathbf { y } _ { 0 } , M ) = p _ { Z } \left( f _ { \psi _ { M } } ( \mathbf { k } ; \mathbf { e } ) \right) \left| \operatorname* { d e t } \mathbf { J } _ { f _ { \psi _ { M } } } \right| .\tag{4}
$$

Minimizing the expected forward Kullback-Leibler (KL) divergence between this approximate density and the true posterior $p ( \mathbf { k } \mid \mathbf { y } _ { 0 } , M )$ is equivalent to minimizing its expected negative log-density over the generative distribution [Radev et al., 2022]:

$$
\mathcal { L } _ { \mathcal { T } } ( \phi , \psi _ { M } ) = \mathbb { E } _ { p ( \mathbf { k } , \mathbf { y } _ { 0 } | M ) } \Big [ - \log p _ { \mathcal { Z } } \left( f _ { \psi _ { M } } ( \mathbf { k } ; T _ { \phi } ( \mathbf { y } _ { 0 } ) ) \right) - \log \Big | \mathrm { d e t } \mathbf { J } _ { f _ { \psi _ { M } } } \Big | \Big ] .\tag{5}
$$

## 2.3.3 The Model Network (Model Selection)

Parameter inference assumes the underlying CRN structure is known. To jointly resolve this, the shared summary embedding e is also fed into a model network $g _ { \omega } : \mathbb { R } ^ { d _ { e } }  \Delta ^ { | \mathcal { M } | }$ . Implemented as an MLP with a softmax as the final layer, it maps the embedding e to a categorical distribution consisting of the posterior model probabilities $p _ { \omega } ( M \mid \mathbf { y } _ { 0 } )$

The shared architecture approximates the joint posterior $p ( M , \mathbf { k } \mid \mathbf { y } _ { 0 } ) = p ( M \mid \mathbf { y } _ { 0 } ) p ( \mathbf { k } \mid \mathbf { y } _ { 0 } , M )$ Thus, minimizing the joint negative log-posterior reduces to minimizing the sum of the discrete and continuous losses:

$$
- \log p ( M , { \bf k } \mid { \bf y } _ { 0 } ) = - \log p ( M \mid { \bf y } _ { 0 } ) - \log p ( { \bf k } \mid { \bf y } _ { 0 } , M ) .\tag{6}
$$

Because the candidate models $M \in \mathcal { M }$ are discrete and mutually exclusive, the posterior $p ( M \mid \mathbf { y } _ { 0 } )$ corresponds to a categorical distribution. Consequently, the expected negative log-posterior in the first term expands into the standard categorical cross-entropy. Minimizing this objective—and assuming that the true data-generating process lies within the candidate set M—makes the model network's softmax outputs converge to the true Bayesian posterior probabilities [Richard and Lippmann, 1991]. Substituting the model network's approximation $p _ { \omega }$ and taking the expectation over the full generative process, the model loss becomes:

$$
\mathcal { L } _ { \mathcal { M } } ( \phi , \omega ) = \mathbb { E } _ { p ( M , { \mathbf { k } } , { \mathbf { y } } _ { 0 } ) } \Big [ - \sum _ { m \in \mathcal { M } } \mathbb { I } ( M = m ) \log p _ { \omega } ( M = m \mid T _ { \phi } ( \mathbf { y } _ { 0 } ) ) \Big ] ;\tag{7}
$$

with $\mathbb { I } ( \cdot )$ the indicator function. Hence, following the chain rule for KL divergence [Cover, 2006] and the factorization of the joint posterior, the global training objective we use to simultaneously address both tasks is the sum of the losses from Equations (5) and (7):

$$
\begin{array} { r } { \mathcal { L } ( \phi , \psi , \omega ) = \mathcal { L } _ { \mathcal { T } } ( \phi , \psi ) + \mathcal { L } _ { \mathcal { M } } ( \phi , \omega ) , } \end{array}\tag{8}
$$

where ${ \pmb { \psi } } = \{ \psi _ { M } \} _ { M \in \mathcal { M } }$ is the collection of all model-specific cINN weights. We minimize the Monte Carlo estimate of (8) across mini-batches of simulated triplets $( M , \mathbf { k } , \mathbf { y } _ { 0 } )$ , amortizing the computational cost of both tasks simultaneously. Once trained, this setup reduces joint inference to a forward pass, eliminating the need for further repetitive and expensive ODE simulations.

## 3 Simulation Study

To validate and assess our proposed neural posterior estimation pipeline, we conducted a simulation study comparing it against the classical SMC-ABC baseline using synthetically generated CRN datasets with known ground-truth reaction mechanisms and parameter sets.

## 3.1 Data and Problem Setting

We consider a model space M consisting of a finite set of candidate CRN topologies drawn from Burés and Larrosa [2023]. Our primary study focuses on three distinct reaction pathways, $\mathcal { M } = \{ M _ { 1 } , M _ { 2 } , M _ { 3 } \}$ assuming a uniform categorical prior $p ( M )$ . Each model $M \in \mathcal { M }$ is characterized by a parameter vector of reaction rate constants $\mathbf { k } _ { M } \in \mathbb { R } _ { + } ^ { d _ { M } }$ governing its elementary reactions. To reflect an uninformative setting where no prior kinetic information is available, individual rate constants are drawn independently from log-uniform priors spanning multiple orders of magnitude, specifically log $k _ { i } \sim \mathcal { U } ( \log { 1 0 ^ { - 5 } } , \log { 1 0 ^ { 5 } } )$ for all $k _ { i } \in \mathbf { k } _ { M }$

Following the training framework of Burés and Larrosa [2023], we consider $N _ { \mathrm { e x p } } = 4$ independent experimental setups per system. To perturb the system and expose different kinetic regimes, each setup is initialized with varying starting catalyst concentrations, uniformly sampled between 1 and 10 mol%. For each mechanism, our forward ODE simulator generated 936, 000 independent training trajectories and 1, 000 test trajectories. Generating these datasets required approximately 22 wall-clock hours per mechanism utilizing 120 CPU cores each. The ODE systems for the mechanisms and complete simulation details are provided in Appendix C.

The observable $\mathbf { y } _ { 0 }$ comprises time-series concentration measurements of a subset (Substrate and Product) of the involved chemical species. For a given experimental setup $i ,$ the continuous trajectory is sampled at $T = 2 0$ time points. To mirror realistic laboratory conditions, these observations are distorted with measurement noise. The data at each time step t is generated as $\mathbf { y } _ { 0 } ^ { ( i ) } ( t ) \sim \mathcal { N } ( \mathbf { X } _ { 0 } ^ { ( i ) } ( t ) , \Sigma )$ 2 where $\mathbf { X } _ { 0 } ^ { ( i ) } ( t )$ represents the subset of measurable species derived from the simulator at time $t ,$ and $\Sigma = \mathrm { d i a g } ( \sigma _ { S } ^ { 2 } , \sigma _ { P } ^ { 2 } )$ . The species-specific noise scale parameters, $\sigma _ { S }$ and $\sigma _ { P } .$ are sampled from a lognormal prior and inferred jointly with the kinetic parameters (from 6 to 8 total parameters, depending on the model).

Including training (performed using a GPU), the three-model architecture took roughly 25 hours to be ready for inference. Once trained, however, evaluating new experimental trajectories takes only a few seconds. Detailed hardware configurations are provided in Appendix B.4.

## 3.2 Baseline Comparison: SMC-ABC

To establish a baseline, we initially implemented the SMC-ABC algorithm to evaluate discrepancies directly on point-wise observed concentrations via the $L _ { 2 }$ distance across the observation vectors concatenated. We executed it using a population size of 3, 000 particles for each candidate model. We evaluated multiple threshold schedules, including adaptive thresholding [Lenormand et al., 2013], where the discrepancy € for a given generation is dynamically set to a specific quantile of the accepted discrepancies from the previous generation.

Evaluating discrepancies directly over the observation vectors proved inefficient. In classical ABC. working with high-dimensional data spaces is affected by the curse of dimensionality, as the probability of a simulated trajectory falling within an e-neighborhood of that observed diminishes exponentially [Sisson et al., 2018]. In our setting, this is compounded by the uninformative priors and the varying parametric complexities of the candidate mechanisms, causing high rejection rates and thus making the implementation very costly. In an attempt to accelerate inference and improve posterior convergence, we augmented the concentration trajectories with additional summary features, such as local trajectory gradients. Yet, incorporating these statistics offered no meaningful improvement in either acceptance rates or model selection accuracy. Further tuning of these features could theoretically improve the performance. However, such problem-specific calibration requires significant manual effort and is exactly what deep learning approaches aim to circumvent.

To demonstrate how these limitations manifest in practice, we consider a representative test case (where the true generative mechanism was $M _ { 2 } )$ . We implemented an adaptive threshold SMC-ABC algorithm across the candidate models, initialized with $\epsilon _ { 0 } = 8 . 5$ and an adaptive threshold quantile of $\alpha = 0 . 5 5$ . For this one case, the algorithm required evaluating 928, 605 ODE simulations to produce 3, 000 samples for each model across 6 generations. Furthermore, this baseline often failed to confidently or even correctly, recover the true model. In this case, the highest estimated posterior probability belonged to $M _ { 1 }$ (53.1%), as opposed to the true mechanism $M _ { 2 } ~ ( 4 6 . 8 \% )$ . ABC inherently penalizes parametric complexity when drawing from broad priors, and so it tends to favor simpler mechanisms that can loosely mimic the data. Visual inspection of the posterior predictive trajectories (App D.1, Figure 6) shows that both models can produce data within the vicinity of the observed curves Ultimately, this approach fails to distinguish between structurally distinct mechanisms that produce similar kinetic profiles. By contrast, our neural architecture is optimized to extract the higher-order temporal correlations and cross-species dependencies that can help make this distinction.

This high computational cost makes classical simulation-based approaches, such as SMC-ABC, impractical for chemical systems, which often require repeated inference across varying experimental conditions. Executing SMC-ABC for just a single case like this requires almost one-third of the total simulations used to train our amortized framework. As demonstrated in the following section, by shifting this simulation cost to a one-time training phase, our neural architecture not only enables fast inference at test time, but also achieves high model selection performance.

## 3.3 Amortized Neural Inference Evaluation

Having established the computational limitations of classical SMC-ABC for this kind of chemical system, we evaluate the performance of our proposed neural posterior approach. We assessed the pipeline on an independent test set of 3,000 concentration trajectories (split equally across the three mechanisms), evaluating its performance across three dimensions: (1) model selection accuracy; (2) the calibration of the estimated posterior probabilities, and (3) the quality of parameter inference within the selected reaction pathway

## 3.3.1 Model Selection Accuracy

To evaluate our approach's model selection capabilities, we measured how frequently it assigned the highest posterior probability to the true underlying mechanism. We assessed this accuracy under four distinct observational settings: (1) both the substrate and product are observed across all $T = 2 0$ time points; (2) both species are observed at $T = 1 0$ time points; (3) only the product is observed across all $T = 2 0$ time points; and (4) only the product is observed at $T = 1 0$ time points. Global accuracy metrics and per-model F1-scores for all four settings are summarized in Table 1.

Table 1: Model Selection Performance of Our Deep Learning Pipeline. We evaluate it across varying observational conditions: the impact of hidden chemical species (Product Only) and reduced temporal resolution (10 vs. 20 points). Global accuracy and model-specific F1-scores are reported for the 3, 000 independent test trajectories.
<table><tr><td>Observed Species</td><td>Time Points</td><td>Global Acc.</td><td> $M _ { 1 }$  F1</td><td> $M _ { 2 }$  F1</td><td> $M _ { 3 } \ { \bf F 1 }$ </td></tr><tr><td>Substrate &amp; Product</td><td>20</td><td>94%</td><td>0.94</td><td>0.98</td><td>0.96</td></tr><tr><td>Substrate &amp; Product</td><td>10</td><td>94%</td><td>0.93</td><td>0.97</td><td>0.93</td></tr><tr><td>Product Only</td><td>20</td><td>93%</td><td>0.91</td><td>0.97</td><td>0.92</td></tr><tr><td>Product Only</td><td>10</td><td>92%</td><td>0.89</td><td>0.95</td><td>0.90</td></tr></table>

Under the richest data scheme (substrate and product measured at 20 time points), our network achieved a global model selection accuracy of 94%. This already contrasts with the classical SMC-ABC baseline, which struggled to confidently identify the true mechanism under various hyperparameter specifications. When the number of measurements was reduced to 10 randomly sampled time points the network demonstrated robustness, maintaining the 94% global accuracy.

Further, we tested the pipeline when assuming the substrate was unobserved and that only measurements of the product were available. We trained the very same architecture using only product trajectories, and used it to infer the mechanism strictly from the product's formation trajectory. Remarkably, the network retained its discriminatory power, achieving 93% accuracy with 20 observation points and 92% with 10.

The confusion matrices across all four observation settings are shown in Appendix D.2, Table 4. Mechanism $M _ { 2 }$ proved to be the most distinguishable pathway, obtaining F1-scores between 0.96 and 0.98 across all observational settings. The network struggled more between $M _ { 1 }$ and $M _ { 3 } ,$ specifically when the substrate was hidden. For example, in the most constrained condition (10 product-only points), the network misclassified $M _ { 1 }$ as $M _ { 3 }$ in 71 instances (7.1% of the cases), and $M _ { 3 }$ as $M _ { 1 }$ in 99 instances (9.9% of the cases). This aligns with the underlying reaction kinetics: in $M _ { 3 } ,$ , by design the off-cycle resting state is only significant during initial conversion $( < 1 0 \% )$ , after which its product formation trajectory closely resembles that of $M _ { 1 }$ . Conversely, $M _ { 2 }$ features a continuous separation of the active catalyst into an off-cycle dimer during the main half of the reaction, causing a distinct deceleration on the product curve that the network can identify even when the substrate is unobserved Ultimately, these results confirm that the neural architecture successfully extracts the dynamics required to discriminate between reaction pathways, without critical reductions in performance when the underlying physical observables are decreased.

## 3.3.2 Posterior Calibration

In real-world chemical systems, multiple reaction pathways often occur simultaneously. While one primary mechanism usually dominates the process, alternative pathways frequently operate in the background, meaning experimental data is rarely the product of a single, perfectly isolated rule [see Chapter 2 of Steinfeld, 1998]. Hence, and even though the network is trained assuming that there is only one true underlying model, a robust inference framework must do more than simply predict the most likely mechanism; it must produce well-calibrated posterior probabilities that correctly quantify model selection uncertainty.

To assess the uncertainty estimates from our network, we computed the Expected Calibration Error (ECE) using 10 uniform bins [Guo et al., 2017] and Brier scores [Brier, 1950]. The network achieves low ECE values for all models: around 0.006. This means the network's estimated posterior probabilities deviate from the testing accuracy by less than $1 \%$ on average. Further, we produced reliability diagrams for all three mechanisms across the test set (under observational scheme (1) above). As shown in Figure 2, the estimated posterior probabilities closely track the identity line. This indicates that the network's uncertainty closely matches its accuracy—when the network expresses a specific level of certainty (posterior probability), the frequency with which the true mechanism is correctly identified is very close to its estimated probability. While there are minor deviations in the intermediate probability bins, with slight underconfidence about $M _ { 2 }$ and some overconfidence about $M _ { 3 }$ , these deviations occur only for a handful of bins and are generally not too large (less than 0.20 in all cases but one). In general, our results suggest that our pipeline produces well-calibrated posterior probabilities.

![](images/14c4cd43946e66a75f544a21d30d7e37f7b9f0dec31fb971b9d6720e2d2efb5f.jpg)  
Figure 2: Reliability diagram evaluating the calibration of the posterior probabilities across the 3,000 test trajectories. Predictions are grouped into 10 uniformly spaced probability bins. The dashed diagonal represents perfect calibration, where the estimated probability exactly matches empirical accuracy. All three mechanisms closely track the identity line, achieving low Expected Calibration Errors.

## 3.3.3 Parameter Inference

Having demonstrated the pipeline's ability to classify generative mechanisms, we next evaluate its accuracy in inferring reaction rate constants and generating parameter posteriors that reliably capture estimation uncertainty.

To measure this, we computed the empirical coverage for 95% credible intervals (CIs) from the posterior of each kinetic parameter within the test set. In a perfectly calibrated Bayesian setup, 95% of the ground-truth parameters should fall within the network's estimated CIs. Table 2 details the obtained coverage for the parameters of $M _ { 1 } , M _ { 2 }$ and $M _ { 3 }$ under the richest observational setting (setup (1)). The network achieves close to perfect coverage across all parameters.

Table 2: Empirical Coverage of the 95% Posterior Credible Intervals. Results for the kinetic parameters and observation noise (σ) under the 20-point (Substrate + Product) setting.
<table><tr><td>Parameter</td><td> $M _ { 1 }$ </td><td> $M _ { 2 }$ </td><td> $M _ { 3 }$ </td></tr><tr><td> $k _ { 1 }$ </td><td>96.2%</td><td>96.2%</td><td>96.2%</td></tr><tr><td> $k _ { - 1 }$ </td><td>96.4%</td><td>95.2%</td><td>96.0%</td></tr><tr><td> $k _ { 2 }$ </td><td>98.1%</td><td>97.3%</td><td>95.1%</td></tr><tr><td> $k _ { - 2 }$ </td><td>97.0%</td><td>94.8%</td><td>96.8%</td></tr><tr><td> $k _ { 3 }$ </td><td></td><td>93.0%</td><td>95.2%</td></tr><tr><td> $k _ { - 3 }$ </td><td></td><td>96.2%</td><td>96.0%</td></tr><tr><td> $\sigma _ { S }$ </td><td>95.1%</td><td>93.0%</td><td>95.8%</td></tr><tr><td> $\sigma _ { P }$ </td><td>95.1%</td><td>93.0%</td><td>94.3%</td></tr></table>

Moreover, the architecture maintains high-quality parameter uncertainty even when observables are restricted. Under the most constrained setting (product at 10 time points), the 95% CI coverage remains high across all parameters, ranging between 89.5% and 97.7%. Full coverage results for all observational settings are provided in $\mathrm { A }$ ppendix D.2, Table 6.

Additionally, we performed Simulation-Based Calibration (SBC) [Talts et al., 2020| to verify the probabilistic calibration of the marginal posteriors. The resulting SBC rank histograms (shown in App. D.2, Figures 9, 10 and 11) exhibit uniformity across the majority of the parameters. Further, it shows how the network handles weakly identifiable parameters. For example, the rate log $k _ { - 1 }$ in $M _ { 3 }$ cannot be fully recovered from the observables, with the posterior mean estimates often being very similar for different true values. Its posterior is more informative than the prior (the variance shrinkage was ≈ 0.26), but still spans most of the prior space. However, its corresponding SBC rank histogram remains close to uniform. This indicates that the network does not produce incorrect, overconfident posteriors, and instead outputs a suitably uncertain distribution. Minor deviations do exist in some cases—as for log $k _ { 3 }$ in $M _ { 2 }$ . But the overall uniformity indicates that the cINNs produce well-calibrated marginal posteriors on top of the underlying model selection.

Finally, we conducted posterior predictive checks (PPCs) to ensure these posteriors translate into sensible concentration profiles. To capture both model and parameter uncertainty, we first sample a mechanism using the estimated posterior probabilities, and then draw rate constants from its corresponding posteriors, using them to produce 100 simulations. As shown in the example in Figure 12 (App. D.2), the estimated parameters yield predictive distributions that closely track the ground-truth trajectories. This indicates that the network successfully captures the true underlying chemical kinetics. while accounting for parameter and model uncertainties given the data observed.

## 3.3.4 Scalability: Four Candidate Models

Unlike classical methods (e.g., SMC-ABC) where inference costs increase significantly with the number of candidate models, our amortized approach can scale more efficiently. We expanded the model space to include a fourth mechanism, M5 (App. C.2.4), simply by adding a classifier output and a corresponding cINN.

Despite keeping the network's parameter capacity fixed (e.g., the summary network dimensions), global classification accuracy across 4, 000 test trajectories remains high at 86% $\mathrm { ( E C E \le 0 . 0 2 2 9 ) }$ . We expect that scaling the hidden layers would easily offset this minor performance drop.

Parameter recovery also handles the expanded space robustly. Although M5 introduces weakly identifiable parameters $( \mathrm { e . g . }$ , log $k _ { - 2 }$ exhibits a low variance shrinkage of 0.080), the 95% credible intervals maintain acceptable coverage (95.3%–97.8%) across all its eight parameters (App. D.3). This indicates that the network accurately quantifies uncertainty when faced with uninformative physical signals.

Moreover, while adding a new mechanism requires one-time upfront simulation and training $( \approx 2 5 . 8 $ hours; Section 3), the inference time for new experimental trajectories remains amortized and is instantaneous. Thus, the architecture supports rapid, repetitive inference over complex model spaces without the computational inefficiencies of classical approaches.

## 4 Applied Example: Protodeboronation

As a proof of concept on a real-world example, we applied our framework to experimental kinetic data from a catalytic protodeboronation—an undesirable degradation process in Suzuki-Miyaura crosscoupling [Ser et al., 2025]. To identify the dominant mechanism driving this degradation, we considered three competing mechanisms for catalyst deactivation from Burés and Larrosa [2023]: a unimolecular deactivation of the catalyst-substrate complex $( M _ { 1 4 } )$ , an inhibitor-induced deactivation $( M _ { 1 5 } )$ , and a substrate-induced deactivation where a second substrate neutralizes the active intermediate $( M _ { 1 6 } )$ (see App. E).

We first trained our pipeline on synthetic simulations generated from these three candidate mechanisms. We then processed the experimental data from Ser et al. [2025], which consisted of product concentration profiles sampled at 8 time points across $N _ { \mathrm { e x p } } = 3$ independent setups with varying initial catalyst loadings (5%, 10%, and 15%). The network assigned a 92.19% probability to $M _ { 1 5 } ,$ considering it a stronger candidate than the alternative unimolecular $( M _ { 1 4 } : 3 . 7 1 \% )$ and substrate-induced $( M _ { 1 6 } : 4 . 1 0 \% )$ degradation hypotheses. This result provides actionable insights; further experimentation could focus on purifying the reaction environment to remove the deactivating agent.

The inference network produced wide posteriors for most parameters (shown alongside point estimates in App. E.4), indicating that measuring the product alone is insufficient to fully identify all individual reaction rate constants. However, as shown in Figure 3, synthetic trajectories drawn from these posteriors are consistent with the experimental observations. The network accurately maps covariant relationships between the parameters compatible with the true dynamics—as shown by the bivariate posterior distributions in App. E, Figure 18—while establishing their underlying uncertainty. By capturing these relationships, the pipeline reveals the underlying kinetic constraints of the system offering deeper mechanistic insights even when absolute rate constants remain non-identifiable.

## 5 Discussion

In this work, we introduced a joint neural posterior architecture for CRNs that combines reaction pathway selection and kinetic parameter estimation. We leverage a permutation-invariant summary network and couple it with cINNs and an MLP with a classification head to overcome the scaling and calibration limitations of classical likelihood-free methods like SMC-ABC. As demonstrated with synthetic cases and on real protodeboronation data, the pipeline extracts the necessary kinetic dynamics to output well-calibrated posterior model probabilities and robust parameter credible intervals, even when used on sparse observations

![](images/d5ef3d5c1088ca107f62a6273f952d2e105bcf1ef40f72105323294b8ccc3b07.jpg)

![](images/c5f23ebe361d2f90d4148ab988da321a04425a042e1f6db6b0ad16390607a0f1.jpg)

![](images/d2beb480f204013fdeaf25118996273599b9c1272a42ceabc2aafc0ab405bf94.jpg)  
Figure 3: Posterior Predictive Check (PPC) for experimental protodeboronation data across three varying initial catalyst loadings. The 300 predictive trajectories are sampled from the joint posterior, drawing from the candidate mechanisms proportionally to their estimated model probabilities.

The current framework was developed to consider three competing reaction mechanisms and was successfully scaled to a four-model space with minimal modifications and little performance loss. A natural next step is scaling the architecture to accommodate a larger number of competing structures. For example, Burés and Larrosa [2023] formalize 20 commonly encountered reaction mechanisms. Scaling the joint architecture to accommodate this magnitude of structures will likely require expanding the capacity of the summary network and potentially exploring hierarchical classification heads to deal with several overlapping structures, a direction we are actively pursuing.

Furthermore, our framework currently relies on a closed-world assumption: it presumes the true data-generating process lies within the predefined candidate set M. If the true mechanism is absent from the candidate space, or if none of the formalized models accurately explain the experimental data, the model network is forced to allocate its probability mass across the available approximations. In practice, this limitation is partially mitigated by chemical intuition, as candidate structures are typically curated by domain experts before inference begins. Nevertheless, moving toward open-world inference remains a critical challenge. Equipping the architecture with the capability to flag an experimental trajectory as an anomaly within the candidates considered is a critical direction for future research.

By shifting the heavy computational burden of forward ODE simulations to a pre-inference training phase, our approach amortizes the cost of joint inference and enables a fast feedback loop between experimentation and robust analysis. This could pave the way for integration into automated laboratories, enabling real-time mechanism discovery and experimental design when studying complex chemical systems.

## Code Availability

The source code for the joint neural inference architecture, implemented via BayesFlow, along with the scripts required to simulate the chemical trajectories and reproduce all experiments, are publicly available at https://github.com/Esquivel-Arturo/JNPE.

## Acknowledgements

J.A.E. is supported by a Doctoral Student Fellowship from the Data Sciences Institute at the University of Toronto. J.A.E. and V.L.-B. acknowledge the support of the Canadian Statistical Sciences Institute (CANSSI) and CANSSI Ontario. V.L.-B. also acknowledges the support of the Natural Sciences and

Engineering Research Council of Canada (NSERC). J.A.C.G.A. acknowledges funding of this project by the NSERC Alliance Grant #ALLRP587593-23 (Quantamole). J.A.C.G.A. also acknowledges support from the Council for Science, Technology and Innovation (CSTI), Cross-ministerial Strategic Innovation Promotion Program (SIP), "Promoting the application of advanced quantum technology platforms to social issues" (Funding agency: QST). J.A.C.G.A. also acknowledges support from the Australian Research Council Centre in Quantum Biotechnology through project number CE230100021. S.P.-G. acknowledges financial support from the Comunidad de Madrid through the “César Nombela' talent attraction program (Grant No. 2025-T1/COM36440)

## References

Lynton Ardizzone, Carsten Lüth, Jakob Kruse, Carsten Rother, and Ullrich Köthe. Conditional Invertible Neural Networks for Guided Image Generation. September 2019. URL https: //openreview.net/forum?id=SyxC9TEtPH.

Mark A Beaumont, Wenyang Zhang, and David J Balding. Approximate Bayesian Computation in Population Genetics. Genetics, 162(4):2025–2035, December 2002. ISSN 1943-2631. doi: 10.1093/ genetics/162.4.2025. URL https://doi.org/10.1093/genetics/162.4.2025.

Samuel M. Blau, Hetal D. Patel, Evan Walter Clark Spotte-Smith, Xiaowei Xie, Shyam Dwaraknath, and Kristin A. Persson. A chemically consistent graph architecture for massive reaction networks applied to solid-electrolyte interphase formation. Chemical Science, 12(13):4931–4939, April 2021. ISSN 2041- 6539. doi: 10.1039/D0SC05647B. URL https://pubs.rsc.org/en/content/articlelanding/ 2021/sc/d0sc05647b.

Glenn W. Brier. VERIFICATION OF FORECASTS EXPRESSED IN TERMS OF PROBABILITY. January 1950. ISSN 1520-0493. URL https://journals-ametsoc-org.us1.proxy.openathens. net/view/journals/mwre/78/1/1520-0493\_1950\_078\_0001\_vofeit\_2\_0\_co\_2.xml.

Jordi Burés and Igor Larrosa. Organic reaction mechanism classification using machine learning. Nature, 613(7945):689–695, January 2023. ISSN 1476-4687. doi: 10.1038/s41586-022-05639-4. URL https://www.nature.com/articles/s41586-022-05639-4.

Yanzhi Chen, Dinghuai Zhang, Michael U. Gutmann, Aaron Courville, and Zhanxing Zhu. Neural Approximate Sufficient Statistics for Implicit Models. October 2020. URL https://openreview. net/forum?id=SRDuJssQud.

T. M. Cover. Elements of information theory. Wiley, 2nd ed. edition, 2006. ISBN 9786610517497.

Xavier Didelot, Richard G. Everitt, Adam M. Johansen, and Daniel J. Lawson. Likelihood-free estimation of model evidence. Bayesian Analysis, 6(1), March 2011. ISSN 1936-0975. doi: 10. 1214/11-BA602. URL https://projecteuclid.org/journals/bayesian-analysis/volume-6/ issue-1/Likelihood-free-estimation-of-model-evidence/10.1214/11-BA602.full.

Yachao Dong, Ting Yang, Yafeng Xing, Jian Du, and Qingwei Meng. Data-Driven Modeling Methods and Techniques for Pharmaceutical Processes. Processes, 11(7):2096, July 2023. ISSN 2227-9717. doi: 10.3390/pr11072096. URL https://www.mdpi.com/2227-9717/11/7/2096. Number: 7.

Lasse Elsemüller, Martin Schnuerch, Paul-Christian Bürkner, and Stefan T. Radev. A deep learning method for comparing Bayesian hierarchical models. Psychological Methods, 2024. ISSN 1939-1463. doi:10.1037/met0000645.

Paul Fearnhead and Dennis Prangle. Constructing summary statistics for approximate Bayesian computation: semi-automatic approximate Bayesian computation [with Discussion]. Journal of the Royal Statistical Society. Series B (Statistical Methodology), 74(3):419–474, 2012. ISSN 1369-7412. URL https://www.jstor.org/stable/41674639.

Paul Fearnhead, Vasilieos Giagos, and Chris Sherlock. Inference for Reaction Networks Using the Linear Noise Approximation. Biometrics, 70(2):457–466, June 2014. ISSN 0006-341X. doi: 10.1111/ biom.12152. URL https://doi.org/10.1111/biom.12152.

Daniel T. Gillespie. Stochastic Simulation of Chemical Kinetics. Annual Review of Physical Chemistry, 58(Volume 58, 2007):35–55, May 2007. ISSN 0066-426X, 1545-1593. doi: 10.1146/annurev.physchem. 58.032806.104637. URL https://www.annualreviews.org/content/journals/10.1146/annurev. physchem.58.032806.104637.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On Calibration of Modern Neural Networks. In Proceedings of the 34th International Conference on Machine Learning, pages 1321–1330. PMLR, July 2017. URL https://proceedings.mlr.press/v70/guo17a.html.

C. L. Heald and J. H. Kroll. The fuel of atmospheric chemistry: Toward a complete description of reactive organic carbon. Science Advances, 6(6):eaay8967, February 2020. doi: 10.1126/sciadv.aay8967. URL https://www.science.org/doi/10.1126/sciadv.aay8967.

Hassan Ismail Fawaz, Germain Forestier, Jonathan Weber, Lhassane Idoumghar, and Pierre-Alain Muller. Deep learning for time series classification: a review. Data Mining and Knowledge Discovery, 33(4):917–963, July 2019. ISSN 1573-756X. doi: 10.1007/s10618-019-00619-1. URL https://doi. org/10.1007/s10618-019-00619-1.

Petar Jovanovski, Andrew Golightly, Umberto Picchini, and Massimiliano Tamborrino. Simulationbased inference using splitting schemes for partially observed diffusions in chemical reaction networks, August 2025. URL http://arxiv.org/abs/2508.11438. arXiv:2508.11438 [stat.ME].

Lars Kühmichel, Jerry M. Huang, Valentin Pratz, Jonas Arruda, Hans Olischläger, Daniel Habermann, Simon Kucharsky, Lasse Elsemüller, Aayush Mishra, Niels Bracher, Svenja Jedhoff, Marvin Schmitt, Paul-Christian Bürkner, and Stefan T. Radev. BayesFlow 2: Multi-Backend Amortized Bayesian Inference in Python, March 2026. URL http://arxiv.org/abs/2602.07098. arXiv:2602.07098 [stat.CO].

Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. In Proceedings of the 36th International Conference on Machine Learning, pages 3744–3753. PMLR, May 2019. URL https://proceedings.mlr.press/v97/1ee19d.html.

Maxime Lenormand, Franck Jabot, and Guillaume Deffuant. Adaptive approximate Bayesian computation for complex models. Computational Statistics, 28(6):2777–2796, December 2013. ISSN 1613-9658. doi: 10.1007/s00180-013-0428-3. URL https://doi.org/10.1007/s00180-013-0428-3.

George Papamakarios and Iain Murray. Fast epsilon -free Inference of Simulation Models with Bayesian Conditional Density Estimation. In Advances in Neural Information Processing Systems volume 29. Curran Associates, Inc., 2016. URL https://proceedings.neurips.cc/paper\_files/ paper/2016/hash/6aca97005c68f1206823815f66102863-Abstract.html.

Stefan T. Radev, Ulf K. Mertens, Andreas Voss, Lynton Ardizzone, and Ullrich Kothe. BayesFlow: Learning Complex Stochastic Models With Invertible Neural Networks. IEEE transaction on neural networks and learning systems, 33(4):1452–1466, 2022. ISSN 2162-237X. doi: 10.1109/TNNLS.2020. 3042395. URL https://ieeexplore.ieee.org/document/9298920. Place: United States.

A. Raue, C. Kreutz, T. Maiwald, J. Bachmann, M. Schilling, U. Klingmüller, and J. Timmer. Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. Bioinformatics, 25(15):1923–1929, August 2009. ISSN 1367-4803. doi: 10.1093/bioinformatics/btp358. URL https://doi.org/10.1093/bioinformatics/btp358.

Michael D. Richard and Richard P. Lippmann. Neural Network Classifiers Estimate Bayesian a

posteriori Probabilities. Neural Computation, 3(4):461–483, December 1991. ISSN 0899-7667. doi: 10.1162/neco.1991.3.4.461. URL https://doi.org/10.1162/neco.1991.3.4.461.

Christian P. Robert, Jean-Marie Cornuet, Jean-Michel Marin, and Natesh S. Pillai. Lack of confidence in approximate Bayesian computation model choice. Proceedings of the National Academy of Sciences, 108(37):15112–15117, September 2011. ISSN 0027-8424, 1091-6490. doi: 10.1073/pnas.1102900108. URL https://pnas.org/doi/ful1/10.1073/pnas.1102900108.

Cher Tian Ser, Han Hao, Sergio Pablo-García, Kjell Jorner, Shangyu Li, Robert Pollice, and Alán Aspuru-Guzik. Bulky phosphine ligands promote palladium-catalyzed protodeboronation. Journal of the American Chemical Society, 147(47):43884–43901, 2025. URL https://pubs.acs.org/jacsat/ article/147/47/43884/3739145.

S. A. Sisson, Y. Fan, and Mark M. Tanaka. Sequential Monte Carlo without likelihoods. Proceedings of the National Academy of Sciences, 104(6):1760–1765, February 2007. doi: 10.1073/pnas.0607208104. URL https://www.pnas.org/doi/10.1073/pnas.0607208104

Scott A. Sisson, Yanan Fan, and Mark Beaumont, editors. Handbook of Approximate Bayesian Computation. Chapman and Hall/CRC, New York, September 2018. ISBN 978-1-315-11719-5. doi: 10.1201/9781315117195.

Jeffrey I. Steinfeld. Chemical kinetics and dynamics. Prentice Hall, Upper Saddle River, N.J, 2nd ed. edition, 1998. ISBN 978-0-13-737123-5.

Sean Talts, Michael Betancourt, Daniel Simpson, Aki Vehtari, and Andrew Gelman. Validating Bayesian Inference Algorithms with Simulation-Based Calibration, October 2020. URL http: //arxiv.org/abs/1804.06788. arXiv:1804.06788[stat.ME].

Tina Toni, David Welch, Natalja Strelkowa, Andreas Ipsen, and Michael P.H Stumpf. Approximate Bayesian computation scheme for parameter inference and model selection in dynamical systems. Journal of The Royal Society Interface, 6(31):187–202, July 2008. doi: 10.1098/rsif.2008.0172. URL https://royalsocietypublishing.org/doi/abs/10.1098/rsif.2008.0172.

David J. Warne, Ruth E. Baker, and Matthew J. Simpson. Simulation and inference algorithms for stochastic biochemical reaction networks: from basic concepts to state-of-the-art. Journal of The Royal Society Interface, 16(151):20180943, February 2019. doi: 10.1098/rsif.2018.0943. URL https://royalsocietypublishing.org/doi/10.1098/rsif.2018.0943.

Yan, Bing. SPARK: Mechanism inference for outer-sphere electron transfer, 2026. URL https: //huggingface.co/spaces/bing-yan/spark.

Manzil Zaheer, Satwik Kottur, Siamak Ravanbhakhsh, Barnabás Póczos, Ruslan Salakhutdinov, and Alexander J Smola. Deep Sets. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS'17, pages 3394–3404, Red Hook, NY, USA, December 2017. Curran Associates Inc. ISBN 978-1-5108-6096-4. URL https://d1.acm.org/doi/10.5555/3294996. 3295098.

# Joint Neural Posterior Estimation and Model Selection for Chemical Reaction Networks: Supplementary Materials

## A Sequential Monte Carlo ABC Formulation

A comprehensive theoretical treatment of Approximate Bayesian Computation (ABC) is beyond the scope of this work (we refer the reader to standard literature such as Toni et al. [2008], Beaumont et al. [2002] and Sisson et al. [2007]). Here we briefly outline the SMC-ABC algorithm utilized as our classical baseline to provide intuition for its computational mechanics.

Instead of evaluating the likelihood directly, ABC simulates synthetic data $\mathbf { y } _ { \mathbf { k } }$ using candidate parameters and compares it to the observed data $\mathbf { y } _ { 0 }$ . A candidate parameter set is accepted if the distance between their respective summary statistics falls below a specified tolerance threshold: $\rho ( T ( \mathbf { y } _ { \mathbf { k } } ) , T ( \mathbf { y } _ { 0 } ) ) \le \epsilon$ , where $T ( \cdot )$ is a summary statistic function and $\rho ( \cdot , \cdot )$ is a distance metric (e.g., Euclidean distance).

To avoid the inefficient rejection rates of naive ABC rejection sampling, SMC-ABC makes use of sequential importance sampling. A population of $N$ particles is propagated through a sequence of intermediate distributions defined by decreasing tolerances $\epsilon _ { 1 } > \cdots > \epsilon _ { R }$ At generation $r ,$ a candidate particle $\mathbf { k } ^ { * * }$ is generated by sampling an ancestor $\mathbf { k } ^ { * }$ from generation $r - 1$ with probability proportional to its importance weight $\dot { w } ^ { ( r - 1 ) }$ , and perturbing it via a Gaussian transition kernel $q ( \cdot \mid \mathbf { k } ^ { * } )$

If the simulated data satisfies the current tolerance $\rho ( T ( \mathbf { y _ { k ^ { * * } } } ) , T ( \mathbf { y } _ { 0 } ) ) \le \epsilon _ { r }$ , the particle is accepted. To account for the proposal distribution, the unnormalized importance weight for the accepted particle i is defined as:

$$
w _ { i } ^ { ( r ) } = \frac { p ( \mathbf { k } _ { i } ^ { ( r ) } ) } { \sum _ { j = 1 } ^ { N } w _ { j } ^ { ( r - 1 ) } q \left( \mathbf { k } _ { i } ^ { ( r ) } \mid \mathbf { k } _ { j } ^ { ( r - 1 ) } \right) } .\tag{9}
$$

Under this framework, estimating the marginal likelihood (evidence) for a specific mechanism M relies on a Monte Carlo approximation over all $N _ { \mathrm { t o t a l } }$ proposed particles. ABC approximates the likelihood through $\mathbb { I } ( \rho ( T ( \mathbf { y } _ { \mathbf { k } } ) , T ( \mathbf { y } _ { 0 } ) ) \le \epsilon _ { R } )$ , which evaluates to 1 if the simulated data falls within the tolerance and 0 otherwise.

By substituting this indicator function into the standard importance sampling estimator, the sum over all proposals reduces to only the $N _ { \mathrm { a c c } }$ accepted particles. Thus, the evidence estimate reduces to:

$$
p ( \mathbf { y } _ { 0 } \mid M ) \approx \frac { 1 } { N _ { \mathrm { t o t a l } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t o t a l } } } \frac { \mathbb { I } ( \rho _ { i } \le \epsilon _ { R } ) p ( \mathbf { k } _ { i } \mid M ) } { \mu _ { R } ( \mathbf { k } _ { i } \mid M ) } \propto \frac { 1 } { N _ { \mathrm { t o t a l } } } \sum _ { i = 1 } ^ { N _ { \mathrm { a c c } } } w _ { R } ^ { ( i ) } ,\tag{10}
$$

where $\begin{array} { r } { \mu _ { R } ( \mathbf { k } \mid M ) = \sum _ { j } \bar { w } _ { j } ^ { ( R - 1 ) } q ( \mathbf { k } \mid \mathbf { k } _ { j } ^ { ( R - 1 ) } ) } \end{array}$ is the proposal density, and $w _ { R } ^ { ( i ) }$ the unnormalized weights at the final generation $R .$

Therefore, for a finite set of competing mechanisms $\mathcal { M }$ selecting between hypotheses $M _ { i }$ and $M _ { j }$ via the posterior odds ratio involves:

$$
{ \frac { p ( M _ { i } \mid \mathbf { y } _ { 0 } ) } { p ( M _ { j } \mid \mathbf { y } _ { 0 } ) } } = { \frac { p ( M _ { i } ) } { p ( M _ { j } ) } } { \underbrace { { \frac { p ( \mathbf { y } _ { 0 } \mid M _ { i } ) } { p ( \mathbf { y } _ { 0 } \mid M _ { j } ) } } } _ { K _ { i j } } } \approx { \frac { p ( M _ { i } ) } { p ( M _ { j } ) } } { \sum } w _ { R } ^ { ( i , M _ { i } ) } / N _ { \mathrm { t o t a l } } ^ { ( M _ { i } ) } .\tag{11}
$$

Note that this corresponds exactly to the Bayes Factor $K _ { i j }$ when the model prior is uniform. Since the marginal likelihood estimate in Equation 10 requires tracking the total number of simulations $N _ { \mathrm { t o t a l } }$ , the SMC-ABC pipeline is executed independently for every candidate model $M \in { \mathcal { M } } .$ As the tolerance $\epsilon _ { R }  0 , N _ { \mathrm { t o t a l } }$ grows exponentially, making classical Bayes Factor estimation computationally prohibitive for large hypothesis spaces. The procedure used to execute this independent model evaluation (inspired by implementations from Warne et al. [2019] and Didelot et al. [2011]) is detailed in Algorithm 1.

Algorithm 1: SMC-ABC with Sequential Importance Resampling   
Data: Observed data $\mathbf { y } _ { 0 } ,$ Prior $p ( \mathbf { k } )$ , Simulator $\overline { { \mathcal { S } ( \mathbf { k } , \mathcal { R } ) } }$ , Metric $\rho ( \cdot , \cdot )$ , Particles N, Tolerances   
$\epsilon _ { 1 } > \cdots > \epsilon _ { R } .$   
Result: Posterior samples $\{ \mathbf { k } _ { i } ^ { ( R ) } \} _ { i = 1 } ^ { N }$ approximating $p _ { \epsilon _ { R } } ( \mathbf { k } \mid \mathbf { y } _ { 0 } )$   
$/ /$ Initialization (Generation $r = 1 )$   
for $i \gets 1$ to $N$ do   
repeat   
Sample $\mathbf { k } ^ { * * } \sim p ( \mathbf { k } ) ;$   
Simulate $\mathbf { y } _ { \mathrm { s i m } } \sim \mathcal { S } ( \mathbf { k } ^ { * * } , \mathcal { R } )$   
until $\begin{array} { r } { \rho ( \mathbf { y } _ { \mathbf { k } ^ { * * } } , \mathbf { y } _ { 0 } ) \le \epsilon _ { 1 } ; } \end{array}$   
Set $\mathbf { k } _ { i } ^ { ( 1 ) } \gets \mathbf { k } ^ { * * } ;$   
Set $w _ { i } ^ { ( 1 ) }  1 / N ;$   
end   
$/ /$ Sequential Importance Resampling   
for $r \gets 2$ to R do   
for $i \gets 1$ to $N$ do   
repeat   
Sample ancestor $j$ from $\{ 1 , \ldots , N \}$ with prob $w _ { j } ^ { ( r - 1 ) }$   
Sample proposed parameter $\mathbf { k } ^ { * * } \sim q ( \mathbf { k } \mid \mathbf { k } _ { j } ^ { ( r - 1 ) } )$   
Simulate $\mathbf { y } _ { \mathbf { k } ^ { * * } } \sim \mathcal { S } ( \mathbf { k } ^ { * * } , \mathcal { R } )$ 2   
until $\rho ( \mathbf { y _ { k } } \ast \ast , \mathbf { y } _ { 0 } ) \leq \epsilon _ { r } ;$   
Set $\mathbf { k } _ { i } ^ { ( r ) * }  \mathbf { k } ^ { * * } ;$   
$\begin{array} { r } { w _ { i } ^ { ( r ) * }  \frac { p ( \mathbf { k } _ { i } ^ { ( r ) * } ) } { \sum _ { j = 1 } ^ { N } w _ { j } ^ { ( r - 1 ) } q ( \mathbf { k } _ { i } ^ { ( r ) * } | \mathbf { k } _ { j } ^ { ( r - 1 ) } ) } ; } \end{array}$   
end   
$/ /$ Normalize and Resample   
W $ \textstyle \sum _ { i = 1 } ^ { N } w _ { i } ^ { ( r ) * } ;$ ,   
for $i \gets 1$ to N do   
$w _ { i } ^ { ( r ) * }  w _ { i } ^ { ( r ) * } / W ;$   
end   
for $i \gets 1$ to N do   
Sample index m with prob $w _ { m } ^ { \left( r \right) * } ;$   
Set $\mathbf { k } _ { i } ^ { ( r ) } \gets \mathbf { k } _ { m } ^ { ( r ) * } ;$   
Set $w _ { i } ^ { ( r ) } \gets 1 / N ;$   
end   
end

## B Neural Architecture and Training Details

## B.1 Deep Learning Architecture

This section details the neural network architectures and training hyperparameters used for the simulation studies and applied example. The framework was implemented using TensorFlow/Keras and the BayesFlow library.

The architecture consists of three core components: (1) a Permutation-Invariant CNN-Transformer summary network that compresses the multivariate concentration trajectories y into a fixed-length embedding; (2) a model selection network utilizing a dense body; and (3) a group of conditional Invertible Neural Networks (cINNs) for model-specific parameter inference. A detailed visual diagram of the summary network is shown in Figure 1. The configurations and training hyperparameters for the entire architecture are summarized in Table 3.

The summary network employs a permutation-invariant structure. First, the 1D-CNN and the initial dense projection layer are applied independently to each of the N experimental observations (timeseries) using shared weights. The data is then reshaped and passed into the Self-Attention block, which allows the observation embeddings to contextualize one another. Finally, a global average pooling layer across the observations combines the interacting sequences into the representation that is passed to the shared dense body.

The output of this shared body is then fed into a dedicated classification head, which outputs logits via a linear activation. During inference, a Softmax function is applied to these logits to yield the final posterior model probabilities $p ( M | \mathbf { y } _ { 0 } )$

In parallel with the model selection task, the shared summary embedding e conditions parameter inference. We instantiate a dedicated cINN for each model. Each cINN is constructed as a normalizing flow comprising a sequence of 8 affine coupling layers [Ardizzone et al., 2019]. Each coupling block is parameterized by internal MLPs configured with two hidden layers of 256 units (BayesFlow's default). Each independent cINN learns to map a standard Gaussian to the joint posterior distribution $p ( \mathbf { k } \mid \mathbf { y } _ { 0 } , M )$ for the specific kinetic parameters of that mechanism.

![](images/574c7c3f681a36c5c86e12ccad650a3e41be4347163460c06aad298721b69d36.jpg)  
Figure 4: Detailed architecture of the summary network $T _ { \phi } .$ Independent experimental observations are processed through a shared 1D-CNN (indicated by dashed connections) to extract temporal signatures. The sequences are globally average pooled across time to ensure length invariance. The independent features are contextualized using a Set Attention Block (SAB), before being globally pooled across experiments and passed through dense layers to produce the unified embedding e.

## B.2 Input Representation and Data Handling

To process the multi-experiment trajectories, the input to the network is structured as a tensor of shape $( B , N _ { \mathrm { e x p } } , T , C )$ , where B is the batch size, $N _ { \mathrm { e x p } } = 4$ is the number of experimental initial conditions, and $T = 2 1$ represents the maximum number of temporal observations (including initial conditions). The feature dimension C = 5 consists of: the observation time t, the measured substrate concentration [S], the measured product concentration [P], the initial catalyst loading for that specific experiment [cat]o, and a binary indicator flagging the initial condition t = 0.

Table 3: Architecture and Training Hyperparameters for our Neural Inference Framework.
<table><tr><td>Component / Hyperparameter</td><td>Specification</td></tr><tr><td>Summary Network: CNN-Transformer 1D-CNN filters (3 layers, kernel size 3)</td><td> $6 4  1 2 8  2 5 6$  Global Average Pooling (across time steps, masked for dropout points)</td></tr><tr><td>Pooling Layer Dense layer  $( d _ { m o d e l } )$ </td><td></td></tr><tr><td>Self-Attention heads</td><td>256 4</td></tr><tr><td>Feed Forward Network</td><td> $5 1 2  2 5 6$ </td></tr><tr><td></td><td></td></tr><tr><td>Normalization</td><td>Add &amp; Norm (LayerNorm)</td></tr><tr><td>Pooling layer</td><td>Global Average Pooling (across experimental observations)</td></tr><tr><td>Dense layers</td><td> $5 1 2  2 5 6$ </td></tr><tr><td>Final embedding</td><td>256</td></tr><tr><td>Activation functions</td><td>ReLU</td></tr><tr><td>Dropout rate</td><td>0.1</td></tr><tr><td>Model Network: MLP</td><td></td></tr><tr><td>Dense layer (classification head)</td><td>256 → 3 (base) / 256 → 4 (four models)</td></tr><tr><td>Activation function</td><td>Linear</td></tr><tr><td>Inference Networks: cINNs</td><td></td></tr><tr><td>Coupling design</td><td>Affine</td></tr><tr><td>Number of coupling blocks</td><td>8</td></tr><tr><td>MLP hidden layers per block</td><td>2</td></tr><tr><td>MLP hidden units per layer</td><td>256</td></tr><tr><td>Inference parameters  $\left( M _ { 1 } , M _ { 2 } , M _ { 3 } , M _ { 5 } \right)$ </td><td>6,8,8,8</td></tr><tr><td>Training Setup</td><td></td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 } ~ \mathrm { ( C o n s t a n t ) }$ </td></tr><tr><td>Gradient clipping (norm)</td><td>1.0</td></tr><tr><td>Batch size (per mechanism)</td><td>180</td></tr><tr><td>Total samples per mechanism</td><td>936,000</td></tr><tr><td>Total training epochs</td><td>55</td></tr></table>

To ensure the summary network learns robust length-invariant representations that can generalize to varying experimental measurement rates (e.g., T = 20 versus $T = 1 0 )$ , we apply temporal dropout during training. For each forward pass, we uniformly sample between 5 and T valid time points per experimental setup. The discarded time points are masked by setting their time channel to -1.0. All target parameters are standardized prior to density estimation.

## B.3 Data Flow

The input tensor is processed through a sequence of transformations designed to extract temporal and contextual features:

• Temporal Modeling (1D-CNN): The $N _ { \mathrm { e x p } }$ observation sequences are flattened into a batch of independent trajectories and passed through the shared 1D-CNN. The sequence length is convolved into 128 and then 256 feature channels.

• Masked Temporal Pooling: For each independent observation, the CNN features are averaged across the valid time steps—ignoring any points removed by temporal dropout—to collapse the sequence into a single vector. A shared dense layer then projects these independent vectors into a $( B , N _ { \mathrm { e x p } } , d _ { \mathrm { m o d e l } } )$ tensor.

• Set Interaction (SAB): To capture cross-dependencies between the varying initial conditions, the tensor is processed by a Multi-Head Self-Attention block (4 heads, $d _ { \mathrm { m o d e l } } = 2 5 6 )$ , followed by Add & Norm and Position-wise Feed-Forward layers

• Summary Embedding: The context-aware set of $N _ { \mathrm { e x p } }$ vectors is mean-pooled across the experimental dimension. A final dense body maps this pooled vector to the fixed summary embedding $\mathbf { e } \in \mathbb { R } ^ { 2 5 6 }$

• Inference and Model Networks: The embedding e is passed through a linear classification head to output the categorical model logits, while simultaneously acting as the conditional input for the independent cINNs.

The architecture is trained end-to-end by minimizing a joint loss function. At each step, the total loss is computed as the sum of the categorical cross-entropy from the Model Network $\left( - \log p ( M \mid \mathbf { y } _ { 0 } ) \right)$ and the negative log-posterior from the corresponding cINN $\left( - \log p ( \mathbf { k } \mid \mathbf { y } _ { 0 } , M ) \right)$ . The network weights are updated using the Adam optimizer with a constant learning rate of $1 \times 1 0 ^ { - 4 }$ and gradient clipping scaled to a maximum norm of 1.0 to prevent overflow during the early training stages. The model is trained over 55 epochs using batch sizes of 180 per mechanism. All other details and hyperparameter specifications are summarized in Table 3.

## B.4 Software Infrastructure and Computational Resources

To generate the training datasets efficiently, simulations were separated from the neural network training. The chemical systems were numerically integrated using Julia's DifferentialEquations.jl suite, accessed via the diffeqpy wrapper in Python. This enabled rapid, multi-core dataset generation performed on an academic high-performance computing cluster.

Our neural posterior inference pipeline was implemented in Python utilizing TensorFlow and the BayesFlow library [Kühmichel et al., 2026]. Generating the 936, 000 synthetic trajectories for each mechanism required approximately 22 wall-clock hours distributed across 120 CPU cores (executed as four parallel 30-core jobs, each provisioned with 240GB of RAM). Training of the full architecture was executed in roughly 3 hours per training configuration utilizing 4 CPU cores, 16GB of RAM, and a 10GB partition of an NVIDIA H100 GPU.

## C Simulation Details and Reaction Mechanisms

This appendix provides the mathematical formulations for the candidate mechanisms evaluated in this study, alongside the parameters and process utilized for data simulation.

## C.1 Simulation Configuration

All synthetic datasets were generated by numerically integrating the deterministic ODEs corresponding to each candidate CRN. All mechanism as well as the experimental settings to simulate from were taken from Burés and Larrosa [2023]. For every mechanism $M \in \mathcal { M }$ , the kinetic constants were drawn randomly from independent log-uniform priors, i.e. log $k _ { i } \sim \mathcal { U } ( \log { 1 0 ^ { - 5 } } , \log { 1 0 ^ { 5 } } )$ . Each kinetic sample is used to produce $N _ { \mathrm { e x p } } = 4$ concentration profiles. To induce varying kinetic regimes, the initial setups are configured as follows: The first three profiles are initialized with a standard substrate concentration of $[ S ] _ { 0 } = 1$ and product concentration of $[ P ] _ { 0 } = 0$ . The catalyst loadings $[ c a t ] _ { 0 }$ for these first three profiles are uniformly sampled at random between 1 and 10 mol%. The fourth profile represents a "same-excess" experiment: the substrate $[ S ] _ { 0 }$ is initialized at a random value between 0.4 and 0.8, with the remaining mass balance added as initial product such that $[ P ] _ { 0 } = 1 - [ S ] _ { 0 }$ . The initial catalyst loading $[ c a t ] _ { 0 }$ for this fourth profile is randomly selected from the first three profiles. All other initial concentrations are set to zero. From each continuous kinetic profile, observations are evaluated at $T = 2 0$ discrete concentration-time points randomly selected across the trajectory. These points are distorted using Gaussian noise model. The independent observation noise scales for the substrate $( \sigma _ { S } )$ and product $\left( \sigma _ { P } \right)$ are sampled from a log-normal prior: log $\sigma \sim \mathcal { N } ( - 6 , 1 )$ . To ensure stable network training and allow the architecture to generalize across varying kinetic regimes, both the temporal grids and the resulting species concentrations for each trajectory were min-max normalized to the range [0, 1] prior to passing them to the summary network

Below we show the CRN for each mechanism and the corresponding systems of ODEs. Each node of the CRN corresponds to a chemical species. Arrows indicate which reactants (origin) lead to which products (end) for an intermediate reaction. The reaction rate constant dictating the propensity of each reaction is shown next to the corresponding arrow. Note that to ensure the generated trajectories accurately reflect their target structural hypotheses, the sampled rate constants for certain mechanisms must satisfy specific yield constraints. Solid arrows correspond to forward reactions, while dashed arrows to the reverse. In all equations, [X] denotes the concentration of species X at time t.

## C.2 Reaction Mechanisms Considered

## C.2.1 Mechanism 1

The catalyst reversibly binds to the substrate to form an active complex, which then converts to the product.

![](images/ac51579e0ca9933ce1e8f5a32c6015160d9d626b89397aa90d866ce6925adbd5.jpg)

$$
\frac { d [ S ] } { d t } = k _ { - 1 } [ c a t { S } ] - k _ { 1 } [ S ] [ c a t ]
$$

$$
\frac { d [ P ] } { d t } = k _ { 2 } [ c a t { S } ] - k _ { - 2 } [ c a t ] [ P ]
$$

$$
\frac { d [ c a t ] } { d t } = ( k _ { - 1 } + k _ { 2 } ) [ c a t S ] - ( k _ { 1 } [ S ] + k _ { - 2 } [ P ] ) [ c a t ]
$$

$$
\frac { d [ c a t S ] } { d t } = ( k _ { 1 } [ S ] + k _ { - 2 } [ P ] ) [ c a t ] - ( k _ { - 1 } + k _ { 2 } ) [ c a t S ]
$$

## C.2.2 Mechanism 2

Mechanism 2 incorporates an off-cycle equilibrium where the active catalyst can reversibly form an inactive dimer $( c a t _ { 2 } )$ . It is defined by any set of kinetic constants such that a reaction run with $[ S ] _ { 0 } = 1$ and $[ c a t ] _ { 0 } = 5$ mol% produces a significant amount $c a t _ { 2 } \ ( > 1 0 \%$ of the total catalyst) during the first half of the reaction (between 20 and 50% of the maximum conversion possible).

![](images/49b7615147a787eade525c1c74996b82a1d0e57db2419b3a3d7832228ee34250.jpg)

## C.2.3 Mechanism 3

Mechanism 3 is characterized by an off-cycle resting state where the catalytic dimer binds with the substrate to form a $( c a t ) _ { 2 } S$ complex, which is most significant during the initial stages of conversion. It is defined by any set of kinetic constants such that the reaction run $[ S ] _ { 0 } = 1$ and $[ c a t ] _ { 0 } = 5$ mol% has a significant amount of cat $( > 5 \%$ of the total catalyst) at the beginning of the reaction $( < 1 0 \%$ of the maximum conversion possible).

![](images/7ed9208367ed23264e41e961cf43d47fec056f70aea7f42c893ec5ecd00a8a27.jpg)

## C.2.4 Mechanism 5

Mechanism 5 evaluates the framework's scalability by introducing a pathway where the catalyst reversibly binds to the formed product, creating catP complex. A second catalyst molecule is required to assist in the product release

![](images/f92ba2c16ea4b6571363b5c07fa74a87ee14967ae4284099be9424432509dffe.jpg)

$$
\begin{array} { l } { \displaystyle \frac { d [ S ] } { d t } = k _ { - 1 } [ c a t S ] - k _ { 1 } [ S ] [ c a t ] } \\ { \displaystyle \frac { d [ P ] } { d t } = k _ { 3 } [ c a t P ] - k _ { - 3 } [ c a t ] [ P ] } \\ { \displaystyle \frac { d [ c a t ] } { d t } = k _ { - 1 } [ c a t S ] + k _ { 3 } [ c a t P ] - ( k _ { 1 } [ S ] + k _ { - 3 } [ P ] ) [ c a t ] } \\ { \displaystyle \frac { d [ c a t S ] } { d t } = ( k _ { 1 } [ S ] + k _ { - 2 } [ c a t P ] ) [ c a t ] - ( k _ { - 1 } + k _ { 2 } [ c a t ] ) [ c a t S ] } \\ { \displaystyle \frac { d [ c a t P ] } { d t } = k _ { 2 } [ c a t S ] [ c a t ] + k _ { - 3 } [ P ] [ c a t ] - ( k _ { 3 } + k _ { - 2 } [ c a t ] ) [ c a t P ] } \end{array}
$$

## D Extended Simulation Results

Here we provide additional results supporting the main text. We include performance metrics for all the implementations described in our simulation study. To maintain clarity, full visual diagnostics are presented only for the highest-resolution setting (Setup 1: both chemical species, $T = 2 0 )$ . For the remaining regimes, inference performance is summarized via the evaluation metrics.

## D.1 SMC-ABC Results

To establish a performance baseline for our parameter inference and model selection framework, we employed an SMC-ABC algorithm. The distance metric was defined as the $L _ { 2 }$ norm between the "experimental" trajectories and simulated trajectories generated from candidate sets of parameter values. The results presented here correspond to a representative case where the true mechanism underlying the observed synthetic data is $M _ { 2 }$ . We implemented the adaptive threshold SMC-ABC algorithm across the candidate models, initialized with $\epsilon _ { 0 } = 8 . 5$ and an adaptive threshold quantile of $\alpha = 0 . 5 5$ . For this single observation, the algorithm required evaluating 928, 605 simulations to produce 3, 000 samples for each model across 6 generations.

Figure 5 highlights the computational inefficiencies of SMC-ABC in the context of our problem. SMC-ABC requires new simulations every time inference is performed on new data. Inference is costly as several simulations must be produced for one parameter sample to be accepted. The acceptance rate curves in the figure illustrate the exponential increase in required simulations as the tolerance threshold € shrinks. From the first to the sixth generation, the acceptance rates dropped from 17.4%, 16.2%, and 6.1% down to 2.9%, 2.4%, and 2.0% for $M _ { 1 } , M _ { 2 }$ , and $M _ { 3 }$ , respectively. This also demonstrates that as model complexity increases, acceptance rates become substantially lower, even during initial generations.

For the case presented here, our implementation of SMC-ABC failed to identify the ground-truth mechanism. The highest estimated posterior probability belonged to $M _ { 1 } ~ ( 5 3 . 1 \% )$ , as opposed to the true mechanism $M _ { 2 } ~ \left( 4 6 . 8 \% \right)$ . Figure 6 presents a Posterior Predictive Check (PPC) for this case. To generate this PPC, a candidate mechanism is first sampled according to the estimated posterior probabilities, followed by sampling kinetic parameters from that model's estimated posterior. These inferred parameters are then passed through the ODE simulator to generate the 100 inferred trajectories. The resulting PPC shows considerable agreement between the observed data and the inferred trajectories. This outcome highlights the overlap within the candidate space: under certain parameter regimes, distinct CRNs can produce very similar macroscopic concentration profiles.

## D.2 Amortized Neural Inference Results (3 Models)

In our primary study considering $\mathcal { M } = \{ M _ { 1 } , M _ { 2 } , M _ { 3 } \}$ , the amortized framework demonstrates both high classification accuracy and well-calibrated parameter estimation. This section details the performance metrics and presents relevant figures resulting from implementing our approach under four distinct observational setups: (1) both substrate and product observed across all $T = 2 0$ discrete time points; (2) both species observed at $T = 1 0$ random time points; (3) only the product observed across all $T = 2 0$ time points; and (4) only the product observed at $T = 1 0$ random time points.

In all cases, performance was evaluated using a holdout test set of 3, 000 synthetic observations (1, 000 per true mechanism). We used two trained networks to handle all four setups. The first network was trained on trajectories observing both species, enabling inference for setups (1) and (2). The second network was trained observing only the product, for inference under setups (3) and (4). Both networks were trained on trajectories with a randomized number of observed time points (uniformly sampled between 5 and 20). During the evaluation for setups (2) and (4), the 10 available time points were a random subset from the full 20-point temporal grid.

![](images/fa7c23494b5ce2385506703fadd2a4d05f746255fbafb8505b19679b42e528c8.jpg)  
Figure 5: Convergence diagnostics for SMC-ABC inference. The solid lines (left axis) track the shrinkage of the discrepancy threshold (€) across generations for candidate mechanisms M1, M2, and M3. The dashed lines (right axis) denote the corresponding acceptance rates required to maintain the target population of N = 3, 000 particles per generation. As the posterior approximation refines and the € tolerance tightens, the acceptance rate exhibits a rapid decline, reflecting the increasing computational cost of proposing valid kinetic parameters from the transition kernel.

Table 4 details the confusion matrix and corresponding classification metrics for all setups of observation, demonstrating the network's capacity to distinguish the underlying CRN structures. For model selection, the predicted mechanism for each test trajectory is chosen to be the one with the highest posterior probability (i.e., the maximum a posteriori estimate). The metrics shown compare this predicted mechanism with the true mechanism in each test case.

To evaluate the calibration of the estimated posterior probabilities, we computed the Expected Calibration Error (ECE) and evaluated the corresponding reliability diagrams. A perfectly calibrated summary network would produce an ECE of zero and a reliability curve along the identity line, indicating that when the network assigns a p probability to a specific mechanism, that prediction turns out to be correct in exactly $( 1 0 0 * p ) \%$ of the cases.

All reliability diagrams were constructed by splitting the network's outputted posterior probabilities across all test cases into 10 equally spaced bins across the [0, 1| interval. For each bin, we computed the average posterior probability and the empirical accuracy (the fraction of cases from the bin that actually correspond to the mechanism evaluated). The ECE was then calculated as the weighted average of the absolute differences between the mean probability and the empirical accuracy across all bins (weighted by the number of cases per bin). Calibration results for each observational setup are presented in Table 5. As the observational data becomes sparser, the ECE remains low, fluctuating around 1%. This suggests that when missing sufficient data to uniquely identify the true mechanism, the network appropriately scales back its estimated probabilities, correctly quantifying the uncertainty about the true mechanism. The reliability diagram for the main setup is not presented here as it is present in the main text (Figure 2). Figure shows 7 the diagram that corresponds to observational setup (4). Even in this case, with only one of the species available and observing only 10 time steps. the posterior probabilities produced by the model network are well-calibrated.

![](images/4f6442bc81fcdbf9f477e576be84801a710eb3194e5be13542c8057254dcaeaa.jpg)  
Figure 6: PPC for the four experimental observations, where SMC-ABC was implemented. Black lines denote the true observed synthetic data, with solid and dashed lines representing the substrate (S) and product (P) concentrations, respectively. Faint colored lines depict the 100 inferred trajectories generated by drawing from the network's estimated posterior.

Beyond model selection, the network facilitates parameter estimation by generating samples from the models' estimated posterior distributions. Figure 17 illustrates the estimated marginal posteriors (2, 000 samples) for a representative test trajectory generated by mechanism $M _ { 3 }$ (main observational setup). For this test case, the network accurately identified the underlying structural pathway, assigning it a posterior probability approaching 1. To evaluate the reliability of these estimates across the entire test set, we calculate the coverage of the 95% credible intervals—tracking the percentage of cases where the true parameter value falls within this bound. Table 6 contains the coverage results for all four settings.

Further, to assess the calibration of the entire posterior distributions, we rely on Simulation-Based Calibration (SBC) rank histograms. To construct the SBC rank histograms, we used the 2, 000 posterior samples for each of the 1, 000 test trajectories per mechanism. In each case, the true parameter value was ranked among the corresponding posterior samples, resulting in an integer rank between 0 and 2, 000 (normalized to 1). The histograms visualize the distribution of these empirical ranks across the entire test set; posterior calibration would produce a uniform distribution across all possible ranks. Figures 9, 10 and 11 show the rank histograms (right pane) for all parameters of $M _ { 1 } , M _ { 2 }$ and $M _ { 3 }$ respectively. They also include a comparison between the posterior mean estimate of the parameters and the true value across all testing observations. As the plots show, histograms are mostly uniform, even when the parameters seem to be weakly identifiable (for example $k _ { 1 } )$ . This suggests that the network correctly expresses uncertainty about the true parameter values.

Lastly, we conduct Posterior Predictive Checks (PPCs) by simulating data (100 samples) from the inferred parameter distributions to evaluate how well our approach models the true concentration profiles. Figure 12 presents the PPC corresponding to the same test trajectory as the posterior distributions from Figure 17 (generated by mechanism $M _ { 3 }$ under the setup (1): both species observed, $T = 2 0 )$ . The network correctly identifies the CRN structure with absolute confidence and the predictive samples perfectly track the ground-truth trajectories for both the substrate and the product across all four experimental conditions. In contrast, Figure 13 illustrates the network's predictive fit under the sparse observational constraints of setup (4) (only the product observed, $T = 1 0 )$ . For this trajectory generated by mechanism $M _ { 1 }$ , the network successfully identifies the true mechanism but presents some model uncertainty, assigning a posterior probability of $p _ { 1 } = 0 . 8 5$ to $M _ { 1 }$ and allocating the remaining $p _ { 3 } = 0 . 1 5$ probability to mechanism $M _ { 3 }$ . The resulting PPC shows this uncertainty: predictive samples are drawn from both candidate models in accordance with their probabilities. Despite the absence of substrate data and a restricted 10-point temporal resolution, the network not only encompasses the observed product data but successfully reconstructs the unobserved substrate dynamics with high accuracy. The slightly wider predictive variance cleanly reflects the network's uncertainty regarding the missing kinetic information.

Table 4: Classification Performance. Results for the three-model setting $\left( { \mathcal { M } } = \{ M _ { 1 } , M _ { 2 } , M _ { 3 } \} \right)$ across all four observational setups. The left pane displays the confusion matrix across the holdout test set, and the right pane details the corresponding performance metrics.
<table><tr><td rowspan="2">Setup</td><td rowspan="2">Truth</td><td colspan="3">Predicted Model</td><td colspan="3">Performance Metrics</td></tr><tr><td> $M _ { 1 }$ </td><td> $M _ { 2 }$ </td><td> $M _ { 3 }$ </td><td>Precision</td><td>Recall</td><td> $F _ { \mathrm { 1 } } { \mathrm { - s c o r e } }$ </td></tr><tr><td rowspan="2">(1)  $\mathrm { ~ S ~ } \& \mathrm { ~ P ~ } , T = 2 0$ </td><td> $M _ { 1 }$ </td><td>954</td><td>9</td><td>37</td><td>0.93</td><td>0.95</td><td>0.94</td></tr><tr><td> $M _ { 2 }$ </td><td>18</td><td>976</td><td>6</td><td>0.99</td><td>0.98</td><td>0.98</td></tr><tr><td rowspan="2">(2)  $\mathrm { ~ S ~ } \& \mathrm { ~ P ~ } , T = 1 0$ </td><td> $M _ { 3 }$ </td><td>51</td><td>5</td><td>944</td><td>0.96</td><td>0.94</td><td>0.95</td></tr><tr><td> $M _ { 1 }$ </td><td>936</td><td>17</td><td>47</td><td>0.92</td><td>0.94</td><td>0.93</td></tr><tr><td rowspan="2"></td><td> $M _ { 2 }$ </td><td>19</td><td>965</td><td>16</td><td>0.98</td><td>0.96</td><td>0.97</td></tr><tr><td> $M _ { 3 }$ </td><td>64</td><td>7</td><td>929</td><td>0.94</td><td>0.93</td><td>0.93</td></tr><tr><td rowspan="3">(3)  $\mathrm { P \ o n l y , \ } T = 2 0$ </td><td> $M _ { 1 }$ </td><td>943</td><td>14</td><td>43</td><td>0.88</td><td>0.94</td><td>0.91</td></tr><tr><td> $M _ { 2 }$ </td><td>23</td><td>971</td><td>6</td><td>0.98</td><td>0.97</td><td>0.97</td></tr><tr><td> $M _ { 3 }$ </td><td>103</td><td>6</td><td>891</td><td>0.95</td><td>0.89</td><td>0.92</td></tr><tr><td rowspan="3">(4)  $\mathrm { P \ o n l y , \ } T = 1 0$ </td><td> $M _ { 1 }$ </td><td>901</td><td>28</td><td>71</td><td>0.88</td><td>0.90</td><td>0.89</td></tr><tr><td> $M _ { 2 }$ </td><td>28</td><td>951</td><td>21</td><td>0.96</td><td>0.95</td><td>0.95</td></tr><tr><td> $M _ { 3 }$ </td><td>99</td><td>12</td><td>889</td><td>0.91</td><td>0.89</td><td>0.90</td></tr></table>

Table 6: Empirical Coverage for Three Models. Coverage (%) for the 95% credible intervals for the kinetic parameters and observation noise across all four observational setups. Values close to 95% indicate wellcalibrated marginal posteriors.
<table><tr><td rowspan="2">Param</td><td colspan="3">(1) S &amp;  $\mathbf { P } ,$   $T = 2 0$ </td><td colspan="3">(2) S&amp;  $\mathbf { P } ,$   $T = 1 0$ </td><td colspan="3">(3)  $\mathbf { P }$  only,  $T = 2 0$ </td><td colspan="3">(4) P only,  $T = 1 0$ </td></tr><tr><td> $M _ { 1 }$ </td><td> $M _ { 2 }$ </td><td> $M _ { 3 }$ </td><td> $M _ { 1 }$ </td><td> $M _ { 2 }$ </td><td> $M _ { 3 }$ </td><td> $M _ { 1 }$ </td><td> $M _ { 2 }$ </td><td> $M _ { 3 }$ </td><td> $M _ { 1 }$ </td><td> $M _ { 2 }$ </td><td> $M _ { 3 }$ </td></tr><tr><td> $k _ { 1 }$ </td><td>96.2</td><td>96.2</td><td>96.2</td><td>95.0</td><td>96.2</td><td>96.0</td><td>97.7</td><td>94.9</td><td>98.0</td><td>95.9</td><td>92.6</td><td>97.4</td></tr><tr><td> $k _ { - 1 }$ </td><td>96.4</td><td>95.2</td><td>96.0</td><td>97.0</td><td>94.9</td><td>96.4</td><td>95.8</td><td>94.4</td><td>94.9</td><td>96.5</td><td>94.8</td><td>94.0</td></tr><tr><td> $k _ { 2 }$ </td><td>98.1</td><td>97.3</td><td>95.1</td><td>97.5</td><td>94.3</td><td>95.4</td><td>97.8</td><td>95.2</td><td>93.9</td><td>96.5</td><td>94.6</td><td>93.9</td></tr><tr><td> $k _ { - 2 }$ </td><td>97.0</td><td>94.8</td><td>96.8</td><td>97.7</td><td>94.8</td><td>97.3</td><td>97.0</td><td>95.4</td><td>94.3</td><td>96.6</td><td>94.6</td><td>94.8</td></tr><tr><td> $k _ { 3 }$ </td><td></td><td>93.0</td><td>95.2</td><td></td><td>89.5</td><td>93.1</td><td></td><td>97.7</td><td>94.0</td><td></td><td>96.5</td><td>93.4</td></tr><tr><td> $k _ { - 3 }$ </td><td></td><td>96.2</td><td>96.0</td><td></td><td>94.0</td><td>96.1</td><td>一</td><td>95.9</td><td>95.2</td><td></td><td>94.6</td><td>96.1</td></tr><tr><td> $\sigma _ { S }$ </td><td>95.1</td><td>93.0</td><td>95.8</td><td>97.4</td><td>97.0</td><td>97.1</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\sigma _ { P }$ </td><td>95.1</td><td>93.0</td><td>94.3</td><td>97.5</td><td>97.2</td><td>96.9</td><td>95.9</td><td>97.3</td><td>96.6</td><td>93.7</td><td>94.0</td><td>95.0</td></tr></table>

![](images/8822d50a1d9e578446dd3a5393abed48b9f9c8597a0401fe718340c035f4e739.jpg)  
Figure 7: Reliability diagram evaluating the calibration of the posterior probabilities across the 3,000 test trajectories (only P is observed, $T = 1 0 )$ . Predictions are grouped into 10 uniformly spaced probability bins. The dashed diagonal represents perfect calibration, where the estimated probability exactly matches empirical accuracy.  
Table 5: Expected Calibration Error. ECE for posterior probabilities across the four observational setups. Lower values indicate better alignment between estimated probabilities and testing accuracy.

<table><tr><td rowspan="2">Setup</td><td colspan="3">ECE</td></tr><tr><td> $M _ { 1 }$ </td><td> $M _ { 2 }$ </td><td> $M _ { 3 }$ </td></tr><tr><td rowspan="3">(1) S &amp; P, T = 20 (2) S &amp; P, T = 10 (3) P only, T = 20</td><td>0.0060</td><td>0.0056</td><td>0.0059</td></tr><tr><td>0.0083</td><td>0.0064</td><td>0.0107</td></tr><tr><td>0.0124</td><td>0.0052</td><td>0.0094</td></tr><tr><td>(4) P only,  $T = 1 0$ </td><td>0.0078</td><td>0.0064</td><td>0.0071</td></tr></table>

Posterior Distributions - PPC Case 82 (True Model: M3)  
![](images/f62fa3c0ad20045ba0aefe025863edcd767cb33c12979f69e92b1de55a600dcf.jpg)  
Figure 8: Estimated posterior distributions, using 2, 000 samples, for all parameters from a test case where the true mechanism was $M _ { 3 }$ . The true value (in the log-space) is marked with black dashed line.

![](images/fa591c6a2553df24c0b51584f4424722d03166749ce77daa4d2aa6ee85c85aa0.jpg)  
Figure 9: Inference calibration across all testing cases from M1. Left Panel: Posterior mean estimates vs. the true values. Right Panel: Rank histograms for all parameters (normalized to (0, 1)). The dashed black line indicated the level for uniform density (1).

![](images/447e7a397391bbdced9359d2aa732f9945023cd44b6853ba2a412f50ef3e3bbc.jpg)  
Figure 10: Inference calibration across all testing cases from M2. Left Panel: Posterior mean estimates vs. the true values. Right Panel: Rank histograms for all parameters (normalized to (0, 1)). The dashed black line indicated the level for uniform density (1).

![](images/c8c16ea74ec7d86205042ccfb1f5c866c57b4ef4ff2ff8fd2696da0a2a64fe9e.jpg)  
Figure 11: Inference calibration across all testing cases from M3. Left Panel: Posterior mean estimates vs. the true values. Right Panel: Rank histograms for all parameters (normalized to (0, 1)). The dashed black line indicated the level for uniform density (1).

![](images/01024f45b4f0f8d0060ea7329c746b396c1eed5d1873a0dbce8e5cdafc08c462.jpg)  
Figure 12: PPC for a test trajectory generated by mechanism $M _ { 3 }$ under the high-resolution setup (1). Black dots denote the 20 observations for both species across four distinct initial conditions. The solid lines represent 100 posterior predictive trajectories sampled from the network's inferred joint posterior.

![](images/90cb307184ec6f1d7c3fd104084a14d31b74d6842458242ae62f3ab5cb8cda47.jpg)  
Figure 13: PPC for a test trajectory generated by mechanism $M _ { 1 }$ under the low-resolution setup (4). Black dots denote the 10 observations for the product across four distinct initial conditions. The gray dots indicate the corresponding substrate, which was not observed by the model. The solid lines represent 100 posterior predictive trajectories sampled from the network's inferred joint posterior.

## D.3 Scalability Results (4 Models)

To evaluate the scalability of our neural posterior framework, the model space was expanded to include Mechanism 5 $\left( { \mathcal { M } } = \{ M _ { 1 } , M _ { 2 } , M _ { 3 } , M _ { 5 } \} \right)$ . Table 7 details the classification performance over the 4, 000-sample testing set under the high-resolution setting. While the overall classification accuracy remains high at 86%, the confusion matrix reveals the structural uncertainty introduced by the inclusion of $M _ { 5 }$ . Mechanisms $M _ { 2 }$ and $M _ { 3 }$ remain highly distinguishable, maintaining $F _ { 1 }$ -scores of 0.98 and 0.93 respectively. However, some confusion emerges between $M _ { 1 }$ and $M _ { 5 }$ 186 $M _ { 1 }$ trajectories are classified as $M _ { 5 } .$ and 190 $M _ { 5 }$ trajectories are classified as $M _ { 1 }$

Table 7: Classification Performance for Four Models. Results for the expanded four-model setting $( \mathcal { M } =$ $\left\{ M _ { 1 } , M _ { 2 } , M _ { 3 } , M _ { 5 } \right\} )$ under the main observational setup (both species observed, $T = 2 0 )$ . The left pane displays the confusion matrix across the holdout test set, and the right pane details the corresponding performance metrics.
<table><tr><td rowspan="2"></td><td colspan="4">Predicted Model</td><td colspan="3">Performance Metrics</td></tr><tr><td>Truth  $M _ { 1 }$ </td><td> $M _ { 2 }$ </td><td> $M _ { 3 }$ </td><td> $M _ { 5 }$ </td><td>Precision</td><td>Recall</td><td> $F _ { 1 }$  -score</td></tr><tr><td> $M _ { 1 }$ </td><td>789</td><td>6</td><td>19</td><td>186</td><td>0.75</td><td>0.79</td><td>0.77</td></tr><tr><td> $M _ { 2 }$ </td><td>17</td><td>977</td><td>6</td><td>0</td><td>0.98</td><td>0.98</td><td>0.98</td></tr><tr><td> $M _ { 3 }$ </td><td>54</td><td>9</td><td>928</td><td>9</td><td>0.93</td><td>0.93</td><td>0.93</td></tr><tr><td> $M _ { 5 }$ </td><td>190</td><td>4</td><td>43</td><td>763</td><td>0.80</td><td>0.76</td><td>0.78</td></tr></table>

![](images/e9a6c66572c48b6af0e5849a5e310e680f596dbfaf317c4b51dfbcf8a37d34ae.jpg)  
Figure 14: Reliability diagram evaluating the calibration of the posterior probabilities across the 4,000 test trajectories $( P$ and S observed, $T = 2 0 )$ . Predictions are grouped into 10 uniformly spaced probability bins. The dashed diagonal represents perfect calibration, where the estimated probability exactly matches empirical accuracy.  
Table 8: Empirical Coverage for Four Models. Coverage $( \% )$ of the 95% posterior credible intervals for the kinetic parameters and observation noise under the expanded four-model space $\left( { \mathcal M } \ = \ \{ M _ { 1 } , M _ { 2 } , M _ { 3 } , M _ { 5 } \} \right)$ . Both substrate and product observed, $T = 2 0$

<table><tr><td>Parameter</td><td> $M _ { 1 }$ </td><td> $M _ { 2 }$ </td><td> $M _ { 3 }$ </td><td> $M _ { 5 }$ </td></tr><tr><td> $k _ { 1 }$ </td><td>97.1</td><td>94.7</td><td>94.9</td><td>97.8</td></tr><tr><td> $k _ { - 1 }$ </td><td>96.4</td><td>91.8</td><td>94.0</td><td>96.9</td></tr><tr><td> $k _ { 2 }$ </td><td>96.2</td><td>94.3</td><td>96.0</td><td>95.3</td></tr><tr><td> $k _ { - 2 }$ </td><td>95.3</td><td>93.4</td><td>96.9</td><td>96.2</td></tr><tr><td> $k _ { 3 }$ </td><td></td><td>94.2</td><td>96.0</td><td>96.8</td></tr><tr><td> $k _ { - 3 }$ </td><td></td><td>94.8</td><td>96.1</td><td>95.4</td></tr><tr><td> $\sigma _ { S }$ </td><td>96.4</td><td>95.4</td><td>95.8</td><td>96.6</td></tr><tr><td> $\sigma _ { P }$ </td><td>95.3</td><td>94.6</td><td>96.4</td><td>96.2</td></tr></table>

This symmetric error seems to quantify the structural similarities between $M _ { 1 }$ and $M _ { 5 }$ . Under specific parameter regimes, the concentration trajectories generated by these two mechanisms are hard to distinguish, even when both the substrate and product are observed. For example, in Figure 15 we show the PPC from a test case in which both $M _ { 1 } , M _ { 5 }$ , (and even $M _ { 3 } )$ can produce trajectories that tightly track the data. The network correctly captures the uncertainty associated with this kind of case. Moreover, all other assessments indicate inference is well-calibrated for this expanded model too. Table 8 shows coverage results, Figure 14 the reliability diagram for the model selection posterior probabilities, and Figure 16 the rank histograms for every parameter of $M _ { 5 }$ (results for all other mechanisms remained consistent with those from the three-model setting). We believe accuracy could be improved by extending the dimensions of the architecture when considering larger model spaces, but did not explore it for this work.

![](images/6305cb9328d6f7da531f138ee059262b27c539214981dbff7ac11a884daeb277.jpg)  
Figure 15: PPC for a test trajectory generated by mechanism $M _ { 1 }$ under the high-resolution setup (1) with four candidate models. Black dots denote the 20 discrete observations for both species across four distinct initial conditions. The solid lines represent 100 posterior predictive trajectories sampled from the network's inferred joint posterior.

![](images/30da19d84947652868201988b4337a3fed1390c3e40e8f5f4c212e5d07ac7ed1.jpg)  
Figure 16: Inference calibration across all testing cases from M5. Left Panel: Posterior mean estimates vs. the true values. Right Panel: Rank histograms for all parameters (normalized to (0, 1)). The dashed black line indicated the level for uniform density (1).

## E Protodeboronation Example: Details and Results

Catalytic protodeboronation is an undesirable degradation process that frequently compromises Suzuki-Miyaura cross-coupling reactions [Ser et al., 2025]. To optimize the reaction conditions and mitigate this degradation, chemists must first identify the primary kinetic pathway driving the catalyst deactivation Using mechanistic candidates formalized by Burés and Larrosa [2023], we formulate this as a model selection problem over three competing deactivation hypotheses: a unimolecular deactivation of the active catalyst-substrate complex $( M _ { 1 4 } )$ , an inhibitor-induced deactivation where an external agent neutralizes the complex $( M _ { 1 5 } )$ , and a substrate-induced deactivation where a second substrate molecule triggers the degradation $( M _ { 1 6 } )$ . We detail this mechanisms below.

## E.1 Mechanism 14

Mechanism 14 is defined with any set of reaction rate constants such that at least one reaction with the reaction run $[ S ] _ { 0 } = 1$ and $[ c a t ] _ { 0 }$ between 3 and 7 mol% of catalyst has between 50 and 90% of active catalyst $( c a t + c a t S )$ at 50% of the maximum conversion.

![](images/3fe1852b7f1f7b1e00f215e00802939a67b7d72d30fbd1fc8932c1e037fb67ba.jpg)

![](images/e69aec56426ffed38a47088c4cb208ac62dd1651a9f0f45a53dc9ca5421625f7.jpg)

## E.2 Mechanism 15

Mechanism 15 is defined with any set of reaction rate constants such that at least one reaction with the reaction run $[ S ] _ { 0 } = 1$ and $[ c a t ] _ { 0 }$ between 3 and 7 mol% of catalyst has more than 10% of inactive catalyst (inactivecatSI) at 50% of the maximum conversion.

![](images/036a1750cbe6d03721dd7c68429ddc93934e35778c469e3008d0f3e31bd21b91.jpg)

![](images/ae7133a174753c9f1a16011ff4a9316a0c07ba92ecd139973af55092e9f0081d.jpg)

## E.3 Mechanism 16

Mechanism 16 is defined with any set of reaction rate constants such that at least one reaction with the reaction run $[ S ] _ { 0 } = 1$ and $[ c a t ] _ { 0 }$ between 3 and 7 mol% of catalyst has between 50 and 90% of active catalyst (cat + catS) at 50% of the maximum conversion.

![](images/366eaf95e94faaf4f9cdc08246bc142baca211833bc94fec9607288e315b7335.jpg)

## E.4 Inference Details and Results

Real-world kinetic data is limited by experimental constraints. In this application, the available observational data consists solely of product yield measurements, sampled at just 8 time points. The experimental data was collected by Ser et al. [2025] (we take them from their Table S22) across $N _ { \mathrm { e x p } } = 3$ independent setups featuring varying initial catalyst loadings (5%, 10%, and 15%). This represents a regime of low observability: intermediate species—such as the active catalyst, and the degraded inactive species—are unobserved. Under these conditions, traditional likelihood-free inference methods struggle significantly due to structural ambiguity and high rejection rates.

To perform inference and mechanism selection, we trained a dedicated instance of our neural inference pipeline tailored to this three-model hypothesis space $\left( \mathcal { M } = \{ M _ { 1 4 } , M _ { 1 5 } , M _ { 1 6 } \} \right)$ . The architecture was trained on synthetic trajectories (936, 000 per model) simulated from the corresponding ODEs, with only the product observed. The permutation-invariant SAB within our summary network naturally accommodates the three independent initial conditions, processing the 5%, 10%, and 15% catalyst loading trajectories as an unordered set to extract global kinetic features.

For inference, the time-normalized experimental time-series were passed through the trained architecture to compute posterior model probabilities and estimate the joint posterior distributions for the kinetic rate constants.

Table 9 provides the marginal posterior point estimates (mean and median) for the kinetic rate constants and measurement noise across all three candidate models $\left( M _ { 1 4 } , M _ { 1 5 } , M _ { 1 6 } \right)$ evaluated in Section 4. The estimates are provided in logarithmic physical units. To quantify the uncertainty surrounding these estimates, Figure 17 visualizes the marginal posterior distributions for the highly favored inhibitor-induced pathway $( M _ { 1 5 } )$ . Lastly, Figure 18 shows the corresponding pairwise posterior distributions. The plots show that even though the posterior distributions for some of the parameters are very wide, the posterior samples account for relationships between them.

Table 9: Parameter Estimates. Posterior parameter estimates for the protodeboronation structural hypotheses.
<table><tr><td rowspan="2">Parameter</td><td colspan="2"> $M _ { 1 4 }$  (3.71%)</td><td colspan="2"> $M _ { 1 5 }$  (92.19%)</td><td colspan="2"> $M _ { 1 6 } ~ ( 4 . 1 0 \% )$ </td></tr><tr><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td></tr><tr><td>log  $k _ { 1 }$ </td><td>3.9572</td><td>2.2728</td><td>3.5822</td><td>1.0382</td><td>4.2962</td><td>2.6632</td></tr><tr><td> $\log k _ { - 1 }$ </td><td>-5.6564</td><td>-6.7500</td><td>-6.1921</td><td>-7.2830</td><td>-5.8420</td><td>-6.8342</td></tr><tr><td>log k2</td><td>-1.8568</td><td>-3.9131</td><td>-1.8872</td><td>-2.5498</td><td>-2.0608</td><td>-4.0818</td></tr><tr><td> $\log { k _ { - 2 } }$ </td><td>-0.9392</td><td>-1.4539</td><td>-0.2301</td><td>-1.2468</td><td>-1.0966</td><td>-1.8808</td></tr><tr><td> $\log k _ { - 3 }$ </td><td>-2.1533</td><td>-4.2273</td><td>9.9645</td><td>10.5861</td><td>2.2611</td><td>0.2867</td></tr><tr><td>log  $\sigma _ { P }$ </td><td>-5.9537</td><td>-5.9104</td><td>-6.0284</td><td>-6.0212</td><td>-5.9492</td><td>-5.9434</td></tr></table>

Parameter Posterior Distributions - Protodeboronation  
![](images/248605789703b03910f3ecddd80eced74079ca530314ba38905d4bf00218bef8.jpg)  
Figure 17: Estimated posterior distributions for $M _ { 1 5 } ,$ which was estimated to be the most likely mechanism from our candidate models. We use 2, 000 samples, for all parameters estimated from real protodeboronation data. The estimated posterior mean (in the log-space) is marked with black dashed line.

![](images/1a1e999b21d64a5dfe7d80ae1bb2f3406c4c6bfb09a74c6fbee8e9d1ed5b90f6.jpg)

![](images/230113b926e36cc9ad8df6d8ecaedb53e205e78653ae8daa4ae4c0062b3d91c8.jpg)

![](images/55ca0c7061ab554094ee485a4a73db8a1aacf6c3e7830d8fafdcd1bc79c5a976.jpg)

![](images/b880a7ec955cdb847222d0ddb18f8e92b773e7ec9a3c3fd9903a57281e59b9fb.jpg)

![](images/29b951c3ce2c94c8fe0e1fd45a2babdf143cec8c8be0266dae6a4a74ec13dbc1.jpg)

![](images/06992dd46dc4f42167d00f2767ebd37c1d43286e19eec57ec4d32e2eb2bd10c9.jpg)

![](images/d6fb500dae2835ec76ce9c4fa88eddfa02916b13bd6fd2aca657e018df1bf448.jpg)

![](images/14c811c51fa2bfbd1df63fbe92a6c667df55f24b8f9c09a053005e0cabb71baf.jpg)

![](images/ee88a9ecd36e5a41d471a4c27e4e7750124c68abb0dce4768083c5b185ca466a.jpg)

![](images/6fdf98d11e198a2db939463f392f6a0a37863f25e8a94ee9ebe2b1fcccaa4639.jpg)

![](images/016bb34186a0d46a80f6460af61cb35c13eed9d32a8edde4267906cc8d80e92c.jpg)

![](images/7187b91f3f176c7948d8f5916a5b7f6e14840475eefc45ab0763a45245c29522.jpg)

![](images/ab7ff85bb9079321a57d4f59ad43e675a3f68cf995f18dfec4e9dd9e4321ace8.jpg)

![](images/6ff67ea582470d143abb0b4b645045db70e3fdb2f238ef39d4477f7bd799a702.jpg)

![](images/011c79b98d247282aff38ed979d110f1ab34272e5bd6769d01af032b0667e7d9.jpg)

![](images/47857e3111a9f8f3df88f4c07082aeb51b7596132945a44a72e82fb6f58dc660.jpg)

![](images/a1549db67a45319095f290f1b37df67e346499014217380a739063d3c460b90b.jpg)

![](images/fe6e67d9f1d5080829a7a48e4597c8c9d54756596aef780dbe936547c6aa8b21.jpg)

![](images/cb84995257ebde5549e03d451c99b9bb21f485c0d91b573a4e9a47762f1439bd.jpg)

![](images/62f3084ec62f39bc934d2ed2d817da0bcc0e8d64de00393eac9c22e077a6ba6c.jpg)

![](images/70bba271a38bc40732a816e38cf60a0d510d929d15420e680416a48e49899402.jpg)  
Figure 18: Bivariate and marginal posterior distributions (2, 000 samples) for the most probable protodeboronation mechanism. The diagonal panels display the marginal posterior distributions. The lower-triangular panels feature bivariate KDE contour maps, illustrating the pairwise uncertainty between the kinetic constants.