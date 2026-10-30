# OCCAMS: A Multi-Task Learning-Based Investigation of the CMB Power Spectra

Ankit Mandal,<sup>1★</sup> Rahul Shah,<sup>1</sup>† Md. Wali Hossain,<sup>2</sup>‡ Somasri Sen,<sup>2</sup>§ Anjan A Sen,<sup>3</sup>¶ Supratik Pal<sup>1</sup>∥

<sup>1</sup>Physics and Applied Mathematics Unit, Indian Statistical Institute, 203 B.T. Road, Kolkata 700 108, India

<sup>2</sup>Department ofPhysics, Jamia Millia Islamia, New Delhi 110025, India

<sup>3</sup>Centrefor Theoretical Physics, Jamia Millia Islamia, New Delhi 110025, India

## ABSTRACT

How many latent variables are needed to fit the cosmic microwave background (CMB) temperature and polarisation power spectra when a neural network is given no cosmological model, and to what extent do they map onto the standard cosmological parameters? We use multi-task learning with variational autoencoders trained on CLASS-generated spectra (TT. TE. EE) over $2 \leq \ell \leq 2 5 0 0$ . No cosmological model is built into the network, and the number of latent variables, $L ,$ is varied over $L = 5 ,$ 6 and 7. Their fits are compared to the Planck 2018 data using $\hat { \chi } ^ { 2 } .$ , AIC, BIC and Bayesian evidence. Their correspondence with cosmological parameters is measured with normalised mutual information post-training. The loss weights are set within a single training run by uncertainty weighting, with the weight of the disentanglement term either fixed (semi-automated, SA) or learned (completely automated, CA). The reconstruction loss is split over multipole ranges in three configurations that separate the roles of temperature and polarisation and of the low- and high-ℓ ranges. In four of the six combinations of configuration and strategy, the fit improves from � = 5 to � = 6 and not beyond, and $L = 6$ is the best, or close to the best, in five of the six cases. So, six latent variables are needed, as many as the free parameters of the standard, model-dependent approach. SA gives clearer disentanglement, while CA needs no a priori choice of the disentanglement weight and has the higher evidence at $L = 5$ and 6: the two strategies are points on a trade-of between interpretability and fit to the data. Notably, the latent variables for � and $A _ { s }$ share almost no information in any model, consistent with large-scale polarisation separating the two, as in standard analyses. A deep-learning approach can thus give a compact, data-driven parametrisation of the CMB spectra.

Key words: cosmological parameters – cosmic background radiation – cosmology: observations – methods: statistical – methods: data analysis – software: machine learning

## 1 INTRODUCTION

Building physically motivated cosmological models that fit the observed data is a central aim of cosmology. Traditionally, this has been done theoretically, using physical understanding of the processes that govern the evolution and structure of the components of the Universe, backed by mathematical modelling that ensures good agreement between the predicted observables and the cosmological observations (Weinberg 2008; Peebles 2020). Among several maximally symmetric spaces that have been employed to model our universe (Landau & Lifschits 1975), the Friedmann–Lemaître–Robertson–Walker (FLRW) cosmology is a natural choice to describe a universe that is homogeneous and isotropic on large scales, at least from recombination to today. Within the FLRW framework, the Λ Cold Dark Matter (ΛCDM) model is considered the benchmark of cosmology, owing to its simplicity and the extent of its agreement with observations. It describes the dynamics of the Universe, from the Big Bang to the present day, within a single framework with only six parameters, namely the physical densities of baryons $( \omega _ { b } )$ and cold dark matter $( \omega _ { c } )$ , the reduced Hubble constant (ℎ), the optical depth to reionisation (�), and the amplitude $\left( A _ { s } \right)$ and spectral index $\left( n _ { s } \right)$ of the primordial scalar perturbations (Dodelson & Schmidt 2020). In that sense, our understanding of the universe depends both on the assumed background geometry (FLRW) and on the assumed model (ΛCDM).

However, modern cosmology has entered an era of unprecedented precision, driven by large volumes of high-quality data across diverse scales and epochs, and ΛCDM now faces several challenges (Verde et al. 2019; Di Valentino et al. 202 $^ \mathrm { 1 b , a ; }$ Perivolaropoulos & Skara 2022; Schöneberg et al. 2022; Abdalla et al. 2022; Kamionkowski & Riess 2023; Giarè 2024; Hu & Wang 2023; Verde et al. 2024; Di Valentino & Brout 2024; Efstathiou 2025; Di Valentino et al. 2025; Riess et al. 2022; Abbott et al. 2022). These challenges warrant a closer look at the assumptions that enter the analysis, with the choice of parameters being one of them. The six parameters of ΛCDM are defined on theoretical grounds and for ease of physical interpretation, and standard Markov chain Monte Carlo (MCMC) analyses typically sample directly in them. The data, however, most strongly constrain combinations of these parameters rather than the individual ones (Efstathiou & Bond 1999; Kosowsky et al. 2002). Sampling in a rigid theoretical parametrisation could therefore lead to complex degenerate posteriors and leave the results vulnerable to prior-volume efects (Trotta 2008; Handley & Lemos 2019).

Various statistical and algorithmic strategies have been developed to address these inferential challenges. Examples include profile likelihood methods, which avoid prior-volume efects (Gómez-Valent 2022; Herold et al. 2022, 2025), and simulation-based inference, which dispenses with an explicit likelihood (Alsing et al. 2019). A further route is to extract the efective underlying dimensionality of the data directly, without imposing a physical parametrisation, using machine-learning techniques such as principal component analysis (Huterer & Starkman 2003; Leach 2006; Peiris & Verde 2010; Hu & Holder 2003), support vector machines (Lochner et al. 2016; Jones & Singal 2017) and neural/deep-learning networks (Charnock et al. 2018; Portillo et al. 2020).

Recently, deep learning architectures that can learn efective representations directly from the data have been applied to a variety of cosmological data sets (Dvorkin et al. 2022; Rojas et al. 2025; Shah et al. 2024). Instead of working within a predefined parametrisation, such an architecture can discover one that is optimised for the data, and it does not rely on any a priori information about specific cosmological models. In particular, encoder-decoder networks such as autoencoders (Hinton & Salakhutdinov 2006) and variational autoencoders (VAE) (Kingma & Welling 2013; Rezende et al. 2014) have become popular for identifying the essential features of cosmological data sets, as well as for reconstructing them independent of a cosmological model. A VAE compresses each data vector into a small number of latent variables from which the data can be reconstructed. It also regularises the distribution of these variables, so that the latent space forms a compact and smooth parametrisation of the data. VAEs have been shown to reconstruct and generate non-linear cosmic structures (Lanusse et al. 2021), to learn compact and semantically rich latent representations of field level cold dark matter simulation data (Kannan et al. 2026), to map dark matter onto baryonic fields when trained on cosmological simulations (Horowitz et al. 2022), and to encode complementary representations of baryonic efects (Lin et al. 2026).

Among all cosmological observations, the temperature and polarisation anisotropies of the cosmic microwave background (CMB) are among the most precisely measured, from the Wilkinson Microwave Anisotropy Probe (WMAP) (Hinshaw et al. 2013) through Planck 2015 (Ade et al. 2016) to the Planck 2018 release (Aghanim et al. 2020b), complemented with similar missions like the South Pole Telescope (SPT) (Benson et al. 2014) and Atacama Cosmology Telescope (ACT) (Louis et al. 2025). These measurements have led to exceptionally tight constraints on the ΛCDM parameters, and CMB physics is among the most rigorously tested and theoretically best understood areas of cosmology. Six parameters are needed to fit the CMB within ΛCDM, and the data leave little room for more: additional parameters do not significantly improve the goodness of fit. Combining the CMB with other probes can make extra parameters useful, but the improved fit then comes at the price of a more complex model, so a compromise has to be struck with the help of model selection criteria (Liddle 2004). In the conventional modeldependent route, this exercise has to be repeated whenever a new model is introduced. A data-driven parametrisation turns it into a one-shot exercise, carried out without bias from any individual predefined model. In this work we follow the data-driven route and ask a more basic question: how many independent degrees of freedom do the CMB data contain if no physical parametrisation is imposed in advance? Before such a method can be trusted on extensions of the standard cosmological model, it has to be tested on a case where the expected answer can be checked. We, hence, do not claim complete theory independence, since the training spectra are generated within ΛCDM. However, no cosmological model is built into the network or its latent space, and this is what makes the outcomes data-driven.

Piras et al. (2025) recently demonstrated the eficacy of VAEs in parametrising the CMB temperature anisotropy power spectrum (TT) at $\ell \geq 3 0$ into a disentangled, data-driven latent space. They manually tuned the weight of the Kullback–Leibler (KL) divergence term to obtain disentangled latent variables, interpreted them with mutual information (MI) (Cover & Thomas 2012), found that five latent variables sufice (one fewer than the six ΛCDM parameters) to explain the data, and obtained constraints from Planck 2018 TT data in latent space. A subsequent study by Sun et al. (2026) included binned TT, TE and EE spectra at $\ell \geq 3 0$ in a parameter-conditioned VAE and again found that five latent variables sufice. In their model, information about each parameter is spread over several latent variables, and no single latent variable corresponds cleanly to a single parameter.

Three aspects of the problem remain open. First, the earlier studies use only $\ell \geq 3 0$ (Piras et al. 2025; Sun et al. 2026). At these multipoles the spectra depend on $A _ { s }$ and � essentially only through the combination $A _ { s } e ^ { - 2 \tau }$ , which makes these two parameters degenerate and accounts for the five latent variables found there instead of the six theoretical parameters of ΛCDM. Whether extending to $\ell \geq 2$ and including polarisation, in particular the reionisation bump (Zaldarriaga 1997) in the large-scale EE spectrum, changes the number of latent variables the data require has, to our knowledge, not been studied with a VAE, and we do not assume the answer a priori. Six parameters generate the spectra, but the data can be sensitive to fewer independent combinations of them, as is the case for $A _ { s }$ and � at $\ell \geq 3 0$ (Aghanim et al. 2020b), and a VAE is free to merge such degrees of freedom. How many latent variables the full CMB data require is therefore an open question. Second, to our knowledge, a clean one-to-one correspondence between latent variables and the base ΛCDM parameters has not been obtained by tuning the KL divergence weight or by conditioning alone. Accurate reconstruction does not guarantee that each latent variable follows one physical direction, and a representation that reconstructs the spectra well can still be hard to interpret (Locatello et al. 2018). Third, the training objective (loss function) with several terms needs a weight for each of them. These weights are usually chosen by hand and tuned over many training runs. A scheme that chooses all of them automatically from the data in a single run would remove this arbitrariness, in addition to reducing overall computational cost.

We address the first two points by asking, first, how many latent variables (�) sufice to fit the full CMB data and, second, to what extent these latent variables can be mapped onto the six standard ΛCDM parameters. The analysis uses the combined TT, TE and EE spectra over $2 \leq \ell \leq 2 5 0 0$ , in three configurations of the reconstruction loss that treat the low- and high-ℓ ranges and the temperature and polarisation spectra diferently, so that the role of polarisation is assessed explicitly rather than being absorbed into a single data vector. The central feature of our approach is that no cosmological model or theory is built into the network architecture, i.e., the reconstruction and KL divergence terms of the loss carry no physical information, and the encoder and decoder act only on the spectra. Moreover, the latent variables are given complete freedom: the parameter each one should follow is not fixed in advance, and all share the same standard normal prior. Nor is the number of latent variables imposed strongly;

we train separate models with $L = 5 ,$ 6 and $^ { 7 , }$ and the comparison of their fits to real data shows how many latent variables are needed. The range is centred on six, i.e. the number of ΛCDM parameters, so that the neighbourhood of the expected answer is searched. Of course, the same scan can in principle be extended to any range of $L$ until the minimum (and optimal) number is found through the statistical measures employed in this work. To encourage the latent variables to follow distinct physical directions, an explicit disentanglement term is added to the training objective. This term uses the parameters of the training samples only as a soft penalty against latent variables that mix several parameters or duplicate one another. It does not fix $L ,$ and it does not assign any latent variable to a given parameter. Each latent variable is still free to follow whichever parameter the data favours, and the resulting mapping to cosmological parameters is inferred only post-training. The aim is not merely accurate reconstruction, but a compact and interpretable data-driven parametrisation of the CMB spectra.

These considerations make the training objective more complex. Because physical efects and noise levels vary across angular scales, the reconstruction loss must be split across multipole ranges, each with its own hyperparameter that sets its importance in the total loss function, and the disentanglement term introduces a further hyperparameter, $w _ { d } .$ , that sets its importance relative to the reconstruction and regularisation terms. This latter hyperparameter is hard to choose by hand: increasing $w _ { d }$ improves the separation of latent information but can afect reconstruction accuracy, and the best balance may depend on the specific configuration and the multipole ranges. On the other hand, a manual scan over $w _ { d } ,$ together with the scale-dependent reconstruction hyperparameters, would require a large number of training runs, and the result would rely on manual tuning. To avoid this, multi-task learning with uncertainty weighting (Cipolla et al. 2018) is used, in which every loss term has its hyperparameter made trainable, so that it is learned together with the network in a single training run. We consider two strategies for the disentanglement weight $w _ { d } .$ . In the semi-automated (SA) strategy, the weights of the reconstruction and KL divergence terms are learned, while $w _ { d }$ is fixed by hand to a large value $( w _ { d } = 5 )$ . In the completely automated (CA) strategy, all the weights, including $w _ { d } .$ , are learned from the data in the same training run, so that no hyperparameter is chosen by hand. We use both because they answer diferent questions: SA shows what a strong disentanglement loss achieves, CA shows whether the weights can be set entirely from the data, and comparing them shows how the number of latent variables and the interpretability depend on the strength of the disentanglement loss. The main contributions of this work are: (i) a deep-learning parametrisation of the CMB in which no cosmological model is built into the network or its latent space, covering the full multipole range, including polarisation, (ii) an answer to the two questions posed above, namely how many latent variables sufice to fit the full CMB data and to what extent they map onto the six ΛCDM parameters, and (iii) the automated tuning of the hyperparameters of the reconstruction and KL divergence terms, and in the CA strategy also of the disentanglement term, within a single training run. The framework is implemented in our code OCCAMS (Optimal Components of CMB with Autoencoders and Model Selection).

The paper is organised as follows. Section 2 introduces VAEs and the basic form of their loss function. Section 3 describes the network architecture, the training data, the loss function with the disentanglement term, and the multi-task tuning of the hyperparameters. Section 4 presents the reconstruction accuracy, the interpretation of the latent space through normalised MI, and the latent-space fits to

Planck 2018 data with the resulting model comparison metrics. We discuss the results in Section 5 and conclude in Section 6.

## 2 DEEP GENERATIVE MODELLING AND VARIATIONAL AUTOENCODERS

A Variational Autoencoder (VAE) (Kingma & Welling 2013; Rezende et al. 2014) learns a compact representation of its input in an unsupervised manner. An encoder network compresses each input spectrum � into a low-dimensional latent vector $z \in \mathbb { R } ^ { L }$ (the “bottleneck”), and a decoder network reconstructs the spectrum from $z .$ Unlike a standard autoencoder, the encoder does not output a single point. For each input it outputs a vector of means $\mu ( x )$ and a vector of standard deviations $\pmb { \sigma } ( \pmb { x } )$ of a Gaussian distribution $q _ { \phi } ( \boldsymbol { z } | \boldsymbol { x } )$ , where $\phi$ denotes the parameters of the encoder network, and � is drawn from this distribution before it is passed to the decoder. The network is trained by minimising,

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { r e c o n } } \left( D _ { \ell , \mathrm { t r u e } } , D _ { \ell , \mathrm { p r e d } } \right) + \beta \mathcal { D } _ { K L } \left[ q _ { \phi } ( z | \boldsymbol { x } ) \parallel p ( z ) \right] .\tag{1}
$$

The first term measures the network’s reconstruction accuracy through the mean squared error (MSE) between the rescaled reconstructed angular power spectra $\begin{array} { r } { D _ { \ell , \mathrm { p r e d } } \ = \ \frac { \ell ( \ell + 1 ) } { 2 \pi } C _ { \ell , \mathrm { p r e d } } } \end{array}$ and the rescaled target spectra $\begin{array} { r } { D _ { \ell , \mathrm { t r u e } } = \frac { \ell ( \ell + 1 ) } { 2 \pi } C _ { \ell , \mathrm { t r u e } } . } \end{array}$ Here the target $\mathrm { \Delta ^ { 6 6 } t r u e } ^ { 3 }$ spectra are the spectra of the training set, which are also the input to the encoder, so that the network is trained to reproduce its own input. A VAE has to be trained on a large set of spectra that covers the range it is expected to represent, and since there is no analytic description of the space of possible CMB spectra, this set has to be produced by a simulator. We use the ΛCDM model purely as such a simulator of realistic spectra, because it is the standard model that fits the CMB data well. Its inputs are the parameters $\Theta = ( \omega _ { \mathrm { b } } , \omega _ { \mathrm { c d m } } , h , \tau , n _ { \mathrm { s } } , \ln ( 1 0 ^ { 1 0 } A _ { \mathrm { s } } ) ) \mathrm { : }$ the physical densities of baryons and cold dark matter, the reduced Hubble constant, the optical depth to reionisation, and the spectral index and amplitude of the primordial scalar perturbations. The network, however, only ever sees the resulting spectra. It is not told how many parameters were used to generate them, nor what they are, and the number of latent variables it needs is not set by this number, since the spectra may depend on fewer independent combinations of the parameters than there are parameters. Further details about the training process are described in Section 3.2. The second term is the KL divergence regulariser, which encourages the learned latent distribution $q _ { \phi } ( \boldsymbol { z } | \boldsymbol { x } )$ to be similar to a predefined prior distribution $p ( z )$ . The prior is taken to be a multivariate Gaussian with zero mean and identity covariance, $p ( z ) = N ( 0 , I )$

A primary challenge in training such VAEs is balancing the reconstruction accuracy against the disentanglement of the latent space. This balance is governed by the hyperparameter $\beta$ (Burgess et al. 2018), which multiplies the KL divergence term in the loss function. An optimal $\beta$ is not known in advance. It is usually found by training the model several times with diferent values and choosing the one that gives the best compromise between reconstruction accuracy and interpretability. The modifications to the loss function required for the objectives of this work introduce further hyperparameters of the same kind beyond $\beta ,$ which are discussed in the following sections. Scanning over all of them by hand is computationally prohibitive, which motivates the multi-task learning approach of Section 3.4.

![](images/0ec1e6f2da4358c9529ff05ce9c1f4f54ebb0c4fbe78ecee8da324d5c8b7680c.jpg)  
Figure 1. The VAE network architecture. The input spectra are compressed via successive 1D convolutions into a latent distribution defined by � and log $\sigma ^ { 2 } .$ . A stochastic sample � is drawn and subsequently upsampled via transposed convolutions to reconstruct the original spectra.

## 3 METHODOLOGY

## 3.1 Architecture

The compression and reconstruction pipeline is a 1D convolutional VAE, designed to map the high-dimensional CMB spectra into a continuous, low-dimensional probabilistic space. The architecture consists of three primary components: an encoder network, a stochastic latent bottleneck, and a decoder network (illustrated in Fig. 1). The full pipeline is implemented in OCCAMS (Optimal Components of CMB with Autoencoders and Model Selection), which is our code developed for this analysis.

Throughout the network we use the trainable activation function adopted by Piras et al. (2025). For an input � it is defined as, $f ( x ) =$ $x \cdot \left[ \gamma + \left( 1 - \gamma \right) \right.$ sigmoid(��)], where � and � are learnable parameters optimised during training together with the other parameters of the network. It adds a trainable non-linearity that interpolates between a linear and a sigmoid-gated response. Each convolutional layer, except the final output layer of the decoder, is followed by this activation and 1D batch normalisation to stabilise training.

The encoder $q _ { \phi } ( z | x )$ serves as the inference network, compressing the multi-channel input sequences into the parameters of the approximate posterior distribution. The input consists of three channels, the spectra $D _ { \ell } ^ { T T } , D _ { \ell } ^ { T E }$ and $D _ { \ell } ^ { E E }$ , each sampled at $N _ { \ell }$ multipoles over $2 \leq \ell \leq \bar { 2 } 5 0 0$ . The network applies three successive 1D convolutional blocks. The first block expands the 3 input channels to 16 channels using a stride of 3 and a kernel size chosen so that the decoder reproduces exactly $N _ { \ell }$ multipoles. The second block expands the 16 channels to 32 channels using a kernel size of 16 and a stride of 3. The third block further expands the 32 channels to 64 channels using a kernel size of 3 and a stride of 3. Following the feature extraction, the 64-channel output is flattened into a 5760-dimensional vector. A fully connected linear layer then projects this representation into a 2�-dimensional vector, which is split equally to parametrise the mean (�) and the log-variance (log $\sigma ^ { 2 } )$ of the latent distribution. To ensure numerical stability during optimisation, the log-variance is clamped between [−10, 10].

The latent representation is an �-dimensional Gaussian distribution with diagonal covariance. � is a fixed choice of the architecture for each model: we train separate models with $L = 5 , 6$ and 7 and compare them in Section 4.3. This range is chosen around the six parameters of ΛCDM, and can be extended to any range of � if the scan does not enclose the minimum number. To allow gradients to backpropagate through the stochastic sampling process, the standard reparametrisation trick is employed: the latent vector is sampled as $\begin{array} { r } { z = \mu + \epsilon \odot \exp ( { \frac { 1 } { 2 } } \log \sigma ^ { 2 } ) } \end{array}$ , where $\epsilon \sim { \cal N } ( 0 , I )$ is standard Gaussian noise and ⊙ represents element-wise multiplication.

The decoder $p _ { \theta } ( x | z )$ , where � denotes the parameters of the decoder network, acts as the generative network, mapping the �- dimensional latent vector � back to the space of the three spectra. The sampled vector � is first passed through a dense layer, expanding it to 5760 features. This is reshaped into a 3D tensor of 64 channels with a spatial length of 90. The spatial resolution is progressively upsampled using a sequence of 1D transposed convolutions (ConvTranspose1d). The first block maintains 64 channels using a kernel size of 3 and a stride of 3, accompanied by the activation and batch normalisation. The second block reduces the feature maps from 64 to 32 channels using a kernel size of 16 and a stride of 3. Finally, the third block projects the 32 channels back to the original 3 output channels $( D _ { \ell } ^ { T T } , D _ { \ell } ^ { T E } , D _ { \ell } ^ { E E } )$ using a stride of 3 and a kernel size matching that of the first encoder block. No activation function is applied at this final layer, allowing the network to directly output the reconstructed values.

## 3.2 Training Data

A total of 200,000 sets of spectra $( C _ { \ell } ^ { T T } , C _ { \ell } ^ { T E } , C _ { \ell } ^ { E E } )$ are generated with the Einstein–Boltzmann solver CLASS (Lesgourgues 2011; Blas et al. 2011). CLASS numerically solves the coupled equations that govern the evolution of the photons, baryons, dark matter and neutrinos in the early Universe, and from their solution computes the CMB angular power spectra. For the purposes of this work it can be regarded as a deterministic simulator that maps a set of input cosmological parameters (six in the case of ΛCDM) to the corresponding temperature and polarisation spectra, and it is used only to produce the training examples. The six ΛCDM parameters are drawn with Latin hypercube sampling within the ranges given in Table 1. In this scheme the range of each parameter is divided into as many equal intervals as there are samples, one value is drawn at random within each interval, and the values of the diferent parameters are combined in a random order. Every parameter range is thus covered evenly, while the combinations of parameters are random. The training set therefore contains no preferred cosmology and no imposed correlations between the parameters, so any dependence of the spectra on fewer than six independent combinations of them is a property of the spectra and not of how the sample was drawn. The spectra cover $2 \leq \ell \leq 2 5 0 0$ . The samples are split into training (80%), validation (10%) and test (10%) sets.

The spectra are converted to $D _ { \ell } \ = \ \ell ( \ell + 1 ) C _ { \ell } / 2 \pi$ and preprocessed following the scheme of Piras et al. (2025). Each spectrum is first divided by a arbitrary reference spectrum (diferent for TT,

<table><tr><td>Parameter</td><td>Sampling Range</td></tr><tr><td> $\omega _ { \mathrm { b } }$ </td><td>[0.020, 0.024]</td></tr><tr><td> $\omega _ { \mathrm { c d m } }$ </td><td>[0.10,0.13]</td></tr><tr><td> $h$ </td><td>[0.62, 0.80]</td></tr><tr><td> $\tau$ </td><td>[0.01,0.13]</td></tr><tr><td> $n _ { s }$ </td><td>[0.92, 1.01]</td></tr><tr><td> $\ln ( 1 0 ^ { 1 0 } A _ { \mathrm { s } } )$ </td><td>[2.90, 3.18]</td></tr></table>

Table 1. Uniform sampling ranges used to generate the training data set via a Latin hypercube. Following Piras et al. (2025), the ranges span approximately 10� around the Planck 2018 baseline best-fit values. There are two exceptions to these bounds: the lower limit for the optical depth (�) is restricted to 0.01 to prevent non-physical negative values, and the upper limit for the reduced Hubble constant (ℎ) is broadened to encompass the local distance-ladder measurement from the SH0ES collaboration (Riess et al. 2022).

TE and EE), which facilitates training. A base-10 logarithm is then taken, followed by normalisation of the resulting data vectors. Since the TE spectra change sign, a vertical ofset is imposed so that the logarithms are well defined. The VAE output is processed through the same steps in reverse order to obtain $D _ { \ell , \mathrm { p r e d } }$

## 3.3 Loss Function

## 3.3.1 Temperature Anisotropies (TT)

The temperature fluctuations of the CMB, $\begin{array} { r l } { \Delta T ( \hat { n } ) / T } & { { } = } \end{array}$ $\begin{array} { r l } {  { \sum _ { \ell , m } a _ { \ell m } Y _ { \ell m } \bigl ( \hat { n } \bigr ) } } \end{array}$ , are described statistically by the angular power spectrum $C _ { \ell } ^ { T T } = \langle | a _ { \ell m } | ^ { 2 } \rangle$ . Its shape changes with angular scale, because diferent physical processes dominate at diferent scales (Hu & Dodelson 2002; Zaldarriaga et al. 1997). The positions of the acoustic peaks give the angular size of the sound horizon $\theta _ { s } ,$ , the relative height of the odd and even peaks is sensitive to $\omega _ { \mathrm { b } } .$ , the peak heights to $\omega _ { \mathrm { c d m } } .$ the spectral index $n _ { s }$ tilts the spectrum as a whole, and the overall amplitude is set by $A _ { s }$ . At the largest scales, cosmic variance limits how well parameters can be measured (Aghanim et al. 2020b).

Piras et al. (2025) trained a VAE on ΛCDM TT spectra at $\ell \geq 3 0$ To find the optimal number of latent variables, they increased the latent dimension step by step, and kept the smallest number for which the reconstruction was well within the Planck 1� error. With four latent variables the reconstruction error became comparable to the Planck error, while five were found to be enough. The latent variables were compared with the cosmological parameters only after training, using MI and latent traversals. They correspond to the overall amplitude, the positions of the acoustic peaks, the peak heights together with the spectral tilt, the even-odd modulation of the peaks, and the smoothing of the peaks by gravitational lensing.

The result has two implications. First, the TT spectrum at $\ell \geq 3 0$ has five independent degrees of freedom, one fewer than the six parameters of ΛCDM, and this number was found from the VAE analysis and not set in advance. Second, the missing degree of freedom is related to the degeneracy between $A _ { s }$ and � discussed in Section $3 . 3 . 2 \colon$ the amplitude latent variable follows the combination $A _ { s } e ^ { - 2 \tau }$ , and not $A _ { s }$ and � separately.

In Piras et al. (2025), the weight of the KL divergence term in the loss was manually tuned by the authors for each model to obtain disentangled latent variables, using the MI between the latent variables as the guide. In this work the weights of the reconstruction and KL divergence terms, and in the CA strategy also of the disentanglement term, are instead learned automatically in a single training run, which removes the need to tune these weights by hand.

<table><tr><td>Configuration Name</td><td>Reconstruction Terms  $\mathcal { T }$ </td></tr><tr><td>lowTTTEEE+full</td><td> $\mathcal { T } = \{ \mathcal { L } _ { \mathrm { l o w } \ell } , ~ \mathcal { L } _ { \mathrm { f u l l } \ell } \}$ </td></tr><tr><td>lowTTTEEE+high</td><td> $\mathcal { T } = \{ \mathcal { L } _ { \mathrm { l o w } \ell } , \ \mathcal { L } _ { \mathrm { h i g h } \ell } \}$ </td></tr><tr><td>lowTT-TEEE+high</td><td> $\mathcal { T } = \{ \mathcal { L } _ { \mathrm { l o w T T } \ell } , ~ \mathcal { L } _ { \mathrm { l o w T E E E } \ell } , ~ \mathcal { L } _ { \mathrm { h i g h } \ell } \}$ </td></tr></table>

Table 2. Summary of the three loss function configurations considered here.

## 3.3.2 Including Polarisation (TE, EE)

The temperature spectrum alone, however, has a built-in degeneracy. Thomson scattering of free electrons after reionisation reduces the power at small scales $( \ell ~ \gtrsim ~ 3 0 )$ by a factor of $e ^ { - 2 \tau }$ , where � is the optical depth. As a result, across the acoustic peaks and the damping tail, $\bar { C } _ { \ell } ^ { T T }$ depends on $A _ { s }$ and � essentially only through the combination $A _ { s } e ^ { - 2 \tau }$ (Aghanim et al. 2020b). Because the large scales are limited by cosmic variance, the temperature spectrum alone can constrain $A _ { s }$ and � separately only weakly. A VAE trained on the temperature spectrum at $\ell \geq 3 0$ can therefore resolve only this combination, and not the two parameters separately.

Polarisation at large scales carries independent information on �. Thomson scattering of the local quadrupole at reionisation produces a peak at low multipoles in the �� spectrum (the “reionisation bump”), and a corresponding feature in $T E$ . Its amplitude scales approximately as $C _ { \ell } ^ { E E } \overset { \cdot } { \propto } A _ { s } \tau ^ { 2 }$ for $\ell \lesssim 1 5 ,$ , so it constrains � separately from the high-ℓ amplitude $A _ { s } e ^ { - 2 \tau }$ . To allow the VAE to fit both the low-ℓ and the high-ℓ regimes, we split the reconstruction loss $\mathcal { L } _ { \mathrm { r e c o n } } \big ( D _ { \ell , \mathrm { t r u e } } , D _ { \ell , \mathrm { p r e d } } \big )$ into MSEs over separate multipole ranges: $\mathcal { L } _ { \mathrm { l o w } \ell }$ for $\ell \leq 1 4$ $\mathcal { L } _ { \mathrm { h i g h } \ell }$ for $\ell \geq 1 5 ,$ , and ${ \mathcal { L } } _ { \mathrm { f u l l } \ell }$ for the full range $2 \leq \ell \leq 2 5 0 0$ . The boundary at $\ell = 1 4$ is chosen to isolate the multipoles $\ell \lesssim 1 5$ at which the reionisation bump dominates, so that the low-ℓ polarisation, which carries the information on �, is not outweighed by the far larger number of high-ℓ multipoles in the loss. Unless stated otherwise, each of these is computed over the TT, TE and EE spectra.

The reconstruction loss is built from these terms in three diferent combinations, which define the three configurations studied in this work. All three use the same training spectra and difer only in how the reconstruction error is grouped and weighted. The total loss function has the general form,

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = \sum _ { i \in { \mathcal { T } } } w _ { i } { \mathcal { L } } _ { i } + \beta { \mathcal { D } } _ { K L } + w _ { d } { \mathcal { L } } _ { \mathrm { d i s } } ,\tag{2}
$$

where $\mathcal { T }$ is the set of reconstruction terms, which difers between configurations, and $w _ { i } , \beta$ and $w _ { d }$ are the loss weights, i.e. the hyperparameters that set the importance of each term. The three configurations were chosen to separate the roles of the temperature and polarisation data and of the low- and high-ℓ ranges: the first two difer in whether the high-ℓ error is weighted separately from, or together with, the low-ℓ error, and the third weights the low-ℓ temperature and polarisation errors separately. They are summarised in Table 2. Here $\mathcal { L } _ { \mathrm { l o w T T } \ell }$ is the MSE of the TT spectrum for $\ell \leq 1 4 .$ and $\mathcal { L } _ { \mathrm { l o w T E E E } \ell }$ is the MSE of the TE and EE spectra for $\ell \leq 1 4$ The third configuration therefore gives separate loss weights to the low-ℓ temperature and polarisation errors. In the lowTTTEEE+full configuration, the low-ℓ range appears in both $\mathcal { L } _ { \mathrm { l o w } \ell }$ and $\mathcal { L } _ { \mathrm { f u l l } \ell }$ . Additionally, ${ \mathcal { L } } _ { \mathrm { d i s } }$ is the diferentiable disentanglement loss function, described in Section 3.3.3.

![](images/a17e1d32c384c69dea2bd024387eb18bb7603523cd4e0cb79c56f9e1eeacfb2d.jpg)  
Figure 2. The end-to-end Multi-Task Learning VAE pipeline for CMB parametrisation, shown for the CA strategy. Top: The theoretical ΛCDM model is simulated via CLASS to generate target temperature and polarisation spectra. Middle (violet dashed box): These spectra are compressed by the encoder into an �-dimensional probabilistic latent space, before being reconstructed by the decoder. Bottom (red dashed box): The extraction of data to compute the individual loss terms during training. A trainable parameter $\sigma _ { i }$ is attached to each term. Backpropagation (solid red curves) updates the network parameters (�, �) and the $\sigma _ { i }$ together, so that the loss weights are set in a single training run. In SA strategy, $w _ { d }$ is fixed and the disentanglement loss term is not assigned any $\sigma _ { i }$

## 3.3.3 Diferentiable Disentanglement Loss Formulation

The reconstruction and KL terms alone do not tie the latent variables to physical quantities. A VAE trained only on them can describe the spectra equally well after its latent coordinates are mixed, and hence physical disentanglement cannot be determined without extra information. For our application, this matters for two reasons. First, in order to interpret the latent variables we need to know which physical efects each of them encodes, and to what extent. This can be assessed in a straightforward manner only if each latent variable can be compared with a physical parameter. Second, if the latent axes correspond to ΛCDM parameters, the latent posteriors can be read directly, rather than only through the decoder.

Therefore a penalty term is added to the loss function with two aims: each latent should depend on as few of the six cosmological parameters � as possible, and diferent latents should depend on different parameters. We do not demand a strict one-to-one mapping, because the spectra constrain some combinations of parameters, such as $A _ { s } e ^ { - 2 \tau }$ , better than the individual parameters, so one latent may reasonably follow two of them. The loss therefore allows each latent to carry up to two parameters (top-2 purity) and favours a strict one-to-one mapping only weakly. It is computed on each batch, from the latents and the cosmological parameters that generated the training samples, using only diferentiable operations, so that it can be minimised together with the reconstruction error. This term is the only place where these parameters enter the training, and it acts as a soft penalty: it does not set � and does not assign any latent variable to a given parameter. The details are described in Appendix A and Fig. A1. Note that the penalty uses the squared Pearson correlation, which measures linear dependence, whereas the normalised MI used later to evaluate the latent variables is not restricted to linear dependence.

## 3.4 Algorithm: Automated Hyperparameter Tuning

The training objective contains several terms: the reconstruction errors on diferent multipole ranges, the KL divergence and the disentanglement loss. The loss weights that set their relative importance must be chosen before training. A good choice, however, is not known in advance, and finding it by trial and error would need many training runs, which is impractical given the number of loss weights and the cost of each run. To avoid this, we adapt the multi-task learning framework of Cipolla et al. (2018), which dynamically tunes the loss weights during a single training run.

We treat each term $\mathcal { L } _ { i }$ as a separate task and attach to it a trainable parameter $\sigma _ { i } > 0$

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \sum _ { i } \left( \frac { \mathcal { L } _ { i } } { 2 \sigma _ { i } ^ { 2 } } + \log \sigma _ { i } \right) ,\tag{3}
$$

where the $\sigma _ { i }$ are optimised by gradient descent together with the network parameters. The loss weight of term � is $w _ { i } = 1 / ( 2 \sigma _ { i } ^ { 2 } )$ , so a larger $\sigma _ { i }$ means a smaller loss weight. In the notation of equation (2), the KL term has $\beta = 1 / ( 2 \sigma _ { \mathrm { K L } } ^ { 2 } )$ , and in the CA strategy of Section 3.4.1 the disentanglement term has $w _ { d } = 1 / ( 2 \sigma _ { d } ^ { 2 } )$ . The log � term prevents the trivial solution $\sigma _ { i }  \infty$ , which would set all loss weights to zero. For fixed network parameters, equation (3) is minimised at $\sigma _ { i } ^ { 2 } = \mathcal { L } _ { i } , \mathrm { i . e . } \ w _ { i } = 1 / ( 2 \mathcal { L } _ { i } )$ . The loss weights are therefore not free hyperparameters with a single optimal value. Rather they follow the current size of each term, so that a term whose loss stays large gets a smaller weight than one that decreases quickly. Consequently, the weight of a term grows as its loss falls. The factor $1 / ( 2 \sigma ^ { 2 } )$ follows from a Gaussian likelihood for regression tasks (Cipolla et al. 2018); applying it to the KL divergence and disentanglement terms is a heuristic extension. Fig. 2 shows the full algorithm.

![](images/8aad9df682e9bc7187e93a5e3ab05f63e7d5ab78802292b86f98b4c0100b9c30.jpg)  
Figure 3. Absolute reconstruction residuals $| \Delta D _ { \ell } | = | D _ { \ell } ^ { \mathrm { V A E } } - D _ { \ell } ^ { \mathrm { t r u e } } |$ of the Planck 2018 best-fit ΛCDM TT (top), TE (middle) and EE (bottom) spectra. Each pair of columns shows one configuration, with SA (left) and CA (right). The latent dimensions are � = 5 (blue), � = 6 (orange) and � = 7 (green).

## 3.4.1 Treatment ofthe Disentanglement Loss Weight

The disentanglement loss ${ \mathcal { L } } _ { \mathrm { d i s } }$ is the only term that uses information about the cosmological parameters, through the parameter values of the training samples. It acts as a soft penalty that encourages each latent variable to follow separate features that are independent of what other latent variables follow. It does not fix the number of latent variables, it does not assign a latent variable to a given parameter, and the ordering of the latent variables remains arbitrary (Section 4.2). All other terms of the loss carry no physical information.

This makes the choice of its loss weight, $w _ { d } ,$ less straightforward than for the other terms, since $w _ { d }$ sets how strongly the latent space is pulled towards the six cosmological parameters. To test the role of the disentanglement loss, the two strategies are implemented in equation (2) as follows:

(i) In the semi-automated (SA) strategy, �<sub>�</sub> and $\beta$ are learned in the same way, but $w _ { d }$ is held fixed at 5.0, and the disentanglement term is excluded from the uncertainty weighting.

(ii) In the completely automated (CA) strategy, every term, including the disentanglement loss, is given its own �<sub>�</sub> as in equation (3), so that $w _ { i } , \beta$ and $w _ { d }$ are all learned in a single training run.

The disentanglement loss therefore acts with a fixed strength in SA and with a learned strength in CA strategies.

## 4 RESULTS

Each VAE is trained for 200 epochs, by which stage the training losses have converged to their minimum values. Combining three configurations (lowTTTEEE+full, lowTTTEEE+high, and lowTT-TEEE+high), two strategies for the disentanglement loss weight (SA and CA), and three latent dimensions, $L = 5 , 6 , 7 ,$ gives a total of 18 models. For each run, the architecture, the training settings and, for SA, the value of $w _ { d }$ are fixed before training and are not tuned manually. The learned loss weights of every trained model are listed in Appendix B. Each model is evaluated in three ways: the accuracy of the reconstructed spectra (Section 4.1), the correspondence between latent variables and cosmological parameters (Section 4.2), and the fit of the latent-space model to Planck 2018 data (Section 4.3), where the models are compared using $\hat { \chi } ^ { 2 }$ , Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC) and Bayesian evidence.

## 4.1 Reconstruction

Fig. 3 shows the absolute residuals between the spectra reconstructed by each of the 18 models and the Planck 2018 best-fit ΛCDM spectra. The trained VAE models reconstruct the CMB power spectra with absolute reconstruction residual $( | \Delta D _ { \ell } | )$ values well within the

Planck 2018 uncertainties. Three features are observed across the models:

(i) The residuals for $L \ = \ 5$ (blue) lie above those for $L \ = \ 6$ (orange) and $L = 7$ (green) in most of the cases, so the reconstruction is generally less accurate with $L = 5$ than with $L \geq 6 .$

(ii) The residuals for $L = 6$ and $L = 7$ are similar, so the seventh latent variable does not improve the reconstruction significantly, despite the extra degree of freedom and the higher model complexity.

(iii) The residuals are largest at the peaks and troughs of the spectra.

## 4.2 Interpreting the Latent Space

We use the normalised MI to determine how much information each latent variable shares with each ΛCDM parameter. MI is a statistical measure of the information shared between two variables (Vergara & Estévez 2013). It is not restricted to linear dependence, which suits latent variables that may depend non-linearly on the parameters, and has been used to interpret latent spaces in cosmology (Lucie-Smith et al. 2022, 2024; Guo et al. 2024). The raw MI values are computed with the GMM-MI package (Piras et al. 2023), which fits a Gaussian mixture model to the joint distribution of the two variables, as also done by Piras et al. (2025).

Raw MI values are in units of nats and, unlike a correlation coefficient, have no fixed upper limit. We therefore divide each raw MI value by the maximum MI that can be reached for the given sample size $( N = 2 0 0$ samples in our case), which bounds the normalised MI between 0 and 1. The same constant is applied to every entry of every model, so the normalisation rescales all values together and does not change their ordering. Our values are therefore not on the same scale as raw MI values in nats, and they should not be compared directly with those of Piras et al. (2025). Larger values mean that a latent variable shares more information with the corresponding cosmological parameter(s). Hereafter, MI refers to the normalised MI.

To summarise the MI matrix of each model with a single number, we define a disentanglement score. It measures both the strength and the exclusivity of the association between latent variables and cosmological parameters, and it penalises information spread over several latent variables, and cosmological parameters left without a latent variable. For a normalised MI matrix $M \in \mathbb { R } ^ { N \times L }$ , with � parameters and � latent variables, each parameter is first matched to one latent variable with the Hungarian linear sum assignment algorithm (Kuhn 1955), which maximises the total MI of the matched pairs. A matched pair is kept only if its MI is at least a threshold $M _ { \mathrm { m i n } } = 0 . 1 0$ chosen to separate clear associations from weak ones, and pairs below it are discarded. The parameter coverage $C = N _ { \mathrm { a s s i g n e d } } / N$ is the fraction of parameters that are matched to a latent variable. For $L = 5$ there are fewer latent variables than parameters, so $C \leq 5 / 6 ,$ which lowers the maximum possible score and should be kept in mind when comparing scores across �. The dependence of the score on the choice $M _ { \mathrm { m i n } } = 0 . 1 0$ is demonstrated in Appendix C.

For each retained pair $( i , j )$ of parameter � and latent variable $j ,$ the quality of the association is evaluated through three quantities: purity, row redundancy and column redundancy. The purity,

$$
P _ { i j } = \frac { M _ { i , j } } { \sum _ { k } M _ { k , j } } ,\tag{4}
$$

is the fraction of the total MI of latent variable � that is due to parameter �. Information of parameter � spread over other latent variables

is penalised by a row factor,

$$
F _ { i } ^ { \mathrm { r o w } } = \frac { 1 } { 1 + \sum _ { k \neq j } M _ { i , k } } ,\tag{5}
$$

and a latent variable � that encodes other parameters as well is penalised by a column factor,

$$
F _ { j } ^ { \mathrm { c o l } } = \frac { 1 } { 1 + \sum _ { \boldsymbol { k } \neq i } M _ { \boldsymbol { k } , j } } .\tag{6}
$$

The product of these terms gives the score of a pair, $S _ { i j } = P _ { i j } \cdot F _ { i } ^ { \mathrm { r o w } }$ $F _ { j } ^ { \mathrm { c o l } }$ . The scores of the retained pairs are averaged and multiplied by the coverage,

$$
S _ { \mathrm { t o t a l } } = C \cdot \frac { 1 } { N _ { \mathrm { a s s i g n e d } } } \sum _ { ( i , j ) } S _ { i j } .\tag{7}
$$

The normalised MI matrices of all 18 models are shown in Fig. 4. In each matrix, the entry in a row and a column is the normalised MI between the parameter of that row and the latent variable of that column. A large MI in an outlined cell does not by itself imply a clean mapping, since the same parameter may also be spread over other latent variables (the rest of its row), or the same latent variable may mix several parameters (the rest of its column). These are the two efects penalised by the row and column factors in equations (5) and (6), so that a clean one-to-one encoding has a large MI in the outlined cell and small values in the rest of its row and column. Hence, the score � above each panel of Fig. 4 combines both aspects.

We note three patterns in Fig. 4. First, only three of the 18 models assign all six parameters to separate latent variables cleanly, that is, with the normalised MI of every matched pair above the threshold: SA lowTTTEEE+full with $L = 6$ and 7, and SA lowTTTEEE+high with $L = 7 .$ . These are also the three models with the highest scores. Second, in every model the latent variable matched to � shares almost no information with $A _ { s } ,$ and the one matched to $A _ { s }$ shares almost none with � (normalised MI of at most 0.03, see Section 5.2). Third, the SA models have a mean disentanglement score of 0.58 and the CA models of 0.40. The gap is not the same everywhere: for lowTT-TEEE+high the two strategies score almost the same at every �, whereas for lowTTTEEE+full, and for lowTTTEEE+high at $L \ = \ 7$ , it is large (Table 3). SA and CA difer only in the loss weight of the disentanglement loss, which is fixed at $w _ { d } = 5$ in SA and learned in CA, where it takes values of $0 . 4 5 \substack { - 0 . 8 1 }$ (Table B1). The gap in the mean disentanglement scores shows the efect of a stronger disentanglement loss weight, which pulls the latent space more towards the physical parameters. These measures indicate how interpretable the latent variables are, but not how well the models fit the real data. Fits to Planck data are hence performed next.

## 4.3 Latent Space Posteriors

We sample the posterior distribution of the latent variables � of each model with MCMC, using the emcee sampler (Foreman-Mackey et al. 2013) with 64 walkers. At each step the decoder gives the spectra $( \hat { D } _ { \ell } ^ { T T } , \hat { D } _ { \ell } ^ { T E } , \hat { D } _ { \ell } ^ { E E } )$ , which is compared with Planck 2018 data using the Plik\_lite likelihood (Prince & Dunkley 2019; Aghanim et al. 2020a). The VAE spectra span $2 \leq \ell \leq 2 5 0 0$ in all three spectra, but the comparison with the data is made over the multipole range that Plik\_lite covers within this span: $2 \leq \ell \leq 2 5 0 0$ for TT, up to the last bin that lies within the VAE range, and $3 0 \leq \ell \leq$ 1996 for TE and EE. A sample of the posterior distributions of the latent variables is shown in Appendix D.

To compare the models fairly, we use the goodness of fit, two model selection criteria and Bayesian evidence, which account for the

CA, L = 6

$$
S = 0 . 6 2
$$

$$
\omega _ { c }
$$

$$
S = 0 . 7 4
$$

<table><tr><td rowspan=1 colspan=1>0.36</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.03</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.16</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.06</td></tr><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.02</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.29</td><td rowspan=1 colspan=1>0.01</td></tr></table>

$$
n _ { s }
$$

$$
\ln { 1 0 ^ { 1 0 } A _ { s } }
$$

$$
S = 0 . 3 9
$$

$$
S = 0 . 4 0
$$

$S = 0 . 5 2$ SA, L = 7
<table><tr><td rowspan=1 colspan=1> $\omega _ { b }$ </td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td></tr><tr><td rowspan=1 colspan=1> $\omega _ { c }$ </td><td rowspan=1 colspan=1>0.22</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>h</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.19</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.01</td></tr><tr><td rowspan=1 colspan=1>τ</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.21</td><td rowspan=1 colspan=1>10.06</td><td rowspan=1 colspan=1>0.01</td></tr><tr><td rowspan=1 colspan=1> $n _ { s }$ </td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>.090</td><td rowspan=1 colspan=1>.06</td></tr><tr><td rowspan=1 colspan=1> $\ln { 1 0 ^ { 1 0 } A _ { s } }$ </td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.030</td><td rowspan=1 colspan=1>.07</td></tr></table>

$$
S = 0 . 1 4
$$

<table><tr><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1> $0 . 0 0 ~</td><td rowspan=1 colspan=1>0 . 0 0$ </td></tr><tr><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td></tr><tr><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.29</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.05</td></tr><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.040</td><td rowspan=1 colspan=1>.09</td></tr></table>

<table><tr><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td></tr><tr><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.15</td><td rowspan=1 colspan=1>10.01</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.03</td></tr><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.08</td></tr><tr><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.02</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.00</td></tr></table>

SA, L = 6

$$
S = 0 . 5 1
$$

$$
\omega _ { b }
$$

$$
\omega _ { c }
$$

<table><tr><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.27</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td></tr><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.21</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.23</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.07</td></tr><tr><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.08</td></tr></table>

<table><tr><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.03</td></tr><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>10.03</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.03</td></tr><tr><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.16</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.06</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.24</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.02</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.16</td><td rowspan=1 colspan=1>0.02</td></tr></table>

$$
\ln { 1 0 ^ { 1 0 } A _ { s } }
$$

$$
S = 0 . 4 9
$$

$$
S = 0 . 4 3
$$

SA, L = 5  
CA, L = 7
<table><tr><td rowspan=1 colspan=1> $\omega _ { b }$ </td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td></tr><tr><td rowspan=1 colspan=1> $\omega _ { c }$ </td><td rowspan=1 colspan=1>0.22</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>h0</td><td rowspan=1 colspan=1>.03</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td></tr><tr><td rowspan=1 colspan=1>τ</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.48</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1> $n _ { s }$ </td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>0.04</td></tr><tr><td rowspan=1 colspan=1> $\ln { 1 0 ^ { 1 0 } A _ { s } }$ </td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.060</td><td rowspan=1 colspan=1>.06</td></tr></table>

<table><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.07</td></tr><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>0.08</td></tr></table>

$$
S = 0 . 4 1
$$

<table><tr><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.05</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.077</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.007</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.06</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.03</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.02</td></tr><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.02</td></tr></table>

$$
S = 0 . 4 5
$$

$$
\omega _ { b }
$$

$$
\omega _ { c }
$$

$$
n _ { s }
$$

$$
\ln { 1 0 ^ { 1 0 } A _ { s } }
$$

$$
\mathbf { C A } , L = 5
$$

$$
\omega _ { b }
$$

$$
 0 . 0 5 | ; 0 . 0 8 \}
$$

$$
S = 0 . 3 8
$$

$$
\omega _ { c }
$$

$$
\mathbf { C A } , L = \mathbf { 6 }
$$

$$
\begin{array} { r } { \boxed { \mathbf { 0 . 1 0 } } \boxed { 0 . 0 6 } } \end{array}
$$

$$
0 . 0 6 \boxed { 0 . 1 1 }
$$

$$
S = 0 . 4 5
$$

$$
n _ { s }
$$

$$
1 0 ^ { 1 0 } A _ { s }
$$

<table><tr><td rowspan=1 colspan=1>0.16</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.13</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>0.12</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.12</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.01</td></tr><tr><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.05</td></tr><tr><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.10</td></tr></table>

<table><tr><td rowspan=1 colspan=1>0.34</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.13</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.03</td></tr><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.12</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.05</td></tr><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.24</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td></tr><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.02</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.065</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.01</td></tr><tr><td rowspan=1 colspan=1>CA, L</td><td rowspan=1 colspan=1>= 7</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $S =</td><td rowspan=1 colspan=1>0 . 4 7$ </td></tr><tr><td rowspan=1 colspan=1>0.31</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td></tr><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>0.12</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td></tr><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.30</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.04</td></tr><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.02</td></tr><tr><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.01</td></tr></table>

Figure 4. Normalised MI between each ΛCDM parameter (rows) and each latent variable (columns) for all 18 trained models. The three blocks are the configurations, each shown for the SA and CA strategies, and the columns of panels correspond to $L \ = \ 5 , 6 , 7 .$ The number above each panel is the disentanglement score �. Solid outline: parameter-latent pair assigned by the matching, with normalised MI of at least 0.10; dashed outline: assigned pair below the threshold of 0.10, counted as not captured. All values are on a common colour scale that follows a power law with exponent 0.6 so that small values can be told apart.

<table><tr><td>Configuration</td><td>Strategy</td><td>L</td><td> $\hat { \chi } ^ { 2 }$ </td><td>ΔAIC</td><td>ΔBIC</td><td>lnB</td><td>Score</td></tr><tr><td rowspan="6">lowTTTEEE+full</td><td rowspan="3">SA</td><td>5</td><td>0.9630</td><td>1.1</td><td>0.0</td><td>0.3</td><td>0.619</td></tr><tr><td>6</td><td>0.9607</td><td>0.7</td><td>4.0</td><td>0.0</td><td>0.737</td></tr><tr><td>7</td><td>0.9578</td><td>0.0</td><td>7.8</td><td>0.4</td><td>0.781</td></tr><tr><td rowspan="3">CA</td><td>5</td><td>0.9575</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.388</td></tr><tr><td>6</td><td>0.9603</td><td>2.8</td><td>7.2</td><td>3.5</td><td>0.144</td></tr><tr><td>7</td><td>0.9621</td><td>4.9</td><td>13.7</td><td>5.5</td><td>0.399</td></tr><tr><td rowspan="6">lowTTTEEE+high</td><td rowspan="3">SA</td><td>5</td><td>0.9782</td><td>11.0</td><td>6.6</td><td>10.6</td><td>0.510</td></tr><tr><td>6</td><td>0.9584</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.519</td></tr><tr><td>7</td><td>0.9662</td><td>5.8</td><td>10.2</td><td>5.2</td><td>0.766</td></tr><tr><td rowspan="3">CA</td><td>5</td><td>0.9687</td><td>7.6</td><td>3.1</td><td>7.5</td><td>0.492</td></tr><tr><td>6</td><td>0.9546</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.433</td></tr><tr><td>7</td><td>0.9664</td><td>8.2</td><td>12.6</td><td>8.2</td><td>0.423</td></tr><tr><td rowspan="6">lowTT-TEEE+high</td><td>SA</td><td>5</td><td>0.9881</td><td>14.7</td><td>8.4</td><td>14.1</td><td>0.408</td></tr><tr><td></td><td>6</td><td>0.9653</td><td>1.8</td><td>0.0</td><td>1.0</td><td>0.451</td></tr><tr><td></td><td>7</td><td>0.9606</td><td>0.0</td><td>2.6</td><td>0.0</td><td>0.458</td></tr><tr><td rowspan="3">CA</td><td>5</td><td>0.9713</td><td>7.0</td><td>2.6</td><td>6.3</td><td>0.384</td></tr><tr><td>6</td><td>0.9581</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.454</td></tr><tr><td>7</td><td>0.9614</td><td>3.0</td><td>7.4</td><td>3.2</td><td>0.471</td></tr></table>

Table 3. Goodness of fit and model comparison for all 18 models, using the MCMC chains and the Plik\_lite likelihood. ΔAIC, ΔBIC and ln � are measured from the best � within each configuration and strategy, and the lowest values are highlighted in bold. Score is the disentanglement score $S _ { \mathrm { t o t a l } }$ of equation (7), computed from the normalised MI matrices of Fig. 4.

<table><tr><td>Configuration</td><td> $L = 5$ </td><td> $L = 6$ </td><td> $L = 7$ </td></tr><tr><td>lowTTTEEE+full</td><td> $+ 4 . 3$ </td><td> $+ 0 . 6$ </td><td>-1.0</td></tr><tr><td>lowTTTEEE+high</td><td>+5.6</td><td>+2.5</td><td>-0.5</td></tr><tr><td>lowTT-TEEE+high</td><td>+10.2</td><td>+3.4</td><td>-0.8</td></tr></table>

Table 4. Evidence comparison of the two disentanglement strategies at fixed latent dimension: ln $B _ { \mathrm { C A , S A } } = \ln Z _ { \mathrm { C A } } - \ln Z _ { \mathrm { S A } }$ for the same configuration. Positive values favour CA and negative values favour SA.

number of latent variables in diferent ways. The first measure is the reduced $\chi ^ { 2 } , \hat { \chi } ^ { 2 } = \chi _ { \mathrm { m i n } } ^ { 2 } / ( n { - } L )$ , where $\chi _ { \mathrm { m i n } } ^ { 2 }$ is the minimum $\chi ^ { 2 }$ found in the chains, $n = 6 1 4$ is the number of binned data points used in the Plik\_lite likelihood (Prince & Dunkley 2019), and all � latent variables are counted as free parameters. The second and third are the model selection criteria $\mathrm { A I C } = \chi _ { \mathrm { m i n } } ^ { 2 } + 2 L$ and $\mathrm { B I C } = \chi _ { \mathrm { m i n } } ^ { 2 } + L$ ln �, which add a fixed penalty for each latent variable.

The Bayesian evidence (�) is estimated from the MCMC samples using the MCEvidence �-nearest-neighbour estimator (Heavens et al. 2017). We report two complementary Bayes factors. First, for a fixed strategy and configuration, ln $B = \ln Z _ { \mathrm { m a x } } - \ln Z$ measures how much a model is disfavoured relative to the model with the highest evidence among $L = 5 , 6 , 7 .$ . Second, ln $B _ { \mathrm { C A , S A } } = \ln Z _ { \mathrm { C A } } - \ln Z _ { \mathrm { S A } }$ compares CA with SA at the same � and configuration. The strength of the preference is interpreted with the Jefreys scale (Jefreys 1939), in which ln � between 2.5 and 5 is moderate and above 5 is strong. For the evidence computation we use the same prior for all models: uniform on [−10, 10] for every latent variable. No latent variable is given any prior information by any cosmological parameter, so the fit to the Planck data is carried out entirely in the free latent space.

Table 3 compares � = 5, 6 and 7 for each configuration and strategy. No single � is preferred in all six cases, but there is a clear trend. Going from $L \ = \ 5$ to � = 6 improves the fit substantially in four of the six cases, and � = 5 is strongly disfavoured by the evidence (ln $B > 5 )$ for lowTTTEEE+high and lowTT-TEEE+high. In the two lowTTTEEE+full cases the sixth latent variable does not improve the fit: for SA, � = 5 and $L \ = \ 6$ are indistinguishable (ln $B = 0 . 3$ and 0.0), and for CA, � = 5 is the best, with $L = 6$ moderately disfavoured (ln $B = 3 . 5 )$ . Going from $L = 6$ to $L = 7$ gives no significant gain in any case: the BIC never prefers � = 7, with $L \ = \ 7$ having the highest evidence only for SA lowTT-TEEE+high, where its advantage over � = 6 is ln $B = 1 . 0 ,$ which is inconclusive. Overall, $L = 6$ is the best, or within ln � ≃ 1 of the best, in five of the six cases. BIC and the evidence select the same � in four cases (CA lowTTTEEE+full, SA lowTTTEEE+high, CA lowTTTEEE+high and CA lowTT-TEEE+high), including all three CA cases. For SA they difer for lowTTTEEE+full (BIC selects $L = 5$ and the evidence $L \ : = \ : 6 )$ and for lowTT-TEEE+high (BIC selects � = 6 and the evidence $L = 7 )$

At fixed � (Table 4), the evidence favours CA over SA at $L = 5$ (ln $B _ { \mathrm { C A , S A } } = 4 . 3 – 1 0 . 2 )$ and at $L = 6 ( 0 . 6 \ – 3 . 4 .$ , from inconclusive to moderate), and favours SA over CA at $L = 7 \ : ( - 1 . 0 \ : \mathrm { t o } \ : - 0 . 5 $ , weak or inconclusive). Since the $L = 7$ models are themselves disfavoured relative to $L = 6$ in most cases, we regard the comparison between � = 5 and � = 6 as more relevant, in which CA is favoured.

## 5 DISCUSSION

The main aim of this work is to find out how many independent degrees of freedom the CMB power spectra (TT, TE and EE) contain, and what they correspond to, without building a cosmological model a priori into the network. Standard analyses sample directly in a predefined set of parameters, whereas here the latent variables are free and are compared with the ΛCDM parameters only after training is completed. We discuss three aspects of the results: the number of latent variables the spectra need, the separation of � from $A _ { s }$ , and the trade-of between an interpretable latent space and the quality of the fit to real data.

## 5.1 The Intrinsic Dimensionality of the CMB

We train models with three latent dimensions, $L = \{ 5 , 6 , 7 \}$ . This range is chosen around the six parameters of ΛCDM, since searching in this neighbourhood could help locate the number of independent degrees of freedom even in this model-independent approach. In principle, the same exercise can be repeated over any range of $L ,$ and the minimum number of latent variables can be found through statistical measures such as those used here.

A comparison of the fits for $L = 5 , 6$ and 7 shows how many latent variables the spectra need. In four of the six cases (SA and CA lowTTTEEE+high, and SA and CA lowTT-TEEE+high), $L = 5$ is clearly disfavoured by the evidence and $L = 7$ gives no significant gain over $L = 6$ . In these cases, six latent variables are needed to fit the data, and seven give no further improvement. This is the number of free parameters of ΛCDM, and one more than the five found for data at $\ell \geq 3 0$ (Piras et al. 2025; Sun et al. 2026). In the two lowTTTEEE+full cases the result is diferent: for SA, $L = 5$ and $L = 6$ are indistinguishable, and for $\mathrm { C A } , L = 5$ is the best. A possible explanation for this preference of $L = 5$ in this case might be the use of lowTTTEEE twice in the loss function (once in the full spectrum term, once separately), which gives extra weight to the low-ℓ data. This gives a hint of how crucial a proper choice of the loss function could be in order to arrive at scientifically valid results. However, this aspect requires further investigations with a series of possible loss functions (both good and bad) and hence we refrain from making any strong comments on this at this point. The CA lowTTTEEE+full model with $L = 6$ is also the least interpretable of all the models: only � passes the MI threshold, and its score is 0.14 (Fig. 4). Once $L \ : = \ : 6$ is reached, the seventh latent variable gives no significant improvement, and the evidence decreases for CA lowTTTEEE+full, SA and CA lowTTTEEE+high, and CA lowTT-TEEE+high. For SA lowTTTEEE+full, $L = 7$ lowers $\hat { \chi } ^ { 2 }$ slightly and its disentanglement score is higher, while its evidence is not higher (ln $B = 0 . 4$ against 0.0 at $L = 6 )$ . Only for SA lowTT-TEEE+high is the evidence higher at $L = 7 ,$ but the diference is inconclusive (ln $B = 1 . 0 )$

The networks are trained on the full spectra, including the low-ℓ polarisation, and the Planck comparison tests how well the resulting latent spaces reproduce the observed spectra. The Plik\_lite likelihood includes the low-ℓ temperature bins, whose amplitude is not attenuated by reionisation and can be compared with the high-ℓ amplitude. So the data carries some information on the split between $A _ { s }$ and �. A natural interpretation of the step from $L = 5$ to $L = 6$ is that the low-ℓ polarisation in the training spectra provides information on � that is independent of $A _ { s }$ , so these spectra contain one more independent degree of freedom than at $\ell \geq 3 0$ alone. This agrees with the MI results of Section 5.2. Since the number of degrees of freedom is not imposed anywhere in the network, the finding that six are needed to fit these spectra is a direct result from the analysis. Whether, and to what extent, the six latent variables correspond to the six ΛCDM parameters has to be judged from the MI matrices (Section 5.3).

With five latent variables, one degree of freedom must be merged or dropped, which degrades the fit (Fig. 3). This costs fit for lowTT-TEEE+high and lowTT-TEEE+high, where $L = 5$ is strongly disfavoured by the evidence, but not for lowTTTEEE+full, where $L = 5$ is competitive with $L \ = \ 6 .$ . Going to $L \ = \ 7$ brings no meaningful gain: the BIC never prefers $L = 7 .$ , and the evidence shows no significant improvement. The four measures $\hat { \big ( } \hat { \chi } ^ { 2 }$ , AIC, BIC and the Bayesian evidence) therefore point in the same direction: they favour $L = 6$ over $L = 5$ in four of the six cases and find no significant gain beyond $L = 6$ . They select the same � in all three CA configurations, whereas for SA the fits at neighbouring � are so close that the preferred � depends on the measure considered (Section 5.4). The Bayesian evidence penalises dimensions that the data do not require, but it depends on the prior of the latent variables, and the latent scale is not the same in diferent networks, so we do not read diferences of order one in ln � as significant. Overall, $L = 6$ is the most consistently favoured value, being the best or within ln $B \simeq 1$ of the best in five of the six cases.

## 5.2 Breaking of the $A _ { \mathrm { s } } { - } \tau$ Degeneracy

A key result of this work is that the latent variables for � and $A _ { s }$ share almost no information in any of the 18 models: the MI between them is at most 0.03. In standard model-dependent analyses, $A _ { s }$ and � are degenerate in the high-ℓ TT spectrum, which depends on them only through the combination $A _ { s } e ^ { - 2 \tau }$ (Section 3.3.1). This degeneracy is broken by adding polarisation data (Aghanim et al. 2020b; Zaldarriaga 1997), since � is separately constrained by the reionisation bump in the EE spectrum (Section 3.3.2). Our models include $\ell \geq 2$ and polarisation, in both the training spectra and the reconstruction loss, and the separation found in their latent space is consistent with the one that including polarisation data provides in standard analyses. It is notable that the separation of these two physical parameters appears as a result of a deep-learning network with neither a cosmological model built into it, nor any of the latent variables assigned to any parameter in advance.

## 5.3 What the Latent Variables Capture

The strength of the mapping difers strongly between parameters (Fig. 4). In most models � is matched to the latent variable with the largest MI, and $\tau , \omega _ { \mathrm { c d m } }$ and ℎ typically follow in that order, whereas $n _ { s }$ and, to a lesser extent, $A _ { s }$ most often fall at or below the threshold of 0.10 used for the disentanglement score. This pattern follows how distinctive the imprint of a parameter on the spectra is.

The shape of $C _ { \ell } ^ { T T }$ changes with angular scale, because diferent physical processes dominate at diferent scales (Hu & Dodelson 2002; Zaldarriaga et al. 1997). At the largest scales $( \ell \lesssim 3 0 )$ , the spectrum is nearly flat, i.e. the Sachs–Wolfe plateau (Sachs & Wolfe 1967). It is set by the primordial gravitational potential wells, whose amplitude is proportional to $A _ { s }$ , together with the late-time decay of these potentials driven by dark energy. Cosmic variance limits how well parameters can be measured at these large scales (Aghanim et al. 2020b). At intermediate scales $( 3 0 \lesssim \ell \lesssim 1 0 0 0 )$ , the spectrum shows acoustic oscillations of the photon-baryon fluid. Here, the positions of the peaks give the angular size of the sound horizon $\theta _ { s } ,$ , which is set by the Hubble constant ℎ together with the matter densities. The relative height of the odd and even peaks is sensitive to $\omega _ { \mathrm { b } }$ , and the peak heights are sensitive to $\omega _ { \mathrm { c d m } }$ . At the smallest scales $( \ell \gtrsim 1 0 0 0 )$ photon difusion during recombination washes out the fluctuations, which gives the exponential Silk damping tail (Silk 1968). The optical depth has its own signature in the form of the reionisation bump in the low-ℓ EE and TE spectra (Section 3.3.2), and it also lowers the high-ℓ amplitude by $e ^ { - 2 \tau }$ . The baryon and cold dark matter densities and ℎ therefore produce large changes in the spectra in the form of acoustic features, which a latent variable can pick up easily. In contrast, $A _ { s }$ and $n _ { s }$ change the spectra smoothly, as an overall amplitude and a tilt without acoustic features. A latent space that encodes the shape of the spectra therefore has little reason to tie itself to either of them prominently.

This ordering agrees with the temperature-only analysis of Piras et al. (2025), where $\omega _ { \mathrm { b } }$ also has the cleanest latent variable, which they identify with the even–odd modulation of the acoustic peaks. Here, only the ordering (instead of specific MI values) is compared since the MI normalisations difer (Section 4.2). Moreover, in Piras et al. (2025) � is only weakly mapped, since in TT alone it enters mainly through $A _ { s } e ^ { - 2 \tau }$ . In our models it is among the best captured parameters, which is expected, since the large-scale polarisation gives it a signature of its own. Their $n _ { s }$ is mapped more clearly in a latent variable mixed with $\omega _ { \mathrm { c d m } }$ , whereas in our CA models, $n _ { s }$ is the weakest captured parameter. In some of our SA models, however, it is captured about as well as $\omega _ { \mathrm { c d m } }$ and ℎ.

At $\ell \geq 3 0$ the spectra depend on $A _ { s }$ and � only through $A _ { s } e ^ { - 2 \tau }$ and five latent variables sufice, for TT alone (Piras et al. 2025), and for binned TT, TE and EE together (Sun et al. 2026). Adding polarisation at high ℓ is hence not seen to prefer an additional independent degree of freedom. The additional degree of freedom in our models (which includes spectra down to $\ell = 2 )$ is associated with the largescale polarisation, which constrains � separately from the amplitude (Section 5.2).

The largest normalised MI cross-term involves $A _ { s }$ and $n _ { s } .$ . In several models, in particular in the CA models and in SA lowTT-TEEE+high, the largest entry in the $A _ { s }$ row outside the matched cell lies in the $n _ { s }$ latent variable, and the largest entry in the $n _ { s }$ row lies in the $A _ { s }$ latent variable. In SA lowTTTEEE+full this cross-term is almost absent. Where it appears the entries are small, at most 0.08, below the threshold of 0.10. Both parameters describe the primordial spectrum $( A _ { s }$ its amplitude and $n _ { s }$ its tilt) and over a limited range of ℓ a change of tilt can be partly traded against a change of amplitude. A latent variable may therefore encode a combination of the two rather than each one separately. A weaker version of the same tendency is seen for $\omega _ { \mathrm { c d m } }$ and $h ,$ which lie along the direction of the geometric degeneracy between the matter density and ℎ at a fixed angular scale of the acoustic peaks. In the temperature-only analysis of Piras et al. (2025), the latent variable that follows the peak positions has higher MI with the angular size of the sound horizon $\theta _ { s }$ than with ℎ. Since $\theta _ { s }$ is set by ℎ and the matter densities together, a latent variable that follows $\theta _ { s }$ would show only a partial mapping to ℎ. The cross-terms hence point to two pairs of parameters whose efects on the spectra partly overlap, $A _ { s }$ with $n _ { s }$ and, more weakly, $\omega _ { \mathrm { c d m } }$ with ℎ.

## 5.4 Semi-Automated vs Completely Automated Disentanglement

SA and CA difer only in the loss weight of the disentanglement loss, which is fixed at $w _ { d } ~ = ~ 5 . 0$ in SA and learned in CA. The learned values lie between 0.45 and 0.81 (Table B1), which are ∼ 5 to ∼ 10 times smaller than the fixed value. The variation of $w _ { d }$ with $L ,$ and with the configuration, is small compared to this diference. The learned weight is therefore always well below the value fixed by hand, so CA applies a much weaker disentanglement loss. We compare the two strategies in three respects.

• Fit to the data. The evidence favours CA over SA at $L = 5$ and $L = 6$ and SA over CA at � = 7 (Table 4), so the advantage of CA decreases with � and is not present at $L = 7$ . When each strategy is taken at its own best �, ln $B _ { \mathrm { C A , S A } } \simeq 4 . 0 , 2 . 5$ and 2.4 for lowTT-TEEE+full, lowTTTEEE+high and lowTT-TEEE+high respectively. ${ \mathrm { S o } } ,$ CA strategy is still preferred. Because the evidence is compared between diferent networks with the same latent prior, we regard these diferences as indicative.

• Interpretability. In SA strategy the disentanglement score increases with �. The diference in the mean score is about 0.09, 0.23 and 0.24 at $L = 5 ,$ 6 and 7. A possible reason is that a large fixed loss weight costs more fit when there are few latent variables, whereas with more of them the same constraint can be met at little cost. The gain in disentanglement is not tied to a better fit. For SA lowTT-TEEE+high, the best fit is at $L \ : = \ : 6$ (score 0.52), but the highest score is at $L = 7 \ ( 0 . 7 7 )$ . A good fit therefore neither requires nor excludes a clean mapping of the latent variables onto the cosmological parameters. Moreover, choosing � by the fit alone would not select the most interpretable latent space. With the disentanglement loss, a one-to-one assignment of all six parameters is reached in three of our SA models, all with $L \geq 6 .$ . This is diferent from the more distributed encoding reported for a conditional VAE by Sun et al. (2026). However, the measures difer (kernel alignment there, normalised MI here), so the comparison here is qualitative.

• Choice of�. For CA, the four measures $\hat { \big ( } \hat { \chi } ^ { 2 }$ , AIC, BIC, B) agree on � in all three configurations, whereas for SA they difer in two of the three. For SA the fits at neighbouring � are close, so the preferred � depends on the measure.

SA and CA are therefore two points on a trade-of between interpretability and fit to the data. SA gives clearer disentanglement, especially at larger �. CA needs no choice of $w _ { d } ,$ , has the higher evidence at $L = 5$ and 6, and gives a consistent preference for �. Since CA prefers a weaker disentanglement loss, its latent variables are spread more widely over the parameters, although they fit the Planck data better. This suggests that a latent space less constrained to follow the standard cosmological parameters can fit the Planck data at least as well, and in our comparison better, than one pulled towards them. A possible reading is that the standard parameters, which are chosen for ease of physical interpretation, may not be the combinations that the CMB data constrain most directly.

## 6 CONCLUSIONS AND OUTLOOK

We asked how many latent variables are required to fit the CMB temperature and polarisation spectra when a neural network is given no cosmological model to start from, and to what extent they map onto the six ΛCDM parameters. Using our OCCAMS (Optimal Components of CMB with Autoencoders and Model Selection) code, a total of 18 VAEs were trained on CLASS spectra of TT, TE and EE for $2 \leq \ell \leq 2 5 0 0$ , for three configurations of the reconstruction loss, two ways of treating the weight of the disentanglement loss (semi-automated, SA, and completely automated, CA, strategies), and three diferent latent dimensions $L = 5 , 6 , 7$ , thereby making our investigation comprehensive. The three configurations were chosen to separate the roles of the temperature and polarisation data and of the low- and high-ℓ ranges. The loss weights of the reconstruction and KL divergence terms, and in CA also of the disentanglement term, were set by multi-task uncertainty weighting within a single training run, without a grid search. The latent dimensions have been considered with an expectation that a well-designed VAE model would lead to the number of minimum dimensions close to the physical ones. However, the process can be repeated with any arbitrary number as a viable test case, in order to arrive at the minimum number of dimensions. All models fit the Planck data well, with $\hat { \chi } ^ { 2 }$ between 0.955 and 0.988. The models were compared using $\hat { \chi } ^ { 2 }$ , AIC, BIC and Bayesian evidence, and the latent variables were interpreted through their normalised mutual information (MI) with the six ΛCDM parameters. Our main results are the following.

(i) The latent variables capture best the parameters that leave distinctive features in the spectra: $\omega _ { \mathrm { b } } ,$ , then $\tau , \omega _ { \mathrm { c d m } }$ and ℎ. The parameters $n _ { s }$ and $A _ { s }$ are captured least by the pipeline, and the small normalised MI cross-terms (smeared) between them suggest that their efects partly overlap.

(ii) The latent variables for � and $A _ { s }$ share almost no information in any of the 18 models (normalised MI of at most 0.03), despite no external bias forced on any latent parameter explicitly a priori. This is consistent with the large-scale polarisation separating the two parameters, as it does in standard analyses, and as TT alone at $\ell \geq 3 0$ cannot.

(iii) In four of the six cases (SA and CA, for lowTTTEEE+high and lowTT-TEEE+high) the fit improves from $L = 5 { \mathrm { ~ t o ~ } } L = 6$ and not beyond, so six latent variables are needed, which is the same as the number of free parameters of ΛCDM. For $\ell \geq 3 0 .$ , earlier analyses find five (Piras et al. 2025; Sun et al. 2026), which matches the single combination $A _ { s } e ^ { - 2 \tau }$ that replaces $A _ { s }$ and � there. The two lowTTTEEE+full cases difer: for SA, $L = 5$ and $L \ : = \ : 6$ are indistinguishable, and for CA the fit is best at $L = 5 .$ , with the choice of this particular loss function being a possible reason for that. The data therefore do not single out one latent dimension in every case, but show a trend to conclude that $L = 6$ is the best, or within ln $B \simeq 1$ of the best, in five of the six cases, and $L = 7$ gives no significant gain over $L = 6$ in any case.

(iv) SA and CA strategies are two points on a trade-of between interpretability and fit to the data. SA gives clearer disentanglement, with a mean score of 0.58 against 0.40 for CA, and the diference grows with �. All six parameters are matched cleanly to a latent variable only in three SA models, all with $L \geq 6 . \mathrm { C A }$ removes the need to fix the weight of the disentanglement loss in advance. It also gives a higher evidence at $L = 5$ and $^ { 6 , }$ and remains preferred when each strategy is compared at the value of � that suits it best. The four model-comparison measures also agree on the preferred � in all three CA cases, which they do not for SA. The gain in disentanglement is not tied to a better fit: for SA lowTTTEEE+full and lowTTTEEE+high the score is highest at $L = 7$ , whereas the evidence is highest at $L = 6$

These results show that a deep-learning approach can produce a compact parametrisation of the CMB temperature and polarisation spectra, with the loss weights set automatically. No cosmological model is built into the network or its latent space, although the training spectra are generated within ΛCDM. Starting from this freedom, six latent variables sufice in five of the six cases studied in this work. This is as many as there are free parameters in ΛCDM, and one more than required for spectra at $\ell \geq 3 0$ . The latent variables map onto the ΛCDM parameters to diferent extents: more clearly for �<sub>b</sub>, �, �<sub>cdm</sub> and ℎ, and less for $A _ { s }$ and $n _ { s }$

This framework can be extended in several directions. The first is to CMB data of higher precision and wider multipole range. Groundbased experiments such as ACT (Louis et al. 2025) and SPT-3G (Benson et al. 2014) already measure the spectra to higher ℓ than Planck. The Simons Observatory (Abitbol et al. 2025) will improve the small-scale polarisation, and LiteBIRD (Aizawa et al. 2026), a full-sky polarisation mission designed mainly to search for primordial B-modes, will also measure the large-scale polarisation that constrains �. Since the separation of � and $A _ { s }$ in our latent variables is expected to depend on the large-scale polarisation, such data would test it directly, and would show how the number of latent variables and the parameter mappings respond to better measurements.

The second is to apply the framework to baryon acoustic oscillations (BAO) and other late-time data, for example from the Dark Energy Spectroscopic Instrument (DESI) (Adame et al. 2025; Abdul Karim et al. 2025), and to combine them with the CMB latent variables. This would allow extensions, such as dynamical dark energy, to be studied in a data-driven parametrisation, as was done for early dark energy in the temperature-only case by Piras et al. (2025). Applying the same pipeline to spectra from extended models would also show whether the preferred latent dimension follows the underlying model. Within the CMB, the latent variables can also be used directly in parameter inference and related to derived combinations of physical parameters such as $\theta _ { s }$

Finally, the architecture and training strategy used here are not final. The VAE, the multi-task weighting and the automated choice of the disentanglement weight can be refined further. We plan to improve the statistical robustness of these techniques, and hope that this work encourages the development of more sophisticated, reliable and widely applicable data-driven approaches in cosmology.

## ACKNOWLEDGMENTS

We acknowledge the computational facilities provided by the SyMeC High-Performance Computing (HPC) cluster of Indian Statistical Institute (ISI) Kolkata and the Pegasus HPC facility at the Inter-University Centre for Astronomy and Astrophysics (IUCAA), Pune, India. AM and RS thank Tanima Duary, Purba Mukherjee, Biswajit Das, Debarun Paul and Arko Bhaumik, for insightful discussions. AM thanks the Anusandhan National Research Foundation (ANRF), Government ofIndia and ISI Kolkata for financial support through Junior Research Fellowship. RS thanks ISI Kolkata for financial support through Senior Research Fellowship. MWH, SS, AAS and SP thank ANRF, Government of India, for partial support through Project No. CRG/2023/003984. SP also thanks IUCAA, Pune for Visiting Associateship during which a part of the work was done.

## DATA AVAILABILITY

The OCCAMS (Optimal Components of CMB with Autoencoders and Model Selection) suite and data for this article will be made public on GitHub once the paper is published. All other data sets used in this work are publicly available. The modified versions of the codes by other authors may be made available upon reasonable request.

## REFERENCES

Abbott T. M. C., et al., 2022, Phys. Rev. D, 105, 023520

Abdalla E., et al., 2022, JHEAp, 34, 49

Abdul Karim M., et al., 2025, Phys. Rev. D, 112, 083515

Abitbol M., et al., 2025, JCAP, 08, 034

Adame A. G., et al., 2025, JCAP, 02, 021

Ade P. A. R., et al., 2016, Astron. Astrophys., 594, A13

Aghanim N., et al., 2020a, Astron. Astrophys., 641, A5

Aghanim N., et al., 2020b, Astron. Astrophys., 641, A6

Aizawa K., et al., 2026, in SPIE Astronomical Telescopes + Instrumentation 2026. (arXiv:2607.25285), doi:10.48550/arXiv.2607.25285

Alsing J., Charnock T., Feeney S., Wandelt B., 2019, Mon. Not. Roy. Astron. Soc., 488, 4440

Benson B. A., et al., 2014, Proc. SPIE Int. Soc. Opt. Eng., 9153, 91531P

Blas D., Lesgourgues J., Tram T., 2011, JCAP, 07, 034

Burgess C. P., Higgins I., Pal A., Matthey L., Watters N., Desjardins G., Lerchner A., 2018, in 2017 NIPS Workshop on Learning Disentangled Representations. (arXiv:1804.03599), doi:10.48550/arXiv.1804.03599

Charnock T., Lavaux G., Wandelt B. D., 2018, Phys. Rev. D, 97, 083004

Cipolla R., Gal Y., Kendall A., 2018, in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, pp 7482–7491, doi:10.1109/CVPR.2018.00781, https: //doi.ieeecomputersociety.org/10.1109/CVPR.2018.00781

Cover T., Thomas J., 2012, Elements of Information Theory. Wiley, https: //books.google.co.in/books?id=VWq5GG6ycxMC

Di Valentino E., Brout D., eds, 2024, The Hubble Constant Tension. Springer Series in Astrophysics and Cosmology, Springer, doi:10.1007/978-981- 99-0177-7

Di Valentino E., et al., 2021a, Class. Quant. Grav., 38, 153001

Di Valentino E., et al., 2021b, Astropart. Phys., 131, 102605

Di Valentino E., et al., 2025, Phys. Dark Univ., 49, 101965

Dodelson S., Schmidt F., 2020, Modern Cosmology. Academic Press, doi:10.1016/C2017-0-01943-2

Dvorkin C., et al., 2022, in Snowmass 2021. (arXiv:2203.08056), doi:10.48550/arXiv.2203.08056

Efstathiou G., 2025, Phil. Trans. Roy. Soc. Lond. A, 383, 20240022

Efstathiou G., Bond J. R., 1999, Mon. Not. Roy. Astron. Soc., 304, 75

Foreman-Mackey D., Hogg D. W., Lang D., Goodman J., 2013, Publ. Astron. Soc. Pac., 125, 306

Giarè W., 2024, CMB Anomalies and the Hubble Tension. Springer Nature Singapore, Singapore, pp 675–686, doi:10.1007/978-981-99-0177-7\_36

Gómez-Valent A., 2022, Phys. Rev. D, 106, 063506

Guo N., Lucie-Smith L., Peiris H. V., Pontzen A., Piras D., 2024, Mon. Not. Roy. Astron. Soc., 532, 4141

Handley W., Lemos P., 2019, Phys. Rev. D, 100, 043504

Heavens A., Fantaye Y., Mootoovaloo A., Eggers H., Hosenie Z., Kroon S., Sellentin E., 2017, arXiv:1704.03472

Herold L., Ferreira E. G. M., Komatsu E., 2022, Astrophys. J. Lett., 929, L16

Herold L., Ferreira E. G. M., Heinrich L., 2025, Phys. Rev. D, 111, 083504

Hinshaw G., et al., 2013, Astrophys. J. Suppl., 208, 19

Hinton G. E., Salakhutdinov R. R., 2006, Science, 313, 1127647

Horowitz B., Dornfest M., Lukić Z., Harrington P., 2022, Astrophys. J., 941, 42

Hu W., Dodelson S., 2002, Ann. Rev. Astron. Astrophys., 40, 171

Hu W., Holder G. P., 2003, Phys. Rev. D, 68, 023001

Hu J.-P., Wang F.-Y., 2023, Universe, 9, 94

Huterer D., Starkman G., 2003, Phys. Rev. Lett., 90, 031301

Jefreys H., 1939, The Theory of Probability. Oxford Classic Texts in the Physical Sciences, Oxford University Press (Clarendon Press)

Jones E., Singal J., 2017, Astron. Astrophys., 600, A113

Kamionkowski M., Riess A. G., 2023, Ann. Rev. Nucl. Part. Sci., 73, 153

Kannan S., Qiu T., Cuesta-Lazaro C., Jeong H., 2026, Mach. Learn. Sci. Tech., 7, 055019

Kingma D. P., Welling M., 2013, arXiv:1312.6114

Kosowsky A., Milosavljevic M., Jimenez R., 2002, Phys. Rev. D, 66, 063007

Kuhn H. W., 1955, Naval Research Logistics Quarterly, 2, 83

Landau L. D., Lifschits E. M., 1975, The Classical Theory of Fields. Course of Theoretical Physics Vol. Volume 2, Pergamon Press, Oxford, doi:10.1016/c2009-0-14608-1

Lanusse F., Mandelbaum R., Ravanbakhsh S., Li C.-L., Freeman P., Póczos B., 2021, Monthly Notices of the Royal Astronomical Society, 504, 5543–5555

Leach S. M., 2006, Mon. Not. Roy. Astron. Soc., 372, 646

Lesgourgues J., 2011, arXiv:1104.2932

Lewis A., 2025, JCAP, 08, 025

Liddle A. R., 2004, Mon. Not. Roy. Astron. Soc., 351, L49

Lin S., Li Y., Genel S., Villaescusa-Navarro F., Dai B., Luo W., Wang Y., 2026, Astrophys. J. Lett., 996, L41

Locatello F., Bauer S., Lucic M., Gelly S., Schölkopf B., Bachem O., 2018, CoRR, abs/1811.12359

Lochner M., McEwen J. D., Peiris H. V., Lahav O., Winter M. K., 2016, Astrophys. J. Suppl., 225, 31

Louis T., et al., 2025, JCAP, 11, 062

Lucie-Smith L., Peiris H. V., Pontzen A., Nord B., Thiyagalingam J., Piras D., 2022, Phys. Rev. D, 105, 103533

Lucie-Smith L., Despali G., Springel V., 2024, Mon. Not. Roy. Astron. Soc., 532, 164

Peebles P. J. E., 2020, Principles of Physical Cosmology. Princeton University Press

Peiris H. V., Verde L., 2010, Phys. Rev. D, 81, 021302

Perivolaropoulos L., Skara F., 2022, New Astron. Rev., 95, 101659

Piras D., Peiris H. V., Pontzen A., Lucie-Smith L., Guo N., Nord B., 2023, Mach. Learn. Sci. Tech., 4, 025006

Piras D., Herold L., Lucie-Smith L., Komatsu E., 2025, Phys. Rev. D, 111, 083537

Portillo S. K. N., Parejko J. K., Vergara J. R., Connolly A. J., 2020, The Astronomical Journal, 160, 45

Prince H., Dunkley J., 2019, Phys. Rev. D, 100, 083502

Rezende D. J., Mohamed S., Wierstra D., 2014, in Proceedings of the 31st International Conference on Machine Learning. (arXiv:1401.4082), doi:10.48550/arXiv.1401.4082

Riess A. G., et al., 2022, Astrophys. J. Lett., 934, L7

Rojas L., Espinoza S., González E., Maldonado C., Luo F., 2025, Galaxies, 13, 114

Sachs R. K., Wolfe A. M., 1967, Astrophys. J., 147, 73

Schöneberg N., Franco Abellán G., Pérez Sánchez A., Witte S. J., Poulin V., Lesgourgues J., 2022, Phys. Rept., 984, 1

Shah R., Saha S., Mukherjee P., Garain U., Pal S., 2024, Astrophys. J. Suppl., 273, 27

Silk J., 1968, Astrophys. J., 151, 459

Sun T.-Y., Li T.-N., Wang H., Zhang J.-F., Zhang X., 2026, Phys. Rev. D, 114, 043547

Trotta R., 2008, Contemp. Phys., 49, 71

Verde L., Treu T., Riess A. G., 2019, Nature Astron., 3, 891

Verde L., Schöneberg N., Gil-Marín H., 2024, Ann. Rev. Astron. Astrophys., 62, 287

Vergara J. R., Estévez P. A., 2013, Neural Computing and Applications, 24, 175–186

Weinberg S., 2008, Cosmology. Oxford University Press (OUP)

Zaldarriaga M., 1997, Phys. Rev. D, 55, 1822

Zaldarriaga M., Spergel D. N., Seljak U., 1997, Astrophys. J., 488, 1

## APPENDIX A: CONSTRUCTION OF $\pmb { \mathcal { L } } _ { \mathbf { d i s } }$

Given a batch of � samples, let $\mathbf { Z } \in \mathbb { R } ^ { N \times L }$ be the latent representations and $\mathbf { \Theta } ( \mathbf { \Theta } ) \in \mathbb { R } ^ { N \times P }$ be the theoretical model parameter labels, where � is the latent dimension of a particular VAE model and $P$ is number of cosmological model parameters (6 for ΛCDM). We first apply zero-mean, unit-variance standardisation to obtain Z<sup>˜</sup> and �<sup>˜</sup> .

Then the squared Pearson correlation matrix is computed between each latent dimension and each physical parameter. To frame this as a probability distribution over the parameters for each latent node, we apply column-wise normalisation to obtain the matrix $\mathbf { M } \in \mathbb { R } ^ { P \times L }$

$$
\mathbf { M } _ { i , j } = \frac { ( \tilde { \mathbf { \Theta } } _ { : , i } \cdot \tilde { \mathbf { Z } } _ { : , j } ) ^ { 2 } / N ^ { 2 } } { \sum _ { k = 1 } ^ { P } ( \tilde { \mathbf { \Theta } } _ { : , k } \cdot \tilde { \mathbf { Z } } _ { : , j } ) ^ { 2 } / N ^ { 2 } + \epsilon } ,
$$

where $\epsilon = 1 0 ^ { - 8 }$ ensures numerical stability. The columns of M sum to 1, representing the fractional information capacity each latent variable dedicates to the respective parameters. Fig. A1 summarises the construction of $\mathbf { M } _ { i , j }$ and of the loss terms as follows. To enforce 1-to-1 mappings while remaining sensitive to possible physical degeneracies $\bar { ( } \mathrm { e . g . , } A _ { s } e ^ { - 2 \tau } )$ , our total disentanglement loss is defined as a weighted sum of three distinct objectives,

$$
\mathcal { L } _ { \mathrm { d i s } } = \mathcal { L } _ { \mathrm { p u r i t y } } + \alpha \mathcal { L } _ { \mathrm { c o l l i s i o n } } + \beta ^ { \prime } \mathcal { L } _ { \mathrm { s t r i c t } } .
$$

1. Top-2 Purity Loss $( \mathcal { L } _ { \mathrm { p u r i t y } } ) \colon$ We penalise latent variables that distribute information across more than two parameters. By extracting the top two values in each column of M (denoted as $\nu _ { 1 , j }$ and $\nu _ { 2 , j }$ for

![](images/df1b887f6d555f64ec0d7f81dff30385a70c0875f4c8dac7b441f0cfb8915ec6.jpg)  
Figure A1. Visualisation of the disentanglement penalty calculation during a training epoch. The network multiplies the transposed standardised parameter batch $( \tilde { \Theta } ^ { T } )$ by the standardised latent batch $( \tilde { Z } )$ . Each element $R _ { i , j }$ is the squared correlation between parameter � (e.g. �) and latent $j .$ Normalising each column of � to sum to one gives M, from which the three terms of $\mathcal { L } _ { \mathrm { d i s } }$ are computed: purity and strict from the columns of M, and collision from the of-diagonal elements of $\mathbf { M } ^ { T } \mathbf { M } .$ Minimising these terms makes each latent follow only a few parameters, and diferent latents follow diferent ones.

the �-th latent variable), we encourage the network to concentrate its predictive power,

$$
\mathcal { L } _ { \mathrm { p u r i t y } } = \sum _ { j = 1 } ^ { L } \left( 1 - ( \nu _ { 1 , j } + \nu _ { 2 , j } ) \right) ^ { 2 } .
$$

2. Collision Loss $( { \mathcal { L } } _ { \mathrm { c o l l i s i o n } } ) { : }$ To prevent mode collapse where multiple latent variables track the same $" \mathrm { e a s y " }$ parameter, we compute the overlap (Gram) matrix $\mathbf { O } = \mathbf { M } ^ { T } \mathbf { M } \in \mathbb { R } ^ { L \times L }$ . We penalise the of-diagonal elements of O, encouraging orthogonality between diferent latent dimensions,

$$
\mathcal { L } _ { \mathrm { c o l l i s i o n } } = \sum _ { i = 1 } ^ { L } \sum _ { j = 1 \atop j \neq i } ^ { L } \mathbf { O } _ { i , j } ^ { 2 } .
$$

3. Strict Purity Tie-Breaker $( \mathcal { L } _ { \mathrm { s t r i c t } } ) \colon$ To ensure the network prefers perfect 1-to-1 mappings and only utilises the 2-parameter degeneracy allowance only when physically necessary to lower the reconstruction error, we apply a minor penalty to the deviation of the absolute maximum column value from 1,

$$
\mathcal { L } _ { \mathrm { s t r i c t } } = \sum _ { j = 1 } ^ { L } \left( 1 - \operatorname* { m a x } _ { i } ( \mathbf { M } _ { i , j } ) \right) ^ { 2 } .
$$

The hyperparameters � and $\beta ^ { \prime }$ control the trade-of between orthogonality and strictness. In practice, $\beta ^ { \prime } \ll 1 . 0$ acts as a soft regulariser against unnecessary use of the two-parameter allowance.

<table><tr><td>Configuration</td><td>Strategy L</td><td></td><td> ${ \bf w } _ { \mathrm { l } } \left( { \bf 1 0 ^ { 4 } } \right)$ </td><td> $\mathbf { w _ { h } } \left( \mathbf { 1 0 ^ { 2 } } \right)$ </td><td>β</td><td>Wd</td></tr><tr><td rowspan="6">lowTTTEEE+full</td><td rowspan="3">SA</td><td>5</td><td>0.168</td><td>0.283</td><td>0.253</td><td>5</td></tr><tr><td>6</td><td>0.994</td><td>1.16</td><td>0.209</td><td>5</td></tr><tr><td>7</td><td>0.973</td><td>1.21</td><td>0.238</td><td>5</td></tr><tr><td rowspan="3">CA</td><td>5</td><td>0.346</td><td>0.225</td><td>0.238</td><td>0.810</td></tr><tr><td>6</td><td>1.28</td><td>1.21</td><td>0.191</td><td>0.587</td></tr><tr><td>7</td><td>1.69</td><td>1.27</td><td>0.192</td><td>0.446</td></tr><tr><td rowspan="6">lowTTTEEE+high</td><td rowspan="3">SA</td><td>5</td><td>0.178</td><td>0.272</td><td>0.250</td><td>5</td></tr><tr><td>6</td><td>0.817</td><td>1.13</td><td>0.212</td><td>5</td></tr><tr><td>7</td><td>1.61</td><td>1.24</td><td>0.184</td><td>5</td></tr><tr><td rowspan="3">CA</td><td>5</td><td>0.323</td><td>0.22</td><td>0.239</td><td>0.799</td></tr><tr><td>6</td><td>1.19</td><td>1.17</td><td>0.199</td><td>0.583</td></tr><tr><td>7</td><td>1.64</td><td>1.24</td><td>0.193</td><td>0.454</td></tr><tr><td rowspan="5">lowTT-TEEE+high</td><td rowspan="3">SA</td><td>5</td><td>1.07,0.274</td><td>0.159</td><td>0.190</td><td>5</td></tr><tr><td>6</td><td>1.58,1.27</td><td>1.06</td><td>0.160</td><td>5</td></tr><tr><td>7</td><td>2.56, 2.35</td><td>1.24</td><td>0.151</td><td>5</td></tr><tr><td rowspan="3">CA</td><td>5</td><td>1.33,0.354</td><td>0.134</td><td>0.185</td><td>0.799</td></tr><tr><td>6</td><td>1.80, 1.42</td><td>1.17</td><td>0.162</td><td>0.596</td></tr><tr><td>7</td><td>1.74, 1.75</td><td>1.24</td><td>0.165</td><td>0.448</td></tr></table>

Table B1. Loss weights at the end of training for all 18 models. �<sub>�</sub> and �ℎ are the weights of the low-ℓ and high-ℓ reconstruction terms, � is that of the KL term, and �<sub>�</sub> is that of the disentanglement term. For lowTT-TEEE+high the �<sub>�</sub> column lists (�<sub>lowTT</sub>, �<sub>lowTEEE</sub> ), and for lowTTTEEE+full the �ℎ column holds $w _ { \mathrm { f u l l } }$

## APPENDIX B: LOSS FUNCTION WEIGHTS

Table B1 lists the loss weights at the end of training for all 18 models, as defined in equation (2). The low-ℓ weights are larger than the highℓ weights by roughly two orders of magnitude, and $\beta$ lies between 0.15 and 0.26 in all models.

## APPENDIX C: DISENTANGLEMENT SCOREDEPENDENCE ON MI THRESHOLD

The disentanglement score of Section 4 counts a parameter as captured only if its matched MI exceeds the threshold $M _ { \mathrm { m i n } } = 0 . 1 0 .$ Fig. C1 shows the score of all 18 models as a function of this threshold between 0.02 and 0.20. Each curve is a step function, because the score drops whenever the threshold passes the matched MI of one parameter and that parameter leaves the coverage. Below about 0.04 all scores sit on their plateau values. The three highest-scoring models at the nominal threshold, SA lowTTTEEE+full with $L = 6$ and 7 and SA lowTTTEEE+high with � = 7, remain the three highest up to a threshold of 0.17; above it no model exceeds a score of 0.58. The CA models lose a larger part of their score at thresholds between 0.05 and 0.10 than the SA models, so their scores depend more strongly on the chosen threshold. The ordering among the lower-scoring models changes with the threshold, and hence diferences between them should not be over-interpreted.

## APPENDIX D: MCMC CONTOURS

Figs. D1, D2 and D3 show the marginal posteriors of the latent variables � for the lowTTTEEE+high models, plotted using getdist (Lewis 2025). The numbering, sign and scale of the latent variables are arbitrary, so the panels cannot be compared latent by latent. For � = 6 and 7, some pairs of latent variables are almost perfectly (anti-)correlated, indicating that the data constrain only a combination of the two and not each separately.

![](images/d3b768e4f30e17425dd7210d0db2531ee5c354c336242d40c527bbb65422df75.jpg)

![](images/3b7dbc017d4018e8580e805160877c32478a20ab2ea45ef4dd2ecf22f5a9ebce.jpg)

![](images/799ee52a79ad7026cdb46870c019d253e7c8853b9cae65dfa0754c040ea3048d.jpg)  
Figure C1. Sensitivity of the disentanglement score to the MI threshold for the three configurations (solid: SA, dashed: CA; colours: �). The dotted line marks the nominal threshold of 0.10 used in the main analysis.

![](images/e6810324d5f032e27171e113567377935d6a9ca5b2c572590337d9f5f84f01b2.jpg)

![](images/643b9982c544ccb70284d06c0a3c359d155d8af62fa829205d6a407f7a3fc411.jpg)  
Figure D1. Posterior distributions of the latent variables for the lowTTTEEE+high models with $L = 5 \colon \mathrm { S A }$ (top) and CA (bottom). Contours enclose 68 and 95 per cent of the posterior.

![](images/95904f1786953a91d11a86fbecbcce137ad224e7f93e15f1370be091dac83a97.jpg)

![](images/26f0a299749c1c31af3f571ea7372d7e9a8e518f1a184b7857acb830f726b2ba.jpg)  
Figure D2. Posterior distributions of the latent variables for the lowTTTEEE+high models with $L = 6 \colon \mathrm { S A }$ (top) and CA (bottom). Contours enclose 68 and 95 per cent of the posterior.

This paper has been typeset from a T X/LAT X file prepared by the author.

![](images/4aef48972f1ac6d9e5a52f8482032b11cfe3673fdb595508a9e8736c1d49ade9.jpg)

![](images/e26254d8e84e0cf09f5be0185ca5d4f2a014bdb8836cfb4bd131d2138fdbad40.jpg)  
Figure D3. Posterior distributions of the latent variables for the lowTTTEEE+high models with $L = 7 \colon \mathrm { S A }$ (top) and CA (bottom). Contours enclose 68 and 95 per cent of the posterior.