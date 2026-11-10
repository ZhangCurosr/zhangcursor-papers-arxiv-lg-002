# TSFFormer: A Frequency-Domain Spike Transformer for Green Human Activity Recognition

Jiale Deng   
Xidian University   
Xi’an 710071, China   
25011211021@stu.xidian.edu.cn

Shuiying Xiang Xidian University Xi’an 710071, China syxiang@xidian.edu.cn

## Abstract

Wearable Inertial Measurement Units (IMUs) enable pervasive Human Activity Recognition (HAR), but traditional Deep Neural Networks (DNNs) sufer from an “arithmetic-latency deadlock” due to dense floating-point MAC operations. While Spiking Neural Networks (SNNs) ofer ultra-low power, existing ones face spatial blind spots, rigid neurons, and gradient degradation. To address these, we propose TSF-Former, a purely spiking Fourier Transformer for Green HAR. It introduces two key innovations: (1) a Multi-HeadFourier attention that performs frequency-domain global mixing, reducing complexity from O(V<sup>2</sup>) to O(V log V ), eliminating spatial blind spots, and saving 50% GPU memory; and (2) a dual-compartment Temporal Spiking LIF (TSLIFNode) that utilizes dynamic thresholds to adapt to non-stationary rhythms and mitigate gradient degradation. Extensive experiments on 8 public IMU benchmarks demonstrate that TSFFormer surpasses state-of-the-art SNNs and even the classic DeepConvLSTM, achieving a 3 faster training speed with an energy footprint of 0.1 pJ. Ablation studies confirm the synergistic efect of our modules, providing a highly accurate, ultra-low-power event-driven paradigm for wearable sensing.

## 1. Introduction

HAR using IMUs is a cornerstone of pervasive computing[24]. Over the past decade, the field has evolved from hand-crafted features and shallow classifiers to DNNs and, more recently, heavy Transformerbased foundation models [22]. However, this relentless pursuit of accuracy has spawned a critical bottleneck in edge deployment: the arithmetic-latency deadlock. DNNs rely on dense 32-bit floating-point (FP32) Multiply-Accumulate (MAC) operations ( 4.6 pJ per step) and rigid “bufer-then-compute” paradigms, resulting in prohibitive power consumption and observation latency that make always-on HAR impractical.

SNNs ofer a neuromorphic alternative by replacing dense MACs with sparse Accumulate (AC) operations ( 0.1 pJ) [2]. Despite this eficiency, applying SNNs to IMU-based HAR faces severe limitations: (1) Spatial blind spots: naive SNNs flatten sensor data, ignoring human biomechanical topology[24]. (2) Neural rigidity: standard Leaky Integrate-and-Fire (LIF) neurons use fixed thresholds, failing to adapt to non-stationary motion rhythms[6]. (3) Gradient degradation: training long sequences via BPTT forces models to rely on long integration windows, precluding low-latency inference.

To bridge these gaps, we propose TSFFormer, a purely spiking Fourier Transformer for Green HAR. It integrates a Temporal Spiking LIF (TSLIF) neuron to dynamically modulate firing thresholds, and a Multi-HeadFourier attention [7] to eficiently capture global spatio-temporal dependencies in the frequency domain. Combined with a non-downsampling Patch Embedding (MS-SPS) to preserve spatial topology, TSFFormer fundamentally resolves the aforementioned limitations. Our main contributions are:

• Purely Spiking Fourier Paradigm: We propose TSF-Former, a HAR architecture that deeply fuses spiking neural networks with Fourier attention. This model significantly reduces the operation energy consumption and greatly improves the inference speed.

• Dual-Adaptive Spatio-Temporal Core:We used a dual adaptive core. The TSLIF neuron prevents catastrophic accuracy drops caused by fixed thresholds. Meanwhile, MultiHeadFourier reduces complexity from O(V<sup>2</sup>) to O(V log V ), accelerating training by 3 and saving 50% GPU memory.

• Topology-Preserving Sparse Computation: We adopt MS-SPS for channel mapping and keep the vertex dimension unchanged, so as to completely preserve the spatial structure of sensor nodes. Combined with the sparse event-driven characteristics of spiking MLP, the model greatly reduces redundant computations while maintaining high accuracy.

## 2. Related Work

## 2.1. Evolution of IMU-based HAR Architectures

IMU-based HAR has evolved from early classifiers with hand-crafted features to automated representation learning. Deep architectures, notably CNNs and recurrent networks (e.g., DeepConvLSTM[14]), established strong baselines by fusing local spatial extraction with temporal modeling. Subsequent research integrated attention mechanisms and advanced macroarchitectures [13] to capture salient spatio-temporal segments. Recently, the field has embraced large-scale foundation models and LLM-driven semantic reasoning [22] to handle extreme cross-domain variability. Among these, LIMU-BERT stands out as a representative IMU-specific foundation model that applies selfsupervised learning to unlabeled sensor data for generalized representation[17].

## 2.2. Spiking Neural Networks and Energy-Efficient Computing

To circumvent the energy bottleneck, SNNs have emerged as a promising ultra-low-power paradigm. By processing asynchronous binary events, SNNs[5] replace power-hungry MACs with eficient AC operations. Driven by Surrogate Gradient (SG) methods and Spatio-Temporal Backpropagation (STBP), deep SNNs have achieved notable progress [2]. The fundamental computational unit is typically the LIF neuron, whose discrete-time dynamics are[11]:

$$
U [ t ] = \tau U [ t - 1 ] ( 1 - S [ t - 1 ] ) + W X [ t ]\tag{1}
$$

$$
S [ t ] = H ( U [ t ] - V _ { t h } )\tag{2}
$$

where $U [ t ]$ is the membrane potential, τ is the decay factor, X[t] is the input current, W represents synaptic weights, and $S [ t ]$ is the binary output spike triggered when U[t] exceeds the firing threshold $V _ { t h } . \quad H ( \cdot )$ denotes the Heaviside step function.

Despite its success, deploying SNNs for IMU-based HAR exposes three critical gaps: (1) Spatial blind spots: existing SNNs typically flatten multi-sensor series into independent 1D channels, lacking structural capability to explicitly route physical topologies. (2) Neural rigidity: standard LIF neurons rely on a fixed $V _ { t h }$ . Adaptive variants $( \mathrm { e . g . , A L I F [ 1 ] } )$ introduce heuristic threshold decay but lack causal modulation for rapidly changing human motion rhythms[6]. (3) Gradient degradation: optimizing deep SNNs over long sequences via BPTT forces networks to rely on long integration windows, precluding ultra-low-latency prediction.

## 2.3. Spiking Transformers and Frequency-Domain Attention

Recent Spiking Transformers[18, 25–27] introduce attention-style mixing into SNNs for global modeling. However, they mostly focus on static images, relying on dot products or linear projections that struggle to capture global dependencies in the frequency domain, often losing biomechanical correlations due to flattened inputs. Conversely, frequency-domain modeling in time-series analysis ofers unique advantages. The Fourier transform converts temporal convolutions into frequency-domain multiplications[8, 9], capturing long-range dependencies with lower computational complexity. Combining multi-head mechanisms with Fourier transforms (MultiHeadFourier) avoids explicit sequence-length squared complexity while retaining sensitivity to periodic patterns[7]. However, seamlessly integrating frequency-domain attention into SNNs while maintaining event-driven sparse computation remains an open problem.

## 2.4. Adaptive Spiking Neurons and Temporal Dynamics

The fixed threshold of standard LIF neurons struggles to adapt to non-stationary motion rhythms. While variants like ALIF adjust sensitivity via threshold decay, they rely on heuristic rules and lack explicit modeling of causal temporal contexts. TSLIF neurons introduce time-dependent threshold modulation, simulating the dynamic threshold characteristics of biological neurons to enable precise feature extraction across diferent motion stages[23]. However, efectively combining TSLIF with Spiking Transformer architectures while maintaining gradient stability during training requires further exploration[3].

## 3. Methodology

## 3.1. Overall Architecture

To achieve high-accuracy, low-power HAR in wearable inertial sensing, we propose TSFFormer—a Green HAR architecture that fuses SNNs, TSLIF neurons, and MultiHeadFourier. Formally, given a raw multinode IMU tensor $X \in \mathbb { R } ^ { T \times B \times C \times \check { V } }$ (time steps $T _ { : }$ , batch size $B ,$ sensor channels $C _ { i }$ , physical nodes V), the model processes data through a purely spike-driven forward pipeline and outputs classification predictions along the temporal dimension.

As shown in Figure 1, TSFFormer consists of several parts: the perception front-end and spiking embedding map continuous inputs into high-dimensional spike sequences; spatio-temporal spiking Transformer blocks are stacked for L layers, each containing MultiHeadFourier attention and a spiking MLP; temporal dimension readout aggregates features along the vertex dimension; and the classification head averages predictions over the time dimension. The entire network replaces dense floating-point MACs with event-driven sparse ACs, utilizes the dynamic threshold modulation of TSLIF neurons to adapt to non-stationary motion rhythms, and eficiently captures global joint correlations through frequency-domain attention.

## 3.2. Perception Front-end and Spiking Embedding

The raw IMU tensor $X \in \mathbb { R } ^ { T \times B \times C \times V }$ is first mapped to the embedding space via a Simple MS-SPS. MS-SPS employs a single-layer 1D convolution (kernel size 1) and Batch Normalization (BatchNorm1d) to achieve channel transformation:

$$
X _ { e m b } = \mathrm { B N } ( \operatorname { C o n v 1 d } _ { 1 \times 1 } ( X ) ) \in \mathbb { R } ^ { T \times B \times D \times V }\tag{3}
$$

where D is the embedding dimension. This operation does not perform spatial downsampling, fully preserving the biomechanical topology of the vertex dimension. For the first layer, no spiking activation is applied; subsequent layers pass through a TSLIF neuron before convolution to introduce spiking sparsity. The output $X _ { e m b }$ serves as the input to the first Transformer block.

## 3.3. Spatio-Temporal Spiking Transformer Block

The network backbone consists of L stacked MS-Blocks. Each block contains two sub-layers: Multi-HeadFourier attention and a spiking MLP, both using residual connections and DropPath regularization. The input tensor maintains the shape $T \times B \times D \times V$ , with all operations performed independently along the temporal dimension.

## 3.3.1. Multi-Head Fourier Attention

To eficiently capture global dependencies between vertices and avoid the quadratic complexity of traditional dot-product attention, we adopt Multi-Head Fourier attention. Given input $X \in \dot { \mathbb { R } ^ { T \times B \times D \times V } }$ , we first reshape it into $B \times T$ independent sequences:

$$
X _ { s e q } = \mathrm { r e s h a p e } ( X . \mathrm { p e r m u t e } ( 1 , 0 , 3 , 2 ) ) \in \mathbb { R } ^ { ( B T ) \times V \times D }\tag{4}
$$

meaning each time slice and sample is processed independently, with the vertex dimension V as the sequence length and D as the feature dimension. Then, MultiHeadFourier executes the following steps:

• Local causal convolution: Use a depthwise separable 1D convolution (kernel size 3, groups D) to extract local temporal inductive bias, with left padding to ensure causality:

$$
X _ { c o n v } = \mathrm { C o n v 1 d _ { d e p t h w i s e } } ( X _ { s e q } )\tag{5}
$$

• Layer Normalization: Apply LayerNorm to the convolution output.

• Content and Gating Stream Projections:

$V = W _ { V } ( \mathrm { L N } ( X _ { c o n v } ) ) \in \mathbb { R } ^ { ( B T ) \times V \times D }$ , split into H heads with dimension $d _ { h } = D / H$

$G = W _ { G 2 } ( \mathrm { S i L U } ( W _ { G 1 } ( \mathrm { L N } ( X _ { c o n v } ) ) ) )$ , where $W _ { G 2 }$ is a grouped 1D convolution (groups H), also split into heads.

• Frequency-Domain Causal Mixing: Apply real Fast Fourier Transform (rFFT) to both content and gating streams, padding length to $N = 2 V$ :

$$
\mathcal V = \mathrm { r F F T } ( V , n = N ) , \quad \mathcal G = \mathrm { r F F T } ( G , n = N )\tag{6}
$$

Multiply element-wise in the frequency domain:

$$
\mathcal { X } = \mathcal { V } \odot \mathcal { G }\tag{7}
$$

Then inverse transform back to the time domain and truncate to the original length V:

$$
X _ { m i x e d } = \mathrm { i r F F T } ( \mathcal { X } , n = N ) [ : , : , : V , : ]\tag{8}
$$

• Output Projection: Concatenate the multi-head results and pass through a linear layer:

$$
Y = W _ { O } ( \mathrm { r e s h a p e } ( X _ { m i x e d } ) ) \in \mathbb { R } ^ { ( B T ) \times V \times D }\tag{9}
$$

Finally, restore the shape to $T \times B \times D \times V$ . This design leverages the property that frequency-domain multiplication is equivalent to time-domain convolution, achieving a global receptive field with O(V log V ) complexity[7], while adaptively adjusting frequencydomain responses via the gating mechanism to enhance modeling of periodic motion patterns.

## 3.3.2. Spiking MLP

The spiking MLP is responsible for channel mixing and non-linear transformation. The input $X \in \mathbb { R } ^ { T \times B \times D \times V }$ passes through two layers sequentially:

$$
X _ { 1 } = \mathrm { T S L I F } ( \mathrm { B N } _ { 1 } ( \mathrm { C o n v } 1 \mathrm { d } _ { 1 \times 1 } ( X ) ) )\tag{10}
$$

$$
X _ { 2 } = \mathrm { B N _ { 2 } } ( \mathrm { C o n v 1 } \mathrm { d } _ { 1 \times 1 } ( X _ { 1 } ) )\tag{11}
$$

$$
Y = \mathrm { T S L I F } ( X _ { 2 } )\tag{12}
$$

Before each convolution, the T and B dimensions are flattened to adapt to 1D convolution. The hidden dimension is $D _ { h i d } = \lfloor D \cdot \mathrm { m l p \_ r a t i o } \rfloor$ . Both TSLIF neurons act on the channel dimension, and the output maintains $T \times B \times D \times V$

![](images/37b00c8f19179f3f91255c796cb22260ebea392469a43661ecd261496ed205de.jpg)  
Figure 1. The overall structure of our proposed TSFFormer framework.

## 3.3.3. TSLIFNode

TSLIFNode is a dual-compartment spiking neuron whose dynamics simulate the dendrite-soma coupling and dynamic threshold modulation of biological neurons[6]. Let the input at time step t be $c _ { t } .$ , and the membrane potential states be $v _ { d }$ (dendritic compartment) and $v _ { s }$ (somatic compartment). The update rules are as follows:

$$
v _ { d } [ t ] = \alpha _ { 1 } v _ { d } [ t - 1 ] + \beta _ { 1 } v _ { s } [ t - 1 ] + ( 1 - \alpha _ { 1 } ) c _ { t } - \gamma _ { 1 } s _ { d } [ t - 1 ]\tag{13}
$$

$$
v _ { s } [ t ] = \alpha _ { 2 } v _ { s } [ t - 1 ] + \beta _ { 2 } v _ { d } [ t ] + ( 1 - \alpha _ { 2 } ) c _ { t } - \gamma _ { 2 } s _ { s } [ t - 1 ] ]\tag{14}
$$

$$
s _ { d } [ t ] = H ( v _ { d } [ t ] - v _ { t h } ) , \quad s _ { s } [ t ] = H ( v _ { s } [ t ] - v _ { t h } )\tag{15}
$$

where $\alpha _ { 1 } , \alpha _ { 2 } , \beta _ { 1 } , \beta _ { 2 }$ are learnable decay factors, γ<sub>1</sub>, γ<sub>2</sub> are inter-compartment coupling strengths, and $v _ { t h }$ is the firing threshold:

$$
v _ { t h } = V _ { t h } \cdot ( 2 - g _ { t } )\tag{16}
$$

where $V _ { t h }$ is the baseline threshold, and $g _ { t }$ is an optional external gating signal (used for contextual modulation in this paper). $s _ { d } [ t ]$ and $s _ { s } [ t ]$ are the spike outputs of the dendritic and somatic compartments, respectively, and $H ( \cdot )$ is the Heaviside step function, with ATan surrogate gradients used in actual training. This design enables neurons to dynamically adjust firing sensitivity based on historical activity, lowering the threshold during explosive movements to enhance response and raising it at rest to suppress noise, thereby adapting to non-stationary motion rhythms.

![](images/6aeec2157f782357312d36c1c7f4b2b2530984e7481f290bf6286ce4697e391b.jpg)  
Figure 2. (a) Structural organization of neuronal signal transmission: focusing on axon-soma synapses and axondendrite synapses; (b) SNN neuron structure; (c) Circuit diagram and formula of the LIF model; (d) Time-series decomposition and spike generation in the TS-LIF model.

## 3.4. Temporal Dimension Readout and Classification Head

After L MS-Blocks, the feature tensor $X \in \mathbb { R } ^ { T \times B \times D \times V }$ is first averaged over the vertex dimension to aggregate information from all physical nodes:

$$
X _ { p o o l } = \frac { 1 } { V } \sum _ { v = 1 } ^ { V } X [ : , : , : , v ] \in \mathbb { R } ^ { T \times B \times D }\tag{17}
$$

Then, it passes through a TSLIF neuron to generate spike activations, followed by a linear classification head mapping to the class space:

$$
\hat { Y } = W _ { c l s } \cdot \mathrm { T S L I F } ( X _ { p o o l } ) \in \mathbb { R } ^ { T \times B \times N _ { c } }\tag{18}
$$

Finally, averaging over the temporal dimension yields the classification prediction for each sample:

$$
\boldsymbol { \hat { y } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \boldsymbol { \hat { Y } } [ t ] \in \mathbb { R } ^ { B \times N _ { c } }\tag{19}
$$

This readout strategy aggregates predictions over time, avoiding latency issues associated with relying only on the final time step, while further reducing computation via the sparsity of spiking neurons.

## 4. Experiments

To comprehensively evaluate TSFFormer, we conduct extensive experiments on eight public IMU benchmarks. Our evaluation is guided by the following research questions (RQs):

• RQ1 (Overall Performance): How does TSFFormer compare against state-of-the-art DNNs and SNNs across various scenarios?

• RQ2 (Training Eficiency): Can TSLIF neurons and Multi-Head Fourier attention efectively accelerate convergence and mitigate gradient degradation?

• RQ3 (Component Contribution): How much do the key designs (e.g., spiking, TSLIF, frequency-domain mixing) specifically contribute to the final performance?

## 4.1. Experimental Setup

## 4.1.1. Datasets

To ensure robust evaluation across diferent wearable scenarios, we selected 8 public IMU datasets covering multi-node, single-node, clinical, and elderly populations. Table 1 summarizes the detailed configurations of these datasets.

• Multi-node sensor environments: PAMAP2 [12], TNDA-HAR, and HuGaDB [4] contain multiple distributed sensor nodes, with TNDA-HAR comprising 50 subjects and 6 asymmetric sensor nodes, making it the largest multi-node benchmark.

• Complex human activities: WISDM[16] and OP-PORTUNITY cover various sensor types, capturing diverse activities in daily life.

• Single-node daily motion: USC-HAD[21] serves as a standard benchmark, using a single right front hip sensor to evaluate basic human motion.

• Elderly population: HAR70 [15] captures the nonstationary motion rhythms of the elderly, used to validate the dynamic threshold modulation capability of TSLIF.

## 4.1.2. Data Preprocessing

All datasets adopt a frequency-adaptive sliding window strategy, with window size and stride dynamically adjusted based on the original sampling rate to ensure an approximately 2-second physical observation window. The Daily-Sports dataset retains its original 5- second pre-segmented snippets. Input signals are normalized to zero mean and unit variance, and converted to $T \times B \times C \times V$ format to adapt to the model. To avoid data leakage caused by window-level random splitting, we adopt a strict subject-independent hold-out protocol for all datasets except WISDM: each dataset is split by unique subject IDs; WISDM, due to its many fine hand movements and large inter-person variance, uses random splitting, with 70% of subjects for training, 15% for validation, and 15% for testing. The test set contains only individuals unseen during training. All results are reported as accuracy scores to handle class imbalance.

## 4.2. Overall Recognition Performance Comparison

We compared TSFFormer with various state-of-the-art (SOTA) models, mainly in two categories:

• Traditional Deep Neural Networks (ANN): DeepConvLSTM[14].

• State-of-the-art Spiking Neural Networks (SNN): LMUformer[10], QKformer[25], Simple-SNN, SpikeGRU, SpikeRNN, SpikeTCN, Spikeformer[26], STATten[8], TSSNN [20], SpikeDrivenTransformer[18], and its improved version SpikeDrivenTransformerV2[19].

Table 2 presents the recognition accuracy of all models across the eight benchmark datasets. TSFFormer establishes new SOTA performance in the vast majority of scenarios.

Comprehensively surpassing existing SNN architectures: TSFFormer achieves the best SNN performance on all 8 datasets. On the most challenging

Table 1. Summary of selected IMU datasets, with window size T and step size adaptively configured according to the original sampling rate.
<table><tr><td>Dataset</td><td>Split</td><td>Subjects</td><td>Classes</td><td>Window Size(T)</td><td>Stride</td><td>Model input</td><td>Sensor Nodes(V)</td></tr><tr><td>Daily-Sports</td><td>[0.70,0.15,0.15]</td><td>8</td><td>19</td><td>125</td><td>50</td><td> $3 \times 5$ </td><td>5</td></tr><tr><td>HAR70</td><td>[0.70,0.15,0.15]</td><td>18</td><td>7</td><td>100</td><td>50</td><td> $3 \times 2$ </td><td>2</td></tr><tr><td>HuGaDB</td><td>[0.70,0.15,0.15]</td><td>18</td><td>12</td><td>128</td><td>64</td><td> $6 \times 6$ </td><td>6</td></tr><tr><td>Opportunity</td><td>[0.70,0.15,0.15]</td><td>4</td><td>4</td><td>60</td><td>30</td><td> $1 1 4 \times 1$ </td><td>1</td></tr><tr><td>PAMAP2</td><td>[0.70,0.15,0.15]</td><td>9</td><td>12</td><td>200</td><td>100</td><td> $6 \times 3$ </td><td>3</td></tr><tr><td>TNDA</td><td>[0.70,0.15,0.15]</td><td>50</td><td>8</td><td>200</td><td>100</td><td> $3 \times 5$ </td><td>6</td></tr><tr><td>USC-HAD</td><td>[0.70,0.15,0.15]</td><td>14</td><td>12</td><td>200</td><td>100</td><td> $3 \times 1$ </td><td>1</td></tr><tr><td>WISDM</td><td>[0.70,0.15,0.15]</td><td>51</td><td>18</td><td>200</td><td>100</td><td> $6 \times 2$ </td><td>2</td></tr></table>

Table 2. Overall recognition performance (accuracy %) on eight diferent IMU datasets. Our model is indicated in bold, and the ANN model is underlined.
<table><tr><td>Type</td><td>Model</td><td>Daily-sports</td><td>HAR70</td><td>HuGaDB</td><td>Opportunity</td><td>PAMAP2</td><td>TNDA</td><td>USC-HAD</td><td>WISDM</td></tr><tr><td>ANN</td><td>DeepConvLSTM</td><td>86.22</td><td>90.60</td><td>89.73</td><td>94.10</td><td>81.16</td><td>92.28</td><td>75.72</td><td>71.23</td></tr><tr><td rowspan="10"></td><td>LMUformer</td><td>65.78</td><td>79.47</td><td>85.62</td><td>91.06</td><td>58.01</td><td>80.36</td><td>57.99</td><td>27.69</td></tr><tr><td>QKformer</td><td>68.46</td><td>85.35</td><td>86.17</td><td>88.99</td><td>64.56</td><td>81.75</td><td>61.81</td><td>41.63</td></tr><tr><td>Simple-SNN</td><td>52.54</td><td>77.22</td><td>81.47</td><td>88.75</td><td>57.40</td><td>71.84</td><td>51.26</td><td>36.62</td></tr><tr><td>SpikeGRU</td><td>6.57</td><td>48.50</td><td>29.01</td><td>45.43</td><td>12.35</td><td>14.36</td><td>11.37</td><td>12.07</td></tr><tr><td>SpikeRNN</td><td>64.07</td><td>86.91</td><td>89.55</td><td>90.45</td><td>64.60</td><td>79.78</td><td>53.93</td><td>20.07</td></tr><tr><td>SpikeTCN</td><td>58.64</td><td>86.93</td><td>80.83</td><td>89.17</td><td>59.12</td><td>85.48</td><td>65.87</td><td>52.43</td></tr><tr><td>Spikeformer</td><td>74.60</td><td>84.70</td><td>89.32</td><td>92.29</td><td>69.92</td><td>86.50</td><td>64.71</td><td>47.84</td></tr><tr><td>STATten</td><td>79.03</td><td>83.26</td><td>87.63</td><td>89.32</td><td>64.05</td><td>86.25</td><td>68.82</td><td>42.25</td></tr><tr><td>TSSNN</td><td>64.16</td><td>82.22</td><td>89.64</td><td>92.27</td><td>66.49</td><td>76.67</td><td>42.29</td><td>24.78</td></tr><tr><td>SpikeDrivenTransformer</td><td>78.20 79.82</td><td>84.34 86.21</td><td>79.56 89.73</td><td>92.43 88.62</td><td>68.26</td><td>77.03</td><td>72.02</td><td>34.01</td></tr><tr><td>SpikeDrivenTransformerV2</td><td></td><td></td><td></td><td></td><td></td><td>71.55</td><td>93.12</td><td>67.00</td><td>49.79</td></tr><tr><td>TSFFormer</td><td></td><td>87.89</td><td>90.54</td><td>90.42</td><td>93.79</td><td>83.33</td><td>98.53</td><td>78.65</td><td>86.35</td></tr></table>

WISDM dataset, our model (86.35%) achieves an absolute accuracy leap of 33.92% compared to the secondbest SpikeTCN (52.43%); on the TNDA dataset, it improves by 5.41% compared to SpikeDrivenTransformerV2 (93.12%). This fully demonstrates the significant advantages of TSLIF dynamic neurons and Multi-Head Fourier attention in capturing long spatiotemporal dependencies and complex kinematic features.

More remarkably, breaking the accuracy barrier between ANN and SNN: TSFFormer not only surpasses all SNN baselines but also comprehensively defeats the best continuous DNN model, DeepConvLSTM, on datasets including Daily-sports (87.89% vs. 86.22%), PAMAP2 (83.33% vs. 81.16%), TNDA (98.53% vs. 92.28%), and WISDM (86.35% vs. 71.23%). This breaks the traditional stereotype that SNNs struggle to balance high accuracy and low power, proving that spiking architectures also possess powerful representation capabilities.This fully address RQ1 and demonstrates the superiority of TSFFormer.

## 4.3. Training Efficiency and Convergence Analysis

To investigate the training dynamics of TSFFormer and address RQ2, we compared the loss descent trajectories (Fig. 3a) and validation accuracy curves (Fig. 3b) of various models on the representative WISDM dataset.

Our model exhibits extremely fast convergence and excellent gradient propagation: As shown in Fig. 3(a), TSFFormer (Ours, red) experiences a clif-like drop in loss values in the early training stage (<10 Epochs), rapidly converging to around 0.5. In contrast, traditional SNNs (e.g., Simple-SNN, Spike-RNN) and some Transformer architectures (e.g., SpikeDriven-Transformer) show extremely slow loss descent and struggle to converge to an ideal state. This indicates that the dual-compartment dynamics and threshold modulation mechanisms in TSLIF neurons efectively alleviate the common gradient degradation issue in deep SNN training via BPTT.

Furthermore, our model exhibits high and stable validation accuracy: In Fig. 3(b), the validation accuracy of TSFFormer rapidly climbs and stabilizes at a high level within a very short number of Epochs, without the severe oscillations or low accuracy traps common in early SNN models. This fast convergence property has extremely important engineering value for practical deployment and fine-tuning on edge devices.

![](images/2c9ff18a2dda2f4c9ae389a0d132c299ca75a0e2a28979803904e5d703439dc4.jpg)  
Figure 3. Training loss curves and validation accuracy curves of the twelve models on the WISDM dataset.

In addition, we not only compared various SNN baselines but also selected the highly representative continuous DNN, DeepConvLSTM, for a strict end-toend training dynamics comparison on the same dataset. As shown in Fig. 4, TSFFormer demonstrates crushing convergence advantages and a high representation ceiling.

As shown in the training loss curve in Fig. 4(a) and the convergence speed comparison in Fig. 4(d), TSFFormer loss undergoes a clif-like drop in the early training stage, requiring only 5 Epochs to break through the 50% validation accuracy threshold. In contrast, DeepConvLSTM takes a lengthy 35 Epochs to reach the same level, meaning r the early convergence speed of TSFFormer is 7 times that of the traditional DNN. This is not only due to the eficient feature routing brought by frequency-domain global mixing but also proves that the dynamic dual-compartment mechanism of TSLIF neurons efectively mitigates gradient vanishing in deep networks during temporal propagation.

From the validation accuracy trajectory in Fig. 4(b), the accuracy curve of TSFFormer rapidly climbs and stabilizes at a high level without the severe oscillations common in traditional DNNs. Ultimately, as shown in Fig. 4(c), TSFFormer achieves a best validation accuracy of 87.37%, while DeepConvLSTM remains at 71.95%, giving TSFFormer an absolute accuracy improvement of 15.42%.This overturns the long-held assumption that SNNs cannot match the accuracy of traditional DNNs, underscoring the immense viability of purely event-driven paradigms for wearable sensing.

![](images/705f6a10c99b8b7e223baed690ca70b8503d9f29e9de778aa369752a73bfae4d.jpg)  
(a) Training Loss

![](images/397e424723b17ea8e300842d888529bccb0f8434b3a50f5df00d1d16a273fa94.jpg)  
(b) Validation Accuracy

![](images/9924eb3d5f9145bd431a59347514b6e144308d5c7a84f3569716cf3ade03be20.jpg)

![](images/37169ed70c0e935d8322b1d1e3d71db3816a707ab1ad0de7a928a0aaa41c7133.jpg)  
Figure 4. Comparison of training dynamics and performance between TSFFormer and DeepConvLSTM on WISDM: (a) Training loss curve; (b) Accuracy trajectory of the validation set; (c) Comparison of the best validation accuracy; (d) Number of Epochs required to reach 50% accuracy (convergence speed).

## 4.4. Ablation Study

To rigorously evaluate the efectiveness of the core architecture of TSFFormer and address RQ3, we constructed four progressive variants for strict ablation studies.

• TSFFormer (Ours):The complete proposed model;

• w/o MultiHeadFourier: replacing the frequencydomain mixing mechanism with standard spatial self-attention;

• w/o TSLIFNode: replace the dual-compartment dynamic neuron with a fixed-threshold standard LIF neuron;

• w/o Both (pure baseline SNN): removing both the frequency-domain mechanism and the dynamic neuron.

We comprehensively evaluated them on two dimensions: recognition accuracy and training dynamics (convergence speed and computational overhead), with results shown in Tables 3 and 4.

## 4.4.1. Effectiveness of TSLIFNode

Removing TSLIFNode caused the most severe performance degradation. As shown in Table 3, on the highly dynamic WISDM dataset, accuracy plummeted from 86.35% to 63.85% (a drop of over 20%); and from Table 4, the $\mathrm { w / o }$ TSLIFNode model even failed to reach the 95% accuracy line (peaking at only 92.71%).

Table 3. Ablation experiment results of core components on eight benchmark datasets (accuracy %).
<table><tr><td>Model</td><td>Daily-Sports</td><td>HAR70</td><td>HuGaDB</td><td>Opportunity</td><td>PAMAP2</td><td>TNDA</td><td>USC-HAD</td><td>WISDM</td></tr><tr><td>TSFFormer</td><td>87.89</td><td>90.54</td><td>90.42</td><td>93.79</td><td>83.33</td><td>98.25</td><td>78.65</td><td>86.35</td></tr><tr><td>w/o MultiHeadFourier</td><td>86.09</td><td>88.95</td><td>88.50</td><td>92.25</td><td>80.85</td><td>96.89</td><td>78.94</td><td>81.18</td></tr><tr><td> $\mathrm { w / o }$  TSLIFNode</td><td>80.13</td><td>88.03</td><td>90.14</td><td>91.85</td><td>79.76</td><td>93.12</td><td>71.39</td><td>63.85</td></tr><tr><td>w/o Both</td><td>64.03</td><td>83.94</td><td>86.86</td><td>87.96</td><td>61.48</td><td>77.36</td><td>54.14</td><td>26.07</td></tr></table>

Table 4. Comparison of training eficiency, convergence, and system resource consumption of the model on the TNDA dataset.
<table><tr><td>Model configuration</td><td>Epochs to 95%</td><td>Mean epoch time (s)</td><td>Peak GPU memory (MB)</td><td>Best validation (%)</td><td>Time to best validation (h)</td></tr><tr><td>TSFFormer</td><td>22</td><td>188.5</td><td>10905</td><td>96.60</td><td>~2.6</td></tr><tr><td> $\mathrm { w / o }$  MultiHeadFourier</td><td>39</td><td>567.2</td><td>20713</td><td>95.45</td><td>~12.4</td></tr><tr><td>w/o TSLIFNode</td><td>Not Achieved</td><td>112.2</td><td>7968</td><td>92.71</td><td>~1.34</td></tr><tr><td>w/o Both</td><td>Not Achieved</td><td>294.91</td><td>10334</td><td>78.19</td><td>~7.13</td></tr></table>

This phenomenon profoundly reveals that the fixed threshold of standard LIF neurons is completely incapable of handling non-stationary, high-burst human motion rhythms, easily leading to “spike explosions” or loss of resting features. Although TSLIF introduces dual-compartment dynamics that slightly increase perepoch time to 188.5s, it brings a massive improvement in convergence speed (reaching 95% in just 22 epochs), making it the core engine for high robustness.

## 4.4.2. Impact of MultiHeadFourier

Removing MultiHeadFourier also leads to general accuracy drops (e.g., HuGaDB dropping by 3.88%, PAMAP2 by 2.48%). More critically, Table 4 shows that traditional spatial dot-product attention incurs catastrophic computational overhead: its per-epoch training time soars to 567.2s, and peak memory usage reaches 20,713 MB. In contrast, MultiHeadFourier uses the Fast Fourier Transform (O(V log V) complexity) to perform element-wise multiplication in the frequency domain, increasing training speed by 3 times, saving nearly 50% of memory, and requiring only 22 epochs to break through 95% accuracy. This fully proves that the frequency-domain mechanism, while capturing periodic topological dependencies between multiple sensors, perfectly aligns with the “green and low-power” design philosophy. (Note: On the single-node USC-HAD dataset, removing FFT actually yields a slight increase of 0.29%. This is mainly because a single node lacks complex spatial topology, and frequency-domain global mixing may introduce slight redundancy; while removing TSLIF on this dataset causes a plunge of 7.26%, once again confirming the fundamental role of temporal dynamic modulation.)

## 4.4.3. Pure Baseline

The w/o Both pure baseline model (using standard self-attention and fixed-threshold LIF) shows complete collapse across all metrics. In Table 3, it achieves catastrophic accuracies of only 61.48% and 26.07% on PAMAP2 and WISDM, respectively; in Table 4, it not only fails to reach 95% accuracy (peaking at only 78.19%), but due to gradient degradation and the superposition of attention matrices, its per-epoch time (294.9s) is even slower than the full model.

This controlled variable experiment forms a perfect logical loop: TSLIFNode provides fine-grained spike sparsification and adaptive modulation in the temporal dimension, while MultiHeadFourier provides a global spatial receptive field in the frequency domain. Neither can be dispensed with; only by deeply combining the two can we simultaneously achieve extremely fast convergence ( 2.6 hours), ultra-low memory usage ( 10.9 GB), and the highest classification accuracy (96.60%) under the strict spiking (pure AC operation) paradigm.

## 5. Conclusion

We propose an end-to-end trained purely spiking architecture, TSFFormer, which applies frequency-domain global mixing [7] and dynamic neural modulation[6] to efectively balance computational cost and recognition performance. The performance of TSFFormer surpasses state-of-the-art baselines in the IMU-HAR domain (even outperforming dense DNNs), while featuring higher convergence speed (3 faster) and lower memory usage ( 50% savings). Furthermore, we incorporate MultiHeadFourier attention and dualcompartment TSLIFNode for spatio-temporal feature extraction, resulting in a highly robust event-driven model that exhibits excellent accuracy on eight benchmark datasets while retaining the ultra-low power consumption of pure AC operations. Our frequencyspiking hybrid structure is highly promising for deployment on resource-constrained wearables and eventdriven neuromorphic processors. We believe that the combination of frequency-domain attention and dynamic spiking dynamics holds promise for migration to broader multimodal time-series tasks, and the proposed TSFFormer is of significant value for promoting the development of Green HAR.

## References

[1] Guillaume Bellec, Darjan Salaj, Anand Subramoney, Robert Legenstein, and Wolfgang Maass. Long shortterm memory and learning-to-learn in networks of spiking neurons. Advances in Neural Information Processing Systems, 31, 2018. 2

[2] Guillaume Bellec, Franz Scherr, Anand Subramoney, Elias Hajek, Darjan Salaj, Robert Legenstein, and Wolfgang Maass. A solution to the learning dilemma for recurrent networks of spiking neurons. Nature Communications, 11(1):3625, 2020. 1, 2

[3] Cristiano Capone, Cosimo Lupo, Paolo Muratore, and Pier Stanislao Paolucci. Beyond spiking networks: The computational advantages of dendritic amplification and input segregation. Proceedings of the National Academy of Sciences, 120(49):e2220743120, 2023. 2

[4] Roman Chereshnev and Attila Kertész-Farkas. HuGaDB: Human gait database for activity recognition from wearable inertial sensor networks. In International Conference on Analysis of Images, Social Networks and Texts, pages 131–141. Springer, 2017. 5

[5] Wei Fang, Yanqi Chen, Jianhao Ding, Zhaofei Yu, Timothée Masquelier, Ding Chen, Liwei Huang, Huihui Zhou, and Yonghong Tian. Spikingjelly: An open-source machine learning infrastructure platform for spike-based intelligence. Science Advances, 9(40): eadi1480, 2023. 2

[6] Shibo Feng, Wanjin Feng, Xingyu Gao, Peilin Zhao, and Zhiqi Shen. TS-LIF: A temporal segment spiking neuron network for time series forecasting. In International Conference on Learning Representations (ICLR), pages 28064–28084, 2025. 1, 2, 4, 8

[7] Bingzheng Gan, Tianyi Zhang, Yusu Li, Jing Huang, Wei Shi, Yangkai Ding, and Tao Yu. Caracal: Causal architecture via spectral mixing. arXiv preprint arXiv:2605.00292, 2026. 1, 2, 3, 8

[8] Donghyun Lee, Yuhang Li, Youngeun Kim, Shiting Xiao, and Priyadarshini Panda. Spiking transformer with spatial-temporal attention. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13948–13958. IEEE, 2025. 2, 5

[9] James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santiago Ontanon. FNet: Mixing tokens with fourier transforms. arXiv preprint arXiv:2105.03824, 2021. 2

[10] Zeyu Liu, Gourav Datta, Anni Li, and Peter Beerel. LMUFormer: Low complexity yet powerful spiking model with legendre memory units. In International Conference on Learning Representations (ICLR), pages 12108–12122, 2024. 5

[11] Nicos A Nicola and Jefrey J Babon. Leukemia inhibitory factor (LIF). Cytokine & Growth Factor Reviews, 26(5):533–544, 2015. 2

[12] Attila Reiss and Didier Stricker. Introducing a new benchmarked dataset for activity monitoring. In 2012 16th International Symposium on Wearable Computers, pages 108–109. IEEE, 2012. 5

[13] Mutegeki Ronald, Alwin Poulose, and Dong Seog Han. iSPLInception: An inception-ResNet deep learning architecture for human activity recognition. IEEE Access, 9:68985–69001, 2021. 2

[14] Satya P Singh, Madan Kumar Sharma, Aimé Lay-Ekuakille, Deepak Gangwar, and Sukrit Gupta. Deep ConvLSTM with self-attention for human activity decoding using wearable sensors. IEEE Sensors Journal, 21(6):8575–8582, 2020. 2, 5

[15] Astrid Ustad, Aleksej Logacjov, Stine Øverengen Trollebø, Pernille Thingstad, Beatrix Vereijken, Kerstin Bach, and Nina Skjæret Maroni. Validation of an activity type recognition model classifying daily physical behavior in older adults: The HAR70+ model. Sensors, 23(5):2368, 2023. 5

[16] Gary Weiss. WISDM Smartphone and Smartwatch Activity and Biometrics Dataset. UCI Machine Learning Repository, 2019. DOI: https://doi.org/10.24432/C5HK59. 5

[17] Huatao Xu, Pengfei Zhou, Rui Tan, Mo Li, and Guobin Shen. LIMU-BERT: Unleashing the potential of unlabeled data for IMU sensing applications. In Proceedings of the 19th ACM Conference on Embedded Networked Sensor Systems, pages 1–14, 2021. 2

[18] Man Yao, Jiakui Hu, Zhaokun Zhou, Li Yuan, Yonghong Tian, Bo Xu, and Guoqi Li. Spike-driven transformer. Advances in Neural Information Processing Systems, 36:64043–64058, 2023. 2, 5

[19] Man Yao, JiaKui Hu, Tianxiang Hu, Yifan Xu, Zhaokun Zhou, Yonghong Tian, Bo Xu, and Guoqi Li. Spike-driven transformer V2: Meta spiking neural network architecture inspiring the design of nextgeneration neuromorphic chips. In International Conference on Learning Representations (ICLR), pages 52885–52907, 2024. 5

[20] Kairong Yu, Tianqing Zhang, Qi Xu, Gang Pan, and Hongwei Wang. TS-SNN: Temporal shift module for spiking neural networks. arXiv preprint arXiv:2505.04165, 2025. 5

[21] Mi Zhang and Alexander A Sawchuk. USC-HAD: A daily activity dataset for ubiquitous activity recognition using wearable sensors. In Proceedings of the 2012 ACM Conference on Ubiquitous Computing, pages 1036–1043, 2012. 5

[22] Xiyuan Zhang, Diyan Teng, Ranak R Chowdhury, Shuheng Li, Dezhi Hong, Rajesh K Gupta, and Jingbo

Shang. UniMTS: Unified pre-training for motion time series. Advances in Neural Information Processing Systems, 37:107469–107493, 2024. 1, 2

[23] Hanle Zheng, Zhong Zheng, Rui Hu, Bo Xiao, Yujie Wu, Fangwen Yu, Xue Liu, Guoqi Li, and Lei Deng. Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics. Nature Communications, 15(1):20, 2024. 2

[24] Naichuan Zheng, Hailun Xia, Zepeng Sun, Weiyi Li, and Yinzhe Zhou. Towards green wearable computing: A physics-aware spiking neural network for energyeficient IMU-based human activity recognition. arXiv preprint arXiv:2604.10458, 2026. 1

[25] Chenlin Zhou, Han Zhang, Zhaokun Zhou, Liutao Yu, Liwei Huang, Xiaopeng Fan, Li Yuan, Zhengyu Ma, Huihui Zhou, and Yonghong Tian. QKFormer: Hierarchical spiking transformer using QK attention. Advances in Neural Information Processing Systems, 37: 13074–13098, 2024. 2, 5

[26] Zhaokun Zhou, Yuesheng Zhu, Chao He, Yaowei Wang, Shuicheng Yan, Yonghong Tian, and Li Yuan. Spikformer: When spiking neural network meets transformer. arXiv preprint arXiv:2209.15425, 2022. 5

[27] Zhaokun Zhou, Yuesheng Zhu, Chao He, Yaowei Wang, Shuicheng Yan, Yonghong Tian, and Li Yuan. Spikformer: When spiking neural network meets transformer. In International Conference on Learning Representations (ICLR), 2023. 2