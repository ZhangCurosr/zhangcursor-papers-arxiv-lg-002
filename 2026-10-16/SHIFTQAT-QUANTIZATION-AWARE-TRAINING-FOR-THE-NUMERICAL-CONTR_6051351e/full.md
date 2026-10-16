# SHIFTQAT: QUANTIZATION-AWARE TRAINING FOR THE NUMERICAL CONTRACT OF A FIXED-FUNCTION EDGE CNN ACCELERATOR

Oguzhan Buyuksolak <sup>1</sup> Osman Erman Okman <sup>1</sup> Mehmet Gorkem Ulkar <sup>1</sup>

## ABSTRACT

Fixed-function CNN accelerators make edge inference fast and energy-eficient, but they execute only networks that obey a precise numerical contract. On the MAX7800x family, this contract requires power-of-two, per-tensor, symmetric scales, 8-bit activations and biases, integer-only arithmetic, and one statically configured shift per layer. We present ShiftQAT, the quantization-aware training (QAT) flow we developed for this family, which produces networks satisfying the contract by construction. It is the default training path of ai8x-training, the family’s opensource training toolchain. ShiftQAT resolves residual, concatenation, and weight-sharing dependencies so that activation scales propagate statically and each layer’s requantization reduces to one ofline-configured shift. It also trains weights and biases on a joint power-of-two grid that fits the 8-bit bias budget. Among the compared methods, only ShiftQAT satisfies all of the contract’s scaling constraints, including signed 8-bit activation storage. Even so, on ImageNet it matches the W8A8 accuracy drop of less constrained QAT methods and stays within 2.9 percentage points of FP32 at W4A8. On MAX78002 silicon, ShiftQAT-trained MobileNetV2 and ResNet-18 classify ImageNet images in 14.6 and 8.4 ms at 5.1 and 2.8 mJ per inference.

## 1 INTRODUCTION

Deep neural networks are increasingly deployed on batterypowered edge devices, close to the sensors that produce their inputs. General-purpose microcontrollers ofer only a few megabytes of memory and limited arithmetic throughput. This has motivated software stacks for running quantized models on MCUs (Lai et al., 2018; David et al., 2021; Lin et al., 2020), as well as dedicated low-power CNN accelerators integrated alongside the MCU core (Giordano et al., 2022; Huang et al., 2024). Quantization is the enabling step for both, because it shrinks the memory footprint and lets inference run in fixed-point arithmetic (Gholami et al., 2022; Nagel et al., 2021).

The quantization literature usually treats hardware friendliness as a spectrum of desirable properties, such as uniform, symmetric, power-of-two, and per-tensor scaling. Methods are then compared by how much accuracy they recover at a given bit-width. Fixed-function accelerators turn these properties into hard requirements. We call the complete set of arithmetic rules that an accelerator’s datapath implements its numerical contract. A network either satisfies this contract or cannot be compiled for the device at all. On Analog Devices’ MAX7800x family (Analog Devices Inc., 2023), the contract requires power-of-two, per-tensor, symmetric scales, 8-bit storage for activations and biases, and integer-only inference. Each layer has a single shift that is configured ofline and cannot rescale dynamically where branches meet (Section 3). Prior hardware-friendly methods each satisfy part of this contract, but none of those we compare satisfies all of it (Tables 2 and 3).

This paper presents ShiftQAT, a QAT flow that produces networks satisfying the full contract by construction. We developed ShiftQAT at Analog Devices as the default quantization-aware training path of ai8x-training, the opensource training toolchain for the MAX7800x family. This paper documents its design, the datapath constraint behind each of its mechanisms, and its evaluation on ImageNet and on MAX78002 silicon. Each of its mechanisms answers a specific constraint of the datapath, and two of them carry the design. The first is static scale propagation. Because the datapath cannot rescale dynamically, ShiftQAT resolves all rescaling before deployment. It folds each layer’s activation scale forward into the next layer’s bias and shift, so that intermediate layers never recover real-valued scale. This requires positively homogeneous activations and a scaleconsistent graph. ShiftQAT therefore resolves every residual addition, concatenation, and shared-weight layer so that propagation remains exact on real topologies. The second is a joint power-of-two weight/bias scale, re-estimated at every training iteration. Once biases have absorbed upstream activation scales and must still fit in 8 bits, weights and biases compete for the same power-of-two grid. Training on that grid lets the network adapt to it. The per-layer shift that the hardware applies is computed by training itself. A trained checkpoint therefore maps deterministically to integer weights, 8-bit biases, and shift values, with no separate calibration step.

## Our contributions are:

• We formalize the numerical contract of a fixedfunction edge CNN accelerator as seven datapath constraints (Table 1). None of the prior methods we compare satisfies all of them.

• We present ShiftQAT, which satisfies the full contract through topology dependency resolution, static scale propagation, and joint power-of-two weight/bias scaling, specified end-to-end in Algorithm 1.

• Under the contract’s scaling constraints, ShiftQAT matches the ImageNet accuracy drop of less constrained QAT methods at W8A8 and stays within 2.9 pp of FP32 at W4A8 on MobileNetV2 and ResNet-18. We quantify the gap to QAT methods that relax the contract at 4 bits.

• We deploy ShiftQAT-trained MobileNetV2 and ResNet-18 end-to-end on MAX78002 silicon and report measured latency, energy, and memory for ImageNet-scale inference on a microcontroller-class device.

• We distill five lessons for building training flows for fixed-function accelerators (Section 6). The complete flow is available as open source in ai8x-training.

Code availability. The code is published at ai8xtraining, https://github.com/analogdevicesinc/ ai8x-training.

## 2 BACKGROUND AND RELATED WORK

Integer-only inference and dyadic requantization. Jacob et al. (2018) formalized integer-only inference. Each layer’s real-valued output multiplier is decomposed ofline into a fixed-point mantissa and a power-of-two shift, and biases are stored as 32-bit integers at the accumulator’s scale. HAWQ-V3 (Yao et al., 2021) restricts these multipliers to dyadic numbers, and TensorRT (Migacz, 2017) and CMSIS-NN (Lai et al., 2018) provide calibrated or Qformat fixed-point kernels for GPUs and Cortex-M CPUs. Figure 1a shows this per-layer recovery structure. On processors with a general integer multiplier it is inexpensive. A fixed-function accelerator changes where the rescaling can happen. It can only be a statically configured power-of-two shift per layer, never a multiplication by an arbitrary mantissa.

Moving scales across layers. An activation f is positively homogeneous if $f ( s z ) ~ = ~ s f ( z )$ for all $s > 0 ,$ , as holds for ReLU and Leaky ReLU. For such activations, a perchannel or per-tensor scale can be moved through the activation into the next layer’s weights. Data-free quantization (Nagel et al., 2019) exploits this property for cross-layer weight equalization. ShiftQAT uses the same identity for a diferent purpose. It defers every activation scale to the network output, so that intermediate layers never recover realvalued scale. It adds the topology handling (Section 4.3) and the bias treatment (Section 4.5) needed to make this hold under an 8-bit bias budget on real CNNs.

Hardware-friendly quantizers. PACT (Choi et al., 2018), FAQ (McKinstry et al., 2019), LSQ (Esser et al., 2020), and diferentiable range/step-size methods (Uhlich et al., 2020) learn clipping ranges or step sizes during QAT. TQT (Jain et al., 2020) trains power-of-two per-tensor thresholds, and APoT (Li et al., 2020) uses sums of powers of two as non-uniform levels. Vandersteegen et al. (2021) combine per-layer power-of-two scales with 4-bit weights, optionally through a lookup table that maps 4-bit codes to non-uniform 8-bit weights. Post-training methods such as AdaRound (Nagel et al., 2020), BRECQ (Li et al., 2021), HPTQ (Habi et al., 2021), and EPTQ (Gordon et al., 2024) avoid retraining. The strongest of them rely on per-channel scales, which the MAX7800x datapath does not provide. HPTQ is closest in spirit to ShiftQAT. It targets uniform, symmetric, power-of-two quantization, and ShiftQAT adopts its activation-threshold search (Section 4.2). Lower-bit and binary/ternary schemes (Hubara et al., 2016; Zhou et al., 2018; Zhang et al., 2018; Jung et al., 2019; Yang et al., 2021) pursue compression beyond the 8/4- bit regime studied here.

Edge CNN accelerators. The MAX78000 and MAX78002 integrate a CNN accelerator with 1/2/4/8- bit weights and 8-bit activations alongside a Cortex-M4 core. Independent studies have characterized their latency and energy on small vision and audio workloads (Wang et al., 2022; Giordano et al., 2022; Moosmann et al., 2023; Huang et al., 2024; Dabbous et al., 2025). This paper describes the training side, namely the QAT flow of the family’s own toolchain, together with ImageNet-scale deployments on it.

## 3 THE ACCELERATOR’S NUMERICAL CONTRACT

Table 1 lists the constraints a network must satisfy to run on the MAX7800x CNN accelerator, together with the ShiftQAT mechanism each constraint forces. Appendix A expands on each property.

Why generic QAT output does not compile. A network produced by standard integer-only QAT (Jacob et al., 2018) typically violates C2 and C4 (non-power-of-two multipliers, 32-bit biases), and often C1 (asymmetric activations). Perchannel PTQ methods (Li et al., 2021; Gordon et al., 2024) additionally violate C3. TQT (Jain et al., 2020) comes closest. It uses power-of-two, per-tensor scales and merges the input scales of element-wise additions and concatenations, which addresses C7. However, it quantizes biases to 16 bits and switches to unsigned activations after ReLU, which violates C4 and C5. HPTQ (Habi et al., 2021) relies on perchannel weight scales, which violates C3. The bias constraint C4 interacts with C7. Once activation scales are folded forward (C6, C7), biases absorb those scales and must still fit in 8 bits. Satisfying the constraints one at a time therefore does not satisfy them jointly, which is why ShiftQAT treats the contract as a whole.

Table 1. The numerical contract of the MAX7800x CNN accelerator, as specified in the public MAX7800x documentation, and the ShiftQAT mechanism each constraint requires. Constraints C1– C7 are properties of the datapath, not design preferences.
<table><tr><td>Constraint</td><td>Required mechanism</td></tr><tr><td>C1</td><td>Signed fixed-point (Q7) num- Symmetric quantizers ber format; uniform, symmet- everywhere (§4.2, §4.5) ric, no zero-point</td></tr><tr><td>C2</td><td>Rescaling only by  $2 ^ { k } , \ k \in \ \mathrm { P o T }$  scales for  $S _ { x } , \ S _ { w }$  [−15, 15], applied to the 32-bit (§4.2, §4.5)</td></tr><tr><td>C3</td><td>accumulator One output shift per layer (per- Per-tensor  $S _ { x } , S _ { w }$  tensor)</td></tr><tr><td>C4</td><td>Biases stored in 8 bits; weights Joint weight/bias scale in 8/4/2/1 bits (§4.5)</td></tr><tr><td>C5</td><td>Activations stored as signed 8- Signed symmetric activa- bit; ReLU outputs clipped to tion quantization [0, 127]</td></tr><tr><td>C6</td><td>No floating point in the infer- All scales folded offline ence path (§4.4)</td></tr><tr><td>C7</td><td>Shift configured statically; Dependency resolution + element-wise operations com- propagation (§4.3, §4.4) bine inputs without per-input rescaling</td></tr></table>

What the contract rules out. Two constraints remove accuracy levers that most published methods rely on. C3 forgoes per-channel scales, which the strongest PTQ results depend on. C5 means that post-ReLU tensors use only 7 of their 8 bits. The hardware clips ReLU outputs to [0, 127] in its signed 8-bit format, so half of the code space is unused for non-negative tensors. ShiftQAT recovers accuracy within these limits through training, as Section 5.2 shows.

## 4 SHIFTQAT MECHANISMS REQUIRED BY THE CONTRACT

Figure 2 and Algorithm 1 summarize the flow. It starts from an FP32 checkpoint and runs an ofline initialization consisting of batch-norm folding, activation-scale proposal, dependency resolution, and propagation. QAT follows, and the weight/bias scale is re-estimated at every iteration. Throughout, $n _ { x } , n _ { w } ,$ and $n _ { b }$ denote the bit-widths of activations, weights, and biases, and every scale $S = 2 ^ { s }$ is a power of two with integer exponent s.

Algorithm 1 ShiftQAT, ofline initialization and QAT loop   
Require: FP32 model M, training set D, bit-widths $n _ { x } , n _ { w } , n _ { b }$   
1: // Ofline initialization   
2: Fold batch normalization into preceding layers   
3: for each layer ℓ do   
4: Collect activations on $\mathcal { D } ;$ drop z-score outliers   
5: $S _ { \mathrm { n c } } $ smallest PoT covering the remaining range   
6: $S _ { x , \ell } \gets \mathrm { a r g }$ min MSE over $\breve { S } _ { \mathrm { n c } } / 2 ^ { i } , i = 0 , \ldots , 4 \ : ( \mathrm { E q . } \ : 3 )$   
7: end for   
8: Resolve dependencies (§4.3). Residual inputs take the add’s   
scale, and concatenation and shared-weight inputs take the   
maximum scale   
9: for each layer ℓ in topological order, with predecessor p do   
10: $b _ { \ell } \gets \bar { b \ell } / S _ { x , p } ; \quad \mathbf { \hat { { S } } } _ { x , \ell } ^ { \prime } \gets S _ { x , \ell } / S _ { x , p }$ (Eq. 5)   
11: end for   
12: // Quantization-aware training   
13: for each training iteration do   
14: for each layer ℓ do   
15: $\begin{array} { r l } { \lambda _ { \ell } } & { { }  } \end{array}$ q-quantile of $| \{ W _ { \ell } , b _ { \ell } \} | ; s _ { w , \ell } \gets$   
$\mathrm { c l i p } ( \lceil \log _ { 2 } \lambda _ { \ell } \rceil ; - 1 5 , 1 5 ) ( \mathrm { E q . ~ } 1 0 )$   
16: Quantize $W _ { \ell } \left( n _ { w } \right.$ bits), b (n bits) with $S _ { w , \ell } = 2 ^ { s _ { w , \ell } }$   
17: $k _ { \ell } \gets \mathrm { c l i p } ( s _ { w , \ell } - s _ { x , \ell } ^ { \prime } ; - 1 5 , 1 5 )$ (PoT scaler, Eq. 11)   
18: end for   
19: Forward pass with fake quantization; backward pass with   
the STE; update $W , b$   
20: end for   
Ensure: Integer weights, 8-bit biases, one shift per layer

## 4.1 Batch-Normalization Folding

Batch normalization (Iofe, 2015) is folded into the preceding convolution or linear layer before quantization, following Jacob et al. (2018). This is required by C6. The accelerator has no normalization stage, so its afine transform must be absorbed into the weights and biases that the contract governs.

## 4.2 Activation-Scale Proposal

For each layer we select a power-of-two activation scale (C2, C3) using the threshold search of HPTQ (Habi et al., 2021). Activation statistics are collected as 2048-bin histograms in one forward pass over the training set, and outliers are removed with a z-score threshold of 8. With $S _ { x } = 2 ^ { k } , k \in \mathbb { Z } ,$ the step size is $\Delta _ { x } = S _ { x } / 2 ^ { n _ { x } - 1 }$ , and activations are quantized symmetrically (C1, C5):

$$
X ^ { \prime } = \mathrm { c l i p } \big ( \lfloor X / \Delta _ { x } \rceil ; - 2 ^ { n _ { x } - 1 } , 2 ^ { n _ { x } - 1 } - 1 \big ) \cdot \Delta _ { x } ,\tag{1}
$$

where  denotes rounding to the nearest integer. The search starts from the smallest non-clipping power of two

![](images/8279ca975b16d6f917ab0fed4077cc44a562420f8496d01f2d73b208742bd34d.jpg)

Figure 1. Per-layer inference datapaths. (a) Per-layer scale recovery as in integer-only QAT (Jacob et al., 2018; Esser et al., 2020). The product $W ^ { \prime } X ^ { \prime }$ is rescaled by $S _ { w }$ and $S _ { x }$ before a full-precision bias is added, and the result is then requantized. (b) The structure the MAX7800x contract requires, as produced by ShiftQAT. Weights and biases share one power-of-two scale $S _ { w } ,$ and activation scales are resolved and propagated ofline. The only runtime rescaling is a single power-of-two shift per layer (the PoT scaler, Eq. 11).  
![](images/eb71931656f56dd2f3e961a938ffe0dba2ab9564ce1aec4b10b741de3ad1690d.jpg)  
Figure 2. The ShiftQAT flow. FP32 training is followed by ofline initialization, which consists of BN folding, activation-scale proposal, and activation-scale refinement (dependency resolution and propagation). QAT then runs with per-iteration weight/bias scale determination.

for the outlier-filtered activations,

$$
S _ { \mathrm { n c } } = 2 ^ { \lceil \log _ { 2 } \operatorname* { m a x } | X | \rceil } .\tag{2}
$$

It evaluates this scale and its first four halvings, and keeps the candidate that minimizes the quantization error weighted by the activation histogram:

$$
S _ { x } ^ { * } = \arg \operatorname* { m i n } _ { S _ { x } \in \{ S _ { \mathrm { n c } } / 2 ^ { i } : i = 0 , \ldots , 4 \} } \mathbb { E } \big [ ( X - X ^ { \prime } ) ^ { 2 } \big ] .\tag{3}
$$

## 4.3 Topology Dependency Resolution

Constraint C7 means that wherever two branches meet, their tensors must already share a scale, because the datapath cannot rescale one branch at runtime to match the other. Independently proposed per-layer scales do not have this property, so they must be reconciled before propagation. ShiftQAT resolves three cases (Figure 3).

Residual connections. For an element-wise addition with inputs from $L _ { 1 }$ and $L _ { 2 }$ (scales $S _ { x _ { 1 } } , S _ { x _ { 2 } } )$ and output scale $S _ { x _ { 3 } }$ , both inputs are reassigned $S _ { x _ { 3 } }$ . The two addends then share a grid, so the accelerator adds them directly and saturates the sum to the 8-bit range, with no runtime alignment.

Concatenation. The output range of a concatenation is set by its widest input, so all inputs adopt the maximum of their proposed scales. If $S _ { x _ { 1 } } > S _ { x _ { 2 } }$ , then $S _ { x _ { 2 } }  S _ { x _ { 1 } }$

Weight sharing. In FPN-style heads (Lin et al., 2016), one layer $L _ { 4 }$ may be invoked on inputs from $L _ { 1 } , L _ { 2 } ,$ , and $L _ { 3 }$ at diferent points in the network. A single set of weights and a single shift must then serve every invocation. The scale $S _ { x _ { 4 } }$ is proposed from the union of $\chi _ { 4 } { \mathrm { : } } $ outputs across invocations, and the inputs $\boldsymbol { L } _ { 1 } { - } \boldsymbol { L } _ { 3 }$ are assigned the maximum of their scales.

Sharing scales at additions and concatenations follows the practice of TQT (Jain et al., 2020). ShiftQAT additionally resolves weight sharing and then folds all scales forward (Section 4.4). In each case the rule favors the larger scale, or the scale of the merge point. This trades some resolution on the narrower branch for the guarantee that no branch clips.

![](images/6d82e50c81d21010d2ab9004d9eed617c398d8dead5f532a30f39b2f1e713923.jpg)  
Figure 3. Dependency resolution. (a) For a residual connection, both input layers adopt the scale of the element-wise addition. (b) For a concatenation, the inputs adopt the largest scale among them. (c) For weight sharing, all layers feeding the shared layer adopt the largest scale among them, so that one set of shared weights and one shift serve every invocation.

## 4.4 Static Scale Propagation

With a scale-consistent graph, each activation scale is folded into the next layer, so that no intermediate layer needs to recover real-valued scale (C6, C7). Consider two layers with activation $f ,$ where $O _ { 1 } = f ( X w _ { 1 } + b _ { 1 } )$ and $O _ { 2 } = f ( O _ { 1 } w _ { 2 } + b _ { 2 } )$ (Figure 4). Layer 1 emits its output normalized by its own scale:

$$
O _ { 1 } ^ { \prime } = \frac { f ( X w _ { 1 } + b _ { 1 } ) } { S _ { x _ { 1 } } } = f \bigg ( \frac { X w _ { 1 } + b _ { 1 } } { S _ { x _ { 1 } } } \bigg ) ,\tag{4}
$$

where the second equality holds because f is positively homogeneous and $S _ { x _ { 1 } } > 0$ . This is the only property of f the method uses, and it is why Sigmoid and Tanh are excluded (Section 7). Layer 2 then consumes $O _ { 1 } ^ { \prime }$ directly, with its bias and scale adjusted ofline:

$$
b _ { 2 } ^ { \prime } = { \frac { b _ { 2 } } { S _ { x _ { 1 } } } } , \qquad S _ { x _ { 2 } } ^ { \prime } = { \frac { S _ { x _ { 2 } } } { S _ { x _ { 1 } } } } ,\tag{5}
$$

$$
O _ { 2 } ^ { \prime } = f \bigg ( \frac { O _ { 1 } ^ { \prime } w _ { 2 } + b _ { 2 } ^ { \prime } } { S _ { x _ { 2 } } ^ { \prime } } \bigg ) = \frac { O _ { 2 } } { S _ { x _ { 2 } } } .\tag{6}
$$

Since all scales are powers of two, $S _ { x _ { 2 } } ^ { \prime }$ is also a power of two. Applied layer by layer in topological order, this leaves a single recovery $O = O ^ { \prime } \cdot S _ { x }$ at the network output. The Arm core applies it when it reads out the results, and it can be omitted entirely for argmax classification. The last layer may instead emit unclipped 32-bit outputs, which avoids saturating the logits to 8 bits.

## 4.5 Joint Weight/Bias Power-of-Two Scale

Propagation (Eq. 5) divides each bias by an upstream activation scale. This changes the biases’ dynamic range relative to the weights, and C4 requires the result to fit in 8 bits. If weights and biases were scaled independently, the datapath would need a separate bias alignment, which violates C2 and C3. ShiftQAT therefore quantizes each layer’s weights and biases with a single shared power-of-two scale $S _ { w } ,$ , each at its own bit-width:

![](images/ab8d2c570aaedcc5edd950192abaca941720e6bfe24765cad9537f95ef66324e.jpg)  
Figure 4. Static scale propagation on a two-layer network. On the left, each layer’s output carries its real-valued scale. On the right, each layer emits scale-normalized outputs, downstream biases and scales are adjusted ofline (Eq. 5), and a single recovery remains at the output. Activations are omitted from the diagram for clarity, but the derivation requires them to be positively homogeneous.

$$
\Delta _ { w } = \frac { S _ { w } } { 2 ^ { n _ { w } - 1 } } , \qquad \Delta _ { b } = \frac { S _ { w } } { 2 ^ { n _ { b } - 1 } } ,\tag{7}
$$

$$
W ^ { \prime } = \frac { \mathrm { c l i p } \big ( \lfloor W / \Delta _ { w } \big \rceil ; - 2 ^ { n _ { w } - 1 } , ~ 2 ^ { n _ { w } - 1 } - 1 \big ) } { 2 ^ { n _ { w } - 1 } } ,\tag{8}
$$

$$
b ^ { \prime } = \frac { \mathrm { c l i p } \big ( \lfloor b / \Delta _ { b } \big 7 ; - 2 ^ { n _ { b } - 1 } , 2 ^ { n _ { b } - 1 } - 1 \big ) } { 2 ^ { n _ { b } - 1 } } ,\tag{9}
$$

so that $W ^ { \prime } , b ^ { \prime } \in [ - 1 , 1 )$ At every training iteration, weights and biases are pooled into one distribution, and outliers are rejected by taking the q-quantile of their absolute values as the range bound λ. We use $q = 0 . 9 9 9$ for 8-bit layers and $q = 0 . 9$ for 4-bit layers. With only 16 levels, clipping the largest 10% of magnitudes buys more resolution for the bulk of the distribution than it costs in clipping error. The scale is then

$$
S _ { w } = 2 ^ { s _ { w } } , \qquad s _ { w } = \mathrm { c l i p } \big ( \lceil \log _ { 2 } \lambda \rceil ; - 1 5 , 1 5 \big ) ,\tag{10}
$$

where λ is the quantile-based absolute-value bound and the clipping range follows the hardware’s shift range (C2). Because $S _ { w }$ is recomputed as the parameters move, the network is trained on the grid it will be deployed on. Recomputing $S _ { w }$ instead of learning it avoids the extra machinery that learned power-of-two scales require to stay stable, such as threshold freezing and carefully bounded optimizer settings (Jain et al., 2020), and works with plain SGD. When weights and biases compete for the same power-oftwo range, training can shift magnitude between them rather than having one operand lose resolution at deployment. Gradients pass through the rounding with the straight-through estimator (Bengio et al., 2013).

## 4.6 The PoT Scaler Is the Per-Layer Shift

After quantization, a layer’s accumulator $A = X ^ { \prime } W ^ { \prime } + b ^ { \prime }$ carries the scale $S _ { w } ,$ , while its output must carry the propagated scale $S _ { x } ^ { \prime }$ . The required rescaling is therefore

$$
S _ { \mathrm { P o T } } = \frac { S _ { w } } { S _ { x } ^ { \prime } } = 2 ^ { k } , \qquad k = \mathrm { c l i p } \big ( s _ { w } - s _ { x } ^ { \prime } ; - 1 5 , 1 5 \big ) .\tag{11}
$$

where $s _ { x } ^ { \prime } \ = \ \log _ { 2 } S _ { x } ^ { \prime }$ Multiplying A by $2 ^ { k }$ is a single arithmetic shift, to the left for $k \ > \ 0$ and to the right for $k \ < \ 0 .$ The hardware applies it to the 32-bit intermediate result and accepts total shifts in [ 15, 15], so the flow clips k to that range during training. The activation is then applied, and the result is quantized with step $2 ^ { - ( n _ { x } - 1 ) }$ and saturated to $[ - 1 , 1 - 2 ^ { - ( \bar { n } _ { x } - 1 ) } ]$ . On MAX78002, this value is exactly the accelerator’s programmable per-layer output shift. The weight-quantization shift and the propagated activation-scale shift are merged into each layer’s single shift value. The PoT scaler is therefore not an extra operation that ShiftQAT adds. It is the training-time computation of the value the hardware register will be programmed with, and QAT keeps that value consistent with the evolving weights at every iteration.

## 5 EVALUATION

## 5.1 Setup

We quantize MobileNetV2 and ResNet-18 starting from the oficial PyTorch FP32 checkpoints (Ansel et al., 2024), trained on ImageNet at 224 224 (Deng et al., 2009). We evaluate two settings, W8A8 and W4A8. For a fair comparison with published results, all methods in Tables 2 and 3, including ShiftQAT, use 32-bit biases $( n _ { b } = 3 2 )$ , as is standard in the quantization literature. These tables therefore isolate the efect of the weight and activation constraints (C1–C3, C5–C7). The deployed networks in Section 5.3 use 8-bit biases $( n _ { b } = 8 )$ ), as the hardware requires (C4). QAT runs for the last 50 of 200 epochs for MobileNetV2 and for the last 30 of 120 epochs for ResNet-18. Appendix B gives the full hyperparameters. All ShiftQAT results come from a single run per configuration.

Baseline numbers are taken from the cited papers. Because the baselines use diferent FP32 references (e.g., 71.878% for our MobileNetV2 checkpoint versus 72.62% in EPTQ’s), we compare accuracy drops (∆) rather than absolute accuracy. We treat diferences below about 1 pp as comparable, given single runs and cross-paper references. We also mark each method’s regime, post-training (PTQ) or quantization-aware (QAT), because PTQ methods use far less compute than a QAT schedule. No baseline can be deployed on MAX78002 without violating part of the contract, so all comparisons are accuracy-only.

## 5.2 Accuracy Under the Contract’s Scaling Constraints

W8A8 (Table 2). ShiftQAT loses 1.026 pp on MobileNetV2 and 0.558 pp on ResNet-18. These drops match those of integer-only QAT (Jacob et al., 2018) on MobileNetV2 (1.0 pp), which uses asymmetric, non-powerof-two scaling, and of FAQ (McKinstry et al., 2019) on ResNet-18 (0.5 pp). At 8 bits, then, meeting all of the contract’s scaling constraints costs no measurable accuracy relative to comparable QAT methods that meet only part of them. HPTQ and LSQ report smaller drops, but HPTQ relies on per-channel scales and LSQ on non-power-of-two step sizes, neither of which the datapath can execute. TQT, which is also power-of-two and per-tensor, reports a 0.1 pp gain at W8A8. However, its results table leaves the W4A8 (INT4/INT8) entry blank for both MobileNetV1 and MobileNetV2, the only networks in that table without a reported INT4 result (Jain et al., 2020). ShiftQAT, by contrast, delivers W4A8 MobileNetV2 within 2.9 pp of FP32 under stricter constraints.

W4A8 (Table 3). With 4-bit weights, ShiftQAT loses 2.864 pp on MobileNetV2 and 1.904 pp on ResNet-18. This is less than AdaRound (3.37 and 2.46 pp), the per-tensor

Table 2. W8A8 ImageNet top-1 accuracy. Baseline MobileNetV2 results are as reported by HPTQ (Habi et al., 2021), and baseline ResNet-18 results are as reported by LSQ (Esser et al., 2020). A “+” marks a satisfied property and a “−” an unsatisfied one. The Static prop. column marks the static scale propagation of Section 4.4, which is this paper’s mechanism and so is satisfied only by ShiftQAT. The Regime column gives post-training (PTQ) or quantization-aware training (QAT). ShiftQAT is the only row satisfying every property column. All methods use 32-bit biases.
<table><tr><td>Model</td><td>Method</td><td>Regime</td><td>FP (%)</td><td>Q (%)</td><td>∆(pp)</td><td>Unif.</td><td>Sym.</td><td>PoT</td><td>Per-ten.</td><td>Static prop.</td></tr><tr><td>MobileNetV2</td><td>ShiftQAT</td><td>QAT</td><td>71.878</td><td>70.852</td><td>1.026</td><td>十</td><td>十</td><td>十</td><td>十</td><td>十</td></tr><tr><td></td><td>TQT (Jain et al., 2020)</td><td>QAT</td><td>71.7</td><td>71.8</td><td>-0.1</td><td>十</td><td>十</td><td>十</td><td>十</td><td>一</td></tr><tr><td></td><td>QAT (Jacob et al., 2018)</td><td>QAT</td><td>71.9</td><td>70.9</td><td>1.0</td><td>十</td><td>1</td><td>一</td><td>十</td><td></td></tr><tr><td></td><td>HPTQ (Habi et al., 2021)</td><td>PTQ</td><td>71.812</td><td>71.46</td><td>0.352</td><td>十</td><td>十</td><td>十</td><td>一</td><td>一</td></tr><tr><td>ResNet-18</td><td>ShiftQAT</td><td>QAT</td><td>69.758</td><td>69.2</td><td>0.558</td><td>十</td><td>十</td><td>十</td><td>十</td><td>十</td></tr><tr><td></td><td>LSQ (Esser et al., 2020)</td><td>QAT</td><td>70.5</td><td>71.1</td><td>-0.6</td><td>十</td><td>十</td><td>一</td><td>十</td><td>一</td></tr><tr><td></td><td>FAQ (McKinstry et al., 2019)</td><td>QAT</td><td>70.5</td><td>70.0</td><td>0.5</td><td>十</td><td>十</td><td>十</td><td>十</td><td></td></tr></table>

Table 3. W4A8 ImageNet top-1 accuracy. PTQ baseline results are as reported by EPTQ (Gordon et al., 2024). QAT baseline results are the per-tensor results of Nagel et al. (2021) (average of three runs). Columns as in Table 2. All methods use 32-bit biases.
<table><tr><td>Model</td><td>Method</td><td>Regime</td><td>FP (%)</td><td>Q (%)</td><td>∆(pp)</td><td>Unif.</td><td>Sym.</td><td>PoT</td><td>Per-ten.</td><td>Static prop.</td></tr><tr><td>MobileNetV2</td><td>ShiftQAT</td><td>QAT</td><td>71.878</td><td>69.014</td><td>2.864</td><td>十</td><td>十</td><td>十</td><td>十</td><td>十</td></tr><tr><td rowspan="5"></td><td>QAT (Nagel et al., 2021)</td><td>QAT</td><td>71.72</td><td>70.17</td><td>1.55</td><td>十</td><td></td><td>一</td><td>十</td><td>一</td></tr><tr><td>AdaRound (Nagel et al., 2020)</td><td>PTQ</td><td>72.62</td><td>69.25</td><td>3.37</td><td>十</td><td></td><td></td><td>十</td><td></td></tr><tr><td>BRECQ (Li et al., 2021)</td><td>PTQ</td><td>72.62</td><td>71.42</td><td>1.2</td><td>十</td><td></td><td></td><td>一</td><td>一</td></tr><tr><td>EPTQ (Gordon et al., 2024)</td><td>PTQ</td><td>72.62</td><td>72.06</td><td>0.56</td><td>十</td><td>十</td><td>一</td><td>一</td><td>一</td></tr><tr><td>ShiftQAT</td><td>QAT</td><td>69.758</td><td>67.854</td><td>1.904</td><td>十</td><td>十</td><td>十</td><td>十</td><td>十</td></tr><tr><td rowspan="5">ResNet-18</td><td>QAT (Nagel et al., 2021)</td><td>QAT</td><td>69.68</td><td>69.76</td><td>-0.08</td><td>十</td><td></td><td></td><td>十</td><td>一</td></tr><tr><td>AdaRound (Nagel et al., 2020)</td><td>PTQ</td><td>71.01</td><td>68.55</td><td>2.46</td><td>十</td><td></td><td></td><td>十</td><td></td></tr><tr><td>BRECQ (Li et al., 2021)</td><td>PTQ</td><td>71.01</td><td>70.58</td><td>0.43</td><td>十</td><td>一</td><td></td><td></td><td></td></tr><tr><td>EPTQ (Gordon et al., 2024)</td><td>PTQ</td><td>71.01</td><td>70.70</td><td>0.31</td><td>十</td><td>十</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

PTQ method in the table. Per-tensor QAT with learned scales (Nagel et al., 2021) loses less, 1.55 and 0.08 pp. It relaxes parts of the contract that ShiftQAT must keep. It uses asymmetric activations, which give post-ReLU tensors all 8 bits where C5 leaves 7, and it learns non-powerof-two scales by backpropagation, whereas ShiftQAT keeps power-of-two activation scales fixed after calibration. The remaining gap is therefore the observed price of the full scaling contract at 4 bits, and separating the contribution of each factor is left to future work (Section 7). BRECQ and EPTQ reach smaller drops with per-channel, non-power-oftwo scales, which violate C2 and C3. ShiftQAT is the only method in the table whose scaling scheme the datapath can execute. With biases constrained to 8 bits, the same recipe produces the deployed networks of Section 5.3. Diferences across rows also reflect difering FP32 references and training regimes.

## 5.3 End-to-End Deployment on MAX78002

MAX78002 supports 1/2/4/8-bit weights with 8-bit activations. We deploy MobileNetV2 and ResNet-18 trained with the same ShiftQAT recipe, with mixed 4/8-bit weights and 8-bit activations. Unlike in Tables 2 and 3, the biases are quantized to 8 bits, as the hardware requires (C4), so these networks satisfy the full contract. The trained checkpoints are compiled with ai8x-synthesis, the toolchain’s open-source synthesis tool, with no additional calibration. Because full-size ImageNet models exceed the accelerator’s on-chip weight memory, the deployed variants are reduced. MobileNetV2 uses width multiplier m = 0.75 and ResNet-18 uses $m = 0 . 5 ,$ , both with 128 128 inputs. Layers wider than 1024 channels are capped at 1024, and each stride-2 convolution is replaced by 2 2 max-pooling followed by a stride-1 convolution (Appendix B.2). The mixedprecision assignment keeps the input and output layers at 8-bit weights and concentrates the 4-bit layers in the interior of the network. This follows a long-standing convention in low-bit quantization, where the first and last layers are kept at higher precision because they are the most sensitive to quantization error (Rastegari et al., 2016; Zhou et al., 2018; Choi et al., 2018; Esser et al., 2020). TQT’s authors likewise attribute part of their INT4 accuracy shortfall to not keeping the first and last layers at higher precision (Jain et al., 2020). Hessian-based analyses also confirm that sensitivity to quantization varies strongly across layers (Dong et al., 2019). The input layer sees raw pixels through only three input channels and holds few parameters, so 8 bits there cost little memory, while the output layer directly produces the logits. Latency and energy are measured with the MAX78002 evaluation kit’s on-board power monitor in its windowed energy-accumulation (CNN power) mode. Firmware instrumentation, generated by the synthesis tool, marks idle and active windows with trigger signals. The monitor accumulates power on the VCOREA rail, which supplies the CNN accelerator as well as the MCU logic, and times the active window. It subtracts the idle power, so that $E = ( A - I ) T$ , where A and I are the active and idle power and T is the active period. Each measured operation is averaged over 100 repetitions. Weight loading is measured as a separate stage and is not included in Table 4. The CNN accelerator runs at 200 MHz. Table 4 reports inference only, obtained by subtracting the input-loading stage from the input-loading-plus-inference stage.

Table 4 reports the results. The deployed MobileNetV2 runs in 14.6 ms (about 68 inferences per second) at 5.1 mJ per inference, and ResNet-18 runs in 8.4 ms (about 119 inferences per second) at 2.8 mJ. Every layer executes on the accelerator in integer arithmetic, and only the final output rescale runs on the Arm core. These are ImageNet-scale CNNs running on a microcontroller-class device at millisecond latency and millijoule energy. The memory columns show what drove the model design. Both variants store about 2 MB of weights, most of MAX78002’s 2,340 KiB weight memory, whose efective capacity is further reduced for layers with more than 64 channels. On average they store roughly 7 bits per weight for MobileNetV2 and roughly 5.5 bits for ResNet-18, so the 4-bit option was used far more heavily for ResNet-18. All biases occupy exactly one byte each, as C4 requires.

Reduced variants. To fit in on-chip weight memory, the deployed networks are narrower than the full-size models, take 128 128 inputs, and use mixed 4/8-bit weights. They also use 8-bit rather than 32-bit biases. Their accuracy is therefore not directly comparable to the full-size results in Section 5.2. Table 4 demonstrates what the flow enables on silicon, not a controlled accuracy comparison.

## 6 LESSONS FROM BUILDING AND DEPLOYING THE FLOW

L1. The contract, not the bit-width, is the specification. “W4A8” underdetermines a deployable network. Two W4A8 networks can difer in symmetry, scale granularity, bias width, and where rescaling happens, and only one of them may compile. Writing the contract down explicitly (Table 1) before designing the training flow is what made each mechanism in Section 4 necessary rather than optional.

L2. Constraints interact, so satisfy them jointly. Each constraint in isolation has a known solution. The dificulty comes from their interaction. Folding activation scales forward (C6, C7) moves them into biases, which must then fit in 8 bits (C4) on the same power-of-two grid as the weights (C2). The joint, per-iteration weight/bias scale exists only because of this interaction.

L3. Topology, not arithmetic, is where static scaling breaks. The propagation identity itself (Eq. 5) is a few lines of algebra. Making it valid on real networks required resolving every point where branches merge or weights are reused (Section 4.3). Any flow targeting a datapath without dynamic cross-branch rescaling must solve this before it can compile ResNet- or FPN-style graphs.

L4. On-chip weight memory shapes the deployed model. The deployed models were sized to fill the weight memory (Table 4). Width, input resolution, the 1024-channel limit, and the 4/8-bit mix were all chosen so that the network fits. Wider or higher-resolution deployments come from more weight memory or lower average bits per weight, and a QAT flow’s leverage lies in making the lower-bit option accurate enough to use. ShiftQAT’s W4A8 results are what make 4-bit interior layers a practical way to fit larger networks.

L5. Make training emit the hardware configuration. The PoT scaler (Section 4.6) computes, at every iteration, the exact value that each layer’s output-shift register will hold. A trained checkpoint therefore maps deterministically to integer weights, 8-bit biases, and shift values. No posttraining calibration step is needed, so the hardware never runs a configuration that training did not see. In practice, we found this property more valuable than any single accuracy trick, because it makes the path from training to silicon predictable.

## 7 DISCUSSION AND SCOPE

Beyond MAX7800x. The constraints in Table 1 are typical of shift-based fixed-point datapaths, and the mechanisms of Section 4 do not depend on MAX7800x specifics beyond them. On accelerators that lack a free per-layer shifter, static propagation would additionally remove per-layer rescaling from the datapath. On MAX78002, whose output shift is free, its value lies in making the network executable at all.

Closing the 4-bit gap. Learned power-of-two thresholds, as in TQT (Jain et al., 2020), remain compatible with the contract, because the learned scales are still powers of two. Adopting them in ShiftQAT is a direct way to narrow the W4A8 gap to QAT methods that relax the contract.

Activation functions. Propagation requires every activation between the input and the output to be positively homogeneous, such as ReLU, Leaky ReLU (Maas et al., 2013), or the modulus (Vallés-Pérez et al., 2023). These activations are standard in edge CNNs. Networks with Sigmoid, Tanh, GELU, or SiLU would need scale recovery at those layers, which a static datapath cannot provide.

Table 4. MobileNetV2 and ResNet-18 deployed end-to-end on MAX78002 with the ShiftQAT flow (CNN clock 200 MHz). Accuracy is ImageNet top-1. Time and energy are per inference, excluding weight and input loading, measured on the VCOREA rail with idle power subtracted.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Input</td><td rowspan="2"> $\mathbf { A c c . } \left( \% \right)$ </td><td colspan="2">Weights</td><td colspan="2">Biases</td><td colspan="2">Inference</td></tr><tr><td>Num. (M)</td><td>Size (MB)</td><td>Num. (k)</td><td>Size (kB)</td><td>Time (ms)</td><td>Energy (µJ)</td></tr><tr><td>MobileNetV2  $\mathrm { ( } m { = } 0 . 7 5 )$ </td><td> $1 2 8 \times 1 2 8$ </td><td>62.812</td><td>2.29</td><td>2.03</td><td>6.6</td><td>6.6</td><td>14.6</td><td>5137</td></tr><tr><td> $\mathrm { R e s N e t - } 1 8 \ ( m { = } 0 . 5 )$ </td><td> $1 2 8 \times 1 2 8$ </td><td>53.230</td><td>3.06</td><td>2.05</td><td>3.4</td><td>3.4</td><td>8.4</td><td>2768</td></tr></table>

## 8 CONCLUSION

Fixed-function edge accelerators define a numerical contract, and a QAT flow for them must be designed to that contract rather than to a bit-width. ShiftQAT meets the full contract of the MAX7800x family. It resolves topology dependencies so that activation scales propagate statically, and it trains weights and biases on a shared power-of-two grid that fits an 8-bit bias budget. Under the contract’s scaling constraints, it matches the W8A8 accuracy drop of less constrained QAT methods and stays within 2.9 pp of FP32 at W4A8. With 8-bit biases, ShiftQAT-trained networks run end-to-end on MAX78002 at millisecond latency and millijoule energy per inference. The flow is open source and is the default training path of the MAX7800x toolchain, and we expect its constraint-to-mechanism design to carry over to other fixed-function accelerators.

## REFERENCES

Analog Devices Inc. MAX78002 datasheet and product info. https://www.analog.com/en/products/ max78002.html, 2023. Accessed: 2026-06-17.

Ansel, J., Yang, E., He, H., Gimelshein, N., Jain, A., Voznesensky, M., Bao, B., Bell, P., Berard, D., Burovski, E., Chauhan, G., Chourdia, A., Constable, W., Desmaison, A., DeVito, Z., Ellison, E., Feng, W., Gong, J., Gschwind, M., Hirsh, B., Huang, S., Kalambarkar, K., Kirsch, L., Lazos, M., Lezcano, M., Liang, Y., Liang, J., Lu, Y., Luk, C., Maher, B., Pan, Y., Puhrsch, C., Reso, M., Saroufim, M., Siraichi, M. Y., Suk, H., Suo, M., Tillet, P., Wang, E., Wang, X., Wen, W., Zhang, S., Zhao, X., Zhou, K., Zou, R., Mathews, A., Chanan, G., Wu, P., and Chintala, S. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. In 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS ’24).

ACM, apr 2024. doi: 10.1145/3620665.3640366. URL https://pytorch.org/assets/pytorch2-2.pdf.

Bengio, Y., Léonard, N., and Courville, A. Estimating or propagating gradients through stochastic neurons for conditional computation, 2013. URL https://arxiv. org/abs/1308.3432.

Choi, J., Wang, Z., Venkataramani, S., Chuang, P. I.-J., Srinivasan, V., and Gopalakrishnan, K. Pact: Parameterized clipping activation for quantized neural networks, 2018. URL https://arxiv.org/abs/1805.06085.

Dabbous, A., Lazzaroni, L., Bellotti, F., Presentación, S. M., Pighetti, A., and Berta, R. Tinyml acceleration with max78000. In Valle, M., Gastaldo, P., and Limiti, E. (eds.), Proceedings of SIE 2024, pp. 468–474, Cham, 2025. Springer Nature Switzerland. ISBN 978-3-031- 71518-1.

David, R., Duke, J., Jain, A., Janapa Reddi, V., Jefries, N., Li, J., Kreeger, N., Nappier, I., Natraj, M., Wang, T., Warden, P., and Rhodes, R. TensorFlow Lite Micro: Embedded machine learning for TinyML systems. In Proceedings of Machine Learning and Systems (MLSys), volume 3, pp. 800–811, 2021.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR09, 2009.

Dong, Z., Yao, Z., Gholami, A., Mahoney, M. W., and Keutzer, K. HAWQ: Hessian AWare quantization of neural networks with mixed-precision. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 293–302, 2019.

Esser, S. K., McKinstry, J. L., Bablani, D., Appuswamy, R., and Modha, D. S. Learned step size quantization. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id= rkgO66VKDS.

Gholami, A., Kim, S., Dong, Z., Yao, Z., Mahoney, M. W., and Keutzer, K. A survey of quantization methods for eficient neural network inference. In Low-Power Computer Vision, pp. 291–326. Chapman and Hall/CRC, 2022.

Giordano, M., Piccinelli, L., and Magno, M. Survey and comparison of milliwatts micro controllers for tiny machine learning at the edge. In 2022 IEEE 4th International Conference on Artificial Intelligence Circuits and Systems (AICAS), pp. 94–97, 2022. doi: 10.1109/ AICAS54282.2022.9870017.

Gordon, O., Cohen, E., Habi, H. V., and Netzer, A. Eptq: Enhanced post-training quantization via hessianguided network-wise optimization, 2024. URL https: //arxiv.org/abs/2309.11531.

Habi, H. V., Peretz, R., Cohen, E., Dikstein, L., Dror, O., Diamant, I., Jennings, R. H., and Netzer, A. Hptq: Hardware-friendly post training quantization, 2021. URL https://arxiv.org/abs/2109.09113.

Huang, Y., Gong, T., Jang, S., Kawsar, F., and Min, C. Energy characterization of tiny ai accelerator-equipped microcontrollers. In Proceedings of the 2nd International Workshop on Human-Centered Sensing, Networking, and Multi-Device Systems, HumanSys ’24, pp. 1– 6, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400713002. doi: 10.1145/ 3698388.3699628. URL https://doi.org/10.1145/ 3698388.3699628.

Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R., and Bengio, Y. Binarized neural networks. In Lee, D., Sugiyama, M., Luxburg, U., Guyon, I., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 29. Curran Associates, Inc., 2016. URL https://proceedings. neurips.cc/paper\_files/paper/2016/file/ d8330f857a17c53d217014ee776bfd50-Paper.pdf.

Iofe, S. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.

Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., and Kalenichenko, D. Quantization and training of neural networks for eficient integerarithmetic-only inference. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

Jain, S. R., Gural, A., Wu, M., and Dick, C. Trained quantization thresholds for accurate and eficient fixed-point inference of deep neural networks. In Proceedings of Machine Learning and Systems (MLSys), volume 2, pp. 112– 128, 2020.

Jung, S., Son, C., Lee, S., Son, J., Han, J.-J., Kwak, Y., Hwang, S. J., and Choi, C. Learning to quantize deep networks by optimizing quantization intervals with task loss. In Proceedings of the IEEE/CVF Conference on

Computer Vision and Pattern Recognition (CVPR), June 2019.

Lai, L., Suda, N., and Chandra, V. CMSIS-NN: Eficient neural network kernels for Arm Cortex-M CPUs, 2018.

Li, Y., Dong, X., and Wang, W. Additive powers-of-two quantization: An eficient non-uniform discretization for neural networks. In International Conference on Learning Representations (ICLR), 2020.

Li, Y., Gong, R., Tan, X., Yang, Y., Hu, P., Zhang, Q., Yu, F., Wang, W., and Gu, S. Brecq: Pushing the limit of post-training quantization by block reconstruction. arXiv preprint arXiv:2102.05426, 2021.

Lin, J., Chen, W.-M., Lin, Y., Cohn, J., Gan, C., and Han, S. MCUNet: Tiny deep learning on IoT devices. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pp. 11711–11722, 2020.

Lin, T., Dollár, P., Girshick, R. B., He, K., Hariharan, B., and Belongie, S. J. Feature pyramid networks for object detection. CoRR, abs/1612.03144, 2016. URL http: //arxiv.org/abs/1612.03144.

Maas, A. L., Hannun, A. Y., and Ng, A. Y. Rectifier nonlinearities improve neural network acoustic models. In Proc. icml, volume 30, pp. 3. Atlanta, GA, 2013.

McKinstry, J. L., Esser, S. K., Appuswamy, R., Bablani, D., Arthur, J. V., Yildiz, I. B., and Modha, D. S. Discovering low-precision networks close to full-precision networks for eficient embedded inference, 2019. URL https:// arxiv.org/abs/1809.04191.

Migacz, S. 8-bit inference with tensorrt. In GPU technology conference, volume 2, pp. 5, 2017.

Moosmann, J., Giordano, M., Vogt, C., and Magno, M. Tinyissimoyolo: A quantized, low-memory footprint, tinyml object detection network for low power microcontrollers. In 2023 IEEE 5th International Conference on Artificial Intelligence Circuits and Systems (AICAS), pp. 1–5, 2023. doi: 10.1109/AICAS57966.2023.10168657.

Nagel, M., van Baalen, M., Blankevoort, T., and Welling, M. Data-free quantization through weight equalization and bias correction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1325–1334, 2019.

Nagel, M., Amjad, R. A., van Baalen, M., Louizos, C., and Blankevoort, T. Up or down? adaptive rounding for post-training quantization. CoRR, abs/2004.10568, 2020. URL https://arxiv.org/abs/2004.10568.

Nagel, M., Fournarakis, M., Amjad, R. A., Bondarenko, Y., van Baalen, M., and Blankevoort, T. A white paper on neural network quantization, 2021.

Rastegari, M., Ordonez, V., Redmon, J., and Farhadi, A. XNOR-Net: ImageNet classification using binary convolutional neural networks. In European Conference on Computer Vision (ECCV), pp. 525–542, 2016.

Uhlich, S., Mauch, L., Cardinaux, F., Yoshiyama, K., Garcia, J. A., Tiedemann, S., Kemp, T., and Nakamura, A. Mixed precision dnns: All you need is a good parametrization. In International Conference on Learning Representations, 2020. URL https:// openreview.net/forum?id=Hyx0slrFvH.

Vallés-Pérez, I., Soria-Olivas, E., Martínez-Sober, M., Serrano-López, A. J., Vila-Francés, J., and Gómez-Sanchís, J. Empirical study of the modulus as activation function in computer vision applications. Engineering Applications of Artificial Intelligence, 120:105863, 2023. ISSN 0952-1976. doi: https://doi.org/10.1016/j.engappai.2023.105863. URL https://www.sciencedirect.com/science/ article/pii/S0952197623000477.

Vandersteegen, M., Van Beeck, K., and Goedemé, T. Integer-only CNNs with 4 bit weights and bit-shift quantization scales at full-precision accuracy. Electronics, 10 (22):2823, 2021. doi: 10.3390/electronics10222823.

Wang, G., Bhat, Z. P., Jiang, Z., Chen, Y.-W., Zha, D., Reyes, A. C., Niktash, A., Ulkar, G., Okman, E., Cai, X., and Hu, X. Bed: A real-time object detection system for edge devices. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, CIKM ’22, pp. 4994–4998, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450392365. doi: 10.1145/3511808.3557168. URL https://doi.org/10.1145/3511808.3557168.

Yang, H., Duan, L., Chen, Y., and Li, H. {BSQ}: Exploring bit-level sparsity for mixed-precision neural network quantization. In International Conference on Learning Representations, 2021. URL https://openreview. net/forum?id=TiXl51SCNw8.

Yao, Z., Dong, Z., Zheng, Z., Gholami, A., Yu, J., Tan, E., Wang, L., Huang, Q., Wang, Y., Mahoney, M. W., and Keutzer, K. HAWQ-V3: Dyadic neural network quantization. In Proceedings of the 38th International Conference on Machine Learning (ICML), volume 139 of PMLR, pp. 11875–11886, 2021.

Zhang, D., Yang, J., Ye, D., and Hua, G. Lq-nets: Learned quantization for highly accurate and compact deep neural networks. In Proceedings of the European Conference on Computer Vision (ECCV), September 2018.

Zhou, S., Wu, Y., Ni, Z., Zhou, X., Wen, H., and Zou, Y. Dorefa-net: Training low bitwidth convolutional neural networks with low bitwidth gradients, 2018. URL https://arxiv.org/abs/1606.06160.

## A HARDWARE-FRIENDLY QUANTIZATIONPROPERTIES AND THEIR MAX7800XMANDATE

Uniform quantization. A uniform quantizer $Q ( x ) \ =$ $\lfloor x / \Delta \rceil + Z$ uses a single step size $\Delta$ and zero-point $Z$ across the range, so multiply-accumulate operations need no lookup tables or branching. The accelerator stores weights, biases, and data as fixed-point Q7 integers, which implies a uniform step; there is no documented non-uniform or lookup-table dequantization path.

Symmetric quantization. Fixing $Z = 0$ gives $Q ( x ) =$ $\lfloor x / \Delta \rceil$ and removes the zero-point cross terms from every product. The accelerator’s accumulate-and-shift datapath assumes zero-centered operands. The documented number format is signed two’s complement with no ofset term, so a nonzero zero-point would need an ofset correction the datapath does not describe (C1).

Power-of-two scaling. With $\Delta = 2 ^ { s } / 2 ^ { n - 1 }$ for integer s, $Q ( x ) ~ = ~ \lfloor x \cdot 2 ^ { n - 1 } / 2 ^ { s } \rceil$ , and every rescaling becomes a shift. On MAX7800x this is not a preference. The perlayer output shift can only multiply by powers of two, so a non-power-of-two scale cannot be realized without a general multiplier on the output path (C2).

Per-tensor quantization. Per-channel scales improve accuracy (Habi et al., 2021; Li et al., 2021; Gordon et al., 2024), but they require a scale per output channel at inference. The accelerator’s output shift is configured per layer, so per-channel scaling would require per-channel shifts or a runtime multiply, neither of which is documented for this hardware generation (C3).

## B ADDITIONAL EXPERIMENTAL DETAILS

## B.1 Hyperparameters

MobileNetV2. We use SGD with momentum 0.9 and weight decay $4 \times 1 0 ^ { - 5 }$ for 200 epochs, with QAT starting at epoch 150. Runs initialized from a pretrained checkpoint resume at epoch 150. The learning rate follows cosine annealing from an initial value of 0.05.

ResNet-18. We use SGD with momentum 0.9 and weight decay $1 0 ^ { - 4 }$ for 120 epochs, with QAT starting at epoch 90. Runs initialized from a pretrained checkpoint resume at epoch 90. The learning rate follows MultiStepLR with milestones at epochs 30, 60, 90, and 105, and $\gamma = 0 . 1$

Scale estimation. Activation statistics are collected as 2048-bin histograms in one forward pass over the training set, without backpropagation, immediately before QAT starts. Activation outliers are removed with a z-score threshold of 8.0. The activation-threshold search evaluates five power-of-two candidates, the non-clipping threshold and its first four halvings. The weight/bias range bound uses the 0.999 quantile for 8-bit layers and the 0.9 quantile for 4-bit layers. Total per-layer shifts are clamped to [ 15, 15].

## B.2 Hardware Implementation Details

To fit the memory and architectural limits of MAX78002, we made the following changes for deployment:

• Input resolution was set to $1 2 8 \times 1 2 8$

• MobileNetV2 used width multiplier 0.75, and ResNet-18 used width multiplier 0.5.

• Layers wider than 1024 channels were capped at 1024 channels, the hardware’s per-layer channel limit.

• Each stride-2 conv2d was replaced by $2 \times 2$ maxpooling followed by a stride-1 conv2d.

• Biases were quantized to 8 bits, as the hardware requires (C4). The accuracy comparisons in Tables 2 and 3 instead use 32-bit biases for all methods, matching the published baselines.

## B.3 Compute Resources

All training ran on two NVIDIA RTX 4090 GPUs with 24 GB each. Runs resumed from a pretrained checkpoint took approximately 12 hours, and runs from scratch took approximately 48 hours.

## B.4 Dataset License

We use the ILSVRC-2012 subset of ImageNet under its noncommercial research license (https://www.image-net. org/download.php).