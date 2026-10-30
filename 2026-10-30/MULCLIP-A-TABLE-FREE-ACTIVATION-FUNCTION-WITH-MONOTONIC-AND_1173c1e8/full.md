# MULCLIP: A TABLE-FREE ACTIVATION FUNCTION WITH MONOTONIC AND FOLDING REGIMES FOR RESERVOIR COMPUTING ON MICROCONTROLLERS

Guren Matsumura, Masato Takemoto, and Shinzo Koyama Panasonic Industry Co., Ltd., Osaka, Japan matsumura.guren@jp.panasonic.com

## ABSTRACT

This letter proposes MulClip, a lightweight activation function for reservoir computing (RC) on microcontrollers (MCUs). In a ring reservoir, a representative sparse RC topology, multiply–accumulate operations are few, and evaluating tanh accounted for 75% of the state-update time on a Cortex-M4. MulClip introduces curvature into the response using only comparison, absolute value, multiplication, and subtraction, and places its clipping threshold not at the extremum but at the point where the response returns to zero after passing the extremum. As a result, a single function has both a monotonic regime and a nonmonotonic folding regime that increases the expressiveness of a single unit. Which regime is used is determined by the usual hyperparameter search, e.g., over the input scale, so no task-specific threshold tuning is required. The state-update time measured on hardware was 0.34 times that of tanh and shorter than that of any of the 11 evaluated tanh approximations. In PC-based evaluation, third-order MulClip achieved a 22% lower NMSE than tanh on NARMA-20, where the pre-activations remained in the monotonic regime, and on 8-bit parity it raised the accuracy from 0.76–0.78 for the monotonic-clip variant without folding to 0.97–0.98. For these two tasks, on-device inference yielded nearly the same values as on the PC.

Keywords Activation function · embedded machine learning · microcontroller · reservoir computing

## I Introduction

In recent years, there has been growing demand for running machine-learning inference on embedded devices, for example, for sequential processing of sensor signals [1]–[3]. However, the microcontrollers (MCUs) widely used in embedded devices have limited computing performance and memory capacity, and it is often difficult to complete inference within the required processing period.

Reservoir computing (RC) [4], [5] is a lightweight approach suited to such environments. RC fixes the reservoir weights and trains only the readout layer. Moreover, sparsifying the reservoir connections reduces the multiply–accumulate operations in the state update. A ring topology, in which connections are restricted to a cycle, achieves performance comparable to that of a randomly connected reservoir while making the number of multiply–accumulate operations roughly proportional to the number of units [6]. However, the fewer the multiply–accumulate operations, the larger the share of the activation function in the state-update cost. The standard tanh is a transcendental function and is expensive relative to multiplication and addition even on MCUs with an FPU [7]. Indeed, for a ring reservoir on a Cortex-M4, tanh evaluated on the FPU (hereafter tanh(FPU)) accounted for 75% of the state-update time (Section III-B).

Alternatives to tanh include implementations that approximate tanh with LUTs, piecewise-linear (PWL) approximations, or rational functions [8], [9], and functions such as hardtanh, softsign [10], and ReLU [11]. The former aim to reproduce tanh, so higher fidelity requires more memory or arithmetic. The latter are inexpensive but have fixed shapes; in our measurements, they increased the NARMA-20 error to 1.6–1.9 times that of tanh(FPU). The shape of the activation function strongly affects RC performance [12], and in particular, responses that are nonmonotonic in the input are known to increase the expressiveness of a single unit [13]. However, most such functions involve transcendental functions and thus suffer from the same problem as tanh on MCUs.

![](images/9b3b29c41d96558e33a94ed30dd15ea8a19a0db62175667dd5f9e69c1d2b250c.jpg)  
Fig. 1. Response of MulClip (blue: $d = 2 ;$ red: $d = 3 )$ . Small dots mark the extremum at $d / ( d - 1 )$ , and open circles mark the clipping threshold $c _ { d } .$ . Gray dashed lines show the monotonic-clip variant, whose clipping threshold is set at $\bar { d / } ( d - 1 )$ . The dotted line is tanh.

This letter therefore proposes MulClip, an activation function that, rather than approaching tanh, inexpensively provides a response that tanh lacks. MulClip introduces curvature using only comparison, absolute value, multiplication, and subtraction, and places the clipping threshold at the point where the output returns to zero, which is determined by the functional form; thus, a single function has both a monotonic regime and a folding regime without task-specific threshold tuning. This is the central contribution of this letter. With this design, both a monotonic operating point and an operating point that exploits folding can be selected solely through the hyperparameter search over, e.g., the input scale, that is routinely performed in RC. Section II presents the design, and Section III evaluates it in terms of both on-device computational cost and task accuracy.

## II Proposed Method—MulClip

The cheapest way to provide a bounded nonlinearity is to clip the input; clipping with a threshold of 1 yields hardtanh. MulClip adds a low-order term with a negative coefficient to the clipped input to give the response curvature. MulClip of order d $( d \geq 2 )$ is defined as follows.

$$
x _ { c } = \operatorname* { m i n } \big ( \operatorname* { m a x } ( x , - c _ { d } ) , c _ { d } \big ) ,\tag{1}
$$

$$
f ( x ) = x _ { c } - \alpha _ { d } | x _ { c } | ^ { d - 1 } x _ { c } , \qquad \alpha _ { d } = \frac { ( d - 1 ) ^ { d - 1 } } { d ^ { d } } ,\tag{2}
$$

$$
c _ { d } = \alpha _ { d } ^ { - 1 / ( d - 1 ) } = \frac { d ^ { d / ( d - 1 ) } } { d - 1 } .\tag{3}
$$

The coefficient $\alpha _ { d }$ is chosen such that the unclipped response attains its extremum of 1 at $| x | = d / ( d - 1 )$ , and $c _ { d }$ is the point where that response returns to 0 after passing the extremum (for $d = 2 , \alpha _ { 2 } = 1 / 4$ and $c _ { 2 } = 4 ;$ for $d = 3 , \alpha _ { 3 } = 4 / 2 7$ and $c _ { 3 } = 3 \sqrt { 3 } / 2 \approx 2 . 6 0 )$ . Evaluation requires only two comparisons, d multiplications, and one subtraction (plus one absolute value for $d = 2 )$ , with no division, transcendental function, or table lookup. The output always lies in [−1, 1].

Fig. 1 shows the response. Since MulClip is an odd function, we describe the $x \ge 0$ side below. For $0 \leq x \leq d / ( d - 1 )$ the response increases monotonically from a slope of 1 at the origin until it reaches 1 (monotonic regime). Beyond this point, it decreases and returns to 0 at $c _ { d }$ (folding regime). Beyond $c _ { d }$ , the output is 0 (zero regime). The folding regime traverses the same output range [0,1] as the monotonic regime in reverse. If the clipping threshold is smaller than $c _ { d } ,$ the folding stops partway; if it is larger, a region arises in which the sign of the output is opposite to that of the input.

If the input to the activation function (the pre-activation) stays within the monotonic regime, the output of MulClip coincides with that of the monotonic function whose clipping threshold is set at $d / ( d - 1 )$ (hereafter, the monotonic-clip variant). In RC, hyperparameters such as the input scale are customarily searched for each task [5], and this search determines which regime is used. Therefore, the clipping threshold need not be tuned for each task.

The polynomial part of MulClip itself is not new. For example, the monotonic-clip variant with $d = 3$ equals the cubic soft clipper [14] scaled so that both the slope at the origin and the maximum are 1. MulClip moves this clipping threshold from the extremum to $c _ { d } .$ where the output returns to 0. Making the function odd through a sign-multiplied even-order term is shared with the polynomial activation functions of Abe et al. [15], but those functions maintain a monotonic response through positive coefficients and output saturation. Hurley et al. showed that nonmonotonic functions are effective for RC [12], but their function involves transcendental functions. Yoshioka and Tamukoh implemented LUT-based RC on low-cost FPGAs [16]. The novelty of this letter lies not in the polynomial or the folding response per se, but in making a monotonic regime and a folding regime coexist in a single activation function, without adding a task-specific threshold parameter, through the clipping threshold $c _ { d }$ uniquely determined by the functional form.

## III Experimental Evaluation

## A Experimental Setup

The state-update time was measured by running a 500-unit ring reservoir, in which each unit receives only the state of the preceding unit,

$$
x _ { i } ( t + 1 ) = f { \big ( } w x _ { i - 1 } ( t ) + \nu _ { i } u ( t ) + b _ { i } { \big ) } \qquad ( x _ { 0 } = x _ { N } , N = 5 0 0 )\tag{4}
$$

for 230 steps on an STM32 NUCLEO-F446RE (Cortex-M4, single-precision FPU, 84 MHz). Here, w is the shared coupling weight, $\nu _ { i }$ and $b _ { i }$ are the input weight and bias, and the argument of $f$ is the pre-activation. The state-update time (the time to compute the pre-activations and evaluate the activation function for all units) was measured with the cycle counter (firmware built with -O2). We compared 20 activations: MulClip $( d = 2 , 3 )$ ; tanh(FPU), i.e., newlib’s tanhf executed on the FPU; 11 tanh approximations (two rational approximations, five nearest-neighbor LUTs, three interpolated LUTs, and an eight-segment PWL approximation, PWL8); hardtanh; softsign; ReLU; and three polynomial activation functions of Abe et al. (F1–F3) [15] (implementation and optimization details in Appendices A and B).

Accuracy was evaluated by PC simulation (reservoir state update in float32, readout in float64) on four tasks: NARMA 20 [17] (NMSE); 8-bit parity, Parity8 [18] (accuracy); autonomous prediction of the Mackey–Glass time series [19], MG (valid prediction time, VPT, in steps); and 10-word recognition on Google Speech Commands [20], KWS (accuracy). The readout was trained by ridge regression. For each activation, the spectral radius (coupling weight w), input scale, and other hyperparameters were searched with 100 trials of Optuna’s TPE [21], and we report averages over 50 random seeds separate from those used in the search (Appendix C). To isolate the effect of the folding response, the monotonic-clip variant was evaluated with the same procedure. The on-device MulClip states agreed with the PC computation within a maximum absolute error of $2 . 0 \dot { \times } 1 0 ^ { - 6 }$ , even under conditions in which pre-activations entered the folding and zero regimes. Furthermore, for NARMA-20 and Parity8, we ran inference over the evaluation segment on the device using readouts trained on the PC (five seeds) and obtained nearly the same NMSE and accuracy as on the PC (Appendix F).

## B Evaluation Cost

The state-update time of MulClip was comparable to that of the cheapest functions, which use neither tables nor transcendental functions, and shorter than that of every tanh approximation. Evaluating tanh(FPU) required 203.2 cycles per activation and accounted for 75% of the 1.615-ms state-update time. MulClip required 30.0 (d = 2) and 31.0 (d = 3) cycles per activation, giving state-update times of 0.543 and 0.549 ms (0.34 times that of tanh(FPU)). These are within 0.06 ms of hardtanh, softsign, and ReLU (0.490–0.531 ms), whereas all 11 tanh approximations (0.592–0.704 ms) took longer. The state-update times of the functions of Abe et al. (F1–F3) were 0.34–0.48 times that of tanh(FPU) (measured in a separate build; Appendix B). The flash usage (text+data) of binaries built with only one activation each exceeded that of the smallest ones, ReLU and softsign, by 40–56 B for MulClip and by 880 B for tanh(FPU). The per-step processing time including the readout was 1.67 ms for tanh(FPU) and 0.60 ms for MulClip with $d = 3 ;$ only tanh(FPU) exceeds a 1-kHz processing period. In terms of cost, MulClip is thus nearly on par with the cheapest fixed-shape functions, and its difference from them appears in accuracy, as shown in the next subsection.

![](images/a955a2cec26facc7c9e352523295a6abb9105275711b4618e70fea1a7a0292d1.jpg)  
Fig. 2. NMSE versus state-update time (500 units) on NARMA-20. Lower left is better. The horizontal axis is broken. The horizontal dashed line indicates the NMSE of tanh(FPU). The connected black squares are, from left to right, hardtanh, MulClip with d = 2, and MulClip with d = 3. Numbers beside the LUT points denote the number of entries. Hexagons denote the functions of Abe et al. [15]; F2 (NMSE 0.60) lies outside the vertical-axis range and is plotted at the top edge. The times for the functions of Abe et al. were measured in a separate build (Appendix B).

## C Accuracy Versus State-Update Time

Fig. 2 shows the relationship between NMSE and state-update time on NARMA-20. Adding curvature to hardtanh lowered the NMSE from 0.150 to 0.087 for d = 2 and to 0.063 for d = 3; the latter is 22% below the 0.081 of tanh(FPU). Because the pre-activations never entered the folding regime on NARMA-20 (Section III-D), this improvement is attributable to curvature. None of the tanh approximations achieved a lower NMSE than tanh(FPU), and tanh(FPU) and all 11 approximations had both longer state-update times and higher NMSEs than MulClip with d = 3.

Table I shows the accuracy on the four tasks. On MG, tanh(FPU) and the 5/4-order rational approximation slightly outperformed MulClip with d = 3 (VPT 628 and 626 vs. 599). On KWS, the accuracies of all activations except the 10-entry LUT and Abe F2 fell within 0.742–0.785. In contrast, on Parity8, only MulClip and the nearest-neighbor LUTs with 100 or more entries reached 0.911–0.990, whereas all others were below 0.85 (0.761 for tanh(FPU)). The interpolated LUTs, which have no steps, yielded 0.736–0.763, suggesting that quantization steps may have contributed to the nearest-neighbor LUT results. The monotonic functions of Abe et al. reached up to 0.848 on Parity8, but their NARMA-20 NMSEs were 0.142–0.604 (1.75–7.46 times that of tanh(FPU)).

## D Effect of Folding

On Parity8, MulClip raised the accuracy from 0.764 (d = 2) and 0.781 (d = 3) for the monotonic-clip variants to 0.981 and 0.968, respectively. Both use the same operations and differ only in the clipping-threshold constant. In this case, about 19% and 17% of the pre-activations fell in the folding regime, and about 11% and 10% in the zero regime. When the number of bits k was varied from 2 to 10 (Appendix E), all activations achieved 0.99 or higher with no difference for $k \leq 6 ,$ whereas MulClip consistently outperformed the monotonic-clip variants for $k \geq { \bar { 7 } } .$ . For $k \geq 8 ,$ MulClip also outperformed all monotonic activations other than the nearest-neighbor LUTs, but the 4096- and 16384-entry nearest-neighbor LUTs achieved accuracy comparable to or better than that of MulClip with d = 2. To separate the contributions of the folding and zero regimes, we evaluated, for d = 2, a variant with only part of the folding regime and a variant with only the zero regime (Appendix D). Their accuracies were 0.877 and 0.810, respectively, exceeding the monotonic-clip variant but falling short of MulClip, which has both regimes and achieved the highest accuracy.

Table I  
ACCURACY (MEAN OVER 50 SEEDS), STATE-UPDATE TIME, AND FLASH USAGE (RELATIVE TO tanh(FPU)) OF EACH ACTIVATION
<table><tr><td>Activation</td><td>NARMA-20 NMSE↓</td><td>Parity8 Acc.↑</td><td>MG VPT↑</td><td>KWS Acc.↑</td><td>State-update time ratio↓</td><td>Flash ratio↓</td></tr><tr><td>MulClip d = 2</td><td>0.087</td><td>0.981</td><td>473</td><td>0.771</td><td>0.34</td><td>0.94</td></tr><tr><td>MulClip d = 3</td><td>0.063</td><td>0.968</td><td>599</td><td>0.781</td><td>0.34</td><td>0.94</td></tr><tr><td>Monotonic clip d = 2</td><td>0.087</td><td>0.764</td><td>472</td><td>0.772</td><td></td><td></td></tr><tr><td>Monotonic clip d = 3</td><td>0.063</td><td>0.781</td><td>582</td><td>0.785</td><td></td><td></td></tr><tr><td>tanh(FPU)</td><td>0.081</td><td>0.761</td><td>628</td><td>0.779</td><td>1.00</td><td>1.00</td></tr><tr><td>Rational 3/2</td><td>0.104</td><td>0.767</td><td>596</td><td>0.779</td><td>0.40</td><td>0.94</td></tr><tr><td>Rational 5/4</td><td>0.081</td><td>0.752</td><td>626</td><td>0.782</td><td>0.44</td><td>0.95</td></tr><tr><td>LUT nearest 10–16384</td><td>0.252–0.110</td><td>0.697–0.990</td><td>54-391</td><td>0.679–0.776</td><td>0.39</td><td>0.95-5.53</td></tr><tr><td>LUT interp. 16–256</td><td>0.144–0.098</td><td>0.763-0.746</td><td>435-469</td><td>0.757–0.779</td><td>0.43</td><td>0.95-1.02</td></tr><tr><td>PWL8</td><td>0.153</td><td>0.777</td><td>363</td><td>0.756</td><td>0.37</td><td>0.95</td></tr><tr><td>hardtanh softsign</td><td>0.150</td><td>0.803</td><td>390</td><td>0.742</td><td>0.32</td><td>0.94</td></tr><tr><td></td><td>0.127</td><td>0.734</td><td>470</td><td>0.748</td><td>0.33</td><td>0.94</td></tr><tr><td>ReLU</td><td>0.152</td><td>0.795</td><td>352</td><td>0.752</td><td>0.30</td><td>0.94</td></tr><tr><td>Abe F1 / F2 / F3 [15]</td><td>0.266 / 0.604</td><td>0.848 / 0.750</td><td>253 / 5</td><td>0.769 / 0.720</td><td>0.34 / 0.48</td><td>0.95</td></tr><tr><td></td><td>/0.142</td><td>/0.844</td><td>/311</td><td>/0.776</td><td>/0.40</td><td></td></tr></table>

Note: ↓ lower is better; ↑ higher is better. Accuracy is the mean over 50 seeds. State-update time and flash usage are ratios to tanh(FPU) (1.615 ms, 14,304 B). The monotonic-clip variants differ from MulClip only in the clipping-threshold constant, so their time and flash usage were not measured. For the LUT rows, the values for the smallest and largest table sizes are given in the order “smallest-size value–largest-size value”; they are not ranges (all values in Appendix C). The state-update time ratios of the functions of Abe et al. are relative to tanh(FPU) measured in the separate build that included them (Appendix B).

As for the other tasks, on NARMA-20 the pre-activations never entered the folding regime, and the results of MulClip and the monotonic-clip variant were identical. On MG, the fraction entering it was 0.4% or less, and the difference was small (Appendix D). On KWS with d = 3, the search selected an operating point that uses the folding regime (9.8% of the pre-activations), and the accuracy was slightly lower than that of the monotonic-clip variant (0.781 vs. 0.785). Because MulClip also reproduces 0.785 when given the hyperparameters of the monotonic-clip variant (Appendix D), that operating point exists within MulClip as well, but the 100-trial search did not select it. Thus, MulClip contains the operating points of the monotonic-clip variant and provides additional ones, but a finite search is not guaranteed to select the best operating point.

The low accuracy of the monotonic-clip variant could also stem from an inability to retain the most recent eight bits. However, when trained on the same input sequence to predict the number of 1s among the most recent eight bits, even the monotonic-clip variant predicted it almost without error $( \mathrm { N M S E } \leq 4 \times 1 0 ^ { - 8 } ;$ Appendix E). Since parity is whether this count is even or odd, the difference on Parity8 is hard to explain by insufficient memory alone.

## IV Discussion and Conclusion

These results indicate that, for RC on MCUs, the design goal of an activation function need not be to approximate tanh.   
Curvature was effective on NARMA-20, and the folding response was effective on Parity8.

We have not verified the mechanism by which the folding response helped on Parity8, so it remains a hypothesis. One hypothesis is that, when the pre-activation changes monotonically with the number of 1s, a folding response allows a single unit to raise and then lower its output, making it easier to represent parity, which alternates between 0 and 1 as the number of 1s increases; this corresponds to the fact that a nonmonotonic response can represent XOR with a single unit [13].

This study has the following limitations. First, the state-update time was measured for a ring reservoir on a Cortex-M4. The share of the activation function is expected to decrease as the number of connections increases, but we did not measure the time for other topologies (Appendix B). In addition, the accuracies in Table I were evaluated on a PC over 50 seeds, and on-device inference was confirmed only for NARMA-20 and Parity8 with five seeds. Second, because MulClip returns 0 for $| x | \geq c _ { d }$ , information may be lost for large inputs that do not appear during training.

Furthermore, for d = 3, the slope just below the threshold is −2, which locally amplifies the input. Third, c is not the accuracy-maximizing threshold. For Parity8 with d = 2, a threshold of c = 5 yielded an accuracy of 0.999, but this setting creates a region in which the output sign is opposite to that of the input, and the output also falls outside [−1, 1]. In contrast, c is the threshold, determined by the functional form, that includes the folding regime to its full extent without causing sign inversion (Appendix D).

In summary, we proposed MulClip, an activation function composed only of comparison, absolute value, multiplication, and subtraction. On a Cortex-M4, MulClip ran with 0.34 times the state-update time of tanh(FPU), and in PC-based evaluation it achieved a lower NMSE on NARMA-20 than tanh(FPU) and all 11 approximations, as well as higher accuracy on Parity8 than the monotonic-clip variant. As for the order, d = 3, which yielded a clearly lower NMSE on NARMA-20 at a cost difference of only one cycle, is a reasonable default. For the evaluated ring reservoir, inexpensively realizing a response with both monotonic and folding regimes was more advantageous than approximating tanh in terms of both state-update time and task accuracy.

## Acknowledgment

The authors used the generative AI tool GitHub Copilot (GitHub Copilot CLI; model: Anthropic’s Claude) to write code for the experimental firmware and for the PC-based experiment, analysis, and plotting scripts; to draft and revise the manuscript text; and to cross-check the numerical values in the text against the experimental data. The authors reviewed all code, numerical values, figures, and text and take full responsibility for the content.

## References

[1] P. Warden and D. Situnayake, TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers. Sebastopol, CA, USA: O’Reilly Media, 2019.

[2] J. Lin, W.-M. Chen, Y. Lin, J. Cohn, C. Gan, and S. Han, “MCUNet: Tiny deep learning on IoT devices,” in Proc. NeurIPS, 2020, pp. 11711–11722.

[3] L. Lai, N. Suda, and V. Chandra, “CMSIS-NN: Efficient neural network kernels for Arm Cortex-M CPUs,” arXiv:1801.06601, 2018.

[4] H. Jaeger, “The ‘echo state’ approach to analysing and training recurrent neural networks,” GMD—German National Research Center for Information Technology, Bonn, Germany, GMD Rep. 148, 2001.

[5] M. Lukoševicius and H. Jaeger, “Reservoir computing approaches to recurrent neural network training,”ˇ Comput. Sci. Rev., vol. 3, no. 3, pp. 127–149, 2009.

[6] A. Rodan and P. Tino, “Minimum complexity echo state network,”ˇ IEEE Trans. Neural Netw., vol. 22, no. 1, pp. 131–144, Jan. 2011.

[7] ARM Ltd., Cortex-M4 Technical Reference Manual, ARM DDI 0439B, 2010.

[8] H. Amin, K. M. Curtis, and B. R. Hayes-Gill, “Piecewise linear approximation applied to nonlinear function of a neural network,” IEE Proc.—Circuits Devices Syst., vol. 144, no. 6, pp. 313–317, 1997.

[9] G. A. Baker, Jr. and P. Graves-Morris, Padé Approximants, 2nd ed. Cambridge, U.K.: Cambridge Univ. Press, 1996.

[10] X. Glorot and Y. Bengio, “Understanding the difficulty of training deep feedforward neural networks,” in Proc. AISTATS, 2010, pp. 249–256.

[11] V. Nair and G. E. Hinton, “Rectified linear units improve restricted Boltzmann machines,” in Proc. ICML, 2010, pp. 807–814.

[12] L. A. Hurley, J. G. Restrepo, and S. E. Shaheen, “Tuning the activation function to optimize the forecast horizon of a reservoir computer,” J. Phys. Complexity, vol. 5, no. 3, Art. no. 035004, Jul. 2024, doi: 10.1088/2632-072X/ad5e55.

[13] M. M. Noel, A. L, A. Trivedi, and P. Dutta, “Growing cosine unit: A novel oscillatory activation function that can speedup training and reduce parameters in convolutional neural networks,” arXiv:2108.12943, 2021.

[14] J. O. Smith III, Physical Audio Signal Processingfor Virtual Musical Instruments and Audio Effects. W3K Publishing, 2010. [Online]. Available: https://ccrma.stanford.edu/\~jos/pasp/ (accessed Oct. 26, 2026).

[15] Y. Abe, K. Nishida, M. Akai-Kasaya, and T. Asai, “High-order polynomial activation function and regenerative internal weights for FPGA implementation of reservoir computing,” Nonlinear Theory Appl., IEICE, vol. 15, no. 2, pp. 262–272, Apr. 2024, doi: 10.1587/nolta.15.262.

[16] K. Yoshioka and H. Tamukoh, “A 340-µW TinyML using LUT-based reservoir computing on low-cost FPGAs,” IEEE Embedded Syst. Lett., vol. 17, no. 5, pp. 357–360, 2025, doi: 10.1109/LES.2025.3598209.

[17] A. F. Atiya and A. G. Parlos, “New results on recurrent network training: Unifying the algorithms and accelerating convergence,” IEEE Trans. Neural Netw., vol. 11, no. 3, pp. 697–709, May 2000.

[18] N. Bertschinger and T. Natschläger, “Real-time computation at the edge of chaos in recurrent neural networks,” Neural Comput., vol. 16, no. 7, pp. 1413–1436, 2004.

[19] M. C. Mackey and L. Glass, “Oscillation and chaos in physiological control systems,” Science, vol. 197, no. 4300, pp. 287–289, 1977.

[20] P. Warden, “Speech Commands: A dataset for limited-vocabulary speech recognition,” arXiv:1804.03209, 2018.

[21] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, “Optuna: A next-generation hyperparameter optimization framework,” in Proc. ACM SIGKDD, 2019, pp. 2623–2631.

## Appendices

The following appendices provide details of the definitions, experimental conditions, and results referred to in the main text. Reference and equation numbers $\left( ( 1 ) ‐ ( 4 ) \right)$ follow the main text. The terminology also follows the main text: $| x | \leq d / ( d - 1 )$ is called the monotonic regime, $d / ( d - 1 ) < | x | < c _ { d }$ the folding regime, and $| x | \geq c _ { d }$ the zero regime. The control whose clipping threshold is placed at the extremum $d / ( d - 1 )$ is called the monotonic-clip variant (“Monotonic clip” in the tables).

## A Implementation and Optimization of the Compared Activations

We give the definitions of the 20 activation functions compared in the main text. All were implemented in singleprecision floating point (float32), and the same expressions and order of operations were used for the accuracy evaluation on the PC (Python) and the measurements on the MCU (C).

MulClip. As in $( 1 ) – ( 3 )$ of the main text. For $\begin{array} { r } { d = 2 , f ( x ) = x _ { c } - \frac { 1 } { 4 } | x _ { c } | x _ { c } } \end{array}$ , and for $\begin{array} { r } { d = 3 , f ( x ) = x _ { c } - \frac { 4 } { 2 7 } x _ { c } ^ { 3 } } \end{array}$ , where $x _ { c }$ is x clipped to $\left[ - c _ { d } , c _ { d } \right] ( c _ { 2 } = 4 , c _ { 3 } = 3 \sqrt { 3 } / 2 \approx 2 . 5 9 8 0 7 6 )$ . The monotonic-clip variant, the control in Section III-D of the main text, uses the same expression with the clipping threshold placed at the extremum $d / ( d - 1 ) \left( 2 \right.$ for $d = 2$ and 1.5 for $d = 3 )$ . On the MCU, after clipping with two comparisons, the output is computed with an absolute value, two multiplications, and one subtraction for $d = 2$ , and with three multiplications and one subtraction for $d = 3$ . For $d = 3$ the multiplications are performed in the order $( 4 / 2 7 ) \cdot ( x _ { c } \cdot x _ { c } ) \cdot x _ { c }$ <sub>c</sub> to match the rounding order of the PC implementation. The clipping threshold appears only as a comparison constant, and its value does not change the type or number of instructions. The timing measurements and the check of agreement with the PC computation were also performed with $c _ { d }$ (Appendices B and F).

tanh(FPU). newlib’s tanhf() built with single-precision FPU instructions (fpv4-sp-d16).

Rational approximations. The 3/2-order approximation is $f ( x ) = x ( 2 7 + x ^ { 2 } ) / ( 2 7 + 9 x ^ { 2 } )$ , which has value 1 and slope 0 at $x = 3 ;$ for $| x | \geq 3$ , the input is clamped to ±3. The 5/4-order approximation is $f ( x ) = x ( 9 4 5 + 1 0 5 x ^ { 2 } +$ $x ^ { 4 } ) / ( 9 4 5 + 4 2 0 x ^ { 2 } + 1 5 x ^ { 4 } )$ , the Padé approximant of tanh; the input is clamped at $x \approx 3 . 6 4 6 7$ , where the value reaches 1. In both cases the denominator is always positive, and one division is required.

LUTs. The values of tanh at n points dividing $[ - 5 , 5 ]$ into $n - 1$ equal intervals were stored as a constant array in flash. For nearest-neighbor lookup, the input is clamped to $[ - 5 , 5 ]$ , the index is computed as idx $= \lfloor ( x _ { c } + 5 ) \cdot ( n - 1 ) / \mathrm { 1 0 } + 0 . 5 \rfloor$ and the value at that point is returned $( n = 1 0$ , 100, 1000, 4096, 16384). For linear interpolation, the integer part i and the fractional part of $t = ( x _ { c } + 5 ) \cdot ( n - 1 ) / 1 0$ are computed, and the values at the two adjacent points are linearly interpolated $( n = 1 6 , 6 4 , 2 5 6 )$

PWL8. A piecewise-linear approximation that divides $[ - 5 , 5 ]$ into eight segments and fits a straight line in each. The segment boundaries were placed to minimize the maximum error; the boundaries on the positive side are 0, 0.6726, 1.2955, 1.7690, and 5.0. The coefficients are odd-symmetric.

softsign, ReL $\mathbf { \delta } _ { \mathbf { u } } \mathbf { u } ,$ and hardtanh. softsign is $x / ( 1 + | x | )$ (one division), ReLU is $\operatorname* { m a x } ( x , 0 )$ , and hardtanh clips x to [−1, 1] (comparisons only).

Polynomial activation functions of Abe et al. (F1–F3) [15]. The output of the ninth-order polynomial

$$
f ( x ) = \sum _ { i = 1 } ^ { 5 } a _ { 2 i - 1 } x ^ { 2 i - 1 } + \operatorname { s g n } ( x ) \sum _ { i = 1 } ^ { 4 } a _ { 2 i } x ^ { 2 i }
$$

is limited to $[ - 1 , 1 ] ,$ , using the three coefficient sets of the original paper. For $\mathrm { F } 1 , a _ { 1 } – a _ { 9 }$ are all 1; for F2, $( a _ { 1 } , \ldots , a _ { 9 } ) =$ $( 1 / 2 5 6 , 1 / 2 5 6 , 1 / 6 4 , 1 / 6 \breve { 4 } , 1 / 1 6 , 1 / 1 6 , 1 / 4 , 1 / 4 , 1 )$ ; and for F3, $( \dot { 1 } , 1 / 4 , 1 / 4 , 1 / 1 6 , 1 / 1 6 , 1 / 6 4 , 1 / 6 4 , 1 / 2 5 6 , 1 / 2 5 6 )$ Because the polynomial $\begin{array} { r } { P ( t ) = \sum _ { k } a _ { k } t ^ { k } } \end{array}$ in $t = \left| x \right|$ has positive coefficients and increases monotonically for $t \geq 0 ,$ |x| was clipped at the saturation point $t _ { \mathrm { s a t } }$ satisfying $P ( t _ { \mathrm { s a t } } ) = 1$ (0.5005 for F1, 0.9390 for F2, and 0.7321 for F3), and sgn(x) · min $( P ( t ) , 1 )$ was computed with Horner’s method (on the MCU, the nine Horner stages were manually unrolled). The original work implemented these functions in fixed point on an FPGA and did not report their evaluation cost on an MCU.

Optimization. To avoid unfairly penalizing the approximate implementations, two optimizations were applied. First, the division in the LUT index computation was replaced by multiplication by the reciprocal $\left( n - 1 \right) / 1 0$ determined at compile time (for the 1000-entry nearest-neighbor LUT, the state-update time decreased from 0.702 ms to 0.637 ms). The fraction of inputs whose index shifts by one owing to the difference in rounding was $2 . 1 \times 1 0 ^ { - 4 }$ even for 16384 entries. Second, the segment search of PWL8 was changed from a binary search over the boundary array to a method that exploits odd symmetry and identifies one of the four segments on the |x| side with at most three comparisons (the state-update time decreased from 1.092 ms to 0.592 ms). The mapping is identical before and after the optimization (except for differences of about 1 ULP due to rounding of the intercepts on the negative side).

## B Details of the Measurements on the MCU

## B.1 Measurement environment

An STM32 NUCLEO-F446RE (Cortex-M4, single-precision FPU) was operated at 84 MHz. The firmware was built with arm-none-eabi-gcc 13.3 with optimization -O2 and hardware floating point (-mfloat-abi=hard -mfpu=fpv4-sp-d16). Measured values were sent to the PC via UART.

## B.2 Measurement method

The 500-unit ring reservoir of (4) in the main text was run for 230 steps of scalar input. The input was the NARMA-20 input sequence (uniform random numbers in [0,0.5]). The state-update time per step was measured with the DWT cycle counter (CYCCNT) over the interval that includes computing the pre-activations of all units and evaluating the activation function. The number of cycles per activation was obtained by separately measuring only the evaluation of the activation function, using the actual pre-activation values of each step (including the function call). Interrupts were disabled during the measured intervals. All activations except those of Abe et al. were included in a single binary and measured in turn with common weights and inputs. The functions of Abe et al. were measured in a separate build that additionally included them, and their state-update time ratios were computed relative to tanh(FPU) measured in the same build (note to Table I of the main text). The readout (one output, a 500-dimensional inner product) was not included in the state-update time and was measured separately. The same binary was run four times, and the mean cycle count of each activation was exactly identical across the four runs.

## B.3 Flash usage

Flash usage was obtained by building, for each activation, a binary that contains only that activation and taking the sum of text and data reported by arm-none-eabi-size. Owing to differences in the build environment, the increments have an uncertainty of up to about 8 B.

Table B1  
Measured values for all activations (500 units, per step)
<table><tr><td>Activation</td><td>Cycles/eval.</td><td>State-update time [ms]</td><td>Flash [B]</td><td>Increment from smallest [B]</td></tr><tr><td>hardtanh</td><td>25.0</td><td>0.513</td><td>13,440</td><td>16</td></tr><tr><td>MulClip d = 2</td><td>30.0</td><td>0.543</td><td>13,464</td><td>40</td></tr><tr><td>MulClip d = 3</td><td>31.0</td><td>0.549</td><td>13,480</td><td>56</td></tr><tr><td>tanh(FPU)</td><td>203.2</td><td>1.615</td><td>14,304</td><td>880</td></tr><tr><td>Rational 3/2</td><td>38.0</td><td>0.638</td><td>13,480</td><td>56</td></tr><tr><td>Rational 5/4</td><td>49.0</td><td>0.704</td><td>13,520</td><td>96</td></tr><tr><td>LUT nearest 10</td><td>44.5</td><td>0.629</td><td>13,544</td><td>120</td></tr><tr><td>LUT nearest 100</td><td>44.6</td><td>0.630</td><td>13,904</td><td>480</td></tr><tr><td>LUT nearest 1000</td><td>45.8</td><td>0.637</td><td>17,512</td><td>4,088</td></tr><tr><td>LUT nearest 4096</td><td>44.0</td><td>0.626</td><td>29,880</td><td>16,456</td></tr><tr><td>LUT nearest 16384</td><td>44.0</td><td>0.626</td><td>79,032</td><td>65,608</td></tr><tr><td>LUT interp. 16</td><td>54.1</td><td>0.687</td><td>13,608</td><td>184</td></tr><tr><td>LUT interp. 64</td><td>55.4</td><td>0.694</td><td>13,800</td><td>376</td></tr><tr><td>LUT interp. 256</td><td>55.6</td><td>0.695</td><td>14,576</td><td>1,152</td></tr><tr><td>PWL8</td><td>38.3</td><td>0.592</td><td>13,584</td><td>160</td></tr><tr><td>softsign</td><td>20.0</td><td>0.531</td><td>13,424</td><td>0</td></tr><tr><td>ReLU</td><td>21.0</td><td>0.490</td><td>13,424</td><td>0</td></tr><tr><td>Abe F1</td><td>31.0</td><td>0.549</td><td>13,536</td><td>112</td></tr><tr><td>Abe F2</td><td>70.0</td><td>0.781</td><td>13,552</td><td>128</td></tr><tr><td>Abe F3</td><td>48.4</td><td>0.652</td><td>13,552</td><td>128</td></tr></table>

Note: The evaluation of tanh(FPU) accounted for 74.9% of the state-update time. The functions of Abe et al. were measured in a separate build (Appendix B.2). The computation time of the readout (one output) was 0.051 ms per step.

## B.4 Projection for reservoirs with denser connections (rough estimate)

The measurements in the main text were performed with a ring topology (number of recurrent connections per unit $m = 1 )$ . We roughly estimated, from the measured values, the share of the activation function as m increases. One step with tanh(FPU) took 135,684 cycles, i.e., 271.4 cycles per unit, of which 203.2 cycles were for the activation function and 68.2 cycles for everything else. Assuming about 5 cycles per nonzero element of the sparse matrix (reading the value, column index, and state, and the multiply–accumulate), we added 5 cycles for each increment of m. Under this assumption, the share of tanh(FPU) in the state-update time is about 64% for $m = 1 0 .$ , about 39% for $m = 5 0$ (10% connection density for $N = 5 0 0 )$ , and about 7% for dense connections $( m = 5 0 0 )$ . Replacing tanh(FPU) with MulClip with $d = 3 \mathrm { a t } m = 5 0$ is expected to reduce the state-update time by about 35%. This is an unmeasured rough estimate, and the number of cycles per nonzero element depends on the data layout and implementation.

## C Conditions and Results of the Accuracy Evaluation

## C.1 Reservoir

The 500-unit ring reservoir of (4) in the main text was computed in float32. Training of the readout (ridge regression) and computation of its output were performed in float64. The coupling weight w was searched as the spectral radius, and the input weights $\nu _ { i }$ and biases $b _ { i }$ were drawn from uniform distributions on [−input scale, input scale] and [−bias scale, bias scale], respectively. For KWS, 10-dimensional MFCC vectors were used as input, together with leaky integration with leak rate $a , x ( t + 1 ) = ( 1 - a ) x ( t ) + a f ( \cdot )$

## C.2 Tasks

NARMA-20 is the task of estimating

$$
y ( t + 1 ) = 0 . 2 y ( t ) + 0 . 0 4 y ( t ) \sum _ { j = 0 } ^ { 1 9 } y ( t - j ) + 1 . 5 u ( t ) u ( t - 1 9 ) + 0 . 0 0 1
$$

for a uniform random input $u ( t )$ in [0, 0.5]. We used 200 washout steps, 5,000 training steps, 1,500 validation steps, and 1,500 test steps, and the features consisted of the state augmented with a constant term and the input. The metric is the NMSE over the test segment. Parity8 is the task of determining the parity of the most recent eight bits of a uniform random bit sequence over {0,1}; a sequence was generated for each seed, with 4,000 training, 2,000 validation, and

2,000 test steps (the first 50 steps of each used as washout). The readout output was limited to [0,1], and the task was evaluated by the accuracy of decisions with a threshold of 0.5. For MG, the standardized Mackey–Glass series with $\tau = 1 7$ was used, with 200 washout, 5,000 training, 1,500 validation, and 2,000 test steps. After training for one-step-ahead prediction, the state was updated with the inputs of the validation segment, and then the test segment was predicted autonomously by feeding the predicted output back as the next input; the metric is the number of steps until the error first exceeds 0.5 times the standard deviation of the true values (VPT). For KWS, ten words of Google Speech Commands (yes, no, up, down, left, right, on, off, stop, go) were used with the official validation and test splits. The training data were limited to at most 300 samples per word. Ten MFCCs were extracted from each utterance (16 kHz, 1 s) with a hop length of 1,024, and classification was performed using the state at the final frame augmented with a constant term; the metric is accuracy.

## C.3 Hyperparameter search

For each activation and each task, a 100-trial search was performed with Optuna’s TPE [21]. The objective of the search was the mean validation error over ten search seeds (1000–1009) (NMSE for NARMA-20, 1 − accuracy for Parity8 and KWS, and the NMSE of the first 100 steps of autonomous prediction for MG). For NARMA-20 and MG, the search ranges were spectral radius [0.1,1.2], input scale [0.01,2.0], bias scale [0,1], and ridge coefficient $[ 1 0 ^ { - 6 } , 1 0 ^ { - 1 } ]$ (log-uniform). For Parity8 and the control experiments in Appendix E, because the selected input scale reached the upper bound, only the input-scale range was widened to [0.01,20] (log-uniform). For KWS, the ranges were spectral radius [0, 2], input scale $[ 1 0 ^ { - 3 } , 1 0 ]$ , bias scale [0, 10], leak rate [0.01, 1], and ridge coefficient $[ 1 0 ^ { - 8 } , 1 ]$ (log-uniform except for the spectral radius and bias scale). With the selected hyperparameters, the readout was retrained on the combined training and validation segments, and the test segment was evaluated with 50 seeds (2000–2049) separate from those used in the search. The seeds determine the reservoir weights (and, for Parity8, also the input sequence). The random seed of the search was common to all activations (1234).

Table C1  
Accuracy for each LUT size (mean over 50 seeds)
<table><tr><td>Activation</td><td>NARMA-20 NMSE</td><td>Parity8 accuracy</td><td>MG VPT</td><td>KWS accuracy</td></tr><tr><td>LUT nearest 10</td><td>0.2516</td><td>0.697</td><td>54</td><td>0.679</td></tr><tr><td>LUT nearest 100</td><td>0.2092</td><td>0.911</td><td>100</td><td>0.746</td></tr><tr><td>LUT nearest 1000</td><td>0.1558</td><td>0.980</td><td>208</td><td>0.771</td></tr><tr><td>LUT nearest 4096</td><td>0.1310</td><td>0.979</td><td>301</td><td>0.772</td></tr><tr><td>LUT nearest 16384</td><td>0.1104</td><td>0.990</td><td>391</td><td>0.776</td></tr><tr><td>LUT interp. 16</td><td>0.1439</td><td>0.763</td><td>435</td><td>0.757</td></tr><tr><td>LUT interp. 64</td><td>0.1271</td><td>0.736</td><td>401</td><td>0.778</td></tr><tr><td>LUT interp. 256</td><td>0.0984</td><td>0.746</td><td>469</td><td>0.779</td></tr></table>

Note: Table I of the main text shows only the values for the smallest and largest table sizes. The accuracies of the other activations are as given in Table I of the main text.

## D Details of the Effect of Folding

## D.1 Difference in accuracy with and without folding

MulClip (clipping threshold $c _ { d } )$ and the monotonic-clip variant (clipping threshold $d / ( d - 1 ) )$ differ only in the value of the clipping threshold, and the hyperparameters of each were searched with the same procedure as in Appendix C. The accuracies of both are shown in Table I of the main text. On Parity8, folding improved the accuracy by 0.217 for $d = 2$ and by 0.187 for $d = 3 ,$ . On NARMA-20, the results of the two, including the selected hyperparameters, were exactly identical. The differences in MG VPT were 1 step for $d = 2$ and 17 steps for $d = 3$ . On KWS with $d = 3$ , MulClip was lower by 0.0043; the origin of this difference is examined in Appendix D.3.

## D.2 Fractions of pre-activations in each regime

For MulClip, the 50-seed evaluation was rerun with the selected hyperparameters, and the pre-activation values input to the activation function were recorded for all steps and all units in the training and evaluation segments (including washout). Table D1 shows the fraction of pre-activations whose absolute value exceeded the extremum $d / ( d - 1 )$ (the sum of the folding and zero regimes) and its breakdown.

Table D1  
Distribution of the pre-activations of MulClip (50 seeds, all steps and all units in training and evaluation)
<table><tr><td></td><td></td><td>Beyond the extremum</td><td>Folding regime  $( d / ( d - 1 ) < | x | < c _ { d } )$ </td><td>Zero regime</td></tr><tr><td>Task</td><td> $d$ </td><td> $( | x | > d / ( d - 1 ) )$ </td><td></td><td> $( | x | \geq c _ { d } )$ </td></tr><tr><td>NARMA-20</td><td>2</td><td>0%</td><td>0% 0%</td><td>0% 0%</td></tr><tr><td>NARMA-20 Parity8</td><td>3 2</td><td>0% 30.1%</td><td>19.3%</td><td>10.8%</td></tr><tr><td>Parity8</td><td>3</td><td>26.8%</td><td>16.6%</td><td>10.2%</td></tr><tr><td>MG</td><td>2</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td>MG</td><td>3</td><td>0.36%</td><td>0.36%</td><td>0%</td></tr><tr><td>KWS</td><td>2</td><td>0.0001%</td><td>0.0001%</td><td>0%</td></tr><tr><td>KWS</td><td>3</td><td>9.8%</td><td>9.8%</td><td>0%</td></tr></table>

Note: The extremum $d / ( d - 1 )$ is 2 for $d = 2$ and 1.5 for $d = 3 ,$ and $c _ { d }$ is 4 for $d = 2$ and about 2.60 for $d = 3 .$ The fraction in the folding regime is the fraction beyond the extremum minus the fraction in the zero regime. 0% indicates that none of the recorded values satisfied the condition.

On NARMA-20, the pre-activations never entered either the folding regime or the zero regime. In this case, the output of MulClip is identical to that of the monotonic-clip variant, which is consistent with the exactly identical results of the two in Appendix D.1. On Parity8, 27–30% of the pre-activations exceeded the extremum. The breakdown is 19.3% $( d = 2 )$ and 16.6% (d = 3) of all pre-activations in the folding regime, and 10.8% and 10.2% in the zero regime. The contributions of the two regimes are separated in Appendix D.4. On MG, the fraction beyond the extremum was 0.4% or less, and on KWS with d = 3, 9.8% entered the folding regime. Which regime is used is determined by the set of hyperparameters selected by the search.

## D.3 Verification by swapping hyperparameters

To determine whether the difference on KWS (d = 3) in Appendix D.1 stems from the difference in functions or from the difference in the operating points reached by the searches, the hyperparameters selected for the monotonic-clip variant were given to MulClip and evaluated with the same 50 seeds. The mean accuracy was 0.785, which agreed with the value of the monotonic-clip variant itself (0.785) to the sixth decimal place. That is, the operating point of the monotonic-clip variant is also contained in MulClip, and the difference in Appendix D.1 arises because the search for MulClip reached a different operating point that uses the folding regime (9.8% in Appendix D.2). Conversely, when the hyperparameters selected for MulClip were given to the monotonic-clip variant, the accuracy decreased to 0.769.

## D.4 Sensitivity analysis of the clipping threshold and separation of the folding and zero regimes

In the main text, the clipping threshold is fixed at $c _ { d }$ and is not searched for each task. Here, for reference, we show the accuracy when the clipping threshold is varied around $c _ { d }$ . This is intended to check the validity of the fixed $c _ { d }$ post hoc, and all results in the main text use $c _ { d } .$ . For each clipping threshold, the hyperparameters were searched again with the same procedure as in Appendix C.

Table D2  
Accuracy on Parity8 $( d = 2 )$ for different clipping thresholds c and for the zero-regime-only variant (mean over 50 seeds)
<table><tr><td>C</td><td>Response shape</td><td>Accuracy</td></tr><tr><td>2.0</td><td>Monotonic (monotonic-clip variant)</td><td>0.764</td></tr><tr><td>3.0</td><td>Folding stops partway</td><td>0.877</td></tr><tr><td>4.0</td><td> $c _ { d }$  (MulClip in the main text)</td><td>0.981</td></tr><tr><td>5.0</td><td>Beyond  $c _ { d }$  (with a sign-inverted region)</td><td>0.999</td></tr><tr><td>4.0</td><td>Zero regime only (constant 1 from the extremum to  $c _ { d } ,$  0 outside)</td><td>0.810</td></tr></table>

On Parity8, as the clipping threshold was widened from the extremum (2.0 for d = 2) to $c _ { d } ,$ , the accuracy increased from 0.764 to 0.877 and then to 0.981. This indicates that a wider folding range is more advantageous for Parity8, consistent with the results in Section III-D of the main text.

Separation of the folding and zero regimes. For Section III-D of the main text, a variant with only the zero regime was also evaluated with the same procedure (Appendix C). This variant returns the same response as the monotonic-clip variant for $| x | \leq d / ( d - 1 )$ , a constant ±1 for $d / ( d - 1 ) < | x | < c _ { d }$ , and 0 for $| x | \geq c _ { d } .$ , and has no folding regime. On the other hand, the variant with $c = 3$ (the “variant with only part of the folding regime” in the main text) has no zero regime, and its output is constant at 0.75 for $| x | > 3$ . Their accuracies were 0.810 and 0.877, respectively, exceeding the monotonic-clip variant (0.764) but falling short of MulClip (0.981). Hence, the folding and zero regimes each raise the Parity8 accuracy on their own, but the accuracy is highest with the MulClip response, which has both.

With $c = 5 . 0 \mathrm { . }$ , beyond $c _ { d } ,$ the accuracy was even higher at 0.999. Therefore, $c _ { d }$ is not the value that maximizes the accuracy on Parity8. We nevertheless use $c _ { d }$ because $c _ { d }$ is determined not by accuracy but by a property of the response shape. For $c > c _ { d } .$ , the sign of the output is opposite to that of the input in the range where |x| exceeds $c _ { d } .$ . Furthermore, the absolute value of the output exceeds 1 for $c > 2 + 2 { \sqrt { 2 } } \approx 4 . 8 3$ when $d = 2 \ ( \mathrm { f o r } \ c = 5 . 0 , f ( \pm 5 ) = \mp 1 . 2 5 )$ , and likewise for $c > 3$ when $d = \bar { 3 }$ . That is, $c _ { d }$ is the largest clipping threshold for which the sign of the output agrees with that of the input, and the property stated in Section II of the main text, “the output always lies in $[ - 1 , 1 ] , ^ { \dag }$ also holds at $c _ { d } .$ Pursuing accuracy beyond $c _ { d }$ would mean returning to the practice of searching the clipping threshold for each task, which is outside the scope of the claims of the main text.

## E Control Experiments (Count8, Parity-k)

## E.1 Memory-length control (Count8)

We give the details of the task described in Section III-D of the main text, predicting the number of 1s in the most recent eight bits (hereafter Count8). The same input sequence as Parity8 (uniform random bit sequence over {0,1} generated with the same seeds), data lengths, washout, readout, and search procedure were used, and only the target was changed. The target is $s / 8$ , the number s of 1s in the most recent eight bits divided by 8, evaluated by NMSE. The target of Parity8 is the parity of s.

Table E1
<table><tr><td colspan="2">NMSE on Count8 (mean over 50 seeds)</td></tr><tr><td>Activation</td><td>Count8 NMSE</td></tr><tr><td>MulClip  $d = 2$ </td><td> $3 . 3 1 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>MulClip  $d = 3$ </td><td> $1 . 1 2 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>Monotonic clip  $d = 2$   $d = 3$ </td><td> $3 . 3 1 \times 1 0 ^ { - 8 }$   $1 . 1 2 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>Monotonic clip</td><td></td></tr><tr><td>tanh(FPU)</td><td> $1 . 0 6 \times 1 0 ^ { - 8 }$ </td></tr></table>

All activations, including the monotonic-clip variants, reached an NMSE of $4 \times 1 0 ^ { - 8 }$ or less on Count8. Therefore, the number of 1s in the most recent eight bits can be linearly read out from the state of the 500-unit ring reservoir, and the low accuracy of the monotonic-clip variant on Parity8 is hard to explain by insufficient memory length alone. However, we have not verified whether the information of each bit required for parity is retained, nor the contributions of individual units.

## E.2 Parity with different numbers of bits (Parity-k)

To check that the Parity8 results are not specific to eight bits, we evaluated Parity-k $( k = 2 , 3 , 4 , 5 , 6 , 7 , 9 , 1 0 )$ , whose target is the parity of the number of 1s in the most recent k bits. Apart from k, the conditions were the same as for Parity8, and the hyperparameters were searched again for each activation at each k. For $k \leq 6 ,$ , the accuracies of all 22 configurations were 0.994 or higher, with no difference among activations; Table E2 therefore shows the results for $k \geq 7$ The $k = 8$ column contains the values of Table I of the main text and Table C1.

Table E2  
Accuracy on Parity-k (k ≥ 7) (mean over 50 seeds)
<table><tr><td>Activation</td><td> $k = 7$ </td><td> $k = 8$ </td><td> $k = 9$ </td><td> $k = 1 0$ </td></tr><tr><td>MulClip  $d = 2$ </td><td>1.000</td><td>0.981</td><td>0.802</td><td>0.637</td></tr><tr><td>MulClip d = 3</td><td>1.000</td><td>0.968</td><td>0.754</td><td>0.625</td></tr><tr><td>Monotonic clip d = 2</td><td>0.972</td><td>0.764</td><td>0.625</td><td>0.559</td></tr><tr><td>Monotonic clip d = 3</td><td>0.960</td><td>0.781</td><td>0.637</td><td>0.568</td></tr><tr><td>tanh(FPU)</td><td>0.937</td><td>0.761</td><td>0.622</td><td>0.561</td></tr><tr><td>Rational 3/2</td><td>0.949</td><td>0.767</td><td>0.624</td><td>0.564</td></tr><tr><td>Rational 5/4</td><td>0.935</td><td>0.752</td><td>0.620</td><td>0.562</td></tr><tr><td>LUT nearest 10</td><td>0.891</td><td>0.697</td><td>0.600</td><td>0.544</td></tr><tr><td>LUT nearest 100</td><td>0.996</td><td>0.911</td><td>0.709</td><td>0.579</td></tr><tr><td>LUT nearest 1000</td><td>1.000</td><td>0.980</td><td>0.774</td><td>0.599</td></tr><tr><td>LUT nearest 4096</td><td>1.000</td><td>0.979</td><td>0.796</td><td>0.647</td></tr><tr><td>LUT nearest 16384</td><td>1.000</td><td>0.990</td><td>0.798</td><td>0.660</td></tr><tr><td>LUT interp. 16</td><td>0.988</td><td>0.763</td><td>0.623</td><td>0.561</td></tr><tr><td>LUT interp. 64</td><td>0.962</td><td>0.736</td><td>0.620</td><td>0.561</td></tr><tr><td>LUT interp. 256</td><td>0.944</td><td>0.746</td><td>0.619</td><td>0.562</td></tr><tr><td>PWL8</td><td>0.985</td><td>0.777</td><td>0.624</td><td>0.561</td></tr><tr><td>hardtanh</td><td>0.997</td><td>0.803</td><td>0.642</td><td>0.566</td></tr><tr><td>softsign</td><td>0.914</td><td>0.734</td><td>0.602</td><td>0.546</td></tr><tr><td>ReLU</td><td>0.997</td><td>0.795</td><td>0.598</td><td>0.531</td></tr><tr><td>Abe F1</td><td>1.000</td><td>0.848</td><td>0.651</td><td>0.561</td></tr><tr><td>Abe F2</td><td>0.958</td><td>0.750</td><td>0.605</td><td>0.544</td></tr><tr><td>Abe F3</td><td>0.998</td><td>0.844</td><td>0.653</td><td>0.568</td></tr></table>

For $k \geq 7 ,$ the accuracy decreased as k increased, and the following trends were observed. First, MulClip outperformed the monotonic-clip variant of the same d for all k = 7–10 (the differences were 0.028–0.217 for $d = 2$ and 0.040–0.187 for $d = 3 )$ . Second, for $k = 8 – 1 0 .$ , MulClip outperformed all monotonic activations other than the nearest-neighbor LUTs. Third, the 4096- and 16384-entry nearest-neighbor LUTs were comparable to or better than MulClip with d = 2. Hence, MulClip’s advantage over the monotonic-clip variant is not specific to $k = 8$ and was observed for all of k = 7–10, where the decision becomes difficult.

## F Agreement Between the PC and the Device, and On-Device Inference

## F.1 Agreement of states

Because the accuracies in the main text were obtained by float32 simulation on the PC, we confirmed, for four representative activations (hardtanh, MulClip d = 2, MulClip d = 3, and tanh(FPU)), that the computation on the device agrees with that on the PC. For this check, we used the same 500-unit ring reservoir as in Appendix B, the same weights as the PC implementation, and the same input (230 steps of the NARMA-20 input sequence). On the device, the states of all units at steps 0–4 and 200–229 were output via UART as raw floating-point bit patterns (in hexadecimal) and compared with the states computed under the same conditions on the PC (500 units × 35 steps = 17,500 elements).

For hardtanh, the states agreed bitwise for all compared elements. Because hardtanh consists only of comparisons, this agreement shows that the weight layout and the state-update expression are identical to those of the PC implementation For tanh(FPU), the maximum absolute error was $7 . 0 \dot { \times } 1 0 ^ { - 3 }$ . This difference results from differences of a few ULPs, due to the different algorithms of newlib’s tanhf and tanh on the PC (NumPy), accumulating through the recurrent connections.

MulClip was checked with the same $c _ { d }$ as in the main text. Because the pre-activations do not reach the folding regime with the weights of Appendix B, only the input scale was changed to 10.0 so that the pre-activations entered both the folding and zero regimes (folding regime: 24.5% for d = 2 and 35.0% for $d = 3 ;$ zero regime: 2.5% and 14.7%, respectively). The maximum absolute error relative to the float32 computation on the PC was $5 . 2 \times 1 0 ^ { - 7 }$ for $d = 2$ and $2 . \overset { \cdot } { 0 } \times 1 0 ^ { - 6 } \mathrm { f o r } d = 3$ . These differences arise because the compiler (-O2) combined the multiply–accumulate and the final subtraction of MulClip into fused multiply–add (FMA) instructions, reducing the number of roundings by one. When these fusions were emulated on the PC, the states agreed bitwise for all 17,500 elements for both d = 2 and $d = 3$

For the approximate implementations in Appendix A, the C code of the firmware was built on the PC, and we confirmed that the outputs for 20,001 input points agreed with those of the PC implementation (except for index shifts due to the optimization in Appendix A and differences within $6 \times 1 0 ^ { - 8 } )$

## F.2 On-device inference (NARMA-20 and Parity8)

In addition to the agreement of states, for the two main tasks, inference over the evaluation segment was performed on the device and the metrics were obtained directly. The readout was trained on the PC with the same procedure as in the main text (same hyperparameters and training segment), and its coefficients, together with the input weights, biases, and inputs of the evaluation segment, were placed in flash (about 260 KB). On the device, the state update and readout over the evaluation segment (for NARMA-20, 1,500 of the 1,700 steps excluding the first 200; for Parity8, 1,950 of the 2,000 steps excluding the first 50) were executed in float32, and the predicted values were output via UART as raw bit patterns. The metrics were computed on the PC from the same target values. The configurations were eight in total: tanh(FPU) and MulClip (d = 2,3) for NARMA-20, and these plus the monotonic-clip variants (d = 2,3) for Parity8, using five of the evaluation seeds (2000–2004).

Table F1  
Comparison of on-device inference and PC computation (mean over five seeds)
<table><tr><td>Task</td><td>Activation</td><td>Device</td><td>PC</td></tr><tr><td>NARMA-20 (NMSE)</td><td>tanh(FPU)</td><td>0.08340</td><td>0.08340</td></tr><tr><td>NARMA-20 (NMSE)</td><td>MulClip d = 2</td><td>0.08999</td><td>0.08999</td></tr><tr><td>NARMA-20 (NMSE)</td><td>MulClip d = 3</td><td>0.06472</td><td>0.06472</td></tr><tr><td>Parity8 (accuracy)</td><td>tanh(FPU)</td><td>0.7483</td><td>0.7487</td></tr><tr><td>Parity8 (accuracy)</td><td>Monotonic clip d = 2</td><td>0.7535</td><td>0.7518</td></tr><tr><td>Parity8 (accuracy)</td><td>Monotonic clip d = 3</td><td>0.7588</td><td>0.7589</td></tr><tr><td>Parity8 (accuracy)</td><td>MulClip d = 2</td><td>0.9825</td><td>0.9825</td></tr><tr><td>Parity8 (accuracy)</td><td> $\mathbf { M u l C l i p } d = 3$ </td><td>0.9670</td><td>0.9670</td></tr></table>

Note: The PC values are those of the procedure in the main text, with the readout computed in float64. Because they are means over five seeds, they differ from the 50-seed means in Table I of the main text.

The differences between the device and PC values arise because the device computes the readout in float32 and because of slight differences in the states (the fused multiply–add and the tanhf differences in Appendix F.1). Accordingly, the main accuracy comparisons in the main text (the advantage of d = 3 on NARMA-20 and the effect of folding on Parity8) were reproduced with nearly the same values by on-device inference. However, only five seeds were checked, and MG and KWS were not run on the device.