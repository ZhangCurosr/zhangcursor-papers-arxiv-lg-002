# Physical Muon: Orthogonalization as an Equilibrium Computation

Yuren Hao

yurenh2@illinois.com

University of Illinois at Urbana-Champaign

## Abstract

Physical neural networks and analog in-memory computing could reduce the energy cost of neural network training. Realizing this potential, however, requires optimizers that combine efective learning with physical implementability. SGD fits local analog updates but struggles on transformers, while Adam family is unstable against analog bias. Muon ofers strong training performance, but its Newton–Schulz orthogonalization relies on dense matrix-matrix products. To address this obstacle, we introduce Physical Muon, which computes the orthogonalization as the equilibrium of a continuous-time flow. Random probes approximate the flow using matrix-vector products, reciprocal reads, and local rank-1 writes. To test whether this replacement preserves training performance, we evaluate it on a 10.95M-parameter transformer. The dense flow’s mean validation cross-entropy is 0.0085 above Newton–Schulz across nine seeds per method; the probe implementation is 0.0188 above the control across two seeds. Circuit simulations further reproduce the flow dynamics and yield comparable training behavior.

## 1. Introduction

Physical neural networks and analog in-memory computing ofer a route to reducing the energy cost of neural computation (Momeni et al., 2025; Wright et al., 2022; Sebastian et al., 2020). For example, resistive arrays store weights as conductances and compute matrix-vector products by summing currents, avoiding the movement of individual weights to a digital processor (Gokmen and Vlasov, 2016; Ambrogio et al., 2018). Extending these advantages to training, however, requires an optimizer that both learns efectively and operates within the constraints of the physical hardware.

SGD’s local updates fit resistive arrays (Gokmen and Vlasov, 2016; Gokmen and Haensch, 2020), but its performance on transformers can fall substantially behind Adam (Zhang et al., 2024). Adam’s adaptive updates address this training-quality gap by maintaining first and second moments and normalizing each parameter separately (Kingma and Ba, 2015; Loshchilov and Hutter, 2019). However, systematic analog bias (Dillavou et al., 2025) can destabilize Adam’s adaptive updates. Physical training therefore requires an optimizer that combines efective learning with robustness to analog bias.

These requirements motivate Muon, which orthogonalizes its momentum matrix and has demonstrated better training compute eficiency than AdamW in language models (Jordan et al., 2024; Liu et al., 2025). Its standard Newton–Schulz (NS) implementation, however, requires dense matrix-matrix products. The target resistive arrays lack this native operation and must decompose a dense product into repeated matrix-vector reads (Gokmen and Vlasov, 2016; Sebastian et al., 2020). To overcome this obstacle, we introduce Physical Muon, which computes the orthogonalization through a continuous-time relaxation directly expressible with array reads and local rank-1 writes.

To evaluate this formulation, we compare dense numerical integration and a randomprobe implementation with NS on a 10.95M-parameter transformer. Both obtain validation loss close to NS, and ablations separately test output reuse and orthogonalization accuracy. These training experiments use digital simulation and standard backpropagation. We then assess physical implementability through device-error tests and circuit simulations, including a small synthetic task with the circuit in the training loop.

Related work. Recent work accelerates digital orthogonalization with improved polynomials and hardware-aware reformulations (Amsel et al., 2026; Zhang et al., 2026), or uses temporal information (Dev et al., 2026). Continuous-time matrix flows are classical (Oja, 1982; Brockett, 1988), and recent work studies Muon’s parameter trajectory across steps (Peyr´e, 2026; Mustafi et al., 2026; Beneventano et al., 2026). Our contribution is an analog execution model for Muon’s per-step orthogonalization, validated through training and circuit simulation. Physical Muon complements methods that obtain gradients through physical relaxation (Scellier and Bengio, 2017; Dillavou et al., 2022, 2024).

## 2. Physical Muon

At each optimizer step, Muon accumulates the gradient $g _ { t }$ into a momentum bufer $m _ { t } =$ $\mu m _ { t - 1 } + g _ { t }$ and forms the Nesterov momentum $u _ { t } = g _ { t } + \mu m _ { t }$ . It orthogonalizes $u _ { t }$ to obtain $O _ { t } ,$ then updates the weight matrix as $W _ { t + 1 } = ( 1 - \operatorname { l r } \lambda ) W _ { t } - \operatorname { l r } s O _ { t }$ , with learning rate lr, weight decay λ, and shape-dependent scale s (Jordan et al., 2024; Liu et al., 2025). For full-rank $u _ { t } = U \Sigma V ^ { T }$ , the target is the polar factor $U V ^ { T }$ , which preserves the singular vectors and maps the singular values to one (Higham, 1986). This is the steepest descent direction under a spectral-norm constraint (Bernstein and Newhouse, 2024); standard Muon approximates it with NS iterations.

Equilibrium formulation. To compute this transformation through relaxation, we hold the current momentum fixed and evolve an auxiliary matrix X from zero:

$$
\dot { X } = \hat { M } - X X ^ { T } \hat { M } , \qquad X ( 0 ) = 0 , \qquad \hat { M } = u _ { t } / \alpha ( u _ { t } ) .\tag{1}
$$

Here $\alpha ( u _ { t } ) > 0$ is a scalar normalizer. Starting from zero keeps X in the singular-vector basis of M<sup>ˆ</sup> , so each singular value evolves independently:

$$
\dot { d } _ { i } = \hat { s } _ { i } ( 1 - d _ { i } ^ { 2 } ) , \qquad d _ { i } ( t ) = \operatorname { t a n h } ( \hat { s } _ { i } t ) ,\tag{2}
$$

where $\hat { s } _ { i }$ is a singular value of $\hat { M }$ . Every nonzero mode approaches one, giving the polar factor at equilibrium; zero modes remain zero, as in odd-polynomial NS iterations. At finite time, smaller modes remain attenuated. Section 3 evaluates the resulting efect on training by varying the integration time and comparing the flow with NS5 and exact orthogonalization.

An analog implementation also needs a practical way to normalize the momentum. The Frobenius norm $\left\| u _ { t } \right\| _ { F } ,$ computed from a sum of squared entries, ofers an alternative to computing the largest singular value $\sigma _ { \mathrm { m a x } }$ . For any positive scalar normalizer, (2) preserves the limiting polar factor while rescaling the convergence rates. Relative to our $\sigma _ { \mathrm { m a x } }$ -normalized reference, Frobenius normalization slows the modes by $\| u _ { t } \| _ { F } / \sigma _ { \operatorname* { m a x } }$ , whose measured median is 2.07 (Appendix B). We therefore double the solve budget in the normalization ablation of Section 3 to test this hardware-compatible alternative.

For numerical evaluation we use T Euler iterations with step size η and return $O _ { t } = X _ { T }$

$$
X _ { k + 1 } = \Pi _ { [ - r , r ] } \Big [ X _ { k } + \eta \big ( \hat { M } - X _ { k } X _ { k } ^ { T } \hat { M } \big ) \Big ] ,\tag{3}
$$

where the element-wise clip models circuit supply rails and is inactive in the reported runs. We reset X at each optimizer step. Since $\operatorname { t a n h } ( 1 . 8 ) \approx 0 . 9 5$ , modes above $\hat { s } ^ { * } = 1 . 8 / ( \eta T )$ reach approximately 95% of their limiting value in the continuous-time solution. Figure 2 shows how this threshold shifts with the solve budget. The flow can also stop when its relative change falls below a threshold, ofering a measurable tradeof between iteration count and training loss.

Analog execution. Random probes express the dense increment in (3) through array reads and local writes. For a probe v with independent ±1 entries (Hutchinson, 1990), we compute

$$
p = \hat { M } \boldsymbol { v } , \qquad \boldsymbol { q } = \boldsymbol { X } ^ { T } \boldsymbol { p } , \qquad r = \boldsymbol { X } q , \qquad \Delta \boldsymbol { X } = \eta ( p - r ) \boldsymbol { v } ^ { T } .\tag{4}
$$

Since $\mathbb { E } [ v v ^ { T } ] = I$ , the expected write at fixed X is $\eta ( \hat { M } - X X ^ { T } \hat { M } )$ , the dense Euler increment. The transposed product reads the same X array in reverse, using reciprocity; cell (i, j) is updated from only the row signal $p _ { i } - r _ { i }$ and column signal $v _ { j }$ (Gokmen and Vlasov, 2016; Gokmen and Haensch, 2020; Ambrogio et al., 2018). We average K probes per iteration, giving a budget of $B = 3 K T$ array passes, counted separately for each probe channel.

Figure 1 maps these operations to an array. Capacitors store the evolving state $X ;$ the signed row error charges each cell with a polarity set by its probe entry. The converged state supplies the weight update through the same read and rank-1 write primitives. We use zero initialization to obtain the singular-mode dynamics in (2); Appendix B reports a preliminary warm-start comparison.

![](images/aff5c0d9487c7490ae8ff7ee225a31ad468590f51400c7b15e39e8d479e1435d.jpg)  
Figure 1: Physical execution of (4). Left: reciprocal reads form the row error for local rank-1 writes. Right: a portion of the X array and an enlarged behavioral cell. A capacitor stores $X _ { i j }$ as a voltage that controls its efective read conductance; signed write currents update the state. Section 4 details circuit validation.

## 3. Training Evaluation

We test the replacement on a 12-layer, width-128 transformer (10.95M parameters), trained on FineWeb (Penedo et al., 2024) for 2,500 steps with batch size 24 and sequence length 256. We change only the orthogonalization, keeping standard backpropagation and the optimizer settings fixed: Muon uses momentum 0.95 and learning rate 0.016 on matrix parameters; AdamW updates the remaining parameters. We report best validation cross-entropy (CE), where lower is better. NS5 denotes five Newton–Schulz iterations. Full settings and per-seed results are in Appendix C.

Equilibrium-flow equivalence. We first test the relaxation itself using the dense flow, which evaluates the full matrix update in (3) with $\eta = 0 . 5$ and $T = 4 0 0$ . We assess practical equivalence with a CE margin of $\delta = 0 . 0 4 2 6$ , the observed cost of reusing an NS5 output for one additional step (Section 3.1). Across nine seeds per method, the dense flow passes the two one-sided tests (TOST) procedure (Schuirmann, 1987). Its mean gap is 0.0085 and its one-sided 95% upper bound is 0.0142, both within this margin (Table 1).

Table 1: We perform an equivalence test using the two one-sided tests (TOST) procedure and find that NS5 and the dense flow have statistically equivalent validation CE within a margin of $\pm 0 . 0 4 2 6$ $( p = 2 . 4 \times 1 0 ^ { - 8 } )$ . CE is mean ± sample std across n training seeds. $\Delta \mathrm { C E }$ is relative to NS5; the upper bound is its one-sided 95% confidence bound.
<table><tr><td>Solver</td><td>n</td><td>Validation CE</td><td>ΔCE</td><td>Upper bound</td></tr><tr><td>NS5 control</td><td>9</td><td> $5 . 0 1 2 4 \pm 0 . 0 0 8 1$ </td><td></td><td></td></tr><tr><td>Dense flow</td><td>9</td><td> $5 . 0 2 0 9 \pm 0 . 0 0 5 4$ </td><td> $+ 0 . 0 0 8 5$ </td><td> $0 . 0 1 4 2$ </td></tr></table>

Hardware-compatible normalization. The normalization ablation changes the dense flow’s normalizer from $\sigma _ { \mathrm { m a x } }$ to $\| u _ { t } \| _ { F }$ and doubles T to 800 to compensate for slower relaxation. Its CE is $5 . 0 1 7 0 \pm 0 . 0 0 8 3$ across four seeds, and it passes the same equivalence test with an upper bound of 0.0142 (Appendix C). Thus Frobenius normalization provides a practical alternative while preserving training quality within the chosen margin.

Array-operation approximation. We next test the probe flow, which simulates the three reads and local writes in (4) inside transformer training. With $K = 3 2$ probes along the shorter matrix dimension, $\eta = 0 . 1 5$ , and $T = 4 1 7 .$ , it uses about 40k passes per matrix per optimizer step. The two runs give a mean CE of 5.0313, or $+ 0 . 0 1 8 8$ relative to NS5. This pilot tests training with the array-operation sequence; additional seeds would further characterize its variability. Circuit-in-the-loop training is evaluated in Section 4.

## 3.1. Ablations and Generality

To interpret the replacement cost, we separately change output accuracy and freshness. Exact SVD gives a mean CE of 5.0119 across three seeds, close to NS5’s 5.0124. SVD maps nonzero singular values exactly to one; the measured NS5 outputs span 0.74–1.13 (Figure 2, right). Thus exact singular-value scaling ofers little observed benefit at this scale. The dense and probe experiments above directly assess our finite-time approximation. More accurate orthogonalization can improve training at larger scales (Amsel et al., 2026).

Direct reuse behaves diferently. Recomputing the NS5 output every S steps and reusing it in between increasingly degrades training (Figure 2, left). Even $S = 2$ adds 0.0426 CE in the standard configuration. Saved momentum trajectories show rotation of the singular subspaces (Appendix B), explaining why an old output may be inaccurate for the current momentum. This ablation isolates direct reuse. Warm starts and history-based methods use past information to accelerate a fresh computation (Dev et al., 2026; Ahn et al., 2025).

The $S = 2$ penalty supplies the reference margin in Table 1. The dense flow’s upper confidence bound is about one third of this penalty, placing its replacement cost below the measured efect of reusing an output once. This margin is estimated from one seed and depends on the configuration; at batch size 6, reuse has little measured cost (Appendix C). Solve duration. Short flow solves degrade training. At $\eta = 0 . 2 5$ , increasing T from 50 to 200 reduces the CE gap from 0.136 to 0.033. This agrees with (2): longer integration brings more small singular modes near one. An adaptive stop at relative change $3 \times 1 0 ^ { - 3 }$ uses 96.7 iterations on average, versus 400 for the reference, with a gap of +0.030 on one seed. Thus the flow ofers a tunable accuracy–cost tradeof. Figure 4 and the full depth and stopping results are in Appendices B–C.

![](images/dab0ce53e30f879954e38026517d6504d5f6ec5a1050aca0221ffc6eac106f26.jpg)

![](images/9df81c8680c968e14d8f18e567deaad15225e445ce2e1c9a7402e366de70a911.jpg)  
Figure 2: Left: directly reusing NS5 outputs degrades training as the reuse interval grows (one seed). Right: singular-value responses of NS5 and the flow at two integration times; vertical lines mark the approximate 95% thresholds. Shading covers the measured NS5 outputs.

Generality. Across widths corresponding to 10.95M–59M parameters, the dense-flow gap remains between 0.003 and 0.012, with one to nine seeds per setting. Reducing batch size from 24 to 3 tests noisier gradients; gaps range from 0.0078 to 0.0219 with at least three seeds per method. At a nearby learning rate, the dense flow’s mean CE remains similar; extending training to 5,000 steps gives a gap of +0.0010 on one seed. These tests support a modest training cost across the studied settings, although the largest model and longer run each have only one seed (Appendix C).

## 4. Physical Feasibility

The transformer experiments test the optimizer numerically. We now assess whether the array operations can be realized in a circuit, used inside training, and sustained under device errors and realistic operation budgets.

Circuit realization and training. An ngspice transient on an $8 \times 8$ momentum block executes the reciprocal reads and local writes in Figure 1. An 8-bit diferential resistor array stores the momentum, and 64 capacitors store X. Each iteration occupies 1 µs, including 40 ns read settling; the cold-start solve runs as one continuous 600-period transient. Its final state matches the numerical probe implementation at cosine 0.999983 and relative Frobenius distance 0.0059. Adding the modeled circuit non-idealities changes the cosine with NS5 by about 0.002. The $8 \times 8$ array uses behavioral elements; separate transistor-level read tests cover a $2 \times 2$ array (Appendix E).

To test the circuit inside training, we use a four-class synthetic task (Figure 3). The figure compares three implementations: digital NS5 Muon uses five Newton–Schulz iterations;

behavioral flow Muon computes the probe flow with a numerical array model; and SPICEpolar Muon executes the same flow through the ngspice circuit in the optimizer loop. All three reach CE below 0.01 in a similar number of updates and attain 100% training accuracy. This small circuit-in-the-loop experiment complements the digital transformer evaluation.

![](images/e05e4a30a74ac724a95373b96740bc3c2b05a2361f81d4963116e0bee3762668.jpg)

![](images/3edbd39c5b9df43abf8c0f8e2d9e48d63ab63dc4998d1bb439953e659d10fa88.jpg)  
Figure 3: Circuit-in-the-loop training on the four-class synthetic task. Left: training CE for the three Muon implementations defined above. Right: cosine similarity between their orthogonalized update directions and the NS5 reference for each of the two weight matrices. The momentum matrix for $W _ { 2 }$ has rank three.

Device errors. On three saved width-128 momentum shapes, the orthogonalized update from the probe flow reaches cosine similarity 0.9 with the NS5 update within 10k–40k array passes; the 40k budget used in transformer training covers all three shapes. At the selected budgets, 20% persistent per-pass gain error changes this cosine by at most 0.011. At 10% gain error, the cosine between perturbed and clean NS5 updates falls to 0.40–0.45, and PolarExpress diverges on two of the three matrices. A persistent ofset of 1% of signal rms and zero-mean write noise of 30% change the probe output’s cosine with NS5 by at most 0.0004 and 0.007, respectively (Appendix D). These tests quantify the directional robustness of orthogonalized updates on saved momentum matrices.

Energy and scaling. For the GPT-2 124M workload, projected orthogonalization energy is 0.45–1.04 J per optimizer step at 60k array passes per matrix, compared with an estimated 1.38–3.00 J for digital NS5. At 240k passes, the projection rises to 1.81–4.18 J, overlapping the digital range. The potential energy advantage therefore depends on the required solve budget.

These budgets extrapolate the measured 10k–40k passes on width-128 matrices linearly to rank 768. The energy ranges use published ReRAM and phase-change chip costs of 88.9 and 205 fJ per multiply-accumulate, respectively (Hung et al., 2021; Le Gallo et al., 2023); the digital range spans peak-throughput and utilization-adjusted estimates. The corresponding break-even budgets are 104–517 passes per unit rank. These estimates assume resident arrays and exclude momentum loading, output readout, and shared bufer trafic. Appendix D gives the cost model, the original small-matrix budgets, and the latency assumptions.

## 5. Discussion and Future Directions

To our knowledge, Physical Muon provides the first analog-compatible realization of Muon’s orthogonalization, using reciprocal matrix-vector reads and local rank-1 writes. The dense flow meets the chosen practical-equivalence margin in transformer training, and circuitin-the-loop simulations demonstrate these operations on a synthetic task. Open problems include scaling the probe budget to larger matrices, sustaining closed-loop training under measured device errors, and integrating the optimizer with physical gradient computation. These directions connect the present orthogonalization mechanism to end-to-end physical training.

## References

Kwangjun Ahn, Byron Xu, Natalie Abreu, Ying Fan, Gagik Magakyan, Pratyusha Sharma, Zheng Zhan, and John Langford. Dion: Distributed orthonormalized updates. arXiv preprint arXiv:2504.05295, 2025.

Stefano Ambrogio, Pritish Narayanan, Hsinyu Tsai, Robert M. Shelby, Irem Boybat, Carmelo di Nolfo, Severin Sidler, Massimo Giordano, Martina Bodini, Nathan C. P. Farinha, Benjamin Killeen, Christina Cheng, Yassine Jaoudi, and Geofrey W. Burr. Equivalentaccuracy accelerated neural-network training using analogue memory. Nature, 558:60–67, 2018. doi: 10.1038/s41586-018-0180-5.

Noah Amsel, David Persson, Christopher Musco, and Robert M. Gower. The polar express: Optimal matrix sign methods and their application to the Muon algorithm. In International Conference on Learning Representations (ICLR), 2026. doi: 10.48550/arXiv.2505.16932. arXiv:2505.16932.

Pierfrancesco Beneventano, Mahmoud Abdelmoneum, and Tomaso Poggio. The spectral dynamics and noise geometry of Muon. arXiv preprint arXiv:2606.08388, 2026. doi: 10.48550/arXiv.2606.08388.

Jeremy Bernstein and Laker Newhouse. Old optimizer, new norm: An anthology. arXiv preprint arXiv:2409.20325, 2024. doi: 10.48550/arXiv.2409.20325.

Roger W. Brockett. Dynamical systems that sort lists, diagonalize matrices and solve linear programming problems. In Proceedings of the 27th IEEE Conference on Decision and Control, 1988. doi: 10.1109/CDC.1988.194420.

Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, and Quoc V. Le. Symbolic discovery of optimization algorithms. In Advances in Neural Information Processing Systems, 2023. arXiv:2302.06675.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. PaLM: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240), 2023.

Bishnu Dev, Sushil Bohara, Martin Tak´aˇc, and Samuel Horv´ath. CacheMuon: Using temporal preconditioning to approximate polar factor. arXiv preprint arXiv:2606.16371, 2026. doi: 10.48550/arXiv.2606.16371.

Sam Dillavou, Menachem Stern, Andrea J. Liu, and Douglas J. Durian. Demonstration of decentralized physics-driven learning. Physical Review Applied, 18:014040, 2022. doi: 10.1103/PhysRevApplied.18.014040.

Sam Dillavou, Benjamin D. Beyer, Menachem Stern, Andrea J. Liu, Marc Z. Miskin, and Douglas J. Durian. Machine learning without a processor: Emergent learning in a nonlinear analog network. Proceedings of the National Academy of Sciences, 121(28):e2319718121, 2024. doi: 10.1073/pnas.2319718121.

Sam Dillavou, Marcelo Guzman, Andrea J. Liu, and Douglas J. Durian. Understanding and embracing imperfection in physical learning networks. arXiv preprint arXiv:2505.22887, 2025.

Tayfun Gokmen and Wilfried Haensch. Algorithm for training neural networks on resistive device arrays. Frontiers in Neuroscience, 14:103, 2020. doi: 10.3389/fnins.2020.00103.

Tayfun Gokmen and Yurii Vlasov. Acceleration of deep neural network training with resistive cross-point devices: Design considerations. Frontiers in Neuroscience, 10:333, 2016. doi: 10.3389/fnins.2016.00333.

Nicholas J. Higham. Computing the polar decomposition—with applications. SIAM Journal on Scientific and Statistical Computing, 7(4):1160–1174, 1986. doi: 10.1137/0907079.

Je-Min Hung, Cheng-Xin Xue, Hui-Yao Kao, Yen-Hsiang Huang, Fu-Chun Chang, Sheng-Po Huang, Ta-Wei Liu, Chuan-Jia Jhang, Chin-I Su, Win-San Khwa, Chung-Chuan Lo, Ren-Shuo Liu, Chih-Cheng Hsieh, Kea-Tiong Tang, Mon-Shu Ho, Chung-Cheng Chou, Yu-Der Chih, Tsung-Yung Chang, and Meng-Fan Chang. A four-megabit compute-in-memory macro with eight-bit precision based on CMOS and resistive random-access memory for AI edge devices. Nature Electronics, 4:921–930, 2021. doi: 10.1038/s41928-021-00676-9.

Michael F. Hutchinson. A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. Communications in Statistics - Simulation and Computation, 19(2):433–450, 1990. doi: 10.1080/03610919008812866.

Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks. https: //kellerjordan.github.io/posts/muon/, 2024.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015. arXiv:1412.6980.

Manuel Le Gallo, Riduan Khaddam-Aljameh, Milos Stanisavljevic, Athanasios Vasilopoulos, Benedikt Kersting, Martino Dazzi, Geethan Karunaratne, Matthias Braendli, Abhairaj Singh, Silvia Mueller, Julian B¨uchel, Xavier Timoneda, Vinay Joshi, Malte Rasch, Urs Egger, Angelo Garofalo, Anastasios Petropoulos, Theodore Antonakopoulos, Kevin Brew,

Samuel Choi, Injo Ok, Timothy Philip, Victor Chan, Claire Silvestre, Ishtiaq Ahsan, Nicole Saulnier, Vijay Narayanan, Pier Andrea Francese, Evangelos Eleftheriou, and Abu Sebastian. A 64-core mixed-signal in-memory compute chip based on phase-change memory for deep neural network inference. Nature Electronics, 6:680–693, 2023. doi: 10.1038/s41928-023-01010-1.

Kaizhao Liang, Lizhang Chen, Bo Liu, and Qiang Liu. Cautious optimizers: Improving training with one line of code. arXiv preprint arXiv:2411.16085, 2024. doi: 10.48550/ arXiv.2411.16085.

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, and Zhilin Yang. Muon is scalable for LLM training. arXiv preprint arXiv:2502.16982, 2025. doi: 10.48550/arXiv.2502.16982.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. arXiv:1711.05101.

Ali Momeni, Babak Rahmani, Benjamin Scellier, Logan G. Wright, Peter L. McMahon, Clara C. Wanjura, Yuhang Li, Anas Skalli, Natalia G. Berlof, Tatsuhiro Onodera, Ilker Oguz, Francesco Morichetti, Philipp del Hougne, Manuel Le Gallo, Abu Sebastian, Azalia Mirhoseini, Cheng Zhang, Danijela Markovi´c, Daniel Brunner, Christophe Moser, Sylvain Gigan, Florian Marquardt, Aydogan Ozcan, Julie Grollier, Andrea J. Liu, Demetri Psaltis, Andrea Al\`u, and Romain Fleury. Training of physical neural networks. Nature, 645, 2025. doi: 10.1038/s41586-025-09384-2.

Aratrika Mustafi, Soumya Mukherjee, and Bharath K. Sriperumbudur. Move on Muon: A Hamiltonian probability gradient flow perspective of Muon optimizer. arXiv preprint arXiv:2605.23871, 2026. doi: 10.48550/arXiv.2605.23871.

NVIDIA Corporation. NVIDIA H100 tensor core GPU datasheet. https://www.nvidia. com/en-us/data-center/h100/, 2023.

Erkki Oja. Simplified neuron model as a principal component analyzer. Journal of Mathematical Biology, 15:267–273, 1982. doi: 10.1007/BF00275687.

Erik Olieman, Anne-Johan Annema, and Bram Nauta. An interleaved full Nyquist highspeed DAC technique. IEEE Journal of Solid-State Circuits, 50(3), 2015. doi: 10.1109/ JSSC.2014.2387946.

Guilherme Penedo, Hynek Kydl´ıˇcek, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colin Rafel, Leandro Von Werra, and Thomas Wolf. The FineWeb datasets: Decanting the web for the finest text data at scale. In Advances in Neural Information Processing Systems, 2024. arXiv:2406.17557.

Gabriel Peyr´e. Muon dynamics as a spectral Wasserstein flow. arXiv preprint arXiv:2604.04891, 2026. doi: 10.48550/arXiv.2604.04891.

Benjamin Scellier and Yoshua Bengio. Equilibrium propagation: Bridging the gap between energy-based models and backpropagation. Frontiers in Computational Neuroscience, 11: 24, 2017. doi: 10.3389/fncom.2017.00024.

Donald J. Schuirmann. A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. Journal of Pharmacokinetics and Biopharmaceutics, 15(6):657–680, 1987. doi: 10.1007/BF01068419.

Abu Sebastian, Manuel Le Gallo, Riduan Khaddam-Aljameh, and Evangelos Eleftheriou. Memory devices and applications for in-memory computing. Nature Nanotechnology, 15: 529–544, 2020. doi: 10.1038/s41565-020-0655-z.

Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. In International Conference on Machine Learning, 2018. arXiv:1804.04235.

Yu-Neng Wang and Sara Achour. Generative models on analog hardware with dynamics. arXiv preprint arXiv:2606.27294, 2026. doi: 10.48550/arXiv.2606.27294.

Zixiao Wang, Yifei Shen, and Huishuai Zhang. OLion: Approaching the Hadamard ideal by intersecting spectral and ℓ<sub>∞</sub> implicit biases. arXiv preprint arXiv:2602.01105, 2026. doi: 10.48550/arXiv.2602.01105.

Logan G. Wright, Tatsuhiro Onodera, Martin M. Stein, Tianyu Wang, Darren T. Schachter, Zoey Hu, and Peter L. McMahon. Deep physical neural networks trained with backpropagation. Nature, 601:549–555, 2022. doi: 10.1038/s41586-021-04223-6.

Jack Zhang, Noah Amsel, Berlin Chen, and Tri Dao. Gram Newton–Schulz: A fast, hardware-aware Newton–Schulz algorithm for Muon. https://tridao.me/blog/2026/ gram-newton-schulz/, 2026.

Yushun Zhang, Congliang Chen, Tian Ding, Ziniu Li, Ruoyu Sun, and Zhi-Quan Luo. Why transformers need Adam: A Hessian perspective. In Advances in Neural Information Processing Systems, 2024. doi: 10.48550/arXiv.2402.16788. arXiv:2402.16788.

## Appendix A. Optimizer Comparison

Table 2 reports a supplementary optimizer comparison. All arms train the same 12-layer, width-128 transformer for the same number of steps with backpropagation, with no weight decay; the learning rate of each method is swept until the optimum is bracketed. The reported quantity is the best validation CE.

The OLion rows with 0 to 5 Newton-Schulz rounds form an ablation series; three rounds land within 0.026 of five.

## Appendix B. Rotation-Rate Measurements and Telemetry

Rotation of the singular subspace. The measurements use momentum matrices saved every step over 601 steps of a width-128 run, evaluated on three matrix shapes in a latetraining window. For a stale interval of k steps, the NS5 output of the matrix saved k steps earlier is compared with the NS5 output of the current matrix. Table 3 lists the cosine over the full matrix and the cosine after projection onto the current leading-16 singular subspace. Between 117 and 126 of the 128 singular modes carry singular values below half of the leading value.

Table 2: Best validation CE for 11 optimizers under one protocol
<table><tr><td>Optimizer</td><td>CE</td></tr><tr><td>Muon (Jordan et al., 2024) OLion (Wang et al., 2026) OLion, 3 NS rounds</td><td>4.9190 4.9263 4.9523</td></tr><tr><td>OLion, 2 NS rounds AdamW (Loshchilov and Hutter, 2019)</td><td>5.0043</td></tr><tr><td>OLion, 1 NS round</td><td>5.0707 5.1503</td></tr><tr><td>Adafactor (Shazeer and Stern, 2018) Cautious Lion (Liang et al., 2024)</td><td>5.2094</td></tr><tr><td>Lion (Chen et al., 2023)</td><td>5.2163 5.2242</td></tr></table>

Table 3: Cosine between stale and fresh NS5 outputs
<table><tr><td>Stale interval (steps)</td><td>5</td><td>10</td><td>20</td></tr><tr><td>Full matrix</td><td>0.60</td><td>0.39</td><td>0.19</td></tr><tr><td>Leading-16 subspace</td><td>0.89</td><td>0.78</td><td>0.52</td></tr></table>

Normalization. On 174 saved momentum samples across three shapes, $\| u \| _ { F } / \sigma _ { \operatorname* { m a x } }$ ranges from 1.02 to 3.19 (median 2.07), below the worst-case bound ${ \sqrt { \mathrm { r a n k } } } \approx 1 1 . 3$ . Doubling the Frobenius-normalized solve to T = 800 approximately compensates the median slowdown; its training result is reported in Section 3. At $T = 4 0 0$ , one seed gives CE 5.0226; at $T = 1 6 0 0$ one seed gives 5.0161. A multi-seed budget sweep would quantify the tradeof between solve cost and training quality.

Reset comparison. On three saved late-training momentum matrices, a 50-iteration solve initialized from the previous step’s state produces an update with cosine 0.85–0.86 against the fresh NS output; a 100-iteration solve from zero produces an update with cosine 0.94. These measurements use diferent budgets; an equal-cost comparison would isolate the efect of initialization. We use zero initialization for the singular-mode dynamics in (2).

Stopping-criterion telemetry. With the relative update sampled every 10 iterations and a cap of 400, halting at a relative update of $3 \times 1 0 ^ { - 3 }$ requires a mean of 96.7 iterations per matrix (median 93.8, range 72.2–132.5) over 60 matrices; no matrix reaches the cap. Training with this rule gives CE 5.0419 on one seed, +0.030 above the NS control mean.

Depth series. Figure 4 plots the flow–control gap against integration time ηT for the dense and probe implementations. Most depth settings have one seed; the diamond and square aggregate nine and two seeds, respectively.

## Appendix C. Training Protocol and Full Language-Model Results

Protocol. Unless varied explicitly, the language-model experiments use a 12-layer, width-128 transformer (10.95M parameters), trained with standard backpropagation on FineWeb (Penedo et al., 2024) for 2,500 steps at sequence length 256 and batch size 24, in fp32. Muon applies momentum 0.95 with Nesterov and learning rate $1 . 6 \times 1 0 ^ { - 2 }$ to two-dimensional matrix parameters, with no weight decay. AdamW governs the remaining parameters at learning rate $1 0 ^ { - 3 }$ and weight decay $1 0 ^ { - 4 }$ . A 250-step linear warmup precedes cosine decay to 0.1×. All CE values are the best validation CE. The optimizer comparison in Appendix A uses its separately described learning-rate sweeps and weight-decay settings.

![](images/d06f4f8517449fc84579f4ad15ae2508f3bac7f70901d1384001ae5e5e492983.jpg)  
Figure 4: Flow–control CE gap against integration time. Filled circles: $\eta = 0 . 2 5 ;$ ; open circles: $\eta = 0 . 5 ;$ diamond: $T = 4 0 0$ , nine-seed mean ± std; square: probe flow, $K = 3 2 , \eta = 0 . 1 5 , T = 4 1 7$ , two-seed mean. The grey band is the NS5 seed spread.

Primary comparison, per seed. Table 4 lists the individual seeds behind the $n = 9$ comparison in Table 1.

Table 4: Best validation CE per seed, NS5 control and flow $( T = 4 0 0 , \eta = 0 . 5 )$
<table><tr><td>Seed</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td></tr><tr><td>NS5 control</td><td>5.0048</td><td>5.0084</td><td>5.0183</td><td>5.0080</td><td>5.0233</td><td>5.0248</td><td>5.0144</td><td>5.0035</td><td>5.0064</td></tr><tr><td>Flow</td><td>5.0251</td><td>5.0152</td><td>5.0217</td><td>5.0185</td><td>5.0178</td><td>5.0285</td><td>5.0124</td><td>5.0220</td><td>5.0269</td></tr></table>

Exact polar factor. The SVD arm records CE values of 5.0092, 5.0120, and 5.0146 across three seeds (mean 5.0119), against the $5 . 0 1 2 4 \pm 0 . 0 0 8 1$ NS5 control.

Probe implementation. The two training runs use $K = 3 2 , \eta = 0 . 1 5 .$ , and $T = 4 1 7$ , with probes along the shorter side of each matrix. Their CE values are 5.0281 and 5.0344, giving a mean of 5.03125, +0.0188 above the NS5 mean and +0.0103 above the dense flow. The budget is $3 K T = 4 0 { , } 0 3 2$ array passes per matrix per optimizer step.

Equivalence test. With the reference margin $\delta = 0 . 0 4 2 6$ from the standard-configuration reuse test, TOST gives $p = 2 . 4 \times 1 0 ^ { - 8 }$ for the dense flow $( t = 1 0 . 5 7$ , Welch df 14). Its one-sided 95% upper confidence bound on the CE diference is 0.0142; the test passes down to approximately this margin $( p = 0 . 0 4 9 \mathrm { a t } 0 . 0 1 4 2 )$ . A two-sided test detects a small positive CE diference $( p = 0 . 0 2 0 )$ , which lies within the chosen equivalence margin. The four-seed

Frobenius-normalized flow at $T = 8 0 0$ also passes at $\delta = 0 . 0 4 2 6 ~ ( p = 1 . 6 \times 1 0 ^ { - 4 } )$ , with the same rounded upper bound of 0.0142.

Reuse penalty in other configurations. The $S = 2$ penalty that sets the equivalence margin in Section 3.1 is +0.0426 at width 128 (seed 1). At width 256 the paired $S = 2$ penalty is +0.0174, above the 0.0142 bound achieved in Section 3. At batch size 6 the $S = 2$ penalty is +0.0025, inside the seed noise, so the freshness argument applies to the standard batch size.

Depth series. Table 5 lists the integration-depth series from Section 3 together with the approximate mode threshold $\hat { s } ^ { * } = 1 . 8 / ( \eta T )$ of each configuration. An additional single-seed run at T = 800, η = 0.5 gives CE 5.0147, 0.0104 below its paired control.

Table 5: Flow degradation against the NS5 control across integration depths (seed 1)
<table><tr><td>η</td><td> $T$ </td><td> $\eta T$ </td><td> $\hat { s } ^ { * }$ </td><td>∆CE</td></tr><tr><td>0.25</td><td>50</td><td>12.5</td><td>0.144</td><td>+0.136</td></tr><tr><td>0.25</td><td>100</td><td>25</td><td>0.072</td><td>+0.076</td></tr><tr><td>0.25</td><td>200</td><td>50</td><td>0.036</td><td>+0.033</td></tr><tr><td>0.5</td><td>200</td><td>100</td><td>0.018</td><td>+0.021</td></tr><tr><td>0.5</td><td>400</td><td>200</td><td>0.009</td><td>+0.0085*</td></tr></table>

∗ n = 9 mean; the other rows are seed-1 measurements.

Learning rate and training duration. At learning rate $1 . 4 2 \times 1 0 ^ { - 2 }$ , the dense flow gives $\mathrm { C E ~ 5 . 0 2 0 5 } \pm 0 . 0 1 1 9 ~ ( n = 5 )$ , compared with $5 . 0 2 0 9 \pm 0 . 0 0 5 4 \ : ( n = 9 )$ at $1 . 6 \times 1 0 ^ { - 2 }$ . NS5 at $1 . 4 2 \times 1 0 ^ { - 2 } \ \mathrm { o r } \ 1 . 8 \times 1 0 ^ { - 2 }$ is approximately +0.010 worse than at $1 . 6 \times 1 0 ^ { - 2 }$ in single-seed comparisons. Extending training to 5,000 steps reduces the flow–control gap to +0.0010 on one seed.

Gradient-noise series. Table 6 lists the flow–control gap across batch sizes at the NS learning rate. Every arm contains at least three seeds; 17 of 21 paired seed diferences are positive.

Table 6: Flow–control gap across batch sizes
<table><tr><td>Batch size</td><td>24</td><td>12</td><td>6</td><td>4</td><td>3</td></tr><tr><td> $\Delta ( \mathrm { H o w - c t l } )$ </td><td>+0.0085</td><td>+0.0175</td><td>+0.0078</td><td>+0.0219</td><td>+0.0149</td></tr></table>

Width series. Table 7 lists the flow–control gap across model widths. The gap stays within [0.003, 0.012] and shows no monotone direction.

Table 7: Flow–control gap across widths at $T = 4 0 0$
<table><tr><td>Width</td><td>128 (10.95M)</td><td>192</td><td>256 (26.4M)</td><td>384 (59M)</td></tr><tr><td>∆</td><td>+0.0085</td><td>+0.0114</td><td>+0.0034</td><td>+0.0112</td></tr><tr><td>Seeds</td><td>9</td><td>3</td><td>2</td><td>1</td></tr></table>

## Appendix D. Probe Budget Frontier, Device Tolerance, and Energy Accounting

Frontier. Three saved momentum matrices from a width-128 run (a 128 × 128 attention matrix and the $3 8 4 \times 1 2 8$ and $1 2 8 \times 3 8 4$ feed-forward matrices) serve as test inputs. For each budget B of array passes $( B = 3 K T$ for K parallel probe channels and T iterations), the grid over $K \in \{ 8 , 1 6 , 3 2 , 6 4 \}$ and $\eta \in \{ 0 . 1 5 , 0 . 3 , 0 . 5 \}$ is searched with three probe seeds, and Table 8 reports the best cosine between the probe-flow output and the digital NS5 output. The step size $\eta = 0 . 1 5$ is the best choice at every entry. The probe-form code path reproduces the dense flow bitwise when probes are replaced by the identity. For reference, the dense-flow outputs at $T = 4 0 0$ have cosines 0.9833, 0.9854, and 0.9861 with the NS5 outputs on these three shapes, respectively. Cosines restricted to the leading-16 singular subspace range from 0.96 to 0.99 over the probe budgets.

Table 8: Best cosine against NS5 (best K in parentheses) across array-pass budgets; entries above the 0.9 screen in bold
<table><tr><td>Budget B</td><td>2k</td><td>5k</td><td>10k</td><td>20k</td><td>40k</td><td>80k</td></tr><tr><td> $1 2 8 \times 1 2 8$  attention</td><td>0.708 (16)</td><td>0.802 (16)</td><td>0.850 (16)</td><td>0.888 (16)</td><td>0.919 (32)</td><td>0.945 (32)</td></tr><tr><td> $3 8 4 \times 1 2 8$ </td><td>0.806 (8)</td><td>0.896 (8)</td><td>0.930 (16)</td><td>0.951 (16)</td><td>0.965 (32)</td><td>0.973 (64)</td></tr><tr><td> $1 2 8 \times 3 8 4$ </td><td>0.677 (16)</td><td>0.803 (16)</td><td>0.873 (32)</td><td>0.912 (32)</td><td>0.946 (64)</td><td>0.953 (64)</td></tr></table>

Device tolerance. Each defect class is injected into the probe form at the frontier configuration of each matrix (three probe seeds), and Table 9 reports the change in the screen value. Gain errors are persistent per-pass scalars $1 + g \xi$ with ξ drawn once; ofsets are persistent per-pass vectors scaled to the signal rms; write noise is fresh zero-mean noise on each rank-1 write.

Table 9: Change in screen value (cosine against NS5) under injected defects
<table><tr><td rowspan="2"></td><td colspan="3">Persistent gain</td><td colspan="2">Persistent offset</td><td colspan="2">Write noise</td></tr><tr><td>5%</td><td>10%</td><td>20%</td><td>1%</td><td>5%</td><td>10%</td><td>30%</td></tr><tr><td> $1 2 8 \times 1 2 8$  attention</td><td>+0.0004</td><td>+0.0008</td><td>+0.0013</td><td>-0.0004</td><td>-0.0129</td><td>-0.0006</td><td>-0.0052</td></tr><tr><td> $3 8 4 \times 1 2 8$ </td><td>+0.0002</td><td>+0.0003</td><td>+0.0001</td><td>-0.0001</td><td>-0.0014</td><td>-0.0002</td><td>-0.0017</td></tr><tr><td> $1 2 8 \times 3 8 4$ </td><td>+0.0005</td><td>-0.0008</td><td>-0.0106</td><td>-0.0002</td><td>-0.0073</td><td>-0.0008</td><td>-0.0069</td></tr></table>

Table 10 compares three orthogonalizers under the same persistent gain error, with the metric defined as the cosine between the defected output and the clean output of the same method; the entries give the range over the three matrices. Uniform output scaling leaves this cosine unchanged. Separate read gains can change the flow’s equilibrium amplitude as well as its timescale; the reported cosine measures directional agreement.

Table 10: Cosine between defected and clean output under persistent per-pass gain error
<table><tr><td>Method</td><td>5%</td><td>10%</td><td>20%</td></tr><tr><td>NS5</td><td>0.913-0.940</td><td>0.397-0.447</td><td>0.366–0.424</td></tr><tr><td>PolarExpress, 5 rounds</td><td>0.366-0.429</td><td>2 of 3 diverged; 0.389</td><td>3 of 3 diverged</td></tr><tr><td>Flow (probe form)</td><td>0.9986–0.9999</td><td>0.9945–0.9998</td><td>0.9763-0.9990</td></tr></table>

Energy model. We use an operation-count estimate, $E ( B ) = B \Sigma _ { r c } e .$ , with per-MAC energy e. This extrapolation from published silicon measurements is analogous to the component-based power estimate of Wang and Achour (2026). The workload unit is GPT-2 124M: 48 momentum matrices containing $\Sigma _ { r c } = 8 4 . 9 \times 1 0 ^ { 6 }$ cells. A fabricated ReRAM chip (Hung et al., 2021) provides a lower anchor of 10.3 fJ per multiply-accumulate (MAC)

for the binary-input probe pass and 128 fJ for the other two passes, averaging 88.9 fJ. A fabricated phase-change chip (Le Gallo et al., 2023) provides a higher anchor of 205 fJ per MAC. These published chip measurements include per-pass conversion circuitry and serve as cost anchors for the proposed loop. The digital model assumes $9 . 7 8 4 \times 1 0 ^ { 1 1 }$ MACs for NS5 on an H100, giving 1.38 J at datasheet peak (NVIDIA Corporation, 2023) or 3.00 J using the 46.2% utilization reported for large-scale training (Chowdhery et al., 2023) as a proxy for NS5 utilization.

Rank scaling and break-even. Starting from zero with inactive rails, KT rank-1 writes give rank $( X ) \ \leq \ K T = B / 3$ under the counted write model. The measured width-128 budgets span 78–312 passes per unit rank. Extrapolating these budgets linearly to rank 768 gives 60k–240k passes; validating this projection requires width-768 probe training. Table 11 applies both the measured small-matrix budgets and the rank-scaled budgets to the same GPT-2 workload. Break-even lies at 239–517 passes per unit rank for the lower anchor and 104–224 for the higher anchor, depending on digital utilization. At 240k passes the projected range overlaps the digital estimate.

Latency and excluded costs. A simulated iteration takes 1.0 µs. Assuming all matrices are resident and the K probe channels operate in parallel at this period, the projected solve time for T = 417 is 0.42 ms, compared with the 2 ms digital reference used in the accounting. A system-level evaluation would test these residency and parallel-execution assumptions. The energy totals exclude loading M<sup>ˆ</sup> (estimated at 0.85 mJ per step using a measured DAC (Olieman et al., 2015)), reading X out once per step, and 1.7 GB of bufer trafic shared by both implementations.

Table 11: Estimated orthogonalization energy per step for the GPT-2 124M workload. The 10k–40k budgets are measured on width-128 matrices; 60k–240k are rank-scaled projections. Pitch and stress denote the lower and higher device energy anchors.
<table><tr><td>Budget per matrix</td><td>E (pitch)</td><td>E (stress)</td><td>Digital / pitch</td><td>Digital / stress</td></tr><tr><td>B = 10k (measured, width 128)</td><td>0.076 J</td><td>0.174 J</td><td>18.3-39.7×</td><td>8.0-17.2×</td></tr><tr><td>B = 20k (measured, width 128)</td><td>0.151 J</td><td>0.348 J</td><td>9.2-19.8×</td><td>4.0-8.6×</td></tr><tr><td>B = 40k (measured, width 128)</td><td>0.302 J</td><td>0.696 J</td><td>4.6-9.9×</td><td>2.0-4.3×</td></tr><tr><td>B = 60k (rank-scaled, r = 768)</td><td>0.453 J</td><td>1.044 J</td><td>3.1-6.6×</td><td>1.3-2.9×</td></tr><tr><td>B = 240k (rank-scaled, r = 768)</td><td>1.812 J</td><td>4.177 J</td><td>0.8-1.7×</td><td>0.3-0.7×</td></tr></table>

## Appendix E. Circuit-Level Validation

Simulation ladder. Three levels connect the mathematical flow to a circuit: the dense Euler iteration of (3), the probe form of (4) with a behavioral array model, and an ngspice transient of the array. Table 12 lists the agreement at each interface.

Array model. The momentum array is a diferential resistor network with 8-bit conductance codes and 0.4% mismatch per resistor; its currents sum into a 0 V virtual ground, and it is written digitally once per optimizer step. The state X is stored on 64 capacitors (one per entry of the 8 × 8 block). The passes $\boldsymbol { q } = \boldsymbol { X } ^ { T } \boldsymbol { p }$ and $r = X q$ read the same stored values, which is the reciprocity the transposed pass relies on. Each pass settles through a 40 ns RC time; the write gate is open for 0.60–0.95 µs of each 1.0 µs period, and the complete cold-start solve runs as one continuous transient of 600 periods (9.0 s of ngspice time clean,

Table 12: Agreement across the simulation ladder
<table><tr><td>Interface</td><td>Quantity</td><td>Value</td></tr><tr><td>Probe form vs. dense flow</td><td>code path with identity probes</td><td>bitwise equal</td></tr><tr><td>Quantized vs. continuous block</td><td>cosine of NS5 outputs</td><td>1.0000</td></tr><tr><td>Circuit vs. probe form</td><td>cosine of converged states</td><td>0.999983</td></tr><tr><td></td><td>relative Frobenius distance maximum element-wise distance</td><td>0.0059 0.0045</td></tr><tr><td>Circuit vs. NS5</td><td>screen value, clean</td><td>0.9627 (behavioral 0.9624)</td></tr><tr><td></td><td>screen value, with non-idealities</td><td>0.9606</td></tr><tr><td></td><td>dense flow at  $T = 4 0 0$  on the same block</td><td>0.9834</td></tr><tr><td>Closed loop (synthetic task)</td><td>steps to CE &lt; 0.01, NS5 / behavioral / circuit</td><td></td></tr><tr><td></td><td>final CE, all three</td><td>43 / 42 / 42 0.0000</td></tr></table>

12.5 s with non-idealities, single core). The test block is the $8 \times 8$ leading sub-block of a saved width-128 attention momentum matrix, with singular values from $3 . 6 5 \times 1 0 ^ { - 3 }$ to $1 . 2 \times 1 0 ^ { - 4 }$ . The element-wise rail at 1.2 stays inactive (max $| X | = 0 . 7 4 9$ clean, $0 . 5 9 2$ with non-idealities). Figure 1 (right) shows one X cell; Figure 5 shows the capacitor voltages against the probe-form state, the cosine values against NS5, and the convergence trajectory.

![](images/89c73f9b8fdcc32c2bc433041cbe12ec90117c41d68c18318deacaf9401527d1.jpg)

![](images/da0763e806e6f29d6585bcb45448e1c0f974f18678b0d3197cc6aa94f500bb1e.jpg)

![](images/59c6cc50fa71f255bab169cceabab86fbf897b35b5e0dceb07db212e1593a90b.jpg)  
Figure 5: Circuit transient on the $8 \times 8$ block. Left: the 64 capacitor voltages against the probe-form state (cosine 0.999983). Middle: screen values of the circuit output against NS5, clean and with non-idealities. Right: convergence over the 600-period transient.

Transistor-level read element. Each entry of X is read through two NMOS channels in the triode region with gates at $V _ { \mathrm { r e f } } \pm V _ { X } ;$ the channels are the only elements in the signal path. The devices use a level-1 NMOS model with discrete-transistor parameters $( V _ { T 0 } = 0 . 7$ V, $K _ { P } = 0 . 1$ m $\mathrm { A } / \mathrm { V } ^ { 2 } , \gamma = 0 . 3 7$ $W / L = 1 0 0 / 1 0 0 \ \mu \mathrm { m } )$ ; a mismatch arm adds 2% transconductance and 5 mV threshold mismatch. The device law $g _ { \mathrm { e f f } } ( V _ { X } ) = 2 K V _ { X }$ holds with a simulated slope of $2 0 0 . 0 0 ~ \mu \mathrm { S / V }$ against a model value of 200 and a deviation from linearity below 0.01% for $| V _ { X } | \leq 0 . 5 ~ \mathrm { V } ;$ the body-efect terms of the two channels cancel in pairs (Figure 6, left). The channel law $I = K ( V _ { A } - V _ { B } ) [ ( V _ { G } - V _ { T } ) - ( V _ { A } + V _ { B } ) / 2 ]$ is symmetric in its two terminals, so the device itself is reciprocal and the measured asymmetries are operating-point efects of the composite cell. The two-transistor cell reads exactly in the virtual-ground direction and shows a reverse asymmetry of $u / ( 2 V _ { X } )$ times a body-efect factor of 1.10 (measured 1.10–1.11); the sub-1% region is $u \lesssim V _ { X } / 5 5$ (3 mV at $V _ { X } = 0 . 2 ~ \mathrm { V } )$

The row-common part of the transpose residual is removed by the probe chopping, so the element-wise transpose error of 2.1% leaves the cosine of r at 0.9997. The four-transistor cell reads exactly in both directions with matched devices and shows 0.00% small-signal and 0.13% full-swing asymmetry under mismatch; the zero crossing is clean, with a mismatch ofset of 1.8 mV (0.0046 in X units). A full flow iteration on a $2 \times 2$ array with transistor reads gives cosines of 1.00000, 0.99972, and 0.99987 for $q , \ r ,$ and $\Delta X$ ; the write lands within $1 0 ^ { - 5 } \ X$ units, the read droop is 0.200 $\mu \mathrm { V }$ per 10 $\mu \mathrm { s } ,$ , the reset injection is 1.7 mV per capacitor, and the diferential residual after reset is 0.00061 (Figure 6, right). The model boundaries are the level-1 device model (no mobility degradation or short-channel efects, adequate for discrete devices of this size), the $2 \times 2$ scale of the transistor-level array, and a behavioral DAC abstraction for the write chain.

![](images/fcf34f27e005e0d2a56eb2ff8add1e0a7281e66d2e3f911a1b02c65a55bce463.jpg)

![](images/9caec67df3f348407e6df848cc4fb001d1fb05f5fecb2f52757de69ce9a7fbeb.jpg)

![](images/03e706576eb8fe059903c157865af21de2452cbabb25bc21d5eb21bac1553a4a.jpg)

![](images/8d52d20eb301457299e8e8eecb6ae0c7beeb9410bca03fd3c2a3bdb8fd122bc7.jpg)  
Figure 6: Left: device law $g _ { \mathrm { e f f } } = 2 K V _ { X }$ of the read element $( 2 0 0 . 0 0 ~ \mu \mathrm { { S } / \mathrm { { V } _ { : } } }$ , deviation below 0.01%). Right: reciprocity of the two- and four-transistor cells and the $2 \times 2$ array cosines for $q , r ,$ and $\Delta X$

Closed loop. A four-class synthetic task trained with batch size 8 for 300 steps compares three orthogonalizers in the loop: digital NS5, the probe form with the behavioral array, and the probe form with the ngspice circuit. The three reach $\mathrm { C E } < 0 . 0 1$ in 43, 42, and 42 steps and end at CE 0.0000 with 100% training accuracy; the in-loop cosine of the orthogonalizer output against NS5 averages 0.895 for the behavioral model and 0.891 for the circuit (Figure 3).