# Learning consistent molecular mechanics force fields from first principles

Berkay Günes <sup>1,2</sup>, Leif Seute <sup>1,3</sup>, Jigyasa Nigam <sup>4</sup>, Frauke Gräter <sup>1,2,3</sup>

<sup>1</sup> Max Planck Institute for Polymer Research Mainz, Germany <sup>2</sup> Heidelberg University, Germany

<sup>3</sup> Heidelberg Institute for Theoretical Studies, Germany

<sup>4</sup> Initiative for Computational Catalysis, Flatiron Institute, United States {guenesb, seutel, graeter}@mpip-mainz.mpg.de, jnigam@flatironinstitute.org

## Abstract

Classical force fields (FFs) remain the workhorse for large-scale simulations even as machine-learned interatomic potentials (MLIPs) approach ab initio accuracy. They decompose total configuration energies into simple effective interactions whose parameters are traditionally assigned based on atom or bond types, enabling efficient simulations but also limiting their ability to adapt across configurations. Recent machine learning approaches have improved the accuracy and transferability of bonded parameters in these FFs by inferring them as functions of local atomic environments, but still rely on empirical nonbonded parameters for practical simulations. In this work, we introduce a unified approach, grappa-fullFF, which learns both bonded and nonbonded parameters consistently and simultaneously from ab initio reference data. By incorporating physically inspired regularization via supervision of the electrostatic potential and an architecture that facilitates charge equilibration, our model recovers accurate electric response properties, achieves state-of-the-art accuracy on geometry optimization benchmarks, and reproduces the conformational sampling of both classical and existing machine-learned FFs, without relying on externally assigned nonbonded parameters.

## 1 Introduction

Simulating molecular systems containing millions of atoms on nanosecond to millisecond timescales underlies numerous applications ranging from drug discovery and biomedical engineering to materials science [1–5]. Such simulations require force fields that are both computationally efficient and sufficiently accurate to describe the potential energy and forces throughout the dynamics. Recent advances in atomistic modeling and deep learning have enabled MLIPs that predict energies and forces directly from three-dimensional molecular configurations with near ab initio accuracy [6–11]. Although MLIPs are orders of magnitude faster than quantum mechanical (QM) calculations, their computational cost remains prohibitive for such large-scale systems and long timescales [12, 13], and classical molecular mechanics (MM) continues to be widely used to drive these simulations due to its exceptional efficiency [14–17]. In MM, the Born-Oppenheimer potential energy of an atomic configuration x is approximated as a sum of physical contributions,

$$
{ \cal E } _ { \mathrm { M M } } ( { \bf x } , \Phi _ { \mathrm { F F } } ) = { \cal E } _ { \mathrm { b o n d e d } } ( { \bf x } , \Phi _ { \mathrm { b o n d e d } } ) + { \cal E } _ { \mathrm { n o n b o n d e d } } ( { \bf x } , \Phi _ { \mathrm { n o n b o n d e d } } )\tag{1}
$$

where $\Phi _ { \mathrm { F F } } = \{ \Phi _ { \mathrm { b o n d e d } } , \Phi _ { \mathrm { n o n b o n d e d } } \}$ represents the set of force field parameters. $\begin{array} { r l } { E _ { \mathrm { b o n d e d } } ( \mathbf { x } , \Phi _ { \mathrm { b o n d e d } } ) = } \end{array}$ $U _ { \mathrm { b o n d s } } + U _ { \mathrm { a n g l e s } } + U _ { \mathrm { d i h e d r a l s } }$ constitute bonded interactions, while $E _ { \mathrm { n o n b o n d e d } } ( \mathbf { x } , \Phi _ { \mathrm { n o n b o n d e d } } ) = U _ { \mathrm { v d W } } +$ $U _ { \mathrm { C o u l o m b } }$ are the nonbonded interactions [18, 19] comprising van der Waals (vdW) (typically the 12-6 Lennard-Jones [20]) and long-range Coulomb interactions. Traditionally, the parameters of these effective potentials have been refined empirically and tabulated for different classes of molecules, which restricts their accuracy and transferability. Recently, graph-based machine-learning frameworks [21–26] have enabled hybrid ML/MM force fields that predict bonded parameters $( \Phi _ { \mathrm { b o n d e d } } )$ directly from reference energies and forces, or model environmental polarization [27]. However, these models still rely on external, empirical nonbonded parameters $( \Phi _ { \mathrm { n o n b o n d e d } } )$ or separate machinelearned and classical descriptions of the simulated system, limiting their ability to generalize across configurations. For instance, partial charges are typically derived from computationally expensive semi-empirical methods such as AM1-BCC [28] or by partitioning the electron density using methods such as MBIS [29]. Lennard-Jones (vdW) parameters, on the other hand, are assigned using rigid, curated typing rules inherited from legacy force fields including GAFF [30] and OpenFF [31].

![](images/40fba4b198c8f68182c5f30a0882972317d54dd433e88bcb23d848c5839d4cb6.jpg)  
Figure 1: Grappa architecture and training routine. The shared graph-attention encoder feeds the atom embeddings in the individual parameter prediction heads. All parameters are trained simultaneously on QM energies and forces. The nonbonded parameter heads are additionally regularized by the electrostatic potential (ESP) and the ratio of the effective and free atomic volume, v<sub>i</sub>.

In the following, we introduce grappa-fullFF, a unified end-to-end approach that learns both bonded and nonbonded MM parameters from ab initio reference data. We extend Grappa [23] (hereafter referred to as grappa-bonded), which has shown state-of-the-art performance on bonded parameters, to jointly predict all parameters. Instead of targeting each parameter independently, we train the model directly against QM energies and forces, allowing all components of the force field to be learned consistently. We regularize the implicitly learned MM parameters using physically motivated constraints on the electrostatic potential and the Lennard-Jones (LJ) parameters through supervision of the effective per-atom volume. We demonstrate that, even without explicit supervision of individual MM parameters, grappa-fullFF recovers the QM electrostatic potential with an accuracy comparable to expensive electron density partitioning schemes, without any electron density calculation at inference, and retains the performance of grappa-bonded on a standard geometry optimization benchmark. In an explicit-solvent molecular dynamics (MD) simulation of alanine-dipeptide, our model reproduces the Ramachandran free-energy surface with accuracy comparable to grappa-bonded relative to AMBER99SB-ILDN [32], and about twice as accurately as espaloma-0.3.2, even without having been trained on condensed phase systems.

## 2 Methodology

Training the model by constraining energies and forces does not uniquely determine how the potential is decomposed into individual components of the FF in Eq. (1). Various combinations of bonded and nonbonded interactions can yield similar total energies and forces, and similarly, different combinations of the electrostatic and vdW parameters can yield the same nonbonded contribution. This ambiguity, referred to as energy degeneracy [13, 33], can allow the model to achieve low errors on the target energies while implicitly learning parameters that are not physically meaningful and do not transfer reliably across molecular environments [12]. We address this ambiguity by incorporating additional physical information, discussed in Sec. 2.2, which directly constrains the nonbonded interactions. Before introducing these constraints, we first describe the model architecture used to predict the MM parameters.

## 2.1 Model Architecture

We extend the grappa-bonded framework [23], which uses a graph attention network to predict the bonded parameters (Fig. 1, shaded yellow), by introducing two new prediction heads for nonbonded interactions (shaded red), namely the antisymmetric bond-charge-flux head for partial charges and a volume-ratio head for LJ parameters. The parameterization of the bonded interactions is unchanged.

Antisymmetric Bond-Charge-Flux Head. To get the electrostatic contribution to the nonbonded energy, the model must learn to assign a partial charge to each atom. Each atomic charge state is initialized (t = 0) as $q _ { i } ^ { ( t = 0 ) } = Q _ { \mathrm { t o t } } / N$ and is iteratively refined through $t = 1 , \dots , T$ messagepassing steps. Instead of predicting atomic charges at each step, the model outputs charge flux between bonded atom pairs, $w _ { i j } ^ { ( t ) }$ , which denotes the charge transferred from atom j to atom i. The fluxes are restricted to be antisymmetric under the permutation of atom indices, $w _ { i j } ^ { ( t ) } = - w _ { j i } ^ { ( t ) }$ , so that the total charge $Q _ { \mathrm { t o t } }$ is conserved. Following the classical Bond-Charge Increment (BCI) [34], we compute the atomic charges using the flux predictions from all bonded neighbors of i (denoted as $\mathcal { N } ( i ) )$ as,

$$
q _ { i } ^ { ( t + 1 ) } = q _ { i } ^ { ( t ) } + \sum _ { j \in N ( i ) } \left( w _ { i j } ^ { ( t ) } - w _ { j i } ^ { ( t ) } \right) .\tag{2}
$$

The initial charge fluxes $w _ { i j } ^ { ( t = 0 ) }$ are computed using the latent atom embeddings $h _ { i } , h _ { j }$

$$
w _ { i j } ^ { ( 0 ) } = \frac { 1 } { 2 } \big [ \mathrm { M L P } ^ { ( 0 ) } ( h _ { i } , h _ { j } ) - \mathrm { M L P } ^ { ( 0 ) } ( h _ { j } , h _ { i } ) \big ]\tag{3}
$$

whereas, subsequent time steps additionally condition the fluxes on the current atomic charge state and its local change $\Delta q _ { i , t } = q _ { i , t } - q _ { i , t - 1 }$ . Denoting these inputs as $\pmb { \xi } _ { i j } ^ { ( t ) } = [ h _ { i } , h _ { j } , q _ { i } ^ { ( t ) } , q _ { j } ^ { ( t ) } , \Delta \bar { q } _ { i } ^ { ( t ) } , \Delta q _ { j } ^ { ( t ) } ]$ nonlinear activation as σ, and element-wise product as ⊙, the update for $t \geq 1$ can be written as,

$$
w _ { i j } ^ { ( t ) } = \sigma \big ( \mathrm { M L P } _ { \mathrm { g a t e } } ^ { ( t ) } ( h _ { i } + h _ { j } , h _ { i } \odot h _ { j } ) \cdot \frac { 1 } { 2 } \big [ \mathrm { M L P } ^ { ( t ) } ( \xi _ { i j } ^ { ( t ) } ) - \mathrm { M L P } ^ { ( t ) } ( \xi _ { j i } ^ { ( t ) } ) \big ] .\tag{4}
$$

The anti-symmetry of the charge fluxes enforces $\begin{array} { r } { \sum _ { i } \Delta q _ { i } ^ { ( t ) } = 0 } \end{array}$ at every refinement step and maintains total charge conservation by construction, eliminating the need for post-hoc normalization or alternative charge equilibration [35, 36].

Volume-ratio LJ Head. Similarly, instead of directly predicting the LJ parameters $( \sigma , \epsilon )$ , a separate MLP predicts per-atom volume ratios $v _ { i } = v _ { i } ^ { \mathrm { e f f } } / \bar { v } _ { i } ^ { \mathrm { f r e e } }$ from the latent atomic representation $h _ { i }$ This formulation couples $\sigma$ and ϵ through a physically motivated relationship, constraining their predicted values to be more meaningful. Following the extension [37] of the TS-vdW formalism [38], which uses $v _ { i }$ to determine effective-atom polarizabilities $\alpha _ { \mathrm { e f f } } = v _ { i } \alpha _ { \mathrm { f r e e } }$ and dispersion coefficients $C _ { 6 } ^ { \mathrm { e f f } } = v _ { i } ^ { 2 } C _ { 6 } ^ { \mathrm { f r e e } }$ , we map LJ parameters to $v _ { i }$ through $\sigma \propto \alpha _ { \mathrm { e f f } } ^ { 1 / 7 }$ and $\epsilon \propto C _ { 6 } ^ { \mathrm { e f f } } / \sigma ^ { 6 }$ . For physical validity, we restrict the predicted volume ratio to be strictly positive using a scaled Exponential Linear Unit (ELU) activation on the network output.

## 2.2 Model training through differentiable MM and physics-informed regularization

To train the model on energies and forces, we pass the predicted bonded parameters, charges, and LJ parameters to a fully differentiable implementation of the energy functional (Eq. (1)), thus allowing loss gradients to propagate through the parameter prediction and graph embeddings. The corresponding atomic forces are obtained as $\mathbf { F } = - \nabla _ { \mathbf { x } } E _ { \mathbf { M M } }$ . Our differentiable functional explicitly implements classical 1-2 and 1-3 exclusion rules, as well as 1-4 scaling factors, to mimic the deployment environment of standard MD engines (e.g., GROMACS[39], OpenMM [40]).

As previously described, optimizing only the QM energy and force errors leaves the energy decomposition underconstrained. We therefore introduce additional physical constraints to the training objective and define the loss function as,

$$
\mathcal { L } = \lambda _ { E } \mathcal { L } _ { E } + \lambda _ { \mathbf { F } } \mathcal { L } _ { \mathbf { F } } + \lambda _ { \mathrm { E S P } } \mathcal { L } _ { \mathrm { E S P } } + \lambda _ { v } \mathcal { L } _ { v } ,\tag{5}
$$

where $\mathcal { L } _ { E }$ and $\mathcal { L } _ { \mathbf { F } }$ are the mean squared errors (MSEs) of the total MM energy and atomic forces relative to the QM references. $\mathcal { L } _ { \mathrm { E S P } }$ and $\mathcal { L } _ { v }$ regularize the predicted charges and LJ parameters, described below, and λs determine the relative weight of each term. Instead of optimizing the partial charges against their target values in eq.(5), which would depend on the choice of partitioning scheme, we constrain them by matching their electrostatic potential (ESP) to the QM reference. We compare this choice against direct charge supervision in Appendix B. We evaluate the ESP at grid points χ (which depend on x implicitly) surrounding the molecule and define

$$
\mathcal { L } _ { \mathrm { E S P } } = \frac { 1 } { N _ { \chi } } \sum _ { \mathrm { s h e l l } } \sum _ { j \in \mathrm { s h e l l } } \left( \frac { \tilde { V } _ { \mathrm { E S P } } ( \{ \tilde { q } _ { i } \} , \chi _ { j } ) - V _ { \mathrm { E S P } } ( \chi _ { j } ) } { \mu _ { \mathrm { s h e l l } } } \right) ^ { 2 } ,\tag{6}
$$

where $\tilde { V } , V$ are the predicted and reference ESP and $\left\{ \tilde { q } _ { i } \right\}$ is the set of predicted atomic charges. Note that unlike $\tilde { V } .$ , the reference ESP does not rely on charges and is computed directly from the electron density. Grid points $x _ { j }$ are obtained using concentric Fibonacci lattice shells at 1.4, 1.6, 1.8, and 2 times the vdW radii around each atom, following conventional ESP fitting schemes[41, 42], and grid points that fall inside the vdW radii of another atom are excluded. Contributions from each grid point are normalized by $\mu _ { \mathrm { s h e l l } }$ , defined as the root-mean-square of the reference ESP over the corresponding shell, to compensate for the rapid decay of the ESP with distance.

For the LJ parameters, we similarly constrain the predicted effective volume ratios $\tilde { v } _ { i }$ using reference ratios $v _ { i }$ obtained by partitioning the QM electron density, and define $\begin{array} { r } { \mathcal { L } _ { v } = \frac { 1 } { N _ { \mathrm { a t o m s } } } \sum _ { i \in \mathbf { x } } \left( \tilde { v } _ { i } - v _ { i } \right) ^ { 2 } } \end{array}$ We use the losses on energies and forces to train all model parameters but restrict $\mathcal { L } _ { \mathrm { E S P } }$ and $\mathcal { L } _ { v }$ to exclusively update the parameters of the charge flux and volume ratio prediction heads.

## 3 Results

In the following, we consider the DES-monomer, dipeptide, and PubChem subsets of SPICE [43] using the QM energy and forces reported in OMol25 at the ωB97M-V/def2-TZVPD level [44]. For direct comparison with Espaloma [21] and the grappa-bonded [23], we similarly filter high-force conformations, retaining 10,240 unique molecules and ∼441,000 conformers for which QM energy and force labels are available. Of these, electron densities are available for only 14,600 conformers in the OMol25 dataset. Therefore, while all conformers contribute to the energy and force losses $( \mathcal { L } _ { E }$ $\mathcal { L } _ { \mathbf { F } } )$ , L and $\mathcal { L } _ { v }$ are evaluated only for a subset of the dataset with electron densities. Reference ESP values are computed using Multiwfn [45, 46], while effective and free atomic volumes are obtained using MBIS partitioning [29] of the available electron densities and averaged across all conformers per molecule to a single value per atom.

Accuracy of the Learned Electrostatics We assess the quality of the learned charges by evaluating how well they reproduce the QM electric response properties. Table 1 compares the RMSE of the ESP and dipole moments across the test dataset partition. grappa-fullFF outperforms traditional empirical charge schemes such as AM1-BCC and stand-alone machine-learned charge models such as espaloma-charge [43]. Fig. 3 shows deviations in the ESP computed from the predicted charges by different charge schemes from the QM reference. Notably, the predicted charges from grappa-fullFF are almost as accurate as those obtained from MBIS partitioning, even though the model is never trained against reference atomic charges and only infers them from ESP supervision.

Table 1: Comparison of ESP $( 1 0 ^ { - 3 } e / \mathring { \mathbf { A } } )$ and dipole moment $( e \mathring \mathbf { A } )$ RMSE across charge models and datasets. Parenthetical values give the standard deviation on the last digits.
<table><tr><td></td><td colspan="2">PubChem</td><td colspan="2">Dipeptide</td><td colspan="2">DES Monomers</td></tr><tr><td>Charge Model</td><td>ESP</td><td>Dipole</td><td>ESP</td><td>Dipole</td><td>ESP</td><td>Dipole</td></tr><tr><td>MBIS1</td><td>7.157(54)</td><td>1.190(26)</td><td>8.47(37)</td><td>1.724(79)</td><td>6.22(25)</td><td>0.622(50)</td></tr><tr><td>AM1-BCC1</td><td>10.414(87)</td><td>1.156(25)</td><td>12.32(54)</td><td>1.88(10)</td><td>8.50(40)</td><td>0.597(47)</td></tr><tr><td>espaloma-charge</td><td>35.7(10)</td><td>2.084(40)</td><td>30.3(11)</td><td>2.174(71)</td><td>18.2(17)</td><td>0.758(66)</td></tr><tr><td>grappa-fullFF</td><td>8.90(58)</td><td>1.048(24)</td><td>12.7(13)</td><td>1.584(81)</td><td>6.12(26)</td><td>0.561(46)</td></tr></table>

Structural Accuracy in Geometry Optimization We next assess whether the learned force field reproduces QM equilibrium structures using a standard geometry optimization benchmark [47, 48]. Starting from unrelaxed conformations, we optimize each molecule and compare the resulting geometries and relative energies with the QM reference in Table 2. The distributions of geometry RMSE, TFD, and $| \Delta \Delta E |$ (ddE) in Fig. 4 show a clear separation between ML and classical force fields. espaloma-0.3.2, grappa-bonded, and grappa-fullFF consistently outperform GAFF-2.11 and OpenFF-2.1.0 across all three metrics. Among ML models, the distributions are nearly indistinguishable, with marginal differences in the median values. This shows that grappa-fullFF retains the geometric fidelity of its bonded-only counterpart, and that jointly learning nonbonded parameters does not degrade the bonded parameters due to energy degeneracy.

Table 2: Comparison of force field performance metrics: median of the Root Mean Square Error (RMSE med) in geometries between MM-optimized and QM-optimized conformers, Torsion Fingerprint Deviation (TFD), and the Mean Absolute Error (MAE) and RMSE of the conformer-averaged relative energy (ddE).
<table><tr><td>Force Field</td><td>RMSE med  $( \mathring \mathrm { A } )$ </td><td>TFD med</td><td>ddE MAE</td><td>ddE RMSE</td></tr><tr><td>gaff-2.11</td><td>0.4997</td><td>0.0476</td><td>2.341</td><td>3.892</td></tr><tr><td>openff-2.1.0</td><td>0.3688</td><td>0.0355</td><td>2.036</td><td>3.495</td></tr><tr><td>espaloma-0.3.2</td><td>0.2725</td><td>0.0241</td><td>1.639</td><td>3.064</td></tr><tr><td>grappa-bonded</td><td>0.2571</td><td>0.0232</td><td>1.608</td><td>2.773</td></tr><tr><td>grappa-fullFF</td><td>0.2661</td><td>0.0221</td><td>1.589</td><td>2.973</td></tr></table>

Conformational Thermodynamics in Explicit Solvent Finally, we evaluate grappa-fullFF in explicit-solvent molecular dynamics of capped alanine dipeptide (ACE-ALA-NME). We run 500 ns of unbiased MD with TIP3P water using AMBER99SB-ILDN, espaloma-0.3.2, grappa-bonded, and grappa-fullFF, and compare the resulting Ramachandran free-energy surfaces (FES). While grappa-bonded and espaloma-0.3.2 combine their learned bonded parameters with empirical or separately learned nonbonded parameters, grappa-fullFF predicts both bonded and nonbonded parameters within a single model. Full simulation details are given in Appendix D. All four force fields recover the characteristic extende $\boldsymbol { \mathrm { 1 / P P I I / \alpha _ { R } } }$ basin as the dominant low-free-energy region and $\alpha _ { L }$ as a distinct, higher-energy minimum. To quantify the differences, we bin the two backbone dihedral angles $( \phi , \psi )$ on a $5 ^ { \circ }$ grid and compute the symmetric Kullback–Leibler (KL) and Jensen– Shannon (JS) divergences relative to AMBER99SB-ILDN (Table 3). Both Grappa models achieve $\mathrm { J S } \lesssim 0 . 0 8$ nats, approximately twice as close to AMBER99SB-ILDN as espaloma-0.3.2, and agree with each other, as the difference in their JS is only $\Delta \mathrm { J S } = 0 . 0 1 3$ nats. Thus, our unified training preserves the conformational sampling of grappa-bonded, and eliminates the need to assign nonbonded parameters separately.

![](images/f650e0dccda9ee4dec0c3ffe95ab9461e7c4c0e19e5fe44d4ab0a3df65fa1bb1.jpg)  
Figure 2: Ramachandran FES of capped Alanine dipeptide from 500 ns explicit-solvent (TIP3P) MD.

## 4 Discussion and Conclusions

Our results demonstrate that grappa-fullFF is an end-to-end force field that learns bonded and nonbonded parameters directly from ab-initio QM energies and forces. One might expect this approach to introduce a degeneracy between interaction terms, but we show that this ambiguity can be resolved through physics-inspired regularization. By constraining charge-conserving bond fluxes through the electrostatic potential and LJ parameters through the TS-vdW volume ratios, our model learns parameters which are physically consistent and transferable. For instance, we obtain reduced LJ σ values for polar hydrogens required for hydrogen bonding, despite the larger values that would result from using TS-vdW volume ratios directly (Appendix C). These nonbonded parameters (charges, LJ parameters) emerge implicitly, without having been optimized against reference values, and offer considerably cheaper alternatives to methods such as MBIS that rely on expensive electron density calculations and partitioning. This parameter flexibility does not compromise the accuracy of bonded interactions. grappa-fullFF matches the state-of-the-art performance of grappa-bonded on geometry optimization and, despite being trained exclusively on isolated gas-phase monomers, reproduces conformational thermodynamics in explicit solvent. Extending training to explicit-solvent configurations will be a next step toward further refining model accuracy, along with additional refinement steps or alternative charge initialization schemes for larger, more delocalized molecules, paving the way for high-throughput, closed-loop simulations.

## Acknowledgments and Disclosure of Funding

This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No. 101002812)

Computations were performed on the HPC system Otter at the Max Planck Computing and Data Facility.

The Flatiron Institute is a division of the Simons Foundation.

## References

[1] David E Shaw, Paul Maragakis, Kresten Lindorff-Larsen, Stefano Piana, Ron O Dror, Michael P Eastwood, Joseph A Bank, John M Jumper, John K Salmon, Yibing Shan, et al. Atomic-level characterization of the structural dynamics of proteins. Science, 330(6002):341–346, 2010.

[2] David W Borhani and David E Shaw. The future of molecular dynamics simulations in drug discovery. Journal of computer-aided molecular design, 26(1):15–26, 2012.

[3] Yasushi Shibuta, Shinji Sakane, Eisuke Miyoshi, Shin Okita, Tomohiro Takaki, and Munekazu Ohno. Heterogeneity in homogeneous nucleation from billion-atom molecular dynamics simulation of solidification of pure metal: Molecular dynamics simulation of pure metal. Nature communications, 8(1):10, 2017.

[4] Tobias Morawietz and Nongnuch Artrith. Machine learning-accelerated quantum mechanicsbased atomistic simulations for industrial applications. Journal ofComputer-Aided Molecular Design, 35(4):557–586, 2021.

[5] Frank Noé. Machine learning for molecular dynamics on long timescales. In Machine learning meets quantum physics, pages 331–372. Springer, 2020.

[6] Albert P Bartók, Mike C Payne, Risi Kondor, and Gábor Csányi. Gaussian approximation potentials: The accuracy of quantum mechanics, without the electrons. Physical review letters, 104(13):136403, 2010.

[7] Justin S Smith, Olexandr Isayev, and Adrian E Roitberg. Ani-1: an extensible neural network potential with dft accuracy at force field computational cost. Chemical science, 8(4):3192–3203, 2017.

[8] Ilyes Batatia, Dávid Péter Kovács, Gregor N. C. Simm, Christoph Ortner, and Gábor Csányi. Mace: Higher order equivariant message passing neural networks for fast and accurate force fields, 2023. URL https://arxiv.org/abs/2206.07697.

[9] Simon Batzner, Albert Musaelian, Lixin Sun, Mario Geiger, Jonathan P Mailoa, Mordechai Kornbluth, Nicola Molinari, Tess E Smidt, and Boris Kozinsky. E (3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nature communications, 13(1): 2453, 2022.

[10] Arslan Mazitov, Filippo Bigi, Matthias Kellner, Paolo Pegolo, Davide Tisi, Guillaume Fraux, Sergey Pozdnyakov, Philip Loche, and Michele Ceriotti. Pet-mad as a lightweight universal interatomic potential for advanced materials modeling. Nature Communications, 16(1):10653, 2025.

[11] Yury Lysogorskiy, Anton Bochkarev, and Ralf Drautz. Graph atomic cluster expansion for foundational machine learning interatomic potentials. npj Computational Materials, 12(1):114, 2026.

[12] Oliver T. Unke, Stefan Chmiela, Huziel E. Sauceda, Michael Gastegger, Igor Poltavsky, Kristof T. Schütt, Alexandre Tkatchenko, and Klaus-Robert Müller. Machine learning force fields. Chemical Reviews, 121(16):10142–10186, 2021. doi: 10.1021/acs.chemrev.0c01111. PMID: 33705118.

[13] Yuanqing Wang, Kenichiro Takaba, Michael S Chen, Marcus Wieder, Yuzhi Xu, Tong Zhu, John ZH Zhang, Arnav Nagle, Kuang Yu, Xinyan Wang, et al. On the design space between molecular mechanics and machine learning force fields. Applied Physics Reviews, 12(2), 2025.

[14] João Morado, Paul N Mortenson, J Willem M Nissink, Jonathan W Essex, and Chris-Kriton Skylaris. Does a machine-learned potential perform better than an optimally tuned traditional force field? a case study on fluorohydrins. Journal ofChemical Information and Modeling, 63 (9):2810, 2023.

[15] M. J. Harvey, G. Giupponi, and G. De Fabritiis. Acemd: Accelerating biomolecular dynamics in the microsecond time scale. Journal ofChemical Theory and Computation, 5(6):1632–1639, 05 2009. ISSN 1549-9618. doi: 10.1021/ct9000685. URL https://doi.org/10.1021/ ct9000685.

[16] Peter Eastman, Jason Swails, John D. Chodera, Robert T. McGibbon, Yutong Zhao, Kyle A. Beauchamp, Lee-Ping Wang, Andrew C. Simmonett, Matthew P. Harrigan, Chaya D. Stern, Rafal P. Wiewiora, Bernard R. Brooks, and Vijay S. Pande. Openmm 7: Rapid development of high performance algorithms for molecular dynamics. PLOS Computational Biology, 13 (7):1–17, 07 2017. doi: 10.1371/journal.pcbi.1005659. URL https://doi.org/10.1371/ journal.pcbi.1005659.

[17] Romelia Salomon-Ferrer, Andreas W. Götz, Duncan Poole, Scott Le Grand, and Ross C. Walker. Routine microsecond molecular dynamics simulations with amber on gpus. 2. explicit solvent particle mesh ewald. Journal of Chemical Theory and Computation, 9(9):3878–3888, 08 2013. ISSN 1549-9618. doi: 10.1021/ct400314y. URL https://doi.org/10.1021/ct400314y.

[18] Pnina Dauber-Osguthorpe and Arnie Hagler. Biomolecular force fields: where have we been, where are we now, where do we need to go and how do we get there? Journal ofComputer-Aided Molecular Design, 33, 02 2019. doi: 10.1007/s10822-018-0111-4.

[19] Arnold T. Hagler. Force field development phase II: relaxation of physics-based criteria... or inclusion of more rigorous physics into the representation of molecular energetics. J. Comput. Aided Mol. Des., 33(2):205–264, 2019. doi: 10.1007/S10822-018-0134-X. URL https://doi.org/10.1007/s10822-018-0134-x.

[20] J. E. Jones. On the determination of molecular fields. ii. from the equation of state of a gas. Proceedings ofthe Royal Society ofLondon. Series A, Containing Papers ofa Mathematical and Physical Character, 106(738):463–477, 1924. ISSN 09501207. URL http://www.jstor. org/stable/94265.

[21] Yuanqing Wang, Josh Fass, Benjamin Kaminow, John E. Herr, Dominic Rufa, Ivy Zhang, Iván Pulido, Mike Henry, Hannah E. Bruce Macdonald, Kenichiro Takaba, and John D. Chodera. End-to-end differentiable construction of molecular mechanics force fields. Chemical Science, 13(41):12016–12033, 2022. ISSN 2041-6539. doi: 10.1039/d2sc02739a. URL http://dx. doi.org/10.1039/D2SC02739A.

[22] Kenichiro Takaba, Anika J. Friedman, Chapin E. Cavender, Pavan Kumar Behara, Iván Pulido, Michael M. Henry, Hugo MacDermott-Opeskin, Christopher R. Iacovella, Arnav M. Nagle, Alexander Matthew Payne, Michael R. Shirts, David L. Mobley, John D. Chodera, and Yuanqing

Wang. Machine-learned molecular mechanics force fields from large-scale quantum chemical data. Chemical Science, 15(32):12861–12878, 08 2024. ISSN 2041-6520. doi: 10.1039/ d4sc00690a. URL https://doi.org/10.1039/d4sc00690a.

[23] Leif Seute, Eric Hartmann, Jan Stühmer, and Frauke Gräter. Grappa – a machine learned molecular mechanics force field. Chemical Science, 16(6):2907–2930, 02 2025. ISSN 2041- 6520. doi: 10.1039/d4sc05465b. URL https://doi.org/10.1039/d4sc05465b.

[24] Gong Chen, Théo Jaffrelot Inizan, Thomas Plé, Louis Lagardère, Jean-Philip Piquemal, and Yvon Maday. Advancing force fields parameterization: A directed graph attention networks approach. Journal ofChemical Theory and Computation, 20(13):5558–5569, 06 2024. ISSN 1549-9618. doi: 10.1021/acs.jctc.3c01421. URL https://doi.org/10.1021/acs.jctc. 3c01421.

[25] Tianze Zheng, Ailun Wang, Xu Han, Yu Xia, Xingyuan Xu, Jiawei Zhan, Yu Liu, Yang Chen, Zhi Wang, Xiaojie Wu, Sheng Gong, and Wen Yan. Data-driven parametrization of molecular mechanics force fields for expansive chemical space coverage. Chem. Sci., pages –, 2025. doi: 10.1039/D4SC06640E. URL http://dx.doi.org/10.1039/D4SC06640E.

[26] Tianze Zheng, Xingyuan Xu, Zhi Wang, Zhenze Yang, Yuanheng Wang, Xu Han, Lei Chen, Zhenliang Mu, Ziqing Zhang, Siyuan Liu, Sheng Gong, Kuang Yu, and Wen Yan. Bridging quantum mechanics to liquid properties via a universal organic force field. Nature Communications, 17(1), May 2026. ISSN 2041-1723. doi: 10.1038/s41467-026-73566-3. URL http://dx.doi.org/10.1038/s41467-026-73566-3.

[27] Ge Song and Weitao Yang. Nepoip/mm: toward accurate biomolecular simulation with a machine learning/molecular mechanics model incorporating polarization effects. Journal of Chemical Theory and Computation, 21(11):5588–5598, 2025.

[28] Araz Jakalian, David B. Jack, and Christopher I. Bayly. Fast, efficient generation of high-quality atomic charges. am1-bcc model: Ii. parameterization and validation. Journal ofComputational Chemistry, 23(16):1623–1641, 2002. doi: https://doi.org/10.1002/jcc.10128. URL https: //onlinelibrary.wiley.com/doi/abs/10.1002/jcc.10128.

[29] Toon Verstraelen, Steven Vandenbrande, Farnaz Heidar-Zadeh, Louis Vanduyfhuys, Veronique Van Speybroeck, Michel Waroquier, and Paul W. Ayers. Minimal basis iterative stockholder: Atoms in molecules for force-field development. Journal ofChemical Theory and Computation, 12(8):3894–3912, 07 2016. ISSN 1549-9618. doi: 10.1021/acs.jctc.6b00456. URL https: //doi.org/10.1021/acs.jctc.6b00456.

[30] Junmei Wang, Romain M. Wolf, James W. Caldwell, Peter A. Kollman, and David A. Case. Development and testing of a general amber force field. Journal of Computational Chemistry, 25 (9):1157–1174, 2004. doi: https://doi.org/10.1002/jcc.20035. URL https://onlinelibrary. wiley.com/doi/abs/10.1002/jcc.20035.

[31] Lily Wang, Pavan Kumar Behara, Matthew W. Thompson, Trevor Gokey, Yuanqing Wang, Jeffrey R. Wagner, Daniel J. Cole, Michael K. Gilson, Michael R. Shirts, and David L. Mobley. The open force field initiative: Open software and open science for molecular modeling. The Journal of Physical Chemistry B, 128(29):7043–7067, 07 2024. ISSN 1520-6106. doi: 10.1021/acs.jpcb.4c01558. URL https://doi.org/10.1021/acs.jpcb.4c01558.

[32] David A. Case, David S. Cerutti, Vinícius Wilian D. Cruzeiro, Thomas A. Darden, Robert E. Duke, Mahdieh Ghazimirsaeed, George M. Giamba¸su, Timothy J. Giese, Andreas W. Götz, Julie A. Harris, Koushik Kasavajhala, Tai-Sung Lee, Zhen Li, Charles Lin, Jian Liu, Yinglong Miao, Romelia Salomon-Ferrrer, Jana Shen, Ryan Snyder, Jason Swails, Ross C. Walker, Jinan Wang, Xiongwu Wu, Jinzhe Zeng, Thomas E. Cheatham III, Daniel R. Roe, Adrian Roitberg, Carlos Simmerling, Darrin M. York, Maria C. Nagan, and Jr. Merz, Kenneth M. Recent developments in amber biomolecular simulations. Journal ofChemical Information and Modeling, 65(15):7835–7843, 07 2025. ISSN 1549-9596. doi: 10.1021/acs.jcim.5c01063. URL https://doi.org/10.1021/acs.jcim.5c01063.

[33] Serhii Tretiakov, AkshatKumar Nigam, and Robert Pollice. Studying noncovalent interactions in molecular systems with machine learning. Chemical Reviews, 125(12):5776–5829, 06 2025. ISSN 0009-2665. doi: 10.1021/acs.chemrev.4c00893. URL https://doi.org/10.1021/ acs.chemrev.4c00893.

[34] Thomas A. Halgren. Merck molecular force field. i. basis, form, scope, parameterization, and performance of mmff94. Journal of Computational Chemistry, 17(5- 6):490–519, 1996. doi: 10.1002/(SICI)1096-987X(199604)17:5/6<490::AID-JCC1>3. 0.CO;2-P. URL https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI% 291096-987X%28199604%2917%3A5/6%3C490%3A%3AAID-JCC1%3E3.0.CO%3B2-P.

[35] Anthony K. Rappe and William A. III Goddard. Charge equilibration for molecular dynamics simulations. The Journal of Physical Chemistry, 95(8):3358–3363, 1991. doi: 10.1021/ j100161a070.

[36] Michael K. Gilson, Hillary S. R. Gilson, and Michael J. Potter. Fast assignment of accurate partial atomic charges: An electronegativity equalization method that accounts for alternate resonance forms. Journal ofChemical Information and Computer Sciences, 43(6):1982–1997, 2003. doi: 10.1021/ci034148o. PMID: 14632449.

[37] Dmitry V. Fedorov, Mainak Sadhukhan, Martin Stöhr, and Alexandre Tkatchenko. Quantummechanical relation between atomic dipole polarizability and the van der waals radius. Phys. Rev. Lett., 121:183401, Nov 2018. doi: 10.1103/PhysRevLett.121.183401. URL https: //link.aps.org/doi/10.1103/PhysRevLett.121.183401.

[38] Alexandre Tkatchenko and Matthias Scheffler. Accurate molecular van der waals interactions from ground-state electron density and free-atom reference data. Phys. Rev. Lett., 102:073005, Feb 2009. doi: 10.1103/PhysRevLett.102.073005. URL https://link.aps.org/doi/10. 1103/PhysRevLett.102.073005.

[39] Mark James Abraham, Teemu Murtola, Roland Schulz, Szilárd Páll, Jeremy C. Smith, Berk Hess, and Erik Lindahl. Gromacs: High performance molecular simulations through multi-level parallelism from laptops to supercomputers. SoftwareX, 1-2:19–25, 2015. ISSN 2352-7110. doi: https://doi.org/10.1016/j.softx.2015.06.001. URL https://www.sciencedirect.com/ science/article/pii/S2352711015000059.

[40] Peter Eastman, Raimondas Galvelis, Raúl P. Peláez, Charlles R. A. Abreu, Stephen E. Farr, Emilio Gallicchio, Anton Gorenko, Michael M. Henry, Frank Hu, Jing Huang, Andreas Krämer, Julien Michel, Joshua A. Mitchell, Vijay S. Pande, João PGLM Rodrigues, Jaime Rodriguez-Guerra, Andrew C. Simmonett, Sukrit Singh, Jason Swails, Philip Turner, Yuanqing Wang, Ivy Zhang, John D. Chodera, Gianni De Fabritiis, and Thomas E. Markland. Openmm 8: Molecular dynamics simulation with machine learning potentials. The Journal of Physical Chemistry B, 128(1):109–116, 12 2023. ISSN 1520-6106. doi: 10.1021/acs.jpcb.3c06662. URL https://doi.org/10.1021/acs.jpcb.3c06662.

[41] Brent H. Besler, Kenneth M. Merz Jr., and Peter A. Kollman. Atomic charges derived from semiempirical methods. Journal of Computational Chemistry, 11(4):431–439, 1990. doi: https://doi.org/10.1002/jcc.540110404. URL https://onlinelibrary.wiley.com/doi/ abs/10.1002/jcc.540110404.

[42] Christopher I. Bayly, Piotr Cieplak, Wendy Cornell, and Peter A. Kollman. A well-behaved electrostatic potential based method using charge restraints for deriving atomic charges: the resp model. The Journal of Physical Chemistry, 97(40):10269–10280, 1993. doi: 10.1021/ j100142a004. URL https://doi.org/10.1021/j100142a004.

[43] Peter Eastman, Pavan Kumar Behara, David L. Dotson, Raimondas Galvelis, John E. Herr, Josh T. Horton, Yuezhi Mao, John D. Chodera, Benjamin P. Pritchard, Yuanqing Wang, Gianni De Fabritiis, and Thomas E. Markland. Spice, a dataset of drug-like molecules and peptides for training machine learning potentials, 2022. URL https://arxiv.org/abs/2209.10702.

[44] Daniel S. Levine, Muhammed Shuaibi, Evan Walter Clark Spotte-Smith, Michael G. Taylor, Muhammad R. Hasyim, Kyle Michel, Ilyes Batatia, Gábor Csányi, Misko Dzamba, Peter

Eastman, Nathan C. Frey, Xiang Fu, Vahe Gharakhanyan, Aditi S. Krishnapriyan, Joshua A. Rackers, Sanjeev Raja, Ammar Rizvi, Andrew S. Rosen, Zachary Ulissi, Santiago Vargas, C. Lawrence Zitnick, Samuel M. Blau, and Brandon M. Wood. The open molecules 2025 (omol25) dataset, evaluations, and models, 2026. URL https://arxiv.org/abs/2505. 08762.

[45] Tian Lu and Feiwu Chen. Multiwfn: A multifunctional wavefunction analyzer. Journal of Computational Chemistry, 33(5):580–592, 2012. doi: https://doi.org/10.1002/jcc.22885. URL https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.22885.

[46] Tian Lu. A comprehensive electron wavefunction analysis toolbox for chemists, multiwfn. The Journal ofChemical Physics, 161(8):082503, 08 2024. ISSN 0021-9606. doi: 10.1063/5. 0216272. URL https://doi.org/10.1063/5.0216272.

[47] Victoria Lim, David Hahn, Gary Tresadern, Christopher Bayly, and David Mobley. Benchmark assessment of molecular geometries and energies from small molecule force fields. F1000Research, 9:1390, 12 2020. doi: 10.12688/f1000research.27141.1.

[48] Kenichiro Takaba, Ivàn Pulido, Pavan Kumar Behara, Chapin E. Cavender, Anika J. Friedman, Michael M. Henry, Hugo MacDermott Opeskin, Christopher R. Iacovella, Arnav M. Nagle, Alexander Matthew Payne, Michael R. Shirts, David L. Mobley, John D. Chodera, and Yuanqing Wang. Machine-learned molecular mechanics force field for the simulation of protein-ligand systems and beyond, 2023.

## A Extended Benchmarking and Visualizations

## A.1 Qualitative Analysis of Electrostatic Potential Surfaces

Although Table 1 summarizes the ESP errors, visually inspecting the ESP projected onto the molecular van der Waals surface provides a more direct intuition for how these models will behave in intermolecular interactions. The 1.4× vdW surface is particularly relevant for nonbonded interactions, as it approximates the intermolecular contact distances. Figure 3 illustrates the ESP surface of a representative polar molecule. grappa-fullFF closely reproduces both the spatial structure and magnitude of the QM reference, comparable to the more computationally expensive MBIS charges. In contrast, empirical or learned charges in AM1-BCC and espaloma-0.3.2 show larger deviations in several regions of the molecular surface, often over- or under-polarizing with respect to QM. These visualizations confirm that ESP supervision constrains the predicted partial charges to reproduce the electrostatics from the underlying ab initio electron density.

![](images/3a8b47ccb94d78c32fac258c756f37627aef7e1a55575dc364ce275abf252096.jpg)  
Figure 3: ESP mapped onto the 1.4× vdW surface of a representative molecule. grappa-fullFF closely recovers the ESP surface derived from DFT with high accuracy, matching the computationally more expensive partial charges derived from the MBIS density partitioning. Standard empirical methods such as AM1-BCC and machine-learned charge models (espaloma-0.3.2) show noticeably higher deviations from the QM reference.

## A.2 Cumulative Error Distributions for Geometry Optimization

Table 2 in the main text reports the median and mean errors for the OpenFF Industry Benchmark. We complement these aggregate metrics with the full cumulative distribution functions (CDFs) for the Root Mean Square Error (RMSE), Torsion Fingerprint Deviation (TFD), and relative conformer energy errors (|ddE|) in Figure 4, which provide a more detailed comparison. The distributions for grappa-fullFF and the bonded-only baseline grappa-bonded are nearly indistinguishable. This indicates that jointly learning the nonbonded parameters preserves the accuracy of hybrid ML/MM models that only predict bonded parameters. Thus, physical regularization constrains the additional learnable nonbonded degrees of freedom sufficiently so that they do not substantially alter the learned bonded interactions despite the underlying energy degeneracy.

![](images/c23d1b8978912d76cc92fc8d2f9038f7b9a6558827999724ac6660978ce75176.jpg)  
Figure 4: Cumulative distribution functions (CDFs) for the geometry Root Mean Square Error (RMSE), Torsion Fingerprint Deviation (TFD), and relative conformer energies (|ddE|) evaluated on the OpenFF Industry Benchmark Season 1 v1.1. The machine-learned force fields (espaloma-0.3.2, grappa-bonded, grappa-fullFF) consistently outperform classical force fields (GAFF-2.11, OpenFF-2.1.0) on geometric fidelity. grappa-fullFF matches the structural accuracy of its bondedonly predecessor, confirming that the joint prediction of nonbonded parameters does not degrade bonded performance.

## A.3 Quantitative Divergence of Ramachandran Free-Energy Surfaces

Although the Ramachandran FESs in Figure 2 qualitatively show that the expected metastable states (α<sub>R</sub>, α<sub>L</sub>, β/PPII) are sampled, we provide quantitative metrics to evaluate the underlying thermodynamic distributions. To do so, we compute the 2D probability density distributions $P ( \phi$ , ψ) from 500 ns MD trajectories using a $5 ^ { \circ } \times 5 ^ { \circ }$ binning grid for each FF, and evaluate the symmetric KL and JS divergences relative to those obtained from AMBER99SB-ILDN.

As shown in Table 3, both grappa-bonded and grappa-fullFF models closely reproduce the AM-BER distribution. grappa-fullFF achieves a JS divergence of just 0.0820 nats, approximately twice as close to the reference distribution as espaloma-0.3.2. The difference between grappa-bonded and grappa-fullFF is negligible $( \Delta \mathbf { J } \mathbf { S } \approx 0 . 0 1 3$ nats), which quantitatively confirms that replacing established empirical nonbonded parameters (AMBER’s TIP3P-compatible charges and LJ) with our jointly learned parameters does not change the complex conformational dynamics of the peptide in explicit solvent.

Table 3: Divergence of each force field’s Ramachandran free-energy surface from AMBER99SB-ILDN, computed from a $5 ^ { \circ } \times 5 ^ { \circ }$ histogram of $( \phi , \psi )$ over 500 ns of unbiased TIP3P MD. Smaller is closer to AMBER99SB-ILDN.
<table><tr><td>Force field</td><td>Symmetric KL [nats]</td><td>Jensen-Shannon [nats]</td></tr><tr><td>AMBER99SB-ILDN (reference)</td><td>0.000</td><td>0.0000</td></tr><tr><td>espaloma-0.3.2</td><td>1.752</td><td>0.1596</td></tr><tr><td>grappa-bonded</td><td>0.432</td><td>0.0691</td></tr><tr><td>grappa-fullFF</td><td>0.502</td><td>0.0820</td></tr></table>

## B Ablation of the physics-informed regularization

As discussed in Sec. 2.2, we use ESP to regularize the predicted charges rather than supervising them directly against reference charges. We compare this strategy against the alternative training objective in which predicted charges are supervised against MBIS reference charges through the MSE loss,

$$
\mathcal { L } _ { q } = \frac { 1 } { N _ { \mathrm { a t o m s } } } \sum _ { i } \left( \tilde { q } _ { i } - q _ { i } ^ { \mathrm { M B I S } } \right) ^ { 2 } ,\tag{7}
$$

where $q _ { i } ^ { \mathrm { M B I S } }$ denotes the reference MBIS charge. This charge loss is incorporated as an additional term in Eq. (5) with weight $\lambda _ { q } .$ Table 4 lists four representative combinations of the hyperparameter sweep across $( \lambda _ { \mathrm { E S P } } , \lambda _ { q } )$ , all other regularization weights are fixed at $\lambda _ { E } = 2 , \lambda _ { \mathbf { F } } = 1$ , and $\lambda _ { v } = 1 0 0$ The last row corresponds to the configuration used to train grappa-fullFF .

Without either charge regularization, $\lambda _ { \mathrm { E S P } } = \lambda _ { q } = 0 ,$ , the ESP RMSE increases by $3 5 - 5 6 \%$ across all three data partitions compared to this work, and the ratio of mean charge magnitudes, $\langle q _ { i } \rangle / \langle q _ { i } ^ { \mathrm { M B I S } } \rangle$ , is only 0.74. Supervising charges alone $( \lambda _ { q } = 1 0 0 0 , \lambda _ { \mathtt { E S P } } = 0 )$ recovers the reference charge magnitude $( \langle q _ { i } \rangle / \langle q _ { i } ^ { \mathrm { M B I S } } \rangle \sim 1 )$ but increases the ESP errors compared to our model. Adding ESP regularization alongside the MBIS charge loss further reduces the ESP error for PubChem and DES, but increases the error on the dipeptide set. Using ESP regularization alone gives the lowest ESP errors for all three dataset partitions, even though the charge magnitude ratio degrades to 0.78. These results indicate that even though ESP regularizations do not exactly recover MBIS charges, it can still provide a more accurate description of the electrostatics and electric response properties.

Table 4: Ablation of the two nonbonded regularization terms. $\lambda _ { q }$ weighs the loss against per-atom MBIS charges as defined in Eq. (7), λ weighs the ESP loss defined in Eq. (6). The energy, force, and volume-ratio loss weights are held fixed at $( \lambda _ { E } = 2 , \lambda _ { F } = 1 , \lambda _ { v } = 1 \dot { 0 } 0 ) . \ \langle q _ { i } \rangle / \langle q _ { i } ^ { \mathrm { M B \bar { I } \tilde { S } } } \rangle$ is the ratio of mean (absolute) charge magnitudes on 30 PubChem molecules. ddE MAE is from the geometry-optimization benchmark of Table 2.
<table><tr><td>Config</td><td> $\lambda _ { q }$ </td><td> $\lambda _ { \mathrm { E S P } }$ </td><td> ${ \mathrm { E S P } } _ { { \mathrm { P u b C h e m } } }$ </td><td> ${ \mathrm { E S P } } _ { \mathrm { D i p e p } } .$ </td><td> $\mathrm { E S P _ { D E S } }$ </td><td> $\frac { \langle q _ { i } \rangle } { \langle q _ { i } ^ { \mathrm { M B I S } } \rangle }$ </td><td>ddE MAE</td></tr><tr><td>No regularization</td><td>0</td><td>0</td><td>13.84</td><td>17.35</td><td>9.10</td><td>0.74</td><td></td></tr><tr><td> $\lambda _ { q } \neq \mathsf { \bar { 0 } } , \lambda _ { \mathrm { E S P } } = 0$ </td><td>1000</td><td>0</td><td>11.80</td><td>16.52</td><td>8.71</td><td>0.99</td><td>1.72</td></tr><tr><td> $\bar { \lambda _ { q } } , \bar { \lambda _ { \mathrm { E S P } } } \ne 0$ </td><td>1000</td><td>100</td><td>11.07</td><td>20.24</td><td>7.55</td><td>≈1.00</td><td></td></tr><tr><td> $\bar { \lambda _ { q } } = 0 , \lambda _ { \mathrm { E S P } } \neq 0$ </td><td>0</td><td>1000</td><td>8.86</td><td>12.84</td><td>6.13</td><td>0.78</td><td>1.58</td></tr></table>

ESP in units of $\overline { { 1 0 ^ { - 3 } e / \mathring { A } } }$

## C Implicit Learning of Polar Hydrogen Radii

By regularizing the LJ parameters toward TS-vdW volume ratios and simultaneously training on ab initio energies and forces, grappa-fullFF learns the environment-dependent LJ σ values required to describe hydrogen bonding. The TS-vdW volume ratios assign relatively large σ values to polar hydrogens, whereas classical force fields assign smaller values to avoid unphysical Pauli repulsion in these hydrogen-bonded environments. As shown in Fig. 5, grappa-fullFF learns this deviation from the TS-vdW reference directly from the energy and force gradients.

![](images/c729cb099e595793bb1f6dd0c1ba5f9370ffe544746dd2eb0e39d49473a3fd23.jpg)

![](images/6faa0be7b2e6eb7114cf9920ca5ddfc571cb2bbc34cab3e42add8b9971d67c5e.jpg)

![](images/d4ba64afc3ba244f6792ee7c09ee0d9c137802fe76abd25e59d12e5a9008a877.jpg)  
Figure 5: Comparison of predicted Lennard-Jones σ parameters against the TS-vdW reference values for the three OMol25 SPICE subsets. The TS-vdW references assign relatively uniform radii to all hydrogen atoms (σ $\approx 2 . 3 - 2 . 5 )$ , while grappa-fullFF learned a systematic downward spread $( \sigma \approx 1 . 1 - 2 . 5 )$ . This correction is learned purely by training the model on ab initio energy and force gradients, which allows grappa-fullFF to accurately model hydrogen bonds without unphysical Pauli repulsion.

## D Molecular dynamics protocol

For the MD in Section 3 all four force fields, AMBER99SB-ILDN, espaloma-0.3.2, grappa-bonded, and grappa-fullFF, simulate an identical system of a capped alanine dipeptide (ACE-ALA-NME) solvated in a cubic box of edge 2.4 nm with 418 TIP3P water molecules and one Na<sup>+</sup>/two Cl<sup>−</sup> ions (0.15 M ionic strength) for 1,278 total atoms. The solvated topology and starting coordinates are generated once with OpenMM’s Modeller.addSolvent and reused identically across all four force fields, so that any differences in the resulting Ramachandran FES (Fig. 2) can be attributed to the force field rather than the simulated system.

For the grappa-fullFF simulations, the solute (ACE-ALA-NME) was parameterized entirely by the NN, whereas the TIP3P water molecules were parameterized conventionally. This setup shows the stability of the learned nonbonded parameters at the solute-solvent interface. For espaloma-0.3.2, partial charges and bonded parameters are assigned to the initial solvated topology through a chargeassignment step, followed by equilibration and the simulation protocol described below.

Each system is relaxed, equilibrated for 100 ps in the NVT ensemble followed by 1 ns in NPT, and then propagated for 500 ns in NPT at a 2 fs timestep using OpenMM’s LangevinMiddleIntegrator (friction coefficient 1 $\cdot \mathrm { p s ^ { - 1 } } )$ at 300 K. The pressure is maintained at 1 atm using a Monte Carlo barostat. Long-range electrostatics are treated with particle-mesh Ewald (real-space cutoff 1.0 nm, default OpenMM Ewald error tolerance). Lennard-Jones interactions use the same 1.0 nm cutoff with an isotropic long-range dispersion correction. All bonds to hydrogen are constrained, and water is kept rigid using OpenMM’s default constraint solver. Frames from the resulting MD trajectory are saved every 1 ps, yielding 500,000 samples of (ϕ, ψ) per force field.