# Thermodynamics-Informed Learning Captures Hierarchical Hydrogen-Bond Networks in Deep Eutectic Solvents

Yanpeng Ye<sup>1</sup>, Chaoran Duan<sup>2</sup>, Mian Xu<sup>1,∗</sup>, S. Mani Sarathy<sup>1</sup>

<sup>1</sup>Clean Energy Research Platform, Physical Science and Engineering Division (PSE), King Abdullah University of Science and Technology, Thuwal 23955-6900, Saudi Arabia;

<sup>2</sup>Key Laboratory of the Three Gorges Reservoir Region’s Eco-environments, Ministry of Education, Institute of Environment and Ecology, Chongqing University, Chongqing 400045, China.

<sup>∗</sup>Corresponding author: mian.xu@kaust.edu.sa

## Abstract

Deep eutectic solvents (DESs) ofer a highly tunable platform for lignocellulose refining, yet their solvation behavior arises from interactions among hydrogen bond acceptors (HBAs), hydrogen bond donors (HBDs), and solutes within multicomponent hydrogen bond networks, rather than from the additive properties of individual constituents. Existing machine learning models represent DES components and operating conditions independently, overlooking the hierarchical and thermodynamic nature of multicomponent solvation and limiting their reliability for new DES formulations and sparse conditions. This work develops TiNN, a thermodynamics-informed neural network that captures hierarchical hydrogen bond networks, from the association between HBAs and HBDs to solute accommodation, and links them to solubility under varying conditions through a learned free energy pathway. Trained on 938 records, TiNN achieves an $R ^ { 2 }$ of 0.917 on the test set, with robust generalization to new HBA and HBD combinations and to sparse conditions. Ablation shows that solute-specific

stabilization makes the largest contribution to prediction accuracy, while the contribution of hydrogen bond compatibility depends on the dissolution mechanism: it is pronounced for carbohydrate and lignin solutes but negligible for gases, which dissolve through other interactions. In new measurements of glucose, xylan, and Kraft lignin in acidic, basic, and neutral DESs, TiNN reduces the geometric mean fold error of solubility from more than 23 with COSMO-RS to about 1.5. Without training on fractionation data, the same molecular solvation information also separates DESs with high and low removal of lignin and hemicellulose from corn stover, linking the solubility of isolated biomass components to the fractionation of real lignocellulose. These results identify hierarchical hydrogen bonding and solute-specific stabilization as key molecular determinants of DES solvation for biomass-derived solutes, and provide a chemically grounded route to select DESs for lignocellulose fractionation before experiments.

## Introduction

Lignocellulose biorefineries serve as a key platform for converting abundant and renewable bioresources into fuels, chemicals, and functional materials, thereby enabling a transition to a circular bioeconomy.<sup>1</sup> The organic components of lignocellulose biomass mainly consist of cellulose, hemicellulose, and lignin, which are chemically and structurally distinct biopolymers that are intrinsically interconnected to form a recalcitrant supramolecular network.<sup>2</sup> This pronounced heterogeneity in physicochemical properties and chemical reactivities substantially impedes the selective component valorization and often results in harsh processing conditions and side product formation.<sup>3</sup> Fractionating lignocellulose into relatively pure individual organic components represents a prevailing trend and essential prerequisite for component-oriented biorefineries, enabling eficient production of biofuels, biochemicals, and biomaterials.<sup>4–7</sup> Deep eutectic solvents (DESs) with negligible vapor pressure, favorable biocompatibility, recyclability,<sup>8</sup> and generally inexpensive synthetic materials have emerged as promising alternatives<sup>9;10</sup> to conventional concentrated acids, strong alkalis, and organic solvents on lignocellulose fractionation. The use of these harsh substance results in equipment corrosion, chemical waste generation, and solvent emissions.<sup>11–13</sup>

DESs can be simply synthesized by mechanically mixing a wide variety of hydrogen bond donors (HBDs) and hydrogen bond acceptors (HBAs) with adjustable molar ratios,<sup>14;15</sup> enabling highly tunable polarity, acidity, and hydrogen-bonding basicity.<sup>16–18</sup> This strong hydrogen bond network (HBN) formed between HBDs and HBAs can facilitate the deconstruction and selective dissolution of lignocellulosic components through a reaction-assisted solvation mechanism<sup>19</sup>: the HBN dynamically reorganizes to disrupt noncovalent interactions within the biomass matrix, while chemical reactions may also cleave lignin–carbohydrate linkages; DES components competitively replace native hydrogen bonds, van der Waals forces, and aromatic interactions; newly formed component-DES hydrogen bonds, together with ion-dipole interactions, dispersion forces, and �–� interactions, compensate for the loss of polymer-polymer cohesive interactions, thereby stabilizing the liberated molecules in the liquid phase.<sup>20</sup> Therefore, the multilevel hydrogen bond network constitutes a key molecular determinant of the lignocellulose fractionation performance of DESs.<sup>19;21</sup> However, the compositional tunability of DESs creates a highly complex and reconfigurable hydrogen bond network such that changes in combinations and stoichiometry of HBDs and HBAs (> 100 types) concurrently alter network polarity, acidity, density, and viscosity. This leads to nonlinear trade-ofs between DES-DES association and competitive DES-biomass solvation.<sup>22</sup> Further coupled with operating conditions and feedstock heterogeneity, this tunability makes lignocellulose fractionation performance dificult to infer directly from the bulk property of individual DES constituent. Consequently, the rational design and formulation of optimal DES for lignocellulose refining is a challenging multidimensional screening problem.<sup>23;24</sup>

Conventionally, the screening of DESs for lignocellulose refining has primarily relied on two approaches, i.e., experimental evaluation and quantum chemical solvation modeling. Although experimental evaluation remains the most reliable route for assessing actual fractionation performance, systematically exploring the enormous space of HBA-HBD-solute combinations and operating parameters is prohibitively time- and resource-intensive.<sup>25–27</sup> Quantum chemical solvation models, such as the conductor-like screening model for real solvents (COSMO-RS), can provide useful molecular information on solute-solvent afinity and preliminarily guide solvent screening.<sup>17;28;29</sup> However, their accuracy is often constrained by simplifying assumptions and highly sensitive to molecular conformations and may require extensive computational resources when applied to large and diverse candidate spaces.<sup>30;31</sup>

Data-driven methods have therefore attracted increasing attention for the rapid screening of DESs, as they can learn complex and nonlinear composition-structure-property relationships from available data and evaluate extensive chemical spaces at lower computational costs.<sup>32–35</sup> Machine learning models, ranging from descriptor-based regressors to graph neural networks, have recently been applied to predict DES properties, including viscosity, density, and melting point.<sup>36–38</sup> Nevertheless, most existing models describe a DES system as a simple collection of discrete constituents, typically by concatenating constituent-wise molecular descriptors or combining independently encoded HBAs and HBDs with fixed molar ratios.<sup>23;39;40</sup> Such coarse-grained representations are chemically incomplete since they fail to explicitly capture the composition-dependent and dynamically reconfigurable multilevel HBN that governs DES formation, solvent-solute interactions, and final lignocellulose fractionation performance. Moreover, these representations treat the HBA, HBD, and solute as static inputs and treat temperature and pressure as ordinary features,<sup>41–43</sup> overlooking both the sequential nature of DES fractionation (the HBA–HBD environment forms prior to solute accommodation) and the thermodynamic origin of the solubilization process.<sup>44–46</sup>

This study develops TiNN, a thermodynamics-informed neural network that represents DES fractionation as two coupled chemical steps: formation of the HBA–HBD hydrogen-bonded environment, followed by accommodation of the solute within this environment. These two steps are linked to a learned free-energy pathway, connecting the molecular representation with the temperature- and pressure-dependent solubility response. Trained on 938 experimental records, TiNN achieves higher prediction accuracy than the conventional COSMO-RS method and captures dissolution trends under varying temperatures, pressures, and HBD:HBA combinations. It also demonstrates compositional generalization across HBA–HBD pairings. The predictions are further validated by solubility measurements on DES-solute pairs spanning acidic, basic, and neutral DES environments. By integrating solvation thermodynamics with the multilevel hydrogen bond network, this chemically interpretable representation provides a readily deployable framework for the rational design of DES in lignocellulose refining.

## Results and Discussion

## Dataset Distribution

The compiled dataset spans a broad range of DES–solute systems and exhibits a strongly heterogeneous solubility distribution (Figure 1). The resulting log � values have a mean of 1.64 and a standard deviation of 0.80, with substantial variation across diferent solute families. The solutes were grouped into carbohydrates (cellulose- and hemicellulose-derived compounds), lignin-derived compounds, and gases, providing chemically distinct subsets for evaluating the ability of TiNN to capture diferent solvation regimes.

![](images/ef733f730641c031c9352fe2756387fba44009f16bc68cd3e42dc74d7fc5a4a6.jpg)

![](images/937709d27f3e2d199955f3dc5a3bf99e0628baf67ae37f5c0cfb854f91d5a85f.jpg)

![](images/ffc9279eeb860bbbe4d0a64f1b0f4b745e083d9a5976d14400f4c6ed83414d34.jpg)

(d)  
![](images/af8936b661c4458e992f3864b8e850bfecd6d86b9f2680f58c38a6e1308e5e58.jpg)  
Figure 1. Statistical overview of the DES solubility dataset. Distribution of log � values for (a) all DES–solute systems, (b) carbohydrates, (c) lignin-derived compounds, and (d) gases. The dashed red lines indicate the mean values.

## Thermodynamics-Informed Learning

TiNN represents the DES-solute complexes through a structure that accounts for intermolecular interactions and reflects the ordered nature of DES solvation (Figure 2a). In contrast to direct feature concatenation, which treats the HBA, HBD, and solute as independent inputs and combines them at a single level (Figure 2a, left), TiNN first forms a DES-level representation by combining the HBA and HBD embeddings with their molar ratio and the internal hydrogen-bond compatibility, and then couples this DES representation with the solute embedding through the DES-solute hydrogen-bond compatibility (Figure 2a, right). This hierarchical construction encodes the

sequence in which the HBA–HBD environment forms before the solute is accommodated.   
Implementation details of the encoders $\Phi _ { \mathrm { D E S } }$ and $\Phi _ { \mathrm { s o l v } }$ are given in the Methods section.

(a)  
![](images/9da192699de63c671b3256292d757b814c1b4536ea45a69b7ae396df0162ed14.jpg)

![](images/60566973b86641115e1aee5ad45d83829da70ad525e46ba29a3a3841dfe16845.jpg)

![](images/8e9196b7d7a99bd29aa180edffb0e6be38d5db12c7df434088425e5af6af10d1.jpg)

![](images/5f68df1187b5bc905965762528acb96747e840dae3024b218296b1b227bc1338.jpg)

(e)  
![](images/016a355d14a97118d21e863d05c9dae48dc12ad096a5543b4b39eee0fc66ebfb.jpg)  
Figure 2. Molecular representation and predictive performance of TiNN. (a) Representation strategies for DES–solute systems. Left: direct concatenation of independently encoded HBA, HBD, and solute embeddings. Right: TiNN first combines the HBA and HBD into a DES representation, incorporating composition and internal hydrogen-bond compatibility, and then integrates the solute through DES–solute hydrogen-bond compatibility. (b–d) Parity plots on the carbohydrate subset for (b) MLP with concatenated HBA, HBD, and solute descriptors, (c) TiNN with concatenated input, and (d) TiNN with the hierarchical three-molecule representation. (e) Training- and test-set predictions of the full TiNN model; the reported test-set metrics are calculated on held-out samples.

The hierarchical representation defines the HBA–HBD–solute complexes, but it does not constrain how solubility changes with temperature by itself. Accordingly, log � reflects a thermodynamic balance between solute stabilization and the cost of incorporation into the DES liquid phase. In contrast, a direct mapping from the system representation to log � treats temperature as an ordinary descriptor and leaves this response unconstrained.

TiNN organizes the prediction through a learned free-energy pathway. Two latent quantities, $\Delta H _ { \theta }$ and $\Delta S _ { \theta }$ , are inferred from the system representation. An enthalpy–entropy consistency loss encourages the learned free energy to follow Eq. (1).

$$
\Delta G _ { \theta } \approx \Delta H _ { \theta } ( \mathbf { z } _ { \mathrm { s y s t e m } } ) - T \Delta S _ { \theta } ( \mathbf { z } _ { \mathrm { s y s t e m } } ) .\tag{1}
$$

Here, $\Delta H _ { \theta }$ and $\Delta S _ { \theta }$ are not calibrated to experimental enthalpy and entropy. They provide a thermodynamic inductive bias that separates the learned response into a temperature-independent contribution and a temperature-scaled contribution. The temperature branch supplies the explicit condition input, while the learned free-energy terms constrain how this input afects log �. This provides a thermodynamic regularization of the learned temperature response.

A solute-specific free-energy term is further included to account for the diferent energetic cost of accommodating diferent solutes in the same DES environment. Real DES systems can also deviate from the idealized thermodynamic form because of ion association, hydrogen-bond rearrangement, and other non-ideal interactions. TiNN therefore writes the final prediction as

$$
\log S _ { \mathrm { p r e d } } = \underbrace { \log S _ { \mathrm { t h e r m o } } ( \mathbf { z } _ { \mathrm { s y s t e m } } , T ) } _ { \mathrm { t h e r m o d y n a m i c s - g u i d e d } } + \underbrace { \delta _ { \theta } ( \mathbf { z } _ { \mathrm { s y s t e m } } , T ) } _ { \mathrm { n o n - i d e a l ~ c o r r e c t i o n } } ,\tag{2}
$$

where log $S _ { \mathrm { t h e r m o } }$ describes the thermodynamics-guided response, and $\delta _ { \theta }$ captures non-ideal deviations. A thermodynamic consistency constraint is applied to keep the learned temperature response aligned with the free-energy form.

## Predictive Performance

## Overall Accuracy of TiNN

After defining the three-molecule representation and the thermodynamics-informed learning module, the capability of TiNN in predicting DES solubility across chemically diverse DES– solute complexes was investigated. Figure 2e shows the training- and test-set predictions of the full TiNN model. TiNN achieves a test-set $R ^ { 2 }$ of 0.917 and maintains close agreement with experimental values over a broad solubility range. A small number of high-error points are observed in the parity plot. This agreement covers cellulose- and hemicellulose-derived carbohydrates, lignin-derived compounds (such as vanillin and syringaldehyde), and gaseous solutes (such as $\mathrm { C O } _ { 2 } , { \mathrm { S O } } _ { 2 }$ , and $\mathrm { H } _ { 2 } \mathbf { S } )$ . The broad coverage indicates that TiNN does not only fit a limited group of chemically similar compounds, but can learn solubility trends across distinct solute families.

We then examined how the two main design choices in TiNN, the thermodynamics-informed learning module and the hierarchical three-molecule representation, contributed to this performance. Carbohydrate solutes were selected as a representative subset because they provide suficient sample size and cover a continuous solubility distribution. As shown in Figure 2b–d, the predictions progressively improve from the MLP baseline to the thermodynamics-informed model with concatenated input, and finally to the full TiNN with the hierarchical three-molecule representation. The improvement from the MLP baseline to the thermodynamics-informed model shows that the learned free-energy pathway helps constrain the solubility response. The further improvement after introducing the hierarchical representation shows that resolving DES formation and DES–solute accommodation provides information that cannot be captured by simple feature aggregation. These results provide an initial indication that DES solubility is better described when both multilevel molecular interaction structure and thermodynamic response are considered, rather than when the system is treated as an additive combination of separated molecular descriptors.

## Comparison with Alternative Models

To examine whether the two central design ideas of TiNN were the main contributors for its superior performance, they were compared with alternative models at three diferent levels. First, the 3D molecular encoder within the TiNN framework was changed to examine the efect of geometry encoding. Second, the EGNN-based representation was kept, but the TiNN head was replaced with conventional prediction heads. Third, the TiNN was compared with descriptor-based, graph-based, and literature baselines that do not explicitly represent the ordered HBA–HBD–solute solvation process. The main results are summarized in Table 1.

EGNN-TiNN provides the best overall performance, with the lowest RMSE and the highest $R ^ { 2 }$ and Spearman correlation. PaiNN-TiNN gives a similar MAE but is slightly weaker in RMSE and $R ^ { 2 }$ , whereas DimeNet++-TiNN performs substantially worse. This comparison indicates that the three-molecule TiNN framework benefits from a stable 3D molecular encoder. The encoder is therefore not only a technical choice, but afects how efectively the complicated HBA–HBD–solute environment is represented.

The second comparison isolates the role of the thermodynamics-informed prediction structure. EGNN-MLP and EGNN-XGBoost use the same EGNN-based representation as EGNN-TiNN, but both give lower overall performance. This result verifies that the molecular embedding alone is not suficient. Instead, the learned free-energy pathway provides an additional constraint that helps convert the HBA–HBD–solute representation into a condition-dependent solubility response.

Descriptor and graph baselines provide a further test of the representation itself. XGBoost and CatBoost achieve relatively low MAE and high Spearman correlation, indicating that descriptor models can capture part of the statistical relationship between molecular identity and solubility. However, their RMSE and $R ^ { 2 }$ are substantially weaker than those of EGNN-TiNN, showing that they are less efective in reproducing the full quantitative response. GIN and the concatenated SMILES baseline also underperform TiNN, consistent with the limitations of concatenation-based representations discussed above. These comparisons indicate that molecular interactions and thermodynamic structure are both essential for describing DES–solute solubility. Therefore, TiNN is able to address this challenge by combining an ordered HBA–HBD–solute representation with a thermodynamics-informed prediction pathway, allowing the model to treat DES solubility as a multicomponent solvation response rather than as a direct descriptor-to-property mapping.

Table 1. Comparison of TiNN with alternative models for DES–solute solubility prediction. The models test the efects of 3D molecular encoding, thermodynamics-informed prediction, and descriptor-based representations.
<table><tr><td>Group</td><td>Model</td><td>MAE RMSE</td><td> $R ^ { 2 }$ </td><td>Spearman</td></tr><tr><td rowspan="3">Hierarchical TiNN variants</td><td>EGNN-TiNN</td><td>0.126 0.242</td><td>0.917</td><td>0.956</td></tr><tr><td>PaiNN-TiNN</td><td>0.123 0.253</td><td>0.909</td><td>0.954</td></tr><tr><td>DimeNet++-TiNN</td><td>0.163 0.665</td><td>0.562</td><td>0.886</td></tr><tr><td>EGNN representation + conventional heads</td><td>EGNN-MLP EGNN-XGBoost</td><td>0.170 0.286 0.145 0.493</td><td>0.884 0.656</td><td>0.913 0.935</td></tr><tr><td rowspan="5">Descriptor and graph baselines</td><td>MLP</td><td></td><td></td><td></td></tr><tr><td>XGBoost</td><td>0.232 0.595</td><td>0.498 0.661</td><td>0.878 0.929</td></tr><tr><td>CatBoost</td><td>0.127 0.489</td><td>0.743</td><td></td></tr><tr><td>GIN</td><td>0.138 0.426</td><td></td><td>0.938</td></tr><tr><td>Concat-SMILES Ridge</td><td>0.301 0.604 0.259 0.485</td><td>0.484 0.667</td><td>0.808 0.848</td></tr></table>

## Ablation Study

To further identify which design can promote the performance of TiNN, the full model was compared with a series of simplified variants trained under the same setup. These variants tested two parts of TiNN. For the molecular representation, the 3D molecular encoder was replaced by 2D descriptors of comparable dimensionality. For the thermodynamics-informed learning module, variants were tested without the residual branch, the solute-specific free-energy term, the thermodynamic consistency constraint, or the temperature branch (Table 2).

The largest performance drop occurs when the solute-specific free-energy term is removed $( \Delta R ^ { 2 } = - 0 . 1 6 3$ ; Table 2), showing that solute-dependent stabilization cannot be recovered by the shared molecular representation alone. Removing the residual branch gives the second largest drop $( \Delta R ^ { 2 } = - 0 . 1 0 2 )$ , demonstrating that the conventional ideal-solubility limit is insuficient for real DES complexes and that non-ideal correction is necessary. Removing the temperature branch or the thermodynamic consistency constraint causes a smaller decrease $( \Delta R ^ { 2 } = - 0 . 0 3 1$ and −0.028), suggesting that TiNN learns temperature dependence through both the explicit temperature encoder and the thermodynamic loss. Replacing the 3D molecular representation with 2D descriptors also leads to consistent degradation $( \Delta R ^ { 2 } = - 0 . 0 2 2 )$ , in agreement with the encoder comparison in Table 1.

Overall, the ablation results show that both parts of TiNN are necessary: the 3D representation provides molecular geometry beyond 2D descriptors, while the thermodynamics-informed module separates solute-specific stabilization, thermodynamic consistency, and non-ideal deviations in the solubility response. These results further indicate that DES solubility is better modeled as a thermodynamically constrained HBA–HBD–solute response than as a direct descriptor-toproperty mapping.

## Robustness and Generalization

The condition-dependent response and compositional generalization of TiNN were further examined across representative DES–solute systems and held-out HBA–HBD pairings.

Table 2. Ablation of TiNN components on the test set. Each variant removes or replaces one component of the full model. $\Delta R ^ { 2 }$ is relative to the full TiNN.
<table><tr><td>Variant</td><td>MAE</td><td>RMSE</td><td> $R ^ { 2 }$ </td><td>Spearman</td><td> $\Delta R ^ { 2 }$ </td></tr><tr><td>Full TiNN</td><td>0.126</td><td>0.242</td><td>0.917</td><td>0.956</td><td></td></tr><tr><td>w/o solute-specific free-energy term</td><td>0.183</td><td>0.417</td><td>0.754</td><td>0.924</td><td>-0.163</td></tr><tr><td>w/o residual branch</td><td>0.173</td><td>0.361</td><td>0.815</td><td>0.937</td><td>-0.102</td></tr><tr><td>w/o temperature branch</td><td>0.149</td><td>0.284</td><td>0.886</td><td>0.941</td><td>-0.031</td></tr><tr><td>w/o thermodynamic consistency</td><td>0.151</td><td>0.280</td><td>0.889</td><td>0.941</td><td>-0.028</td></tr><tr><td>3D encoder → 2D descriptors</td><td>0.155</td><td>0.273</td><td>0.895</td><td>0.935</td><td>-0.022</td></tr></table>

## Condition-Dependent Solubility Response

Moreover, whether TiNN captures the dependence of DES–solute solubility on physical conditions was studied with a focus on the HBD:HBA molar ratio and temperature. Figure 3a,b shows the MAE of TiNN, CatBoost, and MLP across ratio and temperature bins, with the sample count in each bin shown by gray bars. In data-rich regions, the three models show comparable errors. Clearer diferences appear near the boundaries of the dataset. Above 340 K, where the number of test samples decreases sharply, CatBoost and MLP show larger error increases, whereas TiNN remains closer to its overall error level.

The predicted and experimental condition-dependent trends were then compared for representative DES–solute systems. Figure 3c,d shows the temperature dependence of $\mathrm { { S O } } _ { 2 }$ solubility in tetrabutylammonium bromide (TBAB):glycerol (Gly) and TBAB:ethylene glycol (EG) predicted by TiNN and MLP, respectively. TiNN follows both the magnitude and the slope of the experimental trends. In contrast, the MLP predictions are flatter and deviate further from the experimental trend of TBAB:EG.

These results demonstrate that TiNN does not simply interpolate toward an average response over the training distribution but rather retains condition-dependent variation in DES–solute solubility, especially in sparse temperature and ratio regions. This behavior is consistent with the thermodynamics-informed branch, which constrains the temperature-resolved response, and the hierarchical three-molecule representation, which captures system-specific deviations under diferent pressure and composition conditions.

![](images/4b1b830eb71c254ed3e21d233c4eb4be56dbf31aa45589c6ccb7334b28729903.jpg)

![](images/37e924eb17a327ffbb309d56fe2541d557e59e69da2167095b2305dfd51a011d.jpg)

![](images/4ab4d4299351001279d3d160d965770f4788778e4b11d9776dbeb49058260f11.jpg)

![](images/a45a788c65d2bcf6326eda385ddd2983870e4e07c79b02a8b5d4ed85b1592c5b.jpg)

![](images/a442ef95bed5630f63426d589dc12a58b950ddf9a241e0fa83b513cad63619ba.jpg)

(f)  
![](images/157d564081f1d2c04ba65a230ab899df1e0dbfb2b122612d2ed0cbcbf872cf10.jpg)  
Figure 3. Condition-dependent response and compositional generalization of TiNN. (a, b) MAE of TiNN, CatBoost, and MLP across HBD:HBA ratio bins (a) and temperature bins (b); gray bars indicate the number of test samples in each bin. (c, d) Temperature-dependent $\mathrm { S O } _ { 2 }$ solubility in TBAB:Gly and TBAB:EG predicted by TiNN (c) and MLP (d); dash-dotted lines show the experimental trends and gray dashed lines show the predicted trends. (e, f) Out-of-fold parity plots for TiNN (e) and MLP (f) under leave-one-HBA–HBD-pair-out cross-validation. Each of the 81 HBA–HBD combinations was held out once, with all associated records excluded from training. Color indicates point density.

## Compositional Generalization

TiNN’s compositional generalization was investigated across HBA–HBD combinations within the solubility task.

Compositional generalization was evaluated using leave-one-HBA–HBD-pair-out cross-validation across 81 distinct HBA–HBD combinations. In each fold, all records associated with one HBA– HBD pair were excluded from training and used for testing, including records with diferent molar ratios, solutes, temperatures, and pressures. The individual HBA or HBD could still occur in other training combinations, so this evaluation specifically tests whether the model can generalize to new pairings of known molecular components.

The aggregated out-of-fold predictions show that TiNN preserves a clear diagonal parity distribution, whereas the MLP baseline develops horizontal bands indicative of regression toward an averaged response (Figure 3e,f). This behavior indicates that the hierarchical HBA–HBD– solute representation captures composition-level relationships that can be transferred across HBA–HBD pairings, rather than relying primarily on memorization of specific combinations.

## Chemical Significance of Hydrogen-Bond Compatibility

The remarkable performance and generalization results above show that TiNN predicts DESsolute solubility accurately across chemically diverse systems. Accordingly, the origin of this performance was examined to determine whether it reflects a chemically consistent use of the hydrogen-bond compatibility signal incorporated into the model or whether the encoded compatibility merely functions as an arbitrary numerical anchor. However, it cannot be simply inferred from the predictive accuracy alone: the multilevel hydrogen bond network is the molecular determinant of DES-solute interactions and the chemical motivation behind the hierarchical HBA-HBD-solute representation used in TiNN. Therefore, the role of this hydrogen-bonding structure in model predictions was probed through two analyses of the relative hydrogen-bond compatibility descriptor $H ^ { \mathrm { r e l } }$ defined in Methods: an ablation of the $H ^ { \mathrm { r e l } }$ input channel under three perturbations, and an examination of the $H ^ { \mathrm { r e l } }$ value distribution across solute classes (Figure 4a–g).

The TiNN was retrained with three modifications of the $H ^ { \mathrm { r e l } }$ input: values kept unchanged (Full), values set to zero (Zero), and values shufled across DES-solute pairs while preserving the marginal distribution (Shufled). On the full test set, Full and Zero performed within 0.023 in $R ^ { 2 }$ while Shufled dropped by 0.217 (Figure 4a). This asymmetry became more informative when the test set was partitioned by solute type. On the carbohydrate (cellulose- and hemicellulosederived) and lignin-derived subset, where solutes carry O-H donor and acceptor sites, Full outperforms Zero by 0.046 and Shufled by 0.412 in $R ^ { 2 }$ . On the gas subset, Full and Zero were indistinguishable $( \Delta R ^ { 2 } = - 0 . 0 0 1 )$ , consistent with the absence of classical O-H or N-H donor sites on $\mathrm { C O } _ { 2 } , { \mathrm { S O } } _ { 2 }$ , and $\mathrm { H } _ { 2 } \mathbf { S }$ . Rather than relying on classical donor–acceptor hydrogen bonding, gas solutes dissolve in DES through distinct mechanisms: $\mathrm { C O } _ { 2 }$ primarily through cavity formation and dispersion interactions,<sup>47;48</sup> $\mathrm { S O } _ { 2 }$ through charge-transfer and Lewis acid–base interactions with anionic components,<sup>49</sup> and $\mathrm { H } _ { 2 } \mathbf { S }$ through Lewis acid–base interactions.<sup>50</sup> Consequently, the contribution of $H ^ { \mathrm { r e l } }$ to TiNN predictions is mechanism-dependent, i.e., it is pronounced when hydrogen-bond compatibility is chemically informative, but negligible when dissolution is dominated by mechanisms not represented by $H ^ { \mathrm { r e l } }$

The $H ^ { \mathrm { r e l } }$ input carries chemically meaningful patterns across the dataset itself (Figure 4b–g). $H _ { \mathrm { D E S - s o l u t e } } ^ { \mathrm { r e l } }$ spans more than eight orders of magnitude across solute classes: mostly 1–3 for carbohydrates, which present multiple hydroxyl groups as both donors and acceptors; below 2 for lignin-derived compounds, whose aromatic scafolds carry fewer donor sites; and $\sim 1 0 ^ { - 8 }$ for gaseous solutes, which fall to near zero because $\mathrm { C O } _ { 2 } , { \mathrm { S O } } _ { 2 }$ , and $\mathrm { H } _ { 2 } \mathbf { S }$ provide no classical O-H or N-H donor. $H _ { \mathrm { H B A - H B D } } ^ { \mathrm { r e l } } ;$ , which characterizes the internal hydrogen-bond balance of the DES, follows the same rule-based construction. For carbohydrate and lignin-derived solutes, it takes lower values than the DES-solute channel, reflecting the tighter internal donor-acceptor pairing between HBA and HBD relative to the DES-solute interface. For gaseous solutes, the DES-solute channel falls to near zero, so the HBA-HBD channel becomes the larger of the two. These value patterns arise directly from the rule-based donor-acceptor scoring of the constituent molecules and reflect the intrinsic hydrogen-bond structure of the DES-solute system rather than any external labeling or supervision.

The ablation and distribution results show that TiNN incorporates $H ^ { \mathrm { r e l } }$ as a chemically meaningful signal rather than an arbitrary numerical channel. The model responds to the chemical pattern of $H ^ { \mathrm { r e l } }$ values, preferentially uses this signal where hydrogen-bond compatibility governs solvation, and disregards it where dissolution proceeds through other mechanisms. Combined with the ordered HBA-HBD-solute representation and the thermodynamically constrained temperature response established above, this behavior further indicates that TiNN captures DES-solute solubility as a chemically interpretable manifestation of multicomponent solvation, rather than as a direct descriptor-to-property mapping.

## Solubility and Biomass Fractionation Experiment

The predictive capability of TiNN was first experimentally examined using three representative DES environments: acidic ChCl:lactic acid (ChCl:LA), basic TBAB:imidazole (TBAB:Im), and relatively neutral ChCl:glycerol (ChCl:Gly). Their solubilization performance for cellulosederived glucose, hemicellulose-derived xylan, and Kraft lignin was investigated at 323.15 K and 101.325 kPa with an HBD:HBA molar ratio of 2:1. The experimental values were compared with predictions from TiNN and the conventional COSMO-RS method (Figure 4h).

Across the nine DES–solute systems, TiNN achieves a log<sub>10</sub>-scale MAE of 0.18, compared with at least 1.37 for COSMO-RS, corresponding to geometric mean fold errors of approximately 1.5 and at least 23, respectively. For the two systems predicted by COSMO-RS to be completely miscible, a conservative surrogate solubility of $1 0 ^ { 4 } \mathrm { m g } \mathrm { g } ^ { - 1 }$ DES was assigned solely for error calculation. TiNN shows close agreement with the experimental measurements for glucose and Kraft lignin and correctly reproduces the experimental ranking of xylan solubility across the three DESs, although its absolute values are underestimated. This deviation may partly arise from the polydisperse nature of xylan,<sup>51</sup> which is represented by an idealized molecular structure in the model, together with the relatively limited xylan data available for training.

To further examine whether the molecular solubility predictions can support practical DES screening, five DESs spanning diferent predicted lignin–xylan dissolution capacities were evaluated using real lignocellulose (corn stover). The normalized TiNN predictions were compared with the experimentally measured dissolution of lignin and hemicellulose (Figure 4i). Although the intermediate DESs are not reproduced in their exact order, TiNN correctly distinguishes the DESs with the highest and lowest fractionation capacities and captures the overall experimental trend. This result indicates that the HBA–HBD–solute interaction patterns learned by TiNN retain practical relevance in lignocellulose fractionation, where disruption and reorganization of the hydrogen-bond network contributes to lignin and hemicellulose dissolution.

Overall, the two levels of experimental validation show that TiNN not only reproduces molecular solubility trends across chemically distinct DES environments, but can also provide a useful screening signal for distinguishing DESs with relatively high and low lignocellulose fractionation capability. Importantly, the model is not directly trained on whole-biomass fractionation data, and the agreement in Figure 4i therefore reflects transfer of the learned molecular solvation information to a more complex real biomass system.

![](images/5f98845b2e69adf397bdecc7692b852b4966ec7c1d13f6ede39e887ffe09ef72.jpg)

![](images/0c0034ff2b9dd483d3f652635297c2f112af597fcb7f99712840e2d6eb137886.jpg)

![](images/ee86e3cb974259a3bd83fe3ebfa394ad289fd826e264f41919af5808a1ea464f.jpg)

![](images/de8b48a18920804b8a794a16362f6ce8c3f0aa3760b5d9c86d5b79ba00293704.jpg)

![](images/6c834377d195d729cc6b80dbf53a58532bc7f1fcb5c282db5f6997700272db0b.jpg)

![](images/d358a2e6f184eb297341e9c1d65482633a769b5da4ebf77eb3c70bd38435373b.jpg)

![](images/51dcd30f62cd7042094508b52b87ebe86b403f2a3276bb9bdb8515e3e7f71544.jpg)

![](images/72bdcf998acf9f7d980f1b8cd4e79eaab74670a89812e98db5e8a64a09c52648.jpg)

(i)  
![](images/3cd28e8994aabcc1b74b21b0c76b241a08f2c31ad50a87cdae29ace9af3810f8.jpg)  
Figure 4. Hydrogen-bond compatibility and experimental validation of TiNN. (a) Ablation of the $H ^ { \mathrm { r e l } }$ input under three perturbations; bracket labels give $R ^ { 2 } ( \mathrm { F u l l } ) - R ^ { 2 } ( \mathrm { v a r i a n t } )$ and error bars indicate one standard deviation over three splits. (b–g) Distributions of $H _ { \mathrm { D E S - s o l u t e } } ^ { \mathrm { r e l } } \ ( \mathbf { b } , \mathrm { d } , \mathrm { f } )$ and $H _ { \mathrm { H B A - H B D } } ^ { \mathrm { r e l } } \left( \mathrm { c } , \mathrm { e } , \mathrm { g } \right)$ for carbohydrate, lignin-derived, and gaseous solutes. (h) Measured and predicted solubilities in three DESs at 323.15 K; broken bars mark COSMO-RS predictions of complete miscibility. (i) Normalized TiNN predictions and corn stover fractionation for five DESs selected across the predicted dissolution ranking.

## Conclusions

The present work established a thermodynamics-informed framework that resolves DES–solute solvation through hierarchical hydrogen-bond networks and structured multicomponent interactions, enabling molecularly informed DES design for lignocellulose refining. Instead of treating the solvation as a direct descriptor-to-property mapping, the developed TiNN resolves the formation of the HBA–HBD environment and the subsequent accommodation of the solute, while constraining the condition-dependent response through a learned free-energy pathway. This chemically structured representation achieves a test-set $R ^ { 2 }$ of 0.917 and shows condition-dependent behavior and generalization across held-out HBA–HBD pairings. The mechanism-dependent contribution of hydrogen-bond compatibility further supports this interpretation: shufling this signal decreases $R ^ { 2 }$ by 0.412 for carbohydrate and lignin-derived solutes, whereas removing it produces essentially no change for the gas subset.

Furthermore, the experimental results substantiate the remarkable transferability of TiNN. Across independently measured glucose, xylan, and Kraft lignin solubilities in acidic, basic, and neutral DES environments, TiNN achieves a geometric mean error of approximately 1.5-fold, compared with more than 23-fold for COSMO-RS, indicating that the learned HBA–HBD– solute relationships extend beyond the specific combinations represented in the training data. More notably, this molecular solvation information remains informative when transferred to the substantially more complex process of real lignocellulose fractionation. Although TiNN was not trained on fractionation data and does not explicitly represent the structural and kinetic complexity of lignocellulose deconstruction, its predictions still distinguish DES formulations with relatively high and low lignin–hemicellulose fractionation performances. Overall, this cross-scale transfer suggests that the chemically structured solvation learning can provide a meaningful bridge between molecular property prediction and practical solvent screening, ofering a route toward more physically informed and rational design of multicomponent DES systems.

## Methods

## Dataset and Model Representation

The dataset contains 938 DES–solute solubility records, each specifying the HBA, HBD, solute, temperature, pressure, and HBD:HBA ratio. Solubility was expressed in mg $\mathbf { g } ^ { - 1 }$ DES and modeled as log $S = \log _ { 1 0 } ( S )$ . Solute-stratified splitting assigned approximately 72%, 13%, and 15% of records to training, validation, and testing, respectively.

The HBA, HBD, and solute were encoded separately from COSMO-RS molecular conformations using an EGNN. Their embeddings were combined hierarchically, first into a DES representation and then into a DES–solute representation:

$$
\begin{array} { r l } & { \mathbf { z } _ { \mathrm { D E S } } = \Phi _ { \mathrm { D E S } } ( x _ { \mathrm { H B A } } \mathbf { z } _ { \mathrm { H B A } } , x _ { \mathrm { H B D } } \mathbf { z } _ { \mathrm { H B D } } , H _ { \mathrm { H B A - H B D } } ^ { \mathrm { r e l } } ) , } \\ & { \mathbf { z } _ { \mathrm { s y s t e m } } = \Phi _ { \mathrm { s o l v } } ( \mathbf { z } _ { \mathrm { D E S } } , \mathbf { z } _ { \mathrm { s o l u t e } } , H _ { \mathrm { D E S - s o l u t e } } ^ { \mathrm { r e l } } ) , } \end{array}\tag{3}
$$

where � denotes molar fraction. Hydrogen-bond donor and acceptor scores $( D _ { i }$ and $A _ { i } )$ were assigned using rule-based molecular-site identification with local chemical corrections. The relative DES–solute compatibility was defined as

$$
H ^ { \mathrm { r e l } } = \frac { \mathrm { m i n } ( D _ { \mathrm { D E S } } A _ { \mathrm { s o l u t e } } , A _ { \mathrm { D E S } } D _ { \mathrm { s o l u t e } } ) + \epsilon } { A _ { \mathrm { D E S } } + D _ { \mathrm { D E S } } + \epsilon } ,\tag{4}
$$

with DES-level capacities calculated as molar-fraction-weighted sums. The same form was applied to the HBA–HBD pair.

## Thermodynamics-Informed Learning

The latent free energy combined solute-specific, DES–solute coupling, and temperature-dependent contributions:

$$
\Delta G _ { \theta } = \Delta G _ { \theta } ^ { \mathrm { s o l u t e } } ( \mathbf { z } _ { \mathrm { s o l u t e } } ) + \Delta G _ { \theta } ^ { \mathrm { e n v } } ( \mathbf { z } _ { \mathrm { D E S } } , \mathbf { z } _ { \mathrm { s o l u t e } } ) + \Delta G _ { \theta } ^ { T } ( \mathbf { z } _ { \mathrm { s y s t e m } } , \mathbf { z } _ { T } ) ,\tag{5}
$$

where ${ \bf z } _ { T } = \Phi _ { T } ( T )$ . The enthalpy–entropy relation in Eq. (1) and the relation log $S _ { \mathrm { t h e r m o } } +$ $\Delta G _ { \theta } / ( \alpha T ) \approx 0$ were imposed as soft constraints, with trainable scale �. A residual term was added as in Eq. (2). Training jointly minimized log-solubility prediction error and thermodynamic regularization; the checkpoint with the highest validation $R ^ { 2 }$ was selected.

Performance was assessed using MAE, RMSE, $R ^ { 2 }$ , and Spearman correlation. Compositional generalization was evaluated by holding out each of 81 distinct HBA–HBD pairs in turn, regardless of molar ratio. For each fold, the other pairs were split into training (85%) and validation (15%) sets, and held-out predictions were aggregated.

## Experimental Validation

Glucose, xylan, and Kraft lignin solubilities were measured in ChCl:LA, TBAB:Im, and ChCl:Gly (HBD:HBA = 2:1) at 323.15 K. Solute was added incrementally until undissolved particles were observed; solubility was reported as $S = 1 0 0 0 m _ { s } / m _ { \mathrm { D E S } } \mathrm { ~ ( m g ~ g ^ { - 1 } ~ D E S ) }$ , with $m _ { s }$ and $m _ { \mathrm { D E S } }$ in grams.

Five DESs selected across the predicted xylan–Kraft lignin dissolution ranking were tested for corn stover fractionation at 373.15 K using a biomass-to-DES mass ratio of 1:40. Cellulose, hemicellulose, and lignin contents were compared before and after treatment.

## References

[1] Ragauskas, A. J. et al. Lignin valorization: improving lignin processing in the biorefinery. Science 2014, 344, 1246843.

[2] Himmel, M. E.; Ding, S.-Y.; Johnson, D. K.; Adney, W. S.; Nimlos, M. R.; Brady, J. W.; Foust, T. D. Biomass recalcitrance: engineering plants and enzymes for biofuels production. Science 2007, 315, 804–807.

[3] Rinaldi, R.; Jastrzebski, R.; Clough, M. T.; Ralph, J.; Kennema, M.; Bruijnincx, P. C.;

Weckhuysen, B. M. Paving the way for lignin valorisation: recent advances in bioengineering, biorefining and catalysis. Angewandte Chemie International Edition 2016, 55, 8164–8215.

[4] Renders, T.; Van den Bosch, S.; Koelewijn, S.-F.; Schutyser, W.; Sels, B. Lignin-first biomass fractionation: the advent of active stabilisation strategies. Energy & Environmental Science 2017, 10, 1551–1557.

[5] Li, Y.; Yu, Y.; Lou, Y.; Zeng, S.; Sun, Y.; Liu, Y.; Yu, H. Hydrogen-transfer reductive catalytic fractionation of lignocellulose: high monomeric yield with switchable selectivity. Angewandte Chemie International Edition 2023, 62, e202307116.

[6] Xu, M.; Zhu, X.; Lai, Y.; Xia, A.; Huang, Y.; Zhu, X.; Liao, Q. Production of hierarchical porous bio-carbon based on deep eutectic solvent fractionated lignin nanoparticles for high-performance supercapacitor. Applied Energy 2024, 353, 122095.

[7] Xu, M.-M.; Shi, Z.; Zhu, X.-Q.; Lai, Y.; Xia, A.; Huang, Y.; Jiang, X.; He, J.; Zhou, M.; Zhu, X.; Liao, Q. Ex-situ catalytic upgrading of biomass pyrolysis volatiles over thermaldecomposition products of spent lithium-ion batteries for bio-oil deoxygenation and hydrogen-rich syngas production. International Journal of Hydrogen Energy 2024, 52, 83–96.

[8] Zhang, Q.; De Oliveira Vigier, K.; Royer, S.; Jer´ ome, F. Deep eutectic solvents: syntheses,ˆ properties and applications. Chemical Society Reviews 2012, 41, 7108–7146.

[9] Yue, X.; Lin, J.; Mankinen, O.; Suopajarvi, T.; Mikola, M.; Mikkelson, A.; Huttunen, H.;¨ Chen, L.; Ahola, J.; Telkki, V.-V.; Sun, S.; Liimatainen, H. Lignin Dissolution and Direct Ultrasmall-Lignin-Nanoparticle Formation in Acidic and Alkaline Deep Eutectic Solvents: A Molecular-Level Insight. Angewandte Chemie International Edition 2025, 64, e202505975.

[10] Smith, E. L.; Abbott, A. P.; Ryder, K. S. Deep eutectic solvents (DESs) and their applications. Chemical Reviews 2014, 114, 11060–11082.

[11] Francisco, M.; Van Den Bruinhorst, A.; Kroon, M. C. New natural and renewable low transition temperature mixtures (LTTMs): screening as solvents for lignocellulosic biomass processing. Green Chemistry 2012, 14, 2153–2157.

[12] Alvarez-Vasco, C.; Ma, R.; Quintero, M.; Guo, M.; Geleynse, S.; Ramasamy, K. K.; Wolcott, M.; Zhang, X. Unique low-molecular-weight lignin with high purity extracted from wood by deep eutectic solvents (DES): a source of lignin for valorization. Green Chemistry 2016, 18, 5133–5141.

[13] Shen, X.-J.; Wen, J.-L.; Mei, Q.-Q.; Chen, X.; Sun, D.; Yuan, T.-Q.; Sun, R.-C. Facile fractionation of lignocelluloses by biomass-derived deep eutectic solvent (DES) pretreatment for cellulose enzymatic hydrolysis and lignin valorization. Green Chemistry 2019, 21, 275–283.

[14] Abbott, A. P.; Capper, G.; Davies, D. L.; Rasheed, R. K.; Tambyrajah, V. Novel solvent properties of choline chloride/urea mixtures. Chemical Communications 2003, 70–71.

[15] Abbott, A. P.; Boothby, D.; Capper, G.; Davies, D. L.; Rasheed, R. K. Deep eutectic solvents formed between choline chloride and carboxylic acids: versatile alternatives to ionic liquids. Journal ofthe American Chemical Society 2004, 126, 9142–9147.

[16] Zhang, T.; Yu, J.-T.; Lin, T.; Song, Y.-Q.; Li, M.; Feng, Y.; Song, L.; Jiang, Y.; Xie, W.; Lu, Y.; Liu, T.-F.; Zhang, K.; Chen, J. Fluorinated deep eutectic gel electrolytes for sustainable lithium metal batteries. Journal of the American Chemical Society 2025, 147, 32861–32872.

[17] Hansen, B. B. et al. Deep eutectic solvents: a review of fundamentals and applications. Chemical Reviews 2021, 121, 1232–1285.

[18] Pandey, A.; Rai, R.; Pal, M.; Pandey, S. How polar are choline chloride-based deep eutectic solvents? Physical Chemistry Chemical Physics 2013, 16, 1559–1568.

[19] Chen, Z.; Wan, C. Ultrafast fractionation of lignocellulosic biomass by microwave-assisted deep eutectic solvent pretreatment. Bioresource Technology 2018, 250, 532–537.

[20] Ashworth, C. R.; Matthews, R. P.; Welton, T.; Hunt, P. A. Doubly ionic hydrogen bond interactions within the choline chloride–urea deep eutectic solvent. Physical Chemistry Chemical Physics 2016, 18, 18145–18160.

[21] Hammond, O. S.; Bowron, D. T.; Edler, K. J. The efect of water upon deep eutectic solvent nanostructure: an unusual transition from ionic mixture to aqueous solution. Angewandte Chemie International Edition 2017, 56, 9782–9785.

[22] Florindo, C.; Oliveira, F. S.; Rebelo, L. P. N.; Fernandes, A. M.; Marrucho, I. M. Insights into the synthesis and properties of deep eutectic solvents based on cholinium chloride and carboxylic acids. ACS Sustainable Chemistry & Engineering 2014, 2, 2416–2425.

[23] Lemaoui, T.; Boublia, A.; Lemaoui, S.; Darwish, A. S.; Ernst, B.; Alam, M.; Benguerba, Y.; Banat, F.; AlNashef, I. M. Predicting the CO2 capture capability of deep eutectic solvents and screening over 1000 of their combinations using machine learning. ACS Sustainable Chemistry & Engineering 2023, 11, 9564–9580.

[24] Tan, F.; Zhao, Z.-X.; Jiang, Y.-B.; Zhang, W.-C.; Xie, T.-L.; Chen, W.; Ma, M.; Liu, Y.; Ye, Y.; Xiao, Z.; Lei, Q.; Xu, G.-F.; Ren, J.; Zhao, Y.-Y.; Li, Z. Dual engine-driven strategy for advanced copper alloy design employing large language models. Acta Materialia 2026, 121886.

[25] Tan, Y. T.; Chua, A. S. M.; Ngoh, G. C. Deep eutectic solvent for lignocellulosic biomass fractionation and the subsequent conversion to bio-based products–A review. Bioresource Technology 2020, 297, 122522.

[26] Zhou, M.; Fakayode, O. A.; Yagoub, A. E. A.; Ji, Q.; Zhou, C. Lignin fractionation from lignocellulosic biomass using deep eutectic solvents and its valorization. Renewable and Sustainable Energy Reviews 2022, 156, 111986.

[27] Mqoni, N.; Bahadur, I.; Singh, S.; Meng, X.; Ragauskas, A. Deep Eutectic Solvents for Pretreatment of Lignocellulose Biomass: Physical Properties, Solubility Mechanisms, and Their Interactions. Chemical Reviews 2026, 126, 1206–1257.

[28] Al Ibrahim, E.; Morgan, N.; Muller, S.; Motati, S.; Green, W. H. Accurately predicting solubility curves via a thermodynamic cycle, machine learning, and solvent ensembles. Journal of the American Chemical Society 2025, 147, 45057–45069.

[29] Wojeicchowski, J. P.; Ferreira, A. M.; Abranches, D. O.; Mafra, M. R.; Coutinho, J. A. Using COSMO-RS in the design of deep eutectic solvents for the extraction of antioxidants from rosemary. ACS Sustainable Chemistry & Engineering 2020, 8, 12132–12141.

[30] Wang, K.; Peng, D.; Alhadid, A.; Minceva, M. Assessment of COSMO-RS for predicting liquid–liquid equilibrium in systems containing deep eutectic solvents. Industrial & Engineering Chemistry Research 2024, 63, 11110–11120.

[31] Song, Z.; Wang, J.; Sundmacher, K. Evaluation of COSMO-RS for solid–liquid equilibria prediction of binary eutectic solvent systems. Green Energy & Environment 2021, 6, 371–379.

[32] Xue, L.-L.; Guo, J.-T.; Wu, Y.-Q.; Dong, Y.; Li, J.-H.; Yao, M.; Peng, Y.-Y.; Zhang, F.-M.; Liu, S.-C.; Yu, M.; Li, J.-W.; Cheng, F.-Y. A Three-Descriptor Framework for Solvation Design of Low-Temperature Electrolytes in Lithium Batteries. Journal of the American Chemical Society 2026, 148, 44103–44115.

[33] Butler, K. T.; Davies, D. W.; Cartwright, H.; Isayev, O.; Walsh, A. Machine learning for molecular and materials science. Nature 2018, 559, 547–555.

[34] Ye, Y.; Ren, J.; Wang, S.; Wan, Y.; Razzak, I.; Hoex, B.; Wang, H.; Xie, T.; Zhang, W. Construction and application of materials knowledge graph in multidisciplinary materials science via large language model. Advances in Neural Information Processing Systems 2024, 37, 56878–56897.

[35] Zhao, Z.; Tan, F.; Xie, T.; Chen, W.; Yu, J.-P.; Ye, Y.; Liao, Z.-K.; Xu, G.-F.; Lee, B.; Jiang, Y.- B.; Li, Z. Automated distillation of composition–processing–structure–performance linkages from fragmented and multi-modal materials literature. Materials & Design 2026, 116250.

[36] Ma, S.; Li, Q.; Wang, J. CAT-MPNN: A dual-mode network to predict the density of deep eutectic solvents forming an open-access database. Chemical Engineering Science 2025, 305, 121097.

[37] Yu, L.-Y.; Ren, G.-P.; Hou, X.; Wu, K.-J.; He, Y. Transition State Theory-Inspired Neural Network for Estimating the Viscosity of Deep Eutectic Solvents. ACS Central Science 2022, 8, 983–995.

[38] Lavrinenko, A. K.; Chernyshov, I. Y.; Pidko, E. A. Machine Learning Approach for the Prediction of Eutectic Temperatures for Metal-Free Deep Eutectic Solvents. ACS Sustainable Chemistry & Engineering 2023, 11, 15492–15502.

[39] Wang, J.; Song, Z.; Chen, L.; Xu, T.; Deng, L.; Qi, Z. Prediction of CO2 solubility in deep eutectic solvents using random forest model based on COSMO-RS-derived descriptors. Green Chemical Engineering 2021, 2, 431–440.

[40] Sharma, A.; Garg, A.; Li, L.; Chatterjee, I.; Lee, B.-s.; Garg, A. Machine learning for deep eutectic solvents: advances in property prediction and molecular design. Journal of Molecular Liquids 2025, 128317.

[41] Sheikhshoaei, A. H.; Sanati, A. Data-driven insights into CO solubility in deep eutectic solvents. Journal ofCO2 Utilization 2025, 102, 103285.

[42] Xing, Y.; Sun, Y.; Zhou, X.; Xiu, Z. A novel self-attention mechanism-based machine learning model for high-accuracy prediction of CO2 solubility in natural deep eutectic solvents. Journal of Environmental Management 2025, 393, 127275.

[43] Mohan, M.; Demerdash, O. N.; Simmons, B. A.; Singh, S.; Kidder, M. K.; Smith, J. C. Physics-based machine learning models predict carbon dioxide solubility in chemically reactive deep eutectic solvents. ACS Omega 2024, 9, 19548–19559.

[44] Wang, Y.; Wang, H.; Yan, C.; Wang, W.; Jia, B.; Yao, Y.; Chen, L.; Hu, Q.; Xue, Z. Thermodynamics insights of lignin dissolution in deep eutectic solvents. International Journal of Biological Macromolecules 2025, 300, 140224.

[45] Mohan, M.; Naik, P. K.; Banerjee, T.; Goud, V. V.; Paul, S. Solubility of glucose in tetrabutylammonium bromide based deep eutectic solvents: Experimental and molecular dynamic simulations. Fluid Phase Equilibria 2017, 448, 168–177.

[46] Chen, K.; Chen, H.; Zheng, Z.; Liu, Y.; Fang, Y.; Chen, Z.; Cao, Y. Identifying the role of solvation entropy for the solvation chemistry in nonaqueous electrolytes. Angewandte Chemie International Edition 2025, 64, e202501315.

[47] Pelaquim, F. P.; Barbosa Neto, A. M.; Dalmolin, I. A. L.; Costa, M. C. d. Gas solubility using deep eutectic solvents: review and analysis. Industrial & Engineering Chemistry Research 2021, 60, 8607–8620.

[48] Chi, H.-Y.; Tsao, H.-K.; Sheng, Y.-J. Mechanisms of CO2 absorption in amino acid-based deep eutectic solvents: insights from molecular dynamics and DFT calculations. The Journal of Physical Chemistry B 2025, 129, 5779.

[49] Li, H.; Chang, Y.; Zhu, W.; Wang, C.; Wang, C.; Yin, S.; Zhang, M.; Li, H. Theoretical evidence of charge transfer interaction between SO2 and deep eutectic solvents formed by choline chloride and glycerol. Physical Chemistry Chemical Physics 2015, 17, 28729–28742.

[50] Wang, B.; Xie, X.; Wan, L.; Zhao, W.; Chen, Y. H2S absorption with deep eutectic solvents: low partial pressure capture and thermodynamic analysis. AIChE Journal 2023, 69, e18087.

[51] Scheller, H. V.; Ulvskov, P. Hemicelluloses. Annual Review of Plant Biology 2010, 61, 263–289.