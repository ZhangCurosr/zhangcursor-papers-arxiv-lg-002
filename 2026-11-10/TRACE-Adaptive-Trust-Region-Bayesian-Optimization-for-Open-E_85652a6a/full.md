# TRACE: Adaptive Trust-Region Bayesian Optimization for Open-Ended Molecular Design

Madhav R. Muthyala UW-Madison mmuthyala@wisc.edu

Mahsa Askari Texas Tech University Mahsa.Askari@ttu.edu

Farshud Sorourifar The Ohio State University sorourifar.1@osu.edu

Evan Zhou UW-Madison ezhou37@wisc.edu

Joe Gauthier Texas Tech University Joe.Gauthier@ttu.edu

Joel A. Paulson UW-Madison joel.paulson@wisc.edu

## Abstract

Optimizing molecules under small evaluation budgets requires deciding both where in chemical space to search and which candidate within that region to evaluate. We introduce Trust Region Adaptive Chemical Exploration (TRACE), a framework for open-ended molecular Bayesian optimization (BO), where candidates are generated on demand rather than selected from a fixed library. TRACE combines an objective-free graph-perturbation mechanism with a fixed, chemistry-aware Gaussian process (GP) surrogate using fragprints, a representation combining molecular fingerprints and fragment descriptors. Rather than learning the search geometry itself, TRACE makes the scale of chemical exploration explicit through a Tanimotosimilarity threshold controlling chemical extent and a cardinality budget controlling candidate-search breadth. A simple success/failure controller expands or contracts both quantities during optimization, while the GP and acquisition function select which admissible molecules to evaluate. Across six challenging molecular optimization benchmarks under matched 1,000-evaluation budgets, TRACE achieves the strongest overall sample efficiency as measured by the best molecule found as evaluations accumulate. Crossed ablations on two objectives further show that adaptive region control contributes beyond the proposal mechanism, while its benefit depends on the proposal distribution. We further apply TRACE to the stability of iron phthalocyanine (FePc), a non-precious molecular electrocatalyst whose degradation can involve loss of the Fe center. Using a first-principles density functional theory (DFT) workflow designed for accurate energy evaluation, TRACE increases the computed thermodynamic resistance to Fe dissolution from 0.42 to 2.59 eV over 80 designed evaluations.

## 1 Introduction

Molecular optimization is a fundamental challenge in chemistry, with applications ranging from drug discovery to catalyst design [1, 2, 3]. In the “open-ended” setting considered here, candidate molecule are generated on demand rather than selected from a fixed, pre-enumerated library. Optimization then involves two intertwined decisions, specifically, where in chemical space to search and which candidate within that region to evaluate next. The second has received substantial attention through surrogate models and acquisition functions. The first is often encoded in the candidate-generation mechanism, representation-dependent neighborhoods, similarity criteria, or the number of candidate proposed per iteration. Graph-based genetic algorithms traverse chemical space through prescribed mutation and crossover operations [4]; STONED constructs local neighborhoods through discrete modifications of a string representation [5]. These choices often encode considered chemical judgment about what constitutes a sensible neighborhood. In many molecular optimization pipelines, these choices are specified before optimization and remain fixed throughout the run. They nonetheless determine the effective scale of exploration, and thus how a limited oracle budget is distributed across a combinatorially large space.

On the Practical Molecular Optimization (PMO) benchmark, most molecular design algorithms introduced as state of the art fail to outperform substantially simpler predecessors under a fixed allowance of 10<sup>4</sup> oracle calls [6]. Even this budget is large relative to many discovery settings. A firstprinciples electronic structure calculation for a candidate catalyst can require hours of computation, while synthesis and experimental characterization can take days or longer. When evaluation dominates cost, the scale of exploration can strongly affect how the oracle budget is used. Searching too narrowly can concentrate evaluations on marginal variations of a local scaffold, while searching too broadly requires the optimizer to discriminate among candidates spanning a larger and more heterogeneous portion of chemical space. The appropriate scale may also vary with the task, the region currently under consideration, and the observations accumulated during optimization. Bayesian optimization (BO) is well suited to such settings because it uses a probabilistic surrogate and acquisition function to guide expensive evaluations [7]. GAUCHE [8] showed that Tanimoto-kernel Gaussian processes (GPs) using fragprints, which concatenate Morgan fingerprints with RDKit fragment descriptors, perform well in predictive accuracy, uncertainty calibration, and runtime on small- to mid-sized molecular datasets, and demonstrated their use in data-limited molecular BO. A separate study [9] showed that after correcting an incorrect prior width, over-smoothing, and inadequate acquisitionfunction maximization, a basic GP-based BO setup attains the highest overall performance on PMO under their 10,000-call evaluation protocol. Their remedy for inadequate acquisition maximization uses a genetic algorithm to search molecular space more extensively, highlighting the role of candidate generation in the acquisition stage. Together, these results motivate retaining a lightweight molecular GP while making the scale of candidate search adaptive.

In (high-dimensional) continuous domains, TuRBO [10] provides a useful template for adapting the scale of local BO from optimization progress. It restricts search to a hyperrectangle around the incumbent and expands/contracts that region after consecutive successes/failures. The analogous choice in molecular design is less straightforward because locality depends strongly on the representation. Categorical trust region methods use Hamming distance over fixed-length categorical or mixed representations [11, 12], which does not directly encode structural similarity for variable-size molecular graphs. Latent-space molecular BO instead defines local regions in learned embeddings [13, 14, 15, 16]. Recent methods have addressed important aspects of the mismatch between latent geometry and the underlying molecular or objective space, but the resulting notion of locality remains tied to a learned representation. Other methods explore chemical space through genetic search or learned molecular generators [4, 17, 18]. TRACE instead makes chemical search scale explicit by adapting the similarity-defined region and candidate set passed to the acquisition function.

A fingerprint-similarity threshold, however, does not fully specify the finite candidate set over which BO operates. The threshold τ controls chemical extent, or how dissimilar an admitted molecule may be from the current reference, whereas a cardinality budget k controls candidate-search breadth, or how many admissible molecules are passed to the acquisition function. Because the number of generated molecules satisfying a given threshold depends on the current reference and proposal distribution, the same value of τ can correspond to very different candidate set sizes. TRACE therefore couples τ and k through a simple monotone map, giving a single adaptive state that expands or contracts both chemical extent and candidate-search breadth. This coupling is a practical design choice, and other mappings or separate controllers are possible. Fingerprint similarity is also only an imperfect proxy for property similarity, as illustrated by activity cliffs between structurally similar molecules with sharply different properties [19]. Rather than fixing one nominal similarity scale for the entire run, TRACE adjusts the search scale in response to optimization progress.

We introduce TRACE (Trust Region Adaptive Chemical Exploration), a framework for open-ended molecular BO that makes the scale of candidate search explicit and adaptive. An objective-free graphperturbation mechanism generates molecular proposals around the current search center. TRACE filters these proposals using a Tanimoto-similarity threshold τ that controls chemical extent and a cardinality budget k that controls candidate-search breadth, with a common success/failure controller expanding or contracting both during optimization. A Tanimoto-kernel GP over fragprints [8] and an acquisition function then select molecules from the admissible set for evaluation. Figure 1 summarizes this division of roles between proposal generation, adaptive region control, and objectiveaware candidate selection. Across six PMO objectives, TRACE shows strong sample efficiency under a matched low-budget protocol, and crossed ablations on two objectives show that the adaptive region contributes beyond the proposal mechanism while providing no corresponding benefit with Graph-GA proposals. In a closed-loop density functional theory (DFT) campaign on iron phthalocyanine (FePc) derivatives, TRACE improves the computed Fe-dissolution-resistance objective by 2.17 eV over 80 designed evaluations while moving substantially away from the initial molecule. This expensiveoracle setting is directly relevant to AI-guided materials design, where efficiently allocating limited high-fidelity evaluations can help accelerate materials discovery.

![](images/28058d103ba4e3e3972c41c879d3fc747a99ed6d6bc5e2c46588cdadcb2ad27c.jpg)  
Figure 1: Overview of the TRACE framework. At iteration t, an objective-free proposal mechanism generates candidates around the current search center. The adaptive chemical region admits candidates according to a Tanimoto-similarity threshold $\tau _ { t }$ and cardinality budget $k _ { t } .$ , which are updated by a common success/failure controller. Predictions from a GP surrogate fitted to the evaluated set $\mathcal { D } _ { t }$ are used by a batch acquisition function to select molecules from the admissible set $\boldsymbol { A } _ { t }$ for oracle evaluation. The resulting observations update the evaluated set and the incumbent, and whether the batch meets the controller success criterion provides the signal used to expand or contract the region.

## 2 TRACE Design

## 2.1 Search center and admissible set

Let X denote chemical space, $f : \mathcal { X }  \mathbb { R }$ the oracle, and $\mathcal { D } _ { t }$ the evaluated data available at the start of iteration $t ,$ consisting of molecules and their observed oracle values. We write $\mathcal { X } ( \mathcal { D } _ { t } )$ for the molecules in $\mathcal { D } _ { t }$ and let $\boldsymbol { x } _ { t } ^ { \mathrm { b e s t } }$ denote the global incumbent, the best molecule evaluated so far. TRACE also maintains a search center $c _ { t }$ used to generate and filter proposals. In ordinary iterations $c _ { t } = x _ { t } ^ { \mathrm { b e s t } }$ ; a restart may instead select another previously evaluated molecule as the search center without changing the global incumbent.

We measure structural similarity with the Tanimoto coefficient $T ( x , x ^ { \prime } )$ computed on binary Morgan fingerprints with radius 2 and 2048 bits [20]. A Tanimoto-similarity threshold $\tau _ { t }$ defines the conceptual chemical region around the search center,

$$
\mathcal { R } _ { t } = \left\{ x \in \mathcal { X } : T ( x , c _ { t } ) \geq \tau _ { t } \right\} .\tag{1}
$$

TRACE does not enumerate $\mathcal { R } _ { t }$ . Instead, the proposal mechanism supplies a finite, deduplicated set $\mathcal { P } _ { t } .$ . After removing evaluated molecules and applying the similarity threshold, let

$$
\begin{array} { r l } & { S _ { t } = \{ x \in \mathcal { P } _ { t } \setminus \mathcal { X } ( \mathcal { D } _ { t } ) : T ( x , c _ { t } ) \geq \tau _ { t } \} , } \\ & { \mathcal { A } _ { t } \subseteq S _ { t } , \qquad | \mathcal { A } _ { t } | = \operatorname* { m i n } ( k _ { t } , | \mathcal { S } _ { t } | ) . } \end{array}\tag{2}
$$

When $\vert S _ { t } \vert > k _ { t }$ , the admissible set $\boldsymbol { A } _ { t }$ contains the $k _ { t }$ molecules in $S _ { t }$ with largest $T ( \boldsymbol { x } , c _ { t } )$ ; otherwise $\mathbf { \mathcal { A } } _ { t } = \mathbf { \dot { S } } _ { t }$ . Thus, $\tau _ { t }$ controls chemical extent, while the cardinality budget $k _ { t }$ controls candidate-search breadth. The admissible set determines where objective-aware selection operates: the GP posterior and a batch acquisition function $\alpha _ { t }$ select a batch $B _ { t } \subseteq A _ { t }$ with $| B _ { t } | = q _ { t }$ , where $q _ { t }$ is the number of oracle evaluations available in the current batch (normally the nominal batch size $q ,$ reduced only when fewer evaluations remain in the total budget).

## 2.2 Adaptive region control

TRACE uses $k _ { t } \in [ k _ { \operatorname* { m i n } } , k _ { \operatorname* { m a x } } ]$ as a single adaptive state. Let $\tau _ { \mathrm { h i } }$ and $\tau _ { \mathrm { l o } }$ denote the largest and smallest allowed similarity thresholds, respectively, with $0 \leq \tau _ { \mathrm { l o } } < \tau _ { \mathrm { h i } } \leq 1$ , and let $k _ { 0 } \in [ k _ { \operatorname* { m i n } } , k _ { \operatorname* { m a x } } ]$ be the initial cardinality budget. TRACE couples $\tau$ to k through

$$
\tau ( k ) = \tau _ { \mathrm { h i } } - \frac { k - k _ { \mathrm { m i n } } } { k _ { \mathrm { m a x } } - k _ { \mathrm { m i n } } } \left( \tau _ { \mathrm { h i } } - \tau _ { \mathrm { l o } } \right) .\tag{3}
$$

Hence, $\tau ( k _ { \mathrm { m i n } } ) = \tau _ { \mathrm { h i } }$ and $\tau ( k _ { \mathrm { m a x } } ) = \tau _ { \mathrm { l o } }$ . Expanding the region increases k and lowers $\tau ,$ increasing candidate-search breadth while allowing more chemically distant proposals. On the other hand, contracting the region decreases k and raises $\tau ,$ concentrating search nearer the center. Eq. (3) is a deliberately simple monotone coupling that gives TRACE one adaptive state; other mappings or separate controllers are possible.

After each evaluated batch, TRACE updates $x _ { t + 1 } ^ { \mathrm { b e s t } }$ to the best molecule in $\mathcal { D } _ { t + 1 }$ , regardless of the size of the improvement. For a relative tolerance $\varepsilon \ge 0$ , the iteration is a controller success when

$$
f ( \boldsymbol { x } _ { t + 1 } ^ { \mathrm { b e s t } } ) > f ( \boldsymbol { x } _ { t } ^ { \mathrm { b e s t } } ) + \varepsilon \operatorname* { m a x } \left( 1 , \left| f ( \boldsymbol { x } _ { t } ^ { \mathrm { b e s t } } ) \right| \right) ,
$$

and is otherwise a failure. After $n _ { \mathrm { s u c c } }$ consecutive successes, TRACE expands by setting $k \gets$ min $( k _ { \operatorname* { m a x } } , 2 k ) ;$ ; after $n _ { \mathrm { f a i l } }$ consecutive failures, it contracts by halving k. Here $n _ { \mathrm { s u c c } }$ and $n _ { \mathrm { f a i l } }$ are the corresponding consecutive-event thresholds. If another contraction would place k below the feasible batch floor max $( k _ { \operatorname* { m i n } } , q _ { t } )$ , TRACE restarts instead, resetting k to $k _ { 0 }$ and choosing a new search center from evaluated molecules. A restart can also occur when the current proposal supply cannot fill a batch: TRACE first generates an additional relaxed, objective-free fallback pool and restarts only if the augmented admissible set remains underfilled. Algorithm 1 in Appendix ${ \mathrm { A . 1 } }$ gives the complete state-transition logic and restart rule. Implementation details and code are available at https: $\bar { / / \mathrm { g i } }$ thub.com/PaulsonLab/TRACE-AI4MAT.

## 2.3 Objective-free proposal generation and BO selection

The proposal mechanism generates structural variation without observing oracle values. It applies chemistry-aware atom, bond, fragment, and ring edits under constraints derived from the center used to generate each proposal pool. The proposer may adapt its internal behavior using structural and proposal-history information, including operator outcomes, locality, shell coverage, novelty, and molecular complexity. Because the search center is supplied by the outer optimization loop, proposal generation depends indirectly on optimization history; “objective-free” means that the proposal routine itself never observes $f ( x )$ or uses objective values to rank molecules.

Each iteration combines a fresh strict local pool generated around $c _ { t }$ with a broader relaxed reservoir retained from initialization or the most recent restart. If too few molecules survive admission, a temporary relaxed fallback pool augments the proposal set before TRACE considers restarting. Algorithm 2 in Appendix A.2 gives the proposal routine and the roles of these constraint profiles.

For objective-aware selection, TRACE fits a GP to all observations in $\mathcal { D } _ { t }$ using fragprints, which concatenate Morgan fingerprints with RDKit fragment descriptors, and a Tanimoto kernel [8]. This surrogate representation is distinct from the radius-2, 2048-bit Morgan fingerprints used to define T. A batch acquisition function $\alpha _ { t }$ computed from the GP posterior is optimized over the finite admissible set $\boldsymbol { A } _ { t }$ to select $B _ { t }$ . The experiment-specific acquisition choices are reported with the corresponding experimental protocols.

## 3 Experiments

## 3.1 Low-budget molecular optimization benchmarks

We first test whether adaptive control of chemical search scale translates into improved sample efficiency on established molecular optimization objectives. We select six objectives from the Practical Molecular Optimization (PMO) benchmark [6]: QED, median molecule 2, osimertinib MPO, scaffold hopping, decoration hopping, and DRD2. We chose these objectives to cover different optimization tasks with limited redundancy: single-property drug-likeness (QED), multi-target similarity (median molecule 2), multi-property objectives (osimertinib MPO), structural hopping (scaffold and decoration hopping, which differ in what is held fixed), and a learned activity classifier (DRD2). Our aim is not to reproduce the full PMO leaderboard, which contains 23 objectives and uses a substantially larger oracle budget, but to ask whether a consistent low-budget signal emerges when the methods are compared under the same evaluation constraints.

For each objective and seed, all methods start from the same 20 evaluated molecules. They then receive 20 rounds of active optimization with batch size $q = 4 9$ , for a total budget of 1,000 uniquemolecule oracle evaluations including initialization. We use five seeds and apply the same oracle and molecular constraint filter $( \mathrm { S A } \leq 7 ;$ SMILES length $\leq 1 0 0 )$ throughout. For TRACE, the Tanimoto-kernel GP described in Section 2 is refit as observations accumulate, and qLogEI [21] is optimized over the current admissible set to select each batch. qLogEI is the batch form of log expected improvement, a numerically stabilized reformulation of expected improvement designed to preserve its improvement-seeking behavior while making the acquisition function easier to optimize.

We compare against methods that represent several distinct strategies for exploring chemical space. Graph-GA [4] and SMILES-GA [22] provide strong evolutionary baselines that search directly through molecular graph and string modifications, respectively. STONED-SELFIES [5] instead explores local neighborhoods through discrete SELFIES edits, providing a useful comparison to explicit local molecular search without a probabilistic surrogate. REINVENT4 [23] represents a modern reinforcement learning (RL) approach in which a generative policy is adapted toward high-scoring molecules. Two baselines are particularly close to the questions motivating TRACE. GP-BO + Graph-GA [9] combines molecular GP-based BO with a graph genetic algorithm for acquisition-function optimization, providing a strong surrogate-guided comparison without TRACE’s adaptive similarity/cardinality region. LOL-BO [14] provides a complementary local-BO comparison in a learned latent space. In our experiments we use its pretrained SELFIES VAE without further representation updates while retaining the latent trust region, deep kernel surrogate, and Thompson sampling-based selection; we therefore denote this comparator as LOL-BO (frozen VAE).

All methods are evaluated under the same 1,000-call accounting and common initialization. Where a published method uses a different feedback or update frequency, we adapt its evaluation schedule to the common batched setting used here while retaining its underlying search strategy. This is important for interpreting the comparison, i.e., the experiment is designed to compare how these different search mechanisms use the same limited oracle budget, rather than to reproduce each method under the larger or differently scheduled protocol used in its original publication. Additional implementation details for the comparators are given in Appendix B.

Figure 2 shows a consistent advantage for TRACE once active optimization begins. Because every method starts from the same 20-molecule set, the subsequent separation of the trajectories reflects how the methods allocate the remaining oracle calls rather than differences in initialization. The clearest gains appear on median molecule 2, osimertinib MPO, scaffold hopping, and decoration hopping. On these objectives, TRACE continues to identify improvements later in the budget while several of the competing trajectories begin to flatten. This behavior is particularly relevant to the motivation for adaptive search scale: progress is not confined to the first few batches, but continues as the region and candidate breadth are updated from the optimization history. QED is much less discriminating because the strongest methods rapidly approach the top of its score range, while DRD2 is the main case in which TRACE does not lead the best-so-far summary; REINVENT4 attains a slightly higher top-1 AUC.

Appendix B quantifies these trajectories using AUC summaries computed over the active portion of the budget. On top-1 AUC, which summarizes how quickly the best molecule improves, TRACE ranks first on four of the six objectives, matches SMILES-GA at the reported precision on QED, and ranks second on DRD2. Its mean top-1 AUC rank is 1.73, the best among the compared methods.

![](images/33f9b984b992cbbc5547b70cbf3102b9da6c9de8c1e84f71164f6b3a8bfcdec0.jpg)  
Figure 2: Sample-efficient molecular optimization across six PMO objectives. Mean bestso-far score over five seeds as a function of cumulative oracle evaluations, where the 1,000-call budget includes initialization. The gray region denotes the common 20-molecule initial set; active optimization then proceeds for 20 batches of $q = 4 9$ . The broken x-axis enlarges the early low-budget regime so that differences soon after initialization remain visible. Shaded bands are 95% percentilebootstrap intervals obtained by resampling whole runs. Appendix B summarizes the trajectories using top-1 and top-10 AUC.

The sample-efficiency difference is also visible directly from the trajectories: on osimertinib MPO, scaffold hopping, and decoration hopping, TRACE reaches approximately the full-budget best-so-far performance of the strongest comparator after only about half of the available oracle calls.

The conclusion is not specific to scoring each run by a single exceptional molecule. Appendix B also reports top-10 AUC, which tracks the mean score of the ten best molecules accumulated during a run and is closer to the aggregate measure emphasized in PMO. Under this metric, TRACE has the highest AUC on all six objectives and the best mean rank of 1.27. Taken together, the best-so-far trajectories and the two AUC summaries provide encouraging evidence that adapting the scale of candidate search can make more effective use of a small oracle budget (at least across these tests).

## 3.2 Adaptive region control depends on the proposal distribution

The benchmark trajectories in Figure 2 establish the sample efficiency of TRACE under the lowbudget protocol, but they do not separate the contribution of adaptive region control from that of the molecular proposal mechanism. We thus perform a crossed ablation over two proposal mechanisms and two region control settings:

$$
\{ { \mathrm { T R A C E ~ p r o p o s e r , G r a p h - G A ~ p r o p o s e r } } \} \times \{ { \mathrm { a d a p t i v e ~ r e g i o n ~ o n } } , { \mathrm { a d a p t i v e ~ r e g i o n ~ o f f } } \} .
$$

The TRACE proposer is the objective-free graph-perturbation mechanism used throughout the method, while the Graph-GA proposer provides a population-based alternative. Within each paired comparison, the underlying proposal mechanism is held fixed and only the adaptive-region controller is enabled or disabled. When the region is disabled, k is held at $k _ { 0 }$ and neither the Tanimoto-similarity threshold nor the top-k admission truncation is applied before acquisition optimization; Appendix D.1 gives the complete protocol.

For each objective and seed, all four configurations begin from the same 20 evaluated initialization molecules and then perform 100 active iterations with batch size $q = 1 0 ,$ , for 1,020 cumulative oracle evaluations. We evaluate the crossed design on osimertinib MPO and scaffold hopping using five seeds. Figure 3 shows a different effect of adaptive region control under the two proposal distributions. With the TRACE proposer, the region-enabled and region-disabled trajectories initially track one another but separate as optimization proceeds, with adaptive region control improving the mean best-so-far trajectory on both osimertinib MPO and scaffold hopping. Because the underlying proposal mechanism is unchanged within this pair, these comparisons provide evidence that the adaptive region contributes beyond the TRACE proposer itself on the two objectives studied.

![](images/f6146b84cd09d2ad1e24f34d746665948e31930208d6daa319f2580f512f6feb.jpg)  
Figure 3: Adaptive region control depends on the proposal distribution. Best-so-far trajectories for osimertinib MPO and scaffold hopping under the TRACE and Graph-GA proposers, each with and without TRACE’s adaptive region controller. All four configurations share the same 20 evaluated initialization molecules for a given objective and seed. The gray region marks this common initialization, after which active optimization proceeds for 100 iterations with $q = 1 0$ , ending at 1,020 cumulative oracle evaluations. The broken x-axis magnifies the early trajectory only and does not alter oracle-call accounting. Curves show means over five seeds and shaded bands are 95% percentile-bootstrap intervals obtained by resampling whole runs.

The corresponding Graph-GA comparisons do not show the same benefit: enabling the adaptive region yields mildly lower mean trajectories on both objectives in these experiments. Thus, the controller should not be interpreted as a generic wrapper that necessarily improves an arbitrary proposal mechanism. Instead, the results are consistent with an interaction between adaptive locality and the candidate distribution supplied by the proposer. The TRACE proposer generates structured variation around the current search center $c _ { t } .$ , after which the adaptive region controls how much of that variation is exposed to BO. The more dispersed Graph-GA proposal distribution may interact less favorably with the progressively localized admission rule. Controller success and restart diagnostics in Appendix D.1 support this interpretation, while remaining descriptive rather than establishing a general mechanism for other proposal distributions.

## 3.3 DFT-driven optimization of FePc dissolution resistance

We next evaluate TRACE in a setting where the oracle is a first-principles electronic structure calculation rather than an inexpensive benchmark function. Iron phthalocyanine (FePc) has been widely studied as a non-precious molecular electrocatalyst for the oxygen reduction reaction (ORR) [24, 25]. Its operation under acidic conditions, however, can be limited in part by degradation and demetalation of the Fe center [26, 27]. Thus, we use a DFT-based dissolution energy objective to test whether TRACE can identify FePc-derived structures with greater computed thermodynamic resistance to Fe dissolution using a small number of high-fidelity evaluations.

We optimize the Fe dissolution energy, $E _ { \mathrm { d i s s } }$ , using the thermodynamic construction of [28]:

$$
E _ { \mathrm { d i s s } } = E _ { \mathrm { w i t h o u t ~ F e } } - E _ { \mathrm { w i t h ~ F e } } + \mu ( \mathrm { F e } ^ { 2 + } ) ,\tag{4}
$$

![](images/1665b231d6c6c15d0166f47362994ffdde2c6a3785440b256a75bffdfbeb03ee.jpg)  
(a)

![](images/86e08f7b678109158d6c5985b24a46994c2d0998ca99941e911fe5544f1e25ac.jpg)  
(b)  
Figure 4: Closed-loop DFT optimization of FePc derivatives with TRACE. (a) Best-observed dissolution energy as a function of cumulative designed-candidate DFT evaluations. The initial FePc reference at $E _ { \mathrm { d i s s } } = 0 . 4 2 \ : \mathrm { e V }$ is shown at zero designed evaluations and is not included in the subsequent 80 designed-candidate evaluations. TRACE identifies a derivative with $E _ { \mathrm { d i s s } } = 2 . 5 9 \mathrm { e V }$ after those 80 evaluations, an absolute increase of 2.17 eV. (b) Two-dimensional multidimensional-scaling (MDS) visualization constructed from pairwise Morgan-fingerprint Tanimoto distances between the FePc seed and DFT-evaluated molecules. Color denotes optimization round, numbered markers indicate the centroid of each round in the two-dimensional embedding, and stars denote the initial FePc structure and the best molecule found. The projection is used only for qualitative visualization; quantitative similarity statistics in the original fingerprint space are reported in Appendix C.4.

where $E _ { \mathrm { w i t h ~ F e } }$ and $E _ { \mathrm { w i t h o u t \ F e } }$ are the calculated energies of the corresponding Fe-containing and Fe-removed structures. Under this convention, $E _ { \mathrm { d i s s } }$ is the energetic cost of removing the Fe center and transferring it to the $\mathrm { F e ^ { 2 + } }$ reservoir; a larger $E _ { \mathrm { d i s s } }$ therefore corresponds to greater computed thermodynamic resistance to Fe dissolution. The Fe-containing and Fe-removed structures are evaluated using the same staged workflow: UFF relaxation [29], UMA pre-relaxation [30], PBE geometry optimization [31], and final HSE06 single-point energies [32, 33]. Appendix C provides the complete computational details.

The initial FePc reference $( E _ { \mathrm { d i s s } } = 0 . 4 2 \ \mathrm { e V } )$ is evaluated once before closed-loop optimization. Molecular generation is restricted to FePc-derived structures retaining the central Fe-N coordination core. At each round, TRACE generates proposals around the search center $c _ { t } .$ , applies the adaptive $\left( \tau _ { t } , k _ { t } \right)$ region, and uses the fragprint Tanimoto-kernel GP with qLogEI [21] to select the DFT batch. Eight rounds with batch size $q = 1 0$ yield 80 designed-candidate DFT evaluations in addition to the initial reference.

TRACE improves $E _ { \mathrm { d i s s } }$ to approximately 0.52 eV in round 1 and 0.88 eV in round 2. The global incumbent remains near 0.9 eV through the middle of the campaign, then reaches 1.44 eV in round 7 and 2.59 eV in round 8 (Figure 4a), a total increase of 2.17 eV over the initial FePc reference.

Figure 4b provides a qualitative view of the movement through chemical space: early batches lie nearer FePc, while later rounds occupy different regions of the MDS embedding. The quantitative diagnostic in Appendix C.4 measures similarity in the original fingerprint space relative to the global incumbent $\boldsymbol { x } _ { t } ^ { \mathrm { b e s t } }$ , not the search center $c _ { t }$ . By round 8, the median similarity of evaluated molecules to the global incumbent is approximately 0.69, compared with 0.19 to the initial FePc seed. Thus, later candidates remain relatively close to the current best chemistry even as the campaign moves away from the starting molecule.

The final improvement demonstrates locality relative to TRACE’s actual search center. The best round-8 structure has Tanimoto similarity $T = 0 . 6 5$ to that round’s center $c _ { t } .$ , which has $E _ { \mathrm { d i s s } } = 1 . 4 4$ eV and is only approximately $T = 0 . 1 9$ from the FePc seed. The resulting increase to $E _ { \mathrm { d i s s } } = 2 . 5 9 \mathrm { e V }$ is a further improvement of 1.15 eV. This final transition illustrates the “local steps, distant destination”

behavior that motivates TRACE: the improved molecule is local to the active search center, even though that center has moved far from the original FePc.

## 4 Discussion and Limitations

The crossed ablation in Section 3.2 suggests that the adaptive region is most useful when considered together with the proposal mechanism on which it acts. With the TRACE proposer, turning “on” region adaptation improves performance on both objectives studied; applying the same controller to Graph-GA proposals does not. The diagnostics do not tell us exactly why, but they are consistent with the idea that changing the search region is only useful if the proposal mechanism can continue to supply appropriate candidates as that region changes. The pLogP experiment raises a related issue from another direction: what an optimizer can achieve also depends on the chemistry its generator makes accessible. These results make us hesitant to think of locality as a generic wrapper around an otherwise fixed optimizer. How candidates are generated and how the search region adapts appear to be closely connected design choices.

We therefore see the current TRACE implementation as one concrete instance of a broader adaptivesearch idea. Morgan-fingerprint Tanimoto similarity provides a simple and interpretable definition of locality, but there is nothing fundamental about that particular choice, and fingerprint similarity is only a proxy for locality of the objective. The same is true of the linear coupling between τ<sub>t</sub> and $k _ { t }$ . Other representations could define the neighborhood differently, while more flexible controllers could adapt its extent and candidate breadth separately or in a task-dependent way. There is a practical computational question here as well: broadening the candidate pool increases the work spent generating and ranking molecules. A more developed version of this idea could thus account for both oracle progress and the computational cost of searching a region when deciding how that region should change.

The FePc campaign is encouraging because this behavior carries over to a setting where the oracle is genuinely expensive. Over 80 designed-candidate DFT evaluations, TRACE moved well away from the initial FePc chemistry while improving computed resistance to Fe dissolution, and the final highvalue transition remained local to the active search center. The present study is still one computational campaign on one objective, so practical catalyst design will require a broader treatment of competing objectives and constraints together with experimental validation. Because repeated DFT campaigns would be costly and a single run per method would be difficult to interpret, we treat this study as an expensive-oracle demonstration rather than a comparison of optimizers. Continuous-domain work has already shown that trust-region BO can be extended to constrained multi-objective problems [34], providing one starting point for such an extension. Open-ended molecular design adds its own difficulty, however, because the region being adapted and the chemistry available inside it are both determined by the molecular representation and proposal process. Understanding that interaction more systematically is, in our view, an important next step.

## 5 Conclusion

We introduced TRACE, a Bayesian optimization framework that adapts the extent and breadth of molecular search as evaluations accumulate. Across six practical molecular optimization (PMO) benchmark objectives under matched 1,000-evaluation budgets, TRACE showed the strongest overall sample efficiency among the tested methods. Crossed ablations on two objectives showed that adaptive region control helped with TRACE proposals but not with Graph-GA proposals. In a closed-loop DFT campaign, TRACE identified an FePc derivative with $E _ { \mathrm { d i s s } } = 2 . 5 9 \mathrm { e } \mathrm { \bar { V } } ,$ compared with 0.42 eV for the starting molecule, after 80 designed-candidate evaluations. Together, these results support adapting chemical search scale under limited evaluation budgets.

## Acknowledgments

This work was supported by the National Science Foundation (NSF Award No. 2515046). We also thank the High Performance Computing Center at Texas Tech University for providing computational resources for the DFT calculations.

## References

[1] Yuhang Xia, Yongkang Wang, Zhiwei Wang, and Wen Zhang. A comprehensive review of molecular optimization in artificial intelligence-based drug discovery. Quantitative Biology, 12(1):15–29, 2024.

[2] Madhav R Muthyala, Farshud Sorourifar, Tianhong Tan, You Peng, and Joel A Paulson. Generative multiobjective bayesian optimization with scalable batch evaluations for sample-efficient de novo molecular design. Industrial & Engineering Chemistry Research, 65(1):628–642, 2026.

[3] Eleonora Casillo, Thomas Scattolin, and Steven P Nolan. Catalysis meets machine learning: a guide to data-driven discovery and design. Chemical Communications, 61(93):18247–18272, 2025.

[4] Jan H. Jensen. A graph-based genetic algorithm and generative model/Monte Carlo tree search for the exploration of chemical space. Chemical Science, 10(12):3567–3572, 2019.

[5] AkshatKumar Nigam, Robert Pollice, Mario Krenn, Gabriel dos Passos Gomes, and Alan Aspuru-Guzik. Beyond generative models: superfast traversal, optimization, novelty, exploration and discovery (stoned) algorithm for molecules using selfies. Chemical science, 12(20):7079– 7090, 2021.

[6] Wenhao Gao, Tianfan Fu, Jimeng Sun, and Connor W. Coley. Sample efficiency matters: A benchmark for practical molecular optimization. In Advances in Neural Information Processing Systems 35, Datasets and Benchmarks Track, pages 21342–21357, 2022.

[7] Joel A. Paulson and Calvin Tsay. Bayesian optimization as a flexible and efficient design framework for sustainable process systems. Current Opinion in Green and Sustainable Chemistry, 51:100983, 2025.

[8] Ryan-Rhys Griffiths, Leo Klarner, Henry B. Moss, et al. GAUCHE: A library for Gaussian processes in chemistry. In Advances in Neural Information Processing Systems, volume 36, 2023.

[9] Austin Tripp and José Miguel Hernández-Lobato. Diagnosing and fixing common problems in Bayesian optimization for molecule design. In ICML 2024 AIfor Science Workshop, 2024. arXiv:2406.07709.

[10] David Eriksson, Michael Pearce, Jacob Gardner, Ryan D. Turner, and Matthias Poloczek. Scalable global optimization via local Bayesian optimization. In Advances in Neural Information Processing Systems, volume 32, pages 5496–5507, 2019.

[11] Xingchen Wan, Vu Nguyen, Huong Ha, Binxin Ru, Cong Lu, and Michael A. Osborne. Think global and act local: Bayesian optimisation over high-dimensional categorical and mixed search spaces. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of PMLR, pages 10663–10674, 2021.

[12] Leonard Papenmeier, Luigi Nardi, and Matthias Poloczek. Bounce: Reliable high-dimensional Bayesian optimization for combinatorial and mixed spaces. In Advances in Neural Information Processing Systems, volume 36, pages 1764–1793, 2023.

[13] Austin Tripp, Erik Daxberger, and José Miguel Hernández-Lobato. Sample-efficient optimization in the latent space of deep generative models via weighted retraining. Advances in Neural Information Processing Systems, 33:11259–11272, 2020.

[14] Natalie Maus, Haydn T. Jones, Juston S. Moore, Matt J. Kusner, John Bradshaw, and Jacob R. Gardner. Local latent space Bayesian optimization over structured inputs. In Advances in Neural Information Processing Systems, volume 35, pages 34505–34518, 2022.

[15] Seunghun Lee, Jaewon Chu, Sihyeon Kim, Juyeon Ko, and Hyunwoo J. Kim. Advancing Bayesian optimization via learning correlated latent space. In Advances in Neural Information Processing Systems, volume 36, pages 48906–48917, 2023.

[16] Jaewon Chu, Jinyoung Park, Seunghun Lee, and Hyunwoo J. Kim. Inversion-based latent Bayesian optimization. In Advances in Neural Information Processing Systems, volume 37, pages 68258–68286, 2024.

[17] Hyeonah Kim, Minsu Kim, Sanghyeok Choi, and Jinkyoo Park. Genetic-guided gflownets for sample efficient molecular optimization. Advances in Neural Information Processing Systems, 37:42618–42648, 2024.

[18] Jeff Guo and Philippe Schwaller. Augmented memory: Sample-efficient generative molecular design with reinforcement learning. JACS Au, 4(6):2160–2172, 2024.

[19] Gerald M. Maggiora. On outliers and activity cliffs – why QSAR often disappoints. Journal of Chemical Information and Modeling, 46(4):1535–1535, 2006.

[20] David Rogers and Mathew Hahn. Extended-connectivity fingerprints. Journal of Chemical Information and Modeling, 50(5):742–754, 2010.

[21] Sebastian Ament, Samuel Daulton, David Eriksson, Maximilian Balandat, and Eytan Bakshy. Unexpected improvements to expected improvement for Bayesian optimization. Advances in Neural Information Processing Systems, 36:20577–20612, 2023.

[22] Nathan Brown, Marco Fiscato, Marwin HS Segler, and Alain C Vaucher. Guacamol: benchmarking models for de novo molecular design. Journal of chemical information and modeling, 59(3):1096–1108, 2019.

[23] Hannes H Loeffler, Jiazhen He, Alessandro Tibo, Jon Paul Janet, Alexey Voronov, Lewis H Mervin, and Ola Engkvist. Reinvent 4: Modern ai–driven generative molecule design. Journal ofCheminformatics, 16(1):20, 2024.

[24] Kejun Chen, Kang Liu, Pengda An, Huangjingwei Li, Yiyang Lin, Junhua Hu, Chuankun Jia, Junwei Fu, Hongmei Li, Hui Liu, et al. Iron phthalocyanine with coordination induced electronic localization to boost oxygen reduction reaction. Nature communications, 11(1):4173, 2020.

[25] Zhu Chen, Song Jiang, Gyeongwon Kang, Duc Nguyen, George C Schatz, and Richard P Van Duyne. Operando characterization of iron phthalocyanine deactivation during oxygen reduction reaction using electrochemical tip-enhanced raman spectroscopy. Journal of the American Chemical Society, 141(39):15684–15692, 2019.

[26] Steve Baranton, Christophe Coutanceau, Claude Roux, Françoise Hahn, and J-M Léger. Oxygen reduction reaction in acid medium at iron phthalocyanine dispersed on high surface area carbon substrate: tolerance to methanol, stability and kinetics. Journal of Electroanalytical Chemistry, 577(2):223–234, 2005.

[27] Liyang Wan, Kuangmin Zhao, Yu-Cheng Wang, Nian Wei, Pengyang Zhang, Jiayin Yuan, Zhiyou Zhou, and Shi-Gang Sun. Molecular degradation of iron phthalocyanine during the oxygen reduction reaction in acidic media. ACS Catalysis, 12(18):11097–11107, 2022.

[28] Mikael Maraschin and Joseph A. Gauthier. Unraveling fundamental activity–stability relationships in rutile oxides. The Journal ofPhysical Chemistry C, 130(10):3803–3813, 2026.

[29] Anthony K Rappé, Carla J Casewit, Kent S Colwell, William A Goddard III, and W Mason Skiff. Uff, a full periodic table force field for molecular mechanics and molecular dynamics simulations. Journal ofthe American chemical society, 114(25):10024–10035, 1992.

[30] Brandon Wood, Misko Dzamba, Xiang Fu, Meng Gao, Muhammed Shuaibi, Luis Barroso-Luque, Kareem Abdelmaqsoud, Vahe Gharakhanyan, John Kitchin, Daniel Levine, et al. Uma: A family of universal models for atoms. Advances in Neural Information Processing Systems, 38:129391–129427, 2025.

[31] John P Perdew, Kieron Burke, and Matthias Ernzerhof. Generalized gradient approximation made simple. Physical review letters, 77(18):3865, 1996.

[32] Jochen Heyd, Gustavo E Scuseria, and Matthias Ernzerhof. Hybrid functionals based on a screened coulomb potential. The Journal ofchemical physics, 118(18):8207–8215, 2003.

[33] Aliaksandr V Krukau, Oleg A Vydrov, Artur F Izmaylov, and Gustavo E Scuseria. Influence of the exchange screening parameter on the performance of screened hybrid functionals. The Journal ofchemical physics, 125(22), 2006.

[34] Samuel Daulton, David Eriksson, Maximilian Balandat, and Eytan Bakshy. Multi-objective Bayesian optimization over high-dimensional search spaces. In Proceedings ofthe Thirty-Eighth Conference on Uncertainty in Artificial Intelligence, volume 180 of Proceedings of Machine Learning Research, pages 507–517, 2022.

[35] RDKit: Open-source cheminformatics software. https://www.rdkit.org.

[36] Daniel S Levine, Muhammed Shuaibi, Evan Walter Clark Spotte-Smith, Michael G Taylor, Muhammad R Hasyim, Kyle Michel, Ilyes Batatia, Gábor Csányi, Misko Dzamba, Peter Eastman, et al. The open molecules 2025 (omol25) dataset, evaluations, and models. arXiv preprint arXiv:2505.08762, 2025.

[37] Georg Kresse and Jürgen Furthmüller. Efficient iterative schemes for ab initio total-energy calculations using a plane-wave basis set. Physical review B, 54(16):11169, 1996.

[38] JD Gouveia and J Coutinho. Can we rely on hybrid-dft energies of solid-state problems with local-dft geometries? Electronic Structure, 1(1):015008, 2019.

[39] Jens Kehlet Nørskov, Jan Rossmeisl, Ashildur Logadottir, LRKJ Lindqvist, John R Kitchin, Thomas Bligaard, and Hannes Jonsson. Origin of the overpotential for oxygen reduction at a fuel-cell cathode. The Journal ofPhysical Chemistry B, 108(46):17886–17892, 2004.

## A TRACE algorithmic details

## A.1 Outer-loop optimization

Algorithm 1 expands the main-text description into the complete TRACE state transition using the notation of Section 2. In addition, s and ℓ are the current numbers of consecutive controller successes and failures. The fresh strict local pool is denoted $\mathcal { L } _ { t } .$ , the persistent relaxed proposal reservoir by $\mathcal { G } _ { t } .$ and the conditional relaxed underfill pool by $\mathcal { F } _ { t } ;$ their deduplicated union forms the proposal set $\mathcal { P } _ { t }$ used by the admission rule.

To keep the pseudocode readable, we use four helper operations. RELAXEDPOOL $( c , p )$ runs the objective-free generator in Algorithm 2 with the relaxed proposal mode around center c and returns a finite proposal pool; $p \in \{ { \bf r e s e r v o i r , f a l l b a c k } \}$ indicates whether the result is the persistent reservoir or a temporary underfill augmentation. ADMIT $( \mathcal { P } , \mathcal { D } , c , \tau , k )$ returns the admissible set defined by Eq. (2): it removes molecules already present in $\mathcal { D } ,$ , applies the Tanimoto-similarity threshold, and retains at most the k most similar survivors. SELECTB $\mathrm { A T C H } ( \mathcal { A } , \mathcal { M } , q )$ returns q molecules from A by optimizing the chosen batch acquisition function under GP posterior $\mathcal { M }$ Finally, RESTARTCENTER(D, c) returns an evaluated molecule to use as the new center, using the stored center-level optimization history and the restart heuristic described below. The notation dedup(·) denotes removal of duplicate molecular structures.

Controller and restart semantics. The incumbent and the ordinary next search center are updated after every batch, even when the improvement is smaller than ε max(1, $| f ( x _ { t } ^ { \mathrm { b e s t } } ) | )$ and therefore counts as a controller failure. Expansion and contraction change only the adaptive state k; the corresponding similarity threshold is obtained from Eq. (3) at the next iteration. Proposal generation, admission, fallback generation, and a restart occurring before batch selection do not consume oracle evaluations.

RESTARTCENTER does not modify the global incumbent. It first deduplicates previously used iteration centers and considers them in descending order of center objective value, best improvement obtained from that center, and maximum admissible-pool size observed from that center, subject to a Morgan-Tanimoto screening rule against earlier restart centers. If no prior center qualifies, it considers all evaluated molecules in descending objective-value order under the same screening rule; the global incumbent is the final fallback. The current center is excluded from the first two search passes.

## A.2 Objective-free proposal generation

Algorithm 2 defines the generator used for the strict local pool and both relaxed pools in Algorithm 1. Its inputs are the search center c, requested target size M, proposal mode $m \in \{ { \tt s t r i c t } , { \tt r e l a x e d } \}$ graph-edit operator set ${ \mathcal { O } } ,$ and center-derived constraint set $\kappa _ { m } ( c )$ . The accepted proposal library is $\mathcal { L } ;$ x denotes the current molecule from which edits are proposed and y a newly generated candidate. All internal adaptation uses structural or proposal-history information only.

Algorithm 2 PROPOSE: objective-free molecular generation   
Require: search center c; target size M; mode $m \in$ {strict, relaxed}; graph-edit operators $\mathcal { O } ;$ center  
derived constraints ${ \kappa _ { m } ( c ) }$   
Ensure: proposal library L with $| { \mathcal { L } } | \leq M$   
1: $\mathcal { L }  \mathit { \hat { \mathbb { 0 } } } ; \mathit { \hat { x } }  c ;$ initialize objective-free proposal statistics for mode m   
2: while $| { \mathcal { L } } | < M$ and the objective-free stopping criterion is not met do   
3: $y $ EDITCHAIN(x, O, m) using the current proposal statistics   
4: if y is invalid, duplicated, or violates $\kappa _ { m } ( c )$ then   
5: record the rejected outcome and continue   
6: end if   
7: compute structural statistics for $y ;$ add $_ y$ to $\mathcal { L } ;$ update proposal/operator statistics   
8: choose the next edit seed x from c and accepted proposals using the updated objective-free proposal   
state   
9: end while   
10: return $\mathcal { L }$

Algorithm 1 TRACE: adaptive-region Bayesian optimization in chemical space   
Require: evaluated initial data $\mathcal { D } _ { 0 } ;$ total oracle budget N including $\mathcal { D } _ { 0 } ;$ nominal batch size q   
Require: controller parameters $k _ { \operatorname* { m i n } } , k _ { \operatorname* { m a x } } , k _ { 0 } ;$ coupling $\tau ( k )$ from Eq. (3); thresholds $n _ { \mathrm { s u c c } } , n _ { \mathrm { f a i l } } ;$ tolerance ε   
Ensure: global incumbent after exhausting the oracle budget   
1: $t \gets \bar { 0 ; } k _ { t } \gets k _ { 0 } ; s , \ell \gets 0 , 0$ ▷ success/failure counters   
2: $x _ { t } ^ { \mathrm { b e s t } } \gets \arg \operatorname* { m a x } _ { x \in \mathcal { X } ( \mathcal { D } _ { t } ) } f ( x ) ; c _ { t } \gets x _ { t } ^ { \mathrm { b e s t } }$   
3: $\mathcal { G } _ { t } \gets$ RELAXEDPOOL(c<sub>t</sub>, reservoir)   
4: while $| \mathcal { D } _ { t } | < N$ do   
Generate and admit candidates   
5: $q _ { t } \gets \operatorname* { m i n } ( q , N - | \mathcal { D } _ { t } | ) ; \tau _ { t } \gets \tau ( k _ { t } )$   
6: $\mathcal { L } _ { t } \gets \mathrm { P R O P O S E } ( c _ { t } , \operatorname* { m a x } ( 5 0 , k _ { t } ) ,$ , strict) ▷ Algorithm 2   
7: $\mathcal { P } _ { t }  \mathrm { d e d u p } ( \mathcal { G } _ { t } \cup \mathcal { L } _ { t } )$   
8: $\mathcal { A } _ { t } \gets \mathrm { A D M I T } \big ( \mathcal { P } _ { t } , \mathcal { D } _ { t } , c _ { t } , \tau _ { t } , k _ { t } \big )$   
9: if $| { \mathcal { A } } _ { t } | < q _ { t }$ then   
10: $\mathcal { F } _ { t } \gets \mathrm { R E L A X E D P O O L } \left( c _ { t } , \mathbf { f a l l b a c k } \right)$   
11: $\mathcal { P } _ { t } \gets \mathrm { d e d u p } ( \mathcal { P } _ { t } \cup \mathcal { F } _ { t } ) ; \mathcal { A } _ { t } \gets \mathrm { A D M I T } ( \mathcal { P } _ { t } , \mathcal { D } _ { t } , c _ { t } , \tau _ { t } , k _ { t } )$   
12: end if   
13: if $| { \mathcal { A } } _ { t } | < q _ { t }$ then   
14: $\begin{array} { r } { c _ { t } \gets \bar { \mathrm { R E S T A R T C E N T E R } } ( \mathcal { D } _ { t } , c _ { t } ) ; k _ { t } \gets k _ { 0 } ; s , \ell \gets 0 , 0 } \end{array}$   
15: $\mathcal { G } _ { t }  \mathrm { R E L A X E D P o o L } ( c _ { t }$ , reservoir); continue   
16: end if   
Select and evaluate a batch   
17: fit the fragprint Tanimoto-kernel GP to $\mathcal { D } _ { t }$ and denote its posterior by $\mathcal { M } _ { t }$   
18: $B _ { t } \gets \mathrm { S E L E C T B A T C H } ( A _ { t } , \mathcal { M } _ { t } , q _ { t } )$   
19: evaluate f on $B _ { t }$ and set $\mathcal { D } _ { t + 1 }  \mathcal { D } _ { t } \cup \{ ( x , f ( x ) ) : x \in B _ { t } \}$   
20: $x _ { t + 1 } ^ { \mathrm { b e s t } }  \arg \operatorname* { m a x } _ { x \in \mathcal { X } ( \mathcal { D } _ { t + 1 } ) } f ( x )$   
21: $c _ { t + 1 }  x _ { t + 1 } ^ { \mathrm { b e s t } } ; k _ { t + 1 }  k _ { t } ; \mathcal { G } _ { t + 1 }  \mathcal { G } _ { t }$   
Update the adaptive region   
22: if $\cdot f ( x _ { t + 1 } ^ { \mathrm { b e s t } } ) { \stackrel { . } { > } } f ( x _ { t } ^ { \mathrm { b e s t } } ) + \varepsilon \operatorname* { m a x } ( 1 , | f ( x _ { t } ^ { \mathrm { b e s t } } ) | )$ then   
23: $s \gets s + 1 ; \ell \gets 0$   
24: else   
25: $\ell \gets \ell + 1 ; s \gets 0$   
26: end if   
27: if $s \geq n _ { \mathrm { s u c c } }$ then   
28: $k _ { t + 1 } \gets \operatorname* { m i n } ( k _ { \operatorname* { m a x } } , 2 k _ { t } ) ; s \gets 0$   
29: else if $\ell \geq n _ { \mathrm { f a i l } }$ then   
30: if $k _ { t } / 2 < \operatorname* { m a x } ( k _ { \operatorname* { m i n } } , q _ { t } )$ then   
31: $\begin{array} { r } { c _ { t + 1 } \gets \mathrm { R E S T A R T C E N T E R } ( \mathcal { D } _ { t + 1 } , c _ { t + 1 } ) ; k _ { t + 1 } \gets k _ { 0 } ; s , \ell \gets 0 , 0 } \end{array}$   
32: $\mathcal { G } _ { t + 1 }  \mathrm { R E I }$ LAXEDPOOL(c<sub>t+1</sub>, reservoir)   
33: else   
34: $k _ { t + 1 } \gets k _ { t } / 2 ; \ell \gets 0$   
35: end if   
36: end if   
37: $t \gets t + 1$   
38: end while   
39: return $\boldsymbol x _ { t } ^ { \mathrm { b e s t } }$

Edits, constraints, and stopping. EDIT $\mathbf { \bar { C } } \mathbf { H A I N } ( x , \mathcal { O } , m )$ applies one or more chemistry-aware graph edits to the current edit seed x and returns a candidate molecule $y .$ The operator set includes atom and bond mutations, fragment addition and deletion, and ring fuse, bridge, spiro, open, shrink, and peel operations. Invalid and duplicate molecules are rejected. The internal locality, shell-coverage, novelty, and complexity statistics use the Morgan-fingerprint similarity $T$ defined in Section 2; they may alter operator use and edit-seed selection but never use oracle values. The objective-free stopping criterion permits the generator to return fewer than M molecules when valid or novel proposal generation saturates; its numerical thresholds and rejection limits are left to the implementation.

The mode m selects both the proposal behavior and one of two center-derived constraint profiles. The strict mode is used for the fresh local pool $\mathcal { L } _ { t }$ generated every TRACE iteration. The relaxed mode is used for the persistent reservoir $\mathcal { G } _ { t }$ generated at initialization and after restarts and for the conditional fallback pool $\mathcal { F } _ { t } .$ . Each constraint set $\phantom { } \mathcal { K } _ { m } ( c )$ is re-derived from the center used to generate that pool. For the strict local pool, TRACE requests $M = \operatorname* { m a x } ( 5 0 , k _ { t } )$ proposals. Exact constraint bounds, relaxed-pool target sizes, operator probabilities, proposal-controller constants, and other low-level generation settings are left to the implementation.

No oracle value enters Algorithm 2. Oracle information first enters the proposal-to-evaluation pipeline through the GP/acquisition stage in Algorithm 1; the outer TRACE loop also uses oracle values to update the incumbent, choose ordinary search centers, drive the success/failure controller, and inform the restart history.

## B Low-budget benchmark details and summary statistics

## B.1 Benchmark implementation

The benchmark in Section 3.1 uses a common evaluation contract across all methods. For each objective and seed, the methods begin from the same 20 evaluated molecules and then perform 20 active rounds of 49 oracle evaluations. The same PMO oracle and molecular filter $( \mathrm { S A } \leq 7 ;$ SMILES length ≤ 100) are used throughout. This common initialization is also the starting point of the trajectories in Figure 2.

For TRACE, we use the same controller settings for all six objectives: $\tau _ { \mathrm { l o } } = 0 . 1 0 , \tau _ { \mathrm { h i } } = 0 . 5 0$ $k _ { 0 } = 2 0 0 , k _ { \mathrm { m i n } } = 2 0 , k _ { \mathrm { m a x } } = 1 0 0 0 , n _ { \mathrm { s u c c } } = { \bar { 5 } } , n _ { \mathrm { f a i l } } = 3 , \mathrm { a n d } \varepsilon = 1 0 ^ { - 3 }$ . The GP and adaptive region are updated after each evaluated batch, and qLogEI is optimized over the resulting admissible set. No objective-specific tuning of these controller settings is performed.

The comparators retain the search mechanisms described in their cited references, but their evaluation schedules are placed under the same 20-round, 1,000-call contract so that differences in the trajectories reflect how the methods use a common oracle budget. This requires a benchmark-specific change for methods whose original implementations use a different feedback frequency. For example, GP-BO + Graph-GA [9] retains the GP-based BO and Graph-GA acquisition search, but is updated according to the matched batched schedule used here.

The latent-space comparator also differs in one material respect from the full LOL-BO procedure [14]. We keep the pretrained SELFIES VAE fixed throughout optimization while retaining the latent trust region, deep kernel sparse variational GP surrogate, and Thompson sampling-based selection. We therefore refer to this configuration throughout as LOL-BO (frozen VAE). For Graph-GA, SMILES-GA, STONED-SELFIES, and REINVENT4, the main text summarizes the distinct search mechanisms that motivate their inclusion. Exact software settings and implementation-level details for all comparator runs will be provided with the released code at https: $: / / \mathfrak { g } \dot { \mathbf { 1 } }$ thub.com/ PaulsonLab/TRACE-AI4MAT.

## B.2 Top-1 and top-10 AUC

Figure 2 shows the best molecule found as the oracle budget accumulates. We summarize these trajectories with AUC statistics while respecting the batched nature of the experiment. After the common initialization, observations arrive in batches of 49, giving cumulative batch endpoints

$$
n _ { j } = 2 0 + 4 9 j , \qquad j = 0 , \ldots , 2 0 ,
$$

Table 1: Top-1 AUC and final-best performance. Entries report mean top-1 AUC over five seeds under the low-budget protocol described above. Bold marks the highest mean at the reported precision. The two rank columns average each method’s rank over the 30 objective-seed combinations using top-1 AUC and the final best-so-far score, respectively; rows are ordered by mean AUC rank.
<table><tr><td>Method</td><td>QED</td><td>Median 2</td><td>Osim. MPO</td><td>Scaf. hop</td><td>Deco hop</td><td>DRD2</td><td>Rank (AUC)</td><td>Rank (final)</td></tr><tr><td>TRACE</td><td>0.940</td><td>0.154</td><td>0.732</td><td>0.479</td><td>0.603</td><td>0.870</td><td>1.73</td><td>1.77</td></tr><tr><td>SMILES-GA</td><td>0.940</td><td>0.151</td><td>0.699</td><td>0.471</td><td>0.584</td><td>0.796</td><td>2.93</td><td>2.33</td></tr><tr><td>REINVENT4</td><td>0.938</td><td>0.146</td><td>0.694</td><td>0.472</td><td>0.589</td><td>0.874</td><td>3.07</td><td>3.97</td></tr><tr><td>Graph-GA</td><td>0.933</td><td>0.139</td><td>0.699</td><td>0.451</td><td>0.580</td><td>0.848</td><td>4.33</td><td>4.20</td></tr><tr><td>STONED-SELFIES</td><td>0.939</td><td>0.120</td><td>0.679</td><td>0.450</td><td>0.566</td><td>0.676</td><td>4.97</td><td>4.53</td></tr><tr><td>GP-BO + Graph-GA</td><td>0.931</td><td>0.133</td><td>0.674</td><td>0.452</td><td>0.584</td><td>0.548</td><td>5.03</td><td>5.00</td></tr><tr><td>LOL-BO (frozen VAE)</td><td>0.926</td><td>0.117</td><td>0.671</td><td>0.443</td><td>0.565</td><td>0.526</td><td>5.93</td><td>6.20</td></tr></table>

Table 2: Top-10 AUC. Area under the mean-of-ten-best trajectory over the active 980-call interval. Entries are means over five seeds, and bold marks the highest mean at the reported precision. The final column averages the top-10 AUC rank over the 30 objective-seed combinations.
<table><tr><td>Method</td><td>QED</td><td>Median 2</td><td>Osim. MPO</td><td>Scaf. hop</td><td>Deco hop</td><td>DRD2</td><td>Rank (top-10)</td></tr><tr><td>TRACE</td><td>0.932</td><td>0.148</td><td>0.707</td><td>0.474</td><td>0.598</td><td>0.834</td><td>1.27</td></tr><tr><td>SMILES-GA</td><td>0.928</td><td>0.142</td><td>0.671</td><td>0.464</td><td>0.579</td><td>0.727</td><td>2.43</td></tr><tr><td>REINVENT4</td><td>0.906</td><td>0.129</td><td>0.638</td><td>0.449</td><td>0.576</td><td>0.637</td><td>4.00</td></tr><tr><td>STONED-SELFIES</td><td>0.926</td><td>0.111</td><td>0.651</td><td>0.442</td><td>0.562</td><td>0.635</td><td>4.20</td></tr><tr><td>Graph-GA</td><td>0.902</td><td>0.128</td><td>0.661</td><td>0.433</td><td>0.566</td><td>0.719</td><td>4.57</td></tr><tr><td>GP-BO + Graph-GA</td><td>0.896</td><td>0.122</td><td>0.628</td><td>0.432</td><td>0.571</td><td>0.344</td><td>5.33</td></tr><tr><td>LOL-BO (frozen VAE)</td><td>0.898</td><td>0.107</td><td>0.596</td><td>0.430</td><td>0.552</td><td>0.294</td><td>6.20</td></tr></table>

with $n _ { 0 } = 2 0 $ and $n _ { 2 0 } = 1 0 0 0$ . We do not impose an artificial ordering on the molecules within a batch. Instead, the relevant run-level statistic is updated when the complete batch is observed and held fixed until the next batch endpoint.

More generally, let $s _ { K } ( n )$ denote the mean score of the K best molecules observed by cumulative oracle call n, with $K = 1$ corresponding to the best-so-far trajectory and $K = 1 0$ to the mean of the ten best molecules. We compute

$$
\mathrm { A U C } _ { K } = \frac { 1 } { 9 8 0 } \int _ { 2 0 } ^ { 1 0 0 0 } s _ { K } ( n ) d n , \qquad K \in \{ 1 , 1 0 \} .
$$

The common 20-molecule initialization therefore sets the starting value of each curve but is not itself integrated; the normalization is the 980-call active interval. Because we normalize only by the number of oracle calls, the reported AUC values remain in the units of the underlying objective score.

Top-1 AUC directly summarizes the best-so-far behavior shown in Figure 2. Table 1 additionally reports the rank of each method by its final best molecule. The rank columns are computed by ranking the methods within each of the 30 objective-seed combinations (6 objectives × 5 seeds) and then averaging those ranks.

The top-1 summary mirrors the qualitative behavior in Figure 2. TRACE has the best mean AUC rank (1.73) and final-best rank (1.77), leading outright on median molecule 2, osimertinib MPO, scaffold hopping, and decoration hopping. QED is close to saturation, with TRACE and SMILES-GA equal at the reported three-decimal precision. DRD2 is the exception to the otherwise leading top-1 pattern: REINVENT4 obtains an AUC of 0.874, compared with 0.870 for TRACE.

Top-1 AUC emphasizes discovery of the single-best molecule. To test whether the same conclusion holds when performance depends on accumulating several strong candidates, we also compute top-10 AUC using the same runs and the same batch-endpoint convention. This mirrors the top-10 within-run summary emphasized by PMO [6], but here it is evaluated over our 20-1,000 low-budget interval and reported directly in the original objective-score units.

The top-10 results strengthen the aggregate picture without changing its interpretation. TRACE has the highest mean top-10 AUC on all six objectives and the best mean rank, 1.27. The distinction is particularly clear on DRD2: although REINVENT4 has the slightly higher top-1 AUC, TRACE has the higher top-10 AUC (0.834 versus 0.637). Thus, the single-best trajectory on DRD2 does not translate into the strongest ten-molecule summary. More broadly, the agreement between the top-1 and top-10 analyses indicates that the sample-efficiency gains in this six-objective study are not driven only by isolated high-scoring molecules.

## C FePc DFT case study: computational details and diagnostics

## C.1 Electronic structure workflow

Each molecule selected by TRACE is represented initially as a SMILES string and converted into a three-dimensional molecular structure using RDKit [35]. The generated geometry is first optimized using the Universal Force Field (UFF) [29]. We subsequently apply a machine-learning pre-relaxation using the Universal Model for Atoms (UMA), specifically the UMA-medium model with the omol task [30]. The omol task targets molecular electronic structure data represented in the Open Molecules 2025 (OMol25) dataset, which contains more than $1 0 ^ { 8 }$ density functional theory (DFT) calculations at the ωB97M-V/def2-TZVPD level of theory [36]. UMA is used only to improve the geometry supplied to the subsequent first-principles calculations and does not provide the objective value used by TRACE.

Following UMA pre-relaxation, the structures are optimized using the Perdew-Burke-Ernzerhof (PBE) generalized-gradient approximation [31] as implemented in the Vienna Ab initio Simulation Package (VASP) [37]. Final energies used in the dissolution calculation are obtained from singlepoint HSE06 hybrid-functional calculations [32, 33] performed on the PBE-relaxed structures. The same staged procedure is applied to the Fe-containing and corresponding Fe-removed structures entering the dissolution-energy calculation. This workflow avoids the substantially greater cost of full hybrid-functional geometry optimization for every state; the use of hybrid-functional single-point energies on semilocal-DFT-relaxed geometries is a computational approximation whose accuracy can remain system dependent [38].

## C.2 Dissolution-energy construction

We use the metal-dissolution thermodynamic construction of [28]. For each FePc derivative,

$$
E _ { \mathrm { d i s s } } = E _ { \mathrm { w i t h o u t ~ F e } } - E _ { \mathrm { w i t h ~ F e } } + \mu ( \mathrm { F e } ^ { 2 + } ) ,\tag{5}
$$

where $E _ { \mathrm { w i t h ~ F e } }$ and $E _ { \mathrm { w i t h o u t \ F e } }$ are the calculated energies of the corresponding Fe-containing and Fe-removed structures. Under this convention, $E _ { \mathrm { d i s s } }$ is the energetic cost of removing the Fe center and transferring it to the $\mathrm { F e ^ { 2 + } }$ reservoir; a larger $E _ { \mathrm { d i s s } }$ therefore corresponds to greater computed thermodynamic resistance to Fe dissolution.

Rather than directly representing a solvated $\mathrm { F e ^ { 2 + } }$ ion with DFT, its electrochemical potential is referenced to bulk metallic Fe using the $\mathrm { F e ^ { 2 + } / F e ^ { 0 } }$ redox equilibrium,

$$
\mu ( { \mathrm { F e } } ^ { 2 + } ) = \mu ( { \mathrm { F e } } ( { \mathrm { s } } ) ) - 2 e \left( U _ { \mathrm { { S H E } } } - U _ { 0 } \right) ,\tag{6}
$$

where $\mu ( \mathrm { F e ( s ) } )$ is the chemical potential of bulk metallic Fe, $U _ { \mathrm { S H E } }$ is the applied potential relative to the standard hydrogen electrode, and $U _ { 0 }$ is the equilibrium potential of the $\mathrm { F e ^ { 2 + } / F e ^ { 0 } }$ redox couple. This construction avoids direct DFT treatment of the solvated $\mathrm { F e ^ { 2 + } }$ ion by referencing its electrochemical potential to bulk Fe and is analogous in spirit to the computational hydrogen electrode treatment of electrochemical thermodynamics [39].

## C.3 TRACE settings for the DFT campaign

The FePc reference is evaluated once before optimization starts, giving the initial value $E _ { \mathrm { d i s s } } =$ 0.42 eV. TRACE then performs eight optimization rounds with batch size $q = 1 0 ,$ , giving 80 designedcandidate DFT evaluations after this initial reference evaluation. For this campaign, molecular generation is restricted to FePc-derived structures retaining the central Fe-N coordination core.

At round $t ,$ proposals are generated around the current search center $c _ { t }$ using the objective-free graph-perturbation machinery described in Section 2.3. The adaptive region is controlled by the Tanimoto-similarity threshold $\tau _ { t }$ and cardinality budget $k _ { t }$ . The fragprint Tanimoto-kernel $\mathrm { G P }$ is fitted to the accumulated evaluated data, and qLogEI [21] is optimized over the admissible set to select the molecules submitted to the DFT oracle. The resulting dissolution energies are added to the evaluated data before the next round. The DFT campaign uses the controller settings in Table 3.

Table 3: TRACE settings used in the FePc DFT campaign.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Batch size q</td><td>10</td></tr><tr><td>Number of rounds</td><td>8</td></tr><tr><td> $k _ { 0 }$ </td><td>2,000 20</td></tr><tr><td> $k _ { \mathrm { m i n } }$   $k _ { \mathrm { m a x } }$ </td><td>10,000</td></tr><tr><td> $\tau _ { \mathrm { l o } }$ </td><td>0.35</td></tr><tr><td> $\tau _ { \mathrm { h i } }$ </td><td>0.70</td></tr><tr><td> $n _ { \mathrm { s u c c } }$ </td><td> $^ 3$ </td></tr><tr><td> $n _ { \mathrm { f a i l } }$ </td><td> $2$ </td></tr><tr><td> $\varepsilon$ </td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Acquisition function</td><td> $\mathrm { \ q L o g E I { \Sigma } }$ </td></tr></table>

Locality relative to the evolving incumbent  
![](images/99ec7a404e37c79902a1840312e4aef8eb316cabc8d5a4878fbb59d1e4e85439.jpg)  
Figure 5: Locality relative to the evolving incumbent and displacement from the FePc seed. Median Tanimoto similarity of the DFT-evaluated molecules in each round to the corresponding global incumbent $\boldsymbol { x } _ { t } ^ { \mathrm { b e s t } }$ and to the original FePc seed. Similarity to the incumbent increases over the campaign, reaching approximately 0.69 in round $^ { 8 , }$ while similarity to the original FePc molecule decreases to approximately 0.19. The diagnostic therefore shows that later evaluated molecules remain relatively close to the current best chemistry while the campaign as a whole becomes substantially displaced from its starting point. Because the plotted comparison is incumbent-based, it is not interpreted as a direct measurement of similarity to the search center $c _ { t }$ when the two may differ.

## C.4 Chemical space dynamics

To quantify cumulative structural displacement during the DFT campaign, we compute Tanimoto similarity using the radius-2, 2048-bit binary Morgan fingerprints defined in Section 2. For each optimization round, Figure 5 reports the median similarity of the DFT-evaluated molecules to both (i) the global incumbent $\mathbf { \bar { \Phi } } _ { x _ { t } ^ { \mathrm { b e s t } } } ^ { \mathrm { b e s t } }$ used in the round-wise diagnostic and (ii) the original FePc seed.

The two trends separate clearly as optimization proceeds. The median similarity to the initial FePc seed decreases from approximately 0.59 in round 1 to approximately 0.19 in round 8, whereas the median similarity to the round-specific global incumbent rises to approximately 0.69. This incumbent-based diagnostic supports the cumulative-displacement aspect of the TRACE trajectory without requiring the two-dimensional MDS projection in Figure 4b to be interpreted quantitatively.

![](images/de2d98085feb16b86998c7a3727c7ff58460333f41e998ee313ac9719f77319c.jpg)  
Figure 6: Final improvement around the evolved TRACE search center. The FePc seed $x _ { 0 }$ the round-8 search center $c _ { 8 }$ (the round-7 incumbent), and the best molecule evaluated in round 8 are shown with their computed dissolution energies. By the start of round 8, the search center has moved substantially from the initial FePc structure, with $T ( c _ { 8 } , x _ { 0 } ) \approx 0 . 1 9$ . Round 8 is nevertheless conducted relative to this evolved center, and the best evaluated candidate has $T ( x _ { 8 } ^ { \star } , c _ { 8 } ) = 0 . 6 5$ increasing $E _ { \mathrm { d i s s } }$ from 1.44 to 2.59 eV. The two similarities therefore describe distinct quantities: cumulative displacement from the initial chemistry and locality relative to the search center used for the final round.

## C.5 Final transition around the evolved search center

Figure 6 isolates the final transition in the closed-loop campaign and distinguishes locality with respect to the current TRACE search center from similarity to the initial molecule. At the start of round $^ { 8 , }$ the search center $c _ { 8 }$ is the round-7 incumbent, with $E _ { \mathrm { d i s s } } = 1$ .44 eV. Although this center has moved substantially from the initial FePc seed $( T ( c _ { 8 } , x _ { 0 } ) \approx 0 . 1 9 )$ , round-8 proposals are generated and filtered relative to $c _ { 8 } .$ , rather than relative to x .

The best molecule evaluated in round 8 has $T ( x _ { 8 } ^ { \star } , c _ { 8 } ) = 0 . 6 5$ and $E _ { \mathrm { d i s s } } = 2 . 5 9$ eV, corresponding to a 1.15 eV increase over the preceding incumbent. Thus, the final improvement does not require a direct jump from the original FePc structure: it is obtained from the neighborhood of an evolved search center that has itself become distant from the starting chemistry. This example illustrates how repeated re-centering can accumulate substantial global displacement while exploration remains referenced to the current center.

## C.6 Round-wise molecular evolution

For completeness, Figure 7 shows the highest- $E _ { \mathrm { d i s s } }$ molecule identified in each optimization round. The gallery provides a qualitative chemical audit of the trajectory and illustrates the increasing displacement from the initial FePc structure. The limited number of evaluated structures does not support attribution of the dissolution-energy improvements to individual substituents or a causal structure-property relationship.

## D Additional ablations and diagnostics

## D.1 Proposal mechanism × adaptive region control

Section 3.2 compares two proposal mechanisms with and without the coupled TRACE region controller. The crossed design contains four configurations:

1. TRACE proposer + adaptive TRACE region,

## Highest-E<sub>diss</sub> structure in each round

![](images/4a1d261b072bc3b15d93244ff17325d3ed740ff4e5700eb68b838e32a391989a.jpg)  
Figure 7: Highest- $E _ { \mathrm { d i s s } }$ DFT-evaluated FePc-derived structure in each TRACE round. Each panel shows the highest- $. E _ { \mathrm { d i s s } }$ molecule observed during that round together with its dissolution energy and Tanimoto similarity to the original FePc structure. The sequence illustrates the cumulative structural evolution of the search but is not used to infer a causal structure-property relationship or catalytic-performance improvement.

2. TRACE proposer without adaptive region control,

3. Graph-GA proposer + adaptive TRACE region,

4. Graph-GA proposer without adaptive region control.

For each objective and seed, the four configurations use the same 20 evaluated initialization molecules. Active optimization then consists of 100 iterations with batch size $q = 1 0$ , giving $2 0 + 1 0 0 \times 1 0 =$ 1,020 cumulative oracle evaluations. We use five seeds on osimertinib MPO and scaffold hopping. Apart from the different batch size and number of active iterations, the region-enabled runs use the same controller settings as the primary benchmark in Section 3.1 (see Appendix B.1). The molecular constraints, GP surrogate, and qLogEI acquisition procedure are also held fixed across the four configurations.

In a region-enabled configuration, the corresponding proposer supplies the finite proposal set $\mathcal { P } _ { t }$ , and admission follows Section 2. In particular,

$$
S _ { t } = \left\{ x \in \mathcal { P } _ { t } \backslash \mathcal { X } ( \mathcal { D } _ { t } ) : T ( x , c _ { t } ) \geq \tau _ { t } \right\} , \qquad | \mathcal { A } _ { t } | = \operatorname* { m i n } ( k _ { t } , | \mathcal { S } _ { t } | ) ,
$$

![](images/cf83c817882613b5f0902daa6632abe8ac8f2b4e73c89d99b0356247c4fa4fb5.jpg)

![](images/74747fe2b391c4d55f629d158a10e9dc901cd7ac50f2a72069c12c8bb9845e0a.jpg)

![](images/f17081e044dae898fbd86c92dcb7daa098a883d06a67c66525ae777394573e6e.jpg)  
Figure 8: Region controller diagnostics under the two proposal mechanisms. Both configurations shown here enable adaptive $\left( \tau _ { t } , k _ { t } \right)$ control and differ in the proposal mechanism supplying $\mathcal { P } _ { t }$ . An iteration is counted as a controller success when $f ( x _ { t + 1 } ^ { \mathrm { b e s t } } ) > f ( x _ { t } ^ { \mathrm { b e s t } } )$ + ε max(1, $\left| f ( x _ { t } ^ { \mathrm { b e s t } } ) \right| )$ with $\varepsilon = 1 0 ^ { - 3 }$ , matching the success criterion that drives region adaptation. The first half comprises active iterations 1-50, corresponding to the first 500 active evaluations and ending at cumulative oracle call 520; the second half comprises active iterations 51-100, corresponding to the remaining 500 active evaluations and ending at cumulative call 1,020. A restart is counted whenever TRACE executes either restart pathway described in Section $2 \colon$ failure-driven contraction below the feasible batch floor or persistent candidate underfill after relaxed fallback generation. $\mathbf { A }$ restart resets $k$ to $k _ { 0 }$ and may change the search center $c _ { t }$ without discarding the global incumbent $\boldsymbol { x } _ { t } ^ { \mathrm { b e s t } }$ . Bars are means over five seeds and error bars are 95% percentile-bootstrap intervals.

Table 4: Region controller diagnostics under the two proposal mechanisms. Success rates and restart counts are measured in the two region-enabled configurations. Values are means over five seeds with descriptive 95% percentile-bootstrap intervals in brackets.
<table><tr><td rowspan="2">Objective</td><td rowspan="2">Proposer</td><td colspan="2">Region success rate (%)</td><td colspan="2">Restarts</td></tr><tr><td></td><td>First half</td><td>Second half</td><td>per run</td></tr><tr><td rowspan="2">Osimertinib MPO</td><td>TRACE</td><td>28.0 [22.0, 33.6]</td><td></td><td>8.8 [3.2, 16.0]</td><td>5.6 [5.2, 6.0]</td></tr><tr><td>Graph-GA</td><td>21.6 [14.8, 28.4]</td><td></td><td>4.0 [1.6, 6.4]</td><td>6.2 [5.6, 6.8]</td></tr><tr><td rowspan="2">Scaffold hopping</td><td>TRACE</td><td></td><td>39.6 [32.0, 46.0]</td><td>9.6 [4.8, 14.4]</td><td>4.6 [4.2, 5.0]</td></tr><tr><td>Graph-GA</td><td>40.0 [27.6, 51.6]</td><td></td><td>1.6 [0.0, 4.0]</td><td>6.8 [5.0, 9.6]</td></tr></table>

with $\boldsymbol { A } _ { t }$ containing the $k _ { t }$ molecules in $S _ { t }$ most similar to the current search center $c _ { t }$ whenever $\vert S _ { t } \vert > k _ { t }$ . The cardinality budget $k _ { t }$ adapts from the controller success/failure signal and $\tau _ { t }$ follows from the coupling in Eq. (3). For the TRACE proposer, the requested strict local proposal-pool size also follows the method definition, max $( 5 0 , k _ { t } )$ .

In the region-disabled configurations, the corresponding proposal mechanism itself is unchanged, but the coupled adaptive-region control is removed: k is held at $k _ { 0 } .$ , no Tanimoto-similarity threshold is imposed, and no top-k admission truncation is applied before the surrogate/acquisition stage. For the TRACE proposer, fixing $k = k _ { 0 }$ also fixes the requested strict local proposal-pool size at max $( 5 0 , k _ { 0 } )$ rather than adapting it during the run. Other proposer-specific operations, molecular constraints, surrogate fitting, acquisition, and batch evaluation are unchanged. Thus, holding the proposer fixed refers to using the same proposal mechanism within each paired comparison; the candidate sets need not be identical because adapting candidate-search breadth is itself part of the region intervention.

The trajectories in Figure 3 give the primary ablation result. With the TRACE proposer, adaptive region control improves the mean best-so-far trajectory on both objectives. The same intervention does not provide a corresponding benefit with Graph-GA proposals and produces mildly lower mean trajectories in the two cases studied. We next examine the internal controller behavior in the two region-enabled configurations to understand whether their optimization histories exhibit a corresponding difference.

Figure 8 and Table 4 show how the controller behaves when paired with the two proposal mechanisms. During the first 50 active iterations, there is no clear separation at the resolution of these five-seed summaries. On osimertinib MPO, the mean success rates are 28.0% for the TRACE proposer and 21.6% for Graph-GA; on scaffold hopping they are 39.6% and 40.0%, respectively. The corresponding bootstrap intervals overlap in both cases.

The pattern differs in the second half of optimization, when improvements are less frequent for both proposers. The TRACE proposer retains mean controller-success rates of 8.8% on osimertinib MPO and 9.6% on scaffold hopping, compared with 4.0% and 1.6% for Graph-GA. Graph-GA also produces more restarts on average on both objectives, with the larger difference on scaffold hopping (6.8 versus 4.6 restarts per run). Given only five seeds, these bootstrap summaries are best interpreted descriptively rather than as formal tests of pairwise differences.

Together, the later-run success rate and restart patterns are consistent with the interpretation in Section 3.2: the same adaptive controller interacts differently with the candidate distributions supplied by the two proposers. The TRACE proposer supplies structural variation around the current search center $c _ { t } ,$ while the controller determines how much of that variation is admitted to BO through $\tau _ { t }$ and $k _ { t } .$ . A more dispersed proposal distribution can interact differently with this localization, which is consistent with the lower late-stage success rates and more frequent restarts observed for the Graph-GA pairing here. These diagnostics do not directly measure candidate density within the admissible region or establish a general causal mechanism, so we restrict the conclusion to the two objectives and five seeds studied.

## D.2 Batch size sensitivity on scaffold hopping

TRACE updates its adaptive search region after each evaluated batch, so changing the batch size q affects both the number of oracle evaluations between adaptive updates and the minimum region size needed to supply a complete batch, i.e., max $( k _ { \operatorname* { m i n } } , q )$ . We examine sensitivity to these effects on the scaffold hopping objective using $q \in \{ 5 , 1 0 , 2 5 , 4 0 \}$ . For each seed, all methods start from the same 20 evaluated molecules and then receive exactly 1,000 active unique-molecule oracle evaluations. This corresponds to 200, 100, 40, and 25 complete active batches for $q = 5 , 1 0 , 2 5 , 4 0$ , respectively, giving 1,020 cumulative oracle evaluations including initialization in every case. We use five seeds and compare TRACE with Graph-GA, SMILES-GA, GP-BO + Graph-GA, and STONED-SELFIES at each batch size. Figure 9 reports the median final best objective value across runs together with bootstrap 95% intervals.

Batch size, scaffold hop (median, bootstrap 95% CI)  
![](images/29d370e03d1b91db673043e878b2db8317687b0bf35ce6ca93baf5366a55ab1e.jpg)  
Figure 9: Batch size sensitivity on scaffold hopping. Final best score after 20 matched initialization evaluations followed by 1,000 active oracle evaluations (1,020 cumulative evaluations) for batch sizes $q \in \{ 5 , 1 0 , 2 5 , 4 0 \}$ . Bars denote the median across five runs and error bars denote bootstrap 95% intervals. TRACE has the highest observed median at each tested batch size. The modest variation across $q$ and substantial overlap of the uncertainty intervals indicate that the experiment mainly supports robustness across the tested settings, while differences among individual batch sizes are difficult to distinguish with five runs.

TRACE’s median final score varies only modestly across the sweep, from approximately 0.52 to 0.54, and is the highest observed median among the evaluated methods at each tested value of $q \ ( \mathrm { F i g u r e } \ 9 )$ There is no clear monotone relationship between batch size and terminal performance. The median decreases slightly at $q = 2 5$ and increases again at $q = 4 0$ , with overlapping bootstrap intervals across the TRACE settings. Overall, performance remains similar as the batch size varies from five to forty despite the corresponding changes in adaptive update frequency and feasible batch requirement.

Given that this sweep covers one objective and five runs per setting, its main value is as a robustness check for scaffold hopping over the tested range of q. Determining whether a particular batch size is generally preferable would require a broader study across objectives and additional runs.

## D.3 Penalized logP as a diagnostic objective

We also evaluate penalized logP (pLogP), a long-standing molecular optimization benchmark that combines the octanol-water partition coefficient with synthetic-accessibility and large ring penalties [2]. The objective is known to favor increases in molecular size and related compositional changes, so high scores need not reflect broader molecular design quality [6]. To limit this size-seeking behavior, we impose the same 60-heavy-atom maximum on every method. We otherwise follow the primary benchmark schedule, using the same 20 evaluated initialization molecules for each seed followed by 20 active iterations with batch size $q = 4 9$ . Initialization is included in the budget, giving exactly 1,000 unique-molecule oracle evaluations per run, and we use five seeds.

Figure 10 shows a clear difference in the scores reached by the two leading methods. LOL-BO (frozen VAE) attains the highest observed pLogP performance, with a mean final best score of approximately 18.23, compared with 14.84 for TRACE. Inspection of the saved highest-scoring molecules shows extended hydrophobic chains among the LOL-BO (frozen VAE) solutions and, in some runs, unusual phosphorus-rich motifs. The final-best molecules from both methods also lie close to the imposed size limit, with mean heavy-atom counts of approximately 59.2 for LOL-BO (frozen VAE) and 59.8 for TRACE. The same tendency is visible across the full evaluated sets, where pLogP is positively associated with heavy-atom count for both methods (Spearman $\rho \approx 0 . 6 3$ for LOL-BO (frozen VAE) and $\rho \approx 0 . 8 4$ for TRACE).

![](images/9645bee6729779966c8cf8f43e8ae53e54239d1b5c6ee798b4f0d115fcff746b.jpg)  
Figure 10: Penalized logP diagnostic. Mean best-so-far pLogP score against the number of cumulative unique oracle evaluations under the matched 1,000-call budget. The common 20-molecule initialization is included in the total budget and shown by the gray region, after which optimization proceeds for 20 batches of q = 49. The broken x-axis visually expands the early low-budget regime. Shaded regions denote 95% percentile-bootstrap intervals over whole runs across five seeds.

Taken together, these diagnostics show that both methods exploit a size-dependent direction in pLogP, even with the 60-heavy-atom cap, while LOL-BO (frozen VAE) reaches higher scores under its search space. We do not interpret this difference as a general advantage of latent-space BO or a general limitation of TRACE. Rather, pLogP is useful here as a diagnostic because it makes visible how an objective with an easily exploited direction can interact with the chemistry accessible to a molecular generator. We therefore report it separately from the six-objective headline aggregate rather than use it as a measure of general molecular design performance.