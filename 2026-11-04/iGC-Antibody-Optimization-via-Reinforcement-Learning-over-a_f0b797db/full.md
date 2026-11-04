# iGC: Antibody Optimization via Reinforcement Learning over a Learned Somatic Hypermutation Prior

Thao Nguyen<sup>1</sup>\*, Robert G. Alberstein<sup>2</sup>, Ewa Nowara<sup>3</sup>, Homa MohammadiPeyhani<sup>3</sup>, Holly R. Steach<sup>2</sup>, Sarah A. Robinson<sup>2</sup>, Yuval Shechter<sup>2</sup> Donald W. Lee<sup>2</sup>, Jason A. Vander Heiden<sup>2</sup>, Franziska Seeger<sup>2</sup>, Andrew Leaver-Fay<sup>2</sup>, Richard Bonneau<sup>2</sup>, Joshua Yao-Yu Lin<sup>2</sup>

<sup>1</sup>University of Illinois at Urbana-Champaign <sup>2</sup>Genentech <sup>3</sup>F. Hoffmann-La Roche Ltd.

November 4, 2026

## Abstract

Antibody affinity maturation is a natural optimization loop. Individual B cells propose mutations through somatic hypermutation (SHM), variants with higher affinity for antigen are positively selected, and the process repeats cyclically, giving rise to a population of high-affinity antibody producing B cells. A computational model of this process should therefore capture both how mutations arise and how they are selected for antigen-specific affinity. However, modeling these two components jointly is challenging: mutation models capture the sequence patterns of SHM but do not directly optimize antigen binding, while affinity optimization can improve binding without accounting for biologically plausible mutation trajectories. We introduce iGC (in silico Germinal Centers), a two-stage reinforcement learning (RL) framework for antigen-specific antibody optimization. Stage 1 learns an SHM mutation prior from parent–child pairs in B-cell lineage trees and domain-adapts it to the target affinity-maturation dataset. Stage 2 initializes an RL policy from this adapted prior and optimizes antigen binding while remaining Kullback–Leibler (KL) regularized toward a frozen copy of the prior. Direct optimization of affinity rewards can lead to degenerate solutions, including excessive use of a small number of highly rewarding substitutions or convergence toward structurally tolerated framework mutations. To mitigate these effects, we introduce a Reward–Tax Meta-Gradient objective. RewardNet smooths the affinity landscape, while TaxNet discourages reward overexploitation; their difference defines a learned meta-reward for policy optimization. We evaluate whether iGC achieves both objectives: preserving SHM-consistent mutation behavior and improving antigen-specific binding. The mutation prior more than doubles the strongest baseline on mutation-site prediction. Following RL optimization, iGC preserves strong SHM sequence structure, with 57.0% of predicted mutation positions falling within WRC hotspots (canonical DNA motifs preferentially targeted during SHM). iGC achieves a strong exploration–exploitation balance, yielding competitive reward–mutation-diversity products across candidate-selection cutoffs. These results demonstrate that iGC can jointly model biologically plausible mutation generation and antigen-specific affinity selection, providing an in silico framework for iterative antibody affinity maturation.

## 1 Introduction

The immune system naturally optimizes antibody sequence repertoires through the process of affinity maturation in specialized anatomical structures called germinal centers [Victora and Nussenzweig, 2012]. In B cells, somatic hypermutation (SHM) of the antibody-encoding Ig locus introduces mutations concentrated in Complementarity-Determining Regions (CDRs), specifically diversifying antigen binding. Competition for antigen drives selective expansion of high-affinity binders while eliminating sequences that compromise protein structure or exhibit autoimmune cross-reactivity. This can be formally represented as a sequential decision process: a unique antibody sequence represents a state, binding improvement supplies the reward, and each selected mutation alters the epistatic context for subsequent decisions. Iterative cycles of diversification and selection thus function as a natural optimization loop, yielding B cells that express potent, physiologically optimized antibodies (Figure 1a).

![](images/d97353dd14adacd70d69f58c0af452aa099f0dabc9714b4d83905fcc184022e8.jpg)  
Figure 1: (a) Affinity maturation in germinal centers as a repeated mutation–selection process. (b) Structure of an antibody and the heavy-chain variable region (VH).

Computational antibody design typically models only one half of this loop. B-cell-receptor repertoire and antibody language models learn statistical patterns of SHM and natural sequence evolution but do not directly optimize binding to a specified antigen [Ruffolo et al., 2021, Olsen et al., 2024]. Conversely, antigen-specific affinity-optimization methods search directly for stronger binders but do not explicitly model the sequential, biologically regularized mutation trajectories that produce them [Shan et al., 2022, Luo et al., 2022, Tagasovska et al., 2025]. Such optimization can move toward sequences that score well for binding yet lie far from mutation paths naturally accessible to the body, bypassing physiological constraints that preserve structural integrity and prevent off-target polyreactivity. Existing approaches also generally optimize individual variants rather than learning how to make successive, context-dependent mutation decisions across the full mutation–selection process.

We introduce iGC (in silico Germinal Centers), which learns to propose SHM-consistent mutations and select those that improve binding to a specific antigen. iGC separates these roles through two-stage learning. Stage 1 first trains a Transformer on parent–child pairs from B-cell lineage trees, producing a mutation prior that captures where SHM acts and which substitutions it favors, and then domain-adapts this prior to the target affinity-maturation dataset. Stage 2 initializes a policy from the adapted prior and optimizes antigen-specific binding with REINFORCE [Williams 1992]. A frozen copy of the adapted SHM prior remains a KL-regularization anchor throughout Stage 2, allowing affinity-driven optimization without unconstrained drift from repertoire-plausible mutation behavior.

To address this problem, we introduce a Reward–Tax Meta-Gradient (RTMG) objective. Instead of training the policy directly on raw deep mutational scanning (DMS) rewards, RTMG learns a meta-reward equal to a refined benefit from RewardNet minus a refined exploitation penalty from TaxNet. RewardNet smooths the uneven affinity landscape while preserving agreement with measured binding. TaxNet penalizes recurrent or SHM-accessible sequence outcomes that the optimization can repeatedly exploit. Both networks are learned through their effect on a differentiable probe-policy update, rather than from manually specified targets. The deployed policy is then trained on the detached meta-reward while remaining KL-regularized toward the SHM prior.

The resulting policy improves affinity without discarding SHM structure. After RL, 57.0% of iGC-predicted mutation positions lie in WRC hotspots, compared with 2.0% under random background, and predicted mutability remains correlated with S5F, a statistical model of SHM mutability [Yaari et al., 2013] (Pearson 0.474). At the same time, iGC generates high-affinity candidates while maintaining mutation diversity, yielding strong reward–mutation-diversity products across candidate-selection cutoffs. To test generalization, we hold out entire germinal-center lineage trees and ask whether iGC can rediscover positive-reward sequences observed only in this unseen test set. Across mutation budgets of two to four mutations, iGC exactly rediscovers 110 of 799 such sequences. Together, these results show that iGC learns diverse, SHM-like mutations that improve antigen-specific binding and recover beneficial sequences from unseen lineages.

In summary, our contributions are:

• A sequential formulation of antibody optimization. We model germinal-center affinity maturation as repeated

SHM mutation proposals followed by antigen-specific affinity selection.

• Biologically constrained reinforcement learning. A learned SHM prior initializes and KL-regularizes antigenspecific policy optimization, coupling realistic mutation proposals with binding-driven selection.

• Reward–tax meta-gradient optimization. RewardNet smooths the affinity signal and TaxNet penalizes repeatedly exploitable sequence outcomes, yielding stronger reward–diversity trade-offs while preserving SHM hallmarks.

## 2 Background

Antibody Structure and Antigen Recognition. Antibodies are Y-shaped proteins made by B lymphocytes that function both as cell-surface receptors and as secreted effector molecules (Figure 1b). A single antibody consists of two heavy chains and two light chains, each containing variable and constant domains. Antigen recognition is mediated by the variable domains located at the tips of the antibody, whereas the constant domains are responsible for downstream immune effector functions. Within each heavy- and light-chain variable domain, four structural framework regions (FR1 to FR4) provide structural support, while three complementarity-determining regions (CDR1 to CDR3) form the antigen-binding interface [Chothia and Lesk, 1987]. Together, the heavy and light chains contribute six CDRs (CDR-H1 to CDR-H3 and CDR-L1 to CDR-L3) that collectively determine binding specificity and affinity. Among these, CDR-H3 is the most sequence-diverse region and frequently makes the largest contribution to antigen recognition [Chothia et al., 1989]. In this study, we focus exclusively on the heavy-chain variable region (VH), as heavy-chain sequences are more widely available in repertoire datasets and often capture a substantial fraction of antigen-binding determinants, particularly through CDR-H3.

Somatic Hypermutation To balance a vast, continuously evolving antigen landscape against physiological constraints, the immune system employs multi-stage antibody adaptation. Initially, stochastic V(D)J recombination generates unique sequences that provide broad, low-affinity coverage against unseen targets, while autoreactive variants are filtered out. Because the immune system cannot pre-optimize for unpredictable targets, these interactions remain sub-optimal. Consequently, repertoire fine-tuning requires ad hoc exploration during an active immune response, driven by somatic hypermutation (SHM) during germinal center affinity maturation [Mishra and Mariuzza, 2018]. Given a naive sequence, SHM introduces targeted point mutations into the variable region, and positive selection subsequently expands high-affinity binders. By elevating mutation rates a million-fold, SHM enables a Darwinian evolutionary loop that optimizes antibody binding over days to weeks [Pilzecker and Jacobs, 2019], yielding high-affinity populations that mediate lasting immunity.

## 3 Related Work

Learning Somatic Hypermutation. Because SHM follows reproducible sequence-dependent patterns rather than mutating uniformly, many methods aim to learn its mutational grammar from repertoire data [Yaari et al., 2013, Tang et al., 2022, Sung et al., 2025, Lu et al., 2026]. They differ in the mutational context captured and whether selection is modeled. DeepSHM applies a CNN over local sequence windows to predict mutation probabilities, capturing hotspot motifs but with no global view of the sequence [Tang et al., 2022]. Thrifty widens the context by convolving over 5-mer embeddings while avoiding the parameter blow-up of explicit k-mer models [Sung et al., 2025]. DASM separates mutation from selection explicitly, pairing a fixed SHM model with a learned amino-acid selection model [Matsen IV et al., 2026]; its selection signal, however, is inferred from observed antibody evolution rather than measured binding, so it proxies functional fitness rather than affinity directly. Architecturally, these approaches rely on local sequence contexts or predefined mutation models. In contrast, iGC Stage 1 formulates SHM as a full-sequence mutation-prediction problem. Given a parent-child pair (PCP), a Transformer encoder processes the entire VH sequence and jointly predicts both the mutation position and the resulting amino-acid substitution through dedicated position and amino-acid prediction heads. By conditioning on the complete antibody context, iGC can capture long-range dependencies that are inaccessible to local motif-based approaches.

Affinity-Guided Antibody Optimization. While SHM governs how diversity is generated, affinity maturation ultimately selects variants that bind better. This has driven a large body of affinity-aware design work, including supervised affinity predictors [Shan et al., 2022], active-learning loops [Gessner et al., 2024], language-model sampling [Hie et al., 2024], and diffusion- and flow-based generation [Luo et al., 2022, Yu et al., 2026]. Benchmarks such as EPAM [Johnson et al., 2025] and AbBiBench [Zhao et al., 2025] have highlighted complementary challenges for computational affinity maturation. EPAM shows that incorporating SHM-aware evolutionary information substantially improves affinity-maturation prediction, whereas AbBiBench emphasizes accurate evaluation of antibody binding in the context of the target antigen. Taken together, they underscore the importance of modeling both biologically realistic mutation processes and affinity-driven selection. Existing methods, however, typically emphasize one objective at the expense of the other.

Among existing approaches, CoSiNE and SHMReplay are conceptually the closest to iGC, but neither directly trains its mutation generator to optimize antigen binding. CoSiNE learns antibody evolution as a neural continuous-time Markov process; its Guided Gillespie procedure then queries an external binding oracle during sampling to favor higher scoring transitions [Lu et al., 2026]. SHMReplay similarly learns to replay observed affinity-maturation trajectories and uses affinity scores from the GCReplay landscape to select or rank generated sequences [DeWitt et al., 2026]. Thus, their underlying mutation proposals remain binding-blind, with antigen specificity introduced through external oracle-guided selection rather than learned within the mutation policy itself. Unlike these approaches, iGC explicitly formalizes affinity maturation as a reinforcement-learning process, drawing a direct parallel to the biological engine: SHM-driven diversification plays the role of exploration, while affinity-based selection drives policy improvement. Its central optimization mechanism is a reward–tax meta-gradient objective rather than a hand-designed adaptive shaping rule: RewardNet learns how to transform measured affinity signals, TaxNet learns a sequence-dependent penalty from evolutionary prevalence and SHM accessibility, and both are trained through their effect on a differentiable virtual policy update.

## 4 The iGC Framework

iGC models affinity maturation as a policy over single-mutation actions. Given a parent VH sequence, the policy chooses an action $( i , a )$ , where i is an aligned residue position and a is the replacement amino acid. Stage 1 learns this action distribution from unlabeled parent-child pairs and domain-adapts it to the target biological context. Stage 2 then fine-tunes the adapted mutation prior under affinity-derived rewards while a KL penalty keeps proposals biologically plausible.

## 4.1 Stage 1: Learning and Adapting SHM Grammar

Stage 1 performs behavior cloning from observed mutation events, estimating a conditional prior $p ( i , a \mid \mathbf { s } )$ over one-step mutation actions. We represent each observed SHM event as a Parent-Child Pair (PCP): a parent antibody sequence, the mutated position under the Aho numbering scheme, and the resulting amino acid identity. Aho numbering is a standard antibody alignment convention that maps homologous residues from antibodies of different lengths onto shared position indices, making mutation positions comparable across sequences [Honegger and PluÈckthun, 2001]. Each observed mutation contributes one PCP training example, allowing the model to learn from independent mutation events without requiring full lineage trajectories, fitness measurements, or affinity labels. This makes SHM data abundant and broadly available, allowing the model to learn mutation grammar purely from the natural distribution of SHM events.

Given a parent sequence s $\in \mathcal { A } ^ { L }$ where A is the set of 20 standard amino acids and L is the sequence length, a Transformer encoder processes s to produce contextual residue representations. The model learns the joint conditional distribution: $p ( i , a \mid \mathbf { s } ) = p ( i \mid \mathbf { s } ) \cdot p ( a \mid i , \mathbf { s } )$ , where $i \in \{ 1 , \ldots , L \}$ is the mutation site and $a \in { \mathcal { A } }$ is the replacement amino acid. This factorization reflects the biological structure of SHM: site selection and substitution identity are coupled, as local sequence context around position i influences which amino acid is favored. During training, teacher forcing is applied: the ground-truth position $i ^ { ( n ) }$ is used to condition the amino acid head, and the model is trained by minimizing:

$$
\mathcal { L } = - \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \left[ \log p ( i ^ { ( n ) } \mid \mathbf { s } ^ { ( n ) } ) + \log p ( a ^ { ( n ) } \mid i ^ { ( n ) } , \mathbf { s } ^ { ( n ) } ) \right]\tag{1}
$$

over a dataset of N PCPs $\{ ( \mathbf { s } ^ { ( n ) } , i ^ { ( n ) } , a ^ { ( n ) } ) \} _ { n = 1 } ^ { N }$ . Stage 1 then domain-adapts this model with behavior cloning on parent–child pairs from the target affinity-maturation dataset. The resulting Adapted SHM Prior serves as the

![](images/515591d119bc3bcd1880475b5fc3bba602bd9f95ea59a5d145d6daef15819e00.jpg)  
Figure 2: Overview of the two-stage iGC framework. Stage 1 learns and domain-adapts an SHM mutation prior, and Stage 2 optimizes the resulting policy for antigen binding while remaining regularized toward that prior. At inference, the learned policy optimizes an input antibody by rolling out successive mutations under a fixed mutation budget.

initialization and frozen KL reference for Stage 2, preserving the learned mutation grammar while correcting datasetspecific sequence and species mismatch.

## 4.2 Stage 2: Affinity-Guided Clonal Selection

Stage 2 turns the Adapted SHM Prior produced by Stage 1 into a policy that favors affinity-improving mutations while remaining close to realistic SHM behavior. Training is offline: the policy learns from logged mutation outcomes rather than interacting with a live experimental system. At evaluation time, the learned policy is rolled out in silico to generate new antibody variants under a fixed mutation budget (Figure 2).

Affinity Maturation Data. Affinity supervision is provided by clonal trajectory data, which records ordered mutation paths as sequences of parent, action, child, and affinity-change tuples. Each trajectory defines a sequential decisionmaking problem: the current antibody sequence is the state, a mutation is the action, and antigen-binding affinity provides the reward. The RL policy learns from these trajectories to maximize long-term reward, corresponding to the binding affinity of the final sequence produced after multiple mutation decisions. Our main experiments use GCReplay [DeWitt et al., 2026]. Dataset construction and implementation details are given in Appendices C and E, respectively.

Reinforcement Learning Formulation. For a trajectory τ, the state ${ \mathbf s } _ { \tau , t }$ denotes the antibody sequence after t mutations, and the action $u _ { \tau , t } ~ = ~ ( i _ { \tau , t } , a _ { \tau , t } )$ specifies a mutation site $i _ { \tau , t }$ and its replacement amino acid $\boldsymbol { a } _ { \tau , t }$ producing the next sequence $\mathbf { s } _ { \tau , t + 1 }$ . The immediate reward is obtained from the dataset-provided reward\_bind value associated with the selected substitution. Specifically, for $( i _ { \tau , t } , a _ { \tau , t } )$ , we use the DMS-measured change in binding to CGG, where positive values indicate improved binding and negative values indicate reduced binding: $\boldsymbol { r } _ { \tau , t } = \mathrm { r e w a r d \_ b i n d } ( i _ { \tau , t } , a _ { \tau , t } )$ . Because binding measurements are available only for single mutants, the binding affinity of a multi-mutant child is not directly evaluated. Instead, the return of a trajectory is defined as the cumulative reward of its constituent single mutations, $\begin{array} { r } { \dot { R } _ { \tau , t } = \sum _ { k = t } ^ { T - 1 } r _ { \tau , k } } \end{array}$ . Consequently, the learning objective favors mutation trajectories whose constituent substitutions have larger cumulative DMS-measured binding improvements. We optimize the mutation policy $\pi _ { \boldsymbol { \theta } } ( u \mid \mathbf { s } )$ using a REINFORCE-style logged-action objective [Williams, 1992], with returns computed from the observed trajectories.

![](images/8bc3a290ff643a8a3c4afd2da2b32af4f1157ab902daa2fe386d5ac7474cf82b.jpg)  
Figure 3: Reward landscape across antibody structural regions. (a) Raw GCReplay rewards are imbalanced across regions, driving hotspot and safe-region policy collapse. (b) The learned RTMG meta-reward smooths extreme peaks and produces a more balanced reward landscape.

Raw-Reward Policy Collapse. Directly optimizing the raw DMS reward is unstable because the reward distribution is highly skewed and varies substantially across antibody regions. A few large positive rewards can dominate the policy gradient, while low-variance framework mutations provide an easier but less informative optimization path. In GCReplay (Figure 3a), these effects produce two failure modes: hotspot collapse, in which the policy repeatedly selects a small set of high-reward mutations, and safe-region collapse, in which it concentrates on structurally tolerant framework positions. Both modes improve the training objective without learning a diverse affinity-maturation policy. In contrast, the learned RTMG meta-reward smooths extreme reward peaks and reduces regional imbalance, producing the more even reward landscape shown in Figure 3b.

SHM-Prior Initialization and Domain Adaptation. To keep affinity optimization aligned with biologically plausible SHM behavior, Stage 1 includes a supervised domain-adaptation phase on GCReplay parent–child pairs after pretraining on human repertoires. This phase corrects the human-to-mouse sequence mismatch and produces the Adapted SHM Prior. The trainable Stage 2 policy is initialized from this checkpoint, and a frozen copy, π<sub>prior</sub>, is retained as an SHM reference. A KL penalty toward $\pi _ { \mathrm { p r i o r } }$ constrains subsequent RL updates from departing too far from the learned SHM distribution. This prior controls where the policy explores, but it does not correct the imbalance in what the raw reward favors. RTMG addresses this remaining optimization problem.

Reward–Tax Meta-Gradient. RTMG adapts meta-gradient intrinsic-reward learning [Zheng et al., 2018] to learn a training signal for antibody mutation policies. It introduces two auxiliary networks with parameters ϕ and ψ. RewardNet transforms the cumulative measured reward $c _ { \tau , t }$ together with trajectory depth, mutation position, and antibody region into a benefit score. Its pre-normalized output is bounded to suppress extreme peaks, then centered so that its batch mean equals a fixed target without changing within-batch ordering. TaxNet receives the floored log score of the resulting sequence,

$$
\ell _ { \tau , t } = \operatorname* { m a x } \{ \log P _ { \mathrm { s e q } } ( \mathbf { s } _ { \tau , t + 1 } ) , \ell _ { \operatorname* { m i n } } \} ,\tag{2}
$$

where $P _ { \mathrm { s e q } }$ combines empirical sequence frequency with SHM-prior accessibility (Appendix E.1). Sequences that recur across convergent trajectories, or are readily reachable from recurrent sequences under the SHM prior, receive larger scores and can therefore incur a larger exploitation tax. For a training batch B, the meta-reward is

$$
\begin{array} { r } { r _ { \mathrm { m e t a } } ^ { ( \tau , t ) } = \overline { { R } } _ { \phi , \mathcal { B } } ^ { ( \tau , t ) } - T _ { \psi } ( \ell _ { \tau , t } ) , } \end{array}\tag{3}
$$

where $\overline { { R } } _ { \phi , B } ^ { ( \tau , t ) }$ is RewardNet’s normalized benefit and $T _ { \psi } ( \ell _ { \tau , t } )$ is TaxNet’s sequence-dependent penalty. RewardNet therefore determines which outcomes are valuable, whereas TaxNet regulates how strongly recurrent or SHM-accessible sequence outcomes are exploited.

Algorithm 1 Reward–Tax Meta-Gradient training on logged trajectories   
Require: Logged trajectory data $\mathcal { D } ;$ Adapted SHM Prior $\pi _ { \mathrm { p r i o r } } ;$ policy $\pi _ { \theta }$ initialized from $\pi _ { \mathrm { p r i o r } } $ ; RewardNet $R _ { \phi } ;$   
TaxNet $T _ { \psi }$   
1: for each training iteration do   
2: Sample a batch of logged transitions and sub-trajectories $\begin{array} { r } { B \sim \mathcal { D } . } \end{array}$   
3: Compute RewardNet benefits $\overline { { R } } _ { \phi , B } ,$ sequence scores $\ell ,$ taxes $T _ { \psi } ( \ell )$ , and meta-rewards $r _ { \mathrm { m e t a } } = \overline { { R } } _ { \phi , B } - T _ { \psi } ( \ell )$   
4: Compute Monte Carlo meta-reward returns and the inner REINFORCE-plus-KL loss.   
5: Form the differentiable probe update $\theta ^ { \prime }$ using Equation $4 ;$ do not overwrite $\theta .$   
6: Evaluate the capped-entropy term under $\pi _ { \theta ^ { \prime } }$ and the four fidelity losses for RewardNet, TaxNet, and their   
difference.   
7: Update $( \phi , \psi )$ by differentiating the outer objective through $\theta ^ { \prime } .$   
8: Recompute $r _ { \mathrm { m e t a } }$ with the updated auxiliary networks and detach it from $( \phi , \psi )$   
9: Update the deployed policy θ using Equation 5.   
10: end for   
Ensure: Affinity-optimized mutation policy $\pi _ { \theta }$

RewardNet and TaxNet are not fitted to manually specified targets. Instead, they are optimized through the effect of $r _ { \mathrm { m e t a } }$ on a temporary probe policy. Let $\mathcal { L } _ { \mathrm { i n n e r } } ( \theta ; r _ { \mathrm { m e t a } } )$ denote the return-weighted logged-action REINFORCE objective plus the KL penalty toward $\pi _ { \mathrm { p r i o r } } .$ , as defined in Appendix E. One differentiable inner step gives

$$
\begin{array} { r } { \theta ^ { \prime } = \theta - \eta _ { \mathrm { i n } } \nabla _ { \theta } \mathcal { L } _ { \mathrm { i n n e r } } ( \theta ; r _ { \mathrm { m e t a } } ) . } \end{array}\tag{4}
$$

The parameters $\theta ^ { \prime }$ are used only to evaluate the meta-objective and are never deployed. This objective encourages mutation-site diversity through Shannon entropy, capped at a target value so that diversity does not dominate once sufficient exploration is reached. Three Pearson-correlation losses preserve fidelity: i) RewardNet must remain aligned with cumulative DMS reward and with the log sequence-probability score of the trajectory’s terminal sequence; ii) TaxNet must remain aligned with the per-step resulting-sequence score supplied to it; iii) and the net meta-reward must remain aligned with measured DMS reward. Differentiating this objective through Equation 4 updates only $( \phi , \psi )$ The entropy term determines whether the learned shaping improves exploration, while the fidelity terms prevent the auxiliary networks from inventing an arbitrary training signal.

After updating the auxiliary networks, we recompute the meta-reward and update the deployed policy with its gradient detached,

$$
\theta  \theta - \eta _ { \theta } \nabla _ { \theta } \mathcal { L } _ { \mathrm { i n n e r } } ( \theta ; \mathrm { s t o p g r a d } ( r _ { \mathrm { m e t a } } ) ) .\tag{5}
$$

Thus, the second-order meta-gradient learns only the reward-shaping networks, whereas $\pi _ { \theta }$ receives a standard first-order REINFORCE-plus-KL update. Both steps use the same logged trajectories; no mutation is sampled or experimentally executed during training. Algorithm 1 summarizes the update order, while Appendix E provides the architecture in Figure 7 and gives detailed definitions of the network outputs, normalization, returns, and outer objective.

## 5 Experimental Results

We first evaluate whether Stage 1 learns the mutation grammar of SHM, then test whether Stage 2 generates high-affinity, diverse antibody variants under fixed mutation budgets. Additional SHM probing of the optimized policy is reported in Appendix A.1.

## 5.1 Stage 1: SHM Mutation Prediction

Tang BCR Dataset. We evaluate the pretraining phase of Stage 1 on the Tang B-cell receptor (BCR) dataset [Tang et al., 2022], a BCR repertoire collection from 21 human donors processed through a standardized parent–child pair extraction pipeline [Nguyen et al., 2015], yielding 505,915 mutation pairs. Mutation patterns are consistent with known SHM biology: CDRs account for 45.4% of all mutations despite spanning a minority of positions, and 84.2% of substitutions are reachable through a single nucleotide change. The dataset is split at the donor level (70/15/15) to prevent repertoire leakage across splits. Full dataset statistics are provided in Appendix C.1.

Benchmarking Setup. We evaluate Stage 1 against four SHM models: S5F [Yaari et al., 2013], DASM (in two variants: mutation-and-selection and selection-only) [Matsen IV et al., 2026], Thrifty [Sung et al., 2025], and DeepSHM [Tang et al., 2022]. The Stage 1 neural models are trained with batch size 256, and all models are evaluated on the Tang test set. SHM prediction is framed as two independent sub-tasks: given a parent sequence, predict (1) which position mutates next, evaluated by pos@k (whether the true mutation site appears among the top-k ranked positions), and (2) which amino acid it mutates to, evaluated by aa@k (whether the observed substitution appears among the top-k predicted amino acids at the true position). Conditioning the amino-acid sub-task on the ground-truth position isolates substitution accuracy from position accuracy, allowing fair comparison across models with fundamentally different architectures.

Results. Table 1 reports top-k position, amino-acid, and joint accuracy across all models, and Figure 4 visualizes the top-1 and joint results. iGC Stage 1 leads on all position metrics and on joint accuracy. The decisive gap is on position: pos@1 of 11.929% more than doubles the best baseline (DASM mutation×selection, 5.307%). Substitution accuracy is close to the strongest baselines (aa@1 40.777% versus DASM 40.561%), so the compounding of position and substitution performance gives a joint accuracy of 4.864%, more than twice the next best (2.153%). The advantage is concentrated on where rather than what, suggesting that the transformer’s full-sequence context captures hotspot targeting that motif-based and shallow learned models miss. Appendix B.1 further examines where the model attends when predicting mutation sites.

Table 1: Prediction accuracy of mutation position and amino acid identity for different SHM models. Joint accuracy is the fraction of examples where both the predicted mutation position and amino acid substitution are correct. Values are reported as mean ± standard error over three data splits. Arrows indicate the preferred direction for each metric.
<table><tr><td>Model</td><td>pos@1 (↑)</td><td>pos@3 (↑)</td><td>pos@5 (↑)</td><td>aa@1(↑)</td><td>aa@3(↑)</td><td>aa@5 (↑)</td><td>Joint (↑)</td></tr><tr><td>S5F</td><td> $2 . 4 3 3 \pm 0 . 1 7 2$ </td><td> $7 . 8 1 3 \pm 0 . 2 9 3$ </td><td>12.237 ± 0.457</td><td> $2 6 . 1 3 0 \pm 0 . 5 6 9$ </td><td> $6 4 . 3 1 0 \pm 0 . 6 0 1$ </td><td> $8 4 . 9 5 3 \pm 0 . 3 3 1$ </td><td> $0 . 6 3 7 \pm 0 . 0 5 7$ </td></tr><tr><td>Thrifty</td><td> $4 . 2 5 7 \pm 0 . 1 8 3$ </td><td> $1 0 . 6 3 3 \pm 0 . 2 4 9$ </td><td></td><td></td><td></td><td>15.533 ± 0.230 32.777 ± 0.367 68.747 ± 0.450 86.813 ± 0.052</td><td> $1 . 3 9 5 \pm 0 . 0 6 6$ </td></tr><tr><td>DASM selection-only</td><td> $1 . 0 7 0 \pm 0 . 1 4 9$ </td><td> $3 . 9 6 0 \pm 0 . 1 6 3$ </td><td></td><td></td><td></td><td>6.947 ± 0.169 18.043 ± 0.548 38.497 ± 0.649 52.203 ± 0.790</td><td> $0 . 1 9 4 \pm 0 . 0 3 2$ </td></tr><tr><td>DASM (mutation × selection)</td><td> $5 . 3 0 7 \pm 0 . 0 5 8$ </td><td> $1 3 . 1 3 3 \pm 0 . 3 9 3$ </td><td></td><td></td><td></td><td>19.197 ± 0.382 40.561 ± 0.555 75.737 ± 0.36589.435 ± 0.225</td><td> $2 . 1 5 3 \pm 0 . 0 5 2$ </td></tr><tr><td>DeepSHM</td><td> $5 . 0 2 0 \pm 0 . 3 0 2$ </td><td> $1 1 . 7 4 7 \pm 0 . 2 5 3$ </td><td></td><td></td><td></td><td>17.467 ± 0.299 38.850 ± 0.653 74.250 ± 0.644 88.470 ± 0.122</td><td> $1 . 9 4 9 \pm 0 . 1 0 5$ </td></tr><tr><td>iGC (Stage 1 pretraining)</td><td> ${ \bf 1 1 . 9 2 9 \pm 0 . 1 4 1 }$ </td><td> ${ \bf 2 5 . 1 6 9 \pm 0 . 2 1 7 }$ </td><td></td><td></td><td></td><td>33.836 ± 0.23840.777 ± 0.323 75.552 ± 0.321 90.027 ± 0.188</td><td> $\mathbf { 4 . 8 6 4 \pm 0 . 0 5 2 }$ </td></tr></table>

![](images/d072c02dbb272924670b56c1de51e46b82d76e3ea332f9180d73c6d2a67e56d1.jpg)  
Figure 4: Stage 1 SHM mutation prediction on the Tang test set: top-1 mutation-site accuracy (left), top-1 amino-acid accuracy (middle), and joint accuracy (right) for each model. Bars show the mean over three data splits (in %), with error bars indicating the standard error; values are the same as in Table 1.

## 5.2 Stage 2: Affinity-Guided Clonal Selection

GCReplay Dataset. We fine-tune Stage 2 on GCReplay [DeWitt et al., 2026], a controlled in vivo affinity-maturation dataset of clonal trajectories from CGG-immunized mice. All sequences share the IGHV1-2 V-gene background, and each observed mutation step is paired with a DMS-derived binding reward $( \Delta K _ { D }$ against CGG), providing sequential supervision for policy optimization. Cohort statistics, trajectory extraction, data splits, and sub-trajectory augmentation are detailed in Appendix C.2. Benchmarking Setup. We benchmark iGC against two external baselines, SHMReplay [DeWitt et al., 2026] and CoSiNE [Lu et al., 2026], as well as the Adapted SHM Prior produced by Stage 1 and three internal RL ablations: RL with raw reward, adaptive reward shaping, and RewardNet only. All methods use identical data splits and generation budgets. All Stage 2 models are trained with batch size 64. Stage 2 checkpoints are selected on the validation set by maximizing $\overline { { R } } _ { 6 0 0 } \times \mathrm { j o i n t } @ 1 \times \# \mathrm { D i s t }$ . For each checkpoint, the current policy generates 100 validation rollouts at each mutation budget $k \in \{ 2 , 3 , 4 , 5 , 6 , 7 \}$ , giving 600 samples in total; $\overline { { R } } _ { 6 0 0 }$ is their mean cumulative DMS reward, and #Dist is the number of distinct mutation identities across these rollouts. At each mutation budget, every method generates the same number of candidate sequences. For each data split and budget, candidates are ranked by cumulative DMS reward, and we evaluate the top $N \in \{ 1 0 0 , 3 0 , 2 0 , 1 0 \}$ candidates. Within each top-N set, DMS is the mean reward and #Dist is the number of distinct mutation identities. We also report Prod $= \mathrm { D M S } \times \# \mathrm { D i s t }$ , which measures whether a method achieves high reward without collapsing onto a narrow mutation repertoire. Table 2 averages each metric equally across the six mutation budgets and reports means and standard deviations over three data splits. Full baseline configurations are described in Appendix D.2.

Table 2: Stage 2 benchmarking results averaged equally over mutation budgets k = 2 to 7, with the same number of generated sequences at each budget, and reported over three data splits. Each model’s mean row is followed by a parenthesized standard-deviation row across splits. DMS is mean reward, #Dist is the number of distinct mutation identities, and $\mathrm { P r o d } = \mathrm { D M S } \times \# \mathrm { D i s t }$ . Higher is better for all metrics. Bold and underlined Prod values denote the best and second-best means, respectively. denotes RL-based methods initialized from the Adapted SHM Prior.
<table><tr><td></td><td colspan="3">Top 100</td><td colspan="3">Top 30</td><td colspan="3">Top 20</td><td colspan="3">Top 10</td></tr><tr><td>Model</td><td>DMS</td><td>#Dist</td><td>Prod</td><td>DMS</td><td>#Dist</td><td>Prod</td><td>DMS</td><td>#Dist</td><td>Prod</td><td>DMS</td><td>#Dist</td><td>Prod</td></tr><tr><td>SHMReplay</td><td>0.514 (0.018)</td><td>257.6 (2.3)</td><td>132.4 (5.3)</td><td>0.878 (0.033)</td><td>99.3 (1.5)</td><td>87.2 (3.4)</td><td>0.990 (0.040)</td><td>70.7 (0.7)</td><td>70.0 (3.1)</td><td>1.164</td><td>38.5 (0.4)</td><td>44.8</td></tr><tr><td>CoSiNE</td><td>0.019</td><td>195.3</td><td>3.7</td><td>0.990</td><td>70.4</td><td>69.7</td><td>1.155</td><td>50.0</td><td>57.8</td><td>(0.047) 1.372</td><td>27.5</td><td>(1.7) 37.7</td></tr><tr><td>Adapted SHM Prior</td><td>(0.078) 0.185</td><td>(17.0) 166.9</td><td>(15.7) 30.9</td><td>(0.073) 1.045</td><td>(2.5) 62.8</td><td>(5.6) 65.6</td><td>(0.077) 1.195</td><td>(2.3) 44.4</td><td>(4.5) 53.1</td><td>(0.058) 1.414</td><td>(1.6) 25.0</td><td>(2.7) 35.4</td></tr><tr><td>RL w/ raw reward*</td><td>(0.121) 0.844</td><td>(5.4) 92.6</td><td>(19.1) 78.2</td><td>(0.143) 1.670</td><td>(5.7) 36.0</td><td>(5.0) 60.1</td><td>(0.160) 1.800</td><td>(5.2) 26.7</td><td>(4.1) 48.1</td><td>(0.154) 1.970</td><td>(3.1) 16.3</td><td>(2.0) 32.1</td></tr><tr><td>Adaptive reward shaping*</td><td>(0.298) 0.945</td><td>(7.4) 108.2</td><td>(30.9) 102.2</td><td>(0.152) 1.795</td><td>(0.9) 39.0</td><td>(4.8) 70.0</td><td>(0.158) 1.921</td><td>(1.3) 28.4</td><td>(4.7) 54.6</td><td>(0.152) 2.070</td><td>(1.0) 16.9</td><td>(3.6) 35.0</td></tr><tr><td>RewardNet only*</td><td>(0.167) 0.344</td><td>(7.9) 93.4</td><td>(10.6) 32.1</td><td>(0.067) 1.199</td><td>(3.9) 46.4</td><td>(6.0) 55.6</td><td>(0.035) 1.353</td><td>(2.5) 35.2</td><td>(4.3) 47.6</td><td>(0.017) 1.557</td><td>(1.0) 22.2</td><td>(1.8) 34.6</td></tr><tr><td>RTMG w/o entropy*</td><td>(0.150) 0.901</td><td>(19.9) 90</td><td>(14.0) 81.09</td><td>(0.191) 1.695</td><td>(9.2) 50</td><td>(5.4) 84.75</td><td>(0.194) 1.798</td><td>(7.0) 42</td><td>(4.3) 75.52</td><td>(0.209) 1.946</td><td>(3.7)</td><td>(2.5) 44.76</td></tr><tr><td></td><td>(0.214)</td><td>(11.2)</td><td>(14.8)</td><td>(0.118)</td><td>(7.6)</td><td>(11.7)</td><td>(0.109)</td><td>(6.1)</td><td>(10.4)</td><td>(0.097)</td><td>23 (4.3)</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>1.663</td><td>58.2</td><td>96.8</td><td>1.791</td><td>41.7</td><td>74.7</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>iGC*</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>(7.1)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>1.010</td><td>149.4</td><td>150.9</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1.974</td><td>23.5</td><td>46.4</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>(0.201)</td><td>(13.0)</td><td>(16.1)</td><td>(0.157)</td><td>(10.5)</td><td>(10.5)</td><td>(0.157)</td><td>(6.2)</td><td>(5.5)</td><td>(0.137)</td><td>(2.1)</td><td>(1.3)</td></tr></table>

Results. Table 2 shows the central reward–diversity trade-off. The RL-based methods generally achieve higher DMS than the non-RL generators (SHMReplay, CoSiNE, and the Adapted SHM Prior), showing the benefit of affinity-guided policy optimization. However, direct reward optimization concentrates probability on a smaller set of beneficial actions: RL with raw reward has the lowest #Dist at every cutoff, and adaptive reward shaping attains high DMS while still sacrificing substantial diversity. RewardNet alone does not resolve this trade-off. Full iGC achieves the highest Prod for the Top 100, Top 30, and Top 10 candidate sets and ranks second for Top 20, where RTMG without entropy is marginally higher. This ablation indicates that the entropy term generally improves the reward–diversity balance across the generated candidate pool, although the no-entropy variant remains competitive at one cutoff. Comparisons with RewardNet only still isolate the contribution of TaxNet: RewardNet refines the affinity signal, while the learned tax prevents the resulting policy from collapsing onto a narrow set of mutations. Affinity optimization retains measurable SHM structure, with predicted-position mutability remaining correlated with S5F (Table 3; Appendix A.1).

## 6 Conclusion

iGC models antibody affinity maturation by separating mutation generation from antigen-specific selection and then coupling the two processes through reinforcement learning. The Stage 1 Transformer learns an SHM prior from B-cell lineage data and substantially improves mutation-site prediction, while Stage 2 optimizes binding without discarding the sequence structure encoded by that prior. Reward–Tax Meta-Gradient shaping further prevents the policy from collapsing onto a small number of dominant substitutions or structurally tolerated framework sites. Across mutationpolicy and structure-based evaluations, iGC achieves a favorable balance between predicted binding improvement, mutation diversity, and SHM-consistent behavior, supporting learned mutation–selection policies as a framework for iterative antibody optimization. Our evaluation is limited by its reliance on DMS-derived training rewards rather than direct experimental measurements of generated candidates.

## References

C. Chothia and A. M. Lesk. Canonical structures for the hypervariable regions of immunoglobulins. Journal of molecular biology, 196(4):901–917, 1987.

C. Chothia, A. M. Lesk, A. Tramontano, M. Levitt, S. J. Smith-Gill, G. Air, S. Sheriff, E. A. Padlan, D. Davies, W. R. Tulip, et al. Conformations of immunoglobulin hypervariable regions. Nature, 342(6252):877–883, 1989.

W. S. DeWitt, A. A. Vora, T. Araki, J. G. Galloway, T. Alkutkar, J. Bortolatto, T. B. R. Castro, W. Dumm, C. Jennings Shaffer, T. Jia, L. Mesin, G. Ozorowski, J. Pae, D. K. Ralph, J. D. Bloom, A. Nourmohammad, Y. S. Song, A. B. Ward, T. N. Starr, F. A. Matsen IV, and G. D. Victora. Replaying germinal center evolution on a quantified affinity landscape. Cell, 2026. doi: 10.1016/j.cell.2026.05.013.

E. E. Ford, D. Tieri, O. L. Rodriguez, N. J. Francoeur, J. Soto, J. T. Kos, A. Peres, W. S. Gibson, C. A. Silver, G. Deikus, E. Hudson, C. R. Woolley, N. Beckmann, A. Charney, T. C. Mitchell, G. Yaari, R. P. Sebra, C. T. Watson, and M. L. Smith. Flairr-seq: A method for single-molecule resolution of near full-length antibody h chain repertoires. The Journal ofImmunology, 210(10):1607–1619, 2023. doi: 10.4049/jimmunol.2200825.

A. Gessner, S. W. Ober, O. Vickery, D. Oglic, and T. Uçar. Active learning for affinity prediction of antibodies. ´ arXiv preprint arXiv:2406.07263, 2024.

B. L. Hie, V. R. Shanker, D. Xu, T. U. Bruun, P. A. Weidenbacher, S. Tang, W. Wu, J. E. Pak, and P. S. Kim. Efficient evolution of human antibodies from general protein language models. Nature biotechnology, 42(2):275–283, 2024.

A. Honegger and A. PluÈckthun. Yet another numbering scheme for immunoglobulin variable domains: an automatic modeling and analysis tool. Journal of molecular biology, 309(3):657–670, 2001.

D. B. Jaffe, P. Shahi, B. A. Adams, A. M. Chrisman, P. M. Finnegan, N. Raman, A. E. Royall, F. Tsai, T. Vollbrecht, D. S. Reyes, N. L. Hepler, and W. J. McDonnell. Functional antibodies exhibit light chain coherence. Nature, 611 (7935):352–357, 2022. doi: 10.1038/s41586-022-05371-z.

M. M. Johnson, K. Sung, H. K. Haddox, A. A. Vora, T. Araki, G. D. Victora, Y. S. Song, J. Fukuyama, and F. A. Matsen IV. Nucleotide context models outperform protein language models for predicting antibody affinity maturation. PLOS Computational Biology, 21(12):e1013758, 2025.

S. Z. Lu, A. Vermani, K. Sanno, J. Lu, F. A. Matsen IV, M. Jagota, and Y. S. Song. Conditionally site-independent neural evolution of antibody sequences. ArXiv, pages arXiv–2602, 2026.

S. Luo, Y. Su, X. Peng, S. Wang, J. Peng, and J. Ma. Antigen-specific antibody design and optimization with diffusionbased generative models for protein structures. Advances in Neural Information Processing Systems, 35:9754–9767, 2022.

F. A. Matsen IV, W. Dumm, K. Sung, M. M. Johnson, D. H. Rich, T. N. Starr, Y. S. Song, J. Fukuyama, and H. K. Haddox. Separating selection from mutation in antibody language models. Elife, 15:RP109644, 2026.

A. K. Mishra and R. A. Mariuzza. Insights into the structural basis of antibody affinity maturation from next-generation sequencing. Frontiers in immunology, 9:117, 2018.

L.-T. Nguyen, H. A. Schmidt, A. Von Haeseler, and B. Q. Minh. Iq-tree: a fast and effective stochastic algorithm for estimating maximum-likelihood phylogenies. Molecular biology and evolution, 32(1):268–274, 2015.

T. H. Olsen, I. H. Moal, and C. M. Deane. Addressing the antibody germline bias and its effect on language models for improved antibody design. Bioinformatics, 2024.

B. Pilzecker and H. Jacobs. Mutating for good: Dna damage responses during somatic hypermutation. Frontiers in immunology, 10:438, 2019.

O. L. Rodriguez, Y. Safonova, C. A. Silver, K. Shields, W. S. Gibson, J. T. Kos, D. Tieri, H. Ke, K. J. L. Jackson, S. D. Boyd, M. L. Smith, W. A. Marasco, and C. T. Watson. Genetic variation in the immunoglobulin heavy chain locus shapes the human antibody repertoire. Nature Communications, 14(1):4419, 2023. doi: 10.1038/ s41467-023-40070-x.

J. A. Ruffolo, J. J. Gray, and J. Sulam. Deciphering antibody affinity maturation with language models and weakly supervised learning. arXiv preprint arXiv:2112.07782, 2021.

S. Shan, S. Luo, Z. Yang, J. Hong, Y. Su, F. Ding, L. Fu, C. Li, P. Chen, J. Ma, et al. Deep learning guided optimization of human antibody against sars-cov-2 variants with broad neutralization. Proceedings ofthe National Academy of Sciences, 119(11):e2122954119, 2022.

K. Sung, M. M. Johnson, W. Dumm, N. Simon, H. Haddox, J. Fukuyama, and F. A. Matsen. Thrifty wide-context models of b cell receptor somatic hypermutation. Elife, 14:RP105471, 2025.

N. Tagasovska, J. Ludwiczak, A. Loukas, E. K. Makowski, H. Mohammadi-Peyhani, S. P. Mahajan, K. Zadorozhny, D. Lee, A. Goodman, J. Y.-Y. Lin, et al. Property enhancer–a data efficient multi-objective approach for functional antibody optimization. bioRxiv, pages 2025–10, 2025.

C. Tang, A. Krantsevich, and T. MacCarthy. Deep learning model of somatic hypermutation reveals importance of sequence context beyond hotspot targeting. Iscience, 25(1), 2022.

S. Vergani, I. Korsunsky, A. N. Mazzarello, G. Ferrer, N. Chiorazzi, and D. Bagnara. Novel method for high-throughput full-length ighv-d-j sequencing of the immune repertoire from bulk b-cells with single-cell resolution. Frontiers in Immunology, 8:1157, 2017. doi: 10.3389/fimmu.2017.01157.

G. D. Victora and M. C. Nussenzweig. Germinal centers. Annual review of immunology, 30:429–457, 2012.

R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992.

G. Yaari, J. A. Vander Heiden, M. Uduman, D. Gadala-Maria, N. Gupta, J. N. Stern, K. C. O’Connor, D. A. Hafler, U. Laserson, F. Vigneault, et al. Models of somatic hypermutation targeting and substitution based on synonymous mutations from high-throughput immunoglobulin sequencing data. Frontiers in immunology, 4:358, 2013.

Q. Yu, L. Guo, X. Qin, X. Huang, B. Tian, H. Wang, Y. Liu, Y. Lang, D. Wang, Z. Shen, et al. High-affinity protein binder design via flow matching and in silico maturation. bioRxiv, pages 2026–01, 2026.

X. Zhao, Y.-C. Tang, A. Singh, V. J. Cantu, K. An, J. Lee, A. E. Stogsdill, I. M. Hamdi, A. K. Ramesh, Z. An, et al. Abbibench: A benchmark for antibody binding affinity maturation and design. arXiv preprint arXiv:2506.04235, 2025.

Z. Zheng, J. Oh, and S. Singh. On learning intrinsic rewards for policy gradient methods. In Advances in Neural Information Processing Systems, volume 31, 2018.

## A Additional Stage 2 Results

## A.1 SHM Probing of the Final iGC Policy

Section 5.2 establishes that iGC favors high-binding-reward mutations. The complementary question is whether the final RL-tuned policy still proposes mutations consistent with somatic hypermutation rather than drifting into unconstrained amino-acid optimization. We therefore compare canonical SHM motif frequencies at real mutation positions, iGC-predicted positions, and random background positions in Table 3.

Table 3: Canonical SHM motif frequencies for the final iGC policy after Stage 2 fine-tuning.
<table><tr><td></td><td>WRC</td><td>WA</td><td>GYW</td></tr><tr><td>Real mutation positions</td><td>26.9%</td><td>24.9%</td><td>18.7%</td></tr><tr><td>iGC predicted positions</td><td>57.0%</td><td>1.0%</td><td>11.1%</td></tr><tr><td>Random background</td><td>2.0%</td><td>19.4%</td><td>4.1%</td></tr></table>

iGC strongly enriches WRC and GYW motifs relative to random background, showing that affinity optimization preserves substantial AID-associated targeting structure. Its predicted-position mutability also remains correlated with the S5F model (Pearson 0.474, Spearman 0.450). WA motifs are underrepresented. Unlike the three-base WRC and GYW motifs, WA is a short two-base motif that already occurs at 19.4% of random background positions; because many WA occurrences are not mutated, motif presence alone provides a less specific and noisier learning signal. The final policy may therefore capture these SHM mechanisms less readily. Together with the binding results in Table 2, these results support the two central properties of iGC: its mutations favor binding while remaining substantially aligned with SHM biology.

## A.2 Additional Stage 2 Validation

## A.2.1 Held-Out Sequence Rediscovery

This analysis tests whether the improvement can be explained by memorization. We split GCReplay by germinal center, so all lineage trees in the test germinal centers are unseen during training. For each mutation budget from k = 2 to 4, iGC generates 5,000 variants and is evaluated by exact matches to naturally observed positive-reward sequences in these held-out trees. As shown in Table 4, iGC rediscovers 32.0%, 10.9%, and 5.9% of the eligible test sequences at k = 2, 3, 4, respectively.

Table 4: Rediscovery of unseen positive-reward sequences in the held-out test set.
<table><tr><td>k</td><td>Unseen positives rediscovered</td><td>Total unseen positives</td><td>Rediscovery rate</td></tr><tr><td>2</td><td>54</td><td>169</td><td>32.0%</td></tr><tr><td>3</td><td>41</td><td>375</td><td>10.9%</td></tr><tr><td>4</td><td>15</td><td>255</td><td>5.9%</td></tr></table>

## A.3 Critical Stage 2 Design Ablations

We ablate two implementation choices that directly control the learned reward-shaping signal: the information provided to RewardNet and the lower bound applied to TaxNet’s log sequence-probability score.

## A.3.1 RewardNet Input Selection

Table 5 compares candidate RewardNet inputs. The full configuration combines cumulative DMS reward, mutation count, position identity, and antibody-region identity, and yields the highest average DMS reward.

Table 5: RewardNet input selection. The chosen configuration is shown in bold.
<table><tr><td colspan="2">RewardNet input</td><td></td><td>Avg. DMS reward (↑) Avg. #distinct mut. (↑)</td></tr><tr><td>cum_dms + pos_onehot</td><td></td><td>0.526</td><td>129.3</td></tr><tr><td>cum_dms + mut_count</td><td></td><td>0.743</td><td>116.6</td></tr><tr><td></td><td>cum_dms + mut_count + pos_onehot + region_onehot</td><td>1.134</td><td>108.0</td></tr></table>

## A.3.2 TaxNet Sequence-Score Floor

Table 6 reports the sweep used to select the lower bound on TaxNet’s log sequence-probability input. We use $\ell _ { \mathrm { m i n } } = - 4 . 0$ , which provides the selected reward–diversity trade-off: its average DMS reward is close to the maximum at −4.5 (1.134 vs. 1.142), while retaining slightly more distinct mutations (108.0 vs. 106.3).

Table 6: TaxNet log sequence-score floor selection over mutation budgets k = 1 to 7. The selected floor is shown in bold.
<table><tr><td>Floor</td><td>Avg. DMS reward (↑)</td><td>Avg. #distinct mut. (↑)</td></tr><tr><td>-6.0</td><td>0.528</td><td>155.9</td></tr><tr><td>-5.5</td><td>0.263</td><td>178.1</td></tr><tr><td>-5.0</td><td>0.411</td><td>129.4</td></tr><tr><td>-4.5</td><td>1.142</td><td>106.3</td></tr><tr><td>-4.0</td><td>1.134</td><td>108.0</td></tr><tr><td>-3.5</td><td>0.880</td><td>127.1</td></tr><tr><td>-3.0</td><td>0.703</td><td>119.3</td></tr></table>

## A.4 Training Pipeline Ablation

We isolate the contribution of the two-stage pipeline by comparing three variants: P1 omits Stage 1 and trains Stage 2 directly from random initialization; P2 uses only the Tang-pretraining phase of Stage 1 before Stage 2; and P3 uses the complete Stage 1, consisting of Tang pretraining followed by GCReplay domain adaptation, before Stage 2. All three use identical Stage 2 hyperparameters and reward shaping. Results are in Table 7.

Is Stage 1 pretraining necessary? (P1 vs. P2). Tang pretraining provides an 18% improvement in joint accuracy (0.146 vs. 0.124 on test) and a 23% gain in aa@1, confirming that behavior cloning on human repertoire data provides a meaningful prior for amino-acid substitution grammar. Pretraining also accelerates convergence: under the validation $\overline { { R } } _ { 6 0 0 } \times \mathrm { j o i n t } @ 1 \times \# 1$ Dist criterion, P2 selects epoch 4 vs. epoch 12 for P1. P1 achieves higher proposed mutation reward (+0.512 vs. +0.455); without a BC prior anchoring it to any positional bias, RL explores more freely and converges on high-reward positions. However, this comes at the cost of a spurious FR1 artifact (18.3% of argmax predictions vs. 4.5% empirical), analogous to the FR4 artifact in P2.

Does Tang pretraining transfer to GCReplay? (P2). The Tang-pretrained model shows substantial zero-shot transfer from human repertoires to mouse IGHV1-2 sequences before any RL fine-tuning. However, P2 carries a cross-species artifact: the human-trained positional prior concentrates 67.2% of probability on FR4 (mouse J-gene is out-of-distribution), and despite the framework penalty, FR4 remains at 26.6% after RL, 19× the empirical rate. CDR2, the second most mutated region empirically (26.7%), is correspondingly under-targeted at 14.8%.

Does domain adaptation resolve the OOD artifact? (P3 vs. P2). The domain-adaptation phase of Stage 1 directly addresses the cross-species mismatch by fine-tuning the Tang-pretrained prior on GCReplay trajectories before Stage 2. This reduces FR4 from 26.6% to 1.2% (near the empirical 1.4%) and brings CDR2 targeting to 21.6% (empirical: 26.7%). Joint accuracy improves from 0.146 to 0.162 on test, while suppressing the FR4 artifact induced by the human-to-mouse J-gene distribution shift.

Table 7: Pipeline ablation on GCReplay test set. FR4% and CDR2% are argmax region distributions; empirical rates are 1.4% and 26.7% respectively.
<table><tr><td>Pipeline</td><td>pos@1</td><td>aa@1</td><td>joint</td><td>FR4%</td><td>CDR2%</td></tr><tr><td>P1: No Stage 1</td><td>0.296</td><td>0.417</td><td>0.124</td><td>7.4%</td><td>33.7%</td></tr><tr><td>P2: Stage 1 pretrain only</td><td>0.286</td><td>0.512</td><td>0.146</td><td>26.6%$</td><td>14.8%</td></tr><tr><td>P3: Full Stage 1</td><td>0.301</td><td>0.539</td><td>0.162</td><td>1.2%</td><td>21.6%</td></tr></table>

<sup>‡</sup>FR4 artifact from human-to-mouse J-gene distribution shift; see text.

Cross-timepoint generalization (D20-only training). As a supplementary check, we train Stage 2 on D20 GCReplay trajectories only and evaluate on D15 validation and test sets. D20-only training matches or marginally exceeds the full pipeline on all metrics (joint: 0.165 vs. 0.162, aa@1: 0.551 vs. 0.539), with nearly identical regional distributions. This suggests that D15 and D20 carry similar positional targeting signal, and that the learned policy is not simply memorizing timepoint-specific patterns.

## B Additional Stage 1 Results

We complement the main-text Stage 1 mutation-prediction benchmark by inspecting the model’s attention patterns, probing whether the Adapted SHM Prior retains canonical SHM motif preferences, and characterizing how Stage 1 accuracy scales with training-set size.

## B.1 Stage 1 Attention Probing

Figure 5 probes Stage 1 attention when predicting mutation sites. Mean attention peaks in the four- to five-position window around the true mutation site (highlighted in yellow), especially after averaging across layers. This local focus matches AID biology: cytosine deamination is driven by short sequence motifs (WRC/RCY), so mutation targeting depends strongly on nearby nucleotide context. The pattern suggests that the transformer learns SHM’s local context dependence from observed mutation events, without explicit motif supervision.

![](images/848eb2cd33721f8632043a51c6c8fbb568c59b118eeb4f3e0a94b3b8c7f060fb.jpg)  
Figure 5: Mean attention weight per transformer layer as a function of relative sequence position to the predicted mutation site (position 0, dashed red), aggregated across the Tang BCR test set. Yellow bands highlight the ±5 position window. All layers consistently attend most strongly to the local neighborhood of the mutation site, consistent with AID’s known short-motif targeting mechanism.

## B.2 SHM-Prior Motif Probing

Table 8 provides the corresponding diagnostic for the Adapted SHM Prior produced at the end of Stage 1, before Stage 2 fine-tuning. The Adapted SHM Prior has Pearson and Spearman correlations of 0.554 and 0.692 with S5F mutability, respectively.

Table 8: Canonical SHM motif frequencies for the Stage 1 Adapted SHM Prior.
<table><tr><td></td><td>WRC</td><td>WA</td><td>GYW</td></tr><tr><td>Real mutation positions</td><td>26.9%</td><td>24.9%</td><td>18.7%</td></tr><tr><td>Adapted SHM Prior predicted positions</td><td>54.4%</td><td>1.8%</td><td>42.1%</td></tr><tr><td>Random background</td><td>2.0%</td><td>19.4%</td><td>4.1%</td></tr></table>

## B.3 Stage 1 Model Ablation

We ablate the backbone used to learn the Stage 1 SHM mutation policy and track how this architectural choice propagates through the domain-adaptation phase of Stage 1 and subsequent Stage 2 fine-tuning.

## B.3.1 Backbone Architecture for SHM Policy Learning

We ablate three backbone variants across the full two-stage pipeline, including both phases of Stage 1: a small fromscratch Transformer (4.8M parameters, d = 256, 6 layers, 8 heads), a scaled from-scratch Transformer (14.3M, d = 384, 8 layers, 12 heads), and AbLang2 (44.4M), an antibody-specific encoder pretrained on the Observed Antibody Space (OAS) via masked language modeling [Olsen et al., 2024]. AbLang2 uses rotary attention and SwiGLU activations; its MLM head is discarded and two linear heads for position and amino acid prediction are trained from scratch, while the encoder is fine-tuned end-to-end. The from-scratch and scaled models consume Aho-numbered alignments (fixed length 149); AbLang2 consumes raw heavy-chain sequences in its native tokenization, as its OAS-pretrained encoder implicitly encodes antibody sequence structure without explicit alignment.

We report joint accuracy as the fraction of examples for which both the mutation position and amino-acid identity are predicted correctly, along with the marginal top-1 position accuracy (pos@1) and amino-acid accuracy (aa@1) Results are summarized in Table 9.

Stage 1 pretraining and domain adaptation. During the Tang-pretraining phase of Stage 1, the final 4.8M fromscratch model trained on the full Tang dataset matches the main-text result: pos@1 = 11.929%, aa@1 = 40.777%, and joint@1 = 4.864%. The scaled 14.3M model provides only a marginal joint-accuracy change (4.86%), suggesting that capacity alone is not the limiting factor at this data scale. AbLang2 reaches a similar joint value in this ablation run (4.97%) and converges faster, but the final pipeline uses the 4.8M from-scratch backbone for consistency with the main Stage 1 and Stage 2 experiments. After the GCReplay domain-adaptation phase of Stage 1, AbLang2 leads (15.87% vs. 15.62% for from-scratch), while the scaled model degrades noticeably (14.19%), suggesting it overfits under distribution shift. These backbone comparisons are useful for model selection, but the reported full-Tang 4.8M result is kept consistent with Table 1.

RL fine-tuning (Stage 2). Under matched hyperparameters $( \mathrm { l r } = 5 \times 1 0 ^ { - 5 } , \beta _ { \mathrm { K L } } = 0 . 5 )$ , the from-scratch model outperforms AbLang2 on joint after RL training (15.20% vs. 13.37%). The validation $\overline { { R } } _ { 6 0 0 } \times \mathrm { j o i n t } @ 1 \times \# \mathrm { D i s t }$ criterion selects epoch 1 for both models, a characteristic pattern of REINFORCE training: since RL optimizes for reward rather than pattern matching, the joint metric, which measures alignment with empirical mutation patterns, degrades as training progresses. The from-scratch model’s RL-stage advantage likely reflects that its smaller capacity makes it less susceptible to drifting from the BC prior under policy gradient updates, even under the same KL penalty weight. We note that joint accuracy is an imperfect proxy for RL-stage quality, as it does not directly measure binding improvement; downstream evaluation on a binding affinity oracle would better distinguish backbone quality at this stage.

![](images/a0c3097f84eb38e6c457223d97d8137ca564b4856f68ede3adc8173ba6645332.jpg)  
Figure 6: Stage 1 scaling law: test-set accuracy versus number of training trajectories for position (left), amino-acid (middle), and joint (right) prediction. Each point is a separate model trained on one PCP dataset (Ford, Rodriguez, Jaffe, Tang; the Jaffe point is labeled $\mathbf { \tilde { W } } \mathbf { y a t t } ^ { \mathsf { * } }$ in the right panel). Dashed lines are power-law fits $\operatorname { a c c } = a n ^ { b }$ extrapolated beyond the largest dataset; fitted exponents are given in the legends.

Table 9: Backbone ablation across training phases of the two-stage pipeline. Entries report joint@1 accuracy (%).
<table><tr><td>Training phase</td><td>From-scratch (4.8M)</td><td>Scaled (14.3M)</td><td>AbLang2 (44.4M)</td></tr><tr><td>Stage 1: Tang pretraining</td><td>4.864%</td><td>4.86%</td><td>4.97%</td></tr><tr><td>Stage 1: GCReplay adaptation</td><td>15.62%</td><td>14.19%</td><td>15.87%</td></tr><tr><td>Stage 2: GCReplay RL</td><td>15.20%</td><td>9.47%</td><td>13.37%</td></tr></table>

## B.4 Stage 1 Data Scaling

To understand how much of Stage 1 performance is limited by data, we train the same 4.8M-parameter Transformer policy (Appendix B.3) independently on four human heavy-chain parent–child-pair (PCP) collections of increasing size, compiled by Johnson et al. [2025]: Ford [Ford et al., 2023] (2,407 trajectories), Rodriguez [Rodriguez et al., 2023] (18,639), Jaffe [Jaffe et al., 2022] (176,656; labeled “Wyatt” in Figure 6 after the archive’s file prefix), and Tang [Tang et al., 2022, Vergani et al., 2017] (438,302). The Ford, Rodriguez, and Jaffe models are single runs with Aho-numbered input, batch size 512, learning rate $1 0 ^ { - 3 }$ with 1,000 warmup steps, 10 epochs, and seed 42; the Tang point is the main-text Stage 1 result (Table 1; batch size 256, mean over three data splits). All models use donor-stratified 70/15/15 splits. We report top-k position accuracy (pos@k), amino-acid accuracy at the true site (aa@k), and joint top-1 accuracy on each dataset’s held-out test split, and fit power laws $\operatorname { a c c } = a n ^ { b }$ in the number of training trajectories n (Figure 6).

Position prediction scales with data; amino-acid prediction saturates. Position accuracy improves steadily as the number of training trajectories grows, and the power-law fits show that it keeps rising with more data across all top-k cutoffs. In contrast, amino-acid accuracy is nearly flat: even the smallest dataset already performs close to the largest, so adding data barely changes substitution prediction. This is consistent with the observation that most SHM substitutions are reachable by a single nucleotide change, so local codon context already supplies much of the amino-acid signal Joint accuracy therefore inherits the position scaling, indicating that where to mutate, rather than what to mutate to, is the main data-limited component of Stage 1.

## C Datasets

We use one repertoire dataset to learn the SHM prior and two affinity-labeled datasets for reinforcement-learning experiments.

## C.1 BCR Tang for SHM Prior Learning

The Tang dataset is a paired B-cell receptor (BCR) repertoire collection from 21 human donors spanning three cohorts: ten healthy B-series donors (B10 to B21), seven chronic lymphocytic leukaemia (CLL) donors, and three healthy AG-series donors. Raw repertoires are processed through the Matsen lab parent-child pair (PCP) extraction pipeline, comprising IgBLAST V/J annotation (assignment of each sequence to its inherited variable and joining gene segments), DefineClones clonal clustering, IQ-TREE phylogenetic reconstruction, and HKY+G ancestral sequence reconstruction, producing directed mutation pairs where each edge in a clonal lineage tree yields one PCP. In total, the dataset contains 505,915 PCPs across 43,692 clonal lineage trees, decomposing into 438,302 trajectories and 2,024,413 individual single-amino-acid mutation steps. All sequences are IgH variable domains averaging 121 amino acids in length.

V and J Gene Usage. V gene usage spans 117 distinct IGHV genes, with IGHV3-23\*01 the most frequent at 12.7%. J gene usage spans 13 distinct IGHJ genes, with IGHJ4\*02 the most frequent at 55.7%. Inter-donor divergence in V-gene usage is substantial (mean pairwise Jensen-Shannon distance 0.487). CLL donors exhibit a more restricted, oligoclonal V-gene repertoire relative to healthy donors, consistent with the clonal expansion characteristic of the disease.

Mutation Statistics. Mutation burden per trajectory is heavily right-skewed: median 2, mean 4.62, max 47. Mutations concentrate in known SHM hotspots at IMGT positions 35 to 36 (CDR1) and 64 (CDR2). CDRs collectively account for 45.4% of all mutations despite spanning a much smaller fraction of scored positions, consistent with activation-induced cytidine deaminase (AID) targeting of WRC/RCY motifs. 84.2% of observed amino acid substitutions are reachable via a single nucleotide change from the parent codon, matching AID’s single-cytosine-deamination mechanism. The remaining 15.8% reflect multiple AID events or repair-patch mutations accumulated on the same tree edge, rather than sequencing artefacts.

Dataset Splits. The dataset is split at the donor level using greedy bin-packing targeting a 70/15/15 ratio (seed=42), ensuring no donor appears in more than one split. This yields 293,227 trajectories (12 donors) for training, 71,436 (4 donors) for validation, and 73,639 (5 donors) for test, corresponding to 1,439,804, 295,130, and 289,479 non-terminal mutation steps respectively, out of 2,024,413 total steps. V gene usage, J gene usage, and mutation burden distributions are statistically indistinguishable across the three splits, confirming that any generalisation gap reflects model capacity rather than distributional shift.

## C.2 GCReplay for RL Fine-Tuning

GCReplay is a controlled in vivo affinity maturation experiment in which 25 mice are immunized with CGG (chicken gamma-globulin, the antigen target in this assay) and germinal-center B cells are sampled at two timepoints (day 15 and day 20 post-immunization) across 119 distinct germinal centers (4 to 5 per mouse, 14 to 77 trajectories per GC). Unlike the BCR Tang dataset, every sequence in GCReplay derives from a single fixed V-gene background (IGHV1-2), yielding a constant 112-amino-acid heavy-chain length. Every observed mutation step carries a directly measured fitness signal: reward\_bind (∆K<sub>D</sub> against CGG, derived from deep mutational scanning, where single mutants are experimentally scored for binding). We use two trajectory views of GCReplay. For descriptive edge-level statistics, each clonal-tree edge is decomposed into its own short mutation trajectory (6,188 edge trajectories and 23,128 edge-derived sub-trajectories; median 1 mutation per edge trajectory; 65% single-amino-acid edges), preserving raw SHM events including detrimental mutations. For Stage 2 training and evaluation, we use the root-to-leaf trajectory view described below.

Mutation Statistics. 71.8% of mutations are beneficial (versus a 50% null expectation), confirming strong positive selection. Mutational mass concentrates in FR3 (32.8%, near-neutral reward) and CDR2 (26.7%, the dominant antigencontact loop for IGHV1-2), while CDR3 and CDR1 carry the largest per-mutation binding gains (mean $\Delta K _ { D }$ of +0.291 and +0.259 respectively). 98% of substitutions are single-nucleotide accessible, compared to 84% in Tang, consistent with the tighter evolutionary constraint of a single in vivo experiment. Day-20 germinal centers show higher and less variable binding gains than day-15 germinal centers, consistent with progressive affinity maturation over time.

Trajectory Extraction. The Stage 2 policy uses a separate root-to-leaf trajectory view. Trajectories are extracted from each clonal tree and split by germinal center into training, validation, and test sets containing 1,445, 1,046, and 997 trajectories, respectively. These counts are not intended to match the edge-level counts of 6,188 and 23,128 reported above, which describe a different extraction stage. For RL training, the 1,445 training trajectories contain 12,048 action steps; enumerating all contiguous sub-trajectories expands them into 75,022 training items. Under this construction, a transition at position k in a trajectory of length n appears in $n - k + 1$ training items, so earlier transitions receive greater weight than later transitions and the augmented state distribution differs from the original trajectory distribution. We do not importance-correct this augmentation-induced reweighting. The validation set remains at the full-trajectory level without augmentation. All sub-trajectories are retained regardless of reward sign, preserving detrimental mutations in the training distribution.

Domain Gap. GCReplay’s single mouse IGHV1-2 background differs substantially from Tang’s diverse human V-gene repertoire. The Aho-149 canonical alignment bridges this gap at the sequence level by placing residues from different antibodies into a shared 149-position coordinate system, and the domain-adaptation phase of Stage 1 continuously trains on GCReplay PCPs to adapt the SHM grammar prior to the murine sequence context before Stage 2 begins.

## D Benchmarking Protocols

## D.1 Stage 1 Benchmark Protocol

All Stage 1 models are evaluated on the held-out Tang donor split. We separately measure mutation-site ranking with pos@k, amino-acid prediction at the true site with aa@k, and joint top-1 accuracy, so improvements in site selection can be distinguished from improvements in substitution identity.

## D.2 Stage 2 Benchmark Protocol

Baseline Configurations.

SHMReplay. SHMReplay is run in its native fixed-k-mutation rollout mode (Strategy I), where mutations are applied unconditionally regardless of predicted DMS score and scored post hoc by the common structure-based proxy. No acceptance filter is applied during generation, meaning the method reflects the raw SHMReplay mutation distribution without any affinity-guided steering.

CoSiNE. CoSiNE uses an ESM2-150M encoder with a FreeRateMatrix continuous-time Markov chain (CTMC) head trained on GCReplay. Candidate sequences are generated by standard discrete Gillespie simulation directly from the learned CTMC substitution rates, without affinity-oracle reweighting or Guided Gillespie steering. This provides a fair comparison of the learned mutation policies rather than giving CoSiNE access to an additional oracle during generation. No position is revisited within a rollout.

Adaptive Reward Shaping. This baseline uses the hand-designed adaptive reward-shaping rule from the previous version of the method. It reshapes the per-step reward before each policy-gradient update using one penalty for each observed failure mode. For a mutation at position i in region $\rho \in \{ \mathrm { C D R 1 - 3 , F R 1 - 4 } \}$ , let r denote the affinity-derived reward for the current mutation, $\mu$ and σ the mean and standard deviation of the training reward distribution over all mutations, and $\mu _ { \rho } ^ { + }$ the mean positive reward within region $\rho .$ The shaped reward is

$$
\tilde { r } ( i , \rho ) = r - \left\{ \begin{array} { l l } { \alpha r , } & { r > \mu + \sigma \quad \mathrm { ( h o t s p o t ~ p e n a l t y ) } , } \\ { ( 1 - \mu _ { \rho } ^ { + } ) \lambda , } & { \mathrm { o t h e r w i s e ~ ( r e g i o n ~ p e n a l t y ) } . } \end{array} \right.\tag{6}
$$

Here, α specifies the fraction of the current reward subtracted from hotspot mutations, and λ controls the strength of the region penalty.

Hotspots are identified from the empirical reward distribution: any mutation whose reward exceeds one standard deviation above the global mean $( \mu + \sigma )$ is treated as a hotspot. Applying the hotspot penalty suppresses exceptionally rewarding mutations without eliminating their reward advantage, preventing the policy from collapsing onto a small number of dominant mutation events. For all remaining mutations, the region penalty scales with $( 1 - \mu _ { \rho } ^ { + } )$ , assigning larger penalties to regions whose positive mutations are less informative and smaller penalties to regions that consistently contain beneficial mutations. The two penalties are mutually exclusive, ensuring that each mutation is corrected only for its dominant failure mode.

Shaping is applied independently to every mutation before returns are computed. For a sub-trajectory of length T, the shaped return is

$$
G = \sum _ { t = 1 } ^ { T } \tilde { r } _ { t } ,
$$

which weights the log-probabilities of the corresponding actions in the REINFORCE update, while the KL regularization term keeps the learned policy close to the SHM prior.

Shared Evaluation Protocol. All compared methods start from the same IGHV1-2 germline sequence (the inherited, pre-mutation antibody sequence for this V-gene background; Clone 2.1 Fab is reverted to germline with substitutions A28D, R49K, and G64S) and generate 100 candidate sequences at each CDR-mutation budget from $k = 2$ to 7. We write substitutions as XnY, where X is the parent amino acid, n is the heavy-chain residue position, and Y is the replacement amino acid; site-only labels such as A106 omit Y. Each candidate is evaluated in three independent modeling runs, which are averaged, using the 9ODB crystal structure of Clone 2.1 Fab bound to chicken $\mathrm { I g Y }$ CH2, a CGG surrogate antigen. The common structure-based pipeline computes $\Delta \Delta G _ { \mathrm { b i n d } }$ relative to the germline structure, with more negative values indicating stronger predicted binding.

## E Implementation Details

## E.1 Evolutionary Sequence-Probability Estimation

TaxNet uses a relative sequence-probability score that combines evolutionary prevalence in GCReplay with accessibility under the frozen SHM prior. We first extract the trajectory collection T from all GCReplay lineage trees. Every sequence at every step is placed into a multiset,

$$
\mathcal { M } = \underset { \tau \in \mathfrak { T } , 0 \leq t \leq T _ { \tau } } { \biguplus } \{ \mathbf { s } _ { \tau , t } \} ,\tag{7}
$$

where the joint index ranges over every trajectory τ and every step t within that trajectory, and U denotes multiset union. We deliberately retain duplicate sequences. A sequence reached by multiple branches or trajectories therefore receives a larger count, preserving the signal of convergent evolution; such recurrence commonly indicates that the sequence has been repeatedly selected during affinity maturation. The germline sequence is also repeated because it begins every trajectory.

Let $n ( \mathbf { x } )$ be the multiplicity of sequence x in ${ \mathcal { M } } .$ , let $S _ { \mathrm { o b s } }$ be the set of unique observed sequences, and let g denote the germline. The empirical frequency and germline-normalized prevalence of an observed sequence are

$$
\widehat { p } _ { \mathrm { p o o l } } ( \mathbf { x } ) = \frac { n ( \mathbf { x } ) } { | \mathcal { M } | } , \qquad P _ { \mathrm { e m p } } ( \mathbf { x } ) = \frac { \widehat { p } _ { \mathrm { p o o l } } ( \mathbf { x } ) } { \widehat { p } _ { \mathrm { p o o l } } ( \mathbf { g } ) } = \frac { n ( \mathbf { x } ) } { n ( \mathbf { g } ) } .\tag{8}
$$

Thus, $P _ { \mathrm { e m p } } ( \mathbf { g } ) = 1$ . The score preserves relative recurrence while using the germline, which appears in all trajectories, as a common reference.

For an unseen sequence y ∈/ $S _ { \mathrm { o b s } } .$ let $d _ { \mathrm { e d i t } }$ denote sequence edit distance and define its nearest observed set as

$$
d _ { \operatorname* { m i n } } ( \mathbf { y } ) = \operatorname* { m i n } _ { \mathbf { x } \in \mathcal { S } _ { \mathrm { o b s } } } d _ { \mathrm { e d i t } } ( \mathbf { x } , \mathbf { y } ) , \qquad S _ { \operatorname* { m i n } } ( \mathbf { y } ) = \left\{ \mathbf { x } \in S _ { \mathrm { o b s } } : d _ { \mathrm { e d i t } } ( \mathbf { x } , \mathbf { y } ) = d _ { \operatorname* { m i n } } ( \mathbf { y } ) \right\} .\tag{9}
$$

Let $P _ { \mathrm { p r i o r } } ( \mathbf { y } \mid \mathbf { x } )$ be the transition probability assigned by the frozen Adapted SHM Prior produced by Stage 1 for producing y from x. We extend the empirical score to arbitrary sequences by propagating probability mass from every nearest observed sequence,

$$
P _ { \mathrm { s e q } } ( \mathbf { y } ) = \left\{ \begin{array} { l l } { P _ { \mathrm { e m p } } ( \mathbf { y } ) , } & { \mathbf { y } \in \mathcal { S } _ { \mathrm { o b s } } , } \\ { \displaystyle \sum _ { \mathbf { x } \in \mathcal { S } _ { \mathrm { m i n } } ( \mathbf { y } ) } P _ { \mathrm { e m p } } ( \mathbf { x } ) P _ { \mathrm { p r i o r } } ( \mathbf { y } \mid \mathbf { x } ) , } & { \mathbf { y } \notin \mathcal { S } _ { \mathrm { o b s } } . } \end{array} \right.\tag{10}
$$

Equation 10 assigns an unseen sequence a large score when it is both close to recurrent evolutionary states and likely under the SHM mutation prior. Because the scores are normalized relative to the germline rather than over the complete sequence space, $P _ { \mathrm { s e q } }$ is a relative probability score, not a probability mass function required to sum to one. RTMG uses only its logarithm and relative variation: TaxNet receives the score of each resulting sequence $\mathbf { s } _ { \tau , t + 1 }$ , while the RewardNet fidelity objective uses the score of the terminal sequence ${ \mathbf s } _ { \tau , T _ { \tau } }$

## E.2 Reward–Tax Meta-Gradient Details

This section gives the mathematical specification and bilevel optimization procedure for Reward–Tax Meta-Gradient, complementing the overview in Section 4.2.

![](images/9543198d61ae77400f5983f792c8f618418946df7b1bf79704ea05eb2f0980cb.jpg)  
Figure 7: Reward–tax meta-gradient (RTMG) architecture. RewardNet converts the measured DMS reward and mutation context into a learned benefit, while TaxNet penalizes recurrent or SHM-accessible sequence outcomes according to their evolutionary sequence-probability score. Their difference defines the meta-reward used for a differentiable update of the probe policy; feedback from the probe policy then trains the reward and tax networks to improve affinity while preserving mutation diversity. The deployed policy is updated using the resulting meta-reward.

RewardNet learns a refined reward from the measured DMS binding signal together with the mutation history. It takes as input the cumulative measured binding reward $\scriptstyle c _ { \tau , t } ,$ trajectory depth $d _ { \tau , t } ,$ the mutated position represented as a one-hot vector $\mathbf { e } _ { i _ { \tau , t } }$ , and the antibody region represented as a one-hot vector $\mathbf { e } _ { \rho _ { \tau , t } }$ over FR1–FR4 and CDR1–CDR3. Here, τ denotes the trajectory index, t denotes the mutation step within the trajectory, $i _ { \tau , t }$ denotes the mutated position index, and $\rho _ { \tau , t }$ denotes its antibody region. It outputs a scalar reward that suppresses extreme reward peaks while retaining positive linear agreement with the affinity variation measured by DMS.

$$
R _ { \phi } ( c _ { \tau , t } , d _ { \tau , t } , \mathbf { e } _ { i _ { \tau , t } } , \mathbf { e } _ { \rho _ { \tau , t } } ) = C _ { R } \operatorname { t a n h } \left( \frac { f _ { \phi } ( [ c _ { \tau , t } , d _ { \tau , t } , \mathbf { e } _ { i _ { \tau , t } } , \mathbf { e } _ { \rho _ { \tau , t } } ] ) } { C _ { R } } \right) ,\tag{11}
$$

where $f _ { \phi }$ is a multilayer perceptron. The Pearson-correlation terms used to train RewardNet preserve relative variation but do not determine the absolute scale or offset of its output. The $C _ { R } \operatorname { t a n h } ( \cdot / C _ { R } )$ parameterization bounds the pre-normalized reward $R _ { \phi }$ to $[ - C _ { R } , C _ { R } ]$ , preventing unbounded scale growth. Let $R _ { \phi } ^ { ( b ) }$ denote this bounded output for sample b. We remove batch-dependent offset drift by pinning its batch mean to a fixed target $\mu _ { \mathrm { r e w a r d } } .$ , a training hyperparameter that is independent of the current batch. For a training batch $B ,$ we apply the exact forward-pass normalization

$$
\overline { { R } } _ { \phi , \mathcal { B } } ^ { ( b ) } = R _ { \phi } ^ { ( b ) } - \frac { 1 } { \left. \mathcal { B } \right. } \sum _ { b ^ { \prime } \in \mathcal { B } } R _ { \phi } ^ { ( b ^ { \prime } ) } + \mu _ { \mathrm { r e w a r d } } ,\tag{12}
$$

so that

$$
\frac { 1 } { | B | } \sum _ { b \in B } \overline { { R } } _ { \phi , B } ^ { ( b ) } = \mu _ { \mathrm { r e w a r d } }\tag{13}
$$

holds exactly for every training batch. This transformation adds the same constant to every sample after centering, so it preserves within-batch pairwise differences, ordering, and Pearson correlations. Because the mean shift is applied after the tanh bound, $\overline { { R } } _ { \phi , B }$ is not restricted to $[ - C _ { R } , C _ { R } ]$ , but remains bounded within $[ \mu _ { \mathrm { r e w a r d } } - 2 C _ { R } , \mu _ { \mathrm { r e w a r d } } + 2 C _ { R } ]$ We apply this mean constraint only to RewardNet.

TaxNet. TaxNet predicts a sequence-dependent exploitation penalty. For the sequence produced at step t, we compute the evolutionary sequence score from Appendix E.1 and lower-bound its logarithm at $\ell _ { \mathrm { m i n } } = - 4 . 0 $

$$
\ell _ { \tau , t } = \operatorname* { m a x } \{ \log P _ { \mathrm { s e q } } ( \mathbf { s } _ { \tau , t + 1 } ) , \ell _ { \operatorname* { m i n } } \} .\tag{14}
$$

TaxNet takes $\ell _ { \tau , t }$ as input and outputs a scalar tax that can increase for recurrent or SHM-accessible sequence outcomes, thereby discouraging repeated exploitation of a narrow set of evolutionarily likely mutations. Denoting the learned TaxNet map by $T _ { \psi }$ , its scalar output is

$$
\zeta _ { \tau , t } = T _ { \psi } ( \ell _ { \tau , t } ) .\tag{15}
$$

Unlike RewardNet, TaxNet is not batch-mean normalized; its output is used directly in the meta-reward.

The meta-reward supplied to the policy is

$$
r _ { \mathrm { m e t a } } ^ { ( \tau , t ) } = \overline { { R } } _ { \phi , B } ^ { ( \tau , t ) } - \zeta _ { \tau , t } .\tag{16}
$$

Inner objective. Given the learned meta-reward, the mutation policy is optimized with REINFORCE while remaining close to the frozen SHM prior through KL regularization,

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { i n n e r } } ( \boldsymbol { \theta } ; \boldsymbol { r } _ { \mathrm { m e t a } } ) = \displaystyle - \frac { 1 } { | \mathcal { B } | } \sum _ { ( \tau , t ) \in \mathcal { B } } G _ { \tau , t } \log \pi _ { \boldsymbol { \theta } } ( \boldsymbol { u } _ { \tau , t } \mid \mathbf { s } _ { \tau , t } ) } \\ & { \quad \quad \quad + \displaystyle \frac { \beta _ { \mathrm { K L } } } { | \mathcal { B } | } \sum _ { ( \tau , t ) \in \mathcal { B } } D _ { \mathrm { K L } } \big ( \pi _ { \boldsymbol { \theta } } ( \cdot \mid \mathbf { s } _ { \tau , t } ) \mid \big | \pi _ { \mathrm { p r i o r } } ( \cdot \mid \mathbf { s } _ { \tau , t } ) \big ) , } \end{array}\tag{17}
$$

For discount factor $\gamma \in [ 0 , 1 ]$ , the discounted return is

$$
G _ { \tau , t } = \sum _ { j = t } ^ { T _ { \tau } - 1 } \gamma ^ { j - t } r _ { \mathrm { m e t a } } ^ { ( \tau , j ) } ,\tag{18}
$$

which is used directly in the policy loss. We do not train a value function, subtract a learned state-dependent baseline, or use generalized advantage estimation; consequently, the update has the variance of a Monte Carlo return estimator in addition to the off-policy bias described in Section 4.2. Finally, $u _ { \tau , t } = ( i _ { \tau , t } , a _ { \tau , t } )$ denotes the mutation action consisting of the mutation site $i _ { \tau , t }$ and the target amino acid $\boldsymbol { a } _ { \tau , t }$ . Because $r _ { \mathrm { m e t a } }$ depends on $( \phi , \psi )$ , the inner objective inherits this dependence through its return.

Differentiable inner step. RewardNet and TaxNet are optimized indirectly through their effect on policy learning. We therefore perform one differentiable virtual policy update,

$$
\begin{array} { r } { \theta ^ { \prime } ( \phi , \psi ) = \theta - \eta _ { \mathrm { i n } } \nabla _ { \theta } \mathcal { L } _ { \mathrm { i n n e r } } \big ( \theta ; r _ { \mathrm { m e t a } } ( \phi , \psi ) \big ) . } \end{array}\tag{19}
$$

The computation graph is retained, making $\theta ^ { \prime }$ differentiable with respect to $\phi$ and ψ through $r _ { \mathrm { m e t a } }$ . The virtual update is used only for meta-optimization and is never applied to the actual policy parameters.

Outer meta-objective. RewardNet and TaxNet are updated according to the quality of the virtual policy. The objective balances exploration, fidelity to measured affinity, and consistency with the evolutionary sequence-probability score.

Let $H ( \pi _ { \theta ^ { \prime } } )$ denote the mean Shannon entropy of the virtually updated position distribution,

$$
H ( \pi _ { \theta ^ { \prime } } ) = - \frac { 1 } { | \mathcal { B } | } \sum _ { ( \tau , t ) \in \mathcal { B } } \sum _ { i = 1 } ^ { L } p _ { \theta ^ { \prime } } ( i \mid \mathbf { s } _ { \tau , t } ) \log p _ { \theta ^ { \prime } } ( i \mid \mathbf { s } _ { \tau , t } ) , \qquad H _ { \mathrm { c a p } } = \operatorname* { m i n } \{ H ( \pi _ { \theta ^ { \prime } } ) , H _ { \mathrm { m a x } } \} .\tag{20}
$$

For trajectory $\tau$ of length $T _ { \tau }$ , define the log sequence-probability score of its terminal sequence as

$$
q _ { \tau } = \log P _ { \mathrm { s e q } } ( \mathbf { s } _ { \tau , T _ { \tau } } ) .\tag{21}
$$

For each batch element $b = ( \tau , t ) \in B$ , let

$$
\begin{array} { r l } & { \mathbf { r } _ { \mathcal { B } } ^ { ( b ) } = \overline { { R } } _ { \phi , \mathcal { B } } ^ { ( \tau , t ) } , \mathbf { c } _ { \mathcal { B } } ^ { ( b ) } = c _ { \tau , t } , } \\ & { \mathbf { q } _ { \mathcal { B } } ^ { ( b ) } = q _ { \tau } , \ell _ { \mathcal { B } } ^ { ( b ) } = \ell _ { \tau , t } , } \\ & { \zeta _ { \mathcal { B } } ^ { ( b ) } = \zeta _ { \tau , t } . } \end{array}\tag{22}
$$

Thus, the terminal-sequence log score $\mathbf { q } _ { B } ^ { ( b ) }$ is shared by all steps from the same trajectory, whereas the resulting-sequence score $\ell _ { B } ^ { ( b ) }$ and the remaining quantities are step-specific.

We use the Pearson-correlation fidelity loss

$$
{ \mathcal { L } } _ { \mathrm { f i d } } ( x , y ) = 1 - \rho ( x , y ) , \qquad \rho ( x , y ) = { \frac { \sum _ { i } ( x _ { i } - { \bar { x } } ) ( y _ { i } - { \bar { y } } ) } { { \sqrt { \sum _ { i } ( x _ { i } - { \bar { x } } ) ^ { 2 } } } { \sqrt { \sum _ { i } ( y _ { i } - { \bar { y } } ) ^ { 2 } } } + \varepsilon } } ,\tag{23}
$$

where x¯ and $\bar { y }$ are the means of x and $y ,$ respectively, and ε is a small constant for numerical stability. The outer objective is

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { o u t e r } } \big ( \phi , \psi ; \theta ^ { \prime } ( \phi , \psi ) \big ) = - \lambda _ { H } H _ { \mathrm { c a p } } } \\ & { \phantom { { = } } + \lambda _ { R } \Big [ \mathcal { L } _ { \mathrm { f i d } } ( \mathbf { r } _ { \mathcal { B } } , \mathbf { c } _ { \mathcal { B } } ) + \mathcal { L } _ { \mathrm { f i d } } \big ( \mathbf { r } _ { \mathcal { B } } , \mathbf { q } _ { \mathcal { B } } \big ) \Big ] } \\ & { \phantom { { = } } + \lambda _ { T } \mathcal { L } _ { \mathrm { f i d } } \big ( \zeta _ { \mathcal { B } } , \ell _ { \mathcal { B } } \big ) } \\ & { \phantom { { = } } + \lambda _ { \mathrm { n e t } } \mathcal { L } _ { \mathrm { f i d } } \big ( \mathbf { r } _ { \mathcal { B } } - \zeta _ { \mathcal { B } } , \mathbf { c } _ { \mathcal { B } } \big ) . } \end{array}\tag{24}
$$

The entropy term encourages broader exploration. The fidelity terms constrain RewardNet to preserve affinity information and remain consistent with terminal-sequence prevalence, TaxNet to remain aligned with the floored resulting-sequence score, and the combined meta-reward to remain aligned with measured binding.

Parameter updates. Only the auxiliary-network parameters are updated using the outer objective,

$$
\begin{array} { r } { ( \phi , \psi )  ( \phi , \psi ) - \eta _ { \mathrm { o u t } } \nabla _ { \phi , \psi } \mathcal { L } _ { \mathrm { o u t e r } } \big ( \phi , \psi ; \theta ^ { \prime } ( \phi , \psi ) \big ) . } \end{array}\tag{25}
$$

This is a total derivative: the entropy term differentiates through the virtual update $\theta ^ { \prime } ( \phi , \psi )$ , while the fidelity terms also provide direct gradients to RewardNet and TaxNet.

The mutation policy is then updated separately using a fresh forward pass with the meta-reward detached from the auxiliary networks,

$$
\theta  \theta - \eta _ { \theta } \nabla _ { \theta } \mathcal { L } _ { \mathrm { i n n e r } } \big ( \theta ; \mathrm { s t o p g r a d } ( r _ { \mathrm { m e t a } } ) \big ) .\tag{26}
$$

Detaching the reward prevents the meta-gradient from providing a second optimization path into the policy. Each training iteration therefore consists of a differentiable virtual update for meta-learning followed by a standard REINFORCE update for the policy.

## F Hyperparameters

Table 10 reports the hyperparameters used for the full RTMG model. The four outer-objective weights are $\lambda _ { H } = 3 . 0$ and $\lambda _ { R } = \lambda _ { T } = \lambda _ { \mathrm { n e t } } = 1 . 0$ . RewardNet uses the output cap $C _ { R } = 5 . 0$ and pinned batch mean $\mu _ { \mathrm { r e w a r d } } = 0 . 9 9 3 7$ while the entropy contribution is capped at $H _ { \mathrm { m a x } } = 4 . 0$ . We set the log sequence-score floor to $\ell _ { \mathrm { m i n } } = - 4 . 0$ and the KL coefficient to $\beta _ { \mathrm { K L } } = 0 . 1$

The differentiable inner update and auxiliary-network outer update both use a learning rate of $1 0 ^ { - 3 }$ , whereas the deployed mutation policy uses $2 \times 1 0 ^ { - 4 }$ . Stage 1 uses batch size 256, and all Stage 2 models use batch size 64. The policy is initialized from the Stage 1 Adapted SHM Prior and uses a six-layer Transformer with hidden dimension 256, eight attention heads, feed-forward dimension 1024, dropout 0.1, and maximum sequence length 149 corresponding to the fixed AHo alignment.

## G Compute Cost

RTMG’s bilevel update adds overhead relative to a standard single-level REINFORCE plus KL step because it performs one differentiable inner update to the probe policy $\theta ^ { \prime }$ , followed by an outer gradient update to RewardNet and TaxNet. We measured this overhead directly using identical hardware, batch size 64, and initialization from the Stage 1 Adapted SHM Prior. One RTMG training step required 0.171 s, compared with 0.121 s for plain REINFORCE plus KL. Over 140 steps, the corresponding runtimes were 24 s and 17 s, giving a 1.4× overhead. This is below the 2× overhead suggested by simply counting forward and backward passes.

The lower overhead reflects the asymmetry in parameter counts. The outer gradient updates only RewardNet and TaxNet, which contain 6,177 and 52 parameters, respectively, for 6,229 parameters in total. This is 769× fewer than the 4,788,757 parameters in the mutation policy. Consequently, the second backward pass is inexpensive relative to the additional forward pass and the graph-retaining backward pass implemented with create\_graph=True. RTMG and all Stage 2 baselines use batch size 64, so their reward and diversity comparisons are not confounded by different mini-batch sizes.

Table 10: Hyperparameters for the full RTMG model and mutation-policy Transformer.
<table><tr><td>Component</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="7">RTMG objective</td><td>Entropy weight  $\lambda _ { H }$ </td><td>3.0</td></tr><tr><td>Reward fidelity weight  $\lambda _ { R }$ </td><td>1.0</td></tr><tr><td>Tax fidelity weight  $\lambda _ { T }$ </td><td>1.0</td></tr><tr><td>Net-reward fidelity weight  $\lambda _ { \mathrm { n e t } }$ </td><td>1.0</td></tr><tr><td>RewardNet cap  $C _ { R }$ </td><td>5.0</td></tr><tr><td>RewardNet pinned mean  $\mu _ { \mathrm { r e w a r d } }$ </td><td>0.9937</td></tr><tr><td>Entropy cap  $H _ { \mathrm { m a x } }$  Log-score floor  $\ell _ { \mathrm { m i n } }$ </td><td>4.0 -4.0</td></tr><tr><td rowspan="6">Optimization</td><td>KL coefficient  $\beta _ { \mathrm { K L } }$ </td><td>0.1</td></tr><tr><td>Inner-step learning rate  $\eta _ { \mathrm { i n } }$ </td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Outer learning rate  $\eta _ { \mathrm { o u t } }$ </td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Policy learning rate  $\eta _ { \theta }$ </td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Stage 1 batch size</td><td>256</td></tr><tr><td>Stage 2 batch size</td><td>64</td></tr><tr><td rowspan="6">Policy Transformer</td><td>Transformer layers</td><td></td></tr><tr><td></td><td>6</td></tr><tr><td>Hidden dimension</td><td>256</td></tr><tr><td>Attention heads</td><td>8</td></tr><tr><td>Feed-forward dimension</td><td>1024</td></tr><tr><td>Dropout Maximum sequence length</td><td>0.1 149</td></tr></table>

## H Broader Impact and Ethics

iGC is intended to support antibody research by prioritizing mutation trajectories for subsequent experimental study, but computationally generated candidates should not be interpreted as validated therapeutics or used clinically without laboratory characterization, safety assessment, and regulatory review. Antibody-design methods may also have dual-use implications if applied to harmful targets; responsible use therefore requires target-level biosafety review, controlled experimental validation, and compliance with institutional and legal requirements. This study uses previously collected repertoire, animal, and deep-mutational-scanning datasets and does not recruit human participants or perform new animal experiments.

## I LLM usage

During the course of this research, Large Language Models (LLMs) were utilized as assistive tools to enhance both the manuscript and the technical workflow. Specifically, LLMs were employed to polish the academic writing and improve the overall clarity of the text. Furthermore, we leveraged LLMs to assist in configuring our computational experiments and developing the automated data cleaning pipeline, which significantly streamlined our data preprocessing and analysis.

## J Code and Data Availability

Code and processed experiment configurations are available at https://github.com/PLACEHOLDER/iGC. The original Tang and GCReplay datasets are available from their cited sources subject to the corresponding access and usage

terms; derived splits and metadata needed to reproduce this work are provided in the repository where redistribution is permitted.