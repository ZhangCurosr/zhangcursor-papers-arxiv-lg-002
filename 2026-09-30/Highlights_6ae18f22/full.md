![](images/5772dd502ae78508e14bf62f22e889d2af84a0c94f14805aa8ffb3cfa302cddb.jpg)

Graphical Abstract

Benchmarking graph-based models for in-silico toxicity prediction in drug discovery<sub>v</sub>

Noel Suárez-Barro, Juan Carlos Vidal Aguiar, Manuel Lama Penín

## Highlights

Benchmarking graph-based models for in-silico toxicity prediction in drug discovery

Noel Suárez-Barro, Juan Carlos Vidal Aguiar, Manuel Lama Penín

• A unified benchmarking framework for graph-based toxicity prediction.

• Systematic evaluation of 20+ models under consistent experimental conditions.

• Fair and reproducible comparison across multiple datasets and partitioning strategies.

• Analysis of methodological trends and performance claims in the literature.

• Open-source release of the benchmarking framework to support community research.

# Benchmarking graph-based models for in-silico toxicity prediction in drug discovery

Noel Suárez-Barro<sup>a</sup>, Juan Carlos Vidal Aguiar<sup>a</sup>, Manuel Lama Penín<sup>a</sup>

<sup>a</sup>Centro Singular de Investigación en Tecnoloxías Intelixentes (CiTIUS), Universidade de Santiago de Compostela, 15705 Santiago de Compostela, Spain

## Abstract

Drug discovery is a costly and high-risk process, where toxicity-related failures remain a major cause of attrition in both preclinica and clinical stages. As a result, accurate early prediction of chemical toxicity is essential to reduce downstream costs and improve compound prioritization. In this context, graph deep learning (GDL) has emerged as a powerful paradigm for toxicity prediction, leveraging molecular graph representations to learn directly from chemical structure with improved expressivity over traditional approaches.

Despite the growing number of proposed models, current literature-based comparisons are often dificult to interpret due to inconsistencies in datasets, preprocessing pipelines, and evaluation protocols. To address this limitation, we introduce a unified and standardized benchmarking framework for GDL-based toxicity prediction. We systematically evaluate more than 20 representative approaches under consistent experimental conditions and across multiple datasets and partitioning strategies, enabling a fair and reproducible comparison of model performance. In addition, we complement this empirical study with a structured literature analysis to contextualize existing methodological trends and performance claims. Our results provide a clearer and more reliable assessment of the current state of the field, highlighting both the strengths and limitations of existing graph-based approaches. To support transparency and reproducibility, we release our benchmarking framework as open-source software https://gitlab. citius.gal/noel.suarez/benchtox, allowing the community to evaluate and compare models under consistent conditions.

Keywords: Toxicity prediction, Deep learning, Graph neural network, Benchmark

## 1. Introduction

Development of new pharmaceuticals —usually referred as Drug Discovery— remains a lengthy, costly, and high-risk process in which safety and eficacy must be carefully balanced. Despite continuous technological advances, bringing a drug from initial identification to clinical approval can take over a decade and require investments of several billion dollars [1]. A significant portion of these costs is driven by late-stage failures, many of which are associated with toxicity issues [2]. As a result, early and reliable toxicity assessment has become a central objective in modern drug development, aiming to identify unsafe compounds as soon as possible and thereby reduce downstream attrition.

More broadly, drug discovery encompasses a spectrum of computational tasks that support molecular design and optimization, including target identification, prediction of drug–target interactions and binding afinities, de novo molecule generation, and estimation of key physicochemical and bioactivity properties such as solubility and permeability [3]. Within this landscape, toxicity prediction plays a critical role in prioritizing safer candidates and minimizing reliance on costly experimental assays and animal testing [4].

Toxicity itself is a complex, multifactorial phenomenon arising from the interplay between molecular structure, physicochemical properties, and biological processes across multiple scales. Conventional approaches, including in vitro and in vivo assays, remain essential but are often limited by high costs, long turnaround times, and ethical concerns. Consequently, computational methods —particularly quantitative structure–activity relationship (QSAR) models— have been widely adopted to complement experimental studies by linking molecular descriptors to toxicological outcomes [5]. However, traditional QSAR approaches frequently struggle to generalize across diverse chemical spaces and to capture the nonlinear and context-dependent nature of biological responses [6]. Toxic effects can vary substantially depending on factors such as dose, route of administration, metabolism, and inter-individual or inter-species variability, complicating both label definition and model transferability.

In practice, toxicity prediction for small molecules is typically framed as a supervised learning problem, where models take molecular representations as input and predict toxicity indicators derived from experimental assays, ranging from binary safety labels to continuous dose–response measures. The ultimate goal is to generalize this knowledge to unseen compounds and improve the early identification of adverse efects, which often stem from subtle structural features, reactive metabolites, or unintended of-target interactions [7].

With the emergence of large-scale omics data and advances in Artificial Intelligence, deep learning methods have redefined the landscape of computational toxicology [8, 9]. Among these, graph deep learning (GDL) has emerged as a transformative paradigm [10, 11, 12], leveraging graph-based molecular representations that inherently encode the topology and connectivity of atoms and bonds. Graph neural networks (GNNs) learn data-driven molecular embeddings capable of integrating structural, physicochemical, and contextual biological information, enabling more expressive and generalizable predictive models. Unlike traditional fixed molecular descriptors like fingerprints (FPs), these architectures can automatically infer hierarchical structure–property relationships directly from raw molecular graphs [13].

The application of GDL to toxicity prediction ofers unprecedented opportunities to enhance accuracy, interpretability, and transferability across datasets and endpoints. Recent advances have demonstrated its potential to uncover subtle molecular patterns underlying adverse efects and mechanistic toxicities. This work provides both a comprehensive overview of graph-based deep learning methods for toxicity prediction and an extensive comparative evaluation of state-of-the-art approaches. Through systematic and reproducible benchmarking under equal experimental conditions of the most influential and widely adopted GDL architectures across multiple public toxicity datasets, we assess their predictive performance, robustness, and generalization capabilities. By combining a critical synthesis of current research with rigorous empirical analysis, this study delineates both the progress achieved and the limitations that define the current landscape of AI-driven drug safety prediction, while highlighting promising directions for future research in computational toxicology.

The main contributions of this work are summarized below:

• We provide a unified synthesis of existing approaches and architectures, formalizing them within a common framework that defines the methodological landscape for developing deep learning models in toxicity prediction, as well as the types of information typically employed.

• We conduct a comprehensive analysis and systematic classification of the reviewed methods according to the criteria established within this framework.

• We systematically re-execute up to 20 state-of-the-art models using their publicly available implementations under equal and controlled experimental conditions, enabling fair and reproducible benchmarking and alleviating the need for future researchers to independently reproduce these baselines for comparison.

• We release an open-source benchmarking framework to the scientific community, designed to facilitate the evaluation of new approaches and their direct comparison against the most influential models in the field.

The remainder of the paper is structured as follows. Section 2 describes the literature search methodology adopted to ensure an unbiased review. Section 3 presents the related work, covering both early foundational studies and existing frameworks and benchmarking eforts. Section 4 introduces the proposed unifying framework and provides a structured analysis and classification of the reviewed approaches. Section 5 details the benchmarking protocol, including the experimental settings and evaluation criteria. Section 6 reports and discusses the results of the empirical study. Finally, Section 7 concludes the paper and outlines future research directions.

## 2. Search methodology

The objective of this review is to provide a comprehensive and structured overview of recent advances in toxicity prediction for drug discovery, with particular emphasis on deep learning methods and graph-based models. Unlike traditional systematic literature reviews, toxicity-related research spans a highly diverse and heterogeneously described landscape: toxicity endpoints vary widely across studies, the terminology used is non-standardized, and relevant works are often indexed under disparate domains such as pharmacology, cheminformatics, systems biology, ADMET modelling, or risk assessment. In practice, this heterogeneity resulted in two major challenges: (i) broad keyword queries returned an unmanageable number of non-relevant publications, and (ii) narrowing the queries led to the exclusion of core papers that did not explicitly use the expected terminology.

Given these domain-specific constraints, a fully systematic protocol —e.g., PRISMA— was not appropriate nor reproducible without substantial loss of relevant works. Instead, this review follows a structured literature mapping approach informed by principles of the PRISMA-ScR (Scoping Review) framework [14]. Scoping reviews are particularly suited for research areas with difuse terminology, evolving methodologies, and heterogeneous study designs —conditions that align with the current toxicity prediction landscape—. Following this rationale, we adopted a transparent and methodical process adapted to the characteristics of the field based on the PRISMA-ScR methodology.

Search strategy and sources. The literature search covers peer-reviewed work published from January 2022 to January 2026 and follows a multi-stage retrieval and filtering process, as illustrated in Figure 1.

The initial stage employed a keyword-based query across multiple academic databases and publication repositories, including ACM Digital Library, IEEE Xplore, PubMed, ScienceDirect, Scopus, SpringerLink Google Scholar and Linknovate. The search query combined task-related terms with modelling-oriented terminology as can be seen on the figure. The results of the query came to almost 400 records after deduplication.

In the second stage, the retrieved records were screened for the presence of dataset-related keywords in the full text. Specifically, studies were retained if they referenced any of the following widely adopted toxicity benchmark datasets or endpoints: clintox, ames, hepatotoxicity, dili, herg, carcinogenicity, toxic, toxcast, or tox21. This filtering step yielded 71 papers.

Eligibility criteria. The third stage consisted of a full-text eligibility assessment. Studies were included only if they met the following criteria: (i) the proposed approach fell within the scope of graph-based deep learning methods for toxicity prediction in drug discovery —excluding works leveraging diferent methods or focusing exclusively on environmental toxicity, ecotoxicology, regulatory toxicology or wet-lab assays without computational modelling—, and (ii) the study included an original model proposal for in-silico prediction, leaving apart literature reviews or comparative works. This cutted by half the articles that made it to the review.

![](images/2bda8061b082281d7ada634117ac84d4c69d6e78b2cb88bf1c499c1b9a674bc1.jpg)  
Figure 1: Flow diagram for literature search

Lastly, a final stage was conducted to select those approaches with fully executable codebases and reproducible results to take part in the benchmark. This is detailed in the corresponding section later in this work.

## 3. Related work

Early applications of deep learning to computational toxicity prediction can be traced back to DeepTox [15], which demonstrated that multi-task deep neural networks trained on high-dimensional molecular descriptors could substantially outperform traditional machine-learning approaches in the Tox21 Data Challenge [16]. This work established that hierarchical feature learning and end-to-end optimization were particularly well suited to capturing complex structure–activity relationships, and it efectively positioned deep learning as a new paradigm for in silico toxicity assessment.

In the years that followed, a first wave of deep models for molecular property prediction began to move away from hand-crafted descriptors toward learned representations, including early convolutional and recurrent architectures operating on SMILES strings and graph-based networks that treated molecules explicitly as labeled graphs. These models provided important baselines for toxicity prediction tasks and showed that learned representations could transfer across endpoints, inspiring a series of architectures that progressively tightened the coupling between chemical structure and predictive performance.

A major milestone in graph-based molecular modeling was AttentiveFP [17] introduced an attention-based message-passing framework in which both atom-wise and molecule-level attention mechanisms were used to weight the contribution of diferent substructures, yielding interpretable and highly expressive molecular fingerprints. This architecture achieved competitive performance on multiple toxicity datasets and helped establish attention-enhanced graph neural networks as reference baselines for subsequent work, particularly in settings where model interpretability and substructure attribution are critical.

Around the same period, GROVER [18] introduced graph transformers combining message-passing with a self-supervised pretraining on large unlabeled corpora of molecular graphs to learn rich, task-agnostic node and graph embeddings. By jointly leveraging node-level and edge-level contextual information, GROVER set strong baselines on a variety of molecular property benchmarks, including toxicity-related endpoints, and popularized pretraining and fine-tuning pipelines that later became standard in the field. An example of this is SSL-GCN [19], explicitly tailored for toxicity prediction, leveraging self-supervised objectives to better capture structure–toxicity relationships in molecular graphs.

Building on these foundations, subsequent work increasingly focused on enriching molecular graphs with domain knowledge [20, 21, 22, 23] and 3D structural information [24, 25, 26, 27, 28, 29], while also exploiting shared representations across multiple tasks [30, 31] and more modern networks [32, 33] and training strategies [34, 35, 36]. These approaches aimed to better capture complex structure–property relationships and further improve upon established toxicity prediction performance.

As all these architectures proliferated, a growing need emerged for standardized benchmarks and curated datasets to enable fair and reproducible comparison across toxicity prediction methods. Many of the cited models compare themselves using the established benchmark for molecular property prediction MoleculeNet [37], released as part of the DeepChem library [38]. MoleculeNet represented a major step forward in standardizing dataset curation and evaluation protocols for a broad range of molecular machine learning tasks. It includes a variety of physicochemical, biochemical, and bioactivity datasets, establishing unified metrics and splitting strategies that have become widely adopted in the field. However, despite its breadth, MoleculeNet was not specifically designed to ad dress toxicity prediction and therefore provides only a limited representation of toxicological endpoints.

Beyond standardized benchmarks like MoleculeNet, several recent reviews [39, 40, 2] have sought to synthesize progress in computational toxicity prediction by compiling results across a diverse range of classical and modern methods, typically covering a broad spectrum of algorithms such as random forests, support vector machines, gradient boosting and generic neural network architectures, along with relevant proposals. These works usually compare models indirectly, by summarizing and juxtaposing the performance metrics reported in the original publications, often across heterogeneous datasets and evaluation protocols. In contrast, the present work goes beyond a narrative review by introducing a dedicated benchmarking framework that focuses on graph-based deep learning for molecular toxicity prediction and evaluates all considered models under a unified experimental setup, ensuring a fair and controlled comparison across architectures.

More recently, dedicated resources such as TOXRIC [41] have addressed the need for clean and standard data sources of these machine learning methods. TOXRIC is a recently introduced resource that combines a comprehensive toxicology database with machine-learning–ready datasets and baseline benchmarks, providing an extensive suite of compounds from diferent toxicity categories and endpoints that greatly facilitates data access and standardization. Following its release, many studies in computational toxicity prediction [42, 30, 43] have adopted its curated data resources for both training and evaluation. Our contribution likewise builds on the TOXRIC versions of several widely used toxicity datasets, which we select as the core of our benchmark. Apart from data curation, TOXRIC ofers baseline benchmarks constructed by systematically combining multiple feature types with a small set of typical machine learning algorithms, together with visualization tools to inspect molecular representations and benchmark results for each endpoint.

Complementary to resource-oriented eforts such as TOXRIC, our work focuses on the systematic evaluation of state-of-the-art graph-based models for molecular toxicity prediction under a unified and reproducible framework. Rather than exploring combinations of generic feature representations and baseline algorithms, we design a specialized benchmark that standardizes datasets, endpoints, data splitting strategies, and evaluation protocols. This enables rigorous, head-to-head comparisons across architectures while reducing variability introduced by inconsistent experimental setups.

Specifically, our benchmark builds upon curated TOXRIC datasets and extends prior initiatives such as MoleculeNet by incorporating a broader and more representative set of toxicological endpoints, alongside additional splitting strategies that better reflect real-world generalization scenarios. We further provide open-source implementations, detailed experimental protocols, and comprehensive results for all evaluated models, ensuring transparency and facilitating straightforward reuse.

Overall, this work establishes a robust and extensible benchmarking framework tailored to graph-based toxicity prediction, addressing current limitations in comparability and reproducibility. By consolidating modern architectures within a controlled experimental setting, it provides a reliable foundation for future methodological advances and fair performance assessment in computational toxicology.

## 4. Analysis and classification

In Figure 2, we provide an overview of the general framework underlying the works reviewed in this manuscript. The problem is formulated starting from a small molecule, typically represented in SMILES format, for which an unknown property is of interest, in this case toxicity.

This molecule can be represented in multiple ways. The most popular are fingerprints, a binary vector descriptor that captures information about the molecule’s structure and functional groups, and molecular graphs, where each node corresponds to an atom and each edge to a bond. It is also common for approaches to incorporate additional knowledge in the form of a higher-level graph encoding functional groups and strucutal motifs, 3D information, or other known molecular properties.

A subset of this information is fed into a deep learning model, typically graph-based to align with the molecule’s nature. In multimodal models that integrate diferent types of information, each view is usually processed in a separate branch with its own encoder, and the branches are then combined via concatenation or another fusion mechanism. A pooling operation is commonly required to move from a fine-grained representation —at the atomic or functional level— to a single molecular embedding. Once we have this dense representation, a simple predictor network is usually employed to obtain the target value.

During training, these models are optimized to correctly predict toxicity values for a known set of molecules —the training set— and validated on a set of molecules not seen during training —the test set—. Many approaches separate the training process into two stages. First, a pretraining stage on a very large set of molecules to learn to represent this data efectively, typically by optimizing a generic or self-supervised task to obtain a representative embedding —in a task called representation learning—. Then, this model is fine-tuned on a task-specific toxicity dataset, smaller in size, with the goal of achieving accurate predictions by leveraging prior domain knowledge.

In this section, we detail the diferent components employed by the reviewed architectures. A structured summary of how these components are implemented across approaches can be found in Table 1. Each subsection further examines the individual components, outlining the design choices adopted in the literature.

## 4.1. Primary task

First, it is necessary to frame the reviewed architectures according to the primary task or application domain they target. This perspective provides a high-level categorization that helps contextualize the diferent design choices discussed throughout this section, distinguishing between general-purpose molecular representation learning, toxicity-focused models, and domainspecific approaches.

## 4.1.1. Molecular property prediction

This line of work focuses on molecular encoders that learn general-purpose representations of small molecules, independently of any single downstream endpoint. These models are usually evaluated on broad benchmarks that include diverse physicochemical and bioactivity tasks, sometimes with a subset of toxicity-related datasets, like MoleculeNet. Their main goal is to produce transferable embeddings that can be reused across multiple prediction problems with minimal task-specific adaptation.

![](images/844aab4ebb3facbec387a49e15552bcb11870bbb592ddc1cfd2a7a8c13654ccb.jpg)  
Figure 2: Components of the property prediction general framework. 1. Small molecule provided in SMILES format 2. Diferent data representations that can be inputed to the deep learning model as complementary information or separate views in multimodal settings 3. Graph-based neural model that takes at least one graph as input and generates a dense representation for the provided features 4. Pooling mechanism used to summarize the node-level representations into a single embedding representing the molecule 5. Neural network acting as prediction head for generating the target value from a deep representation of the molecule containing presumably all relevant information for the task 6. Method applied to optimize the deep learning model for the task at hand. Diferent learning strategie can be leveraged for this purpose, such as dividing training into representation learning and fine tuning stages.

## 4.1.2. Toxicity prediction

Toxicity prediction can be viewed as a particular case of molecular property prediction, but it is often treated as a distinct task due to its central role in risk assessment and drug safety. Within this setting, models are explicitly optimized to distinguish between toxic and non-toxic compounds or to estimate toxicity-related endpoints, frequently in the context of ADMET profiling, where safety and eficacy must be balanced simultaneously. Compared with general molecular property models, these approaches typically focus on toxicity-relevant endpoints and design choices that favour interpretability and reliability in safety-critical scenarios.

## 4.1.3. Domain-specific toxicity

Toxicity covers a broad spectrum of efects, from organspecific and systemic toxicity to ecological and environmental endpoints. As a result, there is an increasing interest in domainspecific models tailored to particular forms of toxicity, such as hepatotoxicity, cardiotoxicity, or acute and ecotoxic efects. These models are usually trained and evaluated on specialized datasets and incorporate inductive biases or features that reflect the underlying biological mechanisms of the target endpoint, trading some degree of generality for improved performance and relevance within their specific domain.

Although these models are typically evaluated on a single type of toxicity using specialized datasets, some architectures exhibit the potential to generalize beyond their original domain. For this reason, we also consider them within our benchmark, assessing their performance against other state-of-the-art approaches across a broader range of toxicity prediction tasks.

This task-oriented categorization establishes the context in which architectural design choices should be interpreted, and guides the component-wise analysis that follows.

## 4.2. Input data

Molecular features form the essential interface between chemical structures and computational toxicity prediction models, encoding molecular information in formats that preserve physicochemical and topological properties while enabling efficient machine learning. These features are systematically classified by dimensionality, beginning with zero-dimensional (0D) descriptors that capture only constitutional information —such as molecular weight, atom counts, and basic elemental composition— ofering simplicity but limited structural insight. One-dimensional (1D) descriptors extend this by incorporating linear substructural elements, including functional group frequencies, ring counts, and chain lengths, often represented through string-based notations like SMILES, that sequences atoms and bonds but remain sensitive to ordering conventions [55].

Table 1: Classification of reviewed approaches
<table><tr><td>Ref.</td><td>Approach</td><td>Input</td><td>Data Representation</td><td>Encoder</td><td>Pooling</td><td>Learning Strategy</td><td>Primary Task</td></tr><tr><td>[17]</td><td>AttentiveFP</td><td>2D</td><td>AG</td><td>GAT</td><td>GRU</td><td>Classical</td><td>MPP</td></tr><tr><td>[18]</td><td>Grover</td><td>2D</td><td> $\mathsf { A G } + \mathsf { K }$ </td><td>GT</td><td>Attention</td><td>SSL</td><td>MPP</td></tr><tr><td>[19]</td><td>SSL-GCN</td><td>2D</td><td>AG</td><td>GCN</td><td>Max-pool.</td><td>SSL</td><td>TP</td></tr><tr><td>[34]</td><td>MolCLR</td><td>2D</td><td>AG</td><td>GCN + GIN</td><td>GAP</td><td>SSL, Contrastive</td><td>MPP</td></tr><tr><td>[20]</td><td>KPGT</td><td>2D</td><td> $\mathbf { L G } + \mathbf { F P }$ </td><td>LiGhT</td><td>GAP</td><td>SSL</td><td>MPP</td></tr><tr><td>[25]</td><td>GEM</td><td>3D</td><td> $\mathbf { A G } + \mathbf { L G }$ </td><td>GIN (GeoGNN)</td><td>GAP</td><td>SSL</td><td>MPP</td></tr><tr><td>[35]</td><td>CD-MVGNN</td><td>2D</td><td> $\mathbf { A G } + \mathbf { L G }$ </td><td>NodeGNN, EdgeGNN</td><td>Attention</td><td>Disagreement loss</td><td>MPP</td></tr><tr><td>[24]</td><td>GraphMVP</td><td>3D</td><td>AG</td><td>VGAE</td><td></td><td>SSL</td><td>MPP</td></tr><tr><td>[44]</td><td>RG-MPNN</td><td>2D</td><td> $\mathbf { A } \mathbf { G } + M \mathbf { G }$ </td><td>GAT</td><td>Attention</td><td>Classical</td><td>MPP</td></tr><tr><td>[42]</td><td>NYAN</td><td>2D</td><td> $\mathbf { A G } + \mathbf { F P }$ </td><td> $\mathrm { V G A E } + \mathrm { E x t r a T r e e }$ </td><td></td><td>Pretrain</td><td>MPP</td></tr><tr><td>[22]</td><td>PharmaHGT</td><td>2D</td><td> $\mathbf { A } \mathbf { G } + M \mathbf { G }$ </td><td>HGT</td><td>GRU + Attention</td><td>Classical</td><td>MPP</td></tr><tr><td>[21]</td><td>KANO</td><td>2D</td><td> $\mathbf { A G } + \mathbf { K G }$ </td><td>CMPNN</td><td>Max-pool., GRU</td><td>Contrastive, Prompt</td><td>MPP</td></tr><tr><td>[45]</td><td>GeoDILI</td><td>3D</td><td> $\mathbf { A G } + \mathbf { L G }$ </td><td>GIN (GeoGNN)</td><td>GAP</td><td>Pretrain</td><td>DS</td></tr><tr><td>[27]</td><td>ET-Tox</td><td>3D</td><td>AG</td><td>Equivariant GT</td><td>Attention</td><td>Classical</td><td>TP</td></tr><tr><td>[46]</td><td>FS-GNNTR</td><td>2D</td><td>AG</td><td>GIN + Transformer</td><td>GAP</td><td>Pretrain</td><td>MPP</td></tr><tr><td>[47]</td><td>ToxMPNN</td><td>2D</td><td>AG</td><td>Gated GNNs</td><td>Attention, Max-pool.</td><td>Classical</td><td>TP</td></tr><tr><td>[30]</td><td>MMGIN</td><td>2D</td><td> $\mathbf { A G } + \mathbf { F P }$ </td><td>GIN</td><td>GMP</td><td>Multi-task</td><td>TP</td></tr><tr><td>[36]</td><td>3MTox</td><td>2D</td><td> $\mathbf { L G } / M G + \mathrm { F P }$ </td><td>Transformer</td><td>[CLS]</td><td>SSL, Contrastive</td><td>TP</td></tr><tr><td>[31]</td><td>MPCD</td><td>2D</td><td>AG</td><td>GT</td><td>Attention</td><td>Multi-task</td><td>MPP</td></tr><tr><td>[48]</td><td>AttenhERG</td><td>2D</td><td>AG</td><td>Attention</td><td>Attention</td><td>Classical</td><td>DS</td></tr><tr><td>[23]</td><td>DeepPK</td><td>2D</td><td> $\mathrm { A G } + \mathrm { F P } + \mathrm { K }$ </td><td>D-MPNN</td><td></td><td>Classical</td><td>TP</td></tr><tr><td>[49]</td><td>MolProp</td><td>2D</td><td> $\mathbf { A G } + \mathbf { S M I L E S }$ </td><td> $\mathrm { G A T } + \mathrm { B E R T }$ </td><td>Attention</td><td>Pretrain</td><td>MPP</td></tr><tr><td>[50]</td><td>hERGAT</td><td>2D</td><td> $\mathrm { A G } + \mathrm { F P } + \mathrm { f e a t } .$ </td><td> $\mathbf { G A T } + \mathbf { G R U }$ </td><td>Attention + GRU</td><td>Classical</td><td>DS</td></tr><tr><td>[51]</td><td>DILI_GATNN</td><td>2D</td><td> $\mathbf { A G } + \mathbf { F P }$ </td><td> $\mathrm { G A T + D N N }$ </td><td>GAP</td><td>Pretrain</td><td>DS</td></tr><tr><td>[26]</td><td>DumplingGNN</td><td>3D</td><td>AG</td><td> $\mathrm { G A T } + \mathrm { G r a p h S A G E }$ </td><td>GAP</td><td>Classical</td><td>MPP</td></tr><tr><td>[28]</td><td>SynthMol</td><td>3D</td><td> $\mathbf { A G } + \mathbf { F P }$ </td><td> $\mathrm { G A T + U n i M o l }$ </td><td>GRU</td><td>Pretrain</td><td>MPP</td></tr><tr><td>[29]</td><td>FATE-Tox</td><td>3D</td><td> $\mathbf { A G } + M G + \mathbf { F P }$ </td><td>SE(3)-Transformer</td><td>Sum-pooling</td><td>Classical, Multi-task</td><td>TP</td></tr><tr><td>[52]</td><td>Samar Monem</td><td>2D</td><td>AG</td><td>Attention</td><td>Concat, Attention</td><td>Classical</td><td>TP</td></tr><tr><td>[53]</td><td>ToxKG</td><td>2D</td><td> $\mathrm { K G } + \mathrm { F P }$ </td><td>Heterogeneous GraphGPS</td><td></td><td>Classical</td><td>TP</td></tr><tr><td>[54]</td><td>AMPred-LWN</td><td>2D</td><td> $\mathbf { A G } + M G + \mathbf { F P }$ </td><td>GAT, GIN, Mamba</td><td>Mamba</td><td>Pretrain</td><td>DS</td></tr></table>

Approaches are presented in chronological order. Abbr.: AG = Atom Graph, LG = Line Graph, MG = Motif Graph, K = Knowledge, KG = Knowledge Graph, FP = Fingerprint. MPP = Molecular Property Prediction, TP = Toxicity Prediction, DS = Domain-Specific tasks such as hepatotoxicity or cardiotoxicity.

Two-dimensional (2D) descriptors provide substantially richer information by modeling molecules as undirected graphs, thereby encoding adjacency, connectivity, and topological features through graph invariants, connectivity indices, and substructure-based fingerprints. Three-dimensional (3D) descriptors further enhance expressiveness by incorporating spatial coordinates for each atom, enabling the quantification of steric efects, pharmacophoric arrangements, molecular volume, and surface properties that are crucial for predicting receptor interactions, metabolic liabilities, and organ-specific toxicities, albeit at the cost of conformational sampling [55].

This progression from 0D/1D simplicity toward 2D topological and 3D geometric fidelity directly supports the requirements of graph neural networks, which natively process molecular graphs and conformers to learn adaptive, hierarchical embeddings that surpass the limitations of fixed-length descriptors. The predominance of 2D graph-based representations in modern toxicity prediction frameworks, sometimes enriched with 3D information, is coherent with the choice of graph deep learning architectures that are the focus of this review.

## 4.3. Data representation

Molecules can be represented in multiple ways, each providing a diferent view of the same underlying structure, depending on which features are deemed most relevant for the task. Some of these representations are complementary, while others capture fundamentally diferent perspectives.

## 4.3.1. Molecular fingerprints

Molecular fingerprints are a specific class of molecular descriptors that encode chemical structures into fixed-length vectors that can be directly processed by machine-learning models. Depending on the implementation, these vectors can be purely binary —indicating presence/absence of features— or contain integer counts of fragments or motifs, providing more nuanced information on feature multiplicity. Within the broader context of domain knowledge and molecular representation, fingerprints thus act as a bridge between structural chemistry and statistical learning by transforming 2D or 3D structural information into machine-readable features that have been widely used in QSAR and read-across approaches for chemical safety assessment.

Classical key-based fingerprints such as MACCS encode the presence or absence of a predefined set of substructures —e.g., specific ring systems, heteroatoms, functional groups— into fixed-length binary vectors [56]. These fingerprints are computed by matching each key pattern against the molecular graph and setting the corresponding bits, which makes them highly interpretable and eficient for similarity searching, diversity analysis and as a baseline representation in toxicity QSAR models. For instance, MACCS keys combined with machine-learning algorithms have been used to predict acute oral toxicity with good performance [57]. Dictionary- or path-based fingerprints like the PubChem 2D fingerprint [58] or the RDKit substructure/path fingerprints [59] follow a similar principle but rely on larger and often more exhaustive sets of paths or fragments, yielding longer bitstrings with broader coverage of structural motifs that are particularly suitable for virtual screening, clustering and similarity-based read-across in toxicology datasets.

Circular fingerprints such as ECFP [60] and Morgan fingerprint [61] encode local atomic environments by iteratively hashing atom-centered neighborhoods up to a specified radius. In this scheme, each atom’s environs —defined by atom type, neighboring atoms, bond orders and sometimes additional features— are converted into identifiers that are finally folded into a binary or count vector. Since they highlight local substituent patterns and their connectivity, circular fingerprints are particularly efective for capturing structure–toxicity relationships, supporting high-performing classification and regression models, scafold hopping and detection of subtle toxicophoric patterns that may not be captured by simple substructure keys [62]. Avalon fingerprints implement a flexible, hash-based encoding of paths and features that can be configured in size and detail, generating binary or count vectors from combinations of atom and bond features. This tunability makes Avalon attractive for benchmarking models across multiple representation sizes or when optimizing trade-ofs between dimensionality and predictive performance. Furthermore, past research has shown that Avalon molecular fingerprints excel in multiendpoint acute toxicity tasks [42].

Pharmacophore-oriented fingerprints, including ErG-type encodings [63, 64], represent molecules in terms of pharmacophoric features such as hydrogen-bond donors and acceptors, aromatic centers, charged groups and hydrophobic regions, along with their topological or 3D distances [65]. These fingerprints are typically derived from conformer ensembles or pseudo-3D representations and are particularly well suited for modeling endpoints that depend on specific spatial arrangements of features —e.g., target-mediated toxicity or of-target binding—, as illustrated by ErG-based models that successfully classify ligands for E3 ligases and other pharmacological targets [66, 67].

In the context of toxicity prediction, the choice of fingerprint determines which aspects of chemical space and mechanistic information are emphasized. Key-based and dictionary fingerprints are advantageous when interpretability, substructure alerts and fast similarity search are priorities, for example to rationalize structural alerts associated with mutagenicity or hepatotoxicity [68]. Circular fingerprints often deliver superior predictive performance in machine-learning benchmarks [62], making them a common default for large-scale toxicity classification and regression tasks. Pharmacophore and 3D-aware fingerprints become particularly relevant when toxicity is mediated by well-defined ligand–target interactions where spatial feature arrangements are critical. Approaches such as MMGIN [30] and SynthMol [28] integrate multiple complementary fingerprint representations —including MACCS, pharmacophore ErG, and PubChem fingerprints— in an attempt to capture diverse molecular features and enhance the informational richness available to the model.

Even so, all fingerprints are expert-designed automaticallygenerated representations that capture incomplete or partial information about the molecule that might not be relevant or optimal for the problem at hand. Despite their popularity and utility, specially by facilitating the incoporation of domainspecific expertise unknown to computer scientists, these techniques often fall short to archive the desired predictive perfomance for challenging tasks. For this reason, current eforts are directed toward generating richer context-aware representations through dynamic, self-learned embeddings that move beyond static molecular encodings [17, 69].

## 4.3.2. Molecular graphs

Molecules are commonly represented as atom-centered graphs, where nodes correspond to atoms and edges to chemical bonds [19, 24, 34, 70, 52]. In this representation, both nodes and edges are associated with feature vectors encoding physicochemical properties, such as atom type, hybridization state, formal charge, or bond type and aromaticity. These features provide the basis for message passing in graph-based models.

An alternative formulation focuses on bond-centered representations through the use of line graphs [33], where nodes represent bonds and edges encode adjacency between bonds [20, 25, 35]. As illustrated in Figure 3, this transformation shifts the modeling perspective from atoms to interactions between bonds, enabling the model to more explicitly capture patterns related to bond environments, connectivity, and local structural arrangements. This is particularly relevant for toxicity prediction, where many mechanisms are driven by specific substructures or reactive configurations —such as conjugated systems or electrophilic groups— that are more naturally characterized at the bond level than at the level of isolated atoms.

The new graph will have a node for each pair of bonded atoms and edges between nodes sharing atoms. This type of representation is leveraged by KPGT [20] and, implicitly, by other approaches such as CD-MVGNN [35], GEM [25], GeoDILI [45] and 3MTox [36], which incorporate bond-level graphs into their modelling. By operating on bonds, these methods can better model how local structural contexts influence chemical behavior, leading to more informative representations for downstream prediction tasks.

Furthermore, bond-centered modeling facilitates the incorporation of geometric information [33]. For example, GEM and GeoDILI leverage this representation to include bond angles as edge features, since angles are defined between pairs of adja cent bonds [25]. This provides a natural mechanism to encode 3D structural information, allowing the model to account for molecular geometry and spatial constraints that are often critical for accurately predicting toxicity [45].

![](images/a93c030ef631f09871bca7033443aa92b273ea4b3ae42183189774cc63e44124.jpg)  
Figure 3: An illustrative example of the transformation of a molecular graph to a molecular line graph [20].

## 4.3.3. Motif-level graphs

Some approaches incorporate domain knowledge by introducing higher-level graph representations that capture structural motifs, in an attempt to approximate functional groups and toxicophores. These motif-level graphs aim to abstract recurring chemical patterns that are often associated with specific biolog ical or toxicological properties.

There are multiple strategies to construct such representations. Some methods rely on predefined lists of substructures, as in AMPred-LWN [71], RG-MPNN [44] or DeepPK [23], to identify known functional groups or toxicophores. Others apply algorithmic decomposition techniques, such as BRICS fragmentation [72] in PharmHGT [22], to partition molecules into chemically meaningful components. Alternative approaches, such as 3MTox [36], focus on specific structural elements like ring systems. By operating at this higher level of abstraction, these models can capture semantically meaningful patterns that may be less evident at the atom level.

## 4.3.4. 3D information

Beyond topological representations, the three-dimensional (3D) structure of molecules plays a crucial role in determining their properties, including toxicity. This additional information can significantly enhance the model’s ability to learn structure–property relationships.

Molecular geometry arises from the balance of attractive and repulsive forces between atoms, leading to specific spatial conformations. In practice, molecules can adopt multiple conformations —known as conformers— which correspond to diferent local minima in the energy landscape. These conformers can be generated using computational methods such as forcefield optimization or more accurate quantum chemical calculations.

Some approaches explicitly incorporate 3D coordinates or distance-based features into graph models, enabling the capture of geometric relationships such as interatomic distances and angles. This is the case from GEM, GeoDILI, GraphMVP [24],

ET-Tox [27], DumplingGNN [26], FATE-Tox [29] and Synth-Mol [28], which does so by leveraging pre-trained embeddings from the Uni-Mol framework [73].

## 4.3.5. Molecular properties

In addition to structural representations, models may incorporate global molecular descriptors, which encode known physicochemical properties at the molecule level. These include features such as molecular weight (MW), octanol–water partition coeficient (logP), topological polar surface area (TPSA), or the number of hydrogen bond donors and acceptors. Such descriptors, often referred to as 0D representations, provide complementary information that can be directly related to toxicity, for example through their influence on bioavailability, membrane permeability, or reactivity, all of which are closely connected to toxicological behavior.

One approach that incorporates this type of information is hERGAT [50], which has been proposed for hERG blockade prediction, a setting in which cardiotoxicity is strongly influenced by global physicochemical properties and substructural features associated with channel inhibition.

This type of information has also been incorporated in for hERG blockade prediction, a setting in which cardiotoxicity is strongly linked to molecular properties and substructural determinants of channel inhibition [50].

There are also approaches that explore 1D representations, such as SMILES strings, by processing them using sequential or convolutional architectures. However, these meth ods fall outside the scope of this work, as we focus on approaches that leverage richer 2D and 3D representations incorporating topological and geometric information, which are particularly relevant for molecular toxicity prediction. In this context, MolPROP [49] is a representative multimodal ap proach that combines a graph-based molecular representation with a SMILES-based sequence branch encoded using Chem-BERTa [74], thereby integrating complementary structural and SMILES-derived embeddings for property prediction.

## 4.4. Encoder model

The choice of encoder depends on the type of molecular representation and the specific view it provides. For fixed-size representations such as molecular fingerprints or handcrafted molecular descriptors, simple Feed-Forward Networks (FFNs) are typically employed, as they can efectively transform these vectorized inputs into dense latent representations [75].

In contrast, when molecules are represented as graphs, Graph Neural Networks (GNNs) have emerged as a powerful framework for learning over such structured data, enabling the extraction of complex relational and structural information.

## 4.4.1. Message-Passing Neural Networks

Message-Passing Neural Networks (MPNN) are the formal framework that unifies virtually all GNN architectures [76, 77]. The idea is simple: each node sends "messages" to its neighbors, collects messages from its neighbors, and uses them to update its own representation. By this simple trick, the framework enforces an inductive bias inside the network and injects the prior knowledge that "connected things are related".

MPNNs can be dissected into three operations. First, the MESSAGE function (1) creates a "message" from neighbour u to target v using their features and edge attributes. Then, the AGGREGATE function (2) combines all incoming messages. This operation must be permutation-invariant, such as the sum, mean, max or some attention-based operation. Last, the UPDATE function (3) merges the aggregated message with node’s current features to obtain an updated representation. We can formalize the procedure with the following equations:

$$
m _ { \nu  u } = \mathrm { M E S S A G E } ( h _ { u } ^ { ( l ) } , h _ { \nu } ^ { ( l ) } , e _ { u \nu } )\tag{1}
$$

$$
M _ { \nu } = \mathrm { A G G R E G A T E } ( \{ m _ { \nu  u } \mid u \in N ( \nu ) \} )\tag{2}
$$

$$
h _ { \nu } ^ { ( l + 1 ) } = \mathrm { U P D A T E } \left( h _ { \nu } ^ { ( l ) } , M _ { \nu } \right)\tag{3}
$$

where $h _ { u } ^ { ( l ) }$ and $h _ { \nu } ^ { ( l ) }$ are the hidden representation for nodes u and v in the l-th layer, $e _ { u \nu }$ corresponds to the features for the u → v edge, N(v) is the local set of neighbours of v and <sup>(l)</sup> refers to the current layer in the network.

Every GNN architecture follows this pattern and divert only in how they implement these fundamental operations.

## 4.4.2. Graph Convolutional Network

Graph Convolutional Networks (GCN) were the first foundational model for modern GNNs [78]. In this implementation, the message is composed of the representation of the node scaled by a normalization factor $1 / \sqrt { d _ { u } \cdot d _ { \nu } }$ where $d _ { u }$ and $d _ { \nu }$ represent the degree of the nodes, including themselves. This ensures stable learning across graphs with varying degree distributions and prevents high-degree nodes from dominating. The AGGREGATE function corresponds to the mean operation, and the UPDATE is carried out through a linear transformation followed by an activation function that introduces nonlinearity, typically ReLU. This is the neural network behind SSL-GCN [19].

## 4.4.3. Graph Isomorphism Network

The Graph Isomorphism Network (GIN) is designed to be the most expressive GNN possible [32]. To this end, node representation is passed as message without any transformation —preserving injectivity— and all messages are summed together in the AGGREGATE step instead of averaged. This is because the sum preserves multiset cardinality, unlike operations such as mean or max.

Finally, the aggregated message M is added to the central node representation scaled by some factor and a Multi-Layered Perceptron (MLP) is applied to learn complex relations following $\mathrm { U P D A T E } = \mathrm { M L P } ( ( 1 + \epsilon ) \cdot h _ { \nu } + M _ { \nu } )$ . MLP weights are shared among nodes, but not among layers. Parameter ϵ can be set to a small constant like 0 (in GIN-0) or be a lernable parameter (in GIN-ϵ). The learnable variant allows the model to automatically adjust the relative importance of the node against its neighbors, while maintaining the injectivity property required for maximum discriminative power. Theoretically, this term ensures that GIN can distingish between certain non-isomorphic graph structures that would otherwise yield identical representations, directly contributing to GIN’s equivalence to the 1- Weisfeiler-Lehman test in expressive power [32].

The key diference between GCN and GIN lies in their aggregation schemes and expressive power. While GCN employs a normalized average aggregation, GIN utilizes an additive aggregation. This allows GIN to preserve relevant structural information that may be lost or diluted by the mean and distinguish a broader class of molecules. The great power of GIN motives its use in GEM [25] and MMGIN [30], whereas MolCLR exploits the complementary characteristics of both architectures [34].

## 4.4.4. Graph Attention Networks

Graph attention networks (GAT) extend message passing by learning how much attention each node should pay to its neighbors, instead of treating all neighbors equally. In these models, a learnable attention mechanism assigns a weight to each neighbor, so that messages from more relevant neighbors have a stronger influence on the updated node representation, and the resulting attention coeficients can often be interpreted as a form of importance score [79]. The subsequent UPDATE step can be implemented in diferent ways, ranging from simple nonlinear activations such as ReLU to more elaborate transformations that combine the attention-weighted aggregation with the previous node state, a design choice that will be revisited in later architectures.

Attention-based GNNs have become very popular in molecular property and toxicity prediction, and variants of graph attention are employed in models such as AttentiveFP, AttenhERG [48], hERGAT, DILI\_GATNN [51], and many more. Althought pure GAT layers are in principle not guaranteed to reach the same level of discriminative power as GINs, they can work better in practice by capturing rich, context-dependent relationships and focusing on the most informative parts of the molecular graph. Many recent architectures therefore combine both paradigms to balance expressive power and flexibility; for example, AMPred-LWN integrates GIN- and GAT-style components within a single model to exploit the advantages of each design [71].

## 4.4.5. Graph Transformer

The Graph Transformer (GT) generalizes the popular transformer architecture proposed in [80] to graph-structured data [81]. Analogous to the original formulation, they employ key, query and value learnable projection matrices to map node representations into distinct subspaces that interact within the selfattention mechanism. This family of architectures has been adopted by approaches like GROVER, MPCD [31] and FS-GNNTR [70].

This design allows each node in the molecular graph to attend to any other node, rather than being restricted to its local neighborhood, enabling the modeling of long-range interactions between distant parts of the molecule. Structural biases can be incorporated into the attention mechanism to encode relevant graph information, such as shortest-path distances or connectivity patterns, and related ideas have been extended to 3D settings as in FATE-Tox, where spatial relationships between atoms are taken into account. Along similar lines, 3MTox employs a standard transformer encoder augmented with distancebased biases between atom pairs, so that interatomic distances complement the attention scores without altering the core architecture. After the attention operation, a FFN combined with residual connections and normalization is applied to obtain the final node representations.

## 4.4.6. Heterogeneous Graph Transformer

While many molecular graphs are treated as homogeneous —where all nodes and edges share the same semantics— more expressive formulations consider heterogeneous graphs, which include multiple types of nodes and edges representing diferent entities and relations within a unified structure. This is particularly useful in molecular settings where diferent components —such as atoms, functional groups, or higher-level motifs— as well as diferent types of interactions, may carry distinct meanings.

Heterogeneous Graph Transformer (HGT) extend the transformer framework to account for this diversity by adapting message passing to the types of nodes and edges involved [82]. Messages exchanged between nodes are modulated by both the type of the source node and the type of the connecting edge, allowing the model to capture more nuanced, relation-specific interactions. In this sense, attention in heterogeneous graphs closely parallels its homogeneous counterpart, but incorporates node-type-dependent weight matrices and edge-type-specific bias terms. Node-type-specific linear transformations and FFNs are further applied to obtain the final representations.

This design enables a more flexible integration of heterogeneous molecular information, leading to richer and more informative representations. PharmHGT [22] exploits this by using heterogeneous nodes and relations to jointly represent atoms together with higher-level fragments or pharmacophore-like patterns, enriching the molecular graph with multi-granular structural information. In contrast, ToxKG [53] applies a heterogeneous graph model to knowledge graphs where nodes encode entities such as chemicals, targets, or genes and edges encode curated domain relations, allowing the integration of external toxicological knowledge into the learned representations.

## 4.4.7. Line Graph Transformer

Line Graph Transformer (LiGhT) combines the flexibility of attention-based models with a molecular representation that emphasizes bond connectivity and local context [33]. This allows the architecture to capture both fine-grained structural cues and more global molecular patterns within a unified framework. In this sense, LiGhT serves as the base architecture featured in KPGT, tailoring the transformer framework to this type of bond-centered input and better exploiting its structural information.

## 4.4.8. Variational Graph Autoencoder

The Variational Graph Autoencoders (VGAE) extends the VAE framework [83] to graph-structured data, enabling unsupervised learning of interpretable latent representations for undirected graphs [84]. VGAE learns latent representations Z from a graph with adjacency matrix A and node features X, using a GCN-based encoder that parameterizes a multivariate Gaussian posterior and a decoder that reconstructs the adjacency matrix via inner-product similarities. Mean and variance of the latent distribution are obtained from a two-layer GCN that shares weights for the first layer $Z \mu , Z \sigma \ = \ \mathrm { G C N } ( X , A ) .$ Then, the decoder reconstructs the adjacency matrix A<sup>ˆ</sup> probabilistically from the latent node embeddings Z. This architecture has been adopted by NYAN, originally introduced in [85] and later revisited in [42] for toxicity, to obtain a dense molecular representation by jointly encoding the graph structure and fingerprint-based information.

## 4.4.9. Multimodality

Some models simultaneously combine diferent ways of representing the same molecule, such as graph structures, fingerpints and 3D information within a single architecture. These are commonly referred to as multimodal models, as they leverage multiple complementary views of the data at once. A typical pattern is to process each modality in a dedicated encoder and then fuse the resulting representations through concatenation, attention, or more sophisticated fusion mechanisms.

For example, MMGIN employs a two-channel architecture that independently processes molecular graphs and fingerprints, learning parallel views of the same compound and later combining them into a unified representation for multitask toxicity prediction. This design allows the model to exploit both topological patterns from the graph and compact, high-level descriptors from the fingerprint, which can be particularly useful when diferent toxicity endpoints are governed by diferent structural cues.

SynthMol adopts a similar multimodal strategy but integrates pre-trained 3D structural features, graph-based atom representations, and molecular fingerprints. By leveraging geometric information alongside graph and descriptor inputs, SynthMol is able to capture richer structure–property relationships relevant to drug safety, including efects tied to conformation and spatial arrangement. As in MMGIN, the diferent modality-specific encoders are ultimately merged via concatenation, preserving the contribution of each source in a single joint representation.

Similarly, AMPred-LWN is a multimodal multi-granularity model that fuses atomic-level graphs, sequences of functional groups, and molecular fingerprints for Ames mutagenicity prediction. It uses enhanced graph neural networks, sequence-aware Mamba-based modules, and a dedicated fusion mechanism to adaptively modulate the relative contribution of each branch and combine information across diferent levels of abstraction, from local atomic neighborhoods to higher-level functional motifs. Other multimodal toxicity models explore more flexible fusion strategies such as attention-based weighting, exemplified by ToxMPNN [47] and Samar Monem [52].

Altogether, these approaches illustrate how multimodal architectures can exploit diverse sources of molecular information —topological, geometrical, and expert-handcrafted features— while preserving the strengths of each representation and enabling more robust and interpretable toxicity predictions.

## 4.5. Pooling

Graph pooling in graph neural networks is a critical operation designed to hierarchically reduce the size of a graph by coarsening it into a compact representation while preserving structural and semantic information, tailored to the irregular topology of graphs in contrast to fixed-grid pooling in CNNs.

Crucially, pooling mechanisms must be permutation invariant —meaning they produce identical outputs for isomorphic graphs under any node relabeling— to ensure that representations depend solely on intrinsic graph topology and features, not arbitrary indexing conventions. This property is foundational, as graphs lack canonical node orderings, and permutation equivariance in preceding message-passing layers necessitates invariance in pooling to maintain consistent graph-level predictions across equivalent structures.

Pooling serves dual roles: (i) within AGGREGATE-UPDATE phases of GNN layers to locally coarsen neighborhoods and mitigate over-smoothing during message passing, and (ii) as a READOUT function to generate fixed-size embeddings for graph-level prediction. This mechanism enables GNNs to process large-scale graphs eficiently and extract discriminative representations for downstream tasks. Some of the most common operations employed as pooling are:

## 4.5.1. Global Average or Max Pooling

Global average pooling (GAP) computes a fixed-size vector by averaging node features across the entire graph, ofering simplicity, diferentiability, and permutation invariance, what makes it popular among multiple approaches. It captures global consensus at the cost of diluting discriminative local signs. Max pooling (MP), conversely, extracts element-wise maxima, emphasizing salient features and exhibiting robustness to noise, often employed in pseudo-max variants for local coarsening. It extracts the most salient motif but ignores broader context that might be relevant. Both are ubiquitous as MPNN aggregators and baseline readouts.

## 4.5.2. Sequence-based pooling: GRU

Sequence-based pooling methods, such as those employing Gated Recurrent Units (GRU), Short-Term Memory (LSTM) networks or Mamba [54] transform node embeddings into an ordered sequence —typically via sorting by node degree or a learned permutation— prior to recurrent processing. This approach introduces temporal dynamics to capture hierarchical or sequential patterns within the graph, yielding a permutationinvariant readout through the final hidden state. While efective for graphs exhibiting latent ordering, these methods usually incur higher computational overhead and rely on heuristic sorting, potentially introducing bias depending on the chosen strategy.

We can distinguish between two primary usages of recurrent architectures within graph neural networks. First, they are commonly employed inside the AGGREGATE-UPDATE step of message-passing layers, where they act as adaptive update functions for node representations. In this setting, the aggregated neighborhood information is first computed through attention or message aggregation mechanisms, for instance as

$$
C _ { \nu } ^ { ( l ) } = \sum _ { u \in N ( \nu ) } a _ { \nu u } ^ { ( l ) } W h _ { u } ^ { ( l ) }\tag{4}
$$

where $a _ { \nu u } ^ { ( l ) }$ denotes the attention coeficient between nodes u and v, and W is a learnable transformation matrix. Rather than directly replacing the previous node state, the GRU combines the aggregated context $\dot { C } _ { \nu } ^ { ( l ) }$ with the prior hidden representation $h _ { \nu } ^ { ( l ) }$ through gated operations:

$$
h _ { \nu } ^ { ( l + 1 ) } = G R U ( C _ { \nu } ^ { ( l ) } , h _ { \nu } ^ { ( l ) } )\tag{5}
$$

which propagates nonlocal efects while filtering noisy or redundant messages during the aggregation process [17, 50]. By adaptively controlling the amount of newly incorporated information, recurrent updates help stabilize message propagation across multiple layers and preserve relevant contextual dependencies between distant nodes.

Second, recurrent architectures can also be employed as READOUT operators for graph-level representation learning. In this setting, node embeddings produced after the final message-passing layer are transformed into a sequence —typically following a heuristic or learned ordering strategy— and sequentially processed by the recurrent unit. The final hidden state is then used as a compact graph representation [17, 21, 22, 28]. This enables the model to capture higher-order structural dependencies and hierarchical interactions between nodes, although at the cost of higher computational complexity and sensitivity to the chosen node ordering, which may afect robustness and scalability [86].

## 4.5.3. Attention-based pooling

Attention-driven pooling employs learnable importance scores to adaptively weight node contributions during coarsening or readout, ensuring permutation invariance through soft max normalization over node scores. Methods such as DifPool and Set2Set illustrate two representative designs: DifPool integrates into the AGGREGATE–UPDATE phases by learning soft cluster assignments that map nodes to supernodes and produce coarsened graphs, whereas Set2Set acts as a READOUT mechanism that iteratively refines a latent state via attention over node features to obtain a global graph representation. Both approaches can highlight salient substructures in molecular graphs, which is particularly relevant in computational toxicology, but their additional parameters and architectural choices often require careful hyperparameter tuning to avoid overfitting.

Several of the reviwed architectures adopt attention-based pooling or readout to focus on toxicity-relevant regions of the molecule. ToxMPNN [47] uses an attention readout over node embeddings to form graph-level representations, enabling the model to emphasize atoms and substructures most associated with toxicity endpoints. AttenhERG, built on the AttentiveFP framework, combines attention along message passing with attention-based readout to improve hERG blockade prediction and to support interpretation by visualizing highly attended atoms and bonds. Models such as MolPROP and hERGAT also rely on attention mechanisms in their graph encoders to compute weighted graph-level summaries. These designs illustrate the variety of strategies —ranging from simple nodelevel attention pooling to more sophisticated cross-branch attention— while the multimodal architectures proposed by [52] use cross-modal attention to pool and fuse information from diferent molecular representations in a task-aware mannerthat can be used to derive expressive, toxicity-informed graph representations.

## 4.5.4. Virtual node pooling and CLS

Virtual node pooling augments the graph with a trainable supernode connected to all graph nodes, iteratively updated via message passing or attention to serve as a learnable global aggregator without explicit coarsening. Similarly, CLS token approaches —borrowed from transformer architectures like BERT [87]— initialize a dedicated token updated through cross-attention over node embeddings, producing a fixed-size summary that integrates global context in an end-to-end diferentiable manner [88].

These strategies —adopted by models like 3MTox— excel in set-to-vector aggregation for graph-level tasks, maintaining permutation invariance via symmetric aggregation, while preserving expressive power. However, they may underperform on strongly hierarchical structures requiring explicit topology reduction [89, 90].

## 4.6. Prediction head

Once a graph-level embedding has been obtained, that is, a dense vector representation summarizing the molecule and presumed to concentrate the information most relevant for the task, the most common strategy is to use a FFN network as a prediction head. In practice, this is typically implemented as a MLP that progressively reduces the dimensionality of the embedding through a series of linear transformations and nonlinear activations, ultimately mapping it to either a scalar value or a small output vector corresponding to the target of interest. This design separates representation learning in the GNN backbone from task-specific prediction, and naturally accommodates both single- and multi-task settings.

In relation to the target value that we are trying to predict, problems can be divided into:

• Classification. These problems aim to predict a discrete label, generally binary, that distinguishes among compounds that have noticeable toxic efects and those considered to be safe. This label is usually calculated using a pre-defined endpoint-specific threshold, derived from expert knowledge, and that is usually unknown for the user of the dataset.

• Regression. Here the prediction target comes in the form of a continuous value of a toxic-related metric, such as LD50 (the dose necessary to kill half of the studied population), LDlo (the smallest amount of a substance that has been reported to cause death) or IC50 (the concentration required to inhibit a biological process by half).

For the purpose of this work we have focused in the first kind of problems, as it is the most common approach in the field and most of the reference datasets come in this fashion. Regression is not as frequent since the obtention of such values for humans would require unpracticable experimentation and are usually obtained for other small mammals in a laboratory setting.

However, although the benchmark has been designed for classification tasks with toxicity prediction in mind, the structure and methods employed could be easily extended for regression problems or transferred to other domains, particularly those pursuing some kind of property prediction starting from a small molecule.

## 4.7. Learning strategy

Last but not least, it is also important to characterize how diferent models are trained and optimized for the given task. This subsection complements the previous components of the framework by emphasizing not only what is being modeled, but also how the corresponding representations are learned.

## 4.7.1. Supervised Learning

Supervised learning remains the cornerstone paradigm for molecular property and toxicity prediction, training graph neural networks on labeled datasets pairing molecular graphs with experimentally measured endpoints such as binding afinities, solubility, or LD50 values. Regression or classification objectives minimize prediction errors via mean squared error or cross-entropy losses, requiring abundant high-quality annotations that are often scarce for rare toxicities.

## 4.7.2. Unsupervised or Self-Supervised Learning

Unsupervised strategies such as VGAE learn latent molecular representations from unlabeled structures by reconstructing graphs, SMILES strings or fingerprints, enabling downstream fine-tuning for property prediction with minimal labels. NYAN follows this paradigm by encoding the molecular graph into a latent space from which it reconstructs the corresponding fingerprint, efectively injecting the information contained in the fingerprint into the learned graph-based representation [85].

On the other hand, self-supervised learning (SSL) defines pretext tasks that do not require manual annotations, such as predicting masked atoms or bonds, distinguishing between real and corrupted subgraphs, or recovering context information from augmented views of the same molecule, thereby encouraging GNN encoders to capture structural invariances that are useful for downstream toxicity prediction. A concrete implementations of this idea is foud in GROVER, which employ nodeand graph-level pretext tasks lik motif detection to learn chemically meaningful embeddings that can later be fine-tuned on endpoint-specific toxicity datasets.

## 4.7.3. Knowledge-Driven Learning

Knowledge-driven paradigms infuse domain expertise into molecular learning by incorporating physics-informed losses, quantum-chemical descriptors, or pre-established chemical and pharmacophoric knowledge, particularly in settings with sparse experimental data. Hybrid strategies can enforce chemically motivated constraints or exploit equivariant GNNs that preserve molecular symmetries, guiding optimization toward physically and chemically plausible predictions in extrapolation scenarios, such as the design or assessment of novel toxicants.

Representative examples include KANO [21], which leverages an element- and functional-group-oriented knowledge graph to guide contrastive pretraining and employs functional prompts: molecular embeddings are learned under the guidance of knowledge-derived prompts associated with specific biological or toxicological functions, so that the representation space is shaped along chemically meaningful directions relevant for property and toxicity prediction. KPGT, in turn, introduces an additional knowledge node (K node) for each molecular graph, whose features encode global molecular information such as descriptors and fingerprints. This K node is connected to all other nodes and participates in transformer-based message passing, enabling structural nodes to attend to and exchange information with it, thereby integrating external chemical knowledge into the learned molecular representations.

## 4.7.4. Contrastive Learning

Contrastive learning aims to learn representations by bringing similar (positive) examples closer together in embedding space while pushing dissimilar (negative) examples apart, without requiring explicit labels. In the molecular setting, this is typically achieved by generating diferent augmented views of the same molecule —e.g., via subgraph perturbations, atom masking, or feature noise— and training the encoder so that embeddings of these views are consistent, while remaining distinguishable from embeddings of other molecules.

MolCLR [34] instantiates this idea by applying chemically motivated augmentations to molecular graphs —such as atom dropping, bond perturbation, or subgraph masking— and then maximizing the agreement between embeddings of augmented views of the same molecule. This procedure encourages the GNN encoder to capture substructural motifs and global patterns that are stable under reasonable chemical perturbations, which can later be exploited for downstream toxicity prediction in low-label regimes. Along similar lines, KANO combines contrastive learning with a functional prompt mechanism: the model is trained to align molecular embeddings with taskor endpoint-specific prompts, so that the contrastive objective not only enforces consistency across augmented views of a molecule, but also shapes the representation space around functionally meaningful directions associated with diferent biological activities or toxicity endpoints.

## 4.7.5. Multi-Task Learning

Multi-Task Learning (MTL) jointly optimizes multiple related endpoints from shared molecular representations, leveraging correlations between tasks to improve generalization and reduce overfitting through a common GNN encoder and taskspecific prediction heads. This paradigm is particularly suitable for ADMET profiling and toxicity prediction, where co-training on diverse endpoints can amplify the signal available from sparse labels and produce embeddings that transfer well across related assays. MMGIN, MPCD, and FATE-Tox exemplify this strategy by modeling multiple toxicity- or ADMET-related properties simultaneously from a shared molecular backbone, often achieving better performance than their single-task counterparts while providing a more holistic view of compound safety.

## 5. Benchmark

A major challenge in graph-based molecular toxicity prediction lies not only in the development of increasingly sophisticated architectures, but also in the lack of rigorous and standardized evaluation practices. Existing studies are commonly assessed under heterogeneous experimental conditions, often difering in dataset preprocessing, endpoint selection, train– test splitting strategies, evaluation metrics, number of runs, and statistical validation procedures. Such inconsistencies substantially hinder direct comparison across methods and make it difficult to determine whether reported improvements arise from genuine architectural advances or from diferences in experimental design. As a consequence, reproducibility and comparability remain important unresolved issues within the field.

To address these limitations, the principal contribution of this work is the introduction of a unified and reproducible benchmarking framework specifically designed for graph-based deep learning approaches to molecular toxicity prediction. Rather than relying on results reported independently in the literature, we systematically re-evaluate a broad collection of influential state-of-the-art architectures under equal and controlled experimental conditions. This enables fair head-to-head comparisons across models while minimizing confounding factors introduced by inconsistent evaluation protocols.

In contrast to previous benchmarks, our framework builds upon toxicity-specific curated datasets from TOXRIC, chosen to provide a balanced and representative evaluation landscape across diverse toxicological prediction scenarios. Additionally, we assess all models under a unified protocol incorporating several challenging partitioning strategies, together with crossvalidation and statistical ranking analyses to ensure robust and meaningful comparisons. Table 2 summarizes the evaluation methodologies originally employed by the diferent state-ofthe-art approaches and contrasts them with the standardized experimental framework proposed in this work. The full source code, including environment specifications for each model, is publicly available at https://gitlab.citius.gal/noel. suarez/benchtox.

## 5.1. Experimental set-up

All experiments were conducted using a standardized experimental protocol to ensure a fair comparison across benchmark datasets and splitting strategies. A 5-fold cross-validation was performed for all partitions where this configuration was applicable. For those splitting strategies that did not support cross-validation —namely time and maxmin partitions—, models were instead trained and evaluated over five independent runs using diferent seeds.

Table 2: Original evaluation of benchmarked approaches
<table><tr><td>Approach</td><td>Datasets</td><td>Splits</td><td>Val.</td><td>Stat. tests</td></tr><tr><td>AttentiveFP</td><td>MoleculeNet</td><td>random, scaffold</td><td>3-runs</td><td>no</td></tr><tr><td>Grover</td><td>MoleculeNet</td><td>scaffold</td><td>3-runs</td><td>no</td></tr><tr><td>MolCLR</td><td>MoleculeNet</td><td>scaffold</td><td>3-runs</td><td>no</td></tr><tr><td>KPGT</td><td>MoleculeNet</td><td>scaffold</td><td>3-runs</td><td>no</td></tr><tr><td>GEM</td><td>MoleculeNet</td><td>scaffold</td><td>4-runs</td><td>no</td></tr><tr><td>CD-MVGNN</td><td>MoleculeNet</td><td>scaffold</td><td>10-runs</td><td>no</td></tr><tr><td>NYAN</td><td>TOXRIC</td><td>random</td><td>5-cv</td><td>Friedman, Nemenyi</td></tr><tr><td>PharmaHGT</td><td>MoleculeNet</td><td>random,</td><td>5-cv</td><td>no</td></tr><tr><td>KANO</td><td>MoleculeNet</td><td>scaffold scaffold</td><td>3-runs</td><td>no</td></tr><tr><td>GeoDILI</td><td>Hepatotoxicity</td><td>random</td><td>5-cv</td><td>no</td></tr><tr><td>MMGIN</td><td>TOXRIC, Tox21</td><td>random</td><td>1-run</td><td>no</td></tr><tr><td>3MTox</td><td>MoleculeNet</td><td>random,</td><td>10-runs</td><td>no</td></tr><tr><td>hERGAT</td><td>Cardiotoxicity</td><td>scaffold</td><td>1-run</td><td></td></tr><tr><td>DILI_GATNN</td><td>Hepatotoxicity</td><td>random random</td><td>10-cv</td><td>no</td></tr><tr><td>DumplingGNN</td><td>MoleculeNet</td><td>scaffold</td><td>n-runs</td><td>no no</td></tr><tr><td>SynthMol</td><td>MoleculeNet</td><td>random,</td><td>5-runs</td><td>Mann</td></tr><tr><td>AMPred-LWN</td><td></td><td>scaffold</td><td></td><td>-Whitney U</td></tr><tr><td></td><td>Ames TOXRIC,</td><td>random</td><td>1-run</td><td>no</td></tr><tr><td rowspan="4">ours</td><td></td><td>random,</td><td></td><td></td></tr><tr><td>Tox21,</td><td>scaffold, 5-cv,</td><td></td><td>Plackett-Luce</td></tr><tr><td>ClinTox,</td><td>maxmin, 5runs</td><td></td><td></td></tr><tr><td>Ames</td><td>time</td><td></td><td></td></tr></table>

MoleculeNet includes toxicity datasets such as Tox21, ClinTox, ToxCast, and SIDER, together with non-toxicity molecular benchmarks. TOXRIC provides curated toxicity datasets for multiple organ-specific endpoints, as well as curated versions of datasets such as Tox21, Ames, and ClinTox; here, the term TOXRIC refers specifically to the organ-related toxicity datasets used in our benchmark. Abbreviation: cv, cross-validation.

The hyperparameter configurations and training setups strictly follow the original implementations described in each model oficial repository or publication. When a validation set was not originally included, we incorporated one to enable model selection and maintain consistent evaluation conditions. Each model was trained for at least 100 epochs, with an early stopping patience of 20 epochs, even in cases where the original training schedule was shorter.

All experiments were executed on a server equipped with a Dell PowerEdge R750 system featuring 2 × Intel Xeon Gold 6326 processors, 128 GB RAM, and 2 × NVIDIA Ampere A100 80 GB GPUs, running AlmaLinux 8.6 with NVIDIA Driver 515.48.07 (CUDA 11.7). All experiments were implemented in Python, with random seeds fixed for reproducibility.

## 5.1.1. Datasets

For benchmarking purposes, we selected a diverse set of widely used toxicity datasets spanning multiple biological endpoints and prediction tasks. In this work, we employ the clean, curated TOXRIC versions of these datasets [41], as follows:

• Tox21 is the largest dataset employed in this benchmark, featuring a large-scale in vitro screening collection proposed in the 2014 Tox21 Data Challenge [16] and comprising high-throughput screening data from 12 nuclear receptor and stress response assays to identify potential endocrine disruptors and toxicants via multi-task binary classification.

• ClinTox specifically targets clinical toxicity, contrasting small molecules that were approved with those that failed in clinical trials due to safety issues, making it particularly valuable in benchmarks that bridge preclinical toxicity modeling and translational risk assessment. The dataset is highly imbalanced, as its collection of molecules originates from the final stages of the drug discovery pipeline —namely clinical trials— where the number of approved compounds significantly outweighs those that fail due to toxicity.

• Ames dataset captures bacterial mutagenicity outcomes from the Ames test —the canonical benchmark for genotoxicity prediction— often used to assess how well models recover established structure–mutagenicity relationships. It proves to be the simplest task, yielding the best model performance across our benchmark, likely because it stems from a controlled laboratory test where fewer factors contribute to response variability.

• Carcinogenicity dataset compiles long-term in vivo rodent bioassay outcomes from the Carcinogenic Potency Database (CPDB) labeling compounds by tumor induction potential, enabling evaluation of models’ ability to predict oncogenic hazards.

• Hepatotoxicity dataset aggregates drug-induced liver injury (DILI) annotations from seven authoritative sources, including Liver Toxicity Knowledge Base (LTKB), DILIrank and LiverTox, creating the largest collection for liverspecific toxicity. It provides a critical benchmark for identifying structural patterns linked to hepatic adverse efects, a major cause of drug attrition.

• Cardiotoxicity datasets are provided at multiple IC thresholds (1, 5, 10, 30 µM) to assess hERG channel inhibition —a primary mechanism of QT prolongation and Torsades de Pointes arrhythmia— allowing comprehensive evaluation of model performance across varying clinical risk cutofs.

In Figure 4 we can see the number of compounds for each collection, as well as the distribution of the positive and negative labels. We can identify three datasets where negative samples dominate, specially in ClinTox, with very few positive samples. Other three datasets are skewed towards the positive label, and the last three are quite well balanced. Cardiotoxicity datasets are spread among all three categories. This is due to the moving threshold that allows for diferent toxicity consideration of the same molecules.

![](images/bc0a58f955936eb6dfca9198a23ceb94e9ba6341b304bf892619148cb9f95323.jpg)  
Figure 4: Distribution of positive (toxic) and negative (non-toxic) samples by dataset.

## 5.1.2. Data split

Diferent data splitting strategies are crucial in molecular machine learning benchmarks as they simulate real-world deployment scenarios and prevent overly optimistic performance estimates due to data leakage. Most datasets often overrepresent certain regions of the chemical space, with very few molecular representatives for other less usual chemical motifs. Therefore, random splitting ends up overestimating test performance and undermining generalization. On the other hand, domain-appropriate splits ensure robust evaluation across chemical space, time, and biological contexts. Accordingly, the following splitting strategies were considered:

• Random split partitions molecules by simple random sampling, preserving overall class balance but allowing structurally similar compounds to appear in both training and test sets by overrepresenting the same chemical class. This strategy suits i.i.d. assumptions but fails in chemistry where molecular similarity induces leakage, leading to inflated performance that doesn’t generalize to novel chemotypes.

• Scafold split groups molecules by their Bemis-Murcko scafolds [91] —core structures after removing side chains—, ensuring training and test sets contain distinct chemical skeletons. It addresses the similarity leakage of random splits by testing scafold-hopping —the ability to predict activity for novel core structures critical for virtual screening and lead optimization—. However, although widely adopted, scafold splitting has been shown to overestimate virtual screening performance by permitting nearly identical scafolds to be present in both train and test sets [92].

• Maxmin split overcomes this limitation by iteratively selecting test compounds to maximize the minimum distance to any other molecule in the dataset —typically using ECFP or Morgan fingerprints—. This creates the most dissimilar test set possible, rigorously evaluating extrapolation to chemically remote regions of molecular space. This way, the test set has very high coverage of the chemical space and evaluates the generalization of the algorithm to all kinds of compounds in the data [93].

• Time split assigns molecules to splits based on registration and publication dates, ensuring that training data precedes test data chronologically. It mitigates temporal leakage where future knowlege —e.g., recently discovered toxicophores— contaminates training, providing the most realistic estimate for prospective deployment. This split serves as a proxy for extrinsic validation of the models and how they will perform under real-world conditions, rather than the intrinsic validation of theoretical generalizability that the other splitting strategies aim to provide. Therefore, it is essential for retrospective validation of real-world applicability.

Maxmin and scafold splitting constitute dissimilarity-based approaches, designed to create chemically distinct test sets. In these settings, constructing the validation set using the same dissimilarity criteria can introduce an additional distribution shift between training and validation data, potentially biasing model selection. To mitigate this efect, we adopt a random validation strategy, where the validation set is sampled randomly from the training data distribution. This ensures that hyperparameter tuning and early stopping are performed on a distribution that is representative of the data seen during training, avoiding overfitting to artificially skewed validation splits.

To ensure robust and reliable performance estimation, we adopt diferent evaluation strategies depending on the nature of the data split. For splitting strategies that allow repeated partitioning —namely random and scafold splits— we employ 5-fold cross-validation. In this setting, the dataset is divided into five disjoint folds, and each fold is iteratively used as the test set while the remaining folds are used for training and validation. For splitting strategies where partitioning is fixed by design —such as maxmin and time splits— cross-validation is not applicable. Instead, models are trained and evaluated over five independent runs using diferent random seeds, providing an equivalent estimate of variability.

In terms of data allocation, experiments follow a consistent train/validation/test proportion of 60/20/20% for all splitting strategies with explicit partitioning (random, maxmin, and time). For scafold cross-validation, however, the partitioning follows a diferent procedure: molecules are first grouped according to their Bemis–Murcko scafolds, and these scafold groups are then assigned into five folds. Each fold contains a distinct subset of scafolds, and evaluation is performed by iteratively using one fold as the test set while training on the remaining scafold groups, ensuring structural separation between training and test data.

## 5.1.3. Metrics

In binary classification for toxicity prediction, we distinguish correct predictions —True Positives (TP) and True Negatives (TN)— from errors —False Positives (FP), flagging safe compounds as toxic, and False Negatives (FN), missing toxic ones—. Standard metrics like Accuracy gauge overall correctness but falter on imbalanced datasets by favoring the majority class; Recall tracks detected toxics; Precision assesses positive prediction reliability; and Specificity tracks how many true safe compounds we correctly identified. Yet, each metric tells only part of the story and can be gamed by trivial classifiers.

To address these limitations, the F1-score combines Precision and Recall via their harmonic mean, rewarding models that simultaneously catch toxics reliably while minimizing false alarms. This makes F1 particularly efective in imbalanced settings where no single metric sufices. Building on this, the F2- score further prioritizes Recall to penalize missed toxics more heavily —aligning well with toxicity screening, where undetected risks carry greater consequences than overcaution.

Threshold-independent metrics provide a more comprehensive view by aggregating performance across all classification thresholds. The Area Under the Receiver Operating Characteristic curve (AUROC) summarizes the ROC curve —which plots Sensitivity (True Positive Rate) against 1-Specificity (False Positive Rate)— into a single value measuring a model’s ability to rank toxic vs. safe compounds (0.5 = random guessing, 1.0 = perfect discrimination). Unlike threshold-dependent metrics like Precision or Recall, AUROC evaluates the intrinsic ranking quality of predictions by measuring how well the model separates the positive (toxic) and negative (safe) classes across the full spectrum of decision thresholds.

Despite its widespread use, AUROC can be overly optimistic on highly imbalanced datasets typical of toxicity prediction, where abundant negatives make even many False Positives have minimal impact on the False Positive Rate, masking poor toxic detection. The Area Under the Precision-Recall curve (AUPR) addresses this by focusing explicitly on the Precision-Recall trade-of for the positive (toxic) class, ofering a more realistic assessment when toxics are rare. Still, AUPR remains sensitive to positive class prevalence and overlooks negative class performance, so both metrics should be considered jointly alongside dataset characteristics and error costs.

For robust evaluation across highly imbalanced toxicity prediction datasets, we adopt the Matthews Correlation Coeficient (MCC) as the primary metric of our benchmark. MCC integrates all four types of results following Equation 6 into a single balanced correlation coeficient ranging from −1 (complete disagreement) to +1 (perfect prediction), with 0 corresponding to random performance. Unlike metrics such as Accuracy or AUROC, which may remain artificially high under strong class imbalance, MCC evaluates performance over both positive (toxic) and negative (safe) compounds simultaneously, providing a more reliable estimate of overall classification qual ity. Similarly, while metrics such as Recall, F2-score, or AUPR mainly emphasize the positive class, MCC balances the impact of the two types of errors regardless of class prevalence.

$$
{ \mathrm { M C C } } = { \frac { T P \cdot T N - F P \cdot F N } { { \sqrt { ( T P + F P ) ( T P + F N ) ( T N + F P ) ( T N + F N ) } } } }\tag{6}
$$

Consequently, we use MCC as the main reference metric throughout our benchmark discussion, following several recent toxicity prediction and GNN benchmarking studies that advocate its use under strong class imbalance conditions [45, 71]. At the same time, to maintain direct comparability with the broader literature, we also report some other more commonly used evaluation metrics discussed above, all of which can be explored in detail in our online benchmark repository.

## 5.1.4. Selected approaches

Not all approaches included in the review were incorporated into the benchmark. While the review aimed to provide broad coverage of the state-of-the-art approaches, the empirical evaluation required practical constraints to ensure reproducibility and robustness. Therefore, approaches were excluded from the benchmark if they met at least one of the following conditions:

• The model was restricted to regression problems, with no binary classification head described or evaluated by the authors.

• The source code of the proposed approach was not publicly available in an accesible respository.

• The available code could not be successfully executed following the instructions provided in the repository. In these cases, authors were contacted to resolve the issues, but no response was received.

After applying these exclusion criteria, 15 approaches were selected. However, to provide a more complete and exhaustive evaluation, we also included baseline approaches and foundational models in the benchmark, resulting in a total of 20 executed approaches. As baselines, we adopted the implementations of GCN, GAT, and GATGCN presented in [30]. As foundational models, we selected GROVER and AttentiveFP, two approaches proposed in 2020 that are frequently cited across the selected papers as reference points for comparison.

## 6. Results and discussion

Given the breadth of the benchmark —spanning multiple datasets, partitioning strategies, evaluation metrics, and a total of 21 models— an exhaustive presentation of all possible experimental combinations would be impractical.

We first present an aggregated overview of model performance across datasets, followed by statistical ranking analyses. Then, we motivate the selection of the most informative partitioning strategies and provide a detailed, dataset-level discussion under these settings. The complete set of results, including all metrics, models, datasets, and splitting strategies, is made publicly available through the online benchmark repository<sup>1</sup>.

## 6.1. Global perfomance overview

Table 3 summarizes the performance of all evaluated approaches across the selected metrics, aggregated at the dataset level for each partitioning strategy. The best value for each metric and split is highlighted in bold. Additionally, cells are colorcoded (green, orange, and yellow) to indicate first, second, and third-ranked approaches in the overall ranking, respectively.

Overall, KPGT consistently ranks among the top-performing methods across most datasets and evaluation settings, getting the first place in 17 out of 28 comparisons over all considered metrics and splits. Other approaches, including 3MTox, GeoDILI, and CD-MVGNN, also demonstrate competitive performance, with a concentration of second and third best places depending on the split and metric considered. Notably, the magnitude of performance diferences between these methods is often modest, with less than a 3% diference for MCC.

Within this general pattern, hERGAT stands out primarily in the metrics that emphasize positive-class recovery, achieving the best results in F2-score with a 71-75% and also performing strongly in F1-score, where it frequently ranks third or second. However, this advantage does not extend to the remaining metrics, suggesting a classification behavior that prioritizes sensitivity toward toxic compounds rather than uniformly balanced performance across all criteria. In practical terms, hER-GAT may be particularly suitable for safety-oriented screening scenarios, where minimizing false negatives is more important than maintaining a strict balance between precision and recall.

Regarding the comparison with the baseline and foundation models reported in Table 4, simple GNN architectures —such as GCN and GAT— as well as widely used reference models —namely AttentiveFP and GROVER— achieve results that are frequently close to those of more recent approaches, with around a 2% diference in MCC with most approaches and up to a 4% with those just bellow the third-best. This suggests that most of the proposed methods —excluding KPGT, CD-MVGNN, 3MTox, and GeoDILI— provide only limited improvements over established baselines.

In particular, foundational approaches like GROVER and AttentiveFP stand out for their comparatively strong specificity, indicating that they are especially efective at correctly identifying non-toxic compounds. This represents a relevant advantage in toxicity screening, where reducing false positives helps avoid the premature discard of safe and potentially useful candidates. However, this behavior also suggests that these models may achieve their specificity gains by sacrificing sensitivity, that is, by being more conservative in assigning compounds to the toxic class. In contrast, several more recent approaches appear to shift this trade-of toward higher toxic-compound recovery, often at the expense of a lower ability to reject safe molecules. From this perspective, the newer models are not necessarily outperforming the 2020 baselines across the board, but rather redistributing their errors toward a diferent balance between the two classes.

## 6.2. Statistical ranking across experiments

To enable a robust comparison across heterogeneous evaluation settings, we convert the MCC values obtained for each combination of dataset and splitting strategy into model rank ings and aggregate them using the Plackett–Luce model. The model estimates the relative worth of each approach, which, after normalization, can be interpreted as its probability of ranking first among the compared methods. By relying on rankings rather than raw metric values, the analysis is less sensitive to diferences in scale across datasets and enables statistically significant diferences between methods to be identified while accounting for variability across experimental settings.

As shown in Figure 5, the resulting rankings reveal a clear hierarchy among the evaluated models. KPGT occupies the top position with an estimated probability of ranking first of approximately 0.27, placing it well above the remaining approaches. The next-best models lie below 0.10, and their con fidence intervals do not overlap with that of KPGT, indicating that the observed advantage is not only consistent but also statistically robust across the diferent experimental configurations.

Below this top performer, a second separable tier is formed by CD-MVGNN, 3MTox, and GeoDILI. These three methods yield practically indistinguishable performances, as their point estimates cluster tightly together and their confidence intervals overlap extensively, indicating that no meaningful statistical diferences exist among them. When contrasted with the remaining approaches placed further down the ranking, this group demonstrates clear and statistically significant superiority. However, the lower boundary of this tier partially overlaps with the confidence intervals of KANO and GEM, which are positioned slightly lower in the aggregated ranking. This prevents us from unequivocally asserting that CD-MVGNN, 3MTox, and GeoDILI are strictly better than KANO and GEM in every single experimental scenario, even though they do consistently surpass all methods positioned below this overlapping pair.

Table 3: Benchmark performance comparison across all datasets
<table><tr><td>Approach</td><td>Split random</td><td>ACC% 74.58</td><td>F1% 66.21</td><td>F2% 65.26</td><td>AUROC%</td><td>AUPR%</td><td>SP%</td><td>MCC%</td></tr><tr><td rowspan="4"></td><td>MolCLR</td><td>scaffold 72.57 maxmin 70.10 69.45</td><td>63.23 57.67 54.74</td><td>62.79 57.23 53.13</td><td>76.09 72.83 69.60 69.43</td><td>70.67 66.48 60.84 62.47</td><td>71.19 68.71 68.15 69.95</td><td>37.29 32.02 25.40 23.92</td></tr><tr><td>GEM</td><td>random scaffold maxmin time</td><td>76.42 74.21 72.63 72.60 63.49</td><td>69.41 68.51 66.86 66.91 62.31 61.55 64.93</td><td>78.57 74.73 74.23 71.94</td><td>74.32 70.89 65.84 65.31</td><td>73.92 68.68 73.20 65.36</td><td>42.89 36.80 34.47 31.52</td></tr><tr><td>CD-MVGNN</td><td>random scaffold maxmin time</td><td>77.90 74.83 73.05 73.48</td><td>71.73 71.59 68.86 69.35 63.50 63.33 61.85 62.96</td><td>80.28 77.46 74.67 73.61</td><td>76.49 73.48 67.25 66.12</td><td>70.82 67.40 70.40 63.57</td><td>44.73 38.92 34.30 29.71</td></tr><tr><td>KPGT</td><td>random scaffold maxmin time</td><td>79.14 76.76 75.74 75.43</td><td>70.09 67.13 64.39 59.33</td><td>67.62 65.52 60.50</td><td>82.84 79.75 78.93</td><td>79.23 76.15 73.72</td><td>78.26 47.53 72.84 41.58 78.89 41.16</td></tr><tr><td rowspan="3">2023</td><td>KANO</td><td>random scaffold maxmin</td><td>76.71 70.67 74.17 65.27 73.44 66.23</td><td>55.89 71.48 63.91 67.24</td><td>78.52 80.67 77.28 76.00</td><td>68.49 77.44 74.20 70.61</td><td>77.28 69.21 70.63</td><td>34.33 43.44 37.18 36.90</td></tr><tr><td>PharmHGT</td><td>time random scaffold maxmin</td><td>72.67 57.76 75.86 67.93 73.61 63.42 71.92 62.09</td><td>55.35 68.24 62.75 61.91</td><td>71.25 78.14 75.18 73.63</td><td>65.09 73.22 70.03 66.28</td><td>68.54 71.68 70.08 68.94 68.57</td><td>26.54 39.72 33.71 31.79</td></tr><tr><td>NYAN</td><td>time random scaffold maxmin</td><td>72.47 76.97 74.61 73.99</td><td>56.38 63.24 59.18 57.81</td><td>55.00 62.15 59.32 56.82</td><td>70.19 78.70 75.13 75.05</td><td>63.90 69.54 72.66 70.08 69.30 65.68</td><td>26.26 37.00 28.81</td></tr><tr><td rowspan="2">GeoDILI</td><td></td><td>time random scaffold</td><td>72.98 76.66 75.24 66.37</td><td>58.53 58.40 67.79 68.08 65.98</td><td>71.00 79.48 77.96</td><td>66.19 64.70 73.80 72.80</td><td>69.36 63.61 69.98 69.66</td><td>30.91 25.01 40.10 38.56</td></tr><tr><td></td><td>maxmin time random scaffold</td><td>72.97 61.45 74.66 64.97 75.77 68.79 66.49</td><td>60.78 65.92 69.07</td><td>75.51 78.43 76.64</td><td>68.52 70.69 72.10</td><td>70.67 69.10 68.46</td><td>34.15 37.38 39.58</td></tr><tr><td rowspan="2">2024</td><td>MMGINbin</td><td>maxmin time random</td><td>73.82 71.78 72.43 77.20 70.18</td><td>66.99 60.29 60.39 60.63 60.87</td><td>73.62 70.49 70.68</td><td>69.22 62.98 63.04</td><td>66.41 68.35 67.49</td><td>35.43 29.72 29.10</td></tr><tr><td>3MTox</td><td>scaffold maxmin time</td><td>75.36 68.72 73.58 63.95 75.17 65.79</td><td>69.83 68.66 63.43 66.97</td><td>80.83 77.72 75.43 76.40</td><td>77.24 73.86 69.04 70.24</td><td>71.61 68.35 69.06 66.82</td><td>43.64 39.06 34.30 35.98</td></tr><tr><td rowspan="4">2025</td><td>DILI_GATNN</td><td>random scaffold maxmin time</td><td>77.07 75.56 73.20 74.60</td><td>61.23 60.68 55.99</td><td>60.47 59.55 55.31</td><td>80.32 77.44 75.18</td><td>76.15 73.10 68.39</td><td>71.03 40.31 70.52 32.76 70.30 32.66</td><td>29.26</td></tr><tr><td>hERGAT</td><td>random scaffold maxmin time</td><td>74.67 71.57 70.23 69.70</td><td>57.03 69.77 67.98 64.85</td><td>55.56 75.47 74.84 71.63</td><td>75.04 78.16 74.61 73.02 70.38</td><td>66.68 71.09 67.96 63.40</td><td>70.49 57.29 50.19 50.70</td><td>38.64 32.65 28.84 26.27</td></tr><tr><td>DumplingGNN</td><td>random scaffold maxmin time</td><td>70.61 70.43 68.49 67.97</td><td>64.69 54.72 56.22 50.30 51.45</td><td>71.72 55.11 56.51 51.25 51.21</td><td>66.39 67.12 65.07 65.75</td><td>64.51 59.68 61.05 53.33 58.35</td><td>43.31 61.54 61.60 63.43 61.72</td><td>18.49 21.65 16.56 15.07</td></tr><tr><td>SynthMol</td><td>random scaffold maxmin time</td><td>75.09 73.89 72.31 70.65</td><td>66.15 65.03 61.23 58.20</td><td>65.84 64.04 60.12 58.65</td><td>78.19 75.78 74.76 70.61</td><td>74.32 71.43 67.81 63.33</td><td>72.70 71.27 75.20 63.90</td><td>38.96 35.95 34.53 25.58</td></tr><tr><td>2026</td><td>AMPred-LWN</td><td>random scaffold maxmin time</td><td>70.34 68.26 67.88 65.14</td><td>62.49 60.80 57.93 58.58</td><td>62.92 61.63 58.46 60.76</td><td>79.37 76.47 73.72 74.22</td><td>74.48 71.75 65.84 64.40</td><td>77.54 72.10 74.01 62.00</td><td>39.68 34.14 32.08 27.52</td></tr></table>

Best metric for each split in bold. Green, orange and yellow highlights indicate the first, second and third-best approach in the general ranking (continues in next table).

Table 4: Performance comparison of baseline methods across all datasets
<table><tr><td></td><td>Approach</td><td>Split</td><td>ACC%</td><td>F1%</td><td>F2%</td><td>AUROC%</td><td>AUPR%</td><td>SP%</td><td>MCC%</td></tr><tr><td rowspan="3">basline</td><td>GCN</td><td>random scaffold maxmin time</td><td>76.28 73.69 71.73 72.07</td><td>67.87 64.08 61.25 61.98</td><td>67.99 65.02 61.86 63.28</td><td>77.34 74.41 72.70 72.84</td><td>71.28 67.79 64.28 63.52</td><td>70.36 66.96 69.24 65.44</td><td>40.03 33.75 31.74 30.31</td></tr><tr><td>GAT</td><td>random scaffold maxmin time</td><td>76.26 73.28 71.97 71.81</td><td>67.87 63.92 60.60 62.25</td><td>67.88 65.11 60.65 64.44</td><td>77.21 74.09 72.36 72.75</td><td>71.05 67.81 63.84 63.41</td><td>70.72 65.88 70.44 62.99</td><td>40.20 32.94 31.42 29.44</td></tr><tr><td>GATGCN</td><td>random scaffold maxmin time</td><td>76.12 73.73 71.99 71.89</td><td>68.07 63.94 61.83 61.87</td><td>68.72 64.45 61.69 63.28</td><td>77.32 73.97 72.80 72.78</td><td>70.99 67.23 64.34 63.81</td><td>69.89 67.49 70.54 64.65</td><td>39.92 33.98 32.67 29.58</td></tr><tr><td rowspan="4">ou ta0)</td><td></td><td>random scaffold</td><td>75.19 73.96</td><td>55.77 54.93</td><td>52.82 52.61</td><td>76.27 75.05</td><td>70.63 69.41</td><td>75.54 73.79</td><td>31.29 29.16</td></tr><tr><td> $\mathbf { G R O V E R } _ { b a s e }$ </td><td>maxmin time random scaffold</td><td>71.74 71.36 73.55 73.05</td><td>47.36 51.02 55.83 51.54</td><td>44.62 47.63 48.53 49.37</td><td>71.35 69.07 75.20 74.02</td><td>62.06 62.36 68.92 67.29</td><td>75.95 74.61 75.14 73.56</td><td>22.38 24.49 27.04 25.97</td></tr><tr><td> $\mathbf { G R O V E R } _ { l a r g e }$ </td><td>maxmin time random scaffold</td><td>71.65 71.52 70.79 68.65</td><td>46.94 49.81 63.46</td><td>44.39 46.71 65.04</td><td>70.88 68.03 77.08</td><td>61.10 62.68 71.60</td><td>75.07 74.59 71.52</td><td>20.77 23.17</td></tr><tr><td>AttentiveFP</td><td>maxmin time</td><td>65.90 62.58</td><td>59.94 58.07 47.26</td><td>61.99 59.87 44.68</td><td>73.72 71.26 68.15</td><td>68.77 63.56 61.46</td><td>67.43 69.83 75.91</td><td>36.51 30.61 29.87 21.01</td></tr></table>

Best metric for each split and table division is hightlighted in bold. Colour rankings continue those of previous table.

## 6.3. Influence of partitioning strategy

The data splitting strategy strongly influences both model performance and model rankings. As shown in the boxplot in Figure 6, both maxmin and time splits stand out as the most challenging partitioning strategies, yielding average MCC differences of approximately 10% relative to the random split. This diference may indicate that evaluations based on random or scafold-based splits sample from overlapping regions of the chemical space during both training and testing, allowing a degree of structural similarity leakage [92, 93]. In contrast, maxmin and time-based splits provide evaluations of model generalisation to structurally dissimilar and chronologically later compounds, respectively.

Specifically, the maxmin split enforces maximal structural dissimilarity between training and test instances, challenging models to generalize beyond the molecular chemotypes encountered during training, while the time split imposes a strict chronological partition based on the year of compound publication, thereby simulating prospective evaluation and ofering a closer approximation of how these tools would perform in actual screening workflows. Since these scenarios reflect realworld deployment, we focus the remainder of our comparative analysis on them.

As shown in Figure 7, KPGT attains the highest performance under the maxmin split, with a statistically significant margin over all other methods, as its confidence interval does not overlap with any competitor. The remaining models cluster within a narrow performance range, with KANO ranking second. However, the partial overlap between KANO and baseline GNNs indicates that its advantage over foundational architectures is limited.

The performance of KPGT and KANO may be explained by their respective pretraining strategies, designed to learn molecular representations enriched with chemical domain knowledge. KPGT incorporates expert-designed molecular fingerprints into a self-supervised learning framework, in which the model is trained to reconstruct the masked input from the available molecular information. KANO, in turn, exploits a knowledge graph (ElementKG) and functional prompts to integrate knowledge derived from periodic-table properties. It further employs a contrastive learning strategy that dynamically enriches molecular inputs with information about known functional groups.

On the time split, the performance ranking shifts substantially. In Figure 8, GeoDILI and 3MTox attain the highest MCC values, while KPGT falls from first to third place relative to its top-ranked performance under the maxmin split. However, the confidence intervals of the three methods largely overlap under the temporal split; therefore, no clear statistical ranking can be established among them. In fact, even though KPGT remains within the upper tier, its confidence interval overlaps considerably with those of several other models, including foundational GNNs such as GCN, GAT or GATGCN.

A plausible explanation for this result lies in the nature of the knowledge KPGT and KANO encode. Both approaches are strongly oriented toward capturing structural dissimilarity and leveraging predefined chemical knowledge or patterns derived from historical data. However, in time split, recently introduced molecules may not conform to traditional structural patterns or well-characterized functional groups embedded in these models. Consequently, methods such as GeoDILI and 3MTox become more competitive: GeoDILI benefits from incorporating three-dimensional geometric information, particularly bond-angle relationships, while 3MTox leverages motifbased representations —e.g., ring structures— that provide greater flexibility in identifying emerging functional patterns, unlike KANO, which relies on expert-curated functional-group definitions. Nevertheless, KPGT retains a strong position even in this split, which may be attributed to its bond-centric representation and its fingerprint-guided pretraining strategy. This combination embeds rich chemical knowledge and may also enable the model to uncover previously unseen toxic or functional patterns, partially mitigating the limitations associated with temporal extrapolation.

![](images/29ae3a34eacefaff9244882319470d1947225172a4fca8fc901515572cfc376a.jpg)  
Figure 5: Plackett–Luce ranking for MCC over all experiments.

![](images/2abfd3c5442030d575f6c99b4dff74c05cd053264657268f7e76562df3e1b8ea.jpg)  
Figure 6: Boxplot comparing results obtained by diferent approaches under each partition strategy.

Therefore, model rankings are highly dependent on the evaluation split. While knowledge-enhanced approaches appear to be particularly competitive under the maxmin split, no clear advantage emerges under the time split. Instead, a diverse set of methods achieve comparable performance. These results highlight the importance of evaluating toxicity prediction models across multiple partitioning strategies, as conclusions drawn from a single experimental setting may not generalize to more realistic deployment scenarios.

## 6.4. Dataset-level analysis under challenging splits

We now provide a detailed analysis of model performance at the dataset level under the maxmin and time partitioning strategies, focusing on the five top-performing approaches shown in Figure 5: KPGT, CD-MVGNN, 3MTox, GeoDILI, and KANO.

Under the maxmin split (Table 5), KPGT and KANO emerge as the most consistently competitive methods across datasets. KPGT attains the best average performance in all metrics, with particularly strong gains in AUROC and MCC, reaching 78.93 and 41.16, respectively; while KANO follows closely with 76.00 in AUROC and 36.90 in MCC. At dataset level, KPGT achieves the highest AUROC on 8 of the 9 datasets and the highest MCC on 7 of the 9, indicating that its performance advantage is not driven by a single scenario but is consistently observed across chemically diverse toxicological datasets. The largest gaps in MCC are observed in ClinTox, where KPGT exceeds CD-MVGNN and GeoDILI by 6.31 and 11.00 points, respectively; and in Ames, where it improves over KANO and GeoDILI by 5.65 and 6.02 points, respectively. These results suggest that KPGT benefits from representations that general ize particularly well under strict chemical extrapolation, likely due to its large-scale pretraining and stronger capture of latent molecular structure. By contrast, KANO appears competitive on ClinTox, where it clearly achieves the highest AUROC (93.49) and MCC (64.53), showing that ontology-informed features can be advantageous in data-scarce scenarios where biological or dataset-specific prior knowledge provides valuable guidance.

![](images/6ffeaadd76d569826b5fbd0ad008ba49e7333cc75c1c12435f42ed0c108ea582.jpg)  
Figure 7: Plackett–Luce ranking for MCC over maxmin split.

![](images/bf994ccd042944ad885037d6f7045fa0d3be0e39a14d849ba351b30cf2a94e46.jpg)  
Figure 8: Plackett–Luce ranking for MCC over time split.

Regarding the time split (Table 6), GeoDILI, 3MTox, and KPGT occupy the top tier, with each method achieving first place in 3 out of the 9 datasets, showing similar performance across datasets, with no single method clearly dominating. This similarity is also reflected in their average MCC values after excluding ClinTox: 38.77 for KPGT, 38.14 for GeoDILI, and 36.26 for 3MTox.

The inclusion of the ClinTox dataset, however, has a major impact on the overall ranking. In this dataset, clear diferences between methods are observed across all metrics, especially in MCC, which falls below zero for KPGT. While AUROC remains relatively high for all methods, the lower AUPR and MCC values of KPGT indicate that it struggles to identify the minority toxic class under strong class imbalance. In contrast, the higher AUPR values for GeoDILI and 3MTox indicate a better trade-of between precision and recall. This is also reflected in their positive MCC scores and more balanced classification

Table 5: Maxmin-split performance by dataset
<table><tr><td>Approach Metric Tox21</td><td colspan="11">ClinTox Ames Carcino Cardio-1 Cardio-5 Cardio-10 Cardio-30 Hepato AVG.</td></tr><tr><td rowspan="2">CD-MVGNN</td><td rowspan="2">auroc aupr</td><td colspan="7">72.43 89.73 82.63</td></tr><tr><td>62.78 51.28</td><td>78.20</td><td>69.60 65.53</td><td>81.39 56.67</td><td>74.00</td><td></td><td>67.92</td><td>66.00</td><td>68.33 74.67 67.25</td></tr><tr><td rowspan="3"></td><td>mčc</td><td>35.18</td><td>49.49 50.78</td><td>25.21</td><td>40.50</td><td>64.34 34.48</td><td>69.44 30.15</td><td>89.12 12.91</td><td>67.88 30.03</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>34.30</td></tr><tr><td>auroc</td><td>76.13 67.39</td><td>92.17 86.35</td><td>72.82</td><td>84.82</td><td>76.97</td><td>72.43</td><td>75.84</td><td>72.81 78.93</td></tr><tr><td rowspan="3">KPGT</td><td>aupr</td><td></td><td>66.81 83.96</td><td>70.54</td><td>61.02</td><td>68.99</td><td>75.59</td><td>92.99</td><td>76.21 73.72</td></tr><tr><td>mċc</td><td>38.88</td><td>55.80 57.26</td><td>33.76</td><td>49.95</td><td>40.31</td><td>32.96</td><td>29.56</td><td>31.93 41.16</td></tr><tr><td>auroc</td><td>73.66</td><td>93.49 82.52</td><td>70.73</td><td>80.71</td><td>75.89</td><td>67.99</td><td>68.26</td><td>76.00</td></tr><tr><td rowspan="3">KANO</td><td>aupr</td><td>65.07</td><td>72.24 78.21</td><td>68.07</td><td>53.28</td><td>66.31</td><td>70.70</td><td>89.39</td><td>70.73 72.20 70.61</td></tr><tr><td>mċc</td><td>34.90</td><td>64.53 51.61</td><td>33.63</td><td>38.93</td><td>37.26</td><td>27.57</td><td>12.74</td><td>30.90 36.90</td></tr><tr><td>auroc</td><td>72.76</td><td></td><td>67.59</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">GeoDILI</td><td>aupr</td><td>62.82</td><td>87.73 82.65 56.87 78.44</td><td>63.62</td><td>80.27</td><td>74.33</td><td>67.19</td><td>75.36</td><td>71.70 75.51</td></tr><tr><td>mċc</td><td>34.83</td><td>44.80 51.24</td><td>25.42</td><td>51.86 30.25</td><td>64.34 37.28</td><td>70.76 21.51</td><td>92.32 32.01</td><td>75.67 68.52</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>30.02 34.15</td></tr><tr><td rowspan="3">3MTox</td><td>auroc</td><td>74.10</td><td>91.90 82.99</td><td>67.67</td><td>78.84</td><td>75.36</td><td>67.08</td><td>71.76</td><td>69.17 75.43</td></tr><tr><td>aupr</td><td>65.03</td><td>62.48</td><td>79.31 65.58</td><td>50.82</td><td>64.42</td><td>69.34</td><td>91.83</td><td>72.54 69.04</td></tr><tr><td>mcc</td><td>35.96</td><td>53.48 50.29</td><td>25.30</td><td>40.76</td><td>38.76</td><td>22.99</td><td>13.94</td><td>27.22 34.30</td></tr></table>

Table 6: Time-split performance by dataset
<table><tr><td>Approach</td><td colspan="11">Metric Tox21 ClinTox Ames Carcino Cardio-1 Cardio-5 Cardio-10 Cardio-30 Hepato AVG.</td></tr><tr><td rowspan="3">CD-MVGNN</td><td rowspan="3">auroc aupr</td><td>73.55 79.23</td><td>83.89</td><td>70.30</td><td>72.58</td><td>71.69</td><td>72.13</td><td>64.01</td><td>75.15</td><td>73.61</td></tr><tr><td>61.98</td><td>15.04</td><td>88.72</td><td>74.46</td><td>52.22</td><td>66.36</td><td>78.63</td><td>87.31 70.39</td><td>66.12</td></tr><tr><td>mċc</td><td>10.46</td><td>51.90</td><td>30.41</td><td>28.43</td><td>30.56</td><td>31.15</td><td>14.18 34.42</td><td>29.71</td></tr><tr><td rowspan="3">KPGT</td><td></td><td>35.91</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>auroc</td><td>76.22</td><td>90.60</td><td>87.88 71.40</td><td>82.58</td><td>72.86</td><td>72.88</td><td>74.33</td><td>77.96</td><td>78.52</td></tr><tr><td>aupr mċc</td><td>65.87</td><td>6.70 92.26</td><td>76.23</td><td>61.17</td><td>69.29</td><td>78.57</td><td>91.98</td><td>74.33</td><td>68.49</td></tr><tr><td rowspan="3">KANO</td><td></td><td>35.82</td><td>-1.21</td><td>57.77 30.37</td><td>50.03</td><td>34.77</td><td>35.31</td><td>30.59</td><td>35.52</td><td>34.33</td></tr><tr><td>auroc</td><td>74.51</td><td>66.76</td><td>83.81 67.91</td><td></td><td></td><td>71.61</td><td>72.66</td><td>62.47 73.06</td><td>71.25</td></tr><tr><td>aupr</td><td>63.67</td><td>2.95 89.50</td><td>73.08</td><td>68.47 51.17</td><td>69.20</td><td>79.57</td><td>87.86</td><td>68.79</td><td>65.09</td></tr><tr><td rowspan="3">GeoDILI</td><td>mċc</td><td>34.14</td><td>-1.37</td><td>48.17 28.21</td><td>33.47</td><td>32.10</td><td>33.98</td><td>-2.61</td><td>32.80</td><td>26.54</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>auroc</td><td>74.58</td><td>98.59 84.45</td><td>65.09</td><td>79.32</td><td>77.02</td><td>77.19</td><td>72.36</td><td>77.24</td><td>78.43</td></tr><tr><td rowspan="3"></td><td>aupr</td><td>63.85</td><td>37.01</td><td>88.38 69.44</td><td>56.31</td><td>71.74</td><td>84.68</td><td>90.64</td><td>74.15</td><td>70.69</td></tr><tr><td>mċc</td><td>36.43</td><td>31.28</td><td>53.91 26.85</td><td></td><td>42.32</td><td>41.30</td><td>37.37</td><td>27.96 38.96</td><td>37.38</td></tr><tr><td>auroc</td><td>76.02</td><td></td><td>70.42</td><td>75.93</td><td>77.21</td><td>70.62</td><td></td><td>73.64</td><td></td></tr><tr><td rowspan="3">3MTox</td><td></td><td></td><td>92.08</td><td></td><td></td><td></td><td></td><td>67.17</td><td></td><td></td></tr><tr><td>aupr</td><td></td><td>47.28</td><td>84.56 89.08 73.17</td><td></td><td></td><td>70.74</td><td>88.73</td><td>69.73</td><td>76.40</td></tr><tr><td>mċc</td><td>65.16 37.52</td><td>34.06 55.18</td><td>31.88</td><td>52.52 35.01</td><td>40.98</td><td>75.76 34.03</td><td>24.55</td><td>30.63</td><td>70.24 35.98</td></tr></table>

The first, second, and third-best results are highlighted in green, orange, and yellow, respectively. The best value per dataset across both tables is shown in bold, while aggregated metrics averaged across all datasets are reported in italics.

performance.

Across the remaining datasets, the diferences in MCC among the five approaches vary considerably. On Tox21, all methods obtain relatively similar results, with a diference of only 3.38 points between the highest and lowest MCC values. The gap is also moderate on Carcino (5.03) and Cardio-10 (6.22), but is larger on Ames (9.60), Cardio-5 (10.74), and Hepato (8.33). The largest diferences are observed on Cardio-1 (21.60) and Cardio-30 (33.20). Overall, these results show that the relative performance of the approaches under the time split depends strongly on the specific characteristics of each dataset.

These findings show that model performance depends strongly on both the data split and the characteristics of each dataset. First, strong performance under dissimilarity-based splits does not necessarily imply good generalization to newer molecules, and viceversa.

Second, highly imbalanced datasets such as ClinTox or Cardio-30 increase the performance diferences among models, particularly for those with limited sensitivity to the minority class, as reflected in their lower MCC values. In contrast, more balanced datasets such as Ames or Tox21 yield higher and more consistent scores across models, reducing the performance gaps.

## 6.5. Emerging modeling patterns

Our results reveal that the best-performing approaches, including KPGT, GeoDILI, CD-MVGNN, and 3MTox, incorporate some form of bond-centered representations of molecular structure, either explicitly or implicitly. In practice, this is reflected in the use of line graph formulations, edge-aware message passing, or hybrid architectures in which bonds are treated as first-class entities rather than as secondary connections between atoms. From a chemical perspective, this encoding is consistent with the role of bond-dependent properties in toxicity, including molecular reactivity, electronic structure, and metabolic transformations.

By explicitly modeling bond-level information, these approaches allow the network to learn chemical patterns defined by bond types and local bond configurations more directly, rather than inferring them indirectly from atom neighborhoods alone.

Their consistent performance across datasets and partitioning strategies may suggest that bond-centered representations are a promising direction for improving predictive performance while ofering more chemically meaningful molecular representations in toxicity modeling.

## 7. Conclusions and future directions

In this work, we conducted a comprehensive scoping review and benchmarking study of graph-based deep learning approaches for molecular toxicity prediction. By integrating diverse and widely used toxicity datasets spanning multiple domains, dataset sizes, and imbalance regimes, we established a unified and publicly available benchmark for standardized, reproducible, and more realistic model evaluation. In particular, the proposed framework emphasizes diverse partitioning strategies to reduce overly optimistic estimates caused by chemical similarity leakage.

By evaluating diverse architectures under consistent and realistic conditions, we identified shared modeling principles that would likely remain obscured by the fragmented, ad hoc comparisons commonly found in the state of the art.

Overall, we hope this work provides both a practical evaluation framework and a clearer perspective on the architectural directions that appear most promising for developing more robust, generalizable, and chemically meaningful toxicity prediction models, ultimately supporting their use in real-world drug discovery decision-making pipelines. To this end, we release the full codebase to the scientific community at https: //gitlab.citius.gal/noel.suarez/benchtox to enable direct comparison of new models against our publicly available ranking and benchmark results, fostering transparent and reproducible evaluation in this area.

## Acknowledgment

This work has received financial support from the Agencia Estatal de Investigación (Spain) through projects PID2023- 149549NB-I00 and PDC2025-166312-I00, as well as from the Xunta de Galicia – Consellería de Educación, Ciencia, Universidades e Formación under the Centro de investigación de Galicia accreditation 2024–2027 (ED431G-2023/04), the Competitive Reference Groups programme (ED431C 2022/19) and the predoctoral grant ED481A-2024-039. This work was also supported by the European Union through the European Regional Development Fund (ERDF).

Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used Perplexity for language refinement and to improve clarity of the text. The authors reviewed and edited the output as needed and take full responsibility for the content of the published article.

## References

[1] J. P. Hughes, S. Rees, S. B. Kalindjian, K. L. Philpott, Principles of early drug discovery, British Journal of Pharmacology 162 (6) (2011) 1239–1249.

[2] T. T. V. Tran, A. Surya Wibowo, H. Tayara, K. T. Chong, Artificial intelligence in drug toxicity prediction: Recent advances, challenges, and future perspectives, J. Chem. Inf. Model. 63 (9) (2023) 2628–2643.

[3] J. Zhang, H. Li, Y. Zhang, J. Huang, L. Ren, C. Zhang, Q. Zou, Y. Zhang, Computational toxicology in drug discovery: Applications of artificial intelligence in admet and toxicity prediction, Briefings in Bioinformatics 26 (2025).

[4] H. Lee, J. Kim, J.-W. Kim, Y. Lee, Recent advances in aibased toxicity prediction for drug discovery, Frontiers in Chemistry 13 (2025).

[5] J. Mao, J. Akhtar, X. Zhang, L. Sun, S. Guan, G. Wang, et al., Comprehensive strategies of machine-learningbased quantitative structure-activity relationship models, iScience 24 (2021).

[6] P. Wal, J. Dwivedi, K. Pandey, K. K. Sharma, M. Tiwari, M. S. Ali, A. Khan, A. Gasmi, Advances in artificial intelligence for predictive toxicology: From qsar and omics integration to clinical safety translation, Computational Biology and Chemistry 124 (2026) 109120.

[7] A. P. B. Pena-Gralle, M. E. Schnitzer, S.-N. Boureguaa, F. Morin, M.-A. Legault, C. Sirois, A. Dragomir, L. Blais, Do machine learning methods make better predictions than conventional ones in pharmacoepidemiology?, Artificial Intelligence in Medicine 171 (2026) 103312.

[8] Z. Lin, W.-C. Chou, Machine learning and artificial intelligence in toxicological sciences, Toxicological Sciences 189 (1) (2022) 7–19.

[9] Q. Ning, Y. Wang, Y. Zhao, J. Sun, L. Jiang, K. Wang, M. Yin, Dmhgnn: Double multi-view heterogeneous graph neural network framework for drug-target interaction prediction, Artificial Intelligence in Medicine 159 (2025) 103023.

[10] X. Dong, R. Wong, W. Lyu, K. Abell-Hart, J. Deng, Y. Liu, F. Wang, et al., An integrated lstm-heterorgnn model for interpretable opioid overdose risk prediction, Artificial Intelligence in Medicine 135 (2023) 102439.

[11] C. Zhao, D. Han, Z. Zuo, T. Tohti, Kgdb-ddi: Knowledge graph-based drug background data fusion model for drug-drug interaction prediction, Artificial Intelligence in Medicine 168 (2025) 103225.

[12] B. Wang, Y. He, X. Du, L. Zhu, J. Wang, T. Wang, Vaeganmda: A microbe-drug association prediction model integrating variational autoencoders and generative adversarial networks, Artificial Intelligence in Medicine 167 (2025) 103198.

[13] Q. Shen, W.-M. Shi, W. Kong, Modified tabu search approach for variable selection in quantitative structureactivity relationship studies of toxicity of aromatic compounds, Artificial Intelligence in Medicine 49 (1) (2010) 61–66.

[14] A. C. Tricco, E. Lillie, W. Zarin, K. K. O’Brien, H. Colquhoun, D. Levac, D. Moher, M. D. Peters, T. Horsley, L. Weeks, S. Hempel, E. A. Akl, C. Chang, J. Mc-Gowan, L. Stewart, L. Hartling, A. Aldcroft, M. G. Wilson, C. Garritty, S. Lewin, C. M. Godfrey, M. T. Macdonald, E. V. Langlois, K. Soares-Weiser, J. Moriarty, T. Clifford, Ö. Tunçalp, S. E. Straus, Prisma extension for scoping reviews (prisma-scr): Checklist and explanation, Ann. Intern. Med. 169 (7) (2018) 467–473.

[15] A. Mayr, G. Klambauer, T. Unterthiner, S. Hochreiter, Deeptox: Toxicity prediction using deep learning, Front. Environ. Sci. 3 (80) (2016).

[16] A. Abdelaziz, Tox21 challenge publication, mendeley Data (2016).

[17] Z. Xiong, D. Wang, X. Liu, F. Zhong, X. Wan, X. Li, Z. Li, X. Luo, K. Chen, H. Jiang, M. Zheng, Pushing the boundaries of molecular representation for drug discovery with the graph attention mechanism, J. Med. Chem. 63 (16) (2020) 8749–8760.

[18] Y. Rong, Y. Bian, T. Xu, W. Xie, Y. Wei, W. Huang, J. Huang, Self-supervised graph transformer on largescale molecular data, arXiv:2007.02835 (2020).

[19] J. Chen, Y.-W. Si, C.-W. Un, S. W. I. Siu, Chemical toxicity prediction based on semi-supervised learning and graph convolutional neural network, Journal of Cheminformatics 13 (1) (2021) 93.

[20] H. Li, D. Zhao, J. Zeng, Kpgt: Knowledge-guided pretraining of graph transformer for molecular property prediction, in: Proc. 28th ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD), 2022, pp. 857–867.

[21] Y. Fang, Q. Zhang, N. Zhang, Z. Chen, X. Zhuang, X. Shao, X. Fan, H. Chen, Knowledge graph-enhanced molecular contrastive learning with functional prompt, Nat. Mach. Intell. 5 (5) (2023) 542–553.

[22] Y. Jiang, S. Jin, X. Jin, X. Xiao, W. Wu, X. Liu, Q. Zhang, X. Zeng, G. Yang, Z. Niu, Pharmacophoric-constrained heterogeneous graph transformer model for molecular property prediction, Commun. Chem. 6 (1) (2023) 60.

[23] Y. Myung, A. G. C. de Sá, D. B. Ascher, Deep-pk: deep learning for small molecule pharmacokinetic and toxicity prediction, Nucleic Acids Res. 52 (W1) (2024) W469– W475.

[24] S. Liu, H. Wang, W. Liu, J. Lasenby, H. Guo, J. Tang, Pretraining molecular graph representation with 3D geometry, in: Proc. Int. Conf. Learn. Represent. (ICLR), 2022, pp. 1–32.

[25] X. Fang, L. Liu, J. Lei, D. He, S. Zhang, J. Zhou, F. Wang, H. Wu, H. Wang, Geometry-enhanced molecular representation learning for property prediction, Nat. Mach. Intell. 4 (2) (2022) 127–134.

[26] S. Xu, L. Xie, Dumpling GNN: Hybrid GNN enables better ADC payload activity prediction based on chemical structure, arXiv:2410.05278 (2024).

[27] J. Cremer, L. Medrano Sandonas, A. Tkatchenko, D.-A. Clevert, G. De Fabritiis, Equivariant graph neural networks for toxicity prediction, Chem. Res. Toxicol. 36 (10) (2023) 1561–1573.

[28] Z. Su, R. Zhang, X. Fan, B. Tian, SynthMol: A drug safety prediction framework integrating graph attention and molecular descriptors into pre-trained geometric models, J. Chem. Inf. Model. 65 (5) (2025) 2256–2267.

[29] S. Ha, D. Bang, S. Kim, Fate-tox: fragment attention transformer for E(3)-equivariant multi-organ toxicity prediction, J. Cheminform. 17 (1) (2025) 74.

[30] G. Wang, H. Feng, M. Du, Y. Feng, C. Cao, Multimodal representation learning via graph isomorphism network for toxicity multitask learning, J. Chem. Inf. Model. 64 (21) (2024) 8322–8338.

[31] X. Yang, Y. Duan, Z. Cheng, K. Li, Y. Liu, X. Zeng, D. Cao, Mpcd: A multitask graph transformer for molecular property prediction by integrating common and domain knowledge, Journal of Medicinal Chemistry 67 (23) (2024) 21303–21316.

[32] K. Xu, W. Hu, J. Leskovec, S. Jegelka, How powerful are graph neural networks?, in: Proc. Int. Conf. Learn. Represent. (ICLR), 2019, pp. 1–17.

[33] P. Bai, X. Liu, H. Lu, Geometry-aware line graph transformer pre-training for molecular property prediction, arXiv:2309.00483 (2023).

[34] Y. Wang, J. Wang, Z. Cao, A. Barati Farimani, Molecular contrastive learning of representations via graph neural networks, Nat. Mach. Intell. 4 (3) (2022) 279–287.

[35] H. Ma, Y. Bian, Y. Rong, W. Huang, T. Xu, W. Xie, G. Ye, J. Huang, Cross-dependent graph neural networks for molecular property prediction, Bioinformatics 38 (7) (2022) 2003–2009.

[36] Y. Zhu, Y. Zhang, X. Li, L. Wang, 3MTox: A motiflevel graph-based multi-view chemical language model for toxicity identification with deep interpretation, J. Hazard. Mater. 476 (2024) 135114.

[37] Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, V. Pande, Moleculenet: A benchmark for molecular machine learning, arXiv:1703.00564 (2018).

[38] B. Ramsundar, P. Eastman, P. Walters, V. Pande, K. Leswing, Z. Wu, Deep Learning for the Life Sciences, O’Reilly Media, 2019.

[39] A. M. B. Amorim, L. F. Piochi, A. T. Gaspar, A. J. Preto, N. Rosário-Ferreira, I. S. Moreira, Advancing drug safety in drug development: Bridging computational predictions for enhanced toxicity prediction, Chem. Res. Toxicol. 37 (6) (2024) 827–849.

[40] C. N. Cavasotto, V. Scardino, Machine learning toxicity prediction: Latest advances by toxicity end point, ACS Omega 7 (51) (2022) 47536–47546.

[41] L. Wu, B. Yan, J. Han, R. Li, J. Xiao, S. He, X. Bo, Toxric: a comprehensive database of toxicological data and benchmarks, Nucleic Acids Res. 51 (D1) (2022) D1432–D1445.

[42] R. Li, J. Lu, Z. Liu, D. Yi, M. Wan, Y. Zhang, P. Zan, S. He, X. Bo, Reusability report: exploring the utility of variational graph encoders for predicting molecular toxicity in drug design, Nat. Mach. Intell. 6 (12) (2024) 1457– 1466.

[43] H. Xu, Y. Zhao, Y. Zhang, J. Han, P. Zan, S. He, X. Bo, Deep active learning with high structural discriminability for molecular mutagenicity prediction, Communications Biology 7 (1) (2024) 1071.

[44] Y. Kong, X. Zhao, R. Liu, Z. Yang, H. Yin, B. Zhao, J. Wang, B. Qin, A. Yan, Integrating concept of pharmacophore with graph neural networks for chemical property prediction and interpretation, J. Cheminform. 14 (1) (2022) 52.

[45] W. Wu, J. Qian, C. Liang, J. Yang, G. Ge, Q. Zhou, X. Guan, Geodili: A robust and interpretable model for drug-induced liver injury prediction using graph neural network-based molecular geometric representation, Chem. Res. Toxicol. 36 (11) (2023) 1717–1730.

[46] L. H. Torres, B. Ribeiro, J. P. Arrais, Few-shot learning with transformers via graph embeddings for molecular property prediction, Expert Systems with Applications 225 (2023) 120005.

[47] Y. Zhou, C. Ning, Y. Tan, Y. Li, J. Wang, Y. Shu, S. Liang, Z. Liu, Y. Wang, ToxMPNN: A deep learning model for small molecule toxicity prediction, J. Appl. Toxicol. 44 (7) (2024) 953–964.

[48] T. Yang, X. Ding, E. McMichael, F. W. Pun, A. Aliper, F. Ren, A. Zhavoronkov, X. Ding, AttenhERG: a reliable and interpretable graph neural network framework for predicting hERG channel blockers, J. Cheminform. 16 (1) (2024) 143.

[49] Z. A. Rollins, A. C. Cheng, E. Metwally, Molprop: Molecular property prediction with multimodal language and graph fusion, Journal of Cheminformatics 16 (1) (2024) 56.

[50] D. Lee, S. Yoo, hERGAT: predicting hERG blockers using graph attention mechanism through atom- and moleculelevel interaction analyses, J. Cheminform. 17 (1) (2025) 11.

[51] A. S. Wibowo, K. T. Chong, H. Tayara, Enhancing DILI toxicity prediction through integrated graph attention (GATNN) and dense neural networks (DNN), Toxicology 514 (2025) 154108.

[52] S. Monem, A. H. Abdel-Hamid, A. E. Hassanien, Drug toxicity prediction model based on enhanced graph neural network, Computers in Biology and Medicine 185 (2025) 109614.

[53] J. Xie, W. Liu, W. Hu, M. Ouyang, T. Huang, Graph neural network-based toxicity prediction by integrating molecular fingerprints and knowledge graph features, Toxics 13 (11), art. no. 953 (2025).

[54] A. Gu, T. Dao, Mamba: Linear-time sequence modeling with selective state spaces, arXiv:2312.00752 (2024).

[55] W. Chen, X. Liu, S. Zhang, S. Chen, Artificial intelligence for drug discovery: Resources, methods, and applications, Mol. Ther. Nucleic Acids 31 (2023) 691–702.

[56] J. L. Durant, B. A. Leland, D. R. Henry, J. G. Nourse, Reoptimization of mdl keys for use in drug discovery, Journal of Chemical Information and Computer Sciences 42 (6) (2002) 1273–1280.

[57] T. Fan, G. Sun, L. Zhao, X. Cui, R. Zhong, Qsar and classification study on prediction of acute oral toxicity of n-nitroso compounds, International Journal of Molecular Sciences 19 (2018).

[58] K. Y. Helal, M. Maciejewski, E. Gregori-Puigjane, M. Glick, A. M. Wassermann, Public domain hts fingerprints: Design and evaluation of compound bioactivity profiles from pubchem’s bioassay repository, Journal of Chemical Information and Modeling 56 (2) (2016) 390– 398.

[59] U. V. Ucak, I. Ashyrmamatov, J. Lee, Reconstruction of lossless molecular representations from fingerprints, Jour nal of Cheminformatics 15 (1) (2023) 26.

[60] D. Rogers, M. Hahn, Extended-connectivity fingerprints, Journal of Chemical Information and Modeling 50 (5) (2010) 742–754.

[61] H. L. Morgan, The generation of a unique machine description for chemical structures, Journal of Chemical Documentation 5 (2) (1965) 107–113.

[62] S. Riniker, G. A. Landrum, Open-source platform to benchmark fingerprints for ligand-based virtual screening, Journal of Cheminformatics 5 (1) (2013) 26.

[63] C. Garcia-Hernandez, A. Fernandez, F. Serratosa, Ligandbased virtual screening using graph edit distance as molecular similarity measure, Journal of Chemical Information and Modeling 59 (4) (2019) 1410–1421.

[64] N. Stiefl, I. A. Watson, K. Baumann, A. Zaliani, Erg: 2d pharmacophore descriptions for scafold hopping, Journal of Chemical Information and Modeling 46 (1) (2006) 208–220.

[65] D. Giordano, C. Biancaniello, M. A. Argenio, A. Facchiano, Drug design by pharmacophore and virtual screening approach, Pharmaceuticals 15 (2022).

[66] R. Karki, Y. Gadiya, S. Shetty, P. Gribbon, A. Zaliani, Pharmacophoric-based ml model to filter candidate e3 ligands and predict e3 ligase binding probabilities, bioRxiv:2023.08.10.552794 (2023).

[67] W. Xie, J. Zhang, Q. Xie, C. Gong, Y. Ren, J. Xie, Q. Sun, Y. Xu, L. Lai, J. Pei, Accelerating discovery of bioactive ligands with pharmacophore-informed generative models, Nature Communications 16 (1) (2025) 2391.

[68] D. Kim, J. Jeong, J. Choi, Identification of optimal machine learning algorithms and molecular fingerprints for explainable toxicity prediction models using toxcast/tox21 bioassay data, ACS Omega 9 (36) (2024) 37934–37941.

[69] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, G. E. Dahl, Neural message passing for quantum chemistry, in: Proceedings of the 34th International Conference on Machine Learning, 2017, pp. 1263–1272.

[70] L. H. Torres, B. Ribeiro, J. P. Arrais, Few-shot learning with transformers via graph embeddings for molecular property prediction, Expert Syst. Appl. 225 (2023) 120005.

[71] T. Han, Z. Pan, W. Ge, Q. Zhao, Multimodal multigranularity fusion model with mamba architecture for ames mutagenicity prediction, Journal of Medicinal Chemistry 69 (2) (2026) 1766–1778.

[72] J. Degen, C. Wegscheid-Gerlach, A. Zaliani, M. Rarey, On the art of compiling and using ’drug-like’ chemical fragment spaces, ChemMedChem 3 (10) (2008) 1503– 1507.

[73] G. Zhou, Z. Gao, Q. Ding, H. Zheng, H. Xu, Z. Wei, L. Zhang, G. Ke, Uni-mol: A universal 3d molecular representation learning framework, in: Proc. Int. Conf. Learn. Represent. (ICLR), 2023, pp. 1–35.

[74] S. Chithrananda, G. Grand, B. Ramsundar, Chemberta: Large-scale self-supervised pretraining for molecular property prediction, arXiv:2010.09885 (2020).

[75] Y. Xu, Deep neural networks for qsar, Methods in Molecular Biology 2390 (2022) 233–260.

[76] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, G. E. Dahl, Neural message passing for quantum chemistry, in: Proc. Int. Conf. Mach. Learn. (ICML), 2017, pp. 1263– 1272.

[77] W. L. Hamilton, R. Ying, J. Leskovec, Inductive representation learning on large graphs, CoRR abs/1706.02216 (2017).

[78] T. Kipf, M. Welling, Semi-supervised classification with graph convolutional networks, in: Proc. Int. Conf. Learn. Represent. (ICLR), 2017, pp. 1–14.

[79] P. Velickovi ˇ c, G. Cucurull, A. Casanova, A. Romero, ´ P. Liò, Y. Bengio, Graph attention networks, in: Proc. Int. Conf. Learn. Represent. (ICLR), 2018, pp. 1–12.

[80] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, Attention is all you need, in: Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2017, pp. 5998–6008.

[81] V. P. Dwivedi, X. Bresson, A generalization of transformer networks to graphs, arXiv:2012.09699 (2021).

[82] Z. Hu, Y. Dong, K. Wang, Y. Sun, Heterogeneous graph transformer, in: Proc. Web Conf. (WWW), 2020, pp. 2704–2710.

[83] D. P. Kingma, M. Welling, Auto-encoding variational Bayes, in: Proc. Int. Conf. Learn. Represent. (ICLR), 2014, pp. 1–14.

[84] T. N. Kipf, M. Welling, Variational graph auto-encoders, arXiv:1611.07308 (2016).

[85] H. Y. I. Lam, R. Pincket, H. Han, X. E. Ong, Z. Wang, J. Hinks, Y. Wei, W. Li, L. Zheng, Y. Mu, Application of variational graph encoders as an efective generalist algorithm in computer-aided drug design, Nat. Mach. Intell. 5 (7) (2023) 754–764.

[86] K. Xu, W. Hu, J. Leskovec, S. Jegelka, How powerful are graph neural networks?, CoRR abs/1810.00826 (2018).

[87] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, BERT: Pre-training of deep bidirectional transformers for language understanding, in: Proc. Conf. North Am. Chapter Assoc. Comput. Linguistics (NAACL-HLT), 2019, pp. 4171–4186.

[88] Anonymous, On connection between CLS token and virtual node: Are they both sides of the same coin?, Transactions on Machine Learning ResearchSubmitted; withdrawn (2025).

[89] D. Mesquita, A. H. Souza, S. Kaski, Rethinking pooling in graph neural networks, CoRR abs/2010.11418 (2020).

[90] X. Gao, W. Dai, C. Li, H. Xiong, P. Frossard, Graph pooling with node proximity for hierarchical representation learning, CoRR abs/2006.11118 (2020).

[91] G. W. Bemis, M. A. Murcko, The properties of known drugs. 1. molecular frameworks, Journal of Medicinal Chemistry 39 (15) (1996) 2887–2893.

[92] Q. Guo, S. Hernandez-Hernandez, P. J. Ballester, Scaffold splits overestimate virtual screening performance, in: Proc. Int. Conf. Artif. Neural Netw. (ICANN), 2024, pp. 58–72.

[93] J. Adamczyk, J. Poziemski, P. Siedlecki, ApisTox: a new benchmark dataset for the classification of small molecules toxicity on honey bees, Sci. Data 12 (1) (2025) 5.