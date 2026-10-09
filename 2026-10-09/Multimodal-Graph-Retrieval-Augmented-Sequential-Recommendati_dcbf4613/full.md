# Multimodal Graph Retrieval-Augmented Sequential Recommendation via Collaborative Filtering Paths

Jason Marcell Setiadi, Xin Cao, and Lina Yao

University of New South Wales, Sydney, Australia jason.setiadi@unsw.edu.au, xin.cao@unsw.edu.au, lina.yao@unsw.edu.au

Abstract. Multimodal Large Language Models (MLLMs) have demonstrated strong potential for sequential recommendation through their ability to reason over complex multimodal data. However, existing approaches either rely solely on the target user’s own interaction history, neglecting collaborative signals from neighboring users, or incur substantial computational overhead through repeated MLLM inference over long interaction histories. To address these challenges, we propose MGRASRec, a multimodal graph retrieval-augmented framework for sequential recommendation. MGRASRec injects collaborative filtering signals conditioned on the candidate item directly into the MLLM prompt by retrieving structured paths from a user-item interaction graph, extended via multimodal similarity to increase coverage beyond exact co-interaction overlap. This retrieval also surfaces the history items most relevant to the candidate at no additional cost, removing the need for recurrent summarization and keeping inference to a single forward pass per candidate. All components are unified into an augmented prompt for parameter-eficient fine-tuning of an MLLM. Extensive evaluations across three publicly available datasets validate the efectiveness of MGRASRec, achieving the best performance on all metrics with particularly strong gains in ranking quality.

Keywords: Multimodal Sequential Recommendation · Multimodal Large Language Model · Retrieval-Augmented Generation

## 1 Introduction

Sequential recommendation (SR) in real-world applications inherently involves rich multimodal item information (images, videos, audio, and textual metadata), and leveraging these signals enables a deeper understanding of user preferences beyond purely ID-based representations [11]. Recent advances in Multimodal Large Language Models (MLLMs) have significantly improved multimodal comprehension by unifying diverse inputs into a shared semantic space, enabling stronger reasoning over complex multimodal data across many domains. Motivated by these advances, recent works have adapted LLMs and MLLMs for sequential recommendation [9, 20], either as semantic encoders to generate item representations [17, 23] or leveraging their reasoning capabilities directly for recommendation generation [3, 12], showing particular promise for capturing user preferences and high-level semantic relationships across interaction sequences.

Despite this promise, two critical challenges remain unresolved. First, existing MLLM-based approaches operate in isolation from the broader behavioral patterns of similar users. Collaborative filtering signals are a well-established driver of recommendation quality [4], yet integrating them into prompt-based MLLM frameworks remains an open problem [16]. The dificulty is compounded by sparsity, as identifying such signals through co-interaction requires exact item overlap between users, leaving a substantial fraction of user-item pairs without any collaborative signal. Second, sequential recommendation naturally involves interaction histories that accumulate over time, making it infeasible to place every item in the MLLM prompt. A common workaround is to truncate history based on recency or frequency, but this may discard items most relevant to the candidate. While recurrent summarization approaches have been proposed to preserve the full history [18, 22], they introduce substantial computational overhead through repeated MLLM inference, making them impractical for real-world deployment.

We propose MGRASRec, which addresses both challenges with a single design decision. MGRASRec injects candidate-conditioned collaborative signals directly into the MLLM prompt by retrieving structured paths over the user-item interaction graph that connect the target user to the target item via bridge users, extended with a multimodal similarity layer to widen coverage beyond exact co-interaction overlap. As a byproduct, the same retrieved paths surface the history items most structurally relevant to the candidate, enabling path-guided history selection at no additional inference cost. We perform parameter-eficient fine-tuning of an MLLM as a recommender, using an augmented prompt that integrates the retrieved collaborative paths and selected history items. We evaluate MGRASRec on three public datasets from various domains, where it outperforms strong baselines on all metrics. Our main contributions are summarized as follows:

– We inject candidate-conditioned collaborative filtering signals into MLLMbased sequential recommendation via structured graph path retrieval, which ablations identify as the primary driver of performance gains across three datasets.

– We introduce path-guided history selection, which surfaces candidate-relevant history items as a zero-cost byproduct of path retrieval, keeping recommendation to a single MLLM forward pass per candidate.

## 2 Related Work

The strong semantic understanding of LLMs and MLLMs has motivated growing interest in recommendation. One line of work uses LLMs and MLLMs as feature encoders. LLMRG [17] and KAR [21] augment conventional recommenders by extracting structured reasoning and factual knowledge from LLMs, while NoteLLM-2 [23] and MLLMRec [2] employ MLLMs to transform multimodal item information into unified semantic representations. Molar [13] further incorporates collaborative filtering signals with multimodal content through post-alignment contrastive learning. However, these approaches decouple semantic understanding from the recommendation decision, meaning the model’s reasoning capabilities are never directly applied to the final prediction.

Stage 1 · Graph-Based Collaborative Filtering Retrieval Stage 2 • Relevant History Selection  
![](images/e343a5760d5ac98db6039b2d02ce43cb4c8c84ec8bcf5f63491e292b00477068.jpg)  
Fig. 1. The framework of MGRASRec

A complementary line directly formulates recommendation as a natural language generation task. P5 [3] reformulates diverse recommendation tasks as prompt-based text generation, and TALLRec [1] applies parameter-eficient fine-tuning to adapt LLaMA for recommendation. Extending this paradigm to the multimodal setting, MLLM-MSR [22] recurrently summarizes user interaction histories into natural language preference descriptions through multiple rounds of MLLM inference, which makes recommendation-time cost grow linearly with history length. MMSRARec [16] injects collaborative signals by retrieving embedding-similar users and surfacing their subsequent interactions as keyword context, but this signal is not conditioned on the candidate item, as bridge users are selected based on general history similarity rather than whether they actually interacted with the target item, providing unfocused collaborative evidence. Our work addresses these limitations by introducing structured graph-based collaborative retrieval that encodes behavioral evidence conditioned on the candidate item, a direction that remains largely unexplored in prompt-based MLLM frameworks.

## 3 Method

In this section, we introduce a three-stage framework illustrated in Fig. 1, comprising graph-based collaborative filtering retrieval, relevant history selection,

and MLLM LoRA fine-tuning. We formalize the problem setting in Section 3.1, before detailing each stage.

## 3.1 Problem Formulation

We formally define the multimodal sequential recommendation problem as follows. Given a user u, we define their historical behavior sequence as $S _ { u } = [ I _ { 1 } ^ { u } , \ldots , I _ { n } ^ { u } ]$ where $I _ { i } ^ { u }$ represents the i-th item with which the user has interacted, and n denotes the length of the sequence. Each item i in the item catalog I is associated with a textual title $t _ { i }$ and an image $v _ { i }$ (e.g., product image or video cover). Given the historical behavior sequence $S _ { u }$ and a candidate item $I _ { c } ,$ the objective of multimodal sequential recommendation is to predict the probability that $I _ { c }$ is the user’s next interaction, i.e., $P ( I _ { n + 1 } ^ { u } = I _ { c } \mid S _ { u } )$ .

## 3.2 Graph-based Collaborative Filtering Retrieval

To inject focused collaborative signals conditioned on the candidate item directly into the recommendation prompt, we represent the user-item interaction history as a bipartite graph $\mathcal { G } = ( \mathcal { U } \cup \mathcal { T } , \mathcal { E } )$ , where an undirected edge $( u , i ) \in \mathcal { E }$ indicates that user u has interacted with item i. When verbalizing paths in the prompt, each edge is expressed directionally as (user, watched, item) or (item, watched\_by, user), with the verb instantiated per domain (‘watched’ for Microlens, ‘bought for Amazon). It is built ofline over the interaction history, excluding each split’s held-out positive pairs to avoid leakage. All path retrieval described below is likewise performed ofline, decoupled from serving. We then retrieve paths over G connecting the target user to the target item via bridge users, as detailed below.

Table 1. Path retrieval coverage (%) for positive (ground-truth) and negative (sampled) candidates, reported as Train/Val/Test
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Path Type</td><td rowspan=1 colspan=1>Positive</td><td rowspan=1 colspan=1>Negative</td></tr><tr><td rowspan=1 colspan=1>Microlens</td><td rowspan=1 colspan=1>Co-interaction onlySimilarity onlyCombined</td><td rowspan=1 colspan=1>66.4/71.0/69.299.1/99.2/100.099.3/99.4/100.0</td><td rowspan=1 colspan=1>17.0/21.4/19.488.8/91.0/88.089.1/91.4/88.5</td></tr><tr><td rowspan=1 colspan=1>Amazon-Games</td><td rowspan=1 colspan=1>Co-interaction onlySimilarity onlyCombined</td><td rowspan=1 colspan=1>73.8/77.2/78.898.5/98.8/99.298.6/98.8/99.6</td><td rowspan=1 colspan=1>27.6/29.8/31.686.2/89.0/88.886.6/89.0/89.2</td></tr><tr><td rowspan=1 colspan=1>Amazon-Baby</td><td rowspan=1 colspan=1>Co-interaction onlySimilarity onlyCombined</td><td rowspan=1 colspan=1>71.0/71.6/69.298.3/99.6/98.498.5/99.6/98.4</td><td rowspan=1 colspan=1>28.7/34.0/30.793.8/95.6/94.694.3/96.0/94.9</td></tr></table>

Co-Interaction Path Retrieval For each (target user u, target item $i ^ { * } )$ pair, we perform a breadth-first search (BFS) over $\mathcal { G }$ to retrieve 3-hop co-interaction paths of the form:

$$
u \xrightarrow { \mathrm { w a t c h e d } } b \xrightarrow { \mathrm { w a t c h e d } } u ^ { \prime } \xrightarrow { \mathrm { w a t c h e d } } i ^ { * }\tag{1}
$$

where $b \in \mathcal { Z }$ is a bridge item co-interacted by both u and a bridge user $u ^ { \prime }$ and $u ^ { \prime }$ has also interacted with the target item $i ^ { * }$ . This 3-hop structure is the shortest path connecting the target user to the target item through a bridge user, capturing the most direct collaborative evidence possible, compared to longer paths that would pass through additional intermediate users. Concretely, we retrieve $S _ { u }$ and the users who have interacted with $i ^ { * }$ , and identify shared items between the target user and each bridge user as valid bridge items, each yielding one path, verbalized in the prompt as shown in Fig. 2 (with the number of bridge users capped for eficiency at scale). However, this approach requires exact co-interaction overlap, which leaves a significant fraction of user–item pairs without any collaborative signal, especially for negative candidates (Table 1). To extend coverage, we additionally accept bridge items that are visually or semantically similar to items already in the target user’s history.

Similarity Path Retrieval To identify such bridge items, we first compute multimodal embeddings for all items using CLIP (ViT-L/14) [15], where for each item i, the image embedding and title embedding are averaged and L2-normalized to yield a fused item representation:

$$
\mathbf { e } _ { i } = \mathrm { n o r m a l i z e } \left( \frac { \mathrm { C L I P _ { i m g } } ( v _ { i } ) + \mathrm { C L I P _ { t x t } } ( t _ { i } ) } { 2 } \right)\tag{2}
$$

All item embeddings are stored ofline in a FAISS IndexFlatIP index, enabling eficient cosine similarity search via inner product on the normalized vectors. For each pair $( u , i ^ { * } )$ , we batch-query the FAISS index for visually similar items per history item in $S _ { u }$ whose cosine similarity exceeds the 99th percentile of each dataset’s similarity distribution, excluding items already in $S _ { u }$ to avoid duplicating co-interaction paths. For each bridge user $u ^ { \prime }$ that has interacted with the target item $i ^ { * }$ , we check whether $u ^ { \prime }$ has also interacted with any of these similar items, and if so emit a 4-hop similarity path of the form:

$$
u ~ { \xrightarrow { \mathrm { w a t c h e d } } } ~ h ~ { \xrightarrow { \mathrm { s i m i l a r \_ t o } } } ~ b ~ { \xrightarrow { \mathrm { w a t c h e d } } } ~ u ^ { \prime } ~ { \xrightarrow { \mathrm { w a t c h e d } } } ~ i ^ { * }\tag{3}
$$

where $\textit { h } \in \textit { S } _ { u }$ is a history item and b is the similar bridge item, verbalized analogously (Fig. 2). The final hop remains strict, requiring bridge user $u ^ { \prime }$ to have directly interacted with the target item $i ^ { * }$ , preserving the quality of the collaborative signal while widening the set of valid bridge items through visual and semantic similarity. Identifying these paths requires an ANN search over the precomputed FAISS index followed by a BFS traversal over $\mathcal { G }$ to verify the final co-interaction hop. Table 1 confirms that incorporating similarity paths increases path coverage beyond co-interaction alone, covering almost all user-item pairs across all three datasets for both positive and negative candidates.

Path Priority Ranking Since the number of retrieved paths grows unboundedly with interaction density, including all of them would result in excessive prompt length and introduce noise from weakly-supported paths. We therefore select only top-K paths while including a count summary of total available co-interaction and similarity paths in the prompt, preserving signal breadth without additional token cost, as illustrated in Fig. 2. We sweep $K \in \{ 3 , 5 , 7 , 1 0 \}$ and observe no significant performance diference, and fix $K = 5$ as a balance between prompt length and coverage. Paths are ranked by the number of unique bridge users passing through each bridge item, reflecting consensus behavioral evidence. Cointeraction paths are prioritized with at least 60% of top-K slots reserved as they encode more direct collaborative evidence (Section 4.3), while remaining slots are filled by the next highest-ranked paths regardless of type.

## 3.3 Relevant History Selection

A key challenge in MLLM-based sequential recommendation is handling long user interaction histories. Providing the full history in the prompt is infeasible, while recurrent MLLM inference for summarization incurs substantial computational overhead. A natural alternative is to select a subset of the most relevant history items, but naively truncating by recency or frequency provides no guarantee that selected items relate to the target item, as the most relevant interactions may lie far from the tail of the sequence for users with long histories. We address this with a path-guided selection that surfaces the k history items most structurally relevant to the target item at zero additional inference cost. Since path retrieval identifies which history items are connected to the target item via the first hop of each retrieved path, this selection emerges directly as a byproduct of retrieval already performed. Concretely, we extract the first-hop history item from each of the top-K ranked paths, following the same co-interaction priority logic as Section 3.2. For sparse or no-path cases, remaining slots are padded with the most frequent items from the interaction history not already selected, ensuring a non-empty history in all cases. The final selection is restored to chronological order, preserving the sequential structure the model was pretrained to process.

## 3.4 MLLM LoRA Fine-Tuning

The retrieved collaborative signals and selected history items are assembled into an augmented prompt, illustrated in Fig. 2. If no paths exist for a given pair, the CF signals section is omitted and the prompt contains only the history and target item. We apply supervised fine-tuning (SFT) of LLaVA [10], where the training objective follows the next-token prediction paradigm, and the same augmented prompt format is used at both training and inference time, ensuring no distributional shift between the two stages. We adopt LoRA [7] under the PEFT framework to reduce training cost while preserving the model’s in-context reasoning capability. At inference time, the probability $P ( I _ { n + 1 } ^ { u } = I _ { c } \mid S _ { u } )$ is estimated from the first generated token:

![](images/022ed8f1ae621e4cd9644798aaa40f503a737ec2341bf7f9f81860769613b4db.jpg)  
Fig. 2. An example prompt template of MLLM-based sequential recommendation from the Microlens dataset.

$$
p = \frac { p ( \mathrm { ` y e s } ^ { \prime } ) } { p ( \mathrm { ` y e s } ^ { \prime } ) + p ( \mathrm { ` n o } ^ { \prime } ) }\tag{4}
$$

At recommendation time, MGRASRec requires only this single forward pass per candidate. We quantify the resulting eficiency advantage over MLLM-MSR [22] in Section 4.4.

## 4 Experiments

## 4.1 Experimental Setup

Dataset Description We evaluate on three open-source, real-world datasets spanning diverse recommendation domains: (1) Microlens [14], a micro-video recommendation dataset containing user-item interactions, video introductions, and cover images; (2) Amazon-Games [6], reflecting user preferences in the digital goods sector; and (3) Amazon-Baby [6], from the e-commerce domain representing purchasing behavior in the baby product category. All datasets comprise user-item interactions, product descriptions, and images. We applied k-core filtering with thresholds of (7, 5) for Microlens and (6, 5) for Amazon, iteratively removing users and items below the minimum interaction count. Due to MLLM context window limitations, we limit each user’s history to 5 items for Microlens and 10 for Amazon, sorted chronologically. Users were randomly partitioned into training, validation, and test sets at an 8:1:1 ratio. For negative sampling, we applied a 1:1 positive-to-negative ratio for training and validation, and a 1:20 ratio for testing. Negatives are sampled uniformly at random from items the target user has not interacted with.

Baseline Methods To evaluate the efectiveness of MGRASRec, we compare against representative baselines across four categories. For basic SR models, GRU4Rec [5] uses Gated Recurrent Units to model sequential dependencies between items, and SASRec [8] employs self-attention to capture long-term user preferences, both relying solely on item IDs and collaborative signals. For multimodal recommendation models, MMGCN [19] integrates multimodal features into a graph-based framework via message passing, while FREEDOM [24] leverages modality-specific graph structures and edge pruning strategies to improve robustness and mitigate modality noise. For LLM-based SR models, TALLRec [1] fine-tunes an LLM for sequence recommendation via supervised finetuning using exclusively textual item information. Finally, for MLLM-based SR models, we evaluate LLaVA both zero-shot and LoRA fine-tuned with frequency-based history truncation and no collaborative signals, equivalent to the w/o All Components variant in Section 4.3. MLLM-MSR [22] converts item images and text into natural language descriptions via an MLLM and infers user preferences through multiple rounds of reasoning with supervised fine-tuning.<sup>1</sup>

Evaluation Metrics We evaluate under a ranking-stage protocol, where the model re-scores a shortlist consisting of the ground-truth positive and sampled negative candidates rather than searching the full item catalog, consistent with the evaluation convention adopted by the baselines we compare against. To assess performance under this protocol, we employ AUC, HR@5, and MRR@5 as evaluation metrics for baseline methods and our proposed method.

Implementation Details All experiments were conducted on a single NVIDIA H200 GPU. We use LLaVA-v1.6-Mistral-7B as the backbone MLLM, fine-tuned with LoRA (rank 16, alpha 32, dropout 0.1) using the AdamW optimizer with a learning rate of 2e-5, a per-device batch size of 1, gradient accumulation of 4 steps, and 4 training epochs. All baselines are evaluated on the same preprocessed datasets, data splits, and evaluation protocol. For MLLM-MSR we use the authors’ oficial implementation under the same SFT settings as ours.

## 4.2 Overall Performance

The performance of all compared methods is presented in Table 2, averaged over 3 independent runs. MGRASRec achieves the best performance on all metrics across all three datasets. Multimodal models (FREEDOM, MMGCN) and ID-only sequential models (GRU4Rec, SASRec) each capture only part of the relevant signal and generally underperform fine-tuned MLLM-based approaches, underscoring that integrating both sequential and multimodal processing is essential. Among LLM/MLLM-based models, supervised fine-tuning is critical, with LLaVA w/ SFT substantially outperforming zero-shot LLaVA. Notably, MLLM-MSR underperforms both LLaVA w/ SFT and MGRASRec despite sharing the same backbone and SFT setup, which we attribute to item-level information loss from compressing history into a high-level preference summary rather than preserving item-specific signals via chronological prompting. Building on this fine-tuned MLLM, MGRASRec further injects candidate-conditioned collaborative signals directly into the prompt, yielding the strongest performance overall with particularly pronounced gains on the ranking metrics HR@5 and MRR@5, indicating that collaborative signals capture preference information complementary to sequential and multimodal modeling alone.

Table 2. The performance of diferent methods.
<table><tr><td rowspan=2 colspan=2></td><td rowspan=1 colspan=3>Microlens</td><td rowspan=1 colspan=3>Amazon-Games</td><td rowspan=1 colspan=3>Amazon-Baby</td></tr><tr><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>HR@5</td><td rowspan=1 colspan=1>MRR@5</td><td rowspan=1 colspan=1>AUCH</td><td rowspan=1 colspan=1>R@5M</td><td rowspan=1 colspan=1>RR@5</td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>HR@5</td><td rowspan=1 colspan=1>MRR@5</td></tr><tr><td rowspan=1 colspan=2>GRU4Rec</td><td rowspan=1 colspan=1>73.02</td><td rowspan=1 colspan=1>51.92</td><td rowspan=1 colspan=1>26.19</td><td rowspan=1 colspan=1>76.54</td><td rowspan=1 colspan=1>61.82</td><td rowspan=1 colspan=1>32.77</td><td rowspan=1 colspan=1>70.20</td><td rowspan=1 colspan=1>49.57</td><td rowspan=1 colspan=1>24.23</td></tr><tr><td rowspan=1 colspan=2>SASRec</td><td rowspan=1 colspan=1>74.49</td><td rowspan=1 colspan=1>57.48</td><td rowspan=1 colspan=1>29.40</td><td rowspan=1 colspan=1>74.15</td><td rowspan=1 colspan=1>46.31</td><td rowspan=1 colspan=1>22.15</td><td rowspan=1 colspan=1>77.02</td><td rowspan=1 colspan=1>61.45</td><td rowspan=1 colspan=1>40.76</td></tr><tr><td rowspan=6 colspan=2>MMGCNFREEDOMTALLRecLLaVALLaVA w/ SFTMLLM-MSR</td><td rowspan=1 colspan=1>73.82</td><td rowspan=1 colspan=1>53.07</td><td rowspan=1 colspan=1>26.97</td><td rowspan=1 colspan=1>78.00</td><td rowspan=1 colspan=1>58.13</td><td rowspan=1 colspan=1>33.40</td><td rowspan=1 colspan=1>71.40</td><td rowspan=1 colspan=1>48.49</td><td rowspan=1 colspan=1>24.55</td></tr><tr><td rowspan=1 colspan=1>79.62</td><td rowspan=1 colspan=1>66.40</td><td rowspan=1 colspan=1>46.40</td><td rowspan=1 colspan=1>80.12</td><td rowspan=1 colspan=1>65.20</td><td rowspan=1 colspan=1>43.89</td><td rowspan=1 colspan=1>77.72</td><td rowspan=1 colspan=1>64.40</td><td rowspan=1 colspan=1>45.73</td></tr><tr><td rowspan=1 colspan=1>85.34</td><td rowspan=1 colspan=1>73.87</td><td rowspan=1 colspan=1>52.22</td><td rowspan=1 colspan=1>90.33</td><td rowspan=1 colspan=1>86.40</td><td rowspan=1 colspan=1>65.27</td><td rowspan=1 colspan=1>82.91</td><td rowspan=1 colspan=1>70.80</td><td rowspan=1 colspan=1>48.86</td></tr><tr><td rowspan=1 colspan=1>51.07</td><td rowspan=1 colspan=1>25.60</td><td rowspan=1 colspan=1>13.89</td><td rowspan=1 colspan=1>55.07</td><td rowspan=1 colspan=1>30.40</td><td rowspan=1 colspan=1>15.26</td><td rowspan=1 colspan=1>52.27</td><td rowspan=1 colspan=1>30.00</td><td rowspan=1 colspan=1>14.99</td></tr><tr><td rowspan=1 colspan=1>v/ SFT</td><td rowspan=1 colspan=1>87.98</td><td rowspan=1 colspan=1>81.07</td><td rowspan=1 colspan=1>58.42</td><td rowspan=1 colspan=1>90.81</td><td rowspan=1 colspan=1>86.13</td><td rowspan=1 colspan=1>66.62</td><td rowspan=1 colspan=1>83.51</td><td rowspan=1 colspan=1>74.40</td><td rowspan=1 colspan=1>53.03</td></tr><tr><td rowspan=1 colspan=1>82.16</td><td rowspan=1 colspan=1>74.00</td><td rowspan=1 colspan=1>48.30</td><td rowspan=1 colspan=1>83.61</td><td rowspan=1 colspan=1>73.73</td><td rowspan=1 colspan=1>53.72</td><td rowspan=1 colspan=1>78.10</td><td rowspan=1 colspan=1>62.00</td><td rowspan=1 colspan=1>46.48</td></tr><tr><td rowspan=1 colspan=2>MGRASRec</td><td rowspan=1 colspan=1>89.55</td><td rowspan=1 colspan=1>83.07</td><td rowspan=1 colspan=1>62.65</td><td rowspan=1 colspan=1>91.96</td><td rowspan=1 colspan=1>89.47</td><td rowspan=1 colspan=1>69.94</td><td rowspan=1 colspan=1>85.54</td><td rowspan=1 colspan=1>77.40</td><td rowspan=1 colspan=1>56.59</td></tr></table>

## 4.3 Ablation Study

To evaluate the contribution of each component, we ablate MGRASRec against five variants. w/o Similarity Paths removes the CLIP-based retrieval layer, retaining only co-interaction paths. w/o Co-interaction Paths removes 3- hop co-interaction retrieval, retaining only similarity paths. w/o Path-guided History replaces path-guided selection with frequency-based truncation. w/o Collaborative Signals removes both similarity and co-interaction paths from the prompt. w/o All Components reduces the model to a vanilla SFT baseline with frequency-based history truncation and no collaborative signals.

Results are shown in Fig. 3, reporting performance drop in percentage points relative to the full model. Across all three datasets, collaborative filtering signals are consistently the most impactful component, though the relative contribution of co-interaction versus similarity paths varies by domain. Co-interaction paths are the more reliable driver on the Amazon datasets, while similarity paths matter more on Microlens, where visual semantics correlate more strongly with user interest. Path-guided history selection shows a dataset-dependent efect on accuracy, improving all three metrics on Microlens but inconsistent or slightly negative elsewhere. Its primary value lies in eficiency, since it surfaces relevant history items at no extra cost given the retrieved collaborative paths.

![](images/7353a9d2b4cd08278288ab32ff783e9ec5738c041e05ac25e39c63e3d8d57fd8.jpg)

![](images/6c85207205991902b8f5b61b11ae4f0a5f2c766b50d4cd4d49b65460201119d5.jpg)

![](images/db9f99c5a52aeb828b8b303f143144b4a24e5bf4823a2c47e1ca6bd2e7836788.jpg)

Fig. 3. Performance drop (in percentage points) when each component is removed from MGRASRec, across three datasets and three metrics. Larger bars indicate greater contribution of the removed component. Negative values indicate cases where removal improves performance, reflecting dataset-specific variation in component efectiveness.  
![](images/64ff13404af5c3d980e902d16c3bd490cfd9cf6bbe7a3260c01a9344170d168c.jpg)  
Fig. 4. GPU forward passes required at recommendation time as user history scales. MLLM-MSR incurs $\lceil | h i s t | / 3 \rceil$ additional LLM forward passes for recurrent preference summarization. MGRASRec requires exactly one regardless of history length.

## 4.4 Eficiency Analysis

We analyze inference cost at recommendation time for MGRASRec and MLLM-MSR. As shown in Fig. 4, inference cost grows linearly with user history length for MLLM-MSR but remains constant for MGRASRec. This is achieved by shifting retrieval ofline, where path construction requires only a BFS traversal over the interaction graph and lookups into a precomputed item-similarity table, both CPU-bound operations that are inexpensive relative to an MLLM forward pass. Note that this constant cost is with respect to history length only, yet the total number of MLLM forward passes still equals the number of candidate items scored, as MGRASRec evaluates each candidate independently via a single binary yes/no query.

## 5 Conclusion

We propose MGRASRec, a multimodal graph retrieval-augmented framework for sequential recommendation that addresses key limitations of existing MLLM based approaches. By injecting structured collaborative signals from neighboring users directly into the MLLM prompt via graph path retrieval, extending coverage through multimodal similarity-based bridge items, and selecting candidaterelevant history directly from the retrieved paths, MGRASRec leverages shared behavioral patterns beyond individual user history at a constant inference cost. Our results demonstrate that augmenting MLLMs with structured graph-based collaborative paths is the primary driver of MGRASRec’s gains, indicating that this is a promising direction for multimodal sequential recommendation. Currently, our formulation frames recommendation as binary prediction over a constrained candidate set, requiring one forward pass per candidate item. Future work could inject graph-based collaborative signals into open-set generation, where the MLLM directly generates the next item from the full item vocabulary under constrained decoding for item grounding.

## References

1. Bao, K., Zhang, J., Zhang, Y., Wang, W., Feng, F., He, X.: Tallrec: An efective and eficient tuning framework to align large language model with recommendation. In: Proceedings of the 17th ACM Conference on Recommender Systems (2023)

2. Dang, Y., Zhang, X., Pan, Z., Duan, Y., Chen, W., Cai, F., Chen, H.: Mllmrec: Exploring the potential of multimodal large language models in recommender systems. arXiv preprint arXiv:2508.15304 (2025)

3. Geng, S., Liu, S., Fu, Z., Ge, Y., Zhang, Y.: Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5). In: Proceedings of the 16th ACM Conference on Recommender Systems (2022)

4. He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., Wang, M.: Lightgcn: Simplifying and powering graph convolution network for recommendation. In: Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (2020)

5. Hidasi, B., Karatzoglou, A., Baltrunas, L., Tikk, D.: Session-based recommendations with recurrent neural networks. In: International Conference on Learning Representations (ICLR) (2016)

6. Hou, Y., Li, J., He, Z., Yan, A., Chen, X., McAuley, J.: Bridging language and items for retrieval and recommendation. arXiv preprint arXiv:2403.03952 (2024)

7. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-Rank Adaptation of Large Language Models. In: International Conference on Learning Representations (2022)

8. Kang, W.C., McAuley, J.: Self-attentive sequential recommendation. In: 2018 IEEE International Conference on Data Mining (ICDM) (2018)

9. Lin, J., Dai, X., Xi, Y., Liu, W., Chen, B., Zhang, H., Liu, Y., Wu, C., Li, X., Zhu, C., Guo, H., Yu, Y., Tang, R., Zhang, W.: How can recommender systems benefit from large language models: A survey. ACM Trans. Inf. Syst. (2025)

10. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. In: Advances in Neural Information Processing Systems (2023)

11. Liu, Q., Hu, J., Xiao, Y., Zhao, X., Gao, J., Wang, W., Li, Q., Tang, J.: Multimodal recommender systems: A survey. ACM Comput. Surv. (2024)

12. Liu, Y., Wang, Y., Sun, L., Yu, P.S.: Rec-gpt4v: Multimodal recommendation with large vision-language models. arXiv preprint arXiv:2402.08670 (2024)

13. Luo, Y., Qin, Q., Zhang, H., Cheng, M., Yan, R., Wang, K., Ouyang, J.: Molar: Multimodal llms with collaborative filtering alignment for enhanced sequential recommendation. arXiv preprint arXiv:2412.18176 (2024)

14. Ni, Y., Cheng, Y., Liu, X., Fu, J., Li, Y., He, X., Zhang, Y., Yuan, F.: A content-driven micro-video recommendation dataset at scale. arXiv preprint arXiv:2309.15379 (2023)

15. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Proceedings of the 38th International Conference on Machine Learning (2021)

16. Wang, H., Wang, Y., Wang, J.: Mmsrarec: Summarization and retrieval augumented sequential recommendation based on multimodal large language model. arXiv preprint arXiv:2512.20916 (2025)

17. Wang, Y., Chu, Z., Ouyang, X., Wang, S., Hao, H., Shen, Y., Gu, J., Xue, S., Zhang, J., Cui, Q., Li, L., Zhou, J., Li, S.: LLMRG: Improving Recommendations Through Large Language Model Reasoning Graphs. In: Proceedings of the AAAI Conference on Artificial Intelligence (2024)

18. Wang, Z., Liu, B., Huang, W., Hao, T., Zhou, H., Guo, Y.: Leveraging multimodal large language model for multimodal sequential recommendation. Scientific Reports (2025)

19. Wei, Y., Wang, X., Nie, L., He, X., Hong, R., Chua, T.S.: Mmgcn: Multi-modal graph convolution network for personalized recommendation of micro-video. In: Proceedings of the 27th ACM International Conference on Multimedia (2019)

20. Wu, L., Zheng, Z., Qiu, Z., Wang, H., Gu, H., Shen, T., Qin, C., Zhu, C., Zhu, H., Liu, Q., et al.: A survey on large language models for recommendation. World Wide Web (2024)

21. Xi, Y., Liu, W., Lin, J., Cai, X., Zhu, H., Zhu, J., Chen, B., Tang, R., Zhang, W., Yu, Y.: Towards open-world recommendation with knowledge augmentation from large language models. In: Proceedings of the 18th ACM Conference on Recommender Systems (2024)

22. Ye, Y., Zheng, Z., Shen, Y., Wang, T., Zhang, H., Zhu, P., Yu, R., Zhang, K., Xiong, H.: Harnessing multimodal large language models for multimodal sequential recommendation. In: Proceedings of the AAAI Conference on Artificial Intelligence (2025)

23. Zhang, C., Zhang, H., Wu, S., Wu, D., Xu, T., Zhao, X., Gao, Y., Hu, Y., Chen, E.: Notellm-2: Multimodal large representation models for recommendation. In: Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (2025)

24. Zhou, X., Shen, Z.: A tale of two graphs: Freezing and denoising graph structures for multimodal recommendation. In: Proceedings of the 31st ACM International Conference on Multimedia (2023)