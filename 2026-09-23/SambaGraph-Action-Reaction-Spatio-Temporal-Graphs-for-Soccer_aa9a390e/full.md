# SambaGraph: Action–Reaction Spatio-Temporal Graphs for Soccer Tactical Response Modeling

Abel A. Reyes-Angulo<sup>1,4</sup>, Henry O. Velesaca<sup>2,3,4</sup>, and Steven Araujo<sup>3,4</sup>

<sup>1</sup>Michigan Technological University, Houghton, MI, USA <sup>2</sup>ESPOL Polytechnic University, Guayaquil, Ecuador

<sup>3</sup> University of Granada, Granada, Spain <sup>4</sup>SambaSports AI, Guayaquil, Ecuador

areyesan@mtu.edu; hvelesac@espol.edu.ec; saraujo@espol.edu.ec

Abstract—Soccer tactics are interactive: an attacking action changes the opponent’s defensive problem, and the observed response depends on the multi-agent match state. We introduce SambaGraph, an action–reaction spatio-temporal graph dataset and benchmark for soccer tactical response modeling. From tracking and event data for all 64 matches of the 2022 FIFA World Cup, we curate 4,070 action-centered episodes represented as temporally aligned 23-node player–ball graph sequences with attack/defense views, response labels, and 26,270 split-safe attack–defense pairs. We study three questions: whether observed responses can be classified from graph episodes, whether successful defenses can be retrieved for a query attack, and whether graph-derived summaries support grounded LLM reasoning. A compact signature MLP obtains 0.796 ± 0.007 macro-F1 for response classification, while a fused graph–signature dual encoder reaches 0.471 ± 0.029 Hit@5 and 0.655 ± 0.051 Hit@10 for full-bank defensive retrieval. Hard negatives maximize pair discrimination but not retrieval quality. Local LLMs underperform supervised encoders for direct classification and do not improve over a strong original order in eight-candidate reranking, but they provide grounded tactical rationales. These results position SambaGraph as a reproducible benchmark for graph-based soccer strategy-response research. Code and dataset are available at: https://github.com/areyesan/SambaGraph.

Index Terms—sports computer vision, spatio-temporal graphs, tactical response modeling

## I. INTRODUCTION

Soccer is a multi-agent visual reasoning problem. A pass, cross, shot, foul, or set piece immediately changes the opponent’s defensive problem: defenders press, cover, clear, recover shape, or concede a chance. Although tracking and event data make these interactions observable, most public benchmarks focus on event recognition, localization, or outcome prediction rather than action–reaction supervision: given an attacking context, what defensive response followed, and which past defensive examples are tactically relevant?

This work introduces SambaGraph, a curated dataset and benchmark for tactical response modeling from tracking and event data. We use the PFF FC Enhanced 2022 World Cup data release, which provides match-level tracking, event logs, and metadata for all 64 matches [1], [2]. As Fig. 1 illustrates, we convert raw data into action-centered graph episodes: each episode starts from an on-ball anchor, aligns nearby tracking frames, constructs a fixed player–ball graph sequence, assigns a response label, and exports attack/defense views and paired examples for classification and retrieval.

Our contributions are threefold. First, we curate a graphstructured action–reaction dataset with 4,070 episodes from 64 World Cup matches. Second, we define response classification and attack-to-defense retrieval protocols with matchlevel split safety. Third, we report repeated-seed baselines showing that compact signatures are strong for classification, graph–signature fusion improves retrieval, hard-negative pair discrimination does not necessarily improve retrieval, and local LLMs are best used as explanation/reranking modules rather than primary predictors. These contributions are organized around three research questions. RQ1: Can observed tactical responses be classified from action-centered player–ball graph episodes? RQ2: Can successful defensive responses be retrieved for a query attacking context? RQ3: Can graphderived summaries support grounded LLM reasoning without replacing supervised graph/signature models?

## II. RELATED WORK

Tracking data has enabled detailed modeling of team behavior, pass value, spatial control, and tactical decision support in soccer [3]–[5]. Large-scale resources such as SoccerNet have further accelerated soccer video understanding [6], [7]. These works are valuable, but many benchmarks are centered on recognizing events or estimating value from a possession state. SambaGraph instead treats the tactical unit as an action– reaction episode: an attacking anchor, the surrounding multiagent configuration, the observed defensive response, and retrieved examples that may help explain the response.

Graph representations are a natural fit for interacting players and the ball [8]–[10], and spatio-temporal graph models have been successful for structured visual dynamics such as skeleton action recognition [11]. Recent sports-specific systems, including TacticAI for corner-kick analysis [12] and temporal graph models for pass receiver/outcome prediction [13], show the value of graph-based tactical reasoning. Dynamic graph methods such as DyRep, JODIE, TGAT, EvolveGCN, and TGN model interactions that evolve over time [14]–[18]. SambaGraph complements this literature by releasing broad action–response supervision across multiple event types and by evaluating both response prediction and example-based retrieval.

![](images/e0771212f5076ac0d71c7dd7b777728a881d0c83d6abf4129712056f8fc9098a.jpg)  
Fig. 1. SambaGraph curation pipeline. Event and tracking streams are aligned through tactical anchors, converted into synchronized player–ball graph windows labeled by the ensuing possession outcome, and exported as full graphs, team views, metadata, and paired samples.

## III. DATASET CONSTRUCTION

## A. Source data and event anchors

We derive SambaGraph from the PFF FC Enhanced 2022 World Cup dataset [1]. We flatten event logs into a unified table containing timestamps, teams, players, set-piece types, possession events, shot outcomes, and foul outcomes. We select high-signal anchors: shots, goals, corners, free kicks, penalties, and explicit fouls. Each anchor is represented as

$$
\begin{array} { r } { a = ( { \tt g a m e I d } , { \tt p e r i o d } , { \tt a n c h o r T y p e } , { \tt g a m e E v e n t I d } , } \\ { t _ { 0 } , { \tt t e a m I d } ) _ { \tt _ { \tt - \tt } } } \end{array}\tag{1}
$$

and anchors are deduplicated by event identifier and type.

## B. Tracking windows and graph tensors

For an anchor at time $t _ { 0 } ,$ we extract

$$
W ( a ) = \{ f _ { t } \mid t \in [ t _ { 0 } - 5 \mathrm { s } , t _ { 0 } + 1 0 \mathrm { s } ] \} ,\tag{2}
$$

canonicalize attack direction so the attacking team moves toward +x, and uniformly sample each episode to $L = 6 4$ temporal steps for sequence baselines. This converts each 15- second window into a fixed 64-step tensor, corresponding to an effective sampled rate of approximately 4.3 Hz in the released representation. The task therefore models the observed action– reaction episode, not strict pre-anchor forecasting: the input may include early post-anchor defensive motion, while the label summarizes the subsequent possession outcome. Preanchor-only anticipation is a future benchmark variant.

At each sampled step ℓ, the full view is a graph $G _ { \ell } \ =$ $( V _ { \ell } , E _ { \ell } , X _ { \ell } )$ with 23 fixed nodes: 11 home players, 11 away players, and the ball. Node features include location, speed when available, team identity, and jersey identifiers. Let

$$
X _ { \ell } \in \mathbb { R } ^ { 2 3 \times d } , \quad A _ { \ell } ^ { ( r ) } \in \{ 0 , 1 \} ^ { 2 3 \times 2 3 } , \quad r \in \{ \mathrm { t e a m } , \mathrm { o p p } , \mathrm { b a l l } \}\tag{3}
$$

be the node-feature matrix and relation-specific adjacency matrices for intra-team kNN, cross-team nearest-opponent, and ball-player edges. We use $k _ { \mathrm { t e a m } } = 3 , k _ { \mathrm { o p p } } = 2 ,$ , and $k _ { \mathrm { b a l l } } = 5$ , and define $A _ { \ell } = \vee _ { r } A _ { \ell } ^ { ( r ) }$ . Edge attributes include relative displacement, Euclidean distance, and relation type. The episode graph sequence is

$$
\mathcal { G } ( \boldsymbol { a } ) = \{ ( \boldsymbol { X _ { \ell } } , \boldsymbol { A _ { \ell } } , E _ { \ell } ) \} _ { \ell = 1 } ^ { L } ,\tag{4}
$$

with synchronized 12-node attack and defense views formed by selecting one team plus the ball.

## C. Response labels and pair mining

Each episode receives a fine response label from the event stream, including GOAL, SHOT\_<sub>\*</sub>, FOUL\_<sub>\*</sub>, and END\_<sub>\*</sub>. We collapse long-tailed fine labels into four coarse tactical outcomes: SUCCESS for goals and high-value attacking outcomes, STOPPED\_BY\_DEFENSE for unsuccessful shots, clearances, interceptions, and end-of-attack events, FOUL\_STOP for foul-stopped possessions, and UNKNOWN for ambiguous or unresolved cases. This mapping preserves the tactical distinction needed for the benchmark while avoiding unstable fine-label classes. For retrieval, we construct split-safe pairs

TABLE I  
DATASET SCALE AND VALIDATION SUMMARY.
<table><tr><td colspan="2">Item Value</td></tr><tr><td>Matches</td><td>64</td></tr><tr><td>Episodes</td><td>4,070</td></tr><tr><td>Attack-defense pairs</td><td>26,270</td></tr><tr><td>Automated validation checks</td><td>20</td></tr><tr><td>Failed validation checks</td><td>0</td></tr><tr><td>Full NPZ tensor coverage</td><td>100%</td></tr><tr><td>Attack-view NPZ coverage</td><td>100%</td></tr><tr><td>Defense-view NPZ coverage</td><td>100%</td></tr><tr><td>Media coverage Cross-split pair leakage</td><td>100% 0</td></tr></table>

TABLE II  
COARSE RESPONSE LABEL DISTRIBUTION.
<table><tr><td colspan="2">Coarse label Episodes</td></tr><tr><td>STOPPED_BY_DEFENSE</td><td>2,201</td></tr><tr><td>FOUL_STOP</td><td>1,066</td></tr><tr><td>UNKNOWN</td><td>464</td></tr><tr><td>SUCCESS</td><td>339</td></tr></table>

$$
( a _ { i } , d _ { j } , w _ { i j } , \mathtt { p a i r T y p e } ) \in \mathcal { P } ,\tag{5}
$$

where $a _ { i }$ is a query attack and $d _ { j }$ is a candidate defensive response. Pair types include same-episode positives, retrieved successful defensive positives, and hard failure negatives mined using a lightweight signature over ball trajectory and team-shape summaries. Tables I, II, and III summarize scale, labels, and pair composition.

## D. Artifact schema

The curated release contains global metadata and per-match artifacts keyed by attack\_id: flattened events, anchors, episode metadata, full 23-node NPZ windows, 12-node team views, response metadata, training pairs, compact episode contexts, and optional GIF/MP4 visualizations. This schema supports tabular, sequence, and graph baselines without reparsing the raw source files.

## IV. BENCHMARK TASKS AND BASELINES

## A. Response classification

Given an episode graph sequence, the goal is to classify the coarse observed response label. We report accuracy, balanced accuracy, macro-F1, and weighted-F1 over three seeds. Baselines are: (i) a compact MLP over episode-level tactical signatures, (ii) a dynamic kNN graph-GRU that applies framelevel graph message passing before temporal aggregation, and (iii) a graph–signature fusion model. This setup is response modeling over the full action–reaction window, not a claim of pre-anchor anticipation. This clarification is important because the post-anchor frames may contain part of the reaction being modeled; the benchmark is therefore designed to represent, classify, and retrieve observed tactical responses rather than forecast them from only pre-event context.

TABLE III  
PAIR DISTRIBUTION FOR RESPONSE MATCHING.
<table><tr><td>Pair type</td><td>Pairs</td></tr><tr><td>pos_same</td><td>4,070</td></tr><tr><td>pos_retrieved_success</td><td>9,990</td></tr><tr><td>neg_hard_failure</td><td>12,210</td></tr></table>

System. You are a soccer tactics analyst and graph reasoning   
assistant. Compare a query attacking context with candidate   
defensive responses and return valid JSON.   
User. Rank the candidate defensive responses for the query   
attack. Prefer candidates that are tactically plausible and likely to   
correspond to a successful defensive response.   
Query attack: Anchor $\mathtt { t y p e } ~ = ~ \mathtt { c o r n e r } ;$   
ball displacement = (2.8, 6.6); attack   
compactness = 12.4; nearest player to ball   
= 3.1m; players within 10m = 5; ...   
Candidate C1: Anchor type = free kick;   
defense compactness = 7.8; d $\ P \ t { \mathrm { h } } \ = \ 2 9 . 4 ;$   
nearest player to ball = 1.2m; ...   
Candidate C2: Anchor type = corner; defense   
compactness = 12.8; depth = 72.2; nearest   
player to ball = 6.5m; ...   
Output JSON schema: $\{ " \} \mathtt { r a n k i n g " } : [ " \mathbb { C } 1 " , " \mathbb { C } 2 " , . . . ] \ ,$   
"best":"C1", "rationale":"brief reason"}  
Fig. 2. Example of the label-masked LLM reranking prompt built from graphderived summaries.

## B. Attack-to-defense retrieval and pair scoring

Given a query attack, the model ranks candidate defensive responses. A relevant retrieval is a successful defensive response under the success-only protocol. We compare random retrieval, signature cosine retrieval, random/hard/mixednegative dual encoders, and a fused graph–signature dual encoder. We report Hit@1, Hit@5, Hit@10, MRR, and nDCG@10. Full-bank retrieval ranks all eligible same-split candidate responses, which makes the task harder and more realistic than reranking a short list. We also evaluate pair classification with AUC and AP to test whether curated positives and negatives are separable; this diagnostic is intentionally distinct from full-bank retrieval because a model can separate pair labels without producing the best global ranking of successful defenses.

## C. Auxiliary LLM graph-summary reasoning

We evaluate a lightweight language-based protocol in which each graph sequence is converted into a structured text summary with anchor type, ball displacement, team spread, compactness, and local pressure. LLMs either predict the coarse label or rerank eight candidate defenses. Candidate outcome labels are masked from the prompt and used only for evaluation, as shown in Fig. 2. This protocol tests whether graph summaries support tactical reasoning; it is not treated as a replacement for tensor-based graph or signature models.

![](images/1819c70d5e8e5a3e529c9abdff15b19a1c2af1cf22daef93ec4001740060d838.jpg)  
Fig. 3. Coarse response classification macro-F1. Supervised models outperform LLMs that receive only graph-summary text.

## V. EXPERIMENTS AND RESULTS

## A. Dataset validation

Table I shows that all validation checks pass. In particular, all full/attack/defense NPZ files exist, sampled tensors load with expected node counts, pair identifiers join to existing episodes, media artifacts are present, and all curated pairs remain split-safe. These checks are important because leakage can otherwise occur when a retrieved defense comes from a match assigned to a different split than the query attack.

All experiments use match-level train/validation/test splits to prevent leakage across games. We report mean and standard deviation over three seeds for learned baselines. Classification is evaluated with accuracy, balanced accuracy, macro-F1, and weighted-F1, while retrieval is evaluated with Hit@K, MRR, and nDCG@10. The LLM experiments use the same test episodes but receive only label-masked graph-summary text; outcome labels are used only for evaluation.

## B. Response classification

Table IV reports coarse response classification. The signature MLP is strongest, reaching $0 . 8 4 2 \pm 0 . 0 0 9$ accuracy and $0 . 7 9 6 \pm 0 . 0 0 7$ macro-F1. Graph–signature fusion is close, while the graph-only GRU is lower and higher-variance. Local LLM classifiers are weaker than supervised encoders: Qwen2.5-7B obtains the best LLM macro-F1 $( 0 . 4 5 1 \pm 0 . 0 1 7 )$ Fig. 3 visualizes the same trend and supports the interpretation that graph-derived summaries contain useful signal but do not replace specialized supervised encoders. The fact that the signature MLP remains strongest also provides a useful benchmark constraint: future graph models should be compared against compact tactical features rather than only against weak baselines.

## C. Attack-to-defense retrieval and reranking

Table V compares full-bank retrieval with the separate eight-candidate LLM reranking protocol. These protocols are not equivalent: full-bank models rank the same-split retrieval pool, whereas LLMs rerank eight preselected candidates with outcome labels hidden. In full-bank retrieval, the fused graph– signature dual encoder is best, reaching $0 . 4 7 1 \pm 0 . 0 2 9$ Hit@5 and $0 . 6 5 5 { \pm } 0 . 0 5 1$ Hit@10. In reranking, the original candidate order is already very strong $( 0 . 9 8 0 \pm 0 . 0 0 0$ Hit@1); Llama-3.2-3B largely preserves this order $( 0 . 9 4 3 \pm 0 . 0 0 5$ Hit@1), while larger local models degrade top-ranked success. Fig. 4 shows the protocol-specific results side by side.

![](images/b7e0cc495bf39329503c3f2025fc768033490b7d677f54d8c0988233fb3d009f.jpg)

![](images/8b8e0d8570c202b27181addfc7af259ed69327f640a86644e7a295caf83d7c7b.jpg)

Fig. 4. Full-bank retrieval and eight-candidate reranking. The fused graph– signature encoder is strongest for full-bank retrieval; Llama-3.2-3B best preserves the strong original reranking order.  
![](images/3760face1914abd78757ddd5d29564dc75a105afaed56a508b715f4a8b5c9581.jpg)

![](images/2e0933e81cec457f48761d9064b8f74d193fb5fc1a221a0dcfe4a084fea17940.jpg)  
Fig. 5. Comparison of pair discrimination, full-bank retrieval, and eightcandidate LLM reranking. Hard negatives improve pair AP, but retrieval and reranking quality follow different trends.

## D. Negative mining and LLM interpretation

Table VI reports the pair-discrimination sanity check. Hardnegative training gives the best pair AUC/AP, but Fig. 5 shows that this does not translate monotonically to fullbank retrieval. The result is methodological: pair classification tests separability of curated positives and negatives, whereas retrieval tests whether successful defenses are ranked well among many candidates.

The LLM results add a complementary perspective. The LLM classifier is not competitive with supervised numerical encoders, and LLM reranking does not beat the original candidate order. However, Llama-3.2-3B preserves most of that ordering and produces natural-language tactical rationales from label-masked graph summaries. Thus, we treat LLMs as explanation/reranking modules rather than primary graph encoders. Fig. 7 reports the corresponding Hit@1 comparison.

## E. Qualitative retrieval

Beyond aggregate metrics, retrieval outputs are useful only if they are spatially interpretable. Fig. 6 illustrates how retrieved examples can be inspected on the pitch. These qualitative views are not used to compute the metrics, but they are useful for auditing whether retrieved defenses are not only label-correct but also spatially and tactically plausible. This is important for sports analytics applications, where a recommendation should be interpretable to analysts and coaches.

TABLE IV  
COARSE RESPONSE CLASSIFICATION. SUPERVISED MODELS USE TENSORS/SIGNATURES; LLMS USE GRAPH-SUMMARY TEXT. RESULTS ARE MEAN ± STANDARD DEVIATION OVER THREE SEEDS.
<table><tr><td>Model</td><td>Input</td><td>Protocol</td><td>Acc.</td><td>Bal. Acc.</td><td>Macro-F1</td></tr><tr><td>Signature MLP</td><td>Signature vector</td><td>Supervised</td><td> $0 . 8 4 2 \pm 0 . 0 0 9$ </td><td> $0 . 8 2 5 \pm 0 . 0 0 4$ </td><td> $0 . 7 9 6 \pm 0 . 0 0 7$ </td></tr><tr><td>Graph-signature fusion</td><td>Graph + signature</td><td>Supervised</td><td> $0 . 8 4 2 \pm 0 . 0 0 1$ </td><td> $0 . 7 8 8 \pm 0 . 0 1 3$ </td><td> $0 . 7 8 4 \pm 0 . 0 0 7$ </td></tr><tr><td>Dynamic graph-GRU</td><td>Graph tensor</td><td>Supervised</td><td> $0 . 5 4 3 \pm 0 . 0 8 7$ </td><td> $0 . 5 3 8 \pm 0 . 0 9 8$ </td><td> $0 . 4 9 5 \pm 0 . 0 9 7$ </td></tr><tr><td>Qwen2.5-7B</td><td>Graph-summary text</td><td>LLM zero-shot</td><td> $0 . 5 0 4 \pm 0 . 0 2 2$ </td><td> $0 . 4 4 8 \pm 0 . 0 1 7$ </td><td> $0 . 4 5 1 \pm 0 . 0 1 7$ </td></tr><tr><td>Llama-3.1-8B</td><td>Graph-summary text</td><td>LLM zero-shot</td><td> $0 . 5 2 7 \pm 0 . 0 4 0$ </td><td> $0 . 4 6 9 \pm 0 . 0 3 1$ </td><td> $0 . 4 2 4 \pm 0 . 0 2 7$ </td></tr><tr><td>Llama-3.2-3B</td><td>Graph-summary text</td><td>LLM zero-shot</td><td> $0 . 4 6 0 \pm 0 . 0 1 2$ </td><td> $0 . 4 8 2 \pm 0 . 0 1 2$ </td><td> $0 . 3 9 8 \pm 0 . 0 1 1$ </td></tr></table>

TABLE V

RETRIEVAL AND RERANKING RESULTS. FULL-BANK RETRIEVAL SEARCHES SAME-SPLIT CANDIDATES; LLMS RERANK EIGHT PRESELECTED DEFENSES WITH LABELS HIDDEN.
<table><tr><td>Model</td><td>Protocol</td><td>Hit@1</td><td>Hit@5</td><td>Hit@10</td><td>MRR</td><td>nDCG@10</td></tr><tr><td>Fused dual encoder</td><td>Full-bank retrieval</td><td> $0 . 1 9 3 \pm 0 . 0 1 4$ </td><td> $0 . 4 7 1 \pm 0 . 0 2 9$ </td><td> $0 . 6 5 5 \pm 0 . 0 5 1$ </td><td> $0 . 3 2 8 \pm 0 . 0 2 0$ </td><td> $0 . 2 9 5 \pm 0 . 0 0 7$ </td></tr><tr><td>Dual encoder</td><td>Full-bank retrieval</td><td> $0 . 1 6 4 \pm 0 . 0 1 8$ </td><td> $0 . 4 1 4 \pm 0 . 0 2 1$ </td><td> $0 . 5 7 9 \pm 0 . 0 4 1$ </td><td> $0 . 2 9 3 \pm 0 . 0 2 1$ </td><td> $0 . 2 4 7 \pm 0 . 0 2 3$ </td></tr><tr><td>Mixed 70% hard</td><td>Full-bank retrieval</td><td> $0 . 1 3 8 \pm 0 . 0 3 1$ </td><td> $0 . 4 1 1 \pm 0 . 0 4 1$ </td><td> $0 . 5 8 8 \pm 0 . 0 0 4$ </td><td> $0 . 2 7 5 \pm 0 . 0 2 9$ </td><td> $0 . 2 3 5 \pm 0 . 0 2 4$ </td></tr><tr><td>Signature cosine</td><td>Full-bank retrieval</td><td> $0 . 0 1 8 \pm 0 . 0 0 0$ </td><td> $0 . 0 8 3 \pm 0 . 0 0 0$ </td><td> $0 . 1 2 2 \pm 0 . 0 0 0$ </td><td> $0 . 0 5 2 \pm 0 . 0 0 0$ </td><td> $0 . 0 3 9 \pm 0 . 0 0 0$ </td></tr><tr><td>Random</td><td>Full-bank retrieval</td><td> $0 . 0 0 6 \pm 0 . 0 0 2$ </td><td> $0 . 0 3 0 \pm 0 . 0 0 2$ </td><td> $0 . 0 6 1 \pm 0 . 0 0 3$ </td><td> $0 . 0 3 3 \pm 0 . 0 0 2$ </td><td> $0 . 0 1 4 \pm 0 . 0 0 1$ </td></tr><tr><td>Original order</td><td>8-candidate reranking</td><td> $0 . 9 8 0 \pm 0 . 0 0 0$ </td><td> $0 . 9 8 1 \pm 0 . 0 0 2$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 9 8 3 \pm 0 . 0 0 0$ </td><td> $0 . 9 8 6 \pm 0 . 0 0 0$ </td></tr><tr><td>Llama-3.2-3B</td><td>8-candidate reranking</td><td> $0 . 9 4 3 \pm 0 . 0 0 5$ </td><td> $0 . 9 7 7 \pm 0 . 0 0 2$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 9 5 4 \pm 0 . 0 0 3$ </td><td> $0 . 9 5 9 \pm 0 . 0 0 3$ </td></tr><tr><td>Random order</td><td>8-candidate reranking</td><td> $0 . 3 5 2 \pm 0 . 0 0 5$ </td><td> $0 . 9 5 9 \pm 0 . 0 0 1$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 5 8 4 \pm 0 . 0 0 4$ </td><td> $0 . 6 7 7 \pm 0 . 0 0 2$ </td></tr><tr><td>Llama-3.1-8B</td><td>8-candidate reranking</td><td> $0 . 3 1 9 \pm 0 . 0 1 0$ </td><td> $0 . 8 2 1 \pm 0 . 0 1 0$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 5 3 9 \pm 0 . 0 0 7$ </td><td> $0 . 6 9 4 \pm 0 . 0 0 3$ </td></tr><tr><td>Qwen2.5-7B</td><td>8-candidate reranking</td><td> $0 . 1 2 8 \pm 0 . 0 1 1$ </td><td> $0 . 8 0 1 \pm 0 . 0 0 8$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 3 8 8 \pm 0 . 0 0 8$ </td><td> $0 . 5 9 0 \pm 0 . 0 0 5$ </td></tr></table>

![](images/c9225cf1a4d1e79963a9579b01aceabceb683200e877bec6aab01fc666b4c066.jpg)  
Fig. 6. Qualitative retrieval example showing a query attack and retrieved defensive response candidates rendered on the pitch.

TABLE VI  
PAIR CLASSIFICATION ON CURATED ATTACK–DEFENSE PAIRS.
<table><tr><td>Model</td><td>AUC</td><td>AP</td></tr><tr><td>Dual encoder, hard negatives</td><td> $0 . 9 8 3 \pm 0 . 0 1 4$ </td><td> $0 . 9 8 2 \pm 0 . 0 1 4$ </td></tr><tr><td>Dual encoder, mixed 70% hard</td><td> $0 . 9 4 7 \pm 0 . 0 0 8$ </td><td> $0 . 9 4 6 \pm 0 . 0 0 2$ </td></tr><tr><td>Dual encoder, mixed 50% hard</td><td> $0 . 9 2 8 \pm 0 . 0 0 5$ </td><td> $0 . 9 2 6 \pm 0 . 0 0 6$ </td></tr><tr><td>Dual encoder, mixed 30% hard</td><td> $0 . 9 2 4 \pm 0 . 0 0 6$ </td><td> $0 . 9 1 9 \pm 0 . 0 0 3$ </td></tr><tr><td>Fused dual encoder, random negatives</td><td> $0 . 9 0 0 \pm 0 . 0 1 5$ </td><td> $0 . 9 0 8 \pm 0 . 0 1 3$ </td></tr><tr><td>Dual encoder, random negatives</td><td> $0 . 9 0 3 \pm 0 . 0 0 6$ </td><td> $0 . 8 9 7 \pm 0 . 0 0 7$ </td></tr></table>

## VI. DISCUSSION

The experiments answer the motivating questions. First, tracking and event logs can be transformed into validated action–reaction graph episodes with split-safe labels and retrieval supervision. Second, observed response classification is feasible, but compact signatures remain difficult to beat;

![](images/d2a04daaabc3deb2787500a63d4c3a8f60e1b9939126d1a2093984200b34fd4c.jpg)  
Fig. 7. Eight-candidate LLM reranking with outcome labels masked. The original order is strongest; Llama-3.2-3B preserves it better than the other local LLMs.

this weakens any claim that graph encoders alone drive classification performance and establishes a strong baseline for future graph models. This outcome is expected in a limited-data setting: hand-crafted signatures directly encode ball displacement, team spread, compactness, and local pressure, whereas graph encoders must learn these abstractions from relatively few episodes. Third, graph information is useful for the retrieval use case: the fused graph–signature dual encoder is the strongest full-bank retrieval model. This retrieval setting is closest to the intended analytics use case: given a current attacking configuration, an analyst can inspect similar historical defensive responses and compare how different teams contained or failed to contain comparable situations. Fourth, negative mining changes what the embedding learns, with hard negatives improving pair AP but reducing broad success-oriented retrieval. Finally, local LLMs can reason over graph-derived summaries, but their practical role is currently grounded explanation and short-list reranking, not direct prediction.

The benchmark should therefore be read primarily as a dataset and protocol contribution rather than as a new architecture paper. This is intentional. By providing the graph tensor representation, response-label mapping, pair construction, validation checks, and multiple baseline families, SambaGraph makes it possible to study when relational structure helps, when compact tactical descriptors are sufficient, and how retrieval can support interpretable soccer analysis.

## VII. LIMITATIONS AND ETHICS

SambaGraph models observed action–reaction outcomes rather than optimal tactical decisions. It is also not a pure pre-anchor anticipation benchmark, because the released fullwindow representation includes early post-anchor motion. A successful response may depend on factors not fully captured in the graph, such as attacker error, goalkeeper performance, score state, fatigue, or coaching instructions. Coarse labels compress rich tactical behavior, and broadcast tracking may include occlusions, identity switches, or positional noise. The benchmark is intended for aggregate tactical research, not individual player assessment. Any release must respect the sourcedata license; if raw tracking/event data cannot be redistributed, the public release should provide curation code, schemas, manifests, validation notebooks, and benchmark scripts while requiring users to obtain the original data through the official source.

## VIII. CONCLUSION

We introduced SambaGraph, a curated action–reaction graph dataset and benchmark for soccer tactical response modeling. The dataset contains 4,070 validated episodes from 64 World Cup matches, synchronized attack/defense graph views, and 26,270 curated pairs. The baselines show that compact signatures are strong for response classification, graph–signature fusion improves full-bank defensive retrieval, and pair discrimination is not equivalent to retrieval quality. Multi-seed local LLM experiments show that graph summaries support grounded rationales, but current LLMs remain below supervised encoders for classification and do not improve over a strong original reranking order. Together, these results establish SambaGraph as a practical benchmark for graph reasoning and strategy-response research in soccer.

## REFERENCES

[1] PFF FC, “Unleash your inner analyst: PFF FC’s 2022 World Cup dataset now available,” PFF FC Blog, 2024.

[2] PySport/Kloppy contributors, “PFF FC – Kloppy documentation,” Kloppy documentation, 2025.

[3] P. Power, H. Ruiz, X. Wei, and P. Lucey, “Not all passes are created equal: objectively measuring the risk and reward of passes in soccer from tracking data,” in Proc. KDD, 2017, pp. 1605–1613.

[4] J. Fernandez and L. Bornn, “SoccerMap: A deep learning architecture´ for visually-interpretable analysis in soccer,” in ECML PKDD, 2021, pp. 491–506.

[5] H. M. Le, Y. Yue, P. Carr, and P. Lucey, “Coordinated multi-agent imitation learning,” in Proc. ICML, 2017, pp. 1995–2003.

[6] A. Cioppa, A. Deliege, S. Giancola, B. Ghanem, M. Van Droogenbroeck,\` et al., “Scaling up SoccerNet with multi-view spatial localization and re-identification,” Scientific Data, vol. 9, no. 355, 2022.

[7] SoccerNet, “SoccerNet: Open-source tools and utilities,” GitHub repository, 2026.

[8] P. W. Battaglia, J. B. Hamrick, V. Bapst, A. Sanchez-Gonzalez, V. Zambaldi, et al., “Relational inductive biases, deep learning, and graph networks,” arXiv:1806.01261, 2018.

[9] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in Proc. ICLR, 2017.

[10] P. Velickovi ˇ c, G. Cucurull, A. Casanova, A. Romero, P. Li ´ o, and Y.\` Bengio, “Graph attention networks,” in Proc. ICLR, 2018.

[11] S. Yan, Y. Xiong, and D. Lin, “Spatial temporal graph convolutional networks for skeleton-based action recognition,” in Proc. AAAI, 2018, pp. 7444–7452.

[12] Z. Wang, P. Velickovi ˇ c, D. Hennes, N. Toma ´ sev, L. Prince, ˇ et al., “TacticAI: An AI assistant for football tactics,” Nature Communications, vol. 15, p. 1906, 2024.

[13] P. Rahimian, H. Kim, M. Schmid, and L. Toka, “Pass receiver and outcome prediction in soccer using temporal graph networks,” in Machine Learning and Data Mining for Sports Analytics, 2024, pp. 52–63.

[14] R. Trivedi, M. Farajtabar, P. Biswal, and H. Zha, “DyRep: Learning representations over dynamic graphs,” in Proc. ICLR, 2019.

[15] S. Kumar, X. Zhang, and J. Leskovec, “Predicting dynamic embedding trajectory in temporal interaction networks,” in Proc. KDD, 2019, pp. 1269–1278.

[16] D. Xu, C. Ruan, E. Korpeoglu, S. Kumar, and K. Achan, “Inductive representation learning on temporal graphs,” in Proc. ICLR, 2020.

[17] A. Pareja, G. Domeniconi, J. Chen, T. Ma, T. Suzumura, et al., “EvolveGCN: Evolving graph convolutional networks for dynamic graphs,” in Proc. AAAI, 2020, pp. 5363–5370.

[18] E. Rossi, B. Chamberlain, F. Frasca, D. Eynard, F. Monti, and M. Bronstein, “Temporal graph networks for deep learning on dynamic graphs,” arXiv:2006.10637, 2020.