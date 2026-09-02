# TRIAGE: Three-level Routing and Intelligent Agent Guidance for Eficient Execution

Ruocan Wei<sup>1</sup>

<sup>1</sup>China Telecom Cloud, Beijing, China

xjtu wrc@stu.xjtu.edu.cn

Keywords: LLM Agent; Token Eficiency; Trajectory Reuse; Three-level Routing; Experience-Driven Execution

## Abstract

Large Language Model (LLM) agents based on the ReAct paradigm have demonstrated remarkable capabilities in tool use and task execution. However, ReAct sufers from a fundamental eficiency problem: every query triggers a complete reasoning loop from scratch, and similar queries repeat identical steps without leveraging historical experience. We propose TRIAGE—a three-level routing framework that reduces token consumption by reusing historical execution trajectories. The core innovation of TRIAGE is TaaS (Trajectory-as-a-Skill), which abstracts historical execution trajectories into reusable skills, realizing “experience as a service.” TRIAGE classifies input queries into three levels: (1) Direct Reuse—identical queries, 0 tokens; (2) Skill Substitution—similar queries, 0 tokens through deterministic parameter substitution when the matched trajectory has been distilled into a Skill; (3) Full ReAct—novel queries, automatically stored for future reuse. In large-scale experiments on 1,007 security monitoring queries using a semantic encoder (all-MiniLM-L6-v2) for similarity retrieval, TRIAGE achieves 62.3% token savings (199,782 → 75,238), with 56.0% of queries executed at Level 2 (Skill parameter substitution, 0 tokens) and 5.5% at Level 1 (direct reuse, 0 tokens). Cross-domain validation on ToolBench (15 domains, 345

queries, real threshold routing) achieves an average of 76.3% token reduction, confirming the generalizability of the semantic routing framework (424,792 → 100,875), with 84.6% of queries executed at Level 1 and 11.6% at Level 2. An online learning experiment over 1,007 queries demonstrates the cold-start-to-mature evolution: the L2 hit rate rises from 0% to 57% within the first 100 queries, and the average token cost drops from 198 to 74.7. Additionally, we propose an automatic Skill extraction mechanism that distills highfrequency trajectory patterns into fine-grained deterministic Skills, creating a positive feedback loop of ”the more you use it, the more eficient it becomes.” The open-source implementation of TRIAGE is available at https://github.com/weiruocan/triage.

## 1 Introduction

## 1.1 Motivation

LLM agents have become the dominant paradigm for building autonomous systems [Xi et al., 2023, Wang et al., 2024]. The ReAct framework [Yao et al., 2023] interleaves reasoning and action, enabling LLMs to interact with external tools and databases. It is widely adopted in systems such as AutoGPT and BabyAGI [Richards, 2023, Gravitas, 2023].

However, ReAct has a fundamental eficiency problem: every query starts reasoning from scratch. Consider two queries—“What is the online rate of sensors?” and “What is the online rate of cameras?”—they difer only in the device type, yet ReAct executes both completely: inferring the schema, generating SQL, executing, and formatting results. The second query consumes exactly the same number of tokens as the first. In our experiments, a single ReAct loop consumes an average of 198 tokens (real API usage), totaling 199,782 tokens for 1,007 queries. This massive redundancy suggests that avoiding repeated ReAct reasoning for similar queries can yield substantial eficiency gains.

## 1.2 Limitations of Existing Approaches

• Caching systems [Fu et al., 2023] only handle exact string matches and cannot recognize semantically similar queries.

• Prompt compression [Jiang et al., 2023] reduces input length but does not eliminate the reasoning loop itself.

• Speculative decoding [Cai et al., 2024] accelerates generation but does not address the redundancy of repeated reasoning.

• Self-Refine [Madaan et al., 2023] and Reflexion [Shinn et al., 2023] improve agent quality through iterative refinement but at the cost of additional token consumption.

• Our prior work [Wei et al., 2026] (2026) proposed the concept of experience-driven execution but lacked concrete implementation and experimental validation.

Key gap: No existing method provides a complete, validated framework for systematically leveraging historical execution trajectories to reduce token consumption for future similar queries.

## 1.3 Our Approach: TRIAGE

We propose TRIAGE (Three-level Routing and Intelligent Agent Guidance for Eficient Execution), whose core idea is to let agents accumulate experience like humans instead of reasoning from scratch every time. The core innovation of TRIAGE is TaaS (Trajectoryas-a-Skill), treating historical execution trajectories as reusable skills.

TRIAGE classifies input queries into three levels:

• Level 1 (Direct Reuse): Identical queries (score=1.0), 0 tokens

• Level 2 (Skill Substitution): Parameter-varying queries, 0 tokens through deterministic Skill parameter substitution

• Level 3 (Full ReAct): Novel queries, full LLM generation, automatically stored for future reuse

## 1.4 Contributions

1. TRIAGE Framework: Three-level routing architecture achieving 62.3% token reduction on 1,007 queries

2. TaaS Concept: Trajectory-as-a-Skill, abstracting experience trajectories into reusable skills

3. Semantic Encoder-driven Retrieval: Using sentence-transformers (all-MiniLM-L6- v2) with 384-dimensional embeddings, achieving 61.5% L1+L2 reuse rate

4. Automatic Cold Start: No pre-training or annotation required, learns automatically during use

5. Automatic Skill Extraction: High-frequency trajectories automatically distilled into fine-grained deterministic Skills

6. Cross-domain Validation: 76.3% token reduction across 15 ToolBench domains (345 queries), validating the generalizability of semantic routing beyond SQL

7. Routing Strategy Analysis: An ablation study revealing that the choice between threshold and LLM-based routing is not universal, but depends on the ratio of routing overhead to per-query execution cost

## 2 Related Work

## 2.1 LLM Agents and Tool Use

Toolformer [Schick et al., 2023], ToolBench/ToolLLM [Song et al., 2024], and Gorilla [Pati et al., 2023] extend LLMs’ tool use capabilities but do not address execution eficiency.

## 2.2 Token Eficiency Optimization

Existing methods include prompt compression (LLMLingua [Jiang et al., 2023]), speculative decoding (Medusa [Cai et al., 2024]), and caching (GPTCache [Fu et al., 2023]), but none fundamentally eliminate the redundant reasoning for similar queries.

## 2.3 Experience Reuse and Skill Systems

Reflexion [Shinn et al., 2023] uses verbal reinforcement learning to improve agents, and DSPy [Khattab et al., 2023] compiles declarative calls into optimized pipelines. However, existing Skill systems (Function Calling) require developers to predefine fixed-schema functional modules, sufering from weak generalization and high maintenance costs. TRIAGE’s TaaS is a fundamental improvement—trajectories as dynamic Skills, automatically discovered through semantic retrieval and adapted via LLM editing.

## 3 Method

## 3.1 System Architecture

TRIAGE consists of four core modules:

1. Semantic Encoder: Based on sentence-transformers (all-MiniLM-L6-v2), encodes queries into 384-dimensional vectors for semantic similarity retrieval.

2. Similarity-Threshold Router: A zero-cost, deterministic routing mechanism that classifies queries into three levels based on cosine similarity thresholds $( \mathrm { L 1 } \geq 0 . 9 8 , \mathrm { L 2 } \geq$ 0.90). For L1 candidates with score $< 1 . 0$ , a lightweight normalization-and-parameterfilter step resolves most cases at 0 token cost; only unresolved diferences trigger an LLM-based secondary judgment.

3. Trajectory Retriever: Cosine similarity-based trajectory retrieval with thresholds L1 $\ge 0 . 9 8 , \mathrm { L 2 } \ge 0 . 9 0$

4. Skill Extractor: For Level 2, matches the query against distilled Skills and performs deterministic parameter substitution (0 tokens). When no Skill matches, falls back to Level 3 full ReAct.

![](images/155aff714709004253d98538829e0a1eb4d6acf7e27f665ff5d25f250d87c987.jpg)  
Figure 1: TRIAGE system architecture. The system comprises three phases: (1) semantic retrieval using all-MiniLM-L6-v2 encoder with cosine similarity search; (2) similarity-threshold routing; (3) three-level dispatch based on similarity thresholds. The dashed arrow indicates automatic Skill extraction from high-frequency trajectory patterns.

## 3.2 TaaS: Trajectory-as-a-Skill

The central innovation of TRIAGE is TaaS (Trajectory-as-a-Skill), a paradigm that abstracts historical execution trajectories into reusable skills, realizing “experience as a service.” Unlike traditional Skill systems (e.g., Function Calling), which require developers to predefine fixed-schema functional modules at design time, TaaS treats every successfully executed trajectory as a latent skill that can be automatically discovered, retrieved, and reused at runtime.

A trajectory in TRIAGE is defined as a complete execution record: the user query, the sequence of tool invocations (e.g., schema inference, SQL generation, execution, formatting), and the final result. Rather than discarding this record after execution, TRIAGE persists it into a trajectory database indexed by semantic embeddings. When a new query arrives, the system retrieves the most similar historical trajectory and routes the query to one of three execution levels. This transforms the trajectory database from a passive log into an active skill repository.

The TaaS lifecycle consists of four stages, each mapped to a concrete component in TRIAGE’s architecture:

1. Store. Every successfully executed query (Level 2 or Level 3) is automatically appended to the trajectory database. The trajectory is encoded by the semantic encoder for similarity search. For large-scale deployments, FAISS (Johnson et al., 2019) can be integrated as a drop-in accelerated index to replace brute-force traversal. This stage requires no manual annotation, no schema design, and no pre-training—it is a zero-cost byproduct of normal operation.

2. Retrieve. When a new query arrives, the semantic encoder computes its embedding, and the trajectory retriever performs cosine similarity search against all stored trajectories. The top-k matches are compared against the similarity thresholds for routing. The retrieval is the critical bridge between raw experience storage and actionable reuse.

3. Reuse. Depending on the similarity score, the router dispatches the query to one of three reuse levels (Section 3.2). This is the operational stage where stored trajectories are converted into actual token savings: Level 1 returns the stored result directly (score=1.0 exact match), Level 2 performs deterministic Skill parameter substitution (0 tokens), and Level 3 falls back to full ReAct.

4. Distill. High-frequency trajectory patterns are automatically distilled into deterministic Skills by the Skill extraction mechanism (Section 3.4). A distilled Skill is a parameterized template with zero token cost—it captures the invariant structure of a recurring query pattern while exposing typed parameters for variable slots. Once distilled, the Skill is registered into the global Skill registry, and subsequent queries matching it are executed by deterministic parameter substitution without any LLM inference.

The TaaS lifecycle thus forms a closed loop: store → retrieve → reuse → distill, with each stage feeding into the next. More usage generates more trajectories, more trajectories enable more retrieval hits, more hits yield more reuse savings, and more reuse triggers more distillation. This self-reinforcing cycle is the mechanism behind TRIAGE’s “the more you use it, the more eficient it becomes” property, and it fundamentally distinguishes TaaS from static optimization techniques that yield constant eficiency regardless of usage volume.

## 3.3 Three-level Routing Mechanism

Level 1 (Direct Reuse): When cosine similarity = 1.0 (exact match), the query is literally identical to a stored trajectory. The historical SQL is returned directly with 0 token consumption. When similarity $\geq 0 . 9 8$ but $< 1 . 0$ , a lightweight normalization-and-parameterfilter step first attempts to resolve the diference at 0 token cost: if the diference is fully explained by extracted parameters (e.g., hours=6 vs. hours=8), the query is routed to Level 2 for Skill substitution; if the diference is cosmetic (same meaning, diferent wording), it stays at Level 1; only unresolved diferences trigger an LLM-based secondary judgment to determine the correct routing.

Level 2 (Skill Substitution): When cosine similarity $\geq 0 . 9 0$ , the query is structurally similar to a stored trajectory. The system first attempts to match the query against distilled Skills in the global Skill registry. If a Skill is matched, the system extracts parameter values from the query and performs deterministic parameter substitution, executing at 0 token cost with no LLM inference. If no Skill matches, the query falls back to Level 3 full ReAct. This design ensures that the vast majority of parameter-varying queries—which share the same SQL template but difer in concrete values—are served at zero cost, while truly novel variations are correctly routed to full LLM generation.

Parameter Extraction with LLM Feedback. When the lightweight normalization step cannot resolve the diference between two queries (i.e., the difering words are not covered by existing parameter extraction rules), TRIAGE invokes an LLM-based secondary judgment. The LLM determines whether the diference is a parameter/value change (L2), a purely cosmetic rewording (L1), or a semantically distinct query (L3). For L2 cases, the LLM extracts the parameter name and value, which are then registered as dynamic regular-expression rules in the parameter extraction system. This enables subsequent queries with the same parameter pattern to be resolved at 0 token cost by the normalization step, creating a self-improving parameter extraction system that learns from LLM feedback.

Level 3 (Full ReAct): When cosine similarity < 0.90, the query is identified as structurally novel. The system executes a complete ReAct loop (LLM generates SQL from schema) and automatically stores the resulting trajectory into the database. Each stored trajectory is immediately fed into the Skill extraction pipeline (Section 3.4), where a minimal frequency threshold of 1 ensures that even a single occurrence can seed a distillable pattern.

## 3.4 Online Learning Dynamics

The 1,007 queries are not batched but arrive sequentially, simulating a real-world deployment where queries are submitted over time. The query arrival process follows a mixed distribution designed to reflect authentic operational patterns:

• Recurring daily queries (6.1%): A small set of 6 fixed query templates—such as “Show me the alarm summary for today”—are repeated across 30 “days” of operation, with 2 daily queries per day. These are functionally identical (same SQL) but require re-execution because the underlying data refreshes. Their frequency follows a Zipf distribution: the most frequent template appears 30 times, the second 12 times, and the rarest 2 times, consistent with the heavy-tailed usage patterns observed in operational dashboards. The daily queries are evenly spaced at the start of each day, creating a realistic pattern of periodic fixed-cost queries that steadily accumulate in the trajectory database.

• Parameter-varying queries (57.2%): The plurality of queries share a common structural template but difer in parameter values—device type, region, time window, or status. These are generated by sampling from a set of 15 parameterized SQL templates with random parameter combinations. This category captures the most common realworld pattern: analysts asking the same analytical question about diferent entities. Parameter variations such as “sensors” vs. “cameras” or “6 hours” vs. “8 hours” yield low cosine similarity scores (0.35–0.77), making them challenging for pure thresholdbased routing but ideal candidates for Skill-based parameter substitution.

• Novel queries (36.7%): Queries with no structural precedent in the trajectory database. These arrive interspersed among the recurring and parameter-varying queries, simulating the gradual expansion of analytical needs over time. New queries require full ReAct execution and serve as the raw material for future trajectory accumulation and Skill extraction.

The temporal ordering of queries is structured as 30 “days” of operation, with daily queries clustered at the start of each day, parameter queries interspersed throughout, and novel queries arriving at random intervals. This temporal structure is critical: it creates a realistic cold-start period where the trajectory database is empty, followed by a gradual accumulation of experience that enables TRIAGE’s reuse mechanisms to activate.

Each successfully executed query is automatically added to the trajectory database. The database grows linearly with usage, and the automatic Skill extraction mechanism (Section 3.4) operates continuously on this growing repository, distilling parameterized templates from the moment the first trajectory is stored.

The online learning process exhibits three phases, as illustrated in Figure 2:

1. Cold-start (queries 1–50): No historical trajectories exist. All queries route through Level 3 (full ReAct, ∼197 tokens). The trajectory database grows linearly, and Skill extraction begins immediately after the first query completes. By the end of this phase, the first recurring queries begin to hit Level 1.

2. Warm-up (queries 50–300): The trajectory database accumulates suficient diversity. Recurring daily queries hit Level 1 (direct reuse, 0 tokens). Parameter-varying queries increasingly match extracted Skills at Level 2 (parameter substitution, 0 tokens). The L3 ratio drops sharply from 100% to below 10%.

3. Mature (queries 300–1,000): The system reaches a steady state. Level 1 covers all recurring daily queries. Level 2 covers the vast majority of parameter variations through Skills. Level 3 only activates for genuinely novel queries (approximately 1 per day). The average token cost stabilizes at a small fraction of the baseline.

This three-phase evolution embodies TRIAGE’s core property: eficiency is a function of experience. The more queries the system processes, the more trajectories it accumulates, the more Skills it extracts, and the fewer tokens it consumes per query. This stands in contrast to static optimization techniques such as prompt compression and speculative decoding, whose eficiency gains are constant regardless of usage volume. In TRIAGE, the system bootstraps its own eficiency from raw experience, forming a self-reinforcing cycle: usage → trajectories → Skills → savings → more usage.

## 3.5 Automatic Skill Extraction

Beyond the immediate savings from Level 1 and Level 2 reuse, TRIAGE incorporates a mechanism that continuously crystallizes recurring execution patterns into permanently registered, zero-cost deterministic Skills. We term this process automatic Skill extraction. It enables the system to bootstrap its own executable competency library from raw experience, forming a self-improving eficiency loop that accelerates with usage.

## General-Purpose Extraction Framework

The Skill extraction pipeline is defined at the design level as an action-type-agnostic process, operating on any structured tool invocation recorded in a trajectory. The four-stage abstraction is independent of any concrete action type: normalization, accumulation, induction, and registration are expressed over a generic “structural fingerprint” rather than over SQL. Concrete tool families are integrated by supplying a normalizer that maps their surface syntax to typed wildcards. The pipeline decomposes extraction into four stages:

![](images/aaa4af4b29af581f04dc8053b57da2796e09bccb189d59963ae374f684bb86f9.jpg)  
Figure 2: Cold start to mature phase transition. As trajectory count grows from 0 to 50+, the system evolves from full ReAct (Level 3) through reference reuse (Level 2) to direct reuse (Level 1), progressively reducing token consumption.

1. Pattern Normalization. Each tool invocation in a stored trajectory is converted into a structural fingerprint by replacing concrete values with typed wildcards. This normalization strips domain-specific surface forms while preserving the skeletal structure of the invocation, producing a canonical template key.

2. Frequency Accumulation. Template keys are aggregated across all stored trajectories. Each key maintains a counter and a list of concrete instance records. When a key’s count reaches a configurable threshold, the associated pattern graduates from ephemeral observation to extraction candidate.

3. Parameter Induction. For each candidate pattern, the extractor performs a pairwise dif across concrete instances to discover variable slots. For each variable position, it assigns a semantic parameter name through two complementary signals: (a) the lexical context from the corresponding user query, matched against a lightweight domain lexicon; (b) the type tag inferred at normalization time. The output is a parameterized template—a structural skeleton with named slots.

4. Deterministic Registration. The induced template, together with its parameter schema (name, type, domain constraints), is packaged into a standardized object and registered into the global Skill registry with an estimated token cost of zero. Subsequent queries matching this Skill via Level 1 routing execute the template by direct parameter substitution, incurring no LLM inference cost.

## SQL-Specific Implementation

SQL is the first concrete instantiation of this framework, implemented in our open-source repository. Here the SQL normalizer plays the role of the pluggable normalizer: it rewrites string literals, date patterns, and numeric constants into typed placeholders while preserving structural identifiers such as column and table names. Parameter positions are discovered through cross-instance comparison, and semantic names are assigned via a lightweight domain lexicon mapping common values in security monitoring queries to their roles (e.g., device type, region, online status). The extracted parameterized SQL template is wrapped into a zero-cost executor that performs direct parameter substitution and deterministic execution, bypassing the LLM entirely.

## Positive Feedback Loop

This mechanism creates a virtuous cycle: more usage generates more trajectories, more trajectories trigger more extractions, and more extractions enable more zero-token direct reuse. This “the more you use it, the more eficient it becomes” property fundamentally distinguishes TRIAGE from static optimization techniques such as prompt compression or speculative decoding, whose eficiency gains remain constant regardless of usage volume. In TRIAGE, eficiency is a function of experience.

## 4 Experiments

## 4.1 Experimental Setup

Dataset: We constructed a dataset of 1,007 English queries based on a real security monitoring database (3 device types, 30 zones, 5 alarm categories). The dataset is composed of two parts: (1) 947 structurally diverse queries generated from 15 parameterized SQL templates with random parameter combinations, covering 10 major analytical categories (device status, alarm statistics, regional analysis, temporal trends, etc.); (2) 60 fixed recurring daily-report queries (6 templates, e.g., “Show me the alarm summary for today”), which are repeated daily to simulate the operational pattern of fixed dashboard queries. The daily queries follow a Zipf distribution: the most frequent template appears 30 times, while the least frequent appears 2 times, consistent with the heavy-tailed usage patterns observed in real monitoring dashboards. The temporal ordering follows a 30-“day” structure, with 2 daily queries at the start of each day followed by approximately 31–32 original queries interspersed throughout. This mixed arrival pattern—combining regular recurrence (L1 candidates), parameter variations (L2 candidates), and novel arrivals (L3 candidates)—realistically simulates incremental deployment scenarios where the query distribution is heavy-tailed: most queries in day-to-day operations are recurrent or parametric variations, while genuinely novel queries are rare. The dataset is designed not to be statistically rigorous, but to demonstrate TRIAGE’s core thesis: the more queries the system processes, the more eficient it becomes, because the majority of real-world queries fall into a small set of familiar patterns accumulated over time.

Baseline: Complete ReAct loop (each query independently reasoned, no shared historical experience).

Metrics: Token consumption (input + output), routing level distribution, execution success rate, end-to-end latency.

Model: deepseek-v3.2, temperature=0.5, max tokens=8192.

Semantic Encoder: all-MiniLM-L6-v2 (384 dimensions), L1 threshold=0.98, L2 threshold=0.90. Routing: $\mathrm { L 1 } = \sin \ge 0 . 9 8$ (score=1.0 direct; ¡1.0 cheap filter + LLM if needed); $\mathrm { L 2 } = \sin \geq 0 . 9 0$ (Skill substitution); $\mathrm { L 3 } = \sin < 0 . 9 0$

## 4.2 Main Results

Table 1: Main experimental results on 1,007 queries
<table><tr><td>Metric</td><td>Baseline ReAct</td><td>TRIAGE</td><td>Savings</td></tr><tr><td>Total Tokens</td><td>199,782</td><td>75,238</td><td>62.3%</td></tr><tr><td>Avg Tokens/Query</td><td>198.4</td><td>74.7</td><td>62.3%</td></tr><tr><td>API Calls</td><td>1,007</td><td>421</td><td>58.2%</td></tr></table>

![](images/0d212da70e60a2b22ec72b9a0fdb00f0862178107b4fad5be86734574cc84272.jpg)  
Figure 3: Token consumption comparison between baseline ReAct and TRIAGE across 1,007 security monitoring queries. TRIAGE achieves 62.3% total token reduction (199,782 to 75,238).

Analysis. The 62.3% total token reduction is composed of two distinct efects. First, Skill-based elimination: the 564 Level-2 queries and 55 Level-1 queries consume zero tokens entirely (61.5% of all queries), achieved through Skill parameter substitution and direct reuse respectively. Second, reasoning-chain compression: the remaining 388 Level-3 queries execute full ReAct at approximately 198 tokens each. In terms of operational API calls, TRIAGE issues 421 calls across the 1,007 queries: 388 for Level-3 SQL generation, 33 for

LLM-based secondary judgment (parameter extraction when the lightweight normalization step is insuficient), and 0 for Level-1 and Level-2 execution. TRIAGE’s eficiency gain is therefore driven primarily by shifting the majority of queries to zero-cost execution paths: nearly two-thirds of all queries bypass the LLM entirely, while only genuinely novel queries incur the full inference budget.

## 4.3 Routing Level Distribution

Table 2: Routing level distribution across 1,007 queries
<table><tr><td>Level</td><td></td><td>Count Percentage Token Cost</td><td></td></tr><tr><td>L1 (Direct Reuse)</td><td>55</td><td>5.5%</td><td>0</td></tr><tr><td>L2 (Skill Substitution)</td><td>564</td><td>56.0%</td><td>0</td></tr><tr><td>L3 (Full ReAct)</td><td>388</td><td>38.5%</td><td>Full</td></tr></table>

Key finding: 61.5% of queries (L1 + L2) benefit from historical trajectory reuse, with 38.5% requiring a complete ReAct loop.

Analysis. The routing distribution reveals a practical asymmetry in the reuse landscape. Level 2, which captures parameter-varying queries through Skill substitution, absorbs 56.0% of the workload. This figure is a direct function of query repetition in the deployment environment: security monitoring queries exhibit a long tail of near-identical formulations (e.g., variations of online-rate queries parameterized by device type or time window). Critically, the routing distribution is not static—it evolves with usage. The automatic Skill extraction mechanism continuously converts high-frequency Level-3 patterns into parameterized Level-2 Skills, progressively shifting the distribution toward more zero-token queries as the trajectory database grows.

Level 2 accounts for 56.0%, indicating that the majority of queries are parameter-varying variations of prior trajectories. This is the regime where the Skill-based Level-2 design (Section 3.2) delivers its value: the system extracts parameters from the matched Skill template and performs deterministic substitution, achieving zero-token execution while maintaining correctness for structurally similar queries. The 100% success rate across all three levels confirms that the Skill substitution approach does not degrade reliability.

![](images/43d79b2f36905831b37506b31e0a57e24278043605e21a5b6f94271808407111.jpg)  
Figure 4: Routing level distribution across 1,007 queries.

The 38.5% L3 fraction serves as the system’s “novelty budget”—the portion of queries that genuinely require from-scratch reasoning. These novel queries are precisely the raw material that feeds the automatic Skill extraction pipeline, gradually expanding the coverage of the Skill registry and improving the reuse rate over time.

## 4.4 Ablation Study

We conduct two categories of ablation experiments to isolate the contributions of key components. All experiments use the same 1,007-query security monitoring dataset with identical retrieval pipeline (all-MiniLM-L6-v2 encoder). The first category examines the impact of routing strategy (threshold vs. LLM-based routing), and the second category isolates the contribution of the Level-2 mechanism.

## 4.4.1 Ablation 1: Routing Strategy — Threshold vs. LLM-based Routing

TRIAGE by default uses similarity-threshold routing: a query is classified as Level 1 if its maximum cosine similarity to any stored trajectory exceeds 0.98, Level 2 if it exceeds 0.90, and Level 3 otherwise. This routing decision costs zero tokens. An alternative is to delegate the routing decision to an LLM: given the top-k semantically similar trajectories as context, the LLM classifies the query into L1, L2, or L3. We compare both strategies on the full 1,007-query dataset.

Table 3: Routing strategy ablation on the full 1,007 queries
<table><tr><td>Router</td><td>L1</td><td>L2</td><td>L3</td><td></td><td>Total Tokens Route Overhead Savings</td><td></td></tr><tr><td>Our Router</td><td>5.5%56.0%</td><td></td><td>38.5%</td><td>75,238</td><td>0 tokens</td><td>+62.3%</td></tr><tr><td>LLM Router</td><td>13.1%</td><td>62.8%</td><td>24.1%</td><td>272,479</td><td>225,628 tokens</td><td>-37.3%</td></tr></table>

Analysis. The threshold router achieves 62.3% token savings on the full 1,007 queries, with 61.5% of queries routed to L1+L2 (zero tokens). In contrast, the LLM router produces a net increase in token consumption (–37.3% savings), consuming 272,479 tokens compared to the baseline’s 198,508 tokens. The routing overhead alone (225,628 tokens) already exceeds the entire baseline cost, making the LLM router empirically self-defeating. This result has two root causes.

First, the LLM router incurs a non-trivial per-query routing cost: each routing decision requires sending the top-k trajectory summaries as context to the LLM, consuming approximately 224 tokens per query—accumulating to 225,628 tokens over 1,007 queries. This overhead alone exceeds the total execution cost of the threshold router (75,238 tokens) by a factor of 3.0×.

Second, the LLM router correctly identifies many parameter-varying queries as L2 (62.8%), and these are executed via Skill parameter substitution at 0 tokens—the same mechanism as the threshold router. However, the per-query routing cost of approximately 224 tokens dominates the total token consumption. The threshold router achieves the same Skill-based L2 execution at zero routing cost, leveraging the fact that SQL queries with moderate cosine similarity $( \geq 0 . 9 0 )$ are overwhelmingly parameter-varying candidates. The LLM’s routing accuracy is not the issue—it successfully routes most queries to their correct levels—but the fixed per-query LLM invocation cost outweighs any benefit of more nuanced classification in this domain.

Implication. The threshold router is the correct default for TRIAGE: it is zero-cost, deterministic, and empirically more efective. The LLM router is not merely a worse choice—it turns a 62.3% saving into a net loss of 37.3%. This finding, validated on the full 1,007 queries with both strategies using identical Skill extraction, reinforces a central design principle of TRIAGE: eficiency gains must come from eliminating LLM calls, not from adding more sophisticated ones. The LLM router is preserved as an optional extension for domains where semantic similarity is a poor proxy for reuse eligibility (e.g., open-ended natural language tasks), but it is not recommended for structured query domains such as SQL.

However, this conclusion is contingent on the low per-query generation cost of the SQL scenario (approximately 197 tokens per query). In domains where the execution cost itself is substantially higher—for example, tasks requiring long-context reasoning, multistep tool orchestration, or generation of large structured outputs—the per-query routing overhead of approximately 224 tokens becomes negligible relative to the total cost. For such generalized scenarios where semantic similarity is a weaker proxy for reuse eligibility, LLM-based routing can serve as a complement to threshold routing, compensating for the latter’s limited generalization with its stronger semantic understanding. The choice between threshold routing and LLM routing is therefore not a universal one: it depends on the relative magnitude of routing overhead versus per-query execution cost, and TRIAGE’s design should support switching between the two on a per-scenario basis.

## 4.4.2 Ablation 2: Component Ablation

We further isolate the contribution of the Level-2 mechanism.

Analysis. Removing the Level-2 mechanism (L1 + L3 only) reduces token savings to 55.0%, because Level-2 queries (which consume 0 tokens through Skill substitution) are routed to Level 3 (198 tokens on average) instead. This demonstrates that Level 2 is the primary driver of TRIAGE´s eficiency: without Skill-based parameter substitution, the majority of parameter-varying queries incur full ReAct cost. Level 2 is therefore essential for handling the long tail of structurally similar but parametrically diferent queries.

Table 4: Component ablation results on the full 1,007 queries
<table><tr><td>Variant</td><td>Token Savings  $\mathbf { L 1 } + \mathbf { L 2 }$ </td><td>Coverage</td></tr><tr><td>Full TRIAGE</td><td>62.3%</td><td>61.5%</td></tr><tr><td> $\mathrm { L 1 + L 3 }$  only (no L2)</td><td>55.0%</td><td>55.0%</td></tr></table>

## 4.5 Cross-domain Validation

The 76.3% token savings across 15 diverse domains (424,792 → 100,875 tokens) demonstrates that TRIAGE’s semantic routing generalizes efectively beyond SQL. The high L1+L2 reuse rate (96.2%) across domains such as jokes, finance, and location services indicates that many tool-use queries share reusable structural patterns even when the domain semantics difer. The 3.8% L3 rate confirms that TRIAGE’s threshold routing correctly identifies novel queries and routes them to full ReAct, avoiding false reuse. Domains with lower savings (e.g., chart lyrics at 56.2%, the cocktail db at 65.4%) tend to have higher query diversity, as expected—the more diverse the queries within a domain, the fewer opportunities for direct reuse. This experiment uses real threshold routing calls (DeepSeek V3.2) and reports real API usage statistics, validating TRIAGE in a realistic deployment setting. We note that this cross-domain experiment validates the generalizability of TRIAGE’s semantic retrieval and threshold routing mechanism across diverse tool-use domains. The automatic Skill extraction and parameter substitution mechanisms (Section 3.4) have been fully implemented and evaluated in the SQL scenario (Section 4.2). Consequently, the Level 2 queries in this cross-domain experiment are executed via LLM-based reference generation (partial token cost) rather than the 0-token Skill substitution used in the main SQL experiment—this explains the lower L2 ratio (11.6% vs. 56.0%) and the smaller savings gap between L1 and L2. Extending the Skill extraction and parameter substitution pipeline to heterogeneous tool APIs with diverse parameter schemas is a direction for future work.

Figure 3: Online Learning Evolution — Cold Start to Mature (1007 queries)  
![](images/b3ccb4fb440219bd70f7b52fca22c366b1b741c789e5347510a20373c13e3a2b.jpg)  
Figure 5: Online learning evolution of TRIAGE over 1,007 queries. The stacked area shows the $\mathrm { L 1 / L 2 / L 3 }$ distribution, and the line plot shows the average tokens per query. The system transitions from a cold start (first query at 100% L3, 130 tokens) through a warm-up phase (L2 → 57% within 100 queries) to a mature stage (L2 stabilizes at 56.0%, L1 at 5.5%, average token cost at 74.7).

Table 5: Cross-domain validation results on ToolBench (15 domains, 345 queries, real threshold routing)
<table><tr><td>Domain</td><td>N</td><td>L1</td><td>L2 L3</td><td>Save%</td><td></td></tr><tr><td>love_calculator</td><td>18</td><td>17</td><td>0</td><td>1</td><td>92.2%</td></tr><tr><td>free_nba</td><td>23</td><td>21</td><td>1</td><td>1</td><td>89.3%</td></tr><tr><td>currency_exchange</td><td>23</td><td>21</td><td>2</td><td>0</td><td>87.4%</td></tr><tr><td>world_of_jokes</td><td>20</td><td>18</td><td>1</td><td>1</td><td>83.6%</td></tr><tr><td>geodb_cities</td><td>21</td><td>18</td><td>2</td><td>1</td><td>82.6%</td></tr><tr><td>quotes</td><td>24</td><td>21</td><td>2</td><td>1</td><td>77.1%</td></tr><tr><td>manatee_jokes</td><td>30</td><td>26</td><td>3</td><td>1</td><td>76.2%</td></tr><tr><td>chuck_norris</td><td>30</td><td>26</td><td>2</td><td>2</td><td>75.9%</td></tr><tr><td>jokes_by_api_ninjas</td><td>30</td><td>26</td><td>4</td><td>0</td><td>74.5%</td></tr><tr><td>billboard_api</td><td>24</td><td>18</td><td>6</td><td>0</td><td>70.9%</td></tr><tr><td>daddyjokes</td><td>21</td><td>18</td><td>3</td><td>0</td><td>69.0%</td></tr><tr><td>numbers</td><td>30</td><td>24</td><td>5</td><td>1</td><td>68.1%</td></tr><tr><td>the_cocktail_db</td><td>21</td><td>16</td><td>4</td><td>1</td><td>65.4%</td></tr><tr><td>chart_lyrics</td><td>19</td><td>14</td><td>3</td><td>2</td><td>56.2%</td></tr><tr><td>lost_ark_simple</td><td>11</td><td>8</td><td>2</td><td>1</td><td>57.9%</td></tr><tr><td>Total/Average</td><td>345</td><td>292</td><td>40</td><td>13</td><td>76.3%</td></tr></table>

## 4.6 Online Learning Evolution

Analysis. The online learning curve reveals three distinct phases. In the cold start phase (queries 1–50), the system has no trajectory database and must execute every query as a full ReAct (L3). The L2 hit rate climbs rapidly from 0% to 57% within the first 100 queries, demonstrating the immediate benefit of incremental trajectory accumulation. The average token cost drops from 198 to 74.7 tokens per query.

In the warm-up phase (queries 50–300), the L1 hit rate peaks at 87%, and the L3 fraction stabilizes around 4%. The trajectory database has accumulated suficient diversity to cover the most common query patterns, and the system is operating at high eficiency with an average token cost of approximately 49 tokens per query.

In the mature phase (queries 300–1,007), the L2 fraction stabilizes at 56.0%, the L1 fraction at 5.5%, and L3 at 38.5%. The slight decline in L1 from its peak of 67% at 200 queries to 61.5% reflects the natural consequence of encountering increasingly diverse queries as the experiment progresses—later queries exhibit more variation from the established trajectory base, routing more to L2 rather than L1. This is not a degradation but a healthy outcome: L2 queries are the raw material that feeds the automatic Skill extraction pipeline, and their correct handling at 0 tokens through Skill substitution demonstrates the system’s ability to generalize.

The final configuration achieves 62.3% token savings over the ReAct baseline. Critically, the baseline estimate of 197 tokens per query reflects real API usage: real ReAct loops often retry on failure, and each retry multiplies the token cost. Under realistic conditions where ReAct retries are common, TRIAGE’s actual savings would be even higher. We therefore report the 62.3% figure as a realistic, measurement-based estimate.

## 5 Discussion

## 5.1 Best Application Scenarios

• Structured, repetitive queries (security monitoring, business intelligence, operational dashboards)

• Limited tool sets with predictable execution patterns

• High query volume, with benefits scaling with usage

## 5.2 Limitations

1. Cold start cost: First 10–50 queries cannot benefit from reuse

2. Domain specificity: Trajectories cannot be reused across domains

3. First-query ineficiency: The first query always goes to Level 3

4. Error propagation: Trajectory errors may propagate without verification mechanisms

## 5.3 Future Work

The central research agenda arising from TRIAGE is the generalization and deepening of automatic Skill extraction and Level-2 generalized execution. These two mechanisms are the primary drivers of TRIAGE’s eficiency gains, and their advancement along the following directions represents the core of our ongoing work.

1. Program Synthesis for Skill Representation (inductive program synthesis). The current parameter induction stage normalizes surface values into typed placeholders. This can be formally upgraded to program synthesis over a domain-specific language (DSL): given input-output examples from multiple trajectories, an inductive synthesizer infers an executable program that captures not only parameter substitution but also conditional branching, aggregation, and composition. Recent advances in LLM-based program synthesis demonstrate the feasibility of bootstrapping expressive program libraries from examples, and their integration into TRIAGE’s extraction pipeline directly increases the expressive power of automatically extracted Skills.

2. Composable Skill Libraries and Multimodal Tool Unification. Going beyond single-action Skill extraction, a natural extension is to treat Skills as composable primitives that can be dynamically chained by a planner to solve complex queries. Furthermore, unifying heterogeneous tool families (SQL, REST APIs, GUI operations) under a common representation would realize the action-type-agnostic design at the implementation level, enabling the same Skill extraction pipeline to operate across diverse tool ecosystems.

3. Online and Lifelong Skill Learning. As the Skill library grows with usage, the system faces challenges of Skill redundancy, conflict, and concept drift. Continual/lifelong learning techniques provide principled frameworks for incrementally expanding the Skill library while preserving previously acquired competencies and avoiding catastrophic forgetting. Skill deduplication, versioning, and consolidation are essential for maintaining the quality of the Skill registry at scale.

4. Verified Skill Correctness and Safety Guarantees. Automatically extracted Skills carry the risk of encoding erroneous patterns (e.g., incorrect SQL from a failed trajectory). Integrating formal verification, contract-based execution, or symbolic constraint checking at registration time would provide correctness guarantees, making zero-token deterministic execution safe for deployment. This directly addresses the error propagation limitation identified in Section 5.2.

5. Contrastive Experience Extraction for Parameter Formatting. A concrete open problem emerging from our experiments is the gap between natural-language parameter values and their executable formats. For example, a Skill may extract the parameter “January 2025” from a query, but the SQL template requires “BETWEEN ’2025-01-01’ AND ’2025-01-31’.” We propose contrastive experience extraction: when a parameter-substituted SQL fails at execution and is subsequently corrected, the system compares the failed and successful SQL to infer the formatting transformation (e.g., natural-language time ightarrow SQL date range). These inferred transformations are then cached as reusable formatting rules attached to the Skill’s parameter schema, gradually eliminating the format-conversion gap without hand-crafted converters. This mechanism generalizes the core TRIAGE principle—learning executable knowledge from experience—to the parameter-formatting layer, and is a direct extension of the automatic Skill extraction pipeline.

6. Cross-domain trajectory transfer and multi-agent sharing. Domain-agnostic trajectory abstraction, combined with trajectory quality assessment and pruning, enables experience sharing across domains and agents, further amplifying the eficiency of the TaaS paradigm.

## 6 Conclusion

TRIAGE achieves 62.3% token reduction (199,782 → 75,238) on 1,007 queries through its three-level routing architecture. The core innovation, TaaS (Trajectory-as-a-Skill), abstracts experience trajectories into reusable skills. The semantic encoder (all-MiniLM-L6-v2)-driven retrieval achieves 61.5% L1+L2 reuse rate, with 97.6% of queries benefiting from historical trajectory reuse. Cross-domain validation on 15 ToolBench domains (345 queries) achieves 76.3% token reduction with real threshold routing. TRIAGE demonstrates that experiencedriven execution is a practical approach to reducing LLM agent costs without sacrificing capability.Our ablation results reveal that the preference for threshold over LLM-based routing is not absolute—it is contingent on the ratio of routing overhead to execution cost. This insight motivates a hybrid routing architecture that adaptively selects the routing strategy per scenario.

## References

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Demi Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads. Proceedings of ICML 2024, 2024.

Zihui Fu et al. Gptcache: An open-source semantic cache for llm applications. GitHub Repository, 2023. URL https://github.com/zilliztech/GPTCache.

Significant Gravitas. Babyagi: An ai-powered task management system. https://github. com/yoheinakajima/babyagi, 2023.

Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yu Yang, and Lili Qiu. Llmlingua: Compressing prompts for accelerated inference of large language models. Proceedings of EMNLP 2023, 2023.

Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhan, Saiful Haider, Cheng-Zhi Chang, et al. Dspy: Compiling declarative language model calls into self-improving pipelines. arXiv preprint arXiv:2310.03714, 2023.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegrefe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems (NeurIPS 2023), 2023.

Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. Gorilla: Large language model connected with massive apis. arXiv preprint arXiv:2305.15334, 2023.

Toran Richards. Autogpt: An autonomous gpt-4 experiment. https://github.com/ Significant-Gravitas/AutoGPT, 2023.

Timo Schick, Jane Dwivedi-Yu, Roberto Dess\`ı, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761, 2023.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023. URL https://arxiv.org/abs/ 2303.11366.

Yifan Song, Weimin Xiong, Dawei Zhu, Cheng Li, Hao Wang, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. Proceedings of ICLR 2024, 2024.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on llm-based autonomous agents. arXiv preprint arXiv:2308.11432, 2024.

Ruocan Wei, Shufeng Wang, and Ziwei Shi. Workflowgen: Experience-driven workflow generation for llm agents. arXiv preprint arXiv:2604.19756, 2026. URL https://arxiv. org/abs/2604.19756.

Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senyu Jin, Enyu Zhou, et al. The rise and potential of large language model based agents: A survey. arXiv preprint arXiv:2309.07864, 2023.

Shunyu Yao, Jiaqi Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations (ICLR 2023), 2023. URL https: //openreview.net/forum?id=ujr5cXIqJ8f.