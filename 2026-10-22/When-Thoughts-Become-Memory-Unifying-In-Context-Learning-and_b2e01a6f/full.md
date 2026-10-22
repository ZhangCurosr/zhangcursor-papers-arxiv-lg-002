# When Thoughts Become Memory: Unifying In-Context Learning and Fine-Tuning

Leon Chlon<sup>1</sup> Julie Huang<sup>1</sup>

## Abstract

We are the first to give a certified rule for whether unseen data belong in a pretrained model’s context or its weights, built on an exact law for their information value. That value, the code length saved on held-out queries, equals in expectation the bits stored about the query, plus a prior repair, minus a decoding loss, for any model, architecture, dataset and task. The law’s ceilings certify the choice from one measurement, and an anytimevalid test makes it from held-out queries, with error controlled whenever it stops. Every certificate the ceilings issue on a trained transformer, TabPFN, an occupancy network and a stereo network is correct. The test picks TabPFN over XGBoost and a trained surrogate over an in-context learner on airfoil pressure, calls the better of GPT-2’s cache and fine-tune in 61,948 of 64,000 query streams over 32 comparisons, and errs in at most 3.3% of null streams. The law also diagnoses when learning in context helps and when it does not. Deleting part of a context is held-out validation of that part in any model, and adding parts one at a time traces the learning curve in context: one patch that sees a shape feature cuts an occupancy network’s code length there a hundredfold, while eight that miss it nearly double it, and one camera that sees the surface beats four in a stereo network. Ranked by this label-free valuation, a kriging learner’s three best support cars beat four drawn at random, and on a yes-or-no claim the valuation recovers a published evidence adjudicator, with AUROC 0.91.

## 1. Introduction

A pretrained model holds unseen data in one of two places. Read as context, they live in a state the frozen model computes afresh in every forward pass: the feature grid of an occupancy network, the cost volume of a stereo network, the kriging weights of a kriging learner, the key–value cache of a language model. Written by fine-tuning, they become part of the weights. Every model that reads context poses the same question: when should a thought held in context become a memory in the weights? Current accounts of incontext learning, as implicit Bayesian inference (Xie et al., 2022) or as gradient descent carried out by attention (von Oswald et al., 2023; Akyürek et al., 2023), describe stylised models of language and of linear regression, and give no quantity to answer it with.

We resolve it with an information-theoretic construction. Context and weights are both valued by the bits they save the model on held-out queries, and both obey one exact law that separates what a memory stores from what its decoding loses. The two channels can then be compared term by term, and capacity bounds on each term become certificates that choose a channel from one measurement. Deleting part of the context and rerunning the model gives the value of that part in any model, and with it a test of whether evidence supports a claim. The construction asks a model for two things only: a probability for each outcome and a rerun on part of its context. Every tool runs on these two, and the certificates also need exact information terms. Nothing else is needed from the dataset, the architecture or the task, so a hosted model supplies everything through its log-probabilities, and a local model adds its weights, which give the certificates their capacity bounds. We prove what follows from this construction and validate it on language, vision, fluid-dynamics and tabular models.

In expectation the saving splits exactly into three terms, for the context memory and the weights alike (Theorem 2): the information the memory stores about the query; a prior repair, which corrects the model’s prior over outputs and is available to any context, even one that says nothing about the query; and a decoding loss, by which the model’s prediction falls short of the best prediction from its memory. The split refines the resolution–reliability decomposition of the logarithmic score (Bröcker, 2009) by conditioning on the memory a channel builds. That conditioning makes the two channels comparable: it separates what a memory stores from what its decoder uses, and when the context fits the window it caps what any fine-tune can add at the decoding loss in context. No previous work has made context and weights comparable this way or turned the comparison into a certified rule.

The law decides between the two. Beyond the prior repair they share, a fine-tune cannot save more bits than it stores, and a context memory no more than its window holds, so one channel’s measured gain, set against the other’s ceiling, can certify the choice. A full fine-tune opens no new space for what it learns: it writes into the bits that already hold the model. Prequential learning curves, read from the context itself, and an anytime-valid test on held-out queries make the choice against any model trained on the same data, with error controlled whenever the test stops. A storage budget sets how many bits fixed weights must store to reproduce the predictions made in context (Section 5).

Table 1. The results at a glance: what each tool takes, what it guarantees, and how the paper tests and checks it.
<table><tr><td>Tool</td><td>What it takes</td><td>What it guarantees</td><td>Tested on, and checked by</td></tr><tr><td colspan="4">With the information terms of the data</td></tr><tr><td>Exact law (Thm. 2)</td><td>held-out code lengths</td><td> $\mathbb { E } [ \Gamma _ { c } ] = I ( Q ; Z _ { c } ) + { \mathrm { K L } } _ { 0 } - \deg _ { c } , \operatorname { i n }$  context and in the weights alike</td><td>every term in closed form for a linear-Gaussian model, matched by simulation (App. A.10)</td></tr><tr><td>Certificates (Prop. 6(ii))</td><td>one channel&#x27;s code length, the other&#x27;s ceiling</td><td>the channel beats every memory confined to the window, or of at most C bits</td><td>25 of 64 on a trained transformer and 36 of 96 on TabPFN, occupancy and stereo networks, all correct</td></tr><tr><td colspan="4">From the model&#x27;s probabilities and reruns alone</td></tr><tr><td>(Prop. 6(iii))</td><td>Decoding-loss bound a fine-tune&#x27;s advantage, context in the window</td><td>no fine-tune adds more than the decoding loss in context</td><td>GPT-2: at least 0.35–0.55 bits per token, prior repair the cache leaves unrealised</td></tr><tr><td>Anytime-valid test (Thm. 8)</td><td>held-out code lengths of both channels</td><td>a wrong call with probability at most α, whenever the test stops</td><td>tables, airfoil pressure and GPT-2; at most 3.3% wrong calls on nulls, within 5%; a power curve; the same call in 200 reorderings</td></tr><tr><td>Prequential curves (Prop. 7)</td><td>the code lengths of the context&#x27;s own units</td><td>each equals the held-out learning curve at that size, in expectation</td><td>GPT-2: within 1.97 standard errors of the held-out curves; sums within 0.1% and 1.5%</td></tr><tr><td>Storage budget (Thm. 9)</td><td>in-context outputs over resampled contexts</td><td>a fixed state within ε bits of the context memory stores at least I(Ý; D) – ε bits</td><td>GPT-2 on tables of facts: the estimate stays below the exact 5.14 bits over all 4,096 tables</td></tr><tr><td>Value of a part of the context (Cor. 4)</td><td>a rerun without the part</td><td>its exact leave-one-out value, label-free as IG, split into content and form</td><td>occupancy, stereo and kriging models; claims fixed in advance, matched controls, native stereo and kriging</td></tr><tr><td>Evidence budget (Cor. 16)</td><td>a claim&#x27;s probability with</td><td>the bits a claim needs, against the bits its</td><td>code reproduced exactly in 12 and 168 comparisons two language-model verifiers on VitaminC claims, AUROC 0.91</td></tr></table>

The choice follows the data, to context or to the weights. On two OpenML tables the test picks TabPFN, which learns from the rows in its context, over gradient-boosted trees trained on the same 128 rows, in every seed. On airfoil pressure fields it picks a trained surrogate over an in-context learner in every seed, and with 32 designs instead of 128 it picks the in-context learner in two seeds of three. On simulated streams with no true winner it calls one in at most 3.3% of them, within its 5% limit. Where the information terms are exact, the ceilings issue 25 of 64 possible certificates on a trained transformer and 36 of 96 on TabPFN, an occupancy network and a stereo network, and every one agrees with the measured winner (Section 6).

The law also tells when learning in context helps. Deleting a part of the context and rebuilding the memory from the rest is held-out validation of that part, its exact leave-one-out value, run in the forward pass with the weights untouched in any architecture, and a matched control of the same form separates what the part says from the form it takes.

Measured this way, the bits follow the evidence each query depends on. In an occupancy network, the 300 points that observe a local shape feature are worth 17.66 bits per query, while 300 matched points from unchanged geometry are worth 0.001 bits. In a stereo network, a camera that sees the queried surface carries three times the bits of one that does not. The learning curve in context turns on which items arrive. One patch that observes the feature takes the occupancy network’s code length at the feature from 4.99 to 0.05 bits per site, while eight patches that miss the feature raise it to 9.06, so more context makes the prediction worse. A kriging learner that ranks its support cars by its own valuation codes better with three than with four drawn at random. For two language-model verifiers, the law recovers the Trace Budget adjudicator (Chlon et al., 2026): the bits a binary claim needs, set against the bits its evidence supplies, separate supporting from insufficient evidence with AUROC 0.91 (Section 4).

## Contributions. Table 1 states the results.

• Compare context and weights term by term, through an exact law for the expected gain of a context memory and of a fine-tune (Section 3).

• Certify the choice against a whole class from one measured code length, every memory confined to a window or every state of at most � bits, without training it; and, when the context fits the window, bound a model’s decoding loss from below with one fine-tune (Section 5).

• Value each part of a context exactly, and without labels, in any model, separating what it says from its form and tracing learning curves in context (Section 4).

• Choose with error controlled at any stopping time when the information terms are unknown, with evidence across language, vision, fluid-dynamics and tabular models (Section 6). The appendix writes a transformer’s cache exactly as a weight write with query-dependent step sizes and gives a linear-Gaussian model in which every term is closed form.

## 2. Learning from context

All logarithms are base two and all information is in bits, except in Appendices $\ \mathrm { A } . 2 \mathrm { - } \mathrm { A } .$ .4 and their proofs, which work with logits in nats. The code length of a query $Q$ given a context � is $\ell _ { \theta } ( Q \mid c ) = - \log _ { 2 } p _ { \theta } ( Q \mid c )$ , the bits the model needs to encode the observed outcome: for occupancy at a site, a depth at a pixel or a pressure at a surface point, the negative log-probability or log-density of the observed value, and for a sequence $Q = ( q _ { 1 } , \dots , q _ { m } )$ , the sum $\textstyle - \sum _ { t } \log _ { 2 } p _ { \theta } ( q _ { t } \mid c , q _ { < t } )$

A frozen model $\theta _ { 0 }$ with base predictive $p _ { 0 } = p _ { \theta _ { 0 } } ( { \cdot } | \emptyset )$ can absorb a dataset � in two ways. In context (channel A), it computes from � a memory $Z _ { A }$ , such as a feature grid, a cost volume, a set of kriging weights or a key–value cache, and predicts $p _ { A } ( Q \mid D ) \stackrel { - } { = } p _ { \theta _ { 0 } } ( \stackrel { - } { Q } \mid Z _ { A } )$ . By fine-tuning (channel B), it writes � into its weights, $Z _ { B } = \theta _ { T } ( D )$ , and predicts $p _ { B } ( Q \mid D ) = p _ { \theta _ { T } ( D ) } ( Q \mid \mathbf { \bar { \theta } } )$ . A learner without trained weights, such as a kriging learner, has only channel A. Each channel’s gain is

$$
\Gamma _ { c } ( Q ; D ) = \ell _ { \theta _ { 0 } } ( Q \mid \emptyset ) - \ell _ { c } ( Q \mid D ) = \log _ { 2 } \frac { p _ { c } ( Q \mid D ) } { p _ { 0 } ( Q ) } .\tag{1}
$$

For a transformer the context memory is the key–value cache (the cache, from here on), and two teacher-forced passes, one over � and one over � followed by $Q ,$ , measure its gain, $\Gamma _ { A } = \ell ( D ) + \ell ( Q ) - \ell ( D \oplus Q )$ . For a part � of the context and the rest $R , D = ( R , G )$ , deleting � and rebuilding the memory from � gives the ablation gain of $G ,$

$$
\Delta _ { G } ( Q ) = \ell _ { c } ( Q \mid R ) - \ell _ { c } ( Q \mid D ) = \Gamma _ { c } ( Q ; D ) - \Gamma _ { c } ( Q ; R ) .\tag{2}
$$

Assumption 1 (Clean memory). Each memory is built from its own data and exogenous randomness, so that $Q \perp Z _ { c } \mid D$ After a deletion the memory is rebuilt from � alone, and no state computed from � is reused.

A deletion reruns the computation that builds the memory on what remains: the encoder of the feature grid, the construction of the cost volume, the fit of the kriging learner, and in a transformer every cache entry after the deleted span, because those entries have already absorbed the span, and masking attention to it would leave its information in place. The rebuilt memory carries no trace of the deleted evidence (Lemma 10 in Appendix A.1), so a deletion is valued at what the evidence adds beyond the rest, as held-out validation values a training example at what the rest of the training set does not supply.

## 3. The law of in-context gain

The truth is $( D , Q ) \sim P ^ { * }$ , with true posterior predictive $P ^ { * } ( Q \mid D )$ and marginal $P _ { Q } ^ { * } .$

![](images/555cd1a039faa8906f4840b61540bf7ba355c172c5c0af323f665f0017829872.jpg)  
Figure 1. Both channels, measured in the same bits. The data can give at most $G ^ { \star } = I ( Q ; D ) + { \mathrm { K L } } _ { 0 }$ bits per query in expectation. Each channel keeps its expected gain and loses the rest in two parts: the storage loss, which its memory drops, and the decoding loss, by which its prediction falls short of the best prediction from its memory (Equation (4)).

Theorem 2 (Exact law). Under Assumption 1, for either channel,

$$
\mathbb { E } [ \Gamma _ { c } ] = \underbrace { I ( Q ; Z _ { c } ) } _ { b i t s s t o r e d } + \underbrace { \mathrm { K L } ( P _ { Q } ^ { * } \parallel p _ { 0 } ) } _ { p r i o r r e p a i r } - \underbrace { \mathbb { E } \mathrm { K L } \big ( P _ { Q | Z _ { c } } ^ { * } \big \| p _ { c } ( \cdot \mid Z _ { c } ) \big ) } _ { d e c o d i n g l o s s \mathrm { d e c } _ { c } } .\tag{3}
$$

Proof. Pointwise, with $Z = Z _ { c }$

$$
\log \frac { p _ { c } ( Q \mid Z ) } { p _ { 0 } ( Q ) } = \log \frac { P ^ { * } ( Q \mid Z ) } { P ^ { * } ( Q ) } + \log \frac { P ^ { * } ( Q ) } { p _ { 0 } ( Q ) } - \log \frac { P ^ { * } ( Q \mid Z ) } { p _ { c } ( Q \mid Z ) } .
$$

Take expectations under $P ^ { * }$

Conditioning on the forecast $F _ { c } = p _ { c } ( \cdot \mid Z _ { c } )$ in place of the memory turns the first and last terms into the resolution and reliability of the logarithmic score (Bröcker, 2009; Weijs et al., $2 0 \dot { 1 } 0 )$ , and the decoding loss exceeds that reliability by $I ( Q ; Z _ { c } \mid F _ { c } )$ , the information the memory holds and its decoder leaves unused (Appendix B.2). The ceilings in Section 5 act on the bits stored, and the decoding-loss bound on the bits used.

Because $Q \mathrm { ~ - ~ } D \mathrm { ~ - ~ } Z _ { c }$ is a Markov chain, $I ( Q ; Z _ { c } ) = $ $I ( Q ; D ) - I ( Q ; D \mid Z _ { c } )$ , and the law reads

$$
\begin{array} { c } { \mathbb { E } [ \Gamma _ { c } ] = G ^ { \star } - \underbrace { I ( Q ; D \mid Z _ { c } ) } _ { \mathrm { s t o r a g e ~ l o s s ~ s t o r } _ { c } } - \mathrm { d e c } _ { c } , } \\ { G ^ { \star } = I ( Q ; D ) + \mathrm { K L } ( P _ { Q } ^ { * } \parallel p _ { 0 } ) = \mathbb { E } _ { D } \mathrm { K L } ( P _ { Q | D } ^ { * } \parallel p _ { 0 } ) . } \end{array}\tag{4}
$$

The storage loss is the information about the query that the memory drops, and $G ^ { \star }$ is everything the data can give. Figure 1 draws both channels in these terms.

Corollary 3 (Ceiling, channels and content-free context). $( i ) \mathbb { E } [ \Gamma _ { c } ] \ \leq \ G ^ { \star }$ , with equality if and only if $p _ { c } ( \cdot \mid D ) =$ $P ^ { * } ( \cdot \mid D )$ . (ii) For the two channels,

$$
\begin{array} { r } { \mathbb { E } \bigl [ \Gamma _ { A } - \Gamma _ { B } \bigr ] = \mathbb { E } \bigl [ \mathrm { K L } ( P _ { Q | D } ^ { * } \parallel p _ { B } ) - \mathrm { K L } ( P _ { Q | D } ^ { * } \parallel p _ { A } ) \bigr ] . } \end{array}\tag{5}
$$

(iii) If a context $\tilde { D }$ carries no information about the query, $I ( Q ; { \tilde { D } } ) = 0 ;$ , then $\mathbb { E } [ \Gamma _ { c } ( Q ; \tilde { D } ) ] \stackrel {  } { \leq } \mathrm { K L } ( P _ { O } ^ { * } \| p _ { 0 } )$

Whichever channel lands closer to the true posterior predictive wins, by exactly the difference of their divergences from it (part (ii)). Part (iii) says what a context with no information about the query can do: repair the prior, and nothing more. This is why demonstrations with random labels still help (Min et al., 2022): they fix the format, the label space and the domain, and that alone is worth up to KL $( P _ { Q } ^ { * } \| p _ { 0 } )$ bits.

![](images/7e84080bb819dca642ea138e07e7267eed3941d01656dc3298d0d3caf3b866ab.jpg)

![](images/5f291e205372d97fd07a954ee411fa9c94c187cb9f3ac12a19991970688c9b02.jpg)

![](images/c97b19e9e40cb09ca0a9d6866803aab536a13f30f5b0673d9861ad24b2b5b738.jpg)  
Figure 2. Learning curves in context: held-out code length against the number of context items, by order of arrival, with the memory rebuilt from scratch for each set of items. (a) Occupancy network on eight constructed chair pairs: 300 anchor points and up to nine patches of 300 points, one of which observes the feature; code length at the feature sites. (b) MVSNet on five DTU scenes: one to four source views, at the 3,856 pixels that some views observe and others do not, with the views ordered separately at each pixel. (c) Kriging learner on eight DrivAerNet cars: two to six support cars, ranked by the gain $I _ { G }$ its own predictions assign each car; log-density code length, which can be negative. Random orders average four orders, or in (b) all subsets of each size; means over cases.

Corollary 4 (Ablation). For $D = \left( R , G \right)$ , write stor<sub>�</sub> $=$ $I ( Q ; X \mid { \bar { Z } } _ { X } )$ and dec for the storage and decoding losses ofthe memory builtfrom �. Then

$$
\operatorname { \mathbb { E } } [ \Delta _ { G } ] = I ( Q ; G \mid R ) + \left( \operatorname { s t o r } _ { R } - \operatorname { s t o r } _ { D } \right) + \left( \operatorname { d e c } _ { R } - \operatorname { d e c } _ { D } \right) .\tag{6}
$$

The prior repair cancels. Thefirst term is the information � carries about the query beyond the rest, and the storage terms vanish when the memory keeps its context, as a transformer’s cache does within its window (Proposition 5).

A matched deletion $\tilde { G }$ removes a part with the form of a relevant one and content that says nothing about the query beyond the rest, $I ( Q ; { \tilde { G } } \mid D \setminus { \tilde { G } } ) = 0 $ . Its expected gain, E $\Delta _ { \tilde { G } } ] = ( \mathrm { s t o r } _ { D \backslash \tilde { G } } - \mathrm { s t o r } _ { D } ) + ( \mathrm { d e c } _ { D \backslash \tilde { G } } - \mathrm { d e c } _ { D } )$ , is the cost of the form alone. A relevant deletion of the same form differs from it only in content, so content accounts for the whole difference between the two gains, whichever terms of Equation (6) carry it: removing what the decoder reads changes its decoding as well as its information. With Corollary 3(iii), the pair tests whether a part of the context helps through what it says or only through its form, including any repair of the format.

The model’s own valuation. Draw the outcome from the model’s own full-context prediction, $Q \sim p _ { c } ( \cdot \mid D )$ . The full context then has no decoding loss, and the expected ablation gain is

$$
\begin{array} { r } { I _ { G } : = \mathbb { E } _ { Q \sim p _ { c } ( \cdot \vert D ) } \left[ \Delta _ { G } ( Q ) \right] = \mathrm { K L } \big ( p _ { c } ( \cdot \vert D ) \big \Vert p _ { c } ( \cdot \vert R ) \big ) \geq 0 , } \\ { ( 7 ) \qquad } \end{array}
$$

the code length the model saves on its own full-context continuation. It needs no labels, and it is defined for each context separately. For a fixed context, Corollary 4 has no information or storage term: $I _ { G }$ is the decoding loss of the memory rebuilt from �, measured against the full-context prediction. A single deletion can therefore be worth many bits even for a binary outcome, when it overturns a confident prediction.

## 4. Valuing context in any memory

The law turns context ablation into a measurement for any model that conditions on context. The evaluator (Algorithm 1 in Appendix C.1) asks three things of a model: a predictive distribution or a claim probability, a deletion after which the memory can be rebuilt from what remains (Assumption 1), and the same outputs scored with and without the deleted part. Nothing in it is specific to an architecture. For each deletion it records $I _ { G }$ and, for each claim fixed in advance, the claim’s probability with and without the deleted evidence, and it pairs each relevant deletion with a matched control of the same form (Section 3). We run it on four memories (Table 2 in Appendix C.2): the feature grid of a pretrained convolutional occupancy network, ConvONet (Peng et al., 2020), on constructed chairs and ShapeNet objects (Chang et al., 2015); the cost volume of a multi-view stereo network, MVSNet (Yao et al., 2018), on DTU scenes (Jensen et al., 2014); the kriging weights of a kriging learner on DrivAerNet cars (Elrefaie et al., 2024); and the key–value caches of two language-model verifiers on VitaminC (Schuster et al., 2021).

Deleting the 300 points that observe a local shape feature and rebuilding the grid drops the mean occupancy probability at the feature sites from 0.9827 to 0.0014 and costs 17.66 bits per query, while deleting 300 matched points from unchanged geometry costs 0.001 bits (Figure 6a in Appendix C.2). Deleting a source view that observes the queried surface point costs 1.62 bits per depth query, against 0.55 bits for one that does not, and takes 9.52 percentage points more off the probability that the depth lies within five depth units of its full-context estimate, with the larger loss in all five scenes (Figure 6b).

Adding a context’s items one at a time traces the model’s learning curve in context. Each step is the ablation gain of the item added. The information an item adds is never negative, so by Corollary 4 a step that raises the expected loss is one in which the item raised the storage and decoding losses, together, by more than the information it brings.

The curves turn on which items arrive (Figure 2). With 300 anchor points the occupancy network codes the feature sites in 4.99 bits per site. Added first, the one patch that observes the feature takes this to 0.046 bits, while the eight that miss the feature raise it to 9.06, and the feature patch, added after them, brings it down to 0.015. One source view that observes the queried point codes its depth in 8.71 bits, against 13.65 for a view drawn at random and 10.17 for all four views together. The kriging learner can rank its support cars without labels, by the gain $I _ { G }$ its own full-context predictions assign each car. With the top three it codes the pressure better than with four drawn at random, −1.61 against −1.41 bits per value (a log-density code length can be negative).

Fixing a claim turns the gain into a budget, set against the bits the claim needs. If a claim has probability $p$ with the full context and $q$ after a deletion, the deletion supplies $I _ { G } \geq d _ { 2 } ( p \Vert q )$ bits, where $d _ { 2 }$ is the binary divergence, and lifting the claim from � to a target $t > q$ or higher needs at least $d _ { 2 } ( t \| q )$ bits (Corollary 16 in Appendix A.7). The 300 feature points supply 17.66 bits against a requirement of 16.93, and every confident feature claim falls without them, while the 300 matched points supply 0.001 bits against 0.006. An observing view supplies 1.62 bits against 1.27 and accounts for 2.4 times as many lost claims as a nonobserving view (Table 5 in Appendix C.2). With a binary claim and an evidence passage as the deleted part, the budget is the Trace Budget adjudicator of Chlon et al. (2026), and the law places it inside the decomposition: for one context, the bits the evidence supplies are the decoding loss of the memory rebuilt without it, measured against the fullcontext prediction. A language-model verifier scores the claim with its evidence present and removed, and the budget gap, the bits the claim requires minus the bits its evidence supplies, separates supporting from insufficient evidence with AUROC 0.91 for both Qwen2.5-7B and GPT-4.1-mini on VitaminC.

## 5. Choosing between context and weights

The two channels write differently, a transformer’s cache as a weight update whose step sizes are recomputed for every query and a fine-tune as one fixed update (Appendices ${ \mathrm { A } } . 3 -$ A.5), and they differ in what they can store.

Proposition 5 (Capacity). By data processing, the storage loss is at least $[ I ( \bar { Q } ; D ) - I ( D ; Z _ { c } ) ] _ { + }$ . A context memory rebuilt from the part t Dw $D _ { w }$ of the context inside its window, with $\boldsymbol { D } = \left( D _ { w } , D _ { \mathrm { e v } } \right)$ , has stor $_ A \geq I ( Q ; D _ { \mathrm { e v } } \mid D _ { w } )$ , the information evicted (zero when the context fits); equality holds when the memory keeps its window, $I ( \bar { Q } ; Z _ { A } ) \bar { = } I ( \bar { Q } ; D _ { w } )$ as a transformer’s cache does, since its layer-1 keys and values are generically injective in the tokens. A fine-tune with $P _ { \mathrm { t r a i n } }$ trainable parameters stored at � bits each has $I ( D ; \theta _ { T } ) \le b P _ { \mathrm { t r a i n } } .$

Fine-tuning opens no space that grows with the data. A fine-tune writes into $b P _ { \mathrm { t r a i n } }$ bits fixed before the data arrive. A full fine-tune, whose trainable parameters are all of the model’s, writes into the bits that already hold the pretrained model: with � independent of the pretraining data $D _ { 0 }$ $\begin{array} { r } { I ( \theta _ { T } ; D ) + I ( \theta _ { T } ; D _ { 0 } ) ^ { \circ } \leq I ( \theta _ { T } ; D , D _ { 0 } ) ^ { \circ } \leq H ( \theta _ { T } ) ^ { \circ } \leq b P _ { \mathrm { t r a i n } } . } \end{array}$ so the new data and the old share one budget. A context memory is built afresh from each context and leaves the weights untouched. Because both channels face the same $G ^ { \star }$ , Equation (4) compares them exactly:

$$
\operatorname { \mathbb { E } } [ \Gamma _ { A } ] - \operatorname { \mathbb { E } } [ \Gamma _ { B } ] = ( \operatorname { s t o r } _ { B } - \operatorname { s t o r } _ { A } ) + ( \operatorname* { d e c } _ { B } - \operatorname* { d e c } _ { A } ) .\tag{8}
$$

## 5.1. Ceilings and certificates

Let the context hold � examples which, like the query, are drawn independently from one task. The context memory reads the $W$ of them inside its window. Write $\mathrm { K L } _ { 0 } =$ $\mathrm { K L } ( P _ { O } ^ { * } \| p _ { 0 } )$ for the prior repair the two channels share, and let $C _ { B }$ be any bound on the bits the fine-tune stores, $I ( D ; Z _ { B } ) \le C _ { B } .$ , such as $b P _ { \mathrm { t r a i n } }$ or the noisy-step bound of Proposition 14 in Appendix A.6.

Proposition 6 (Choosing the channel). Under Assumption 1, with the context memory rebuiltfrom the window $D _ { w } .$

(i) $\mathbb { E } [ \Gamma _ { A } ] \le I ( Q ; D _ { w } ) + \mathrm { K L } _ { 0 }$ and E[Γ<sub>�</sub>] ≤ min{�(�; �), �<sub>�</sub>}+ $\mathrm { K L } _ { 0 } .$

(ii) Learning in context wins whenever $\mathbb { E } [ \Gamma _ { A } ]$ exceeds min $\{ I ( Q ; D ) , C _ { B } \} + \mathrm { K L } _ { 0 } ,$ , and fine-tuning wins whenever $\mathbb { E } [ \Gamma _ { B } ]$ exceeds $I ( Q ; D _ { w } ) + \mathrm { K L } _ { 0 }$

(iii) If the context fits the window and the context memory keeps it, as a transformer’s cache does (Proposition $\dot { 5 ) } _ { ; }$ then $\mathbb { E } [ \Gamma _ { B } ] - \mathbb { E } [ \Gamma _ { A } ] \leq { \mathrm { d e c } } _ { A } .$

(iv) For $N \geq W , \mathbb { E } [ \Gamma _ { A } ]$ does not depend on $N ,$ so afine-tune whose capacity bound satisfies $C _ { B } + { \mathrm { K L } } _ { 0 } < \mathbb { E } [ \Gamma _ { A } ]$ at every � loses at every $N \geq W$

Each certificate in (ii) needs one channel’s measured gain and the other’s ceiling, so when it fires it settles the choice without running the other channel. When the context fits and the memory keeps it, part (iii) caps what fine-tuning can add at ${ \mathrm { d e c } } _ { A } .$ , the decoding loss in context. Turned around, any fine-tune’s advantage over learning in context is a lower bound on de $; A \cdot$ . That makes it a test of whether the model uses what its memory holds, and the bound holds context by context (Appendix $\mathsf { B } . 7 )$ . Part (iv) turns the capacity ceiling into a statement about data: a fine-tune whose capacity bound, plus the prior repair, falls short of the gain in context never catches up, however much data it sees.

Each tool covers its own regime. The fine-tune certificate needs a context that overflows the window: when the context fits, the in-context ceiling $I ( Q ; D _ { w } ) + \mathrm { K L } _ { 0 }$ equals $G ^ { \star }$ , which no channel can exceed, and part (iii) takes over. The certificate for learning in context can fire only when $C _ { B } < I ( Q ; D _ { w } )$ , and it takes any valid storage bound: an adapter’s �� bits (Section 6.1), or, for a fine-tune with noisy steps, the bound of Proposition 14. The transfer bound of Proposition 17 replaces the capacity ceiling altogether. The certificates need the information terms and the prior repair, which are exact for designed distributions of contexts (Section 6.1); part (iii), the learning curves, the storage budget and the anytime-valid test need neither, so Section 6.2 runs them on GPT-2, tables and airfoil pressure, where those terms are unknown.

## 5.2. Choosing without the true distribution

Let the context consist of units $d _ { 1 } , \ldots , d _ { n }$ , such as examples, observations, records or question–answer pairs, write $D _ { m } =$ $( d _ { 1 } , \ldots , d _ { m } )$ , and let the query � be a further unit. The learning curve of channel � is $L _ { c } ( m ) = \mathbb { E } \ell _ { c } ( Q \mid D _ { m } )$ , and fine-tuning on � units beats a context memory holding � units exactly when $L _ { B } ( n ) < L _ { A } ( N )$

Proposition 7 (Prequential learning curves). Let $d _ { 1 } , \ldots , d _ { n }$ and � be exchangeable, and let each channel build its memory after � units from those units and from randomness independent of the data: the context memory by reading them through its window, and the fine-tune by one online update per unit, $\theta _ { m } = U ( \theta _ { m - 1 } , \dot { d _ { m } } , \xi _ { m } )$ . Then for every $j \leq n ,$

$$
\mathbb { E } \ell _ { c } ( d _ { j } \mid d _ { < j } ) = L _ { c } ( j - 1 ) , \quad \mathbb { E } \sum _ { j = 1 } ^ { n } \ell _ { c } ( d _ { j } \mid d _ { < j } ) = \sum _ { m = 0 } ^ { n - 1 } L _ { c } ( m ) .\tag{9}
$$

In context, the �-th term rebuilds the memory from the units before unit � and scores unit $j ;$ for a transformer whose window holds the context, one teacher-forced pass returns every term. For the online fine-tune, $\ell _ { B } ( d _ { j } \mid d _ { < j } ) =$ $\ell _ { \theta _ { j - 1 } } ( d _ { j } )$ is the first-epoch loss on unit $j .$ Each sum is a prequential code length of � (Dawid, 1984; Rissanen, 1984; Blier and Ollivier, 2018). A sum adds up the whole learning curve, and learning curves cross, so the winner at a given size comes from the terms at that size, averaged over orderings, or from held-out queries.

Theorem 8 (Anytime-valid choice). Fix the task and the context �, build the context memory from � and train any model on �, by fine-tuning or from scratch, and let $q _ { 1 } , q _ { 2 } , \dots$ . be held-out queries drawn independently from the task. Mix the context memory’s probability for each output symbol with � of the uniform distribution over the alphabet �, so that for queries of at most � symbols $\Delta _ { j } =$ $\ell _ { B } ( q _ { j } ) - \ell _ { A } ( q _ { j } \mid D ) \geq - c w i t h c = L \log _ { 2 } ( | \dot { V } | / \varepsilon )$ . Under the hypothesis � that the trained model is no worse than this context memory on the task, $\mathbb { E } [ \Delta _ { j } ] \le 0$ , the process

$$
E _ { t } = \prod _ { j \leq t } { \Bigl ( } 1 + \lambda _ { j } { \frac { \Delta _ { j } } { c } } { \Bigr ) } , \quad \lambda _ { j } \in [ 0 , 1 ) c h o s e n f r o m D a n d q _ { < j } ,\tag{10}
$$

is a nonnegative supermartingale with $E _ { 0 } ~ = ~ 1$ , and $\begin{array} { r } { \operatorname* { P r } ( \operatorname* { s u p } _ { t } E _ { t } \geq 1 / \alpha ) \leq \alpha } \end{array}$

The process is a test by betting (Ramdas et al., 2023; Waudby-Smith and Ramdas, 2024), and its guarantee is Ville’s inequality (Ville, 1939). Whenever the test stops, reaching $1 / \alpha$ rejects $H _ { 0 }$ at level � and calls learning in context the winner on this task with this context. Exchanging the channels gives the test in favour of the trained model, whether fine-tuned or trained from scratch. Continuous outputs are discretised first, so the test applies to any memory.

Storage budget. Fine-tuning replaces the context by a fixed state. Let � be any such state built from �, with randomness independent of the sampled in-context outputs: fine-tuned weights, an adapter, or a compiled or compressed replacement of the context memory. Let any decoder $p _ { S } ( \cdot \mid x )$ read the state, and let a workload of � inputs $x _ { 1 } , \ldots , x _ { K }$ have incontext outputs $\hat { Y } _ { j } \sim p _ { A } ( \cdot \mid D , x _ { j } )$ , sampled independently given �.

Theorem 9 (Storage budget).

$$
\sum _ { j = 1 } ^ { K } \mathbb { E } \operatorname { K L } \bigl ( p _ { A } ( \cdot \mid D , x _ { j } ) \bigr \| p _ { S } ( \cdot \mid x _ { j } ) \bigr ) \geq I ( \hat { Y } _ { 1 : K } ; D ) - I ( D ; S ) ,\tag{11}
$$

and $I ( D ; S ) \le H ( S )$ when � takes finitely many values.

A state that reproduces the context memory on the workload to within � bits in total stores at least $I ( \tilde { Y } _ { 1 : K } ; D ) - \varepsilon$ bits about �, so no fine-tune whose capacity bound is below $I ( \hat { Y } _ { 1 : K } ; D ) - \varepsilon$ can reproduce it that closely. The information can be estimated from the model alone, by resampling the content of � and scoring the in-context outputs under every variant (Appendix A.8).

## 6. Testing the choice

## 6.1. Certificates with exact information terms

Proposition 6 is tested where the information terms are exact. A three-layer transformer with 104,194 parameters is trained to learn linear-Gaussian regression tasks in context $( d = 8$ signal-to-noise ratio $\rho = 4 ; \mathsf { A }$ ppendix A.10). A query is a block of 16 teacher-forced examples, and the context holds � further examples of the task, $N = 4$ to 1,024. The cache reads the last $\bar { W } = 8$ or $W = 1 6$ examples; the full finetune trains all weights with a recipe tuned for each � on separate tasks; and adapters train � pseudo-labels quantised to � bits, so they store at most $C _ { B } = k b$ bits. A certificate is issued when its margin, the measured gain minus the other channel’s ceiling, has a 95% interval over 64 tasks above zero, and the measured winner is the channel whose paired gain difference has a 95% interval excluding zero, otherwise a tie (Appendices C.3 and D.5).

![](images/e1ea52631818181178973dcdcf3dc11d5113c796180a19f0c973ac86ec2f46d9.jpg)  
Figure 3. Cache against fine-tuning on a trained transformer with exact information terms: gain on a 16-example query against the number of context examples. Dotted: ceilings �(�; $D _ { w } ) + { \mathrm { K L } } _ { 0 }$ of the caches with windows 8 and 16; dashed: ceiling $I ( Q ; D ) + \mathrm { K L } _ { 0 }$ of any channel; bars: sizes at which the fine-tune’s gain passes the cache’s ceiling, which certifies fine-tuning. 95% intervals over 64 tasks.

Of the 64 distinct comparisons between a cache and a finetune, 25 are certified, and every certified call agrees with the measured winner (Table 9 in Appendix C.3). The cache’s gain stops growing once the context passes the window and stays within 0.9 bits of its ceiling, while the fine-tune’s rises from 0.9 bits at $N = 3 2$ to 10.7 bits at $N = 1 { , } 0 2 4$ , against a ceiling of 11.5 bits for any channel (Figure 3). The fine-tune beats the cache from $N \stackrel { \cdot } { = } 1 2 8$ with the window of 8 and from $N = 2 5 6$ with the window of 16, and from there on its gain passes the cache’s ceiling, which certifies fine-tuning against every cache with that window. When the context fits the window, every fine-tuning schedule lowered the gain on the tuning tasks, so the tuned recipe leaves the weights unchanged and the cache wins by 2.7 to 7.0 bits. There, Proposition 6(iii) caps what any fine-tune could add at the cache’s decoding loss, estimated at 0.27 to 0.43 bits. Even with the prior repair of 0.37 bits added, the capacity of the 2-bit and 4-bit adapters falls short of the gain of the cache with the window of 16, so by Proposition 6(iv) they lose at every $N \geq 1 6$ . They lose at every size tested, and the certificate for the cache fires against both (Figure 10b in Appendix C.3).

Across TabPFN, an occupancy network and a stereo network, with contexts drawn from designed distributions, 36 of 96 certificates are issued and all 86 checks against measured winners agree (Appendix C.6). TabPFN reading 1,024 rows is certified against every memory of at most 8 bits, and logistic regression trained on the same rows against every memory confined to the last 16.

## 6.2. Choosing without the true distribution

We run Theorem 8 in both directions at level 0.025 each, so that a call errs with probability at most 0.05, on two choices outside language, each against a model trained from scratch (Appendices C.4 and D.7). On tables, TabPFN-v2 (Hollmann et al., 2025) reads 128 labelled rows in its context, XGBoost (Chen and Guestrin, 2016) trains on the same rows, and the queries are 4,096 and 4,079 fresh rows of the OpenML phoneme and spambase datasets (Vanschoren et al., 2013). On airfoil pressure, from 1,000 AirfRANS simulations (Bonnet et al., 2022), an in-context learner predicts the 64 surface pressure values of a new design from the pressure fields of its support designs, weighted by their similarity to it, and a two-layer MLP surrogate trains on the same 128 designs. Both calibrate their predictive spread on 128 further designs, and the queries are 254 designs from unseen geometry groups.

The test picks TabPFN in all three seeds on both tables, after 151 to 1,390 queries, and the trained surrogate in all three seeds on airfoil pressure, after 53 to 66. It reaches the same call in each of 200 random reorderings of every seed’s queries (Figure 4b). The test compares code lengths, which reward calibrated probabilities. TabPFN codes a label in 0.61 bits against XGBoost’s 0.81 on phoneme, where XGBoost’s labels are slightly more accurate (Table 10), and in 0.33 against 0.56 on spambase; the surrogate codes a pressure value in 2.34 bits against the in-context learner’s 3.65. The learning curves show where the airfoil call comes from (Figure 4a): the in-context learner improves at every size tested, from 4.96 bits per value with 4 designs to 3.30 with 512, but the surrogate overtakes it by 32 to 128 designs, and the test follows the crossing.

The guarantee holds where it can be checked. On 10,000 simulated streams of score differences with a known mean, the test calls a winner within 4,096 queries in 3.25%, 2.87% and 3.31% of streams under three nulls (symmetric, skewed and of changing variance) and in none under an exact tie, while an advantage of 0.5 bits per query is called correctly in every stream, after a median of 127 queries (Figure 4c; Table 12 in Appendix C.4).

On GPT-2 (Radford et al., 2019) the same tools apply unchanged (Appendices C.5 and D.6). Over 32 comparisons between its cache and an online fine-tune, on Shakespeare passages (Karpathy, 2015) and document recall, each run on 2,000 query streams, the test calls the better memory in 61,948 of the 64,000 streams and the worse one in 4, within its error limit. Four passages fit GPT-2’s window and its cache keeps them, so by Proposition 6(iii) the fine-tune’s advantage on held-out passages, 0.35 to 0.55 bits per token, is a lower bound on GPT-2’s decoding loss: its prediction falls at least that far short of the best prediction from its prompt.

![](images/9dbc015c19512ef5f5933285304d998688c6c256352c71b6865b3fb24ede2c2e.jpg)

![](images/d9b70b4d04d8ac8d64483b9a61594517bfc72e4e29139bcb5110c0c764abf227.jpg)

![](images/055508a156dd10b567535e9568b82fcf105aeebb9f6f7391519b7beb89a13986.jpg)  
Figure 4. The choice across domains. (a) Airfoil pressure: held-out code length per pressure value against the number of designs that the in-context learner reads or the surrogate trains on, mean over three seeds (thin lines: seeds); the in-context learner codes better with few designs and the surrogate from 128 on. (b) log $E _ { t }$ of the test that calls the winner, in each seed’s query order, fixed in advance: TabPFN over XGBoost on phoneme (solid) and spambase (dashed), and the surrogate over the in-context learner on airfoil pressure with 128 designs; dots mark the calls. (c) Share of 10,000 simulated streams called by query �: correct calls for advantages of 0.5, 0.1 and 0.03 bits per query, and wrong calls under the worst of three nulls, against � = 0.05.

## 7. Related work

Minimum description length identifies learning with compression (Rissanen, 1984; Grünwald, 2007), and prequential codes score sequential forecasters (Dawid, 1984). Deep networks compress their training data well under prequential codes (Blier and Ollivier, 2018), the next-token loss that trains an in-context learner is a prequential code length (Elmoznino et al., 2025), and language models are strong compressors (Delétang et al., 2024). This account carries the view from the weights to the context of any model, and splits each code length saved into bits stored and bits used. The gain is pointwise usable information (Xu et al., 2020; Ethayarajh et al., 2022) with the family reduced to one frozen model and its memory, and Theorem 2 splits it exactly into the Shannon information the memory stores, a prior repair and a decoding loss. Data valuation scores training examples by retraining or by influence approximations (Koh and Liang, 2017; Ghorbani and Zou, 2019; Ilyas et al., 2022), and context attribution scores parts of a prompt by ablation (Fernandes et al., 2021; Cohen-Wang et al., 2024). In context, deleting a part and rebuilding the memory gives its exact leave-one-out value at one rerun per part, and Corollary 4 with a matched control splits that value into content and form.

In-context learning has been interpreted as implicit Bayesian inference (Xie et al., 2022), as a learned algorithm (Garg et al., 2022), as gradient descent or ridge regression run by attention (von Oswald et al., 2023; Akyürek et al., 2023; Mahankali et al., 2024) and as implicit fine-tuning (Dai et al., 2023), and training, conversely, as attention over training patterns (Irie et al., 2022). Information-theoretic analyses express its error through the information a context carries about a latent task (Jeon et al., 2024), and in-context learning now runs beyond language, as in prior-fitted networks for tables (Hollmann et al., 2025). Proposition 13 (Appendix A.4) writes softmax attention exactly as a weight write with query-dependent step sizes, using the context/prefix split of He et al. (2022) and the divided-difference identity of Huang et al. (2026), and the law prices that write and a fine-tune in the same bits.

Prompt-based fine-tuning often follows kernel dynamics (Jacot et al., 2018; Malladi et al., 2023), prefix tuning and low-rank adaptation write small states (Li and Liang, 2021; Hu et al., 2022), and language models store a few bits of knowledge per parameter (Allen-Zhu and Li, 2025; Morris et al., 2025); Proposition 6 turns any valid bound on what a channel stores into a certificate. Context distillation reproduces a context’s predictions in the weights (Askell et al., 2021; Snell et al., 2022), and Theorem 9 sets the floor of its divergence for a state that stores a given number of bits about the context. Lampinen et al. (2025) and Mosbach et al. (2023) study in-context learning and fine-tuning side by side; Proposition 6 settles the choice from one measured gain when a certificate fires, and Theorem 8 settles it from held-out queries with anytime-valid error control, as a comparison of forecasters by betting (Henzi and Ziegel, 2022; Choe and Ramdas, 2024). Appendix E discusses each line in full.

## 8. Conclusion

Measured in code length, what data give a pretrained model obeys one law in expectation, whether they sit in its context or in its weights: bits stored, plus prior repair, minus decoding loss. Its ceilings certified the choice between context and weights on a trained transformer, TabPFN, an occupancy network and a stereo network, and every certificate was correct; its anytime-valid test made the choice on tables, airfoil pressure and GPT-2 within its error limit. The same evaluator valued the parts of a context in occupancy, stereo and kriging models and recovered a published evidence adjudicator for two language-model verifiers. The law gives a compression account of learning from context.

## Impact Statement

This paper gives a measured basis for deciding whether new data should go into a model’s context or into its weights, and a test of whether a prediction rests on the evidence supplied with it. The evidence adjudicator flags claims that their cited evidence does not support, which bears on the reliability of retrieval-augmented and fact-checking systems. The methods measure what a context supplies and add no new capability to the models they evaluate.

## References

Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? Investigations with linear models. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/ 2211.15661.

Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.3, knowledge capacity scaling laws. In International Conference on Learning Representations, 2025.

Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, and Jared Kaplan. A general language assistant as a laboratory for alignment. arXiv:2112.00861, 2021. URL https://arxiv.org/abs/2112.00861.

Léonard Blier and Yann Ollivier. The description length of deep learning models. In Advances in Neural Information Processing Systems, volume 31, pages 2216–2226, 2018. URL https://arxiv.org/abs/1802.07044.

Florent Bonnet, Jocelyn Mazari, Paola Cinnella, and Patrick Gallinari. AirfRANS: High fidelity computational fluid dynamics dataset for approximating Reynolds-averaged Navier–Stokes solutions. In Advances in Neural Information Processing Systems 35, Datasets and Benchmarks Track, pages 23463–23478, 2022.

Jochen Bröcker. Reliability, sufficiency, and the decomposition of proper scores. Quarterly Journal of the Royal Meteorological Society, 135(643):1512–1519, 2009. doi: 10.1002/qj.456.

Angel X. Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, Jianxiong Xiao, Li Yi, and Fisher Yu. ShapeNet: An information-rich 3D model repository. arXiv:1512.03012, 2015. URL https://arxiv.org/abs/1512.03012.

Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. In Proceedings ofthe 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 785–794, 2016. doi: 10.1145/2939672.2939785.

Leon Chlon. Trace Budget benchmark pipeline. Software, 2026.

Leon Chlon, Ahmed Karim, and MarcAntonio Awada. Predictable compression failures: Order sensitivity and information budgeting for evidence-grounded binary adjudication. In Proceedings of the International Conference on Machine Learning (ICML), 2026. URL https:// arxiv.org/abs/2509.11208. arXiv:2509.11208.

Yo Joong Choe and Aaditya Ramdas. Comparing sequential forecasters. Operations Research, 72(4):1368–1387, 2024.

Benjamin Cohen-Wang, Harshay Shah, Kristian Georgiev, and Aleksander Madry. ContextCite: Attributing model generation to context. In Advances in Neural Information Processing Systems, volume 37, 2024.

Imre Csiszár. I-divergence geometry of probability distributions and minimization problems. The Annals of Probability, 3(1):146–158, 1975.

Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, and Furu Wei. Why can GPT learn incontext? Language models secretly perform gradient descent as meta-optimizers. In Findings of the Association for Computational Linguistics: ACL 2023, pages 4005–4019, 2023. URL https://aclanthology. org/2023.findings-acl.247/.

A. Philip Dawid. Present position and potential developments: Some personal views: Statistical theory: The prequential approach. Journal of the Royal Statistical Society. Series A (General), 147(2):278–292, 1984.

Grégoire Delétang, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya, Li Kevin Wenliang, Matthew Aitchison, Laurent Orseau, Marcus Hutter, and Joel Veness. Language modeling is compression. In International Conference on Learning Representations, 2024. URL https: //arxiv.org/abs/2309.10668.

Eric Elmoznino, Tom Marty, Tejas Kasetty, Leo Gagnon, Sarthak Mittal, Mahan Fathi, Dhanya Sridhar, and Guillaume Lajoie. In-context learning and Occam’s razor. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 15296–15319, 2025.

Mohamed Elrefaie, Angela Dai, and Faez Ahmed. DrivAerNet: A parametric car dataset for data-driven aerodynamic design and prediction. arXiv:2403.08055, 2024. URL https://arxiv.org/abs/2403.08055.

Kawin Ethayarajh, Yejin Choi, and Swabha Swayamdipta. Understanding dataset difficulty with V-usable information. In Proceedings of the International Conference on Machine Learning, volume 162 of Proceedings ofMachine Learning Research, pages 5988–6008, 2022. URL https://proceedings.mlr.press/ v162/ethayarajh22a.html.

Patrick Fernandes, Kayo Yin, Graham Neubig, and André F. T. Martins. Measuring and increasing context usage in context-aware machine translation. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6467–6478, 2021.

Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. What can transformers learn in-context? A case study of simple function classes. In Advances in Neural Information Processing Systems, volume 35, pages 30583–30598, 2022. URL https://arxiv.org/ abs/2208.01066.

Amirata Ghorbani and James Zou. Data Shapley: Equitable valuation of data for machine learning. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, 2019.

Peter D. Grünwald. The Minimum Description Length Principle. MIT Press, 2007.

Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, and Graham Neubig. Towards a unified view of parameter-efficient transfer learning. In International Conference on Learning Representations, 2022. URL https://arxiv.org/abs/2110.04366.

Alexander Henzi and Johanna F. Ziegel. Valid sequential inference on probability forecast performance. Biometrika, 109(3):647–663, 2022.

Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 637(8045):319–326, 2025. doi: 10.1038/ s41586-024-08328-6.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://arxiv.org/abs/ 2106.09685.

Julie Huang, Maggie Chlon, Gregory Gutin, and Leon Chlon. Exact finite attention responses from RoPE derivatives. arXiv:2609.14127, 2026. URL https: //arxiv.org/abs/2609.14127.

Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, and Aleksander Madry. Datamodels: Understanding predictions with data and data with predictions. In Proceedings ofthe 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, 2022.

Kazuki Irie, Róbert Csordás, and Jürgen Schmidhuber. The dual form of neural networks revisited: Connecting test time predictions to training patterns via spotlights of attention. In Proceedings of the International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 9639–9659, 2022. URL https://proceedings.mlr.press/ v162/irie22a.html.

Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems, volume 31, pages 8571–8580, 2018. URL https://arxiv.org/abs/1806.07572.

Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and Henrik Aanæs. Large scale multiview stereopsis evaluation. In IEEE Conference on Computer Vision and Pattern Recognition, 2014. URL https://roboimagedata.compute.dtu. dk/?page\_id=36.

Hong Jun Jeon, Jason D. Lee, Qi Lei, and Benjamin Van Roy. An information-theoretic analysis of in-context learning. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 21522–21554, 2024.

Andrej Karpathy. char-rnn: Multi-layer recurrent neural networks for character-level language models. https:// github.com/karpathy/char-rnn, 2015. Shakespeare text in data/tinyshakespeare.

Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, 2017.

Andrew K. Lampinen, Arslan Chaudhry, Stephanie C. Y. Chan, Cody Wild, Diane Wan, Alex Ku, Jörg Bornschein, Razvan Pascanu, Murray Shanahan, and James L. McClelland. On the generalization of language models from in-context learning and finetuning: A controlled study. arXiv:2505.00661, 2025. URL https: //arxiv.org/abs/2505.00661.

Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings

of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4582–4597, 2021. URL https: //aclanthology.org/2021.acl-long.353/.

Arvind V. Mahankali, Tatsunori Hashimoto, and Tengyu Ma. One step of gradient descent is provably the optimal in-context learner with one layer of linear selfattention. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/ 2307.03576.

Sadhika Malladi, Alexander Wettig, Dingli Yu, Danqi Chen, and Sanjeev Arora. A kernel-based view of language model fine-tuning. In Proceedings of the International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 23610– 23641, 2023. URL https://proceedings.mlr. press/v202/malladi23a.html.

Vladimir A. Marchenko and Leonid A. Pastur. Distribution of eigenvalues for some sets of random matrices. Mathematics ofthe USSR-Sbornik, 1(4):457–483, 1967.

Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes incontext learning work? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 2022. URL https://arxiv.org/abs/ 2202.12837.

John X. Morris, Chawin Sitawarin, Chuan Guo, Narine Kokhlikyan, G. Edward Suh, Alexander M. Rush, Kamalika Chaudhuri, and Saeed Mahloujifar. How much do language models memorize? arXiv:2505.24832, 2025.

Marius Mosbach, Tiago Pimentel, Shauli Ravfogel, Dietrich Klakow, and Yanai Elazar. Few-shot fine-tuning vs. in-context learning: A fair comparison and evaluation. In Findings of the Association for Computational Linguistics: ACL 2023, pages 12284–12314, 2023. URL https://aclanthology.org/2023. findings-acl.779/.

ONNX Model Zoo contributors. ONNX Model Zoo: GPT-2. https://github.com/onnx/models, 2020. GPT-2-LM-HEAD, opset 10.

Songyou Peng, Michael Niemeyer, Lars Mescheder, Marc Pollefeys, and Andreas Geiger. Convolutional occupancy networks. In European Conference on Computer Vision, 2020. URL https://arxiv.org/abs/2003. 04618.

Ankit Pensia, Varun Jog, and Po-Ling Loh. Generalization error bounds for noisy, iterative algorithms. In IEEE International Symposium on Information Theory (ISIT), 2018. URL https://arxiv.org/abs/1801.04295.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. Technical report, OpenAI, 2019.

Aaditya Ramdas, Peter Grünwald, Vladimir Vovk, and Glenn Shafer. Game-theoretic statistics and safe anytimevalid inference. Statistical Science, 38(4):576–601, 2023.

Jorma Rissanen. Universal coding, information, prediction, and estimation. IEEE Transactions on Information Theory, 30(4):629–636, 1984.

Tal Schuster, Adam Fisch, and Regina Barzilay. Get your vitamin C! robust fact verification with contrastive evidence. In Proceedings ofthe 2021 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 624–643, 2021. URL https://aclanthology.org/2021. naacl-main.52/.

Charlie Snell, Dan Klein, and Ruiqi Zhong. Learning by distilling context. arXiv:2209.15189, 2022. URL https://arxiv.org/abs/2209.15189.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568: 127063, 2024.

Joaquin Vanschoren, Jan N. van Rijn, Bernd Bischl, and Luis Torgo. OpenML: Networked science in machine learning. SIGKDD Explorations, 15(2):49–60, 2013.

Jean Ville. Étude critique de la notion de collectif. Gauthier-Villars, Paris, 1939.

Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn incontext by gradient descent. In Proceedings ofthe International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 35151–35174, 2023. URL https://arxiv.org/ abs/2212.07677.

Ian Waudby-Smith and Aaditya Ramdas. Estimating means of bounded random variables by betting. Journal of the Royal Statistical Society Series B: Statistical Methodology, 86(1):1–27, 2024.

Steven V. Weijs, Ronald van Nooijen, and Nick van de Giesen. Kullback–Leibler divergence as a forecast skill score with classic reliability–resolution–uncertainty decomposition. Monthly Weather Review, 138(9):3387– 3399, 2010. doi: 10.1175/2010MWR3229.1.

Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit Bayesian inference. In International Conference on Learning Representations, 2022. URL https:// arxiv.org/abs/2111.02080.

Yilun Xu, Shengjia Zhao, Jiaming Song, Russell Stewart, and Stefano Ermon. A theory of usable information under computational constraints. In International Conference on Learning Representations, 2020. URL https:// arxiv.org/abs/2002.10689.

Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. MVSNet: Depth inference for unstructured multi-view stereo. In European Conference on Computer Vision, pages 785–801, 2018. URL https: //openaccess.thecvf.com/content\_ ECCV\_2018/html/Yao\_Yao\_MVSNet\_Depth\_ Inference\_ECCV\_2018\_paper.html.

## A. Further results

This appendix gives the results that the main text uses or points to: the deleted-evidence lemma, the token-level form of the gain, the two writes, capacity under noisy steps, the lift budget, the estimator of the storage budget, the transfer integral, and a model in which every term is closed form. Proofs are in Appendix B.

## A.1. Deleted-evidence exclusion

Lemma 10 (Deleted-evidence exclusion). Under Assumption 1, two contexts that share the remaining part � induce the same lawfor the rebuilt memory $Z _ { R } ,$ , whatever was deleted. In particular � ⊥⊥ $Z _ { R } \mid R ,$ , provided the exogenous randomness is conditionally independent of� given �.

The rebuilt memory carries no trace of the deleted evidence, so a deletion is valued at what the evidence adds beyond the rest of the context, the query and the weights, just as held-out validation values a training example at what the rest of the training set does not already supply. The proof applies the fixed transition kernels step by step (Appendix B.1).

## A.2. Where the gain comes from, token by token

Theorem 11 (Lift identity). At a query position, let � be the base next-token distribution and $p ^ { \prime } = p e ^ { \Delta z } / Z$ the channel’s, where Δ� is the logit change the channel causes. The teacher-forced token � gains, in nats,

$$
\log { \frac { p ^ { \prime } ( q ) } { p ( q ) } } = \underbrace { \Delta z _ { q } - \mathbb { E } _ { p } [ \Delta z ] } _ { \lambda ( c e n t r e d l i f t ) } - \mathrm { K L } ( p \parallel p ^ { \prime } ) .\tag{12}
$$

Proof. log $\mathcal { Z } = \mathbb { E } _ { p } \Delta z + \mathrm { K L } ( p \| p ^ { \prime } )$

Hoeffding’s lemma gives $0 \leq \mathrm { K L } ( p \| p ^ { \prime } ) \leq \mathrm { r a n g e } ( \Delta z ) ^ { 2 } / 8 .$ , so the per-token gain $\Gamma _ { t }$ obeys

$$
\begin{array} { r } { \lambda _ { t } - \frac { 1 } { 8 } \mathrm { r a n g e } ( \Delta z _ { t } ) ^ { 2 } \ \leq \ \Gamma _ { t } \ \leq \ \lambda _ { t } . } \end{array}\tag{13}
$$

Over the query, $\textstyle \sum _ { t } \lambda _ { t } > 0$ is necessary for $Q$ to become more compressible, and $\begin{array} { r } { \sum _ { t } \lambda _ { t } > \sum _ { i } } \end{array}$ range(Δ�<sub>�</sub> )<sup>2</sup>/8 is sufficient. In expectation over the true next token,

$$
\begin{array} { r } { \mathbb { E } [ \Gamma _ { t } ] = \langle P _ { t } ^ { * } - p _ { t } , \Delta z _ { t } \rangle - \mathrm { K L } ( p _ { t } \| p _ { t } ^ { \prime } ) = \mathrm { K L } ( P _ { t } ^ { * } \| p _ { t } ) - \mathrm { K L } ( P _ { t } ^ { * } \| p _ { t } ^ { \prime } ) . } \end{array}\tag{14}
$$

The cost $\mathbf { K L } ( p \| p ^ { \prime } )$ grows with the size of the move, which is how a context can move predictions far more than a gradient step does and still gain little. To second order, with $h ^ { * } = \log ( P ^ { * } / p )$ and the Fisher inner product $\langle a , b \rangle _ { p } = \mathbf { C o v } _ { p } ( a , b )$

$$
\begin{array} { r } { \mathbb { E } [ \Gamma _ { t } ] \approx \langle h ^ { * } , \Delta z \rangle _ { p } - \frac { 1 } { 2 } \| \Delta z \| _ { p } ^ { 2 } . } \end{array}\tag{15}
$$

To this order, the gain is positive when $\cos _ { p } ( h ^ { \ast } , \Delta z ) > \| \Delta z \| _ { p } / ( 2 \| h ^ { \ast } \| _ { p } )$ , and a channel that moves logits along a direction ℎ captures at most $\cos _ { p } ^ { 2 } ( h , h ^ { * } )$ of the available KL $( P ^ { * } \| p )$

## A.3. The fine-tune

Proposition 12 (Fine-tuning gain). Under gradientflow on ℓ(�) at rate �, in nats,

$$
\begin{array} { c } { \displaystyle \Gamma _ { B } ( Q ) = \eta \int _ { 0 } ^ { T } K _ { \nu } ( Q , D ) d \nu , } \\ { \displaystyle K _ { \nu } = \left. \nabla \log p _ { \theta _ { \nu } } ( Q ) , \nabla \log p _ { \theta _ { \nu } } ( D ) \right. = \sum _ { t \in Q } \sum _ { s \in D } { r _ { t } ^ { \top } J _ { t } J _ { s } ^ { \top } r _ { s } } , } \end{array}\tag{16}
$$

where � is training time, � and � index tokens of� and $D , r = e _ { t o k e n } - p$ is the logit residual and $J = \partial z / \partial \theta ;$ for Adam, its preconditioner sits between the two gradients. For one discrete step $\theta _ { k + 1 } = \theta _ { k } + \eta g _ { D }$ , Taylor’s theorem with integral remainder is exact:

$$
\begin{array} { r l } & { \Delta _ { k } \log p ( Q ) = \eta \langle g _ { Q } , g _ { D } \rangle } \\ & { \qquad + \eta ^ { 2 } \displaystyle \int _ { 0 } ^ { 1 } ( 1 - a ) g _ { D } ^ { \top } \nabla ^ { 2 } \log p _ { \theta _ { k } + a \eta g _ { D } } ( Q ) g _ { D } d a . } \end{array}\tag{17}
$$

The kernel $K _ { \nu }$ is the empirical neural tangent kernel $J _ { t } J _ { s } ^ { \top }$ (Jacot et al., 2018), contracted with the residuals of � and �. Let $\beta$ bound the absolute curvature of log $p ( Q )$ in the direction $g _ { D }$ along the step. Then � becomes strictly more compressible if $\langle g _ { Q } , g _ { D } \rangle > \frac { \beta \eta } { 2 } \| g _ { D } \| ^ { 2 }$ , and it cannot improve if $\begin{array} { r } { \langle g _ { Q } , g _ { D } \rangle < - \frac { \beta \eta } { 2 } \| g _ { D } \| ^ { 2 } } \end{array}$ . Over $T$ steps, $\begin{array} { r } { \Gamma _ { B } = \eta \sum _ { k } K _ { k } \pm } \end{array}$ $\begin{array} { r l } { \frac { \beta \eta ^ { 2 } } { 2 } \sum _ { k } \| g _ { D } ( \theta _ { k } ) \| ^ { 2 } } & { { } } \end{array}$ . The first-order logit change is $\begin{array} { r } { \Delta z _ { t } \approx \eta \sum _ { s } J _ { t } J _ { s } ^ { \top } r _ { s } } \end{array}$ , and placing it in Equation (15) gives a one-step efficiency $\mathbb { E } [ \Gamma _ { t } ] / \mathrm { K L } ( P _ { t } ^ { * } \| p _ { t } )$ of at most $\cos _ { p } ^ { 2 } ( \Delta z _ { t } , h ^ { * } )$ , reached at the best step size.

## A.4. The cache

Proposition 13 (The cache is a query-dependent weight write). Take any head with query vector � and keys $k _ { i } ,$ , after rotary embedding (Su et al., 2024). Let � be the context and � the query’s own prefix. Exactly,

$$
\begin{array} { c } { { \displaystyle o ( u ) = \left( 1 - \alpha ( u ) \right) o _ { S } ( u ) + \alpha ( u ) o _ { D } ( u ) , } } \\ { { \displaystyle \alpha ( u ) = \frac { \mathcal { Z } _ { D } ( u ) } { \mathcal { Z } _ { D } ( u ) + \mathcal { Z } _ { S } ( u ) } , \qquad \mathcal { Z } _ { X } ( u ) = \displaystyle \sum _ { i \in X } e ^ { \langle u , k _ { i } \rangle } . } } \end{array}\tag{18}
$$

With $\varphi ( x ) = ( e ^ { x } - 1 ) / x$ and an anchor query �<sub>0</sub>, the context numerator is exactly

$$
\begin{array} { r l r } {  { \sum _ { i \in D } e ^ { \langle u , k _ { i } \rangle } \nu _ { i } = N _ { D } ( u _ { 0 } ) } } \\ & { } & { + [ \sum _ { i \in D } e ^ { \langle u _ { 0 } , k _ { i } \rangle } \varphi \big ( \langle u - u _ { 0 } , k _ { i } \rangle \big ) \nu _ { i } k _ { i } ^ { \top } ] ( u - u _ { 0 } ) , } \end{array}\tag{19}
$$

and the same holds for $z _ { D }$ with $\nu _ { i }$ replaced by 1.

Equation (18) is the prefix decomposition of He et al. (2022), with the context in the role of the prefix, and Equation (19) is the divided-difference identity of Huang et al. (2026). The bracket is a weight matrix of rank at most min $( | D | , d _ { h } )$ , and its per-example coefficients are recomputed for every query. Per head, the cache is a weight update whose step sizes depend on the query. A static weight edit, such as a fine-tune or a compiled replacement of the cache, writes the frozen-coefficient version, and the gap between the two is the transfer residual. Linear attention replaces the exponential by the identity, so its context numerator is exactly $\begin{array} { r } { \left( \sum _ { i } \nu _ { i } k _ { i } ^ { \top } \right. } \end{array}$ �: the write is $\textstyle \sum _ { i } \nu _ { i } k _ { i } ^ { \intercal }$ , one unit step of gradient descent on $\begin{array} { r } { \frac { 1 } { 2 } \sum _ { i } \| \nu _ { i } - W k _ { i } \| ^ { 2 } } \end{array}$ from $W = 0$ (von Oswald et al., 2023). A logit perturbation � changes every log-probability by at most range(�), so any static replacement of the cache costs at most range(�<sub>�</sub>) nats per token, where $\delta _ { t }$ is the logit discrepancy it induces.

## A.5. Attention without a cache

A kriging learner predicts the value at a query site as a weighted sum of support values. The weights come from the agreement between the query and each support point, whitened by the agreement among the support points themselves through the Gram system, and constrained to sum to one. That is attention, with support descriptors as keys, support values as values and the Gram system in place of the softmax. A transformer stores its keys and values because its context is long and reused; a kriging learner recomputes its weights for each query because its support set is small. Deleting a support item and refitting is the same intervention as rebuilding a cache. The learner of Appendix C.2.5 also selects its own settings by leave-one-out prediction on its support set, so it runs held-out validation inside its own forward pass.

## A.6. Capacity under noisy steps

Proposition 14 (Noisy steps). For noisy gradient steps with noise variance $\sigma _ { k } ^ { 2 }$ and gradient norm at most � (Pensia et al., 2018), afine-tune with $P _ { \mathrm { t r a i n } }$ trainable parameters satisfies

$$
I ( D ; \theta _ { T } ) \leq \sum _ { k } \frac { P _ { \mathrm { t r a i n } } } { 2 } \log \Big ( 1 + \frac { \eta _ { k } ^ { 2 } L ^ { 2 } } { P _ { \mathrm { t r a i n } } \sigma _ { k } ^ { 2 } } \Big ) .
$$

The evicted information is the quantity any compression of the context memory has to bound. For a context that fits the window of a memory that keeps it, as a transformer’s cache does, deterministic, high-precision fine-tuning is generically injective in � as well, so storage separates the two channels only for noisy, low-precision or low-dimensional updates; otherwise the difference between them is decoding loss. By Equation (8), with $C _ { B }$ the fine-tune’s information capacity, learning in context wins whenever $\left[ I ( Q ; D ) - C _ { B } \right] _ { + } > \mathrm { s t o r } _ { A } + \mathrm { d e c } _ { A } .$ , and, for a context memory that keeps its window, fine-tuning wins if and only if $I ( Q ; D _ { \mathrm { e v } } \mid D _ { w } ) > \mathrm { s t o r } _ { B } + \mathrm { d e c } _ { B } - \mathrm { d e c } _ { A }$

## A.7. The lift budget

Theorem 15 (Lift budget). For any event $E ,$ such as a correct answer, let $\hat { Y } _ { c } \sim p _ { c } ( \cdot \mid Z _ { c } )$ be the channel’s output, $\pi _ { c } ( \cdot \mid D ) = \mathbb { E } [ p _ { c } ( \cdot \mid Z _ { c } ) \mid D ]$ its law given $D ,$ and $\bar { \pi } _ { c } = \mathbb { E } _ { D } \pi _ { c } ( \cdot \mid D )$ . Then

$$
\begin{array} { r l } & { \mathbb { E } _ { D } d _ { 2 } \big ( \pi _ { c } ( E \mid D ) \big \| p _ { 0 } ( E ) \big ) \leq I ( \hat { Y } _ { c } ; D ) + \mathrm { K L } ( \bar { \pi } _ { c } \| p _ { 0 } ) } \\ & { \qquad \leq C _ { c } + \mathrm { K L } ( \bar { \pi } _ { c } \| p _ { 0 } ) , } \end{array}\tag{20}
$$

where $\begin{array} { r } { d _ { 2 } ( a \| b ) = a \log _ { 2 } \frac { a } { b } + ( 1 - a ) \log _ { 2 } \frac { 1 - a } { 1 - b } } \end{array}$ and $C _ { c }$ is the channel’s capacity.

The probability lift a channel can buy is capped by the bits it writes, plus a drift term that does not depend on �. A memory computed deterministically from $D ,$ , such as a feature grid or a key–value cache, has $\pi _ { c } ( \cdot \mid D ) = p _ { c } ( \cdot \mid D )$ . For a single context the budget is sharp. Write $b _ { t } ( q ) = d _ { 2 } ( t \| q )$ for $q < t$ and $b _ { t } ( q ) = 0$ otherwise.

Corollary 16 (Sharp lift budget). Fix a claim A with $p = p _ { c } ( { \mathcal { A } } \mid D )$ and $q = p _ { c } ( \mathcal { A } \mid R )$ . Then $I _ { G } \geq d _ { 2 } ( p \Vert q )$ , with gap $\begin{array} { r } { p \mathrm { \normalfont ~ K L } \big ( p _ { c } ( \cdot \mid D , \mathcal { A } ) \| p _ { c } ( \cdot \mid R , \mathcal { A } ) \big ) + ( 1 - p ) \mathrm { \normalfont ~ K L } \big ( p _ { c } ( \cdot \mid D , \mathcal { A } ^ { c } ) \| p _ { c } ( \cdot \mid R , \mathcal { A } ^ { c } ) \big ) . \mathrm { \normalfont ~ } F o r 0 < q < t < 1 } \end{array}$

$$
\operatorname* { m i n } _ { P : P ( \mathcal { A } ) \geq t } \mathrm { K L } \bigl ( P \parallel p _ { c } ( \cdot \mid R ) \bigr ) = b _ { t } ( q ) ,\tag{21}
$$

attained by the law that keeps $p _ { c } ( \cdot \mid R )$ inside A and $\mathcal { A } ^ { c }$ and rescales their masses to � and $1 - t .$ Consequently $p \geq t > q$ implies $I _ { G } \geq b _ { t } ( q )$ , and $I _ { G } < b _ { t } ( q )$ implies $p < t .$

The minimiser is the information projection onto $\{ P : P ( { \mathcal { A } } ) \geq t \}$ (Csiszár, 1975), and the budget is the Expectation-level Decompression Law of Chlon et al. (2026) applied to one context and its ablation.

## A.8. Estimating the storage budget

The information on the right needs only the model. Resample the content of � into � variants ${ D ^ { ( 1 ) } , \dots , D ^ { ( M ) } }$ , sample in-context outputs under each variant and score them under every variant; for � drawn uniformly from the variants, and with $\begin{array} { r } { p _ { A } ( \hat { Y } _ { 1 : K } \mid D ) = \prod _ { j } p _ { A } ( \hat { Y } _ { j } \mid D , x _ { j } ) } \end{array}$ ,

$$
I ( \hat { Y } _ { 1 : K } ; D ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mathbb { E } _ { \hat { Y } \sim p _ { A } ( \cdot \vert D ^ { ( m ) } ) } \log _ { 2 } \frac { p _ { A } ( \hat { Y } \mid D ^ { ( m ) } ) } { \frac { 1 } { M } \sum _ { m ^ { \prime } = 1 } ^ { M } p _ { A } ( \hat { Y } \mid D ^ { ( m ^ { \prime } ) } ) } .\tag{22}
$$

With the variants drawn from the data distribution, the expectation of the right side over the draws is a lower bound on $I ( \hat { Y } _ { 1 : K } ; D )$ that saturates at log � bits, so the estimate certifies the budget for states that store fewer bits than that. This is the information budget of Chlon et al. (2026), computed over content-resampled datasets as that budget is computed over skeletons, with the bits the state stores about � in the part of the bits-to-trust, and keeping $D$ in context in the part of abstention.

## A.9. The transfer integral

Write $\Gamma _ { B } ^ { \mathrm { t r } } ( t ) = \ell _ { \theta _ { 0 } } ( D ) - \ell _ { \theta _ { t } } ( D )$ for the code length that fine-tuning has removed from � itself.

Proposition 17 (Transfer integral). Under gradient flow on $\ell ( D )$ with any positive semidefinite preconditioner $P _ { t }$ , while $g _ { D } ^ { \top } \bar { P } _ { t } g _ { D } > 0 ,$

$$
\frac { d \Gamma _ { B } ( Q ) } { d \Gamma _ { B } ^ { \mathrm { t r } } } = r _ { t } ( Q ) = \frac { g _ { Q } ^ { \top } P _ { t } g _ { D } } { g _ { D } ^ { \top } P _ { t } g _ { D } } , \quad \quad \Gamma _ { B } ( Q ) = \int _ { 0 } ^ { \Gamma _ { B } ^ { \mathrm { t r } } ( T ) } r d \Gamma _ { B } ^ { \mathrm { t r } } ,\tag{23}
$$

so that $\begin{array} { r } { | \Gamma _ { B } ( Q ) | \leq \operatorname* { s u p } _ { t } | r _ { t } ( Q ) | \Gamma _ { B } ^ { \mathrm { t r } } ( T ) } \end{array}$ , which for discrete � is at most sup $| r _ { t } ( Q ) | \ell _ { \theta _ { 0 } } ( D )$

The ratio $r _ { t }$ has no units, so the integral holds in bits. Under gradient flow, every bit removed from $D ' s$ code length transfers $r _ { t }$ bits to $Q \mathrm { { : } }$ : without compressing $D .$ , fine-tuning transfers nothing to $Q$ , and along a run the integral splits the fine-tune’s gain into the compression of � and its transfer to the query. When the ratio stays within a band $| r _ { t } ( Q ) | \leq \rho$ along the run, the fine-tune gains at most $\rho \ell _ { \theta _ { 0 } } ( D )$ bits on $Q ,$ , so a fine-tune with $\rho \ell _ { \theta _ { 0 } } ( D ) < \mathbb { E } [ \Gamma _ { A } ]$ cannot deliver what the context memory delivers, whatever its capacity, and a ratio that stays negative means interference. The ratio at initialisation, $r _ { 0 } ( Q )$ costs one backward pass per query and one for $D$ and fixes where the band starts; orthogonal gradients, as between a fact and its reversal, give $r _ { 0 } = 0$

![](images/450efa23f8f3b1a2b8b3189e1e3e655292e892a23d4874ddc348baf8610f955f.jpg)  
Figure 5. Gain on a held-out query in the linear-Gaussian model at $\rho = 4 .$ . Lines are the closed forms of Proposition 18; dots are simulations at $d = 2 0 0$ , 300 repeats each. One attention layer and one gradient step coincide, and five gradient steps nearly reach the ideal cache. The grey dashed line is the ceiling of every channel and the grey dotted line is the ceiling of a rank-�/4 adapter.

## A.10. A model where every term is exact

Let $\begin{array} { r } { w \sim N ( 0 , \frac { { \bar { \tau } } ^ { 2 } } { d } I ) , x \sim N ( 0 , I _ { d } ) } \end{array}$ and $y = w ^ { \top } x + \varepsilon$ with $\varepsilon \sim N ( 0 , \sigma ^ { 2 } )$ . A dataset holds � examples; write $\gamma = d / n$ and $\rho = \bar { \tau } ^ { 2 } / \sigma ^ { 2 }$ , and let the base predictive be the prior predictive of � given �. A channel whose calibrated Gaussian predictive has excess risk $\mathcal { E } _ { c }$ gains, in the proportional limi $n , d \to \infty$ with � fixed,

$$
\mathbb { E } [ \Gamma _ { c } ] = \frac { 1 } { 2 } \log _ { 2 } { \frac { \sigma ^ { 2 } + \bar { \tau } ^ { 2 } } { \sigma ^ { 2 } + \mathcal { E } _ { c } } } .\tag{24}
$$

Proposition 18 (Closed forms). (i) The ideal cache, the Bayes posterior predictive, gains $I ( y _ { q } ; D \mid x _ { q } )$ , which in the proportional limit is Equation (24) with

$$
\begin{array} { c } { { \displaystyle \mathcal { E } _ { A } = \sigma ^ { 2 } \gamma s _ { \gamma } ( \kappa ) , \qquad \kappa = \gamma / \rho , } } \\ { { \displaystyle s _ { \gamma } ( \kappa ) = \frac { - ( \kappa + 1 - \gamma ) + \sqrt { ( \kappa + 1 - \gamma ) ^ { 2 } + 4 \gamma \kappa } } { 2 \gamma \kappa } . } } \end{array}\tag{25}
$$

(ii) One linear-attention layer, with its step size learned in pretraining, has, exactly at finite � and � with $\tau ^ { 2 } = \bar { \tau } ^ { 2 } / d ,$

$$
\mathcal { E } _ { L A } = \tau ^ { 2 } d \frac { \tau ^ { 2 } ( d + 1 ) + \sigma ^ { 2 } } { \tau ^ { 2 } ( n + d + 1 ) + \sigma ^ { 2 } } .\tag{26}
$$

(iii) Afine-tune of� gradient-descent steps ofsize � from zero has, in the proportional limit,

$$
\begin{array} { l } { \displaystyle \mathcal { E } _ { B } ( T , \eta ) = \bar { \tau } ^ { 2 } \sum _ { k = 0 } ^ { 2 T } \binom { 2 T } { k } ( - \eta ) ^ { k } M _ { k } } \\ { \displaystyle \qquad + \sigma ^ { 2 } \gamma \sum _ { k , l = 1 } ^ { T } \binom { T } { k } \binom { T } { l } ( - 1 ) ^ { k + l } \eta ^ { k + l } M _ { k + l - 1 } , } \end{array}\tag{27}
$$

with $M _ { 0 } = 1$ and $\begin{array} { r } { M _ { k } = \sum _ { j = 0 } ^ { k - 1 } \frac { 1 } { j + 1 } \binom { k } { j } \binom { k - 1 } { j } \gamma ^ { j } } \end{array}$ , the moments of the Marchenko–Pastur law (Marchenko and Pastur, 1967).

Equations (24)–(27) are the learning curves of Section 5.2 in closed form, and they give four results (Figure 5).

• One step equals one layer. One gradient step at its best size has excess risk exactly $\mathcal { E } _ { L A }$ at every � and �, and in the proportional limit the best size is $\overline { { { \eta } } } ^ { * } = \bar { \tau } ^ { 2 } / ( \bar { \tau } ^ { 2 } ( 1 + \gamma ) + \sigma ^ { 2 } \gamma )$ , so $\mathcal { E } _ { B } ( 1 , \eta ^ { * } ) = \mathcal { E } _ { L A }$ . This turns the optimality of one gradient step for one layer of linear attention (Mahankali et al., 2024) into an identity of excess risks, and so of bits.

• Ordering. $\mathcal { E } _ { A } \leq \mathcal { E } _ { B } ( T , \eta _ { T } ^ { * } ) \leq \mathcal { E } _ { L A }$ , where $\eta _ { T } ^ { * }$ is the best step; at $\rho = 4$ this holds for every � from 1 to 10 and every $n / d$ from 0.1 to 10.

• Adapter ceiling. A �-dimensional adapter with a frozen random down-projection never exceeds $\begin{array} { r } { \frac { 1 } { 2 } \log _ { 2 } \frac { 1 + \rho } { 1 + ( 1 - k / d ) \rho } } \end{array}$ . At $\rho = 4$ and $k = d / 4$ , that is 14% of the cache’s ceiling $\textstyle { \frac { 1 } { 2 } } \log _ { 2 } ( 1 + \rho )$

• Crossover. $\mathrm { A t } \rho = 4$ , an ideal cache holding � examples is matched by a fine-tune on $n ^ { * }$ examples, with $n ^ { * } / N = 1 . 0 1 -$ 1.06 for five or more steps and $n ^ { * } / N = 1 . 2 \bar { - 3 } . 8$ for one step, over $N / d$ from 1/4 to 4.

Simulation at $d = 2 0 0 .$ , with 300 repeats per point, reproduces the closed forms: over the points of Figure 5, the largest gap between the two is 0.005 bits.

## B. Proofs

## B.1. Deleted-evidence exclusion

Fix the remaining inputs $R = r$ . Let $T _ { 0 }$ denote the initialised state and write the successive transitions of the rebuilt execution as

$$
T _ { 0 } = \phi _ { 0 } ( r , U _ { 0 } ) , \qquad T _ { j + 1 } = \phi _ { j + 1 } ( r , T _ { 0 } , \ldots , T _ { j } , U _ { j + 1 } ) .\tag{28}
$$

The random sequence � has a conditional law that depends on � but not on the deleted content $G .$ By induction, the conditional law of each transcript prefix $( T _ { 0 } , \dots , T _ { j } )$ depends only on $r ,$ and is therefore identical for every value of � compatible with �. The same holds for a variable-length transcript when stopping is a measurable function of the available transcript and randomness. This proves Lemma 10.

A state cached from the full context is admissible only if it is independent of the removed contents given the remaining inputs. In a transformer, the entries of the deleted span and every entry computed after it are recomputed. A cached geometry-only state qualifies when its arguments are unchanged; a chart or hyperparameter choice computed from deleted geometry does not. Predictions reconstructed from the model’s parameters or the remaining context are always admissible. The conditional independence statement concerns the provenance of the rebuilt execution and makes no assumption that context examples are statistically independent.

## B.2. The law and its corollaries

Theorem 2. The pointwise identity in the proof holds wherever the densities are positive, and its three terms have   
expectations $I ( Q ; Z _ { c } ) , { \mathrm { K L } } ( P _ { Q } ^ { * } \| p _ { 0 } )$ and $\mathbb { E } _ { Z } \bar { \mathbf { K L } } ( P _ { Q | Z } ^ { * } | | p _ { c } ( \cdot \mid Z ) )$ ) under $P ^ { * }$ , which gives Equation (3) whenever the three   
terms are finite.

Equation (4). By the chain rule, $I ( Q ; D , Z _ { c } ) = I ( Q ; D ) + I ( Q ; Z _ { c } \mid D ) = I ( Q ; D )$ , because ${ \cal Q } \perp { \cal Z } _ { c } \mid { \cal D } ,$ and also $I ( Q ; D , Z _ { c } ) = I ( Q ; Z _ { c } ) + I ( Q ; D \mid Z _ { c } )$ . Hence $I ( Q ; Z _ { c } ) = I ( Q ; D ) - I ( Q ; D \mid Z _ { c } )$ . Moreover $\mathbb { E } _ { D } { \mathrm { K L } } ( P _ { Q | D } ^ { * } | | p _ { 0 } ) =$ $\begin{array} { r } { \mathbb { E } \log _ { 2 } \frac { P ^ { * } ( Q | D ) } { P ^ { * } ( Q ) } + \mathbb { E } \log _ { 2 } \frac { P ^ { * } ( Q ) } { p _ { 0 } ( Q ) } = I ( Q ; D ) + \mathrm { K L } ( P _ { Q } ^ { * } \| p _ { 0 } ) } \end{array}$

Corollary 3. For a predictor that sees $\begin{array} { r } { D , \mathbb { E } [ \Gamma _ { c } ] = \mathbb { E } \log _ { 2 } \frac { P ^ { * } ( Q | D ) } { p _ { 0 } ( Q ) } - \mathbb { E } \log _ { 2 } \frac { P ^ { * } ( Q | D ) } { p _ { c } ( Q | D ) } = G ^ { \star } - \mathbb { E } \mathbf { K } \mathbf { L } ( P _ { Q | D } ^ { * } | | p _ { c } ( \cdot | D ) ) } \end{array}$ , with the expectation over � and the memory, which gives (i) and, by subtraction, (ii). For (iii), data processing along $Q - \tilde { D } - Z _ { c }$ gives $\bar { I } ( Q ; Z _ { c } ) \le I ( Q ; \tilde { D } ) = 0$ , and the decoding loss is nonnegative.

Corollary 4. By Equation (2), $\Xi \big [ \Delta _ { G } \big ] \ = \ \mathbb { E } \big [ \Gamma _ { c } \big ( Q ; D \big ) \big ] \ - \ \mathbb { E } \big [ \Gamma _ { c } \big ( Q ; R \big ) \big ]$ . Equation (4) applied to each context gives $\mathbb { E } [ \Gamma _ { c } ( Q ; X ) ] = I ( \bar { Q ; X } ) + \mathrm { K L } ( P _ { O } ^ { * } \| p _ { 0 } ) - \mathrm { s t o r } _ { X } - \mathrm { d e c } _ { X }$ for $X \in \{ D , R \}$ ; the prior-repair term is common to both and cancels, and the chain rule gives $\tilde { I } ( Q ; D ) - I ( Q ; R ) = I ( Q ; G \mid R )$ , which is Equation (6). When $I ( Q ; Z _ { X } ) = I ( Q ; X )$ sto $\mathrm { r } _ { X } = 0$ . For a matched deletion �<sup>˜</sup> the content term $I ( Q ; { \tilde { G } } \mid D \setminus { \tilde { G } } )$ is zero, which leaves the storage and decoding change.

The model’s own valuation. With $\begin{array} { r } { Q \sim p _ { c } ( \cdot \mid D ) , \mathbb { E } [ \Delta _ { G } ] = \mathbb { E } \log _ { 2 } { \frac { p _ { c } ( Q \mid D ) } { p _ { c } ( Q \mid R ) } } = \mathbf { K } \mathbf { L } ( p _ { c } ( \cdot \mid D ) \| p _ { c } ( \cdot \mid R ) ) } \end{array}$ . For a spatial output, the pair $( U , Y _ { U } )$ with $U \sim \mu$ under both conditions has divergence $\begin{array} { r } { \mathrm { K L } ( \mu \| \mu ) + \int \mathrm { K L } ( p _ { D } ^ { u } \| p _ { R } ^ { u } ) \mu ( d u ) } \end{array}$ by the chain rule, and the first term is zero. The construction needs no law over the whole field.

Memory and forecast. Let $F _ { c } ~ = ~ p _ { c } ( \cdot ~ | ~ Z _ { c } )$ , a function of $Z _ { c }$ . The chain rule for the divergence gives $\begin{array} { r l } { \operatorname* { d e c } _ { c } } & { { } = } \end{array}$ $\mathbb { E } \mathrm { K L } ( P _ { O | Z _ { c } } ^ { \mathrm { * } } \parallel F _ { c } ) = I ( Q ; Z _ { c } \mid F _ { c } ) \mathrm { \neq } \mathbb { E } \mathrm { K L } ( P _ { O | F _ { c } } ^ { \mathrm { * } } \parallel F _ { c } )$ , and $I ( Q ; Z _ { c } ) = I ( Q ; F _ { c } ) + I ( Q ; Z _ { c } \mid F _ { c } )$ . Substituting both into Theorem 2 gives $\mathbb { E } [ \Gamma _ { c } ] = I ( Q ; F _ { c } ) + \mathrm { K L } ( P _ { O } ^ { * } \| _ { p _ { 0 } } ) - \mathbb { E } \mathrm { K L } ( P _ { O | F _ { c } } ^ { * } \| F _ { c } )$ , the resolution and reliability of the logarithmic score (Bröcker, 2009; Weijs et al., 2010) measured against $p _ { 0 } ;$ the term $I ( Q ; Z _ { c } \mid F _ { c } )$ cancels from this form and stays inside the decoding loss of the law.

## B.3. The lift identity

With $\mathcal { Z } = \mathbb { E } _ { p } e ^ { \Delta z }$ , log $\begin{array} { r } { \frac { p ^ { \prime } ( q ) } { p ( q ) } = \Delta z _ { q } - \log Z } \end{array}$ and $\mathrm { K L } ( p \| p ^ { \prime } ) = \mathbb { E } _ { p } [ \log \mathcal { Z } - \Delta z ] = \log \mathcal { Z } - \mathbb { E } _ { p } \Delta z$ , which gives Equation (12). Hoeffding’s lemma, log $\mathbb { E } _ { p } e ^ { X } \le \mathbb { E } _ { p } X + \mathrm { r a n g e } ( X ) ^ { 2 } / 8$ , bounds $\mathrm { K L } ( p \| p ^ { \prime } )$ . Under the true next token, E<sub>�</sub>∗ log $\begin{array} { r } { \frac { p ^ { \prime } } { p } ^ { \prime } = } \end{array}$ $\begin{array} { r } { \mathbb { E } _ { P ^ { * } } \Delta z - \log { \mathcal { Z } } = \langle P ^ { * } - p , \Delta z \rangle - { \mathrm { K L } } ( p \| p ^ { \prime } ) } \end{array}$ , and the same expectation equals KL $( P ^ { * } \| p ) - \mathrm { K L } ( P ^ { * } \| p ^ { \prime } )$ . For small moves, $\begin{array} { r } { \mathrm { K L } ( p \| p ^ { \prime } ) = \frac { 1 } { 2 } \mathrm { V a r } _ { p } ( \Delta z ) + O ( \| \Delta z \| ^ { 3 } ) } \end{array}$ and $\langle P ^ { * } - p , \Delta z \rangle = \mathbb { E } _ { p } [ ( P ^ { * } / p - 1 ) \Delta z ] = \mathbf { C o v } _ { p } ( h ^ { * } , \Delta z ) + O ( \| h ^ { * } \| ^ { 2 } \| \Delta z \| )$ , which gives Equation (15). Along $\Delta z = a h$ the quadratic is maximised at $a = \langle h ^ { * } , h \rangle _ { p } / \| h \| _ { p } ^ { 2 }$ with value $\begin{array} { r } { \frac 1 2 \| h ^ { * } \| _ { p } ^ { 2 } \cos _ { p } ^ { 2 } ( h , h ^ { * } ) } \end{array}$ while KL $\begin{array} { r } { ( P ^ { * } \| p ) = \frac { 1 } { 2 } \| h ^ { * } \| _ { p } ^ { 2 } } \end{array}$ to the same order.

## B.4. The fine-tune

Under gradient flow $\dot { \theta } _ { \nu } = \eta \nabla$ log $p _ { \theta _ { \nu } } ( D ) , { \frac { d } { d \nu } }$ log $p _ { \theta _ { \nu } } ( Q ) = \eta \langle \nabla$ log $p _ { \theta _ { \nu } } ( Q )$ , ∇ log $p _ { \theta _ { \nu } } ( D ) )$ ⟩, and integrating gives Equation (16). The token-level form follows from $\nabla _ { \theta } \log p _ { \theta } ( y ) = J ^ { \top } r$ for a softmax output. For one step, apply $f ( 1 ) =$ $\begin{array} { r } { f ( 0 ) + f ^ { \prime } ( 0 ) + \int _ { 0 } ^ { 1 } ( 1 - a ) f ^ { \prime \prime } ( a ) } \end{array}$ �� to $f ( a ) = \log p _ { \theta _ { k } + a \eta g _ { D } } ( Q )$ . If $| g ^ { \top } \nabla ^ { 2 }$ log $p g | \leq \beta \| g \| ^ { 2 }$ along the step, the remainder lies within $\pm \frac { \beta \eta ^ { 2 } } { 2 } \lVert g _ { D } \rVert ^ { 2 }$ , which gives both step conditions and, summed over steps, the �-step bound.

## B.5. The cache

Softmax attention over the union of the context keys � and the prefix keys � gives $\begin{array} { r } { o ( u ) = \frac { N _ { S } ( u ) + N _ { D } ( u ) } { \mathcal { Z } _ { S } ( u ) + \mathcal { Z } _ { D } ( u ) } } \end{array}$ with $N _ { X } ( u ) =$ $\textstyle \sum _ { i \in X } e ^ { \left. u , k _ { i } \right. } \nu _ { i }$ , and dividing the two parts by their own normalisers gives Equation (18). For each key, ${ \dot { e } } ^ { \langle u , k \rangle } = e ^ { \langle u _ { 0 } , k \rangle } e ^ { x }$ with $x = \left. u - u _ { 0 } , k \right.$ , and $e ^ { x } = 1 + \varphi ( x ) x , \mathrm { s o } e ^ { \langle u , k \rangle } \nu = e ^ { \langle u _ { 0 } , k \rangle } \nu + e ^ { \langle u _ { 0 } , k \rangle } \varphi ( x ) \nu k ^ { \top } ( u - u _ { 0 } )$ ; summing over the context gives Equation (19). In linear attention the gradient of $\begin{array} { r } { \frac { 1 } { 7 } \sum _ { i } \| \nu _ { i } - W k _ { i } \| ^ { 2 } \mathrm { a t } W = 0 \mathrm { i s } - \sum _ { i } \nu _ { i } k _ { i } ^ { \intercal } } \end{array}$ , so one unit step writes $\textstyle \sum _ { i } \nu _ { i } k _ { i } ^ { \intercal }$ For logits � and a perturbation $\delta ,$ log softmax $( z + \bar { \delta } ) _ { q } - \log$ sof $\mathrm { t m a x } ( z ) _ { q } = \delta _ { q } - [ \mathrm { l s e } ( z + \delta ) - \mathrm { l s e } ( z ) ]$ , and the bracket lies between min � and max �, so the change is at most range(�) in absolute value.

## B.6. Capacity and crossover

Data processing along $Q { - } D { - } Z _ { c }$ gives $I ( Q ; Z _ { c } ) \le I ( D ; Z _ { c } )$ , so $\operatorname { s t o r } _ { c } = I ( Q ; D ) - I ( Q ; Z _ { c } ) \geq I ( Q ; D ) - I ( D ; Z _ { c } )$ . A context memory is rebuilt from the window with randomness independent of $( D , Q )$ , so data processing along $Q - D _ { w } - Z _ { A }$ gives $I ( Q ; Z _ { A } ) \le I ( Q ; D _ { w } )$ , and with $\boldsymbol { D } = \left( D _ { w } , D _ { \mathrm { e v } } \right)$ the chain rule gives $\mathrm { s t o r } _ { A } = I ( Q ; D ) - I ( Q ; Z _ { A } ) \geq I ( Q ; D ) - I ( Q ; D _ { w } ) =$ $\begin{array} { r } { I ( Q ; D _ { \mathrm { e v } } \ | \ D _ { w } ) } \end{array}$ , with equality when $I ( Q ; Z _ { A } ) = I ( Q ; D _ { w } )$ . For a transformer, $Z _ { A } = \mathrm { K V } ( D _ { w } ) { : }$ if the layer-1 entries determine the tokens in the window, $\mathrm { K V } ( D _ { w } )$ determines $D _ { w } .$ , and equality holds. With $P _ { \mathrm { t r a i n } }$ parameters stored at � bits each, $\theta _ { T }$ has entropy at most $b P _ { \mathrm { t r a i n } }$ , which bounds $I ( D ; \theta _ { T } )$ . Equation (8) is the difference of two instances of Equation (4), which share $G ^ { \star }$ . Since stor $\textsf { 3 } \geq \left[ I ( Q ; D ) - C _ { B } \right]$ <sub>+</sub> and de $_ { ; B } \geq 0$ , the condition $\left[ I ( Q ; D ) - C _ { B } \right] _ { + } > \mathrm { s t o r } _ { A } + \mathrm { d e c } _ { A }$ makes the difference positive; with stor $\dot { \boldsymbol { \mathbf { \rho } } } _ { A } = I ( \boldsymbol { Q } ; D _ { \mathrm { e v } } \mid D _ { w } )$ , the difference is negative exactly when $I ( Q ; D _ { \mathrm { e v } } \mid D _ { w } ) > \mathrm { s t o r } _ { B } + \mathrm { d e c } _ { B } - \mathrm { d e c } _ { A }$

## B.7. Ceilings and certificates

(i) Theorem 2 with de ${ \mathrm { : } } _ { c } \geq 0$ gives $\mathbb { E } [ \Gamma _ { c } ] \leq I ( Q ; Z _ { c } ) + \mathrm { K L } _ { 0 }$ . The context memory is rebuilt from $D _ { w }$ with randomness independent of $( D , Q )$ , so $Q - D _ { w } - Z _ { A }$ is a Markov chain and $I ( Q ; Z _ { A } ) \le I ( Q ; D _ { w } )$ . For the fine-tune, $Q - D - Z _ { B }$ gives $I ( Q ; Z _ { B } ) \le I ( Q ; D )$ and $I ( Q ; Z _ { B } ) \leq I ( D ; Z _ { B } ) \leq C _ { B } . { \mathrm { ( i i ) } }$ follows from (i). (iii) When the context fits and the context memory keeps $\mathfrak { i } \mathrm { t } , I ( Q ; Z _ { A } ) = I ( Q ; D )$ , so sto $\mathrm { r } _ { A } = 0$ , and Equation (8) gives $\operatorname { \mathbb { E } } [ \Gamma _ { B } ] - \operatorname { \mathbb { E } } [ \Gamma _ { A } ] = \det _ { A } - \det _ { B } - \operatorname { s t o r } _ { B } \leq \det _ { A }$ (iv) For $N \geq W$ the window holds � examples drawn independently from the task whatever � is, so the joint law of $( D _ { w } , Q )$ , and with it $\mathbb { E } [ \Gamma _ { A } ]$ , does not depend on �; since min $\bar { \{ I ( Q ; D ) , C _ { B } \} } \le C _ { B }$ , (ii) applies at every $N \geq W .$ . Statements

$( \mathrm { i } ) { - } ( \mathrm { i } \mathrm { i } \mathrm { i } )$ also hold conditionally on inputs that both channels observe, and (iv) holds for gains averaged over the inputs;   
Section 6.1 evaluates them this way. Both channels observe the context, so conditioning on $D = d \mathrm { \ g i v e s } \mathbb { E } [ \Gamma _ { B } - \Gamma _ { A } | \mathbf { \bar { \Pi } } D =$   
$d \big | = \mathbb { E } \big [ \mathrm { K L } ( P _ { \mathcal { O } | d } ^ { \circ } \| p _ { A } ( \cdot \ | \ Z _ { A } ) ) \ | \ \dot { D } = d \big ] - \mathbb { E } \big [ \mathrm { K L } ( P _ { \mathcal { O } | d } ^ { \circ } \| p _ { B } ) \ | \ D = d \big ] \leq \mathbb { E } \big [ \mathrm { K L } ( P _ { \mathcal { O } | d } ^ { \circ } \| p _ { A } ( \cdot \ | \ \xrightarrow [ Z _ { A } ) ) \ | \ D ^ { \circ } = d \big ]$ , which is Q   
the decoding loss in context on � when the context memory keeps �.

## B.8. Learning curves, the anytime-valid test and the budgets

Proposition 7. By exchangeability, $( d _ { < j } , d _ { j } )$ has the law of $( D _ { j - 1 } , Q )$ . In both cases the channel’s memory after $j - 1$ units is the same measurable function of those units and of randomness independent of the data, so $\ell _ { c } ( d _ { j } \mid d _ { < j } )$ and $\ell _ { c } ( Q \mid D _ { j - 1 } )$ have the same law and the same expectation $L _ { c } ( j - 1 )$ . Summing over � gives the second identity.

Theorem 8. Let $\mathcal { F } _ { j }$ hold �, the two memories and $q _ { \leq j } .$ . With the task and � fixed, the memories are fixed and the queries are independent and identically distributed, so $\mathbb { E } [ \bar { \Delta _ { t } } \ | \ \mathcal { F } _ { t - 1 } ] = \mathbb { E } [ \Delta _ { t } ] \le 0$ under $H _ { 0 }$ . The mixed context memory gives $\ell _ { A } ( q \mid D ) \leq L \log _ { 2 } ( | V | / \varepsilon )$ and $\ell _ { B } ( q ) \geq 0$ , so $\Delta _ { j } \geq - c$ and each factor satisfies $1 + \lambda _ { j } \Delta _ { j } / c \geq 1 - \lambda _ { j } > 0$ . Because $\lambda _ { t }$ is fixed by $\mathcal { F } _ { t - 1 } , \mathbb { E } [ E _ { t } \mid \mathcal { F } _ { t - 1 } ] = E _ { t - 1 } \big ( 1 + \lambda _ { t } \mathbb { E } [ \Delta _ { t } \mid \mathcal { F } _ { t - 1 } ] / c \big ) \leq E _ { t - 1 }$ , and Ville’s inequality for nonnegative supermartingales (Ville, 1939) gives Pr(sup<sub>�</sub> $E _ { t } \ge 1 / \dot { \alpha } ) \le \alpha E _ { 0 } = \alpha$ . On the prequential increments, with $\mathcal { F } _ { j }$ holding the first � units and the channels’ randomness, the same computation holds under the conditional hypothesis $\mathbb { E } [ \tilde { \Delta _ { j } } \mid \mathcal { F } _ { j - 1 } ] \leq 0$ for every �.

Theorem 9. Write $P _ { j } ( \cdot \mid D ) = p _ { A } ( \cdot \mid D , x _ { j } )$ and $P _ { j } ( \cdot \mid S ) = \mathbb { E } [ P _ { j } ( \cdot \mid D ) \mid S ]$ . Since $\hat { Y } _ { j }$ is sampled given � independently of $S , \hat { Y } _ { j } \perp \perp S \mid D$ , and for any decoder that reads �,

$$
\begin{array} { r } { \mathbb { E } \mathrm { K L } \big ( P _ { j } ( \cdot \mid D ) \mid \mid p _ { S } ( \cdot \mid x _ { j } ) \big ) = \mathbb { E } \mathrm { K L } \big ( P _ { j } ( \cdot \mid D ) \mid \mid P _ { j } ( \cdot \mid S ) \big ) + \mathbb { E } \mathrm { K L } \big ( P _ { j } ( \cdot \mid S ) \mid \mid p _ { S } ( \cdot \mid x _ { j } ) \big ) \geq I ( \hat { Y } _ { j } ; D \mid S ) . } \end{array}
$$

Given $( D , S )$ the outputs are independent, so $H ( \hat { Y } _ { j } \mid D , S , \hat { Y } _ { < j } ) = H ( \hat { Y } _ { j } \mid D , S )$ while $H ( \hat { Y } _ { j } \mid S , \hat { Y } _ { < j } ) \le H ( \hat { Y } _ { j } \mid S )$ ; hence $I ( \hat { Y } _ { j } ; D \mid S , \hat { Y } _ { < j } ) \le I ( \hat { Y } _ { j } ; D \mid S )$ , and the chain rule gives $\begin{array} { r } { \sum _ { j } I ( \hat { Y } _ { j } ; D \mid S ) \ge I ( \hat { Y } _ { 1 : K } ; D \mid S ) } \end{array}$ . Finally $I ( \hat { Y } _ { 1 : K } ; D \mid S ) =$ $I ( \hat { Y } _ { 1 : K } ; D ) + I ( \hat { Y } _ { 1 : K } ; S \mid D ) - I ( \hat { Y } _ { 1 : K } ; S ) = I ( \hat { Y } _ { 1 : K } ; D ) - I ( \hat { Y } _ { 1 : K } ; S )$ , and $I ( \hat { Y } _ { 1 : K } ; S ) \le I ( D ; S )$ by data processing along $\hat { Y } _ { 1 : K } - D - S ;$ when � takes finitely many values, $I ( D ; S ) \le H ( S )$ . The first term of the decomposition is the storage term and the second the decoding term; if � determines $D .$ , the storage term is zero.

Equation (22). With � uniform on the variants, the marginal law of $\hat { Y }$ is $\begin{array} { r } { \bar { p } = \frac { 1 } { M } \sum _ { m ^ { \prime } } p _ { A } ( \cdot \mid D ^ { ( m ^ { \prime } ) } ) } \end{array}$ , and $I ( { \hat { Y } } ; D ) =$ $\begin{array} { r } { { \frac { 1 } { M } } \sum _ { m } \mathrm { K L } \big ( p _ { A } ( \cdot \mid D ^ { ( m ) } ) \big \| \bar { p } \big ) } \end{array}$ , which is the right side. If instead $V = ( D ^ { ( 1 ) } , \dots , D ^ { ( \bar { M } ) } )$ is drawn independently from the data distribution and � is uniform on $\{ 1 , \ldots , M \}$ , then $D = D ^ { ( U ) }$ has the data distribution, $\hat { Y }$ depends on $( U , V )$ only through �, and the right side equals $I ( \hat { Y } ; U \mid V = \nu )$ . Its expectation over � is $I ( \hat { Y } ; U \mid V ) = I ( \hat { Y } ; D \big ) ^ { \circ } - I ( \hat { Y } ; V ) ^ { \circ } \leq I ( \hat { Y } ; D \big )$ and it never exceeds $H ( U ) = \log _ { 2 } M$

Proposition 17. Under $\dot { \theta } _ { t } = \eta P _ { t } g _ { D }$ with $g = \nabla _ { \theta }$ log $\begin{array} { r } { p _ { \theta } , \frac { d } { d t } \Gamma _ { B } ( Q ) = \eta g _ { O } ^ { \top } P _ { t } g _ { D } } \end{array}$ and $\begin{array} { r } { \frac { d } { d t } \Gamma _ { B } ^ { \mathrm { t r } } = \eta g _ { D } ^ { \top } P _ { t } g _ { D } \geq 0 } \end{array}$ . Dividing gives $r _ { t } .$ and since $\Gamma _ { B } ^ { \mathrm { t r } }$ is nondecreasing, the change of variables gives the integral. Then $\begin{array} { r } { | \Gamma _ { B } ( Q ) | \leq \int | r | d \Gamma _ { B } ^ { \mathrm { t r } } \leq \operatorname* { s u p } _ { t } | r _ { t } | \Gamma _ { B } ^ { \mathrm { t r } } ( T ) } \end{array}$ and $\Gamma _ { B } ^ { \mathrm { t r } } ( T ) \leq \ell _ { \theta _ { 0 } } ( D )$ because a discrete code length is nonnegative. The ratio is unchanged when both gains are converted from nats to bits

## B.9. The lift budget

Given $D _ { \colon }$ , the output $\hat { Y } _ { c }$ has law $\pi _ { c } ( \cdot \mid D )$ , so as for $G ^ { \star } , \mathbb { E } _ { D } \mathrm { K L } ( \pi _ { c } ( \cdot \lfloor D ) \| p _ { 0 } ) = I ( \hat { Y } _ { c } ; D ) + \mathrm { K L } ( \bar { \pi } _ { c } \| p _ { 0 } )$ . Data processing through $\mathbf { 1 } _ { E }$ gives $d _ { 2 } ( \pi _ { c } ( E \mid D ) \| p _ { 0 } ( E ) ) \le \mathrm { K L } ( \pi _ { c } ( \cdot \mid D ) \| p _ { 0 } )$ , and the Markov chain $D - Z _ { c } - \hat { Y } _ { c }$ gives $\hat { I } ( \hat { Y } _ { c } ; D ) \ \bar { \le }$ $I ( Z _ { c } ; D ) \le C _ { c }$ . This proves Theorem 15.

For Corollary 16, let $P \ = \ p _ { c } ( \cdot \ | \ D ) , \ P ^ { \prime } \ = \ p _ { c } ( \cdot \ | \ R ) , \ p \ = \ P ( \mathcal { A } )$ and $q \ = \ P ^ { \prime } ( { \mathcal { A } } )$ with $0 ~ < ~ p , q ~ < ~ 1$ . On A, $d P / d P ^ { \prime } = ( \dot { p _ { \ L } } / q ) \left( d P _ { \ L \mathcal { A } } / d P _ { \ L \mathcal { A } } ^ { \prime } \right)$ for the conditional laws $P _ { \mathcal { A } }$ and $P _ { \mathcal { A } } ^ { \prime }$ , and on $\mathcal { A } ^ { c }$ the analogous ratio uses $( 1 - p ) / ( 1 - q )$ Integrating over the two sets gives the exact coarse-graining identity

$$
\mathrm { K L } ( P \| P ^ { \prime } ) = d _ { 2 } ( p \| q ) + p \mathrm { K L } ( P _ { \mathcal { R } } \| P _ { \mathcal { R } } ^ { \prime } ) + ( 1 - p ) \mathrm { K L } ( P _ { \mathcal { R } ^ { c } } \| P _ { \mathcal { R } ^ { c } } ^ { \prime } ) ,\tag{29}
$$

so $I _ { G } \geq d _ { 2 } ( p \Vert q )$ with the stated gap. The derivative $\begin{array} { r } { \frac { \partial } { \partial r } d _ { 2 } ( r \| q ) = \log _ { 2 } \frac { r ( 1 - q ) } { q ( 1 - r ) } } \end{array}$ is positive on $( q , 1 )$ , so for $q < t$ every law $\tilde { P }$ with $\tilde { P } ( \mathcal { R } ) \geq t$ has $\mathrm { K L } ( \tilde { P } \| P ^ { \prime } ) \geq d _ { 2 } ( \tilde { P } ( \mathcal { A } ) \| q ) \geq d _ { 2 } ( t \| q )$ . The law with density �/ $t / q$ on A and $( 1 - t ) / ( 1 - q )$ on $\mathcal { A } ^ { c }$ relative to $P ^ { \prime }$ integrates to one, keeps both conditional laws, and attains that value by making both conditional terms in Equation (29) zero. For $q < t$ and $p \geq q$ , strict monotonicity gives $d _ { 2 } ( p \| q ) \geq d _ { 2 } ( t \| q )$ if and only if $p \geq t ,$ , which yields both implications.

Algorithm 1 Context ablation as held-out validation   
Require: Predictor $f _ { \boldsymbol { \theta } } ,$ query �, kept context $C _ { 0 } ,$ context �, deletions ${ \mathcal { G } } ,$ observation rule �, target �   
1: � ← PREDICTFRESH $( f _ { \theta } , x , C _ { 0 } ^ { \bar { \theta } , C ) }$ ⊲ memory from �   
2: Fix claims and observation indices for every branch   
3: for $G \in { \mathcal { G } }$ do   
4: � ← PREDICTFRESH( $f _ { \theta } , x , C _ { 0 } , C \setminus G )$ ⊲ rebuilt   
5: Check identical coordinates, readout and inference policy   
6: $I  \mathrm { K L } ( P O \parallel P _ { G } O )$ ⊲ Equation (7)   
7: for each fixed claim $\mathcal { A }$ on the observed output do   
8: $p \gets P O ( \mathcal { A } ) ; q \gets P _ { G } O ( \mathcal { A } )$   
9: $J \gets d _ { 2 } ( p \| q ) ; b \gets b _ { t } ( q )$   
10: Record $( I , J , b , p , q )$ and whether $p \geq t > q$   
11: end for   
12: end for

## B.10. Closed forms

Gain. Given �, the prior predictive of � is $N ( 0 , \sigma ^ { 2 } + \bar { \tau } ^ { 2 } \| x \| ^ { 2 } / d )$ , and $\| \boldsymbol { x } \| ^ { 2 } / d \to 1$ . A calibrated Gaussian predictive has variance equal to its expected squared error, which for an error $w - \hat { w }$ independent of the query input is $\sigma ^ { \bar { 2 } } + x ^ { \top } \mathbb { E } [ ( w -$ $\hat { w } \big ) \big ( w - \hat { w } \big ) ^ { \top } \big ] x .$ , and this concentrates at $\sigma ^ { 2 } + \mathcal { E } _ { c }$ . The expected log-density of a Gaussian predictive whose variance � equals its expected squared error $\begin{array} { r } { \mathrm { i s } - { \frac { 1 } { 2 } } \log _ { 2 } ( 2 \pi e \nu ) } \end{array}$ , so the difference of the two tends to $\begin{array} { r } { \frac { 1 } { 2 } \log _ { 2 } { \frac { \sigma ^ { 2 } + \bar { \tau } ^ { 2 } } { \sigma ^ { 2 } + \mathcal { E } _ { c } } } } \end{array}$ , which is Equation (24).

Ideal cache. The posterior mean is $\hat { w } \ = \ ( X ^ { \top } X + \lambda I ) ^ { - 1 } X ^ { \top } y$ with $\lambda = \sigma ^ { 2 } / \tau ^ { 2 }$ , and its error is $\lambda ( X ^ { \top } X + \lambda I ) ^ { - 1 } w -$ $( X ^ { \top } X + \lambda I ) ^ { - 1 } X ^ { \top } \varepsilon .$ . Its expected squared norm is $\sigma ^ { 2 } \mathbb { E } \operatorname { t r } ( X ^ { \top } X + \lambda I ) ^ { - 1 } = \sigma ^ { 2 } \gamma m ( - \kappa )$ , where � is the Stieltjes transform of the empirical spectrum of $X ^ { \top } \bar { X } / n$ and $\lambda / n = \gamma / \rho = \kappa .$ In the proportional limit, $m ( - \kappa )$ is the positive root of $\gamma \kappa m ^ { 2 } + \bigl ( \bar { \kappa } + 1 - \gamma \bar { \bigr ) } m - 1 = 0$ , which is $s _ { \gamma } ( \boldsymbol { \kappa } )$ . The base predictive is the prior predictive, so the prior repair is zero, and the Bayes predictive has zero decoding loss; by Theorem 2 its gain is $I ( y _ { q } ; D \mid x _ { q } )$

One layer, one step. One step of size � from zero gives $\hat { w } = \eta X ^ { \top } y / n .$ , with excess risk $\bar { \tau } ^ { 2 } - 2 \eta \bar { \tau } ^ { 2 } + \eta ^ { 2 } \mathbb { E } \| X ^ { \top } y / n \| ^ { 2 }$ . Since $\mathbb { E } \operatorname { t r } ( ( \bar { X ^ { \top } } X ) ^ { 2 } ) = n \hat { d } ( n + d + \hat { 1 } ) , \mathbb { E } \| X ^ { \top } \dot { y ^ { \prime } } / n \| ^ { 2 } = [ \tau ^ { 2 } \bar { d } ( n + d + \dot { 1 } ) + \dot { \sigma } ^ { 2 } d ] / n$ . Minimising over � gives Equation (26) exactly at finite � and $d ,$ the risk of one linear-attention layer with its best scalar step; in the proportional limit the minimiser is $\eta ^ { * }$ and the risk is $\bar { \tau } ^ { 2 } \gamma ( \bar { \tau } ^ { 2 } + \sigma ^ { 2 } ) / ( \bar { \tau } ^ { 2 } ( 1 + \gamma ) + \sigma ^ { 2 } \gamma )$

� steps. With $S = X ^ { \top } X / n , T$ steps from zero give $w { - } w _ { T } = ( I { - } \eta S ) ^ { T } w { - } [ I { - } ( I { - } \eta S ) ^ { T } ] S ^ { - 1 } X ^ { \top } { \varepsilon } / n$ , where $[ I - ( I - \eta S ) ^ { T } ] S ^ { - 1 }$ denotes the polynomial $\begin{array} { r } { \sum _ { k = 1 } ^ { T } \binom { \dot { T } } { k } ( - 1 ) ^ { k + 1 } \eta ^ { k } \overline { { S } } ^ { k - 1 } } \end{array}$ , defined for every � including $d > n$ . Its expected squared norm is $\begin{array} { r } { \frac { \bar { \tau } ^ { 2 } } { d } \mathbb { E } \operatorname { t r } ( I - \eta S ) ^ { 2 T } + \frac { \sigma ^ { 2 } } { n } \mathbb { E } \operatorname { t r } \big ( [ I - ( I - \eta S ) ^ { T } ] ^ { 2 } S ^ { - 1 } \big ) } \end{array}$ . Expanding both powers binomially and replacing ${ \frac { 1 } { d } } \operatorname { t r } S ^ { k }$ by its limit $M _ { k }$ gives Equation (27).

Adapter. An adapter that moves the predictor only within a �-dimensional random subspace � independent of � leaves at least $\operatorname { \mathbb { E } } \lVert P _ { U ^ { \perp } } w \rVert ^ { 2 } = ( 1 - k / d ) \bar { \tau } ^ { 2 }$ of excess risk, and Equation (24) turns that floor into the ceiling.

Numbers. The ordering, the crossover ratios and Figure 5 are computed from Equations (25)–(27) with the best step found by one-dimensional minimisation, and the simulations draw �, � and � afresh in every repeat.

## C. Experiments in full

This appendix gives the full results summarised in Sections 4 and 6, in the same order; the protocols are in Appendix D.

## C.1. One evaluator for any memory

One scorer, task-specific adapters. Algorithm 1 is the whole evaluator. For a spatial output with sites � drawn from a fixed measure $\mu ,$ the bits per observation are $\begin{array} { r } { I _ { G } ^ { \mu } = \int \mathrm { K L } ( p _ { D } ^ { u } \| p _ { R } ^ { u } ) \mu ( d u ) } \end{array}$ . An adapter supplies a prediction law or a claim probability, an admissible deletion and a common output observation; the scorer itself contains nothing architecture-specific. Every deletion rebuilds the memory from what remains (Assumption 1): the occupancy network re-encodes its grid, the stereo network rebuilds its cost volume, the kriging learner refits on the remaining cars, and the language verifier recomputes its cache. The gain $I _ { G }$ is computed whenever the law is available, and a claim-only adapter supplies �. The language experiments run the published Trace Budget adjudicator, the binary case of the same test (Appendix C.2.6).

![](images/131addfc82a9ece68f2f9cd25ac2428b6346f91e15d58cfcf270606f0da3def2.jpg)  
(a) The feature grid of an occupancy network

![](images/105d7414e911faf62f30d0790fde9196655fe376d4c7cda1ad27e1559b7363cb.jpg)  
(b) The cost volume of a stereo network  
Figure 6. Relevant evidence against a matched control of the same form. (a) Occupancy probability at the feature sites of eight constructed chair pairs, with the full context, after deleting 300 matched points from unchanged geometry, and after deleting the 300 points that observe the feature; the last row observes the recessed shape. Points are per-pair means over 1,024 sites and diamonds are means over pairs. (b) Extra loss of claim probability, in percentage points, from deleting a source view that observes the queried surface point rather than one that does not, by DTU scene; each point averages five reference views. Scene 9 was used to develop the protocol, and scenes 4, 48, 11 and 29 were drawn by seed before any of their predictions were seen. Claims whose interval misses the reference (oracle) depth are shown for comparison.

Claims are fixed before any deletion, either from geometry before any prediction (occupancy features) or from the full-context prediction (depth intervals and pressure tolerances), and no claim is re-centred afterwards. Deleting nothing reproduces the full prediction, and the depth and field adapters reproduce their native implementations exactly, including preprocessing and context-dependent state (Appendices D.2 and D.3). For a fixed query and � single-item deletions the protocol runs � + 1 predictions; group deletions add predictions, not scoring rules.

Matched controls and statistical units. Every experiment pairs the deletion of relevant evidence with a matched control of the same form (Table 2): an equal-size deletion from unchanged geometry, a camera that does not observe the queried point, or donor evidence whose agreement with the claim is known from independent geometry. By Corollary 4, the control’s gain is what a deletion of that form costs by itself, and the content of the relevant evidence accounts for everything beyond it. Mesh and depth references define whether evidence supports a claim, so relevance never rests on the source of the evidence, and reference pressures test the predictor’s accuracy independently of its gains.

Spatial means are computed within each query population. Geometry results weight objects or constructed pairs equally; multi-view results average five reference views per scene and then the scenes; field results weight sites by area within a car and weight cars equally. Crossing counts are pooled and reported with their denominators. Scene-level intervals resample scenes, so pixel counts never act as sample sizes.

## C.2. Results across four memories

## C.2.1. A FEATURE GRID: 17.7 BITS PER QUERY FROM A LOCAL SHAPE FEATURE

ConvONet encodes the observed points into a feature grid and decodes Bernoulli occupancy at any spatial query by reading the grid there (Peng et al., 2020). The grid is its memory of the context. We construct eight chair pairs that share their surrounding geometry and differ by a local raised feature versus a recess. Each condition keeps 300 common anchor points and nine optional items of 300 points: item 0 holds the observations that distinguish the pair, and item 1 holds a matched set from unchanged geometry. Query sites and claims (“this site is occupied”) are fixed from the constructed geometry before any prediction (Appendix D.1).

Deleting the distinguishing item and rebuilding the grid drops mean occupancy probability from 0.9827 to 0.0014 and costs 17.6642 bits per query on the full-context continuation (Table 3 and Figure 6a). Without the item the rebuilt grid predicts free space at the feature sites, as for the recess, and the 17.66 bits are what the item’s observations save by overturning that confident prediction, priced site by site by Equation (7). Deleting the same number of points from unchanged geometry leaves the probability at 0.9834 and costs 0.00111 bits. All 7,473 confident feature claims fall below 0.95 under the relevant deletion, against 70 under the matched control, and the full present-versus-absent contrast is positive in all eight pairs (mean 0.981).

Table 2. One intervention, four memories. Counts are examples, objects, scenes or cars, which are the units of replication.
<table><tr><td>Task</td><td>Model and data</td><td>Memory built from the context</td><td>Deleted context</td><td>Output law and claim</td><td>Scale</td></tr><tr><td>Controlled occupancy</td><td>Pretrained ConvONet; constructed chairs</td><td>Feature grid from the observed points</td><td>300 observed surface points</td><td>Bernoulli occupancy; the site is occupied</td><td>8 shape pairs; 1,024 feature sites per pair; relevant and matched deletions of equal size 16 objects; 8,492 fixed claims;</td></tr><tr><td>Donor replacement</td><td>Same ConvONet; ShapeNet objects</td><td>Feature grid</td><td>Optional point-cloud observations</td><td>Bernoulli occupancy at fixed world coordinates</td><td>independent donor-mesh labels</td></tr><tr><td>Multi-view depth</td><td>Pretrained MVSNet; DTU</td><td>Cost volume over the source views</td><td>Source image and camera</td><td>Categorical depth law; depth in a fixed interval</td><td rowspan="2">5 scenes, 25 reference views; paired observing and non-observing deletions 8 query cars, 6 support cars</td></tr><tr><td>Pressure fields</td><td>Kriging learner; DrivAerNet</td><td>Kriging weights, recomputed for each site</td><td>Support car (geometry and pressure)</td><td>Gaussian pressure law; pressure within a fixed tolerance</td></tr><tr><td>Language verification</td><td>Trace Budget adjudicator; VitaminC</td><td>Key-value cache</td><td>Evidence passage</td><td>Probability of the fixed claim</td><td>800 examples; Qwen2.5-7B-Instruct and GPT-4.1-mini</td></tr></table>

Table 3. Feature grid: eight pairs, 8,192 feature-site queries, � = 0.9827 with the full feature-present context. Both deletions remove 300 points. The gain is averaged site by site. � counts confident claims (7,473 with the full context) that fall below 0.95.
<table><tr><td>Condition</td><td>Mean q</td><td>Gain  $I _ { G }$  (bits)</td><td>X</td></tr><tr><td>Feature item deleted</td><td>0.001410</td><td>17.6642</td><td>7,473</td></tr><tr><td>Matched item deleted</td><td>0.983387</td><td>0.00111</td><td>70</td></tr><tr><td>Recessed shape observed</td><td>0.002191</td><td>17.4116</td><td></td></tr></table>

Once the distinguishing item is deleted, the two shapes in a pair present identical inputs and receive identical predictions, which verifies the intervention end to end. Single-item gains also locate the relevant evidence: deleting the two items with the largest single-item gains (600 points) costs 17.58 bits, against 4.99 bits for a random pair of items.

## C.2.2. DONOR EVIDENCE COUNTS WHERE ITS GEOMETRY AGREES

The natural-object experiments use 16 ShapeNet shapes, two in each of eight categories (Chang et al., 2015). Donor replacement keeps 300 anchor points of the target and substitutes the optional observations of another object in the same category. We follow the 8,492 claims that the target’s own optional observations lift above 0.95; with donor observations in their place, 5,369 of them stay confident. Independent donor meshes explain which ones:

<table><tr><td colspan="2">still at 0.95 / claims</td><td>rate</td></tr><tr><td>Donor agrees</td><td>5,305/5,902</td><td>89.88%</td></tr><tr><td>Donor disagrees</td><td>64/2,589</td><td>2.47%</td></tr></table>

In total, 98.81% of the surviving confidence sits where the donor’s geometry supports the claim (Figure 7); equal-object rates are 78.34% and 2.30%, and one boundary-ambiguous claim is reported separately (Table 15).

A coherent-object experiment replaces the whole observation set, anchors included. At interior coordinates occupied by the target but empty in the donor, mean occupancy probability is 0.9587 with the target’s observations and 0.0351 with the donor’s, averaging the eligible donors within each target and then the targets, and the contrast is positive for all 16 targets. Confidence tracks whether the evidence supports the claim at that coordinate, not which object the evidence came from.

## C.2.3. A COST VOLUME: CAMERAS THAT SEE THE SURFACE CARRY THREE TIMES THE BITS

MVSNet builds a cost volume from the reference and source views and reads a categorical distribution over depth hypotheses from it (Yao et al., 2018). We keep the reference image and camera and delete one of four source image–camera pairs, holding the depth hypotheses and each claim’s interval fixed under native preprocessing. The experiment covers five DTU scenes (Jensen et al., 2014) with five reference views each. Independent source depths classify each source view as observing the queried surface point, occluded, outside or behind the camera, or unknown. At each eligible query we delete one observing and one non-observing source, chosen by slot order rather than by model output, and views of unknown visibility are never used as non-observing. Each claim states that depth lies in a fixed interval around the full-context expected depth, and claims are grouped by whether the reference depth lies inside the interval, outside it, or near its boundary (Appendix D.2).

![](images/f1d5b869264a9d33a74fee61150eb405efcc968ffefc159f0f6871c9c05fad50.jpg)  
Figure 7. Donor replacement on 8,492 fixed claims. Bars give the share of claims still at 0.95 when a same-category donor’s observations replace the target’s optional observations, split by whether the donor mesh agrees with the claim; rates are pooled over locations.

Table 4. Cost volume: depth claims whose reference depth lies inside the fixed interval, 11,808 paired locations, of which 8,251 reach the target with the full context. Means weight scenes equally; crossing counts are pooled. Each deletion removes one source view.
<table><tr><td>Deleted source view</td><td>Observing</td><td>Non-observing</td></tr><tr><td>Full probability p</td><td>0.8952</td><td>0.8952</td></tr><tr><td>Ablated probability q</td><td>0.7552</td><td>0.8504</td></tr><tr><td>Gain IG (bits)</td><td>1.6204</td><td>0.5452</td></tr><tr><td>Claim-level bits J</td><td>0.8391</td><td>0.1872</td></tr><tr><td>Target crossings</td><td>1,840</td><td>760</td></tr><tr><td>Reverse crossings</td><td>153</td><td>455</td></tr></table>

Deleting an observing view and rebuilding the cost volume costs 1.6204 bits per query; deleting a non-observing view costs 0.5452 (Table 4). Through the depth claim the values are 0.8391 and 0.1872 bits, each below its gain as Corollary 16 requires. Deleting the observing view also costs 9.52 more percentage points of claim probability, with the larger loss in all five scenes (Figure 6b; 95% scene-bootstrap interval [5.04, 13.99]). On the four scenes added beyond the development scene, the effect is 11.06 points (interval [6.45, 14.83]). The effect is 4.5 times larger for correct depth intervals than for incorrect ones, whose extra loss is 2.11 points.

## C.2.4. REQUIRED AND SUPPLIED BITS

Corollary 16 turns every gain into a decision. For each deletion, Table 5 gives the bits a claim needs to return to 0.95 from its ablated probability, the bits the deleted context supplies, and how many claims actually cross the target. The supplied column is code length: when outcomes follow the full-context prediction, coding them with the rebuilt memory costs that many extra bits per query.

The feature item supplies 17.66 bits against a 16.93-bit requirement, and every confident feature claim falls without it; the matched control supplies 0.001 bits against 0.006. An observing view supplies 1.62 bits against a 1.27-bit requirement and accounts for 2.4 times as many lost claims as a non-observing view, which supplies 0.55 bits against 0.53. A single support car in the kriging learner supplies 0.23 bits against 0.74, and deleting it changes the decision on 4.3% of the surface. In both tasks with a matched control, the relevant deletion supplies more bits and moves more claims across the target.

Table 5. Bits required and supplied per query. Required: mean site-wise budget $b _ { 0 . 9 5 } ( q )$ after the deletion (zero when $q \geq 0 . 9 5 )$ ). Supplied: mean gain $I _ { G }$ . Claim: mean ${ \bar { J } } .$ Crossings: confident claims that fall below 0.95. Means of site-wise values with the weighting of Tables 3 and 4; pressure uses native inference with claims at the native answer, weighting sites by area and cars equally.
<table><tr><td>Deletion</td><td>Required</td><td>Supplied</td><td>Claim</td><td>Crossings</td></tr><tr><td>Feature item</td><td>16.93</td><td>17.66</td><td>17.66</td><td>7,473/7,473</td></tr><tr><td>Matched item</td><td>0.006</td><td>0.001</td><td>0.001</td><td>70/7,473</td></tr><tr><td>Observing view</td><td>1.27</td><td>1.62</td><td>0.84</td><td>1,840/8,251</td></tr><tr><td>Non-observing view</td><td>0.53</td><td>0.55</td><td>0.19</td><td>760/8,251</td></tr><tr><td>One support car</td><td>0.74</td><td>0.23</td><td>0.04</td><td>4.3% of area</td></tr></table>

![](images/b7d9fe871615ff9eb43f493866c0a128874e5e465a6025f31fa2317fb98b40d0.jpg)  
Figure 8. Pressure error with the reduced input and with native inference. Lines connect the same query car at the same evaluation sites; the learner and the scorer are identical in both conditions.

## C.2.5. ATTENTION WITHOUT A CACHE: NATIVE INFERENCE IS PART OF THE INTERVENTION

The field predictor is a context-conditioned kriging learner for surface pressure on eight DrivAerNet query cars (Elrefaie et al., 2024), each with six support cars that pair geometry with CFD surface pressure. It attends to its support set without caching it (Appendix A.5). The learner has no offline training: it selects its chart and regression settings from the support set by leave-one-car-out prediction, then transports a mean and an error-based spread to the query, giving a Gaussian law at every surface site (Appendix D.3). Every deletion refits the learner on the remaining cars.

The learner’s descriptors and bandwidth depend on the query geometry it receives. Passing only the 512 evaluation sites into inference therefore defines a different predictor from passing the full 32,000-point query and reading out the same 512 sites afterwards (Table 6). Native inference cuts mean relative $L _ { 2 }$ pressure error from 0.5276 to 0.2992, a 43.3% reduction in the mean, with every one of the eight cars improving (Figure 8); the full query geometry alone gives 39.1%. It also changes the measured gain: the mean single-car value falls from 0.79 to 0.23 bits, so the more accurate predictor relies less on any one support car. All 48 native single-car deletions carry positive gain, and the evaluated learner reproduces the origina implementation’s means, variances and selected configurations exactly in 168 comparisons.

Claims fix the native full-context answer at each site: pressure lies within 0.05 of it. Deleting a single car moves 4.3% of the surface area across the 0.95 target, and a second claim bank fixed at the reduced-input answer gives 1.3% (Table 17).

## C.2.6. A KEY–VALUE CACHE: THE BINARY CASE IS EVIDENCE ADJUDICATION

With a binary claim and an evidence passage as the deleted span, the lift budget of Corollary 16 is the Trace Budget adjudicator of Chlon et al. (2026). A verifier scores the claim with the evidence present and with the cited evidence scrubbed.

Table 6. Pressure prediction on eight fixed query cars and evaluation sites. The single-car gain averages the six deletions and eight cars. Reference pressure enters only the error.
<table><tr><td></td><td>Reduced input</td><td>Native inference</td></tr><tr><td>Geometry points</td><td>512</td><td>32,000</td></tr><tr><td>Chart anchors</td><td>128</td><td>1,024</td></tr><tr><td>Leave-one-out points</td><td>512</td><td>4,000</td></tr><tr><td>Mean relative  $L _ { 2 }$  error</td><td>0.5276</td><td>0.2992</td></tr><tr><td>Mean single-car gain  $I _ { G }$  (bits)</td><td>0.7939</td><td>0.2296</td></tr></table>

Table 7. Evidence adjudication on VitaminC. Higher scores indicate less support. Flag counts are the adjudicator’s fixed decisions at target 0.95. The 390 claim groups pair the same claim with supporting and with NEI or refuting evidence.
<table><tr><td>Metric</td><td>Qwen2.5-7B</td><td>GPT-4.1-mini</td></tr><tr><td>Sufficiency AUROC (515)</td><td>0.9126</td><td>0.9143</td></tr><tr><td>Full-grounding AUROC (800)</td><td>0.8990</td><td>0.9297</td></tr><tr><td>NEI flagged / 120</td><td>114</td><td>112</td></tr><tr><td>REFUTES flagged / 285</td><td>261</td><td>275</td></tr><tr><td>SUPPORTS flagged / 395</td><td>103</td><td>99</td></tr><tr><td>Correct paired direction / 390</td><td>370</td><td>371</td></tr></table>

The adjudicator passes the claim when the evidence lifts it across the target: the full-context probability reaches 0.95 and the scrubbed probability does not, so the evidence supplies at least $d _ { 2 } ( 0 . 9 5 \| q )$ bits. Its score is the budget gap, required minus supplied bits. We run the published adjudicator (Chlon, 2026) unchanged on VitaminC contrastive evidence (Schuster et al., 2021) with Qwen2.5-7B-Instruct and GPT-4.1-mini as verifiers, holding each claim fixed while its evidence changes.

The 800 examples contain 395 SUPPORTS, 120 NOT ENOUGH INFO (NEI) and 285 REFUTES cases. Sufficiency AUROC compares SUPPORTS with NEI (515 examples); full-grounding AUROC treats NEI and REFUTES together as unsupported (800 examples).

The budget gap separates supporting from insufficient evidence with AUROC 0.9126 for Qwen and 0.9143 for GPT-4.1-mini, and supporting from all unsupported evidence with AUROC 0.8990 and 0.9297 (Table 7 and Figure 9). For the same claim, supporting evidence receives the lower score in 370 and 371 of 390 claim groups. At its fixed operating point the adjudicator flags 95.0% and 93.3% of NEI cases and 91.6% and 96.5% of refuting cases, with flag rates of 26.1% and 25.1% on supporting evidence (Appendix D.4).

## C.2.7. LEARNING CURVES IN CONTEXT

Table 8 gives the curves of Figure 2. In every one of the eight chair pairs, the eight patches that do not observe the feature leave the feature sites costlier than the anchor points alone, and the feature patch alone brings them to between 0.001 and 0.22 bits per site. One observing source view codes the depth better than one view drawn at random in all five scenes, and better than all four views together in scenes 4, 29 and 48. With three, four and five support cars, the order of the learner’s own valuation codes better than the mean of the random orders in six, five and six of the eight cars.

## C.3. Choosing the channel on a trained transformer

Proposition 6 is tested where the information terms are exact. A three-layer transformer with 104,194 parameters is trained to learn the tasks of Appendix A.10 in context $( d = 8 , \rho = 4 ) ;$ ; each token carries the current input and the previous example, and the head returns a Gaussian predictive for the next label. Its code length is within 0.025 bits per label of the Bayes predictive on average. A query is a block of 16 examples whose labels are teacher-forced, and the context holds � further examples of the same task, for � from 4 to 1,024. Three channels share the trained weights:

• the cache reads the last � context examples and then the query block, with a window of $W = 8$ or $W = 1 6 ;$

• the full fine-tune trains all weights on the � examples with Adam and then reads the query block alone, with its learning rate and number of steps chosen for each � on separate tasks to maximise its gain;

• �-bit adapters train � pseudo-labels for fixed inputs, placed in front of every sequence, and quantise them to � bits, so that they store at most $C _ { B } = k b$ bits, for (�, �) equal to (2, 1), (4, 1), (8, 1) and (8, 2).

Given the inputs, $I ( Q ; D )$ and $I ( Q ; D _ { w } )$ are Gaussian log-determinants and $\ell ^ { * } ( Q )$ is a Gaussian log-density. A certificate is issued when the lower end of the 95% interval of its margin over 64 tasks is positive, and the measured winner is the channel whose paired gain difference has a 95% interval that excludes zero (Appendix D.5).

![](images/7afe64a1b3e468dcbd573ec9c93d1cd11fe77e65811a497644dd3135926071f4.jpg)  
Figure 9. Sufficiency ROC from the budget gap on 395 supporting and 120 NEI examples. Table 7 adds the full-grounding result with all 285 refuting examples.

Table 8. Learning curves in context: mean held-out code length at each number of context items, by order of arrival (Figure 2). Occupancy: bits per feature site, eight pairs, count 0 is the 300 anchor points alone. Stereo: bits per depth query, five scenes, 3,856 pixels. Kriging: log-density bits per pressure value, eight cars.
<table><tr><td>Occupancy network, patches</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td></td></tr><tr><td>Feature patch first</td><td>4.99</td><td>0.046</td><td>0.021</td><td>0.019</td><td>0.014</td><td>0.016</td><td>0.014</td><td>0.015</td><td>0.015</td><td>0.015</td></tr><tr><td>Random order</td><td>4.99</td><td>5.13</td><td>5.26</td><td>4.68</td><td>4.31</td><td>3.86</td><td>3.40</td><td>2.78</td><td>1.96</td><td>0.015</td></tr><tr><td>Feature patch last</td><td>4.99</td><td>5.71</td><td>6.39</td><td>7.25</td><td>8.01</td><td>8.25</td><td>8.72</td><td>8.90</td><td>9.06</td><td>0.015</td></tr><tr><td>Stereo network, source views</td><td>1</td><td>2</td><td>3</td><td>4</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Observing views first</td><td>8.71</td><td>8.61</td><td>9.73</td><td>10.17</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>All subsets of the size</td><td>13.65</td><td>10.97</td><td>10.14</td><td>10.17</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Non-observing views first</td><td>15.70</td><td>11.04</td><td>10.81</td><td>10.17</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Kriging learner, support cars</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Highest own valuation first</td><td>-0.43</td><td>-1.61</td><td>-1.78</td><td>-1.88</td><td>-1.95</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random order</td><td>0.05</td><td>-0.81</td><td>-1.41</td><td>-1.82</td><td>-1.95</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Every certified call agrees with the measured winner. Of the 64 distinct comparisons between a cache and a fine-tune (for $N \leq 8$ both windows hold the whole context and give the same cache), 25 are certified, and every certified call agrees with the measured winner on the same tasks (Table 9). The cache’s gain stops growing once the context passes the window, at 4.1–4.8 bits for $W = 8$ and 6.8–7.3 bits for $W = 1 6$ , within 0.9 bits of its ceiling $I ( Q ; D _ { w } ) + \mathrm { K L } _ { 0 }$ (Figure 10a). The fine-tune’s gain keeps rising with the data: it rises from 0.9 bits at $N = 3 2$ to 10.7 bits at $N = 1 { , } 0 2 4$ , against a ceiling of 11.5 bits for any channel. It beats the cache from $N = 1 2 8$ with the window of 8 and from $N = 2 5 6$ with the window of 16, and at those data sizes and every larger one its gain passes the cache’s ceiling, which certifies fine-tuning against every cache with that window. In both cases the full fine-tune first wins at sixteen times the window, after a tie at eight times: the larger window leaves less evicted information for fine-tuning to exploit.

When the context fits. For $N \leq W$ , every fine-tuning schedule lowered the gain on the tuning tasks, so the tuned recipe leaves the weights unchanged, and the cache wins with 2.7 to 7.0 bits. Part (iii) caps what any fine-tune could have added there at the cache’s decoding loss, estimated at 0.27 to 0.43 bits.

![](images/00fb3af7c94a30512910a77e1429374c7545f2b473295d7f1deca13419a6546c.jpg)

![](images/ed7b947ad4b9e452f9291f4eb341b64dff3ef1d196aede6bcee46a01d76d916f.jpg)  
Figure 10. Choosing between the cache and fine-tuning on a trained transformer, with exact information terms. (a) Gain on a 16-example query against the number of context examples. Dotted lines are the ceilings $I ( Q ; D _ { w } ) + \mathrm { K L } _ { 0 }$ of the two caches and the dashed line is the ceiling $\bar { I } ( Q ; D ) + \mathrm { K L } _ { 0 }$ of any channel; the bars mark the data sizes at which the fine-tune’s gain passes the cache’s ceiling, which certifies fine-tuning. (b) Gain of �-bit adapters at $N = 2 5 6$ against the bits they store, with their ceilings min $\{ I ( Q ; D ) , C _ { B } \} + \mathrm { K L } _ { 0 }$ and the cache with a window of 16; rings mark the adapters against which the cache is certified. Intervals are 95% over 64 tasks

Table 9. Certified and measured winners across data sizes: nine for the full fine-tune (� from 4 to 1,024) and seven for the adapters (� from 4 to 256). The columns ‘cache’ and ‘tune’ count the data sizes at which the cache certificate and the fine-tune certificate of Proposition 6 fire; ‘won’ gives the data sizes won by the cache, won by the fine-tune and tied. For $N \leq 8$ both windows hold the whole context, so those comparisons appear under both; all 25 distinct certified calls agree with the measured winner.
<table><tr><td></td><td colspan="4">Window 8</td><td colspan="3">Window 16</td></tr><tr><td>Fine-tune</td><td> $C _ { B }$ </td><td>cache</td><td>tune</td><td>won</td><td>cache</td><td>tune</td><td>won</td></tr><tr><td>Full</td><td>∞</td><td>0</td><td>4</td><td>4/4/1</td><td>0</td><td>3</td><td>5/3/1</td></tr><tr><td>Adapter  $2 \times 1$ </td><td>2</td><td>6</td><td>0</td><td>7/0/0</td><td>6</td><td>0</td><td>7/0/0</td></tr><tr><td>Adapter  $4 \times 1$ </td><td>4</td><td>0</td><td>0</td><td>7/0/0</td><td>5</td><td>0</td><td>7/0/0</td></tr><tr><td>Adapter  $8 \times 1$ </td><td>8</td><td>0</td><td>0</td><td>6/0/1</td><td>0</td><td>0</td><td>7/0/0</td></tr><tr><td>Adapter  $8 \times 2$ </td><td>16</td><td>0</td><td>2</td><td>2/4/1</td><td>0</td><td>0</td><td>7/0/0</td></tr></table>

A fine-tune that stores few bits never catches up. Each adapter’s gain stays below its ceiling min $\{ I ( Q ; D ) , C _ { B } \} + \mathrm { K L } _ { 0 } \colon$ at $N = 2 5 6$ the adapters storing 2, 4, 8 and 16 bits gain 1.1, 1.9, 3.6 and 5.8 bits (Figure 10b). The cache with the window of 16 saves 6.8–7.3 bits, more than the 2-bit and 4-bit adapters can store with the prior repair of 0.37 bits added, so part (iv) says they lose at every $N \geq 1 6 ;$ they lose at every data size tested, and the certificate fires at every $N \geq$ 16 for both. The 16-bit adapter, whose ceiling exceeds the cache’s gain, beats the cache with the window of 8 from $N = 3 2 \cdot$ ; its gain exceeds that cache’s ceiling from $N = 3 2 { \mathrm { ~ o n } }$ , and the fine-tune certificate fires at $N = 6 4$ and $N = 2 5 6$

## C.4. The choice across domains in full

Table 10 lists every run behind Section 6.2. On both tables TabPFN encodes the held-out labels in fewer bits than XGBoost in every seed, and the test calls it in every seed and in all 200 orders. On airfoil pressure the trained surrogate encodes the fields in fewer bits in every seed, and the test calls it after 53 to 66 queries. The original pair of airfoil predictors, fitted with 149, 149 and 145 calibration designs before the fresh cohort was drawn, is called the same way after 70 to 122 queries. Table 11 gives the learning curves of Figure 4a with the call at each size, and Table 12 the simulated streams of Figure 4c.

## C.5. The measurable tools on GPT-2

Section 5 reads both channels without the true distribution. We run its three measurable tools on GPT-2, the pretrained 124Mparameter model of Radford et al. (2019), with the original weights (ONNX Model Zoo contributors, 2020) (Appendix D.6).

Table 10. The anytime-valid choice across domains, run by run. Bits are the mean held-out code length per label (tables) or per pressure value (airfoil) without mixing; accuracy is for the tables only. Call and queries are for each seed’s designated query order; the last column counts the 200 random orders of the same cohort that reach the same call.
<table><tr><td>Comparison</td><td>Seed</td><td>In context</td><td>Trained</td><td>Accuracy (%)</td><td>Call</td><td>Queries</td><td>Orders</td></tr><tr><td rowspan="3">Phoneme: TabPFN vs XGBoost</td><td>0</td><td>0.587</td><td>0.761</td><td>78.2 / 78.3</td><td>in context</td><td>348</td><td>200/200</td></tr><tr><td>1</td><td>0.605</td><td>0.691</td><td>76.6 / 78.9</td><td>in context</td><td>1,390</td><td>200/200</td></tr><tr><td>2</td><td>0.640</td><td>0.965</td><td>77.9 /78.2</td><td>in context</td><td>151</td><td>200/200</td></tr><tr><td rowspan="3">Spambase: TabPFN vs XGBoost</td><td>0</td><td>0.379</td><td>0.716</td><td>91.5 / 85.8</td><td>in context</td><td>198</td><td>200/200</td></tr><tr><td>1</td><td>0.299</td><td>0.501</td><td>92.7 / 88.7</td><td>in context</td><td>247</td><td>200/200</td></tr><tr><td>2</td><td>0.318</td><td>0.454</td><td>91.7 / 88.1</td><td>in context</td><td>358</td><td>200/200</td></tr><tr><td rowspan="3">Airfoil: support set vs MLP</td><td>0</td><td>3.654</td><td>2.412</td><td></td><td>trained</td><td>62</td><td>200/200</td></tr><tr><td>1</td><td>3.623</td><td>2.384</td><td></td><td>trained</td><td>66</td><td>200/200</td></tr><tr><td>2</td><td>3.670</td><td>2.227</td><td></td><td>trained</td><td>53</td><td>200/200</td></tr><tr><td rowspan="3">Airfoil, original predictors</td><td>0</td><td>3.787</td><td>3.063</td><td></td><td>trained</td><td>122</td><td>200/200</td></tr><tr><td>1</td><td>3.654</td><td>2.399</td><td></td><td>trained</td><td>71</td><td>200/200</td></tr><tr><td>2</td><td>3.705</td><td>2.566</td><td></td><td>trained</td><td>70</td><td>200/200</td></tr></table>

Table 11. Airfoil pressure against the number of designs in the support set (in context) or the training set (trained), with the calibration designs and the 254 fresh queries fixed in each seed. Bits are the mean held-out code length per pressure value over three seeds; the last column gives the anytime-valid call in each seed’s designated order (seeds 0, 1, 2). The trained surrogate’s recipe stops early on a validation split, which needs at least 16 designs.
<table><tr><td>Designs</td><td>In context</td><td>Trained</td><td>Call in seeds 0, 1, 2</td></tr><tr><td>4</td><td>4.964</td><td></td><td></td></tr><tr><td>8</td><td>4.718</td><td></td><td></td></tr><tr><td>16</td><td>4.583</td><td>4.746</td><td>none, context, none</td></tr><tr><td>32</td><td>4.258</td><td>4.292</td><td>trained, context, context</td></tr><tr><td>64</td><td>3.951</td><td>3.573</td><td>trained, context, trained</td></tr><tr><td>128</td><td>3.649</td><td>2.341</td><td>trained, trained, trained</td></tr><tr><td>256</td><td>3.458</td><td>2.382</td><td>trained, trained, trained</td></tr><tr><td>512</td><td>3.298</td><td>1.516</td><td>trained, trained, trained</td></tr></table>

Learning curves from the context. The units are 64-token passages of Shakespeare (Karpathy, 2015), split at random into a context pool and a held-out pool. The cache is GPT-2 reading the previous units as separate passages in its 1,024-token window, which holds the last 14; the online fine-tune takes one Adam step on each unit as it arrives. Over four streams of 256 units, the cache’s increments are read by teacher-forcing each unit after the units in its window, and the fine-tune’s are its losses before each step (Proposition 7). In all eight size bins and for both channels, the increments agree with the held-out learning curves within the sampling error of the per-unit means, at most 1.97 standard errors apart; from 14 units on they lie within 0.16 bits per token of the curves. Their sums, the prequential code lengths, come within 0.1% for the cache and 1.5% for the fine-tune of the sums of the held-out curves (Figure 11a). Both readings make the same call in every bin. The cache levels off at 6.8 to 6.9 bits per token from 8 units on, while the fine-tune keeps improving: on the mean curves it leads at every size, by 0.16 bits per token after one unit and by 1.28 after 256, where it reaches 5.49.

An anytime-valid choice. The test of Theorem 8 runs in both directions on one stream of held-out queries, each direction at level 0.025, and the first to reach 40 makes the call (Appendix D.6). On Shakespeare the online fine-tune is the better memory with 4 context units and with 256, and the test calls it in all 16,000 query streams, after a median of 205–324 and 81–103 queries (Table 13). In document recall a query is a span of a passage, cued by its first 8 tokens and scored on the next 24, and GPT-2 copies the passage from its window. On passages of 512 tokens the cache needs 0.62–1.92 bits per token, against 2.91–6.64 for fine-tunes of one and four epochs, and the test calls the cache after a median of 25–69 queries; a fine-tune of sixteen epochs memorises the passage, needs 0.23–0.36 bits per token, and the test calls it after 70–300. On passages of 4,096 tokens the window holds the last 960, and fine-tunes of four and sixteen epochs win and are called after 25–53 queries. After one epoch the fine-tune and the cache lie within 0.76 bits per token of each other on these passages: the test calls the better memory in 5,948 of 8,000 streams and leaves 2,048 undecided after 1,000 queries, 2,046 of them in the two comparisons separated by 0.17 and 0.21 bits per token. Over all 64,000 query streams, 4 call the losing memory, all in the comparison separated by 0.21 bits per token and within the 2.5% error the test allows there, and on increments centred at zero the procedure reaches its threshold in at most 0.66% of 10,000 streams for any of the 32 comparisons (Figure 11b)

Table 12. The anytime-valid test on 10,000 simulated streams per case, each of up to 4,096 queries, with the settings of the real comparisons and the bound of the table comparisons, $c = \log _ { 2 } ( 2 / \varepsilon )$ . Advantage is the mean score difference in bits per query, with independent ±2-bit noise for the nonzero cases; the nulls have mean zero. Wrong calls are calls for a side that is not better, with the upper end of a 95% Wilson interval; correct calls are shares of all streams by the given number of queries.
<table><tr><td>Case</td><td>Wrong calls</td><td>95% upper</td><td colspan="3">Correct calls by 256 / 1,024 / 4,096</td></tr><tr><td>Exact tie</td><td>0.00%</td><td>0.04%</td><td></td><td></td><td></td></tr><tr><td>Symmetric null</td><td>3.25%</td><td>3.62%</td><td></td><td></td><td></td></tr><tr><td>Skewed null</td><td>2.87%</td><td>3.22%</td><td></td><td></td><td></td></tr><tr><td>Null of changing variance</td><td>3.31%</td><td>3.68%</td><td></td><td></td><td></td></tr><tr><td>Advantage 0.5</td><td>0.00%</td><td>0.04%</td><td>88.1%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>Advantage 0.1</td><td>0.17%</td><td>0.27%</td><td>3.7%</td><td>12.3%</td><td>54.9%</td></tr><tr><td>Advantage 0.03</td><td>0.74%</td><td>0.93%</td><td>1.1%</td><td>2.2%</td><td>4.5%</td></tr><tr><td>Advantage 0.005</td><td>1.21%</td><td>1.44%</td><td>0.9%</td><td>1.3%</td><td>1.8%</td></tr><tr><td>Advantage -0.1</td><td>0.16%</td><td>0.26%</td><td>4.0%</td><td>12.3%</td><td>54.7%</td></tr></table>

![](images/93db29292320e5605a8a03c86b58161c05d4bcd3d4ff0fcc1cdf39e73a876fbe.jpg)  
Units absorbed m (64 tokens each)

![](images/75bc25ae56dd1f0b76947f06573a5f710d5e04067c932b9de1e6c36b9b5b1217.jpg)

(c) storage budget  
![](images/e33c4a5b634c535c1895eb3da5d8a70be1a665cc05ce13c365941068b86669c2.jpg)  
Bits the replacement stores, H(S)  
Figure 11. The measurable tools of Section 5 on GPT-2. (a) Learning curves of the cache and the online fine-tune on Shakespeare passages: code length on held-out passages (circles) and the prequential increments read from the context itself, binned over sizes (diamonds); the dotted line marks the full window. Intervals are ±1.96 standard errors of the per-unit means, pooled over four streams. (b) log $E _ { t }$ of the test that calls the winner, median over 2,000 query streams for each context (thin lines) and the median over contexts (thick lines): recall of 512-token passages against a one-epoch fine-tune, where the cache wins, and Shakespeare with 256 and 4 context units, where the fine-tune wins; the dashed line is the threshold log(2/�). (c) Total divergence from the cache over the four-key workload against the bits a compiled replacement stores, for the replacement’s Bayes decoder and for GPT-2 reading it; the dashed line is the budget of Theorem 9.

A lower bound on GPT-2’s decoding loss. Four units fit the window, and the cache keeps them (Proposition 5), so by Proposition 6(iii), applied to each context as in Appendix B.7, the online fine-tune’s advantage over the cache on each context is a lower bound on the cache’s decoding loss there. In each of the four streams the fine-tune beats the cache by 0.35 to 0.55 bits per token on held-out passages, with standard errors below 0.03: on each of these contexts GPT-2’s prediction falls at least that far short of the best prediction from its prompt. The held-out passages are drawn independently of those in context, so beyond the task the context carries no information about them, and the shortfall is prior repair that the cache leaves unrealised.

The storage budget. The context is a table of four facts, each a fixed key of three tokens followed by one of eight candidate tokens, so it ranges over $8 ^ { 4 } = 4 { , } 0 9 6$ tables and holds 12 bits. The workload is the four keys, and an output is GPT-2’s next token after a key, read as one of the candidates or as none of them. From GPT-2’s own in-context outputs over all 4,096 tables, $I ( \hat { Y } _ { 1 : 4 } ; \hat { D ) } = 5 .$ .14 bits: the cache carries 43% of the table into its outputs. Equation (22) with $M = 8 , 3 2$ , 128 and 512 tables drawn at random gives 2.57, 3.75, 4.52 and 4.93 bits, each below $\log _ { 2 } M$ , and 5.14 bits with all 4,096. A compiled replacement of the cache lists the first � facts with their words restricted to the first $2 ^ { b }$ candidates, and each table takes the state that GPT-2 reads closest to its cache, so the replacement stores �(�) bits about the table. The 13 replacements store from 0 to 12 bits, and the eight that store less than 5.14 bits all miss the cache by at least $5 . 1 4 - H ( S )$ bits on the workload (Figure 11c). The divergence splits as in the proof of Theorem 9. With the Bayes decoder of the state, which leaves only the storage term, the replacement that stores nothing misses by 5.18 bits against a budget of 5.14, and the three that list one fact stay within 0.52 bits of the budget. GPT-2 reading the same compiled prefix adds 3.6 to 25.6 bits of decoding loss to every

Table 13. The anytime-valid choice on GPT-2. Code lengths are in bits per token on each task’s pool of held-out queries, ranges over the four contexts (streams or passages). ‘Calls’ counts the query streams, 2,000 per context, in which the test called the better memory, and ‘queries’ gives the median number of held-out queries to the call over the streams that called, range over contexts. In the 4,096-token, one-epoch row the fine-tune is better on three passages and the cache on one; there 2,048 streams are still undecided after 1,000 queries and 4 call the losing memory. No other stream calls the losing memory.
<table><tr><td></td><td></td><td></td><td colspan="2">Code length</td><td></td><td></td><td></td></tr><tr><td>Task</td><td>Context</td><td>Recipe</td><td>Cache</td><td>Fine-tune</td><td>Better memory</td><td>Calls</td><td>Queries</td></tr><tr><td>Shakespeare</td><td>4 units</td><td>online</td><td>6.98-7.23</td><td>6.59–6.68</td><td>fine-tune</td><td>8,000/8,000</td><td>205-324</td></tr><tr><td>Shakespeare</td><td>256 units</td><td>online</td><td>6.80-7.13</td><td>5.68–5.70</td><td>fine-tune</td><td>8,000/8,000</td><td>81-103</td></tr><tr><td>Recall</td><td>512 tokens</td><td>1 epoch</td><td>0.62-1.92</td><td>5.44–6.64</td><td>cache</td><td>8,000/8,000</td><td>25-29</td></tr><tr><td>Recall</td><td>512 tokens</td><td>4 epochs</td><td>0.62-1.92</td><td>2.91-4.46</td><td>cache</td><td>8,000/8,000</td><td>46-69</td></tr><tr><td>Recall</td><td>512 tokens</td><td>16 epochs</td><td>0.62–1.92</td><td>0.23-0.36</td><td>fine-tune</td><td>8,000/8,000</td><td>70-300</td></tr><tr><td>Recall</td><td>4,096 tokens</td><td>1 epoch</td><td>5.16–5.52</td><td>4.70-5.32</td><td>fine-tune (3), cache (1)</td><td>5,948/8,000</td><td>165-668</td></tr><tr><td>Recall</td><td>4,096 tokens</td><td>4 epochs</td><td>5.16–5.52</td><td>2.44–2.96</td><td>fine-tune</td><td>8,000/8,000</td><td>41-53</td></tr><tr><td>Recall</td><td>4,096 tokens</td><td>16 epochs</td><td>5.16–5.52</td><td>0.30–0.39</td><td>fine-tune</td><td>8,000/8,000</td><td>25-26</td></tr></table>

replacement short of the full table.

## C.6. Certificates on pretrained models outside language

Proposition 6 is tested on three pretrained models outside language, each on a designed distribution of contexts for which the information terms are exact. For any channel $c , \mathbb { E } [ \Gamma _ { c } ] - \mathrm { \bar { K L _ { 0 } } } = H ( \mathscr { Q } ) - \mathbb { E } [ \ell _ { c } ] $ , so the prior repair cancels from both certificates of part (ii): a memory beats every context memory confined to the window $D _ { w }$ when its code length falls below $H ( Q \mid D _ { w } )$ , and the in-context model beats every memory that stores at most � bits about the context, any fine-tune, adapter or compiled state of that size, when its code length falls below $H ( Q ) - C$ . Each model runs 64 episodes, every predictor is mixed with $\varepsilon = 0 . 0 1$ of the uniform law, a certificate is issued when the 95% interval of its paired margin lies above zero, and each issued certificate is checked against the measured winner of every implemented pair it covers, as in Appendix C.3. Of 96 certificates tested, 36 are issued, and all 86 checks agree with the measured winner (Table 14).

Table 14. Certificates on pretrained models outside language, 64 episodes per model: certificates issued out of those tested, against a window and against a capacity, and checks of the issued certificates against the measured winners of the implemented pairs they cover.
<table><tr><td>Model</td><td>Window</td><td>Capacity</td><td>Checks</td><td>Agree</td><td>Disagree</td></tr><tr><td>TabPFN-v2</td><td>8/30</td><td>17/48</td><td>50</td><td>50</td><td>0</td></tr><tr><td>Occupancy network</td><td>2/5</td><td>3/6</td><td>14</td><td>14</td><td>0</td></tr><tr><td>MVSNet</td><td>2/3</td><td>4/4</td><td>22</td><td>22</td><td>0</td></tr><tr><td>All</td><td>12/38</td><td>24/58</td><td>86</td><td>86</td><td>0</td></tr></table>

TabPFN. Tasks are drawn uniformly from a pool of 4,096 logistic classifiers on four Gaussian features, with weights of standard deviation 3 and biases of standard deviation 1. A context holds $N = 8$ to 1,024 labelled rows of one task, nested across sizes, and a query is a block of 64 further rows. The pool makes the true posterior predictive a finite mixture, so $H ( Q )$ $H ( Q \mid D )$ and $H ( Q \mid D _ { w } )$ are exact in each episode, and $I ( Q ; D )$ rises from 3.4 bits at $N = 8$ to 9.5 bits at $N = 1 { , } 0 2 4$ TabPFN-v2 reads all rows in its context, or only the last 16 or 64; logistic regression and XGBoost train on all rows; and the bounded memories are logistic regressions whose weights and bias lie on a grid of $2 ^ { b }$ values, fitted over the whole grid, which store at most 5� bits for $b \overset { \_ } { = } 1 , 2$ , and the best of $2 ^ { k }$ fixed classifiers by likelihood, which stores at most � bits for $k = 2$ to 10. With 1,024 rows TabPFN codes a block within 0.67 bits of the true posterior predictive. The certificate for learning in context fires from $N = 6 4 \mathrm { o n }$ , and at $N = 1 { , } 0 2 4$ it covers every memory of at most 8 bits. Logistic regression, and TabPFN reading all rows, are certified against every memory confined to the last 16 rows from $N = 1 2 { \bar { 8 } }$ on. Against the last 64 rows, whose ceiling the trained models’ code lengths reach at 512 and 1,024 rows, no certificate is issued.

Occupancy network. A procedural chair carries six bumps, each present or absent with probability $1 / 2$ . The context is a point cloud read as items in a random order: 300 anchor points on the shared geometry, always read; one patch of 300 points around each bump, which lies on the bump if it is present and on the flat surface beneath it if not; and six patches of 300 points from the shared geometry. The query is the occupancy of one site inside each bump, averaged over 32 sites per bump, so $H ( Q ) = 6$ bits, $\dot { H ( Q \mid D ) = 0 }$ , and $\hat { H ( Q \mid D _ { w } ) }$ counts the bumps whose patch lies outside the first � items. The pretrained occupancy network, a three-plane model for partial point clouds, reads every item and codes the six sites in 2.72 bits, so it is certified against every memory that stores at most 2 bits about the shape, and against every memory confined to the first 2 or 4 items, whose ceilings are 5.00 and 3.97 bits. Reading its anchors alone, the same network codes the sites in 19.4 bits against a ceiling of 6: where no patch observes a bump it predicts with confidence, and the excess is decoding loss.

Stereo network. A textured card faces the reference camera at one of 16 depths, drawn uniformly, in front of a textured background at a fixed depth. Textures are defined in reference-image coordinates, so the reference image is the same at every depth. One of four source views observes the card; the other three are turned away so that the card lies outside their frame at every designed depth, and the views are read in a random order. The query is which of the 16 depths the card lies at, read from the network’s law over its 192 hypotheses at pixels inside the card, so $H ( Q ) = 4 \mathrm { b i t s } , H ( \bar { Q } \mid D ) = 0 .$ , and $H ( Q \mid D _ { w } ) = 4$ bits unless the observing view is among the first � read. MVSNet with the checkpoint of Appendix D.2, reading all views, codes the depth in 0.67 bits, so it is certified against every memory that stores at most 3 bits about the scene, and against every memory confined to the first one or two views.

Bounded memories of the two networks. The certificate for learning in context covers every memory of at most � bits. For the two networks it is checked against compiled states, which store � of the bumps or the top � bits of the depth index and are read by their Bayes decoder and by the frozen network, given the stored bumps’ patches or a depth range narrowed to the stored bits.

## D. Experimental protocols

## D.1. Geometry: local features and natural objects

Controlled feature construction. The preparation seed is 20261017. Eight independently seeded parametric chair pairs differ by a raised seat feature versus a recess of matched extent, with unchanged surrounding geometry and a common frame. Sampling starts from 100,000 area-uniform surface samples, applies the partial-crop policy, draws 3,000 points with replacement and adds Gaussian noise of standard deviation 0.005. Sampling allocations and random draws are paired wherever the geometry is shared, and the deletion units are built before the model is evaluated.

A pair is eligible when both distinguishing surfaces survive the crop, the changed observations fit within the 300-point item, and at least 300 anchor observations remain outside the feature plus a 0.01 safety band; eligibility is decided from geometry alone. Three site populations are fixed from geometry: 1,024 feature-positive sites, 1,024 unchanged occupied sites and 1,024 unchanged free sites. The main text reports the feature-positive population, and all three are in geometry\_feature.csv. The mesh ambiguity tolerance is $1 0 ^ { - 5 }$

The relevant deletion removes the distinguishing observations, while the coherent absent arm supplies the observed recess floor and walls. Under the relevant deletion the paired remaining inputs agree exactly. The pretrained model is used without fine-tuning.

Donor replacement. The cohort contains two objects in each of eight ShapeNet categories. Donor replacement keeps 300 target anchor points and substitutes the optional observations of another object in the same category. The 8,492 claims are those that the target’s optional observations lift above 0.95, defined from the target’s own predictions, so donor confidence never enters their selection. Donor-mesh agreement uses the same geometric predicate as the claim. Table 15 gives pooled and equal-object rates.

Coherent-object replacement. This protocol replaces the entire observation set, anchors included. Query strata (target and donor interior disagreement, exterior, and near-boundary) are sampled before prediction, with a boundary band of 0.01 in the normalized frame and query locations in [−0.55, 0.55]<sup>3</sup>. Mesh occupancy uses validated distance checks and paired ray-parity checks, with boundary ambiguity as a separate state. The protocol requests 1,024 queries per stratum, subject to geometric eligibility, and measures occupancy at fixed coordinates where the two shapes disagree. Each target is compared with every eligible donor among the other 15 objects (237 comparisons).

Table 15. Donor replacement by the donor mesh’s relation to the fixed claim. Equal-object rates differ from pooled rates because objects contribute different numbers of claims.
<table><tr><td>Donor mesh relation</td><td>Claims</td><td>Still at 0.95</td><td>Pooled rate</td><td>Equal-object rate</td></tr><tr><td>Agrees with fixed claim</td><td>5,902</td><td>5,305</td><td>89.88%</td><td>78.34%</td></tr><tr><td>Disagrees with fixed claim</td><td>2,589</td><td>64</td><td>2.47%</td><td>2.30%</td></tr><tr><td>Boundary ambiguous</td><td>1</td><td>0</td><td>0%</td><td>0%</td></tr></table>

## D.2. Multi-view preprocessing, claims, and visibility

The native 3D-CNN MVSNet checkpoint is model.ckpt-150000. Preprocessing uses maximum width 1,152 and height 864 with aspect-preserving scaling, cropping and matching camera updates, and the memory-efficient native inference path exposes the categorical law on 192 depth hypotheses. The reference image and camera are present in every condition, the four source views follow the original pairing order, and the physical depth hypotheses are identical across deletions.

Scene 9 was used to develop the protocol. Scenes 4, 48, 11 and 29 were drawn with seed 20261017 from the MVSNet tes scenes. The first five eligible reference views per scene and their first four distinct ranked source views are fixed without reference to predictions, and each reference starts from 4,096 seeded pixel queries sampled after dense inference.

The depth claim is an interval centred at the full-context expected depth with half-width five depth units. A reference depth more than one unit inside the interval defines the oracle-inside stratum, more than one unit outside defines oracle-outside, and the remaining cases form a boundary stratum; invalid reference depths are counted separately. The same intervals are used under every deletion.

Visibility projects the reference surface point into each source and compares the independently provided source depth, with absolute tolerance one depth unit, relative tolerance 0.002 and reprojection tolerance one original-image pixel. The categories are observing, occluded, outside or behind, and unknown. The observing and non-observing deletions take the lowest original slot index within their category, and unknown views are never treated as non-observing. The paired oracle-inside cohort contains 11,808 query locations across 25 references.

Means are computed within reference and stratum, then over references within scene, then over scenes, and crossing counts use the same paired cohorts. Table 16 reports every scene-level contrast, and the bootstrap resamples scenes. The evaluated pipeline matches the native MVSNet implementation exactly in 12 preprocessing, forward-pass and condition comparisons and in two further regression comparisons on scene 4.

Table 16. Extra claim-probability loss from deleting an observing rather than a non-observing view, in percentage points. Five references contribute equally within each scene.
<table><tr><td>Scene</td><td>Oracle inside</td><td>Oracle outside</td><td>Boundary</td></tr><tr><td>9 (protocol development)</td><td>3.3232</td><td>0.7735</td><td>1.9261</td></tr><tr><td>4</td><td>3.9743</td><td>-0.7445</td><td>6.3977</td></tr><tr><td>48</td><td>13.8833</td><td>4.1997</td><td>7.1883</td></tr><tr><td>11</td><td>10.6232</td><td>3.7646</td><td>4.4055</td></tr><tr><td>29</td><td>15.7790</td><td>2.5745</td><td>11.3640</td></tr><tr><td>Mean, all five scenes</td><td>9.5166</td><td>2.1136</td><td>6.2563</td></tr><tr><td>Mean, scenes 4, 48, 11 and 29</td><td>11.0650</td><td>2.4486</td><td>7.3389</td></tr></table>

## D.3. Kriging learner and native inference

The learner receives each support car as points, normals, positive area weights and surface pressure coefficients; the query supplies geometry only. Chart anchors are farthest points of the support union, and local descriptors contain kernel-mean offsets and normals. Centred Gram systems give affine kriging weights summing to one, followed by chart smoothing, and kernel transport uses position and normal agreement to carry support pressures to the query. Descriptor scale, transport scale, ridge and smoothing are selected by leave-one-car-out prediction on the support set, so the learner is a fixed inference procedure with context-dependent fitted state.

The spread transports each support car’s leave-one-out squared-residual field to the query and averages these fields, with a floor of 0.05 times the median squared leave-one-out error and a small positive safeguard. The result is an explicit error-based Gaussian predictive law; any Gaussian predictive law, including a conventional GP posterior with fixed hyperparameters, enters the evaluator in the same way.

Each query has 32,000 prepared geometry points and 512 fixed scoring sites. Query geometry enters the local descriptors and the coarsest argument that sets descriptor bandwidth, so subsampling the geometry before inference changes the predictions even at the scored sites. The fidelity grid crosses two geometry sizes (512 and 32,000), two chart-anchor counts (128 and 1,024) and two leave-one-out selection sizes (512 and 4,000), and two further cells change the supplied geometry and the bandwidth argument separately. Support names and family order are fixed because the learner’s selection rule uses name-seeded sampling.

Two claim banks are fixed in advance and held across every inference setting and deletion: the native-answer bank fixes the native full-context mean at each site, and the reduced-input bank fixes the reduced-input mean. Both use pressure tolerance 0.05 and target 0.95. Under native inference, the native-answer claim has full-context probability $2 \Phi ( 0 . 0 5 / s ) - 1$ , where � is the learner’s predictive spread at the site. The gain does not depend on the bank, while claim-level bits and crossings do (Table 17).

Table 17. Field scores for both claim banks. Means weight sites by area within a query and weight queries and single-car deletions equally. Crossing fractions are percentages of area-weighted site observations.
<table><tr><td>Inference</td><td>Claim bank</td><td> $p$ </td><td>q</td><td>Gain  $I _ { G }$ </td><td>Claim bits J</td><td>Crossing</td><td>Reverse</td></tr><tr><td>Reduced input</td><td>Reduced-input answer</td><td>0.5761</td><td>0.5312</td><td>0.7939</td><td>0.1108</td><td>4.6045%</td><td>1.6273%</td></tr><tr><td>Reduced input</td><td>Native answer</td><td>0.4623</td><td>0.4806</td><td>0.7939</td><td>0.1056</td><td>1.4881%</td><td>2.6273%</td></tr><tr><td>Native</td><td>Reduced-input answer</td><td>0.4629</td><td>0.4584</td><td>0.2296</td><td>0.0382</td><td>1.2688%</td><td>1.3699%</td></tr><tr><td>Native</td><td>Native answer</td><td>0.6131</td><td>0.5895</td><td>0.2296</td><td>0.0419</td><td>4.2862%</td><td>2.2225%</td></tr></table>

Reference pressure enters only the error,

$$
\mathrm { r e l a t i v e } L _ { 2 } = \sqrt { \frac { \sum _ { j } w _ { j } ( \hat { y } _ { j } - y _ { j } ^ { \mathrm { r e f } } ) ^ { 2 } } { \sum _ { j } w _ { j } ( y _ { j } ^ { \mathrm { r e f } } ) ^ { 2 } } } .\tag{30}
$$

The reported mean is an equal-query mean of this ratio, and the query car is the unit of replication. Native inference improves all eight cars. The evaluated learner reproduces the original implementation’s means, variances and selected configurations exactly in 168 comparisons. The learner needs at least two support cars, so every deletion leaves at least two.

## D.4. Language adjudication

The adjudicator is the Trace Budget implementation (Chlon, 2026), run with Qwen2.5-7B-Instruct and with GPT-4.1-mini (version 2025-04-14). Temperature is zero, the top-label log-probability request is five, and the target is 0.95. Claim normalization, budget-gap extraction and the decision policy run unchanged, and the flag is the adjudicator’s own decision rather than a threshold fitted to these data.

The score is the budget gap in bits, required minus supplied, so larger values indicate less support. The adjudicator clips probabilities to $[ 1 0 ^ { - 1 2 } , 1 - 1 0 ^ { - 1 2 } ]$ , which bounds the gap at $d _ { 2 } ( 0 . 9 5 \rVert 1 0 ^ { - 1 2 } ) = 3 7 . 5 8$ bits; 47.0% of Qwen scores and 51.5% of GPT-4.1-mini scores lie within 0.1 bits of this bound, and AUROC uses the full score ordering with ties counted as half a pair.

The evaluation set is the first 800 VitaminC examples selected by the benchmark pipeline: 395 SUPPORTS, 120 NEI and 285 REFUTES. The 390 claim groups contain the same claim with supporting evidence and with NEI or refuting evidence; the Qwen score orders 370 of them correctly with no ties, and the GPT-4.1-mini score orders 371 correctly with one tie.

Three GPT-4.1-mini requests (examples 762–764; two NEI and one SUPPORTS) returned content-filter errors and count as flagged. On the 797 successful examples, sufficiency AUROC is 0.9152 (394 SUPPORTS and 118 NEI) and full-grounding AUROC is 0.9318.

## D.5. Decision test

In-context learner. Tasks follow Appendix A.10 with $d = 8 , \bar { \tau } ^ { 2 } = 4 { \mathrm { ~ a n d ~ } } \sigma ^ { 2 } = 1$ . The learner is a pre-norm causal transformer with three layers, width 64, four heads, MLP width 128 and learned positions, 104,194 parameters in all. Token � is $[ x _ { i } , x _ { i - 1 } , y _ { i - 1 } , \mathbf { 1 } \{ i > \mathrm { ~ \dot { 1 } ~ } \} ]$ , and the head at position � returns the mean and log-variance of a Gaussian predictive for $y _ { i } ,$ so position � sees the � − 1 preceding examples. Training draws 20,000 batches of 64 sequences of 32 examples from the prior and minimises the prequential Gaussian negative log-likelihood with AdamW (peak learning rate $1 0 ^ { - 3 }$ , 1,000 warm-up steps, cosine decay, weight decay $1 0 ^ { - 4 }$ , gradient clipping at 1). On 256 validation sequences its code length exceeds that of the Bayes posterior predictive by 0.025 bits per label on average.

Channels. The base reads the 16-example query block alone. The cache reads the last min $\{ N , W \}$ context examples and then the block, and only the block’s labels are scored. The full fine-tune trains all weights with Adam on $\ell ( D )$ , computed at every step on a fresh random split of the context into sequences of at most 16 examples. Its learning rate and number of steps are chosen for each � to maximise the mean gain on 32 tuning tasks, from learning rates $1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 3 \times 1 0 ^ { \frac { 1 } { - 3 } }$ and $1 0 ^ { - 2 }$ (with $3 \times 1 0 ^ { - 5 }$ added for $N \in \{ 1 6 , 3 2 \}$ and $3 \times 1 0 ^ { - 2 }$ for $N \geq 3 2 )$ and from 0, 5, 10, 20, 40 and 80 steps (up to 160 at $1 0 ^ { - 4 }$ and 320 at $3 \times 1 0 ^ { - 5 } )$ . Every chosen learning rate lies inside this grid. The chosen recipes leave the weights unchanged for $N \leq 1 6 .$ , take 80 steps at $1 0 ^ { - 4 }$ for $N = 3 \bar { 2 }$ and 40 steps at $1 0 ^ { - 2 } \mathrm { f o r } N = 6 4$ , and take 40 steps at $3 \times \bar { 1 0 } ^ { - 3 }$ beyond. The adapters place � pseudo-examples, with inputs fixed by a seed and trainable labels, in front of every sequence; the labels are trained with Adam (learning rate 0.1, 100 steps) on $\dot { \ell ( D ) }$ and then quantised to � bits on a uniform grid over $[ - R , R ]$ , with � chosen for each � on the tuning tasks from $\{ 1 , 2 , 3 , 4 \}$ , or from {1, 2, 3, 4, 5, 6, 8} for the 16-bit adapter. The quantised labels are the adapter’s only trainable state, so it takes at most $2 ^ { k b }$ values and $I ( D ; Z _ { B } ) \le k b$

Evaluation. The 64 evaluation tasks are shared by every channel and data size; each has a stream of 1,024 context examples and a 16-example query block, and the context of size � is the first � examples of the stream. Given the inputs, $I ( Q ; D )$ and $I ( Q ; D _ { w } )$ are Gaussian log-determinants, $\ell ^ { * } ( Q )$ is a Gaussian log-density, and the prior repair is the mean of $\ell _ { 0 } ( Q ) - \ell ^ { * } ( Q )$ 0.37 bits (95% interval 0.12 to 0.61). The cache certificate uses the per-task margin $\ell ^ { * } ( Q ) { \bar { - } } \ell _ { A } ( Q \mid D ) - \operatorname* { m i n } \{ I ( Q ; D ) , C _ { B } \}$ and the fine-tune certificate uses $\ell ^ { * } ( Q ) - \ell _ { B } ( Q ) - I ( Q ; D _ { w } ) ;$ ; a certificate is issued when the lower end of the 95% interval of the mean margin is positive. The measured winner is the channel whose paired gain difference has a 95% interval that excludes zero, and the comparison is a tie otherwise. Adapters are run for $N \leq 2 5 6$ , where their gains have already levelled off.

## D.6. GPT-2 experiments

Model. GPT-2 has 124,439,808 parameters; we use the OpenAI weights exported in the ONNX Model Zoo and their bytelevel BPE vocabulary. The implementation reproduces GPT-2’s behaviour: 3.99 bits per token on a Wikipedia paragraph, and 0.27 bits per token on the second copy of a repeated sequence of 50 random tokens, after its first token. Fine-tuning updates all weights with Adam.

Shakespeare streams. The text is cut into 5,281 units of 64 tokens and permuted with a fixed seed: 1,024 units form the held-out pool and the rest the context pool. Each of four streams draws 256 context units in random order and 64 held-out units for its learning curves, from a sub-pool of $3 8 4 ; \mathrm { a }$ further 256 held-out units define the task of the anytime-valid test. Both channels read every unit as a separate passage, the end-of-text token followed by its 64 tokens. The cache’s context is the end-of-text token followed by the last 14 units, each closed by an end-of-text token, so the context and a query passage fit the 1,024-token window. The online fine-tune takes one Adam step per unit at learning rate $3 \times 1 0 ^ { - 5 }$ , chosen from $1 0 ^ { - 5 }$ $3 \times 1 0 ^ { - 5 }$ and $1 0 ^ { - 4 }$ by the held-out code length after 64 units of a separate ordering of the context pool, on held-out units disjoint from those reported. Increments are read before each step for the fine-tune and with the window rule for the cache. Held-out learning curves are read at 9 sizes for the cache and 11 for the fine-tune; Figure 11a bins the increments over � in $\{ 1 \} , \{ 2 , 3 \} , \{ 4 , \ldots , 7 \} , \{ 8 , \ldots , 1 3 \} , \{ 1 4 , \ldots , 3 1 \} , \{ 3 2 , \ldots , 6 3 \} , \{ 6 4 , \ldots , 1 2 7 \}$ and {128, . . . , 255}, and compares each bin with the held-out curve, interpolated linearly in � and averaged over the same sizes.

Document recall. Four passages of 512 tokens and four of 4,096 tokens are cut from the text at random positions. A query is a span with a uniformly random start: an 8-token cue followed by 24 tokens, whose code length is the query’s, and each passage has a pool of 256 spans. The cache reads the end-of-text token, the last 960 tokens of the passage and then the span, and the fine-tune reads the end-of-text token and the span. Fine-tuning uses Adam at learning rate $1 0 ^ { - 4 }$ on pieces of at most 256 tokens, cut at a random offset each epoch so that they cover the passage once, two pieces per step, and the pool is scored after 1, 4 and 16 epochs.

Anytime-valid test. A query is drawn uniformly from the task’s pool, so the per-query code lengths of the pool under both memories determine every query stream and the exact winner. The channel under test is mixed with $\varepsilon = 0 . 0 1$ of the uniform law, so $c = L \log _ { 2 } ( | V | / \varepsilon )$ : 1,425 bits for passages $( L = 6 4 )$ and 534 bits for spans $( L = 2 4 )$ . The bet is $\lambda _ { j } = \operatorname* { m i n } \{ 0 . 7 5$ , max $\{ 0 , \hat { m } _ { j - 1 } / ( \hat { \nu } _ { j - 1 } + \hat { m } _ { j - 1 } ^ { 2 } ) \} \}$ , with �ˆ and �ˆ the running mean and variance of the past increments divided by �; the mean is shrunk toward zero by one pseudo-observation, and a pseudo-variance of 0.01 is added. Both directions run on each stream at level 0.025, and the first to reach 40 makes the call; each comparison is summarised over 2,000 streams of at most 1,000 queries. The null error is the fraction of 10,000 streams on which the same procedure, run on the increments centred to mean zero over the pool, with � widened by the absolute mean, reaches 40 within 1,000 queries.

Storage budget. The four keys are fixed phrases of three tokens and the eight candidates are further tokens, all drawn with a fixed seed from the single tokens that consist of a space and at least four lowercase letters. A table assigns a candidate to each key, and its context is the end-of-text token followed by “key word,” for the four keys. An output is GPT-2’s next token after a key, read as one of the eight candidates or as other. Reading the outputs this coarsely can only lower the information and the divergences, so the inequality of Theorem 9 holds for the coarse outputs, and the budget computed from them bounds the full outputs as well. $I ( \hat { Y } _ { 1 : 4 } ; D )$ is exact over the $9 ^ { 4 }$ joint outputs and all 4,096 tables, and Equation (22) is averaged over 20 draws of � tables (5 for $M = 5 1 2 )$ . A replacement with parameters $( k , b ) , 0 \leq k \leq 4$ and $1 \leq b \leq 3 .$ , is the prefix that lists the first � keys with words from the first $2 ^ { b }$ candidates, $2 ^ { k b }$ states in all; each table takes the state with the smallest total divergence from its cache as GPT-2 reads it, and $H ( S )$ is the entropy of the chosen state when the table is uniform. The Bayes decoder of a state returns the average of the cache’s output laws over the tables assigned to it.

## D.7. Tables and airfoil pressure

Both comparisons fix their settings before any query is scored. The level is $\alpha = 0 . 0 5$ per comparison, run as two tests at level $\alpha / 2$ of which the first to reach $2 / \alpha$ makes the call. Both predictors are mixed with $\varepsilon = 0 . 0 1$ of the uniform law over their outcomes, so the test compares the two mixed predictors, to which Theorem 8 applies directly. The bet on increment � is $\lambda _ { j } = \operatorname* { m i n } \{ \operatorname* { m a x } \{ m _ { j } / ( \nu _ { j } + \bar { m } _ { j } ^ { 2 } ) , 0 \} , 0 . 7 5 \}$ , from the running mean $m _ { j }$ and variance $\nu _ { j }$ of the past normalised increments $\Delta / c _ { ; }$ shrunk by one pseudo-observation of variance 0.01. Each query’s length is bounded in advance, one label of two classes for a table and 64 values of 482 outcomes each for an airfoil, which gives $c = \log _ { 2 } ( 2 / \varepsilon ) = 7$ .64 and $c = 6 4 \log _ { 2 } ( 4 8 2 / \varepsilon ) = 9 9 5 . 6$ bits. Each cohort is scored once in a designated random order and replayed in 200 further random orders without repetition; the orders add no data.

Tables. OpenML phoneme (1489) and spambase (44), with duplicate feature vectors removed, give 4,096 and 4,079 held-out rows after 128 training rows are drawn in each of three seeds. TabPFN-v2, the public classification checkpoint with two estimators, reads the 128 rows in its context; XGBoost, with 100 trees of depth 3, trains on them.

Airfoil pressure. The AirfRANS derivative holds 1,000 simulations. Each design has 18 inputs, the inlet speed, the angle of attack and 16 samples of the surface geometry, and 64 pressure coefficients at 32 chord positions on each surface. One case represents each geometry group. The 254 query designs come from groups absent from every earlier training, calibration and test set, and the 128 training and 128 calibration designs from the groups used before. The in-context learner averages the support designs’ fields with weights from a Gaussian kernel on standardised inputs, with the bandwidth chosen from six values by leave-one-out prediction on the support set; the surrogate is a two-layer MLP of width 64 with early stopping. Each gets a Gaussian spread per pressure value from its squared errors on the calibration designs, floored at 0.05 times their median, and both laws are discretised into 480 bins of width 0.05 over [−12 12] standardised units and two open tails. The learning curves rerun this pipeline with nested sets of 4 to 512 designs, the first 128 being those of the seed’s run, keep that run’s calibration designs, queries and standardisation, and reproduce its 128-design predictions exactly.

Simulated streams. Each case draws 10,000 independent streams of up to 4,096 increments with $c = 7$ .64: zero for the exact tie; uniform on $[ - c , c ]$ for the symmetric null; 0.1 with probability $c / ( c + 0 . 1 )$ and −� otherwise for the skewed null; uniform on $[ - c , c ]$ , scaled by 0.02 over the first 256 queries, for the null of changing variance; and the advantage plus independent ±2 bits for the rest.

The TabPFN-v2 checkpoint is distributed under the Prior Labs licence, the OpenML datasets under their own terms, and the AirfRANS database and its derivative under the Open Database Licence 1.0.

## D.8. Learning curves in context

Each sweep fixes its queries, items and orders before any subset is scored. Every subset that an order reaches is run through the model, which rebuilds its memory from that subset alone, with no fitted decoder and no parameter update; a subset identical to one of an earlier run reuses that run’s saved predictions.

Occupancy. Each shape of the eight constructed chair pairs of Appendix D.1 has 300 fixed anchor points and nine optional patches of 300 points, the first of which observes the feature. The orders are four random orders drawn with a fixed seed, the order with the feature patch first and the order with it last, which reach 48 distinct subsets per pair. The code length is that of the occupancy at the 1,024 feature sites of each shape, averaged over the two shapes of the pair.

Stereo. Reference view 0 of each of the five DTU scenes of Appendix D.2 is reconstructed from every nonempty subset of its four source views. The code length is that of the depth hypothesis nearest the oracle depth among the network’s 192, at the pixels where at least one source view observes the surface point and at least one does not, by visibility labels computed independently of the network. The observing views first and the non-observing views first are ordered for each pixel, with ties in view order, and the random curve is the mean over all subsets of each size.

Kriging. Each of eight DrivAerNet query cars has six support cars from its family, and the kriging learner of Appendix D.3 needs two of them for its leave-one-out selection, so its curves start at two. The orders are four random orders and the order of the gains $I _ { G }$ that the learner’s own full-context predictions assign each car, computed before the sweep. The code length is the Gaussian log-density of the pressure in fixed $C _ { p }$ units, weighted by area over 512 sites.

## D.9. Numerical conventions

The scorer for the occupancy, depth and pressure predictors computes categorical and Bernoulli log probabilities in a stable log domain. A zero ablated mass with positive full-context mass gives an infinite gain, which is kept. Gaussian variances are strictly positive. Numerical tests allow a declared tolerance for round-off and check that claim-level bits never exceed the gain. For a categorical law on common physical bins $d _ { 1 } , \ldots , d _ { m }$

$$
I _ { G } = \sum _ { j = 1 } ^ { m } p _ { j } \log _ { 2 } ( p _ { j } / q _ { j } ) , \qquad p ( \mathcal { A } ) = \sum _ { j : d _ { j } \in \mathcal { A } } p _ { j } .\tag{31}
$$

For scalar Gaussian laws $N ( \mu _ { 1 } , \nu _ { 1 } )$ and $N ( \mu _ { 0 } , \nu _ { 0 } )$ ，

$$
I _ { G } = \frac { 1 } { 2 \ln 2 } \left[ \ln \frac { \nu _ { 0 } } { \nu _ { 1 } } + \frac { \nu _ { 1 } + ( \mu _ { 1 } - \mu _ { 0 } ) ^ { 2 } } { \nu _ { 0 } } - 1 \right] ,\tag{32}
$$

and the probability of a fixed interval $\begin{array} { r } { [ a - \epsilon , a + \epsilon ] \mathrm { ~ i s ~ } \Phi \bigl ( \frac { a + \epsilon - \mu } { \sqrt { \nu } } \bigr ) - \Phi \bigl ( \frac { a - \epsilon - \mu } { \sqrt { \nu } } \bigr ) } \end{array}$ , computed from whichever tail avoids cancellation. For occupancy the local law is already Bernoulli, so the gain and the claim-level bits agree.

## E. Extended related work

Code length and prequential learning curves. Minimum description length identifies learning with compression (Rissanen, 1984; Grünwald, 2007), and the prequential approach scores a forecaster by the code length of its sequential predictions (Dawid, 1984). Blier and Ollivier (2018) showed that deep networks compress their training data far better under prequential codes than under variational ones, and Delétang et al. (2024) showed that large language models are strong general-purpose compressors. The present account carries this view from the weights to the context: the value of a context is the code length it saves, the learning curves of the context memory and of the fine-tune are both prequential code lengths of the context (Proposition 7), and a transformer’s curve in context costs one teacher-forced pass.

Usable information. Xu et al. (2020) define the information a computationally bounded observer can use, V-information, as the reduction in the best log loss a predictive family achieves, and Ethayarajh et al. (2022) measure its pointwise form on single examples. The gain Γ is pointwise usable information with the family reduced to one frozen model and its memory, and Theorem 2 ties it exactly to Shannon information: its expectation is the information the memory stores, plus the prior repair, minus the decoding loss of the model’s readout. Pointwise, one context can save more bits than the entropy of the query (Appendix C.2.1); in expectation, the information term cannot.

In-context learning as inference and as optimisation. In-context learning has been read as implicit Bayesian inference (Xie et al., 2022), as a learned algorithm for simple function classes (Garg et al., 2022), as learning algorithms such as gradient descent and ridge regression run by attention (von Oswald et al., 2023; Akyürek et al., 2023), and as implicit fine-tuning (Dai et al., 2023); for one layer of linear self-attention, one step of gradient descent is the optimal in-context learner (Mahankali et al., 2024). From the side of training, a linear layer trained by gradient descent answers a test input by attending over its training patterns (Irie et al., 2022). Proposition 13 gives the exact form for softmax attention: with the context/prefix split of He et al. (2022) and the divided-difference identity of Huang et al. (2026), the cache is a weight write of rank at most min $\left[ \left. D \right| , d _ { h } \right)$ whose per-example step sizes are recomputed for every query, and the law prices that write and a fine-tune in the same bits.

Fine-tuning as a channel. Under gradient flow, the fine-tune’s gain on a query is a time integral of the empirical neural tangent kernel (Jacot et al., 2018) contracted with the residuals of the query and the context (Proposition 12), and Malladi et al. (2023) show that prompt-based fine-tuning of language models often follows such kernel dynamics, with the kernel extended to Adam. The transfer integral normalises the same kernel by the context’s own and turns it into a budget: every bit removed from the context’s code length moves $r _ { t }$ bits to the query (Proposition 17). Prefix tuning and low-rank adaptation write a small state into the network (Li and Liang, 2021; Hu et al., 2022), and He et al. (2022) place them in one family by writing prefix tuning as a gated mixture of attention over the prefix and over the input; the storage budget (Theorem 9) bounds how much of the cache any such state can reproduce.

Context distillation and the choice between channels. Context distillation fine-tunes a model to reproduce, without the context, the predictions it makes with it (Askell et al., 2021; Snell et al., 2022). It minimises a divergence of the form on the left of Theorem 9, and the theorem sets the floor of that divergence, averaged over contexts, for a state that stores a given number of bits about the context. Lampinen et al. (2025) compare in-context learning and fine-tuning on knowledge held out of pretraining, find that with matched data in-context learning generalises several kinds of inference more flexibly, and improve fine-tuning by adding in-context inferences to its data; Mosbach et al. (2023) compare the two at matched model size and number of examples. Proposition 6 settles the choice from one measurement whenever a certificate fires, Theorem 8 settles it from held-out queries with anytime-valid error control, and the transfer integral gives the route by which a fine-tune that compresses its data moves a query.

Anytime-valid comparison of forecasters. Ville’s inequality bounds the chance that a nonnegative supermartingale ever crosses 1/� (Ville, 1939), and testing by betting builds on it tests and confidence sequences that stay valid at any stopping time (Ramdas et al., 2023; Waudby-Smith and Ramdas, 2024). Henzi and Ziegel (2022) construct e-values for score differences between competing probability forecasts, and Choe and Ramdas (2024) construct anytime-valid confidence sequences and e-processes for the average score difference between two sequential forecasters. Theorem 8 is such a comparison, with the context memory and the fine-tune as the two forecasters and code length as the score.

## F. Reproducibility

The supplementary material holds the code and saved results behind the experiments, and its README maps each script to the section, figure and table it reproduces.

The package. The package shippable in the supplementary material implements the evaluator, the certificates, the anytime-valid test and the adjudicator for any model that returns a probability law. Its saved laws and stopping traces for the comparisons of Section 6.2 replay without model weights, and scripts/domains/airfrans\_curves.py recomputes the learning curves of Figure 4a from the included AirfRANS derivative. The learning curves in context of Figure 2 are recomputed by scripts/counts/count\_curves.py from the network outputs saved for every subset of context items.

Scripts. scripts/closed\_forms.py evaluates Equations (25)–(27), the ordering and the crossover ratios, and scripts/simulate\_and\_plot.py runs the simulations and draws Figure 5. In scripts/decision/, train\_icl.py trains the in-context learner, decision\_test.py runs every channel and computes the exact information terms, decision\_extend.py and decision\_extend2.py widen the recipe grids, analyze\_decision.py issues the certificates and plot\_decision.py draws Figure 10; the trained weights and ev ery per-task result are in scripts/decision/results/. In scripts/gpt2/, setup\_gpt2.py downloads and checks the GPT-2 weights, the vocabulary and the text; run\_shakes.py, run\_docs.py and run\_storage.py run the three experiments of Section 6.2; analyze\_shakes.py, analyze\_eprocess.py and analyze\_docs.py compute the reported numbers; and plot\_gpt2.py draws Figure 11. Every per-stream, per-passage and per-table result is in scripts/gpt2/results/. Repeating the model executions requires the original datasets and pretrained weights.