# REMOVING TIMING SHORTCUTS IMPROVES NON-INVASIVE BRAIN-TO-TEXT

Dulhan Jayalath & Oiwi Parker Jones

Neural Processing Lab (PNPL ) Department of Engineering Science, University of Oxford {dulhan, oiwi}@robots.ox.ac.uk

## ABSTRACT

We find that major reported improvements in decoding words from non-invasive brain recordings are largely reproducible without any brain data. In the influential work of d’Ascoli et al. (2025), time series of brain activity from subjects perceiving continuous speech are segmented into fixed-length windows starting at each word. A neural network then generates predictions for all of the words in a sentence together. Neighbouring windows partially overlap, implicitly revealing the interval between words. Since these intervals indicate the duration of the words spoken, and different words tend to have different durations—for example, “the” is much shorter than “supercalifragilisticexpialidocious”—the neural network can improve its predictions of words without relying on the underlying brain activity. Consistent with this, the method reaches 22.0% balanced accuracy on synthetic signals containing no brain information, compared with 22.3% on real brain recordings. To prevent the network from learning this shortcut, we make a single, simple change. Instead of jointly encoding all windows in a sentence, we process each independently. As a result, the neural network achieves better performance by learning underlying word-specific information from brain recordings. This makes two existing strategies become much more effective than before. Both aggregating predictions from distinct neural responses to the same word and using a pretrained LLM as a linguistic prior now substantially improve results. On our clinically motivated perceived speech benchmark, this simple recipe (SimpleB2T) achieves a word error rate of 36.6% with five observations per word, approaching past invasive speech decoding performance, albeit under different conditions. The results in this work expose an important shortcut in brain-to-text decoding and show that removing it leads to a simple and considerably more effective strategy.

Code & Notebooks <sup>§</sup> github.com/neural-processing-lab/SimpleB2T Benchmark õ github.com/neural-processing-lab/pnpl

## 1 INTRODUCTION

Restoring communication to people who have lost the ability to speak by decoding brain activity into speech is a north star in brain–computer interface (BCI) research. Non-invasive approaches based on magnetoencephalography (MEG) or electroencephalography (EEG) offer a safe route towards this goal, but must recover speech information from weak and spatially blurred signals measured outside the brain. Nevertheless, the state of the art in the field has progressed from detecting the presence of speech (Dash et al., 2020), to matching audio with corresponding neural responses (Defossez et al.,´ 2023), and more recently to decoding individual words (d’Ascoli et al., 2025).

In this paper, we study word-aligned brain-to-text (B2T) from perceived speech in non-invasive brain recordings. Here, a subject listens to speech (e.g. from an audiobook) or reads text while their brain activity is recorded with M/EEG and the task is to decode the words they perceived from their brain activity, knowing only when they perceived each word. Perceived speech is often used as a stepping stone towards the grander goal of decoding internal speech, such as inner monologues, because it provides stronger and more easily aligned neural responses (Martin et al., 2014). Thus, perceived speech is a common test-bed for developing non-invasive speech decoding methods (Defossez et al.,´ 2023; Tang et al., 2023; Ozdogan et al., 2025; Mantegna et al., 2026a).<sup>¨</sup>

a  
![](images/8d30f6920182ae471356339c6862a54da8df01d77b3623a9988b733bca67f0d1.jpg)

![](images/67dee0eaf20cd4aeecb1b954b6a08d3fc0e0074e21c7bf083e027e1ab48b03c6.jpg)  
Figure 1: Overlapping word-aligned windows leak word duration. (a) Overlapping wordaligned windows contain shared samples whose relative offsets reveal the duration of words. (b) To test whether jointly decoding words uses neural information, we compare MEG with synthetic signals having no relation to brain activity. With MEG, isolated means we decode each window independently; with synthetic data, it means we make windows independent. Timing tests what happens when the same model receives an encoding of the interval between words directly. Jointly decoding words reaches nearly the same accuracy on synthetic and real inputs, whereas removing the shortcut substantially reduces performance. Error bars indicate ±1σ across training seeds.

One influential recent direction in word-aligned B2T, introduced by d’Ascoli et al. (2025), has been to jointly decode neural responses to all words in a sentence from a continuous M/EEG time series. In this setup, d’Ascoli et al. extract a fixed-length window from the continuous recording at each word onset and then jointly encode all of the windows in a sentence with a neural network, predicting all of the words in the sentence at once. Predicting all of the words in the sentence together improves word classification by an average of 50% compared with decoding each window independently (d’Ascoli et al., 2025). The setup has since been extended, evaluated, and incorporated into a series of subsequent non-invasive brain-to-text studies and benchmarks (Zhang et al., 2025; Jayalath & Parker Jones, 2026; Wang et al., 2026; Li et al., 2026; Jayalath et al., 2025a; Ozdogan et al., 2025; Mantegna et al., 2026a; Landau et al., 2026; Banville et al., 2026; L<sup>¨</sup> evy et al.,´ 2026; Mantegna et al., 2026b; Jayalath et al., 2026). However, we identify an important property of this approach. Neighbouring fixed-length windows typically overlap as the windows are longer than the individual words’ durations. The same signal samples therefore appear at different relative positions in adjacent inputs, revealing the interval between word onsets and indicating the duration of words. Since typical word duration differs across words, the neural network can use this timing information to narrow the set of plausible words and improve its predictions, without using brain activity. This is an instance of shortcut learning (Geirhos et al., 2020), where a model can exploi an unintended decision rule that does not transfer to the intended use of the model.

We find that this effect is large enough to reproduce almost the entire gain from jointly decoding words without using brain activity at all (Figure 1). On real MEG, jointly decoding words reaches 22.3% balanced word accuracy, compared with 9.5% when words are decoded independently. When we replace the MEG with a synthetic continuous signal that contains no information about the stimulus but preserves the same window overlap, the decoder reaches 22.0%. Removing the overlap structure reduces accuracy to 5.8%. Thus, most of the apparent benefit of jointly decoding words can be recovered from the information exposed by overlapping inputs alone.

We explore the simplest solution to this problem: decode words individually instead of jointly so that the neural network can never learn this shortcut. We otherwise retain the setup of d’Ascoli et al. (2025). Once words are decoded independently, two techniques that previously provided limited improvements become much more effective. Firstly, multiple occurrences of distinct neural responses to the same stimulus are commonly used in non-invasive BCIs to improve signal-to-noise ratio (Farwell & Donchin, 1988), and we apply the same principle here by combining predictions from multiple observations of the same word. Secondly, invasive speech BCIs routinely combine neural predictions with explicit linguistic priors by leveraging LLMs while non-invasive methods have so far struggled to do the same (Jayalath et al., 2025a). We find that predictions driven partly by timing are poorly suited to aggregation or combination with a linguistic prior, whereas when the shortcut is removed, predictions provide information that can be combined effectively with both. Compared with jointly decoding words, both strategies now yield much larger benefits from only a few aggregated observations. On a core subset of clinically motivated sentences, this approach, which we refer to as SimpleB2T, achieves 65.6% WER with one observation per word and 36.6% with five observations.

Our work makes three points: (1) major improvements in non-invasive word-aligned B2T are largely reproducible without brain information; (2) we establish controls that expose the source of this shortcut as overlapping word-aligned windows; and (3) decoding words independently provides a simple fix that removes the shortcut, forcing the model to use information from brain activity and making aggregation of predictions and combination with a linguistic prior much more effective.

## 2 WORD DURATION LEAKAGE IN WORD-ALIGNED DECODING

We revisit the setup of d’Ascoli et al. (2025) and show how neighbouring, overlapping word-aligned windows reveal the duration of words to the decoder. We then test whether this non-neural information is sufficient to explain the large improvements from jointly decoding words.

## 2.1 DECODING WORDS JOINTLY FROM NEURAL RESPONSES

We describe the task and decoder of d’Ascoli et al. (2025). In word-aligned B2T from perceived speech, the task is to decode the words a subject perceives through an auditory or visual stimulus from their recorded brain activity. The word-aligned nature of the task assumes that the onset time of each perceived word in the stimulus is known, but no further information is assumed.

Let $X ( t ) \in \mathbb { R } ^ { C }$ denote a continuous M/EEG recording with $C$ channels and $t _ { i }$ the known onset of word $w _ { i }$ . For each word, the model extracts a three-second window beginning at its onset,

$$
\boldsymbol { x } _ { i } = \boldsymbol { X } [ t _ { i } : t _ { i } + 3 { \mathrm { s } } ] \in \mathbb { R } ^ { C \times L } ,\tag{1}
$$

where L is the number of time samples in three seconds. Windows $x _ { 1 } , \ldots , x _ { n }$ from all words $w _ { 1 } , \ldots , w _ { n }$ in a sentence are then jointly encoded by a neural network into representations $\hat { z } _ { 1 } , \dots , \hat { z } _ { n }$ where these representations are semantic word embeddings. For each word w, a target embedding $e _ { w }$ is obtained by encoding w with T5-large (Raffel et al., 2020). The neural network is trained with a contrastive objective which encourages the predicted representation $\hat { z } _ { i }$ to be similar to the target embedding $e _ { w _ { i } }$ and dissimilar to embeddings of other words in the batch.

At test time, the model predicts a word by nearest-neighbour retrieval. Given a candidate vocabulary V, each predicted embedding is compared with the corresponding frozen T5 embeddings using cosine similarity, and the highest-scoring candidate is selected:

$$
\hat { w } _ { i } = \underset { w \in \mathcal { V } } { \arg \operatorname* { m a x } } \cos ( \hat { z } _ { i } , e _ { w } ) .\tag{2}
$$

Therefore, the neural network predicts a point in the T5 embedding space and retrieves the nearest candidate word at evaluation. Jointly mapping all inputs $x _ { 1 } , \ldots , x _ { n }$ for a sentence to $\hat { z } _ { 1 } , \dots , \hat { z } _ { n }$ instead of mapping each word-aligned window $x _ { i }$ to $\hat { z } _ { i }$ individually, yields a 50% improvement on average in standard evaluations with continuous brain recordings (d’Ascoli et al., 2025).

## 2.2 OVERLAPPING WINDOWS REVEAL WORD DURATION

An important property of this construction is that consecutive word windows are not independent. Natural speech contains several words within three seconds, so adjacent windows overlap:

$$
x _ { i } = X [ t _ { i } : t _ { i } + 3 { \mathrm { s } } ] , \qquad x _ { i + 1 } = X [ t _ { i + 1 } : t _ { i + 1 } + 3 { \mathrm { s } } ] ,\tag{3}
$$

where $t _ { i + 1 }$ is often much less than $t _ { i } + 3 \mathrm { s }$ . The same underlying samples therefore occur in neighbouring inputs at a relative displacement of $t _ { i + 1 } - t _ { i }$

Indeed, this interval can be recovered directly from two adjacent inputs. In discrete time, let d denote the number of samples between their onsets. Because the windows are extracted from the same continuous recording, $x _ { i } [ n + d ] = x _ { i + 1 } [ n ]$ throughout their overlapping region. Hence

$$
d = \arg \operatorname* { m i n } _ { \ell } \sum _ { c } \sum _ { n } \left( x _ { i } ^ { c } [ n + \ell ] - x _ { i + 1 } ^ { c } [ n ] \right) ^ { 2 } ,\tag{4}
$$

![](images/ff42b4ca287eb6dcb7046de46d3dbdeaf718a8f5253293cd73ca9921839b6240.jpg)

![](images/44f4343acdb2ca6bd50d4d723112c9929d1063368955447b48d34672dedabc71.jpg)  
Figure 2: Jointly decoding words is sensitive to word duration. (a) Each point is a word with accuracy plotted against the standard deviation of its duration across training occurrences. (b) We test whether the decoder predicts words whose typical durations match the test occurrence. We measure how far the typical durations of the model’s top-10 predictions are from an occurrence’s duration. We then replace the duration with that of another occurrence of the same true word and recompute the error. The bar shows how much the error increases after this replacement, so positive values mean the predictions have durations more similar to the specific occurrence than expected. Error bars show standard deviation. $^ { * } p < . 0 5 , { } ^ { * * * } p < . 0 0 1$

where c is the sensor channel, up to preprocessing and repeated signal patterns. Thus, recovering the interval between words (and thereby word duration) requires only matching the shared samples between neighbouring windows and is trivially solvable by minimising a sum of squared differences.

When the input construction is applied to the natural speech data in this work, 99.9996% of adjacent word pairs have overlapping windows and these share 90.6% of their samples on average. Moreover, the interval between word onsets correlates strongly with word duration (r = 0.90; Appendix A.1).

This raises a simple question: how much of the reported improvement from jointly decoding words comes from neural information, and how much can be recovered from the timing information exposed by overlapping windows?

## 2.3 SYNTHETIC SIGNALS REPRODUCE DECODING IMPROVEMENTS

We test whether improvements from jointly decoding words require neural information. The concern is that fixed-length windows for neighbouring words overlap, so decoding words jointly may use word durations revealed by shared signal samples. We therefore construct controls that preserve or remove this overlap, allowing us to test both whether overlap alone can reproduce this gain and whether jointly decoding words remains useful when it is removed.

Experimental setup. We train variants of d’Ascoli et al.’s decoder and evaluate them on subject 0 of LibriBrain100 (Mantegna et al., 2026b), the largest publicly available neural speech decoding dataset at the time of writing. In this dataset, the subject listened to around 80 hours of auditory stimuli from audiobooks, podcasts, and random spoken sentences while their brain activity was recorded with MEG. We split the dataset such that no stimuli overlap between splits and all examples from a session belong exclusively to training, validation, or test. We report balanced top-1 word accuracy over the fifty most frequent words in the dataset. Exact splits are in Appendix B.1 and additional experimental details are in Appendix B.2.

We compare five conditions. Joint (MEG) uses the full decoder described by d’Ascoli et al. (2025) on real MEG data. Isolated (MEG) is similar, except it processes each word window independently in isolation, providing a reference for the gain from jointly decoding words. To test whether this gain requires neural information, in the Joint (synthetic) control, we replace MEG with a continuous synthetic signal that is unrelated to the stimulus, but extract windows at the same true word onsets. Adjacent synthetic windows therefore share the same underlying samples at offsets determined by the original word onsets. In the Isolated (synthetic) control, we generate a separate synthetic signal for each word window rather than extracting all windows from one continuous signal.

Neighbouring inputs therefore share no samples, so their relative positions no longer reveal the true intervals between words. Comparing the synthetic joint and synthetic isolated conditions shows the information provided by overlapping windows. Lastly, in the Timing condition, we encode the log interval between words and supply only this to the same model.

Results. Figure 1 shows that the gain from jointly decoding words can be reproduced almost entirely without neural information. On MEG, jointly decoding words increases word accuracy from 9.5% to 22.3%. A synthetic signal with the same overlap structure reaches 22.0%, despite containing no brain activity. Both reach similar accuracy to the model trained directly on word intervals as inputs (22.9%). When this overlap is removed, accuracy falls to 5.8%. We reproduce this effect across a further 32 subjects (Appendix A.3) and across more datasets (Appendix A.4). We also find that predictions from jointly decoding words align more strongly with duration-based word probabilities than isolated predictions (Appendix A.5). As an additional control, we retrain and evaluate on nonoverlapping natural sentence constructions, preserving the original word sequences while removing shared samples. This does not recover an advantage over isolated word decoding (Appendix A.6).

Figure 2 provides further evidence that jointly decoding words relies on word duration. In panel a, this model is substantially more accurate for words whose durations are consistent across training occurrences, an effect that is much weaker for isolated decoding. Panel b shows that this model also favours candidate words whose typical durations match the duration of the specific test occurrence, again much more strongly than the isolated model.

These results do not imply that jointly decoding words is inherently unhelpful. It could still be useful when neighbouring responses can be modelled jointly without exposing word timing, for example when words are sufficiently separated in time to avoid overlap. In naturalistic speech, however, avoiding this timing information in word-aligned B2T is difficult because truncating or masking windows at word boundaries can itself reveal the same timing information. In settings where word onsets are unknown, this kind of leakage is not an issue. However, such a setting represents a substantially different and more difficult task, which we do not evaluate here. In the word-aligned B2T setting, decoding words individually presents a way to force a neural network to predict words from brain activity without using this shortcut.

## 3 REMOVING TIMING SHORTCUTS IMPROVES SENTENCE RECONSTRUCTION

Having shown that much of the gain from jointly decoding words can arise from non-neural word duration information, we next ask what happens when we decode words independently to see how the neural network performs when it does not use the shortcut. We introduce a simple isolated word decoder and test whether its predictions can be more effectively used to reconstruct sentences, and whether these predictions can now benefit from established strategies (aggregating observations and using a language model prior). We test these questions on a clinically motivated communication benchmark and provide additional technical details of all experiments in Appendix B.2.

## 3.1 DECODING WORDS INDEPENDENTLY WITH A SIMPLE NEURAL NETWORK

We retain the word-level encoder, semantic targets, and contrastive training framework of d’Ascoli et al. (2025), while decoding words independently.

Let $x _ { i }$ denote the three-second MEG window aligned to word $w _ { i }$ . A neural network $f _ { \theta }$ maps this window to a unit-normalised embedding

$$
{ \hat { z } } _ { i } = f _ { \theta } ( x _ { i } ) .\tag{5}
$$

Here, $f _ { \theta }$ consists of a small CNN, whose output embeddings are temporally pooled, followed by a residual MLP that outputs $\hat { z } _ { i }$ . Effectively, $f _ { \theta }$ is a simplification of the d’Ascoli et al. encoder, which processes all $x _ { i }$ in a sentence through the same small CNN with temporal pooling, and then jointly processes the pooled embeddings across a sentence with a large 16-layer, 16-head bidirectional transformer. The d’Ascoli et al. model has approximately 200 million parameters, while our simplification has about 20 million. Unlike d’Ascoli et al.’s encoder, $f _ { \theta }$ has no access to the other windows in the sentence. A three-second window can nevertheless contain neural responses to subsequent words. We examine whether the predictability of this subsequent context is associated with decoding accuracy in Appendix C.1.

c  
![](images/3d12ba11985fb48539a78738995ebcb5baaae67c7e260ff17440072b44ba117c.jpg)

![](images/5f185396ba3a528c33912cc1f0d6dd04dd4aa4d9306ca442f3ae36fa176dc9bc.jpg)  
Figure 3: SimpleB2T. (a) Instead of jointly decoding words, we decode each neural response independently. We also test the effect of aggregating responses from distinct observations and using an LLM with beam search for sentence decoding. (b) Combining predictions with an LLM substantially reduces WER. (c) Decoded examples. Results are on our core set of 100 clinical sentences with $k = 5$ observations per word. Error bars indicate ±1 standard deviation across five training seeds.

As before, $f _ { \theta }$ is trained contrastively, following d’Ascoli et al. (2025), so that $\hat { z } _ { i }$ is close to $e _ { w _ { i } }$ and far from embeddings of other words. At test time, the cosine similarity $\hat { z } _ { i } ^ { \top } e _ { w }$ therefore measures the support for candidate word $w$

For a candidate vocabulary of words $\nu ,$ we convert these similarities into a distribution over words:

$$
p _ { \mathrm { b r a i n } } ( w \mid x _ { i } ) = \frac { \exp ( \hat { z } _ { i } ^ { \top } e _ { w } / T ) } { \sum _ { v \in \mathcal { V } } \exp ( \hat { z } _ { i } ^ { \top } e _ { v } / T ) } .\tag{6}
$$

The temperature $T$ is selected once on validation data and then fixed. This step turns the retrieval scores of the neural decoder into probabilities that can later be combined with a language model.

## 3.2 COMBINING OBSERVATIONS AND LINGUISTIC PRIORS

Since non-invasive neural responses have limited signal-to-noise ratio, we also test two simple ways to strengthen word predictions for sentence reconstruction. Firstly, when multiple responses to the same word are available, we aggregate their predictions. Secondly, we combine the resulting neural probabilities with an off-the-shelf language model, providing a linguistic prior to favour plausible word sequences. We refer to the recipe of independently decoding words, using aggregation, and leveraging an LLM prior as SimpleB2T. We provide further technical details in Appendix B.3.

Aggregating observations. When k distinct observations of the same word are available, we encode each independently and average their predicted embeddings,

$$
m _ { i } = \frac { 1 } { k } \sum _ { j = 1 } ^ { k } \hat { z } _ { i , j } , \qquad r _ { i } = \| m _ { i } \| _ { 2 } , \qquad u _ { i } = \frac { m _ { i } } { r _ { i } } .\tag{7}
$$

As each predicted embedding is unit normalised, $r _ { i }$ measures agreement between observations and approaches one when their predictions point in similar directions and decreases when they disagree. For candidate word w, we define the aggregated neural score

$$
A _ { i } ( w ) = \frac { 1 + \alpha r _ { i } } { T } u _ { i } ^ { \top } e _ { w } ,\tag{8}
$$

where the agreement weight α controls how strongly agreement between observations influences the score. Normalising $m _ { i }$ to obtain the consensus direction $u _ { i }$ would discard this information, so we use $r _ { i }$ to modulate the sharpness of the neural scores.

LLM rescoring. For a candidate sentence $y = ( w _ { 1 } , \ldots , w _ { n } )$ , we combine the neural scores with the autoregressive likelihood assigned by a language model:

$$
S ( y ) = \sum _ { i = 1 } ^ { n } A _ { i } ( w _ { i } ) + \lambda \log p _ { \mathrm { L L M } } ( y \mid q ) ,\tag{9}
$$

where $q$ is a fixed prompt and λ controls the contribution of the language model. In this paper, p<sub>LLM</sub> is the base model of Qwen3-8B (Qwen Team, 2025). We approximately maximise Equation (9) using left-to-right beam search over the candidate vocabulary. As in d’Ascoli et al. (2025), the number of word positions is known for each test sentence, so decoding terminates after that many words. This is part of the word-aligned B2T setting.

## 3.3 EXPERIMENTAL SETUP

We largely follow the experimental setup described in Section 2.3. For this evaluation, however, validation and test decoding are performed over a vocabulary of 92 words on the sentences in our clinical communication benchmark, described next. We report WER and the proportion of sentences where all words are decoded correctly, which we call the sentence match rate (SMR).

Clinical communication benchmark. Our primary evaluation is a benchmark of 200 clinically motivated communication sentences constructed from our held-out test recordings. The sentences are designed to represent simple patient communication needs such as requesting help, water, repositioning, environmental changes, or interaction with another person. They are divided into two 100-sentence subsets: Core, containing short care-related utterances, and Expanded, containing a broader range of questions, statements, temporal expressions, and social interactions. These clinically motivated sentences are inspired by the task design of the surgical decoding work from Moses et al. (2021), who used a smaller 50-word vocabulary and 50 total sentences. This constrained communication domain also provides a stronger linguistic prior, helping an LLM resolve uncertain neural predictions into plausible patient messages. Each sentence position is constructed using a recorded occurrence of the corresponding word from the held-out test sessions. Consequently, neighbouring sentence positions come from distinct recording events and do not share overlapping signal samples, so the shortcut exploited by jointly decoding words is unavailable. To allow for aggregating neural responses, each sentence position is assigned up to five distinct occurrences of the same word. Occurrences are sampled without replacement and are not reused across benchmark positions. Thus, five occurrences correspond to five separately recorded MEG responses to the same word. A list of all sentences is provided in Appendix D.

## 3.4 RESULTS

When decoding words jointly, aggregation provides little benefit, and LM rescoring yields performance close to or worse than the LM-only baseline (Table 1). After decoding words independently, both become substantially more effective. A likely explanation on our benchmark is that the shortcut can no longer be exploited, so predictions from d’Ascoli et al.’s model provide little information beyond the LLM. More generally, predictions from jointly decoding words may not have benefitted from linguistic priors because the information output by a decoder that learns the shortcut partly reflects word duration statistics, which may be less complementary to an external LLM prior. We analyse a version of the model retrained without overlapping windows in Appendix C.2, where we continue to find no benefit from jointly decoding words. By contrast, isolated word decoding must make each prediction from a single neural response, producing information from brain activity that can be combined more effectively across observations and with an LLM. At $k = 5$ , LLM rescoring reduces WER from 74.3% to 36.6%, while increasing the number of observations from one to five reduces WER from 65.6% to 36.6%.

Table 1: Effect of decoding words independently. We compare SimpleB2T and d’Ascoli et al., with and without the same LLM rescoring. The table tests how using k observations and LLM rescoring behave when words are decoded jointly and when words are decoded independently. Results are means across five seeds and subscripts show sample standard deviations.
<table><tr><td colspan="2"></td><td colspan="2">Core</td><td colspan="2">Expanded</td><td colspan="2">Full</td></tr><tr><td>Method</td><td>k</td><td>WER (%) ↓</td><td>SMR (%) ↑</td><td>WER (%) ↓</td><td>SMR (%) ↑</td><td>WER (%) ↓</td><td>SMR (%) ↑</td></tr><tr><td>LM only</td><td>-</td><td>73.8</td><td>2.0</td><td>91.8</td><td>0.0</td><td>83.4</td><td>1.0</td></tr><tr><td>d’Ascoli</td><td>1</td><td> $9 8 . 9 { \scriptstyle \pm 0 . 3 }$ </td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td> $9 8 . 8 { \scriptstyle \pm 0 . 1 }$ </td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td> $9 8 . 8 { \scriptstyle \pm 0 . 2 }$ </td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td></tr><tr><td>d&#x27;Ascoli + LM</td><td>1</td><td> $7 7 . 3 { \pm } 1 . 1$ </td><td> $2 . 4 { \pm } 0 . 3$ </td><td> $8 7 . 5 { \scriptstyle \pm 0 . 8 }$ </td><td> $0 . 2 { \scriptstyle \pm 0 . 1 }$ </td><td> $8 2 . 7 { \scriptstyle \pm 0 . 2 }$ </td><td> $1 . 3 { \pm } 0 . 2$ </td></tr><tr><td>SimpleB2T</td><td>1</td><td> $\mathbf { 6 5 . 6 { \scriptstyle \pm 0 . 8 } }$  </td><td> ${ \bf 6 . 4 \pm 0 . 5 }$ </td><td> ${ \bf 7 6 . 3 { \scriptstyle \pm 0 . 5 } }$ </td><td> ${ \bf 4 . 2 \pm 0 . 4 }$ </td><td> ${ \bf 7 1 . 3 { \scriptstyle \pm 0 . 5 } }$  </td><td> ${ \bf 5 . 3 \pm 0 . 3 }$ </td></tr><tr><td>d&#x27;Ascoli</td><td>5</td><td> $9 8 . 9 { \scriptstyle \pm 0 . 3 }$ </td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td> $9 8 . 8 { \scriptstyle \pm 0 . 4 }$ </td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td> $9 8 . 9 { \scriptstyle \pm 0 . 1 }$ </td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td></tr><tr><td>d’Ascoli + LM</td><td>5</td><td> $7 9 . 4 { \scriptstyle \pm 2 . 5 }$ </td><td> $2 . 4 { \pm } 0 . 5$ </td><td> $8 5 . 5 { \scriptstyle \pm 2 . 0 }$ </td><td> $0 . 2 { \scriptstyle \pm 0 . 4 }$ </td><td> $8 2 . 7 { \scriptstyle \pm 0 . 6 }$ </td><td> $1 . 3 { \pm } 0 . 3$ </td></tr><tr><td>SimpleB2T</td><td>5</td><td>_  ${ \bf 3 6 . 6 { \scriptstyle \pm 1 . 7 } }$ </td><td> ${ \bf 2 6 . 0 { \bf _ { \pm 2 . 5 } } }$ </td><td> ${ \bf 4 4 . 8 { \scriptstyle \pm 3 . 7 } }$ </td><td> ${ \bf 2 0 . 2 } _ { \pm 4 . 3 }$ </td><td> ${ \bf 4 1 . 0 2 . 2 }$ </td><td> $\mathbf { 2 3 . 1 } \mathbf { \pm 3 . 0 }$ </td></tr></table>

Table 2: Ablations. Values selected on a development set of 50 sentences, except for (c) which is retrospective and did not inform $q .$ Results on the core set of test sentences with $k = 5$ . Bold indicates the selected value. Results are means across training seeds, with sample standard deviations.
<table><tr><td>LM</td><td>WER (%)</td></tr><tr><td>Without</td><td> $7 4 . 3 { \scriptstyle \pm 2 . 0 }$ </td></tr><tr><td>With</td><td> ${ \bf 3 6 . 6 _ { \pm 1 . 7 } }$ </td></tr></table>

(a) LM contribution. LM rescoring halves WER.

<table><tr><td>Input</td><td>WER (%)</td></tr><tr><td>LM only</td><td>73.8</td></tr><tr><td>Noise + LM</td><td> $9 7 . 1 _ { \pm 1 . 7 }$ </td></tr><tr><td>Shuff. brain + LM</td><td> $8 8 . 3 { \scriptstyle \pm 1 . 9 }$ </td></tr><tr><td> $\mathrm { B r a i n } + \mathrm { L M }$ </td><td> ${ \bf 3 6 . 6 _ { \pm 1 . 7 } }$ </td></tr></table>

<table><tr><td>Prompt q</td><td>WER (%)</td></tr><tr><td>Generic</td><td> $5 5 . 2 { \scriptstyle \pm 1 . 2 }$ </td></tr><tr><td>Task-oriented</td><td> ${ \bf 3 6 . 6 _ { \pm 1 . 7 } }$ </td></tr></table>

(c) LM prompt. Describing the task and domain reduces error.

(b) Controls. Results depend on example-specific information.
<table><tr><td>Beam width</td><td>WER (%)</td></tr><tr><td>1</td><td> $6 7 . 7 _ { \pm 1 . 6 }$ </td></tr><tr><td>25</td><td> $3 7 . 9 { \scriptstyle \pm 2 . 4 }$ </td></tr><tr><td>50</td><td> ${ \bf 3 6 . 6 { \scriptstyle \pm 1 . 7 } }$ </td></tr><tr><td>200</td><td> $3 5 . 9 { \scriptstyle \pm 2 . 5 }$ </td></tr><tr><td>500</td><td> $3 6 . 0 { \scriptstyle \pm 2 . 7 }$ </td></tr></table>

(d) Beam width. WER plateaus beyond width 200.

<table><tr><td>LM weight λ</td><td>WER (%)</td></tr><tr><td>0</td><td> $7 4 . 3 { \scriptstyle \pm 2 . 0 }$ </td></tr><tr><td>0.25</td><td> $4 0 . 8 { \scriptstyle \pm 2 . 6 }$ </td></tr><tr><td>0.5</td><td> ${ \bf 3 6 . 6 { \scriptstyle \pm 1 . 7 } }$ </td></tr><tr><td>1</td><td> $4 7 . 7 { \pm } 1 . 3$ </td></tr><tr><td>2</td><td> $5 9 . 7 { \scriptstyle \pm 0 . 8 }$ </td></tr></table>

(e) LM weight. LM evidence is most useful at half-weight.

<table><tr><td>Agreement wt. α</td><td>WER (%)</td></tr><tr><td>0</td><td> $4 9 . 3 { \scriptstyle \pm 1 . 5 }$ </td></tr><tr><td>0.5</td><td> $4 2 . 7 _ { \pm 1 . 0 }$ </td></tr><tr><td>1</td><td> $4 0 . 7 { \scriptstyle \pm 1 . 5 }$ </td></tr><tr><td>2</td><td>1  ${ \bf 3 6 . 6 { \scriptstyle \pm 1 . 7 } }$ </td></tr><tr><td>4</td><td> $3 8 . 1 { \pm } 2 . 6 $ </td></tr></table>

(f) Agreement weight. Weight 2 gives the lowest WER.

Aggregating observations provides a trade-off between measurement and accuracy. Figures 4a–b show that sentence decoding improves steadily as additional responses to the same word are combined, without retraining the neural decoder. Thus, collecting more observations provides a simple way to trade additional recording burden for more reliable predictions. We also note that the language-model weight selected on the development set decreases with $k ,$ consistent with the neural predictions becoming more informative as observations are combined (Appendix B.3).

Neural and linguistic information are complementary. $\mathrm { A t } \ k = 5 ,$ the neural decoder obtains 74.3% WER and the LLM alone 73.8%. Combining them reduces WER to 36.6% (Table 2). This improvement depends on information specific to the corresponding brain response. Noise inputs matching the statistics of MEG give 97.1% WER and shuffling the predicted embeddings across test occurrences gives 88.3%. The complementarity also shows across parts of speech (Figure 4f), where neural evidence helps verbs while the LLM contributes strongly to function words. Additional analyses are provided in Appendix C.3.

Errors may be reduced without changing the neural decoder. $\mathrm { A t } k = 5 ,$ better sentences are of ten already present within the beam than the sentence ultimately selected (Figure 4c). This suggests that the neural decoder and language model often recover enough information to generate good reconstructions, but the current scoring rule does not always rank them correctly. The linguistic prior itself is also important. Explicitly describing the patient communication setting substantially improves decoding (Table 2c; Appendix C.4), presumably because it concentrates probability on the restricted set of messages that are plausible in this setting and therefore helps resolve ambiguous neural predictions. These results suggest that an important remaining bottleneck is how neural evidence and linguistic context are used to select among plausible sentences.

![](images/b1e749d0237d2cdea7f8fcff1b83a9500c1b55e87d3650b5cc4e2a2a396870f5.jpg)

![](images/7fbb7e0af32ada419c9ca2bb5dcc3f37bd40b381cb2ae74cb39d6f5773f6357e.jpg)

c  
![](images/b74c15d04b125172b11c8c5ac7b5122ad96e94d045862ebad921c44dc726bf11.jpg)

d  
![](images/4f64bc4416bab12e699231aa3bf234aac5721c1a3ef9fd2d05bed39c57f3ea09.jpg)

![](images/24cb3dade9325dc28def519ddd041bbb676a4990f5b49004e0c2cb95c2bd3b6c.jpg)

![](images/8e943a3d2075bb9deeda045b82a19e5da77e168fbe5234dab234bc44f1b96709.jpg)  
Figure 4: SimpleB2T results. (a, b) WER and exact sentence match as the number of observations increases. (c) Oracle selection among the top-N beam candidates. (d) Sentence WER distribution. (e) Confusion matrix across the 92-word vocabulary. (f) Word accuracy by part of speech for the LM, neural decoder, and their combination. Panels c–f use $k = 5 ;$ all results are on the Core test set. Shading and error bars indicate ±1 standard deviation across five training seeds.

## 4 DISCUSSION

Much of the gain from jointly decoding words reported by d’Ascoli et al. (2025) can be reproduced without brain information. Improvements arise because overlapping word-aligned windows expose word durations, which are informative about word identity. Therefore, when providing external information, one should be careful not to enable artificial shortcuts. To avoid the shortcut here, we decode words independently instead of jointly and find that two established neural decoding techniques—combining multiple observations and using an explicit linguistic prior—become substantially more effective. When predictions are driven by timing, aggregating observations gives a better estimate of a word’s typical duration, improving predictions through the same shortcut. Without the shortcut, aggregation instead improves the estimate of the word-specific neural signal. Similarly, this signal is more complementary to LLM priors than information from the shortcut.

When does the shortcut arise? The shortcut found here occurs when jointly decoding words where the interval between word onsets vary. This is common in natural speech datasets and in reading paradigms where presentation timing varies by word. Of the nine datasets analysed in d’Ascoli et al. (2025), only LittlePrinceRead and Nieuwland are unaffected as they use reading protocols that provide the same amount of time to each word. Intriguingly, these are also the only two datasets for which the authors find that jointly decoding words does not improve performance compared to decoding words independently. This is consistent with our own finding that training a model to jointly decode words without the shortcut does not improve results (Appendix C.2).

Scope of the timing-leakage result. Since its publication in late 2025, the approach introduced by d’Ascoli et al. has already been extended by several later brain-to-text methods, e.g. Jayalath et al. (2025a), Levy et al. (2026), Wang et al. (2026), and Li et al. (2026), broadening the issue be-´ yond the original work. This method has also been evaluated in several subsequent studies (Zhang et al., 2025; Jayalath & Parker Jones, 2026; Ozdogan et al., 2025; Mantegna et al., 2026a; Landau<sup>¨</sup> et al., 2026), included in benchmarks (Banville et al., 2026; Jayalath et al., 2026), and discussed in recent reviews (Yang et al., 2026). Brain2Qwerty (Levy et al., 2026) is distinct in that it operates´ on character-aligned windows from typing, potentially providing an analogous timing shortcut for keystroke intervals. However, our synthetic control suggests that timing explains little of its performance (Appendix A.7). Brain2Qwerty v2 (Zhang et al., 2026), by contrast, does not assume aligned windows. The timing shortcut is therefore relevant to a growing line of non-invasive brain-to-text work, although its importance depends on how informative timing is in each task. More generally, analogous shortcuts could arise whenever aligned, overlapping inputs are modelled jointly.

Repeated observations and word alignment. Our evaluation assumes that the temporal location of each word is known (word-aligned B2T), following d’Ascoli et al. (2025). This differs from solving speech segmentationjointly with word decoding, but is not uncommon among existing BCIs: for example, Moses et al. (2021) used a separate neural speech-detection model to identify the onset and offset of attempted words before classification. A non-invasive system could similarly use an interface in which individual word attempts are made separately, making segmentation substantially easier with non-invasive speech detection models (Dash et al., 2020; Jayalath et al., 2025b). Such a setting would also make gathering multiple observations natural by repeating attempts. Repetition is already widely used in simple non-invasive BCIs such as P300 spellers, where multiple responses are combined to improve signal-to-noise ratio (Farwell & Donchin, 1988). The scale of this effect here is notable relative to comparable speech decoding work. In the 2025 PNPL competition (Landau et al., 2025), averaging 100 observations per phoneme was used to improve decoding reliability, whereas SimpleB2T obtains large sentence-level gains from only five observations per word.

Towards useful non-invasive speech decoding. In 2021, Moses et al. provided a landmark demonstration that attempted speech could be decoded into sentences in a person with paralysis, achieving a median WER of 25.6% using a 50-word vocabulary and a set of 50 test sentences. While we do not decode speech from paralysed patients, and have not yet solved the problem of segmenting words without knowing their onsets, SimpleB2T shows that in decoding perceived speech from non-invasive recordings, it is possible to achieve WERs much closer to that landmark invasive result. This is possible while using multiple observations and operating over a larger 92-word vocabulary within a broader communication benchmark. Like their work, we focus on a constrained communi cation setting which allows for a stronger linguistic prior than unrestricted language decoding.

Once independent word decoding removes the timing shortcut introduced by the word-aligned input construction, the remaining challenges become clearer. Better selection among the top beam search candidates and reducing the number of observations required are opportunities for future work. More importantly, extending these results to decoding internally generated speech, such as imagined speech, will be a key step towards practical non-invasive communication interfaces.

## ACKNOWLEDGMENTS

We thank Gilad Landau, Francesco Mantegna, and Yonatan Gideoni for helpful comments. We also thank Stephane d’Ascoli for his correspondence and for checking a draft of this work.´

We are grateful to Modal Labs, Inc. for a generous compute grant which helped support this project.   
In particular, we thank Adam Azzam for extending the grant timeline to meet conference deadlines.

## AI USE STATEMENT

We used generative AI tools to assist with construction of the clinically motivated communication benchmark and to assist with drafting and editing the manuscript. All generated benchmark sentences were reviewed by the authors before use. Generative AI was also used to assist with code generation and debugging. All resulting code was inspected and its outputs were verified by the authors. All experimental designs, analyses, results, and scientific claims were reviewed and verified by the authors. We take responsibility for the final content of the paper, including text and artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

This work involves no new human-subject data collection. All neural recordings come from the publicly released LibriBrain100 dataset where ethical approval and participant consent for the original data collection are described by Mantegna et al. (2026b). The communication benchmark we constructed from this data is not clinically validated. Thus, we encourage user-centred evaluation before any real-world use. More broadly, progress in neural speech decoding raises important questions about privacy and consent. Practical systems should only decode deliberately provided signals with the informed consent of their users. However, we do not interpret the reported method as yet demonstrating a practical BCI since it operates on perceived and not imagined speech.

## REPRODUCIBILITY STATEMENT

We provide the information required to reproduce the experiments throughout the paper and appendices. Section 2 describes the protocol for our controls testing the overlap-window timing leakage; Section 3 defines SimpleB2T and its scoring procedure; Appendix B.3 gives the preprocessing, architecture, training, and language-model decoding settings; Appendix B.2 specifies the experimental controls; Appendix B.1 lists the exact train, validation, and test splits; and Appendix D gives the 200-sentence communication benchmark and its construction procedure. We report the random seeds used for training and evaluate models across up to five seeds.

## REFERENCES

Kristijan Armeni, Umut Guc¸l¨ u, Marcel van Gerven, and Jan-Mathijs Schoffelen. A 10-hour within-¨ participant magnetoencephalography narrative dataset to test models of language comprehension. Scientific Data, 9(1):278, 2022. License: CC-BY-4.0.

Hubert Banville, Stephane d’Ascoli, Simon Dahan, J´ er´ emy Rapin, Marl´ ene Careil, Yohann\` Benchetrit, Jarod Levy, Saarang Panchavati, Antoine Ratouchniak, Elisa Cascardi, et al. Neural-´ Bench: A unifying framework to benchmark NeuroAI models. arXiv preprint arXiv:2605.08495, 2026.

Debadatta Dash, Paul Ferrari, Satwik Dutta, and Jun Wang. NeuroVAD: Real-time voice activity detection from non-invasive neuromagnetic signals. Sensors (Basel, Switzerland), 20, 2020.

Alexandre Defossez, Charlotte Caucheteux, J ´ er´ emy Rapin, Ori Kabeli, and Jean-R ´ emi King. De-´ coding speech perception from non-invasive brain recordings. Nature Machine Intelligence, 5: 1097 – 1107, 2023.

Stephane d’Ascoli, Corentin Bel, J ´ er´ emy Rapin, Hubert J. Banville, Yohann Benchetrit, Christophe ´ Pallier, and Jean-Remi King. Towards decoding individual words from non-invasive brain record-´ ings. Nature Communications, 16:10521, 2025.

Lawrence A. Farwell and Emanuel Donchin. Talking off the top of your head: Toward a mental prosthesis utilizing event-related brain potentials. Electroencephalography & Clinical Neurophysiology, 70(6):510–23, 1988.

Robert Geirhos, Jorn-Henrik Jacobsen, Claudio Michaelis, Richard S. Zemel, Wieland Brendel,¨ Matthias Bethge, and Felix Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2:665 – 673, 2020.

Dulhan Jayalath and Oiwi Parker Jones. MEG-XL: Data-efficient brain-to-text via long-context pre-training. International Conference on Machine Learning (ICML), 2026. arXiv preprint arXiv:2602.02494.

Dulhan Jayalath, Gilad Landau, and Oiwi Parker Jones. Unlocking non-invasive brain-to-text. International Conference on Machine Learning (ICML), Workshop on Generative AI and Biology, 2025a. arXiv preprint arXiv:2505.13446.

Dulhan Jayalath, Gilad Landau, Brendan Shillingford, Mark W. Woolrich, and Oiwi Parker Jones. The Brain’s Bitter Lesson: Scaling speech decoding with self-supervised learning. International Conference on Machine Learning (ICML), 2025b. arXiv preprint arXiv:2406.04328.

Dulhan Jayalath, Benjamin Ballyk, and Oiwi Parker Jones. A common measure of communication for speech brain-computer interfaces. arXiv preprint arXiv:2609.02887, 2026.

Gilad Landau, Miran Ozdogan, Gereon Elvers, Francesco Mantegna, Pratik Somaiya, Dulhan Jay-<sup>¨</sup> alath, Luisa Kurth, Teyun Kwon, Brendan Shillingford, Greg Farquhar, Minqi Jiang, Karim Jerbi, Hamza Abdelhedi, Yorguin Mantilla Ramos, Caglar Gulcehre, Mark Woolrich, Natalie Voets, and Oiwi Parker Jones. The 2025 PNPL competition: Speech detection and phoneme classification in the LibriBrain dataset. Advances in Neural Information Processing Systems (NeurIPS), Competition Track, 2025. arXiv preprint arXiv:2506.10165.

Gilad D Landau, Dulhan Jayalath, and Oiwi Parker Jones. The semantic bottleneck: Leveraging semantic representations for non-invasive speech decoding. arXiv preprint arXiv:2609.10296, 2026.

Jarod Levy, Mingfang Zhang, Svetlana Pinet, J ´ er´ emy Rapin, Hubert J. Banville, St ´ ephane d’Ascoli,´ and Jean-Remi King. Noninvasive decoding of typed sentences from human brain activity. ´ Nature neuroscience, 2026.

Yueyang Li, Shuran Chen, Wai Ting Siok, and Nizhuan Wang. HDND: Hierarchical dynamic neural decoding for multilingual word/character retrieval from non-invasive brain recordings. arXiv preprint arXiv:2609.24095, 2026.

Ilya Loshchilov and Frank Hutter. SGDR: stochastic gradient descent with warm restarts. International Conference on Learning Representations (ICLR), 2017.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. International Conference on Learning Representations (ICLR), 2019.

Francesco Mantegna, Gereon Elvers, Dulhan Jayalath, Gilad Landau, Tasha Kim, Miran Ozdogan,<sup>¨</sup> Luisa Kurth, Teyun Kwon, SungJun Cho, Benjamin Ballyk, Alex Fung, Anna Greer, Pratik Somaiya, Christian Herff, Jean-Remi King, Yorguin Mantilla Ramos, Hamza Abdelhedi, Karim´ Jerbi, Greg Farquhar, Brendan Shillingford, Mark Woolrich, and Oiwi Parker Jones. The 2026 PNPL competition: Word-classification and efficient cross-subject generalisation in Lib riBrain100. arXiv preprint arXiv:2609.03231, 2026a.

Francesco Mantegna, Dulhan Jayalath, Gereon Elvers, Tasha Kim, Benjamin Ballyk, Alex Fung, SungJun Cho, Teyun Kwon, Luisa Kurth, Miran Ozdogan, Gilad Landau, Pratik Somaiya, Natalie<sup>¨</sup> Voets, Mark Woolrich, and Oiwi Parker Jones. LibriBrain100: One hundred hours of broad and deep MEG data for neural speech decoding at scale. arXiv preprint arXiv:2608.25204, 2026b.

Stephanie Martin, Peter Brunner, Chris Holdgraf, Hans-Jochen Heinze, Nathan E. Crone, Jochem W.´ Rieger, Gerwin Schalk, Robert T. Knight, and Brian N. Pasley. Decoding spectrotemporal features of overt and covert speech from the human cortex. Frontiers in Neuroengineering, 7, 2014.

David A. Moses, Sean L. Metzger, Jessie R. Liu, Gopala K. Anumanchipalli, Joseph G. Makin, Pengfei F. Sun, Josh Chartier, Maximilian E. Dougherty, Patricia M. Liu, Gary M. Abrams, Adelyn Tu-Chan, Karunesh Ganguly, and Edward F. Chang. Neuroprosthesis for decoding speech in a paralyzed person with anarthria. New England Journal ofMedicine, 385(3):217–227, 2021.

OpenAI. GPT-6 Astra: A New Generation of Intelligence. https://openai.com/index/ gpt-6-astra/, 2026. Accessed 2026-09-19.

Fabian Pedregosa, Gael Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier¨ Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, and Edouard Duch-<sup>´</sup> esnay. Scikit-learn: Machine learning in Python. Journal ofMachine Learning Research, 12(85): 2825–2830, 2011. RobustScaler available from https://scikit-learn.org/stable/ modules/generated/sklearn.preprocessing.RobustScaler.html.

Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofMachine Learning Research (JMLR), 21(140):1–67, 2020.

Jerry Tang, Amanda LeBel, Shailee Jain, and Alexander G. Huth. Semantic reconstruction of continuous language from non-invasive brain recordings. Nature Neuroscience, 26:858–866, 2023.

Jiaqi Wang, Huawen Hu, and Shu Zhang. Margin-regularized structured semantic alignment for brain-language correspondence. arXiv preprint arXiv:2608.16975, 2026.

Yiqian Yang, Yiqun Duan, Chenyu Liu, Yiqi Wang, Xinliang Zhou, Chin-Teng Lin, and Yu Zhang. Brain-to-language decoding: Tasks, signals, methods, evaluation, practical use and beyond. arXiv preprint arXiv:2609.27650, 2026.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. International Conference on Computer Vision (ICCV), pp. 11941–11952, 2023.

Mingfang Zhang, Jarod Levy, St´ ephane d’Ascoli, J´ er´ emy Rapin, F Alario, Pierre Bourdillon, Svet-´ lana Pinet, Jean-Remi King, et al. From thought to action: How a hierarchy of neural dynamics´ supports language production. arXiv preprint arXiv:2502.07429, 2025.

Mingfang Zhang, Jarod Levy, Cedric Rommel, J´ er´ emy Rapin, Corentin Bel, Julie Bonnaire, Daniel´ Nieto, Pierre Bourdillon, Svetlana Pinet, Stephane d’Ascoli, et al. Accurate decoding of natural´ sentences from non-invasive brain recordings. arXiv preprint arXiv:2608.18114, 2026.

Miran Ozdogan, Gilad Landau, Gereon Elvers, Dulhan Jayalath, Pratik Somaiya, Francesco Man-<sup>¨</sup> tegna, Mark Woolrich, and Oiwi Parker Jones. LibriBrain: Over 50 hours of within-subject MEG to improve speech decoding methods at scale. Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2025. arXiv preprint arXiv:2506.02098.

## APPENDIX CONTENTS

A Timing-Shortcut Analyses 15   
A.1 Relationship Between Word Timing and Duration 15   
A.2 Synthetic Control Details . 15   
A.3 Timing Shortcut Reproduction Across Multiple Subjects 16   
A.4 Timing Shortcut Reproduction Across Datasets 16   
A.5 Agreement with Duration-Based Word Predictions 16   
A.6 Jointly Decoding Words Without Overlapping Inputs 18   
A.7 Timing Shortcut Control for Brain2Qwerty 18   
B Data and Experimental Details 19   
B.1 Data Splits . 19   
B.2 Sentence Decoding Control Details 19   
B.3 Training and Decoding Details 21   
C Additional SimpleB2T Analyses 22   
C.1 Predictability from Subsequent Words . 22   
C.2 Train–Test Mismatch Control in the Clinical Benchmark 23   
C.3 Word-level Analysis . 24   
C.4 Effect of the Linguistic Prior 24   
C.5 OVMI Scores 24   
D Clinical Communication Benchmark 26

![](images/3dfcbf0fba522dca5a3637df744d2d89098236ef20cc973db7f23055fe364f38.jpg)  
Figure 5: Intervals between word onsets closely track word duration. Relationship between the annotated duration of a word and the interval from its onset to the onset of the following word across adjacent word pairs. The two quantities are strongly correlated $( r = 0 . 9 0 )$ ). The dashed line indicates equality and the solid line shows a linear fit.

## A TIMING-SHORTCUT ANALYSES

## A.1 RELATIONSHIP BETWEEN WORD TIMING AND DURATION

The timing shortcut we identify relies on two properties of the word-aligned construction, namely that neighbouring windows must overlap, and the resulting onset-to-onset intervals must contain information about word duration. We find that both hold strongly in the perceived speech data we study. Three-second windows overlap for 99.9996% of adjacent word pairs. Across 122,120 adjacent word pairs, the onset-to-onset interval is strongly correlated with the annotated duration of the preceding word $( r = 0 . 9 0$ ; Figure 5).

## A.2 SYNTHETIC CONTROL DETAILS

Joint synthetic signal. To test whether a contextual decoder can use window overlap without any neural information, we replace the MEG with a synthetic continuous signal. A simple Gaussiannoise signal would preserve identical samples across overlapping windows, but provides little structured variation for the CNN to encode. We therefore generate a sparse, smoothly varying signal with recognisable local features whose positions can be recovered after a temporal shift.

Specifically, we generate a 306-channel signal sampled at 50 Hz from three independent pulse processes, smoothed with Gaussian kernels of standard deviation 60, 120, and 240 ms. Pulses occur at a combined rate of 3 Hz. Pulse locations are shared across channels, while their amplitudes are sampled independently for each channel. The three smoothed components are summed and normalised to unit population variance. Importantly, pulse generation is entirely independent of the words and their onset times.

For each sentence, we generate one continuous synthetic signal and extract 3-second windows at the true word onsets, using the same windowing procedure as for the real MEG. Neighbouring windows therefore contain the same synthetic features at relative offsets determined by the true inter-word timing. Each window is baseline-corrected using its first 0.5 s and clipped to [−5, 5].

Isolated synthetic signal. The independent control uses the same synthetic signal generator, but samples a separate signal for every word window. Consequently, neighbouring windows share no underlying samples and the true relative word timings are no longer encoded in the inputs.

![](images/150f6041147acd50b7f8208249d56754253a2ad6119030b7c37a4c735494e56f.jpg)  
Figure 6: The overlapping window shortcut reproduces across all subjects. Subject-0-trained MEG and shared-synthetic decoders evaluated on the 32 other subjects in LibriBrain100. (a) Mean across subjects. (b) Per-subject results. Error bars show standard deviation across five training seeds.

During training, synthetic signals are regenerated on every forward pass. Validation and test signals are generated deterministically using sentence-specific random seeds. Comparing the shared and independent conditions therefore isolates the information provided by overlap between word-aligned windows.

Timing. To test whether word intervals alone support decoding, we train a timing-only control that replaces each MEG window with the interval to the next word onset within the same sentence. Intervals are log-transformed and standardised using training-set statistics. A two-layer MLP maps each scalar to a 1024-dimensional, ℓ -normalised representation, which is passed to the same sentencelevel transformer and trained with the same contrastive objective.

## A.3 TIMING SHORTCUT REPRODUCTION ACROSS MULTIPLE SUBJECTS

To test whether the overlapping-window shortcut generalises beyond the subject used for training, we evaluate the same checkpoint on all 32 remaining LibriBrain100 subjects. As shown in Figure 6, the shared synthetic decoder remains highly accurate across subjects despite receiving no neural information, and achieves performance comparable to the MEG-trained decoder. This shows that the shortcut is not specific to subject 0 and the overlap structure alone provides a transferable signal that can be exploited across unseen subjects and unseen speech material.

## A.4 TIMING SHORTCUT REPRODUCTION ACROSS DATASETS

Figure 7 shows that Figure 1b replicates on other datasets. Specifically, we conduct the same experiment on the perceived speech dataset by Armeni et al. (2022) and Le Petit Prince MEG (d’Ascoli et al., 2025). For Armeni et al., we use all subjects and sessions 1–8 for training, session 9 for validation, and session 10 for test. For Le Petit Prince, we use the listening component with all subjects and split by book section. We use runs 1–6 for training, 7 for validation, and 8–9 for testing. On both datasets, the synthetic control performs on par with using real MEG data.

## A.5 AGREEMENT WITH DURATION-BASED WORD PREDICTIONS

To examine whether jointly decoding words captures timing information, we measure how closely its predicted word probabilities align with those inferred from word durations alone. We approximate duration by the interval between consecutive word onsets within a sentence. For each word in the 50-word vocabulary, we fit a Gaussian distribution over log intervals using training occurrences. Combining these likelihoods with training word-frequency priors gives a duration-based posterior $p _ { \mathrm { d u r } } ( w \mid \bar { d } )$ where w is a word type and d is the duration in samples.

We compare this posterior with predictions from the joint MEG decoder, the joint synthetic decoder, and our isolated MEG decoder. Each decoder’s temperature is calibrated by minimising word-level negative log likelihood on validation data. For each training seed, we compute the change in Jensen– Shannon divergence

![](images/b1a0a32d49813f979199d699eac5ccd98a8146cf3eeafb675272be55ba38f81e.jpg)

b  
![](images/f9a740d747e65a51571ac904fe60971631a0c3f14b189a037f07a5704e4293f8.jpg)  
Figure 7: Replication of Figure 1b with Armeni et al. (2022) and the listening component of Le Petit Prince MEG (d’Ascoli et al., 2025).

![](images/4c0a78ab265c2becbfe48622fcf4458499010e7d36b67e29fb88bbefcb23d452.jpg)  
Figure 8: Joint decoders aligns with duration-based predictions. We measure how much the Jensen–Shannon divergence between the decoder probabilities and a duration-based word predictor’s probabilities increase when their pairing across test occurrences is randomly shuffled. Thus, larger values indicate stronger alignment with the duration-based word predictor. Bars show means across five training seeds and error bars indicate sample standard deviations.

$$
\Delta _ { \mathrm { J S } } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \left[ \mathrm { J S } \big ( p _ { i } , q _ { \pi ( i ) } \big ) - \mathrm { J S } ( p _ { i } , q _ { i } ) \right] ,\tag{10}
$$

where $p _ { i }$ is the neural word distribution, $q _ { i }$ is the duration-based posterior, and π is a random permutation of test occurrences. Shuffling preserves the collection of duration-based predictions while breaking their correspondence with individual inputs.

The increase in divergence is substantially larger for the joint MEG and synthetic decoders than for the isolated decoder (Figure 8). Thus, both joint decoders exhibit stronger alignment with evidence derived from the duration of words. The similar effect for synthetic inputs supports the interpretation that the overlap structure contributes to performance from jointly decoding words.

![](images/16d7da6b1ccc2db517036567795d1aa3b5a82fb5d7b50598d78df588fe2e0c1f.jpg)  
Figure 9: Jointly decoding words with and without overlapping inputs. We evaluate d’Ascoli et al. (2025), the same architecture trained without overlapping neighbouring windows, and isolated decoding on both natural overlapping sentences and non-overlapping versions of the same word sequences. Error bars indicate ±1 sample standard deviation across training seeds.

## A.6 JOINTLY DECODING WORDS WITHOUT OVERLAPPING INPUTS

The synthetic signal experiment in Section 2.3 shows that overlapping word windows are sufficient to reproduce most of the gain from jointly decoding words without neural information. We perform an additional control to test whether this result can instead be explained by a train–test mismatch when overlap is removed.

We preserve the original word sequence of each natural sentence, but replace every position with a separately recorded occurrence of the same word. Any two windows presented in the same sentence must either come from different recordings or begin at least three seconds apart, ensuring that they share no MEG samples. This preserves the linguistic structure of the natural sentences while removing the overlap between neighbouring neural inputs. We construct new versions of our training, validation, and test sets in this way independently where any neural response to a word may only be used once. Sentences for which we could not avoid overlaps were discarded. We were able to retain 91% of all the original sentences in the test recordings and 96% of the original sentences in the train recordings.

We compare three models: d’Ascoli et al. (2025)’s decoder trained on overlapping natural sentences; the same architecture retrained on non-overlapping sentences; and the isolated neural decoder used by SimpleB2T. All models are evaluated with one observation per word using balanced top-1 accuracy over the same 50-word vocabulary as in Section 2.3.

Figure 9 shows that removing overlap at test time causes the original model to collapse, but this alone could reflect a train–test mismatch. Retraining the same architecture without overlap recovers some performance, yet still does not outperform isolated decoding. Constructing non-overlapping sentences from separate recordings also removes coherent neural information shared across neighbouring words, leaving the joint decoding model primarily able to exploit regularities in the word sequence (effectively acting as an implicit language model).

## A.7 TIMING SHORTCUT CONTROL FOR BRAIN2QWERTY

Brain2Qwerty (Levy et al., 2026) also jointly processes aligned neural windows, raising the pos-´ sibility of an analogous timing shortcut. We therefore repeat our synthetic-input control in this setting. Unlike word decoding, synthetic inputs perform only slightly above chance, reaching 6.5% balanced character accuracy on validation and 6.8% on test, compared with 37.0% and 34.9% using real MEG. Thus, while aligned typing windows may contain some timing information, it cannot explain the strong performance of Brain2Qwerty.

![](images/acdfce1d62a6519953c4fca1f78b7643fb8709a5eaa2ad3957129baafa5f6a0c.jpg)  
Figure 10: The timing shortcut does not explain Brain2Qwerty performance. We repeat our synthetic-input control for Brain2Qwerty (Levy et al., 2026). Replacing MEG with synthetic signals´ that preserve the aligned-window structure reduces balanced character accuracy from 37.0% to 6.5% on validation and from 34.9% to 6.8% on test, close to the 3.45% chance level. Dots show individual training seeds and error bars show standard deviations.

## B DATA AND EXPERIMENTAL DETAILS

## B.1 DATA SPLITS

We partition subject-0 recordings into training, validation, and test sets as detailed in Table 3, assigning all recordings sharing a session label to the same partition across tasks. Test sessions were selected using word counts to supply at least five distinct occurrences for every word position in the core 100-sentence clinical benchmark while limiting the amount of data reserved from training. The additional expanded 100 sentences use previously unassigned occurrences from the same test sessions. Session 11 was reserved for validation, with all remaining sessions used for training.

## B.2 SENTENCE DECODING CONTROL DETAILS

LM-only baseline. To measure how much of the benchmark can be solved from linguistic priors alone, we run the same sentence-decoding procedure with the neural evidence removed. The baseline uses the frozen Qwen3-8B-Base model with the same clinical prompt as the corresponding Brain + LM condition. Beam search generates each candidate prefix autoregressively, with no access to MEG signals or reference words. The only difference from Brain + LM is therefore the absence of the neural score. Because decoding is deterministic once the prompt and sentence length are fixed, we report a single LM-only result. This baseline quantifies how much performance is explained by the language model under the vocabulary and length constraints of our benchmark.

Noise control. The language model may generate plausible clinical sentences even when the neural input contains no useful information. We therefore replace each MEG window with Gaussian noise matched to the first two moments of the preprocessed training data. We sample $M = 1 0 \small { , } 0 0 0$ training windows without replacement and estimate the mean and population variance separately fo each sensor c and onset-relative time point t:

$$
\mu _ { c , t } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } x _ { j , c , t } , \qquad \sigma _ { c , t } ^ { 2 } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \left( x _ { j , c , t } - \mu _ { c , t } \right) ^ { 2 } .\tag{11}
$$

Table 3: LibriBrain100 subject-0 session split. Session IDs and task names are taken from recording filenames. Task ranges are inclusive: for example, Sherlock1–9 denotes the nine tasks Sherlock1 through Sherlock9. Each listed task contributes one recording for that session ID. All recordings sharing a session label are assigned to the same split, including across tasks. The split contains 121 training recordings (66.29 hours), 10 validation recordings (5.57 hours), and 31 test recordings (14.71 hours). Durations refer to complete recordings before windowing. Core and Expanded use the test partition; development sentences use the validation partition.
<table><tr><td>Session ID</td><td>Split</td><td>Tasks</td><td>Recordings</td><td>Hours</td></tr><tr><td></td><td>1 Train</td><td>MOCHATIMIT, Sherlock1–9, TIMIT, TheMoth</td><td>12</td><td>6.96</td></tr><tr><td>2</td><td>Train</td><td>MOCHATIMIT, Sherlock1–9, TIMIT, TheMoth</td><td>12</td><td>6.85</td></tr><tr><td>3</td><td>Train</td><td>MOCHATIMIT, Sherlock1–9, TIMIT, TheMoth</td><td>12</td><td>6.45</td></tr><tr><td>4</td><td>Train</td><td>MOCHATIMIT, Sherlock1–9, TIMIT, TheMoth</td><td>12</td><td>6.47</td></tr><tr><td>5</td><td>Train</td><td>Sherlock1–9, TIMIT, TheMoth</td><td>11</td><td>5.51</td></tr><tr><td>6</td><td>Train</td><td>Sherlock1–9, TIMIT, TheMoth</td><td>11</td><td>6.61</td></tr><tr><td>8</td><td>Train</td><td>Sherlock1–9, TIMIT, TheMoth</td><td>11</td><td>6.39</td></tr><tr><td>9</td><td>Train</td><td>Sherlock1–9, TIMIT, TheMoth</td><td>11</td><td>7.02</td></tr><tr><td>10</td><td>Train</td><td>Sherlock1–9, TIMIT, TheMoth</td><td>11</td><td>5.77</td></tr><tr><td>12</td><td>Train</td><td>Sherlock1–7, Sherlock9, TIMIT, TheMoth</td><td>10</td><td>6.43</td></tr><tr><td>17</td><td>Train</td><td>TheMoth</td><td>1</td><td>0.25</td></tr><tr><td>19</td><td>Train</td><td>TheMoth</td><td>1</td><td>0.18</td></tr><tr><td>23</td><td>Train</td><td>TheMoth</td><td>1</td><td>0.24</td></tr><tr><td>24</td><td>Train</td><td>TheMoth</td><td>1</td><td>0.24</td></tr><tr><td>25</td><td>Train</td><td>TheMoth</td><td>1</td><td>0.29</td></tr><tr><td>26</td><td>Train</td><td>TheMoth</td><td>1</td><td>0.24</td></tr><tr><td>27</td><td>Train</td><td>TheMoth</td><td>1</td><td>0.13</td></tr><tr><td>28</td><td>Train</td><td>TheMoth</td><td>1</td><td>0.24</td></tr><tr><td>11</td><td>Validation</td><td>Sherlock1–7, Sherlock9, TIMIT, TheMoth</td><td>10</td><td>5.57</td></tr><tr><td>0</td><td>Test</td><td>Sherlock9</td><td>1</td><td>0.08</td></tr><tr><td>7</td><td>Test</td><td>Sherlock1–9, TIMIT, TheMoth</td><td>11</td><td>7.00</td></tr><tr><td>13</td><td>Test</td><td>Sherlock5–7, TIMIT, TheMoth</td><td>5</td><td>2.63</td></tr><tr><td>14</td><td>Test</td><td>Sherlock5–7, TIMIT, TheMoth</td><td>5</td><td>2.84</td></tr><tr><td>15</td><td>Test</td><td>Sherlock5, TheMoth</td><td>2</td><td>0.70</td></tr><tr><td>16</td><td>Test</td><td>TheMoth</td><td>1</td><td>0.22</td></tr><tr><td>18</td><td>Test</td><td>TheMoth</td><td>1</td><td>0.18</td></tr><tr><td>20</td><td>Test</td><td>TheMoth</td><td>1</td><td>0.23</td></tr><tr><td>21</td><td>Test</td><td>TheMoth</td><td>1</td><td>0.21</td></tr><tr><td>22</td><td>Test</td><td>TheMoth</td><td>1</td><td>0.21</td></tr><tr><td>29</td><td>Test</td><td>TheMoth</td><td>1</td><td>0.22</td></tr><tr><td>30</td><td>Test</td><td>TheMoth</td><td>1</td><td>0.19</td></tr><tr><td>Total: 31 session labels</td><td></td><td></td><td>162</td><td>86.57</td></tr></table>

Total: 31 session labels

Table 4: SimpleB2T hyperparameters. The ablations use $\lambda = 0 . 5 .$ , while other results use λ = (1.5, 1, 0.5, 0.5, 0.5) for $\bar { k } = \bar { 1 } , \ldots , 5 .$
<table><tr><td>Hyperparameter</td><td>Setting</td></tr><tr><td>Input</td><td></td></tr><tr><td>MEG channels / sampling rate</td><td>306 / 50 Hz</td></tr><tr><td>Word window / baseline correction</td><td>3 s from onset / subtract channel-wise mean of first 0.5 s</td></tr><tr><td>Bandpass filter / clipping</td><td>0.1–40 Hz / [—5, 5] after scaling</td></tr><tr><td>Channel scaling</td><td>Recording-wise, channel-wise RobustScaler (Pedregosa et al., 2011)</td></tr><tr><td>Model CNN</td><td>Temporal CNN with spatial channel merging and temporal attention</td></tr><tr><td></td><td>pooling (Défossez et al., 2023; d’Ascoli et al., 2025)</td></tr><tr><td>CNN depth / hidden channels</td><td>5 / 160</td></tr><tr><td>Kernel size / dilation period</td><td>3/5</td></tr><tr><td>Input dropout / batch normalization Initial linear width</td><td>0.1 / enabled</td></tr><tr><td>Spatial merger positional dimension</td><td>512</td></tr><tr><td>Residual MLP blocks</td><td>2048</td></tr><tr><td>MLP dimensions</td><td>4; LayerNorm-Linear-GELU-Linear</td></tr><tr><td>Embedding normalisation</td><td>1024 → 2048 → 1024</td></tr><tr><td>Word targets</td><td>Unit l2 norm before and after MLP</td></tr><tr><td></td><td>Mean embeddings from layer 12 of T5-large (Raffel et al., 2020)</td></tr><tr><td>Training and model selection</td><td></td></tr><tr><td>Training observations</td><td>Individual word occurrences (k = 1)</td></tr><tr><td>Objective</td><td>D-SigLIP (Zhai et al., 2023; d’Ascoli et al., 2025)</td></tr><tr><td>Optimizer / learning rate</td><td>AdamW (Loshchilov &amp; Hutter, 2019)  $/ 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay / batch size</td><td>0 /128</td></tr><tr><td>Schedule / maximum epochs</td><td>Cosine annealing (Loshchilov &amp; Hutter, 2017) / 50</td></tr><tr><td>Early-stopping patience</td><td>10 epochs without improvement</td></tr><tr><td>Checkpoint selection</td><td>Validation balanced word accuracy (k = 5)</td></tr><tr><td>Training seeds</td><td>0, 100, 200, 300, 400</td></tr><tr><td>Sentence decoding</td><td></td></tr><tr><td>Temperature T</td><td></td></tr><tr><td>Individual-evidence α</td><td>0.049</td></tr><tr><td>Language model</td><td>2</td></tr><tr><td></td><td>Qwen3-8B-Base (Qwen Team, 2025)</td></tr><tr><td>LM weight λ</td><td>Default: 0.5; observation-dependent settings below</td></tr><tr><td>Beam width</td><td>50</td></tr></table>

Each test occurrence is then replaced by

$$
\tilde { x } _ { c , t } = \mu _ { c , t } + \sigma _ { c , t } \epsilon _ { c , t } , \qquad \epsilon _ { c , t } \sim \mathcal { N } ( 0 , 1 ) ,\tag{12}
$$

with independent draws across sensors, time points, and occurrences. The resulting inputs therefore match the sensor- and time-specific mean and variance of real MEG in expectation, but contain no information about the perceived word.

Shuffled control. The Gaussian-noise control also removes the spatial and temporal structure of real MEG. We therefore use a second control that preserves the neural representations produced from real brain recordings while removing their correspondence to the correct words. We apply a single random permutation, chosen independently of the labels and without replacement, to the predicted embeddings before sentence decoding. This preserves the distribution and structure of the neural representations, but assigns them to the wrong test occurrences. A substantial advantage for correctly aligned embeddings over this shuffled control therefore shows that decoding depends on example-specific evidence decoded from the brain.

## B.3 TRAINING AND DECODING DETAILS

Table 4 summarises the preprocessing, architecture, training, and sentence decoding settings.

Language-model scoring. We use the fixed prompt A patient communicates a short request to hospital staff., followed by a newline and Patient:. We do not apply a

a  
![](images/debf3e5ebe53854cde1960557b94fee9276657b9a957929b35e579b9ccf912e2.jpg)

![](images/bd14982ee8f5600c60bdebd6715731dbebf2a0288e37f1fb0e03b09cfeedb84b.jpg)  
Figure 11: Language-model weight decreases as neural evidence improves. (a) Developmentselected λ falls as more observations are aggregated. (b) WER is minimised at lower LM weights for larger k.

chat template or insert any additional special tokens. Candidate words are formatted with a leading space, the first word of the sentence is capitalised, and the pronoun $^ { 6 6 } \mathrm { I } ^ { \prime }$ is always uppercase.

If a candidate word $w$ is tokenised as $\left( u _ { 1 } , \ldots , u _ { m } \right)$ , we score it by summing the autoregressive log probabilities of its constituent tokens:

$$
\ell _ { \mathrm { L M } } ( w \mid h ) = \sum _ { j = 1 } ^ { m } \log p _ { \mathrm { L M } } ( u _ { j } \mid h , u _ { < j } ) ,\tag{13}
$$

where $h$ contains the fixed prompt together with the current candidate sentence prefix. Each token probability is computed using the model’s full tokenizer vocabulary.

Beam search. At each sentence position, beam search expands every selected prefix with all 92 candidate words. For each expansion, we add the neural decoder score and the language-model score weighted by λ. The number of word positions is known in advance, and the decoder therefore generates exactly one word at each position.

After the final word, we also score the end of the sentence. For each remaining candidate sentence, we add

$$
\lambda \log \left[ p _ { \mathrm { L M } } ( .  { | } h ) + p _ { \mathrm { L M } } ( ?  { | } h ) \right] ,\tag{14}
$$

where both punctuation marks are single tokens and $h$ now contains the complete candidate sentence.   
The highest-scoring completed candidate is returned.

Language-model weight across observations. We select the language-model weight λ separately for each number of observations k using the 50-sentence development set, and fix the selected value before test evaluation. The selected weights are

$$
\frac { k } { \lambda } \ \frac { 1 } { 1 . 5 } \ \begin{array} { c c c c } { { 2 } } & { { 3 } } & { { 4 } } & { { 5 } } \\ { { 1 . 5 } } & { { 1 . 0 } } & { { 0 . 5 } } & { { 0 . 5 } } & { { 0 . 5 } } \end{array}
$$

As more observations are aggregated, the selected language-model weight decreases and then stabilises (Figure 11).

## C ADDITIONAL SIMPLEB2T ANALYSES

## C.1 PREDICTABILITY FROM SUBSEQUENT WORDS

This analysis tests whether SimpleB2T benefits from information about later words contained within each individual three-second window.

For each core test occurrence, we identify all later words whose annotated durations fall entirely within the same three-second window. We then use frozen Qwen3-8B-Base to measure how strongly this continuation predicts the target word. Specifically, for each of the 92 candidate target words, we compute the likelihood of the observed continuation conditioned on that candidate, and normalise these likelihoods assuming a uniform prior over candidates. This gives a probability distribution over possible target words using only the subsequent transcript. For each sentence position, we average these distributions across its five assigned occurrences.

We next test whether SimpleB2T performs better when this subsequent context is more predictive. Predictability varies substantially across positions, from 0.04% to 84.6%, with a median of 20.7% and an interquartile range of 8.1–36.9%. Thus, the benchmark contains both continuations that provide little information about the target word and continuations that strongly constrain its identity. We split examples into low- and high-predictability groups. Because some words may be intrinsically easier to decode than others, we also construct a word-matched split. For each target word separately, we rank its positions by subsequent-context predictability and assign equal numbers to the low- and high-predictability groups.

Table 5 reports positional word accuracy using the unchanged k = 5, beam-50 predictions. In the word-matched analysis, Brain accuracy rises modestly from 24.0% on low-predictability posi tions to 27.1% on high-predictability positions. In contrast, Brain + LM accuracy is almost unchanged, at 65.3% and 64.9%, respectively. The overall split shows no consistent advantage for high-predictability positions. Therefore, SimpleB2T does not appear to significantly benefit from other words in each three-second window.

Table 5: Word accuracy grouped by the predictability of subsequent context. Subscripts denote sample standard deviations across five training seeds. The LM is deterministic.
<table><tr><td></td><td colspan="2">Overall split</td><td colspan="2">Word-matched split</td></tr><tr><td>Method</td><td>Low</td><td>High</td><td>Low</td><td>High</td></tr><tr><td>LM</td><td>26.3</td><td>25.8</td><td>28.3</td><td>28.3</td></tr><tr><td>Brain</td><td> $2 7 . 5 { \scriptstyle \pm 2 . 0 }$ </td><td> $2 3 . 8 { \scriptstyle \pm 2 . 7 }$ </td><td> $2 4 . 0 { \scriptstyle \pm 2 . 1 }$ </td><td> $2 7 . 1 { \pm } 2 . 6 $ </td></tr><tr><td> $\mathrm { B r a i n } + \mathrm { L M }$ </td><td> $6 2 . 6 { \pm } 1 . 3$ </td><td> $6 4 . 0 { \scriptstyle \pm 2 . 2 }$ </td><td> $6 5 . 3 { \scriptstyle \pm 2 . 0 }$ </td><td> $6 4 . 9 { \pm } 1 . 9$ </td></tr></table>

## C.2 TRAIN–TEST MISMATCH CONTROL IN THE CLINICAL BENCHMARK

One possible explanation for the poor transfer of jointly decoding words to our clinical benchmark is a mismatch between training and evaluation. The original model is trained on sentences with overlapping neighbouring windows, whereas benchmark positions are assembled from separately recorded occurrences and therefore do not overlap. To test this, we retrain the same model on nonoverlapping sentence constructions matched to the evaluation setting.

We construct non-overlapping versions of the natural training sentences while preserving their original word sequences. For each word position, we replace the original MEG window with a three-second window from a different recorded occurrence of the same word. We require that no two windows assigned to the same constructed sentence overlap in the underlying recording. They must either come from different recordings or begin at least three seconds apart. The model therefore sees exactly the same linguistic sequence as before, but cannot recover word intervals from shared MEG samples.

To construct these sentences without reusing data, we permute recorded occurrences among positions of the same word type, so that each retained occurrence is assigned to exactly one position per epoch. We then resolve any remaining overlaps within sentences by swapping assignments between positions with the same target word. Sentences for which no valid assignment can be found are discarded. This retains 96.3% of eligible training occurrences. Assignments are re-randomised between epochs while preserving the same non-overlap constraint.

Retraining does not materially improve sentence reconstruction (Table 6). With LLM rescoring at k = 5, the matched model obtains 79.2%, 85.5%, and 82.6% WER on Core, Expanded, and Full, respectively, compared with 79.4%, 85.5%, and 82.7% for the original overlap-trained model. Our clinical benchmark is constructed from separately recorded occurrences at each sentence position, so there is no coherent cross-word neural activity for the d’Ascoli et al. decoder to exploit. The control shows that matching the decoder to the non-overlapping, word-by-word setting does not recover the performance of independent decoding.

Table 6: Controlling for train–test mismatch in the joint word decoder. We retrain the decoder of d’Ascoli et al. (2025) on non-overlapping sentence constructions and evaluate at $k = 5$ . Jointly decoding words remains much less accurate than SimpleB2T. Values are means across training seeds and subscripts show sample standard deviations.
<table><tr><td></td><td></td><td colspan="2">Core</td><td colspan="2">Expanded</td><td colspan="2">Full</td></tr><tr><td>Method</td><td>k</td><td>WER (%) ↓</td><td>SMR (%) ↑</td><td>WER  $( \% ) \downarrow$ </td><td>SMR (%) ↑</td><td>WER (%) ↓</td><td>SMR (%) ↑</td></tr><tr><td>d&#x27;Ascoli</td><td>1</td><td> $9 7 . 7 _ { \pm 0 . 4 }$ </td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td> $9 7 . 0 { \scriptstyle \pm 0 . 7 }$ </td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td> $9 7 . 3 { \scriptstyle \pm 0 . 6 }$ </td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td></tr><tr><td>d&#x27;Ascoli + LM</td><td>1</td><td> $7 9 . 9 { \scriptstyle \pm 1 . 5 }$ </td><td> $2 . 3 { \scriptstyle \pm 0 . 2 }$ </td><td> $8 6 . 2 { \scriptstyle \pm 0 . 9 }$ </td><td> $0 . 6 { \scriptstyle \pm 0 . 0 }$ </td><td> $8 3 . 2 _ { \pm 0 . 2 }$ </td><td> $1 . 4 { \scriptstyle \pm 0 . 1 }$ </td></tr><tr><td>d’Ascoli</td><td>5</td><td> $9 8 . 0 { \scriptstyle \pm 0 . 6 }$ </td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td> $9 7 . 2 { \scriptstyle \pm 1 . 1 }$ </td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td> $9 7 . 5 { \scriptstyle \pm 0 . 8 }$ </td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td></tr><tr><td> $\mathrm { d } ^ { \prime } \mathrm { A s c o l i + L M }$ </td><td>5</td><td> $7 9 . 2 { \scriptstyle \pm 1 . 6 }$ </td><td> $2 . 3 { \scriptstyle \pm 0 . 6 }$ </td><td> $8 5 . 5 { \scriptstyle \pm 0 . 7 }$ </td><td> $1 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td> $8 2 . 6 { \scriptstyle \pm 0 . 5 }$ </td><td> $1 . 7 _ { \pm 0 . 3 }$ </td></tr></table>

## C.3 WORD-LEVEL ANALYSIS

We further examine which words are easiest to decode. Figure 12 relates mean Brain + LM word accuracy across five training seeds to properties of the 92 vocabulary words. Phonetic distinguishability is the minimum phoneme edit distance to another vocabulary word, normalised by the longer pronunciation’s length, using CMUdict pronunciations with stress markers removed and taking the minimum if there are pronunciation variants. Semantic distinguishability is one minus the maximum cosine similarity to another word’s frozen T5 target embedding. Neither measure is strongly associated with accuracy $( r = - 0 . 0 2 $ and $r = - 0 . 1 1$ , respectively). Training frequency $( r = 0 . 0 9 )$ and mean word duration $( r = - 0 . 1 0 )$ also show weak associations. Duration variability, computed as the sample standard deviation of annotated word durations across the used training occurrences, has a stronger negative association $( r = - 0 . 2 6 )$ . Therefore, words with more consistent durations tend to be decoded more accurately. The strongest association among these measures is with LM predictability $( r = 0 . 4 2 )$ , defined as the target word’s probability under the frozen LM, conditioned on the fixed prompt and preceding reference words. All correlations are Pearson correlations. The accuracy-binned word groups illustrate the range of performance, with both function and content words appearing across accuracy bins.

## C.4 EFFECT OF THE LINGUISTIC PRIOR

We examine how the language-model prior changes with the prompt (Table 7). These experiments were conducted after all main SimpleB2T experiments were complete and were not used to select the prompt reported in the main results. The prompt used throughout the paper was fixed before this ablation. We compare our prompt with several alternatives ranging from a minimal prefix to broader descriptions of English, everyday speech, and clinical communication. The results show that specifying the intended patient-communication setting improves decoding relative to more generic prompts. This supports the view that part of SimpleB2T’s success comes from using a linguistic prior matched to the constrained communication task. Future work may benefit from optimising the prompt on a development set to further improve the linguistic prior.

## C.5 OVMI SCORES

WER depends on the vocabulary and language distribution used for evaluation, making it difficult to compare performance across communication settings and with other work. We therefore also report OVMI (Jayalath et al., 2026), which measures the information conveyed by a decoder relative to a reference communication distribution. Table 8 reports normalised OVMI under four reference distributions.

![](images/fe5425d9722282c21964d8151a524270a0224e0feccac1073903d5798650f57e.jpg)

![](images/b033aab03cf975b3cb165e81f434065023dfe60b5bcf0332ddc25d6c4236e993.jpg)

c  
![](images/227ec88d86aa9b23a5343b8b93b9724dd22ed59f2b739d68aaa2c8648e934e40.jpg)

d  
![](images/7b06ffc4a90f4dbbdad2dd4345610ff7861b62b3f8cd3bafd718bd15ae242171.jpg)

e  
![](images/d49cb095b6dc046728b40e74198c960a5f8a72fabd77c410383353ee930814ab.jpg)  
g

![](images/cad764c6880bc9d3a2089ac5cc07c10b158729d9bb7c8525123b8d270faaa82e.jpg)  
Figure 12: Word-level decoding analysis. Word accuracy is weakly related to (a) phonetic distinguishability, (b) semantic distinguishability, and (c) mean word duration. It is negatively related to (d) word duration variability and positively related to (e) LM predictability, with little association with (f) training frequency. (g) Words grouped by accuracy bins.

Table 7: Prompt specificity improves sentence reconstruction. Brain + LM uses $k = 5 ,$ beam width 50, and frozen Qwen3-8B-Base. Subscripts report sample standard deviations. LM only decoding is deterministic. Bold, shaded cells mark the lowest WER within each condition and split. A line break before Patient: is part of prompts E and F.
<table><tr><td></td><td></td><td colspan="2">Core</td><td colspan="2">Expanded</td><td colspan="2">Full</td></tr><tr><td>Prompt</td><td>λ</td><td>LM only</td><td> $\mathrm { B r a i n } + \mathrm { L M }$ </td><td>LM only</td><td> $\mathrm { B r a i n } + \mathrm { L M }$ </td><td>LM only</td><td> $\mathrm { B r a i n } + \mathrm { L M }$ </td></tr><tr><td>A</td><td>0.25</td><td>97.3</td><td> $5 5 . 2 { \scriptstyle \pm 1 . 2 }$ </td><td>98.1</td><td> $5 3 . 9 { \scriptstyle \pm 0 . 8 }$ </td><td>97.7</td><td> $5 4 . 5 { \scriptstyle \pm 1 . 0 }$ </td></tr><tr><td>B</td><td>0.5</td><td>96.0</td><td> $5 1 . 3 { \scriptstyle \pm 3 . 6 }$ </td><td>91.0</td><td> $5 2 . 5 { \scriptstyle \pm 2 . 6 }$ </td><td>93.3</td><td> $5 1 . 9 { \scriptstyle \pm 3 . 1 }$ </td></tr><tr><td>C</td><td>0.25</td><td>91.8</td><td> $5 1 . 8 { \scriptstyle \pm 3 . 3 }$ </td><td>90.0</td><td> $4 9 . 8 { \scriptstyle \pm 3 . 1 }$ </td><td>90.8</td><td> $5 0 . 7 _ { \pm 3 . 2 }$ </td></tr><tr><td>D</td><td>0.25</td><td>84.4</td><td> $4 9 . 3 { \scriptstyle \pm 2 . 0 }$ </td><td>95.0</td><td> $4 8 . 1 _ { \pm 3 . 2 }$ </td><td>90.0</td><td> $4 8 . 7 _ { \pm 2 . 6 }$ </td></tr><tr><td>E (ours)</td><td>0.5</td><td>73.8</td><td>_  ${ \bf 3 6 . 6 _ { \pm 1 . 7 } }$  </td><td>91.8</td><td> $4 4 . 8 _ { \pm 3 . 7 }$ </td><td>83.4</td><td> $4 1 . 0 _ { \pm 2 . 2 }$ </td></tr><tr><td>F</td><td>0.5</td><td>78.8</td><td> $3 8 . 7 { \scriptstyle \pm 4 . 3 }$ </td><td>87.2</td><td> ${ \bf 4 2 . 6 { \bf _ { \pm 5 . 1 } } }$ </td><td>83.3</td><td> $\mathbf { 4 0 . 8 \pm 4 . 6 }$ </td></tr></table>

ID Exact prompt  
A Sentence:  
B The following is a sentence from an English passage:  
C The following is a sentence from everyday speech:  
D The following is a sentence spoken in a clinical setting:  
E A patient communicates a short request to hospital staff. Patient:  
F A patient who cannot speak communicates a short message to hospital staff about comfort, positioning, personal care, their surroundings, or interaction with other people. Patient:

Table 8: SimpleB2T normalised OVMI scores across reference distributions. OVMI measures the information conveyed per word by a model relative to a reference communication distribution. Values are $1 0 0 \times \mathrm { O V } \mathrm { \bar { M } I } / \bar { H } ( p )$ , where $H ( p )$ is the entropy of the full reference distribution. These estimates use balanced word accuracies of 16.04/13.29/14.80% at k = 1 and 46.83/45.08/45.93% at k = 5 for Core/Expanded/Full, respectively. Results report means across five training seeds, with sample standard deviations as subscripts. See Jayalath et al. (2026) for further details on OVMI.
<table><tr><td>Reference p</td><td>k</td><td>Core</td><td>Expanded</td><td>Full</td></tr><tr><td rowspan="2">SUBTLEX-UK</td><td>1</td><td> $1 . 4 0 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $1 . 0 5 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $1 . 2 3 { \scriptstyle \pm 0 . 0 5 }$ </td></tr><tr><td>5</td><td> $6 . 4 1 _ { \pm 0 . 4 8 }$ </td><td> $6 . 0 9 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $6 . 2 5 { \scriptstyle \pm 0 . 5 4 }$ </td></tr><tr><td rowspan="2">Switchboard</td><td>1</td><td> $1 . 8 8 _ { \pm 0 . 1 2 }$ </td><td> $1 . 4 1 _ { \pm 0 . 0 4 }$ </td><td> $1 . 6 6 _ { \pm 0 . 0 6 }$ </td></tr><tr><td>5</td><td> $8 . 5 3 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $8 . 1 1 { \scriptstyle \pm 1 . 0 9 }$ </td><td> $8 . 3 2 _ { \pm 0 . 7 1 }$ </td></tr><tr><td rowspan="2">UCV</td><td>1</td><td> $5 . 6 8 _ { \pm 0 . 3 6 }$ </td><td> $4 . 2 9 _ { \pm 0 . 1 3 }$ </td><td> $5 . 0 4 _ { \pm 0 . 1 8 }$ </td></tr><tr><td>5</td><td> $2 4 . 3 2 { \scriptstyle \pm 1 . 7 1 }$ </td><td> $2 3 . 1 7 { \scriptstyle \pm 2 . 9 7 }$ </td><td> $2 3 . 7 2 { \scriptstyle \pm 1 . 9 3 }$ </td></tr><tr><td rowspan="2">Sherlock</td><td>1</td><td> $1 . 2 4 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 9 3 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $1 . 1 0 { \scriptstyle \pm 0 . 0 4 }$ </td></tr><tr><td>5</td><td> $5 . 6 5 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $5 . 3 7 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $5 . 5 0 { \scriptstyle \pm 0 . 4 7 }$ </td></tr></table>

## D CLINICAL COMMUNICATION BENCHMARK

Tables 9 (core) and 10 (expanded) list the 200 clinically motivated sentences used in our communication benchmark. The sentences were generated with assistance from GPT-6 Astra (OpenAI, 2026) and subsequently reviewed to ensure they represented plausible patient communication needs. The MEG samples for the benchmark were constructed by assigning recorded word occurrences to positions in predefined sentences. For each position, we select five distinct occurrences of the required word from the held-out test sessions, irrespective of their original sentence context, and extract the corresponding word-aligned MEG windows. Each occurrence is assigned to exactly one position across the 200-sentence benchmark, so no recording event is reused.

Table 9: Core communication benchmark sentences.
<table><tr><td>ID</td><td>Sentence</td><td>ID</td><td>Sentence</td></tr><tr><td>1</td><td>Can you help me?</td><td>51</td><td>Can you open the door?</td></tr><tr><td>2</td><td>I would like some help.</td><td>52</td><td>Can you open the window?</td></tr><tr><td>3</td><td>Can you come here?</td><td>53</td><td>Can you put the light on?</td></tr><tr><td>4</td><td>Can you come back?</td><td>54</td><td>Can you take this away?</td></tr><tr><td>5</td><td>Can you be here with me?</td><td>55</td><td>Can you put that over me?</td></tr><tr><td>6</td><td>Can you give me more time?</td><td>56</td><td>Can you put this by my hand?</td></tr><tr><td>7</td><td>I would like to be on my own.</td><td>57</td><td>Can you put this on the other side?</td></tr><tr><td>8</td><td>Do not go away.</td><td>58</td><td>Can you put that down?</td></tr><tr><td>9</td><td>Can you come back in a little while?</td><td>59</td><td>Can you put that here?</td></tr><tr><td>10</td><td>Can you ask for help?</td><td>60</td><td>Can you put that back?</td></tr><tr><td>11</td><td>I would like some water.</td><td>61</td><td>Can you tell me your name?</td></tr><tr><td>12</td><td>Can you give me some water?</td><td>62</td><td>Can you tell me what that is?</td></tr><tr><td>13</td><td>I would like more water.</td><td>63</td><td>Can you tell me what this is for?</td></tr><tr><td>14</td><td>That is enough water.</td><td>64</td><td>Can you say that again?</td></tr><tr><td>15</td><td>Can you put the water here?</td><td>65</td><td>Can you tell me more?</td></tr><tr><td>16</td><td>Can you help me with the water?</td><td>66</td><td>Can you give me some paper?</td></tr><tr><td>17</td><td>Can you take the water away?</td><td>67</td><td>Can you look at this paper?</td></tr><tr><td>18</td><td>I would like a little more.</td><td>68</td><td>I don&#x27;t see that.</td></tr><tr><td>19</td><td>That is too much.</td><td>69</td><td>I don&#x27;t see your face.</td></tr><tr><td>20</td><td>No more for now.</td><td>70</td><td>Can you ask me one thing at a time?</td></tr><tr><td>21</td><td>Can you help me up?</td><td>71</td><td>Can you give me a little time?</td></tr><tr><td>22</td><td>Can you help me down?</td><td>72</td><td>I have something to say.</td></tr><tr><td>23</td><td>Can you put my head up?</td><td>73</td><td>That is not what I said.</td></tr><tr><td>24</td><td>Can you put my head down?</td><td>74</td><td>Yes, that is right.</td></tr><tr><td>25</td><td>Can you put my hand here?</td><td>75</td><td>No, that is not right.</td></tr><tr><td>26</td><td>Can you put my hand there?</td><td>76</td><td>I do not know.</td></tr><tr><td>27</td><td>Can you take my hand?</td><td>77</td><td>I would like to know why.</td></tr><tr><td>28</td><td>Can you put this under my head?</td><td>78</td><td>Can you tell me when?</td></tr><tr><td>29</td><td>Can you put this under my back?</td><td>79</td><td>Can you tell me how?</td></tr><tr><td>30</td><td>Can you help me get on my side?</td><td>80</td><td>I would like to ask something.</td></tr><tr><td>31</td><td>I would like to be on my left side.</td><td>81</td><td>I would like to see my friend.</td></tr><tr><td>32</td><td>I would like to be on my right side.</td><td>82</td><td>Can you tell my friend to come here?</td></tr><tr><td>33</td><td>I would like to be on my back.</td><td>83</td><td>Can you ask my friend to come back?</td></tr><tr><td>34</td><td>Can you help me with my head?</td><td>84</td><td>Can you tell him I am here?</td></tr><tr><td>35</td><td>Can you help me with my back?</td><td>85</td><td>Can you tell her I am here?</td></tr><tr><td>36</td><td>Can you help me with my left hand?</td><td>86</td><td>I would like to see him.</td></tr><tr><td>37</td><td>Can you help me with my right hand?</td><td>87</td><td>I would like to see her.</td></tr><tr><td>38</td><td>Do not put that on my head.</td><td>88</td><td>Can you ask them to come in?</td></tr><tr><td>39</td><td>Do not put that on my back.</td><td>89</td><td>Can you ask them to come back?</td></tr><tr><td>40</td><td>That is not the right place.</td><td>90</td><td>I would like more time with them.</td></tr><tr><td>41</td><td>Can you wash my face?</td><td>91</td><td>Can you help me now?</td></tr><tr><td>42</td><td>Can you wash my hands?</td><td>92</td><td>That is too much light.</td></tr><tr><td>43</td><td>Can you wash my head?</td><td>93</td><td>There is something in my eyes.</td></tr><tr><td>44</td><td>Can you wash my back?</td><td>94</td><td>There is something on my face.</td></tr><tr><td>45</td><td>Can you wash my left hand?</td><td>95</td><td>Can you look at my hand?</td></tr><tr><td>46</td><td>Can you wash my right hand?</td><td>96</td><td>Can you look at my head?</td></tr><tr><td>47</td><td></td><td>97</td><td>Can you look at my back?</td></tr><tr><td>48</td><td>Can you wash here? Can you wash there?</td><td>98</td><td>Can you look at my eyes?</td></tr><tr><td>49</td><td>Can you help me wash?</td><td>99</td><td>Can you tell me the time?</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>50</td><td>Do not get water in my eyes.</td><td>100</td><td>Can you come back before night?</td></tr></table>

Table 10: Expanded communication benchmark sentences.
<table><tr><td>ID</td><td>Sentence</td><td>ID</td><td>Sentence</td></tr><tr><td>1</td><td>I would like some water now.</td><td>51</td><td>Tell her to come back in a little while.</td></tr><tr><td>2</td><td>Give me a little water now.</td><td>52</td><td>Tell him to come back before night.</td></tr><tr><td>3</td><td>I would like water before you go.</td><td>53</td><td>I would like to see them now.</td></tr><tr><td>4</td><td>Take the water away now.</td><td>54</td><td>I would like to have more time with her.</td></tr><tr><td>5</td><td>I have enough water for now.</td><td>55</td><td>I would like to have more time with</td></tr><tr><td>6</td><td>I would like the water by my hand.</td><td>56</td><td>him. Ask them to be here with me.</td></tr><tr><td>7</td><td>Do not give me more water.</td><td>57</td><td>I do not know your name.</td></tr><tr><td>8</td><td>Give me some more time with the water.</td><td>58</td><td>Tell me your name again.</td></tr><tr><td>9</td><td>There is water on my face.</td><td></td><td>Tell me why you have come back.</td></tr><tr><td>10</td><td>There is water in my eyes.</td><td>59 60</td><td>I would like to know what this is.</td></tr><tr><td>11</td><td>I would like to wash my face.</td><td>61</td><td>I would like to know what that is for.</td></tr><tr><td>12</td><td>I would like to wash my hands.</td><td>62</td><td>I do not know why you said that.</td></tr><tr><td>13</td><td>Wash my face with a little water.</td><td>63</td><td>Say that one more time.</td></tr><tr><td>14</td><td>Do not wash my eyes.</td><td>64</td><td>Say one thing at a time.</td></tr><tr><td>15</td><td>Wash my hands again.</td><td>65</td><td>Give me time to say what I would like.</td></tr><tr><td>16</td><td>I would like to wash before night.</td><td>66</td><td>I have more to say.</td></tr><tr><td>17</td><td>Wash the other hand too.</td><td>67</td><td>That is what I would like.</td></tr><tr><td>18</td><td>There is something under my back.</td><td>68</td><td>That is not what I would like.</td></tr><tr><td>19</td><td>There is something under my head.</td><td>69</td><td>I said no.</td></tr><tr><td>20</td><td>I would like my head a little more up.</td><td>70</td><td>I said yes.</td></tr><tr><td>21</td><td>I would like my head a little more down.</td><td>71</td><td>I would like to ask you something.</td></tr><tr><td>22</td><td>My hand is not in the right place.</td><td>72</td><td>Ask me again in a little while.</td></tr><tr><td>23</td><td>Take this away now.</td><td>73</td><td>Do not ask me now.</td></tr><tr><td>24</td><td>I would like to be on the other side.</td><td>74</td><td>I do not know how to say this.</td></tr><tr><td>25</td><td>Do not take my hand away.</td><td>75</td><td>I would like some paper now.</td></tr><tr><td>26</td><td>Put this under my hand.</td><td>76</td><td>Give me the paper again.</td></tr><tr><td>27</td><td>Put that by my side.</td><td>77</td><td>Look at what is on the paper.</td></tr><tr><td>28</td><td>Put the water on the other side.</td><td></td><td>I don&#x27;t see the paper.</td></tr><tr><td>29</td><td>Put this paper by my hand.</td><td>78 79</td><td>I would like the paper here.</td></tr><tr><td>30</td><td>I would like my back down a little.</td><td>80</td><td>There is something I would like you to</td></tr><tr><td>31</td><td>The light is in my eyes.</td><td>81</td><td>see. Look at my face when I say this.</td></tr><tr><td>32</td><td>I would like a little more light.</td><td>82</td><td>I don&#x27;t see what is in your hand.</td></tr><tr><td>33</td><td>I would like the light on now.</td><td>83</td><td>I would like to see your face.</td></tr><tr><td>34</td><td>Do not have the light on at night.</td><td>84</td><td>I don&#x27;t know what to do.</td></tr><tr><td>35</td><td>The window is open.</td><td>85</td><td>Tell me what you would like me to do.</td></tr><tr><td>36</td><td>I would like the window open for a</td><td>86</td><td>Tell me before you wash my face.</td></tr><tr><td></td><td>little while. Do not open the window now.</td><td></td><td>Tell me before you take this away.</td></tr><tr><td>37 38</td><td>I would like the door open.</td><td>87 88</td><td>I would like to know when you would</td></tr><tr><td>39</td><td>Do not open the door now.</td><td></td><td>be back. How much time do I have?</td></tr><tr><td>40</td><td>Do not have this by my eyes.</td><td>89 90</td><td>What is this water for?</td></tr><tr><td>41</td><td>Come here for a little while.</td><td>91</td><td>Is that my name on the paper?</td></tr><tr><td>42</td><td>Be here while I have some water.</td><td>92</td><td>Is my friend here now?</td></tr><tr><td>43</td><td>I would like you here with me.</td><td>93</td><td>Do you know when my friend would</td></tr><tr><td>44</td><td>Do not go while I have something to</td><td>94</td><td>be here? Do you know when I would see him</td></tr><tr><td></td><td>say.</td><td></td><td>again?</td></tr><tr><td>45</td><td>Come back at night.</td><td>95</td><td>I would like you to look at my eyes.</td></tr></table>

Continued on next page

<table><tr><td>ID</td><td>Sentence</td><td>ID</td><td>Sentence</td></tr><tr><td>46</td><td>Come back in a little while.</td><td>96</td><td>I would like you to look at my back.</td></tr><tr><td>47</td><td>I would like a little time on my own.</td><td>97</td><td>Do not give me that now.</td></tr><tr><td>48</td><td>I would like my friend here with me.</td><td>98</td><td>That is enough for now.</td></tr><tr><td>49</td><td>Ask my friend to come in.</td><td>99</td><td>I would like more time before you go.</td></tr><tr><td>50</td><td>Ask my friend to come back at night.</td><td>100</td><td>I would like to know why you said no.</td></tr></table>

Table 10 continued from previous page