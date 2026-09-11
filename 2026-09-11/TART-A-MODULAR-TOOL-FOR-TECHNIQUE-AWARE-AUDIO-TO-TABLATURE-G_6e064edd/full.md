# TART: A MODULAR TOOL FOR TECHNIQUE-AWARE AUDIO-TO-TABLATURE GUITAR TRANSCRIPTION

Akshaj Gupta Samhita Konduri

Hwi Joo Park Jiachen Lian

Andrea Guzman

Robin Netzorg

Shamak Gowda Gopala Anumanchipalli

University of California, Berkeley

akshaj.gupta@berkeley.edu

## ABSTRACT

Automatic Music Transcription (AMT) for guitar remains limited by three challenges: existing systems often fail to capture expressive techniques such as slides, bends, and percussive hits; they often assign notes to incorrect stringfret combinations; and they are typically trained on clean recordings, limiting generalization to noisy real-world audio. To address these challenges, we propose TART, a modular four-stage audio-to-tablature pipeline consisting of (1) an audio-to-MIDI transcription model, (2) an expressive technique classifier, (3) an audio-conditioned T5 encoderdecoder for string-fret assignment, and (4) an automated tablature generator. We evaluate TART in a zero-shot setting on GuitarSet, EGDB, and two augmented benchmarks, Noisy GuitarSet and Noisy EGDB. Averaged across these four benchmarks, TART achieves 81.35% audio-to-MIDI F50 (+6.67 points over the best prior baseline), 71.8% string-fret Tab F1 (+8.5 points over the best prior baseline), and an overall 54.08% end-to-end Tab F1. To our knowledge, TART is the first framework to generate guitar tablature with both fingering and expressive technique annotations directly from guitar audio.

## 1. INTRODUCTION

Automatic Music Transcription (AMT) is the process of converting audio recordings into symbolic representations such as sheet music, MIDI, or tablature (tab). Audio-to-tab transcription is a long-standing problem and has proven to be much harder than transcribing piano, where deep learning has driven CRNN-based models to high note-level accuracy [1, 2]. While guitar transcription systems have adopted similar architectures [3–6], they still fall short of producing accurate tabs for several reasons.

First, guitarists rely heavily on expressive techniques such as slides, bends, harmonics, and percussive hits, none of which current AMT systems capture. Second, the guitar is pitch-redundant, meaning the same pitch can be played at several different string-fret combinations. Existing transcription systems often assign incorrect combinations, producing tablature that does not reflect how a guitarist would physically play the piece. Third, most guitar-specific models are trained on small, professionally-recorded datasets and generalize poorly to noisy, real-world recordings where transcription is most useful.

To address these issues, we propose the Technique-Aware Audio-to-Tablature Representation Tool (TART), a modular four-stage pipeline that transcribes raw guitar audio into performance-ready tablature (illustrated in Figure 1). Stage 1 transcribes the input audio into a timealigned MIDI note sequence using a high-resolution CRNN. Stage 2 labels each note with one of nine expressive techniques (e.g., slide, bend, harmonic) using a temporal CNN-BiLSTM classifier. Stage 3 resolves the guitar’s pitch redundancy problem by assigning each note a string and fret position using a T5 encoder-decoder conditioned on both the symbolic pitch sequence and the original audio. Stage 4 merges the resulting annotations into a beat-aligned MusicXML tablature score. TART achieves new state-of-theart zero-shot results across multiple benchmarks.

## 2. PRIOR WORK

AMT has already advanced significantly for piano. Hawthorne et al. [1] introduced a dual-objective CRNN that detects onsets and frame-level pitch jointly, and Kong et al. [2] extended this with a high-resolution regressionbased model that predicts precise onset and offset times. Conversely, guitar transcription has lagged behind in part due to limited paired audio-MIDI data and the polyphonic nature of the instrument. Riley et al. [3] adapted the Kong et al. backbone to acoustic guitar via domain adaptation and achieved strong results. Maman and Bermano [6] focused on addressing data scarcity through unaligned supervision. Despite these advances, current systems struggle with electric pickups, distortion, and noisy real-world recordings.

Expressive technique recognition remains challenging due to limited data and inconsistent labels across datasets. Magcil [7], AGPT [8], IDMT-SMT-Chords [9], and EG-IPT [10] have expanded coverage to bends, slides, harmonics, and percussive techniques, but each focuses on a different subset of techniques and uses incompatible label schemes. Existing technique classifiers are typically feedforward models operating on fixed-length feature vectors [11], which discard the temporal dynamics that distinguish similar techniques. As a result, no prior system has produced a unified taxonomy with reliable performance.

![](images/17b43364f70a16965b82daecace1a0227fef820242953312b2b4a73f251223eb.jpg)  
Figure 1: Flowchart depicting how the four stages are combined to transcribe input audio (left) to output tabs (right).

Resolving string-fret ambiguity has historically been framed as an optimal search problem, but modern solutions fall into two machine learning approaches. Audio-based models such as TabCNN [12] and FretNet [13] predict tablature directly from frame-level audio features, but operate without explicit pitch supervision and struggle with crossdataset generalization. Symbolic-based models treat fingering as sequence-to-sequence translation. Edwards et al. [14] used a BART-style model to predict strings from symbolic pitch, deriving frets deterministically, while the Fretting-Transformer [5] maps MIDI tokens directly to string-fret tokens with a T5-style encoder-decoder. These symbolic approaches produce playable fingerings from clean MIDI but cannot exploit timbral cues that distinguish identical pitches played on different strings.

## 3. FIRST STAGE: AUDIO-TO-MIDI CONVERSION

The goal of Stage 1 is to map an audio waveform $\boldsymbol { x } \in \mathbb { R } ^ { N }$ (16 kHz) to a set of MIDI note events

$$
\mathcal { E } = \{ ( p _ { i } , t _ { i } ^ { \mathrm { o n } } , t _ { i } ^ { \mathrm { o f f } } , v _ { i } ) \} _ { i = 1 } ^ { | \mathcal { E } | } ,\tag{1}
$$

where $p _ { i }$ is pitch, $t _ { i } ^ { \mathrm { o n } } / t _ { i } ^ { \mathrm { o f f } }$ are onset and offset times, and $v _ { i }$ is velocity.

## 3.1 Architecture

We adopt the note-only high-resolution CRNN from Kong et al. [2] as our backbone, which takes a log-mel spectrogram as input (16 kHz audio, 100 fps, 229 mel bins)

and predicts four per-frame, per-class maps: onset confidence, offset confidence, frame activation, and velocity. We leave the base architecture unchanged and instead focus our contributions on adapting the training procedure and post-processing for guitar-specific transcription.

## 3.2 Training

A major failure of guitar AMT systems trained on clean acoustic audio is poor transfer to electric pickups, distortion, and noisy consumer recording conditions. To reduce this domain shift, we pool four datasets that span a wide range of instrument domains and recording environments: GAPS [15], Guitar-TECHS [16], François Leduc [3], and the DI subset of GOAT [17]. For each dataset, we create an 80/20 train/validation split and merge the splits into a unified pool. We fine-tune the CRNN backbone with Adam (weight decay $1 0 ^ { - 4 } )$ , batch size 4, and learning rate $1 0 ^ { - 5 }$ on 30 s segments with a 10 s hop, decaying the learning rate by a factor of 0.9 every 10,000 iterations. We select the best model on the validation set.

Beyond dataset diversity, we also propose a new data augmentation strategy. Kong et al.’s [2] piano augmentation pipeline applies pitch shifting, dynamic-range compression, equalization, reverberation, and additive noise, but several of these are destructive for guitar: reverb smears attack transients, and aggressive EQ removes harmonics critical for pitch tracking. We instead develop a stochastic noise augmentor that preserves onset alignment while simulating realistic recording-condition noise. With probability $ { p _ { \mathrm { a u g } } } { = } 0 . 5$ per batch, we apply a zero-phase 80 Hz high-pass filter and additively mix a random subset of {white, pink, 60 Hz hum} noise at SNR ∼ U(25, 45) dB, then peak-normalize to 0.9. To integrate the augmentor into training, we take the checkpoint 2,000 iterations before the best non-augmented model and train it for 10,000 additional iterations with augmentation enabled, decaying the learning rate by 0.9 every 5,000 iterations and selecting the best model by validation score. To mitigate false positives, we also discard any predicted note shorter than $T _ { \mathrm { m i n } } = 3 0 \mathrm { m s } .$ Even at extreme playing speeds (15 notes per second), fully articulated guitar notes typically exceed 50 ms in duration.

<table><tr><td>Model</td><td>GS</td><td>EGDB</td><td>GS noi</td><td>EGDB noi</td><td>Avg</td></tr><tr><td>FretNet [13]</td><td>69.10</td><td>40.90</td><td>37.30</td><td>23.60</td><td>42.73</td></tr><tr><td>NoteEM [6]</td><td>82.90</td><td>59.00</td><td>70.00</td><td>67.60</td><td>69.88</td></tr><tr><td>Riley et al. [15]</td><td>88.10</td><td>68.90</td><td>74.20</td><td>67.50</td><td>74.68</td></tr><tr><td>TART (Ours)</td><td>87.40</td><td>79.00</td><td>82.20</td><td>76.80</td><td>81.35</td></tr></table>

Table 1: Zero-shot audio-to-MIDI transcription results (F50) on GuitarSet (GS), EGDB (EGDB), GuitarSet Noisy (GS noi), and EGDB Noisy (EGDB noi).

## 3.3 Evaluation

To evaluate generalization to realistic recording conditions, we generate two new benchmarks: Noisy GuitarSet and Noisy EGDB. To generate them, we apply the same corruption used in our stochastic noise augmentor (Section 3.2), with the additional inclusion of a room impulse response (IR) convolution. For each clip in the hex-debleeded subset of GuitarSet [18] and the DI subset of EGDB [19], we sample an IR from the EchoThief collection [20], convolve it with the input signal, and truncate the result to the original length to preserve frame-level alignment between audio and labels. The resulting benchmarks simulate diverse environments (e.g., nature, sanctuaries, venues) and low-quality microphones (e.g., phones, laptops).

We report standard precision, recall, and F1 with an onset tolerance of ±50 ms and a pitch tolerance of ±50 cents, denoted P50, R50, and F50 [21]. A predicted note is counted as correct only if both its onset and pitch match a groundtruth note within these tolerances. All evaluation is performed in a zero-shot setting on GuitarSet, EGDB, and their noisy counterparts.

Table 1 reports zero-shot F50 across all four benchmarks. While the Riley et al. model narrowly leads on GuitarSet (88.1% vs. 87.4%), its performance collapses on EGDB, a pattern consistent with prior systems that tune post-processing thresholds to maximize scores on specific datasets at the expense of generalization. TART achieves the best average F50 score of 81.35%, outperforming the next-best system (Riley et al., 74.68%) by 6.67 points, with particularly large margins on EGDB (+10.1 points), Noisy GuitarSet (+8.0 points), and Noisy EGDB (+9.3 points).

## 4. SECOND STAGE: EXPRESSIVE TECHNIQUE CLASSIFICATION

Given Stage 1’s note events $\mathcal { E } ~ = ~ \{ ( p _ { i } , t _ { i } ^ { \mathrm { o n } } , t _ { i } ^ { \mathrm { o f f } } , v _ { i } ) \} _ { i = 1 } ^ { | \mathcal { E } | }$ and the input audio $\boldsymbol { x } \in \mathbb { R } ^ { N }$ , Stage 2 assigns each note a

technique label $\tau _ { i } \in \mathcal { T }$ , producing

$$
\mathcal { E } ^ { \tau } = \{ ( p _ { i } , t _ { i } ^ { \mathrm { o n } } , t _ { i } ^ { \mathrm { o f f } } , v _ { i } , \tau _ { i } ) \} _ { i = 1 } ^ { | \mathcal { E } | } .\tag{2}
$$

The label space $\tau$ consists of the classes listed in Table 2: bend, hammer-on/pull-off, harmonics, kick drum, palm muting, picking/no technique, slide, snare drum, and vibrato.

## 4.1 Architecture

For each note event in E, we extract the audio chunk spanning the note’s duration (from $t _ { i } ^ { \mathrm { o n } } \mathrm { t o } t _ { i } ^ { \mathrm { o f f } } )$ and compute a feature sequence at a 23 ms hop (sample rate 22.05 kHz, FFT window 1024). Each frame stacks 40 MFCCs, 40 log-mel bands, and 12 chroma coefficients into a 92-dimensional vector, z-normalized per feature across time. Sequences are padded or truncated to a length of 128 frames $( \sim 3 \mathrm { s ) }$ yielding a uniform $1 2 8 \times 9 2$ input tensor.

The classifier itself is a temporal CNN-BiLSTM, illustrated in Figure 2. Two 1D convolutional blocks with 64 and 128 filters (kernel size 3), each followed by batch normalization, max pooling (factor 2), and dropout (0.3), feed into a bidirectional LSTM with 64 units per direction (128 total). A fully-connected classification head (128 units, ReLU, batch normalization, dropout) and a softmax over the nine technique classes produce the final prediction. The model has approximately 160,000 parameters in total.

## 4.2 Training

A central challenge in expressive technique classification is that no single public dataset covers the full range of techniques used in real guitar performance, and existing datasets adopt incompatible label schemes. We consolidate five publicly available guitar technique datasets into a single unified training set: IDMT-SMT-Chords [9], Guitar-TECHS [16], AGPT [8], EG-IPT [10], and Magcil [7]. We preprocess each dataset to extract individual technique segments under a common nine-class taxonomy, then apply a stratified 72/8/20 train/validation/test split, ensuring that multiple microphone captures of the same performance never cross split boundaries.

We train with Adam (learning rate $1 0 ^ { - 3 } )$ for 200 epochs using sparse categorical cross-entropy loss, with per-class weighting inversely proportional to sample frequency to handle class imbalance. We halve the learning rate when validation loss plateaus for 5 epochs and stop early after 15 epochs without improvement on validation accuracy.

## 4.3 Evaluation

Table 2 reports per-class performance on the unseen test split. All classes exceed 95% recall, with percussion-style techniques (kick drum, snare drum) reaching 99.5%. The hardest class is vibrato (89.4% F1), which is primarily confused with bend due to their shared characteristics. To contextualize these numbers, we compare against two prior methods on the same unified dataset under identical settings: a convolutional network from Fiorini et al. [10] and a dense MLP from Stefani et al. [11], both reimplemented as baselines. As shown in Table 3, our CNN-BiLSTM substantially outperforms both (95.9% Macro F1 vs. 71.6% and 62.7%) with a 13× parameter reduction relative to Stefani et al.’s MLP, suggesting that explicitly modeling temporal dependencies is more effective and parameter-efficient than scaling feedforward capacity.

![](images/b434686984c8bf33134b97a0a18237884e4b5200b1a8aeab44d0f3bdf460a181.jpg)  
Figure 2: CNN-BiLSTM for technique classification. Tensor dimensions shown in gray (time × features).

<table><tr><td>Class</td><td>Precision</td><td>Recall</td><td>F1</td><td>Support</td></tr><tr><td>Bend</td><td>92.6%</td><td>95.1%</td><td>93.8%</td><td>183</td></tr><tr><td>Hammer/Pull-off</td><td>98.3%</td><td>96.4%</td><td>97.4%</td><td>2,343</td></tr><tr><td>Harmonics</td><td>92.2%</td><td>97.7%</td><td>94.9%</td><td>617</td></tr><tr><td>Kick Drum</td><td>96.7%</td><td>99.5%</td><td>98.1%</td><td>441</td></tr><tr><td>Palm Muting</td><td>98.8%</td><td>97.0%</td><td>97.9%</td><td>1,952</td></tr><tr><td>Picking/No Technique</td><td>98.7%</td><td>97.3%</td><td>98.0%</td><td>3,919</td></tr><tr><td>Slide</td><td>90.4%</td><td>99.3%</td><td>94.7%</td><td>410</td></tr><tr><td>Snare Drum</td><td>99.1%</td><td>99.5%</td><td>99.3%</td><td>1,265</td></tr><tr><td>Vibrato</td><td>83.5%</td><td>96.3%</td><td>89.4%</td><td>242</td></tr><tr><td>Macro Avg</td><td>94.5%</td><td>97.6%</td><td>95.9%</td><td>11,372</td></tr><tr><td>Weighted Avg</td><td>97.5%</td><td>97.4%</td><td>97.4%</td><td>11,372</td></tr></table>

Table 2: Per-class results on the unified dataset test set.
<table><tr><td>Model</td><td>Params</td><td>Accuracy</td><td>Macro F1</td></tr><tr><td>Fiorini et al. [10]</td><td>3.70M</td><td>64.3%</td><td>62.7%</td></tr><tr><td>Stefani et al. [11]</td><td>2.08M</td><td>86.7%</td><td>71.6%</td></tr><tr><td>TART (Ours)</td><td>160K</td><td>97.4%</td><td>95.9%</td></tr></table>

Table 3: Comparison with prior models on the unified dataset. All models use identical splits.

## 5. THIRD STAGE: STRING AND FRET ASSIGNMENT

The guitar is pitch-redundant, meaning the same MIDI pitch $p _ { i }$ can be produced at multiple different string-fret combinations $( s , f ) \in \mathcal { S } \times \mathcal { F }$ , where $\textit { S } = \{ 1 , \ldots , 6 \}$ and $\mathcal { F } ~ = ~ \{ 0 , \ldots , 2 4 \}$ Given Stage 1’s note events $\mathcal { E } = \{ ( p _ { i } , t _ { i } ^ { \mathrm { o n } } , t _ { i } ^ { \mathrm { o f f } } , v _ { i } ) \} _ { i = 1 } ^ { | \mathcal { E } | }$ and the input audio $\boldsymbol { x } \in \mathbb { R } ^ { N }$ Stage 3 assigns each note a string-fret pair $( s _ { i } , f _ { i } )$ , producing

$$
\mathcal { E } ^ { s f } = \{ ( p _ { i } , t _ { i } ^ { \mathrm { o n } } , t _ { i } ^ { \mathrm { o f f } } , v _ { i } , s _ { i } , f _ { i } ) \} _ { i = 1 } ^ { | \mathcal { E } | } .\tag{3}
$$

Because simple playability heuristics may not align with the performer’s ground-truth fingering choices, we infer performer-consistent fingering patterns from both musical context and audio timbral cues.

## 5.1 Architecture

We build upon the Fretting-Transformer [5], which treats string and fret assignment as sequence-to-sequence translation using a T5-style encoder-decoder architecture [22] with a unified vocabulary for MIDI and tablature tokens. We extend this architecture to produce a new model, which we call AudioFret (Figure 3). First, we scale the backbone from the original smaller configuration $( d _ { \mathrm { m o d e l } } = 1 2 8 ,$ 3 layers) to a larger variant $( d _ { \mathrm { m o d e l } } = 2 5 6$ , 6 layers, 8 heads, ∼15 M parameters) with gated-GELU feed-forward layers [23], which we find improves cross-dataset generalization. Second, per-note timbral features from the raw audio are injected into the encoder alongside the symbolic MIDI tokens. This directly addresses the pitch-redundancy problem by allowing the model to exploit the fact that the same pitch played on different strings produces audibly different spectra due to differences in string gauge, tension, and overtone structure.

For each input note we extract a 200 ms mel-spectrogram around its onset, pass it through a lightweight CNN (three convolutional blocks with batch normalization and ReLU), and prepend the resulting per-note audio embeddings as a contiguous block immediately before the MIDI token sequence (after the capo and tuning conditioning tokens). The T5-style self-attention then fuses audio and symbolic information end-to-end without explicit fusion hyperparameters. We retain the tokenization scheme and the capo and tuning conditioning tokens from the original Fretting-Transformer.

## 5.2 Training

Training proceeds in two phases. In the first phase, the scaled T5-style backbone is pre-trained from scratch on the combined datasets of SynthTab [24] and DadaGP [25], consisting of 17,255 training and 1,859 validation tracks. To improve robustness, we augment the symbolic data with capo positions 0–7 and four tuning variants (standard, halfstep down, full-step down, drop D), represented as conditioning tokens prepended to each sequence. We pre-train with Adafactor (learning rate $1 \times 1 0 ^ { - 4 }$ , batch size 16) for up to 120 epochs, with early stopping on validation accuracy (patience 15). In the second phase, we pool together GAPS [15], GOAT [17] (DI subset only), and Guitar-TECHS [16] and apply an 85/15 train/validation split. We first pretrain a CNN audio encoder as a string classifier on this dataset (40 epochs, AdamW, learning rate $1 \times 1 0 ^ { - 3 }$ batch size 64). We then jointly fine-tune the CNN audio encoder and the scaled T5-style backbone end-to-end for 30 epochs with AdamW (base learning rate $5 \times 1 0 ^ { - 5 }$ for the backbone and $2 . 5 \times 1 0 ^ { - 4 }$ for the audio encoder, batch size 8, weight decay 0.01, and early stopping with patience 8).

![](images/0d49dcef6ff013e12fd3a9d4d5ee124f5e547233ef722c8c1dd0d0c660c3bafd.jpg)  
Figure 3: AudioFret architecture for string-fret assignment.

## 5.3 Post-processing

At inference time we replace greedy decoding with beam search (beam width 4) and apply two constraints at each generation step. First, we enforce the tablature token grammar: every note must be followed by a valid duration token, and within a single chord, no string may be assigned twice. Second, we mask out any chord combination that requires a fret span greater than 5 to play, satisfying the realistic reach limit of a guitarist’s hand.

<table><tr><td>Model</td><td>GS</td><td>GS noi</td><td>EGDB</td><td>EGDB noi</td><td>Avg</td></tr><tr><td>TabCNN [12]</td><td>一</td><td>一</td><td>30.4</td><td>25.4</td><td>27.9</td></tr><tr><td>Fretting-Transformer [5]</td><td>60.4</td><td>60.4</td><td>66.3</td><td>66.3</td><td>63.3</td></tr><tr><td>Audio-only (ours)</td><td>54.8</td><td>55.2</td><td>64.8</td><td>58.9</td><td>58.4</td></tr><tr><td>Symbolic-only (ours)</td><td>61.3</td><td>61.3</td><td>72.7</td><td>72.7</td><td>67.0</td></tr><tr><td>AudioFret (ours)</td><td>69.2</td><td>69.5</td><td>74.8</td><td>73.7</td><td>71.8</td></tr></table>

Table 4: Zero-shot note-level Tab F1 (%) on GuitarSet, EGDB, Noisy GuitarSet, and Noisy EGDB. Since TabCNN was trained on GuitarSet, we report only its EGDB results.

![](images/332743e6f8f03a17311cb55bb5e6e04e0a99d27ecf8637483c3cb54103e05729.jpg)  
Figure 4: A sample of the chorus of “Tears In Heaven” by Eric Clapton, played by YouTube guitarist Kenneth Acoustic (bottom) and the transcription by TART (top). TART occasionally confuses slides with hammer-ons, but otherwise closely matches the ground-truth tablature.

## 5.4 Evaluation

We report Tab F1 as the primary metric, where a predicted note is counted as correct only if onset, pitch, and string assignment all match the ground truth (using the same ±50 ms onset / ±50 cents pitch tolerances as in Stage 1). All results are reported in a zero-shot setting on GuitarSet and EGDB, including their noisy counterparts. Stage 3 is evaluated in an oracle setting, where the model receives ground-truth MIDI events along with the corresponding audio to make predictions. We compare AudioFret against TabCNN [12], the original Fretting-Transformer [5], and two ablations of our model: a Symbolic-only variant (our scaled T5 backbone with no audio conditioning) and an Audio-only variant (the CNN string classifier with no symbolic decoder).

As shown in Table 4, scaling the symbolic backbone from the original Fretting-Transformer raises average Tab F1 from 63.3% to 67.0%, a gain of +3.7 percentage points without introducing any audio information. Adding audio conditioning on top of the scaled backbone yields a further +4.8 percentage points, for a total improvement of +8.5 percentage points over the original Fretting-Transformer baseline. AudioFret achieves the best overall Tab F1 of 71.8%, along with a pitch accuracy of 100.0%.

## 6. FOURTH STAGE: TABLATURE GENERATION

Stage 4 merges the parallel outputs of Stages 2 and 3 (the technique-annotated events ${ \mathcal { E } } ^ { \tau }$ and the string-fret assignments $\bar { \mathcal { E } } ^ { s f } )$ together with a tempo estimate $\beta$ from Beat-Net [26], into a single fully-annotated event stream stored

<table><tr><td>Setting</td><td>GS</td><td>GS noi</td><td>EGDB</td><td>EGDB noi</td><td>Avg</td></tr><tr><td>Oracle Tab F1</td><td>69.2</td><td>69.5</td><td>74.8</td><td>73.7</td><td>71.83</td></tr><tr><td>End-to-end Tab F1</td><td>56.0</td><td>51.2</td><td>55.2</td><td>53.9</td><td>54.08</td></tr><tr><td>Propagation Cost (∆)</td><td>13.2</td><td>18.4</td><td>19.7</td><td>19.8</td><td>17.75</td></tr></table>

Table 5: Error propagation for AudioFret.

in a JAMS annotation file:

$$
\mathcal { E } ^ { \mathrm { f i n a l } } = \{ ( p _ { i } , t _ { i } ^ { \mathrm { o n } } , t _ { i } ^ { \mathrm { o f f } } , v _ { i } , \tau _ { i } , s _ { i } , f _ { i } ) \} _ { i = 1 } ^ { | \mathcal { E } | } .\tag{4}
$$

Stage 4 then renders the pair $( \mathcal { E } ^ { \mathrm { f i n a l } } , \beta )$ into a beat-aligned MusicXML tablature score.

The first step for conversion to MusicXML is rhythmic quantization. We compile a quantized event series from ${ \mathcal { E } } ^ { \mathrm { f i n a l } }$ by clustering onsets within 30 ms to remove microtiming errors and quantizing both onsets and durations to a 1/16-note grid using BeatNet’s estimated tempo. Notes sharing the same quantized onset are grouped into chords. Same-string collisions are resolved by retaining the longerduration note.

The second step is score rendering. We export the quantized event stream to MusicXML. Technique annotations from the JAMS file that pertain to a single note (bend, harmonic, vibrato, palm muting, kick drum, snare drum, and picking) are written directly, while techniques that pertain to two consecutive notes (hammer-on/pull-off and slide) are inferred by pairing the note with the next note on the same string within a 1-beat window. Users may optionally provide information on capo position, tuning, and tempo if known; otherwise standard tuning and no capo are assumed. Figure 1 shows the full TART pipeline, and Figure 4 compares tablature generated from TART to ground truth on a section of the song “Tears in Heaven.”

## 7. END-TO-END EVALUATION

TART’s pipeline is structured in a way such that technique detection and string-fret predictions require Stage 1’s MIDI output as input. Therefore, Stage 1 errors directly impact Stage 2’s and Stage 3’s end-to-end performance. To measure this effect, we compare AudioFret results using two separate metrics:

• Oracle Tab F1. AudioFret is given the ground-truth MIDI as input (same as Stage 3 results).

• End-to-end Tab F1. AudioFret is given Stage 1’s predicted MIDI as input.

The difference between the two, which we call the propagation cost, is the Tab F1 that the system loses because Stage 1 is imperfect.

As shown in Table 5, the full pipeline achieves an endto-end Tab F1 score of 54.08% across the four zero-shot test datasets, with consistent performance on both clean and noisy audio. The drop in overall performance due to propagation error between Stage 1 and Stage 3 is 17.75 percentage points.

## 8. CONCLUSION

We present TART, a four-stage pipeline that converts guitar audio into tablature with both expressive technique labels and playable string-fret fingerings. TART combines a noise-robust audio-to-MIDI transcriber, a temporal technique classifier trained on a newly unified nine-class dataset, and AudioFret, a novel audio-conditioned T5 model that exploits timbral cues that purely symbolic models cannot access. Across four zero-shot benchmarks, TART outperforms prior systems.

Several limitations remain. The audio-to-MIDI stage does not detect unpitched notes such as percussion, which prevents downstream technique annotation. In addition, the expressive technique classification stage only assigns a single technique per note, failing to capture simultaneous techniques (e.g., a bend performed with vibrato). Furthermore, tablature generation relies on fixed-rhythmic quantization for chord grouping, which can produce occasional misgroupings. Nevertheless, TART closes a long-standing gap in guitar AMT by jointly modeling pitch, expressive technique, and string-fret position in generated tablature.

## 9. REFERENCES

[1] C. Hawthorne, E. Elsen, J. Song, A. Roberts, I. Simon, C. Raffel, J. H. Engel, S. Oore, and D. Eck, “Onsets and frames: Dual-objective piano transcription,” in Proceedings ofthe 19th International Society for Music Information Retrieval Conference, ISMIR 2018, Paris, France, September 23-27, 2018, E. Gómez, X. Hu, E. Humphrey, and E. Benetos, Eds., 2018, pp. 50–57. [Online]. Available: http://ismir2018.ircam.fr/doc/pdfs/19\_Paper.pdf

[2] Q. Kong, B. Li, X. Song, Y. Wan, and Y. Wang, “High-resolution piano transcription with pedals by regressing onset and offset times,” IEEE/ACM Trans. Audio, Speech and Lang. Proc., vol. 29, p. 3707–3717, Oct. 2021. [Online]. Available: https: //doi.org/10.1109/TASLP.2021.3121991

[3] X. Riley, D. Edwards, and S. Dixon, “High resolution guitar transcription via domain adaptation,” in ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024, pp. 1051–1055.

[4] R. M. Bittner, J. J. Bosch, D. Rubinstein, G. Meseguer-Brocal, and S. Ewert, “A lightweight instrumentagnostic model for polyphonic note transcription and multipitch estimation,” in Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Singapore, 2022.

[5] A. Hamberger, S. Murgul, J. Schmidt, and M. Heizmann, “Fretting-transformer: Encoder-decoder model for midi to tablature transcription,” in Proceedings of the 50th International Computer Music Conference (ICMC 2025). Boston, MA, USA: Composers Society of Singapore, Jun. 2025, pp. 438–445.

[6] B. Maman and A. H. Bermano, “Unaligned supervision for automatic music transcription in the wild,” in Proceedings ofthe 39th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 162. PMLR, 2022, pp. 14 918–14 934.

[7] A. Mitsou, A. Petrogianni, E. A. Vakalaki, C. Nikou, T. Psallidas, and T. Giannakopoulos, “A multimodal dataset for electric guitar playing technique recognition,” Data in Brief, vol. 52, p. 109842, 2024. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S2352340923009046

[8] D. Stefani, G. A. Giudici, and L. Turchet, “On the importance of temporally precise onset annotations for real-time music information retrieval: Findings from the ag-pt-set dataset,” in Proceedings ofthe 19th International Audio Mostly Conference: Explorations in Sonic Cultures, ser. AM ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 270–284. [Online]. Available: https: //doi.org/10.1145/3678299.3678325

[9] C.-R. Nadar, J. Abeßer, and S. Grollmisch, “Idmtsmt-chords dataset,” Zenodo, Jan. 2023, version 1.0.0. [Online]. Available: https://doi.org/10.5281/zenodo. 7544213

[10] M. Fiorini, N. Brochec, J. Borg, and R. Pasini, “Eg-ipt dataset (electric guitar instrumental playing techniques),” Apr. 2025. [Online]. Available: https: //doi.org/10.5281/zenodo.15205644

[11] D. Stefani, S. Peroni, L. Turchet et al., “A comparison of deep learning inference engines for embedded real-time audio classification,” in Proceedings of the International Conference on Digital Audio Effects, DAFx, vol. 3. DAFx, 2022, pp. 256–263.

[12] A. Wiggins and Y. E. Kim, “Guitar tablature estimation with a convolutional neural network,” in Proceedings ofthe 20th International Societyfor Music Information Retrieval Conference (ISMIR), 2019, pp. 284–291.

[13] F. Cwitkowitz, T. Hirvonen, and A. Klapuri, “Fretnet: Continuous-valued pitch contour streaming for polyphonic guitar tablature transcription,” in ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, Jun. 2023, p. 1–5. [Online]. Available: http: //dx.doi.org/10.1109/ICASSP49357.2023.10094825

[14] D. Edwards, X. Riley, P. Sarmento, and S. Dixon, “MIDI-to-Tab: Guitar tablature inference via masked language modeling,” in Proceedings of the 25th International Societyfor Music Information Retrieval Conference (ISMIR), 2024. [Online]. Available: https: //arxiv.org/abs/2408.05024

[15] X. Riley, Z. Guo, D. Edwards, and S. Dixon, “GAPS: A large and diverse classical guitar dataset and benchmark transcription model,” in Proceedings ofthe 25th

International Societyfor Music Information Retrieval Conference (ISMIR), 2024.

[16] H. Pedroza, W. Abreu, R. M. Corey, and I. R. Roman, “Guitar-techs: An electric guitar dataset covering techniques, musical excerpts, chords and scales using a diverse array of hardware,” in ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025, pp. 1–5.

[17] J. Loth, P. Sarmento, S. Sarkar, Z. Guo, M. Barthet, and M. Sandler, “GOAT: A large dataset of paired guitar audio recordings and tablatures,” in Proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR), 2025.

[18] Q. Xi, R. M. Bittner, J. Pauwels, X. Ye, and J. P. Bello, “Guitarset: A dataset for guitar transcription,” in International Society for Music Information Retrieval Conference (ISMIR), Paris, France, 2018, pp. 453–460.

[19] Y.-H. Chen, W.-Y. Hsiao, T.-K. Hsieh, J.-S. R. Jang, and Y.-H. Yang, “Towards automatic transcription of polyphonic electric guitar music: A new dataset and a multi-loss transformer model,” ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 786–790, 2022. [Online]. Available: https://api.semanticscholar.org/ CorpusID:247011871

[20] C. Warren, “EchoThief: A Library of Real-world Impulse Responses,” http://www.echothief.com/, 2015, accessed: 2026-01-17.

[21] C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto, D. Liang, D. P. W. Ellis, and C. C. Raffel, “mir\_eval: A transparent implementation of common mir metrics,” in Proceedings of the 15th International Society for Music Information Retrieval Conference (ISMIR), 2014, pp. 367–372.

[22] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” Journal ofMachine Learning Research, vol. 21, no. 140, pp. 1–67, 2020.

[23] N. Shazeer, “Glu variants improve transformer,” arXiv preprint arXiv:2002.05202, 2020.

[24] Y. Zang, Y. Zhong, F. Cwitkowitz, and Z. Duan, “Synthtab: Leveraging synthesized data for guitar tablature transcription,” in ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, Apr. 2024, p. 1286–1290. [Online]. Available: http://dx.doi.org/10. 1109/ICASSP48485.2024.10447902

[25] P. Sarmento, A. Kumar, C. J. Carr, Z. Zukowski, M. Barthet, and Y.-H. Yang, “Dadagp: A dataset of tokenized guitarpro songs for sequence models,” in Proceedings

ofthe 22nd International Societyfor Music Information Retrieval Conference (ISMIR), 2021, pp. 610–617.

[26] M. Heydari, F. Cwitkowitz, and Z. Duan, “BEATNET: CRNN AND PARTICLE FILTERING FOR ONLINE JOINT BEAT DOWNBEAT AND METER TRACK-ING,” Journal of New Music Research, 2021.