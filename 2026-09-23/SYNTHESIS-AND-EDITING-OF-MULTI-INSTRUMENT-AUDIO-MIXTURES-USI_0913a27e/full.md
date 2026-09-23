# SYNTHESIS AND EDITING OF MULTI-INSTRUMENT AUDIO MIXTURES USINGSCALAR-QUANTISED LATENTS WITH MIDI SPAN CONDITIONING

Sungkyun Chang

Keshav Bhandari

Simon Dixon Emmanouil Benetos

Centre for Digital Music, Queen Mary University of London

## ABSTRACT

Music creation often involves iterative refinement, changing selected musical details while retaining the rest. To support such refinement, we introduce SpanSynth-Edit, a flow-matching model for MIDIguided synthesis and editing of multi-instrument audio mixtures using low-frame-rate scalar-quantised latents. MIDI Span encodes instrument-labelled note lifecycles as unordered event sets with continuous-valued attributes and pools each set into one conditioning vector per audio-latent frame. The model uses contextual audio for instrument-specific timbre guidance and supports editing by resynthesising the target region from revised MIDI. Experiments on single- and multi-instrument benchmarks show competitive performance and demonstrate within-frame onset control. We also discuss limitations of transcription-based note-adherence evaluation.

Index Terms— MIDI-to-audio, multi-instrument synthesis, audio editing, flow matching, note representation

## 1. INTRODUCTION

Recent advances have enabled music-audio editing from naturallanguage prompts [1, 2]. Refining existing music, however, often requires adding, removing, or adjusting individual notes while retaining other musical content [3]. MIDI provides explicit control over pitches, timing, velocities, and instrument assignments for such revisions. Realising these changes in audio remains challenging: instrumental sounds overlap in a mixture, and the requested note changes must be rendered without altering concurrent notes or their timbres.

To this end, we propose SpanSynth-Edit, a MIDI-guided approach to fine-grained music-audio editing. It resynthesises a selected region from revised MIDI, using contextual audio from the original recording for timbre guidance. Aligned contextual MIDI links these sounds to instruments and notes. When adding an instrument absent from the audio context, the model draws on its learned knowledge to generate its sound.

Related work. MIDI-to-audio models differ in whether they synthesise instrument parts independently or as a mixture. CTD [4], P-MUSE [5], and TokenSynth [6] generate single-instrument tracks, while FlowSynth [7] generates individual notes for sampled instruments. SpecDiff [8], MAC [9], and U-MusT [10] synthesise multiinstrument mixtures jointly, as does SpanSynth-Edit.

SpecDiff and U-MusT use only preceding audio for continuation, whereas CTD and TokenSynth use global timbre embeddings. MAC relies on learned, dataset-specific performer embeddings for acoustic control. P-MUSE and our model, SpanSynth-Edit, accept contextual audio from one or both sides of the target, with or without aligned contextual MIDI. Our model also renders mixtures containing instruments absent from the context.

MIDI-guided audio editing has been benchmarked for single instruments [5], while related audio-to-audio methods address timbre transfer [11]. Selected MIDI-to-audio models can in principle support editing through resynthesis from revised MIDI, motivating their use as baselines. However, the challenge is to render the requested note changes or new instrument parts while preserving the notes and timbres of other concurrent parts. We also explore adapting FlowEdit [12], a training-free method for content-preserving image editing, to MIDI-guided audio editing.

SpanSynth-Edit combines joint multi-instrument synthesis and editing using reference context. Our contributions address three complementary aspects of this setting.

Audio representation. We perform flow matching with scalarquantised (SQ) audio latents [13, 14], rather than mel-spectrograms [5, 8], codebook tokens [6, 10, 15], or unquantised latents [4, 11]. The low frame rate of SQ latents keeps sequences short to support efficient training and inference.

Note representation. We propose MIDI Span, a frame-aligned note representation derived from MIDI. Conventional piano rolls use one grid per instrument [5, 9] and quantise note timing to frames, losing precision at low frame rates. Note sequences [15, 16] retain fine timing but serialise concurrent notes. MIDI Span instead represents each instrument-labelled note throughout its lifecycle as frame-aligned events with real-valued numerical attributes that retain within-frame timing. A permutation-invariant encoder [17] pools each frame’s variable-sized event set into one conditioning feature.

Evaluation suite. We evaluate single- and multi-instrument synthesis and editing for audio quality, similarity, and note and instrument adherence. For editing, we construct paired original and revised versions, each with ground-truth audio and MIDI, to assess adherence to requested changes and preservation of unchanged notes. The model demo, checkpoints, data, and benchmark samples are available<sup>1</sup>.

## 2. MODEL

Given target MIDI and audio context, SpanSynth-Edit generates the audio mixture in the target region (Fig. 1(a)). Observed context comprises audio and optional aligned MIDI preceding and succeeding that region. Editing resynthesises changed and unchanged notes together from revised target MIDI, without additional training.

## 2.1. Scalar-Quantised Audio Representation

Codec. We use the frozen HeartCodec [14] encoder E<sub>SQ</sub> and decoder D<sub>SQ</sub>. The encoder maps 48 kHz mono audio y to 128- dimensional frames at 25 Hz (hop H = 40 ms), bounded by tanh. Coordinate-wise quantisation $Q ( \bar { x } ) = \mathrm { r o u n d } ( 9 x ) / 9$ gives 19 levels in [−1, 1] and produces the clean SQ sequence Z. The final generated latent sequence $\hat { \mathbf { Z } }$ is clipped to $[ - 1 , 1 ]$ and requantised before decoding to audio $\hat { y } = D _ { \mathrm { S Q } } ( Q ( \mathrm { c l i p } ( \mathbf { \hat { Z } } , - 1 , 1 ) ) )$ ).

![](images/65c8b14230461d85fa5c61a71bceab7aac21b0f19a4b380046d6fbae5d180f51.jpg)  
Fig. 1. Frame-aligned conditioning and MIDI Span. (a) The DiT combines noisy audio with fixed audio context, MIDI features, and binary masks, with ⊕ denoting feature-wise concatenation. The dashed red arrow shows training interpolation, and the upper codec path shows reconstruction. (b) Per-frame note-event sets retain within-frame timing and undergo permutation-invariant pooling.

Masking. Two binary mask channels B indicate the target region and contextual MIDI. The region channel $\mathbf { b } _ { \mathrm { o b s } }$ is 1 on observed frames and 0 on target frames, giving the fixed audio condition A:

$$
{ \bf Z } = Q ( E _ { \mathrm { S Q } } ( y ) ) , \qquad { \bf A } = { \bf b } _ { \mathrm { o b s } } \odot { \bf Z } .\tag{1}
$$

Thus A contains only observed SQ codes and remains fixed, without added noise, during generation.

## 2.2. Frame-Aligned MIDI Span

MIDI Span encodes notes from target and contextual MIDI as framealigned event sets with real-valued within-frame timing, then pools each set into one conditioning vector independently of slot order. Each non-drum note $j ,$ spanning $t _ { j } ^ { \mathrm { o n } }$ to $t _ { j } ^ { \mathrm { o f } }$ , is represented by one event $e _ { j , k }$ in each frame k that overlaps this interval. The event state is ONSET in the onset frame, OFFSET in the offset frame, and SUSTAIN in between. If both boundaries fall in one frame, only an ONSET event is used. Drum hits use a single ONSET event with zero remaining duration.

Attributes. Each event has two categorical attributes: event state and instrument class (including drums). Its four real-valued attributes are pitch, velocity, boundary position, and remaining duration, each normalised to [−1, 1]:

$$
e _ { j , k } = \left( c _ { \mathrm { s t a t e } } , c _ { \mathrm { i n s t r } } , \boldsymbol { x } _ { \mathrm { p i t c h } } , \boldsymbol { x } _ { \mathrm { v e l } } , \boldsymbol { x } _ { \mathrm { b o u n d a r y } } , \boldsymbol { x } _ { \mathrm { r e m } } \right) .
$$

For an onset or offset at audio time $t _ { \mathrm { e v e n t } }$ in a frame starting at $t _ { k } .$ the boundary position is

$$
x _ { \mathrm { b o u n d a r y } } = 2 ( t _ { \mathrm { e v e n t } } - t _ { k } ) / H - 1 .\tag{2}
$$

This linearly maps within-frame timing to a continuous value in [−1, 1]. At shared frame boundaries, onsets belong to the later frame and offsets to the earlier one. SUSTAIN events use $x _ { \mathrm { b o u n d a r y } } = 0 $ Pitch and velocity are linearly scaled. For non-drum notes, remaining duration is the time to offset from onset (ONSET) or frame start (otherwise), clipped to 20.48 s and log-scaled.

Event-set encoding. Frame-k events form an unordered set $E _ { k } .$ padded to 128 slots. A shared encoder maps events to 128- dimensional vectors. Following Deep Sets [17], we pool independently of slot order by concatenating one sum and five learned sigmoid-gated sums, each root-mean-square (RMS) normalised. Because RMS-norm can suppress magnitude differences reflecting event counts, we add projected, log-scaled counts of melodic/drum events from $c _ { \mathrm { i n s t r } }$ and ONSET/SUSTAIN/OFFSET events from $c _ { \mathrm { s t a t e } } , \mathrm { y i e l d i n g } m _ { k } \in \mathbb { R } ^ { 7 6 8 }$ . Padding is masked and slot-index embeddings are omitted. Empty sets, including context frames without MIDI, yield zeros. Stacking $m _ { k }$ in audio-frame order gives the MIDI condition M.

## 2.3. Conditional Flow Matching

Training. Conditional flow matching [18] mixes the clean SQ sequence $\bar { \bf z }$ with Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , I )$ at flow time $t \sim \mathcal { U } [ 0 , 1 ]$ The noisy input $\mathbf { Z } _ { t }$ and target vector field $\dot { \mathbf { V } } ^ { \star }$ are

$$
{ \bf Z } _ { t } = ( 1 - t ) \epsilon + t { \bf Z } , \mathrm { ~ } \mathbf { V } ^ { \star } = { \bf Z } - \epsilon .\tag{3}
$$

Observed and target frames share this path. Observed audio enters through both noisy $\mathbf { Z } _ { t }$ and fixed, clean A (Fig. 1(a)).

Our generator (480.8M parameters, codec excluded) uses a 25- block diffusion transformer (DiT, width 1,024) v [19] to predict the vector field $\widehat { \mathbf { V } } _ { t }$ from noisy $\mathbf { Z } _ { t } ,$ conditioned on A, M, and B:

$$
\begin{array} { r } { \widehat { { \bf V } } _ { t } = v _ { \theta } ( { \bf Z } _ { t } , t ; { \bf A } , { \bf M } , { \bf B } ) . } \end{array}\tag{4}
$$

At the DiT input, we add separate projections of $\mathbf { [ Z } _ { t } \lVert \mathbf { M } \rVert \mathbf { B } \mathbf { ] }$ and A, where ∥ denotes feature-wise concatenation. Padding is masked, and mean squared error between $\widehat { \mathbf { V } } _ { t }$ <sub>t</sub> and $\mathbf { V } ^ { \star }$ is minimised on target frames.

Inference. Euler integration starts from noise at $t = 0$ and reaches the final latent sequence $\hat { \mathbf { Z } }$ at $t = 1$ , with A, M, and B held fixed. At each step, observed frames follow their known interpolation in Eq. (3), recovering their original SQ codes at $t = 1$ . Intermediate flow states are not quantised, and only the final sequence is decoded as described in Sec. 2.1.

Optional FlowEdit. We adapt FlowEdit [12] from text-guided image editing to MIDI-guided audio editing without retraining. The edited latent sequence starts from before-edit SQ codes. Each step mixes these codes with fresh noise for the first DiT input. The second adds the difference between the current edited sequence and the before-edit codes to this noisy input. Both share A but use beforeedit and revised MIDI, respectively. An Euler step subtracts the first predicted vector field from the second to update only target frames, preserving observed SQ codes.

Table 1. Synthesis performance. Bold: best per group. $\mathrm { M S S \times 1 0 ^ { 3 } } .$ . F-scores in $\% . \longrightarrow : F _ { \mathrm { P 3 7 } } / F _ { \mathrm { P 1 3 } }$ omitted for single-instrument data.
<table><tr><td></td><td></td><td colspan="2">Audio Quality</td><td colspan="4">Audio Similarity</td><td colspan="4">Note Adherence</td></tr><tr><td>Dataset</td><td>Model</td><td> $\mathrm { M u Q } _ { \mathrm { e v a l } } \uparrow$ </td><td>PQ↑</td><td> $\mathrm { F A D } _ { \mathrm { M E R T } } \downarrow$ </td><td> $\mathrm { M u Q } _ { \mathrm { c o s } } \uparrow$ </td><td> $\mathrm { C L A P _ { c o s } \uparrow }$ </td><td>MSS↓</td><td> $F _ { \mathrm { O n } } \uparrow$ </td><td> $F _ { \mathrm { P 3 7 } } \uparrow$ </td><td> $F _ { \mathrm { P 1 3 } } \uparrow$ </td><td> $F _ { \mathrm { O p e n M I C } } \uparrow$ </td></tr><tr><td>Slakh</td><td>CTD†</td><td>4.20</td><td>7.49</td><td>2.00</td><td>0.81</td><td>0.78</td><td>36.46</td><td>46.91</td><td>33.31</td><td>38.36</td><td>12.25</td></tr><tr><td></td><td>TokenSynth†</td><td>4.10</td><td>7.27</td><td>4.63</td><td>0.73</td><td>0.51</td><td>129.45</td><td>42.13</td><td>24.46</td><td>31.34</td><td>16.42</td></tr><tr><td></td><td>Ours</td><td>4.67</td><td>8.02</td><td>1.41</td><td>0.89</td><td>0.86</td><td>12.11</td><td>50.88</td><td>29.57</td><td>37.61</td><td>19.30</td></tr><tr><td>+drums</td><td>SpecDiff</td><td>4.51</td><td>7.80</td><td>4.85</td><td>0.78</td><td>0.64</td><td>49.03</td><td>51.84</td><td>39.64</td><td>43.12</td><td>9.63</td></tr><tr><td></td><td>U-MusT</td><td>3.75</td><td>7.95</td><td>4.16</td><td>0.63</td><td>0.73</td><td>49.02</td><td>23.17</td><td>11.00</td><td>13.75</td><td>5.89</td></tr><tr><td></td><td>Ours</td><td>4.65</td><td>8.08</td><td>1.33</td><td>0.89</td><td>0.90</td><td>29.42</td><td>54.06</td><td>37.78</td><td>44.84</td><td>9.51</td></tr><tr><td>MusicNet</td><td>SpecDiff</td><td>4.38</td><td>6.87</td><td>3.50</td><td>0.90</td><td>0.75</td><td>25.21</td><td>68.25</td><td>66.79</td><td>67.24</td><td>71.67</td></tr><tr><td></td><td>U-MusT</td><td>4.16</td><td>7.09</td><td>4.03</td><td>0.84</td><td>0.84</td><td>30.84</td><td>44.31</td><td>41.19</td><td>42.53</td><td>75.00</td></tr><tr><td></td><td>Ours</td><td>4.75</td><td>7.60</td><td>3.15</td><td>0.92</td><td>0.87</td><td>22.02</td><td>76.56</td><td>71.94</td><td>74.69</td><td>68.33</td></tr><tr><td>URMP</td><td>SpecDiff</td><td>4.58</td><td>7.52</td><td>6.53</td><td>0.89</td><td>0.67</td><td>17.23</td><td>50.50</td><td>34.62</td><td>44.80</td><td>69.19</td></tr><tr><td></td><td>CTD†</td><td>3.91</td><td>7.13</td><td>10.89</td><td>0.77</td><td>0.55</td><td>77.02</td><td>35.39</td><td>2.89</td><td>22.56</td><td>6.67</td></tr><tr><td></td><td>Ours</td><td>4.82</td><td>7.90</td><td>4.95</td><td>0.91</td><td>0.83</td><td>9.50</td><td>54.24</td><td>37.85</td><td>47.90</td><td>53.00</td></tr><tr><td>GuitarSet</td><td>SpecDiff</td><td>4.06</td><td>7.89</td><td>5.92</td><td>0.89</td><td>0.71</td><td>17.76</td><td>83.67</td><td></td><td></td><td>80.83</td></tr><tr><td></td><td>CTD</td><td>3.56</td><td>7.95</td><td>8.67</td><td>0.76</td><td>0.57</td><td>48.41</td><td>47.37</td><td></td><td></td><td>20.00</td></tr><tr><td></td><td>TokenSynth</td><td>3.70</td><td>7.33</td><td>14.77</td><td>0.69</td><td>0.37</td><td>110.17</td><td>34.36</td><td></td><td></td><td>11.67</td></tr><tr><td></td><td>Ours</td><td>4.14</td><td>8.40</td><td>4.05</td><td>0.92</td><td>0.91</td><td>14.29</td><td>88.01</td><td></td><td></td><td>85.00</td></tr><tr><td>MAESTRO</td><td>SpecDiff</td><td>4.47</td><td>6.58</td><td>4.46</td><td>0.93</td><td>0.71</td><td>31.24</td><td>66.63</td><td></td><td></td><td>100.00</td></tr><tr><td>Piano V3</td><td>CTD</td><td>3.75</td><td>6.89</td><td>6.82</td><td>0.74</td><td>0.66</td><td>49.22</td><td>16.67</td><td></td><td></td><td>96.67</td></tr><tr><td></td><td>TokenSynth</td><td>3.95</td><td>6.99</td><td>9.00</td><td>0.79</td><td>0.47</td><td>63.15</td><td>22.90</td><td></td><td></td><td>60.00</td></tr><tr><td></td><td>MIDI-VALLE</td><td>4.37</td><td>7.64</td><td>3.67</td><td>0.93</td><td>0.86</td><td>23.56</td><td>58.27</td><td></td><td></td><td>100.00</td></tr><tr><td></td><td>U-MusT</td><td>4.35</td><td>7.24</td><td>2.59</td><td>0.91</td><td>0.86</td><td>30.58</td><td>36.73</td><td></td><td></td><td>100.00</td></tr><tr><td></td><td>Ours</td><td>4.61</td><td>7.50</td><td>2.89</td><td>0.96</td><td>0.86</td><td>20.23</td><td>76.49</td><td></td><td></td><td>100.00</td></tr></table>

## 3. EXPERIMENTS

## 3.1. Experimental Setup

Tasks and data. For synthesis, we test whether generated audio follows target MIDI and joins surrounding audio smoothly with matching timbres and acoustics (Table 1). For editing, we test preservation of original timbres and unchanged notes under revised MIDI (Table 2). Each edit provides paired MIDI and ground-truth audio before and after revision. Adding or removing stems or stem segments from Slakh test songs [21] yields 668 original–revised pairs. When adding a new instrument, the reference audio excludes it. Rendering original and Modulator-revised MIDI [20] for 147 POP909 songs [22] with the Salamander piano SoundFont [23] yields 882 such pairs, half dry and half with matched reverberation. Context is unchanged and excluded from evaluation.

Training and inference. We train on 660 h of performance MIDI paired with 48-kHz mono audio from 17 public instrumental datasets (15 with real recordings). We use Adam for 125k steps (effective batch 80, peak learning rate $1 0 ^ { - 4 }$ , linear warmup and decay). In Tables $1 { - } 2 .$ , our model uses contextual audio and MIDI, 32 Euler steps, and classifier-free guidance (CFG 2). On a GH200 (BF16, batch 1), MIDI-only generation with SQ decoding takes 0.93 s per 20.48 s of audio, with 3.37 GiB peak reserved memory.

Baselines. We prioritise baselines trained on the corresponding dataset and use official checkpoints. For POP909, all models, including ours, were trained on piano audio but not the Salamander timbre. For multi-instrument results (†), we give CTD [4] and TokenSynth [6] isolated stem context and mix their generated stems at fixed gains without rebalancing. SpecDiff [8] and U-MusT [10] generate mixtures, while MIDI-VALLE [15] is piano-only. All three use only preceding audio, with contextual MIDI only for U-MusT and MIDI-VALLE.

Metrics. Audio Quality evaluates generated audio alone using MuQ-Eval’s musical impression score [24] and Audiobox production quality (PQ) [25]. Audio Similarity compares generated and ground-truth audio using MERT-based FAD [26, 27], MuQ/CLAP cosine similarity [28, 29], and Smooth MSS spectral error [30]. Note Adherence compares YourMT3+ [31] transcriptions of generated audio with target MIDI at a 50-ms onset tolerance. $F _ { \mathrm { O n } }$ is instrument-agnostic note onset F1, while Multi Onset F1 also matches instrument labels using 37-class fine $\left( F _ { \mathrm { P 3 7 } } \right)$ and 13-class coarse $\left( F _ { \mathrm { P 1 3 } } \right)$ vocabularies [31]. For editing, 37-class matching measures recall of requested additions $( R _ { \mathrm { a d d } } )$ and F1 of unchanged notes $( F _ { \mathrm { k e e p } } )$ . For $F _ { \mathrm { k e e p } } ,$ precision excludes predictions matched to requested additions or deletions. $R _ { \mathrm { d e l } }$ is the fraction of notes requested for deletion that remain detected. $F _ { \mathrm { O p e n M I C } }$ measures 20-class instrument-presence F1 [32]. Our repository will include dataset references and 95% bootstrap confidence intervals for per-example metrics.

## 3.2. Synthesis and Editing

Synthesis. SpanSynth-Edit (denoted Ours in Tables 1–2) matches or outperforms the reported baselines on the audio-quality and audiosimilarity metrics in Table 1, except for PQ and FAD on MAESTRO. There, MIDI-VALLE, a piano-only model trained on a large piano dataset [15], leads in PQ, while U-MusT leads in FAD. In each Table 1 group, our model’s $F _ { \mathrm { O n } }$ exceeds the highest baseline score by 2.22–9.86 percentage points, but this advantage is inconsistent for scores requiring instrument-label matches. On Slakh (+drums), our model exceeds SpecDiff in $F _ { \mathrm { P 1 3 } }$ but falls behind in $F _ { \mathrm { P 3 7 } }$

Editing. Our model without FlowEdit outperforms the reported baselines in MuQ-Eval and PQ in every Table 2 group. Applying FlowEdit (denoted Ours + FlowEdit) yields the best FAD and MSS among the compared systems in each group but reduces MuQ-Eval relative to our model without FlowEdit.

The highest $R _ { \mathrm { a d d } }$ and $F _ { \mathrm { k e e p } }$ scores in each Table 2 group come from our model with or without FlowEdit. However, our models obtain worse $R _ { \mathrm { d e l } }$ scores than CTD on non-drum Slakh and SpecDiff on Slakh (+drums) and POP909. Relative to Ours, Ours + FlowEdit increases $R _ { \mathrm { d e l } }$ in every Table 2 deletion group and decreases $R _ { \mathrm { a d d } }$ in three of four groups. Its effect on $F _ { \mathrm { k e e p } }$ varies by task.

Note-adherence scores assess whether audio follows MIDI but also reflect transcription errors, especially in multi-instrument mixtures.

Table 2. Editing performance. R- and F-scores in $\% . - .$ no requested deletions. Other notation follows Table 1.
<table><tr><td rowspan="2">Dataset / Task</td><td rowspan="2">Model</td><td colspan="2">Audio Quality</td><td colspan="4">Audio Similarity</td><td colspan="4">Note Adherence</td></tr><tr><td> $\mathrm { M u Q } _ { \mathrm { e v a l } } \uparrow$ </td><td>PQ↑</td><td> $\mathrm { F A D } _ { \mathrm { M E R T } } \downarrow$ </td><td> $\mathrm { M u Q } _ { \mathrm { c o s } } \uparrow$ </td><td> $\mathrm { C L A P _ { c o s } \uparrow }$ </td><td>MSS↓</td><td> $R _ { \mathrm { a d d } } \uparrow$ </td><td> $R _ { \mathrm { d e l } } \downarrow$ </td><td> $F _ { \mathrm { k e e p } } \uparrow$ </td><td> $F _ { \mathrm { O p e n M I C } } \uparrow$ </td></tr><tr><td>Slakh (+drums)</td><td>SpecDiff</td><td>4.29</td><td>7.71</td><td>4.87</td><td>0.79</td><td>0.63</td><td>45.04</td><td>43.68</td><td></td><td>32.54</td><td>12.86</td></tr><tr><td>Add new</td><td>U-MusT</td><td>3.72</td><td>7.98</td><td>3.45</td><td>0.70</td><td>0.76</td><td>41.39</td><td>21.32</td><td></td><td>8.34</td><td>11.66</td></tr><tr><td>instrument</td><td>Ours</td><td>4.44</td><td>8.13</td><td>1.68</td><td>0.87</td><td>0.84</td><td>31.54</td><td>49.13</td><td></td><td>31.09</td><td>14.73</td></tr><tr><td></td><td>Ours + FlowEdit</td><td>4.34</td><td>8.01</td><td>1.43</td><td>0.88</td><td>0.83</td><td>24.40</td><td>42.14</td><td></td><td>32.68</td><td>14.85</td></tr><tr><td>Slakh</td><td>CTD†</td><td>3.92</td><td>7.28</td><td>1.82</td><td>0.82</td><td>0.72</td><td>80.55</td><td>43.20</td><td>3.82</td><td>28.91</td><td>17.33</td></tr><tr><td>Note ins/del</td><td>Ours</td><td>4.40</td><td>7.88</td><td>1.25</td><td>0.90</td><td>0.84</td><td>21.37</td><td>42.51</td><td>6.08</td><td>33.24</td><td>26.56</td></tr><tr><td></td><td>Ours + FlowEdit</td><td>4.32</td><td>7.76</td><td>1.06</td><td>0.91</td><td>0.86</td><td>11.15</td><td>43.49</td><td>8.01</td><td>35.75</td><td>24.56</td></tr><tr><td>+drums</td><td>SpecDiff</td><td>4.25</td><td>7.72</td><td>4.15</td><td>0.80</td><td>0.62</td><td>40.33</td><td>52.64</td><td>2.62</td><td>39.19</td><td>15.48</td></tr><tr><td></td><td>U-MusT</td><td>3.72</td><td>7.95</td><td>2.65</td><td>0.70</td><td>0.74</td><td>37.81</td><td>20.50</td><td>3.72</td><td>9.56</td><td>13.28</td></tr><tr><td></td><td>Ours</td><td>4.44</td><td>8.03</td><td>1.03</td><td>0.89</td><td>0.87</td><td>23.26</td><td>56.44</td><td>4.56</td><td>41.90</td><td>16.92</td></tr><tr><td></td><td>Ours + FlowEdit</td><td>4.36</td><td>7.93</td><td>0.83</td><td>0.91</td><td>0.88</td><td>12.65</td><td>50.54</td><td>7.78</td><td>41.55</td><td>16.21</td></tr><tr><td>POP909 (piano)</td><td>SpecDiff</td><td>4.35</td><td>7.47</td><td>5.46</td><td>0.87</td><td>0.76</td><td>30.92</td><td>83.03</td><td>2.60</td><td>12.85</td><td>98.85</td></tr><tr><td>AI edits [20]</td><td>CTD</td><td>3.66</td><td>7.42</td><td>7.99</td><td>0.72</td><td>0.65</td><td>45.11</td><td>30.16</td><td>6.74</td><td>3.23</td><td>97.03</td></tr><tr><td></td><td>TokenSynth</td><td>3.79</td><td>7.41</td><td>9.21</td><td>0.74</td><td>0.63</td><td>87.71</td><td>27.24</td><td>5.27</td><td>7.66</td><td>80.50</td></tr><tr><td></td><td>MIDI-VALLE</td><td>4.13</td><td>7.95</td><td>5.19</td><td>0.86</td><td>0.73</td><td>23.93</td><td>72.74</td><td>4.86</td><td>10.41</td><td>99.74</td></tr><tr><td></td><td>U-MusT</td><td>4.26</td><td>7.78</td><td>6.50</td><td>0.84</td><td>0.81</td><td>29.00</td><td>42.56</td><td>5.39</td><td>3.77</td><td>99.96</td></tr><tr><td></td><td>Ours</td><td>4.42</td><td>7.99</td><td>3.33</td><td>0.92</td><td>0.77</td><td>16.82</td><td>89.86</td><td>3.40</td><td>13.84</td><td>99.81</td></tr><tr><td></td><td>Ours + FlowEdit</td><td>4.36</td><td>8.03</td><td>2.31</td><td>0.91</td><td>0.78</td><td>15.64</td><td>86.81</td><td>5.06</td><td>12.46</td><td>99.89</td></tr></table>

Table 3. Tracing note-adherence error sources on Slakh (+drums). Scores in %; ∆: drop from the preceding row (percentage points). $\pmb { \bigtriangledown } : \geq$ 10; ▽: < 10. AMT: automatic music transcription.
<table><tr><td>(a) Slakh / Synthesis</td><td> $F _ { \mathrm { P 3 7 } } \uparrow ( \Delta )$ </td><td> $F _ { \mathrm { O n } } \uparrow \left( \Delta \right)$ </td></tr><tr><td>Ground Truth (GT) w/ Perfect AMT</td><td>100</td><td>100</td></tr><tr><td>GT w/ AMT</td><td>78.40 (21.60)</td><td>82.79(17.21)</td></tr><tr><td>— Reconstruction (no SQ) w/ AMT</td><td>44.43(33.97)</td><td>55.94(26.85)</td></tr><tr><td>— Reconstruction (SQ) w/ AMT</td><td>43.98 (∇ 0.45)</td><td>55.79(∇ 0.15)</td></tr><tr><td>— Generated (ours)</td><td>37.78(∇ 6.20)</td><td>54.06(∇ 1.73)</td></tr><tr><td>(b) Slakh / Add new instrument</td><td> $R _ { \mathrm { a d d } } \uparrow ( \Delta )$ </td><td> $F _ { \mathrm { k e e p } } \uparrow ( \Delta )$ </td></tr><tr><td>Ground Truth (GT) w/ Perfect AMT</td><td>100</td><td>100</td></tr><tr><td>-GT w/ AMT</td><td>71.40(28.60)</td><td>72.80 (27.20)</td></tr><tr><td>— Reconstruction (no SQ) w/ AMT</td><td>55.99(15.41)</td><td> $3 9 . 0 1 ( \pmb { \bigtriangledown } 3 3 . 7 9 )$ </td></tr><tr><td>— Reconstruction (SQ) w/ AMT</td><td>54.54(∇ 1.45)</td><td>37.50(∇ 1.51)</td></tr><tr><td>— Generated (ours)</td><td>49.13 (∇ 5.41)</td><td>31.09(∇ 6.41)</td></tr></table>

Table 4. Inference-time ablations. CM: contextual MIDI. CA: clean audio condition A. Bold: best per group.
<table><tr><td rowspan="3"></td><td colspan="5">Euler steps (Base)</td><td colspan="4">32-step ablations</td></tr><tr><td>4</td><td>8</td><td>16</td><td>32</td><td>64</td><td>Base</td><td>-CM</td><td>-CA</td><td>-CFG</td></tr><tr><td> $\mathrm { M u Q } _ { \mathrm { e v a l } } \uparrow$ </td><td>4.48</td><td>4.46</td><td>4.43</td><td>4.41</td><td>4.40</td><td>4.41</td><td>4.44</td><td>3.48</td><td>4.29</td></tr><tr><td> $\mathrm { F A D } _ { \mathrm { M E R T } } \downarrow$ </td><td>3.65</td><td>3.29</td><td>3.09</td><td>2.99</td><td>2.95</td><td>2.99</td><td>3.36</td><td>9.95</td><td>2.98</td></tr><tr><td> $F _ { \mathrm { O n } } ~ ( \% ) \uparrow$ </td><td>74.60</td><td>76.14</td><td>75.90</td><td>73.34</td><td>73.03</td><td>73.34</td><td>76.01</td><td>45.66</td><td>65.67</td></tr></table>

## 3.3. Analysis

Note-adherence errors. In Table 3(a,b), we compare ground-truth, reconstructed, and generated audio to break down the main sources of error. We observe substantial note-adherence score losses originating from the YourMT3+ transcription model itself (F : 21.60, $R _ { \mathrm { a d d } } \colon 2 8 . 6 0 )$ . This is especially pronounced in mixtures with many instruments. Additional losses arise from encoder-decoder reconstruction without $\mathrm { S Q } \left( F _ { \mathrm { P 3 7 } } \mathrm { : } 3 3 . 9 7 , R _ { \mathrm { a d d } } \mathrm { : } 1 5 . 4 1 \right)$ . Quantisation adds only small losses $( F _ { \mathrm { P 3 7 } } \colon 0 . 4 5 , R _ { \mathrm { a d d } } \colon 1 . 4 5 )$ . Transcription and reconstruction without SQ account for most losses across all four metrics, with smaller losses from generation. Reconstruction losses may reflect a distribution mismatch between HeartCodec’s pretraining data and ours.

Timing sensitivity. We test the timing precision sensitivity of the synthesised audio to the MIDI span encoder by delaying MIDI onsets by 1–40 ms. We quantify the deviations between the synthesised reference and time-shifted re-synthesised audio via a crosscorrelation of pitch harmonic energy (or drum-onset curves [33] with peak interpolation). With a perfect MIDI encoder and synthesis model, a 5 ms onset shift, for example, will reflect a 5 ms shift in the output, representing the dotted diagonal in Figure 2. In Fig. 2, although the model is not sensitive to small onset shifts in the range of 1–5 ms, possibly due to annotation errors, it starts becoming receptive at noticeable shifts from 10–40 ms. Strings and Pipe show weak measured responses, possibly reflecting difficulties in localising non-percussive onsets [33].

![](images/dcfc08f1e5e05f44455957ec6f2363858ac1ea0c6206c7a171c0a940b3a4f0d8.jpg)  
Fig. 2. Median acoustic responses to 1–40 ms MIDI onset shifts. Dotted lines: equal shifts. Zoom: 1–4 ms.

Ablations. Table 4 reveals a trade-off beyond 8 Euler steps: FAD decreases while onset F1 and MuQ-Eval decline. At 32 steps, contextual MIDI provides no consistent benefit. Removing clean contextual audio (CA; A) severely degrades all three metrics. These results strongly support our design choice of conditioning on clean contextual audio latents (Sec. 2.1). Removing CFG also reduces MuQ-Eval and onset F1 with little FAD change.

## 4. CONCLUSION

SpanSynth-Edit generates and edits multi-instrument mixtures with SQ latents and MIDI Span. On synthesis and editing benchmarks, our model outperforms the evaluated baselines on most objective audio-quality and audio-similarity metrics, while achieving comparable note adherence. We analysed the origin of note-adherence errors, timing precision of MIDI span to the generated output, and support for contextual audio conditioning. Codec limitations motivate future decoder fine-tuning. We leave listening tests with music experts for future work. Audio examples are on our project page<sup>1</sup>.

## 5. ACKNOWLEDGEMENT

Keshav Bhandari was supported by the UKRI Centre for Doctoral Training in Artificial Intelligence and Music (EP/S022694/1). EmotionWave provided GPU resources for part of the model training. We thank TaeGyun Kwon and Dabin Kim for their helpful feedback.

## 6. REFERENCES

[1] G. Le Lan et al., “High fidelity text-guided music editing via single-stage flow matching,” arXiv:2407.03648, 2024.

[2] Y. Zhang et al., “Instruct-musicgen: Unlocking text-to-music editing for music language models via instruction tuning,” in Proc. ISMIR, 2025, pp. 328–336.

[3] K. Bhandari et al., “ImprovNet - generating controllable musical improvisations with iterative corruption refinement,” in Proc. IJCNN, 2025, pp. 1–10.

[4] N. Demerle, P. Esling, G. Doras, and D. Genova, “Combining´ audio control and style transfer using latent diffusion,” in Proc. ISMIR, 2024, pp. 721–728.

[5] C. Jing, J. Zhang, J. Yang, Y. Wu, F. Fan, and Z. Wu, “P-MUSE: Prompt-MIDI-optional model for unified instrumental music synthesis and editing,” arXiv:2608.01920, 2026.

[6] K. Kim, J. Koo, S. Lee, H. Joung, and K. Lee, “TokenSynth: A token-based neural synthesizer for instrument cloning and text-to-instrument,” in Proc. IEEE ICASSP, 2025.

[7] Q. Yang, R. Leistikow, and Y. Zang, “Instrument generation through distributional flow matching and test-time search,” in Proc. IEEE ICASSP, 2026.

[8] C. Hawthorne et al., “Multi-instrument music synthesis with spectrogram diffusion,” in Proc. ISMIR, 2022, pp. 598–607.

[9] B. Maman, J. Zeitler, M. Muller, and A. H. Bermano, “Multi-¨ aspect conditioning for diffusion-based music synthesis: Enhancing realism and acoustic control,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 33, pp. 68–81, 2025.

[10] J. Jung et al., “U-MusT: A unified framework for cross-modal translation of score images, symbolic music, and performance audio,” IEEE Trans. Audio, Speech, Lang. Process., vol. 34, pp. 1876–1891, 2026.

[11] C. H. Lee, J. Nistal, S. Lattner, M. Pasini, and G. Fazekas, “Diffusion timbre transfer via mutual information guided inpainting,” in Proc. IEEE ICASSP, 2026, pp. 15077–15081.

[12] V. Kulikov, M. Kleiner, I. Huberman-Spiegelglas, and T. Michaeli, “FlowEdit: Inversion-free text-based editing using pre-trained flow models,” in Proc. IEEE/CVF ICCV, 2025, pp. 19721–19730.

[13] D. Yang et al., “SimpleSpeech 2: Towards simple and efficient text-to-speech with flow-based scalar latent transformer diffusion models,” IEEE Trans. Audio, Speech, Lang. Process., vol. 33, pp. 2634–2646, 2025.

[14] D. Yang et al., “HeartMuLa: A family of open-sourced music foundation models,” arXiv:2601.10547, 2026.

[15] J. Tang, X. Wang, Z. Zhang, J. Yamagishi, G. Wiggins, and G. Fazekas, “MIDI-VALLE: Improving expressive piano performance synthesis through neural codec language modelling,” in Proc. ISMIR, 2025, pp. 623–630.

[16] I. Borovik, D. Gavrilev, and V. Viro, “SyMuPe: Affective and controllable symbolic music performance,” in Proc. ACM MM, 2025, pp. 10699–10708.

[17] M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhut-´ dinov, and A. J. Smola, “Deep sets,” in Adv. Neural Inf. Process. Syst., 2017, vol. 30, pp. 3391–3401.

[18] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in Proc. ICLR, 2023.

[19] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proc. IEEE/CVF ICCV, 2023, pp. 4195–4205.

[20] K. Bhandari, M. Bizzarri, G. A. Wiggins, and S. Colton, “Change is Key: A generative framework for controllable musical modulations,” Hugging Face model repository, 2026, https://huggingface.co/keshavbhandari/ modulator.

[21] E. Manilow, G. Wichern, P. Seetharaman, and J. Le Roux, “Cutting music source separation some Slakh: A dataset to study the impact of training data quality and quantity,” in Proc. IEEE WASPAA, 2019, pp. 45–49.

[22] Z. Wang et al., “POP909: A pop-song dataset for music arrangement generation,” in Proc. ISMIR, 2020, pp. 38–45.

[23] A. Holm, “Salamander Grand Piano,” FreePats, SoundFont version V3+2020-06-02, 2020, https://freepats. zenvoid.org/Piano/acoustic-grand-piano. html.

[24] D. Zhu and Z. Li, “MuQ-Eval: An open-source persample quality metric for AI music generation evaluation,” arXiv:2603.22677, 2026.

[25] A. Tjandra et al., “Meta Audiobox Aesthetics: Unified automatic assessment for speech, music and sound,” in Proc. IEEE ASRU, 2025, pp. 1–8.

[26] K. Kilgour, M. Zuluaga, D. Roblek, and M. Sharifi, “Frechet´ audio distance: A reference-free metric for evaluating music enhancement algorithms,” in Proc. Interspeech, 2019, pp. 2350–2354.

[27] Y. Li et al., “MERT: Acoustic music understanding model with large-scale self-supervised training,” in Proc. ICLR, 2024.

[28] H. Zhu et al., “MuQ: Self-supervised music representation learning with mel residual vector quantization,” IEEE Trans. Audio, Speech, Lang. Process., vol. 33, pp. 3653–3664, 2025.

[29] Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov, “Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation,” in Proc. IEEE ICASSP, 2023, pp. 1–5.

[30] S. Schwar and M. M ¨ uller, “Multi-scale spectral loss revisited,”¨ IEEE Signal Process. Lett., vol. 30, pp. 1712–1716, 2023.

[31] S. Chang, E. Benetos, H. Kirchhoff, and S. Dixon, “YourMT3+: Multi-instrument music transcription with enhanced transformer architectures and cross-dataset stem augmentation,” in Proc. IEEE MLSP, 2024.

[32] K. Koutini, J. Schluter, H. Eghbal-zadeh, and G. Widmer,¨ “Efficient training of audio transformers with patchout,” in Proc. Interspeech, 2022, pp. 2753–2757, OpenMIC checkpoint (v0.0.5).

[33] J. P. Bello, L. Daudet, S. Abdallah, C. Duxbury, M. Davies, and M. B. Sandler, “A tutorial on onset detection in music signals,” IEEE Trans. Speech Audio Process., vol. 13, no. 5, pp. 1035–1047, 2005.

## A. DATASET LIST, REFERENCES, AND SPLITS

We train on the 17 instrumental dataset families below, using instrumental stems where applicable. MusicNet denotes MusicNetEM, with revised note annotations for the original MusicNet recordings.

<table><tr><td>Dataset</td><td></td><td>Audio content</td></tr><tr><td>1</td><td>Slakh [A1]</td><td>Synthesised multi-instrument mixtures, including drums</td></tr><tr><td>2</td><td>AAM [A2]</td><td>Artificial audio multitracks</td></tr><tr><td>3</td><td>MAESTRO [A3]</td><td>Solo piano recordings, mostly classical</td></tr><tr><td>4</td><td>PianoVAM [A4]</td><td>Piano performances</td></tr><tr><td>5</td><td>MusicNetEM [A5, A6]</td><td>Classical instrumental recordings</td></tr><tr><td>6</td><td>BSD / BSED [A7]</td><td>Beethoven symphony recordings / evaluation excerpts</td></tr><tr><td>7</td><td>GOAT [A8]</td><td>Electric guitar recordings with clean and distorted tones</td></tr><tr><td>8</td><td>GuitarSet [A9]</td><td>Acoustic guitar: jazz, rock, funk, bossa nova, singer-songwriter</td></tr><tr><td>9</td><td>URMP [A10]</td><td>Classical ensembles of 2–5 separately recorded parts</td></tr><tr><td>10</td><td>IDMT-SMT-Bass [A11]</td><td>Electric bass recordings</td></tr><tr><td>11</td><td>KRAISLER [A12]</td><td>Piano and violin duet recordings</td></tr><tr><td>12</td><td>ChoraleBricks [A13]</td><td>Wind-instrument multitracks</td></tr><tr><td>13</td><td>FiloBass [A14]</td><td>Jazz bass recordings</td></tr><tr><td>14</td><td>MDB Drums [A15]</td><td>Annotated drum stems</td></tr><tr><td>15</td><td>EGSet12 [A16]</td><td>Solo electric guitar performances</td></tr><tr><td>16</td><td>ENST-Drums [A17]</td><td>Drum performances</td></tr><tr><td>17</td><td>STAR Drums [A18]</td><td>Drum transcription data</td></tr></table>

Official partitions. Slakh and MAESTRO combine their official training and validation sets for training. PianoVAM uses its training and extended-training sets, while GOAT and STAR Drums use their official training sets without adding validation data. Their official test sets are reserved for evaluation. MDB Drums follows the published MIREX partition of 12 training and 11 test songs.

Adopted partitions. MusicNetEM uses the extended ten-recording test set rather than the original three-recording test set. These ten recordings are excluded from both acoustic and synthesised training versions. URMP follows the MT3 split [A19], with 35 training and nine test pieces. GuitarSet uses progressions 1 and 2 for training and progression 3 for evaluation. ENST-Drums uses drummers 1 and 2 for training and drummer 3 for evaluation. IDMT-SMT-Bass retains the stored approximately 80:20 training/validation partition, with the latter reserved for evaluation.

Custom partitions. AAM, ChoraleBricks, FiloBass, and EGSet12 use our approximately 8:1:1 training/validation/test splits, assigned at the song or recording level. Only the training partition is included in the training pool. KRAISLER holds out tracks 05, 10, 15, and 20 for evaluation, using their studio and hall mixtures. The remaining 16 tracks provide training stems from all three room conditions.

BSD and BSED. BSD provides the training recordings, while BSED provides evaluation excerpts. These collections share symphonies and score passages, so this is not a composition-disjoint partition. The prepared BSED evaluation selects recordings not used for training and supplements them with synthesised renditions.

For editing, POP909 [A20] supplies piano arrangements of pop songs. Original and Modulator-revised MIDI [A21] are rendered with the Salamander piano SoundFont [A22] (Sec. 3.1).

Audio preprocessing and SQ storage. Audio is downmixed to mono and resampled to 48 kHz as needed, including upsampling 44.1-kHz sources. We extract 20.48-s crops at 5.12-s intervals (75% overlap), adding a final end-aligned crop and zero-padding short recordings. The frozen HeartCodec encoder also receives 2 s of preceding audio, zero-padded when unavailable. Its first 50 latent frames are discarded, and the remaining 512 × 128 SQ codes are stored as signed 8-bit integers.

Augmentation. We precompute mixtures retaining the base arrangement with up to two additional recordings. Training requests zero, one, or two additions in 50%, 35%, and 15% of samples. Conflicting instrument parts are excluded, and additions exceeding the 128-event frame budget are rejected. Eligible drum-free crops receive length-preserving Rubber Band R3 pitch shifts of ±1 or ±2 semitones, with matched MIDI transposition before SQ encoding. Training requests shifted crops 25% of the time, falling back to unshifted crops when no eligible shifted version is available.

## B. SYNTHESIS RESULTS WITH CONFIDENCE INTERVALS (TABLE 1)

Reporting convention. Each per-example metric is reported as mean [95% CI lower bound, upper bound]. Intervals use a source-recording cluster bootstrap with 10,000 resamples and the 2.5th and 97.5th percentiles. Each resampled recording contributes all its evaluated excerpts, and the mean is taken over excerpts. These intervals quantify variation across the evaluation recordings, not across training runs, and are not tests of paired differences between models. FAD is a distributional score computed from pooled features, so only its point estimate is shown. MSS is multiplied by 10<sup>3</sup>, and all note and instrument scores are percentages. Cosine similarities are shown to three decimal places.

Comparison conditions. Models and conditioning follow Table 1 and Sec. 3.1. A dagger denotes separate stem generation followed by mixing. Each Slakh group has 100 excerpts from 100 recordings. MusicNetEM has 20 excerpts from 10 recordings, URMP has 20 from 9, and GuitarSet and MAESTRO each have 20 from 20. The small number of recordings limits the precision of the MusicNetEM and URMP intervals. The single-instrument tables omit $F _ { \mathrm { P 3 7 } }$ and $F _ { \mathrm { P 1 3 } } ,$ , as in the main paper.

## B.1. Slakh (without drums)

<table><tr><td>Model</td><td> $\mathrm { M u Q _ { e v a l } }$  ←</td><td>PQ↑</td><td>FADMERT ↓</td></tr><tr><td>CTD†</td><td>4.20 [4.14, 4.27]</td><td>7.49 [7.39, 7.59]</td><td>2.00</td></tr><tr><td>TokenSynth†</td><td>4.10 [4.03, 4.16]</td><td>7.27 [7.18, 7.35]</td><td>4.63</td></tr><tr><td>Ours</td><td>4.67 [4.60, 4.74]</td><td>8.02 [7.96, 8.07]</td><td>1.41</td></tr></table>

<table><tr><td>Model</td><td> $\mathrm { M u Q } _ { \mathrm { c o s } } \uparrow$ </td><td> $\mathrm { C L A P _ { c o s } \uparrow }$ </td><td>MSS↓</td></tr><tr><td>CTD†</td><td>0.813 [0.796, 0.829]</td><td>0.778 [0.763, 0.792]</td><td>36.46 [35.03, 37.94]</td></tr><tr><td>TokenSynth†</td><td>0.733 [0.716, 0.749]</td><td>0.513 [0.495, 0.531]</td><td>129.45 [123.31, 135.62]</td></tr><tr><td>Ours</td><td>0.890 [0.880, 0.899]</td><td>0.856 [0.839, 0.871]</td><td>12.11 [11.49, 12.72]</td></tr></table>

<table><tr><td>Model</td><td> $F _ { \mathrm { O n } }$  ↑</td><td> $F _ { \mathrm { P 3 7 } } \uparrow$ </td><td> $F _ { \mathrm { P 1 3 } } \uparrow$ </td><td> $F _ { \mathrm { O p e n M I C } } \uparrow$ </td></tr><tr><td>CTD{†</td><td>46.91 [44.52, 49.23]</td><td>33.31 [30.96, 35.69]</td><td>38.36 [35.84, 40.81]</td><td>12.25 [8.64, 15.99]</td></tr><tr><td>TokenSynth†</td><td>42.13 [39.96, 44.27]</td><td>24.46 [22.38, 26.49]</td><td>31.34 [29.22, 33.46]</td><td>16.42 [12.39, 20.64]</td></tr><tr><td>Ours</td><td>50.88 [48.45, 53.30]</td><td>29.57 [26.80, 32.41]</td><td>37.61 [34.98, 40.29]</td><td>19.30 [15.09, 23.57]</td></tr></table>

## B.2. Slakh (+drums)

<table><tr><td>Model</td><td> $\mathrm { M u Q } _ { \mathrm { e v a l } }$  ←</td><td>PQ↑</td><td> $\mathrm { F A D } _ { \mathrm { M E R T } } \downarrow$ </td></tr><tr><td>SpecDiff</td><td>4.51 [4.44, 4.57]</td><td>7.80 [7.74, 7.85]</td><td>4.85</td></tr><tr><td>U-MusT</td><td>3.75 [3.67, 3.83]</td><td>7.95 [7.89, 8.02]</td><td>4.16</td></tr><tr><td>Ours</td><td>4.65 [4.59, 4.70]</td><td>8.08 [8.04, 8.12]</td><td>1.33</td></tr></table>

<table><tr><td>Model</td><td> $\mathrm { M u Q } _ { \mathrm { c o s } } \uparrow$ </td><td> $\mathrm { C L A P _ { c o s } \uparrow }$ </td></tr><tr><td>SpecDiff</td><td>0.778 [0.764, 0.790]</td><td>0.637 [0.617, 0.657] 49.03 [45.87, 52.62]</td></tr><tr><td>U-MusT</td><td>0.627 [0.600, 0.653]</td><td>0.734 [0.712, 0.754] 49.02 [46.04, 52.16]</td></tr><tr><td>Ours</td><td>0.889 [0.880, 0.897]</td><td>0.895 [0.883, 0.907] 29.42 [28.13, 30.71]</td></tr></table>

<table><tr><td>Model</td><td> $F _ { \mathrm { O n } }$  ←</td><td> $F _ { \mathrm { P 3 7 } } \uparrow$ </td><td> $F _ { \mathrm { P 1 3 } } \uparrow$ </td><td> $F _ { \mathrm { O p e n M I C } } \uparrow$ </td></tr><tr><td>SpecDiff</td><td>51.84 [49.18, 54.43]</td><td>39.64 [36.62, 42.68]</td><td>43.12 [40.24, 46.01]</td><td>9.63 [7.00, 12.39]</td></tr><tr><td>U-MusT</td><td>23.17 [21.52, 24.98]</td><td>11.00 [9.57, 12.52]</td><td>13.75 [12.20, 15.40]</td><td>5.89 [3.83, 8.18]</td></tr><tr><td>Ours</td><td>54.06 [51.59, 56.48]</td><td>37.78 [35.17, 40.40]</td><td>44.84 [42.19, 47.40]</td><td>9.51 [6.88, 12.28]</td></tr></table>

B.3. MusicNetEM
<table><tr><td>Model</td><td> $\mathrm { M u Q } _ { \mathrm { e v a l } } \uparrow$  PQ↑</td><td> $\mathrm { F A D } _ { \mathrm { M E R T } } \downarrow$ </td></tr><tr><td>SpecDiff</td><td>4.38 [4.18, 4.61]</td><td>6.87 [6.43, 7.26] 3.50</td></tr><tr><td>U-MusT</td><td>4.16 [3.85, 4.49] 7.09 [6.76, 7.40]</td><td>4.03</td></tr><tr><td>Ours</td><td>4.75 [4.48, 5.00] 7.60 [7.30, 7.87]</td><td>3.15</td></tr></table>

<table><tr><td>Model</td><td> $\mathrm { M u Q } _ { \mathrm { c o s } } \uparrow$ </td><td> $\mathrm { C L A P _ { c o s } \uparrow }$ </td><td>MSS↓</td></tr><tr><td>SpecDiff</td><td>0.896 [0.873, 0.917]</td><td>0.745 [0.696, 0.789]</td><td>25.21 [18.54, 32.35]</td></tr><tr><td>U-MusT</td><td>0.837 [0.797, 0.876]</td><td>0.842 [0.794, 0.881]</td><td>30.84 [21.69, 42.02]</td></tr><tr><td>Ours</td><td>0.921 [0.906, 0.934]</td><td>0.869 [0.830, 0.899]</td><td>22.02 [16.21, 28.54]</td></tr></table>

<table><tr><td>Model</td><td> $F _ { \mathrm { O n } }$  ←</td><td> $F _ { \mathrm { P 3 7 } } \uparrow$ </td><td> $F _ { \mathrm { P 1 3 } }$  ←</td><td> $F _ { \mathrm { O p e n M I C } } \uparrow$ </td></tr><tr><td>SpecDiff</td><td>68.25 [57.59, 79.36]</td><td>66.79 [55.45, 78.54]</td><td>67.24 [56.21, 78.79]</td><td>71.67 [46.67, 93.33]</td></tr><tr><td>U-MusT</td><td>44.31 [36.06, 53.93]</td><td>41.19 [31.90, 51.88]</td><td>42.53 [33.54, 52.79]</td><td>75.00 [48.33, 96.67]</td></tr><tr><td>Ours</td><td>76.56 [69.55, 84.40]</td><td>71.94 [62.55, 82.21]</td><td>74.69 [66.49, 83.60]</td><td>68.33 [43.33, 90.00]</td></tr></table>

## B.4. URMP

<table><tr><td>Model</td><td> $\mathrm { M u Q } _ { \mathrm { e v a l } } \uparrow$  PQ↑</td><td>FADMERT ↓</td></tr><tr><td>SpecDiff</td><td>4.58 [4.48, 4.65]</td><td>7.52 [7.25, 7.70] 6.53</td></tr><tr><td>CTD†</td><td>3.91 [3.71, 4.13] 7.13 [6.98, 7.28]</td><td>10.89</td></tr><tr><td>Ours</td><td>4.82 [4.67, 4.96] 7.90 [7.67, 8.07]</td><td>4.95</td></tr></table>

<table><tr><td>Model</td><td> $\mathrm { M u Q } _ { \mathrm { c o s } } \uparrow$ </td><td> $\mathrm { C L A P _ { c o s } \uparrow }$ </td><td>MSS↓</td></tr><tr><td>SpecDiff</td><td>0.885 [0.865, 0.900]</td><td>0.671 [0.647, 0.699]</td><td>17.23 [12.13, 25.03]</td></tr><tr><td>CTD†</td><td>0.767 [0.748, 0.789]</td><td>0.555 [0.518, 0.600]</td><td>77.02 [72.30, 80.74]</td></tr><tr><td>Ours</td><td>0.905 [0.886, 0.920]</td><td>0.833 [0.806, 0.869]</td><td>9.50 [8.51, 10.45]</td></tr></table>

<table><tr><td>Model</td><td> $F _ { \mathrm { O n } } \uparrow$ </td><td> $F _ { \mathrm { P 3 7 } } \uparrow$ </td><td> $F _ { \mathrm { P 1 3 } } \uparrow$ </td><td> $F _ { \mathrm { O p e n M I C } } \uparrow$ </td></tr><tr><td>SpecDiff</td><td>50.50 [42.64, 58.63]</td><td>34.62 [26.44, 42.78]</td><td>44.80 [38.26, 50.78]</td><td>69.19 [53.08, 84.47]</td></tr><tr><td>CTD†</td><td>35.39 [26.62, 43.60]</td><td>2.89 [0.57, 6.28]</td><td>22.56 [13.38, 32.71]</td><td>6.67 [0.00, 16.67]</td></tr><tr><td>Ours</td><td>54.24 [48.28, 60.37]</td><td>37.85 [26.85, 49.78]</td><td>47.90 [40.66, 55.63]</td><td>53.00 [32.00, 74.12]</td></tr></table>

B.5. GuitarSet
<table><tr><td>Model</td><td> $\mathrm { M u Q } _ { \mathrm { e v a l } }$  ←</td><td>PQ↑</td><td> $\mathrm { F A D } _ { \mathrm { M E R T } } \downarrow$ </td></tr><tr><td>SpecDiff</td><td>4.06 [3.95, 4.17]</td><td>7.89 [7.75, 8.02]</td><td>5.92</td></tr><tr><td>CTD</td><td>3.56 [3.48, 3.64]</td><td>7.95 [7.80, 8.08]</td><td>8.67</td></tr><tr><td>TokenSynth</td><td>3.70 [3.49, 3.91]</td><td>7.33 [6.98, 7.67]</td><td>14.77</td></tr><tr><td>Ours</td><td>4.14 [4.01, 4.27]</td><td>8.40 [8.33, 8.46]</td><td>4.05</td></tr></table>

<table><tr><td>Model</td><td> $\mathrm { M u Q } _ { \mathrm { c o s } }$  ←</td><td> $\mathrm { C L A P _ { c o s } } \mathrm { ~ \cdot ~ }$  ←</td><td>MSS↓</td></tr><tr><td>SpecDiff</td><td>0.892 [0.881, 0.903]</td><td>0.710 [0.686, 0.733]</td><td>17.76 [14.43, 21.21]</td></tr><tr><td>CTD</td><td>0.756 [0.723, 0.788]</td><td>0.574 [0.503, 0.646]</td><td>48.41 [42.98, 54.31]</td></tr><tr><td>TokenSynth</td><td>0.686 [0.630, 0.734]</td><td>0.367 [0.279, 0.457]</td><td>110.17 [95.44, 125.50]</td></tr><tr><td>Ours</td><td>0.921 [0.913, 0.929]</td><td>0.908 [0.890, 0.923]</td><td>14.29 [11.28, 17.46]</td></tr></table>

<table><tr><td>Model</td><td> $F _ { \mathrm { O n } } \uparrow$   $F _ { \mathrm { O p e n M I C } } \uparrow$ </td></tr><tr><td>SpecDiff</td><td>83.67 [75.22, 90.60] 80.83 [71.67, 89.17]</td></tr><tr><td>CTD</td><td>47.37 [37.99, 57.23] 20.00 [5.00, 40.00]</td></tr><tr><td>TokenSynth</td><td>34.36 [25.30, 44.18] 11.67 [0.00, 25.00]</td></tr><tr><td>Ours</td><td>88.01 [83.50, 92.08] 85.00 [70.00, 100.00]</td></tr></table>

## B.6. MAESTRO

<table><tr><td>Model</td><td> $\mathrm { M u Q } _ { \mathrm { e v a l } }$  ←</td><td>PQ↑</td><td> $\mathrm { F A D } _ { \mathrm { M E R T } } \downarrow$ </td></tr><tr><td>SpecDiff</td><td>4.47 [4.36, 4.56]</td><td>6.58 [6.44, 6.72]</td><td>4.46</td></tr><tr><td>CTD</td><td>3.75 [3.60, 3.90]</td><td>6.89 [6.65, 7.12]</td><td>6.82</td></tr><tr><td>TokenSynth</td><td>3.95 [3.82, 4.10]</td><td>6.99 [6.78, 7.19]</td><td>9.00</td></tr><tr><td>MIDI-VALLE</td><td>4.37 [4.22, 4.52]</td><td>7.64 [7.56, 7.72]</td><td>3.67</td></tr><tr><td>U-MusT</td><td>4.35 [4.21, 4.49]</td><td>7.24 [7.03, 7.43]</td><td>2.59</td></tr><tr><td>Ours</td><td>4.61 [4.46, 4.75]</td><td>7.50 [7.38, 7.60]</td><td>2.89</td></tr></table>

<table><tr><td>Model</td><td> $\mathrm { M u Q } _ { \mathrm { c o s } }$  ←</td><td> $\mathrm { C L A P _ { c o s } } $  个</td><td>MSS↓</td></tr><tr><td>SpecDiff</td><td>0.925 [0.914, 0.936]</td><td>0.705 [0.670, 0.740]</td><td>31.24 [27.21, 35.15]</td></tr><tr><td>CTD</td><td>0.741 [0.715, 0.766]</td><td>0.662 [0.623, 0.699]</td><td>49.22 [46.47, 52.13]</td></tr><tr><td>TokenSynth</td><td>0.785 [0.758, 0.813]</td><td>0.473 [0.430, 0.515]</td><td>63.15 [57.00, 70.76]</td></tr><tr><td>MIDI-VALLE</td><td>0.932 [0.920, 0.943]</td><td>0.863 [0.826, 0.891]</td><td>23.56 [20.32, 26.67]</td></tr><tr><td>U-MusT</td><td>0.915 [0.899, 0.930]</td><td>0.858 [0.825, 0.888]</td><td>30.58 [26.04, 35.22]</td></tr><tr><td>Ours</td><td>0.964 [0.958, 0.970]</td><td>0.863 [0.831, 0.888]</td><td>20.23 [17.51, 22.94]</td></tr></table>

<table><tr><td>Model</td><td> $F _ { \mathrm { O n } } \uparrow$   $F _ { \mathrm { O p e n M I C } } \uparrow$ </td></tr><tr><td>SpecDiff</td><td>66.63 [59.01, 74.06] 100.00 [100.00, 100.00]</td></tr><tr><td>CTD</td><td>16.67 [13.78, 19.87] 96.67 [91.67, 100.00]</td></tr><tr><td>TokenSynth</td><td>22.90 [19.07, 26.84] 60.00 [48.29, 71.67]</td></tr><tr><td>MIDI-VALLE</td><td>58.27 [51.13, 65.43] 100.00 [100.00, 100.00]</td></tr><tr><td>U-MusT</td><td>36.73 [30.12, 43.46] 100.00 [100.00, 100.00]</td></tr><tr><td>Ours</td><td>76.49 [70.54, 81.74] 100.00 [100.00, 100.00]</td></tr></table>

## C. EDITING RESULTS WITH CONFIDENCE INTERVALS (TABLE 2)

The reporting convention in Appendix B also applies here. Each original song is one bootstrap cluster, keeping its related edit cases, target durations, and dry/wet renders together. Within each group, all models are evaluated on the same pairs. Per-example scores are averaged over the relevant pairs in each group, as specified below. Ours and Ours + FlowEdit use the same Base SQ codec. The dagger retains its meaning from Table 2.

Slakh. The editing set contains 668 original–revised pairs. Each 20.48-second crop contains an editing region spanning 30%, 50%, or 70% of its duration. Track insertion, reported as “Add new instrument”, comprises 100 drum-inclusive pairs from 100 songs. “Note ins/del” pools note insertion, note deletion, and track deletion: 93/91/91 pairs without drums and 97/98/98 with drums, respectively. These give 275 pairs from 127 songs and 293 pairs from 129 songs. In the two note insertion/deletion groups, $R _ { \mathrm { a d d } }$ uses the note-insertion cases, while $R _ { \mathrm { d e l } }$ uses the note- and track-deletion cases (182 and 196 pairs). $F _ { \mathrm { k e e p } }$ uses cases with unchanged notes (273 and 293 pairs). A dash indicates that no deletions were requested.

POP909. Modulator supplies revised MIDI for 5-, 7-, and 10-second editing regions within 20.48-second crops. Each of 147 songs contributes one original–revised pair per duration and rendering condition (dry or matched reverberation), giving 882 pairs. Per-example scores are averaged over pairs from all six conditions. $F _ { \mathrm { k e e p } }$ uses the 874 pairs with unchanged notes.

## C.1. Slakh (+drums): add new instrument

<table><tr><td>Model</td><td> $\mathbf { M u Q _ { \mathrm { e v a l } } } \ \uparrow$ </td><td>PQ↑</td><td>FADMERT ↓</td></tr><tr><td>SpecDiff</td><td>4.29 [4.22, 4.36]</td><td>7.71 [7.63, 7.79]</td><td>4.87</td></tr><tr><td>U-MusT</td><td>3.72 [3.64, 3.79]</td><td>7.98 [7.92, 8.04]</td><td>3.45</td></tr><tr><td>Ours</td><td>4.44 [4.36, 4.52]</td><td>8.13 [8.08, 8.18]</td><td>1.68</td></tr><tr><td>Ours + FlowEdit</td><td>4.34 [4.26, 4.42]</td><td>8.01 [7.94, 8.07]</td><td>1.43</td></tr></table>

<table><tr><td>Model</td><td> $\mathrm { M u Q } _ { \mathrm { c o s } } $  个</td><td> $\mathrm { C L A P _ { c o s } \uparrow }$ </td><td>MSS↓</td></tr><tr><td>SpecDiff</td><td>0.787 [0.773, 0.801]</td><td>0.631 [0.610, 0.650]</td><td>45.04 [42.47, 48.03]</td></tr><tr><td>U-MusT</td><td>0.699 [0.680, 0.717]</td><td>0.757 [0.739, 0.775]</td><td>41.39 [38.94, 43.96]</td></tr><tr><td>Ours</td><td>0.874 [0.863, 0.884]</td><td>0.845 [0.829, 0.860]</td><td>31.54 [30.09, 32.98]</td></tr><tr><td>Ours + FlowEdit</td><td>0.882 [0.872, 0.892]</td><td>0.833 [0.815, 0.850]</td><td>24.40 [22.95, 25.92]</td></tr></table>

<table><tr><td>Model</td><td> $R _ { \mathrm { a d d } }$  ↑</td><td> $R _ { \mathrm { d e l } } \downarrow$ </td><td> $F _ { \mathrm { k e e p } }$  ←</td><td> $F _ { \mathrm { O p e n M I C } } \uparrow$ </td></tr><tr><td>SpecDiff</td><td>43.68 [38.22, 49.15]</td><td></td><td>32.54 [29.15, 35.92]</td><td>12.86 [9.95, 15.86]</td></tr><tr><td>U-MusT</td><td>21.32 [18.07, 24.78]</td><td></td><td>8.34 [6.75, 9.96]</td><td>11.66 [8.63, 14.81]</td></tr><tr><td>Ours</td><td>49.13 [43.55, 54.65]</td><td></td><td>31.09 [27.87, 34.33]</td><td>14.73 [11.45, 18.10]</td></tr><tr><td>Ours + FlowEdit</td><td>42.14 [36.64, 47.79]</td><td></td><td>32.68 [29.12, 36.24]</td><td>14.85 [11.63, 18.00]</td></tr></table>

## C.2. Slakh (without drums): note insertion/deletion

<table><tr><td>Model</td><td> $\mathbf { M u Q _ { \mathrm { e v a l } } } \ \uparrow$ </td><td>PQ↑ FADMERT ↓</td></tr><tr><td>CTD†</td><td>3.92 [3.85, 3.99]</td><td>7.28 [7.19, 7.36] 1.82</td></tr><tr><td>Ours</td><td>4.40 [4.33, 4.46]</td><td>7.88 [7.83, 7.93] 1.25</td></tr><tr><td>Ours + FlowEdit</td><td>4.32 [4.25, 4.39]</td><td>7.76 [7.69, 7.83] 1.06</td></tr></table>

<table><tr><td>Model</td><td> $\mathrm { M u Q } _ { \mathrm { c o s } } \uparrow$ </td><td> $\mathrm { C L A P _ { c o s } \uparrow }$ </td><td>MSS↓</td></tr><tr><td>CTD†</td><td>0.817 [0.807, 0.827]</td><td>0.722 [0.709, 0.735]</td><td>80.55 [76.79, 84.43]</td></tr><tr><td>Ours</td><td>0.897 [0.890, 0.903]</td><td>0.840 [0.830, 0.851]</td><td>21.37 [20.30, 22.43]</td></tr><tr><td>Ours + FlowEdit</td><td>0.911 [0.904, 0.918]</td><td>0.861 [0.849, 0.872]</td><td>11.15 [10.52, 11.79]</td></tr></table>

<table><tr><td>Model</td><td> $R _ { \mathrm { a d d } } ~ \uparrow$ </td><td> $R _ { \mathrm { d e l } } \downarrow$ </td><td> $F _ { \mathrm { k e e p } } \uparrow$ </td><td> $F _ { \mathrm { O p e n M I C } } \uparrow$ </td></tr><tr><td>CTD†</td><td>43.20 [37.98, 48.28]</td><td>3.82 [2.80, 4.98]</td><td>28.91 [26.64, 31.23]</td><td>17.33 [14.21, 20.51]</td></tr><tr><td>Ours</td><td>42.51 [36.31, 48.78]</td><td>6.08 [4.32, 8.05]</td><td>33.24 [30.65, 35.84]</td><td>26.56 [23.29, 29.78]</td></tr><tr><td>Ours + FlowEdit</td><td>43.49 [37.19, 49.84]</td><td>8.01 [6.16, 9.96]</td><td>35.75 [33.16, 38.46]</td><td>24.56 [21.22, 27.95]</td></tr></table>

C.3. Slakh (+drums): note insertion/deletion
<table><tr><td>Model</td><td> $\mathrm { M u Q } _ { \mathrm { e v a l } } \ \cdot$  个</td><td>PQ↑</td><td> $\mathrm { F A D } _ { \mathrm { M E R T } } \downarrow$ </td></tr><tr><td>SpecDiff</td><td>4.25 [4.19, 4.31]</td><td>7.72 [7.65, 7.78]</td><td>4.15</td></tr><tr><td>U-MusT</td><td>3.72 [3.67, 3.78]</td><td>7.95 [7.89, 8.00]</td><td>2.65</td></tr><tr><td>Ours</td><td>4.44 [4.39, 4.50]</td><td>8.03 [7.98, 8.08]</td><td>1.03</td></tr><tr><td>Ours + FlowEdit</td><td>4.36 [4.31, 4.42]</td><td>7.93 [7.87, 7.99]</td><td>0.83</td></tr></table>

<table><tr><td>Model</td><td> $\mathrm { M u Q } _ { \mathrm { c o s } } \uparrow$ </td><td> $\mathrm { C L A P _ { c o s } \uparrow }$ </td><td>MSS↓</td></tr><tr><td>SpecDiff</td><td>0.797 [0.787, 0.806]</td><td>0.619 [0.605, 0.632]</td><td>40.33 [38.86, 41.89]</td></tr><tr><td>U-MusT</td><td>0.699 [0.685, 0.712]</td><td>0.744 [0.730, 0.756]</td><td>37.81 [36.31, 39.38]</td></tr><tr><td>Ours</td><td>0.894 [0.885, 0.902]</td><td>0.870 [0.858, 0.881]</td><td>23.26 [22.32, 24.22]</td></tr><tr><td>Ours + FlowEdit</td><td>0.913 [0.905, 0.920]</td><td>0.882 [0.872, 0.891]</td><td>12.65 [12.05, 13.31]</td></tr></table>

<table><tr><td>Model</td><td> $R _ { \mathrm { a d d } }$  个</td><td> $R _ { \mathrm { d e l } } \downarrow$ </td><td> $F _ { \mathrm { k e e p } } \uparrow$ </td><td> $F _ { \mathrm { O p e n M I C } } \uparrow$ </td></tr><tr><td>SpecDiff</td><td>52.64 [46.67, 58.47]</td><td>2.62 [1.67, 3.80]</td><td>39.19 [36.44, 41.91]</td><td>15.48 [13.07, 17.98]</td></tr><tr><td>U-MusT</td><td>20.50 [16.53, 24.67]</td><td>3.72 [2.83, 4.72]</td><td>9.56 [8.45, 10.82]</td><td>13.28 [11.25, 15.43]</td></tr><tr><td>Ours</td><td>56.44 [50.64, 62.13]</td><td>4.56 [3.21, 6.24]</td><td>41.90 [39.34, 44.52]</td><td>16.92 [14.61, 19.28]</td></tr><tr><td>Ours + FlowEdit</td><td>50.54 [44.85, 56.27]</td><td>7.78 [5.99, 9.71]</td><td>41.55 [38.90, 44.22]</td><td>16.21 [13.71, 18.78]</td></tr></table>

## C.4. POP909: AI edits

<table><tr><td>Model</td><td> $\mathrm { M u Q } _ { \mathrm { e v a l } }$  ↑</td><td>PQ↑</td><td> $\mathrm { F A D } _ { \mathrm { M E R T } } \downarrow$ </td></tr><tr><td>SpecDiff</td><td>4.35 [4.31, 4.39]</td><td>7.47 [7.42, 7.51]</td><td>5.46</td></tr><tr><td>CTD</td><td>3.66 [3.62, 3.70]</td><td>7.42 [7.38, 7.46]</td><td>7.99</td></tr><tr><td>TokenSynth</td><td>3.79 [3.75, 3.83]</td><td>7.41 [7.38, 7.45]</td><td>9.21</td></tr><tr><td>MIDI-VALLE</td><td>4.13 [4.09, 4.16]</td><td>7.95 [7.94, 7.97]</td><td>5.19</td></tr><tr><td>U-MusT</td><td>4.26 [4.22, 4.29]</td><td>7.78 [7.76, 7.80]</td><td>6.50</td></tr><tr><td>Ours</td><td>4.42 [4.38, 4.45]</td><td>7.99 [7.97, 8.01]</td><td>3.33</td></tr><tr><td>Ours + FlowEdit</td><td>4.36 [4.33, 4.40]</td><td>8.03 [8.01, 8.05]</td><td>2.31</td></tr></table>

<table><tr><td>Model</td><td> $\mathrm { M u Q } _ { \mathrm { c o s } }$  ←</td><td> $\mathrm { C L A P _ { c o s } } $  个</td><td>MSS↓</td></tr><tr><td>SpecDiff</td><td>0.873 [0.869, 0.876]</td><td>0.762 [0.755, 0.768]</td><td>30.92 [29.59, 32.31]</td></tr><tr><td>CTD</td><td>0.716 [0.708, 0.723]</td><td>0.652 [0.642, 0.662]</td><td>45.11 [43.91, 46.36]</td></tr><tr><td>TokenSynth</td><td>0.741 [0.732, 0.749]</td><td>0.625 [0.617, 0.634]</td><td>87.71 [85.24, 90.29]</td></tr><tr><td>MIDI-VALLE</td><td>0.865 [0.861, 0.869]</td><td>0.729 [0.720, 0.737]</td><td>23.93 [22.48, 25.35]</td></tr><tr><td>U-MusT</td><td>0.842 [0.838, 0.846]</td><td>0.810 [0.802, 0.818]</td><td>29.00 [27.84, 30.27]</td></tr><tr><td>Ours</td><td>0.920 [0.918, 0.922]</td><td>0.766 [0.761, 0.772]</td><td>16.82 [15.99, 17.65]</td></tr><tr><td>Ours + FlowEdit</td><td>0.915 [0.912, 0.917]</td><td>0.777 [0.771, 0.783]</td><td>15.64 [14.78, 16.52]</td></tr></table>

<table><tr><td>Model</td><td> $R _ { \mathrm { a d d } } \cdot$  个</td><td> $R _ { \mathrm { d e l } } \downarrow$ </td><td> $F _ { \mathrm { k e e p } } \uparrow$ </td><td> $F _ { \mathrm { O p e n M I C } } \uparrow$ </td></tr><tr><td>SpecDiff</td><td>83.03 [81.06, 84.78]</td><td>2.60 [2.25, 2.96]</td><td>12.85 [11.12, 14.67]</td><td>98.85 [97.51, 99.69]</td></tr><tr><td>CTD</td><td>30.16 [28.43, 31.83]</td><td>6.74 [6.06, 7.43]</td><td>3.23 [2.73, 3.80]</td><td>97.03 [96.11, 97.88]</td></tr><tr><td>TokenSynth</td><td>27.24 [25.93, 28.54]</td><td>5.27 [4.77, 5.79]</td><td>7.66 [6.95, 8.47]</td><td>80.50 [78.63, 82.39]</td></tr><tr><td>MIDI-VALLE</td><td>72.74 [71.05, 74.38]</td><td>4.86 [4.36, 5.40]</td><td>10.41 [9.12, 11.77]</td><td>99.74 [99.24, 100.00]</td></tr><tr><td>U-MusT</td><td>42.56 [40.88, 44.28]</td><td>5.39 [4.88, 5.92]</td><td>3.77 [3.21, 4.33]</td><td>99.96 [99.89, 100.00]</td></tr><tr><td>Ours</td><td>89.86 [88.93, 90.71]</td><td>3.40 [3.07, 3.74]</td><td>13.84 [11.96, 15.72]</td><td>99.81 [99.58, 100.00]</td></tr><tr><td>Ours + FlowEdit</td><td>86.81 [85.81, 87.77]</td><td>5.06 [4.58, 5.55]</td><td>12.46 [10.71, 14.30]</td><td>99.89 [99.70, 100.00]</td></tr></table>

## D. NOTE-ADHERENCE ERROR ANALYSIS WITH CONFIDENCE INTERVALS (TABLE 3)

Both panels use Slakh (+drums), with 100 examples from 100 songs per panel. All scores are percentages evaluated with YourMT3+ at a 50-ms onset tolerance. Ground-truth audio, continuous reconstruction without SQ, and Base SQ reconstruction use the same target MIDI as the generated output. Panel (b) evaluates the revised ground-truth audio and its reconstructions against the requested edit. Perfect AMT is a theoretical 100% control, not an empirical estimate, and therefore has no CI.

## D.1. Synthesis

<table><tr><td>Evaluated audio / transcription</td><td> $F _ { \mathrm { P 3 7 } }$  个</td><td> $F _ { \mathrm { O n } }$  ↑</td></tr><tr><td>Ground Truth (GT) w/ Perfect AMT</td><td>100.00</td><td>100.00</td></tr><tr><td>GT w/ AMT</td><td>78.40 [76.58, 80.16]</td><td>82.79 [81.32, 84.21]</td></tr><tr><td>Reconstruction (no SQ) w/ AMT</td><td>44.43 [41.90, 46.93]</td><td>55.94 [53.59, 58.27]</td></tr><tr><td>Reconstruction (SQ) w/ AMT</td><td>43.98 [41.36, 46.57]</td><td>55.79 [53.48, 58.01]</td></tr><tr><td>Generated (ours)</td><td>37.78 [35.17, 40.40]</td><td>54.06 [51.59, 56.48]</td></tr></table>

## D.2. Add new instrument

<table><tr><td>Evaluated audio / transcription</td><td> $R _ { \mathrm { a d d } } \cdot$  个</td><td> $F _ { \mathrm { k e e p } } \cdot$  个</td></tr><tr><td>Ground Truth (GT) w/ Perfect AMT</td><td>100.00</td><td>100.00</td></tr><tr><td>GT w/ AMT</td><td>71.40 [66.79, 75.86]</td><td>72.80 [69.92, 75.55]</td></tr><tr><td>Reconstruction (no SQ) w/ AMT</td><td>55.99 [50.95, 60.89]</td><td>39.01 [35.33, 42.71]</td></tr><tr><td>Reconstruction (SQ) w/ AMT</td><td>54.54 [49.65, 59.44]</td><td>37.50 [33.88, 41.21]</td></tr><tr><td>Generated (ours)</td><td>49.13 [43.55, 54.65]</td><td>31.09 [27.87, 34.33]</td></tr></table>

## E. INFERENCE-TIME ABLATIONS WITH CONFIDENCE INTERVALS (TABLE 4)

The same trained checkpoint is evaluated on 56 target intervals, eight from one recording each in Slakh, MAESTRO, PianoVAM, Music-NetEM, URMP, KRAISLER, and FiloBass (dataset references in Appendix A). The bootstrap resamples these seven recordings, keeping their intervals together. The intervals describe uncertainty for this small, mixed-dataset collection and should not be interpreted as dataset-specific estimates.

Base uses contextual MIDI (CM), the clean contextual audio condition A (CA), and CFG 2. The 32-step ablations remove CM, zero only A, or disable CFG (CFG 1), without retraining. The observed frames in the noisy input remain available in the −CA condition. FAD uses pooled features from all 56 intervals and is reported without a CI.

<table><tr><td>Setting</td><td> $\mathbf { M u Q _ { \mathrm { e v a l } } } \ \uparrow$ </td><td>FADMERT↓</td><td> $F _ { \mathrm { O n } } \uparrow$ </td></tr><tr><td>Base, 4 steps</td><td>4.48 [4.27, 4.68]</td><td>3.65</td><td>74.60 [59.55, 86.92]</td></tr><tr><td>Base, 8 steps</td><td>4.46 [4.26, 4.65]</td><td>3.29</td><td>76.14 [62.07, 87.79]</td></tr><tr><td>Base, 16 steps</td><td>4.43 [4.24, 4.61]</td><td>3.09</td><td>75.90 [61.03, 87.61]</td></tr><tr><td>Base, 32 steps</td><td>4.41 [4.22, 4.60]</td><td>2.99</td><td>73.34 [58.36, 85.86]</td></tr><tr><td>Base, 64 steps</td><td>4.40 [4.22, 4.57]</td><td>2.95</td><td>73.03 [58.28, 85.92]</td></tr><tr><td>−CM, 32 steps</td><td>4.44 [4.26, 4.62]</td><td>3.36</td><td>76.01 [62.27, 87.04]</td></tr><tr><td>–CA, 32 steps</td><td>3.48 [3.00, 3.90]</td><td>9.95</td><td>45.66 [29.92, 65.58]</td></tr><tr><td>-CFG, 32 steps</td><td>4.29 [4.11, 4.44]</td><td>2.98</td><td>65.67 [51.17, 79.98]</td></tr></table>

[A1] E. Manilow, G. Wichern, P. Seetharaman, and J. Le Roux, “Cutting music source separation some Slakh: A dataset to study the impact of training data quality and quantity,” in Proc. IEEE WASPAA, 2019, pp. 45–49.

[A2] F. Ostermann, I. Vatolkin, and M. Ebeling, “AAM: A dataset of artificial audio multitracks for diverse music information retrieval tasks,” EURASIP J. Audio, Speech, Music Process., vol. 2023, no. 1, pp. 13, 2023.

[A3] C. Hawthorne et al., “Enabling factorized piano music modeling and generation with the MAESTRO dataset,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019.

[A4] Y. Kim et al., “PianoVAM: A multimodal piano performance dataset,” in Proc. ISMIR, 2025.

[A5] J. Thickstun, Z. Harchaoui, and S. M. Kakade, “Learning features of music from scratch,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2017.

[A6] B. Maman and A. H. Bermano, “Unaligned supervision for automatic music transcription in the wild,” in Proc. Int. Conf. Mach. Learn. (ICML), 2022, vol. 162, pp. 14918–14934.

[A7] H.-U. Berendes, A. Saha, B. Maman, V. Arifi-Muller, and M. M¨ uller, “Beethoven symphony excerpt dataset (BSED): An evaluation¨ dataset for orchestral music transcription,” Trans. Int. Soc. Music Inf. Retr., vol. 9, no. 1, pp. 405–422, 2026.

[A8] J. Loth, P. Sarmento, S. Sarkar, Z. Guo, M. Barthet, and M. Sandler, “GOAT: A large dataset of paired guitar audio recordings and tablatures,” in Proc. ISMIR, 2025, pp. 655–662.

[A9] Q. Xi, R. M. Bittner, J. Pauwels, X. Ye, and J. P. Bello, “GuitarSet: A dataset for guitar transcription,” in Proc. 19th Int. Soc. Music Inf. Retrieval Conf. (ISMIR), 2018.

[A10] B. Li, X. Liu, K. Dinesh, Z. Duan, and G. Sharma, “Creating a multi-track classical music performance dataset for multi-modal music analysis: Challenges, insights, and applications,” IEEE Transactions on Multimedia, 2018.

[A11] J. Abeßer, H. Lukashevich, and G. Schuller, “Feature-based extraction of plucking and expression styles of the electric bass guitar,” in Proc. IEEE ICASSP, 2010, pp. 2290–2293.

[A12] H. Kim, J. Park, S. Lee, T. Kwon, S. Won, and J. Nam, “KRAISLER: A multi-track dataset of piano and violin duet recordings for music information retrieval research,” Trans. Int. Soc. Music Inf. Retr., vol. 9, no. 1, pp. 456–473, 2026.

[A13] S. Balke, A. Berndt, and M. Muller, “ChoraleBricks: A modular multitrack dataset for wind music research,”¨ Trans. Int. Soc. Music Inf. Retr., vol. 8, no. 1, pp. 39–54, 2025.

[A14] X. Riley and S. Dixon, “FiloBass: A dataset and corpus based study of jazz basslines,” in Proc. ISMIR, 2023, pp. 500–507.

[A15] C. Southall, C.-W. Wu, A. Lerch, and J. Hockman, “MDB Drums: An annotated subset of MedleyDB for automatic drum transcription,” in Proc. ISMIR Late-Breaking Demo Session, 2017.

[A16] H. Pedroza, W. Abreu, R. M. Corey, and I. R. Roman, “Leveraging real electric guitar tones and effects to improve robustness in guitar tablature transcription modeling,” in Proc. Int. Conf. Digital Audio Effects (DAFx), 2024.

[A17] O. Gillet and G. Richard, “ENST-Drums: An extensive audio-visual database for drum signals processing,” in Proc. ISMIR, 2006, pp. 156–159.

[A18] P. Weber, C. Uhle, M. Muller, and M. Lang, “STAR Drums: A dataset for automatic drum transcription,”¨ Trans. Int. Soc. Music Inf. Retr., vol. 8, no. 1, pp. 248–264, 2025.

[A19] J. Gardner, I. Simon, E. Manilow, C. Hawthorne, and J. Engel, “MT3: Multi-task multitrack music transcription,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.

[A20] Z. Wang et al., “POP909: A pop-song dataset for music arrangement generation,” in Proc. ISMIR, 2020, pp. 38–45.

[A21] K. Bhandari, M. Bizzarri, G. A. Wiggins, and S. Colton, “Change is Key: A generative framework for controllable musical modulations,” Hugging Face model repository, 2026, https://huggingface.co/keshavbhandari/modulator.

[A22] A. Holm, “Salamander Grand Piano,” FreePats, SoundFont version V3+2020-06-02, 2020, https://freepats.zenvoid. org/Piano/acoustic-grand-piano.html.