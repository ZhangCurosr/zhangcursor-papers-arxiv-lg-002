# iMINDBench: iEEG Multi-Institution Neural Decoding Benchmark

Geeling Chau<sup>1∗</sup> Saba Hashemi<sup>2†</sup> Yonghyeon Gwon<sup>3†</sup> Eshani Patel<sup>1,4</sup> Jan DeWitt<sup>5</sup> Christopher Wang<sup>5</sup> Andrii Zahorodnii<sup>5</sup> Sabera J. Talukder<sup>1</sup> Danny Dongyeop Han<sup>3</sup> Chun Kee Chung<sup>3</sup> Maryam M. Shanechi<sup>2</sup> Yisong Yue<sup>1</sup>

<sup>1</sup>Caltech <sup>2</sup>USC <sup>3</sup>Seoul National University <sup>4</sup>CMU <sup>5</sup>MIT

## Abstract

Intracranial electroencephalography (iEEG) is widely used to record electrical activity directly from electrodes inside the human brain, making it an attractive modality for neural decoding. However, progress in iEEG decoding, especially toward general-purpose foundation models, remains difficult to measure reliably: datasets are task- or institution-specific, limiting evidence of generalization across tasks and recording environments, and preprocessing choices can strongly influence performance, making model improvements difficult to distinguish from preprocessing gains. Thus, we introduce IMINDBENCH, an iEEG Multi-Institution Neural Decoding Benchmark that evaluates models on a shared suite of fifteen decoding tasks across three naturalistic movie-watching datasets. The benchmark additionally defines standardized preprocessing tracks and fixed evaluation splits to support consistent model comparisons. Using iMINDBench, we find that the evaluated pretrained systems generally outperform baselines within their respective preprocessing tracks, while strong spectral baselines remain competitive across institutional datasets. In our scaling study, adding up to 25 times more supervised data from other subjects or institutions yields only small or taskdependent gains over within-session training. Together, these findings highlight the need for iEEG models that improve on strong preprocessing baselines and make more effective use of data across subjects and institutions. Project website: https://imindbench.github.io/

## 1 Introduction

Neural decoding seeks models that generalize across subjects, institutions, and recording conditions while requiring minimal patient-specific calibration. Intracranial electroencephalography (iEEG) is an attractive setting for this goal because it records neural activity directly from electrodes inside the human brain with high temporal resolution. However, progress in iEEG decoding remains difficult to evaluate because existing studies often differ not only in dataset and task selection, but also in preprocessing pipelines and split construction. As a result, apparent benchmark improvements can reflect changes in evaluation setup rather than advances in neural modeling itself.

Existing resources address important parts of this problem. Multi-institution iEEG benchmarks such as Omni-iEEG [1] and Mayo/FNUSA [2] demonstrate that clinical recordings can be harmonized across sites, but primarily focus on seizure, pathology, high-frequency oscillation, or related epileptology tasks. Naturalistic decoding resources such as Neuroprobe [3], Brain Treebank [4], and AJILE12 [5] provide interpretable naturalistic decoding tasks, but evaluations often remain tied to a single dataset or institutional setting. Current models also differ in neural input representation and preprocessing pipeline, making improvements difficult to separate from changes in evaluation setup.

We therefore introduce iMINDBench, an iEEG Multi-Institution Neural Decoding Benchmark spanning fifteen language, auditory, and visual decoding tasks across naturalistic movie-watching datasets from three institutions. The benchmark uses fixed evaluation splits and specified preprocessing tracks, with evaluations spanning scaling domains and unit decodability subsets. Results are reported separately by dataset and preprocessing track to reveal how model performance varies across institutions and input representations.

The benchmark reveals why these controls matter. First, the evaluated pretrained systems generally outperform non-pretrained baselines within their respective preprocessing tracks on Main units. However, strong Multi-STFT baselines remain competitive with pretrained systems across institutional datasets. These baselines provide a reference point for developing learned representations that match or exceed the benefits of engineered spectral features across institutions. Second, by aligning tasks and neural-signal metadata across institutions, the benchmark lets us examine whether models benefit from additional subjects and institutions on the same task. In our scaling study, adding other subjects from the same dataset is broadly beneficial, although the gains remain smaller than those from additional target-session data. Adding data from other datasets and institutions produces weaker and more task-dependent improvements. These results highlight the difficulty of turning broader supervision into reliable improvements in target-session decoding.

By supporting these two complementary assessments, iMINDBench enables more systematic evaluation of general-purpose iEEG models across tasks, institutions, and preprocessing tracks, helping identify where their benefits generalize and where they remain specific to the evaluation setting.

## Contributions.

• Benchmark artifact. We introduce a multi-institution iEEG decoding benchmark spanning three independently collected movie-watching datasets with fifteen aligned decoding tasks.

• Evaluation contract. iMINDBench defines fixed splits, decodability subsets, standardized Multi-STFT and Waveform preprocessing tracks, and supervised training scaling domains.

• Benchmark findings. Strong spectral baselines remain competitive with pretrained systems, providing a reference point for learned representations. Broader supervised fine-tuning yields limited, task-dependent gains in our scaling study, motivating models that learn more effectively from heterogeneous neural data.

## 2 Related Work

Datasets and benchmarks. Intracranial electroencephalography is attractive for naturalistic neural decoding because it records human neural activity at high temporal resolution during behavior, but clinical recordings are sparse, nonuniform, and institution-specific [6]. Existing iEEG benchmarks cover important parts of this landscape. The Mayo/FNUSA graphoelement dataset [2] and OmniiEEG [1] show that multi-institution clinical iEEG can be harmonized, but focus on seizure, pathology, high-frequency oscillation, or related epileptology targets. Naturalistic resources such as Brain Treebank [4], Neuroprobe [3], BYD [7], Pippi [8], and AJILE12 [5] provide movie, language, audiovisual, motor, or activity labels, but do not jointly control task, institution, and preprocessing heterogeneity in one evaluation protocol.

While scalp EEG benchmarks such as MOABB [9] and recent EEG foundation-model evaluations [10, 11] exist, they do not transfer cleanly to iEEG: scalp EEG is volume-conducted and bandlimited, whereas iEEG records directly from cortex and retains the high-frequency activity (roughly 70–250 Hz) which is important for cognitive decoding [6, 12]. In addition, iEEG electrode placement is clinically determined rather than fixed, breaking the standardized-montage assumption behind most EEG pipelines and foundation models. These differences necessitate separate benchmarking frameworks for intracranial EEG.

Neural foundation models and preprocessing. Recent electrophysiology foundation models make shared evaluation increasingly necessary. iEEG models such as BrainBERT [13], Brant [14], PopT [15], BaRISTA [16], MVPFormer [17], and DIVER-1 [18], together with EEG foundation models such as LaBraM [19] and EEGPT [20], address scale, subject variability, channel mismatch, spatial layout, and low signal-to-noise ratio. However, their evaluation conditions differ across waveform inputs, spectrograms or superlets, learned tokens, frozen embeddings, spatial metadata, pretraining corpora, and split units. A recent EEG foundation-model review argues that evaluations remain heterogeneous and limited [10], and a broad EEG-FM benchmark finds that fine-tuned foundation models can perform well in data-rich settings but often do not clearly outperform compact neural or classical decoders in data-scarce regimes [11]. This benchmark targets the corresponding iEEG problem: model comparisons should be made within explicit task, institution, split, and preprocessing specifications, with matched baseline reruns whenever the input representation changes.

## 3 Benchmark Design

The benchmark is designed to evaluate whether decoding improvements survive changes in institution, preprocessing track, and behavioral target under shared evaluation settings. To support this, iMIND-Bench standardizes preprocessing and evaluation across multiple naturalistic iEEG datasets while exposing performance differences across tasks and institutions. It also supports controlled scaling experiments that test whether additional supervised data improves decoding across sessions, subjects, and institutions. Figure 1 summarizes the benchmark workflow, including datasets, preprocessing tracks, splits, and decoding domains.

![](images/e23d5f8b3d43dcdd2ec8e23235b45a07f8950dd91cac74bae5378230ab5d4ae0.jpg)  
Figure 1: Multi-institution, multi-task neural decoding benchmark. The benchmark aligns naturalistic decoding tasks and patient metadata across three institutions and defines preprocessing tracks, splits, and decoding domains. (a) Three naturalistic movie-watching intracranial electroencephalography (iEEG) datasets are included: Neuroprobe, Bang! You’re Dead (BYD), and Pippi. (b) iEEG neural recordings are routed through explicit preprocessing tracks, which may include multi-resolution short-time Fourier transform (Multi-STFT), waveform, and custom preprocessors to be used as inputs to models that decode language, vision, and auditory task families. (c) The benchmark defines challenging temporal splits, data-scale regimes, and unit decodability subsets.

## 3.1 Datasets and Tasks

Datasets. The current release combines Neuroprobe/Brain Treebank [3, 4], the Bang! You’re Dead (BYD) movie-watching dataset [7], and Pippi [8]. Compared with the original Neuroprobe benchmark, we restrict evaluation of the Neuroprobe/Brain Treebank component to five held-out subject/session units so that remaining Brain Treebank recordings can be used for model pretraining without overlapping benchmark targets. Each dataset contains neural recordings acquired during movie or audiovisual stimulus presentation, but the datasets differ in recording duration, electrode coverage, subject/session structure, and institutional practices. Neuroprobe/Brain Treebank provides longer movie-watching recordings and comparatively large within-session training sets, BYD introduces a different institutional and stimulus context with shorter recordings and more explicit subject/session structure, and Pippi provides a smaller independent dataset with different electrode coverage and stimulus statistics. Figure 2 summarizes the resulting variation in dataset scale, electrode coverage, and task support across institutions.

Harmonization. iMINDBench aligns decoding targets and electrode metadata to support comparisons across institutions. BYD and Pippi use the Neuroprobe/Brain Treebank stimulus-feature extraction pipeline, with dataset-specific annotation and timing alignment. Available electrode coordinates and anatomical labels are harmonized to shared conventions. Differences in acquisition, electrode layout, and coverage remain part of the evaluation setting. Appendix A describes the alignment procedures and documents the source datasets’ acquisition metadata.

Tasks. The benchmark includes fifteen decoding tasks following the original Neuroprobe task definitions [3], spanning language, auditory, and visual domains. These include speech presence, sentence onset, GPT-2 surprisal, pitch, volume, optical flow, brightness, and face-count decoding. Both binary and multiclass label modes are supported; this paper reports binary classification results. Each example pairs a 1-second neural window with a stimulus label. Continuous annotations are binarized by contrasting low- and high-value ranges within each session, while categorical annotations are mapped to binary labels. Complete task definitions and dataset-specific label-construction details are provided in Appendix B.

## 3.2 Splits and Domains

Splits. We follow Neuroprobe’s two-fold within-session splitting procedure for each dataset– subject/session–task combination, termed an evaluation unit. Applying this procedure across datasets maintains compatibility with Neuroprobe, including reuse of compatible runs for Neuroprobe submission. Within each fold, training, validation, and test examples come from largely contiguous portions of the movie. Preprocessing statistics are estimated from the training split and applied unchanged to validation and test data. Each unit’s final score is averaged across the two folds.

Scaling domains. Scaling domains vary the source of supervised fine-tuning data while preserving the same target units and evaluation splits. Within-session uses only target-session data, representing adaptation to a new subject/session using its calibration labels. Within-dataset adds supervised examples from other subject/session units in the same dataset. Multi-dataset further adds supervised examples from other institutions. These domains test whether broader supervision improves decoding beyond target-session calibration.

![](images/cf6538bb446b140f42a3b5d8a55e479d40fd68ef70d2b9b3d4d53ac1412cedf1.jpg)

![](images/1f02ace32ad3adbc8db88e9581c4cfe4b0574a1689b8a2a4477a013a91a0f8a5.jpg)

![](images/f1275564f01e3365e652ef8f17ae2713c646aa8b4253e9d242162b1be0d4a83c.jpg)  
Figure 2: Dataset coverage is heterogeneous across institutions and tasks. (a) Average number of training samples, Main subject/session–task units, and electrodes per subject. (b) Electrode locations are shown in left/inferior coordinate space for the same datasets. (c) Task bars report the number of Main subject/session–task units for each aligned decoding task.

Unit decodability subsets. Many evaluation units remain near chance under current baselines. This may reflect weak task-relevant signal at the recorded electrode locations, poor recording quality, or limitations of current models in extracting task-relevant information. We therefore report Main, Challenge, and All unit decodability subsets to distinguish improvements on units that are decodable by baseline within-session models from progress on those that are presently near chance.

We operationalize near-chance performance using two screening decoders: an STFT decoder and HTNet at 500 Hz. The STFT decoder uses logistic regression with the single-STFT configuration that achieves the highest mean validation ROC-AUC for each unit. A supported unit u is assigned to Main when

$$
\operatorname* { m a x } \Bigl \{ \overline { { \mathrm { A U C } } } _ { \mathrm { v a l } } ^ { \mathrm { S T F T } } ( u ) , \overline { { \mathrm { A U C } } } _ { \mathrm { v a l } } ^ { \mathrm { H T N e t - 5 0 0 H z } } ( u ) \Bigr \} > 0 . 6 0 ,
$$

where each score is averaged across the two validation folds. Challenge contains the remaining supported units, classified as near chance because both screening scores are at or below 0.60. All contains both subsets. These labels describe decodability under the screening models; they do not establish that Challenge units lack task-relevant neural signal. The resulting Main/Challenge performance separation, task-level Main-unit model breakouts, and Main subject/session coverage are summarized in Appendix Figures 5, 6, and 7.

## 3.3 Preprocessing Tracks

Track definitions. Decoding performance depends on both the model and how neural recordings are preprocessed, making it necessary to account for both when interpreting improvements. Each preprocessing track specifies how neural recordings are transformed into model inputs, helping control for preprocessing variation. We provide two complementary tracks. The Multi-STFT track incorporates established signal-processing operations and spectral features [21] that enable strong performance with simple baseline models. The Waveform track instead provides inputs close to the raw recorded waveforms, preserving phase and temporal structure that the Multi-STFT representation does not explicitly represent. It tests whether models can learn representations that match or exceed the benefits of engineered spectral features. Comparisons under matched preprocessing assess model improvements using the same inputs, while comparisons across tracks assess complete decoding systems, including their preprocessing. The benchmark also supports custom preprocessing routes, allowing compatible reference baselines to be rerun on new inputs for matched comparisons.

Multi-STFT track. The Multi-STFT track is the benchmark’s standardized spectral preprocessing route. The spectral track first applies 60 Hz + harmonics notch filtering for line noise removal and Laplacian rereferencing for common electrode noise removal as done in many prior works [13, 22]. Because neural activity spans multiple temporal and frequency scales, a single fixed STFT resolution can unevenly represent different frequency bands and tasks. The Multi-STFT representation instead uses separate STFT resolutions for low-, mid-, and high-frequency ranges before combining them into a spectrogram-like representation compatible with common neural decoding models (Appendix Figure 10). The Multi-STFT sections use duration-defined windows and an approximately 62-ms hop consistently across dataset sampling rates; Appendix Figure 10 shows the frequency ranges, and Table 7 gives the exact sample settings and retained upper bins. The Multi-STFT track selects frequency bins within 2–250 Hz, excluding the lowest-frequency components while retaining bands including high gamma [12]. For each fold, the Multi-STFT track estimates normalization statistics exclusively from training samples, separately for each channel and frequency bin, and applies those fixed statistics to validation and test samples. This preserves cross-temporal information while standardizing the signals for machine-learning models. We compare normalization strategies and their downstream performances on Multi-STFT baselines in Figure 3a.

Waveform track. The Waveform track uses a 0.5 Hz high-pass filter, notch filtering for line noise, and Laplacian rereferencing, while permitting model-specific sampling rates and input scaling and normalization conventions. Supervised baselines use a standardized 500 Hz recipe with 60 Hz and harmonic notch filtering and robust global normalization fitted exclusively on the training split. This normalization uses a single estimated median center and a scaled median absolute deviation across training channels and timepoints, applied unchanged to validation and test samples, preserving relative amplitude differences across windows and channels. We use 500 Hz for the supervised waveform baselines because matched comparisons did not show consistent gains at higher sampling rates. Model-specific sampling, filtering, and normalization variations are documented in Table 8. Within-track comparisons involving model-specific variants evaluate complete decoding systems; they do not isolate architecture or pretraining effects. We visualize how preprocessing changes affect baseline performance on our strongest supervised waveform model, HTNet, in Figure 3a.

Custom preprocessing routes. Users can evaluate custom preprocessing routes by supplying preprocessing code and configuration files. These routes must be documented explicitly and, where com patible inputs or embeddings are available, evaluated with a simple matched baseline. Leaderboard entries report preprocessing details to make these comparisons transparent (Appendix Figure 11). Appendix D illustrates this approach for BrainBERT, comparing its pretrained representations with simple decoders using the corresponding single-STFT inputs.

## 3.4 Models and Reporting

Model families. We evaluate generic decoder baselines alongside pretrained and specialized iEEG models. Logistic regression, multilayer perceptron (MLP), and convolutional neural network (CNN) baselines are evaluated on both Multi-STFT and Waveform inputs. The Multi-STFT suite additionally includes pretrained PopT-v2 [15, 23]. The Waveform suite includes HTNet [24], trained from scratch, and pretrained BaRISTA [16] and DIVER-1 [18]. We also examined MVPFormer and Brant, whose inclusion requires further validation of adaptations from their released temporal input representations to one-second decoding windows (Appendix I.1). Appendix I provides model and training details.

Reporting. Results are reported with their preprocessing track, model family, and dataset-level performance. For pretrained models, we also report the datasets used for pretraining to contextualize their training exposure. For each dataset and model, we report the mean ROC-AUC across the available evaluation units. The Overall score is the unweighted mean of the three dataset-level scores. The benchmark website and leaderboard break down model comparisons by dataset, task, preprocessing track, scaling domain, and unit decodability subset, helping identify where model advantages are consistent and where they depend on the evaluation setting.

## 4 Benchmark Results

## 4.1 Decoding Performance

<table><tr><td>Track Model</td><td>Pretraining</td><td>Overall</td><td>Neuroprobe</td><td>BYD</td><td>Pippi</td></tr><tr><td colspan="6">Multi-STFT</td></tr><tr><td>Logistic</td><td>not pretrained</td><td>0.628</td><td>0.697</td><td>0.624</td><td>0.564</td></tr><tr><td>MLP</td><td>not pretrained</td><td>0.631</td><td>0.702</td><td>0.625</td><td>0.567</td></tr><tr><td>CNN</td><td>not pretrained</td><td>0.620</td><td>0.710</td><td>0.611</td><td>0.539</td></tr><tr><td>PopT-v2</td><td>BrainTreeBank</td><td>0.645</td><td>0.723</td><td>0.632</td><td>0.579</td></tr><tr><td colspan="6">Waveform</td></tr><tr><td rowspan="5"></td><td>Logistic</td><td>not pretrained</td><td>0.554</td><td>0.617</td><td>0.519 0.525</td></tr><tr><td>MLP</td><td>not pretrained</td><td>0.573</td><td>0.653</td><td>0.532 0.534</td></tr><tr><td>CNN</td><td>not pretrained</td><td>0.576</td><td>0.666</td><td>0.537 0.526</td></tr><tr><td>HTNet</td><td>not pretrained</td><td>0.614</td><td>0.693</td><td>0.597 0.552</td></tr><tr><td>BaRISTA</td><td>BrainTreeBank</td><td>0.636</td><td>0.716</td><td>0.632 0.558</td></tr><tr><td colspan="2">DIVER-1</td><td>private+ AJILE12 0.634</td><td>0.728</td><td>0.605</td><td>0.569</td></tr></table>

Table 1: Main benchmark results. Within-session ROC-AUC performance on Main units, grouped by preprocessing track. Cells are shaded with a normalized color scheme within column and track. Best per column and track are bolded, and second-best values are underlined.

Table 1 compares within-session decoding performance on Main units across the three datasets. The evaluated pretrained systems outperform non-pretrained baselines within their respective preprocessing tracks. PopT-v2 achieves the highest Overall ROC-AUC of 0.645, compared with 0.631 for the strongest non-pretrained Multi-STFT baseline. Within the Waveform track, BaRISTA and DIVER-1 achieve 0.636 and 0.634, respectively, compared with 0.614 for HTNet. However, the Multi-STFT MLP remains competitive with both pretrained waveform systems, particularly on BYD and Pippi, which have smaller downstream training sets than Neuroprobe. This highlights the importance of evaluating pretrained systems against strong preprocessing baselines under limited supervision. Because these comparisons involve different input representations, they assess complete decoding systems rather than isolating the contribution of the model.

The benefit of more complex spectral decoders varies across datasets. On Neuroprobe, CNN outperforms both logistic regression and MLP, whereas on BYD and Pippi, logistic regression and MLP outperform CNN. This pattern suggests that engineered spectral features support effective decoding with relatively simple models when labeled data are limited, highlighting the need for learned feature extractors that generalize reliably under limited supervision.

Waveform inputs show a different pattern. HTNet consistently outperforms generic logistic, MLP, and CNN baselines across all three datasets, indicating that the choice of architecture matters for extracting useful information from waveform inputs. Pretrained BaRISTA and DIVER-1 further outperform these within-session-trained baselines, although their relative strengths vary across datasets: DIVER-1 performs better on Neuroprobe and Pippi, while BaRISTA performs better on BYD.

Together, these results show that model advantages depend on the input representation and institutional dataset, motivating evaluation across both dimensions. Additional scorecards for All and Challenge units are provided in Appendix C.

## 4.2 Preprocessing

Figure 3 evaluates how preprocessing choices affect benchmark conclusions under otherwise matched evaluation settings. Baseline performance varies across preprocessing routes, indicating that neural decoding comparisons can depend strongly on input representation and normalization choices.

In Figure 3a, we compare two normalization strategies that encode different assumptions about signal scale. Sample Norm, following approaches used in wav2vec 2.0 and BrainBERT [13, 25], standardizes each input window independently. This may provide invariance to nuisance variation in recording gain and overall amplitude, but it can also remove between-window amplitude differences that are informative for decoding. Global Norm instead uses statistics estimated from the training set, preserving relative amplitude variation across examples while applying the same transformation to validation and test samples. Global Norm yields consistently stronger baseline performance, improving ROC-AUC by more than 0.10 in some evaluated settings. This result suggests that information removed by sample-wise normalization is often task-relevant in our evaluation settings. We therefore select Global Norm for the standardized baseline recipes in our Multi-STFT and Waveform preprocessing tracks; Section 3.3 details its implementation for each track.

We next test the Waveform track’s sensitivity to high-pass filtering while holding the remaining HTNet preprocessing fixed. High-pass filtering suppresses slow baseline drift and other low-frequency variation that can dominate waveform amplitude, although it may also attenuate slowly varying taskrelated signals. Omitting the 0.5-Hz high-pass filter reduces mean ROC-AUC from 0.612 to 0.589, with lower performance in all three datasets (Figure 3a). This consistent reduction suggests that, under our evaluation protocol, the benefit of suppressing very-low-frequency variation outweighs any useful information attenuated by the filter. We therefore retain the 0.5-Hz high-pass filter in the standardized Waveform baseline recipe.

Finally, we test the spectral track’s sensitivity to replacing Multi-STFT with a single STFT resolution. Aggregate performance is comparable (Figure 3a), but validation-selected Single-STFT parameters vary across datasets and tasks (Figure 3b,c). We therefore retain Multi-STFT as the flagship spectral track: its band-specific window lengths favor frequency resolution at lower frequencies and temporal resolution at higher frequencies without selecting a task- or dataset-specific resolution.

These results show that preprocessing choices can substantially affect decoding performance and support our standardized tracks through both empirical comparisons and signal-processing considera tions. Comparing models under matched preprocessing helps distinguish model improvements from gains due to preprocessing changes.

(a) Preprocessors change Baseline performances  
![](images/c15916edf02fba4b80ef665b6d3e9d04b540c083e7c9891cbee8c3d9eb2b9f59.jpg)  
(c) Improvement from STFT param selection

(b) Val ROC AUC on different STFT Params Sentence Onset  
![](images/611f5d0383fd6a00273b79a27095a00944a6c2567c1c8358c560d3decc32f8b3.jpg)

![](images/4931fd99c66419cdb9176a3285e0d2962612ef8c90ad3274db4ad441407b9b50.jpg)

![](images/8fd4abea2562683ebf1d320db9dc168617ae355751f69a77bbfa0bd584fcf7f6.jpg)  
Figure 3: Preprocessing choices have large effects on baseline performance. (a) Baseline model performance under different STFT and Waveform preprocessing choices. Bars show the unweighted mean of dataset-level averages; dots show individual dataset averages. Our proposed STFT and Wav preprocessing tracks are highlighted with the darkest color in each model family. (b) Validation ROC-AUC for STFT parameter sweeps. We sweep max frequency (y-axis), number of timepoints per segment (x-axis) and percent overlap (depth) at 50%, 75%, and 87.5%. Different datasets (columns) may have different optimal single-STFT params for the same task (rows). (c) Mean paired test ROC AUC change for logistic regression using validation-selected versus fixed single-STFT parameters (150 Hz maximum frequency, 250 ms segments, 75% overlap). Error bars indicate 95% bootstrap confidence intervals. All panels report results on the Main unit decodability subset.

## 4.3 Data Scaling

Transfer question. Data scaling is the benchmark’s direct test of the transfer promise behind population and foundation-style iEEG modeling. If broader supervised neural data are useful for naturalistic decoding, they should improve a target unit when target-session labels are scarce. Figure 4 separates that question into three data sources. Within-session scaling measures the value of additional examples from the same target unit. Within-dataset scaling asks whether other Main units from the same dataset help once institution, acquisition practice, stimulus family, and label construction are mostly shared. Multi-dataset scaling is the harder test: it asks whether supervised data from other institutions and stimulus contexts improves the same target units. We evaluate within-session scaling across spectral models and use PopT-v2, which supports joint supervised training across subjects, for within-dataset and multi-dataset scaling.

Scaling pattern. The scaling analysis exposes an important asymmetry between target-session and broader supervised data. Across evaluated models, within-session scaling produces the largest and most consistent gains, with mean test ROC-AUC increasing by approximately 0.025 for each doubling of the training data (Figure 4a). Using approximately 3–18 times as many training samples as withinsession training, within-dataset scaling improves mean ROC-AUC by 0.001 on Neuroprobe, 0.014 on Bang! You’re Dead, and 0.004 on Pippi (Figure 4b). Multi-dataset training uses approximately 6–25 times as many samples as within-session training, yet underperforms within-dataset training on Neuroprobe and Pippi. On Bang! You’re Dead, multi-dataset training improves mean ROC-AUC over within-dataset training by only 0.001 despite using approximately 30% more training samples.

(a) Neuroprobe Scaling Performance  
![](images/34410a22708f254239bbeeab615b44bce451dc35bca64b26fe3e024f3f3a1ba1.jpg)

(b)  
Data Scaling Beyond Within-Session  
![](images/10ebdfb207347cf8ccb1f4db1f705bd1c51deb2dea0eace986363d298c8ac9b2.jpg)  
Figure 4: Data scaling highlights a boundary in current model generalization. (a) Neuroprobe within-session decoding performance (y-axis) as the available training data increase (x-axis) across models evaluated on the Multi-STFT track (legend). (b) For each dataset (columns and colors), the top subplot shows the paired change in test ROC-AUC relative to within-session training (y-axis) when PopT-v2 is provided supervised data from within the same dataset (medium-shaded circles) or from multiple datasets (dark-shaded circles). The bottom subplot shows the corresponding training-data multipliers relative to within-session training (colored labels); the black label above the leftmost bar gives the mean number of within-session training samples per task. Error bars indicate 95% confidence intervals for paired changes in ROC-AUC. Task-level decompositions are provided in Appendix Figures 8 and 9. All panels report results on the Main unit decodability subset.

Within-dataset training increases mean test ROC-AUC over within-session training in 30 of 44 task–dataset pairs; adding other datasets yields further increases in 19 of 44 pairs (Appendix Figure 9). The benefit of broader supervision therefore depends on the decoding task. More effective crosssubject and cross-institution learning remains an important direction for improving downstream neural decoding.

## 5 Release and Extensibility

The evaluation code, preprocessing configurations, and baseline implementations will be released in the iMINDBench repository. Dataset preparation pipelines and benchmark split metadata will be available through TorchBrain and its Brainsets integration, built on the Neuro-Galaxy ecosystem [26]. These pipelines provide a shared interface for preparing recordings, task labels, and evaluation splits.

A publicly hosted benchmark website and leaderboard will support comparisons across datasets, tasks, preprocessing tracks, scaling domains, and unit decodability subsets. To submit a model, users will run the evaluation pipeline and submit generated result artifacts and model metadata through a pull request to the leaderboard repository, where automated checks validate the submission. Detailed instructions will be provided in the submission guide.

Additional model-native systems can be evaluated through the custom preprocessing route when coverage is complete and matched baseline reruns are available. Future releases may add datasets, tasks, preprocessors, models, or hidden-test evaluation while preserving archived benchmark versions.

## 6 Discussion

The results show that preprocessing choices and cross-dataset heterogeneity remain central to interpreting and scaling iEEG models.

First, the practical value of learned representations depends on the preprocessing used by each system. Strong Multi-STFT baselines remain competitive with pretrained waveform systems, particularly on datasets with smaller downstream training sets. This provides a useful reference point for model development: engineered spectral features already support effective decoding with relatively simple models, while waveform models must learn useful features from less processed inputs. The normalization and filtering analyses further show that apparent model advantages can depend strongly on preprocessing choices. Evaluating learned representations against baselines under matched preprocessing is therefore essential for establishing what the model contributes beyond its input pipeline. Comparisons across tracks additionally assess whether these learned representations yield competitive decoding systems across institutional datasets.

Second, our scaling results show that the source of additional supervision matters for target-session decoding. Within-session scaling produces the largest and most consistent gains, while broader supervision yields smaller, task-dependent improvements. Several residual distribution shifts may contribute to this result. Although iMINDBench harmonizes electrode metadata, decoding targets, and preprocessing, the source recordings still differ in acquisition referencing schemes, electrode contact sizes and geometries, anatomical coverage, amplifier characteristics, and environmental noise. These differences can change the observed signal distribution even when datasets share a decoding target, potentially making additional institutional data difficult to exploit through pooling alone. Our experiments do not isolate the contributions of these individual factors, but they motivate representations that are robust or adaptable to heterogeneous recordings.

## 7 Limitations and Future Work

Evaluation scope. The current evaluation covers three naturalistic movie-watching datasets and assumes access to target-session calibration labels. Extending the benchmark to other stimulus paradigms and evaluation without target-session labels would test broader forms of generalization.

Dataset harmonization. Dataset harmonization is not fully automated: incorporating a new dataset can require manual alignment of electrode locations, brain-area labels, and other metadata, alongside dataset-specific stimulus processing. Automated alignment and validation would reduce this curation effort while preserving a consistent evaluation protocol.

Scaling coverage. The within-dataset and multi-dataset scaling experiments currently evaluate a single pretrained model, whose implementation supports joint supervised fine-tuning across subjects. The other pretrained systems are evaluated using linear decoding heads over output embeddings, without a corresponding mechanism for joint cross-subject fine-tuning in our implementation. Extending these evaluations to additional models would test how broadly the observed scaling pattern holds.

Model tuning. Our model comparisons also reflect the training recipes and tuning budgets used in this study; further optimization of pretraining and fine-tuning could change their relative performance. Appendix K explores this issue through a human-guided AutoResearch-style feasibility study, in which tuning a pretraining recipe improves downstream decoding. Such workflows could help establish stronger model baselines, provided that tuning budgets and validation-based selection procedures are documented consistently across models.

## 8 Conclusion

iMINDBench provides a shared framework for evaluating iEEG decoding across tasks, institutions, and preprocessing tracks. Our results show that model advantages depend on the input representation and evaluation setting, while additional supervised data from other subjects and institutions do not reliably improve target-session decoding in our scaling experiments. Progress on this benchmark will require models that extract useful features under limited supervision and learn more effectively from heterogeneous recordings, with representations robust or adaptable to differences in anatomy, acquisition, and recording conditions. Matched-preprocessing comparisons and cross-institution evaluations provide complementary ways to measure these advances toward general-purpose iEEG decoding.

## References

[1] Chenda Duan, Yipeng Zhang, Sotaro Kanai, Yuanyi Ding, Atsuro Daida, Pengyue Yu, Tiancheng Zheng, Naoto Kuroda, Shaun A. Hussain, Eishi Asano, Hiroki Nariai, and Vwani Roychowdhury. Omni-iEEG: A large-scale, comprehensive iEEG dataset and benchmark for epilepsy research. In International Conference on Learning Representations, 2026.

[2] Petr Nejedly, Vaclav Kremen, Vladimir Sladky, Jan Cimbalnik, Petr Klimes, Filip Plesinger, Filip Mivalt, Vojtech Travnicek, Ivo Viscor, Martin Pail, Josef Halamek, Benjamin H. Brinkmann, Milan Brazdil, Pavel Jurak, and Gregory Worrell. Multicenter intracranial EEG dataset for classification of graphoelements and artifactual signals. Scientific Data, 7(179), 2020.

[3] Andrii Zahorodnii, Christopher Wang, Bennett Stankovits, Charikleia Moraitaki, Geeling Chau, Andrei Barbu, Boris Katz, and Ila R. Fiete. Neuroprobe: Evaluating intracranial brain responses to naturalistic stimuli, 2025. arXiv preprint.

[4] Christopher Wang, Adam Uri Yaari, Aaditya K. Singh, Vighnesh Subramaniam, Dana Rosenfarb, Jan DeWitt, Pranav Misra, Joseph R. Madsen, Scellig Stone, Gabriel Kreiman, Boris Katz, Ignacio Cases, and Andrei Barbu. Brain treebank: Large-scale intracranial recordings from naturalistic language stimuli. In Advances in Neural Information Processing Systems, volume 37, 2024.

[5] Steven M. Peterson, Satpreet H. Singh, Benjamin Dichter, Michael Scheid, Rajesh P. N. Rao, and Bingni W. Brunton. AJILE12: Long-term naturalistic human intracranial neural recordings and pose. Scientific Data, 9(184), 2022.

[6] Josef Parvizi and Sabine Kastner. Promises and limitations of human intracranial electroencephalography. Nature Neuroscience, 21:474–483, 2018.

[7] Umit Keles, Julien Dubois, Kevin J. M. Le, J. Michael Tyszka, David A. Kahn, Chrystal M. Reed, Jeffrey M. Chung, Adam N. Mamelak, Ralph Adolphs, and Ueli Rutishauser. Multimodal single-neuron, intracranial EEG, and fMRI brain responses during movie watching in human patients. Scientific Data, 11(214), 2024.

[8] Julia Berezutskaya, Mariska J. Vansteensel, Erik J. Aarnoutse, Zachary V. Freudenburg, Giovanni Piantoni, Mariana P. Branco, and Nick F. Ramsey. Open multimodal iEEG-fMRI dataset from naturalistic stimulation with a short audiovisual film. Scientific Data, 9(91), 2022.

[9] Vinay Jayaram and Alexandre Barachant. MOABB: trustworthy algorithm benchmarking for BCIs. Journal ofNeural Engineering, 15(6):066011, 2018.

[10] Gayal Kuruppu, Neeraj Wagh, and Yogatheesan Varatharajah. EEG foundation models: A critical review of current progress and future directions. In NeurIPS 2025 Workshop on BrainBodyFM, 2025.

[11] Liuyin Yang, Qiang Sun, Ang Li, and Marc M. Van Hulle. Are EEG foundation models worth it? comparative evaluation with traditional decoders in diverse BCI tasks. In The Fourteenth International Conference on Learning Representations, 2026.

[12] Jean-Philippe Lachaux, Nikolai Axmacher, Florian Mormann, Eric Halgren, and Nathan E. Crone. Highfrequency neural activity and human cognition: Past, present and possible future of intracranial EEG research. Progress in Neurobiology, 98(3):279–301, 2012.

[13] Christopher Wang, Vighnesh Subramaniam, Adam Uri Yaari, Gabriel Kreiman, Boris Katz, Ignacio Cases, and Andrei Barbu. BrainBERT: Self-supervised representation learning for intracranial recordings. In International Conference on Learning Representations, 2023.

[14] Daoze Zhang, Zhizhang Yuan, Yang Yang, Junru Chen, Jingjing Wang, and Yafeng Li. Brant: Foundation model for intracranial neural signal. In Advances in Neural Information Processing Systems, volume 36, pages 26304–26321. Curran Associates, Inc., 2023.

[15] Geeling Chau, Christopher Wang, Sabera J Talukder, Vighnesh Subramaniam, Saraswati Soedarmadji, Yisong Yue, Boris Katz, and Andrei Barbu. Population transformer: Learning population-level representations of neural activity. In The Thirteenth International Conference on Learning Representations, 2025.

[16] Lucine L. Oganesian, Saba Hashemi, and Maryam M. Shanechi. BaRISTA: Brain scale informed spatiotemporal representation of human intracranial neural activity. In The Thirty-ninth Conference on Neural Information Processing Systems, 2025.

[17] Francesco S. Carzaniga, Michael Hersche, Abu Sebastian, Kaspar Schindler, and Abbas Rahimi. A foundation model with multi-variate parallel attention to generate neuronal activity. In The Fourteenth International Conference on Learning Representations, 2026.

[18] Danny Dongyeop Han, Yonghyeon Gwon, Ahhyun Lucy Lee, Taeyang Lee, Seong Jin Lee, Jubin Choi, Sebin Lee, Jihyun Bang, Seungju Lee, David Keetae Park, Shinjae Yoo, Chun Kee Chung, and Jiook Cha. DIVER-1: Scaling intracranial EEG foundation models for transferable representations, 2026. arXiv preprint.

[19] Weibang Jiang, Liming Zhao, and Bao-liang Lu. Large brain model for learning generic representations with tremendous EEG data in BCI. In The Twelfth International Conference on Learning Representations, 2024.

[20] Guangyu Wang, Wenchao Liu, Yuhong He, Cong Xu, Lin Ma, and Haifeng Li. EEGPT: Pretrained transformer for universal and reliable representation of EEG signals. In Advances in Neural Information Processing Systems, volume 37, 2024.

[21] György Buzsáki, Costas A Anastassiou, and Christof Koch. The origin of extracellular fields and currents—eeg, ecog, lfp and spikes. Nature reviews neuroscience, 13(6):407–420, 2012.

[22] Guangye Li, Shize Jiang, Sivylla E Paraskevopoulou, Meng Wang, Yang Xu, Zehan Wu, Liang Chen, Dingguo Zhang, and Gerwin Schalk. Optimal referencing for stereo-electroencephalographic (seeg) recordings. NeuroImage, 183:327–335, 2018.

[23] Eshani Patel, Yisong Yue, and Geeling Chau. Learning time-scale invariant population-level neural representations. In NeurIPS 2025 Workshop on Foundation Modelsfor the Brain and Body, 2025.

[24] Steven M. Peterson, Zoe Steine-Hanson, Nathan Davis, Rajesh P. N. Rao, and Bingni W. Brunton. Generalized neural decoders for transfer learning across participants and recording modalities. Journal of Neural Engineering, 18(2):026014, mar 2021.

[25] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33:12449–12460, 2020.

[26] Mehdi Azabou, Vinam Arora, Venkataramana Ganesh, Ximeng Mao, Santosh Nachimuthu, Michael Mendelson, Blake Richards, Matthew Perich, Guillaume Lajoie, and Eva L. Dyer. A unified, scalable framework for neural population decoding. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

[27] Vernon J. Lawhern, Amelia J. Solon, Nicholas R. Waytowich, Stephen M. Gordon, Chou P. Hung, and Brent J. Lance. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. Journal of Neural Engineering, 15(5):056013, 2018.

[28] Andrej Karpathy. autoresearch: AI agents running research on single-GPU nanochat training automatically. https://github.com/karpathy/autoresearch, 2026. GitHub repository.

## Appendix

## A Dataset Harmonization and Acquisition Metadata

Controlled cross-institution evaluation requires alignment along two benchmark-specific dimensions: decoding labels derived from independently collected movie stimuli and iEEG electrode metadata reported under different localization conventions. Together, these steps support common decoding targets and spatial metadata without treating the source datasets as uniform.

Task and temporal alignment. For the BYD and Pippi components, task and label alignment starts from the Neuroprobe/Brain Treebank stimulus-feature extraction pipeline, applied to each dataset’s video stimulus where source annotations support it. Task labels, scalar thresholds, and lexical/nonverbal row construction are defined in Appendix B. Sentence-level linguistic attributes are labeled using the Brain Treebank annotation protocol, and sentence and word onset times are initialized with Whisper automatic speech recognition (ASR) before manual alignment using 4× slowed video playback. We manually correct face\_num labels because the automatic face-count detector is noisy on these stimuli.

Electrode metadata alignment. Electrode layouts and cortical coverage vary by patient, and localization metadata use different coordinate systems and anatomical labels. Electrode locations are transformed into Brain Treebank’s left-posterior-inferior (LPI) coordinate space by first aligning coordinate dimensions to a sensible brain orientation and then matching the electrode coordinate ranges provided in Brain Treebank. For brain-area metadata, we harmonize labels to the Destrieux parcellation used by Brain Treebank: Pippi electrodes are overlaid onto Destrieux areas with a FreeSurfer alignment pipeline using the publicly available MRI scans, and BYD electrode labels are manually converted to the same Destrieux area set.

Remaining acquisition differences. Table 2 summarizes the remaining acquisition and stimulus differences relevant to multi-dataset model transfer, using metadata reported in the source dataset papers.

Table 2: Institution-specific stimulus and acquisition metadata. Dataset components differ in stimulus duration and language, electrode type and coverage, sampling, reference conventions, and coordinate reporting. BYD and Pippi entries focus on macroelectrode and clinical sEEG recordings, respectively.
<table><tr><td>Property</td><td>Neuroprobe / Brain Treebank [3, 4]</td><td>BYD / Cedars-Sinai [7]</td><td>Pippi / Utrecht [8]</td></tr><tr><td>Stimulus</td><td>English feature-length Hollywood 8-minute black-and-white films. Brain Treebank reports 26 Hitchcock excerpt with English films and 43.5 total hours; subjects watched 2.6 movies on average.</td><td>audio-visual content.</td><td>6.5-minute Dutch-dubbed short film from Pippi on the Run, with interleaved speech and music blocks.</td></tr><tr><td>Electrodes</td><td>sEEG depth probes with 6–16 mm diameter, 2 mm long.</td><td>Macroelectrodes on hybrid contacts per probe. Contacts: 0.8 Behnke-Fried depth electrodes.</td><td>Clinical sEEG recordings.</td></tr><tr><td>Sampling</td><td>2048 Hz.</td><td>1000 Hz.</td><td>2048 Hz.</td></tr><tr><td>Source filtering</td><td>60 Hz notch filtering and harmonics.</td><td>60 Hz notch and 0.1 Hz high-pass Source analysis applies 50 Hz filtering.</td><td>notch filtering.</td></tr><tr><td>Referencing</td><td>Neuroprobe evaluates several benchmark models.</td><td>Source processing uses Laplacian-rereferenced inputs for common-average rereferencing.</td><td>External mastoid reference during acquisition; common-average rereferencing in source analysis.</td></tr><tr><td>Localization</td><td>Electrode positions localized to an Native-space and MNI152 average cortical atlas.</td><td>coordinates provided.</td><td>Native-space electrode locations; MNI projection used for visualization.</td></tr></table>

sEEG: stereoelectroencephalography; MNI: Montreal Neurological Institute.

## B Decoding Tasks

Task definitions follow the Neuroprobe task suite [3] and Population Transformer benchmark protocol [15] where source annotations support them. For scalar binary tasks, Neuroprobe uses bottomversus-top quartiles, while BYD and Pippi use bottom-versus-top terciles to improve class support in the shorter stimuli. Lexical and surprisal tasks are restricted to word rows.

For BYD and Pippi visual and auditory tasks, feature tables include lexical word rows and sampled nonverbal rows. Nonverbal rows are sampled in 500 ms windows stepped every 500 ms from gaps longer than 2.0 s between the preceding word’s offset and the next word’s onset. They inherit the preceding word row’s sentence identifier and provide the negative class for speech and sentence-onset tasks.

Table 3: Aligned visual, auditory, and language decoding tasks. All tasks use binary classification on 1-second neural windows aligned to word onset or sampled nonverbal-interval onset. Classes are rebalanced before training.
<table><tr><td>#</td><td>Feature</td><td>Description</td><td>Benchmark task</td></tr><tr><td>1</td><td>onset (language)</td><td>Whether a new sentence starts.</td><td>Sentence onset vs. nonverbal interval.</td></tr><tr><td>2</td><td>speech (language)</td><td>Whether speech is present.</td><td>Word interval vs. nonverbal interval.</td></tr><tr><td>3</td><td>word_index (language)</td><td>Word index within its sentence.</td><td>First word (0) vs. second word (1); later words excluded.</td></tr><tr><td>4</td><td>word_gap (language)</td><td>Time from the preceding word&#x27;s offset to the current word&#x27;s onset within the same sentence (ms).</td><td>Low vs. high value.</td></tr><tr><td>5</td><td>gpt2_surprisal (language)</td><td>Negative log probability of the word under GPT-2, conditioned on the preceding 20 s of language context.</td><td>Low vs. high value.</td></tr><tr><td>6</td><td>word_head_pos (language)</td><td>Relative position of the word&#x27;s dependency-tree head.</td><td>Head to the left/root/self vs. head to the right.</td></tr><tr><td>7</td><td>word_length (language)</td><td>Word duration in milliseconds.</td><td>Low vs. high value.</td></tr><tr><td>8</td><td>word_part_speech (language)</td><td>Universal part-of-speech (UPOS) tag of the word.</td><td>Noun vs. verb; other tags excluded.</td></tr><tr><td>9</td><td>delta_volume (auditory)</td><td>Difference in average RMS audio power between the 500 ms Low vs. high value. windows before and after target onset.</td><td></td></tr><tr><td>10</td><td>volume (auditory)</td><td>Average root mean square (RMS) audio power.</td><td>Low vs. high value.</td></tr><tr><td>11</td><td>pitch (auditory)</td><td>Average audio pitch.</td><td>Low vs. high value.</td></tr><tr><td>12</td><td>global_flow (visual)</td><td>Camera-motion proxy computed as the maximal average dense optical-flow vector magnitude.</td><td>Low vs. high value.</td></tr><tr><td>13</td><td>local_flow (visual)</td><td>Large-displacement proxy computed as the maximal individual optical-flow vector magnitude.</td><td>Low vs. high value.</td></tr><tr><td>14</td><td>frame_brightness (visual)</td><td>Mean HSV value (V) over pixels.</td><td>Low vs. high value.</td></tr><tr><td>15</td><td>face_num (visual)</td><td>Maximum number of faces per frame during the target interval.</td><td>No face (0) vs. at least one face (≥ 1).</td></tr></table>

## C Additional Unit Decodability Subset Scorecards

Table 4 reports the same model and track rows as the Main scorecard on the All subset. Table 5 reports the same rows on the Challenge subset.
<table><tr><td>Track Model</td><td></td><td>Pretraining</td><td>Overall</td><td>Neuroprobe</td><td>BYD</td><td>Pippi</td></tr><tr><td colspan="7">Multi-STFT</td></tr><tr><td rowspan="4"></td><td>Logistic</td><td>not pretrained</td><td>0.587</td><td>0.641</td><td>0.576</td><td>0.545</td></tr><tr><td>MLP</td><td>not pretrained</td><td>0.592</td><td>0.646</td><td>0.578</td><td>0.551</td></tr><tr><td>CNN</td><td>not pretrained</td><td>0.583</td><td>0.654</td><td>0.570</td><td>0.525</td></tr><tr><td>PopT-v2</td><td>BrainTreeBank</td><td>0.599</td><td>0.659</td><td>0.585</td><td>0.553</td></tr><tr><td colspan="7">Waveform</td></tr><tr><td rowspan="5"></td><td>Logistic</td><td>not pretrained</td><td>0.541</td><td>0.589</td><td></td><td>0.515 0.520</td></tr><tr><td>MLP</td><td>not pretrained</td><td>0.554</td><td>0.612</td><td>0.523</td><td>0.527</td></tr><tr><td>CNN</td><td>not pretrained</td><td>0.554</td><td>0.616</td><td>0.524</td><td>0.523</td></tr><tr><td>HTNet</td><td>not pretrained</td><td>0.574</td><td>0.636</td><td>0.556</td><td>0.530</td></tr><tr><td>BaRISTA DIVER-1</td><td>BrainTreeBank private+ AJILE12</td><td>0.593 0.593</td><td>0.655 0.666</td><td>0.583 0.569</td><td>0.540 0.545</td></tr></table>

Table 4: All benchmark results. Within-session ROC-AUC performance on all supported units, grouped by the same fixed-track and model-native families as Table 1. Cells are shaded with a normalized color scheme within column and track. Best per column and track are bolded, and second-best values are underlined.

<table><tr><td>Track Model</td><td>Pretraining</td><td>Overall</td><td>Neuroprobe</td><td>BYD</td><td>Pippi</td></tr><tr><td colspan="6">Multi-STFT</td></tr><tr><td>Logistic</td><td>not pretrained</td><td>0.521</td><td>0.536</td><td>0.509</td><td>0.519</td></tr><tr><td>MLP</td><td>not pretrained</td><td>0.527</td><td>0.541</td><td>0.512</td><td>0.529</td></tr><tr><td>CNN</td><td>not pretrained</td><td>0.523</td><td>0.550</td><td>0.512</td><td>0.506</td></tr><tr><td>PopT-v2</td><td>BrainTreeBank</td><td>0.525</td><td>0.539</td><td>0.521</td><td>0.515</td></tr><tr><td colspan="6">Waveform</td></tr><tr><td></td><td>Logistic not pretrained</td><td>0.519</td><td>0.535</td><td>0.510</td><td>0.513</td></tr><tr><td>MLP</td><td>not pretrained</td><td>0.520</td><td>0.534</td><td>0.510</td><td>0.518</td></tr><tr><td>CNN</td><td>not pretrained</td><td>0.515</td><td>0.522</td><td>0.505</td><td>0.518</td></tr><tr><td>HTNet</td><td>not pretrained</td><td>0.509</td><td>0.528</td><td>0.500</td><td>0.500</td></tr><tr><td>BaRISTA</td><td>BrainTreeBank</td><td>0.523</td><td>0.541</td><td>0.513</td><td>0.514</td></tr><tr><td>DIVER-1</td><td>private+ AJILE12</td><td>0.526</td><td>0.548</td><td>0.518</td><td>0.512</td></tr></table>

Table 5: Challenge benchmark results. Within-session ROC-AUC performance on Challenge units, grouped by the same fixed-track and model-native families as Table 1. Cells are shaded with a normalized color scheme within column and track. Best per column and track are bolded, and second-best values are underlined.

## D BrainBERT Model-Native Scorecard

Table 6 evaluates BrainBERT [13] through its model-native single-STFT route and includes corresponding non-pretrained single-STFT decoders on the Main unit decodability subset. We evaluate the released BrainBERT checkpoint as a frozen feature extractor with a trained linear head. Direct single-STFT baselines achieve higher overall ROC-AUC under the benchmark protocol. The evaluation uses one-second windows and training-split normalization, differing from the checkpoint’s five-second pretraining windows and original normalization; it therefore characterizes the released representation under these conditions.

<table><tr><td>Track Model</td><td>Pretraining</td><td>Overall</td><td>Neuroprobe</td><td>BYD</td><td>Pippi</td></tr><tr><td colspan="6">BrainBERT STFT</td></tr><tr><td>Logistic</td><td>not pretrained</td><td>0.630</td><td>0.695</td><td>0.627 0.569</td><td></td></tr><tr><td>MLP</td><td>not pretrained</td><td>0.634</td><td>0.696</td><td>0.622</td><td>0.583</td></tr><tr><td>CNN BrainBERT</td><td>not pretrained</td><td>0.620</td><td>0.702</td><td>0.603 0.553</td><td></td></tr><tr><td>(frozen + linear head) BrainTreeBank</td><td></td><td>0.619</td><td>0.664</td><td>0.619 0.575</td><td></td></tr></table>

Table 6: BrainBERT model-native preprocessing scorecard on Main units. Within-session ROC-AUC performance on Main units for BrainBERT and corresponding non-pretrained single-STFT decoders. BrainBERT is evaluated using frozen representations pretrained on BrainTreeBank, while Logistic, MLP, and CNN are fit directly on the dataset-specific single-STFT route. Overall is the unweighted mean of the three dataset-level scores; each dataset score is a unit-weighted mean. Cells are shaded within column, with the best value bolded and the second-best value underlined.

## E Main and Challenge Unit Coverage

Main Means across Datasets by Task  
![](images/9fbda146d5751ec165444dd1b33edab49c63e427872434e20729ee83ab21fb3c.jpg)  
Figure 5: Main and Challenge unit performance across tasks. Each point shows an evaluation unit’s fold-averaged ROC-AUC. The top panel reports test performance for the STFT screening decoder, while the bottom panel reports the validation screening score: the maximum mean validation ROC-AUC across the STFT screening decoder and HTNet at 500 Hz. The dashed line marks the Main-unit threshold (validation ROC-AUC > 0.60), and the solid line marks chance performance.

![](images/f93fb59350e9fecd4e9ab01667d9db4d22dca16bb96c77538f00e85111278fb6.jpg)  
Figure 6: Main-unit task breakout by model family. Bars show mean test ROC-AUC on Main units for each task and model family, with points showing dataset means. The dashed horizontal line marks chance performance. Error bars indicate SEM across the 3 dataset means.

![](images/e8738692dfab0be956e72030ae927423259d8b5eba3936ab6de438970ab3d033.jpg)  
Figure 7: Main subject/session coverage across benchmark tasks. Cells show fold-averaged test ROC-AUC from the STFT screening decoder for Main subject/session–task units. Blank cells correspond to Challenge or unsupported units, and higher values indicate stronger screening-decoder test performance.

![](images/3eda8fb7536085a6d9bfa2ea928dfabe49beb607d65d305612b08ff0fde4a270.jpg)  
Figure 8: Neuroprobe within-session sample efficiency by task. Each panel decomposes Figure 4a by decoding task and model family, showing unit-weighted mean test ROC-AUC as the available target-session training data increase from one-sixteenth to the full training set. Only complete-case task/subject-session/fold units containing every model family and data fraction are retained. Panel titles report the number of unique supported subjects (n), and the dashed horizontal line marks chance performance.

## G Task-Level Scaling Trajectories

![](images/a572323f89466727287b752b13fd4bc04acae1100aba34757c837e2c4c15e3a8.jpg)  
Figure 9: Task-level scaling trajectories across datasets. For PopT-v2, points show the change in ROC-AUC for each decoding task relative to within-session evaluation when scaling to within-dataset and multi-dataset training; within-session values are therefore fixed at zero. Rows correspond to Neuroprobe, Bang! You’re Dead (BYD), and Pippi. Error bars denote 95% confidence intervals, and labels above each task report the number of unique supported subjects (n). Positive values indicate improvement over within-session training, while negative values indicate reduced performance.

## H Multi-STFT Input Example

![](images/4d08ca9fc67e3a076203b3c119c1677be13e85a075c8c558ee52853c0bdc7c36.jpg)

Figure 10: Example Multi-STFT input representation. The low-, mid-, and high-frequency sections use 0.5-, 0.25-, and 0.125-second windows with an approximately 62-ms hop and target 2–40, 20–150, and 80–250 Hz, respectively. The sections are concatenated along frequency with aligned time bins. Colors show feature magnitudes standardized per channel and frequency bin using training-split statistics.
<table><tr><td>Setting</td><td>Low</td><td>Mid</td><td>High</td></tr><tr><td>Window samples at 2048 Hz</td><td>1024</td><td>512</td><td>256</td></tr><tr><td>Window samples at 1000 Hz (BYD)</td><td>500</td><td>250</td><td>125</td></tr><tr><td>Retained upper frequency at 2048 Hz (Hz)</td><td>40</td><td>148</td><td>248</td></tr></table>

Table 7: Exact Multi-STFT settings. Window sample counts specify nperseg; hops are 128 samples at 2048 Hz and 62 samples at 1000 Hz. Retained upper frequencies reflect discrete-bin spacing.

## I Model Details

Generic decoders. Logistic regression uses a scikit-learn linear classifier on flattened alignedchannel inputs with maximum 10,000 iterations, tolerance $1 0 ^ { - 3 }$ , and random seed 42. The MLP flattens each input sample and applies two 128-unit hidden layers with ReLU activations and 0.2 dropout. The CNN uses three convolutional blocks with 32, 64, and 128 channels, kernel size 3, max pooling, and 0.5 dropout, followed by a 128-unit fully connected layer. Convolutions are 1D for waveform inputs and 2D for STFT-style inputs.

We keep these model hyperparameters fixed across datasets, tasks, and the Multi-STFT and Waveform tracks. The MLP and CNN use Adam, learning rate $1 0 ^ { - 4 }$ , batch size 200, and validation-based early stopping for up to 100 epochs. For waveform inputs, filtering uses a 15-second context around each one-second target window to reduce boundary effects; the signal is then cropped to the target window before rereferencing, resampling, and normalization.

PopT-v2. PopT-v2 uses a PopT-style Transformer to aggregate sparse and variable electrode sets across subjects [15]. It has 6 layers, hidden dimension 512, 8 attention heads, 2048-dimensional feed-forward layers, multi-subject positional encoding, and a classification-token pretraining head.

We pretrain PopT-v2 from scratch on the Brain Treebank subset used by PopT: 10 subjects acros 19 movie recordings [15]. Following TSAP’s finding that PopT pretraining benefits from matching pretraining and downstream window lengths [23], we use one-second Laplacian Multi-STFT windows with the duration-defined track settings and per-channel keyed standardization. Pretraining uses a next-segment prediction and token-replacement objective, with 1-second windows sampled every 0.2 seconds, variable channel subsets of 10–100 channels, and 10% token replacement. Training runs for 1,000,000 steps with LAMB, learning rate $1 0 ^ { - 4 }$ , batch size 256, gradient clipping at 1.0, a 2.5% warmup ramp-up schedule with decay factor 0.99, and bfloat16 mixed precision. Pretraining takes 48 hours on a single A100 GPU.

For benchmark fine-tuning, we update the pretrained Transformer and a new classification head with AdamW, weight decay $1 0 ^ { - 2 }$ , learning rates $5 \times 1 0 ^ { - 5 }$ and $5 \times 1 0 ^ { - 4 }$ respectively, batch size 128, and 1,000 gradient-update steps.

HTNet. We evaluate an HTNet-style implementation [24], based on the EEGNet-derived architecture [27], on one-second waveform inputs resampled to 500 Hz. We retain $F _ { 1 } = 8$ , depth multiplier $D = 2 , F _ { 2 } = 1 6$ , and dropout 0.5, and adapt the temporal and separable kernel lengths to 32 and 8 samples for one-second inputs. Training uses Adam, learning rate $1 0 ^ { - 3 }$ , batch size 128, and validation-based early stopping with patience 10 for up to 100 epochs. HTNet uses the same filtering and cropping procedure as the generic waveform decoders.

BaRISTA. BaRISTA [16] is a self-supervised intracranial waveform model whose central design feature is spatial encoding and masking at configurable spatial scales beyond individual channels: temporal patch representations can be encoded and masked at the level of individual channels, parcels, or lobes, making the granularity of spatial representation an explicit modeling choice. It is pretrained on 30 hours of Brain Treebank data, segmented into 3-second chunks sampled at 2048 Hz with 250 ms temporal patches. Pretraining excludes all recording sessions used for testing in the benchmark. Pretraining waveforms are Laplacian rereferenced, notch filtered, and high-pass filtered at 0.5 Hz. Destrieux atlas labels define the spatial categories, following the original paper’s best configuration: parcel-level encoding and channel-level masking.

For benchmark evaluation, we freeze the tokenizer and retain the 250 ms temporal patch size. Finetuning uses learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 2 }$ , batch size 128, and early stopping on validation ROC-AUC with patience 15 for up to 100 epochs. We extend the Destrieux spatial encoding to Destrieux-ASEG to support more subcortical regions.

Evaluation filtering is performed separately within each recording and train, validation, or test split. Selected windows are ordered by start time, concatenated for 0.5 Hz high-pass and notch filtering, then divided back into the original windows; the procedure does not filter the continuous recording or mix samples across splits. BYD is filtered at 1000 Hz and upsampled to 2048 Hz; Neuroprobe and Pippi remain at 2048 Hz. Filtering is followed by Laplacian rereferencing, robust global scaling, and per-window, per-channel temporal standardization before tokenization.

DIVER-1. We evaluate the 0.1-second-patch, Tiny-width variant of DIVER-1 [18]. It was pretrained to reconstruct masked raw inputs and STFT/FFT features at the patch level for 32 epochs on private iEEG data and AJILE12 [5], totaling 5,310 hours. Pretraining inputs use a 0.5 Hz high-pass filter and notch filtering.

Benchmark evaluation uses the original paper’s frozen setting: learning rate $2 \times 1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 2 }$ , batch size 32, and early stopping on validation ROC-AUC for up to 40 epochs. DIVER-1 uses Montreal Neurological Institute (MNI) coordinates when available; Pippi is evaluated without coordinate inputs because the benchmark release does not include Pippi MNI coordinates.

The DIVER-native evaluation route filters a 15-second context with a 0.5 Hz high-pass filter followed by notch filters at 60, 120, and 180 Hz, crops to the one-second target window, applies Laplacian rereferencing, and resamples to 500 Hz. No additional standardization is applied.

Waveform preprocessing. Table 8 summarizes the sampling, filtering, and normalization conventions for each system. Baseline comparisons did not show consistent improvements from higher sampling rates. BaRISTA and DIVER-1 retain model-specific preprocessing and pretraining exposure, so comparisons with the fixed-track baselines evaluate complete systems.

<table><tr><td>System</td><td>Input rate</td><td>Filtering procedure</td><td>Input normalization</td></tr><tr><td>Logistic, MLP, CNN, 500 Hz HTNet</td><td></td><td>dow.</td><td>Filter with 15-second context, then Training-fitted robust global normalization; crop to the one-second target win- one median and median-absolute-deviation scale across channels and timepoints.</td></tr><tr><td>BaRISTA</td><td>2048 Hz</td><td>ing and split for filtering; upsample tion. BYD from 1000 Hz.</td><td>Concatenate selected windows in Robust global scaling followed by per- temporal order within each record- window, per-channel temporal standardiza-</td></tr><tr><td>DIVER-1</td><td>500 Hz</td><td>crop to the one-second target win- dow.</td><td>Filter with 15-second context, then No additional input standardization.</td></tr></table>

Table 8: Waveform-track preprocessing recipes. All systems use 0.5 Hz high-pass filtering, linenoise notch filtering, and Laplacian rereferencing. Model-specific variants retain their sampling and normalization conventions.

## I.1 Additional model coverage

MVPFormer and Brant use temporal input representations that require additional adaptation for iMINDBench’s one-second naturalistic decoding windows.

MVPFormer. The released MVPFormer configuration we examined [17] encodes each channel in five-second segments and combines 25 segments of temporal context. This temporal granularity differs from iMINDBench’s one-second prediction windows. Retaining the released context would provide information beyond the benchmark window and require explicit handling of chronological fold boundaries; shortening or rescaling the input changes the signal presented to the pretrained encoder. Including MVPFormer therefore requires a validated mapping between its temporal representation and the benchmark’s prediction targets.

Brant. Brant’s published preprocessing [14] uses six-second patches with both temporal and frequency-domain representations. Adapting one-second recordings to this input format changes the temporal support of those representations. In particular, stretching a short window to the expected patch length changes its effective timescale and requires care when computing frequency features. A benchmark-compatible adaptation must establish how these representations are constructed while preserving the intended signal interpretation.

We defer inclusion of these models in the main comparison pending validation of these adaptations. These cases motivate documenting model-specific input requirements alongside standardized benchmark protocols.

## I.2 Benchmark Runtime Estimates

Cumulative per-job timing records are normalized to four concurrent workers. Preprocessing and CPU fitting assume a node with two 32-core AMD EPYC 7513 processors; neural fitting assumes four NVIDIA A100 SXM4 GPUs (80 GB each), with one worker per GPU. Estimates cover all three datasets and both folds and are not measured end-to-end wall-clock times.

<table><tr><td>Preprocessing route</td><td>Worker-hours</td><td>Four-worker wall time</td></tr><tr><td>Multi-STFT</td><td>~5.5 h</td><td>~1.4 h</td></tr><tr><td>Waveform</td><td>~80 h</td><td>~20 h</td></tr></table>

Table 9: Estimated preprocessing time. Waveform preprocessing is estimated from the HTNet runs that built the cache reused by waveform Logistic, MLP, and CNN.

<table><tr><td>Track</td><td>Model</td><td>Fitting worker-hours</td><td>Four-worker wall time</td></tr><tr><td>Multi-STFT</td><td>Logistic</td><td>0.8 h</td><td> ${ \sim } 0 . 2 \mathrm { h }$ </td></tr><tr><td>Multi-STFT</td><td>MLP</td><td>3.9 h</td><td>~1.0 h</td></tr><tr><td>Multi-STFT</td><td>CNN</td><td>5.6 h</td><td>~1.4 h</td></tr><tr><td>Multi-STFT</td><td>PopT-v2</td><td>72.6 h</td><td>~18.2 h</td></tr><tr><td>Waveform</td><td>Logistic</td><td>0.4 h</td><td>~0.1 h</td></tr><tr><td>Waveform</td><td>MLP</td><td>79.1 h</td><td>~19.8 h</td></tr><tr><td>Waveform</td><td>CNN</td><td>64.0 h</td><td>~16.0 h</td></tr><tr><td>Waveform</td><td>HTNet</td><td>14.1 h</td><td>~3.5 h</td></tr><tr><td>Model-native</td><td>BaRISTA</td><td>27.4 h</td><td>~6.8 h</td></tr><tr><td>Model-native</td><td>DIVER-1</td><td>68.0 h</td><td>~17.0 h</td></tr></table>

Table 10: Estimated model-fitting time. Fitting excludes subject loading and preprocessing.

## J Leaderboard

The interactive leaderboard is available at https://imindbench.github.io/.

![](images/19ee1cb20506738bddb8b76a370cb74e8f3d482bf03548905135fd760eb33fc5.jpg)  
Figure 11: Decomposed results viewer. The leaderboard supports comparisons across datasets, tasks, preprocessing tracks, scaling domains, and unit decodability subsets.

## K AutoResearch Feasibility Study

As a scoped proof of use, we applied an AutoResearch-style agentic experimentation loop [28] to tune the PopT-v2 pretraining recipe starting from a fixed 780k-step single-STFT checkpoint. The best selected variant improved Neuroprobe within-session decoding over the original 1M-step single-STFT baseline by ∆ROC-AUC = 0.0066 (95% CI: 0.0021–0.0111, p = 0.0049; Figure 12). This demonstrates how automated tuning systems may support scalable, controlled model comparisons.

Pretraining Loss vs Decoding Gain  
![](images/5464ca6fdae6f06196f9cc6f61d4d001b19a88a629b7b8552edc34a7905039fa.jpg)  
Figure 12: AutoResearch feasibility result. AutoResearch-selected PopT-v2 variants with lower pretraining validation loss (colors) generally improve Neuroprobe within-session decoding. Points show variants evaluated across 150 paired units; error bars show 95% bootstrap confidence intervals.

AutoResearch was run in a human-in-the-loop mode from the same 780k-step single-STFT PopT-v2 checkpoint, with human decisions limited to approving phase transitions and extending promising candidates. The search varied optimization and regularization hyperparameters (learning rate, scheduler type, batch size, gradient clipping, weight decay, and dropout) while model architecture, dataset, preprocessing track (laplacian\_stft\_time\_pooled), split policy (per-subject, seed 42, 1% validation / 10% test), and validation objective (nsp\_replace\_only\_pretrain loss) were held fixed. Across 26 launched jobs (25 valid), the sweep consumed approximately 228.3 GPU-hours over approximately 184.5 active wall-clock hours. Initial screening produced small or inconsistent effects; dropout removal, identified within the initial 780k–820k window (approximately 16.2 GPU-hours, compared to approximately 48 GPU-hours for a full 1M steps pretraining run), was the first robust improvement and was subsequently extended to 2M steps with variants of both scheduled and flat learning rates. As shown in Figure 13, no-dropout variants separate clearly from dropout variants beyond approximately 1.2M steps, with the best model (2M no-dropout flat-LR) achieving validation loss 0.687 versus 0.829 for the original 1M baseline. Because this model uses a single-STFT rather than the benchmark’s Multi-STFT track, it is not directly comparable to Table 4; however, its within-session All-subset performance of 0.664 sits above the Multi-STFT PopT-v2 baseline (0.599 overall, 0.659 on Neuroprobe). The total compute cost was approximately \$342 (228.3 GPU-hours at \$1.50/A100-hour).

![](images/d5122d2a9e38983e63d27057cd13cdc8719b21b299f95036b31e6930ba668ff9.jpg)

Figure 13: AutoResearch pretraining validation curves. All runs continue from a shared 780k-step checkpoint. The 1M baseline (dropout=0.1, flat LR) and 1M no-dropout model terminate early; the four 2M continuations separate beyond approximately 1.2M steps, with no-dropout variants consistently outperforming their dropout counterparts. The best model, 2M no-dropout flat-LR, achieves the lowest final validation loss (0.687), confirming dropout removal as the primary driver of improvement and the flat learning rate as an additional benefit at longer horizons.