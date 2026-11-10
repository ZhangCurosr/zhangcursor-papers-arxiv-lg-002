# Spike-to-Field Mechanisms of Turbulence-Like Dynamics in Spatial Spiking Neural Networks

Zequan Liu

ZeQome

Zurich, Switzerland

zequan@zeqome.com

## Abstract

Neural avalanches and turbulence-like brain dynamics are usually measured after neural activity has been coarse-grained into continuous fields, leaving open which spike-level mechanisms can generate these macroscopic observables. We build a spatial leaky integrate-and-fire spiking neural network and explicitly transform spikes into activity-density and Hilbert phase fields. A delayed, distance-dependent, passive-dendrite regime reproducibly generates turbulence-like field structure, including high-amplitude phase-defect tracks, structure-function scaling, and powerlaw-favored continuous-field avalanche tails. Controls show that firing rate, vortex count, and avalanche tail fits are individually insufficient: destroying delayed propagation, dendritic filtering, or spike timing suppresses key field signatures, while random-space nulls can retain many mathematical phase defects but show weaker spatial structure and synaptic-delay transfer. Biologically motivated gates further shape the regime: PV-like perisomatic inhibition clamps phase dynamics, SST-like dendritic inhibition and projection-specific inhibitory facilitation preserve long-lived tracks, and a rate-constrained projection-STP regime expresses a roughly 200-ms episode at 12.25 Hz (full-record mean) across 12 seeds. These results use SNNs as a mechanistic testbed for spike-derived brain turbulence observables and show why phase defects and avalanche-like statistics must be interpreted jointly with spatial scaling and spike-level propagation.

## 1 Introduction

Large-scale brain activity is often modeled as a continuous dynamical field, even though it ultimately emerges from discrete spikes and synaptic events. This separation of levels is useful, but it leaves a mechanistic gap: if turbulence-like phase and amplitude statistics are measured in empirical neuroimaging, what spike-level circuit mechanisms can generate them? Conversely, if neuronal avalanches are measured as scale-free cascades in activity, can they be connected to the same spatial phase structures used in turbulence analyses?

Recent work on human brain dynamics has proposed that intrinsic activity can exhibit turbulence-like signatures that support cross-scale information transfer [5]. Related work on cortical travelling and metastable waves shows that neural activity can organize into propagating, rotating, and source/sinklike spatiotemporal patterns across scales [12, 14]. In parallel, the neuronal avalanche literature has emphasized scale-free activity cascades as a signature of critical dynamics [1, 8, 6]. Both lines of work rely on a coarse-grained description: empirical fMRI, MEG, LFP, or binned population activity is treated as a field. The present paper studies the spike-level mechanisms that could support such turbulence dynamics by asking whether a minimal spatial SNN can generate the same classes of observables after an explicit spike-to-field transformation.

We focus on mechanism rather than whole-brain reconstruction. A small two-dimensional SNN is not intended to replace empirical studies of human brain turbulence; instead, it gives an executable bridge between three descriptions: binary spike rasters, continuous activity-density and phase fields, and turbulence-like statistics. This bridge allows us to test which microscopic ingredients support measured field organization: distance-dependent recurrent connectivity, finite conduction delays, excitation/inhibition balance, dendritic filtering, and projection-specific short-term synaptic plasticity (STP).

We use the term “turbulence-like” operationally. In this paper it denotes a joint pattern of spikederived field observables: spatially structured activity-density fluctuations, phase-defect tracks in high-amplitude regions, structure-function scaling, and spike-level delayed propagation. It does not mean that the model solves the Navier–Stokes equations, reproduces a whole human brain, or proves that any single graph motif is sufficient for empirical brain turbulence. These observables are not sufficient evidence of physical turbulence.

The contributions are:

• A reproducible pipeline from discrete SNN spikes to continuous density fields, Hilbert phase fields, phase vortices, avalanche candidates, and spatial structure functions.

• A mechanism-guided passive-dendrite delayed-propagation regime with reproducible highamplitude phase vortices across 12 random seeds.

• Control experiments showing that avalanche tail fits alone are insufficient: vortex density and vortex track lifetime distinguish mechanisms more sharply than power-law-vs-exponential avalanche comparisons.

• A separation between localized turbulence-field assays and classical avalanche assays: sparse post-stimulus rasters do not support a Beggs–Plenz criticality claim, whereas long weakdrive spontaneous-like records produce many spike-bin avalanches with branching ratios near one.

• A biologically motivated extension in which dendritic inhibition and projection-specific STP modulate turbulence-like dynamics, with SST-like dendrite-targeting inhibitory facilitation producing the highest long-track count in the tested set.

• A topology control showing that feedforward graphs favor direct pairwise spike transfer, whereas recurrent graphs preserve rate-matched activity re-entry, separating transmission from activity re-entry.

## 2 Related work

Brain turbulence and neural fields. Turbulence-like brain dynamics have been studied in wholebrain empirical and oscillator models, where amplitude fluctuations and their spatial correlations are used to quantify cross-scale communication [5]. These approaches are related to a longer tradition of neural field and large-scale dynamical modeling [18, 2], cortical travelling waves [12], metastable brain waves [14], and mesoscopic phase-vortex descriptions of brain activity [7]. Our goal is complementary: we study the circuit-level mechanisms that can generate the continuous-field observables used to study brain turbulence, starting from spikes in a spatially embedded recurrent circuit.

Neuronal avalanches. Neuronal avalanches were introduced as scale-free cascades in cortical circuits [1]. Later work has connected avalanche statistics to critical dynamics, scaling functions, cortical state transitions, and possible computational benefits [8, 6, 15]. Standard avalanche analyses use binned events and cluster sizes; our main turbulence-field pipeline computes connected highactivity volumes in a continuous density field. This is closer to a non-equilibrium field-theoretic view of avalanches, but it inherits threshold and smoothing choices. Because power-law-like tails can arise spuriously and sparse sampling can distort inferred branching statistics [3, 17, 13, 19], we also include a direct traditional spike-bin diagnostic on sparse electrode samples, distinguishing short post-stimulus records from long weak-drive spontaneous-like records.

Dendrites, inhibition, and STP. Dendrites are not passive cables in all regimes: pyramidal neurons contain domain-specific synaptic inputs, active conductances, and dendritically initiated events [16, 10]. Inhibition is also strongly target- and cell-type-specific, with PV-like perisomatic and SST-like dendritic effects representing different network controls. Short-term synaptic plasticity can be projection-specific: a single axon can signal differently to different targets [11], and inhibitory microcircuits use diverse synaptic dynamics [9]. These facts motivate our simplified tests of passive dendrites, active dendrites, dendritic inhibition, and projection-specific STP.

## 3 Model and analysis pipeline

## 3.1 Spatial LIF network

We simulate a $2 4 \times 2 4$ grid of leaky integrate-and-fire neurons. Each neuron is excitatory with probability 0.8 and inhibitory otherwise. The membrane voltage obeys

$$
\tau _ { m } \frac { d v _ { i } } { d t } = E _ { L } - v _ { i } + I _ { 0 } + I _ { i } ^ { \mathrm { s t i m } } ( t ) + I _ { i } ^ { \mathrm { s y n } } ( t ) + \xi _ { i } ( t ) ,\tag{1}
$$

with reset and refractory dynamics after threshold crossing. A short stimulus is applied to a fixed left-side patch only to initiate activity; subsequent waves and activity clusters are generated by recurrent interactions.

Outgoing connections are sampled according to an exponential distance rule,

$$
p _ { i j } \propto \exp ( - \| r _ { i } - r _ { j } \| / \lambda ) ,\tag{2}
$$

with separate out-degree parameters for excitatory and inhibitory neurons. Conduction delays are distance-dependent:

$$
d _ { i j } = 1 + \operatorname { r o u n d } ( \| r _ { i } - r _ { j } \| / v _ { c } ) ,\tag{3}
$$

clipped by a maximum delay. This makes wavefronts and collisions possible without prescribing a moving wave in the input, consistent with the broader view that recurrent cortical circuits can support travelling-wave dynamics [12].

The passive-dendrite candidate used in the main experiments has grid size 24, 340 time steps, passive dendrites, $\lambda = 8 . 5 5$ cells, conduction speed $v _ { c } = 1 . 0 6$ cells/ms, $k _ { E } = k _ { I } = 3 2$ , excitatory weight 11.97, inhibitory weight magnitude 12.31, tonic drive 11.31 mV, noise standard deviation 1.36 mV, stimulus amplitude 26.45 mV, dendrite coupling 0.759, and dendrite time constant 20.19 ms.

## 3.2 Spike-to-field transformation

Let $s _ { i } ( t ) ~ \in ~ \{ 0 , 1 \}$ denote whether neuron i spikes at time t. We reshape spikes onto the twodimensional grid and apply separable Gaussian smoothing in time and space:

$$
\rho ( \boldsymbol { r } , t ) = \frac { 1 } { \Delta t } \sum _ { i , t ^ { \prime } } s _ { i } ( t ^ { \prime } ) G _ { \sigma _ { r } } ( \boldsymbol { r } - \boldsymbol { r } _ { i } ) G _ { \sigma _ { t } } ( t - t ^ { \prime } ) .\tag{4}
$$

The resulting $\rho ( r , t )$ is a continuous activity-density field in Hz. To obtain a phase field, we band-pass filter $\rho$ in the 20–80 Hz range and apply the Hilbert transform at each spatial location:

$$
z ( r , t ) = A ( r , t ) e ^ { i \theta ( r , t ) } .\tag{5}
$$

Here $A ( r , t )$ is the local band-limited amplitude and $\theta ( r , t )$ is the phase field. Gaussian widths are 2 ms/1.2 cells; filtering uses third-order forward–backward Butterworth.

## 3.3 Turbulence-like observables

We compute phase vortices as topological defects on each plaquette. For plaquette $p$ with cyclic corner phases $\theta _ { p , 1 } , \ldots , \theta _ { p , 4 }$ and $\theta _ { p , 5 } = \theta _ { p , 1 }$

$$
q _ { p } = \frac { 1 } { 2 \pi } \sum _ { k = 1 } ^ { 4 } { \mathrm { w r a p } ( \theta _ { p , k + 1 } - \theta _ { p , k } ) } ,\tag{6}
$$

where $q _ { p } = \pm 1$ indicates a candidate vortex. We count only defects whose surrounding amplitude exceeds the 60th percentile, because phase is unreliable in low-amplitude regions. Vortices are linked across frames by same-sign nearest-neighbor matching with a maximum distance of 2.6 cells and gap of two frames, yielding track lifetimes.

We estimate continuous-field avalanche candidates by thresholding ρ and computing threedimensional connected components in $( x , y , t )$ . We fit discrete power-law tails by maximum likelihood over candidate $x _ { \mathrm { m i n } }$ values and compare against an exponential tail using AIC, following the principle that power-law claims require likelihood-based tail fitting rather than visual log-log regression [3]. We also compute a second-order spatial structure function,

$$
S _ { 2 } ( \ell ) = \big \langle | \rho ( r + \ell , t ) - \rho ( r , t ) | ^ { 2 } \big \rangle _ { r , t } ,\tag{7}
$$

and report the log-log slope over the central lag range.

The key point is that no single observable defines turbulence-like dynamics in this setting. Density fields measure amplitude organization, phase fields measure local oscillatory timing, vortices measure phase topology, structure functions measure spatial scaling, and synaptic-delay transfer checks whether field structure is supported by spike-level propagation.

## 4 Experiments

The experiments are organized around a compact main evidence chain. First, we show that a delayed spatial SNN can reproducibly generate spike-derived field structure. Second, we show that avalanche fits and vortex counts alone are insufficient by using mechanistic and rate-matched controls. Third, we test biological gates–dendritic inhibition, PV/SST/VIP microcircuits, and projection-specific STP–that move the same field dynamics into a lower-rate regime. Fourth, we separate two avalanche settings: sparse post-stimulus rasters are not enough to claim classical criticality, while long weakdrive spontaneous-like records can produce traditional spike-bin avalanches with branching ratios near one. Finally, a topology control separates direct spike transfer from recurrent field re-entry. Additional diagnostic figures are provided in Appendix A.

## 4.1 Candidate regime and reproducibility

We first instantiated a mechanism-guided passive-dendrite operating point combining finite delayed propagation, balanced excitation/inhibition, noise-driven variability, and a localized trigger. The pilot candidate produced mean firing rate 105.18 Hz, mean high-amplitude vortices 6.58 per sampled frame, peak high-amplitude vortices 37, vortex occupancy 0.54, and $S _ { 2 }$ slope 0.965.

We then fixed the candidate parameters and reran 12 random seeds, resampling network connectivity, initial voltages, and noise. All 12 seeds passed an exploratory field criterion. The mean firing rate was 115.43 Hz, mean high-amplitude vortices were 6.05, mean vortex tracks with lifetime at least 8 frames were 82.5, and the mean structure-function slope was 0.94.

This baseline should be interpreted as a high-excitability operating point rather than a biologically typical spontaneous cortical firing-rate regime. Its role is to expose a reproducible spike-derived field state before adding circuit mechanisms that can regulate activity level. The projection-specific STP experiment below shows that similar long-lived phase dynamics can also occur episodically at much lower mean firing rate.

## 4.2 Avalanche tail fitting

Pooling avalanche candidates across the 12-seed candidate regime yielded 1825 samples. The fitted power-law tail had $x _ { \operatorname* { m i n } } = 4$ , 747 tail samples, exponent $\alpha = 2 . 0 6$ , and $\Delta \mathrm { A I C } = \mathrm { A I C } _ { \exp } -$ $\mathrm { \bar { A } I C _ { p o w e r } = 5 3 4 7 . 2 }$ , favoring the power-law tail over the exponential alternative. Each of the 12 individual seeds also favored the power-law tail by AIC.

This result is encouraging but not by itself sufficient evidence for turbulence. Avalanche statistics depend on thresholding, smoothing, and finite-size effects, and several controls below also show power-law-favored tails. We therefore treat avalanche tail fits as one axis of evidence, not the defining criterion. Five-family tail comparisons retain power-law preference in 12/12 high-rate records, versus 8/11 episodes; cutoffs remain power-law-selected.

Table 1: Mechanistic controls over 12 seeds. Power-law-favored avalanche tails remain common, while vortex statistics distinguish propagation and dendritic mechanisms more sharply.
<table><tr><td>Model</td><td>Rate</td><td>Vortices</td><td>Tracks  $\geq 8$ </td><td> $S _ { 2 }$  slope</td><td>α</td></tr><tr><td>Passive + distance delay</td><td>115.43</td><td>6.05</td><td>82.50</td><td>0.94</td><td>2.05</td></tr><tr><td>No distance delay</td><td>128.67</td><td>3.70</td><td>58.33</td><td>0.96</td><td>2.01</td></tr><tr><td>Short range + delay</td><td>113.65</td><td>6.10</td><td>92.00</td><td>1.15</td><td>1.77</td></tr><tr><td>No passive dendrite</td><td>164.25</td><td>2.29</td><td>37.25</td><td>0.98</td><td>2.06</td></tr><tr><td>Uniform fast control</td><td>129.23</td><td>4.28</td><td>64.83</td><td>0.84</td><td>1.92</td></tr></table>

## 4.3 Mechanistic controls

Table 1 summarizes the main controls. Removing distance-dependent delays reduced mean highamplitude vortices from 6.05 to 3.70 and long tracks from 82.5 to 58.3. Removing passive dendrites reduced vortices further to 2.29 and long tracks to 37.25 while increasing mean firing rate to 164.25 Hz. A short-range delayed regime preserved vortex statistics, suggesting that finite delayed propagation is more important than simply having long connections.

## 4.4 Baseline and mechanism-control field morphology

To make the contrast explicit, we consolidated the candidate regime and several mechanism-ablated or inhibition-clamped controls into one comparison. The baseline candidate has 6.05 mean highamplitude vortices and 82.5 long tracks. Removing distance-dependent delays reduces these to 3.70 vortices and 58.33 long tracks. Removing passive dendrites increases firing rate to 164.25 Hz but reduces vortices to 2.29 and long tracks to 37.25, showing that high firing rate alone is not sufficient for turbulence-like dynamics. A PV-clamped microcircuit is an even stronger negative control, with only 0.35 vortices and 4.08 long tracks. In contrast, an SST-only dendritic-inhibition circuit preserves turbulence-like dynamics, with 5.93 vortices and 94.33 long tracks.

Figure 1 gives the visual version of this comparison. The left column shows the raw spiking raster on the two-dimensional grid before any field analysis. The next columns show how those same spikes become a smoothed activity-density field, a phase field with detected high-amplitude vortices, and a population spike-count trace. This figure is important because the field observables are not introduced as abstract post-processing objects; they are visually traceable back to discrete spike patterns.

## 4.5 Rate-matched spike nulls

The previous controls change both mechanism and firing statistics. We therefore added rate-matched spike nulls that start from the exact same candidate spike tensors and preserve total spike count. A global-rate shuffle preserves the number of spikes in each frame but randomizes their spatial locations; it reduces mean vortices from 6.05 to 0.69 and long tracks from 82.5 to 8.92. Independent circular time shifts preserve each cell’s spike count and spatial position but destroy simultaneity; they reduce vortices to 1.80 and long tracks to 29.25. These controls indicate that firing rate alone is not sufficient.

The fixed spatial-permutation null is more diagnostic. It preserves every spike train and all pairwise temporal synchrony but assigns cells to random spatial positions. This null retains many phase defects (5.72 vortices and 84.0 long tracks) while reducing the structure-function slope from 0.94 to 0.72. We also measured excitatory synaptic-delay transfer: for each excitatory edge, we counted whether the postsynaptic neuron spiked at the edge-specific conduction delay after a presynaptic spike, and subtracted a random-target control. The original SNN has a positive transfer excess $\mathrm { \bar { ( 1 . 5 8 ) } \times 1 0 ^ { - 3 } ) }$ , whereas the global-rate shuffle and fixed spatial permutation are near zero $( - 1 . 4 5 \times 1 0 ^ { - 4 }$ and $- 1 . 3 8 \times 1 0 ^ { - 5 } )$ . Thus, the spatial permutation can preserve phase-defect counts while disrupting edge-aligned spike timing. We therefore interpret turbulence-like dynamics as a joint signature of phase defects, long tracks, spatial scaling, and delayed propagation, not as a vortex-count threshold alone (Table 2).

![](images/5158b56c14dac71de6e9d631773ba9fc11da2e23a9ae056e69ab631115b93d2f.jpg)

![](images/dd78c975c677700f626fdf5f3b25e2e7f34f561d7fd5a133d6a2c67a7e5eb3a3.jpg)  
Figure 1: Visual spike-to-field comparison. Each row is a different SNN regime and each column shows a different representation of the same simulated activity: raw spikes on the grid, smoothed activity density $\rho ,$ Hilbert phase with high-amplitude vortex markers, and the population spike-count trace. The turbulence-like candidate has structured spike clusters that coarse-grain into moving density and phase patterns; the no-passive-dendrite high-rate control fires strongly but produces weaker long-lived phase organization; the PV-clamped control suppresses both spiking and field structure.

Table 2: Positive contrasts (pointwise 95% paired seed-bootstrap intervals; 20,000 resamples; 12 seeds/state). $D \colon$ unfiltered-density $S _ { 2 } ; T \colon$ : raw-spike delay transfer; $P \colon$ defect/track counts. Episode shifts preserve within-window counts; this is not turbulence specificity.
<table><tr><td>State</td><td></td><td>Global-rate shuffle Spatial permutation Time shift</td><td></td></tr><tr><td>High-rate baseline</td><td> $D , T , P$ </td><td> $D , T$ </td><td> $D , P$ </td></tr><tr><td>Low-rate episode</td><td> $D , T$ </td><td> $D , T , P$ </td><td> $T , P$ </td></tr></table>

## 4.6 Dendritic inhibition

We next asked whether inhibitory input location changes the turbulence-like observables. A PVlike soma-only inhibition condition reduced mean vortices to 3.40 and long tracks to 54.83. Mixed PV/SST routing recovered mean vortices to 5.95 and increased long tracks to 90.33. SST-rich dendritic inhibition produced similar vortex density and 86.0 long tracks. This supports the interpretation that perisomatic inhibition clamps firing and propagation, while dendritic inhibition reshapes the integration window in a way that better preserves spatial phase structure.

## 4.7 Explicit inhibitory microcircuits

We then replaced the lumped inhibitory population with explicit PV-like, SST-like, and VIP-like inhibitory subtypes. PV-like neurons route inhibition preferentially to the soma, SST-like neurons route inhibition preferentially to dendrites, and VIP-like neurons preferentially inhibit SST-like neurons, implementing a simplified disinhibitory gate.

This cell-type split strongly affected the spike-derived field observables. A lumped inhibitory baseline produced 6.15 mean high-amplitude vortices and 89.58 long tracks. A PV-only network nearly eliminated phase-defect dynamics, with 0.35 mean vortices and 4.08 long tracks. An SST-only network preserved measured phase structure, with 5.93 mean vortices and 94.33 long tracks. A PV+SST network was strongly suppressed (1.07 mean vortices, 15.92 long tracks), while adding VIP partially restored phase structure (3.77 mean vortices, 62.0 long tracks). Thus, explicit inhibitory cell-type structure supports the same qualitative mechanism suggested by the routing experiment: perisomatic inhibition clamps propagation, dendritic inhibition preserves structured phase dynamics, and VIP-like disinhibition can reopen the dendritic pathway.

![](images/f156a2cfb20195a14ddd92019ea8ce465e1c37aad7e712c83c38b260141fa801.jpg)

![](images/46e5b1e6ed69046fc29c209678c793d7d00078ec4475cc059b8cf2d18b41a445.jpg)

![](images/e2c308edf1c841fb51853783a07ed6d859e4c0b15908861aa063eb7b7972c08b.jpg)

(a) Local phase-defect zoom. The highlighted plaquette has a wrapped phase winding of approximately −2π, making the center a topological phase defect rather than an ordinary high-activity point. 3D perspective of high-amplitude phase-defect tracks  
![](images/b77fbbd11264068b279e46008c27bfa8c200182cfa7a7f92af9dcaaeccbfdf28.jpg)  
(b) Three-dimensional perspective of high-amplitude phase-defect tracks in $( x , y , t )$ . The thick trajectory is the same defect track used for the local zoom.  
Figure 2: Visual audit of phase defects and vortex tracks. Single-frame phase defects are detected by local phase winding; tracks link same-charge defects across time, turning a topological singularity into a space-time trajectory.

## 4.8 Microcircuit phase diagram

Finally, we asked whether the PV/SST/VIP result is an isolated operating point or part of a broader parameter region. We fixed the VIP fraction at 0.15 and mapped the PV fraction among non-VIP inhibitory neurons against the VIP→SST disinhibition scale, using four seeds per grid point. The strongest morphology score occurred at a low PV setting (PV axis 0.25, actual PV fraction 0.2125, actual SST fraction 0.6375) with VIP→SST scale 0.5, yielding 6.37 mean vortices and 92.75 long tracks. Fully PV-dominated networks were suppressed across all VIP→SST strengths, with about

0.71 mean vortices and 10.75 long tracks. Fully SST-dominated networks preserved strong vortex dynamics, but the low-PV mixed regime scored higher under the tested heuristic.

## 4.9 Short-term synaptic plasticity

A presynaptic-type STP model showed that STP does not simply increase turbulence-like structure. Pure short-term depression reduced mean vortices to 0.41 and long tracks to 5.67. Pure facilitation increased firing rate to 163.62 Hz but did not exceed the baseline vortex density. A coarse motif with excitatory depression and inhibitory facilitation preserved long tracks (85.5) at a low firing rate (21.05 Hz), suggesting that STP can shift the network into intermittent low-rate dynamics rather than uniformly amplifying activity.

We then refined STP by projection type: $E  E , E  I , I  E$ , and I → I. This is projection-specific than assigning one STP rule to all excitatory or inhibitory outputs. The strongest tested motif was SST-like $I \to E$ facilitation, with 8.05 mean high-amplitude vortices, 120.42 long tracks, and 15.62 Hz mean firing rate. In contrast, a PV-like $I \to \bar { E }$ depression motif nearly eliminated vortex structure, with 0.54 mean vortices and 7.67 long tracks.

Thus, the high-rate passive-dendrite baseline is not the only route to turbulence-like field structure. Projection-specific inhibitory facilitation provides a lower-rate, intermittent route in which long-lived phase-defect dynamics persist while the mean firing rate drops by more than sevenfold relative to the baseline candidate. This supports the interpretation that STP is a biophysical regulator of the operating regime rather than a cosmetic addition to the model.

## 4.10 Rate-constrained low-rate STP episode

Because the passive-dendrite baseline has a high mean firing rate, we next made firing rate an explicit design constraint. We evaluated projection-specific STP configurations around the SST-like facilitation motif under an exploratory full-record mean-rate window of $5 { - } 3 0 \mathrm { H z } ,$ while requiring preserved vortices, long tracks, spatial scaling, and positive synaptic-delay transfer. Pilot calibration was followed by six-seed validation of the strongest candidate regimes.

For full 340-ms records, the best validated candidate stayed inside the target rate window for all six pilot-validation seeds. It produced mean firing rate 12.03 Hz, population spike-count CV 1.50, 8.87 high-amplitude vortices, 129.33 tracks with lifetime at least 8 frames, structure-function slope 0.93, and positive synaptic-delay transfer $3 . 3 4 \times 1 0 ^ { - 3 }$ . We then fixed this parameter set and ran 12 additional seeds held out from pilot calibration. All 12 remained in the 5–30 Hz target window, with mean firing rate $1 2 . 2 5 \pm 1 . 2 1$ Hz, 8.70±0.59 high-amplitude vortices, 122.58±8.56 long vortex tracks, structure-function slope $0 . 9 4 \pm 0 . 0 3$ , and synaptic-delay transfer $3 . 8 0 \times 1 0 ^ { - 3 } \pm 4 . 1 8 \times 1 0 ^ { - 4 }$ This candidate therefore addresses the biological-rate objection for finite episodes: turbulence-like field observables do not require the 115 Hz high-excitability operating point when projection-specific STP is included. Last spikes occur at $1 9 5 . \bar { 2 } \pm 1 1 . 9$ ms; active-support rate is $2 1 . 3 1 \pm 2 . { \bar { 5 } } 1$ Hz (onset-to-last-spike denominator). Corrected fields yield $3 . 4 2 \pm 0 . 9 2$ defects, $2 1 . 0 8 \pm 9 . 0 7$ tracks and $S _ { 2 } = 0 . 7 6 8 \pm 0 . 0 2 9$ . After transformation, retain [max(50 ms, stimulus offset), min(last spike+20 ms, 290 ms)]. These supersede the full-record field estimates above.

The original full-record null analysis gave the following values. As in the high-rate baseline, the result is not a simple vortex-count story. The global-rate shuffle preserved mean rate (12.25 Hz) and retained many phase defects (8.11 vortices, 122.75 long tracks), but its structure-function slope dropped from 0.94 to 0.73 and its excitatory synaptic-delay transfer collapsed from $3 . 7 6 \times 1 0 ^ { - 3 }$ to $2 . 1 \dot { 8 } \times 1 0 ^ { - 4 }$ . A fixed spatial permutation similarly retained phase defects (7.76 vortices, $1 1 0 . 9 2$ long tracks) while reducing the structure-function slope to 0.73 and eliminating synaptic-delay transfer $( - 9 . 4 2 \times 1 0 ^ { - 5 } )$ . Independent time shifts strongly reduced phase-defect dynamics (0.53 vortices, 7.75 long tracks). Thus, even in the biologically more plausible low-rate regime, smooth band-limited spike-derived fields can contain many mathematical phase singularities; the mechanistic evidence comes from their conjunction with spatial scaling and delayed propagation. Corrected-window $S _ { 2 }$ rate-null contrast is 0.0337 [0.0137,0.0565]; topology reverses against this null.

We also asked whether the $2 4 \times 2 4$ grid is too small to support the claim. Without retuning the low-rate $\operatorname { S T P }$ parameters, we repeated the simulation on $1 6 \times 1 6 , 2 4 \times 2 4 , 3 2 \times 3 2$ , and $4 0 \times 4 0$ grids for three seeds each. Vortex density remained nonzero across sizes (about 13.5–14.9 vortices per 1000 plaquettes), and the mean firing rate stayed near 5 Hz per neuron. With the original size-dependent fitting windows, the structure-function slope changed with size, from 1.23 at $1 6 \times 1 6$ to 0.72 at $4 0 \times 4 0$ , so this is a robustness check rather than a finite-size scaling proof. Common-lag (2–5 cells), activity-supported slopes are 0.757/0.753/0.758/0.779 (three seeds/size). We then computed a traditional Beggs–Plenz style spike-bin avalanche diagnostic on sparse post-stimulus electrode samples. This post-stimulus diagnostic was not strong enough to claim classical criticality: each seed produced only about 8–10 traditional avalanches after sparse sampling, the size and duration tails lacked enough samples for power-law model comparison, and the branching ratio remained below 1 (0.74 at $2 4 \times 2 4 )$ .

This negative result is specific to the short localized-stimulus record. To test classical spike avalanches more directly, we ran long weak-drive records with no localized stimulus: stimulus\_amplitude\_mv=0, homogeneous tonic drive, independent noise, 3000 ms records, and sparse 64-electrode sampling. At tonic drive 14.0 mV on $2 4 \times 2 4$ , the mean firing rate was $6 . 9 1 \pm 0 . 1 8$ Hz per neuron, the pooled electrode-resample avalanche count was $1 7 8 8 \pm 2 3 5$ per seed, the branching ratio was $0 . 9 9 7 \pm \mathrm { { \bar { 0 } } . 0 0 1 }$ , and the avalanche-size tail favored a power law over an exponential by $\Delta \mathrm { A I C } = 7 3 6 . 8 \pm 9 9 . 6$ with exponent $\alpha = 2 . 9 1 \pm 0 . 1 3$ . Repeating the same weak-drive point on $2 4 \times 2 4 , 3 2 \times 3 2$ , and $4 0 \times 4 0$ grids produced firing rates $6 . 4 8 – 6 . 8 1$ Hz and branching ratios 0.994– 1.003. Thus, the same SNN model class can show near-unit branching, not established criticality under ongoing weak drive, but this is a separate assay from the localized spike-to-field turbulence experiment and still requires stronger independent-record, bootstrap, and finite-size tests. Five-family comparison favors log-normal at five of six drives and truncated power law at one. Diagnostic plots are provided in Appendix A.

## 4.11 Topology control: transfer versus field re-entry

The topology experiment is a control on interpretation rather than a claim that graph cycles alone explain brain turbulence. The previous experiments show that delayed spatial SNNs can generate long-lived phase-field dynamics, but they do not by themselves say whether these dynamics are simply direct feedforward spike transmission. We therefore performed a topology ablation while keeping the low-rate projection-STP cell and synapse parameters fixed. We compared the original recurrent exponential-distance graph, a strictly feedforward acyclic graph with only increasing x-direction edges, a same-degree forward-biased graph, and a same-degree loop-enhanced graph with additional reciprocal edges.

The first result is that feedforward topology favors direct pairwise spike transfer. Across six seeds, the feedforward acyclic graph had the largest delay-aligned pairwise information proxy, $3 . 5 0 \times 1 0 ^ { - 3 }$ bits per presynaptic opportunity, whereas the original recurrent graph had $3 . 3 5 \times 1 0 ^ { - 4 }$ and the loop-enhanced graph had $2 . 1 2 \times 1 0 ^ { - 4 }$ . This is consistent with a feedforward chain acting as a more direct transmitter. However, the feedforward graph eliminates closed loops: its cycle-edge fraction and random-walk return probability are both zero. In contrast, recurrent and loop-enhanced graphs have cycle-edge fraction 1.0 and nonzero return probability.

The second result is that recurrent topology supports activity re-entry after controlling firing rate. We adjusted tonic drive values for each topology and seed to match the mean firing rate near 10.5 Hz. After rate matching, the feedforward acyclic graph still had no graph cycles and had density-field reentry probability $0 . 5 8 8 \pm 0 . 0 3 6$ . The original recurrent graph had re-entry probability $0 . 7 3 6 \pm 0 . 0 1 2$ and the loop-enhanced graph had $0 . 7 5 7 { \pm } 0 . 0 1 9$ . To test whether this was only an artifact of smoothing $\rho ,$ we repeated the measurement on raw spike trains and compared against a per-neuron spike-count matched random-time null. Raw spike re-entry exceeded the null in all topologies. The recurrent and loop-enhanced graphs had modestly higher excess over the null (0.138 and 0.134) than the feedforward acyclic graph (0.107). Thus, topology does not define turbulence-like dynamics by itself; it separates a feedforward transmission mode from a recurrent field-re-entry mode without establishing a computational advantage.

## 5 Discussion

The main lesson is that spike-derived turbulence-like observables are possible in a minimal spatial SNN, but only after an explicit scale transition from binary spikes to continuous density and phase fields. The strongest evidence is not any single statistic: avalanche tails and vortex counts can both be misleading, because controls can preserve power-law-favored tails or mathematical phase defects while degrading long tracks, structure-function scaling, and delayed synaptic transfer.

Table 3: Topology ablation separating direct pairwise spike transfer from recurrent activity re-entry. Pairwise bits/opportunity are from the six-seed topology ablation. Cycle and re-entry metrics are from the rate-matched and raw-spike null controls.
<table><tr><td>Topology</td><td>Bits/opportunity</td><td>Cycle edges</td><td>ρre-entry</td><td>Raw spike excess</td></tr><tr><td>Recurrent EDR</td><td> $3 . 3 5 \times 1 0 ^ { - 4 }$ </td><td>1.00</td><td>0.736</td><td>0.138</td></tr><tr><td>Feedforward acyclic</td><td> $3 . 5 0 \times 1 0 ^ { - 3 }$ </td><td>0.00</td><td>0.588</td><td>0.107</td></tr><tr><td>Forward-biased same-degree</td><td> $3 . 9 2 \times 1 0 ^ { - 4 }$ </td><td>1.00</td><td>0.652</td><td>0.118</td></tr><tr><td>Loop-enhanced same-degree</td><td> $2 . 1 2 \times 1 0 ^ { - 4 }$ </td><td>1.00</td><td>0.757</td><td>0.134</td></tr></table>

The avalanche results sharpen this separation. Short post-stimulus rasters are too sparse to support a Beggs–Plenz criticality claim, whereas long weak-drive records produce many spike-bin avalanches and branching ratios near one at biologically plausible rates. This is consistent with prior warnings about thresholding and subsampling [17, 13, 19]: localized initiation is useful for testing propagationgenerated fields, while ongoing weak drive is better suited to testing spike-bin criticality.

The biological controls identify candidate gates for the field dynamics. PV-like perisomatic inhibition suppresses detected phase-defect activity, SST-like dendritic inhibition preserves it, VIP-like disinhibition partially restores it, and projection-specific inhibitory facilitation provides a low-rate route to long-lived phase-defect tracks. The topology control adds a guardrail: feedforward graphs favor direct pairwise transfer, whereas recurrent graphs support rate-matched field re-entry, without proving enhanced information mixing.

High-rate density lacks a 20–80 Hz peak above fitted aperiodic background; Butterworth/FFT defects/frame are 6.51/4.39 with 50-ms guards. Separately, five frozen configurations sustain low-rate organization for 2.8 s across 12 unseen seeds under different, phase-free criteria.

## 6 Limitations

This is a simplified simulation study. The grid is two-dimensional and small, neurons are singlecompartment LIF units with reduced dendritic variables, and PV/SST/VIP labels are functional motifs rather than validated cell classes. The high-rate passive-dendrite candidate should not be interpreted as typical spontaneous cortex; the low-rate STP replication shows that high rate is not necessary, but it is still a finite episode rather than a biological fit. The phase diagram is a four-seed regional map, STP parameters are motif hypotheses, and avalanche evidence remains preliminary: continuous-field tails have conditional alternative-family and seed-bootstrap tests [3], while spike-bin avalanches still require bin-width, threshold, independent-record, and finite-size analyses. Finally, null models show that phase defects can survive spatial scrambling, topology ablations do not prove an empirical claim about cortical loops, and the model is not a reproduction of human fMRI or MEG measurement physics. Appendix B reports a small event-data boundary check in which the current frozen SNN does not improve task readout over raw event features.

## 7 Conclusion

We present a spike-to-field SNN framework in which delayed spatial propagation, dendritic filtering, inhibitory routing, and projection-specific STP can generate turbulence-like density and phase observables. The main result is a guarded one: long-lived phase-defect dynamics require interpretation jointly with spatial scaling, delayed synaptic transfer, firing-rate controls, and avalanche diagnostics. This provides a mechanistic testbed for brain turbulence observables, with the next step being stronger finite-size, avalanche, null-model, and empirical-recording comparisons.

## Acknowledgments and Disclosures

ZeQome provided all funding, equipment and computing resources. Zequan Liu is employed by ZeQome.

## References

[1] John M. Beggs and Dietmar Plenz. Neuronal avalanches in neocortical circuits. Journal of Neuroscience, 23(35):11167–11177, 2003. doi: 10.1523/JNEUROSCI.23-35-11167.2003.

[2] Michael Breakspear. Dynamic models of large-scale brain activity. Nature Neuroscience, 20: 340–352, 2017. doi: 10.1038/nn.4497.

[3] Aaron Clauset, Cosma Rohilla Shalizi, and M. E. J. Newman. Power-law distributions in empirical data. SIAM Review, 51(4):661–703, 2009. doi: 10.1137/070710111.

[4] Benjamin Cramer, Yannik Stradmann, Johannes Schemmel, and Friedemann Zenke. The heidelberg spiking data sets for the systematic evaluation of spiking neural networks. IEEE Transactions on Neural Networks and Learning Systems, 33(7):2744–2757, 2022. doi: 10.1109/ TNNLS.2020.3044364.

[5] Gustavo Deco and Morten L. Kringelbach. Turbulent-like dynamics in the human brain. Cell Reports, 33(10):108471, 2020. doi: 10.1016/j.celrep.2020.108471.

[6] Antonio J. Fontenele, Nivaldo A. P. de Vasconcelos, Thaís Feliciano, Leandro A. A. Aguiar, Carina Soares-Cunha, Bárbara Coimbra, Leonardo Dalla Porta, Sidarta Ribeiro, Ana João Rodrigues, Nuno Sousa, Pedro V. Carelli, and Mauro Copelli. Criticality between cortical states. Physical Review Letters, 122:208101, 2019. doi: 10.1103/PhysRevLett.122.208101.

[7] Walter J. Freeman. Vortices in brain activity: Their mechanism and significance for perception. Neural Networks, 22(5–6):491–501, 2009. doi: 10.1016/j.neunet.2009.06.050.

[8] Nir Friedman, Shinya Ito, Braden A. W. Brinkman, Masanori Shimono, R. E. Lee DeVille, Karin A. Dahmen, John M. Beggs, and Thomas C. Butler. Universal critical dynamics in high resolution neuronal avalanche data. Physical Review Letters, 108:208102, 2012. doi: 10.1103/PhysRevLett.108.208102.

[9] Anirudh Gupta, Yun Wang, and Henry Markram. Organizing principles for a diversity of GABAergic interneurons and synapses in the neocortex. Science, 287(5451):273–278, 2000. doi: 10.1126/science.287.5451.273.

[10] Guy Major, Matthew E. Larkum, and Jackie Schiller. Active properties of neocortical pyramidal neuron dendrites. Annual Review of Neuroscience, 36:1–24, 2013. doi: 10.1146/annurev-neuro-062111-150343.

[11] Henry Markram, Yun Wang, and Misha Tsodyks. Differential signaling via the same axon of neocortical pyramidal neurons. Proceedings of the National Academy of Sciences, 95(9): 5323–5328, 1998. doi: 10.1073/pnas.95.9.5323.

[12] Lyle Muller, Frédéric Chavane, John Reynolds, and Terrence J. Sejnowski. Cortical travelling waves: Mechanisms and computational principles. Nature Reviews Neuroscience, 19:255–268, 2018. doi: 10.1038/nrn.2018.20.

[13] Viola Priesemann, Matthias H. J. Munk, and Michael Wibral. Subsampling effects in neuronal avalanche distributions recorded in vivo. BMC Neuroscience, 10:40, 2009. doi: 10.1186/ 1471-2202-10-40.

[14] James A. Roberts, Leonardo L. Gollo, Romesh G. Abeysuriya, Gloria Roberts, Philip B. Mitchell, Mark W. Woolrich, and Michael Breakspear. Metastable brain waves. Nature Communications, 10:1056, 2019. doi: 10.1038/s41467-019-08999-0.

[15] Woodrow L. Shew and Dietmar Plenz. The functional benefits of criticality in the cortex. The Neuroscientist, 19(1):88–100, 2013. doi: 10.1177/1073858412445487.

[16] Nelson Spruston. Pyramidal neurons: dendritic structure and synaptic integration. Nature Reviews Neuroscience, 9:206–221, 2008. doi: 10.1038/nrn2286.

[17] Jonathan Touboul and Alain Destexhe. Can power-law scaling and neuronal avalanches arise from stochastic dynamics? PLOS ONE, 5(2):e8982, 2010. doi: 10.1371/journal.pone.0008982.

[18] Hugh R. Wilson and Jack D. Cowan. Excitatory and inhibitory interactions in localized populations of model neurons. Biophysical Journal, 12(1):1–24, 1972. doi: 10.1016/S0006-3495(72) 86068-5.

[19] Jens Wilting and Viola Priesemann. Inferring collective dynamical states from widely unobserved systems. Nature Communications, 9:2325, 2018. doi: 10.1038/s41467-018-04725-4.

## A Additional diagnostic figures

This section collects supporting visual diagnostics that are useful for auditability but are not required to follow the main argument. Panels retain original windows/operators; corrected low-rate estimates supersede full-record values (Section 4.10).

![](images/fb13fbd33bbd2f28d2c1dec45085ab077ac2dfa3641933a58850b5c491fb01de.jpg)  
Figure 3: Multi-seed reproducibility of the candidate delayed-propagation passive-dendrite regime.

![](images/867998a651e28af3dd472826e5bcef4dd8311a7953c645c6bd4a12909cc62ea8.jpg)

![](images/f5dde653279bc14e413a773aa3eb333c74f0554ce4856f04df0f29716a2de765.jpg)

![](images/4fefa17e60f17ab97120f6cb2d68086c1ad651a4d857911fb5a7957ee0f203fa.jpg)

![](images/820815b2d827f37c74156d18e85202bc995f891b50ce02bb51b4ecda96022431.jpg)  
Figure 4: Baseline versus mechanism controls.

![](images/7f51f98769c82502dd4b151ae31c7075f85c6f7128c66d7389cbe2e7d2a5e77c.jpg)

![](images/28c124f79a9ab9e371b71d7af9edd248ad0edd15c2dadebb8b729987c80183f8.jpg)

![](images/fc3c525bb7490a04f969447e2eeb606652265783c1fa5492efa5c4a020c1d46f.jpg)

![](images/9d07375e338f169885dd63974d4592877cfac7a18ba8d4e49ad6aa6f3ccec413.jpg)

![](images/e2c09250e5d4355bc6189a4e413674c287289aad51d2695e1a8d1875eecf6399.jpg)

![](images/c322ae318378f6439cf109ac8526ca02981c2dcc1463afbccd0f7c0ba2918976.jpg)  
Figure 5: Rate-matched nulls constructed from the same high-rate turbulence-like SNN spike tensors. These are not separate non-turbulent SNN models: the original spike data are held fixed and then shuffled to preserve selected firing statistics while destroying selected spatiotemporal structure. The global-rate shuffle preserves the total spike count in each frame but randomizes spike locations; independent time shifts preserve each neuron’s spike count and spatial identity but disrupt synchrony; the fixed spatial permutation preserves every spike train and temporal synchrony but scrambles the physical map. The comparison shows why vortex count alone is insufficient: some shuffled fields can still contain mathematical phase defects, but spatial structure and delayed transfer distinguish complementary nulls.

![](images/4fc8af0424d57867e17a450d7f7b96724dd938a02519310e4c56cccdaa7e1cd4.jpg)  
Figure 6: PV/SST/VIP microcircuit phase diagram.

![](images/98b450287b2eb266d4517a2c09ea613c8dbb81ac68b83cd31bcee3365b6cd76e.jpg)  
Figure 7: Projection-specific STP comparison.

![](images/8d5d2dc22efad029b1024257d689683f8b7c0d39415dbd1d1816a8be76217bd8.jpg)

![](images/7e225dc5c3cc0d42a50639de17637922da7f2961d69271c1c6446e8281b54638.jpg)

![](images/b09be982f4357024474bbf5265558b9fc9e23638d8c317547fe4f0e32ee77907.jpg)

![](images/e67eb6a5b57527fc13bdfc566dd72b0d09748e75439bada1626b42420cbaf102.jpg)

![](images/bf56f4eb673f776343ecde31e9a54e2e529f035b16b74aecd14f500c80899242.jpg)

![](images/f1ce920f1e0d9179d223ddc3ee2a0e15f75437f72a38c2b9a47060b67911de9e.jpg)

(a) Twelve-seed reproducibility of the best low-rate projection-STP candidate.  
![](images/3af7025be9c26fb3c8b6214bb1aeb082249ef4df30d16226163cdb0ecd83a49f.jpg)

![](images/b533564b8ddc5a8a43ef271c8156715f87b094b540fdafae4a17217816fcf2c6.jpg)

![](images/bb191e52d019e34191c3ae6a4fa44cd106dfd075f9ab0c96f4b97889596adcf2.jpg)

![](images/f5dde2af0041a931c7bb832a2099cad4d14c3240151881a118e029e280867dd7.jpg)

![](images/5fc38e26197c66af0aae30140c9f146502aae2dda95db41a973412ad08617766.jpg)  
(b) Rate-matched nulls for the same low-rate spike tensors.

![](images/2610d84f114a70321698e659e2fb0a31467e388d95be67394f03f7702e883037.jpg)  
Figure 8: Historical full-record episode summaries and null controls.

![](images/58a0db186ffeb98f43d23177328c57117284f901e39d6dd371a220bb54ccc81c.jpg)

![](images/82f84d70c9f0ff5c362408a3854bd3664d4b1207670d3beadc3766c19d17724e.jpg)

![](images/1b2304c885e18a3c53fcb7546e83b761b4a8e4c8bb311671d9890b4891886835.jpg)

![](images/552f10a8bc31a5f4d7f5e8639dae9f0c5481ad0bd2850eb68570fea8e470d23c.jpg)

![](images/4dd21ed82084055c9482b15c89f13b53fe21e88fcd71a18cdf18995fe306113b.jpg)

![](images/0057f8a461d8c4d148afb54b2d3503113ec3f9bca9531c0838ed2f9ccafdbb12.jpg)  
Figure 9: Continuous-field avalanche analysis. High-density connected components in $\rho ( x , y , t )$ yield candidate avalanche sizes. The pooled tail is better fit by a power law than an exponential under the tested AIC comparison, but this statistic is interpreted jointly with vortex and structure-function metrics.

![](images/6281dd9bc59a5c9cc72ef5433948036f3c91343efb73e3940d5365d201111898.jpg)

![](images/63754fff79a74bbf41cc9f314ec3b16fa3f8dcd58cf7021c595c86f65b294b72.jpg)

![](images/425b880ceccf88b33b4c73768a6de69fe4ef9d736aef7a125b86225487ed35fe.jpg)

![](images/6831c923d58bc18dea6d781ca3e3bcea4eff46f56d86f36b693fb805cb594941.jpg)

![](images/350ef4d55970b4aee22ba3ca80eac1990abfea1ac73c21225a43502699f1ab9c.jpg)

![](images/b1a6100b8f96d1838dfd877d7330e603bb0e1004d8e4247f0dfd21bd169aeb29.jpg)  
Figure 10: Threshold sensitivity and mechanistic controls. Avalanche power-law preference is robust across tested thresholds, but the phase-vortex and long-track metrics expose stronger dependence on propagation delay and passive dendritic filtering.

![](images/20e1c6af0090ba57ed7166bcc794cb466f1e5d3a3b70fe8238497feeba1b3b31.jpg)

![](images/b9e2a7e4814400ca6f07419a917c6b27a7283f3b6686e59015cfefe4f4d8b785.jpg)

![](images/0c2f011a26cbeca7577d1c212994489d27573e1820e84a2d474b6ec251653658.jpg)

![](images/2026342a70ffb531f1b7a6516537255f029a31bcaf514d84f2c5bc089d9d82c9.jpg)

![](images/6cac98be89f5cf03a00af49f7a52746d4ae4ce433ab81d74f588d401ec8f616c.jpg)

![](images/5c8ca595c16170cd084b7c9fb4ba6d46a97bbf6ebccf9b3823972a36bc3b2259.jpg)  
Figure 11: Explicit inhibitory microcircuit comparison. PV-only perisomatic inhibition suppresses vortex dynamics, SST-only dendritic inhibition preserves long tracks, and VIP-like disinhibition partially restores phase structure in the PV+SST setting.

![](images/9b6025884097c94a496b7cad6daec9298e5531af8f1926deaaf6041e5fe4ccf3.jpg)

![](images/660d4a5a32a30b21eba9f18aaaa4e8170ee2f6cf9164c54d3911002da3e58258.jpg)

![](images/a7cee6ac2544e0a036555fb99b1869ad3be8562bb18da10465b7c46962251756.jpg)

![](images/d262e05b0915ceb11ba005c67c37d9522db1b20b33110f3f500454c55679442c.jpg)

![](images/e1914e5076f61f3a014396d46f4f96ed8b3c2cfa2ce5ef90329ceda3aa0134b9.jpg)

![](images/23432bbb1233dd06e2bd14a8cf79902dce736f56cd0af7e42a31806b113a32f8.jpg)  
Figure 12: Rate-constrained projection-STP candidates for biologically more plausible turbulence-like regimes. The shaded firing-rate window is 5–30 Hz. The validated candidates preserve high-amplitude vortices, long tracks, spatial scaling, and positive synaptic-delay transfer at much lower full-record mean firing rates than the passive-dendrite baseline.

![](images/bf4aedefbd88edff3eab596ff9a90ee637ad926895149de0d25f51457c98de6d.jpg)  
(b) Raw-spike re-entry compared with per-neuron spike-count matched random-time nulls.

Figure 13: Topology control diagnostics. Feedforward topology improves direct delay-aligned pairwise spike transfer but removes graph cycles. Recurrent and loop-enhanced topologies preserve closed-loop structure and retain higher rate-matched activity re-entry. Raw spike re-entry also exceeds spike-count matched nulls, but the raw-spike topology effect is moderate.

![](images/0c0131d12268d277a779c44de2bfd893852f7cb989ef130b25fa27cc90b50d1f.jpg)

![](images/b9321293cff967e515ca2355e808786eaea9457d3d2426173797d4d977e0f119.jpg)

![](images/9903c61de2a7c87f2952089935ba4b82b450835e94d24ec8a4a75c341f3e4728.jpg)

![](images/5da94624ef0ba7b85bf2ebc224a645b57d015fcbf15de400ee9074de4d44026f.jpg)

![](images/df531406a7794a155bc0dae2dd400609a5b3d579c42957f39fe4f41b3284102e.jpg)

![](images/8f2694298ba8c4a0f5a9e575007adb666c88fe675147a31d4f105233d5545e80.jpg)  
Figure 14: Traditional spike-bin avalanche and larger-grid diagnostics. The low-rate projection-STP candidate was run without retuning on $1 6 \times 1 6 , 2 4 \times 2 4 , 3 2 \times 3 2$ , and $4 0 \times 4 0$ grids. Vortex density and field structure persist beyond $2 4 \times 2 4$ , but sparse-raster traditional avalanche samples are too few for reliable size/duration power-law comparisons; branching ratios remain below 1.

![](images/1482d3ea96258108f30ec8c7ccf4f85ef33fd11c3117e37b090050ccc8e4ddbc.jpg)

![](images/c1e1eabc500d7596be4a6b672f9a09d4e0b428397a321d2829e41c6c97855054.jpg)

![](images/941664514ea4f7cdaa70331bd865e718a9ee1d9ab12609bcce032ad99312fb73.jpg)

![](images/9dc12881f02c5b7534e2745ce52b6dd362f3fc201b24e7ea21cdf21a2ace0b36.jpg)

![](images/3594e84cf7bd29f6f513aa61949349e234008c1f38e0df036a77fef89241b616.jpg)

![](images/4e5537a2ee427b24670942bf41eea501d09ba3ef941f096c48da5711d155ebf7.jpg)  
Figure 15: Traditional spike-bin avalanche diagnostics under long weak-drive spontaneous-like recording. The localized stimulus is disabled and homogeneous drive/noise maintain low-rate ongoing activity. At moderate drive, sparse-electrode spike-bin avalanches are numerous, branching ratios are near one, power-law tails are favored over exponential tails by $\mathrm { A I C } ,$ and the effect persists from $2 4 \times 2 4 \mathrm { t o } 4 0 \times 4 0$ grids.

## B Real event-data readout sanity check

The main paper studies spike-level mechanisms for turbulence-like field dynamics, not task performance. We nevertheless ran a small real event-data sanity check to test whether the current frozen SNN regimes automatically improve event-stream readout. We used the Spiking Heidelberg Digits (SHD) dataset [4], selecting five classes with 40 training and 20 test examples. SHD is event-based audio rather than two-dimensional DVS vision; its 700 input channels were mapped onto a 28 × 25 region of the SNN input grid. This experiment is therefore a diagnostic boundary condition rather than a main benchmark.

The result is negative. With rich temporal event-bin features, a direct linear readout on raw SHD event frames achieved 0.95 test accuracy (training-CV-selected readout) on this small subset. The frozen turbulence-like low-rate STP SNN reached 0.45 linear-ridge test accuracy using temporal state bins, the ordinary balanced E/I LIF control reached 0.45. Thus, the present turbulence-like dynamics do not automatically improve task readout; they may mix or discard discriminative event structure unless input projection and readout mechanisms are explicitly trained or otherwise matched to the task.

Table 4: Small SHD event-data readout sanity check. These results are reported as an appendix boundary condition, not as a main contribution.
<table><tr><td>Representation / decoder</td><td>Features</td><td>CV accuracy</td><td>Test accuracy</td></tr><tr><td>Raw / linear ridge</td><td>Event temporal bins</td><td>0.625</td><td>0.95</td></tr><tr><td>Projection STP / linear ridge</td><td>State temporal bins</td><td>0.450</td><td>0.45</td></tr><tr><td>Projection STP / RBF ridge</td><td>State temporal bins</td><td>0.425</td><td>0.50</td></tr><tr><td>Balanced E/I / linear ridge</td><td>State temporal bins</td><td>0.325</td><td>0.45</td></tr></table>

![](images/e08e3b0ba6da84c1e96cfe1303bfebc1c30a342d3e2e33e4abcdcd08d934a393.jpg)

(a) Information-retention diagnostics comparing raw event features and frozen SNN states.  
![](images/9a3d255221dd009f75f208011c9a526b4cfe91aadd799e4deeb6b93aad215cd0.jpg)

![](images/ab6214d8e2680750aa0d574c20169d2217c55b4760bacb9fc182ff3ce9faf458.jpg)

![](images/48e3b3f46b52b5ae97617a8e654b5ca0a81d66479c307ce019eea95ee3eced9d.jpg)

![](images/c451b0ec7f1b74b9fca0a667255b78fb7f764af1fedd600058b8af1f7e8d76e7.jpg)  
(b) Readout bottleneck experiment using single-time, multi-time state, spike, and combined readout features.

Figure 16: Real event-data sanity check on a small SHD subset. Raw event features remain more linearly readable than the current frozen turbulence-like SNN states, reinforcing that turbulence-like dynamics should not be interpreted as automatic task-performance gains.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes].

Justification: The abstract and introduction state that this is a simplified SNN simulation framework and avoid claiming a direct model of biological human brain turbulence.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes].

Justification: A dedicated Limitations section discusses the reduced grid, simplified neuron and dendrite models, motif-level STP assumptions, and conditional avalanche model comparisons.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A].

Justification: The paper proposes an empirical simulation pipeline and does not claim formal theorems.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes].

Justification: The model parameters, analysis steps, and random-seed protocol are described in the paper and appendix.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [No].

Justification: A review artifact exists. The code and generated data exist in the local repository, but a complete permanent public release remains unverified.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes].

Justification: Simulations have no optimizer; SHD uses training cross-validation and heldout testing; simulation parameters and candidate-regime protocol are described in the model and experiment sections.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes].

Justification: The paper reports 12-seed means, SD and pointwise paired seed-bootstrap intervals.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [No].

Justification: The current draft does not yet report wall-clock time, CPU model, or memory usage.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes].

Justification: The work is a synthetic simulation study and uses no private or scraped data;   
SHD reuses public speech.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No].

Justification: Broader impacts have not yet been expanded here; the current scope is foundational computational neuroscience.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A].

Justification: The paper does not release a high-risk model or scraped dataset.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [No].

Justification: The draft cites the scientific literature, but repository and dependency license information has not yet been formalized.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes].

Justification: The generated simulation scripts, CSV outputs, figures, and HTML reports are documented in the local README and appendix.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A].

Justification: The work does not involve new crowdsourcing or participant recruitment.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A].

Justification: The work does not recruit human subjects.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A].

Justification: LLMs are not part of the simulated network; they were used for code and manuscript assistance, subject to author verification.