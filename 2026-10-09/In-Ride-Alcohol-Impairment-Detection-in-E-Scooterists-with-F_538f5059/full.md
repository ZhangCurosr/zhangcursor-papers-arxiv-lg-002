# In-Ride Alcohol-Impairment Detection in E-Scooterists with False-Alarm Control

Marco Capuccini<sup>1,2</sup> and Rahul Rajendra Pai<sup>1,3</sup>

Abstract—Shared e-scooter services have become a widely adopted urban transport mode. While most users ride responsibly, alcohol intoxication stands out among the factors contributing to severe crashes. Nonetheless, countermeasures remain limited to single-point reaction tests and night bans that suspend the service altogether. This paper proposes a new approach in which onboard sensors evaluate the rider as the trip unfolds, raising an alarm as soon as enough evidence of impairment has accumulated. Specifically, we introduce a detector that operates on inertial and throttle measurements, with a provable bound on the rate of false alarms. Experiments on sensor data from 141 rides, in which 25 participants rode while sober and at two target blood alcohol concentration levels, confirm that the bound holds, whereas baselines and ablations either exceed it or lose detection performance, and in some cases delay the alarm. At a bound of 0.023, the detector identifies 91% of the rides performed at the higher concentration and 50% of those at the lower one, with median detection times of 25 and 27 seconds, respectively. We further show that an embedded implementation meets the realtime requirement, making mitigation actions feasible onboard— without requiring data to leave the vehicle. Overall, this work lays the ground for interventions that reach impaired riders as soon as possible, sparing the sober ones the burden of a preride test or the suspension of the service at night, while letting operators budget false alarms against user experience.

Index Terms—Alcohol impairment detection, micromobility safety, anytime-valid inference, conformal prediction.

## I. INTRODUCTION

HARED micromobility, in particular rental e-scooters, has S transformed cities at a pace unmatched by other forms of transportation. The scale of this shift is evident in Europe, with 312M trips in 2025 alone [1]. As ridership continues to grow, safety has become a pressing challenge, underscored by crash reports [2]. Among the contributing factors, alcohol intoxication stands out across studies. In Sweden, 44% of fatal e-scooter crashes recorded between 2016 and 2024 involved intoxicated riders, against 13% for bicyclists and 27% for ecyclists [3]. Non-fatal injury data follow the same pattern, with intoxication reported in 12% of injured riders in Washington, D.C. during 2019 [4], 32% in a German emergency department during 2022 [5], and 47% in Helsinki during 2021, before nighttime restrictions [6]. Despite this evidence, preventive interventions remain limited: the in-app reaction tests currently adopted by operators are easily circumvented and screen the rider only at the start of the trip [7], while nighttime bans suspend the service for everyone, sober riders included [6].

In this paper, we propose an approach in which onboard sensors monitor rider kinematics continuously throughout the ride. This approach removes the need for a pre-ride reaction test, keeps the service available at night, and enables in-ride intervention which occurs only when impairment is detected. For such a system to be viable, we also provide a falsealarm-rate (FAR) control mechanism with an anytime-valid statistical guarantee, allowing operators to bound the rate of incorrect detections and thus limit unnecessary alarms for sober riders. Anytime validity means that the bound holds at every step of the ride rather than at a single pre-specified decision point; the detector can therefore raise an alarm as soon as sufficient evidence accumulates, without inflating the FAR. This extends our prior work, which established alcohol impairment as detectable from the permutation entropy (PE) of sensor signals, though only from complete rides in postprocessing and without FAR control [8].

The proposed detector evaluates the ride by extracting the PE of triaxial accelerometer, gyroscope, and throttle lever position measurements, computed over a short lookback window. The resulting seven-dimensional vector is then fed to a machine learning (ML) model that yields an impairment score for the current window. As the ride progresses, evidence is accumulated through a running statistic of the scores, and an alarm is raised as soon as it crosses a threshold calibrated on sober rides. The threshold is obtained through a procedure introduced in this work, which builds on inductive conformal prediction (ICP) [9], extending its FAR control to every step of the ride.

We evaluate the detector on the dataset we publicly released with our prior work, comprising signals from 141 instrumented e-scooter rides, in which 25 participants rode while sober and at two target blood alcohol concentration (BAC) levels [8]. The detector keeps the FAR within the bound, unlike baselines that do not account for the temporal dependence of sensor measurements. At a FAR bound of 0.023, it detects 91% of the rides at the higher BAC level within a median of 25 s of riding, and 50% of the rides at the lower level within a median of 27 s, a rate no in-ride baseline matches under FAR control. We also run an embedded implementation of the detector on an STM Nucleo board [10], where one detection cycle takes 262 ms, showing that in-ride monitoring is feasible onboard.

In summary, the key contributions of this work are as follows:

• We introduce an alcohol impairment detector that monitors e-scooter riding continuously and raises an alarm as soon as sufficient evidence accumulates, reducing the need for pre-ride screening and nighttime bans.

• We propose a threshold calibration procedure that statistically bounds the FAR at any step of the ride, and we prove its validity.

• Using instrumented e-scooter signals, we empirically show that (1) the detector keeps the FAR within the bound where baselines that disregard temporal dependence do not, (2) it outperforms in-ride baselines under FAR control with strong detection performance at the higher BAC level and moderate performance at the lower, and (3) it runs in real time on an embedded device.

• We publicly release the full experimental code.

## II. RELATED WORK

Alcohol impairment has been studied most extensively in the automotive domain. A systematic review of 26 studies applying ML to in-vehicle sensor data, most often tracking vehicle dynamics and control inputs, reports consistently strong discriminative performance [11]. In micromobility, impaired riding has been examined for both cyclists and e-scooterists, with the evidence pointing to riding fitness degrading measurably with BAC [12], [13], while detection from onboard sensors has so far been demonstrated only in our prior work [8]. To our knowledge, threshold calibration procedures and anytime-valid FAR control, enabling in-trip detection, remain unexplored across both domains.

ICP provides a computationally efficient mechanism to provably bound the FAR in ML systems, without prior distributional knowledge [9]. This property is particularly compelling, as in real applications the underlying distribution is seldom known and guarantees derived from approximations can be unreliable. ICP validity instead holds under exchangeability alone [14], weaker than the i.i.d. sampling which ML methods commonly assume. Exchangeability martingales, first developed to test this condition [15], provide valid FAR control and have also been applied to anomaly detection in flight behavior [16] and, in our prior work, to e-scooter GNSS traces [17]. However, when readings are temporally dependent, as in sensor streams, exchangeability breaks even in the absence of anomalies and the FAR guarantee no longer holds. The flight behavior study neither discusses this nor reports the empirical FAR, whereas our prior work addresses it with a shuffling buffer, which restores exchangeability at the cost of a considerable detection delay. Conformal methods that instead relax the condition also fall short for in-ride detection, as weighted quantiles bound the loss of coverage rather than removing it [14], and block permutation is only approximately valid under dependence [18], both targeting coverage at a single test point rather than anytime validity.

## III. DETECTION METHOD

The detector operates on rides consisting of sensor readings. Formally, a ride $R = \{ \mathbf { x } ( t _ { k } ) \} _ { k = 1 } ^ { T }$ is a discrete-time signal of

Algorithm 1: In-Ride Inference   
Input: Window stream $\overline { { W _ { 1 } , W _ { 2 } , \ldots , W _ { J } ; } }$ scoring   
function $f ;$ running evidence function $^ { g ; }$   
subject center $\bar { \mathbf { z } } ;$ threshold $\tau _ { \alpha }$   
1 for $j \in \{ 1 , 2 , \dots , J \}$ do   
2 $\mathbf { c } _ { j } \gets \big ( H ( W _ { j } ^ { ( 1 ) } ) , \dots , H ( W _ { j } ^ { ( d ) } ) \big ) - \bar { \mathbf { z } }$   
3 $s _ { j }  f ( \mathbf { c } _ { j } )$   
4 $E _ { j }  g ( s _ { 1 } , . . . , s _ { j } )$   
5 if $E _ { j } > \tau _ { \alpha }$ then   
6 Raise an alarm $\blacktriangle$ and exit   
7 end   
8 end

length $T$ sampled at a rate $\begin{array} { r } { F _ { s } = 1 0 0 \mathrm { H z } , } \end{array}$ , with

$$
\mathbf { x } ( t _ { k } ) \in \mathbb { R } ^ { d } , \quad k = 1 , \dots , T ,
$$

where $d = 7$ is the number of sensor channels, i.e., triaxial acceleration, triaxial angular velocity, and throttle position. The inertial readings pass through a causal 64th-order finite impulse response low-pass filter, using the Hamming window at a 10 Hz cutoff [19], which suppresses chassis vibration while preserving voluntary motor input [20]. As the ride is monitored while it unfolds, the detector can in principle access all the readings up to the current step $t _ { k } .$ However, for resource efficiency, it retains only a short lookback window of $w = 1 0 0 0$ samples, updated every $\Delta = 1 0 0$ samples, i.e., once per second. It follows that the resulting window stream is given by

$$
W _ { j } = \left\{ { \bf x } ( t _ { k } ) \right\} _ { k = k _ { j } - w + 1 } ^ { k _ { j } } , \quad k _ { j } = w + ( j - 1 ) \Delta ,
$$

for $j = 1 , \dots , J$ and $J = \lfloor ( T - w ) / \Delta \rfloor + 1 .$

Features are extracted from each window as the normalized PE of its channels [21], with embedding dimension 5 and a delay matched to the Nyquist rate of the filtered signal. Letting H denote PE and $W _ { j } ^ { ( 1 ) } , \ldots , W _ { j } ^ { ( d ) }$ the channels of $W _ { j }$ , the stream of feature vectors is defined as

$$
\mathbf z _ { j } = \big ( H ( W _ { j } ^ { ( 1 ) } ) , \dots , H ( W _ { j } ^ { ( d ) } ) \big ) \in \mathbb R ^ { d } , \quad j = 1 , \dots , J .
$$

To account for subject variability, as in our prior work [8], each $\mathbf { z } _ { j }$ is then centered by subtracting a channel-wise mean, giving

$$
\mathbf { c } _ { j } = \mathbf { z } _ { j } - \bar { \mathbf { z } } , \quad \bar { \mathbf { z } } = \big ( \bar { z } ^ { ( 1 ) } , \dots , \bar { z } ^ { ( d ) } \big ) ,
$$

where $\bar { z } ^ { ( 1 ) } , \dots , \bar { z } ^ { ( d ) }$ are the channel means of the feature vectors over a set of centering rides from the same rider. Note that this operation requires no knowledge of their impairment level, as it only removes a rider-specific offset.

The remainder of this section describes inference and threshold calibration, then proves the anytime-valid FAR guarantee.

## A. Inference

Algorithm 1 shows how inference is carried out on the stream. The procedure requires a scoring function $f ,$ which rates each window such that higher values indicate stronger evidence of impairment. For this, we fit a logistic regression on the centered feature vectors of a set of training rides and take its impairment probability as the score, a modeling choice that in our prior work matched more complex alternatives at a lower computational cost [8]. In contrast with the centering rides, training rides may come from other riders, though their impairment level must be known. Inference further requires a running evidence function $^ { g , }$ chosen as the shrunk average

$$
g ( s _ { 1 } , \ldots , s _ { j } ) = { \frac { 1 } { j + \kappa } } \Big ( \sum _ { i = 1 } ^ { j } s _ { i } + \kappa s _ { 0 } \Big ) ,
$$

where $s _ { 1 } , \ldots , s _ { j }$ are the scores of the windows observed so far and the shrinkage $\kappa = 1 0$ acts as a number of pseudo-windows pulling the average toward the sober baseline s , i.e., the mean score over sober rides used for fitting, thereby damping smallsample fluctuations early in the ride. The remaining inputs are the subject center z¯ and the threshold $\tau _ { \alpha } ,$ which results from the calibration in Section III-B for a FAR bound α.

The procedure runs one cycle per window, as these become available, with J only known when the ride ends. First, it extracts and centers the features (line 2), then scores them (line 3) and folds the score into the running evidence $E _ { j }$ (line 4). If $E _ { j }$ exceeds $\tau _ { \alpha }$ (line 5), an alarm is raised and the procedure exits (line 6), otherwise it proceeds to the next cycle. As the shrunk average can be computed from a running sum, resource utilization does not grow with $j .$

## B. Threshold Calibration

Let $R _ { 1 } , \ldots , R _ { K }$ be sober calibration rides sampled at random, disjoint from the training and centering rides. Each $R _ { i } ,$ $i = 1 , \ldots , K ,$ , is processed as in Algorithm 1, without testing for impairment (lines 5–7 disabled), and is summarized by the peak of its running evidence

$$
S _ { i } = \operatorname* { m a x } _ { j = 1 , \ldots , J _ { i } } g \big ( f ( \mathbf { c } _ { 1 } ) , \ldots , f ( \mathbf { c } _ { j } ) \big ) ,
$$

where $\mathbf { c } _ { 1 } , \ldots , \mathbf { c } _ { J _ { i } }$ are the centered feature vectors of $R _ { i }$ , with $J _ { i }$ denoting its total number of windows. Writing the peaks in increasing order as $S _ { ( 1 ) } \leq \dots \leq S _ { ( K ) }$ , the threshold is set to

$$
\tau _ { \alpha } = S _ { ( \lceil ( K + 1 ) ( 1 - \alpha ) \rceil ) } ,
$$

namely the empirical (1−α) quantile of the calibration peaks, where the count includes the ride yet to be observed and the rank is rounded up. In the next section we prove that testing against $\tau _ { \alpha }$ , as in lines 5–7 of Algorithm 1, keeps the FAR within α at every step of the ride.

## C. Anytime-Valid False-Alarm Guarantee

The argument in this section borrows the validity proof of ICP [9], reproduced in brief to keep this work self-contained. Our contribution is the reduction of a ride to the peak of its running evidence, which lifts the bound from a single test point to every step of the ride.

Let $S = \mathrm { m a x } _ { j } E _ { j } , j = 1 , \ldots , J $ , denote the peak of the ride under test, exchangeable with the calibration rides and therefore sober, i.e., any $E _ { j }$ crossing the threshold yields a false alarm. $\mathbf { A s } \ S \geq E _ { j }$ at every step, such a crossing occurs if and only if $S > \tau _ { \alpha } .$ , hence

$$
\mathbb { P } \big ( \exists j : E _ { j } > \tau _ { \alpha } \big ) = \mathbb { P } \big ( S > \tau _ { \alpha } \big ) .
$$

Since $f , g ,$ , and the centering are fixed before the calibration rides are drawn, the peak is a deterministic function of a ride, so $S _ { 1 } , \ldots , S _ { K } , S$ inherit the exchangeability of the rides they summarize, ensured by random sampling for $R _ { 1 } , \ldots , R _ { K }$ and assumed for the ride under test. Ties among the peaks almost surely do not occur for a continuous score, and would in any case only make the test more conservative [22]. Setting them aside, $S$ is therefore equally likely to occupy any of the $K + 1$ ranks, giving

$$
\mathbb { P } \big ( S > \tau _ { \alpha } \big ) = \frac { K + 1 - \lceil ( K + 1 ) ( 1 - \alpha ) \rceil } { K + 1 } \le \alpha .
$$

The anytime-valid bound follows directly:

$$
\mathbb { P } \big ( \exists j : E _ { j } > \tau _ { \alpha } \big ) = \mathbb { P } \big ( S > \tau _ { \alpha } \big ) \leq \alpha .\tag{1}
$$

## IV. DETECTION PERFORMANCE

In the previous section we have shown the FAR to be bounded by construction, without saying anything about how often impairment is detected, nor how quickly, both of which depend on the scoring function f and on the running evidence function $g .$ This section confirms the bound empirically and evaluates detection rate and time, in either case against baselines with and without FAR control guarantees. The dataset, the metrics, the experimental setup, and the baselines are described next, followed by the results.

## A. Dataset

The evaluation relies on the dataset released with our prior work [8], gathered in a controlled experiment approved by the Swedish Ethical Review Authority (ref. 2025-08327-02). Thirty-three participants rode a commercial e-scooter limited to 7 km/h on an indoor track combining a straight line, a slalom with narrowing cone spacing, a braking zone, and a figure-eight. Before data collection, they practiced until they felt comfortable with the vehicle and the layout. Each then covered the track twice while sober, followed by two trials at each of two target BAC levels in increasing order, 0.05% and 0.08%, confirmed with a law-enforcement-grade breathalyzer before every trial. The vehicle logged the channels listed in Section III at 100 Hz. Intermittent memory exhaustion of the onboard logger led to the exclusion of eight participants and the loss of nine trials across eight others, leaving 25 (19 male), aged 26.5±4.6 years. The resulting dataset contains 141 rides: 45 sober, 50 at the lower level, and 46 at the higher one; at least one trial per participant is retained in each condition. Missing samples remain within these recordings, fewer than 10% in any of them, which we fill by linear interpolation. We refer the reader to [8] for the full protocol.

## B. Evaluation Metrics

We measure performance in terms of FAR, detection rate, and detection time, at different values of the bound α. Given a set of test rides $\mathcal { R } = \{ R _ { 1 } , . . . , R _ { N } \}$ , the first two are computed as the alarm rate

$$
{ \frac { \left| \left\{ R \in { \mathcal { R } } : E _ { j } > \tau _ { \alpha } { \mathrm { ~ f o r ~ s o m e ~ } } j \right\} \right| } { N } } ,
$$

where $E _ { j } , \ j \ = \ 1 , \ldots , J ,$ is the running evidence of $R .$ Specifically, R contains sober rides for the FAR and impaired ones for the detection rate. Detection time is instead computed, on each impaired ride raising an alarm, as

$$
\frac { w + ( j ^ { * } - 1 ) \Delta } { F _ { s } } , \quad j ^ { * } = \operatorname* { m i n } \{ j : E _ { j } > \tau _ { \alpha } \} ,
$$

which amounts to the ride time elapsed when the alarming window $j ^ { * }$ closes.

## C. Experimental Setup

To make the most of the available data, the evaluation follows a leave-one-participant-out protocol. At every fold, a distinct scoring function is fit on the rides of the participants left in, whose sober rides also provide the baseline $s _ { 0 } .$ , whereas centering is carried out within each participant, one ride at a time against the remaining ones. For the participant left out, a distinct threshold is calibrated on the sober rides left in, each contributing the peak it reaches under the scoring function of the fold that holds it out, so that no calibration peak is in-sample. Each ride is therefore scored once, out of fold, giving one running evidence path per ride and one threshold per participant, over which the metrics above are computed. Furthermore, for both rates we derive a 95% confidence interval (CI) from 1000 bootstrap resamples of the out-offold results, drawn such that the rides of each participant stay together, whereas detection time is summarized by its median and interquartile range (IQR). Note that no leakage occurs, as no ride is ever scored by a function fit on data from the participant who rode it, whose other rides enter only through the label-free centering.

In this protocol, the scoring function and the threshold vary from one fold to the next, hence each threshold is computed from peaks produced by distinct models. Each scoring function nonetheless comes from the same deterministic procedure, applied to all participants but the one left out. It follows that the peaks do not depend on the order in which the participants are taken, thereby inheriting their exchangeability. Rides are however not drawn at random, as those of the participant under test are left out. We argue that centering, by removing rider-specific offsets, makes this distinction negligible. The argument in Section III-C thus applies within each fold, and the bound carries over to the pooled metrics, as Section IV-E confirms. The same reasoning covers the $\mathrm { C I s , }$ as resampling draws participants at random.

Finally, we point out that the smallest attainable $\alpha$ follows from the size of the calibration set. Holding out a participant leaves at least $K \ = \ 4 3$ sober rides, placing the tightest threshold at the $1 / ( K + 1 ) \approx 0 . 0 2 3$ level. Hence, we show results for α set to 0.023, 0.05, 0.1, and 0.2 in Section IV-E, higher values being of no practical interest.

## D. Baselines

We compare the detector against five baselines under the protocol introduced in the previous section. First, we ablate inride detection and the calibrated running evidence, to quantify what each contributes to the evaluation metrics. Then, we turn to exchangeability martingales, to assess whether they control FAR in practice under time dependence, and how they compare once recalibrated. Finally, we ablate and sweep the shrinkage parameter. We leave the window size at $w = 1 0 0 0$ samples, which strikes a good balance between PE estimation accuracy and the minimum detection delay. Each baseline is detailed next.

1) Batch Classifier: This baseline ablates in-ride detection, thus reproducing our prior work [8]. In summary, the scoring function is fit on PE computed over the whole ride without windowing, and the ride is flagged as impaired at test time when its score exceeds $1 - \alpha$ , with no calibration involved. Detection time is undefined, as the alarm cannot precede the end of the ride.

2) Naive Windowed Classifier: We re-introduce in-ride detection, and thus windowing, but keep running evidence and calibration ablated. Specifically, the scoring function is fit on windowed PE and tests each window separately, with the ride flagged as impaired as soon as the score exceeds $1 - \alpha$

3) Exchangeability Martingale: The window scores in the proposed detector are turned into smoothed conformal p-values against the sober calibration rides, and the Simple Jumper strategy bets on them sequentially as in [23], with jump probability 0.01. The resulting wealth process is tested against $1 / \alpha$ , following Ville’s inequality [24].

4) Recalibrated Martingale: The wealth process introduced above serves as running evidence in the proposed detector, with the threshold calibrated as in Section III-B instead of taken from Ville’s inequality.

5) Shrinkage Sweep: The proposed detector is run with the shrinkage κ set to 0, 10, 20, and 30, the first of which ablates the pull toward the sober baseline.

## E. Results

Table I shows the FAR attained at the four levels of α. The proposed detector stays within the bound throughout, with point estimates consistently below $\alpha ,$ and never significantly above it—as the lower CI ends show. Since the guarantee holds in probability and not in realization, the upper ends do lie above $\alpha ,$ two to three times at the two lowest bounds, though expected of an estimate over only 45 sober rides. The recalibrated martingale holds the bound as well, save at $\alpha = 0 . 1$ , where one sober ride beyond what the bound allows brings the point estimate to 0.11, an excess its lower CI end leaves insignificant. This is unsurprising, as this baseline inherits the full procedure introduced in this work, but for the running evidence, keeping the anytime-valid guarantee intact. It is also worth noting that for both variants the point estimates fall close to the bound, as one would expect from (1), where the rate falls short of α only by the rounding up of the rank. Conversely, the batch classifier, while keeping the FAR well below α, at 0.02 with the lower CI end at 0.00 throughout, yields overly conservative predictions, which cost detection rate at lower BAC level, as shown later at $\alpha = 0 . 0 2 3$ . Finally, both the naive windowed classifier and the exchangeability martingale significantly inflate the FAR, with point estimates several times the bound, and lower CI ends above it. This confirms that testing each window on its own, or betting on time-dependent p-values, forfeits FAR control.

TABLE I  
FALSE ALARM RATE AT FOUR BOUNDS, WITH 95% CIS IN BRACKETS
<table><tr><td>Method</td><td> $\alpha = 0 . 0 2 3$ </td><td> $\alpha = 0 . 0 5$ </td><td> $\alpha = 0 . 1$ </td><td> $\alpha = 0 . 2$ </td></tr><tr><td>Proposed detector</td><td>0.02 [0.00–0.07]</td><td>0.04 [0.00–0.11]</td><td> $0 . 0 9 \ [ 0 . 0 2 – 0 . 1 7 ]$ </td><td> $0 . 1 8 \ [ 0 . 0 7 - 0 . 2 9 ]$ </td></tr><tr><td>Batch classifier</td><td>0.02 [0.00–0.07]</td><td>0.02 [0.00–0.07]</td><td>0.02 [0.00–0.07]</td><td>0.02 [0.00–0.07]</td></tr><tr><td>Naive windowed classifier</td><td>0.11 [0.04–0.20]</td><td>0.27 [0.13–0.42]</td><td>0.44 [0.30–0.58]</td><td>0.80 [0.66–0.93]</td></tr><tr><td>Exchangeability martingale</td><td>0.67 [0.52–0.80]</td><td>0.71 [0.58–0.84]</td><td>0.78 [0.64–0.91]</td><td>0.87 [0.77–0.95]</td></tr><tr><td>Recalibrated martingale</td><td>0.02 [0.00–0.07]</td><td>0.04 [0.00–0.11]</td><td>0.11 [0.02–0.22]</td><td>0.18 [0.07–0.31]</td></tr></table>

TABLE II

DETECTION RATE AND MEDIAN TIME AT $\alpha = 0 . 0 2 3 ,$ , WITH 95% CIS IN BRACKETS AND IQRS IN PARENTHESES
<table><tr><td rowspan="2"></td><td colspan="2">0.08% BAC</td><td colspan="2">0.05% BAC</td></tr><tr><td>Rate</td><td>Time, s</td><td>Rate</td><td>Time, s</td></tr><tr><td>Method Proposed detector  $( \kappa = 1 0 )$ </td><td>0.91 [0.82–0.98]</td><td>25 (22–29)</td><td>0.50 [0.36–0.64]</td><td>27 (22–35)</td></tr><tr><td>Batch classifier</td><td>0.96 [0.88–1.00]</td><td></td><td>0.24 [0.12–0.36]</td><td></td></tr><tr><td>Recalibrated martingale</td><td>0.65 [0.50–0.79]</td><td>54 (52–57)</td><td>0.14 [0.04–0.24]</td><td>60 (56–64)</td></tr><tr><td>Shrinkage sweep of the proposed detector</td><td></td><td></td><td></td><td></td></tr><tr><td> $\kappa = 0$ </td><td>0.63 [0.48–0.78]</td><td>10 (10–10)</td><td>0.22 [0.12–0.34]</td><td>10 (10–10)</td></tr><tr><td> $\kappa = 2 0$ </td><td>0.89 [0.80–0.96]</td><td>29 (26–33)</td><td>0.46 [0.32–0.60]</td><td>34 (28–42)</td></tr><tr><td> $\kappa = 3 0$ </td><td>0.85 [0.74–0.94]</td><td>33 (30–37)</td><td>0.38 [0.24–0.52]</td><td>41 (33–50)</td></tr></table>

Table II shows detection rate at $\alpha = 0 . 0 2 3$ , for the methods that control the FAR. The proposed detector alarms on 91% of the rides at the higher BAC level, within a median of 25 s of riding (IQR 22–29 s), and on 50% of those at the lower level, within a median of 27 s (IQR 22–35 s). Conversely, the recalibrated martingale falls short, detecting only 65% and 14% of them, respectively, at medians of 54 s and 60 s of riding, with its CIs never overlapping those of the proposed detector. Lastly, the batch classifier detects 96% of the rides at the higher BAC level, a margin that the strong overlap of their CIs suggests is insignificant. At the lower level, its conservative predictions bring detection down to 24%, its CI meeting our detector’s at 0.36 without overlapping.

We conclude this section by discussing the shrinkage sweep shown in the bottom half of Table II. When shrinkage is turned off, i.e., $\kappa = 0 .$ , most alarms arrive at the earliest step the window affords, 10 s at either BAC level. This comes at the cost of detection rate, decreasing to 63% and 22%, with neither CI overlapping those of the proposed detector; the fluctuations triggering these early alarms also inflate the sober peaks that set the threshold. For the other values considered, the rate peaks at $\kappa = 1 0$ , the value chosen in this work. It then recedes as the pull toward the sober baseline grows, down to 85% and 38% at $\kappa = 3 0$ , where the median alarm is further delayed at both levels. The FAR stays at 0.02 across the sweep.

## V. ON-DEVICE LATENCY

To assess the feasibility of real-time detection, we implement Algorithm 1 in C, and evaluate it bare metal on a NUCLEO-H533RE board [10], at the default 32 MHz system clock. Specifically, we time 1000 cycles against the 1 ms system tick. In each, the incoming samples are filtered and accumulated into a window of size $\begin{array} { r } {  { w } = \ 1 0 0 0 . } \end{array}$ , with the procedure then carrying on as described, up to the update of the running evidence, where the cycle ends. As specific values do not bear on the size of the computation, both the learned parameters and the input signal are drawn at random within their valid ranges, outside of the timed region.

Over the 1000 cycles, latency averages 262.26 ms, with a standard deviation of 0.91 ms and a range from 260 to 265 ms. A cycle thus takes about a quarter of the second that separates consecutive windows, leaving generous margin for the realtime requirement.

## VI. DISCUSSION AND LIMITATIONS

This paper proposes a new approach in which riders are evaluated for impairment as they ride, rather than at a single point, before or after the trip. Evidence builds from riding itself, reducing the need for pre-ride screening, yet the alarm comes as soon as enough of it has accumulated. The rate at which sober rides raise one is itself a design parameter, with a provably valid guarantee, allowing operators to budget the cost in riding experience beforehand. The calibration procedure assumes only knowledge of sober outcomes, matching the reality of production-grade implementations, where an adequate number of impaired rides cannot be collected due to scarce reporting and ethical concerns. In practice, we envision the scoring function fit as a one-class model on rides from periods in which impairment is unlikely, weekday mornings for instance, with the threshold calibrated on a large number of disjoint rides from the same periods—lowering the smallest attainable bound well below 0.023. This would likely cause a small degree of contamination, though only inflating peaks at the cost of detection rate, leaving the guarantee effectively intact. Centering is equally undemanding, as it needs only a handful of prior rides from the rider, none of which carry labels.

Ethical considerations bear on how an alarm should be acted upon. Interventions should be tiered, transparent, and subject to human review where the response carries a significant cost for the rider. In fact, a single alarm signals an atypical riding pattern with respect to the sober baseline, not a determination of intoxication. It should thus inform soft measures, e.g., a warning or a speed reduction, rather than accusation or sanction. Such measures can be taken onboard, as Section V shows, with no data leaving the vehicle, thus mitigating privacy concerns.

While we presented encouraging results, some limitations should be acknowledged. First, detection performance appears to be stronger at the higher BAC level, that is 0.08%. We argue that, as the aim is to reduce the number of serious injuries, such a concentration should still be considered low: among fatally injured riders with a positive BAC in Sweden, the median ranged from 0.12% to 0.20% across modes [3]. However, detection at such levels remains untested, as inducing that degree of intoxication would expose participants to a higher risk of injury. At the same time, ground truth requires controlled dosing under ethical approval, which puts naturalistic labels out of reach and keeps datasets such as the one used in this study small. The leave-one-participant-out protocol makes the most of what is available, yet cannot push the bound below 0.023 or narrow the CIs (see Section IV-E). The data-collection setup also leaves us with a fixed repertoire of maneuvers on a single vehicle, whereas traffic, slopes, and surfaces add variation in real-world scenarios, which may raise the sober peaks or mask impairment. Recalibrating per vehicle and geographical location is however inexpensive, hence the FAR can effectively be kept under control; precise predictions would then follow from running the detector where and when intoxicated riding is suspected to be prevalent.

## VII. CONCLUSION

This paper presents an alcohol impairment detector that monitors e-scooterists as riding unfolds, raising an alarm as soon as enough evidence accumulates. The FAR is turned into a design parameter and bounded at every step of the ride, a guarantee we prove and confirm empirically, whereas baselines that disregard the temporal dependence of sensor measurements forfeit it. Detection is strong at the higher BAC level and moderate at the lower one, with alarms typically raised within half a minute of riding. In addition, we show that an embedded implementation leaves ample margin for realtime operation, allowing data to stay on the vehicle. The public release of the experimental code supports reproducibility and further work on the topic. Overall, the proposed detector provides a foundation for interventions that reach impaired riders while the trip is still underway, reducing the need for pre-ride screening and nighttime bans.

## DATA AND CODE AVAILABILITY

All data and experimental code required to reproduce the reported results are publicly available at https://github.com/v oi-oss/in-ride-impairment-detection and https://doi.org/10.6 084/m9.figshare.34070439.

## ACKNOWLEDGMENTS

We thank Rahman Amandius from Voi Technology for his valuable feedback on this paper. We also thank Sven Akersten<sup>˚</sup> and Mia Muminovic from Voi Technology for their help with the embedded implementation.

The data used in this paper come from the experiment reported in our prior work [8], run at MicroLab, Chalmers University of Technology, Sweden. We sincerely thank the participants for their time, Marco Dozza, Niyathi Kini, Alexander Rasch, and Ali Mohammadi from Chalmers for the data collection, and Andrea de Bejczy at the University of Gothenburg for medical support and oversight. We are also grateful to Rikard Karlsson and Tord Hansson for access to the Eventhallen, as well as to Henrik Horlin, Peter B¨ ackgren, and Kristina¨ Henricson Briggs for the Tracks facilities.

The data collection experiment was supported by the MicroTox project, funded by VINNOVA (Sweden’s innovation agency) and Drive Sweden under grant 2025-00431, the MinTOX project, funded by the Area of Advance Transport at Chalmers, and Stod till MinTOX, funded by Trafikverket under¨ grant TRV2024/106354. This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation.

## DISCLOSURE OF INTERESTS

The authors have no competing interests to declare that are relevant to the content of this article.

## USE OF GENERATIVE AI

Generative AI tools were used to assist with information retrieval, experimental code development, and text editing. The authors take full responsibility for the content of this paper.

## REFERENCES

[1] Fluctuo, “European shared mobility annual review 2025,” 2026, accessed: Jul. 31, 2026. [Online]. Available: https://fluctuo.com/pdf/e smi-2025.pdf

[2] H. Stigson, I. Malakuti, and M. Klingegard, “Electric scooters accidents:˚ Analyses of two Swedish accident data sets,” Accident Analysis & Prevention, vol. 163, p. 106466, Dec. 2021. [Online]. Available: http://dx.doi.org/10.1016/j.aap.2021.106466

[3] R. R. Pai, R. Fredriksson, and M. Dozza, “Three modes, three profiles: Characterizing fatal crashes on e-scooters, e-bikes, and conventional bicycles in Sweden,” Journal of Safety Research, vol. 97, p. 533–542, Jun. 2026. [Online]. Available: http://dx.doi.org/10.1016/j.jsr.2026.05.0 01

[4] J. B. Cicchino, P. E. Kulie, and M. L. McCarthy, “Severity of e-scooter rider injuries associated with trip characteristics,” Journal of Safety Research, vol. 76, p. 256–261, Feb. 2021. [Online]. Available: http://dx.doi.org/10.1016/j.jsr.2020.12.016

[5] F. Hartz, P. Zehnder, T. Resch, G. Rommermann, V. Hartmann,¨ M. Schwarz, C. Kirchhoff, P. Biberthaler, and M. Zyskowski, “Characteristics of e-scooter and bicycle injuries at a university hospital in a large German city – a one-year analysis,” Injury Epidemiology, vol. 12, no. 1, Jan. 2025. [Online]. Available: http://dx.doi.org/10.1186/s40621-024-00554-w

[6] S. Dibaj, S. Vosough, K. Kazemzadeh, S. O’Hern, and M. N. Mladenovic, “An exploration of e-scooter injuries and severity:´ Impact of restriction policies in Helsinki, Finland,” Journal of Safety Research, vol. 91, p. 271–282, Dec. 2024. [Online]. Available: http://dx.doi.org/10.1016/j.jsr.2024.09.006

[7] Voi Technology, “Tap the helmets: Voi introduces world’s first Reaction Test for e-scooters to discourage drunk riding,” Voi Technology blog, Sep. 2020, accessed: Jul. 31, 2026. [Online]. Available: https://www.voi.com/blog/voi-reaction-test

[8] R. R. Pai, M. Dozza, A. Rasch, A. Mohammadi, and M. Capuccini, “Kinematic signatures of impairment: Detecting alcohol intoxication in e-scooter riders using sensor data and machine learning,” 2026. [Online]. Available: https://arxiv.org/abs/2609.38276

[9] H. Papadopoulos, K. Proedrou, V. Vovk, and A. Gammerman, “Inductive confidence machines for regression,” in Machine Learning: ECML 2002, 13th European Conference on Machine Learning, Helsinki, Finland, August 19-23, 2002, Proceedings, ser. Lecture Notes in Computer Science, T. Elomaa, H. Mannila, and H. Toivonen, Eds., vol. 2430. Springer, 2002, pp. 345–356. [Online]. Available: https://doi.org/10.1007/3-540-36755-1 29

[10] STMicroelectronics, “NUCLEO-H533RE,” Product page, accessed: Aug. 3, 2026. [Online]. Available: https://www.st.com/en/evaluation-t ools/nucleo-h533re.html

[11] B. G. Devcich, L.-m. Ang, M. Wang, and G. S. Larue, “How machine learning has been used to detect alcohol-induced driver impairment using in-vehicle sensors: A systematic review,” Journal of Safety Research, vol. 97, p. 52–66, Jun. 2026. [Online]. Available: http://dx.doi.org/10.1016/j.jsr.2026.01.021

[12] B. Hartung, N. Mindiashvili, R. Maatz, H. Schwender, E. H. Roth, S. Ritz-Timme, J. Moody, A. Malczyk, and T. Daldrup, “Regarding the fitness to ride a bicycle under the acute influence of alcohol,” International Journal of Legal Medicine, vol. 129, no. 3, p. 471–480, Nov. 2014. [Online]. Available: http://dx.doi.org/10.1007/s00414-014-1 104-z

[13] K. Zube, T. Daldrup, M. Lau, R. Maatz, A. Tank, I. Steiner, H. Schwender, and B. Hartung, “E-scooter driving under the acute influence of alcohol—a real-driving fitness study,” International Journal of Legal Medicine, vol. 136, no. 5, p. 1281–1290, Feb. 2022. [Online]. Available: http://dx.doi.org/10.1007/s00414-022-02792-3

[14] R. F. Barber, E. J. Candes, A. Ramdas, and R. J. Tibshirani,\` “Conformal prediction beyond exchangeability,” The Annals of Statistics, vol. 51, no. 2, Apr. 2023. [Online]. Available: http: //dx.doi.org/10.1214/23-aos2276

[15] V. Vovk, I. Nouretdinov, and A. Gammerman, “Testing exchangeability on-line,” in Machine Learning, Proceedings of the Twentieth International Conference (ICML 2003), August 21- 24, 2003, Washington, DC, USA, T. Fawcett and N. Mishra, Eds. AAAI Press, 2003, pp. 768–775. [Online]. Available: http://www.aaai.org/Library/ICML/2003/icml03-100.php

[16] S. Ho, M. Schofield, B. Sun, J. Snouffer, and J. Kirschner, “A martingalebased approach for flight behavior anomaly detection,” in 20th IEEE International Conference on Mobile Data Management, MDM 2019, Hong Kong, SAR, China, June 10-13, 2019. IEEE, 2019, pp. 43–52. [Online]. Available: https://doi.org/10.1109/MDM.2019.00-75

[17] M. Capuccini, R. R. Pai, and L. Carlsson, “Testing by betting for anomaly detection in rental e-scooter GNSS traces,” in Fourteenth Symposium on Conformal and Probabilistic Prediction with Applications, COPA 2025, 10-12 September 2025, London, UK, ser. Proceedings of Machine Learning Research, K. A. Nguyen, Z. Luo, H. Papadopoulos, T. Lofstr ¨ om, L. Carlsson, and H. Bostr ¨ om,¨ Eds., vol. 266. PMLR, 2025, pp. 633–644. [Online]. Available: https://proceedings.mlr.press/v266/capuccini25a.html

[18] V. Chernozhukov, K. Wuthrich, and Y. Zhu, “Exact and robust¨ conformal inference methods for predictive machine learning with dependent data,” in Conference On Learning Theory, COLT 2018, Stockholm, Sweden, 6-9 July 2018, ser. Proceedings of Machine Learning Research, S. Bubeck, V. Perchet, and P. Rigollet, Eds., vol. 75. PMLR, 2018, pp. 732–749. [Online]. Available: http: //proceedings.mlr.press/v75/chernozhukov18a.html

[19] A. V. Oppenheim and R. W. Schafer, Discrete-Time Signal Processing, 3rd ed. Pearson, 2010.

[20] D. A. Winter, Biomechanics and Motor Control of Human Movement. Wiley, Sep. 2009. [Online]. Available: http://dx.doi.org/10.1002/97804 70549148

[21] C. Bandt and B. Pompe, “Permutation entropy: A natural complexity measure for time series,” Physical Review Letters, vol. 88, no. 17, Apr.

2002. [Online]. Available: http://dx.doi.org/10.1103/physrevlett.88.1741 02

[22] R. J. Tibshirani, R. F. Barber, E. J. Candes, and A. Ramdas,\` “Conformal prediction under covariate shift,” in Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, H. M. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alche-Buc, E. B. Fox, and R. Garnett, Eds., 2019,´ pp. 2526–2536. [Online]. Available: https://proceedings.neurips.cc/pap er/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html

[23] V. Vovk, I. Petej, I. Nouretdinov, E. Ahlberg, L. Carlsson, and A. Gammerman, “Retrain or not retrain: conformal test martingales for change-point detection,” in Conformal and Probabilistic Prediction and Applications, 8-10 September 2021, Virtual Event, ser. Proceedings of Machine Learning Research, L. Carlsson, Z. Luo, G. Cherubin, and K. A. Nguyen, Eds., vol. 152. PMLR, 2021, pp. 191–210. [Online]. Available: https://proceedings.mlr.press/v152/vovk21b.html

[24] J. Ville, “Etude critique de la notion de collectif,” Ph.D. dissertation,<sup>´</sup> 1939. [Online]. Available: https://www.numdam.org/item/THESE 193 9 218 1 0