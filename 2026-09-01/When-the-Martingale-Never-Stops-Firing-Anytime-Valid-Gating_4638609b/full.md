# When the Martingale Never Stops Firing: Anytime-Valid Gating on Real Forecast Streams

Weijia Han   
University of Washington   
Seattle, USA   
wh73@uw.edu   
Lisha Qu   
University of Washington   
Seattle, USA   
lishaq@uw.edu

## Abstract

Machine learning systems are increasingly corrected while they run, and the decision of when to intervene is increasingly delegated to statistical monitors. Anytime-valid inference promises evidence that can be acted on at any moment, exactly the guarantee this setting needs, and it is moving from theory into deployed monitoring. Conformal test martingales are the change-detection instrument, and Ville’s inequality caps their false-alarm probability on exchangeable data. The guarantee is conditional. A deployment inherits it only if the stream it monitors behaves exchangeably. The premise is hardest to satisfy where these monitors are most useful, on dependent data and inside loops where the monitor modifies the learner whose scores it reads. It is also rarely measured. We measure it in a pre-specified case study, where such a monitor gates the online updates of a Kalman adapter correcting frozen time-series foundation models on five forecasting streams. On exchangeable synthetic streams, the same implementation fires in at most 1 of 60 runs. On the real streams, at α = 0.05, 135 of 135 clean-stream runs fired. The construction does not explain the firing; the failure comes from the deployed score stream itself. Repeated fires hold the gate’s drift response active, and the gated filter amplifies the very transient it was designed to prevent. The component worth keeping makes no validity claim. Huber-style gating of the filter’s own updates cuts isolated-spike degradation by an order of magnitude with no dataset-specific tuning. Anytime-valid methods proposed for dependent data should therefore be accompanied by null-calibration controls and mechanism traces.

## 1 A pre-specified martingale-gated deployment

Anytime-valid monitors are advocated precisely for online deployment. They fire as soon as evidence accumulates, at a finite-sample false-alarm level that holds at every stopping time [Ramdas et al., 2023]. That level rests on a premise, exchangeability of the monitored score stream. We measure that premise on a single deployment, frozen before any run.

Backbones and adapter. Frozen time-series foundation models, hereafter backbones, are increasingly corrected online by lightweight adapters. We use TiRex [Auer et al., 2025], Chronos-2 [Ansari et al., 2024, 2025], and TimesFM 2.5 [Das et al., 2024], the last a checkpoint release of the cited model line. We correct each backbone with a per-step linear “black-box stacking” adapter (forecast horizon 96, state dimension 192) maintained by a Kalman filter whose hyperparameters (λ, Q, R) adapt online by Adam on the innovation likelihood [West and Harrison, 1997]. An exact block-structured form makes the filter and its hyper-gradient cost 0.163× one-step SGD; the gate’s conformal and martingale path is not counted.

Motivation. The staged study preceding the gate left exactly one problem for a gate to solve. Validation-tuned static forgetting diverges on test; its chosen cell finishes 1.4 to 3.6 orders of magnitude worse than the zero-tuning filter on one dataset across all three backbones. The zerotuning adaptive filter never diverged in any of the 45 staged-study runs (Figure 1). Its one weakness is a single self-excited transient on the weather stream. Cumulative MSE stays near $2 \times 1 0 ^ { - 4 }$ through roughly step 18,400, a brief elevated-error episode follows, and one event at step 18,536 dominates the final cumulative MSE.

![](images/1f77fc0e63cc3cd595f24e68285ffc43c43314e6d9f6422caeb2fca1a1a956a3.jpg)  
Figure 1: Clean-stream divergences per learner across the 45 staged-study runs (5 datasets $\times \ 3$ backbones $\times 3$ seeds). Hatching marks de facto divergences the magnitude guard missed (SGD on traffic under TimesFM 2.5, cumulative MSE 7.97 against the committed threshold of 5, no guard flag, in all three seeds).

Table 1: Fires on clean streams at $\alpha = 0 . 0 5 ;$ each range spans 3 backbones $\times \ 3$ seeds (9 runs). Right column: largest Ljung-Box $p$ across the cell’s full-gate runs; the weather value is a lower-tail degeneracy (see text). <sup>†</sup>Counts cover a shorter stream, truncated by clean-stream divergence (6 of 9 weather full-gate and 3 of 9 traffic SGD-surrogate runs, Section 3).
<table><tr><td>dataset</td><td>Huber-only</td><td>full gate</td><td>SGD surrogate</td><td>max LB p (full gate)</td></tr><tr><td>ETTh2</td><td>31-72</td><td>18-35</td><td>121-173</td><td> $< 1 0 ^ { - 8 }$ </td></tr><tr><td>ETTm1</td><td>459-557</td><td>146-176</td><td>634-712</td><td> $< 1 0 ^ { - 1 5 }$ </td></tr><tr><td>weather</td><td>462-567</td><td>115–277†</td><td>787-1,340</td><td>≈ 1 (degenerate)</td></tr><tr><td>electricity</td><td>77-94</td><td>36-56</td><td>57-75</td><td>0.25</td></tr><tr><td>traffic</td><td>26-43</td><td>22-33</td><td>118–210†</td><td> $< 1 0 ^ { - 1 0 }$ </td></tr></table>

This motivates a gate that (i) robustifies isolated spikes and (ii) declares sustained drift so adaptation can be boosted, with a conformal test martingale [Vovk et al., 2003, 2021, Ville, 1939, Shafer, 2021, Ramdas et al., 2023] making the decision.

Gate construction. The nonconformity score is the filter’s own standardized innovation, and the monitor is a simple-mixture conformal test martingale over randomized conformal p-values from a 250-step sliding calibration window [Vovk et al., 2022]:

$$
s _ { t } = e _ { t } ^ { \top } S _ { t } ^ { - 1 } e _ { t } , \qquad M _ { t } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \prod \varepsilon _ { k } p _ { \tau } ^ { \varepsilon _ { k } - 1 } ,
$$

with $K { = } 1 9$ grid values $\varepsilon _ { k } \in \{ 0 . 0 5 , \ldots , 0 . 9 5 \}$ , threshold $1 / \alpha \ \mathrm { a t } \alpha \ \stackrel { } { = } \ 0 . 0 5$ , and reset on fire. A decision table maps each step to normal, huberize [Huber, 1964], skip, or sustained drift: drift is declared on $\bar { M _ { t } } \geq 1 / \alpha$ (checked first), isolated spikes on $s _ { t }$ above the 99.9th percentile of its fixed chi-square null distribution; the drift response boosts $Q$ by 10×, decaying linearly over 50 recovery steps, applies a covariance soft reset, and moves the forgetting factor to a drift value. Spike classification is suppressed during those recovery steps.

Protocol. Five datasets [Zhou et al., 2021, Wu et al., 2021] (ETTh2, ETTm1, weather, electricity, traffic; test blocks of 12,922 to 52,117 steps), three backbones, three seeds, deterministic cachedforecast replay. The online protocol is the immediate-reveal streaming convention: stride one, each step’s full 96-step label revealed before the next update, so consecutive labels overlap in 95 of 96 entries, which is itself a mechanical source of serial dependence in the scores; the full-label reveal is optimistic for real deployment, and clean means uncontaminated replay with no stationarity claim. Label contamination is fraction-pinned (exactly round(fT) steps at $f \in \{ 1 , 5 , 1 0 \% \}$ ; spike, level-shift, and stuck injectors), applied to the update channel only; accuracy is scored against clean labels. A run that crosses a magnitude-divergence guard is truncated at that step and marked diverged. All verdicts, exclusion rules, and contamination severities were version-stamped before the runs; Appendix A states the composite criterion.

## 2 Premise failure on real streams and its attribution

Three gated arms isolate the gate’s components. The full gate is the deployed object. Huber-only keeps the huberize and skip branches with the drift response disabled. The SGD surrogate drives the same gate from a trailing mean-square score on an SGD learner. Table 1 reports the primary count: across 45 clean-stream reports with 3 seeds each, 135 of 135 runs fired at least once, all 45 full-gate runs among them (first fires at steps 6 to 4,274, median 460; 18 to 1,340 total fires). Under the premise and the textbook guarantee, Ville’s bound at $\alpha = 0 . 0 5$ would cap the expected number of runs with any fire near 7 of 135. Every SGD-surrogate run fired, so firing is not specific to the filter’s scores. Two explanations compete: the score streams themselves, and the deployed deviations from the textbook protocol.

![](images/cde171aa423bc0d73595905ed5e50cd5bf40284a79a04e9d6415a8a5c4708d47.jpg)  
Figure 2: Single-step trace around the natural weather transient (clean stream, TimesFM 2.5, seed 0). (a) Per-step MSE: the full gate (red) tracks the ungated filter until the event, then exceeds it by 12×. (b) Q-boost multiplier with fire ticks (8 plotted of 264): re-fires hold an 8 to 10× boost active at the event.

Inconclusive whiteness diagnostics. Ljung-Box [Ljung and Box, 1978] at 10 lags, on the first-step (h=1) innovation column, non-overlapping in target time, rejects at $p < 0 . 0 1$ in 119 of 135 runs, consistent with serially dependent innovations. But the portmanteau over-rejects under conditional heteroskedasticity even for uncorrelated series, which a shipped illustration demonstrates. The weather full-gate runs report a portmanteau statistic between 0.05 and 0.64 against a $\chi _ { 1 0 } ^ { 2 }$ null with mean 10, so $p \approx 1$ by lower-tail degeneracy, an artifact of the very transient traced in Section 3.

Null controls of the construction. The deployed construction departs from the classical online protocol in three ways: a sliding calibration window in place of the growing bag, scores from an adaptively learned $S _ { t } ,$ and reset on fire. We ran the null control: the same window, randomized p-values, mixture, and reset logic on exchangeable i.i.d. score streams, under both a pre-filled window and the deployed empty-window initialization, at T=20,000 and the longest deployed T=52,117. The pre-filled conditions fired in 0 of 60 runs each; the deployed initialization fired in 1 of 60 at either horizon; heavy-tailed $( | t _ { 2 } | )$ scores fired in 0 of 60. Sixty runs cannot certify strict conservatism. The observed fire fraction stayed below α in every condition, and reset on fire cannot affect the first crossing, the event that Ville’s inequality bounds. On exchangeable inputs the implementation stays quiet, so the pervasive firing does not come from the construction alone; the deployed variant carries no proven Ville guarantee of its own.

A complementary control keeps the full deployed pipeline, drift response included, and feeds it drift-free synthetic streams whose labels are a smooth deterministic pattern plus i.i.d. Gaussian noise, so every monitored score comes from the pipeline’s own adaptation. Were the premise to hold on those scores, at most 3 of 60 runs would be expected to fire. 60 of 60 fired, with 33 to 42 fires per run and first fires inside 67 steps of the test block; the i.i.d.-score control shares the deployed empty-window start and fired in 1 of 60, so the early fires are not a window artifact. With no drift anywhere in the data, the pipeline’s own scores break the premise, though the control does not isolate the responsible element of the loop.

## 3 Fire-triggered amplification of the weather transient

A premise failure matters only through what the fires trigger; we trace the worst case, the self-excited weather transient of Section 1. Figure 2 shows that event at single-step resolution on TimesFM 2.5. Through the elevated-error episode preceding the event, ungated per-step MSE runs 8 to 37 where it typically remains below 1. The martingale re-fires every 5 to 10 steps through that episode, so the drift response’s Q-boost never decays below 8. At the event step the configured boost multiplier reads 9.6 and logged process noise is 2.0× the ungated filter’s.

Table 2: Isolated-spike label contamination (5% of steps, 6 robust-σ): mean per-run degradation (contaminated − clean)/clean, paired within (backbone, seed). Ungated arms are the same learners with no gate; cells losing runs to divergence show [n used/9].
<table><tr><td>dataset</td><td>ungated KF</td><td>Huber-only</td><td>full gate</td><td>ungated SGD</td><td>SGD surrogate</td></tr><tr><td>ETTh2</td><td>+3.32</td><td> $+ 0 . 1 5$ </td><td>+0.24</td><td>+9.04</td><td>+0.15</td></tr><tr><td>traffic</td><td>+3.73</td><td>+0.11</td><td>+0.35</td><td> $+ 2 . 0 7 [ 3 / 9 ]$ </td><td> $+ 3 . 4 0 [ 6 / 9 ]$ </td></tr><tr><td>ETTm1</td><td>+2.68</td><td> $+ 0 . 6 3$ </td><td>+6.28</td><td>+2.90</td><td>+0.06</td></tr><tr><td>electricity</td><td>+0.04</td><td>+0.01</td><td>-0.31</td><td>+1.95</td><td>+2.08</td></tr><tr><td>weather</td><td>+0.01</td><td> $\left. 0 . 0 3 [ 8 / 9 ] \right.$  一</td><td>+0.03 [3/9]</td><td>+0.00</td><td>-0.02</td></tr></table>

Under the accumulated drift response, TimesFM 2.5’s per-step MSE at the event reaches $1 . 1 5 5 \times 1 0 ^ { 5 }$ against the ungated filter’s $9 . 8 \times 1 0 ^ { 3 }$ , and its final cumulative MSE rises from 0.36 to between 2.6 and 3.1 across seeds. On TiRex and Chronos-2, where amplification is larger, the same event crosses the magnitude-divergence guard on the clean stream itself, in six of six runs, all at step 18,536. It crosses again in all 54 of 54 contaminated weather runs on those two backbones. Component and feedback controls in Appendix B attribute the divergence to the covariance soft reset and to the immediate reveal; the boost alone never diverges.

The spike branch of the decision table never activated. The recorded action at the event was a normal update, recovery-window suppression was active, and the event’s conformal p of 0.011 reflected a window already adapted to the episode. The trace also exposed a configuration incoherence in the deployed gate. The base forgetting factor is pinned at the guard floor (0.90), below the drift value (0.95), so the drift response simultaneously slowed forgetting and boosted process noise.

## 4 Component ablations under contamination and drift

The drift response caused the damage. The remaining question is which component is worth keeping. Table 2 isolates the robust-update half (huberize and skip). It delivers the spike robustness without the drift response and matches or beats the full gate on ETTh2, traffic, and ETTm1, the three datasets where the ungated learner degrades appreciably. The same gating transfers to SGD through the surrogate score on ETTh2 and ETTm1. On traffic the surrogate arm does worse than the ungated SGD learner. Divergence is itself an outcome, so the bracketed survivor-only cells favor the arms that lose runs; we read them qualitatively. The Huber-only arm still uses the martingale, though only as a classifier, and nothing here needs an anytime-validity claim.

The same protocol also produced negative results. Sustained contamination defeats innovation gating. Level shift at 5% degrades every arm by $+ 7 . 8 \ \mathrm { t o } + 1 9 . 3 $ on ETTh2 and ETTm1, with the gated arms often worst. On traffic it degrades every Kalman arm by $+ 9 . 5 \mathrm { t o } + 1 8 . 3 .$ . The stuck injector degrades the full gate by +1.7 to +14.2 on the four datasets outside weather, against at worst +1.8 for the ungated filter and +0.5 for the SGD arms. On synthetic piecewise drift the full gate adapts more slowly than the ungated filter on three of three seeds (Appendix C). The pre-specified composite gate criterion (Appendix A) fails on three of three backbones.

## 5 Implications for e-value methods

Deployment reports should pair every anytime-valid claim with raw fire counts on clean data, a null-calibration control ofthe deployed implementation at its horizon, and dependence diagnostics. Guarantee-free components should be evaluated separately from guaranteed ones. Here the robustupdate half carried the measured benefit and the martingale-triggered response caused the damage, so an “anytime-valid gating” framing would credit the benefit to the guarantee and hide the source of the damage.

On the detection half, e-detectors offer sequential change detection under average-run-length control [Shin et al., 2024], though replayed on the deployed p-streams they alarm in 135 of 135 runs at every tested threshold (Appendix B), so a detector swap does not repair the premise. E-processes can test the exchangeability assumption itself [Ramdas et al., 2022] and extend across filtrations [Choe and Ramdas, 2026], and adaptive conformal updating tracks calibration under shift [Gibbs and Candès, 2021]. None of these supplies a response policy with bounded cost under wrong alarms, and that policy is what this study finds missing.

## References

Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, and Yuyang Wang. Chronos: Learning the language of time series. Transactions on Machine Learning Research, 2024.

Abdul Fatir Ansari, Oleksandr Shchur, Jaris Küken, Andreas Auer, Boran Han, Pedro Mercado, Syama Sundar Rangapuram, Huibin Shen, Lorenzo Stella, Xiyuan Zhang, Mononito Goswami, Shubham Kapoor, Danielle C. Maddix, Pablo Guerron, Tony Hu, Junming Yin, Nick Erickson, Prateek Mutalik Desai, Hao Wang, Huzefa Rangwala, George Karypis, Yuyang Wang, and Michael Bohlke-Schneider. Chronos-2: From univariate to universal forecasting, 2025. arXiv:2510.15821.

Andreas Auer, Patrick Podest, Daniel Klotz, Sebastian Böck, Günter Klambauer, and Sepp Hochreiter. Tirex: Zero-shot forecasting across long and short horizons with enhanced in-context learning, 2025. arXiv:2505.23719.

Yo Joong Choe and Aaditya Ramdas. Combining evidence across filtrations. Journal of the Royal Statistical Society Series B: Statistical Methodology, 2026. doi: 10.1093/jrsssb/qkag058. Advance article.

Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for time-series forecasting. In Proceedings of the 41st International Conference on Machine Learning (ICML 2024), volume 235 of Proceedings of Machine Learning Research, pages 10148–10167, 2024.

Isaac Gibbs and Emmanuel J. Candès. Adaptive conformal inference under distribution shift. In Advances in Neural Information Processing Systems, volume 34, pages 1660–1672, 2021.

Peter J. Huber. Robust estimation of a location parameter. Annals ofMathematical Statistics, 35(1): 73–101, 1964.

G. M. Ljung and G. E. P. Box. On a measure of lack of fit in time series models. Biometrika, 65(2): 297–303, 1978.

Aaditya Ramdas, Johannes Ruf, Martin Larsson, and Wouter M. Koolen. Testing exchangeability: Fork-convexity, supermartingales and e-processes. International Journal of Approximate Reasoning, 141:83–109, 2022.

Aaditya Ramdas, Peter Grünwald, Vladimir Vovk, and Glenn Shafer. Game-theoretic statistics and safe anytime-valid inference. Statistical Science, 38(4):576–601, 2023.

Glenn Shafer. Testing by betting: A strategy for statistical and scientific communication. Journal of the Royal Statistical Society Series A: Statistics in Society, 184(2):407–431, 2021.

Jaehyeok Shin, Aaditya Ramdas, and Alessandro Rinaldo. E-detectors: A nonparametric framework for sequential change detection. New England Journal ofStatistics in Data Science, 2024.

Jean Ville. Étude critique de la notion de collectif. Gauthier-Villars, Paris, 1939.

Vladimir Vovk, Ilia Nouretdinov, and Alex Gammerman. Testing exchangeability on-line. In Proceedings of the Twentieth International Conference on Machine Learning (ICML 2003), pages 768–775, 2003.

Vladimir Vovk, Ivan Petej, Ilia Nouretdinov, Ernst Ahlberg, Lars Carlsson, and Alex Gammerman. Retrain or not retrain: Conformal test martingales for change-point detection. In Proceedings of the Tenth Symposium on Conformal and Probabilistic Prediction and Applications (COPA 2021), volume 152 of Proceedings ofMachine Learning Research, pages 191–210, 2021.

Vladimir Vovk, Alex Gammerman, and Glenn Shafer. Algorithmic Learning in a Random World. Springer, 2nd edition, 2022.

Mike West and Jeff Harrison. Bayesian Forecasting and Dynamic Models. Springer, 2nd edition, 1997.

Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. In Advances in Neural Information Processing Systems, volume 34, 2021.

Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 11106–11115, 2021. doi: 10.1609/aaai.v35i12.17325.

## A Reproducibility and scope

We release deterministic cached-forecast replay, pre-specified verdicts with an errata trail, raw counts beside every p-value, and the null-control scripts. All results use the single gate configuration of Section 1, so every claim here is a claim about that deployment. Seeds are Monte Carlo replicates conditional on each fixed cached stream. Each gated arm draws its own randomized conformal p-values per seed, and each contaminated run spawns its own RNG. The pre-specified composite criterion requires the gated arm to beat the ungated arm, per contamination kind, on at least three of five comparable dataset cells (cells where the ungated arm itself degrades by less than five percent are excluded as incomparable); the gate passes only if every evaluated kind passes. Contamination magnitudes are in units of a robust per-horizon scale of the clean labels (1.4826 times the median absolute deviation). Spike alters isolated uniformly chosen steps by six scale units with per-step random sign; level-shift applies a three-unit offset over one contiguous block with one sign per horizon; stuck freezes one contiguous block at its onset value with no magnitude at all. The TimesFM 2.5 backbone is the google/timesfm-2.5-200m-pytorch checkpoint release, pinned by revision alongside the other two backbones in the code release. Code and artifacts will be released, anonymized during review.

## B Component and feedback ablations

All numbers replay the committed run artifacts; MSE figures quote TimesFM 2.5, the backbone of the Section 3 trace, and divergence counts cover all three backbones. Response components. On clean weather with immediate feedback the soft-reset-only arm reproduces the full gate’s divergence in six of six TiRex and Chronos-2 runs at step 18,536 and ends inside the full gate’s range on TimesFM 2.5 (2.50 to 3.03 against 2.57 to 3.06), while the boost-only arm diverges in zero of nine runs and ends near the ungated filter (0.378 to 0.383 against 0.361); the forgetting-only arm is likewise benign (0.341 to 0.342). The covariance soft reset is sufficient for the catastrophe; the boost alone is not.

Delayed feedback. With labels revealed 96 steps late the clean-weather full gate diverges in zero of nine runs, and final cumulative MSE stays above the ungated filter on every backbone (TiRex 0.50 to 0.53 against 0.286, Chronos-2 8.3 to 9.1 against 5.75, TimesFM 2.5 1.18 to 1.23 against 0.517). The step-18,536 catastrophe requires the immediate full-vector reveal.

A martingale-free cutoff. A fixed-threshold huberize-and-skip arm with frozen hyperparameters and no martingale holds isolated-spike degradation to at most +0.05 in every dataset, stronger than Huber-only on the three materially affected datasets. Its clean-weather cost is backbone-dependent, from well below the ungated filter on TiRex and Chronos-2 (0.37 against 2.03; 0.23 against 3.58) to 9.5× above it on TimesFM 2.5 (3.41 against 0.36).

E-detector replay. SR and CUSUM e-detectors [Shin et al., 2024] built from the deployed betting family, replayed on the committed p-value streams, first reproduce the recorded fires exactly in 405 of 405 runs. Both then alarm in 135 of 135 paper runs at every threshold $A \in \{ 1 0 ^ { 3 } , 1 0 ^ { 4 } , 1 0 ^ { 5 } \}$ , with SR at $A { = } 1 0 ^ { 4 }$ producing 193/320/1,464 min/median/max alarms per run against null expected counts of 1.3 to 5.2, while the same detectors stay calibrated on i.i.d. p-streams and on i.i.d. scores passed through the deployed empty-start sliding window. The detector family inherits the premise failure; the remedy must restore score exchangeability.

## C Synthetic drift latency

Table 3 reports the synthetic-drift latency study over three seeds with declared break points and a per-break bounded search. It gives the mean adaptation latency, the number of steps after each declared break until a trailing-error recovery criterion is met (never-recovered breaks are capped at the bounded search horizon), and the final cumulative MSE per arm. The full gate is slower than the ungated filter at its own motivating task on every seed, and its final error is roughly four times worse.

Table 3: Mean latency (steps) and final cumulative MSE on the synthetic drift benchmark.
<table><tr><td></td><td colspan="3">mean latency</td><td colspan="3">final MSE</td></tr><tr><td>arm</td><td>seed 0</td><td>seed 1</td><td>seed 2</td><td>seed 0</td><td>seed 1</td><td>seed 2</td></tr><tr><td>ungated filter</td><td>126.8</td><td>128.0</td><td>95.2</td><td>0.092</td><td>0.101</td><td>0.098</td></tr><tr><td>full gate</td><td>154.8</td><td>137.5</td><td>128.0</td><td>0.342</td><td>0.447</td><td>0.365</td></tr><tr><td>SGD ungated</td><td>226.0</td><td>250.0</td><td>216.8</td><td>0.327</td><td>0.333</td><td>0.330</td></tr></table>