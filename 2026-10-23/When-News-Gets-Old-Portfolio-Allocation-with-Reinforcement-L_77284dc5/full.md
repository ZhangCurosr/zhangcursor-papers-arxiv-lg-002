# When News Gets Old: Portfolio Allocation with Reinforcement Learning and Large Language Models

Bolin Song

Cornell University

bs833@cornell.edu

Surui Xiang

United Nations

surui.xiang@un.org

Abstract—Financial news can influence investment decisions long after publication. Timely news may reach an investor before prices fully adjust, whereas older news may already be reflected in prices. We examine two ways to handle aging signals: retain them and give a portfolio policy timing information, or filter them out before portfolio selection. A fixed large language model (LLM) extracts signals from public SEC disclosures. We compare these treatments in reinforcement learning (RL) policies and LLMs that choose portfolios directly. We evaluate returns after trading costs in historical tests over 2023–2025. RL choices respond to timing information in a controlled test, but historical returns show no consistent benefit from adding it. For the LLM, filtering old news produced the highest average net return, reaching 18.43% per year when news delivery was delayed by four weeks. This exceeded the returns of all RL strategies that used news.

Index Terms—portfolio allocation, reinforcement learning, large language models, news timing, signal filtering

## I. INTRODUCTION

A company reports stronger sales, and a large language model (LLM) assigns the report a positive signal. A month later, should that signal still influence the portfolio in the same way? Prices may already reflect the news, and the company’s outlook may have changed. The report still describes a favorable development, but its usefulness for a new trade is less certain.

Reinforcement learning (RL) has been used to learn trading decisions that account for existing positions and transaction costs [1], [2]. LLM-assisted systems supply textual signals to RL [3], while other LLM agents use financial memory to make decisions directly [4]. The value of acting on a signal depends on how long it remains useful and on trading costs [5], [6]. Tetlock [7] examines stale information through repeated news content. We focus on a practical choice as a report grows older: retain its signal and tell the portfolio policy about its timing, or remove the signal before the policy makes a decision.

We compare these approaches in a weekly allocation task on US stocks. An LLM reads periodic SEC disclosures and assigns positive, neutral, or negative signals based on the company’s business performance or outlook. Either an RL policy or an LLM then uses these signals, market data, and current holdings to choose a portfolio. Both choose from the same predefined portfolios and can keep their existing holdings. We compare three input settings: signals without timing information, signals with timing information, and signals filtered out once the source report is more than eight weeks old.

We measure returns after trading costs over 2023–2025, making reports available without added delay or holding them back for one or four weeks. Separately, we make already available reports appear older while keeping market data, holdings, and extracted signals unchanged. This second test checks whether news age alone changes the selected portfolio.

Providing timing information changes RL choices in the controlled test but does not consistently improve historical net returns. For the LLM portfolio policy, filtering old signals yields the highest average net return. With a four-week delivery delay, it earns 18.43% per year after costs, exceeding all three RL policies that use news. A simple age filter can therefore offer a useful alternative to retaining older signals and asking the portfolio model to interpret their timing.

## II. DECISION FRAMEWORK

## A. Observations and News Timing

We use an LLM to extract positive, neutral, or negative signals from companies’ periodic SEC disclosures. Each week, RL or an LLM combines these signals with market data and holdings to choose from the same target portfolios or keep its holdings. Signals persist until newer reports arrive, unless filtered.

Consider N assets and weekly decisions indexed by t. The policy observes

$$
o _ { t } = \big ( \bar { \mathbf { w } } _ { t } , \mathbf { m } _ { t } , \mathbf { z } _ { t } , \tau _ { t } , \mathbf { g } _ { t } \big ) ,\tag{1}
$$

where $\bar { \mathbf { w } } _ { t }$ contains portfolio weights known at the decision time, m contains past returns and volatility, and $z _ { i , t } \in$ {−1, 0, 1} is a negative, neutral, or positive signal about the company’s business performance or outlook.

News age measures the number of weeks since a report was published. Time since arrival measures how long the report has been available to the policy. We refer to these two inputs together as timing information. For example, a report published four weeks ago but arriving today has a news age of four weeks and a time since arrival of zero. These quantities form $\tau _ { t }$ and g<sub>t</sub>, respectively.

![](images/cbbc84bba5dea6721decb7b24115655fef8417bbeec5fe190cac7868c8740de8.jpg)  
Fig. 1. News Signals and Portfolio Decisions. An LLM extracts news signals, while report dates determine news age. The RL policy and direct LLM baseline choose portfolios independently, with separate holdings under common backtesting rules. Table I specifies each method’s inputs.

Let $b _ { t }$ be decision time in weeks, $u _ { i , t }$ the publication time of the latest visible report, and $\boldsymbol { v } _ { i , t } = \boldsymbol { u } _ { i , t } + \delta$ its arrival under delay δ. Then

$$
\tau _ { i , t } = b _ { t } - u _ { i , t } , \qquad g _ { i , t } = b _ { t } - v _ { i , t } , \qquad v _ { i , t } \leq b _ { t } .\tag{2}
$$

A fixed LLM reads each report once and returns a report signal with a short quotation from the report. Its input contains only the report. All policies share the stored signals. A zero signal can mean a neutral report, no available report, or an output that failed a format or evidence check. Appendix C gives the prompts and a recorded report-to-portfolio example.

We compare each news treatment against News, which uses market data, holdings, and report signals without timing information. News + Age adds the two timing inputs. Filtered News sets a report’s signal to zero once it is more than eight weeks old, without supplying timing inputs. The stock remains available for investment. The filtered signal is

$$
\widetilde { z } _ { i , t } = z _ { i , t } \mathbf { 1 } \{ \tau _ { i , t } \leq h \} ,\tag{3}
$$

where the common age limit is $h = 8$ weeks for RL and LLM portfolio policies. Table I includes the market-only and passive benchmarks.

## B. Actions, Reward, and Learning

A target portfolio specifies how much capital to allocate to each stock. For example, one target assigns 40% to one stock and spreads the remaining 60% equally across the remaining stocks. Each week, the policy selects a target from the predefined list or keeps its current holdings. Rebalancing to the selected target incurs trading costs. HOLD keeps the existing shares. All strategies are fully invested and use the same execution schedule. Equal Allocation restores $q _ { i } = 1 / N$ each week. Buy and Hold starts at the same weights and preserves its shares, allowing weights to drift.

Let $V _ { t }$ be wealth before trading, with subsequent wealth $V _ { t + 1 }$ already reflecting trading costs. The reward and objective are

$$
r _ { t + 1 } = \kappa \log \frac { V _ { t + 1 } } { V _ { t } } , \qquad \operatorname* { m a x } _ { \pi } \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { T - 1 } \gamma ^ { t } r _ { t + 1 } \right] .\tag{4}
$$

Here $T$ is the horizon, $\gamma$ discounts later rewards, and $\kappa > 0$ sets reward scale. Appendix B gives the execution and cost accounting.

Each RL information setting has a separately trained Double DQN policy [8], with the same architecture and training budget. During evaluation, the trained network chooses the action with the highest predicted value, without random exploration or further training. LLM portfolio policies receive the corresponding numerical fields and choose from the same list of portfolios. Assets are represented by anonymous indices; company names, dates, and raw reports are omitted. News and filtered-news prompts are identical; only the supplied signals change. Invalid action outputs map to HOLD. Each policy maintains its own evolving holdings.

## III. EXPERIMENTAL DESIGN

## A. Historical Comparison

We evaluate weekly portfolio allocation for ten US stocks over 2023–2025, using historical prices and signals from periodic SEC disclosures. RL training and validation use data from years before each annual test. Every method begins each test year with an equal-weight portfolio.

Each week, the portfolio is chosen using information available at that week’s last close. Trades execute at the following week’s first open and cost 10 basis points per dollar bought or sold. We test report delivery delays of zero, one, and four weeks; report text and publication dates remain unchanged. All methods use the same underlying data and trading rules. Each RL policy is trained using all three delay settings. Stock selection and training details are given in Appendix B.

GPT-4.1 mini extracts the report signals and selects portfolios for the LLM results in Table II. At a four-week delivery delay, we also compare Llama 3.3 and GPT-6 Astra as portfolio decision makers using the same signals. Model settings are given in Appendix C.

## B. Decision Changes and Net Returns

Historical tests compare investment returns when reports arrive with an additional delay of zero, one, or four weeks.

A separate test asks whether older reports change RL’s portfolio choice. Reports, market data, holdings, and original signals stay fixed; news age and time since arrival increase equally, leaving delivery delay unchanged. Filtered News reapplies the eight-week cutoff, potentially zeroing more signals. We measure changes in portfolio choice, not returns from those changes. Let $o _ { t } ^ { + \ell }$ denote the input after adding ℓ weeks to both timing inputs.

TABLE I  
DECISION INPUTS AND COMPARISON METHODS.
<table><tr><td>Policy Input setting</td><td></td><td>Market data &amp; holdings</td><td>News signals</td><td>News age &amp; time since arrival</td><td>Decision maker</td></tr><tr><td colspan="2">Equal Allocation</td><td></td><td></td><td></td><td>Fixed weights</td></tr><tr><td>Buy and Hold</td><td></td><td></td><td></td><td></td><td>Fixed positions</td></tr><tr><td>RL</td><td>Market Data</td><td>√</td><td></td><td></td><td>RL</td></tr><tr><td></td><td>Market Data + News</td><td>√</td><td>√</td><td></td><td>RL</td></tr><tr><td></td><td>Market Data + News + News Age</td><td>√</td><td>√</td><td></td><td>RL</td></tr><tr><td></td><td>Market Data + Filtered News</td><td>√</td><td>√*</td><td></td><td>RL</td></tr><tr><td>LLM</td><td>Market Data</td><td>√</td><td></td><td></td><td>LLM</td></tr><tr><td></td><td>Market Data + News</td><td>√</td><td>V</td><td></td><td>LLM</td></tr><tr><td></td><td>Market Data + News + News Age</td><td>√</td><td>√</td><td></td><td>LLM</td></tr><tr><td></td><td>Market Data + Filtered News</td><td>√</td><td>√*</td><td></td><td>LLM</td></tr></table>

<sup>∗</sup>Reports older than eight weeks have their signals set to zero; numerical ages are not passed to the filtered policy.

TABLE II  
HISTORICAL RETURNS AND TRADING OUTCOMES.
<table><tr><td colspan="2"></td><td colspan="3">Annualized return after costs</td><td colspan="4">Results with a four-week news delay</td></tr><tr><td></td><td>Policy Input setting</td><td>0 weeks</td><td>1 week</td><td>4 weeks</td><td>Sharpe ratio</td><td>Maximum drawdown</td><td>Annual trading volume (×)</td><td>Total trading costs</td></tr><tr><td colspan="2">Equal Allocation</td><td>17.63%</td><td>17.63%</td><td>17.63%</td><td>1.13</td><td>10.44%</td><td>1.09</td><td>0.12%</td></tr><tr><td colspan="2">Buy and Hold</td><td>17.19%</td><td>17.19%</td><td>17.19%</td><td>1.13</td><td>10.45%</td><td>0.00</td><td>0.00%</td></tr><tr><td>RL</td><td>Market Data</td><td>19.07%</td><td>19.07%</td><td>19.07%</td><td>1.00</td><td>12.99%</td><td>25.12</td><td>2.69%</td></tr><tr><td></td><td>Market Data + News</td><td>16.22%</td><td>16.95%</td><td>17.81%</td><td>0.96</td><td>12.49%</td><td>23.72</td><td>2.55%</td></tr><tr><td></td><td>Market Data + News + News Age</td><td>15.41%</td><td>15.11%</td><td>18.02%</td><td>0.98</td><td>12.88%</td><td>24.75</td><td>2.64%</td></tr><tr><td></td><td>Market Data + Filtered News</td><td>18.54%</td><td>19.20%</td><td>18.03%</td><td>0.97</td><td>13.03%</td><td>25.40</td><td>2.73%</td></tr><tr><td>LLM</td><td>Market Data</td><td>18.02%</td><td>18.02%</td><td>18.02%</td><td>1.09</td><td>11.07%</td><td>20.97</td><td>2.25%</td></tr><tr><td></td><td>Market Data + News</td><td>17.60%</td><td>16.19%</td><td>16.02%</td><td>0.98</td><td>12.01%</td><td>18.87</td><td>2.02%</td></tr><tr><td></td><td>Market Data + News + News Age</td><td>16.73%</td><td>13.26%</td><td>17.17%</td><td>1.07</td><td>11.31%</td><td>13.01</td><td>1.36%</td></tr><tr><td></td><td>Market Data + Filtered News</td><td>20.16%</td><td>18.35%</td><td>18.43%</td><td>1.12</td><td>10.83%</td><td>19.03</td><td>2.05%</td></tr></table>

Metrics are computed within each annual test and averaged equally; RL also averages separately trained policies. The Sharpe ratio uses a zero risk-free rate. Maximum drawdown is the largest peak-to-trough decline. Annual trading volume sums buys and sells relative to portfolio value. Total costs are fees relative to each year’s initial capital, then averaged. All methods use the same equal-weight starting portfolio with no initial trading fee and no forced terminal sale.

$$
\rho ( \ell ) = \frac { 1 } { | \mathcal { B } | } \sum _ { t \in \mathcal { B } } \mathbf { 1 } \{ \pi ( o _ { t } ^ { + \ell } ) \neq \pi ( o _ { t } ) \} ,\tag{5}
$$

computed over fixed validation observations B for each trained policy. News-only inputs do not change.

Investment value is evaluated on historical prices using net annualized return, Sharpe ratio, maximum drawdown, trading volume, and fees. Within each decision method, we subtract the annualized return of News from those of News + Age and Filtered News, reporting these differences in %. Metrics are averaged equally across annual tests.

Following uncertainty-aware RL evaluation [9], [10], we report training variation separately from uncertainty in return differences. Paired time-block intervals resample the same weeks across methods and trained policies. Appendix A gives annual results, training repetitions, and resampling details. Supporting checks vary fees and examine signal persistence in a synthetic market (Appendix B).

## IV. RESULTS

## A. LLM Portfolio Returns

Filtering old news gives the LLM its highest average net return (Table II). With a four-week delivery delay, removing signals from reports more than eight weeks old produces an average annualized net return of 18.43%. Keeping the signals produces 16.02% without timing information and 17.17% with timing information. The LLM with filtering also earns a higher return than all three RL policies that use news. RL with Market Data returns 19.07%, above the LLM’s 18.43% with filtering. News does not always improve on market data alone, and its treatment matters.

![](images/5b0d4cfb4a54156b01a5c4448cda72b226302651e9df175223e688f604d47efa.jpg)  
Fig. 2. Portfolio decisions as reports age. Market features, holdings, and original LLM signals stay fixed while publication and arrival times move backward together. Filtering reapplies its cutoff. Lines average trained policies; Appendix A reports variation across training runs.

Additional LLM comparisons show that the effect of adding timing information varies by model and year (Appendix A).

## B. RL Responses to News Age

RL changes its portfolio choices in response to news age (Fig. 2). When the same reports are made twelve weeks older, RL with News + Age selects a different portfolio in 52% of cases, averaged across trained policies on 39 fixed validation observations. Market data, holdings, and the original report signals remain unchanged in this test.

RL with Filtered News changes its choice in 26% of cases under the same twelve-week increase. Here, making reports older causes more signals to cross the eight-week cutoff and be removed. Its choices stop changing once no additional signals are removed. RL with News keeps the same choices throughout because its inputs do not change. Decision changes show sensitivity to timing information, without establishing that RL systematically reduces the influence of older signals.

## C. RL Portfolio Returns

In the historical backtests, adding timing information does not consistently improve RL net returns. It gives RL lower average returns than News at delivery delays of zero and one week. With a four-week delay, the average annualized net return increases only from 17.81% for News to 18.02% for News + Age. Filtered News produces a similar return of 18.03%. The 95% confidence intervals for both gains over News include zero (Fig. 3).

At the four-week delay, filtering removes roughly 70% of nonzero news signals. Keeping these older signals and giving RL timing information offers no clear return advantage over filtering them out.

![](images/be51508489acbc6a2bb03ae66b2b87b1d33a001f09bcf6dbbc9e398a0f3fbb0d.jpg)  
Fig. 3. Changes in RL annualized net return relative to Market Data + News at a four-week delivery delay. Providing news age adds both timing inputs; filtering removes signals older than eight weeks. Dots are paired mean differences; bars are 95% time-block intervals (Appendix A).

## D. Trading Costs

Higher trading fees substantially reduce the return of RL with News + Age. At a four-week delivery delay, we increase the fee from 10 to 30 basis points per dollar traded while keeping the portfolio decisions fixed. Its average annualized net return falls from 18.02% to 12.32%. Buy and Hold remains at 17.19% because it makes no trades after the initial allocation.

## V. CONCLUSION

We compared retaining older signals with timing information and filtering them before portfolio selection, using RL and LLM policies on US stocks over 2023–2025. All policies shared signals extracted from SEC disclosures. RL choices responded to timing information in the controlled test, while historical tests showed no consistent net return improvement from adding it. The LLM policy performed best when old signals were filtered out. With a four-week delivery delay, it earned an average annualized net return of 18.43%, exceeding all three RL policies that used news.

These results highlight signal filtering as a practical part of LLM-based portfolio design. Retaining older signals and supplying their timing was less effective for the LLM policy than simply removing them. For RL, the findings show that responding to news timing and improving investment performance are distinct outcomes that should be evaluated together.

## REFERENCES

[1] J. Moody and M. Saffell, “Learning to trade via direct reinforcement,” IEEE Transactions on Neural Networks, vol. 12, no. 4, pp. 875–889, 2001.

[2] Z. Jiang, D. Xu, and J. Liang, “A deep reinforcement learning framework for the financial portfolio management problem,” arXiv preprint arXiv:1706.10059, 2017.

[3] M. Benhenda, “FinRL-DeepSeek: LLM-infused risk-sensitive reinforcement learning for trading agents,” arXiv preprint arXiv:2502.07393, 2025.

[4] Y. Yu, H. Li, Z. Chen, Y. Jiang, Y. Li, D. Zhang, R. Liu, J. W. Suchow, and K. Khashanah, “FinMem: A performance-enhanced LLM trading agent with layered memory and character design,” arXiv preprint arXiv:2311.13743, 2023.

[5] N. Gârleanu and L. H. Pedersen, “Dynamic trading with predictable returns and transaction costs,” The Journal of Finance, vol. 68, no. 6, pp. 2309–2340, 2013.

[6] ——, “Dynamic portfolio choice with frictions,” Journal of Economic Theory, vol. 165, pp. 487–516, 2016.

[7] P. C. Tetlock, “All the news that’s fit to reprint: Do investors react to stale information?” The Review of Financial Studies, vol. 24, no. 5, pp. 1481–1512, 2011.

[8] H. van Hasselt, A. Guez, and D. Silver, “Deep reinforcement learning with double q-learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 30, no. 1, 2016, pp. 2094–2100.

[9] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger, “Deep reinforcement learning that matters,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018, pp. 3207– 3214.

[10] R. Agarwal, M. Schwarzer, P. S. Castro, A. C. Courville, and M. G. Bellemare, “Deep reinforcement learning at the edge of the statistical precipice,” in Advances in Neural Information Processing Systems, vol. 34, 2021. [Online]. Available: https://papers.nips.cc/paper/2021/ hash/f514cec81cb148559cf475e7426eed5e-Abstract.html

[11] SPDR Dow Jones Industrial Average ETF Trust, “Annual report, october 31, 2019,” SEC Form N-30D, 2019, accessed: Oct. 20, 2026. [Online]. Available: https://www.sec.gov/Archives/edgar/data/1041130/ 000119312519320843/d823731dn30d.htm

[12] Yahoo Finance, “What is the adjusted close?” accessed: Oct. 20, 2026. [Online]. Available: https://in.help.yahoo.com/kb/ adjusted-close-sln28256.html

## APPENDIX A

## ANNUAL RESULTS AND ADDITIONAL DIAGNOSTICS

## A. Uncertainty and Annual Variation

Table III gives the paired intervals for the average return differences in Fig. 3. For each bootstrap replicate, we sample circular blocks of eight consecutive weeks separately within each year. A year’s sampled indices are shared by all compared methods and training seeds. Each trained policy’s sampled path is compounded and annualized before averaging across policies and then equally across years. Percentile intervals use 10,000 replicates, conditional on the recorded policies and historical periods. Annual segments are resampled independently, which neglects dependence across year boundaries. Training seeds quantify variation across training runs and are held together during time resampling.

TABLE III  
PAIRED RETURN DIFFERENCES AT FOUR-WEEK DELAY.
<table><tr><td>Comparison</td><td>Difference (%)</td><td>95% interval (%)</td></tr><tr><td>News + Age — News</td><td>+0.21</td><td>[−3.73,4.29]</td></tr><tr><td>Filtered News — News</td><td>+0.22</td><td>[−2.93, 3.44]</td></tr></table>

Table V gives the annual paired contrasts. None of the 36 primary annual timing/filter comparisons has an eight-weekblock 95% interval excluding zero. Four-week blocks give the same count; with 13-week blocks, only mini filtering at zero delay in 2025 excludes zero. These are unadjusted exploratory intervals. Variation across training runs is reported separately: the four-week RL timing contrast spans positive and negative values across training runs in every year.

## B. LLM Portfolio Policies

The direct-decision comparison keeps the mini-extracted signals fixed and changes only the model selecting a portfolio. Table IV shows the News + Age versus News return contrasts at four-week delay. All nine corresponding paired intervals contain zero. The effect of timing varies with both the model and the annual period.

TABLE IV  
TIMING EFFECTS IN LLM PORTFOLIO POLICIES.
<table><tr><td></td><td colspan="3">Return change (%)</td></tr><tr><td>Model</td><td>2023</td><td>2024</td><td>2025</td></tr><tr><td>GPT-4.1 mini</td><td>-3.42</td><td>+6.40</td><td>+0.48</td></tr><tr><td>Llama 3.3</td><td>+0.72</td><td>+0.54</td><td>-1.76</td></tr><tr><td>GPT-6 Astra</td><td>+5.35</td><td>+0.50</td><td>-4.36</td></tr></table>

## C. Variation in the Behavioral Response

At twelve weeks of added age, the action-change rates across trained policies range from 25.6% to 74.4% for News + Age and from 7.7% to 41.0% for Filtered News. Figure 2 displays their respective means.

For the same fixed inputs as Fig. 2, we also measure $\textstyle \sum _ { i } z _ { i } ( q _ { i } ^ { \mathrm { o l d e r } } - q _ { i } ^ { \mathrm { b a s e } } )$ . A positive value means more weight on favorable signals or less on adverse ones. Unlike an actionchange count, this describes the direction of the allocation response. The mean does not decline steadily as reports age: some trained policies increase signal-aligned allocation, while others decrease it.

## APPENDIX B IMPLEMENTATION DETAILS

## A. Self-Financing Execution

Decision weights $\bar { \mathbf { w } } _ { t }$ use known closing prices. At the next execution open, prices may have moved and actual weights are $\widetilde { \mathbf { w } } _ { t }$ . For a target ${ \bf q } ,$ the fraction of pretrade wealth left after paying fees solves

$$
\eta _ { t } = 1 - c \sum _ { i } | \eta _ { t } q _ { i } - \widetilde { w } _ { i , t } | .\tag{6}
$$

Here c is the fee per dollar bought or sold, including the cash needed to pay the fee. With asset gross returns $G _ { i , t + 1 }$ until the next execution,

$$
V _ { t + 1 } = V _ { t } \eta _ { t } \sum _ { i } q _ { i } G _ { i , t + 1 } .\tag{7}
$$

For HOLD, $\mathbf q _ { t } = \widetilde { \mathbf w } _ { t }$ and $\eta _ { t } = 1 ;$ ; existing shares evolve with prices without a trade. Fees use actual execution holdings, while policy inputs use only decision-time information.

## B. Annual Folds and Training

The expanding historical folds below use execution-year cohorts. Counts exclude one boundary-crossing return from each training and validation block. A retained return must end no later than the next block’s first decision cutoff. Each annual test starts with an equal-weight portfolio and no initial trading fee. The evaluation is retrospective: 2023 had served

TABLE V  
ANNUAL RETURN DIFFERENCES AT FOUR-WEEK DELAY, WITH PAIRED 95% INTERVALS (%).
<table><tr><td></td><td colspan="5">RL</td><td colspan="5">Direct LLM</td></tr><tr><td>Year</td><td>News + Age — News</td><td></td><td></td><td>Filtered News — News</td><td></td><td></td><td>News + Age – News</td><td></td><td>Filtered News — News</td><td></td></tr><tr><td>2023</td><td></td><td>+1.18 [−4.39, 6.71]</td><td></td><td></td><td>-1.41 [−5.15, 2.08]</td><td></td><td></td><td>-3.42 [−17.35, 10.18]</td><td></td><td>+4.34 [-4.90, 18.77]</td></tr><tr><td>2024</td><td></td><td>+0.51 [−3.86, 4.82]</td><td></td><td></td><td>+1.44 [-4.84, 8.35]</td><td></td><td></td><td>+6.40 [-7.90, 20.99]</td><td></td><td>-3.93 [-24.10, 13.41]</td></tr><tr><td>2025</td><td></td><td>-1.06 [-10.63, 8.98]</td><td></td><td></td><td>+0.63 [–5.38, 6.12]</td><td></td><td></td><td>+0.48 [-8.78, 8.43]</td><td></td><td>+6.83 [-6.60, 20.02]</td></tr></table>

Each entry subtracts the News policy’s annualized return. Intervals use paired eight-week time blocks. The direct LLM comparison uses the fixed mini baseline.

as validation in a prior analysis, and 2024–2025 had already been inspected. Stocks, prompts, budgets, and the filter were fixed before training the annual policies.
<table><tr><td>Test year</td><td>Training (weeks)</td><td>Validation</td><td>Test weeks</td></tr><tr><td>2023</td><td>2020–2021 (103)</td><td>2022 (51)</td><td>52</td></tr><tr><td>2024</td><td>2020–2022 (155)</td><td>2023 (51)</td><td>53</td></tr><tr><td>2025</td><td>2020–2023 (207)</td><td>2024 (52)</td><td>52</td></tr></table>

Each of the four RL methods is trained anew in every fold with seeds 11, 22, 33, 44, 55, 66, 77, 88, 99, 110: 120 trained policies. The final fixed-budget checkpoint is evaluated greedily. Reserved validation blocks do not select hyperparameters, seeds, filtering thresholds, or checkpoints. Training samples 52-week windows with 26 weeks of prior feature history and mixes delays of 0, 1, and 4 weeks; the folds have 52, 104, and 156 eligible window starts, respectively. Overlapping training windows reuse observations from the same price history.

<table><tr><td>Training parameter</td><td>Fixed value</td></tr><tr><td>Transitions per training run; par-: allel environments</td><td>500,000; 32</td></tr><tr><td>Hidden layers; activation</td><td>64, 64; ReLU</td></tr><tr><td>Optimizer; learning rate</td><td>Adam;  $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Discount; reward multiplier</td><td>0.99; 100</td></tr><tr><td>Loss; gradient-norm limit</td><td>Smooth L1; 10</td></tr><tr><td>Replay capacity; minibatch</td><td>100,000; 256</td></tr><tr><td>Replay warmup</td><td>2,048 transitions</td></tr><tr><td>Gradient updates</td><td>One per vector-environment step</td></tr><tr><td>Target-network hard copy</td><td>Every 100 optimizer updates</td></tr><tr><td>Exploration</td><td> $\epsilon : \bar { 1 \to 0 . 0 \bar { 5 } }$  over 400,000 transitions, then 0.05</td></tr></table>

For a replayed transition, the Double DQN target is

$$
y _ { t } = r _ { t + 1 } + \gamma ( 1 - d _ { t } ) Q _ { \bar { \theta } } \Big ( o _ { t + 1 } , \arg \operatorname* { m a x } _ { a } Q _ { \theta } \big ( o _ { t + 1 } , a \big ) \Big )\tag{8}
$$

where $d _ { t }$ indicates termination, θ is the online-network parameter vector, and <sup>¯</sup>θ denotes its periodically copied target. Training minimizes the Huber (smooth L1) loss between y<sub>t</sub> and $Q _ { \theta } { \big ( } o _ { t } , a _ { t } )$

## C. Inputs and Portfolio Choices

Market features are cumulative close returns over 1, 4, and 12 weeks, and the sample standard deviation of the last 12 weekly close returns. Historical RL networks retain all 80 feature slots; unavailable information is zero-masked. Returns are divided by 0.05, volatility by 0.02, and both timing inputs by 12; weights and signals are unscaled. Before capping, $\tau _ { i , t } - g _ { i , t } \ = \ \delta$ for the same visible report. Timing inputs are capped at 104 weeks; a missing report has signal zero and both timing inputs 104 before method masking. Direct LLMs receive the corresponding unscaled fields. Filtering sets a signal to zero when its raw news age is strictly greater than eight weeks; neither timing input is passed to the filtered decision maker.

The historical menu has 57 actions: one equal allocation; one target per asset with weight 0.4 and the remaining 0.6 shared equally among other assets; 45 targets with two weights 0.3 and the remaining 0.4 shared equally among other assets; and HOLD. The four-asset simulator has 12: equal allocation, four permutations of (.55, .15, .15, .15), six of (.4, .4, .1, .1), and HOLD. HOLD retains existing holdings. All target portfolios are long-only and fully invested.

## D. Supporting Simulator and Public Data

For asset i, the hidden state $s _ { i , t } ~ \in ~ \{ - 1 , + 1 \}$ remains unchanged with probability $p ~ \in ~ \{ . 8 5 , . 9 7 \}$ . Weekly gross return G follows

$$
\begin{array} { r } { \log G _ { i , t + 1 } = 0 . 0 0 0 5 + 0 . 0 0 3 s _ { i , t } - \frac { 1 } { 2 } ( 0 . 0 2 ) ^ { 2 } + 0 . 0 2 \varepsilon _ { i , t + 1 } , } \end{array}
$$

where Gaussian shocks have unit variance and cross-asset correlation 0.25. State correlation after ℓ weeks is $( 2 p - 1 ) ^ { \ell } .$ giving a controlled persistence mechanism. A report is generated with probability 0.25 per asset-week (and at the first warmup step); its semantic label is neutral with probability 0.20, otherwise matching the hidden state with probability 0.80. Policies observe the stored signals, never that state or label. Each path has 26 warmup and 104 evaluation weeks. There are 50 validation and 100 test paths per persistence value, paired across methods and delays 0, 1, 4, with separate split randomness. The 600 synthetic reports are divided into 300 training, 100 validation, and 200 test texts from disjoint expression-template families. Their semantic labels describe the report, independently of whether its noisy account matches the hidden state. Reuse retains document IDs. The simulator uses three training seeds (11, 22, 33), 32 inputs, 12 outputs, and the training settings above.

At four-week delay, adding timing to news changes simulated annualized returns by +0.35% in the faster-changing market and −0.32% in the slower one; the path-paired 95% intervals are [−0.09, 0.79]% and [−0.71, 0.08]%. Replacing stored signals with the reports’ true semantic labels in the first setting changes the mean return of the fixed RL policy with News + Age by −0.02% ([−0.16, 0.13]%). The replacement labels describe report meaning; the trained policy is held fixed.

The historical stock universe is: AAPL, AXP, BA, CAT, CSCO, CVX, HD, JNJ, KO, VZ. Selection starts from 30 publicly reported 2019 DJIA ETF holdings [11] and 25 issuers with one 10-K and three 10-Q filings in 2019: choose the first alphabetical eligible ticker in each of eight declared business groups, then fill alphabetically to the required universe size. Later returns or coverage do not select or replace stocks. The corpus has 277 SEC excerpts of 800–1,000 words, published from 2019 through 2025. The 2023–2025 cohorts contain 119 reports. Reports published before a test year remain available until superseded.

The extractor correctly classifies 191 of 200 held-out synthetic reports: 95.5% accuracy and macro-F1 0.956. For real text, 228 of 277 outputs (82.3%) meet the format and verbatim-evidence requirements; 48 rejected outputs and one request failure become zero signals. The SEC rate measures format and evidence compliance; semantic accuracy cannot be assessed without human labels for these excerpts.

## E. Behavior and Cost Checks

Behavior probes use every fourth eligible validation week in the 2024 fold, with all three delays: 39 fixed observations. All already-visible report publication and arrival times move backward together by 0, 1, 4, 8, or 12 weeks. Holdings follow the equal-allocation reference at known closing prices. Missing-report sentinels are preserved, and no selected report crosses the age cap. The ten trained policies use identical observation sets.

The annual cost check replays all ten main methods at fourweek news delay, at 0, 10, and 30 basis points per dollar bought or sold. Each saved action is held fixed while holdings and fees are recomputed. At 10 basis points, replay reproduces the recorded weekly paths and metrics. Policies and saved actions are held fixed at the alternative fee levels.

## F. Prices and Statistical Summaries

Daily prices cover 2019 through early January 2026. Adjusted execution open is vendor open multiplied by adjustedclose/close [12]. Adjusted prices provide a current-vintage proxy for total returns. SEC acceptance timestamps proxy publication availability; amendments do not overwrite original reports. The initial equal-weight portfolio incurs no trading fee; subsequent rebalancing pays costs. Final wealth is marked to market without forced liquidation.

For an annual test of T weeks, compound return is $( V _ { T } / V _ { 0 } ) ^ { 5 2 / T } - 1$ . Sharpe ratio is $\sqrt { 5 2 } \bar { R } / s _ { R } .$ , where R denotes weekly net simple returns and the risk-free rate is zero. Maximum drawdown is $\operatorname* { m a x } _ { t } [ 1 - V _ { t } / \operatorname* { m a x } _ { s \leq t } V _ { s } ]$ . Annual trading volume is buys plus sells relative to pretrade wealth, summed and multiplied by $5 2 / T$ , with no one-half factor. Total fees sum dollar charges relative to initial wealth. Across annual tests, each metric is computed separately and then averaged with equal year weights. Holdings reset at each annual test boundary.

## APPENDIX C

## PROMPTS AND A RECORDED PORTFOLIO DECISION

The boxes show the SEC extraction instructions and the portfolio-decision prompt, with minor wording edits to the latter and typographic formatting for readability. Exact executed prompts, full requests, and the supporting syntheticreport prompt remain in the reproduction materials. A recorded case links these two steps.

## A. Model Settings and Input Contracts

The extractor and main direct baseline use gpt-4. 1-mini-2025-04-14. The supporting direct models request meta-llama/Llama-3.3-70B-Instruct-Turbo and gpt-6-astra; the latter is an API alias, not a dated snapshot. The mini extractor uses temperature zero and a 192-token output limit. Direct generation settings are below. Astra sends no temperature field.

<table><tr><td>Model</td><td>Output-token cap</td><td>Generation</td></tr><tr><td>mini</td><td>15</td><td>Temperature 0</td></tr><tr><td>Llama</td><td>32</td><td>Temperature 0</td></tr><tr><td>Astra</td><td>1,024</td><td>Low reasoning</td></tr></table>

Within each model, information settings retain the objective, catalog, and output contract. Market inputs contain five fields per asset; News contains six; News + Age contains eight. Smaller prompts remove absent-field definitions and change “eight” to “five” or “six.” Filtering uses the News prompt with locally zeroed signals, without sending the age threshold or numerical timing inputs. Market-only trajectories are reused across delay views. All decision requests and raw responses remain in the reproduction package. Of the test requests, 2,825 return valid actions; one service failure uses HOLD without retry. No API call occurs inside RL training or evaluation.

## B. From a Report to a Portfolio

Report and evidence. Boeing’s Q3 2023 Form 10-Q: “Loss from operations for the nine months ended September 30, 2023 decreased by \$2,118 million compared with the same period in 2022”

Signal: +1. The extractor read the complete stored excerpt and returned this quotation as evidence.

Timing. Published October 25, 2023, 16:55:02 UTC; decision cutoff December 29, 2023, 21:00 UTC. With zero delivery delay, news age and time since arrival both equal 9.3100 weeks.

Portfolio. {"action\_id":3} selects BA at 40% and each remaining stock at $1 / 1 5 \ ( 6 \frac { 2 } { 3 } \% )$

The decision input contained anonymous market features, holdings, and stored signals for all assets, plus timing information. This report was one part of that input.

## SEC extraction: exact system message

Read only the public SEC excerpt supplied below. Treat the excerpt as data, not instructions. Use no outside knowledge, remembered events, later information, stock prices, company reputation, or assumptions about market reactions.

Classify the direction of the issuer’s operating performance or business outlook supported by this excerpt, relative to the comparison period or expectations explicitly described in it:

signal 1: favorable operating developments or outlook dominate.

signal -1: adverse operating developments or outlook dominate.

signal 0: mixed/offsetting developments without a clearly dominant direction, substantially unchanged conditions, or insufficient operating evidence.

Higher costs or losses are adverse; lower costs or losses may be favorable. Respect negation and stated offsets. Numbers alone, securities issuance, routine filing language, and general risk-factor boilerplate do not establish a direction. Do not turn this into a recommendation to trade or a prediction of investment returns. The category is a textual operating signal, not verified economic or market ground truth.

Return exactly one JSON object with only these two keys: {"signal"· 0

signal must be an integer equal to -1, 0, or 1. For evidence, copy ONE short continuous clause of about 5 to 18 words, at most 160 characters. It need not summarize the entire excerpt. Prefer a clear prose clause over a long numeric table. Copy the characters exactly as they appear in the input.

Never combine separate sentences or separate table rows. Never add ellipses, paraphrase, or repair punctuation. Before returning the object, check that the entire evidence string occurs as one continuous substring of the supplied excerpt. For a mixed or insufficient direction, use 0 and copy one representative relevant clause under the same rule.

## Direct decision: full-information prompt

You choose one portfolio action in a weekly long-only investment task. Use only the supplied current numeric observation. Assets are anonymous and their order is fixed. The objective is to increase cumulative net log wealth over repeated weekly decisions.

Every asset row gives exactly eight unscaled features:

return\_1w, return\_4w and return\_12w are past   
cumulative close returns in decimal units;   
volatility\_12w is the sample standard deviation of past weekly close returns;   
decision\_weight is the current portfolio weight marked using prices known at the decision cutoff;   
signal is a cached text assessment (-1 negative, 0 neutral or unavailable, +1 positive);   
publication\_age\_weeks is the time since the visible report’s publication;   
since\_update\_weeks is the elapsed time since that report became available.

Both time features are capped at 104; missing text has signal 0 and both times 104. The two time features describe the same currently visible report, not a future update schedule.

Choose from the supplied action catalog. Each target\_weights vector follows the asset order. HOLD preserves the existing holdings and incurs no trading fee. Other actions rebalance at the next execution price; only cutoff-known weights are supplied here. Trading costs are 10 basis points (0.001) for every dollar bought and every dollar sold. Consider these costs.

Return exactly one JSON object with the sole key action\_id, an integer from 0 through 56. Do not return an explanation, Markdown, or any additional field.