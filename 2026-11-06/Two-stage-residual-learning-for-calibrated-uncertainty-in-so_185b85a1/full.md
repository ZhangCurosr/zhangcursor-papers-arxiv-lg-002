# Two-stage residual learning for calibrated uncertainty in solar irradiance forecasts

RR Flanagan<sup>1</sup>, E Kasha<sup>1</sup>, PW Talbot<sup>2</sup>, AS Epiny<sup>2</sup>, MR Deinert<sup>1\*</sup>

<sup>1</sup> Department of Mechanical Engineering, Colorado School of Mines

<sup>2</sup> Idaho National Laboratory

\*Corresponding author. MR Deinert, Department of Mechanical Engineering, Colorado School of Mines. mdeinert@mines.edu

Abstract. Accurate forecasting of renewable generation requires not only point predictions but also reliable estimates of the uncertainty attached to them, particularly at extended horizons. Machine learning models are increasingly used to predict solar output, but they typically report a single value and capture the associated uncertainty poorly. The standard remedy is to train a model to predict the parameters of an assumed probability distribution directly. Here we develop an alternative two-stage method, motivated by synthetic solar signal generation, in which the forecast error is modeled separately from the expected irradiance. A first long short-term memory (LSTM) network predicts the expected global horizontal irradiance, and a second network models the distribution of the forecast error, which is added back to the mean to give both the forecast and its uncertainty. Using historical weather data from the National Solar Radiation Database and a probability integral transform evaluation, we show that explicitly modeling the error brings the predicted standard deviation markedly closer to the historical value, reducing the deviation between the two by 34.6 percent for a standalone LSTM and by 56.5 percent when the LSTM is coupled to Gaussian process regression, the most reliable of the configurations tested. Because the method models the total forecast error directly, it captures the full predictive uncertainty that matters for decisions, rather than only the model contribution that dropout-based approaches estimate.

## Introduction

As the penetration of renewable generators on electrical grids increases, their variability places growing stress on grid operations [1]. Accurate forecasts of renewable generation tell operators how much compensatory generation will be needed [2,3] and support efficient bidding in electricity markets. The field has moved steadily from point forecasts toward probabilistic forecasts that report a full distribution of outcomes [4], a shift that is now well established for solar and photovoltaic power in particular [5]. Solar and wind signals have historically been predicted with statistical methods. The autoregressive movingaverage model (ARMA) and its integrated extension (ARIMA) have shown promise for forecasting [6–8]. The inertia in these models makes them well suited to short horizons, but their accuracy falls as the horizon grows. Numerical weather prediction has also been used widely for wind and solar forecasting, though its computational cost can be significant [9].

Machine learning offers another route to predicting solar output [10,11]. A least absolute shrinkage and selection operator (LASSO) approach has been used to forecast solar power generation from five-minute weather data in Massachusetts [12]. Several reviews report that the best machine learning method often depends on the geographic area [13,14]. Others combine statistical and machine learning techniques. Xie and colleagues combined variational mode decomposition, deep belief networks, and ARMA to predict solar signals, and reported improvements over each method used alone [15]. Long short-term memory (LSTM) networks have been used for hour-ahead and day-ahead solar irradiance prediction and can outperform feedforward neural networks [16], and an LSTM trained partly on non-local data has been used to predict the next day of hourly irradiance from local weather forecasts [17]. Reviews of deep learning for solar forecasting find that hybrid models such as the convolutional LSTM can outperform a standalone LSTM, although the strongest results in those reviews came from notably dry, clear-sky locations [18].

Neural networks have also been applied to other natural signals. A discrete wavelet transform combined with an LSTM has outperformed wavelet combinations with regression and backpropagation networks for short-term wind power [19]. An LSTM has also outperformed a moving-average approach and a multilayer perceptron on mean square error in turbine SCADA data [20]. Whatever the forecasting approach, the uncertainty attached to a prediction is both essential and too often missing. This is a recognized problem for machine learning models generally [21]. One way to estimate uncertainty is to assume the modeled quantity is Gaussian and to train a network to predict the mean and standard deviation of that Gaussian for a given set of weather inputs [22], which gives both an expected value and a distribution. A network can also be trained to output the parameters of a richer conditional distribution, as in a mixture density network [23]. This family of methods has been used widely, including for the estimation of nuclear cross sections [24]. Gaussian process models provide another route to uncertainty [25]. Gaussian process regression has been used to predict hour-ahead solar panel output with attached confidence intervals [26]. For solar power, coupling an LSTM to a Gaussian process reduced the uncertainty in the result relative to an LSTM alone [27], and the same pairing has been applied to short-term wind speed [28]. Gaussian process models are, however, computationally expensive and scale poorly.

Despite these advances, current methods remain limited in their ability to capture uncertainty fully. Traditional machine learning often fails to account for both aleatoric uncertainty, which arises from irreducible randomness, and epistemic uncertainty, which reflects gaps in knowledge and is in principle reducible [21]. Most methods address aleatoric uncertainty by modeling noise in the data but rarely capture the model's own ignorance, so predictions can appear overconfident where data are sparse or unrepresentative [21]. Several approaches target the epistemic part directly. Monte Carlo dropout deactivates connections at random during both training and inference, producing an ensemble of stochastic forward passes whose spread estimates uncertainty [29]. Deep ensembles train several networks from different initializations and combine their predictions [30]. A single network can also be trained to output both the data noise and a measure of model uncertainty at once [31], or to return the parameters of an evidential distribution from one forward pass [32]. Each of these mainly captures one component, and the spread they report is a proxy rather than a direct measure of the total error.

Isolating the uncertainty in a prediction would allow the aleatoric and epistemic contributions to be measured together. This is difficult for time series forecasts of probability distributions, because there is usually no ground truth distribution to compare against, only single realized outcomes. Earlier work on synthetic solar signals showed that isolating and modeling the noise component, separately from the clearsky irradiance, improves signal generation [33]. Building on that result, we combine noise modeling with machine learning for solar forecasting and compare two approaches. In the first, an LSTM predicts the parameters of a modified Gaussian for the irradiance directly. In the second, the expected value is subtracted from the ground truth to build a dataset of forecast errors, and a second LSTM predicts the distribution of that error, which is added back to the expected value to give both the forecast and its distribution. Both approaches use historical weather data from the National Solar Radiation Database [34]. The difference is that the two-stage method gives an explicit estimate of the error distribution, whereas the single-stage method gives only an implicit one. We evaluate both with the probability integral transform [35], and we extend the comparison to configurations that couple each method to Gaussian process regression. A standalone LSTM was chosen for the core comparison so that the benefit of the two-stage method could be isolated, though hybrid architectures such as the convolutional LSTM have shown promise for solar forecasting [18].

## Methods

## Data

Ground truth data were collected from the National Solar Radiation Database (NSRDB) [34] for Austin, Texas and Seattle, Washington, together with the Department of Energy Open Energy Data Initiative. Each record contains wind speed, wind direction, solar strength, dew point, temperature, humidity, air pressure, time of day, and time of year. An additional field representing cloud cover was derived by subtracting the historical value from the calculated clear-sky maximum. The historical data are used to make hour-ahead and day-ahead predictions.

## Data processing

Records for Austin and Seattle were drawn at 30-minute resolution. Weather conditions from 24 surrounding regions, upstream and downstream of each target site, were included to provide pseudoforecast information. All fields were scaled to the interval from zero to one to improve the performance of the network.

Two fields were derived from the raw data: normalized global horizontal irradiance (GHI) and estimated cloud cover. Normalized GHI is the GHI divided by the maximum solar irradiance at the same time, and is the prediction target. Estimated cloud cover is the clear-sky maximum minus the historical GHI, normalized by the clear-sky maximum, and represents the fraction of sunlight blocked by clouds.

The calculated maximum solar irradiance is used to identify intervals when irradiance is necessarily zero. These nighttime intervals are excluded from the root mean square error, the coefficient of determination, and the error calculations, because irradiance is known to be zero at night and a network is not required to predict it.

## Parameter prediction method

The parameter prediction method uses an LSTM network to predict the parameters of a modified Gaussian, Equation (1):

$$
\begin{array} { r } { f ( t + \Delta t ) = \frac { 1 } { \sqrt { 2 \pi c } } e ^ { - \frac { ( x - a ) ^ { 2 } } { 2 b ^ { 2 } } } } \end{array}\tag{1}
$$

Here f is the probability density of the GHI at time $\mathrm { t } { + } \Delta \mathrm { t } ,$ and the network predicts the parameters a, b, and c, from which the probability of a given value of GHI(t+Δt) follows. Figure 1 (PPM) shows the process. Equation (1) has the form of a Gaussian, but the three parameters are allowed to vary independently, which makes the function more general than a standard Gaussian, whose normalization is fixed by its width. The parameters are obtained by training the network to maximize the likelihood of the observed irradiance under Equation (1), equivalently by minimizing the negative log-likelihood, Equation (2), averaged over the training data:

$$
\begin{array} { r } { L = \frac { 1 } { N } { \sum _ { i = 1 } ^ { N } } [ \frac { ( x _ { i } - a _ { i } ) ^ { 2 } } { 2 { b _ { i } } ^ { 2 } } + \frac { 1 } { 2 } l n ( 2 \pi c _ { i } ) ] } \end{array}\tag{2}
$$

## Augmented error method

The parameter prediction method gives a prediction of the uncertainty. The uncertainty can also be estimated by comparing the predicted value with the historical value. For this, the predicted value is subtracted from the historical GHI, and the difference is the error of that prediction.

The forecast errors are generated with a leave-one-year-out procedure, so that no year is predicted by a model that was trained on it. For each year in the twenty-year record from 2002 to 2021, a parameter prediction network is trained on the remaining nineteen years and used to predict that year. The expected value of each prediction is subtracted from the corresponding ground truth to give the forecast error at 30- minute resolution across the full record. Figure 1 (AEM) shows the process, where f in Equation (1) now represents the error at time t+Δt rather than the GHI.

A second LSTM is then trained on the historical weather data from 2002 to 2021 together with these errors, and is applied over the test period from 2022 to 2024 to predict an error distribution for each 30-minute interval. Adding each error distribution back to the expected GHI gives the forecast and its distribution. The test period yields approximately 16,000 predicted distributions per configuration, which are analyzed below. The specifications of each network are given in the Supplementary Information. This two-stage structure is a form of residual learning where the second network models the residual of the first-stage prediction rather than the quantity of interest directly.

![](images/710e1f202fa6fe5f5aa8d95fc56e9f3f5ace3f62bb900da0c651b39540a59712.jpg)  
Figure 1. Schematic for training the parameter prediction method (PPM) and the augmented error method (AEM). Historical ground truth data at time t, for both GHI and weather, are used to train the PPM to predict the GHI distribution at time t+Δt. The training data contain both the t and t+Δt values, which are used to generate a semi-synthetic error dataset for time t+Δt. The AEM uses this dataset to predict the GHI distribution at t+Δt. Once trained, both models require only weather and GHI values at time t to predict the GHI distribution at t+Δt.

## LSTM setup

Long short-term memory networks process sequences rather than single data points, which suits time series because earlier values can influence the prediction of later ones. An LSTM cell has three gates that manage its information: an input gate selects which information is updated in the cell, a forget gate determines which information is kept and which is discarded, and an output gate returns the cell result [36,37]. The networks were built with the Keras interface to TensorFlow [38]. The network hyperparameters, including the number of epochs, layers, neurons, batch size, and dropout rate, were selected with the Optuna optimization framework [39]. The search ranges and the selected values are reported in the Supplementary Information.

## Gaussian process regression

A combined LSTM and Gaussian process regression model was also tested with the same procedures. Gaussian process models are an established route to uncertainty estimation in machine learning [25–28]. Here the LSTM was trained to predict the GHI rather than the parameters of a distribution, and the Gaussian process supplied the distribution. The combined model was also run through the augmented error procedure, in which the LSTM again predicted an error rather than a distribution of error. As with the parameter prediction network, residuals from 2002 to 2021 were used to build this model.

## Evaluating the predicted distributions

If a distribution correctly represents the uncertainty in a prediction, then ground truth values fall within it at the frequencies the distribution implies [35]. This property is the basis of the probability integral transform, and uniformity of the transformed values is the standard diagnostic for a calibrated probabilistic forecast. We use it to compare the methods. We also summarize accuracy with the root mean square error of the point forecast, the continuous ranked probability score [40], and the empirical coverage of the central prediction intervals at the 95, 90, and 50 percent levels. We do not report the logarithmic score. It evaluates the predicted probability density at the realized value, and the modified Gaussian of Equation (1), whose parameters are independent, does not integrate to one, so its density is fixed only up to a constant. The probability integral transform and the continuous ranked probability score depend only on the cumulative distribution, in which that constant cancels, and so remain well defined.

Each predicted distribution is divided into 1000 bins of equal probability, each holding one part in a thousand of the total probability. The bin edges are found by integrating the distribution until the cumulative probability reaches each successive multiple of one part in a thousand. The ground truth value for that time step is placed in the bin into which it falls. Counts are accumulated across all predicted distributions in the test period, which gives a histogram over the 1000 probability bins. Figure 2 shows the process.

![](images/e428c6b7e6b9dac642d53764ae6f72db49d1c2d285076821be64262b62d089ab.jpg)  
Figure 2. The probability integral transform. The model produces a set of values for a, b, and c in Equation (1) at each time point. Because there is no ground truth distribution to compare against, each predicted distribution is divided into bins of equal cumulative probability, and the ground truth value is placed in the bin into which it falls. If the predicted distributions are correct, the binned values form a uniform histogram.

If the distributions are correct, the histogram is uniform, with each bin holding the total number of values divided by the number of bins. With approximately 16,000 values this is about 16 per bin. Departures from a uniform histogram indicate that the predicted distributions do not match the data.

We test each histogram against the uniform expectation with a chi-square goodness-of-fit test, Equation (3), where $\mathrm { ~ O ~ } _ { - }$ i is the count in bin i and E is the expected count, equal to the number of values divided by 1000. The test has 999 degrees of freedom, one fewer than the number of bins:

$$
\begin{array} { r } { \chi ^ { 2 } = \sum _ { i = 1 } ^ { 1 0 0 0 } \frac { ( O _ { i } - E ) ^ { 2 } } { E } , \ E = \frac { N } { 1 0 0 0 } } \end{array}\tag{3}
$$

The chi-square test is the appropriate choice here. Because every ground truth value falls into exactly one bin, the bin counts sum to the number of predicted values and their mean is fixed at about 16 for every method, regardless of calibration. The information that distinguishes a calibrated forecast from a miscalibrated one is therefore carried entirely by the spread of the counts about this value, which is exactly

the quantity the chi-square statistic measures.

## Results and discussion

Many forecasting methods, including those built on deep learning, focus on point predictions and do not report the associated probability distribution. Methods that do address uncertainty often assume a single distribution for the forecast error, usually Gaussian [22]. Forecast errors can, however, show skew, multiple peaks, and heavy tails that a single assumed distribution does not capture [28]. Capturing both the aleatoric and epistemic parts of the uncertainty is also difficult, particularly the latter [21]. The method developed here addresses these issues by making an explicit estimate of the uncertainty in a predicted irradiance and then using an LSTM to choose the parameters of a modified Gaussian that fits it.

Table 1 reports the average predicted standard deviation over a year and the average historical values for Austin and Seattle, for the parameter prediction method, the augmented error method, the LSTM with Gaussian process regression, and the LSTM with Gaussian process regression in its augmented error form, at both horizons. Nighttime intervals, where the signal is known to be zero, are excluded, since including them would inflate the apparent accuracy. The parameter prediction and augmented error methods share the same base LSTM and differ in how the forecast error is modeled. The historical standard deviation at a single time step does not represent the true value at that step and is shown only for reference. In every case the augmented error method gives an average predicted standard deviation closer to the historical value than its single-stage counterpart. Averaged across the cases, the augmented error method reduces the deviation between the predicted and historical standard deviation by 34.6 percent for the standalone LSTM and by 56.5 percent for the LSTM with Gaussian process regression. The latter is closest to the historical standard deviation in every case examined.

<table><tr><td colspan="1" rowspan="1">Method</td><td colspan="1" rowspan="1">Austin predictedstandarddeviation (W/m²)</td><td colspan="1" rowspan="1">Austin historicalstandarddeviation (W/m²)</td><td colspan="1" rowspan="1">Seattle predictedstandarddeviation (W/m²)</td><td colspan="1" rowspan="1">Seattle historicalstandarddeviation (W/m²)</td></tr><tr><td colspan="5" rowspan="1">One-hour prediction</td></tr><tr><td colspan="1" rowspan="1">Parameterprediction method</td><td colspan="1" rowspan="1">30.2</td><td colspan="1" rowspan="1">34.4</td><td colspan="1" rowspan="1">41.3</td><td colspan="1" rowspan="1">45.1</td></tr><tr><td colspan="1" rowspan="1">Augmented errormethod</td><td colspan="1" rowspan="1">36.9</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">47.6</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">LSTM with</td><td colspan="1" rowspan="1">36.2</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">47.2</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">Gaussian processregression</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">LSTM withGaussian processregression,augmented error</td><td colspan="1" rowspan="1">35.4</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">44.3</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">24-hour prediction</td><td colspan="2" rowspan="1"></td><td colspan="2" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">Parameterprediction method</td><td colspan="1" rowspan="1">29.4</td><td colspan="1" rowspan="1">34.4</td><td colspan="1" rowspan="1">39.9</td><td colspan="1" rowspan="1">45.1</td></tr><tr><td colspan="1" rowspan="1">Augmented errormethod</td><td colspan="1" rowspan="1">31.2</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">41.4</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">LSTM withGaussian processregression</td><td colspan="1" rowspan="1">37.6</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">49.5</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">LSTM withGaussian processregression,augmented error</td><td colspan="1" rowspan="1">32.4</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">43.9</td><td colspan="1" rowspan="1"></td></tr></table>

Table 1. Average predicted standard deviation over a year and average historical standard deviation for Austin and Seattle, for the parameter prediction method, the augmented error method, the LSTM with Gaussian process regression, and its augmented error form, at the one-hour and 24-hour horizons. The augmented error method is closer to the historical standard deviation in every case. The historical standard deviation is shown once per horizon as a reference and does not represent the true standard deviation at a single time step.

Figure 3 shows the probability integral transform histograms for the hour-ahead GHI distributions from the parameter prediction and augmented error methods. The top two panels are for Austin, with the parameter prediction method on the left and the augmented error method on the right, and the bottom two panels are the same for Seattle. A perfectly calibrated set of distributions would place about 16 values in each bin with no systematic departure, and the further a histogram departs from this, the less accurate the predicted distributions. Figure 4 gives the corresponding results for day-ahead predictions. Figures 5 and 6 show the results for the LSTM with Gaussian process regression and its augmented error form. The augmented error variants give visibly flatter histograms, and the augmented error form of the LSTM with Gaussian process regression is the flattest of the four configurations.

![](images/521a7917e0f5878d60e9ff6b1e8cf2024dcda8444a4735d7c610b49c4cac8294.jpg)

![](images/5c5968211259121b1abac6fe046520382b97ac88d3f3b6397d7522ed5b730baa.jpg)

![](images/a0d7de52e61249e6d8df0b59a3d1ec17f75bfc5e2226076f0af84f48fcf199c3.jpg)

![](images/8742ca39324755506651e8c38668b76ac46b9ef328b3f5e1f5f10b7ea8884705.jpg)  
Figure 3. Probability integral transform histograms of the hour-ahead GHI distributions from the parameter prediction and augmented error methods. The top panels are for Austin, Texas, with the parameter prediction method on the left and the augmented error method on the right. The bottom panels are the same for Seattle, Washington.

![](images/03fc4cdcd98953ba77b6cde269c2e96c3740c9029c53ffe9c5f46e2c52cfe895.jpg)

![](images/e55caaa786bc8e6873dd4ee54576d19297971d17fc84a1d4b311a83b13185490.jpg)

![](images/deb52ff5852342d8451a454a1f581251ee677aef1c0bca78e6494b139ff82b52.jpg)

![](images/d9c475eaf4f8dc6afe1971a0d3e7c917efd95963f0be91a329ba0f3e7de4cf41.jpg)  
Figure 4. Probability integral transform histograms of the day-ahead GHI distributions from the parameter prediction and augmented error methods. The top panels are for Austin, Texas, with the parameter prediction method on the left and the augmented error method on the right. The bottom panels are the same for Seattle, Washington.

![](images/15d3431ecf92b09c314e26b4aa437cd1de21ee52bf789465a73ad421d4bc745b.jpg)

![](images/4e6f19c482c16c733b9c4e06a8b125288275706534e1eb810d113b934392e297.jpg)

![](images/cb445baa821454ba0380ae645823bb22b5905bcd80f6913a725460a9719eeca4.jpg)

![](images/0270aaf6b246a2ec31305e1ab62aa316c48e8a651ada1be21dcc33dcb603ff5a.jpg)  
Figure 5. Probability integral transform histograms of the hour-ahead GHI distributions from the LSTM with Gaussian process regression. The top panels are for Austin, Texas, with the standalone model on the left and its augmented error form on the right. The bottom panels are the same for Seattle, Washington.

![](images/e12d65c099495bfeae4432f9f0774040dbf8f10079217f8c395c2c79cbad641f.jpg)  
Figure 6. Probability integral transform histograms of the day-ahead GHI distributions from the LSTM with Gaussian process regression. The top panels are for Austin, Texas, with the standalone model on the left and its augmented error form on the right. The bottom panels are the same for Seattle, Washington.

We compared each histogram with the uniform expectation using a chi-square goodness-of-fit test with 999 degrees of freedom. The results are given in Table 2. The histograms from the parameter prediction method and from the standalone LSTM with Gaussian process regression differ significantly from the uniform expectation, with p-values below 0.05, so these methods do not reproduce the expected distribution. Every configuration that uses the augmented error method has a p-value above 0.05, and so is not distinguishable from the uniform expectation. Taken together with Table 1, the augmented error form of the LSTM with Gaussian process regression is the most reliable configuration: it passes the calibration test and its predicted standard deviation is closest to the historical standard deviation in every case.

<table><tr><td colspan="1" rowspan="1">Method</td><td colspan="1" rowspan="1">Austin</td><td colspan="1" rowspan="1">Seattle</td></tr><tr><td colspan="1" rowspan="1">Parameter prediction method, 1h</td><td colspan="1" rowspan="1"> $1 . 1 \times 1 0 ^ { - 8 }$ </td><td colspan="1" rowspan="1"> $\overline { { 3 . 0 { \times } 1 0 ^ { - 9 } } }$ </td></tr><tr><td colspan="1" rowspan="1">Parameter prediction method, 24h</td><td colspan="1" rowspan="1"> $4 . 6 { \times } 1 0 ^ { - 1 1 }$ </td><td colspan="1" rowspan="1"> $1 . 5 { \times } 1 0 ^ { - 1 3 }$ </td></tr><tr><td colspan="1" rowspan="1">Augmented error method, 1 h</td><td colspan="1" rowspan="1">0.47</td><td colspan="1" rowspan="1">0.38</td></tr><tr><td colspan="1" rowspan="1">Augmented error method, 24 h</td><td colspan="1" rowspan="1">0.18</td><td colspan="1" rowspan="1">0.47</td></tr><tr><td colspan="1" rowspan="1">LSTM with Gaussian processregression, 1 h</td><td colspan="1" rowspan="1"> $2 . 2 \times 1 0 ^ { - 3 }$ </td><td colspan="1" rowspan="1"> $2 . 0 { \times } 1 0 ^ { - 4 }$ </td></tr><tr><td colspan="1" rowspan="1">LSTM with Gaussian processregression, 24 h</td><td colspan="1" rowspan="1"> $9 . 0 { \times } 1 0 ^ { - 4 }$ </td><td colspan="1" rowspan="1">1.1×10⁻⁴</td></tr><tr><td colspan="1" rowspan="1">LSTM with Gaussian processregression, augmented error, 1 h</td><td colspan="1" rowspan="1">0.39</td><td colspan="1" rowspan="1">0.25</td></tr><tr><td colspan="1" rowspan="1">LSTM with Gaussian processregression, augmented error, 24h</td><td colspan="1" rowspan="1">0.45</td><td colspan="1" rowspan="1">0.21</td></tr></table>

Table 2. Chi-square goodness-of-fit p-values comparing each probability integral transform histogram with the uniform expectation, for each city, method, and horizon, with 999 degrees of freedom. A p-value at or below 0.05 indicates a histogram that differs significantly from the uniform expectation.

The probability integral transform evaluates the entire predicted distribution against the realized values at every probability level. A method that passes it has reproduced the conditional distribution of the outcome, not merely a summary statistic of it. A flat histogram is, strictly, a necessary rather than a sufficient condition, because opposing miscalibrations could in principle cancel, but such cancellation would have to hold across the entire record at every probability level, which a single consistently trained model does not produce and which the clear rejection of the parameter prediction and standalone Gaussian process configurations shows the test still detects. A single summary score such as the continuous ranked probability score compresses calibration and sharpness into one value, and a model can attain a competitive score while remaining miscalibrated [40]. Because operational decisions depend on the probability assigned to each possible outcome, we treat the probability integral transform as the primary measure of calibration and report the summary scores below as a complement.

Calibration describes the shape of the predicted distribution. The accuracy of the forecasts can also be summarized with scores that compare the distribution, or its center, with the realized values. Table 3 reports the root mean square error of the point forecast and the continuous ranked probability score, and Table 4 reports the empirical coverage of the central prediction intervals at the 95, 90, and 50 percent levels, for both horizons and both sites.

The augmented error method lowers both the root mean square error and the continuous ranked probability score relative to its base in every case. Averaged over the cases, the continuous ranked probability score falls by about 33 percent for the standalone LSTM and by about 22 percent for the LSTM with Gaussian process regression. The augmented error form of the LSTM with Gaussian process regression gives the lowest root mean square error and the lowest continuous ranked probability score in most cases, which agrees with its being the best calibrated. As an external baseline we also evaluate Monte Carlo dropout applied to the base LSTM, with the predictive distribution taken from the spread of 500 stochastic forward passes at inference. It improves on the single-stage parameter prediction method but does not reach either augmented error method on the root mean square error or the continuous ranked probability score in any case, which leaves the augmented error form of the LSTM with Gaussian process regression the best overall. The central prediction intervals are close to their nominal levels for the four methods and the Monte Carlo dropout baseline alike, and the configurations that use Gaussian process regression are marginally closer to the nominal 95 percent coverage than the others. Because the two methods share the same base LSTM, the improvement in root mean square error is attributable to second stage. Here, the predicted error distribution can have a nonzero expected value, which acts as an effective bias correction to the first-stage point forecast.

<table><tr><td colspan="1" rowspan="1">Method</td><td colspan="1" rowspan="1">Austin RMSE(W/m²)</td><td colspan="1" rowspan="1">Austin CRPS(W/m²)</td><td colspan="1" rowspan="1">Seattle RMSE(W/m²)</td><td colspan="1" rowspan="1">Seattle CRPS(W/m²)</td></tr><tr><td colspan="5" rowspan="1">One-hour prediction</td></tr><tr><td colspan="1" rowspan="1">Parameterprediction method</td><td colspan="1" rowspan="1">33.6</td><td colspan="1" rowspan="1">16.9</td><td colspan="1" rowspan="1">29.0</td><td colspan="1" rowspan="1">14.2</td></tr><tr><td colspan="1" rowspan="1">Augmented errormethod</td><td colspan="1" rowspan="1">22.8</td><td colspan="1" rowspan="1">11.5</td><td colspan="1" rowspan="1">20.7</td><td colspan="1" rowspan="1">10.1</td></tr><tr><td colspan="1" rowspan="1">LSTM withGaussian processregression</td><td colspan="1" rowspan="1">27.8</td><td colspan="1" rowspan="1">14.0</td><td colspan="1" rowspan="1">24.2</td><td colspan="1" rowspan="1">11.8</td></tr><tr><td colspan="1" rowspan="1">LSTM withGaussian processregression,augmented error</td><td colspan="1" rowspan="1">21.8</td><td colspan="1" rowspan="1">11.0</td><td colspan="1" rowspan="1">18.8</td><td colspan="1" rowspan="1">9.2</td></tr><tr><td colspan="1" rowspan="1">Monte Carlodropout</td><td colspan="1" rowspan="1">28.3</td><td colspan="1" rowspan="1">14.3</td><td colspan="1" rowspan="1">25.9</td><td colspan="1" rowspan="1">12.6</td></tr><tr><td colspan="5" rowspan="1">24-hour prediction</td></tr><tr><td colspan="1" rowspan="1">Parameterprediction method</td><td colspan="1" rowspan="1">36.5</td><td colspan="1" rowspan="1">18.3</td><td colspan="1" rowspan="1">33.0</td><td colspan="1" rowspan="1">16.0</td></tr><tr><td colspan="1" rowspan="1">Augmented errormethod</td><td colspan="1" rowspan="1">23.4</td><td colspan="1" rowspan="1">11.8</td><td colspan="1" rowspan="1">21.5</td><td colspan="1" rowspan="1">10.4</td></tr><tr><td colspan="1" rowspan="1">LSTM withGaussian processregression</td><td colspan="1" rowspan="1">28.4</td><td colspan="1" rowspan="1">14.3</td><td colspan="1" rowspan="1">25.4</td><td colspan="1" rowspan="1">12.3</td></tr><tr><td colspan="1" rowspan="1">LSTM withGaussian processregression,augmented error</td><td colspan="1" rowspan="1">23.7</td><td colspan="1" rowspan="1">12.0</td><td colspan="1" rowspan="1">18.0</td><td colspan="1" rowspan="1">8.8</td></tr><tr><td colspan="1" rowspan="1">Monte Carlodropout</td><td colspan="1" rowspan="1">33.0</td><td colspan="1" rowspan="1">16.6</td><td colspan="1" rowspan="1">29.1</td><td colspan="1" rowspan="1">14.1</td></tr></table>

Table 3. Root mean square error and continuous ranked probability score, both in watts per square meter, for the four methods and a Monte Carlo dropout baseline at the one-hour and 24-hour horizons for Austin and Seattle. Lower values are better. The augmented error method improves both over its base in every case, and both augmented methods improve on the Monte Carlo dropout baseline.

<table><tr><td colspan="1" rowspan="1">Method</td><td colspan="1" rowspan="1">Austin 95%</td><td colspan="1" rowspan="1">Austin 90%</td><td colspan="1" rowspan="1">Austin 50%</td><td colspan="1" rowspan="1">Seattle95%</td><td colspan="1" rowspan="1">Seattle90%</td><td colspan="1" rowspan="1">Seattle50%</td></tr><tr><td colspan="7" rowspan="1">One-hour prediction</td></tr><tr><td colspan="1" rowspan="1">Parameterpredictionmethod</td><td colspan="1" rowspan="1">92.6</td><td colspan="1" rowspan="1">86.6</td><td colspan="1" rowspan="1">46.4</td><td colspan="1" rowspan="1">92.7</td><td colspan="1" rowspan="1">86.7</td><td colspan="1" rowspan="1">46.2</td></tr><tr><td colspan="1" rowspan="1">Augmentederrormethod</td><td colspan="1" rowspan="1">92.5</td><td colspan="1" rowspan="1">86.4</td><td colspan="1" rowspan="1">46.1</td><td colspan="1" rowspan="1">91.7</td><td colspan="1" rowspan="1">85.2</td><td colspan="1" rowspan="1">44.7</td></tr><tr><td colspan="1" rowspan="1">LSTM withGaussianprocessregression</td><td colspan="1" rowspan="1">93.5</td><td colspan="1" rowspan="1">87.7</td><td colspan="1" rowspan="1">47.1</td><td colspan="1" rowspan="1">93.2</td><td colspan="1" rowspan="1">87.5</td><td colspan="1" rowspan="1">46.9</td></tr><tr><td colspan="1" rowspan="1">LSTM withGaussianprocess</td><td colspan="1" rowspan="1">93.1</td><td colspan="1" rowspan="1">87.4</td><td colspan="1" rowspan="1">46.9</td><td colspan="1" rowspan="1">93.2</td><td colspan="1" rowspan="1">87.4</td><td colspan="1" rowspan="1">46.9</td></tr><tr><td colspan="1" rowspan="1">regression,augmentederror</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">Monte Carlodropout</td><td colspan="1" rowspan="1">92.6</td><td colspan="1" rowspan="1">86.6</td><td colspan="1" rowspan="1">46.4</td><td colspan="1" rowspan="1">91.7</td><td colspan="1" rowspan="1">85.3</td><td colspan="1" rowspan="1">45.5</td></tr><tr><td colspan="7" rowspan="1">24-hour prediction</td></tr><tr><td colspan="1" rowspan="1">Parameterpredictionmethod</td><td colspan="1" rowspan="1">90.6</td><td colspan="1" rowspan="1">84.5</td><td colspan="1" rowspan="1">44.2</td><td colspan="1" rowspan="1">92.1</td><td colspan="1" rowspan="1">86.1</td><td colspan="1" rowspan="1">45.6</td></tr><tr><td colspan="1" rowspan="1">Augmentederrormethod</td><td colspan="1" rowspan="1">91.9</td><td colspan="1" rowspan="1">85.6</td><td colspan="1" rowspan="1">45.2</td><td colspan="1" rowspan="1">90.8</td><td colspan="1" rowspan="1">84.0</td><td colspan="1" rowspan="1">43.8</td></tr><tr><td colspan="1" rowspan="1">LSTM withGaussianprocessregression</td><td colspan="1" rowspan="1">92.2</td><td colspan="1" rowspan="1">86.3</td><td colspan="1" rowspan="1">45.7</td><td colspan="1" rowspan="1">92.2</td><td colspan="1" rowspan="1">86.3</td><td colspan="1" rowspan="1">45.8</td></tr><tr><td colspan="1" rowspan="1">LSTM withGaussianprocessregression,augmentederror</td><td colspan="1" rowspan="1">93.5</td><td colspan="1" rowspan="1">87.9</td><td colspan="1" rowspan="1">47.1</td><td colspan="1" rowspan="1">94.0</td><td colspan="1" rowspan="1">88.5</td><td colspan="1" rowspan="1">47.9</td></tr><tr><td colspan="1" rowspan="1">Monte Carlodropout</td><td colspan="1" rowspan="1">89.1</td><td colspan="1" rowspan="1">82.5</td><td colspan="1" rowspan="1">42.2</td><td colspan="1" rowspan="1">90.6</td><td colspan="1" rowspan="1">83.8</td><td colspan="1" rowspan="1">44.0</td></tr></table>

Table 4. Empirical coverage, in percent, of the central prediction intervals at the 95, 90, and 50 percent nominal levels, for the four methods and a Monte Carlo dropout baseline at the one-hour and 24-hour horizons for Austin and Seattle. Values close to the nominal level indicate well-sized intervals.

There are several limitations to the current work. An LSTM can in principle shift historical data to reproduce a signal. To ensure that a predicted value is not a copy of the historical signal, leading or lagging, a correlation test was run comparing the predicted signal against the historical signal shifted by up to 100 time steps in each direction, and any near-perfect match was inspected visually to confirm that no time shifting had occurred. Many physical weather variables could not be included here because of the limited public data, among them the convective available potential energy and atmospheric temperature gradients that underlie deterministic weather models. Including such variables in the training set could improve the results. Finally, the comparison here uses a standalone LSTM by design, so that the benefit of the two-stage method can be isolated. Other probabilistic approaches, such as boosting-based distributional regression [41], could be evaluated with the same probability integral transform in future work.

## Conclusions

This work introduces a two-stage augmented error method for modeling the probability distributions of solar irradiance forecasts and evaluates it against a conventional single-stage approach and against configurations coupled to Gaussian process regression. The method addresses several persistent gaps in machine learning forecasting. Conventional models, including LSTMs, tend to report point forecasts without the associated distribution, which limits their use where decisions depend on risk, such as grid operation and market bidding. Dedicating a second network to the forecast error provides that distribution explicitly. Where earlier work assumes a single fixed error distribution and so misses skew, heavy tails, and multiple peaks, the augmented error method fits a modified Gaussian to the explicitly modeled error, and could be extended to other distributions. By modeling the total forecast error directly, it captures the full predictive uncertainty that decisions depend on. Methods built on Monte Carlo dropout estimate only the model contribution, and methods that assume a single noise distribution estimate only the data contribution. The augmented error method requires neither component to be isolated, because the total error is what a forecast user needs.

Evaluating uncertainty is itself difficult, because there is no ground truth distribution to compare against. The probability integral transform addresses this by testing whether ground truth values fall within the predicted distributions at the expected frequencies. Under a chi-square goodness-of-fit test, the augmented error distributions are not distinguishable from the uniform expectation, while the single-stage parameter prediction and the standalone Gaussian process configurations are. Coupling the augmented error method to Gaussian process regression gives the best result overall, since it is the only configuration whose predicted standard deviation is closest to the historical standard deviation in every case while also passing the calibration test.

Across the cases studied, the augmented error method reduces the deviation between the predicted and historical standard deviation by 34.6 percent for the standalone LSTM and by 56.5 percent for the LSTM with Gaussian process regression, and it produces distributions that better reflect the range of solar irradiance outcomes. By making the uncertainty explicit and testing it directly, the approach improves the reliability of machine learning forecasts for renewable energy integration and for other settings where the distribution of outcomes matters.

## Data availability

The training and test data will be made available upon request.

## References

1. Anees AS. Grid integration of renewable energy sources: challenges, issues and possible solutions.   
2012 IEEE 5th India International Conference on Power Electronics (IICPE) 2012:1-6.

2. Ciarreta A, Pizarro-Irizar C, Zarraga A. Renewable energy regulation and structural breaks: an empirical analysis of Spanish electricity price volatility. Energy Economics 2020;88:104749. doi:10.1016/j.eneco.2020.104749.

3. Schmietendorf K, Peinke J, Kamps O. The impact of turbulent renewable energy production on power grid stability and quality. European Physical Journal B 2017;90:222. doi:10.1140/epjb/e2017-80352-8.

4. Hong T, Pinson P, Fan S, Zareipour H, Troccoli A, Hyndman RJ. Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond. International Journal of Forecasting 2016;32:896-913. doi:10.1016/j.ijforecast.2016.02.001.

5. van der Meer DW, Widen J, Munkhammar J. Review on probabilistic forecasting of photovoltaic power production and electricity consumption. Renewable and Sustainable Energy Reviews 2018;81:1484-1512. doi:10.1016/j.rser.2017.05.212.

6. Colak I, Yesilbudak M, Genc N, Bayindir R. Multi-period prediction of solar radiation using ARMA and ARIMA models. 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA) 2015:1045-1049.

7. Wang J, Zhou Q, Zhang X. Wind power forecasting based on time series ARMA model. IOP Conference Series: Earth and Environmental Science 2018;199:022015. doi:10.1088/1755- 1315/199/2/022015.

8. Zhang J, Wang C. Application of ARMA model in ultra-short term prediction of wind power. 2013

International Conference on Computer Sciences and Applications 2013:361-364.

9. Mathiesen P, Kleissl J. Evaluation of numerical weather prediction for intra-day solar forecasting in the continental United States. Solar Energy 2011;85:967-977. doi:10.1016/j.solener.2011.02.013.

10. Izgi E, Oztopal A, Yerli B, Kaymak MK, Sahin AD. Short-mid-term solar power prediction by using artificial neural networks. Solar Energy 2012;86:725-733. doi:10.1016/j.solener.2011.11.013.

11. Abuella M, Chowdhury B. Solar power forecasting using artificial neural networks. 2015 North American Power Symposium (NAPS) 2015:1-5.

12. Tang N, Mao S, Wang Y, Nelms RM. Solar power generation forecasting with a LASSO-based approach. IEEE Internet of Things Journal 2018;5:1090-1099. doi:10.1109/JIOT.2018.2812155.

13. Voyant C, Notton G, Kalogirou S, Nivet M-L, Paoli C, Motte F, Fouilloy A. Machine learning methods for solar radiation forecasting: a review. Renewable Energy 2017;105:569-582. doi:10.1016/j.renene.2016.12.095.

14. Guermoui M, Melgani F, Gairaa K, Mekhalfi ML. A comprehensive review of hybrid models for solar radiation forecasting. Journal of Cleaner Production 2020;258:120357. doi:10.1016/j.jclepro.2020.120357.

15. Xie T, Zhang G, Liu H, Liu F, Du P. A hybrid forecasting method for solar output power based on variational mode decomposition, deep belief networks and auto-regressive moving average. Applied Sciences 2018;8:1901. doi:10.3390/app8101901.

16. Qing X, Niu Y. Hourly day-ahead solar irradiance prediction using weather forecasts by LSTM.   
Energy 2018;148:461-468. doi:10.1016/j.energy.2018.01.177.

17. Jeon B, Kim E-J. Next-day prediction of hourly solar irradiance using local weather forecasts and LSTM trained with non-local data. Energies 2020;13:5258. doi:10.3390/en13205258.

18. Rajagukguk RA, Ramadhan RAA, Lee H-J. A review on deep learning models for forecasting time series data of solar irradiance and photovoltaic power. Energies 2020;13:6623. doi:10.3390/en13246623.

19. Liu Y, Guan L, Hou C, Han H, Liu Z, Sun Y, Zheng M. Wind power short-term prediction based on LSTM and discrete wavelet transform. Applied Sciences 2019;9:1108. doi:10.3390/app9061108.

20. Delgado I, Fahim M. Wind turbine data analysis and LSTM-based prediction in SCADA system.   
Energies 2021;14:125. doi:10.3390/en14010125.

21. Hullermeier E, Waegeman W. Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods. Machine Learning 2021;110:457-506. doi:10.1007/s10994-021- 05946-3.

22. Nix DA, Weigend AS. Estimating the mean and variance of the target probability distribution.   
Proceedings of 1994 IEEE International Conference on Neural Networks (ICNN'94) 1994;1:55-60.   
doi:10.1109/ICNN.1994.374138.

23. Bishop CM. Mixture density networks. Technical Report NCRG/94/004, Aston University 1994.

24. Wilkinson IM, Bhattacharjee RR, Shafer JC, Osborne AG. Confidence estimation in the prediction of epithermal neutron resonance self-shielding factors in irradiation samples using an ensemble neural network. Energy and AI 2022;7:100131. doi:10.1016/j.egyai.2021.100131.

25. Rasmussen CE, Williams CKI. Gaussian Processes for Machine Learning. MIT Press, Cambridge, MA 2006.

26. Najibi F, Apostolopoulou D, Alonso E. Enhanced performance Gaussian process regression for probabilistic short-term solar output forecast. International Journal of Electrical Power & Energy Systems 2021;130:106916. doi:10.1016/j.ijepes.2021.106916.

27. Wang Y, Feng B, Hua Q-S, Sun L. Short-term solar power forecasting: a combined long short-term memory and Gaussian process regression method. Sustainability 2021;13:3665. doi:10.3390/su13073665.

28. Zhang Z, Ye L, Qin H, Liu Y, Wang C, Yu X, Yin X, Li J. Wind speed prediction method using shared weight long short-term memory network and Gaussian process regression. Applied Energy 2019;247:270-284. doi:10.1016/j.apenergy.2019.04.047.

29. Gal Y, Ghahramani Z. Dropout as a Bayesian approximation: representing model uncertainty in deep learning. Proceedings of the 33rd International Conference on Machine Learning (ICML), PMLR 2016;48:1050-1059.

30. Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in Neural Information Processing Systems 30 (NeurIPS) 2017:6402- 6413. doi:10.48550/arXiv.1612.01474.

31. Kendall A, Gal Y. What uncertainties do we need in Bayesian deep learning for computer vision?.   
Advances in Neural Information Processing Systems 30 (NeurIPS) 2017:5574-5584.   
doi:10.48550/arXiv.1703.04977.

32. Amini A, Schwarting W, Soleimany A, Rus D. Deep evidential regression. Advances in Neural Information Processing Systems 33 (NeurIPS) 2020:14927-14937. doi:10.48550/arXiv.1910.02600.

33. Flanagan R, Talbot P, Osborne A, Rabiti C, Deinert M. Isolating cloud-induced noise to improve generation of synthetic surface solar irradiances. Advances in Applied Energy 2021;3:100045. doi:10.1016/j.adapen.2021.100045.

34. Sengupta M, Xie Y, Lopez A, Habte A, Maclaurin G, Shelby J. The National Solar Radiation Data Base (NSRDB). Renewable and Sustainable Energy Reviews 2018;89:51-60. doi:10.1016/j.rser.2018.03.003.

35. Diebold FX, Gunther TA, Tay AS. Evaluating density forecasts with applications to financial risk management. International Economic Review 1998;39:863-883. doi:10.2307/2527342.

36. Hochreiter S, Schmidhuber J. Long short-term memory. Neural Computation 1997;9:1735-1780.   
doi:10.1162/neco.1997.9.8.1735.

37. Staudemeyer RC, Morris ER. Understanding LSTM: a tutorial into long short-term memory recurrent neural networks. arXiv:1909.09586 2019. doi:10.48550/arXiv.1909.09586.

38. Abadi M, Agarwal A, Barham P, et al. TensorFlow: large-scale machine learning on heterogeneous

distributed systems. arXiv:1603.04467 2016. doi:10.48550/arXiv.1603.04467.

39. Akiba T, Sano S, Yanase T, Ohta T, Koyama M. Optuna: a next-generation hyperparameter optimization framework. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining 2019:2623-2631. doi:10.1145/3292500.3330701.

40. Gneiting T, Raftery AE. Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association 2007;102:359-378. doi:10.1198/016214506000001437.

41. Duan T, Avati A, Ding DY, Thai KK, Basu S, Ng AY, Schuler A. NGBoost: natural gradient boosting for probabilistic prediction. Proceedings of the 37th International Conference on Machine Learning (ICML), PMLR 2020;119:2690-2700. doi:10.48550/arXiv.1910.03225.

## Supplementary Information

Two-stage residual learning for calibrated uncertainty in solar irradiance forecasts

## Machine learning algorithm operation

The hyperparameters of each network were selected with the Optuna optimization framework, as described in the main text. The parameters swept were the number of epochs, the batch size, the number of neurons, and the dropout rate, which sets the fraction of neuron connections switched off. The dropout rate was held constant across layers when a network had more than one layer, and the number of layers was optimized by running the tuner for each candidate number of layers. Table S1 gives the range searched for each parameter.

<table><tr><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Low</td><td rowspan=1 colspan=1>High</td></tr><tr><td rowspan=1 colspan=1>Epochs</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2000</td></tr><tr><td rowspan=1 colspan=1>Layers</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>Batch size</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>500</td></tr><tr><td rowspan=1 colspan=1>Neurons</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>500</td></tr><tr><td rowspan=1 colspan=1>Dropout rate</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.3</td></tr></table>

Table S1. Range searched for each hyperparameter in the Optuna sweep.

After the sweep, the configuration that gave the smallest root mean square error for each field was selected. These configurations are listed in Tables S2 to S5.

<table><tr><td colspan="1" rowspan="1">Field</td><td colspan="1" rowspan="1">Epochs</td><td colspan="1" rowspan="1">Layers</td><td colspan="1" rowspan="1">Batch size</td><td colspan="1" rowspan="1">Neurons</td><td colspan="1" rowspan="1">Dropout rate</td></tr><tr><td colspan="1" rowspan="1">1 hour, Austin</td><td colspan="1" rowspan="1">1266</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">47</td><td colspan="1" rowspan="1">13</td><td colspan="1" rowspan="1">0.035</td></tr><tr><td colspan="1" rowspan="1">1 hour, Seattle</td><td colspan="1" rowspan="1">1399</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">49</td><td colspan="1" rowspan="1">14</td><td colspan="1" rowspan="1">0.044</td></tr><tr><td colspan="1" rowspan="1">24 hour,</td><td colspan="1" rowspan="1">1874</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">46</td><td colspan="1" rowspan="1">13</td><td colspan="1" rowspan="1">0.023</td></tr><tr><td colspan="1" rowspan="1">Austin</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">24 hour,Seattle</td><td colspan="1" rowspan="1">1801</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">46</td><td colspan="1" rowspan="1">20</td><td colspan="1" rowspan="1">0.021</td></tr></table>

Table S2. Optimized network hyperparameters for the parameter prediction method.

<table><tr><td rowspan=1 colspan=1>Field</td><td rowspan=1 colspan=1>Epochs</td><td rowspan=1 colspan=1>Layers</td><td rowspan=1 colspan=1>Batch size</td><td rowspan=1 colspan=1>Neurons</td><td rowspan=1 colspan=1>Dropout rate</td></tr><tr><td rowspan=1 colspan=1>1 hour, Austin</td><td rowspan=1 colspan=1>1681</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>0.033</td></tr><tr><td rowspan=1 colspan=1>1 hour, Seattle</td><td rowspan=1 colspan=1>1722</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>0.023</td></tr><tr><td rowspan=1 colspan=1>24 hour,Austin</td><td rowspan=1 colspan=1>1337</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>0.041</td></tr><tr><td rowspan=1 colspan=1>24 hour,Seattle</td><td rowspan=1 colspan=1>1201</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>0.039</td></tr></table>

Table S3. Optimized network hyperparameters for the augmented error method.

<table><tr><td rowspan=1 colspan=1>Field</td><td rowspan=1 colspan=1>Epochs</td><td rowspan=1 colspan=1>Layers</td><td rowspan=1 colspan=1>Batch size</td><td rowspan=1 colspan=1>Neurons</td><td rowspan=1 colspan=1>Dropout rate</td></tr><tr><td rowspan=1 colspan=1>1 hour, Austin</td><td rowspan=1 colspan=1>854</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>0.022</td></tr><tr><td rowspan=1 colspan=1>1 hour, Seattle</td><td rowspan=1 colspan=1>777</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>0.036</td></tr><tr><td rowspan=1 colspan=1>24 hour,Austin</td><td rowspan=1 colspan=1>916</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>0.038</td></tr><tr><td rowspan=1 colspan=1>24 hour,Seattle</td><td rowspan=1 colspan=1>988</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>0.041</td></tr></table>

Table S4. Optimized network hyperparameters for the LSTM with Gaussian process regression.

<table><tr><td colspan="1" rowspan="1">Field</td><td colspan="1" rowspan="1">Epochs</td><td colspan="1" rowspan="1">Layers</td><td colspan="1" rowspan="1">Batch size</td><td colspan="1" rowspan="1">Neurons</td><td colspan="1" rowspan="1">Dropout rate</td></tr><tr><td colspan="1" rowspan="1">1 hour, Austin</td><td colspan="1" rowspan="1">335</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">18</td><td colspan="1" rowspan="1">6</td><td colspan="1" rowspan="1">0.043</td></tr><tr><td colspan="1" rowspan="1">1 hour, Seattle</td><td colspan="1" rowspan="1">197</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">17</td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">0.051</td></tr><tr><td colspan="1" rowspan="1">24 hour,Austin</td><td colspan="1" rowspan="1">118</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">28</td><td colspan="1" rowspan="1">16</td><td colspan="1" rowspan="1">0.062</td></tr><tr><td colspan="1" rowspan="1">24 hour,Seattle</td><td colspan="1" rowspan="1">234</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">21</td><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">0.098</td></tr></table>

Table S5. Optimized network hyperparameters for the augmented error form of the LSTM with Gaussian process regression.