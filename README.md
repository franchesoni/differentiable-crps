
# Differentiable CRPS and an application to probabilistic solar time-series forecasting

If you came here for the loss, check the `crps_loss_batch` function in `main.py`. We predict bin probabilities, but you could fix them and predict bin borders, or predict both.
We build this repo with simplicity in mind. We use torch, which is just like numpy, but with automated differentiation. 

## Results

|Loss name| Test MAE | Test CRPS|
|---|---|---|
|MAE|0.155|0.155|
|CE|0.171|0.128|
|**CRPS (ours)**|**0.130**|**0.0975**|

## Solar forecasting

The solar data is in a CSV. The Global Horizontal Irradiance (GHI) is the most important factor determining the PV power output. However it depends on the position of the sun. For forecasting we need to get rid of this deterministic dependency. This leaves us with the clear sky index (CSI), which tell us how clear is the sky. To get a GHI forecast, we multiply the forecasted CSI by the clear sky model.

We only predict CSI. It should be easy for you to go to GHI. 
Article coming soon. 


## Training of networks

We build a multilayer perceptron with the last 18 observations (3 hours) as input. 
To evaluate, we sample a test set use the second half of the data (a whole year). We also remove the first 24*6 datapoints of the beggining of the test set to avoid any overlap.
From the train set, we build samples by taking slices of 18 samples and the sample one hour afterwards, which is 6 datapoints.

The validation set is about 10% of the training data taken in two intervals, one at the begginging and one at the middle. This basically evaluates the performance on winter and summer and reduces overlap with the training set.

