# BIOMEDE-517
FINAL PROJECT-KALMANNET

This repository has 2 code files: Kalmanlstm_newmodel & Kalman-RNN-LSTM.
Kalman-RNN-LSTM:
Traditional KalmanNet Architecture: Here we train observation and trajectory models in the beginning with the entire training dataset and then batchwise train and predict gain for every timestep and calculate the present states.


Kalmanlstm_newmodel:
Here we train observation and trajectory models along with kalman gain for every batch with the deeplearnining model and predict it from Neural layer for every timestep. Then use this to calculate current states. This approach ensures that the predicted states are more finely tuned with the evolution matrices, capitalizing on deep learning's strengths to optimize these parameters and improve the model's overall accuracy and performance.

Dataset:
