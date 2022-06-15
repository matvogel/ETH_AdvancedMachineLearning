# Project 2
This project was about time-series classification. The original data came from different mobile devices which
measured the ECG, sampled at 300Hz. One could apply RNN or extract features from the time series, where the latter was used. The final goal was to detect different kind of heart diseases from the data.

Features were extracted with libraries like ``biosppy`` or hand-crafted.
They included classic PQRST information, morpholigical features or simple RMS and SR values, to detect noise-only-data. There was also filtering applied, to remove the 50Hz noise and baseline-wandering.

The dataset cannot be provided here, only the calculated features. There is an example image of how the data looked like in the notebook. The feature calculation is still in the script if anyone is interested in how the features are calculated.

XGB was used as classifier and it achieved satisfactory results but was not amongst the best. One should definitely try recurrent neural networks on it! In this project I learned about feature extraction from time-series, in both the time domain and frequency domain.
