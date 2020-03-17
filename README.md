# DILATE: DIstortion Loss with shApe and tImE
[Vincent Le Guen](https://www.linkedin.com/in/vincentleguen/),  [Nicolas Thome](http://cedric.cnam.fr/~thomen/)

Code for our NeurIPS 2019 paper "Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models"

![](https://github.com/vincent-leguen/DILATE/blob/master/fig2.png)

If you find this code useful for your research, please cite our [paper](https://papers.nips.cc/paper/8672-shape-and-time-distortion-loss-for-training-deep-time-series-forecasting-models):

```
@incollection{leguen19dilate,
title = {Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models},
author = {Le Guen, Vincent and Thome, Nicolas},
booktitle = {Advances in Neural Information Processing Systems},
pages = {4191--4203},
year = {2019}
}
```

## Abstract
This paper addresses the problem of time series forecasting for non-stationary signals and multiple future steps prediction. To handle this challenging task, we introduce DILATE (DIstortion Loss including shApe and TimE), a new objective function for training deep neural networks. DILATE aims at accurately predicting sudden changes, and explicitly incorporates two terms supporting precise shape and temporal change detection. We introduce a differentiable loss function suitable for training deep neural nets, and provide a custom back-prop implementation for speeding up optimization. We also introduce a variant of DILATE, which provides a smooth generalization of temporally-constrained Dynamic Time Warping (DTW). Experiments carried out on various non-stationary datasets reveal the very good behaviour of DILATE compared to models trained with the standard Mean Squared Error (MSE) loss function, and also to DTW and variants. DILATE is also agnostic to the choice of the model, and we highlight its benefit for training fully connected networks as well as specialized recurrent architectures, showing its capacity to improve over state-of-the-art trajectory forecasting approaches.
