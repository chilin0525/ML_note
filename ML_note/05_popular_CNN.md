# popular CNN Architectures

## LeNet-5

LeNet-5 架構如下圖:

<div align="center">
<img src="img/lenet5.png" width=700>
</div>

由 2 convolutional layers, 2 subsampling layers 與 3 fully connected layers 組成, LeNet-5 的 5 是因為 5 layers with learnable parameters

input 為 channel = 1 的黑白照片:

<div align="center">
<img src="img/lenet5-1.png" width=100>
</div>

<div align="center">
<img src="img/lenet5-2.png" width=600>
</div>

<br>

details(計算方法可以參考 [04_CNN:terminology](04_CNN.md#terminology)):

|Layer|filter size|filters|stride|activation function|feature map size|
|:---:|:---:|:---:|:---:|:---:|:---:|
|input|-|-|-|-|32x32x1|
|Conv1|5x5|6|1|tanh|28x28x6|
|Avg. Pooling1|2x2|-|2|-|14x14x6|
|Conv2|5x5|16|1|tanh|10x10x6|
|Avg. Pooling2|2x2|-|2|-|5x5x16|
|Conv3|5x5|120|1|tanh|120|
|Fully Connected1|-|-|-|tanh|84|
|Fully Connected2|-|-|-|softmax|10|

* TODO: trainable parameters
* TODO: connections
    * [LeNet-5 nice slide](https://www.slideshare.net/ssuser2e52e8/lenet5)
---

## ref

* [LeNet-5 論文: Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
* [LeNet-5 網站](http://yann.lecun.com/exdb/lenet/index.html)
* [towardsdatascience:Illustrated: 10 CNN Architectures](https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d#c5a6)