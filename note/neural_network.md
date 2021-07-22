# neural network

在 linear classification 中我們將 score function 寫為下面形式:

<div align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;s=Wx&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;s=Wx&space;\end{align*}" title="\begin{align*} s=Wx \end{align*}" /></a>
</div>

<br>

* 其中以 CIFAR-10 舉例的話:
    * x = [3072x1] column vector, because each image in CIFAR-10 is 32x32x3
    * W = [10x3072] matrix, 同時我們可以把 W 想成是 10 class 的模板, 跟模板相近的就越可能是該 class
    * f(x,W) 的結果就是對 10 個 class 的分數

<br>

<div align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;s=W_{2}max(0,W_{1})&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;s=W_{2}max(0,W_{1})&space;\end{align*}" title="\begin{align*} s=W_{2}max(0,W_{1}) \end{align*}" /></a>
</div>

<br>

<div align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;s=W_{3}max(W_{2}max(0,W_{1}x)&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;s=W_{3}max(W_{2}max(0,W_{1}x)&space;\end{align*}" title="\begin{align*} s=W_{3}max(W_{2}max(0,W_{1}x) \end{align*}" /></a>
</div>

## the architecture of neural networks
