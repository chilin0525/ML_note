# backpropagation

backpropagation 為實際上用 gradient descent 的方法 train 一個 neural network 的 algorithm

## computation graph

利用 computation graph 來表達任意 function, 在 computation graph 中的 node 表示我們要執行的每一步運算, 一旦 function 可以利用 computation graph 來表達的話就可以利用 backpropagation 技術 recurively 利用 chain rule 算出每個維度的 gradient

Example:

<div align="center">
<img src="img/computation_graph.png" width=400>
</div>

<br>
<div align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;&f(x,y,z)=(x&plus;y)z&space;\\&space;&e.q.\:&space;x=-2,y=5,z=-4&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;&f(x,y,z)=(x&plus;y)z&space;\\&space;&e.q.\:&space;x=-2,y=5,z=-4&space;\end{align*}" title="\begin{align*} &f(x,y,z)=(x+y)z \\ &e.q.\: x=-2,y=5,z=-4 \end{align*}" /></a>
</div>

<br>
<div align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;want:\:&space;\frac{\partial&space;f}{\partial&space;x},\frac{\partial&space;f}{\partial&space;},\frac{\partial&space;f}{\partial&space;z}&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;want:\:&space;\frac{\partial&space;f}{\partial&space;x},\frac{\partial&space;f}{\partial&space;},\frac{\partial&space;f}{\partial&space;z}&space;\end{align*}" title="\begin{align*} want:\: \frac{\partial f}{\partial x},\frac{\partial f}{\partial },\frac{\partial f}{\partial z} \end{align*}" /></a>
</div>

<br>

利用變數取代中間的計算結果並計算偏微分的結果:

<div align="center">
<img src="img/computation_graph2.png" width=400>
</div>

<br>

<div align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;&q=x&plus;y\:&space;\:&space;\:&space;\:&space;\frac{\partial&space;q}{\partial&space;x}=1,\frac{\partial&space;q}{\partial&space;y}=1&space;\\&space;&f=qz\:&space;\:&space;\:&space;\:&space;\:&space;\:&space;\:&space;\:&space;\:&space;\frac{\partial&space;f}{\partial&space;q}=z,\frac{\partial&space;f}{\partial&space;z}=q&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;&q=x&plus;y\:&space;\:&space;\:&space;\:&space;\frac{\partial&space;q}{\partial&space;x}=1,\frac{\partial&space;q}{\partial&space;y}=1&space;\\&space;&f=qz\:&space;\:&space;\:&space;\:&space;\:&space;\:&space;\:&space;\:&space;\:&space;\frac{\partial&space;f}{\partial&space;q}=z,\frac{\partial&space;f}{\partial&space;z}=q&space;\end{align*}" title="\begin{align*} &q=x+y\: \: \: \: \frac{\partial q}{\partial x}=1,\frac{\partial q}{\partial y}=1 \\ &f=qz\: \: \: \: \: \: \: \: \: \frac{\partial f}{\partial q}=z,\frac{\partial f}{\partial z}=q \end{align*}" /></a>
</div>

<br>

backpropagation 顧名思義就是從後面往前進行計算:

<div align="center">
<img src="img/computation_graph3.png" width=400>
</div>

<div align="center">
<img src="img/computation_graph4.png" width=400>
</div>

<div align="center">
<img src="img/computation_graph5.png" width=400>
</div>

<div align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;&\frac{\partial&space;f}{\partial&space;z}=x&plus;y=3&space;\\&space;&\frac{\partial&space;f}{\partial&space;q}=z=-4&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;&\frac{\partial&space;f}{\partial&space;z}=x&plus;y=3&space;\\&space;&\frac{\partial&space;f}{\partial&space;q}=z=-4&space;\end{align*}" title="\begin{align*} &\frac{\partial f}{\partial z}=x+y=3 \\ &\frac{\partial f}{\partial q}=z=-4 \end{align*}" /></a>
</div>

<br>
<div align="center">
<img src="img/computation_graph6.png" width=400>
</div>

<div align="center">
<img src="img/computation_graph7.png" width=400>
</div>

<br>
<div align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;&\frac{\partial&space;f}{\partial&space;x}=\frac{\partial&space;f}{\partial&space;q}\frac{\partial&space;q}{\partial&space;x}=z=-4&space;\\&space;&\frac{\partial&space;f}{\partial&space;y}=\frac{\partial&space;f}{\partial&space;q}\frac{\partial&space;q}{\partial&space;y}=z=-4&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;&\frac{\partial&space;f}{\partial&space;x}=\frac{\partial&space;f}{\partial&space;q}\frac{\partial&space;q}{\partial&space;x}=z=-4&space;\\&space;&\frac{\partial&space;f}{\partial&space;y}=\frac{\partial&space;f}{\partial&space;q}\frac{\partial&space;q}{\partial&space;y}=z=-4&space;\end{align*}" title="\begin{align*} &\frac{\partial f}{\partial x}=\frac{\partial f}{\partial q}\frac{\partial q}{\partial x}=z=-4 \\ &\frac{\partial f}{\partial y}=\frac{\partial f}{\partial q}\frac{\partial q}{\partial y}=z=-4 \end{align*}" /></a>
</div>

<br>

為什麼不直接做偏微分, 也可以直接得到一樣的結果是因為這邊的 function 太過簡單, 很容易做偏微分, 但如果是一個很複雜的 function 依然可以利用 computation graph 做出來且容易 implement

<br>

<div align="center">
<img src="img/computation_graph8.png" width=400>
</div>

<br>

因此總結 computation graph 要如何計算 local gradient, 就是用上游得到的 gradients 乘上 local gradient 就能得到 input gradient

Example:

<div align="center">
<img src="img/computation_graph9.png">
</div>

<div align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}&space;&f(w,x)=\frac{1}{1&plus;e^{-(w_0x_0&plus;w_1x_1&plus;w_2)}}&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}&space;&f(w,x)=\frac{1}{1&plus;e^{-(w_0x_0&plus;w_1x_1&plus;w_2)}}&space;\end{align*}" title="\begin{align*} &f(w,x)=\frac{1}{1+e^{-(w_0x_0+w_1x_1+w_2)}} \end{align*}" /></a>
</div>



---

## ref

* [cs231n Course Notes : Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/)
* [YT: cs231n Lecture 4 | Introduction to Neural Networks](https://www.youtube.com/watch?v=d14TUNcbn1k&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=4&ab_channel=StanfordUniversitySchoolofEngineering)