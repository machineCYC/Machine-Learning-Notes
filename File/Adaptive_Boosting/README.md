# Adaptive Boosting

AdaBoosting 在各大競賽中常常被使用，能應用在 machine learning 的各個分支，分類、回歸、排序問題...等。

AdaBoost 算法，是一種改進的 Boosting 分類算法。透過提高前幾個分類器線性組合的分類錯誤樣本的權重，這樣做可以讓每次訓練新的分類器的時後都聚焦在容易分類錯誤的訓練樣本上。每個弱分類器使用加權投票機制取代平均投票機制，準確率較大的弱分類器有較大的權重，反之，準確率低的弱分類器權重較低。

最基本的想法對資料找一組新的權重使得 M_t-1 表現得不好，進而讓 M_t 學得比較好。透過這樣的方式在每次產生新模型的時候都會跟之前的模型擁有不同特性，可以理解成每個模型在某個特徵比較特別好，但在整體卻沒有很好，也就是所謂的弱分類器。

至於要怎麼對資料找到一組新的權重呢?舉一個簡單的分類例子來說明:

| Entries | weight(old) | M_1 predict result | weight(new) |
| --- | --- | --- | --- |
| X_1, y_1 | u_0_1=1 | right | u_1_1=1/sqrt(3) |
| X_2, y_2 | u_0_2=1 | wrong | u_1_2=sqrt(3) |
| X_3, y_3 | u_0_3=1 | right | u_1_3=1/sqrt(3) |
| X_4, y_4 | u_0_4=1 | right | u_1_4=1/sqrt(3) |

如上表，是個二元分類的問題，在二元分類問題中，如果一個模型的錯誤率為 0.5 意味著跟用猜的沒什麼兩樣，也就是模型表現得不好。因此在這 case 我們會希望找到一組權重，使得原來的模型錯誤預測率為 0.5，其中錯誤預測率的計算方式為

<a href="https://www.codecogs.com/eqnedit.php?latex=\varepsilon&space;_{n}=\frac{\sum_{i}^{&space;}u_{i}^{n}\delta&space;(M_{n}(x_{i})\neq&space;y_{i})}{\sum_{i}^{&space;}u_{i}^{n}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\varepsilon&space;_{n}=\frac{\sum_{i}^{&space;}u_{i}^{n}\delta&space;(M_{n}(x_{i})\neq&space;y_{i})}{\sum_{i}^{&space;}u_{i}^{n}}" title="\varepsilon _{n}=\frac{\sum_{i}^{ }u_{i}^{n}\delta (M_{n}(x_{i})\neq y_{i})}{\sum_{i}^{ }u_{i}^{n}}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\delta&space;(M_{n}(x_{i})\neq&space;y_{i})=\left\{\begin{matrix}&space;1,&space;M_{n}(x_{i})\neq&space;y_{i}&space;\\&space;0,&space;M_{n}(x_{i})=&space;y_{i}&space;\end{matrix}\right" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta&space;(M_{n}(x_{i})\neq&space;y_{i})=\left\{\begin{matrix}&space;1,&space;M_{n}(x_{i})\neq&space;y_{i}&space;\\&space;0,&space;M_{n}(x_{i})=&space;y_{i}&space;\end{matrix}\right" title="\delta (M_{n}(x_{i})\neq y_{i})=\left\{\begin{matrix} 1, M_{n}(x_{i})\neq y_{i} \\ 0, M_{n}(x_{i})= y_{i} \end{matrix}\right" /></a>

u_n_i 為第 i 個 training data 在第 n 個模型的權重。

假設有 4 筆資料 <a href="https://www.codecogs.com/eqnedit.php?latex=(x_{1},&space;y_{1})&space;...&space;(x_{4},&space;y_{4})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(x_{1},&space;y_{1})&space;...&space;(x_{4},&space;y_{4})" title="(x_{1}, y_{1}) ... (x_{4}, y_{4})" /></a>，每筆資料對應的權重分別為 <a href="https://www.codecogs.com/eqnedit.php?latex=u_{1},&space;u_{2},...,u_{4}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u_{1},&space;u_{2},...,u_{4}" title="u_{1}, u_{2},...,u_{4}" /></a> ，經過 M_1 預測過後，分類錯誤率為 0.25 ，其中第二筆資料預測錯誤，這時我們希望提高錯誤資料的權重和降低正確資料的權重，weight (new) 所代表調整過後的權重，調整過後導致 M_1 錯誤預測率為 0.5。

有了這個範例可以得知，

當預測正確時，<a href="https://www.codecogs.com/eqnedit.php?latex=u_{i}^{n}\leftarrow&space;u_{i}^{n}/d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u_{i}^{n}\leftarrow&space;u_{i}^{n}/d" title="u_{i}^{n}\leftarrow u_{i}^{n}/d" /></a>，預測錯誤時，<a href="https://www.codecogs.com/eqnedit.php?latex=u_{i}^{n}\leftarrow&space;u_{i}^{n}*d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u_{i}^{n}\leftarrow&space;u_{i}^{n}*d" title="u_{i}^{n}\leftarrow u_{i}^{n}*d" /></a>，其中 d > 1

至於這個 d 該如何計算，其實也只是解下列等式

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\sum_{i}^{&space;}u_{i}^{n&plus;1}\delta(M_{n}(x_{i})\neq&space;y_{i})}{\sum_{i}^{&space;}u_{i}^{n&plus;1}}=0.5" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\sum_{i}^{&space;}u_{i}^{n&plus;1}\delta(M_{n}(x_{i})\neq&space;y_{i})}{\sum_{i}^{&space;}u_{i}^{n&plus;1}}=0.5" title="\frac{\sum_{i}^{ }u_{i}^{n+1}\delta(M_{n}(x_{i})\neq y_{i})}{\sum_{i}^{ }u_{i}^{n+1}}=0.5" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\Rightarrow&space;\frac{\sum_{M_{n}(x_{i})\neq&space;y_{i}}^{&space;}u_{i}^{n}*d^{n}}{\sum_{M_{n}(x_{i})\neq&space;y_{i}}^{&space;}u_{i}^{n}*d^{n}&space;&plus;&space;\sum_{M_{n}(x_{i})\neq&space;y_{i}}^{&space;}u_{i}^{n}/d^{n}}=0.5" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Rightarrow&space;\frac{\sum_{M_{n}(x_{i})\neq&space;y_{i}}^{&space;}u_{i}^{n}*d^{n}}{\sum_{M_{n}(x_{i})\neq&space;y_{i}}^{&space;}u_{i}^{n}*d^{n}&space;&plus;&space;\sum_{M_{n}(x_{i})\neq&space;y_{i}}^{&space;}u_{i}^{n}/d^{n}}=0.5" title="\Rightarrow \frac{\sum_{M_{n}(x_{i})\neq y_{i}}^{ }u_{i}^{n}*d^{n}}{\sum_{M_{n}(x_{i})\neq y_{i}}^{ }u_{i}^{n}*d^{n} + \sum_{M_{n}(x_{i})\neq y_{i}}^{ }u_{i}^{n}/d^{n}}=0.5" /></a>

最後就可以得到

<a href="https://www.codecogs.com/eqnedit.php?latex=d^{n}=\sqrt{\frac{1-\varepsilon&space;_{n}}{\varepsilon&space;_{n}}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d^{n}=\sqrt{\frac{1-\varepsilon&space;_{t}}{\varepsilon&space;_{n}}}" title="d^{n}=\sqrt{\frac{1-\varepsilon _{n}}{\varepsilon _{n}}}" /></a>

至於模型的權證可以透過模型的預測錯誤率來計算

<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha&space;_{n}=\ln&space;\sqrt{\frac{1-\varepsilon&space;_{n}}{\varepsilon&space;_{n}}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha&space;_{n}=\ln&space;\sqrt{\frac{1-\varepsilon&space;_{n}}{\varepsilon&space;_{n}}}" title="\alpha _{n}=\ln \sqrt{\frac{1-\varepsilon _{n}}{\varepsilon _{n}}}" /></a>

透過上述的過程就可以不斷地去創造新模型，最就將這些模型透過加權的模式計算出結果。

<a href="https://www.codecogs.com/eqnedit.php?latex=H(x_{i})=sign(\sum_{n}^{&space;}\alpha&space;_{n}*M_{n}(x_{i}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?H(x_{i})=sign(\sum_{n}^{&space;}\alpha&space;_{n}*M_{n}(x_{i}))" title="H(x_{i})=sign(\sum_{n}^{ }\alpha _{n}*M_{n}(x_{i}))" /></a>

AdaBoost 的手法就是讓判斷錯誤的 train data 提高權重，讓產生新的權重的 training set 讓舊的模型 M_n fail掉，但在新的模型上就去加強學這些權重較大的 training set。最後將所有模型根據權重加權起來得到結果。

## Parameter tunning


## Reference

[ML Lecture 22: Ensemble (李弘毅)](https://www.youtube.com/watch?v=tH9FH1DH5n0)