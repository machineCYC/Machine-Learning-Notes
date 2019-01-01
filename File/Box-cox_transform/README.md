# Box-cox transform

Box-cox 轉換是統計建模中常使用的方式，透過 box-cox 轉換過後可以消除原來資料的偏度 (skewness)，盡可能的接近常態分佈。至於要逼近常態分佈的主要原因是，在統計模型中很多的前提假設都會跟常態分佈有關。

在 box-cox 轉換當中有個參數 <a href="https://www.codecogs.com/eqnedit.php?latex=\lambda" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /></a> ，估計這參數主要的方法是透過最大概四估計法，詳細細節可以參考 reference。

在 python3 中，可以透過 scipy 中的 stats 來幫我們估計 <a href="https://www.codecogs.com/eqnedit.php?latex=\lambda" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /></a> ，另外 scipy.special.boxcox1p 來轉換資料。參考程式碼如下:

```
from scipy.special import boxcox1p
from scipy import stats

lambda_range = np.linspace(-2, 2, 100)
loglikelihood = np.zeros(lambda_range.shape, dtype=float)

for idx, lam in enumerate(lambda_range):
    loglikelihood = stats.boxcox_llf(lambda, data)
lam_best = lambda_range[loglikelihood.argmax()]

transform_data = boxcox1p(data, round(lam_best, 2))

```

Box-cox 轉換有時候會遇到一個問題，由於轉換公式跟 log 有關，所以資料不可以為負的，但實際資料常常不能避免這情況，所以會先透過 shirft data 來達到此條件。

另外 box-cox 轉換其實是一種比較廣泛的 power 轉換，透過不同的參數可以達到 log-transform、sqrt-transform...等。

## Reference

[Box-Cox Transformations](http://onlinestatbook.com/2/transformations/box-cox.html)