# Gradient Boosting (GBM)

Gradient Boosting 在各大競賽中常常被使用，能應用在 machine learning 的各個分支，分類、回歸、排序問題...等。

Gradient boosting 的概念是疊代生成多個（N個）弱的模型，然後將每個弱模型的預測結果相加，後面的模型 Fn+1(x) 基於前面學習模型的 Fn(x) 的效果生成的。

用一個簡單的例子來說明:

如果我們有個模型 M，目前模型 M 的 MSE 是 100.25 (可以是任何的衡量方式)，我們透過 Gradient boosting 的方式來改進。

```
Y = M(x) + error1
```

摸型 M 所預測出來的值跟真實值有 error1 的差距，我們可以透過另一個模型 M2 來捕捉 error1。

```
error1 = M2(x) + error2
```

利用 M2 來預測 error1 也會產生 error2，再利用另一個模型 M3 來捕捉 error2。

```
error2 = M3(x) + error3
```

而 M3 也會產生相對應的 error3，但 error3 相對而言會比 error1 小。

```
Y = M(x) + M2(x) + M3(x) + error3
```

最後將所有的模型加起來可以得到如上的式子，這樣就可以獲得一個相對好的模型。可以發現每個模型都是基於上個模型的 error 來建模，這 error 又稱為殘差 (Residual)，而這殘差剛好就是負的 MSE 一階倒數。但 MSE 這 loss function 又很容易受到 outlier 影響，所以常常都會用 MAE 或 Huber loss 替代。


## Parameter tunning

- GBM 參數大致可分為三類

    - tree 相關的參數: 其實也不一定是 tree，只是常見的方式都是透過tree base model 去實踐
    - boosting 相關的參數: boosting 相關的設定，例如:learning rate、tree 的個數 (n_estimators)...等
    - 其他參數: loss function、random seed...等

step1. 選定一個相對高的 learning rate (0.05~0.2)

step2. 根據這個 learning rate 決定 n_estimators，通常都選擇可以快速計算完的個數

step3. tune tree 相關的參數

step4. 調降 learning rate 和增加 n_estimator，進而獲得相對 robust 模型

## Relative Project

[House Prices: Advanced Regression Techniques](https://github.com/machineCYC/SideProjects/tree/master/06-Kaggle-HousePricesAdvancedRegTech)

## Reference
