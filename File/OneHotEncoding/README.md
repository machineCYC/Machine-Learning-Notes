# One Hot Encoder

在資料科學的領域中，模型終究只能透過數字來做計算，但在客戶交易資料或者是自然語言處理方面所遇到的資料通常會有很多不是數字 e.g.性別、客戶類別、促銷月份...等。

而將這些類別的變數轉成數值變數就會需要一些方法，One Hot Encoder 只是其中一種方法。

**One Hot Encoder 通常針對沒有順序性的類別變數去做轉換，不同類別支間不存在任何的順序關係**。例如:性別就沒有男生大於女生或女生大於男生這種情況。

但 One Hot Encoder 有時候也會衍伸出許多問題

1. 當類別變數中的類別數量分常多的時候，One hot encodeing 會導致資料非常高維度，進而影響模型訓練時間及記憶體的消耗

2. One hot encodeing 容易導致模型 overfitting 資料

3. 如果測試資料中有新的類別，不再訓練資料中出現，One hot encodeing 就會產生問題，像是 device type 會隨時間一直出現新的編碼，這時使用 One hot encodeing 就會造成很多困擾。

在 python3 中可以透過 sklearn 來幫我們完成 One Hot Encoder 的轉換，上述第三點的問題也可以透過 handle_unknown="ignore" 來解決。如果出現新的類別則 0 向量來表示。程式如下所示:


```
from sklearn.preprocessing import OneHotEncoder
import numpy as np

data = np.array(["a", "b", "a"])
data = data.reshape(-1, 1)

OHE = OneHotEncoder(handle_unknown="ignore")
OHE.fit(temp)

transform_data = OHE.transform(data).toarray()
```

# Reference

[scikit-learn OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)