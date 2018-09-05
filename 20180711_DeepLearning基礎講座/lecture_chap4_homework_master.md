
# 第4回講義 宿題

## 課題. MNISTデータセットを多層パーセプトロン(MLP)で学習せよ

### 注意
- homework関数を完成させて提出してください
    - 訓練データはtrain_X, train_y, テストデータはtest_Xで与えられます
    - train_Xとtrain_yをtrain_X, train_yとvalid_X, valid_yに分けるなどしてモデルを学習させてください
    - test_Xに対して予想ラベルpred_yを作り, homework関数の戻り値としてください\
- pred_yのtest_yに対する精度(F値)で評価します
- 全体の実行時間がiLect上で60分を超えないようにしてください
- homework関数の外には何も書かないでください (必要なものは全てhomework関数に入れてください)
- 解答提出時には Answer Cell の内容のみを提出してください

- MLPの実装にTensorflowなどのライブラリを使わないでください

### ヒント
- 出力yはone-of-k表現
- 最終層の活性化関数はソフトマックス関数, 誤差関数は多クラス交差エントロピー
- 最終層のデルタは教科書参照

次のセルのhomework関数を完成させて提出してください

# Answer Cell


```python
def homework(train_X, train_y, test_X):
    from sklearn.model_selection import train_test_split

    # Layer
    class Layer:
        # Constructor
        def __init__(self, in_dim, out_dim, function, deriv_function):
            self.W = np.random.uniform(low=-0.08, high=0.08,
                                       size=(in_dim, out_dim))
            self.b = np.zeros(out_dim)
            self.function = function
            self.deriv_function = deriv_function
            self.u = None
            self.delta = None

        # Forward Propagation
        def f_prop(self, x):
            self.u = np.dot(x, self.W) + self.b
            self.z = self.function(self.u)
            return self.z

        # Back Propagation
        def b_prop(self, delta, W):
            self.delta = self.deriv_function(self.u)*np.dot(delta, W.T)
            return self.delta

    # Forward Propagation
    def f_props(layers, x):
        z = x
        for layer in layers:
            z = layer.f_prop(z)
        return z

    # Back Propagation
    def b_props(layers, delta):
        for i, layer in enumerate(layers[::-1]):
            if i == 0:
                layer.delta = delta
            else:
                delta = layer.b_prop(delta, _W)
            _W = layer.W

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def deriv_sigmoid(x):
        return sigmoid(x)*(1 - sigmoid(x))

    def softmax(x):
        tem = np.exp(x)
        return tem/np.sum(tem, axis=1)[:, np.newaxis]

    def deriv_softmax(x):
        return softmax(x)*(np.ones(x.shape) - softmax(x))

    def tanh(x):
        return np.tanh(x)

    def deriv_tanh(x):
        return 1 - tanh(x)**2

    layers = [
        Layer(784, 200, sigmoid, deriv_sigmoid),
        Layer(200, 200, sigmoid, deriv_sigmoid),
        Layer(200, 10, softmax, deriv_softmax)
    ]

    def train(X, d, eps=0.1):
        # Forward Propagation
        y = f_props(layers, X)

        # Cost Function & Delta
        cost = - np.sum(d*np.log(y))
        delta = y - d

        # Back Propagation
        b_props(layers, delta)

        # Update Parameters
        z = X
        for i, layer in enumerate(layers):
            dW = np.dot(z.T, layer.delta)
            db = np.dot(np.ones(len(z)), layer.delta)

            layer.W = layer.W - eps*dW
            layer.b = layer.b - eps*db

            if i != len(layers) - 1:
                z = layer.z
        # Train Cost
        y = f_props(layers, X)
        cost = - np.sum(d*np.log(y))

        return cost

    def valid(X, d):
        # Test Cost
        y = f_props(layers, X)
        cost = - np.sum(d*np.log(y))
        return cost, y

    def test(X):
        # Test Cost
        y = f_props(layers, X)
        return y

    train_y = np.eye(10)[train_y]

    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y,
                                                          test_size=0.2,
                                                          random_state=42)

    # Epoch
    for epoch in range(2):
        # Online Learning
        train_X, train_y = shuffle(train_X, train_y)
        for x, y in zip(train_X, train_y):
            cost = train(x[np.newaxis, :], y[np.newaxis, :], eps=0.1)

        cost, pred_y = valid(valid_X, valid_y)

    pred_y = np.argmax(test(test_X), axis=1)
    return pred_y
```

- 以下のvalidate_homework関数を用いてエラーが起きないか動作確認をして下さい。
- 提出に際して、 以下のscore_homework関数で60分で実行が終わることを確認して下さい。
- 評価は以下のscore_homework関数で行われますが、random_stateの値は変更されます。

# Checker Cell (for student)


```python
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

import numpy as np

def load_mnist():
    mnist = fetch_mldata('MNIST original')
    mnist_X, mnist_y = shuffle(mnist.data.astype('float32'),
                               mnist.target.astype('int32'), random_state=42)

    mnist_X = mnist_X / 255.0

    return train_test_split(mnist_X, mnist_y,
                test_size=0.2,
                random_state=42)

def validate_homework():
    train_X, test_X, train_y, test_y = load_mnist()

    # validate for small dataset
    train_X_mini = train_X[:100]
    train_y_mini = train_y[:100]
    test_X_mini = test_X[:100]
    test_y_mini = test_y[:100]

    pred_y = homework(train_X_mini, train_y_mini, test_X_mini)
    print(f1_score(test_y_mini, pred_y, average='macro'))

def score_homework():
    train_X, test_X, train_y, test_y = load_mnist()
    pred_y = homework(train_X, train_y, test_X)
    print(f1_score(test_y, pred_y, average='macro'))
```


```python
#validate_homework()
score_homework()
```

    0.951486846066

