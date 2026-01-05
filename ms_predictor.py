import pandas as pd
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops, context

# CPU is fine
ms.set_device("CPU")


class UtilPredictor1to1:
    """
    MindSpore forecasting model:
    input  = util_n(t)  (one number)
    output = util_n(t+1) (one number)
    """

    def __init__(self):
        self.net = nn.SequentialCell(
            nn.Dense(1, 16),
            nn.ReLU(),
            nn.Dense(16, 1)
        )
        self.loss_fn = nn.MSELoss()
        self.opt = nn.Adam(self.net.trainable_params(), learning_rate=0.01)

        # MindSpore training wrapper (this is the correct way)
        self.train_step = nn.TrainOneStepCell(
            nn.WithLossCell(self.net, self.loss_fn),
            self.opt
        )
        self.train_step.set_train()

    def fit(self, csv_path: str, epochs=200):
        df = pd.read_csv(csv_path)

        util = (df["server_utilization"].values.astype(np.float32) / 100.0)

        X = util[:-1].reshape(-1, 1)
        y = util[1:].reshape(-1, 1)

        X = Tensor(X, ms.float32)
        y = Tensor(y, ms.float32)

        for ep in range(epochs):
            loss = self.train_step(X, y)

        self.net.set_train(False)

    def predict_next_util(self, util_n: float) -> float:
        x = Tensor(np.array([[util_n]], dtype=np.float32))
        yhat = self.net(x)
        return float(yhat.asnumpy()[0, 0])
