import pandas as pd
import numpy as np
import torch
import torch.nn as nn


class MLPptorch(nn.Module):
    def __init__(self,
                 in_size,
                 first_hidden_size,
                 second_hidden_size,
                 third_hidden_size,
                 out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(nn.Linear(in_size, first_hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(first_hidden_size, second_hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(second_hidden_size, third_hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(third_hidden_size, out_size),
                                    nn.Sigmoid())

    # прямой проход
    def forward(self, x):
        return self.layers(x)


# функция обучения
def train(x, y, num_iter):
    for i in range(0, num_iter):
        pred = net.forward(x)
        loss = lossFn(pred, y)
        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            print('Ошибка на ' + str(i) + ' итерации: ', loss.item())
    return loss.item()


df = pd.read_csv('data.csv')
df = df.iloc[np.random.permutation(len(df))]

X = df.iloc[0:100, 0:3].values
y = df.iloc[0:100, 4]
y = y.map({'Iris-setosa': 1, 'Iris-virginica': 2, 'Iris-versicolor': 3}).values.reshape(-1, 1)
Y = np.zeros((y.shape[0], np.unique(y).shape[0]))
for i in np.unique(y):
    Y[:, i - 1] = np.where(y == i, 1, 0).reshape(1, -1)

X_test = df.iloc[100:150, 0:3].values
y = df.iloc[100:150, 4]
y = y.map({'Iris-setosa': 1, 'Iris-virginica': 2, 'Iris-versicolor': 3}).values.reshape(-1, 1)
Y_test = np.zeros((y.shape[0], np.unique(y).shape[0]))
for i in np.unique(y):
    Y_test[:, i - 1] = np.where(y == i, 1, 0).reshape(1, -1)

inputSize = X.shape[1]  # количество входных сигналов равно количеству признаков задачи
first_hidden_size = 50  # задаем число нейронов скрытого слоя
second_hidden_size = 20  # задаем число нейронов скрытого слоя
third_hidden_size = 10  # задаем число нейронов скрытого слоя
outputSize = Y.shape[1] if len(Y.shape) else 1  # количество выходных сигналов равно количеству классов задачи

net = MLPptorch(inputSize,
                first_hidden_size,
                second_hidden_size,
                third_hidden_size,
                outputSize)
lossFn = nn.MSELoss()

# создает оптимизатор Stochastic Gradient Descent (SGD) для обучения сети net
# net.parameters() - веса
optimizer = torch.optim.SGD(net.parameters(), lr=0.009)

#  запуск процесса обучения сети net с использованием функции train.
loss_ = train(torch.from_numpy(X.astype(np.float32)),
              torch.from_numpy(Y.astype(np.float32)), 5000)


pred = net.forward(torch.from_numpy(X.astype(np.float32))).detach().numpy()
err = sum(abs((pred > 0.5) - Y))
print(err)

pred = net.forward(torch.from_numpy(X_test.astype(np.float32))).detach().numpy()
err = sum(abs((pred > 0.5) - Y_test))
print(err)
