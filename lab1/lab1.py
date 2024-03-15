import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)

X = df.iloc[0:100, [0, 2]].values

input_size = X.shape[1]
neurons_hidden_layer = 10
neurons_output_layer = 1 if len(y.shape) else y.shape[1]  # количество выходных сигналов равно количеству классов задачи
print('input_size:', input_size)
print('neurons_hidden_layer:', neurons_hidden_layer)
print('neurons_output_layer:', neurons_output_layer)

# матрица весов и порого первого слоя
W_1 = np.zeros((1 + input_size, neurons_hidden_layer))
# пороговые значения
W_1[0, :] = (np.random.randint(0, 3, size=(neurons_hidden_layer)))
# веса
W_1[1:, :] = (np.random.randint(-1, 2, size=(input_size, neurons_hidden_layer)))
print('W_1:', W_1)

# матрица весов и порогов второго слоя
W_2 = np.random.randint(0, 2, size=(1 + neurons_hidden_layer, neurons_output_layer)).astype(np.float64)
print('W_2:', W_2)


def predict(X):
    W_1_out = np.where((np.dot(X, W_1[1:, :]) + W_1[0, :]) >= 0.0, 1, -1).astype(np.float64)
    W_2_out = np.where((np.dot(W_1_out, W_2[1:, :]) + W_2[0, :]) >= 0.0, 1, -1).astype(np.float64)
    return W_2_out, W_1_out


# количетсво обучающих итераций
n_iter = 0
# шаг обучения
step = 0.01
# количество
check_iter = 5

# список для хранения  матрицы весов второго слоя
list_w_2_weights = [];

# обучение
while (True):

    for x_input, expected, j in zip(X, y, range(X.shape[0])):
        W_2_out, W_1_out = predict(x_input)
        W_2[1:] += ((step * (expected - W_2_out)) * W_1_out).reshape(-1, 1)
        W_2[0] += step * (expected - W_2_out)

    # список матрицы весов второго слоя
    list_w_2_weights.append(W_2.tobytes())

    W_2_out, W_1_out = predict(X)
    sum_errors = sum(W_2_out.reshape(-1) - y)
    print('sum_errors', sum_errors)
    if (sum_errors == 0):
        print('Все примеры обучающей выборки решены:')
        break

    # проверка наличия дубликатов в списке весов
    break_out_flag = False
    for item in list_w_2_weights:
        if list_w_2_weights.count(item) > 1:
            print('Повторение весов:')
            break_out_flag = True
            break
    if break_out_flag:
        break

W_2_out, W_1_out = predict(X)
sum_errors = sum(W_2_out.reshape(-1) - y)
print('sum_errors', sum_errors)
