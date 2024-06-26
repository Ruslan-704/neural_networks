import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt

# Функция активации сигмоида.
def sigmoid_activation(x):
    return 1 / (1 + math.exp(-x))

# Производная функции активации.
def sigmoid_derivative_activation(x):
    return sigmoid_activation(x) * (1 - sigmoid_activation(x))

# Возвденние в квадрат
def square(x):
    return x * x

def errors(result, expected_class):
    expected_vector = np.array([0, 0, 0])
    expected_vector[expected_class] = 1
    # MSE для каждого класса и складываем
    sum_errors = sum(square(result - expected_vector))
    return (sum_errors, np.array([result - expected_vector]))

square = np.vectorize(square)
size_input = 4
first_layer = 6
second_layer = 3
gradient_step = 0.08

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    df = df.iloc[np.random.permutation(len(df))]
    y = df.iloc[0:150, 4].values
    y = np.where(y == "Iris-setosa", 0, y)
    y = np.where(y == "Iris-versicolor", 1, y)
    y = np.where(y == "Iris-virginica", 2, y)

    X = df.iloc[0:150, 0:4].values

    W_1 = np.random.rand(size_input, first_layer)
    W_2 = np.random.randn(first_layer, second_layer)

    for x_input in X:
        print(x_input)

    vector_sigmoid_activation = np.vectorize(sigmoid_activation)
    vector_sigmoid_derivative_activation = np.vectorize(sigmoid_derivative_activation)

    ziped = zip(X, y)
    zipped_list = list(ziped)

    epoch = 4
    errors_epoch = []
    for i in range(epoch):
        error_epoch = 0
        random.shuffle(zipped_list)
        for j in range(1, len(zipped_list)):

            random_element = random.sample(zipped_list, 1)
            random_element = random_element[0]
            x_input, expected = random_element

            new_W_1 = np.zeros((size_input, first_layer))
            new_W_2 = np.zeros((first_layer, second_layer))
            sum_errors = 0
            o1 = x_input @ W_1
            o1_activated = vector_sigmoid_activation(o1)
            o2 = o1_activated @ W_2
            o2_activated = vector_sigmoid_activation(o2)
            error, expected_vector = errors(o2_activated, expected)
            sum_errors = sum_errors + error

            de_e = vector_sigmoid_derivative_activation(o2) * expected_vector
            de_dw2 = np.dot(np.array([o1_activated]).T, de_e)
            de_o1_activated = de_e @ W_2.T
            de_01 = vector_sigmoid_derivative_activation(o1) * de_o1_activated
            de_dw1 = np.array([x_input]).T @ de_01

            new_W_1 = new_W_1 + de_dw1
            new_W_2 = new_W_2 + de_dw2

            W_1 = W_1 - gradient_step * new_W_1
            W_2 = W_2 - gradient_step * new_W_2
            error_epoch += sum_errors
        print('error in epoch', sum_errors)
        errors_epoch.append(error_epoch)

    plt.plot(errors_epoch)
    plt.show()

    correct = 0
    incorrect = 0
    for x_input, expected in zipped_list:
        o1 = x_input @ W_1
        o1_activated = vector_sigmoid_activation(o1)
        o2 = o1_activated @ W_2
        o2_activated = vector_sigmoid_activation(o2)
        error, expected_vector = errors(o2_activated, expected)
        sum_errors = sum_errors + error
        if (np.argmax(o2_activated, axis=0) == expected):
            correct += 1
        else:
            incorrect += 1

    print('correct', correct)
    print('incorrect', incorrect)