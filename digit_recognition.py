from nn_class import NeuralNetwork
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt

data = loadmat('ex4data1.mat')

# Разделяем входные данные на два набора: обучающую выборку (training set) и тестовую (test set) в пропорции 80/20
m = int(data['y'].size)
m_training = int(np.round(m * 0.8, 0))
m_test = m - m_training

# Элементы для тестовой выборки из общего набора берем в случайном порядке
random_indexes = np.arange(m)
np.random.shuffle(random_indexes)
X_test = data['X'][random_indexes[:m_test], :]
Y_test = data['y'][random_indexes[:m_test], :]
X_training = data['X'][random_indexes[m_test:], :]
Y_training = data['y'][random_indexes[m_test:], :]

nn = NeuralNetwork(400, 50, 10)
nn.train(X_training, Y_training, 1, max_iter=250)

pred_training = nn.predict(X_training)
accuracy_training = np.mean(np.array(pred_training == Y_training.flatten(), dtype='int')) * 100
print(f'Точность распознавания на обучающей выборке: {accuracy_training}')

pred_test = nn.predict(X_test)
accuracy_test = np.mean(np.array(pred_test == Y_test.flatten(), dtype='int')) * 100
print(f'Точность распознавания на тестовой выборке: {accuracy_test}')

# Визуализируем случайные 100 примеров из тестовой выборки
random_indexes = np.arange(m_test)
np.random.shuffle(random_indexes)

fig, ax = plt.subplots(5, 20)

for i in range(5):
    for j in range(20):
        index = random_indexes[i*20 + j]
        display_array = np.transpose(X_test[index].reshape(20, 20))
        ax[i, j].imshow(display_array, cmap='Greys_r')
        ax[i, j].axis('off')

        nn_answer = 0 if pred_test[index] == 10 else pred_test[index]
        correct_answer = 0 if Y_test[index][0] == 10 else Y_test[index][0]
        ax[i, j].set_title(f'{nn_answer} / {correct_answer}',
                           color='green' if nn_answer == correct_answer else 'red')

fig.suptitle(f'Точность распознавания на тестовой выборке: {accuracy_test}%')