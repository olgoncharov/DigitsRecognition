import numpy as np
from scipy.optimize import minimize as sc_minimize

class NeuralNetworkLayer():
    """
    Реализует слой нейронной сети.

    Свойства
    --------
    number_of_units: int
        Количество нейронов.
    index: int
        Номер слоя по порядку в структуре нейронной сети.
    theta: ndarray
        Матрица весов.
    a_term: ndarray, None
        Значения функции активации a(z).
    z_term: ndarray, None
        Аргумент функции активации z.
    delta: ndarray, None
        Вектор ошибок.
    derivate: ndarray, None
        Значения частных производных, рассчитанных по методу обратного распространения ошибки (backpropagation)
    """

    def __init__(self, number_of_units, index):
        """
        Параметры
        ---------
        number_of_units: int
            Количество нейронов.
        index: int
            Номер слоя по порядку в структуре нейронной сети.
        """
        self.number_of_units = number_of_units
        self.index = index

        # Эти свойства будут заполняться при обучении и прогнозировании
        self.theta = None
        self.a_term = None
        self.z_term = None
        self.delta = None
        self.derivate = None

    def initialize_weights(self, number_of_outputs):
        """Инициализирует матрицу весов случайными числами. Выполняется перед обучением."""
        eps = np.round(np.sqrt(6) / np.sqrt(self.number_of_units + number_of_outputs), 2)
        self.theta = np.random.rand(number_of_outputs, self.number_of_units + 1) * 2 * eps - eps


class NeuralNetwork():
    """
    Реализует нейронную сеть прямого распространения (feed forward neural network) с логистической функцией активации.

    Свойства
    --------
    layers: list
        Список объектов NeuralNetworkLayer, описывающих слои нейронной сети.
    """

    def __init__(self, *args):
        """
        Параметры
        ---------
        *args
            В качестве аргументов передаются целые числа, обозначающие количество нейронов на каждом слое сети.
            Количество переданных чисел равняется количеству слоев создаваемой нейронной сети.

        Примечание
        ----------
        Класс позволяет создать нейронную сеть не менее, чем с тремя слоями - один входной, один скрытый и один
        выходной. Если при инициализации указано менее трех аргументов, то будет вызвано исключение.

        Примеры
        -------
        Создание 4-х слойной нейронной сети с 400 входами, 50 нейронами на 2-м слое,  25 нейронами на 3-м слое и 10 выходами:

        >> net = NeuralNetwork(400, 50, 25, 10)
        """
        assert \
            len(args) > 2 and all(isinstance(item, int) for item in args),\
            'Переданы неверные параметры для инициализации нейронной сети'

        self.layers = [NeuralNetworkLayer(s, ind) for ind, s in enumerate(args)]

    def __iter__(self):
        return NeuralNetworkIterator(self.layers)

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, item):
        return self.layers[item]

    def set_weights(self, unroll_theta):
        """
        Заполняет матрицы весов для каждого слоя.

        Параметры
        ---------
        unroll_theta: ndarray
            Одномерный массив, содержащий элементы матриц весов каждого слоя.
        """
        cursor = 0
        for layer in self[:-1]:
            # Количество строк матрицы весов = количество нейронов следующего слоя
            # количество столбцов = количество нейронов текущего слоя + 1 (добавляем bias unit)
            shape = (self[layer.index + 1].number_of_units, layer.number_of_units + 1)
            layer.theta = unroll_theta[cursor: cursor + shape[0] * shape[1]].reshape(shape, order='F')
            cursor += shape[0] * shape[1]

    def forward_propagation(self, X):
        """
        Рассчитывает функции активации a(z) и аргументы z на каждом слое для переданного входного сигнала.

        Параметры
        ---------
        X: ndarray
            Входной сигнал в виде матрицы m x n, где m - количество поданных на вход примеров,
            n - количество входов нейронной сети.
        """
        self[0].a_term = np.transpose(X)
        for layer in self[1:]:
            layer.z_term = np.dot(self[layer.index-1].theta, add_ones(self[layer.index-1].a_term, 0))
            layer.a_term = sigmoid(layer.z_term)

    def backpropagation(self, X, Y):
        """
        Рассчитывает ошибку и градиент на каждом слое, используя принцип обратного распространения ошибки.

        Параметры
        ---------
        X: ndarray
            Входной сигнал в виде матрицы m x n, где m - количество поданных на вход примеров,
            n - количество входов нейронной сети.
        Y: ndarray
            Ожидаемый выходной сигнал.
        """
        m = X.shape[0]
        num_labels = self[-1].number_of_units

        for layer in self[:-1]:
            layer.derivate = np.zeros(layer.theta.shape)

        for i in range(m):
            a_i = self[-1].a_term[:, i]
            y_i = np.zeros(num_labels, dtype='int')
            y_i[Y[i][0] - 1] = 1

            self[-1].delta = (a_i - y_i).reshape((num_labels, 1))

            for layer in self[-2: :-1]:
                a_l = add_ones(layer.a_term[:, i].reshape((1, layer.number_of_units)), 1)
                layer.derivate += np.dot(self[layer.index + 1].delta, a_l)
                if layer.index != 0:
                    z_l = layer.z_term[:, i]
                    layer.delta = np.dot(np.transpose(layer.theta), self[layer.index + 1].delta)[1:] * sigmoid_gradient(z_l).reshape(z_l.size, 1)

        for layer in self[0:-1]:
            layer.derivate /= m

    def gradient(self, X, Y, unroll_theta):
        """
        Рассчитывает градиент.

        Параметры
        ---------
        X: ndarray
            Входной сигнал в виде матрицы m x n, где m - количество поданных на вход примеров,
            n - количество входов нейронной сети.
        Y: ndarray
            Ожидаемый выходной сигнал.
        unroll_theta: ndarray
            Веса нейронной сети, представленные в виде одномерного массива.

        Возвращаемое значение
        ---------------------
        grad: ndarray
            Одномерный вектор частных производных.
        """
        self.set_weights(unroll_theta)
        self.forward_propagation(X)
        self.backpropagation(X, Y)

        return unroll_arrays(*[layer.derivate for layer in self.layers[0:-1]])

    def cost_function(self, X, Y, unroll_theta, lambda_=0):
        """
        Функция затрат.

        Параметры
        ---------
        X: ndarray
            Входной сигнал в виде матрицы m x n, где m - количество поданных на вход примеров,
            n - количество входов нейронной сети.
        Y: ndarray
            Ожидаемый выходной сигнал.
        unroll_theta: ndarray
            Веса нейронной сети, представленные в виде одномерного массива.
        lambda_: number
            Параметр регуляризации.
        """
        m = X.shape[0]
        num_labels = self[-1].number_of_units
        J = 0
        self.set_weights(unroll_theta)
        self.forward_propagation(X)

        for k in range(num_labels):
            if num_labels < 10:
                yk = np.array(Y == (k + 1), dtype='int').flatten()
                Hk = self[-1].a_term[k, :]
            elif k == 0:
                # в обучающей выборке цифра '0' помечена меткой 10
                yk = np.array(Y == 10, dtype='int').flatten()
                Hk = self[-1].a_term[9, :]
            else:
                yk = np.array(Y == k, dtype='int').flatten()
                Hk = self[-1].a_term[k - 1, :]
            J += np.sum(-yk * np.log(Hk) - ((1 - yk) * np.log(1 - Hk)))

        J /= m

        if lambda_:
            reg_term = 0
            for layer in self[:-1]:
                reg_term += np.sum(np.square(layer.theta[:, 1:]))

            reg_term *= lambda_ / 2 / m
            J += reg_term

        return J

    def train(self, X, Y, lambda_, max_iter=150):
        """
        Обучает нейронную сеть.

        Параметры
        ---------
        X: ndarray
            Входной сигнал обучающей выборки в виде матрицы m x n, где m - количество обучающих примеров,
            n - количество входов нейронной сети.
        Y: ndarray
            Ожидаемый выходной сигнал.
        lambda_: number
            Параметр регуляризации.
        max_iter: integer
            Максимальное количество итераций, используемых для оптимизации функции затрат методом сопряженных градиентов.
        """
        for layer in self[:-1]:
            layer.initialize_weights(self[layer.index + 1].number_of_units)

        unroll_theta = unroll_arrays(*[layer.theta for layer in self.layers[0:-1]])

        opt_result = sc_minimize(
            fun=lambda t: self.cost_function(X, Y, t, lambda_),
            x0=unroll_theta,
            method='CG',
            jac=lambda t: self.gradient(X, Y, t),
            options={'maxiter': max_iter}
        )

        self.set_weights(opt_result.x)

    def predict(self, X):
        """
        Выполняет прогноз для переданного входного сигнала.

        Параметры
        ---------
        X: ndarray
            Входной сигнал в виде матрицы m x n, где m - количество поданных на вход примеров,
            n - количество входов нейронной сети.
        """
        self.forward_propagation(X)
        return np.argmax(self[-1].a_term, axis=0) + 1

class NeuralNetworkIterator():
    """Итератор для класса NeuralNetwork."""
    def __init__(self, collection):
        self.collection = collection
        self.cursor = -1

    def __next__(self):
        self.cursor += 1
        if self.cursor < len(self.collection):
            return self.collection[self.cursor]
        else:
            raise StopIteration

def add_ones(arr, ax):
    """
    Добавляет строку или столбец с единицами в матрицу.

    Параметры
    ---------
    arr: array like
        Исходный массив, к которому необходимо присоединить строку / столбец с единицами.
    ax: int {0, 1}
        Ось, вдоль которой присоединяются единицы. Если 0 - добавляется строка сверху, если 1 - столбец слева.

    Возвращаемые значения
    ---------------------
    d: array
        Результат конкатенации единичного вектора и исходной матрицы
    """
    if ax == 0:
        return np.concatenate((np.ones((1, arr.shape[-1])), arr), axis=0)
    else:
        return np.concatenate((np.ones((arr.shape[0], 1)), arr), axis=1)


def unroll_arrays(*args):
    """Преобразует многомерные массивы в один одномерный массив."""
    return np.concatenate(tuple([arr.flatten(order='F') for arr in args]))


def sigmoid(z):
    """Вычисляет значение логистической функции."""
    return 1 / (1 + np.power(np.e, -z))


def sigmoid_gradient(z):
    """Вычисляет значение градиента логистической функции"""
    return sigmoid(z) * (1 - sigmoid(z))
