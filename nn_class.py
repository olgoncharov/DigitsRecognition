import numpy as np
from scipy.optimize import minimize as sc_minimize
import time
import functools


def measure_time(func):
    """Декортатор, реализующий вывод времени выполнения функции"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        finish_time = time.time()
        print(f'Время выполнения {func.__name__}: {finish_time - start_time:.3f} секунд')

    return wrapper


class NeuralNetworkLayer():
    """
    Реализует слой нейронной сети.

    Свойства
    --------
    number_of_units: int
        Количество нейронов.
    network: NeuralNetwork
        Нейронная сеть, которой принадлежит слой.
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

    def __init__(self, network, number_of_units, index):
        """
        Параметры
        ---------
        number_of_units: int
            Количество нейронов.
        index: int
            Номер слоя по порядку в структуре нейронной сети.
        """
        self.network = network
        self.number_of_units = number_of_units
        self.index = index

        # Эти свойства будут заполняться при обучении и прогнозировании
        self.theta = None
        self.a_term = None
        self.z_term = None
        self.delta = None
        self.derivate = None

    def __add__(self, other):
        assert isinstance(other, int)
        return self.network[self.index + other]

    def __sub__(self, other):
        assert isinstance(other, int)
        assert other <= self.index
        return self.network[self.index - other]


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
            Количество переданных аргументов равняется количеству слоев создаваемой нейронной сети.

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

        self.layers = [NeuralNetworkLayer(self, s, ind) for ind, s in enumerate(args)]

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
            shape = ((layer + 1).number_of_units, layer.number_of_units + 1)
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
            layer.z_term = np.dot(
                (layer - 1).theta,
                np.vstack((np.ones(((layer - 1).a_term.shape[1])), (layer - 1).a_term))
            )
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
        numbers = np.arange(1, 11)
        Y_binary = np.transpose((Y == numbers)).astype('int')

        self[-1].delta = self[-1].a_term - Y_binary
        for layer in self[-2: :-1]:
            a_l = np.vstack((np.ones((1, layer.a_term.shape[1])), layer.a_term))
            layer.derivate = np.dot((layer + 1).delta, np.transpose(a_l)) / X.shape[0]
            if layer.index != 0:
                layer.delta = np.dot(np.transpose(layer.theta), (layer + 1).delta)[1:] * sigmoid_gradient(layer.z_term)

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

    @measure_time
    def train(self, X, Y, lambda_, init_weights=None, max_iter=150):
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
        init_weights: ndarray
            Веса нейронной сети, используемые в качестве начального приближения для алгоритма оптимизации. Заданы в виде
            одномерного массива. Если начальные веса не заданы, то они инициализируются случайными значениями.
        max_iter: integer
            Максимальное количество итераций, используемых для оптимизации функции затрат методом сопряженных градиентов.
        """
        if init_weights is None:
            for layer in self[:-1]:
                layer.initialize_weights((layer + 1).number_of_units)
        else:
            self.set_weights(init_weights)

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


def unroll_arrays(*args):
    """Преобразует многомерные массивы в один одномерный массив."""
    return np.concatenate(tuple([arr.flatten(order='F') for arr in args]))


def sigmoid(z):
    """Вычисляет значение логистической функции."""
    return 1 / (1 + np.power(np.e, -z))


def sigmoid_gradient(z):
    """Вычисляет значение градиента логистической функции"""
    return sigmoid(z) * (1 - sigmoid(z))
