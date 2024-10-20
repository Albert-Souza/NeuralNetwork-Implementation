import numpy as np
from Neural_Network import Network

# Rede neural que soma e subtrai números entre 0 e 100
# F([x1, x2, ..., xn]) : [x1+x2+..+xn, x1-x2-..-xn]

# Dados

num_samples = 1000
num_entries = 3

X = np.random.randint(0, 100, (num_samples, num_entries))

Y_sum = np.sum(X, axis=1).reshape(num_samples, 1)
Y_diff = (X[:, 0] - np.sum(X[:, 1:], axis=1)).reshape(num_samples, 1)

Y = np.stack((Y_sum, Y_diff), axis=1).reshape(num_samples, 2)

# Treinamento

network = Network([num_entries, 2], activation_function='linear')
network.train(X, Y, training_rate=0.0001, batch_size=20, epochs=5)

# Teste

num_tests = 5
Xn = np.random.randint(0, 100, (num_tests, num_entries))
Yn = network.evaluate(Xn).round(0).astype('int')
for x, y in zip(Xn, Yn):
    print(f'{x}: {y}')
