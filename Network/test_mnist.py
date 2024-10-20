import numpy as np
from Neural_Network import Network
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

X, Y = zip(*training_data)
X, Y = np.array(X), np.array(Y)
X, Y = X[:,:,0], Y[:,:,0]

network = Network([784, 30, 10], activation_function='sigmoid')
network.train(X, Y, training_rate=3, batch_size=10, epochs=5)

X_val, Y_val = zip(*validation_data)

X_val, Y_val = np.array(X_val), np.array(Y_val)
X_val = X_val[:,:,0]
Y_val = np.reshape(Y_val, (len(Y_val),1))

Y_hat = network.evaluate(X_val)

hits, misses = 0, 0
for y_v, y_h in zip(Y_val, Y_hat):
    if y_v == np.argmax(y_h):
        hits += 1
    else:
        misses += 1
    
print(f'Hits: {hits}, Misses: {misses}')
print(f'Accuracy: {hits/(hits+misses)}')
