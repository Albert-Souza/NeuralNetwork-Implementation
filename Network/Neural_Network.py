import numpy as np

class Network:
    weights = None
    activation_function = None
    activation_function_prime = None
    
    
    def __init__(self, shape, activation_function='sigmoid'):
        self.weights = [None] + [np.random.randn(j, k+1) for j, k in zip(shape[1:], shape[:-1])]
        if activation_function == 'sigmoid':
            self.activation_function = self.sigmoid
            self.activation_function_prime = self.sigmoid_prime
        elif activation_function == 'linear':
            self.activation_function = self.linear
            self.activation_function_prime = self.linear_prime
        
    
    def train(self, X, Y, training_rate=0.01, batch_size=None, epochs=1, log=False, log_name=None):
        if not batch_size:
            batch_size = X.shape[0]

        batches_X, batches_Y = self.create_batches(X, Y, batch_size)

        if log:
            log_file = open(log_name, 'w')
            log_file = open(log_name, 'a')
        else:
            log_file = None

        for epoch in range(1, epochs+1):
            if log:
                print(f'-- EPOCH {epoch} --\n', file=log_file)

            for batch_number, batch_X, batch_Y in zip(range(1, len(batches_X)+1), batches_X, batches_Y):
                batch_gradient, total_error = self.backpropagation(batch_X.T, batch_Y.T)
                self.weights = self.gradient_descent(batch_gradient, training_rate, batch_size)
                
                if log:
                    
                    print(f'Mini-Batch {batch_number}\n', file=log_file)
                    for l in range(1, len(self.weights)):
                        print(f'Weights {l}:\n{self.weights[l]}\n', file=log_file)
                    print(f'Total Error: {total_error}\n\n---------------\n', file=log_file)

        if log_file:
            log_file.close()

    def evaluate(self, X):
        a = np.pad(X, ((0,0), (0,1)), 'constant',  constant_values=1).T
        for weight in self.weights[1:]:
            z = weight @ a
            a = np.pad(self.activation_function(z), ((0,1), (0,0)), 'constant',  constant_values=1)

        return a[:-1].T
    
    
    def backpropagation(self, X, Y):
        zs, activations = self.feedforward(X)
        final_z = zs[-1]
        final_output = activations[-1][:-1]
        final_error = self.cost_derivative(Y, final_output)*self.activation_function_prime(final_z)
        final_gradient = final_error @ activations[-2].T


        errors = [final_error]
        gradients = [final_gradient]
        for l in range(2, len(activations)):
            error = self.weights[-l+1].T[:-1] @ errors[-1] * self.activation_function_prime(zs[-l])
            gradient = error @ activations[-l-1].T
            errors = [error] + errors
            gradients = [gradient] + gradients
        errors = [None] + errors
        gradients = [None] + gradients

        return gradients, final_error.sum()
        
    
    def feedforward(self, X):
        zs = [None]
        activations = [np.pad(X, ((0,1), (0,0)), 'constant', constant_values=1)]
        for weight in self.weights[1:]:
            z = weight @ activations[-1]
            activation = np.pad(self.activation_function(z), ((0,1), (0,0)), 'constant', constant_values=1)
            zs.append(z)
            activations.append(activation)
    
        return zs, activations
    

    def gradient_descent(self, gradients, training_rate, batch_size):
        new_weights = [None]
        for l in range(1, len(gradients)):
            new_weights.append(self.weights[l] - (training_rate/batch_size)*gradients[l])

        return new_weights


    def cost(self, Y, output):
        return np.square(Y-output)/2
    

    def cost_derivative(self, Y, output):
        return output-Y


    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    
    def sigmoid_prime(self, z):
        sigmoid = self.sigmoid(z)
        return sigmoid*(1-sigmoid)
    

    def linear(self, z):
        return z
    
    
    def linear_prime(self, z):
        return np.ones_like(z)
    
    
    def create_batches(self, X, Y, batch_size):
        num_samples = X.shape[0]
        new_order_indexes = np.random.permutation(num_samples)
        new_order_X, new_order_Y = X[new_order_indexes], Y[new_order_indexes]
        batches_X, batches_Y = [], []
        for i in range(0, num_samples, batch_size):
            batches_X.append(new_order_X[i:i+batch_size])
            batches_Y.append(new_order_Y[i:i+batch_size])
        
        return batches_X, batches_Y
