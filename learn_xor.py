import numpy as np

class Layer:
    def __init__(self):
        self.input=None
        self.output=None
    
    def forward_propagation(self, input_data):
        raise NotImplementedError
    def backward_propagation(self,output_error, learning_rate):
        raise NotImplementedError

class FCLayer(Layer):
    def __init__(self,input_size, output_size):
        self.weights = np.random.rand(input_size,output_size)-0.5
        self.bias = np.random.rand(1,output_size)-0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
        
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size




class Network:
    def __init__(self,loss, loss_prime):
        self.layers = []
        self.loss = loss
        self.loss_prime = loss_prime
    
    def add(self, layer):
        self.layers.append(layer)
    
    def predict(self, input_data):
        result = np.zeros((len(input_data),1))
        for i in range(len(input_data)):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result[i]=output[0][0]
            
        return result
    
    def fit(self, X,y,epochs=10,learning_rate=0.1):
        for i in range(epochs):
            err = 0
            for j in range(len(X)):
                output = X[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                
                err += self.loss(y[j],output)
                
                error = self.loss_prime(y[j],output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error,learning_rate)
                
            err /= len(X)
            print('epoch {}/{} error = {}'.format(i+1,epochs,err))

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network(mse,mse_prime)
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)








