import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

# Encode to onehot vector
def encode_onehot(dataset_input):
    if len(np.array(dataset_input).shape) == 1:
        return np.eye(np.array(dataset_input).max() + 1)[dataset_input]
    elif len(np.array(dataset_input).shape) == 2:
        return dataset_input

# Encode to list of scalars
def encode_scalar(dataset_input):
    if len(np.array(dataset_input).shape) == 1:
        return dataset_input
    elif len(np.array(dataset_input).shape) == 2:
        return np.argmax(dataset_input, axis=1)

# Calculate accuracy
def accuracy(model_input:np.ndarray, dataset_input:np.ndarray) -> float:
    y_model = encode_scalar(model_input),
    y_dataset = encode_scalar(dataset_input)                                         
    return np.mean(y_model == y_dataset) 

# Activation function ReLU
def ReLU(model_input, derivative=False):
    if derivative: 
        return (model_input > 0.0)
    return model_input * (model_input > 0.0)

# Activation function Sigmoid
def Sigmoid(model_input, derivative=False):
    sigmoid = 1.0 / (np.exp(-model_input) + 1.0)
    if derivative: 
        return sigmoid * (1.0 - sigmoid)
    return sigmoid

# Activation function Tanh
def Tanh(model_input, derivative=False):
    if derivative: 
        return 1.0 - np.tanh(model_input) ** 2
    return np.tanh(model_input)

# Activation function GCU
def GCU(model_input, derivative=False):
    if derivative: 
        return np.cos(model_input) - np.sin(model_input) * model_input
    return np.cos(model_input) * model_input

# Activation function Softmax
def Softmax(model_input, derivative=False):
    exponents = np.exp(model_input)
    if derivative:
        raise ValueError()
    return exponents / np.sum(exponents, axis=1, keepdims=True)

# Cost function CCE
def CCE(model_input, dataset_input, derivative=False):
    model_input_clipped = model_input - np.max(model_input, axis=1, keepdims=True)
    # Apply activation function
    model_input_activated = Softmax(model_input_clipped, derivative=False)
    # Apply cost function derivative
    if derivative:
        error_derivative = (model_input_activated - encode_onehot(dataset_input))
        cost_derivative = error_derivative / len(model_input_activated)
        return cost_derivative
    # Apply cost function
    error = -np.log(np.sum(np.clip(model_input_activated, 1e-7, 1-1e-7) * encode_onehot(dataset_input), axis=1, keepdims=True))
    cost = np.mean(error)

    return model_input_activated, cost, Regulization.regulization_cost

# Network layers:

class Dense:

    def __init__(self, input_size:int=0.0, output_size:int=0.0, 
                 weights:np.ndarray=None, bias:np.ndarray=None, activation=None, cost=None,
                 l1w:float=0.0, l1b:float=0.0, l2w:float=0.0, l2b:float=0.0) -> None:

        self.TYPE = "Dense"

        # Set layer weights if given or generate new ones based on input_size and output_size
        self.weights = np.random.uniform(-0.5, 0.5, (input_size, output_size)) if weights == None else weights
        # Set layer bias if given or generate new ones based on input_size and output_size
        self.bias = np.random.uniform(-0.1, 0.1, (1, output_size)) if bias == None else bias
        # Set activation function and cost function
        self.activation, self.cost = activation, cost

        # Momentum for weights for SDG/Adam optimizer
        self.vt_w = np.zeros_like(self.weights)
        # Momentum for bias for SDG/Adam optimizer 
        self.vt_b = np.zeros_like(self.bias) 
        # Cache for weights for AdaGrad/RMSprop/Adam optimizer
        self.cache_w = np.zeros_like(self.weights)
        # Cache for bias for AdaGrad/RMSprop/Adam optimizer  
        self.cache_b = np.zeros_like(self.bias)

        # L1 regulizer for weights and bias
        self.l1_weights, self.l1_bias = l1w, l1b
        # L2 regulizer for weights and bias
        self.l2_weights, self.l2_bias = l2w, l2b
        # L1 regulizer derivative for weights and bias 
        self.l1d_weights = np.ones_like(self.weights) 
        self.l1d_bias  = np.ones_like(self.bias)


    def forward(self, layer_input:np.ndarray, dataset_input:np.ndarray=None, compiled=False) -> np.ndarray:

        self.x, self.y = layer_input, dataset_input
        # Calculate unactivated neuron output
        self.z = self.x @ self.weights + self.bias

        # L1 & l2 regulizer
        Regulization.forward(self)

        # Calculate activated neuron output
        if compiled == False or self.cost == None: 
            # Calculate with activation function if not compiled 
            self.a = self.activation(self.z, derivative=False)

        elif compiled == True and self.cost != None:
            # Calculate with cost function if compiled and last layer
            self.a = self.cost(self.z, self.y, derivative=False)

        return self.a # Contains; softmax output, cost, regulization cost

    def backward(self, layer_input:np.ndarray, compiled=False) -> np.ndarray:

        # Calculate derivative of activated neuron output
        if compiled == False or self.cost == None:
            # Calculate with activation function if not compiled
            self.dc = self.activation(self.z, derivative=True)

        elif compiled == True and self.cost != None:
            # Calculate with cost function if compiled and last layer
            self.dc = self.cost(self.z, self.y, derivative=True)

        # Calculate derivative of unactivated neuron output
        self.dc_dz = self.dc * layer_input
        # Calculate derivative for weight
        self.dc_dw = self.x.T @ self.dc_dz
        # Calculate derivative for bias
        self.dc_db = np.sum(self.dc_dz, axis=0, keepdims=True)
        # Calculate derivative for input
        self.dc_dx = self.dc_dz @ self.weights.T

        # L1 & l2 regulizer
        Regulization.backward(self)
        
        return self.dc_dx 

class Regulization:

    @classmethod
    def forward(cls, layer):

        cls.regulization_cost = 0

        # Calculate regulization_cost for the layer
        cls.regulization_cost += layer.l1_weights * np.sum(np.abs(layer.weights))
        cls.regulization_cost += layer.l1_bias * np.sum(np.abs(layer.bias))
        cls.regulization_cost += layer.l2_weights * np.sum(layer.weights**2)
        cls.regulization_cost += layer.l2_bias * np.sum(layer.bias**2)

        return cls.regulization_cost

    @classmethod
    def backward(cls, layer):

        # Calculate derivative matrix of l1 regulizer of weights and bias
        layer.l1d_weights[layer.weights < 0] = -1
        layer.l1d_bias[layer.bias < 0] = -1

        # Add regulization l1 to derivative of weights and bias
        layer.dc_dw += layer.l1_weights * layer.l1d_weights
        layer.dc_db += layer.l1_bias * layer.l1d_bias

        # Add regulization l2 to derivative of weights and bias
        layer.dc_dw += 2 * layer.l2_weights * layer.weights
        layer.dc_db += 2 * layer.l2_bias * layer.bias

class Flatten:

    def __init__(self, input_shape:tuple) -> None:

        self.TYPE = "Flatten"

        self.input_shape = input_shape
    
    def forward(self, layer_input:np.ndarray) -> np.ndarray:

        # Amount of data in a sample
        self.data_size = np.prod(self.input_shape)
        # Amount of samples in a dataset
        self.sample_size = np.array(layer_input).shape[0]
        # Shape of flattened matrix
        self.output_shape = (self.sample_size, self.data_size)
        # Return reshaped input
        return np.array(layer_input).reshape(self.output_shape)

# Optimizer:

class SGD:
    
    def __init__(self, learning_rate:float=1.0, learning_decay:float=1e-3,
                 gamma:float=1e-7) -> None:

        self.learning_rate_init = learning_rate
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.gamma = gamma
    
    def optimize(self, layer, epoch:int) -> float:

        # Ajust learning rate based on delay
        self.learning_rate = self.learning_rate_init / (1.0 + self.learning_decay * epoch)
        # Calculate learning momentum 
        layer.vt_w = -self.learning_rate * layer.dc_dw + self.gamma * layer.vt_w 
        layer.vt_b = -self.learning_rate * layer.dc_db + self.gamma * layer.vt_b
        # Ajust weights and bias
        layer.weights += layer.vt_w
        layer.bias += layer.vt_b
        # Return learning rate for statistics
        return self.learning_rate

class AdaGrad:
    
    def __init__(self, learning_rate:float=0.05, learning_decay:float=1e-5, 
                 epsilon:float=1e-7) -> None:

        self.learning_rate_init = learning_rate
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.epsilon = epsilon

    def optimize(self, layer, epoch:int) -> float:

        # Ajust learning rate based on delay
        self.learning_rate = self.learning_rate_init / (1.0 + self.learning_decay * epoch)
        # Calculate learning cache
        layer.cache_w += layer.dc_dw ** 2
        layer.cache_b += layer.dc_db ** 2
        # Ajust weights and bias
        layer.weights += -self.learning_rate * layer.dc_dw / (np.sqrt(layer.cache_w) + self.epsilon)
        layer.bias += -self.learning_rate * layer.dc_db / (np.sqrt(layer.cache_b) + self.epsilon) 
        # Return learning rate for statistics
        return self.learning_rate

class RMSprop:
    
    def __init__(self, learning_rate:float=0.001, learning_decay:float=0.0,
                 epsilon:float=1e-7 , rho:float=0.9) -> None:

        self.learning_rate_init = learning_rate
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.epsilon = epsilon
        self.rho = rho 

    def optimize(self, layer, epoch:int) -> float:

        # Ajust learning rate based on delay
        self.learning_rate = self.learning_rate_init / (1.0 + self.learning_decay * epoch)
        # Calculate learning cache
        layer.cache_w = self.rho * layer.cache_w + (1.0 - self.rho) * layer.dc_dw ** 2
        layer.cache_b = self.rho * layer.cache_b + (1.0 - self.rho) * layer.dc_db ** 2
        # Ajust weights and bias
        layer.weights += -self.learning_rate * layer.dc_dw / (np.sqrt(layer.cache_w) + self.epsilon)
        layer.bias += -self.learning_rate * layer.dc_db / (np.sqrt(layer.cache_b) + self.epsilon)
        # Return learning rate for statistics
        return self.learning_rate

class Adam:
    
    def __init__(self, learning_rate:float=0.05, learning_decay:float=1e-5,
                 epsilon:float=1e-7, beta_1:float=0.9, beta_2:float=0.999) -> None:

        self.learning_rate_init = learning_rate
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2 
    
    def optimize(self, layer, epoch:int) -> float:

        # Ajust learning rate based on delay
        self.learning_rate = self.learning_rate_init / (1.0 + self.learning_decay * epoch)
        # Calculate learning momentum 
        layer.vt_w = self.beta_1 * layer.vt_w + (1.0 - self.beta_1) * layer.dc_dw
        layer.vt_b = self.beta_1 * layer.vt_b + (1.0 - self.beta_1) * layer.dc_db
        # Calculate learning cache
        layer.cache_w = self.beta_2 * layer.cache_w + (1.0 - self.beta_2) * layer.dc_dw ** 2.0
        layer.cache_b = self.beta_2 * layer.cache_b + (1.0 - self.beta_2) * layer.dc_db ** 2.0
        # Correct learning momentum
        vt_w_corrected = layer.vt_w / (1.0 - self.beta_1 ** (epoch + 1.0)) 
        vt_b_corrected = layer.vt_b / (1.0 - self.beta_1 ** (epoch + 1.0))
        # Correct learning cache
        cache_w_corrected = layer.cache_w / (1.0 - self.beta_2 ** (epoch + 1.0))
        cache_b_corrected = layer.cache_b / (1.0 - self.beta_2 ** (epoch + 1.0)) 
        # Ajust weights and bias
        layer.weights += -self.learning_rate * vt_w_corrected / (np.sqrt(cache_w_corrected) + self.epsilon)
        layer.bias += -self.learning_rate * vt_b_corrected / (np.sqrt(cache_b_corrected) + self.epsilon)
        # Return learning rate for statistics
        return self.learning_rate

# Network:

class Network:

    def __init__(self, *layers, optimizer=Adam()) -> None:

        self.layers = layers
        self.optimizer = optimizer
        self.log_acc, self.log_val_acc = [], []
        self.log_cost, self.log_val_cost = [], []
        self.log_reg_cost, self.log_val_reg_cost = [], []
    

    def run(self, model_input:np.ndarray) -> np.ndarray:

        # Layer input forward
        self.layer_input_fw = model_input

        # Forward pass for model input
        for layer in self.layers:
            if layer.TYPE == "Dense":
                self.layer_input_fw = layer.forward(self.layer_input_fw, compiled=False)
            elif layer.TYPE == "Flatten":
                self.layer_input_fw = layer.forward(self.layer_input_fw)
            elif layer.TYPE == "Convolution":
                self.layer_input_fw = layer.forward(self.layer_input_fw)
            elif layer.TYPE == "Pooling":
                self.layer_input_fw = layer.forward(self.layer_input_fw) 
        self.output, self.cost, self.regulization_cost = self.layer_input_fw
        return self.output, self.cost, self.regulization_cost
    
    def train(self, model_input:np.ndarray, dataset_input:np.ndarray, val_model_input:np.ndarray, val_dataset_input:np.ndarray,
              epochs:int, interval:int):

        for epoch in range(epochs):

            # Training layer input forward
            self.layer_input_fw = model_input
            # Training layer input backward
            self.layer_input_bw = 1.0
            # Validation layer input forward
            self.val_layer_input_fw = val_model_input

            # Forward pass for valisation model input
            for layer in self.layers:
                if layer.TYPE == "Dense":
                    self.val_layer_input_fw = layer.forward(self.val_layer_input_fw, val_dataset_input, compiled=True)
                elif layer.TYPE == "Flatten":
                    self.val_layer_input_fw = layer.forward(self.val_layer_input_fw)
                elif layer.TYPE == "Convolution":
                    pass
                elif layer.TYPE == "Pooling":
                    pass      
            self.val_output, self.val_cost, self.val_regulization_cost = self.val_layer_input_fw

            # Forward pass for training model input
            for layer in self.layers:
                if layer.TYPE == "Dense":
                    self.layer_input_fw = layer.forward(self.layer_input_fw, dataset_input, compiled=True)
                elif layer.TYPE == "Flatten":
                    self.layer_input_fw = layer.forward(self.layer_input_fw)
                elif layer.TYPE == "Convolution":
                    pass
                elif layer.TYPE == "Pooling":
                    pass
            self.output, self.cost, self.regulization_cost = self.layer_input_fw

            # Backward pass for training model input
            for layer in reversed(self.layers):
                if layer.TYPE == "Dense":
                    self.layer_input_bw = layer.backward(self.layer_input_bw, compiled=True)
                elif layer.TYPE == "Flatten":
                    pass
                elif layer.TYPE == "Convolution":
                    pass
                elif layer.Type == "Pooling":
                    pass
                
            # Optimization process for training model input
            for layer in reversed(self.layers):
                if layer.TYPE == "Dense":
                    self.lr = self.optimizer.optimize(layer, epoch)
                elif layer.TYPE == "Flatten":
                    pass
                elif layer.TYPE == "Convolution":
                    pass
                elif layer.Type == "Pooling":
                    pass

            if epoch % interval == 0:

                # Accuracy
                self.accuracy = accuracy(self.output, dataset_input)
                self.val_accuracy = accuracy(self.val_output, val_dataset_input)
                self.log_acc.append(self.accuracy)
                self.log_val_acc.append(self.val_accuracy)

                # Cost
                self.log_cost.append(self.cost)
                self.log_val_cost.append(self.val_cost)  
                self.log_reg_cost.append(self.regulization_cost)
                self.log_val_reg_cost.append(self.val_regulization_cost)

                # Output result of training epoch
                print(f"| acc: {self.accuracy:.1%}", end=" ") 
                print(f"val_acc: {self.val_accuracy:.1%} |", end=" ") 
                print(f"| cost: {self.cost:.2f}", end=" ")
                print(f"reg_cost: {self.regulization_cost:.2f}", end=" ")
                print(f"val_cost: {self.val_cost:.2f}", end=" ")
                print(f"val_reg_cost: {self.val_regulization_cost:.2f} |", end=" ")
                print(f"lr: {self.lr:.3f}", end=" ")
                print(f"Epoch: [{epoch} / {epochs}]")

        # Plot result of network training
        plt.subplot(211)
        plt.title('Trainingsprozess')
        plt.plot(np.arange(len(self.log_cost)), self.log_cost, label="cost")
        plt.plot(np.arange(len(self.log_val_cost)), self.log_val_cost, label="validation cost") 
        plt.ylabel("Fehler")
        plt.legend()
        plt.subplot(212)
        plt.plot(np.arange(len(self.log_acc)), self.log_acc, label="accuracy")
        plt.plot(np.arange(len(self.log_val_acc)), self.log_val_acc, label="validation accuracy") 
        plt.xlabel("Epoche")
        plt.ylabel("Präzision")
        plt.legend()  
        plt.show()


# When I wrote this code only I and God knew what it was doing. Now God alone knows.
# - Jake Paul Richter
 