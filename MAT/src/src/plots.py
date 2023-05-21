"""
THIS PROGRAM IST ONLY USED FOR THE GENERATION OF PLOTS AND FIGURES FOR THE WRITTEN PART OF THIS MARTURA WORK.
THEREFORE THIS PART DOES NOT CONCERN THE IMPLEMENTATION OF THE ALGORITHM ITSELF OR THE APPLICATION OF THE NEURAL NETWORK ON A CERTAIN DATASET.
"""

import numpy as np

import matplotlib.pyplot as plt
from keras.datasets import mnist

import datasets as data
import network as nn


# -> SPIRAL DATASET
# --> ReLU

X, Y = data.Spiral.generate(1000, 3)
X_val, Y_val = data.Spiral.generate(300, 3)
 
modelReLU_spiral = nn.Network(nn.Dense(input_size=2, output_size=128, activation=nn.ReLU, cost=None, l2w=5e-4, l2b=5e-4),
                   nn.Dense(input_size=128, output_size=3, activation=nn.Softmax, cost=nn.CCE, l2w=5e-5, l2b=5e-4),
                   optimizer=nn.Adam())                                  

modelReLU_spiral.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
            epochs=10_000, interval=25) 

# --> Sigmoid

X, Y = data.Spiral.generate(1000, 3)
X_val, Y_val = data.Spiral.generate(300, 3)
 
modelSigmoid_spiral = nn.Network(nn.Dense(input_size=2, output_size=128, activation=nn.Sigmoid, cost=None, l2w=5e-4, l2b=5e-4),
                   nn.Dense(input_size=128, output_size=3, activation=nn.Softmax, cost=nn.CCE, l2w=5e-5, l2b=5e-4),
                   optimizer=nn.Adam())                                  

modelSigmoid_spiral.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
            epochs=10_000, interval=25) 

# --> Tanh

X, Y = data.Spiral.generate(1000, 3)
X_val, Y_val = data.Spiral.generate(300, 3)
    
modelTanh_spiral = nn.Network(nn.Dense(input_size=2, output_size=128, activation=nn.Tanh, cost=None, l2w=5e-4, l2b=5e-4),
                   nn.Dense(input_size=128, output_size=3, activation=nn.Softmax, cost=nn.CCE, l2w=5e-5, l2b=5e-4),
                   optimizer=nn.Adam())                                  

modelTanh_spiral.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
            epochs=10_000, interval=25) 

# --> GCU

X, Y = data.Spiral.generate(1000, 3)
X_val, Y_val = data.Spiral.generate(300, 3)
     
modelGCU_spiral = nn.Network(nn.Dense(input_size=2, output_size=128, activation=nn.GCU, cost=None, l2w=5e-4, l2b=5e-4),
                   nn.Dense(input_size=128, output_size=3, activation=nn.Softmax, cost=nn.CCE, l2w=5e-5, l2b=5e-4),
                   optimizer=nn.Adam())                                  

modelGCU_spiral.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
            epochs=10_000, interval=25) 

# -> MNSIT

# --> ReLU

X, Y, X_val, Y_val = data.Mnist.generate(samples_training_data=10_000, samples_val_data=3_000)

modelReLU_mnist = nn.Network(nn.Flatten(input_shape=(28, 28)),
                   nn.Dense(input_size=784, output_size=64, activation=nn.ReLU, cost=None, l2w=5e-4, l2b=5e-4),
                   nn.Dense(input_size=64, output_size=10, activation=nn.Softmax, cost=nn.CCE, l2w=5e-4, l2b=5e-4),
                   optimizer=nn.Adam())

modelReLU_mnist.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
            epochs=400, interval=1)

# --> Sigmoid

X, Y, X_val, Y_val = data.Mnist.generate(samples_training_data=10_000, samples_val_data=3_000)

modelSigmoid_mnist = nn.Network(nn.Flatten(input_shape=(28, 28)),
                   nn.Dense(input_size=784, output_size=64, activation=nn.Sigmoid, cost=None, l2w=5e-4, l2b=5e-4),
                   nn.Dense(input_size=64, output_size=10, activation=nn.Softmax, cost=nn.CCE, l2w=5e-4, l2b=5e-4),
                   optimizer=nn.Adam())

modelSigmoid_mnist.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
            epochs=400, interval=1)

# --> Tanh

X, Y, X_val, Y_val = data.Mnist.generate(samples_training_data=10_000, samples_val_data=3_000)

modelTanh_mnist = nn.Network(nn.Flatten(input_shape=(28, 28)),
                   nn.Dense(input_size=784, output_size=64, activation=nn.Tanh, cost=None, l2w=5e-4, l2b=5e-4),
                   nn.Dense(input_size=64, output_size=10, activation=nn.Softmax, cost=nn.CCE, l2w=5e-4, l2b=5e-4),
                   optimizer=nn.Adam())

modelTanh_mnist.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
            epochs=400, interval=1)

# --> GCU

X, Y, X_val, Y_val = data.Mnist.generate(samples_training_data=10_000, samples_val_data=3_000)

modelGCU_mnist = nn.Network(nn.Flatten(input_shape=(28, 28)),
                   nn.Dense(input_size=784, output_size=64, activation=nn.GCU, cost=None, l2w=5e-4, l2b=5e-4),
                   nn.Dense(input_size=64, output_size=10, activation=nn.Softmax, cost=nn.CCE, l2w=5e-4, l2b=5e-4),
                   optimizer=nn.Adam())

modelGCU_mnist.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
            epochs=400, interval=1)



# -> SPIRAL

# --> Adam

X, Y = data.Spiral.generate(1000, 3)
X_val, Y_val = data.Spiral.generate(300, 3)
 
modelAdam_spiral = nn.Network(nn.Dense(input_size=2, output_size=128, activation=nn.ReLU, cost=None, l2w=5e-4, l2b=5e-4),
                   nn.Dense(input_size=128, output_size=3, activation=nn.Softmax, cost=nn.CCE, l2w=5e-5, l2b=5e-4),
                   optimizer=nn.Adam())                                  

modelAdam_spiral.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
            epochs=10_000, interval=25) 

# --> RMSprop

X, Y = data.Spiral.generate(1000, 3)
X_val, Y_val = data.Spiral.generate(300, 3)
 
modelRMSprop_spiral = nn.Network(nn.Dense(input_size=2, output_size=128, activation=nn.ReLU, cost=None, l2w=5e-4, l2b=5e-4),
                   nn.Dense(input_size=128, output_size=3, activation=nn.Softmax, cost=nn.CCE, l2w=5e-5, l2b=5e-4),
                   optimizer=nn.RMSprop())                                  

modelRMSprop_spiral.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
            epochs=10_000, interval=25) 

# --> AdaGrad

X, Y = data.Spiral.generate(1000, 3)
X_val, Y_val = data.Spiral.generate(300, 3)
    
modelAdaGrad_spiral = nn.Network(nn.Dense(input_size=2, output_size=128, activation=nn.ReLU, cost=None, l2w=5e-4, l2b=5e-4),
                   nn.Dense(input_size=128, output_size=3, activation=nn.Softmax, cost=nn.CCE, l2w=5e-5, l2b=5e-4),
                   optimizer=nn.AdaGrad())                                  

modelAdaGrad_spiral.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
            epochs=10_000, interval=25) 

# --> SGD

X, Y = data.Spiral.generate(1000, 3)
X_val, Y_val = data.Spiral.generate(300, 3)
     
modelSGD_spiral = nn.Network(nn.Dense(input_size=2, output_size=128, activation=nn.ReLU, cost=None, l2w=5e-4, l2b=5e-4),
                   nn.Dense(input_size=128, output_size=3, activation=nn.Softmax, cost=nn.CCE, l2w=5e-5, l2b=5e-4),
                   optimizer=nn.SGD())                                  

modelSGD_spiral.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
            epochs=10_000, interval=25) 


# -> MNSIT

# --> Adam

X, Y, X_val, Y_val = data.Mnist.generate(samples_training_data=10_000, samples_val_data=3_000)

modelAdam_mnist = nn.Network(nn.Flatten(input_shape=(28, 28)),
                   nn.Dense(input_size=784, output_size=64, activation=nn.ReLU, cost=None, l2w=5e-4, l2b=5e-4),
                   nn.Dense(input_size=64, output_size=10, activation=nn.Softmax, cost=nn.CCE, l2w=5e-4, l2b=5e-4),
                   optimizer=nn.Adam())

modelAdam_mnist.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
            epochs=400, interval=1)

# --> RMSprop

X, Y, X_val, Y_val = data.Mnist.generate(samples_training_data=10_000, samples_val_data=3_000)

modelRMSprop_mnist = nn.Network(nn.Flatten(input_shape=(28, 28)),
                   nn.Dense(input_size=784, output_size=64, activation=nn.ReLU, cost=None, l2w=5e-4, l2b=5e-4),
                   nn.Dense(input_size=64, output_size=10, activation=nn.Softmax, cost=nn.CCE, l2w=5e-4, l2b=5e-4),
                   optimizer=nn.RMSprop())

modelRMSprop_mnist.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
            epochs=400, interval=1)

# --> AdaGrad

X, Y, X_val, Y_val = data.Mnist.generate(samples_training_data=10_000, samples_val_data=3_000)

modelAdaGrad_mnist = nn.Network(nn.Flatten(input_shape=(28, 28)),
                   nn.Dense(input_size=784, output_size=64, activation=nn.ReLU, cost=None, l2w=5e-4, l2b=5e-4),
                   nn.Dense(input_size=64, output_size=10, activation=nn.Softmax, cost=nn.CCE, l2w=5e-4, l2b=5e-4),
                   optimizer=nn.AdaGrad())

modelAdaGrad_mnist.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
            epochs=400, interval=1)

# --> SGD

X, Y, X_val, Y_val = data.Mnist.generate(samples_training_data=10_000, samples_val_data=3_000)

modelSGD_mnist = nn.Network(nn.Flatten(input_shape=(28, 28)),
                   nn.Dense(input_size=784, output_size=64, activation=nn.ReLU, cost=None, l2w=5e-4, l2b=5e-4),
                   nn.Dense(input_size=64, output_size=10, activation=nn.Softmax, cost=nn.CCE, l2w=5e-4, l2b=5e-4),
                   optimizer=nn.SGD())

modelSGD_mnist.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
            epochs=400, interval=1)


# Plot:

# Spiral activation functions comparison
# accuracy
#plt.plot(np.arange(len(modelReLU_spiral.log_acc)), modelReLU_spiral.log_acc, label="ReLU Aktivierungsfunktion")
#plt.plot(np.arange(len(modelSigmoid_spiral.log_acc)), modelSigmoid_spiral.log_acc, label="Sigmoid Aktivierungsfunktion")
#plt.plot(np.arange(len(modelTanh_spiral.log_acc)), modelTanh_spiral.log_acc, label="Tanh Aktivierungsfunktion")
#plt.plot(np.arange(len(modelGCU_spiral.log_acc)), modelGCU_spiral.log_acc, label="CCE Aktivierungsfunktion")
#plt.legend()
#plt.show()

# cost
#plt.plot(np.arange(len(modelReLU_spiral.log_acc)), modelReLU_spiral.log_cost, label="ReLU Aktivierungsfunktion")
#plt.plot(np.arange(len(modelSigmoid_spiral.log_acc)), modelSigmoid_spiral.log_cost, label="Sigmoid Aktivierungsfunktion")
#plt.plot(np.arange(len(modelTanh_spiral.log_acc)), modelTanh_spiral.log_cost, label="Tanh Aktivierungsfunktion")
#plt.plot(np.arange(len(modelGCU_spiral.log_acc)), modelGCU_spiral.log_cost, label="CCE Aktivierungsfunktion")
#plt.legend()
#plt.show()

# Mnist activation functions comparison
# accuracy
#plt.plot(np.arange(len(modelReLU_mnist.log_acc)), modelReLU_mnist.log_acc, label="ReLU Aktivierungsfunktion")
#plt.plot(np.arange(len(modelSigmoid_mnist.log_acc)), modelSigmoid_mnist.log_acc, label="Sigmoid Aktivierungsfunktion")
#plt.plot(np.arange(len(modelTanh_mnist.log_acc)), modelTanh_mnist.log_acc, label="Tanh Aktivierungsfunktion")
#plt.plot(np.arange(len(modelGCU_mnist.log_acc)), modelGCU_mnist.log_acc, label="CCE Aktivierungsfunktion")
#plt.legend()
#plt.show()

# cost
#plt.plot(np.arange(len(modelReLU_mnist.log_acc)), modelReLU_mnist.log_cost, label="ReLU Aktivierungsfunktion")
#plt.plot(np.arange(len(modelSigmoid_mnist.log_acc)), modelSigmoid_mnist.log_cost, label="Sigmoid Aktivierungsfunktion")
#plt.plot(np.arange(len(modelTanh_mnist.log_acc)), modelTanh_mnist.log_cost, label="Tanh Aktivierungsfunktion")
#plt.plot(np.arange(len(modelGCU_mnist.log_acc)), modelGCU_mnist.log_cost, label="CCE Aktivierungsfunktion")
#plt.legend()
#plt.show()

# Spiral optimizer comparison
# accuracy
#plt.plot(np.arange(len(modelAdam_spiral.log_acc)), modelAdam_spiral.log_acc, label="Adam Optimierer")
#plt.plot(np.arange(len(modelRMSprop_spiral.log_acc)), modelRMSprop_spiral.log_acc, label="RMSprop Optimierer")
#plt.plot(np.arange(len(modelAdaGrad_spiral.log_acc)), modelAdaGrad_spiral.log_acc, label="AdaGrad Optimierer")
#plt.plot(np.arange(len(modelSGD_spiral.log_acc)), modelSGD_spiral.log_acc, label="SGD Optimierer")
#plt.legend()
#plt.show()

# cost
#plt.plot(np.arange(len(modelAdam_spiral.log_acc)), modelAdam_spiral.log_cost, label="Adam Optimierer")
#plt.plot(np.arange(len(modelRMSprop_spiral.log_acc)), modelRMSprop_spiral.log_cost, label="RMSprop Optimierer")
#plt.plot(np.arange(len(modelAdaGrad_spiral.log_acc)), modelAdaGrad_spiral.log_cost, label="AdaGrad Optimierer")
#plt.plot(np.arange(len(modelSGD_spiral.log_acc)), modelSGD_spiral.log_cost, label="SGD Optimierer")
#plt.legend()
#plt.show()

# Mnist optimizer comparison
# accuracy
#plt.plot(np.arange(len(modelAdam_mnist.log_acc)), modelAdam_mnist.log_acc, label="Adam Optimierer")
#plt.plot(np.arange(len(modelRMSprop_mnist.log_acc)), modelRMSprop_mnist.log_acc, label="RMSprop Optimierer")
#plt.plot(np.arange(len(modelAdaGrad_mnist.log_acc)), modelAdaGrad_mnist.log_acc, label="AdaGrad Optimierer")
#plt.plot(np.arange(len(modelSGD_mnist.log_acc)), modelSGD_mnist.log_acc, label="SGD Optimierer")
#plt.legend()
#plt.show()

# cost
#plt.plot(np.arange(len(modelAdam_mnist.log_acc)), modelAdam_mnist.log_cost, label="Adam Optimierer")
#plt.plot(np.arange(len(modelRMSprop_mnist.log_acc)), modelRMSprop_mnist.log_cost, label="RMSprop Optimierer")
#plt.plot(np.arange(len(modelAdaGrad_mnist.log_acc)), modelAdaGrad_mnist.log_cost, label="AdaGrad Optimierer")
#plt.plot(np.arange(len(modelSGD_mnist.log_acc)), modelSGD_mnist.log_cost, label="SGD Optimierer")
#plt.legend()
#plt.show()
