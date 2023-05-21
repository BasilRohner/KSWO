import numpy as np
import matplotlib.pyplot as plt

import datasets as data
import network as nn

import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential

# Spiral-dataset training without regulization
def spiral_training():
    
    # Generate training dataset
    X, Y = data.Spiral.generate(1000, 3)
    # Generate validation dataset 
    X_val, Y_val = data.Spiral.generate(300, 3)
 
    # Display training dataset
    data.Spiral.display(X, Y) 
    # Display validation dataset             
    data.Spiral.display(X_val, Y_val)        
    
    # Instanciate network with layers & optimiser
    model = nn.Network(nn.Dense(input_size=2, output_size=128, activation=nn.ReLU, cost=None),
                       nn.Dense(input_size=128, output_size=3, activation=nn.Softmax, cost=nn.CCE),
                       optimizer=nn.Adam())                                  

    # Train network   
    model.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
                epochs=10_000, interval=25) 

# Spiral-dataset training with regulization
def spiral_training_regulization():
    
    # Generate training dataset
    X, Y = data.Spiral.generate(1000, 3)
    # Generate validation dataset 
    X_val, Y_val = data.Spiral.generate(300, 3)
 
    # Display training dataset
    data.Spiral.display(X, Y) 
    # Display validation dataset            
    data.Spiral.display(X_val, Y_val)        
    
    # Instanciate network with layers & optimiser
    model = nn.Network(nn.Dense(input_size=2, output_size=128, activation=nn.ReLU, cost=None, l2w=5e-4, l2b=5e-4),
                       nn.Dense(input_size=128, output_size=3, activation=nn.Softmax, cost=nn.CCE, l2w=5e-5, l2b=5e-4),
                       optimizer=nn.Adam())                                  

    # Train network  
    model.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
                epochs=10_000, interval=25) 

# Spiral-dataset training with keras
def spiral_training_keras():

    # Generate training dataset
    X, Y = data.Spiral.generate(1000, 3)
    # Generate validation dataset
    X_val, Y_val = data.Spiral.generate(300, 3)

    # Encode labels
    Y = nn.encode_onehot(Y)
    Y_val = nn.encode_onehot(Y_val)

    # Display training dataset
    data.Spiral.display(X, Y) 
    # Display validation dataset            
    data.Spiral.display(X_val, Y_val) 

    # Instanciate network with layers
    model = Sequential()
    model.add(Dense(128, input_dim=2, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    # Compile network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train network
    history = model.fit(X, Y, validation_split=0.1, epochs=100, batch_size=50)

    # Plot training & validation accuracy values
    plt.subplot(211)
    plt.title('Trainingsprozess')
    plt.plot(history.history['loss'], label='cost')
    plt.plot(history.history['val_loss'], label='valalidation cost')
    plt.ylabel("Fehler")
    plt.legend()
    plt.subplot(212)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.xlabel("Epoche")
    plt.ylabel("Präzision")
    plt.legend()
    plt.show()

# MNIST-dataset training without regulization
def mnist_training():

    # Generate training & validation dataset
    X, Y, X_val, Y_val = data.Mnist.generate(samples_training_data=10_000, samples_val_data=3_000)

    # Display training dataset
    data.Mnist.display(X)
    # Display validation dataset
    data.Mnist.display(X_val)

    # Instanciate network with layers & optimiser
    model = nn.Network(nn.Flatten(input_shape=(28, 28)),
                       nn.Dense(input_size=784, output_size=64, activation=nn.ReLU, cost=None),
                       nn.Dense(input_size=64, output_size=10, activation=nn.Softmax, cost=nn.CCE),
                       optimizer=nn.Adam())

    # Train network
    model.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
                epochs=400, interval=1)

# MNIST-dataset training with regulization
def mnist_training_regulization():

    # Generate training & validation dataset
    X, Y, X_val, Y_val = data.Mnist.generate(samples_training_data=10_000, samples_val_data=3_000)

    # Display training dataset
    data.Mnist.display(X)
    # Display validation dataset
    data.Mnist.display(X_val)

    # Instanciate network with layers & optimiser
    model = nn.Network(nn.Flatten(input_shape=(28, 28)),
                       nn.Dense(input_size=784, output_size=64, activation=nn.ReLU, cost=None, l2w=5e-4, l2b=5e-4),
                       nn.Dense(input_size=64, output_size=10, activation=nn.Softmax, cost=nn.CCE, l2w=5e-4, l2b=5e-4),
                       optimizer=nn.Adam())

    # Train network
    model.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
                epochs=400, interval=1)

# MNIST-dataset training with keras
def mnist_training_keras():

    # Generate training & validation dataset
    X, Y, X_val, Y_val = data.Mnist.generate(samples_training_data=10_000, samples_val_data=3_000)
    
    # Encode labels
    Y = nn.encode_onehot(Y)
    Y_val = nn.encode_onehot(Y_val)

    # Display training dataset
    data.Mnist.display(X)
    # Display validation dataset
    data.Mnist.display(X_val)

    # Instanciate network with layers
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train network
    history = model.fit(X, Y, validation_split=0.1, epochs=100, batch_size=50)

    # Plot training & validation accuracy values
    plt.subplot(211)
    plt.title('Trainingsprozess')
    plt.plot(history.history['loss'], label='cost')
    plt.plot(history.history['val_loss'], label='valalidation cost')
    plt.ylabel("Fehler")
    plt.legend()
    plt.subplot(212)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.xlabel("Epoche")
    plt.ylabel("Präzision")
    plt.legend()
    plt.show()

# Cat_dog-dataset training without regulization
def Cat_Dog_training():

    # Generate training & validation dataset
    X, Y, X_val, Y_val = data.Cat_Dogs.generate(samples_training_data=10_000, samples_val_data=3_000)

    # Encode labels
    Y = nn.encode_onehot(Y)
    Y_val = nn.encode_onehot(Y_val)

    # Display training dataset
    data.Cat_Dogs.display(X)
    # Display validation dataset
    data.Cat_Dogs.display(X_val)

    # Instanciate network with layers
    model = nn.Network(nn.Flatten(input_shape=(50, 50)),
                       nn.Dense(input_size=2500, output_size=64, activation=nn.ReLU, cost=None),
                       nn.Dense(input_size=64, output_size=2, activation=nn.Softmax, cost=nn.CCE),
                       optimizer=nn.Adam())
    
    # Train network
    model.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
                epochs=400, interval=1)

# Cat_dog-dataset training with regulization
def Cat_Dog_training_regulization():

    # Generate training & validation dataset
    X, Y, X_val, Y_val = data.Cat_Dogs.generate(samples_training_data=10_000, samples_val_data=3_000)

    # Encode labels
    Y = nn.encode_onehot(Y)
    Y_val = nn.encode_onehot(Y_val)

    # Display training dataset
    data.Cat_Dogs.display(X)
    # Display validation dataset
    data.Cat_Dogs.display(X_val)

    # Instanciate network with layers
    model = nn.Network(nn.Flatten(input_shape=(50, 50)),
                       nn.Dense(input_size=2500, output_size=64, activation=nn.ReLU, cost=None, l2w=5e-4, l2b=5e-4),
                       nn.Dense(input_size=64, output_size=2, activation=nn.Softmax, cost=nn.CCE, l2w=5e-4, l2b=5e-4),
                       optimizer=nn.Adam())

    # Train network
    model.train(model_input=X, dataset_input=Y, val_model_input=X_val, val_dataset_input=Y_val,
                epochs=400, interval=1)

# Cat_dog-dataset training with keras
def Cat_Dog_training_keras():

    # Generate training & validation dataset
    X, Y, X_val, Y_val = data.Cat_Dogs.generate(samples_training_data=10_000, samples_val_data=3_000)

    # Display training dataset
    data.Cat_Dogs.display(X)
    # Display validation dataset
    data.Cat_Dogs.display(X_val)

    # Encode labels
    Y = nn.encode_onehot(Y)
    Y_val = nn.encode_onehot(Y_val)

    # Instanciate network with layers
    model = Sequential()
    model.add(Flatten(input_shape=(50,50)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Compile network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train network
    history = model.fit(X, Y, validation_split=0.1, epochs=100, batch_size=50)

    # Plot training & validation accuracy values
    plt.subplot(211)
    plt.title('Trainingsprozess')
    plt.plot(history.history['loss'], label='cost')
    plt.plot(history.history['val_loss'], label='valalidation cost')
    plt.ylabel("Fehler")
    plt.legend()
    plt.subplot(212)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.xlabel("Epoche")
    plt.ylabel("Präzision")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #spiral_training()
    #spiral_training_regulization()
    #spiral_training_keras()
    #mnist_training()
    #mnist_training_regulization()
    #mnist_training_keras()
    #Cat_Dog_training()
    #Cat_Dog_training_regulization()
    #Cat_Dog_training_keras()
    pass