# Audio-classification
Audio classification using CNN+LSTM

### About Project:
Sound is a form of energy that is produced when an object vibrates and propagates as waves through a medium, such as air, water, or solids. 

It is an essential aspect of our lives which enables us to perceive our surroundings and experience emotions through music and other auditory sensations.

Inspired by the functioning of our brain's auditory cortex, which processes sound by convolving audio signals, we designed Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) models.

We won't be using raw audio, since it can be of any length and our CNN and LSTM model learns from images, therefore we perform some feature extraction, here we are creating mel spectrograms and then feeding them to the model for training.

### Spectrogram of street music:
![image](https://github.com/avrilnandini/Audio-classification/assets/28782334/ce9aa9a5-0467-4aa4-933d-afc657756a5f)

### About data set:

We have taken dataset from Kaggle https://www.kaggle.com/datasets/chrisfilo/urbansound8k, this data set contains 8732 labelled urban sounds of street music, dog bark, aeroplane, engine idling, drilling, children playing, car honk, air conditioner, gunshot, jackhammer and siren.

### Python libraries required:
pip install numpy pandas sklearn tensorflow pyaudio librosa matplotlib 

### Training of CNN+LSTM model:

We have used train_test_split function from sklearn.model_selection to split the spectrograms (input data) and window_labels (output labels) into training and testing sets. The training set will be used to train the model, while the testing set will be used to evaluate its performance.

LabelEncoder from sklearn.preprocessing is used to convert the categorical window_labels into numerical class indices. Label encoding maps each class to a unique integer, making it suitable for neural network training.

One-hot encoding is done that converts categorical data into binary vectors, where each vector has only one element as 1 (indicating the class) and all other elements as 0 (indicating non-class).

We then defined the model architecture using the Keras Sequential API. It begins with a series of CNN layers, followed by Dense layers and finally two LSTM layers.

The CNN part of the model comprises Conv2D layers with max-pooling, dropout, and batch normalization.

The Conv2D layer: short for 2D Convolutional Layer is the building block in convolutional neural networks (CNNs). It performs a mathematical operation called convolution on the input data.

MaxPooling: The MaxPooling2D layer has a pool size of (2, 2), which means it takes the maximum value within a 2x2 window. The effect of this operation is to reduce the size of the feature map in both dimensions (width and height) by half. By taking the maximum value within the pooling window, MaxPooling2D captures the most prominent features in each region, making the model more robust to slight shifts or translations in the input.

Batch Normalization: It normalizes the output from the activation function, multiplies the normalized output with arbitrary,g and then adds arbitrary, b to the resulting product.

Dropout: Dropout randomly deactivates neurons during training, preventing overfitting by encouraging the network to learn robust and diverse features from different subsets of neurons.

The output of the Dense layer is reshaped to have a time step dimension since the subsequent LSTM layers expect input in a time series format.

Two LSTM layers are added with dropout and batch normalization between them. LSTM layers process the temporal dependencies in the data.

The model ends with a Dense layer with num_classes output neurons and softmax activation for classification tasks.

The model is compiled using Adam optimizer with a learning rate of 0.0001 and categorical cross-entropy loss for multi-class classification.

The function trains the 
model on the training data using fit and evaluates it on the test data using evaluate.

By this, we achieved test accuracy of around 92%.

### Output curves:

![image](https://github.com/avrilnandini/Audio-classification/assets/28782334/4a3666e6-03d4-40c7-bbc7-1dbbf1bbe197)

![image](https://github.com/avrilnandini/Audio-classification/assets/28782334/e01dd083-73da-40d0-afa4-45d1c79f67bf)



