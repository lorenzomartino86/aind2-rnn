import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    p = len(series)
    
    # containers for input/output pairs
    X = [series[index:window_size+index] for index in range(p-window_size)]
    y = series[window_size : ]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    print (X.shape, y.shape)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(units=5, input_shape = (window_size,1), activation='tanh'))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    punctuation.extend(['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '"','$', '%', '&', "'", '(', ')', '*', '/', '@'])
    for char in punctuation:
        text = text.replace(char, ' ')
    text.replace('à', 'a')
    text.replace('â', 'a')
    text.replace('ò', 'o')
    text.replace('è', 'e')
    text.replace('é', 'e')
    text.replace('ù', 'u')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    p = len(text)
    
    inputs = [text[i:i+window_size] for i in range(0, p - window_size, step_size)]
    outputs = [text[i+step_size] for i in range(window_size,p - step_size)]

    return inputs,outputs


# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(units=200, input_shape = (window_size,num_chars), activation='tanh'))
    model.add(Dense(num_chars, activation='softmax'))
    return model
