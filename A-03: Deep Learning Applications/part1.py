import os
import pandas as panda
import kagglehub

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

from tensorflow.keras.layers import LSTM

import matplotlib.pyplot as plt

# ---------- DOWNLOAD AND LOAD DATASET ----------

# Download the dataset
path = kagglehub.dataset_download("PromptCloudHQ/imdb-data")
print("Path to dataset files:", path)

# List all files in the dataset
files = os.listdir(path)
print("Files in the dataset:", files)

# Load the CSV file
csvFilePath = os.path.join(path, 'IMDB-Movie-Data.csv')
data = panda.read_csv(csvFilePath)

# Display Columns and Sample Data
print(data.columns)
print(data['Title'][1])
print(data['Genre'][1])
print(data['Description'][1])

# ---------- PREPROCESS THE DATA ----------

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize stopwords and lemmatizer
stopWords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define a function to preprocess text
def preprocessText(text): 
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stopwords and non-alphanumeric tokens, and lemmatize remaining tokens
    filteredTokens = [lemmatizer.lemmatize(token) for token in
                      tokens if token not in stopWords and token.isalnum()]
    return ' '.join(filteredTokens)

# Apply preprocessing to the 'Description' column
data['CleanedDescription'] = data['Description'].apply(preprocessText)

# Display the cleaned description
print(data['CleanedDescription'][1])

# ---------- TOKENIZE TEXT DATA AND PAD SEQUENCES ----------

# Initialize the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['CleanedDescription'])

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(data['CleanedDescription'])

# Define max sequence length
maxSeqLength = 100

# Pad the sequences
paddedSequences = pad_sequences(sequences, maxlen=maxSeqLength)

# Load GloVe word embeddings
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip

# Create embedding index
embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create embedding matrix
embeddingDim = 100
wordIndex = tokenizer.word_index
embeddingMatrix = np.zeros((len(wordIndex) + 1, embeddingDim))
for word, i in wordIndex.items():
    embeddingVector = embeddings_index.get(word)
    if embeddingVector is not None:
        embeddingMatrix[i] = embeddingVector

# Print statements
print ('Found %s word vectors.' % len(embeddings_index))
print('Shape of embedding matrix:', embeddingMatrix.shape)
print('Shape of padded sequences:', paddedSequences.shape)

# ---------- PREPROCESS LABELS ----------

# Split genres into list of genres
data['GenreList'] = data['Genre'].apply(lambda x: x.split(','))

# Print genre list
print(data['GenreList'][1])

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Fit and transform the genres
genresEncoded = mlb.fit_transform(data['GenreList'])

# Display encoded genres and classes
print(genresEncoded[0])
print(mlb.classes_)

# ---------- SPLIT DATA INTO TRAINING (700), VALIDATION (100), TEST (200) SETS ----------

# Split the data into training, validation, and test sets

# XTrain = Training Features
# XTemp = Temporary Features
# YTrain = Training Target
# YTemp = Temporary Target
# XVal = Validation Features
# XTest = Testing Features
# YVal = Validation Target
# YTest = Testing Target
XTrain, XTemp, YTrain, YTemp = train_test_split(paddedSequences, genresEncoded,
                                                train_size=700, random_state=88)
XVal, XTest, YVal, YTest = train_test_split(XTemp, YTemp, test_size=200,
                                            random_state=88)

print('Training set shape:', XTrain.shape, YTrain.shape)
print('Validation set shape:', XVal.shape, YVal.shape)
print('Test set shape:', XTest.shape, YTest.shape)

# ---------- CREATE DATALOADERS ----------

# Create data loaders
trainDataset = tf.data.Dataset.from_tensor_slices((XTrain, YTrain)).batch(32)
valDataset = tf.data.Dataset.from_tensor_slices((XVal, YVal)).batch(32)
testDataset = tf.data.Dataset.from_tensor_slices((XTest, YTest)).batch(32)

# ---------- BUILD AND TRAIN THE RNN MODEL ----------

# Build RNN Model
rnnModel = Sequential([
    Embedding(input_dim=len(wordIndex) + 1, output_dim=embeddingDim,
              weights=[embeddingMatrix], input_length=maxSeqLength,
              trainable=False),
    SimpleRNN(128, return_sequences=True),
    SimpleRNN(128),
    Dense(20, activation='sigmoid')
])

# Compile the model
rnnModel.compile(optimizer='adam', loss='binary_crossentropy',
                 metrics=['accuracy'])

# Train the model
historyRnn = rnnModel.fit(trainDataset, validation_data=valDataset,
                          epochs=20)

# Evaluate the model
print("RNN Test Set Model Evaluation:")
rnnModel.evaluate(testDataset)

# ---------- BUILD AND TRAIN THE LSTM MODEL ----------

# Build LSTM Model

lstmModel = Sequential([
    Embedding(input_dim=len(wordIndex) + 1, output_dim=embeddingDim,
              weights = [embeddingMatrix], trainable = False),
    LSTM(128, return_sequences=True),
    LSTM(128),
    Dense(20, activation='sigmoid')
])

# Compile the model
lstmModel.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

# Train the model
historyLSTM = lstmModel.fit(trainDataset, validation_data=valDataset,
                            epochs=20) 

# Evaluate the model
print("LSTM Test Set Model Evaluation:")
lstmModel.evaluate(testDataset)

# ---------- PLOT RESULTS ----------

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(historyRnn.history['accuracy'])
plt.plot(historyRnn.history['val_accuracy'])
plt.title('RNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(historyLSTM.history['accuracy'])
plt.plot(historyLSTM.history['val_accuracy'])
plt.title('LSTM Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Plot training and validation loss
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(historyRnn.history['loss'])
plt.plot(historyRnn.history['val_loss'])
plt.title('RNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(historyLSTM.history['loss'])
plt.plot(historyLSTM.history['val_loss'])
plt.title('LSTM Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# ---------- REPORT THE RESULTS ----------

# Evaluate the RNN model on the test set
rnnTestLoss, rnnTestAccuracy = rnnModel.evaluate(testDataset)
print(f'RNN Model - Test Loss: {round(rnnTestLoss, 2)}, Test Accuracy: {round(rnnTestAccuracy, 2)}')

# Evaluate the LSTM model on the test set
OldLstmTestLoss, OldLstmTestAccuracy = lstmModel.evaluate(testDataset)
print(f'LSTM Model - Test Loss: {round(OldLstmTestLoss, 2)}, Test Accuracy: {round(OldLstmTestAccuracy, 2)}')

# ---------- ADD TITLES TO TEXT DESCRIPTION ----------

# Combine titles and descriptions 
data['TitleDescription'] = data['Title'] + ' ' + data['CleanedDescription']

# Tokenize the combined text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['TitleDescription'])
sequences = tokenizer.texts_to_sequences(data['TitleDescription'])

# Pad the sequences
maxSeqLength = 100
paddedSequences = pad_sequences(sequences, maxlen=maxSeqLength)

# Create an embedding matrix based on the updated word index
wordIndex = tokenizer.word_index
embeddingMatrix = np.zeros((len(wordIndex) + 1, embeddingDim))
for word, i in wordIndex.items():
    embeddingVector = embeddings_index.get(word)
    if embeddingVector is not None:
        embeddingMatrix[i] = embeddingVector

# Split the data into training, validation, and test sets
XTrain, XTemp, YTrain, YTemp = train_test_split(paddedSequences, genresEncoded,
                                                train_size=700, random_state=88)
XVal, XTest, YVal, YTest = train_test_split(XTemp, YTemp, test_size=200,
                                            random_state=88)

# Create data loaders
trainDataset = tf.data.Dataset.from_tensor_slices((XTrain, YTrain)).batch(32)
valDataset = tf.data.Dataset.from_tensor_slices((XVal, YVal)).batch(32)
testDataset = tf.data.Dataset.from_tensor_slices((XTest, YTest)).batch(32)

# Build LSTM Model
lstmModel = Sequential([
    Embedding(input_dim=len(wordIndex) + 1, output_dim=embeddingDim,
              weights=[embeddingMatrix], trainable=False),
    LSTM(128, return_sequences=True),
    LSTM(128),
    Dense(20, activation='sigmoid')
])

# Compile the model
lstmModel.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

# Train the model
historyLSTM = lstmModel.fit(trainDataset, validation_data=valDataset,
                            epochs=20)

# ---------- PRINT NEW RESULTS ----------

# Evaluate the model
print("LSTM Test Set Model Evaluation:")
lstmTestLoss, lstmTestAccuracy = lstmModel.evaluate(testDataset)

# Print the test loss and accuracy
print(f'LSTM Model without Titles - Test Loss: {round(OldLstmTestLoss, 2)},
      Test Accuracy: {round(OldLstmTestAccuracy, 2)}')
print(f'LSTM Model with Titles - Test Loss: {round(lstmTestLoss, 2)},
      Test Accuracy: {round(lstmTestAccuracy, 2)}')