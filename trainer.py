import pandas as pd
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping

def create_sequences(df, sequence_length):
    data = []
    for i in range(len(df) - sequence_length):
        data.append(df.iloc[i:i + sequence_length])
    return np.array(data)

import pickle

def train_and_save_model_and_scaler(data_path, model_path, scaler_path):
    sequence_length = 20
    num_features = 4

    df = pd.read_csv(data_path)

    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(df[['open', 'max', 'min', 'close']])

    # Guardar el objeto scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)


    # Crea una instancia del objeto EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=14, restore_best_weights=True)
    
    X = create_sequences(pd.DataFrame(data_normalized), sequence_length)
    y = (df['close'].iloc[sequence_length:] > df['open'].iloc[sequence_length:]).astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32, shuffle=True)

    model = Sequential()
    model.add(LSTM(75, input_shape=(sequence_length, num_features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(75))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    #model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    #model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=65)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=65, callbacks=[early_stopping])

    model.save(model_path)

data_path = 'candle_data.csv'
model_path = 'trained_model.h5'
scaler_path = 'scaler.pkl'
train_and_save_model_and_scaler(data_path, model_path, scaler_path)

