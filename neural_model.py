import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

def build_vanguard_model(num_actions=10):
    """
    Builds a temporal neural network to classify sequence data.
    Input Shape: (30 frames, 126 coordinates)
    """
    model = Sequential()
    
    # Layer 1: The deep sequence reader
    model.add(LSTM(128, return_sequences=True, activation='tanh', input_shape=(30, 126)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    # Layer 2: Feature abstraction
    model.add(LSTM(256, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))
    
    # Layer 3: Compression
    model.add(LSTM(128, return_sequences=False, activation='tanh'))
    model.add(BatchNormalization())
    
    # Fully Connected output layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    
    # Final classification (Softmax outputs a probability distribution across 'num_actions')
    model.add(Dense(num_actions, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    return model

if __name__ == "__main__":
    model = build_vanguard_model()
    model.summary()
    print("[SYS] NEURAL ARCHITECTURE COMPILED.")