import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class NeuralModels:
    def __init__(self, rf_path="data/rf_model.pkl", gb_path="data/gb_model.pkl", lstm_path="data/lstm_model.h5"):
        self.rf_path, self.gb_path, self.lstm_path = rf_path, gb_path, lstm_path
        self.rf = RandomForestRegressor(n_estimators=300)
        self.gb = GradientBoostingRegressor(n_estimators=200)
        self.lstm = self.build_lstm()
        
        if rf_path and os.path.exists(rf_path):
            self.rf = joblib.load(rf_path)
        if gb_path and os.path.exists(gb_path):
            self.gb = joblib.load(gb_path)
        if lstm_path and os.path.exists(lstm_path):
            self.lstm.load_weights(lstm_path)

    def build_lstm(self):
        model = Sequential()
        model.add(LSTM(50, input_shape=(5,1), return_sequences=False))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict(self, features):
        X = np.array(features).reshape(1, -1)
        pred_rf = self.rf.predict(X)[0]
        pred_gb = self.gb.predict(X)[0]
        X_lstm = np.array(features).reshape(1,5,1)
        pred_lstm = self.lstm.predict(X_lstm, verbose=0)[0][0]
        # Ensamble promedio
        return (pred_rf + pred_gb + pred_lstm)/3

    def train_incremental(self, X, y):
        self.rf.fit(X, y)
        self.gb.fit(X, y)
        X_lstm = X.reshape(X.shape[0], 5, 1)
        self.lstm.fit(X_lstm, y, epochs=1, verbose=0)
        joblib.dump(self.rf, self.rf_path)
        joblib.dump(self.gb, self.gb_path)
        self.lstm.save(self.lstm_path)
