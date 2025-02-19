import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Carga el dataset, normaliza las características y divide en entrenamiento y prueba."""
    df = pd.read_csv(file_path)
    df['quality'] = (df['quality'] > 6.5).astype(int)
    X = df.drop(columns=['quality']).values
    y = df['quality'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class WineModel:
    """Modelo de clasificación para la calidad del vino basado en regresión logística."""
    def __init__(self, input_dim):
        self.weights = np.random.rand(input_dim) * 0.1
        self.bias = np.random.rand() * 0.1

    def predict(self, X):
        return sigmoid(np.dot(X, self.weights) + self.bias)

class Trainer:
    """Clase para entrenar el modelo utilizando SGD o Adam, con opciones de regularización."""
    def __init__(self, model, learning_rate=0.01, use_adam=False, lambda_=0.01, reg_type="L2"):
        self.model = model
        self.learning_rate = learning_rate
        self.use_adam = use_adam
        self.lambda_ = lambda_
        self.reg_type = reg_type
        self.epochs = 100
        self.history = {"loss": [], "accuracy": []}
        
        if self.use_adam:
            self.m = np.zeros_like(self.model.weights)
            self.v = np.zeros_like(self.model.weights)
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8

    def binary_cross_entropy(self, y_true, y_pred):
        """Calcula la pérdida de entropía cruzada binaria con regularización opcional."""
        epsilon = 1e-8
        loss = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
        
        if self.reg_type == "L2":
            loss += self.lambda_ * np.sum(self.model.weights ** 2)
        elif self.reg_type == "L1":
            loss += self.lambda_ * np.sum(np.abs(self.model.weights))
        elif self.reg_type == "ElasticNet":
            loss += self.lambda_ * (0.5 * np.sum(np.abs(self.model.weights)) + 0.5 * np.sum(self.model.weights ** 2))
        
        return loss
    
    def compute_gradients(self, X, y_true, y_pred):
        """Calcula los gradientes para actualizar los pesos y bias."""
        error = y_pred - y_true
        dW = np.dot(X.T, error) / len(y_true)
        dB = np.mean(error)
        return dW, dB
    
    def update_weights(self, dW, dB, t):
        """Actualiza los pesos usando SGD o Adam."""
        if self.use_adam:
            self.m = self.beta1 * self.m + (1 - self.beta1) * dW
            self.v = self.beta2 * self.v + (1 - self.beta2) * (dW ** 2)
            m_hat = self.m / (1 - self.beta1 ** t)
            v_hat = self.v / (1 - self.beta2 ** t)
            self.model.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            self.model.bias -= self.learning_rate * dB
        else:
            self.model.weights -= self.learning_rate * dW
            self.model.bias -= self.learning_rate * dB
    
    def train(self, X_train, y_train):
        """Entrena el modelo durante varias épocas."""
        for epoch in range(self.epochs):
            y_pred = self.model.predict(X_train)
            loss = self.binary_cross_entropy(y_train, y_pred)
            dW, dB = self.compute_gradients(X_train, y_train, y_pred)
            self.update_weights(dW, dB, epoch + 1)
            accuracy = np.mean((y_pred >= 0.5) == y_train)
            self.history["loss"].append(loss)
            self.history["accuracy"].append(accuracy)
            if (epoch + 1) % 10 == 0:
                print(f"Época {epoch + 1}/{self.epochs}, Pérdida: {loss:.4f}, Precisión: {accuracy:.4f}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data("../datasets/winequality-red.csv")
    
    print("\nEntrenando con SGD...")
    model_sgd = WineModel(X_train.shape[1])
    trainer_sgd = Trainer(model_sgd, learning_rate=0.01, use_adam=False, reg_type="L2")
    trainer_sgd.train(X_train, y_train)
    
    print("\nEntrenando con Adam...")
    model_adam = WineModel(X_train.shape[1])
    trainer_adam = Trainer(model_adam, learning_rate=0.01, use_adam=True, reg_type="L2")
    trainer_adam.train(X_train, y_train)
    
    plt.plot(trainer_sgd.history["loss"], label="SGD - Pérdida")
    plt.plot(trainer_adam.history["loss"], label="Adam - Pérdida")
    plt.legend()
    plt.title("Comparación de Pérdida (SGD vs Adam)")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.show()
