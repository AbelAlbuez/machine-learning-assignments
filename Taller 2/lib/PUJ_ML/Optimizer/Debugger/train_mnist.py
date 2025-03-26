## =========================================================================
## Script mejorado para entrenar modelos en MNIST con optimizaciones de memoria
## =========================================================================

import numpy as np
import gc
import time
import sys
from PUJ_ML.Neural.FeedForward import FeedForward
from PUJ_ML.Optimization.Adam import Adam
from ImprovedDebugger import ImprovedDebugger

def train_mnist(model_file, epochs=30, batch_size=128, lr=0.001):
    """
    Entrena un modelo de red neuronal en el conjunto de datos MNIST
    
    Args:
        model_file: Archivo de definición del modelo
        epochs: Número máximo de épocas
        batch_size: Tamaño del lote
        lr: Tasa de aprendizaje inicial
    """
    print(f"Cargando modelo desde {model_file}...")
    model = FeedForward()
    model.load(model_file)
    
    # Cargar datos MNIST
    print("Cargando datos MNIST...")
    try:
        # Asumiendo que tienes una función para cargar MNIST o ya los tienes preprocesados
        from PUJ_ML.Data import load_mnist
        (X_train, y_train), (X_test, y_test) = load_mnist()
        
        # Convertir a float32 para reducir memoria
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
        
        print(f"Datos cargados: {X_train.shape[0]} muestras de entrenamiento, {X_test.shape[0]} muestras de prueba")
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        return
    
    # Configurar el optimizador
    print("Configurando optimizador...")
    optimizer = Adam(model)
    optimizer.m_Alpha = lr
    
    # Regularización (opcional, ajusta según necesidad)
    optimizer.m_Lambda1 = 0.0001  # L1
    optimizer.m_Lambda2 = 0.0005  # L2
    
    # Configurar depurador con early stopping
    print("Configurando depurador...")
    debugger = ImprovedDebugger(epochs, model, (X_train, y_train), (X_test, y_test), patience=10)
    debugger.set_lr_decay(optimizer, initial_lr=lr, decay_rate=0.95, decay_epochs=5)
    optimizer.m_Debug = debugger
    
    # Liberar memoria antes de comenzar
    gc.collect()
    
    # Entrenar modelo
    print("Iniciando entrenamiento...")
    start_time = time.time()
    try:
        optimizer.fit((X_train, y_train), (X_test, y_test), batch_size=batch_size)
        
        # Mostrar resumen
        debugger.print_summary()
        
        # Guardar modelo (opcional)
        # model.save("mnist_trained_model.txt")
        
        print(f"Entrenamiento completado en {time.time() - start_time:.2f} segundos")
    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        # Mostrar información de uso de memoria para depuración
        try:
            import psutil
            process = psutil.Process()
            print(f"Uso de memoria: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        except:
            pass
    
    return model, debugger

if __name__ == "__main__":
    # Obtener argumentos de línea de comandos
    model_file = "mnist_optimizado.txt"  # Por defecto
    epochs = 30
    batch_size = 128
    lr = 0.001
    
    # Permitir que el usuario especifique estos parámetros
    if len(sys.argv) > 1:
        model_file = sys.argv[1]
    if len(sys.argv) > 2:
        epochs = int(sys.argv[2])
    if len(sys.argv) > 3:
        batch_size = int(sys.argv[3])
    if len(sys.argv) > 4:
        lr = float(sys.argv[4])
    
    print(f"Parámetros: modelo={model_file}, épocas={epochs}, batch_size={batch_size}, lr={lr}")
    train_mnist(model_file, epochs, batch_size, lr)