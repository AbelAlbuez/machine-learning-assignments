## =========================================================================
## Script para visualizar resultados de entrenamiento
## =========================================================================

import matplotlib.pyplot as plt
import numpy as np

def visualize_history(debugger):
    """
    Visualiza el historial de entrenamiento del depurador mejorado
    
    Args:
        debugger: Instancia de ImprovedDebugger con historial de entrenamiento
    """
    history = debugger.m_History
    
    # Crear figura con subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Gráfico 1: Pérdida de entrenamiento y prueba
    axs[0, 0].plot(history['epoch'], history['train_loss'], 'b-', label='Entrenamiento')
    if 'test_loss' in history and history['test_loss']:
        axs[0, 0].plot(history['epoch'], history['test_loss'], 'r-', label='Prueba')
    axs[0, 0].set_title('Pérdida')
    axs[0, 0].set_xlabel('Época')
    axs[0, 0].set_ylabel('Pérdida')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Gráfico 2: Precisión de entrenamiento y prueba
    if 'train_acc' in history and history['train_acc']:
        axs[0, 1].plot(
            [history['epoch'][i] for i in range(len(history['epoch'])) if i < len(history['train_acc'])], 
            history['train_acc'], 'b-', label='Entrenamiento'
        )
    if 'test_acc' in history and history['test_acc']:
        axs[0, 1].plot(
            [history['epoch'][i] for i in range(len(history['epoch'])) if i < len(history['test_acc'])], 
            history['test_acc'], 'r-', label='Prueba'
        )
    axs[0, 1].set_title('Precisión')
    axs[0, 1].set_xlabel('Época')
    axs[0, 1].set_ylabel('Precisión')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Gráfico 3: Norma del gradiente
    axs[1, 0].plot(history['epoch'], history['grad_norm'], 'g-')
    axs[1, 0].set_title('Norma del Gradiente')
    axs[1, 0].set_xlabel('Época')
    axs[1, 0].set_ylabel('Norma')
    axs[1, 0].grid(True)
    
    # Gráfico 4: Diferencia entre pérdida de entrenamiento y prueba (generalización)
    if 'test_loss' in history and history['test_loss']:
        # Calcular diferencia de pérdida (pérdida de prueba - pérdida de entrenamiento)
        loss_diff = np.array(history['test_loss']) - np.array(history['train_loss'])
        axs[1, 1].plot(history['epoch'], loss_diff, 'm-')
        axs[1, 1].set_title('Diferencia Pérdida (Prueba - Entrenamiento)')
        axs[1, 1].set_xlabel('Época')
        axs[1, 1].set_ylabel('Diferencia')
        axs[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_visualization.png')
    plt.show()

def visualize_confusion_matrix(model, X_test, y_test, class_names=None):
    """
    Visualiza la matriz de confusión para el modelo en el conjunto de prueba
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas de prueba
        class_names: Nombres de las clases (opcional)
    """
    # Si las etiquetas son one-hot, convertirlas a índices de clase
    if y_test.shape[1] > 1:
        true_labels = np.argmax(y_test, axis=1)
    else:
        true_labels = y_test.flatten().astype(int)
    
    # Predecir con el modelo
    y_pred = model(X_test)
    if y_pred.shape[1] > 1:
        pred_labels = np.argmax(y_pred, axis=1)
    else:
        pred_labels = (y_pred >= 0.5).astype(int).flatten()
    
    # Calcular matriz de confusión
    num_classes = max(np.max(true_labels) + 1, np.max(pred_labels) + 1)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for i in range(len(true_labels)):
        conf_matrix[true_labels[i], pred_labels[i]] += 1
    
    # Visualizar matriz de confusión
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Etiquetas de los ejes
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Matriz de Confusión',
           ylabel='Etiqueta Verdadera',
           xlabel='Etiqueta Predicha')
    
    # Rotar etiquetas del eje x
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Mostrar valores en la matriz
    thresh = conf_matrix.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Calcular métricas globales
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    print(f"Precisión global: {accuracy:.4f}")
    
    # Calcular precisión y recall por clase
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)
    
    for i in range(num_classes):
        precision[i] = conf_matrix[i, i] / np.sum(conf_matrix[:, i]) if np.sum(conf_matrix[:, i]) > 0 else 0
        recall[i] = conf_matrix[i, i] / np.sum(conf_matrix[i, :]) if np.sum(conf_matrix[i, :]) > 0 else 0
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    # Imprimir métricas por clase
    print("\nMétricas por clase:")
    for i in range(num_classes):
        print(f"Clase {class_names[i]}: Precisión={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-Score={f1_score[i]:.4f}")

# Ejemplo de uso:
# from ImprovedDebugger import ImprovedDebugger
# from train_mnist import train_mnist
# model, debugger = train_mnist("mnist_optimizado.txt")
# visualize_history(debugger)
# 
# # Asumiendo que tienes los datos de prueba disponibles
# from PUJ_ML.Data import load_mnist
# (_, _), (X_test, y_test) = load_mnist()
# X_test = X_test.astype(np.float32) / 255.0
# visualize_confusion_matrix(model, X_test, y_test, class_names=[str(i) for i in range(10)])