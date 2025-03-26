## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

"""
Depurador mejorado que muestra métricas adicionales sin consumir demasiada memoria
"""
class ImprovedDebugger:
    '''
    Inicialización
    '''
    def __init__(self, max_epochs, m, D_tr, D_te=None, patience=10, min_delta=0.001):
        self.m_MaxEpochs = max_epochs
        self.m_Model = m
        self.m_X_tr = D_tr[0]
        self.m_y_tr = D_tr[1]
        
        self.m_X_te = None
        self.m_y_te = None
        if D_te is not None:
            self.m_X_te = D_te[0]
            self.m_y_te = D_te[1]
        
        # Para early stopping
        self.m_Patience = patience
        self.m_MinDelta = min_delta
        self.m_BestLoss = float('inf')
        self.m_WaitCount = 0
        
        # Para aprendizaje adaptativo
        self.m_InitialLR = None
        self.m_DecayRate = 0.95
        self.m_DecayEpochs = 5
        
        # Historial para visualización
        self.m_History = {
            'epoch': [], 
            'grad_norm': [], 
            'train_loss': [], 
            'test_loss': [],
            'train_acc': [],
            'test_acc': []
        }
        
        # Para seguimiento de tiempo
        self.m_StartTime = None
        self.m_LastEpochTime = None
    # end def
    
    '''
    Muestra una barra de progreso en la terminal
    '''
    def _show_progress_bar(self, current, total, width=50):
        progress = min(1.0, current / total)
        filled_length = int(width * progress)
        bar = '█' * filled_length + '-' * (width - filled_length)
        
        # Calcular tiempo estimado restante
        if self.m_StartTime is None:
            import time
            self.m_StartTime = time.time()
            self.m_LastEpochTime = self.m_StartTime
            eta = "Calculando..."
        else:
            import time
            current_time = time.time()
            epoch_time = current_time - self.m_LastEpochTime
            self.m_LastEpochTime = current_time
            
            elapsed = current_time - self.m_StartTime
            if progress > 0:
                total_estimated = elapsed / progress
                remaining = total_estimated - elapsed
                eta = self._format_time(remaining)
            else:
                eta = "Calculando..."
        
        print(f"\rProgreso: |{bar}| {progress*100:.1f}% Época: {current}/{total} ETA: {eta}", end='')
        
        if current == total:
            total_time = time.time() - self.m_StartTime
            print(f"\nEntrenamiento completado en {self._format_time(total_time)}")
    # end def
    
    '''
    Formatea segundos en horas:minutos:segundos
    '''
    def _format_time(self, seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    # end def
    
    '''
    Configurar decaimiento de tasa de aprendizaje
    '''
    def set_lr_decay(self, optimizer, initial_lr=0.001, decay_rate=0.95, decay_epochs=5):
        self.m_InitialLR = initial_lr
        optimizer.m_Alpha = initial_lr
        self.m_DecayRate = decay_rate
        self.m_DecayEpochs = decay_epochs
        self.m_Optimizer = optimizer
    # end def
    
    '''
    Calcular precisión para un conjunto de datos
    '''
    def _calculate_accuracy(self, X, y):
        # Procesar en lotes para evitar problemas de memoria
        batch_size = 128
        total_samples = X.shape[0]
        correct = 0
        
        for i in range(0, total_samples, batch_size):
            end = min(i + batch_size, total_samples)
            X_batch = X[i:end]
            y_batch = y[i:end]
            
            # Predecir
            y_pred = self.m_Model(X_batch)
            
            # Determinar el tipo de problema (clasificación multiclase, binaria, etc.)
            if len(y_batch.shape) > 1 and y_batch.shape[1] > 1:
                # Caso de etiquetas one-hot (clasificación multiclase)
                pred_labels = numpy.argmax(y_pred, axis=1)
                true_labels = numpy.argmax(y_batch, axis=1)
                correct += numpy.sum(pred_labels == true_labels)
            elif len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                # Caso donde las predicciones son multiclase pero las etiquetas son índices
                pred_labels = numpy.argmax(y_pred, axis=1)
                if len(y_batch.shape) > 1:
                    y_batch = y_batch.flatten()
                correct += numpy.sum(pred_labels == y_batch)
            else:
                # Caso de clasificación binaria
                pred_labels = (y_pred >= 0.5).astype(int)
                if len(pred_labels.shape) > 1:
                    pred_labels = pred_labels.flatten()
                if len(y_batch.shape) > 1:
                    y_batch = y_batch.flatten()
                correct += numpy.sum(pred_labels == y_batch)
        
        # Calcular precisión (valores entre 0 y 1)
        accuracy = float(correct) / float(total_samples)
        
        # Verificar que la precisión esté en el rango correcto
        if accuracy < 0 or accuracy > 1:
            print(f"\nADVERTENCIA: Se calculó una precisión fuera de rango: {accuracy}")
            # Limitar a [0,1] si está fuera de rango
            accuracy = max(0, min(1, accuracy))
        
        return accuracy
    # end def
    
    '''
    Callback llamado en cada época
    '''
    def __call__(self, t, nG, J_tr, J_te):
        # Early stopping flag
        stop = not (t < self.m_MaxEpochs)
        
        # Mostrar progreso
        self._show_progress_bar(t, self.m_MaxEpochs)
        
        # Calcular métricas adicionales
        tr_acc = None
        te_acc = None
        
        # Solo calcular métricas cada 5 épocas para ahorrar tiempo y memoria
        if t % 5 == 0 or t == 1 or stop:
            # Calcular precisión de entrenamiento en un subconjunto para ahorrar memoria
            sample_size = min(1000, self.m_X_tr.shape[0])
            indices = numpy.random.choice(self.m_X_tr.shape[0], sample_size, replace=False)
            tr_acc = self._calculate_accuracy(self.m_X_tr[indices], self.m_y_tr[indices])
            
            # Calcular precisión de prueba si hay datos de prueba
            if self.m_X_te is not None and self.m_y_te is not None:
                te_acc = self._calculate_accuracy(self.m_X_te, self.m_y_te)
        
        # Aplicar decaimiento de tasa de aprendizaje si está configurado
        if hasattr(self, 'm_Optimizer') and t > 0 and t % self.m_DecayEpochs == 0:
            self.m_Optimizer.m_Alpha *= self.m_DecayRate
            print(f"\nLR ajustada a: {self.m_Optimizer.m_Alpha:.6f}")
        
        # Actualizar historial
        self.m_History['epoch'].append(t)
        self.m_History['grad_norm'].append(nG)
        self.m_History['train_loss'].append(J_tr)
        if J_te is not None:
            self.m_History['test_loss'].append(J_te)
        if tr_acc is not None:
            self.m_History['train_acc'].append(tr_acc)
        if te_acc is not None:
            self.m_History['test_acc'].append(te_acc)
        
        # Imprimir resultados detallados solo cada cierto número de épocas o al final
        if t % 5 == 0 or t == 1 or stop:
            print("\n" + "=" * 80)
            print(f"RESUMEN ÉPOCA: {t}/{self.m_MaxEpochs} ({t/self.m_MaxEpochs*100:.1f}% completado)")
            print("-" * 80)
            print(f"Norma del Gradiente: {nG:.6f}")
            print(f"Error Entrenamiento: {J_tr:.6f}")
            
            if J_te is not None:
                print(f"Error Test:          {J_te:.6f}")
                print(f"Diferencia (Test-Train): {J_te-J_tr:.6f}")
            
            if tr_acc is not None:
                print(f"Precisión Entrenamiento: {tr_acc:.4f} ({tr_acc*100:.2f}%)")
            
            if te_acc is not None:
                print(f"Precisión Test:          {te_acc:.4f} ({te_acc*100:.2f}%)")
            
            # Mostrar mejores valores hasta ahora
            best_te_acc = max(self.m_History['test_acc']) if 'test_acc' in self.m_History and self.m_History['test_acc'] else None
            if best_te_acc is not None:
                best_epoch = self.m_History['epoch'][self.m_History['test_acc'].index(best_te_acc)]
                print(f"Mejor precisión test:     {best_te_acc:.4f} (época {best_epoch})")
            
            # Mostrar información de early stopping
            if hasattr(self, 'm_WaitCount'):
                if self.m_WaitCount > 0:
                    print(f"Sin mejora durante {self.m_WaitCount}/{self.m_Patience} épocas")
            
            # Mostrar tasa de aprendizaje actual si está configurada
            if hasattr(self, 'm_Optimizer'):
                print(f"Tasa de aprendizaje:     {self.m_Optimizer.m_Alpha:.6f}")
            
            print("=" * 80)
        
        # Mostrar información compacta en otras épocas
        else:
            print(f" | E: {t} | Loss: {J_tr:.4f}", end='')
            if J_te is not None:
                print(f" | Test: {J_te:.4f}", end='')
            if tr_acc is not None and t % 2 == 0:  # Mostrar precisión cada 2 épocas
                print(f" | Acc: {tr_acc:.4f}", end='')
            print("")
        
        # Early stopping basado en pérdida de prueba
        if J_te is not None:
            if J_te < self.m_BestLoss - self.m_MinDelta:
                self.m_BestLoss = J_te
                self.m_WaitCount = 0
            else:
                self.m_WaitCount += 1
                # if self.m_WaitCount >= self.m_Patience:
                #     print(f"\nEarly stopping en época {t} - No hay mejora durante {self.m_Patience} épocas")
                #     stop = True
        
        return stop
    # end def
    
    '''
    Imprimir resumen del entrenamiento
    '''
    def print_summary(self):
        if not self.m_History['epoch']:
            print("No hay historial de entrenamiento disponible.")
            return
        
        print("\n" + "=" * 80)
        print("                        RESUMEN DEL ENTRENAMIENTO                        ")
        print("=" * 80)
        
        # Información general
        print(f"INFORMACIÓN GENERAL:")
        print(f"- Épocas completadas: {self.m_History['epoch'][-1]}/{self.m_MaxEpochs}")
        
        # Calcular tiempo total
        if self.m_StartTime is not None:
            import time
            total_time = time.time() - self.m_StartTime
            print(f"- Tiempo total de entrenamiento: {self._format_time(total_time)}")
            print(f"- Tiempo promedio por época: {self._format_time(total_time/self.m_History['epoch'][-1])}")
        
        # Métricas finales
        print("\nMÉTRICAS FINALES:")
        if self.m_History['train_loss']:
            print(f"- Pérdida final (entrenamiento): {self.m_History['train_loss'][-1]:.6f}")
        
        if 'test_loss' in self.m_History and self.m_History['test_loss']:
            print(f"- Pérdida final (test): {self.m_History['test_loss'][-1]:.6f}")
            print(f"- Diferencia final (test-train): {self.m_History['test_loss'][-1] - self.m_History['train_loss'][-1]:.6f}")
        
        if self.m_History['train_acc']:
            print(f"- Precisión final (entrenamiento): {self.m_History['train_acc'][-1]:.4f} ({self.m_History['train_acc'][-1]*100:.2f}%)")
        
        if 'test_acc' in self.m_History and self.m_History['test_acc']:
            print(f"- Precisión final (test): {self.m_History['test_acc'][-1]:.4f} ({self.m_History['test_acc'][-1]*100:.2f}%)")
        
        # Mejores resultados
        print("\nMEJORES RESULTADOS:")
        if self.m_History['train_loss']:
            best_train_loss_idx = numpy.argmin(self.m_History['train_loss'])
            print(f"- Mejor pérdida (entrenamiento): {self.m_History['train_loss'][best_train_loss_idx]:.6f} (época {self.m_History['epoch'][best_train_loss_idx]})")
        
        if 'test_loss' in self.m_History and self.m_History['test_loss']:
            best_test_loss_idx = numpy.argmin(self.m_History['test_loss'])
            print(f"- Mejor pérdida (test): {self.m_History['test_loss'][best_test_loss_idx]:.6f} (época {self.m_History['epoch'][best_test_loss_idx]})")
        
        if self.m_History['train_acc']:
            best_train_acc_idx = numpy.argmax(self.m_History['train_acc'])
            print(f"- Mejor precisión (entrenamiento): {self.m_History['train_acc'][best_train_acc_idx]:.4f} ({self.m_History['train_acc'][best_train_acc_idx]*100:.2f}%) (época {self.m_History['epoch'][best_train_acc_idx]})")
        
        if 'test_acc' in self.m_History and self.m_History['test_acc']:
            best_test_acc_idx = numpy.argmax(self.m_History['test_acc'])
            print(f"- Mejor precisión (test): {self.m_History['test_acc'][best_test_acc_idx]:.4f} ({self.m_History['test_acc'][best_test_acc_idx]*100:.2f}%) (época {self.m_History['epoch'][best_test_acc_idx]})")
        
        # Norma del gradiente
        print("\nINFORMACIÓN DEL GRADIENTE:")
        print(f"- Norma inicial: {self.m_History['grad_norm'][0]:.6f}")
        print(f"- Norma final: {self.m_History['grad_norm'][-1]:.6f}")
        print(f"- Reducción de la norma: {(1 - self.m_History['grad_norm'][-1]/self.m_History['grad_norm'][0])*100:.2f}%")
        
        # Información sobre early stopping
        if hasattr(self, 'm_WaitCount') and hasattr(self, 'm_Patience'):
            print("\nEARLY STOPPING:")
            if self.m_WaitCount >= self.m_Patience:
                print(f"- Activado en época {self.m_History['epoch'][-1]}")
                print(f"- Razón: No hubo mejora durante {self.m_Patience} épocas consecutivas")
            else:
                print(f"- No activado (contador final: {self.m_WaitCount}/{self.m_Patience})")
        
        # Información sobre tasa de aprendizaje
        if hasattr(self, 'm_Optimizer') and hasattr(self, 'm_InitialLR'):
            print("\nTASA DE APRENDIZAJE:")
            print(f"- Inicial: {self.m_InitialLR:.6f}")
            print(f"- Final: {self.m_Optimizer.m_Alpha:.6f}")
            print(f"- Factor de decaimiento: {self.m_DecayRate}")
            print(f"- Épocas entre decaimientos: {self.m_DecayEpochs}")
        
        print("=" * 80)
        print("Para visualizar gráficamente estos resultados, ejecute:")
        print("from visualize_training import visualize_history")
        print("visualize_history(debugger)")
        print("=" * 80)
    # end def
# end class

## eof - ImprovedDebugger.py