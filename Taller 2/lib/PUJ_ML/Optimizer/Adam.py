## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, numpy
import gc  # Recolector de basura
from .GradientDescent import GradientDescent

"""
Implementación optimizada de Adam para reducir el consumo de memoria
"""
class Adam(GradientDescent):
  '''
  Optimizador Adam con gestión de memoria mejorada
  '''
  m_Beta1 = 0.9
  m_Beta2 = 0.999

  '''
  '''
  def __init__(self, m):
    super().__init__(m)
  # end def

  '''
  Implementación optimizada de _fit para reducir el consumo de memoria
  '''
  def _fit(self, X_tr, y_tr, X_te, y_te, batches):
    e = self.m_Epsilon
    a = self.m_Alpha
    l1 = self.m_Lambda1
    l2 = self.m_Lambda2
    b1 = self.m_Beta1
    b2 = self.m_Beta2
    b1t = b1
    b2t = b2
    cb1 = float(1) - b1
    cb2 = float(1) - b2
    
    # Inicializar mt y vt con dimensiones optimizadas
    model_size = self.m_Model.size()
    mt = numpy.zeros((1, model_size), dtype=numpy.float32)  # Usar float32 en lugar de float64
    vt = numpy.zeros((1, model_size), dtype=numpy.float32)  # Usar float32 en lugar de float64

    self.m_Model.init()
    t = 0
    stop = False
    latest_G = None  # Variable para almacenar el último gradiente
    
    while not stop:
      t += 1

      for batch in batches:
        # Liberar memoria no utilizada antes de cada lote
        gc.collect()
        
        # Calcular costo y gradiente
        J_tr, G = self.m_Model.cost_gradient(X_tr[batch, :], y_tr[batch, :], l1, l2)
        latest_G = G  # Guardar referencia al último gradiente

        # Actualizar momentos
        mt = (mt * b1) + (G * cb1)
        vt = (vt * b2) + (numpy.multiply(G, G) * cb2)

        # Calcular actualización de parámetros
        mhat = mt / (1.0 - b1t)
        vhat = vt / (1.0 - b2t)
        
        # Calcular y aplicar la actualización en partes para reducir el uso de memoria
        D = numpy.divide(mhat, numpy.sqrt(vhat) + e)
        self.m_Model -= D * a
        
        # Liberar memoria explícitamente
        del D
        del mhat
        del vhat
        gc.collect()
      # end for

      if not math.isnan(J_tr) and not math.isinf(J_tr):
        J_te = None
        if X_te is not None:
          J_te = self.m_Model.cost(X_te, y_te)
        # end if

        if self.m_Debug is not None:
          # Calcular la norma del gradiente solo si es necesario
          if latest_G is not None:
            grad_norm = numpy.sqrt((latest_G * latest_G).sum())
          else:
            grad_norm = 0
          stop = self.m_Debug(t, grad_norm, J_tr, J_te)
        # end if
      else:
        stop = True
      # end if

      b1t *= b1
      b2t *= b2
    # end while
  # end def

## eof - Adam.py