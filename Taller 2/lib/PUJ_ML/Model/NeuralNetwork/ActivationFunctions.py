## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

class ActivationFunctions:

  '''
  '''
  def get( name ):
    if name.lower( ) == 'identity':
      return ( 'Identity', ActivationFunctions.Identity )
    elif name.lower( ) == 'relu':
      return ( 'ReLU', ActivationFunctions.ReLU )
    elif name.lower( ) == 'leakyrelu':
      return ( 'LeakyReLU', ActivationFunctions.LeakyReLU )
    elif name.lower( ) == 'clippedrelu':
      return ( 'ClippedReLU', ActivationFunctions.ClippedReLU )
    elif name.lower( ) == 'sigmoid':
      return ( 'Sigmoid', ActivationFunctions.Sigmoid )
    elif name.lower( ) == 'tanh':
      return ( 'Tanh', ActivationFunctions.Tanh )
    elif name.lower( ) == 'softmax':
      return ( 'SoftMax', ActivationFunctions.SoftMax )
    else:
      return None
    # end if
  # end def

  '''
  '''
  def Identity( Z, d = False ):
    if d:
      return numpy.ones( Z.shape )
    else:
      return Z
    # end if
  # end def

  '''
  Función ReLU con un límite para prevenir valores extremos
  '''
  def ReLU( Z, d = False ):
    if d:
      # Derivada
      return ( Z > 0 ).astype( float )
    else:
      # ReLU con límite superior
      return numpy.clip(Z, 0, 20.0)  # Limitar valores a un máximo de 20
    # end if
  # end def
  
  '''
  LeakyReLU: una alternativa a ReLU que previene neuronas muertas
  '''
  def LeakyReLU( Z, d = False, alpha = 0.01 ):
    if d:
      # Derivada: alpha para x < 0, 1 para x >= 0
      return numpy.where(Z > 0, 1.0, alpha)
    else:
      Z_array = numpy.asarray(Z)
      # Implementación: max(alpha*x, x)
      result = numpy.where(Z_array > 0, Z_array, alpha * Z_array)
      # Limitar valores extremos
      result = numpy.clip(result, -100, 100)
      
      # Convertir de vuelta al tipo original
      if isinstance(Z, numpy.matrix):
        return numpy.asmatrix(result)
      else:
        return result
    # end if
  # end def
  
  '''
  ClippedReLU: ReLU con límite superior explícito
  '''
  def ClippedReLU( Z, d = False, threshold = 20.0 ):
    if d:
      # Derivada: 1 para 0 < x < threshold, 0 en otro caso
      return numpy.where((Z > 0) & (Z < threshold), 1.0, 0.0)
    else:
      Z_array = numpy.asarray(Z)
      # min(max(0, x), threshold)
      result = numpy.clip(Z_array, 0, threshold)
      
      # Convertir de vuelta al tipo original
      if isinstance(Z, numpy.matrix):
        return numpy.asmatrix(result)
      else:
        return result
    # end if
  # end def

  '''
  '''
  def Sigmoid( Z, d = False ):
    if d:
      s = ActivationFunctions.Sigmoid( Z, False )
      return numpy.multiply( s, float( 1 ) - s )
    else:
      # Implementación estable para evitar overflow
      Z_array = numpy.asarray(Z)
      Z_safe = numpy.clip(Z_array, -500, 500)  # Limitar valores extremos
      result = float( 1 ) / ( float( 1 ) + numpy.exp( -Z_safe ) )
      
      # Convertir de vuelta al tipo original
      if isinstance(Z, numpy.matrix):
        return numpy.asmatrix(result)
      else:
        return result
    # end if
  # end def

  '''
  '''
  def Tanh( Z, d = False ):
    if d:
      T = ActivationFunctions.Tanh( Z, False )
      return float( 1 ) - numpy.multiply( T, T )
    else:
      # Limitar valores extremos para prevenir overflow
      Z_array = numpy.asarray(Z)
      Z_safe = numpy.clip(Z_array, -500, 500)
      result = numpy.tanh( Z_safe )
      
      # Convertir de vuelta al tipo original
      if isinstance(Z, numpy.matrix):
        return numpy.asmatrix(result)
      else:
        return result
    # end if
  # end def
  
  '''
  Implementación optimizada de SoftMax para reducir el consumo de memoria
  '''
  def SoftMax(Z, d=False):
    if d:
        # La derivada de SoftMax es compleja y depende del contexto
        # Esta es una aproximación simplificada
        s = ActivationFunctions.SoftMax(Z, False)
        return numpy.multiply(s, float(1) - s)
    else:
        # Convertir a array si es una matriz
        Z_array = numpy.asarray(Z)
        
        # Siempre usar la implementación por batches para conjuntos grandes
        batch_size = min(100, Z_array.shape[0])  # Limitar tamaño del batch
        result = ActivationFunctions.batched_softmax(Z_array, batch_size)
            
        # Convertir de vuelta al tipo original si era una matriz
        if isinstance(Z, numpy.matrix):
            return numpy.asmatrix(result)
        else:
            return result
    # end if
  # end def

  '''
  Implementación de SoftMax por batches optimizada para memoria
  '''
  def batched_softmax(Z, batch_size=100):
    """Implementación por batches de SoftMax para grandes conjuntos de datos"""
    # Asegurarse de que Z es un array, no una matriz
    Z_array = numpy.asarray(Z, dtype=numpy.float32)  # Usar float32 para reducir memoria
    
    m = Z_array.shape[0]
    n = Z_array.shape[1]
    result = numpy.zeros((m, n), dtype=numpy.float32)
    
    for i in range(0, m, batch_size):
        end = min(i + batch_size, m)
        batch = Z_array[i:end]
        
        # Implementación estable para evitar overflow
        batch_max = numpy.max(batch, axis=1, keepdims=True)
        batch_shifted = batch - batch_max
        
        # Usar operaciones in-place para reducir memoria
        numpy.exp(batch_shifted, out=batch_shifted)
        batch_sum = numpy.sum(batch_shifted, axis=1, keepdims=True)
        numpy.divide(batch_shifted, batch_sum, out=batch_shifted)
        
        result[i:end] = batch_shifted
        
        # Liberar memoria explícitamente
        del batch, batch_max, batch_shifted, batch_sum
        import gc
        gc.collect()
    
    return result
  # end def

# end class

## eof - ActivationFunctions.py