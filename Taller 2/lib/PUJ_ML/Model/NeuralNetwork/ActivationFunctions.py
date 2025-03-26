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
  '''
  def ReLU( Z, d = False ):
    if d:
      return ( Z > 0 ).astype( float )
    else:
      return numpy.multiply( Z, ( Z > 0 ).astype( float ) )
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
      Z_safe = numpy.clip(Z, -500, 500)  # Limitar valores extremos
    return float( 1 ) / ( float( 1 ) + numpy.exp( -Z_safe ) )
    # end if
  # end def

  '''
  '''
  def Tanh( Z, d = False ):
    if d:
      T = ActivationFunctions.Tanh( Z, False )  # Corregido para usar el prefijo de clase
      return float( 1 ) - numpy.multiply( T, T )
    else:
      return numpy.tanh( Z )
    # end if
  # end def
  
  '''
  '''
  def SoftMax(Z, d=False):
    if d:
        # Si realmente necesitas la derivada de SoftMax
        # Nota: esta no es la derivada correcta de SoftMax
        # La derivada de SoftMax es más compleja y matricial
        s = ActivationFunctions.SoftMax(Z, False)
        return numpy.multiply(s, float(1) - s)
    else:
        # Convertir a array si es una matriz
        Z_array = numpy.asarray(Z)
        
        # Verificar el tamaño para decidir si usar batches
        if Z_array.shape[0] > 1000:  # Umbral para usar batches
            result = ActivationFunctions.batched_softmax(Z_array)
        else:
            # Implementación estable para batches pequeños
            Z_shifted = Z_array - numpy.max(Z_array, axis=1, keepdims=True)
            exp_Z = numpy.exp(Z_shifted)
            result = exp_Z / numpy.sum(exp_Z, axis=1, keepdims=True)
            
        # Convertir de vuelta al tipo original si era una matriz
        if isinstance(Z, numpy.matrix):
            return numpy.asmatrix(result)
        else:
            return result
    # end if
  # end def

  '''
  '''
  def batched_softmax(Z, batch_size=100):
    """Implementación por batches de SoftMax para grandes conjuntos de datos"""
    # Asegurarse de que Z es un array, no una matriz
    Z_array = numpy.asarray(Z)
    
    m = Z_array.shape[0]
    n = Z_array.shape[1]
    result = numpy.zeros((m, n), dtype=Z_array.dtype)
    
    for i in range(0, m, batch_size):
        end = min(i + batch_size, m)
        batch = Z_array[i:end]
        
        # Implementación estable para evitar overflow
        batch_shifted = batch - numpy.max(batch, axis=1, keepdims=True)
        exp_batch = numpy.exp(batch_shifted)
        result[i:end] = exp_batch / numpy.sum(exp_batch, axis=1, keepdims=True)
        
        # Liberar memoria explícitamente
        del batch, batch_shifted, exp_batch
    
    return result
  # end def

# end class

## eof - ActivationFunctions.py