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
      s = Sigmoid( Z, False )
      return numpy.multiply( s, float( 1 ) - s )
    else:
      return float( 1 ) / ( float( 1 ) + numpy.exp( -Z ) )
    # end if
  # end def

  '''
  '''
  def Tanh( Z, d = False ):
    if d:
      T = Tanh( Z, False )
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
      # Verificar el tamaño para decidir si usar batches
      if Z.shape[0] > 1000:  # Umbral para usar batches
        return ActivationFunctions.batched_softmax(Z)
      else:
        # Implementación estable para batches pequeños
        Z_shifted = Z - numpy.max(Z, axis=1, keepdims=True)
        exp_Z = numpy.exp(Z_shifted)
        return exp_Z / numpy.sum(exp_Z, axis=1, keepdims=True)
    # end if
  # end def

  '''
  '''
  def batched_softmax(Z, batch_size=100):
    """Implementación por batches de SoftMax para grandes conjuntos de datos"""
    m = Z.shape[0]
    n = Z.shape[1]
    result = numpy.zeros((m, n), dtype=Z.dtype)
    
    for i in range(0, m, batch_size):
      end = min(i + batch_size, m)
      batch = Z[i:end]
      
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
