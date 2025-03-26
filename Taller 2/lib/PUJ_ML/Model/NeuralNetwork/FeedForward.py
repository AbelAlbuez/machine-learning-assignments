## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import sys




import numpy, random
from ..Base import Base
from .ActivationFunctions import ActivationFunctions

"""
"""
class FeedForward( Base ):

  '''
  '''
  m_W = []
  m_B = []
  m_A = []

  '''
  '''
  def __init__( self, n = 0 ):
    super( ).__init__( 0 )
  # end def

  '''
  '''
  def __getitem__( self, i ):
    return 0
    #if l < self.number_of_layers( ):
    #  return self.m_B[ l ][ 0 , 1 ]
    #else:
    #  return 0
    ## end if
  # end def

  '''
  '''
  def __setitem__( self, i, v ):
    pass
    #if l < self.number_of_layers( ):
    #self.m_B[ l ][ 0 , 1 ] = v
    # end if
  # end def

  '''
  '''
  def __iadd__( self, w ):
    L = self.number_of_layers( )
    k = 0
    for l in range( L ):
      i = self.m_W[ l ].shape[ 0 ]
      o = self.m_W[ l ].shape[ 1 ]
      self.m_W[ l ] \
        += \
        numpy.reshape( w[ 0 , k : k + ( i * o ) ], self.m_W[ l ].shape )
      k += i * o
      self.m_B[ l ] \
        += \
        numpy.reshape( w[ 0 , k : k + o ], self.m_B[ l ].shape );
      k += o
    # end for
    return self
  # end def

  '''
  '''
  def __isub__( self, w ):
    L = self.number_of_layers( )
    k = 0
    for l in range( L ):
      i = self.m_W[ l ].shape[ 0 ]
      o = self.m_W[ l ].shape[ 1 ]
      self.m_W[ l ] \
        -= \
        numpy.reshape( w[ 0 , k : k + ( i * o ) ], self.m_W[ l ].shape )
      k += i * o
      self.m_B[ l ] \
        -= \
        numpy.reshape( w[ 0 , k : k + o ], self.m_B[ l ].shape );
      k += o
    # end for
    return self
  # end def

  '''
  '''
  def __str__( self ):
    h = '0'
    p = ''

    for l in range( self.number_of_layers( ) ):
      in_size = self.m_W[ l ].shape[ 0 ]
      out_size = self.m_W[ l ].shape[ 1 ]

      if l == 0:
        h = str( in_size ) + '\n'
      # end if
      h += str( out_size ) + ' ' + self.m_A[ l ][ 0 ] + '\n'

      for i in range( in_size ):
        for o in range( out_size ):
          p += str( self.m_W[ l ][ i, o ] ) + ' '
        # end for
      # end for
      for o in range( out_size ):
        p += str( self.m_B[ l ][ 0 , o ] ) + ' '
      # end for

    # end for

    return h + p
  # end def

  '''
  '''
  def load( self, fname ):
    fstr = open( fname, 'r' )
    lines = [ l.strip( ) for l in fstr.readlines( ) ]
    fstr.close( )

    n0 = lines[ 0 ]
    n1, a = lines[ 1 ].split( )
    self.add_input_layer( int( n0 ), int( n1 ), ActivationFunctions.get( a ) )

    for i in range( 2, len( lines ) - 1 ):
      n, a = lines[ i ].split( )
      self.add_layer( int( n ), ActivationFunctions.get( a ) )
    # end for

    if lines[ -1 ] == 'random':
      self.init( )
    else:
      # *** TODO ***
      pass
    # end if

  # end def

  '''
  '''
  def size( self ):
    s = 0
    for l in range( self.number_of_layers( ) ):
      s += self.m_W[ l ].shape[ 0 ] * self.m_W[ l ].shape[ 1 ]
      s += self.m_B[ l ].shape[ 0 ] * self.m_B[ l ].shape[ 1 ]
    # end for
    return s
  # end def

  '''
  '''
  def add_input_layer( self, in_size, out_size, activation ):
    self.m_W = [ numpy.zeros( ( in_size, out_size ) ) ]
    self.m_B = [ numpy.zeros( ( 1, out_size ) ) ]
    self.m_A = [ activation ]
  # end def

  '''
  '''
  def add_layer( self, out_size, activation ):
    if self.number_of_layers( ) > 0:
      in_size = self.m_B[ -1 ].shape[ 1 ]
      self.m_W += [ numpy.zeros( ( in_size, out_size ) ) ]
      self.m_B += [ numpy.zeros( ( 1, out_size ) ) ]
      self.m_A += [ activation ]
    else:
      raise AssertionError( 'Input layer not yet defined.' )
    # end if
  # end def

  '''
  '''
  def activation( self, l ):
    return self.m_A[ l ]
  # end def

  '''
  '''
  def init( self ):
    for l in range( self.number_of_layers( ) ):
      in_size = self.m_W[ l ].shape[ 0 ]
      out_size = self.m_W[ l ].shape[ 1 ]

      for i in range( in_size ):
        for o in range( out_size ):
          self.m_W[ l ][ i , o ] = random.random( ) - 0.5
        # end for
      # end for
      for o in range( out_size ):
        self.m_B[ l ][ 0 , o ] = random.random( ) - 0.5
      # end for
    # end for
  # end def

  '''
  '''
  def input_size( self ):
    if self.number_of_layers( ) > 0:
      return self.m_W[ 0 ].shape[ 0 ]
    else:
      return 0
    # end if
  # end def

  '''
  '''
  def number_of_layers( self ):
    return len( self.m_W )
  # end def

  '''
  Fix for MNIST: Modified cost_gradient function that handles multi-class better
  while maintaining the same interface
  '''
  def cost_gradient( self, X, Y, L1, L2 ):
    # Forward pass (same as original)
    A = [ X ]
    Z = []
    L = self.number_of_layers( )
    for l in range( L ):
      Z += [ ( A[ -1 ] @ self.m_W[ l ] ) + self.m_B[ l ] ]
      A += [ self.m_A[ l ][ 1 ]( Z[ -1 ] ) ]
    # end for

    G = numpy.zeros( ( 1, self.size( ) ) )
    
    # For MNIST multi-class case with SoftMax
    if Y.shape[1] == 10 and self.m_A[-1][0] == 'SoftMax':
      # Better gradient calculation for multi-class
      m = float( X.shape[0] )
      DL = A[L] - Y  # This works well with SoftMax + categorical cross-entropy
    else:
      # Original code for binary classification (backward compatibility)
      m = float( 1 ) / float( X.shape[ 0 ] )
      DL = A[ L ] - Y
    
    # Rest of backpropagation (same as original)
    i = self.m_B[ L - 2 ].size
    o = self.m_B[ L - 1 ].size
    k = self.size( ) - o
    G[ 0 , k : k + o ] = ( DL.sum( axis = 0 ) * m ).flatten( )
    k -= i * o
    G[ 0 , k : k + ( i * o ) ] = ( ( A[ L - 1 ].T @ DL ) * m ).flatten( )

    # Backpropagate remaining layers (same as original)
    for l in range( L - 1, 0, -1 ):
      o = i
      i = self.m_W[ l - 1 ].shape[ 0 ]

      DL = numpy.multiply(
        ( DL @ self.m_W[ l ].T ),
        self.m_A[ l - 1 ][ 1 ]( Z[ l - 1 ], True )
        )
      k -= o
      G[ 0 , k : k + o ] = ( DL.sum( axis = 0 ) * m ).flatten( )
      k -= i * o
      G[ 0 , k : k + ( i * o ) ] = ( ( A[ l - 1 ].T @ DL ) * m ).flatten( )
    # end for

    # Cost calculation for multi-class MNIST
    if Y.shape[1] == 10 and self.m_A[-1][0] == 'SoftMax':
      # For multi-class, use proper categorical cross-entropy
      z_safe = numpy.clip(A[L], self.m_Epsilon, 1.0 - self.m_Epsilon)
      J = -numpy.sum(numpy.multiply(Y, numpy.log(z_safe))) / float(X.shape[0])
    else:
      # Original code for binary classification
      zi = numpy.where( Y == 0 )[ 0 ].tolist( )
      oi = numpy.where( Y == 1 )[ 0 ].tolist( )

      J  = numpy.log( float( 1 ) - A[ -1 ][ zi , : ] + self.m_Epsilon ).sum( )
      J += numpy.log( A[ -1 ][ oi , : ] + self.m_Epsilon ).sum( )
      J /= float( X.shape[ 0 ] )
      J = -J

    return ( J, G )
  # end def

 
  '''
  Fix for MNIST: Modified cost function that handles multi-class classification better
  while maintaining the same interface
  '''
  def cost( self, X, y ):
    # Get predictions from forward pass usando _evaluate directamente
    z = self._evaluate(X)
    
    # Handle MNIST's multi-class case (10 classes)
    if y.shape[1] == 10 and self.m_A[-1][0] == 'SoftMax':
      # For multi-class, use proper categorical cross-entropy
      m = float( X.shape[0] )
      
      # Clip values to avoid log(0)
      z_safe = numpy.clip(z, self.m_Epsilon, 1.0 - self.m_Epsilon)
      
      # Categorical cross-entropy: -sum(y_true * log(y_pred))
      # Element-wise multiplication and sum across classes
      J = -numpy.sum(numpy.multiply(y, numpy.log(z_safe))) / m
      
      return J
    else:
      # Original binary classification code for backward compatibility
      zi = numpy.where( y == 0 )[ 0 ].tolist( )
      oi = numpy.where( y == 1 )[ 0 ].tolist( )

      J  = numpy.log( float( 1 ) - z[ zi , : ] + self.m_Epsilon ).sum( )
      J += numpy.log( z[ oi , : ] + self.m_Epsilon ).sum( )
      J /= float( X.shape[ 0 ] )

      return -J
  # end def
  
  '''
  Added helper method to calculate classification accuracy
  This doesn't modify the interface but adds a useful method
  '''
  def accuracy(self, X, y):
    # Get predictions
    predictions = self(X)
    
    # For multi-class (MNIST)
    if y.shape[1] > 2:  # More than binary classification
      # Convert to class indices
      pred_classes = numpy.argmax(predictions, axis=1)
      true_classes = numpy.argmax(y, axis=1)
      
      # Calculate accuracy
      correct = numpy.sum(pred_classes == true_classes)
    else:
      # Binary classification (original behavior)
      predicted_binary = (predictions >= 0.5).astype(float)
      correct = numpy.sum(predicted_binary == y)
    
    # Return accuracy as percentage
    return float(correct) / float(y.shape[0] * y.shape[1]) * 100.0
  # end def

  '''
  Improved initialization that works better for ReLU networks
  '''
  def init_he( self ):
    for l in range( self.number_of_layers( ) ):
      in_size = self.m_W[ l ].shape[ 0 ]
      out_size = self.m_W[ l ].shape[ 1 ]
      
      # Standard He initialization - better for ReLU
      std_dev = numpy.sqrt(2.0 / in_size)
      
      # Initialize weights with normal distribution
      for i in range( in_size ):
        for o in range( out_size ):
          self.m_W[ l ][ i , o ] = random.gauss(0, std_dev)
        # end for
      # end for
      
      # Initialize biases with small positive values for ReLU
      for o in range( out_size ):
        self.m_B[ l ][ 0 , o ] = 0.01
      # end for
    # end for
  # end def

  '''
  Este método es requerido por Base.py - Es la función principal de evaluación
  '''
  def _evaluate(self, X, threshold=None):
      """
      Forward pass through the network
      
      Parameters:
      X : Input data matrix
      threshold : Whether to apply threshold for binary classification
      
      Returns:
      numpy.ndarray: Network output after forward pass
      """
      L = self.number_of_layers()
      if L > 0:
          a = X
          for l in range(L):
              z = (a @ self.m_W[l]) + self.m_B[l]
              a = self.m_A[l][1](z)
          # end for
          if threshold is not None and threshold:
              # Only apply threshold for binary classification
              if a.shape[1] == 1:
                  return (a >= 0.5).astype(float)
              else:
                  return a  # For multi-class, return raw probabilities
          else:
              return a
      else:
          return None
      # end if
  # end def
  
  '''
  Método para calcular la precisión del modelo
  '''
  def calculate_accuracy(self, X, y):
      """
      Calculate classification accuracy
      
      Parameters:
      X : Input data
      y : Target labels
      
      Returns:
      float: Classification accuracy (0-1)
      """
      # Get predictions using _evaluate directly to avoid recursion
      predictions = self._evaluate(X)
      
      # For multi-class (MNIST)
      if y.shape[1] > 1:  # More than binary classification
          # Convert to class indices
          pred_classes = numpy.argmax(predictions, axis=1)
          true_classes = numpy.argmax(y, axis=1)
          
          # Calculate accuracy
          correct = numpy.sum(pred_classes == true_classes)
          total = y.shape[0]
      else:
          # Binary classification
          predicted_binary = (predictions >= 0.5).astype(float)
          correct = numpy.sum(predicted_binary == y)
          total = y.shape[0]
      
      # Return accuracy as decimal (0-1)
      return float(correct) / float(total)
  # end def

  '''
  '''
  def _regularization( self, L1, L2 ):
    r = numpy.zeros( ( 1, self.size( ) ) )

    if L1 != float( 0 ) or L2 != float( 0 ):
      _L2 = float( 2 ) * L2
      L = self.number_of_layers( )
      k = 0
      for l in range( L ):
        i = self.m_W[ l ].shape[ 0 ]
        o = self.m_W[ l ].shape[ 1 ]
        W = self.m_W[ l ].flatten( )
        B = self.m_B[ l ].flatten( )
        for j in range( W.size ):
          r[ 0 , k ] = W[ j ] * _L2
          if W[ j ] > 0:
            r[ 0 , k ] += L1
          # end if
          if W[ j ] < 0:
            r[ 0 , k ] -= L1
          # end if
          k += 1
        # end for
        for j in range( B.size ):
          r[ 0 , k ] = B[ j ] * _L2
          if B[ j ] > 0:
            r[ 0 , k ] += L1
          # end if
          if B[ j ] < 0:
            r[ 0 , k ] -= L1
          # end if
          k += 1
        # end for
      # end for
    # end if
    return r
  # end def

# end class

## eof - FeedForward.py
