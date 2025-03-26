## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy, random, sys

D = numpy.genfromtxt( sys.argv[ 1 ], delimiter = ',', skip_header = 1 )
X = numpy.asmatrix( D[ : , 0 : D.shape[ 1 ] - 1 ] )
y = numpy.asmatrix( D[ : , -1 ] ).T
L = numpy.unique( numpy.asarray( y ) )

print( 'Histogram' )
for l in L:
  print( l, numpy.sum( y == l ) )
# end for
print( '============' )

## eof - sandbox.py
