import numpy as np

n = 7
grid = np.zeros((7,7))

a1 = np.copy(grid) 
a1[1,1] = 1

a2 = np.copy(grid) 
a2[0,1] = 1

a3 = np.copy(grid) 
a3[1,0] = 1

b1 = np.copy(grid) 
b1[0,2] = 1

b2 = np.copy(grid) 
b2[1,2] = 1

b3 = np.copy(grid) 
b3[2,2] = 1

b4 = np.copy(grid) 
b4[2,1] = 1

b5 = np.copy(grid) 
b5[2,0] = 1


possible_moves = [a1, a2, a3, b1, b2, b3, b4, b5]

a1 = [(0,0),(0,1)]
a2 = [(0,0),(1,1)]
a3 = [(0,0),(1,0)]
b1 = [(0,0),(0,2)]
b2 = [(0,0),(1,2)]
b3 = [(0,0),(2,2)]
b4 = [(0,0),(2,1)]
b5 = [(0,0),(2,0)]

possible_moves = [a1, a2, a3, b1, b2, b3, b4, b5]

