#import numpy as np
#from lab_env import LabEnv, yield_true

#env = LabEnv()

#print("Global maximum noise-free yield:", env.max_yield)
#print("Best T, C:", env.best_T, env.best_C)


import numpy as np
from lab_env import yield_true

T = np.linspace(20, 100, 11)
C = np.linspace(0.1, 2.0, 11)

Y = np.array([[yield_true(t, c) for c in C] for t in T])

best_index = np.unravel_index(np.argmax(Y), Y.shape)
best_T = T[best_index[0]]
best_C = C[best_index[1]]

print("Global maximum noise-free yield:", Y.max())
print("Best temperature:", best_T)
print("Best concentration:", best_C)