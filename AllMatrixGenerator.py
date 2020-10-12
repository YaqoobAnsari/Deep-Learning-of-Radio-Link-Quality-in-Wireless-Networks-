import numpy as np
final = []
y = np.load('Filenames_ALL_Files.npy')
for filenamelist in y:
    filename = filenamelist[0]
    temp = np.load(filename)
    final.append(temp)
    
np.save("ALL_MATRICES.npy",final)
    
