import numpy as np

file = str(input("Enter name of file with coordinates : "))
filepath = file + '.txt'
filenames = []
with open(filepath,"r") as fp:

        while True:
            line = fp.readline()
            if not line:
                break
            line = line[:-1]
            line = line.split(",")

            x = int(line[0])
            y = int(line[1])
            z = int(line[2])
               
            newfile_name = str("Matrix_") + str(x) + "_" + str(y) + "_" + str(z)+ ".npy"
            newfile_90 = str("Matrix_90_") + str(x) + "_" + str(y) + "_" + str(z)+ ".npy"
            newfile_180 = str("Matrix_180_") + str(x) + "_" + str(y) + "_" + str(z)+ ".npy"
            newfile_270 = str("Matrix_270_") + str(x) + "_" + str(y) + "_" + str(z)+ ".npy"

            filenames.append(newfile_name)
            filenames.append(newfile_90)
            filenames.append(newfile_180)
            filenames.append(newfile_270)


filenames = np.array(filenames)
np.save("Filenames_ALL_Files",filenames)
