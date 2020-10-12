import numpy as np

allcord = []

AP_x = 11#int(input("Enter the x coordinate of the access point : "))
AP_y = 11#int(input("Enter the y coordinate of the access point : "))
AP_z = 151#int(input("Enter the z coordinate of the access point : "))

def distance_from_AP(x,y,z):
    
    distance = int((((int(AP_x - x))**2) + ((int(AP_y - y))**2) + ((int(AP_z - z))**2))**0.5)
    return distance


def add_distance(X,Y,Z,newMatrix):

    for height in range (0,26):
        for row in range (0,255):
            for col in range (0,255):
                newMatrix[height][row][col][2] = distance_from_AP(X,Y,Z)
                
def color_cube_innermost(x,y,z,val,newMatrix):
    
    l = 255
    h = 26
    b = 100
    AP_x = 11
    AP_y = 11
    AP_z = 151
    for y in range (0,h,1):
        if(AP_y <= y):
            for z in range (AP_y,y+1,1):
                if(AP_x <= x):
                    for x in range (AP_x,x+1,1):
                        newMatrix[x][y][z][1] = val
                        
                else:
                    for x in range (x,AP_x+1,1):
                        newMatrix[x][y][z][1] = val
                        
        else:
            for z in range (y,AP_y+1,1):
                if(AP_x <= x):
                    for x in range (AP_x,x+1,1):
                        newMatrix[x][y][z][1] = val
                        
                else:
                    for x in range (x,AP_x+1,1):
                        newMatrix[x][y][z][1] = val
                

c = 0
allarray = []   
def copy_color():
    c = 0
    final = []
    
    file = str(input("Enter name of file with coordinates : "))
    filepath = file + '.txt'
    filename = str(input("Enter name of Output file : "))
    Matrix = np.load('normal_original_255_26_255.npy')
    print("Matrix Loaded")
    
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

            allcord.append(newfile_name)
            #print(newfile_name)

            newMatrix = np.copy(Matrix)

            # Coordinate and voxel on its right 
            newMatrix[y][x][z][0] = 1
            #newMatrix[y][x][z][2] = distance_from_AP(x,y,z)
            
            newMatrix[y][x+1][z][0] = 1
            #newMatrix[y][x+1][z][2] = distance_from_AP(x,y,z)

            # Voxels below the coordiante and one to its right 
            newMatrix[y-1][x][z][0] = 1
            #newMatrix[y-1][x][z][2] = distance_from_AP(x,y,z)
            
            newMatrix[y-1][x+1][z][0] = 1
            #newMatrix[y-1][x+1][z][2] = distance_from_AP(x,y,z)

            # Voxels infront of the coordinate and to its right
            newMatrix[y][x][z-1][0] = 1
            #newMatrix[y][x][z-1][2] = distance_from_AP(x,y,z)
            
            newMatrix[y][x+1][z-1][0]= 1
            #newMatrix[y][x+1][z-1][2]= distance_from_AP(x,y,z)

            # Voxels infront of the voxels below the coordinate and to its right
            newMatrix[y-1][x][z-1][0] = 1
            #newMatrix[y-1][x][z-1][2] = distance_from_AP(x,y,z)
            
            newMatrix[y-1][x+1][z-1][0] = 1
            #newMatrix[y-1][x+1][z-1][2] = distance_from_AP(x,y,z)
            

            color_cube_innermost(y,x,z,10,newMatrix)
            add_distance(x,y,z,newMatrix)
            #np.save(newfile_name,newMatrix)
            
            #rotote_90_newmatrix = np.rot90(newMatrix, k=1,axes=(2, 1))
            #np.save(newfile_90,rotote_90_newmatrix)

            #rotote_180_newmatrix = np.rot90(rotote_90_newmatrix, k=1,axes=(2, 1))
            #np.save(newfile_180,rotote_180_newmatrix)

            #rotote_270_newmatrix = np.rot90(rotote_180_newmatrix,k=1, axes=(2, 1))
            #np.save(newfile_270,rotote_270_newmatrix)


            final.append(newMatrix)
            #final.append(rotote_90_newmatrix)
            #final.append(rotote_180_newmatrix)
            #final.append(rotote_270_newmatrix)
            
            print("File Number = ", c+1, " Saved!")
            c +=1

    return final            

b = copy_color()
b = np.asarray(b)
name = str("New_floorplans"+".npy")
np.save(name,b)
'''
allarray = np.array(allarray)

print("Array converted to numpy")
np.save("All_FloorPlans",allarray)
'''
print("ALL ARRAY SAVED")

print("")
