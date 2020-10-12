# Deep-Learning-of-Indoor-Radio-Link-Quality-in-Wireless-Networks-using-Floor-Plans
The dataset and scripts for recreation of a deep learning model for measuring indoor radio link quality in wireless networks using floor plans.

# Requirements:
1) Python3
2) Keras
3) WireShark
4) Voxelizer
5) Voxelizer Editor

# Data Requirements
* A 2D floorplan of the location where data will be collected.
* A 3D reconstruction of the floor plan (can be created using SweetHome3D software).
* RSSI values at different locations of the floor plan (can be collected using WireShark).
* A list of all the coordinates of locations where data was collected.
* A deepnet architecture to train with the data.

# Repository Contents 
1) TimeTracker.py: A script that notes down the time when data collection was started and stopped and writes the values to a text file.
2) Parser.py: A script that splits the output .csv file containing RSSI values from WireShark into a .csv with all the RSSI and coordinates mapped.
3) DataCleaner.py: A script that removes outlier values from the data. Reads from a .csv file using Pandas library and returns a numpy list with no outliers. Applied IQD method for outlier removal.
4) MatrixShrinker.py: A script that uses the matrix representation of the voxels and creates a cubic matrix using the matrix representation of voxels.
5) CopyCreator.py: A script that the original Matrix file as input and creates copies for every location. In addition, it adds features to the elements. These are distances from the Access Point, a metric for Area of Importance, and colors the location and access points red.
6) AllMatrixGenerator.py: A script that appends all individual matrices representations of the 3D floorplan into one larger ndarray. 
7) TrainTestSplit.py: A script that splits the matrix of all 3D floorplans and the matrix of RSSIs into training and testing data.
8) FilenameGen.py: A script that returns a numpy array with all the filenames of the created matrices copies.
9) GRNN_simple.py: A script containing the Keras architecture for training with the data files.

# Procedure 
* A 2D floorplan of the location is created/obtained and reconstructed into a 3D floorplan (using SweetHome3D software).
* The 3D floorplan created is fed to a Voxelizer that returns a python file with a list implementation of the voxels and a voxelized file.
* Data is collected on the location, following a sequenced locations written in a text file, using WireShark and is obtained in a .csv format.
* Data is parsed into required .csv files using Parser.py which is then cleared of outliers using the DataCleaner.py script and converted into a numpy array.
* The python file with a list implementation of the voxels is converted into a matrix using the MatrixShrinker.py file.
* The created matrix is fed to Copycreator.py which takes in the file with the text coordinates to create copies of the matrix for every location. Lastly, the FilenamesGen.py which creates a numpy array with the names of all the matrix created, is used to feed the AllMatrixGenerator.py which returns a single numpy matrix of all the location matrices. 
* Both the final numpy matrix (one for RSSI, and one for 3D representation of model) are used to train the model (in GRNN_simply.py).

