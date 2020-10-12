import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#3D Floorplan Input Matrix
X = np.load("FLOORPLANS.npy")

#RSSI Value matrix of corresponding 3D Floorplan's Matrix
Y = np.load("RSSI.npy")
Y = Y.astype(np.float64)
Y = Y.reshape(len(Y),1)
print(Y.shape)
print(X.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
# Scaler computes all feature to generate a value b/t 0 and 1
# Normalizing the RSSI value (between 0 and 1)
np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)


print("")
print("X_train:  ",X_train.shape)
print("Y_train:  ",Y_train.shape)
print("X_test:  ",X_test.shape)
print("Y_test:  ",Y_test.shape)

#scaler_y = MinMaxScaler()
#Y_train = scaler_y.fit_transform(Y_train)
#Y_test = scaler_y.transform(Y_test)
#np.save("Y_train_scaled.npy", Y_train)
#np.save("Y_test_scaled.npy", Y_test)


print("ALL FILES CREATED")
