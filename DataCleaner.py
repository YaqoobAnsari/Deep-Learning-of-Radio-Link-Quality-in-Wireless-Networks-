import pandas as pd
import numpy as np
import operator

#filepath = str(input("Enter the name of file with all filenames: ")) + ".txt"

totalval = []
    
def cleaner(file):
    filepath = str(file) + ".csv"
    df = pd.read_csv(filepath)
    rssi = df["RSSI"]
    rssi = np.array(rssi)
    data = []
    for i in rssi:
        j = int(i[:-3])
        data.append(j)

    data = np.array(data)

    freq = {} 
    for item in data: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1
      
    value = max(freq.items(), key=operator.itemgetter(1))[0]
    print(value)
    '''
    old_median = np.median (data)
    P = np.percentile(data, [5, 95])
    final = []
    final = np.where((data < P[0]) | (data > P[1]), old_median,data)

    mean = str(int(round(np.mean(final)))) + "\n"
    median = str(int(np.median(final))) + "\n"
    std_dev = str(np.std(final))+ "\n"
    '''
    textfile_name = str(file) + ".txt"
    with open(textfile_name,"w") as fileout:
        fileout.write(value)
        #fileout.write(median)
        #fileout.write(std_dev)

cleaner("File0")
# Going through the file with file names
with open(filepath,"r") as fileout:

    singlename = fileout.read()
    singlename = singlename.split()

    for name in singlename:
        #print(type(name))
        cleaner(name)
        #print("HERE")
        text_name = str(name)+".txt"
        with open(text_name,"r") as datafile:
            data = datafile.read()
            data = data.split()
            value = data[0]
            totalval.append(value)
            #print(mean)

print(totalval)

filename = "Rssi.npy"
# define data
data = np.asarray(totalval)
# save to csv file
np.save(filename, data)

        



