import csv

'''
import glob
import os
import pandas as pd

combined_csv = pd.concat( [ pd.read_csv(f) for f in filenames ] )
combined_csv.to_csv( "combined_csv.csv", index=False )

for f in glob.glob("data*.txt"):
     os.system("cat "+f+" >> OutFile.txt")

for f in glob.glob("capture*.csv"):
     os.system("cat "+f+" >> captureFile.csv")
'''

print ("insert time stamps file name")

timeFile = str(input())

print("insert capture file name")

wiresharkFile = str(input())


with open(timeFile,"r") as fileout:
	with open(wiresharkFile, mode = 'r') as csv_file:
		csv_reader = csv.DictReader(csv_file)

		num = int(fileout.readline())
		# num == number of readings

		for i in range(num):
			safetyCount = 0
			start = fileout.readline()
			end = fileout.readline()

			currentRow = csv_reader.next()
			time = str(currentRow['Time'])[:8]

			while time != start and time != end:

				currentRow = csv_reader.next()
				time = str(currentRow['Time'])[:8]

			if time == start:
				# start writing into csv file
				# if file exists create new file
				files = fnmatch.filter((f for f in os.listdir('./Users/Nouha Tiyal/Desktop/QSUIRP/parse')), 'output*.csv')
				if not files:  # is empty
				    mun = ''
				elif len(files) == 1:
				    mun = '(1)'
				else:
				    # files is supposed to contain 'somefile.txt'
				    files.remove('output.csv')
				    mun = '(%i)' % (int(re.search(r'\(([0-9]+)\)', max(files)).group(1))+1)
				####
				with open("output%s.csv" % mun, mode = 'w') as output:
					fieldnames = ['TimeStamps','Protocol','RSSI']
					writer = csv.DictWriter(output, fieldnames = fieldnames)
					writer.writeheader()
					writer.writerow({'TimeStamps': time, 'Protocol': currentRow['Protocol'], 'RSSI': currentRow['RSSI Value']})

					while time != end and safetyCount < 100:

						currentRow = csv_reader.next()
						time = str(currentRow['Time'])[:8]
						writer.writerow({'TimeStamps': time, 'Protocol': currentRow['Protocol'], 'RSSI': currentRow['RSSI Value']})
						safetyCount += 1

