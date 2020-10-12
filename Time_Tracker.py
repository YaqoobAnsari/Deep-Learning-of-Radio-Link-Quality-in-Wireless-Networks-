from datetime import datetime

filepath = 'bedroom1ground.txt'
with open(filepath,"w") as fileout:
    # datetime object containing current date and time
    now = datetime.now()

    print("Enter the number of readings to take ::", end=' ')
    reading = int(input())
    filereading = str(reading) + "\n"
    #fileout.write(filereading)
    
    print()
    
    for i in range (0,reading):
        print("READING ",(i+1))
        
        #roundnum = "Reading number " + str(i+1)+ "\n"
        #fileout.write(roundnum)
        #fileout.write("\n")
        
        print("Press Y/y to START time", end=' ')
        start = input()
        if ( start == "Y" or start == "y"):
            start_time = datetime.now().time()#.strftime("%H:%M:%S")
            print(start_time)
            stime = str(start_time) + "\n"
            fileout.write(stime)

        print("Press N/n to STOP time", end=' ')
        stop = input()
        if ( stop == "N" or stop == "n"):
            stop_time = datetime.now().time()#.strftime("%H:%M:%S")
            print(stop_time)
            etime = str(stop_time) + "\n"
            fileout.write(etime)
            
        print()

    
