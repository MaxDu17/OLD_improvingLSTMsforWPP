from ftplib import FTP
import csv
import os
import time
import pygrib #library that only works in linux

key_word = "RETR "
base_command = 'ruc2anl_130_'
year = '2011'
suffix = "_000.grb2"
date_dict = {
1:"01", 2:"02", 3:"03",
4:"04", 5:"05", 6:"06",
7:"07", 8:"08", 9:"09",
10:"10", 11:"11", 12:"12",
13:"13", 14:"14", 15:"15",
16:"16", 17:"17", 18:"18",
19:"19", 20:"20", 21:"21",
23:"23", 24:"24", 25:"25",
26:"26", 27:"27", 28:"28",
29:"29", 30:"30", 31:"31",
22:"22"
} #how date is expressed

time_dict = {
    1: "0100", 2: "0200", 3: "0300",
    4: "0400", 5: "0500", 6: "0600",
    7: "0700", 8: "0800", 9: "0900",
    10: "1000", 11: "1100", 12: "1200",
    13: "1300", 14: "1400", 15: "1500",
    16: "1600", 17: "1700", 18: "1800",
    19: "1900", 20: "2000", 21: "2100",
    22: "2200",23: "2300", 0:"0000",
} #how time is expressed

keepers = [223,230,300,295,310] #the data points to keep

point_to_keep_i =186
point_to_keep_j = 388 #among the large list, it is this single point that we want to keep. This changes with location
lowbound = list()
headers = ["year", "month", "date", "hour", "surface_pressure", "2M_temperature", "wind_gust_speed",
           "2M_rel_humidity", "surface_temperature"]
error_headers = ["year", "month", "date"]

try:
    k = open("2011_TOTALSET.csv")
    print("existing file detected!")
    r = csv.reader(k) #this is crash protection to ensure that everything doesn't get erased
    lines = list(r)
    lines = [int(m) for m in k]
    lowbound = lines.pop()
    print("starting from (Y,M,D,H):", str(lowbound[0:3]))
    k.close()
    big_data_ = open("2011_TOTALSET.csv")  # here we get the large file
    big_data = csv.writer(big_data_, lineterminator="\n")
    big_data.writerows(lines)  # we write the headers here
except:
    print("starting from scratch")
    big_data_ = open("2011_TOTALSET.csv", "w") #here we get the large file
    big_data = csv.writer(big_data_, lineterminator = "\n")
    big_data.writerow(headers) #we write the headers here
    lowbound = [1,1,0]

error_file_ = open("error_file.csv", "w")
error_file = csv.writer(error_file_, lineterminator = "\n")
error_file.writerow(error_headers)


for l in range(lowbound[0],13):
    print("I'm on month: " + str(l))
    for j in range(lowbound[1],32):
        print("I'm on day: " + str(j))
        for i in range(lowbound[2],24):
            print("I'm on hour: " + str(i))

            try:
                pass # do something here
            except:
                print("I am not in the right date for the month! Not a problem! Passing...")
                continue


            try:
                pass #do somethign here
            except:
                print("file not found, this is recorded in the database")
                error_file.writerow([l,j,i])
                big_data.writerow(["error------------------------------------------------"])
                continue

            opened_file = pygrib.open("temp.grb2")

            base_template = [2011, l, j, i]
            for number in keepers:
                selection = opened_file.select()[number]
                selection_ = selection.values
                single_pt = selection_[point_to_keep_i][point_to_keep_j]
                base_template.append(single_pt)
                print("extracting: " + str(number))

            big_data.writerow(base_template)
