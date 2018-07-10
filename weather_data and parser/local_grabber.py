from ftplib import FTP
import csv
import os
import time
import pygrib #library that only works in linux

hour_big = "00.g2/"
hour_sub = "_0000_" #change me!

file_path = "/home/max/DRIVE/data/0000/"
base_command = 'ruc2_130_'
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
    1: "001", 2: "002", 3: "003",
    4: "004", 5: "005", 6: "006",
    7: "007", 8: "008", 9: "009",
    10: "010", 11: "011", 12: "012",
    13: "013", 14: "014", 15: "015",
    16: "016", 17: "017", 18: "018", 0:"000",
} #how forcast time is expressed

category_dict = {0: "surface_pressure", 1: "temp@2M", 2: "wind_gust_speed", 3: "2_M_rel_humid", 4: "temp_gnd_lvl"}

keepers = [223,230,300,295,310] #the data points to keep

point_to_keep_i =186
point_to_keep_j = 388 #among the large list, it is this single point that we want to keep. This changes with location

headers = ["year", "month", "date", "hour"]

for i in range(19):
    for j in range(5):
        time = "forecast " + str(i) + "-"
        category = category_dict[j]
        concat = time + category
        headers.append(concat)

error_headers = headers
try:
    k = open("2011_TOTALSET.csv")
    print("existing file detected!")
    r = csv.reader(k) #this is crash protection to ensure that everything doesn't get erased
    lines = list(r)
    lines = [int(m) for m in k]
    input("loaded previous data: " + str(len(lines)) + " lines of data. Press enter to continue")
    k.close()
    big_data_ = open("2011_TOTALSET.csv")  # here we get the large file
    big_data = csv.writer(big_data_, lineterminator="\n")
    big_data.writerows(lines)  # we write the headers here
except:
    input("no filled file detected. Starting from scratch. Press enter to continue")
    big_data_ = open("2011_TOTALSET.csv", "w") #here we get the large file
    big_data = csv.writer(big_data_, lineterminator = "\n")
    big_data.writerow(headers) #we write the headers here

error_file_ = open("error_file.csv", "w")
error_file = csv.writer(error_file_, lineterminator = "\n")
error_file.writerow(error_headers)

for l in range(1,13):
    print("I'm on month: " + str(l))
    for j in range(1,32):
        print("I'm on day: " + str(j))
        base_template = [2011, l, j, 0]
        for i in range(0,19):
            print("I'm on forecast hour " + str(i))
            address = file_path + base_command + year + date_dict[l]+date_dict[j] + hour_big +\
                base_command + year + date_dict[l]+date_dict[j] + hour_sub + time_dict[i] + ".grb2"

            try:
                opened_file = pygrib.open(address)
            except:
                if j == 31:
                    print("whoops! This month doesn't have a 31! No problem! Passing ........")
                    base_template = []
                else:
                    print("file not found, this is recorded in the database")
                    error_file.writerow([l, j, hour_sub, "forecast hour " + str(i)])
                    base_template.extend(["err","err","err","err","err",]) #makes it robust to missing files
                continue



            for number in keepers:
                selection = opened_file.select()[number]
                selection_ = selection.values
                single_pt = selection_[point_to_keep_i][point_to_keep_j]
                base_template.append(single_pt)
                print("extracting: " + str(number))

        big_data.writerow(base_template)
