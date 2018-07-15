from ftplib import FTP
import csv
import os
import time
import pygrib #library that only works in linux
from sys import argv

script, dir_name = argv

category_dict = {0: "surface_pressure", 1: "temp@2M", 2: "wind_gust_speed", 3: "2_M_rel_humid", 4: "temp_gnd_lvl"}

keepers = [223,230,300,295,310] #the data points to keep
headers = ["year", "month", "date", "hour"]
point_to_keep_i =186
point_to_keep_j = 388 #among the large list, it is this single point that we want to keep. This changes with location

#delta = [0,0,2,2,0,2,2,0,2,2,0,2,2,0,2,2,0,2,2] #this compensates for the index-hopping that the dataset does
delta = [0,0,2]
gate_delta = [0,0,1,1,1]
lower_bound = list()

for i in range(3):
    for j in range(5):
        time = "forecast " + str(i) + "-"
        category = category_dict[j]
        concat = time + category
        headers.append(concat)


big_data_ = open(dir_name, "w") #here we get the large file
big_data = csv.writer(big_data_, lineterminator = "\n")
big_data.writerow() #we write the headers here

base_template = list()

file_names = os.listdir(dir_name)
for file_name in file_names[0:3]:
    #ruc2_130_20110102_0500_004
    year = file_name[9:13]
    month = file_name[13:15]
    date = file_name[15:17]
    hour = file_name[19]
    base_template = [year, month, date, hour]

    opened_file = pygrib.open(dir_name + file_name)

    delta_list = [k * delta[i] for k in gate_delta]
    ok_list = [sum(x) for x in zip(delta_list, keepers)]
    for number in ok_list:
        selection = opened_file.select()[number]
        print(selection)
        selection_ = selection.values
        single_pt = selection_[point_to_keep_i][point_to_keep_j]
        base_template.append(single_pt)
        print("extracted: " + str(number) + "\n")

big_data.writerow(base_template)
