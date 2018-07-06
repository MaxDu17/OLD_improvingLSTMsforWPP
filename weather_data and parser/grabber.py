from ftplib import FTP
import csv
import os
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
19:"19", 20:"20", 21:"22",
23:"23", 24:"24", 25:"25",
26:"26", 27:"27", 28:"28",
29:"29", 30:"30", 31:"31",
} #how date is expressed

time_dict = {
    1:"0100", 2:"0200", 3:"0300"
, 4:"0400", 5:"0500", 6:"0600"
, 7:"0700", 8:"0800", 9:"0900", 0:"0000",
} #how time is expressed

keepers = [223,230,300,295,310] #the data points to keep

point_to_keep_i =186
point_to_keep_j = 388 #among the large list, it is this single point that we want to keep. This changes with location

headers = ["year", "month", "date", "hour", "surface pressure", "2meter tmp", "precipitation",
           "rain", "rel humic", "temperature" ]

big_data_ = open("ruc2anl_130_TOTALSET.csv", "w") #here we get the large file
big_data = csv.writer(big_data_, lineterminator = "\n")
big_data.writerow(headers) #we write the headers here

ftp = FTP("nomads.ncdc.noaa.gov")#logging into ftp w/noaa
ftp.login()

constructed_directory = "/RUC/analysis_only/" + year + date_dict[1] + "/" + year + date_dict[1] + date_dict[1]
#constructed directory is the query command that accesses a specific file location
ftp.cwd(constructed_directory)
for l in range(1,13):
    for j in range(1,32):

        for i in range(24):
            constructed_directory = "/RUC/analysis_only/" + year + date_dict[l] + "/" + year + date_dict[l] + date_dict[j]
            try:
                ftp.cwd(constructed_directory)
                break
            except:
                print("I am not in the right date for the month! Not a problem! Passing...")
                continue

            constructed_command = key_word + base_command + year + date_dict[l] + date_dict[j] +  "_" + time_dict[i] + suffix
            #constructed_file_path = base_command + year + date_dict[2] + date_dict[2] +  "_" + time_dict[i] + suffix
            ftp.retrbinary(constructed_command, open("temp.grb2", 'wb').write)
            opened_file = pygrib.open("temp.grb2")

            base_template = [2011, 1, 1, i]
            for number in keepers:
                print("this is number "+ str(number))
                selection = opened_file.select()[number]
                selection_ = selection.values
                print(selection_)
                single_pt = selection_[point_to_keep_i][point_to_keep_j]
                base_template.append(single_pt)

            big_data.writerow(base_template)
