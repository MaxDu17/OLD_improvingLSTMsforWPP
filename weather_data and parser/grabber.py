from ftplib import FTP
import csv
import os
key_word = "RETR "
base_command = 'ruc2anl_130_'
year = '2012'
suffix = "_000.grb2"
date_dict = {
    1:"01", 2:"02", 3:"03"
, 4:"04", 5:"05", 6:"06"
, 7:"07", 8:"08", 9:"09"
}

time_dict = {
    1:"0100", 2:"0200", 3:"0300"
, 4:"0400", 5:"0500", 6:"0600"
, 7:"0700", 8:"0800", 9:"0900", 0:"0000",
}

ftp = FTP("nomads.ncdc.noaa.gov")
ftp.login()
constructed_directory = "RUC/analysis_only/" + year + date_dict[1] + "/" + year + date_dict[1] + date_dict[1]
ftp.cwd(constructed_directory)
for i in range(10):
    constructed_command = key_word + base_command + year + date_dict[1] + date_dict[1] +  "_" + time_dict[i] + suffix
    constructed_file_path = base_command + year + date_dict[1] + date_dict[1] +  "_" + time_dict[i] + suffix
    ftp.retrbinary(constructed_command, open(constructed_file_path, 'wb').write)
