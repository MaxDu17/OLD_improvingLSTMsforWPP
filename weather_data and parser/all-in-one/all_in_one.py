#!/usr/bin/env python3

import subprocess, os
from ftplib import FTP
path = "/home/max/DRIVE/data/"
len_path = len(path)
directory = "HAS011154726"
# Connect to FTP server and go to the folder
ftp = FTP('ftp.ncdc.noaa.gov')
ftp.login()
ftp.cwd('pub/has/model/' + directory + '/')
try:
    content = ftp.nlst()
except:
    print("error, no files found. Quitting...")
    quit()

# Read the list from a file, and remove white spaces
#with open('tarfilelist.txt') as f:
with open('tarfilelist.txt') as f:
    content = f.readlines()
content = [x.strip() for x in content] 

for filename in content: 
    # Download the file from the FTP server
    command = 'RETR ' + filename
    print("Downloading: " + filename)
    overarching_name = path + filename
    ftp.retrbinary(command, open(overarching_name, 'wb').write)

    # Untar each file to its own folder, after it is done, delete the tar file
    dirname = overarching_name.replace('.tar','')
    tarcommand = 'tar -xf '+overarching_name + ' -C ' + dirname + '; rm '+overarching_name + \
                 '; cd /home/max/SHARED; python3 local_grabber_single.py ' + dirname
    print("Extracting tar: " + filename)
    subprocess.call(["mkdir", dirname])
    subprocess.Popen(['/bin/sh', '-c', tarcommand], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
