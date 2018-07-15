#!/usr/bin/env python3

import subprocess, os
from ftplib import FTP

directory = ""
# Connect to FTP server and go to the folder
ftp = FTP('ftp.ncdc.noaa.gov')
ftp.login()
ftp.cwd('pub/has/model/HAS011154722/')

# Read the list from a file, and remove white spaces
#with open('tarfilelist.txt') as f:
with open('tarfilelist.txt') as f:
    content = f.readlines()
content = [x.strip() for x in content] 

for filename in content: 
    # Download the file from the FTP server
    command = 'RETR ' + filename
    print("Downloading: " + filename)
    ftp.retrbinary(command, open(filename, 'wb').write)

    # Untar each file to its own folder, after it is done, delete the tar file
    dirname = filename.replace('.tar','')
    tarcommand = 'tar -xf '+filename + ' -C ./'+dirname + '; rm '+filename 
    print("Run it in a subprocess: " + tarcommand)
    subprocess.call(["mkdir", dirname])
    subprocess.Popen(['/bin/sh', '-c', tarcommand], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
