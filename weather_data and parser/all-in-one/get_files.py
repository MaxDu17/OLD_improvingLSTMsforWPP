from ftplib import FTP
path = "/home/max/DRIVE/data/"
len_path = len(path)
directory = "HAS011154726"
# Connect to FTP server and go to the folder
ftp = FTP('ftp.ncdc.noaa.gov')
ftp.login()
k = open("tarfiles.csv", "w")
tarfile_list = csv.writer(k, lineterminator ='\'')
ftp.cwd('pub/has/model/' + directory + '/')
try:
    content = ftp.nlst()
except:
    print("error, no files found. Quitting...")
    quit()
