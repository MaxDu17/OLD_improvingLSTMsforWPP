from ftplib import FTP

ftp = FTP("nomads.ncdc.noaa.gov")
ftp.login()
ftp.cwd("RUC")
ftp.cwd("analysis_only")
ftp.cwd("201201")
ftp.cwd("20120101")

#print(ftp.retrlines('LIST'))

ftp.retrbinary('RETR ruc2anl_130_20120101_0000_000.grb2', open('ruc2anl_130_20120101_0000_000.grb2', 'wb').write)