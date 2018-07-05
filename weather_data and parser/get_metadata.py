import pygrib
import csv
writerlat = open("data.csv", "w")
writer_objectlat = csv.writer(writerlat, lineterminator="\n")

writerlon = open("data.csv", "w")
writer_objectlon = csv.writer(writerlon, lineterminator="\n")
test_file = "ruc2anl_130_20110101_0000_000.grb2"
opened_file = pygrib.open(test_file)
selection = opened_file.select()[0]
lats, lons = selection.latlons()
writer_objectlat.writerows(lats)
writer_objectlon.writerows(lons)