import numpy as np
import pandas as pd
import csv

lats = pd.read_csv("lats_ruc2anl_130.csv", header=None) #header = none just prevents the first row from being the index
lons = pd.read_csv("lons_ruc2anl_130.csv", header=None)

lons = lons.values
lats = lats.values


#oncat = zip(lats[0], lons[0])
big_list = list()
for i in range(len(lats)):
    concat = zip(lats[i], lons[i])
    concat = list(concat)
    big_list.append(concat)

writer = open("coordinates_ruc2anl_130.csv", "w")
writer_object = csv.writer(writer, lineterminator="\n")
writer_object.writerows(big_list)

