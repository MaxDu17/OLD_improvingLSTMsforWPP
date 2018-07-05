import pygrib
import csv
writer = open("data.csv", "w")
writer_object = csv.writer(writer, lineterminator="\n")
test_file = "test.grb"
opened_file = pygrib.open(test_file)
for i in range(100):
    carrier = opened_file.select()[i]