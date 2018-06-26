import csv

file = open("test_file.csv","w")
writer_log = csv.writer(file, lineterminator="\n")
for i in range(100):
    carrier = [i]
    writer_log.writerow(carrier)

print("done!")
