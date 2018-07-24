import csv


k = open("../sortedtotalset.csv")
rawset = list(csv.reader(k))
m = open("../test.csv", "w")
writer = csv.writer(m, lineterminator="\n")

for j in range(1,366*24):

    if int(rawset[j][3]) != int((j-1)%24):
        print(rawset[j])
        j = j+1
