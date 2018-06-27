import pandas as pd
from sqlalchemy import create_engine

class DataParser:
    data = pd.read_csv("2012DATA_HEADLESS.csv")
    clean_data = data[["Month", "Day", "Hour", "Minute", "power (MW)"]]
    power_ds = data[["power (MW)"]]
    #power_ds.set_index("power (MW)",inplace= True)

    def print_from_start(self, number):
        return self.power_ds.head(number)


    def grab_list_range(self,start,end):
        self.power_ds.index.name = "index"
        command = str(start)+ "<=index<=" + str(end)
        subset = self.power_ds.query(command)
        clean = [k[0] for k in subset.values]
        return clean

dp = DataParser()

print(dp.grab_list_range(10,20))
