import pandas as pd
import numpy as np
from sqlalchemy import create_engine

class DataParser:
    data = pd.read_csv("2012DATA_HEADLESS.csv")
    clean_data = data[["Month", "Day", "Hour", "Minute", "power (MW)"]]
    power_ds = data[["power (MW)"]]
    power_ds.set_index("power (MW)",inplace= True)

    def print_from_start(self, number):
        return self.power_ds.head(number)
    def grab_list(self,start, end):
        engine = create_engine('sqlite://', echo=False)
        self.power_ds.to_sql(name = "power", con = engine)
        command = "SELECT * FROM power LIMIT " + str(start) + "," + str(end)
        a = engine.execute(command).fetchall()
        clean = [k[0] for k in a]
        return clean

dp = DataParser()
answer = dp.grab_list(0,1000)

print(answer[4])