import pandas as pd
import numpy as np
from sqlalchemy import create_engine

class DataParser:
    data = pd.read_csv("2012DATA_HEADLESS.csv")
    clean_data = data[["Month", "Day", "Hour", "Minute", "power (MW)"]]
    power_ds = data[["power (MW)"]]
    power_ds.set_index("power (MW)")

    def print_from_start(self, number):
        return self.power_ds.head(number)
    def grab_list(self):
        engine = create_engine('sqlite://', echo=False)
        self.power_ds.to_sql(name = "power", con = engine)
        return engine

dp = DataParser()
engine = dp.grab_list()
a= engine.execute("SELECT * FROM power LIMIT 0,1000").fetchall()
print(a[1])