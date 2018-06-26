import pandas as pd
from sqlalchemy import create_engine

class DataParser:
    data = pd.read_csv("2012DATA_HEADLESS.csv")
    clean_data = data[["Month", "Day", "Hour", "Minute", "power (MW)"]]
    power_ds = data[["power (MW)"]]
    power_ds.set_index("power (MW)",inplace= True)

    def print_from_start(self, number):
        return self.power_ds.head(number)

    def grab_list_range(self,start,end):
        #this allows you to get sequential listings
        engine = create_engine('sqlite://', echo=False) #make sql engine
        self.power_ds.to_sql(name = "power", con = engine) #make sql table
        command = "SELECT * FROM power LIMIT " + str(start) + "," + str(end) #make command
        a = engine.execute(command).fetchall() #run command
        clean = [k[0] for k in a] #extract single number
        return clean

