import pandas as pd

class DataParser:
    data = pd.read_csv("2012DATA_HEADLESS.csv") #read file
    #clean_data = data[["Month", "Day", "Hour", "Minute", "power (MW)"]] #extract critical data, not used here
    power_ds = data[["power (MW)"]] #extract a single column

    def print_from_start(self, number):
        return self.power_ds.head(number) #print everything. Seldom used, but is an option

    def grab_list_range(self,start,end): #selects a range to query
        self.power_ds.index.name = "index" #sets index to "index" for ease of query
        command = str(start)+ "<=index<" + str(end) #makes command
        subset = self.power_ds.query(command) #querys the pandas data frame
        clean = [k[0] for k in subset.values] #extracts the value and discards the index value
        return clean #returns the query in a form of a list

