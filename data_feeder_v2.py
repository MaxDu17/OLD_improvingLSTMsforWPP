import pandas as pd

class DataParser_Weather:
    """
    this database has a query interval of [x, y)
    meaning that xth number is included, while the yth number is not

    This is the lowest level library; everything else depends on this.
    """
    def __init__(self):
        #self.data = pd.read_csv("Training_sets/83863-2011.csv", skiprows = 3) #read file
        self.data = pd.read_csv("Training_sets/2012DATA.csv", skiprows=3)  # read file
        #clean_data = data[["Month", "Day", "Hour", "Minute", "power (MW)"]] #extract critical data, not used here
        self.combined_data = self.data[["power (MW)"],["wind direction at 100m (deg)"],["wind speed at 100m (m/s)"],
                              ["air temperature at 2m "],["surface air pressure (Pa)"],["density at hub height (kg/m^3)"]] #extracts ALL data
    '''
    def use_foreign(self, file_name):
        self.data = pd.read_csv(file_name, skiprows = 3) #read file
        #clean_data = data[["Month", "Day", "Hour", "Minute", "power (MW)"]] #extract critical data, not used here
        self.combined_data = self.data[["power (MW)"]] #extract a single column
    '''
    def print_from_start(self, number):
        return self.combined_data.head(number) #print everything. Seldom used, but is an option

    def dataset_size(self):
        return self.combined_data.size

    def grab_list_range(self,start,end): #selects a range to query
        self.combined_data.index.name = "index" #sets index to "index" for ease of query
        command = str(start) + "<=index<" + str(end) #makes command
        subset = self.combined_data.query(command) #querys the pandas data frame
        clean = [round(k[0],3) for k in subset.values] #extracts the value and discards the index value
        return clean #returns the query in a form of a list

    def grab_element(self, element):
        array = self.combined_data.values
        clean_array = [k[0] for k in array]
        return clean_array[element]

