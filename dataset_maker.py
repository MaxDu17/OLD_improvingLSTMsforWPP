from data_feeder import DataParser

from hyperparameters import Hyperparameters
class SetMaker:
    dp = DataParser()
    hyp = Hyperparameters()
    master_list = list()
    #small_list = list()
    counter = 0
    batch_counter = 0
    def test_database(self):
        test = self.dp.grab_list_range(10,20)
        print(len(test))
        print(test)
    def next_epoch(self):
        self.master_list = list()
        self.master_list = self.dp.grab_list_range(self.counter, self.counter+self.hyp.FOOTPRINT+1)
        self.counter += (self.hyp.FOOTPRINT+1)
        self.batch_counter = 0
        print(self.counter)
        print(self.master_list)
    def clear_counter(self):
        self.counter = 0
    def next_sample(self):
        self.batch_counter += 1
        return self.master_list[self.batch_counter]
