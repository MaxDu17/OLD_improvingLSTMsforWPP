from data_feeder import DataParser

from hyperparameters import Hyperparameters
class SetMaker:
    dp = DataParser()
    hyp = Hyperparameters()
    master_list = list()
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
        #print(self.counter) #for debugging purposes
        #print(self.master_list)
    def clear_counter(self):
        self.counter = 0

    def get_label(self):
        return self.master_list[self.hyp.FOOTPRINT]

    def next_sample(self):
        if self.batch_counter >=self.hyp.FOOTPRINT:
            raise ValueError("you are infiltrating into key territory! Traceback: dataset_maker/next_sample. "
                             "violation: batch_counter > self.hyp.FOOTPRINT")
        else:
            carrier = self.master_list[self.batch_counter]
            self.batch_counter += 1
            return carrier
