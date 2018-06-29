from data_feeder import DataParser
from hyperparameters import Hyperparameters

class SetMaker:
    def __init__(self):
        self.dp = DataParser()
        self.hyp = Hyperparameters()
        self.master_list = list()
        self.counter = 0
        self.batch_counter = 0
        self.training_set_size = 0
        self.valid_counter = 0
        self.validation_set_size = 0

    def test_database(self): #checks that the query is in good shape.
        test = self.dp.grab_list_range(10,20)
        print(len(test))
        print(test)

    def create_training_set(self):
        self.training_set_size = self.hyp.TRAIN_PERCENT * self.dp.dataset_size()
        self.test_counter = self.training_set_size

    def create_validation_set(self):
        self.validation_set_size = int(self.hyp.VALIDATION_PERCENT * self.dp.dataset_size()) #just casting to whole #

    def next_epoch(self):
        self.master_list = list()
        if self.counter + self.hyp.FOOTPRINT+1 > self.training_set_size:
            self.clear_counter()
        self.master_list = self.dp.grab_list_range(self.counter, self.counter+self.hyp.FOOTPRINT+1)
        self.counter += 1
        self.batch_counter = 0
        #print(self.counter) #for debugging purposes
        #print(self.master_list)

    def next_epoch_test(self):
        if self.test_counter + self.hyp.FOOTPRINT + 1 > self.dp.dataset_size():
            raise ValueError("you have reached the end of the test set. Violation dataset_maker/next_epoch_test")
        self.master_list = list()
        self.master_list = self.dp.grab_list_range(self.test_counter, self.test_counter + self.hyp.FOOTPRINT + 1)
        self.test_counter += 1
        self.batch_counter = 0

    def next_epoch_valid(self):
        if self.valid_counter + self.hyp.FOOTPRINT + 1 > self.validation_set_size:
            raise ValueError("you have reached the end of the validation. Please check your code"
                             " for boundary cases. Violation dataset_maker/next_epoch_valid")
        self.master_list = list()
        self.master_list = self.dp.grab_list_range(self.valid_counter, self.valid_counter + self.hyp.FOOTPRINT + 1)
        self.valid_counter += 1
        self.batch_counter = 0

    def clear_valid_counter(self):
        self.valid_counter =0

    def clear_counter(self):
        self.counter = 0

    def get_label(self):
        return self.master_list[self.hyp.FOOTPRINT]

    def next_sample(self):
        if self.batch_counter >=self.hyp.FOOTPRINT:
            raise ValueError("you are infiltrating into key territory! Traceback: dataset_maker/next_sample. "
                             "Violation: batch_counter > self.hyp.FOOTPRINT")
        else:
            carrier = self.master_list[self.batch_counter]
            self.batch_counter += 1
            return carrier
