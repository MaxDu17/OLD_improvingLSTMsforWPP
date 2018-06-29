class Hyperparameters: #the class that defines the hyperparameters is here
    FOOTPRINT = 9 #how many steps back you take. This is a critical adjustment point
    LEARNING_RATE = 0.001
    EPOCHS = 20001
    TRAIN_PERCENT = 0.6
    VALIDATION_PERCENT = 0.002 #nullifies for now
    VALIDATION_NUMBER = 30
    cell_dim = 25
    hidden_dim = 25
    TEST = True
    SAVER_JUMP = 2000
    SUMMARY_JUMP = 50

    class Info: #not used in real code, just as metadata
        DATASET_SIZE = 150120
        TRAIN_SIZE = 63072
        #TEST_SIZE = 42048
        TEST_SIZE = 1000
        EVAULATE_TEST_SIZE = 1000
        VALID_SIZE = 210
        VALID_SIZE_LIMIT = 200 #highest number you can ask for
        #TEST_SIZE_LIMIT = 42038 #this is the last number you can ask for
        LOSS_FUNCTION = "squared_error"

