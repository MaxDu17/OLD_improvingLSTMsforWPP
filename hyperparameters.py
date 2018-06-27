class Hyperparameters: #the class that defines the hyperparameters is here
    FOOTPRINT = 9 #how many steps back you take. This is a critical adjustment point
    LEARNING_RATE = 0.01
    EPOCHS = 25000
    TRAIN_PERCENT = 0.6
    VALIDATION_PERCENT = 0.002
    class Info: #not used in real code, just as metadata
        DATASET_SIZE = 150120
        TRAIN_SIZE = 63072
        TEST_SIZE = 42048
        TEST_SIZE_LIMIT = 42038 #this is the last number you can ask for
        LOSS_FUNCTION = "squared error"

