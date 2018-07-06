from pipeline import SetMaker

sm = SetMaker()

sm.create_training_set()

for i in range(10):
    test = sm.next_epoch_test_continuous()
    print(test)