from pipeline import SetMaker

sm = SetMaker()
sm.create_training_set()
sm.create_validation_set()
sm.next_epoch_test()



for j in range(26):
    m = list()
    for i in range(9):
        m.append(sm.next_sample())
    print(sm.get_label())

    sm.next_epoch()
    print(m)
    print("xxxx")