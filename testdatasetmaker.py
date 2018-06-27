from pipeline import SetMaker

sm = SetMaker()
sm.create_training_set()
sm.create_validation_set()
sm.next_epoch_test()
k = sm.next_sample()
print(k)
while k < 0.1:
    sm.next_epoch_test()
    k = sm.next_sample()
    print(k)
