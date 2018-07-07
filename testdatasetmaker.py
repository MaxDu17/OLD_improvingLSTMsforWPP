from pipeline import SetMaker

sm = SetMaker()

sm.create_training_set()
sm.next_epoch_test()
for i in range(9):
    a = sm.next_sample()
    print(a)
k = sm.get_label()
print(k)
for i in range(10):
    test = sm.next_epoch_test_continuous()
    print(test)