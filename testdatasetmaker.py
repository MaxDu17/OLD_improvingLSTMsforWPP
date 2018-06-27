from dataset_maker import SetMaker
sm = SetMaker()

for k in range(2):
    sm.next_epoch()

    for i in range(9):
        carrier = sm.next_sample()
        print(carrier)
    print(sm.get_label())
    print("\n")

    sm.create_training_set()