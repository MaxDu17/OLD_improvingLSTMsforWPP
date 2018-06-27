from dataset_maker import SetMaker
sm = SetMaker()

sm.next_epoch()

for i in range(9):
    carrier = sm.next_sample()
    print(carrier)
print(sm.get_label())