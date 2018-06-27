from dataset_maker import SetMaker
sm = SetMaker()
sm.create_training_set()
sm.create_validation_set()
for k in range(300):
    sm.next_epoch_valid()

