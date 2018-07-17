from pipeline import DataParser_Weather
from pipeline import SetMaker_Weather
'''
dp = DataParser_Weather()
print(dp.grab_list_range(0,10))
'''
SM = SetMaker_Weather()

SM.create_training_set()
print(SM.next_epoch_waterfall())
print(SM.get_label())
print("\n")
print(SM.next_epoch_waterfall())