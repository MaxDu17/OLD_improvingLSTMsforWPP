from pipeline import DataParser_Weather
from pipeline import SetMaker_Weather

dp = DataParser_Weather(True)
print(dp.grab_list_range(0,10))
'''
SM = SetMaker_Weather()

SM.create_training_set()
print(SM.next_epoch_waterfall())
'''