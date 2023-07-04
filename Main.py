from ML_Helpers.Analyses import *
from ML_Helpers.ResultInterpretationHelpers import *

a1_description = "Effect of lookback days in predicting score"
test_parameter = "Lookback days"
name = "Lookback_Analysis"
path = "./ModelResults"

a2_description = "Effect of lookback days in predicting score - Temp + Humid"
test_parameter_a2 = "Lookback days - Temp and Humid"
name_a2 = "Lookback_Analysis_Temp_Humid"

a3_description = "Effect of lookback days in predicting score - Humid only"
test_parameter_a3 = "Lookback days - Humid only"
name_a3 = "Lookback_Analysis_Humid"


a1 = Analysis1(name=name, path=path, description=a1_description,
               parameter="lookback_days")
a1.run()
# a2 = Analysis2(name=name_a2, path=path, description=a2_description,
#                parameter="lookback_days")
# a3 = Analysis3(name=name_a3, path=path, description=a3_description,
#                parameter="lookback_days")
# runMultipleTimes(a2, 3, name_a2, path)
# a2.plot_r2_curves()
#
# runMultipleTimes(a3, 3, name_a3, path)
# a2.plot_r2_curves()

# I will add a comment here

# plot.show()