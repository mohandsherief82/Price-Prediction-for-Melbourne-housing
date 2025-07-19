from helpers import *
import pandas as pd


print("Welcome to Housing Price Predictor")

r = int(input("Number of Rooms: "))
d = float(input("Distance: "))
lds = float(input("Land Size: "))
pc = float(input("Postcode: "))
bd = float(input("Bedroom2: "))
bth = float(input("Bathroom: "))
cr = float(input("Car: "))
bldga = float(input("BuildingArea: "))
yb = float(input("Year Built: "))
ltt = float(input("Lattitude: "))
lgd = float(input("Longtitude: "))
ptc = float(input("Propertycount: "))
# typ = str(input("Housing (H) or Urban (U) or Transportation (T): "))

validation = pd.DataFrame({
    'Rooms': r,
    'Distance': d,
    'Postcode': pc,
    'Bedroom2': bd,
    'Bathroom': bth,
    'Car': cr,
    'Landsize': lds,
    'BuildingArea': bldga,
    'YearBuilt': yb,
    'Lattitude': ltt,
    'Longtitude': lgd,
    'Propertycount': ptc
}, index=['validation'])

# if typ.lower() == 'h' or typ.lower() == 'housing':
#     validation = pd.DataFrame({
#         'Rooms': r,
#         'Distance': d,
#         'Landsize': lds,
#         'h': 1.0,
#         't': 0.0,
#         'u': 0.0
#     }, index=['validation'])
# elif typ.lower() == 't' or typ.lower() == 'urban':
#     validation = pd.DataFrame({
#         'Rooms': r,
#         'Distance': d,
#         'Landsize': lds,
#         'h': 0.0,
#         't': 1.0,
#         'u': 0.0
#     }, index=['validation'])
# elif typ.lower() == 'u' or typ.lower() == 'transportation':
#     validation = pd.DataFrame({
#         'h': 0.0,
#         't': 0.0,
#         'u': 1.0,
#         'PI': 0.0,
#         'S': 0.0,
#         'SA': 0.0,
#         'SP': 0.0,
#         'VB': 0.0,
#         'Rooms': r,
#         'Distance': d,
#         'Landsize': lds
#     }, index=['validation'])


s = (validation.dtypes == 'object')
object_cols = list(s[s].index)
validation = encode1(validation, object_cols)

my_model = model_user('model.pkl')

value = float(my_model.predict(validation))

get_value(value)

print("Done")
