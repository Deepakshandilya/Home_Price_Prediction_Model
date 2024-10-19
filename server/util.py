import  json
import pickle
import numpy as np


__locations=None
__data_columns=None
__model=None
# global variable


def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index=-1

    x=np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)


def get_location_name():
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts.....start")
    global __data_columns
    global __locations

    with open("./artifacts/columns.json", 'r') as f:
        __data_columns=json.load(f)['data_columns']
        __locations=__data_columns[3:]  # location start after 2 ...0,1,2
    global __model
    with open("./artifacts/banglore_home_prices_model.pickle", 'rb') as f:
        # since it is binary model so we used rb
        __model=pickle.load(f)
    print("loading the artifacts is done")


#         it is json file and it will convert into dictionary


if __name__=="__main__":
    load_saved_artifacts()
    print(get_location_name())

    print(get_estimated_price("1st phase jp nagar", 1000,3,3))
    print(get_estimated_price("Whitefield", 1000,3,3))