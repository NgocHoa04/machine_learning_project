import numpy as np
from sklearn.preprocessing import FunctionTransformer

def precip_scale_function():
    return FunctionTransformer(func=lambda x: np.log1p(x),
                               validate=True,
                               feature_names_out='one-to-one')

def solarradiation_scale_function():
    return FunctionTransformer(
        lambda x: np.log1p(x)/np.log1p(x.max()), 
        validate=True,
        feature_names_out='one-to-one'
    )

def humidity_scale_function():
    return FunctionTransformer(
        lambda x: x / 100.0, 
        validate=True,
        feature_names_out='one-to-one'
    )

def percentage_scale_function():
    return FunctionTransformer(
        lambda x: x / 100.0, 
        validate=True,
        feature_names_out='one-to-one'
    )

def windspeed_scale_function():
    return FunctionTransformer(
        lambda x: x / 100.0, 
        validate=True,
        feature_names_out='one-to-one'
    )