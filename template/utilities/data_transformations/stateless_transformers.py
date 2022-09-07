from sklearn.preprocessing import FunctionTransformer
from numpy import log1p, sqrt, vectorize

def LogTransformer():

    return FunctionTransformer(func=log1p)

def SqrtTransformer():
    
    return FunctionTransformer(func=sqrt)

def standardize_capitalization(a, case='upper'):
    
    if case == 'upper':
        return a.upper()
    if case == 'lower':
        return a.lower()
    else: 
        raise ValueError("'case' argument must be one of ['lower', 'upper']")
        
def CaseStandardizer(case='upper'):
    
    return FunctionTransformer(func=vectorize(standardize_capitalization),
                              kw_args={'case':case})
    