import pickle
import pandas as pd
from flask import Flask, request, Response 
from healthinsurance.healthinsurance import HealthInsurance

#loading model
path = 'model/'
model = pickle.load(open(path + 'model.pkl','rb'))

app = Flask(__name__)

@app.route('/predict', methods = ['POST'])
def health_insurance_predict():
    test_json = request.get_json()
    
    if test_json: # there is data
        if isinstance( test_json, dict ): # unique example
            test_raw = pd.DataFrame( test_json, index=[0] )
            
        else: # multiple example
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
            
        # Instantiate Rossmann class
        pipeline = HealthInsurance()
        print("1")
        # feature engineering
        df1 = pipeline.feature_engineering( test_raw )
        print('2')
        
        # data preparation
        df2 = pipeline.data_preparation( df1 )
        
        print("3")
        # prediction
        df_response = pipeline.get_prediction( model, test_raw, df2 )
        
        return df_response
    
    else:
        return Response( '{}', status=200, mimetype='application/json' )
    
if __name__ == '__main__':
    app.run( '127.0.0.1' )