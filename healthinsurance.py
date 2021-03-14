import pickle
import numpy as np
import pandas as pd

class HealthInsurance:
    
    def __init__(self):
        self.home_path 							= 'parameters/'
        self.age_scaler 						= pickle.load(open(self.home_path + 'rescaling_age.pkl','rb'))
        self.annual_premium_scaler  			= pickle.load(open(self.home_path + 'rescaling_annual_premium.pkl','rb'))
        self.vintage_scaler 					= pickle.load(open(self.home_path + "rescaling_vintage.pkl",'rb'))
        self.target_encode_gender_scaler 		=      pickle.load( open( self.home_path + 'target_encode_gender.pkl' ,'rb') )
        self.target_encode_region_code_scaler 	= pickle.load( open( self.home_path + 'target_encode_region_code.pkl' ,'rb') )
        self.fe_policy_sales_channel_scaler 	=   pickle.load( open( self.home_path + 'fe_policy_sales_channel.pkl','rb' ) )
              
    def feature_engineering(self, df1):
        #vehicle_age
        df1['vehicle_age']					 = df1['vehicle_age'].apply(lambda x:'over_2_years' if x == "> 2 Years" 
                                              else 'between_1_2_year' if x == '1-2 Years'
                                              else 'below_1_year')

        #vehicle_damage
        df1['vehicle_damage'] 				= df1['vehicle_damage'].apply(lambda x: 1 if x == "Yes" else 0)    
        return df1   

    def data_preparation(self, df2):
        ### RESCALING ###
        #age
        df2['age'] 							= self.age_scaler.fit_transform(df2[['age']])
        #annual_premium
        df2['annual_premium'] 				= self.annual_premium_scaler.fit_transform(df2[['annual_premium']])
        #vintage
        df2['vintage'] 						= self.vintage_scaler.fit_transform(df2[['vintage']])
        
        ### ENCODING ###
        df2.loc[:,'gender'] 				= df2['gender'].map(self.target_encode_gender_scaler)  
        
        df2.loc[:,'region_code'] 			= df2['region_code'].map(self.target_encode_region_code_scaler)        
        
        df2.loc[:,'policy_sales_channel'] 	= df2['policy_sales_channel'].map(self.fe_policy_sales_channel_scaler)       
        
        df2 								= pd.get_dummies(df2, prefix = 'vehicle_age', columns = ['vehicle_age'])

        cols_selected 						= ['vintage','annual_premium','age','region_code','vehicle_damage','policy_sales_channel','previously_insured']

        
        return df2[cols_selected]
    
    def get_prediction( self, model, original_data, test_data ):
        # model prediction
        pred = model.predict_proba( test_data )

        # join prediction into original data
        original_data['prediction'] = pred

        return original_data.to_json( orient='records', date_format='iso' )