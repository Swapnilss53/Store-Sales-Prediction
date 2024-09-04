import os 
import sys
import pandas as pd

from src.config.configuration import PREPROCESSING_OBJ_PATH,MODEL_FILE_PATH
from src.exception import CustomException
from src.logger import logging
from src.utils import load_model



class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path = PREPROCESSING_OBJ_PATH
            model_path = MODEL_FILE_PATH
            
            preprocessor = load_model(preprocessor_path)
            model = load_model(model_path)
            
            data_scaled = preprocessor.transform(features)
            
            pred = model.predict(data_scaled)
            
            return pred 
        
        except Exception as e:
            logging.info("Error occured in prediction pipeline")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Item_Fat_Content:str,  
                 Item_Visibility:float, 
                 Item_Type:str, 
                 Item_MRP:float,  
                 Outlet_Identifier:str,  
                 Outlet_Establishment_Year:int,
                 Outlet_Location_Type:str,
                 Outlet_Type:str):
        
        self.Item_Fat_Content = Item_Fat_Content
        self.Item_Visibility = Item_Visibility
        self.Item_Type = Item_Type
        self.Item_MRP = Item_MRP
        self.Outlet_Identifier = Outlet_Identifier
        self.Outlet_Establishment_Year = Outlet_Establishment_Year
        self.Outlet_Location_Type = Outlet_Location_Type
        self.Outlet_Type=Outlet_Type
        

        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Item_Fat_Content':[self.Item_Fat_Content],
                'Item_Visibility':[self.Item_Visibility],
                'Item_Type':[self.Item_Type],
                'Item_MRP':[self.Item_MRP],
                'Outlet_Identifier':[self.Outlet_Identifier],
                'Outlet_Establishment_Year':[self.Outlet_Establishment_Year],
                'Outlet_Location_Type':[self.Outlet_Location_Type],
                'Outlet_Type':[self.Outlet_Type]
                

            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame gatherd")
            
            return df
        except Exception as e:
            logging.info("Exception occured in Custom data")
            raise CustomException(e,sys)