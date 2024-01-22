from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cachetools import TTLCache
import pandas as pd
from pandasai import Agent
import os
from pandasai import SmartDataframe
from pandasai.llm import AzureOpenAI
import base64
import orjson
import typing
# from loguru import logger
import numpy as np
from fastapi.responses import JSONResponse

from dotenv import load_dotenv
load_dotenv()


class ORJSONResponse(JSONResponse):
    media_type = "application/json"
    def render(self, content: typing.Any) -> bytes:
        return orjson.dumps(content)
    
app = FastAPI(default_response_class=ORJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class QueryRequest(BaseModel):
    """    Pydantic BaseModel for representing the structure of the request for querying data."""
    query: str

class QueryResult(BaseModel):
    """    Pydantic BaseModel for representing the structure of the response for a query result."""
    result: str
    last_code_generated: str
    

class BankDataAnalysis:
    def __init__(self):
        """ Initializes the BankDataAnalysis class with necessary configurations, including cache setup."""

        self.OPENAI_API_KEY=os.environ['OPENAI_API_KEY']
        self.api_base=os.environ["api_base"]
        self.AZURE_OPENAI_API_ENDPOINT=os.environ["AZURE_OPENAI_API_ENDPOINT"]
        self.OPENAI_API_VERSION=os.environ["OPENAI_API_VERSION"]
        self.OPENAI_API_TYPE=os.environ['OPENAI_API_TYPE']
        self.dl = None  # Initialize dl to None
        
        # Configure cache with a time-to-live (TTL) of 1 hour
        self.cache = TTLCache(maxsize=300, ttl=3600)
        self.keywords= ["Acc_Holder", "Acc_Status", "Acc_Type", "Acount", "Add_Type", "Adress","Branch","Currency","custom_address", "Cust_Tele_Num","customer", "Emp","House_Own","Inden_Doc", "InterestRate", "MS", "Prod_IR",
                        "Prod_Type","Product","Proof_Add","Tel_Num_Type","Tel_Number","Trans"]


    def load_data(self):

        """Loads data from Parquet files into Pandas DataFrames."""

        self.df_Acc_Holder= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/AccountHolder.csv")
        self.df_customer= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/customer.csv")
        self.df_Acount= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/Account.csv" )
        self.df_Acc_Status= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/AccountStatus.csv")
        self.df_Acc_Type= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/AccountType.csv")
        self.df_Add_Type= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/AddressType.csv")
        self.df_Adress= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/Address.csv")
        self.df_Branch= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/Branch.csv")
        self.df_Currency= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/Currency.csv")
        self.df_custom_address= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/CustomerAddress.csv")
        self.df_Cust_Tele_Num= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/CustomerTelephoneNumber.csv")
        self.df_Emp= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/Employee.csv")
        self.df_House_Own= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/HouseOwnership.csv")
        self.df_Inden_Doc= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/IndentificationDocument.csv")
        self.df_InterestRate= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/InterestRate.csv")
        self.df_MS= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/MaritalStatus.csv")
        self.df_Prod_IR= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/ProductInterestRate.csv")
        self.df_Prod_Type= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/ProductType.csv")
        self.df_Product= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/Product.csv")
        self.df_Proof_Add= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/ProofOfAddress.csv")
        self.df_Tel_Num_Type= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/TelephoneNumberType.csv")
        self.df_Tel_Number= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/TelephoneNumber.csv")
        self.df_Trans= pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Data_Model/Transaction.csv")    
        
        pass

    def initialize_llm(self):

        """Initializes the Language Model (llm) for natural language processing."""

        llm = AzureOpenAI(deployment_name="chat",
                          model_name="gpt-35-turbo",
                          api_base=os.environ['api_base'],
                          azure_endpoint=os.environ['AZURE_OPENAI_API_ENDPOINT'],
                          temperature=0,
                          model_kwargs={"api_type": "azure",
                                        "api_version": "2023-07-01-preview"})
        
        self.Acc_Holder = SmartDataframe(self.df_Acc_Holder, name="Acc_Holder")
        self.Acc_Status = SmartDataframe(self.df_Acc_Status, name="Acc_Status")
        self.Acc_Type = SmartDataframe(self.df_Acc_Type, name="Acc_Type")
        self.Acount = SmartDataframe(self.df_Acount, name="Acount")
        self.Add_Type = SmartDataframe(self.df_Add_Type, name="Add_Type")
        self.Adress = SmartDataframe(self.df_Adress, name="Adress")
        self.Branch = SmartDataframe(self.df_Branch, name="Branch")
        self.Currency = SmartDataframe(self.df_Currency, name="Currency")
        self.custom_address = SmartDataframe(self.df_custom_address, name="custom_address")
        self.Cust_Tele_Num = SmartDataframe(self.df_Cust_Tele_Num, name="Cust_Tele_Num")
        self.customer = SmartDataframe(self.df_customer, name="customer")
        self.Emp = SmartDataframe(self.df_Emp, name="Emp")
        self.House_Own = SmartDataframe(self.df_House_Own, name="House_Own")
        self.Inden_Doc = SmartDataframe(self.df_Inden_Doc, name="Inden_Doc")
        self.InterestRate = SmartDataframe(self.df_InterestRate, name="InterestRate")
        self.MS = SmartDataframe(self.df_MS, name="MS")
        self.Prod_IR = SmartDataframe(self.df_Prod_IR, name="Prod_IR")
        self.Prod_Type = SmartDataframe(self.df_Prod_Type, name="Prod_Type")
        self.Product = SmartDataframe(self.df_Product, name="Product")
        self.Proof_Add = SmartDataframe(self.df_Proof_Add, name="Proof_Add")
        self.Tel_Num_Type = SmartDataframe(self.df_Tel_Num_Type, name="Tel_Num_Type")
        self.Tel_Number = SmartDataframe(self.df_Tel_Number, name="Tel_Number")       
        self.Trans = SmartDataframe(self.df_Trans, name="Trans")

        self.dl = Agent([self.Acc_Holder, self.Acc_Status, self.Acc_Type, self.Acount, self.Add_Type, self.Adress, self.Branch, self.Currency,
                        self.custom_address, self.Cust_Tele_Num, self.customer, self.Emp, self.House_Own, self.Inden_Doc, self.InterestRate, self.MS, self.Prod_IR,
                        self.Prod_Type, self.Product, self.Proof_Add, self.Tel_Num_Type, self.Tel_Number, self.Trans],config={"llm": llm, "custom_whitelisted_dependencies": ["os"], "verbose": True, "conversational": False},memory_size=20)
        
        # self.dl = Agent([self.Acc_Holder,self.Acount,self.customer],config={"llm": llm, "custom_whitelisted_dependencies": ["os"], "verbose": True, "conversational": False},memory_size=20)

        return self.dl
    
    def process_keywords_and_print_dataframes(self, result_keyword):
        for df_name in result_keyword:
            dataframe = getattr(self, df_name, None)
            if dataframe is not None:
                print(f"Dataframe for {df_name}:")
                result = {df_name: dataframe.to_dict(orient='records') for i, df_name in enumerate(result_keyword, start=1)}
 
            else:
                print(f"Dataframe {df_name} not found.")
        return result

    def query_data(self, query_req: QueryRequest):

        """ Queries data using the initialized data lake and returns the result."""

        # Check if result is already in the cache
        query_hashable = hash(str(query_req))

        cached_result = self.cache.get(query_hashable)
        if cached_result:
            return cached_result

        if self.dl is None:
            raise HTTPException(status_code=500, detail="Datalake not initialized")

        result = self.dl.chat(query_req.query)
        
        code_df=self.dl.last_code_executed
        print("final",(code_df))

        result_keyword =self.check_keywords_occurrence(code_df, self.keywords)

        if result_keyword:
            print("Keywords with multiple occurrences:", result_keyword)
        else:
            print("No keywords appear more than once.")

        self.process_keywords_and_print_dataframes(result_keyword)
        response_df=self.process_keywords_and_print_dataframes(result_keyword)

        print("response_df",self.process_keywords_and_print_dataframes(result_keyword))
        response = {"result": result, "last_code_generated": self.dl.last_code_executed,"dataframe":response_df}
        # Store the result in the cache
        self.cache[query_hashable] = response
        print("response",self.cache[query_hashable])

        return response
    
    def check_keywords_occurrence(self,result, keywords):


        keyword_count = {keyword: result.count(keyword) for keyword in keywords}

        # Filter keywords that appear more than once
        multiple_occurrence_keywords = [keyword for keyword, count in keyword_count.items() if count > 1]

        return multiple_occurrence_keywords
        

bank_analysis = BankDataAnalysis()
bank_analysis.load_data()
bank_analysis.initialize_llm()

def image_to_base64(image_path):

    """Converts an image file to base64 encoding."""

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_image

def resultvalue(result):

    """Processes the query result and handles image conversion to base64 if applicable."""

    result['result'] = str(result['result'])
    # logger.debug("2", type(result['result']), result['result'])

    if ".png" not in result['result']:
        print("no image here")
        response = result
        print(response)
        # logger.debug("result= = =",result)
        return response
        
    elif ".png" in result['result']:
        print("Image path:", result)
        base64_image = image_to_base64(result['result'])
        response = {
            "image_path": result,
            "base64_image": base64_image
        }
        return response

@app.post("/query/")
async def query_endpoint(query_req: QueryRequest):
    
    """FastAPI endpoint for handling queries and returning the results."""

    result = bank_analysis.query_data(query_req)
    print("Output",result)
    return resultvalue(result)


