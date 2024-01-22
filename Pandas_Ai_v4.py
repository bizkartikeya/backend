from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cachetools import TTLCache
from datetime import timedelta
import pandas as pd
from pandasai import Agent
import os
from pandasai import SmartDataframe
from pandasai.llm import AzureOpenAI
from pandasai import SmartDatalake
import base64
# from loguru import logger
# from dotenv import load_dotenv
# load_dotenv()


app = FastAPI()

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
        os.environ['OPENAI_API_KEY'] = "d06dfb8d9f2e4d9696bfb614c8da2a69"
        os.environ["api_base"] = "https://cog-ccgt327nxe3tc.openai.azure.com/"
        os.environ["AZURE_OPENAI_API_ENDPOINT"] = "https://cog-ccgt327nxe3tc.openai.azure.com/"
        os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
        os.environ['OPENAI_API_TYPE'] = "azure"
        self.dl = None  # Initialize dl to None
        # Configure cache with a time-to-live (TTL) of 1 hour
        self.cache = TTLCache(maxsize=300, ttl=3600)

    def load_data(self):
        """Loads data from Parquet files into Pandas DataFrames."""

        self.df_Acc_Holder= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/AccountHolder.parquet", engine='pyarrow')
        self.df_Acc_Status= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/AccountStatus.parquet", engine='pyarrow')
        self.df_Acc_Type= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/AccountType.parquet", engine='pyarrow')
        self.df_Acc= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/Account.parquet", engine='pyarrow')
        self.df_Add_Type= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/AddressType.parquet", engine='pyarrow')
        self.df_Add= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/Address.parquet", engine='pyarrow')
        self.df_Branch= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/Branch.parquet", engine='pyarrow')
        self.df_Curr= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/Currency.parquet", engine='pyarrow')
        self.df_cust_add= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/CustomerAddress.parquet", engine='pyarrow')
        self.df_Cust_Tele_Num= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/CustomerTelephoneNumber.parquet", engine='pyarrow')
        self.df_cust= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/customer.parquet", engine='pyarrow')
        self.df_Emp= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/Employee.parquet", engine='pyarrow')
        self.df_House_Own= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/HouseOwnership.parquet", engine='pyarrow')
        self.df_Inden_Doc= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/IndentificationDocument.parquet", engine='pyarrow')
        self.df_IR= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/InterestRate.parquet", engine='pyarrow')
        self.df_MS= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/MaritalStatus.parquet", engine='pyarrow')
        self.df_Prod_IR= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/ProductInterestRate.parquet", engine='pyarrow')
        self.df_Prod_Type= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/ProductType.parquet", engine='pyarrow')
        self.df_Prod= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/Product.parquet", engine='pyarrow')
        self.df_Proof_Add= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/ProofOfAddress.parquet", engine='pyarrow')
        self.df_Tel_Num_Type= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/TelephoneNumberType.parquet", engine='pyarrow')
        self.df_Tel_Num= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/TelephoneNumber.parquet", engine='pyarrow')
        self.df_Trans= pd.read_parquet("/Users/kartikeyshrivastav/Desktop/backend/Data_Model_Parquet/Transaction.parquet", engine='pyarrow')    
        
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
        self.Acc = SmartDataframe(self.df_Acc, name="Acc")
        self.Add_Type = SmartDataframe(self.df_Add_Type, name="Add_Type")
        self.Add = SmartDataframe(self.df_Add, name="Add")
        self.Branch = SmartDataframe(self.df_Branch, name="Branch")
        self.Curr = SmartDataframe(self.df_Curr, name="Curr")
        self.cust_add = SmartDataframe(self.df_cust_add, name="cust_add")
        self.Cust_Tele_Num = SmartDataframe(self.df_Cust_Tele_Num, name="Cust_Tele_Num")
        self.cust = SmartDataframe(self.df_cust, name="cust")
        self.Emp = SmartDataframe(self.df_Emp, name="Emp")
        self.House_Own = SmartDataframe(self.df_House_Own, name="House_Own")
        self.Inden_Doc = SmartDataframe(self.df_Inden_Doc, name="Inden_Doc")
        self.IR = SmartDataframe(self.df_IR, name="IR")
        self.MS = SmartDataframe(self.df_MS, name="MS")
        self.Prod_IR = SmartDataframe(self.df_Prod_IR, name="Prod_IR")
        self.Prod_Type = SmartDataframe(self.df_Prod_Type, name="Prod_Type")
        self.Prod = SmartDataframe(self.df_Prod, name="Prod")
        self.Proof_Add = SmartDataframe(self.df_Proof_Add, name="Proof_Add")
        self.Tel_Num_Type = SmartDataframe(self.df_Tel_Num_Type, name="Tel_Num_Type")
        self.Tel_Num = SmartDataframe(self.df_Tel_Num, name="Tel_Num")       
        self.Trans = SmartDataframe(self.df_Trans, name="Trans")

        self.dl = Agent([self.Acc_Holder, self.Acc_Status, self.Acc_Type, self.Acc, self.Add_Type, self.Add, self.Branch, self.Curr,
                        self.cust_add, self.Cust_Tele_Num, self.cust, self.Emp, self.House_Own, self.Inden_Doc, self.IR, self.MS, self.Prod_IR,
                        self.Prod_Type, self.Prod, self.Proof_Add, self.Tel_Num_Type, self.Tel_Num, self.Trans],config={"llm": llm, "custom_whitelisted_dependencies": ["os"], "verbose": True, "conversational": True},memory_size=20)
        # self.agent=Agent(self.dl,config={"llm": llm, "custom_whitelisted_dependencies": ["os"], "verbose": True, "conversational": True},memory_size=20)
        return self.dl
    
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
        response = {"result": result, "last_code_generated": self.dl.last_code_executed}

        # Store the result in the cache
        self.cache[query_hashable] = response
        # logger.debug("response",self.cache[query_hashable])
        print("response",self.cache[query_hashable])

        return response

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
    print("2", type(result['result']), result['result'])
    # logger.debug("2", type(result['result']), result['result'])

    if ".png" not in result['result']:
        print("no image here")
        # logger.debug("no image here")
        response = result
        print(response)
        # logger.debug(response)
        # logger.debug("result= = =",result)
        print("result= = =",result)
        return response
        
    elif ".png" in result['result']:
        print("Image path:", result)
        print("I am here")
        # logger.debug("Image path:", result)
        # logger.debug("I am here")
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
    # logger.debug("Output",result)
    return resultvalue(result)