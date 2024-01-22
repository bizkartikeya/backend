from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import AzureOpenAI
import os
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResult(BaseModel):
    result: str
    last_code_generated: str

class BankDataAnalysis:
    def __init__(self):
        os.environ['OPENAI_API_KEY'] = "d06dfb8d9f2e4d9696bfb614c8da2a69"
        os.environ["AZURE_OPENAI_API_ENDPOINT"] = "https://cog-ccgt327nxe3tc.openai.azure.com/"
        os.environ["OPENAI_API_BASE"] = "https://cog-ccgt327nxe3tc.openai.azure.com/"
        os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
        os.environ['OPENAI_API_TYPE'] = "azure"
        # self.conn = sqlite3.connect('Bank_details.db')
        # self.cursor = self.conn.cursor()
        self.df = pd.DataFrame()
        print("DataFrame value ---:", len(self.df)) #here DF.size is 0

    def load_data(self):
        self.df=pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Bank_Data.csv")
 
        # try:
        #     self.df = pd.read_sql_query("SELECT * FROM Bank_Data", self.conn)
        # except Exception as e:
        #     print(f"Error loading data: {e}")
        print("DataFrame value -1-:", len(self.df)) #here DF.size is 0

    def initialize_llm(self):
        try:
            llm = AzureOpenAI(deployment_name="chat",
                          model_name="gpt-35-turbo",
                          azure_endpoint=os.environ['AZURE_OPENAI_API_ENDPOINT'],
                          api_base=os.environ['OPENAI_API_BASE'],
                          temperature=0,
                          model_kwargs={"api_type": "azure",
                                        "api_version": "2023-07-01-preview"})
            self.df = SmartDataframe(self.df, config={"llm": llm})
        except Exception as e:
            print(f"Error initializing language model: {e}")


    def query_data(self, query):
        try:
            print("DataFrame size:", len(self.df))
            result = self.df.chat(query)
            return {"result": result, "last_code_generated": self.df.last_code_generated}
        except Exception as e:
            print(f"Error querying data: {e}")
            return {"error": str(e)}

bank_analysis = BankDataAnalysis()
bank_analysis.load_data()
bank_analysis.initialize_llm()

@app.post("/query/")
async def query_endpoint(query_req: QueryRequest):
    result = bank_analysis.query_data(query_req.query)
    print("Output",result)
    return resultvalue(result)
    
        
def resultvalue(result):
    result['result'] = str(result['result'])
    if ".png" not in result['result']:
        print("no image here")
        response = result
        print(response)
        print("result= = =",result)
        return response
        
    elif ".png" in result['result']:
        print("I am here")
        print("Image path:", result)
        base64_image = image_to_base64(result['result'])
        response = {
            "image_path": result,
            "base64_image": base64_image
        }
        return response

    

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        print("Base64 image:", base64_image)
        return base64_image