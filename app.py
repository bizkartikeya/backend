import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, MetaData
from pandasai import SmartDatalake
from pandasai.llm import AzureOpenAI


app = FastAPI()

class DataProcessorAnalyzer:
    def __init__(self, azure_openai_endpoint, api_version, deployment_name, api_key,
                 db_file, tables=None):
        self.llm = self.initialize_llm(azure_openai_endpoint, api_version, deployment_name, api_key)
        self.engine = self.connect_to_database(db_file)
        self.table_names = self.get_table_names()
        self.tables_to_retrieve = tables if tables else self.table_names
        self.data_frames = self.read_data_frames()
        self.sdl = self.smart_datalake(self.data_frames, {
            "llm": self.llm,
            "enable_cache": False,
            "max_retries": 10,
            "use_error_correction_framework": True,
            "verbose": True,
            "enforce_privacy": False,
        })

    def initialize_llm(self, endpoint, api_version, deployment_name, api_token, is_chat_model=True):
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            deployment_name=deployment_name,
            api_token=api_token,
            is_chat_model=is_chat_model
        )

    def connect_to_database(self, db_file):
        connection_string = f"sqlite:///{db_file}"
        return create_engine(connection_string)

    def get_table_names(self):
        with self.engine.connect() as connection:
            metadata = MetaData()
            metadata.reflect(bind=connection)
            return metadata.tables.keys()

    def read_data_frames(self):
        data_frames = {}
        for table_name in self.tables_to_retrieve:
            if table_name not in self.table_names:
                raise ValueError(f"Table '{table_name}' not found in the database metadata.")
            with self.engine.connect() as connection:
                data_frames[table_name] = pd.read_sql_table(table_name, connection)
        return data_frames

    def smart_datalake(self, data_frames, config):
        return SmartDatalake(list(data_frames.values()), config)

    def query_data(self, query):
        return self.sdl.chat(query)

# Example usage
load_dotenv()

analyzer_processor = DataProcessorAnalyzer(
    azure_openai_endpoint=os.getenv("GPT3_ENDPOINT"),
    api_version=os.getenv("GPT3_API_VERSION"),
    deployment_name=os.getenv("GPT3_DEPLOYMENT_NAME"),
    api_key=os.getenv("GPT3_KEY"),
    db_file=r"D:\Biz_master_pandasai\09-01\backend\backend\Bank_details.db",  # Replace with the path to your SQLite database file
    # tables=["Bank_details"],  # Replace with the list of tables you want to analyze OR keep None if you want to use all tables
)

result = analyzer_processor.query_data("How many rows in the users table?")
print(result)