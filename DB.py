from pathlib import Path
import pandas as pd
import sqlite3

conn=sqlite3.connect("Bank_details.db")
cursor=conn.cursor()
# cursor.execute("""CREATE TABLE Bank_data(age int, job text, marital text, education text, balance int ,housing text, loan text,contact text,day int, month text, duration int) """)

df=pd.read_csv("/Users/kartikeyshrivastav/Desktop/backend/Bank_Data.csv")
print("DataFrame size01100 ---:", len(df)) #here DF.size is 4932 
df.to_sql("Bank_details",conn, if_exists='replace',index= False)
conn.commit()
conn.close()



