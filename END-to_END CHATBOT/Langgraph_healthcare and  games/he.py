import pyodbc

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost,1433;"
    "DATABASE=master;"
    "UID=sa;"
    "PWD=YourStrong@Passw0rd",
    autocommit=True  # <--- Important!
)

cursor = conn.cursor()
cursor.execute("CREATE DATABASE RX_AI_ML")
print("✅ Database created successfully")
