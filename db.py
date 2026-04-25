import mysql.connector
import os

def get_connection():
    return mysql.connector.connect(
        host=os.environ.get("mysql.railway.internal"),
        user=os.environ.get("root"),
        password=os.environ.get("nENQPEBbPltAwXdlTjGoDYGSuZCsdqpQ"),
        database=os.environ.get("railway"),
        port=int(os.environ.get("3306", 3306))
    )