import pyodbc as pyo


server = "localhost"
database = "bitdb"
username = "bit"
password = "1234"

cnxn = pyo.connect("DRIVER={ODBC Driver 13 for SQL Server}; SERVER="+server+
                    '; PORT=1433; DATABASE='+database+ ";UID="+username+";PWD="+password)

cursor = cnxn.cursor()

tsql = "SELECT * FROM iris2;"


from flask import Flask, render_template
app = Flask(__name__)

@app.route("/sqltable")
def showsql():
    cursor.execute(tsql)
    return render_template("myweb.html", rows=cursor.fetchall())
    

if __name__ == "__main__":
    app.run(host="127.0.0.1", port = 5000, debug=False)



# with cursor.execute(tsql):
#     row = cursor.fetchone()
#     #행이 존재할때까지 하나씩 행을 증가시키면서 모든 컬럼을 공백으로 구분
#     while row:
#         print(str(row[0])+" "+str(row[1])+" "+str(row[2])+" "+str(row[3])+" "+str(row[4]))
#         row = cursor.fetchone()


# cnxn.close()