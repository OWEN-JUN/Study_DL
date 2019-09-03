# 모듈을 불러옵니다.
import sqlite3

# test.db 연결합니다.(SQLite는 없으면 자동으로 생성)
conn = sqlite3.connect("test.db")
cursor = conn.cursor()

# 테이블이 없다면 해당 테이블을 생성
cursor.execute("""DROP TABLE any """)
cursor.execute("""CREATE TABLE IF NOT EXISTS supermarket(ITemno INTEGER, Category TEXT, FoodName TEXT, Company TEXT, P INTEGER )""")

# 테이블의 내용을 모두 지웁니다.
sql = "DELETE FROM supermarket"
cursor.execute(sql)

# 데이터를 2건 입력 합니다.
sql = "INSERT into supermarket(ITemno, Category, FoodName, Company,P) values (?, ?, ?, ?,?)"
cursor.execute(sql, (1, '과일', '자몽', '마트',100))

sql = "INSERT into supermarket(ITemno, Category, FoodName, Company,P) values (?,?,?,?,?)"
cursor.execute(sql, (2, '음료수', '망고주스', '편의점',100))

# 입력된 데이터를 조회
sql = "select * from supermarket"
cursor.execute(sql)

# 데이터를 모두 가져옵니다.
rows = cursor.fetchall()

# 가져온 내용을 한 줄씩 가져와서, 각 컬럼의 내용을 공백으로 구분해 출력
for row in rows:
    print(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + " " + str(row[3]) + " " +str(row[4]))
        
# 연결을 닫습니다.
conn.close()