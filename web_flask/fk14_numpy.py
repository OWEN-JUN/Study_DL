import pymssql as ms
import numpy as np

# 데이터 베이스에 연결
conn = ms.connect(server="localhost",user="bit", password='1234',database='bitdb')
# 커서를 만듬
cursor = conn.cursor()
#커서에 쿼리를 입력해 실행
cursor.execute("SELECT TOP (1000)* FROM Iris2;")
#한행을 가져옵니다.

rows = cursor.fetchall()
for row in rows:
    print(row)

a= np.asarray(rows)
print(a)
print(a.shape)
print(type(a))

# np.save("test_a.npy",a)
conn.close()







'''
row = cursor.fetchone()
print(type(row))
#행이 존재할 때까지, 하나씩 행을 증가시키면서 1번째 컬럼을 숫자 2째번 컬럼을 문자로 출력
while row:
    # print("첫컬럼=%s, 둘컬럼=%s"%(frow[0],row[1]))
    print(row)
    row = cursor.fetchone()
#연결을 닫습니다.
conn.close()
'''