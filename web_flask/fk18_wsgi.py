from wsgiref.simple_server import make_server

def my_app (environ, start_response):

    status = "200 ok"
    headers = [("Content-Type", 'text/plain')]
    start_response(status, headers)

    response = [b"this is a sample WSGI Application."]

    return response

if __name__ == "__main__":
    print("Started WSGI Server on port 8888...")
    server = make_server('', 8888, my_app)
    server.serve_forever()

'''

WSGI 규격 : WEB server gateway interface
파이썬 애플리케이션을 실행하고자 하는 웹 서버는 이 규격을 준수해야한다.
웹서버와 웹 애플리케이션을 연결해주는 규격
장소와 같은 파이썬 웹 프레임워크를 개발하거나,
이런 웹 프레임워크를 아파치와 같은 웹서버와 연동시 사용
'''