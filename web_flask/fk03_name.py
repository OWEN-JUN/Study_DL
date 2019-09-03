from flask import Flask

app = Flask(__name__)


@app.route("/<name>")
def user(name):

    return "<h1>hello , %s !!!</h1>"%name
    
@app.route("/user/<name>")
def user2(name):
    return "<h1>hello , user/%s !!!</h1>"%name

@app.route("/user/add/<add>")
def add(add):
    a,b = map(int,add.split(","))
    c = a+b
    return "<h1>a+b = %d</h1>"%(c)
    


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)