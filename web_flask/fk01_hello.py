from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "<h1>hello world<h2>hello world</h2></h1>"

@app.route("/bit")
def bit():
    return "<h1>bit가 bit하다</h1>"

if __name__ == "__main__":
    app.run(host="192.168.0.158",port=8888, debug=False)