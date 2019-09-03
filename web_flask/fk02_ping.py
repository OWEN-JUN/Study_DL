from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "<h1>hello world<h2>hello world</h2></h1>"

@app.route("/ping",methods=["GET"])
def ping():
    return "<h1>pong</h1>"

if __name__ == "__main__":
    app.run(host="192.168.0.158",port=8888, debug=False)