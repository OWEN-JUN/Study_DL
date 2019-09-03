from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('generic.html')

@app.route('/images/<name>')
def user(name):
    return render_template('%s.jpg'%name, name= name)
@app.route('/assets/css/<name>')
def user1(name):
    return render_template('%s'%name, name= name)
@app.route('/assets/fonts/<name>')
def user2(name):
    return render_template('%s'%name, name= name)
@app.route('/assets/js/<name>')
def user3(name):
    return render_template('%s'%name, name= name)

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=8888, debug=False)
