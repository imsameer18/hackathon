from flask import Flask,request,jsonify
from new import df1
app = Flask(__name__)
@app.route('/')
def home():
    return df1

if __name__== '__main__':
    app.run(debug=True)
