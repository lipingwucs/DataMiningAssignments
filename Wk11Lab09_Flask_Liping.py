# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:33:04 2020

@author: Liping
"""

from flask import Flask
app = Flask(__name__)
@app.route("/myex")
def hello():
    return "Welcome to machine learning model APIs!, Liping"
if __name__ == '__main__':
 #   
    app.run(debug=True,port=12345)
