from flask import Flask, render_template, request      
import sys

app = Flask(__name__)

@app.route("/generateMCQ")
def generatemcq():
	return render_template("home.html")
@app.route("/")
def home():
	return render_template("a.html")

if __name__ == "__main__":
	app.run(debug=True)
