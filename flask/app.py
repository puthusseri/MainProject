from flask import Flask, render_template, request      
import sys
import maincode as maincode
app = Flask(__name__)

@app.route("/generateMCQ")
def generatemcq():
	return render_template("home.html")
	
	
@app.route("/a")
def generatemcqaa():
        a= maincode.returnQuestionSetJSON('My friends are eating in the palace.')
        return render_template("home.html",abc=a)
    
@app.route("/abcd")
def generatemcqssaassa():
        a= maincode.returnQuestionSet('My friends are eating in the palace.')
        return render_template("home.html",abc=a)    
       
@app.route("/")
def home():
	return render_template("a.html")

if __name__ == "__main__":
	app.run(debug=True)
