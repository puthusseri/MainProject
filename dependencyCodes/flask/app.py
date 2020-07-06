from flask import Flask, render_template, request      
import sys
import maincode as maincode
app = Flask(__name__)

@app.route("/a")
def generatemcq():
	return render_template("home.html")
	

    
@app.route("/abcd", methods=["POST"])
def generatemcqssaassa():
        if request.method == "POST":
                paragraph = request.form['para']
        a= maincode.returnQuestionSet('My friends are eating in the palace.')
        return render_template("home.html",abc=a)    


@app.route("/generateMCQ", methods=["POST"])
def setee():
        if request.method == "POST":
                paragraph = request.form['para']
        a= maincode.returnAllQuestions(paragraph)
        return render_template("home.html",abc=a,length = len(a)) 
        
        
        
        
        
@app.route("/")
def home():
	return render_template("a.html")

if __name__ == "__main__":
	app.run(debug=True)
