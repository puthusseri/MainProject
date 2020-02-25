from flask import Flask, render_template, request      
import sys
import maincode as maincode
app = Flask(__name__)

@app.route("/generateMCQ")
def generatemcq():
	return render_template("home.html")
	

    
@app.route("/abcd")
def generatemcqssaassa():
        a= maincode.returnQuestionSet('My friends are eating in the palace.')
        return render_template("home.html",abc=a)    


@app.route("/a")
def setee():
        para = '''My friends are eating in the palace. Your friends waited for you for over an hour. It is not worth paying so much money for this concert. When I reached the station, the train had left. I visited the Taj Mahal last month. The criminal attacked the victim with a blunt object. His company is greatly sought after. The terrified people fled to the mountains.'''


        a= maincode.returnAllQuestions(para)
        return render_template("home.html",abc=a,length = len(a)) 
        
        
        
        
        
@app.route("/")
def home():
	return render_template("a.html")

if __name__ == "__main__":
	app.run(debug=True)
