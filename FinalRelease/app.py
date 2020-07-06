from flask import Flask, render_template, request      
import sys
import demo as d
import maincode as maincode
app = Flask(__name__)


@app.route("/generateMCQ", methods=["POST"])
def setee():
        if request.method == "POST":
                paragraph = request.form['para']
                f = open("input.txt","w")
                f.write(paragraph)
                f.close()
        d.sentensify()
        d.ansewringQuestions()
        d.generateDistractors()
        a = d.questions_answers_distractors_list
        li = maincode.generateOtherTypeQue()
        return render_template("ans.html",abc=a,length = len(a), li = li, l = len(li) ) 

@app.route("/home")
def home():
        return render_template("vyshak.html")
       
@app.route("/")
def hometemp():
        return render_template("vyshak.html")

if __name__ == "__main__":
        app.run(debug=True)
