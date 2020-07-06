from flask import Flask, render_template, request      
import sys
import demo as d
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

        return render_template("home.html",abc=a,length = len(a)) 

@app.route("/")
def home():
        return render_template("a.html")

if __name__ == "__main__":
        app.run(debug=True)
