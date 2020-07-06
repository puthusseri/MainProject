# Automatic Generation of MCQs for Reading Comprehension
### Introduction
This project was done under the guidance of Prof. Baby Syla for the completion of Masters degree in Computer Applications. This study aims at creating the MCQ's for a given project. As a starting step Im trying to recreate the state of art result. The repository contains the code that I tried for learning mode about this area. The whole project has three main task.
 - Generating Question
 - Answering those question
 - Generating distractors

I focused on the third part, which is to generate the distractors. In in run of studying deeplearning I found it difficult to understand the intution behind the LSTM cells. This project was the end resutl of that study.

### Project Setup
Clone this repository If you are using the anaconda, you can directly clone the environment as :
```sh
$ conda env create -f environment.yml
$ conda activate tensornlp
```
- Install Spacy
`pip install -U spacy`
- For lemetisation
`pip install -U spacy-lookups-data`
- Download English Data
`python -m spacy download en_core_web_sm`


### Progress
- [x] Generate the fill in the blanks <br>
- [x] Generate the distractors for the fill in the blanks <br>
- [x] Generate the wh questions <br>
- [x] Generate the answers for the wh questions <br>
- [x] Generate the distractors for the sequence
### Setting the virtual environment

The distractor generation part requires the tensorflow = 1.5.0
Upgrade the pip to version> 19


License
----

MIT


**Free Software, Hell Yeah!**
### A small tip
If you feel difficult in understanding LSTM , dont listen to anyone. Go direclty into its equation. Raise an issue if any found.


