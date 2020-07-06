# The required modules are
import pandas as pd
import _pickle as cPickle
from pathlib import Path
from nltk.tokenize import sent_tokenize


import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = 'glove.6B.100d.txt'
tmp_file = 'word2vec-glove.6B.100d.txt'

glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)



def retunVerb(inputString):
        doc = nlp(inputString)
        temp = []
        lis = []
        se = []
        indexCount = 0
        for token in doc:
                if(token.pos_ == 'VERB' and not(token.is_stop)):
                        se = [token.text,indexCount]
                        lis.append(se)
                indexCount = indexCount + 1
        return(lis)
        
       
#This can include the fill in the blanks
def generateFillintheblanks(inputString,key):
        stri = inputString.split() # Replace the key with index
        stri[key[1]] = '..........'
        output = ''
        #combine the string 
        for i in range(len(stri)):
                output = output + ' ' + stri[i]    
        return(output)
        
        
 # Generate the distractor
def generateDistractor(answer,count):
        answer = str.lower(answer)

        ##Extracting closest words for the answer. 
        try:
                closestWords = model.most_similar(positive=[answer], topn=count)
        except:
                #In case the word is not in the vocabulary, or other problem not loading embeddings
                return []

        #Return count many distractors
        distractors = list(map(lambda x: x[0], closestWords))[0:count]
        return distractors

questionCount = 1
def returnQuestionSet(sentence):
        global questionCount
        verbs = retunVerb(sentence)
        que = generateFillintheblanks(sentence,verbs[0])
        distractor = generateDistractor(verbs[0][0],3)
        print("\n",questionCount,que)
        
        print("a.",distractor[0],"\nb.",distractor[1])
        print("c.",distractor[2],"\nd.",verbs[0][0])

        li = []
        li.append(questionCount)
        li.append(que)
        li.append(distractor[0])
        li.append(distractor[1])
        li.append(distractor[2])
        li.append(verbs[0][0])
        questionCount = questionCount + 1
        return li
questionCount = 1
def returnQuestionSetasList(sentence):
        global questionCount
        verbs = retunVerb(sentence)
        que = generateFillintheblanks(sentence,verbs[0])
        distractor = generateDistractor(verbs[0][0],3)
        print("\n",questionCount,que)
        
        print("a.",distractor[0],"\nb.",distractor[1])
        print("c.",distractor[2],"\nd.",verbs[0][0])

        li = []
        li.append(questionCount)
        li.append(que)
        li.append(distractor[0])
        li.append(distractor[1])
        li.append(distractor[2])
        li.append(verbs[0][0])
        questionCount = questionCount + 1
        return li

# Now given a paragraph , return all the quesions
def returnAllQuestions(paragraph):
        li = []
        sentences = sent_tokenize(paragraph)
        for sentence in sentences:
                li.append(returnQuestionSetasList(sentence))
        # Display the thing in the html
        return(li)
       
       
def generateOtherTypeQue():
        fp = open('input.txt')
        data = fp.read()
        sentences = sent_tokenize(data)
        new_paragraph = []
        for sentence in sentences:
              verb = retunVerb(sentence)
              if not verb:
                    pass
              else:
                    new_paragraph.append(sentence)
            
        new_paragraph = " ".join(new_paragraph)
        li = returnAllQuestions(new_paragraph)
        return(li)
               


#para = '''My friends are eating in the palace. Your friends waited for you for over an hour. It is not worth paying so much money for this concert. When I reached the station, the train had left. I visited the Taj Mahal last month. The criminal attacked the victim with a blunt object. His company is greatly sought after. The terrified people fled to the mountains.'''



