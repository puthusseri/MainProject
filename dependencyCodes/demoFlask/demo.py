import nltk
import nltk.data
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import *
from keras.models import Sequential,Model
import collections
from keras.models import load_model

from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz")


nlp = spacy.load('en_core_web_sm')
stemmer = LancasterStemmer()
sentences = []
disc_sentences = {}
nondisc_sentences = []
questions_list = []
questions_answers_list = []
questions_answers_distractors_list = []
yes_or_no_questions = []
aux_list = ['am', 'are', 'is', 'was', 'were', 'can', 'could', 'does', 'do', 'did', 'has', 'had', 'may', 'might', 'must','need', 'ought', 'shall', 'should', 'will', 'would']
discourse_markers = ['because', 'as a result', 'since', 'when', 'although', 'for example', 'for instance']
qtype = {'because': ['Why'], 'since': ['When', 'Why'], 'when': ['When'], 'although': ['Yes/No'], 'as a result': ['Why'],'for example': ['Give an example where'], 'for instance': ['Give an instance where'], 'to': ['Why']}
target_arg = {'because': 1, 'since': 1, 'when': 1, 'although': 1, 'as a result': 2, 'for example': 1, 'for instance': 1, 'to': 1}


def sentensify():
    global sentences
    global questions_list
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    fp = open('input.txt')
    data = fp.read()
    sentences = tokenizer.tokenize(data)
    question_answer = discourse()
    for i in range(len(question_answer)):
        questions_list.append(question_answer[i][1])

def generate_question(question_part, type):
    question_part = question_part[0].lower() + question_part[1:]
    if(question_part[-1] == '.' or question_part[-1] == ','):
        question_part = question_part[:-1]
    for i in range(0, len(question_part)):
        if(question_part[i] == 'i'):
            if((i == 0 and question_part[i+1] == ' ') or (question_part[i-1] == ' ' and question_part[i+1] == ' ')):
                question_part = question_part[:i] + 'I' + question_part[i + 1: ]      
    question = ""
    if(type == 'Give an example where' or type == 'Give an instance where'):
        question = type + " " + question_part + '?'
        return question
    aux_verb = False
    res = None
    for i in range(len(aux_list)):
        if(aux_list[i] in question_part.split()):
            aux_verb = True
            pos = i
            break
    if(aux_verb):
        text = nltk.word_tokenize(question_part)
        tags = nltk.pos_tag(text)
        question_part = ""
        fP = False
        for word, tag in tags:
            if(word in ['I', 'We', 'we']):
                question_part += 'you' + " "
                fP = True
                continue
            question_part += word + " "
        question = question_part.split(" " + aux_list[pos])
        if(fP):
             question = ["were "] + question
        else:
            question = [aux_list[pos] + " "] + question
        if(type == 'Yes/No'):
            question += ['?']
        elif(type != "non_disc"):
            question = [type + " "] + question + ["?"]
        else:
            question = question + ["?"]
        question = ''.join(question)
    else:
        aux = None
        text = nltk.word_tokenize(question_part)
        tags = nltk.pos_tag(text)
        comb = ""
        for tag in tags:
            if(comb == ""):
                if(tag[1] == 'NN' or tag[1] == 'NNP'):
                    comb = 'NN'
                elif(tag[1] == 'NNS' or tag[1] == 'NNPS'):
                    comb = 'NNS'
                elif(tag[1] == 'PRP'):
                    if tag[0] in ['He','She','It']:
                        comb = 'PRPS'
                    else:
                        comb = 'PRPP'
                        tmp = question_part.split(" ")
                        tmp = tmp[1: ]
                        if(tag[0] in ['I', 'we', 'We']):
                            question_part = 'you ' + ' '.join(tmp)                            
            if(res == None):
                res = re.match(r"VB*", tag[1])
                if(res):
                    question_part = question_part.replace(tag[0], stemmer.stem(tag[0]))
                res = re.match(r"VBN", tag[1])
                res = re.match(r"VBD", tag[1])
        if(comb == 'NN'):
            aux = 'does'            
        elif(comb == 'NNS'):
            aux = 'do'         
        elif(comb == 'PRPS'):
            aux = 'does'           
        elif(comb == 'PRPP'):
            aux = 'do'        
        if(res and res.group() in ['VBD', 'VBN']):
            aux = 'did'
        if(aux):
            if(type == "non_disc" or type == "Yes/No"):
                question = aux + " " + question_part + "?"
            else:
                question = type + " " + aux + " " + question_part + "?"
    if(question != ""):
        question = question[0].upper() + question[1:]
    return question
        
def get_named_entities(sent):
    doc = nlp(sent)
    named_entities = [(X.text, X.label_) for X in doc.ents]
    return named_entities        

def get_wh_word(entity, sent):
    wh_word = ""
    if entity[1] in ['TIME', 'DATE']:
        wh_word = 'When'        
    elif entity[1] == ['PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']:
        wh_word = 'What'        
    elif entity[1] in ['PERSON']:
            wh_word = 'Who'            
    elif entity[1] in ['NORP', 'FAC' ,'ORG', 'GPE', 'LOC']:
        index = sent.find(entity[0])
        if index == 0:
            wh_word = "Who"            
        else:
            wh_word = "Where"            
    else:
        wh_word = "Where"
    return wh_word

def generate_one_word_questions(sent):    
    named_entities = get_named_entities(sent)
    questions = []    
    if not named_entities:
        return questions    
    for entity in named_entities:
        wh_word = get_wh_word(entity, sent)        
        if(sent[-1] == '.'):
            sent = sent[:-1]        
        if sent.find(entity[0]) == 0:
            questions.append(sent.replace(entity[0],wh_word) + '?')
            continue       
        question = ""
        aux_verb = False
        res = None
        for i in range(len(aux_list)):
            if(aux_list[i] in sent.split()):
                aux_verb = True
                pos = i
                break            
        if not aux_verb:
            pos = 9        
        text = nltk.word_tokenize(sent)
        tags = nltk.pos_tag(text)
        question_part = ""        
        if wh_word == 'When':
            word_list = sent.split(entity[0])[0].split()
            if word_list[-1] in ['in', 'at', 'on']:
                question_part = " ".join(word_list[:-1])
            else:
                question_part = " ".join(word_list)            
            qp_text = nltk.word_tokenize(question_part)
            qp_tags = nltk.pos_tag(qp_text)            
            question_part = ""            
            for i, grp in enumerate(qp_tags):
                word = grp[0]
                tag = grp[1]
                if(re.match("VB*", tag) and word not in aux_list):
                    question_part += WordNetLemmatizer().lemmatize(word,'v') + " "
                else:
                    question_part += word + " "                
            if question_part[-1] == ' ':
                question_part = question_part[:-1]        
        else:
            for i, grp in enumerate(tags):                
                word = grp[0]
                tag = grp[1]
                if(re.match("VB*", tag) and word not in aux_list):
                    question_part += word
                    if i<len(tags) and 'NN' not in tags[i+1][1] and wh_word != 'When':
                        question_part += " "+ tags[i+1][0]
                    break
                question_part += word + " "
        question = question_part.split(" "+ aux_list[pos])
        question = [aux_list[pos] + " "] + question
        question = [wh_word+ " "] + question + ["?"]
        question = ''.join(question)
        questions.append(question)    
    return questions      


def discourse():
    temp = []
    target = ""
    questions = []
    global disc_sentences
    global yes_or_no_questions
    yes_or_no_questions = []
    disc_sentences = {}
    for i in range(len(sentences)):
        maxLen = 9999999
        val = -1
        for j in discourse_markers:
            tmp = len(sentences[i].split(j)[0].split(' '))  
            if(len(sentences[i].split(j)) > 1 and tmp >= 3 and tmp < maxLen):
                maxLen = tmp
                val = j                
        if(val != -1):
            if(disc_sentences.get(val, 'empty') == 'empty'):
                disc_sentences[val] = []                
            disc_sentences[val].append(sentences[i])
            temp.append(sentences[i])
    nondisc_sentences = list(set(sentences) - set(temp))    
    t = []
    for k, v in disc_sentences.items():
        for val in range(len(v)):
            question_part = disc_sentences[k][val].split(k)[target_arg[k] - 1]
            q = generate_question(question_part, qtype[k][0])
            if(q != ""):
                if qtype[k][0] == "Yes/No":
                    yes_or_no_questions.append(q)
                else:
                    questions.append([disc_sentences[k][val],q])         
    for question_part in nondisc_sentences:
        s = "non_disc"
        sentence = question_part
        text = nltk.word_tokenize(question_part)
        if(text[0] == 'Yes'):
            question_part = question_part[5:]
            s = "Yes/No"
            
        elif(text[0] == 'No'):
            question_part = question_part[4:]
            s = "Yes/No"            
        q = generate_question(question_part, s)
        if(q != ""):
            if s=="Yes/No":
                yes_or_no_questions_or_no_questions.append(q)
            else:
                questions.append([sentence,q])                
        l = generate_one_word_questions(question_part)
        questions += [[sentence,i] for i in l]
    print(len(questions))
    return questions     

#Question Answering
def ansewringQuestions():
    global questions_answers_list
    questions_answers_list = []
    f = open("input.txt")
    data = f.read()
    for i in questions_list:
        result = predictor.predict(passage=data,question=i)
        questions_answers_list.append([i,result['best_span_str']])    


#Distractor generation
def clean_text(sentence):
    sentence = sentence.lower()
    sentence = re.sub("[^a-z0-9]+"," " , sentence)
    sentence = sentence.split()
    sentence = [s for s in sentence if((len(s)>1) or (re.match("[0-9]+",s) is not None))]
    sentence = " ".join(sentence)
    return sentence

def data_generator(train_distractors,answers,word_to_idx,max_len,batch_size):
    X1,X2,X3,y = [],[],[],[]
    n=0
    while True:
        for key,dist_list in train_distractors.items():
            n+=1
            question = key
            answer = answers[key]
            seqq = [word_to_idx[wordQ] for wordQ in question.split() if wordQ in word_to_idx]
            question= pad_sequences([seqq],maxlen=max_q,value=0,padding='post')[0]
            seqa = [word_to_idx[wordA] for wordA in answer.split() if wordA in word_to_idx]
            answer = pad_sequences([seqa],maxlen=max_a,value=0,padding='post')[0]
            for dist in dist_list:
                seq = [word_to_idx[word] for word in dist.split() if word in word_to_idx]
                for i in range(1,len(seq)):
                    xi = seq[0:i]
                    yi = seq[i]
                    xi = pad_sequences([xi],maxlen=max_len,value = 0,padding='post')[0] 
                    yi = to_categorical([yi],num_classes = vocab_size)[0]
                    X1.append(question)
                    X2.append(answer)
                    X3.append(xi)
                    y.append(yi)
                if n==batch_size:
                    yield[[np.array(X1),np.array(X2),np.array(X3)],np.array(y)]
                    X1,X2,X3,y = [],[],[],[]
                    n=0    

def getEmbeddingMatrix():
    emb_dim=100
    matrix = np.zeros((vocab_size,emb_dim))
    for word,idx in word_to_idx.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            matrix[idx] = embedding_vector
    return matrix

def generateDistractors():
    train = pd.read_csv("Train.csv")
    test = pd.read_csv("Test.csv")
    test = test.values
    train = train.values
    answers = {}
    distractors = {}
    count = 0
    for x in range(train.shape[0]):
        answers[train[x][0]] = train[x][1]
        a=[]
        for y in train[x][2].split(", "):
            a.append(str(y[1:-1]))
        distractors[train[x][0]] = a
        count = count+1
    a={}
    d={}
    for key , dist_list in distractors.items():
        for i in range(len(dist_list)):
            dist_list[i] = clean_text(dist_list[i])
        answer=clean_text(answers[key])
        key=clean_text(key)
        a[key]=answer
        d[key]=dist_list
    answers=a
    distractors=d    
    with open("answers.txt","w") as f:
        f.write(str(answers))
    with open("distractors.txt","w") as f:
        f.write(str(distractors))
    vocab = set()
    for key in answers.keys():
        [vocab.update(key.split())]
        [vocab.update(answers[key].split())]
        [vocab.update(sentence.split()) for sentence in distractors[key]]
    total = []
    for key in answers.keys():
        [total.append(i) for i in key.split()]
        [total.append(i) for i in answers[key].split()]
        [total.append(i) for des in distractors[key] for i in des.split()]
    counter = collections.Counter(total)
    freq_cnt = dict(counter)
    sorted_freq_cnt = sorted(freq_cnt.items(),reverse = True,key = lambda x:x[1])
    threshold =10
    sorted_freq_cnt = [x for x in sorted_freq_cnt if x[1]>threshold]
    total_words = [x[0] for x in sorted_freq_cnt]
    train_distractors = {}
    for key in distractors.keys():
        train_distractors[key] = []
        for dist in distractors[key]:
            dist_to_append = "StartSeq " + dist + " EndSeq"
            train_distractors[key].append(dist_to_append)
    word_to_idx = {}
    idx_to_word = {}
    for i,word in enumerate(total_words):
        word_to_idx[word] = i+1
        idx_to_word[i+1] = word
    word_to_idx["StartSeq"]=4724
    idx_to_word[4724] = "StartSeq"
    word_to_idx["EndSeq"]=4725
    idx_to_word[4725] = "EndSeq"
    vocab_size = len(word_to_idx)
    vocab_size= vocab_size+1
    max_len=0
    for key in train_distractors.keys():
        for dist in train_distractors[key]:
            max_len = max(max_len,len(dist.split()))
    max_q=0
    for key in train_distractors.keys():
        max_q = max(max_q,len(key.split()))
    max_a = 0
    for key in answers.keys():
        max_a = max(max_a,len(answers[key].split()))
    f=open("glove.6B.100d.txt",encoding="utf8")
    embedding_index = {}
    for line in f:
        values=line.split()
        word = values[0]
        embedding_index[word]=np.array(values[1:],dtype='float')

    emb_dim=100
    matrix = np.zeros((vocab_size,emb_dim))
    for word,idx in word_to_idx.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            matrix[idx] = embedding_vector
    embedding_matrix = matrix

    #architecture
    input_dist = Input(shape = (max_len,))
    input_dist1=  Embedding(input_dim=vocab_size,output_dim=100,mask_zero=True)(input_dist)
    input_dist2 = Dropout(0.3)(input_dist1)
    input_dist3 = LSTM(256)(input_dist2)
    input_ques = Input(shape = (max_q,))
    input_ques1=  Embedding(input_dim=vocab_size,output_dim=100,mask_zero=True)(input_ques)
    input_ques2 = Dropout(0.3)(input_ques1)
    input_ques3 = LSTM(256)(input_ques2)
    input_ans = Input(shape = (max_a,))
    input_ans1=  Embedding(input_dim=vocab_size,output_dim=100,mask_zero=True)(input_ans)
    input_ans2 = Dropout(0.3)(input_ans1)
    input_ans3 = LSTM(256)(input_ans2)
    decoder1 = add([input_dist3,input_ques3,input_ans3])
    decoder2 = Dense(512 ,activation = 'relu')(decoder1)
    outputs = Dense(vocab_size,activation= 'softmax')(decoder2)
    model = Model(inputs = [input_ques,input_ans,input_dist],outputs = outputs)
    model.layers[3].set_weights([embedding_matrix])
    model.layers[3].trainable = False  
    model.layers[4].set_weights([embedding_matrix])
    model.layers[4].trainable = False  
    model.layers[5].set_weights([embedding_matrix])
    model.layers[5].trainable = False
    model.compile(loss='categorical_crossentropy',optimizer = 'adam')
    model.load_weights("./model_weights/model_58.h5")
    answers_t = {}
    count = 0
    for x in range(test.shape[0]):
        answers_t[test[x][0]] = test[x][1]
        count = count+1
        
    a={}
    for key , answer in answers_t.items():
        answer=clean_text(answers_t[key])
        key=clean_text(key)
        a[key]=answer
    answers_t=a

    def predict_distractors(X1,X2):
        dists = []
        for j in range(3):
            in_text = "StartSeq"
            for i in range(max_len):
                sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
                sequence = pad_sequences([sequence],maxlen=max_len,padding = 'post')[0]
                XQ = []
                XA = []
                XI = []
                XQ.append(X1)
                XA.append(X2)
                XI.append(sequence)
                y_pred = model.predict([np.array(XQ),np.array(XA),np.array(XI)])
                if(i<=1):
                    y_pred=np.array(y_pred)
                    y_pred = y_pred.argsort()
                    y_pred=y_pred[0][:]
                    y_pred=y_pred[len(y_pred)-1-j]
                else:
                    y_pred=y_pred.argmax()
                word = idx_to_word[y_pred]
                in_text += (' ' + word)
                if word == 'EndSeq':
                    break
            final_dists = in_text.split()[1:-1]
            final_dists = ' '.join(final_dists)
            dists.append(final_dists)
        return dists

    def generate_MCQ(que,ans):
        global questions_answers_distractors_list
        question= clean_text(que)
        answer = clean_text(ans)
        seqq = [word_to_idx[wordQ] for wordQ in question.split() if wordQ in word_to_idx]
        question= pad_sequences([seqq],maxlen=max_q,value=0,padding='post')[0]
        seqa = [word_to_idx[wordA] for wordA in answer.split() if wordA in word_to_idx]
        answer = pad_sequences([seqa],maxlen=max_a,value=0,padding='post')[0]
        distractor = predict_distractors(question,answer)
        distractor = str(distractor)
        distractor = distractor.replace("[","")
        distractor = distractor.replace("]","")
        distractor = distractor.replace("'","")
        distractor = distractor.split(",")
        questions_answers_distractors_list.append([que,ans,distractor[0],distractor[1],distractor[2]])

    for i in range(len(questions_answers_list)):
        que = questions_answers_list[i][0]
        ans = questions_answers_list[i][1]
        generate_MCQ(que,ans)
    

