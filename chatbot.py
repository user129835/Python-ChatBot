
# Current ChatBot is unfinished, it needs more responses and smarter response algorithm
#
#
#
# Still in progress...

# All necessary imported libs
import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages

# uncomment the following only the first time
nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only



#Reading the corpus module
with open('chatbot.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()


#TOkenisation (not necessary) 
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words


# Preprocessing the code
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching and User Inputs
GREETING_INPUTS = ["Hello", "Hi", "Greetings", "Hey", "Wsup", "Sup", "Yo", "Hey", "Howdy", "How's it going?", "Whazzup"]
GREETING_RESPONSES = ["How are you feeling today?", "What has upsetting you?", "Hi there, how may I help you?", "I will like to hear what is upsetting you today?"]
INCOMP_ANSWERS = ["Sory, can you explain me more briefly", "Sory, I misunderstood, can you explain more", "Sory, i dont understand, can you explain it"]
POSITIVE_USER_RESPONSE = ("acceptable", "excellent", "great", "marvelous", "positive", "cool", "normal", "satisfactory", "fine", "superb", "beautiful")
NEGATIVE_USER_RESPONSE = ("bad", "inferior", "second-class", "inadequate", "unacceptable", "imperfect", "unpleasant", "disagreeable", "unwelcome", "unfortunate", "unfavourable", "unlucky", "sad", "tearful", "adverse", "nasty", "terrible", "dreadful", "awful", "grim", "distressing", "regrettable") 


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


flag=True
print("Hello. My name is Sophia and I will be your assistant today! (If you want to exit, type Bye)")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='Bye'):
        if(user_response=='thanks' or user_response=='thank you' or user_response=='thank you so much' ):
            flag=False
            print("You are welcome :)")
        else:
            if(greeting(user_response)!=None):
                print(greeting(user_response))
            else:
                print(end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("Bye! Have a fantastic day :)")    
        
        






