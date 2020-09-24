import nltk
import numpy as np
import random
import warnings
import io
import string 
warnings.filterwarnings('ignore')

f=open('D:\gb.txt','r',errors='ignore')
raw=f.read() #to read a file
raw=raw.lower() # convert to lower case
nltk.download('punkt') #download punkt for forming sentences and words out of strings
nltk.download('wordnet') # english dictionary
sent_tokens=nltk.sent_tokenize(raw) # list of sentences
word_tokens=nltk.word_tokenize(raw) #list of words

#preprocessing
lemmer=nltk.stem.WordNetLemmatizer()
#nltk will stem through the wordnetlemmatizer for the base word of the word present int the text
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
#returns the  normalized value of the tokens
remove_punct_dict= dict((ord(punct),None) for punct in string.punctuation)
#this removes the punctuation marks present in the given text
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
#removes the punctuation mark from the usertext after converting to the lowercase


#greetings
grt_inputs=("hey there", "hi", "hey you","hey","hola", "hello")
grt_responses=["hi","hello","hi there","hey","I'm glad you are talking to me","How can i help you?"]
gdbye_ips=("bye", "goodbye")
gdbye_resps=["see you!", "have a nice day", "bye! Come back again soon."]
thank_ips=("thanks", "thank you", "helpful", "thanks for helping me")
thank_resps=["Happy to help!", "Any time!", "My pleasure"]
noans_ips=("sad","loser","heartbroken")
noans_resps=["don't be sad","be strong","enjoy every moment","don't worry! you will be fine","Stay positive"]
bo_ips=("boring","bored","alone","lonely")
bo_resps=["read books","have a walk","I am there with you"]
help_ips=("help me?")
help_resps=["I can give you info abou global warming"]

def greeting(sentence):

    for word in sentence.split():
        if word.lower() in grt_inputs:
            return random.choice(grt_responses)
        elif word.lower() in gdbye_ips:
            return random.choice(gdbye_resps)
        elif word.lower() in thank_ips:
            return random.choice(thank_resps)
        elif word.lower() in bo_ips:
             return random.choice(bo_resps)
        elif word.lower() in help_ips:
            return random.choice(help_resps)
        elif word.lower() in noans_ips:
            return random.choice(noans_resps)    
        else:
            print()
           
 #vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# for converting characters to vectors
#matching the vectors with the given text

def response(user_response):
    #defing a response function
    chatbot_response=''
    #initially the reponse will be empty
    sent_tokens.append(user_response)
    #later appending the reponses to sent_tokens
    TfidfVec=TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')
    #for the vectorization the englich words are given
    tfidf=TfidfVec.fit_transform(sent_tokens)
    #transforming the sent_tokens as a vector
    vals=cosine_similarity(tfidf[-1],tfidf)
    #to find the similarity betn the vextors 
    idx=vals.argsort()[0][-2]
    #when it finds the similarity particular id is stored in idx
    flat=vals.flatten()
    #it will convert it into a row /a column matrix
    flat.sort()
    req_tfidf= flat[-2]
    if(req_tfidf==0):
        #if there is no match betn the user ip and the text file text
        chatbot_response=chatbot_response+"I am sorry! I don't understand you"
        return chatbot_response
    else:
        chatbot_response=chatbot_response+sent_tokens[idx]
        #it will return the reponse that is matching the id generated when similarity is found betn the text
        return chatbot_response
    
flag=True
print("chatbot: Hey I am here you answer your queries about global warming..If you want to exit say Bye..")
while(flag==True):
    user_response=input()
        #takes the user ip
    user_response=user_response.lower()
        #converts to lower case
    if(user_response!='see you later'):
            #if its bye and if its thanks or thank you flag is set as false and it says welcome
        if(user_response=="that's grateful"  or user_response=='really helped'):
                flag=False
                print("chatbot:hey No problem!!")
        else:
                #else it gives it for the greeting function to generate a random response
            if(greeting(user_response)!=None):
                print("chatbot:"+greeting(user_response))
            else:
                    #if it already has a response prints that and removes the previous response
                print("chatbot:")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
            #else prints bye
            flag=False
            print("chatbot:Have a good day!!")
                