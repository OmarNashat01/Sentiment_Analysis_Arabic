# Commented out IPython magic to ensure Python compatibility.
from __future__ import unicode_literals
# Import the required libraires
import numpy as np
import pandas as pd

# visualization libraries
import matplotlib.pyplot as plt

# %matplotlib inline

import re
import unicodedata

# Pre-processing library
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import nltk
import gensim
import arabicstopwords.arabicstopwords as stp
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
import pyarabic.araby as araby
from nltk.stem.isri import ISRIStemmer

import unicodedata
# !pip install unidecode
from unidecode import unidecode
# !pip install emoji
import emoji

# Vectorizers
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import Word2Vec
from gensim.models import FastText

# Mectrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc


# Models 
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


if __name__ == "__main__":
    test_neg = pd.read_csv('./data/test_Arabic_tweets_negative.tsv',header=None, sep='\t')
    test_pos = pd.read_csv('./data/test_Arabic_tweets_positive.tsv',header=None, sep='\t')
    train_neg = pd.read_csv('./data/train_Arabic_tweets_negative.tsv',header=None, sep='\t')
    train_pos = pd.read_csv('./data/train_Arabic_tweets_positive.tsv',header=None, sep='\t')

class ml_model():

    def __init__(self,  model='xgboost',vectorizer='tfidf'):
        
        self.vectorizer_type = vectorizer    
        if vectorizer == 'tfidf':
            self.vectorizer = TfidfVectorizer(ngram_range=(1,2))
        elif vectorizer == 'word2vec':
            self.vectorizer = gensim.models.Word2Vec()
        elif vectorizer == 'fasttext':
          self.vectorizer = FastText()
            
        self.model_type = model
        if model == 'xgboost':
            self.model = XGBClassifier(random_state=42, seed=2, colsample_bytree=0.9, subsample=0.7,learning_rate=0.08)
        elif model == 'svm':
            self.model = svm.SVC(class_weight='balanced')
        elif model == 'rfc':
            self.model = RandomForestClassifier(class_weight='balanced')
        elif model == 'nb':
            self.model = GaussianNB()
        elif model == 'lr':
            self.model = LogisticRegression(max_iter=300)
            
#         self.pipeline = Pipeline([ ('vectorizer', self.vectorizer), ('model',self.model)]) 

        
        self.stopwords = set(stp.stopwords_list())
        self.emojis_ar = {}
        with open('emojis.csv','r',encoding='utf-8') as f:
            lines = f.readlines()
            
            for line in lines:
                line = line.strip('\n').split(';')
                self.emojis_ar.update({line[0].strip():line[1].strip()})
              
          
        
        
    def load_data(self, df):
        self.df = df.copy()
    
    def clean_review(self, text):
            
        text = re.sub("[إأٱآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        text = re.sub("ة", "ه", text)
        noise = re.compile(""" ّ    | # Tashdid
                                َ    | # Fatha
                                ً    | # Tanwin Fath
                                ُ    | # Damma
                                ٌ    | # Tanwin Damm
                                ِ    | # Kasra
                                ٍ    | # Tanwin Kasr
                                ْ    | # Sukun
                                ـ     # Tatwil/Kashida
                            """, re.VERBOSE)
        text = re.sub(noise, '', text)
        text = re.sub(r'(.)\1+', r"\1\1", text) # Remove longation
        text = word_tokenize(text)
        # text = " ".join(ISRIStemmer().stem(i) for i in text if i not in self.stopwords)
        text = " ".join(WordNetLemmatizer().lemmatize(i) for i in text if i not in self.stopwords)
    
        return text

        

    def remove_emoji(self, text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        return text



    def emoji_native_translation(self, text):
      if not text:
        return text
      loves = ["<3", "♥",'❤']
      smilefaces = []
      sadfaces = []
      neutralfaces = []

      eyes = ["8",":","=",";"]
      nose = ["'","`","-",r"\\"]
      for e in eyes:
          for n in nose:
              for s in ["\)", "d", "]", "}","p"]:
                  smilefaces.append(e+n+s)
                  smilefaces.append(e+s)
              for s in ["\(", "\[", "{"]:
                  sadfaces.append(e+n+s)
                  sadfaces.append(e+s)
              for s in ["\|", "\/", r"\\"]:
                  neutralfaces.append(e+n+s)
                  neutralfaces.append(e+s)
              #reversed
              for s in ["\(", "\[", "{"]:
                  smilefaces.append(s+n+e)
                  smilefaces.append(s+e)
              for s in ["\)", "\]", "}"]:
                  sadfaces.append(s+n+e)
                  sadfaces.append(s+e)
              for s in ["\|", "\/", r"\\"]:
                  neutralfaces.append(s+n+e)
                  neutralfaces.append(s+e)

      smilefaces = list(set(smilefaces))
      sadfaces = list(set(sadfaces))
      neutralfaces = list(set(neutralfaces))
      t = []

      for w in text.split():
          if w in loves:
              t.append("حب")
          elif w in smilefaces:
              t.append("مضحك")
          elif w in neutralfaces:
              t.append("عادي")
          elif w in sadfaces:
              t.append("محزن")
          else:
              t.append(w)
      newText = " ".join(t)
      return newText

    def is_emoji(self, word):
        if word in self.emojis_ar:
            return True
        else:
            return False
    
    def add_space(self, text):
        return ''.join(' ' + char if self.is_emoji(char) else char for char in text).strip()

    # !pip install aiogoogletrans
    # !pip install asyncio
    from aiogoogletrans import Translator
    translator = Translator()
    import asyncio
    loop = asyncio.get_event_loop()

    def translate_emojis(self, words):
        word_list = list()
        words_to_translate = list()
        for word in words :
            t = self.emojis_ar.get(word.get('emoji'),None)
            if t is None:
                word.update({'translation':'عادي','translated':True})
                #words_to_translate.append('normal')
            else:
                word.update({'translated':False,'translation':t})
                words_to_translate.append(t.replace(':','').replace('_',' '))
            word_list.append(word)
        return word_list

    def emoji_unicode_translation(self, text):
        if not text:
          return text
        text = self.add_space(text)
        words = text.split()
        text_list = list()
        emojis_list = list()
        c = 0
        for word in words:
            if self.is_emoji(word):
                emojis_list.append({'emoji':word,'emplacement':c})
            else:
                text_list.append(word)
            c+=1
        emojis_translated = self.translate_emojis(emojis_list)
        for em in emojis_translated:
            text_list.insert(em.get('emplacement'),em.get('translation'))
        text = " ".join(text_list)
        return text
        
    def clean_emoji(self, text):
        text = self.emoji_native_translation(text)
        text = self.emoji_unicode_translation(text)
        return text



    def preprocess(self, x):
        
        x = [self.clean_emoji(rev) for rev in x]
        x = [self.remove_emoji(rev) for rev in x]
        
        self.data = [self.clean_review(rev) for rev in x]
        
        print(self.data[0:10])
        
        return self.data
    
    def vectorize_data(self, x):
        if self.vectorizer_type == 'tfidf':
            return self.vectorizer.transform(x)
        elif self.vectorizer_type == 'fasttext' or self.vectorizer_type == 'word2vec':
            ft_arr = np.zeros((len(x), 500))
            for i in range(len(x)):
                ft_arr[i,:] = self.wordvec(x[i], 500)

            return ft_arr

    def wordvec(self, x, size):
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        for word in x:
            try:
                vec += self.vectorizer[word].reshape((1, size))
                count += 1.
            except KeyError: # handling the case where the token is not in vocabulary
                            
                continue
        if count != 0:
            vec /= count
        return vec
        
        
    def train(self, x, y):
        x = self.preprocess(x)
        
        if self.vectorizer_type == 'tfidf':
            self.vectorizer.fit_transform(x)
        elif self.vectorizer_type == 'fasttext':
            self.vectorizer = FastText(sentences=x, size= 500)
        elif self.vectorizer == 'word2vec':
            self.vectorizer = Word2Vec(sentences = x, size= 500)



        x = self.vectorize_data(x)
        
        if self.model_type == 'nb':
            self.model.fit(x.toarray(),y)
        else:
            self.model.fit(x,y)
        
        
    def score(self, x_test, y_test, show=True):
        x_test = self.preprocess(x_test)
        
        x_test = self.vectorize_data(x_test)
        
        if self.model_type == 'nb':
            y_pred = self.model.predict(x_test.toarray())
        else:
            y_pred = self.model.predict(x_test)
        
        self.f1 = f1_score(y_test, y_pred, average="macro")*100
        self.precision = precision_score(y_test, y_pred, average="macro")*(100)
        self.recall = recall_score(y_test, y_pred, average="macro")*100
        self.accuracy = accuracy_score(y_test, y_pred)*100
        self.auc = roc_auc_score(y_test,y_pred)

        if not show:
          return y_pred

        print(f'Confusion matrix: \n{confusion_matrix(y_test, y_pred)}\n')
        print(f'The F1 score is: {self.f1}')
        print(f'The precision score is: {self.precision}')
        print(f'The recall score is: {self.recall}') 
        print(f'The accuracy score is: {self.accuracy}\n\n\n')
        
        print(f'AUC score is: {self.auc}')
        
        return y_pred
        
    def predict(self, x):
        x = self.clean_review(x)
        
        x = self.vectorize_data([x])
        
        return(self.model.predict(x))
    
    def get_params(self):
        return self.pipeline.get_params()
    
    def draw_roc(self, x, y):
        
        y_pred = self.score(x,y)
        
        clf_fpr, clf_tpr, threshold = roc_curve(y, y_pred)
        auc_clf = auc(clf_fpr, clf_tpr)
        
        plt.figure(figsize=(5, 5), dpi=100)
        plt.plot(clf_fpr, clf_tpr, linestyle='-', label='clf (auc = %0.3f)' % auc_clf)

        plt.xlabel('False Positive Rate -->')
        plt.ylabel('True Positive Rate -->')

        plt.legend()

        plt.show()

    def draw_roc_mult(x, y, models):
        plt.figure(figsize=(5, 5), dpi=100)
        for model in models:
           y_pred = model[0].score(x,y,show=False)
           clf_fpr, clf_tpr, threshold = roc_curve(y, y_pred)
           auc_clf = auc(clf_fpr, clf_tpr)
           
           
           plt.plot(clf_fpr, clf_tpr, linestyle='-', label=f'{model[1]} (auc = %0.3f, f1 = %0.3f)' % (auc_clf,model[0].f1))
        
        plt.xlabel('False Positive Rate -->')
        plt.ylabel('True Positive Rate -->')

        plt.legend()

        plt.show()

if __name__ == "__main__":
    test_df = pd.concat([test_neg,test_pos], ignore_index=True)
    test_df.head()

    train_df = pd.concat([train_neg, train_pos], ignore_index=True)
    train_df.head()

    test_df[0][test_df[0] == 'neg'] = 0
    test_df[0][test_df[0] == 'pos'] = 1

    train_df[0][train_df[0] == 'neg'] = 0
    train_df[0][train_df[0] == 'pos'] = 1

    types_dict = { 0: bool}
    test_df = test_df.astype(types_dict)
    train_df = train_df.astype(types_dict)

    train_df.info()

    train_df.columns = ['feedback', 'review']
    test_df.columns = ['feedback', 'review']

    train_df = shuffle(train_df)
    test_df = shuffle(test_df)

    valid_df = pd.DataFrame()
    print(train_df.shape, valid_df.shape)
    _, valid_df['review'], _, valid_df['feedback'] = train_test_split(train_df['review'],train_df['feedback'],test_size=0.2,random_state=42)


    valid_df.shape, train_df.shape

    print(train_df['feedback'].value_counts())
    print(test_df['feedback'].value_counts())

    print(valid_df['review'].head(10))

    # xg_ft = ml_model('xgboost',vectorizer='fasttext')
    rfc_tf = ml_model('rfc', vectorizer='tfidf')
    svm_tf = ml_model('svm',vectorizer='tfidf')

    # xg_ft.train(train_df['review'], train_df['feedback'])
    # rfc_ft.train(train_df['review'], train_df['feedback'])
    rfc_tf.train(train_df['review'], train_df['feedback'])
    svm_tf.train(train_df['review'], train_df['feedback'])

    # xg_ft.score(test_df['review'],test_df['feedback'])
    # rfc_ft.score(test_df['review'],test_df['feedback'])
    # rfc_tf.score(test_df['review'],test_df['feedback'])
    # svm_tf.score(test_df['review'],test_df['feedback'])

    # xg_ft.draw_roc(test_df['review'],test_df['feedback'])
    # rfc_ft.draw_roc(test_df['review'],test_df['feedback'])
    # rfc_tf.draw_roc(test_df['review'],test_df['feedback'])
    # svm_tf.draw_roc(test_df['review'],test_df['feedback'])

    ml_model.draw_roc_mult(test_df['review'],test_df['feedback'], [(rfc_tf,'rfc_tf'), (svm_tf,'svm_tf')])

    import pickle

    pickle.dump(rfc_tf,open('rfc_tf_model','wb'))
    pickle.dump(svm_tf,open('svm_tf_model','wb'))

    pickle.dump(rfc_tf,open('rfc_tf_model','wb'))

    svm_tf.predict('الدودو جايه تكمل علي 💔')
    rfc_tf.predict('الدودو جايه تكمل علي 💔')

    model = pickle.load(open('svm_tf_model','rb'))

    ml_model.draw_roc_mult(test_df['review'],test_df['feedback'], [(rfc_tf,'rfc_tf'), (model,'svm_tf')])

