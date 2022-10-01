from Sentiment_Analysis_OOP import ml_model

import pickle
class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'ml_model':
            from Sentiment_Analysis_OOP import ml_model
            return ml_model
        return super().find_class(module, name)

model = CustomUnpickler(open('svm_tf_model','rb')).load()

from fastapi import FastAPI

from pydantic import BaseModel

# Import the required libraires
from camel_tools.tokenizers.word import simple_word_tokenize
from textblob import TextBlob
from camel_tools.ner import NERecognizer

ner = NERecognizer.pretrained()


app = FastAPI()


class analyzer_item(BaseModel):
    sent: str = None

# @app.route("/")
@app.get("/")
def sentiment_analyzer(sent: str = None):

    if not sent:
        return ('No sentence provided!'),200
    sent = model.clean_review(sent)

    feedback = list(model.predict(sent))
    sent_pos = TextBlob(sent).tags

    sent_ner = list(zip(sent.split(),ner.predict_sentence(sent.split())))
    response = {'feedback': 'Positive' if feedback[0] else 'Negative',
                'sentence': sent,
                'POS': sent_pos,
                'NER': sent_ner}
    return (response, 200)

@app.post("/")
def home(item: analyzer_item):

    # if not sent:
    #     return ('No sentence provided!'),200
    sent = model.clean_review(item.sent)

    feedback = list(model.predict(sent))
    sent_pos = TextBlob(sent).tags

    sent_ner = list(zip(sent.split(),ner.predict_sentence(sent.split())))
    response = {'feedback': 'Positive' if feedback[0] else 'Negative',
                'sentence': sent,
                'POS': sent_pos,
                'NER': sent_ner}
    return (response, 200)

