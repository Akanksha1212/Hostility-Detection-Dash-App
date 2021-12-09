import argparse
from pathlib import Path
from typing import List, Any
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from lime.lime_text import LimeTextExplainer
import sklearn.pipeline
import scipy.stats
import spacy
from sklearn.pipeline import make_pipeline

METHODS = {
    'textblob': {
        'class': "TextBlobExplainer",
        'file': None,
        'name': "TextBlob",
        'lowercase': False,
    },
    'vader': {
        'class': "VaderExplainer",
        'file': None,
        'name': "VADER",
        'lowercase': False,
    },
    'logistic': {
        'class': "LogisticExplainer",
        'file': "models/merged_data.csv",
        'name': "Logistic Regression",
        'lowercase': False,
    },
    'svm': {
        'class': "SVMExplainer",
        'file': "models/merged_data.csv",
        'name': "Support Vector Machine",
        'lowercase': False,
    },
    'fasttext': {
        'class': "FastTextExplainer",
        'file': "models/fasttext/multinomialClassifier1.pkl",
        'name': "Multinomial",
        'lowercase': False,
    },
}


def tokenizer(s):
    return s.split(' ') or s.split('  ')
# def tokenizer(text: str) -> str:
#     "Tokenize input string using a spaCy pipeline"
#     nlp = spacy.blank('hi')
#     nlp.add_pipe('sentencizer')  # Very basic NLP pipeline in spaCy
#     doc = nlp(text)
#     tokenized_text = ' '.join(token.text for token in doc)
#     print(tokenized_text)
#     return tokenized_text


# def explainer_class(method: str, filename: str) -> Any:
#     "Instantiate class using its string name"
#     classname = METHODS[method]['class']
#     class_ = globals()[classname]
    
#     return class_(filename)



# class LogisticExplainer:
#     """Class to explain classification results of a scikit-learn
#        Logistic Regression Pipeline. The model is trained within this class.
#     """
#     def __init__(self, path_to_train_data: str) -> None:
#         # "Input training data path for training Logistic Regression classifier"
#         import pandas as pd
#         # Read in training data set
#         self.train_df = pd.read_csv(path_to_train_data, header=0, index_col=0)
#         print(self.train_df['Filtered_Post_Stopword_Removed'])

#     def train(self) -> sklearn.pipeline.Pipeline:
#         # "Create sklearn logistic regression model pipeline"
#         from sklearn.feature_extraction.text import CountVectorizer
#         from sklearn.pipeline import Pipeline
#         from sklearn.naive_bayes import MultinomialNB
#         import pickle

#         # classifier=pickle.load(open('models/fasttext/multinomialClassifier1.pkl', 'rb'))
#         pipeline = Pipeline(
#             [
#                 ('vect', CountVectorizer()),
#                 ('clf', self.MultinomialNB(alpha=0.8)),
#             ]
#         )
#         # Train model
#         classifier = pipeline.fit(self.train_df['Filtered_Post_Stopword_Removed'], self.train_df['Hostile/Non-Hostile'])
#         return classifier

#     def predict(self, texts: List[str]) -> np.array([float, ...]):
#         """Generate an array of predicted scores (probabilities) from sklearn
#         Logistic Regression Pipeline."""
#         classifier = self.train()
#         probs = classifier.predict_proba(texts)
#         return probs

def explainer(text: str) -> LimeTextExplainer:
    """Run LIME explainer on provided classifier"""
    from sklearn.naive_bayes import MultinomialNB
    from lime.lime_text import LimeTextExplainer
    from lime.lime_text import TextDomainMapper
    from lime.lime_tabular import LimeTabularExplainer
    from sklearn.feature_extraction.text import CountVectorizer
    merged_data = pd.read_csv('models/merged_data.csv', header=0, index_col=0)
    
    hindi_stopwords = pd.read_csv('models/hindi_stopwords.csv', header=None)[0].tolist()
    stop_words = ['','.',' ','!','! ','!  ','!!','!!','!!!','।','।।','।।।','?','...',':','  ','ये चैनल्स','अरे ये','करा','मैं','है।']
    hindi_stopwords.extend(stop_words)
    train_y = merged_data['Hostile/Non-Hostile']
    vectorizer = CountVectorizer(encoding='ISCII',tokenizer=tokenizer, stop_words=hindi_stopwords).fit(merged_data['Filtered_Post_Stopword_Removed'])

    merged_X_train_vectorized = vectorizer.transform(merged_data['Filtered_Post_Stopword_Removed'].tolist())

    model = MultinomialNB(alpha=0.8)
    model.fit(merged_X_train_vectorized, merged_data['Hostile/Non-Hostile'].values)
    # predictor = model.predict
    

    class_names = ['Non-Hostile', 'Hostile']

    
    explainer = LimeTextExplainer(class_names=class_names, split_expression= tokenizer)

    c = make_pipeline(vectorizer, model)
    # Make a prediction and explain it:
    exp = explainer.explain_instance(text, c.predict_proba, num_features=50)

    return exp

def main(samples: List[str]) -> None:
    # Get list of available methods:
    # method_list = [method for method in METHODS.keys()]
    # Arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--method', type=str, nargs='+', help="Enter one or more methods \
    #                     (Choose from following: {})".format(", ".join(method_list)),
    #                     required=True)
    # parser.add_argument('-n', '--num_samples', type=int, help="Number of samples for explainer \
    #                     instance", default=1000)
    # args = parser.parse_args()
    exp = explainer()
    # for method in args.method:
    #     if method not in METHODS.keys():
    #         parser.error("Please choose from the below existing methods! \n{}".format(", ".join(method_list)))
    #     #  path_to_file = METHODS[method]['file']
    #     # ENABLE_LOWER_CASE = METHODS[method]['lowercase']
    #     # Run explainer function
    #     print("Method: {}".format(method.upper()))
        
        # for i, text in enumerate(samples):
        #     text = tokenizer(text)  # Tokenize text using spaCy before explaining
        #     print("Generating LIME explanation for example {}: `{}`".format(i+1, text))
        #     exp = explainer()
        #     # Output to HTML
        #     output_filename = Path(__file__).parent / "{}-explanation-{}.html".format(i+1, method)
        #     exp.save_to_file(output_filename)


if __name__ == "__main__":
    # Evaluation text
    samples = [
       "भारत, पाकिस्तान और चीन सहित दुनिया भर की ताज़ा ख़बरें",
        "भरोसेमंद हो, एक दिन साथ छोड़ ही जाते है",
    ]
    main(samples)



