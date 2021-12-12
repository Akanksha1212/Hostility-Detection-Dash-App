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
    
    'logistic': {
        'name': "Logistic Regression",
    },
    'svm': {
        'name': "Support Vector Machine",
    },
    'multinomial': {
        'name': "Multinomial",
    },
    'logisticTFIDF': {
        'name': "Logistic Regression TF-IDF",
    },
    'svmTFIDF': {
        'name': "Support Vector Machine TF-IDF",
    },
    'multinomialTFIDF': {
        'name': "Multinomial TF-IDF",
    },
}


def tokenizer(text:str)->str:
    return text.split(' ') or text.split('  ')


# def explainer_class(method: str):
#     "Instantiate class using its string name"
#     classname = METHODS[method]['class']
#     # class_ = globals()[classname]
#     print(classname)
#     return classname




def multinomialModel():
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.8)
    return model

def svmModel():
    from sklearn import svm
    from sklearn.svm import SVC
    model = svm.SVC(kernel='rbf',probability=True)
    return model

def logisticModel():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(solver= 'newton-cg', penalty= 'l2', C = 0.5)
    return model




def explainer(method: str,text: str) -> LimeTextExplainer:
    """Run LIME explainer on provided classifier"""
    from sklearn.naive_bayes import MultinomialNB
    from lime.lime_text import LimeTextExplainer
    from lime.lime_text import TextDomainMapper
    from lime.lime_tabular import LimeTabularExplainer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    merged_data = pd.read_csv('models/merged_data.csv', header=0, index_col=0)
    
    hindi_stopwords = pd.read_csv('models/hindi_stopwords.csv', header=None)[0].tolist()
    stop_words = ['','.',' ','!','! ','!  ','!!','!!','!!!','।','।।','।।।','?','...',':','  ','ये चैनल्स','अरे ये','करा','मैं','है।']
    hindi_stopwords.extend(stop_words)
    train_y = merged_data['Hostile/Non-Hostile']
    vectorizer = CountVectorizer(encoding='ISCII',tokenizer=tokenizer, stop_words=hindi_stopwords).fit(merged_data['Filtered_Post_Stopword_Removed'])
    vectorizer1 = TfidfVectorizer(encoding='ISCII',tokenizer=tokenizer, stop_words=hindi_stopwords).fit(merged_data['Filtered_Post_Stopword_Removed'])
    merged_X_train_vectorized = vectorizer.transform(merged_data['Filtered_Post_Stopword_Removed'].tolist())
    merged_X_train_tfidf = vectorizer1.transform(merged_data['Filtered_Post_Stopword_Removed'].tolist())
    class_names = ['Non-Hostile', 'Hostile']
    
    a=method
    print(a)
    if a=="multinomial":
        model = multinomialModel()
        train_x=merged_X_train_vectorized
        vectori=vectorizer
    if a=="svm":
        model = svmModel()
        train_x=merged_X_train_vectorized
        vectori=vectorizer
    if a=="logistic":
        model = logisticModel()
        train_x=merged_X_train_vectorized
        vectori=vectorizer
    # model.fit(, merged_data['Hostile/Non-Hostile'].values)
    # predictor = model.predict
    if a=="multinomialTFIDF":
        model = multinomialModel()
        train_x=merged_X_train_vectorized
        vectori=vectorizer1
    if a=="svmTFIDF":
        model = svmModel()
        train_x=merged_X_train_vectorized
        vectori=vectorizer1
    if a=="logisticTFIDF":
        model = logisticModel()
        train_x=merged_X_train_vectorized
        vectori=vectorizer1

    model.fit(train_x, merged_data['Hostile/Non-Hostile'].values)
    

    

    
    explainer = LimeTextExplainer(class_names=class_names, split_expression= tokenizer)

    c = make_pipeline(vectori, model)
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



