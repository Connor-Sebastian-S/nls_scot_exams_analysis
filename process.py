import os
import re
import csv
import textstat
import matplotlib.pyplot as plt
import numpy as np
from genbit.genbit_metrics import GenBitMetrics
import pprint
import requests
import nltk
import regex as re
import pandas as pd
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

from bertopic import BERTopic
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from rake_nltk import Rake
from bertopic import BERTopic

from gpt4all import GPT4All
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import pipeline
import numpy as np
from sklearn.cluster import DBSCAN
import re
from collections import defaultdict
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from rake_nltk import Rake
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import BartTokenizer, BartForConditionalGeneration
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM
from kneed import KneeLocator
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TextClassificationPipeline



# Load the base LLaMA model
def get_intent(question):
    classifier = pipeline("text-classification", model="gokuls/distilbert-emotion-intent")
    text = question
    result = classifier(text)
    print(result)


def get_themes(question):
    
    nlp = spacy.load("en_core_web_sm")
    
    # Process the text with spaCy
    doc = nlp(question)
    
    # Extract intent dynamically using dependency parsing and POS tagging
    intent = None
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ == "ROOT":  # Focus on the main verb/root
            # Capture the root verb and its modifiers for multi-word intents
            intent = token.text
            # Add related modifiers (e.g., "compare and contrast")
            #intent_phrase = " ".join([child.text for child in token.children])# if child.dep_ in {"prep", "conj", "cc"}])
            #if intent_phrase:
            #    intent += f" {intent_phrase}"
            break
    
    # Extract named entities
    named_entities = [ent.text for ent in doc.ents]
    
    # Extract thematic keywords (noun chunks)
    thematic_keywords = []
    for chunk in doc.noun_chunks:
        # Simple filter for thematic relevance: contains at least one noun
        if any(token.pos_ == "NOUN" for token in chunk):
            thematic_keywords.append(chunk.text)
    
    return {"intent": intent, "named_entities": named_entities, "thematic_keywords": thematic_keywords}

  
def tokenize_text(text: str):
    
    # lowercase the text
    text = text.lower()
    
    # remove punctuation from text
    text = re.sub(r"[^\w\s]", "", text)
    
    # tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # remove stopwords from txt_tokens and word_tokens
    from nltk.corpus import stopwords
    english_stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in english_stop_words]
    
    # return your tokens
    return tokens

def lemmatize_tokens(tokens):
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # return your lemmatized tokens
    return lemmatized_tokens

# return the most common tokens
def return_top_tokens(tokens,
                      top_N = 10):

    # first, count the frequency of every unique token
    word_token_distribution = nltk.FreqDist(tokens)
    
    # next, filter for only the most common top_N tokens
    # also, put this in a dataframe
    top_tokens = pd.DataFrame(word_token_distribution.most_common(top_N),
                              columns=['Word', 'Frequency'])
    
    # return the top_tokens dataframe
    return top_tokens


# return the most common bi-grams
from nltk.collocations import BigramCollocationFinder

def return_top_bigrams(tokens,
                       top_N = 10):
    
    # collect bigrams
    bcf = BigramCollocationFinder.from_words(tokens)
    
    # put bigrams into a dataframe
    bigram_df = pd.DataFrame(data = bcf.ngram_fd.items(),
                             columns = ['Bigram', 'Frequency'])
    
    # sort the dataframe by frequency
    bigram_df = bigram_df.sort_values(by=['Frequency'],ascending = False).reset_index(drop=True)
    
    # filter for only top bigrams
    bigram_df = bigram_df[0:top_N]
    
    # return the bigram dataframe
    return bigram_df

from nltk.sentiment import SentimentIntensityAnalyzer

def return_sentiment_df(tokens):

	# initialize sentiment analyzer
	sia = SentimentIntensityAnalyzer()
	
	# create some counters for sentiment of each token
	positive_tokens = 0
	negative_tokens = 0
	neutral_tokens = 0
	compound_scores = []
		
	# loop through each token
	for token in tokens:
		
		if sia.polarity_scores(token)["compound"] > 0:
			
			positive_tokens += 1
			compound_scores.append(sia.polarity_scores(token)["compound"])
			
		elif sia.polarity_scores(token)["compound"] < 0:
			
			negative_tokens += 1
			compound_scores.append(sia.polarity_scores(token)["compound"])
			  
		elif sia.polarity_scores(token)["compound"] == 0:
			
			neutral_tokens += 1
			compound_scores.append(sia.polarity_scores(token)["compound"])
      
	
			
	# put sentiment results into a dataframe
	compound_score_numbers = [num for num in compound_scores if num != 0]
	csc = 0
	if len(compound_score_numbers) > 0:
		csc = sum(compound_score_numbers) / len(compound_score_numbers)
	sentiment_df = pd.DataFrame(data = {"total_tokens" : len(tokens),
										"positive_tokens" : positive_tokens,
										"negative_tokens" : negative_tokens,
										"neutral_tokens" : neutral_tokens,
										"compound_sentiment_score" : csc},
								index = [0])

	# return sentiment_df
	return sentiment_df


def process_exam_text(file_path):
    """
    Process an exam text file to extract metadata and questions.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        if len(lines) < 3:
            print(f"File {file_path} does not have enough lines for metadata extraction.")
            return None
        
        level = lines[0].strip()
        year = lines[1].strip()
        subject = lines[2].strip()

        text_from_questions = " ".join(line.strip() for line in lines[3:])
        question_pattern = r'\d+\.\s*(?:\([a-z]\)\s*)?'
        questions = re.split(question_pattern, text_from_questions)
        matches = re.findall(question_pattern, text_from_questions)
        
        structured_questions = []
        for i, question in enumerate(questions[1:]):
            question_number = matches[i].strip()
            structured_questions.append((question_number, question.strip()))

        return {
            "level": level,
            "year": year,
            "subject": subject,
            "questions": structured_questions
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


    
def Analyse_and_save_questions(metadata, output_file):
    """
    Analyse questions and save results to a CSV file.
    """
    question_pattern = r'\([a-z]\.?'  # Matches subquestion patterns like (a)

    textstat.set_lang("en_GB")
    rows = []
    
    def clean_question_number(number):
        """
        Cleans up the question identifier, removing errant parentheses or formatting issues.
        """
        #return re.sub(r'(\d+)\s*\(\w\)', r'\1', number)  # Matches cases like "1 (b)" and cleans to "1"
        return re.sub(r'\(\w\)', '', number)

    for q in metadata["questions"]:
        main_question_number = clean_question_number(q[0])  # Clean main question number
        main_question_text = re.split(question_pattern, q[1], maxsplit=1)[0].strip(')')
        
        # Analyse main question
        main_score = textstat.dale_chall_readability_score(main_question_text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(main_question_text)
        gunning_fog = textstat.gunning_fog(main_question_text)
        
        tokens = tokenize_text(main_question_text)
        lemmatized_tokens = lemmatize_tokens(tokens)
        sentiment_df = return_sentiment_df(lemmatized_tokens)
        
        #topics = get_themes(main_question_text)
        #get_intent(main_question_text)
        
        rows.append({
            "year": metadata["year"],
            "level": metadata["level"],
            "subject": metadata["subject"],
            "question": main_question_number.rstrip('.'),
            "text": main_question_text,
            "coleman_liau": main_score,
            "flesch_kincaid": flesch_kincaid_grade,
            "gunning_fog": gunning_fog,
            "total_tokens": int(sentiment_df["total_tokens"].iloc[0]),  # Extract scalar value
            "positive_tokens": int(sentiment_df["positive_tokens"].iloc[0]),
            "negative_tokens": int(sentiment_df["negative_tokens"].iloc[0]),
            "neutral_tokens": int(sentiment_df["neutral_tokens"].iloc[0]),
            "compound_sentiment_score": float(sentiment_df["compound_sentiment_score"].iloc[0])#,  
            #"intent": topics["intent"],
            #"named_entities": topics["named_entities"], 
            #"thematic_keywords": topics["thematic_keywords"]
        })

        # Analyse subquestions
        subquestions = re.split(question_pattern, q[1])
        subquestion_markers = re.findall(question_pattern, q[1])

        for idx, subtext in enumerate(subquestions[1:]):  # Skip main question text
            marker = subquestion_markers[idx].strip("()")  # Extract "a", "b", etc.
            sub_score = textstat.coleman_liau_index(subtext.strip()) 
            sub_flesch_kincaid_grade = textstat.flesch_kincaid_grade(subtext.strip())
            sub_gunning_fog = textstat.gunning_fog(subtext.strip())
            subtext = re.sub(r'^\)', '', subtext)
            
            tokens = tokenize_text(subtext)
            lemmatized_tokens = lemmatize_tokens(tokens)
            sentiment_df = return_sentiment_df(lemmatized_tokens)
            questionText = subtext.strip()
            #topics = get_themes(questionText)
            
            #get_intent(questionText)

            rows.append({
                "year": metadata["year"],
                "level": metadata["level"],
                "subject": metadata["subject"],
                "question": f"{main_question_number.rstrip('.')}{marker}",
                "text": questionText,
                "coleman_liau": sub_score,
                "flesch_kincaid": sub_flesch_kincaid_grade,
                "gunning_fog": sub_gunning_fog,
                "total_tokens": int(sentiment_df["total_tokens"].iloc[0]),  # Extract scalar value
                "positive_tokens": int(sentiment_df["positive_tokens"].iloc[0]),
                "negative_tokens": int(sentiment_df["negative_tokens"].iloc[0]),
                "neutral_tokens": int(sentiment_df["neutral_tokens"].iloc[0]),
                "compound_sentiment_score": float(sentiment_df["compound_sentiment_score"].iloc[0])#, 
                #"intent": topics["intent"],
                #"named_entities": topics["named_entities"], 
                #"thematic_keywords": topics["thematic_keywords"]
            })
            
    # Write to CSV
    with open(f"{output_file}_results.csv", 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "year", "level", "subject", "question", "text", 
            "coleman_liau", "flesch_kincaid", "gunning_fog",
            "total_tokens", "positive_tokens", "negative_tokens", 
            "neutral_tokens", "compound_sentiment_score", "intent", "named_entities", "thematic_keywords"
        ])
        writer.writeheader()
        writer.writerows(rows)
        
    
        
    score_value = np.array([row['gunning_fog'] for row in rows])
    question_id = np.arange(0,len(score_value))
    question_labels = [row['question'] for row in rows]

    plt.plot(question_id, score_value, 'o-')
    plt.xticks(question_id, question_labels)
    plt.xlabel('Question')
    plt.ylabel('Gunning Fog score')
    plt.title(f"{metadata["year"]} - {metadata["level"]} - {metadata["subject"]} - Gunning Fog score")
    plt.savefig(f"{output_file}_fog.png")
    plt.clf()

    print(f"Saved results to {output_file}")
    
    compound_sentiment_value = np.array([row['compound_sentiment_score'] for row in rows])
    plt.plot(question_id, compound_sentiment_value, 'o-')
    plt.xticks(question_id, question_labels)
    plt.xlabel('Question')
    plt.ylabel('Compound Sentiment score')
    plt.title(f"{metadata["year"]} - {metadata["level"]} - {metadata["subject"]} - Compound Sentiment score")
    plt.savefig(f"{output_file}_css.png")
    plt.clf()

    print(f"Saved results to {output_file}")
    
    
    

def process_all_files(folder_path, output_dir):
    """
    Process all files in the folder and save results to separate CSV files.
    """
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if not file_name.endswith('.txt'):
            continue  # Skip non-text files
        
        metadata = process_exam_text(file_path)
        #test(file_path)
        if not metadata:
            continue

        output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}")
        Analyse_and_save_questions(metadata, output_file)

# Example usage

#nltk.download()

# Summarizer pipeline


folder_path = './data/text/test/'  # Directory containing exam text files
output_dir = './output/'     # Directory to save CSV files
os.makedirs(output_dir, exist_ok=True)
        
process_all_files(folder_path, output_dir)
