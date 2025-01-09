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
    question_pattern = r'\([a-z]\.?'

    textstat.set_lang("en_GB")
    rows = []
    
    def print_pred(texts, pred):
        for txt, p in zip(texts, pred):
            print("formal: ", txt) if p == 0 else print("informal: ", txt)
            


    for q in metadata["questions"]:
        main_question_number = q[0]  # e.g., "1.", "2."
        main_question_text = re.split(question_pattern, q[1], maxsplit=1)[0].strip()
        
        # Analyse main question
        main_score = textstat.dale_chall_readability_score(main_question_text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(main_question_text)
        gunning_fog = textstat.gunning_fog(main_question_text)
        
        def remove_format(text):
            # Regular expression pattern to match "number. (letter)"
            cleaned_text = re.sub(r'(\d+)\.\s?\([a-zA-Z]\)', r'\1', text)

            return cleaned_text
            
        main_question_number = remove_format(main_question_number)
        
        tokens = tokenize_text(text = main_question_text)
        from nltk.stem import WordNetLemmatizer
        lemmatized_tokens = lemmatize_tokens(tokens = tokens)
        sentiment_df = return_sentiment_df(tokens = lemmatized_tokens)
        
        rows.append({
            "year": metadata["year"],
            "level": metadata["level"],
            "subject": metadata["subject"],
            "question": main_question_number.rstrip('.'),
            "text": main_question_text,
            "coleman_liau": main_score,
            "flesch_kincaid": flesch_kincaid_grade,
            "gunning_fog": gunning_fog,
            "total_tokens" : sentiment_df["total_tokens"],
            "positive_tokens" : sentiment_df["positive_tokens"],
            "negative_tokens" : sentiment_df["negative_tokens"],
            "neutral_tokens" : sentiment_df["neutral_tokens"],
            "compound_sentiment_score" : sentiment_df["compound_sentiment_score"]
        })

        # Analyse subquestions
        subquestions = re.split(question_pattern, q[1])
        subquestion_markers = re.findall(question_pattern, q[1])

        for idx, subtext in enumerate(subquestions[1:]):  # Skip main question text
            marker = subquestion_markers[idx].strip("()")  # Extract "a", "b", etc.
            sub_score = textstat.coleman_liau_index(subtext.strip()) 
            sub_flesch_kincaid_grade = textstat.flesch_kincaid_grade(subtext.strip())
            sub_gunning_fog = textstat.gunning_fog(subtext.strip())
            
            tokens = tokenize_text(text = subtext)
            from nltk.stem import WordNetLemmatizer
            lemmatized_tokens = lemmatize_tokens(tokens = tokens)
            sentiment_df = return_sentiment_df(tokens = lemmatized_tokens)     

            rows.append({
                "year": metadata["year"],
                "level": metadata["level"],
                "subject": metadata["subject"],
                "question": f"{main_question_number.rstrip('.')}{marker}",
                "text": subtext.strip(),
                "coleman_liau": sub_score,
                "flesch_kincaid": sub_flesch_kincaid_grade,
                "gunning_fog": sub_gunning_fog,
                "total_tokens" : sentiment_df["total_tokens"],
                "positive_tokens" : sentiment_df["positive_tokens"],
                "negative_tokens" : sentiment_df["negative_tokens"],
                "neutral_tokens" : sentiment_df["neutral_tokens"],
                "compound_sentiment_score" : sentiment_df["compound_sentiment_score"]
            })
            
    #for idx, q in enumerate([row['text'] for row in rows]): 
        #print(q)
        #tokens = tokenize_text(text = q)

        #from nltk.stem import WordNetLemmatizer
       # lemmatized_tokens = lemmatize_tokens(tokens = tokens)

        #top_tokens = return_top_tokens(tokens = lemmatized_tokens,
                               #top_N = 10)
        #print(top_tokens)

        # run the return_top_bigrams function and print the results
        #bigram_df = return_top_bigrams(tokens = lemmatized_tokens,
                               #top_N = 10)
        #print(bigram_df)

        #sentiment_df = return_sentiment_df(tokens = lemmatized_tokens)
        #print(sentiment_df)


        
    # Write to CSV
    with open(f"{output_file}_results.csv", 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["year", "level", "subject", "question", "text", "coleman_liau", "flesch_kincaid", "gunning_fog", "total_tokens", "positive_tokens", "negative_tokens", "neutral_tokens", "compound_sentiment_score"])
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
        if not metadata:
            continue

        output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}")
        Analyse_and_save_questions(metadata, output_file)

# Example usage

#nltk.download()
folder_path = './data/text/test/'  # Directory containing exam text files
output_dir = './output/'     # Directory to save CSV files
os.makedirs(output_dir, exist_ok=True)
        
process_all_files(folder_path, output_dir)
