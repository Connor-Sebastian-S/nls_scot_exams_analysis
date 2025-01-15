import os
import re
import csv
import textstat
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline

# load tokeniser and model
tokeniser = DistilBertTokenizer.from_pretrained("./results/checkpoint-final")
model = DistilBertForSequenceClassification.from_pretrained("./results/checkpoint-final")
classifier = pipeline("text-classification", model=model, tokenizer=tokeniser)    
label_mapping = {
    0: "discuss",
    1: "describe",
    2: "compare",
    3: "explain",
    4: "argue",
    5: "reason",
    6: "other"
}

def get_intent(question):

    # get predictions
    predictions = classifier(question)
    
    # return prediction
    for pred in predictions:
        # map label ID to name
        pred['label'] = label_mapping[int(pred['label'].split('_')[-1])]
        return (pred)

def tokenise_text(text: str):
    
    # lowercase the text
    text = text.lower()
    
    # remove punctuation from text
    text = re.sub(r"[^\w\s]", "", text)
    
    # tokenise the text
    tokens = nltk.word_tokenize(text)
    
    # remove stopwords from txt_tokens and word_tokens
    from nltk.corpus import stopwords
    english_stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in english_stop_words]
    
    # return your tokens
    return tokens

def lemmatise_tokens(tokens):
    
    # initiate lemmatiser
    lemmatiser = WordNetLemmatizer()
    
    # lemmatise tokens
    lemmatiser = WordNetLemmatizer()
    lemmatised_tokens = [lemmatiser.lemmatize(word) for word in tokens]
    
    # return your lemmatised tokens
    return lemmatised_tokens

# return the most common tokens
def return_top_tokens(tokens, top_N = 10):

    # first, count the frequency of every unique token
    word_token_distribution = nltk.FreqDist(tokens)
    
    # next, filter for only the most common top_N tokens
    # also, put this in a dataframe
    top_tokens = pd.DataFrame(word_token_distribution.most_common(top_N),
                              columns=['Word', 'Frequency'])
    
    # return the top_tokens dataframe
    return top_tokens


def return_top_bigrams(tokens, top_N = 10):
    
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

def return_sentiment_df(tokens):

	# initialise sentiment analyzer
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
    # matches subquestion patterns like (a)
    question_pattern = r'\([a-z]\.?'

    textstat.set_lang("en_GB")
    rows = []
    
    def clean_question_number(number):
        """
        Cleans up the question identifier, removing errant parentheses or formatting issues.
        """
        return re.sub(r'\(\w\)', '', number)

    for q in metadata["questions"]:
        main_question_number = clean_question_number(q[0])  # Clean main question number
        main_question_text = re.split(question_pattern, q[1], maxsplit=1)[0].strip(')')
        
        # analyse main question
        main_score = textstat.dale_chall_readability_score(main_question_text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(main_question_text)
        gunning_fog = textstat.gunning_fog(main_question_text)
        
        tokens = tokenise_text(main_question_text)
        lemmatised_tokens = lemmatise_tokens(tokens)
        sentiment_df = return_sentiment_df(lemmatised_tokens)
        intent_ = get_intent(main_question_text)

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
            "compound_sentiment_score": float(sentiment_df["compound_sentiment_score"].iloc[0]),  
            "intent": intent_["label"],
            "intent_certainty": intent_["score"]      
            #"named_entities": topics["named_entities"], 
            #"thematic_keywords": topics["thematic_keywords"]
        })

        # analyse subquestions
        subquestions = re.split(question_pattern, q[1])
        subquestion_markers = re.findall(question_pattern, q[1])

        for idx, subtext in enumerate(subquestions[1:]):  # Skip main question text
            marker = subquestion_markers[idx].strip("()")  # Extract "a", "b", etc.
            sub_score = textstat.coleman_liau_index(subtext.strip()) 
            sub_flesch_kincaid_grade = textstat.flesch_kincaid_grade(subtext.strip())
            sub_gunning_fog = textstat.gunning_fog(subtext.strip())
            subtext = re.sub(r'^\)', '', subtext)
            
            tokens = tokenise_text(subtext)
            lemmatised_tokens = lemmatise_tokens(tokens)
            sentiment_df = return_sentiment_df(lemmatised_tokens)
            questionText = subtext.strip()
            intent_ = get_intent(subtext)

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
                "compound_sentiment_score": float(sentiment_df["compound_sentiment_score"].iloc[0]), 
                "intent": intent_["label"],
                "intent_certainty": intent_["score"]
                #"named_entities": topics["named_entities"], 
                #"thematic_keywords": topics["thematic_keywords"]
            })
            
    # Write to CSV
    with open(output_file + re.sub(r"[^\w\s]", "", metadata["subject"]) + 
    ".csv", 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "year", "level", "subject", 
            "question", "text", 
            "coleman_liau", "flesch_kincaid", "gunning_fog",
            "total_tokens", "positive_tokens", "negative_tokens", "neutral_tokens", 
            "compound_sentiment_score", 
            "intent", "intent_certainty",
            "named_entities", "thematic_keywords"
        ])
        writer.writeheader()
        writer.writerows(rows)

def process_all_files(folder_path, output_dir):
    """
    Process all files in the folder and save results to separate CSV files.
    """
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if not file_name.endswith('.txt'):
            # skip non-text files
            continue  
        
        metadata = process_exam_text(file_path)
        if not metadata:
            continue
        
        csv_file_name = (
            metadata["year"] + 
            "/" + 
            re.sub(r"[^\w\s]", "", metadata["level"]) + 
            "/")
        print (csv_file_name)
        
        output_file = os.path.join(output_dir, csv_file_name)
        print (output_file)
        os.makedirs(output_file)
        Analyse_and_save_questions(metadata, output_file)

# directory containing exam text files
folder_path = './data/text/test/'  
# directory to save CSV files
output_dir = './output/'     
os.makedirs(output_dir, exist_ok=True)
        
process_all_files(folder_path, output_dir)
