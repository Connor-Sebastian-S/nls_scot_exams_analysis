import os
import re
import csv
import textstat
import matplotlib.pyplot as plt
import numpy as np
from genbit.genbit_metrics import GenBitMetrics
import pprint

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
    
    #genbit_metrics_object = GenBitMetrics("EN", context_window=5, distance_weight=0.95, percentile_cutoff=80)

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
        
        
        rows.append({
            "year": metadata["year"],
            "level": metadata["level"],
            "subject": metadata["subject"],
            "question": main_question_number.rstrip('.'),
            "text": main_question_text,
            "coleman_liau": main_score,
            "flesch_kincaid": flesch_kincaid_grade,
            "gunning_fog": gunning_fog
        })

        # Analyse subquestions
        subquestions = re.split(question_pattern, q[1])
        subquestion_markers = re.findall(question_pattern, q[1])

        for idx, subtext in enumerate(subquestions[1:]):  # Skip main question text
            marker = subquestion_markers[idx].strip("()")  # Extract "a", "b", etc.
            sub_score = textstat.coleman_liau_index(subtext.strip()) 
            sub_flesch_kincaid_grade = textstat.flesch_kincaid_grade(subtext.strip())
            sub_gunning_fog = textstat.gunning_fog(subtext.strip())
            rows.append({
                "year": metadata["year"],
                "level": metadata["level"],
                "subject": metadata["subject"],
                "question": f"{main_question_number.rstrip('.')}{marker}",
                "text": subtext.strip(),
                "coleman_liau": sub_score,
                "flesch_kincaid": sub_flesch_kincaid_grade,
                "gunning_fog": sub_gunning_fog
            })
            
    #for idx, q in enumerate([row['text'] for row in rows]): 


        
    # Write to CSV
    with open(f"{output_file}_results.csv", 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["year", "level", "subject", "question", "text", "coleman_liau", "flesch_kincaid", "gunning_fog"])
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
folder_path = './data/text/test/'  # Directory containing exam text files
output_dir = './output/'     # Directory to save CSV files
os.makedirs(output_dir, exist_ok=True)
process_all_files(folder_path, output_dir)
