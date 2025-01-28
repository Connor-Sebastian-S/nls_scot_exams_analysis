import os
import pandas as pd
import stanza

# Directory containing CSV files
DATA_DIR = "output"

def parse_directory(data_dir):
    directory_info = {}
    for year in os.listdir(data_dir):
        year_path = os.path.join(data_dir, year)
        if os.path.isdir(year_path):
            directory_info[year] = {}
            for level in os.listdir(year_path):
                level_path = os.path.join(year_path, level)
                if os.path.isdir(level_path):
                    directory_info[year][level] = {}
                    for paper in os.listdir(level_path):  # Handle paper subfolders
                        paper_path = os.path.join(level_path, paper)
                        if os.path.isdir(paper_path):
                            # Build full paths for each .csv file
                            subjects = {
                                os.path.splitext(file)[0]: os.path.join(paper_path, file)
                                for file in os.listdir(paper_path)
                                if file.endswith(".csv")
                            }
                            # Keep "Paper 1", "Paper 2" as keys, map to paths
                            directory_info[year][level][paper] = subjects
    return directory_info


directory_info = parse_directory(DATA_DIR)

def build_paths_with_extension(directory_info, base_path=""):
    paths = []
    for key, value in directory_info.items():
        current_path = os.path.join(base_path, key)
        if isinstance(value, dict):
            paths.extend(build_paths_with_extension(value, current_path))
        else:
            paths.append(f"{current_path}.csv")
    return paths

all_paths = build_paths_with_extension(directory_info)

from stanfordcorenlp import StanfordCoreNLP
import pathlib
import shlex
from pathlib import Path
from nltk.parse.corenlp import CoreNLPDependencyParser



from stanza.server import CoreNLPClient

os.environ["CORENLP_HOME"] = r"C:/stanford-corenlp"

print(os.environ.get("CORENLP_HOME"))

client = CoreNLPClient(
    endpoint="http://localhost:9000",  # Server URL
    timeout=30000,                    # Timeout in ms
    annotators=["tokenize", "ssplit", "ner"],  # Annotators to use
    be_quiet=True
)

dataframes = []
#for path in all_paths:
try:
    p = os.path.join(DATA_DIR, all_paths[0])
    p = os.path.abspath(p)
    p = os.path.normpath(p)


    df = pd.read_csv(p, dtype={'text': 'string'}, encoding='utf-8')

    # Start CoreNLPClient after loading the file
    for t in df['text']:
        t = str(t)
        ann = client.annotate(t)
        for sentence in ann.sentence:
            for entity in sentence.mentions:
                print(f"Entity: {entity.entityMentionText}, Type: {entity.entityType}")



except Exception as e:
    print(f"Error: {e}")

#finally:
   # client.stop()

    