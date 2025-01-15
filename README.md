# Linguistic Analysis of Scottish School Exam Papers #
## Intent ##

Data available at: https://data.nls.uk/data/digitised-collections/scottish-exams/
The Scottish School Exam Papers dataset provides a unique opportunity to explore how the language used in education reflects evolving societal, pedagogical, and linguistic trends. By examining the phrasing, structure, and vocabulary of exam questions over time, this project aims to uncover shifts in accessibility, inclusivity, and readability within Scotland's education system. Such an analysis can offer insights into how language in assessments aligns with broader historical, social, and educational changes.

By measuring linguistic features such as sentence length, complexity, and vocabulary diversity, trends in the evolution of exam question readability can be identified. For instance, earlier exams may reveal long, complex sentence structures reflective of 19th-century academic norms, while more modern exams could display shorter, more direct phrasing influenced by contemporary pedagogical principles emphasising clarity and accessibility.

Another dimension of the analysis focuses on inclusivity in language. Historical exams may reveal implicit biases in phrasing or content, such as gendered language or cultural assumptions that privilege specific societal groups. Tracking changes in the language used to describe historical figures, literature, or scientific phenomena can highlight how educational materials have adapted to incorporate broader perspectives and ensure representation.

## Progress ##
* Done
  * Read NLS text files
  * Split into question (1, 2, 3, etc.)
  * Split into subquestions (a, b, c, etc.)
  * Perform "coleman liau", "flesch kincaid", and "gunning fog" analysis on questions and subquestions
  * Output to CSV
  * Create and save labelled plots of "gunning fog" score per paper

## Testing and development ##
The first test uses English papers from 1901, 1961, and 2024 and calculates their per-question Gunning Fog Index, Coleman Liau score, and their Flesch Kincaid score. It also calculates the total tokens per question, positive tokens, negative tokens , and neutral_tokens. It then calculates the Compound Sentiment Score. Finally it uses a custom trained BART model to determine the "intent" of each question, this represents what the question is asking the reader to do. Labels for this are "discuss", "describe", "compare", "explain", "argue",  "reason",  or "other". 

The text files for the exam papers don't need a particular name, however their structure must be specific.

The folder structure of the outputted CSV files are as follows:
* Output
 * Year
  * Level
   * SUBJECT_NAME.csv

* process.py - reads the text files, splits into questions, calculates the aforementioned data, and saves as a CSV per exam paper.
* train.py - trains our model for intent calculation.
* eval.py - test the model on a single sentence.
* analyse.py - presents the data in various forms, compares trends over time, etc. (makes pretty plots)

The trained model is too big to upload here :(
