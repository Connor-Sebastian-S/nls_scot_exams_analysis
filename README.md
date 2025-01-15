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
├── 1945
│   ├── National 5
│       ├── ENGLISH.csv
│   └── Higher
│       ├── HISTORY.csv
│       ├── MODERN_STUDIES.csv
├── 1985
│   ├── National 4
│       ├── HISTORY.csv
│   ├── National 5
│       ├── HISTORY.csv
│   └── Higher
│       ├── ENGLISH.csv
├── 2024
│   └── Higher
│       ├── HISTORY.csv
│       ├── ENGLISH.csv

* process.py - reads the text files, splits into questions, calculates the aforementioned data, and saves as a CSV per exam paper.
* train.py - trains our model for intent calculation.
* eval.py - test the model on a single sentence.
* analyse.py - presents the data in various forms, compares trends over time, etc. (makes pretty plots)

The trained model is too big to upload here :(

## Qualification History ##

I've attempted to map any of the old-style grade names to their modern equivalent under the Curriculum for Excellence.

Scottish secondary education has evolved significantly since 1900, with changes reflecting broader educational reforms, societal shifts, and the establishment of a standardized curriculum. Here's a summary of the key levels and stages over time:

---

### **1900-1947: Pre-WWII to Post-War Period**
1. **Primary Education** (Ages 5-12):  
   Primary school was the main form of education until the introduction of secondary stages.

2. **Secondary Education** (Ages 12-14/15):  
   - Junior Secondary: For students not pursuing academic qualifications.
   - Senior Secondary: For students aiming for formal qualifications like the **Scottish Leaving Certificate** (introduced in 1888, expanded over time).

---

### **1947-1960: Post-War Reforms**
1. **Primary Education**: Extended until age 12.  
2. **Secondary Education**: Divided based on academic or vocational paths:
   - **Junior Secondary Schools**: Focused on vocational training.
   - **Senior Secondary Schools**: Prepared students for higher education or professional careers.  

   Key qualification: **Scottish Leaving Certificate** (updated with new subjects and grades).

---

### **1960s-1970s: Comprehensive Education System**
1. **Introduction of Comprehensive Schools**:  
   The move towards a non-selective, comprehensive system reduced the division between junior and senior secondary schools.

2. **Scottish Certificate of Education (SCE)**:  
   Replaced the Leaving Certificate in 1962.  
   Levels:
   - **O-Grades** (Ordinary): General qualifications taken at age 16.  
   - **H-Grades** (Higher): Advanced qualifications for students aged 17-18.

---

### **1980s-1990s: Standard Grades Era**
1. **Primary Education**: Ages 5-12.  
2. **Secondary Education** (Ages 12-18):  
   Levels introduced with the **Standard Grades** (1986):  
   - Foundation Level.  
   - General Level.  
   - Credit Level.  
   
   Higher qualifications included:  
   - **Highers**: Advanced qualifications (post-16).  
   - **Certificate of Sixth Year Studies (CSYS)**: Optional advanced level post-Highers.

---

### **2000s: Curriculum for Excellence (CfE)**
1. **Broad General Education (BGE)** (Ages 3-15):  
   - Covers early years, primary, and the first three years of secondary education (S1-S3).  
   
2. **Senior Phase (S4-S6)** (Ages 15-18):  
   Replaces Standard Grades with:
   - **National 1-5**: Basic to more advanced qualifications.  
   - **Highers**: Intermediate advanced level.  
   - **Advanced Highers**: Pre-university qualifications.

---

### **Current Levels (2025)**  
**Primary School**: P1-P7 (Ages 5-12).  
**Secondary School**: S1-S6 (Ages 12-18).  
- **Broad General Education (BGE)**: S1-S3.  
- **Senior Phase**: S4-S6.  
   Key qualifications:
   - **National 1-5**.  
   - **Highers**.  
   - **Advanced Highers**.

---
#### The mapping itself
```
    # Mapping of historical grades to modern equivalents
    mapping = {
        "Lower Grade": "National 4",
        "Intermediate Grade": "National 5",
        "Higher Grade": "Higher",
        "Ordinary Grade": "National 5",
        "Foundation Standard Grade": "National 3",
        "General Standard Grade": "National 4",
        "Credit Standard Grade": "National 5",
        "CSYS": "Advanced Higher",
        "National 1": "National 1",
        "National 2": "National 2",
        "National 3": "National 3",
        "National 4": "National 4",
        "National 5": "National 5",
        "Higher": "Higher",
        "Advanced Higher": "Advanced Higher",
    }
```
