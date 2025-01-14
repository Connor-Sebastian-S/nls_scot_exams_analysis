import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import spacy
import re
from typing import List, Dict, Any

class ExamQuestionAnalyzer:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with exam question data.
        
        Args:
            data (pd.DataFrame): DataFrame containing exam questions with at least 'text' column
        """
        self.df = data
        self.nlp = spacy.load('en_core_web_sm')
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize text for analysis.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ''
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.?]', ' ', text.lower())
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_question_types(self) -> Dict[str, List[str]]:
        """
        Extract different types of questions based on sentence structure and verbs.
        
        Returns:
            Dict[str, List[str]]: Dictionary of question types and their examples
        """
        question_types = {
            'analytical': [],
            'comparative': [],
            'descriptive': [],
            'evaluative': [],
            'other': []
        }
        
        for text in self.df['text']:
            doc = self.nlp(self.preprocess_text(text))
            
            # Analyze sentence structure and main verbs
            main_verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB']
            
            # Categorize based on verb patterns and sentence structure
            if any(v in ['analyze', 'examine', 'discuss'] for v in main_verbs):
                question_types['analytical'].append(text)
            elif any(v in ['compare', 'contrast', 'differentiate'] for v in main_verbs):
                question_types['comparative'].append(text)
            elif any(v in ['describe', 'explain', 'illustrate'] for v in main_verbs):
                question_types['descriptive'].append(text)
            elif any(v in ['evaluate', 'assess', 'judge'] for v in main_verbs):
                question_types['evaluative'].append(text)
            else:
                question_types['other'].append(text)
                
        return question_types
    
    def identify_topics(self, n_topics: int = 5) -> Dict[str, List[str]]:
        """
        Identify main topics in questions using TF-IDF and clustering.
        
        Args:
            n_topics (int): Number of topics to identify
            
        Returns:
            Dict[str, List[str]]: Dictionary of topics and their related questions
        """
        # Create TF-IDF matrix
        texts = [self.preprocess_text(text) for text in self.df['text']]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_topics, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Get top terms for each cluster
        feature_names = self.vectorizer.get_feature_names_out()
        topics = {}
        
        for i in range(n_topics):
            cluster_docs = [text for j, text in enumerate(self.df['text']) if clusters[j] == i]
            topics[f'Topic_{i+1}'] = cluster_docs
            
        return topics
    
    def analyze_complexity(self) -> pd.DataFrame:
        """
        Analyze question complexity using available metrics.
        
        Returns:
            pd.DataFrame: DataFrame with complexity metrics
        """
        metrics = pd.DataFrame()
        
        # Use available readability scores
        if 'coleman_liau' in self.df.columns:
            metrics['coleman_liau'] = self.df['coleman_liau']
        if 'flesch_kincaid' in self.df.columns:
            metrics['flesch_kincaid'] = self.df['flesch_kincaid']
        if 'gunning_fog' in self.df.columns:
            metrics['gunning_fog'] = self.df['gunning_fog']
            
        # Add basic complexity metrics
        metrics['word_count'] = self.df['text'].apply(lambda x: len(str(x).split()))
        metrics['sentence_count'] = self.df['text'].apply(lambda x: len(str(x).split('.')))
        
        return metrics
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.
        
        Returns:
            Dict[str, Any]: Dictionary containing analysis results
        """
        report = {
            'question_types': self.extract_question_types(),
            'topics': self.identify_topics(),
            'complexity_analysis': self.analyze_complexity().describe(),
            'summary_stats': {
                'total_questions': len(self.df),
                'avg_word_count': self.df['text'].apply(lambda x: len(str(x).split())).mean(),
                'complexity_distribution': self.analyze_complexity().mean().to_dict()
            }
        }
        
        return report

def main():
    # Example usage
    df = pd.read_csv('./output/test_results.csv')
    analyzer = ExamQuestionAnalyzer(df)
    report = analyzer.generate_report()
    
    # Print summary of findings
    print("\n=== Exam Question Analysis Report ===")
    print("\nQuestion Types Distribution:")
    for qtype, questions in report['question_types'].items():
        print(f"{qtype}: {len(questions)} questions")
        
    print("\nComplexity Metrics (Average):")
    for metric, value in report['summary_stats']['complexity_distribution'].items():
        print(f"{metric}: {value:.2f}")
        
    print("\nIdentified Topics:")
    for topic, questions in report['topics'].items():
        print(f"\n{topic}: {len(questions)} questions")
        if questions:
            print(f"Sample question: {questions[0][:100]}...")

if __name__ == "__main__":
    main()