from stanfordcorenlp import StanfordCoreNLP

import logging
import json

import os
os.environ["CORENLP_HOME"] = r"C:\stanford-corenlp"

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000 , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        try:
            response = self.nlp.annotate(sentence, properties=self.props)
            return json.loads(response)
        except Exception as e:
            print(f"Error annotating sentence: {e}")
            return None

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens
        
    @staticmethod
    def post_process_ner(ner_tokens):
        """
        Post-process the NER tokens to merge Roman numerals with preceding PERSON entities
        and remove possessive suffixes like 's.
        """
        processed_entities = []
        current_entity = []
        
        for word, ner in ner_tokens:
            # Merge Roman numerals with preceding PERSON tokens
            if ner == "PERSON" or (current_entity and word.isdigit()):
                current_entity.append(word)
            elif current_entity:
                # Finalize the current entity
                processed_entities.append((" ".join(current_entity).rstrip("'s"), "PERSON"))
                current_entity = []
            # Exclude possessive suffix
            if ner != "O" and word != "'s":
                processed_entities.append((word, ner))
        
        # Finalize any remaining entity
        if current_entity:
            processed_entities.append((" ".join(current_entity).rstrip("'s"), "PERSON"))
        
        return processed_entities

if __name__ == '__main__':
    sNLP = StanfordNLP()
    text = "Charles I's treatment of Scotland."
    
    ano = sNLP.annotate(text)
    pos = sNLP.pos(text)
    tokens = sNLP.word_tokenize(text)
    ner = sNLP.ner(text)
    
    # Apply post-processing
    processed_ner = StanfordNLP.post_process_ner(ner)
    
    print("Original NER:", ner)
    print("Post-processed NER:", processed_ner)
    

 