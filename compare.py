import spacy
import stanza
import logging

from typing import List
from abc import abstractmethod


class NERFacade:

    @abstractmethod
    def transform(self, text: str) -> List[str]:
        pass


class StanzaNERFacade(NERFacade):

    def __init__(self, lang='pl'):
        logging.getLogger('stanza').setLevel(logging.ERROR)
        stanza.download('pl')
        self.nlp = stanza.Pipeline(lang)

    def transform(self, text: str) -> List[str]:
        doc = self.nlp(text)
        entities = [{
            "text": ent.text,
            "type": ent.type,
            "start_char": ent.start_char,
            "end_char": ent.end_char
        } for ent in doc.entities]
        return entities


class SpacyNERFacade(NERFacade):

    def __init__(self, model='pl_core_news_lg'):
        self.nlp = spacy.load(model)

    def transform(self, text: str) -> List[str]:
        doc = self.nlp(text)
        entities = [{
            "text": ent.text,
            "type": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char
        } for ent in doc.ents]
        return entities
    


if __name__ == '__main__':
    text = ''
    with open('data/fida.md') as f:
        text = f.read().rstrip()
    
    nlps = [
        StanzaNERFacade(),
        SpacyNERFacade()
    ]

    for nlp in nlps:
        ents = nlp.transform(text)
        print(ents)