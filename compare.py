import spacy
import stanza
import logging
import pandas as pd

from typing import List
from abc import abstractmethod
from transformers import pipeline


class NERFacade:

    @abstractmethod
    def transform(self, text: str) -> List[str]:
        pass


class FastPDNNERFacade(NERFacade):

    def __init__(self):
        self.nlp = pipeline('ner', model='clarin-pl/FastPDN',
                            aggregation_strategy='simple')

    def transform(self, text: str) -> List[str]:
        doc = self.nlp(text)
        entities = [{
            "text": ent.get('word'),
            "type": ent.get('entity_group'),
            "start_char": ent.get('start'),
            "end_char": ent.get('end')
        } for ent in doc]
        return entities


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


if __name__ == '__main__':
    # execute all on text and save Pandas pickle
    text = ''
    with open('data/fida.md') as f:
        text = f.read().rstrip()

    nlps = {
        'Stanza-Default': StanzaNERFacade(),
        'Spacy-Default': SpacyNERFacade(),
        'FastPDN': FastPDNNERFacade(),
    }

    dfs = []
    for nlp_name, nlp in nlps.items():
        ents = nlp.transform(text)
        df_dict = {
            'nlp': [],
            'text': [],
            'type': [],
            'start_char': [],
            'end_char': [],
        }
        for ent in ents:
            df_dict['nlp'].append(nlp_name)
            df_dict['text'].append(ent['text'])
            df_dict['type'].append(ent['type'])
            df_dict['start_char'].append(ent['start_char'])
            df_dict['end_char'].append(ent['end_char'])
        nlp_df = pd.DataFrame.from_dict(df_dict)
        dfs.append(nlp_df)
    df = pd.concat(dfs)
    df.to_pickle('data.pkl')
