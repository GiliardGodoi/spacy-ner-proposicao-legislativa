from pathlib import Path

import spacy
import spacy_streamlit
import streamlit as st

from spacy_streamlit import visualize_ner

visualizers = ["ner",]

@st.cache_resource
def get_spacy_model():
    return spacy.load(Path('..', 'ner_demo_replace', 'training', 'model-best'))

if __name__ == "__main__":

    nlp = get_spacy_model()
    labels = nlp.get_pipe('ner').labels

    text = st.text_area(label='Requerimento')

    if text:
        visualize_ner(nlp(text),
                    labels=labels,
                    show_table=False,
                    )
