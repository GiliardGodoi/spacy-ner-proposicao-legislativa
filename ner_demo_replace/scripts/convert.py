import json
import re
import srsly
import typer
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split

import spacy
from spacy.tokens import DocBin


# def convert(lang: str, input_path: Path, output_path: Path):
#     nlp = spacy.blank(lang)
#     db = DocBin()
#     for text, annot in srsly.read_json(input_path):
#         doc = nlp.make_doc(text)
#         ents = []
#         for start, end, label in annot["entities"]:
#             span = doc.char_span(start, end, label=label)
#             if span is None:
#                 msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
#                 warnings.warn(msg)
#             else:
#                 ents.append(span)
#         doc.ents = ents
#         db.add(doc)
#     db.to_disk(output_path)

def _create_docs_bin(nlp, examples, output_path):
    db = DocBin()
    for example in examples:
        text = re.sub(r'/', r'\\', example['text'])
        doc = nlp.make_doc(text)
        ents = list()
        for entity in example['entities']:
            start = entity['start_offset']
            end = entity['end_offset']
            label = entity['label']
            if label == 'DOC_NUMBER':
                continue
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print('Skipping entity')
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)

    db.to_disk(output_path)

def convert(lang : str, input_path : Path, output_dir : Path):

    with open(input_path, 'r', encoding='utf8') as file:
        data = [json.loads(line) for line in file]
    nlp = spacy.blank(lang)
    train, test = train_test_split(data, test_size=0.40, shuffle=True)
    output_train = output_dir / 'train.spacy'
    _create_docs_bin(nlp, train, output_train)

    output_test = output_dir / 'dev.spacy'
    _create_docs_bin(nlp, test, output_test)


if __name__ == "__main__":
    typer.run(convert)
