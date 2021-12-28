# import spacy
# from spacy.cli.train import train
from __future__ import unicode_literals, print_function
# import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import os
import argparse
from spacy.tokens import Doc
from spacy.training import Example
from tqdm import tqdm
from data_reader import ConllReader
# Construction from scratch
from spacy.vocab import Vocab
from spacy.language import Language
# from spacy.gold import GoldParse

from spacy_conll import example_from_conllu_sentence

# nlp = ConllParser(init_parser("en_core_web_sm", "spacy"))
#
# doc = nlp.parse_conll_file_as_spacy("path/to/your/conll-sample.txt")

nlp = Language(Vocab())
# nlp = spacy.load("en_core_web_sm")


# doc = nlp("I eat an apple")
# for word in doc:
#     print(f"word index: {word.i}, word: {word.text}, pos: {word.pos_}, dep: {word.dep_}, head: {word.head.text}, head index: {word.head.i}")
# train("base_config.cfg", overrides={"paths.train": "spacy_data/he_htb-ud-train.json", "paths.dev": "spacy_data/he_htb-ud-dev.json"})
# model_name = "en_core_web_sm"

# nlp = spacy.load(model_name)
# print(f"Loaded model {model_name}")
# parser = nlp.get_pipe('parser')
#
# for _, annotations in TRAIN_DATA:
#     for ent in annotations.get('entities'):
#         parser.add_label(ent[2])
#
# other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
# with nlp.disable_pipes(*other_pipes):  # only train NER
#     optimizer = nlp.begin_training()
#     for itn in range(n_iter):
#         random.shuffle(TRAIN_DATA)
#         losses = {}
#         for text, annotations in tqdm(TRAIN_DATA):
#             nlp.update(
#                 [text],
#                 [annotations],
#                 drop=0.5,
#                 sgd=optimizer,
#                 losses=losses)
#         print(losses)


def get_sentences(file_path):
    sentences = []

    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()
        curr_sentence = []
        for line in lines:

            if line == "\n":
                sentences.append(curr_sentence)
                curr_sentence = []
                continue

            if line.startswith("#"):
                continue

            curr_sentence.append(line)

    return sentences

TRAIN_DATA = [
    ("Stefflon Don is on the periphery of global greatness.", {
        'deps': ['compound', 'nsubj', 'cop', 'case', 'det', 'root', 'case', 'amod', 'nmod', 'punct'],
        "heads": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }),
    ("From Jools Holland to the BBC Sound Poll, Steff's powerful presence commands attention both on record and in real life.", {
        'deps': ['case', 'compound', 'nmod', 'case', 'det', 'compound', 'compound', 'nmod', 'punct', 'nmod:poss', 'case', 'amod', 'nsubj', 'root', 'dobj', 'cc:preconj', 'case', 'nmod', 'cc', 'case', 'amod', 'conj', 'punct'],
        "heads": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    })
]

def pipeline(train_path, model_name, n_epochs):
    # train_data = spacy.spacy_data_reader.spacy_load_data(train_file)
    # train_data = [(["I", "eat", "an", "apple"], ["DET", "NOUN", "DET", "NOUN"], [1, 0, 3, 1], ["A", "B", "C", "D"]), (["you", "eat", "an", "apple"], ["DET", "NOUN", "DET", "NOUN"], [1, 0, 3, 1], ["A", "B", "C", "D"])]
    # nlp = spacy.load(model_name)
    # nlp = spacy.blank('en')
    train_data = ConllReader("train").get_examples()
    # train_data = get_sentences("data\he_htb-ud-train.conllu")
    # add the tagger to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    # parser = nlp.create_pipe('parser')
    # parser = nlp.create_pipe('parser')
    parser = nlp.add_pipe("parser", first=True)
    # parser.initialize(lambda: [], nlp=nlp)
    # Add the tags. This needs to be done before you start training.
    for texts, pos, heads, deps in train_data:
        for dep in deps:
            parser.add_label(dep)
    # nlp.add_pipe(parser)
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'parser']
    with nlp.disable_pipes(*other_pipes):
        # optimizer = nlp.begin_training()
        optimizer = nlp.initialize()
        # optimizer = nlp.create_optimizer()
        for i in range(n_epochs):
            random.shuffle(train_data)
            # random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(64, 2, 1.001))
            for batch in batches:
                texts, pos, heads, deps = zip(*batch)
                examples = []
                try:
                    for j in range(len(texts)):
                #         doc = example_from_conllu_sentence(nlp.vocab, texts[j])
                #         # doc = Doc(words=texts[i], pos=pos[i], heads=heads[i], deps=deps[i])
                #         doc = Doc(words=texts[i], heads=heads[i], deps=deps[i])

                        doc = Doc(nlp.vocab, words=texts[j], pos=pos[j])
                #
                #         # for word in doc:
                #         #     print(word)
                #         example = Example.from_dict(doc, {"heads": doc.head_, "deps": doc.dep_})
                        example = Example.from_dict(doc, {"heads": heads[j], "deps": deps[j]})
                        examples.append(example)
                #         # nlp.update(texts, annotations, sgd=optimizer, losses=losses)
                    losses = parser.update(examples, sgd=optimizer, losses=losses)
                except Exception as error:
                    print(error)
            # for text, annotations in TRAIN_DATA:
            #     nlp.update([text], [annotations], sgd=optimizer, losses=losses)
            print(f'Loss in epoch {i} is {losses["parser"]}')
        print("Training is done!")


def main(args):
    model_name = None
    if args.model == "small":
        model_name = "en_core_web_sm"
    elif args.model == "large":
        model_name = "en_core_web_lg"
    elif args.model == "trf":
        model_name = "en_core_web_trf"

    if not model_name:
        print(f"The model {args.model} is not valid")
        return None

    try:
        pipeline(args.train_path, model_name, args.epochs)
    except Exception as error:
        print(error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aligner model')

    parser.add_argument('model', help='model name small/large/trf', type=str)
    parser.add_argument('-tp', '--train_path', help='path to the training data', type=str)
    parser.add_argument('-e', '--epochs', help='number of epochs', default=10, type=int)

    args = parser.parse_args()
    main(args)
