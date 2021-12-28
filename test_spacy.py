from __future__ import unicode_literals, print_function
import random
from spacy.util import minibatch, compounding
import argparse
from spacy.tokens import Doc
from spacy.training import Example
from data_reader import ConllReader
from spacy.vocab import Vocab
from spacy.language import Language

nlp = Language(Vocab())


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


def evaluate_model(parsing_model):
    dev_data = ConllReader("dev").get_examples()
    unlabeled = 0
    all_words = 0
    labeled = 0

    for text, pos, heads, deps in dev_data:
        all_words += len(text)

        analyzed_sent = parsing_model(Doc(nlp.vocab, words=text, pos=pos))
        for i, word in enumerate(analyzed_sent):
            if word.head.i == heads[i]:
                unlabeled += 1
                if word.dep_ == deps[i]:
                    labeled += 1
    print(f"unlabeled: {round(unlabeled*100/all_words, 2)}%")
    print(f"labeled: {round(labeled*100/all_words, 2)}%")


def pipeline(model_path, n_epochs, batch_size):
    train_data = ConllReader("train").get_examples()
    parser = nlp.add_pipe("parser", first=True)
    # parser.initialize(lambda: [], nlp=nlp)

    # Add the tags. This needs to be done before you start training.
    for texts, pos, heads, deps in train_data:
        for dep in deps:
            parser.add_label(dep)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'parser']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.initialize()
        # optimizer = nlp.begin_training()
        # optimizer = nlp.create_optimizer()

        print("Start training!")
        for i in range(n_epochs):
            random.shuffle(train_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(batch_size, 2, 1.001))
            for batch in batches:
                texts, pos, heads, deps = zip(*batch)
                examples = []
                try:
                    for j in range(len(texts)):
                        doc = Doc(nlp.vocab, words=texts[j], pos=pos[j])
                        example = Example.from_dict(doc, {"heads": heads[j], "deps": deps[j]})
                        examples.append(example)
                    losses = parser.update(examples, sgd=optimizer, losses=losses)
                except Exception as error:
                    print(error)
            print(f'Loss in epoch {i+1} is {losses["parser"]}')
            evaluate_model(parser)
        print("Training is done!")
        parser.to_disk(model_path)
        print(f"Parser saved to {model_path}")


def main(args):
    try:
        pipeline(args.model_path_to_save, args.epochs, args.batch_size)
        # parser = nlp.add_pipe("parser")
        # parser.from_disk("models/trained_parser")
        # evaluate_model(parser)
    except Exception as error:
        print(error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aligner model')

    parser.add_argument('model_path_to_save', help='path to save the trained model', type=str)
    parser.add_argument('-tp', '--train_path', help='path to the training data', type=str)
    parser.add_argument('-e', '--epochs', help='number of epochs', default=10, type=int)
    parser.add_argument('-b', '--batch_size', help='batch size', default=64, type=int)

    args = parser.parse_args()
    main(args)
