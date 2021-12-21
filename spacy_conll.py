import argparse

from spacy.tokens import Doc, Token

NOUN_AND_PROP = {"PROPN", "NOUN", "PRON"}

VERB_ADJ = {'VERB', 'AUX'}

LOC_ADVERB = {
    "פה",
    "כאן",
    "שם",
    "למטה",
    "קדימה",
    "אחורה",
    "בצד",
    "מזרח",
    "מערב",
    "צפון",
    "דרום"
}

TIME_ADVERB = {
    "אתמול",
    "שלשום",
    "אמש",
    "מחר",
    "אז",
    "פעם",
    "השנה",
    "שנה",
    "החודש",
    "חודש",
    "היום",
    "יום"

}

LOC_AND_TIME_ADVERB = LOC_ADVERB.union(TIME_ADVERB)


def merge_conllu_subtokens(lines, doc):
    # identify and process all subtoken spans to prepare attrs for merging
    subtok_spans = []
    for line in lines:
        parts = line.split("\t")
        id_, word, lemma, pos, tag, morph, head, dep, _1, misc = parts
        if "-" in id_:
            subtok_start, subtok_end = id_.split("-")
            subtok_span = doc[int(subtok_start) - 1: int(subtok_end)]
            subtok_spans.append(subtok_span)
            # create merged tag, morph, and lemma values
            tags = []
            morphs = {}
            lemmas = []
            for token in subtok_span:
                tags.append(token.tag_)
                lemmas.append(token.lemma_)
                if token._.merged_morph:
                    for feature in token._.merged_morph.split("|"):
                        field, values = feature.split("=", 1)
                        if field not in morphs:
                            morphs[field] = set()
                        for value in values.split(","):
                            morphs[field].add(value)
            # create merged features for each morph field
            for field, values in morphs.items():
                morphs[field] = field + "=" + ",".join(sorted(values))
            # set the same attrs on all subtok tokens so that whatever head the
            # retokenizer chooses, the final attrs are available on that token
            for token in subtok_span:
                token._.merged_orth = token.orth_
                token._.merged_lemma = " ".join(lemmas)
                token.tag_ = "_".join(tags)
                token._.merged_morph = "|".join(sorted(morphs.values()))
                token._.merged_spaceafter = (True if subtok_span[-1].whitespace_ else False)

    with doc.retokenize() as retokenizer:
        for span in subtok_spans:
            retokenizer.merge(span)

    return doc


def example_from_conllu_sentence(vocab, lines, merge_subtoken=False, ):
    """Create an Example from the lines for one CoNLL-U sentence, merging
    subtokens and appending morphology to tags if required.
    lines (str): The non-comment lines for a CoNLL-U sentence
    ner_tag_pattern (str): The regex pattern for matching NER in MISC col
    RETURNS (Example): An example containing the annotation
    """
    # create a Doc with each subtoken as its own token
    # if merging subtokens, each subtoken orth is the merged subtoken form
    if not Token.has_extension("merged_orth"):
        Token.set_extension("merged_orth", default="")
    if not Token.has_extension("merged_lemma"):
        Token.set_extension("merged_lemma", default="")
    if not Token.has_extension("merged_morph"):
        Token.set_extension("merged_morph", default="")
    if not Token.has_extension("merged_spaceafter"):
        Token.set_extension("merged_spaceafter", default="")
    words, spaces, tags, poses, morphs, lemmas = [], [], [], [], [], []
    heads, deps = [], []
    subtok_word = ""
    in_subtok = False
    for i in range(len(lines)):
        line = lines[i]
        parts = line.split("\t")
        id_, word, lemma, pos, tag, morph, head, dep, _1, misc = parts
        if "." in id_:
            continue
        if "-" in id_:
            in_subtok = True
        if "-" in id_:
            in_subtok = True
            subtok_word = word
            subtok_start, subtok_end = id_.split("-")
            subtok_spaceafter = "SpaceAfter=No" not in misc
            continue
        if merge_subtoken and in_subtok:
            words.append(subtok_word)
        else:
            words.append(word)
        if in_subtok:
            if id_ == subtok_end:
                spaces.append(subtok_spaceafter)
            else:
                spaces.append(False)
        elif "SpaceAfter=No" in misc:
            spaces.append(False)
        else:
            spaces.append(True)
        if in_subtok and id_ == subtok_end:
            subtok_word = ""
            in_subtok = False
        id_ = int(id_) - 1
        head = (int(head) - 1) if head not in ("0", "_") else id_
        tag = pos if tag == "_" else tag
        morph = morph if morph != "_" else ""
        dep = "ROOT" if dep == "root" else dep
        lemmas.append(lemma)
        poses.append(pos)
        tags.append(tag)
        morphs.append(morph)
        heads.append(head)
        deps.append(dep)

    doc = Doc(vocab, words=words, spaces=spaces)
    try:
        for i in range(len(doc)):
            doc[i].tag_ = tags[i]
            doc[i].pos_ = poses[i]
            doc[i].dep_ = deps[i]
            doc[i].lemma_ = lemmas[i]
            doc[i].head = doc[heads[i]]
            doc[i]._.merged_orth = words[i]
            doc[i]._.merged_morph = morphs[i]
            doc[i]._.merged_lemma = lemmas[i]
            doc[i]._.merged_spaceafter = spaces[i]
        if merge_subtoken:
            doc = merge_conllu_subtokens(lines, doc)
    except Exception as e:
        print(f"Cant process {doc}")
        print(e)
    return doc


def read_conllx(input_data, vocab, merge_subtoken=False, ner_tag_pattern="", ner_map=None,

                ):
    """ Yield examples, one for each sentence """
    for sent in input_data.strip().split("\n\n"):
        lines = sent.strip().split("\n")
        if lines:
            while lines[0].startswith("#"):
                lines.pop(0)
            # Dummy vocab
            example = example_from_conllu_sentence(vocab, lines, merge_subtoken=merge_subtoken, )
            yield example


left_labels = ["det", "fixed", "nmod:poss", "amod", "nummod", "appos", "compound:smixut", "flat:name"]
right_labels = ["fixed", "nmod:poss", "amod", "nummod", "appos", "compound:smixut", "flat:name"]
stop_labels = ["punct"]

np_label = "NP"


def is_time_and_location_adv(token):
    return token.pos_ == "ADV" and token.lemma_ in LOC_AND_TIME_ADVERB


def get_noun_chunks(spacy_doc, bio=True, nested=False):
    def is_verb_adj_token(tok):
        return tok.pos_ in VERB_ADJ

    def get_left_bound(root):
        left_bound = root
        for tok in reversed(list(root.lefts)):
            if tok.dep_ in left_labels:
                left_bound = tok
        return left_bound

    def get_right_bound(doc, root):
        right_bound = root
        for tok in root.rights:
            if tok.dep_ in right_labels or (tok.dep_ == "det" and tok.pos_ == "PRON"):
                right = get_right_bound(doc, tok)
                if list(filter(lambda t: is_verb_adj_token(t) or (t.dep_ in stop_labels and t.text != "-"),
                               doc[root.i: right.i], )):
                    break
                else:
                    right_bound = right
        return right_bound

    def get_bounds(doc, root):
        return get_left_bound(root), get_right_bound(doc, root)

    chunks = []
    for token in spacy_doc:
        if token.pos_ in NOUN_AND_PROP or is_time_and_location_adv(token):  # כך, כן ?? ||| יתר??
            left, right = get_bounds(spacy_doc, token)
            chunks.append((left.i, right.i + 1, np_label))

    is_chunk = [True for _ in chunks]
    if not nested:
        # remove nested chunks
        remove_nested(chunks, is_chunk)

    final_chunks = [c for c, ischk in zip(chunks, is_chunk) if ischk]
    return _chunks2bio(final_chunks, len(spacy_doc)) if bio else final_chunks


def remove_nested(chunks, is_chunk):
    for i, i_chunk in enumerate(chunks[:-1]):
        i_left, i_right, _ = i_chunk
        for j, j_chunk in enumerate(chunks[i + 1:], start=i + 1):
            j_left, j_right, _ = j_chunk
            if j_left <= i_left < i_right <= j_right:
                is_chunk[i] = False
            if i_left <= j_left < j_right <= i_right:
                is_chunk[j] = False


def _chunks2bio(chunks, sent_len):
    bio_tags = ['O'] * sent_len
    for (start, end, label) in chunks:
        bio_tags[start] = 'B-' + label
        for j in range(start + 1, end):
            bio_tags[j] = 'I-' + label
    return bio_tags


def parse_arguments():
    p = argparse.ArgumentParser(description='Np chunker flow')
    p.add_argument('input', help="input file expect UD conll file")
    p.add_argument('output', help="output file")
    return p.parse_args()

#
# def main():
#     args = parse_arguments()
#
#     dummy_vocab = spacy_udpipe.load("he").vocab
#     with open(args.input, encoding='utf-8') as f:
#         lines = "".join(f.readlines())
#         doc = read_conllx(lines, dummy_vocab, merge_subtoken=False)
#
#     with open(args.output, "w", encoding='utf-8') as f:
#         for e in doc:
#             for w, t in zip(e, get_noun_chunks(e, nested=False)):
#                 f.write(f"{w}\t{t}\n")
#             f.write("\n")
#
#
# if __name__ == '__main__':
#     main()