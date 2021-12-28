import os
import re

GENDER_RE = r"Gender=([A-Za-z,]*)"
NUMBER_RE = r"Number=([A-Za-z,]*)"
PERSON_RE = r"Person=([1-3,]*)"


def get_info(splitted_line):
    id_token = splitted_line[0]
    form = splitted_line[1]
    pos = splitted_line[3]
    head_id = int(splitted_line[6])
    deprel = splitted_line[7]
    features = splitted_line[5]
    gender = "_"
    number = "_"
    person = "_"
    if len(form) > 1 and form != "__":
        form = form.replace("_", "")
    if features != "_":
        re_gender = re.search(GENDER_RE, features)
        if re_gender:
            gender = re_gender.group(1)

        re_number = re.search(NUMBER_RE, features)
        if re_number:
            number = re_number.group(1)

        re_person = re.search(PERSON_RE, features)
        if re_person:
            person = re_person.group(1)

    return id_token, form, pos, deprel, head_id, gender, number, person


class ConllReader:
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

    def __init__(self, data_set):
        self.data_path = os.path.join(self.BASE_PATH, f"he_htb-ud-{data_set}.conllu")
        self.sentences, self.poses, self.deps, self.heads, self.genders, self.numbers, self.persons = self.get_data(self.data_path)

    def get_examples(self):
        examples = []
        for i in range(len(self.sentences)):
            examples.append((self.sentences[i], self.poses[i], self.heads[i], self.deps[i]))
        return examples

    def get_data(self, file_path):
        sentences = []
        poses = []
        deps = []
        heads = []
        genders = []
        numbers = []
        persons = []

        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()
            curr_sentence = []
            curr_pos = []
            curr_deps = []
            curr_head = []
            curr_genders = []
            curr_numbers = []
            curr_persons = []
            for line in lines:

                if line == "\n":
                    sentences.append(curr_sentence)
                    poses.append(curr_pos)
                    deps.append(curr_deps)
                    genders.append(curr_genders)
                    numbers.append(curr_numbers)
                    persons.append(curr_persons)
                    heads.append(curr_head)
                    curr_sentence = []
                    curr_pos = []
                    curr_deps = []
                    curr_head = []
                    curr_genders = []
                    curr_numbers = []
                    curr_persons = []
                    continue

                if "sent_id = 2035" in line:
                    print()

                if line.startswith("#"):
                    continue

                splitted_line = line.split("\t")

                # if there is - so the token contains multiple tokens (will be added separately)
                if "-" not in splitted_line[0]:
                    id_token, form, pos, deprel, head_id, gender, number, person = get_info(splitted_line)
                    # in spacy there is no root, instead the head is the word itself
                    if head_id == 0:
                        head_id = int(id_token)
                    head_id -= 1
                    curr_sentence.append(form)
                    curr_pos.append(pos)
                    curr_deps.append(deprel)
                    # curr_head.append(head_id - int(id_token))
                    curr_head.append(head_id)
                    curr_genders.append(gender)
                    curr_numbers.append(number)
                    curr_persons.append(person)

        return sentences, poses, deps, heads, genders, numbers, persons
