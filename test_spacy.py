# import spacy
from spacy.cli.train import train

# nlp = spacy.load("en_core_web_sm")
train("base_config.cfg", overrides={"paths.train": "spacy_data/he_htb-ud-train.json", "paths.dev": "spacy_data/he_htb-ud-dev.json"})

