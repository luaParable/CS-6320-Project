import spacy
import json
import re

nlp = spacy.blank("en")
patterns = [{"label":"CALLSIGN","pattern":[{"TEXT":{"REGEX":"[A-Z]{2,3}\\d{1,4}"}}]}]
nlp.add_pipe("entity_ruler").add_patterns(patterns)
def annotate(text): return [(ent.text, ent.label_) for ent in nlp(text).ents]