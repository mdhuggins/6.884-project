import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from components.assertion import Assertion
from tqdm import tqdm


def text_to_assertions(input_file, n=-1, general=False,concept_filter=[]):
    assertions = []
    with open(input_file) as f:
        lines = f.readlines()
        if n == -1:
            n = len(lines)
        print("Loading at most,",n)
        for line in tqdm(lines[0:n+1]):
            row = line.strip().split('\t')
            if len(concept_filter) > 0:
                stop = True
                for concept in concept_filter:
                    if concept in row[1] or concept in row[2]:
                        stop = False
                        break
                if stop:
                    continue
            assertions.append(Assertion(subject=row[1], object=row[2], relation=row[0], general=general))
    return assertions
