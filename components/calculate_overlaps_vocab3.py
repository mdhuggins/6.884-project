import sys

import torch

sys.path.insert(0, '../')

if __name__ == '__main__':
    print("Overlap for 3")
    def dedupe(input_Dict):
        for key1 in input_Dict:
            for key2 in input_Dict:
                if key1 == key2:
                    continue
                elif key1 in key2:
                    input_Dict[key1]-= input_Dict[key2]
        return input_Dict
    with open("100k overlap") as f:
        d = eval(f.readlines()[1])
        tot = sum(dedupe(d).values())
        print(tot)
    with open("300k overlap") as f:
        d = eval(f.readlines()[1])
        tot = sum(dedupe(d).values())
        print(tot)
    with open("600k overlap") as f:
        d = eval(f.readlines()[1])
        tot = sum(dedupe(d).values())
        print(tot)
    with open("nb overlap") as f:
        d = eval(f.readlines()[1])
        tot = sum(dedupe(d).values())
        print(tot)