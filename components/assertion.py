import hashlib
from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Assertion:
    story: str = ""
    sentence: str = ""
    subject: str = ""
    object: str = ""
    relation: str = ""
    general: bool = False
    additional: str = ""

    def __str__(self):
        if self.general:
            preface="Generally,"
        else:
            preface="Specifically,"

        str_rep = " ".join([preface,self.subject,self.relation,self.object+"."])
        return str_rep

    def __hash__(self):
        a = str(self)
        b = hashlib.md5(a.encode('utf-8')).hexdigest()
        return int(b, 16)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()