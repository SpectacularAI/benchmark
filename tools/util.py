import json

def readJsonl(filePath):
    with open(filePath) as f:
        for l in f: yield(json.loads(l))
