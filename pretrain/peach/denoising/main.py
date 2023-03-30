import nltk
import json
import random
import numpy as np
from numpy.core.defchararray import islower as npislower
import argparse
from os.path import exists

inputFile = None
words = None

regxTokenizer = nltk.RegexpTokenizer(r"\w+")

config = {}
languageMap = { 
    "en": "english",
    "fr": "french",
    "de": "german",
    "mk": "macedonian"
}

configFilePath = 'config.json'

inSentences = []
outSentences = []

def listDiff(li1, li2):
    return list(set(li1) - set(li2))

def readConfig():
    global config
    f = open(configFilePath)
    config = json.load(f)
    
    if config["language"] in languageMap.keys():
        config["language"] = languageMap[config["language"]]
    else:
        print("Not supported language.")
        quit()
    
    if "-" not in config["percentage of replace"] or "-" not in config["percentage of add"]:
        print("Invalid format for percentages")
        quit()

    config["floor replace"] = int(config["percentage of replace"].split("-")[0])
    config["ceil replace"] = int(config["percentage of replace"].split("-")[1])

    config["floor add"] = int(config["percentage of add"].split("-")[0])
    config["ceil add"] = int(config["percentage of add"].split("-")[1])

    if config["floor replace"] > config["ceil replace"] or config["floor add"] > config["ceil add"]:
        print("Floor should not be greater the ceil")
        quit()

    f.close()

    del config["percentage of add"]
    del config["percentage of replace"]

def setInAndOutSentences():
    for i in range(config["number of lines"]):
        inSentences.append([])
        outSentences.append("")

def readNLines():
    global inputFile
    global words

    count = 0

    words = []

    while count < config["number of lines"]:
        line = inputFile.readline()
        if not line:
            break

        inSentences[count] = nltk.sent_tokenize(line)
        for i in range(len(inSentences[count])):
            inSentences[count][i] = nltk.word_tokenize(inSentences[count][i])
            tokenizedNumpyArray = np.array(inSentences[count][i])
            mask = npislower(np.array(list(map(lambda x: x.lower(), tokenizedNumpyArray))))
            words.extend(tokenizedNumpyArray[mask])

        count += 1
    words = list(set(words))
    return count

def writeOutput(numberOfLines):
    f = None
    if config["append first"]:
        f = open(config["output file"], mode="a", encoding="utf-8")
    else:
        f = open(config["output file"], mode="w", encoding="utf-8")
        config["append first"] = True

    for i in range(numberOfLines):
        f.write(outSentences[i] + "\n")
    f.close()


def shuffleSentence(sentence):
    punctuations = []
    punctuationIndexes = []
    for i in range(len(sentence)-1, -1, -1):
        word = sentence[i]
        lowered_word = word.lower()
        if not (lowered_word.isupper() or lowered_word.islower()):
            punctuations.append((i, word))
            punctuationIndexes.append(i)
            del sentence[i]

    random.shuffle(sentence)

    for i in range(len(punctuations)-1, -1, -1):
        punctuation = punctuations[i]
        sentence.insert(punctuation[0], punctuation[1])

    return punctuationIndexes

def replaceWords(sentence, punctuationIndexes):
    wantedIndexes = listDiff(range(len(sentence)), punctuationIndexes)
    random.shuffle(wantedIndexes)
    numberOfReplace = round(len(wantedIndexes) * (random.randint(config["floor replace"], config["ceil replace"])/100))
    wantedIndexes = wantedIndexes[:numberOfReplace+1]
    selectedWords = random.sample(range(len(words)), len(sentence)+numberOfReplace)
    selectedWords = list(map(lambda x:words[x], selectedWords))
    selectedWords = listDiff(selectedWords, sentence)

    for i in range(numberOfReplace):
        sentence[wantedIndexes[i]] = selectedWords[i]

    return wantedIndexes

def removeWords(sentence, punctuationIndexes, replacedIndexes):
    removePercentage = abs(np.random.normal(config["remove mean"], config["remove std"], 1)[0])
    if removePercentage == 1:
        removePercentage -= 0.001
    if removePercentage > 1:
        removePercentage -= int(removePercentage)

    wantedIndexes = listDiff(range(len(sentence)), punctuationIndexes + replacedIndexes)
    random.shuffle(wantedIndexes)
    numberOfRemoves = round(len(wantedIndexes) * removePercentage)
    wantedIndexes = wantedIndexes[:numberOfRemoves+1]
    wantedIndexes.sort()

    for i in range(len(wantedIndexes)-1, -1, -1):
        del sentence[wantedIndexes[i]]

def addWords(sentence, punctuationIndexes):
    wantedIndexes = listDiff(range(len(sentence)), punctuationIndexes)
    numberOfAdd = round(len(wantedIndexes) * (random.randint(config["floor add"], config["ceil add"])/100))
    selectedWords = random.sample(range(len(words)), numberOfAdd)
    selectedWords = list(map(lambda x:words[x], selectedWords))

    for i in range(numberOfAdd):
        idx = random.randint(0, len(sentence)-1)
        sentence.insert(idx, selectedWords[i])

    return wantedIndexes

def preprocessOneLine(line):
    result = []
    for sentence in line:
        punctuationIndexes = shuffleSentence(sentence)
        replacedIndexes = replaceWords(sentence, punctuationIndexes)
        removeWords(sentence, punctuationIndexes, replacedIndexes)
        addWords(sentence, punctuationIndexes)
        result.append(" ".join(sentence))
    return " ".join(result)

def preprocess(numberOfLines):
    global inSentences
    global outSentences

    for i in range(numberOfLines):
        outSentences[i] = preprocessOneLine(inSentences[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("input", help="Path to input file")
    parser.add_argument("output", help="Path to output file")
    parser.add_argument("--verbose", help="Turn on output verbosity", action="store_true")
    args = parser.parse_args()

    verbose = args.verbose
    configFilePath = args.config

    if verbose:
        print("Setting arguments ...")
    
    readConfig()
    config["input file"] = args.input
    config["output file"] = args.output

    if not exists(config["input file"]):
        print("Input file does not exists")
        quit()

    setInAndOutSentences()

    inputFile = open(config["input file"], mode="r", encoding="utf-8")

    if verbose:
        print("Started preprocess ...")

    batchNumber = 1
    while True:
        numberOfLinesRead = readNLines()
        if numberOfLinesRead == 0:
            break

        preprocess(numberOfLinesRead)
        writeOutput(numberOfLinesRead)

        if verbose:
            print(f"Batch: {batchNumber} :: Preprocess of {numberOfLinesRead} completed.")

        batchNumber += 1

    if verbose:
        print("Preprocess completed.")

    inputFile.close()
