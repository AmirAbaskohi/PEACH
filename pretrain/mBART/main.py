import nltk
import json
import numpy as np
import argparse
from os.path import exists
import random

inputFile = None

regxTokenizer = nltk.RegexpTokenizer(r"\w+")

config = {}
configFilePath = 'config.json'

inSentences = []
targetSentences = []
outSentences = []

languageMap = { 
    "en": "english",
    "fr": "french",
    "de": "german"
}

def readConfig():
    global config
    f = open(configFilePath)
    config = json.load(f)
    f.close()

def setInAndOutSentences():
    for i in range(config["number of lines"]):
        inSentences.append([])
        outSentences.append("")
        targetSentences.append("")

def readNLines():
    global inputFile
    global languageMap

    count = 0

    while count < config["number of lines"]:
        line = inputFile.readline()
        if not line:
            break

        if not line.strip():
            continue

        sentences = nltk.tokenize.sent_tokenize(line, language=languageMap[config["input language"]])
        random.shuffle(sentences)
        inSentences[count] = nltk.word_tokenize(" ".join(sentences))
        targetSentences[count] = " ".join(nltk.word_tokenize(line))

        count += 1
    return count

def writeOutput(numberOfLines):
    global targetSentences

    f = None
    t = None
    if config["append first"]:
        f = open(config["output file"], mode="a", encoding="utf-8")
        t = open(config["target file"], mode="a", encoding="utf-8")
    else:
        f = open(config["output file"], mode="w", encoding="utf-8")
        t = open(config["target file"], mode="w", encoding="utf-8")
        config["append first"] = True

    for i in range(numberOfLines):
        f.write(outSentences[i] + "\n")
        t.write(targetSentences[i] + "\n")
    f.close()
    t.close()

def getSpanLength():
    global config

    return np.random.poisson(lam=config["lambda"])

def randomTokenOneLine(line):
    maskedSize = round(config["random token percentage"] * len(line))

    while maskedSize > 0:
        spanSize = getSpanLength()
        spanSize = min(maskedSize, spanSize)
        start = random.randint(0, len(line)-spanSize)

        maskedUsed = 0
        for i in range(start, start+spanSize):
            if not (len(line[i]) > 2 and line[i][0] == '<' and line[i][-1] == '>' and line[i][1:-1].isdigit()):
                maskedUsed += 1

        del line[start:start+spanSize]
        line.insert(start, "<1>")

        maskedSize -= maskedUsed

    return " ".join(line)

def preprocess(numberOfLines):
    global inSentences
    global outSentences
    global config

    for i in range(numberOfLines):
        outSentences[i] = f'<{config["input language"]}><{config["input language"]}> ' + randomTokenOneLine(inSentences[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("input", help="Path to input file")
    parser.add_argument("output", help="Path to output file")
    parser.add_argument("target", help="Path to target file")
    parser.add_argument("--verbose", help="Turn on output verbosity", action="store_true")
    args = parser.parse_args()

    verbose = args.verbose
    configFilePath = args.config

    if verbose:
        print("Setting arguments ...")
    
    readConfig()
    config["input file"] = args.input
    config["output file"] = args.output
    config["target file"] = args.target

    if not exists(config["input file"]):
        print("Input file does not exists")
        quit()

    setInAndOutSentences()

    inputFile = open(config["input file"], mode="r", encoding="utf-8")

    if verbose:
        print("Started preprocess ...")
    while True:
        numberOfLinesRead = readNLines()
        if numberOfLinesRead == 0:
            break

        preprocess(numberOfLinesRead)
        writeOutput(numberOfLinesRead)

        if verbose:
            print(f"Preprocess of {numberOfLinesRead} completed.")

    if verbose:
        print("Preprocess completed.")

    inputFile.close()
