import nltk
import json
import numpy as np
import argparse
from os.path import exists
import random
import re

inputFile = None

regxTokenizer = nltk.RegexpTokenizer(r"\w+")

config = {}
configFilePath = 'config.json'

inSentences = []
outSentences = []
targetSentences = []

pattern = '<[0-9]+>'

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

    count = 0

    while count < config["number of lines"]:
        line = inputFile.readline()
        if not line:
            break

        inSentences[count] = nltk.word_tokenize(line)

        count += 1
    return count

def writeOutput(numberOfLines):
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
    lengths = range(1, config["max span length"]+1)
    prb = np.asarray([config["geometric p"]*(1.0-config["geometric p"])**k for k in range(0, len(lengths))])
    prb /= np.sum(prb)

    return np.random.choice(lengths, size=1, replace = True, p=prb)[0]

def randomTokenOneLine(line):
    maskedSize = round(config["random token percentage"] * len(line))

    tokenId = 1

    target = []
    while maskedSize > 0:
        spanSize = getSpanLength()
        spanSize = min(spanSize, maskedSize)
        start = random.randint(0, len(line)-spanSize)

        t = []
        maskedUsed = 0
        for i in range(start, start+spanSize):
            if len(line[i]) > 2 and line[i][0] == '<' and line[i][-1] == '>' and line[i][1:-1].isdigit():
                t.append(target[int(line[i][1:-1])-1])
            else:
                t.append(line[i])
                maskedUsed += 1
        target.append(" ".join(t))

        del line[start:start+spanSize]
        line.insert(start, f"<{tokenId}>")

        maskedSize -= maskedUsed
        tokenId += 1

    resultTarget = ""
    out = " ".join(line)
    ind = [(m.start(0), m.end(0)) for m in re.finditer(pattern, out)]
    for mask in ind:
        start, end = mask
        targetIndex = int(out[start:end][1:-1])
        resultTarget += f" <{targetIndex}> "
        resultTarget += target[targetIndex-1]
    resultTarget += " <z> "
    
    resultTarget = " ".join(resultTarget.split())

    return out, resultTarget

def orderMaskTokens(masked):
    i = 1
    result = ""
    ind = [(m.start(0), m.end(0)) for m in re.finditer(pattern, masked)]
    lastStart = 0
    for mask in ind:
        start, end = mask
        result += masked[lastStart:start]
        result += f"<{i}>"
        i+=1
        lastStart = end
    result += masked[lastStart:]
    return result

def preprocess(numberOfLines):
    global inSentences
    global outSentences
    global targetSentences

    for i in range(numberOfLines):
        out, target = randomTokenOneLine(inSentences[i])
        outSentences[i], targetSentences[i] = orderMaskTokens(out), orderMaskTokens(target)


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
