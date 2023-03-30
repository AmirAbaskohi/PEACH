import nltk
import json
import random
from os.path import exists
import re
from copy import copy
import argparse
from os.path import exists, isdir
import os
import de_core_news_sm
import fr_core_news_sm
import en_core_web_sm
import mk_core_news_sm
import numpy as np
from polyglot.text import Text

inputFile = None
pipeLine = None
dictionaries = {}
dictionaryExist = {}

outputDir = "./"

configFilePath = "config.json"

regxTokenizer = nltk.RegexpTokenizer(r"\w+")

config = {}

acceptableLanguages = ["en", "mk", "fr", "de"]
allLanguages = ["en", "fr", "de", "mk"]

inSentences = []
outSentences = []
batchNamedEntities = []

unwantedNamedEntities = ["the", "le", "la", "les", "der", "die", "das", "на", "овие", "ова", "оваа", "овој"]

def listDiff(li1, li2):
    return list(set(li1) - set(li2))

def readConfig():
    global config
    global pipeLine

    f = open(configFilePath)
    config = json.load(f)

    acceptableLanguages.remove(config["language"])
    
    if config["language"] not in allLanguages:
        print("Not supported language.")
        quit()
    f.close()

    if config["language"] == "en":
      pipeLine = en_core_web_sm.load()
    elif config["language"] == "fr":
      pipeLine = fr_core_news_sm.load()
    elif config["language"] == "de":
      pipeLine = de_core_news_sm.load()
    elif config["language"] == "mk":
      pipeLine = mk_core_news_sm.load()

def setInAndOutSentences():
    global inSentences
    global outSentences

    oneLanguageOut = []
    for i in range(config["number of lines"]):
        inSentences.append("")
        oneLanguageOut.append("")
    for i in range(len(acceptableLanguages)):
        outSentences.append(copy(oneLanguageOut))

def readDictionaries():
    for i in range(len(allLanguages)):
        for j in range(len(allLanguages)):
            if i == j:
                continue
            name = allLanguages[i] + "-" + allLanguages[j]
            reversedName = allLanguages[j] + "-" + allLanguages[i]
            path = config["dictionaries path"] + name + ".txt"
            reversedPath = config["dictionaries path"] + reversedName + ".txt"

            if not exists(path) and not exists(reversedPath):
                dictionaryExist[name] = False

            else:
                dictionaryExist[name] = True
                oneDictionary = {}
                if exists(path):
                    f  = open(path)
                    while True:
                        line = f.readline()

                        if not line:
                            break

                        line = line.lower()

                        line = line.replace('\n', '')
                        line = line.replace('\t', ' ')
                        line = re.split(' +', line)
                        if line[0] in oneDictionary:
                            oneDictionary[line[0]].append(line[1])
                        else:
                            oneDictionary[line[0]] = [line[1]]

                    f.close()
                if exists(reversedPath):
                    f  = open(reversedPath)
                    while True:
                        line = f.readline()

                        if not line:
                            break

                        line = line.lower()

                        line = line.replace('\n', '')
                        line = line.replace('\t', ' ')
                        line = re.split(' +', line)
                        if line[1] in oneDictionary:
                            if line[0] not in oneDictionary[line[1]]:
                                oneDictionary[line[1]].append(line[0])
                        else:
                            oneDictionary[line[1]] = [line[0]]
                    f.close()
                dictionaries[name] = oneDictionary


def readNLines():
    global inputFile
    global batchNamedEntities

    count = 0

    fullBatch = ""
    while count < config["number of lines"]:
        line = inputFile.readline()
        if not line:
            break

        if not line.strip():
            continue

        fullBatch += line + "\n"
        
        line = line.lower()
        inSentences[count] = nltk.word_tokenize(line)

        count += 1

    doc = pipeLine(fullBatch)
    batchNamedEntities = []
    for ent in doc.ents:
      text = ent.text
      text = text.lower()
      text = text.split(' ')
      batchNamedEntities.extend(text)
    batchNamedEntities = listDiff(batchNamedEntities, unwantedNamedEntities)

    return count

def writeOutput(numberOfLines):
    f = None
    if config["append first"]:
        f = open(outputDir + "input.txt", mode="a", encoding="utf-8")
    else:
        f = open(outputDir + "input.txt", mode="w", encoding="utf-8")
    for i in range(numberOfLines):
        f.write(" ".join(inSentences[i]) + "\n")
    f.close()

    for i in range(len(acceptableLanguages)):
        f = None
        if config["append first"]:
            f = open(outputDir + acceptableLanguages[i]+".txt", mode="a", encoding="utf-8")
        else:
            f = open(outputDir + acceptableLanguages[i]+".txt", mode="w", encoding="utf-8")
        for j in range(numberOfLines):
            f.write(outSentences[i][j] + "\n")
        f.close()
    
    config["append first"] = True

def translateUsingEnglish(word, fromLanguage, toLanguage):
    if word not in dictionaries[f"{fromLanguage}-en"]:
        return ""

    englishTranslations = random.sample((dictionaries[f"{fromLanguage}-en"][word]), len((dictionaries[f"{fromLanguage}-en"][word])))

    for englishTranslation in englishTranslations:
        if englishTranslation in dictionaries[f"en-{toLanguage}"]:
            return random.choice(dictionaries[f"en-{toLanguage}"][englishTranslation])

    return ""

def translateOneWord(word, fromLanguage, toLanguage):
    if dictionaryExist[f"{fromLanguage}-{toLanguage}"]:
        if word not in dictionaries[f"{fromLanguage}-{toLanguage}"]:
            if fromLanguage!="en" and toLanguage!="en":
                return translateUsingEnglish(word, fromLanguage, toLanguage)
            else:
                return ""

        return random.choice(dictionaries[f"{fromLanguage}-{toLanguage}"][word])
    else:
        return translateUsingEnglish(word, fromLanguage, toLanguage)
    

def translateOneLine(line):
    global batchNamedEntities

    result = []
    fromLanguage = config["language"]

    for i in range(len(acceptableLanguages)):
        result.append([])

    for word in line:
        for i in range(len(acceptableLanguages)):
            if word in batchNamedEntities:
                translated = translateOneWord(word, fromLanguage, acceptableLanguages[i])
                if translated == "":
                  text = Text(word, fromLanguage)
                  translitratedWord = list(text.transliterate(acceptableLanguages[i]))[0]
                  result[i].append(translitratedWord)
                else:
                  result[i].append(translated)
            elif (not word.isalnum()) or word.isnumeric():
                if '-' in word:
                    splittedWords = word.split('-')
                    translatedWords = []
                    for splittedWord in splittedWords:
                        if (not splittedWord.isalnum()) or splittedWord.isnumeric():
                            translatedWords.append(splittedWord)
                        else:
                            translated = translateOneWord(splittedWord, fromLanguage, acceptableLanguages[i])
                            if translated != "":
                              translatedWords.append(translated)
                    result[i].append("-".join(translatedWords))
                else:
                    result[i].append(word)
            else:
                translated = translateOneWord(word, fromLanguage, acceptableLanguages[i])
                if translated != "":
                  result[i].append(translated)


    for i in range(len(acceptableLanguages)):
        result[i] = " ".join(result[i])

    return result

def translate(numberOfLines):
    global inSentences
    global outSentences
    
    for i in range(numberOfLines):
        translations = translateOneLine(inSentences[i])
        for j in range(len(acceptableLanguages)):
            outSentences[j][i] = translations[j]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("input", help="Path to input file")
    parser.add_argument("output", help="Path to output directory")
    parser.add_argument("--verbose", help="Turn on output verbosity", action="store_true")
    args = parser.parse_args()

    verbose = args.verbose

    if verbose:
        print("Setting arguments ...")
    configFilePath = args.config
    readConfig()
    config["input file"] = args.input

    if not exists(config["input file"]):
        print("Input file does not exists")
        quit()
    if not isdir(args.output):
        os.mkdir(args.output)

    outputDir = args.output
    if outputDir[-1] != '\\' and outputDir[-1] != '/':
        outputDir += os.sep

    setInAndOutSentences()

    if verbose:
        print("Setting dictionaries ...")
    readDictionaries()

    inputFile = open(config["input file"], mode="r", encoding="utf-8")

    if verbose:
        print("Started preprocess ...")

    batchNumber = 1
    while True:
        numberOfLinesRead = readNLines()
        if numberOfLinesRead == 0:
            break

        translate(numberOfLinesRead)
        writeOutput(numberOfLinesRead)

        if verbose:
            print(f"Batch: {batchNumber} :: Preprocess of {numberOfLinesRead} completed.")
        batchNumber += 1

    if verbose:
        print("Preprocess completed.")

    inputFile.close()