# How to use?
The Python script for word-by-word translation:
Paths to the following files are required: 
* `config`: which will be further explained 
* `input`: which is the file we want to use as input
* `output`: which is the directory we want to use to put output files in
* `verbose`: which is used to display some logs to show the script's status.

## Config file
Some parameters and hyperparameters for the runtime are included in configuration files.

* `number of lines`: batch size
* `append first`: when this parameter is set to true, the output file will open in append mode
* `language`: language of the input file
* `dictoinaries path`: path to dictionary files

Config files which we used to run for different languages are available in the repository.

You can use your own dictionary as well, but remember to put them in the following format:
```
Name: fromLaguageAbbr-toLanguageAbbr

word1InFromLanguage    Translation1InToLanguage
word2InFromLanguage    Translation2InToLanguage
word3InFromLanguage    Translation3InToLanguage
word4InFromLanguage    Translation4InToLanguage
...
```

## Command to run
Use the following command to run the code:
```
python3 usage: main.py [-h] [--verbose] config input output

positional arguments:
  config      Path to config file
  input       Path to input file
  output      Path to output directory

optional arguments:
  -h, --help  show this help message and exit
  --verbose   Turn on output verbosity
```

# How it works?
 
The word-by-word translation was performed in batches of 1K documents. The batch size does not affect the algorithm's performance and should be chosen based on the available resources. 

After lower casing the documents in a batch, name entities are extracted using the spaCy toolkit. The identified entities should be divided by white space characters since the named entities sometimes consist of multiple words. Since the spaCy toolkit for named entity recognition sometimes chooses definite articles as a part of named entities, we filter out definite articles such as "the," "le," "la," "les," "der," "die," and "das" and translate them using dictionaries in following steps.

In order to perform word-by-word translation, we first tokenize the document. We search for the translation of each token from the source language to the destination language using the appropriate dictionary. If we found more than one possible translation for a token, we uniformly select one of them. Suppose we can not find any translations for a token in the source to the destination language dictionary. In that case, we use source to English and English to destination dictionary to find a translation for the mentioned token. First, we search for a translation from the source language to English using the source to English dictionary. Next, we search for a translation from English to the destination language in the English to the destination dictionary. It does not need to be mentioned that this technique is just helpful when there is a translation from the source token to English. If we can not find any translations for a token, we mark it as unknown to decide about it later.

For the terms that were marked as unknown, if the token contains numbers or punctuations, we transfer it without any change to the output as a translated word. Otherwise, we check if the word is in the extracted name entities. In this case, we transliterate the word into the destination language using polyglot library and put it in the output as a translated word. For complex words such as "high-end," we break the word into its alphabetical components and search them in the dictionary. If we could find a translation for all components, we would translate each component and concatenate them using the proper separator character. In the case that none of the aforementioned scenarios happens, we omit the word and hope the denoising pre-trained model can find a proper translation for it.
