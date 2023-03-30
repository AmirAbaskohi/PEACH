# How to use?
The Python script for denoising utilizes the following arguments:
Paths to the following files are required: 
* `config`: which will be further explained 
* `input`: which is the file we want to use as input
* `output`: which is the file we want to use as an output
* `verbose`: which is used to display some logs to show the script's status.

## Config file
Some parameters and hyperparameters for the runtime are included in configuration files.

* `remove std`: This is the standard deviation that is used to generate a random number distribution and remove words from it.
* `remove mean`: this is the mean that is used to generate the normal distribution for the purpose of removing words from random numbers.
* `percentage of replace`: here are the lower and upper boundaries for a random number generated using a uniform distribution for the proportion of a sentence's length that should be replaced with words.
* `percentage of add`: the lower and upper boundaries for the uniform distribution's random number generation for the proportion of a sentence's length that should be added in words are as follows.
* `number of lines`: batch size
* `append first`: when this parameter is set to true, the output file will open in append mode
* `language`: language of the input file

Config files which we used to run for different languages are available in the repository.

## Command to run
Use the following command to run the code:
```
python3 usage: main.py [-h] [--verbose] config input output

positional arguments:
  config      Path to config file
  input       Path to input file
  output      Path to output file

optional arguments:
  -h, --help  show this help message and exit
  --verbose   Turn on output verbosity
```

# How it works?
 This procedure consists of sentence shuffling, word deletion, word addition, and word substitution.
 
 The first step in generating pre-training data is loading a batch of pre-training documents into the memory as the current batch. We used a batch size of 1K for generating pre-training data for the denoising model. The batch size plays an essential role in this procedure because our algorithm selects candidates for replacing some words in a document from the words available in other documents in the current batch.
 
 After tokenizing the separating sentences using the NLTK toolkit, we shuffle the words in each sentence but keep the relative order of words in different sentences. It helps the denoising model learn the relative order of sentences, which is crucial since the word-by-word translation algorithm might face documents with multiple sentences. Therefore, this will teach the denoising model how to figure out the boundaries of different sentences.

Next, for each sentence, we select `m x c` words to be replaced, in which `m` is the length of the sentence and `c` is a random number from a uniform distribution between the reported rates in the paper. The algorithm selects `m x c` unique words from other documents in the current batch uniformly to be substituted with the selected words from the current document. The word addition objective works the same way as the substitution, but the algorithm does not replace any words. The word deletion objective works the same way, but it uses a normal distribution for generating the random number, and it just omits some words from each sentence without replacing them with any other words.

The word substitution and addition rates in the paper were selected based on a few observations of the outputs of the word-by-word translation algorithm. On the other hand, we computed the mean and standard deviation for the percent of words that the word-by-word translation algorithm could not find any translation for them on the pre-training corpus. The main purpose of the word deletion objective is to find a translation for the words that the word-by-word translation algorithm could not find any translation for them by considering the context of the sentence. Therefore, computing this number on the pre-training corpus that the final multilingual model is going to be trained on it will improve the ability of the denoising model in denoising the word-by-word translation algorithm's outputs, which decreases the number of words that the word-by-word translation algorithm or the denoising model could not find any translation for them.