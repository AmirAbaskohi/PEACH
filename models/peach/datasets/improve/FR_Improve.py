
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds
import re

_DOCUMENT = "_DOCUMENT"
_OUTPUT = "_OUTPUT"



class FrImprove(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version('0.1.0')
    MANUAL_DOWNLOAD_INSTRUCTIONS = 'Here to be description of the dataset'
    SKIP_REGISTERING = True
    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            # This is the description that will appear on the datasets page.
            description=("dataset."),
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            _DOCUMENT: tfds.features.Text(),
            _OUTPUT: tfds.features.Text(),
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=(_DOCUMENT, _OUTPUT),
        )

    def _split_generators(self, dl_manager):
        # Download source data
        
        test_inputs = ['/*Address to outputs of word-by-word translation with the name fr.txt*/']
        test_outputs = ['/*Address to outputs of word-by-word translation with the name input.txt corresponding to fr.txt*/']
        test_lang_tokens = ['/*<mk><fr> or <en><fr> or <de><fr> based on the input language of word-by-word tranlation fr.txt output */']

        # Specify the splits
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "raw_texts": test_inputs,
                    "target_texts": test_outputs,
                    "lang_tokens" :test_lang_tokens
                },
            )
        ]
            
    def _generate_examples(self, raw_texts,target_texts,lang_tokens):
        count = 0
        for raw_text,target_text,token_start in zip(raw_texts,target_texts,lang_tokens):
            with tf.io.gfile.GFile(raw_text,"rb") as input_file,tf.io.gfile.GFile(target_text,"rb") as target_file:
                temp_count = count
                for i, (text, target) in enumerate(zip(input_file, [line for line in target_file.readlines() if line.strip()])):
                    target = (token_start+target.decode("utf-8")).encode()
                    yield temp_count + i, {_DOCUMENT: text, _OUTPUT: target}
                    count += 1