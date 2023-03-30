
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds
import re

_DOCUMENT = "_DOCUMENT"
_OUTPUT = "_OUTPUT"



class WMT14ENFR(tfds.core.GeneratorBasedBuilder):

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
        
        train_inputs = ["/*Address to the inputs of train set of the dataset*/"]
        train_outputs =["/*Address to the outputs of train set of the dataset*/"]
        
        val_inputs = ["/*Address to the inputs of validation set of the dataset*/"]
        val_outputs =["/*Address to the outputs of validation set of the dataset*/"]
        
        test_inputs = ["/*Address to the inputs of test set of the dataset*/"]
        test_outputs =["/*Address to the outputs of test set of the dataset*/"]
        
        # Specify the splits
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "raw_texts": train_inputs,
                    "target_texts": train_outputs
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    "raw_texts": val_inputs,
                    "target_texts": val_outputs
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "raw_texts": test_inputs,
                    "target_texts": test_outputs
                },
            )
        ]
            
    def _generate_examples(self, raw_texts,target_texts):
        count = 0
        for raw_text,target_text in zip(raw_texts,target_texts):
            with tf.io.gfile.GFile(raw_text,"rb") as input_file,tf.io.gfile.GFile(target_text,"rb") as target_file:
                temp_count = count
                for i, (text, target) in enumerate(zip(input_file, target_file)):
                    text = ("<en><fr> "+text.decode("utf-8")).encode()
                    yield temp_count + i, {_DOCUMENT: text, _OUTPUT: target}
                    count += 1