# Copyright 2022 The PEACH Authors..
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Summarization params of baseline models for downstream datasets."""
import functools

from peach.data import parsers
from peach.eval import estimator_metrics
from peach.eval import text_eval
from peach.models import transformer
from peach.ops import public_parsing_ops
from peach.params import peach_params
from peach.params import registry
from tensorflow.contrib import training as contrib_training


def transformer_params(patterns, param_overrides):
  """Params for TransformerEncoderDecoderMLModel.

  Args:
    patterns: a dict include train_pattern, dev_pattern, test_pattern
    param_overrides: a string, comma separated list of name=value

  Returns:
    A instance of HParams
  """

  hparams = contrib_training.HParams(
      train_pattern=patterns["train_pattern"],
      dev_pattern=patterns["dev_pattern"],
      test_pattern=patterns["test_pattern"],
      vocab_filename="peach/ops/testdata/sp_test.model",
      encoder_type="sentencepiece_newline",
      length_bucket_size=0,
      add_task_id=False,
      batch_size=patterns["batch_size"],
      max_input_len=patterns["max_input_len"],
      max_target_len=patterns["max_output_len"],
      max_decode_len=patterns["max_output_len"],
      hidden_size=768,
      filter_size=3072,
      num_heads=12,
      num_encoder_layers=12,
      num_decoder_layers=12,
      beam_size=1,
      beam_start=5,
      beam_alpha=0.6,
      beam_min=0,
      beam_max=-1,
      temperature=0.0,
      top_k=0,
      top_p=0.0,
      optimizer_name="adafactor",
      train_steps=patterns["train_steps"],
      learning_rate=patterns["learning_rate"],
      label_smoothing=0.0,
      dropout=0.1,
      eval_max_predictions=patterns.get("eval_steps", 1000),
      use_bfloat16=False,
      model=None,
      parser=None,
      encoder=None,
      estimator_prediction_fn=None,
      eval=None,
      estimator_eval_metrics_fn=estimator_metrics.gen_eval_metrics_fn,
  )

  if param_overrides:
    hparams.parse(param_overrides)

  hparams.parser = functools.partial(
      parsers.supervised_strings_parser,
      hparams.vocab_filename,
      hparams.encoder_type,
      hparams.max_input_len,
      hparams.max_target_len,
      length_bucket_size=hparams.length_bucket_size,
      length_bucket_start_id=peach_params.LENGTH_BUCKET_START_ID,
      length_bucket_max_id=peach_params.TASK_START_ID - 1,
      add_task_id=hparams.add_task_id,
      task_start_id=peach_params.TASK_START_ID)

  hparams.encoder = public_parsing_ops.create_text_encoder(
      hparams.encoder_type, hparams.vocab_filename)

  hparams.model = functools.partial(
      transformer.TransformerEncoderDecoderModel, hparams.encoder.vocab_size,
      hparams.hidden_size, hparams.filter_size, hparams.num_heads,
      hparams.num_encoder_layers, hparams.num_decoder_layers,
      hparams.label_smoothing, hparams.dropout)

  beam_keys = ("beam_start", "beam_alpha", "beam_min", "beam_max",
               "temperature", "top_k", "top_p")
  beam_kwargs = {k: hparams.get(k) for k in beam_keys if k in hparams.values()}

  def decode_fn(features):
    return hparams.model().predict(features, hparams.max_decode_len,
                                   hparams.beam_size, **beam_kwargs)

  hparams.estimator_prediction_fn = decode_fn
  hparams.eval = functools.partial(
      text_eval.text_eval,
      hparams.encoder,
      num_reserved=peach_params.NUM_RESERVED_TOKENS)

  return hparams

@registry.register("en-denoising")
def billsum_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:EnDenoising-train",
          "dev_pattern": "tfds:EnDenoising-train",
          "test_pattern": "tfds:EnDenoising-train",
          "max_input_len": 512,
          "max_output_len": 512,
          "train_steps": 500000,
          "learning_rate": 0.01,
          "batch_size": 96,
      }, param_overrides)

@registry.register("fr-denoising")
def billsum_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:FrDenoising-train",
          "dev_pattern": "tfds:FrDenoising-train",
          "test_pattern": "tfds:FrDenoising-train",
          "max_input_len": 512,
          "max_output_len": 512,
          "train_steps": 500000,
          "learning_rate": 0.01,
          "batch_size": 96,
      }, param_overrides)

@registry.register("de-denoising")
def billsum_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:DeDenoising-train",
          "dev_pattern": "tfds:DeDenoising-train",
          "test_pattern": "tfds:DeDenoising-train",
          "max_input_len": 512,
          "max_output_len": 512,
          "train_steps": 500000,
          "learning_rate": 0.01,
          "batch_size": 96,
      }, param_overrides)

@registry.register("en-improve")
def billsum_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:EnImprove-test",
          "dev_pattern": "tfds:EnImprove-test",
          "test_pattern": "tfds:EnImprove-test",
          "max_input_len": 512,
          "max_output_len": 512,
          "train_steps": 500000,
          "learning_rate": 0.01,
          "batch_size": 96,
      }, param_overrides)

@registry.register("fr-improve")
def billsum_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:FrImprove-test",
          "dev_pattern": "tfds:FrImprove-test",
          "test_pattern": "tfds:FrImprove-test",
          "max_input_len": 512,
          "max_output_len": 512,
          "train_steps": 500000,
          "learning_rate": 0.01,
          "batch_size": 96,
      }, param_overrides)

@registry.register("de-improve")
def billsum_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:DeImprove-test",
          "dev_pattern": "tfds:DeImprove-test",
          "test_pattern": "tfds:DeImprove-test",
          "max_input_len": 512,
          "max_output_len": 512,
          "train_steps": 500000,
          "learning_rate": 0.01,
          "batch_size": 96,
      }, param_overrides)


@registry.register("en-fr")
def billsum_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:WMT14ENFR-train",
          "dev_pattern": "tfds:WMT14ENFR-validation",
          "test_pattern": "tfds:WMT14ENFR-test",
          "max_input_len": 512,
          "max_output_len": 512,
          "train_steps": 50000,
          "learning_rate": 0.00005,
          "batch_size": 96,
      }, param_overrides)

@registry.register("fr-en")
def billsum_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:WMT14FREN-train",
          "dev_pattern": "tfds:WMT14FREN-validation",
          "test_pattern": "tfds:WMT14FREN-test",
          "max_input_len": 512,
          "max_output_len": 512,
          "train_steps": 50000,
          "learning_rate": 0.00005,
          "batch_size": 96,
      }, param_overrides)

@registry.register("de-en")
def billsum_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:WMT14DEEN-train",
          "dev_pattern": "tfds:WMT14DEEN-validation",
          "test_pattern": "tfds:WMT14DEEN-test",
          "max_input_len": 512,
          "max_output_len": 512,
          "train_steps": 50000,
          "learning_rate": 0.00005,
          "batch_size": 96,
      }, param_overrides)

@registry.register("en-de")
def billsum_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:WMT14ENDE-train",
          "dev_pattern": "tfds:WMT14ENDE-validation",
          "test_pattern": "tfds:WMT14ENDE-test",
          "max_input_len": 512,
          "max_output_len": 512,
          "train_steps": 50000,
          "learning_rate": 0.00005,
          "batch_size": 96,
      }, param_overrides)

@registry.register("fr-de")
def billsum_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:WMT19FRDE-train",
          "dev_pattern": "tfds:WMT19FRDE-validation",
          "test_pattern": "tfds:WMT19FRDE-test",
          "max_input_len": 512,
          "max_output_len": 512,
          "train_steps": 50000,
          "learning_rate": 0.00005,
          "batch_size": 96,
      }, param_overrides)

@registry.register("de-fr")
def billsum_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:WMT19DEFR-train",
          "dev_pattern": "tfds:WMT19DEFR-validation",
          "test_pattern": "tfds:WMT19DEFR-test",
          "max_input_len": 512,
          "max_output_len": 512,
          "train_steps": 50000,
          "learning_rate": 0.00005,
          "batch_size": 96,
      }, param_overrides)

@registry.register("en-mk")
def billsum_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:TatoebaENMK-train",
          "dev_pattern": "tfds:TatoebaENMK-validation",
          "test_pattern": "tfds:TatoebaENMK-test",
          "max_input_len": 512,
          "max_output_len": 512,
          "train_steps": 20000,
          "learning_rate": 0.00005,
          "batch_size": 96,
      }, param_overrides)

@registry.register("mk-en")
def billsum_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": "tfds:TatoebaMKEN-train",
          "dev_pattern": "tfds:TatoebaMKEN-validation",
          "test_pattern": "tfds:TatoebaMKEN-test",
          "max_input_len": 512,
          "max_output_len": 512,
          "train_steps": 20000,
          "learning_rate": 0.00005,
          "batch_size": 96,
      }, param_overrides)

