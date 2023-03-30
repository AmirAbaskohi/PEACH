# Copyright 2022 The peach Authors..
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

"""peach Params."""

import functools

from peach.data import parsers
from peach.eval import estimator_metrics
from peach.eval import text_eval
from peach.models import transformer
from peach.ops import public_parsing_ops
from peach.params import registry
from tensorflow.contrib import training as contrib_training

# Shift of special tokens id in the vocab file.
# I.e. the starting id of ordinary tokens in the vocab file.
NUM_RESERVED_TOKENS = 105
LENGTH_BUCKET_START_ID = 20
TASK_START_ID = 50


@registry.register("peach_base")
def peach_large_params(param_overrides):
  """Params for peachLarge."""

  hparams = contrib_training.HParams(
      length_bucket_size=0,
      add_task_id=False,
      batch_size=16,
      max_input_len=512,
      max_target_len=512,
      max_decode_len=512,
      max_total_words=0,
      pretrain_target_filter_min=0,
      hidden_size=768,
      filter_size=3072,
      num_heads=12,
      num_encoder_layers=12,
      num_decoder_layers=12,
      optimizer_name="adafactor",
      learning_rate=0.01,
      label_smoothing=0.0,
      dropout=0.1,
      train_steps=500000,
      beam_size=1,
      eval_max_predictions=1000,
      use_bfloat16=False,
      model=None,
      encoder=None,
      parser=None,
      estimator_prediction_fn=None,
      eval=None,
      estimator_eval_metrics_fn=estimator_metrics.pretrain_eval_metrics_fn,
  )

  if param_overrides:
    hparams.parse(param_overrides)

  hparams.encoder = public_parsing_ops.create_text_encoder(
      hparams.encoder_type, hparams.vocab_filename)
  hparams.parser = functools.partial(
      parsers.string_features_for_pretraining_parser,
      hparams.vocab_filename,
      hparams.encoder_type,
      hparams.max_input_len,
      hparams.max_target_len,
      hparams.max_total_words,
      hparams.parser_strategy,
      hparams.parser_masked_sentence_ratio,
      hparams.parser_masked_words_ratio, [
          hparams.parser_mask_word_by_msk_token_prob,
          hparams.parser_mask_word_by_random_token_prob,
          hparams.parser_mask_word_by_intact_prob
      ], [
          hparams.parser_mask_sentence_by_msk_token_prob,
          hparams.parser_mask_sentence_by_random_sentence_prob,
          hparams.parser_mask_sentence_by_intact_prob,
          hparams.parser_mask_sentence_by_remove_prob
      ],
      hparams.parser_rouge_ngrams_size,
      hparams.parser_rouge_metric_type,
      hparams.parser_rouge_compute_option,
      hparams.parser_rouge_stopwords_filename,
      NUM_RESERVED_TOKENS,
      parser_rouge_noise_ratio=hparams.parser_rouge_noise_ratio,
      parser_dynamic_mask_min_ratio=hparams.parser_dynamic_mask_min_ratio,
      input_feature="inputs",
      pretrain_target_filter_min=hparams.pretrain_target_filter_min,
      length_bucket_size=hparams.length_bucket_size,
      length_bucket_start_id=LENGTH_BUCKET_START_ID,
      length_bucket_max_id=TASK_START_ID - 1,
      add_task_id=hparams.add_task_id,
      task_start_id=TASK_START_ID)
  hparams.model = functools.partial(
      transformer.TransformerEncoderDecoderModel, hparams.encoder.vocab_size,
      hparams.hidden_size, hparams.filter_size, hparams.num_heads,
      hparams.num_encoder_layers, hparams.num_decoder_layers,
      hparams.label_smoothing, hparams.dropout)

  def decode_fn(features):
    return hparams.model().predict(features, hparams.max_decode_len,
                                   hparams.beam_size)

  hparams.estimator_prediction_fn = decode_fn
  hparams.eval = functools.partial(
      text_eval.text_eval, hparams.encoder, num_reserved=NUM_RESERVED_TOKENS)
  return hparams
