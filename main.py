# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Training and evaluation"""

import run_lib_score, run_lib_classifier
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import tensorflow as tf

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("restore", None, "Path to the checkpoint of a pretrained score model.")
flags.DEFINE_string("model", "score", "Running mode: train classifier, train uda model, or train score function")
flags.mark_flags_as_required(["workdir", "config", "model"])


def main(argv):
  workdir = os.path.join('results', FLAGS.workdir)
  config = FLAGS.config
  config.model.score_restore_path = FLAGS.restore
  tf.io.gfile.makedirs(workdir)
  # Run the training pipeline
  if FLAGS.model == "classifier":
    run_lib_classifier.train(config, workdir)
  elif FLAGS.model == "score":
    run_lib_score.train(config, workdir)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
