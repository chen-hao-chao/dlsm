import numpy as np
import io
import os
import time
import tensorflow as tf
import torch
from torchvision.utils import make_grid, save_image
from ml_collections.config_flags import config_flags
from absl import flags
from absl import app

import sampling
import datasets
import sde_lib
from models import ncsnpp, classifier
from models import utils as mutils
from models.ema import ExponentialMovingAverage

FLAGS = flags.FLAGS

def sampling_function(config, workdir):
  # Create directories for experimental logs.
  sample_dir = os.path.join(workdir, "eval_samples")
  tf.io.gfile.makedirs(sample_dir)
  
  # Initialize models.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  checkpoint = torch.load(config.model.score_restore_path, map_location=config.device)
  score_model.load_state_dict(checkpoint['model'], strict=False)
  ema.load_state_dict(checkpoint['ema'])
  ema.copy_to(score_model.parameters())
  classifier_model = mutils.create_classifier(config)
  checkpoint = torch.load(config.model.classifier_restore_path)
  classifier_model.load_state_dict(checkpoint['model'])

  # Create data normalizer and its inverse.
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Building sampling functions.
  sampling_shape = (config.training.batch_size, config.data.num_channels,
                    config.data.image_size, config.data.image_size)
  sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, eps)
  
  if config.eval.mode == 'full':
    # Conditionally generate the images for each class.
    num_sampling_rounds = config.eval.num_samples // config.training.batch_size 
    num_classes = config.classifier.classes
    with torch.no_grad():
      for c in range(num_classes):
        sample_c_dir = os.path.join(sample_dir, str(c))
        tf.io.gfile.makedirs(sample_c_dir)
        for r in range(0, num_sampling_rounds // num_classes):
          print("Class {} || Rounds: {}/{}".format(c, r+1, num_sampling_rounds//num_classes))
          now = time.time()
          
          # Generate the images using the sampling function.
          cond = torch.ones((sampling_shape[0],), dtype=torch.long).to(config.device)*c
          samples, _ = sampling_fn(score_model, classifier_model, cond)
          # Save the samples.
          nrow = int(np.sqrt(samples.shape[0]))
          image_grid = make_grid(samples, nrow, padding=2)
          samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
          samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))
          with tf.io.gfile.GFile(os.path.join(sample_c_dir, f"samples_{r}.npz"), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, samples=samples)
            fout.write(io_buffer.getvalue())
          with tf.io.gfile.GFile(os.path.join(sample_c_dir, f"samples_{r}.png"), "wb") as fout:
            save_image(image_grid, fout)
          
          later = time.time()
          difference = int(later - now)
          print("Time consumption: ", str(difference), " sec.")
    
  elif config.eval.mode == 'class':
    # Conditionally generate the images for a class or certain classes
    num_sampling_rounds = config.eval.num_samples // config.training.batch_size 
    num_classes = config.classifier.classes
    with torch.no_grad():
      for c in range(config.eval.class_id, config.eval.class_id_end+1):
        sample_c_dir = os.path.join(sample_dir, str(c))
        tf.io.gfile.makedirs(sample_c_dir)
        for r in range(0, num_sampling_rounds // num_classes):
          print("Class {} || Rounds: {}/{}".format(c, r+1, num_sampling_rounds//num_classes))
          now = time.time()
          
          # Generate the images using the sampling function.
          cond = torch.ones((sampling_shape[0],), dtype=torch.long).to(config.device)*c
          samples, _ = sampling_fn(score_model, classifier_model, cond)
          # Save the samples.
          nrow = int(np.sqrt(samples.shape[0]))
          image_grid = make_grid(samples, nrow, padding=2)
          samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
          samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))
          with tf.io.gfile.GFile(os.path.join(sample_c_dir, f"samples_{r}.npz"), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, samples=samples)
            fout.write(io_buffer.getvalue())
          with tf.io.gfile.GFile(os.path.join(sample_c_dir, f"samples_{r}.png"), "wb") as fout:
            save_image(image_grid, fout)
          
          later = time.time()
          difference = int(later - now)
          print("Time consumption: ", str(difference), " sec.")
  else:
    raise ValueError(f"Mode {config.eval.mode} not recognized.")

config_flags.DEFINE_config_file("config", None, "Sampling configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("mode", 'full', "Mode for evaluation.")
flags.DEFINE_integer("classid", 0, "The starting class index.")
flags.DEFINE_integer("classidend", -1, "The ending class index.")
flags.DEFINE_string("restore", None, "Path to the checkpoint of a pretrained score model.")
flags.DEFINE_string("restore_classifier", None, "Path to the checkpoint of a pretrained classifier.")
flags.DEFINE_float("scale", 1.0, "Scaling factor")
flags.mark_flags_as_required(["workdir", "config", "mode"])

def main(argv):
  config = FLAGS.config
  workdir = os.path.join('results', FLAGS.workdir)
  tf.io.gfile.makedirs(workdir)
  # Adjust the config file
  config.eval.mode = FLAGS.mode
  config.eval.class_id = FLAGS.classid
  config.eval.class_id_end = FLAGS.classid if FLAGS.classidend == -1 else FLAGS.classidend
  config.model.classifier_restore_path = FLAGS.restore_classifier
  config.model.score_restore_path = FLAGS.restore
  config.sampling.scaling_factor = FLAGS.scale
  # Run the code
  sampling_function(config, workdir)

if __name__ == "__main__":
  app.run(main)