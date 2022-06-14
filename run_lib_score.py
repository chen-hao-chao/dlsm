import os
import numpy as np
import tensorflow as tf
import logging
from absl import flags

from models import utils as mutils
from models import ddpm, ncsnv2, ncsnpp
from models.ema import ExponentialMovingAverage
from utils import save_checkpoint
import losses
import sampling
import datasets
import sde_lib
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image

FLAGS = flags.FLAGS


def train(config, workdir):
  """Execute the training procedure for the score model.
  Args:
    config: (dict) Experimental configuration file that specifies the setups and hyper-parameters.
    workdir: (str) Working directory for checkpoints and TF summaries.
  """

  # Create directories for experimental logs.
  sample_dir = os.path.join(workdir, "samples")
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(sample_dir)
  tf.io.gfile.makedirs(tb_dir)

  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  if config.model.score_restore_path:
    checkpoint = torch.load(config.model.score_restore_path, map_location=config.device)
    score_model.load_state_dict(checkpoint['model'], strict=False)
    ema.load_state_dict(checkpoint['ema'])
    ema.copy_to(score_model.parameters())
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
  initial_step = int(state['step'])

  # Build data iterators.
  train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)
  train_iter = iter(train_ds)
  eval_iter = iter(eval_ds)

  # Create data normalizer and its inverse.
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs.
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions.
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn, reduce_mean=reduce_mean, 
                                      continuous=continuous, likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn, reduce_mean=reduce_mean,
                                      continuous=continuous, likelihood_weighting=likelihood_weighting)

  # Build sampling functions.
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  # Start training.
  num_train_steps = config.training.n_iters
  logging.info("Starting training loop at step %d." % (initial_step,))

  for step in range(initial_step, num_train_steps + 1):    
    batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
    batch = batch.permute(0, 3, 1, 2)
    batch = scaler(batch)

    # Execute one training step.
    loss = train_step_fn(state, batch)
    if step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
      writer.add_scalar("training_loss", loss, step)

    # Report the loss on an evaluation dataset periodically.
    if step % config.training.eval_freq == 0:
      eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
      eval_batch = eval_batch.permute(0, 3, 1, 2)
      eval_batch = scaler(eval_batch)
      eval_loss = eval_step_fn(state, eval_batch)
      logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
      writer.add_scalar("eval_loss", eval_loss.item(), step)

    # Save a checkpoint periodically and generate samples if needed.
    if step % config.training.snapshot_freq == 0 or step == num_train_steps:
      save_step = step // config.training.snapshot_freq
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

      # Generate and save samples.
      if config.training.snapshot_sampling and step != 0:
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        sample, n = sampling_fn(score_model)
        ema.restore(score_model.parameters())
        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
        tf.io.gfile.makedirs(this_sample_dir)
        nrow = int(np.sqrt(sample.shape[0]))
        image_grid = make_grid(sample, nrow, padding=2)
        sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
          np.save(fout, sample)

        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
          save_image(image_grid, fout)