import os
import tensorflow as tf
import logging
from absl import flags

from models import utils as mutils
from models.ema import ExponentialMovingAverage
from models import ddpm, ncsnv2, ncsnpp, classifier
from utils import save_checkpoint
import losses as losses
import datasets
import sde_lib

import torch
from torch.utils import tensorboard

FLAGS = flags.FLAGS

def train(config, workdir):
  """Execute the training procedure for the classifier.
  Args:
    config: (dict) Experimental configuration file that specifies the setups and hyper-parameters.
    workdir: (str) Working directory for checkpoints and TF summaries.
  """

  # Create directories for experimental logs.
  sample_dir = os.path.join(workdir, "samples")
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(sample_dir)
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(tb_dir)

  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  checkpoint = torch.load(config.model.score_restore_path, map_location=config.device)
  score_model.load_state_dict(checkpoint['model'], strict=False)
  ema.load_state_dict(checkpoint['ema'])
  ema.copy_to(score_model.parameters())
  
  classifier_model = mutils.create_classifier(config)
  optimizer = losses.get_optimizer(config, classifier_model.parameters())
  state = dict(optimizer=optimizer, model=classifier_model, step=0)

  initial_step = int(state['step'])

  # Build data iterators.
  train_ds, _, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)
  _, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization, evaluation=True)
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
  num_classes = config.classifier.classes
  loss_type = config.model.loss_type
  weighting_dlsm = config.model.weighting_dlsm
  weighting_ce = config.model.weighting_ce
  coef = config.model.coef
  
  train_step_fn = losses.get_classifier_step_fn(sde, True, optimize_fn, continuous=continuous, num_classes=num_classes, loss_type=loss_type,
                                                weighting_dlsm=weighting_dlsm, weighting_ce=weighting_ce, coef=coef, eps=sampling_eps)
  eval_step_fn = losses.get_classifier_step_fn(sde, False, optimize_fn, continuous=continuous, num_classes=num_classes, loss_type=loss_type,
                                                weighting_dlsm=weighting_dlsm, weighting_ce=weighting_ce, coef=coef, eps=sampling_eps)

  num_train_steps = config.training.n_iters

  # Start training.
  logging.info("Starting training loop at step %d." % (initial_step,))
  
  for step in range(initial_step, num_train_steps + 1):
    data = next(train_iter)
    batch = torch.from_numpy(data['image']._numpy()).to(config.device).float()
    batch = batch.permute(0, 3, 1, 2)
    batch = scaler(batch)
    labels = torch.from_numpy(data['label']._numpy()).to(config.device).long()
    
    # Execute one training step.
    loss = train_step_fn(state, batch, labels, score_model)
    if step % config.training.log_freq == 0:
      logging.info("step: %d, loss: %.5e" % (step, loss.item()))
      writer.add_scalar("loss", loss, step)
    
    # Save a checkpoint periodically and generate samples if needed.
    if step % config.training.snapshot_freq == 0 or step == num_train_steps:
      save_step = step // config.training.snapshot_freq
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state, ema=False)

    # Report the loss on an evaluation dataset periodically.
    if step % config.training.eval_freq == 0:
      all_num = 0
      all_correct = 0
      while True:
        try:
          eval_data = next(eval_iter)
        except:
          eval_iter = iter(eval_ds)
          break
        # Get evaluation data.
        eval_batch = torch.from_numpy(eval_data['image']._numpy()).to(config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch)
        eval_labels = torch.from_numpy(eval_data['label']._numpy()).to(config.device).long()
        # Execute one evaluation step.
        correct_num, batch_num = eval_step_fn(state, eval_batch, eval_labels, score_model)
        all_num += batch_num
        all_correct += correct_num

      # Compute the overall accuracy.
      logging.info("Evaluation Accuracy: %.2e" % ((all_correct / all_num).item()*100))
      writer.add_scalar("eval_acc", (all_correct / all_num)*100, step)