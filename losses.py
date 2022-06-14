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

"""All functions related to loss computation and optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE
from functools import reduce

def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  elif config.optim.optimizer == 'SGD':
    optimizer = optim.SGD(params, lr=config.optim.lr, momentum=config.optim.momentum)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer

def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn

def get_sde_classifier_loss_fn(sde, train, num_classes, continuous, loss_type, weighting_dlsm, weighting_ce, coef, eps):
  """Construct a one-step training/evaluation function.
  Args: 
    train: (bool) The indication for training. It is set as True for training mode.
    continuous: (bool) The indication for continuous.
    num_classes: (int) The number of classes.
    loss_type: (str) The indication for the type of loss.
    weighting_dlsm: (int) The power of the balancing coefficient for the DLSM loss. For example, 
                     if weighting_dlsm=2, the coefficient is 1/std^(2*2).
    weighting_ce: (int) The power of the balancing coefficient for the CE loss. For example, 
                     if weighting_ce=0, the coefficient is 1/std^(2*0).
    coef: (float) The coefficient for balancing the DLSM and the CE losses.
    eps: (float) An exetremely small value. It is used for preventing overflow.
  Returns:
    loss_fn: (func) A one-step training/evaluation function.
  """
  def loss_fn(classifier_model, score_model, batch, labels):
    """Compute the loss function for training.
    Args:
      score_model: (nn.Module) A parameterized score model.
      classifier_model: (nn.Module) A parameterized classifier.
      batch: (tensor) A mini-batch of training data.
      labels: (tensor) A mini-batch of labels of the training data.
    Returns:
      loss: (float) The average loss value across the mini-batch.
    """
    # Define classifier, softmax and ce functions.
    sm = nn.Softmax(dim=1)
    loss_ce_fn = torch.nn.CrossEntropyLoss(reduce=False)
    classifier_fn = mutils.get_classifier_fn(sde, classifier_model, train=train, continuous=continuous)

    # Perturb the images.
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std_value_cond = sde.marginal_prob(batch, t)
    std = std_value_cond[:, None, None, None]
    perturbed_data = mean + std * z

    # Get score function.
    with torch.no_grad():
      score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=continuous)
      score = score_fn(perturbed_data, t)

    # Make predictions.
    perturbed_data_var = torch.tensor(perturbed_data, requires_grad=True)
    out = classifier_fn(perturbed_data_var, t)
    
    # Calculate the losses
    if loss_type == 'total' or loss_type == 'dlsm':
      # Calculate the dlsm loss
      log_prob_class = torch.log(sm(out)+ 1e-8)
      label_mask = F.one_hot(labels, num_classes=num_classes)
      grads_prob_class, = torch.autograd.grad(log_prob_class, perturbed_data_var, 
                          grad_outputs=label_mask, create_graph=True)
      loss_dlsm = torch.mean(0.5 * torch.square(grads_prob_class * (std ** weighting_dlsm) + score * (std ** weighting_dlsm) + z * (std ** (weighting_dlsm-1)) ))

    if loss_type == 'total' or loss_type == 'ce':
      # Calculate the ce loss
      loss_ce = torch.mean(loss_ce_fn(out, labels)*(std_value_cond ** (-2 * weighting_ce)))
    
    loss = (loss_dlsm + coef * loss_ce) if loss_type == 'total' else (loss_dlsm if loss_type == 'dlsm' else loss_ce)
    return loss
  
  def acc_fn(classifier_model, batch, labels):
    """Compute the accuracy for evaluation.
    Args:
      classifier_model: (nn.Module) A parameterized classifier.
      batch: (tensor) A mini-batch of training data.
      labels: (tensor) A mini-batch of labels of the training data.
    Returns:
      loss: (float) The average loss value across the mini-batch.
    """
    # Define classifier, softmax and ce functions.
    sm = nn.Softmax(dim=1)
    classifier_fn = mutils.get_classifier_fn(sde, classifier_model, train=train, continuous=continuous)

    # Perturb the images
    t = torch.zeros(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std_value_cond = sde.marginal_prob(batch, t)
    std = std_value_cond[:, None, None, None]
    perturbed_data = mean + std * z
    
    # Make predictions
    pred = classifier_fn(perturbed_data, t)
    pred = sm(pred)
    pred = torch.argmax(pred, dim=1)
    correct_num = (pred == labels).sum()
    all_num = pred.shape[0]

    return correct_num, all_num

  return loss_fn, acc_fn


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-3):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    # Perturb the images
    n = torch.rand(batch.shape[0], device=batch.device)
    t = n * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    # Make predictions
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    score = score_fn(perturbed_data, t)
    # Calculate the losses
    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[:, None, None, None]
    perturbed_data = noise + batch
    score = model_fn(perturbed_data, labels)
    target = -noise / (sigmas ** 2)[:, None, None, None]
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(state, batch):
    """Running one step of training or evaluation.
    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.
    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.
    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model, batch)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch)
        ema.restore(model.parameters())
    return loss

  return step_fn

def get_classifier_step_fn(sde, train, optimize_fn, continuous=True, num_classes=10, loss_type='total', weighting_dlsm=0, weighting_ce=0, coef=1.0, eps=1e-3):
  if continuous:
    loss_fn, acc_fn = get_sde_classifier_loss_fn(sde, train, num_classes, continuous, loss_type, weighting_dlsm, weighting_ce, coef, eps)
  else:
    raise ValueError(f"Discrete training for classifier is not yet supported.")
  
  def step_fn(state, batch, labels, score_model):
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model, score_model, batch, labels)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      return loss
    else:
      with torch.no_grad():
        correct_num, batch_num = acc_fn(model, batch, labels)
      return correct_num, batch_num
  return step_fn