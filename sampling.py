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

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import torch.nn as nn
import numpy as np
import abc

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils
import torch.nn.functional as F

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,
                                  shape=shape,
                                  inverse_scaler=inverse_scaler,
                                  denoise=config.sampling.noise_removal,
                                  eps=eps,
                                  device=config.device)
  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    sampling_fn = get_pc_sampler(sde=sde,
                                 shape=shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 inverse_scaler=inverse_scaler,
                                 snr=config.sampling.snr,
                                 n_steps=config.sampling.n_steps_each,
                                 probability_flow=config.sampling.probability_flow,
                                 continuous=config.training.continuous,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=config.device,
                                 conditional=config.sampling.conditional,
                                 adaptive=(config.sampling.predictor == 'adaptive'),
                                 scaling_factor=config.sampling.scaling_factor)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, classifier_fn=None, probability_flow=False, conditional=False, cond=None, scaling_factor=1):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn=score_fn, classifier_fn=classifier_fn, probability_flow=probability_flow, conditional=conditional, cond=cond, scaling_factor=scaling_factor)
    self.score_fn = score_fn
    self.classifier_fn = classifier_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps, classifier_fn=None, conditional=False, cond=None, scaling_factor=1):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.classifier_fn = classifier_fn
    self.snr = snr
    self.n_steps = n_steps
    self.cond = cond
    self.scaling_factor = scaling_factor
    self.conditional = conditional

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, classifier_fn=None, probability_flow=False, conditional=False, cond=None, scaling_factor=1):
    super().__init__(sde, score_fn, classifier_fn, probability_flow, conditional, cond, scaling_factor)
    self.rsde = sde.reverse(score_fn=score_fn, classifier_fn=classifier_fn, probability_flow=probability_flow, conditional=conditional, cond=cond, scaling_factor=scaling_factor)

  def update_fn(self, x, t):
    dt = -1. / self.rsde.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, t)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
    return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, classifier_fn=None, probability_flow=False, conditional=False, cond=None, scaling_factor=1):
    super().__init__(sde, score_fn, classifier_fn, probability_flow, conditional, cond, scaling_factor)
    self.rsde = sde.reverse(score_fn=score_fn, classifier_fn=classifier_fn, probability_flow=probability_flow, conditional=conditional, cond=cond, scaling_factor=scaling_factor)

  def update_fn(self, x, t):
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * z
    return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, classifier_fn=None, probability_flow=False, conditional=False, cond=None, scaling_factor=1):
    super().__init__(sde, score_fn, classifier_fn, probability_flow, conditional, cond)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"
    self.rsde = sde.reverse(score_fn=score_fn, classifier_fn=classifier_fn, probability_flow=probability_flow, conditional=conditional, cond=cond, scaling_factor=scaling_factor)

  def vesde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
    score = self.score_fn(x, t)
    x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
    std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = torch.randn_like(x)
    x = x_mean + std[:, None, None, None] * noise
    return x, x_mean

  def vpsde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = self.score_fn(x, t)
    x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
    noise = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
    return x, x_mean

  def update_fn(self, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, classifier_fn=None, probability_flow=False, conditional=False, cond=None, scaling_factor=1):
    pass

  def update_fn(self, x, t):
    return x, x
  

# EM or Improved-Euler (Heun's method) with adaptive step-sizes
@register_predictor(name='adaptive')
class AdaptivePredictor(Predictor):
  def __init__(self, sde, score_fn, classifier_fn=None, probability_flow=False, conditional=False, cond=None, 
    eps=1e-5, abstol = 0.0078, reltol = 1e-2, error_use_prev=True, norm = "L2_scaled",
    safety = .9, sde_improved_euler=True, extrapolation = True, exp=0.9, variance=1, scaling_factor=1):
    super().__init__( sde, score_fn, classifier_fn, probability_flow, conditional, cond, scaling_factor)
    self.h_min = 1e-10 # min step-size
    self.t = sde.T # starting t
    self.eps = eps # end t
    self.abstol = abstol
    self.reltol = reltol
    self.error_use_prev = error_use_prev
    self.norm = norm
    self.safety = safety
    self.sde_improved_euler = sde_improved_euler
    self.extrapolation = extrapolation
    self.exp = exp
    self.variance = variance
    self.rsde = sde.reverse(score_fn=score_fn, classifier_fn=classifier_fn, probability_flow=probability_flow, conditional=conditional, cond=cond, scaling_factor=scaling_factor)
    
    if self.norm == "L2_scaled":
      def norm_fn(x):
        n = x.shape[1]*x.shape[2]*x.shape[3]
        return torch.sqrt(torch.sum((x)**2, dim=(1,2,3), keepdims=True)/n)
    elif self.norm == "L2":
      def norm_fn(x):
        return torch.sqrt(torch.sum((x)**2, dim=(1,2,3), keepdims=True))
    elif self.norm == "Linf":
      def norm_fn(x):
        return torch.max(torch.abs(x), dim=(1,2,3), keepdims=True)
    else:
      raise NotImplementedError(self.norm)
    self.norm_fn = norm_fn


  def update_fn(self, x, x_prev, t, h): 
    # Note: both h and t are vectors with batch_size elems (this is because we want adaptive step-sizes for each sample separately)
    # drift: [batch_size, channels, img_size, img_size]
    # diffusion: [batch_size]
    h_ = h[:, None, None, None] # [batch_size, 1, 1, 1]
    t_ = t[:, None, None, None] # expand for multiplications

    z = torch.randn_like(x)*self.variance
    drift, diffusion = self.rsde.sde(x, t)

    # Heun's method for SDE (while Lamba method only focuses on the non-stochastic part, this also includes the stochastic part)
    K1 = -(h_ * drift) + (diffusion[:, None, None, None] * torch.sqrt(h_) * z) 
    drift_Heun, diffusion_Heun = self.rsde.sde(x + K1, t - h)
    K2 = -(h_ * drift_Heun) + (diffusion_Heun[:, None, None, None] * torch.sqrt(h_) * z)
    E = (K2 - K1) / 2 # local-error between EM and Heun (SDEs) (right one)

    if self.extrapolation: # Extrapolate using the Heun's method result
      x_new = x + (K1 + K2) / 2
      x_check = x + K1 # x_prime in the algorithm
    else:
      x_new = x + K1
      x_check = x + (K1 + K2) / 2

    # Calculating the error-control
    if self.error_use_prev:
      reltol_ctl = torch.maximum(torch.abs(x_prev), torch.abs(x_check)) * self.reltol
    else:
      reltol_ctl = torch.abs(x_check) * self.reltol
    err_ctl = torch.maximum(reltol_ctl, torch.ones(reltol_ctl.shape).to(reltol_ctl.device)*self.abstol) # [batch_size, channels, img_size, img_size]

    # Normalizing for each sample separately
    E_scaled_norm = self.norm_fn( E/err_ctl ) # [batch_size, 1, 1, 1]

    # Accept or reject x_{n+1} and t_{n+1} for each sample separately
    accept = torch.le(E_scaled_norm, 1) # T/F tensor
    x = torch.where(accept, x_new, x)
    x_prev = torch.where(accept, x_check, x_prev)
    t_ = torch.where(accept, t_ - h_, t_)

    # Change the step-size
    h_max = torch.maximum(t_ - self.eps, torch.zeros(t_.shape).to(t_.device)) # max step-size must be the distance to the end (we use maximum between that and zero in case of a tiny but negative value: -1e-10)
    E_pow = torch.where(h_ == 0, h_, torch.pow(E_scaled_norm, -self.exp)) # (E^-r) Only applies power when not zero, otherwise, we get nans
    h_new = torch.minimum(h_max, self.safety*h_*E_pow)

    return x, x_prev, t_.reshape((-1)), h_new.reshape((-1))

@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps, classifier_fn=None, conditional=False, cond=None, scaling_factor=1):
    super().__init__(sde, score_fn, snr, n_steps, classifier_fn, conditional, cond, scaling_factor)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    
  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    for i in range(n_steps):
      score = score_fn(x, t)
      if self.conditional:
        with torch.enable_grad():
          x = x.clone().detach().requires_grad_(True)
          labels = torch.ones(x.shape[0], device=x.device, dtype=torch.long) * self.cond
          sm = nn.Softmax(dim=1)
          log_prob_class = torch.log(sm(self.classifier_fn(x, t))+ 1e-8)
          label_mask = F.one_hot(labels, num_classes=log_prob_class.shape[1])
          grads_prob_class, = torch.autograd.grad(log_prob_class, x, grad_outputs=label_mask)
        score = score + grads_prob_class*self.scaling_factor

      noise = torch.randn_like(x)
      grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * score
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps, classifier_fn=None, conditional=False, cond=None, scaling_factor=1):
    super().__init__(sde, score_fn, snr, n_steps, classifier_fn, conditional, cond, scaling_factor)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps, classifier_fn=None, conditional=False, cond=None, scaling_factor=1):
    pass

  def update_fn(self, x, t):
    return x, x


def shared_predictor_update_fn(x, t, sde, score_fn, classifier_fn, predictor, probability_flow, 
                                conditional, cond, h=None, x_prev=None, scaling_factor=1):
  """A wrapper that configures and returns the update function of predictors."""
  if predictor is None:
    predictor_obj = NonePredictor(sde=sde, score_fn=score_fn, classifier_fn=classifier_fn,
                                    probability_flow=probability_flow, conditional=conditional, cond=cond, scaling_factor=scaling_factor)
  else:
    predictor_obj = predictor(sde=sde, score_fn=score_fn, classifier_fn=classifier_fn,
                                    probability_flow=probability_flow, conditional=conditional, cond=cond, scaling_factor=scaling_factor)
  if h is not None:
    return predictor_obj.update_fn(x, x_prev, t, h)
  else:
    return predictor_obj.update_fn(x, t)
  


def shared_corrector_update_fn(x, t, sde, score_fn, classifier_fn, corrector,
                                conditional, cond, snr, n_steps, scaling_factor=1):
  """A wrapper tha configures and returns the update function of correctors."""
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde=sde, score_fn=score_fn, snr=snr, n_steps=n_steps,
                                classifier_fn=classifier_fn, conditional=conditional, cond=cond, scaling_factor=scaling_factor)
  else:
    corrector_obj = corrector(sde=sde, score_fn=score_fn, snr=snr, n_steps=n_steps, classifier_fn=classifier_fn, conditional=conditional, cond=cond, scaling_factor=scaling_factor)
  return corrector_obj.update_fn(x, t)


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-5, h_init=5e-1, device='cuda',
                   conditional=False, adaptive=False, scaling_factor=1):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          conditional=conditional,
                                          scaling_factor=scaling_factor)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          snr=snr,
                                          n_steps=n_steps,
                                          conditional=conditional,
                                          scaling_factor=scaling_factor)

  def pc_sampler(score_model, classifier_model=None, cond=None):
    """ The PC sampler funciton.

    Args:
      score_fn: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      
      score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=continuous)
      if classifier_model is not None:
        classifier_fn = mutils.get_classifier_fn(sde, classifier_model, train=False, continuous=continuous)
      else:
        classifier_fn = None

      if adaptive:
        # Initial sample
        x = sde.prior_sampling(shape).to(device)
        h = torch.ones(shape[0]).to(device) * h_init # initial step_size
        t = torch.ones(shape[0]).to(device) * sde.T # initial time
        n_iter = 0

        while((torch.abs(t - eps) > 1e-6).any()):
          x, x_prev = corrector_update_fn(x=x, t=t, score_fn=score_fn, classifier_fn=classifier_fn, cond=cond)
          x, x_prev, t, h = predictor_update_fn(x=x, t=t, h=h, x_prev=x_prev, score_fn=score_fn, classifier_fn=classifier_fn, cond=cond)
          n_iter += 1

        return inverse_scaler(x), n_iter

      else:
        x = sde.prior_sampling(shape).to(device)
        timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
        for i in range(sde.N):
          t = timesteps[i]
          vec_t = torch.ones(shape[0], device=t.device) * t
          x, x_mean = corrector_update_fn(x=x, t=vec_t, score_fn=score_fn, classifier_fn=classifier_fn, cond=cond)
          x, x_mean = predictor_update_fn(x=x, t=vec_t, score_fn=score_fn, classifier_fn=classifier_fn, cond=cond)
          
        return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)
  
  return pc_sampler



def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  def denoise_update_fn(model, x):
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x

  def drift_fn(model, x, t):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def ode_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      if z is None:
        # If not represent, sample the latent code from the prior distibution of the SDE.
        x = sde.prior_sampling(shape).to(device)
      else:
        x = z

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      # Denoising is equivalent to running one predictor step without adding noise
      if denoise:
        x = denoise_update_fn(model, x)

      x = inverse_scaler(x)
      return x, nfe

  return ode_sampler
