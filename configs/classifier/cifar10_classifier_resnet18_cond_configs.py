from configs.default_cifar_configs import get_default_configs

def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True
  training.n_iters = 60000
  training.batch_size = 150
  training.log_freq = 50
  training.eval_freq = 500
  training.snapshot_freq = 5000

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'
  sampling.conditional = True

  # data
  data = config.data
  data.centered = False
  data.random_flip = False
  data.dataset = 'CIFAR10'
  data.image_size = 32
  data.num_channels = 3

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.fourier_scale = 16
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 8
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.0
  model.conv_size = 3
  model.classifier_restore_path = None 
  model.score_restore_path = 'results/checkpoint.pth'
  
  # optimize
  optim = config.optim
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.grad_clip = 1.
  optim.momentum = 0.9
  optim.warmup = 0
  optim.gamma = 0.9

  # classifier
  ### ($)
  classifier = config.classifier
  classifier.name = 'classifier'
  classifier.model = 'resnet18_cond'
  classifier.nf = 32
  classifier.embedding_type = 'fourier'
  classifier.fourier_scale = 16
  classifier.classes = 10

  # eval
  eval = config.eval
  eval.batch_size = 2000
  eval.num_samples = 50000
  eval.freq = 5

  plot = config.plot
  plot.type = "precision"

  return config
