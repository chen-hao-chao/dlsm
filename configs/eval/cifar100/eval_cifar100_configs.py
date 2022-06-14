from configs.default_cifar_configs import get_default_configs

def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True
  training.n_iters = 1000001
  training.batch_size = 100
  training.log_freq = 50
  training.eval_freq = 500

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'
  sampling.conditional = True
  sampling.classifier_restore_path = None
  sampling.snr = 0.16
  sampling.scaling_factor = 1

  # data
  data = config.data
  data.centered = False
  data.random_flip = False
  data.dataset = 'CIFAR100'
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

  # classifier
  classifier = config.classifier
  classifier.name = 'classifier'
  classifier.model = 'resnet34_cond'
  classifier.nf = 32
  classifier.embedding_type = 'fourier'
  classifier.fourier_scale = 16
  classifier.classes = 100

  # eval
  eval = config.eval
  eval.freq = 5
  eval.batch_size = 200
  eval.num_samples = 60000
  eval.num_samples_all = 50000
  eval.mode = 'class'
  eval.class_id = 0
  eval.class_id_end = 1

  plot = config.plot
  plot.type = "precision"

  return config
