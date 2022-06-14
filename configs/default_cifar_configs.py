import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 128
  training.n_iters = 1300000
  training.snapshot_freq = 50000
  training.log_freq = 50
  training.eval_freq = 100
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False
  

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16
  sampling.scaling_factor = 1.0
  sampling.conditional = False
  

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 9
  evaluate.end_ckpt = 26
  evaluate.batch_size = 1024
  evaluate.enable_sampling = False
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CIFAR10'
  data.image_size = 32
  data.random_flip = True
  data.centered = False
  data.uniform_dequantization = False
  data.num_channels = 3

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'
  model.weighting_dlsm = 0
  model.weighting_ce = 0
  model.coef = 1.0
  model.scaling_factor = 1
  model.loss_type = 'total'
  model.classifier_restore_path = None
  model.score_restore_path = None

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  # ($) classifier
  config.classifier = classifier = ml_collections.ConfigDict()
  classifier.nf = 128
  classifier.embedding_type = 'fourier'
  classifier.fourier_scale = 16
  classifier.classes = 10
  classifier.noise_list = [0.0001, 0.3333, 0.6666, 0.9999]

  # ($) classifier
  config.plot = plot = ml_collections.ConfigDict()
  plot.type = "precision"

  return config