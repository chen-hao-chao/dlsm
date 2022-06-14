import gc
import numpy as np
import os
import glob
import tensorflow as tf
import tensorflow_gan as tfgan
from ml_collections.config_flags import config_flags
from absl import flags
from absl import app

import sampling
import datasets
import sde_lib
import evaluation
from models import ncsnpp, classifier
from models import utils as mutils
from models.ema import ExponentialMovingAverage

from prdc import compute_prdc

FLAGS = flags.FLAGS

def create_stat_files(config, workdir):
  '''
  This function converts the samples from the dataset to the np.array format,
  and stores it as `stat.npz' in a class-wise order.
  '''
  stat_dir = os.path.join(workdir, "stat")
  tf.io.gfile.makedirs(stat_dir)
  # Convert dataset to stat file
  config.training.batch_size = config.eval.num_samples_all
  training_ds, _, _ = datasets.get_dataset(config, uniform_dequantization=False)
  iter_ds = iter(training_ds)
  data = next(iter_ds)
  batch = data['image']._numpy()
  labels = data['label']._numpy()
  
  images = np.zeros(batch.shape)
  sizes = np.zeros(config.classifier.classes)
  starts = np.zeros(config.classifier.classes)
  start = 0
  
  for c in range(config.classifier.classes):
      indeces = np.where(labels == c)
      class_images = batch[indeces, :, :, :]
      print(class_images.shape)
      size = class_images.shape[1]
      print("Class {} || Number of Images: {}".format(c, size))
      print("======================")
      sizes[c] = size
      starts[c] = start
      images[start:start+size, :, :, :] = class_images
      start = start + size
  
  np.savez(os.path.join(stat_dir, 'stat.npz'), batch=images, size=sizes, start=starts)

def create_latent_files(config, workdir, datastat):
  '''
  This function encodes the generated samples as well as the samples in `stat.npz' into
  the latents with a pretrained Inception model, and store them in a class-wise order.
  '''
  # Construct the inception model.
  inceptionv3 = config.data.image_size >= 256
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  # Encodes the samples from the dataset.
  stat_dir = os.path.join(workdir, "stat")
  stats = np.load(os.path.join(stat_dir, "stat.npz"))
  batch_all = stats['batch']
  size = stats['size']
  start = stats['start']
  pools = None
  logits = None
  for c in range(config.classifier.classes):
    print("Encode the images for class {}.".format(c))
    ss = int(start[c])
    if config.data.dataset == 'CIFAR10':
      # If the number of samples in a batch is too large, the batch will be split into mini-batches.
      for minibatch_idx in range(config.eval.batch_splits):
        if minibatch_idx != config.eval.batch_splits-1:
          se = int(size[c] // config.eval.batch_splits) 
        else:
          se = int(size[c] // config.eval.batch_splits) + int(size[c] % config.eval.batch_splits)
        batch = np.clip(batch_all[ss:ss+se, :, :, :] * 255.0, 0, 255).astype(np.uint8)
        gc.collect()
        latent = evaluation.run_inception_distributed(batch, inception_model, inceptionv3=inceptionv3)
        gc.collect()
        pools = np.concatenate((pools, latent['pool_3']), axis=0) if pools is not None else latent['pool_3']
        logits = np.concatenate((logits, latent['logits']), axis=0) if logits is not None else latent['logits']
        ss += se
    else:
      se = int(size[c])
      batch = np.clip(batch_all[ss:ss+se, :, :, :] * 255.0, 0, 255).astype(np.uint8)
      gc.collect()
      latent = evaluation.run_inception_distributed(batch, inception_model, inceptionv3=inceptionv3)
      gc.collect()
      pools = np.concatenate((pools, latent['pool_3']), axis=0) if pools is not None else latent['pool_3']
      logits = np.concatenate((logits, latent['logits']), axis=0) if logits is not None else latent['logits']
      ss += se

  # Save as .npz files
  np.savez(os.path.join(stat_dir, 'stat.npz'), batch=batch_all, size=size, start=start, pool_3=pools, logit=logits)
  print(pools.shape)
  print(logits.shape)
  print("Finish encoding the samples from the dataset.")
  
  # Encodes the generated samples.
  sample_dir = os.path.join(workdir, "eval_samples")
  latent_dir = os.path.join(workdir, "latent")
  tf.io.gfile.makedirs(latent_dir)
  for c in range(config.classifier.classes):
    print("Encode the images for class {}.".format(c))
    samples = None
    sample_c_dir = os.path.join(sample_dir, str(c))
    latent_c_dir = os.path.join(latent_dir, str(c))
    tf.io.gfile.makedirs(latent_c_dir)
    for file in glob.glob(os.path.join(sample_c_dir, '*.npz')):
      sample = np.load(file)['samples']
      gc.collect()
      latent = evaluation.run_inception_distributed(sample, inception_model, inceptionv3=inceptionv3)
      gc.collect()
      # Save as .npz files
      file_name = file.split('/')[-1].replace(".npz", "")
      np.savez_compressed(os.path.join(latent_c_dir, file_name+'_latent.npz'), pool_3=latent["pool_3"], logits=latent["logits"])
  
  print("Finish encoding the generated samples.")

def evaluate_prdc(config, workdir):
  '''
  This function reads the latent files and the `stat.npz' file, and calculates the P / R / D / C metrics.
  '''
  stat_dir = os.path.join(workdir, "stat")
  latent_dir = os.path.join(workdir, "latent")
  # Load stats files.
  stats = np.load(os.path.join(stat_dir, "stat.npz"))
  all_data_pool = stats['pool_3']
  size = stats['size']
  start = stats['start']
  nearest_k = config.eval.nearest_k

  if config.eval.mode == 'full':
    # Concate the loaded latents and check the shapes are the same.
    pools = None
    for c in range(config.classifier.classes):
      latent_c_dir = os.path.join(latent_dir, str(c))
      for file in glob.glob(os.path.join(latent_c_dir, '*.npz')):
        data = np.load(file)
        pool = data['pool_3']
        pools = np.concatenate((pools, pool), axis=0) if pools is not None else pool
    all_data_pool = all_data_pool.reshape((all_data_pool.shape[0], 2048))
    pools = pools.reshape((pools.shape[0], 2048))[0:all_data_pool.shape[0], :]

    # Compute PRDC metrics.
    metrics = compute_prdc(real_features=all_data_pool, fake_features=pools, nearest_k=nearest_k)
    print("Precision: {:2.2%} || Recall:   {:2.2%}".format(metrics['precision'], metrics['recall']))
    print("Density:   {:2.2%} || Coverage: {:2.2%}".format(metrics['density'], metrics['coverage']))
    
  elif config.eval.mode == 'class':
    avg_p = []
    avg_r = []
    avg_d = []
    avg_c = []
    
    for c in range(config.classifier.classes):
      # Concate the loaded latents and check the shapes are the same.
      pools = None
      latent_c_dir = os.path.join(latent_dir, str(c))
      for file in glob.glob(os.path.join(latent_c_dir, '*.npz')):
        data = np.load(file)
        pool = data['pool_3']
        pools = np.concatenate((pools, pool), axis=0) if pools is not None else pool
      
      data_pool = all_data_pool[int(start[c]):int(start[c]+size[c]), :]
      data_pool = data_pool.reshape((data_pool.shape[0], 2048))
      pools = pools.reshape((pools.shape[0], 2048))[0:int(size[c]), :]

      # Compute P / R / D / C metrics.
      metrics = compute_prdc(real_features=data_pool, fake_features=pools, nearest_k=nearest_k)
      avg_p.append(metrics['precision'])
      avg_r.append(metrics['recall'])
      avg_d.append(metrics['density'])
      avg_c.append(metrics['coverage'])

    # Print average class-wise P / R / D / C metrics.
    print("Avg CW Precision: {:2.2%} || Avg CW Recall:   {:2.2%}".format(np.sum(avg_p)/config.classifier.classes, np.sum(avg_r)/config.classifier.classes))
    print("Avg CW Density:   {:2.2%} || Avg CW Coverage: {:2.2%}".format(np.sum(avg_d)/config.classifier.classes, np.sum(avg_c)/config.classifier.classes))
    print("======================")

    # Save class-wise P / R / D / C metrics.
    with open(os.path.join(workdir, "prdc.txt"),'w') as f:
      f.write(', '.join(avg_p))
      f.write('\n')
      f.write(', '.join(avg_r))
      f.write('\n')
      f.write(', '.join(avg_d))
      f.write('\n')
      f.write(', '.join(avg_c))

  else:
    raise ValueError(f"Mode {config.eval.mode} not recognized.")

def evaluate_fidis(config, workdir):
  '''
  This function reads the latent files and the `stat.npz' file, and calculates the FID / IS metrics.
  '''
  if config.eval.mode == 'full':
    stat_dir = os.path.join(workdir, "stat")
    latent_dir = os.path.join(workdir, "latent")
    # Load stats files
    stats = np.load(os.path.join(stat_dir, 'stat.npz'))
    all_data_pool = stats['pool_3']
    all_sample_pools = None
    all_sample_logits = None
    # Concate the loaded latents and check the shapes are the same.
    for c in range(config.classifier.classes):
      pools = None
      logits = None
      latent_c_dir = os.path.join(latent_dir, str(c))
      for file in glob.glob(os.path.join(latent_c_dir, '*.npz')):
        data = np.load(file)
        pool = data['pool_3']
        logit = data['logits']
        pools = np.concatenate((pools, pool), axis=0) if pools is not None else pool
        logits = np.concatenate((logits, logit), axis=0) if logits is not None else logit
      all_sample_pools = np.concatenate((all_sample_pools, pools), axis=0) if all_sample_pools is not None else pools
      all_sample_logits = np.concatenate((all_sample_logits, logits), axis=0) if all_sample_logits is not None else logits

    all_sample_pools = all_sample_pools[:all_data_pool.shape[0], :]
    all_sample_logits = all_sample_logits[:all_data_pool.shape[0], :]
    
    # Compute FID / IS metrics.
    fid = tfgan.eval.frechet_classifier_distance_from_activations(all_data_pool, all_sample_pools)
    inception_score = tfgan.eval.classifier_score_from_logits(all_sample_logits)
    print("FID: {:.2f} || IS: {:.2f}".format(fid.numpy(), inception_score.numpy()))

  elif config.eval.mode == 'class':
    stat_dir = os.path.join(workdir, "stat")
    latent_dir = os.path.join(workdir, "latent")
    fid_list = []
    is_list = []

    # Load stats files
    stats = np.load(os.path.join(stat_dir, 'stat.npz'))
    size = stats['size']
    start = stats['start']
    all_data_pool = stats['pool_3']
    all_sample_pools = None
    all_sample_logits = None

    for c in range(config.classifier.classes):
      # Concate the loaded latents and check the shapes are the same.
      print("Class {}.".format(c))
      pools = None
      logits = None
      latent_c_dir = os.path.join(latent_dir, str(c))
      data_pools = all_data_pool[int(start[c]):int(start[c]+size[c]), :]

      for file in glob.glob(os.path.join(latent_c_dir, '*.npz')):
        data = np.load(file)
        pool = data['pool_3']
        logit = data['logits']
        pools = np.concatenate((pools, pool), axis=0) if pools is not None else pool
        logits = np.concatenate((logits, logit), axis=0) if logits is not None else logit
      
      # Compute class-wise FID / IS metrics.
      fid = tfgan.eval.frechet_classifier_distance_from_activations(data_pools, pools[0:int(size[c]), :])
      inception_score = tfgan.eval.classifier_score_from_logits(logits[0:int(size[c]), :])
      print("FID: {:.2f} || IS: {:.2f}".format(fid.numpy(), inception_score.numpy()))
      fid_list.append(fid.numpy())
      is_list.append(inception_score.numpy())
      gc.collect()

      all_sample_pools = np.concatenate((all_sample_pools, pools), axis=0) if all_sample_pools is not None else pools
      all_sample_logits = np.concatenate((all_sample_logits, logits), axis=0) if all_sample_logits is not None else logits

    # Compute average class-wise FID / IS metrics.
    print("Avg CW FID: {:.2f} || Avg CW IS: {:.2f}".format(np.sum(fid_list) / config.classifier.classes, np.sum(is_list) / config.classifier.classes))
    # Save class-wise FID / IS metrics.
    with open(os.path.join(workdir, "fid_is.txt"),'w') as f:
      f.write(', '.join(is_list))
      f.write('\n')
      f.write(', '.join(fid_list))

  else:
    raise ValueError(f"Mode {config.eval.mode} not recognized.")

config_flags.DEFINE_config_file("config", None, "Evaluation configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("mode", 'full', "Mode for evaluation: `full' for normal evaluation, `class' for class-wise evaluation.")
flags.DEFINE_boolean("stat", False, "Flag indicates whether to create stats files.")
flags.DEFINE_boolean("latent", False, "Flag indicates whether to create latent files.")
flags.DEFINE_boolean("fidis", False, "Flag indicates whether to calculate FID / IS.")
flags.DEFINE_boolean("prdc", False, "Flag indicates whether to calculate P / R / D / C.")
flags.mark_flags_as_required(["workdir", "config", "mode"])

def main(argv):
  config = FLAGS.config
  workdir = os.path.join('results', FLAGS.workdir)
  tf.io.gfile.makedirs(workdir)
  # Adjust the config file
  config.eval.mode = FLAGS.mode
  # Run the code
  if FLAGS.stat:
    create_stat_files(config, workdir)
  if FLAGS.latent:
    create_latent_files(config, workdir, True)
  if FLAGS.prdc:
    evaluate_prdc(config, workdir)
  if FLAGS.fidis:
    evaluate_fidis(config, workdir)

if __name__ == "__main__":
  app.run(main)