import os
import logging
import tensorflow as tf
import torch
import torch.optim as optim
from torch.utils import tensorboard
from absl import flags
from absl import app
from ml_collections.config_flags import config_flags

import datasets
from models import classifier_cas

FLAGS = flags.FLAGS

def training_function(config, workdir):
  """Execute the training procedure for the classifier (for calculating CAS).
  Args:
    config: (dict) Experimental configuration file that specifies the setups and hyper-parameters.
    workdir: (str) Working directory for checkpoints and TF summaries.
  """
  # Create directories for experimental logs.
  tb_dir = os.path.join(workdir, "tensorboard")
  checkpoint_dir = os.path.join(workdir, "checkpoints_cas")
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(tb_dir)
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(sample_dir)

  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize the classifier and the optimizer.
  classifier_model = classifier_cas.classifier_cas(config).to(config.device)
  optimizer = optim.Adam(classifier_model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)

  # Build the data iterators.
  _, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=False, evaluation=True)
  config.data.dataset = config.data.cas_dataset
  train_ds, _, _ = datasets.get_dataset(config, uniform_dequantization=False)
  train_iter = iter(train_ds)
  eval_iter = iter(eval_ds)

  scaler = datasets.get_data_scaler(config)
  
  # Training
  for step in range(config.training.n_iters):
    # Get data and execute one training step.
    data = next(train_iter)
    batch = torch.from_numpy(data['image']._numpy()).to(config.device).float()
    batch = batch.permute(0, 3, 1, 2)
    batch = scaler(batch)
    labels = torch.from_numpy(data['label']._numpy()).to(config.device).long()
    optimizer.zero_grad()
    classifier_model.train()
    pred = classifier_model(batch)
    loss_ce_fn = torch.nn.CrossEntropyLoss()
    loss_ce = loss_ce_fn(pred, labels)
    loss_ce.backward()
    optimizer.step()

    if step % config.training.log_freq == 0:
      logging.info("step: %d, loss_ce: %.5e" % (step, loss_ce.item()))
      writer.add_scalar("loss_ce", loss_ce, step)

    # Report the loss and accuracy periodically
    if step % config.training.eval_freq == 0:
      all_correct = 0
      all_number = 0
      while True:
        try:
          eval_data = next(eval_iter)
        except:
          eval_iter = iter(eval_ds)
          break
        batch = torch.from_numpy(eval_data['image']._numpy()).to(config.device).float()
        batch = batch.permute(0, 3, 1, 2)
        batch = scaler(batch)
        labels = torch.from_numpy(eval_data['label']._numpy()).to(config.device).long()
        classifier_model.eval()
        sm_fn = torch.nn.Softmax(dim=1)
        with torch.no_grad():
          pred = classifier_model(batch)
          pred = sm_fn(pred)
          pred = torch.argmax(pred, dim=1)
        all_correct += (pred == labels).sum()
        all_number += pred.shape[0]

      print("Accuracy: {:2.2%}".format((all_correct/all_number).item()))
      writer.add_scalar("eval_acc", (all_correct/all_number)*100, step)
      torch.save({'model': classifier_model.state_dict(),}, os.path.join(checkpoint_dir, f'checkpoint_{step}.pth'))

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("setup", "base", "The experimental setups. (available choices: `base', `ours', `scale')")
flags.mark_flags_as_required(["workdir", "config"])

def main(argv):
  config = FLAGS.config
  workdir = os.path.join('results', FLAGS.workdir)
  tf.io.gfile.makedirs(workdir)
  # NOTE! The costumized directories should be specified in `sample/samples.py'.
  # NOTE! The generated samples should be placed at `${work_dir}/results/classifier_cifar10_${setup}_resnet18_cond/sample' for cifar-10,
  # and `${work_dir}/results/classifier_cifar100_${setup}_resnet34_cond/sample' for cifar-100.
  print("NOTE! The generated samples should be placed at `\${work_dir}/results/classifier_cifar10_\${setup}_resnet18_cond/sample' for cifar-10, and `\${work_dir}/results/classifier_cifar100_\${setup}_resnet34_cond/sample' for cifar-100.")
  print("NOTE! The costumized directories should be specified in `sample/samples.py'.")
  # Adjust the config file
  config.model.classifier_restore_path = os.path.join(workdir, 'checkpoints/checkpoint.pth')
  # Run the code
  config.data.cas_dataset = "_".join(('samples', FLAGS.setup, config.data.dataset.lower()))
  training_function(config, workdir)

if __name__ == "__main__":
  app.run(main)