import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf
import numpy as np
import os
import glob

### SAMPLES contains
SAMPLES_IMAGE_SIZE = 32
SAMPLES_IMAGE_SHAPE = (SAMPLES_IMAGE_SIZE, SAMPLES_IMAGE_SIZE, 3)
_NUM_CLASSES = 10
SAMPLES_NUM_POINTS = 50000

SAMPLES_DESCRIPTION = """
The sample dataset.
"""
SAMPLES_CITATION = """ """
SAMPLES_DIR_OURS = "results/classifier_cifar10_ours_resnet18_cond/sample"
SAMPLES_DIR_BASE = "results/classifier_cifar10_base_resnet18_cond/sample"
SAMPLES_DIR_SCALE = "results/classifier_cifar10_base_resnet18_cond_scale10/sample"

_NUM_CLASSES_CIFAR100 = 100
SAMPLES_DIR_OURS_CIFAR100 = "results/classifier_cifar100_ours_resnet34_cond/sample"
SAMPLES_DIR_BASE_CIFAR100 = "results/classifier_cifar100_base_resnet34_cond/sample"
SAMPLES_DIR_SCALE_CIFAR100 = "results/classifier_cifar100_base_resnet34_cond_scale10/sample"

#### Cifar-10 ####

class samples_ours_cifar10(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for SAMPLES dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  You do not have to do anything for the dataset downloading.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=SAMPLES_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=SAMPLES_IMAGE_SHAPE),
            'label': tfds.features.ClassLabel(num_classes=_NUM_CLASSES),
        }),
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='None',
        citation=SAMPLES_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
        
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(
                split='train',
            )),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(
                split='test',
            )),
    ]

  def _generate_examples(self, split='x'):
    """Yields examples."""
    labels = None
    samples = None
    for c in range(_NUM_CLASSES):
      sample_c_dir = os.path.join(SAMPLES_DIR_OURS, str(c))
      batch_sample = None
      for file in glob.glob(os.path.join(sample_c_dir, '*.npz')):
        sample = np.load(file)['samples']
        batch_sample = np.concatenate((batch_sample, sample), axis=0) if batch_sample is not None else sample
      
      samples = np.concatenate((samples, batch_sample), axis=0) if samples is not None else batch_sample
      batch_label = np.ones(batch_sample.shape[0], dtype=np.long)*c
      labels = np.concatenate((labels, batch_label), axis=0) if labels is not None else batch_label

    data = list(zip(samples, labels))
    for index, (sample, label) in enumerate(data):
      record = {"image": sample, "label": label}
      yield index, record


class samples_scale_cifar10(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for SAMPLES dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  You do not have to do anything for the dataset downloading.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=SAMPLES_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=SAMPLES_IMAGE_SHAPE),
            'label': tfds.features.ClassLabel(num_classes=_NUM_CLASSES),
        }),
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='None',
        citation=SAMPLES_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
        
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(
                split='train',
            )),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(
                split='test',
            )),
    ]

  def _generate_examples(self, split='x'):
    """Yields examples."""
    labels = None
    samples = None
    for c in range(_NUM_CLASSES):
      sample_c_dir = os.path.join(SAMPLES_DIR_SCALE, str(c))
      batch_sample = None
      for file in glob.glob(os.path.join(sample_c_dir, '*.npz')):
        sample = np.load(file)['samples']
        batch_sample = np.concatenate((batch_sample, sample), axis=0) if batch_sample is not None else sample
      
      samples = np.concatenate((samples, batch_sample), axis=0) if samples is not None else batch_sample
      batch_label = np.ones(batch_sample.shape[0], dtype=np.long)*c
      labels = np.concatenate((labels, batch_label), axis=0) if labels is not None else batch_label

    data = list(zip(samples, labels))
    for index, (sample, label) in enumerate(data):
      record = {"image": sample, "label": label}
      yield index, record

class samples_base_cifar10(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for SAMPLES dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  You do not have to do anything for the dataset downloading.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=SAMPLES_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=SAMPLES_IMAGE_SHAPE),
            'label': tfds.features.ClassLabel(num_classes=_NUM_CLASSES),
        }),
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='None',
        citation=SAMPLES_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
        
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(
                split='train',
            )),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(
                split='test',
            )),
    ]

  def _generate_examples(self, split='x'):
    """Yields examples."""
    labels = None
    samples = None
    for c in range(_NUM_CLASSES):
      sample_c_dir = os.path.join(SAMPLES_DIR_BASE, str(c))
      batch_sample = None
      for file in glob.glob(os.path.join(sample_c_dir, '*.npz')):
        sample = np.load(file)['samples']
        batch_sample = np.concatenate((batch_sample, sample), axis=0) if batch_sample is not None else sample
      
      samples = np.concatenate((samples, batch_sample), axis=0) if samples is not None else batch_sample
      batch_label = np.ones(batch_sample.shape[0], dtype=np.long)*c
      labels = np.concatenate((labels, batch_label), axis=0) if labels is not None else batch_label

    data = list(zip(samples, labels))
    for index, (sample, label) in enumerate(data):
      record = {"image": sample, "label": label}
      yield index, record


#### Cifar-100 ####

class samples_ours_cifar100(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for SAMPLES dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  You do not have to do anything for the dataset downloading.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=SAMPLES_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=SAMPLES_IMAGE_SHAPE),
            'label': tfds.features.ClassLabel(num_classes=_NUM_CLASSES_CIFAR100),
        }),
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='None',
        citation=SAMPLES_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
        
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(
                split='train',
            )),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(
                split='test',
            )),
    ]

  def _generate_examples(self, split='x'):
    """Yields examples."""
    labels = None
    samples = None
    for c in range(_NUM_CLASSES_CIFAR100):
      sample_c_dir = os.path.join(SAMPLES_DIR_OURS_CIFAR100, str(c))
      batch_sample = None
      for file in glob.glob(os.path.join(sample_c_dir, '*.npz')):
        sample = np.load(file)['samples']
        batch_sample = np.concatenate((batch_sample, sample), axis=0) if batch_sample is not None else sample
      
      samples = np.concatenate((samples, batch_sample), axis=0) if samples is not None else batch_sample
      batch_label = np.ones(batch_sample.shape[0], dtype=np.long)*c
      labels = np.concatenate((labels, batch_label), axis=0) if labels is not None else batch_label

    data = list(zip(samples, labels))
    for index, (sample, label) in enumerate(data):
      record = {"image": sample, "label": label}
      yield index, record


class samples_scale_cifar100(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for SAMPLES dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  You do not have to do anything for the dataset downloading.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=SAMPLES_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=SAMPLES_IMAGE_SHAPE),
            'label': tfds.features.ClassLabel(num_classes=_NUM_CLASSES_CIFAR100),
        }),
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='None',
        citation=SAMPLES_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
        
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(
                split='train',
            )),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(
                split='test',
            )),
    ]

  def _generate_examples(self, split='x'):
    """Yields examples."""
    labels = None
    samples = None
    for c in range(_NUM_CLASSES_CIFAR100):
      sample_c_dir = os.path.join(SAMPLES_DIR_SCALE_CIFAR100, str(c))
      batch_sample = None
      for file in glob.glob(os.path.join(sample_c_dir, '*.npz')):
        sample = np.load(file)['samples']
        batch_sample = np.concatenate((batch_sample, sample), axis=0) if batch_sample is not None else sample
      
      samples = np.concatenate((samples, batch_sample), axis=0) if samples is not None else batch_sample
      batch_label = np.ones(batch_sample.shape[0], dtype=np.long)*c
      labels = np.concatenate((labels, batch_label), axis=0) if labels is not None else batch_label

    data = list(zip(samples, labels))
    for index, (sample, label) in enumerate(data):
      record = {"image": sample, "label": label}
      yield index, record

class samples_base_cifar100(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for SAMPLES dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  You do not have to do anything for the dataset downloading.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=SAMPLES_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=SAMPLES_IMAGE_SHAPE),
            'label': tfds.features.ClassLabel(num_classes=_NUM_CLASSES_CIFAR100),
        }),
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='None',
        citation=SAMPLES_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
        
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(
                split='train',
            )),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(
                split='test',
            )),
    ]

  def _generate_examples(self, split='x'):
    """Yields examples."""
    labels = None
    samples = None
    for c in range(_NUM_CLASSES_CIFAR100):
      sample_c_dir = os.path.join(SAMPLES_DIR_BASE_CIFAR100, str(c))
      batch_sample = None
      for file in glob.glob(os.path.join(sample_c_dir, '*.npz')):
        sample = np.load(file)['samples']
        batch_sample = np.concatenate((batch_sample, sample), axis=0) if batch_sample is not None else sample
      
      samples = np.concatenate((samples, batch_sample), axis=0) if samples is not None else batch_sample
      batch_label = np.ones(batch_sample.shape[0], dtype=np.long)*c
      labels = np.concatenate((labels, batch_label), axis=0) if labels is not None else batch_label

    data = list(zip(samples, labels))
    for index, (sample, label) in enumerate(data):
      record = {"image": sample, "label": label}
      yield index, record