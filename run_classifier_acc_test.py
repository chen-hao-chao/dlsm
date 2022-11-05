from absl import flags, app
import os
import tensorflow as tf
import torch
import torch.nn as nn
import datasets

from ml_collections.config_flags import config_flags
from models import ncsnpp, classifier
from models import utils
import sde_lib
import losses
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Evaluation configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("restore_classifier", None, "Path to the checkpoint of a pretrained classifier.")
flags.mark_flags_as_required(["workdir", "config", "restore_classifier"])


def test_classifier_accuracy(config):
    scaler = datasets.get_data_scaler(config)
    classifier_model = utils.create_classifier(config)
    checkpoint = torch.load(config.model.classifier_restore_path)
    classifier_model.load_state_dict(checkpoint['model'])
    
    # Setup SDEs.
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step evaluation functions.
    continuous = config.training.continuous

    
    def acc_fn(classifier_model, batch, labels, eps):
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
        classifier_fn = utils.get_classifier_fn(sde, classifier_model, train=False, continuous=continuous)

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

        return int(correct_num), int(all_num)

    _, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization, evaluation=True)
    
    eps_list = []
    acc_list = []
    eps = 0
    eval_iter = iter(eval_ds)
    for i in range(21):
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
            correct_num, batch_num = acc_fn(classifier_model, eval_batch, eval_labels, eps)
            all_num += batch_num
            all_correct += correct_num
        
        eps_list.append(eps)
        acc_list.append(round(all_correct / all_num, 2))

        print(f"eps: {eps}")
        print("Correct: ", all_correct)
        print("All: ", all_num)
        print("Accuracy: ", all_correct / all_num)
        print("----------------------------------------")
        
        eps += 5e-2
    
    plt.plot(eps_list, acc_list, 'ro')
    plt.xlabel("eps")
    plt.ylabel("accuracy")
    print(eps_list, acc_list)
    plt.savefig("accuracy.png")


def main(argv):
    config = FLAGS.config
    workdir = os.path.join('results', FLAGS.workdir)
    tf.io.gfile.makedirs(workdir)
    config.model.classifier_restore_path = FLAGS.restore_classifier
    test_classifier_accuracy(config)


if __name__=="__main__":
    app.run(main)
