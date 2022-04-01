import pytest

import optax
from sunyata.equinox.archs.map_between_categorical_probabilities_and_hidden_features import ComputeExponentialValues
from sunyata.equinox.layers.linear import Linear
from sunyata.equinox.train.trainer import HiddenBayesianNetTrainer
from jax import random

from sunyata.utils import setup_colab_tpu_or_emulate_it_by_cpus

@pytest.mark.skip(reason="setup tpu cores can not be tested now")
def test_hidden_bayesian_net_trainer():
    # set the directory of your tfrecords files, only support absolute path till now!!
    directory = "/home/fengwf/sunyata/resources/openwebtext2/tfrecords/"

    batch_size_per_core = 2
    sequence_len = 1024
    seed = 1
    dim_categorical_probabilities = 50257
    dim_hidden_features = 128

    trainer = HiddenBayesianNetTrainer(directory, batch_size_per_core, sequence_len, dim_hidden_features,
                                        dim_categorical_probabilities, seed)

    trainer.setup_tpu_cores()
    trainer.setup_data()

    key = random.PRNGKey(seed)
    layers = [Linear(key, in_features=dim_hidden_features, out_features=dim_hidden_features) for _ in range(2)]
    trainer.create_hidden_bayesian_net(layers)

    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    trainer.setup_optimizer(optimizer)

    batch = trainer.tfrecords_dataset.next()
    trainer.train_one_batch(batch)