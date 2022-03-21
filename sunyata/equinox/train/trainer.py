from dataclasses import dataclass
from functools import partial
from typing import Iterator, List

import equinox as eqx
import jax
import optax
from jax import random
from sunyata.data import ReadTfrecordsFiles
from sunyata.equinox.archs import (
    ComputeExponentialValues, HiddenBayesianNet,
    MapBetweenCategoricalProbabilitiesAndHiddenFeatures)
from sunyata.equinox.archs.map_between_categorical_probabilities_and_hidden_features import \
    MapValuesToNonNegative
from sunyata.equinox.layers import BayesianIteration
from sunyata.equinox.train.lm import PrePostProcessLM
from sunyata.equinox.utils import (compute_cross_entropy,
                                   split_first_dim_of_array_by_core)
from sunyata.utils import (
    get_all_files_with_specific_filetypes_in_a_directory,
    setup_colab_tpu_or_emulate_it_by_cpus)


@dataclass
class HiddenBayesianNetTrainer:
    directory: str
    batch_size_per_core: int
    sequence_len: int
    dim_hidden_features: int
    dim_categorical_probabilities: int = 50257
    seed: int = 1

    num_of_cores: int = None
    batch_size: int = None

    tfrecords_dataset: Iterator = None
    pre_post_process_lm: PrePostProcessLM = None

    hidden_bayesian_net: HiddenBayesianNet = None

    optimizer = None
    opt_state: list = None

    def create_hidden_bayesian_net(self, layers: List[eqx.Module]):
        self.hidden_bayesian_net = HiddenBayesianNet.create(layers, self.seed, 
                                                self.dim_categorical_probabilities, self.dim_hidden_features)


    def setup_tpu_cores(self):
        setup_colab_tpu_or_emulate_it_by_cpus()

        self.num_of_cores = jax.local_device_count()
        self.batch_size = self.batch_size_per_core * self.num_of_cores

    def setup_data(self, language_model="clm1", dataset="openwebtext2"):
        tfrecords_files = get_all_files_with_specific_filetypes_in_a_directory(self.directory, filetypes=["tfrecords"])

        read_tfrecords_files = ReadTfrecordsFiles(tfrecords_files, self.batch_size, chunk_size=self.sequence_len)
        self.tfrecords_dataset = read_tfrecords_files.read()

        self.pre_post_process_lm = PrePostProcessLM(language_model, dataset, self.dim_categorical_probabilities)

    def setup_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.opt_state = self.optimizer.init(self.hidden_bayesian_net)

    def train(self, num_of_epochs: int):
        for epoch in range(num_of_epochs):
            for batch in self.tfrecords_dataset:
                self.train_one_batch(batch)

    def train_one_batch(self, batch: dict):
        input, target = self.pre_post_process_lm.preprocess(batch)
        shard_input = split_first_dim_of_array_by_core(input, self.num_of_cores)
        shard_target = split_first_dim_of_array_by_core(target, self.num_of_cores)
        replicated_hidden_bayesian_net = jax.device_put_replicated(self.hidden_bayesian_net, jax.local_devices())
        replicated_opt_state = jax.device_put_replicated(self.opt_state, jax.local_devices())
        replicated_hidden_bayesian_net, replicated_opt_state, metrics = self.train_step(
            replicated_hidden_bayesian_net, replicated_opt_state, shard_input, shard_target)
        print(f"Train Loss: {metrics['loss'][0]}")

    @partial(jax.pmap, axis_name="cores")
    def train_step(self, model, opt_state, input, target):
        loss, grads = self.forward(model, input, target)
        grads = jax.lax.pmean(grads, axis_name="cores")
        updates, opt_state = self.optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        metrics = jax.lax.pmean(
            {
                "loss": loss
            },
            axis_name="cores"
        )
        return model, opt_state, metrics

    @partial(eqx.filter_value_and_grad)  # , has_aux=True
    def forward(self, model, input, target):
        output = model(input)
        predicted = self.pre_post_process_lm.postprocess(output)
        losses = compute_cross_entropy(predicted, target)
        loss = losses.mean()
        return loss
