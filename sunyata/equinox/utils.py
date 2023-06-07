import jax.numpy as jnp

def setup_colab_tpu_or_emulate_it_by_cpus():
    """prepare tpu environment when running on Colab, otherwise, emulate 8-core tpu by local cpus."""

    import jax.tools.colab_tpu
    try:
        jax.tools.colab_tpu.setup_tpu()
    except Exception as e:
        print("TPU cores doesn't exist: ", e)
        print("Emulate 8 TPU cores using CPUs...")
        import os
        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

    import jax
    print(jax.local_device_count())
    assert jax.local_device_count() == 8
    print("Success gotten 8 TPU cores.")    

    return True


def compute_cross_entropy(predicted_categorical_probabilities: jnp.ndarray, target_categorical_probabilities: jnp.ndarray):
    """Compute the cross entropy between predicated categorical distribution and the true categorical distribution"""
    return - jnp.sum(jnp.log(predicted_categorical_probabilities) * target_categorical_probabilities, axis=-1)


def split_first_dim_of_array_by_core(array: jnp.ndarray, num_of_cores: int):
    return array.reshape((num_of_cores, -1) + array.shape[1:])


# copy from flax.training.common_utils.onehot
def one_hot_encode(labels: jnp.ndarray, dim_categorical_probabilities: int, on_value: float=1.0, off_value: float=0.0):
    x = (labels[..., None] == jnp.arange(dim_categorical_probabilities).reshape((1,) * labels.ndim + (-1,)))
    x = lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
    return x.astype(jnp.float32)