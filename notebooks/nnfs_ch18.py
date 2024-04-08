# %%
import numpy as np

from sunyata.grokking.nnfs import (
    Model,
    Layer_Dense,
    Activation_ReLU,
    Layer_Dropout,
    Activation_Softmax,
    Activation_Sigmoid,
    Loss_CategoricalCrossentropy,
    Loss_BinaryCrossentropy,
    Optimizer_Adam,
    Accuracy_Categorical,
) 

# %%
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
# Create dataset
X, y = spiral_data(samples=1000, classes=2)
X_test, y_test = spiral_data(samples=100, classes=2)

# %%
model = Model()
# %%
model.add(Layer_Dense(
    n_inputs=2,
    n_neurons=512,
    weight_regularizer_l2=5e-4,
    bias_regularizer_l2=5e-4,
))
# %%
model.add(Activation_ReLU())
# %%
model.add(Layer_Dropout(0.1))
# %%
model.add(Layer_Dense(512, 3))
# %%
model.add(Activation_Softmax())
# %%
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Categorical(),
)
# %%
model.finalize()
# %%
model.train(
    X,
    y,
    validation_data=(X_test, y_test),
    epochs=10000,
    print_every=100,
)

#################################################
# %%
n = 30  # length of bianry string
k = 3  # parity of the first k bits
train_size = 900
test_size = 1000
hidden_size = 32
weight_init_scale = 2.
weight_regularizer_l1 = 2e-5
learning_rate = 0.003
beta_1 = .99
beta_2 = .98

unique_binary_strings = set()
while len(unique_binary_strings) < train_size + test_size:
    binary_string = tuple(np.random.randint(2, size=n))
    unique_binary_strings.add(binary_string)

inputs = np.array(list(unique_binary_strings), dtype=np.float32)
outputs = np.sum(inputs[:, :k], axis=-1) % 2
outputs = outputs.reshape(-1, 1)

ones_column = np.ones((inputs.shape[0], 1), dtype=np.float32)
inputs = np.concatenate((inputs, ones_column), axis=1)

indices = np.random.permutation(len(inputs))
split_idx = train_size
train_batch = inputs[indices[:split_idx]], outputs[indices[:split_idx]]
eval_batch = inputs[indices[split_idx:]], outputs[indices[split_idx:]]

# %%
layer_dense_1 = Layer_Dense(
    n_inputs = n + 1, 
    n_neurons=hidden_size,
    with_bias=False,
    weight_init_scale=weight_init_scale,
    weight_regularizer_l1=weight_regularizer_l1,
)

layer_dense_2 = Layer_Dense(
    n_inputs = hidden_size, 
    n_neurons=1,
    with_bias=False,
    weight_init_scale=1.,
    weight_regularizer_l1=weight_regularizer_l1,
)

activation1 = Activation_ReLU()
activation2 = Activation_Sigmoid()

loss_function = Loss_BinaryCrossentropy()

optimizer = Optimizer_Adam(
    learning_rate=learning_rate,
    beta_1=beta_1,
    beta_2=beta_2,
)
# %%
model = Model()
model.add(layer_dense_1)
model.add(activation1)
model.add(layer_dense_2)
model.add(activation2)

model.set(
    loss=loss_function,
    optimizer=optimizer,
    accuracy=Accuracy_Categorical(binary=True),
)

model.finalize()
# %%
model.train(
    train_batch[0],
    train_batch[1],
    validation_data=(eval_batch[0], eval_batch[1]),
    epochs=8000,
    print_every=100,
)


# %%
