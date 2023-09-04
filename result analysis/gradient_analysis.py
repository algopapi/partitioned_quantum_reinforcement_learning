# load the weights into the model
import os
import pickle

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm
from model_gen import Part_ReuploadingPQC, ReuploadingPQC
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Computer Modern Roman'
matplotlib.rcParams['text.usetex'] = True

plt.style.use("fivethirtyeight")

normalization = tf.keras.layers.Normalization(axis=-1)

# Create directory if it doesn't exist
directory = "ga_outputs"
if not os.path.exists(directory):
    os.makedirs(directory)

print(os.getcwd())
# Define the environment
env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="human")
state_space = env.observation_space.shape[0]

n_partitions = 2

# Define some constants
n_qubits = int(state_space/n_partitions)
n_terms = 20
n_layers = 5
input_dim = 1,
n_actions = env.action_space.n
beta = 1
x_l = False
trainable_lambdas = True
 
partitioned = True

model = None

if partitioned:
    model = Part_ReuploadingPQC( 
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_actions=n_actions,
        n_partitions=n_partitions,
        n_terms=n_terms,
        input_dim=input_dim,
        beta=beta,
        trainable_lambdas=trainable_lambdas,
        rescaling_scheme="fraction"
    )
    print("building partitioned")
    model.build(input_shape=(None, n_qubits*n_partitions))
else:
    model = ReuploadingPQC(
        n_qubits=4,
        n_layers=5,
        n_actions=2,
        beta=beta,
    )
    print("building unpartitioned")
    model.build(input_shape=(None, 4))
# build the model


# Define the environment
rescale_input = np.array([2.4, 2.5, 0.21, 2.5])
input_tensor = tf.convert_to_tensor(rescale_input)

# Load the weights of the model
# The naming convention is as follows:
# PartitionedPQC Policy GradientCartPole-v1{n-terms}terms.h5
# These terms should match the n_terms defined above
model.load_weights("PartitionedPQC Policy GradientCartPole-v120terms.h5")


# This code does a foul sweer of our policy:
# These are the carpole dimensions and their respective bounds
# 0: Cart Position - min 4,8 max 4,8
# 1: Cart Velocity - -0.418 0.418
# 2: Pole Angle min -3 , max 3
# 3: Pole Velocit: min -3 max 3
# Create a full sweep that goes through the 6 permutations of
# two dimensional combinations so: 
# (pole angle, cart position) (pole angle, cart velocity) (pole angle, pole velocity)
# for each sweep, set the other two dimensions both to 0. 
# then plot in a 3d plot the output of the model for each of these sweeps.# CartPole state space dimensions and their respective bounds
dim_names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Velocity"]
dim_bounds = [(-2.4, 2.4), (-2.5, 2.5), (-0.21, 0.21), (-2.5, 2.5)]
state_bounds = np.array([2.4, 2.5, 0.21, 2.5])  # normalization bounds

# Prepare a grid for each pair of dimensions
pairs = [(0, 2), (1, 2), (2, 3)]
grid_size = 50 # number of points in each dimension

# Prepare 3D plots
figs, axs = plt.subplots(nrows=1, ncols=3, subplot_kw={'projection':'3d'}, figsize=(20, 6))
for ax, (dim1, dim2) in zip(axs, pairs):
    # Prepare a grid of states
    grid1 = np.linspace(*dim_bounds[dim1], grid_size)
    grid2 = np.linspace(*dim_bounds[dim2], grid_size)

    print("grid1: ", grid1[:5])
    print("grid2: ", grid2[:5])

    states = np.zeros((grid_size, grid_size, 4))
    states[..., dim1] = grid1[:, None]
    states[..., dim2] = grid2[None, :]

    print('states: ', states)

    # Check if the model's output for this pair is saved
    pickle_file = os.path.join(directory, f"outputs_dim{dim1}_dim{dim2}_terms{n_terms}.pickle")
    print("pickle file: ", pickle_file)
    try:
        with open(pickle_file, "rb") as file:
            outputs = pickle.load(file)
    except FileNotFoundError:
        print("File not found, calculating outputs...")
        # Fetch the pre-softmax output for each state
        outputs = np.empty((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                state = states[i, j]
                state_norm = state / state_bounds  # normalize the state
                print("state norm: ", state_norm)
                input_tensor = tf.convert_to_tensor(state_norm)
                input_tensor = tf.expand_dims(input_tensor, 0) # adjust dimensions for the model
                output = model.call_presoftmax(input_tensor)
                outputs[i, j] = output[0, 0].numpy() # adjust dimensions according to your model output

        # Save the model's output
        with open(pickle_file, "wb") as file:
            pickle.dump(outputs, file)
          
    # Plot the outputs
    X, Y = np.meshgrid(grid1, grid2)

    print("X: ", X[:5])
    print("Y: ", Y[:5])
    
    print("x label", dim_names[dim1])
    print("y label", dim_names[dim2])

    surf = ax.plot_surface(X, Y, outputs, cmap='viridis')
    ax.set_xlabel(dim_names[dim1]) # (0 )
    ax.set_ylabel(dim_names[dim2])
    ax.set_zlabel('Model Output')
    ax.set_title(f'Sweeping "{dim_names[dim1]}" and "{dim_names[dim2]}"')
    figs.colorbar(surf, ax=ax, shrink=0, aspect=5) # add a color bar

plt.tight_layout()
plt.show()


# Increase the default font size
matplotlib.rcParams.update({'font.size': 14})


# Define normalization layer
model = ReuploadingPQC(
    n_qubits=4, 
    n_layers=5, 
    n_actions=2, 
    beta=beta
)
model.build(input_shape=(None, 4))
model.load_weights(f"PartitionedPQC Policy GradientCartPole-v10terms.h5")
# create a tf.keras.layers.Normalization layer for each (dim1, dim2) pair

outputs_unpartitioned =[0, 0, 0]
for ax_idx, (dim1, dim2) in enumerate(pairs):
    # Prepare a grid of states
    grid1 = np.linspace(*dim_bounds[dim1], grid_size)
    grid2 = np.linspace(*dim_bounds[dim2], grid_size)
    states = np.zeros((grid_size, grid_size, 4))
    states[..., dim1] = grid1[:, None]
    states[..., dim2] = grid2[None, :]

    # load the outputs from pickle
    pickle_file = os.path.join(directory, f"outputs_dim{dim1}_dim{dim2}_terms0.pickle")
    print("pickle file: ", pickle_file)
    try:
        with open(pickle_file, "rb") as file:
            outputs = pickle.load(file)
    except FileNotFoundError:
        print("File not found, calculating outputs...")

    # Outputs is a (50, 50) shape array of outputs for each grid in the state space. 
    # I want to normalize the outputs of other models to the same magnitude as this one so i can do a 
    # mse comparion. 
    scaler = MinMaxScaler().fit(outputs.reshape(-1, 1))
    outputs_unpartitioned[ax_idx] = scaler.transform(outputs.reshape(-1, 1)).reshape(outputs.shape)


terms = [1, 3, 6, 20, 0]
figs, axs = plt.subplots(nrows=len(terms), ncols=3, subplot_kw={'projection':'3d'}, figsize=(20, 6*len(terms)))


for term_idx, n_terms in enumerate(terms):
    mse = 0
    if n_terms is not 0:
        model = Part_ReuploadingPQC(
            n_qubits=2,
            n_layers=5,
            n_actions=2,
            n_partitions=2,
            n_terms=n_terms,
            input_dim=1,
            beta=1,
            trainable_lambdas=trainable_lambdas,
            rescaling_scheme="fraction"
        )
    else :
        model = ReuploadingPQC(
            n_qubits=4,
            n_layers=5,
            n_actions=2,
            beta=beta,
        )
    model.build(input_shape=(None, 4))
    
    # load the weights of the model
    model.load_weights(f"PartitionedPQC Policy GradientCartPole-v1{n_terms}terms.h5")
    if n_terms is not 0:
        lambdas = model.partitioned_pqc.lambdas.numpy()
        observable_weights = model.alternating.w.numpy()
    # print the lambda paramater of the model
    print("final lambda distribution:", lambdas)
    print("Observable weights distribution:", observable_weights)
   
    for ax_idx, (dim1, dim2) in enumerate(pairs):
        # Prepare a grid of states
        grid1 = np.linspace(*dim_bounds[dim1], grid_size)
        grid2 = np.linspace(*dim_bounds[dim2], grid_size)
        states = np.zeros((grid_size, grid_size, 4))
        states[..., dim1] = grid1[:, None]
        states[..., dim2] = grid2[None, :]

        # Check if the model's output for this pair is saved
        pickle_file = os.path.join(directory, f"outputs_dim{dim1}_dim{dim2}_terms{n_terms}.pickle")
        #print("pickle file: ", pickle_file)
        try:
            with open(pickle_file, "rb") as file:
                outputs = pickle.load(file) # theses outputs need to be normalized so i can compare them

        except FileNotFoundError:
            print("File not found, calculating outputs...")
  
        #outputs = layer(outputs)
        #print("outputs post norm", outputs[0][:4])
        scaler = MinMaxScaler().fit(outputs.reshape(-1, 1))
        outputs = scaler.transform(outputs.reshape(-1, 1)).reshape(outputs.shape)

        mse += mean_squared_error(outputs_unpartitioned[ax_idx], outputs)
        print(f"MSE of term {n_terms} with unpartitioned: {mse}")


        X, Y = np.meshgrid(grid1, grid2)
        surf = axs[term_idx, ax_idx].plot_surface(X, Y, outputs, cmap='viridis', antialiased=True)
        axs[term_idx, ax_idx].set_xlabel(dim_names[dim1], fontsize=20, labelpad=8)
        axs[term_idx, ax_idx].set_ylabel(dim_names[dim2], fontsize=20, labelpad=8)
        axs[term_idx, ax_idx].set_zlabel('Model Output', fontsize=20, labelpad=8)

        axs[term_idx, ax_idx].xaxis.pane.fill = False
        axs[term_idx, ax_idx].yaxis.pane.fill = False
        axs[term_idx, ax_idx].zaxis.pane.fill = False

        axs[term_idx, ax_idx].xaxis.pane.set_edgecolor('w')
        axs[term_idx, ax_idx].yaxis.pane.set_edgecolor('w')
        axs[term_idx, ax_idx].zaxis.pane.set_edgecolor('w')
        #axs[term_idx, ax_idx].text2D(0.05, 0.95, f"MSE: {mse:.2f}", transform=axs[term_idx, ax_idx].transAxes, zorder=5)
      
        # Add a title to the middle plot of each row
        if ax_idx == 1:
            if n_terms == 0:
                axs[term_idx, ax_idx].set_title(r'\textbf{' + f'Unpartitioned Circuit' + '}', fontsize=24)
            else: 
                axs[term_idx, ax_idx].set_title(r'\textbf{' + f'Terms: {n_terms}' + '}', fontsize=24)
        if ax_idx == 2: 
            if n_terms is not 0:
                # round mse to 3 decimal places
                mse = round(mse, 3)
                axs[term_idx, ax_idx].set_title(f'Total MSE {mse}', fontsize=20)


plt.tight_layout()
plt.show()
figs.savefig("Policy Comparison.pdf", dpi=300, transparent=True)





























# max_episodes = 10
# for e in range(max_episodes):
#     state = env.reset()
#     state = state[0]
#     #state = np.reshape(state, [1, state_space])
#     done, truncated = False, False
#     i = 0

    
#     while not (done or truncated):
#         env.render()
#         #print("state",state)
#         normalized_state = state / state_bounds
#         #print("normalized state", normalized_state)
#         normalized_tensor = tf.convert_to_tensor(normalized_state)
        
#         expended_tensor = tf.expand_dims(normalized_tensor, 0)
#         policy = model(expended_tensor)
#         #print(policy.numpy()[0])
#         action = np.random.choice(n_actions, p=policy.numpy()[0])
#         state, reward, done, truncated, _ = env.step(action)
#         # state = np.reshape(next_state, [1, state_space])
#         i += 1
#         if (done or truncated):
#             print("episode: {}/{}, score: {}".format(e, max_episodes, i))
#             break