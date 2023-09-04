# Hyperparamter Tuning Imports 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from ray import tune

experiment_path_1 = "~/ray_results_3/ray_results/variable_layers_and_terms"
tuner = tune.Tuner.restore(experiment_path_1)
result_grid = tuner.get_results()

result_dataframe = result_grid.get_dataframe()

error_free = []
for result in result_grid:
    if result.error:
        print("error", result.error)
        continue
    else:
        error_free.append(result)
print("error_free", error_free)

trail_dict = {}
for result in error_free:
    n_layers = result.config["n_layers"]
    n_terms = result.config["n_terms"]
    if (n_layers, n_terms) not in trail_dict:
        trail_dict[(n_layers, n_terms)] = []
    trail_dict[(n_layers, n_terms)].append(result)

# Initialize the plot outside the loop
fig, ax = plt.subplots()

for key, result_grid in trail_dict.items():
    dataframes = [result.metrics_dataframe for result in result_grid]
    all_data_1 = pd.concat([df.assign(trail_id=i) for i, df in enumerate(dataframes)])
    
    mean_data_1 = all_data_1.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
    mean_data_1.columns = ["mean", "lower_bound", "upper_bound"]
    mean_data_1.index = mean_data_1.index * 10
    
    # Plot the average mean_reward and the 95% confidence interval
    mean_data_1["mean"].plot(ax=ax, label=f"n_layer={key[0]} n_terms={key[1]}")

# Set the plot title, labels, and legend
ax.set_title("Mean Reward")
ax.set_xlabel("Training Iteration")
ax.set_ylabel("Mean Reward")
ax.legend()

plt.show()




trail_dict = {}
for result in error_free:
    n_terms = result.config["n_terms"]
    if (n_terms) not in trail_dict:
        trail_dict[(n_terms)] = []
    trail_dict[(n_terms)].append(result)

# Initialize the plot outside the loop
fig, ax = plt.subplots()

for key, result_grid in trail_dict.items():
    dataframes = [result.metrics_dataframe for result in result_grid]
    all_data_1 = pd.concat([df.assign(trail_id=i) for i, df in enumerate(dataframes)])
    
    mean_data_1 = all_data_1.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
    mean_data_1.columns = ["mean", "lower_bound", "upper_bound"]
    mean_data_1.index = mean_data_1.index * 10

    # Plot the average mean_reward and the 95% confidence interval
    mean_data_1["mean"].plot(ax=ax, label=f"n_terms={key}")

# Set the plot title, labels, and legend
ax.set_title("Mean Reward")
ax.set_xlabel("Training Iteration")
ax.set_ylabel("Mean Reward")
ax.legend()

plt.show()



# Separate results by layers first
layer_dict = {}
for result in error_free:
    n_layers = result.config["n_layers"]
    if n_layers not in layer_dict:
        layer_dict[n_layers] = []
    layer_dict[n_layers].append(result)

# Create subplots for each number of layers
fig, axs = plt.subplots(nrows=1, ncols=len(layer_dict), figsize=(15, 5))  # Adjust figsize to your preference

for ax, (n_layers, results) in zip(axs, layer_dict.items()):
    # Within each layer, separate results by terms
    term_dict = {}
    for result in results:
        n_terms = result.config["n_terms"]
        if n_terms not in term_dict:
            term_dict[n_terms] = []
        term_dict[n_terms].append(result)

    # Plot each term in the current layer's subplot
    for n_terms, term_results in term_dict.items():
        dataframes = [result.metrics_dataframe for result in term_results]
        all_data = pd.concat([df.assign(trail_id=i) for i, df in enumerate(dataframes)])

        mean_data = all_data.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
        mean_data.columns = ["mean", "lower_bound", "upper_bound"]
        mean_data.index = mean_data.index * 10

        ax.fill_between(mean_data.index, mean_data["lower_bound"], mean_data["upper_bound"], alpha=0.3)
        mean_data["mean"].plot(ax=ax, label=f"n_terms={n_terms}")

    # Set the plot title, labels, and legend
    ax.set_title(f"Mean Reward for n_layers={n_layers}")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Mean Reward")
    ax.legend()

plt.tight_layout()
plt.show()