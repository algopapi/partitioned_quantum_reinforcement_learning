# Hyperparamter Tuning Imports 
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from ray import tune
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Computer Modern Roman'
matplotlib.rcParams['text.usetex'] = True


experiment_path_1 = "./results/variable_layers_and_terms/variable_layers_and_terms"
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

scheme_dict = {}

for result in error_free:
    scheme = result.config["rescaling_scheme"]
    if (scheme) not in scheme_dict:
        scheme_dict[(scheme)] = []
    scheme_dict[(scheme)].append(result)
# Initialize figure

fig, ax = plt.subplots(figsize=(8, 6))

# Define custom color schemes and linestyles
color_schemes = [
  "steelblue", "pink", "orange"
]
linestyles = ['-', '--', '-.']  # solid, dashed, dashdot

# Define alpha values for fill_between
alpha_values = [0.7, 0.8, 1]
fill_alpha_values = [0.12, 0.2, 0.15]
zorders = [2, 1, 3]

# Loop over the permutations
# Loop over the permutations
for i, (key, result_grid) in enumerate(scheme_dict.items()):
    print("key", key)
    print('values', result_grid)

    dataframes = [result.metrics_dataframe for result in result_grid]
    all_data_1 = pd.concat([df.assign(trail_id=i) for i, df in enumerate(dataframes)])

    mean_data_1 = all_data_1.groupby("training_iteration")["mean_reward"].agg(["mean", "min", "max"])
    mean_data_1.columns = ["mean", "lower_bound", "upper_bound"]
    mean_data_1.index = mean_data_1.index * 10

    # GPR fitting for min, and max
    kernel = RBF(length_scale=1.3, length_scale_bounds=(1, 100.0))
    X = mean_data_1.index.to_numpy().reshape(-1, 1)

    # Fit GPR for lower_bound
    y_lower = mean_data_1["lower_bound"].to_numpy()
    gpr_lower = GaussianProcessRegressor(kernel=kernel, alpha=0.1).fit(X, y_lower)
    y_lower_smooth = gpr_lower.predict(X)

    # Fit GPR for upper_bound
    y_upper = mean_data_1["upper_bound"].to_numpy()
    gpr_upper = GaussianProcessRegressor(kernel=kernel, alpha=0.0).fit(X, y_upper)
    y_upper_smooth = gpr_upper.predict(X)

    # Plotting
    current_colors = color_schemes[i]
    current_zorder = zorders[i]
    current_fill_alpha = fill_alpha_values[i]
    alpha_value = alpha_values[i]

    # Plot mean as line
    ax.plot(X, mean_data_1["mean"], label=f"{key[0].upper() + key[1:]}", color=current_colors, alpha=alpha_value, linewidth=1.22, zorder=current_zorder)
    
    # Plot confidence interval as filled area
    ax.fill_between(X.flatten(), y_lower_smooth, y_upper_smooth, color=current_colors, alpha=current_fill_alpha, zorder=current_zorder)

    ax.set_ylabel("Mean Reward")
    ax.set_xlabel("Training Iteration")



# Cut off the x-axis at 1200
ax.set_xlim([0, 1200])

plt.legend()
# Plot the figure
plt.show()

# Print the figure to a file
fig.savefig("lambda_schemes.pdf", dpi=300, bbox_inches='tight', transparent=True)

# ax = None
# exp_results = scheme_dict["exponential"] 
# for result in exp_results:
#     print("result")
#     if result.metrics_dataframe is not None:
#         result.metrics_dataframe.plot("training_iteration", "mean_reward", ax=ax)


