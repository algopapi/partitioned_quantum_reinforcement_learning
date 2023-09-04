# Hyperparamter Tuning Imports 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from ray import tune

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Computer Modern Roman'
matplotlib.rcParams['text.usetex'] = True


experiment_path_1 = f"~/ray_results_2/ray_results/test_run_3_zlt_nterms_nlayers_4"
tuner = tune.Tuner.restore(experiment_path_1)
result_grid = tuner.get_results()
result_dataframe = result_grid.get_dataframe()

#plt.style.use('fivethirtyeight')

error_free = []
for result in result_grid:
    #print("result config", result.config)
  
    if result.error:
        print("error", result.error)
        continue
    else:
        error_free.append(result)
print("error_free", error_free)


trail_dict = {}
# create a dict for every permutation of zeta and lambda
for result in error_free:
    #theta_lr = result.config["theta_lr"]
    zeta_lr = result.config["zeta_lr"]
    lambda_lr = result.config["lambda_lr"]
    if (zeta_lr, lambda_lr) not in trail_dict:
        trail_dict[(zeta_lr, lambda_lr)] = []
    trail_dict[(zeta_lr, lambda_lr)].append(result)

# print("trail_dict", trail_dict)
# for key, result_grid in trail_dict.items():
#     print("key", key)
#     print('values', result_grid)

#     dataframes = [result.metrics_dataframe for result in result_grid]
#     all_data_1 = pd.concat([df.assign(trail_id=i) for i, df in enumerate(dataframes)])
    
#     mean_data_1 = all_data_1.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
#     mean_data_1.columns = ["mean", "lower_bound", "upper_bound"]
#     mean_data_1.index = mean_data_1.index * 10

#     # Plot the average mean_reward and the 95% confidence interval
#     fig, ax = plt.subplots()
#     mean_data_1["mean"].plot(ax=ax, label="Average")
#     ax.fill_between(mean_data_1.index, mean_data_1["lower_bound"], mean_data_1["upper_bound"], alpha=0.3)
#     ax.set_title(f"Mean Reward for $\zeta$={key[0]}, $\lambda$={key[1]}, num trails = {len(result_grid)}")
#     ax.set_ylabel("Mean Reward")
#     ax.legend()

#     # Add experiment information to the info box
#     # ax.text(1.1, 0.5, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
#     plt.show()
    # Initiate plot outside the loop


# # Remove the previous legend
# ax.get_legend().remove()


# fig, ax = plt.subplots(figsize=(8, 6.5))
# cmap = plt.get_cmap('jet')
# colors = cmap(np.linspace(0, 1.0, len(trail_dict)))

# # create empty list to store table data
# table_data = []

# for color, (key, result_grid) in zip(colors, trail_dict.items()):
#     dataframes = [result.metrics_dataframe for result in result_grid]
#     all_data_1 = pd.concat([df.assign(trail_id=i) for i, df in enumerate(dataframes)])
    
#     mean_data_1 = all_data_1.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
#     mean_data_1.columns = ["mean", "lower_bound", "upper_bound"]
#     mean_data_1.index = mean_data_1.index * 10

#     # Plot the average mean_reward
#     ax.plot(mean_data_1.index, mean_data_1["mean"], color=color)


#     # Add data to table_data list
#     table_data.append((color, key[0], key[1]))
    
# # set labels and title
# ax.set_title("Cartpole Different Parameters")
# ax.set_ylabel("Average Reward")
# ax.set_xlabel("Episode")

# # Set the y limit: lower limit = 0, upper limit = 120% of the max mean value
# ax.set_aspect('auto')

# # Create a subplot for the custom table
# table_ax = fig.add_axes([0.1, 0.643, 0.2, 0.3], frame_on=True)

# # Turn off ticks and labels for the table subplot
# table_ax.xaxis.set_visible(False)
# table_ax.yaxis.set_visible(False)

# # Add colored rectangles for each row in the table
# for i, (color, zeta, lambda_val) in enumerate(table_data):
#     # Create a colored rectangle for the 'color' cell with reduced height
#     color_rect = Rectangle((0, i * 0.8), 1, 0.8, facecolor=color)  # reduce the height of the rectangle
#     table_ax.add_patch(color_rect)

#     # Add text for the 'zeta' and 'lambda' cells with adjusted vertical position
#     table_ax.text(2.6, i * 0.8 + 0.4, f'{zeta}', va='center', ha='center')  # adjust the vertical position
#     table_ax.text(5.8, i * 0.8 + 0.4, f'{lambda_val}', va='center', ha='center')  # adjust the vertical position

# # Add column labels
# table_ax.text(2.6, len(table_data) * 0.8, '$\zeta$', va='bottom', ha='center')
# table_ax.text(5.8, len(table_data) * 0.8, '$\lambda$', va='bottom', ha='center')

# # Set the limits and aspect ratio of the table subplot to fit the data
# table_ax.set_xlim(0, 7.5)
# table_ax.set_ylim(0, len(table_data) * 0.8)  # adjust the y limit))
# table_ax.set_aspect('equal')

# # Adjust the layout of the plot
# fig.tight_layout()
# plt.show()
# fig.savefig("Paramater_sharp.pdf", dpi=300)



######################## SIDE BY SIDE ########################

# Define color maps with shades of red, green, and blue respectively
cmap_red = LinearSegmentedColormap.from_list("custom_cmap", ["#FFC0CB", "#FF0000"])
cmap_blue = LinearSegmentedColormap.from_list("custom_cmap", ["#ADD8E6", "#0000FF"])
cmap_green = LinearSegmentedColormap.from_list("custom_cmap", ["#90EE90", "#008000"])

# Create side by side subplots for different paramters
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))  # Define 3 subplots in a row

# Zeta specific trails
zeta_dict = {}
# create a dict for every permutation of zeta and lambda
for result in error_free:
    #theta_lr = result.config["theta_lr"]
    zeta_lr = result.config["zeta_lr"]
    if (zeta_lr) not in zeta_dict:
        zeta_dict[(zeta_lr)] = []
    zeta_dict[(zeta_lr)].append(result)


layer_dict = {}
for result in error_free:
    n_layer = result.config["n_layers"]
    if (n_layer) not in layer_dict:
        layer_dict[(n_layer)] = []
    layer_dict[(n_layer)].append(result)
# print only the keys
print(layer_dict.keys())

term_dict = {}
for result in error_free:
    n_terms = result.config["n_terms"]
    if (n_terms) not in term_dict:
        term_dict[(n_terms)] = []
    term_dict[(n_terms)].append(result)
# print only the keys
print(term_dict.keys())


# Define a colormap with nicer shades of red, green, and blue
cmap = LinearSegmentedColormap.from_list(
    "custom_cmap", ["#D7191C", "#ABDDA4", "#2B83BA"]
)

# Get the colors from the colormap
colors = cmap_red(np.linspace(0, 1.0, len(zeta_dict)))


# create empty list to store table data
table_data = []

for color, (key, result_grid) in zip(colors, zeta_dict.items()):
    dataframes = [result.metrics_dataframe for result in result_grid]
    all_data_1 = pd.concat([df.assign(trail_id=i) for i, df in enumerate(dataframes)])
    
    mean_data_1 = all_data_1.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
    mean_data_1.columns = ["mean", "lower_bound", "upper_bound"]
    mean_data_1.index = mean_data_1.index * 10

    # Plot the average mean_reward with fill_between for confidence intervals
    axs[0].plot(mean_data_1.index, mean_data_1["mean"], color=color, label=f"Zeta : {key}")  # Adding label
    axs[0].set_title(f"Variable Zeta's")
    axs[0].set_ylabel("Mean Reward")
    axs[0].set_xlabel("Episode")
    #axs[0].legend()

    # Add data to table_data list
    # table_data.append((color, key[0]))
axs[0].legend(loc='upper left')


######################## Lambda specific trails ########################
# Lambda specific trails
zeta_dict = {}
# create a dict for every permutation of zeta and lambda
for result in error_free:
    #theta_lr = result.config["theta_lr"]
    zeta_lr = result.config["lambda_lr"]
    if (zeta_lr) not in zeta_dict:
        zeta_dict[(zeta_lr)] = []
    zeta_dict[(zeta_lr)].append(result)


cmap = plt.get_cmap('jet')
colors = cmap_blue(np.linspace(0, 1.0, len(zeta_dict)))

# create empty list to store table data
table_data = []

for color, (key, result_grid) in zip(colors, zeta_dict.items()):
    dataframes = [result.metrics_dataframe for result in result_grid]
    all_data_1 = pd.concat([df.assign(trail_id=i) for i, df in enumerate(dataframes)])
    
    mean_data_1 = all_data_1.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
    mean_data_1.columns = ["mean", "lower_bound", "upper_bound"]
    mean_data_1.index = mean_data_1.index * 10

    # Plot the average mean_reward
    axs[1].plot(mean_data_1.index, mean_data_1["mean"],color=color, label=f"Lambda : {key}")
    axs[1].set_title(f"Variable Lambda's")
    #axs[1].set_ylabel("Mean Reward")
    axs[1].set_xlabel("Episode")
    # Add data to table_data list
    # table_data.append((color, key[0]))
axs[1].legend(loc='upper left')


######################## Theta specific trails ########################
# Theeta specific trails
zeta_dict = {}
# create a dict for every permutation of zeta and lambda
for result in error_free:
    #theta_lr = result.config["theta_lr"]
    zeta_lr = result.config["theta_lr"]
    if (zeta_lr) not in zeta_dict:
        zeta_dict[(zeta_lr)] = []
    zeta_dict[(zeta_lr)].append(result)


cmap = plt.get_cmap('jet')
colors = cmap_green(np.linspace(0, 1.0, len(zeta_dict)))

# create empty list to store table data
table_data = []

for color, (key, result_grid) in zip(colors, zeta_dict.items()):
    dataframes = [result.metrics_dataframe for result in result_grid]
    all_data_1 = pd.concat([df.assign(trail_id=i) for i, df in enumerate(dataframes)])
    
    mean_data_1 = all_data_1.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
    mean_data_1.columns = ["mean", "lower_bound", "upper_bound"]
    mean_data_1.index = mean_data_1.index * 10

    # Plot the average mean_reward
    axs[2].plot(mean_data_1.index, mean_data_1["mean"], color=color, label=f"Theta : {key}")
    axs[2].set_title(f"Variable Theta's")
    #axs[2].set_ylabel("Mean Reward")
    axs[2].set_xlabel("Episode")
    # Add data to table_data list
    # table_data.append((color, key[0]))
axs[2].legend(loc='upper left')

# normalize y axis
axs[0].set_ylim([0, 450])
axs[1].set_ylim([0, 450])
axs[2].set_ylim([0, 450])
axs[0].set_facecolor('none')
axs[1].set_facecolor('none')
axs[2].set_facecolor('none')

#Remove grids on all plots
axs[0].grid(False)
axs[1].grid(False)
axs[2].grid(False)

fig.tight_layout()
plt.show()
fig.savefig("ParameterSearch.pdf", dpi=300, transparent=True)



# layer_dict = {}
# # create a dict for every permutation of zeta and lambda
# for result in error_free:
#     #theta_lr = result.config["theta_lr"]
#     n_layers = result.config["zeta_lr"]
#     lambda_lr = result.config["lambda_lr"]
#     if (zeta_lr, lambda_lr) not in layer_dict:
#         layer_dict[(zeta_lr, lambda_lr)] = []
#     layer_dict[(zeta_lr, lambda_lr)].append(result)