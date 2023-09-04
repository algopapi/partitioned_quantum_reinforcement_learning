# Hyperparamter Tuning Imports 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import ray
import seaborn as sns
from ray import air, tune
from ray.air import Result
from ray.air.checkpoint import Checkpoint
from ray.tune import CLIReporter, ExperimentAnalysis, Stopper
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler

#analysis = ExperimentAnalysis("~/ray_results/ray_results/testrun")
#print("best trail",  analysis.get_best_trial(metric="mean_reward", mode="max"))
#print("best result",  analysis.get_best_trial(metric="mean_reward", mode="max").best_result)
#print("best result config",  analysis.get_best_trial(metric="mean_reward", mode="max").config)


experiment_path = f"~/ray_results/ray_results/testrun"
experiment_path_2 = f"~/ray_results/testrun_2"
#experiment_path_3 = f"~/ray_results/ray_results/testrun_3/"
restored_tuner = tune.Tuner.restore(experiment_path)
restored_tuner_2 = tune.Tuner.restore(experiment_path_2)



#restored_tuner_3 = tune.Tuner.restore(experiment_path_3)

result_grid = restored_tuner.get_results()
result_grid_2 = restored_tuner_2.get_results()

best_result = result_grid_2.get_best_result("mean_reward", "max")
print("best_result", best_result)
#result_grid_3 = restored_tuner_3.get_results()

# Check if there have been errors
# if result_grid.errors:
#     print("One of the trials failed!")
# else:
#     print("No errors!")


# for i, result in enumerate(result_grid):
#     if result.error:
#         print(f"Trial #{i} had an error:", result.error)
#         continue
    
#     if "mean_reward" in result.metrics:
#       print(
#           f"Trial #{i} finished successfully with a mean accuracy metric of:",
#           result.metrics["mean_reward"]
#       )

relevant_results = []
for result in result_grid:
    #print("result", result)
    if result.error:
        print("error", result.error)
        continue
    else:
        # add the results with 6 layers and 6 or 10 terms to the dictionary
        # check if result.config is not Nonetype  
        if result.config is not None:
            # then do this
            if result.config["n_layers"] == 6 and (result.config["n_terms"] == 6 or result.config["n_terms"] == 10):
                # append result to dict
                relevant_results.append(result)
            else:
                continue

dataframes = []
for result in relevant_results:
    if hasattr(result, 'metrics_dataframe') and result.metrics_dataframe is not None:
        dataframes.append(result.metrics_dataframe)


# get the average trajectory of all trails in each permuatataion. First loop over the permutations

# Concatenate the individual dataframes into one large dataframe
all_data = pd.concat([df.assign(trial_id=i) for i, df in enumerate(dataframes)])

# Calculate mean and 95% confidence interval
mean_data = all_data.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
mean_data.columns = ["mean", "lower_bound", "upper_bound"]
mean_data.index = mean_data.index * 10

# Plot the mean and the 95% confidence interval
fig, ax = plt.subplots()
mean_data["mean"].plot(ax=ax, label="Average reward")
ax.fill_between(mean_data.index, mean_data["lower_bound"], mean_data["upper_bound"], alpha=0.3)
ax.set_title("Average Reward over Training Iterations")
ax.set_ylabel("Mean Reward")
ax.legend()
plt.show()

plt.legend()
#plot the figure
plt.show()

# results_df = result_grid.get_dataframe()
# results = results_df.dropna(subset=['mean_reward'])

#for index, row in results_df.iterrows():
#    print(f"mean_reward: {row['mean_reward']}, episodes: {row['episode']}, n_layers: {row['config/n_layers']}, n_terms: {row['config/n_terms']},  input_lr: {row['config/input_lr']}, lambda_lr: {row['config/lambda_lr']},   obs_lr: {row['config/obs_lr']}, theta_lr: {row['config/theta_lr']}, zeta_lr: {row['config/zeta_lr']}")

# x_values = results['config/n_layers']
# y_values = results['mean_reward']

# plt.scatter(x_values, y_values)
# plt.xlabel('n_terms')
# plt.ylabel('mean_reward')

# plt.title('Mean Reward vs n_terms')
# plt.show()


# fig = px.scatter_3d(
#     results,
#     x='config/n_layers',
#     y='config/n_terms',
#     z='mean_reward',
#     color='mean_reward',
#     color_continuous_scale=px.colors.diverging.RdYlBu_r,  # Red to Blue colormap
#     range_color=(0, 500),
# )

# fig.update_layout(
#     scene=dict(
#         xaxis_title='n_layers',
#         yaxis_title='n_terms',
#         zaxis_title='mean_reward',
#     )
# )

# fig.show()
# Group the data by 'n_terms'
#grouped_data = results_df.groupby(['config/n_terms', 'episode']).agg({'mean_reward': 'mean'}).reset_index()
#mean_reward_500 = results_df[results_df["mean_reward"]==500]

# for index, row in results_df.iterrows():
#     if row['config/n_layers'] == 10 and row['config/n_terms'] == 10 and row['mean_reward'] >= 400:
#         print(f"mean_reward: {row['mean_reward']}, episodes: {row['episode']}, n_layers: {row['config/n_layers']}, n_terms: {row['config/n_terms']},  input_lr: {row['config/input_lr']}, lambda_lr: {row['config/lambda_lr']},   obs_lr: {row['config/obs_lr']}, theta_lr: {row['config/theta_lr']}, zeta_lr: {row['config/zeta_lr']}")


# fig = px.scatter_3d(
#     results,
#     x='config/n_layers',
#     y='config/n_terms',
#     z='mean_reward',
#     color='mean_reward',
#     color_continuous_scale=px.colors.diverging.RdYlBu_r,  # Red to Blue colormap
#     range_color=(0, 500),
# )

#

#for index, row in mean_reward_500.iterrows():
#    print(f"Trial: {row['trial_id']}, episodes: {row['episode']}, n_layers: {row['config/n_layers']}, n_terms: {row['config/n_terms']},  input_lr: {row['config/input_lr']}, lambda_lr: {row['config/lambda_lr']},   obs_lr: {row['config/obs_lr']}, theta_lr: {row['config/theta_lr']}, zeta_lr: {row['config/zeta_lr']}")

#ax = None
#for result in result_grid_2:
#    print("result", result)
#    #label = f"Trial {result.metrics.trial_id}"
#    if ax is None:
#        ax = result.metrics_dataframe.plot("training_iteration", "mean_reward")
#    else:
#        result.metrics_dataframe.plot("training_iteration", "mean_reward", ax=ax)
#ax.set_title("Mean Accuracy vs. Training Iteration for All Trials")
#ax.set_ylabel("Mean Test Accuracy")

#print("result_grid_2", result_grid_2.param_space)

for result in result_grid_2:
    print("result2", result)
    #label = f"Trial {result.metrics.trial_id}"

dataframes = [result.metrics_dataframe for result in result_grid_2]
all_data_1 = pd.concat([df.assign(trail_id=i) for i, df in enumerate(dataframes)])

mean_data_1 = all_data_1.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
mean_data_1.columns = ["mean", "lower_bound", "upper_bound"]
mean_data_1.index = mean_data_1.index * 10

# Plot the average mean_reward and the 95% confidence interval
fig, ax = plt.subplots()
mean_data_1["mean"].plot(ax=ax, label="Average")
ax.fill_between(mean_data_1.index, mean_data_1["lower_bound"], mean_data_1["upper_bound"], alpha=0.3)
ax.set_title(f"Mean Reward for some experiment")
ax.set_ylabel("Mean Reward")
ax.legend()

# Add experiment information to the info box
# ax.text(1.1, 0.5, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
plt.show()


