# Hyperparamter Tuning Imports 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from ray import air, tune
from ray.air import Result
from ray.air.checkpoint import Checkpoint
from ray.tune import CLIReporter, ExperimentAnalysis, Stopper
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Computer Modern Roman'
matplotlib.rcParams['text.usetex'] = True

experiment_path_1 = f"~/ray_results/testrun_1_term_correct"
experiment_path_1_reshuffled = f"~/ray_results/reshuffled_inputs"
experiment_path_3 = f"~/ray_results/testrun_3_term_correct"
experiment_path_6 = f"~/ray_results/testrun_6_term_correct"
experiment_path_6_2 = f"~/ray_results/testrun_6_term_correct_fixed"
experiment_path_10 = f"~/ray_results/testrun_10_term_correct_fixed"
experiment_path_20 = f"~/ray_results/test_run_20_terms"

restored_tuner_1 = tune.Tuner.restore(experiment_path_1)
restored_tunder_1_alt = tune.Tuner.restore(experiment_path_1_reshuffled)
restored_tuner_3 = tune.Tuner.restore(experiment_path_3)
restored_tuner_6 = tune.Tuner.restore(experiment_path_6)
restored_tuner_6_2 = tune.Tuner.restore(experiment_path_6_2)
restored_tuner_10 = tune.Tuner.restore(experiment_path_10)
restored_tuner_20 = tune.Tuner.restore(experiment_path_20)


result_grid_1 = restored_tuner_1.get_results()
result_grid_1_alt = restored_tunder_1_alt.get_results()
result_grid_3 = restored_tuner_3.get_results()
result_grid_6 = restored_tuner_6.get_results()
result_grid_6_2 = restored_tuner_6_2.get_results()
result_grid_10 = restored_tuner_10.get_results()
result_grid_20 = restored_tuner_20.get_results()


best_result_1 = result_grid_1.get_best_result("mean_reward", mode="max")
best_result_3 = result_grid_3.get_best_result("mean_reward", mode="max")
best_result_6 = result_grid_6.get_best_result("mean_reward", mode="max")
best_result_6_2 = result_grid_6_2.get_best_result("mean_reward", mode="max")
best_result_10 = result_grid_10.get_best_result("mean_reward", mode="max")
best_result_20 = result_grid_20.get_best_result("mean_reward", mode="max")


print("1 term config" , best_result_1.config)
print("3 term config" , best_result_3.config)
print("6 term  config" , best_result_6.config)
print("6 term fixed config" , best_result_6_2.config)
print("10 term configconfig" , best_result_10.config)
print("20 term config" , best_result_20.config)


### 1 terms #############################################################
ax = None
for result in result_grid_1:
   if ax is None:
       ax = result.metrics_dataframe.plot("training_iteration", "mean_reward")
   else:
       result.metrics_dataframe.plot("training_iteration", "mean_reward", ax=ax)
ax.set_title("1 Term individual learning Curves")
ax.set_ylabel("Mean Test Accuracy")
plt.show()

# Collect all the dataframes in a list
dataframes = [result.metrics_dataframe for result in result_grid_1]

# Concatenate all the dataframes, adding a column for trial_id
all_data_1 = pd.concat([df.assign(trial_id=i) for i, df in enumerate(dataframes)])

# Calculate the mean and 2.5% and 97.5% quantiles for each training_iteration
mean_data_1 = all_data_1.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
mean_data_1.columns = ["mean", "lower_bound", "upper_bound"]
mean_data_1.index = mean_data_1.index * 10


# Collect all the dataframes in a list
dataframes_alt = [result.metrics_dataframe for result in result_grid_1_alt]

# Concatenate all the dataframes, adding a column for trial_id
all_data_1_alt = pd.concat([df.assign(trial_id=i) for i, df in enumerate(dataframes_alt)])

# Calculate the mean and 2.5% and 97.5% quantiles for each training_iteration
mean_data_1_alt = all_data_1_alt.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
mean_data_1_alt.columns = ["mean", "lower_bound", "upper_bound"]
mean_data_1_alt.index = mean_data_1_alt.index * 10


# Plot the average mean_reward and the 95% confidence interval
fig, ax = plt.subplots()
mean_data_1["mean"].plot(ax=ax, label=r"$[x, \dot{x}, \theta, \dot{\theta} ]$")
mean_data_1_alt["mean"].plot(ax=ax, label=r"$[x, \dot{\theta}, \theta, \dot{x} ]$")
ax.fill_between(mean_data_1.index, mean_data_1["lower_bound"], mean_data_1["upper_bound"], alpha=0.3)
ax.fill_between(mean_data_1_alt.index, mean_data_1_alt["lower_bound"], mean_data_1_alt["upper_bound"], alpha=0.3)
#ax.set_title("1 Term Average Learning Curve")
ax.set_ylabel("Mean Reward")
ax.set_xlabel("Episode")
ax.legend()

# Add experiment information to the info box
experiment_info = f"""Experiment Information:
- Number of trials: {len(dataframes)}
- Batch dim: {best_result_1.config["batch_size"]}
- layers: {best_result_1.config["n_layers"]}
- terms: {best_result_1.config["n_terms"]}
- theta lr: {best_result_1.config["theta_lr"]}
- zeta lr: {best_result_1.config["zeta_lr"]}
- lambda lr: {best_result_1.config["lambda_lr"]}
- input lr: {best_result_1.config["input_lr"]}
- obs lr: {best_result_1.config["obs_lr"]}
"""
#ax.text(1.1, 0.5, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
plt.show()
fig.savefig("Reshuffled.pdf", dpi=400, transparent=True)

### 3 terms #############################################################
ax = None
for result in result_grid_3:
   if ax is None:
       ax = result.metrics_dataframe.plot("training_iteration", "mean_reward")
   else:
       result.metrics_dataframe.plot("training_iteration", "mean_reward", ax=ax)
ax.set_title("3 Term individual learning curves")
ax.set_ylabel("Mean Test Accuracy")
plt.show()   

# Collect all the dataframes in a list
dataframes_2= [result.metrics_dataframe for result in result_grid_3]

# Concatenate all the dataframes, adding a column for trial_id
all_data_2 = pd.concat([df.assign(trial_id=i) for i, df in enumerate(dataframes_2)])

# Calculate the mean and 2.5% and 97.5% quantiles for each training_iteration
mean_data_2 = all_data_2.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
mean_data_2.columns = ["mean", "lower_bound", "upper_bound"]
mean_data_2.index = mean_data_2.index * 10

# Plot the average mean_reward and the 95% confidence interval
fig, ax = plt.subplots()
mean_data_2["mean"].plot(ax=ax, label="Average")
ax.fill_between(mean_data_2.index, mean_data_2["lower_bound"], mean_data_2["upper_bound"], alpha=0.3)
ax.set_title("3 Term Average Learning Curve")
ax.set_ylabel("Mean Reward")
ax.legend()

# Add experiment information to the info box
experiment_info = f"""Experiment Information:
- Number of trials: {len(dataframes)}
- Batch dim: {best_result_3.config["batch_size"]}
- layers: {best_result_3.config["n_layers"]}
- terms: {best_result_3.config["n_terms"]}
- theta lr: {best_result_3.config["theta_lr"]}
- zeta lr: {best_result_3.config["zeta_lr"]}
- lambda lr: {best_result_3.config["lambda_lr"]}
- input lr: {best_result_3.config["input_lr"]}
- obs lr: {best_result_3.config["obs_lr"]}
"""
ax.text(1.1, 0.5, experiment_info, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
plt.show()

### 6 terms #############################################################
ax = None
for i, result in enumerate(result_grid_6):
    if ax is None:
        ax = result.metrics_dataframe.plot("training_iteration", "mean_reward")
    else:
        result.metrics_dataframe.plot("training_iteration", "mean_reward", ax=ax)
ax.set_title("6 Term Runs")
ax.set_ylabel("Mean Test Accuracy")
plt.show()


# Collect all the dataframes in a list
dataframes_3= [result.metrics_dataframe for result in result_grid_6]

# Concatenate all the dataframes, adding a column for trial_id
all_data_3 = pd.concat([df.assign(trial_id=i) for i, df in enumerate(dataframes_3)])

# Calculate the mean and 2.5% and 97.5% quantiles for each training_iteration
mean_data_3 = all_data_3.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
mean_data_3.columns = ["mean", "lower_bound", "upper_bound"]
mean_data_3.index = mean_data_3.index * 10

# Plot the average mean_reward and the 95% confidence interval
fig, ax = plt.subplots()
mean_data_3["mean"].plot(ax=ax, label="Average")
ax.fill_between(mean_data_3.index, mean_data_3["lower_bound"], mean_data_3["upper_bound"], alpha=0.3)
ax.set_title("6 Term Average Learning Curve")
ax.set_ylabel("Mean Reward")
ax.legend()

# Add experiment information to the info box
experiment_info = f"""Experiment Information:
- Number of trials: {len(dataframes)}
- Batch dim: {best_result_6.config["batch_size"]}
- layers: {best_result_6.config["n_layers"]}
- terms: {best_result_6.config["n_terms"]}
- theta lr: {best_result_6.config["theta_lr"]}
- zeta lr: {best_result_6.config["zeta_lr"]}
- lambda lr: {best_result_6.config["lambda_lr"]}
- input lr: {best_result_6.config["input_lr"]}
- obs lr: {best_result_6.config["obs_lr"]}
"""
ax.text(1.1, 0.5, experiment_info, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
plt.show()

### 6 terms Fixed #############################################################
ax = None
for i, result in enumerate(result_grid_6_2):
    if ax is None:
        ax = result.metrics_dataframe.plot("training_iteration", "mean_reward")
    else:
        result.metrics_dataframe.plot("training_iteration", "mean_reward", ax=ax)
ax.set_title("6 Term Fixed Runs")
ax.set_ylabel("Mean Test Accuracy")
plt.show()


# Collect all the dataframes in a list
dataframes_3_2 = [result.metrics_dataframe for result in result_grid_6_2]

# Concatenate all the dataframes, adding a column for trial_id
all_data_3_2 = pd.concat([df.assign(trial_id=i) for i, df in enumerate(dataframes_3_2)])

# Calculate the mean and 2.5% and 97.5% quantiles for each training_iteration
mean_data_3_2 = all_data_3_2.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
mean_data_3_2.columns = ["mean", "lower_bound", "upper_bound"]
mean_data_3_2.index = mean_data_3_2.index * 10

# Plot the average mean_reward and the 95% confidence interval
fig, ax = plt.subplots()
mean_data_3_2["mean"].plot(ax=ax, label="Average")
ax.fill_between(mean_data_3_2.index, mean_data_3_2["lower_bound"], mean_data_3_2["upper_bound"], alpha=0.3)
ax.set_title("6 Term Fixed Average Learning Curve")
ax.set_ylabel("Mean Reward")
ax.legend()

# Add experiment information to the info box
experiment_info = f"""Experiment Information:
- Number of trials: {len(dataframes)}
- Batch dim: {best_result_6_2.config["batch_size"]}
- layers: {best_result_6_2.config["n_layers"]}
- terms: {best_result_6_2.config["n_terms"]}
- theta lr: {best_result_6_2.config["theta_lr"]}
- zeta lr: {best_result_6_2.config["zeta_lr"]}
- lambda lr: {best_result_6_2.config["lambda_lr"]}
- input lr: {best_result_6_2.config["input_lr"]}
- obs lr: {best_result_6_2.config["obs_lr"]}
"""
ax.text(1.1, 0.5, experiment_info, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
plt.show()



### 10 Terms #############################################################
ax = None
i = 0
print("result_grid_6", result_grid_10)
for i, result in enumerate(result_grid_10):
  # if i is 0, 2, 4
  print("result", result)
  if ax is None:
      ax = result.metrics_dataframe.plot("training_iteration", "mean_reward")
  else:
      result.metrics_dataframe.plot("training_iteration", "mean_reward", ax=ax)
ax.set_title("10 Terms individual runs")
ax.set_ylabel("Mean Test Accuracy")
plt.show()


# Collect all the dataframes in a list
dataframes_4= [result.metrics_dataframe for result in result_grid_10]

# Concatenate all the dataframes, adding a column for trial_id
all_data_4 = pd.concat([df.assign(trial_id=i) for i, df in enumerate(dataframes_4)])

# Calculate the mean and 2.5% and 97.5% quantiles for each training_iteration
mean_data_4 = all_data_4.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
mean_data_4.columns = ["mean", "lower_bound", "upper_bound"]
mean_data_4.index = mean_data_4.index * 10

# Plot the average mean_reward and the 95% confidence interval
fig, ax = plt.subplots()
mean_data_4["mean"].plot(ax=ax, label="Average")
ax.fill_between(mean_data_4.index, mean_data_4["lower_bound"], mean_data_4["upper_bound"], alpha=0.3)
ax.set_title("10 Terms Average Learning Curve")
ax.set_ylabel("Mean Reward")
ax.legend()

# Add experiment information to the info box
experiment_info = f"""Experiment Information:
- Number of trials: {len(dataframes)}
- Batch dim: {best_result_10.config["batch_size"]}
- layers: {best_result_10.config["n_layers"]}
- terms: {best_result_10.config["n_terms"]}
- theta lr: {best_result_10.config["theta_lr"]}
- zeta lr: {best_result_10.config["zeta_lr"]}
- lambda lr: {best_result_10.config["lambda_lr"]}
- input lr: {best_result_10.config["input_lr"]}
- obs lr: {best_result_10.config["obs_lr"]}
"""
ax.text(1.1, 0.5, experiment_info, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
plt.show()

### 20 Terms #############################################################
ax = None
i = 0
print("result_grid_6", result_grid_20)
for i, result in enumerate(result_grid_20):
  # if i is 0, 2, 4
  print("result", result)
  if ax is None:
      ax = result.metrics_dataframe.plot("training_iteration", "mean_reward")
  else:
      result.metrics_dataframe.plot("training_iteration", "mean_reward", ax=ax)
ax.set_title("20 Terms individual runs")
ax.set_ylabel("Mean Test Accuracy")
plt.show()


# Collect all the dataframes in a list
dataframes_5= [result.metrics_dataframe for result in result_grid_20]

# Concatenate all the dataframes, adding a column for trial_id
all_data_5 = pd.concat([df.assign(trial_id=i) for i, df in enumerate(dataframes_5)])

# Calculate the mean and 2.5% and 97.5% quantiles for each training_iteration
mean_data_5 = all_data_5.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
mean_data_5.columns = ["mean", "lower_bound", "upper_bound"]
mean_data_5.index = mean_data_5.index * 10

# Plot the average mean_reward and the 95% confidence interval
fig, ax = plt.subplots()
mean_data_4["mean"].plot(ax=ax, label="Average")
ax.fill_between(mean_data_4.index, mean_data_4["lower_bound"], mean_data_4["upper_bound"], alpha=0.3)
ax.set_title("20 Terms Average Learning Curve")
ax.set_ylabel("Mean Reward")
ax.legend()

# Add experiment information to the info box
experiment_info = f"""Experiment Information:
- Number of trials: {len(dataframes)}
- Batch dim: {best_result_20.config["batch_size"]}
- layers: {best_result_20.config["n_layers"]}
- terms: {best_result_20.config["n_terms"]}
- theta lr: {best_result_20.config["theta_lr"]}
- zeta lr: {best_result_20.config["zeta_lr"]}
- lambda lr: {best_result_20.config["lambda_lr"]}
- input lr: {best_result_20.config["input_lr"]}
- obs lr: {best_result_20.config["obs_lr"]}
"""
ax.text(1.1, 0.5, experiment_info, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
plt.show()


# Create a new figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot all the average mean_rewards and their respective 95% confidence intervals
mean_data_1["mean"].plot(ax=ax, label="1 term", color='blue', alpha=0.8)
ax.fill_between(mean_data_1.index, mean_data_1["lower_bound"], mean_data_1["upper_bound"], color='blue', alpha=0.07)

mean_data_2["mean"].plot(ax=ax, label="3 terms", color='red',  alpha=0.85)
ax.fill_between(mean_data_2.index, mean_data_2["lower_bound"], mean_data_2["upper_bound"], color='red', alpha=0.08)

# mean_data_3["mean"].plot(ax=ax, label="6 terms", color='green')
# ax.fill_between(mean_data_3.index, mean_data_3["lower_bound"], mean_data_3["upper_bound"], color='green', alpha=0.1)

mean_data_4["mean"].plot(ax=ax, label="10 terms", color='green', alpha=0.9)
ax.fill_between(mean_data_4.index, mean_data_4["lower_bound"], mean_data_4["upper_bound"], color='green', alpha=0.09)

mean_data_5["mean"].plot(ax=ax, label="20 terms", color='orange', alpha=0.95)
ax.fill_between(mean_data_5.index, mean_data_5["lower_bound"], mean_data_5["upper_bound"], color='orange', alpha=0.1)

# Add title, label, and legend to the plot
#ax.set_title("Comparison of Test Runs with Different Terms")
ax.set_ylabel("Mean Reward")
ax.set_xlabel("Episode")
ax.legend()
plt.show()
fig.savefig("TermComparison.pdf", dpi=400, transparent=True)

# Create a new figure that shows 6 terms fixed vs 6 term cloud plots in 1 plot
fig, ax = plt.subplots()

mean_data_3["mean"].plot(ax=ax, label="Average 6 terms", color='red')
ax.fill_between(mean_data_3.index, mean_data_3["lower_bound"], mean_data_3["upper_bound"], color='red', alpha=0.1)

mean_data_3_2["mean"].plot(ax=ax, label="Average 6 Fixed terms", color='blue')
ax.fill_between(mean_data_3_2.index, mean_data_3_2["lower_bound"], mean_data_3_2["upper_bound"], color='blue', alpha=0.1)

ax.set_title("6 terms fixed vs trainable lambda")
ax.set_ylabel("Mean Reward")
ax.legend()
plt.show()


# Create a new figure
fig, ax = plt.subplots()

# Plot all the average mean_rewards and their respective 95% confidence intervals
mean_data_1["mean"].plot(ax=ax, label="1 term", color='blue')
ax.fill_between(mean_data_1.index, mean_data_1["lower_bound"], mean_data_1["upper_bound"], color='blue', alpha=0.1)

mean_data_2["mean"].plot(ax=ax, label="3 terms", color='red')
ax.fill_between(mean_data_2.index, mean_data_2["lower_bound"], mean_data_2["upper_bound"], color='red', alpha=0.1)

mean_data_5["mean"].plot(ax=ax, label="20 terms", color='orange')
ax.fill_between(mean_data_5.index, mean_data_5["lower_bound"], mean_data_5["upper_bound"], color='orange', alpha=0.1)

# Add title, label, and legend to the plot
ax.set_title("Comparison of Test Runs with Different Terms")
ax.set_ylabel("Mean Reward")
ax.set_xlabel("Episode")
ax.legend()
plt.show()


# # Create a new figure that shows 6 terms fixed vs 6 term cloud plots in 1 plot
# fig, ax = plt.subplots()

# mean_data_3["mean"].plot(ax=ax, label="Average 6 terms", color='red')
# ax.fill_between(mean_data_3.index, mean_data_3["lower_bound"], mean_data_3["upper_bound"], color='red', alpha=0.1)

# mean_data_3_2["mean"].plot(ax=ax, label="Average 6 Fixed terms", color='blue')
# ax.fill_between(mean_data_3_2.index, mean_data_3_2["lower_bound"], mean_data_3_2["upper_bound"], color='blue', alpha=0.1)

# ax.set_title("6 terms fixed vs trainable lambda")
# ax.set_ylabel("Mean Reward")
# ax.legend()

# plt.show()


# Create a 1x3 subplot (one row and three columns)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 1 term plot
mean_data_1["mean"].plot(ax=axs[0], label="1 term", color='blue')
axs[0].fill_between(mean_data_1.index, mean_data_1["lower_bound"], mean_data_1["upper_bound"], color='blue', alpha=0.1)
axs[0].set_title("1 term")
axs[0].set_ylabel("Mean Reward")
axs[0].set_xlabel("Episode")

# 3 terms plot
mean_data_2["mean"].plot(ax=axs[1], label="3 terms", color='red')
axs[1].fill_between(mean_data_2.index, mean_data_2["lower_bound"], mean_data_2["upper_bound"], color='red', alpha=0.1)
axs[1].set_title("3 terms")
axs[1].set_xlabel("Episode")

# 20 terms plot
mean_data_5["mean"].plot(ax=axs[2], label="20 terms", color='orange')
axs[2].fill_between(mean_data_5.index, mean_data_5["lower_bound"], mean_data_5["upper_bound"], color='orange', alpha=0.1)
axs[2].set_title("20 terms")
axs[2].set_xlabel("Episode")

# Show the plot
plt.tight_layout()  # This helps to fit the subplots nicely in the figure
plt.show()
fig.savefig("TermSideBySide.pdf", dpi=400, transparent=True)