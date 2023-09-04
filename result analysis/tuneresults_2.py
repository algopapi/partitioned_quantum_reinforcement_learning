# Hyperparamter Tuning Imports 
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

experiment_path = f"~/ray_results/rebalanced/testrun_1"
experiment_path_alt = f"~/ray_results/reshuffled_inputs"
experiment_path_2 = f"~/ray_results/rebalanced/testrun_2"
experiment_path_3 = f"~/ray_results/rebalanced/testrun_3"
experiment_path_4 = f"~/ray_results/testrun_3_term_fixed_lambdas"
experiment_path_5 = f"~/ray_results/testrun_6_tl_ntl"
experiment_path_6 = f"~/ray_results/testrun_10_tl"


restored_tuner = tune.Tuner.restore(experiment_path)
restored_tuner_alt = tune.Tuner.restore(experiment_path_alt)
restored_tuner_2 = tune.Tuner.restore(experiment_path_2)
restored_tuner_3 = tune.Tuner.restore(experiment_path_3)
restored_tuner_4 = tune.Tuner.restore(experiment_path_4)
restored_tuner_5 = tune.Tuner.restore(experiment_path_5)
restored_tuner_6 = tune.Tuner.restore(experiment_path_6)

print("Restored tuner", restored_tuner)


result_grid = restored_tuner.get_results()
result_grid_alt = restored_tuner_alt.get_result()
result_grid_2 = restored_tuner_2.get_results()
result_grid_3 = restored_tuner_3.get_results()
result_grid_4 = restored_tuner_4.get_results()
result_grid_5 = restored_tuner_5.get_results()
result_grid_6 = restored_tuner_6.get_results()



best_result_1 = result_grid.get_best_result("mean_reward", mode="max")
best_result_2 = result_grid_2.get_best_result("mean_reward", mode="max")
best_result_3 = result_grid_3.get_best_result("mean_reward", mode="max")
best_result_4 = result_grid_4.get_best_result("mean_reward", mode="max")
best_result_5 = result_grid_5.get_best_result("mean_reward", mode="max")
best_result_6 = result_grid_6.get_best_result("mean_reward", mode="max")



print("best_result_1 config" , best_result_1.config)
print("best_result_2 config" , best_result_2.config)
print("best_result_3 config" , best_result_3.config)
print("best_result_4 config" , best_result_4.config)
print("best_result_5 config" , best_result_5.config)
print("best_result_6 config" , best_result_6.config)



#Create a dictionary to store the axes for each group of trials with the same number of terms
axes_dict = {}

#Iterate over the results and plot them
for result in result_grid:
    #print("result", result)
    
    # Extract the number of terms from the experiment_tag
    n_terms = int(result.metrics["experiment_tag"].split('_')[2].split('=')[-1])
    key = f'n_terms={n_terms}'

    if result.metrics_dataframe is not None:
        if key not in axes_dict:
            fig, ax = plt.subplots()
            ax.set_title(f"{key}")
            ax.set_ylabel("Mean Test Accuracy")
            ax.set_xlabel("Training Iteration")
            axes_dict[key] = ax

        # Plot the trials on the corresponding axes based on the number of terms
        result.metrics_dataframe.plot("training_iteration", "mean_reward", ax=axes_dict[key])

plt.show()

# cloud plot 
ax = None
for result in result_grid_2:
   #print("result", result)
   #label = f"Trial {result.metrics.trial_id}"
   if ax is None:
       ax = result.metrics_dataframe.plot("training_iteration", "mean_reward")
   else:
       result.metrics_dataframe.plot("training_iteration", "mean_reward", ax=ax)
ax.set_title("Mean Accuracy vs. Training Iteration for All Trials")
ax.set_ylabel("Mean Test Accuracy")
plt.show()


# Collect all the dataframes in a list
dataframes = [result.metrics_dataframe for result in result_grid_2]

# Concatenate all the dataframes, adding a column for trial_id
all_data = pd.concat([df.assign(trial_id=i) for i, df in enumerate(dataframes)])

# Calculate the mean and 2.5% and 97.5% quantiles for each training_iteration
mean_data = all_data.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
mean_data.columns = ["mean", "lower_bound", "upper_bound"]

mean_data.index = mean_data.index * 5

# Plot the average mean_reward and the 95% confidence interval
fig, ax = plt.subplots()
mean_data["mean"].plot(ax=ax, label="Average")
ax.fill_between(mean_data.index, mean_data["lower_bound"], mean_data["upper_bound"], alpha=0.3)
ax.set_title("Mean Reward vs. Episode (Average and 95% Confidence Interval of All Trials)")
ax.set_ylabel("Mean Reward")
ax.legend()

# Add experiment information to the info box
experiment_info = f"""Experiment Information:
- Number of trials: {len(dataframes)}
- Batch dim: 5
- layers: 5
- terms: 6
- theta lr: 0.01
- zeta lr: 0.001
- lambda lr: 3e-4
- input lr: 0.1
- obs lr: 0.1
"""

ax.text(1.1, 0.5, experiment_info, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
plt.show()

# cloud plot end

ax = None
for result in result_grid_3:

   if ax is None:
       ax = result.metrics_dataframe.plot("training_iteration", "mean_reward")
   else:
       result.metrics_dataframe.plot("training_iteration", "mean_reward", ax=ax)
ax.set_title("Mean Accuracy vs. Training Iteration for All Trials")
ax.set_ylabel("Mean Test Accuracy")
plt.show()


ax = None
for result in result_grid_4:
   if ax is None:
       ax = result.metrics_dataframe.plot("training_iteration", "mean_reward")
   else:
       result.metrics_dataframe.plot("training_iteration", "mean_reward", ax=ax)
ax.set_title("Mean Accuracy vs. Training Iteration for All Trials")
ax.set_ylabel("Mean Test Accuracy")
plt.show()   

# Collect all the dataframes in a list
dataframes_2 = [result.metrics_dataframe for result in result_grid_4]

# Concatenate all the dataframes, adding a column for trial_id
all_data_2 = pd.concat([df.assign(trial_id=i) for i, df in enumerate(dataframes_2)])

# Calculate the mean and 2.5% and 97.5% quantiles for each training_iteration
mean_data_2 = all_data_2.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
mean_data_2.columns = ["mean", "lower_bound", "upper_bound"]

mean_data_2.index = mean_data_2.index * 5

# Plot the average mean_reward and the 95% confidence interval
fig, ax = plt.subplots()
mean_data_2["mean"].plot(ax=ax, label="Average")
ax.fill_between(mean_data.index, mean_data["lower_bound"], mean_data["upper_bound"], alpha=0.3)
ax.set_title("Mean Reward vs. Episode (Average and 95% Confidence Interval of All Trials)")
ax.set_ylabel("Mean Reward")
ax.legend()

# Add experiment information to the info box
experiment_info = f"""Experiment Information:
- Number of trials: {len(dataframes)}
- Batch dim: 10
- layers: 5
- terms: 6
- theta lr: 0.001
- zeta lr: 0.0001
- lambda lr: 3e-4
- input lr: 0.1 
- obs lr: 0.1
"""

ax.text(1.1, 0.5, experiment_info, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
plt.show()


ax = None
i = 0
for result in result_grid_5:
   if i == 2:
       continue
   if ax is None:
       ax = result.metrics_dataframe.plot("training_iteration", "mean_reward")
   else:
       result.metrics_dataframe.plot("training_iteration", "mean_reward", ax=ax)
   i = i + 1
ax.set_title("Fixed Lambdas")
ax.set_ylabel("Mean Test Accuracy")
plt.show()



# cloud plot 

ax = None
print("result_grid_6", result_grid_6)
for result in result_grid_6:
   print("result", result)
#    if result.error:
#          print("error", result.error)
#          continue
   #label = f"Trial {result.metrics.trial_id}"
   if ax is None:
       ax = result.metrics_dataframe.plot("training_iteration", "mean_reward")
   else:
       result.metrics_dataframe.plot("training_iteration", "mean_reward", ax=ax)
ax.set_title("Trainable Lambdas")
ax.set_ylabel("Mean Test Accuracy")
plt.show()


# Collect all the dataframes in a list
dataframes = [result.metrics_dataframe for result in result_grid_6]

# Concatenate all the dataframes, adding a column for trial_id
all_data = pd.concat([df.assign(trial_id=i) for i, df in enumerate(dataframes)])

# Calculate the mean and 2.5% and 97.5% quantiles for each training_iteration
mean_data = all_data.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
mean_data.columns = ["mean", "lower_bound", "upper_bound"]

mean_data.index = mean_data.index * 10

# Plot the average mean_reward and the 95% confidence interval
fig, ax = plt.subplots()
mean_data["mean"].plot(ax=ax, label="Average")
ax.fill_between(mean_data.index, mean_data["lower_bound"], mean_data["upper_bound"], alpha=0.3)
ax.set_title("Single Term")
ax.set_ylabel("Mean Reward")
ax.legend()

# Add experiment information to the info box
experiment_info = f"""Experiment Information:
- Number of trials: {len(dataframes)}
- Batch dim: 5
- layers: 5
- terms: 1
- theta lr: 0.001
- zeta lr: 0.0001
- lambda lr: 3e-4
- input lr: 0.1
- obs lr: 0.1
"""

ax.text(1.1, 0.5, experiment_info, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
plt.show()

# cloud plot end