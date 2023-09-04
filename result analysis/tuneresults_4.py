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

experiment_path_1 = f"~/ray_results/test_run_3_zeta_lambda_permutations"

tuner = tune.Tuner.restore(experiment_path_1)
result_grid = tuner.get_results()
result_dataframe = result_grid.get_dataframe()


error_free = []
for result in result_grid:
    print("result config", result.config)
  
    if result.error:
        print("error", result.error)
        continue
    else:
        error_free.append(result)
print("error_free", error_free)


trail_dict = {}
# create a dict for every permutation of zeta and lambda
for result in error_free:
    zeta_lr = result.config["zeta_lr"]
    lambda_lr = result.config["lambda_lr"]
    if (zeta_lr, lambda_lr) not in trail_dict:
        trail_dict[(zeta_lr, lambda_lr)] = []
    trail_dict[(zeta_lr, lambda_lr)].append(result)

print("trail_dict", trail_dict)
# get the average trajectory of all trails in each permuatataion. First loop over the permutations
for key, result_grid in trail_dict.items():
    print("key", key)
    print('values', result_grid)

    dataframes = [result.metrics_dataframe for result in result_grid]
    all_data_1 = pd.concat([df.assign(trail_id=i) for i, df in enumerate(dataframes)])
    
    mean_data_1 = all_data_1.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
    mean_data_1.columns = ["mean", "lower_bound", "upper_bound"]
    mean_data_1.index = mean_data_1.index * 10

    # Plot the average mean_reward and the 95% confidence interval
    fig, ax = plt.subplots()
    mean_data_1["mean"].plot(ax=ax, label="Average")
    ax.fill_between(mean_data_1.index, mean_data_1["lower_bound"], mean_data_1["upper_bound"], alpha=0.3)
    ax.set_title(f"Mean Reward for zeta_lr={key[0]}, lambda_lr={key[1]}, num trails = {len(result_grid)}")
    ax.set_ylabel("Mean Reward")
    ax.legend()

    # Add experiment information to the info box
    # ax.text(1.1, 0.5, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
    plt.show()