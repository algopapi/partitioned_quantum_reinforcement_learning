import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from ray import tune

n_terms = 0

experiment_path_1 = "~/ray_results/unpartitioned_pqc"
experiment_path_10 = f"~/ray_results/testrun_10_term_correct_fixed"
experiment_path_20 = f"~/ray_results/test_run_20_terms"
tuner = tune.Tuner.restore(experiment_path_1)
tuner_2 = tune.Tuner.restore(experiment_path_20)
tuner_3 = tune.Tuner.restore(experiment_path_10)
result_grid = tuner.get_results()
result_grid_2 = tuner_2.get_results()
result_grid_3 = tuner_3.get_results()

result_grid_short = []
i = 0
for result in result_grid:
    # take every one result so 0, 2, 4, 6, 8
    if i % 2 == 0:
        result_grid_short.append(result)
    i += 1
        

dataframes = [result.metrics_dataframe for result in result_grid_short]
print("result _grid short length", len(result_grid_short))


fig, ax = plt.subplots()
all_data = pd.concat([df.assign(trail_id=i) for i, df in enumerate(dataframes)])
mean_data = all_data.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
mean_data.columns = ["mean", "lower_bound", "upper_bound"]
mean_data.index = mean_data.index * 10

ax.fill_between(mean_data.index, mean_data["lower_bound"], mean_data["upper_bound"], alpha=0.3)
mean_data["mean"].plot(ax=ax, label=f"Unpartitioned")

dataframes = [result.metrics_dataframe for result in result_grid_2]

all_data = pd.concat([df.assign(trail_id=i) for i, df in enumerate(dataframes)])
mean_data = all_data.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
mean_data.columns = ["mean", "lower_bound", "upper_bound"]
mean_data.index = mean_data.index * 10

ax.fill_between(mean_data.index, mean_data["lower_bound"], mean_data["upper_bound"], alpha=0.3)
mean_data["mean"].plot(ax=ax, label=f"20 terms")

dataframes = [result.metrics_dataframe for result in result_grid_3]

# all_data = pd.concat([df.assign(trail_id=i) for i, df in enumerate(dataframes)])
# mean_data = all_data.groupby("training_iteration")["mean_reward"].agg(["mean", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
# mean_data.columns = ["mean", "lower_bound", "upper_bound"]
# mean_data.index = mean_data.index * 10

# ax.fill_between(mean_data.index, mean_data["lower_bound"], mean_data["upper_bound"], alpha=0.3)
# mean_data["mean"].plot(ax=ax, label=f"n_terms={n_terms}")

plt.legend()
plt.show()
fig.savefig(f"unpartitioned_comp.pdf", dpi=300, bbox_inches='tight')