# Hyperparamter Tuning Imports 
import os

os.system("source ray_env/bin/activate")
import argparse

import numpy as np
import ray
import tensorflow as tf
from ray import air, tune
from ray.air.checkpoint import Checkpoint
from ray.tune import CLIReporter, Stopper
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler

from Agent.agent import PGAgent

#ray.init("auto")
        
class MyTrainableClass(tune.Trainable):
    def setup(self, config):
        self.config = config
        self.agent = PGAgent(config=config)

        self.episode_reward_history = []
        self.batch = 0
        self.max_batch = self.agent.max_episodes // self.agent.batch_size
    
    def step(self):
        
        # Calculate the current episodes
        episode = (self.batch + 1) * self.agent.batch_size
        # Increment the batch counter
        self.batch = self.batch + 1

        # Gather the trajectories
        episodes = self.agent.gather_trajectories()

        # Group states, actions and returns in numpy arrays
        # (make a single array of all states, actions and returns from all trajectories)
        states = np.concatenate([ep['states'] for ep in episodes])
        actions = np.concatenate([ep['actions'] for ep in episodes])
        rewards = [ep['rewards'] for ep in episodes]
        returns = np.concatenate([self.agent.compute_returns(ep_rwds) for ep_rwds in rewards])
        returns = np.array(returns, dtype=np.float32)

        id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

        self.agent.reinforce(states, id_action_pairs, returns, episode)

        # Store collected rewards
        for ep_rwds in rewards:
            self.episode_reward_history.append(np.sum(ep_rwds))

        avg_rewards = np.mean(self.episode_reward_history[-10:])
        
        # with self.agent.writer.as_default():
        #     tf.summary.scalar("model/avg_rew", avg_rewards, step=episode)
        if avg_rewards >= self.agent.max_average:
            self.agent.Actor.save_weights(self.agent.agent_name + self.agent.env_name + ".h5")
            
        return {"mean_reward": avg_rewards, "episode": episode, "batch": self.batch}
    

    def reset_config(self, new_config):
        self.config.update(new_config)
        self.agent = PGAgent(config = new_config)
        return True
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.h5")
        self.agent.Actor.save_weights(checkpoint_path)
        return tmp_checkpoint_dir
    
    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.h5")
        self.agent.Actor.load_weights(checkpoint_path)


if __name__ == "__main__":
    #ray.init()

    # search space
    best_trail = {
        "env_name": "CartPole-v1", # Name of the environment
        "input_dim": 1, # Input dimension of the environment (CartPole-v1 = 1)
        "agent_name": "PartitionedPQC Policy Gradient", # Name of the agent
       
        # Hyperparamters for the training loop
        "max_episodes" : 1200, # Number of episodes to run
        "max_average": 500, # Average score to reach to stop training
        "batch_size": 10, # the number of trajectories to collect before each REINFORCE update

        # Hyperparamets for the dimensions of the PQC
        "n_partitions": 2, # Number of partitions in the PQC.
        "n_terms": 1, # A hyperparamter for number of terms to avaluate in the sum. (run time cost vs accuracy tradeoff)
        "n_layers": 5, #Depth of the PQC.
        # "rescale_parameter": tune.grid_search([0.001, 0,1, 1]), # Lambda Rescale parameter search. WHAT SHOULD THIS BE?

        # Hyperparamters for the Policy Graidnet algorithm
        "gamma": 1, # Discount factor for the rewards
        "beta": 1, # Regularization parameter for the PQC.

        # Hyperparamter for the learning rates of the MODEL
        "theta_lr": 0.002, # learning rate for the rotational gates (Keep the same as orginal paper)
        "zeta_lr":  0.0002, # Learning rate for the zeta paramters
        "lambda_lr": 0.0002, # Learning rate for the Adam optimizer # 0.001 before
        
        "x_l": False, # Do we take the exponential of the lambda paramters? (this tend to make the model very unstable for some reason)
        "trainable_lambdas": True,
        "rescaling_scheme": "factoring",

        # Testing currently.
        "input_lr":  0.1, # Learning rate for the Input encodings (Keep the same as orginal paper)
        "obs_lr":  0.1, # Learning rate for the Adam optimizer (Keep the same as orginal paper)
    }
       
    # Experiment name
    exp_name = "reshuffled_inputs"

    # Define the logger
    reporter = CLIReporter(max_progress_rows=100)
    reporter.add_metric_column("mean_reward")
    reporter.add_metric_column("episode")

    class CustomStopper(Stopper):
        def __init__(self):
            self.should_stop = {}
            self.should_stop_all = False
            self.count_below_threshold = {}
            self.stop_conditions = {
                500: float("inf")
            }

        def __call__(self, trial_id, result):
            if trial_id not in self.should_stop:
                self.should_stop[trial_id] = False
                self.count_below_threshold[trial_id] = {}
                for threshold in self.stop_conditions:
                    self.count_below_threshold[trial_id][threshold] = 0
            
            if not self.should_stop[trial_id]:

                # if result["mean_reward"] >= 500:
                #     self.should_stop[trial_id] = True
                    
                if result["episode"] >= 1200:
                    self.should_stop[trial_id] = True
                
                for threshold, episode in self.stop_conditions.items():
                    if result["episode"] > episode:
                        if result["mean_reward"] < threshold:
                            self.count_below_threshold[trial_id][threshold] += 1
                        else:
                            self.count_below_threshold[trial_id][threshold] = 0
                        
                        if self.count_below_threshold[trial_id][threshold] >= 3:
                            self.should_stop[trial_id] = True

            return self.should_stop[trial_id]
        
        def stop_all(self):
            return False
  
    stopper=CustomStopper()

    # This is the trail that uses 
    resource_group = tune.PlacementGroupFactory([{"CPU": 4, "CustomResource": 1}])
    trainable_with_recources = tune.with_resources(MyTrainableClass, resource_group)
                                                                                                                                              
    #The tuner object
    # tuner = tune.Tuner(
    #     MyTrainableClass,
    #     run_config=air.RunConfig(
    #         name=exp_name,
    #         stop=stopper,
    #         progress_reporter=reporter,
    #         failure_config=air.FailureConfig(max_failures=2),
    #         verbose=1,
    #     ),
    #     tune_config=tune.TuneConfig(
    #         num_samples=10,
    #         mode="max",
    #         metric="mean_reward",
    #         max_concurrent_trials=1,
    #     ),
    #     param_space=best_trail,
    # )
    
    tuner = tune.Tuner.restore(
        "~/ray_results/reshuffled_inputs",
        MyTrainableClass,
        resume_errored=False,
        restart_errored=True,
    )

    results = tuner.fit()