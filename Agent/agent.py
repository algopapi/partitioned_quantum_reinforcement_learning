import datetime
import gc
import os
from collections import defaultdict

import gym
import numpy as np
import pylab
import tensorflow as tf
from ray.air import session

from Agent.model_gen import Part_ReuploadingPQC, ReuploadingPQC

# Import the model generator
# from model_gen import ReuploadingPQC

class PGAgent():
    # Policy Gradient Main Opimization Algorithm
    def __init__(self, config):
        # Environment and PG parameters
        self.agent_name = config["agent_name"]
        self.env_name = config["env_name"]

        # Get environment dimensions
        self.env = gym.make(self.env_name, render_mode="human")
        self.action_space = self.env.action_space.n
        self.state_space = self.env.observation_space.shape[0]
        
        # Select the model we want
        if config["n_partitions"] > 1:
            self.partitioned = True
        else:
            self.partitioned = False
    
        self.trainable_lambdas = config["trainable_lambdas"]
        #self.normalize_lambdas = config["normalize_lambdas"]
        self.rescaling_scheme = config["rescaling_scheme"] # : "exponential", "constant", "factoring"

        # Model Dimension Parameters
        self.input_dim = config["input_dim"]
        self.n_partitions = config["n_partitions"]
        self.n_qubits = int(self.state_space / self.n_partitions)
 
        print("state space :", self.state_space)
        print("action space : ", self.action_space)
        print("Number of qubits :", self.n_qubits)

        # Agent Hyperparameters
        self.max_episodes = config["max_episodes"]
        self.max_average = config["max_average"] # Average Score Target
        self.gamma = config["gamma"]

        # state that we user to normalize the input
        self.state_bounds = np.array([2.4, 2.5, 0.21, 2.5])
        self.state_bounds_s = np.array([0.21, 2.5, 2.4, 2.5])

          # input which we use to rescale the lambdas
        self.state_bounds_res = np.reshape(self.state_bounds, [1, 1, 4])
        self.rescale_input = np.array([2.4, 2.5, 0.21, 2.5])
        self.rescale_input_res = np.reshape(self.rescale_input, [1, 1, 4])

        # Network Hyperparameters
        self.n_terms = config["n_terms"] # A hyperparamter for number of terms to avaluate in the sum. (run time cost vs accuracy tradeoff)
        self.n_layers = config["n_layers"] # Depth of the PQC
        self.beta = config["beta"]
        self.batch_size = config["batch_size"] # Amount of trajectories we collect in a single loop

        # initialize memory (for plots)
        self.states, self.actions, self.rewards = [], [], []
        self.episodes, self.scores, self.average = [], [], []

        self.save_path = 'Models'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.path = '{}_{}_LR'.format(
            self.agent_name, self.env_name)
        self.model_name = os.path.join(self.save_path, self.path)


        # Define the Model
        if self.partitioned:
            self.Actor = Part_ReuploadingPQC(
                self.n_qubits,
                self.n_layers,
                self.action_space,
                self.n_partitions,
                self.n_terms,
                self.input_dim,
                self.beta,
                self.trainable_lambdas,
                self.rescaling_scheme,
            )
            # Learning Rates
            self.theta_lr = config["theta_lr"]
            self.input_lr =  config["input_lr"]
            self.zeta_lr =  config["zeta_lr"]
            self.lambda_lr = config["lambda_lr"]
            self.obs_lr = config["obs_lr"]

            #PQC Optimizers
            self.optimizer_theta = tf.keras.optimizers.Adam(
                learning_rate=self.theta_lr, amsgrad=True)
            self.optimizer_in = tf.keras.optimizers.Adam(
                learning_rate=self.input_lr, amsgrad=True)
            self.optimizer_zeta = tf.keras.optimizers.Adam(
                learning_rate= self.zeta_lr, amsgrad=True)
            self.optimizer_lambda = tf.keras.optimizers.Adam(
                learning_rate=self.lambda_lr, amsgrad=True)
        
            # Alternating layer Optimizer
            self.optimizer_out = tf.keras.optimizers.Adam(
                learning_rate=self.obs_lr, amsgrad=True)
            
            if self.trainable_lambdas:
                self.w_theta, self.w_in, self.w_zeta, self.w_lambda, self.w_out = 0, 1, 2, 3, 4
            else: 
                self.w_theta, self.w_in, self.w_zeta, self.w_out = 0, 1, 2, 3
                
        else:
            self.n_layers = 5
            self.n_qubits = 4
            self.Actor = ReuploadingPQC(
                self.n_qubits, self.n_layers, self.action_space, self.beta
            )
           
            self.input_lr = 0.1
            self.theta_lr = 0.01
            self.obs_lr = 0.1

            self.optimizer_in = tf.keras.optimizers.Adam(
                learning_rate=self.input_lr, amsgrad=True)
            self.optimizer_theta = tf.keras.optimizers.Adam(
                learning_rate= self.theta_lr, amsgrad=True)
            self.optimizer_out = tf.keras.optimizers.Adam(
                learning_rate=self.obs_lr, amsgrad=True)
            self.w_in, self.w_theta, self.w_out = 1, 0, 2

        self.UnpartitionedModel = ReuploadingPQC(
            self.n_qubits*2, self.n_layers, self.action_space, self.beta
        )
        
        # Rescale lambdas of our model
        if self.partitioned:
            if self.rescaling_scheme == 'factoring':
                print("Rescaling lambdas for factoring method ...")
                self.Actor.rescale_lambdas(self.rescale_input_res)

        # Tensorboard Setup
        self.current_time = datetime.datetime.now().strftime("[%m:%d-%H:%M]")
        self.file = 'p={}_t={}_l={}_{}'.format(
            self.n_partitions, self.n_terms, self.n_layers, self.current_time
        )
        self.train_log_dir = 'logs/' + self.current_time
        self.writer = tf.summary.create_file_writer(self.train_log_dir)

     
        with self.writer.as_default():
            tf.summary.scalar("test", 1, step=0)


    def load(self):
        self.Actor.load_weights(f"Models/{self.agent_name}-{self.n_terms}.h5")

    def save(self):
        self.Actor.save_weights(f"Models/{self.agent_name}-{self.n_terms}.h5")
       

    def compute_returns(self, rewards_history):
        """Compute discounted returns with discount factor `gamma`."""
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        returns = returns.tolist()

        return returns


    def reinforce(self, states, actions, returns, episode):
        with tf.profiler.experimental.Trace('train', step_num=episode, _r=1):
            states = tf.convert_to_tensor(states)
            actions = tf.convert_to_tensor(actions)
            returns = tf.convert_to_tensor(returns)
            
            with tf.GradientTape() as tape:
                tape.watch(self.Actor.trainable_variables)
                logits = self.Actor(states) # logits are tn
                p_actions = tf.gather_nd(logits, actions)
                log_probs = tf.math.log(p_actions)
                loss = tf.math.reduce_sum(-log_probs * returns) / self.batch_size
            grads = tape.gradient(loss, self.Actor.trainable_variables)

            with self.writer.as_default():
                for idx, var in enumerate(self.Actor.trainable_variables):
                    tf.summary.histogram(var.name, var, step=episode)
                    if grads is not None:
                        gradient_name = var.name.replace(':', '_') + '_gradient'
                        tf.summary.histogram(gradient_name, grads[idx], step=episode)

            del tape
            if self.partitioned:
                if self.trainable_lambdas:  # check if lambdas are trainable
                    for optimizer, w in zip([self.optimizer_theta, self.optimizer_in, self.optimizer_zeta, self.optimizer_lambda, self.optimizer_out], [self.w_theta, self.w_in, self.w_zeta, self.w_lambda, self.w_out]):
                        optimizer.apply_gradients([(grads[w], self.Actor.trainable_variables[w])])
                else:  # if lambdas are not trainable, skip applying gradients to them
                    for optimizer, w in zip([self.optimizer_theta, self.optimizer_in, self.optimizer_zeta, self.optimizer_out], [self.w_theta, self.w_in, self.w_zeta, self.w_out]):
                        optimizer.apply_gradients([(grads[w], self.Actor.trainable_variables[w])])
            else:
                for optimizer, w in zip([self.optimizer_in, self.optimizer_theta, self.optimizer_out], [self.w_in, self.w_theta, self.w_out]):
                    optimizer.apply_gradients([(grads[w], self.Actor.trainable_variables[w])])

            
            del grads # I dont know why but if you dont do this it saves the graphs and leak memory
            gc.collect()


  
    def gather_trajectories(self):
        """ Gather trajectories from environment in a batched fashion"""
        trajectories = [defaultdict(list) for _ in range(self.batch_size)] # Dicts for trajectories
        envs = [gym.make(self.env_name) for _ in range(self.batch_size)] # Create environments for each trajectory
        # envs[0] = gym.make(self.env_name, render_mode="human") # make 1 env that we can actually observe
        
        done = [False for _ in range(self.batch_size)] # Done booleans
        states = [e.reset() for e in envs] # Initial states
        states = [e[0] for e in states] # Deal with the rather annoying assymetry between env.reset() aand env.step()

        while not all (done):
            unfinished_ids = [i for i in range(self.batch_size) if not done[i]]
            
            normalized_states = [s / self.state_bounds for i, s in enumerate(states) if not done[i]]

            for i, state in zip(unfinished_ids, normalized_states):
                trajectories[i]['states'].append(state)
            
            states = tf.convert_to_tensor(normalized_states)
            action_probs = self.Actor(states) # Get action probabilities for all (unfinished trajectories)
            # Retrieve next states and store
            states = [None for i in range(self.batch_size)]
            for i, policy in zip(unfinished_ids, action_probs):
                action = np.random.choice(self.action_space, p=policy.numpy()) # Sample action from policy
                states[i], reward, d, t, _ = envs[i].step(action)
                done[i] = d or t
                trajectories[i]['actions'].append(action)
                trajectories[i]['rewards'].append(reward)      
        
        return trajectories
    
    def gather_trajectories_swapped(self):
        """ Gather trajectories from environment in a batched fashion but then swapped"""
        trajectories = [defaultdict(list) for _ in range(self.batch_size)] # Dicts for trajectories
        envs = [gym.make(self.env_name) for _ in range(self.batch_size)] # Create environments for each trajectory
    
        done = [False for _ in range(self.batch_size)] # Done booleans
        states = [e.reset() for e in envs] # Initial states
        states = [e[0] for e in states] # Deal with the rather annoying assymetry between env.reset() and env.step()
        # Reshuffle the states
        states = [np.array([s[0], s[3], s[2], s[1]]) for s in states]
        #print(" initial states after swap", states)

        while not all(done):
            unfinished_ids = [i for i in range(self.batch_size) if not done[i]]

            normalized_states = [s / self.state_bounds for i, s in enumerate(states) if not done[i]]

            for i, state in zip(unfinished_ids, normalized_states):
                # Swap the 2nd and 4th elements before appending to the trajectory
                # state = np.array([state[2], state[3], state[0], state[1]])
                trajectories[i]['states'].append(state)

            states = tf.convert_to_tensor(normalized_states)
            action_probs = self.Actor(states) # Get action probabilities for all (unfinished trjectories)
            # Retrieve next states and store
            states = [None for i in range(self.batch_size)]
            for i, policy in zip(unfinished_ids, action_probs):
                action = np.random.choice(self.action_space, p=policy.numpy()) # Sample action from policy
                next_state, reward, d, t, _ = envs[i].step(action)
                done[i] = d or t
                #print("next_state preswap" , next_state)
                # Swap the 2nd and 4th elements before appending to the states
                next_state = np.array([next_state[0], next_state[3], next_state[2], next_state[1]])
                #print("next state", next_state)
                states[i] = next_state

                trajectories[i]['actions'].append(action)
                trajectories[i]['rewards'].append(reward)
                
        return trajectories


    def learn(self):
        """ Learn from the gathered trajectories """
        print("Learning...")
        episode_reward_history = []
        max_batch = self.max_episodes // self.batch_size
        loaded_checkpoint = session.get_checkpoint() # RayTune related stuff
        c_batch = 0

        if loaded_checkpoint:
            print("Loaded checkpoint from {}".format(loaded_checkpoint))
            last_batch = loaded_checkpoint.to_dict()["batch"]
            c_batch = last_batch + 1
        
        for batch in range(c_batch, max_batch):
            episode = (batch + 1) * self.batch_size
        
            # Gather episodes
            episodes = self.gather_trajectories()

            # Group states, actions and returns in numpy arrays
            states = np.concatenate([ep['states'] for ep in episodes])
            actions = np.concatenate([ep['actions'] for ep in episodes])
            rewards = [ep['rewards'] for ep in episodes]
            returns = np.concatenate([self.compute_returns(ep_rwds) for ep_rwds in rewards])
            returns = np.array(returns, dtype=np.float32)
            id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

            # Update model parameters.
            self.reinforce(states, id_action_pairs, returns, episode)
            
            # Store collected rewards
            for ep_rwds in rewards:
                episode_reward_history.append(np.sum(ep_rwds))

            avg_rewards = np.mean(episode_reward_history[-10:])

            with self.writer.as_default():
                tf.summary.scalar("model/avg_rew", avg_rewards, step=episode)
            
            self.save_plot(avg_rewards, episode)
            print('\033[32m' + f'Finished episode {(batch + 1) * self.batch_size}, Average rewards: {avg_rewards}' + '\033[0m')

            if avg_rewards >= self.max_average or episode >= self.max_episodes:   
                self.save()
                return {"mean_reward": avg_rewards, "episode": episode, "batch": batch}
        

        return {"mean_reward": avg_rewards, "episode": episode, "batch": batch} # This is for raytune
        

    def save_plot(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        
        #Log variables to tensorboard
        with self.writer.as_default():
            tf.summary.scalar("model/score", score, step=episode)
            tf.summary.scalar("model/avg_rew", self.average[-1], step=episode)

    
    def plot(self):
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.plot(self.episodes, self.average, 'r')
        pylab.title(self.agent_name, fontsize=18)
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Episodes', fontsize=18)


    def call_model(self):
        self.Actor.predict(self.state_bounds_res)


    def test(self):
      """ Test previously learned models"""

      self.load()
    
      for e in range(self.max_episodes):
          state = self.env.reset()
          state = state[0]
          state = np.reshape(state, [1, self.state_space])
          done, truncated = False, False
          i = 0
          while not (done or truncated):
            self.env.render()
            action = np.argmax(self.Actor.predict(state))
            next_state, _ , done, truncated, _ = self.env.step(action)
            state = np.reshape(next_state, [1, self.state_space])
            i += 1
            if (done or truncated):
                print("episode: {}/{}, score: {}".format(e, self.max_episodes, i))
                break

