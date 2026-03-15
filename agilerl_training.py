import numpy as np
import torch
import gymnasium as gym
from gymnasium.spaces import utils

# Import AgileRL components
from agilerl.algorithms.ppo import PPO

# Import your custom environment (assuming it's saved in a file named splendor_env.py)
# If this script is in the same file as the class, you can omit this import.
from env.splendor_env import SplendorEnv

class OnPolicyRolloutBuffer:
    """
    Custom buffer to store on-policy trajectories for PPO.
    """
    def __init__(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def get_experiences(self, next_state):
        experiences = (
            self.states, 
            self.actions, 
            self.log_probs, 
            self.rewards, 
            self.dones, 
            self.values, 
            next_state
        )
        self.clear()
        return experiences

    def clear(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []
        
    def __len__(self):
        return len(self.states)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Initialize the Splendor Environment
    env = SplendorEnv(num_players=4)
    env.reset()

    # 2. Extract and Flatten Spaces for AgileRL
    dummy_agent = env.possible_agents[0]
    
    # We only want to flatten the "observation" dict, not the "action_mask"
    base_obs_space = env.observation_space(dummy_agent)["observation"]
    flat_obs_space = utils.flatten_space(base_obs_space)
    action_space = env.action_space(dummy_agent)

    # 3. Initialize AgileRL PPO Agent
    ppo_agent = PPO(
        observation_space=flat_obs_space,
        action_space=action_space,
        device=device,
        net_config={
            "encoder_config": {"hidden_size": [256, 256, 256]}, 
            "head_config": {"hidden_size": [256]} 
        },                                                     # Wider, deeper network
        batch_size=256,                                        # Larger batch size
        ent_coef=0.02,                                         # Higher exploration
        update_epochs=8                                        # More epochs per rollout
    )

    buffer = OnPolicyRolloutBuffer()
    
    # PPO hyperparameters
    max_episodes = 1000
    update_steps = 2048  # Number of steps to collect before calling .learn()

    # 4. Main AEC Training Loop
    total_steps = 0

    for episode in range(max_episodes):
        env.reset()
        
        # Cache to align delayed AEC rewards with the correct state/action
        last_step_data = {a: None for a in env.possible_agents}
        episode_rewards = {a: 0.0 for a in env.possible_agents}
        
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation
            
            # Extract raw nested dict and the mask
            raw_obs = observation["observation"]
            mask = observation["action_mask"]
            
            # Flatten the nested dict into a 1D numpy array for the neural network
            flat_state = utils.flatten(base_obs_space, raw_obs)
            
            # If this agent acted previously, complete the transition and add to buffer
            if last_step_data[agent_id] is not None:
                prev_s, prev_a, prev_lp, prev_v = last_step_data[agent_id]
                buffer.add(prev_s, prev_a, prev_lp, reward, done, prev_v)
                episode_rewards[agent_id] += reward
                
            if done:
                action = None
            else:
                # Unpack the action, log prob, and value from the PPO agent
                # Note: getAction expects a batched dimension, so we wrap/unwrap if necessary,
                # but AgileRL typically handles single unbatched numpy arrays gracefully.
                action, log_prob, _, value = ppo_agent.get_action(flat_state, action_mask=mask)
                
                # Action comes back as an array, extract the integer for the discrete environment
                if isinstance(action, np.ndarray):
                    action = int(action[0])
                    
                last_step_data[agent_id] = (flat_state, action, log_prob, value)
                total_steps += 1
                
            env.step(action)
            
            # Periodically train the network when the buffer is full
            if len(buffer) >= update_steps:
                experiences = buffer.get_experiences(next_state=flat_state)
                ppo_agent.learn(experiences)
                
        # Logging progress
        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards.values()) / env.num_players
            print(f"Episode: {episode + 1} | Total Steps: {total_steps} | Avg Reward: {avg_reward:.2f}")

if __name__ == "__main__":
    train()