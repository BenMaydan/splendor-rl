import os
import sys
import time
import numpy as np
import torch
import gymnasium as gym
from gymnasium.spaces import utils
from torch.utils.tensorboard import SummaryWriter

# Import AgileRL components
from agilerl.algorithms.ppo import PPO

# Import your custom environment (assuming it's saved in a file named splendor_env.py)
from env.splendor_env import SplendorEnv

class OnPolicyRolloutBuffer:
    """
    Custom buffer to store on-policy trajectories for PPO.
    """
    def __init__(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.terminations, self.truncations, self.values = [], [], [], []

    def add(self, state, action, log_prob, reward, termination, truncation, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.terminations.append(termination)
        self.truncations.append(truncation)
        self.values.append(value)

    def get_experiences(self, next_state):
        experiences = (
            self.states, 
            self.actions, 
            self.log_probs, 
            self.rewards, 
            self.terminations,
            self.truncations,
            self.values, 
            next_state 
        )
        self.clear()
        return experiences

    def clear(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.terminations, self.truncations, self.values = [], [], [], []
        
    def __len__(self):
        return len(self.states)

def train(max_episodes=1000, max_hours=None, checkpoint_minutes=None, checkpoint_episodes=None, checkpoint_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir="runs/splendor_ppo")
    
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

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
    update_steps = 2048  # Number of steps to collect before calling .learn()

    # 4. Main AEC Training Loop
    total_steps = 0
    episode = 0
    
    # Time tracking variables
    start_time = time.time()
    last_checkpoint_time = start_time

    try:
        while True:
            # Check maximum hours limit
            if max_hours is not None:
                elapsed_hours = (time.time() - start_time) / 3600.0
                if elapsed_hours >= max_hours:
                    print(f"Reached maximum training time of {max_hours:.2f} hours. Stopping training.")
                    break
                    
            # Check maximum episodes limit
            if episode >= max_episodes:
                print(f"Reached maximum episodes of {max_episodes}. Stopping training.")
                break

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
                    # Pass termination and truncation separately
                    buffer.add(prev_s, prev_a, prev_lp, reward, termination, truncation, prev_v)
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
                    # Ensure the PPO learn method gets called. Adding a generic loss return for logging if supported.
                    loss = ppo_agent.learn(experiences) 

            # --- Logging and Checkpointing Logic ---
            
            # Logging progress
            if (episode + 1) % 10 == 0:
                avg_reward = sum(episode_rewards.values()) / env.num_players
                print(f"Episode: {episode + 1} | Total Steps: {total_steps} | Avg Reward: {avg_reward:.2f}")
                
                # Write to TensorBoard
                writer.add_scalar("Training/Average_Reward", avg_reward, episode + 1)
                writer.add_scalar("Training/Total_Steps", total_steps, episode + 1)
                # If ppo_agent.learn returns loss metrics, you can log them here too.

            # Checkpointing
            current_time = time.time()
            elapsed_minutes_since_ckpt = (current_time - last_checkpoint_time) / 60.0
            
            time_to_ckpt = checkpoint_minutes is not None and elapsed_minutes_since_ckpt >= checkpoint_minutes
            episode_to_ckpt = checkpoint_episodes is not None and (episode + 1) % checkpoint_episodes == 0

            if time_to_ckpt or episode_to_ckpt:
                ckpt_filename = f"ppo_splendor_ep{episode+1}_steps{total_steps}.pt"
                ckpt_path = os.path.join(checkpoint_dir, ckpt_filename)
                
                # Save using AgileRL's built-in saveCheckpoint (or fallback to PyTorch save if needed)
                ppo_agent.saveCheckpoint(ckpt_path) 
                last_checkpoint_time = current_time
                print(f"--> Checkpoint saved to {ckpt_path}")

            episode += 1
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C). Initiating graceful shutdown...")
    finally:
        # Final save and cleanup (this runs whether the loop finishes naturally or is interrupted)
        final_path = os.path.join(checkpoint_dir, "ppo_splendor_final.pt")
        ppo_agent.save_checkpoint(final_path)
        print(f"Final model saved to {final_path}")
        writer.flush()
        writer.close()
        print("TensorBoard writer closed. Shutdown complete.")
        sys.exit()

if __name__ == "__main__":
    # Example execution with constraints:
    # Train for max 12 hours, save checkpoint every 30 mins OR every 500 episodes
    train(
        max_episodes=5_000_000, 
        max_hours=12.0, 
        checkpoint_minutes=30.0, 
        checkpoint_episodes=500
    )