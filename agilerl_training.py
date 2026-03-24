import os
import time
import copy
import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium.spaces import Dict, utils
from torch.utils.tensorboard import SummaryWriter

from agilerl.algorithms.ppo import PPO

from env.splendor_env import SplendorEnv

class AgileRLSelfPlayWrapper(gym.Env):
    """
    A single-agent wrapper that plays the opponents' turns automatically using
    the current AgileRL PPO agent's policy. The external `.step()` only controls
    the Learner agent.
    """
    def __init__(self, aec_env):
        super().__init__()
        self.env = aec_env
        self.ppo_agent = None  # Must be set after PPO initialization
        
        representative_agent = self.env.possible_agents[0]
        orig_obs_space = self.env.observation_space(representative_agent)
        
        self.base_obs_space = orig_obs_space["observation"]
        flat_inner_obs_space = utils.flatten_space(self.base_obs_space)
        
        self.observation_space = Dict({
            "observation": flat_inner_obs_space,
            "action_mask": orig_obs_space["action_mask"]
        })
        self.action_space = self.env.action_space(representative_agent)
        self.learner_agent = None

    def _get_obs_and_mask(self):
        obs, reward, terminated, truncated, info = self.env.last()
        flat_inner_state = utils.flatten(self.base_obs_space, obs["observation"])
        
        flat_obs = {
            "observation": flat_inner_state,
            "action_mask": np.copy(obs["action_mask"])
        }
        return flat_obs, float(reward), terminated, truncated, info

    def _advance_to_learner(self):
        while self.env.agent_selection != self.learner_agent:
            curr_agent = self.env.agent_selection
            
            if self.env.terminations[curr_agent] or self.env.truncations[curr_agent]:
                self.env.step(None)
                if not self.env.agents:
                    break
                continue
                
            obs, _, _, _, _ = self._get_obs_and_mask()
            
            if self.ppo_agent is not None:
                # Format to batch dimension for AgileRL
                batched_state = {
                    "observation": np.expand_dims(obs["observation"], axis=0),
                    "action_mask": np.expand_dims(obs["action_mask"], axis=0).astype(bool)
                }
                action, _, _, _ = self.ppo_agent.get_action(batched_state, action_mask=batched_state["action_mask"])
                if isinstance(action, np.ndarray):
                    action = int(action[0])
            else:
                valid_actions = np.where(obs["action_mask"])[0]
                action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
                
            self.env.step(action)
            
        if self.env.agent_selection == self.learner_agent:
            obs, reward, term, trunc, info = self._get_obs_and_mask()
        else:
            obs = self.observation_space.sample()
            reward, term, trunc, info = 0.0, True, False, {}
            
        return obs, float(reward), term, trunc, info

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        self.learner_agent = random.choice(self.env.possible_agents)
        obs, _, _, _, info = self._advance_to_learner()
        return obs, info

    def step(self, action):
        self.env.step(action)
        obs, reward, term, trunc, info = self._advance_to_learner()
        return obs, float(reward), term, trunc, info


def train(max_episodes=1000, max_hours=None, checkpoint_minutes=None, checkpoint_episodes=None, checkpoint_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    writer = SummaryWriter(log_dir="runs/splendor_ppo")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Initialize the AEC Splendor Environment & Gym Wrapper
    env = SplendorEnv(num_players=4)
    gym_env = AgileRLSelfPlayWrapper(env)
    
    agilerl_obs_space = gym_env.observation_space
    action_space = gym_env.action_space

    # 2. Initialize AgileRL PPO Agent (WITH rollout buffer)
    update_steps = 2048
    
    ppo_agent = PPO(
        observation_space=agilerl_obs_space,
        action_space=action_space,
        device=device,
        net_config={
            "encoder_config": {
                "latent_dim": 256,
                "max_latent_dim": 512,
                "mlp_config": {"hidden_size": [256, 256]}
            }, 
            "head_config": {
                "hidden_size": [256] 
            } 
        },
        batch_size=256,
        ent_coef=0.02,
        update_epochs=8,
        learn_step=update_steps,
        use_rollout_buffer=True, # USE THE NATIVE INTERNAL BUFFER
    )
    
    # 3. Connect the wrapper's opponent policy to the new PPO agent
    gym_env.ppo_agent = ppo_agent

    total_steps = 0
    episode = 0
    start_time = time.time()
    last_checkpoint_time = start_time

    try:
        while True:
            if max_hours is not None:
                elapsed_hours = (time.time() - start_time) / 3600.0
                if elapsed_hours >= max_hours:
                    print(f"Reached max training time of {max_hours:.2f} hours. Stopping.")
                    break
                    
            if episode >= max_episodes:
                print(f"Reached max episodes of {max_episodes}. Stopping.")
                break

            obs, info = gym_env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                # Format obs for AgileRL
                batched_state = {
                    "observation": np.expand_dims(obs["observation"], axis=0),
                    "action_mask": np.expand_dims(obs["action_mask"], axis=0).astype(bool)
                }

                # Get action from PPO
                action, log_prob, _, value = ppo_agent.get_action(batched_state, action_mask=batched_state["action_mask"])
                scalar_action = int(action[0]) if isinstance(action, np.ndarray) else action
                
                # Step the Gym environment (advances Learner AND Opponents)
                next_obs, reward, term, trunc, info = gym_env.step(scalar_action)
                done = term or trunc

                # Save the ONE contiguous transition directly to the internal buffer
                ppo_agent.rollout_buffer.add(
                    obs=batched_state,
                    action=np.array([scalar_action]),
                    reward=np.array([reward]),
                    done=np.array([done]),
                    value=value,
                    log_prob=log_prob,
                    action_mask=batched_state["action_mask"]
                )

                obs = next_obs
                episode_reward += reward
                total_steps += 1

                # Train the network when the native buffer reaches capacity
                if ppo_agent.rollout_buffer.size() >= update_steps:
                    # 1. We must bootstrap the final value before computing advantages
                    # Grab value of the final state
                    next_batched_state = {
                        "observation": np.expand_dims(obs["observation"], axis=0),
                        "action_mask": np.expand_dims(obs["action_mask"], axis=0).astype(bool)
                    }
                    _, _, _, next_value = ppo_agent.get_action(next_batched_state, action_mask=next_batched_state["action_mask"])
                    
                    # 2. Tell the buffer to calculate advantages
                    ppo_agent.rollout_buffer.compute_returns_and_advantages(
                        last_value=next_value,
                        last_done=np.array([done])
                    )

                    # 3. Call `.learn()` natively
                    loss = ppo_agent.learn()
                    
                    # 4. Empty the buffer once trained
                    ppo_agent.rollout_buffer.reset()

            # Logging and Checkpointing
            if (episode + 1) % 10 == 0:
                print(f"Episode: {episode + 1} | Total Steps: {total_steps} | Episodic Reward: {episode_reward:.2f}")
                writer.add_scalar("Training/Episodic_Reward", episode_reward, episode + 1)
                writer.add_scalar("Training/Total_Steps", total_steps, episode + 1)

            current_time = time.time()
            elapsed_minutes_since_ckpt = (current_time - last_checkpoint_time) / 60.0
            
            if (checkpoint_minutes and elapsed_minutes_since_ckpt >= checkpoint_minutes) or \
               (checkpoint_episodes and (episode + 1) % checkpoint_episodes == 0):
                ckpt_path = os.path.join(checkpoint_dir, f"ppo_splendor_ep{episode+1}_steps{total_steps}.pt")
                ppo_agent.save_checkpoint(ckpt_path) 
                last_checkpoint_time = current_time
                print(f"--> Checkpoint saved to {ckpt_path}")

            episode += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Shutting down...")
    finally:
        final_path = os.path.join(checkpoint_dir, "ppo_splendor_final.pt")
        ppo_agent.save_checkpoint(final_path)
        print(f"Final model saved to {final_path}")
        writer.flush()
        writer.close()

if __name__ == "__main__":
    train(
        max_episodes=20_000_000,
        checkpoint_minutes=30.0, 
        checkpoint_episodes=100_000
    )
