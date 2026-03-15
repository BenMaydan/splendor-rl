import os
import time
import numpy as np
import torch
from gymnasium.spaces import utils
from torch.utils.tensorboard import SummaryWriter

# Import AgileRL components
from agilerl.algorithms.ppo import PPO
from agilerl.components.rollout_buffer import RolloutBuffer

# Import your custom environment
from env.splendor_env import SplendorEnv

def train(max_episodes=1000, max_hours=None, checkpoint_minutes=None, checkpoint_episodes=None, checkpoint_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    writer = SummaryWriter(log_dir="runs/splendor_ppo")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Initialize the Splendor Environment
    env = SplendorEnv(num_players=4)
    env.reset()

    # 2. Extract and Flatten Spaces for AgileRL
    dummy_agent = env.possible_agents[0]
    base_obs_space = env.observation_space(dummy_agent)["observation"]
    flat_obs_space = utils.flatten_space(base_obs_space)
    action_space = env.action_space(dummy_agent)

    # 3. Initialize AgileRL PPO Agent (Acting as a shared brain for all players)
    ppo_agent = PPO(
        observation_space=flat_obs_space,
        action_space=action_space,
        device=device,
        net_config={
            "encoder_config": {"hidden_size": [256, 256, 256]}, 
            "head_config": {"hidden_size": [256]} 
        },
        batch_size=256,
        ent_coef=0.02,
        update_epochs=8
    )

    update_steps = 2048

    # Initialize AgileRL's native RolloutBuffer with Gym spaces
    buffer = RolloutBuffer(
        capacity=update_steps,
        observation_space=flat_obs_space,
        action_space=action_space,
        num_envs=1,  # Since you process AEC steps sequentially
        device=device
    )
    
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

            env.reset()
            last_step_data = {a: None for a in env.possible_agents}
            episode_rewards = {a: 0.0 for a in env.possible_agents}
            
            for agent_id in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                done = int(termination or truncation)
                
                raw_obs = observation["observation"]
                mask = observation["action_mask"]
                flat_state = utils.flatten(base_obs_space, raw_obs)
                
                # Complete transition and add to AgileRL buffer
                if last_step_data[agent_id] is not None:
                    prev_s, prev_a, prev_lp, prev_v = last_step_data[agent_id]
                    
                    # AgileRL's buffer .add() method expects specific named kwargs and batched arrays
                    buffer.add(
                        obs=np.expand_dims(prev_s, axis=0),
                        action=np.array([prev_a]),
                        reward=np.array([reward]),
                        done=np.array([done]),
                        value=np.array([prev_v]),
                        log_prob=np.array([prev_lp])
                    )
                    episode_rewards[agent_id] += reward
                    
                if done:
                    action = None
                else:
                    action, log_prob, _, value = ppo_agent.get_action(flat_state, action_mask=mask)
                    
                    if isinstance(action, np.ndarray):
                        action = int(action[0])
                        
                    last_step_data[agent_id] = (flat_state, action, log_prob, value)
                    total_steps += 1
                    
                env.step(action)
                
                # Train the network when the AgileRL buffer is full
                if buffer.size() >= update_steps:
                    # 1. Get the dictionary from the buffer
                    exp_dict = buffer.get()
                    
                    # 2. Manually construct the 8-item tuple using your exact dictionary keys:
                    # Order: (states, actions, log_probs, rewards, dones, values, next_states, next_dones)
                    experiences_tuple = (
                        exp_dict['observations'],
                        exp_dict['actions'],
                        exp_dict['log_probs'],
                        exp_dict['rewards'],
                        exp_dict['dones'],
                        exp_dict['values'],
                        np.expand_dims(flat_state, axis=0), # Use current state as next_state
                        np.array([done])                    # Use current done as next_done
                    )
                    
                    # 3. Pass the tuple to learn()
                    loss = ppo_agent.learn(experiences_tuple)
                    
                    # 4. Reset the buffer
                    buffer.reset()

            # Logging and Checkpointing
            if (episode + 1) % 10 == 0:
                avg_reward = sum(episode_rewards.values()) / env.num_players
                print(f"Episode: {episode + 1} | Total Steps: {total_steps} | Avg Reward: {avg_reward:.2f}")
                writer.add_scalar("Training/Average_Reward", avg_reward, episode + 1)
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
        max_episodes=1_000_000,
        # max_hours=12.0, 
        checkpoint_minutes=30.0, 
        checkpoint_episodes=2_000
    )
