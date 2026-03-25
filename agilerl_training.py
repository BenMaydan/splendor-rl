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
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations

from env.splendor_env import SplendorEnv


def train(max_episodes=1000, max_hours=None, checkpoint_minutes=None, checkpoint_episodes=None, checkpoint_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    writer = SummaryWriter(log_dir="runs/splendor_ppo")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Initialize the AEC Splendor Environment
    env = SplendorEnv(num_players=4)
    
    # Extract spaces for AgileRL
    representative_agent = env.possible_agents[0]
    orig_obs_space = env.observation_space(representative_agent)
    
    base_obs_space = orig_obs_space["observation"]
    flat_inner_obs_space = utils.flatten_space(base_obs_space)
    
    agilerl_obs_space = Dict({
        "observation": flat_inner_obs_space,
        "action_mask": orig_obs_space["action_mask"]
    })
    action_space = env.action_space(representative_agent)

    # 2. Initialize AgileRL PPO Population & HPO Setup
    update_steps = 2048
    POP_SIZE = 6
    
    pop = []
    for i in range(POP_SIZE):
        ppo_agent = PPO(
            observation_space=agilerl_obs_space,
            action_space=action_space,
            device=device,
            index=i,
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
            use_rollout_buffer=True,
        )
        # Track episodic rewards for this specific agent manually
        ppo_agent.fitness = [] # AgileRL expects a list
        pop.append(ppo_agent)

    tournament = TournamentSelection(
        tournament_size=2,
        elitism=True,
        population_size=POP_SIZE,
        eval_loop=1
    )

    mutations = Mutations(
        no_mutation=0.1,
        architecture=0.1,
        new_layer_prob=0.1,
        parameters=0.1,
        activation=0.0, # disabled because it is unsupported for PPO
        rl_hp=0.1,
        mutation_sd=0.1,
        rand_seed=42,
        device=device
    )
    
    evo_steps = 10000
    next_evo_step = evo_steps

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
            
            # Select 4 unique agents from population to play in this episode
            playing_agents = random.sample(pop, 4)
            agent_mapping = {
                env.possible_agents[i]: playing_agents[i] 
                for i in range(4)
            }
            
            # Agent-specific trajectories for the episode
            trajectories = {
                agent: {
                    "obs": [], "action": [], "reward": [], "done": [], "value": [], "log_prob": [], "action_mask": []
                } for agent in env.possible_agents
            }
            
            # Caches for continuous transition collection from all agents
            last_state = {agent: None for agent in env.possible_agents}
            last_action_mask = {agent: None for agent in env.possible_agents}
            last_action = {agent: None for agent in env.possible_agents}
            last_value = {agent: None for agent in env.possible_agents}
            last_log_prob = {agent: None for agent in env.possible_agents}
            
            episode_reward = 0.0

            for agent in env.agent_iter():
                obs, reward, term, trunc, info = env.last()
                done = term or trunc
                
                # Flatten observation to match AgileRL expectations
                flat_inner_state = utils.flatten(base_obs_space, obs["observation"])
                batched_state = {
                    "observation": np.expand_dims(flat_inner_state, axis=0),
                    "action_mask": np.expand_dims(obs["action_mask"], axis=0).astype(bool)
                }

                if last_state[agent] is not None:
                    # Append the transition to this agent's temporary local list
                    trajectories[agent]["obs"].append(last_state[agent])
                    trajectories[agent]["action"].append(last_action[agent])
                    trajectories[agent]["reward"].append(reward)
                    trajectories[agent]["done"].append(done)
                    trajectories[agent]["value"].append(last_value[agent])
                    trajectories[agent]["log_prob"].append(last_log_prob[agent])
                    trajectories[agent]["action_mask"].append(last_action_mask[agent])

                    episode_reward += reward

                if done:
                    env.step(None)
                    continue
                    
                # Action Selection
                current_ppo_agent = agent_mapping[agent]
                action, log_prob, _, value = current_ppo_agent.get_action(batched_state, action_mask=batched_state["action_mask"])
                scalar_action = int(action[0]) if isinstance(action, np.ndarray) else action
                
                # Cache parameters for the NEXT loop for this agent
                last_state[agent] = batched_state
                last_action_mask[agent] = batched_state["action_mask"]
                last_action[agent] = scalar_action
                last_value[agent] = value
                last_log_prob[agent] = log_prob
                
                env.step(scalar_action)
                total_steps += 1

            # Episode ends. Now sequence all completed trajectories contiguously into the PPO buffer
            for env_agent in env.possible_agents:
                traj_len = len(trajectories[env_agent]["obs"])
                current_ppo_agent = agent_mapping[env_agent]
                
                # Accumulate the total reward for this individual's fitness score
                ep_reward = sum(trajectories[env_agent]["reward"])
                current_ppo_agent.fitness.append(ep_reward)
                
                # Check if adding this complete trajectory would overflow the buffer capacity
                if current_ppo_agent.rollout_buffer.size() + traj_len > update_steps:
                    if current_ppo_agent.rollout_buffer.size() > 0:
                        # Since we just finished an episode, we know it ended in a terminal state
                        current_ppo_agent.rollout_buffer.compute_returns_and_advantages(
                            last_value=np.array([0.0]),
                            last_done=np.array([True])
                        )
                        loss = current_ppo_agent.learn()
                        current_ppo_agent.rollout_buffer.reset()

                for t in range(traj_len):
                    current_ppo_agent.rollout_buffer.add(
                        obs=trajectories[env_agent]["obs"][t],
                        action=np.array([trajectories[env_agent]["action"][t]]),
                        reward=np.array([trajectories[env_agent]["reward"][t]]),
                        done=np.array([trajectories[env_agent]["done"][t]]),
                        value=trajectories[env_agent]["value"][t],
                        log_prob=trajectories[env_agent]["log_prob"][t],
                        action_mask=trajectories[env_agent]["action_mask"][t]
                    )

            # Episode ends
            if (episode + 1) % 10 == 0:
                # episode_reward here represents the total combined rewards of all 4 agents
                # meaning it's 4x larger nominally than a single agent view, but since zero-sum, it might be 0.
                print(f"Episode: {episode + 1} | Total Steps (across all agents): {total_steps} | Combined Env Reward: {episode_reward:.2f}")
                writer.add_scalar("Training/Combined_Episodic_Reward", episode_reward, episode + 1)
                writer.add_scalar("Training/Total_Steps", total_steps, episode + 1)
            
            # Checkpoint the elite dynamically before potentially doing a reduction on the fitness buffer
            current_time = time.time()
            elapsed_minutes_since_ckpt = (current_time - last_checkpoint_time) / 60.0
            if (checkpoint_minutes and elapsed_minutes_since_ckpt >= checkpoint_minutes) or (checkpoint_episodes and (episode + 1) % checkpoint_episodes == 0):
                # Calculate mean on the fly to find the best agent without altering the actual fitness tracking
                best_agent = max(pop, key=lambda x: np.mean(x.fitness) if len(x.fitness) > 0 else -1000)
                ckpt_path = os.path.join(checkpoint_dir, f"ppo_elite_ep{episode+1}.pt")
                os.makedirs(checkpoint_dir, exist_ok=True)
                best_agent.save_checkpoint(ckpt_path) 
                last_checkpoint_time = current_time
                print(f"--> Checkpoint saved to {ckpt_path}")

            # Evolution Step
            if total_steps >= next_evo_step:
                print(f"--- Evolution Step triggered at {total_steps} steps ---")
                next_evo_step += evo_steps
                
                # Condense the generation's noisy array of episodic rewards into a single stable mean
                # so that TournamentSelection (with eval_loop=1) evaluates true generational average performance!
                for p_agent in pop:
                    p_agent.fitness = [sum(p_agent.fitness) / len(p_agent.fitness)] if len(p_agent.fitness) > 0 else [-1000.0]

                # Tournament Selection
                elite, survivors = tournament.select(pop)
                
                # Mutation
                pop = mutations.mutation(survivors)
                
                # State Clearing: Prevent off-policy buffer poisoning and reset generational fitness
                for p_agent in pop:
                    p_agent.rollout_buffer.reset()
                    p_agent.fitness = []

            episode += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Shutting down...")
    finally:
        elite = max(pop, key=lambda x: np.mean(x.fitness) if len(x.fitness) > 0 else -1000)
        final_path = os.path.join(checkpoint_dir, "ppo_elite_final.pt")
        os.makedirs(checkpoint_dir, exist_ok=True)
        elite.save_checkpoint(final_path)
        print(f"Final elite model saved to {final_path}")
        writer.flush()
        writer.close()

if __name__ == "__main__":
    train(
        max_episodes=20_000_000,
        checkpoint_minutes=10.0, 
        checkpoint_episodes=1_000
    )
