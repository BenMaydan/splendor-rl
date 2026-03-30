import argparse
import os
import time
import random
import numpy as np
import torch
from gymnasium.spaces import utils
from torch.utils.tensorboard import SummaryWriter
from tensordict import TensorDict

from agilerl.algorithms.dqn_rainbow import RainbowDQN
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
from agilerl.components.replay_buffer import MultiStepReplayBuffer, PrioritizedReplayBuffer, ReplayBuffer

from env.splendor_env import SplendorEnv
from utils import TRAINING_CURRICULUM, apply_curriculum_mask

def train(
        max_episodes=500_000,
        max_hours=None,
        checkpoint_minutes=None,
        checkpoint_episodes=None,
        checkpoint_dir="checkpoints_rdqn",
        load_path=None,
        use_training_curriculum=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    writer = SummaryWriter(log_dir="runs/splendor_rdqn")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize the AEC Splendor Environment
    env = SplendorEnv(num_players=4)

    # Validate all action types in curriculum map are valid
    all_action_types = env.get_action_types()
    for action_types in TRAINING_CURRICULUM.values():
        if action_types is None:
            continue
        for action_type in action_types:
            assert action_type in all_action_types, f"Action type {action_type} is not in all environment action types!"
    
    # Extract spaces for AgileRL
    representative_agent = env.possible_agents[0]
    orig_obs_space = env.observation_space(representative_agent)
    
    base_obs_space = orig_obs_space["observation"]
    flat_inner_obs_space = utils.flatten_space(base_obs_space)
    action_mask_space = orig_obs_space["action_mask"]

    # RainbowDQN accepts action_mask as a separate argument to get_action,
    # so the network only needs the flat observation Box (no Dict encoder).
    agilerl_obs_space = flat_inner_obs_space
    action_space = env.action_space(representative_agent)

    # Initialize AgileRL Population
    POP_SIZE = 16
    EVO_STEPS = 25_000

    net_config={
        "encoder_config": {
            "hidden_size": [512, 256]
        },
        "head_config": {
            "hidden_size": [256, 128]
        }
    }

    hp_config = HyperparameterConfig(
        lr=RLParameter(min=1e-5, max=1e-3),
        batch_size=RLParameter(min=64, max=512, dtype=int),
        gamma=RLParameter(min=0.90, max=0.999), 
        tau=RLParameter(min=1e-3, max=0.1)
    )
    
    pop = []
    for i in range(POP_SIZE):
        dqn_agent = RainbowDQN(
            observation_space=agilerl_obs_space,
            action_space=action_space,
            index=i,
            hp_config=hp_config,
            device=device,
            net_config=net_config,
            batch_size=256,
            lr=1e-4,
            learn_step=4,
            gamma=0.99,
            n_step=3,
        )
        # Load checkpoint if provided
        if load_path and os.path.exists(load_path):
            print(f"Loading checkpoint for agent {i} from {load_path}")
            dqn_agent.load_checkpoint(load_path)
            
        dqn_agent.fitness = [] 
        pop.append(dqn_agent)

    tournament = TournamentSelection(
        tournament_size=3,
        elitism=True,
        population_size=POP_SIZE,
        eval_loop=1
    )

    mutations = Mutations(
        no_mutation=0.20,
        architecture=0.30,
        parameters=0.20,
        rl_hp=0.30,
        activation=0.0,
        new_layer_prob=0.30,
        mutation_sd=0.1,
        rand_seed=42,
        device=device
    )
    
    # Off-Policy Replay Buffers
    n_step_buffers = {
        agent: MultiStepReplayBuffer(max_size=1000, n_step=3, gamma=0.99, device=device)
        for agent in env.possible_agents
    }
    # Main population shared experience buffer
    # memory = ReplayBuffer(max_size=250_000, device=device)
    # Replace standard ReplayBuffer with PER
    memory = PrioritizedReplayBuffer(max_size=250_000, alpha=0.6, device=device)

    next_evo_step = EVO_STEPS

    total_steps = 0
    episode = 0
    start_time = time.time()
    last_checkpoint_time = start_time

    # Metric tracking windows
    recent_game_lengths = []
    recent_winning_scores = []
    recent_losing_scores = []

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
            
            # Local caches for continuous transition collection from each agent
            last_state = {agent: None for agent in env.possible_agents}
            last_action_mask = {agent: None for agent in env.possible_agents}
            last_action = {agent: None for agent in env.possible_agents}
            
            # Episode-specific metrics
            ep_turns = 0
            ep_purchases = 0
            ep_reserves = 0
            final_scores = {agent: 0 for agent in env.possible_agents}
            winner_agent = None

            for agent in env.agent_iter():
                obs, reward, term, trunc, info = env.last()
                done = term or trunc

                if use_training_curriculum:
                    obs = apply_curriculum_mask(TRAINING_CURRICULUM, env, obs, episode)
                
                # Flatten observation; action_mask is passed separately to get_action
                flat_inner_state = utils.flatten(base_obs_space, obs["observation"])
                batched_obs = np.expand_dims(flat_inner_state, axis=0).astype(np.float32)
                action_mask = np.expand_dims(obs["action_mask"], axis=0).astype(bool)

                if "points" in info:
                    final_scores[agent] = info["points"]

                if last_state[agent] is not None:
                    # Construct MDP transition (prev_obs, prev_act, rew, obs, done)
                    # All arrays must be float32 to match the buffer's internal dtype
                    td = TensorDict({
                        "obs": last_state[agent],
                        "action": np.array([[last_action[agent]]], dtype=np.int64),
                        "reward": np.array([[reward]], dtype=np.float32),
                        "next_obs": batched_obs,
                        "done": np.array([[float(done)]], dtype=np.float32),
                        # CRITICAL: Add masks so the target network respects game rules
                        "action_mask": last_action_mask[agent],
                        "next_action_mask": action_mask
                    }, batch_size=[1])
                    
                    n_step_td = n_step_buffers[agent].add(td)
                    if n_step_td is not None:
                        memory.add(n_step_td)

                if done:
                    if info.get("is_winner", False):
                        winner_agent = agent
                    env.step(None)
                    continue
                    
                # Action Selection
                current_dqn_agent = agent_mapping[agent]
                action = current_dqn_agent.get_action(batched_obs, action_mask=action_mask)
                scalar_action = int(action[0]) if isinstance(action, np.ndarray) else action

                env.step(scalar_action)
                
                # Metric tracking: Action parsing
                action_type = env.infos[agent]["action_type"]
                if "buy" in action_type:
                    ep_purchases += 1
                elif "reserve" in action_type:
                    ep_reserves += 1

                last_state[agent] = batched_obs
                last_action_mask[agent] = action_mask
                last_action[agent] = scalar_action
                
                ep_turns += 1
                total_steps += 1
                reward -= 0.1

                # # Step-wise DQN updates
                # if len(memory) >= 256 and total_steps % current_dqn_agent.learn_step == 0:
                #     experiences = memory.sample(current_dqn_agent.batch_size)
                #     current_dqn_agent.learn(experiences, per=False)
                
                # Step-wise DQN updates (with PrioritizedReplayBuffer)
                if len(memory) >= 256 and total_steps % current_dqn_agent.learn_step == 0:
                    # Sample with beta (controls importance sampling correction)
                    experiences = memory.sample(current_dqn_agent.batch_size, beta=0.4)
                    
                    # Enable PER and multi-step in the learn function
                    loss, idxs, priorities = current_dqn_agent.learn(experiences, per=True)
                    
                    # Update the buffer with the new priorities based on TD error
                    memory.update_priorities(idxs, priorities)

            # Process Episode Completion
            recent_game_lengths.append(ep_turns)
            
            if winner_agent:
                recent_winning_scores.append(final_scores[winner_agent])
                losing_scores = [score for a, score in final_scores.items() if a != winner_agent]
                if losing_scores:
                    recent_losing_scores.append(np.mean(losing_scores))

            for env_agent in env.possible_agents:
                current_dqn_agent = agent_mapping[env_agent]
                
                # Fitness is directly factored into evolutionary success
                is_win = 1 if env_agent == winner_agent else 0
                points = final_scores[env_agent]
                
                agent_fitness = (is_win * 100) + points
                current_dqn_agent.fitness.append(agent_fitness)
                
                slot_index = pop.index(current_dqn_agent)
                writer.add_scalar(f"Win_Rate/Slot_{slot_index}", is_win, episode)
                
                # Clear trailing data from episode off agent's unrolled multistep buffer
                n_step_buffers[env_agent].clear()

            # Tensorboard Logging every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_game_len = np.mean(recent_game_lengths[-10:])
                avg_win_score = np.mean(recent_winning_scores[-10:]) if recent_winning_scores else 0
                avg_lose_score = np.mean(recent_losing_scores[-10:]) if recent_losing_scores else 0
                buy_reserve_ratio = ep_purchases / max(ep_reserves, 1) # Prevent div by zero
                
                writer.add_scalar("Gameplay/Average_Turns", avg_game_len, episode + 1)
                writer.add_scalar("Gameplay/Winner_Average_Points", avg_win_score, episode + 1)
                writer.add_scalar("Gameplay/Loser_Average_Points", avg_lose_score, episode + 1)
                writer.add_scalar("Gameplay/Purchase_to_Reserve_Ratio", buy_reserve_ratio, episode + 1)
                
                print(f"Ep: {episode + 1} | Turns: {avg_game_len:.1f} | Win Pts: {avg_win_score:.1f} | Lose Pts: {avg_lose_score:.1f} | Buy/Res: {buy_reserve_ratio:.2f}")
            
            # Checkpoint Logic
            current_time = time.time()
            elapsed_minutes_since_ckpt = (current_time - last_checkpoint_time) / 60.0
            if (checkpoint_minutes and elapsed_minutes_since_ckpt >= checkpoint_minutes) or (checkpoint_episodes and (episode + 1) % checkpoint_episodes == 0):
                # Calculate mean on the fly to find the best agent without altering the actual fitness tracking
                best_agent = max(pop, key=lambda x: np.mean(x.fitness) if len(x.fitness) > 0 else -1000)
                ckpt_path = os.path.join(checkpoint_dir, f"rdqn_elite_ep{episode+1}.pt")
                os.makedirs(checkpoint_dir, exist_ok=True)
                best_agent.save_checkpoint(ckpt_path) 
                last_checkpoint_time = current_time
                print(f"--> Checkpoint saved to {ckpt_path}")

            # Evolution Step
            if total_steps >= next_evo_step:
                print(f"--- Evolution Step triggered at {total_steps} steps ---")
                gen_fitness = np.mean([np.mean(p.fitness) for p in pop if len(p.fitness) > 0])
                writer.add_scalar("Evolution/Generational_Mean_Fitness", gen_fitness, episode)
                next_evo_step += EVO_STEPS
               
                for p_agent in pop:
                    p_agent.fitness = [sum(p_agent.fitness) / max(len(p_agent.fitness), 1)] if len(p_agent.fitness) > 0 else [-1000.0]

                elite, survivors = tournament.select(pop)
                pop = mutations.mutation(survivors)
                
                # State Clearing: Prevent off-policy buffer poisoning and reset generational fitness
                for p_agent in pop:
                    p_agent.fitness = []

            episode += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Shutting down...")
    finally:
        elite = max(pop, key=lambda x: np.mean(x.fitness) if len(x.fitness) > 0 else -1000)
        final_path = os.path.join(checkpoint_dir, "rdqn_elite_final.pt")
        os.makedirs(checkpoint_dir, exist_ok=True)
        elite.save_checkpoint(final_path)
        print(f"Final elite model saved to {final_path}")
        writer.flush()
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AgileRL RainbowDQN on Splendor")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint .pt file to resume training")
    parser.add_argument("--episodes", type=int, default=500_000, help="Max episodes to train")
    args = parser.parse_args()

    train(
        max_episodes=args.episodes,
        checkpoint_minutes=10.0,
        checkpoint_episodes=1_000,
        checkpoint_dir="checkpoints_rdqn",
        load_path=args.checkpoint,
        use_training_curriculum=False
    )
