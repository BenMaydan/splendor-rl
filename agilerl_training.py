import argparse
import os
import time
import random
import numpy as np
import torch
from gymnasium.spaces import Dict, utils
from torch.utils.tensorboard import SummaryWriter

from agilerl.algorithms.ppo import PPO
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations

from env.splendor_env import SplendorEnv


# Define the stages of your curriculum
# Format: {start_episode: [list_of_types_to_ALLOW]}
TRAINING_CURRICULUM = {
    0: ["take_3_diff_tokens", "take_2_diff_tokens", "take_1_token", "buy_face_up", "discard", "pick_noble"],
    100000: ["take_3_diff_tokens", "take_2_diff_tokens", "take_1_token", "buy_face_up", "buy_reserved", "reserve_face_up", "discard", "pick_noble"],
    250000: None # Disable curriculum (Allow all legal moves)
}


def apply_curriculum_mask(curriculum: dict[int, list[str]], env: SplendorEnv, observation, episode):
    """
    Masks specific action types based on the current training episode.
    """
    # Find the current stage
    active_stage = None
    for start_ep in sorted(curriculum.keys()):
        if episode >= start_ep:
            active_stage = curriculum[start_ep]
    
    # If we are in a restricted stage, refine the mask
    if active_stage is not None:
        new_mask = np.zeros_like(observation["action_mask"])
        for action_type in active_stage:
            s, e = env._action_indices_map[action_type]
            # Only allow the action if it was already legal in the base environment
            new_mask[s:e] = observation["action_mask"][s:e]
        
        # Ensure 'pass' is always available as a safety valve
        s_p, e_p = env._action_indices_map["pass"]
        if np.sum(new_mask) == 0:
            new_mask[s_p:e_p] = 1
            
        observation["action_mask"] = new_mask
        env.action_mask = new_mask

    return observation


def train(
        max_episodes=1000,
        max_hours=None,
        checkpoint_minutes=None,
        checkpoint_episodes=None,
        checkpoint_dir="checkpoints",
        load_path=None,
        use_training_curriculum=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    writer = SummaryWriter(log_dir="runs/splendor_ppo")
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
    
    agilerl_obs_space = Dict({
        "observation": flat_inner_obs_space,
        "action_mask": orig_obs_space["action_mask"]
    })
    action_space = env.action_space(representative_agent)

    # Initialize AgileRL PPO Population & HPO Setup
    update_steps = 2048
    POP_SIZE = 16
    EVO_STEPS = 25_000

    net_config={
        "encoder_config": {
            "latent_dim": 256,
            "max_latent_dim": 1024,
            "mlp_config": {"hidden_size": [512, 512]}
        },
        "head_config": {
            # A slightly deeper head for complex action evaluation
            "hidden_size": [256, 256]
        }
    }

    hp_config = HyperparameterConfig(
        # Logarithmic Parameters: Can handle 20% swings
        lr=RLParameter(min=1e-5, max=1e-3, grow_factor=1.2, shrink_factor=0.8),
        batch_size=RLParameter(min=64, max=512, dtype=int, grow_factor=1.2, shrink_factor=0.8),
        
        # Sensitive Coefficients: Restrict to 10% swings to prevent loss function collapse
        ent_coef=RLParameter(min=0.001, max=0.05, grow_factor=1.1, shrink_factor=0.9),
        clip_coef=RLParameter(min=0.1, max=0.3, grow_factor=1.1, shrink_factor=0.9),
        vf_coef=RLParameter(min=0.1, max=1.0, grow_factor=1.1, shrink_factor=0.9),
        
        # Hyper-Sensitive Horizon Parameters: Microscopic swings (0.2% to 0.5%)
        # Your previous 1.0010204... was on the right track, but 1.002 is much cleaner and just as safe.
        gamma=RLParameter(min=0.98, max=0.999, grow_factor=1.002, shrink_factor=0.998),
        gae_lambda=RLParameter(min=0.90, max=0.99, grow_factor=1.005, shrink_factor=0.995)
    )
    
    pop = []
    for i in range(POP_SIZE):
        ppo_agent = PPO(
            observation_space=agilerl_obs_space,
            action_space=action_space,
            index=i,
            hp_config=hp_config,
            device=device,
            net_config=net_config,

            # --- Baseline for Evolvable Params ---
            batch_size=256,
            ent_coef=0.01,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,       # Baseline clip
            vf_coef=0.5,         # Baseline value coefficient
            
            # --- Static Safety Parameters ---
            max_grad_norm=0.5,   # Prevents math errors (exploding gradients)
            target_kl=0.015,     # Early stopping for the epochs
            update_epochs=8,     # Safe to leave at 8 because target_kl will stop it early if needed
            learn_step=update_steps,
            
            # --- Architecture Flags ---
            share_encoders=True, # Efficient learning for board games
            use_rollout_buffer=True,
        )
        # Load checkpoint if provided
        if load_path and os.path.exists(load_path):
            print(f"Loading checkpoint for agent {i} from {load_path}")
            ppo_agent.load_checkpoint(load_path)
            
        # Track episodic rewards for this specific agent manually
        ppo_agent.fitness = [] # AgileRL expects a list
        pop.append(ppo_agent)

    tournament = TournamentSelection(
        tournament_size=3,
        elitism=True,
        population_size=POP_SIZE,
        eval_loop=1
    )

    mutations = Mutations(
        # Relative probabilities normalized to sum to 1.0
        no_mutation=0.10,   # 10% chance to pass through untouched
        architecture=0.0,   # 0% chance - Architecture is now completely locked
        parameters=0.40,    # 40% chance to mutate existing weights (increased)
        rl_hp=0.50,         # 50% chance to mutate learning rate, gamma, etc. (increased)
        activation=0.0,
        
        # Conditional probability: If architecture mutation triggers, 30% chance it's a new layer
        new_layer_prob=0.10,
        
        mutation_sd=0.1,
        
        rand_seed=42,
        device=device
    )
    
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
            
            # Episode-specific metrics
            ep_turns = 0
            ep_purchases = 0
            ep_reserves = 0
            final_scores = {agent: 0 for agent in env.possible_agents}
            winner_agent = None

            for agent in env.agent_iter():
                obs, reward, term, trunc, info = env.last()
                done = term or trunc

                # Apply curriculum for training
                # We apply the mask to the raw observation before flattening
                if use_training_curriculum:
                    obs = apply_curriculum_mask(TRAINING_CURRICULUM, env, obs, episode)
                
                # Flatten observation to match AgileRL expectations
                flat_inner_state = utils.flatten(base_obs_space, obs["observation"])
                batched_state = {
                    "observation": np.expand_dims(flat_inner_state, axis=0),
                    "action_mask": np.expand_dims(obs["action_mask"], axis=0).astype(bool)
                }

                # Update final scores if provided in info dict
                if "points" in info:
                    final_scores[agent] = info["points"]

                if last_state[agent] is not None:
                    # Append the transition to this agent's temporary local list
                    trajectories[agent]["obs"].append(last_state[agent])
                    trajectories[agent]["action"].append(last_action[agent])
                    trajectories[agent]["reward"].append(reward)
                    trajectories[agent]["done"].append(done)
                    trajectories[agent]["value"].append(last_value[agent])
                    trajectories[agent]["log_prob"].append(last_log_prob[agent])
                    trajectories[agent]["action_mask"].append(last_action_mask[agent])

                if done:
                    if info.get("is_winner", False):
                        winner_agent = agent
                    env.step(None)
                    continue
                    
                # Action Selection
                current_ppo_agent = agent_mapping[agent]
                action, log_prob, _, value = current_ppo_agent.get_action(batched_state, action_mask=batched_state["action_mask"])
                scalar_action = int(action[0]) if isinstance(action, np.ndarray) else action

                env.step(scalar_action)
                
                # Metric tracking: Action parsing
                action_type = env.infos[agent]["action_type"]
                if "buy" in action_type:
                    ep_purchases += 1
                elif "reserve" in action_type:
                    ep_reserves += 1

                last_state[agent] = batched_state
                last_action_mask[agent] = batched_state["action_mask"]
                last_action[agent] = scalar_action
                last_value[agent] = value
                last_log_prob[agent] = log_prob
                
                ep_turns += 1
                total_steps += 1

            # Process Episode Completion
            recent_game_lengths.append(ep_turns)
            
            if winner_agent:
                recent_winning_scores.append(final_scores[winner_agent])
                losing_scores = [score for a, score in final_scores.items() if a != winner_agent]
                if losing_scores:
                    recent_losing_scores.append(np.mean(losing_scores))

            for env_agent in env.possible_agents:
                traj_len = len(trajectories[env_agent]["obs"])
                current_ppo_agent = agent_mapping[env_agent]
                
                # -- FITNESS: Cumulative episode reward --
                agent_fitness = sum(trajectories[env_agent]["reward"])
                current_ppo_agent.fitness.append(agent_fitness)
                
                # Monitoring only: game-level outcomes for your own analysis
                slot_index = pop.index(current_ppo_agent)
                is_win = 1 if env_agent == winner_agent else 0
                points = final_scores[env_agent]
                writer.add_scalar(f"Population/WinRate_Slot_{slot_index}", is_win, episode)
                writer.add_scalar(f"Population/Points_Slot_{slot_index}", points, episode)
                
                if current_ppo_agent.rollout_buffer.size() + traj_len > update_steps:
                    if current_ppo_agent.rollout_buffer.size() > 0:
                        current_ppo_agent.rollout_buffer.compute_returns_and_advantages(
                            last_value=np.array([0.0]),
                            last_done=np.array([True])
                        )
                        loss = current_ppo_agent.learn()
                        writer.add_scalar("Loss/Policy", loss, total_steps)
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
                ckpt_path = os.path.join(checkpoint_dir, f"ppo_elite_ep{episode+1}.pt")
                os.makedirs(checkpoint_dir, exist_ok=True)
                best_agent.save_checkpoint(ckpt_path) 
                last_checkpoint_time = current_time
                print(f"--> Checkpoint saved to {ckpt_path}")

            # Evolution Step
            if total_steps >= next_evo_step:
                print(f"--- Evolution Step triggered at {total_steps} steps ---")
                # Log Generational Mean Fitness before selection
                gen_fitness = np.mean([np.mean(p.fitness) for p in pop if len(p.fitness) > 0])
                writer.add_scalar("Evolution/Generational_Mean_Fitness", gen_fitness, episode)
                next_evo_step += EVO_STEPS
                
                # Condense the generation's noisy array of episodic rewards into a single stable mean
                # so that TournamentSelection (with eval_loop=1) evaluates true generational average performance!
                for p_agent in pop:
                    p_agent.fitness = [sum(p_agent.fitness) / len(p_agent.fitness)] if len(p_agent.fitness) > 0 else [-1000.0]

                elite, pop = tournament.select(pop)

                # Log Architecture Hyperparameters of the Elite
                # Access the net_config which AgileRL updates during mutation
                if hasattr(elite, 'net_config'):
                    # Log Encoder Hidden Sizes
                    encoder_layers = elite.net_config.get("encoder_config", {}).get("mlp_config", {}).get("hidden_size", [])
                    for idx, size in enumerate(encoder_layers):
                        writer.add_scalar(f"Hyperparameters/Elite_Encoder_Layer_{idx}_Size", size, episode)
                    
                    # Log Head Hidden Sizes
                    head_layers = elite.net_config.get("head_config", {}).get("hidden_size", [])
                    for idx, size in enumerate(head_layers):
                        writer.add_scalar(f"Hyperparameters/Elite_Head_Layer_{idx}_Size", size, episode)
                        
                    # Log Latent Dimension
                    latent_dim = elite.net_config.get("encoder_config", {}).get("latent_dim", 0)
                    writer.add_scalar("Hyperparameters/Elite_Latent_Dim", latent_dim, episode)
                
                writer.add_scalar("Hyperparameters/Elite_LR", elite.lr, episode)
                writer.add_scalar("Hyperparameters/Elite_Batch_Size", elite.batch_size, episode)
                writer.add_scalar("Hyperparameters/Elite_Ent_Coef", elite.ent_coef, episode)
                writer.add_scalar("Hyperparameters/Elite_Clip_Coef", elite.clip_coef, episode)
                writer.add_scalar("Hyperparameters/Elite_Vf_Coef", elite.vf_coef, episode)
                writer.add_scalar("Hyperparameters/Elite_Gamma", elite.gamma, episode)
                writer.add_scalar("Hyperparameters/Elite_Gae_Lambda", elite.gae_lambda, episode)

                # Proceed with mutations
                pop = mutations.mutation(pop)
                
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
    parser = argparse.ArgumentParser(description="Train AgileRL PPO on Splendor")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint .pt file to resume training")
    parser.add_argument("--episodes", type=int, default=500_000, help="Max episodes to train")
    args = parser.parse_args()

    # Pass the checkpoint path to the train function
    train(
        max_episodes=args.episodes,
        checkpoint_minutes=10.0,
        checkpoint_episodes=1_000,
        checkpoint_dir="checkpoints",
        load_path=args.checkpoint,
        use_training_curriculum=False
    )