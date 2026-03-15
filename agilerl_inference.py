import torch
import numpy as np
from gymnasium.spaces import utils
from agilerl.algorithms.ppo import PPO
from env.splendor_env import SplendorEnv

def play_against_ai(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Environment
    env = SplendorEnv(num_players=4, render_mode="console")
    env.reset()
    
    # 2. Setup Agent Space Logic
    dummy_agent = env.possible_agents[0]
    base_obs_space = env.observation_space(dummy_agent)["observation"]
    flat_obs_space = utils.flatten_space(base_obs_space)
    action_space = env.action_space(dummy_agent)

    # 3. Load the Trained Agent
    ppo_agent = PPO(
        observation_space=flat_obs_space,
        action_space=action_space,
        device=device,
        net_config={
            "encoder_config": {"hidden_size": [256, 256, 256]}, 
            "head_config": {"hidden_size": [256]} 
        }
    )
    ppo_agent.load_checkpoint(checkpoint_path)
    print(f"Loaded model from {checkpoint_path}")

    # 4. Game Loop
    env.reset()
    human_player = "player_0"
    
    for agent_id in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            action = None
        elif agent_id == human_player:
            # --- Human Turn ---
            env.render()
            mask = observation["action_mask"]
            valid_actions = np.where(mask == 1)[0]
            
            print(f"\n--- Your Turn ({agent_id}) ---")
            print("Action Mapping:")
            for act in valid_actions:
                # Access the 'desc' key from your env's action_mapping
                action_info = env.unwrapped.action_mapping.get(act, {})
                desc = action_info.get("desc", f"Action {act}")
                print(f"  [{act}]: {desc}")
            
            action = None
            while action not in valid_actions:
                try:
                    action = int(input("Enter action ID: "))
                except ValueError:
                    print("Please enter a valid integer.")
        else:
            # --- AI Turn ---
            raw_obs = observation["observation"]
            mask = observation["action_mask"]
            flat_state = utils.flatten(base_obs_space, raw_obs)
            
            # Use training=False to get the greedy (best) action
            action, _, _, _ = ppo_agent.get_action(
                flat_state, 
                action_mask=mask,
            )
            
            if isinstance(action, np.ndarray):
                action = int(action[0])
            
            action_info = env.unwrapped.action_mapping.get(action, {})
            ai_desc = action_info.get("desc", f"Action {action}")
            print(f"AI ({agent_id}) chose: [{action}] {ai_desc}")

        env.step(action)
        
    print("\nGame Over!")
    env.close()

if __name__ == "__main__":
    # Replace with your actual checkpoint file
    PATH = "checkpoints/ppo_splendor_final.pt"
    play_against_ai(PATH)