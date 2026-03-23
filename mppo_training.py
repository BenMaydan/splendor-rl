import gymnasium as gym
import numpy as np
import os

from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy

# Import your custom environment here
from env.splendor_env import SplendorEnv

class PettingZooToGymWrapper(gym.Env):
    def __init__(self, aec_env):
        super().__init__()
        self.env = aec_env
        
        # Use the first possible agent just to define the SHAPE of the spaces.
        # In Splendor, all agents should have identical space structures.
        representative_agent = self.env.possible_agents[0]
        
        orig_obs_space = self.env.observation_space(representative_agent)
        new_spaces = {k: v for k, v in orig_obs_space["observation"].spaces.items()}
        new_spaces["action_mask"] = orig_obs_space["action_mask"]
        
        self.observation_space = gym.spaces.Dict(new_spaces)
        self.action_space = self.env.action_space(representative_agent)
        self.metadata = self.env.metadata

    def _get_obs_and_mask(self):
        """Helper to get current agent's observation and mask."""
        obs, reward, terminated, truncated, info = self.env.last()
        
        # Use np.copy() to ensure we aren't passing references to 
        # internal environment arrays that will change next step.
        flat_obs = {k: np.copy(v) for k, v in obs["observation"].items()}
        flat_obs["action_mask"] = np.copy(obs["action_mask"])
        
        return flat_obs, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        # AEC reset doesn't return anything; we must call last()
        obs, reward, terminated, truncated, info = self._get_obs_and_mask()
        return obs, info
        
    def step(self, action):
        # Identify who is acting right now
        current_agent = self.env.agent_selection
        
        # 1. Execute the action
        self.env.step(action)
        
        # 2. Get the reward specifically for the agent that just acted
        actual_reward = self.env.rewards[current_agent]
        
        # 3. Get the state for the NEXT agent in the AEC cycle
        obs, _, terminated, truncated, info = self._get_obs_and_mask()
        
        # Return the captured reward, not the next agent's reward
        return obs, float(actual_reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        # Crucial: Return the mask for whoever the AEC says is next
        obs, _, _, _, _ = self.env.last()
        return np.copy(obs["action_mask"])

class TrackPassCallback(BaseCallback):
    def __init__(self, pass_action_index, verbose=0):
        super().__init__(verbose)
        self.pass_action_index = pass_action_index
        self.pass_count = 0

    def _on_step(self) -> bool:
        # Extract the action taken in the last step
        # actions is a numpy array because of VecEnv
        actions = self.locals.get("actions")

        if actions is not None:
            # Increment count for every instance of a 'Pass' action in the batch
            self.pass_count += np.sum(actions == self.pass_action_index)
        
        # Track Deadlocks
        # 'infos' is a list of dicts (one for each env in a VecEnv)
        infos = self.locals.get("infos")
        if infos is not None:
            for info in infos:
                # In SB3, when an episode ends, the 'info' dict contains 
                # the final info of the episode.
                if "is_deadlock" in info and info.get("is_deadlock"):
                    # Check if the episode actually ended this step
                    # (Standard SB3 logic for detecting end of episode in VecEnvs)
                    if "terminal_observation" in info or info.get("TimeLimit.truncated"):
                         self.deadlock_count += 1

        # Log to TensorBoard
        self.logger.record("stats/cumulative_pass_actions", self.pass_count)
        self.logger.record("stats/cumulative_deadlocks", self.deadlock_count)
        
        return True

def mask_fn(env: gym.Env) -> np.ndarray:
    """Helper function for ActionMasker wrapper."""
    return env.unwrapped.action_masks()

def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

def main():
    # 1. Initialize your base AEC environment
    aec_env = SplendorEnv(num_players=4, render_mode="console")

    def make_env():
        env = PettingZooToGymWrapper(aec_env)
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        return env

    # 3. Create Vectorized Env (Required for VecNormalize)
    # Even if you only have 1 env, SB3 needs the VecEnv interface
    venv = DummyVecEnv([make_env])

    # 4. Wrap with VecNormalize
    # norm_obs=True scales the gem counts/board state
    # norm_reward=True scales that 350 max reward down to ~1.0 range
    # For now we don't normalize observation since we have non spaces.Box observations, which are incompatible
    env = VecNormalize(
        venv,
        norm_obs=False,
        norm_reward=True,
        clip_obs=10.,
        gamma=0.998,
    )

    # 5. Set up Callbacks

    # Saves the model every N steps
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000,
        save_path="./logs/checkpoints/",
        name_prefix="splendor_ppo_mask"
    )

    # Set up Maskable Eval Callback
    # We create a separate identical environment for evaluation to prevent messing up training state
    eval_aec_env = SplendorEnv(num_players=4, render_mode=None)

    def make_eval_env():
        env = PettingZooToGymWrapper(eval_aec_env)
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        return env

    # Wrap in DummyVecEnv just like the training environment
    eval_venv = DummyVecEnv([make_eval_env])

    # Wrap with VecNormalize, ensuring training and reward normalization are disabled
    eval_env = VecNormalize(
        eval_venv,
        training=False,
        norm_reward=False,
        norm_obs=False,
        clip_obs=10.
    )
    
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/results",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # Get the index of the Pass action from the environment's internal mapping
    # Based on your SplendorEnv, it's the start of the 'pass' slice
    pass_action_idx = aec_env._action_indices_map["pass"][0] 
    pass_tracking_callback = TrackPassCallback(pass_action_index=pass_action_idx)
    callback_list = CallbackList([eval_callback, pass_tracking_callback, checkpoint_callback])

    # 6. Initialize MaskablePPO
    # MultiInputPolicy is used because observation_space is a spaces.Dict
    print("Initializing MaskablePPO...")

    # Define a wider and deeper network
    # For example: 3 hidden layers (depth), with 256 neurons each (width)
    custom_policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256])
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=custom_policy_kwargs,
        gamma=0.99,
        learning_rate=linear_schedule(3e-4), # Starts at 3e-4 and drops to 0
        n_steps=8192,
        seed=42,
        verbose=1,
        tensorboard_log="./logs/tensorboard/"
    )

    # 7. Train the agent
    print("Starting training...")
    model.learn(total_timesteps=20_000_000, callback=callback_list)

    # 8. Save the final model and VecNormalize stats
    model.save("splendor_ppo_mask")
    env.save("vec_normalize.pkl")  # Save the normalization stats!
    print("Training complete and model saved.")

    # 9. Clean up and demonstrate loading/running
    del model
    model = MaskablePPO.load("splendor_ppo_mask")

    # Run a test game
    print("\nRunning a test game with the trained agent...")
    test_aec_env = SplendorEnv(num_players=4, render_mode="console")

    def make_test_env():
        test_wrapped = PettingZooToGymWrapper(test_aec_env)
        return ActionMasker(test_wrapped, mask_fn)

    test_venv = DummyVecEnv([make_test_env])

    # Load the saved normalization stats and disable training/reward norm
    test_env = VecNormalize.load("vec_normalize.pkl", test_venv)
    test_env.training = False
    test_env.norm_reward = False

    obs = test_env.reset()
    done = False

    while not done:
        # ActionMasker requires pulling the mask from the unwrapped env directly
        # Since test_env is now wrapped in VecNormalize and DummyVecEnv, we access the unwrapped env
        action_masks = test_env.venv.envs[0].unwrapped.action_masks()

        # Predict the action, ensuring we pass the action mask
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)

        # Step the environment
        obs, reward, done, info = test_env.step(action)

    print("Test game finished!")

if __name__ == "__main__":
    main()
