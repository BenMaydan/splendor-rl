import gymnasium as gym
import numpy as np
import random
import copy

from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

# Import your custom environment here
from env.splendor_env import SplendorEnv

class SelfPlayPoolWrapper(gym.Env):
    """
    A single-agent wrapper that assigns one agent as the 'Learner' and uses 
    a shared pool of historical policies to control the opponents.
    """
    def __init__(self, aec_env, shared_pool, current_policy_container, is_eval=False):
        super().__init__()
        self.env = aec_env
        self.shared_pool = shared_pool
        self.current_policy_container = current_policy_container
        self.is_eval = is_eval
        
        # Define Spaces based on a representative agent
        representative_agent = self.env.possible_agents[0]
        orig_obs_space = self.env.observation_space(representative_agent)
        new_spaces = {k: v for k, v in orig_obs_space["observation"].spaces.items()}
        new_spaces["action_mask"] = orig_obs_space["action_mask"]
        
        self.observation_space = gym.spaces.Dict(new_spaces)
        self.action_space = self.env.action_space(representative_agent)
        self.metadata = self.env.metadata
        
        self.learner_agent = None
        self.opponent_policies = {}

    def _get_obs_and_mask(self):
        """Helper to get current agent's observation and mask."""
        obs, reward, terminated, truncated, info = self.env.last()
        flat_obs = {k: np.copy(v) for k, v in obs["observation"].items()}
        flat_obs["action_mask"] = np.copy(obs["action_mask"])
        return flat_obs, float(reward), terminated, truncated, info

    def _advance_to_learner(self):
        """Steps the environment forward using opponent policies until it's the Learner's turn."""
        while self.env.agent_selection != self.learner_agent:
            curr_agent = self.env.agent_selection
            
            # Check if game is over or agent is dead
            if self.env.terminations[curr_agent] or self.env.truncations[curr_agent]:
                self.env.step(None) # Execute dead step
                
                # If all agents are removed, game is fully over
                if not self.env.agents:
                    break
                continue
            
            # Get opponent's state
            flat_obs, _, _, _, _ = self._get_obs_and_mask()
            policy = self.opponent_policies.get(curr_agent)
            
            # Decide opponent action
            if policy is not None:
                # SB3 policies accept raw observation dicts natively
                action, _ = policy.predict(flat_obs, action_masks=flat_obs["action_mask"], deterministic=False)
                if isinstance(action, np.ndarray):
                    action = action.item()
            else:
                # Fallback to random if pool is completely empty and no current policy exists
                valid_actions = np.where(flat_obs["action_mask"])[0]
                action = np.random.choice(valid_actions)
                
            self.env.step(action)

        # Now it's the Learner's turn, or the game is over and the Learner is waiting for its dead step
        if self.env.agent_selection == self.learner_agent:
            obs, reward, term, trunc, info = self._get_obs_and_mask()
        else:
            # Fallback for unexpected AEC removal
            obs = self.observation_space.sample()
            reward, term, trunc, info = 0.0, True, False, {}
            
        return obs, reward, term, trunc, info

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        
        # Randomize which seat the Learner sits in to prevent turn-order bias
        self.learner_agent = random.choice(self.env.possible_agents)
        
        # Assign policies to opponents for this match
        self.opponent_policies = {}
        for agent in self.env.possible_agents:
            if agent != self.learner_agent:
                if self.is_eval:
                    # In eval, always test against historical models if available
                    if self.shared_pool:
                        self.opponent_policies[agent] = random.choice(self.shared_pool)
                    else:
                        self.opponent_policies[agent] = self.current_policy_container[0]
                else:
                    # In training, 80% chance to play history, 20% chance to play latest self
                    if self.shared_pool and random.random() < 0.80:
                        self.opponent_policies[agent] = random.choice(self.shared_pool)
                    else:
                        self.opponent_policies[agent] = self.current_policy_container[0]
        
        obs, reward, term, trunc, info = self._advance_to_learner()
        return obs, info
        
    def step(self, action):
        # 1. Execute the Learner's action
        self.env.step(action)
        
        # 2. Advance the game through all opponents' turns
        obs, reward, term, trunc, info = self._advance_to_learner()
        
        return obs, float(reward), term, trunc, info

    def action_masks(self) -> np.ndarray:
        obs, _, _, _, _ = self.env.last()
        return np.copy(obs["action_mask"])


class PolicyPoolCallback(BaseCallback):
    """Saves snapshots of the policy network into a shared pool periodically."""
    def __init__(self, shared_pool, current_policy_container, save_freq, max_pool_size=10, verbose=0):
        super().__init__(verbose)
        self.shared_pool = shared_pool
        self.current_policy_container = current_policy_container
        self.save_freq = save_freq
        self.max_pool_size = max_pool_size

    def _on_step(self):
        # Always maintain a reference to the latest active policy
        if self.n_calls == 1:
            self.current_policy_container[0] = self.model.policy

        # Periodically freeze a copy of the weights for the historical pool
        if self.n_calls % self.save_freq == 0:
            historical_policy = copy.deepcopy(self.model.policy)
            
            self.shared_pool.append(historical_policy)
            if len(self.shared_pool) > self.max_pool_size:
                self.shared_pool.pop(0) # Remove oldest
                
        return True


class SplendorStatsCallback(BaseCallback):
    """Tracks custom stats for Splendor including Win Rate and Deadlock Rate."""
    def __init__(self, pass_action_index, verbose=0):
        super().__init__(verbose)
        self.pass_action_index = pass_action_index
        self.total_episodes = 0
        self.deadlocks = 0
        self.wins = 0
        self.pass_count = 0

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            self.pass_count += np.sum(actions == self.pass_action_index)

        infos = self.locals.get("infos")
        if infos is not None:
            for info in infos:
                # Track episodes only when they finish
                if "episode" in info:
                    self.total_episodes += 1
                    
                    if info.get("is_deadlock", False):
                        self.deadlocks += 1
                    else:
                        self.wins += 1

                    if self.total_episodes > 0:
                        win_rate = self.wins / self.total_episodes
                        deadlock_rate = self.deadlocks / self.total_episodes
                        
                        self.logger.record("rates/win_rate", win_rate)
                        self.logger.record("rates/deadlock_rate", deadlock_rate)

        self.logger.record("stats/cumulative_pass_actions", self.pass_count)
        return True

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.action_masks()

def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

def main():
    # --- Shared Memory for Pool ---
    # We use lists to pass by reference between Callbacks and Environments
    shared_pool = []
    current_policy_container = [None] 

    # 1. Initialize Train Env
    aec_env = SplendorEnv(num_players=4, render_mode="console")
    gym_env = SelfPlayPoolWrapper(aec_env, shared_pool, current_policy_container, is_eval=False)
    gym_env = Monitor(gym_env)
    env = ActionMasker(gym_env, mask_fn)

    # 2. Initialize Eval Env
    eval_aec_env = SplendorEnv(num_players=4, render_mode=None)
    eval_gym_env = SelfPlayPoolWrapper(eval_aec_env, shared_pool, current_policy_container, is_eval=True)
    eval_env = ActionMasker(Monitor(eval_gym_env), mask_fn)

    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=500_000, save_path="./logs/checkpoints/", name_prefix="splendor_ppo_mask")
    
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/results",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    pass_action_idx = aec_env._action_indices_map["pass"][0] 
    stats_callback = SplendorStatsCallback(pass_action_index=pass_action_idx)
    
    # Save a new snapshot to the historical pool every 50,000 steps (max 10 past models)
    pool_callback = PolicyPoolCallback(shared_pool, current_policy_container, save_freq=50_000, max_pool_size=10)

    callback_list = CallbackList([eval_callback, stats_callback, pool_callback, checkpoint_callback])

    # 4. Initialize MaskablePPO
    print("Initializing MaskablePPO...")
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        gamma=0.99,
        learning_rate=linear_schedule(3e-4),
        n_steps=8192,
        seed=42,
        verbose=1,
        tensorboard_log="./logs/tensorboard/"
    )

    # 5. Train the agent
    print("Starting training...")
    model.learn(total_timesteps=20_000_000, callback=callback_list)

    # 6. Save the final model
    model.save("splendor_ppo_mask")
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
