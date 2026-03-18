import argparse
import gymnasium as gym
import numpy as np
import random
import copy
import os
import glob

from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

# Import your custom environment here
from env.splendor_env import SplendorEnv


class CurriculumSchedule:
    def __init__(self, total_timesteps, start_step=2_000_000, min_weight=0.2, max_weight=0.8):
        self.total_timesteps = total_timesteps
        self.start_step = start_step
        self.end_step = total_timesteps
        self.min_weight = min_weight
        self.max_weight = max_weight

    def get_pool_weight(self, current_decisions):
        # Phase 1: Flat 0% (Pure self-play)
        if current_decisions < self.start_step:
            return 0.0
        
        # Phase 2: Linear increase from min_weight (20%) to max_weight (80%)
        # Calculate progress between start_step and end_step
        progress = (current_decisions - self.start_step) / (self.end_step - self.start_step)
        progress = min(max(progress, 0.0), 1.0) # Clip between 0 and 1
        
        return self.min_weight + (progress * (self.max_weight - self.min_weight))
    
    def get_start_steps(self):
        return self.start_step


class SelfPlayPoolWrapper(gym.Env):
    """
    A single-agent wrapper that assigns one agent as the 'Learner' and uses 
    a shared pool of historical policies to control the opponents.
    """
    def __init__(self, aec_env, shared_pool, current_policy_container, shared_decision_counter, curriculum_schedule=None, is_eval=False):
        super().__init__()
        self.env = aec_env
        self.shared_pool = shared_pool
        self.current_policy_container = current_policy_container
        self.shared_decision_counter = shared_decision_counter
        self.is_eval = is_eval
        self.curriculum_schedule = curriculum_schedule
        
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
                # Track if the opponent is using the latest policy
                if not self.is_eval and policy == self.current_policy_container[0]:
                    self.shared_decision_counter[0] += 1

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

        # Get the current weight from the schedule
        if self.curriculum_schedule and not self.is_eval:
            # Sync schedule with the true number of decisions made
            pool_weight = self.curriculum_schedule.get_pool_weight(self.shared_decision_counter[0])
        else:
            pool_weight = 1.0 if self.is_eval else 0.5 # Default behavior with no curriculum schedule
        
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
                    # Use the scheduled weight to decide Pool vs. Latest
                    if self.shared_pool and random.random() < pool_weight:
                        self.opponent_policies[agent] = random.choice(self.shared_pool)
                    else:
                        self.opponent_policies[agent] = self.current_policy_container[0]
        
        obs, reward, term, trunc, info = self._advance_to_learner()
        return obs, info
        
    def step(self, action):
        # The Learner always uses the latest policy, so we track its decision
        if not self.is_eval:
            self.shared_decision_counter[0] += 1

        # Execute the Learner's action
        self.env.step(action)

        # Advance the game through all opponents' turns
        obs, reward, term, trunc, info = self._advance_to_learner()
        
        return obs, float(reward), term, trunc, info

    def action_masks(self) -> np.ndarray:
        obs, _, _, _, _ = self.env.last()
        return np.copy(obs["action_mask"])


class PolicyPoolCallback(BaseCallback):
    """Saves snapshots of the policy network into a shared pool periodically."""
    def __init__(self, shared_pool, current_policy_container, shared_decision_counter, save_freq, curriculum_schedule: CurriculumSchedule=None, max_pool_size=10, verbose=0):
        super().__init__(verbose)
        self.shared_pool = shared_pool
        self.current_policy_container = current_policy_container
        self.shared_decision_counter = shared_decision_counter # Read the shared state
        self.save_freq = save_freq
        self.max_pool_size = max_pool_size
        self.curriculum_schedule = curriculum_schedule

    def _on_step(self):
        # Always maintain a reference to the latest active policy
        if self.n_calls == 1:
            self.current_policy_container[0] = self.model.policy

        # Periodically freeze a copy of the weights for the historical pool
        start_steps = self.curriculum_schedule.get_start_steps() if self.curriculum_schedule else 0
        
        # We still save every N 'n_calls' (SB3 steps) for stability, 
        # but we gate the start of the saving based on the actual true decisions.
        if self.n_calls % self.save_freq == 0 and self.shared_decision_counter[0] >= start_steps:
            historical_policy = copy.deepcopy(self.model.policy)
            
            self.shared_pool.append(historical_policy)
            if len(self.shared_pool) > self.max_pool_size:
                self.shared_pool.pop(0) # Remove oldest
                
        return True


class SplendorStatsCallback(BaseCallback):
    """Tracks custom stats for Splendor including Win Rate and Deadlock Rate."""
    def __init__(self, pass_action_index, shared_decision_counter, verbose=0):
        super().__init__(verbose)
        self.pass_action_index = pass_action_index
        self.shared_decision_counter = shared_decision_counter
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

        self.logger.record("stats/total_actions_taken", self.shared_decision_counter[0])
        self.logger.record("stats/cumulative_pass_actions", self.pass_count)
        return True

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.action_masks()

def linear_schedule(initial_value, total_timesteps, already_done=0):
    """
    Linearly decreases learning rate from initial_value to 0, 
    accounting for steps already completed in a previous run.
    """
    def func(progress_remaining):
        # progress_remaining starts at 1.0 and goes to 0.0 for THIS learn() call.
        # We need to map this back to the global progress.
        
        current_steps_in_this_run = (1.0 - progress_remaining) * (total_timesteps - already_done)
        total_global_steps = already_done + current_steps_in_this_run
        
        global_progress_remaining = 1.0 - (total_global_steps / total_timesteps)
        return max(0, global_progress_remaining * initial_value)
        
    return func

def main():
    parser = argparse.ArgumentParser(description="Train Splendor RL Agent")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to the checkpoint .zip file to resume from.")
    parser.add_argument("--initial-decisions", type=int, default=0, 
                        help="The number of decisions already made, to correctly resume the curriculum schedule.")
    parser.add_argument("--total-timesteps", type=int, default=20_000_000, 
                        help="Total timesteps for this training run.")
    args = parser.parse_args()

    # --- Shared Memory for Pool ---
    shared_pool = []
    current_policy_container = [None] 
    shared_decision_counter = [args.initial_decisions] # Resume the internal game clock

    # Initialize Schedule (Assuming you want the pool to max out around the end of training)
    curriculum = CurriculumSchedule(
        total_timesteps=args.total_timesteps,
        start_step=2_000_000
    )

    # 1. Initialize Train Env
    aec_env = SplendorEnv(num_players=4, render_mode="console")
    gym_env = SelfPlayPoolWrapper(aec_env, shared_pool, current_policy_container, shared_decision_counter, curriculum_schedule=curriculum, is_eval=False)
    gym_env = Monitor(gym_env)
    env = ActionMasker(gym_env, mask_fn)

    # 2. Initialize Eval Env
    eval_aec_env = SplendorEnv(num_players=4, render_mode=None)
    eval_gym_env = SelfPlayPoolWrapper(eval_aec_env, shared_pool, current_policy_container, shared_decision_counter, curriculum_schedule=curriculum, is_eval=True)
    eval_env = ActionMasker(Monitor(eval_gym_env), mask_fn)

    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path="./logs/checkpoints/", name_prefix="splendor_ppo_mask")
    
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/results",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    pass_action_idx = aec_env._action_indices_map["pass"][0] 
    stats_callback = SplendorStatsCallback(pass_action_index=pass_action_idx, shared_decision_counter=shared_decision_counter)
    
    # Pass the shared counter and curriculum to the pool callback
    pool_callback = PolicyPoolCallback(shared_pool, current_policy_container, shared_decision_counter, save_freq=50_000, curriculum_schedule=curriculum, max_pool_size=10)

    callback_list = CallbackList([eval_callback, stats_callback, pool_callback, checkpoint_callback])

    # Calculate the new schedule
    lr_schedule = linear_schedule(
        initial_value=3e-4, 
        total_timesteps=args.total_timesteps, 
        already_done=args.initial_decisions
    )

    # 4. Initialize or Load MaskablePPO
    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}...")
        model = MaskablePPO.load(
            args.checkpoint,
            env=env,
            custom_objects={"learning_rate": lr_schedule}, # This overrides the old schedule
            tensorboard_log="./logs/tensorboard/" # Ensure logging connects to the existing directory
        )

        checkpoint_dir = os.path.dirname(args.checkpoint)
        # Find all .zip files in the checkpoint directory
        all_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
        
        # Sort them by modification time (or you could parse the step number from the name)
        all_checkpoints.sort(key=os.path.getmtime)
        
        # Take the 10 most recent checkpoints (excluding the one we are currently loading to train)
        # We want "historical" versions, not the current one.
        pool_candidates = [cp for cp in all_checkpoints if cp != args.checkpoint][-10:]
        
        print(f"Pre-filling pool with {len(pool_candidates)} historical checkpoints...")
        for cp_path in pool_candidates:
            print(f"Loading weights from {os.path.basename(cp_path)}...")

            # 1. Pass custom_objects to prevent cloudpickle segfaults
            # 2. Append the whole model to prevent orphaned C++ tensor references
            temp_model = MaskablePPO.load(
                cp_path,
                device="cpu",
                custom_objects={"learning_rate": lr_schedule}
            )
            temp_model.policy.eval()
            shared_pool.append(temp_model)
    else:
        print("Initializing MaskablePPO from scratch...")
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            gamma=0.99,
            learning_rate=lr_schedule,
            n_steps=32768,
            seed=42,
            verbose=1,
            tensorboard_log="./logs/tensorboard/"
        )

    # 5. Train the agent
    print("Starting training...")
    model.learn(
        total_timesteps=args.total_timesteps, 
        callback=callback_list,
        reset_num_timesteps=not bool(args.checkpoint) # CRITICAL: False when resuming to maintain schedules
    )

    # 6. Save the final model
    model.save("splendor_ppo_mask")
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()