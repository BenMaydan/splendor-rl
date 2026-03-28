import numpy as np
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
