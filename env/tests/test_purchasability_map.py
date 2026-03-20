import numpy as np
from numpy.typing import NDArray
from splendor_env import SplendorEnv
from tqdm import tqdm
from tests.utils import initialize_tokens_in_hand, initialize_discounts
from numba import njit

@njit
def is_purchasable(color_indices, gold_index, card, tokens_in_hand, discounts):
    gold_needed = 0
    for i in range(len(color_indices)):
        # Using a simple if instead of max() is often faster in JIT
        diff = card[color_indices[i]] - discounts[i] - tokens_in_hand[i]
        if diff > 0:
            gold_needed += diff
    return tokens_in_hand[gold_index] >= gold_needed

def test_single_player_single_card(env: SplendorEnv):
    player = 0
    tier, slot = 0, 0
    
    card = env.dealt[tier, slot]
    tokens = env.tokens_in_hand[player]
    discounts = env.discounts[player]
    
    # Function should return a 0D boolean scalar (or 1D if you forced flattening, but our latest returns native spatial shape)
    purchasable_map = env.get_purchasability_map(tokens, discounts, card)
    expected = is_purchasable(env.color_indices, env.gold_index, card, tokens, discounts)
    
    assert purchasable_map.shape == (), f"Expected scalar shape (), got {purchasable_map.shape}"
    assert purchasable_map == expected, "Single player single card logic mismatch"

def test_single_player_multiple_cards(env: SplendorEnv):
    player = 0
    tokens = env.tokens_in_hand[player]
    discounts = env.discounts[player]
    cards = env.dealt # Shape: (tiers, slots, features)
    
    purchasable_map = env.get_purchasability_map(tokens, discounts, cards)
    assert purchasable_map.shape == (env.num_tiers, env.num_slots), f"Expected {(env.num_tiers, env.num_slots)}, got {purchasable_map.shape}"

    color_indices = env.color_indices
    gold_index = env.gold_index
    
    expected_map = np.zeros((env.num_tiers, env.num_slots), dtype=bool)
    for tier in range(env.num_tiers):
        for slot in range(env.num_slots):
            expected_map[tier, slot] = is_purchasable(color_indices, gold_index, cards[tier, slot], tokens, discounts)
            
    assert np.all(purchasable_map == expected_map), "Single player multiple cards logic mismatch"

def test_multiple_players_multiple_cards(env: SplendorEnv):
    # Repeat the dealt cards along a new player axis to cleanly pass the function's strict shape assertions
    batched_dealt = np.repeat(env.dealt[np.newaxis, ...], env.num_players, axis=0)
    
    purchasability_map = env.get_purchasability_map(env.tokens_in_hand, env.discounts, batched_dealt)
    assert purchasability_map.shape == (env.num_players, env.num_tiers, env.num_slots)

    color_indices = env.color_indices
    gold_index = env.gold_index
    
    non_vectorized_purchasibility_map = np.zeros((env.num_players, env.num_tiers, env.num_slots), dtype=bool)
    for player in range(env.num_players):
        for tier in range(env.num_tiers):
            for slot in range(env.num_slots):
                non_vectorized_purchasibility_map[player, tier, slot] = is_purchasable(
                    color_indices,
                    gold_index,
                    env.dealt[tier, slot],
                    env.tokens_in_hand[player],
                    env.discounts[player]
                )
                
    assert np.all(purchasability_map == non_vectorized_purchasibility_map), "Multiple players multiple cards logic mismatch"


if __name__ == "__main__":
    env = SplendorEnv(num_players=4)
    env.reset(seed=1293987)

    # Fuzz tests
    print("Testing vectorized purchasability maps...")
    for test in tqdm(range(10_000)):
        env.reset()
        env.tokens_in_hand = initialize_tokens_in_hand(env)
        env.discounts = initialize_discounts(env)
        
        test_single_player_single_card(env)
        test_single_player_multiple_cards(env)
        test_multiple_players_multiple_cards(env)
        
    print("All vectorized purchasability tests passed successfully!")