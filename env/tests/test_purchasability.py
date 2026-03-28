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
    available = card[env.card_column_indexer['available']] == 1
    tokens = env.tokens_in_hand[player]
    discounts = env.discounts[player]
    
    # Function should return a 0D boolean scalar (or 1D if you forced flattening, but our latest returns native spatial shape)
    purchasable_map = env.get_purchasability_map(tokens, discounts, card)
    expected = available and is_purchasable(env.color_indices, env.gold_index, card, tokens, discounts)
    
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
            card = cards[tier, slot]
            available = card[env.card_column_indexer['available']] == 1
            expected_map[tier, slot] = available and is_purchasable(color_indices, gold_index, card, tokens, discounts)
            
    assert np.all(purchasable_map == expected_map), "Single player multiple cards logic mismatch"

def test_multiple_players_multiple_cards(env: SplendorEnv):
    # Repeat the dealt cards along a new player axis to cleanly pass the function's strict shape assertions
    purchasability_map = env.get_purchasability_map(env.tokens_in_hand, env.discounts, env.dealt)
    assert purchasability_map.shape == (env.num_players, env.num_tiers, env.num_slots)

    color_indices = env.color_indices
    gold_index = env.gold_index
    
    non_vectorized_purchasibility_map = np.zeros((env.num_players, env.num_tiers, env.num_slots), dtype=bool)
    for player in range(env.num_players):
        for tier in range(env.num_tiers):
            for slot in range(env.num_slots):
                card = env.dealt[tier, slot]
                available = card[env.card_column_indexer['available']] == 1
                non_vectorized_purchasibility_map[player, tier, slot] = available and is_purchasable(
                    color_indices,
                    gold_index,
                    card,
                    env.tokens_in_hand[player],
                    env.discounts[player]
                )
                
    assert np.all(purchasability_map == non_vectorized_purchasibility_map), "Multiple players multiple cards logic mismatch"

def test_relative_costs_single_player(env: SplendorEnv):
    player = 0
    tokens = env.tokens_in_hand[player]
    discounts = env.discounts[player]
    cards = env.dealt
    
    deficit, gold = env.get_relative_costs(tokens, discounts, cards)
    
    assert deficit.shape == (env.num_tiers, env.num_slots, len(env.colors))
    assert gold.shape == ()
    
    for tier in range(env.num_tiers):
        for slot in range(env.num_slots):
            card = cards[tier, slot]
            raw_costs = card[env.color_indices]
            expected_deficit = np.maximum(0, raw_costs - discounts - tokens[:len(env.colors)])
            assert np.all(deficit[tier, slot] == expected_deficit)
            assert gold == tokens[env.gold_index]

def test_relative_costs_multiple_players(env: SplendorEnv):
    cards = env.dealt
    deficit, gold = env.get_relative_costs(env.tokens_in_hand, env.discounts, cards)
    
    assert deficit.shape == (env.num_players, env.num_tiers, env.num_slots, len(env.colors))
    assert gold.shape == (env.num_players, 1, 1)
    
    for player in range(env.num_players):
        tokens = env.tokens_in_hand[player]
        discounts = env.discounts[player]
        for tier in range(env.num_tiers):
            for slot in range(env.num_slots):
                card = cards[tier, slot]
                raw_costs = card[env.color_indices]
                expected_deficit = np.maximum(0, raw_costs - discounts - tokens[:len(env.colors)])
                assert np.all(deficit[player, tier, slot] == expected_deficit)
                assert gold[player, 0, 0] == tokens[env.gold_index]

def test_purchasability_threat_single_player(env: SplendorEnv):
    player = 0
    tokens = env.tokens_in_hand[player]
    discounts = env.discounts[player]
    cards = env.dealt
    
    deficit_per_color, gold_tokens = env.get_relative_costs(tokens, discounts, cards)
    threat_map = env.get_purchasability_threat(tokens, cards, deficit_per_color, gold_tokens)
    
    assert threat_map.shape == (env.num_tiers, env.num_slots)
    
    bank_regular = env.tokens_remaining[:len(env.colors)]
    bank_gold = env.tokens_remaining[env.gold_index]
    
    for tier in range(env.num_tiers):
        for slot in range(env.num_slots):
            card = cards[tier, slot]
            available = card[env.card_column_indexer['available']] == 1
            if not available:
                assert threat_map[tier, slot] == False
                continue
                
            deficit = deficit_per_color[tier, slot]
            gold = gold_tokens
            total_deficit = np.sum(deficit)
            net_deficit = total_deficit - gold
            
            needed_and_available = (deficit > 0) & (bank_regular > 0)
            max_gain_from_diff = min(3, np.sum(needed_and_available))
            
            useful_double_gain = [min(2, deficit[i]) if bank_regular[i] >= 4 else 0 for i in range(len(env.colors))]
            max_gain_from_double = max(useful_double_gain) if useful_double_gain else 0
            
            max_gain_from_reserve = 1 if bank_gold > 0 else 0
            
            is_threat = (net_deficit <= 0) or (net_deficit <= max_gain_from_diff) or (net_deficit <= max_gain_from_double) or (net_deficit <= max_gain_from_reserve)
            assert threat_map[tier, slot] == is_threat

def test_purchasability_threat_multiple_players(env: SplendorEnv):
    cards = env.dealt
    deficit_per_color, gold_tokens = env.get_relative_costs(env.tokens_in_hand, env.discounts, cards)
    threat_map = env.get_purchasability_threat(env.tokens_in_hand, cards, deficit_per_color, gold_tokens)
    
    assert threat_map.shape == (env.num_players, env.num_tiers, env.num_slots)
    
    bank_regular = env.tokens_remaining[:len(env.colors)]
    bank_gold = env.tokens_remaining[env.gold_index]
    
    for player in range(env.num_players):
        for tier in range(env.num_tiers):
            for slot in range(env.num_slots):
                card = cards[tier, slot]
                available = card[env.card_column_indexer['available']] == 1
                if not available:
                    assert threat_map[player, tier, slot] == False
                    continue
                    
                deficit = deficit_per_color[player, tier, slot]
                gold = gold_tokens[player, 0, 0]
                total_deficit = np.sum(deficit)
                net_deficit = total_deficit - gold
                
                needed_and_available = (deficit > 0) & (bank_regular > 0)
                max_gain_from_diff = min(3, np.sum(needed_and_available))
                
                useful_double_gain = [min(2, deficit[i]) if bank_regular[i] >= 4 else 0 for i in range(len(env.colors))]
                max_gain_from_double = max(useful_double_gain) if useful_double_gain else 0
                
                max_gain_from_reserve = 1 if bank_gold > 0 else 0
                
                is_threat = (net_deficit <= 0) or (net_deficit <= max_gain_from_diff) or (net_deficit <= max_gain_from_double) or (net_deficit <= max_gain_from_reserve)
                assert threat_map[player, tier, slot] == is_threat


if __name__ == "__main__":
    env = SplendorEnv(num_players=4)
    env.reset(seed=1293987)

    # Fuzz tests
    print("Testing vectorized purchasability maps...")
    for test in tqdm(range(10_000)):
        env.reset()
        env.tokens_in_hand = initialize_tokens_in_hand(env)
        env.discounts = initialize_discounts(env)
        
        # randomly set some cards to be unavailable - with 10% chance
        num_cards_dealt = env.num_tiers * env.num_slots
        for tier in range(env.num_tiers):
            for slot in range(env.num_slots):
                if np.random.random() <= (1.0 / num_cards_dealt):
                    env.dealt[tier, slot, env.card_column_indexer['available']] = 0
        
        test_single_player_single_card(env)
        test_single_player_multiple_cards(env)
        test_multiple_players_multiple_cards(env)
        test_relative_costs_single_player(env)
        test_relative_costs_multiple_players(env)
        test_purchasability_threat_single_player(env)
        test_purchasability_threat_multiple_players(env)
        
    print("All vectorized purchasability tests passed successfully!")