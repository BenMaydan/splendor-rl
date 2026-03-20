import numpy as np
from numpy.typing import NDArray
from splendor_env import SplendorEnv


def initialize_tokens_in_bank(env: SplendorEnv) -> NDArray[np.int8]:
    tokens_in_bank = np.zeros((1 + len(env.colors),), dtype=np.int8)
    if env.num_players == 4:
        tokens_in_bank += 7
    elif env.num_players == 3:
        tokens_in_bank += 5
    elif env.num_players == 2:
        tokens_in_bank += 4
    tokens_in_bank[env.gold_index] = 5
    return tokens_in_bank


def initialize_tokens_in_hand(env: SplendorEnv):
    # 1. Get the starting bank state
    tokens_in_bank = initialize_tokens_in_bank(env)
    
    # 2. Create a "pool" representing every individual token in the bank
    # e.g., if bank is [2, 1], pool is [0, 0, 1]
    pool = np.repeat(np.arange(len(tokens_in_bank)), tokens_in_bank)
    
    # 3. Shuffle the entire bank once
    np.random.shuffle(pool)
    
    # 4. Determine how many tokens each player takes
    # This assumes every player fills their hand to max_tokens_allowed
    needed = env.num_players * env.max_tokens_allowed
    
    # Safety check: don't try to take more than exists in the bank
    take_amount = min(needed, len(pool))
    drawn_tokens = pool[:take_amount]
    
    # 5. Distribute into the (num_players, num_colors) shape
    tokens_in_hand = np.zeros((env.num_players, len(tokens_in_bank)), dtype=np.int8)
    
    for p in range(env.num_players):
        start = p * env.max_tokens_allowed
        end = start + env.max_tokens_allowed
        if start >= len(drawn_tokens):
            break
        
        # Count occurrences of each token type for this player's slice
        player_slice = drawn_tokens[start:end]
        counts = np.bincount(player_slice, minlength=len(tokens_in_bank))
        tokens_in_hand[p] = counts
        
    return tokens_in_hand


def initialize_discounts(env: SplendorEnv):
    total_cards = np.sum(env._max_num_cards_at_tier)
    max_per_person = total_cards // env.num_players
    cards = np.zeros((len(env.colors,)), dtype=np.uint8)
    cards += total_cards // len(env.colors)

    pool = np.repeat(np.arange(len(cards)), cards)
    start = 0

    # Distribute into the (num_players, num_colors) shape
    tokens_in_hand = np.zeros((env.num_players, len(env.colors)), dtype=np.int8)

    for p in range(env.num_players):
        num_cards_to_take = np.random.randint(0, max_per_person + 1)
        player_slice = pool[start:start + num_cards_to_take]
        counts = np.bincount(player_slice, minlength=len(env.colors))
        tokens_in_hand[p] = counts
        start += num_cards_to_take
    
    return tokens_in_hand
