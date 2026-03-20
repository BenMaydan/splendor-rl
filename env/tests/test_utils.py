import numpy as np
from tqdm import tqdm
from splendor_env import SplendorEnv
from tests.utils import initialize_tokens_in_bank, initialize_tokens_in_hand, initialize_discounts


if __name__ == "__main__":
    env = SplendorEnv(num_players=4)
    env.reset(seed=1293987)

    print("Testing initialize_tokens_in_hand...")
    for test in tqdm(range(100_000)):
        tokens_in_hand = initialize_tokens_in_hand(env)
        assert tokens_in_hand.shape == (env.num_players, 1 + len(env.colors))
        assert np.all(np.sum(tokens_in_hand, axis=-1) <= env.max_tokens_allowed)
        assert np.all(np.sum(tokens_in_hand, axis=0) <= initialize_tokens_in_bank(env))
    print("Success!\n")

    print("Testing initialize_discounts...")
    max_num_cards = np.sum(env._max_num_cards_at_tier)
    max_cards_per_color = max_num_cards // len(env.colors)
    for test in tqdm(range(100_000)):
        discounts = initialize_discounts(env)
        assert discounts.shape == (env.num_players, len(env.colors))
        assert np.all(np.sum(discounts) <= max_num_cards)
        assert np.all(np.sum(discounts, axis=0) <= max_cards_per_color)
    print("Success!\n")