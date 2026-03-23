import pandas as pd
import pytest
import numpy as np
import os

# Note: Adjust this import to match the actual filename of your environment
from splendor_env import SplendorEnv

@pytest.fixture
def env():
    """Fixture to create a fresh, standard 4-player environment for each test."""
    return SplendorEnv(num_players=4, render_mode="console")


class TestSplendorInitialization:

    ### -------------------------------------------------------------------
    ### NOBLE INITIALIZATION TESTS
    ### -------------------------------------------------------------------

    def test_initialize_nobles_shape_and_availability(self, env):
        """Verifies the nobles tensor is correctly shaped and flags active cards."""
        # Check shape matches max_num_nobles x number of noble columns
        assert env.nobles.shape == (env.max_num_nobles, len(env.nobles_columns))

        # In a 4 player game, 5 nobles should be available
        assert env.num_nobles_available == 5

        # Check that exactly 'num_nobles_available' nobles have their 'available' flag set to 1
        available_idx = env.nobles_column_indexer['available']
        assert np.all(env.nobles[:env.num_nobles_available, available_idx] == 1)

        # Check that the remaining empty noble slots are zeroed out (inactive)
        if env.num_nobles_available < env.max_num_nobles:
            assert np.all(env.nobles[env.num_nobles_available:, :] == 0)

    @pytest.mark.parametrize("num_players, expected_nobles", [
        (2, 3), 
        (3, 4), 
        (4, 5)
    ])
    def test_initialize_nobles_different_player_counts(self, num_players, expected_nobles):
        """Ensures the correct number of nobles are dealt based on variable player counts."""
        temp_env = SplendorEnv(num_players=num_players)
        
        assert temp_env.num_nobles_available == expected_nobles
        
        available_idx = temp_env.nobles_column_indexer['available']
        assert np.all(temp_env.nobles[:expected_nobles, available_idx] == 1)
        
        # Verify remainder padding is zeroed
        if expected_nobles < temp_env.max_num_nobles:
            assert np.all(temp_env.nobles[expected_nobles:, :] == 0)

    def test_initialize_nobles_randomization(self, env):
        """Checks that nobles are pulled from the master CSV data correctly."""
        # Check that the dealt nobles match rows from the pre-loaded _all_nobles
        # We just test the active nobles (first 'num_nobles_available' rows)
        active_nobles = env.nobles[:env.num_nobles_available]
        
        for noble in active_nobles:
            # Look for a matching row in the master array
            # We slice [1:] to ignore the 'available' flag since it might differ
            matches = np.all(env._all_nobles[:, 1:] == noble[1:], axis=1)
            assert np.any(matches), f"Dealt noble {noble} does not exist in master _all_nobles array."


    ### -------------------------------------------------------------------
    ### DECK INITIALIZATION TESTS
    ### -------------------------------------------------------------------

    def test_initialize_deck_structure(self, env):
        """Verifies the base deck array is split into 3 tiers with proper numpy types."""
        assert len(env._deck) == env.num_tiers

        # Check that each tier is a 2D numpy array of type uint8
        for tier in range(env.num_tiers):
            assert isinstance(env._deck[tier], np.ndarray)
            assert env._deck[tier].dtype == np.uint8
            assert env._deck[tier].ndim == 2

        # Check dimensions: max_num_cards_at_tier x card_num_columns
        for tier in range(env.num_tiers):
            expected_shape = (env._max_num_cards_at_tier[tier], env.card_num_columns)
            assert env._deck[tier].shape == expected_shape

    def test_initialize_deck_availability_and_color_mapping(self, env):
        """Verifies string colors from the CSV were successfully mapped to uint8 indices."""
        available_idx = env.card_column_indexer['available']
        color_idx = env.card_column_indexer['color']

        for tier in range(env.num_tiers):
            # All initialized cards in the baseline deck should be marked as available (1)
            assert np.all(env._deck[tier][:, available_idx] == 1)

            # Check that colors were successfully mapped to numeric indices (0 to len(colors)-1)
            colors_in_tier = env._deck[tier][:, color_idx]
            assert np.all(colors_in_tier >= 0)
            assert np.all(colors_in_tier < len(env.colors))

    def test_deck_setup_on_reset(self, env):
        """Verifies that `reset()` correctly deals cards from `_deck` into `dealt`."""
        env.reset()

        # The playable deck tensor should match the padded dimensions
        expected_shape = (env.num_tiers, max(env._max_num_cards_at_tier), env.card_num_columns)
        assert env.deck.shape == expected_shape

        # Verify that cards were dealt out to the active 4 slots per tier
        assert env.dealt.shape == (env.num_tiers, env.num_slots, env.card_num_columns)
        assert np.all(env.num_dealt_at_tier == 4)
        
        # Verify the dealt cards are available
        available_idx = env.card_column_indexer['available']
        assert np.all(env.dealt[:, :, available_idx] == 1)
    
    ### -------------------------------------------------------------------
    ### CSV INTEGRITY & UNIQUENESS TESTS
    ### -------------------------------------------------------------------

    def test_nobles_match_csv_and_are_unique(self, env):
        """Verifies all nobles from the CSV are loaded without duplicates."""
        # Resolve path from env/tests/... to data/nobles.csv
        current_dir = os.path.dirname(os.path.abspath(__file__))
        nobles_csv_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'data', 'nobles.csv'))
        
        # Load the raw CSV independently
        df_nobles = pd.read_csv(nobles_csv_path)
        
        # Verify the total number of nobles matches the CSV
        assert len(env._all_nobles) == len(df_nobles), "Number of parsed nobles does not match nobles.csv row count."
        
        # Verify there are no duplicate nobles in the master array
        # axis=0 ensures we check for unique rows
        unique_nobles = np.unique(env._all_nobles, axis=0)
        assert len(unique_nobles) == len(env._all_nobles), "Duplicate noble entries found in initialization."

    def test_cards_match_csv_and_are_unique(self, env):
        """Verifies all cards from the CSV are loaded into tiers without duplicates."""
        # Resolve path from env/tests/... to data/cards.csv
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cards_csv_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'data', 'cards.csv'))
        
        # Load the raw CSV independently
        df_cards = pd.read_csv(cards_csv_path)
        
        # Verify the total number of cards across all tiers matches the CSV
        total_initialized_cards = sum(len(tier_deck) for tier_deck in env._deck)
        assert total_initialized_cards == len(df_cards), "Total parsed cards across all tiers does not match cards.csv row count."
        
        # Stack all tiers together to check for global duplicates
        all_cards_combined = np.vstack(env._deck)
        
        # Verify there are no duplicate cards in the entire deck
        unique_cards = np.unique(all_cards_combined, axis=0)
        assert len(unique_cards) == len(all_cards_combined), "Duplicate card entries found in the initialized deck."