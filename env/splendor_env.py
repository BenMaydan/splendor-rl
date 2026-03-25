from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
import math
import random
import pandas as pd
import itertools
import os
import functools


def read_nobles_csv(num_nobles_columns, nobles_column_indexer, nobles_color_indices, colors):
    """
    Writes all the possible noble cards to the constant deck variable for initialization
    """
    basedir = os.path.dirname(os.path.abspath(__file__))
    nobles_csv_path = os.path.abspath(os.path.join(basedir, '..', 'data', 'nobles.csv'))
    df = pd.read_csv(os.path.join(basedir, nobles_csv_path), dtype=np.uint8)

    # (available, *color_requirements)
    all_nobles = np.zeros((len(df), num_nobles_columns), dtype=np.uint8)
    all_nobles[..., nobles_column_indexer['available']] = 1
    all_nobles[..., nobles_color_indices] = df[colors].to_numpy(dtype=np.uint8)

    return all_nobles


class SplendorEnv(AECEnv):
    """
    Custom Environment that follows gymnasium interface.
    This is where the rules of Splendor will live.
    """
    metadata = {
        "render_modes": ["console", "human"], 
        "name": "splendor_v0", 
        "is_parallelizable": False,
        "render_fps": 30
    }

    def __init__(self, num_players=4, maximum_total_turns=400, render_mode="console"):
        super(SplendorEnv, self).__init__()
        self.render_mode = render_mode
        self.max_num_players = 4
        self.num_players = num_players

        # PettingZoo Agent Initialization
        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        self.agents = self.possible_agents[:]

        self.num_tiers = 3
        self.num_slots = 4
        self.colors = ['Red', 'Green', 'Blue', 'White', 'Black']
        self.gold_index = len(self.colors)
        self.max_num_of_color = None

        # types of actions
        self._action_take_3_diff_tokens = math.comb(len(self.colors), 3)
        self._action_take_2_diff_tokens = math.comb(len(self.colors), 2)
        self._action_take_1_token = math.comb(len(self.colors), 1)
        self._action_take_2_identical = len(self.colors)
        self._action_reserve_face_up = self.num_tiers * 4
        self._action_reserve_face_down = self.num_tiers
        self._action_buy_face_up = self.num_tiers * 4
        self._action_buy_reserved = 3
        self._action_pick_noble = self.max_num_players + 1
        self._action_discard = 1 + len(self.colors) # if you have too many tokens - includes ability to discard a gold even though it's objectively never the right move
        self._action_pass = 1
        
        self.num_total_actions = (
            self._action_take_3_diff_tokens + 
            self._action_take_2_diff_tokens + 
            self._action_take_1_token + 
            self._action_take_2_identical + 
            self._action_reserve_face_up + 
            self._action_reserve_face_down + 
            self._action_buy_face_up + 
            self._action_buy_reserved + 
            self._action_pick_noble + 
            self._action_discard + 
            self._action_pass
        )

        # here we precompute some things to make it easier to create the action mask very quickly
        all_combs_take_three_tokens = list(itertools.combinations(np.arange(len(self.colors)), 3))
        self._precomputed_combs_take_three_tokens = np.zeros((len(all_combs_take_three_tokens), 3), dtype=np.uint8)
        for i, comb in enumerate(all_combs_take_three_tokens):
            self._precomputed_combs_take_three_tokens[i] = comb
        
        all_combs_take_two_tokens = list(itertools.combinations(np.arange(len(self.colors)), 2))
        self._precomputed_combs_take_two_tokens = np.zeros((len(all_combs_take_two_tokens), 2), dtype=np.uint8)
        for i, comb in enumerate(all_combs_take_two_tokens):
            self._precomputed_combs_take_two_tokens[i] = comb
        
        all_combs_take_one_token = list(itertools.combinations(np.arange(len(self.colors)), 1))
        self._precomputed_combs_take_one_token = np.zeros((len(all_combs_take_one_token), 1), dtype=np.uint8)
        for i, comb in enumerate(all_combs_take_one_token):
            self._precomputed_combs_take_one_token[i] = comb

        # we want to precompute starting indices of the action mask for a given action type (for quick masking)
        # the ending index of the action type is the action_indices[action_type_index + 1]
        self._action_indices = np.zeros((12,), dtype=np.uint8)
        self._action_indices[1] = self._action_take_3_diff_tokens
        self._action_indices[2] = self._action_indices[1] + self._action_take_2_diff_tokens
        self._action_indices[3] = self._action_indices[2] + self._action_take_1_token
        self._action_indices[4] = self._action_indices[3] + self._action_take_2_identical
        self._action_indices[5] = self._action_indices[4] + self._action_reserve_face_up
        self._action_indices[6] = self._action_indices[5] + self._action_reserve_face_down
        self._action_indices[7] = self._action_indices[6] + self._action_buy_face_up
        self._action_indices[8] = self._action_indices[7] + self._action_buy_reserved
        self._action_indices[9] = self._action_indices[8] + self._action_pick_noble
        self._action_indices[10] = self._action_indices[9] + self._action_discard
        self._action_indices[11] = self._action_indices[10] + self._action_pass
        self._action_indices_map = {
            "take_3_diff_tokens": [self._action_indices[0], self._action_indices[1]],
            "take_2_diff_tokens": [self._action_indices[1], self._action_indices[2]],
            "take_1_token": [self._action_indices[2], self._action_indices[3]],
            "take_2_tokens": [self._action_indices[3], self._action_indices[4]],
            "reserve_face_up": [self._action_indices[4], self._action_indices[5]],
            "reserve_face_down": [self._action_indices[5], self._action_indices[6]],
            "buy_face_up": [self._action_indices[6], self._action_indices[7]],
            "buy_reserved": [self._action_indices[7], self._action_indices[8]],
            "pick_noble": [self._action_indices[8], self._action_indices[9]],
            "discard": [self._action_indices[9], self._action_indices[10]],
            "pass": [self._action_indices[10], self._action_indices[11]],
        }

        # Initializing internal game state variables
        self.get_next_player = lambda: (self.current_player + 1) % self.num_players

        # data about the card, we store it here to avoid hardcoding
        card_columns = ['available', 'points', 'color'] + self.colors
        self.card_column_indexer = {column: i for i, column in enumerate(card_columns)}
        self.color_indices = [self.card_column_indexer[c] for c in self.colors]
        self.card_num_columns = len(card_columns)

        self._deck = None
        self.deck = None
        self.dealt = None
        self._max_num_cards_at_tier = None
        self.num_dealt_at_tier = None

        # the purchasability map is (num_players, num_tiers, num_slots)
        purchasability_shape = (self.max_num_players, self.num_tiers, self.num_slots)

        # setting observation limits about the dealt tensor
        dealt_shape = (self.num_tiers, self.num_slots, self.card_num_columns)
        self.dealt_observation_limits_high = np.zeros(dealt_shape, dtype=np.uint8)
        self.dealt_observation_limits_high[:, :, self.card_column_indexer['available']] = 1
        self.dealt_observation_limits_high[:, :, self.card_column_indexer['points']] = 5
        self.dealt_observation_limits_high[:, :, self.card_column_indexer['color']] = len(self.colors)
        self.dealt_observation_limits_high[:, :, self.color_indices] = 7

        self.max_able_to_reserve = 3
        self.max_tokens_allowed = 10
        self.points = np.zeros((self.max_num_players,), dtype=np.int8)
        self.reserved = np.zeros((self.max_num_players, self.max_able_to_reserve, self.card_num_columns), dtype=np.uint8)
        self.num_reserved = np.zeros((self.max_num_players,), dtype=np.uint8)
        self.num_cards_in_hand = np.zeros((self.max_num_players,), dtype=np.uint8)
        self.discounts = np.zeros((self.max_num_players, len(self.colors)), dtype=np.int8)
        self.tokens_remaining = np.zeros((1 + len(self.colors),), dtype=np.uint8)
        self.tokens_in_hand = np.zeros((self.max_num_players, 1 + len(self.colors)), dtype=np.int8)

        self.num_nobles_points = 3
        self.nobles_columns = ['available'] + self.colors
        self.nobles_column_indexer = {column: i for i, column in enumerate(self.nobles_columns)}
        self.nobles_color_indices = [self.nobles_column_indexer[c] for c in self.colors]
        self._all_nobles = read_nobles_csv(
            len(self.nobles_columns),
            self.nobles_column_indexer,
            self.nobles_color_indices,
            self.colors
        )
        self.nobles = None
        self.num_nobles_available = num_players + 1
        self.max_num_nobles = self.max_num_players + 1
        self.initialize_nobles()

        # initialize nobles observation limits only once
        self.nobles_observation_limits_low = np.zeros(self.nobles.shape, dtype=np.uint8)
        self.nobles_observation_limits_high = np.zeros(self.nobles.shape, dtype=np.uint8)
        self.nobles_observation_limits_low[:] = np.min(self._all_nobles, axis=0)
        self.nobles_observation_limits_high[:] = np.max(self._all_nobles, axis=0)

        self.phases = ['main', 'pick_noble', 'discard']
        self.current_phase = 'main'
        self.starting_player = 0
        self.current_player = 0
        self.num_turns = 0
        self.termination_condition = lambda: (self.num_turns % self.num_players == 0 and np.any(self.points >= 15))
        self.truncation_condition = lambda: (self.num_turns >= maximum_total_turns)

        self.mini_rewards = {
            'buy_card': 0.5,
            'get_noble': 3.0,
            'get_point': 1.0,
        }
        self.lose_points = 100
        self.win_points = (self.num_players - 1) * self.lose_points
        self.discourage_stalling = -0.03
        self.deadlock_tax = -50

        self.initialize_deck()
        self.initialize_misc()

        # initializing observation space
        # TODO: make all observations about costs be relative to the player
        # so dealt for example should be the cost for the player who can buy it
        self.single_observation_space = spaces.Dict({
            "observation": spaces.Dict({
                "phase": spaces.Discrete(len(self.phases)),
                "relative_player_seat": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
                "tier_1_remaining": spaces.Box(low=0, high=self._max_num_cards_at_tier[0], shape=(1,), dtype=np.uint8),
                "tier_2_remaining": spaces.Box(low=0, high=self._max_num_cards_at_tier[1], shape=(1,), dtype=np.uint8),
                "tier_3_remaining": spaces.Box(low=0, high=self._max_num_cards_at_tier[2], shape=(1,), dtype=np.uint8),
                "nobles_remaining": spaces.Box(low=0, high=5, shape=(1,), dtype=np.uint8),
                "tokens_remaining": spaces.Box(low=0, high=7, shape=self.tokens_remaining.shape, dtype=np.uint8),
                "purchasability": spaces.MultiBinary(purchasability_shape),
                "dealt": spaces.Box(low=0, high=self.dealt_observation_limits_high, shape=dealt_shape, dtype=np.uint8),
                "nobles": spaces.Box(low=self.nobles_observation_limits_low, high=self.nobles_observation_limits_high, shape=self.nobles.shape, dtype=np.uint8),
                "points": spaces.Box(low=0, high=22, shape=self.points.shape, dtype=np.int8),
                "reserved": spaces.Box(low=0, high=7, shape=self.reserved.shape, dtype=np.uint8),
                "discounts": spaces.Box(low=0, high=self.max_num_of_color, shape=self.discounts.shape, dtype=np.int8),
                "num_cards_in_hand": spaces.Box(low=0, high=30, shape=self.num_cards_in_hand.shape, dtype=np.uint8),
                "tokens_in_hand": spaces.Box(low=0, high=7, shape=self.tokens_in_hand.shape, dtype=np.int8)
            }),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.num_total_actions,), dtype=np.uint8)
        })

        # initializing action mapping and mask
        self.action_mapping = {}
        self.action_mask = np.ones((self.num_total_actions,), dtype=np.uint8)
        self._build_action_space()

        # for global deadlocks - an early truncation mechanism if all players pass in a row
        self.num_passes_in_a_row = None
    
    def initialize_nobles(self):
        """
        Writes all the possible nobles to the deck to initialize, to start the game
        """
        # (available, *color_requirements)
        self.nobles = np.zeros((self.max_num_nobles, len(self.nobles_columns)), dtype=np.uint8)

        # Select random indices without replacement.
        chosen_indices = np.random.choice(
            self._all_nobles.shape[0],
            size=self.num_nobles_available,
            replace=False
        )

        # Assign the randomly selected nobles to the active slots
        self.nobles[:self.num_nobles_available, :] = self._all_nobles[chosen_indices]

        # Assert only on the nobles that were actually dealt into play
        assert np.all(self.nobles[:self.num_nobles_available, self.nobles_column_indexer['available']] == 1)
    
    def initialize_deck(self):
        """
        Writes all the possible cards to the deck to initialize, to start the game
        """
        # Read without forcing uint8 on string columns - then shuffle
        basedir = os.path.dirname(os.path.abspath(__file__))
        cards_csv_path = os.path.abspath(os.path.join(basedir, '..', 'data', 'cards.csv'))
        df = pd.read_csv(os.path.join(basedir, cards_csv_path))

        # Handle color mapping BEFORE casting to uint8
        self.max_num_of_color = np.count_nonzero(df['color'] == self.colors[0])
        color_indices = {color: i for i, color in enumerate(self.colors)}
        df['color'] = df['color'].map(color_indices)

        # Now that 'color' is numeric, cast the whole dataframe to uint8
        # Since everything is now an integer it should not throw an error
        df = df.astype(np.uint8)

        self._max_num_cards_at_tier = np.zeros((self.num_tiers,), dtype=np.uint8)
        for tier in range(self.num_tiers):
            df_at_tier = df[df['level'] == (tier + 1)]
            self._max_num_cards_at_tier[tier] = len(df_at_tier)

        # deck should be a list of arrays per tier
        self._deck = [np.zeros((self._max_num_cards_at_tier[tier], self.card_num_columns), dtype=np.uint8) for tier in range(self.num_tiers)]

        # fill in our numpy deck array with the csv data
        for tier in range(self.num_tiers):
            df_at_tier = df[df['level'] == (tier + 1)]

            # available is hardcoded to 1 for now until cards start to run out
            self._deck[tier][:, self.card_column_indexer['available']] = 1

            # now we fill in our cards numpy array using the data in the csv
            for column, column_index in self.card_column_indexer.items():
                if column == 'available':
                    continue
                self._deck[tier][:, column_index] = df_at_tier[column].values

    def initialize_misc(self):
        """
        Initialize observational data about purchased self + opponents card
        """
        # TODO: fix this so we don't allocate new arrays every time reset is called.
        # need to store num_players x (discount_red, discount_blue, ..., discount_gold)
        self.points[:] = 0
        self.reserved[:] = 0
        self.num_reserved[:] = 0
        self.num_cards_in_hand[:] = 0
        self.discounts[:] = 0
        self.tokens_remaining[:] = 0
        self.tokens_in_hand[:] = 0

        # actually initialize tokens remaining tensor
        if self.num_players == 4:
            self.tokens_remaining += 7
        elif self.num_players == 3:
            self.tokens_remaining += 5
        elif self.num_players == 2:
            self.tokens_remaining += 4
        self.tokens_remaining[self.gold_index] = 5

        self.num_passes_in_a_row = 0

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Resolve player count FIRST
        if options and 'num_players' in options:
            self.num_players = options['num_players']
        
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Reset the internal game state to the start of a Splendor game
        self.current_phase = 'main'
        
        # initialize starting player index randomly
        self.starting_player = random.randint(0, self.num_players - 1)
        self.current_player = self.starting_player
        self.num_turns = 0

        self.initialize_nobles()
        self.initialize_misc()
        # deck initialization
        self.deck = np.zeros((self.num_tiers, max(self._max_num_cards_at_tier), self.card_num_columns), dtype=np.uint8)
        for tier in range(self.num_tiers):
            deck_at_tier = np.copy(self._deck[tier])
            assert deck_at_tier.ndim == 2
            np.random.shuffle(deck_at_tier)
            self.deck[tier, :self._max_num_cards_at_tier[tier], :] = deck_at_tier
        self.dealt = self.deck[:, :self.num_slots, :]
        self.num_dealt_at_tier = np.zeros((self.num_tiers,), dtype=np.uint8) + 4

        # re-initialize action mask
        self.action_mask = np.ones((self.num_total_actions,), dtype=np.uint8)
        self._generate_action_mask()
        
        # Initialize PettingZoo's agent selector
        # You can construct this list based on your dynamic turn order
        # The important part is we can override not calling agent selector if necessary
        turn_order = [f"player_{(self.starting_player + i) % self.num_players}" for i in range(self.num_players)]
        self._agent_selector = agent_selector(turn_order)
        self.agent_selection = self._agent_selector.next()

    def _build_action_space(self):
        """
        Populates self.action_mapping with human-readable dictionaries.
        """
        action_idx = 0

        # Action 1: taking 3 diff tokens
        for comb in itertools.combinations(np.arange(len(self.colors)), 3):
            desc = ", ".join([f"1 {self.colors[comb[i]]}" for i in range(len(comb))])
            self.action_mapping[action_idx] = {
                "type": "take_3_diff_tokens",
                "indices": list(comb),
                "desc": f"Take Gems ({desc})"
            }
            action_idx += 1
        
        # Action 2: taking 2 diff tokens
        for comb in itertools.combinations(np.arange(len(self.colors)), 2):
            desc = ", ".join([f"1 {self.colors[comb[i]]}" for i in range(len(comb))])
            self.action_mapping[action_idx] = {
                "type": "take_2_diff_tokens",
                "indices": list(comb),
                "desc": f"Take Gems ({desc})"
            }
            action_idx += 1
        
        # Action 3: taking 1 token
        for comb in itertools.combinations(np.arange(len(self.colors)), 1):
            desc = ", ".join([f"1 {self.colors[comb[i]]}" for i in range(len(comb))])
            self.action_mapping[action_idx] = {
                "type": "take_1_token",
                "indices": list(comb),
                "desc": f"Take Gems ({desc})"
            }
            action_idx += 1
        
        # Action 4: taking 2 identical tokens
        for color_index in range(len(self.colors)):
            self.action_mapping[action_idx] = {
                "type": "take_2_identical_tokens",
                "index": color_index,
                "desc": f"Take 2 {self.colors[color_index]} Gems"
            }
            action_idx += 1
        
        # # Action 5: Reserving a face up card
        # 3 tiers, 4 cards each = 12 actions
        for tier in range(self.num_tiers):
            for slot in range(4):
                self.action_mapping[action_idx] = {
                    "type": "reserve_face_up",
                    "tier": tier,
                    "slot": slot,
                    "desc": f"Reserve Card (Tier {tier}, Slot {slot})"
                }
                action_idx += 1
        
        # # Action 6: Reserving a face down card
        # 3 tiers = 3 actions
        for tier in range(self.num_tiers):
            self.action_mapping[action_idx] = {
                "type": "reserve_face_down",
                "tier": tier,
                "desc": f"Reserve Face Down Card in Tier {tier}"
            }
            action_idx += 1
                
        # Action 7: Buying Face-Up Cards
        for tier in range(self.num_tiers):
            for slot in range(4):
                self.action_mapping[action_idx] = {
                    "type": "buy_face_up",
                    "tier": tier,
                    "slot": slot,
                    "desc": f"Buy Card (Tier {tier}, Slot {slot})"
                }
                action_idx += 1
        
        # Action 8: Buying Reserved Cards
        # Can reserve at most 3 cards
        desc_number = ["1st", "2nd", "3rd", "4th", "5th"]
        for index in range(3):
            self.action_mapping[action_idx] = {
                "type": "buy_reserved",
                "index": index,
                "desc": f"Buy {desc_number[index]} Reserved Card"
            }
            action_idx += 1
        
        # Action 9: Pick a noble if you can
        for index in range(self.max_num_nobles):
            self.action_mapping[action_idx] = {
                "type": "pick_noble",
                "index": index,
                "desc": f"Pick {desc_number[index]} Noble"
            }
            action_idx += 1

        # Action 10: Discard tokens (if you have too many)
        desc_colors = self.colors.copy()
        desc_colors.insert(self.gold_index, "Gold")
        for token_type in range(1 + len(self.colors)):
            self.action_mapping[action_idx] = {
                "type": "discard_token",
                "index": token_type,
                "desc": f"Discard {desc_colors[token_type]} Gem"
            }
            action_idx += 1
        
        # Action 11: Pass if you can't do anything
        self.action_mapping[action_idx] = {
            "type": "pass",
            "desc": f"Pass (No Legal Moves)"
        }
    
    def _generate_action_mask(self) -> None:
        """
        Generate the numpy array of the action mask given all the observations we know
        """
        match self.current_phase:
            case "main":
                self.action_mask[:] = 1

                # mask out taking 2 diff and 1 token since that is a special case of there not being enough 3 diff tokens
                s, e = self._action_indices_map["take_2_diff_tokens"]
                self.action_mask[s:e] = 0
                s, e = self._action_indices_map["take_1_token"]
                self.action_mask[s:e] = 0
                # mask out picking noble and discarding since that is not allowed in the main phase
                s, e = self._action_indices_map["pick_noble"]
                self.action_mask[s:e] = 0
                s, e = self._action_indices_map["discard"]
                self.action_mask[s:e] = 0

                num_reserved = self.num_reserved[self.current_player]

                # mask out invalid actions of taking three tokens
                s, e = self._action_indices_map["take_3_diff_tokens"]
                invalid_action_indices = np.any(self.tokens_remaining[self._precomputed_combs_take_three_tokens] == 0, axis=1)
                self.action_mask[s:e][invalid_action_indices] = 0

                # means it is not possible to take three different tokens, so you take two different tokens
                if np.sum(self.action_mask[s:e]) == 0:
                    s, e = self._action_indices_map["take_2_diff_tokens"]
                    self.action_mask[s:e] = 1
                    invalid_action_indices = np.any(self.tokens_remaining[self._precomputed_combs_take_two_tokens] == 0, axis=1)
                    self.action_mask[s:e][invalid_action_indices] = 0

                    # means it is not possible to take three different tokens, so you take one token
                    if np.sum(self.action_mask[s:e]) == 0:
                        s, e = self._action_indices_map["take_1_token"]
                        self.action_mask[s:e] = 1
                        invalid_action_indices = np.any(self.tokens_remaining[self._precomputed_combs_take_one_token] == 0, axis=1)
                        self.action_mask[s:e][invalid_action_indices] = 0


                # MOVING ON ->
                # mask out invalid actions of taking two identical tokens
                s, e = self._action_indices_map["take_2_tokens"]
                invalid_action_indices = self.tokens_remaining[:len(self.colors)] < 4
                self.action_mask[s:e][invalid_action_indices] = 0

                # mask out invalid actions of reserving face up card
                # the only cards you cannot reserve are ones that don't exist because that tier was already completely dealt out
                s, e = self._action_indices_map["reserve_face_up"]
                if num_reserved >= self.max_able_to_reserve:
                    self.action_mask[s:e] = 0
                else:
                    invalid_action_indices = (self.dealt[..., self.card_column_indexer['available']] == 0).flatten()
                    self.action_mask[s:e][invalid_action_indices] = 0

                # mask out invalid actions of reserving face down card
                # only time this is invalid is if that tier has been completely dealt out or you already reserved the max
                s, e = self._action_indices_map["reserve_face_down"]
                if num_reserved >= self.max_able_to_reserve:
                    self.action_mask[s:e] = 0
                else:
                    invalid_action_indices = (self.num_dealt_at_tier >= self._max_num_cards_at_tier)
                    self.action_mask[s:e][invalid_action_indices] = 0

                # mask out invalid actions of buying face up card
                # an invalid action is: not (card available and card purchasable), which is de'morgans law
                # we can reuse logic to see which face up cards are available
                s, e = self._action_indices_map["buy_face_up"]
                invalid_action_indices = np.logical_not(
                    self.get_purchasability_map(self.tokens_in_hand[self.current_player], self.discounts[self.current_player], self.dealt)
                ).flatten()
                self.action_mask[s:e][invalid_action_indices] = 0

                # mask out invalid actions of buying reserved card
                s, e = self._action_indices_map["buy_reserved"]
                invalid_action_indices = np.logical_not(
                    self.get_purchasability_map(self.tokens_in_hand[self.current_player], self.discounts[self.current_player], self.reserved[self.current_player])
                ).flatten()
                self.action_mask[s:e][invalid_action_indices] = 0
            
            case "pick_noble":
                assert np.any(self.nobles[..., self.nobles_column_indexer['available']] == 1)
                self.action_mask[:] = 0
                # only allow actions for correct nobles you can pick
                # most of the time there will only be one allowable action
                # but rarely there will be a choice -- will be interesting to see what the agent decides
                s, e = self._action_indices_map["pick_noble"]
                valid_action_indices = np.logical_and(
                    self.nobles[..., self.nobles_column_indexer['available']] == 1,
                    np.all(self.discounts[self.current_player] >= self.nobles[..., self.nobles_color_indices], axis=-1)
                ).flatten()
                self.action_mask[s:e][valid_action_indices] = 1

            case "discard":
                assert np.sum(self.tokens_in_hand[self.current_player]) > 0
                self.action_mask[:] = 0
                # only allow actions for tokens you are able to discard
                s, e = self._action_indices_map["discard"]
                valid_action_indices = (self.tokens_in_hand[self.current_player] > 0)
                self.action_mask[s:e][valid_action_indices] = 1
        

        # ---------------------------------------------------------
        # GLOBAL FALLBACK: Evaluate only after ALL masking is finished
        # This only happens because it is possible to be in an absolute deadlock, just really unlikely
        # ---------------------------------------------------------
        # 1. Mask out the pass action by default just to be safe
        s_pass, e_pass = self._action_indices_map["pass"]
        self.action_mask[s_pass:e_pass] = 0
        
        # 2. If absolutely no actions survived the phase logic, unlock pass
        if np.sum(self.action_mask) == 0:
            self.action_mask[s_pass:e_pass] = 1


        # A guardrail to ensure there is always one valid action - this should ALWAYS pass
        try:
            assert np.sum(self.action_mask) > 0
        except AssertionError:
            print(f"DEADLOCK DETECTED!")
            print(f"Phase: {self.current_phase}")
            print(f"Player: {self.current_player}")
            print(f"Bank: {self.tokens_remaining}")
            print(f"Nobles: {self.nobles}")
            print(f"Reserved: {self.reserved}")
            print(f"Hand: {self.tokens_in_hand[self.current_player]}")
            print(f"All Tokens: {self.tokens_in_hand}, Sum: {np.sum(self.tokens_in_hand, axis=0)}")
            print(f"Dealt: {self.dealt}")
            print(f"Num Dealt: {self.num_dealt_at_tier}")
            raise

    def _token_cost(self, tokens_in_hand, discounts, card) -> None | NDArray[np.uint8]:
        """
        Determines the token cost (and if gold tokens are necessary) to buy a card
        Return inf if player doesn't have enough tokens (including gold tokens)
        """
        # tokens_available: (*colors, gold_index)
        # card: (*colors)
        raw_costs = card[..., self.color_indices]
        deficit_per_color = np.maximum(0, raw_costs - discounts - tokens_in_hand[:len(self.colors)])
        gold_needed = np.sum(deficit_per_color, axis=-1)
        if gold_needed > tokens_in_hand[self.gold_index]:
            return None
        actual_gem_cost = np.maximum(0, np.minimum(raw_costs - discounts, tokens_in_hand[:len(self.colors)]))

        result = np.empty(1 + len(self.colors), dtype=np.int8)
        result[:len(self.colors)] = actual_gem_cost
        result[self.gold_index] = gold_needed
        return result
    
    def get_purchasability_map(self, tokens_in_hand, discounts, cards) -> NDArray[np.bool_]:
        """
        Determines the token cost (and if gold tokens are necessary) to buy a card.
        Returns an unflattened mask matching the spatial dimensions of the cards.
        """
        
        NUM_COLORS = len(self.colors)

        # --- INPUT SHAPE ASSERTIONS ---
        # 1. tokens_in_hand checks
        assert tokens_in_hand.ndim in [1, 2], "tokens_in_hand must be 1D (single player) or 2D (multi-player)"
        assert tokens_in_hand.shape[-1] == NUM_COLORS + 1, f"tokens_in_hand last dim must be {NUM_COLORS + 1} (colors + gold)"
        if tokens_in_hand.ndim == 2:
            assert tokens_in_hand.shape[0] == self.max_num_players, f"tokens_in_hand dim 0 must be {self.max_num_players}"

        # 2. discounts checks
        assert discounts.ndim in [1, 2], "discounts must be 1D (single player) or 2D (multi-player)"
        assert discounts.shape[-1] == NUM_COLORS, f"discounts last dim must be {NUM_COLORS}"
        if discounts.ndim == 2:
            assert discounts.shape[0] == self.max_num_players, f"discounts dim 0 must be {self.max_num_players}"
            
        # 3. cards checks
        assert cards.ndim >= 1, "cards must have at least 1 dimension (card features)"
        assert cards.shape[-1] == self.card_num_columns, f"cards last dim must be {self.card_num_columns}"
        
        # 4. Consistency checks
        assert tokens_in_hand.ndim == discounts.ndim, "tokens_in_hand and discounts must both be either single-player or multi-player"
        # ------------------------------

        # Slice the final axis for colors/gold
        regular_tokens = tokens_in_hand[..., :NUM_COLORS]
        gold_tokens = tokens_in_hand[..., self.gold_index]
        raw_costs = cards[..., self.color_indices]

        # Align dimensions dynamically for multi-player batched inputs
        if tokens_in_hand.ndim > 1:
            # Calculate the number of spatial dimensions on the board
            # e.g., dealt cards (3, 4, 5) -> 2 spatial dims.
            num_spatial_dims = raw_costs.ndim - 1
            
            # Insert the required number of '1' dimensions right after the player dimension
            for _ in range(num_spatial_dims):
                regular_tokens = np.expand_dims(regular_tokens, axis=1)
                discounts = np.expand_dims(discounts, axis=1)
                gold_tokens = np.expand_dims(gold_tokens, axis=1)

        # Compute cost
        deficit_per_color = np.maximum(0, raw_costs - discounts - regular_tokens)
        gold_needed = np.sum(deficit_per_color, axis=-1)
        
        # 1. Determine if the player can afford the raw cost
        affordability_map = gold_tokens >= gold_needed
        
        # 2. Extract the availability mask directly from the cards tensor
        availability_mask = cards[..., self.card_column_indexer['available']] == 1
        
        # 3. Create the final map via logical AND
        purchasability_map = np.logical_and(affordability_map, availability_mask)
        
        # --- OUTPUT SHAPE ASSERTION ---
        # The output shape should perfectly match the cards array, minus the feature dimension
        expected_output_shape = cards.shape[:-1]
        if tokens_in_hand.ndim > 1:
            # If batched, prepend the num_players dimension to the expected output
            expected_output_shape = (tokens_in_hand.shape[0],) + expected_output_shape

        assert purchasability_map.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, got {purchasability_map.shape}"
        # ------------------------------
        
        return purchasability_map

    def _end_of_turn_check(self) -> tuple[str, int]:
        """
        Evaluates noble eligibility at the very end of a player's turn sequence.
        Returns (next_phase, next_player).
        """
        available_nobles_mask = self.nobles[:, self.nobles_column_indexer['available']] == 1
        
        if np.any(available_nobles_mask):
            available_nobles_cost = self.nobles[available_nobles_mask][..., self.nobles_color_indices]
            # If the player's engine affords any available noble
            if np.any(np.all(self.discounts[self.current_player] >= available_nobles_cost, axis=-1)):
                return "pick_noble", self.current_player
                
        # If no noble is triggered, officially pass the turn
        next_player = (self.current_player + 1) % self.num_players
        return "main", next_player

    def _apply_action(self, action) -> tuple[int, str, int]:
        """
        Applies the action to the current environment
        Returns:
            (mini_reward, next phase, next player index)
            Sometimes next player index does not change if there is a phase change
        """
        # print(f"Action: {action}, player: {current_player}, phase: {self.current_phase}")
        action_type = action["type"]
        next_player = (self.current_player + 1) % self.num_players

        # Update the pass counter based on the action taken
        if action_type == "pass":
            self.num_passes_in_a_row += 1
        else:
            self.num_passes_in_a_row = 0

        def buy_card(card) -> None:
            # can't be none since we know the action is valid
            cost = self._token_cost(self.tokens_in_hand[self.current_player], self.discounts[self.current_player], card)
            assert cost is not None and isinstance(cost, np.ndarray), f"Cost = {cost}"
            self.tokens_in_hand[self.current_player] -= cost
            self.tokens_remaining += cost.astype(self.tokens_remaining.dtype)
            self.points[self.current_player] += card[self.card_column_indexer['points']]
            self.num_cards_in_hand[self.current_player] += 1
            self.discounts[self.current_player][card[self.card_column_indexer['color']]] += 1
        
        def deal_new_card(tier, slot=None):
            # deal new card to dealt assuming there are cards left
            if slot is not None:
                if self.num_dealt_at_tier[tier] < self._max_num_cards_at_tier[tier]:
                    self.dealt[tier, slot] = self.deck[tier, self.num_dealt_at_tier[tier]]
                else:
                    self.dealt[tier, slot, self.card_column_indexer['available']] = 0
                    return
            
            self.num_dealt_at_tier[tier] += 1

        match action_type:
            case "take_3_diff_tokens" | "take_2_diff_tokens" | "take_1_token":
                token_indices = action["indices"]
                self.tokens_in_hand[self.current_player, token_indices] += 1
                self.tokens_remaining[token_indices] -= 1
                if np.sum(self.tokens_in_hand[self.current_player]) > self.max_tokens_allowed:
                    return (0, "discard", self.current_player)

                next_phase, next_p = self._end_of_turn_check()
                return (0, next_phase, next_p)

            case "take_2_identical_tokens":
                token_index = action["index"]
                self.tokens_in_hand[self.current_player, token_index] += 2
                self.tokens_remaining[token_index] -= 2
                if np.sum(self.tokens_in_hand[self.current_player]) > self.max_tokens_allowed:
                    return (0, "discard", self.current_player)

                next_phase, next_p = self._end_of_turn_check()
                return (0, next_phase, next_p)

            case "reserve_face_up":
                tier, slot = action["tier"], action["slot"]
                # put reserved card in players reserved pile
                self.reserved[self.current_player, self.num_reserved[self.current_player]] = self.dealt[tier, slot]
                self.num_reserved[self.current_player] += 1

                # take gold token for this player if there are tokens left
                if self.tokens_remaining[self.gold_index] > 0:
                    self.tokens_in_hand[self.current_player, self.gold_index] += 1
                    self.tokens_remaining[self.gold_index] -= 1
                
                deal_new_card(tier, slot)
                
                # make sure to move to discard if getting gold token brings you over the limit of allowed tokens
                if np.sum(self.tokens_in_hand[self.current_player]) > self.max_tokens_allowed:
                    return (0, "discard", self.current_player)
                
                next_phase, next_p = self._end_of_turn_check()
                return (0, next_phase, next_p)

            case "reserve_face_down":
                tier = action["tier"]
                # put reserved card in players reserved pile
                self.reserved[self.current_player, self.num_reserved[self.current_player]] = self.deck[tier, self.num_dealt_at_tier[tier]]
                self.num_reserved[self.current_player] += 1

                deal_new_card(tier)

                # take gold token for this player if there are tokens left
                if self.tokens_remaining[self.gold_index] > 0:
                    self.tokens_in_hand[self.current_player, self.gold_index] += 1
                    self.tokens_remaining[self.gold_index] -= 1
                
                # make sure to move to discard if getting gold token brings you over the limit of allowed tokens
                if np.sum(self.tokens_in_hand[self.current_player]) > self.max_tokens_allowed:
                    return (0, "discard", self.current_player)
                
                next_phase, next_p = self._end_of_turn_check()
                return (0, next_phase, next_p)

            case "buy_face_up":
                tier, slot = action["tier"], action["slot"]
                card = self.dealt[tier, slot]
                points_in_card = card[self.card_column_indexer['points']]

                buy_card(card)
                deal_new_card(tier, slot)

                next_phase, next_p = self._end_of_turn_check()
                return (self.mini_rewards['buy_card'] + points_in_card * self.mini_rewards['get_point'], next_phase, next_p)

            case "buy_reserved":
                index = action["index"]
                card = self.reserved[self.current_player, index]
                points_in_card = card[self.card_column_indexer['points']]

                buy_card(card)

                self.reserved[self.current_player, index, self.card_column_indexer['available']] = 0
                self.num_reserved[self.current_player] -= 1
                # we shift over the remaining reserved cards
                new_available_reserved_cards = self.reserved[self.current_player, :, self.card_column_indexer['available']] == 1
                self.reserved[self.current_player, :self.num_reserved[self.current_player]] = self.reserved[self.current_player, new_available_reserved_cards]
                self.reserved[self.current_player, self.num_reserved[self.current_player]:, :] = 0

                next_phase, next_p = self._end_of_turn_check()
                return (self.mini_rewards['buy_card'] + points_in_card * self.mini_rewards['get_point'], next_phase, next_p)

            case "pick_noble":
                assert self.current_phase == "pick_noble", f"Pick noble action taken in {self.current_phase} phase!"
                index = action["index"]
                self.points[self.current_player] += 3
                self.nobles[index, self.nobles_column_indexer['available']] = 0

                # Rule enforcement: Turn ends immediately after picking a noble.
                return (self.mini_rewards['get_noble'], "main", next_player)

            case "discard_token":
                assert self.current_phase == "discard", f"Discard action taken in {self.current_phase} phase!"
                index = action["index"]
                self.tokens_remaining[index] += 1
                self.tokens_in_hand[self.current_player][index] -= 1

                if np.sum(self.tokens_in_hand[self.current_player]) > self.max_tokens_allowed:
                    return (0, "discard", self.current_player)

                # Discarding finishes the turn sequence, so check for nobles now
                next_phase, next_p = self._end_of_turn_check()
                return (0, next_phase, next_p)

            case "pass":
                return (0, "main", next_player)

            case _:
                raise gym.error.InvalidAction(f"Action {action} is ill-defined!")

    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.single_observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(self.num_total_actions)

    def observe(self, agent):
        """
        Returns the observation for a specific agent.
        """
        # Extract the integer index from the agent string (e.g., "player_0" -> 0)
        player_idx = int(agent.split('_')[1])
        
        # Calculate the shift required to move this player's data to index 0
        shift = -player_idx
        
        # Roll all player-specific arrays along the player dimension (axis 0)
        ego_points = np.roll(self.points, shift, axis=0)
        ego_reserved = np.roll(self.reserved, shift, axis=0)
        ego_discounts = np.roll(self.discounts, shift, axis=0)
        ego_num_cards_in_hand = np.roll(self.num_cards_in_hand, shift, axis=0)
        ego_tokens_in_hand = np.roll(self.tokens_in_hand, shift, axis=0)

        # Compute purchasability using the ego-centric arrays
        # The agent now has a boolean map of exactly what IT can afford at index 0
        ego_purchasability = self.get_purchasability_map(ego_tokens_in_hand, ego_discounts, self.dealt)
        
        # Generate the new observation state
        # Notice we use player_idx to structure the perspective
        observation = {
            "observation": {
                "phase": self.phases.index(self.current_phase),
                "relative_player_seat": np.array([(player_idx + self.num_players - self.starting_player) % self.num_players], dtype=np.uint8),

                "tier_1_remaining": np.array([self._max_num_cards_at_tier[0] - self.num_dealt_at_tier[0]], dtype=np.uint8),
                "tier_2_remaining": np.array([self._max_num_cards_at_tier[1] - self.num_dealt_at_tier[1]], dtype=np.uint8),
                "tier_3_remaining": np.array([self._max_num_cards_at_tier[2] - self.num_dealt_at_tier[2]], dtype=np.uint8),
                "nobles_remaining": np.array([self.num_nobles_available], dtype=np.uint8),
                "tokens_remaining": self.tokens_remaining,

                "purchasability": ego_purchasability,
                
                "dealt": self.dealt,
                "nobles": self.nobles,
                
                "points": ego_points,
                "reserved": ego_reserved,
                "discounts": ego_discounts,
                "num_cards_in_hand": ego_num_cards_in_hand,
                "tokens_in_hand": ego_tokens_in_hand
            },
            
            # The mask must reflect the current state.
            # If it's not this agent's turn, the mask should ideally be all zeros.
            "action_mask": self.action_mask if agent == self.agent_selection else np.zeros((self.num_total_actions,), dtype=np.uint8)
        }

        return observation

    def step(self, action):
        """
        Executes one time step within the environment based on the given action.
        """
        # Handle Dead Steps Manually to avoid _agent_selector desync
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            agent = self.agent_selection
            # _was_dead_step clears rewards and sets the agent as done
            self._was_dead_step(action)
            
            # Manually advance to the next agent instead of relying on the broken _agent_selector
            current_idx = self.possible_agents.index(agent)
            next_idx = (current_idx + 1) % self.num_players
            self.agent_selection = self.possible_agents[next_idx]
            return
        
        # A guardrail to make sure AgileRL respects the action mask
        if self.action_mask[action] == 0:
            raise ValueError(
                f"Agent {self.agent_selection} selected invalid action to {self.action_mapping[action]['desc']} in the {self.current_phase} phase."
            )
        
        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        # this is directly from the petting zoo documentation
        self._cumulative_rewards[agent] = 0

        self.current_player = int(agent.split('_')[1])
        action_dict = self.action_mapping[action]

        self._clear_rewards()

        # ---------------------------------------------------------
        # 1. APPLY THE ACTION FIRST
        # ---------------------------------------------------------
        if self.current_phase == self.phases[0]:
            self.num_turns += 1
            
        mini_reward, next_phase, next_player = self._apply_action(action_dict)
        
        # Update internal state tracking
        self.current_phase = next_phase
        self.current_player = next_player
        self._generate_action_mask()
        
        # Assign mini rewards
        self.rewards[agent] += mini_reward
        self.rewards[agent] += self.discourage_stalling

        # ---------------------------------------------------------
        # 2. EVALUATE TERMINATION AFTER STATE IS UPDATED
        # ---------------------------------------------------------
        is_deadlock = (self.num_passes_in_a_row == self.num_players)
        is_truncated = is_deadlock or self.truncation_condition()
        is_terminated = self.termination_condition()
        
        if is_terminated or is_truncated:
            # first we need to find the winner, taking into account potential ties in points
            max_points = np.max(self.points)
            candidates = np.where(self.points == max_points)[0]
            
            # Apply the tie-breaker if necessary
            if len(candidates) > 1:
                # Get the card counts only for the tied players
                candidate_cards = self.num_cards_in_hand[candidates]
                # Find the index of the candidate with the minimum cards
                winning_candidate_idx = np.argmin(candidate_cards)
                winner = candidates[winning_candidate_idx]
            else:
                winner = candidates[0]

            winner_points = self.points[winner]

            mask = np.ones(self.points.shape, dtype=bool)
            mask[winner] = False
            sum_loser_points = np.sum(self.points[mask])
            
            for i in range(self.num_players):
                if i == winner:
                    base_reward = self.win_points + (self.num_players - 1) * winner_points - sum_loser_points
                else:
                    base_reward = -self.lose_points - (winner_points - self.points[i])
                
                # Apply the deadlock tax to EVERYONE if the game was exhausted
                if is_deadlock:
                    base_reward += self.deadlock_tax
                
                self.rewards[f"player_{i}"] += base_reward
            
            # Terminate all agents
            for a in self.agents:
                self.terminations[a] = is_terminated
                self.truncations[a] = is_truncated
                
            # DO NOT update self.agent_selection here.
            # Leaving it on the current agent triggers the dead step logic on the next call.
        else:
            # ADVANCE TO NEXT PLAYER ONLY IF GAME CONTINUES
            self.agent_selection = f"player_{next_player}"
             
        self._accumulate_rewards()
        
        self.infos[agent] = {
            'phase': self.current_phase,
            'turn': self.current_player,
            'action': action_dict,
            'action_mask': self.action_mask,
            'is_deadlock': is_deadlock
        }

    def render(self):
        """
        Renders the environment to the console.
        """
        if self.render_mode != "console":
            return

        print("\n" + "=" * 70)
        print(f"--- Turn: {self.num_turns} | Current Player: {self.current_player} | Phase: {self.current_phase} ---")
        print("-" * 70)

        # 1. Bank (Tokens Remaining)
        tokens_names = self.colors + ['Gold']
        bank_strs = []
        for i, token_name in enumerate(tokens_names):
            bank_strs.append(f"{token_name}: {self.tokens_remaining[i]}")
        print("BANK:  " + " | ".join(bank_strs))
        print("-" * 70)

        # 2. Nobles
        print("NOBLES:")
        for i in range(self.num_nobles_available):
            noble = self.nobles[i]
            if noble[self.nobles_column_indexer['available']] == 1:
                reqs = []
                for color_name in self.colors:
                    cost = noble[self.nobles_column_indexer[color_name]]
                    if cost > 0:
                        reqs.append(f"{cost} {color_name}")
                print(f"  [Noble {i}] 3 Pts | Requires: {', '.join(reqs)}")
            else:
                print(f"  [Noble {i}] CLAIMED")
        print("-" * 70)

        # 3. Market (Dealt Cards)
        print("MARKET:")
        for tier in reversed(range(self.num_tiers)):
            print(f"  Tier {tier + 1}:")
            for slot in range(4):
                card = self.dealt[tier][slot]
                if card[self.card_column_indexer['available']] == 1:
                    points = card[self.card_column_indexer['points']]
                    color_idx = card[self.card_column_indexer['color']]
                    color_name = self.colors[color_idx] if color_idx < len(self.colors) else "None"
                    
                    costs = []
                    for color in self.colors:
                        cost = card[self.card_column_indexer[color]]
                        if cost > 0:
                            costs.append(f"{cost}{color[0]}") # e.g., "3R" for 3 Red
                            
                    print(f"    Slot {slot} | {points} Pts | {color_name:5} | Cost: {' '.join(costs)}")
                else:
                    print(f"    Slot {slot} | [ EMPTY ]")
        print("-" * 70)

        # 4. Player States
        print("PLAYERS:")
        for p in range(self.num_players):
            active_marker = ">>" if p == self.current_player else "  "
            points = self.points[p]
            
            # Format hand tokens
            hand_tokens = []
            for i, token_name in enumerate(tokens_names):
                count = self.tokens_in_hand[p][i]
                if count > 0:
                    hand_tokens.append(f"{count}{token_name[0]}")
                    
            # Format engine discounts
            engine = []
            for i, color_name in enumerate(self.colors):
                count = self.discounts[p][i]
                if count > 0:
                    engine.append(f"{count}{color_name[0]}")
                    
            num_res = self.num_reserved[p]
            
            print(f"{active_marker} Player {p}: {points:2} Pts | Tokens: [{', '.join(hand_tokens):15}] | Engine: [{', '.join(engine):15}] | Reserved: {num_res}")
            
        print("=" * 70 + "\n")
