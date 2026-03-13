"""
Dependencies required:
pip install gymnasium numpy stable-baselines3
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import math
import random
import pandas as pd
import itertools

class SplendorEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    This is where the rules of Splendor will live.
    """
    metadata = {"render_modes": ["console", "human"], "render_fps": 30}

    def __init__(self, num_players=4, maximum_total_turns=400, render_mode="console"):
        super(SplendorEnv, self).__init__()
        self.render_mode = render_mode

        self.num_tiers = 3
        self.colors = ['Red', 'Green', 'Blue', 'White', 'Black']
        self.gold_index = len(self.colors)
        self.max_num_of_color = None

        # types of actions
        self._action_take_3_tokens = math.comb(len(self.colors), self.num_tiers)
        self._action_take_2_identical = len(self.colors)
        self._action_reserve_face_up = self.num_tiers * 4
        self._action_reserve_face_down = self.num_tiers
        self._action_buy_face_up = self.num_tiers * 4
        self._action_buy_reserved = 3
        self._action_pick_noble = num_players + 1
        self._action_discard = 1 + len(self.colors) # if you have too many tokens - includes ability to discard a gold even though it's objectively never the right move
        self.num_total_actions = self._action_take_3_tokens + self._action_take_2_identical + self._action_reserve_face_up + self._action_reserve_face_down + self._action_buy_face_up + self._action_buy_reserved + self._action_pick_noble + self._action_discard

        # here we precompute some things to make it easier to create the action mask very quickly
        all_combs_take_three_tokens = itertools.combinations(np.arange(len(self.colors)), 3)
        self._precomputed_combs_take_three_tokens = np.zeros((len(all_combs_take_three_tokens), 3))
        for i, comb in enumerate(all_combs_take_three_tokens):
            self._precomputed_combs_take_three_tokens[i] = comb
        
        self._precomputed_deck_linear_indices = np.zeros((self.num_tiers * 4,), dtype=np.uint8)
        i = 0
        for tier in range(self.num_tiers):
            for slot in range(4):
                self._precomputed_deck_linear_indices[tier]
                i += 1

        # we want to precompute starting indices of the action mask for a given action type (for quick masking)
        # the ending index of the action type is the action_indices[action_type_index + 1]
        self._action_indices = np.zeros((9,), dtype=np.uint8)
        self._action_indices[1] = self._action_take_3_tokens
        self._action_indices[2] = self._action_indices[1] + self._action_take_2_identical
        self._action_indices[3] = self._action_indices[2] + self._action_reserve_face_up
        self._action_indices[4] = self._action_indices[3] + self._action_reserve_face_down
        self._action_indices[5] = self._action_indices[4] + self._action_buy_face_up
        self._action_indices[6] = self._action_indices[5] + self._action_buy_reserved
        self._action_indices[7] = self._action_indices[6] + self._action_pick_noble
        self._action_indices[8] = self.num_total_actions
        self._action_indices_map = {
            "take_3_tokens": [self._action_indices[0], self._action_indices[1]],
            "take_2_tokens": [self._action_indices[1], self._action_indices[2]],
            "reserve_face_up": [self._action_indices[2], self._action_indices[3]],
            "reserve_face_down": [self._action_indices[3], self._action_indices[4]],
            "buy_face_up": [self._action_indices[4], self._action_indices[5]],
            "buy_reserved": [self._action_indices[5], self._action_indices[6]],
            "pick_noble": [self._action_indices[6], self._action_indices[7]],
            "discard": [self._action_indices[7], self._action_indices[8]],
        }

        # initializing action space + mask
        self.action_mapping = {}
        self._build_action_space()
        self.action_space = spaces.Discrete(self.num_total_actions)
        self.action_mask = np.ones((self.num_total_actions,), dtype=np.uint8)

        # Initializing internal game state variables
        self.get_next_player = lambda: (self.current_player + 1) % self.num_players

        # data about the card, we store it here to avoid hardcoding
        card_columns = ['available', 'points', 'color'] + self.colors
        self.card_column_indexer = {column: i for i, column in enumerate(card_columns)}
        self.color_indices = [self.card_column_indexer[c] for c in self.colors]
        self.card_num_columns = len(card_columns)

        self.deck = None
        self.dealt = None
        self.max_num_cards_at_tier = None
        self.num_dealt_at_tier = None
        self.card_observation_limits_low = None
        self.card_observation_limits_high = None

        self.points = None
        self.reserved = None
        self.num_reserved = None
        self.max_able_to_reserve = 3
        self.num_cards_in_hand = None
        self.discounts = None
        self.tokens_remaining = None
        self.tokens_in_hand = None
        self.max_tokens_allowed = 10

        self.nobles = None
        self.num_nobles_available = num_players + 1
        self.nobles_observation_limits_low = None
        self.nobles_observation_limits_high = None
        self.num_nobles_points = 3
        nobles_columns = ['available'] + self.colors
        self.nobles_column_indexer = {column: i for i, column in enumerate(nobles_columns)}
        self.nobles_color_indices = [self.nobles_column_indexer[c] for c in self.colors]

        self.phases = ['main', 'pick_noble', 'discard']
        self.current_phase = 'main'
        self.num_players = num_players
        self.starting_player = 0
        self.current_player = 0
        self.num_turns = 0
        self.termination_condition = lambda: (self.num_turns % self.num_players == 0 and np.any(self.points >= 15))
        self.truncation_condition = lambda: (self.num_turns >= maximum_total_turns)

        self.mini_rewards = {
            'buy_card': 0.01,
            'get_noble': 0.05,
        }
        self.discourage_stalling = -0.01

        self.initialize_nobles()
        self.initialize_deck()
        self.initialize_misc()

        # initializing observation space
        self.observation_space = spaces.Dict({
            "phase": spaces.Discrete(len(self.phases), dtype=np.uint8),
            "relative_player_seat": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
            "tier_1_remaining": spaces.Box(low=0, high=self.max_num_cards_at_tier[0], shape=(1,), dtype=np.uint8),
            "tier_2_remaining": spaces.Box(low=0, high=self.max_num_cards_at_tier[1], shape=(1,), dtype=np.uint8),
            "tier_3_remaining": spaces.Box(low=0, high=self.max_num_cards_at_tier[2], shape=(1,), dtype=np.uint8),
            "nobles_remaining": spaces.Box(low=0, high=5, shape=(1,), dtype=np.uint8),
            "tokens_remaining": spaces.Box(low=0, high=7, shape=self.tokens_remaining.shape, dtype=np.uint8),
            "dealt": spaces.Box(low=self.card_observation_limits_low, high=self.card_observation_limits_high, shape=self.dealt.shape, dtype=np.uint8),
            "nobles": spaces.Box(low=self.nobles_observation_limits_low, high=self.nobles_observation_limits_high, shape=self.nobles.shape, dtype=np.uint8),
            "points": spaces.Box(low=0, high=22, shape=self.points.shape, dtype=np.uint8),
            "reserved": spaces.Box(low=0, high=7, shape=self.reserved.shape, dtype=np.uint8),
            "discounts": spaces.Box(low=0, high=self.max_num_of_color, shape=self.discounts.shape, dtype=np.uint8),
            "num_cards_in_hand": spaces.Box(low=0, high=30, shape=self.num_cards_in_hand.shape, dtype=np.uint8),
            "tokens_in_hand": spaces.Box(low=0, high=7, shape=self.tokens_in_hand.shape, dtype=np.uint8),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.num_total_actions,), dtype=np.int8)
        })
    
    def initialize_nobles(self, seed=None):
        """
        Writes all the possible nobles to the deck to initialize, to start the game
        """
        df = pd.read_csv('nobles.csv', dtype=np.uint8)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        self.nobles_observation_limits_low = np.zeros(self.nobles.shape, dtype=np.uint8)
        self.nobles_observation_limits_high = np.zeros(self.nobles.shape, dtype=np.uint8)

        df_min = df.min()
        df_max = df.max()
        for i, color in enumerate(self.colors):
            self.nobles_observation_limits_low[:, i] = df_min[color]
            self.nobles_observation_limits_high[:, i] = df_max[color]
        
        # (available, *color_requirements)
        self.nobles = np.zeros((self.num_nobles_available, 2 + len(self.colors)), dtype=np.uint8)
        self.nobles[:, self.nobles_column_indexer['available']] = 1
        self.nobles[:, 1:] = df[self.colors].to_numpy(dtype=np.uint8)[:self.num_nobles_available]
    
    def initialize_deck(self, seed=None):
        """
        Writes all the possible cards to the deck to initialize, to start the game
        """
        df = pd.read_csv('cards.csv', dtype=np.uint8)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        self.max_num_of_color = len(df['color'] == self.colors[0])

        color_indices = {color: i for i, color in enumerate(self.colors)}
        df['color'] = df['color'].map(color_indices)
        num_tiers = max(df['level'])
        num_cards = [len(df['level'] == i) for i in range(1, num_tiers + 1)]

        self.deck = np.zeros((num_tiers, max(num_cards), self.card_num_columns), dtype=np.uint8)
        self.max_num_cards_at_tier = np.zeros((num_tiers,), dtype=np.uint8)
        self.num_dealt_at_tier = np.zeros((num_tiers,), dtype=np.uint8) + 4

        self.card_observation_limits_low = np.zeros(self.deck.shape, dtype=np.uint8)
        self.card_observation_limits_high = np.zeros(self.deck.shape, dtype=np.uint8)

        # setting global maximum observation limits for data about the cards
        # this is so the gymnasium API knows how to normalize the columns
        self.card_observation_limits_high[:, :, self.card_column_indexer['available']] = 1
        self.card_observation_limits_high[:, :, self.card_column_indexer['points']] = 5
        self.card_observation_limits_high[:, :, self.card_column_indexer['color']] = len(self.colors)
        self.card_observation_limits_high[:, :, self.color_indices] = 7

        # fill in our numpy deck array with the csv data
        for tier in range(1, num_tiers + 1):
            df_at_tier = (df['level'] == tier)
            self.max_num_cards_at_tier[tier - 1] = len(df_at_tier)

            # available is hardcoded to 1 for now until cards start to run out
            self.deck[tier, :len(df_at_tier), self.card_column_indexer['available']] = 1

            # now we fill in our cards numpy array using the data in the csv
            for column, column_index in self.card_column_indexer.items():
                if column == 'available':
                    continue
                self.deck[tier, :len(df_at_tier), column_index] = df_at_tier[column].values
        
        self.dealt = self.deck[:, :4, :]

    def initialize_misc(self):
        """
        Initialize observational data about purchased self + opponents card
        """
        # need to store num_players x (discount_red, discount_blue, ..., discount_gold)
        self.points = np.zeros((self.num_players,), dtype=np.uint8)
        self.reserved = np.zeros((self.num_players, self.max_able_to_reserve, self.card_num_columns), dtype=np.uint8)
        self.num_reserved = np.zeros((self.num_players,), dtype=np.uint8)
        self.num_cards_in_hand = np.zeros((self.num_players,), dtype=np.uint8)
        self.discounts = np.zeros((self.num_players, len(self.colors)), dtype=np.uint8)
        self.tokens_remaining = np.zeros((1 + len(self.colors),), dtype=np.uint8)
        if self.num_players == 3:
            self.tokens_remaining += 5
        elif self.num_players == 2:
            self.tokens_remaining += 4
        self.tokens_remaining[self.gold_index] = 5
        self.tokens_in_hand = np.zeros((self.num_players, 1 + len(self.colors)), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.
        """
        super().reset(seed=seed)
        random.seed(seed)
        
        # Reset the internal game state to the start of a Splendor game
        self.current_phase = 'main'
        if options:
            self.num_players = options['num_players']
        else:
            self.num_players = 4
        
        # TODO: initialize starting player index randomly
        self.starting_player = random.randint(0, self.num_players - 1)
        self.current_player = self.starting_player
        self.num_turns = 0

        self.initialize_nobles(seed=seed)
        self.initialize_deck(seed=seed)
        self.initialize_misc()

        # re-initialize action mask
        self.action_mask = self._generate_action_mask()
        
        # Generate the starting observation based on the reset state
        observation = {
            "phase": self.current_phase,
            "relative_player_seat": (self.current_player + self.num_players - self.starting_player) % 4,

            "tier_1_remaining": np.uint8(self.max_num_cards_at_tier[0]),
            "tier_2_remaining": np.uint8(self.max_num_cards_at_tier[1]),
            "tier_3_remaining": np.uint8(self.max_num_cards_at_tier[2]),
            "nobles_remaining": np.uint8(self.num_nobles_available),
            
            "dealt": self.dealt,
            "nobles": self.nobles,
            
            # initialization of player stats (zeros for a new game)
            "points": self.points,
            "reserved": self.reserved,
            "discounts": self.discounts,
            "num_cards_in_hand": self.num_cards_in_hand,
            "tokens_in_hand": self.tokens_in_hand,
            
            "action_mask": self.action_mask
        }
        
        # info can contain auxiliary diagnostic information
        info = {
            'current_phase': self.current_phase,
            'current_player': self.current_player,
        }
        
        return observation, info

    def _build_action_space(self):
        """
        Populates self.action_mapping with human-readable dictionaries.
        """
        action_idx = 0

        # Action 1: taking 3 tokens
        for comb in itertools.combinations(np.arange(len(self.colors)), 3):
            self.action_mapping[action_idx] = {
                "type": "take_3_tokens",
                "indices": comb
            }
            action_idx += 1
        
        # Action 2: taking 2 identical tokens
        for color_index in range(len(self.colors)):
            self.action_mapping[action_idx] = {
                "type": "take_2_identical_tokens",
                "index": color_index
            }
            action_idx += 1
        
        # # Action 3: Reserving a face up card
        # 3 tiers, 4 cards each = 12 actions
        for tier in range(self.num_tiers):
            for slot in range(4):
                self.action_mapping[action_idx] = {
                    "type": "reserve_face_up",
                    "tier": tier,
                    "slot": slot
                }
                action_idx += 1
        
        # # Action 4: Reserving a face down card
        # 3 tiers = 3 actions
        for tier in range(self.num_tiers):
            self.action_mapping[action_idx] = {
                "type": "reserve_face_down",
                "tier": tier,
            }
            action_idx += 1
                
        # Action 5: Buying Face-Up Cards
        for tier in range(self.num_tiers):
            for slot in range(4):
                self.action_mapping[action_idx] = {
                    "type": "buy_face_up",
                    "tier": tier,
                    "slot": slot
                }
                action_idx += 1
        
        # Action 6: Buying Reserved Cards
        # Can reserve at most 3 cards
        for index in range(3):
            self.action_mapping[action_idx] = {
                "type": "buy_reserved",
                "index": index,
            }
            action_idx += 1
        
        # Action 7: Pick a noble if you can
        for index in range(self.num_nobles_available):
            self.action_mapping[action_idx] = {
                "type": "pick_noble",
                "index": "index"
            }
            action_idx += 1

        # Action 8: Discard tokens (if you have too many)
        for token_type in range(1 + len(self.colors)):
            self.action_mapping[action_idx] = {
                "type": "discard_token",
                "index": token_type,
            }
            action_idx += 1
    
    def _generate_action_mask(self):
        """
        Generate the numpy array of the action mask given all the observations we know
        """
        match self.current_phase:
            case "main":
                self.action_mask[:] = 1

                # mask out invalid actions of taking three tokens using advanced numpy magic to avoid python for loops
                s, e = self._action_indices_map["take_3_tokens"]
                invalid_action_indices = np.any(self.tokens_remaining[self._precomputed_combs_take_three_tokens] == 0, axis=1)
                self.action_mask[s:e][invalid_action_indices] = 0

                # mask out invalid actions of taking two identical tokens
                s, e = self._action_indices_map["take_2_tokens"]
                invalid_action_indices = self.tokens_remaining < 2
                self.action_mask[s:e][invalid_action_indices] = 0

                # mask out invalid actions of reserving face up card
                # the only cards you cannot reserve are ones that don't exist because that tier was already completely dealt out
                s, e = self._action_indices_map["reserve_face_up"]
                invalid_action_indices = (self.dealt[..., self.card_column_indexer['available']] == 0).flatten()
                self.action_mask[s:e][invalid_action_indices] = 0

                # mask out invalid actions of reserving face down card
                # only time this is invalid is if that tier has been completely dealt out
                s, e = self._action_indices_map["reserve_face_down"]
                invalid_action_indices = (self.num_dealt_at_tier == self.max_num_cards_at_tier)
                self.action_mask[s:e][invalid_action_indices] = 0

                # mask out invalid actions of buying face up card
                # an invalid action is: not (card available and card purchasable), which is de'morgans law
                # we can reuse logic to see which face up cards are available
                s, e = self._action_indices_map["buy_face_up"]
                invalid_action_indices = np.logical_not(np.logical_and(
                    self.dealt[..., self.card_column_indexer['available']] == 1,
                    self.get_purchasibility_map(self.tokens_in_hand[self.current_player], self.discounts[self.current_player], self.dealt)
                )).flatten()
                self.action_mask[s:e][invalid_action_indices] = 0

                # mask out invalid actions of buying reserved card
                s, e = self._action_indices_map["buy_reserved"]
                invalid_action_indices = np.logical_not(np.logical_and(
                    self.reserved[..., self.card_column_indexer['available']] == 1,
                    self.get_purchasibility_map(self.tokens_in_hand[self.current_player], self.discounts[self.current_player], self.reserved)
                )).flatten()
                self.action_mask[s:e][invalid_action_indices] = 0
            
            case "pick_noble":
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
                self.action_mask[:] = 0
                # only allow actions for tokens you are able to discard
                s, e = self._action_indices_map["discard"]
                valid_action_indices = (self.tokens_in_hand[self.current_player] > 0)
                self.action_mask[s:e][valid_action_indices] = 1

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
        actual_gem_cost = np.minimum(raw_costs, tokens_in_hand[:len(self.colors)])
        return np.append(actual_gem_cost, gold_needed).astype(np.uint8)
    
    def get_purchasibility_map(self, tokens_in_hand, discounts, cards) -> NDArray[np.uint8]:
        """
        Determines the token cost (and if gold tokens are necessary) to buy a card
        Return inf if player doesn't have enough tokens (including gold tokens)
        """
        # tokens_available: (*colors, gold_index)
        # card: (*colors)
        raw_costs = cards[..., self.color_indices]
        deficit_per_color = np.maximum(0, raw_costs - discounts - tokens_in_hand[:len(self.colors)])
        gold_needed = np.sum(deficit_per_color, axis=-1)
        return (tokens_in_hand[self.gold_index] >= gold_needed).flatten()

    def _apply_action(self, current_player, action) -> tuple[int, str, int]:
        """
        Applies the action to the current environment
        Returns:
            (mini_reward, next phase, next player index)
            Sometimes next player index does not change if there is a phase change
        """
        action_type = action["type"]
        next_player = (current_player + 1) % self.num_players

        def buy_card(card) -> tuple[str, int]:
            cost = self._token_cost(self.tokens_in_hand[current_player], self.discounts[current_player], card) # can't be none since we know the action is valid
            self.tokens_in_hand[current_player] -= cost
            self.points[current_player] += card[self.card_column_indexer['points']]
            self.num_cards_in_hand[current_player] += 1
            self.discounts[current_player][card[self.card_column_indexer['color']]] += 1

            # Need to take into account awarding the noble of the players choice if he can get it
            available_nobles_cost = self.nobles[self.nobles[:, self.nobles_column_indexer['available']] == 1]
            deficit = self.discounts[current_player] - available_nobles_cost
            if np.any(np.all(deficit > 0, axis=1), axis=0):
                return "pick_noble", current_player
            return self.current_phase, next_player

        match action_type:
            case "take_3_tokens":
                token_indices = action["indices"]
                self.tokens_in_hand[current_player, token_indices] += 1
                self.tokens_remaining[token_indices] -= 1
                if np.sum(self.tokens_in_hand[current_player]) > self.max_tokens_allowed:
                    return ("discard", current_player)
                return (0, self.current_phase, next_player)
            case "take_2_identical_tokens":
                token_index = action["index"]
                self.tokens_in_hand[current_player, token_index] += 2
                self.tokens_remaining[token_index] -= 2
                if np.sum(self.tokens_in_hand[current_player]) > self.max_tokens_allowed:
                    return ("discard", current_player)
                return (0, self.current_phase, next_player)
            case "reserve_face_up":
                tier, slot = action["tier"], action["slot"]
                # put reserved card in players reserved pile
                self.reserved[current_player, self.num_reserved[current_player]] = self.dealt[tier][slot]
                self.num_reserved[current_player] += 1

                # take gold token for this player if there are tokens left
                if self.tokens_remaining[self.gold_index] > 0:
                    self.tokens_in_hand[current_player, self.gold_index] += 1
                    self.tokens_remaining[self.gold_index] -= 1

                # deal new card to dealt assuming there are cards left
                self.num_dealt_at_tier[tier] += 1
                if self.num_dealt_at_tier[tier] < self.max_num_cards_at_tier[tier]:
                    self.dealt[tier][slot] = self.deck[tier][self.num_dealt_at_tier[tier]]
                
                return (0, self.current_phase, next_player)
            case "reserve_face_down":
                tier, slot = action["tier"], action["slot"]
                # put reserved card in players reserved pile
                self.reserved[current_player, self.num_reserved[current_player]] = self.deck[tier, self.num_dealt_at_tier[tier]]
                self.num_reserved[current_player] += 1
                self.num_dealt_at_tier[tier] += 1

                # take gold token for this player if there are tokens left
                if self.tokens_remaining[self.gold_index] > 0:
                    self.tokens_in_hand[current_player, self.gold_index] += 1
                    self.tokens_remaining[self.gold_index] -= 1
                
                return (0, self.current_phase, next_player)
            case "buy_face_up":
                tier, slot = action["tier"], action["slot"]
                next_phase, player = buy_card(self.dealt[tier][slot])
                return (self.mini_rewards['buy_card'], next_phase, player)
            case "buy_reserved":
                index = action["index"]
                next_phase, player = buy_card(self.reserved[current_player, index])
                return (self.mini_rewards['buy_card'], next_phase, player)
            case "pick_noble":
                index = action["index"]
                self.points[current_player] += 3
                self.nobles[index, self.nobles_column_indexer['available']] = 0
                return (self.mini_rewards['get_noble'], "main", next_player)
            case "discard_token":
                index = action["index"]
                self.tokens_remaining[index] += 1
                self.tokens_in_hand[current_player][index] -= 1
                if np.sum(self.tokens_in_hand[current_player]) > self.max_tokens_allowed:
                    return ("discard", current_player)
                return ("main", next_player)
            case _:
                raise gym.error.InvalidAction(f"Action {action} is ill-defined!")

    def step(self, action):
        """
        Executes one time step within the environment based on the given action.
        """
        self.current_step += 1
        action = self.action_mapping[action]

        # Determine if the game has ended naturally (e.g., someone hit 15 points) and everyone has finished their last turn
        if self.termination_condition():
            winner = np.argmax(self.points)
            winner_points = self.points[winner]
            self.points[winner] = 0
            best_opponent_points = np.max(self.points)
            reward = {}
            for i in range(self.num_players):
                if i == winner:
                    reward[f"player_{i}"] = self.win_points + winner_points - best_opponent_points
                else:
                    reward[f"player_{i}"] = -self.win_points + self.points[i] - winner_points
        
        # Implementing game logic for the action taken
        # 1. Update internal state (tokens, cards, points) based on 'action'
        # 2. Handle invalid actions (e.g., negative reward and ignore, or mask them)
        if self.current_phase == self.phases[0]:
            self.num_turns += 1
        mini_reward, next_phase, next_player = self._apply_action(self.current_player, action)
        assert next_phase in self.phases
        assert next_player < self.num_players
        self.current_phase = next_phase
        self.current_player = next_player
        
        # Assign mini rewards to encourage learning something at the beginning of training
        for i in range(self.num_players):
            if i == self.current_player:
                reward[f"player_{i}"] += mini_reward
            reward[f"player_{i}"] += self.discourage_stalling
        
        # Generate the action mask -- very complicated so we use a helper function
        self.action_mask = self._generate_action_mask()
        
        # Generate the new observation state
        observation = {
            "phase": self.current_phase,
            "relative_player_seat": (self.current_player + self.num_players - self.starting_player) % 4,

            "tier_1_remaining": np.uint8(self.max_num_cards_at_tier[0]),
            "tier_2_remaining": np.uint8(self.max_num_cards_at_tier[1]),
            "tier_3_remaining": np.uint8(self.max_num_cards_at_tier[2]),
            "nobles_remaining": np.uint8(self.num_nobles_available),
            
            "dealt": self.dealt,
            "nobles": self.nobles,
            
            # initialization of player stats (zeros for a new game)
            "points": self.points,
            "reserved": self.reserved,
            "discounts": self.discounts,
            "num_cards_in_hand": self.num_cards_in_hand,
            "tokens_in_hand": self.tokens_in_hand,
            
            "action_mask": self.action_mask
        }
        
        info = {
            'phase': self.current_phase,
            'turn': self.current_player,
            'action': self.action_mapping[action]
        }
        
        return observation, reward, terminated, self.truncation_condition(), info

    def render(self):
        """
        Renders the environment.
        """
        if self.render_mode == "console":
            # TODO: Print a text representation of the board state
            pass


if __name__ == "__main__":
    # 1. Instantiate the environment
    env = SplendorEnv()
    
    # Optional: Verify the environment follows the Gym API before training
    # check_env(env, warn=True)

    # 2. Initialize the RL model
    # PPO (Proximal Policy Optimization) is a strong default algorithm.
    # MlpPolicy is used for standard vector observations.
    model = PPO("MlpPolicy", env, verbose=1)

    # 3. Train the agent
    # TODO: Adjust total_timesteps based on training needs
    print("Starting training...")
    model.learn(total_timesteps=100_000)

    # 4. Save the trained model
    model.save("splendor_ppo_model")
    print("Model saved.")

    # 5. Evaluate / Test the trained agent
    # Load the model (just to show how it's done)
    # model = PPO.load("splendor_ppo_model")
    
    obs, info = env.reset()
    for _ in range(50):
        # The agent predicts the best action based on the observation
        action, _states = model.predict(obs, deterministic=True)
        
        # The environment steps forward based on the action
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        
        if terminated or truncated:
            obs, info = env.reset()