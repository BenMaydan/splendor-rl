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
        self._action_pick_noble = self.max_num_players + 1
        self._action_discard = 1 + len(self.colors) # if you have too many tokens - includes ability to discard a gold even though it's objectively never the right move
        self.num_total_actions = self._action_take_3_tokens + self._action_take_2_identical + self._action_reserve_face_up + self._action_reserve_face_down + self._action_buy_face_up + self._action_buy_reserved + self._action_pick_noble + self._action_discard

        # here we precompute some things to make it easier to create the action mask very quickly
        all_combs_take_three_tokens = list(itertools.combinations(np.arange(len(self.colors)), 3))
        self._precomputed_combs_take_three_tokens = np.zeros((len(all_combs_take_three_tokens), 3), dtype=np.uint8)
        for i, comb in enumerate(all_combs_take_three_tokens):
            self._precomputed_combs_take_three_tokens[i] = comb

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
        self.dealt_observation_limits_low = None
        self.dealt_observation_limits_high = None

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
        self.max_num_nobles = self.max_num_players + 1
        self.nobles_observation_limits_low = None
        self.nobles_observation_limits_high = None
        self.num_nobles_points = 3
        self.nobles_columns = ['available'] + self.colors
        self.nobles_column_indexer = {column: i for i, column in enumerate(self.nobles_columns)}
        self.nobles_color_indices = [self.nobles_column_indexer[c] for c in self.colors]

        self.phases = ['main', 'pick_noble', 'discard']
        self.current_phase = 'main'
        self.starting_player = 0
        self.current_player = 0
        self.num_turns = 0
        self.termination_condition = lambda: (self.num_turns % self.num_players == 0 and np.any(self.points >= 15))
        self.truncation_condition = lambda: (self.num_turns >= maximum_total_turns)

        self.mini_rewards = {
            'buy_card': 0.01,
            'get_noble': 0.05,
        }
        self.lose_points = 100
        self.win_points = (self.num_players - 1) * self.lose_points
        self.discourage_stalling = -0.01

        self.initialize_nobles()
        self.initialize_deck()
        self.initialize_misc()

        # initializing observation space
        self.single_observation_space = spaces.Dict({
            "observation": spaces.Dict({
                "phase": spaces.Discrete(len(self.phases)),
                "relative_player_seat": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
                "tier_1_remaining": spaces.Box(low=0, high=self.max_num_cards_at_tier[0], shape=(1,), dtype=np.uint8),
                "tier_2_remaining": spaces.Box(low=0, high=self.max_num_cards_at_tier[1], shape=(1,), dtype=np.uint8),
                "tier_3_remaining": spaces.Box(low=0, high=self.max_num_cards_at_tier[2], shape=(1,), dtype=np.uint8),
                "nobles_remaining": spaces.Box(low=0, high=5, shape=(1,), dtype=np.uint8),
                "tokens_remaining": spaces.Box(low=0, high=7, shape=self.tokens_remaining.shape, dtype=np.uint8),
                "dealt": spaces.Box(low=self.dealt_observation_limits_low, high=self.dealt_observation_limits_high, shape=self.dealt.shape, dtype=np.uint8),
                "nobles": spaces.Box(low=self.nobles_observation_limits_low, high=self.nobles_observation_limits_high, shape=self.nobles.shape, dtype=np.uint8),
                "points": spaces.Box(low=0, high=22, shape=self.points.shape, dtype=np.int8),
                "reserved": spaces.Box(low=0, high=7, shape=self.reserved.shape, dtype=np.uint8),
                "discounts": spaces.Box(low=0, high=self.max_num_of_color, shape=self.discounts.shape, dtype=np.int8),
                "num_cards_in_hand": spaces.Box(low=0, high=30, shape=self.num_cards_in_hand.shape, dtype=np.uint8),
                "tokens_in_hand": spaces.Box(low=0, high=7, shape=self.tokens_in_hand.shape, dtype=np.int8)
            }),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.num_total_actions,), dtype=np.uint8)
        })

        # Action Spaces (Dict mapping agent to space)
        self.action_spaces = {
            agent: spaces.Discrete(self.num_total_actions) for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: self.single_observation_space for agent in self.possible_agents
        }

        # initializing action mapping and mask
        self.action_mapping = {}
        self.action_mask = np.ones((self.num_total_actions,), dtype=np.uint8)
        self._build_action_space()
    
    def initialize_nobles(self, seed=None):
        """
        Writes all the possible nobles to the deck to initialize, to start the game
        """
        # (available, *color_requirements)
        self.nobles = np.zeros((self.max_num_nobles, len(self.nobles_columns)), dtype=np.uint8)

        df = pd.read_csv(os.path.join(os.getcwd(), 'env', 'nobles.csv'), dtype=np.uint8)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        self.nobles_observation_limits_low = np.zeros(self.nobles.shape, dtype=np.uint8)
        self.nobles_observation_limits_high = np.zeros(self.nobles.shape, dtype=np.uint8)

        df_min = df.min()
        df_max = df.max()
        for i, color in enumerate(self.colors):
            self.nobles_observation_limits_low[:, i] = df_min[color]
            self.nobles_observation_limits_high[:, i] = df_max[color]
        
        self.nobles[:self.num_nobles_available, self.nobles_column_indexer['available']] = 1
        self.nobles[:self.num_nobles_available, self.nobles_color_indices] = df[self.colors].to_numpy(dtype=np.uint8)[:self.num_nobles_available]
    
    def initialize_deck(self, seed=None):
        """
        Writes all the possible cards to the deck to initialize, to start the game
        """
        # Read without forcing uint8 on string columns - then shuffle
        df = pd.read_csv(os.path.join(os.getcwd(), 'env', 'cards.csv'))
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Handle color mapping BEFORE casting to uint8
        self.max_num_of_color = np.count_nonzero(df['color'] == self.colors[0])
        color_indices = {color: i for i, color in enumerate(self.colors)}
        df['color'] = df['color'].map(color_indices)

        # Now that 'color' is numeric, cast the whole dataframe to uint8
        # Since everything is now an integer it should not throw an error
        df = df.astype(np.uint8)

        num_tiers = max(df['level'])
        num_cards = [np.count_nonzero(df['level'] == i) for i in range(1, num_tiers + 1)]

        self.deck = np.zeros((num_tiers, max(num_cards), self.card_num_columns), dtype=np.uint8)
        self.max_num_cards_at_tier = np.zeros((num_tiers,), dtype=np.uint8)
        self.num_dealt_at_tier = np.zeros((num_tiers,), dtype=np.uint8) + 4

        dealt_shape = (num_tiers, 4, self.card_num_columns)
        self.dealt_observation_limits_low = np.zeros(dealt_shape, dtype=np.uint8)
        self.dealt_observation_limits_high = np.zeros(dealt_shape, dtype=np.uint8)

        # setting global maximum observation limits for data about the cards
        # this is so the gymnasium API knows how to normalize the columns
        self.dealt_observation_limits_high[:, :, self.card_column_indexer['available']] = 1
        self.dealt_observation_limits_high[:, :, self.card_column_indexer['points']] = 5
        self.dealt_observation_limits_high[:, :, self.card_column_indexer['color']] = len(self.colors)
        self.dealt_observation_limits_high[:, :, self.color_indices] = 7

        # fill in our numpy deck array with the csv data
        for tier in range(num_tiers):
            df_at_tier = df[df['level'] == (tier + 1)]
            self.max_num_cards_at_tier[tier] = len(df_at_tier)

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
        self.points = np.zeros((self.num_players,), dtype=np.int8)
        self.reserved = np.zeros((self.num_players, self.max_able_to_reserve, self.card_num_columns), dtype=np.uint8)
        self.num_reserved = np.zeros((self.num_players,), dtype=np.uint8)
        self.num_cards_in_hand = np.zeros((self.num_players,), dtype=np.uint8)
        self.discounts = np.zeros((self.num_players, len(self.colors)), dtype=np.int8)
        self.tokens_remaining = np.zeros((1 + len(self.colors),), dtype=np.uint8)
        if self.num_players == 4:
            self.tokens_remaining += 7
        elif self.num_players == 3:
            self.tokens_remaining += 5
        elif self.num_players == 2:
            self.tokens_remaining += 4
        self.tokens_remaining[self.gold_index] = 5
        self.tokens_in_hand = np.zeros((self.num_players, 1 + len(self.colors)), dtype=np.int8)

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

        self.initialize_nobles(seed=seed)
        self.initialize_deck(seed=seed)
        self.initialize_misc()

        # re-initialize action mask
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

        # Action 1: taking 3 tokens
        for comb in itertools.combinations(np.arange(len(self.colors)), 3):
            desc = ", ".join([f"1 {self.colors[comb[i]]}" for i in range(len(comb))])
            self.action_mapping[action_idx] = {
                "type": "take_3_tokens",
                "indices": list(comb),
                "desc": f"Take Gems ({desc})"
            }
            action_idx += 1
        
        # Action 2: taking 2 identical tokens
        for color_index in range(len(self.colors)):
            self.action_mapping[action_idx] = {
                "type": "take_2_identical_tokens",
                "index": color_index,
                "desc": f"Take 2 {self.colors[color_index]} Gems"
            }
            action_idx += 1
        
        # # Action 3: Reserving a face up card
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
        
        # # Action 4: Reserving a face down card
        # 3 tiers = 3 actions
        for tier in range(self.num_tiers):
            self.action_mapping[action_idx] = {
                "type": "reserve_face_down",
                "tier": tier,
                "desc": f"Reserve Face Down Card in Tier {tier}"
            }
            action_idx += 1
                
        # Action 5: Buying Face-Up Cards
        for tier in range(self.num_tiers):
            for slot in range(4):
                self.action_mapping[action_idx] = {
                    "type": "buy_face_up",
                    "tier": tier,
                    "slot": slot,
                    "desc": f"Buy Card (Tier {tier}, Slot {slot})"
                }
                action_idx += 1
        
        # Action 6: Buying Reserved Cards
        # Can reserve at most 3 cards
        desc_number = ["1st", "2nd", "3rd", "4th", "5th"]
        for index in range(3):
            self.action_mapping[action_idx] = {
                "type": "buy_reserved",
                "index": index,
                "desc": f"Buy {desc_number[index]} Reserved Card"
            }
            action_idx += 1
        
        # Action 7: Pick a noble if you can
        for index in range(self.num_nobles_available):
            self.action_mapping[action_idx] = {
                "type": "pick_noble",
                "index": index,
                "desc": f"Pick {desc_number[index]} Noble"
            }
            action_idx += 1

        # Action 8: Discard tokens (if you have too many)
        desc_colors = self.colors.copy()
        desc_colors.insert(self.gold_index, "Gold")
        for token_type in range(1 + len(self.colors)):
            self.action_mapping[action_idx] = {
                "type": "discard_token",
                "index": token_type,
                "desc": f"Discard {desc_colors[token_type]} Gem"
            }
            action_idx += 1
    
    def _generate_action_mask(self) -> None:
        """
        Generate the numpy array of the action mask given all the observations we know
        """
        match self.current_phase:
            case "main":
                self.action_mask[:] = 1

                # mask out picking noble and discarding since that is not allowed in the main phase
                s, e = self._action_indices_map["pick_noble"]
                self.action_mask[s:e] = 0
                s, e = self._action_indices_map["discard"]
                self.action_mask[s:e] = 0

                num_reserved = self.num_reserved[self.current_player]

                # mask out invalid actions of taking three tokens using advanced numpy magic to avoid python for loops
                s, e = self._action_indices_map["take_3_tokens"]
                invalid_action_indices = np.any(self.tokens_remaining[self._precomputed_combs_take_three_tokens] == 0, axis=1)
                self.action_mask[s:e][invalid_action_indices] = 0

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
                    invalid_action_indices = (self.num_dealt_at_tier >= self.max_num_cards_at_tier)
                    self.action_mask[s:e][invalid_action_indices] = 0

                # mask out invalid actions of buying face up card
                # an invalid action is: not (card available and card purchasable), which is de'morgans law
                # we can reuse logic to see which face up cards are available
                s, e = self._action_indices_map["buy_face_up"]
                invalid_action_indices = np.logical_not(np.logical_and(
                    (self.dealt[..., self.card_column_indexer['available']] == 1).flatten(),
                    self.get_purchasibility_map(self.tokens_in_hand[self.current_player], self.discounts[self.current_player], self.dealt)
                )).flatten()
                self.action_mask[s:e][invalid_action_indices] = 0

                # mask out invalid actions of buying reserved card
                s, e = self._action_indices_map["buy_reserved"]
                invalid_action_indices = np.logical_not(np.logical_and(
                    self.reserved[self.current_player, ..., self.card_column_indexer['available']] == 1,
                    self.get_purchasibility_map(self.tokens_in_hand[self.current_player], self.discounts[self.current_player], self.reserved[self.current_player])
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
        actual_gem_cost = np.maximum(0, np.minimum(raw_costs - discounts, tokens_in_hand[:len(self.colors)]))
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
            self.tokens_remaining += cost
            self.points[current_player] += card[self.card_column_indexer['points']]
            self.num_cards_in_hand[current_player] += 1
            self.discounts[current_player][card[self.card_column_indexer['color']]] += 1

            # Need to take into account awarding the noble of the players choice if he can get it
            available_nobles_cost = self.nobles[self.nobles[:, self.nobles_column_indexer['available']] == 1][..., self.nobles_color_indices]
            if np.any(np.all(self.discounts[current_player] >= available_nobles_cost, axis=-1), axis=0):
                return "pick_noble", current_player
            return self.current_phase, next_player
        
        def deal_new_card(tier, slot=None):
            # deal new card to dealt assuming there are cards left
            if slot is not None:
                if self.num_dealt_at_tier[tier] < self.max_num_cards_at_tier[tier]:
                    self.dealt[tier, slot] = self.deck[tier, self.num_dealt_at_tier[tier]]
                else:
                    self.dealt[tier, slot, self.card_column_indexer['available']] = 0
                    return
            
            self.num_dealt_at_tier[tier] += 1

        match action_type:
            case "take_3_tokens":
                token_indices = action["indices"]
                self.tokens_in_hand[current_player, token_indices] += 1
                self.tokens_remaining[token_indices] -= 1
                if np.sum(self.tokens_in_hand[current_player]) > self.max_tokens_allowed:
                    return (0, "discard", current_player)
                return (0, self.current_phase, next_player)
            case "take_2_identical_tokens":
                token_index = action["index"]
                self.tokens_in_hand[current_player, token_index] += 2
                self.tokens_remaining[token_index] -= 2
                if np.sum(self.tokens_in_hand[current_player]) > self.max_tokens_allowed:
                    return (0, "discard", current_player)
                return (0, self.current_phase, next_player)
            case "reserve_face_up":
                tier, slot = action["tier"], action["slot"]
                # put reserved card in players reserved pile
                self.reserved[current_player, self.num_reserved[current_player]] = self.dealt[tier, slot]
                self.num_reserved[current_player] += 1

                # take gold token for this player if there are tokens left
                if self.tokens_remaining[self.gold_index] > 0:
                    self.tokens_in_hand[current_player, self.gold_index] += 1
                    self.tokens_remaining[self.gold_index] -= 1
                
                # make sure to move to discard if getting gold token brings you over the limit of allowed tokens
                if np.sum(self.tokens_in_hand[current_player]) > self.max_tokens_allowed:
                    next_phase = 'discard'
                    next_player = self.current_player
                else:
                    next_phase = self.current_phase
                
                deal_new_card(tier, slot)
                
                return (0, next_phase, next_player)
            case "reserve_face_down":
                tier = action["tier"]
                # put reserved card in players reserved pile
                self.reserved[current_player, self.num_reserved[current_player]] = self.deck[tier, self.num_dealt_at_tier[tier]]
                self.num_reserved[current_player] += 1

                deal_new_card(tier)

                # take gold token for this player if there are tokens left
                if self.tokens_remaining[self.gold_index] > 0:
                    self.tokens_in_hand[current_player, self.gold_index] += 1
                    self.tokens_remaining[self.gold_index] -= 1
                
                # make sure to move to discard if getting gold token brings you over the limit of allowed tokens
                if np.sum(self.tokens_in_hand[current_player]) > self.max_tokens_allowed:
                    next_phase = 'discard'
                    next_player = self.current_player
                else:
                    next_phase = self.current_phase
                
                return (0, next_phase, next_player)
            case "buy_face_up":
                tier, slot = action["tier"], action["slot"]
                next_phase, player = buy_card(self.dealt[tier][slot])
                deal_new_card(tier, slot)
                return (self.mini_rewards['buy_card'], next_phase, player)
            case "buy_reserved":
                index = action["index"]
                next_phase, player = buy_card(self.reserved[current_player, index])
                self.reserved[current_player, index, self.card_column_indexer['available']] = 0
                self.num_reserved[current_player] -= 1
                # we shift over the remaining reserved cards
                new_available_reserved_cards = self.reserved[current_player, :, self.card_column_indexer['available']] == 1
                self.reserved[current_player, :self.num_reserved[current_player]] = self.reserved[current_player, new_available_reserved_cards]
                self.reserved[current_player, self.num_reserved[current_player]:, :] = 0
                return (self.mini_rewards['buy_card'], next_phase, player)
            case "pick_noble":
                assert self.current_phase == "pick_noble", "Pick noble action taken outside of pick noble phase!"
                index = action["index"]
                self.points[current_player] += 3
                self.nobles[index, self.nobles_column_indexer['available']] = 0
                return (self.mini_rewards['get_noble'], "main", next_player)
            case "discard_token":
                assert self.current_phase == "discard", "Discard action taken outside of discard phase!"
                index = action["index"]
                self.tokens_remaining[index] += 1
                self.tokens_in_hand[current_player][index] -= 1
                if np.sum(self.tokens_in_hand[current_player]) > self.max_tokens_allowed:
                    return (0, "discard", current_player)
                return (0, "main", next_player)
            case _:
                raise gym.error.InvalidAction(f"Action {action} is ill-defined!")

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        """
        Returns the observation for a specific agent.
        """
        # Extract the integer index from the agent string (e.g., "player_0" -> 0)
        player_idx = int(agent.split('_')[1])
        
        # Generate the new observation state
        # Notice we use player_idx to structure the perspective
        observation = {
            "observation": {
                "phase": self.phases.index(self.current_phase),
                "relative_player_seat": np.array([(player_idx + self.num_players - self.starting_player) % 4], dtype=np.uint8),

                "tier_1_remaining": np.array([self.max_num_cards_at_tier[0]], dtype=np.uint8),
                "tier_2_remaining": np.array([self.max_num_cards_at_tier[1]], dtype=np.uint8),
                "tier_3_remaining": np.array([self.max_num_cards_at_tier[2]], dtype=np.uint8),
                "nobles_remaining": np.array([self.num_nobles_available], dtype=np.uint8),
                "tokens_remaining": self.tokens_remaining,
                
                "dealt": self.dealt,
                "nobles": self.nobles,
                
                "points": self.points,
                "reserved": self.reserved,
                "discounts": self.discounts,
                "num_cards_in_hand": self.num_cards_in_hand,
                "tokens_in_hand": self.tokens_in_hand
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
        # Check if the agent is dead (terminated/truncated)
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)
        
        agent = self.agent_selection
        self.current_player = int(agent.split('_')[1])
        
        # Clear step-wise rewards
        self.rewards = {a: 0 for a in self.agents}
        
        # get important info about the action for execution
        action_dict = self.action_mapping[action]

        # Clear step-wise rewards
        self.rewards = {a: 0 for a in self.agents}

        # Determine if the game has ended naturally (e.g., someone hit 15 points) and everyone has finished their last turn
        # I think this logic also needs to happen if the game has been truncated due to lack of progress
        is_truncated = self.truncation_condition()
        is_terminated = self.termination_condition()
        
        if is_terminated or is_truncated:
            winner = np.argmax(self.points)
            winner_points = self.points[winner]
            mask = np.ones(self.points.shape, dtype=bool)
            mask[winner] = False
            sum_loser_points = np.sum(self.points[mask])
            for i in range(self.num_players):
                if i == winner:
                    self.rewards[f"player_{i}"] += self.win_points + (self.num_players - 1) * winner_points - sum_loser_points
                else:
                    self.rewards[f"player_{i}"] += -self.lose_points - (winner_points - self.points[i])
            
            # we terminate/truncate all the agents so the environment can be reset to play another game
            for a in self.agents:
                self.terminations[a] = is_terminated
                self.truncations[a] = is_truncated
        
        # We let the agent apply its action and update the current player and phase appropriately
        if self.current_phase == self.phases[0]:
            self.num_turns += 1
        mini_reward, next_phase, next_player = self._apply_action(self.current_player, action_dict)
        assert next_phase in self.phases
        assert next_player < self.num_players
        self.current_phase = next_phase
        self.current_player = next_player
        self._generate_action_mask()
        
        # Assign mini rewards to encourage learning something at the beginning of training
        self.rewards[agent] += mini_reward
        self.rewards[agent] += self.discourage_stalling
        
        # Advance to the next agent
        if not (is_terminated or is_truncated):
            self.agent_selection = f"player_{next_player}"
             
        self._accumulate_rewards()
        
        # TODO: maybe we don't want this?
        info = {
            'phase': self.current_phase,
            'turn': self.current_player,
            'action': action_dict
        }
        self.infos[agent] = info

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
