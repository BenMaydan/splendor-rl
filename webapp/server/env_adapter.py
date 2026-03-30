import sys
import os
import uuid
import numpy as np
import time

# Add the root directory to path so we can import env.splendor_env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from env.splendor_env import SplendorEnv

COLOR_INDEX_TO_STR = {0: 'r', 1: 'g', 2: 'u', 3: 'w', 4: 'b', 5: '*'}
STR_TO_COLOR_INDEX = {'r': 0, 'g': 1, 'u': 2, 'w': 3, 'b': 4, '*': 5}

class SplendorEnvAdapter:
    def __init__(self, num_players=4):
        self.env = SplendorEnv(num_players=num_players)
        self.env.reset(seed=int(time.time() * 1000) % (2**31 - 1))
        
        self.sig_to_uuid = {}
        self.uuid_to_sig = {}
        
        self.noble_sig_to_uuid = {}
        self.uuid_to_noble_sig = {}
        
        self.pending_take = [] # list of color strings
        self.pids = list(range(num_players))
        
        # Pre-assign UUIDs to all cards in the deck
        for tier in range(self.env.num_tiers):
            tier_cards = self.env.deck[tier]
            for i in range(self.env._max_num_cards_at_tier[tier]):
                card = tier_cards[i]
                sig = self._card_sig(card)
                self._assign_uuid(sig)
                
        # Pre-assign UUIDs to all nobles
        for i in range(self.env.max_num_nobles):
            noble = self.env.nobles[i]
            sig = self._noble_sig(noble)
            self._assign_noble_uuid(sig)
            
    def _card_sig(self, card):
        """Signature is points, color, and cost features. Ignoring 'available' flag."""
        return tuple(card[1:])
        
    def _noble_sig(self, noble):
        """Signature is features ignoring 'available'."""
        return tuple(noble[1:])
        
    def _assign_uuid(self, sig):
        if sig not in self.sig_to_uuid:
            id_val = uuid.uuid4().hex
            self.sig_to_uuid[sig] = id_val
            self.uuid_to_sig[id_val] = sig
        return self.sig_to_uuid[sig]

    def _assign_noble_uuid(self, sig):
        if sig not in self.noble_sig_to_uuid:
            id_val = uuid.uuid4().hex
            self.noble_sig_to_uuid[sig] = id_val
            self.uuid_to_noble_sig[id_val] = sig
        return self.noble_sig_to_uuid[sig]

    def _get_action_index(self, condition):
        """Find the gym integer action where the mapped dict satisfies the condition AND is unmasked."""
        for action_idx, mapping in self.env.action_mapping.items():
            if self.env.action_mask[action_idx] == 1 and condition(mapping):
                return action_idx
        return None
        
    def _card_to_dict(self, card, level_idx=None):
        sig = self._card_sig(card)
        card_uuid = self.sig_to_uuid.get(sig, "")
        color_idx = card[self.env.card_column_indexer['color']]
        points = card[self.env.card_column_indexer['points']]
        
        cost = {}
        for c_idx in range(5):
            c_val = card[self.env.card_column_indexer[self.env.colors[c_idx]]]
            if c_val > 0:
                cost[COLOR_INDEX_TO_STR[c_idx]] = int(c_val)
                
        return {
            'color': COLOR_INDEX_TO_STR[color_idx] if color_idx < 5 else '',
            'points': int(points),
            'uuid': card_uuid,
            'cost': cost,
            'level': f"level{level_idx+1}" if level_idx is not None else ""
        }
        
    def dict(self, player_id=None):
        """Translate the SplendorEnv state to the JSON format expected by React Game.dict()"""
        state = {
            'players': [],
            'cards': {'level1': [], 'level2': [], 'level3': []},
            'decks': {},
            'log': getattr(self, 'logs', []),
            'gems': {},
            'nobles': [],
            'winner': None,
            'turn': self.env.current_player
        }
        
        # Fill gems
        for c_idx in range(5):
            color_str = COLOR_INDEX_TO_STR[c_idx]
            val = int(self.env.tokens_remaining[c_idx])
            if color_str in self.pending_take:
                val -= self.pending_take.count(color_str)
            state['gems'][color_str] = val
        state['gems']['*'] = int(self.env.tokens_remaining[self.env.gold_index])
        
        # Fill cards (dealt)
        for tier in range(self.env.num_tiers):
            t_name = f"level{tier+1}"
            state['decks'][t_name] = int(self.env._max_num_cards_at_tier[tier] - self.env.num_dealt_at_tier[tier])
            for slot in range(4):
                card = self.env.dealt[tier][slot]
                if card[self.env.card_column_indexer['available']] == 1:
                    state['cards'][t_name].append(self._card_to_dict(card, tier))
                    
        # Fill nobles
        for i in range(self.env.num_nobles_available):
            noble = self.env.nobles[i]
            if noble[self.env.nobles_column_indexer['available']] == 1:
                sig = self._noble_sig(noble)
                req = {}
                for c_idx in range(5):
                    c_val = noble[self.env.nobles_column_indexer[self.env.colors[c_idx]]]
                    if c_val > 0:
                        req[COLOR_INDEX_TO_STR[c_idx]] = int(c_val)
                state['nobles'].append({
                    'id': i,
                    'points': int(self.env.num_nobles_points),
                    'uuid': self.noble_sig_to_uuid.get(sig, ""),
                    'requirement': req
                })
                
        # Fill players
        for p in range(self.env.num_players):
            cards = {'w': [], 'u': [], 'g': [], 'b': [], 'r': []}
            # Instead of keeping arrays of cards, the UI just shows counts, EXCEPT for drawing. 
            # Wait, the React UI uses player.cards to render bought cards.
            # But the gym environment only tracks self.discounts, which are the COUNTS of engines, not the actual cards!
            # Since all cards of the same color are equivalent for engine purposes, we can just generate dummy cards.
            for c_idx in range(5):
                count = self.env.discounts[p][c_idx]
                for _ in range(count):
                    # Dummy card to satisfy the UI
                    cards[COLOR_INDEX_TO_STR[c_idx]].append({
                        'color': COLOR_INDEX_TO_STR[c_idx],
                        'points': 0, 'uuid': uuid.uuid4().hex, 'cost': {}, 'level': ''
                    })
                    
            gems = {}
            for c_idx in range(5):
                color_str = COLOR_INDEX_TO_STR[c_idx]
                val = int(self.env.tokens_in_hand[p][c_idx])
                if p == self.env.current_player and color_str in self.pending_take:
                    val += self.pending_take.count(color_str)
                gems[color_str] = val
            gems['*'] = int(self.env.tokens_in_hand[p][self.env.gold_index])
            
            reserved = []
            for r_idx in range(self.env.num_reserved[p]):
                r_card = self.env.reserved[p][r_idx]
                if r_card[self.env.card_column_indexer['available']] == 1:
                    reserved.append(self._card_to_dict(r_card))
                    
            p_dict = {
                'id': p,
                'name': getattr(self, f"player_name_{p}", f"Player {p+1}"),
                'uuid': getattr(self, f"player_uuid_{p}", ""),
                'reserved': reserved,
                'nobles': [], # Gym env doesn't store which nobles you got, just updates points!
                'cards': cards,
                'gems': gems,
                'score': int(self.env.points[p])
            }
            # Find winner logic
            if self.env.terminations[f"player_{p}"] or self.env.truncations[f"player_{p}"]:
                # The winner is computed sequentially by gym, but we can just use score if it's over
                pass 
            state['players'].append(p_dict)
            
        # Determine winner if game is over
        if all(self.env.terminations.values()) or all(self.env.truncations.values()):
            # Game over
            best_score = -1
            winner = 0
            for p in range(self.env.num_players):
                if self.env.points[p] > best_score:
                    best_score = self.env.points[p]
                    winner = p
            state['winner'] = winner
            
        state['phase'] = getattr(self.env.unwrapped, "current_phase", "main")
            
        # Add valid_actions mask translated to feature identifiers
        valid_actions = { 'buy': [], 'reserve': [], 'noble': [], 'take': [], 'discard': [] }
        if hasattr(self.env, 'action_mapping'):
            for action_idx, mapping in self.env.action_mapping.items():
                if self.env.action_mask[action_idx] == 1:
                    typ = mapping['type']
                    if typ == 'buy_face_up':
                        card = self.env.dealt[mapping['tier']][mapping['slot']]
                        valid_actions['buy'].append(self.sig_to_uuid.get(self._card_sig(card)))
                    elif typ == 'buy_reserved':
                        card = self.env.reserved[self.env.current_player][mapping['index']]
                        valid_actions['buy'].append(self.sig_to_uuid.get(self._card_sig(card)))
                    elif typ == 'reserve_face_up':
                        card = self.env.dealt[mapping['tier']][mapping['slot']]
                        valid_actions['reserve'].append(self.sig_to_uuid.get(self._card_sig(card)))
                    elif typ == 'reserve_face_down':
                        valid_actions['reserve'].append(f"level{mapping['tier']+1}")
                    elif typ == 'pick_noble':
                        noble = self.env.nobles[mapping['index']]
                        valid_actions['noble'].append(self.noble_sig_to_uuid.get(self._noble_sig(noble)))
                    elif typ == 'discard':
                        valid_actions['discard'].append(COLOR_INDEX_TO_STR[mapping['index']])
                    elif typ in ['take_3_diff_tokens', 'take_2_diff_tokens', 'take_1_token']:
                        for i in mapping['indices']:
                            valid_actions['take'].append(COLOR_INDEX_TO_STR[i])
                    elif typ in ['take_2_identical_tokens', 'take_2_tokens']:
                        valid_actions['take'].append(COLOR_INDEX_TO_STR[mapping['index']])
            valid_actions['take'] = list(set(valid_actions['take']))
        state['valid_actions'] = valid_actions

        return state

    def start_turn(self):
        self.pending_take = []
        
    def _generate_log_msg(self, mapping):
        t = mapping.get('type')
        p = self.env.current_player
        name = getattr(self, f"player_name_{p}", f"Player {p+1}")
        
        def format_card(card, tier=None, is_noble=False):
            if is_noble:
                reqs = []
                for i, c in enumerate(self.env.colors):
                    if card[self.env.nobles_column_indexer[c]] > 0:
                        reqs.append(f"{card[self.env.nobles_column_indexer[c]]}[{COLOR_INDEX_TO_STR[i]}]")
                return f"Noble (3 points, requires {' '.join(reqs)})"
            
            pts = card[self.env.card_column_indexer['points']]
            col_idx = card[self.env.card_column_indexer['color']]
            color_str = f"[{COLOR_INDEX_TO_STR[col_idx]}] " if col_idx < 5 else ""
            
            costs = []
            for i, c in enumerate(self.env.colors):
                if card[self.env.card_column_indexer[c]] > 0:
                    costs.append(f"{card[self.env.card_column_indexer[c]]}[{COLOR_INDEX_TO_STR[i]}]")
                    
            tier_str = f"level{tier+1} " if tier is not None else ""
            return f"a {color_str}{tier_str}card ({pts} points, cost: {' '.join(costs)})"

        if t in ['take_3_diff_tokens', 'take_2_diff_tokens', 'take_1_token', 'take_2_identical_tokens', 'take_2_tokens']:
            if 'indices' in mapping:
                colors = [f"[{COLOR_INDEX_TO_STR[i]}]" for i in mapping['indices']]
                return f"{name} took {' '.join(colors)}"
            elif 'index' in mapping:
                c = f"[{COLOR_INDEX_TO_STR[mapping['index']]}]"
                return f"{name} took {c} {c}"
        elif t == 'buy_face_up':
            card = self.env.dealt[mapping['tier']][mapping['slot']]
            return f"{name} bought {format_card(card, mapping['tier'])}"
        elif t == 'buy_reserved':
            card = self.env.reserved[p][mapping['index']]
            return f"{name} bought a reserved card, revealed as {format_card(card)}"
        elif t == 'reserve_face_up':
            card = self.env.dealt[mapping['tier']][mapping['slot']]
            return f"{name} reserved {format_card(card, mapping['tier'])}"
        elif t == 'reserve_face_down':
            return f"{name} reserved a level{mapping['tier']+1} card face down"
        elif t == 'pick_noble':
            noble = self.env.nobles[mapping['index']]
            return f"{name} was visited by a {format_card(noble, is_noble=True)}"
        elif t == 'discard':
            c = f"[{COLOR_INDEX_TO_STR[mapping['index']]}]"
            return f"{name} discarded {c}"
        elif t == 'pass':
            return f"{name} passed"
        return ""

    def _execute_action_idx(self, a_idx):
        if a_idx is None:
            return {'error': 'Invalid action'}
            
        mapping = self.env.action_mapping[a_idx]
        msg = self._generate_log_msg(mapping)
        p = self.env.current_player
        
        self.env.step(a_idx)
        
        if msg:
            if not hasattr(self, 'logs'):
                self.logs = []
            self.logs.append({'pid': p, 'time': time.time(), 'msg': msg})
            
        self.start_turn()
        return {}
        
    def buy(self, card_uuid):
        if self.env.current_phase != "main":
            return {'error': "Not in main phase"}
            
        sig = self.uuid_to_sig.get(card_uuid)
        if not sig:
            return {'error': "Unknown card"}
            
        # Check if it's dealt
        for tier in range(self.env.num_tiers):
            for slot in range(4):
                card = self.env.dealt[tier][slot]
                if card[self.env.card_column_indexer['available']] == 1 and self._card_sig(card) == sig:
                    return self._execute_action_idx(self._get_action_index(
                        lambda m: m['type'] == 'buy_face_up' and m['tier'] == tier and m['slot'] == slot
                    ))
                    
        # Check if reserved
        p = self.env.current_player
        for index in range(self.env.num_reserved[p]):
            card = self.env.reserved[p][index]
            if card[self.env.card_column_indexer['available']] == 1 and self._card_sig(card) == sig:
                return self._execute_action_idx(self._get_action_index(
                    lambda m: m['type'] == 'buy_reserved' and m['index'] == index
                ))
                
        return {'error': "Card not available"}

    def reserve(self, target):
        if self.env.current_phase != "main":
            return {'error': "Not in main phase"}
        
        # target can be a level string like 'level1' or a UUID
        if target in ('level1', 'level2', 'level3'):
            tier = int(target[-1]) - 1
            return self._execute_action_idx(self._get_action_index(
                lambda m: m['type'] == 'reserve_face_down' and m['tier'] == tier
            ))
            
        sig = self.uuid_to_sig.get(target)
        if not sig:
            return {'error': "Unknown card"}
            
        for tier in range(self.env.num_tiers):
            for slot in range(4):
                card = self.env.dealt[tier][slot]
                if card[self.env.card_column_indexer['available']] == 1 and self._card_sig(card) == sig:
                    return self._execute_action_idx(self._get_action_index(
                        lambda m: m['type'] == 'reserve_face_up' and m['tier'] == tier and m['slot'] == slot
                    ))
                    
        return {'error': "Card not Face Up"}

    def noble_visit(self, noble_uuid):
        if self.env.current_phase != "pick_noble":
            return {'error': "Not in pick noble phase"}
            
        sig = self.uuid_to_noble_sig.get(noble_uuid)
        if not sig:
            return {'error': "Unknown noble"}
            
        for index in range(self.env.num_nobles_available):
            noble = self.env.nobles[index]
            if noble[self.env.nobles_column_indexer['available']] == 1 and self._noble_sig(noble) == sig:
                return self._execute_action_idx(self._get_action_index(
                    lambda m: m['type'] == 'pick_noble' and m['index'] == index
                ))
                
        return {'error': "Noble not available"}
        
    def discard(self, color):
        if self.env.current_phase != "discard":
            return {'error': "Not in discard phase"}
        
        # Use the updated STR_TO_COLOR_INDEX which now contains '*'
        if color not in STR_TO_COLOR_INDEX:
            return {'error': f"Unknown color: {color}"}
            
        c_idx = STR_TO_COLOR_INDEX[color]
        a_idx = self._get_action_index(lambda m: m['type'] == 'discard' and m['index'] == c_idx)
        if a_idx is not None:
            return self._execute_action_idx(a_idx)
        return {'error': "Invalid discard"}

    def take(self, color):
        if self.env.current_phase != "main":
            return {'error': "Not in main phase"}
            
        c_idx = STR_TO_COLOR_INDEX[color]
        if self.env.tokens_remaining[c_idx] <= 0:
            return {'error': "No gems remaining"}
            
        if len(self.pending_take) == 0:
            self.pending_take.append(color)
            return {}
            
        if len(self.pending_take) == 1:
            if self.pending_take[0] == color:
                a_idx = self._get_action_index(lambda m: m['type'] == 'take_2_identical_tokens' and m['index'] == c_idx)
                if a_idx is not None:
                    return self._execute_action_idx(a_idx)
                else:
                    return {'error': "Cannot take 2 of that color"}
            else:
                self.pending_take.append(color)
                # check if take 2 diff is complete (if take 3 is masked out)
                idx1 = STR_TO_COLOR_INDEX[self.pending_take[0]]
                idx2 = STR_TO_COLOR_INDEX[self.pending_take[1]]
                a_idx = self._get_action_index(lambda m: m['type'] == 'take_2_diff_tokens' and set(m['indices']) == {idx1, idx2})
                if a_idx is not None:
                    # Execute take 2 only if take 3 is not an option
                    s, e = self.env._action_indices_map["take_3_diff_tokens"]
                    if np.sum(self.env.action_mask[s:e]) == 0:
                        return self._execute_action_idx(a_idx)
                return {} # waiting for 3rd
                
        if len(self.pending_take) == 2:
            if color in self.pending_take:
                return {'error': "Cannot take 2 of the same color and 1 different"}
                
            self.pending_take.append(color)
            indices = {STR_TO_COLOR_INDEX[c] for c in self.pending_take}
            a_idx = self._get_action_index(lambda m: m['type'] == 'take_3_diff_tokens' and set(m['indices']) == indices)
            if a_idx is not None:
                return self._execute_action_idx(a_idx)
            else:
                self.pending_take.pop()
                return {'error': "Invalid token combination"}
                
        return {'error': 'Unknown state'}

    def next(self):
        # Allow passing the turn if partial tokens were taken but valid
        if len(self.pending_take) == 1:
            c_idx = STR_TO_COLOR_INDEX[self.pending_take[0]]
            a_idx = self._get_action_index(lambda m: m['type'] == 'take_1_token' and m['indices'][0] == c_idx)
            if a_idx is not None:
                return self._execute_action_idx(a_idx)
                
        if len(self.pending_take) == 2:
            idx1 = STR_TO_COLOR_INDEX[self.pending_take[0]]
            idx2 = STR_TO_COLOR_INDEX[self.pending_take[1]]
            a_idx = self._get_action_index(lambda m: m['type'] == 'take_2_diff_tokens' and set(m['indices']) == {idx1, idx2})
            if a_idx is not None:
                return self._execute_action_idx(a_idx)
                
        a_idx = self._get_action_index(lambda m: m['type'] == 'pass')
        if a_idx is not None:
            return self._execute_action_idx(a_idx)
            
        return {'error': 'Cannot skip turn right now'}

