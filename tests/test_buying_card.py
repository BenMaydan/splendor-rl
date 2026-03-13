import numpy as np
from numpy.typing import NDArray


color_indices = [1,2,]
colors = ['Red', 'Green']
gold_index = len(colors)


def token_cost(tokens_in_hand, discounts, card) -> NDArray[np.uint8]:
    """
    Determines the token cost (and if gold tokens are necessary) to buy a card
    Return inf if player doesn't have enough tokens (including gold tokens)
    """
    # tokens_available: (*colors, gold_index)
    # card: (*colors)
    raw_costs = card[..., color_indices]
    deficit_per_color = np.maximum(0, raw_costs - discounts - tokens_in_hand[:len(colors)])
    gold_needed = np.sum(deficit_per_color, axis=-1)
    if gold_needed > tokens_in_hand[gold_index]:
        return None
    actual_gem_cost = np.minimum(raw_costs, tokens_in_hand[:len(colors)])
    return np.append(actual_gem_cost, gold_needed).astype(np.uint8)


def get_purchasibility_map(tokens_in_hand, discounts, cards) -> NDArray[np.uint8]:
    """
    Determines the token cost (and if gold tokens are necessary) to buy a card
    Return inf if player doesn't have enough tokens (including gold tokens)
    """
    # tokens_available: (*colors, gold_index)
    # card: (*colors)
    raw_costs = cards[..., color_indices]
    deficit_per_color = np.maximum(0, raw_costs - discounts - tokens_in_hand[:len(colors)])
    gold_needed = np.sum(deficit_per_color, axis=-1)
    return (tokens_in_hand[gold_index] >= gold_needed).flatten()


if __name__ == "__main__":
    card = np.array([1,2,2])
    cards = np.array([[[0,1,1],[1,2,2]],[[1,3,3],[1,4,4]],[[1,5,5],[1,0,1]]])
    reserved = np.array([[0,1,1],[1,2,3]])
    discounts = np.zeros((2,), dtype=np.uint8)
    tokens = np.array([1,2,1])

    print(token_cost(tokens, discounts, card))
    print(get_purchasibility_map(tokens, discounts, cards))
    print(get_purchasibility_map(tokens, discounts, reserved))