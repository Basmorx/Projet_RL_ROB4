import random

def draw_card():
    """Tire une carte: 1-10. Rouge (neg) prob 1/3, Noir (pos) prob 2/3."""
    value = random.randint(1, 10)
    if random.random() < 1/3:
        return -value  # Rouge
    else:
        return value   # Noir

def step(state, action):
    """
    Exécute un pas dans l'environnement Easy21.
    state: tuple (dealer_card, player_sum)
    action: 0 (stick) ou 1 (hit)
    Retourne: next_state, reward, terminal
    """
    dealer_card, player_sum = state
    
    if action == 1:  # Hit
        player_sum += draw_card()
        if player_sum > 21 or player_sum < 1:
            return (dealer_card, player_sum), -1, True # Perdu
        else:
            return (dealer_card, player_sum), 0, False # Continue
            
    else:  # Stick
        while dealer_card < 17 and dealer_card >= 1:
            dealer_card += draw_card()
            
        if dealer_card > 21 or dealer_card < 1: # Dealer bust
            return (dealer_card, player_sum), 1, True
        
        if player_sum > dealer_card:
            return (dealer_card, player_sum), 1, True
        elif player_sum < dealer_card:
            return (dealer_card, player_sum), -1, True
        else:
            return (dealer_card, player_sum), 0, True # Egalité

def init_game():
    return (random.randint(1, 10), random.randint(1, 10))