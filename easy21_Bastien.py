import random


def step(s , a):
    # s: (dealer_card, player_sum)
    # a: 0 (stick) or 1 (hit)
    dealer_card, player_sum = s
    if a == 0:  # stick
        while dealer_card < 17:
            dealer_card += draw_card()
        if dealer_card > 21 or dealer_card < player_sum:
            return (dealer_card, player_sum), 1  # win
        elif dealer_card == player_sum:
            return (dealer_card, player_sum), 0  # draw
        else:
            return (dealer_card, player_sum), -1  # lose
    else:  # hit
        new_card = draw_card()
        player_sum += new_card
        if player_sum > 21:
            return (dealer_card, player_sum), -1  # lose
        else:
            return (dealer_card, player_sum), None  # continue
        


def draw_card():
    card = random.randint(1, 10)
    if random.random() < 1/3:
        return -card  # black card
    else:
        return card  # red card
    


def init_game():
    dealer_card = random.randint(1, 10)
    player_sum = random.randint(1, 10)
    return (dealer_card, player_sum)

