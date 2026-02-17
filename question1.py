import numpy as np
import random

class Easy21Env:
    """Exo 1 : Envirpnement Easy21 
    State space: (dealer_card, player_sum)
    Actions: 0 (Hit), 1 (Stick)
    """
    def __init__(self):
        self.min_val = 1
        self.max_val = 10
    
    def draw_card(self):
        """
        Tire une carte avec une valeur de 1 à 10.
        Noir (prob 2/3) -> value positive.
        rouge (prob 1/3) -> value negative.
        """
        val = random.randint(1, 10) #NB infini de carte donc pas de changement
        color_type = np.random.choice(['noir', 'rouge'], p=[2/3, 1/3])
        
        if color_type == 'black':
            return val
        else:
            return -val

    def init_game(self):
        """
        Initialisation du jeu, dealer tire et joueur aussi retourne sous forme 
        (Dealer , Joueur)
        """
        dealer_card = random.randint(1, 10) # Dealer draws black
        player_card = random.randint(1, 10) # Player draws black
        
        return (dealer_card, player_card)
    
    def step(self, state, action):
        """
        Fait un pas du jeu (tout les étapes pour un tour de jeu)
        Arguments : 
            state: Tuple (dealer_card, player_sum)
            action: 0 for Hit, 1 for Stick
        return :
            next_state: Tuple (dealer_card, player_sum)
            reward: Float 
            terminal: Boolean    #True si fin
        """

        dealer_card, player_sum = state
        #Cas Hit
        if action == 0:  
            card = self.draw_card()
            player_sum += card
            
            # Check for Player Bust
            if player_sum < 1 or player_sum > 21:
                return (dealer_card, player_sum), -1.0, True 
            else:
                return (dealer_card, player_sum), 0.0, False #Partie continue
        
        # --- ACTION: STICK (1) ---
        else: 
            
            dealer_sum = dealer_card
            
            # Dealer tire jusque somme >= 17 ou bust
            while dealer_sum < 17 and dealer_sum > 0: 
                dealer_sum += self.draw_card()
            
            # Determine Winner
            if dealer_sum < 1 or dealer_sum > 21:
                return (dealer_card, player_sum), 1.0, True # Dealer busts, Player wins
            
            if player_sum > dealer_sum:
                return (dealer_card, player_sum), 1.0, True # Win
            elif player_sum < dealer_sum:
                return (dealer_card, player_sum), -1.0, True # Lose
            else:
                return (dealer_card, player_sum), 0.0, True # Draw
            

if __name__ == "__main__":
    env = Easy21Env()
    s = env.init_game()
    print(f"Start: {s}")
    ns, r, term = env.step(s, 0) # Hit
    print(f"Hit result: {ns}, Reward: {r}, Terminal: {term}")