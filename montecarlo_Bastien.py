import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import easy21_Bastien  # Assurez-vous que le fichier easy21.py est dans le même dossier
import pickle

# Constantes
N0 = 100
NUM_EPISODES = 500000  # Nombre élevé pour converger vers la "Vérité Terrain" Q*

class MonteCarloAgent:
    def __init__(self):
        # Q-table: dealer (1-10) -> index 0-9, player (1-21) -> index 0-20, actions (2)
        self.q_table = np.zeros((10, 21, 2))
        self.n_state = np.zeros((10, 21))     # N(s)
        self.n_state_action = np.zeros((10, 21, 2)) # N(s,a)

    def get_action(self, state):
        """Stratégie Epsilon-Greedy dépendante de N0"""
        d_idx, p_idx = state[0]-1, state[1]-1
        # Epsilon_t = N0 / (N0 + N(s_t)) [cite: 39]
        epsilon = N0 / (N0 + self.n_state[d_idx, p_idx])
        
        if np.random.random() < epsilon:
            return np.random.randint(0, 2)
        else:
            # En cas d'égalité, on mélange pour éviter un biais
            values = self.q_table[d_idx, p_idx, :]
            return np.random.choice(np.flatnonzero(values == values.max()))

    def train(self, episodes):
        for _ in range(episodes):
            trajectory = []
            state = easy21_Bastien.init_game()
            terminal = False
            
            # 1. Générer l'épisode complet
            while not terminal:
                action = self.get_action(state)
                # Incrémenter N(s) pour le calcul d'epsilon au prochain pas
                self.n_state[state[0]-1, state[1]-1] += 1
                
                next_state, reward, terminal = easy21_Bastien.step(state, action)
                trajectory.append((state, action))
                state = next_state
            
            # 2. Mise à jour (Update) à la fin de l'épisode
            # Gt est juste la reward finale car gamma=1 [cite: 27]
            gt = reward
            
            for (s, a) in trajectory:
                d_idx, p_idx = s[0]-1, s[1]-1
                
                self.n_state_action[d_idx, p_idx, a] += 1
                # Alpha_t = 1 / N(s, a) [cite: 38]
                alpha = 1.0 / self.n_state_action[d_idx, p_idx, a]
                
                error = gt - self.q_table[d_idx, p_idx, a]
                self.q_table[d_idx, p_idx, a] += alpha * error

def plot_value_function(q_table):
    """Trace V*(s) = max_a Q*(s,a) en 3D [cite: 42]"""
    v_star = np.max(q_table, axis=2)
    
    dealer_range = np.arange(1, 11)
    player_range = np.arange(1, 22)
    X, Y = np.meshgrid(dealer_range, player_range)
    
    # Attention: Matplotlib attend X, Y et Z de la même forme (transpose nécessaire selon l'indexation)
    Z = v_star.T  # Transpose pour aligner (21, 10) avec le meshgrid
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax.set_zlabel('Value V*')
    ax.set_title('Optimal Value Function (Monte Carlo)')
    fig.colorbar(surf)
    plt.show()

if __name__ == "__main__":
    print(f"Entraînement Monte-Carlo sur {NUM_EPISODES} épisodes...")
    agent = MonteCarloAgent()
    agent.train(NUM_EPISODES)
    
    # Sauvegarde de Q* pour l'exercice 3 (calcul MSE)
    with open('q_star.pkl', 'wb') as f:
        pickle.dump(agent.q_table, f)
    print("Q* sauvegardé dans 'q_star.pkl'.")
    
    plot_value_function(agent.q_table)