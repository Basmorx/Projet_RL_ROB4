import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import easy21
import pickle
import copy

EPISODES = 100000  # Nombre d'épisodes pour l'entraînement de Q-Learning et Sarsa


# --- 1. Agent Q-Learning ---
class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((10, 21, 2))
        self.n_state = np.zeros((10, 21))
        self.n_state_action = np.zeros((10, 21, 2)) 
        self.N0 = 100

    def get_action(self, state):
        """Stratégie Epsilon-Greedy dépendante de N0"""
        d_idx, p_idx = state[0]-1, state[1]-1
        epsilon = self.N0 / (self.N0 + self.n_state[d_idx, p_idx])
        
        if np.random.random() < epsilon:
            return np.random.randint(0, 2)
        else:
            values = self.q_table[d_idx, p_idx, :]
            return np.random.choice(np.flatnonzero(values == values.max()))

    def train_episode(self):
        state = easy21.init_game()
        terminal = False
        
        while not terminal:
            d_idx, p_idx = state[0]-1, state[1]-1
            
            # Sélection de l'action selon la politique (comportement)
            action = self.get_action(state)
            
            # Incrémentation des compteurs de visite
            self.n_state[d_idx, p_idx] += 1
            self.n_state_action[d_idx, p_idx, action] += 1
            
            # Exécution de l'action
            next_state, reward, terminal = easy21.step(state, action)
            
            # Cible Q-Learning (mise à jour hors-politique sur max a')
            if terminal:
                target = reward
            else:
                n_d_idx, n_p_idx = next_state[0]-1, next_state[1]-1
                target = reward + np.max(self.q_table[n_d_idx, n_p_idx, :])
                
            # Pas d'apprentissage alpha variable
            alpha = 1.0 / self.n_state_action[d_idx, p_idx, action]
            
            # Mise à jour de Q
            self.q_table[d_idx, p_idx, action] += alpha * (target - self.q_table[d_idx, p_idx, action])
            
            state = next_state

# --- Fonctions utilitaires pour la comparaison (Sarsa) ---
# (Intégrées ici pour générer le graphique commun demandé sur 10000 épisodes)
class TabularSarsaAgent:
    def __init__(self, lmbda):
        self.lmbda = lmbda
        self.q_table = np.zeros((10, 21, 2))
        self.n_state = np.zeros((10, 21))
        self.n_state_action = np.zeros((10, 21, 2))
        self.e_trace = np.zeros((10, 21, 2))
        self.N0 = 100

    def get_action(self, state):
        d_idx, p_idx = state[0]-1, state[1]-1
        epsilon = self.N0 / (self.N0 + self.n_state[d_idx, p_idx])
        if np.random.random() < epsilon: return np.random.randint(0, 2)
        else:
            values = self.q_table[d_idx, p_idx, :]
            return np.random.choice(np.flatnonzero(values == values.max()))

    def train_episode(self):
        state = easy21.init_game()
        action = self.get_action(state)
        self.e_trace[:] = 0.0 
        terminal = False
        
        while not terminal:
            d_idx, p_idx = state[0]-1, state[1]-1
            self.n_state[d_idx, p_idx] += 1
            self.n_state_action[d_idx, p_idx, action] += 1
            
            next_state, reward, terminal = easy21.step(state, action)
            
            if not terminal:
                next_action = self.get_action(next_state)
                q_next = self.q_table[next_state[0]-1, next_state[1]-1, next_action]
            else:
                next_action = 0; q_next = 0.0
                
            q_current = self.q_table[d_idx, p_idx, action]
            delta = reward + q_next - q_current
            self.e_trace[d_idx, p_idx, action] += 1
            alpha = 1.0 / self.n_state_action[d_idx, p_idx, action]
            
            self.q_table += alpha * delta * self.e_trace
            self.e_trace *= self.lmbda
            
            state, action = next_state, next_action

def compute_mse(q_pred, q_star):
    return np.sum((q_pred - q_star)**2)

def plot_value_function(q_table, title):
    v_star = np.max(q_table, axis=2)
    dealer_range = np.arange(1, 11)
    player_range = np.arange(1, 22)
    X, Y = np.meshgrid(dealer_range, player_range)
    Z = v_star.T 
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax.set_zlabel('Value V*')
    ax.set_title(title)
    fig.colorbar(surf)
    plt.savefig('qlearning_3d.png')
    plt.show()

if __name__ == "__main__":
    try:
        with open('q_star.pkl', 'rb') as f:
            q_star = pickle.load(f)
    except FileNotFoundError:
        print("Erreur: q_star.pkl introuvable.")
        exit()

    
    print(f"Entraînement des agents sur {EPISODES} épisodes...")
    
    # Entraînement Q-Learning
    ql_agent = QLearningAgent()
    ql_mse_history = []
    for _ in range(EPISODES):
        ql_agent.train_episode()
        ql_mse_history.append(compute_mse(ql_agent.q_table, q_star))
    print(f"Q-Learning MSE finale: {ql_mse_history[-1]:.4f}")

    # Entraînement Sarsa(0)
    sarsa_0_agent = TabularSarsaAgent(0.0)
    sarsa_0_mse_history = []
    for _ in range(EPISODES):
        sarsa_0_agent.train_episode()
        sarsa_0_mse_history.append(compute_mse(sarsa_0_agent.q_table, q_star))
    print(f"Sarsa(0) MSE finale: {sarsa_0_mse_history[-1]:.4f}")

    # Entraînement Sarsa(1)
    sarsa_1_agent = TabularSarsaAgent(1.0)
    sarsa_1_mse_history = []
    for _ in range(EPISODES):
        sarsa_1_agent.train_episode()
        sarsa_1_mse_history.append(compute_mse(sarsa_1_agent.q_table, q_star))
    print(f"Sarsa(1) MSE finale: {sarsa_1_mse_history[-1]:.4f}")

    # --- Plot des comparaisons (Q-Learning vs Sarsa) ---
    plt.figure(figsize=(10, 5))
    plt.plot(ql_mse_history, label='Q-Learning', alpha=0.8)
    plt.plot(sarsa_0_mse_history, label='Sarsa(lambda=0)', alpha=0.8)
    plt.plot(sarsa_1_mse_history, label='Sarsa(lambda=1) / Monte Carlo', alpha=0.8)
    plt.title('Comparaison des MSE sur 10000 épisodes (Tabulaire)')
    plt.xlabel("Numéro d'épisode")
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_ql_sarsa.png')
    plt.show()

    # --- Plot 3D de la fonction de valeur optimale de Q-Learning ---
    plot_value_function(ql_agent.q_table, 'Optimal Value Function (Q-Learning)')