import numpy as np
import matplotlib.pyplot as plt
import easy21_Bastien as easy21
import pickle

# Paramètres
N0 = 100
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # [cite: 46]
EPISODES_PER_LAMBDA = 1000  # [cite: 46]

class SarsaAgent:
    def __init__(self, lmbda):
        self.lmbda = lmbda
        self.q_table = np.zeros((10, 21, 2))
        self.n_state = np.zeros((10, 21))
        self.n_state_action = np.zeros((10, 21, 2))
        # Trace d'éligibilité
        self.e_trace = np.zeros((10, 21, 2))

    def get_action(self, state):
        d_idx, p_idx = state[0]-1, state[1]-1
        epsilon = N0 / (N0 + self.n_state[d_idx, p_idx])
        
        if np.random.random() < epsilon:
            return np.random.randint(0, 2)
        else:
            values = self.q_table[d_idx, p_idx, :]
            return np.random.choice(np.flatnonzero(values == values.max()))

    def train_episode(self):
        """Joue un seul épisode et retourne l'erreur quadratique (si Q* fourni)"""
        state = easy21.init_game()
        action = self.get_action(state)
        self.e_trace[:] = 0.0 # Reset traces au début de l'épisode
        
        terminal = False
        while not terminal:
            d_idx, p_idx = state[0]-1, state[1]-1
            
            # Incrément compteurs
            self.n_state[d_idx, p_idx] += 1
            self.n_state_action[d_idx, p_idx, action] += 1
            
            # Exécution action
            next_state, reward, terminal = easy21.step(state, action)
            
            # Choix action suivante (s', a')
            if not terminal:
                next_action = self.get_action(next_state)
                q_next = self.q_table[next_state[0]-1, next_state[1]-1, next_action]
            else:
                next_action = 0
                q_next = 0.0
            
            q_current = self.q_table[d_idx, p_idx, action]
            
            # Erreur TD
            delta = reward + q_next - q_current
            
            # Mise à jour trace (Accumulating traces)
            self.e_trace[d_idx, p_idx, action] += 1
            
            # Alpha variable [cite: 38, 45]
            alpha = 1.0 / self.n_state_action[d_idx, p_idx, action]
            
            # Mise à jour de TOUS les états via la trace
            # Q(s,a) <- Q(s,a) + alpha * delta * E(s,a)
            # Note: On utilise l'alpha de l'état visité pour l'update global (standard approx)
            self.q_table += alpha * delta * self.e_trace
            
            # Décroissance trace
            self.e_trace *= self.lmbda # gamma = 1, donc juste lambda
            
            state = next_state
            action = next_action

def compute_mse(q_pred, q_star):
    """Calcule MSE sur tous les états et actions [cite: 48]"""
    # MSE = sum((Q - Q*)^2)
    return np.sum((q_pred - q_star)**2)

if __name__ == "__main__":
    # Charger la vérité terrain Q* (générée par question2.py)
    try:
        with open('q_star.pkl', 'rb') as f:
            q_star = pickle.load(f)
    except FileNotFoundError:
        print("Erreur: Exécutez question2.py d'abord pour générer q_star.pkl")
        exit()

    final_mses = []
    learning_curves = {} # Pour stocker lambda=0 et lambda=1

    print("Début de l'entraînement Sarsa(lambda)...")

    for lmbda in LAMBDAS:
        agent = SarsaAgent(lmbda)
        mse_history = []
        
        for ep in range(EPISODES_PER_LAMBDA):
            agent.train_episode()
            # Calcul MSE après chaque épisode
            mse = compute_mse(agent.q_table, q_star)
            mse_history.append(mse)
        
        final_mses.append(mse_history[-1])
        print(f"Lambda {lmbda}: MSE finale = {mse_history[-1]:.4f}")
        
        if lmbda == 0.0 or lmbda == 1.0:
            learning_curves[lmbda] = mse_history

    # --- Plot 1: MSE vs Lambda [cite: 50] ---
    plt.figure(figsize=(10, 5))
    plt.plot(LAMBDAS, final_mses, marker='o')
    plt.title('MSE vs Lambda (après 1000 épisodes)')
    plt.xlabel('Lambda')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.show()

    # --- Plot 2: Learning Curves (Lambda 0 vs 1) [cite: 51] ---
    plt.figure(figsize=(10, 5))
    plt.plot(learning_curves[0.0], label='Lambda = 0 (TD)')
    plt.plot(learning_curves[1.0], label='Lambda = 1 (Monte Carlo)')
    plt.title('Courbe d\'apprentissage (MSE vs Episodes)')
    plt.xlabel('Numéro d\'épisode')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()