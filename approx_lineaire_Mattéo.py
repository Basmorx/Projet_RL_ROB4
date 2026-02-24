import numpy as np
import matplotlib.pyplot as plt
import easy21
import pickle

# --- 1. Encodage par codage grossier ---
def feature_vector(state, action):
    """
    Construit le vecteur de caractéristiques binaires phi(s, a).
    Retourne un numpy array de dimension 36.
    """
    dealer_card, player_sum = state
    
    # Initialisation du tenseur 3x6x2
    phi = np.zeros((3, 6, 2))
    
    # Définition des intervalles (inclusifs)
    dealer_intervals = [(1, 4), (4, 7), (7, 10)]
    player_intervals = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]
    
    # Remplissage du vecteur avec chevauchements
    for i, (d_min, d_max) in enumerate(dealer_intervals):
        if d_min <= dealer_card <= d_max:
            for j, (p_min, p_max) in enumerate(player_intervals):
                if p_min <= player_sum <= p_max:
                    phi[i, j, action] = 1.0
                    
    # Aplatissement en un vecteur 1D de dimension 36
    return phi.flatten()

# --- 2. Sarsa(lambda) avec approximation linéaire ---
class SarsaApproxAgent:
    def __init__(self, lmbda):
        self.lmbda = lmbda
        self.epsilon = 0.05  # Exploration constante
        self.alpha = 0.01    # Pas d'apprentissage constant
        self.theta = np.zeros(36) # Paramètres à apprendre
        self.e_trace = np.zeros(36)

    def get_q_value(self, state, action):
        """Calcule Q(s,a) = phi(s,a)^T * theta"""
        phi = feature_vector(state, action)
        return np.dot(phi, self.theta)

    def get_action(self, state):
        """Stratégie epsilon-greedy constante"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 2)
        else:
            q_hit = self.get_q_value(state, 0)
            q_stick = self.get_q_value(state, 1)
            
            if q_hit > q_stick:
                return 0
            elif q_stick > q_hit:
                return 1
            else:
                return np.random.choice([0, 1])

    def train_episode(self):
        state = easy21.init_game()
        action = self.get_action(state)
        self.e_trace[:] = 0.0 # Réinitialisation des traces au début de l'épisode
        
        terminal = False
        while not terminal:
            phi = feature_vector(state, action)
            
            next_state, reward, terminal = easy21.step(state, action)
            
            if not terminal:
                next_action = self.get_action(next_state)
                q_next = self.get_q_value(next_state, next_action)
            else:
                next_action = 0
                q_next = 0.0
                
            q_current = np.dot(phi, self.theta)
            
            # Erreur TD
            delta = reward + q_next - q_current
            
            # Mise à jour des traces (accumulées avec gamma=1)
            self.e_trace = self.lmbda * self.e_trace + phi
            
            # Mise à jour des poids theta
            self.theta += self.alpha * delta * self.e_trace
            
            state = next_state
            action = next_action

# --- 3. Analyse des résultats ---
def get_full_q_table(agent):
    """Reconstruit la table Q complète (10x21x2) à partir des poids theta"""
    q_table = np.zeros((10, 21, 2))
    for d in range(1, 11):
        for p in range(1, 22):
            for a in range(2):
                q_table[d-1, p-1, a] = agent.get_q_value((d, p), a)
    return q_table

def compute_mse(q_pred, q_star):
    """Calcule la MSE entre la table approximée et Q*"""
    return np.sum((q_pred - q_star)**2)

if __name__ == "__main__":
    # Chargement de la vérité terrain Q* du TP précédent
    try:
        with open('q_star.pkl', 'rb') as f:
            q_star = pickle.load(f)
    except FileNotFoundError:
        print("Erreur: q_star.pkl introuvable. Exécutez montecarlo.py d'abord.")
        exit()

    LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    EPISODES = 1000
    
    final_mses = []
    learning_curves = {}

    print("Début de l'entraînement Sarsa(lambda) avec approximation linéaire...")

    for lmbda in LAMBDAS:
        agent = SarsaApproxAgent(lmbda)
        mse_history = []
        
        for ep in range(EPISODES):
            agent.train_episode()
            q_pred = get_full_q_table(agent)
            mse = compute_mse(q_pred, q_star)
            mse_history.append(mse)
            
        final_mse = mse_history[-1]
        final_mses.append(final_mse)
        print(f"Lambda {lmbda}: MSE finale = {final_mse:.4f}")
        
        if lmbda == 0.0 or lmbda == 1.0:
            learning_curves[lmbda] = mse_history

    # --- Tracé 1: MSE vs Lambda ---
    plt.figure(figsize=(10, 5))
    plt.plot(LAMBDAS, final_mses, marker='o', color='purple')
    plt.title('MSE vs Lambda (Approx Linéaire, 1000 épisodes)')
    plt.xlabel('Lambda')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.savefig('mse_vs_lambda_approx.png')
    plt.show()

    # --- Tracé 2: Courbes d'apprentissage (Lambda 0 vs 1) ---
    plt.figure(figsize=(10, 5))
    plt.plot(learning_curves[0.0], label='Lambda = 0 (TD)')
    plt.plot(learning_curves[1.0], label='Lambda = 1 (Monte Carlo)')
    plt.title('Courbe d\'apprentissage Approx Linéaire (MSE vs Episodes)')
    plt.xlabel('Numéro d\'épisode')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curves_approx.png')
    plt.show()