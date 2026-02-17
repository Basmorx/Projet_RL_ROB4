

import numpy as np

class MonteCarloAgent:
    def __init__(self, n_states, n_actions, gamma=0.99, n0=100):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.n0 = n0  # La constante N0 (ajustable)
        
        # Initialisation de la Q-table avec des zéros
        self.q_table = np.zeros((n_states, n_actions))
        
        # Initialisation de la table de comptage N(s, a)
        self.n_count = np.zeros((n_states, n_actions))

    def get_epsilon(self, state):
        """
        Calcule epsilon selon la formule : N0 / (N0 + N(s))
        N(s) est la somme des visites de toutes les actions dans cet état.
        """
        n_s = np.sum(self.n_count[state, :]) # N(s)
        epsilon = self.n0 / (self.n0 + n_s)
        return epsilon

    def choose_action(self, state):
        """
        Stratégie epsilon-greedy avec epsilon variable
        """
        epsilon = self.get_epsilon(state)
        
        # Exploration : on tire un nombre aléatoire
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        
        # Exploitation : on prend la meilleure action (argmax)
        # Note : on utilise une petite astuce pour casser les égalités aléatoirement
        # au lieu de toujours prendre le premier index en cas d'égalité.
        max_q = np.max(self.q_table[state, :])
        actions_with_max_q = np.where(self.q_table[state, :] == max_q)[0]
        return np.random.choice(actions_with_max_q)

    def update(self, state, action, reward, next_state):
        """
        Mise à jour avec pas d'apprentissage alpha variable.
        alpha = 1 / N(s, a)
        """
        # 1. Incrémenter le compteur N(s, a) AVANT de calculer alpha
        self.n_count[state, action] += 1
        
        # 2. Calculer le pas d'apprentissage dynamique alpha
        # Comme on a incrémenté juste avant, n_count est au minimum 1, donc pas de division par 0.
        alpha = 1.0 / self.n_count[state, action]
        
        # 3. Calculer la cible (Target) - Exemple pour Q-Learning
        best_next_action = np.max(self.q_table[next_state, :])
        td_target = reward + self.gamma * best_next_action
        
        # 4. Mise à jour de la Q-valeur
        current_q = self.q_table[state, action]
        td_error = td_target - current_q
        
        self.q_table[state, action] = current_q + alpha * td_error