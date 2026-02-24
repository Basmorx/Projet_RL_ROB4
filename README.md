# Projet_RL_ROB4

MUZATON Mattéo
MOREAUX Bastien


## Exercice 2 - MonteCarlo

Il s'agit d'une méthode pour trouver la policy optimale à partir de la moyenne de tous les gains cumulés obtenus à partir d'un état précis.

```bash
python3 montecarlo.py
```

## Exercice 3 - Sarsa Algorithm

L'algorithme de Sarsa (State - Action - Reward - State - Action) :

1. Observer l'état s
2. Choisir une action a à executer en fonction du reward qu'on connait déjà avec une probabilité de (1 - Ԑ) ou une action aléatoire avec une probabilité Ԑ où Ԑ est le biais d'exploration.
3. Obtention d'un reward R
4. Observe le nouvel état s'
5. Choisir une nouvelle action a' à executer

Ensuite on met à jour la politique Q(s, a) := Q(s, a) + ɑ[ r + γ Q(s', a') - Q(s, a)]

ɑ est le taux d'apprentissage

Il s'agit d'une méthode pour trouver la policy optimale à partir de la moyenne de tous les gains cumulés obtenus à partir d'un état précis.

```bash
python3 sarsa.py
```