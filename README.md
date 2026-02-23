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


## Exercice 4 - Discussion

### 1. Avantages et inconvenients du bootstrapping dans easy21

#### Avantages :

1. Plus d'efficacité dans l'apprentissage : le modèle n'a pas besoin de jouer la partie jusqu'au bout pour savoir que certains résultats sont meilleurs que d'autre.

2. Moins de données bruité : le jeu peut se terminer grâce à un coup de chance, le bootstrapping permettrait de moins se reposer sur la chance.

3. Plus rapide

#### Inconvénients :

1. Plus complexe à implémenter qu'un modèle simple.

2. Beaucoup de dépendance aux paramètres initiaux.


### 2. Bootstrapping plus utile en BlackJack ou dans Easy21 ?

Le bootstrapping est plus utile dans le jeu easy21 car les parties sont plus longues à cause des cartes négatives, ce qui rend l'estimation de la value des états plus facile.


### 3. Avantages et inconvenients de l'approximation de fonction dans easy21

#### Avantages :

1. L'agent peut généraliser des états qu'il n'a jamais rencontré.

#### Iconvénients :

1. La méthode avec approximation peut converger sur une solution non optimale si le taux d'apprentissage est trop élevé.

2. Il peut y avoir des décision imprécises près des limites comme 21 ou 0 à cause des cassures dans la value fonction optimale.

### 4. Quels changements faire dans l'approximateur de fonction ?

1. On peut faire en sorte de faire décroitre le taux d'apprentissage pour garantir la convergence de la maximum value fonction vers la maximum value fonction réelle.

2. Utiliser des noyaux gaussiens pour représenter l'espace, ce qui permettrait à l'agent de mieux capturer la continuité de la somme du joueur tout en restant précis sur les zones critiques (autour de 21).