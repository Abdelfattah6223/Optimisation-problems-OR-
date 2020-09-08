## Importation des libraries nécessaires

import numpy as np
import math
import copy
from itertools import permutations
import matplotlib.pyplot as plt
import pandas as pd
import time

## Importation de la matrice des distances

# INSTANCE_10/20
inst = pd.read_excel('data/INSTANCE_10.xlsx')
villes = inst.columns.tolist()[1:]
distance_matrix = inst[villes].values

## Fonctions de base

def Fonction_Mvt_Candidats(RT_solution, nmbr_noeuds):

    '''
       Entrée (input) : une solution + nombre de villes (noeuds).
       Sortie (output) : une liste des mouvements candidats possibles pour la solution (actuelle) donnée.
    '''
    temp_solution = copy.copy(RT_solution)
    liste_mvt_candidats = []

    index_iter = nmbr_noeuds - 1
    index_step = 1

    for i in range(nmbr_noeuds - 2):
        for j in range(1, index_iter):
            temp_mvt_tableau = np.zeros(3)
            temp_mvt_tableau[0] = temp_solution[j]
            temp_mvt_tableau[1] = temp_solution[j+index_step]

            # Échanger le candidat
            temp2_solution = copy.copy(RT_solution)
            temp2_solution[j] = temp_mvt_tableau[1]
            temp2_solution[j+index_step] = temp_mvt_tableau[0]

            # Calculer la distance totale (de la solution associée au mouvement candidat)
            for k in range(nmbr_noeuds - 1):
                temp_mvt_tableau[2] += distance_matrix[temp2_solution[k], temp2_solution[k+1]]

            temp_mvt_tableau[2] += distance_matrix[temp2_solution[nmbr_noeuds - 1], temp2_solution[0]]
            liste_mvt_candidats.append(temp_mvt_tableau)


        index_step += 1
        index_iter -= 1


    liste_mvt_candidats = np.array(liste_mvt_candidats)
    return liste_mvt_candidats

def Mettre_Jour_RT_solution_Fonction(RT_solution, index_0, index_1, nmbr_noeuds):

    '''
        Input : une solution, les indices des noeuds qu'on veut permuter et le nombre de noeuds.
        Output : Cette fonction génére un mouvement en faisant une permutation entre deux villes
        Remarque : cette fonction vient après avoir la liste des mouvements candidats, donc après avoir trier la liste
                   en se basant sur la valeur de la distance, on choisit le mouvement le plus avantageux et on appelle
                   cette fonction
    '''

    temp_solution = copy.copy(RT_solution)

    for i in range(1, nmbr_noeuds):
        if (temp_solution[i] == index_0):
            temp_solution[i] = index_1
        elif(temp_solution[i] == index_1):
            temp_solution[i] = index_0

    return temp_solution


## Initialiser les paramètres de l'algorithme de la recherche taboue

nmbr_noeuds = 16
nbr_iteration = 1000
critere_aspiration = 300
tabu_tenure = 10

# Start time
t1 = time.perf_counter()

## Programme principal
Meilleure_Route_Solution = []   # la mémoire s*
Meilleure_Shortest_Distance_Valeur = []  # la mémoire z*

# Initialiser une solution pour l'algrithme de RT

RT_solution = np.arange(1, nmbr_noeuds)
np.random.shuffle(RT_solution)
RT_solution = list(RT_solution)
RT_solution.insert(0, 0)

RT_solution = np.array(RT_solution)

# Calculer l'initialisation de la distance la plus courte à partir de la solution de recherche Tabu
RT_Shortest_Distance = 0
for i in range(nmbr_noeuds - 1):
    RT_Shortest_Distance += distance_matrix[RT_solution[i], RT_solution[i+1]]

RT_Shortest_Distance += distance_matrix [RT_solution[nmbr_noeuds-1], RT_solution[0]]

Meilleure_Route_Solution.append(RT_solution)
Meilleure_Shortest_Distance_Valeur.append(RT_Shortest_Distance)

# Initialiser le liste Tabu (matrice carrée au nombre des noeuds)
tabu_list = np.zeros((nmbr_noeuds, nmbr_noeuds))

## l'Algorithme de RT
for i in range(nbr_iteration):
    Nv_liste_mvt_candidats = []
    Nv_liste_mvt_candidats = Fonction_Mvt_Candidats(RT_solution, nmbr_noeuds)
# Trier la liste de mouvements candidats en fonction de la distance la plus courte
    tableau_mvt_candidats =  copy.copy(Nv_liste_mvt_candidats)
    tableau_mvt_candidats = tableau_mvt_candidats[np.argsort(tableau_mvt_candidats[:, 2])] # <--pour choisir le voisin le plus avantageux

    length_tableau_mvt_candidats = len(tableau_mvt_candidats)

    ## Pénaliser la valeur de fonction objective par fréquence en utilisant la diagonale inférieure (Diversification)

    for j in range(length_tableau_mvt_candidats):
        if(tableau_mvt_candidats[j, 0] < tableau_mvt_candidats[j, 1]):
            index_row = tableau_mvt_candidats[j, 1]
            index_col = tableau_mvt_candidats[j, 0]
        else:
            index_row = tableau_mvt_candidats[j, 0]
            index_col = tableau_mvt_candidats[j, 1]

        index_row = int(index_row)
        index_col = int(index_col)

        tableau_mvt_candidats[j, 2] += tabu_list[index_row, index_col]

    # Trier la liste de mouvements candidats en fonction de la distance la plus courte après l'avoir pénalisée
    tableau_mvt_candidats = copy.copy(Nv_liste_mvt_candidats)
    tableau_mvt_candidats = tableau_mvt_candidats[np.argsort(tableau_mvt_candidats[:, 2])]

    # Choisir le mouvement de tableau_mvt_candidats
    for j in range(length_tableau_mvt_candidats):
        if(tableau_mvt_candidats[j, 0] > tableau_mvt_candidats[j, 1]):
            index_row = tableau_mvt_candidats[j, 1]
            index_col = tableau_mvt_candidats[j, 0]
        else:
            index_row = tableau_mvt_candidats[j, 0]
            index_col = tableau_mvt_candidats[j, 1]

        index_row = int(index_row)
        index_col = int(index_col)
    # Vérifier si le mouvement est tabou ou non en utilisant la diagonale supérieure
        if(tabu_list[index_row, index_col] == 0):
            RT_Shortest_Distance = tableau_mvt_candidats[j, 2]

            # Mettre à jour la route (solution)
            Nv_RT_solution = []
            Nv_RT_solution = Mettre_Jour_RT_solution_Fonction(RT_solution, tableau_mvt_candidats[j, 0], tableau_mvt_candidats[j, 1], nmbr_noeuds)

            RT_solution = copy.copy(Nv_RT_solution)
            ## Remplissage de tabu_list
            # Ajouter le tenure dans la diagonale sup
            tabu_list[index_row, index_col] = tabu_tenure

            # Ajouter la fréquence dans la diagonale inf
            tabu_list[index_col, index_row] += 1

            break
        ## Critére d'aspiration ( si la solution vérifie ce critére elle peut sortir de tabu_list avant son tabou tenure)
        elif(tabu_list[index_row, index_col] > 0):
            # Vérifier si elle répond au critère d'aspiration
            temp_diff = RT_Shortest_Distance - tableau_mvt_candidats[j, 2]

            if (temp_diff > critere_aspiration):
                RT_Shortest_Distance = tableau_mvt_candidats[j, 2]

                Nv_RT_solution = []
                Nv_RT_solution = Mettre_Jour_RT_solution_Fonction(RT_solution, tableau_mvt_candidats[j, 0], tableau_mvt_candidats[j, 1], nmbr_noeuds)

                RT_solution = copy.copy(Nv_RT_solution)

                # Réinitialiser le tenure dans la diagonale sup
                tabu_list[index_row, index_col] = tabu_tenure

                # Réinitialiser la fréquence dans la diagonale inf
                tabu_list[index_col, index_row] += 1

                break

    ## Mettre à jour le Tabu Tenure dans la liste (ICI à chaque iteration le tenure tabu décroit jusqu'à ce qu'il devient nul et donc la solution correspondante sort de tabu_list)
    for j in range(nmbr_noeuds):
        for k in range(j+1, nmbr_noeuds):
            if (tabu_list[j, k] > 0):
                tabu_list[j, k] -= 1

    if(Meilleure_Shortest_Distance_Valeur[i] < RT_Shortest_Distance):
        Meilleure_Route_Solution.append(Meilleure_Route_Solution[i])
        Meilleure_Shortest_Distance_Valeur.append(Meilleure_Shortest_Distance_Valeur[i])
    else:
        Meilleure_Route_Solution.append(RT_solution)
        Meilleure_Shortest_Distance_Valeur.append(RT_Shortest_Distance)

# Stop time
t2 = time.perf_counter()

# Tableau de Tabu tenure et les pénalités de fréquence
print(tabu_list)
print("\n=====================================================================================")
print ("La solution (Route) à l'aide de l'algorithme de recherche Tabu : ")
RT_solution = Meilleure_Route_Solution[-1]
RT_solution = list(RT_solution) + [0]
RT_solution = np.array(RT_solution)

RT_solution_villes = [villes[i] for i in RT_solution]

print (RT_solution)
print(" -> ".join(RT_solution_villes)+"\n")
print ("La distance parcourue correspondante : ")
RT_Shortest_Distance = Meilleure_Shortest_Distance_Valeur[-1]
print (RT_Shortest_Distance)

print ("\n")

print("CPU Time (s):", (t2-t1))


plt.plot(Meilleure_Shortest_Distance_Valeur,color='red')
plt.ylabel('Valeur de la fonction objective')
plt.xlabel('Iteration')
plt.savefig('Evolution_RT.png')
plt.show()
