import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


np.random.seed(20)

def check_type(variable):
    """
    Affiche le type de la variable donnée en argument
    """
    if variable is None:
        print("La variable est de type: None")
    else:
        print("La variable est de type:", type(variable).__name__)

class Node:
    """
        Classe représentant la structure de base d'un nœud dans l'arbre de décision
    """
    def __init__(self,attrib = None, seuil = None, nd_gauche = None, nd_droit = None, prediction = None):
        self.attrib = attrib # attribut sur lequel on realise la division à ce noeud
        self.seuil = seuil # la valeur seuil pour la division
        self.nd_gauche = nd_gauche # noeud héritié gauche
        self.nd_droit = nd_droit # noeud héritié droit
        self.prediction = prediction # valeur prédite pour ce noeud

class DecisionTree:
    """
        Classe représentant l'arbre et sa construction
    """
    def __init__(self, ech_min = 2, taille_max = 100, racine = None):
        self.ech_min = ech_min # nombre min d'échantillons nécessaire pour divier
        self.taille_max = taille_max # taille max de l'arbre
        self.racine = racine # racine de l'arbre

    def choix_attribut(self, df, method = "GINI"):
        """
            Choix de l'attribut optimal
        """
        gain_max = 0
        attrib = None
        seuil = None
        heritier_g = df.iloc[0:0]
        heritier_d = df.iloc[0:0]

        # Calcul de l'impureté parent
        impurete_parent = self.calcul_impurete(df, method)

        # parcours tous les attibuts du dataset
        for attribut in range(df.shape[1] - 1): 
            val_unq = df.iloc[:, attribut].unique() 
            # Pour chaque valeur unique
            for val in val_unq:
                
                # divise sur l'attribut actuel
                h_g, h_d = self.separation(df,attribut,val)

                # calcule le gain d'information seulement si on a deux df non vide 
                if isinstance(h_g, pd.DataFrame) and isinstance(h_d, pd.DataFrame):
                    gain = self.calcul_gain(df, h_g, h_d, method)

                    if gain > gain_max: 
                        gain_max = gain
                        attrib = attribut
                        seuil = val
                        heritier_g = h_g
                        heritier_d = h_d

        return attrib, seuil, heritier_g,heritier_d

    def separation (self, df, attribut, seuil):
        """
            Divise le dataset en deux en fonction de l'attribut et du seuil
        """

        # Création de deux listes vides pour les héritiers
        heritier_g = []
        heritier_d = []

        # Remplissage
        heritier_g = df[df.iloc[:, attribut] <= seuil]
        heritier_d = df[df.iloc[:, attribut] > seuil]

        return heritier_g,heritier_d

    def calcul_gain(self, df, heritier_g, heritier_d, method):
        """
            Calcule le gain d'information
        """

        # Poids noeuds
        poids_g = poids_d = 0
        if isinstance(heritier_g, pd.DataFrame):
            poids_g = len(heritier_g)/len(df)
        if isinstance(heritier_d, pd.DataFrame):
            poids_d = len(heritier_d)/len(df)

        # Calcul de l'impureté parent
        impurete_parent = self.calcul_impurete(df, method)

        # Calcul du gain d'information
        if method == "GINI":
            gain = impurete_parent - (poids_g * self.GINI(heritier_g) + poids_d * self.GINI(heritier_d))
        elif method == "ENTROPIE":
            gain = impurete_parent - (poids_g * self.ENTROPIE(heritier_g) + poids_d * self.ENTROPIE(heritier_d))
        else:
            gain = (poids_g * self.CHIdeux(heritier_g) + poids_d * self.CHIdeux(heritier_d))
        return gain

    def calcul_impurete(self, df, method):
        """
            Calcule l'impureté en fonction de la méthode choisie 
        """

        if method == "GINI":
            return self.GINI(df)
        elif method == "ENTROPIE":
            return self.ENTROPIE(df)
        elif method == "CHIdeux":
            return self.CHIdeux(df)
        else:
            print('/!\ ERREUR /!\ ')
            print(" Saisie de méthode incorecte : ")
            print("1\ ... GINI ... ")
            print("2\ ... ENTROPIE ... ")
            print("3\ ... CHIdeux ... ")
            print("Merci de respecter l'orthographe et les majuscules. Merci")
            print("Méthode par défaut : GINI. Merci")
            method = "GINI"
            return self.calcul_impurete(df, method)

    def GINI(self, df):
        """
            Calcule de l'indice de GINI
        """

        # On obtient les valeurs des données target dan sla dernière colonne
        val_unq = df.iloc[:, -1].unique()  

        gini = 0.0

        # Pour chaque valeur unique
        for val in val_unq:
            # Calcul de la proportion
            proportion = len(df[df.iloc[:, -1] == val]) / len(df)
            gini = gini - (proportion ** 2)

        return gini

    def CHIdeux(self,df):
        """
            Calcule de de l'indice du Chi2
        """

        # On obtient les valeurs des données target dan sla dernière colonne
        val_unq = df.iloc[:, -1].unique()  

        chid = 0.0
        # Pour chaque valeur unique
        for val in val_unq:
            O = len(df[df.iloc[:, -1] == val]) # O est le nombre d'occurrences de la valeur unique actuelle
            E = len(df) / len(val_unq) # E est le nombre attendu d'occurrences 
            chid = chid + (((O - E) ** 2) / E) # Calcul du Chi carré
        return chid

    def ENTROPIE(self,df):
        """
            Calcule de l'entropie de Shannon
        """

        # On obtient les valeurs des données target dan sla dernière colonne
        val_unq = df.iloc[:, -1].unique()
          
        entrop = 0.0

        # Pour chaque valeur unique
        for val in val_unq:
            # Calcul de la proportion
            proportion = len(df[df.iloc[:, -1] == val]) / len(df)
            # Calcul de l'entropie de Shannon
            entrop = entrop + ((- proportion) * np.log2(proportion))

        return entrop


    def calcule_valeur_feuille(self, x):
        """
            Détermine la valeur de la feuille avec la classe la plus fréquente
        """

        # Convertir les valeurs d'entrée en une liste
        x = list(x)

        # la valeur la plus fréquente dans la liste x
        val = max(x, key=x.count)
        return val

    def construction(self, df, niveau = 0, method = "GINI", historique_attrib = []):
        """
            Construit l'arbre
        """
       
        X = df.iloc[:, :-1]  # Toutes les colonnes sauf la dernière
        y = df.iloc[:, -1]  # La dernière colonne représente les target

        nb_ech, nb_attrib = X.shape # nombre d'échantillons et d'attributs

        # Critère d'arrêt : pureté du nœud, profondeur maximale ou nombre d'échantillons est inférieur au nombre minimum 
        if len(y.unique()) == 1 or niveau == self.taille_max or nb_ech < self.ech_min:
 
            val_actu = self.calcule_valeur_feuille(y) # Calcule la valeur de classe la plus fréquente
            return Node(prediction=val_actu) 

        # Choix de l'attribut qui maximise le gain d'information
        attrib, seuil, heritier_g, heritier_d, = self.choix_attribut(df,method)
        # Construction arbre
        niveau = niveau + 1
        nd_g = self.construction(heritier_g, niveau, method)
        nd_d = self.construction(heritier_d, niveau, method)
        return Node(attrib=attrib, seuil=seuil, nd_gauche=nd_g, nd_droit=nd_d)

    def prediction(self, df):
        """
            Prédiction des étiquettes de classe
        """

        predicts = [] # Liste pour stocker les prédictions

        # Parcourt chaque ligne du df
        for i, rw in df.iterrows():
            # prédiction pour la ligne actuelle
            #print(self.racine)
            pred = self.predire(rw, self.racine)
            # Ajoute la prédiction à la liste
            predicts.append(pred)


        return predicts

    def predire(self, x, nd):
        """
            Parcourt l'arbre de décision pour prédire la valeur cible
        """

        # Parcourt l'arbre jusqu'à ce qu'un nœud de prédiction soit atteint
        while nd.prediction is None:
            if x.iloc[nd.attrib] <= nd.seuil:

                nd = nd.nd_gauche
            else:

                nd = nd.nd_droit
        return nd.prediction

    def fit(self, x, y,method):
        """
            Entraîne l'arbre de décision
        """

        # Concatène x et y en un df
        df = pd.concat([x, y], axis=1)

        # Construit un arbre de décision
        self.racine = self.construction(df,niveau=0,method=method)


# Chargement des données
iris = load_iris()
x = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

# Transformation en df
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

# Gini
arbreG = DecisionTree(ech_min = 2, taille_max = 10)
arbreG.fit(X_train, y_train, method='GINI')
predictionsG = arbreG.prediction(X_test)
print(f'Précision de votre implémentation de l\'arbre de décision avec la méthode GINI : {accuracy_score(y_test, predictionsG)}')
matrice_confusionG = confusion_matrix(y_test, predictionsG)



# Entropie
arbreE = DecisionTree(ech_min = 2, taille_max = 10)
arbreE.fit(X_train, y_train, method='ENTROPIE')
predictionsE = arbreE.prediction(X_test)
print(f'Précision de votre implémentation de l\'arbre de décision avec la méthode Entropie : {accuracy_score(y_test, predictionsE)}')
matrice_confusionE = confusion_matrix(y_test, predictionsE)
print(matrice_confusionE)



# CHi2
arbreC = DecisionTree(ech_min = 2, taille_max = 10)
arbreC.fit(X_train, y_train, method='CHIdeux')
predictionsC = arbreC.prediction(X_test)
print(f'Précision de votre implémentation de l\'arbre de décision avec la méthode Chi2 : {accuracy_score(y_test, predictionsC)}')
matrice_confusionC = confusion_matrix(y_test, predictionsC)
print(matrice_confusionC)




X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)
# Initialisation du classificateur
dtc = DecisionTreeClassifier()
# Entraînement du classificateur
dtc.fit(X_train, y_train)
# Prédiction sur l'ensemble de test
predictions_skl = dtc.predict(X_test)
# Calcul et affichage de la précision du classificateur de Scikit-learn
print(f'Précision de l\'implémentation de l\'arbre de décision de Scikit-learn : {accuracy_score(y_test, predictions_skl)}')
matrice_confusion = confusion_matrix(y_test, predictions_skl)
print(matrice_confusion)
