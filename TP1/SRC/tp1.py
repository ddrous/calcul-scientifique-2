import numpy as np
import scipy as sp
import scipy.linalg as spli
import scipy.sparse.linalg as spspli
import scipy.sparse as spsp
import matplotlib.pyplot as plt
import time


# Partie 1. (Format Compressed Sparse Row)

row = sp.array([0, 0, 1, 2, 2, 2])
col = sp.array([0, 2, 2, 0, 1, 2])
data = sp.array([1, 2, 3, 4, 5, 6])
A = spsp.csr_matrix((data, (row, col)), shape = (3, 3))

def test1_1():  # Question 1, premiere partie: signification de A.data, A.indices, et de A.indptr
    print('matrice A: \n', A.toarray())
    print('A.data:    ', A.data)             # A.data Correspond aux éléments non nuls de la matrice CSR
    print('A.indices: ', A.indices)          # # A.indptr correspond aux indices des colones dans data dans data
    print('A.indptr:  ', A.indptr)           # A.indptr correspond aux indices qui pointes au debut de chaque ligne dans data


def test1_2():  # Question 1, deuxième partie: definition de la fonction 'print'
    print('\n matrice A en csr: \n', A)

def test2_1():      # Resultat de A[0, :]
    s_time = time.time()
    print(A[0, :])      # Affiche tout les éléments de A d'indice [0,j] (0 <= j <= 2) dont les valeurs sont non nulles
                        # Plus rapide parce que les éléments de la matrice sont stockés par ligne dans A.data
    print(time.time()-s_time, "seconds")       # Donne le temps d'execution de la fonction

def test2_2():      # Resultat de A[:, 0]
    s_time = time.time()
    print(A[:, 0])      # Affiche tout les éléments de A d'indice [i,0] (0 <= i <= 2) dont les valeurs sont non nulles
    print(time.time() - s_time, "seconds")

def test3(A, B):    # Addition de matrice csr?
    C = spsp.csr_matrix(A) + spsp.csr_matrix(B)
    return C     # Oui il est possible d'ajouter deux matrices au format csr car meme si seuls les tableaux data, indices, et indptr
                 # sont stockés et sont eventuellment differents, n'oublions que les elements sont nuls ailleur, ce qui facilite les calculs

a = spsp.csc_matrix(([1, 5], ([0, 0], [0, 2])), shape = (1, 3))     # LA REPRESENTATION CSR DES MATRICES DE 1 SEULE LIGNE EST TROP BIZARE
                                                                    # C'EST EN FAIT LA REPRESENTATION DE LEUR TRANSPOSEE (qui est tout le
                                                                    # temps la même chose avec scipy.sparse)
b = np.array([3, 1, -1])
def matvect_multiply(A, b):      # Question 4. Mutiplication de A (au format csr) par b
    y = np.zeros((len(b)))
    for i in range(len(A.toarray())):
        for j in range(A.indptr[i], A.indptr[i+1]):
            y[i] += A.data[j] * b[A.indices[j]]
    return y

def test4_2(A, b):      # Test de matvect_multiply(A, b). Verification du produit A par b
    print('matrice A: \n', A.toarray())
    print('vecteur b: ', b)
    print('vecteur Ab: ', A.dot(b))
    # print('vecteur Ab: ', A@b)
    # print('vecteur Ab: ', np.dot(A.toarray(), b))     # La commande 'np.dot(A, b)' ne marche pas car elle s'atend a
                                                        # des matrices sous forme dense, et non au format csr


# Partie 2. (Factorisation LU)

B = np.array([[2.5, 0.4, -0.5],[0, -1, 0.25],[0, 0, 4]])
def Facto_LU(A):        # Question 5. Factorisation LU inplace
    n = A.shape[0]      # En supposant que la matrice est carree
    if (n != A.shape[1]):
        print("Matrice non carree!")
        return -1
    A = A.astype('float64')     # Pour éviter les divisions entieres
    if spsp.issparse(A):        # PROBLEME: CERTAINS CAS DE FACTORISATION AVEC CSR NE SONT PAS BONS! (Voire Question 9, matrice B)
        A = A.tolil()       # Accelere considerablement le temps de calcul pour les matrices au format csr
        # A = A.toarray()     # Accelere encore plus les calculs!
    for k in range(n-1):
        if (A[k, k] == 0):      # Attention! Utiliser A[k, k] et non A[k][k]
            print("Pivot nul! car A[" + str(k) + ", " + str(k) + "] devient nul")
            # print('facto LU: \n', A)
            return -1
        for i in range(k+1, n):
            A[i, k] = A[i, k]/A[k, k]
            A[i, k+1:] = A[i, k+1:] - A[i, k]*A[k, k+1:]
    if (A[n-1, n-1] == 0):      # JE PENSE QUE CERTAINS CAS DE FACTO LU CSR NE MARCHENT PAS CAR LE DERNIER PIVOT EST MAL CALCULE,
                                # LA STRUCTURE CSR DE LA MATRICE N'EST PAS MISE A JOUR. (Voire Question 9, matrice B (test9_2()))
        print("Avertissement: Dernier pivot A[" + str(n-1) + ", " + str(n-1) + "] nul!")
        return -1
    if(spsp.isspmatrix_lil(A)):     # On a transforme la matrice en linked list pour aller plus vite en haut
        A = A.tocsr()
    return A
# Penser a changer le format de A pour une csr_matrix vers une lil_matrix

def test5_1(A, LU):     # Verification de la factorisation LU
    n = LU.shape[0]
    L = np.zeros((n, n))        # Creation de L
    U = np.zeros((n, n))        # Creation de U
    for i in range(n):
        for j in range(n):
            if (i == j):
                L[i, j] = 1
                U[i, j] = LU[i, j]
            elif (i > j):
                L[i, j] = LU[i, j]
            else:
                U[i, j] = LU[i, j]
    print('matrice L:\n', L)
    print('matrice U:\n', U)
    print('matrice A:\n', A)
    print('produit matriciel LU:\n', np.dot(L, U))
    if (np.array_equal(A.toarray().astype('float64'), np.dot(L, U))):
        print('Facto LU correcte')
    else:
        print('Facto LU echouee')

def solve_LU(A, b):        # Question 6, Fonction 'solve_LU'
    x = np.zeros(shape=b.shape)
    y = np.zeros(shape=b.shape)
    A = Facto_LU(A)
    if (spsp.issparse(A) == False):
        A = spsp.csr_matrix(A)
    n = A.shape[0]
    # Algorithme de descente, A[i][i] = 1
    for i in range(0, n):
        # y[i] = (b[i] - np.dot(A[i, :i], y[:i]).sum())       # Methode classique de descente
        sum = 0
        for j in range(A.indptr[i], A.indptr[i+1]-1):       # On retire 1 a l'extremite droite parce qu'on sait que B[i, i] = 1 != 0
            if(A.indices[j] < i):                           # Eviter de prendre aussi l'element x[i]*A[i, i]
                sum += A.data[j] * y[A.indices[j]]
        y[i] = b[i]-sum
    # Algorithme de remontee
    for i in range(n-1, -1, -1):
        # x[i] = (y[i] - np.dot(A[i, i+1:], x[i+1:]).sum())/A[i, i]     # Methode classique de remontee
        sum = 0
        for j in range(A.indptr[i]+1, A.indptr[i+1]):
            if (A.indices[j] > i):
                sum += A.data[j] * x[A.indices[j]]
        x[i] = (y[i]-sum)/A[i, i]
    return x

def test7(b):        # Verification de la resolution avec A definie comme dans l'enonce
    n = len(b)
    A = spsp.diags([- np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1])
    A = A.toarray()     # Important car A est definie creuse en haut
    print('matrice A:\n', A)
    print('vecteur b:', b)
    print('ma methode:    x =', solve_LU(A, b))
    print('methode scipy: x =', spli.solve(A, b))
    if(np.array_equal(solve_LU(A, b), spli.solve(A, b))):
        print('Les deux solutions sont bien egales')


# Partie 3 (Remplissage - fill in)

# Pour la question 8, * on trouve que alpha doit etre different de 1, 2, ..., n-1 A pour que A soit factorisable LU
#                     * B admet une facto LU pour alpha different de n-1 (verifiable grace au test9_2(), mais seulement en format dense)

def test9_1(n, alpha):       # Creation de la matrice A de taille n et de valeur alpha a preciser
    A = spsp.dok_matrix((n, n), dtype=np.float64)
    A[0, :] = 1
    A[:, 0] = 1
    for i in range(n):
        A[i, i] = 1
    A[0, 0] = alpha
    return A.tocsr()

def test9_2(n, alpha):      # Creation de la matrice B de taille n et de valeur alpha a preciser
    B = spsp.dok_matrix((n, n), dtype=np.float64)
    B[n-1, :] = 1
    B[:, n-1] = 1
    for i in range(n):
        B[i, i] = 1     # Mettre B[i, i] et non B[i][i]
    B[n-1, n-1] = alpha
    return B.tocsr()

def test10(A):      # Structure de la matrice avant et apres la facto LU. La matrice A est au format csr
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.spy(A)
    plt.title("Avant la facto LU")
    plt.subplot(1, 2, 2)
    plt.title("Apres la facto LU")
    plt.spy(spsp.csc_matrix(Facto_LU(A.toarray())))
    plt.show()
    # La facto LU inplace transforme A en une matrice pleine car toutes les lignes de A sont modifiees pendant la factorisation
    # La meme facto conserve la forme de B car seule la derniere ligne est constament modifiee pour obtenir une forme triangulaire superieure


# Partie 4 (temps de calcul)

def test11(n):      # Assemblage de la matrice
    d = np.sqrt(n)
    data = [4*np.ones(shape = (n)), -np.ones(shape = (n)), -np.ones(shape = (n)), -np.ones(shape = (n)), -np.ones(shape = (n))]
    diags = np.array([0, 1, -1, d, -d])
    A = spsp.spdiags(data, diags, n, n)
    return A.tocsc()

def test12_1(n):      # Temps de calcul en format csr qui varie avec la taille n de la matrice
    deb = time.time()
    Facto_LU(test11(n))
    fin = time.time()
    print("temps de calcul avec facto_LU =", fin - deb, "sec")

def test12_2(n):      # Temps de calcul en format dense
    deb = time.time()
    Facto_LU(test11(n).todense())
    fin = time.time()
    print("temps de calcul avec facto_LU =", fin - deb, "sec")

def test13(n):      # Le temps de calcul avec la fonction splu
    deb = time.time()
    spspli.splu(test11(n))
    fin = time.time()
    print("temps de calcul avec module scipy =", fin - deb, "sec")

# Le temps de calcul avec le module scipy est beauuucoup plus rapide !! La lenteur relative de la facto
# LU en format csr est du au fait que python doit recalculer les tableaux A.data, A.indices, et A.indptr
# a chaque fois qu'un element non nul apparait ou il y avait auparavant un element nul dans A