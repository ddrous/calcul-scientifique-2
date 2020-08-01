import numpy as np
import numpy.linalg as nplin
import scipy as sp
import scipy.sparse as spsp
import scipy.sparse.linalg as spsplin
import matplotlib.pyplot as plt
import time as time

""" PARTIE 1 """


""" QUESTION 1 """

def iter_Arnoldi_sym( A, v, vold, beta):
    # n = len(vold)
    w = A@v
    alpha = w.T@v
    w -= alpha*v + beta*vold
    beta = np.linalg.norm(w)
    
    vold = v
    v = w / beta

    return v, vold, alpha, beta


""" QUESTION 2 """

def Lanczos(A, nbiter):
    n, n = np.shape(A)
    eigval = np.zeros((nbiter, nbiter))
    
    T = np.zeros((nbiter, nbiter), dtype=np.float)
    # print(T)
        
    v0 = np.random.rand(n)
    v0 = v0 / np.linalg.norm(v0)

    vold = np.zeros(n)
    v = v0

    # Matrice V
    # V = np.zeros((n, nbiter), dtype=np.float)

    beta = 0

    for i in range(1, nbiter+1):
        #Remplissage de V
        # V[:, i-1] = v

        v, vold, alpha, beta = iter_Arnoldi_sym(A, v, vold, beta)

        T[i-1, i-1] = alpha
        if i < nbiter:
            T[i, i-1] = beta
            T[i-1, i] = beta

        # val_prpr, vec_prpr = np.linalg.eig(T[:i, :i])
        val_prpr = np.linalg.eig(T[:i, :i])[0]
        
        val_prpr = np.sort(val_prpr)
        eigval[i-1, 0:i] = val_prpr

    # ESTIMATION DU CONDITIONNEMENT DE A: cond(A) = cond(T)
    cond_A_approchee = nplin.cond(T)
    print("Estimation du contitionnement de A:   ", cond_A_approchee)
    print("Valeur exacte du contitionnement de A:", nplin.cond(A.todense()))

    return eigval


""" QUESTION 3 """

def testLanczos():
    d = 10
    n = d**2
    nbiter = 40
    A = spsp.diags([[4.]*n,[-1]*(n-1),[-1]*(n-1),[-1]*(n-d),[-1]*(n-d)], [0,1,-1,d,-d])
    
    valeurs_propres_simulees = Lanczos(A, nbiter)
    # print("Valeurs propres simulees\n", valeurs_propres_simulees)

    x = np.arange(nbiter)

    # PLUS GRANGE VALEUR PROPRE
    # print(np.linalg.eig(A)[0])
    val_propre_max_exacte = np.ones(nbiter)
    val_propre_max_exacte = np.linalg.eig(A.todense())[0].max() * val_propre_max_exacte
    # print(val_propre_max_exacte)
    # Remplissage des valeurs propres simulees max
    eigen_simulees_max = []
    # Remplissage avec les plus grandes valeurs de Ritz (sur la diagonales)
    for i in range(nbiter):
        eigen_simulees_max.append(valeurs_propres_simulees[i][i])
    # print("valeurs propres simulees\n", eigen_simulees_max)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Convergence des valeurs propres de Ritz maximales')
    ax[0].plot(x, eigen_simulees_max, label='Valeurs de Ritz')
    ax[0].plot(x, val_propre_max_exacte, label='Valeur propre exacte maximale')
    ax[0].legend()
    # y = np.abs(eigen_simulees_max - val_propre_max_exacte)
    z = np.log(np.abs(eigen_simulees_max - val_propre_max_exacte))
    ax[1].plot(x, z, label='Difference logarithmique')
    ax[1].legend()
    # fig.tight_layout()
    # plt.show()

    # PLUS PETITE VALEUR PROPRE
    val_propre_min_exacte = np.linalg.eig(A.todense())[0].min() * np.ones(nbiter)
    # Remplissage des valeurs propres simulees min
    eigen_simulees_min = []
    # Remplissage avec les plus petites valeurs de Ritz
    for i in range(nbiter):
        eigen_simulees_min.append(valeurs_propres_simulees[i][0])
    # print("valeurs propres simulees\n", eigen_simulees_min)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Convergence des valeurs propres de Ritz minimales')
    ax[0].plot(x, eigen_simulees_min, label='Valeurs de Ritz')
    ax[0].plot(x, val_propre_min_exacte, label='Valeur propre exacte minimale')
    ax[0].legend()
    # y = np.abs(eigen_simulees_min - val_propre_min_exacte)
    z = np.log(np.abs(eigen_simulees_min - val_propre_min_exacte))
    ax[1].plot(x, z, label='Difference logarithmique')
    ax[1].legend()
    # fig.tight_layout()
    # plt.show()

    """ QUESTION 4 """

    # SIMULATION DE L'EVOLUTION DES VALEURS DE RITZ
    eigen_simulees = np.zeros((n, nbiter))
    fig, ax = plt.subplots(1,1, figsize=(10, 4)) 
    fig.suptitle('Convergence de toutes les valeurs propres de Ritz')
    for k in range(n):
        for i in range(nbiter):
            if i < k:
                eigen_simulees[k, i] = valeurs_propres_simulees[i, i]
            else:
                eigen_simulees[k, i] = valeurs_propres_simulees[k, k]
        
        # for i in range(k):
        #     eigen_simulees[k, i] = valeurs_propres_simulees[i, i]
        # for i in range(k, nbiter):
        #     eigen_simulees[k, i] = valeurs_propres_simulees[k, k]

        ax.plot(x, eigen_simulees[k, :], label=str(k+1)+'-eme valeur propre')
        ax.legend()
    # fig.tight_layout()
    plt.show()
    # valeurs_propres_exacte = np.sort(np.linalg.eig(A.todense())[0])
    # print("Valeurs propres exactes\n", valeurs_propres_exacte)
    # print("Valeurs propres totales\n", valeurs_propres_simulees)
    # print("Valeurs de ritz\n", eigen_simulees)


""" QUESTION 5 """

def testLanczos2():
    L = np.zeros(shape=(203))
    L[0] = 0
    for i in range(1, 201):
        L[i] = L[i-1] + 1/100
    L[-2] = 2.5
    L[-1] = 3
    B = spsp.diags (L, 0, dtype=np.float64)
    # print("L\n", L)

    n = np.shape(B)[0]
    nbiter = n + 10
    x = np.arange(nbiter)

    valeurs_propres_simulees = Lanczos(B, nbiter)
    
    # SIMULATION DE L"EVOLUTION DES VALEURS DE RITZ
    eigen_simulees = np.zeros((2, nbiter))
    fig, ax = plt.subplots(1,1, figsize=(10, 4)) 
    fig.suptitle('Convergence des deux valeurs de Ritz les plus grandes')
    for k in range(n-2, n):        
        for i in range(k):
            eigen_simulees[k-n+2, i] = valeurs_propres_simulees[i, i]
        for i in range(k, nbiter):
            eigen_simulees[k-n+2, i] = valeurs_propres_simulees[k, k]

        ax.plot(x, eigen_simulees[k-n+2, :], label=str(k+1)+'-eme valeur propre')
        ax.legend()
    # fig.tight_layout()

    print("Avant derniere valeur propre (Ritz):\n", eigen_simulees[0])
    print("Derniere valeur propre (Ritz):\n", eigen_simulees[1])
    valeurs_propres_exacte = np.sort(np.linalg.eig(B.todense())[0])
    print("Avant derniere valeur propre exacte:", valeurs_propres_exacte[-2])
    print("Derniere valeur propre exacte:", valeurs_propres_exacte[-1])
    commentaire = """On remarque qu'il n'ya tout simplement pas de difference entre les valeurs de Ritz. \
Elles convergent vers les deux valeurs les plus grandes. """
    commentaire_suite = """D'entree, ces deux valeurs propres ont la meme suite de valeur de Ritz jusqu'a l'iteration 202. \
Le cacul des dernieres valeurs de la suite (pour l'avant derniere valeur propres) ne correspond plus a la valeur voulue, \
ceci du au "mauvais" calcul/choix des valeurs de Ritz precedentes  mais surtout aux erreurs d'arrondi qui rendent le calcul des \
valeurs de Ritz restantes identique aux valeurs de Ritz pour la plus grande valeur propre """
    print("Commentaire:") 
    print(commentaire)
    print(commentaire_suite)
    plt.show()


""" PARTIE 2 """


""" QUESTION 6 """

def facto_QR_hessenberg(A):
    n = np.shape(A)[0]
    G = np.zeros_like(A)
    for i in range(n):
        G[i, i] = 1
    R = A
    Q = G # Matrice identite
    for k in range(n-1):
        # R[k+1, k] est deja nul
        if (R[k, k]**2 + R[k+1, k]**2) < 1e-200:
            continue

        c = -(R[k, k] / np.sqrt(R[k, k]**2 + R[k+1, k]**2)) 
        s = (R[k+1, k] / np.sqrt(R[k, k]**2 + R[k+1, k]**2))

        G[k, k] = c
        G[k, k+1] = -s
        G[k+1, k] = s
        G[k+1, k+1] = c
        # print("c^2 + s^2 =", c**2 + s**2)
        # print("G:\n", G)

        R = G@R
        Q = G@Q
        # print("R:\n", R)

        G[k, k] = 1
        G[k, k+1] = 0
        G[k+1, k] = 0
        G[k+1, k+1] = 1
    
    return Q.T, R


""" QUESTION 7 """

def testFacto_QR_hessenberg():
    A = np.triu([[1,2,3],[4,5,6],[7,8,-0.49]], -1).astype(np.float64)
    A = np.triu(np.random.randint(-5, 5, size=(4, 4)), -1).astype(np.float64)
    A[0, 0] = 2
    A[2, 1] = 0
    Q, R = facto_QR_hessenberg(A)
    print("A:\n", A)
    print("Q:\n", Q)
    # print("Q@Q.T:\n", Q@Q.T)
    print("R:\n", R)
        

""" QUESTION 8 """

def QR_hessenberg(A):
    n = np.shape(A)[0]
    A = A
    Q, R = facto_QR_hessenberg(A)
    valeurs_propres = np.zeros(np.shape(A)[0])

    # La norme est le maximum en valeur absolue des coefficients de la sous-diagonale -1
    def norm(A):
        # max_norm = np.abs(A[1][0])
        max_norm = 0
        for j in range(n-1):
            if np.abs(A[j+1, j]) > max_norm:
                max_norm = np.abs(A[j+1, j])
        return max_norm    

    max_norm = norm(A)
    nb_iter = 0
    # for _ in range(1000):
    while nb_iter < 5000 and max_norm > 1e-6:      # Trop lent avec la boucle while, le calcul de la norme semble inutilement couteux
        A = R@Q
        Q, R = facto_QR_hessenberg(A)
        max_norm = norm(A)
        nb_iter += 1
    
    for i in range(n):
        valeurs_propres[i] = A[i, i]

    # print("A final ---------\n", A)
    return valeurs_propres


""" QUESTION 9 """

def testQR_hessenberg():
    print("""\nVERIFICATION POUR UNE MATRICE SYMETRIQUE TRIDIAGONALE """)
    diag = np.random.randint(-5, 5, size=(4)).astype(np.float64)
    diag_sup = np.random.randint(-5, 5, size=(3)).astype(np.float64)
    A = np.diag(diag) + np.diag(diag_sup,1) + np.diag(diag_sup,-1)
    print("A initial ---------\n", A)
    val_pr = QR_hessenberg(A)
    real_val_pr = np.real(nplin.eig(A)[0])
    print("Valeurs Propres Exactes -------------- \n", np.sort(real_val_pr))
    print("Valeurs Propres Calculees ------------- \n", np.sort(val_pr))
    print("Erreur sur ces valeurs propres ------------- \n", np.abs(np.sort(val_pr)-np.sort(real_val_pr)))

    print("""\nVERIFICATION POUR UNE MATRICE DE HESSENBERG QUELCONQUE """)
    A = np.triu(np.random.randint(-5, 5, size=(5, 5)), -1).astype(np.float64)
    print("A initial ---------\n", A)
    val_pr = QR_hessenberg(A)
    real_val_pr = np.real(nplin.eig(A)[0])
    print("Valeurs Propres Exactes -------------- \n", np.sort(real_val_pr))
    print("Valeurs Propres Calculees ------------- \n", np.sort(val_pr))
    print("Erreur sur ces valeurs propres ------------- \n", np.abs(np.sort(val_pr)-np.sort(real_val_pr)))
#     commentaire1 = """On observe qu'il y a convergence dans la mojorite des cas, mais dans plusieurs autres cas, meme\
# le cas sysmetrique ne conduit a aucun resultat satisfaisant. Ceci se produit quand le systeme ne satisfait pas les \
# condition pour appliquer la methode QR (Deux valeurs propres identiques par exemple)"""
    commentaire2 = """On constate que pour des matrices de petite taille, tres souvent on n'obtient pas de bonnes approximations \
de nos valeurs propres. Par contre, plus la taille de la matrice de Hessenberg augmente, plus nos aproximations sont precises."""
    print("\nCommentaire -----------")
    print(commentaire2)


""" QUESTION 10 """
def Lanczos_Prime(A, nbiter):
    n, n = np.shape(A)
    eigval = np.zeros((nbiter, nbiter))
    T = np.zeros((nbiter, nbiter), dtype=np.float)
    v0 = np.random.rand(n)
    v0 = v0 / np.linalg.norm(v0)
    vold = np.zeros(n)
    v = v0
    beta = 0
    for i in range(1, nbiter+1):

        v, vold, alpha, beta = iter_Arnoldi_sym(A, v, vold, beta)

        T[i-1, i-1] = alpha
        if i < nbiter:
            T[i, i-1] = beta
            T[i-1, i] = beta

        # val_prpr, vec_prpr = np.linalg.eig(T[:i, :i])
        # if i != 1:
        val_prpr = QR_hessenberg(T[:i, :i])
        
        val_prpr = np.sort(val_prpr)
        eigval[i-1, 0:i] = val_prpr

    # ESTIMATION DU CONDITIONNEMENT DE A: cond(A) = cond(T)
    cond_A_approchee = nplin.cond(T)
    print("Estimation du contitionnement de A:   ", cond_A_approchee)
    print("Valeur exacte du contitionnement de A:", nplin.cond(A.todense()))

    return eigval

def comparaison():
    d = 4
    n = d**2
    nbiter = 20
    A = spsp.diags([[4.]*n,[-1]*(n-1),[-1]*(n-1),[-1]*(n-d),[-1]*(n-d)], [0,1,-1,d,-d])
    val_propres_exactes = nplin.eig(A.todense())[0]
    print("\nValeurs propres EXACTES:\n", val_propres_exactes)

    start = time.time()
    val_propres_lanczos = Lanczos(A, nbiter)
    time_lanczos = time.time() - start
    print("\nValeurs propres de Ritz avec EIGEN:\n", val_propres_lanczos)
    print("Temps avec EIGEN:", time_lanczos, "secondes")
    
    start = time.time()
    val_propres_hessenberg = Lanczos_Prime(A, nbiter)
    time_hessenberg = time.time() - start
    print("\nValeurs propres de Ritz avec HESSENBERG:\n", val_propres_hessenberg)
    print("Temps avec HESSENBERG:", time_hessenberg, "secondes")

    print("Le calul reste correcte (losque les tailles des matrices sont assez elevees). La methose QR est definitivement plus lente (et moins precise) que la recherche des valeurs propres avec avec EIGEN.")



""" POUR L"EXECUTION """

""" Questions 3 et 4 """
# testLanczos()
""" Question 5 """
# testLanczos2()
""" Question 7 """
# testFacto_QR_hessenberg()
""" Question 9 """
testQR_hessenberg()
""" Question 10 """
# comparaison()
