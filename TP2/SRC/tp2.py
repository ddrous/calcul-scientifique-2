import numpy as np
import numpy.linalg as npalg
import scipy as sp
import scipy.sparse as spsp
import scipy.sparse.linalg as spsplin
import matplotlib.pyplot as plt
import time

plt.style.use("seaborn")
plt.figure(figsize=(10, 5))

# Defnition de n et d pour tout le reste du programme
# d = 10
n = 100
d = int(np.sqrt(n))


# Vecteur second membre pour toutes les mothodes
b = np.ones(n)
for i in range(n):
  b[i] = 10
b[0] = -0.75
b[-1] = -12.05

# vecteur initial pour toutes les methodes iteratives
x0 = np.zeros(n)

# q nombres comparaisons a effectuer
q = 6

# Parametre p pour la question 4
p = 2

""" Tests a executer a la fin du programme pour afficher les resultats correspondants a chaque question """
""" Les parametres p et q sont optionnels! """
# question_3()
# question_4(p)
# question_6()
# question_7(q)
# question_8(q)
# question_10(q)
""" Remarque: 
              Les autres questions correspondent aux fonctions demandées
              Pour la question 4, la fonction demandée gmres(A, b, xexact, p) correspond a gmres2(A, b, x_exact, p) """



""" PARTIE 1 """

""" Question 1 """
def arnoldi(A, V, H):
  n = np.shape(A)[0]
  k = np.shape(V)[1]

  Vp = np.zeros((n, k+1))
  Hp = np.zeros((k+1, k))

  Vp[:, :-1] = V
  # for i in range(n):
  #   for j in range(k):
  #     Vp[i, j] = V[i, j]

  Hp[:-1, :-1] = H
  Hp[k, :-1] = 0
  # for i in range(k):
  #   for j in range(k-1):
  #     Hp[i, j] = H[i, j]

  for j in range(k):
    Hp[j, -1] = (A@V[:, -1])@V[:, j]
  
  wk = A@V[:, k-1]
  for j in range(k):
    # wk = wk - ((A@V[:, -1])@V[:, j])*V[:, j]
    wk -= Hp[j, -1] * V[:, j] 

  Hp[k, k-1] = np.linalg.norm(wk)
  Vp[:, k] = wk / np.linalg.norm(wk)

  return Vp, Hp

""" Question 2 """
def gmres(A, b, x_exact = None):
  n = np.shape(A)[0]

  # x0 sera une variable globale
  r0 = b - A@x0
  r0_norm = npalg.norm(r0)

  v0 = r0 / npalg.norm(r0)
  V = np.zeros((n, 1))
  V[:, 0] = v0

  H = np.zeros((1, 0))

  x = x0
  r_norm = r0_norm
  eps = 1e-4

  # Si la solution exacte n'est pas fournie, alors la calculer 
  if x_exact is None:
    if spsp.issparse(A):
      A = A.todense()
    x_exact = npalg.solve(A, b)
  x_exact_norm = npalg.norm(x_exact)

  # Matrice ligne pour les  erreurs relatives
  erreur_rel = []


  # Matrice ligne pour les residus relatifs
  res_rel = []


  while r_norm > eps:

    r_norm = np.linalg.norm(b - A@x)
    res_rel.append(r_norm / r0_norm)

    erreur_rel.append(npalg.norm(x_exact - x) / x_exact_norm)

    V, H = arnoldi(A, V, H)

    # On factorise en mode compplete sinon les tailles de Q et R ne correspondent pas.
    # En locurence, Q n'est pas caree...
    Q, R = np.linalg.qr(H, mode='complete')

    # k = np.shape(R)[0] - 2
    # e0 = np.zeros((k+2, 1))
    # e0[0] = 1
    
    # y = np.linalg.solve(R[:, -1], r0_norm * (Q.transpose()[:-1, :-1]@e0))
    y = np.linalg.solve(R[:-1, :], r0_norm * Q[0, :-1])

    # Calcul de x_k+1 a partit de V_k et non V_k+1 calcule plus haut
    x = x0 + V[:, :-1]@y

  return x, erreur_rel, res_rel


""" Question 3 """
A = np.diag(2*np.ones(n)) + 0.5 * np.random.rand(n, n)/np.sqrt(n)

b = np.ones(n)
for i in range(n):
  b[i] = 10
b[0] = -0.75
b[-1] = -12.05

# x_exact pour A
x_exact = npalg.solve(A, b)

# vecteur initial pour toutes les methodes iteratives
x0 = np.zeros(n)

def afficher_scipy(A, b, x_exact = None):
  print("\nMETHODE SCIPY ------------------------------------------------------------------")
  if spsp.issparse(A):
    A = A.todense()
  print("Matrice A: \n", A)
  print("Vecteur b: \n", b)
  x_exact = npalg.solve(A, b)
  print("Solution exacte du systeme: \n", x_exact)


def afficher_gmres(A, b, x_exact = None):
  solution, erreurs, residus = gmres(A, b, x_exact)
  print("\n\nGMRES ------------------------------------------------------------------------")
  print("Solution du systeme (GMRES): \n", solution)
  print("Liste des erreurs relatives (en echelle logarithmique): \n", np.log(erreurs))
  print("Liste des residus relatifs (en echelle logarithmique): \n", np.log(residus))  
  
  x = np.arange(0, len(erreurs)) 
  plt.subplot(1, 2, 1)
  plt.plot(x, erreurs, label="erreurs relatives")
  plt.legend()
  plt.subplot(1, 2, 2)
  plt.plot(x, residus, label="residus relatifs", color="g")
  plt.legend()
  plt.tight_layout()
  plt.suptitle("Erreurs et residus pour la methode gmres")
  plt.show()


def question_3():
  # A = np.diag(2*np.ones(n)) + 0.5 * np.random.rand(n, n)/np.sqrt(n)
  print("\nQUESTION 3\n")
  
  afficher_scipy(A, b)
  afficher_gmres(A, b)
  print()

# question_3()

""" Question 4 """
def gmres_2(A, b, x_exact, p):
  n = np.shape(A)[0]

  # Si la solution exacte n'est pas fournie, alors la definir 
  if x_exact is None:
    if spsp.issparse(A):
      A = A.todense()
    x_exact = npalg.solve(A, b)
  x_exact_norm = npalg.norm(x_exact)

  # Matrice des solutions initiales pour chaque iteration
  X = np.array([x0])
  X = X.T

  # Matrice ligne pour les  erreurs relatives
  erreur_rel = []

  # Matrice ligne pour les residus relatifs
  res_rel = []

  x0_prime = X[:, -1]
  r0 = b - A@x0_prime
  r_norm = npalg.norm(r0)
  eps = 1e-4

  while r_norm > eps:
    
    # x0_prime devient le vecteur initial a chaque reinitialisation, tout comme r0, v0, etc ...
    x0_prime = X[:, -1]
    r0 = b - A@x0_prime
    r0_norm = npalg.norm(r0)

    v0 = r0 / npalg.norm(r0)
    V = np.zeros((n, 1))
    V[:, 0] = v0

    H = np.zeros((1, 0))

    x = x0_prime
    r_norm = r0_norm

    ctr = 0
    # for i in range(p):
    while r_norm > eps and ctr < p:
      ctr += 1

      r_norm = np.linalg.norm(b - A@x)
      res_rel.append(r_norm / r0_norm)

      erreur_rel.append(npalg.norm(x_exact - x) / x_exact_norm)

      V, H = arnoldi(A, V, H)

      Q, R = np.linalg.qr(H, mode='complete')

      y = np.linalg.solve(R[:-1, :], r0_norm * Q[0, :-1])

      x = x0_prime + V[:, :-1]@y
      
      x = np.array([x]).T
      X = np.concatenate((X, x), axis=1)

  return X, erreur_rel, res_rel

def afficher_gmres_reinitialise(A, b, x_exact, p):
  SOLUTIONS, erreurs, residus = gmres_2(A, b, x_exact, p)
  print("\n\nGMRES REINITIALISE ------------------------------------------------------------")
  # print("\nSolutions initiales du systeme (GMRES reinitialise): \n", SOLUTIONS)
  print("Solution finale du systeme (GMRES reinitialise): \n", SOLUTIONS[:, -1])
  print("Liste des erreurs relatives (en echelle logarithmique): \n", np.log(erreurs))
  print("Liste des residus relatifs (en echelle logarithmique): \n", np.log(residus))
    
  x = np.arange(0, len(erreurs)) 
  plt.subplot(1, 2, 1)
  plt.plot(x, np.log(erreurs), label="erreurs relatives (logarithmes)")
  plt.legend()
  plt.subplot(1, 2, 2)
  plt.plot(x, np.log(residus), label="residus relatifs (logarithmes)", color="g")
  plt.legend()
  plt.tight_layout()
  plt.suptitle("Erreurs et residus pour la methode gmres reinitialise (avec p = " + str(p) + ")")
  plt.show()



def question_4(p = 2):
  # A = np.diag(2*np.ones(n)) + 0.5 * np.random.rand(n, n)/np.sqrt(n)
  print("\nQUESTION 4\n")
  
  afficher_scipy(A, b)
  # afficher_gmres(A, b)
  afficher_gmres_reinitialise(A, b, npalg.solve(A, b), p)

  print()

# question_4(2)


""" PARTIE 2 """

""" Question 5 """
def gradient_conjugue(A, b, x_exact = None):

  # Si la solution exacte n'est pas fournie, alors la definir 
  if x_exact is None:
    # Indispenable pour utiliser npalg.solve()
    A = A.todense()
    x_exact = npalg.solve(A, b)
  x_exact_norm = npalg.norm(x_exact)

  # Matrice ligne pour les  erreurs relatives
  erreur_rel = []

  # Matrice ligne pour les residus relatifs
  res_rel = []

  # x0 = np.zeros(n)
  x = x0 
  r0 = b - A@x
  d = r0
  r0_norm = npalg.norm(r0)
  r = r0
  r_norm = r0_norm
  eps = 1e-4

  # Puisque x, d et r sont matrice ligne, transformons les en vecteurs (matrices colones)  
  x = np.matrix(x).T    # Car x n'est pas encore une matrice
  d = np.matrix(d).T  
  r = np.matrix(r).T

  while r_norm > eps:
    res_rel.append(r_norm / r0_norm)
    erreur_rel.append(npalg.norm(x_exact - x) / x_exact_norm)

    s = (r.T @ r)[0, 0] / ((A@d).T @ d)[0, 0]
    x = x + (s * d)    
    r_plus_1 = r - s * (A@d)
    beta = (r_plus_1.T @ r_plus_1)[0, 0] / (r.T @ r)[0, 0]
    d = r_plus_1 + beta * d

    # Pour assurer l'iteration
    r = r_plus_1

    r_norm = npalg.norm(r)


  return x.T, erreur_rel, res_rel

""" Question 6 """
B = spsp.diags([[4.]*n,[-1]*(n-1),[-1] *(n-1),[-1] *(n-d),[-1] *(n-d)],[0,1,-1,d,-d])

def afficher_gradient_conjugue(B, b, x_exact=None):
  solution, erreurs, residus = gradient_conjugue(B, b)
  print("\n\nGRADIENT CONJUGUE ------------------------------------------------------------------------")
  print("Solution du systeme: \n", solution)
  print("Liste des erreurs relatives (en echelle logarithmique): \n", np.log(erreurs))
  print("Liste des residus relatifs (en echelle logarithmique): \n", np.log(residus))

  x = np.arange(0, len(erreurs))
  plt.subplot(1, 2, 1)
  plt.plot(x, erreurs, label="erreurs relatives")
  plt.legend()
  plt.subplot(1, 2, 2)
  plt.plot(x, residus, label="residus relatifs", color="g")
  plt.suptitle("erreurs et residus pour le gradient conjugue")
  plt.legend()
  plt.show()


# afficher_scipy(B, b, x_exact)
# afficher_gradient_conjugue(B, b, x_exact)

# Conparaison des temps
def comparaison_gmres_grad_conj():
  start_time = time.time()
  gmres(B, b, x_exact)
  gmres_time = time.time() - start_time
  print("Methode GMRES:            ", gmres_time, "sec")

  start_time = time.time()
  gradient_conjugue(B, b, x_exact)
  grad_conj_time = time.time() - start_time
  print("Methode GRADIENT CONJUGUE:", grad_conj_time, "sec")

# comparaison_gmres_grad_conj()

texte1 = """\nCommentaire:
La methode GMRES est beacoup plus rapide que celle du gradient conjugué (pour n = 5000, GMRES prend 3.5 sec
alors que GRADIENT ONCJUGUE prends 77.9 sec) par exemple !\n"""

texte2 = """\nRaison de l'adaptabilité aux structures creuses
Ces algorithmes sont adaptes aux structures creuses parce qu'ils reposent tous deux sur le calcul de 
l'espace de Krylov jusqu'a un certain degré. A etant creuse, cela facilite les produit Matrice x Vecteur 
et le calcul des differents vecteurs de la base de l'espace de Krylov (a savoir les v_k pour GMRES, 
et les d_k pour le GRADIENT CONJUGUE) se trouven simplifié\n"""

def question_6():
  print("\nQUESTION 6\n")
  afficher_scipy(B, b, x_exact)
  afficher_gradient_conjugue(B, b, x_exact)

  print("\nTemps de calcul")
  comparaison_gmres_grad_conj()
  print(texte1, texte2)

# question_6()

""" PARTIE 3 """
""" Question 7 """
C = np.diag(2+np.arange(n)) - np.diag(np.ones(n-1),1) - np.diag(np.ones(n-1),-1)
M = np.zeros_like(C)
for i in range(np.shape(C)[0]):
  M[i, i] = C[i, i]

# Comparaison des resultats et affichage des conditionnement
def afficher_preconditionne(C, M, b, x_exact = None):
  solution, erreurs, residus = gmres(C, b, None)
  print("\n\nGMRES STANDARD ------------------------------------------------------------------------")
  print("Condionnement de la matrice:", npalg.cond(C))
  print("Solution du systeme: \n", solution)
  print("Liste des erreurs relatives (en echelle logarithmique): \n", np.log(erreurs))
  print("Liste des residus relatifs (en echelle logarithmique): \n", np.log(residus))

  M_1 = npalg.inv(M)
  solution, erreurs, residus = gmres(M_1@C, M_1@b, None)
  print("\n\nGMRES PRECONDITIONNE------------------------------------------------------------------------")
  print("Condionnement de inv(M) x Matrice :", npalg.cond(M_1@C))
  print("Solution du systeme: \n", solution)
  print("Liste des erreurs relatives (en echelle logarithmique): \n", np.log(erreurs))
  print("Liste des residus relatifs (en echelle logarithmique): \n", np.log(residus))


def comparaison_gmres_gmres_precond(C, M, b):
  start_time = time.time()
  gmres(C, b, None)
  gmres_time = time.time() - start_time

  C_prime = npalg.inv(M)@C
  b_prime = npalg.inv(M)@b
  start_time = time.time()
  gmres(C_prime, b_prime, None)
  gmres_precond_time = time.time() - start_time

  return gmres_time, gmres_precond_time


def observations(q):
  obs = np.zeros((3, q))
  n = 4

  for i in range(q):
    n *= 2
    obs[0, i] = n

    C = np.diag(2+np.arange(n)) - np.diag(np.ones(n-1),1) - np.diag(np.ones(n-1),-1)
    M = np.zeros_like(C)
    for j in range(np.shape(C)[0]):
      M[j, j] = C[j, j]
    
    b = 10 * np.ones(n)
    b[0] = -0.75
    b[-1] = -12.05

    global x0
    x0 = np.zeros(n)

    comparaison = comparaison_gmres_gmres_precond(C, M, b)
    obs[1, i] = comparaison[0]
    obs[2, i] = comparaison[1]

  # Restituons la matrice x0
  list_of_globals = globals()
  x0 = np.zeros(list_of_globals["n"])

  return obs

def comparaisons_gmres_gmres_precond(q):
  observation = observations(q)
  print("Les tailles de n:\n", observation[0])
  print("temps gmres - temps gmres_precond (en secondes):\n", observation[1]-observation[2])
  plt.plot(observation[0], observation[1], label = "temps gmres")
  plt.plot(observation[0], observation[2], label = "temps gmres preconditionne")
  plt.xlabel("taille de la matrice")
  plt.xticks(observation[0])
  plt.ylabel("temps (seconde)")
  plt.suptitle("Comparison des temps d'execution gmres et gmres preconditionne (Matrice C)")
  plt.legend()
  plt.show()


def question_7(q = 6):
  # C = np.diag(2+np.arange(n)) - np.diag(np.ones(n-1),1) - np.diag(np.ones(n-1),-1)
  print("\nQUESTION 7\n")
  # global M
  M = np.zeros_like(C)
  for i in range(np.shape(C)[0]):
    M[i, i] = C[i, i]

  print("Matrice C:\n", C)
  print("Matrice b:\n", b)
  print("Solution exacte (SCIPY):\n", npalg.solve(C, b))
  
  afficher_preconditionne(C, M, b)
  
  print("\nComparaison des temps de calculs")
  comparaisons_gmres_gmres_precond(q)
  
  print("\nComentaire:\nIl est clair que le  temps gmres est plus grand que le temps gmres_precond (la metode preconditionnee est bien plus rapide), \nce qui n'est que normal car le conditionnement de C a ete significativement reduit par le preconditionnement!")

# question_7(q)

""" Question 8 """

D = np.diag(2*np.ones(n)) - np.diag(np.ones(n-1),1) - np.diag(np.ones(n-1),-1)
M = np.zeros_like(D)
for i in range(np.shape(D)[0]):
  M[i, i] = D[i, i]

def observations_2(q):
  obs = np.zeros((3, q))
  n = 4

  for i in range(q):
    n *= 2
    obs[0, i] = n

    D = np.diag(2*np.ones(n)) - np.diag(np.ones(n-1),1) - np.diag(np.ones(n-1),-1)
    M = np.zeros_like(D)
    for j in range(np.shape(D)[0]):
      M[j, j] = D[j, j]
    
    b = 10 * np.ones(n)
    b[0] = -0.75
    b[-1] = -12.05

    global x0
    x0 = np.zeros(n)

    comparaison = comparaison_gmres_gmres_precond(D, M, b)
    obs[1, i] = comparaison[0]
    obs[2, i] = comparaison[1]

  # Restituons la matrice x0
  list_of_globals = globals()
  x0 = np.zeros(list_of_globals["n"])

  return obs

def comparaisons_gmres_gmres_precond_2(q):
  observation = observations_2(q)
  print("Les tailles de n:\n", observation[0])
  print("temps gmres - temps gmres_precond (en secondes):\n", observation[1]-observation[2])
  
  plt.plot(observation[0], observation[1], label = "temps gmres")
  plt.plot(observation[0], observation[2], label = "temps gmres preconditionne")
  plt.xlabel("taille de la matrice")
  plt.xticks(observation[0])
  plt.ylabel("temps (seconde)")
  plt.suptitle("Comparison des temps d'execution gmres et gmres preconditionne (Matrice D)")
  plt.legend()
  plt.show()


def question_8(q = 6):
  # D = np.diag(2*np.ones(n)) - np.diag(np.ones(n-1),1) - np.diag(np.ones(n-1),-1)
  print("\nQUESTION 8\n")
  # global M
  M = np.zeros_like(D)
  for i in range(np.shape(D)[0]):
    M[i, i] = D[i, i]
  
  print("Matrice D:\n", D)
  print("Matrice b:\n", b)
  print("Solution exacte (SCIPY):\n", npalg.solve(D, b))
  
  afficher_preconditionne(D, M, b)
  
  print("\nComparaison des temps de calculs")
  comparaisons_gmres_gmres_precond_2(q)
  print("\nComentaire:\nLes deux methodes ont asymptotiquement les memmes temps d'execution.")
  print("Vu que le conditionnnement de D est egal a celui de inv(M) x D, on a aucune amelioration!")
  print("Bien que les matrices C et D soient toutes deux symetriques, on constate que le conditionnement de C \nest bien plus proche de 1 que celui de D, autrement dit, la matrice inv(M) x C est plus proche de Id que inv(M) x D, \nd'ou l'acceleration de la methode gmres_preconditionne (avec la matrice C).\n")

# question_8(q)

""" Question 9 """
def gradient_conjugue_precond(A, b, M, x_exact = None):
  # Si la solution exacte n'est pas fournie, alors la definir 
  if x_exact is None:
    # Indispenable pour utiliser npalg.solve()
    # A = A.todense()
    x_exact = npalg.solve(A.todense(), b)
  x_exact_norm = npalg.norm(x_exact)

  # Matrice ligne pour les  erreurs relatives
  erreur_rel = []

  # Matrice ligne pour les residus relatifs
  res_rel = []

  global x0
  x = x0 
  r0 = b - A@x
  r0_norm = npalg.norm(r0)
  r = r0

  if isinstance(M, spsplin.SuperLU):
    z = M.solve(r0.T)
  else:
    z = npalg.solve(M, r0.T)
  
  d = z
  r_norm = r0_norm
  eps = 1e-4

  # Puisque x, d et r sont matrice ligne, transformons les en vecteurs (matrices colones)  
  # x = np.matrix(x).T    # Car x et r n'est pas encore une matrice mais d et z le sont deja
  # r = np.matrix(r).T

  while r_norm > eps:
  # for i in range(n):    # Puisque le systeme atteitn la solution finale en au plus n iterations   
                        # Les resultats sont bizarres a partir de n = 110 !!
    
    res_rel.append(r_norm / r0_norm)
    erreur_rel.append(npalg.norm(x_exact - x) / x_exact_norm)

    # s = (r.T @ z)[0, 0] / ((A@d).T @ d)[0, 0]
    # s = (r.T @ z) / ((A@d).T @ d)
    s = (r.T @ z) / (d.T@A@d)
    x = x + (s * d)    
    r_prime = r - s * (A@d)

    # Boucle utile pour l'utilisation de la boucle for
    if npalg.norm(r_prime) < eps:
      break

    if isinstance(M, spsplin.SuperLU):
      z_prime = M.solve(r_prime)
    else: 
      z_prime = npalg.solve(M, r_prime)
    # beta = (r_prime.T @ z_prime)[0, 0] / (r.T @ z)[0, 0]
    beta = (r_prime.T @ z_prime) / (r.T @ z)
    d = r_prime + beta * d

    # Pour assurer l'iteration
    r = r_prime
    z = z_prime

    r_norm = npalg.norm(r)

  return x.T, erreur_rel, res_rel

""" Question 10 """
M = spsplin.spilu(B.tocsc())

def afficher_gradient_conjugue_precond(B, b, M, x_exact=None):
  solution, erreurs, residus = gradient_conjugue_precond(B, b, M)
  print("\n\nGRADIENT CONJUGUE PRECONDITIONNE-----------------------------------------------------------")
  print("Solution du systeme: \n", solution)
  print("Liste des erreurs relatives (en echelle logarithmique): \n", np.log(erreurs))
  print("Liste des residus relatifs (en echelle logarithmique): \n", np.log(residus))

  x = np.arange(0, len(erreurs))
  plt.subplot(1, 2, 1)
  plt.plot(x, erreurs, label="erreurs relatives (logarithme)")
  plt.legend()
  plt.subplot(1, 2, 2)
  plt.plot(x, np.log(residus), label="residus relatifs (logarithme)", color="g")
  plt.suptitle("erreurs et residus pour le gradient conjugue preconditionne")
  plt.legend()
  plt.show()


def comparaison_grad_conj_precond(n):
  d = int(np.sqrt(n))
  B = spsp.diags([[4.]*n,[-1]*(n-1),[-1] *(n-1),[-1] *(n-d),[-1] *(n-d)],[0,1,-1,d,-d])
  M = spsplin.spilu(B.tocsc())
  b = 10 * np.ones(n)
  b[0] = -0.75
  b[-1] = -12.05
  global x0
  x0 = np.zeros(n)

  start_time = time.time()
  gradient_conjugue(B, b, None)
  grad_conj_time = time.time() - start_time

  start_time = time.time()
  gradient_conjugue_precond(B, b, M, None)
  grad_conj_precond_time = time.time() - start_time

  # Restitution de la taille de x0
  list_of_globals = globals()
  x0 = np.zeros(list_of_globals['n'])

  return grad_conj_time, grad_conj_precond_time


def comparaisons_grad_conj_precond(q):
  obs = np.zeros((3, q))
  n = 2
  for i in range(q):
    n *= 2
    obs[0, i] = n
    comparaison = comparaison_grad_conj_precond(n)
    obs[1, i] = comparaison[0]
    obs[2, i] = comparaison[1]

  print("Les tailles de n:\n", obs[0])
  print("temps grad_conj - temps grad_conj_precond (en secondes):\n", obs[1]-obs[2])


def question_10(q = 5):
  # B = spsp.diags([[4.]*n,[-1]*(n-1),[-1] *(n-1),[-1] *(n-d),[-1] *(n-d)],[0,1,-1,d,-d])
  B = spsp.diags([[4.]*n,[-1]*(n-1),[-1] *(n-1),[-1] *(n-d),[-1] *(n-d)],[0,1,-1,d,-d])
  M = spsplin.spilu(B.tocsc())
  print("\nQUESTION 10\n")
  print("Matrice B:\n", B.todense())
  print("Matrice b:\n", b)
  print("Solution exacte (SCIPY):\n", npalg.solve(B.todense(), b))

  afficher_gradient_conjugue_precond(B, b, M)
  
  print("\nComparaison entre gradient conjugué et gradient conjugué predonditionnné ------------")
  comparaisons_grad_conj_precond(q)
  
  print("\nComentaire:\nDe façon globale, le gradient conjugue préconditionné est plus rapide que le gradient conjugué, ce n'est que normal !\n")

# question_10(q)





""" Décommenter ces lignes pour afficher les résultats correspondants à chaque question """
question_3()
question_4()
question_6()
question_7()
question_8()
question_10()
""" Remarque : 
              Les autres questions correspondent aux fonctions demandées
              Pour la question 4, la fonction demandee gmres(A, b, xexact, p) correspond a gmres2(A, b, x_exact, p)
 """
