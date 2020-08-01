import numpy as np
import numpy.linalg as nplin
import scipy as sp
import scipy.sparse as spsp
import scipy.sparse.linalg as spsplin
import matplotlib.pyplot as plt

class mesh:
    def __init__(self, Nel, xmin, xmax, deg=1):
        self.Nel = Nel      # Correspond a N+1 dans le cours
        self.xmin = xmin
        self.xmax = xmax
        self.deg = deg
        self.dof = np.zeros(self.deg*Nel+1)
        self.Ndof = Nel*self.deg - 1       # Correspond a N (dans le cas P1) dans le cours
    
    def connect(self, el, k):
        return self.deg*el + k
    
    def init_uniform(self):
        self.nodes = np.linspace(self.xmin, self.xmax, self.Nel+1)
        self.h = self.nodes[1:] - self.nodes[0:-1]
        for j in range(self.Nel):
            for k in range(self.deg):
                self.dof[self.connect(j, k)] = self.nodes[j] + self.h[j]*k/self.deg
        self.dof[-1] = self.nodes[-1]       # dof[0] = nodes[0] = xmin

    def init_random(self):
        # nodes = (self.xmin*np.random.rand(self.Ndof+2) + self.xmax*np.random.rand(self.Ndof+2)) / 2
        nodes = self.xmin + (self.xmax - self.xmin)*np.random.rand(self.Nel+1)
        nodes[0] = self.xmin
        nodes = np.sort(nodes)
        nodes[self.Nel] = self.xmax
        self.nodes = nodes
        self.h = self.nodes[1:] - self.nodes[0:-1]
        for j in range(self.Nel):
            for k in range(self.deg):
                self.dof[self.connect(j, k)] = self.nodes[j] + self.h[j]*k/self.deg
        self.dof[-1] = self.nodes[-1]
    
    def norm_P1(self, u):
        h = self.h
        norme_L2 = np.sqrt(np.sum(h*u[:-1]*u[:-1]))
        semi_norme_H1 = np.sqrt(norme_L2 + np.sum((u[1:]-u[:-1])**2/h))
        return norme_L2, semi_norme_H1

    def norm(self, u):
        norme_L2 = 0
        for j in range(1, self.Ndof+1, 2):
            norme_L2 += (self.dof[j+1]-self.dof[j-1]) * (u[j-1]**2 + 4*u[j]**2 + u[j+1]**2) / 6
        return np.sqrt(norme_L2)
        
def test_question_2():
    Mh = mesh(5, -2, 2)
    Mh.init_uniform()
    # Mh.init_random()
    print("nodes\n", Mh.nodes)
    # Mh = mesh(50, -2, 2)
    # Mh.init_uniform()
    # print("nodes\n", Mh.nodes)


# test_question_2() 

class fem:
    def __init__(self, mesh):
        self.mesh = mesh
    
    def matrixA_P1(self):
        self.A = spsp.dok_matrix((self.mesh.Ndof, self.mesh.Ndof), dtype=np.float64)
        h = self.mesh.h
        for i in range(1, self.mesh.Ndof+1):
            for j in range(1, self.mesh.Ndof+1):
                if i == j:
                    self.A[i-1, j-1] = 1/h[i-1] + 1/h[i] + h[i-1]/3 + h[i]/3
                elif i == j+1:
                    self.A[i-1, j-1] = -1/h[i-1] + h[i-1]/6
                elif i == j-1:
                    self.A[i-1, j-1] = -1/h[i] + h[i]/6
                # else:
                #     self.A[i-1, j-1] = 0
    
    def matrixA(self):
        self.A = spsp.dok_matrix((self.mesh.Ndof, self.mesh.Ndof), dtype=np.float64)
        M = np.array([[2,1,-0.5], [1,8,1], [-0.5,1,2]]) / 15
        K = np.array([[7,-8,1], [-8,16,-8], [1,-8,7]]) / 3
        for el in range(self.mesh.Nel):
            for ni in range(self.mesh.deg+1):
                i = self.mesh.connect(el, ni)
                for nj in range(self.mesh.deg+1):
                    j = self.mesh.connect(el, nj)
                    if 0 < i <= self.mesh.Ndof and 0 < j <= self.mesh.Ndof:
                        self.A[i-1,j-1] += K[ni, nj] / self.mesh.h[el]
                        self.A[i-1,j-1] += M[ni, nj] * self.mesh.h[el]

    def rhs_P1(self, f):
        N = self.mesh.Ndof
        h = self.mesh.h
        b = np.zeros(N)
        for i in range(1, N+1):
            b[i-1] = (h[i] + h[i-1]) * f(self.mesh.nodes[i]) / 2
        return b

    def rhs(self, f):
        # Matrice des valeurs de Phi_bar_0, Phi_bar_1, Phi_bar_2 (les lignes) en 0, 0.5, et 1 (les colones)
        Phi_bar = spsp.diags([1, 1, 1], 0).todense()      # Egale a la matrice identite
        b = np.zeros(self.mesh.Ndof)
        dof = self.mesh.dof
        for el in range(self.mesh.Nel):
            for ni in range(self.mesh.deg+1):
                i = self.mesh.connect(el, ni)
                if 0 < i <= self.mesh.Ndof:
                    # if ni == 0 and el != 0:
                    #     b[i-1] += 4 * (f(self.mesh.nodes[el])) * (self.mesh.h[el]+self.mesh.h[el-1]) / 6
                    b[i-1] += (f(dof[i-1])*Phi_bar[ni, 0] + 4*f(dof[i])*Phi_bar[ni, 1] + f(dof[i+1])*Phi_bar[ni, 2]) * (dof[i+1]-dof[i-1]) / 6
        return b
    
    def solve(self, f, plot=True):
        self.matrixA_P1()
        b = self.rhs_P1(f)
        u = nplin.solve(self.A.todense(), b)
        # u_tilde est le prologement de u par zero sur les bords
        u_tilde = np.zeros(self.mesh.Nel+1)
        u_tilde[1:-1] = u
        # Les abcisees sont les noeuds
        x = self.mesh.nodes
        u_exact = np.sin(np.pi*x)
        if (plot == True):
            plt.plot(x, u_tilde, label="solution approchee")
            plt.plot(x, u_exact, label="solution exacte")
            plt.suptitle("Elements finis P1")
            plt.legend()
            plt.show()
        return self.mesh.norm_P1(u_exact-u_tilde)[0]

    # La fonction solve_P2() sert a resoudre le cas P2 tandis que solve() resoud le cas P1
    def solve_P2(self, f, plot=True):
        # self.matrixA_P1()
        self.matrixA()
        # b = self.rhs_P1(f)
        b = self.rhs(f)
        u = nplin.solve(self.A.todense(), b)
        # u_tilde est le prologement de u par zero sur les bords
        # u_tilde = np.zeros(self.mesh.Nel+1)
        u_tilde = np.zeros(self.mesh.Ndof+2)
        u_tilde[1:-1] = u
        # Les abcisees sont les noeuds
        # x = self.mesh.nodes
        x = self.mesh.dof
        u_exact = np.sin(np.pi*x)
        if (plot == True):
            plt.plot(x, u_tilde, label="solution approchee")
            plt.plot(x, u_exact, label="solution exacte")
            plt.suptitle("Elements finis P2")
            plt.legend()
            plt.show()
        # return self.mesh.norm_P1(u_exact-u_tilde)[0]
        return self.mesh.norm(u_exact-u_tilde)

def f(x):
    return (np.pi**2 + 1)*np.sin(np.pi*x)

def test_question_4():
    Mh = mesh(5, -1, 1)
    Mh.init_uniform()
    # Mh.init_random()
    elt_finis = fem(Mh)
    # print("mesh's nodes\n", elt_finis.mesh.nodes)
    elt_finis.matrixA_P1()
    print("Mesh's A matrix\n", elt_finis.A.todense())

# test_question_4()

def test_question_6():
    Mh = mesh(5, -1, 1)
    Mh.init_uniform()
    # Mh.init_random()
    elt_finis = fem(Mh)
    b = elt_finis.rhs_P1(f)
    print("b manufacture par la methode des trapezes:\n", b)
    nodes = Mh.nodes
    h = Mh.h
    # Calcul de b par une integration normale, valeurs exactes
    b_real = (1+1/np.pi**2)*((np.sin(np.pi*nodes[1:-1])-np.sin(np.pi*nodes[0:-2]))/h[0:-1] - (np.sin(np.pi*nodes[2:])-np.sin(np.pi*nodes[1:-1]))/h[1:])
    print("b manufacture exacte :\n", b_real)

test_question_6()

def test_question_7():
    # Prendre xmin et xmax de sorte que u(xmin) = sin(pi*xmin) = u(xmax) = sin(pi*xmax) = 0
    Mh = mesh(100, -1, 1)
    Mh.init_random()
    elt_finis = fem(Mh)
    elt_finis.solve(f, True)

# test_question_7()

def test_question_8():
    Mh = mesh(500, -2, 2)
    Mh.init_uniform()
    # Mh.init_random()
    u = np.sin(np.pi*Mh.nodes)
    norme = Mh.norm_P1(u)[0]
    print("norme L2:", norme)

# test_question_8()

def test_question_9(xmin=-1, xmax=1):
    h_max = []
    normes_L2 = []
    for Nel in [20, 40, 80, 160, 320, 640]:
        Mh = mesh(Nel, xmin, xmax, 1)
        Mh.init_random()
        # Mh.init_uniform()
        h_max.append(np.max(Mh.h)) 
        elt_finis = fem(Mh)
        normes_L2.append(elt_finis.solve(f, False))
    print ("h_max\n", h_max)
    # plt.plot(h_max, normes_L2, label="Erreur en fonction du pas")
    # plt.plot(np.log10(h_max), np.log10(normes_L2), label="Erreur en fonction du pas")
    plt.loglog(h_max, normes_L2, label="Erreur logarithmique en fonction du pas")
    plt.suptitle("Elements finis P1")
    plt.legend()
    log_h_max = np.log10(np.array(h_max))
    log_normes_L2 = np.log10(np.array(normes_L2))
    max_slope = np.max((log_normes_L2[1:]-log_normes_L2[:-1])/(log_h_max[1:]-log_h_max[:-1]))
    print("Pour un maillage aleatoire, on obtient un ordre de convergence de:", max_slope)      # Car log(norme_L2) < max_slope*log(h_max) sur chaque intervalle
    print("L'ordre de convergence attendu est de:", 2)
    plt.show()

test_question_9()

def test_question_10(xmin=-1, xmax=1):
    Nels = [20, 40, 80, 160, 320, 640]
    cond_unif = []
    cond_rand = []
    for Nel in Nels:
        Mh = mesh(Nel, xmin, xmax)
        # Conditionnement pour le maiialge uniforme
        Mh.init_uniform()
        probleme = fem(Mh)
        probleme.matrixA_P1()
        cond_unif.append(nplin.cond(probleme.A.todense()))
        # Conditionnement pour le maiialge aleatoire
        Mh.init_random()
        probleme = fem(Mh)
        probleme.matrixA_P1()
        cond_rand.append(nplin.cond(probleme.A.todense()))
    print("Valeurs de Nel:\n", Nels)
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    print("Conditionnement de la matrice pour le maillage uniforme:\n", np.array(cond_unif))
    print("Conditionnement de la matrice pour le maillage aleatoire:\n", np.array(cond_rand))
    commentaire = """Commentaire: ------\nOn constate que le conditionnement est generalement plus grand quand le maillage est aleatoire.\
        \nLe conditionnement traduit la facilite a inverser une matrice; et d'ainsi resoudre le systeme lineaire associe.\
        \nIl est donc preferable de choisir un maillage uniforme pour avoir de meilleurs performances de calcul. """
    print(commentaire)   

# test_question_10()

def test_question_13():
    Mh = mesh(5, -2, 2)
    Mh.init_uniform()
    # Mh.init_random()
    print("nodes\n", Mh.nodes)
    print("dof\n", Mh.dof)

# test_question_13()

def test_question_14():
    Mh = mesh(4, -1, 1, 2)
    Mh.init_random()
    probleme = fem(Mh)
    probleme.matrixA()
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    print("matrice A:\n", probleme.A.todense())

# test_question_14()

def test_question_15():
    Mh = mesh(4, -1, 1, 2)
    Mh.init_random()
    probleme = fem(Mh)
    print("vecteur b:", probleme.rhs(f))

# test_question_15()

def test_question_16():
    Mh = mesh(500, -2, 2, 1)
    Mh.init_uniform()
    # Mh.init_random()
    u = np.sin(np.pi*Mh.dof)
    norme = Mh.norm(u)
    print("norme L2:", norme)

# test_question_16()

def test_question_17(xmin=-1, xmax=1):
    """ Affichage de la solution des elements finis P2 """
    Mh = mesh(25, xmin, xmax, 2)
    Mh.init_random()
    # Mh.init_uniform()
    probleme = fem(Mh)
    probleme.solve_P2(f, True)      # Pour afficher les resultat au moins une fois
    
    """ Etude de la convergence """
    print("ELEMENTS FINIS P1")
    test_question_9(xmin, xmax)
    
    print("ELEMENTS FINIS P2")
    h_max = []
    normes_L2 = []
    for Nel in [20, 40, 80, 160, 320, 640]:
        Mh = mesh(Nel, xmin, xmax, 2)
        Mh.init_random()
        # Mh.init_uniform()
        h_max.append(np.max(Mh.h))
        probleme = fem(Mh)
        normes_L2.append(probleme.solve_P2(f, False))
    # plt.plot(np.log(h_max), np.log(normes_L2), label="Erreur en fonction du pas")
    # plt.plot(np.log10(h_max), np.log10(normes_L2), label="Erreur en fonction du pas")
    plt.loglog(h_max, normes_L2, label="Erreur logarithmique en fonction du pas")
    plt.suptitle("Elements finis P2")
    plt.legend()
    log_h_max = np.log10(np.array(h_max))
    log_normes_L2 = np.log10(np.array(normes_L2))
    max_slope = np.max((log_normes_L2[1:]-log_normes_L2[:-1])/(log_h_max[1:]-log_h_max[:-1]))
    print("Pour un maillage aleatoire, on obtient un ordre de convergence de:", max_slope)      # Car log(norme_L2) < max_slope*log(h_max) sur chaque intervalle
    print("L'ordre de convergence attendu est de:", 2)
    commentaire = """Commentaire: ------\nOn obtient en general une meilleure approximation dans le cas P2. L'ordre de convergence calcule se rapproche considerablement de l'ordre de convergence attendu. """
    print(commentaire)
    plt.show()

test_question_17()