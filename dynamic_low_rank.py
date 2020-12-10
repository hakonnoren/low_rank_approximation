import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import time
import random

from utils import I,inv,fro_norm,check_orth, get_initial_decomp
from utils import animate_matrix_func
from bidiag_svd import lanczos_svd


################### Cayley transformation ###########################


def cay(B):
    n = B.shape[0]
    return inv(I(n) - 0.5*B)@(I(n) + 0.5*B)

def cay_naive(FU,U):
    B = FU@U.T - U@FU.T
    n = B.shape[0]
    return inv(I(n) - 0.5*B)@(I(n) + 0.5*B)

def cay_test_1(FU,U):
    k = U.shape[1]
    m = U.shape[0]    
    D = np.concatenate([U,FU],axis = 1)
    C = np.concatenate([FU,-U],axis = 1)
    
    DTC = D.T@C

    return I(m) + C@inv(I(2*k) - 0.5*DTC)@D.T

def cay_test_2(FU,U):
    k = U.shape[1]
    m = U.shape[0]    
    
    D = np.concatenate([U,FU],axis = 1)
    C = np.concatenate([FU,-U],axis = 1)
    
    K4 = inv(I(k) - 0.25*FU.T@FU)
    K3 = 2*(K4 - I(k))
    K2 = inv(-2*I(k) - 0.5*FU.T@FU)
    K1 = -2*K2
    
    K12 = np.concatenate([K1,K2],axis=1)
    K34 = np.concatenate([K3,K4],axis=1)
    K = np.concatenate([K12,K34])
    
    return I(m)+C@K@D.T


def cay_test_3(FU,U):
    k = U.shape[1]
    Q,R = np.linalg.qr(FU)
    K = np.zeros((2*k,2*k))
    K[0:k,k:] = -R.T
    K[k:,0:k] = R
    L = np.concatenate([U,Q],axis = 1)
    
    return L@cay(K)@L.T
    
    
################### end Cayley transformation ###########################


################### rkII ###########################


def rk2(U0,S0,V0,Adot,h,t,select_cay = "naive"):
    
    cay = lambda d:0
    m = U0.shape[0]
    n = V0.shape[0]
    
    FU = (I(m) - U0@U0.T)@Adot(t)@V0@inv(S0)
    FV = (I(n) - V0@V0.T)@Adot(t).T@U0@inv(S0).T
    
    if select_cay == "naive":
        cay = cay_naive
    elif select_cay == "test_1":
        cay = cay_test_1
    elif select_cay == "test_2":
        cay = cay_test_2
    elif select_cay == "test_3":
        cay = cay_test_3
    elif select_cay == "test_accuracy":

        n_t1 = 0.5*np.linalg.norm(cay_naive(FU,U0) - cay_test_1(FU,U0),ord='fro')
        n_t1 += 0.5*np.linalg.norm(cay_naive(FV,V0) - cay_test_1(FV,V0),ord='fro')
        n_t2 = 0.5*np.linalg.norm(cay_naive(FU,U0) - cay_test_2(FU,U0),ord='fro')
        n_t2 += 0.5*np.linalg.norm(cay_naive(FV,V0) - cay_test_2(FV,V0),ord='fro')
        n_t3 = 0.5*np.linalg.norm(cay_naive(FU,U0) - cay_test_3(FU,U0),ord='fro')
        n_t3 += 0.5*np.linalg.norm(cay_naive(FV,V0) - cay_test_3(FV,V0),ord='fro')
        
        print("Accuracy of method 1: ", n_t1)
        print("Accuracy of method 2: ", n_t2)
        print("Accuracy of method 3: ", n_t2)
        
        return   
    
    K1S = h*U0.T@Adot(t)@V0
    Shlf = S0 + 0.5*K1S
    
    K1U = h*(FU@U0.T - U0@FU.T)
    K1V = h*(FV@V0.T - V0@FV.T)
    
    S1est = Shlf + 0.5*K1S
    
    U1est = cay(h*FU,U0)@U0
    V1est = cay(h*FV,V0)@V0
    
    Uhlf = cay(h*0.5*FU,U0)@U0
    Vhlf = cay(h*0.5*FV,V0)@V0
    
    FUhlf = (I(m) - Uhlf@Uhlf.T)@Adot(t+0.5*h)@Vhlf@inv(Shlf)
    FVhlf = (I(n) - Vhlf@Vhlf.T)@Adot(t+0.5*h).T@Uhlf@inv(Shlf).T
    
    K2S = h*Uhlf.T@Adot(t + 0.5*h)@Vhlf
    K2U = h*(FUhlf@Uhlf.T - Uhlf@FUhlf.T)
    K2V = h*(FVhlf@Vhlf.T - Vhlf@FVhlf.T)
    
    S1 = S0 + K2S
    U1 = cay(h*FUhlf,Uhlf)@U0
    V1 = cay(h*FVhlf,Vhlf)@V0
    
    return S1est,U1est,V1est,S1,U1,V1

def rk2_old(U0,S0,V0,Adot,h,t):

    m = U0.shape[0]
    n = V0.shape[0]
    
    FU = (I(m) - U0@U0.T)@Adot(t)@V0@inv(S0)
    FV = (I(n) - V0@V0.T)@Adot(t).T@U0@inv(S0).T
    
    K1S = h*U0.T@Adot(t)@V0
    Shlf = S0 + 0.5*K1S
    
    K1U = h*(FU@U0.T - U0@FU.T)
    K1V = h*(FV@V0.T - V0@FV.T)
    
    S1est = Shlf + 0.5*K1S
    U1est = cay(K1U)@U0
    V1est = cay(K1V)@V0
    
    Uhlf = cay(0.5*K1U)@U0
    Vhlf = cay(0.5*K1V)@V0
    
    FUhlf = (I(m) - Uhlf@Uhlf.T)@Adot(t+0.5*h)@Vhlf@inv(Shlf)
    FVhlf = (I(n) - Vhlf@Vhlf.T)@Adot(t+0.5*h).T@Uhlf@inv(Shlf).T
    
    K2S = h*Uhlf.T@Adot(t + 0.5*h)@Vhlf
    K2U = h*(FUhlf@Uhlf.T - Uhlf@FUhlf.T)
    K2V = h*(FVhlf@Vhlf.T - Vhlf@FVhlf.T)
    
    S1 = S0 + K2S
    U1 = cay(K2U)@U0
    V1 = cay(K2V)@V0
    
    return S1est,U1est,V1est,S1,U1,V1


################### end rkII ###########################


################### ODE solver ###########################


def get_Ydot(U,S,V,Adot):
    m = U.shape[0]
    n = V.shape[0]
    Sdot = U.T@Adot@V
    Udot = (I(m) - U@U.T)@Adot@V@inv(S)
    Vdot = (I(n) - V@V.T)@Adot.T@U@inv(S).T
    Ydot = Udot@S@V.T + U@Sdot@V.T + U@S@Vdot.T
    return Ydot

def solve_ode(t0,h0,T,U0,S0,V0,Adot,A,k,tol,compare_W,store = False,check_error = True):
    
    if check_error:
        e_AYt = []
        e_AXt = []
        e_YXt = []
        e_AdotYdot = []
        e_Wt = []
        sigmas_X = np.zeros((1,k))
        sigmas_Y = np.zeros((1,k))
        
    if store:
        n = A(0).shape[0]
        Ys =  []
        
    t = t0
    h = h0
    count = 0
    
    def take_step(U0,S0,V0,h,t):
        Sest,Uest,Vest,S1,U1,V1 = rk2(U0,S0,V0,Adot,h,t,select_cay = "test_1")
        sigma = np.linalg.norm(U1@S1@V1.T - Uest@Sest@Vest.T,ord='fro')
        t += h
        t1,h1 = step_control(sigma,tol,h,t)
        return U1,S1,V1,t,t1,h1
    
    def get_X(A,k):
        UT,ST,VT = get_initial_decomp(A,k)
        return UT@ST@VT.T,ST
    
    
    while t <= T:
        
        U1,S1,V1,t,t1,h1 = take_step(U0,S0,V0,h,t)            
        t2,h2 = t1,h1
        while t1 < t and count <= 3:
            t,h = t2,h2
            U1,S1,V1,t,t1,h1 = take_step(U1,S1,V1,h,t)
            t2,h2 = t1,h1
            count += 1
        count = 0
        
        if check_error:
            Y = U1@S1@V1.T
            X,S = get_X(A(t),k)
            e_AYt.append(fro_norm(A(t) - Y))
            e_AXt.append(fro_norm(A(t) - X))
            e_YXt.append(fro_norm(X-Y))
            e_AdotYdot.append(fro_norm(get_Ydot(U1,S1,V1,Adot(t)) - Adot(t)))
            
            sigmas_X = np.concatenate([sigmas_X,[np.diag(S)]])
            sigmas_Y = np.concatenate([sigmas_Y,[np.diag(S1)]])
            
            if compare_W:
                Ul,Sl,VlT = lanczos_svd(A(t),k,reorth = True,eig_vec = True)
                e_Wt.append(fro_norm(A(t) - Ul@Sl@VlT))
                
        t,h,U0,S0,V0 = t1,h1,U1,S1,V1
        
        if store:
            Y = U1@S1@V1.T
            Ys.append(Y)

    if t > T:
        U1,S1,V1,t,t1,h1 = take_step(U0,S0,V0,T-t,t-h)
        
    if store:
        return np.stack(Ys)

    if compare_W:
        return U1,S1,V1,[e_AYt,e_AXt,e_YXt,e_AdotYdot,e_Wt],sigmas_X[1:,:],sigmas_Y[1:,:]
    else:
        return U1,S1,V1,[e_AYt,e_AXt,e_YXt,e_AdotYdot],sigmas_X[1:,:],sigmas_Y[1:,:]

def step_control(sigma,tol,h,t):
    R = 1
    if sigma > tol:        
        t1 = t-h
        h1 = 0.5*h
        return t1,h1
    else:
        h1 = h
        t1 = t
        if sigma > 0.5*tol:
            R = (tol/sigma)**(1/3)
            if R > 0.9 or R < 0.5:
                R = 0.7
        else:
            if sigma > (1/16)*tol:
                R = 1
            else:
                R = 2
        return t1,R*h1

################### end ODE solver ###########################


def plot_error(e,T):
    n = len(e[0])
    x = np.linspace(0,T,n)
    labels = ["$|| A-Y_k||_F$","$||A-X_k||_F$","$||Y_k-X_k||_F$","$|| \dot A - \dot Y_k||_F$","$||A - W_k||_F$"]
    for i,es in enumerate(e):
        plt.plot(x,es,label = labels[i])
        
    plt.legend()
    plt.title("Dynamic low rank approximation error")
    plt.xlabel("t")
    plt.show()
    
def plot_singular_values(sigmas_X,sigmas_Y,T):
    n,k = sigmas_X.shape
    x = np.linspace(0,T,n)
    #labels = [f"$\sigma_{i}$" for i in range(1,k+1)]
    
    for i in range(k):
        plt.plot(x,sigmas_X[:,i])
        plt.plot(x[::40],sigmas_Y[::40,i],marker='.',linestyle="None")
        
    plt.title("Singular values for $X_k$ and $Y_k$")
    plt.show()
    

def run_test(A,Adot,k,h0,t0,T,tol,plot_singular = False,compare_W = True):
    U0,S0,V0 = get_initial_decomp(A(t0),k)
    U1,S1,V1,e,sigmas_X,sigmas_Y = solve_ode(t0,h0,T,U0,S0,V0,Adot,A,k,tol,compare_W)
    plot_error(e,T)
    if plot_singular:
        plot_singular_values(sigmas_X,sigmas_Y,T)
        
def animate_experiment(A,Adot,k,h0,t0,T,tol):
    U0,S0,V0 = get_initial_decomp(A(t0),k)
    Ys = solve_ode(t0,h0,T,U0,S0,V0,Adot,A,k,tol,compare_W = False,store = True,check_error = False)
    n_frames = Ys.shape[0]
    fps = n_frames//T
    return animate_matrix_func(Ys,T,fps,approx_input = True)
    
    
def experiment_cayley(ms,k):
    h0 = 1e-1
    t = 3*h0
    
    n_exp = len(ms)
    
    t_naive = [0]*n_exp
    t_1 = [0]*n_exp
    t_2 = [0]*n_exp
    t_3 = [0]*n_exp
    
    times = [t_naive,t_1,t_2,t_3]
    test_names = ["naive","test_1","test_2","test_3"]
    order = [0,1,2,3]
    
    for i,m in enumerate(ms):
        
        A,Adot = get_A1(m**2,m,k)
        U0,S0,V0 = get_initial_decomp(A(0),k)
        
        random.shuffle(order)
        for e in order:
        
            t0 = time.perf_counter()
            rk2(U0,S0,V0,Adot,h0,t,select_cay=test_names[e])
            times[e][i] = time.perf_counter() - t0
            
    for i in range(4):
        plt.semilogy(ms,times[i],label=test_names[i])
    plt.legend()
    plt.xlabel("$n = m^2$")
    plt.ylabel("$t \; (s)$")
    plt.show()
    
    rk2(U0,S0,V0,Adot,h0,t,select_cay="test_accuracy")
    

    
################### GENERATORS FOR pt II ###########################
    
def get_A0(m,k):
    return np.random.rand(m,k)@np.random.rand(m,k).T

def get_Laplacian(n):
    L = np.zeros((n**2,n**2))
    I = np.identity(n)
    B = np.zeros((n,n))
    for d,i in zip([1,-4,1],[-1,0,1]):
        B += np.diag([d]*(n-np.abs(i)),i) 
    for k in range(1,n):
        L[(k-1)*n:k*n,(k-1)*n:k*n] = B
        L[(k-1)*n:k*n,k*n:(k+1)*n] = I
        L[k*n:(k+1)*n,(k-1)*n:k*n] = I
    L[(n-1)*n:n**2,(n-1)*n:n**2] = B
    return -L/(n**2)


def get_A1(m,l,k):
    A0 = get_A0(m,k+2)
    B = get_Laplacian(l)
    
    A1 = lambda t : expm(t*B)@A0
    A1dot = lambda t : B@A1(t)
    
    return A1,A1dot

################### end GENERATORS FOR pt II ###########################
    

################### GENERATORS FOR pt III ###########################
    
def get_Ai(epsilon,m,seed = 0):
    np.random.seed(seed)
    A = np.zeros((m**2,m**2))
    A[0:m,0:m] =  np.random.rand(m,m)*0.5 + I(m)
    return A + epsilon*np.random.rand(m**2,m**2)
        
def get_Qi(i,m):
    Ti = np.zeros((m**2,m**2))
    
    if i == 1:
        diags = zip([1,0,-1],[-1,0,1])
    elif i == 2:
        diags = zip([1,0.5,0,-0.5,-1],[-2,-1,0,1,2])
    
    for d,j in diags:
        Ti += np.diag([d]*(m**2-np.abs(j)),j) 
        
    Qi = lambda t: np.identity(m**2)@expm(t*Ti)
    
    return Qi,Ti
    
def get_A2(epsilon,m):    
    A2_1 = get_Ai(epsilon,m,seed = 3)
    A2_2 = get_Ai(epsilon,m,seed = 4)
    
    Q1,T1 = get_Qi(1,m)
    Q2,T2 = get_Qi(2,m)
    
    A2 = lambda t: Q1(t)@(A2_1 + np.exp(t)*A2_2)@Q2(t).T
    A2dot = lambda t: T1@A2(t) + A2(t)@T2.T + Q1(t)@(np.exp(t)*A2_2)@Q2(t).T
    
    return A2,A2dot


################### end GENERATORS FOR pt III ###########################


################### GENERATORS FOR pt IV ###########################

def get_A3(epsilon,m):
    #m = 10
    
    A2_1 = get_Ai(epsilon,m,seed = 1)
    A2_2 = get_Ai(epsilon,m,seed = 2)
    
    Q1,T1 = get_Qi(1,m)
    Q2,T2 = get_Qi(2,m)
    
    A2 = lambda t: Q1(t)@(A2_1 + np.cos(t)*A2_2)@Q2(t).T
    A2dot = lambda t: T1@A2(t) + A2(t)@T2.T - Q1(t)@(np.sin(t)*A2_2)@Q2(t).T
    
    return A2,A2dot


################### end GENERATORS FOR pt IV ###########################