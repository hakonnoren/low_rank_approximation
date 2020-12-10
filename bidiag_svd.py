import numpy as np
import matplotlib.pyplot as plt
from utils import I,inv,fro_norm,check_orth, get_initial_decomp


np.random.seed(0)

################## Lanczos bidiagonalization ###########

def lanczos_bidiag(A,k,b,reorth=True):
    m,n = A.shape
    B,P,Q = np.zeros((k,k)),np.zeros((m,k)),np.zeros((n,k))
    
    get_normalized = lambda v: (np.linalg.norm(v,ord=2), v/np.linalg.norm(v,ord=2))
    
    beta1,u1 = get_normalized(b)
    alpha1,v1 = get_normalized(A.T@u1)
    
    B[0,0] = alpha1
    P[:,0] = u1
    Q[:,0] = v1
    
    for i in range(1,k):
        beta2,u2 = get_normalized(A@v1-alpha1*u1)
        alpha2,v2 = get_normalized(A.T@u2-beta2*v1)
        
        if reorth:
            v2 *= alpha2
            for l in range(0,i):
                v2 -= (Q[:,l]@v2)*Q[:,l]
            alpha2,v2 = get_normalized(v2)
            
        B[i,i] = alpha2
        B[i,i-1] = beta2
        P[:,i] = u2
        Q[:,i] = v2
            
        beta1,alpha1,v1,u1 = beta2,alpha2,v2,u2
        
    return P,B,Q
    

def bidiag_svd_step(B):
    n = B.shape[1]-1

    m = n-1
    a0 = B[m,m]**2 + B[m,m-1]**2
    a1 = B[n,n]**2 + B[n,m]**2
    b0 = B[m,m]*B[n,m]
    
    d = 0.5*(a0-a1)
    mu = a1 + d - np.sign(d)*np.sqrt(d**2 + b0**2)
    
    y = B[0,0]**2 - mu
    z = B[0,0]*B[1,0]
    
    c = lambda z,y : y/np.sqrt(z**2+y**2)
    s = lambda z,y : z/np.sqrt(z**2+y**2)
    
    def G(k,z,y):
        R = np.identity(n+1)
        R[k:k+2,k:k+2] = np.array([[c(z,y),-s(z,y)],[s(z,y),c(z,y)]])
        return R
        
    Q = np.identity(n+1)
    P = np.identity(n+1)
    
    for k in range(0,n):
        B = G(k,z,y).T@B
        P = G(k,z,y).T@P
        y = B[k,k]
        z = B[k,k+1]
        B = B@G(k,z,y)
        Q = Q@G(k,z,y)
        
        if k < n-1:
            y = B[k+1,k]
            z = B[k+2,k]
            
    return P,Q

def lanczos_svd(A,k,reorth = True,eig_vec = False):
    n = A.shape[1]
    b = np.zeros(n)
    b[0] = 1
    P1,B,Q1 = lanczos_bidiag(A,k,b,reorth=reorth)
    B_old = np.copy(B)
    
    n = k-1
    tol = 1e-4
    count = 0
    q = 0
    r = 0
    
    if eig_vec:
        P2 = np.identity(k)
        Q2 = np.identity(k)

    while q<=n:
        
        for i in range(n):
            if np.abs(B[i+1,i]) <= tol*(np.abs(B[i,i]) + np.abs(B[i+1,i+1])):
                B[i+1,i] = 0
        q = 0

        for i in range(1,n):
            if B[n+1-i,n-i] == 0:
                q+=1
            else:
                break

        r = q

        for i in range(q+1,n+1):
            if B[n+1-i,n-i] == 0:
                break
            else:
                r += 1
                
        B33 = B[n-q+1:n+1,n-q+1:n+1]
        B22 = B[n-r:n-q+1,n-r:n-q+1]

        if B22.shape == (1,1):
            break
        
        P22,Q22 = bidiag_svd_step(B22)
        
        if eig_vec:

            P2_temp = np.identity(k)
            Q2_temp = np.identity(k)

            P2_temp[n-r:n-q+1,n-r:n-q+1] = P22
            Q2_temp[n-r:n-q+1,n-r:n-q+1] = Q22

            P2 = P2_temp@P2
            Q2 = Q2@Q2_temp
        
        B[n-r:n-q+1,n-r:n-q+1] = P22@B22@Q22
        
        count += 1

    if eig_vec:
        return P1@P2.T,B,Q2.T@Q1.T
    else:
        return B

##################  end Lanczos bidiagonalization ###########
    
################## Lanczos bidiagonalization tests ###########    
    

def test_lanczos_svd(A,k,reorth = True):
    n = A.shape[0]
    Ul,Sl,VlT = lanczos_svd(A,k,reorth = reorth,eig_vec = True)
    Ub,Sb,Vb = get_initial_decomp(A,k)
    
    sig_bidiag = np.flip(np.sort(np.diag(Sl)))
    sig_x = np.diag(Sb)
    
    
    e_sigma = np.linalg.norm(sig_bidiag - sig_x)
    e_orth_U = fro_norm(Ul.T@Ul - np.identity(k))
    e_orth_V = fro_norm(VlT@VlT.T - np.identity(k))
    
    e_x = fro_norm(A-Ub@Sb@Vb.T)
    e_bidiag = fro_norm(A-Ul@Sl@VlT)
    
    return e_sigma,e_orth_U,e_orth_V,e_x,e_bidiag

def experiment_lanczos(n,ks,matrix_generator):
    A = matrix_generator(n)
    n_exp = len(ks)
    
    e_sigma = [0]*n_exp
    e_x = [0]*n_exp
    e_bidiag = [0]*n_exp
    
    U,S,V = get_initial_decomp(A,n)
    x = np.arange(0,n)
    
    plt.plot(x,np.diag(S),'.',label = "$\sigma_i$")
    plt.title("Singular values")
    plt.legend()
    plt.show()
    
    
    for i in range(n_exp):
        e_sigma[i],e_orth_U,e_orth_V,e_x[i],e_bidiag[i] = test_lanczos_svd(A,ks[i])
    
    plt.plot(ks,e_sigma,label="$||\sigma_X - \sigma_W||_2$")
    plt.plot(ks,e_x,label="$||A - X_k||_F$")
    plt.plot(ks,e_bidiag,label="$||A - W_k||_F$")
    plt.xlabel("$k$")
    plt.title("Approximation error")
    plt.legend()
    plt.show()
    
    print("||A-W_n|| = ", e_bidiag[-1])

def experiment_orthogonality(n,k,n_exp,matrix_generator):
    e_q_re = [0]*n_exp
    e_p_re = [0]*n_exp
    e_q = [0]*n_exp
    e_p = [0]*n_exp
    
    for i in range(n_exp):
        A = matrix_generator(n)
        Pre,Bre,QTre = lanczos_svd(A,k,reorth = True,eig_vec = True)
        P,B,QT = lanczos_svd(A,k,reorth = False,eig_vec = True)
        
        e_q_re[i] = check_orth(QTre.T)
        e_p_re[i] = check_orth(Pre)
        e_q[i] = check_orth(QT.T)
        e_p[i] = check_orth(P)
        
        
    plt.plot(e_q_re,e_p_re,'r.')
    plt.title("Orthogonality error with reorthogonalization")
    plt.xlabel("$||Q^TQ - I||_F$")
    plt.ylabel("$||P^TP - I||_F$")
    plt.show()
    
    plt.plot(e_q,e_p,'b.')
    plt.title("Orthogonality error without reorthogonalization")
    plt.xlabel("$||Q^TQ - I||_F$")
    plt.ylabel("$||P^TP - I||_F$")
    plt.show()
    
    
################## end Lanczos bidiagonalization tests ########### 