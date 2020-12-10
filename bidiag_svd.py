import numpy as np
import matplotlib.pyplot as plt
from utils import I,inv,fro_norm,check_orth, get_initial_decomp


################## Lanczos bidiagonalization ###########

def lanczos_bidiag(A,k,b,reorth=True):
    
    """
    Lanczos bidiagonalization algorithm with reorthogonalization i.e. A = P_kBQ_k^T
    
    INPUT:
        A: m x n matrix to be diagonalized
        k: rank of the approximation
        b: m x 1 initial vector for selecting u_1,
        reorth: boolean to turn reorthogonalization on / off
        
    OUTPUT:
        P_k: k x m orthogonal matrix 
        B: k x k (lower) bidiagonal matrix
        Q_k: k x n orthogonal matrix
    
    """
    
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
        
        #Performs reorthogonalization
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
    
    """
    SVD step for the bidiagonal Lanczos SVD algorithm. 
    Given bidiagonal submatrix B, this algorithm performs 
    something similar to a truncated QR iteration on T = B^TB
    
    INPUT:
        B: n x n lower bidiagonal matrix
        
    OUTPUT:
        U: n x n Product of Givens rotations left multiplied to B
        V: n x n Product of Givens rotations right multiplied to B
    
    """
    
    
    n = B.shape[1]-1
    m = n-1
    
    #Truncation step:
    #approximating the eigenvalue of the last diagonal entry of the 2x2 trailing submatrix
    a0 = B[m,m]**2 + B[m,m-1]**2
    a1 = B[n,n]**2 + B[n,m]**2
    b0 = B[m,m]*B[n,m]
    
    d = 0.5*(a0-a1)
    mu = a1 + d - np.sign(d)*np.sqrt(d**2 + b0**2)
    
    y = B[0,0]**2 - mu
    z = B[0,0]*B[1,0]
    
    #Entries in a Givens rotation
    c = lambda z,y : y/np.sqrt(z**2+y**2)
    s = lambda z,y : z/np.sqrt(z**2+y**2)
    
    #Function yielding a Givens rotation matrix
    def G(k,z,y):
        R = np.identity(n+1)
        R[k:k+2,k:k+2] = np.array([[c(z,y),-s(z,y)],[s(z,y),c(z,y)]])
        return R
        
    V = np.identity(n+1)
    U = np.identity(n+1)
    
    for k in range(0,n):
        B = G(k,z,y).T@B
        U = U@G(k,z,y)
        y = B[k,k]
        z = B[k,k+1]
        B = B@G(k,z,y)
        V = V@G(k,z,y)
        
        if k < n-1:
            y = B[k+1,k]
            z = B[k+2,k]
            
    return U,V

def lanczos_svd(A,k,reorth = True,eig_vec = False):
    
    """
    Performs the full Lancozs bidiagonal SVD algorithm of A m x n. Yields the SVD: A = U\SigmaV.T
    For simplicity we assume m = n. 
    
    INPUT:
        A: n x n real matrix to be approximated
        k: rank of the apprimation
        reorth: boolean variable to turn reorthogonalization on / off.
        eig_vec: boolean variable which if True disables the storage and return of the orthogonal U,V
        
    OUTPUT:
        U: n x k: Orthogonal matrix approximating the right hand side eigenvectors of the SVD
        B: k x k: Diagonal matrix approximating the sigular values of the SVD. Note: these are not in descending order
        V: n x k: The orthogonal matrix approximating the left hand side eigenvectors of the SVD
    
    """
    
    n = A.shape[1]
    b = np.zeros(n)
    b[0] = 1
    
    #Get the Lanczos bidiagonal decomposition
    P1,B,Q1 = lanczos_bidiag(A,k,b,reorth=reorth)
    
    n = k-1
    tol = 1e-6
    q = 0
    r = 0
    
    if eig_vec:
        U = np.identity(k)
        V = np.identity(k)

    while q<=n:
        
        #Identifies the submatrices B22 and B33
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
        
        #Performs the SVD step on B22
        U22,V22 = bidiag_svd_step(B22)
        
        #Stores U,V
        if eig_vec:

            U_temp = np.identity(k)
            V_temp = np.identity(k)

            U_temp[n-r:n-q+1,n-r:n-q+1] = U22
            V_temp[n-r:n-q+1,n-r:n-q+1] = V22

            U = U@U_temp
            V = V@V_temp
            
        B[n-r:n-q+1,n-r:n-q+1] = U22.T@B22@V22

    if eig_vec:
        return P1@U,B,Q1@V
    else:
        return B

##################  end Lanczos bidiagonalization ###########
    
################## Lanczos bidiagonalization tests ###########    
    

def test_lanczos_svd(A,k,reorth = True):
    
    """
    Testing the bidiagonal Lanczos SVD W_k. 
    Find approximation error rank k approximation W_k and the best approximation X_k
    
    INPUT:
        A: n x n real matrix to be approximated
        k: rank of the apprimation
        reorth: boolean variable to turn reorthogonalization on / off.
        
    OUTPUT:
        e_sigma: 2 norm of the difference in singular values between LAPACK SVD and Lanczos SVD
        e_x: Frobenious norm of the approximation error of the LAPACK SVD of rank k: \|A - X_k\|_F
        e_bidiag: Frobenious norm of the approximation error of the Lanczos bidiagonal SVD of rank k: \|A - W_k\|_F
    
    """
    
    n = A.shape[0]
    Ul,Sl,Vl = lanczos_svd(A,k,reorth = reorth,eig_vec = True)
    
    #Get the LAPACK SVD implemented by Numpy
    Ub,Sb,Vb = get_initial_decomp(A,k)
    
    #Orders the singular values
    #For some reason unknown, there is an increase in the error if one also permutes U,V
    #as one would expect to be reasonable
    
    permutation = np.flip(np.argsort(np.diag(Sl)))
    sig_bidiag = np.diag(Sl)[permutation]

    sig_x = np.diag(Sb)
    
    e_sigma = np.linalg.norm(sig_bidiag - sig_x)
    e_orth_U = fro_norm(Ul.T@Ul - np.identity(k))
    e_orth_V = fro_norm(Vl.T@Vl - np.identity(k))
    
    e_x = fro_norm(A-Ub@Sb@Vb.T)
    e_bidiag = fro_norm(A-Ul@Sl@Vl.T)
    
    return e_sigma,e_x,e_bidiag

def experiment_lanczos(n,ks,matrix_generator):
    
    """
    Runs multiple Lanczos SVD tests.
    
    INPUT:
        n: size of the test matrix A n x n
        ks: list of several k: the rank of the approximation
        matrix_generator: a function which takes n and returns a matrix A n x n
        
    OUTPUT:
        Plot of the singular values of A n x n
        Plot of the approximation error for W_k and X_k for the different k in the k-list ks.
        Print of the approximation error for k = n of W_n
    
    """
    
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
    
    #Performs all experiments and collects approximation error
    for i in range(n_exp):
        e_sigma[i],e_x[i],e_bidiag[i] = test_lanczos_svd(A,ks[i])
    
    plt.plot(ks,e_sigma,label="$||\sigma_X - \sigma_W||_2$")
    plt.plot(ks,e_x,label="$||A - X_k||_F$")
    plt.plot(ks,e_bidiag,label="$||A - W_k||_F$")
    plt.xlabel("$k$")
    plt.title("Approximation error")
    plt.legend()
    plt.show()
    
    print("||A-W_n||_F = ", e_bidiag[-1])

def experiment_orthogonality(n,k,n_exp,matrix_generator):
    
    """
    Runs multiple experiments and plots the orthogonalization error
    
    INPUT:
        n: Size of the test matrix A n x n
        k: The rank of the approximation
        n_exp: number of repetitions of one orthogonality experiment
        matrix_generator: a function which takes n and returns a matrix A n x n
        
    OUTPUT:
        Plot of orthogonality error with reorthogonalization with error in V on the x-axis and error in U on the y-axis
        Plot of orthogonality error with reorthogonalization with error in V on the x-axis and error in U on the y-axis
    """
    
    e_v_re = [0]*n_exp
    e_u_re = [0]*n_exp
    e_v = [0]*n_exp
    e_u = [0]*n_exp
    
    for i in range(n_exp):
        A = matrix_generator(n)
        Ure,Bre,Vre = lanczos_svd(A,k,reorth = True,eig_vec = True)
        U,B,V = lanczos_svd(A,k,reorth = False,eig_vec = True)
        
        e_v_re[i] = check_orth(Vre)
        e_u_re[i] = check_orth(Ure)
        e_v[i] = check_orth(V)
        e_u[i] = check_orth(U)
        
        
    plt.plot(e_v_re,e_u_re,'r.')
    plt.title("Orthogonality error with reorthogonalization")
    plt.xlabel("$||V^TV - I||_F$")
    plt.ylabel("$||U^TU - I||_F$")
    plt.show()
    
    plt.plot(e_v,e_u,'b.')
    plt.title("Orthogonality error without reorthogonalization")
    plt.xlabel("$||V^TV - I||_F$")
    plt.ylabel("$||U^TU - I||_F$")
    plt.show()
    
    
################## end Lanczos bidiagonalization tests ########### 