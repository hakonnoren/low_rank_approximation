import numpy as np
import matplotlib.animation
import matplotlib
import matplotlib.pyplot as plt

I = lambda n: np.identity(n)
inv = lambda A: np.linalg.inv(A)
fro_norm = lambda A : np.linalg.norm(A,ord='fro')

def check_orth(U):
    k = U.shape[1]
    return np.linalg.norm(U.T@U - np.identity(k),ord='fro')


def get_initial_decomp(A0,k):
    U0,S0,V0 = np.linalg.svd(A0)
    return U0[:,0:k],np.diag(S0[0:k]),V0.T[:,0:k]


def animate_matrix_func(At,T,fps,approx_input = False):
    
    if approx_input:
        snapshots = At
        m = At[0].shape[0]
    else:
        ts = np.arange(0,T,1/fps)
        snapshots = [At(t) for t in ts]
        m = At(0).shape[0]

    fig = plt.figure( figsize=(m,m) )
    im = plt.imshow(snapshots[0], 
                    cmap='hot', 
                    interpolation='nearest')

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )
        im.set_array(snapshots[i])
        return [im]

    anim = matplotlib.animation.FuncAnimation(
                                   fig, 
                                   animate_func, 
                                   frames = T * fps,
                                   interval = 1000 / fps, 
                                   )
    from IPython.display import HTML
    return HTML(anim.to_jshtml())