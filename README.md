# Low rank approximation of time dependent matrices

This repository presents a method to obtain a low rank approximation of time dependent matrices; the dynamic low rank approximation proposed by Koch and Lubich [2]. 
In order to evaluate the performance of this method, an alternative method for computing the singular value decomposition, proposed by Golub and Kahan [1], is presented. 
Both methods are compared to the native SVD algorithm implemented in the Numpy library, relying on highly accurate and efficient LAPACK routines.

The numerical experiments are presented and performed in the Jupyter notebook, however all experiments are easily run from the command prompt by calling the appropriate functions. 

<a name="foot1">[1]</a>  G.H. Golub and C.F. Van Loan, *Matrix Computations*, Johns Hopkins Studies in the Mathematical Sciences.

<a name="foot1">[2]</a> O. Koch and C.Lubic *Dynamical low-rank approximartion* SIAM J. on Matrix Anal. and Appl. (2007), DOI 10.1137/050639703.
