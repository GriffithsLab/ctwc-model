#! /usr/bin/env python

'''
Symmetric orthogonalization leakage correction
Paper:
Colclough, G. L., Brookes, M., Smith, S. M. and Woolrich, M. W., "A symmetric multivariate leakage correction for MEG connectomes," NeuroImage 117, pp. 439-448 (2015)
Translated from MATLAB:
https://github.com/OHBA-analysis/MEG-ROI-nets/blob/master/%2BROInets/symmetric_orthogonalise.m
Main function:
closest_ortho_matrix(dat)
dat: np.array with k x n shape
k: number of regions or sensors or sources of interest
n: number of samples
'''

import numpy as np
from numpy.linalg import svd, eig

def symmetric_ortho(dat):
    U,S,V = svd(dat,full_matrices=0)
    #rank checking
    S = S #this is different from matlab, as the diagonal component is automatically obtained in np svd, for matlab, we need diag(S)
    tol = max(dat.shape)*S[0]*(np.finfo(dat.dtype).eps) #tolerance level
    r = np.sum(S>tol) #number of S larger than tolerance
    isFullRank = (r >= dat.shape[0]) #dat.shape[0] here is number of ROIs
    #in matlab -> [U,S,V] = svd(a)
    #in python U, S, Vh = linalg.svd(a) and V = Vh.T
    '''if isFullRank == False:
        print('Warning: The input ts matrix is not full rank.')
        print(r)
        print(dat.shape[0])'''
    L = U.dot(np.conj(V))
    #W = V.T.dot(np.diag(1/S)).dot(V) #working weights, but not using
    return(L,isFullRank)

#fast_svd.py assumes dat is already transposed
def fast_svd(dat,N):
    #N = 1
    if N < dat.shape[1]:
        eigs2 = eig(dat.dot(dat.T))
        #eigs2 = eig(dat.dot(dat))
        d = max(eigs2[0])

        U = eigs2[1][:,0]

        S = np.sqrt(np.abs(d))
        V = dat.T.dot(U.dot(1/S))

        #U = dat.dot(V.dot(1/S))
    return(S) # for the purpose of tolerance finding, only need S, need a constant

def scale_cols(dat,s):
    newdat = dat * s
    return(newdat)

def reldiff(a,b):
    if a == 0 or b == 0:
        outcome = 0
    else:
        outcome = (2*np.abs(a-b) / (np.abs(a)+np.abs(b)))
    return(outcome)

def closest_ortho_matrix(dat,verbose=True):
    if verbose: print('Starting symmetric orthogonalization leakage correction')
    #dat = dat.T # data has to be transposed before processing as per their matlab code... weird practice
    itere = 0
    #dat = dat.astype(np.float64) #use double precision
    MAX_ITER = 2e2
    #slightly different because of single precision float32
    tol = np.finfo(dat.dtype).eps
    if verbose: print(tol)
    A_b = np.conj(dat)
    d = np.sqrt(np.sum(dat.conj()*A_b,axis=0))
    rho = []
    Ls = []
    '''
    dot(A,B) of same size is simply in matlab:
    sum(conj(A).*B)

    in python it is:
    np.sum(A.conj()*B, axis=0)
    '''
    isFullRank = True
    while itere < MAX_ITER:
        V, isFullRank = symmetric_ortho(scale_cols(dat,d))
        d = np.sum(A_b.conj()*V,axis=0)
        L = scale_cols(V,d)
        Ls.append(L)
        if isFullRank == False:
            if verbose: print('  No longer full rank. Optimal matrix reached at iteration %s' % (str(itere)))
            break
        E = dat - L
        rho.append(np.sqrt(np.sum(np.sum(E.conj()*np.conj(E),axis=0))))
        if itere > 0:
            val = reldiff(rho[itere],rho[itere-1])
            if verbose: print('  Iteration: %s\n   Tolerance: %s\n   Relative difference: %s\n   Rhos: %s' % (str(itere+1),str(tol),str(val),str(rho[itere])))
            if val <= tol:
                if verbose: print('  Optimal matrix reached at iteration %s\n  Tolerance: %s\n  Relative difference: %s\n  Rhos: %s' % (str(itere+1),str(tol),str(val),str(rho[itere])))
                break
        itere+=1
    if isFullRank == False:
        return(Ls[-1])
    else:
        return(L)
