import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from src.matfuncb.krylov_basis import arnoldi
from src.matfuncb.matfuncb import matfuncb
from test_np.utils.matrix_factory import get_symmetric_matrix_by_evals
from semi_md.matrix_functions import sinc_sqrtm, sinc_sqrtm_non_sym, sinc_sqrtm_non_clip
from src.matfuncb.error_bounds import *


# Investigate how the error bounds for the Lanczos method look for the rough matrix we have in MD

def sinc_sqrt(w):
    return np.sinc(np.sqrt(w) / np.pi)


if __name__ == "__main__":
    # evals = np.array([-2.96080487e-10, -2.88257522e-10, -2.25755073e-10, -2.14112029e-10,
    #    -1.99241086e-10, -1.94973036e-10, -1.70022735e-10, -1.54998901e-10,
    #    -1.41914174e-10, -1.35449631e-10, -1.33769510e-10, -1.22004599e-10,
    #    -1.20919393e-10, -1.18216721e-10, -1.15719487e-10, -1.10793316e-10,
    #    -1.10093111e-10, -1.03136887e-10, -9.70804929e-11, -9.13786085e-11,
    #    -8.87151977e-11, -8.71486879e-11, -8.40187587e-11, -8.16146463e-11,
    #    -7.99983429e-11, -7.67988712e-11, -7.41844970e-11, -6.83307989e-11,
    #    -6.44629934e-11, -6.03643401e-11, -5.73149016e-11, -5.64460579e-11,
    #    -5.34011197e-11, -5.10594581e-11, -4.55296383e-11, -4.36406256e-11,
    #    -4.01048626e-11, -3.90853633e-11, -3.83414507e-11, -3.54115504e-11,
    #    -3.36201642e-11, -3.15804179e-11, -2.79977161e-11, -2.49569489e-11,
    #    -2.16193607e-11, -1.64686201e-11, -1.53448804e-11, -1.16526890e-11,
    #    -9.19075203e-12, -8.13343674e-12, -4.69089455e-12, -3.08865851e-12,
    #     4.36280004e-13,  1.01887214e-12,  3.42618576e-12,  6.24653073e-12,
    #     1.20830526e-11,  1.32899834e-11,  1.87171812e-11,  2.25819898e-11,
    #     2.95662497e-11,  3.60406579e-11,  4.33123963e-11,  5.17097007e-11,
    #     5.75729117e-11,  7.87087185e-11,  8.87656271e-11,  1.05905123e-10,
    #     1.36606644e-10,  9.03990048e+03,  1.18839755e+04,  1.49054264e+04,
    #     1.71305935e+04,  1.94291088e+04,  2.42077381e+04,  2.60022631e+04,
    #     3.06961436e+04,  3.38481198e+04,  3.83366526e+04,  3.87971128e+04,
    #     4.81799229e+04,  5.02595063e+04,  5.46617202e+04,  6.47976743e+04,
    #     6.71085001e+04,  6.98895615e+04,  7.49349628e+04,  7.62598595e+04,
    #     3.08178987e+05,  3.08205577e+05,  3.15559676e+05,  3.17043481e+05,
    #     3.17218300e+05,  3.30322674e+05,  3.31058021e+05,  3.31090131e+05,
    #     3.31153684e+05,  3.32638205e+05,  3.32670016e+05,  3.33431375e+05,
    #     3.50105936e+05,  3.50507280e+05,  4.16995658e+05,  4.42318326e+05,
    #     4.72114910e+05])
    evals = np.array([ 1.35198226e+06,  9.64169943e+05,  9.63565003e+05,  9.15377733e+05,
        9.09163404e+05,  8.81238675e+05,  8.67572893e+05,  8.53605808e+05,
        8.33118770e+05,  8.25384113e+05,  8.45734448e+05,  7.73713122e+04,
        1.00405234e+05,  1.07940498e+05,  1.55371311e+05,  1.33942262e+05,
        7.83904016e+05,  1.37521489e+05,  1.75617778e+05,  1.85112980e+05,
        2.09970884e+05,  2.31174244e+05,  2.41676822e+05,  3.17869540e+05,
        3.47143916e+05,  3.57276528e+05,  3.92183002e+05,  3.72261809e+05,
        7.25558748e+05,  4.00404558e+05,  4.26269438e+05,  4.59561142e+05,
        4.82419594e+05,  4.98398260e+05,  2.80462808e+05,  2.76864456e+05,
        5.19864163e+05,  5.33620377e+05,  6.65668723e+05,  6.38575336e+05,
        6.50278050e+05,  6.45116666e+05,  6.89242030e+05,  7.13237051e+05,
        7.30634467e+05,  7.29480182e+05,  7.31269658e+05,  4.32013238e+05,
        2.88351440e+05,  2.64532331e+05,  4.43875982e+05,  7.02545826e+05,
        6.93957583e+05,  6.95926082e+05,  5.85513896e+05,  5.56040594e+05,
        5.54846788e+05,  5.70762125e+05,  6.05827462e+05,  6.03493464e+05,
        5.96058060e+05,  5.39067260e+05,  7.06316344e+05,  5.95698369e+05,
        5.96871102e+05,  7.03698350e+05, -9.81875670e-10, -8.79186533e-10,
       -8.06774116e-10, -6.58101872e-10, -6.45422501e-10, -6.36302228e-10,
       -5.73707020e-10, -4.98804443e-10, -4.84071218e-10, -4.27029956e-10,
       -4.27029956e-10,  1.18821928e-10,  1.08957094e-10, -3.84776971e-10,
       -3.52616822e-10, -3.52616822e-10, -3.55498812e-10, -3.43058013e-10,
       -3.14246405e-10, -3.14246405e-10, -2.81905865e-10, -2.81905865e-10,
       -2.65286291e-10, -2.65286291e-10, -2.59076838e-10,  5.67808114e-11,
        5.67808114e-11,  4.08706666e-11,  4.08706666e-11, -2.23009485e-10,
       -2.23009485e-10, -2.46242811e-10, -2.46242811e-10, -2.40893320e-10,
       -2.40893320e-10, -2.22136787e-10, -2.22136787e-10, -2.06245350e-10,
       -2.06245350e-10, -1.99126722e-10,  8.95103307e-12,  8.95103307e-12,
       -1.56777429e-11, -1.56777429e-11,  2.96580281e-11,  9.56485916e-12,
        9.56485916e-12,  9.79272904e-12,  9.79272904e-12, -1.45485476e-10,
       -1.45485476e-10, -1.72789696e-10, -1.72789696e-10, -1.76676655e-10,
       -1.76676655e-10, -1.73641766e-10,  4.24626873e-12, -2.10805459e-11,
       -2.10805459e-11, -1.55280240e-10, -1.55280240e-10, -1.66991030e-10,
       -1.57838830e-10, -6.94522571e-11, -6.94522571e-11, -5.29991351e-11,
       -5.29991351e-11, -1.62308622e-11, -1.62308622e-11, -1.44619258e-10,
       -1.30369615e-10, -1.30369615e-10, -1.51335602e-11, -1.51335602e-11,
       -1.55926303e-11, -1.30553827e-10, -2.07908469e-11, -2.07908469e-11,
       -1.25591936e-10, -1.25591936e-10, -5.19165842e-11, -5.19165842e-11,
       -8.64861147e-11, -8.64861147e-11, -1.22126698e-10, -6.37401556e-11,
       -6.37401556e-11, -1.05677982e-10, -1.05677982e-10, -1.09772482e-10,
       -1.09772482e-10, -3.26675524e-11, -3.26675524e-11, -4.17695635e-11,
       -4.17695635e-11, -3.70527451e-11, -3.95696843e-11, -4.13885385e-11,
       -4.13885385e-11, -9.21676238e-11, -9.21676238e-11, -9.78779944e-11,
       -4.63631855e-11, -4.63631855e-11, -8.44255937e-11, -8.44255937e-11,
       -6.40752346e-11, -6.40752346e-11, -5.15432360e-11, -9.04323836e-11,
       -9.04323836e-11, -8.72485086e-11, -8.72485086e-11, -7.04265194e-11,
       -7.04265194e-11, -7.75841509e-11, -7.75841509e-11])
    evals = np.clip(evals, 0, np.inf)
    #evals[random.sample(range(38), k=18)] = 0
    evals = np.sort(evals)[::-1]  # sort returns ascending
    n = min(len(evals), 80)
    Omega2, S = get_symmetric_matrix_by_evals(evals, True, "./precalculated", load=True, save=False)
    t = 0.003  # seconds
    b = np.random.random(len(evals))

    f =  sinc_sqrt # np.exp
    fm_sparse = sinc_sqrtm #scipy.sparse.linalg.expm
    fm = sinc_sqrtm #scipy.linalg.expm
    fm_non_symmetric = sinc_sqrtm_non_sym #scipy.linalg.expm
    plt.title(r"Error bounds for $sinc(t\sqrt{A})$ with $\lambda(tA)_max=" + f"{t* evals[0]}" + r"$, $" + f"n={len(evals)}" + r"$.")
    # plt.title(
    #     r"Error bounds for $exp(-t\sqrt{A})$ with $\lambda(tA)_max=" + f"{t * evals[0]}" + r"$, $" + f"n={len(evals)}" + r"$.")

    sign = 1
    Omega2 = Omega2
    evals = evals

    sm_eval = evals[-1] #evals[0]
    la_eval = evals[0] #evals[-1]
    shift = -1

    exact = S  @ np.diag(f(t ** 2 * evals)) @ S.T @ b
    max_acc = 1e-16
    errors = []
    i_s = []
    for i in range(2, 100):  # int(len(evals) / 2)):
        app, info = matfuncb(t ** 2 * Omega2, b, fm_sparse, k=i, symmetric=True)
        #app, info = matfuncb(t ** 2 * Omega2, b, sinc_sqrtm_non_clip, k=i, symmetric=True)
        err = np.linalg.norm(app - exact)
        errors.append(err)
        i_s.append(i)
        if err <= max_acc:
            break
    plt.plot(i_s, errors, label="Lanczos")
   # print(hochbruck_lubich(-evals[0], t**2, 20))
    # print(hochbruck_lubich(-evals[0], t**2, n))
    plt.plot(*hochbruck_lubich(-1 * max(np.abs(la_eval), np.abs(sm_eval)), t**2, n), label="HL", linestyle="solid")
    if np.abs(la_eval) * t**2 <= 1:
        plt.plot(*saad(np.abs(la_eval), t**2, n), label="Saad", linestyle="dashed")
    # plt.plot(*ye(evals[-1], evals[0], t=t**2, n=n, alpha=0), label="Ye 0", linestyle="dotted")
    # plt.plot(*ye(evals[-1], evals[0], t=t**2, n=n, alpha=0.5 * t**2), label="Ye 0.5t", linestyle="dotted")
    # plt.plot(*ye(evals[-1], evals[0], t=t**2, n=n, alpha=t**2), label="Ye t", linestyle="dotted")
    plt.plot(*chen_musco(t**2 * sm_eval, t**2 * la_eval, w=shift, n=n, f=f), label="CGMM", linestyle="dashdot")

    w = b / np.linalg.norm(b)
    (w, V, T, breakdown) = arnoldi(t**2 * Omega2, w, 41, trunc=2)
    if T.shape[0] >= 10:
        plt.plot(*chen_musco_post(T[:10, :10], w=shift, f=f, fix_0_eval=True), "or", label="CGMM")
        lower, upper = afanasjew_post(T, V, t**2*Omega2, m=10, f=fm_non_symmetric)
        plt.plot(10, lower, "ob", label="AEEG")
        plt.plot(10, upper, "og", label="AEEG")

    if T.shape[0] >= 25:
        plt.plot(*chen_musco_post(T[:25, :25], w=shift, f=f, fix_0_eval=True), "or")
        lower, upper = afanasjew_post(T, V, t ** 2 * Omega2, m=25, f=fm_non_symmetric)
        plt.plot(25, lower, "ob")
        plt.plot(25, upper, "og")
    if T.shape[0] >= 40:
        lower, upper = afanasjew_post(T, V, t ** 2 * Omega2, m=40, f=fm_non_symmetric)
        plt.plot(40, lower, "ob")
        plt.plot(40, upper, "og")
        plt.plot(*chen_musco_post(T[:40, :40], w=shift, f=f, fix_0_eval=True), "or")
    plt.yscale("log")
    plt.ylim(top=10, bottom=1e-18)
    plt.legend()
    plt.show()
