from scipy.sparse import find
from pycvxcluster.algos.helpers import fnorm
from pycvxcluster.algos.helpers import prox_l2
from pycvxcluster.algos.helpers import proj_l2
from pycvxcluster.algos.admm import admm_l2
import numpy as np
import math
import time

EPS = np.finfo(np.float64).eps


class AInput:
    def __init__(self, node_arc_matrix):
        self.A0 = node_arc_matrix
        self.AT = node_arc_matrix.T
        self.ATAmatT = (self.A0.T @ self.A0).T

    def Amap(self, x):
        # return x @ self.A0
        return (self.AT @ x.T).T

    def ATmap(self, x):
        # return x @ self.A0.T
        return (self.A0 @ x.T).T

    def ATAmap(self, x):
        # return x @ self.ATAmat
        return (self.ATAmatT @ x.T).T


class Dim:
    def __init__(self, X, weight_vec):
        self.n = X.shape[1]
        self.d = X.shape[0]
        self.E = len(weight_vec)


def ssnal_wrapper(X, weight_vec, node_arc_matrix, **kwargs):
    Ain = AInput(node_arc_matrix)
    dim = Dim(X, weight_vec)
    return ssnal(Ain, X, dim, weight_vec, **kwargs)


def ssnal(
    Ainput,
    data,
    dim,
    weight_vec,
    sigma=1,
    maxiter=100,
    stoptol=1e-6,
    scale=0,
    use_kkt=1,
    admm_iter=0,
    ncgtolconst=0.5,
    verbose=1,
    x0=None,
    z0=None,
    y0=None,
):
    if verbose > 0:
        print("Starting SSNAL...")
    start_time = time.perf_counter()
    if x0 is None:
        xi = data
    else:
        xi = x0
    if z0 is None:
        z = np.zeros((dim.d, dim.E))
    else:
        z = z0
    if y0 is None:
        y = Ainput.Amap(xi)
    else:
        y = y0

    if admm_iter > 0:
        xi, y, z, admm_status, admm_eta, *_ = admm_l2(
            data,
            Ainput.A0,
            weight_vec,
            max_iter=admm_iter,
            sigma=sigma,
            rho=1.618,
            stop_tol=stoptol,
            verbose=verbose,
            xi0=xi,
            x0=z,
            y0=y,
        )
        if admm_eta < stoptol:
            print("ADMM converged in {} iterations.".format(admm_status))

    breakyes = 0
    msg = "error"
    normy = fnorm(y)
    Axi = Ainput.Amap(xi)
    Atz = Ainput.ATmap(z)
    Rp = Axi - y
    proj_z = proj_l2(z, weight_vec)
    Rd = z - proj_z
    primfeas = fnorm(Rp) / (1 + normy)
    dualfeas = fnorm(Rd) / (1 + fnorm(z))
    maxfeas = max(primfeas, dualfeas)
    primfeasorg = primfeas
    dualfeasorg = dualfeas
    primobj = get_primobj(data, weight_vec, xi, Axi)
    dualobj = get_dualobj(data, Atz)
    relgap = get_relgap(primobj, dualobj)
    eta = None

    if maxfeas < max(1e-6, stoptol):
        if use_kkt:
            eta = kkt_eta(data, weight_vec, xi, z, y, Atz)
        else:
            primobj = get_primobj(data, weight_vec, xi, Axi)
            dualobj = get_dualobj(data, Atz)
            relgap = get_relgap(primobj, dualobj)
            eta = relgap
        etaorg = eta
        if eta < stoptol:
            breakyes = 1
            msg = "Converged in ADMM"
    if breakyes == 1:
        pass

    ncgsigma = sigma
    ncgdim = dim

    maxitersub = 10
    prim_win = 0
    dual_win = 0
    ncgtol = stoptol
    precond = 0
    bscale = 1
    cscale = 1

    for iter in range(maxiter):
        zold = z.copy()
        ncgsigma = sigma
        if primfeas < 1e-5:
            maxitersub = max(maxitersub, 30)
        elif primfeas < 1e-3:
            maxitersub = max(maxitersub, 30)
        elif primfeas < 1e-1:
            maxitersub = max(maxitersub, 20)
        y, Axi, xi, par_rr, breakyesncg = ssncg(
            data,
            Ainput,
            z,
            Axi,
            xi,
            weight_vec,
            ncgsigma,
            ncgdim,
            maxitersub,
            ncgtol,
            ncgtolconst,
            precond,
            bscale,
            cscale,
        )
        if breakyesncg < 0:
            ncgtolconst = max(ncgtolconst / 1.06, 1e-3)

        Rp = Axi - y
        z = zold + sigma * Rp
        Atz = Ainput.ATmap(z)
        normRp = fnorm(Rp)
        normy = fnorm(y)
        primfeasorg = primfeas
        primfeas = normRp / (1 + normy)
        proj_z = proj_l2(z, weight_vec)
        Rd = z - proj_z
        dualfeasorg = dualfeas
        dualfeas = fnorm(Rd) / (1 + fnorm(z))
        maxfeasorg = maxfeas
        maxfeas = max(primfeas, dualfeas)

        if maxfeas < max(1e-6, stoptol):
            etaorg = eta
            primobj = get_primobj(data, weight_vec, xi, Axi)
            dualobj = get_dualobj(data, Atz)
            relgap = get_relgap(primobj, dualobj)
            if use_kkt:
                eta = kkt_eta(data, weight_vec, xi, z, y, Atz)
            else:
                eta = relgap
            if eta < stoptol:
                breakyes = 1
                msg = "converged"

        if breakyes > 0:
            # print(msg)
            break
        if primfeasorg < dualfeasorg:
            prim_win += 1
        else:
            dual_win += 1

        sigma_update_iter = sigma_update(iter)
        sigmascale = 5
        sigmamax = 135
        if iter % sigma_update_iter == 0:
            sigmamin = 1e-4
            if prim_win > max(1, 1.2 * dual_win):
                prim_win = 0
                sigma = max(sigmamin, sigma / sigmascale)
            elif dual_win > max(1, 1.2 * prim_win):
                dual_win = 0
                sigma = min(sigmamax, sigma * sigmascale)
        #print(sigma)

    if iter == maxiter - 1:
        msg = "max iterations reached"
        primobj = get_primobj(data, weight_vec, xi, Axi)
        dualobj = get_dualobj(data, Atz)
        relgap = get_relgap(primobj, dualobj)
        if use_kkt:
            etaorg = eta
            eta = kkt_eta(data, weight_vec, xi, z, y, Atz)
        else:
            etaorg = eta
            eta = relgap
    end_time = time.perf_counter()
    if verbose > 0:
        print("SSNAL terminated in {} seconds.".format(end_time - start_time))
        print(f"Status: {msg}, Iterations: {iter+1}")
    return (
        primobj,
        dualobj,
        y,
        xi,
        z,
        eta,
        msg,
        iter + 1,
        breakyes,
        end_time - start_time,
    )


def sigma_update(iter):
    if iter < 10:
        sigma_update_iter = 2
    elif iter < 20:
        sigma_update_iter = 3
    elif iter < 200:
        sigma_update_iter = 3
    elif iter < 500:
        sigma_update_iter = 10
    else:
        sigma_update_iter = 10
    return sigma_update_iter


def ssncg(
    data,
    Ainput: AInput,
    z0,
    Axi0,
    xi0,
    weight_vec,
    sigma,
    dim,
    maxitersub=50,
    tol=1e-6,
    tolconst=0.5,
    precond=0,
    bscale=1,
    cscale=1,
    tiny=1e-10,
    verbose=1,
    **kwargs,
):
    maxitpsqmr = 500
    sig = sigma
    breakyes = 0
    normborg = 1 + fnorm(data) * np.sqrt(bscale * cscale)

    yinput = Axi0 + z0 / sig
    norm_yinput = np.sqrt(np.sum(yinput * yinput, axis=0))
    alpha_now = np.ones(dim.E)
    weight_sig = weight_vec / sig
    _, idx_temp, _ = find(norm_yinput > weight_sig)
    alpha_now[idx_temp] = weight_sig[idx_temp] / norm_yinput[idx_temp]

    y, rr, norm_yinput = prox_l2(yinput, (1 / sig) * weight_vec)
    normy = fnorm(y)
    Rp = Axi0 - y
    normRp = fnorm(Rp)
    ytmp = yinput - y
    Axi = Axi0
    xi = xi0

    Rd = np.maximum(np.sqrt(np.sum(z0 * z0, axis=0)) - weight_vec, 0)
    normRd = np.sum(Rd)
    #print(Rd)

    priminf_hist = np.zeros(maxitersub)
    dualinf_hist = np.zeros(maxitersub)
    Ly_hist = np.zeros(maxitersub)
    solve_ok_hist = np.zeros(maxitersub)
    psqmr_hist = np.zeros(maxitersub)
    psqmr_hist[0] = 0

    par_rr = None

    for itersub in range(maxitersub):
        Ly = 0.5 * fnorm(xi - data) ** 2
        Ly = Ly + np.dot(weight_vec, np.sqrt(np.einsum("ij,ij->j", y, y)))
        GradLxi = data - xi - sig * Ainput.ATmap(ytmp)
        normGradLxi = fnorm(GradLxi)
        priminf_sub = normGradLxi
        dualinf_sub = normRd * cscale / (1 + normborg * cscale)

        if max(priminf_sub, dualinf_sub) < tol:
            tolsubconst = 0.9
        else:
            tolsubconst = 0.5

        tolsub = max(min(1, tolconst * normRp / (1 + normy)), tolsubconst * tol)
        priminf_hist[itersub] = priminf_sub
        dualinf_hist[itersub] = dualinf_sub
        Ly_hist[itersub] = Ly
        #print(np.nanmax(normGradLxi), np.nanmax(tolsub))
        if np.nanmax(normGradLxi) < np.nanmax(tolsub) and itersub > 1:
            #print('here')
            breakyes = -1
            break
        elif max(priminf_sub, dualinf_sub) < 0.5 * tol:
            msg = f"max(priminf_sub, dualinf_sub) < {0.5 * tol:.3f}"
            #print(msg)
            breakyes = -1
            break

        epsilon = min(1e-3, 0.1 * normGradLxi)
        precond = precond

        maxitpsqmr = set_maxitpsqmr(maxitpsqmr, itersub, priminf_sub)

        if itersub >= 1:
            prim_ratio = priminf_sub / priminf_hist[itersub - 1]
            dual_ratio = dualinf_sub / dualinf_hist[itersub - 1]
        else:
            prim_ratio = 0
            dual_ratio = 0

        rhs = GradLxi
        tolpsqmr = min(5e-3, 0.1 * fnorm(rhs))
        const2 = 1

        if itersub >= 1 and (prim_ratio > 0.5 or priminf_sub > 0.1 * priminf_hist[0]):
            const2 *= 0.5
        if dual_ratio > 1.1:
            const2 *= 0.5
        tolpsqmr *= const2
        dirtol = tolpsqmr
        maxit = maxitpsqmr
        _, idx, _ = find(rr > 0)
        nzidx = idx
        normytmp = norm_yinput[idx]
        Dsub = yinput[:, idx] / normytmp
        alpha = weight_vec[idx] / (sigma * normytmp)

        dxi, resnrm, solve_ok = ssncg_direction(
            Ainput, rhs, dirtol, maxit, nzidx, alpha, Dsub, sigma
        )
        
        Adxi = Ainput.Amap(dxi)
        iterpsqmr = len(resnrm) - 1

        pariter = itersub

        if itersub <= 2 and dualinf_sub > 1e-4 or pariter < 2:
            stepopt = 1
        else:
            stepopt = 2

        steptol = 1e-4
        xi, Axi, y, ytmp, alp, iterstep, yinput, norm_yinput, rr, par_rr = findstep(
            data, weight_vec, xi, Axi, y, ytmp, dxi, Adxi, steptol, stepopt, Ly, sigma
        )
        #print(alp)
        solve_ok_hist[itersub] = solve_ok
        psqmr_hist[itersub] = iterpsqmr

        if alp < tiny:
            break
        Ly_ratio = 1
        if itersub >= 1:
            Ly_ratio = (Ly_hist[itersub - 1] - Ly) / (abs(Ly) + EPS)

        if itersub >= 4:
            breakyes = update_breakyes(
                breakyes,
                tol,
                priminf_hist,
                dualinf_hist,
                solve_ok_hist,
                psqmr_hist,
                itersub,
                priminf_sub,
                dualinf_sub,
            )
            if breakyes > 0:
                Rp = Axi - y
                normRp = fnorm(Rp)
                break
    if par_rr is None:
        par_rr = 0
    #print(itersub, xi)
    return y, Axi, xi, par_rr, breakyes


def update_breakyes(
    breakyes,
    tol,
    priminf_hist,
    dualinf_hist,
    solve_ok_hist,
    psqmr_hist,
    itersub,
    priminf_sub,
    dualinf_sub,
):
    idx = np.arange(max(0, itersub - 3), itersub + 1)
    tmp = priminf_hist[idx]
    ratio = np.nanmin(tmp) / np.nanmax(tmp)

    if (
        np.all(solve_ok_hist[idx] <= -1)
        and ratio > 0.9
        and np.nanmin(psqmr_hist[idx]) == np.nanmax(psqmr_hist[idx])
        and np.nanmax(tmp) < 5 * tol
    ):
        breakyes = 1

    const3 = 0.7
    priminf_1half = np.nanmin(priminf_hist[: math.ceil(itersub * const3)])
    priminf_2half = np.nanmin(priminf_hist[math.ceil(itersub * const3) :])
    priminf_best = np.nanmin(priminf_hist[:itersub])
    priminf_ratio = priminf_hist[itersub] / priminf_hist[itersub - 1]
    dualinf_ratio = dualinf_hist[itersub] / dualinf_hist[itersub - 1]
    _, stagnate_idx, _ = find(solve_ok_hist[: itersub + 1] <= -1)
    stagnate_count = len(stagnate_idx)
    idx2 = np.arange(max(0, itersub - 7), itersub + 1)

    if (
        itersub >= 9
        and np.all(solve_ok_hist[idx2] == -1)
        and priminf_best < 1e-2
        and dualinf_sub < 1e-3
    ):
        tmp = priminf_hist[idx2]
        ratio = np.nanmin(tmp) / np.nanmax(tmp)
        if ratio > 0.5:
            breakyes = 2
    if (
        itersub >= 14
        and priminf_1half < min(2e-3, priminf_2half)
        and dualinf_sub < 0.8 * dualinf_hist[0]
        and dualinf_sub < 1e-3
        and stagnate_count >= 3
    ):
        breakyes = 3
    if (
        itersub >= 14
        and priminf_ratio < 0.1
        and priminf_sub < 0.8 * priminf_1half
        and dualinf_sub < min(1e-3, 2 * priminf_sub)
        and (priminf_sub < 2e-3 or (dualinf_sub < 1e-5 and priminf_sub < 5e-3))
        and stagnate_count >= 3
    ):
        breakyes = 4
    if (
        itersub >= 9
        and dualinf_sub > 5 * np.nanmin(dualinf_hist)
        and priminf_sub > 2 * np.nanmin(priminf_hist)
    ):
        breakyes = 5
    if itersub >= 19:
        dualinf_ratioall = dualinf_hist[1 : itersub + 1] / dualinf_hist[:itersub]
        _, idx, _ = find(dualinf_ratioall > 1)
        if len(idx) >= 3:
            dualinf_increment = np.mean(dualinf_ratioall[idx])
            if dualinf_increment > 1.25:
                breakyes = 6
    return breakyes


def findstep(
    data, weight_vec, xi0, Axi0, y0, ytmp0, dxi, Adxi, tol, stepopt, Ly0, sigma
):
    maxit = max(10, np.ceil(np.log(1 / (tol + EPS) / np.log(2))))
    c1 = 1e-4
    c2 = 0.9
    sig = sigma

    g0 = np.sum(dxi * (data - xi0)) - sig * np.sum(Adxi * ytmp0)

    if g0 < -1e-10:
        alp = 0
        iter = 0
        xi = xi0
        Axi = Axi0
        y = y0
        ytmp = ytmp0
        par_rr = 0
        return xi, Axi, y, ytmp, alp, iter, None, None, None, par_rr
    alp = 1
    alpconst = 0.5

    for iter in range(maxit):
        if iter == 0:
            alp = 1
            LB = 0
            UB = 1
        else:
            alp = alpconst * (UB + LB)
        xi = xi0 + alp * dxi
        Axi = Axi0 + alp * Adxi
        yinput = ytmp0 + y0 + alp * Adxi
        y, rr, norm_yinput = prox_l2(yinput, (1 / sig) * weight_vec)
        par_rr = rr
        ytmp = yinput - y
        # galp = np.sum(dxi * (data - xi)) - sig * np.sum(Adxi * ytmp)
        galp = np.einsum("ij,ij->", dxi, data - xi) - sig * np.einsum(
            "ij,ij->", Adxi, ytmp
        )

        if iter == 0:
            gLB = g0
            gUB = galp
            if np.sign(gLB) * np.sign(gUB) > 0:
                #print('here in find')
                return xi, Axi, y, ytmp, alp, iter, yinput, norm_yinput, rr, par_rr

        if abs(galp) < c2 * abs(g0):
            #print('in this now')
            Ly = 0.5 * fnorm(xi - data) ** 2
            Ly = (
                Ly
                # + np.sum(weight_vec * np.sqrt(np.sum(y * y, axis=0)))
                + np.dot(weight_vec, np.sqrt(np.einsum("ij,ij->j", y, y)))
                + 0.5 * sig * fnorm(ytmp) ** 2
            )

            if (Ly - Ly0 + c1 * alp * g0 < 1e-8 / max(1, abs(Ly0))) and (
                stepopt == 1 or (stepopt == 2 and abs(galp) < tol)
            ):
                return xi, Axi, y, ytmp, alp, iter, yinput, norm_yinput, rr, par_rr

        if np.sign(galp) * np.sign(gUB) < 0:
            LB = alp
            gLB = galp
        elif np.sign(galp) * np.sign(gLB) < 0:
            UB = alp
            gUB = galp
    return xi, Axi, y, ytmp, alp, iter, yinput, norm_yinput, rr, par_rr


def ssncg_direction(Ainput, rhs, tol, maxit, nzidx, alpha, Dsub, sigma):
    d, n = rhs.shape
    x = np.zeros((d, n))
    r = rhs
    res_temp = fnorm(r)
    resnrm = np.zeros(maxit + 1)
    resnrm[0] = res_temp

    if res_temp < tol:
        return x, resnrm, 1

    y = -r
    z = Matvec(y, nzidx, alpha, Dsub, sigma, Ainput)
    # make y, z float64
    # y = y.astype(np.float64)
    # z = z.astype(np.float64)
    s = np.sum(y * z)
    t = np.sum(r * y) / s
    x = x + t * y

    for k in range(maxit):
        r = r - t * z
        res_temp = fnorm(r)
        resnrm[k + 1] = res_temp

        if res_temp < tol:
            break

        # B = np.sum(r * z) / s
        B = 1 / s * np.einsum("ij,ij->", r, z)
        y = -r + B * y
        z = Matvec(y, nzidx, alpha, Dsub, sigma, Ainput)
        # s = np.sum(y * z)
        s = np.einsum("ij,ij->", y, z)
        # t = np.sum(r * y) / s
        t = 1 / s * np.einsum("ij,ij->", r, y)
        x = x + t * y

    if k < maxit:
        solve_ok = 1
    else:
        solve_ok = -1
    #print(x)
    return x, resnrm, solve_ok


def Matvec(y, nzidx, alpha, Dsub, sigma, Ainput):
    idx = nzidx
    lenn = len(idx)
    Ay = Ainput.Amap(y)
    if lenn > 0:
        Aytmp = Ay[:, idx]
        rho = np.einsum("j, ij,ij->j", alpha, Aytmp, Dsub)
        Ay[:, idx] = Aytmp * alpha - Dsub * rho
    My = y + sigma * Ainput.ATmap(Ay)
    return My


def set_maxitpsqmr(maxitpsqmr, itersub, priminf_sub):
    if (priminf_sub > 1e-3) or itersub <= 5:
        maxitpsqmr = max(maxitpsqmr, 200)
    elif priminf_sub > 1e-4:
        maxitpsqmr = max(maxitpsqmr, 300)
    elif priminf_sub > 1e-5:
        maxitpsqmr = max(maxitpsqmr, 400)
    elif priminf_sub > 5e-6:
        maxitpsqmr = max(maxitpsqmr, 500)
    return maxitpsqmr


def kkt_eta(data, weight_vec, xi, z, y, Atz):
    grad = Atz + xi - data
    eta = fnorm(grad) / (1 + fnorm(xi))
    ypz = y + z
    ypz_prox, _, _ = prox_l2(ypz, weight_vec)
    res_vec = y - ypz_prox
    eta = eta + fnorm(res_vec) / (1 + fnorm(y))
    return eta


def get_relgap(primobj, dualobj):
    return np.abs(primobj - dualobj) / (1 + np.abs(primobj) + np.abs(dualobj))


def get_dualobj(data, Atz):
    return -0.5 * fnorm(Atz) ** 2 + np.einsum("ij,ij->", data, Atz)


def get_primobj(data, weight_vec, xi, Axi):
    return 0.5 * fnorm(xi - data) ** 2 + np.dot(
        weight_vec, np.sqrt(np.einsum("ij,ij->j", Axi, Axi))
    )
