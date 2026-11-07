# Line-by-line pure-python (no numpy) transliteration of HarmonicEpochClass.swift
# (prepareEpochs + classStats), compared bit-exactly to the numpy goldens.
import json, struct, math, sys


def B(x): return struct.unpack('<Q', struct.pack('<d', x))[0]
def D(b): return struct.unpack('<d', struct.pack('<Q', b))[0]
def rint(x): return float(round(x))   # python round: half-even


def median(x):
    s = sorted(x); n = len(s); h = n // 2
    return s[h] if n % 2 == 1 else (s[h - 1] + s[h]) * 0.5


def tc(t0, d, c): return t0 if c else t0 + 0.5 * d


def prep(time, flux, period, t0, dur, excl, centre):
    t = []; y = []
    for i in range(len(time)):
        if not (math.isfinite(time[i]) and math.isfinite(flux[i])): continue
        if excl is not None and excl[i]: continue
        t.append(time[i]); y.append(flux[i])
    if len(t) < 10 or not period > 0 or not dur > 0: return None
    ref = median(y)
    raw = 1.4826 * median([abs(v - ref) for v in y])
    sigma = 1e-12 if 1e-12 > raw else raw
    diffs = [t[i + 1] - t[i] for i in range(len(t) - 1)]
    dt = median(diffs); half = 0.5 * dur
    expected = (2.0 * half) / dt if dt > 0 else 1.0
    cn = 0.5 * expected; need = cn if cn > 3.0 else 3.0
    c0 = tc(t0, dur, centre); tmid = 0.5 * (t[0] + t[-1])
    tref = c0 + rint((tmid - c0) / period) * period
    ep = [rint((v - tref) / period) for v in t]
    span = t[-1] - t[0]
    wA = 1e-4 * span; wB = 2.0 * dur; W = wB if wB < wA else wA
    step = 0.25 * dur
    J = 0
    if W > 0: J = int(math.ceil(W / step))
    edge = (0.5 * span) / period; nedge = edge if edge > 1.0 else 1.0
    bmax = 2.0 * step; lim = (half + W) + bmax
    ts = []; ys = []; es = []
    for i in range(len(t)):
        if abs(t[i] - (tref + ep[i] * period)) < lim:
            ts.append(t[i]); ys.append(y[i]); es.append(ep[i])

    def score(a, b):
        cnt = 0; acc = 0.0
        for i in range(len(ts)):
            c = tref + es[i] * period; c = c + b; c = c + a * es[i]
            if abs(ts[i] - c) < half: cnt += 1; acc += ys[i]
        if cnt < 3: return None
        return ref - acc / float(cnt)

    ba = 0.0; bb = 0.0; bs = score(0.0, 0.0)
    if J > 0 and ts:
        for j in range(-J, J + 1):
            a = (float(j) * step) / nedge
            for i in range(-2, 3):
                if j == 0 and i == 0: continue
                b = float(i) * step; s = score(a, b)
                if s is not None and (bs is None or s > bs): bs = s; ba = a; bb = b
    p2 = period + ba; c2 = tref + bb
    inidx = []; inep = []
    for i in range(len(t)):
        e = int(rint((t[i] - c2) / p2))
        if abs(t[i] - (c2 + float(e) * p2)) < half: inidx.append(i); inep.append(e)
    if not inep: return None
    ep0 = min(inep); m = max(inep) - ep0 + 1
    nE = [0.0] * m; sE = [0.0] * m
    for q in range(len(inidx)):
        k = inep[q] - ep0; nE[k] += 1.0; sE[k] += (y[inidx[q]] - ref)
    cov = [False] * m; dE = [0.0] * m
    for e in range(m):
        if nE[e] >= need: cov[e] = True; dE[e] = -sE[e] / nE[e]
    return dict(ep0=ep0, nE=nE, dE=dE, cov=cov, sigma=sigma, period=p2, centre=c2)


def cstats(p, k):
    nan = float('nan')
    m = len(p['nE']); nep = [0] * k; ndip = [0] * k; sD = [0.0] * k; sN = [0.0] * k; cls = [0] * m
    for e in range(m):
        c = ((p['ep0'] + e) % k + k) % k; cls[e] = c
        if not p['cov'][e]: continue
        nep[c] += 1; sD[c] += p['dE'][e]; sN[c] += p['nE'][e]
        if p['dE'][e] >= (2.0 * p['sigma']) / math.sqrt(p['nE'][e]): ndip[c] += 1
    depth = [nan] * k; ew = [nan] * k
    for c in range(k):
        if nep[c] > 0: depth[c] = sD[c] / float(nep[c]); ew[c] = p['sigma'] / math.sqrt(sN[c])
    sd2 = [0.0] * k
    for e in range(m):
        if p['cov'][e]:
            dev = p['dE'][e] - depth[cls[e]]; sd2[cls[e]] += dev * dev
    err = [nan] * k
    for c in range(k):
        if nep[c] > 0:
            err[c] = ew[c]
            if nep[c] >= 2:
                sc = math.sqrt((sd2[c] / float(nep[c] - 1)) / float(nep[c]))
                err[c] = ew[c] if ew[c] >= sc else sc
    return depth, err, nep, ndip


def same(a, b):
    return (a != a and D(b) != D(b)) or B(a) == b


d = json.load(open(sys.argv[1]))
bad = 0; n = 0
for c in d['cases']:
    time = [D(b) for b in c['time']]; flux = [D(b) for b in c['flux']]
    ex = None if c['excluded'] is None else [bool(v) for v in c['excluded']]
    for e in c['ephemerides']:
        p = prep(time, flux, D(e['period']), D(e['t0']), D(e['duration']),
                 ex if e['use_excluded'] else None, e['centre'])
        g = e['prep']
        if g is None:
            assert p is None; continue
        ok = (p['ep0'] == g['ep0']
              and all(same(a, b) for a, b in zip(p['nE'], g['n_e']))
              and all(same(a, b) for a, b in zip(p['dE'], g['d_e']))
              and same(p['sigma'], g['sigma']) and same(p['period'], g['period'])
              and same(p['centre'], g['centre']))
        for k in range(2, 6):
            dep, err, nep, ndip = cstats(p, k); s = e['stats'][str(k)]
            ok = (ok and all(same(a, b) for a, b in zip(dep, s['depth']))
                  and all(same(a, b) for a, b in zip(err, s['err']))
                  and nep == s['nep'] and ndip == s['ndip'])
        n += 1
        if not ok: bad += 1; print('MISMATCH', c['name'], D(e['period']))
print('ephemerides checked', n, 'mismatches', bad)
