# ============================================================
# scripts/compare_distance_metrics.py
# Does replacing cosine with another distance help? (Arun, 8 Oct 2026)
#
# Proxy task, judge-free and needing no embedder: take node vectors
# already stored in the five FAISS indexes, hold out half, and ask which
# domain graph each held-out node came from.
#   - nearest centroid  ~ how ROUTING works (query vs one point per domain)
#   - 10-nearest-neighbour ~ how SEED SEARCH works (nearest concept nodes)
# Metrics: cosine (current), Euclidean, Manhattan, and Mahalanobis using a
# Ledoit-Wolf shrinkage estimate of the pooled within-domain covariance.
#
# Limitation: this is domain classification of nodes, not retrieval of
# relevant concepts for real queries. Treat it as a screen; confirm any
# winner with test_routing_accuracy.py and retrieval_eval.py.
#
# Usage: python scripts/compare_distance_metrics.py
# ============================================================
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(ROOT, "indexes")
import faiss, numpy as np, time
from sklearn.covariance import LedoitWolf
rng=np.random.default_rng(0)
D=["database","deep_learning","software_engineering","operating_systems","computer_architecture"]
tr,te=[],[]
for i,d in enumerate(D):
    ix=faiss.read_index(os.path.join(INDEX_DIR, f"graphrag_{d}.faiss")); ix.make_direct_map()
    v=np.zeros((ix.ntotal,ix.d),dtype='float32'); ix.reconstruct_n(0,ix.ntotal,v)
    p=rng.permutation(len(v))[:2000]
    tr.append((v[p[:1000]],i)); te.append((v[p[1000:]],i))
Xtr=np.vstack([x for x,_ in tr]); ytr=np.concatenate([[i]*len(x) for x,i in tr])
Xte=np.vstack([x for x,_ in te]); yte=np.concatenate([[i]*len(x) for x,i in te])
Xc=Xtr-np.array([Xtr[ytr==c].mean(0) for c in range(5)])[ytr]
lw=LedoitWolf().fit(Xc); L=np.linalg.cholesky(lw.precision_).astype(np.float32)
print(f"train {Xtr.shape} test {Xte.shape}  Ledoit-Wolf shrinkage={lw.shrinkage_:.3f}",flush=True)
def knn(Xa,Xb,dist,k=10):
    pred=[]
    for s in range(0,len(Xb),100):
        B=Xb[s:s+100]
        if dist=="cos": Dm=-(B@Xa.T)
        elif dist=="l2": Dm=(B**2).sum(1)[:,None]+(Xa**2).sum(1)[None,:]-2*B@Xa.T
        else: Dm=np.stack([np.abs(Xa-b).sum(1) for b in B])
        nn=np.argpartition(Dm,k,axis=1)[:,:k]
        pred.extend(np.bincount(ytr[r],minlength=5).argmax() for r in nn)
    return (np.array(pred)==yte).mean()
def cent(Xa,Xb,dist):
    C=np.array([Xa[ytr==c].mean(0) for c in range(5)])
    if dist=="cos": s=-(Xb@(C/np.linalg.norm(C,axis=1,keepdims=True)).T)
    elif dist=="l2": s=((Xb[:,None,:]-C[None])**2).sum(2)
    else: s=np.abs(Xb[:,None,:]-C[None]).sum(2)
    return (s.argmin(1)==yte).mean()
print(f"\n{'metric':30}{'nearest centroid':>18}{'10-NN':>9}",flush=True)
for name,dist,A,B in [("Cosine (current)","cos",Xtr,Xte),("Euclidean (L2)","l2",Xtr,Xte),
                      ("Manhattan (L1)","l1",Xtr,Xte),("Mahalanobis (shrinkage)","l2",Xtr@L,Xte@L)]:
    t=time.time(); print(f"{name:30}{cent(A,B,dist):>17.1%}{knn(A,B,dist):>9.1%}   {time.time()-t:.0f}s",flush=True)
