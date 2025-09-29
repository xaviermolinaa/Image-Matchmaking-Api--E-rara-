# %%
import numpy as np
import cv2
from collections import defaultdict
from dataclasses import dataclass, field
import pickle
import os
from pathlib import Path
from tqdm import tqdm
import time
import joblib
import math
import faiss

try:
    import faiss  # pip install faiss-cpu  (or faiss-gpu)
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False


# %%
# ---------- Feature Extraction ----------
def extract_sift(gray_img, nfeatures=500):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kps, desc = sift.detectAndCompute(gray_img, None)
    if desc is None:
        return np.empty((0,128), np.float32), []
    return desc.astype(np.float32), kps
def to_rootsift(desc, eps=1e-12, l2_after=True):
    """
    Convert SIFT -> RootSIFT (Arandjelović & Zisserman, 2012).
    Steps: L1-normalize, then sqrt. Optionally L2-normalize after sqrt.
    desc: (N,128) float32
    """
    if desc is None or len(desc) == 0:
        return np.empty((0,128), np.float32)
    # L1-normalize
    desc /= (np.sum(desc, axis=1, keepdims=True) + eps)
    # element-wise sqrt
    desc = np.sqrt(desc, dtype=np.float32)
    if l2_after:
        # optional: stabilize numerics
        norms = np.linalg.norm(desc, axis=1, keepdims=True) + eps
        desc /= norms
    return desc.astype(np.float32)


# %%
def keypoints_to_tuples(kps):
    return [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            for kp in kps]

# %%
@dataclass
class VocabNode:
    centroid: np.ndarray
    children: list = field(default_factory=list)
    is_leaf: bool = False
    node_id: int = -1  # unique id for all nodes; leaves get final "visual word" ids
    df: int = 0        # document frequency (images that hit this node)
    idf: float = 0.0   # computed after training
    # optional: entropy if you prefer that formulation


# %%


# ---------- Hierarchical K-means (Vocabulary Tree) ----------
class VocabTree:
    def __init__(self, k=10, L=6, min_cluster_size=30, max_iter=20, seed=0):
        self.k = int(k)
        self.L = int(L)
        self.min_cluster_size = int(min_cluster_size)
        self.max_iter = int(max_iter)
        self.rng = np.random.RandomState(seed)
        self.root = None
        self.leaf_nodes = []   # populated after build
        self._next_id = 0
        # If elsewhere you build a compact id map: self.leaf_id_map = {leaf.node_id: i, ...}

    # ---- FAISS KMeans ----
    def _kmeans(self, X, k):
        X = np.ascontiguousarray(X.astype(np.float32))
        N, D = X.shape
        if N < k:
            k = max(1, N)
        seed = int(self.rng.randint(0, 2**31 - 1))
        clus = faiss.Kmeans(d=D, k=k, niter=50, nredo=3, verbose=True, seed=seed)
        try:
            if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0 and hasattr(clus, "gpu"):
                clus.gpu = True
        except Exception:
            pass
        clus.train(X)
        centroids = np.ascontiguousarray(clus.centroids.reshape(k, D))
        index = faiss.IndexFlatL2(D); index.add(centroids)
        _, I = index.search(X, 1)
        return centroids, I.ravel().astype(np.int32)


    def _build_rec(self, X, level):
        node = VocabNode(centroid=X.mean(0), node_id=self._next_id)
        self._next_id += 1
        
         # FAISS-aware early stop
        required = max(self.min_cluster_size, 39 * self.k)
        if level == self.L or len(X) < required:
            node.is_leaf = True
            self.leaf_nodes.append(node)
            return node


        centroids, labels = self._kmeans(X, self.k)
        for j in range(centroids.shape[0]):
            child_points = X[labels == j]
            if len(child_points) == 0:
                continue
            child = self._build_rec(child_points, level + 1)
            child.centroid = centroids[j]
            node.children.append(child)
        if len(node.children) == 0:
            node.is_leaf = True
            self.leaf_nodes.append(node)
        return node

    def fit(self, all_descs):
        """
        all_descs: list of np.ndarray, each (Ni, D) SIFT/RootSIFT descriptors per image.
        """
        X = np.vstack([d for d in all_descs if len(d)])
        print("Training descriptors:", len(X))
        self.root = self._build_rec(X, level=0)

    # ---- Hard (single-path) assignment, unchanged ----
    def quantize_path(self, desc):
        """
        Return the leaf node reached by this descriptor (hard assignment).
        """
        node = self.root
        while not node.is_leaf and node.children:
            # pick child with nearest centroid
            cents = np.stack([ch.centroid for ch in node.children], axis=0)
            j = int(np.argmin(((cents - desc) ** 2).sum(1)))
            node = node.children[j]
        return node

    # ---- Soft assignment (new): descend to multiple children when close ----
    def quantize_descriptor_soft(self, desc, ratio=1.15, max_branch=2, max_soft_levels=2):
        """
        Descend the tree allowing up to `max_branch` children per level if their
        distance is within `ratio` of the best distance. Only apply this for the
        first `max_soft_levels` levels; afterwards use single best child.
        Returns a list of *leaf nodes* reached (deduplicated).
        """
        assert self.root is not None
        stack = [(self.root, 0)]
        leaves = []

        while stack:
            node, depth = stack.pop()
            if node.is_leaf or not node.children:
                leaves.append(node)
                continue

            cents = np.stack([c.centroid for c in node.children], axis=0)
            dist = ((cents - desc) ** 2).sum(1)
            order = np.argsort(dist)

            if depth < max_soft_levels:
                best = dist[order[0]]
                chosen = [node.children[i]
                          for i in order
                          if dist[i] <= best * ratio][:max_branch]
            else:
                chosen = [node.children[int(order[0])]]

            for ch in chosen:
                stack.append((ch, depth + 1))

        # Deduplicate in case multiple paths converge to the same leaf
        uniq = {}
        for lf in leaves:
            uniq[lf.node_id] = lf
        return list(uniq.values())

    def quantize_descriptors_soft(self, D, **kwargs):
        """
        Quantize many descriptors with soft assignment,
        returning a list of leaf node_ids (or compact ids if you map them outside).
        """
        if D is None or len(D) == 0:
            return []
        lids = []
        for d in D:
            leaf_nodes = self.quantize_descriptor_soft(d, **kwargs)
            lids.extend([lf.node_id for lf in leaf_nodes])
        return lids

class InvertedIndex:
    """
    Dense-by-leaf postings: postings[leaf_id] is a list of (doc_id, tf).
    IDF is a NumPy array of length num_leaves.
    External IDs are mapped to compact internal doc_ids.
    """
    def __init__(self, num_leaves: int):
        self.num_leaves = int(num_leaves)
        self.postings = [list() for _ in range(self.num_leaves)]  # leaf_id -> [(doc_id, tf)]
        self.doc_tf = {}                   # doc_id(int) -> {leaf_id(int): tf(int)}
        self.idf = np.zeros((self.num_leaves,), dtype=np.float32)
        self.doc_norm = {}                 # doc_id -> L2 norm of TF-IDF vector
        self.N = 0                         # number of docs

        # external<->internal mapping (kept)
        self.ext2int = {}                  # external image_id (str/int) -> doc_id (int)
        self.int2ext = []                  # index = doc_id, value = external id

        # stopwords (optional)
        self.stop_leaves = set()

    # ---------- ID mapping ----------
    def _get_doc_id(self, image_id):
        if image_id in self.ext2int:
            return self.ext2int[image_id]
        doc_id = len(self.int2ext)
        self.ext2int[image_id] = doc_id
        self.int2ext.append(image_id)
        return doc_id

    # ---------- Add one image's TF ----------
    def add_image(self, image_id, leaf_ids):
        """
        leaf_ids MUST be compact leaf IDs in [0..num_leaves-1].
        """
        doc_id = self._get_doc_id(image_id)
        if leaf_ids is None or len(leaf_ids) == 0:
            self.doc_tf[doc_id] = {}
        else:
            uniq, counts = np.unique(np.asarray(leaf_ids, dtype=np.int64), return_counts=True)
            # safety: clip to valid range
            uniq = uniq[(uniq >= 0) & (uniq < self.num_leaves)]
            counts = counts[:len(uniq)]
            self.doc_tf[doc_id] = {int(w): int(c) for w, c in zip(uniq, counts)}
        self.N = len(self.doc_tf)

    # ---------- Build postings from doc_tf (call before compute_idf) ----------
    def _build_postings(self):
        self.postings = [list() for _ in range(self.num_leaves)]
        for doc_id, tfmap in self.doc_tf.items():
            for lid, tf in tfmap.items():
                # skip stop leaves if already defined pre-IDF (rare)
                if lid in self.stop_leaves:
                    continue
                self.postings[lid].append((doc_id, tf))

    # ---------- Choose stop leaves (optional) ----------
    def _choose_stop_leaves(self, top_percent=0.0, frac_thresh=None):
        if (top_percent is None or top_percent <= 0.0) and (frac_thresh is None):
            self.stop_leaves = set()
            return
        df = np.array([len(plist) for plist in self.postings], dtype=np.int32)
        N = max(1, self.N)
        stops = set()
        if top_percent and top_percent > 0.0:
            k = max(1, int(self.num_leaves * top_percent))
            top_idx = np.argpartition(df, -k)[-k:]
            stops.update(int(i) for i in top_idx.tolist())
        if frac_thresh is not None:
            for lid, d in enumerate(df):
                if d / N > frac_thresh:
                    stops.add(lid)
        self.stop_leaves = stops

    # ---------- Compute IDF (after postings are built) ----------
    def compute_idf(self, use_entropy=False, stop_percent=0.0, stop_frac=None, hard_purge=False):
        """
        Computes IDF (or entropy weights). Supports optional stopwords:
          - stop_percent: drop top p fraction of most frequent leaves (e.g., 0.005 for 0.5%)
          - stop_frac: drop leaves with df/N > stop_frac (e.g., 0.05)
          - hard_purge: if True, also purge their postings
        """
        # rebuild postings from doc_tf to be sure they're aligned with current docs
        self._build_postings()

        # choose stop leaves (optional)
        self._choose_stop_leaves(top_percent=stop_percent, frac_thresh=stop_frac)
        if hard_purge and self.stop_leaves:
            for lid in self.stop_leaves:
                self.postings[lid] = []

        N = max(1, self.N)
        idf = np.zeros((self.num_leaves,), dtype=np.float32)
        for lid, plist in enumerate(self.postings):
            if lid in self.stop_leaves:
                idf[lid] = 0.0
                continue
            df = len(plist)
            if df == 0:
                idf[lid] = 0.0
            else:
                if use_entropy:
                    p = min(1 - 1e-12, max(1e-12, df / N))
                    idf[lid] = float(-(p * math.log(p) + (1 - p) * math.log(1 - p)))
                else:
                    idf[lid] = float(math.log(N / df))
        self.idf = idf

        # precompute per-doc norms for cosine similarity
        self.doc_norm = {}
        for doc_id, tfmap in self.doc_tf.items():
            s = 0.0
            for lid, tf in tfmap.items():
                w = tf * self.idf[lid]
                s += w * w
            self.doc_norm[doc_id] = math.sqrt(s) if s > 0 else 1.0

    # ---------- Build vectors ----------
    def image_vector(self, image_id):
        """Return normalized TF-IDF vector as dict {leaf_id: weight} using EXTERNAL image_id."""
        if image_id not in self.ext2int:
            return {}
        doc_id = self.ext2int[image_id]
        tf_map = self.doc_tf.get(doc_id, {})
        if not tf_map:
            return {}
        w = {lid: tf * float(self.idf[lid]) for lid, tf in tf_map.items() if self.idf[lid] > 0.0}
        norm = self.doc_norm.get(doc_id, None)
        if norm is None:
            # fallback compute norm if not cached
            s = sum(v * v for v in w.values())
            norm = math.sqrt(s) if s > 0 else 1.0
        if norm == 0:
            norm = 1.0
        return {lid: v / norm for lid, v in w.items()}

    def query_vector(self, leaf_ids):
        tf = defaultdict(int)
        for lid in leaf_ids:
            if 0 <= lid < self.num_leaves and self.idf[lid] > 0.0:
                tf[lid] += 1
        if not tf:
            return {}
        w = {lid: c * float(self.idf[lid]) for lid, c in tf.items()}
        s = sum(v * v for v in w.values())
        norm = math.sqrt(s) if s > 0 else 1.0
        return {lid: v / norm for lid, v in w.items()}

    # ---------- Scoring ----------
    def score(self, qvec, topk=50, return_external_ids=True):
        """
        Cosine similarity via sparse postings. Assumes doc_norms are precomputed.
        Returns [(doc_id or external_id, score), ...]
        """
        if not qvec:
            return []
        acc = defaultdict(float)
        for lid, qw in qvec.items():
            if self.idf[lid] == 0.0:
                continue
            for doc_id, tf in self.postings[lid]:
                iw = tf * float(self.idf[lid])
                acc[doc_id] += qw * iw

        results = []
        for doc_id, dot in acc.items():
            denom = self.doc_norm.get(doc_id, 1.0)
            results.append((doc_id, dot / denom if denom > 0 else 0.0))
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:topk]
        if return_external_ids:
            return [(self.int2ext[doc_id], score) for doc_id, score in results]
        return results


class VocabTreeDB:
    def __init__(self, k=10, L=6, min_cluster_size=25, max_iter=40, seed=0):
        self.tree = VocabTree(k=k, L=L, min_cluster_size=min_cluster_size, max_iter=max_iter, seed=seed)
        self.index = None
        self.image_meta = {}  # optional: {external_id: {...}}

    def spatial_verify(self, qdesc, qkps, candidates, ratio_thresh=0.75, cap=500, lam=0.001):
        # quick exits
        if qdesc is None or len(qdesc) == 0 or not qkps:
            return [(img_id, score, 0) for img_id, score in candidates]

        qdesc = qdesc.astype(np.float32, copy=False)
        bf = cv2.BFMatcher(cv2.NORM_L2)

        reranked = []
        for img_id, base_score in candidates:
            meta = self.image_meta.get(img_id, None)
            if not meta:
                reranked.append((img_id, base_score, 0)); continue

            desc = meta.get('descs', None)
            kps  = meta.get('kps', None)
            if desc is None or len(desc) == 0 or not kps:
                reranked.append((img_id, base_score, 0)); continue

            desc = desc.astype(np.float32, copy=False)

            # k-NN matches (defensive against short lists)
            matches = bf.knnMatch(qdesc, desc, k=2)
            good = []
            for pair in matches[:cap]:
                if len(pair) < 2:     # sometimes only one neighbor exists
                    continue
                m, n = pair
                if n is not None and m.distance < ratio_thresh * n.distance:
                    good.append(m)

            if len(good) < 4:
                reranked.append((img_id, base_score, 0)); continue

            # build correspondence arrays
            src = np.float32([qkps[m.queryIdx].pt for m in good])
            dst = np.float32([kps[m.trainIdx].pt for m in good])

            H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0)

            inliers = int(mask.ravel().sum()) if mask is not None else 0

            final = float(base_score) + lam * min(inliers, cap)
            reranked.append((img_id, final, inliers))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked



    def train(self, image_descs):
        """
        image_descs: list of np.ndarray (Ni,128) used to train the vocab.
        After fit, we build a compact leaf-id map and allocate InvertedIndex with num_leaves.
        """
        self.tree.fit(image_descs)
        # compact leaf IDs: node_id -> [0..num_leaves-1]
        self.tree.leaf_id_map = {leaf.node_id: i for i, leaf in enumerate(self.tree.leaves)} \
                                if hasattr(self.tree, "leaves") else \
                                {leaf.node_id: i for i, leaf in enumerate(self.tree.leaf_nodes)}
        num_leaves = len(self.tree.leaves) if hasattr(self.tree, "leaves") else len(self.tree.leaf_nodes)
        self.index = InvertedIndex(num_leaves=num_leaves)

    def add_image(self, external_image_id, descs, kps=None, path=None):
        # get multiple leaves per descriptor for better recall
        raw_leaf_ids = self.tree.quantize_descriptors_soft(
            descs,
            ratio=1.15,
            max_branch=2,
            max_soft_levels=2
        )

        # map raw node_ids -> compact ids
        leaf_ids = [self.tree.leaf_id_map[lid] for lid in raw_leaf_ids]

        self.index.add_image(external_image_id, leaf_ids)
        
        self.image_meta[external_image_id] = dict(path=path, kps=keypoints_to_tuples(kps), descs=descs)


    def finalize(self, use_entropy=False, stop_percent=0.0, stop_frac=None, hard_purge=False):
        """
        Must be called AFTER all add_image() calls.
        Computes IDF (or entropy) and precomputes doc norms.
        Optional stopwords:
        - stop_percent=0.005  -> drop top 0.5% by df
        - stop_frac=0.05      -> drop leaves with df/N > 0.05
        """
        assert self.index is not None, "Index not initialized."
        self.index.compute_idf(use_entropy=use_entropy,
                            stop_percent=stop_percent,
                            stop_frac=stop_frac,
                            hard_purge=hard_purge)

    def query(self, q_descs, topk=20):
        raw_leaf_ids = self.tree.quantize_descriptors_soft(
            q_descs,
            ratio=1.15,
            max_branch=2,
            max_soft_levels=2
        )
        leaf_ids = [self.tree.leaf_id_map[lid] for lid in raw_leaf_ids]

        qvec = self.index.query_vector(leaf_ids)
        return self.index.score(qvec, topk=topk)


def train_model():
    descriptors_list = []
    images_descriptors={}
    DIR_NAME='images/'
    for image_name in tqdm(os.listdir('images')):
        image=cv2.imread(f'{DIR_NAME}{image_name}', cv2.IMREAD_GRAYSCALE)
        # image8bit = cv2.normalize(image_gs, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        descriptors, keypoints = extract_sift(image,nfeatures=1000)
        descriptors = to_rootsift(descriptors)
        images_descriptors[image_name.split('.')[0]]={'desc':descriptors,'kps':keypoints,'path':f'{DIR_NAME}{image_name}'}
        descriptors_list.append(descriptors)


    # with open('filename.pickle', 'rb') as handle:
    #     b = pickle.load(handle)

    # %%
    db=VocabTreeDB(k=10, L=6)
    start_time = time.time()
    db.image_meta=images_descriptors
    db.train(descriptors_list)
    end_time = time.time()


    # %%
    print(f"Training time: {(end_time - start_time/10)} ,minutes")

    # %%
    for image_id in tqdm(images_descriptors.keys()):
        db.add_image(image_id, descs=images_descriptors[image_id]['desc'], kps=images_descriptors[image_id]['kps'], path=images_descriptors[image_id]['path'])

    # %%
    # Compute IDF + stopwords (can be rerun after more additions)
    db.finalize(use_entropy=True,stop_percent=0.02, stop_frac=0.05, hard_purge=False)
    #
    # %%
    print("idf size:", len(db.index.idf))
    print("num docs:", db.index.N)
    print("sample tf:", list(db.index.doc_tf.items())[0])
    return db

# %%
# Query
db = train_model()
def query_db(image, topk=20):
    q_descs, q_kps = extract_sift(image,nfeatures=1500)
    qdesc_root = to_rootsift(q_descs)
    cands = db.query(q_descs, topk=200)
    reranked = db.spatial_verify(qdesc_root,q_kps, cands, ratio_thresh=0.75, cap=1000, lam=0.01)
    top10 = [(img, score) for img, score, inl in reranked[:10]]
    print("Top 10 after RANSAC re-ranking:", top10)
    return top10


# %%

    q_descs, q_kps = extract_sift(image,nfeatures=1500)
    qdesc_root = to_rootsift(q_descs)
    cands = db.query(q_descs, topk=topk)
    reranked = db.spatial_verify(qdesc_root,q_kps, cands, ratio_thresh=0.75, cap=1000, lam=0.01)
    topk = [(img, score) for img, score, inl in reranked[:topk]]
    return topk
