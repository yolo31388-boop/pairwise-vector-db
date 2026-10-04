"""
游戏内向量数据库与相似度检索系统 - 核心模块

功能：
- 向量插入（自动ID、L2归一化、支持更新/删除）
- 相似度检索（余弦相似度、TopK、过滤、排除已删除向量）
- ANN索引（IVF倒排索引，支持构建与增量更新）
- 文本嵌入（动态词表、OOV哈希回退、固定维度）
- 聚类（KMeans + inertia/silhouette 质量评估）
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
import hashlib
import math
import random
import re
import time


@dataclass
class Config:
    dim: int = 64
    nlist: int = 16        # ANN 倒排桶数量
    nprobe: int = 4        # 检索时探测的桶数量
    ann_threshold: int = 256  # 向量数超过该值才启用ANN


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _tokenize(text: str) -> List[str]:
    # 英文/数字按词切分，CJK 按单字切分
    return re.findall(r"[a-z0-9]+|[一-鿿]", text.lower())


class VectorDB:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.dim = int(self.config.get("dim", Config.dim))
        self.nlist = int(self.config.get("nlist", Config.nlist))
        self.nprobe = int(self.config.get("nprobe", Config.nprobe))
        self.ann_threshold = int(self.config.get("ann_threshold", Config.ann_threshold))

        self._vectors: Dict[int, List[float]] = {}      # id -> 归一化后的向量
        self._metadata: Dict[int, Dict] = {}
        self._deleted: set = set()
        self._next_id = 1

        self._vocab: Dict[str, int] = {}                # 词 -> 词表索引

        # ANN (IVF) 索引
        self._centroids: List[List[float]] = []
        self._buckets: Dict[int, List[int]] = {}
        self._index_built = False

        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 向量插入 / 更新 / 删除
    # ------------------------------------------------------------------
    def insert_vector(self, vector: List[float], metadata: Optional[Dict] = None,
                      id: Optional[int] = None) -> int:
        """插入向量：自动分配ID、L2归一化、同步更新ANN索引。"""
        vec = [float(x) for x in vector]
        if len(vec) != self.dim:
            raise ValueError(f"向量维度 {len(vec)} 与数据库维度 {self.dim} 不一致")
        if id is None:
            id = self._next_id
            self._next_id += 1
        else:
            if id in self._vectors and id not in self._deleted:
                raise ValueError(f"ID {id} 已存在")
            self._next_id = max(self._next_id, id + 1)

        self._vectors[id] = _normalize(vec)
        self._metadata[id] = dict(metadata or {})
        self._deleted.discard(id)

        if self._index_built:
            self._assign_to_index(id)
        self._history.append(("insert", id))
        return id

    def update_vector(self, id: int, vector: Optional[List[float]] = None,
                      metadata: Optional[Dict] = None) -> bool:
        """按ID更新向量和/或元数据。"""
        if id not in self._vectors or id in self._deleted:
            return False
        if vector is not None:
            vec = [float(x) for x in vector]
            if len(vec) != self.dim:
                raise ValueError(f"向量维度 {len(vec)} 与数据库维度 {self.dim} 不一致")
            self._vectors[id] = _normalize(vec)
            if self._index_built:
                self._remove_from_index(id)
                self._assign_to_index(id)
        if metadata is not None:
            self._metadata[id] = dict(metadata)
        self._history.append(("update", id))
        return True

    def delete_vector(self, id: int) -> bool:
        """按ID删除向量（逻辑删除，检索时过滤）。"""
        if id not in self._vectors or id in self._deleted:
            return False
        self._deleted.add(id)
        if self._index_built:
            self._remove_from_index(id)
        self._history.append(("delete", id))
        return True

    def get_vector(self, id: int) -> Optional[List[float]]:
        if id in self._deleted:
            return None
        return self._vectors.get(id)

    def __len__(self) -> int:
        return len(self._vectors) - len(self._deleted)

    # ------------------------------------------------------------------
    # 相似度检索（余弦相似度 + TopK + 过滤）
    # ------------------------------------------------------------------
    def search_similar(self, query: List[float], top_k: int = 10,
                       filter: Optional[Union[Callable[[Dict], bool], Dict]] = None
                       ) -> List[Tuple[int, float]]:
        """余弦相似度检索，返回 TopK 的 (id, score)，按相似度降序。

        filter 可以是 callable(metadata) -> bool，或 {key: value} 精确匹配字典。
        已删除的向量不会返回。
        """
        q = _normalize([float(x) for x in query])
        if len(q) != self.dim:
            raise ValueError(f"查询维度 {len(q)} 与数据库维度 {self.dim} 不一致")

        candidate_ids = self._ann_candidates(q)
        if candidate_ids is None:
            candidate_ids = self._vectors.keys()

        filter_fn = self._build_filter(filter)
        scored = []
        for vid in candidate_ids:
            if vid in self._deleted:
                continue
            meta = self._metadata.get(vid, {})
            if filter_fn is not None and not filter_fn(meta):
                continue
            scored.append((vid, _dot(q, self._vectors[vid])))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:max(0, top_k)]

    @staticmethod
    def _build_filter(filter):
        if filter is None:
            return None
        if callable(filter):
            return filter
        if isinstance(filter, dict):
            return lambda meta: all(meta.get(k) == v for k, v in filter.items())
        raise TypeError("filter 必须是 callable 或 dict")

    # ------------------------------------------------------------------
    # ANN 索引（IVF：构建 + 增量更新）
    # ------------------------------------------------------------------
    def ann_index(self, nlist: Optional[int] = None) -> Dict:
        """构建IVF倒排索引，返回索引统计信息。"""
        if nlist is not None:
            self.nlist = nlist
        active_ids = [vid for vid in self._vectors if vid not in self._deleted]
        self._centroids = []
        self._buckets = {}
        if not active_ids:
            self._index_built = True
            return {"nlist": 0, "indexed": 0, "bucket_sizes": {}}

        k = min(self.nlist, len(active_ids))
        all_vectors = [self._vectors[vid] for vid in active_ids]
        # 大样本时抽样训练质心，再全量分桶，保证构建速度
        train_ids = active_ids
        if len(active_ids) > 4096:
            train_ids = random.Random(0).sample(active_ids, 4096)
        train_vectors = [self._vectors[vid] for vid in train_ids]
        centroids, _ = _kmeans(train_vectors, k, max_iter=15,
                               rng=random.Random(0))
        self._centroids = centroids
        self._buckets = {i: [] for i in range(len(centroids))}
        for vid, vec in zip(active_ids, all_vectors):
            ci = max(range(len(centroids)),
                     key=lambda c: _dot(vec, centroids[c]))
            self._buckets[ci].append(vid)
        self._index_built = True
        self._history.append(("ann_build", len(active_ids)))
        return {
            "nlist": len(self._centroids),
            "indexed": len(active_ids),
            "bucket_sizes": {ci: len(ids) for ci, ids in self._buckets.items()},
        }

    def _nearest_centroid(self, vec: List[float], n: int = 1) -> List[int]:
        scored = [(_dot(vec, c), i) for i, c in enumerate(self._centroids)]
        scored.sort(reverse=True)
        return [i for _, i in scored[:n]]

    def _assign_to_index(self, id: int):
        """新增向量增量加入索引，保证插入后立即可被检索到。"""
        if not self._centroids:
            self._centroids = [list(self._vectors[id])]
            self._buckets = {0: [id]}
            return
        ci = self._nearest_centroid(self._vectors[id])[0]
        self._buckets.setdefault(ci, []).append(id)

    def _remove_from_index(self, id: int):
        for ids in self._buckets.values():
            if id in ids:
                ids.remove(id)
                return

    def _ann_candidates(self, query: List[float]):
        """索引可用时返回候选ID集合，否则返回None（退化为全量暴力搜索）。"""
        if not self._index_built or not self._centroids:
            return None
        if len(self) < self.ann_threshold:
            return None
        probes = self._nearest_centroid(query, n=min(self.nprobe, len(self._centroids)))
        candidates = []
        for ci in probes:
            candidates.extend(self._buckets.get(ci, []))
        return candidates

    # ------------------------------------------------------------------
    # 文本嵌入（词表 + OOV哈希 + 固定维度）
    # ------------------------------------------------------------------
    def embedding(self, text: str) -> List[float]:
        """将文本编码为固定维度的归一化向量。

        已知词用词表中的稳定索引，OOV词通过哈希回退，保证任意文本都能嵌入。
        """
        if not isinstance(text, str) or not text.strip():
            return [0.0] * self.dim
        tokens = _tokenize(text)
        if not tokens:
            return [0.0] * self.dim

        vec = [0.0] * self.dim
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
            idx = self._vocab[token]
            # 词表索引 + 哈希混合定位到固定维度空间，OOV 也能稳定落位
            h = int(hashlib.md5(f"{idx}:{token}".encode("utf-8")).hexdigest(), 16)
            dim_idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 == 0 else -1.0
            vec[dim_idx] += sign * 1.0
        return _normalize(vec)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    # ------------------------------------------------------------------
    # 聚类（KMeans + 质量评估）
    # ------------------------------------------------------------------
    def clustering(self, k: int, max_iter: int = 50) -> Dict:
        """对库内向量做KMeans聚类，返回分组与质量评估指标。"""
        active_ids = [vid for vid in self._vectors if vid not in self._deleted]
        if not active_ids:
            return {"clusters": {}, "centroids": [], "inertia": 0.0, "silhouette": 0.0}
        k = max(1, min(k, len(active_ids)))
        vectors = [self._vectors[vid] for vid in active_ids]
        centroids, assignments = _kmeans(vectors, k, max_iter=max_iter,
                                         rng=random.Random(0))

        clusters: Dict[int, List[int]] = {i: [] for i in range(k)}
        for vid, ci in zip(active_ids, assignments):
            clusters[ci].append(vid)
        clusters = {ci: ids for ci, ids in clusters.items() if ids}

        inertia = sum(
            1.0 - _dot(vectors[i], centroids[assignments[i]])
            for i in range(len(vectors))
        )
        silhouette = _silhouette(vectors, assignments, k)
        self._history.append(("clustering", k))
        return {
            "clusters": clusters,
            "centroids": centroids,
            "inertia": inertia,
            "silhouette": silhouette,
        }

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def update(self, dt: float):
        pass

    def reset(self):
        self._vectors = {}
        self._metadata = {}
        self._deleted = set()
        self._next_id = 1
        self._vocab = {}
        self._centroids = []
        self._buckets = {}
        self._index_built = False
        self._state = {}
        self._history = []


# ----------------------------------------------------------------------
# 内部算法
# ----------------------------------------------------------------------
def _kmeans(vectors: List[List[float]], k: int, max_iter: int = 50,
            rng: Optional[random.Random] = None
            ) -> Tuple[List[List[float]], List[int]]:
    """KMeans（余弦距离，k-means++ 初始化），返回 (centroids, assignments)。"""
    rng = rng or random.Random(0)
    n = len(vectors)
    k = max(1, min(k, n))

    # k-means++ 初始化
    centroids = [list(vectors[rng.randrange(n)])]
    while len(centroids) < k:
        dists = [1.0 - max(_dot(v, c) for c in centroids) for v in vectors]
        total = sum(dists)
        if total <= 0:
            centroids.append(list(vectors[rng.randrange(n)]))
            continue
        r = rng.random() * total
        acc = 0.0
        chosen = n - 1
        for i, d in enumerate(dists):
            acc += d
            if acc >= r:
                chosen = i
                break
        centroids.append(list(vectors[chosen]))

    assignments = [0] * n
    for _ in range(max_iter):
        changed = False
        for i, v in enumerate(vectors):
            best = max(range(k), key=lambda ci: _dot(v, centroids[ci]))
            if best != assignments[i]:
                assignments[i] = best
                changed = True
        new_centroids = []
        for ci in range(k):
            members = [vectors[i] for i in range(n) if assignments[i] == ci]
            if not members:
                new_centroids.append(list(vectors[rng.randrange(n)]))
                continue
            mean = [sum(m[d] for m in members) / len(members)
                    for d in range(len(vectors[0]))]
            new_centroids.append(_normalize(mean))
        centroids = new_centroids
        if not changed:
            break
    return centroids, assignments


def _silhouette(vectors: List[List[float]], assignments: List[int], k: int,
                sample_size: int = 200) -> float:
    """轮廓系数（余弦距离），样本过多时抽样评估。返回 [-1, 1]。"""
    n = len(vectors)
    if k <= 1 or n <= k:
        return 0.0
    idxs = list(range(n))
    if n > sample_size:
        rng = random.Random(0)
        idxs = sorted(rng.sample(idxs, sample_size))

    clusters: Dict[int, List[int]] = {}
    for i, ci in enumerate(assignments):
        clusters.setdefault(ci, []).append(i)

    def mean_dist(i, members):
        others = [j for j in members if j != i]
        if not others:
            return 0.0
        return sum(1.0 - _dot(vectors[i], vectors[j]) for j in others) / len(others)

    scores = []
    for i in idxs:
        ci = assignments[i]
        a = mean_dist(i, clusters.get(ci, []))
        b = min(
            (mean_dist(i, members) for cj, members in clusters.items() if cj != ci),
            default=0.0,
        )
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0
