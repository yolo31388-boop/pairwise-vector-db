"""
游戏内向量数据库与相似度检索系统 - 核心模块

功能：
- insert_vector: 带 ID 的向量插入，自动 L2 归一化，增量更新 ANN 索引
- search_similar: 余弦相似度检索，支持 TopK 与过滤，已删除向量不返回
- ann_index: IVF（倒排）ANN 索引的构建与增量更新
- embedding: 基于词表 + 哈希的文本嵌入，固定维度，支持 OOV
- clustering: KMeans 聚类，含 inertia 与 silhouette 质量评估
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import hashlib
import math
import random
import re
import time


@dataclass
class Config:
    dim: int = 0
    embedding_dim: int = 64
    ann_clusters: int = 64
    ann_probe: int = 4


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")


def _l2_normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return list(vector)
    return [x / norm for x in vector]


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class VectorDB:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.dim = int(self.config.get("dim", 0))
        self.embedding_dim = int(self.config.get("embedding_dim", 64))
        self.ann_clusters = int(self.config.get("ann_clusters", 64))
        self.ann_probe = int(self.config.get("ann_probe", 4))
        self._vectors: Dict[int, List[float]] = {}
        self._metadata: Dict[int, Dict] = {}
        self._deleted: set = set()
        self._next_id = 1
        self._vocab: Dict[str, int] = {}
        self._index: Optional[Dict] = None
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 基础生命周期
    # ------------------------------------------------------------------
    def update(self, dt: float):
        pass

    def reset(self):
        self._vectors = {}
        self._metadata = {}
        self._deleted = set()
        self._next_id = 1
        self._vocab = {}
        self._index = None
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 向量插入：带 ID、归一化、增量更新索引
    # ------------------------------------------------------------------
    def insert_vector(
        self,
        vector: List[float],
        metadata: Optional[Dict] = None,
        vector_id: Optional[int] = None,
    ) -> int:
        vector = [float(x) for x in vector]
        if self.dim == 0:
            self.dim = len(vector)
        if len(vector) != self.dim:
            raise ValueError(
                f"向量维度 {len(vector)} 与数据库维度 {self.dim} 不一致"
            )
        if vector_id is None:
            vector_id = self._next_id
            self._next_id += 1
        else:
            if vector_id in self._vectors and vector_id not in self._deleted:
                raise ValueError(f"向量 ID {vector_id} 已存在")
            self._next_id = max(self._next_id, vector_id + 1)

        normalized = _l2_normalize(vector)
        self._vectors[vector_id] = normalized
        self._metadata[vector_id] = dict(metadata) if metadata else {}
        self._deleted.discard(vector_id)
        self._index_add(vector_id, normalized)
        self._history.append(("insert", vector_id, time.time()))
        return vector_id

    def update_vector(
        self,
        vector_id: int,
        vector: List[float],
        metadata: Optional[Dict] = None,
    ) -> None:
        if vector_id not in self._vectors or vector_id in self._deleted:
            raise KeyError(f"向量 ID {vector_id} 不存在或已删除")
        vector = [float(x) for x in vector]
        if len(vector) != self.dim:
            raise ValueError(
                f"向量维度 {len(vector)} 与数据库维度 {self.dim} 不一致"
            )
        self._index_remove(vector_id)
        normalized = _l2_normalize(vector)
        self._vectors[vector_id] = normalized
        if metadata is not None:
            self._metadata[vector_id] = dict(metadata)
        self._index_add(vector_id, normalized)
        self._history.append(("update", vector_id, time.time()))

    def delete_vector(self, vector_id: int) -> None:
        if vector_id not in self._vectors or vector_id in self._deleted:
            raise KeyError(f"向量 ID {vector_id} 不存在或已删除")
        self._deleted.add(vector_id)
        self._index_remove(vector_id)
        self._history.append(("delete", vector_id, time.time()))

    def get_vector(self, vector_id: int) -> Optional[List[float]]:
        if vector_id in self._deleted:
            return None
        vec = self._vectors.get(vector_id)
        return list(vec) if vec is not None else None

    def _live_ids(self) -> List[int]:
        return [vid for vid in self._vectors if vid not in self._deleted]

    # ------------------------------------------------------------------
    # 相似度检索：余弦相似度 + TopK + 过滤
    # ------------------------------------------------------------------
    def search_similar(
        self,
        query: List[float],
        top_k: int = 10,
        filter_fn: Optional[Callable[[int, Dict], bool]] = None,
    ) -> List[Tuple[int, float]]:
        query = [float(x) for x in query]
        if self.dim == 0:
            return []
        if len(query) != self.dim:
            raise ValueError(
                f"查询维度 {len(query)} 与数据库维度 {self.dim} 不一致"
            )
        query = _l2_normalize(query)

        candidate_ids = self._ann_candidates(query)
        if candidate_ids is None:
            candidate_ids = self._live_ids()

        scored = []
        for vid in candidate_ids:
            if vid in self._deleted:
                continue
            if filter_fn is not None and not filter_fn(vid, self._metadata[vid]):
                continue
            # 向量已归一化，点积即余弦相似度
            scored.append((vid, _dot(query, self._vectors[vid])))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(0, top_k)]

    # ------------------------------------------------------------------
    # ANN 索引：IVF 倒排索引，支持构建与增量更新
    # ------------------------------------------------------------------
    def ann_index(self, n_clusters: Optional[int] = None) -> Optional[Dict]:
        """基于当前全部存活向量重建 IVF 索引，返回索引结构。

        质心用采样向量做 KMeans 训练，随后将全部向量归入最近质心。
        """
        ids = self._live_ids()
        if not ids:
            self._index = None
            return None
        k = min(n_clusters or self.ann_clusters, len(ids))
        # 训练数据采样，避免超大数据集上 KMeans 过慢
        sample_ids = ids
        if len(ids) > 5000:
            rng = random.Random(42)
            sample_ids = rng.sample(ids, 5000)
        sample = [self._vectors[vid] for vid in sample_ids]
        centroids, _ = self._kmeans(sample, k)
        buckets: Dict[int, List[int]] = {c: [] for c in range(k)}
        counts = [0] * k
        for vid in ids:
            label = self._nearest_centroid(self._vectors[vid], centroids)
            buckets[label].append(vid)
            counts[label] += 1
        self._index = {"centroids": centroids, "buckets": buckets,
                       "counts": counts}
        return self._index

    def _index_add(self, vector_id: int, normalized: List[float]) -> None:
        if self._index is None:
            # 累积到足够初始质心训练后自动构建一次，之后增量更新
            if len(self._vectors) >= max(2 * self.ann_clusters, 256):
                self.ann_index()
            return
        centroid_id = self._nearest_centroid(normalized)
        self._index["buckets"].setdefault(centroid_id, []).append(vector_id)
        # 在线更新质心（增量均值），保证新增向量可被检索且质心跟随数据
        centroid = self._index["centroids"][centroid_id]
        count = self._index["counts"][centroid_id]
        step = 1.0 / (count + 1)
        for d in range(len(centroid)):
            centroid[d] += (normalized[d] - centroid[d]) * step
        self._index["counts"][centroid_id] = count + 1

    def _index_remove(self, vector_id: int) -> None:
        if self._index is None:
            return
        for bucket in self._index["buckets"].values():
            if vector_id in bucket:
                bucket.remove(vector_id)
                return

    def _nearest_centroid(
        self,
        vector: List[float],
        centroids: Optional[List[List[float]]] = None,
    ) -> int:
        centroids = centroids if centroids is not None else self._index["centroids"]
        best, best_d = 0, float("inf")
        for i, centroid in enumerate(centroids):
            d = 0.0
            for x, y in zip(vector, centroid):
                d += (x - y) * (x - y)
            if d < best_d:
                best, best_d = i, d
        return best

    def _ann_candidates(self, query: List[float]) -> Optional[List[int]]:
        if self._index is None:
            return None
        centroids = self._index["centroids"]
        sims = []
        for i, centroid in enumerate(centroids):
            d = 0.0
            for x, y in zip(query, centroid):
                d += (x - y) * (x - y)
            sims.append((d, i))
        sims.sort()
        probe = min(self.ann_probe, len(sims))
        candidates: List[int] = []
        for _, cid in sims[:probe]:
            candidates.extend(self._index["buckets"].get(cid, []))
        return candidates

    # ------------------------------------------------------------------
    # 嵌入：词表 + 哈希，固定维度，OOV 可处理
    # ------------------------------------------------------------------
    def build_vocab(self, texts: List[str]) -> Dict[str, int]:
        for text in texts:
            for token in self._tokenize(text):
                if token not in self._vocab:
                    self._vocab[token] = self._token_index(token)
        return dict(self._vocab)

    def embedding(self, text: str, dim: Optional[int] = None) -> List[float]:
        dim = int(dim or self.embedding_dim)
        if dim <= 0:
            raise ValueError("嵌入维度必须为正整数")
        vector = [0.0] * dim
        for token in self._tokenize(text):
            if token not in self._vocab:
                # OOV 词通过哈希进入词表，保证同一词永远映射到同一维度
                self._vocab[token] = self._token_index(token) % dim
            idx = self._vocab[token] % dim
            sign = 1.0 if self._token_sign(token) else -1.0
            vector[idx] += sign
        return _l2_normalize(vector)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return _TOKEN_RE.findall(text.lower())

    @staticmethod
    def _token_index(token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    @staticmethod
    def _token_sign(token: str) -> bool:
        digest = hashlib.md5(("#" + token).encode("utf-8")).hexdigest()
        return int(digest[:2], 16) % 2 == 0

    # ------------------------------------------------------------------
    # 聚类：KMeans + 质量评估（inertia / silhouette）
    # ------------------------------------------------------------------
    def clustering(
        self,
        vectors: Optional[List[List[float]]] = None,
        k: int = 8,
        max_iter: int = 100,
    ) -> Dict:
        if vectors is None:
            ids = self._live_ids()
            vectors = [self._vectors[vid] for vid in ids]
        vectors = [list(v) for v in vectors]
        if not vectors:
            return {"labels": [], "centroids": [], "inertia": 0.0,
                    "silhouette": 0.0, "k": 0}
        k = max(1, min(k, len(vectors)))
        centroids, labels = self._kmeans(vectors, k, max_iter=max_iter)
        inertia = self._inertia(vectors, centroids, labels)
        silhouette = self._silhouette(vectors, labels, k)
        return {"labels": labels, "centroids": centroids,
                "inertia": inertia, "silhouette": silhouette, "k": k}

    @staticmethod
    def _kmeans(
        vectors: List[List[float]],
        k: int,
        max_iter: int = 100,
        seed: int = 42,
    ) -> Tuple[List[List[float]], List[int]]:
        rng = random.Random(seed)
        n = len(vectors)
        # k-means++ 初始化
        centroids = [list(vectors[rng.randrange(n)])]
        while len(centroids) < k:
            dists = []
            for v in vectors:
                d = min(_euclidean(v, c) for c in centroids)
                dists.append(d * d)
            total = sum(dists)
            if total == 0.0:
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

        labels = [0] * n
        for _ in range(max_iter):
            changed = False
            for i, v in enumerate(vectors):
                best, best_d = 0, float("inf")
                for c_idx, c in enumerate(centroids):
                    d = _euclidean(v, c)
                    if d < best_d:
                        best, best_d = c_idx, d
                if labels[i] != best:
                    changed = True
                    labels[i] = best
            new_centroids = []
            for c_idx in range(k):
                members = [vectors[i] for i in range(n) if labels[i] == c_idx]
                if not members:
                    # 空簇：用离现有质心最远的点重新初始化
                    farthest = max(
                        vectors,
                        key=lambda v: min(_euclidean(v, c) for c in centroids),
                    )
                    new_centroids.append(list(farthest))
                else:
                    dim = len(members[0])
                    new_centroids.append(
                        [sum(m[d] for m in members) / len(members)
                         for d in range(dim)]
                    )
            centroids = new_centroids
            if not changed:
                break
        return centroids, labels

    @staticmethod
    def _inertia(
        vectors: List[List[float]],
        centroids: List[List[float]],
        labels: List[int],
    ) -> float:
        total = 0.0
        for v, label in zip(vectors, labels):
            d = _euclidean(v, centroids[label])
            total += d * d
        return total

    @staticmethod
    def _silhouette(
        vectors: List[List[float]],
        labels: List[int],
        k: int,
        sample_size: int = 200,
    ) -> float:
        n = len(vectors)
        if n < 2 or k < 2:
            return 0.0
        indices = list(range(n))
        if n > sample_size:
            rng = random.Random(42)
            indices = sorted(rng.sample(indices, sample_size))
        clusters: Dict[int, List[int]] = {c: [] for c in range(k)}
        for i, label in enumerate(labels):
            clusters[label].append(i)

        scores = []
        for i in indices:
            own = clusters[labels[i]]
            if len(own) > 1:
                a = sum(_euclidean(vectors[i], vectors[j]) for j in own if j != i)
                a /= len(own) - 1
            else:
                a = 0.0
            b = float("inf")
            for c, members in clusters.items():
                if c == labels[i] or not members:
                    continue
                mean_d = sum(_euclidean(vectors[i], vectors[j]) for j in members)
                mean_d /= len(members)
                b = min(b, mean_d)
            if b == float("inf"):
                continue
            denom = max(a, b)
            scores.append((b - a) / denom if denom > 0 else 0.0)
        return sum(scores) / len(scores) if scores else 0.0
