"""
游戏内向量数据库与相似度检索系统 - 验收测试
运行方式：python -m pytest tests/test_vector_db.py -q
共 15 个测试用例
"""
import math
import random
import time

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.vector_db import VectorDB


def _unit(v):
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


class TestVectorDB:
    def setup_method(self):
        self.system = VectorDB()

    # ---- 向量插入：ID + 归一化 ----
    def test_case_01(self):
        id1 = self.system.insert_vector([1.0, 0.0, 0.0])
        id2 = self.system.insert_vector([0.0, 1.0, 0.0])
        assert id1 != id2
        id3 = self.system.insert_vector([0.0, 0.0, 1.0], vector_id=100)
        assert id3 == 100
        with pytest.raises(ValueError):
            self.system.insert_vector([1.0, 1.0, 1.0], vector_id=100)

    def test_case_02(self):
        vid = self.system.insert_vector([3.0, 4.0, 0.0])
        stored = self.system.get_vector(vid)
        norm = math.sqrt(sum(x * x for x in stored))
        assert abs(norm - 1.0) < 1e-9
        assert abs(stored[0] - 0.6) < 1e-9
        assert abs(stored[1] - 0.8) < 1e-9

    def test_case_03(self):
        vid = self.system.insert_vector([1.0, 0.0, 0.0])
        self.system.update_vector(vid, [0.0, 5.0, 0.0])
        stored = self.system.get_vector(vid)
        assert abs(stored[1] - 1.0) < 1e-9
        with pytest.raises(KeyError):
            self.system.update_vector(9999, [1.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            self.system.insert_vector([1.0, 2.0])

    # ---- 相似度检索：余弦相似度 + TopK + 过滤 ----
    def test_case_04(self):
        # 余弦相似度：同方向不同量级得分应一致，正交向量得分为 0
        self.system.insert_vector([1.0, 0.0, 0.0])
        self.system.insert_vector([100.0, 0.0, 0.0])
        self.system.insert_vector([0.0, 1.0, 0.0])
        results = self.system.search_similar([5.0, 0.0, 0.0], top_k=3)
        scores = {vid: s for vid, s in results}
        assert abs(scores[1] - 1.0) < 1e-9
        assert abs(scores[2] - 1.0) < 1e-9
        assert abs(scores[3] - 0.0) < 1e-9

    def test_case_05(self):
        for i in range(20):
            self.system.insert_vector([float(i + 1), 1.0, 0.0])
        results = self.system.search_similar([1.0, 0.0, 0.0], top_k=5)
        assert len(results) == 5
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_case_06(self):
        self.system.insert_vector([1.0, 0.0, 0.0], metadata={"type": "weapon"})
        self.system.insert_vector([0.9, 0.1, 0.0], metadata={"type": "armor"})
        self.system.insert_vector([0.8, 0.2, 0.0], metadata={"type": "weapon"})
        results = self.system.search_similar(
            [1.0, 0.0, 0.0],
            top_k=10,
            filter_fn=lambda vid, meta: meta.get("type") == "weapon",
        )
        assert len(results) == 2
        for vid, _ in results:
            assert vid != 2

    def test_case_07(self):
        vid1 = self.system.insert_vector([1.0, 0.0, 0.0])
        vid2 = self.system.insert_vector([0.9, 0.1, 0.0])
        self.system.delete_vector(vid1)
        results = self.system.search_similar([1.0, 0.0, 0.0], top_k=10)
        returned_ids = [vid for vid, _ in results]
        assert vid1 not in returned_ids
        assert vid2 in returned_ids
        assert self.system.get_vector(vid1) is None
        with pytest.raises(KeyError):
            self.system.delete_vector(vid1)

    # ---- ANN 索引：构建 + 更新 ----
    def test_case_08(self):
        rng = random.Random(0)
        for _ in range(100):
            self.system.insert_vector([rng.random() for _ in range(8)])
        index = self.system.ann_index(n_clusters=8)
        assert index is not None
        assert len(index["centroids"]) == 8
        indexed_ids = [vid for bucket in index["buckets"].values() for vid in bucket]
        assert sorted(indexed_ids) == list(range(1, 101))

    def test_case_09(self):
        rng = random.Random(1)
        for _ in range(100):
            self.system.insert_vector([rng.random() for _ in range(8)])
        self.system.ann_index(n_clusters=8)
        # 索引构建后新增的向量也必须能被检索到（索引增量更新）
        new_id = self.system.insert_vector([1.0] + [0.0] * 7)
        results = self.system.search_similar([1.0] + [0.0] * 7, top_k=1)
        assert results[0][0] == new_id
        assert results[0][1] > 0.99

    def test_case_10(self):
        rng = random.Random(2)
        base = [[rng.random() for _ in range(16)] for _ in range(2000)]
        for v in base:
            self.system.insert_vector(v)
        # 大规模向量下 ANN 检索应快速且能召回精确匹配
        start = time.time()
        hits = 0
        for i in range(0, 2000, 100):
            results = self.system.search_similar(base[i], top_k=1)
            if results and results[0][0] == i + 1:
                hits += 1
        elapsed = time.time() - start
        assert hits >= 18  # 20 次查询至少召回 18 次精确匹配
        assert elapsed < 5.0

    # ---- 嵌入：词表 + 维度 + OOV ----
    def test_case_11(self):
        vec = self.system.embedding("火焰之剑 攻击力 强大", dim=32)
        assert len(vec) == 32
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-9
        assert len(self.system._vocab) > 0

    def test_case_12(self):
        # OOV 词不报错，且同一文本嵌入结果确定一致
        text = "xyzzy_unknown_token_从未见过的词"
        vec1 = self.system.embedding(text)
        vec2 = self.system.embedding(text)
        assert len(vec1) == self.system.embedding_dim
        assert vec1 == vec2

    def test_case_13(self):
        sword = self.system.embedding("火焰之剑 攻击力 锋利 武器")
        sword2 = self.system.embedding("冰霜之剑 攻击力 锋利 武器")
        potion = self.system.embedding("治疗药水 恢复生命 饮用")
        def cosine(a, b):
            return sum(x * y for x, y in zip(a, b))
        assert cosine(sword, sword2) > cosine(sword, potion)

    # ---- 聚类：KMeans + 评估 ----
    def test_case_14(self):
        rng = random.Random(3)
        group_a = [[10 + rng.gauss(0, 0.1), rng.gauss(0, 0.1)] for _ in range(30)]
        group_b = [[rng.gauss(0, 0.1), 10 + rng.gauss(0, 0.1)] for _ in range(30)]
        vectors = group_a + group_b
        result = self.system.clustering(vectors=vectors, k=2)
        assert result["k"] == 2
        assert len(result["labels"]) == 60
        assert len(result["centroids"]) == 2
        # 同一组的点应被分到同一簇
        labels_a = set(result["labels"][:30])
        labels_b = set(result["labels"][30:])
        assert len(labels_a) == 1
        assert len(labels_b) == 1
        assert labels_a != labels_b

    def test_case_15(self):
        rng = random.Random(4)
        vectors = []
        for cx, cy in [(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)]:
            vectors.extend(
                [[cx + rng.gauss(0, 0.5), cy + rng.gauss(0, 0.5)]
                 for _ in range(20)]
            )
        result = self.system.clustering(vectors=vectors, k=3)
        assert result["inertia"] >= 0.0
        assert -1.0 <= result["silhouette"] <= 1.0
        # 三个分离良好的簇，轮廓系数应明显为正
        assert result["silhouette"] > 0.5
        # 空库聚类不报错
        empty = VectorDB().clustering(k=3)
        assert empty["labels"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
