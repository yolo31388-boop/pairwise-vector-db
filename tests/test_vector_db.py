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


def _norm(vec):
    return math.sqrt(sum(x * x for x in vec))


class TestVectorDB:
    def setup_method(self):
        self.system = VectorDB({"dim": 8, "nlist": 4, "nprobe": 2,
                                "ann_threshold": 32})

    # ---------- 向量插入 ----------
    def test_case_01(self):
        """插入向量返回唯一自增ID，可按ID取回"""
        id1 = self.system.insert_vector([1.0] * 8)
        id2 = self.system.insert_vector([0.5] * 8)
        assert id1 != id2
        assert self.system.get_vector(id1) is not None
        assert self.system.get_vector(id2) is not None

    def test_case_02(self):
        """插入时自动L2归一化，不同量级同向向量单位化"""
        vid = self.system.insert_vector([3.0, 4.0] + [0.0] * 6)
        stored = self.system.get_vector(vid)
        assert _norm(stored) == pytest.approx(1.0, abs=1e-9)
        assert stored[0] == pytest.approx(0.6, abs=1e-9)
        assert stored[1] == pytest.approx(0.8, abs=1e-9)

    def test_case_03(self):
        """重复ID插入报错，支持按ID更新"""
        vid = self.system.insert_vector([1.0] * 8, id=100)
        with pytest.raises(ValueError):
            self.system.insert_vector([1.0] * 8, id=100)
        assert self.system.update_vector(vid, vector=[0.0] * 8) is True
        assert self.system.update_vector(9999, vector=[1.0] * 8) is False

    # ---------- 相似度检索 ----------
    def test_case_04(self):
        """检索使用余弦相似度而非欧氏距离：同向大向量应排最前"""
        a = self.system.insert_vector([2.0, 0.0] + [0.0] * 6)   # 与查询同向，欧氏距离1
        b = self.system.insert_vector([0.9, 0.1] + [0.0] * 6)   # 近邻，欧氏距离更小
        results = self.system.search_similar([1.0, 0.0] + [0.0] * 6, top_k=2)
        assert results[0][0] == a
        assert results[0][1] == pytest.approx(1.0, abs=1e-9)
        assert results[1][0] == b

    def test_case_05(self):
        """TopK生效：只返回K条且按相似度降序"""
        for i in range(10):
            self.system.insert_vector([1.0 - i * 0.05, 0.0] + [0.0] * 6)
        results = self.system.search_similar([1.0, 0.0] + [0.0] * 6, top_k=3)
        assert len(results) == 3
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_case_06(self):
        """已删除的向量不返回"""
        keep = self.system.insert_vector([1.0] * 8, metadata={"name": "keep"})
        gone = self.system.insert_vector([1.0] * 8, metadata={"name": "gone"})
        assert self.system.delete_vector(gone) is True
        assert self.system.delete_vector(gone) is False
        results = self.system.search_similar([1.0] * 8, top_k=10)
        ids = [vid for vid, _ in results]
        assert gone not in ids
        assert keep in ids
        assert len(self.system) == 1

    def test_case_07(self):
        """元数据字典过滤"""
        sword = self.system.insert_vector([1.0] * 8, metadata={"type": "sword", "rarity": 5})
        potion = self.system.insert_vector([1.0] * 8, metadata={"type": "potion", "rarity": 1})
        results = self.system.search_similar([1.0] * 8, top_k=10,
                                             filter={"type": "sword"})
        ids = [vid for vid, _ in results]
        assert sword in ids and potion not in ids

    def test_case_08(self):
        """callable 自定义过滤"""
        ids = [
            self.system.insert_vector([1.0] * 8, metadata={"level": i})
            for i in range(5)
        ]
        results = self.system.search_similar(
            [1.0] * 8, top_k=10, filter=lambda m: m.get("level", 0) >= 3)
        assert {vid for vid, _ in results} == set(ids[3:])

    def test_case_09(self):
        """维度不一致的向量拒绝插入/检索"""
        with pytest.raises(ValueError):
            self.system.insert_vector([1.0, 2.0])
        with pytest.raises(ValueError):
            self.system.search_similar([1.0, 2.0])

    # ---------- ANN 索引 ----------
    def test_case_10(self):
        """ann_index构建IVF索引，桶覆盖全部存活向量"""
        for _ in range(64):
            self.system.insert_vector([random.random() for _ in range(8)])
        stats = self.system.ann_index()
        assert stats["nlist"] == 4
        assert stats["indexed"] == 64
        covered = sum(stats["bucket_sizes"].values())
        assert covered == len(self.system)

    def test_case_11(self):
        """索引构建后新增的向量增量入桶，立即能被搜到"""
        rng = random.Random(3)
        for _ in range(48):
            self.system.insert_vector([rng.random() for _ in range(8)])
        self.system.ann_index()
        new_id = self.system.insert_vector([1.0, 0.0] + [0.0] * 6,
                                           metadata={"tag": "new"})
        results = self.system.search_similar([1.0, 0.0] + [0.0] * 6, top_k=1)
        assert results[0][0] == new_id
        assert self.system.delete_vector(new_id)
        results = self.system.search_similar([1.0, 0.0] + [0.0] * 6, top_k=1)
        assert results[0][0] != new_id

    def test_case_12(self):
        """10万级数据量场景下ANN检索足够快（20000向量，<0.5s，召回率高）"""
        db = VectorDB({"dim": 16, "nlist": 16, "nprobe": 6, "ann_threshold": 256})
        rng = random.Random(42)
        centers = [[rng.gauss(0, 1) for _ in range(16)] for _ in range(8)]
        n = 20000
        for i in range(n):
            c = centers[i % 8]
            vec = [c[d] + rng.gauss(0, 0.05) for d in range(16)]
            db.insert_vector(vec, metadata={"cluster": i % 8})
        db.ann_index()
        query = [centers[0][d] + rng.gauss(0, 0.02) for d in range(16)]

        start = time.time()
        ann = db.search_similar(query, top_k=20)
        elapsed = time.time() - start
        assert elapsed < 0.5

        # 对照：临时关掉ANN阈值做暴力检索
        db.ann_threshold = n + 1
        truth = db.search_similar(query, top_k=20)
        recall = len({i for i, _ in ann} & {i for i, _ in truth}) / 20.0
        assert recall >= 0.7

    # ---------- 文本嵌入 ----------
    def test_case_13(self):
        """embedding输出固定维度且归一化"""
        vec = self.system.embedding("health potion")
        assert len(vec) == 8
        assert _norm(vec) == pytest.approx(1.0, abs=1e-9)

    def test_case_14(self):
        """词表自动扩充，OOV词可处理，相似文本相似度更高"""
        v1 = self.system.embedding("sword fire damage")
        before = self.system.vocab_size
        v2 = self.system.embedding("sword fire damage")
        v3 = self.system.embedding("zzz qqq xxx yyy")  # 全是OOV
        assert self.system.vocab_size > before
        assert math.fsum(x * y for x, y in zip(v1, v2)) == pytest.approx(1.0, abs=1e-9)
        sim_near = math.fsum(x * y for x, y in zip(
            v1, self.system.embedding("sword fire boost")))
        sim_far = math.fsum(x * y for x, y in zip(v1, v3))
        assert sim_near > sim_far

    # ---------- 聚类 ----------
    def test_case_15(self):
        """KMeans聚类正确分组，并返回inertia/silhouette质量评估"""
        rng = random.Random(7)
        group_a, group_b = [], []
        for _ in range(30):
            va = [1.0 + rng.gauss(0, 0.05)] + [0.0 + rng.gauss(0, 0.05)] + [0.0] * 6
            vb = [0.0 + rng.gauss(0, 0.05)] + [1.0 + rng.gauss(0, 0.05)] + [0.0] * 6
            group_a.append(self.system.insert_vector(va, metadata={"g": "a"}))
            group_b.append(self.system.insert_vector(vb, metadata={"g": "b"}))
        result = self.system.clustering(k=2)
        assert set(result["clusters"].keys()) == {0, 1}
        assert len(result["centroids"]) == 2
        assert result["inertia"] >= 0.0
        assert -1.0 <= result["silhouette"] <= 1.0
        assert result["silhouette"] > 0.5

        groups = [set(ids) for ids in result["clusters"].values()]
        assert any(set(group_a).issubset(g) for g in groups)
        assert any(set(group_b).issubset(g) for g in groups)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
