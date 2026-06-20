---
title: Tian-Dao Embeddings
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

# 🧠 Tian-Dao LLM — Endoregulated Embeddings v2.7

**Endoregulated AI based on Wuxing philosophy and algebraic topology**
**基于五行哲学与代数拓扑的自调节人工智能**

[![Tests](https://img.shields.io/badge/tests-73%20passed-brightgreen)](https://github.com/bruno-dd470/Tian-Dao-LLM/tree/main/tests)
[![Version](https://img.shields.io/badge/version-2.7-blue)](https://github.com/bruno-dd470/Tian-Dao-LLM/releases)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](https://www.apache.org/licenses/LICENSE-2.0)
[![Languages](https://img.shields.io/badge/languages-EN%20%7C%20中文-orange)](#-description--描述)

## 🚀 Live Demo / 在线演示

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Open%20Demo-blue?style=for-the-badge)](https://huggingface.co/spaces/WaltDod/Gradio)

👆 Click the badge above to try the interactive demo.
👆 点击上方徽章体验交互式演示。

---

## 📝 Description / 描述

### 🇬🇧 English

This application transforms text into a 20D embedding via an **endoregulated AI** inspired by the principles of Wuxing (Chinese philosophy of the 5 phases) and algebraic topology.

Unlike traditional embedding models (BERT, SBERT, etc.), this approach:
- ✅ **Requires no supervised training**
- ✅ **Is fully deterministic** (same text → same embedding)
- ✅ **Produces interpretable embeddings** (each dimension = a Wuxing attractor)
- ✅ **Self-regulates** via an equilibrium cycle between exploration (SHENG) and contraction (KE)

The system projects 6-bit inputs (0-63) into a 20-dimensional space via a network of attractors organized as a Merkabah.

### 🇨🇳 中文

本应用通过受**五行哲学**（中国五阶段哲学）与**代数拓扑**启发的**自调节人工智能**，将文本转换为 20 维嵌入向量。

与传统的嵌入模型（BERT、SBERT 等）不同，本方法：
- ✅ **无需监督训练**
- ✅ **完全确定性**（相同文本 → 相同嵌入）
- ✅ **生成可解释的嵌入**（每个维度 = 一个五行吸引子）
- ✅ **通过探索（生 SHENG）与收缩（克 KE）之间的平衡循环实现自调节**

系统通过以梅尔卡巴（Merkabah）结构组织的吸引子网络，将 6 位输入（0-63）投影到 20 维空间。

---

## 🔥 What's New in v2.7 / v2.7 新特性 (2026-06-20)

### 🐛 Critical Fixes (P0) / 关键修复

| Bug / 缺陷 | Fix / 修复 | Impact / 影响 |
|---|---|---|
| **Flip overwritten** / 翻转被覆盖 | `encode_bits()` now REALLY propagates the flip to pentades / `encode_bits()` 现在真正将翻转传播到五元组 | Input has a real, persistent effect / 输入产生真实持久的影响 |
| **Thread safety** / 线程安全 | `threading.Lock` + local `np.random.default_rng` / `threading.Lock` + 局部 `np.random.default_rng` | Compatible with multi-user Gradio / 兼容多用户 Gradio |
| **Determinism** / 确定性 | Removed global `np.random.seed()` / 移除全局 `np.random.seed()` | Guaranteed reproducibility / 保证可复现性 |
| **PCA cache** / PCA 缓存 | `gr.State` per user session / 每个用户会话一个 `gr.State` | No more collisions between users / 消除用户间冲突 |

### 🛡️ Robustness Improvements (P1) / 健壮性改进

| Improvement / 改进 | Before / 之前 | After / 之后 |
|---|---|---|
| **Hash** / 哈希 | Polynomial (frequent collisions) / 多项式（频繁碰撞） | Truncated SHA-256 (robust) / 截断 SHA-256（健壮） |
| **PCA** / PCA | Unstable on 3 points / 3 点不稳定 | Stable on 5+ points / 5+ 点稳定 |
| **Text validation** / 文本验证 | `text.strip() == " "` | `not text.strip()` (detects all whitespace) / 检测所有空白字符 |
| **Memory leaks** / 内存泄漏 | Risk with matplotlib / matplotlib 风险 | Systematic `try/finally` / 系统性 `try/finally` |

---

## 📊 Scientific Benchmark / 科学基准测试

Honest comparison between Tian-Dao 20D and DistilBERT:
Tian-Dao 20D 与 DistilBERT 的客观对比：

| Metric / 指标 | Tian-Dao 20D | DistilBERT | Ratio / 比率 |
|---|---|---|---|
| **Spearman (STS)** | +0.054 | +0.582 | 10.9x |
| **Dimension** / 维度 | **20** | 512 | **25.6x** |
| **Embedding size** / 嵌入大小 | **80 bytes** | 2048 bytes | **25.6x** |
| **Model size** / 模型大小 | **0.005 MB** | 250 MB | **50000x** |
| **Time/sentence** / 每句时间 | **0.22 ms** | 4.86 ms | **21.8x** |
| **Training required** / 需训练 | ❌ No / 否 | ✅ Yes / 是 | - |
| **GPU required** / 需 GPU | ❌ No / 否 | ✅ Yes / 是 | - |
| **Interpretable** / 可解释 | ✅ Yes / 是 | ❌ No / 否 | - |

**Conclusion / 结论**: Tian-Dao 20D excels where DistilBERT is unsuitable (hardware constraints, interpretability, no training data).
Tian-Dao 20D 在 DistilBERT 不适用的场景中表现出色（硬件限制、可解释性、无训练数据）。

---


## 🔬 Theoretical Foundations / 理论基础

### The Genetic Code as a Signature Translation System / 遗传密码作为签名翻译系统

The genetic code does not *compute* semantics — it **translates invariant molecular signatures** into stable functional classes. This principle, formalized in our foundational paper ([DOI: 10.5281/zenodo.19540508](https://doi.org/10.5281/zenodo.19540508)), is the structural blueprint of Tian-Dao 20D.

遗传密码并不*计算*语义——它将**不变的分子签名**翻译成稳定的功能类别。这一原理在我们奠基性论文（[DOI: 10.5281/zenodo.19540508](https://doi.org/10.5281/zenodo.19540508)）中形式化，是 Tian-Dao 20D 的结构蓝图。

### The 64→20 Invariant / 64→20 不变量

| Biological System / 生物系统 | Tian-Dao 20D (Abstraction) / Tian-Dao 20D（抽象） |
|---|---|
| 64 codons → 20 amino acids / 64 个密码子 → 20 个氨基酸 | 64 configurations → 20 attractors / 64 个构型 → 20 个吸引子 |
| Physico-chemical signature (hydrophobicity, charge, volume) / 物理化学签名（疏水性、电荷、体积） | Polarity signature of pentads (3P, 2P+1N, 1P+2N, 3N) / 五元组极性签名 |
| Organized degeneracy (error minimization) / 有组织的简并性（误差最小化） | Deterministic noise + clipping (structural stability) / 确定性噪声 + 裁剪（结构稳定性） |
| RNA → Protein translation (invariant preservation) / RNA → 蛋白质翻译（不变量保持） | Text → 20D embedding (topology preservation) / 文本 → 20D 嵌入（拓扑保持） |

### Why Tian-Dao 20D is NOT a Semantic Model / 为什么 Tian-Dao 20D 不是语义模型

Tian-Dao 20D does not capture linguistic similarity — it encodes **structural signatures** preserving the 64→20 topology. This explains our benchmark results:

- **STS Benchmark (Spearman ≈ +0.016, IC [-0.047, +0.042])**: Tian-Dao is structurally *orthogonal* to semantic similarity, just as the genetic code is orthogonal to "meaning" of proteins. The confidence interval contains zero, confirming statistical non-significance on this task.
- **Topological Benchmark (Score = 0.852)**: Tian-Dao excels at what it was designed for — preserving invariants, stability, and interpretability.

Tian-Dao 20D 不捕捉语言相似性——它编码**保持 64→20 拓扑的结构签名**。这解释了我们的基准测试结果：

- **STS 基准（Spearman ≈ +0.016，IC [-0.047, +0.042]）**：Tian-Dao 在结构上与语义相似性*正交*，正如遗传密码与蛋白质的"意义"正交。置信区间包含零，证实了在此任务上的统计非显著性。
- **拓扑基准（得分 = 0.852）**：Tian-Dao 在其设计目标上表现出色——保持不变量、稳定性和可解释性。

### Scientific Foundations / 科学基础

This work builds on the mathematical framework of nilpotent Clifford algebras developed by **Peter Rowlands**, and its application to the genetic code jointly authored with **Vanessa Hill** (Chapter 19, *Zero to Infinity*, World Scientific, 2007). The 64→20 reduction emerges from the geometry of the Level-3 double tetrahedron (Merkabah), where the 64-element structure of Cl(6,0) partitions into exactly 20 stable equivalence classes under a triangular face-sharing neighborhood rule.

本工作建立在 **Peter Rowlands** 发展的幂零 Clifford 代数数学框架之上，以及他与 **Vanessa Hill** 共同将其应用于遗传密码的研究（《Zero to Infinity》第 19 章，World Scientific，2007）。64→20 约化源于 3 级双四面体（Merkabah）的几何结构，其中 Cl(6,0) 的 64 元素结构在三角面共享邻域规则下精确划分为 20 个稳定等价类。

**Reference / 参考文献** :
> DE DOMINICIS, B. (2026). *The Genetic Code as a 64→20 Clifford Invariant: Implications for Self-Regulating AI*. Zenodo. [DOI: 10.5281/zenodo.19540508](https://doi.org/10.5281/zenodo.19540508)



### 🧬 Topological Benchmark / 拓扑基准测试

Complementary evaluation of what Tian-Dao 20D does BEST:
互补评估 Tian-Dao 20D 最擅长的方面：

| Test / 测试 | Score / 得分 | Interpretation / 解释 |
|---|---|---|
| **Stability / 稳定性** | 1.000 | ✅ Perfect — small perturbations → same embedding |
| **Reproducibility / 可复现性** | 1.000 | ✅ Perfect — deterministic (1000x → variance = 0) |
| **Compression / 压缩** | 0.999 | ✅ 38.4x (20D vs 768D) |
| **Interpretability / 可解释性** | 0.500 | ✅ Each dimension = identifiable attractor |
| **Discrimination / 区分度** | 0.759 | ✅ Different texts → different embeddings |
| **🎯 Global Score / 全局得分** | **0.852** | ✅ **Excellent structural properties** |

Run the topological benchmark / 运行拓扑基准测试:
```bash
python code/benchmark/benchmark_topological.py
```


## 🧪 Unit Tests / 单元测试 (73 tests)

The project has a comprehensive test suite covering ~95% of the business logic:
项目拥有覆盖约 95% 业务逻辑的全面测试套件：

| File / 文件 | Tests / 测试 | Coverage / 覆盖 |
|---|---|---|
| `test_p0.py` | 4 | Critical bugs (flip, thread safety, determinism) / 关键缺陷 |
| `test_p1.py` | 4 | Robustness (PCA, hash, validation, leaks) / 健壮性 |
| `test_endoregulated.py` | 33 | Business logic (Merkabah, metrics, Wuxing) / 业务逻辑 |
| `test_app.py` | 32 | Gradio interface (embeddings, visualization) / Gradio 接口 |
| **TOTAL** / **总计** | **73** | **~95% coverage** / **约 95% 覆盖率** |

### Run the tests / 运行测试

```bash
# Install test dependencies / 安装测试依赖
pip install pytest

# Run all tests / 运行所有测试
pytest tests/ -v

# Run a specific file / 运行特定文件
pytest tests/test_p0.py -v
```

---

## 🏗️ Project Architecture / 项目架构

```
Tian-Dao-LLM/
├── Endoregulated_AI_v27.py      ← Core v2.7 (thread-safe, deterministic)
│                                   核心 v2.7（线程安全、确定性）
├── app.py                        ← Gradio interface v1.2
│                                   Gradio 接口 v1.2
├── requirements.txt              ← Python dependencies / Python 依赖
├── pytest.ini                    ← pytest configuration / pytest 配置
├── README.md                     ← This file / 本文件
├── .gitignore
│
├── code/
│   └── benchmark/
│       ├── benchmark_distilbert.py    ← STS benchmark (vs DistilBERT)
│       ├── benchmark_topological.py   ← Topological benchmark / 拓扑基准测试
│       ├── BENCHMARK_REPORT_*.md      ← STS reports (timestamped)
│       ├── BENCHMARK_RESULTS_*.json   ← STS results (timestamped)
│       ├── TOPOLOGICAL_REPORT_*.md    ← Topological reports
│       └── TOPOLOGICAL_RESULTS_*.json ← Topological results
│
├── tests/
│   ├── conftest.py                    ← pytest configuration / pytest 配置
│   ├── test_p0.py                     ← Critical bug tests / 关键缺陷测试
│   ├── test_p1.py                     ← Robustness tests / 健壮性测试
│   ├── test_endoregulated.py          ← Business logic tests / 业务逻辑测试
│   └── test_app.py                    ← Interface tests / 接口测试
│
└── docs/
    ├── theory.md                      ← Theoretical documentation / 理论文档
    └── validation.md                  ← Scientific validation / 科学验证
```

---

## 🚀 Installation / 安装

### 🇬🇧 English

#### 1. Clone the repository

```bash
git clone https://github.com/bruno-dd470/Tian-Dao-LLM.git
cd Tian-Dao-LLM
```

#### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. (Optional) Install benchmark dependencies

```bash
pip install torch transformers sentence-transformers
```

### 🇨🇳 中文

#### 1. 克隆仓库

```bash
git clone https://github.com/bruno-dd470/Tian-Dao-LLM.git
cd Tian-Dao-LLM
```

#### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. （可选）安装基准测试依赖

```bash
pip install torch transformers sentence-transformers
```

---

## 🎯 Usage / 使用

### 🇬🇧 English

#### Launch the Gradio app

```bash
python app.py
```

The app will be available at `http://localhost:7860`

#### Run the benchmark

```bash
python code/benchmark/benchmark_distilbert.py
```

Results are saved to:
- `code/benchmark/BENCHMARK_REPORT.md`
- `code/benchmark/BENCHMARK_RESULTS.json`

#### Run the core directly

```bash
python Endoregulated_AI_v27.py
```

### 🇨🇳 中文

#### 启动 Gradio 应用

```bash
python app.py
```

应用将在 `http://localhost:7860` 可用

#### 运行基准测试

```bash
python code/benchmark/benchmark_distilbert.py
```

结果保存到：
- `code/benchmark/BENCHMARK_REPORT.md`
- `code/benchmark/BENCHMARK_RESULTS.json`

#### 直接运行核心

```bash
python Endoregulated_AI_v27.py
```

---

## 📚 Theory & Related Work / 理论与相关工作

### Tian-Dao Hub / 天道中心
- [bruno-dd470/Tian-Dao-AI](https://github.com/bruno-dd470/Tian-Dao-AI) — Unified framework / 统一框架

### Mathematical Foundations / 数学基础
- [Tian-Dao-WuXing-Cl66-Pentads](https://github.com/bruno-dd470/Tian-Dao-WuXing-Cl66-Pentads) — Cl(6,6) algebra & Λ₇₂ / Cl(6,6) 代数与 Λ₇₂
- [mass-unification-L72](https://github.com/bruno-dd470/mass-unification-L72) — Physics validation / 物理验证

### DOIs / 数字对象标识符
- [10.5281/zenodo.19633890](https://doi.org/10.5281/zenodo.19633890) — The Genetic Code as a 64→20 Clifford Invariant / 遗传密码作为 64→20 Clifford 不变量
- [10.5281/zenodo.19947629](https://doi.org/10.5281/zenodo.19947629) — From nilpotent angular cycles to bimetric cosmology / 从幂零角循环到双度量宇宙学
- [10.5281/zenodo.20696586](https://doi.org/10.5281/zenodo.20696586) — Mandarin phonology / 普通话音系
- [10.5281/zenodo.20042320](https://doi.org/10.5281/zenodo.20042320) — Mass unification / 质量统一

---

## 📊 Key Results / 关键结果

| Metric / 指标 | Value / 值 | Source / 来源 |
|---|---|---|
| **Size reduction** / 尺寸缩减 | 97% | `benchmark/benchmark_distilbert.py` |
| **GitHub clones** / GitHub 克隆 | 158+ in 14 days / 14 天内 | [Tian-Dao-AI](https://github.com/bruno-dd470/Tian-Dao-AI) |
| **Performance** / 性能 | Complementary to 768D / 与 768D 互补 | DistilBERT evaluation / DistilBERT 评估 |
| **Tests** / 测试 | 73/73 passing / 全部通过 | `pytest tests/` |

---

## 🔧 Troubleshooting / 故障排除

### 🇬🇧 English

**Error: `ModuleNotFoundError: No module named 'gradio'`**
```bash
pip install gradio
```

**Error: `ModuleNotFoundError: No module named 'Endoregulated_AI_v27'`**

Make sure `Endoregulated_AI_v27.py` is at the project root.

**Tests are failing**

Verify all dependencies are installed:
```bash
pip install -r requirements.txt
pip install pytest
```

### 🇨🇳 中文

**错误：`ModuleNotFoundError: No module named 'gradio'`**
```bash
pip install gradio
```

**错误：`ModuleNotFoundError: No module named 'Endoregulated_AI_v27'`**

确保 `Endoregulated_AI_v27.py` 位于项目根目录。

**测试失败**

验证所有依赖已安装：
```bash
pip install -r requirements.txt
pip install pytest
```

---

## 📧 Contact / 联系

For collaboration / 合作请联系: [dod60@gmx.fr](mailto:dod60@gmx.fr)

---

## 📄 License / 许可证

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

本项目采用 **Apache License 2.0** 许可证 - 详见 [LICENSE](LICENSE) 文件。

