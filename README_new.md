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

# 🧠 Tian-Dao LLM — Endoregulated Topological Signatures v2.7
**Endoregulated AI based on Wuxing philosophy and algebraic topology**  
基于五行哲学与代数拓扑的自调节人工智能 | IA endorégulée basée sur la philosophie Wuxing et la topologie algébrique

![tests](https://img.shields.io/badge/tests-73%20passed-brightgreen)
![version](https://img.shields.io/badge/version-2.7-blue)
![python](https://img.shields.io/badge/python-3.11+-blue)
![license](https://img.shields.io/badge/license-Apache%202.0-green)
![languages](https://img.shields.io/badge/languages-EN%20%7C%20FR%20%7C%20中文-orange)

---

## ⚠️ Important: Topological Signatures, Not Semantic Embeddings
**Tian-Dao 20D is NOT a semantic model.** It encodes **structural signatures** preserving the 64→20 Clifford invariant. This is orthogonal to linguistic similarity, exactly as the genetic code is orthogonal to the "meaning" of proteins.  
**STS Benchmark:** Spearman ≈ +0.016 (IC [-0.047, +0.042]) → structurally orthogonal to semantics.  
**Topological Benchmark:** Score = 0.852 → excellent at invariant preservation, stability, and interpretability.

---

## 🚀 Live Demo / 在线演示

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Open%20Demo-blue?style=for-the-badge)](https://waltdod-gradio.hf.space)

[![Hugging Face README](https://img.shields.io/badge/📄%20Hugging%20Face-README-blue?style=for-the-badge)](https://huggingface.co/spaces/WaltDod/Gradio/blob/main/README.md)

Bilingual interface (FR/EN) with real-time 20D signature generation, Wuxing regime tracking, and PCA projection.

👆 Click the badges above to try the demo or view the README on Hugging Face.
👆 点击上方徽章体验演示或查看 Hugging Face 上的 README。

---

## 📖 Theoretical Foundations / Fondements théoriques / 理论基础

This implementation is the executable counterpart of the mathematical framework described in the companion repository:

🔗 **[Tian-Dao-AI Repository](https://github.com/bruno-dd470/Tian-Dao-AI)**  
📄 Full theoretical manuscript: **[complexity.pdf](https://github.com/bruno-dd470/Tian-Dao-AI/blob/main/docs/pdf/complexity.pdf)** | **[complexité.pdf](https://github.com/bruno-dd470/Tian-Dao-AI/blob/main/docs/pdf/complexité.pdf)**  
🔬 Zenodo DOI: [10.5281/zenodo.19540508](https://doi.org/10.5281/zenodo.19540508)

Key theoretical concepts implemented:
- `Cl(6,0)` → 20 stable attractors (Merkabah geometry)
- Pentadic generators (P1–P6, N1–N6) and tropical belts (CP/CN)
- Wuxing dynamics (Sheng/KE cycles) for endogenous regulation
- 64→20 topological filtration via triangular face-sharing rule

---

## 🧬 Benchmark Results / Résultats de benchmark / 基准测试结果

### 📊 Topological Benchmark (What Tian-Dao excels at)
| Test | Score | Interpretation |
|------|-------|----------------|
| Stability | 1.000 | ✅ Perfect — small perturbations → same signature |
| Reproducibility | 1.000 | ✅ Perfect — deterministic (1000x → variance = 0) |
| Compression | 0.999 | ✅ 38.4x (20D vs 768D) |
| Interpretability | 0.500 | ✅ Each dimension = identifiable attractor |
| Discrimination | 0.759 | ✅ Different texts → different signatures |
| 🎯 **Global Score** | **0.852** | ✅ **Excellent structural properties** |

Run it: `python code/benchmark/benchmark_topological.py`

### 📐 STS Benchmark (Semantic Similarity)
| Metric | Tian-Dao 20D | DistilBERT | Notes |
|--------|--------------|------------|-------|
| Spearman | +0.016 | +0.582 | Tian-Dao is orthogonal to semantics |
| Dimension | 20 | 512 | 25.6x smaller |
| Signature size | 80 bytes | 2048 bytes | 25.6x smaller |
| Model size | 0.005 MB | 250 MB | 50,000x smaller |
| Training | ❌ None | ✅ Required | Zero data needed |
| GPU | ❌ Not required | ✅ Required | CPU-only |

**Conclusion:** Tian-Dao 20D is designed for **hardware-constrained, interpretable, training-free topological encoding** — NOT for semantic search or text classification.

---

## 🏗️ Architecture & v2.7 Features / Architecture et fonctionnalités v2.7 / 架构与 v2.7 特性

| Feature | Before (v2.5) | After (v2.7) |
|---------|---------------|--------------|
| Flip propagation | Overwritten | ✅ Real pentade-level propagation |
| Thread safety | Global state | ✅ `threading.Lock` + local RNG |
| Determinism | `np.random.seed()` | ✅ Removed, fully reproducible |
| Hashing | Polynomial (collisions) | ✅ Truncated SHA-256 |
| PCA cache | Shared state | ✅ `gr.State` per session |
| Tests | Partial | ✅ 73/73 passing (95% coverage) |

Files:
- `Endoregulated_AI_v27.py` — Core engine (thread-safe, deterministic)
- `app.py` — Gradio interface (bilingual FR/EN)
- `requirements.txt` — Dependencies
- `tests/` — Full test suite (`pytest`)

---

## 🧪 Unit Tests / Tests unitaires / 单元测试
| File | Tests | Coverage |
|------|-------|----------|
| `test_p0.py` | 4 | Critical bugs (flip, thread safety, determinism) |
| `test_p1.py` | 4 | Robustness (PCA, hash, validation, leaks) |
| `test_endoregulated.py` | 33 | Business logic (Merkabah, metrics, Wuxing) |
| `test_app.py` | 32 | Gradio interface (signatures, visualization) |
| **TOTAL** | **73** | **~95%** |

Run: `pytest tests/ -v`

---

## 🚀 Installation & Usage / Installation et utilisation / 安装与使用

### 🇬🇧 English
1. Clone & install:
   ```bash
   git clone https://github.com/bruno-dd470/Tian-Dao-LLM.git
   cd Tian-Dao-LLM
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Launch app: `python app.py` → `http://localhost:7860`
3. Run benchmarks: `python code/benchmark/benchmark_topological.py`

### 🇨🇳 中文
1. 克隆并安装（见上文）
2. 启动应用：`python app.py` → `http://localhost:7860`
3. 运行基准测试：`python code/benchmark/benchmark_topological.py`

---

## 🔗 Cross-Repository Structure / Structure inter-dépôts / 跨仓库结构

| Repository | Focus | Key Files |
|------------|-------|-----------|
| **[Tian-Dao-LLM](https://github.com/bruno-dd470/Tian-Dao-LLM)** | 💻 Code v2.7, Gradio, benchmarks, tests | `Endoregulated_AI_v27.py`, `app.py`, `tests/` |
| **[Tian-Dao-AI](https://github.com/bruno-dd470/Tian-Dao-AI)** | 📖 Theory, `complexity.pdf`, philosophy | `docs/pdf/complexity.pdf`, Zenodo DOIs |
| **[Tian-Dao-WuXing-Cl66-Pentads](https://github.com/bruno-dd470/Tian-Dao-WuXing-Cl66-Pentads)** | 📐 Cl(6,6) algebra & Λ₇₂ | Algebraic derivations, pentadic mappings |

---

## 📚 Related Work / Travaux connexes / 相关工作
- [bruno-dd470/Tian-Dao-AI](https://github.com/bruno-dd470/Tian-Dao-AI) — Theoretical framework & manuscripts
- [Tian-Dao-WuXing-Cl66-Pentads](https://github.com/bruno-dd470/Tian-Dao-WuXing-Cl66-Pentads) — Mathematical foundations
- [mass-unification-L72](https://github.com/bruno-dd470/mass-unification-L72) — Physics validation
- Peter Rowlands & Vanessa Hill, *Zero to Infinity* (Ch. 19) — Genetic code & Clifford algebras

---

## 📧 Contact / Contact / 联系
Collaboration / Coopération / 合作: [dod60@gmx.fr](mailto:dod60@gmx.fr)

---

## 📄 License / Licence / 许可证
Apache License 2.0 — See [LICENSE](LICENSE) for details.  
This project is open for academic, institutional, and industrial use, provided proper attribution is given.

