# 🧪 Cikgu AI Kimia — SPM Chemistry AI Tutor

> Production-ready AI tutor for Malaysian SPM Chemistry students
> Built with FastAPI + FAISS + Groq LLM + Telegram Bot + Firebase Smart Memory
> Now with Universal Solver + Multi-State SPM Coverage 📚

---

## 🌐 Live URLs

| Service | URL |
|---|---|
| API (FastAPI) | https://cikgiaikimia.onrender.com |
| API Docs | https://cikgiaikimia.onrender.com/docs |
| Health Check | https://cikgiaikimia.onrender.com/api/health |
| Telegram Bot | @TicerHawaAIBot |
| GitHub | https://github.com/msallehroslan/CikgiAIKimia |
| Firebase | https://console.firebase.google.com/u/1/project/cikgu-kimia-66b7b/firestore |

---

## ✅ FULLY WORKING FEATURES (v3.2.0 deployed)

### Infrastructure
- [x] FastAPI backend deployed on Render.com (Singapore)
- [x] Python 3.11.9 pinned (.python-version file)
- [x] fastembed (ONNX) — RAM ~150MB (free tier compatible)
- [x] FAISS indexes: 514 theory + 176 calculations + 13 QA = 703 vectors
- [x] Groq LLM — 3 model architecture (70b theory, 8b explain, scout vision)
- [x] Firebase Firestore connected (project: cikgu-kimia-66b7b)
- [x] Render auto-deploy on git push
- [x] Telegram webhook inside FastAPI (no separate worker)

### Solvers (Deterministic Python — v3.2.0 deployed)
- [x] Mol calculations (moles_from_mass, moles_from_volume, moles_multi)
- [x] pH / pOH calculations
- [x] Titration — find molarity dan find volume (nisbah 1:1)
- [x] Thermochemistry — entalpi peneutralan, pemelarutan, pembakaran
- [x] Stoichiometry mass→mass dan mass→volume
- [x] Voltaic cell — E0cell = E0katod - E0anod
- [x] Molarity from delta H
- [x] Concentration g/dm3, Dilution (M1V1=M2V2)
- [x] Empirical formula
- [x] Relative atomic mass dari isotop
- [x] Atomic structure
- [x] Oxidation number
- [x] Rate of reaction
- [x] JMR / Molar mass
- [x] Mass from molarity

### Photo / Vision Support (v3.2.0)
- [x] Pelajar hantar gambar soalan ke Telegram — bot jawab
- [x] Scout extract + interpret dalam SATU call
- [x] Auto-derive formula dari nama IUPAC
- [x] Support MCQ + struktur organik

---

## 🔧 PENDING FIXES — v3.4.0 (BELUM PUSH)

### Fail: solver/solver_engine.py
**6 Bug Kritikal yang dah difix (dalam solver_engine_fixed.py):**

| # | Bug | Kesan | Fix |
|---|---|---|---|
| 1 | Concentration/dilution crash | Ralat bila tanya kepekatan | Handle exception + fix routing |
| 2 | Thermochem J→kJ salah | ΔH=-2.86 bukan -53.76 | Divide by 1000 betul |
| 3 | Stoich crash bila input mol | Crash untuk soalan "0.5 mol KI →" | Accept mol terus |
| 4 | Kepekatan molar tidak dikira | Dapat g/dm³ sahaja, bukan mol/dm³ | Kira kedua-dua |
| 5 | Titration nisbah diabaikan | H₂SO₄+2NaOH dapat 40cm³ bukan 80cm³ | Ambil kira stoich ratio |
| 6 | MOLFROMDH solver tidak lengkap | Hanya kira Q, tidak sambung ke kemolaran | Lengkapkan 3 langkah |

### Fail: solver/solver_engine.py (universal_spm_solver.py)
**Solver Baru — cover semua pattern SPM pelbagai negeri:**

| Solver Baru | Pattern | Contoh (Negeri) |
|---|---|---|
| stoich_from_molarity | Kemolaran → jisim produk | Terengganu Q38 |
| thermochem_reverse | Beri Q, cari ΔT | Terengganu Q34 |
| stoich_vol_to_vol | Gas → gas (nisbah) | Terengganu Q37 |
| ph_from_OH | OH⁻ → pH | Terengganu Q25 |
| stoich_mass_to_vol | Jisim → isipadu gas | Terengganu Q33 |

### Fail: api/vision.py
**Vision Prompt dikemas kini (dalam vision_prompt_johor2021.py):**
- Format output berstruktur: JENIS, SOALAN, DATA_NOMBOR, FORMULA_KIMIA, PERSAMAAN_KIMIA, JENIS_PENGIRAAN
- Type detection automatik → route ke solver betul
- Cover semua 10 jenis soalan SPM (graf, jadual, struktur, neraca, dll)

---

## 📊 SPM MULTI-STATE COVERAGE (selepas v3.4.0)

| Negeri | Tahun | Kertas | Status |
|---|---|---|---|
| Johor | 2021 | K1 + K2 + Skema | ✅ Dianalisis |
| Terengganu | 2021 | K1 | ✅ Dianalisis |
| Selangor | 2024 | K1 (scanned) | ⏳ Pending OCR |
| Kedah/Pahang/Perak | - | - | 🔄 Pending upload |

### Soalan Pengiraan — Status Coverage

| Jenis Soalan | Johor | Terengganu | Status |
|---|---|---|---|
| Stoich mass→mass | ✅ | ✅ | Selesai |
| Stoich mass→vol | ✅ | ✅ | Selesai |
| Stoich vol→mass | ✅ | - | Selesai |
| Stoich vol→vol | - | ✅ | Selesai |
| Stoich dari kemolaran | - | ✅ | Selesai |
| Thermochem forward (ΔH) | ✅ | ✅ | Selesai |
| Thermochem reverse (ΔT) | - | ✅ | Selesai |
| pH dari H⁺ | ✅ | ✅ | Selesai |
| pH dari OH⁻ | ✅ | ✅ | Selesai |
| Titration 1:1 | ✅ | - | Selesai |
| Titration 1:2 | ✅ | ✅ | Selesai |
| Voltaic cell | ✅ | - | Selesai |
| JMR/Molar mass | ✅ | ✅ | Selesai |
| Formula empirik | ✅ | - | Selesai |
| Kadar tindak balas | ✅ | - | Selesai |
| Jisim atom relatif (isotop) | ✅ | - | Selesai |
| % komposisi (hidrat) | ✅ | - | Selesai |
| Molarity dari ΔH | ✅ | - | Selesai |

---

## 📁 PROJECT STRUCTURE

```
CikgiAIKimia/
├── api/
│   ├── main.py              ← FastAPI + Telegram webhook (v3.2.0)
│   ├── memory.py            ← Smart memory (shared cache + personal)
│   └── vision.py            ← Multi-provider vision (Groq/Gemini/Tesseract)
│                               ⚠️ PENDING: Update VISION_SYSTEM_PROMPT
├── rag/
│   ├── embedder.py          ← fastembed ONNX
│   ├── retriever.py         ← FAISS retriever
│   ├── chunker.py           ← Markdown chunker
│   ├── indexer.py           ← FAISS index manager
│   └── metadata_tagger.py   ← chunk metadata tagger
├── solver/
│   ├── solver_engine.py     ← deterministic chemistry solver
│   │                           ⚠️ PENDING: Merge solver_engine_fixed.py
│   │                           ⚠️ PENDING: Merge universal_spm_solver.py
│   ├── extractor.py         ← question parser
│   │                           ⚠️ PENDING: Add vision output extractor
│   ├── router.py            ← task router
│   │                           ⚠️ PENDING: Add new solver type mappings
│   ├── formula_parser.py    ← chemical formula parser
│   ├── equation_parser.py   ← equation parser
│   └── units.py             ← unit conversions
├── scripts/
│   └── build_index_v2.py   ← FAISS index builder
├── faiss_indexes/
│   ├── index_theory.faiss       (514 vectors)
│   ├── index_calculations.faiss (176 vectors)
│   └── index_qa.faiss           (13 vectors)
├── knowledge_base/          ← SPM Markdown notes
├── requirements.txt
├── render.yaml
└── .python-version          ← 3.11.9
```

---

## ⚙️ ENVIRONMENT VARIABLES (Render)

| Key | Value |
|---|---|
| GROQ_API_KEY | your_groq_key |
| GROQ_MODEL | llama-3.3-70b-versatile |
| GROQ_EXPLAIN_MODEL | llama-3.1-8b-instant |
| GROQ_VISION_MODEL | meta-llama/llama-4-scout-17b-16e-instruct |
| VISION_PROVIDER | groq |
| TELEGRAM_BOT_TOKEN | your_bot_token |
| API_BASE_URL | https://cikgiaikimia.onrender.com |
| FAISS_INDEX_DIR | ./faiss_indexes |
| KB_DIR | ./knowledge_base |
| RETRIEVAL_THRESHOLD | 0.30 |
| MAX_CONTEXT_CHARS | 3000 |
| GOOGLE_APPLICATION_CREDENTIALS_JSON | {entire Firebase JSON} |

---

## 📦 KEY PACKAGES

```txt
fastembed==0.3.6
faiss-cpu==1.8.0
langchain==0.2.16
fastapi==0.111.0
groq==0.9.0
python-telegram-bot==21.3
firebase-admin==6.5.0
Pillow==10.4.0
numpy==1.26.4
```

---

## 🧠 ANSWER PIPELINE (v3.4.0 — target)

```
TEKS masuk              GAMBAR masuk
     |                       |
Detect BM/EN         Groq Scout (1 call)
     |               UNIVERSAL_VISION_PROMPT
Firebase cache           |
HIT -> Return        Structured output:
     |               JENIS/SOALAN/DATA/FORMULA
     |               PERSAMAAN/JENIS_PENGIRAAN
Extractor                |
     |               extract_from_vision_output()
     |               → detect solver type
     |               → build clean question
Universal Router <----------/
     |
JENIS_PENGIRAAN → solver function mapping:
stoich_mass         → solve_stoichiometry(want="mass")
stoich_vol          → solve_stoichiometry(want="volume_rtp")
stoich_from_molarity→ solve_stoichiometry(given_molarity=...)
thermochem_forward  → solve_thermochemistry_full(want="delta_H")
thermochem_reverse  → solve_thermochemistry_full(want="delta_T") [BARU]
ph_from_H           → solve_ph_universal(ion_type="H+")
ph_from_OH          → solve_ph_universal(ion_type="OH-") [BARU]
titration           → solve_titration_universal(coeff1, coeff2)
voltaic_cell        → solve_voltaic_cell()
...
     |
llama-3.1-8b (explain 3-4 ayat)
     |
Save Firebase → Return answer
```

---

## 🗓️ VERSION HISTORY

| Version | Date | Changes |
|---|---|---|
| v1.0.0 | May 2026 | Initial deploy |
| v2.0.0 | May 2026 | fastembed (no torch) |
| v2.1.0 | May 2026 | Short explanations, bilingual |
| v3.0.0 | 09 May 2026 | Smart memory — shared cache + personal |
| v3.1.0 | 10 May 2026 | Major extractor fixes + LLM discipline |
| v3.2.0 | 10 May 2026 | Photo/Vision + 3-model + 3 new solvers |
| v3.3.0 | 10 May 2026 | 6 bug fixes based on stress test results |
| v3.4.0 | 10 May 2026 | Universal solver (Johor+Terengganu patterns) |

---

## 🐛 BUG FIXES LOG (v3.3.0)

| # | Bug | Fail | Fix |
|---|---|---|---|
| 1 | Concentration crash | solver_engine.py | Try/except + fix routing |
| 2 | Thermochem J→kJ | solver_engine.py | ÷1000 betul |
| 3 | Stoich crash bila mol | solver_engine.py | Accept given_mol param |
| 4 | Kepekatan molar missing | solver_engine.py | Kira mol/dm³ sekali |
| 5 | Titration nisbah salah | solver_engine.py | Formula (M1V1)/c1=(M2V2)/c2 |
| 6 | MOLFROMDH incomplete | solver_engine.py | Tambah langkah kira M |
| 7 | "Berapa" = formula | extractor.py | BM stopwords |
| 8 | Cas ion salah | extractor.py | extract_ion_charge() fix |
| 9 | Thermochem EN route salah | router.py | Tambah EN keywords |
| 10 | Mass from molarity route salah | router.py | Tambah "buat larutan" keyword |
| 11 | OOS format kimia | main.py | Fix rejection message |

---

## 🧪 STRESS TEST RESULTS (10 May 2026)

| Sesi | Soalan | Lulus | Pass Rate |
|---|---|---|---|
| Sesi 1 (teks) | 44 | 28 | 63% |
| Sesi 2 (vision) | 10 | 6 | 60% |
| Sesi 3 (teks) | 12 | 8 | 67% |
| **Jumlah** | **66** | **42** | **63.6%** |

**Jangkaan selepas v3.4.0:** ~90% pass rate

---

## 🚧 PENDING TASKS

### KRITIKAL — Perlu push segera
1. Merge `solver_engine_fixed.py` → `solver/solver_engine.py`
2. Merge `universal_spm_solver.py` → `solver/solver_engine.py`
3. Update `api/vision.py` → ganti VISION_SYSTEM_PROMPT
4. Update `solver/extractor.py` → tambah extract_from_vision_output()
5. Update `solver/router.py` → tambah SOLVER_TYPE_MAP

### MEDIUM — Selepas push
6. Analisis kertas negeri lain (Kedah, Pahang, Perak, Kelantan)
7. Tambah Ar override dari soalan (Cu=64 dalam soalan vs Cu=63.5 standard)
8. Selangor 2024 — OCR untuk scanned PDF
9. Tambah more past year ke QA index

### LOW
10. Admin dashboard
11. Rate limiting middleware
12. CI/CD auto-rebuild FAISS

---

## 🔨 USEFUL COMMANDS

```bash
# Push
git add . && git commit -m "msg" && git push origin main

# Health check
curl https://cikgiaikimia.onrender.com/api/health

# Test API
curl -X POST https://cikgiaikimia.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Hitung mol 4g NaOH", "session_id": "test123"}'

# Clear Firebase cache (WAJIB selepas deploy baru)
# Firebase Console -> Firestore -> qa_cache -> Delete collection
```

---

## 💬 HOW TO CONTINUE IN NEW CHAT

```
I am continuing "Cikgu AI Kimia" SPM Chemistry AI Tutor.
Attached README.md has full project status.

GitHub: https://github.com/msallehroslan/CikgiAIKimia
Live: https://cikgiaikimia.onrender.com
Bot: @TicerHawaAIBot
Firebase: cikgu-kimia-66b7b
Current version: v3.4.0 (pending push)

FILES TO UPLOAD TOGETHER:
1. README.md (this file)
2. solver_engine_v340.py (gabungan semua solver fix)
3. vision_extractor_v340.py (vision prompt + extractor)

Task for this session:
[describe your task here]
```

---

*Last updated: 10 May 2026*
*Current deployed: v3.2.0*
*Pending push: v3.4.0 (solver fixes + universal solver + vision update)*
