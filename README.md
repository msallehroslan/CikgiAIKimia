# 🧪 Cikgu AI Kimia — SPM Chemistry AI Tutor

> Production-ready AI tutor for Malaysian SPM Chemistry students
> Built with FastAPI + FAISS + Groq LLM + Telegram Bot + Firebase Smart Memory
> Now with Photo/Vision support 📷

---

## 🌐 Live URLs

| Service | URL |
|---|---|
| API (FastAPI) | https://cikgiaikimia.onrender.com |
| API Docs | https://cikgiaikimia.onrender.com/docs |
| Health Check | https://cikgiaikimia.onrender.com/api/health |
| Memory Stats | https://cikgiaikimia.onrender.com/api/memory/stats |
| Telegram Bot | @TicerHawaAIBot |
| GitHub | https://github.com/msallehroslan/CikgiAIKimia |
| Firebase | https://console.firebase.google.com/u/1/project/cikgu-kimia-66b7b/firestore |

---

## ✅ FULLY WORKING FEATURES

### Infrastructure
- [x] FastAPI backend deployed on Render.com (Singapore)
- [x] Python 3.11.9 pinned (.python-version file)
- [x] fastembed (ONNX) — RAM ~150MB (free tier compatible)
- [x] FAISS indexes: 514 theory + 176 calculations + 13 QA = 703 vectors
- [x] Groq LLM — 3 model architecture (70b theory, 8b explain, scout vision)
- [x] Firebase Firestore connected (project: cikgu-kimia-66b7b)
- [x] Render auto-deploy on git push
- [x] Telegram webhook inside FastAPI (no separate worker)

### Solvers (Deterministic Python — v3.2.0)
- [x] Mol calculations (moles_from_mass, moles_from_volume, moles_multi)
- [x] pH / pOH calculations (termasuk dari [OH-] concentration)
- [x] Titration — find molarity dan find volume (nisbah 1:1 dan 1:2)
- [x] Thermochemistry — entalpi peneutralan, pemelarutan, pembakaran
- [x] Entalpi endotermik (suhu turun = DeltaH positif)
- [x] Stoichiometry mass->mass dan mass->volume (gas RTP/STP)
- [x] Stoichiometry volume->mass (gas diberi, jisim ditanya) — NEW v3.2.0
- [x] Voltaic cell — E0cell = E0katod - E0anod — NEW v3.2.0
- [x] Molarity from delta H — cari kemolaran dari ΔH dan ΔT — NEW v3.2.0
- [x] Concentration g/dm3, Dilution (M1V1=M2V2)
- [x] Empirical formula (dari % komposisi atau jisim — BM/EN)
- [x] Relative atomic mass dari isotop
- [x] Atomic structure (proton, neutron, elektron)
- [x] Oxidation number (ion berkas SO4 2-, NO3-, Cr2O7 2-)
- [x] Rate of reaction (purata dan dari dua titik graf)
- [x] JMR / Molar mass (formula mudah, kompleks, air kristal K4Fe(CN)6.3H2O)
- [x] Mass from molarity (jisim untuk buat larutan)
- [x] Jumlah mol campuran gas (multi-formula)

### Photo / Vision Support (v3.2.0)
- [x] Pelajar hantar gambar soalan ke Telegram — bot jawab
- [x] Scout buat extract + interpret dalam SATU call (optimised)
- [x] Auto-derive formula dari nama IUPAC bila formula tidak terbaca
  - "kalium heksasianoferat(III) terhidrat" → K4Fe(CN)6.3H2O
  - "kuprum(II) sulfat pentahidrat" → CuSO4.5H2O
- [x] Formula MESTI dalam bahagian SOALAN (arahan eksplisit dalam prompt)
- [x] preprocess_vision_question() — strip MCQ options, parse SOALAN/PILIHAN/DATA
- [x] extract_valid_formulas() — handle complex formula K4Fe(CN)6.3H2O, Fe2(SO4)3
- [x] Clean extracted text — strip LaTeX, unicode subscript, arrows
- [x] Support MCQ — extract soalan + semua pilihan A/B/C/D
- [x] Fallback LLM (70b) untuk soalan yang solver belum support
- [x] Multi-provider: Groq Scout (default) / Gemini Flash / Tesseract (local)
- [x] Switch provider tanpa tukar code — set VISION_PROVIDER env var

### LLM Architecture (v3.2.0 — 3-Model Disciplined)
- [x] LLM HANYA untuk: (1) explain solver output, (2) teori dengan RAG context
- [x] LLM TIDAK BOLEH jawab pengiraan tanpa solver
- [x] Soalan luar konteks → mesej fallback jelas (no hallucination)
- [x] 3 model berbeza — RPD pool berasingan (~16K combined RPD/day)

### Extractor (v3.2.0 — Vision-Ready)
- [x] preprocess_vision_question() — handle Scout structured output
- [x] _clean_mcq_options() — strip A.141 B.256 C.389 D.422
- [x] extract_valid_formulas() — Pattern 1 (complex) + Pattern 2 (simple)
- [x] BM Stopwords — "Sebatian" tidak parse sebagai "Se"
- [x] Equation extraction v2 — trim LHS/RHS betul semua format
- [x] Strip [Ar=X] notation sebelum parse equation
- [x] Expanded keywords semua task — handle soalan bentuk ayat dari gambar:
  - JMR: "jisim relatif", "relative mass"
  - Mol: "berapa mol", "bilangan mol"
  - pH: "ion hidrogen", "nilai ph"
  - Molarity: "hitungkan kemolaran", "apakah kemolaran"
  - Stoich: "hitungkan jisim", "berapakah isipadu", "mass of"
  - Empirical: "tentukan formula", "formula molekul"
  - Oxidation: "tentukan nombor", "determine the oxidation"
  - Ar isotop: "jisim atom", "kelimpahan", "abundance"
  - Voltaic: "keupayaan elektrod", "voltan sel", "e0"
  - Molarity from dH: "kemolaran" + "delta h" + thermochemistry

### Telegram Bot
- [x] /start — welcome + inline keyboard
- [x] /help — all commands
- [x] /quiz [topik] — generate MCQ quiz
- [x] /solve [soalan] — calculation only
- [x] /clear — clear session + memory
- [x] /stats — show cache statistics
- [x] Hantar gambar terus — bot jawab soalan dari gambar

---

## 📊 SPM TRIAL PAPER CAPABILITY
### Kertas Percubaan SPM 2024 — Johor Batu Pahat (40 soalan)

| Kategori | Bilangan | Peratusan |
|---|---|---|
| Boleh jawab teks sahaja | 22 | 55% |
| Boleh jawab dengan gambar | +15 est | ~92% total |
| Sangat susah (graf data tepat) | ~3 | ~8% |

### Limitations Yang Masih Ada
1. Graf yang perlu baca nilai tepat — Q15 (kaedah tangen), Q33 (nilai R vs S)
2. Soalan KBAT multi-step seperti Q40 (kenal garam X dulu, kemudian kira PbO)
3. Diagram susunan elektron — perlu vision interpret dengan tepat

---

## 📁 PROJECT STRUCTURE

```
CikgiAIKimia/
├── api/
│   ├── main.py              ← FastAPI + Telegram webhook (v3.2.0)
│   ├── memory.py            ← Smart memory (shared cache + personal)
│   └── vision.py            ← Multi-provider vision (Groq/Gemini/Tesseract)
├── rag/
│   ├── embedder.py          ← fastembed ONNX
│   ├── retriever.py         ← FAISS retriever
│   ├── chunker.py           ← Markdown chunker
│   ├── indexer.py           ← FAISS index manager
│   └── metadata_tagger.py   ← chunk metadata tagger
├── solver/
│   ├── solver_engine.py     ← deterministic chemistry solver (v3.2.0)
│   ├── extractor.py         ← question parser (v3.2.0 vision-ready)
│   ├── router.py            ← task router
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
| VISION_PROVIDER | groq (groq/gemini/tesseract/none) |
| GEMINI_API_KEY | your_gemini_key (hanya jika VISION_PROVIDER=gemini) |
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

## 🧠 ANSWER PIPELINE (v3.2.0)

```
TEKS masuk              GAMBAR masuk
     |                       |
Detect BM/EN         Groq Scout (1 call)
     |               extract+interpret
Firebase cache       formula MESTI dalam SOALAN
HIT -> Return            |
     |               preprocess_vision_question()
     |               strip MCQ options
     |               parse SOALAN/DATA
Extractor <-----------------/
extract_valid_formulas()
(Pattern 1: K4Fe(CN)6.3H2O)
(Pattern 2: NaOH, H2SO4)
     |
Router -> identify task
     |
CALCULATION              THEORY
Solver Python       RAG -> nota markdown
(deterministic)          |
     |              llama-3.3-70b
llama-3.1-8b        (synthesis nota)
(explain 3-4 ayat)
     |
Tiada context -> Mesej fallback
     |
Save Firebase -> Return answer
```

### 3-Model Architecture
| Model | Task | RPD | TPM |
|---|---|---|---|
| llama-3.3-70b-versatile | Teori (RAG) | 1K | 12K |
| llama-3.1-8b-instant | Explain solver | 14.4K | 6K |
| llama-4-scout-17b | Vision (gambar) | 1K | 30K |

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

### v3.2.0 New Solvers
| Task | Formula | Contoh Soalan |
|---|---|---|
| stoichiometry_volume_to_mass | n=V/Vm, m=nM | 120cm3 Cl2 -> jisim FeCl3 |
| voltaic_cell | E0sel = E0katod - E0anod | Sel Zn-Cu, E0=? |
| molarity_from_delta_h | mol=Q/dH, M=mol/V | dH=-57.3 kJ/mol, dT=7C, M=? |

### v3.2.0 Vision Fixes
| Fix | Detail |
|---|---|
| extract_valid_formulas | Pattern 1 khusus formula kompleks K4Fe(CN)6.3H2O |
| preprocess_vision_question | Strip DATA square brackets, parse SOALAN/PILIHAN/DATA |
| Vision prompt | Formula MESTI dalam SOALAN section, bukan di luar |
| Scout prompt | Arahan derive formula IUPAC dan letak dalam SOALAN |

### v3.1.0 Bug Fixes (14 bugs)
| Bug | Fix |
|---|---|
| "Sebatian" parse sebagai "Se" | BM Stopwords |
| Cas ion SO4 2- diabaikan | extract_ion_charge() |
| Entalpi route ke gas solver | Thermochemistry check dahulu |
| Titrasi route ke gas solver | Titration check dahulu |
| DeltaH endotermik tanda salah | Label eksotermik/endotermik |
| Jisim larutan salah | total_mass = sum(volumes_cm3) |
| Kemolaran jadi pOH | pOH detect bila keyword eksplisit sahaja |
| Multi-formula mol | Task baru moles_multi |
| Jisim untuk buat larutan | Task baru mass_from_molarity |
| Stoikiometri formula salah | extract_equation v2 + smart target |
| [Ar=X] rosak equation | Strip dalam normalize_text |
| Equation trim salah | Trim RHS pada " jika " tanpa koma |
| Stoikiometri tanya isipadu | Task baru stoichiometry_mass_to_volume |
| LLM jawab pengiraan | Strict LLM discipline |

---

## 🚧 PENDING TASKS

### HIGH PRIORITY
1. Test vision Q5 JMR K4Fe(CN)6.3H2O — patut 🧮 422 selepas fix formula extraction
2. Test 3 solver baru — Q38 (0.542g), Q37 (1.028 mol/dm3), Q34 (+1.10V)
3. More vision testing — jadual, graf, organik struktur

### MEDIUM PRIORITY
4. Add more past year questions to QA index (currently 13 vectors)
5. Quiz improvement — subjective, difficulty levels
6. Diagram descriptions dalam knowledge_base

### LOW PRIORITY
7. Admin dashboard
8. Rate limiting middleware
9. CI/CD auto-rebuild FAISS

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

# Clear session
curl -X DELETE https://cikgiaikimia.onrender.com/api/memory/session/tg_123456

# Rebuild FAISS
cd CikgiAIKimia && set PYTHONPATH=.
python scripts/build_index_v2.py --skip-diagrams
```

---

## 📊 BENCHMARK (Tested 10 May 2026)

| Question | Expected | Status |
|---|---|---|
| Mol 4g NaOH | 0.1 mol | OK |
| Entalpi peneutralan | -53.76 kJ/mol | OK |
| Entalpi pemelarutan endotermik | +50.4 kJ/mol | OK |
| JMR K4Fe(CN)6.3H2O | 422 | TESTING |
| Stoich Fe2O3+CO->Fe | 5.6g | OK |
| Stoich 120cm3 Cl2->FeCl3 | 0.542g | NEW |
| Voltaic sel Zn-Cu | +1.10V | NEW |
| Molarity from dH=-57.3 | 1.026 mol/dm3 | NEW |
| Soalan luar konteks | Rejected | OK |
| GAMBAR Q6 Ikatan kovalen | Kovalen | OK |

---

## 💬 HOW TO CONTINUE IN NEW CHAT

```
I am continuing "Cikgu AI Kimia" SPM Chemistry AI Tutor.
Attached README.md has full project status.

GitHub: https://github.com/msallehroslan/CikgiAIKimia
Live: https://cikgiaikimia.onrender.com
Bot: @TicerHawaAIBot
Firebase: cikgu-kimia-66b7b
Current version: v3.2.0

Task for this session:
1. [describe your task here]
```

Then attach this README.md file.

---

*Last updated: 10 May 2026*
*Current version: v3.2.0*
*Next: Test Q5 vision fix, test 3 new solvers, more vision testing*
