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

### Solvers (Deterministic Python — v3.1.0)
- [x] Mol calculations (moles_from_mass, moles_from_volume, moles_multi)
- [x] pH / pOH calculations (termasuk dari [OH⁻] concentration)
- [x] Titration — find molarity dan find volume (nisbah 1:1 dan 1:2)
- [x] Thermochemistry — entalpi peneutralan, pemelarutan, pembakaran
- [x] Entalpi endotermik (suhu turun → ΔH positif) ✅
- [x] Stoichiometry mass→mass dan mass→volume (gas RTP/STP)
- [x] Concentration g/dm³, Dilution (M₁V₁=M₂V₂)
- [x] Empirical formula (dari % komposisi atau jisim — BM/EN)
- [x] Relative atomic mass dari isotop
- [x] Atomic structure (proton, neutron, elektron)
- [x] Oxidation number (termasuk ion berkas seperti SO₄²⁻, NO₃⁻, Cr₂O₇²⁻)
- [x] Rate of reaction (purata dan dari dua titik graf)
- [x] JMR / Molar mass (formula mudah dan kompleks)
- [x] Mass from molarity (jisim yang diperlukan untuk buat larutan)
- [x] Jumlah mol campuran gas (multi-formula)

### LLM Architecture (v3.2.0 — 3-Model Disciplined)
- [x] LLM HANYA untuk: (1) explain langkah pengiraan solver, (2) teori dengan RAG context
- [x] LLM TIDAK BOLEH jawab pengiraan tanpa solver
- [x] LLM TIDAK BOLEH jawab teori tanpa RAG context dari nota
- [x] Jika RAG tiada context → mesej fallback jelas (bukan hallucination)
- [x] Soalan luar konteks kimia → ditolak dengan mesej betul
- [x] 3 model berbeza untuk 3 task berbeza (RPD pool berasingan)

### Photo / Vision Support (v3.2.0) 📷
- [x] Pelajar boleh hantar gambar soalan terus ke Telegram bot
- [x] Multi-provider: Groq Vision (default) / Gemini Flash / Tesseract (local)
- [x] Auto-detect image format (JPEG, PNG, WebP, GIF)
- [x] Show extracted text preview sebelum jawab
- [x] Fallback mesej jelas jika gambar tidak dapat dibaca
- [x] Switch provider tanpa tukar code — set env var VISION_PROVIDER
- [x] Flow: Gambar → Vision AI → extract teks → Solver/RAG → Jawapan SPM

### Telegram Bot Commands
- [x] /start — welcome + inline keyboard
- [x] /help — all commands
- [x] /quiz [topik] — generate MCQ quiz
- [x] /solve [soalan] — calculation only
- [x] /clear — clear session + memory
- [x] /stats — show cache statistics

### Smart Memory System v3.0.0 ✅
- [x] LAYER 1: Shared Q&A cache (all students benefit)
  - First question: ~1167ms (fresh solve)
  - Same question again: ⚡ ~400-550ms (from cache)
- [x] LAYER 2: Personal memory per student
  - "soalan saya tadi, terangkan lagi" → works ✅
  - Bot remembers last 4 messages per session
- [x] Auto language detection BM/EN
- [x] Firestore collections: sessions/ + qa_cache/

### Extractor Improvements (v3.1.0)
- [x] BM Stopwords — "Sebatian" tidak lagi parse sebagai "Se" (Selenium)
- [x] Ion charge parsing — "SO₄ 2-" detect charge=−2 betul
- [x] Equation extraction v2 — trim LHS/RHS betul untuk semua format
- [x] Strip [Ar=X] notation sebelum parse equation
- [x] Smart target formula detection dalam stoichiometry
- [x] Thermochemistry priority check SEBELUM mol/volume chain
- [x] Titration priority check SEBELUM mol/volume chain
- [x] pOH hanya detect bila keyword "poh" eksplisit ada
- [x] Kemolaran tidak tersalah jadi pOH

---

## 📊 SPM TRIAL PAPER CAPABILITY ANALYSIS
### Kertas Percubaan SPM 2024 — Johor Batu Pahat (40 soalan)

| Kategori | Bilangan | Peratusan |
|---|---|---|
| ✅ Bot boleh jawab | 22 | 55% |
| ⚠️ Boleh sebahagian | 8 | 20% |
| ❌ Tidak boleh | 10 | 25% |

### ✅ Boleh dijawab
- Semua soalan pengiraan numerik (mol, pH, kemolaran, entalpi, stoikiometri)
- Soalan teori yang ada dalam nota RAG (ikatan kimia, asid-bes, redoks, kadar tindak balas)
- Soalan fakta SPM standard (isotop radioaktif, pH jus gastrik, dll)

### ❌ Tidak boleh dijawab — Limitations

**1. Soalan bergantung pada GAMBAR/RAJAH (kritikal)**
- ~10 soalan setiap kertas SPM melibatkan gambar rajah, graf, formula struktur bergambar
- Bot tidak boleh "lihat" gambar yang dicetak dalam kertas soalan
- Contoh: Q2 (daya tarikan zarah), Q8 (ikatan hidrogen), Q10 (susunan elektron), Q15 (kaedah kadar dari graf), Q33 (interpret graf R vs S)
- **FIX DALAM PEMBANGUNAN: Groq Vision API untuk soalan bergambar**

**2. Stoikiometri GAS → JISIM (terbalik)**
- Sekarang ada: jisim→jisim, jisim→gas
- Belum ada: gas→jisim (Q38: 120cm³ Cl₂ → jisim FeCl₃)
- Belum ada: gas→gas

**3. Sel Elektrokimia**
- Belum ada solver untuk E⁰cell = E⁰katod − E⁰anod
- Belum ada reverse-calculate molarity dari ΔH (Q37)

**4. Nama IUPAC organik**
- Bot boleh terangkan konsep tapi tidak auto-generate nama IUPAC dari formula struktur bergambar
- Q21 (2-metilbutana), Q27 (2-metilbut-1,3-diena)

**5. Soalan KBAT multi-langkah padu**
- Q40: kenal pasti garam X → kira mol → kira pepejal Z (3 langkah berlainan)
- Bot boleh buat setiap langkah berasingan tapi tidak secara automatik end-to-end

---

## 📁 PROJECT STRUCTURE

```
CikgiAIKimia/
├── api/
│   ├── main.py              ← FastAPI + Telegram webhook (v3.2.0)
│   ├── memory.py            ← Smart memory (shared cache + personal)
│   └── vision.py            ← Multi-provider vision (Groq/Gemini/Tesseract)
├── rag/
│   ├── embedder.py          ← fastembed ONNX (replaces torch)
│   ├── retriever.py         ← FAISS retriever + query augmentation
│   ├── chunker.py           ← Markdown chunker
│   ├── indexer.py           ← FAISS index manager
│   └── metadata_tagger.py   ← chunk metadata tagger
├── solver/
│   ├── solver_engine.py     ← deterministic chemistry solver (v3.2.0)
│   ├── extractor.py         ← question parser (v3.2.0 — major fixes)
│   ├── router.py            ← task router (v3.1.0)
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
├── requirements.txt         ← pinned deps (fastembed, no torch)
├── render.yaml              ← Render deployment config
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
| GEMINI_API_KEY | your_gemini_key (only if VISION_PROVIDER=gemini) |
| TELEGRAM_BOT_TOKEN | your_bot_token |
| API_BASE_URL | https://cikgiaikimia.onrender.com |
| FAISS_INDEX_DIR | ./faiss_indexes |
| KB_DIR | ./knowledge_base |
| RETRIEVAL_THRESHOLD | 0.30 |
| MAX_CONTEXT_CHARS | 3000 |
| GOOGLE_APPLICATION_CREDENTIALS_JSON | {entire Firebase JSON} |

---

## 📦 KEY PACKAGES (requirements.txt)

```txt
fastembed==0.3.6              # ONNX embeddings (replaces torch)
faiss-cpu==1.8.0              # vector store
langchain==0.2.16             # RAG framework
fastapi==0.111.0              # web framework
groq==0.9.0                   # LLM API
python-telegram-bot==21.3     # Telegram bot
firebase-admin==6.5.0         # Firestore smart memory
Pillow==10.4.0                # image processing
numpy==1.26.4                 # numerics
```

---

## 🧠 ANSWER PIPELINE (v3.1.0)

```
Soalan TEKS masuk          Gambar/FOTO masuk
      ↓                           ↓
1. Detect language         vision.py
      ↓                    (Groq/Gemini/Tesseract)
2. Firebase cache               ↓
   HIT → Return ⚡         Extract teks soalan
      ↓                           ↓
3. Extractor.py ←──────────────────┘
      ↓
4. Router.py → identify task
      ↓
┌──────────────────────┬─────────────────────┐
│ CALCULATION          │ THEORY              │
│ Solver Python        │ RAG → nota markdown │
│ (deterministic)      │       ↓             │
│       ↓              │ llama-3.3-70b ←─────┤ Teori
│ llama-3.1-8b ←───────┤ (synthesis nota)    │
│ (explain 3-4 ayat)   │                     │
└──────────────────────┴─────────────────────┘
      ↓
5. Tiada context → Mesej fallback (no hallucination)
      ↓
6. Save to Firebase (cache + personal memory)
      ↓
7. Return answer
```

---

## 🗓️ VERSION HISTORY

| Version | Date | Changes |
|---|---|---|
| v1.0.0 | May 2026 | Initial deploy — FastAPI + FAISS + Groq |
| v2.0.0 | May 2026 | fastembed (no torch), free tier compatible |
| v2.1.0 | May 2026 | Short explanations, correct RAG sources, bilingual |
| v3.0.0 | 09 May 2026 | Smart memory — shared cache + personal history |
| v3.1.0 | 10 May 2026 | Major extractor fixes + LLM discipline + new solvers |
| v3.2.0 | 10 May 2026 | Photo/Vision support + 3-model architecture |

### v3.2.0 New Features
| Feature | Detail |
|---|---|
| Photo support | Pelajar hantar gambar soalan → bot jawab |
| Vision provider | Groq Llama 4 Scout (default, free, dalam account) |
| Multi-provider | Switch Groq/Gemini/Tesseract via env var |
| 3-model architecture | 70b theory, 8b explain, scout vision |
| RPD combined | 1K + 14.4K + 1K = ~16K requests/day |

### 3-Model Architecture
| Model | Task | RPD | TPM | Kenapa |
|---|---|---|---|---|
| llama-3.3-70b-versatile | Teori (RAG) | 1K | 12K | Better reasoning untuk synthesis nota |
| llama-3.1-8b-instant | Explain solver | 14.4K | 6K | Fast, cheap, cukup untuk 3-4 ayat |
| llama-4-scout-17b | Vision (gambar) | 1K | 30K | Multimodal, support image input |

### v3.1.0 Bug Fixes Detail
| Bug | Fix |
|---|---|
| "Sebatian" parse sebagai "Se" (Selenium) | BM Stopwords list dalam extractor |
| Cas ion SO₄²⁻ diabaikan → oxidation number salah | extract_ion_charge() hanya match selepas whitespace |
| Entalpi route ke gas solver | Thermochemistry check SEBELUM mol/volume chain |
| Titrasi route ke gas solver | Titration check SEBELUM mol/volume chain |
| ΔH endotermik tanda salah | Label eksotermik/endotermik + tanda betul |
| Jisim larutan salah (ambil jisim terlarut) | total_mass = sum(volumes_cm3) |
| Kemolaran 0.4 jadi pOH | pOH hanya detect bila "poh" keyword eksplisit |
| Multi-formula mol abaikan komponen kedua | Task baru moles_multi |
| "Jisim untuk buat larutan" route ke gas solver | Task baru mass_from_molarity |
| Stoikiometri formula salah (CO₂ bukan CaCO₃) | extract_equation v2 + smart target detection |
| [Ar=X] dalam soalan rosak equation parse | Strip [Ar=X] dalam normalize_text |
| Equation trim terlalu awal/lewat | Trim RHS pada " jika ", " apabila " tanpa koma |
| Stoikiometri tanya isipadu tapi jawab jisim | Task baru stoichiometry_mass_to_volume |
| LLM jawab pengiraan tanpa solver | main.py v3.1.0 — strict LLM discipline |

---

## 🚧 PENDING TASKS

### HIGH PRIORITY
1. **~~Photo/Screenshot support~~ ✅ SELESAI v3.2.0**
   - vision.py — multi-provider (Groq/Gemini/Tesseract)
   - Groq Llama 4 Scout — default provider
   - Set VISION_PROVIDER env var untuk switch

2. **Stoikiometri GAS → JISIM**
   - Tambah task `stoichiometry_volume_to_mass` dalam solver_engine.py
   - Contoh: "120cm³ Cl₂ + Fe → FeCl₃, hitungkan jisim FeCl₃"
   - Extractor perlu detect "isipadu gas diberi" + "jisim ditanya"

3. **Sel Elektrokimia solver**
   - Tambah task `voltaic_cell` dalam solver_engine.py
   - Formula: E⁰cell = E⁰katod − E⁰anod
   - Extractor detect "keupayaan elektrod" atau "voltan sel"

### MEDIUM PRIORITY
4. **Add more past year questions to QA index**
   - Currently only 13 QA vectors
   - Add SPM past year Q&A to knowledge_base/questions/past_years/
   - Rebuild: `python scripts/build_index_v2.py --skip-diagrams`

5. **Quiz improvement**
   - Add subjective question type (not just MCQ)
   - Add difficulty levels (mudah/sederhana/susah)
   - Chapter-specific: /quiz bab3

6. **Diagram support dalam knowledge_base**
   - Currently --skip-diagrams during build
   - Add diagram descriptions to knowledge_base/
   - Rebuild with diagram injection enabled

7. **IUPAC naming solver**
   - Bot boleh terangkan konsep tapi tidak auto-generate nama
   - Perlu parser untuk formula struktur → nama IUPAC

### LOW PRIORITY
8. **Admin dashboard**
   - Simple web UI untuk monitor cache hits
   - View top questions, manage sessions

9. **Rate limiting at API level**
   - Currently only in bot handler
   - Add FastAPI middleware rate limiting

10. **CI/CD pipeline**
    - Auto rebuild FAISS when knowledge_base/ changes
    - GitHub Actions workflow

---

## 🔨 USEFUL COMMANDS

```bash
# Rebuild FAISS indexes locally
cd CikgiAIKimia
set PYTHONPATH=.
python scripts/build_index_v2.py --skip-diagrams --skip-questions

# Full rebuild with QA
python scripts/build_index_v2.py --skip-diagrams

# Push to GitHub
git add .
git commit -m "your message"
git push origin main

# Check health
curl https://cikgiaikimia.onrender.com/api/health

# Check memory/cache stats
curl https://cikgiaikimia.onrender.com/api/memory/stats

# Test question via API
curl -X POST https://cikgiaikimia.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Hitung mol 4g NaOH", "session_id": "test123"}'

# Clear a session
curl -X DELETE https://cikgiaikimia.onrender.com/api/memory/session/tg_123456

# Delete Telegram webhook
curl https://api.telegram.org/bot<TOKEN>/deleteWebhook

# Check webhook status
curl https://api.telegram.org/bot<TOKEN>/getWebhookInfo
```

---

## 📊 PERFORMANCE BENCHMARK (Tested 10 May 2026)

| Question | Answer | Correct | Fresh | Cached |
|---|---|---|---|---|
| Mol 4g NaOH | 0.1 mol | ✅ | ~1200ms | ~400ms |
| Entalpi peneutralan HCl+NaOH | −53.76 kJ/mol | ✅ | ~1500ms | ~400ms |
| Entalpi pemelarutan NaOH endotermik | +50.4 kJ/mol | ✅ | ~1400ms | ~400ms |
| Formula empirik "Sebatian X" 52.2%C | C₂H₆O | ✅ | ~1800ms | ~400ms |
| Nombor pengoksidaan SO₄²⁻ | S=+6 | ✅ | ~1600ms | ~400ms |
| Jumlah mol N₂+O₂ campuran | 0.3 mol | ✅ | ~1400ms | ~400ms |
| Jisim NaOH untuk buat larutan | 10g | ✅ | ~1600ms | ~400ms |
| Stoikiometri CaCO₃→CO₂ pada RTP | 1.2 dm³ | ✅ | ~900ms | ~400ms |
| Stoikiometri Fe₂O₃+CO→Fe | 5.6g | ✅ | ~1400ms | ~400ms |
| Soalan luar konteks (PM Malaysia) | Rejected ✅ | ✅ | ~800ms | N/A |
| **TOTAL** | **10/10** | **100%** | avg 1.4s | avg 0.4s |

---

## 💬 HOW TO CONTINUE IN NEW CHAT

Paste this at the start of new chat:

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
*Next priority: stoichiometry_volume_to_mass + voltaic cell solver + more past year questions*
