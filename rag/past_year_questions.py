"""
past_year_questions.py — Cikgu AI Kimia
========================================
Complete SPM Chemistry past year question bank.

This file serves TWO purposes:
  1. A ready-to-use question bank for the `index_qa` FAISS index
  2. A template showing exactly how to format your own past year questions

Structure of each question:
  - soalan: question text (BM)
  - jawapan: model answer / marking scheme
  - chapter, tingkatan, exam_year
  - question_type: mcq | struktur | esei
  - marks: mark allocation
  - topic, subtopic
  - diagrams: list of diagram references (if any)
  - keywords_bm, keywords_en

HOW TO ADD YOUR OWN PAST YEAR QUESTIONS:
  1. Copy a question block below
  2. Fill in the fields from your SPM paper
  3. Run: python scripts/build_index.py --kb-dir knowledge_base
  The new question will be indexed automatically.

Author: Cikgu AI Kimia Project
"""

from __future__ import annotations
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# QUESTION BANK
# ---------------------------------------------------------------------------

SPM_QUESTIONS: List[Dict[str, Any]] = [

    # ════════════════════════════════════════════════════════════════════════
    # BAB 3 FORM 4 — KONSEP MOL
    # ════════════════════════════════════════════════════════════════════════
    {
        "soalan": (
            "Hitungkan bilangan mol bagi 11.2 g besi, Fe. "
            "[Ar: Fe = 56]"
        ),
        "jawapan": (
            "Diberi:\n"
            "Jisim besi, m = 11.2 g\n"
            "Jisim atom relatif Fe = 56\n\n"
            "Formula:\n"
            "n = m ÷ Ar\n\n"
            "Pengiraan:\n"
            "n = 11.2 ÷ 56\n"
            "n = 0.2 mol\n\n"
            "Jawapan:\n"
            "Bilangan mol besi = 0.2 mol"
        ),
        "chapter": 3, "tingkatan": 4,
        "exam_year": 2022, "question_type": "struktur", "marks": 2,
        "topic": "Konsep Mol", "subtopic": "Mol dan Jisim",
        "keywords_bm": ["mol", "jisim", "besi", "jisim atom relatif"],
        "keywords_en": ["mole", "mass", "iron", "relative atomic mass"],
        "diagrams": [],
    },
    {
        "soalan": (
            "Berapakah bilangan atom dalam 0.5 mol kuprum, Cu? "
            "[NA = 6.02 × 10²³ mol⁻¹]"
        ),
        "jawapan": (
            "Diberi:\n"
            "Bilangan mol Cu = 0.5 mol\n"
            "NA = 6.02 × 10²³ mol⁻¹\n\n"
            "Formula:\n"
            "Bilangan atom = n × NA\n\n"
            "Pengiraan:\n"
            "Bilangan atom = 0.5 × 6.02 × 10²³\n"
            "Bilangan atom = 3.01 × 10²³ atom\n\n"
            "Jawapan:\n"
            "Bilangan atom kuprum = 3.01 × 10²³ atom"
        ),
        "chapter": 3, "tingkatan": 4,
        "exam_year": 2021, "question_type": "struktur", "marks": 2,
        "topic": "Konsep Mol", "subtopic": "Mol dan Bilangan Zarah",
        "keywords_bm": ["bilangan atom", "mol", "nombor avogadro", "kuprum"],
        "keywords_en": ["number of atoms", "mole", "avogadro", "copper"],
        "diagrams": [],
    },
    {
        "soalan": (
            "Hitungkan isipadu gas hidrogen, H₂ pada keadaan bilik yang dihasilkan "
            "apabila 1.2 g magnesium bertindak balas dengan asid sulfurik berlebihan. "
            "[Ar: Mg = 24; Isipadu molar gas pada keadaan bilik = 24 dm³ mol⁻¹]\n"
            "Persamaan: Mg + H₂SO₄ → MgSO₄ + H₂"
        ),
        "jawapan": (
            "Diberi:\n"
            "Jisim Mg = 1.2 g\n"
            "Ar Mg = 24\n"
            "Vm = 24 dm³ mol⁻¹\n\n"
            "Langkah 1: Mol Mg\n"
            "n(Mg) = 1.2 ÷ 24 = 0.05 mol\n\n"
            "Langkah 2: Nisbah mol dari persamaan\n"
            "Mg : H₂ = 1 : 1\n"
            "n(H₂) = 0.05 mol\n\n"
            "Langkah 3: Isipadu H₂\n"
            "V = n × Vm = 0.05 × 24 = 1.2 dm³\n\n"
            "Jawapan:\n"
            "Isipadu gas H₂ = 1.2 dm³"
        ),
        "chapter": 3, "tingkatan": 4,
        "exam_year": 2023, "question_type": "struktur", "marks": 3,
        "topic": "Konsep Mol", "subtopic": "Stoikiometri",
        "keywords_bm": ["stoikiometri", "isipadu gas", "magnesium", "nisbah mol"],
        "keywords_en": ["stoichiometry", "volume gas", "magnesium", "mole ratio"],
        "diagrams": [],
    },

    # ════════════════════════════════════════════════════════════════════════
    # BAB 6 FORM 5 — ASID BES GARAM
    # ════════════════════════════════════════════════════════════════════════
    {
        "soalan": (
            "Hitungkan nilai pH asid hidroklorik, HCl yang mempunyai kepekatan "
            "0.01 mol dm⁻³. [HCl → H⁺ + Cl⁻]"
        ),
        "jawapan": (
            "Diberi:\n"
            "[HCl] = 0.01 mol dm⁻³\n"
            "HCl mengion lengkap → [H⁺] = 0.01 mol dm⁻³\n\n"
            "Formula:\n"
            "pH = −log[H⁺]\n\n"
            "Pengiraan:\n"
            "pH = −log(0.01)\n"
            "pH = −log(10⁻²)\n"
            "pH = 2\n\n"
            "Jawapan:\n"
            "pH = 2"
        ),
        "chapter": 6, "tingkatan": 5,
        "exam_year": 2022, "question_type": "struktur", "marks": 2,
        "topic": "Asid Bes Garam", "subtopic": "Pengiraan pH",
        "keywords_bm": ["pH", "asid hidroklorik", "kepekatan ion hidrogen"],
        "keywords_en": ["pH", "hydrochloric acid", "hydrogen ion concentration"],
        "diagrams": [],
    },
    {
        "soalan": (
            "25 cm³ larutan natrium hidroksida, NaOH 0.1 mol dm⁻³ dititratkan dengan "
            "asid hidroklorik, HCl 0.2 mol dm⁻³.\n"
            "Hitungkan isipadu HCl yang diperlukan untuk peneutralan lengkap.\n"
            "Persamaan: NaOH + HCl → NaCl + H₂O"
        ),
        "jawapan": (
            "Diberi:\n"
            "V(NaOH) = 25 cm³ = 0.025 dm³\n"
            "M(NaOH) = 0.1 mol dm⁻³\n"
            "M(HCl) = 0.2 mol dm⁻³\n\n"
            "Langkah 1: Mol NaOH\n"
            "n(NaOH) = M × V = 0.1 × 0.025 = 0.0025 mol\n\n"
            "Langkah 2: Nisbah mol\n"
            "NaOH : HCl = 1 : 1\n"
            "n(HCl) = 0.0025 mol\n\n"
            "Langkah 3: Isipadu HCl\n"
            "V = n ÷ M = 0.0025 ÷ 0.2 = 0.0125 dm³ = 12.5 cm³\n\n"
            "Jawapan:\n"
            "Isipadu HCl = 12.5 cm³"
        ),
        "chapter": 6, "tingkatan": 5,
        "exam_year": 2023, "question_type": "struktur", "marks": 3,
        "topic": "Asid Bes Garam", "subtopic": "Pentitratan",
        "keywords_bm": ["pentitratan", "peneutralan", "NaOH", "HCl", "nisbah mol"],
        "keywords_en": ["titration", "neutralisation", "sodium hydroxide", "mole ratio"],
        "diagrams": [],
    },

    # ════════════════════════════════════════════════════════════════════════
    # BAB 7 FORM 5 — KADAR TINDAK BALAS
    # ════════════════════════════════════════════════════════════════════════
    {
        "soalan": (
            "Rajah menunjukkan graf isipadu gas CO₂ melawan masa untuk tindak balas "
            "antara kalsium karbonat dengan asid hidroklorik berlebihan.\n\n"
            "Jadual data:\n"
            "Masa (min) | Isipadu CO₂ (cm³)\n"
            "0          | 0\n"
            "1          | 40\n"
            "2          | 65\n"
            "3          | 80\n"
            "4          | 88\n"
            "5          | 90\n"
            "6          | 90\n\n"
            "(a) Hitungkan kadar tindak balas purata bagi minit pertama.\n"
            "(b) Nyatakan masa tindak balas selesai.\n"
            "(c) Mengapakah kadar tindak balas berkurang dengan masa?"
        ),
        "jawapan": (
            "(a) Kadar tindak balas purata bagi minit pertama:\n"
            "Kadar = perubahan isipadu ÷ masa\n"
            "Kadar = (40 − 0) ÷ 1\n"
            "Kadar = 40 cm³ min⁻¹\n\n"
            "(b) Tindak balas selesai pada minit ke-5\n"
            "(isipadu gas tidak berubah selepas minit ke-5)\n\n"
            "(c) Kadar tindak balas berkurang dengan masa kerana:\n"
            "kepekatan bahan tindak balas (CaCO₃ dan HCl) berkurang.\n"
            "Frekuensi perlanggaran berkesan antara zarah berkurang.\n"
            "Oleh itu, kadar tindak balas menjadi lebih perlahan."
        ),
        "chapter": 7, "tingkatan": 5,
        "exam_year": 2023, "question_type": "struktur", "marks": 5,
        "topic": "Kadar Tindak Balas", "subtopic": "Pengiraan Kadar",
        "keywords_bm": ["kadar tindak balas", "graf isipadu gas", "perlanggaran", "kepekatan"],
        "keywords_en": ["rate of reaction", "gas volume graph", "collision frequency", "concentration"],
        "diagrams": ["images/graf_kadar_tindak_balas.png"],
    },
    {
        "soalan": (
            "Terangkan kesan suhu terhadap kadar tindak balas berdasarkan teori perlanggaran."
        ),
        "jawapan": (
            "Apabila suhu meningkat:\n\n"
            "1. Tenaga kinetik zarah meningkat\n"
            "2. Zarah bergerak lebih laju\n"
            "3. Frekuensi perlanggaran antara zarah meningkat\n"
            "4. Lebih banyak zarah mempunyai tenaga sama atau melebihi tenaga pengaktifan\n"
            "5. Bilangan perlanggaran berkesan meningkat\n"
            "6. Oleh itu, kadar tindak balas meningkat\n\n"
            "Peraturan umum: Setiap kenaikan suhu 10°C menggandakan kadar tindak balas."
        ),
        "chapter": 7, "tingkatan": 5,
        "exam_year": 2021, "question_type": "esei", "marks": 4,
        "topic": "Kadar Tindak Balas", "subtopic": "Teori Perlanggaran",
        "keywords_bm": ["suhu", "tenaga kinetik", "perlanggaran berkesan", "tenaga pengaktifan"],
        "keywords_en": ["temperature", "kinetic energy", "effective collision", "activation energy"],
        "diagrams": [],
    },

    # ════════════════════════════════════════════════════════════════════════
    # BAB 3 FORM 5 — TERMOKIMIA
    # ════════════════════════════════════════════════════════════════════════
    {
        "soalan": (
            "50 cm³ larutan asid hidroklorik, HCl 1.0 mol dm⁻³ dicampurkan dengan "
            "50 cm³ larutan natrium hidroksida, NaOH 1.0 mol dm⁻³. "
            "Suhu awal kedua-dua larutan ialah 28.0°C. "
            "Suhu akhir campuran ialah 34.5°C.\n\n"
            "(a) Hitungkan haba yang dibebaskan dalam tindak balas ini.\n"
            "(b) Hitungkan perubahan entalpi peneutralan, ΔH.\n"
            "[c = 4.2 J g⁻¹ °C⁻¹; 1 cm³ larutan = 1 g]"
        ),
        "jawapan": (
            "(a) Haba yang dibebaskan:\n"
            "Diberi:\n"
            "m = 100 g (jumlah isipadu = 50 + 50 = 100 cm³)\n"
            "c = 4.2 J g⁻¹ °C⁻¹\n"
            "ΔT = 34.5 − 28.0 = 6.5°C\n\n"
            "Formula: Q = mcΔT\n"
            "Q = 100 × 4.2 × 6.5\n"
            "Q = 2730 J\n\n"
            "(b) Perubahan entalpi:\n"
            "Mol HCl = 0.05 dm³ × 1.0 mol dm⁻³ = 0.05 mol\n"
            "Mol NaOH = 0.05 mol\n"
            "Mol H₂O terbentuk = 0.05 mol\n\n"
            "ΔH = −Q ÷ mol\n"
            "ΔH = −2730 ÷ 0.05\n"
            "ΔH = −54600 J mol⁻¹\n"
            "ΔH = −54.6 kJ mol⁻¹\n\n"
            "Jawapan:\n"
            "(a) Q = 2730 J\n"
            "(b) ΔH = −54.6 kJ mol⁻¹ (tanda negatif menunjukkan eksotermik)"
        ),
        "chapter": 3, "tingkatan": 5,
        "exam_year": 2022, "question_type": "struktur", "marks": 5,
        "topic": "Termokimia", "subtopic": "Kalorimetri dan Entalpi Peneutralan",
        "keywords_bm": ["haba peneutralan", "kalorimetri", "perubahan entalpi", "Q=mcΔT"],
        "keywords_en": ["heat of neutralisation", "calorimetry", "enthalpy change"],
        "diagrams": ["images/bab3_kalorimeter.png"],
    },

    # ════════════════════════════════════════════════════════════════════════
    # BAB 1 FORM 5 — REDOKS
    # ════════════════════════════════════════════════════════════════════════
    {
        "soalan": (
            "Tentukan nombor pengoksidaan manganes dalam KMnO₄."
        ),
        "jawapan": (
            "Diberi:\n"
            "KMnO₄ adalah sebatian neutral\n"
            "K = +1\n"
            "O = −2\n\n"
            "Pengiraan:\n"
            "Biarkan nombor pengoksidaan Mn = x\n"
            "(+1) + x + 4(−2) = 0\n"
            "1 + x − 8 = 0\n"
            "x = +7\n\n"
            "Jawapan:\n"
            "Nombor pengoksidaan Mn dalam KMnO₄ = +7"
        ),
        "chapter": 1, "tingkatan": 5,
        "exam_year": 2021, "question_type": "struktur", "marks": 2,
        "topic": "Redoks", "subtopic": "Nombor Pengoksidaan",
        "keywords_bm": ["nombor pengoksidaan", "KMnO4", "kalium manganat"],
        "keywords_en": ["oxidation number", "potassium permanganate", "manganese"],
        "diagrams": [],
    },
    {
        "soalan": (
            "Tindak balas berikut berlaku:\n"
            "Fe + CuSO₄ → FeSO₄ + Cu\n\n"
            "(a) Tentukan spesies yang dioksidakan.\n"
            "(b) Tentukan agen penurun dalam tindak balas ini.\n"
            "(c) Tulis setengah persamaan untuk pengoksidaan besi."
        ),
        "jawapan": (
            "(a) Spesies yang dioksidakan: Fe (besi)\n"
            "Nombor pengoksidaan Fe: 0 → +2 (meningkat = dioksidakan)\n\n"
            "(b) Agen penurun: Fe (besi)\n"
            "Agen penurun menyebabkan bahan lain diturunkan.\n"
            "Fe menyebabkan Cu²⁺ diturunkan kepada Cu.\n\n"
            "(c) Setengah persamaan pengoksidaan besi:\n"
            "Fe → Fe²⁺ + 2e⁻"
        ),
        "chapter": 1, "tingkatan": 5,
        "exam_year": 2023, "question_type": "struktur", "marks": 4,
        "topic": "Redoks", "subtopic": "Agen Pengoksidaan dan Penurunan",
        "keywords_bm": ["agen penurun", "pengoksidaan", "setengah persamaan", "Fe", "Cu2+"],
        "keywords_en": ["reducing agent", "oxidation", "half equation", "iron", "copper"],
        "diagrams": [],
    },

    # ════════════════════════════════════════════════════════════════════════
    # BAB 4 FORM 5 — POLIMER
    # ════════════════════════════════════════════════════════════════════════
    {
        "soalan": (
            "Terangkan perbezaan antara pempolimeran penambahan dan pempolimeran kondensasi."
        ),
        "jawapan": (
            "Pempolimeran Penambahan:\n"
            "• Monomer: satu jenis monomer sahaja\n"
            "• Syarat: monomer mesti mempunyai ikatan ganda dua (C=C)\n"
            "• Hasil sampingan: tiada\n"
            "• Jisim molekul polimer = n × jisim monomer\n"
            "• Contoh: etena → polietena\n\n"
            "Pempolimeran Kondensasi:\n"
            "• Monomer: dua jenis monomer yang berbeza\n"
            "• Syarat: monomer mesti mempunyai dua kumpulan berfungsi\n"
            "• Hasil sampingan: molekul kecil dibebaskan (air, HCl)\n"
            "• Jisim polimer < n × jisim monomer\n"
            "• Contoh: diamin + asid dikarboksilik → nilon"
        ),
        "chapter": 4, "tingkatan": 5,
        "exam_year": 2022, "question_type": "esei", "marks": 4,
        "topic": "Polimer", "subtopic": "Jenis Pempolimeran",
        "keywords_bm": ["pempolimeran penambahan", "pempolimeran kondensasi", "monomer", "ikatan ganda dua"],
        "keywords_en": ["addition polymerisation", "condensation polymerisation", "monomer", "double bond"],
        "diagrams": ["images/bab4_pempolimeran_penambahan.png"],
    },

    # ════════════════════════════════════════════════════════════════════════
    # MCQ EXAMPLES
    # ════════════════════════════════════════════════════════════════════════
    {
        "soalan": (
            "Antara berikut, yang manakah betul tentang tindak balas eksotermik?\n\n"
            "A. ΔH bernilai positif\n"
            "B. Tenaga diserap dari persekitaran\n"
            "C. Suhu persekitaran meningkat\n"
            "D. Kandungan tenaga produk lebih tinggi dari reaktan"
        ),
        "jawapan": (
            "Jawapan: C\n\n"
            "Penjelasan:\n"
            "Tindak balas eksotermik membebaskan tenaga haba ke persekitaran.\n"
            "Oleh itu, suhu persekitaran (larutan) meningkat.\n\n"
            "A — Salah. ΔH eksotermik adalah NEGATIF.\n"
            "B — Salah. Tenaga DIBEBASKAN (bukan diserap).\n"
            "C — BETUL. Suhu meningkat kerana tenaga dibebaskan.\n"
            "D — Salah. Kandungan tenaga produk LEBIH RENDAH dari reaktan."
        ),
        "chapter": 3, "tingkatan": 5,
        "exam_year": 2021, "question_type": "mcq", "marks": 1,
        "topic": "Termokimia", "subtopic": "Tindak Balas Eksotermik",
        "keywords_bm": ["eksotermik", "ΔH", "suhu", "tenaga"],
        "keywords_en": ["exothermic", "enthalpy", "temperature", "energy"],
        "diagrams": [],
    },
    {
        "soalan": (
            "Antara berikut, yang manakah merupakan koloid?\n\n"
            "A. Larutan gula dalam air\n"
            "B. Susu\n"
            "C. Air suling\n"
            "D. Larutan kuprum sulfat"
        ),
        "jawapan": (
            "Jawapan: B\n\n"
            "Penjelasan:\n"
            "Susu adalah koloid — zarah tersebar (lemak) bersaiz 1–100 nm dalam medium cecair.\n"
            "Koloid kelihatan homogen tetapi zarah tidak larut sepenuhnya.\n"
            "A, C, D adalah larutan tulen (homogen, zarah < 1 nm)."
        ),
        "chapter": 1, "tingkatan": 4,
        "exam_year": 2020, "question_type": "mcq", "marks": 1,
        "topic": "Pengenalan Kimia", "subtopic": "Pengelasan Jirim",
        "keywords_bm": ["koloid", "susu", "larutan", "campuran"],
        "keywords_en": ["colloid", "milk", "solution", "mixture"],
        "diagrams": [],
    },
]


# ---------------------------------------------------------------------------
# CONVERTER: Question dict → ChemistryChunk-compatible dict
# ---------------------------------------------------------------------------

def questions_to_chunks(questions: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Convert SPM_QUESTIONS list into chunk dicts ready for FAISS indexing.
    Each question becomes one chunk in index_qa.
    """
    if questions is None:
        questions = SPM_QUESTIONS

    chunks = []
    for i, q in enumerate(questions):
        # Build full text content
        content = (
            f"Soalan SPM {q.get('exam_year', '')} "
            f"(Bab {q.get('chapter', '?')}, Tingkatan {q.get('tingkatan', '?')}):\n\n"
            f"{q['soalan']}\n\n"
            f"Jawapan / Skema Pemarkahan:\n{q['jawapan']}"
        )

        # Build embed text with rich keywords
        embed_text = (
            f"Soalan SPM kimia: {q['soalan'][:200]}\n"
            f"Topik: {q.get('topic', '')}\n"
            f"Subtopik: {q.get('subtopic', '')}\n"
            f"Kata kunci: {', '.join(q.get('keywords_bm', []) + q.get('keywords_en', []))}\n"
            f"{content}"
        )

        chunk_id = (
            f"spm_{q.get('exam_year', '0000')}_"
            f"bab{q.get('chapter', '0')}_"
            f"t{q.get('tingkatan', '0')}_"
            f"{q.get('question_type', 'q')}_{i:04d}"
        )

        chunks.append({
            "chunk_id": chunk_id,
            "source_file": f"spm_{q.get('exam_year', 'unknown')}.md",
            "content_type": "qa_scheme",
            "chapter": q.get("chapter"),
            "tingkatan": q.get("tingkatan"),
            "topic": q.get("topic", ""),
            "subtopic": q.get("subtopic", ""),
            "content": content,
            "embed_text": embed_text,
            "keywords_bm": q.get("keywords_bm", []),
            "keywords_en": q.get("keywords_en", []),
            "formulas": [],
            "equations": [],
            "diagrams": [{"path": d, "alt": ""} for d in q.get("diagrams", [])],
            "has_worked_example": True,
            "has_diagram": len(q.get("diagrams", [])) > 0,
            "has_table": False,
            "language": "BM",
            "exam_year": q.get("exam_year"),
            "question_type": q.get("question_type"),
            "marks": q.get("marks", 0),
        })

    return chunks


def load_from_markdown_files(questions_dir: str) -> List[Dict[str, Any]]:
    """
    Load past year questions from Markdown files in knowledge_base/questions/
    This is the preferred way to add large numbers of questions.

    Expected file format (one file per year):
        knowledge_base/questions/past_years/SPM_2023_Chemistry.md

    Each question block starts with a level-2 heading:
        ## Soalan 1 [Bab 3, Tingkatan 4, 3 markah]
        ...soalan...

        ### Jawapan
        ...jawapan/skema...

        Keywords: mol, jisim, ...
    """
    from pathlib import Path
    import re

    q_dir = Path(questions_dir)
    if not q_dir.exists():
        return []

    all_chunks = []
    for md_file in sorted(q_dir.rglob("*.md")):
        # Extract year from filename
        year_match = re.search(r'(\d{4})', md_file.stem)
        exam_year = int(year_match.group(1)) if year_match else None

        content = md_file.read_text(encoding='utf-8')

        # Use the existing chunker for these files
        from chunker import MarkdownChunker
        from metadata_tagger import MetadataTagger
        chunker = MarkdownChunker()
        tagger = MetadataTagger()

        raw_chunks = chunker.chunk_file(md_file)
        for rc in raw_chunks:
            tagged = tagger.tag(rc)
            tagged.content_type = "qa_scheme"
            if exam_year:
                tagged.exam_year = exam_year
            d = tagged.to_dict()
            all_chunks.append(d)

    return all_chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    chunks = questions_to_chunks()
    print(f"Loaded {len(chunks)} questions from built-in bank")
    for c in chunks[:3]:
        print(f"\n{'─'*50}")
        print(f"ID     : {c['chunk_id']}")
        print(f"Type   : {c['question_type']} | Marks: {c['marks']}")
        print(f"Topic  : {c['topic']} > {c['subtopic']}")
        print(f"Year   : {c['exam_year']}")
        print(f"Content: {c['content'][:200]}...")
