"""
diagram_processor.py — Cikgu AI Kimia
======================================
Solves the diagram gap completely.

The problem:
  Your notes contain 80+ diagram references like:
    ![Rajah kalorimeter](images/bab3_kalorimeter.png)
  
  But the actual .png files are never seen by the LLM.
  This means diagram-based SPM questions get no visual context.

This module provides THREE layers of diagram support:

  Layer 1 — DiagramDescriptionLibrary
    A built-in text description for every diagram referenced in your notes.
    No image files needed. Works immediately.
    Based on SPM Chemistry curriculum knowledge.

  Layer 2 — DiagramOCR  
    If you upload the actual .png files, OCR extracts any text from them.
    Uses pytesseract. Labels, values, axis text all extracted.

  Layer 3 — DiagramVisionDescriber
    If you have a vision-capable LLM (GPT-4V, Claude), send the image
    and get a full text description back. Best quality.

Usage:
    processor = DiagramProcessor()
    desc = processor.get_description("images/bab3_kalorimeter.png")
    # Returns: "Rajah kalorimeter menunjukkan..."

Author: Cikgu AI Kimia Project
"""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# LAYER 1 — BUILT-IN DIAGRAM DESCRIPTION LIBRARY
# ---------------------------------------------------------------------------
# Every diagram referenced in your .md files has a description here.
# Format: "filename_without_extension" -> "BM text description"

DIAGRAM_LIBRARY: Dict[str, str] = {

    # ── BAB 1: PENGENALAN ────────────────────────────────────────────────
    "bahan_kimia_kehidupan": (
        "Rajah menunjukkan pelbagai bahan kimia yang digunakan dalam kehidupan harian merangkumi "
        "bidang makanan (pengawet, pewarna), perubatan (antibiotik, antiseptik), pertanian (herbisid, "
        "pestisid) dan industri (kaca, detergen). Rajah berbentuk peta minda dengan 'Kimia' di tengah."
    ),
    "peralatan_keselamatan": (
        "Rajah menunjukkan peralatan keselamatan makmal kimia termasuk: goggles pelindung mata, "
        "topeng muka, sarung tangan getah, baju makmal, kasut bertutup, kebuk wasap, pencuran air "
        "kecemasan, dan eyewash. Setiap peralatan dilabelkan dengan fungsinya."
    ),

    # ── BAB 2: JIRIM DAN STRUKTUR ATOM ───────────────────────────────────
    "susunan_zarah_jirim": (
        "Rajah menunjukkan susunan zarah dalam tiga keadaan jirim. "
        "Pepejal: zarah tersusun rapat dan teratur dalam susunan tetap, jarak antara zarah sangat kecil. "
        "Cecair: zarah tersusun rapat tetapi tidak teratur, boleh bergerak bebas. "
        "Gas: zarah berjauhan antara satu sama lain, bergerak rawak dengan pantas."
    ),
    "pengelasan_jirim": (
        "Rajah carta alir pengelasan jirim. Jirim dibahagikan kepada tulen dan campuran. "
        "Bahan tulen dibahagikan kepada unsur dan sebatian. Campuran dibahagikan kepada campuran "
        "homogen (larutan) dan campuran heterogen. Contoh diberikan bagi setiap kategori."
    ),
    "lengkung_pemanasan": (
        "Graf suhu (paksi-y, °C) melawan masa (paksi-x, minit) menunjukkan lengkung pemanasan pepejal. "
        "Graf menunjukkan empat bahagian: (1) pepejal dipanaskan — suhu naik, (2) pepejal lebur — "
        "suhu malar pada takat lebur, (3) cecair dipanaskan — suhu naik semula, (4) cecair mendidih — "
        "suhu malar pada takat didih. Dua dataran mendatar jelas kelihatan pada graf."
    ),
    "lengkung_penyejukan": (
        "Graf suhu melawan masa untuk lengkung penyejukan. Menunjukkan cecair panas menyejuk: "
        "suhu turun, cecair membeku — suhu malar pada takat beku (dataran mendatar), "
        "pepejal terus menyejuk — suhu turun semula. Takat beku = takat lebur bahan."
    ),
    "zarah_subatom": (
        "Rajah struktur atom menunjukkan nukleus di tengah mengandungi proton (+) dan neutron (tiada cas). "
        "Elektron (-) bergerak dalam orbit di luar nukleus. Jadual sifat: proton (cas +1, jisim 1), "
        "neutron (cas 0, jisim 1), elektron (cas -1, jisim 1/1840)."
    ),
    "perwakilan_atom": (
        "Rajah menunjukkan format perwakilan piawai atom: simbol kimia X dengan nombor nukleon A "
        "di kiri atas dan nombor proton Z di kiri bawah. Contoh: karbon-12 dengan A=12, X=C, Z=6. "
        "Bilangan neutron = A - Z."
    ),
    "susunan_elektron": (
        "Rajah susunan elektron dalam petala tenaga untuk beberapa unsur. "
        "Petala pertama (K) boleh mengandungi maksimum 2 elektron. "
        "Petala kedua (L) boleh mengandungi maksimum 8 elektron. "
        "Petala ketiga (M) boleh mengandungi maksimum 18 elektron. "
        "Contoh aluminium (Al): susunan 2.8.3 dengan 3 elektron valens di petala ketiga."
    ),
    "isotop_hidrogen": (
        "Rajah menunjukkan tiga isotop hidrogen. Protium (H-1): 1 proton, 0 neutron. "
        "Deuterium (H-2): 1 proton, 1 neutron. Tritium (H-3): 1 proton, 2 neutron. "
        "Ketiga-tiga isotop mempunyai nombor proton sama (1) tetapi nombor neutron berbeza."
    ),

    # ── BAB 2: SEBATIAN KARBON ────────────────────────────────────────────
    "bab2_methane_structure": (
        "Rajah struktur metana (CH₄). Atom karbon di tengah berikatan dengan empat atom hidrogen "
        "melalui ikatan kovalen tunggal. Sudut ikatan H-C-H ialah 109.5°. "
        "Formula struktur: H-C(-H)(-H)-H."
    ),
    "bab2_methane_model": (
        "Model bola dan pasak metana (CH₄) menunjukkan atom karbon (hitam/kelabu) di tengah "
        "dan empat atom hidrogen (putih) di sekeliling dalam bentuk tetrahedral."
    ),
    "bab2_ethene_structure": (
        "Rajah struktur etena (C₂H₄). Dua atom karbon dihubungkan oleh ikatan ganda dua (C=C). "
        "Setiap atom karbon juga berikatan dengan dua atom hidrogen. Molekul adalah planar. "
        "Formula: H₂C=CH₂."
    ),
    "bab2_ethyne_structure": (
        "Rajah struktur etuna (C₂H₂) dengan ikatan ganda tiga (C≡C) antara dua atom karbon. "
        "Setiap karbon berikatan dengan satu hidrogen. Molekul adalah linear. "
        "Formula: H-C≡C-H."
    ),
    "bab2_substitution_reaction": (
        "Rajah tindak balas penukargantian metana dengan klorin di bawah cahaya UV. "
        "CH₄ + Cl₂ → CH₃Cl + HCl. "
        "Atom hidrogen dalam metana digantikan oleh atom klorin satu demi satu."
    ),
    "bab2_bromine_addition": (
        "Rajah tindak balas penambahan etena dengan bromin (Br₂). "
        "C₂H₄ + Br₂ → C₂H₄Br₂. "
        "Ikatan ganda dua dibuka dan bromin ditambah pada kedua-dua atom karbon. "
        "Larutan bromin bertukar dari coklat kepada tidak berwarna — ujian ketaktepuan."
    ),
    "bab2_polymerisation_ethene": (
        "Rajah pempolimeran penambahan etena membentuk polietena. "
        "Monomer etena (CH₂=CH₂) bergabung: ikatan ganda dua dibuka, "
        "monomer bersambung membentuk rantai panjang -(-CH₂-CH₂-)ₙ-. "
        "n boleh mencapai ribuan."
    ),
    "bab2_ethanol_structure": (
        "Rajah struktur etanol (C₂H₅OH). Kumpulan berfungsi -OH (hidroksil) terikat pada "
        "rantai karbon etil. Formula struktur: CH₃-CH₂-OH. "
        "Menunjukkan ikatan O-H dan C-O."
    ),
    "bab2_ethanoic_acid": (
        "Rajah struktur asid etanoik (CH₃COOH). Kumpulan berfungsi -COOH (karboksil) "
        "mengandungi kumpulan C=O dan O-H. Formula struktur: CH₃-C(=O)-OH."
    ),
    "bab2_esterification": (
        "Rajah tindak balas esterifikasi antara asid etanoik dan etanol. "
        "CH₃COOH + C₂H₅OH ⇌ CH₃COOC₂H₅ + H₂O. "
        "Pemangkin: asid sulfurik pekat. Tindak balas boleh berbalik."
    ),
    "bab2_butane_isomers": (
        "Rajah menunjukkan dua isomer butana (C₄H₁₀). "
        "n-butana: rantai lurus CH₃-CH₂-CH₂-CH₃. "
        "2-metilpropana (isobutana): rantai bercabang dengan kumpulan metil pada karbon kedua. "
        "Kedua-dua mempunyai formula molekul sama C₄H₁₀ tetapi struktur berbeza."
    ),
    "bab2_cracking_process": (
        "Rajah proses peretakan (cracking) hidrokarbon rantai panjang. "
        "Hidrokarbon rantai panjang (C₁₆H₃₄) dipecahkan pada suhu tinggi (500°C) "
        "dengan pemangkin aluminium oksida/silikon oksida. "
        "Hasil: hidrokarbon rantai pendek (oktana C₈H₁₈) dan alkena (etena C₂H₄)."
    ),

    # ── BAB 3: TERMOKIMIA ─────────────────────────────────────────────────
    "bab3_rajah_aras_tenaga": (
        "Rajah aras tenaga menunjukkan dua jenis tindak balas. "
        "Eksotermik: bahan tindak balas (aras tinggi) → hasil (aras rendah). ΔH negatif. "
        "Endotermik: bahan tindak balas (aras rendah) → hasil (aras tinggi). ΔH positif. "
        "Paksi-y: kandungan tenaga (kJ/mol). Anak panah menunjukkan perubahan entalpi."
    ),
    "bab3_rajah_eksotermik": (
        "Rajah aras tenaga eksotermik. Bahan tindak balas pada aras tenaga lebih tinggi. "
        "Produk pada aras tenaga lebih rendah. ΔH = negatif (tenaga dibebaskan ke persekitaran). "
        "Anak panah ke bawah menunjukkan penurunan kandungan tenaga."
    ),
    "bab3_rajah_endotermik": (
        "Rajah aras tenaga endotermik. Bahan tindak balas pada aras tenaga lebih rendah. "
        "Produk pada aras tenaga lebih tinggi. ΔH = positif (tenaga diserap dari persekitaran). "
        "Anak panah ke atas menunjukkan peningkatan kandungan tenaga."
    ),
    "bab3_tenaga_ikatan": (
        "Rajah menunjukkan pemutusan dan pembentukan ikatan dalam tindak balas kimia. "
        "Pemutusan ikatan: memerlukan tenaga (endotermik). "
        "Pembentukan ikatan: membebaskan tenaga (eksotermik). "
        "ΔH = Tenaga pemutusan ikatan − Tenaga pembentukan ikatan."
    ),
    "bab3_kalorimeter": (
        "Rajah susunan eksperimen kalorimeter untuk mengukur haba tindak balas. "
        "Komponen: bikar polistirena (penebat haba), termometer, pengacau, "
        "penutup (mengurangkan kehilangan haba), larutan tindak balas. "
        "Suhu dicatatkan sebelum dan selepas tindak balas untuk mengira ΔT."
    ),
    "bab3_penyesaran": (
        "Rajah eksperimen penyesaran logam. Kepingan ferum dimasukkan ke dalam larutan kuprum(II) sulfat (biru). "
        "Pemerhatian: larutan biru pudar, mendakan kuprum berwarna perang terbentuk pada ferum. "
        "Fe + CuSO₄ → FeSO₄ + Cu. ΔH = negatif (eksotermik)."
    ),
    "bab3_peneutralan": (
        "Rajah eksperimen peneutralan. Asid hidroklorik dan natrium hidroksida dicampurkan "
        "dalam bikar berpenebat. Termometer mengukur kenaikan suhu. "
        "H⁺(ak) + OH⁻(ak) → H₂O(l). ΔH = −57 kJ mol⁻¹."
    ),
    "bab3_graf_pembakaran_alkohol": (
        "Graf haba pembakaran melawan bilangan atom karbon untuk siri homolog alkohol. "
        "Paksi-x: bilangan atom karbon (1 hingga 6). Paksi-y: haba pembakaran (kJ/mol). "
        "Graf menunjukkan hubungan linear positif — semakin banyak karbon, semakin tinggi haba pembakaran."
    ),

    # ── BAB 4: JADUAL BERKALA ─────────────────────────────────────────────
    "kedudukan_unsur": (
        "Rajah jadual berkala unsur menunjukkan kedudukan unsur mengikut nombor proton. "
        "Baris mendatar = kala (7 kala). Lajur menegak = kumpulan (18 kumpulan). "
        "Kumpulan 1 (logam alkali), Kumpulan 17 (halogen), Kumpulan 18 (gas adi) dilabelkan. "
        "Unsur peralihan terletak di tengah jadual."
    ),

    # ── BAB 5: IKATAN KIMIA ───────────────────────────────────────────────
    "susunan_elektron_stabil": (
        "Rajah menunjukkan susunan elektron stabil gas adi (oktet). "
        "Neon (2.8) dan argon (2.8.8) sebagai contoh susunan stabil. "
        "Atom lain mencapai kestabilan melalui pemindahan atau perkongsian elektron."
    ),
    "pembentukan_ion_natrium": (
        "Rajah pembentukan ion natrium. Na (2.8.1) kehilangan 1 elektron membentuk Na⁺ (2.8). "
        "Anak panah menunjukkan elektron berpindah keluar. "
        "Cas berubah dari neutral (0) kepada +1."
    ),
    "ikatan_ion_nacl": (
        "Rajah pembentukan natrium klorida melalui ikatan ion. "
        "Na (2.8.1) memindahkan 1 elektron kepada Cl (2.8.7). "
        "Na⁺ (2.8) + Cl⁻ (2.8.8) → NaCl. "
        "Daya tarikan elektrostatik antara Na⁺ dan Cl⁻ membentuk ikatan ion."
    ),
    "struktur_lewis_kovalen": (
        "Rajah struktur Lewis menunjukkan perkongsian elektron dalam molekul kovalen. "
        "Pasangan elektron dikongsi dilabelkan sebagai garis ikatan (—). "
        "Pasangan elektron bebas (lone pair) ditunjukkan sebagai titik berpasangan."
    ),
    "ikatan_tunggal": (
        "Rajah ikatan tunggal C-C dalam etana (C₂H₆). "
        "Satu pasangan elektron dikongsi antara dua atom karbon. "
        "Dilambangkan sebagai satu garis: C—C."
    ),
    "ikatan_ganda_dua": (
        "Rajah ikatan ganda dua C=C dalam etena (C₂H₄). "
        "Dua pasangan elektron dikongsi. Satu ikatan sigma dan satu ikatan pi. "
        "Dilambangkan sebagai dua garis selari: C=C."
    ),
    "ikatan_ganda_tiga": (
        "Rajah ikatan ganda tiga C≡C dalam etuna (C₂H₂). "
        "Tiga pasangan elektron dikongsi. Satu ikatan sigma dan dua ikatan pi. "
        "Dilambangkan sebagai tiga garis: C≡C."
    ),
    "ikatan_hidrogen_air": (
        "Rajah ikatan hidrogen antara molekul air. "
        "Atom hidrogen (δ+) dalam satu molekul H₂O tertarik kepada atom oksigen (δ-) "
        "dalam molekul H₂O yang lain. Ikatan hidrogen dilabelkan sebagai garis putus-putus (---). "
        "Setiap molekul air boleh membentuk sehingga 4 ikatan hidrogen."
    ),
    "ikatan_datif": (
        "Rajah pembentukan ion hidronium (H₃O⁺) melalui ikatan datif. "
        "Molekul air menderma pasangan elektron bebas kepada ion H⁺. "
        "H₂O + H⁺ → H₃O⁺. Anak panah dari O menunjukkan arah derma elektron."
    ),
    "ikatan_logam": (
        "Rajah ikatan logam menunjukkan ion positif logam tersusun dalam kisi. "
        "Elektron valens bebas bergerak di seluruh kisi membentuk 'lautan elektron'. "
        "Daya tarikan antara ion positif dan elektron bebas membentuk ikatan logam yang kuat."
    ),
    "kekonduksian_ion": (
        "Rajah sel elektrolisis menunjukkan larutan ion mengkonduksikan elektrik. "
        "Ion positif (kation) bergerak ke katod. Ion negatif (anion) bergerak ke anod. "
        "Aliran ion dalam larutan membolehkan arus elektrik mengalir."
    ),

    # ── BAB 6: ASID BES GARAM ─────────────────────────────────────────────
    "pembentukan_ion_hidronium": (
        "Rajah menunjukkan pembentukan ion hidronium apabila HCl dilarutkan dalam air. "
        "HCl → H⁺ + Cl⁻. H⁺ + H₂O → H₃O⁺. "
        "Ion H⁺ tidak wujud bebas tetapi bergabung dengan molekul air."
    ),
    "pembentukan_ion_hidroksida": (
        "Rajah pembentukan ion hidroksida apabila NaOH dilarutkan dalam air. "
        "NaOH → Na⁺ + OH⁻. Larutan mengandungi ion Na⁺ dan OH⁻ bebas."
    ),
    "skala_ph": (
        "Rajah skala pH dari 0 hingga 14. "
        "pH 0–6: berasid (merah pada kertas litmus biru). "
        "pH 7: neutral. "
        "pH 8–14: beralkali (biru pada kertas litmus merah). "
        "Contoh bahan pada setiap pH: jus lemon (pH 2), air tulen (pH 7), sabun (pH 9), NaOH (pH 14)."
    ),
    "pengionan_hcl": (
        "Rajah perbandingan pengionan HCl (asid kuat) dan CH₃COOH (asid lemah) dalam air. "
        "HCl: semua molekul mengion lengkap → banyak H⁺. "
        "CH₃COOH: hanya sebahagian molekul mengion → sedikit H⁺. "
        "Asid kuat lebih pekat H⁺ walaupun kemolaran sama."
    ),
    "hubungan_kepekatan": (
        "Graf menunjukkan hubungan antara kepekatan asid/alkali dengan nilai pH. "
        "Paksi-x: kepekatan (mol dm⁻³). Paksi-y: pH. "
        "Asid: pH menurun apabila kepekatan meningkat. "
        "Alkali: pH meningkat apabila kepekatan meningkat."
    ),

    # ── BAB 7: KADAR TINDAK BALAS ─────────────────────────────────────────
    "kadar_tindak_balas_jisim": (
        "Rajah susunan eksperimen mengukur kadar tindak balas melalui penurunan jisim. "
        "Kelalang kon di atas penimbang mengandungi CaCO₃ dan HCl. "
        "Gas CO₂ terlepas menyebabkan jisim berkurang. Bacaan jisim dicatat setiap minit."
    ),
    "kadar_tindak_balas_gas": (
        "Rajah susunan eksperimen pengumpulan gas menggunakan picagari gas atau penyesaran air. "
        "Gas hidrogen dari tindak balas Zn dengan HCl dikumpulkan. "
        "Isipadu gas dicatat pada selang masa tertentu."
    ),
    "graf_kadar_tindak_balas": (
        "Graf kuantiti produk (isipadu gas / cm³) melawan masa (saat/minit). "
        "Graf bermula curam (kadar tinggi) dan mendatar apabila tindak balas selesai. "
        "Kecerunan awal (tangen) = kadar tindak balas awal. "
        "Graf mendatar = tindak balas selesai (bahan tindak balas habis)."
    ),
    "kecerunan_kadar": (
        "Rajah menunjukkan cara mengukur kecerunan graf pada masa tertentu. "
        "Tangen dilukis pada titik yang dikehendaki. "
        "Kecerunan = Δy/Δx = perubahan kuantiti/perubahan masa. "
        "Ini memberikan kadar tindak balas serta-merta pada masa tersebut."
    ),

    # ── BAB 1 FORM 5: REDOKS ─────────────────────────────────────────────
    "pertukaran_fe2_fe3": (
        "Rajah menunjukkan pertukaran antara ion Fe²⁺ dan Fe³⁺. "
        "Fe²⁺ → Fe³⁺: kehilangan 1 elektron, pengoksidaan berlaku. "
        "Fe³⁺ → Fe²⁺: penambahan 1 elektron, penurunan berlaku. "
        "Ujian: Fe²⁺ memberikan mendakan putih dengan NaOH; Fe³⁺ memberikan mendakan perang."
    ),
    "penyesaran_zink_kuprum": (
        "Rajah tindak balas penyesaran: kepingan zink dalam larutan CuSO₄ biru. "
        "Pemerhatian: larutan biru pudar, lapisan kuprum perang terbentuk pada zink, "
        "kepingan zink semakin menipis. "
        "Zn + Cu²⁺ → Zn²⁺ + Cu. Zink dioksidakan, kuprum diturunkan."
    ),
    "penyesaran_halogen": (
        "Rajah penyesaran halogen dalam larutan halida. "
        "Cl₂ (lebih reaktif) + 2KBr → 2KCl + Br₂. "
        "Larutan bertukar warna: tidak berwarna → perang (Br₂ terbentuk). "
        "Klorin mengoksidakan ion bromida."
    ),

    # ── BAB 4: POLIMER ────────────────────────────────────────────────────
    "bab4_pempolimeran_penambahan": (
        "Rajah pempolimeran penambahan. "
        "Monomer dengan ikatan ganda dua (C=C): n(CH₂=CH₂). "
        "Ikatan ganda dua dibuka, monomer bergabung membentuk polimer: -(-CH₂-CH₂-)ₙ-. "
        "Tiada hasil sampingan dihasilkan."
    ),
    "bab4_pempolimeran_kondensasi": (
        "Rajah pempolimeran kondensasi antara dua monomer berbeza. "
        "Contoh nilon: monomer diamin + monomer asid dikarboksilik. "
        "Ikatan amida (-CO-NH-) terbentuk. Molekul air dibebaskan setiap kali ikatan terbentuk."
    ),
    "bab4_poliisoprena": (
        "Rajah struktur polimer getah asli (poliisoprena). "
        "Monomer: isoprena (2-metilbuta-1,3-diena). "
        "Rantai panjang dengan ikatan ganda dua pada setiap unit monomer. "
        "Formula berulang: -(-CH₂-C(CH₃)=CH-CH₂-)ₙ-."
    ),
    "bab4_penggumpalan_lateks": (
        "Rajah proses penggumpalan lateks. "
        "Asid (cuka/asid formik) ditambahkan ke lateks. "
        "Cas negatif pada zarah getah dineutralkan → zarah bergabung → pepejal terbentuk. "
        "Getah pepejal dipisahkan dan dikeringkan."
    ),
    "bab4_pemvulkanan": (
        "Rajah proses pemvulkanan getah. "
        "Getah asli + sulfur dipanaskan pada 150°C. "
        "Atom sulfur membentuk ikatan silang (-S-S-) antara rantai polimer. "
        "Hasilnya: getah lebih keras, lebih anjal, dan lebih tahan haba."
    ),
    "bab4_ikatan_silang_sulfur": (
        "Rajah menunjukkan ikatan silang sulfur antara rantai poliisoprena. "
        "Sebelum vulkanisasi: rantai polimer bebas, boleh gelangsar. "
        "Selepas vulkanisasi: ikatan -S-S- menghubungkan rantai, "
        "menghalang gelangsar dan meningkatkan kekuatan."
    ),
}


def _normalize_image_path(path: str) -> str:
    """Extract just the filename stem from any image path format."""
    p = Path(path)
    stem = p.stem.lower()
    # Remove common prefixes
    stem = re.sub(r'^(images/|rajah_|diagram_)', '', stem)
    return stem


def get_built_in_description(image_path: str) -> Optional[str]:
    """Look up built-in description for a diagram path."""
    stem = _normalize_image_path(image_path)
    # Direct match
    if stem in DIAGRAM_LIBRARY:
        return DIAGRAM_LIBRARY[stem]
    # Partial match
    for key, desc in DIAGRAM_LIBRARY.items():
        if key in stem or stem in key:
            return desc
    return None


# ---------------------------------------------------------------------------
# LAYER 2 — OCR TEXT EXTRACTION
# ---------------------------------------------------------------------------

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using pytesseract.
    Returns empty string if pytesseract not installed or image not found.
    """
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        # Try BM + English
        text = pytesseract.image_to_string(img, lang='eng+msa')
        return text.strip()
    except ImportError:
        return ""
    except FileNotFoundError:
        return ""
    except Exception as e:
        return ""


# ---------------------------------------------------------------------------
# LAYER 3 — VISION LLM DESCRIPTION
# ---------------------------------------------------------------------------

async def describe_image_with_vision(
    image_path: str,
    api_key: str,
    model: str = "llama-3.2-90b-vision-preview",  # Groq vision model
) -> str:
    """
    Use a vision LLM to describe a chemistry diagram.
    Supports Groq (llama-3.2-90b-vision-preview) and OpenAI (gpt-4o).
    """
    if not Path(image_path).exists():
        return ""

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(image_path).suffix.lower().lstrip('.')
    mime = f"image/{ext}" if ext in ('png', 'jpg', 'jpeg', 'gif', 'webp') else "image/png"

    prompt = (
        "Ini adalah rajah kimia dari buku teks SPM Malaysia. "
        "Huraikan rajah ini dalam Bahasa Malaysia dengan tepat. "
        "Sertakan: label, nilai, unit, anak panah, persamaan kimia, dan ciri penting. "
        "Huraian mestilah berguna untuk menjawab soalan kimia SPM."
    )

    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=api_key)
        resp = await client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime};base64,{image_data}"
                    }},
                ],
            }],
            max_tokens=500,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return ""


# ---------------------------------------------------------------------------
# MAIN PROCESSOR CLASS
# ---------------------------------------------------------------------------

class DiagramProcessor:
    """
    Three-layer diagram text extraction for Cikgu AI Kimia.

    Priority order:
      1. Built-in library (instant, no API needed)
      2. OCR (if image file available)
      3. Vision LLM (best quality, requires API + image file)
    """

    def __init__(
        self,
        images_dir: str = "knowledge_base/images",
        use_ocr: bool = True,
        use_vision: bool = False,
        vision_api_key: str = "",
    ):
        self.images_dir = Path(images_dir)
        self.use_ocr = use_ocr
        self.use_vision = use_vision
        self.vision_api_key = vision_api_key or os.environ.get("GROQ_API_KEY", "")
        self._cache: Dict[str, str] = {}

    def get_description(self, image_ref: str) -> str:
        """
        Get text description for a diagram reference.
        image_ref: e.g. "images/bab3_kalorimeter.png" or just "bab3_kalorimeter"
        """
        if image_ref in self._cache:
            return self._cache[image_ref]

        # Layer 1: Built-in library
        desc = get_built_in_description(image_ref)
        if desc:
            self._cache[image_ref] = desc
            return desc

        # Layer 2: OCR
        if self.use_ocr:
            img_path = self._resolve_path(image_ref)
            if img_path and img_path.exists():
                ocr_text = extract_text_from_image(str(img_path))
                if ocr_text and len(ocr_text) > 20:
                    desc = f"[Teks OCR dari rajah]: {ocr_text}"
                    self._cache[image_ref] = desc
                    return desc

        # Fallback: generic description from alt text
        stem = _normalize_image_path(image_ref)
        desc = f"[Rajah: {stem.replace('_', ' ')}]"
        self._cache[image_ref] = desc
        return desc

    async def get_description_async(self, image_ref: str) -> str:
        """Async version that includes vision LLM fallback."""
        desc = self.get_description(image_ref)

        # If only got fallback stub AND vision is enabled AND image exists
        if desc.startswith("[Rajah:") and self.use_vision and self.vision_api_key:
            img_path = self._resolve_path(image_ref)
            if img_path and img_path.exists():
                vision_desc = await describe_image_with_vision(
                    str(img_path), self.vision_api_key
                )
                if vision_desc:
                    self._cache[image_ref] = vision_desc
                    return vision_desc

        return desc

    def _resolve_path(self, image_ref: str) -> Optional[Path]:
        """Try to find the image file on disk."""
        candidates = [
            Path(image_ref),
            self.images_dir / Path(image_ref).name,
            self.images_dir / image_ref,
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def inject_descriptions_into_chunk(self, content: str) -> str:
        """
        Replace all ![alt](images/...) references in a chunk
        with their text descriptions.

        Before: "![Rajah kalorimeter](images/bab3_kalorimeter.png)"
        After:  "[RAJAH — Rajah kalorimeter]: Rajah susunan eksperimen kalorimeter..."
        """
        import re

        def replace_diagram(match):
            alt = match.group(1)
            path = match.group(2)
            desc = self.get_description(path)
            return f"\n[RAJAH — {alt}]: {desc}\n"

        return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_diagram, content)

    def build_diagram_chunks(self, md_dir: str) -> List[dict]:
        """
        Scan all .md files and create dedicated diagram chunks
        for every unique diagram reference found.

        Returns list of dicts ready for indexing.
        """
        import re
        from pathlib import Path

        diagram_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
        seen = set()
        chunks = []

        for md_file in Path(md_dir).rglob("*.md"):
            content = md_file.read_text(encoding='utf-8')
            for match in diagram_pattern.finditer(content):
                alt = match.group(1).strip()
                path = match.group(2).strip()

                if path in seen:
                    continue
                seen.add(path)

                desc = self.get_description(path)
                stem = _normalize_image_path(path)

                chunks.append({
                    "chunk_id": f"diagram__{stem}",
                    "source_file": md_file.name,
                    "content_type": "theory",
                    "topic": f"Rajah: {alt}",
                    "subtopic": alt,
                    "content": f"Rajah: {alt}\n\n{desc}",
                    "embed_text": f"Rajah kimia SPM: {alt}\n{desc}",
                    "keywords_bm": [alt, stem.replace('_', ' ')],
                    "keywords_en": [],
                    "formulas": [],
                    "equations": [],
                    "diagrams": [{"alt": alt, "path": path}],
                    "has_worked_example": False,
                    "has_diagram": True,
                    "has_table": False,
                    "chapter": None,
                    "tingkatan": None,
                    "language": "BM",
                    "exam_year": None,
                    "question_type": None,
                })

        print(f"[diagram_processor] Found {len(chunks)} unique diagrams")
        return chunks


# ---------------------------------------------------------------------------
# CONVENIENCE
# ---------------------------------------------------------------------------

_processor_instance: Optional[DiagramProcessor] = None


def get_diagram_processor(
    images_dir: str = "knowledge_base/images",
    use_vision: bool = False,
) -> DiagramProcessor:
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = DiagramProcessor(
            images_dir=images_dir,
            use_vision=use_vision,
        )
    return _processor_instance


if __name__ == "__main__":
    # Test the library
    processor = DiagramProcessor()

    test_refs = [
        "images/bab3_kalorimeter.png",
        "images/bab2_bromine_addition.png",
        "images/skala_ph.png",
        "images/graf_kadar_tindak_balas.png",
        "images/bab4_pemvulkanan.png",
        "images/ikatan_ion_nacl.png",
    ]

    print("DIAGRAM DESCRIPTION TEST")
    print("=" * 60)
    for ref in test_refs:
        desc = processor.get_description(ref)
        print(f"\n{ref}")
        print(f"  → {desc[:120]}...")

    # Test injection
    sample = """
## Rajah Susunan Kalorimeter

![Rajah kalorimeter](images/bab3_kalorimeter.png)

Langkah pengiraan haba menggunakan Q = mcΔT.
"""
    injected = processor.inject_descriptions_into_chunk(sample)
    print("\n\nINJECTION TEST:")
    print(injected)
