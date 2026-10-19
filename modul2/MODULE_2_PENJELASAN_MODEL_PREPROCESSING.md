# **PENJELASAN MODEL & PREPROCESSING — MODULE 2**

Dokumen ini menjelaskan **bagian preprocessing** dan **bagian model** secara rinci: *apa* yang dipakai, *bagaimana cara kerjanya*, *mengapa dipilih*, dan **bukti eksperimen** yang mendasarinya.

> Dokumen pendamping: `MODULE_2_ANALISIS.md` (analisis mendalam hasil) dan `MODULE_2_EKSPERIMEN_PREPROCESSING.ipynb` (kode eksperimen).

---

# BAGIAN A — PREPROCESSING

## A.0 Gambaran Umum Pipeline

Urutan yang dipakai (konfigurasi final = "V4"):

```
teks mentah
   │
   ├─ 1. Perbaikan encoding (mojibake)      → emoji utuh kembali
   ├─ 2. Lowercase                          → menyatukan kosakata
   ├─ 3. Hapus URL & mention                → buang noise
   ├─ 4. Pertahankan KATA hashtag (#tag→tag) → jaga sinyal terkuat
   ├─ 5. Ekspansi kontraksi (can't→can not) → rapikan bentuk
   ├─ 6. Emoji → teks (😭→"loudly crying face") → ubah simbol jadi kata
   ├─ 7. Normalisasi slang (bihday→birthday) → satukan varian kata
   ├─ 8. Hapus non-alfabet + rapikan spasi  → buang sisa noise
   ├─ 9. Tokenization                       → pecah jadi token
   ├─ 10. Stopword removal                  → buang kata umum
   ├─ 11. Lemmatization                     → normalkan ke bentuk dasar
   └─ 12. Deduplikasi                       → cegah data leakage
   │
teks bersih  →  masuk ke model
```

## A.1 Tabel Lengkap Teknik Preprocessing

| # | Teknik | Apa yang dilakukan | Bukti di data |
|---|---|---|---|
| 1 | Perbaikan encoding | `encode('latin-1').decode('utf-8')` | 10.756 tweet (33,7%) rusak |
| 2 | Lowercase | semua huruf → kecil | 0 tweet pakai ALL-CAPS |
| 3 | Hapus URL & mention | regex | noise, bukan sinyal kelas |
| 4 | **Pertahankan kata hashtag** | `#tag` → `tag` | hashtag token paling diskriminatif |
| 5 | Normalisasi slang & kontraksi | kamus + regex | `u`=763, `bihday`=703, `ur`=443 |
| 6 | Emoji → teks | `emoji.demojize` | 33,7% tweet punya emoji |
| 7 | Hapus non-alfabet | regex `[^a-z\s]` | angka, tanda baca, emoji sisa |
| 8 | Tokenization | Penn Treebank tokenizer | — |
| 9 | **Stopword removal** | set lookup (198 kata) | A1 unggul tipis (lihat A.3) |
| 10 | **Lemmatization** | WordNet + POS | A2 (stemming) kalah |
| 11 | **Deduplikasi** | hash baris identik | 2.432 duplikat |

---

## A.2 Cara Kerja Masing-Masing Teknik Preprocessing

### A.2.1 Perbaikan Encoding (Mojibake)

**Masalah:** emoji memakai UTF-8 sepanjang 4 byte. Kalau dibaca sebagai Latin-1 (1 byte = 1 karakter), satu emoji pecah jadi 4 karakter aneh.

Contoh: `😭` = byte `F0 9F 98 AD` → dibaca Latin-1 menjadi `ð` `Ÿ` `˜` ``.

**Cara kerja:**

1. `text.encode('latin-1')` — mengembalikan setiap karakter ke **byte asalnya** (bit-nya masih utuh).
   `'angryð'` → `b'angry\xf0'`
2. `.decode('utf-8')` — menafsirkan ulang byte itu dengan aturan yang benar → emoji utuh.

**Syarat:** seluruh string harus bisa di-encode Latin-1; kalau tidak → dikembalikan apa adanya (`try/except`).

### A.2.2 Case Folding (Lowercase)

Mengubah semua huruf menjadi kecil. Bagi komputer `"Data"` ≠ `"data"`. Efek sampingnya kehilangan penekanan huruf kapital — aman di sini karena **tidak ada** tweet ber-kata ALL-CAPS.

### A.2.3 Regex: URL, Mention, dan Hashtag

| Pola | Arti | Aksi |
|---|---|---|
| `http\S+\|www\.\S+` | "http"/"www." + non-spasi | hapus URL |
| `@\w+` | `@` + karakter kata | hapus mention |
| `#(\w+)` | `#` + kata (**grup tangkap**) | ganti `\1` → `#run` jadi `run` |
| `[^a-z\s]` | bukan huruf/spasi | hapus angka, tanda baca, emoji sisa |
| `\s+` | 1+ spasi | jadikan satu spasi |

**Kenapa `#(\w+)` penting:** karena memakai **grup tangkap** `(...)`, kita membuang hanya simbol `#` sementara **katanya dipertahankan**. Inilah inti teknik A3.

### A.2.4 HTML Entity & Emoji → Teks

- **HTML entity**: web menyimpan karakter khusus sebagai entitas — `&amp;`→`&`, `&lt;`→`<`. `html.unescape()` mengembalikannya. 1.816 tweet terdampak.
- **Emoji → teks**: `emoji.demojize()` memetakan emoji ke **nama resmi (CLDR short name)**: `😭`→`loudly_crying_face`, `😎`→`smiling_face_with_sunglasses`; lalu `_` diganti spasi. Tujuannya: emoji yang tadinya dibuang regex `[^a-z\s]` berubah menjadi **kata** yang bisa dipelajari model.

### A.2.5 Normalisasi Slang & Kontraksi

Berbasis **kamus (lookup table)**:

- Tingkat token: `SLANG_DICT.get(token, token)` → `u`→`you`, `bihday`→`birthday`, `gr8`→`great`.
- Tingkat string: regex `n't` → ` not`, sehingga `can't` → `can not`.

**Catatan mekanis:** banyak hasil normalisasi justru kata umum yang **ada di daftar stopword** (`you`, `not`, `are`) → terbuang lagi di tahap stopword removal. Karena itu efeknya paling kecil (§A.3, V1).

### A.2.6 Tokenization

NLTK `word_tokenize` memakai **Penn Treebank tokenizer** — berbasis aturan regex, **bukan** `split(" ")`:

- Memisahkan kontraksi: `"don't stop"` → `['do', "n't", 'stop']`; `"can't"` → `['ca', "n't"]`
- Memisahkan tanda baca jadi token sendiri: `"i'm u r"` → `['i', "'m", 'u', 'r']`

**Konsekuensi:** karena `can't` dipecah jadi `ca` + `n't`, ekspansi kontraksi **harus** dijalankan di tingkat string **sebelum** tokenisasi.

### A.2.7 Stopword Removal

1. Ambil daftar stopword NLTK → **198 kata** (`a, about, above, after, again, against, ...`).
2. Ubah jadi `set` → lookup O(1).
3. Buang token yang ada di set.

**Temuan kritis:** `not`, `no`, `against` **ada** di daftar itu → penanda negasi ikut terbuang. Inilah alasan teknis kenapa A1 (tanpa stopword removal) justru unggul tipis.

### A.2.8 Stemming (Porter) — Cara Kerja

Memotong afiks berdasarkan **aturan**, tanpa kamus. Porter = rangkaian langkah (step 1a–5b) berisi aturan penggantian akhiran dengan syarat *measure* m > 0.

Hasil nyata NLTK:

| Input | Output | Catatan |
|---|---|---|
| running | run | benar |
| racists | racist | benar |
| happily | happili | **bukan kata asli** |
| universities | univers | **bukan kata asli** |
| studies | studi | **bukan kata asli** |
| generously | gener | **bukan kata asli** |
| dysfunctional | dysfunct | **bukan kata asli** |

Kelebihan: cepat, tanpa kamus. Kekurangan (**over-stemming**): makna kabur → kalah dari lemmatization di dataset ini.

### A.2.9 Lemmatization (WordNet) — Cara Kerja

Mencari bentuk dasar lewat **kamus WordNet**, dan **butuh POS tag**.

| Input | POS | Output |
|---|---|---|
| running | verb | run |
| was | verb | be |
| better | adjective | good |
| mice | noun | mouse |
| children | noun | child |
| studies | noun | study |

**Bukti POS itu krusial** — tanpa POS: `running` → `running`, `better` → `better` (tidak berubah!). Karena itu pipeline menjalankan lemmatization **dua kali**: `pos='v'` lalu default (noun).

### A.2.10 Deduplikasi

Setiap nilai dibandingkan lewat **hash**-nya; hash identik = duplikat persis → sisakan satu. Kompleksitas O(n).

**Efeknya bukan sekadar "data bersih":** mencegah tweet yang sama muncul di **train** dan **validation** sekaligus → menghindari **data leakage** (skor validasi palsu tinggi).

---

## A.3 Eksperimen Preprocessing

Semua varian dijalankan dengan **model & split identik** (Embedding+BiGRU, 2 seed), jadi hanya preprocessing yang berubah.

### Eksperimen 1 — Variasi dasar

| Varian | n | Akurasi | macro-F1 | F1 kelas 1 | Vocab |
|---|---|---|---|---|---|
| A0 baseline (stopwords + lemma) | 31.962 | 94,74% | 0,752 | 0,533 | 15.141 |
| A1 tanpa stopword removal | 31.962 | 95,00% | 0,782 | 0,591 | 15.249 |
| A2 stemming (Porter) | 31.962 | 94,61% | 0,748 | 0,525 | 14.060 |
| **A3 pertahankan kata hashtag** | 31.962 | **95,50%** | **0,818** | **0,661** | 28.787 |

**Kesimpulan:** A3 menang. Kata hashtag menambah ~13.600 token informatif (vocab hampir 2×), dan **F1 kelas minoritas melonjak 0,533 → 0,661** → sinyal hashtag memang memisahkan kelas 1, bukan sekadar menambah kapasitas.

### Eksperimen 2 — 3 teknik tambahan di atas A3

| Varian | n | Akurasi | ±std | macro-F1 | F1 kelas 1 |
|---|---|---|---|---|---|
| V1 A3 + slang/kontraksi | 31.962 | 95,69% | 0,0015 | 0,827 | 0,678 |
| V2 A3 + encoding/emoji | 31.962 | 95,87% | 0,0006 | 0,828 | 0,677 |
| V3 A3 + deduplikasi | 29.530 | 96,09% | 0,0041 | 0,830 | 0,682 |
| **V4 A3 + semua teknik** | 29.530 | **96,16%** | 0,0029 | **0,834** | **0,688** |

**Kesimpulan:** tiap teknik menambah perbaikan, dan efeknya **saling menumpuk** → V4 jadi konfigurasi final.

---

# BAGIAN B — MODEL

## B.1 Arsitektur Lengkap

```
Input (string, shape=(1,))
   ↓
TextVectorization  →  (20,)        vocab 30.000, panjang dipatok 20
   ↓
Embedding(30000, 64, mask_zero=True)  →  (20, 64)
   ↓
SpatialDropout1D(0.3)  →  (20, 64)
   ↓
Bidirectional(GRU(64, tanh))  →  (128,)
   ↓
Dense(64, relu)  →  (64,)
   ↓
Dropout(0.4)  →  (64,)
   ↓
Dense(1, sigmoid)  →  (1,)         probabilitas kelas 1
```

**Rincian parameter:**

| Layer | Output shape | Params | Fungsi |
|---|---|---|---|
| Embedding | (None, 20, 64) | 1.920.000 | vektor kata yang dipelajari |
| Bidirectional GRU | (None, 128) | 49.920 | konteks dua arah |
| Dense (relu) | (None, 64) | 8.256 | kombinasi non-linear |
| Dense (sigmoid) | (None, 1) | 65 | probabilitas |
| **Total** | | **1.978.241 (~7,6 MB)** | didominasi tabel embedding |

## B.2 Cara Kerja Tiap Komponen + Alasan Pemilihannya

**1. TextVectorization** — tiga tahap: standardize → split → konversi token ke indeks. Saat `adapt(X_train)` ia **membangun kosakata hanya dari data training** (hitung frekuensi, ambil 30.000 teratas) → indeks 0 untuk padding. `output_sequence_length=20` memotong/menambal setiap kalimat jadi tepat 20 angka.
*Alasan:* terintegrasi dalam model, dan mencegah kebocoran kosakata dari data validasi.

**2. Embedding** — tabel pencarian `(30.000 × 64)`. Setiap indeks kata menunjuk ke satu vektor 64 angka yang **dipelajari** selama training (bukan dihitung manual), sehingga kata bermakna mirip berdekatan. `mask_zero=True` membuat padding (indeks 0) diabaikan.
*Alasan:* menangkap kemiripan makna antar kata, jauh lebih kaya daripada one-hot.

**3. SpatialDropout1D** — mematikan **seluruh channel** (satu dimensi embedding penuh) secara acak saat training. Berbeda dari Dropout biasa yang mematikan elemen acak.
*Alasan:* dimensi dalam satu vektor kata saling berkaitan → mematikan per-channel lebih tepat.

**4. Bidirectional GRU(64)** — GRU memproses token satu per satu sambil menyimpan memori (`h`), dengan dua gate:
- **Reset gate (`r`)** — seberapa banyak memori lama "dilupakan".
- **Update gate (`z`)** — seberapa banyak memori lama dipertahankan.

  `h̃ = tanh(W·[r ⊙ h_prev, x])`, lalu `h = (1−z) ⊙ h_prev + z ⊙ h̃`

`Bidirectional` menjalankan **dua** GRU (kiri→kanan dan kanan→kiri) lalu menggabungkan outputnya → 64+64 = 128 dimensi.
*Alasan:* (a) gate mengurangi **vanishing gradient**, (b) konteks dibaca dua arah sehingga setiap kata "melihat" kata sebelum dan sesudahnya.

**5. Dense(64, relu)** — lapisan fully-connected untuk mengombinasikan fitur sekuens menjadi representasi non-linear.

**6. Dropout(0.4)** — mematikan 40% neuron acak saat training agar model tidak bergantung pada satu jalur.

**7. Dense(1, sigmoid)** — `σ(z)=1/(1+e^(−z))` memetakan skor ke (0,1) = probabilitas kelas 1.

## B.3 Setup Pelatihan

| Komponen | Pilihan | Alasan |
|---|---|---|
| Loss | `binary_crossentropy` | tepat untuk klasifikasi biner; `−(y·log p + (1−y)·log(1−p))` |
| Optimizer | `Adam(1e-3)` | adaptive per-parameter → konvergen cepat & stabil |
| Batch size | 128 | kompromi kecepatan & kestabilan gradien |
| Epoch | maks 15 | — |
| EarlyStopping | `patience=3`, `restore_best_weights=True` | berhenti sebelum overfitting, kembalikan bobot terbaik |
| ReduceLROnPlateau | `factor=0.5`, `patience=2` | perkecil LR saat val_loss stagnan → konvergen lebih halus |
| Class weight | **tidak dipakai** | lihat §B.4 — terbukti merugikan akurasi |

## B.4 Eksperimen Pemilihan Model

> Semua eksperimen di bagian ini dijalankan pada **preprocessing lama** (sebelum teknik V1–V4). Angka absolutnya **tidak** sebanding dengan hasil final; yang valid adalah **perbandingan relatif** antar konfigurasi pada kondisi yang sama.

### B.4.1 Pemilihan arsitektur

| Arsitektur | Akurasi | macro-F1 | F1 kelas 1 |
|---|---|---|---|
| BiLSTM(64), emb 64 | 94,57% | 0,766 | 0,561 |
| 2× BiLSTM (64→32), emb 64 | 94,87% | 0,745 | 0,516 |
| BiLSTM(64), emb 128 | 94,85% | 0,743 | 0,514 |
| **BiGRU(64), emb 64** | **94,85%** | **0,776** | **0,579** |

**Kesimpulan:** BiGRU dipilih — macro-F1 terbaik, **dan** paling cepat (30s vs 44s). Menambah layer/embedding tidak membantu (overfitting).

### B.4.2 Penanganan ketidakseimbangan kelas

Validation: 6.393 sampel (5.945 mayoritas / 448 minoritas).

| Strategi | Akurasi | macro-F1 | F1 kelas 1 | FP | TP |
|---|---|---|---|---|---|
| **Tanpa bobot kelas** | **94,57%** | 0,766 | 0,561 | 121 | 222 |
| `class_weight='balanced'` | 86,19% | 0,676 | 0,430 | 768 | 333 |
| balanced + oversample | 92,01% | 0,737 | 0,518 | 337 | 274 |

**Aritmatika keputusannya:**

```
class_weight='balanced':  TP +111 (222→333)  tapi  FP +647 (121→768)
Δbenar = +111 − 647 = −536  →  akurasi turun ~8,4 poin
```

**Kesimpulan:** pembobotan kelas adalah alat untuk menaikkan **recall minoritas** dengan **mengorbankan akurasi**. Pada data 93:7, kerugian FP (mayoritas 93% dari data) jauh melebihi keuntungan TP → model final **tanpa** `class_weight`.

### B.4.3 Ambang keputusan (threshold)

| Threshold | macro-F1 |
|---|---|
| 0,50 (default) | 0,776 |
| **0,35** | **0,780** |

Menurunkan threshold menaikkan macro-F1 — berguna bila recall kelas minoritas lebih diprioritaskan.

## B.5 Hasil Model Final

Validation set, 5.906 sampel:

```
              precision    recall  f1-score   support
Normal (0)        0.976     0.984     0.980      5503
Hate (1)          0.757     0.672     0.712       403
accuracy                              0.963      5906
macro avg         0.867     0.828     0.846      5906
```

| | Prediksi 0 | Prediksi 1 |
|---|---|---|
| **Aktual 0** | 5.415 (TN) | 88 (FP) |
| **Aktual 1** | 132 (FN) | 271 (TP) |

**Diagnosis overfitting** (bukti per epoch):

| Epoch | Train acc | Val acc | Val loss |
|---|---|---|---|
| 1 | 0,9375 | 0,9567 | 0,1180 |
| **2 (terbaik)** | **0,9725** | **0,9629** | **0,1132** |
| 3 | 0,9873 | 0,9563 | 0,1483 |
| 4 | 0,9952 | 0,9534 | 0,1976 |
| 5 (stop) | 0,9974 | 0,9519 | 0,2026 |

Train naik ke 0,9974 tapi val **turun** setelah epoch 2 (gap melebar dari ~1 poin ke ~4,5 poin) → `EarlyStopping` mengembalikan bobot epoch 2.

---

# BAGIAN C — RANGKUMAN SELURUH EKSPERIMEN

| # | Eksperimen | Yang dibandingkan | Pemenang | Bukti kunci |
|---|---|---|---|---|
| 1 | Arsitektur model | BiLSTM vs 2×BiLSTM vs emb128 vs BiGRU | **BiGRU(64)** | macro-F1 0,776 (terbaik) & tercepat |
| 2 | Imbalance | no-weight vs balanced vs balanced+oversample | **no-weight** | akurasi 94,57% vs 86,19% |
| 3 | Threshold | 0,50 vs 0,35 | **0,35** (bila recall prioritas) | macro-F1 0,780 vs 0,776 |
| 4 | Preprocessing dasar | A0 vs A1 vs A2 vs A3 | **A3 (hashtag)** | macro-F1 0,818 vs 0,752 |
| 5 | Teknik tambahan | V1 vs V2 vs V3 vs V4 | **V4 (semua)** | akurasi 96,16%, macro-F1 0,834 |

**Konfigurasi final** = Arsitektur #1 + Imbalance #2 + Preprocessing #5:
**BiGRU(64), tanpa class_weight, preprocessing V4** → **akurasi 96,29%, macro-F1 0,846.**

---

*Semua angka berasal dari eksekusi nyata pada `~/.venvs/pembelajaran-mesin` (TensorFlow 2.21.0, Python 3.12).*
