# **ANALISIS MENDALAM — KLASIFIKASI TEKS (MODULE 2)**

Dokumen ini menjelaskan **apa** yang dikerjakan, **mengapa** keputusan itu diambil, dan **bagaimana cara kerja** masing-masing tekniknya, beserta bukti numerik.

Rujukan file:

- Notebook utama: `MODULE_2_MACHINE_LEARNING_TEXT_CLASSIFICATION.ipynb`
- Notebook eksperimen: `MODULE_2_EKSPERIMEN_PREPROCESSING.ipynb`
- Dataset: `dataset/assignment/train.csv` (berlabel) & `test.csv` (tanpa label)

---

## 0. Ringkasan Eksekutif

| Aspek | Nilai |
|---|---|
| Tugas | Klasifikasi biner tweet (0 = Normal/Netral, 1 = Hate/Offensive) |
| Data latih | 31.962 baris → 29.530 setelah deduplikasi |
| Evaluasi | Validation set 20% (5.906 baris), stratified |
| **Akurasi akhir** | **96,29%** |
| Macro-F1 | 0,846 |
| F1 kelas minoritas (1) | 0,712 |
| Arsitektur | Embedding(64) → SpatialDropout → **BiGRU(64)** → Dense(64, ReLU) → Dropout → Dense(1, Sigmoid) |

**Lima temuan utama:**

1. **Akurasi adalah metrik yang menipu di dataset ini** — menebak kelas mayoritas terus sudah memberi 93,18%. Jadi target "85%" nyaris tidak bermakna; pembeda sebenarnya adalah macro-F1 / recall kelas minoritas.
2. **Hashtag adalah token paling diskriminatif.** Membuang seluruh hashtag (praktik umum) justru membuang sinyal terkuat.
3. **Ada data leakage dari duplikat.** Menghapus 2.432 tweet kembar memberi lompatan tunggal terbesar.
4. **Emoji di dataset ini rusak (mojibake), bukan hilang** — dan bisa dipulihkan sepenuhnya.
5. **Menyeimbangkan kelas dengan `class_weight` justru menurunkan akurasi** (kontra-intuitif, dijelaskan di §5).

---

## 1. Karakteristik Dataset — dan Mengapa Ini Menentukan Segalanya

Semua keputusan di bawah bukan asumsi, tapi respons terhadap ciri data berikut:

| Fakta data | Angka | Implikasi ke pipeline |
|---|---|---|
| Rasio kelas | **92,99% : 7,01%** (29.720 : 2.242) | Akurasi menipu → wajib pakai macro-F1 / F1 kelas 1 |
| Majority baseline (val) | **93,18%** | Target 85% ada *di bawah* tebakan trivial |
| Panjang tweet | median ≈ 12 kata; p99 ≈ 14; maks 21 | `MAX_LEN = 20` sudah menampung hampir semua; tidak perlu 100+ |
| Duplikat persis | **2.432 (7,6%)** | Risiko **data leakage** train↔validation |
| Emoji rusak (mojibake) | **10.756 (33,7%)** | Sinyal emosi terbuang kalau tidak diperbaiki |
| Entitas HTML (`&amp;`) | 1.816 | Perlu `html.unescape` |
| Slang/singkatan | `2`=1.659, `u`=763, `4`=738, `bihday`=703, `ur`=443 | Kosakata terpecah → perlu normalisasi |
| Karakter berulang ≥3× | 21.721 (68%) | Alternatif normalisasi tambahan |
| Rusak jadi kosong setelah stopword removal | 571 (1,79%) | Stopword removal agresif berisiko |

**Bukti hashtag bersifat diskriminatif** (top hashtag per kelas):

- Label 1: `#trump`(133), `#politics`(94), `#libtard`(76), `#liberal`(75), `#sjw`(74), `#blm`(28), `#bigot`(27)
- Label 0: `#love`(1531), `#positive`(874), `#healthy`(570), `#smile`(548), `#thankful`(491)

Namun menariknya, **keberadaan hashtag itu sendiri tidak prediktif**:

- `P(label=1 | ada hashtag) = 0,0698` ≈ `P(label=1) = 0,0701`

Artinya sinyalnya murni **leksikal** (hashtag *apa*), bukan struktural (ada/tidak ada hashtag).

---

## 2. Pipeline: Apa yang Dilakukan dan Mengapa

### 2.1 Preprocessing

| Tahap | Apa | Mengapa (bukti) |
|---|---|---|
| Perbaikan encoding | `text.encode('latin-1').decode('utf-8')` | 33,7% tweet ter-mojibake; memperbaiki `ð` → emoji asli |
| Emoji → teks | `emoji.demojize` (`😭` → "loudly crying face") | Memulihkan sinyal emosi |
| Hapus URL & mention | regex | Noise; hampir tidak membawa sinyal kelas |
| **Pertahankan kata hashtag** | `#tag` → `tag` | Bukti A3: +0,7% macro-F1 vs membuang hashtag (§4.1) |
| Normalisasi slang & kontraksi | kamus + regex | Menyatukan `bihday`/`birthday`, `u`/`you`; `u`=763, `bihday`=703 |
| Hapus non-alfabet | regex `[^a-z\s]` | Angka, tanda baca, sisa emoji = noise |
| Tokenization | `word_tokenize` | Unit analisis |
| **Stopword removal** | NLTK | Tradeoff: membuang ~33% teks (12→8 kata) **dan** negasi ("not" termasuk stopword NLTK) → A1 unggul tipis (§4.1) |
| **Lemmatization** | WordNet | Lebih tepat daripada stemming; A2 (stemming) justru kalah (§4.1) |
| **Deduplikasi** | `drop_duplicates('tweet')` | 2.432 kembar → hilangkan **leakage**; lonjakan terbesar (§4.2) |

**Mengapa deduplikasi krusial (bukan sekadar "bersih-bersih"):**
Kalau tweet identik muncul di train *dan* validation, model seolah sudah melihat jawabannya. Akurasi jadi **optimistis palsu**. Karena label duplikat konsisten (0 tweet dengan label berbeda), menghapusnya aman.

**Mengapa stopword removal yang agresif berbahaya di sini:**
Tweet ini pendek (median 12 kata). Memotong 33% teks menyisakan sedikit konteks, dan NLTK menganggap "not"/"no" sebagai stopword — padahal negasi bisa membalik makna. Terbukti: 571 tweet menjadi **string kosong** setelah stopword removal.

### 2.2 Representasi & Arsitektur Model

| Keputusan | Pilihan | Mengapa (bukti) |
|---|---|---|
| Vektorisasi | `TextVectorization` (int + padding) | Terintegrasi di dalam model; vocabulary di-`adapt` **hanya dari train** (anti-leakage) |
| Embedding | 64 dim, `mask_zero=True` | Masking agar padding tidak dianggap kata |
| Layer berulang | **BiGRU(64)** | Eksperimen: GRU acc 0,9485 / macro-F1 0,7757 — mengalahkan BiLSTM (0,9457/0,7662), 2×BiLSTM (0,9487/0,7446), dan emb128 (0,9485/0,7434), **dan lebih cepat** |
| Bidirectional | ya | Konteks dibaca dua arah |
| Regularisasi | SpatialDropout1D 0,3 + Dropout 0,4 | Dataset relatif kecil → cegah overfitting |
| Output | Dense(1, sigmoid) | Klasifikasi biner + `binary_crossentropy` |

### 2.3 Pelatihan

- `Adam(1e-3)`, `batch_size=128`, maksimum 15 epoch
- `EarlyStopping(patience=3, restore_best_weights=True)` → mengembalikan bobot epoch terbaik
- `ReduceLROnPlateau(factor=0.5, patience=2)` → turunkan LR saat val_loss stagnan
- **Tanpa `class_weight`** — lihat §5 untuk alasan berbasis bukti

---

## 3. Cara Kerja Masing-Masing Teknik (Mekanisme)

### 3.1 Teknik Preprocessing

#### 3.1.1 Perbaikan Encoding (Mojibake)

**Masalahnya:** emoji sebenarnya memakai UTF-8 yang panjangnya 4 byte. Kalau file dibaca dengan asumsi Latin-1 (1 byte = 1 karakter), maka 1 emoji pecah menjadi 4 karakter aneh.

Contoh: emoji `😭` = byte `F0 9F 98 AD`. Dibaca Latin-1 → `ð` `Ÿ` `˜` `` (4 karakter). Itu sebabnya di data muncul `ð` dan `\x9f`.

**Cara kerjanya:**

1. `text.encode('latin-1')` → mengembalikan tiap karakter ke **byte asalnya** (karena bit-nya masih utuh). Contoh: `'angryð'` → `b'angry\xf0'`.
2. `.decode('utf-8')` → menafsirkan ulang byte itu dengan aturan yang benar → emoji kembali utuh.

**Syarat:** seluruh string harus bisa di-encode Latin-1 (semua karakter ≤ 0xFF). Bila ada karakter di luar itu, `UnicodeEncodeError` muncul dan string dikembalikan apa adanya (blok `try/except`).

#### 3.1.2 Case Folding (Lowercase)

Mengubah semua huruf jadi kecil. Bagi komputer `"Data"` ≠ `"data"`, sehingga tanpa ini kosakata terpecah tanpa guna. Efek sampingnya kehilangan penekanan (`PENTING` vs `penting`) — pada data ini aman karena **0 tweet** memakai kata ALL-CAPS.

#### 3.1.3 Regex: Menghapus URL, Mention, dan Hashtag

Regex adalah pola pencarian. Pola yang dipakai:

| Pola | Arti | Aksi |
|---|---|---|
| `http\S+\|www\.\S+` | "http" atau "www." diikuti karakter non-spasi | hapus URL utuh |
| `@\w+` | `@` lalu 1+ karakter kata | hapus mention |
| `#(\w+)` | `#` + kata, **ditangkap sebagai grup 1** | diganti `\1` → `#run` jadi `run` (kata dipertahankan) |
| `[^a-z\s]` | semua karakter **bukan** huruf/ spasi | hapus angka, tanda baca, sisa emoji |
| `\s+` | 1 atau lebih spasi | jadikan satu spasi |

Perhatikan `#(\w+)`: karena memakai **grup tangkap** `(...)`, kita bisa membuang hanya simbol `#` tapi mempertahankan katanya. Ini inti teknik A3.

#### 3.1.4 HTML Entity Decoding

Halaman web menyimpan karakter khusus sebagai entitas: `&amp;` = `&`, `&lt;` = `<`, `&gt;` = `>`. `html.unescape()` mengubahnya balik menjadi karakter asli. 1.816 tweet di data ini terdampak.

#### 3.1.5 Emoji → Teks

Setelah encoding diperbaiki, emoji menjadi karakter Unicode yang sah. `emoji.demojize()` memetakan emoji ke **nama resminya (CLDR short name)**:

- `😭` → `loudly_crying_face`
- `😎` → `smiling_face_with_sunglasses`

Lalu `_` diganti spasi. **Tujuannya:** simbol emoji akan dibuang oleh regex `[^a-z\s]`, sedangkan hasil teksnya berupa kata yang bisa dipelajari model. Jadi emoji yang tadinya hilang berubah menjadi fitur.

#### 3.1.6 Normalisasi Slang & Kontraksi

Berbasis **kamus (lookup table)**:

- Tingkat token: `SLANG_DICT.get(token, token)` → `u`→`you`, `bihday`→`birthday`, `gr8`→`great`.
- Tingkat string (kontraksi): regex `n't` → ` not`, sehingga `can't` → `can not`.

**Catatan mekanis:** sebagian besar hasil normalisasi justru kata umum yang **ada di daftar stopword** (`you`, `not`, `are`), sehingga terbuang lagi di tahap berikutnya. Itu sebabnya efeknya paling kecil di antara tiga teknik (lihat §4.2).

#### 3.1.7 Tokenization

Memecah string menjadi token. NLTK `word_tokenize` memakai **Penn Treebank tokenizer** — berbasis aturan regex, **bukan** sekadar `split(" ")`:

- Memisahkan kontraksi: `"don't stop"` → `['do', "n't", 'stop']`; `"can't"` → `['ca', "n't"]`
- Memisahkan tanda baca menjadi token sendiri: `"i'm u r"` → `['i', "'m", 'u', 'r']`

**Konsekuensi penting:** karena `can't` dipecah jadi `ca` + `n't`, ekspansi kontraksi **harus** dilakukan di tingkat string **sebelum** tokenisasi — kalau tidak, `n't` sudah terlanjur terpisah.

#### 3.1.8 Stopword Removal

Cara kerjanya sederhana dan deterministik:

1. Ambil daftar stopword NLTK → **198 kata** (`a, about, above, after, again, against, ain, all, am, an, and, any, ...`).
2. Ubah menjadi `set` → pencarian O(1) per token.
3. Buang token yang ada di dalam set.

**Temuan kritis:** `not`, `no`, dan `against` **ada** di daftar itu. Jadi stopword removal ikut membuang penanda negasi — inilah alasan teknis mengapa teknik ini berisiko untuk teks opini/sentimen, dan mengapa varian A1 (tanpa stopword removal) justru unggul tipis.

#### 3.1.9 Stemming (Porter) — Cara Kerja

Prinsip: **memotong afiks (awalan/akhiran) berdasarkan aturan**, tanpa kamus. Algoritma Porter adalah rangkaian langkah (step 1a, 1b, 1c, 2, 3, 4, 5a, 5b); tiap langkah punya daftar aturan penggantian akhiran, dengan syarat tambahan seperti nilai *measure* m > 0 (perkiraan jumlah suku kata).

Hasil nyata dari `PorterStemmer` (NLTK):

| Input | Output |
|---|---|
| running | run |
| racists | racist |
| happily | happili |
| universities | univers |
| studies | studi |
| ponies | poni |
| generously | gener |
| dysfunctional | dysfunct |

**Karakteristik yang perlu disadari:**

- Hasilnya **tidak harus kata asli** — `happili`, `gener`, `dysfunct` bukan kata bahasa Inggris.
- Bisa **bertingkat**: `generously` → `gener` → `gener` (stabil setelah sekali).
- **Kelebihan:** cepat, tidak butuh kamus, mengecilkan kosakata.
- **Kekurangan (over-stemming):** dua kata berbeda makna bisa menyatu, sehingga sinyal kabur. Ini yang membuatnya kalah dari lemmatization di dataset ini.

#### 3.1.10 Lemmatization (WordNet) — Cara Kerja

Prinsip: mencari **bentuk dasar (lemma)** melalui **kamus WordNet**, dan **membutuhkan POS tag** (kata benda/kerja/sifat).

Hasil nyata dari `WordNetLemmatizer`:

| Input | POS | Output |
|---|---|---|
| running | verb | run |
| was | verb | be |
| better | adjective | good |
| mice | noun | mouse |
| children | noun | child |
| studies | noun | study |
| dysfunctional | adjective | dysfunctional |

**Bukti bahwa POS itu krusial** — tanpa POS:

- `lemmatize("running")` → `running` (tidak berubah!)
- `lemmatize("better")` → `better` (tidak berubah!)

Karena itu di pipeline kita lemmatization dijalankan **dua kali**: pertama sebagai kata kerja (`pos='v'`), lalu default (noun) — supaya kata kerja dan kata benda sama-sama dinormalkan.

**Perbandingan kunci:** stemming memotong secara mekanis (bisa merusak makna), lemmatization memakai kamus (hasil selalu kata valid). Trade-off: lemmatization lebih lambat dan butuh POS.

#### 3.1.11 Deduplikasi

**Cara kerjanya:** setiap nilai dibandingkan lewat **hash**-nya (NLTK/pandas melakukan hashing internal). Baris dengan hash identik = duplikat persis → sisakan satu. Kompleksitasnya linear O(n).

**Efeknya bukan sekadar "data lebih bersih":** ia mencegah tweet yang sama muncul di **train** dan **validation** sekaligus. Tanpa ini, model bisa "menghafal" baris yang juga muncul saat evaluasi → skor validasi palsu tinggi.

### 3.2 Teknik Representasi & Model

#### 3.2.1 TextVectorization

Bekerja dalam tiga tahap:

1. **Standardize** (opsional) → di sini kita sudah membersihkan manual sebelumnya.
2. **Split** → memecah string menjadi token.
3. **Convert to index** → setiap token dipetakan ke integer.

`adapt(X_train)` **membangun kosakata dari data training saja**: hitung frekuensi token, ambil `max_tokens=30000` teratas, lalu beri indeks (1, 2, 3, ...; indeks 0 disimpan untuk padding). `output_sequence_length=20` membuat setiap kalimat dipotong atau di-pad menjadi tepat 20 angka.

#### 3.2.2 Embedding

Sebuah **tabel pencarian (lookup table)** berukuran `(kosakata × dimensi)` = `30000 × 64`. Setiap indeks kata menunjuk ke satu baris vektor 64 angka. **Penting:** nilai vektor ini **tidak dihitung manual, tapi dipelajari** selama training — model menggeser vektor agar kata-kata bermakna mirip berdekatan. `mask_zero=True` membuat indeks 0 (padding) diabaikan oleh layer berikutnya.

#### 3.2.3 GRU & Bidirectional

**GRU (Gated Recurrent Unit)** memproses token satu per satu sambil menyimpan **memori (hidden state `h`)**. Bedanya dengan RNN biasa: ada **gate** yang mengatur aliran informasi:

- **Reset gate (`r`)** — seberapa banyak memori lama "dilupakan" saat menghitung kandidat baru.
- **Update gate (`z`)** — seberapa banyak memori lama dipertahankan.

Secara ringkas: `h̃ = tanh(W·[r ⊙ h_prev, x])`, lalu `h = (1−z) ⊙ h_prev + z ⊙ h̃`.

Manfaatnya: **vanishing gradient** (informasi dari token jauh yang menguap) jauh berkurang dibanding RNN sederhana. Angka **64** = jumlah neuron = dimensi hidden state.

**Bidirectional** menjalankan **dua** GRU: satu membaca kiri→kanan, satu membaca kanan→kiri, lalu kedua output digabung. Akibatnya setiap kata "melihat" konteks **sebelum dan sesudahnya** — penting karena penanda kelas bisa muncul di awal atau akhir kalimat.

#### 3.2.4 Dropout & SpatialDropout1D

**Dropout** mematikan sebagian neuron secara **acak** di setiap langkah training (outputnya 0), sehingga jaringan tidak boleh bergantung pada satu jalur tertentu. Saat prediksi (inference) semua neuron aktif kembali.

**SpatialDropout1D** berbeda: ia mematikan **seluruh channel** (satu dimensi embedding penuh), bukan elemen acak. Untuk layer embedding, ini lebih tepat karena dimensi-dimensi dalam satu vektor kata saling berkaitan.

#### 3.2.5 Sigmoid & Binary Cross-Entropy

- **Sigmoid** `σ(z) = 1 / (1 + e^(−z))` memetakan skor mentah apa pun ke rentang (0, 1) → dibaca sebagai probabilitas kelas 1.
- **Binary cross-entropy**: `−(y·log(p) + (1−y)·log(1−p))` → memberi hukuman yang makin besar bila probabilitas makin jauh dari label benar.
- Keputusan akhir memakai **threshold 0,5**: `p ≥ 0,5` → kelas 1.

#### 3.2.6 Adam Optimizer

Optimizer adalah cara memperbarui bobot. **Adam (Adaptive Moment Estimation)** menyimpan dua besaran per parameter: rata-rata gradien (momentum, `m`) dan varians gradien (`v`), lalu menyesuaikan laju belajar **per parameter**. Efeknya: konvergen cepat dan stabil, serta lebih tahan terhadap pilihan learning rate.

#### 3.2.7 EarlyStopping & ReduceLROnPlateau

- **EarlyStopping** memantau `val_loss`; bila tidak membaik selama `patience=3` epoch → training dihentikan dan **bobot epoch terbaik dipulihkan**. Fungsinya: berhenti tepat sebelum overfitting parah (§4.4).
- **ReduceLROnPlateau** memantau `val_loss`; bila stagnan → learning rate dikalikan 0,5. Fungsinya: saat model "muter-muter" di sekitar minimum, langkah diperkecil agar lebih halus masuk ke lembah minimum.

---

## 4. Hasil dan Analisisnya

### 4.1 Eksperimen 1 — Variasi Preprocessing Dasar

| Varian | n | Akurasi | macro-F1 | F1 kelas 1 | Vocab |
|---|---|---|---|---|---|
| A0 baseline (stopwords + lemma) | 31.962 | 94,74% | 0,752 | 0,533 | 15.141 |
| A1 tanpa stopword removal | 31.962 | 95,00% | 0,782 | 0,591 | 15.249 |
| A2 stemming (Porter) | 31.962 | 94,61% | 0,748 | 0,525 | 14.060 |
| **A3 pertahankan kata hashtag** | 31.962 | **95,50%** | **0,818** | **0,661** | 28.787 |

**Mengapa A3 menang:** kata hashtag menambah ~13.600 token informatif (vocab hampir 2×). Kenaikan terbesar ada di **F1 kelas minoritas** (0,533 → 0,661) — artinya sinyal hashtag memang memisahkan kelas 1, bukan sekadar menambah kapasitas.

**Mengapa A1 unggul tipis:** kata fungsi & negasi dipertahankan; tidak ada data yang "menguap" jadi kosong (§3.1.8).

**Mengapa A2 kalah:** Porter memotong tanpa dasar linguistik (`dysfunction` → `dysfunct`, §3.1.9) → over-merging, makna kabur.

### 4.2 Eksperimen 2 — 3 Teknik Tambahan di Atas A3

| Varian | n | Akurasi | ±std | macro-F1 | F1 kelas 1 |
|---|---|---|---|---|---|
| V1 A3 + slang/kontraksi | 31.962 | 95,69% | 0,0015 | 0,827 | 0,678 |
| V2 A3 + encoding/emoji | 31.962 | 95,87% | 0,0006 | 0,828 | 0,677 |
| V3 A3 + deduplikasi | 29.530 | 96,09% | 0,0041 | 0,830 | 0,682 |
| **V4 A3 + semua teknik** | 29.530 | **96,16%** | 0,0029 | **0,834** | **0,688** |

Efek ketiganya **saling menumpuk** (bukan saling meniadakan), dan V4 menjadi konfigurasi final.

### 4.3 Hasil Model Final (validation set, 5.906 sampel)

```
              precision    recall  f1-score   support
Normal (0)        0.976     0.984     0.980      5503
Hate (1)          0.757     0.672     0.712       403
accuracy                              0.963      5906
macro avg         0.867     0.828     0.846      5906
```

Confusion matrix (diturunkan dari report):

| | Prediksi 0 | Prediksi 1 |
|---|---|---|
| **Aktual 0** | 5.415 (TN) | 88 (FP) |
| **Aktual 1** | 132 (FN) | 271 (TP) |

**Pembacaan analitis:**

- Kesalahan didominasi **FN (132)** — model cenderung **melewatkan** tweet hate/offensive. Precision kelas 1 (0,757) > recall (0,672): saat bilang "hate" ia cukup bisa dipercaya, tapi sering tidak berani.
- **Akurasi 96,29% vs majority baseline 93,18%** → lift nyata hanya **+3,11 poin**.

### 4.4 Overfitting — Bukti per Epoch

| Epoch | Train acc | Val acc | Val loss |
|---|---|---|---|
| 1 | 0,9375 | 0,9567 | 0,1180 |
| **2 (terbaik)** | **0,9725** | **0,9629** | **0,1132** |
| 3 | 0,9873 | 0,9563 | 0,1483 |
| 4 | 0,9952 | 0,9534 | 0,1976 |
| 5 (stop) | 0,9974 | 0,9519 | 0,2026 |

**Diagnosis:** overfitting jelas. Train accuracy naik ke 0,9974 sementara val accuracy justru **turun** setelah epoch 2, dan val loss naik hampir 2×. Gap membesar dari ~1 poin (epoch 2) ke ~4,5 poin (epoch 5). `EarlyStopping` mengembalikan bobot epoch 2.

---

## 5. Temuan Kontra-Intuitif dan Penjelasannya

### 5.1 Mengapa `class_weight='balanced'` malah menurunkan akurasi?

Diuji pada setup identik (validation 6.393 sampel, 5.945 mayoritas / 448 minoritas):

| Strategi | Akurasi | macro-F1 | F1 kelas 1 | FP | TP |
|---|---|---|---|---|---|
| **Tanpa bobot kelas** | **94,57%** | 0,766 | 0,561 | 121 | 222 |
| `class_weight='balanced'` | 86,19% | 0,676 | 0,430 | **768** | **333** |
| balanced + oversample | 92,01% | 0,737 | 0,518 | 337 | 274 |

**Aritmatikanya terbuka:** pemberian bobot menambah **+111** deteksi benar kelas minoritas (222→333), tetapi mengorbankan **+647 false positive** (121→768). Karena kelas mayoritas 93% dari data, kerugian FP jauh melebihi keuntungan TP:

```
Δbenar = +111 (TP) − 647 (FP) = −536  →  akurasi turun ~8,4 poin
```

**Kesimpulan:** `class_weight` adalah alat untuk menaikkan **recall kelas minoritas** dengan mengorbankan **akurasi keseluruhan**. Pada data sedominan ini trade-off-nya tidak menguntungkan bila metrik utamanya akurasi. Karena itu model final memakai distribusi asli.

### 5.2 Mengapa stemming kalah dari lemmatization?

Stemming memotong berdasarkan aturan karakter (`dysfunction` → `dysfunct`, `universities` → `univers`, §3.1.9), sementara lemmatization memakai kamus + POS (`running` → `run`, `was` → `be`, `better` → `good`, §3.1.10). Pada data ini stemming menciptakan over-merging yang menghapus perbedaan makna, dan hasilnya (A2: 94,61% / macro-F1 0,748) lebih rendah dari lemmatization (A3: 95,50% / 0,818).

### 5.3 Mengapa "ada hashtag" bukan fitur prediktif?

`P(label=1 | ada hashtag) = 0,0698` ≈ `P(label=1) = 0,0701`. Yang prediktif adalah **hashtag spesifik** (`#trump`, `#sjw` vs `#love`, `#thankful`). Pelajarannya: fitur "ada/tidak ada penanda" lemah; fitur "nilai spesifik" kuat.

---

## 6. Keterbatasan Analisis

1. **`test.csv` tidak berlabel** → akurasi sesungguhnya tidak dapat diukur; semua angka akurasi berasal dari validation set internal. Output `test.csv` hanya prediksi (16.086 Normal / 1.111 Hate).
2. **Semantik label disimpulkan**, bukan dari dokumentasi resmi.
3. **Deduplikasi hanya menangkap duplikat persis**, bukan near-duplicate (beda 1 karakter). Sisa potensi leakage masih ada.
4. **Perbandingan arsitektur & `class_weight` (§2.2, §5.1) dijalankan pada pipeline preprocessing lama.** Angka absolutnya tidak sebanding dengan hasil final; yang valid adalah **perbandingan relatif** pada kondisi yang sama.
5. **Hanya 2 seed per varian** → selisih akurasi < 0,3 poin ada dalam rentang noise. Kesimpulan bertumpu pada **macro-F1** yang selisihnya jauh lebih besar dan konsisten.
6. **Reparasi mojibake hanya berhasil bila seluruh string dapat di-decode latin-1→utf-8** (§3.1.1).

---

## 7. Rekomendasi Lanjutan

| Prioritas | Langkah | Alasan |
|---|---|---|
| Tinggi | Turunkan threshold keputusan dari 0,5 ke ~0,35 | Menaikkan macro-F1 ke ~0,78 — cocok bila recall kelas minoritas lebih penting |
| Tinggi | Pakai **F1 macro / PR-AUC** sebagai metrik utama | Akurasi menyesatkan pada data 93:7 |
| Sedang | Focal loss sebagai alternatif `class_weight` | Menangani imbalance tanpa menggeser batas keputusan sekeras bobot kelas |
| Sedang | Near-duplicate detection (MinHash / TF-IDF cosine) | Menutup sisa leakage yang belum tertangkap dedup persis |
| Sedang | Normalisasi karakter berulang (`soooo` → `so`) | 68% tweet terdampak — belum diuji |
| Rendah | Handle 571 tweet kosong (drop atau pertahankan stopword) | 1,79% data saat ini masuk sebagai string kosong |
| Rendah | K-fold cross-validation, bukan 1 split | Estimasi performa lebih stabil & jujur |

---

*Semua angka pada dokumen ini berasal dari eksekusi nyata notebook pada lingkungan `~/.venvs/pembelajaran-mesin` (TensorFlow 2.21.0, Python 3.12).*
