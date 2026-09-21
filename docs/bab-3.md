3.1 Gambaran Umum Pipeline

Alur end-to-end: EDA → prapemrosesan → rekayasa fitur multi-blok → dua jalur pemodelan paralel (GBDT tabular + neural cross-encoder) → kalibrasi & blending → threshold tuning dari OOF → retrain penuh → submission
Prinsip yang dipegang di seluruh pipeline: from-scratch, train-only fitting, test hanya untuk inferensi

3.2 Prapemrosesan dan Strategi Validasi

Normalisasi teks (lowercase, whitespace)
Union-find grouping atas judul/isi ternormalisasi untuk mendeteksi baris yang berbagi judul atau isi
StratifiedGroupKFold berbasis grup ini, alasan mencegah kebocoran validasi lewat pasangan identik/near-identik

3.3 Rekayasa Fitur
Dipecah jadi sub-subsection karena blok fiturnya cukup banyak dan berbeda karakter:

3.3.1 Fitur Semantik Dasar (Blok SEM) — coverage/overlap, coverage berbobot IDF, jendela selaras terbaik, n-gram verbatim, fitur urutan (Kendall tau, pasangan berurutan), fitur dukungan (exact/sinonim akronim/stem/char-trigram fuzzy), entitas berbasis cap rate, cloze slot filling, rival similarity & PMI, mismatch negasi, mismatch angka
3.3.2 Fitur Penyelarasan Monoton (Blok ALG) — LIS-based alignment, token yang gagal selaras, analisis gap di sekitar posisi ekspektasi (entitas asing, negasi asing)
3.3.3 Fitur Retrieval (Blok RET) — cosine similarity TF-IDF word 1-2gram & char 3-5gram judul terhadap korpus lead isi, posisi rank di antara kandidat
3.3.4 Fitur Perbandingan Berpasangan (Blok DUP & NDP) — deteksi struktur duplikat persis, identifikasi rival near-duplicate, cross-feature differencing (fitur "milik sendiri vs milik rival")
3.3.5 Fitur Leksikon Eksternal (Blok EXT) — gazetteer wilayah berjenjang, leksikon nama orang, normalisasi KBBA, sinonim tesaurus; semua di-fit train-only

3.4 Arsitektur Model

3.4.1 Ensemble GBDT — LightGBM, XGBoost, CatBoost, multi-seed
3.4.2 Cross-Encoder Neural BiGRU + Atensi (gaya ESIM) — embedding acak, GRU judul/isi, atensi silang, komposisi, pooling avg+max
3.4.3 Skema Data Latih Tambahan Eksternal Single-Pass (NN-EXT) — ini bagian yang wajib dijelaskan jelas: kenapa skema lama (pretrain lalu fine-tune) diganti, bagaimana CLICK-ID dipakai sebagai baris latih tambahan (natural + terkorupsi) digabung satu training dengan data fold-lomba, kepatuhan terhadap klarifikasi Q&A soal larangan pretraining

3.5 Strategi Pelatihan

Seed locking & determinism (PYTHONHASHSEED, cudnn deterministic, dsb)
Generator korupsi sintetis 7 resep (SUB_ENT, NEG_DEL, NEG_INS, PERM, SUB_NUM, SUB_MULTI, PAIR) dan pembobotan tetangga Word2Vec untuk substitusi entitas
Penanganan imbalance: oversampling kelas minoritas pada training GRU, scale_pos_weight/class_weight pada GBDT
Early stopping dan cross-validation yang group-aware

3.6 Kalibrasi, Threshold Tuning, dan Inferensi

Pencarian threshold dari OOF yang memaksimalkan Macro F1
Kalibrasi Platt (logistic 1D) untuk menyamakan ruang skor GBDT ensemble, NN-EXT, dan GRU sebelum blending
Bobot blend tetap dari pencarian coordinate descent pada OOF
Retrain penuh lalu prediksi test di ruang skor yang sama

3.7 Reproduksibilitas dan Audit

Kontrol seed di seluruh tahap
Validasi format submission terhadap sample_submission
Hash audit untuk jejak reproduksi
