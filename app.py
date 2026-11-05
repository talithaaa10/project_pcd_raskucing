import io
import os
import struct
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# Dukungan tambahan jika pengguna mengunggah foto format iPhone (HEIC)
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

# ==================== 1. PENGATURAN HALAMAN ====================
st.set_page_config(
    page_title="Aplikasi Identifikasi Ras Kucing",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Desain tampilan (CSS) agar rapi di layar Laptop maupun HP
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1150px;
    }
    /* Sembunyikan ikon link di sebelah judul/heading Streamlit */
    [data-testid="stHeaderActionElements"],
    .stMarkdown h1 a,
    .stMarkdown h2 a,
    .stMarkdown h3 a,
    .stMarkdown h4 a {
        display: none !important;
    }
    .judul-utama {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: #ffffff;
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        margin-bottom: 1rem;
    }
    .judul-utama h1 {
        font-size: 1.55rem;
        margin: 0;
        color: #ffffff !important;
    }
    .kotak-file-terpilih {
        background: #eff6ff;
        border: 2px solid #3b82f6;
        color: #1e3a8a;
        padding: 0.85rem 1.1rem;
        border-radius: 12px;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .kotak-hasil {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
        color: white;
        padding: 1.2rem 1.4rem;
        border-radius: 14px;
        margin-bottom: 1rem;
    }
    .kotak-hasil h2 {
        margin: 0;
        font-size: 1.45rem;
        color: #ffffff !important;
    }
    .label-akurasi {
        display: inline-block;
        margin-top: 0.45rem;
        background: rgba(255, 255, 255, 0.22);
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .judul-ras-katalog {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        min-height: 2.9rem;
        font-weight: 700;
        font-size: 1rem;
    }
    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.6rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
        .judul-utama h1 {
            font-size: 1.25rem;
        }
        .kotak-hasil h2 {
            font-size: 1.2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================== 2. DATA 5 RAS KUCING ====================
DAFTAR_RAS = [
    "Bengal",
    "British Shorthair",
    "Domestic Short Hair",
    "Persian",
    "Turkish Angora",
]

INFO_RAS_KUCING = {
    "Bengal": {
        "nama_lengkap": "Kucing Bengal",
        "jumlah_data": 2477,
        "ciri_fisik": (
            "Tubuh ramping dan berotot dengan corak bulu tutul atau marmer seperti macan tutul. "
            "Bulunya pendek, rapat, dan tampak berkilau."
        ),
        "karakter": "Sangat aktif, cerdas, suka memanjat tempat tinggi, dan sering tertarik bermain air.",
        "perawatan": (
            "Sediakan tempat bermain atau panjatan, ajak bermain setiap hari, "
            "dan sikat bulunya seminggu sekali."
        ),
        "kesehatan": "Perlu pemeriksaan rutin kesehatan jantung dan mata.",
    },
    "British Shorthair": {
        "nama_lengkap": "Kucing British Shorthair",
        "jumlah_data": 567,
        "ciri_fisik": (
            "Tubuh bulat dan berisi, dada lebar, pipi tembem, mata bulat besar, "
            "serta bulu pendek yang sangat tebal dan halus."
        ),
        "karakter": "Tenang, santai, mandiri, dan ramah di dalam rumah.",
        "perawatan": (
            "Atur porsi makan agar tidak kegemukan (obesitas) dan sisir bulunya 2–3 kali seminggu."
        ),
        "kesehatan": "Jaga berat badan tetap ideal serta perhatikan kebersihan gigi dan gusi.",
    },
    "Domestic Short Hair": {
        "nama_lengkap": "Kucing Domestik (Domestic Short Hair)",
        "jumlah_data": 53027,
        "ciri_fisik": (
            "Tubuh proporsional dan lincah dengan bulu pendek rapat. Memiliki banyak variasi warna "
            "seperti belang (tabby), oranye, belang tiga (calico), hitam, atau putih."
        ),
        "karakter": "Mudah beradaptasi, pintar, mandiri, dan memiliki daya tahan tubuh yang kuat.",
        "perawatan": (
            "Perawatan sangat mudah, cukup disisir seminggu sekali, beri makanan bergizi, "
            "serta vaksin dan obat cacing secara rutin."
        ),
        "kesehatan": "Umumnya memiliki fisik yang kuat; pastikan rutin vaksinasi dan bebas kutu.",
    },
    "Persian": {
        "nama_lengkap": "Kucing Persia (Persian)",
        "jumlah_data": 4018,
        "ciri_fisik": (
            "Wajah bulat dan datar, hidung pesek, mata bulat besar, "
            "tubuh pendek berisi dengan bulu yang sangat panjang dan lebat."
        ),
        "karakter": "Kalem, manja, lembut, dan menyukai suasana rumah yang tenang serta sejuk.",
        "perawatan": (
            "Wajib disisir setiap hari agar bulu tidak kusut, bersihkan area sekitar mata secara rutin, "
            "dan pelihara di dalam ruangan."
        ),
        "kesehatan": "Jaga kebersihan kulit dari jamur serta perhatikan saluran pernapasan dan ginjal.",
    },
    "Turkish Angora": {
        "nama_lengkap": "Kucing Angora Turki (Turkish Angora)",
        "jumlah_data": 750,
        "ciri_fisik": (
            "Tubuh ramping dan anggun dengan bulu sedang-panjang yang halus seperti sutra. "
            "Wajah sedikit meruncing, telinga tegak, dan bulu ekor mengembang."
        ),
        "karakter": "Aktif, cerdas, lincah, dan senang bermain bersama pemiliknya.",
        "perawatan": (
            "Sisir bulu 2–3 kali seminggu, sediakan mainan, serta jaga kebersihan telinga dan mata."
        ),
        "kesehatan": "Perhatikan kesehatan pendengaran (terutama yang berbulu putih mata biru) dan jantung.",
    },
}


# ==================== 3. PEMBACA MODEL TFLITE ====================
class PenjalanModelTFLite:
    """
    Menjalankan file model_kucing.tflite menggunakan NumPy jika library TensorFlow
    belum terpasang di lingkungan Python.
    """

    def __init__(self, lokasi_model: str):
        with open(lokasi_model, "rb") as f:
            self.buf = f.read()
        self._siapkan_model()

    def _baca_vtable(self, pos):
        offset = struct.unpack_from("<i", self.buf, pos)[0]
        vtable = pos - offset
        ukuran, _ = struct.unpack_from("<HH", self.buf, vtable)
        return [
            struct.unpack_from("<H", self.buf, vtable + 4 + 2 * i)[0]
            for i in range((ukuran - 4) // 2)
        ]

    def _baca_vektor_tabel(self, pos, idx):
        offsets = self._baca_vtable(pos)
        if idx >= len(offsets) or offsets[idx] == 0:
            return []
        vpos = pos + offsets[idx]
        vpos += struct.unpack_from("<I", self.buf, vpos)[0]
        panjang = struct.unpack_from("<I", self.buf, vpos)[0]
        return [
            vpos + 4 + 4 * i + struct.unpack_from("<I", self.buf, vpos + 4 + 4 * i)[0]
            for i in range(panjang)
        ]

    def _baca_vektor_int(self, pos, idx):
        offsets = self._baca_vtable(pos)
        if idx >= len(offsets) or offsets[idx] == 0:
            return []
        vpos = pos + offsets[idx]
        vpos += struct.unpack_from("<I", self.buf, vpos)[0]
        panjang = struct.unpack_from("<I", self.buf, vpos)[0]
        return list(struct.unpack_from(f"<{panjang}i", self.buf, vpos + 4))

    def _baca_byte(self, pos, idx):
        offsets = self._baca_vtable(pos)
        if idx >= len(offsets) or offsets[idx] == 0:
            return b""
        vpos = pos + offsets[idx]
        vpos += struct.unpack_from("<I", self.buf, vpos)[0]
        panjang = struct.unpack_from("<I", self.buf, vpos)[0]
        return memoryview(self.buf)[vpos + 4 : vpos + 4 + panjang]

    def _baca_nilai(self, pos, idx, fmt, default=0):
        offsets = self._baca_vtable(pos)
        if idx >= len(offsets) or offsets[idx] == 0:
            return default
        return struct.unpack_from(fmt, self.buf, pos + offsets[idx])[0]

    def _baca_tabel(self, pos, idx):
        offsets = self._baca_vtable(pos)
        if idx >= len(offsets) or offsets[idx] == 0:
            return None
        tpos = pos + offsets[idx]
        return tpos + struct.unpack_from("<I", self.buf, tpos)[0]

    def _siapkan_model(self):
        root = struct.unpack_from("<I", self.buf, 0)[0]
        op_codes = self._baca_vektor_tabel(root, 1)
        buffers = self._baca_vektor_tabel(root, 4)
        sg = self._baca_vektor_tabel(root, 2)[0]
        tensor_tables = self._baca_vektor_tabel(sg, 0)
        self.inputs = self._baca_vektor_int(sg, 1)
        self.outputs = self._baca_vektor_int(sg, 2)
        operators = self._baca_vektor_tabel(sg, 3)

        peta_tipe = {0: np.float32, 1: np.float16, 2: np.int32, 9: np.int8}
        self.tensors = [None] * len(tensor_tables)

        for idx, t in enumerate(tensor_tables):
            shape = tuple(self._baca_vektor_int(t, 0))
            ttype = self._baca_nilai(t, 1, "<b", 0)
            buf_idx = self._baca_nilai(t, 2, "<I", 0)
            raw = self._baca_byte(buffers[buf_idx], 0)
            if len(raw) > 0 and ttype in peta_tipe:
                arr = np.frombuffer(raw, dtype=peta_tipe[ttype]).reshape(shape)
                self.tensors[idx] = arr.astype(np.float32) if ttype == 1 else arr.copy()

        self.daftar_operasi = []
        for op in operators:
            opcode_idx = self._baca_nilai(op, 0, "<I", 0)
            oc = op_codes[opcode_idx]
            builtin = self._baca_nilai(oc, 3, "<i", self._baca_nilai(oc, 0, "<b", 0))
            op_in = self._baca_vektor_int(op, 1)
            op_out = self._baca_vektor_int(op, 2)
            opt = self._baca_tabel(op, 4)

            if builtin == 6:
                self.tensors[op_out[0]] = self.tensors[op_in[0]].astype(np.float32)
            elif builtin == 3:
                pad = self._baca_nilai(opt, 0, "<b", 0)
                sw = self._baca_nilai(opt, 1, "<i", 1)
                sh = self._baca_nilai(opt, 2, "<i", 1)
                act = self._baca_nilai(opt, 3, "<b", 0)
                w = self.tensors[op_in[1]]
                b = self.tensors[op_in[2]]
                if w.shape[1] == 1 and w.shape[2] == 1 and sw == 1 and sh == 1:
                    w_mat = np.ascontiguousarray(w[:, 0, 0, :].T, dtype=np.float32)
                    self.daftar_operasi.append(("PW_CONV", op_in[0], op_out[0], w_mat, b, act))
                else:
                    self.daftar_operasi.append(("CONV_2D", op_in[0], op_out[0], w, b, pad, sw, sh, act))
            elif builtin == 4:
                pad = self._baca_nilai(opt, 0, "<b", 0)
                sw = self._baca_nilai(opt, 1, "<i", 1)
                sh = self._baca_nilai(opt, 2, "<i", 1)
                act = self._baca_nilai(opt, 4, "<b", 0)
                w = self.tensors[op_in[1]][0]
                b = self.tensors[op_in[2]]
                self.daftar_operasi.append(("DW_CONV", op_in[0], op_out[0], w, b, pad, sw, sh, act))
            elif builtin == 0:
                act = self._baca_nilai(opt, 0, "<b", 0)
                self.daftar_operasi.append(("ADD", op_in[0], op_in[1], op_out[0], act))
            elif builtin == 40:
                axes = tuple(int(a) for a in self.tensors[op_in[1]].flatten())
                self.daftar_operasi.append(("MEAN", op_in[0], op_out[0], axes))
            elif builtin == 9:
                act = self._baca_nilai(opt, 0, "<b", 0) if opt else 0
                w_mat = np.ascontiguousarray(self.tensors[op_in[1]].T, dtype=np.float32)
                b = self.tensors[op_in[2]]
                self.daftar_operasi.append(("FC", op_in[0], op_out[0], w_mat, b, act))
            elif builtin == 25:
                self.daftar_operasi.append(("SOFTMAX", op_in[0], op_out[0]))

    @staticmethod
    def _fungsi_aktivasi(x, act):
        if act == 1:
            np.maximum(x, 0.0, out=x)
        elif act == 3:
            np.clip(x, 0.0, 6.0, out=x)
        return x

    @staticmethod
    def _tambah_padding(x, kh, kw, sh, sw):
        _, h, w, _ = x.shape
        out_h = (h + sh - 1) // sh
        out_w = (w + sw - 1) // sw
        pad_h = max((out_h - 1) * sh + kh - h, 0)
        pad_w = max((out_w - 1) * sw + kw - w, 0)
        pt, pb = pad_h // 2, pad_h - pad_h // 2
        pl, pr = pad_w // 2, pad_w - pad_w // 2
        if pt or pb or pl or pr:
            x = np.pad(x, ((0, 0), (pt, pb), (pl, pr), (0, 0)), mode="constant")
        return x, out_h, out_w

    def prediksi(self, input_tensor: np.ndarray) -> np.ndarray:
        t = list(self.tensors)
        t[self.inputs[0]] = np.ascontiguousarray(input_tensor, dtype=np.float32)

        for op in self.daftar_operasi:
            jenis = op[0]
            if jenis == "PW_CONV":
                _, in_idx, out_idx, w_mat, b, act = op
                x = t[in_idx]
                n, h, w_dim, c_in = x.shape
                y = (x.reshape(-1, c_in) @ w_mat).reshape(n, h, w_dim, -1) + b
                t[out_idx] = self._fungsi_aktivasi(y, act)
            elif jenis == "CONV_2D":
                _, in_idx, out_idx, w, b, pad, sw, sh, act = op
                x = t[in_idx]
                out_c, kh, kw, _ = w.shape
                if pad == 0:
                    x, out_h, out_w = self._tambah_padding(x, kh, kw, sh, sw)
                else:
                    out_h = (x.shape[1] - kh) // sh + 1
                    out_w = (x.shape[2] - kw) // sw + 1
                y = np.broadcast_to(b, (x.shape[0], out_h, out_w, out_c)).copy()
                for ky in range(kh):
                    for kx in range(kw):
                        patch = x[:, ky : ky + out_h * sh : sh, kx : kx + out_w * sw : sw, :]
                        y += patch @ w[:, ky, kx, :].T
                t[out_idx] = self._fungsi_aktivasi(y, act)
            elif jenis == "DW_CONV":
                _, in_idx, out_idx, w, b, pad, sw, sh, act = op
                x = t[in_idx]
                kh, kw, c = w.shape
                if pad == 0:
                    x, out_h, out_w = self._tambah_padding(x, kh, kw, sh, sw)
                else:
                    out_h = (x.shape[1] - kh) // sh + 1
                    out_w = (x.shape[2] - kw) // sw + 1
                y = np.broadcast_to(b, (x.shape[0], out_h, out_w, c)).copy()
                for ky in range(kh):
                    for kx in range(kw):
                        patch = x[:, ky : ky + out_h * sh : sh, kx : kx + out_w * sw : sw, :]
                        y += patch * w[ky, kx, :]
                t[out_idx] = self._fungsi_aktivasi(y, act)
            elif jenis == "ADD":
                _, in1, in2, out_idx, act = op
                t[out_idx] = self._fungsi_aktivasi(t[in1] + t[in2], act)
            elif jenis == "MEAN":
                _, in_idx, out_idx, axes = op
                t[out_idx] = np.mean(t[in_idx], axis=axes)
            elif jenis == "FC":
                _, in_idx, out_idx, w_mat, b, act = op
                t[out_idx] = self._fungsi_aktivasi(t[in_idx] @ w_mat + b, act)
            elif jenis == "SOFTMAX":
                _, in_idx, out_idx = op
                x = t[in_idx]
                e = np.exp(x - np.max(x, axis=-1, keepdims=True))
                t[out_idx] = e / np.sum(e, axis=-1, keepdims=True)

        return t[self.outputs[0]][0]


# ==================== 4. FUNGSI PEMROSESAN GAMBAR & PREDIKSI ====================
@st.cache_resource(show_spinner="Memuat model MobileNetV2...")
def muat_model():
    folder_aplikasi = os.path.dirname(os.path.abspath(__file__))
    lokasi_tflite = os.path.join(folder_aplikasi, "model_kucing.tflite")

    if not os.path.exists(lokasi_tflite):
        return None

    # Coba gunakan interpreter bawaan jika tersedia, jika tidak gunakan pembaca TFLite NumPy
    for nama_modul in ("tensorflow", "ai_edge_litert.interpreter", "tflite_runtime.interpreter"):
        try:
            if nama_modul == "tensorflow":
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=lokasi_tflite)
            elif nama_modul == "ai_edge_litert.interpreter":
                from ai_edge_litert.interpreter import Interpreter
                interpreter = Interpreter(model_path=lokasi_tflite)
            else:
                from tflite_runtime.interpreter import Interpreter
                interpreter = Interpreter(model_path=lokasi_tflite)
            interpreter.allocate_tensors()
            return ("interpreter", interpreter)
        except Exception:
            continue

    return ("numpy", PenjalanModelTFLite(lokasi_tflite))


def siapkan_gambar(gambar: Image.Image):
    """
    Menyiapkan foto sebelum dimasukkan ke model MobileNetV2:
    1. Memperbaiki posisi rotasi foto dari kamera HP
    2. Mengubah warna menjadi RGB dan ukuran menjadi 224 x 224 piksel
    3. Mengubah skala nilai piksel menjadi -1 sampai 1
    """
    gambar_rapi = ImageOps.exif_transpose(gambar)
    gambar_rgb = gambar_rapi.convert("RGB")
    gambar_224 = gambar_rgb.resize((224, 224))
    array_angka = np.asarray(gambar_224, dtype=np.float32)
    array_normal = (array_angka / 127.5) - 1.0
    input_model = np.expand_dims(array_normal, axis=0)
    return gambar_rgb, gambar_224, array_normal, input_model


def prediksi_ras_kucing(input_model: np.ndarray):
    model_aktif = muat_model()
    waktu_mulai = time.perf_counter()

    if model_aktif is None:
        raise FileNotFoundError("File model_kucing.tflite tidak ditemukan di folder aplikasi.")

    mode, objek_model = model_aktif
    if mode == "interpreter":
        info_input = objek_model.get_input_details()
        info_output = objek_model.get_output_details()
        objek_model.set_tensor(info_input[0]["index"], input_model.astype(info_input[0]["dtype"]))
        objek_model.invoke()
        skor_probabilitas = objek_model.get_tensor(info_output[0]["index"])[0]
    else:
        skor_probabilitas = objek_model.prediksi(input_model)

    durasi_ms = (time.perf_counter() - waktu_mulai) * 1000.0
    hasil = {DAFTAR_RAS[i]: float(skor_probabilitas[i]) for i in range(len(DAFTAR_RAS))}
    return hasil, durasi_ms


# Muat model di awal agar aplikasi siap digunakan
muat_model()


# ==================== 5. SIDEBAR (INFORMASI SINGKAT) ====================
with st.sidebar:
    st.title("Identifikasi Ras Kucing", anchor=False)
    st.caption("Aplikasi Pendeteksi Ras Kucing Menggunakan MobileNetV2")
    st.markdown("---")
    st.subheader("Informasi Model", anchor=False)
    st.metric("Model CNN", "MobileNetV2")
    st.metric("Akurasi Pengujian", "94,20%")
    st.metric("Akurasi Validasi", "94,08%")
    st.metric("Ukuran Gambar Input", "224 × 224 (RGB)")
    st.markdown("**Daftar 5 Ras Kucing:**")
    for indeks, nama_ras in enumerate(DAFTAR_RAS):
        st.markdown(f"`{indeks}` — **{nama_ras}**")


# ==================== 6. JUDUL & MENU TAB UTAMA ====================
st.markdown(
    """
    <div class="judul-utama">
        <h1>Identifikasi Ras Kucing</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_scan, tab_ensiklopedia, tab_ringkasan = st.tabs(
    [
        "Scan & Identifikasi Kucing",
        "Ensiklopedia 5 Ras Kucing",
        "Ringkasan Model",
    ]
)


# ==================== TAB 1: SCAN & IDENTIFIKASI ====================
with tab_scan:
    kolom_kiri, kolom_kanan = st.columns([1.05, 1.25], gap="large")

    gambar_terpilih = None
    nama_file_gambar = ""
    ukuran_file_kb = 0.0
    pesan_error = None

    with kolom_kiri:
        st.subheader("1. Pilih & Upload Foto Kucing", anchor=False)

        cara_input = st.radio(
            "Pilih Cara Memasukkan Foto:",
            options=[
                "Pilih File / Galeri HP",
                "Kamera Langsung",
            ],
            horizontal=True,
        )

        if cara_input == "Pilih File / Galeri HP":
            st.caption(
                "Tekan tombol **Browse files** untuk memilih foto kucing dari Galeri HP atau komputer."
            )
            file_diunggah = st.file_uploader(
                "Pilih Foto Kucing (JPG / PNG / WEBP / HEIC):",
                accept_multiple_files=False,
                key="upload_foto_kucing",
            )

            if file_diunggah is not None:
                try:
                    data_bytes = file_diunggah.getvalue()
                    nama_file_gambar = file_diunggah.name
                    ukuran_file_kb = len(data_bytes) / 1024.0
                    gambar_terpilih = Image.open(io.BytesIO(data_bytes))
                    gambar_terpilih.load()
                except Exception as e:
                    pesan_error = (
                        f"File '{file_diunggah.name}' tidak dapat dibaca sebagai gambar ({e}). "
                        "Silakan pilih file foto yang benar."
                    )

        else:
            st.caption(
                "Arahkan kamera ke kucing Anda lalu tekan tombol **Take Photo**."
            )
            foto_kamera = st.camera_input("Ambil Foto Kucing", key="kamera_kucing")
            if foto_kamera is not None:
                try:
                    data_bytes = foto_kamera.getvalue()
                    nama_file_gambar = "foto_kamera.jpg"
                    ukuran_file_kb = len(data_bytes) / 1024.0
                    gambar_terpilih = Image.open(io.BytesIO(data_bytes))
                    gambar_terpilih.load()
                except Exception as e:
                    pesan_error = f"Gagal membaca foto dari kamera: {e}"

        if pesan_error:
            st.error(pesan_error)

        # Jika file sudah dipilih, tampilkan nama file dan tombol Upload & Identifikasi
        if gambar_terpilih is not None:
            lebar_img, tinggi_img = gambar_terpilih.size
            id_file = f"{nama_file_gambar}_{len(data_bytes)}"
            sudah_diproses = st.session_state.get("id_file_terproses") == id_file

            st.markdown(
                f"""
                <div class="kotak-file-terpilih">
                    <div style="font-size:0.8rem; font-weight:700; color:#1d4ed8;">
                        LANGKAH 1 SELESAI — FILE GAMBAR TERPILIH
                    </div>
                    <div style="font-size:1.05rem; font-weight:800; margin-top:0.25rem; word-break:break-all;">
                        Nama Image: {nama_file_gambar}
                    </div>
                    <div style="font-size:0.85rem; margin-top:0.15rem; color:#334155;">
                        Ukuran: <b>{ukuran_file_kb:.1f} KB</b> • Resolusi: <b>{lebar_img} × {tinggi_img} px</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("Upload & Identifikasi Foto Ini Sekarang", type="primary"):
                st.session_state["id_file_terproses"] = id_file
                sudah_diproses = True

            st.image(
                ImageOps.exif_transpose(gambar_terpilih).convert("RGB"),
                caption=f"Nama Image: {nama_file_gambar} ({lebar_img}×{tinggi_img} px)",
                width=280,
            )
        else:
            sudah_diproses = False

    with kolom_kanan:
        st.subheader("2. Hasil Identifikasi Ras Kucing", anchor=False)

        if gambar_terpilih is None:
            st.info(
                "**Langkah 1 (Select File):** Klik tombol **Browse files** untuk memilih foto kucing.\n\n"
                "**Langkah 2 (Cek Nama Image):** Pastikan nama gambar muncul di layar.\n\n"
                "**Langkah 3 (Upload):** Klik tombol **Upload & Identifikasi Foto Ini Sekarang**."
            )
            with st.container(border=True):
                st.markdown("**5 Ras Kucing yang Dapat Dikenali:**")
                st.markdown(
                    "- **Bengal**\n"
                    "- **British Shorthair**\n"
                    "- **Domestic Short Hair** (Kucing Domestik / Kampung)\n"
                    "- **Persian** (Kucing Persia)\n"
                    "- **Turkish Angora** (Kucing Angora)"
                )

        elif not sudah_diproses:
            st.warning(
                f"File **`{nama_file_gambar}`** sudah dipilih.\n\n"
                "Silakan klik tombol **Upload & Identifikasi Foto Ini Sekarang** untuk melihat hasil deteksi ras kucing."
            )

        else:
            with st.spinner(f"Sedang mengenali foto '{nama_file_gambar}'..."):
                gambar_rgb, gambar_224, array_normal, input_model = siapkan_gambar(gambar_terpilih)
                hasil_prediksi, waktu_ms = prediksi_ras_kucing(input_model)
                urutan_prediksi = sorted(hasil_prediksi.items(), key=lambda x: x[1], reverse=True)
                ras_teratas, skor_teratas = urutan_prediksi[0]
                persen_teratas = skor_teratas * 100.0
                detail_ras = INFO_RAS_KUCING.get(ras_teratas, {})

            nama_tampil = detail_ras.get("nama_lengkap", ras_teratas)

            st.markdown(
                f"""
                <div class="kotak-hasil">
                    <div style="font-size:0.84rem; opacity:0.92; margin-bottom:0.2rem;">
                        Nama File: <b>{nama_file_gambar}</b>
                    </div>
                    <h2>{nama_tampil}</h2>
                    <div class="label-akurasi">Tingkat Keyakinan: {persen_teratas:.2f}%</div>
                    <div style="font-size:0.8rem; opacity:0.85; margin-top:0.45rem;">
                        Waktu Proses: {waktu_ms:.0f} ms
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if persen_teratas < 55.0:
                st.warning(
                    "Tingkat keyakinan di bawah 55%. Pastikan foto kucing terlihat jelas dan memiliki pencahayaan yang cukup."
                )

            st.markdown("**Persentase Kecocokan untuk 5 Ras Kucing:**")
            for nama_ras, nilai_skor in urutan_prediksi:
                persen = nilai_skor * 100.0
                st.progress(
                    min(max(int(round(persen)), 1), 100),
                    text=f"**{nama_ras}**: **{persen:.2f}%**",
                )

            if detail_ras:
                with st.container(border=True):
                    st.markdown(f"**Informasi & Cara Perawatan: {detail_ras['nama_lengkap']}**")
                    st.markdown(f"**Ciri Fisik:**\n{detail_ras['ciri_fisik']}")
                    kol_info1, kol_info2 = st.columns(2)
                    with kol_info1:
                        st.markdown(f"**Karakter:**\n{detail_ras['karakter']}")
                    with kol_info2:
                        st.markdown(f"**Perawatan:**\n{detail_ras['perawatan']}")
                    st.markdown(f"**Catatan Kesehatan:**\n{detail_ras['kesehatan']}")

            with st.expander("Lihat Detail Ukuran Gambar (224 × 224 RGB)"):
                p1, p2 = st.columns(2)
                with p1:
                    st.image(
                        gambar_224,
                        caption="Gambar Setelah Diubah ke 224 × 224 px",
                        width="stretch",
                    )
                with p2:
                    st.markdown("**Informasi Gambar Input:**")
                    st.code(
                        f"Nama File     : {nama_file_gambar}\n"
                        f"Ukuran Asli   : {gambar_rgb.size[0]} × {gambar_rgb.size[1]} px\n"
                        f"Mode Warna    : {gambar_rgb.mode}\n"
                        f"Ukuran Model  : 224 × 224 px\n"
                        f"Bentuk Data   : {input_model.shape}\n"
                        f"Nilai Minimum : {array_normal.min():.2f}\n"
                        f"Nilai Maksimum: {array_normal.max():.2f}"
                    )


# ==================== TAB 2: ENSIKLOPEDIA 5 RAS KUCING ====================
with tab_ensiklopedia:
    st.subheader("Informasi Lengkap 5 Ras Kucing", anchor=False)
    kolom_katalog = st.columns(2)

    for indeks, nama_ras in enumerate(DAFTAR_RAS):
        info = INFO_RAS_KUCING.get(nama_ras, {})
        with kolom_katalog[indeks % 2]:
            with st.container(border=True):
                # Menggunakan div biasa (bukan heading ###) agar tidak muncul ikon link di sebelah nama ras
                st.markdown(
                    f'<div class="judul-ras-katalog">{info.get("nama_lengkap", nama_ras)}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Nomor Kelas: {indeks} • Jumlah Data Latih: {info.get('jumlah_data', 0):,} foto"
                )
                st.markdown(f"- **Ciri Fisik:** {info.get('ciri_fisik', '-')}")
                st.markdown(f"- **Karakter:** {info.get('karakter', '-')}")
                st.markdown(f"- **Perawatan:** {info.get('perawatan', '-')}")
                st.markdown(f"- **Kesehatan:** {info.get('kesehatan', '-')}")


# ==================== TAB 3: RINGKASAN MODEL ====================
with tab_ringkasan:
    st.subheader("Ringkasan & Evaluasi Model", anchor=False)
    st.markdown(
        "Ringkasan jumlah data, pembagian data, susunan jaringan **MobileNetV2**, dan hasil akurasi pelatihan."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Data Awal (67 Kelas)", "126.607")
    m2.metric("Data Terpilih (5 Kelas)", "60.839")
    m3.metric("Akurasi Validasi", "94,08%")
    m4.metric("Akurasi Pengujian", "94,20%")

    kol_tabel1, kol_tabel2 = st.columns(2)
    with kol_tabel1:
        st.markdown("**1. Jumlah Foto pada 5 Ras Kucing**")
        tabel_distribusi = pd.DataFrame(
            {
                "Nomor Kelas": [0, 1, 2, 3, 4],
                "Nama Ras Kucing": [
                    "Bengal",
                    "British Shorthair",
                    "Domestic Short Hair",
                    "Persian",
                    "Turkish Angora",
                ],
                "Jumlah Foto": [2477, 567, 53027, 4018, 750],
            }
        )
        st.dataframe(tabel_distribusi, width="stretch", hide_index=True)

        st.markdown("**2. Pembagian Data (80% Latih, 10% Validasi, 10% Uji)**")
        tabel_pembagian = pd.DataFrame(
            {
                "Bagian Data": ["Data Latih (80%)", "Data Validasi (10%)", "Data Uji (10%)"],
                "Jumlah Foto": ["48.671 foto", "6.084 foto", "6.084 foto"],
            }
        )
        st.dataframe(tabel_pembagian, width="stretch", hide_index=True)

    with kol_tabel2:
        st.markdown("**3. Susunan Lapisan Model MobileNetV2**")
        tabel_arsitektur = pd.DataFrame(
            {
                "Lapisan (Layer)": [
                    "Input Gambar",
                    "Augmentasi Data",
                    "MobileNetV2 (Pre-trained)",
                    "GlobalAveragePooling2D",
                    "Dense (128 Neuron, ReLU)",
                    "Dropout (0.4)",
                    "Output (5 Kelas, Softmax)",
                ],
                "Ukuran Output": [
                    "(224, 224, 3)",
                    "(224, 224, 3)",
                    "(7, 7, 1280)",
                    "(1280)",
                    "(128)",
                    "(128)",
                    "(5)",
                ],
                "Jumlah Parameter": ["0", "0", "2.257.984", "0", "163.968", "0", "645"],
            }
        )
        st.dataframe(tabel_arsitektur, width="stretch", hide_index=True)

        st.markdown("**4. Hasil Pelatihan Model (10 Epoch)**")
        tabel_pelatihan = pd.DataFrame(
            {
                "Epoch": [1, 2, 4, 6, 8, 10],
                "Akurasi Latih": ["91,98%", "93,07%", "93,53%", "93,86%", "94,01%", "94,01%"],
                "Akurasi Validasi": ["93,69%", "93,92%", "94,10%", "93,92%", "94,05%", "94,08%"],
                "Loss Validasi": ["0,3737", "0,3225", "0,2798", "0,2615", "0,2493", "0,2416"],
            }
        )
        st.dataframe(tabel_pelatihan, width="stretch", hide_index=True)
