import io
import os
import socket
import struct
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

# ==================== KONFIGURASI HALAMAN STREAMLIT ====================
st.set_page_config(
    page_title="CatScan AI - Identifikasi Ras Kucing",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================== CSS RESPONSIF (DESKTOP & HP / MOBILE) ====================
st.markdown(
    """
    <style>
    /* Kontainer utama responsif untuk Desktop & HP */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }
    /* Header Banner */
    .hero-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: #ffffff;
        padding: 1.3rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
    }
    .hero-card h1 {
        font-size: 1.65rem;
        margin: 0 0 0.35rem 0;
        color: #ffffff !important;
    }
    .hero-card p {
        font-size: 0.95rem;
        margin: 0;
        opacity: 0.92;
        line-height: 1.45;
    }
    /* Kartu Hasil Prediksi Utama */
    .pred-badge {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
        color: white;
        padding: 1.2rem 1.4rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.25);
    }
    .pred-badge h2 {
        margin: 0;
        font-size: 1.5rem;
        color: #ffffff !important;
    }
    .pred-badge .conf-pill {
        display: inline-block;
        margin-top: 0.45rem;
        background: rgba(255,255,255,0.22);
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    /* Kartu Ringkas Langsung di Bawah Upload (Sangat Membantu di Layar HP) */
    .quick-detect-box {
        background: #f0fdf4;
        border: 2px solid #16a34a;
        color: #14532d;
        padding: 0.9rem 1.1rem;
        border-radius: 12px;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }
    /* Tombol ramah layar sentuh HP */
    .stButton > button, .stDownloadButton > button {
        width: 100%;
        border-radius: 10px;
        min-height: 2.8rem;
        font-weight: 600;
    }
    /* Penyesuaian khusus layar HP (Mobile) */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.6rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
        .hero-card {
            padding: 0.95rem 1.05rem;
            border-radius: 12px;
        }
        .hero-card h1 {
            font-size: 1.25rem;
        }
        .hero-card p {
            font-size: 0.85rem;
        }
        .pred-badge h2 {
            font-size: 1.25rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================== ENGINE INFERENSI TFLITE MURNI (TANPA DEPENDENSI C++) ====================
class NativeTFLiteRunner:

    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            self.buf = f.read()
        self._parse_and_compile()

    def _get_vtable(self, pos):
        vtable_offset = struct.unpack_from("<i", self.buf, pos)[0]
        vtable = pos - vtable_offset
        vsize, _ = struct.unpack_from("<HH", self.buf, vtable)
        return [
            struct.unpack_from("<H", self.buf, vtable + 4 + 2 * i)[0]
            for i in range((vsize - 4) // 2)
        ]

    def _get_table_vector(self, pos, field_idx):
        offsets = self._get_vtable(pos)
        if field_idx >= len(offsets) or offsets[field_idx] == 0:
            return []
        vec_pos = pos + offsets[field_idx]
        vec_pos += struct.unpack_from("<I", self.buf, vec_pos)[0]
        length = struct.unpack_from("<I", self.buf, vec_pos)[0]
        return [
            vec_pos + 4 + 4 * i + struct.unpack_from("<I", self.buf, vec_pos + 4 + 4 * i)[0]
            for i in range(length)
        ]

    def _get_int_vector(self, pos, field_idx):
        offsets = self._get_vtable(pos)
        if field_idx >= len(offsets) or offsets[field_idx] == 0:
            return []
        vec_pos = pos + offsets[field_idx]
        vec_pos += struct.unpack_from("<I", self.buf, vec_pos)[0]
        length = struct.unpack_from("<I", self.buf, vec_pos)[0]
        return list(struct.unpack_from(f"<{length}i", self.buf, vec_pos + 4))

    def _get_byte_slice(self, pos, field_idx):
        offsets = self._get_vtable(pos)
        if field_idx >= len(offsets) or offsets[field_idx] == 0:
            return b""
        vec_pos = pos + offsets[field_idx]
        vec_pos += struct.unpack_from("<I", self.buf, vec_pos)[0]
        length = struct.unpack_from("<I", self.buf, vec_pos)[0]
        return memoryview(self.buf)[vec_pos + 4 : vec_pos + 4 + length]

    def _get_scalar(self, pos, field_idx, fmt, default=0):
        offsets = self._get_vtable(pos)
        if field_idx >= len(offsets) or offsets[field_idx] == 0:
            return default
        return struct.unpack_from(fmt, self.buf, pos + offsets[field_idx])[0]

    def _get_table(self, pos, field_idx):
        offsets = self._get_vtable(pos)
        if field_idx >= len(offsets) or offsets[field_idx] == 0:
            return None
        t_pos = pos + offsets[field_idx]
        return t_pos + struct.unpack_from("<I", self.buf, t_pos)[0]

    def _parse_and_compile(self):
        root = struct.unpack_from("<I", self.buf, 0)[0]
        op_codes = self._get_table_vector(root, 1)
        buffers = self._get_table_vector(root, 4)
        sg = self._get_table_vector(root, 2)[0]
        tensor_tables = self._get_table_vector(sg, 0)
        self.inputs = self._get_int_vector(sg, 1)
        self.outputs = self._get_int_vector(sg, 2)
        operators = self._get_table_vector(sg, 3)

        dtype_map = {0: np.float32, 1: np.float16, 2: np.int32, 9: np.int8}
        self.tensors = [None] * len(tensor_tables)

        for idx, t in enumerate(tensor_tables):
            shape = tuple(self._get_int_vector(t, 0))
            ttype = self._get_scalar(t, 1, "<b", 0)
            buf_idx = self._get_scalar(t, 2, "<I", 0)
            raw = self._get_byte_slice(buffers[buf_idx], 0)
            if len(raw) > 0 and ttype in dtype_map:
                arr = np.frombuffer(raw, dtype=dtype_map[ttype]).reshape(shape)
                self.tensors[idx] = arr.astype(np.float32) if ttype == 1 else arr.copy()

        self.exec_ops = []
        for op in operators:
            opcode_idx = self._get_scalar(op, 0, "<I", 0)
            oc = op_codes[opcode_idx]
            builtin = self._get_scalar(oc, 3, "<i", self._get_scalar(oc, 0, "<b", 0))
            op_in = self._get_int_vector(op, 1)
            op_out = self._get_int_vector(op, 2)
            opt = self._get_table(op, 4)

            if builtin == 6:  # DEQUANTIZE
                self.tensors[op_out[0]] = self.tensors[op_in[0]].astype(np.float32)
            elif builtin == 3:  # CONV_2D
                pad = self._get_scalar(opt, 0, "<b", 0)
                sw = self._get_scalar(opt, 1, "<i", 1)
                sh = self._get_scalar(opt, 2, "<i", 1)
                act = self._get_scalar(opt, 3, "<b", 0)
                w = self.tensors[op_in[1]]
                b = self.tensors[op_in[2]]
                if w.shape[1] == 1 and w.shape[2] == 1 and sw == 1 and sh == 1:
                    w_mat = np.ascontiguousarray(w[:, 0, 0, :].T, dtype=np.float32)
                    self.exec_ops.append(("PW_CONV", op_in[0], op_out[0], w_mat, b, act))
                else:
                    self.exec_ops.append(("CONV_2D", op_in[0], op_out[0], w, b, pad, sw, sh, act))
            elif builtin == 4:  # DEPTHWISE_CONV_2D
                pad = self._get_scalar(opt, 0, "<b", 0)
                sw = self._get_scalar(opt, 1, "<i", 1)
                sh = self._get_scalar(opt, 2, "<i", 1)
                act = self._get_scalar(opt, 4, "<b", 0)
                w = self.tensors[op_in[1]][0]
                b = self.tensors[op_in[2]]
                self.exec_ops.append(("DW_CONV", op_in[0], op_out[0], w, b, pad, sw, sh, act))
            elif builtin == 0:  # ADD
                act = self._get_scalar(opt, 0, "<b", 0)
                self.exec_ops.append(("ADD", op_in[0], op_in[1], op_out[0], act))
            elif builtin == 40:  # MEAN
                axes = tuple(int(a) for a in self.tensors[op_in[1]].flatten())
                self.exec_ops.append(("MEAN", op_in[0], op_out[0], axes))
            elif builtin == 9:  # FULLY_CONNECTED
                act = self._get_scalar(opt, 0, "<b", 0) if opt else 0
                w_mat = np.ascontiguousarray(self.tensors[op_in[1]].T, dtype=np.float32)
                b = self.tensors[op_in[2]]
                self.exec_ops.append(("FC", op_in[0], op_out[0], w_mat, b, act))
            elif builtin == 25:  # SOFTMAX
                self.exec_ops.append(("SOFTMAX", op_in[0], op_out[0]))

    @staticmethod
    def _apply_act(x, act):
        if act == 1:  # RELU
            np.maximum(x, 0.0, out=x)
        elif act == 3:  # RELU6
            np.clip(x, 0.0, 6.0, out=x)
        return x

    @staticmethod
    def _pad_same(x, kh, kw, sh, sw):
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

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        t = list(self.tensors)
        t[self.inputs[0]] = np.ascontiguousarray(input_tensor, dtype=np.float32)

        for op in self.exec_ops:
            kind = op[0]
            if kind == "PW_CONV":
                _, in_idx, out_idx, w_mat, b, act = op
                x = t[in_idx]
                n, h, w_dim, c_in = x.shape
                y = (x.reshape(-1, c_in) @ w_mat).reshape(n, h, w_dim, -1) + b
                t[out_idx] = self._apply_act(y, act)
            elif kind == "CONV_2D":
                _, in_idx, out_idx, w, b, pad, sw, sh, act = op
                x = t[in_idx]
                out_c, kh, kw, _ = w.shape
                if pad == 0:
                    x, out_h, out_w = self._pad_same(x, kh, kw, sh, sw)
                else:
                    out_h = (x.shape[1] - kh) // sh + 1
                    out_w = (x.shape[2] - kw) // sw + 1
                y = np.broadcast_to(b, (x.shape[0], out_h, out_w, out_c)).copy()
                for ky in range(kh):
                    for kx in range(kw):
                        patch = x[:, ky : ky + out_h * sh : sh, kx : kx + out_w * sw : sw, :]
                        y += patch @ w[:, ky, kx, :].T
                t[out_idx] = self._apply_act(y, act)
            elif kind == "DW_CONV":
                _, in_idx, out_idx, w, b, pad, sw, sh, act = op
                x = t[in_idx]
                kh, kw, c = w.shape
                if pad == 0:
                    x, out_h, out_w = self._pad_same(x, kh, kw, sh, sw)
                else:
                    out_h = (x.shape[1] - kh) // sh + 1
                    out_w = (x.shape[2] - kw) // sw + 1
                y = np.broadcast_to(b, (x.shape[0], out_h, out_w, c)).copy()
                for ky in range(kh):
                    for kx in range(kw):
                        patch = x[:, ky : ky + out_h * sh : sh, kx : kx + out_w * sw : sw, :]
                        y += patch * w[ky, kx, :]
                t[out_idx] = self._apply_act(y, act)
            elif kind == "ADD":
                _, in1, in2, out_idx, act = op
                y = t[in1] + t[in2]
                t[out_idx] = self._apply_act(y, act)
            elif kind == "MEAN":
                _, in_idx, out_idx, axes = op
                t[out_idx] = np.mean(t[in_idx], axis=axes)
            elif kind == "FC":
                _, in_idx, out_idx, w_mat, b, act = op
                y = t[in_idx] @ w_mat + b
                t[out_idx] = self._apply_act(y, act)
            elif kind == "SOFTMAX":
                _, in_idx, out_idx = op
                x = t[in_idx]
                e = np.exp(x - np.max(x, axis=-1, keepdims=True))
                t[out_idx] = e / np.sum(e, axis=-1, keepdims=True)

        return t[self.outputs[0]][0]


# ==================== BASIS PENGETAHUAN 5 RAS KUCING (SESUAI TUBES_PCD.IPYNB) ====================
BREED_KNOWLEDGE_BASE = {
    "Bengal": {
        "nama_lengkap": "Kucing Bengal (Bengal Cat)",
        "nama_populer": "Bengal / Macan Tutul Mini",
        "jumlah_data_notebook": 2477,
        "ikon": "🐆",
        "ciri_fisik": (
            "Tubuh atletis, berotot, dan ramping dengan corak bulu tutul (spotted/rosetted) atau marmer (marbled) "
            "yang eksotis menyerupai macan tutul. Bulu pendek, sangat rapat, dan berkilau (glitter coat)."
        ),
        "karakter": "Sangat energik, cerdas, suka memanjat tempat tinggi, penasaran, dan sering menyukai air.",
        "perawatan": (
            "Sediakan ruang bermain vertikal (cat tree/cat wheel), ajak bermain aktif setiap hari, "
            "dan sikat bulu 1 kali seminggu untuk menjaga kilau alami."
        ),
        "estimasi_harga": "Rp 3.500.000 – Rp 15.000.000+",
        "kesehatan": "Perhatikan skrining Hypertrophic Cardiomyopathy (HCM) dan Progressive Retinal Atrophy (PRA-b).",
    },
    "British Shorthair": {
        "nama_lengkap": "Kucing British Shorthair (BSH)",
        "nama_populer": "British Shorthair",
        "jumlah_data_notebook": 567,
        "ikon": "🐻",
        "ciri_fisik": (
            "Postur tubuh bulat dan kekar (cobby), dada bidang, pipi tembem (chubby cheeks), mata bulat besar "
            "(sering berwarna tembaga/emas), serta bulu pendek yang sangat padat dan empuk seperti boneka plush."
        ),
        "karakter": "Tenang, santai, mandiri, penyayang namun tidak terlalu menuntut digendong terus-menerus.",
        "perawatan": (
            "Kontrol porsi makan untuk mencegah obesitas karena sifatnya yang santai, "
            "serta sisir bulu 2–3 kali seminggu terutama saat musim rontok."
        ),
        "estimasi_harga": "Rp 4.000.000 – Rp 18.000.000+",
        "kesehatan": "Pantau berat badan ideal, pemeriksaan jantung (HCM), dan kesehatan gusi/gigi.",
    },
    "Domestic Short Hair": {
        "nama_lengkap": "Kucing Domestik / Kampung (Domestic Short Hair)",
        "nama_populer": "Domestic Short Hair (Kucing Domestik / Kampung)",
        "jumlah_data_notebook": 53027,
        "ikon": "🐈",
        "ciri_fisik": (
            "Postur tubuh proporsional, lincah, dan tangguh dengan bulu pendek rapat. Memiliki variasi warna "
            "dan pola bulu yang sangat kaya (tabby/belang, oyen/ginger, calico/belang tiga, tuxedo, putih/hitam solid)."
        ),
        "karakter": "Adaptif, cerdas, mandiri, memiliki daya tahan tubuh kuat, dan setia pada pemiliknya.",
        "perawatan": (
            "Perawatan bulu sangat praktis (cukup disisir seminggu sekali), berikan pakan bergizi seimbang, "
            "obat cacing/kutu berkala, vaksinasi dasar, dan sterilisasi."
        ),
        "estimasi_harga": "Adopsi Kasih / Rp 150.000 – Rp 500.000 (Pengganti Biaya Vaksin & Steril)",
        "kesehatan": "Daya tahan genetik sangat prima (hybrid vigor); utamakan pencegahan parasit dan vaksin FVRCP.",
    },
    "Persian": {
        "nama_lengkap": "Kucing Persia (Persian Cat)",
        "nama_populer": "Persian (Kucing Persia)",
        "jumlah_data_notebook": 4018,
        "ikon": "👑",
        "ciri_fisik": (
            "Wajah bulat dan datar (flat-face/peke-face atau doll-face), hidung pesek, mata bulat besar, "
            "telinga kecil berjauhan, tubuh pendek gempal dengan bulu sangat panjang dan lebat."
        ),
        "karakter": "Sangat kalem, manja, lembut, menyukai lingkungan rumah yang tenang dan sejuk.",
        "perawatan": (
            "Wajib disisir setiap hari (10–15 menit) agar bulu tidak menggumpal (matting), bersihkan area air mata "
            "setiap pagi, dan pelihara di dalam ruangan (indoor) yang sejuk."
        ),
        "estimasi_harga": "Rp 2.000.000 – Rp 8.500.000",
        "kesehatan": "Rentan terhadap Polycystic Kidney Disease (PKD), penyempitan lubang hidung, dan jamur kulit bila lembap.",
    },
    "Turkish Angora": {
        "nama_lengkap": "Kucing Angora Turki (Turkish Angora)",
        "nama_populer": "Turkish Angora (Kucing Angora)",
        "jumlah_data_notebook": 750,
        "ikon": "🦢",
        "ciri_fisik": (
            "Tubuh ramping, anggun, dan lentur dengan bulu semi-panjang sehalus sutra (single coat tanpa lapisan bawah tebal). "
            "Wajah agak meruncing (wedge-shaped), telinga besar tegak, dan ekor mengembang indah."
        ),
        "karakter": "Aktif, cerdas, lincah, ekspresif, dan senang berinteraksi dengan anggota keluarga.",
        "perawatan": (
            "Sisir bulu 2–3 kali seminggu (tidak mudah kusut karena single coat), sediakan mainan interaktif, "
            "dan jaga kebersihan telinga serta mata."
        ),
        "estimasi_harga": "Rp 1.500.000 – Rp 4.500.000",
        "kesehatan": "Perhatikan uji pendengaran pada Angora putih bermata biru serta kesehatan jantung (HCM).",
    },
}


# ==================== FUNGSI UTILITAS & PEMUATAN MODEL ====================
def get_local_ip() -> str:
    """Mendapatkan alamat IP lokal agar aplikasi bisa dibuka dari HP di jaringan Wi-Fi yang sama."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


@st.cache_data
def load_labels(label_file: str = "labels_kucing.txt") -> list[str]:
    """Memuat daftar kelas dari labels_kucing.txt yang dihasilkan oleh tubes_pcd.ipynb."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, label_file)
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        if labels:
            return labels
    return [
        "Bengal",
        "British Shorthair",
        "Domestic Short Hair",
        "Persian",
        "Turkish Angora",
    ]


@st.cache_resource(show_spinner="Memuat model MobileNetV2 (model_kucing.tflite)...")
def load_inference_engine():
    """
    Memuat model_kucing.tflite dari direktori proyek.
    Prioritas:
    1. tensorflow.lite.Interpreter / ai_edge_litert / tflite_runtime (jika tersedia)
    2. NativeTFLiteRunner (menjalankan bobot asli model_kucing.tflite langsung dengan NumPy)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tflite_path = os.path.join(base_dir, "model_kucing.tflite")

    if not os.path.exists(tflite_path):
        return None, "File model_kucing.tflite tidak ditemukan"

    for mod_name in ("tensorflow", "ai_edge_litert.interpreter", "tflite_runtime.interpreter"):
        try:
            if mod_name == "tensorflow":
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=tflite_path)
            elif mod_name == "ai_edge_litert.interpreter":
                from ai_edge_litert.interpreter import Interpreter
                interpreter = Interpreter(model_path=tflite_path)
            else:
                from tflite_runtime.interpreter import Interpreter
                interpreter = Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            return ("tflite_c", interpreter), f"TFLite Interpreter ({mod_name} - model_kucing.tflite)"
        except Exception:
            continue

    runner = NativeTFLiteRunner(tflite_path)
    return ("native_tflite", runner), "Native TFLite Engine (model_kucing.tflite - MobileNetV2 Float16)"


def preprocess_image_for_mobilenetv2(image: Image.Image):
    """
    Tahapan Preprocessing sesuai Bagian B pada tubes_pcd.ipynb:
    1. Koreksi orientasi EXIF (penting untuk foto kamera HP) & konversi ke RGB (3 channel)
    2. Resize citra ke ukuran (224, 224)
    3. Normalisasi piksel dengan preprocess_input MobileNetV2: (x / 127.5) - 1.0 -> rentang [-1.0, 1.0]
    """
    img_fixed = ImageOps.exif_transpose(image)
    img_rgb = img_fixed.convert("RGB")
    img_resized = img_rgb.resize((224, 224))
    arr_float = np.asarray(img_resized, dtype=np.float32)
    arr_normalized = (arr_float / 127.5) - 1.0
    input_tensor = np.expand_dims(arr_normalized, axis=0)
    return img_rgb, img_resized, arr_normalized, input_tensor


def predict_breed(input_tensor: np.ndarray, labels: list[str]):
    """Menjalankan prediksi ras kucing menggunakan model_kucing.tflite."""
    engine_obj, engine_name = load_inference_engine()
    t0 = time.perf_counter()

    if engine_obj is None:
        raise FileNotFoundError("File model_kucing.tflite tidak ditemukan di folder aplikasi.")

    mode, runner = engine_obj
    if mode == "tflite_c":
        in_details = runner.get_input_details()
        out_details = runner.get_output_details()
        runner.set_tensor(in_details[0]["index"], input_tensor.astype(in_details[0]["dtype"]))
        runner.invoke()
        probs = runner.get_tensor(out_details[0]["index"])[0]
    else:
        probs = runner.predict(input_tensor)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    result = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return result, engine_name, elapsed_ms


@st.cache_data
def get_notebook_sample_images() -> dict[str, bytes]:
    """Mengekstrak contoh citra kucing asli dari output Cell 15 di tubes_pcd.ipynb sebagai bytes PNG."""
    samples = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nb_path = os.path.join(base_dir, "tubes_pcd.ipynb")
    if os.path.exists(nb_path):
        try:
            import json
            import base64

            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            b64 = nb["cells"][15]["outputs"][0]["data"]["image/png"]
            grid = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            w, h = grid.size
            cw, ch = w / 5.0, h / 2.0
            crops = {
                "Contoh 1 - Domestic Short Hair (Tabby)": grid.crop(
                    (int(2 * cw + 12), int(32), int(3 * cw - 12), int(ch - 8))
                ),
                "Contoh 2 - Domestic Short Hair (Grey & White)": grid.crop(
                    (int(3 * cw + 12), int(32), int(4 * cw - 12), int(ch - 8))
                ),
                "Contoh 3 - Persian Cat": grid.crop(
                    (int(4 * cw + 10), int(ch + 85), int(5 * cw - 10), int(h - 60))
                ),
                "Contoh 4 - Domestic Short Hair (Orange)": grid.crop(
                    (int(1 * cw + 10), int(55), int(2 * cw - 10), int(ch - 55))
                ),
            }
            for k, im in crops.items():
                bio = io.BytesIO()
                im.save(bio, format="PNG")
                samples[k] = bio.getvalue()
        except Exception:
            pass
    return samples


# ==================== STATE MANAGEMENT UNTUK INPUT GALERI HP / KAMERA ====================
if "active_source" not in st.session_state:
    st.session_state["active_source"] = "upload"


def _on_upload_change():
    if st.session_state.get("uploader_kucing") is not None:
        st.session_state["active_source"] = "upload"


def _on_camera_change():
    if st.session_state.get("cam_kucing") is not None:
        st.session_state["active_source"] = "camera"


def _on_sample_change():
    st.session_state["active_source"] = "sample"


# ==================== PRE-LOAD MODEL AGAR SAAT UPLOAD DARI HP LANGSUNG CEPAT ====================
labels_list = load_labels()
local_ip = get_local_ip()
load_inference_engine()

# ==================== SIDEBAR: INFO MODEL & AKSES HP ====================
with st.sidebar:
    st.title("🐾 CatScan AI")
    st.caption("Aplikasi Identifikasi Ras Kucing berbasis MobileNetV2 (`tubes_pcd.ipynb`)")
    st.markdown("---")

    st.subheader("📱 Buka di HP (Satu Wi-Fi)")
    st.markdown(
        f"Jika dijalankan di laptop/PC, buka browser di HP Anda yang terhubung ke Wi-Fi yang sama lalu ketik:\n"
        f"**`http://{local_ip}:8501`**"
    )
    st.caption(
        "Tip HP: Gunakan tombol **Upload / Galeri HP** untuk memilih foto dari Galeri atau memotret langsung dengan Kamera HP."
    )

    st.markdown("---")
    st.subheader("📊 Spesifikasi Model (`tubes_pcd.ipynb`)")
    st.metric("Arsitektur CNN", "MobileNetV2")
    st.metric("Akurasi Testing", "94,20%", "Loss: 0,2385")
    st.metric("Akurasi Validasi", "94,08%", "Val Loss: 0,2416")
    st.metric("Input Tensor", "224 × 224 × 3 (RGB)")
    st.markdown("**5 Kelas Ras Kucing (`labels_kucing.txt`):**")
    for idx, lbl in enumerate(labels_list):
        ikon = BREED_KNOWLEDGE_BASE.get(lbl, {}).get("ikon", "🐱")
        st.markdown(f"`{idx}` — {ikon} **{lbl}**")


# ==================== HEADER UTAMA ====================
st.markdown(
    """
    <div class="hero-card">
        <h1>🐱 Identifikasi Ras Kucing AI (Scan Kamera & Upload Galeri HP)</h1>
        <p>
            Pilih foto kucing dari <b>Galeri HP / Komputer</b> atau gunakan <b>Scan Kamera</b>.
            Begitu foto dipilih, sistem <b>otomatis langsung mendeteksi</b> ras kucing menggunakan model <code>model_kucing.tflite</code>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================== NAVIGASI UTAMA ====================
main_tab_scan, main_tab_katalog, main_tab_model = st.tabs(
    [
        "📸 Scan & Identifikasi Kucing",
        "📘 Ensiklopedia 5 Ras Kucing",
        "📊 Ringkasan Model (tubes_pcd.ipynb)",
    ]
)

# ==================== TAB 1: SCAN & IDENTIFIKASI ====================
with main_tab_scan:
    col_left, col_right = st.columns([1.0, 1.25], gap="large")

    selected_image = None
    source_caption = ""
    upload_error_msg = None

    with col_left:
        st.subheader("1️⃣ Masukkan Foto Kucing")
        mode_tab_upload, mode_tab_camera, mode_tab_sample = st.tabs(
            [
                "📤 Upload / Galeri HP",
                "📷 Scan Kamera",
            ]
        )

        with mode_tab_upload:
            st.caption(
                "Pilih foto kucing dari **Galeri HP**, **Kamera HP**, atau folder komputer Anda (mendukung JPG, JPEG, PNG, WEBP, JFIF, BMP)."
            )
            uploaded_file = st.file_uploader(
                "Pilih foto kucing dari Galeri HP / Komputer:",
                type=["jpg", "jpeg", "png", "webp", "jfif", "bmp", "heic", "heif"],
                key="uploader_kucing",
                on_change=_on_upload_change,
            )

        with mode_tab_camera:
            st.caption(
                "Arahkan kamera HP atau webcam laptop ke kucing Anda lalu tekan tombol **Take Photo**."
            )
            enable_cam = st.checkbox("Aktifkan Kamera Live", value=False)
            camera_photo = None
            if enable_cam:
                camera_photo = st.camera_input(
                    "Ambil Foto Kucing",
                    key="cam_kucing",
                    on_change=_on_camera_change,
                )

        with mode_tab_sample:
            st.caption("Uji model secara instan menggunakan sampel citra kucing dari notebook:")
            sample_dict = get_notebook_sample_images()
            if sample_dict:
                sample_name = st.selectbox(
                    "Pilih Contoh Citra:",
                    options=list(sample_dict.keys()),
                    key="sample_selector",
                    on_change=_on_sample_change,
                )
                if st.button("🐱 Gunakan Contoh Foto Ini", type="primary"):
                    st.session_state["active_source"] = "sample"

        # Tentukan gambar yang aktif berdasarkan aksi terakhir pengguna
        active_src = st.session_state.get("active_source", "upload")

        if active_src == "upload" and uploaded_file is not None:
            try:
                raw_bytes = uploaded_file.getvalue()
                selected_image = Image.open(io.BytesIO(raw_bytes))
                selected_image.load()
                source_caption = f"Galeri / Upload: {uploaded_file.name}"
            except Exception as e:
                upload_error_msg = f"Gagal membaca format gambar ({e}). Pastikan format foto berupa JPG, PNG, atau WEBP."
        elif active_src == "camera" and camera_photo is not None:
            try:
                raw_bytes = camera_photo.getvalue()
                selected_image = Image.open(io.BytesIO(raw_bytes))
                selected_image.load()
                source_caption = "Hasil Scan Kamera Langsung"
            except Exception as e:
                upload_error_msg = f"Gagal membaca foto kamera: {e}"
        elif active_src == "sample" and sample_dict:
            chosen_sample = st.session_state.get("sample_selector", list(sample_dict.keys())[0])
            selected_image = Image.open(io.BytesIO(sample_dict[chosen_sample]))
            source_caption = chosen_sample
        elif uploaded_file is not None:
            try:
                raw_bytes = uploaded_file.getvalue()
                selected_image = Image.open(io.BytesIO(raw_bytes))
                selected_image.load()
                source_caption = f"Galeri / Upload: {uploaded_file.name}"
            except Exception as e:
                upload_error_msg = f"Gagal membaca format gambar ({e})."
        elif camera_photo is not None:
            raw_bytes = camera_photo.getvalue()
            selected_image = Image.open(io.BytesIO(raw_bytes))
            source_caption = "Hasil Scan Kamera Langsung"

        if upload_error_msg:
            st.error(upload_error_msg)

    # Jalankan Preprocessing & Inferensi Begitu Gambar Tersedia
    prediction_data = None
    if selected_image is not None:
        with st.spinner("🔍 Sedang menganalisis & mendeteksi ras kucing..."):
            img_rgb, img_resized, arr_norm, input_tensor = preprocess_image_for_mobilenetv2(
                selected_image
            )
            probs_dict, engine_label, infer_ms = predict_breed(input_tensor, labels_list)
            sorted_preds = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
            top_breed, top_prob = sorted_preds[0]
            top_pct = top_prob * 100.0
            breed_info = BREED_KNOWLEDGE_BASE.get(top_breed, {})
            prediction_data = {
                "img_rgb": img_rgb,
                "img_resized": img_resized,
                "arr_norm": arr_norm,
                "input_tensor": input_tensor,
                "sorted_preds": sorted_preds,
                "top_breed": top_breed,
                "top_pct": top_pct,
                "breed_info": breed_info,
                "engine_label": engine_label,
                "infer_ms": infer_ms,
            }

    # Tampilkan Pratinjau & Notifikasi Hasil Cepat di Kolom Kiri (Agar di Layar HP Langsung Terlihat Tanpa Scroll!)
    with col_left:
        if prediction_data is not None:
            b_info = prediction_data["breed_info"]
            b_ikon = b_info.get("ikon", "🐱")
            b_nama = b_info.get("nama_lengkap", prediction_data["top_breed"])
            st.markdown(
                f"""
                <div class="quick-detect-box">
                    <div style="font-size:0.82rem; font-weight:700; text-transform:uppercase; color:#15803d;">
                        ✅ Berhasil Terdeteksi ({prediction_data['infer_ms']:.0f} ms)
                    </div>
                    <div style="font-size:1.2rem; font-weight:800; margin-top:0.15rem;">
                        {b_ikon} {b_nama}
                    </div>
                    <div style="font-size:0.92rem; margin-top:0.15rem;">
                        Tingkat Keyakinan (Confidence): <b>{prediction_data['top_pct']:.2f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.image(
                prediction_data["img_rgb"],
                caption=f"📷 {source_caption}",
                width="stretch",
            )

            with st.expander("🔍 Detail Tahapan Preprocessing (Sesuai tubes_pcd.ipynb)"):
                p1, p2 = st.columns(2)
                with p1:
                    st.image(
                        prediction_data["img_resized"],
                        caption="Hasil Resize: 224 × 224 px (RGB)",
                        width="stretch",
                    )
                with p2:
                    st.markdown("**Parameter Tensor Input:**")
                    st.code(
                        f"Resolusi Asli : {prediction_data['img_rgb'].size[0]} × {prediction_data['img_rgb'].size[1]} px\n"
                        f"Mode Warna    : {prediction_data['img_rgb'].mode} (3 Channel)\n"
                        f"Ukuran Resize : 224 × 224 px\n"
                        f"Bentuk Tensor : {prediction_data['input_tensor'].shape}\n"
                        f"Nilai Minimum : {prediction_data['arr_norm'].min():.3f}\n"
                        f"Nilai Maksimum: {prediction_data['arr_norm'].max():.3f}\n"
                        f"Normalisasi   : MobileNetV2 [-1, 1]"
                    )

    # Tampilkan Detail Lengkap di Kolom Kanan
    with col_right:
        st.subheader("2️⃣ Detail Hasil Identifikasi Ras Kucing")
        if prediction_data is None:
            st.info(
                "👈 Silakan pilih foto kucing pada tombol **Browse files** di tab **Upload / Galeri HP**, "
                "gunakan **Scan Kamera**, atau klik **Gunakan Contoh Foto Ini**."
            )
            with st.container(border=True):
                st.markdown("#### 💡 Tips Upload dari Galeri HP")
                st.markdown(
                    "- Tekan tombol **Browse files** pada tab **Upload / Galeri HP**, lalu pilih foto kucing dari **Galeri** atau pilih **Kamera**.\n"
                    "- Tunggu hingga bar loading file selesai 100% — hasil ras kucing akan **otomatis langsung muncul tepat di bawah tombol upload** maupun pada panel detail ini.\n"
                    "- Model mengenali **5 ras kucing**: **Bengal**, **British Shorthair**, **Domestic Short Hair (Kampung/Domestik)**, **Persian**, dan **Turkish Angora**."
                )
        else:
            breed_info = prediction_data["breed_info"]
            ikon = breed_info.get("ikon", "🐱")
            nama_lengkap = breed_info.get("nama_lengkap", prediction_data["top_breed"])
            top_pct = prediction_data["top_pct"]

            st.markdown(
                f"""
                <div class="pred-badge">
                    <div style="font-size:0.88rem; opacity:0.9; margin-bottom:0.2rem;">HASIL IDENTIFIKASI RAS KUCING</div>
                    <h2>{ikon} {nama_lengkap}</h2>
                    <div class="conf-pill">🎯 Tingkat Keyakinan (Confidence): {top_pct:.2f}%</div>
                    <div style="font-size:0.8rem; opacity:0.85; margin-top:0.45rem;">
                        Engine: {prediction_data['engine_label']} • Waktu Inferensi: {prediction_data['infer_ms']:.0f} ms
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if top_pct < 55.0:
                st.warning(
                    "⚠️ Tingkat keyakinan di bawah 55%. Pastikan objek kucing terlihat jelas, dekat, dan memiliki pencahayaan yang baik."
                )

            st.markdown("##### 📊 Persentase Probabilitas 5 Kelas (`Softmax Output`)")
            for breed_name, prob_val in prediction_data["sorted_preds"]:
                pct = prob_val * 100.0
                b_icon = BREED_KNOWLEDGE_BASE.get(breed_name, {}).get("ikon", "🐱")
                st.progress(
                    min(max(int(round(pct)), 1), 100),
                    text=f"{b_icon} **{breed_name}**: **{pct:.2f}%**",
                )

            if breed_info:
                with st.container(border=True):
                    st.markdown(
                        f"#### {ikon} Profil & Panduan Perawatan: **{breed_info['nama_populer']}**"
                    )
                    st.markdown(f"**🔎 Ciri Fisik Utama:**\n{breed_info['ciri_fisik']}")
                    c_info1, c_info2 = st.columns(2)
                    with c_info1:
                        st.markdown(f"**💛 Karakter & Sifat:**\n{breed_info['karakter']}")
                        st.markdown(
                            f"**💰 Estimasi Harga / Adopsi:**\n`{breed_info['estimasi_harga']}`"
                        )
                    with c_info2:
                        st.markdown(f"**🧼 Panduan Perawatan:**\n{breed_info['perawatan']}")
                        st.markdown(f"**🩺 Catatan Kesehatan:**\n{breed_info['kesehatan']}")


# ==================== TAB 2: ENSIKLOPEDIA 5 RAS KUCING ====================
with main_tab_katalog:
    st.subheader("📘 Informasi Lengkap 5 Kelas Ras Kucing")
    st.caption(
        "Daftar 5 ras kucing yang diklasifikasikan oleh model sesuai `labels_kucing.txt` pada `tubes_pcd.ipynb`."
    )
    cols_kat = st.columns(2)
    for idx, breed_key in enumerate(labels_list):
        info = BREED_KNOWLEDGE_BASE.get(breed_key, {})
        with cols_kat[idx % 2]:
            with st.container(border=True):
                st.markdown(f"### {info.get('ikon', '🐱')} {info.get('nama_lengkap', breed_key)}")
                st.caption(
                    f"Indeks Kelas: `{idx}` • Jumlah Citra pada Dataset Terpilih: `{info.get('jumlah_data_notebook', 0):,} citra`"
                )
                st.markdown(f"- **Ciri Fisik:** {info.get('ciri_fisik', '-')}")
                st.markdown(f"- **Karakter:** {info.get('karakter', '-')}")
                st.markdown(f"- **Perawatan:** {info.get('perawatan', '-')}")
                st.markdown(f"- **Catatan Medis:** {info.get('kesehatan', '-')}")


# ==================== TAB 3: RINGKASAN MODEL (TUBES_PCD.IPYNB) ====================
with main_tab_model:
    st.subheader("📊 Dokumentasi & Evaluasi Model (`tubes_pcd.ipynb`)")
    st.markdown(
        "Ringkasan tahapan pemilihan data, *preprocessing*, arsitektur **MobileNetV2**, dan evaluasi dari notebook [`tubes_pcd.ipynb`]."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Citra Awal (67 Kelas)", "126.607")
    m2.metric("Total Citra Terpilih (5 Kelas)", "60.839")
    m3.metric("Akurasi Validasi (Epoch 10)", "94,08%")
    m4.metric("Akurasi Testing (6.084 Citra)", "94,20%")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("##### 1. Distribusi Citra pada 5 Kelas Terpilih")
        df_dist = pd.DataFrame(
            {
                "Indeks Label": [0, 1, 2, 3, 4],
                "Nama Kelas (labels_kucing.txt)": [
                    "Bengal",
                    "British Shorthair",
                    "Domestic Short Hair",
                    "Persian",
                    "Turkish Angora",
                ],
                "Jumlah Citra Valid": [2477, 567, 53027, 4018, 750],
            }
        )
        st.dataframe(df_dist, width="stretch", hide_index=True)

        st.markdown("##### 2. Pembagian Dataset (`Stratified Split 80 : 10 : 10`)")
        df_split = pd.DataFrame(
            {
                "Subset Data": ["Data Training (80%)", "Data Validation (10%)", "Data Testing (10%)"],
                "Jumlah Citra": ["48.671 citra", "6.084 citra", "6.084 citra"],
            }
        )
        st.dataframe(df_split, width="stretch", hide_index=True)

    with col_d2:
        st.markdown("##### 3. Arsitektur CNN MobileNetV2 (`model_kucing.tflite`)")
        df_arch = pd.DataFrame(
            {
                "Layer": [
                    "Input Layer",
                    "Data Augmentation",
                    "MobileNetV2 (ImageNet, frozen)",
                    "GlobalAveragePooling2D",
                    "Dense (ReLU, L2=0.001)",
                    "Dropout (0.4)",
                    "Dense Output (Softmax)",
                ],
                "Output Shape": [
                    "(None, 224, 224, 3)",
                    "(None, 224, 224, 3)",
                    "(None, 7, 7, 1280)",
                    "(None, 1280)",
                    "(None, 128)",
                    "(None, 128)",
                    "(None, 5)",
                ],
                "Parameter": ["0", "0", "2.257.984", "0", "163.968", "0", "645"],
            }
        )
        st.dataframe(df_arch, width="stretch", hide_index=True)

        st.markdown("##### 4. Riwayat Pelatihan 10 Epoch (`Adam lr=1e-4`)")
        df_hist = pd.DataFrame(
            {
                "Epoch": [1, 2, 4, 6, 8, 10],
                "Train Accuracy": ["91,98%", "93,07%", "93,53%", "93,86%", "94,01%", "94,01%"],
                "Val Accuracy": ["93,69%", "93,92%", "94,10%", "93,92%", "94,05%", "94,08%"],
                "Val Loss": ["0,3737", "0,3225", "0,2798", "0,2615", "0,2493", "0,2416"],
            }
        )
        st.dataframe(df_hist, width="stretch", hide_index=True)
