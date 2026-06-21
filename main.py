import re
import shutil
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_FILES
from cover_processor import CoverProcessor

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

proc = CoverProcessor()

# ── 색상 팔레트 ──────────────────────────────────
ACCENT      = "#a78bfa"   # 보라 계열 (vocal-separator의 파랑과 구분)
SUCCESS     = "#2e7d32"
CARD_BG     = "#1e1e2e"
LOG_BG      = "#0d0d0d"
LOG_FG      = "#d4b8ff"
GRAY        = "#6b7280"
BORDER      = "#2a2a3e"


# ── CTk + tkinterdnd2 결합 ───────────────────────
class App(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)
        self.title("AI Cover Maker")
        self.geometry("860x820")
        self.minsize(720, 680)
        self.configure(fg_color="#13131f")
        self.busy = False
        self._build()

    # ─────────────────────────────────────────────
    # 레이아웃
    # ─────────────────────────────────────────────
    def _build(self):
        self._header()
        self.tabs = ctk.CTkTabview(self, fg_color=CARD_BG, corner_radius=12,
                                   segmented_button_selected_color="#4a1a7a",
                                   segmented_button_selected_hover_color="#6a2a9a")
        self.tabs.pack(fill="both", expand=True, padx=20, pady=(10, 0))

        for name in ("📚  모델 학습", "🎤  목소리 변환", "🎵  AI 커버 생성", "📖  사용법"):
            self.tabs.add(name)

        self._tab_train(self.tabs.tab("📚  모델 학습"))
        self._tab_convert(self.tabs.tab("🎤  목소리 변환"))
        self._tab_cover(self.tabs.tab("🎵  AI 커버 생성"))
        self._tab_help(self.tabs.tab("📖  사용법"))
        self._storage_bar()
        self._log_box()

    def _header(self):
        hdr = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=0, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="  🎤  AI Cover Maker",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=ACCENT).pack(side="left", padx=16)
        ctk.CTkLabel(hdr, text="RVC 기반 AI 커버곡 생성 도구",
                     font=ctk.CTkFont(size=11), text_color=GRAY).pack(side="left")

    # ─────────────────────────────────────────────
    # Tab 1 — 모델 학습
    # ─────────────────────────────────────────────
    def _tab_train(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # 1) 학습 데이터 소스
        src = self._card(scroll, "1.  학습 데이터 소스 선택")
        ctk.CTkLabel(src, text="vocal-separator 의 결과 폴더 또는 보컬 WAV 가 있는 폴더",
                     font=ctk.CTkFont(size=11), text_color=GRAY).pack(anchor="w", padx=14, pady=(0, 8))

        row = ctk.CTkFrame(src, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(0, 14))
        self.train_src = ctk.CTkEntry(row, placeholder_text="폴더 경로…", height=36, border_color=BORDER)
        self.train_src.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(row, text="폴더 선택", width=100, height=36,
                      fg_color="#1565c0", hover_color="#1976d2",
                      command=lambda: self._pick_folder(self.train_src)).pack(side="left")

        # 2) 모델 이름 + 전처리
        cfg = self._card(scroll, "2.  전처리 설정")
        r1 = ctk.CTkFrame(cfg, fg_color="transparent")
        r1.pack(fill="x", padx=14, pady=(0, 8))
        ctk.CTkLabel(r1, text="모델 이름", width=90).pack(side="left")
        self.train_name = ctk.CTkEntry(r1, placeholder_text="예) singer_홍길동", height=34, border_color=BORDER)
        self.train_name.pack(side="left", fill="x", expand=True)

        r2 = ctk.CTkFrame(cfg, fg_color="transparent")
        r2.pack(fill="x", padx=14, pady=(0, 14))
        ctk.CTkLabel(r2, text="세그먼트 길이(초)", width=120).pack(side="left")
        self.seg_sec = ctk.CTkSlider(r2, from_=5, to=20, number_of_steps=15)
        self.seg_sec.set(10)
        self.seg_sec.pack(side="left", fill="x", expand=True, padx=8)
        self.seg_label = ctk.CTkLabel(r2, text="10 초", width=50)
        self.seg_label.pack(side="left")
        self.seg_sec.configure(command=lambda v: self.seg_label.configure(text=f"{int(v)} 초"))

        ctk.CTkButton(cfg, text="▶  전처리 시작", height=38,
                      fg_color="#1565c0", hover_color="#1976d2",
                      command=self._on_preprocess).pack(fill="x", padx=14, pady=(0, 14))

        # 3) 학습
        trn = self._card(scroll, "3.  모델 학습")
        r3 = ctk.CTkFrame(trn, fg_color="transparent")
        r3.pack(fill="x", padx=14, pady=(0, 8))
        ctk.CTkLabel(r3, text="Epochs", width=90).pack(side="left")
        self.epochs_var = tk.IntVar(value=100)
        ctk.CTkEntry(r3, textvariable=self.epochs_var, width=80, height=34,
                     border_color=BORDER).pack(side="left")
        ctk.CTkLabel(r3, text="  (CPU: epoch 당 약 3–8분)",
                     text_color=GRAY, font=ctk.CTkFont(size=10)).pack(side="left", padx=8)

        ctk.CTkButton(trn, text="▶  학습 시작", height=38,
                      fg_color=SUCCESS, hover_color="#388e3c",
                      command=self._on_train).pack(fill="x", padx=14, pady=(0, 8))

        note = ctk.CTkLabel(trn,
            text="⚠  CPU 환경에서 100 epoch 는 수 시간 소요됩니다.\n"
                 "    먼저 10–20 epoch 로 테스트 후 품질을 확인하세요.",
            font=ctk.CTkFont(size=10), text_color="#f59e0b", justify="left")
        note.pack(anchor="w", padx=14, pady=(0, 14))

        # 4) 외부 모델 가져오기
        imp = self._card(scroll, "4.  외부 모델 가져오기 (Applio / RVC 커뮤니티)")
        ctk.CTkLabel(imp, text="사전 학습된 .pth 파일을 불러와 변환에 바로 사용할 수 있습니다.",
                     font=ctk.CTkFont(size=11), text_color=GRAY).pack(anchor="w", padx=14, pady=(0, 6))

        r4 = ctk.CTkFrame(imp, fg_color="transparent")
        r4.pack(fill="x", padx=14, pady=(0, 6))
        self.import_pth = ctk.CTkEntry(r4, placeholder_text=".pth 파일 경로", height=34, border_color=BORDER)
        self.import_pth.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(r4, text=".pth", width=60, height=34,
                      fg_color="#4a1a7a", hover_color="#6a2a9a",
                      command=lambda: self._pick_file(self.import_pth, [("PTH", "*.pth")])).pack(side="left")

        r5 = ctk.CTkFrame(imp, fg_color="transparent")
        r5.pack(fill="x", padx=14, pady=(0, 6))
        self.import_idx = ctk.CTkEntry(r5, placeholder_text=".index 파일 경로 (선택)", height=34, border_color=BORDER)
        self.import_idx.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(r5, text=".index", width=60, height=34,
                      fg_color="#4a1a7a", hover_color="#6a2a9a",
                      command=lambda: self._pick_file(self.import_idx, [("INDEX", "*.index")])).pack(side="left")

        r6 = ctk.CTkFrame(imp, fg_color="transparent")
        r6.pack(fill="x", padx=14, pady=(0, 14))
        ctk.CTkLabel(r6, text="모델 이름", width=80).pack(side="left")
        self.import_name = ctk.CTkEntry(r6, placeholder_text="예) singer_아이유", height=34, border_color=BORDER)
        self.import_name.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(r6, text="가져오기", width=90, height=34,
                      fg_color="#1565c0", hover_color="#1976d2",
                      command=self._on_import).pack(side="left")

    # ─────────────────────────────────────────────
    # Tab 2 — 목소리 변환
    # ─────────────────────────────────────────────
    def _tab_convert(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        inp = self._card(scroll, "입력 보컬")
        self._drop_entry(inp, "conv_input", "변환할 WAV 파일을 드래그하거나 선택하세요",
                         [("WAV", "*.wav"), ("MP3", "*.mp3"), ("전체", "*.*")])

        mdl = self._card(scroll, "모델 선택")
        self.conv_model = self._model_combo(mdl)
        ctk.CTkButton(mdl, text="↻  새로고침", width=90, height=30,
                      fg_color="transparent", border_width=1, border_color=BORDER,
                      command=self._refresh_models).pack(anchor="e", padx=14, pady=(0, 4))

        opt = self._card(scroll, "변환 옵션")
        self._slider_row(opt, "음정 조절 (반음)", "transpose_var",
                         from_=-12, to=12, default=0, fmt=lambda v: f"{int(v):+d} 반음")
        self._slider_row(opt, "Index Rate", "index_rate_var",
                         from_=0, to=1, default=0.75, fmt=lambda v: f"{v:.2f}",
                         steps=100)
        ctk.CTkLabel(opt,
            text="Index Rate: 학습 데이터 목소리 비율 (높을수록 원본 가수에 가까움)",
            font=ctk.CTkFont(size=10), text_color=GRAY).pack(anchor="w", padx=14, pady=(0, 10))

        ctk.CTkButton(scroll, text="▶   목소리 변환 시작", height=48,
                      font=ctk.CTkFont(size=15, weight="bold"),
                      fg_color=SUCCESS, hover_color="#388e3c",
                      command=self._on_convert).pack(fill="x", padx=4, pady=(4, 14))

    # ─────────────────────────────────────────────
    # Tab 3 — AI 커버 생성
    # ─────────────────────────────────────────────
    def _tab_cover(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        inp = self._card(scroll, "커버할 곡")
        self._drop_entry(inp, "cover_input",
                         "원곡 MP3 / WAV 를 드래그하거나 선택하세요",
                         [("오디오", "*.mp3 *.wav *.flac"), ("전체", "*.*")])

        mdl = self._card(scroll, "사용할 목소리 모델")
        self.cover_model = self._model_combo(mdl)
        ctk.CTkButton(mdl, text="↻  새로고침", width=90, height=30,
                      fg_color="transparent", border_width=1, border_color=BORDER,
                      command=self._refresh_models).pack(anchor="e", padx=14, pady=(0, 4))

        opt = self._card(scroll, "커버 옵션")
        self._slider_row(opt, "음정 조절 (반음)", "cover_transpose_var",
                         from_=-12, to=12, default=0, fmt=lambda v: f"{int(v):+d} 반음")
        self._slider_row(opt, "보컬 볼륨 조절 (dB)", "vocal_gain_var",
                         from_=-6, to=6, default=0, fmt=lambda v: f"{v:+.1f} dB",
                         steps=120)

        ctk.CTkLabel(opt,
            text="파이프라인: 스템 분리 (Demucs) → 목소리 변환 (RVC) → 합성 (pydub)",
            font=ctk.CTkFont(size=10), text_color=GRAY).pack(anchor="w", padx=14, pady=(0, 10))

        ctk.CTkButton(scroll, text="🎵   AI 커버 생성", height=52,
                      font=ctk.CTkFont(size=16, weight="bold"),
                      fg_color="#4a1a7a", hover_color="#6a2a9a",
                      command=self._on_cover).pack(fill="x", padx=4, pady=(4, 14))

    # ─────────────────────────────────────────────
    # Tab 4 — 사용법
    # ─────────────────────────────────────────────
    def _tab_help(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # ── 워크플로우 ──
        wf = self._card(scroll, "🔄  전체 워크플로우")
        steps = [
            ("Step 1", "vocal-separator 앱으로 원하는 가수의 노래에서 보컬 분리",
             "여러 곡 분리할수록 모델 품질 향상 (10곡 이상 권장)"),
            ("Step 2", "📚 모델 학습 탭 → 소스 폴더 선택 → 전처리 시작",
             "vocals.wav 를 10초 단위로 슬라이싱하여 학습 데이터셋 생성"),
            ("Step 3", "학습 시작 (처음엔 20–30 epoch 로 테스트)",
             "CPU 환경: epoch 당 약 3–8분 / 100 epoch = 약 5–13시간"),
            ("Step 4", "🎵 AI 커버 생성 탭 → 원곡 + 모델 선택 → 생성",
             "Demucs 분리 → RVC 변환 → 합성 순서로 자동 처리"),
        ]
        for title, desc, tip in steps:
            row = ctk.CTkFrame(wf, fg_color="#16213e", corner_radius=8)
            row.pack(fill="x", padx=14, pady=(0, 6))
            ctk.CTkLabel(row, text=title,
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color=ACCENT, width=60).pack(side="left", padx=(12, 8), pady=10)
            col = ctk.CTkFrame(row, fg_color="transparent")
            col.pack(side="left", fill="x", expand=True, pady=8)
            ctk.CTkLabel(col, text=desc,
                         font=ctk.CTkFont(size=11), anchor="w").pack(anchor="w")
            ctk.CTkLabel(col, text=f"💡 {tip}",
                         font=ctk.CTkFont(size=10), text_color=GRAY, anchor="w").pack(anchor="w")
        ctk.CTkFrame(wf, height=6, fg_color="transparent").pack()

        # ── 저장 위치 ──
        st = self._card(scroll, "💾  저장 위치")
        storage_info = [
            ("학습 데이터셋",
             "models/<모델명>/dataset/",
             "노래 1곡(4분) 기준 약 30–80 MB\n곡 수에 따라 선형 증가"),
            ("모델 파일 (.pth)",
             "models/<모델명>/*.pth",
             "모델 1개 약 60–200 MB\n(학습 epoch 수에 무관하게 고정)"),
            ("Index 파일 (.index)",
             "models/<모델명>/*.index",
             "약 50–300 MB\n(학습 데이터 양에 비례)"),
            ("변환 결과 / 커버곡",
             "output/",
             "WAV: 곡당 약 30–150 MB\nMP3 (320k): 곡당 약 8–15 MB"),
        ]
        for name, path, note in storage_info:
            row = ctk.CTkFrame(st, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=(0, 8))
            ctk.CTkLabel(row, text=name,
                         font=ctk.CTkFont(size=11, weight="bold"),
                         width=150, anchor="w").pack(side="left")
            col = ctk.CTkFrame(row, fg_color="transparent")
            col.pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(col, text=path,
                         font=ctk.CTkFont(family="Consolas", size=10),
                         text_color=ACCENT, anchor="w").pack(anchor="w")
            ctk.CTkLabel(col, text=note,
                         font=ctk.CTkFont(size=10), text_color=GRAY, anchor="w").pack(anchor="w")
        ctk.CTkFrame(st, height=4, fg_color="transparent").pack()

        # ── 음정 조절 가이드 ──
        tp = self._card(scroll, "🎵  음정 조절 (Transpose) 가이드")
        ctk.CTkLabel(tp,
            text="남성 → 여성 목소리 모델:  +6 ~ +12 반음\n"
                 "여성 → 남성 목소리 모델:  -6 ~ -12 반음\n"
                 "같은 성별 목소리:          0 ~ ±3 반음\n\n"
                 "※ 값이 너무 크면 음정이 부자연스러워집니다. ±6 단위로 테스트하세요.",
            font=ctk.CTkFont(size=11), text_color="white",
            justify="left").pack(anchor="w", padx=14, pady=(0, 14))

        # ── CPU 학습 예상 시간 ──
        tm = self._card(scroll, "⏱  CPU 학습 예상 소요 시간")
        time_data = [
            ("10 epoch  (테스트용)",   "30분 – 1시간 20분",  "목소리 특성 확인 가능"),
            ("50 epoch  (기본)",       "2.5 – 7시간",        "어느 정도 자연스러운 변환"),
            ("100 epoch (권장)",       "5 – 13시간",         "전반적으로 안정적인 품질"),
            ("200 epoch (고품질)",     "10 – 27시간",        "밤새 학습 권장"),
        ]
        header = ctk.CTkFrame(tm, fg_color="#1a1a2e", corner_radius=6)
        header.pack(fill="x", padx=14, pady=(0, 2))
        for col_text, w in [("설정", 140), ("소요 시간", 160), ("결과", 300)]:
            ctk.CTkLabel(header, text=col_text, width=w,
                         font=ctk.CTkFont(size=10, weight="bold"),
                         text_color=GRAY).pack(side="left", padx=6, pady=4)

        for epoch, duration, quality in time_data:
            row = ctk.CTkFrame(tm, fg_color="transparent")
            row.pack(fill="x", padx=14)
            for text, w, color in [
                (epoch,    140, "white"),
                (duration, 160, ACCENT),
                (quality,  300, GRAY),
            ]:
                ctk.CTkLabel(row, text=text, width=w,
                             font=ctk.CTkFont(size=10), text_color=color,
                             anchor="w").pack(side="left", padx=6, pady=3)
        ctk.CTkFrame(tm, height=8, fg_color="transparent").pack()

    # ─────────────────────────────────────────────
    # 저장공간 현황 바
    # ─────────────────────────────────────────────
    def _storage_bar(self):
        bar = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=8, height=36)
        bar.pack(fill="x", padx=20, pady=(6, 0))
        bar.pack_propagate(False)

        ctk.CTkLabel(bar, text="💾 저장공간",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=GRAY).pack(side="left", padx=(12, 16))

        self.lbl_models  = ctk.CTkLabel(bar, text="models/ : —",
                                        font=ctk.CTkFont(size=10), text_color="white")
        self.lbl_models.pack(side="left", padx=8)

        self.lbl_output  = ctk.CTkLabel(bar, text="output/ : —",
                                        font=ctk.CTkFont(size=10), text_color="white")
        self.lbl_output.pack(side="left", padx=8)

        self.lbl_free    = ctk.CTkLabel(bar, text="디스크 여유 : —",
                                        font=ctk.CTkFont(size=10), text_color=GRAY)
        self.lbl_free.pack(side="left", padx=8)

        ctk.CTkButton(bar, text="↻", width=28, height=22,
                      fg_color="transparent", hover_color="#2a2a3e",
                      font=ctk.CTkFont(size=12),
                      command=self._refresh_storage).pack(side="right", padx=8)

        self._refresh_storage()

    def _refresh_storage(self):
        def calc():
            def dir_size(p: Path) -> int:
                return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

            def fmt(b: int) -> str:
                for unit in ("B", "KB", "MB", "GB"):
                    if b < 1024:
                        return f"{b:.1f} {unit}"
                    b /= 1024
                return f"{b:.1f} TB"

            m_size = dir_size(proc.models_dir) if proc.models_dir.exists() else 0
            o_size = dir_size(proc.output_dir)  if proc.output_dir.exists()  else 0
            free   = shutil.disk_usage(Path(".")).free

            self.after(0, lambda: (
                self.lbl_models.configure(text=f"models/ : {fmt(m_size)}"),
                self.lbl_output.configure(text=f"output/ : {fmt(o_size)}"),
                self.lbl_free.configure(text=f"디스크 여유 : {fmt(free)}"),
            ))

        threading.Thread(target=calc, daemon=True).start()

    # ─────────────────────────────────────────────
    # 로그 박스 (탭 아래 공용)
    # ─────────────────────────────────────────────
    def _log_box(self):
        ctk.CTkLabel(self, text="처리 로그",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=GRAY).pack(anchor="w", padx=20)
        self.progress = ctk.CTkProgressBar(self, height=6, corner_radius=3,
                                           fg_color=CARD_BG, progress_color=ACCENT)
        self.progress.pack(fill="x", padx=20, pady=(2, 4))
        self.progress.set(0)
        self.log = ctk.CTkTextbox(self, height=160,
                                  font=ctk.CTkFont(family="Consolas", size=10),
                                  fg_color=LOG_BG, text_color=LOG_FG,
                                  border_color=BORDER, border_width=1, corner_radius=8)
        self.log.pack(fill="x", padx=20, pady=(0, 12))

    # ─────────────────────────────────────────────
    # 재사용 UI 헬퍼
    # ─────────────────────────────────────────────
    def _card(self, parent, title):
        f = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=12)
        f.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(f, text=title,
                     font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=14, pady=(12, 8))
        return f

    def _drop_entry(self, parent, attr, placeholder, filetypes):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(0, 14))
        entry = ctk.CTkEntry(row, placeholder_text=placeholder, height=36, border_color=BORDER)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        setattr(self, attr, entry)
        ctk.CTkButton(row, text="선택", width=70, height=36,
                      fg_color="#1565c0", hover_color="#1976d2",
                      command=lambda e=entry, ft=filetypes: self._pick_file(e, ft)).pack(side="left")
        for w in (row, entry):
            w.drop_target_register(DND_FILES)
            w.dnd_bind("<<Drop>>", lambda ev, e=entry: self._on_drop_entry(ev, e))

    def _model_combo(self, parent):
        models = proc.list_models() or ["(모델 없음)"]
        combo = ctk.CTkComboBox(parent, values=models, width=300, height=36, border_color=BORDER)
        combo.pack(anchor="w", padx=14, pady=(0, 4))
        return combo

    def _slider_row(self, parent, label, attr, from_, to, default, fmt, steps=24):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(0, 6))
        ctk.CTkLabel(row, text=label, width=160).pack(side="left")
        var = tk.DoubleVar(value=default)
        setattr(self, attr, var)
        val_label = ctk.CTkLabel(row, text=fmt(default), width=80)
        val_label.pack(side="right")
        sl = ctk.CTkSlider(row, from_=from_, to=to, number_of_steps=steps, variable=var,
                           command=lambda v, lbl=val_label, f=fmt: lbl.configure(text=f(v)))
        sl.pack(side="left", fill="x", expand=True, padx=8)

    def _refresh_models(self):
        models = proc.list_models() or ["(모델 없음)"]
        self.conv_model.configure(values=models)
        self.cover_model.configure(values=models)
        if models:
            self.conv_model.set(models[0])
            self.cover_model.set(models[0])

    # ─────────────────────────────────────────────
    # 파일/폴더 선택
    # ─────────────────────────────────────────────
    def _pick_file(self, entry, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def _pick_folder(self, entry):
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def _on_drop_entry(self, event, entry):
        try:
            paths = self.tk.splitlist(event.data)
        except Exception:
            paths = [event.data.strip().strip("{}")]
        if paths:
            entry.delete(0, "end")
            entry.insert(0, paths[0])

    # ─────────────────────────────────────────────
    # 로그 / 진행률
    # ─────────────────────────────────────────────
    def _emit(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    def _set_busy(self, busy: bool, btn=None, label=""):
        self.busy = busy
        if btn:
            btn.configure(state="disabled" if busy else "normal",
                          text=("⏳  처리 중..." if busy else label))
        if busy:
            self.progress.configure(mode="indeterminate")
            self.progress.start()
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self.progress.set(1 if not busy else 0)
            self._refresh_storage()   # 작업 완료 후 자동 갱신

    def _run(self, fn, btn=None, btn_label=""):
        if self.busy:
            return
        self._set_busy(True, btn, btn_label)
        def worker():
            try:
                fn()
            except Exception as e:
                self._emit(f"[오류] ❌ {e}")
            finally:
                self.after(0, self._set_busy, False, btn, btn_label)
        threading.Thread(target=worker, daemon=True).start()

    # ─────────────────────────────────────────────
    # 이벤트 핸들러
    # ─────────────────────────────────────────────
    def _on_preprocess(self):
        src  = self.train_src.get().strip()
        name = self.train_name.get().strip()
        sec  = int(self.seg_sec.get())
        if not src or not name:
            self._emit("❌ 소스 폴더와 모델 이름을 입력하세요.")
            return

        def run():
            self._emit(f"\n── 전처리 시작: {name} ──")
            _, count = proc.preprocess_for_training(
                src, name, segment_sec=sec,
                progress_cb=self._emit)
            self._emit(f"✅ 전처리 완료 — {count}개 세그먼트")

        self._run(run)

    def _on_train(self):
        name   = self.train_name.get().strip()
        epochs = self.epochs_var.get()
        if not name:
            self._emit("❌ 모델 이름을 입력하세요.")
            return

        def run():
            self._emit(f"\n── 학습 시작: {name} | {epochs} epochs ──")
            proc.train_model(name, epochs=epochs, progress_cb=self._emit)
            self._emit("✅ 학습 완료")
            self.after(0, self._refresh_models)

        self._run(run)

    def _on_import(self):
        pth  = self.import_pth.get().strip()
        idx  = self.import_idx.get().strip() or None
        name = self.import_name.get().strip()
        if not pth or not name:
            self._emit("❌ .pth 파일과 모델 이름을 입력하세요.")
            return

        def run():
            proc.import_model(pth, idx, name)
            self._emit(f"✅ 모델 가져오기 완료: {name}")
            self.after(0, self._refresh_models)

        self._run(run)

    def _on_convert(self):
        inp   = self.conv_input.get().strip()
        model = self.conv_model.get()
        trans = int(self.transpose_var.get())
        irate = self.index_rate_var.get()
        if not inp or model == "(모델 없음)":
            self._emit("❌ 입력 파일과 모델을 선택하세요.")
            return

        def run():
            self._emit(f"\n── 목소리 변환: {Path(inp).name} → [{model}] ──")
            out = proc.convert_voice(inp, model, transpose=trans,
                                     index_rate=irate, progress_cb=self._emit)
            self._emit(f"✅ 변환 완료 → {out}")
            subprocess.Popen(f'explorer /select,"{out}"')

        self._run(run)

    def _on_cover(self):
        inp   = self.cover_input.get().strip()
        model = self.cover_model.get()
        trans = int(self.cover_transpose_var.get())
        gain  = self.vocal_gain_var.get()
        if not inp or model == "(모델 없음)":
            self._emit("❌ 원곡 파일과 모델을 선택하세요.")
            return

        def run():
            self._emit(f"\n── AI 커버 생성 ──")
            self._emit(f"   곡: {Path(inp).name}")
            self._emit(f"   모델: {model}  음정: {trans:+d}  볼륨: {gain:+.1f}dB")
            out = proc.generate_cover(inp, model, transpose=trans,
                                       vocal_gain_db=gain, progress_cb=self._emit)
            self._emit(f"✅ 커버 완성 → {out}")
            subprocess.Popen(f'explorer /select,"{out}"')

        self._run(run)


if __name__ == "__main__":
    App().mainloop()
