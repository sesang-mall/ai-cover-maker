import re
import subprocess
from pathlib import Path

import librosa
import soundfile as sf
import numpy as np


class CoverProcessor:
    def __init__(self, models_dir="models", output_dir="output"):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    # ──────────────────────────────────────────────
    # 모델 관리
    # ──────────────────────────────────────────────
    def list_models(self) -> list[str]:
        """models/ 아래 .pth 파일을 가진 폴더명 목록 반환"""
        return [
            p.parent.name
            for p in self.models_dir.glob("**/*.pth")
        ]

    def model_path(self, model_name: str):
        d = self.models_dir / model_name
        pths = list(d.glob("*.pth"))
        return pths[0] if pths else None

    def index_path(self, model_name: str):
        d = self.models_dir / model_name
        idxs = list(d.glob("*.index"))
        return idxs[0] if idxs else None

    def import_model(self, pth_src: str, index_src: str | None, model_name: str):
        """외부 .pth / .index 파일을 models/ 로 복사"""
        import shutil
        dest = self.models_dir / model_name
        dest.mkdir(exist_ok=True)
        shutil.copy2(pth_src, dest / Path(pth_src).name)
        if index_src:
            shutil.copy2(index_src, dest / Path(index_src).name)

    # ──────────────────────────────────────────────
    # 학습 전처리 — 보컬 세그먼트 슬라이싱
    # ──────────────────────────────────────────────
    def preprocess_for_training(
        self,
        source_dir: str,
        model_name: str,
        segment_sec: int = 10,
        min_sec: int = 3,
        target_sr: int = 40000,
        progress_cb=None,
    ) -> tuple[Path, int]:
        """
        source_dir 아래 vocals.wav 파일들을 segment_sec 단위로 슬라이싱.
        결과는 models/<model_name>/dataset/ 에 저장.
        """
        source_dir = Path(source_dir)
        dataset_dir = self.models_dir / model_name / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        wav_files = list(source_dir.rglob("vocals.wav"))
        if not wav_files:
            # 일반 WAV/MP3도 허용
            wav_files = [
                f for f in source_dir.rglob("*")
                if f.suffix.lower() in (".wav", ".mp3", ".flac")
            ]

        if not wav_files:
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {source_dir}")

        seg_idx = 0
        for i, wav_path in enumerate(wav_files):
            if progress_cb:
                progress_cb(f"[{i+1}/{len(wav_files)}] {wav_path.name} 처리 중")

            y, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)
            seg_samples = segment_sec * sr
            min_samples = min_sec * sr

            pos = 0
            while pos < len(y):
                seg = y[pos: pos + seg_samples]
                if len(seg) < min_samples:
                    break
                out = dataset_dir / f"seg_{seg_idx:05d}.wav"
                sf.write(str(out), seg, target_sr, subtype="PCM_16")
                seg_idx += 1
                pos += seg_samples

        if progress_cb:
            progress_cb(f"전처리 완료: {seg_idx}개 세그먼트 → {dataset_dir}")
        return dataset_dir, seg_idx

    # ──────────────────────────────────────────────
    # RVC 모델 학습
    # ──────────────────────────────────────────────
    def train_model(
        self,
        model_name: str,
        epochs: int = 100,
        batch_size: int = 4,
        progress_cb=None,
    ):
        """
        rvc-python 을 사용해 모델 학습.
        CPU 환경에서는 epoch 당 수 분 소요.
        """
        try:
            from rvc_python.train import train_model as rvc_train
        except ImportError:
            raise RuntimeError(
                "rvc-python 이 설치되어 있지 않습니다.\n"
                "pip install rvc-python 으로 설치하세요."
            )

        dataset_dir = self.models_dir / model_name / "dataset"
        if not dataset_dir.exists() or not list(dataset_dir.glob("*.wav")):
            raise FileNotFoundError(
                f"학습 데이터가 없습니다. 먼저 전처리를 실행하세요: {dataset_dir}"
            )

        save_dir = self.models_dir / model_name
        if progress_cb:
            progress_cb(f"학습 시작 — 모델: {model_name}  epochs: {epochs}")

        rvc_train(
            model_name=model_name,
            dataset_path=str(dataset_dir),
            save_path=str(save_dir),
            total_epoch=epochs,
            batch_size=batch_size,
            sr="40k",
            if_f0=True,
            progress_callback=progress_cb,
        )

        if progress_cb:
            progress_cb(f"학습 완료 → {save_dir}")
        return save_dir

    # ──────────────────────────────────────────────
    # 목소리 변환 (Inference)
    # ──────────────────────────────────────────────
    def convert_voice(
        self,
        input_wav: str | Path,
        model_name: str,
        transpose: int = 0,
        index_rate: float = 0.75,
        f0_method: str = "rmvpe",
        progress_cb=None,
    ) -> Path:
        """학습된 모델로 보컬을 변환하여 WAV 반환"""
        try:
            from rvc_python.infer import RVCInference
        except ImportError:
            raise RuntimeError("rvc-python 이 설치되어 있지 않습니다.")

        pth = self.model_path(model_name)
        if pth is None:
            raise FileNotFoundError(f"모델 파일(.pth)이 없습니다: models/{model_name}/")
        idx = self.index_path(model_name)

        if progress_cb:
            progress_cb(f"모델 로드: {pth.name}")

        output_wav = self.output_dir / f"converted_{Path(input_wav).stem}.wav"

        rvc = RVCInference(device="cpu")
        rvc.load_model(str(pth), str(idx) if idx else None)
        rvc.infer_file(
            input_path=str(input_wav),
            output_path=str(output_wav),
            f0_up_key=transpose,
            f0_method=f0_method,
            index_rate=index_rate,
        )

        if progress_cb:
            progress_cb(f"변환 완료: {output_wav.name}")
        return output_wav

    # ──────────────────────────────────────────────
    # 스템 분리 (Demucs, 커버 생성용)
    # ──────────────────────────────────────────────
    def separate_stems(self, audio_path: str | Path, progress_cb=None) -> Path:
        cmd = [
            "python", "-m", "demucs",
            "-n", "htdemucs",
            "-o", str(self.output_dir),
            str(audio_path),
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        buf = b""
        while True:
            chunk = proc.stdout.read(256)
            if not chunk:
                break
            buf += chunk
            parts = re.split(b"[\r\n]", buf)
            for part in parts[:-1]:
                line = part.decode("utf-8", errors="replace").strip()
                if line and progress_cb:
                    progress_cb(line)
            buf = parts[-1]
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("Demucs 분리 실패")
        return self.output_dir / "htdemucs" / Path(audio_path).stem

    # ──────────────────────────────────────────────
    # Instrumental 합성 (drums + bass + other)
    # ──────────────────────────────────────────────
    def merge_stems_to_instrumental(self, stem_dir: Path, progress_cb=None) -> Path:
        try:
            from pydub import AudioSegment
        except ImportError:
            raise RuntimeError("pydub 이 설치되어 있지 않습니다.")

        mixed = None
        for stem in ("drums", "bass", "other"):
            wav = stem_dir / f"{stem}.wav"
            if not wav.exists():
                continue
            seg = AudioSegment.from_wav(str(wav))
            mixed = seg if mixed is None else mixed.overlay(seg)

        if mixed is None:
            raise FileNotFoundError("instrumental 스템이 없습니다.")

        out = stem_dir / "instrumental.wav"
        mixed.export(str(out), format="wav")
        if progress_cb:
            progress_cb(f"instrumental 생성: {out.name}")
        return out

    # ──────────────────────────────────────────────
    # 최종 커버 합성
    # ──────────────────────────────────────────────
    def mix_cover(
        self,
        instrumental: Path,
        converted_vocals: Path,
        output_name: str,
        vocal_gain_db: float = 0.0,
        progress_cb=None,
    ) -> Path:
        from pydub import AudioSegment

        inst = AudioSegment.from_wav(str(instrumental))
        voc  = AudioSegment.from_wav(str(converted_vocals))

        if vocal_gain_db != 0:
            voc = voc + vocal_gain_db

        # 길이를 instrumental 기준으로 맞춤
        if len(voc) > len(inst):
            voc = voc[:len(inst)]
        elif len(voc) < len(inst):
            silence = AudioSegment.silent(duration=len(inst) - len(voc))
            voc = voc + silence

        cover = inst.overlay(voc)
        out = self.output_dir / f"{output_name}_cover.mp3"
        cover.export(str(out), format="mp3", bitrate="320k")
        if progress_cb:
            progress_cb(f"커버 저장: {out.name}")
        return out

    # ──────────────────────────────────────────────
    # 원스텝 커버 생성 파이프라인
    # ──────────────────────────────────────────────
    def generate_cover(
        self,
        audio_path: str | Path,
        model_name: str,
        transpose: int = 0,
        vocal_gain_db: float = 0.0,
        progress_cb=None,
    ) -> Path:
        audio_path = Path(audio_path)

        if progress_cb:
            progress_cb("[1/4] 스템 분리 중 (Demucs)...")
        stem_dir = self.separate_stems(audio_path, progress_cb=progress_cb)

        if progress_cb:
            progress_cb("[2/4] Instrumental 합성 중...")
        instrumental = self.merge_stems_to_instrumental(stem_dir, progress_cb=progress_cb)

        if progress_cb:
            progress_cb("[3/4] 목소리 변환 중 (RVC)...")
        converted = self.convert_voice(
            stem_dir / "vocals.wav",
            model_name,
            transpose=transpose,
            progress_cb=progress_cb,
        )

        if progress_cb:
            progress_cb("[4/4] 최종 합성 중...")
        cover = self.mix_cover(
            instrumental, converted, audio_path.stem,
            vocal_gain_db=vocal_gain_db, progress_cb=progress_cb,
        )

        return cover
