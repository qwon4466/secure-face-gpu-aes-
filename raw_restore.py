from __future__ import annotations
from pathlib import Path
import os
import json
from typing import Any
import numpy as np
from utils.aes_crypto import encrypt_and_remove
import config as c


class RawRecorder:
    """
    원본 프레임을 raw.bin으로 저장하고
    close() 시 AES-256으로 암호화 후
    raw.bin을 삭제한다.
    """

    def __init__(
        self,
        save_root: str | Path,
        fps: float,
        width: int,
        height: int,
        channels: int = 3,
        dtype: str = "uint8",
    ):
        
        from datetime import datetime

        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.dtype = np.dtype(dtype)

        now = datetime.now()
        date_dir = now.strftime("%Y-%m-%d")
        time_dir = now.strftime("%H-%M-%S")
        
        # 2. "raw_data/날짜/시간" 형태의 기본 폴더 경로 지정
        base_folder = Path("raw_data") / date_dir / time_dir

        # 3. camera_stream에서 넘어온 원본 경로에서 파일명만 추출
        original_path = Path(save_root)
        
        # 파일명(확장자)이 지정되어 있지 않으면 기본값으로 "raw.bin" 사용
        if original_path.suffix == "":
            filename = original_path.with_suffix(".bin").name
        else:
            filename = original_path.name

        # 4. 최종 저장 경로 합치기
        path = base_folder / filename

        # 5. 지정된 경로의 폴더가 없다면 모두 자동 생성
        path.parent.mkdir(parents=True, exist_ok=True)

        self.path = path
        self.meta_path = self.path.with_name(self.path.name + ".json")

        self._file = self.path.open("wb")
        self._closed = False

        self._write_metadata()

    def _write_metadata(self):

        metadata = {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "dtype": str(self.dtype),
            "format": "raw",
        }

        with self.meta_path.open(
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=4)

    def write(self, frame: Any):

        if self._closed:
            raise RuntimeError("Recorder already closed.")

        arr = np.asarray(frame)

        if arr.dtype != self.dtype:
            arr = arr.astype(self.dtype)

        expected = (
            (self.height, self.width)
            if self.channels == 1
            else (self.height, self.width, self.channels)
        )

        if arr.shape != expected:
            raise ValueError(
                f"Expected {expected}, got {arr.shape}"
            )

        self._file.write(arr.tobytes())

    def close(self):

        if self._closed:
            return

        self._file.close()

        encrypt_and_remove(str(self.path))

        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self,
                 exc_type,
                 exc_val,
                 exc_tb):
        self.close()

    @staticmethod
    def find_chunk_file(chunk_id: str):
        root = Path(c.RAW_ENCRYPT_DIR)

        # 암호화 파일 검색
        for file in root.rglob(f"{chunk_id}.enc"):
            return file

        # 혹시 암호화 전 bin이 남아있는 경우
        for file in root.rglob(f"{chunk_id}.bin"):
            return file

        return None