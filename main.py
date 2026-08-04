"""
SecureFace-RX v2 — 통합 FastAPI 서버

  GET  /                        → 실시간 모니터링 (SCR-002)
  GET  /register                → 사원 등록 UI
  GET  /assets                  → 보호 자산 목록 (SCR-003/004)
  GET  /employees               → 사원 관리 UI (SCR-006)
  GET  /stream/cam_0            → MJPEG 익명화 스트림
  GET  /api/stats               → 실시간 통계
  GET  /api/users               → 등록 사원 목록
  DELETE /api/users/{name}      → 사원 삭제 (SCR-006)
  POST /api/users/reload        → 카메라 DB 즉시 갱신
  POST /api/register            → 얼굴 등록 (3각도)
  GET  /api/assets              → 녹화 청크 목록
  GET  /api/assets/{chunk_id}   → 청크 상세
  GET  /recordings/{chunk_id}/thumb            → 청크 썸네일 JPEG
  GET  /recordings/{chunk_id}/{frame_id}/frame → 개별 프레임 JPEG
  POST /api/restore             → INN 역변환 복원 (SCR-005)

실행:
  python main.py
  또는
  uvicorn main:app --host 0.0.0.0 --port 8000
"""
import asyncio
import base64
import hashlib
import json
import io
import os
import secrets
import sqlite3
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse, Response, StreamingResponse)
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from insightface.utils import face_align
from utils.aes_crypto import decrypt_file

import torch  # GPU 메모리 및 컨텍스트 관리용 추가

import config as c
import event_log
from fastapi.staticfiles import StaticFiles

from camera_stream import (
    CameraProcessor,
    DB_PATH,
    _adapt_array,
    _convert_array,
)
from recorder import PSFRecorder

IMAGE_DIR = "registered_faces"
os.makedirs(IMAGE_DIR, exist_ok=True)

camera: CameraProcessor | None = None
recorder: PSFRecorder | None = None


# ── DB 초기화 ─────────────────────────────────────────────────────────────
def _init_db():
    sqlite3.register_adapter(np.ndarray, _adapt_array)
    sqlite3.register_converter("array", _convert_array)
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            auth_group TEXT NOT NULL,
            image_path TEXT NOT NULL,
            vector     array NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


# ── Lifespan (FastAPI 0.93+ 권장 방식) ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global camera, recorder
    
    # 1. 시스템 DB 초기화
    _init_db()
    
    # 2. GPU(CUDA) 상태 초기화 (메모리 파편화 방지)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ CUDA is available. GPU 메모리 정리를 완료했습니다.")
    else:
        print("⚠️ CUDA is unavailable. CPU 모드로 동작합니다.")

    # 3. CameraProcessor 초기화 (내부에서 Hailo 및 INN 모델 로드)
    camera = CameraProcessor()
    camera.start(cam_id=getattr(c, "CAMERA_INDEX", 0))
    
    # 4. Recorder 초기화 및 시작
    recorder = PSFRecorder(camera, interval_sec=getattr(c, "RECORD_INTERVAL", 5))
    recorder.start()
    
    yield  # --- 서버 가동 중 ---
    
    # 5. 서버 종료 시 안전한 자원 해제
    if recorder:
        recorder.stop()
    if camera:
        camera.stop()
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── 앱 초기화 ─────────────────────────────────────────────────────────────
app = FastAPI(title="SecureFace-RX v2", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")
app.mount("/recordings", StaticFiles(directory=c.RECORD_RAM_DIR), name="recordings")

# ── 관리자 인증 (사원 등록/관리 접근 제어) ────────────────────────────────
# 서버 측 세션(쿠키) 기반. 로그인 성공 시 토큰을 발급해 쿠키로 저장하고,
# 보호 대상 페이지·API는 이 쿠키를 검증한다. 토큰은 메모리 보관(서버 재시작 시 초기화).
_ADMIN_SESSIONS: dict = {}   # token -> admin_id
_ADMIN_COOKIE = "admin_session"
_ADMIN_STORE = "admin_users.json"   # 회원가입으로 추가된 관리자 계정 저장


def _is_admin(request: Request) -> bool:
    tok = request.cookies.get(_ADMIN_COOKIE)
    return bool(tok) and tok in _ADMIN_SESSIONS


def _admin_id(request: Request) -> str:
    tok = request.cookies.get(_ADMIN_COOKIE, "")
    return _ADMIN_SESSIONS.get(tok, "")


def require_admin(request: Request):
    """보호 API용 의존성. 미로그인 시 401."""
    if not _is_admin(request):
        raise HTTPException(status_code=401, detail="관리자 로그인이 필요합니다.")


def _hash_pw(pw: str, salt: str) -> str:
    return hashlib.sha256((salt + pw).encode("utf-8")).hexdigest()


def _load_admins() -> dict:
    try:
        with open(_ADMIN_STORE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_admins(d: dict):
    with open(_ADMIN_STORE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)


def _check_admin(admin_id: str, pw: str) -> bool:
    # 1) config의 기본 관리자 계정
    if admin_id == getattr(c, "ADMIN_ID", "") and pw == getattr(c, "ADMIN_PASSWORD", ""):
        return True
    # 2) 회원가입으로 등록된 계정
    rec = _load_admins().get(admin_id)
    return bool(rec) and _hash_pw(pw, rec.get("salt", "")) == rec.get("hash")


class AdminLogin(BaseModel):
    id: str = ""
    password: str


@app.get("/admin/login", response_class=HTMLResponse)
async def page_admin_login(request: Request):
    # 이미 로그인돼 있으면 사원 관리로
    if _is_admin(request):
        return RedirectResponse("/employees", status_code=302)
    return templates.TemplateResponse(request=request, name="admin_login.html")


@app.post("/api/admin/login")
async def api_admin_login(data: AdminLogin):
    if not _check_admin(data.id, data.password):
        raise HTTPException(status_code=401,
                            detail="아이디 또는 비밀번호가 일치하지 않습니다.")
    tok = secrets.token_urlsafe(24)
    _ADMIN_SESSIONS[tok] = data.id
    resp = JSONResponse({"ok": True})
    resp.set_cookie(_ADMIN_COOKIE, tok, httponly=True,
                    max_age=8 * 3600, samesite="lax")
    return resp


@app.post("/api/admin/signup")
async def api_admin_signup(data: AdminLogin):
    """관리자 회원가입 — 새 관리자 계정 생성."""
    if not getattr(c, "ADMIN_ALLOW_SIGNUP", False):
        raise HTTPException(status_code=403,
                            detail="회원가입이 비활성화되어 있습니다. 관리자에게 문의하세요.")
    admin_id = (data.id or "").strip()
    if not admin_id or not data.password:
        raise HTTPException(status_code=400, detail="아이디와 비밀번호를 입력하세요.")
    if len(data.password) < 4:
        raise HTTPException(status_code=400, detail="비밀번호는 4자 이상이어야 합니다.")
    admins = _load_admins()
    if admin_id == getattr(c, "ADMIN_ID", "") or admin_id in admins:
        raise HTTPException(status_code=409, detail="이미 존재하는 아이디입니다.")
    salt = secrets.token_hex(8)
    admins[admin_id] = {"salt": salt, "hash": _hash_pw(data.password, salt)}
    _save_admins(admins)
    return {"ok": True}


@app.post("/api/admin/logout")
async def api_admin_logout(request: Request):
    tok = request.cookies.get(_ADMIN_COOKIE)
    if tok:
        _ADMIN_SESSIONS.pop(tok, None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(_ADMIN_COOKIE)
    return resp


@app.get("/api/admin/status")
async def api_admin_status(request: Request):
    return {
        "logged_in": _is_admin(request),
        "who": _admin_id(request),
        "allow_signup": bool(getattr(c, "ADMIN_ALLOW_SIGNUP", False)),
    }


# ── 페이지 ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def page_monitor(request: Request):
    return templates.TemplateResponse(request=request, name="monitor.html")


@app.get("/register", response_class=HTMLResponse)
async def page_register(request: Request):
    # 관리자 로그인 필요 → 미로그인 시 로그인 페이지로
    if not _is_admin(request):
        return RedirectResponse("/admin/login?next=/register", status_code=302)
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/assets", response_class=HTMLResponse)
async def page_assets(request: Request):
    return templates.TemplateResponse(request=request, name="assets.html")


@app.get("/employees", response_class=HTMLResponse)
async def page_employees(request: Request):
    # 관리자 로그인 필요 → 미로그인 시 로그인 페이지로
    if not _is_admin(request):
        return RedirectResponse("/admin/login?next=/employees", status_code=302)
    return templates.TemplateResponse(request=request, name="employees.html")


# ── 스냅샷 (단일 JPEG, JS 폴링용) ────────────────────────────────────────
@app.get("/snapshot/cam_0")
async def snapshot_cam0():
    jpeg = camera.get_jpeg() if camera else None
    if jpeg is None:
        # 카메라 준비 중 — 빈 1×1 회색 JPEG 반환
        import base64
        placeholder = base64.b64decode(
            "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
            "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
            "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
            "MjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAA"
            "AAAAAAAAAAAAAP/EABQBAQAAAAAAAAAAAAAAAAAAAAD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA"
            "/9oADAMBAAIRAxEAPwCwABmX/9k="
        )
        return Response(content=placeholder, media_type="image/jpeg")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store"},
    )

# ── 등록용 원본 스냅샷 (익명화 전) ──────────────────────────────────────
@app.get("/snapshot/raw")
async def snapshot_raw():
    jpeg = camera.get_raw_jpeg() if camera else None
    if jpeg is None:
        raise HTTPException(status_code=503, detail="카메라 준비 중")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store"},
    )


# ── MJPEG 스트리밍 (호환 브라우저용, 유지) ───────────────────────────────
@app.get("/stream/cam_0")
async def stream_cam0():
    async def generate():
        while True:
            jpeg = camera.get_jpeg() if camera else None
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(0.033)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── API — 통계 ────────────────────────────────────────────────────────────
@app.get("/api/stats")
async def api_stats():
    s = camera.get_stats() if camera else {"employee_count": 0, "unknown_count": 0}
    s["unknown_alerts"] = event_log.count_detections()   # 누적 비허가자 경고 수
    return s


# ── API — 디버그 (카메라 상태 확인) ──────────────────────────────────────
@app.get("/api/debug")
async def api_debug():
    if camera is None:
        return {"error": "camera not initialized"}
    return camera.get_debug_info()


# ── API — 사원 목록 ───────────────────────────────────────────────────────
@app.get("/api/users", dependencies=[Depends(require_admin)])
async def api_users():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT DISTINCT SUBSTR(name, 1, INSTR(name||'_', '_') - 1) as base_name, "
        "auth_group FROM users"
    ).fetchall()
    conn.close()
    seen = {}
    for base_name, group in rows:
        seen[base_name] = group
    return [{"name": k, "group": v} for k, v in seen.items()]


# ── API — DB 재로드 ───────────────────────────────────────────────────────
@app.post("/api/users/reload", dependencies=[Depends(require_admin)])
async def api_reload():
    if camera:
        camera.reload_db()
    return {"status": "ok"}


# ── API — 사원 삭제 (Phase 6) ─────────────────────────────────────────────
@app.delete("/api/users/{name}", dependencies=[Depends(require_admin)])
async def api_delete_user(name: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "DELETE FROM users WHERE name LIKE ?", (f"{name}_%",)
    )
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    if deleted == 0:
        raise HTTPException(status_code=404, detail="사원을 찾을 수 없습니다.")
    if camera:
        camera.reload_db()
    return {"status": "ok", "deleted": deleted}


# ── API — 얼굴 등록 ───────────────────────────────────────────────────────
class RegisterData(BaseModel):
    name: str
    group: str
    image_base64: str


class RegisterCapture(BaseModel):
    name: str
    group: str


def _do_register(frame, name: str, group: str) -> dict:
    """단일 프레임으로 얼굴 탐지 → 각도 판별 → DB 등록."""
    if frame is None:
        return {"status": "error", "message": "❌ 카메라 프레임이 없습니다."}

    detector = camera.detector
    recognizer = camera.recognizer

    bboxes, kpss = detector.detect(frame, max_num=1, metric="default")
    if bboxes is None or len(bboxes) == 0:
        return {"status": "error", "message": "❌ 얼굴을 찾을 수 없습니다."}

    x1, y1, x2, y2 = bboxes[0, :4].astype(int).tolist()
    lm = kpss[0]

    # 측면 판별 (눈-코 거리 비율)
    le, re, nose = lm[0], lm[1], lm[2]
    d_left = np.linalg.norm(le - nose)
    d_right = np.linalg.norm(re - nose)
    ratio = max(d_left, d_right) / (min(d_left, d_right) + 1e-5)
    eye_dist = np.linalg.norm(le - re)
    is_side = ratio > 1.5 or (eye_dist / (x2 - x1 + 1e-5)) < 0.25

    aligned = face_align.norm_crop(frame, landmark=lm, image_size=112)
    emb = recognizer.get_feat(aligned)
    if emb is None:
        return {"status": "error", "message": "❌ 임베딩 추출 실패"}

    if not is_side:
        tag, fname = "정면", f"{name}_정면.jpg"
    else:
        if d_left < d_right:
            tag, fname = "좌측면", f"{name}_측면1(좌).jpg"
        else:
            tag, fname = "우측면", f"{name}_측면2(우).jpg"

    fpath = os.path.join(IMAGE_DIR, fname)
    cv2.imwrite(fpath, frame)

    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute(
        "INSERT INTO users (name, auth_group, image_path, vector) VALUES (?,?,?,?)",
        (f"{name}_{tag}", group, fpath, emb),
    )
    conn.commit()
    conn.close()

    camera.reload_db()
    return {"status": "success", "message": f"✅ [{name}] {tag} 등록 성공!"}


@app.post("/api/register", dependencies=[Depends(require_admin)])
async def api_register(data: RegisterData):
    """브라우저가 캡처한 base64 이미지로 등록 (기존 호환)."""
    try:
        if camera is None:
            return {"status": "error", "message": "서버 초기화 중입니다. 잠시 후 시도하세요."}
        _, encoded = data.image_base64.split(",", 1)
        frame = cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded), np.uint8),
            cv2.IMREAD_COLOR,
        )
        return _do_register(frame, data.name, data.group)
    except Exception as e:
        return {"status": "error", "message": f"서버 오류: {e}"}


@app.post("/api/register_capture", dependencies=[Depends(require_admin)])
async def api_register_capture(data: RegisterCapture):
    """서버(RealSense) 카메라의 현재 원본 프레임으로 등록."""
    try:
        if camera is None:
            return {"status": "error", "message": "서버 초기화 중입니다. 잠시 후 시도하세요."}
        frame = camera.capture_raw_frame()
        return _do_register(frame, data.name, data.group)
    except Exception as e:
        return {"status": "error", "message": f"서버 오류: {e}"}

# ── API — 연속 촬영 얼굴 등록 (애플 Face ID 스타일) ──────────────────────
class RegisterContinuous(BaseModel):
    name: str
    group: str

@app.post("/api/register_continuous_realsense", dependencies=[Depends(require_admin)])
async def api_register_continuous_realsense(data: RegisterContinuous):
    """
    프론트엔드의 트리거 신호를 받아 약 3~4초간 연속으로 프레임을 캡처하고,
    얼굴을 탐지하여 다각도의 특징점(Vector)을 DB에 일괄 등록합니다.
    """
    if camera is None:
        return {"status": "error", "message": "서버 초기화 중입니다. 잠시 후 시도하세요."}

    print(f"▶ [{data.name}] 연속 촬영 등록 시작...")
    
    capture_duration = 5.0  # 5초간 캡처
    interval = 0.2          # 0.2초 간격 (초당 5프레임)
    max_frames = int(capture_duration / interval)
    
    valid_faces = 0
    detector = camera.detector
    recognizer = camera.recognizer
    
    # DB 연결
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    
    try:
        for i in range(max_frames):
            # 1. RealSense 카메라 원본 프레임 가져오기
            frame = camera.capture_raw_frame()
            if frame is None:
                await asyncio.sleep(interval)
                continue
            
            # 2. 얼굴 탐지 (SCRFD)
            bboxes, kpss = detector.detect(frame, max_num=1, metric="default")
            if bboxes is not None and len(bboxes) > 0:
                lm = kpss[0]
                
                # 3. 얼굴 정렬 및 임베딩(특징점) 추출
                aligned = face_align.norm_crop(frame, landmark=lm, image_size=112)
                emb = recognizer.get_feat(aligned)
                
                if emb is not None:
                    # 파일명 생성 및 이미지 저장
                    fname = f"{data.name}_연속_{i}.jpg"
                    fpath = os.path.join(IMAGE_DIR, fname)
                    cv2.imwrite(fpath, frame)
                    
                    # DB 저장 (이름 뒤에 _연속_번호 를 붙여 구분)
                    conn.execute(
                        "INSERT INTO users (name, auth_group, image_path, vector) VALUES (?,?,?,?)",
                        (f"{data.name}_연속_{i}", data.group, fpath, emb),
                    )
                    valid_faces += 1
            
            # 다음 캡처까지 대기 (비동기 딜레이)
            await asyncio.sleep(interval)
            
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"❌ 등록 중 오류 발생: {e}")
        return {"status": "error", "message": f"등록 중 서버 오류 발생: {e}"}
    finally:
        conn.close()

    # 4. 결과 판별 및 DB 리로드
    if valid_faces > 0:
        camera.reload_db()  # 새로 등록된 데이터 즉시 반영
        print(f"✅ [{data.name}] 총 {valid_faces}개의 얼굴 각도 등록 완료.")
        return {"status": "success", "message": f"총 {valid_faces}개의 얼굴 각도가 성공적으로 등록되었습니다."}
    else:
        return {"status": "error", "message": "얼굴을 제대로 인식하지 못했습니다. 밝은 곳에서 정면을 보고 다시 시도해주세요."}

# ── API — 보호 자산 목록 (Phase 4) ───────────────────────────────────────
@app.get("/api/assets")
async def api_assets():
    if recorder is None:
        return []
    return recorder.list_chunks()


@app.get("/api/save_progress")
async def api_save_progress():
    """진행 중인 청크의 저장 진행률·ETA (자산 화면 상단 배너용)."""
    if recorder is None:
        return {"in_progress": False}
    return recorder.save_progress()


# ── API — 감사/이벤트 로그 + GPU 상태 ──────────────────────────────────────
@app.get("/api/audit/restore", dependencies=[Depends(require_admin)])
async def api_audit_restore():
    """복원(익명화 해제) 접근 이력."""
    return event_log.read_restore(300)


@app.get("/api/audit/detections", dependencies=[Depends(require_admin)])
async def api_audit_detections():
    """비허가자 감지 이벤트 이력."""
    return event_log.read_detections(300)


@app.get("/api/gpu_status")
async def api_gpu_status():
    """원격 GPU 서버 연결 상태 (화면 배지용)."""
    url = (getattr(c, "REMOTE_RESTORE_URL", None)
           or getattr(c, "REMOTE_PROTECT_URL", None))
    if not url:
        return {"configured": False, "connected": False}
    health = url.rsplit("/", 1)[0] + "/health"
    try:
        import requests
        r = await asyncio.to_thread(lambda: requests.get(health, timeout=3))
        if r.ok:
            d = r.json()
            return {"configured": True, "connected": True,
                    "device": d.get("device", ""), "cuda": d.get("cuda", False)}
    except Exception:
        pass
    return {"configured": True, "connected": False}


@app.get("/audit", response_class=HTMLResponse)
async def page_audit(request: Request):
    if not _is_admin(request):
        return RedirectResponse("/admin/login?next=/audit", status_code=302)
    return templates.TemplateResponse(request=request, name="audit.html")


@app.get("/api/assets/{chunk_id}")
async def api_asset_detail(chunk_id: str):
    if recorder is None:
        raise HTTPException(status_code=503, detail="Recorder not ready")
    detail = recorder.get_chunk_detail(chunk_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="청크를 찾을 수 없습니다.")
    return detail


@app.get("/api/frame/{chunk_id}/{frame_id}")
async def api_frame(chunk_id: str, frame_id: str):
    """보호본 프레임 JPEG (미리보기 재생용). /recordings 마운트에 가로채이지 않게 /api 경로 사용."""
    if recorder is None:
        raise HTTPException(status_code=503, detail="Recorder not ready")
    data = recorder.get_frame_jpeg(chunk_id, frame_id)
    if data is None:
        raise HTTPException(status_code=404, detail="프레임 없음")
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=60"})


@app.get("/recordings/{chunk_id}/thumb")
async def recording_thumb(chunk_id: str):
    if recorder is None:
        raise HTTPException(status_code=503, detail="Recorder not ready")
    data = recorder.get_thumb_jpeg(chunk_id)
    if data is None:
        raise HTTPException(status_code=404, detail="썸네일 없음")
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=60"})


@app.get("/recordings/{chunk_id}/{frame_id}/frame")
async def recording_frame(chunk_id: str, frame_id: str):
    if recorder is None:
        raise HTTPException(status_code=503, detail="Recorder not ready")
    data = recorder.get_frame_jpeg(chunk_id, frame_id)
    if data is None:
        raise HTTPException(status_code=404, detail="프레임 없음")
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache"})


# ── API — INN 복원 (Phase 5) ──────────────────────────────────────────────
class RestoreRequest(BaseModel):
    chunk_id: str
    frame_id: str
    password: str


@app.post("/api/restore")
async def api_restore(req: RestoreRequest):
    if recorder is None:
        raise HTTPException(status_code=503, detail="Recorder not ready")
        
    # 역변환(추론) 시 그래디언트 계산 방지 및 GPU 안전 확보
    with torch.no_grad():
        data = recorder.restore_frame(req.chunk_id, req.frame_id, req.password)
        
    if data is None:
        raise HTTPException(status_code=404, detail="프레임을 찾을 수 없습니다.")
        
    encoded = base64.b64encode(data).decode()
    return {"status": "ok", "image_base64": f"data:image/jpeg;base64,{encoded}"}


# ── API — 청크 전체 복원 영상 ─────────────────────────────────────────────
class RestoreVideoRequest(BaseModel):
    chunk_id: str
    password: str


@app.post("/api/restore_video")
async def api_restore_video(req: RestoreVideoRequest):
    if recorder is None:
        raise HTTPException(status_code=503, detail="Recorder not ready")
        
    # 동영상 복원은 다량의 프레임을 처리하므로 VRAM 관리가 필수
    with torch.no_grad():
        path = await asyncio.to_thread(
            recorder.restore_chunk_video, req.chunk_id, req.password
        )
        
    # 작업 완료 후 캐시 즉시 반환
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    if path is None or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="복원할 프레임이 없습니다.")
        
    return FileResponse(path, media_type="video/mp4", filename="restored.mp4")


def _remote_gpu_restore(chunk_id: str, password: str) -> str | None:
    """
    청크 폴더를 tar.gz로 묶어 원격 GPU 서버(REMOTE_RESTORE_URL)로 보내고,
    복원된 mp4를 받아 임시 파일로 저장한 뒤 경로를 반환한다. 실패 시 None.
    (PC의 pc_restore_server.py 또는 Modal 등, /restore 계약이 같은 서버 대상)
    """
    import io
    import tarfile
    import tempfile

    import requests
    from recorder import _id_to_path

    url = getattr(c, "REMOTE_RESTORE_URL", None)
    if not url:
        return None
    chunk_dir = _id_to_path(chunk_id)
    if not os.path.isdir(chunk_dir):
        print(f"[GPU복원] 청크 폴더 없음: {chunk_dir}")
        return None

    # 청크 폴더를 메모리 상 tar.gz로 압축
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add(chunk_dir, arcname=os.path.basename(chunk_dir))
    buf.seek(0)

    print(f"[GPU복원] 원격 전송 → {url}  ({len(buf.getvalue())//1024}KB)")
    resp = requests.post(
        url,
        files={"file": ("chunk.tar.gz", buf, "application/gzip")},
        data={"password": password},
        timeout=1800,
    )
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    if "video" not in ctype:
        # 서버가 에러 JSON을 준 경우
        print(f"[GPU복원] 원격 서버 오류 응답: {resp.text[:300]}")
        return None
    out = tempfile.mktemp(suffix=".mp4")
    with open(out, "wb") as f:
        f.write(resp.content)
    return out


@app.post("/api/restore_video_gpu")
async def api_restore_video_gpu(req: RestoreVideoRequest, request: Request):
    if recorder is None:
        raise HTTPException(status_code=503, detail="Recorder not ready")

    who = _admin_id(request) or "미로그인"
    ip = request.client.host if request.client else "-"
    path = None

    # 1순위: 원격 GPU 서버(PC)로 오프로드
    if getattr(c, "REMOTE_RESTORE_URL", None):
        try:
            path = await asyncio.to_thread(
                _remote_gpu_restore, req.chunk_id, req.password
            )
            if not (path and os.path.exists(path)):
                path = None
                print("[GPU복원] 원격 복원 결과 없음 → 로컬 복원으로 폴백")
        except Exception as e:
            path = None
            print(f"[GPU복원] 원격 복원 실패 → 로컬 복원으로 폴백: {e}")

    # 2순위(폴백): 이 서버(라즈베리파이) 로컬 복원
    if path is None:
        with torch.no_grad():
            path = await asyncio.to_thread(
                recorder.restore_chunk_video, req.chunk_id, req.password
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ok = bool(path and os.path.exists(path))
    # 복원 접근 감사 기록 (누가/언제/무엇을 복원했는지)
    event_log.log_restore(req.chunk_id, who, ip, ok, mode="gpu")
    if not ok:
        raise HTTPException(status_code=404, detail="복원할 프레임이 없습니다.")
    return FileResponse(path, media_type="video/mp4", filename="restored_gpu.mp4")

@app.get("/decryption_page")
async def decryption_page(request: Request):
    """상단 메뉴에서 '원본 복호화' 버튼 클릭 시 열리는 전용 웹페이지"""
    return templates.TemplateResponse(request=request, name="decryption.html", context={"request": request})

@app.get("/api/admin/chunks")
async def get_chunk_list():
    """raw_data 폴더를 탐색하여 날짜/시간별 청크 목록을 반환합니다."""
    base_dir = "raw_data"
    chunks = {}
    
    if not os.path.exists(base_dir):
        return JSONResponse(content={})
        
    for date_folder in sorted(os.listdir(base_dir), reverse=True):
        date_path = os.path.join(base_dir, date_folder)
        
        if os.path.isdir(date_path):
            chunks[date_folder] = []
            for time_folder in sorted(os.listdir(date_path), reverse=True):
                time_path = os.path.join(date_path, time_folder)
                
                # 👇 바로 이 부분의 os.path.listdir를 os.listdir로 수정했습니다 👇
                if os.path.isdir(time_path) and "encrypted_raw.bin.enc" in os.listdir(time_path):
                    chunks[date_folder].append(time_folder)
                    
    return JSONResponse(content=chunks)

@app.post("/api/admin/verify_chunk")
async def verify_chunk(folder_path: str = Form(...), password: str = Form(...)):
    """관리자 비밀번호로 청크를 복호화하고 첫 프레임을 반환합니다."""
    enc_file = os.path.join(folder_path, "encrypted_raw.bin.enc")
    json_file = os.path.join(folder_path, "encrypted_raw.bin.json")
    
    # 1. 파일 존재 여부 사전 검증 (에러 방지)
    if not os.path.exists(enc_file) or not os.path.exists(json_file):
        raise HTTPException(status_code=404, detail="암호화된 파일이나 메타데이터를 찾을 수 없습니다.")
        
    temp_decrypted_bin = os.path.join(folder_path, "temp_admin_verify.bin")
    
    # 2. AES-256 복호화 시도
    try:
        decrypt_file(enc_file, temp_decrypted_bin, password)
    except Exception as e:
        if os.path.exists(temp_decrypted_bin):
            os.remove(temp_decrypted_bin)
        raise HTTPException(status_code=401, detail="복호화 실패: 비밀번호가 일치하지 않거나 파일이 손상되었습니다.")
        
    # 3. 메타데이터 읽기 및 이미지 추출 (try-finally로 안전망 구축)
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        width = meta["width"]
        height = meta["height"]
        channels = meta.get("channels", 3)
        frame_size = width * height * channels
        
        with open(temp_decrypted_bin, "rb") as f:
            raw_data = f.read(frame_size)
            
    finally:
        # 무조건 실행 보장: 에러가 나더라도 임시 복호화된 파일은 서버에 남지 않고 즉시 삭제됨
        if os.path.exists(temp_decrypted_bin):
            os.remove(temp_decrypted_bin)
            
    # 4. 추출된 데이터 검증
    if not raw_data or len(raw_data) != frame_size:
        raise HTTPException(status_code=500, detail="프레임 데이터를 정상적으로 읽지 못했습니다.")
        
    # 5. numpy 배열을 JPEG 이미지로 인코딩
    img_array = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, channels))
    success, encoded_img = cv2.imencode('.jpg', img_array)
    
    if not success:
        raise HTTPException(status_code=500, detail="이미지 렌더링에 실패했습니다.")
        
    # 성공 시 브라우저에 이미지 데이터 바로 스트리밍
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

# ── 직접 실행 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
