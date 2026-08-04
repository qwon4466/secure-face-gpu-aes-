"""
SecureFace-RX 전역 설정
(123.txt의 설정을 기준으로 zxc.txt의 디바이스, 아키텍처, 난독화 등 누락된 내용 병합)
"""

import os
import torch
import platform

# ─── 디바이스 (zxc.txt 기준) ──────────────────────────────────────
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ─── INN 아키텍처 (zxc.txt 기준) ──────────────────────────────────
INV_BLOCKS   = 3       # INV_block_affine 반복 수 (config.INV_BLOCKS)
channels_in  = 3       # 입력 채널 수 (RGB)
clamp        = 2.0     # affine 스케일 클램핑 계수

# ─── 오복원(Wrong Recovery) 모드 (zxc.txt 기준) ───────────────────
# 'Random': RandWR — 랜덤 노이즈형 오복원 (PSNR<11dB)
# 'Obfs'  : ObfsWR — 난독화 유지형 오복원
WRONG_RECOVER_TYPE = 'Random'

# ─── 키 보조입력 정책 (zxc.txt 기준) ──────────────────────────────
SECRET_KEY_AS_NOISE = True  
# 복원 보조입력으로 K를 3채널 반복

# ─── Utility 조건부 기능 (기본 비활성) (zxc.txt 기준) ──────────────
ADJ_UTILITY = False

# ─── 정규화 해상도 (zxc.txt 기준) ─────────────────────────────────
# 원본 config: cropsize=224, SRS: NORM_RESOLUTION=256
# 공식 가중치 사용 시 학습된 해상도에 맞춰야 함
NORM_RESOLUTION = 256   # 변경 시 key 길이도 달라짐

# ─── 사전 난독화 (zxc.txt 기준) ───────────────────────────────────
DEFAULT_OBFUSCATOR = 'blur'
BLUR_KERNEL_SIZE   = 61
BLUR_SIGMA         = 21.0     # 원본 hybridAll: Blur(61, 9, 21)
BLUR_SIGMA_MIN     = 9.0      # 원본 hybridAll blur sigma_min
PIXELATE_BLOCK     = 20       
# 원본 hybridAll: Pixelate(20)
MEDIAN_KERNEL      = 23       # 원본 hybridAll: MedianBlur(23)

# ─── 검출기 (zxc.txt 기준) ────────────────────────────────────────
DETECTOR_CONF_THRESHOLD = 0.25
DETECTOR_NMS_IOU        = 0.4
FACE_MARGIN             = 0.10


# ─── 학습 하이퍼파라미터 (123.txt 기준) ───────────────────────────
lr           = 0.00001
batch_size   = 6
weight_decay = 1e-5
init_scale   = 0.01
TRIPLET_MARGIN         = 1.2
LAMBDA_RECONSTRUCTION  = 5
LAMBDA_GUIDE           = 1
LAMBDA_LOW_FREQUENCY   = 1

SAVE_IMAGE_INTERVAL = 1000
SAVE_MODEL_INTERVAL = 5000

# ─── 사전학습 가중치 파일명 ───────────────────────────────────────
CHECKPOINT_ID = "hybridAll_inv3_recTypeRandom_secretAsNoise_TripMargin1.2_ep12_iter15000"

# ─── KeyGen (PBKDF2) ── NFR-SEC-2 경고 ───────────────────────────
# !!
# salt=1, count=10 은 논문의 "demonstration only" 값 !!
# 운영 배포 시 임의 salt + OWASP 권고 반복 수(≥600000)로 교체할 것
KEY_SALT  = 1
KEY_COUNT = 10

# ─── 기타 ─────────────────────────────────────────────────────────
debug = False
recognizer = 'AdaFaceIR100'

# ─── 통합 시스템 설정 (123.txt 우선) ───────────────────────────────
# 복원 비밀번호 (데모용 — 실제 배포 시 환경변수로 교체)
DEMO_PASSWORD = "0000"

# 관리자 로그인 계정 (사원 등록/관리 접근 제어). 실제 배포 시 반드시 변경.
ADMIN_ID = "0000"
ADMIN_PASSWORD = "0000"

# 관리자 회원가입 개방 여부. False면 로그인 화면의 회원가입이 막힌다(보안).
# 데모에서 새 계정을 만들어 보이려면 True로. (실운영 시 False 권장)
ADMIN_ALLOW_SIGNUP = False

# 청크 보관 기간(일). 이보다 오래된 녹화 청크는 자동 삭제. 0이면 자동삭제 안 함.
RETENTION_DAYS = 7

# 같은 비허가자 감지 이벤트를 이 간격(초) 내에는 중복 기록하지 않음(로그 폭주 방지).
DETECT_LOG_COOLDOWN = 10

# INN 가중치 경로 (None이면 랜덤 초기화 — 형태 확인용)
# 실제 가중치가 있으면 아래 경로 지정:
# INN_CHECKPOINT = "checkpoints/hybridAll_inv3_...pth"
INN_CHECKPOINT = f"checkpoints/{CHECKPOINT_ID}.pth"

# ArcFace 코사인 유사도 임계값
MATCH_THRESHOLD = 0.45

# Hailo-8L 가속: SCRFD 탐지 + ArcFace 인식을 Hailo에서 실행.
# True여도 hailo_platform이 없으면 자동으로 insightface(CPU)로 폴백.
# ⚠️ Hailo ArcFace는 임베딩 공간이 달라 전환 시 재등록 필요.
USE_HAILO = True
SCRFD_HEF_PATH = "/usr/share/hailo-models/scrfd_2.5g_h8l.hef"
ARCFACE_HEF_PATH = "/usr/share/hailo-models/arcface_mobilefacenet.hef"
HAILO_DET_THRESH = 0.5

# (참고) zxc.txt 환경에서의 Hailo 모델 경로 변수명
HAILO_SCRFD_HEF   = "/usr/share/hailo-models/scrfd_2.5g_h8l.hef"
HAILO_ARCFACE_HEF = "/usr/share/hailo-models/arcface_mobilefacenet.hef"

# True면 내부인(등록 사원)도 익명화 (전원 보호, 권한자만 복원).
# False면 외부인만 익명화하고 내부인은 신원 표시.
ANONYMIZE_ALL = True

# 실시간 화면 익명화 방식:
#   "mosaic" — 가벼운 모자이크(실시간 부드러움).
# INN 보호본은 녹화 시 별도 생성.
#   "inn"    — 실시간 화면도 INN 보호본(느림, 라즈베리파이 비권장).
# 어느 쪽이든 녹화/복원은 INN으로 동작 → 복원하면 원본이 나옴.
REALTIME_ANON = "mosaic"

# 실시간 화면(모자이크 표시) 갱신 fps 상한.
# 높을수록 화면이 부드러움.
PROCESS_MAX_FPS = 20

# pending 저장 fps (INN 보호본 대상). 복원 영상 부드러움의 기준.
# 짧은 청크(20초)면 5fps여도 완성이 수 분이라 감당 가능. 높일수록 부드럽지만
# 청크 완성이 느려짐(INN 0.4fps 처리라 큐가 쌓임).
SAVE_FPS = 15

# INN 보호본 생성을 병렬로 돌릴 워커 프로세스 수 (GPU 없이 저장 가속).
# 라즈베리파이는 코어 4개 → 3 권장(1코어는 카메라/탐지/OS 몫).
# 1이면 기존 단일 프로세스로 동작(폴백). 탐지·인식은 항상 메인(Hailo)에서만 수행.
# 워커마다 INN 모델 사본을 메모리에 올리므로 값이 클수록 RAM을 더 씀.
PARALLEL_WORKERS = 3

# 워커 1개가 INN 추론에 쓸 CPU 스레드 수 (torch intra-op).
# 권장: PARALLEL_WORKERS × INN_THREADS_PER_WORKER ≈ 코어 수(4)가 되게 맞출 것.
#   예) 3워커 × 1스레드=3, 2워커 × 2스레드=4, 4워커 × 1스레드=4
# 1장당 처리시간과 병렬도의 균형점 → 로그의 fps로 조합을 비교해 최적값을 찾는다.
# 0이면 자동(max(1, 코어수 ÷ 워커수)).
INN_THREADS_PER_WORKER = 1

# (구) 프레임 스킵. 시간 기반 PROCESS_MAX_FPS를 쓰므로 1로 둠.
PROCESS_EVERY_N = 1

# recorder가 뒤처질 때 원본 프레임을 쌓아두는 큐 최대 크기(장).
# 클수록 프레임을 덜 버리지만 메모리↑ (JPEG 압축 저장, 장당 ~50KB).
FRAME_QUEUE_MAX = 10

# 처리 전 프레임 가로 해상도 축소(px).
# 0이면 원본. 탐지 속도↑.
PROCESS_WIDTH = 640

# 카메라 종류: "webcam"(cv2.VideoCapture) 또는 "realsense"(pyrealsense2)
# RealSense D455도 UVC 장치라 OpenCV(webcam)로 컬러 스트림을 받을 수 있음.
CAMERA_TYPE = "webcam"

# 카메라 인덱스. RealSense는 여러 /dev/videoN 중 컬러 노드를 골라야 함.
# scan_camera.py 로 어느 인덱스가 컬러 영상을 주는지 확인 후 지정.
CAMERA_INDEX = 4

# RealSense(pyrealsense2 사용 시) 컬러 스트림 설정
REALSENSE_WIDTH = 640
REALSENSE_HEIGHT = 480
REALSENSE_FPS = 30

# 카메라 장치 이름 (None이면 정수 인덱스 자동 탐색)
# Windows DSHOW: "video=<장치이름>" 형식으로 열림
CAMERA_DEVICE_NAME = "Logi C310 HD WebCam"

# 카메라 실패 시 폴백 동영상 (mp4 등).
# None이면 "Reconnecting" 재시도.
# 카메라 인식에만 집중 → 폴백 영상 사용 안 함.
VIDEO_FALLBACK = None
VIDEO_FALLBACK_FPS = 25   # 재생 속도 (원본 영상 fps에 맞춤)

# True이면 카메라 탐색을 건너뛰고 무조건 VIDEO_FALLBACK 영상을 사용.
# (이 PC 카메라가 검은 프레임만 줄 때 demo.mp4로 강제하는 용도)
FORCE_VIDEO = False

# PSF 녹화 간격(초).
# 0이면 INN이 처리 가능한 최대 속도로 쉬지 않고 저장
# (가장 부드럽지만 CPU 최대 사용). 실제 저장 fps는 로그로 확인.
RECORD_INTERVAL = 0

# 청크 길이(초). 프레임 수 기준 청크에서 "청크당 몇 초 분량"을 정하는 기준값.
# 실제 청크 분할은 벽시계가 아니라 저장된 프레임 수(FRAMES_PER_CHUNK)로 이뤄진다.
# 저장 계층: recordings/월/일/오전오후/시/HH-MM-SS 청크
CHUNK_SECONDS = 20   # 20초 (나중에 10분=600으로 수동 변경)

# 청크당 저장 프레임 수. 이 수만큼 프레임이 쌓이면 청크를 닫고 다음 청크 시작.
# 벽시계 시간과 무관하게 항상 동일한 프레임 수 → 복원 영상 길이가 일정해진다.
# 0이면 CHUNK_SECONDS * SAVE_FPS 로 자동 계산 (600초 * 15fps = 9000장 = 10분).
FRAMES_PER_CHUNK = 0

# 복원 영상 출력 fps (부드러움용). 실제 영상 길이는 프레임 타임스탬프로 맞춰짐.
RESTORE_VIDEO_FPS = 15

# 원격 GPU 복원 엔드포인트 (PC의 pc_restore_server.py 또는 Modal 등).
# "GPU 복원" 버튼이 이 주소로 청크를 보내 GPU에서 복원한다.
# None이면 원격 오프로드 없이 이 서버(라즈베리파이) 로컬에서 복원(느림).
#   - PC 로컬 GPU 서버 사용 예:  "http://192.168.0.10:8500/restore"
#     (PC에서 pc_restore_server.py 실행 후, 그 PC의 LAN IP로 교체)
#   - 같은 공유기/네트워크에 있어야 하고, PC 방화벽에서 8500 포트를 열어야 함.
REMOTE_RESTORE_URL = "http://192.168.0.3:8500/restore"

# (구) Modal 서버리스 GPU 복원 엔드포인트. 크레딧 소진으로 PC 서버로 대체.
MODAL_RESTORE_URL = "https://yena07--securefacerx-restore-restore.modal.run"

# [시험용] 저장(INN 보호본 생성)도 원격 PC GPU로 오프로드하는 엔드포인트.
# 설정하면 Pi는 탐지만(Hailo) 하고, 원본 프레임을 이 주소로 보내 PC GPU가
# INN 보호본을 만들어 돌려준다. None이면 Pi에서 로컬 처리(병렬/단일).
#   예) "https://xxxx.trycloudflare.com/protect"  (복원과 같은 서버, 경로만 /protect)
# ⚠️ 익명화 前 '원본' 프레임이 네트워크로 나감 → 온디바이스 익명화 원칙과 상충.
#    연속 업로드라 대역폭 부담도 큼. 시험/데모 용도로만 사용할 것.
REMOTE_PROTECT_URL = "http://192.168.0.3:8500/protect"

# [실험] 저장 시 여러 프레임을 한 번에 묶어 전송(배치). 1이면 기존 단일 전송.
# 왕복/포장 오버헤드를 줄여 저장 fps를 높일 수 있음. 4~8 정도로 실험.
# (병목이 GPU 연산이면 효과가 제한적일 수 있음)
PROTECT_BATCH_SIZE = 1

# ffmpeg 실행 파일 경로 (None이면 PATH에서 탐색).
# 브라우저 호환 H.264 변환에 사용.
# winget 설치 시 PATH에 없을 수 있어 직접 지정.
#FFMPEG_PATH = r"C:\Users\HOSEO\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
FFMPEG_PATH = "ffmpeg"  # PATH에 있으면 그냥 ffmpeg로 됨

INN_ONNX_PROTECT = "checkpoints/inn_protect.onnx"

# AES 암호화 원본 저장 위치
RAW_ENCRYPT_DIR = "encrypted_originals"
# ─── 디스크 I/O 최적화 (Ramdisk & SD 하이브리드 설정) ────────────────────────
if platform.system() == "Linux":
    # 라즈베리파이: 실시간 저장은 RAM, 영구 보관은 SD카드
    RECORD_RAM_DIR = "/dev/shm/SecureFace_recordings"
    RECORD_SD_DIR = "recordings"
else:
    # 윈도우/맥 (로컬 테스트용): 구별할 필요 없이 기존 폴더 사용
    RECORD_RAM_DIR = "recordings"
    RECORD_SD_DIR = "recordings"

# 폴더가 없으면 미리 생성
os.makedirs(RECORD_RAM_DIR, exist_ok=True)
os.makedirs(RECORD_SD_DIR, exist_ok=True)