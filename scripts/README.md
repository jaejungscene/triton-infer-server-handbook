# Scripts 디렉토리 — 운영 자동화 스크립트

## 스크립트 목록

| 스크립트 | 시나리오 | 실행 시점 |
|----------|----------|-----------|
| `build.sh` | manifest.yaml 기반으로 models/serving/ → model_repository/ 동기화 | CI/CD, 로컬 개발 |
| `validate.sh` | 모든 config.pbtxt 유효성 검사 | PR 시, 배포 전 |
| `health_check.sh` | 서버 및 모델별 상태 확인 | 배포 후, 모니터링 |
| `convert/to_onnx.py` | PyTorch → ONNX 변환 | 모델 업데이트 시 |
| `convert/to_tensorrt.sh` | ONNX → TensorRT 변환 | CI/CD |
| `convert/to_torchscript.py` | PyTorch → TorchScript 변환 | 모델 업데이트 시 |
| `convert/to_fil.py` | sklearn/XGBoost → FIL 변환 | 모델 업데이트 시 |
| `model_control/load.sh` | 런타임 모델 로드 후 ready 확인 | explicit mode 운영 |
| `model_control/unload.sh` | 런타임 모델 언로드 완료 확인 | explicit mode 운영 |
| `model_control/reload.sh` | 검증된 artifact를 unload → load | 단일 replica에서는 가용성 공백 발생 |

`health_check.sh`의 URL은 credential이나 path가 없는 HTTP(S) origin만 허용합니다. 인증이 필요한
내부 운영 endpoint는 `TRITON_AUTH_TOKEN` 환경변수를 사용하며 live, ready, Repository Index
요청에 같은 Bearer token을 전달합니다. HTTPS 인증서 검증은 기본으로 유지합니다.

```bash
TRITON_AUTH_TOKEN="$(secret-tool lookup service triton)" \
  scripts/health_check.sh https://triton-ops.example.com
```

모델 이름은 URL 경로에 사용되므로 영문·숫자·점·밑줄·하이픈만 허용합니다. 세 스크립트의
기본 timeout은 120초이며 세 번째 인자로 조정할 수 있습니다. `reload.sh`는 artifact를
배치하지 않으며, CI/CD가 immutable model revision을 먼저 게시한 뒤 실행해야 합니다.
단일 replica의 explicit reload는 무중단이 아닙니다. 무중단 교체가 필요하면 새 Deployment나
새 model name으로 먼저 load·warmup한 후 traffic을 전환하는 blue/green 방식을 사용합니다.

model-control base URL은 credential이나 path가 없는 HTTP(S) origin만 허용합니다. 인증이
필요하면 token을 인수로 전달하지 말고 `TRITON_AUTH_TOKEN` 환경변수로 주입합니다. 스크립트는
권한이 `0600`인 임시 header 파일을 만들고 종료 시 제거하므로 token이 curl 명령행에 직접
나타나지 않습니다. HTTPS는 curl의 기본 CA 검증을 유지하며 인증서 검증을 끄는 옵션은
제공하지 않습니다.

```bash
TRITON_AUTH_TOKEN="$(secret-tool lookup service triton)" \
  scripts/model_control/load.sh text_classifier https://triton.example.com 180
```

`build.sh`는 PyYAML로 manifest 전체를 검증하고 모든 선택 모델을 임시 디렉토리에 먼저
복사한 뒤 성공한 경우에만 `model_repository`를 교체합니다. enabled source·required artifact가
없거나 `target`과 `config.pbtxt`의 `name`이 다르거나 선택 결과가 0개면 기존 repository를
수정하지 않고 실패합니다. `--env`는 각 manifest 항목의 선택적 `environments` 허용 목록을
적용하며, 이 필드를 생략한 모델은 세 환경 모두에 배치할 수 있습니다. `.env.*`는 Compose
설정이며 build 과정에서 shell script로 실행하지 않습니다.
