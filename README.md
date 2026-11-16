# Лабораторна №3 — CI/CD для SpeechCNN

Індивідуальний базовий варіант за методичкою. Класи `yes`, `no`, `up`, `down`, Google Speech Commands v0.02, FastAPI `/health`, `/model-info`, `/predict`.

## Послідовність

`pull_request / push / workflow_dispatch` → `lint` → `train` → `verify` → `build-push` (лише push main або ручний запуск з `push_image=true`).

- `train`: Docker trainer запускається через `docker run`, модель і `train.log` записуються в `artifacts/`, Actions artifact `model`. WAV із незалежної тестової вибірки передаються окремим артефактом `golden-samples`; очікувані класи зберігаються в `samples/expected.json`.
- `verify`: завантажує артефакти, запускає FastAPI, перевіряє всі чотири WAV через HTTP `/predict` (помилка = exit 1), поріг accuracy 60% для PR і 80% для main, виконує 100 запитів після 10 прогрівальних, формує `metrics/report.md`, `metrics/latency.json`, `metrics/verify.json`, `metrics/service.log` і Job Summary.
- `build-push`: лише після успішного `verify` публікує образ у GHCR з SHA-тегом та `latest` на main. Використовує GITHUB_TOKEN з `packages: write` лише у цьому job.

## Локально (PowerShell, із папки проєкту)

```powershell
docker build --target trainer -t kws-trainer:ci .
New-Item -ItemType Directory -Force artifacts,data,samples | Out-Null
docker run --rm -v "${PWD}/artifacts:/artifacts" -v "${PWD}/data:/data" kws-trainer:ci --classes yes no up down --epochs 10 --seed 42 --out /artifacts
docker run --rm --entrypoint python -v "${PWD}/data:/data" -v "${PWD}/samples:/samples" kws-trainer:ci -m scripts.make_samples
docker build --target runtime -t kws-api:ci .
docker run -d --name kws -p 8000:8000 kws-api:ci
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/model-info
curl.exe -X POST -F "file=@samples/yes.wav" http://localhost:8000/predict
python scripts/verify_model.py --expected samples/expected.json --out metrics/verify.json
python scripts/benchmark.py --n 100 --warmup 10 --out metrics/latency.json
python scripts/make_report.py --out metrics/report.md
```

Перший запуск завантажує великий датасет. Еталонні WAV беруться з тестової вибірки, не з навчальної. Модель може помилитися на окремому WAV — тоді `verify` правильно зупиниться; потрібно покращувати модель або обґрунтовано вибрати стабільні незалежні тестові записи, а не підробляти результати.

## GitHub: обов'язкові ручні дії

1. Завантажити вміст папки в репозиторій (включно з `.github`) і перевірити, що гілка `main`.
2. В Actions перевірити три події: PR (2 епохи, без публікації), push main (10 епох, публікація), Run workflow (поле epochs і перемикач push_image).
3. Settings → Rules → Rulesets або Branches: для `main` увімкнути Require pull request before merging і Require status checks to pass; вибрати `lint`, `train`, `verify` після першого запуску. Перевірити блокування PR із помилковим `expected.json`, потім відкотити помилку.
4. У Settings → Actions → General перевірити workflow permissions. Токен не вводити в код. Перевірити `docker history --no-trunc`.
5. Зробити справжні скріншоти трьох запусків, артефактів model/report, Job Summary, GHCR, правил захисту, невдалого verify і заблокованого Merge.
6. Заповнити таблицю результатів за реальними запусками: тригер, епохи, accuracy %, latency mean/p95 мс, розмір моделі КБ, тривалість пайплайна хв, образ опубліковано.

## Важливі обмеження

Це вихідний код, а не підтвердження успішного GitHub Actions запуску. Пороги 60%/80% — початкові, потребують перевірки та обґрунтування. Перші незалежні тестові WAV не гарантовано правильно розпізнаються. Опціональні командні завдання 3.1–3.9 не включені.

## PR test
Automatic CI checks for pull requests.
