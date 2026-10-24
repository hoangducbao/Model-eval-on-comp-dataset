# Checkpoint

## Task 8 — Publish GitHub repository and usage guide (2026-10-24)
- Status: in progress.
- Scope: create private GitHub repository Model-eval-on-comp-dataset and Vietnamese README covering APK build, installation, QDC SSH setup, dataset processing and results retrieval.
- Preparation: no existing Git repository or GitHub CLI; use authenticated browser to create the repository and existing Git credentials for push if available. Exclude keys, outputs, downloaded assets and local machine configuration.

## Task 7 — Investigate translation latency and refine run controls (2026-10-24)
- Status: complete.
- Scope: compare translation execution with the previous APK/reference implementation, explain measured versus apparent latency, merge pause/resume into one control, and verify/fix manual-audio target selection and dataset language indicators.
- Starting evidence: screenshot shows quadata sample 205/560, top selector English, active track target Vietnamese; the current UI treats missing MT timing as “processing” and hides track-target behavior.
- Findings: fixed 256-token budget makes `ids.length + budget <= 255` impossible in HyMtNpuDecoder; every decoding step uses the larger AR64 graph instead of AR1. This was already present before the 1.0.2 UI change. Reference app supports adaptive budgets and shared/reused NPU buffers; this minimal app uses per-step allocations. No matched on-device timing comparison is available, so no quantitative regression claim is justified.
- Implementation: unified pause/resume uses persisted request or per-sample checkpoint, including pause during download/import; manual language selector persists immediately, dataset track behavior is labelled explicitly; export uses a bright enabled color only when result files exist and work is idle. Pending MT now reads “chưa có” instead of incorrectly claiming inference is underway.
- Verification: assembleDebug, lintDebug and 7 unit tests passed, including cancelled/completed checkpoint eligibility. Device UI and matched performance timing remain unverified. Exported `outputs/larplarpsahur-debug-1.0.3-controls.apk`, SHA-256 `15c53ec842e4d2c6b78cbecdd450df4c731668b3617930707631d805ba002e87`.

## Task 6 — Refine the ASR / translation workspace UI (2026-10-24)
- Status: complete.
- Scope: replace the form-style screen with a focused, dark two-panel workspace: live audio/latency line, separate ASR and translation outputs with language indicators, and clear actions for individual upload, pause, quadata download/process, and ZIP export.
- Starting findings: the current screen combines ASR and translation in one output field and requires a separate run button after selecting audio; the service already produces per-sample language, ASR latency and MT latency data, so no inference pipeline change is required.
- Result: `Upfile` now opens the audio picker and starts the selected audio pipeline immediately. The upper ASR panel shows detected language and transcript; the lower panel shows the target language and translation. The line above them displays the active file identifier and each measured processing latency. `Tạm dừng` uses the same cooperative cancellation path for manual and quadata work; the Qualcomm-labelled action continues to fetch the pinned Hugging Face quadata ZIP.
- Verification: `:app:assembleDebug`, `:app:testDebugUnitTest` (7 tests), and `:app:lintDebug` passed. APK: `outputs/larplarpsahur-debug-1.0.2-ui.apk`, versionCode 3 / versionName 1.0.2, SHA-256 `b353988fbbfc8b9a38404adc528d86e3d6ed2ed35d5f048df1890a1669110542`.

## Task 5 — Replace Google Drive dataset source with Hugging Face quadata (2026-10-24)
- Status: complete.
- Scope: download only the public Round 3 ZIP from Echdipi/quadata; ignore duplicate files outside that ZIP, retain validation/extraction/scan/export behavior, update the UI and build a replacement APK.
- Verified source: public tree lists `Round 3-Bộ dữ liệu mẫu-20261024T014123Z-1-001.zip` at revision `fed18c73b4350e07a02124df4f4c716a95e26743`, LFS SHA-256 `cde6b58e391094c7578a7bdd37a8202db0735ccd165f91cc18c6f7ded1f8f01e`.
- Result: removed Drive URL/token/SHA UI and Drive downloader. The sole dataset action downloads the pinned Hugging Face ZIP over HTTPS, verifies its pinned hash, then extracts/scans it. Build/lint/test passed (7 tests).
- Artifact: `outputs/larplarpsahur-debug-1.0.1-quadata.apk`, versionCode 2 / versionName 1.0.1, arm64-v8a, SHA-256 `7a28243e4f557a4a216671dbc99bf9bceea80090c5418c2fad34189211b05d11`.

## Task 1 — Inspect plan and reference app
- Status: complete
- Findings: `qahapp` contains only `android_asr_mt_benchmark_plan.md`; the reference is a single-module Java Android app using sherpa-onnx streaming ASR, ONNX Runtime HY-MT, and SHA-256 validated model downloads.

## Task 2 — Implement `larplarpsahur` and dataset workflow
- Status: complete
- Result: implemented the minimal Java Android app, Input/Output-first workflow, first-launch SHA-256 model provisioning, Drive ZIP ingestion, safe extraction, Round 3 ID pairing, export/resume and configurable dataset run path.

## Task 3 — Resume and complete integration (2026-10-24)
- Status: complete.
- Starting findings: Android sources and Gradle wrapper exist; no completed APK has been confirmed. Previous compile output was truncated because the command session id was discarded. The current scanner/exporter does not yet meet the Round 3 reference matching and corpus metric requirements. The model installer incorrectly used local checksum cache names as remote asset names; the reference downloads `SHA256SUMS` from each release.
- Work: correct model provisioning, complete dataset/reference matching and export, retain background work safely, build APK, document evidence and remaining device-only verification.
- User clarification: retain a basic Input box and Output box as the main UI; file selection and explicit Run action above the optional Drive workflow.
- Verified GitHub release metadata: required model graphs and weights exist; remote manifests are named SHA256SUMS. HY-MT tokenizer is absent from these releases, so use the matching Tencent tokenizer at a pinned commit and hash, with a configurable mirror URL.
- Integration implemented: explicit Input/Run/Output UI, foreground worker, first-launch download validation/retry, Drive ZIP import, ID-based XLSX/TXT/manifest references, two-phase processing and recoverable per-sample exports. Verification is underway.
- Result: native runtime is used only for Qwen/HY-MT; inference processes Qwen then HY-MT to limit memory. OrtEnvironment remains alive across runs so a completed run does not break a later run.

## Task 4 — Verify and export APK (2026-10-24)
- Status: complete.
- Starting checks: compile against the existing QNN/ORT dependency versions; test dataset matching, archive safety, metric totals, JSON/CSV export and resume; produce an installable arm64 APK.
- Device limitation: native ASR and translation require an SM8850 cloud/physical device. Do not claim live inference has passed without running on that device.
- Verification: :app:assembleDebug :app:lintDebug :app:testDebugUnitTest succeeded. Unit tests cover Unicode/corpus metrics, BLEU edge cases, Round 3 direction/reference joins, XLSX sparse/shared/inline strings, traversal/duplicate archive rejection, result export, ZIP and resume.
- Artifact: outputs/larplarpsahur-debug-1.0.0.apk, 43,941,768 bytes, SHA-256 1bacaaaf76e20d2a909d91004d3e5bc2f5eb6004eec54bf87ca48ffa67d2af56; verified package com.larplarpsahur, API 27+, arm64-v8a.
- Remaining: run a real smoke test on an SM8850/QNN device with the intended dataset and Drive credential.
