# Checkpoint

## Task 13 — Evaluate the first 1.1.0 device run (2026-11-12)
- Status: complete. Report: `reports/2026-11-12-quadata-v1.1.0.md`; script: `tools/rescore_results.py`.
- Input: `larplarpsahur-results.zip` from QDC ("Canoe for arm64", SM8850), app 1.1.0, upstream `499d54ee`, quadata `fed18c73`, `mt_mode=pipeline`, track targets on. 560 samples (VE-vi 100, VM-vi 180, VM-zh 180, EV-en 100), DONE, 0 errors, 539.8 s wall-clock for 2,124 s of audio. EV references came from the bundled E→V table (100/100).
- Metric check: sacreBLEU 2.x on the app's normalised strings reproduces the app's corpus BLEU and chrF for VE-vi, VM-vi and VM-zh to all printed digits; EV differs by 0.6 BLEU because the script's number spelling also says "giờ" for clock times.
- ASR: app CER VE-vi 1.71 %, VM-vi 13.57 %, VM-zh 3.89 %, EV-en 6.49 %. Format-neutral (digits, 她/他, 的/得/地, fillers, contractions) and without four truncated references: 1.27 %, 8.05 %, 0.51 %, 5.85 % (all 5.41 %). Language tags 558/560 right; EV-2748 produced an empty transcript.
- MT: app BLEU/chrF VE 31.8/56.2, VM-vi 33.7/34.0 (Chinese target), VM-zh 34.4/59.4, EV 30.1/54.1. Sentences with a correct transcript score 6.9–12.1 BLEU higher than those with ASR errors. Hand spot check of 48 random samples with correct ASR: 41 correct or acceptable, 4 translation errors, 3 broken by one-letter ASR slips.
- Main findings: *anh/chị* = "you" translated as he/she in 8/16 VE sentences; 22/33 VM-zh sentences about a woman translated as male because the ASR writes 他 for *tā*; one hallucinated entity (越南 → Thái Lan, VM-1730). Dataset: 4 truncated `vie_text` rows (30 % of VM-vi ASR edits), one vi/zh mismatch (VM-4999), spoken-style E→V references.
- Speed: ASR median 247–449 ms per sample (real-time factor 0.08–0.10), MT median 340–930 ms; model load 0.7 s + 1.2 s.
- Not done: logcat was not provided, so `HtpShared` / MT-context log lines are unverified; no oracle (`mt_mode=both`) run yet.

## Task 12 — Documentation and progress tracking (2026-11-12)
- Status: complete.
- Result: added `CLAUDE.md` (layout, commands, pipeline, metric definitions, vendoring rules, task list with open items), updated README (build requirements, translation path, EV references, new result files, running the Drive folder from device storage) and plan §23. `CLAUDE.md` "Tasks and progress" is the live list from now on; this file keeps the history.

## Task 11 — Translation accuracy against the Round 3 translation tables (2026-11-12)
- Status: complete in code and JVM tests; not run on a device.
- Source: organisers' Drive folder "Round 3-Bộ dữ liệu mẫu" (`1Gm8ziMP-XwbP5Ce1InVCusy-S0uNWf62`). Tables: `OneVoice_V_to_E_samples_100.xlsx` (sheet Final: ID | Tiếng Việt | Tiếng Anh - Final), `OneVoice_E_to_V_samples_100.xlsx` (sheet Final: ID | Tiếng Anh (normalized) | Tiếng Việt (normalized)), `OneVoice_V_to_M_samples_180.xlsx` (sheet Samples: id | vie_text | man_text | paths), `OneVoice_V_to_K_samples_180.xlsx` (out of scope). Every table ID has its audio file and vice versa (100/100, 100/100, 180/180 ×2).
- Finding: the E→V table is new. The pinned Hugging Face ZIP (snapshot of 2026-10-24) has a copy of the V→E table in the Eng-Vie folder, which is why EV-en had no reference in 1.0.x. The E→V table is normalized text: lowercase, no punctuation, numbers spelled out; 13 of 100 rows contain numbers, several with readings that cannot be derived from digits.
- Implementation: `DatasetScanner` matches text columns by header prefix (so `Tiếng Anh (normalized)` etc. are found), no longer excludes EV from table joins, and falls back to the tables bundled in `assets/reference_tables` (SHA-256 `b7088a1d…` E_to_V, `66794c82…` V_to_E, `b0b9f1ba…` V_to_M) for IDs the dataset has no row for. Join stays ID-only. Samples carry `ref_source` (sidecar / dataset_table / bundled_table / manifest / none) and `ref_normalized`.
- Metrics added: chrF2 (sacreBLEU defaults), character accuracy `100*(1-CER)` floored at 0, exact match after normalization, alongside the existing BLEU and CER. Normalized-table samples are scored with digits spelled out and symbols dropped (`Numbers`, profile `spell-digits-v1`) for both ASR (WER/CER) and MT.
- Output: `eval_mt.csv` gains accuracy, sent_chrf, exact_match, chrf, samples, ref_source; new `eval_mt_summary.csv` (one row per track and MT input); `eval_summary.json` records the chrF signature and accuracy definition; the completion message lists ASR CER/WER and one translation line per track.
- Verification: BLEU and chrF match sacreBLEU to 1e-6 on all four reference columns of the real tables with synthetic hypotheses (`TranslationAccuracyTest`). First attempt at chrF was off by 0.06 on Chinese: sacreBLEU does not count hypothesis n-grams of an order the reference is too short for; fixed.
- Limits: number spelling is plain cardinals (+ clock split, thousands groups, decimals); numeric EV rows remain approximate. Hugging Face CDN was not reachable from the work environment, so the pinned ZIP's own tables were not compared byte-for-byte with the Drive ones.

## Task 10 — Integrate OnSpeak47's translation path (2026-11-12)
- Status: complete in code and JVM tests; not run on a device.
- Before: `PipelineRunner` called `HyMtOnnxModule.translateSentenceWithBudget` directly on 180-character chunks with a 256-token budget, source language = Qwen's tag.
- Now: `SegmentTranslator` (local file in the vendored package) reproduces `PipelineOrchestrator.translateSegmentWithDraft` without drafts/TTS: `ScriptLid.label` decides the source language from the transcript, a row already in the target language is copied, a mixed row is prompted from its minority language, `lowerEnglish` + `writtenForm` (+ `viMtForm` for Vietnamese) shape the input, and `TranslationModule.translateWithDraft` decodes with `AsrState.COMMIT_MT_MAX_TOKENS` (128; 256 only for a single sentence over 200 chars). Transcripts over 200 chars are packed by sentence, and each segment gets the previous one as forced-prefix context, as upstream does between commits of a session.
- ASR side: `QwenFileRecognizer` keeps two forms of the transcript — as written (`asr_hyp`, scored) and in the app's script convention (`asr_routed`, English words uppercase, which is what `ScriptLid` reads). Oracle mode applies the same convention to the reference transcript using its known language. Audio over 18 s is cut at the quietest 100 ms of the last 2 s instead of at a fixed sample.
- A paused translation returns partial text from the NPU decoder; the runner re-checks cancellation and never stores it.
- New per-sample fields: `asr_routed`, `mt_src_lang`, `mt_segments`, `mt_copied`; run config records `onspeak47_commit` and `translation_path`.
- Consequence: scores from 1.0.x and 1.1.0 are not comparable without re-running; the token budget change should also let short prompts decode on the AR1 256-slot graph (Task 7 finding). Neither is measured yet.

## Task 9 — Sync vendored OnSpeak47 code to upstream `499d54e` (2026-11-12)
- Status: complete; builds and passes JVM tests; not run on a device.
- Finding: vendored files matched upstream `c9b9950` (2026-10-23; `HyMtNpuDecoder` `9040143`, `HyMtOnnxModule` `4cb9a81`) with three local edits. Upstream branch `hymt2+qwen3asr` is at `499d54ee` (2026-11-11). Model releases and REVs are unchanged (`v2.2.0-npu` Qwen REV 1, `v2.1.0-npu` Hy-MT2 REV 8): no change to `ModelRepository`.
- Method: `tools/sync_onspeak47.sh <checkout>` copies 19 main files, 5 JNI files and 5 test classes verbatim, applies `tools/onspeak47-local.patch`, and writes `android/ONSPEAK47_UPSTREAM`. Local edits are now two hunks: `QwenAsrNpuEngine.transcribeFileBlock` and not closing the shared `OrtEnvironment` in `HyMtOnnxModule.close`. The earlier edit that stripped glossary/Tatoeba from the prompt is gone: `Glossary`, `TatoebaText`, `TatoebaAssist` and `TranslationModule` are vendored instead and stay inert (empty glossary, no database).
- What the sync brings: NPU graph I/O in QNN HTP shared memory (`HtpShared` JNI + `NpuGraph`, falls back to direct buffers when the library is missing), `LoopGuard` on the decode loop, MT forced-prefix context, UTF-8/newline cleanup of MT output, email preservation, the full `ScriptLid`, Qwen streaming fixes (dropped audio before the 20 s cut, name lock, quiet cut), `cool_run` HTP performance mode.
- Build: `externalNativeBuild` (NDK 28.2.13676358, CMake 3.22.1) builds `libhtpshared.so`; `-PnoHtpShared` skips it. `unitTests.returnDefaultValues = true` for the vendored tests. Version 1.1.0 (versionCode 5).
- Verification (Linux, JDK 21, Gradle 9.5.1 wrapper): `:app:assembleDebug :app:testDebugUnitTest :app:lintDebug` succeeded; 60 tests, 0 failures, 0 skipped with `-PhymtTokenizer` (5 BPE golden tests skip without it). APK 43,001,443 bytes, contains `lib/arm64-v8a/libhtpshared.so` and `assets/reference_tables/*.xlsx`. APK not published; build locally.
- Not done: no SM8850 run, so shared-memory I/O, latency and translation quality of 1.1.0 are unverified.

## Task 8 — Publish GitHub repository and usage guide (2026-10-24)
- Status: complete.
- Scope: create private GitHub repository Model-eval-on-comp-dataset and Vietnamese README covering APK build, installation, QDC SSH setup, dataset processing and results retrieval.
- Preparation: no existing Git repository or GitHub CLI; use authenticated browser to create the repository and existing Git credentials for push if available. Exclude keys, outputs, downloaded assets and local machine configuration.
- Result: created `https://github.com/hoangducbao/Model-eval-on-comp-dataset` as a private repository and pushed initial source commit `3de5626`. README documents Windows build, debug APK installation, secure per-session PEM handling, QDC SSH tunnel on local port 5038, ADB operations, app behavior, result ZIP export, `adb pull`, direct recovery path and expected error handling. Git ignores private keys, keystores, APKs, downloaded/build output and local Android configuration.

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
