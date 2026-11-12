# CLAUDE.md

Guidance for working in this repository, plus the running log of tasks and progress.
Keep both halves current: when behaviour changes, update "How it works"; when work is
done or planned, update "Tasks and progress" (and add the matching entry to `checkpoint.md`).

## What this is

**larplarpsahur** (`com.larplarpsahur`, single-module Java Android app under `android/`) benchmarks
the speech-translation pipeline of [DS-PNQ/OnSpeak47](https://github.com/DS-PNQ/OnSpeak47)
(branch `hymt2+qwen3asr`): **Qwen3-ASR → Hy-MT2 translation**, both on the Hexagon NPU of a
Snapdragon 8 Elite Gen 5 (SM8850), usually a Qualcomm Device Cloud phone. It runs single audio
files or the Qualcomm "Round 3" sample dataset and scores the output in the app:

- ASR: WER, CER against the reference transcript.
- Translation: BLEU, chrF2, CER, character accuracy and exact match against the reference
  translation table that ships with the dataset.

There is no server and no Python in the product; everything a run needs is in the APK or
downloaded by the app on first launch.

## Layout

| Path | Contents |
|---|---|
| `android/app/src/main/java/com/larplarpsahur/` | This app: UI (`MainActivity`), foreground worker (`WorkService`), `dataset/` (download, scan, XLSX), `model/` (model download + SHA-256), `pipeline/` (audio decode, runner, metrics, export). |
| `android/app/src/main/java/com/omnivoice/onspeak47/` | **Vendored** OnSpeak47 sources (see below). Do not edit by hand. |
| `android/app/src/main/jni/` | Vendored `htpshared` JNI shim (QNN HTP shared-memory tensors) + ONNX Runtime C headers. |
| `android/app/src/main/assets/reference_tables/` | Round 3 translation tables from the organisers' Drive folder (fallback references). |
| `android/app/src/test/java/` | JVM unit tests: `com/larplarpsahur` (ours) and `com/omnivoice/onspeak47` (vendored upstream tests + `SegmentTranslatorTest`). |
| `android/ONSPEAK47_UPSTREAM` | Upstream repo, branch and commit the vendored files were copied from. Read by `build.gradle` into `BuildConfig.ONSPEAK47_COMMIT`. |
| `tools/sync_onspeak47.sh`, `tools/onspeak47-local.patch` | Re-vendoring script and the only local edits allowed inside vendored files. |
| `README.md` | User guide in Vietnamese (build, QDC SSH tunnel, running, pulling results). |
| `android_asr_mt_benchmark_plan.md` | Design plan in Vietnamese; §22–§23 describe what is actually implemented. |
| `checkpoint.md` | Task-by-task history, newest first. |

## Commands

All from `android/`. Needs JDK 21, Android SDK Platform 34, and (for the JNI shim) NDK
`28.2.13676358` + CMake `3.22.1`; `android/local.properties` (`sdk.dir=…`) is git-ignored.

```bash
./gradlew :app:assembleDebug :app:testDebugUnitTest :app:lintDebug --console=plain
```

- `-PnoHtpShared` builds without the NDK: the app falls back to plain direct buffers (same
  results, slower per token).
- `-PhymtTokenizer=/path/to/tokenizer.json` enables the five vendored BPE golden tests that
  otherwise skip (file: `tencent/Hy-MT2-1.8B` at the revision pinned in `build.gradle`).
- One test class: `./gradlew :app:testDebugUnitTest --tests 'com.larplarpsahur.TranslationAccuracyTest'`.
- APK: `android/app/build/outputs/apk/debug/app-debug.apk` (arm64-v8a only, never committed).
- Maven Central sometimes answers 429 through a proxy; re-run, nothing is wrong with the build.

JVM tests cover dataset joins, XLSX parsing, metrics, export/resume and the vendored pure-Java
logic. They do **not** exercise the NPU: inference only runs on an SM8850 device.

## How it works

1. **Models** (`ModelRepository`): first launch downloads Qwen3-ASR (release `v2.2.0-npu`, REV 1)
   and Hy-MT2-1.8B-1.25Bit (release `v2.1.0-npu`, REV 8) from `DS-PNQ/OnSpeak47-npu-models`, plus
   the Tencent tokenizer, each pinned by size and SHA-256. SM8850 + Android 12 only, no CPU fallback.
2. **Input**: picked audio files (`RUN`), the pinned Hugging Face ZIP `Echdipi/quadata`
   (`DATASET`), or a dataset folder already on the device (`LOCAL`, started with
   `am start … --ez auto_run true --es dataset_root …`).
3. **Scan** (`DatasetScanner`): Round 3 tracks by ID prefix — `VM-vi` (vi→zh), `VM-zh` (zh→vi),
   `VE-vi` (vi→en), `EV-en` (en→vi); `VK`/Vie-Kor is skipped. References are joined **by ID only**,
   never by row order, from: sidecar `.txt` → XLSX tables inside the dataset → the bundled tables
   in `assets/reference_tables` for IDs the dataset has no row for. `ref_source` records which.
4. **Run** (`PipelineRunner`): ASR for every sample, close Qwen, then translate every sample
   (two phases keep both NPU model sets out of memory together). Per-sample state is written
   atomically, so a paused or killed run resumes.
5. **Translate** (`SegmentTranslator`, in the vendored package): OnSpeak47's commit path —
   `ScriptLid` reads the source language from the transcript, a row already in the target
   language is copied, numbers/emails go to written form, text is cut into ≤200-char segments
   decoded with the 128-token commit budget, and the previous segment of the same audio is
   forced-prefix context.
6. **Score + export** (`Metrics`, `ResultExporter`): see below. Results ZIP is saved through
   the Android document picker.

### Translation accuracy

Scored against the reference translation, after the normalization profile
`nfc-lower-stripP-space-v1` (NFC, lowercase, Unicode punctuation removed, whitespace collapsed,
all spaces removed for Chinese). Corpus numbers are sums over samples, never averages of
per-sample scores.

| Metric | Definition |
|---|---|
| `bleu` | sacreBLEU-compatible, tokenizer `13a` (`zh` for Chinese), exp smoothing; corpus = fixed 4-gram, sentence = effective order. |
| `chrf` | sacreBLEU chrF2 defaults: character 1–6-grams without whitespace, β = 2, no word n-grams. |
| `cer` | Levenshtein distance over code points / reference length. |
| `accuracy` | `100 × (1 − CER)`, floored at 0 — share of reference characters reproduced. |
| `exact_match` | % of samples whose normalized translation equals the normalized reference. |

`TranslationAccuracyTest` checks BLEU and chrF against sacreBLEU golden values computed on the
real tables. Output files: `eval_mt.csv` (per sample + `TOTAL` rows), `eval_mt_summary.csv`
(one row per track and MT input), `eval_summary.json`, and the one-line summary on screen.

**Normalized reference tables.** The E→V table (`OneVoice_E_to_V_samples_100.xlsx`, headers
`Tiếng Anh (normalized)` / `Tiếng Việt (normalized)`) is lowercase, unpunctuated, with numbers
spelled out. A sample whose reference column header contains `normalized` gets
`ref_normalized = true` and is scored with the language code `vi+num` / `en+num`: digits in
both strings are spelled out (`Numbers`, profile `spell-digits-v1`) and symbols are dropped.
This is approximate — the table also uses readings no rule recovers from digits
(`sixteen forty five`, `eight hundred` for 8:00) — so expect residual errors on the ~13 % of
EV rows that contain numbers. The signature column says which profile scored a row.

### Dataset sources

- Organisers' Drive folder "Round 3-Bộ dữ liệu mẫu":
  <https://drive.google.com/drive/folders/1Gm8ziMP-XwbP5Ce1InVCusy-S0uNWf62>
  (Vie-Eng: 100 VE + 100 EV wav with one table each; Vie-Man: 180+180 wav, txt, one table;
  Vie-Kor: out of scope). Table IDs match the audio file names 100 %.
- The app's one-tap download is the pinned Hugging Face snapshot of that folder taken on
  2026-10-24, **before** the E→V table existed (its Eng-Vie folder holds a copy of the V→E
  table). That is why the three tables are bundled: with them EV audio from the snapshot is
  scored too. Tables found inside a dataset always win over bundled ones.
- To refresh the bundled tables, download the XLSX files from Drive into
  `android/app/src/main/assets/reference_tables/`, keep the file names, and re-run the tests
  (the golden numbers in `TranslationAccuracyTest` change if the text changed).

## Vendored OnSpeak47 code

`com.omnivoice.onspeak47` is copied from the upstream branch, not forked. Rules:

- **Never edit a vendored file directly.** Change `tools/onspeak47-local.patch` instead (it
  currently adds `QwenAsrNpuEngine.transcribeFileBlock` and stops `HyMtOnnxModule.close` from
  closing the process-wide `OrtEnvironment`).
- To move to a newer upstream commit: check out OnSpeak47 `hymt2+qwen3asr`, run
  `tools/sync_onspeak47.sh <checkout>`, fix the patch if it no longer applies, run the tests,
  and re-read `PipelineOrchestrator.translateSegmentWithDraft` upstream — `SegmentTranslator`
  mirrors it by hand and must follow it.
- If upstream changes model graphs (a new `REV` in `HyMtNpuDecoder` / `QwenAsrNpuEngine` or new
  release URLs in `LoadingActivity`), update the file list, sizes, SHA-256 and REV values in
  `ModelRepository` and the URLs in `android/gradle.properties`.
- Three files in that package are **ours**, kept there for package-private access:
  `asr/StreamingAsrEngine.java` (trimmed base class), `asr/QwenFileRecognizer.java`,
  `pipeline/SegmentTranslator.java` (+ `SegmentTranslatorTest`). The sync script never touches them.
- `TatoebaAssist` and `Glossary` come along because `TranslationModule` / `HyMtOnnxModule`
  reference them; no Tatoeba database is bundled and the glossary is empty, so both are inert.

## Conventions

- Java 11 source level, no Kotlin, no new dependencies without a reason. `com.larplarpsahur`
  code is written densely (several statements per line); match the file you are editing.
- UI strings are Vietnamese; code comments, identifiers and this file are English.
- Never commit APKs, keys, `local.properties`, model files or run outputs (`.gitignore`).
- Never claim on-device behaviour (latency, QNN loading, translation quality) from a JVM
  test or a build; say "not verified on device" until someone ran it on an SM8850.
- Bump `versionCode` / `versionName` in `android/app/build.gradle` for any behaviour change
  and record it in `checkpoint.md`.

## Tasks and progress

Status as of 2026-11-12, app version **1.1.0** (versionCode 5), upstream `499d54e` (2026-11-11).

### Done in this round (2026-11-12)

- [x] **Survey.** Established that the vendored code matched upstream `c9b9950` (2026-10-23) plus
  small local edits; upstream head is `499d54e`. Model releases and graph REVs are unchanged
  (Qwen `v2.2.0-npu` REV 1, Hy-MT2 `v2.1.0-npu` REV 8), so no model re-download is needed.
- [x] **Upstream sync.** Re-vendored 19 source files, the JNI shim and 5 upstream test classes
  with `tools/sync_onspeak47.sh`; local edits reduced to a two-hunk patch. Brings HTP
  shared-memory graph I/O (`HtpShared`, `NpuGraph`), the decode loop guard, MT forced-prefix
  context, the full `ScriptLid` (written-form numbers/emails, mixed-row routing), Qwen streaming
  fixes and the `cool_run` HTP mode.
- [x] **OnSpeak47 translation path.** `PipelineRunner` now translates through `SegmentTranslator`
  → `TranslationModule` instead of calling `HyMtOnnxModule` with fixed 180-char chunks and a
  256-token budget. Side effect: the 128-token commit budget lets short prompts use the AR1
  256-slot graph again (the 1.0.3 latency finding in checkpoint Task 7).
- [x] **Whole-file ASR.** `QwenFileRecognizer` cuts audio longer than 18 s at the quietest spot of
  the last 2 s (upstream `quietCut`) instead of a hard cut, and keeps the transcript both as
  written (scored) and in the app's script convention (MT input).
- [x] **Translation tables.** Scanner reads the new E→V table and header variants; the three
  Drive tables are bundled as fallback references; `EV-en` is now scored (it had no reference
  before). `dataset_scan.json` reports `reference_sources` per track.
- [x] **Translation accuracy.** Added chrF2, character accuracy, exact match, number spelling
  for normalized tables, `eval_mt_summary.csv`, and the on-screen ASR/MT summary.
- [x] **Tests.** 60 JVM tests (13 ours in `com.larplarpsahur`, 3 `SegmentTranslatorTest`,
  44 vendored upstream), all passing; BLEU/chrF verified against sacreBLEU on the real tables.
- [x] **Build.** `assembleDebug` produces a 43 MB arm64 APK containing `libhtpshared.so` and the
  bundled tables; `lintDebug` passes.
- [x] **Docs.** This file, `checkpoint.md` Tasks 9–12, README, plan §23.

### Not verified / open

- [ ] **On-device run of 1.1.0.** Nothing in this round ran on an SM8850. To confirm on QDC:
  logcat `HtpShared` should print `NPU graph I/O in HTP shared memory`; `HyMtOnnxModule` should
  show `mt context:` lines only for multi-segment audio; compare `asr_ms` / `mt_ms` and the
  metric tables with a 1.0.3 run of the same dataset.
- [ ] **Compare 1.0.3 vs 1.1.0 scores.** The translation path changed (script-based source
  language, copies, written-form input, budget), so BLEU/CER are not comparable across versions
  without re-running. Check `mt_copied` > 0 rows by hand: a copy means `ScriptLid` judged the
  transcript to be in the target language already.
- [ ] **Number spelling coverage.** Decide with real EV output whether clock times, years read
  in pairs and digit-by-digit codes need rules, or whether numeric rows should be reported
  separately.
- [ ] **Hugging Face snapshot is stale.** `DatasetDownloader` still pins the 2026-10-24 ZIP. A
  fresh snapshot of the Drive folder (with the E→V table inside) would make the bundled tables
  unnecessary; needs someone with write access to `Echdipi/quadata`, then update `REVISION`,
  the file name and `ZIP_SHA256`.
- [ ] **Vie-Kor.** Still skipped. Hy-MT2 and Qwen3-ASR know Korean, but upstream's prompt and
  routing (`HyMtOnnxModule.enName`, `ScriptLid`) only handle vi/en/zh.
- [ ] **Unused AAR.** `android/app/libs/sherpa-onnx-static-link-onnxruntime-1.13.8.aar` (37 MB)
  is not referenced by `build.gradle`; remove it unless a sherpa-based engine is planned.
