#!/usr/bin/env bash
# Re-vendor the OnSpeak47 sources this app reuses.
#
#   tools/sync_onspeak47.sh /path/to/OnSpeak47-checkout   (branch hymt2+qwen3asr)
#
# Copies the files listed below verbatim, then re-applies tools/onspeak47-local.patch
# (the only local edits allowed inside vendored files). Files that exist only here
# (LOCAL_ONLY) are never touched. Records the upstream commit in android/ONSPEAK47_UPSTREAM.
set -euo pipefail
UP=${1:?usage: tools/sync_onspeak47.sh <OnSpeak47 checkout>}
HERE=$(cd "$(dirname "$0")/.." && pwd)
SRC=android/app/src
PKG=java/com/omnivoice/onspeak47

MAIN_FILES=(
  util/FileUtils.java util/NpuStats.java util/OrtSessionConfig.java
  util/HtpShared.java util/NpuGraph.java util/LoopGuard.java
  asr/AsrLanguage.java asr/AsrModelType.java asr/AsrState.java
  asr/QwenAsr.java asr/QwenAsrNpuEngine.java asr/ScriptLid.java
  pipeline/HyMtBpeTokenizer.java pipeline/HyMtNpuDecoder.java pipeline/HyMtOnnxModule.java
  pipeline/Glossary.java pipeline/TatoebaText.java pipeline/TatoebaAssist.java
  pipeline/TranslationModule.java
)
TEST_FILES=(
  asr/QwenAsrTest.java asr/ScriptLidTest.java
  pipeline/GlossaryTest.java pipeline/HyMtOnnxTest.java pipeline/MtContextTest.java
)
JNI_FILES=(CMakeLists.txt htpshared.c include/onnxruntime_c_api.h
  include/onnxruntime_ep_c_api.h include/onnxruntime_error_code.h)
# Local-only files living in the vendored package (need its package-private API):
#   asr/StreamingAsrEngine.java   trimmed base class (upstream's pulls in sherpa-onnx engines)
#   asr/QwenFileRecognizer.java   whole-file recognition on top of QwenAsrNpuEngine
#   pipeline/SegmentTranslator.java  OnSpeak47's commit translation path without TTS/UI

copy() { mkdir -p "$(dirname "$2")"; cp "$1" "$2"; }
for f in "${MAIN_FILES[@]}"; do copy "$UP/$SRC/main/$PKG/$f" "$HERE/$SRC/main/$PKG/$f"; done
for f in "${TEST_FILES[@]}"; do copy "$UP/$SRC/test/$PKG/$f" "$HERE/$SRC/test/$PKG/$f"; done
for f in "${JNI_FILES[@]}"; do copy "$UP/$SRC/main/jni/$f" "$HERE/$SRC/main/jni/$f"; done

if [ -s "$HERE/tools/onspeak47-local.patch" ]; then
  git -C "$HERE" apply --whitespace=nowarn tools/onspeak47-local.patch
fi
{
  echo "repo: https://github.com/DS-PNQ/OnSpeak47"
  echo "branch: $(git -C "$UP" rev-parse --abbrev-ref HEAD)"
  echo "commit: $(git -C "$UP" rev-parse HEAD)"
  echo "commit_date: $(git -C "$UP" log -1 --format=%cI)"
} > "$HERE/android/ONSPEAK47_UPSTREAM"
echo "Synced to $(git -C "$UP" rev-parse --short HEAD). Now run the unit tests."
