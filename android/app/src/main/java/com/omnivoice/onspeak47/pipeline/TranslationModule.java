/*
 * OmniVoice — Translation Module (Tencent Hy-MT1.5, enhanced INT8 ONNX only)
 *
 * Single backend: the enhanced INT8 KV-cache ONNX model
 * (model_int8_final.onnx + .onnx.data, ~2.1 GB, from
 * niedev/OnnxModelsEnhancer) run by plain ONNX Runtime via
 * {@link HyMtOnnxModule} — same prompting contract (instruction +
 * hy_User/hy_Assistant framing, BOS prepended by hand, greedy decode,
 * stop on EOS 120020, one call per transcript).
 *
 * The 2.1 GB weights cannot ship inside the APK: they are loaded from
 * getExternalFilesDir("hymt_int8/") — stage with
 * `python optimize/09_stage_hymt_int8.py --adb-push`.
 *
 * No fallback: when the ONNX asset is not staged, translate() returns an
 * explicit error string. A new fallback is planned separately — do NOT
 * re-add the GGUF path here.
 */

package com.omnivoice.onspeak47.pipeline;

import android.content.Context;
import android.icu.text.BreakIterator;
import android.util.Log;

import java.io.File;
import java.util.ArrayList;
import java.util.Locale;

import com.omnivoice.onspeak47.asr.AsrState;

public class TranslationModule {

    private static final String TAG = "TranslationModule";

    private static final int MAX_OUTPUT_TOKENS = 128;
    private static final int MAX_OUTPUT_TOKENS_LONG = 256;
    // Single-call threshold: ASR transcripts (~1-2 sentences, <400 chars) go
    // through ONE prefill. Only long paragraphs pay per-sentence prefills.
    // Mirrors RTranslator's join-up-to-maxLength strategy (5000 tok) instead
    // of the old split-every-sentence loop that multiplied prefill cost.
    private static final int SINGLE_CALL_CHAR_THRESHOLD = 400;

    // Obsolete dual-profile extraction (see class comment); reclaimed on boot.
    private static final String DEAD_INT4_DIR = "hymt_int4_onnx";

    private HyMtOnnxModule onnxModule;
    private TatoebaAssist tatoebaAssist;

    public TranslationModule(Context context) throws Exception {
        File baseDir = context.getExternalFilesDir(null);
        if (baseDir == null) baseDir = context.getFilesDir();

        // Reclaim the ~1.25 GB INT4 extraction written by the old dual-profile
        // builds on devices that still have it on disk.
        File wasted = new File(baseDir, DEAD_INT4_DIR);
        if (wasted.isDirectory()) {
            Log.i(TAG, "Removing obsolete INT4 model extraction: " + wasted);
            deleteRecursive(wasted);
        }

        // Tatoeba assist (dictionary + TM few-shot): best-effort, never
        // fatal — a missing/corrupt DB only means unaided translation.
        try {
            TatoebaAssist assist = new TatoebaAssist(context);
            if (assist.isAvailable()) tatoebaAssist = assist;
            else Log.i(TAG, "tatoeba assist absent — translations continue unaided");
        } catch (Exception e) {
            Log.w(TAG, "tatoeba assist init failed — continuing unaided", e);
        }

        // Single backend: enhanced INT8 ONNX (external files).
        // No fallback — a missing stage is a hard error with staging
        // instructions, never a silent GGUF downgrade.
        HyMtOnnxModule onnx = new HyMtOnnxModule(context);
        if (!onnx.isReady()) {
            String err = onnx.getInitError();
            onnx.close();
            throw new IllegalStateException(
                    "ONNX translation backend unavailable (" + err + ")"
                    + " — stage with: python optimize/09_stage_hymt_int8.py --adb-push");
        }
        onnxModule = onnx;
        Log.i(TAG, "Using enhanced INT8 ONNX translation backend");
    }

    private static void deleteRecursive(File f) {
        File[] children = f.listFiles();
        if (children != null) {
            for (File c : children) deleteRecursive(c);
        }
        if (!f.delete()) {
            Log.w(TAG, "Could not delete " + f);
        }
    }

    // ----------------------------------------------------------------
    // Public API (unchanged contract)
    // ----------------------------------------------------------------

    public TranslationResult translate(String text, String srcLang, String tgtLang) {
        long startTime = System.currentTimeMillis();
        if (text == null || text.trim().isEmpty()) {
            return new TranslationResult("", 0);
        }
        String cleaned = correctText(text);
        // Fast path (common ASR case): ONE LLM call, ONE prefill.
        if (cleaned.length() <= SINGLE_CALL_CHAR_THRESHOLD) {
            String out = translateSentence(cleaned, srcLang, tgtLang);
            Log.i(TAG, "translate single-call chars=" + cleaned.length()
                    + " (" + (System.currentTimeMillis() - startTime) + "ms)");
            return new TranslationResult(out, System.currentTimeMillis() - startTime);
        }
        // Slow path (long paragraph): per-sentence calls.
        ArrayList<String> sentences = splitIntoSentences(cleaned, srcLang);
        StringBuilder result = new StringBuilder();
        for (String sentence : sentences) {
            if (sentence.trim().isEmpty()) continue;
            String translated = translateSentence(sentence, srcLang, tgtLang);
            if (result.length() > 0) result.append(" ");
            result.append(translated);
        }
        Log.i(TAG, "translate multi-call sentences=" + sentences.size()
                + " (" + (System.currentTimeMillis() - startTime) + "ms)");
        return new TranslationResult(result.toString(), System.currentTimeMillis() - startTime);
    }

    // ----------------------------------------------------------------
    // HY-MT prompting (decoder-only: instruction instead of lang tokens)
    // ----------------------------------------------------------------

    private String translateSentence(String text, String srcLang, String tgtLang) {
        // Tatoeba assist first: confident dictionary hits bypass the model,
        // else TM examples ride along in the instruction.
        java.util.List<String[]> examples = java.util.Collections.emptyList();
        if (tatoebaAssist != null) {
            try {
                TatoebaAssist.AssistResult assist =
                        tatoebaAssist.assist(text, srcLang, tgtLang);
                if (assist.dictHit) {
                    Log.i(TAG, "tatoeba dict hit [" + srcLang + "->" + tgtLang + "]");
                    return assist.direct;
                }
                examples = assist.examples;
            } catch (Exception e) {
                Log.w(TAG, "tatoeba assist failed — continuing unaided", e);
            }
        }
        return onnxModule.translateSentence(text, srcLang, tgtLang, examples);
    }

    /** Token budget by input length: short ASR -> 48-128, long -> 256 cap. */
    private static int estimateMaxTokens(String text) {
        int len = text == null ? 0 : text.length();
        if (len <= 60) return 64;
        if (len <= SINGLE_CALL_CHAR_THRESHOLD) return MAX_OUTPUT_TOKENS;
        return MAX_OUTPUT_TOKENS_LONG;
    }

    /**
     * Budget-capped translation for streaming re-translation (no model change).
     * Same single-call contract as translate(), but the caller caps the
     * generation budget: live tails use 64, commits use 128.
     */
    public TranslationResult translateWithBudget(String text, String srcLang,
                                                String tgtLang, int maxTokens) {
        long startTime = System.currentTimeMillis();
        if (text == null || text.trim().isEmpty()) {
            return new TranslationResult("", 0);
        }
        String cleaned = correctText(text);
        String out = translateSentenceWithBudget(cleaned, srcLang, tgtLang, maxTokens);
        Log.i(TAG, "translate budgeted chars=" + cleaned.length()
                + " budget=" + maxTokens
                + " (" + (System.currentTimeMillis() - startTime) + "ms)");
        return new TranslationResult(out, System.currentTimeMillis() - startTime);
    }

    private String translateSentenceWithBudget(String text, String srcLang,
                                              String tgtLang, int maxTokens) {
        java.util.List<String[]> examples = java.util.Collections.emptyList();
        if (tatoebaAssist != null) {
            try {
                TatoebaAssist.AssistResult assist =
                        tatoebaAssist.assist(text, srcLang, tgtLang);
                if (assist.dictHit) {
                    Log.i(TAG, "tatoeba dict hit [" + srcLang + "->" + tgtLang + "]");
                    return assist.direct;
                }
                examples = assist.examples;
            } catch (Exception e) {
                Log.w(TAG, "tatoeba assist failed — continuing unaided", e);
            }
        }
        int budget = maxTokens > 0
                ? Math.min(maxTokens, MAX_OUTPUT_TOKENS_LONG) : estimateMaxTokens(text);
        // ONNX generate() is synchronized internally; no extra lock needed
        // now the GGUF native context is gone.
        return onnxModule.translateSentenceWithBudget(text, srcLang, tgtLang,
                examples, budget);
    }

    /**
     * SSBD-assisted translation (Stage A+B, ONNX-only): the previous output
     * (as token ids) is verified against the grown instruction in one forward
     * pass; generation resumes from the first divergence with retained KV.
     * Gated by {@code enabled}: when false, or when the draft is unusable,
     * falls back to plain budgeted translation (identical output contract).
     * Biased verification uses {@link AsrState#SSBD_BETA} (paper §3.3, 0.2).
     * SSBD stays OFF by default ({@link AsrState#SSBD_ENABLED}); plain path
     * is byte-identical to pre-SSBD.
     */
    public SsbdTranslationResult translateWithDraft(String text, String srcLang, String tgtLang,
                                                   int[] draftIds, int maxTokens, boolean enabled) {
        return translateWithDraft(text, srcLang, tgtLang, draftIds, maxTokens, enabled,
                AsrState.SSBD_BETA);
    }

    /**
     * Full Stage-B entry point with explicit beta for A/B tests.
     * {@code beta<=0} = unbiased Stage A; {@code 0.2} = paper sweet spot.
     * {@code draftExamplesKey} is the fingerprint stored with the draft; when
     * it differs from this call's examples the prompt differs too, so the
     * draft is rejected WITHOUT paying a verify forward (plain path + a
     * "ssbd skipped: examples changed" line). Pass "" when unknown.
     */
    public SsbdTranslationResult translateWithDraft(String text, String srcLang, String tgtLang,
                                                   int[] draftIds, int maxTokens, boolean enabled,
                                                   float beta) {
        return translateWithDraft(text, srcLang, tgtLang, draftIds, "",
                maxTokens, enabled, beta);
    }

    public SsbdTranslationResult translateWithDraft(String text, String srcLang, String tgtLang,
                                                   int[] draftIds, String draftExamplesKey,
                                                   int maxTokens, boolean enabled, float beta) {
        long startTime = System.currentTimeMillis();
        if (text == null || text.trim().isEmpty()) {
            return new SsbdTranslationResult("", 0, 0, 0, false, "");
        }
        String cleaned = correctText(text);
        java.util.List<String[]> examples = java.util.Collections.emptyList();
        if (tatoebaAssist != null) {
            try {
                TatoebaAssist.AssistResult assist =
                        tatoebaAssist.assist(cleaned, srcLang, tgtLang);
                if (assist.dictHit) {
                    Log.i(TAG, "tatoeba dict hit [" + srcLang + "->" + tgtLang + "]");
                    return new SsbdTranslationResult(assist.direct, 0, 0,
                            System.currentTimeMillis() - startTime, false, "");
                }
                examples = assist.examples;
            } catch (Exception e) {
                Log.w(TAG, "tatoeba assist failed — continuing unaided", e);
            }
        }
        String key = examplesKey(examples);
        int budget = maxTokens > 0
                ? Math.min(maxTokens, MAX_OUTPUT_TOKENS_LONG) : estimateMaxTokens(cleaned);
        if (!enabled || draftIds == null || draftIds.length == 0) {
            String out = onnxModule.translateSentenceWithBudget(
                    cleaned, srcLang, tgtLang, examples, budget);
            Log.i(TAG, "translate budgeted chars=" + cleaned.length()
                    + " budget=" + budget
                    + " (" + (System.currentTimeMillis() - startTime) + "ms)");
            return new SsbdTranslationResult(out, 0, 0,
                    System.currentTimeMillis() - startTime, false, key);
        }
        if (draftExamplesKey == null || !draftExamplesKey.equals(key)) {
            String out = onnxModule.translateSentenceWithBudget(
                    cleaned, srcLang, tgtLang, examples, budget);
            Log.i(TAG, "ssbd skipped: examples changed chars=" + cleaned.length()
                    + " budget=" + budget
                    + " (" + (System.currentTimeMillis() - startTime) + "ms)");
            return new SsbdTranslationResult(out, 0, 0,
                    System.currentTimeMillis() - startTime, false, key);
        }
        HyMtOnnxModule.SsbdResult r = onnxModule.translateSentenceWithDraft(
                cleaned, srcLang, tgtLang, examples, draftIds, budget, beta);
        Log.i(TAG, "translate ssbd chars=" + cleaned.length()
                + " budget=" + budget + " beta=" + beta
                + " accepted=" + r.accepted + "/" + r.draftLength
                + " verified=" + r.verified
                + " (" + (System.currentTimeMillis() - startTime) + "ms)");
        return new SsbdTranslationResult(r.text, r.accepted, r.draftLength,
                System.currentTimeMillis() - startTime, r.verified, key);
    }

    /**
     * Fingerprint of the few-shot examples baked into the prompt. Two calls
     * verify-compatibly only when their keys are equal; "" means no examples.
     */
    static String examplesKey(java.util.List<String[]> examples) {
        if (examples == null || examples.isEmpty()) return "";
        StringBuilder sb = new StringBuilder();
        for (String[] pair : examples) {
            if (pair == null) continue;
            for (String s : pair) sb.append(s == null ? "" : s).append((char) 1);
            sb.append((char) 2);
        }
        return sb.toString();
    }

    /** SSBD translation result with paper §4 diagnostics (A/D). */
    public static class SsbdTranslationResult extends TranslationResult {
        public final int accepted;
        public final int draftLength;
        public final boolean verified;
        /** Fingerprint of the examples used for THIS output ("" = none). */
        public final String examplesKey;

        public SsbdTranslationResult(String text, int accepted, int draftLength,
                                     long processingMs, boolean verified) {
            this(text, accepted, draftLength, processingMs, verified, "");
        }

        public SsbdTranslationResult(String text, int accepted, int draftLength,
                                     long processingMs, boolean verified, String examplesKey) {
            super(text, processingMs);
            this.accepted = accepted;
            this.draftLength = draftLength;
            this.verified = verified;
            this.examplesKey = examplesKey == null ? "" : examplesKey;
        }

        public float acceptRate() {
            return draftLength <= 0 ? 0f : (float) accepted / draftLength;
        }
    }

    /** Encode a previous translation to draft token ids (null when unavailable). */
    public int[] encodeDraft(String text) {
        try {
            if (onnxModule == null || !onnxModule.isReady()) return null;
            return onnxModule.encodeTarget(text);
        } catch (Exception e) {
            return null;
        }
    }

    /** Port of RTranslator correctText(): terminator + whitespace collapse. */
    private static String correctText(String text) {
        String t = text.trim().replaceAll("\\s+", " ");
        if (t.length() >= 2 && Character.isLetterOrDigit(t.charAt(t.length() - 1))) {
            t += ".";
        }
        return t;
    }

    /** ZH<=>XX → Chinese instruction; XX<=>XX → English instruction (Hy-MT README). */
    private static String buildInstruction(String text, String srcLang, String tgtLang) {
        return buildInstruction(text, srcLang, tgtLang,
                java.util.Collections.emptyList());
    }

    /** Few-shot variant: Tatoeba examples ride inline, task line shape kept. */
    static String buildInstruction(String text, String srcLang, String tgtLang,
                                   java.util.List<String[]> examples) {
        boolean zhInvolved = srcLang.startsWith("zh") || tgtLang.startsWith("zh");
        if (zhInvolved) {
            return "将以下文本翻译为" + zhName(tgtLang) + "，注意只需要输出翻译后的结果，不要额外解释"
                    + TatoebaText.examplesFragment(examples, true) + "：\n\n" + text;
        }
        return "Translate the following segment into " + enName(tgtLang)
                + TatoebaText.examplesFragment(examples, false)
                + ", without additional explanation.\n\n" + text;
    }

    private static String enName(String code) {
        switch (code) {
            case "vi": return "Vietnamese";
            case "zh_hant": return "Traditional Chinese";
            case "zh": case "zh_hans": return "Chinese";
            default: return "English";
        }
    }

    private static String zhName(String code) {
        switch (code) {
            case "vi": return "越南语";
            case "en": return "英语";
            case "zh_hant": return "繁体中文";
            default: return "中文";
        }
    }

    private static String clean(String raw) {
        if (raw == null) return "";
        int cut = raw.indexOf("<｜hy_");
        if (cut >= 0) raw = raw.substring(0, cut);
        raw = raw.replaceAll("</?(think|answer)>", "");
        return raw.trim();
    }

    private ArrayList<String> splitIntoSentences(String text, String langCode) {
        ArrayList<String> sentences = new ArrayList<>();
        Locale locale = langCode != null && langCode.equals("vi") ? new Locale("vi") : Locale.US;
        BreakIterator iterator = BreakIterator.getSentenceInstance(locale);
        iterator.setText(text);
        int start = iterator.first();
        for (int end = iterator.next(); end != BreakIterator.DONE; start = end, end = iterator.next()) {
            sentences.add(text.substring(start, end).trim());
        }
        return sentences;
    }

    public void close() {
        if (onnxModule != null) {
            onnxModule.close();
            onnxModule = null;
        }
        if (tatoebaAssist != null) {
            tatoebaAssist.close();
            tatoebaAssist = null;
        }
    }

    /** Always true: ONNX is the sole backend (no fallback). */
    public boolean isOnnxBackend() {
        return onnxModule != null;
    }

    public static class TranslationResult {
        public final String text;
        public final long processingMs;
        public TranslationResult(String text, long processingMs) {
            this.text = text;
            this.processingMs = processingMs;
        }
    }
}
