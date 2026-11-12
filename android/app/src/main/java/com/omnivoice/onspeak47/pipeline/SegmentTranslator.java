package com.omnivoice.onspeak47.pipeline;

import android.content.Context;

import com.omnivoice.onspeak47.asr.AsrState;
import com.omnivoice.onspeak47.asr.ScriptLid;

import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.BooleanSupplier;

/**
 * LOCAL file (not in OnSpeak47): the app's commit translation path without TTS, drafts or UI.
 *
 * Mirrors {@code PipelineOrchestrator.translateSegmentWithDraft} at the upstream commit in
 * android/ONSPEAK47_UPSTREAM, step for step:
 *   1. ScriptLid.label(row) overrides the acoustic source language;
 *   2. a row already in the target language is copied, a mixed row is prompted from its minority;
 *   3. lowerEnglish + writtenForm (+ viMtForm for Vietnamese) shape the MT input;
 *   4. the previous segment of the same audio rides as forced-prefix context on the NPU;
 *   5. TranslationModule decodes with the commit budget.
 * It lives in this package because CANCEL / CONTEXT on HyMtNpuDecoder are package-private.
 * Keep it in step with upstream when tools/sync_onspeak47.sh moves the vendored files.
 */
public final class SegmentTranslator implements AutoCloseable {

    /** Upstream conference rows are cut at 200 chars; longer file transcripts are packed the same way. */
    static final int MAX_SEGMENT_CHARS = 200;
    private static final int LONG_SEGMENT_TOKENS = 256;

    public static final class Result {
        /** Joined translation of every segment. */
        public final String text;
        /** Language the prompt was built from (ScriptLid verdict, else the caller's label). */
        public final String sourceLanguage;
        /** Number of segments, and how many of them were copies (already in the target language). */
        public final int segments, copied;

        Result(String text, String sourceLanguage, int segments, int copied) {
            this.text = text;
            this.sourceLanguage = sourceLanguage;
            this.segments = segments;
            this.copied = copied;
        }
    }

    private final TranslationModule translator;

    public SegmentTranslator(Context context) throws Exception {
        translator = new TranslationModule(context);
    }

    /**
     * @param row     transcript in the app's script convention (QwenFileRecognizer.routed)
     * @param srcLang acoustic / dataset language, used when the script gives no verdict
     * @param stop    polled between tokens; a stopped decode returns its partial text, so
     *                callers must re-check their own cancellation before keeping the result
     */
    public Result translate(String row, String srcLang, String tgtLang, BooleanSupplier stop) {
        StringBuilder out = new StringBuilder();
        String[] previous = null;   // {src, tgt, source, translation} of the last decoded segment
        String used = srcLang;
        int count = 0, copied = 0;
        for (String segment : segments(row, srcLang)) {
            if (stop != null && stop.getAsBoolean()) break;
            String clean = segment.trim();
            String src = srcLang;
            String detected = ScriptLid.label(clean);
            if (detected != null) src = detected;
            String foreign = ScriptLid.foreign(clean, tgtLang);
            String translated;
            if (detected != null && detected.equals(tgtLang) && foreign == null) {
                translated = ScriptLid.writtenForm(clean);
                previous = null;
                copied++;
            } else {
                if (detected != null && detected.equals(tgtLang)) src = foreign;
                clean = ScriptLid.writtenForm(ScriptLid.lowerEnglish(clean));
                HyMtNpuDecoder.CANCEL.set(stop);
                if (previous != null && previous[0].equals(src) && previous[1].equals(tgtLang)) {
                    HyMtNpuDecoder.CONTEXT.set(new String[]{previous[2], previous[3]});
                }
                try {
                    int budget = clean.length() > MAX_SEGMENT_CHARS
                            ? LONG_SEGMENT_TOKENS : AsrState.COMMIT_MT_MAX_TOKENS;
                    translated = translator.translateWithDraft(
                            "vi".equals(src) ? ScriptLid.viMtForm(clean) : clean, src, tgtLang,
                            null, "", budget, false, AsrState.SSBD_BETA).text;
                } finally {
                    HyMtNpuDecoder.CANCEL.remove();
                    HyMtNpuDecoder.CONTEXT.remove();
                }
                translated = translated == null ? "" : translated.trim();
                previous = translated.isEmpty() ? null : new String[]{src, tgtLang, clean, translated};
                used = src;
            }
            count++;
            if (translated.isEmpty()) continue;
            if (out.length() > 0 && !"zh".equals(tgtLang)) out.append(' ');
            out.append(translated);
        }
        return new Result(out.toString(), used, count, copied);
    }

    /**
     * One segment up to {@link #MAX_SEGMENT_CHARS}; longer text is split at sentence ends and the
     * sentences packed greedily. A single longer sentence stays whole.
     */
    static List<String> segments(String text, String lang) {
        List<String> out = new ArrayList<>();
        String t = text == null ? "" : text.trim().replaceAll("\\s+", " ");
        if (t.isEmpty()) return out;
        if (t.length() <= MAX_SEGMENT_CHARS) {
            out.add(t);
            return out;
        }
        BreakIterator it = BreakIterator.getSentenceInstance(
                "vi".equals(lang) ? new Locale("vi") : "zh".equals(lang) ? Locale.CHINESE : Locale.US);
        it.setText(t);
        StringBuilder current = new StringBuilder();
        for (int start = it.first(), end = it.next(); end != BreakIterator.DONE; start = end, end = it.next()) {
            String sentence = t.substring(start, end);
            if (current.length() > 0 && current.length() + sentence.length() > MAX_SEGMENT_CHARS) {
                out.add(current.toString().trim());
                current.setLength(0);
            }
            current.append(sentence);
        }
        if (current.toString().trim().length() > 0) out.add(current.toString().trim());
        return out;
    }

    @Override public void close() {
        translator.close();
    }
}
