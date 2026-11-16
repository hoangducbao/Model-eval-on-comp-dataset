package com.onspeak47.pipeline;

import android.content.Context;

import com.onspeak47.asr.AsrState;
import com.onspeak47.asr.ScriptLid;

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
 *   3. lowerEnglish + writtenForm shape the MT input, then viMtForm for Vietnamese or
 *      zhMtForm for Chinese (他 -> 她 after a female word, here or in the previous segment);
 *   4. forced-prefix context on the NPU, oldest first: the Vietnamese kinship primer
 *      (HyMtOnnxModule.primer, so anh/chị read as "you") and the previous segment of the
 *      same audio, stored as the MT read it;
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
        /** Text actually sent to the MT, one line per segment (after ScriptLid's rewrites). */
        public final String input;

        Result(String text, String sourceLanguage, int segments, int copied, String input) {
            this.text = text;
            this.sourceLanguage = sourceLanguage;
            this.segments = segments;
            this.copied = copied;
            this.input = input;
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
        StringBuilder out = new StringBuilder(), inputs = new StringBuilder();
        String[] previous = null;   // {src, tgt, MT input, translation} of the last decoded segment
        String used = srcLang;
        int count = 0, copied = 0;
        for (String segment : segments(row, srcLang)) {
            if (stop != null && stop.getAsBoolean()) break;
            Prepared p = prepare(segment, srcLang, tgtLang, previous);
            String translated;
            if (p.copy) {
                translated = p.input;
                previous = null;
                copied++;
            } else {
                HyMtNpuDecoder.CANCEL.set(stop);
                HyMtNpuDecoder.CONTEXT.set(p.context);
                try {
                    int budget = p.input.length() > MAX_SEGMENT_CHARS
                            ? LONG_SEGMENT_TOKENS : AsrState.COMMIT_MT_MAX_TOKENS;
                    translated = translator.translateWithDraft(p.input, p.sourceLanguage, tgtLang,
                            null, "", budget, false, AsrState.SSBD_BETA).text;
                } finally {
                    HyMtNpuDecoder.CANCEL.remove();
                    HyMtNpuDecoder.CONTEXT.remove();
                }
                translated = translated == null ? "" : translated.trim();
                previous = translated.isEmpty() ? null : new String[]{p.sourceLanguage, tgtLang, p.input, translated};
                used = p.sourceLanguage;
            }
            count++;
            if (inputs.length() > 0) inputs.append('\n');
            inputs.append(p.input);
            if (translated.isEmpty()) continue;
            if (out.length() > 0 && !"zh".equals(tgtLang)) out.append(' ');
            out.append(translated);
        }
        return new Result(out.toString(), used, count, copied, inputs.toString());
    }

    /** One segment ready for the MT: what to send, from which language, with which context. */
    static final class Prepared {
        final boolean copy;
        final String sourceLanguage, input;
        /** Context lines {source, translation}, oldest first (HyMtNpuDecoder.CONTEXT). */
        final String[][] context;

        Prepared(boolean copy, String sourceLanguage, String input, String[][] context) {
            this.copy = copy;
            this.sourceLanguage = sourceLanguage;
            this.input = input;
            this.context = context;
        }
    }

    /**
     * The routing and input shaping of PipelineOrchestrator.translateSegmentWithDraft (no I/O).
     * @param previous {src, tgt, MT input, translation} of the previous segment, or null
     */
    static Prepared prepare(String segment, String srcLang, String tgtLang, String[] previous) {
        String clean = segment.trim();
        String src = srcLang;
        String detected = ScriptLid.label(clean);
        if (detected != null) src = detected;
        String foreign = ScriptLid.foreign(clean, tgtLang);
        if (detected != null && detected.equals(tgtLang) && foreign == null) {
            return new Prepared(true, src, ScriptLid.writtenForm(clean), new String[0][]);
        }
        if (detected != null && detected.equals(tgtLang)) src = foreign;
        clean = ScriptLid.writtenForm(ScriptLid.lowerEnglish(clean));
        String[] prev = null;   // {source-language line, target-language line}
        if (previous != null && previous[0].equals(src) && previous[1].equals(tgtLang)) {
            prev = new String[]{previous[2], previous[3]};
        } else if (previous != null && previous[0].equals(tgtLang) && previous[1].equals(src)) {
            prev = new String[]{previous[3], previous[2]};   // reverse pair, as upstream
        }
        String input = "vi".equals(src) ? ScriptLid.viMtForm(clean)
                : "zh".equals(src) ? ScriptLid.zhMtForm(clean, prev == null ? "" : prev[0]) : clean;
        List<String[]> lines = new ArrayList<>();
        String[] primer = HyMtOnnxModule.primer(src, tgtLang);
        if (primer != null) lines.add(primer);
        if (prev != null) lines.add(prev);
        return new Prepared(false, src, input, lines.toArray(new String[0][]));
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
