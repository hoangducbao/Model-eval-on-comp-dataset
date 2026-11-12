/*
 * OmniVoice — Qwen3-ASR-0.6B streaming recognizer core (no Android, no ORT).
 *
 * Java twin of backend/streaming_asr/qwen3asr_npu.py (QwenAsrNpu + Stream), whose
 * self-check runs the exported graphs on the parity clips. The graphs come from
 * optimize/18_qwen3asr.py and run behind {@link Model} (QwenAsrNpuEngine: QNN HTP;
 * QwenAsrTest: a fake). Every CHUNK_SAMPLES of new audio the whole utterance is
 * encoded and decoded again with the previous output minus its last ROLLBACK
 * tokens as a forced prefix (official qwen-asr streaming); unchanged cache slots
 * are kept, finished 8 s encoder blocks are cached, and the rolled-back tokens
 * ride as a draft in the last prefill chunk's padding.
 */
package com.omnivoice.onspeak47.asr;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

final class QwenAsr {

    /** The exported graphs (optimize/18_qwen3asr.py docstring). */
    interface Model {
        /** mel [128][800] row-major + additive key mask [104] -> features [104][1024]. */
        float[] encode(float[] mel, float[] mask) throws Exception;

        /**
         * x [AR][1024] embeddings, pos [AR], mask [AR][CTX+AR] -> argmax id per row. Writes the
         * K/V of rows [0, valid) into its cache at slots [slot, slot + valid).
         */
        long[] decode(float[] x, long[] pos, float[] mask, int slot, int valid) throws Exception;
    }

    static final int SR = 16000, HOP = 160, NFFT = 400, BINS = NFFT / 2 + 1, MELS = 128;
    static final int W = 8, TOK = 13, BLOCK = W * 100, ROWS = W * TOK, HID = 1024;
    static final int CTX = 512, PREFILL = 64, STEP_BUDGET = 64;
    // Official init_streaming_state (2 s). Not 1 s: equal on FLEURS streams (twin: WER vi 6.4 -> 6.7 %,
    // en 10.5 %, CER zh 15.3 -> 14.5 %, partials ~0.55 s earlier) but ASCEND zh-en went 17.5 -> 122 %
    // (repetition loops locked into the forced prefix) and a spliced en+zh clip overflowed the cache.
    static final int CHUNK_SAMPLES = 2 * SR, UNFIXED_CHUNKS = 2;
    static final int ROLLBACK = 10;   // official 5; twin: spliced vi>zh 20.8 -> 18.7, FLEURS vi 10.0 -> 9.0
    // Phone: `Porsche Macan` at 2 s -> `Portia Maccon` at 4 s. FLEURS 40 name sentences: WER 8.9 -> 9.1 %,
    // names 113 -> 114/146, flips 16 -> 9; with no rate limit 12.1 % (invented text locked).
    static final float NAME_HEARD_RATE = 2f;
    static final int MAX_CONTEXT_TOKENS = 48;   // cache slots taken from the 512 (head, audio, text)
    static final int MAX_UTT_SAMPLES = 20 * SR;  // ponytail: quietest spot, not a word boundary; VAD endpoints come first
    static final float MASKED = -1e4f;
    private static final int[] HEAD = {151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669};
    private static final int[] TAIL = {151670, 151645, 198, 151644, 77091, 198};
    private static final int AUDIO_PAD = 151676;
    private static final int[] EOS = {151645, 151643};

    /** qasr_assets.bin (optimize/18_qwen3asr.write_assets), read from a mapped file. */
    static final class Assets {
        final int[] keep;          // new id -> original id, sorted
        final float[] scale;       // int8 embedding row scales
        final int[] vocabOff;
        final byte[] vocab;        // utf-8 bytes per new id
        final float[] mel;         // [201][128] Slaney filters
        final ByteBuffer emb;      // int8 [V][1024]

        Assets(ByteBuffer file) {
            ByteBuffer b = file.duplicate().order(ByteOrder.LITTLE_ENDIAN);
            if (b.get(0) != 'Q' || b.get(1) != 'A' || b.get(2) != 'S' || b.get(3) != 'R' || b.getInt(4) != 1) {
                throw new IllegalArgumentException("not a qasr_assets.bin v1");
            }
            int v = b.getInt(8), nv = b.getInt(12), o = 16;
            keep = new int[v];
            ((ByteBuffer) b.position(o)).asIntBuffer().get(keep);
            o += 4 * v;
            scale = new float[v];
            ((ByteBuffer) b.position(o)).asFloatBuffer().get(scale);
            o += 4 * v;
            vocabOff = new int[v + 1];
            ((ByteBuffer) b.position(o)).asIntBuffer().get(vocabOff);
            o += 4 * (v + 1);
            mel = new float[BINS * MELS];
            ((ByteBuffer) b.position(o)).asFloatBuffer().get(mel);
            o += 4 * mel.length;
            vocab = new byte[nv];
            ((ByteBuffer) b.position(o)).get(vocab);
            o += nv;
            o += (64 - o % 64) % 64;
            ((ByteBuffer) b.position(o)).limit(o + v * HID);
            emb = b.slice();
        }

        int size() {
            return keep.length;
        }

        /** New id of an original token id, -1 when pruned. */
        int id(int original) {
            int i = Arrays.binarySearch(keep, original);
            return i < 0 ? -1 : i;
        }

        void embed(int t, float[] out, int off) {
            float s = scale[t];
            for (int i = 0; i < HID; i++) out[off + i] = emb.get(t * HID + i) * s;
        }
    }

    private final Assets a;
    private final Model model;
    private final int[] tail;
    int[] head;                   // HEAD with the context (hotwords) as the system text
    private final int audioId;
    private final boolean[] eos;
    // 400-point DFT as 16 x 25 Cooley-Tukey (~4x fewer ops than the direct 201 x 400 sum,
    // ~25 ms per 2 s step on a desktop JIT before): n = 25 n1 + n2, k = k1 + 16 k2.
    private static final int N1 = 16, N2 = 25;
    private final float[] c16 = new float[N1 * N1], s16 = new float[N1 * N1];        // W16^(n1 k1)
    private final float[] twc = new float[N2 * N1], tws = new float[N2 * N1];        // W400^(n2 k1)
    private final float[] c25 = new float[N2 * N2], s25 = new float[N2 * N2];        // W25^(n2 k2)
    private final float[] yr = new float[N2 * N1], yi = new float[N2 * N1];          // worker scratch
    private final float[] window = new float[NFFT];
    private final int[] melLo = new int[MELS], melHi = new int[MELS];   // nonzero filter bins per band

    // ---- one utterance ----
    private final List<float[]> melCache = new ArrayList<>();  // log10 mel frames final for good
    private final List<float[][]> blocks = new ArrayList<>();  // finished encoder blocks, valid rows
    private final List<Object> slots = new ArrayList<>();      // per cache slot: Integer id or float[] row
    private final List<Integer> raw = new ArrayList<>();       // last output (new ids)
    private final List<String> done = new ArrayList<>();       // texts of utterances cut at MAX_UTT
    volatile int start;           // first sample of the current internal utterance (app utterance index)
    private int chunks;
    private int heard;                  // samples the previous step decoded
    int prefillRuns, decodeRuns;  // cost counters (logs)
    boolean looped;               // the last generate() stopped at a repetition loop (LoopGuard)

    QwenAsr(Assets assets, Model model) {
        this.a = assets;
        this.model = model;
        head = ids(HEAD);
        tail = ids(TAIL);
        audioId = assets.id(AUDIO_PAD);
        eos = new boolean[assets.size()];
        for (int e : EOS) if (assets.id(e) >= 0) eos[assets.id(e)] = true;
        for (int m = 0; m < MELS; m++) {
            melLo[m] = BINS;
            for (int k = 0; k < BINS; k++) {
                if (assets.mel[k * MELS + m] != 0f) {
                    melLo[m] = Math.min(melLo[m], k);
                    melHi[m] = k + 1;
                }
            }
        }
        for (int j = 0; j < NFFT; j++) window[j] = (float) (0.5 - 0.5 * Math.cos(2 * Math.PI * j / NFFT));
        for (int i = 0; i < N1 * N1; i++) {
            double ph = -2 * Math.PI * (i / N1) * (i % N1) / N1;
            c16[i] = (float) Math.cos(ph);
            s16[i] = (float) Math.sin(ph);
        }
        for (int i = 0; i < N2 * N1; i++) {
            double ph = -2 * Math.PI * (i / N1) * (i % N1) / NFFT;
            twc[i] = (float) Math.cos(ph);
            tws[i] = (float) Math.sin(ph);
        }
        for (int i = 0; i < N2 * N2; i++) {
            double ph = -2 * Math.PI * ((i / N2) * (i % N2) % N2) / N2;
            c25[i] = (float) Math.cos(ph);
            s25[i] = (float) Math.sin(ph);
        }
    }

    private int[] ids(int[] original) {
        int[] out = new int[original.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = a.id(original[i]);
            if (out[i] < 0) throw new IllegalArgumentException("prompt token " + original[i] + " pruned from the vocab");
        }
        return out;
    }

    /** New utterance. */
    void reset() {
        resetInternal();
        done.clear();
        start = 0;
    }

    private void resetInternal() {
        melCache.clear();
        blocks.clear();
        slots.clear();
        raw.clear();
        chunks = 0;
        heard = 0;
    }

    private java.util.Map<String, Integer> pieces;   // vocab bytes (ISO-8859-1 string) -> new id
    int contextTokens;                               // context tokens in the head (log: 0 = all skipped)

    /**
     * Qwen3-ASR context biasing (system text, twin: QwenAsrNpu context=): names and terms the
     * model should prefer, e.g. "Thắng" over the word "thắng". Greedy longest-match over the
     * pruned vocab, not the exact BPE split; text past MAX_CONTEXT_TOKENS is dropped at a whole
     * term. Takes effect on the next decode (cache slots from the head on are recomputed).
     */
    void context(List<String> terms) {
        if (pieces == null) {
            pieces = new java.util.HashMap<>();
            for (int t = 0; t < a.size(); t++) {
                int lo = a.vocabOff[t], hi = a.vocabOff[t + 1];
                if (hi > lo) pieces.putIfAbsent(new String(a.vocab, lo, hi - lo, StandardCharsets.ISO_8859_1), t);
            }
        }
        List<Integer> ctx = new ArrayList<>();
        for (String term : terms) {
            List<Integer> ids = tokenize((ctx.isEmpty() ? "" : ", ") + term);
            if (ids == null || ctx.size() + ids.size() > MAX_CONTEXT_TOKENS) continue;
            ctx.addAll(ids);
        }
        contextTokens = ctx.size();
        int[] base = ids(HEAD), h = new int[base.length + ctx.size()];
        System.arraycopy(base, 0, h, 0, 3);
        for (int i = 0; i < ctx.size(); i++) h[3 + i] = ctx.get(i);
        System.arraycopy(base, 3, h, 3 + ctx.size(), base.length - 3);
        head = h;
    }

    /** Greedy longest match over UTF-8 bytes; null when a byte has no token (pruned). */
    private List<Integer> tokenize(String s) {
        String b = new String(s.getBytes(StandardCharsets.UTF_8), StandardCharsets.ISO_8859_1);
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < b.length(); ) {
            int j = Math.min(b.length(), i + 32);
            Integer t = null;
            for (; j > i && (t = pieces.get(b.substring(i, j))) == null; j--) {}
            if (t == null) return null;
            out.add(t);
            i = j;
        }
        return out;
    }

    // ---- streaming policy (twin: Stream) ----

    /**
     * One re-decode over samples [0, n) of the app's utterance; {@code audio} holds them from
     * sample {@code offset} on ({@code offset <= start}). {@code fin} = last call of the
     * utterance (whole budget, keeps at least one prefix token). Returns the transcript.
     */
    String step(float[] audio, int offset, int n, boolean fin) throws Exception {
        if (n - start < NFFT) return text();
        // At the limit this step ends the part at the quietest spot of the new audio instead of at
        // sample n (D:\apk31jawgege.txt: `2019.` | `2.3`, `转变。` | `转变也好`); the rest starts the next.
        boolean cut = !fin && n - start >= MAX_UTT_SAMPLES;
        int to = cut ? offset + quietCut(audio, n - offset - CHUNK_SAMPLES, n - offset) : n;
        int end = chunks < UNFIXED_CHUNKS ? 0 : Math.max(fin || cut ? 1 : 0, raw.size() - ROLLBACK);
        end = nameLock(end);
        while (end > 0 && decode(raw.subList(0, end)).indexOf('�') >= 0) end--;
        List<Integer> prefix = new ArrayList<>(raw.subList(0, end));
        List<Integer> draft = new ArrayList<>(raw.subList(end, raw.size()));
        List<Integer> out;
        try {
            out = transcribe(Arrays.copyOfRange(audio, start - offset, to - offset), prefix, draft,
                    fin || cut ? CTX - 1 : STEP_BUDGET);
        } catch (IllegalStateException overflow) {
            // Runaway text (a repetition loop locked into the prefix) outgrew the cache: cut here,
            // or every later step of an endpoint-free utterance fails the same way.
            done.add(current());
            resetInternal();
            start = to;
            return text();
        }
        raw.clear();
        raw.addAll(out);
        chunks++;
        heard = to - start;
        // A loop is cut out of the text and the utterance restarts here: kept as the forced
        // prefix it locked every later step until the 20 s cut (D:\apk23 (1).txt).
        if (cut || (!fin && looped)) {  // twin: Stream._step / push cut
            done.add(cut ? dropStop(current()) : current());
            resetInternal();
            start = to;
        }
        return text();
    }

    /** The model ends every text with a stop, here usually mid-sentence (`you guys just.`). */
    static String dropStop(String s) {
        return s.endsWith(".") || s.endsWith("。") ? s.substring(0, s.length() - 1) : s;
    }

    /** Middle of the quietest 100 ms of a[lo, hi). */
    static int quietCut(float[] a, int lo, int hi) {
        int w = SR / 10, hop = SR / 50, best = hi - w / 2;
        float low = Float.MAX_VALUE;
        for (int s = lo; s + w <= hi; s += hop) {
            float e = 0;
            for (int i = s; i < s + w; i++) e += a[i] * a[i];
            if (e < low) {
                low = e;
                best = s + w / 2;
            }
        }
        return best;
    }

    private static final Pattern UNIT = Pattern.compile("[㐀-鿿]|[\\p{L}&&[^㐀-鿿]]+(?:'\\p{L}+)?");

    /** Tokens of raw through the last name the previous step heard, at least {@code end}. */
    private int nameLock(int end) {
        String full = decode(raw);
        int t = full.indexOf("<asr_text>");
        if (t < 0) return end;
        int stop = heardNameEnd(full.substring(t + 10), heard / (float) SR);
        if (stop < 0) return end;
        int need = full.substring(0, t + 10 + stop).getBytes(StandardCharsets.UTF_8).length, k = 0;
        for (int b = 0; k < raw.size() && b < need; k++) b += a.vocabOff[raw.get(k) + 1] - a.vocabOff[raw.get(k)];
        return Math.max(end, k);
    }

    /**
     * End of the last mid-sentence capitalized word (not I, not the text's last word) that lies
     * within NAME_HEARD_RATE units per second of heard audio, or -1. An unknown name flips with
     * every re-decode, so what a step wrote is kept; text past that rate may be invented.
     */
    static int heardNameEnd(String body, float heardSec) {
        List<int[]> units = new ArrayList<>();
        for (Matcher m = UNIT.matcher(body); m.find(); ) units.add(new int[]{m.start(), m.end()});
        int stop = -1;
        for (int i = 0; i + 1 < units.size() && i + 1 <= NAME_HEARD_RATE * heardSec; i++) {
            int s = units.get(i)[0], e = units.get(i)[1];
            String w = body.substring(s, e), before = body.substring(0, s).trim();
            if (Character.isUpperCase(w.charAt(0)) && !w.equals("I") && !w.startsWith("I'")
                    && !before.isEmpty() && ".!?…".indexOf(before.charAt(before.length() - 1)) < 0) stop = e;
        }
        return stop;
    }

    /** Transcript so far: earlier cut parts + the current one (model text, no tag). */
    String text() {
        String s = "";
        for (String d : done) s = joinPart(s, d);
        return joinPart(s, current());
    }

    private static final Pattern PART_START = Pattern.compile("^(\\p{Lu})\\p{Ll}+(?= \\p{Ll})");

    /**
     * A new part always starts with a capital, here usually mid-sentence behind a dropped stop:
     * `gia đình Anh` -> "England", `bà Anh` -> "Mrs. Anh", `bác Miễn là` -> "uncle Miễn" (D:\apk32.txt,
     * one every ~20 s). Kept: a word the part before had capitalized mid-sentence, i.e. a name.
     * ponytail: a new name that opens a part is lowered too; 0 of 30 joins in that log.
     */
    static String joinPart(String head, String part) {
        if (head.isEmpty() || part.isEmpty()) return head + part;
        Matcher m = PART_START.matcher(part);
        if (m.find() && ".!?…。！？".indexOf(head.charAt(head.length() - 1)) < 0
                && !Pattern.compile("[\\p{L},] " + Pattern.quote(m.group()) + "(?!\\p{L})").matcher(head).find()) {
            part = m.group(1).toLowerCase(java.util.Locale.ROOT) + part.substring(1);
        }
        return head + " " + part;
    }

    private String current() {
        String[] p = parse(decode(raw));
        return scriptConvention(p[0], p[1]);
    }

    /** "language X<asr_text>text" -> {X, text}; no tag yet -> {"", raw}. */
    static String[] parse(String s) {
        s = s.trim();
        int i = s.indexOf("<asr_text>");
        if (i < 0) return new String[]{"", s};
        String lang = s.substring(0, i).trim();
        if (lang.startsWith("language")) lang = lang.substring(8).trim();
        if (lang.equalsIgnoreCase("none")) lang = "";
        return new String[]{lang, s.substring(i + 10).trim()};
    }

    private static final Pattern ASCII_WORD = Pattern.compile(
            "(?<![A-Za-z\\u00C0-\\u024F\\u1E00-\\u1EFF'])[A-Za-z][A-Za-z']*(?![A-Za-z\\u00C0-\\u024F\\u1E00-\\u1EFF])");

    /**
     * The app reads a transcript's language off the 2e Zipformer's output format
     * (ScriptLid: Chinese as characters, English UPPERCASE, Vietnamese lowercase),
     * and MT routing copies rows it reads as the target. Qwen writes English in
     * normal case, which ScriptLid would read as diacritic-free Vietnamese, so
     * whole-ASCII words go uppercase in rows the model tagged English or Chinese.
     * Vietnamese rows keep their case: an ASCII word there can be either language.
     */
    static String scriptConvention(String tag, String text) {
        if (!tag.equals("English") && !tag.equals("Chinese")) return text;
        Matcher s = SENTENCE.matcher(text);
        StringBuilder sb = new StringBuilder();
        while (s.find()) {
            if (vietnameseSentence(s.group())) {
                sb.append(s.group());
                continue;
            }
            Matcher m = ASCII_WORD.matcher(s.group());
            StringBuffer up = new StringBuffer();
            while (m.find()) m.appendReplacement(up, m.group().toUpperCase(java.util.Locale.ROOT));
            sb.append(m.appendTail(up));
        }
        return sb.toString();
    }

    private static final Pattern SENTENCE = Pattern.compile("[^.!?。！？]*[.!?。！？]*");

    /**
     * The tag covers a whole 20 s part, so Vietnamese spoken after an English start came out as
     * `mình XEM … người TA` (D:\apk32.txt), read as English and re-translated into Vietnamese.
     * Lowercase diacritic words outnumber the ASCII ones here; names (`call Nguyễn now`) don't count.
     */
    static boolean vietnameseSentence(String s) {
        int vi = 0, ascii = 0;
        for (String w : s.split("[^\\p{L}']+")) {
            if (w.isEmpty()) continue;
            if (w.chars().allMatch(c -> c < 128)) ascii++;
            else if (Character.isLowerCase(w.charAt(0)) && w.chars().noneMatch(c -> c >= 0x3400)) vi++;
        }
        return vi > ascii;
    }

    String decode(List<Integer> ids) {
        java.io.ByteArrayOutputStream b = new java.io.ByteArrayOutputStream();
        for (int t : ids) b.write(a.vocab, a.vocabOff[t], a.vocabOff[t + 1] - a.vocabOff[t]);
        return new String(b.toByteArray(), StandardCharsets.UTF_8);
    }

    // ---- one decode of the utterance (twin: QwenAsrNpu.transcribe) ----

    List<Integer> transcribe(float[] audio, List<Integer> prefix, List<Integer> draft, int budget)
            throws Exception {
        float[][] feats = encode(logMel(audio, audio.length));
        List<Integer> ids = new ArrayList<>();
        for (int t : head) ids.add(t);
        java.util.Map<Integer, float[]> rows = new java.util.HashMap<>();
        for (float[] f : feats) {
            rows.put(ids.size(), f);
            ids.add(audioId);
        }
        for (int t : tail) ids.add(t);
        ids.addAll(prefix);
        List<Integer> out = new ArrayList<>(prefix);
        out.addAll(generate(ids, rows, Math.min(budget, CTX - ids.size()), draft));
        return out;
    }

    /** Greedy continuation of ids; cache slots whose content did not change are reused. */
    List<Integer> generate(List<Integer> ids, java.util.Map<Integer, float[]> rows, int budget,
                           List<Integer> draft) throws Exception {
        int n = 0;
        while (n < Math.min(ids.size(), slots.size())) {
            Object was = slots.get(n);
            float[] row = rows.get(n);
            boolean same = row != null ? was instanceof float[] && Arrays.equals((float[]) was, row)
                    : was instanceof Integer && was.equals(ids.get(n));
            if (!same) break;
            n++;
        }
        n = Math.min(n, ids.size() - 1);  // always run the last prompt token
        int room = (PREFILL - (ids.size() - n) % PREFILL) % PREFILL;
        List<Integer> d = draft.subList(0, Math.min(room, draft.size()));
        List<Integer> seq = new ArrayList<>(ids);
        seq.addAll(d);
        List<Integer> am = prefill(seq, rows, n);
        int base = ids.size() - 1 - n, acc = 0;
        int nxt = am.get(base);
        while (acc < d.size() && nxt == d.get(acc) && !eos[nxt]) {
            acc++;
            nxt = am.get(base + acc);
        }
        List<Integer> out = new ArrayList<>(d.subList(0, acc));
        slots.clear();
        for (int i = 0; i < ids.size(); i++) slots.add(rows.containsKey(i) ? rows.get(i) : ids.get(i));
        slots.addAll(out);
        int p = slots.size();
        float[] x = new float[HID];
        looped = false;
        while (!eos[nxt] && out.size() < budget && p < CTX) {
            out.add(nxt);
            int keep = com.omnivoice.onspeak47.util.LoopGuard.cut(
                    out.stream().mapToInt(Integer::intValue).toArray(), out.size());
            if (keep >= 0) {
                out.subList(keep, out.size()).clear();
                looped = true;
                break;
            }
            if (out.size() == budget || p + 1 > CTX) break;
            a.embed(nxt, x, 0);
            long[] r = model.decode(x, new long[]{p}, mask(p, 1), p, 1);
            decodeRuns++;
            slots.add(nxt);
            p++;
            nxt = (int) r[0];
        }
        return out;
    }

    /** Per-position argmax of seq[n:], AR64 runs from slot n. */
    private List<Integer> prefill(List<Integer> seq, java.util.Map<Integer, float[]> rows, int n)
            throws Exception {
        List<Integer> out = new ArrayList<>();
        for (int s = n; s < seq.size(); s += PREFILL) {
            int m = Math.min(PREFILL, seq.size() - s);
            if (s + PREFILL > CTX) throw new IllegalStateException("utterance longer than the cache");
            float[] x = new float[PREFILL * HID];
            for (int j = 0; j < m; j++) {
                float[] row = rows.get(s + j);
                if (row != null) System.arraycopy(row, 0, x, j * HID, HID);
                else a.embed(seq.get(s + j), x, j * HID);
            }
            long[] pos = new long[PREFILL];
            for (int j = 0; j < PREFILL; j++) pos[j] = s + j;
            long[] next = model.decode(x, pos, mask(s, PREFILL), s, m);
            prefillRuns++;
            for (int j = 0; j < m; j++) out.add((int) next[j]);
        }
        return out;
    }

    /** [AR][CTX+AR]: past slots < s visible, then causal over this run's AR keys. */
    static float[] mask(int s, int ar) {
        int w = CTX + ar;
        float[] m = new float[ar * w];
        Arrays.fill(m, MASKED);
        for (int r = 0; r < ar; r++) {
            Arrays.fill(m, r * w, r * w + s, 0f);
            Arrays.fill(m, r * w + CTX, r * w + CTX + r + 1, 0f);
        }
        return m;
    }

    // ---- encoder (finished 8 s blocks cached) ----

    private float[][] encode(float[][] mel) throws Exception {
        int frames = mel[0].length;
        int nblk = (frames + BLOCK - 1) / BLOCK;
        while (blocks.size() < nblk - 1) blocks.add(block(mel, blocks.size()));
        List<float[]> rows = new ArrayList<>();
        for (int b = 0; b < nblk - 1; b++) rows.addAll(Arrays.asList(blocks.get(b)));
        rows.addAll(Arrays.asList(block(mel, nblk - 1)));
        return rows.toArray(new float[0][]);
    }

    private float[][] block(float[][] mel, int b) throws Exception {
        int segFrames = Math.min(BLOCK, mel[0].length - b * BLOCK);
        float[] x = new float[MELS * BLOCK];
        for (int k = 0; k < MELS; k++) System.arraycopy(mel[k], b * BLOCK, x, k * BLOCK, segFrames);
        float[] mask = new float[ROWS];
        Arrays.fill(mask, MASKED);
        int valid = 0;
        for (int c = 0; c < W; c++) {
            int f = Math.min(100, segFrames - c * 100);
            if (f <= 0) continue;
            int t = chunkTokens(f);
            Arrays.fill(mask, c * TOK, c * TOK + t, 0f);
            valid += t;
        }
        float[] feats = model.encode(x, mask);
        float[][] out = new float[valid][];
        int r = 0;
        for (int i = 0; i < ROWS; i++) {
            if (mask[i] == 0f) out[r++] = Arrays.copyOfRange(feats, i * HID, (i + 1) * HID);
        }
        return out;
    }

    static int chunkTokens(int frames) {
        for (int i = 0; i < 3; i++) frames = (frames - 1) / 2 + 1;
        return frames;
    }

    // ---- Whisper log-mel (twin: log_mel; frames are cached once their window is final) ----

    /** [128][len/160] log-mel of audio[0, len): 128 bins, hop 160, centered, max-8 floor. */
    float[][] logMel(float[] audio, int len) {
        int frames = len / HOP;
        List<float[]> f = new ArrayList<>(melCache.subList(0, Math.min(melCache.size(), frames)));
        for (int i = f.size(); i < frames; i++) {
            float[] fr = melFrame(audio, len, i);
            f.add(fr);
            if (i == melCache.size() && i * HOP + NFFT / 2 <= len) melCache.add(fr);  // no reflected tail
        }
        float max = Float.NEGATIVE_INFINITY;
        for (float[] fr : f) for (float v : fr) max = Math.max(max, v);
        float[][] out = new float[MELS][frames];
        for (int i = 0; i < frames; i++) {
            float[] fr = f.get(i);
            for (int k = 0; k < MELS; k++) out[k][i] = (Math.max(fr[k], max - 8f) + 4f) / 4f;
        }
        return out;
    }

    private float[] melFrame(float[] audio, int len, int i) {
        float[] seg = new float[NFFT];
        for (int j = 0; j < NFFT; j++) {
            int t = i * HOP - NFFT / 2 + j;
            if (t < 0) t = -t;                       // reflect padding (numpy "reflect")
            if (t >= len) t = 2 * (len - 1) - t;
            seg[j] = audio[t] * window[j];
        }
        float[] out = new float[MELS];
        double[] pow = new double[BINS];
        for (int n2 = 0; n2 < N2; n2++) {           // y[n2][k1] = W400^(n2 k1) * DFT16 over n1
            for (int k1 = 0; k1 < N1; k1++) {
                float re = 0, im = 0;
                for (int n1 = 0; n1 < N1; n1++) {
                    float v = seg[N2 * n1 + n2];
                    re += v * c16[n1 * N1 + k1];
                    im += v * s16[n1 * N1 + k1];
                }
                int t = n2 * N1 + k1;
                yr[t] = re * twc[t] - im * tws[t];
                yi[t] = re * tws[t] + im * twc[t];
            }
        }
        for (int k = 0; k < BINS; k++) {            // X[k1 + 16 k2] = DFT25 over n2
            int k1 = k % N1, k2 = k / N1;
            float re = 0, im = 0;
            for (int n2 = 0; n2 < N2; n2++) {
                float c = c25[n2 * N2 + k2], s = s25[n2 * N2 + k2];
                float a = yr[n2 * N1 + k1], b = yi[n2 * N1 + k1];
                re += a * c - b * s;
                im += a * s + b * c;
            }
            pow[k] = (double) re * re + (double) im * im;
        }
        for (int m = 0; m < MELS; m++) {
            double e = 0;
            for (int k = melLo[m]; k < melHi[m]; k++) e += pow[k] * a.mel[k * MELS + m];
            out[m] = (float) Math.log10(Math.max(e, 1e-10));
        }
        return out;
    }
}
