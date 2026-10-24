/*
 * OmniVoice — HY-MT 1.5 greedy decode on the Hexagon NPU.
 *
 * Java twin of backend/translation_hymt_npu.py (self-checked against the
 * original export). Three EPContext graphs from optimize/16_qnn_compile.py
 * share one weight copy in a single QNN context, every node on the HTP
 * (the embedding Gather too, so the wrappers hold no CPU table copy):
 *
 *   hymt_ar64.<soc>.ctx.onnx      prefill, 64 tokens per run, cache ctx 1024
 *   hymt_ar64_256.<soc>.ctx.onnx  short prefill, 64 tokens per run, cache ctx 256
 *   hymt_ar1.<soc>.ctx.onnx       decode, 1 token per run, cache ctx 256
 *   inputs : input_ids [1,AR], attention_mask [1,ctx], position_ids [1,AR],
 *            past_key_values.{0..31}.{key,value} fp16 [1,4,ctx-AR,128]
 *   outputs: next_ids [1,AR] (argmax), new.{i}.{key,value} fp16 [1,4,AR,128]
 *
 * This class owns fixed caches (192 short-prefill slots, 255 decode slots,
 * and 960 long-prefill slots allocated on the first long prompt):
 * slots [0,n) hold the tokens seen so far, each step's K/V is written at
 * slot n, the mask marks the valid slots plus the current tokens (concat
 * index ctx-AR+j), and RoPE gets the true positions n+j. Every run streams
 * its whole cache through the attention ops, so short calls (prompt +
 * budget <= 255) decode over the small cache, and prefill on the 256-slot
 * AR64 graph when prompt + draft <= 192; longer ones prefill and keep
 * decoding on the 1024-slot AR64 graph with one valid token per run.
 *
 * Draft verification (lossless speculative decoding): a run costs about the
 * same for 1 or 64 tokens, and the last prefill chunk is mostly padding. A
 * draft (the previous translation of a source prefix) rides in that padding
 * and the run's per-position argmax checks it: the longest prefix equal to
 * greedy is kept, so each accepted token saves one ~90 ms decode run and the
 * output is the greedy output. The padded part costs no run (draftRoom); a
 * longer draft that matched all of it is verified 64 tokens per extra
 * prefill run while it keeps matching (log `ver=`).
 */
package com.omnivoice.onspeak47.pipeline;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import com.omnivoice.onspeak47.util.FileUtils;
import com.omnivoice.onspeak47.util.NpuStats;
import com.omnivoice.onspeak47.util.OrtSessionConfig;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Locale;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public final class HyMtNpuDecoder implements AutoCloseable {

    private static final String TAG = "HyMtNpuDecoder";
    public static final String EXT_DIR = "hymt_npu";
    /**
     * Graph revision, written to EXT_DIR/REV by the in-app install and
     * optimize/16 --adb-push (HYMT_NPU_REV). 2 = FP16-safe mask fill
     * (rev 1 decoded EOS first on the HTP); 3 = q/k norms as elementwise ops
     * (QNN RmsNorm zeroed K heads 1-3: junk token, then repeats / EOS);
     * 4 = split ArgMax (HTP ArgMax flipped bit 16 of ids > 65535, so EOS
     * 120020 read as 54484 " goats" and decoding never stopped);
     * 5 = 256-slot decode graph (was 1024: ~95 ms/token on the 8 Elite Gen 5);
     * 6 = Hy-MT2-1.8B weights (same layout and surgery as 5);
     * 7 = embedding Gather on the HTP (QDQ) + 256-slot prefill graph;
     * 8 = Hy-MT2-1.8B-1.25Bit weights (ternary linears as 2-bit MatMulNBits).
     * Older graphs are never loaded.
     */
    public static final String REV = "8";

    /** True when the staged graphs carry the current {@link #REV}. */
    public static boolean isCurrent(File dir) {
        return REV.equals(FileUtils.readTextFile(new File(dir, "REV")));
    }
    private static final int CTX = 1024, PREFILL = 64, DECODE_CTX = 256, LAYERS = 32, HEADS = 4, DIM = 128;
    private static final int PAST = CTX - PREFILL, SLOTS = DECODE_CTX - 1;   // 960, 255
    private static final int SHORT = DECODE_CTX - PREFILL;                     // 192

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession prefill, prefillShort, decode;
    private final String[] kvNames = new String[LAYERS * 2];
    private final ShortBuffer[] past = new ShortBuffer[LAYERS * 2];   // [4][960][128], lazy
    private final ShortBuffer[] shortPast = new ShortBuffer[LAYERS * 2];  // [4][192][128]
    private final ShortBuffer[] cache = new ShortBuffer[LAYERS * 2];  // [4][255][128]
    private final Map<String, OnnxTensor> pastFeed = new HashMap<>();  // zero-copy views
    private final Map<String, OnnxTensor> shortFeed = new HashMap<>();
    private final Map<String, OnnxTensor> cacheFeed = new HashMap<>();
    /** Draft tokens accepted by the last generate() call (caller holds the module lock). */
    int lastAccepted;

    /** @return the decoder, or null when not staged for this SoC / no QNN NPU. */
    static HyMtNpuDecoder tryCreate(Context context, File dir) {
        String soc = Build.VERSION.SDK_INT >= 31 ? Build.SOC_MODEL.toLowerCase(Locale.ROOT) : "";
        File p = new File(dir, "hymt_ar64." + soc + ".ctx.onnx");
        File ps = new File(dir, "hymt_ar64_256." + soc + ".ctx.onnx");
        File d = new File(dir, "hymt_ar1." + soc + ".ctx.onnx");
        if (soc.isEmpty() || !p.isFile() || !ps.isFile() || !d.isFile()) return null;
        if (!isCurrent(dir)) {
            Log.w(TAG, "stale HY-MT NPU graphs in " + dir + " (want REV " + REV + "), not loaded");
            return null;
        }
        try {
            return new HyMtNpuDecoder(context, p, ps, d);
        } catch (Exception e) {
            Log.w(TAG, "HY-MT NPU unavailable, CPU INT8 path: " + e);
            return null;
        }
    }

    private HyMtNpuDecoder(Context context, File p, File ps, File d) throws Exception {
        long t0 = System.currentTimeMillis();
        prefill = open(context, p);
        prefillShort = open(context, ps);
        decode = open(context, d);
        for (int i = 0; i < LAYERS; i++) {
            kvNames[2 * i] = "past_key_values." + i + ".key";
            kvNames[2 * i + 1] = "past_key_values." + i + ".value";
        }
        for (int k = 0; k < kvNames.length; k++) {
            shortPast[k] = fp16(HEADS * SHORT * DIM);
            cache[k] = fp16(HEADS * SLOTS * DIM);
            shortFeed.put(kvNames[k], OnnxTensor.createTensor(env, shortPast[k],
                    new long[]{1, HEADS, SHORT, DIM}, OnnxJavaType.FLOAT16));
            cacheFeed.put(kvNames[k], OnnxTensor.createTensor(env, cache[k],
                    new long[]{1, HEADS, SLOTS, DIM}, OnnxJavaType.FLOAT16));
        }
        NpuStats.mtOnNpu = true;
        Log.i(TAG, "HY-MT on QNN HTP ready in " + (System.currentTimeMillis() - t0) + " ms");
    }

    private OrtSession open(Context context, File f) throws Exception {
        OrtSession.SessionOptions so = new OrtSession.SessionOptions();
        if (!OrtSessionConfig.addQnnHtp(context, so)) {
            so.close();
            throw new IllegalStateException("QNN HTP device not found");
        }
        so.addConfigEntry("ep.share_ep_contexts", "1");   // all graphs, one weight copy
        OrtSession s = env.createSession(f.getAbsolutePath(), so);
        so.close();
        return s;
    }

    /** The 960-slot prefill cache (63 MB), only once a prompt needs it. */
    private Map<String, OnnxTensor> longFeed() throws Exception {
        if (pastFeed.isEmpty()) {
            for (int k = 0; k < kvNames.length; k++) {
                past[k] = fp16(HEADS * PAST * DIM);
                pastFeed.put(kvNames[k], OnnxTensor.createTensor(env, past[k],
                        new long[]{1, HEADS, PAST, DIM}, OnnxJavaType.FLOAT16));
            }
        }
        return pastFeed;
    }

    private static ShortBuffer fp16(int n) {
        return ByteBuffer.allocateDirect(n * 2).order(ByteOrder.nativeOrder()).asShortBuffer();
    }

    /** Free slots in the last prefill chunk: a draft that fits there costs no run. */
    static int draftRoom(int promptLen) {
        int r = promptLen % PREFILL;
        return r == 0 ? 0 : PREFILL - r;
    }

    /** Greedy continuation of {@code ids} (BOS-prefixed prompt); stops on EOS. */
    int[] generate(int[] ids, int budget) throws Exception {
        return generate(ids, budget, null);
    }

    /** Same output as without {@code draft}; accepted draft tokens skip their decode runs. */
    int[] generate(int[] ids, int budget, int[] draft) throws Exception {
        lastAccepted = 0;
        if (ids.length >= PAST) return new int[0];
        int dk = draft == null ? 0 : Math.min(Math.min(draft.length, budget), draftRoom(ids.length));
        int[] seq = java.util.Arrays.copyOf(ids, ids.length + dk);
        if (dk > 0) System.arraycopy(draft, 0, seq, ids.length, dk);
        long t0 = System.nanoTime();
        boolean small = ids.length + budget <= SLOTS;   // decodes on the 256-slot graph
        boolean shortPre = small && seq.length <= SHORT;
        OrtSession pre = shortPre ? prefillShort : prefill;
        Map<String, OnnxTensor> preFeed = shortPre ? shortFeed : longFeed();
        ShortBuffer[] preBuf = shortPre ? shortPast : past;
        int preSlots = shortPre ? SHORT : PAST;
        long[] next = null;
        int s = 0;
        for (; s < seq.length; s += PREFILL) {
            int m = Math.min(PREFILL, seq.length - s);
            long[] x = new long[PREFILL];
            for (int j = 0; j < m; j++) x[j] = seq[s + j];
            next = step(pre, x, s, m, preFeed, preBuf, preSlots);
            if (s + m >= seq.length) break;
        }
        // next[j] = greedy token after seq[s + j]; the last prompt token and the
        // whole draft sit in this last chunk (dk <= draftRoom).
        int base = ids.length - 1 - s, acc = 0;
        while (acc < dk && next[base + acc] == draft[acc]) acc++;
        long tok = next[base + acc];
        // Draft longer than the padding (device log D:\2.0log.txt: 21/66 calls accepted
        // every padded slot and still matched beyond it): while every token so far
        // matched, verify the rest 64 per prefill run. tok == draft[acc] is already
        // proven, so a run only replaces the decode step that token would cost.
        int dmax = draft == null ? 0 : Math.min(draft.length, budget), ver = 0;
        boolean onTrack = acc == dk;
        while (onTrack && dmax - acc >= 2 && tok == draft[acc] && ids.length + acc + 2 <= preSlots) {
            int at = ids.length + acc, m = Math.min(Math.min(PREFILL, dmax - acc), preSlots - at);
            long[] x = new long[PREFILL];
            for (int j = 0; j < m; j++) x[j] = draft[acc + j];
            next = step(pre, x, at, m, preFeed, preBuf, preSlots);
            int j = 0;   // next[j] = greedy token after draft[acc + j]
            while (j < m - 1 && next[j] == draft[acc + 1 + j]) j++;
            acc += 1 + j;
            tok = next[j];
            onTrack = j == m - 1;
            ver++;
        }
        int n = ids.length + acc;   // rejected draft rows beyond n stay masked out
        lastAccepted = acc;
        long t1 = System.nanoTime();
        if (small) for (int k = 0; k < kvNames.length; k++) copySlots(preBuf[k], preSlots, cache[k], SLOTS, n);
        OrtSession sess = small ? decode : prefill;
        Map<String, OnnxTensor> feed = small ? cacheFeed : pastFeed;
        ShortBuffer[] buf = small ? cache : past;
        int slots = small ? SLOTS : PAST;
        long[] x = new long[small ? 1 : PREFILL];
        int[] out = new int[budget];
        int len = acc;
        if (acc > 0) System.arraycopy(draft, 0, out, 0, acc);
        while (len < budget && tok != HyMtBpeTokenizer.EOS_ID && n < slots) {
            out[len++] = (int) tok;
            x[0] = tok;
            tok = step(sess, x, n, 1, feed, buf, slots)[0];
            n++;
        }
        long t2 = System.nanoTime();
        // On-device triage: an immediate EOS / special-token run here means
        // HTP FP16 numerics, not the loop (the CPU twin decodes correctly).
        // msPerTok is the decode-latency number to watch (~95 at REV 4 with
        // 1023 slots); slots=255 is the decode graph, 960 the AR64 fallback;
        // pre=192 is the 256-slot prefill graph, 960 the 1024-slot one.
        Log.i(TAG, "prompt=" + ids.length + " out=" + len + " draft=" + acc + "/" + dk + " ver=" + ver
                + " last=" + tok + " pre=" + preSlots + " slots=" + slots
                + " prefillMs=" + (t1 - t0) / 1_000_000
                + " msPerTok=" + (len - acc == 0 ? 0 : (t2 - t1) / 1_000_000 / (len - acc)) + " first="
                + java.util.Arrays.toString(java.util.Arrays.copyOf(out, Math.min(len, 8))));
        return java.util.Arrays.copyOf(out, len);
    }

    /** One run over {@code buf} ([4][slots][128]); writes the m new K/V rows at slots [n, n+m). */
    private long[] step(OrtSession sess, long[] ids, int n, int m, Map<String, OnnxTensor> past,
                        ShortBuffer[] buf, int slots) throws Exception {
        int ar = ids.length;
        long[] mask = new long[slots + ar];
        for (int i = 0; i < n; i++) mask[i] = 1;
        for (int j = 0; j < m; j++) mask[slots + j] = 1;
        long[] pos = new long[ar];
        for (int j = 0; j < ar; j++) pos[j] = n + j;
        try (OnnxTensor x = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), new long[]{1, ar});
             OnnxTensor am = OnnxTensor.createTensor(env, LongBuffer.wrap(mask), new long[]{1, slots + ar});
             OnnxTensor p = OnnxTensor.createTensor(env, LongBuffer.wrap(pos), new long[]{1, ar})) {
            Map<String, OnnxTensor> feed = new HashMap<>(past);
            feed.put("input_ids", x);
            feed.put("attention_mask", am);
            feed.put("position_ids", p);
            try (OrtSession.Result r = NpuStats.run(sess, feed, NpuStats.MT)) {
                for (int k = 0; k < kvNames.length; k++) {
                    String name = kvNames[k].replace("past_key_values.", "new.");
                    ShortBuffer src = ((OnnxTensor) r.get(name).get()).getShortBuffer();
                    for (int h = 0; h < HEADS; h++) {
                        for (int j = 0; j < m; j++) {
                            ShortBuffer row = src.duplicate();
                            row.position((h * ar + j) * DIM).limit((h * ar + j + 1) * DIM);
                            ShortBuffer dst = buf[k].duplicate();
                            dst.position((h * slots + n + j) * DIM);
                            dst.put(row);
                        }
                    }
                }
                long[] next = new long[ar];
                ((OnnxTensor) r.get("next_ids").get()).getLongBuffer().get(next);
                return next;
            }
        }
    }

    /** First n slots of every head, [4][from][128] -> [4][to][128]. */
    private static void copySlots(ShortBuffer src, int from, ShortBuffer dst, int to, int n) {
        for (int h = 0; h < HEADS; h++) {
            ShortBuffer s = src.duplicate();
            s.position(h * from * DIM).limit((h * from + n) * DIM);
            ShortBuffer d = dst.duplicate();
            d.position(h * to * DIM);
            d.put(s);
        }
    }

    @Override public void close() {
        for (OnnxTensor t : pastFeed.values()) t.close();
        for (OnnxTensor t : shortFeed.values()) t.close();
        for (OnnxTensor t : cacheFeed.values()) t.close();
        for (OrtSession s : new OrtSession[]{prefill, prefillShort, decode}) {
            try {
                s.close();
            } catch (Exception ignored) {
            }
        }
    }
}
