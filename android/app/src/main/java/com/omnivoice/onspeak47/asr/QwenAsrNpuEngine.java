/*
 * OmniVoice — Qwen3-ASR-0.6B (vi/en/zh code-switching LoRA) on the Hexagon NPU.
 *
 * Models (optimize/18_qwen3asr.py export -> optimize/16_qnn_compile.py qwen3asr,
 * staged with `16_qnn_compile.py qwen3asr --adb-push`) in
 * getExternalFilesDir("qwen3asr_npu"):
 *   qasr_enc.<soc>.ctx.onnx                      encoder, one 8 s block per run
 *   qasr_ar{64,1}.<soc>.ctx.onnx (+ qasr_ar64.<soc>.ctx_qnn.bin)  decoder, shared weights
 *   qasr_assets.bin                              int8 embedding, vocab, mel filters (mapped)
 *   REV                                          = {@link #REV}
 * Missing / stale -> null, and StreamingAsrEngine falls back to the Zipformer.
 *
 * The recognizer (QwenAsr) re-decodes the utterance every 2 s of new audio. A step
 * costs an encoder run, a few prefill runs and ~10 decode runs, far more than the
 * pipeline's 160 ms audio-thread budget, so steps run on this engine's worker
 * thread and partials return at once. endOfSpeech() makes the next decode the
 * synchronous final one (all audio, whole budget).
 */
package com.omnivoice.onspeak47.asr;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import com.omnivoice.onspeak47.util.FileUtils;
import com.omnivoice.onspeak47.util.NpuStats;
import com.omnivoice.onspeak47.util.OrtSessionConfig;

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Locale;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public final class QwenAsrNpuEngine extends StreamingAsrEngine implements QwenAsr.Model {

    private static final String TAG = "QwenAsrNpuEngine";
    public static final String EXT_DIR = "qwen3asr_npu";
    /** Graph revision (= QWEN3ASR_NPU_REV in optimize/16_qnn_compile.py); older graphs are never loaded. */
    public static final String REV = "1";
    private static final int L = 28, KVH = 8, HD = 128, CTX = QwenAsr.CTX, HID = QwenAsr.HID;
    /** Graphs give argmax ids only, no scores; the router just needs a steady value. */
    private static final float CONFIDENCE = 0.8f;

    private final AsrLanguage lang;
    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession enc, ar64, ar1;
    private final ShortBuffer[] cache = new ShortBuffer[2 * L];         // [8][512][128] fp16, k_0..27, v_0..27
    private final Map<String, OnnxTensor> pastFeed = new HashMap<>();   // zero-copy views of `cache`
    private final QwenAsr asr;
    private final ExecutorService worker = Executors.newSingleThreadExecutor(r -> new Thread(r, "QwenAsr"));

    // utterance audio + scheduling, guarded by `this`. ponytail: the whole utterance stays in
    // `audio` (64 KB/s), only the part after the recognizer's last 20 s cut is copied per step;
    // trim it too if endpoint-free speech runs for many minutes.
    private float[] audio = new float[QwenAsr.SR * 8];
    private int len, scheduled, finalLen = -1, generation, asrGeneration = -1;
    private boolean ending, cutKnown;   // cutKnown: asr.start belongs to this utterance
    private Future<?> pending;
    private volatile String text = "";

    /** @return the engine, or null when not staged / stale / no QNN HTP. */
    static StreamingAsrEngine tryCreate(Context context, AsrLanguage lang) {
        File dir = context.getExternalFilesDir(EXT_DIR);
        String soc = Build.VERSION.SDK_INT >= 31 ? Build.SOC_MODEL.toLowerCase(Locale.ROOT) : "";
        if (dir == null || soc.isEmpty()) return null;
        File e = new File(dir, "qasr_enc." + soc + ".ctx.onnx");
        File p = new File(dir, "qasr_ar64." + soc + ".ctx.onnx");
        File d = new File(dir, "qasr_ar1." + soc + ".ctx.onnx");
        File assets = new File(dir, "qasr_assets.bin");
        if (!e.isFile() || !p.isFile() || !d.isFile() || !assets.isFile()) return null;
        if (!REV.equals(FileUtils.readTextFile(new File(dir, "REV")))) {
            Log.w(TAG, "stale Qwen3-ASR graphs in " + dir + " (want REV " + REV + "), not loaded");
            return null;
        }
        try {
            return new QwenAsrNpuEngine(context, lang, e, p, d, assets);
        } catch (Exception ex) {
            Log.w(TAG, "Qwen3-ASR NPU unavailable, Zipformer path: " + ex);
            return null;
        }
    }

    private QwenAsrNpuEngine(Context context, AsrLanguage lang, File e, File p, File d, File assets)
            throws Exception {
        this.lang = lang;
        long t0 = System.currentTimeMillis();
        enc = open(context, e, false);
        ar64 = open(context, p, true);
        ar1 = open(context, d, true);
        for (int k = 0; k < 2 * L; k++) {
            cache[k] = ByteBuffer.allocateDirect(KVH * CTX * HD * 2).order(ByteOrder.nativeOrder()).asShortBuffer();
            pastFeed.put((k < L ? "past_k_" : "past_v_") + k % L, OnnxTensor.createTensor(env, cache[k],
                    new long[]{1, KVH, CTX, HD}, OnnxJavaType.FLOAT16));
        }
        ByteBuffer map;
        try (RandomAccessFile f = new RandomAccessFile(assets, "r")) {
            map = f.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, f.length());  // stays valid after close
        }
        asr = new QwenAsr(new QwenAsr.Assets(map), this);
        NpuStats.asrOnNpu = true;
        Log.i(TAG, "Qwen3-ASR on QNN HTP ready in " + (System.currentTimeMillis() - t0) + " ms");
    }

    private OrtSession open(Context context, File f, boolean shared) throws Exception {
        OrtSession.SessionOptions so = new OrtSession.SessionOptions();
        if (!OrtSessionConfig.addQnnHtp(context, so)) {
            so.close();
            throw new IllegalStateException("QNN HTP device not found");
        }
        if (shared) so.addConfigEntry("ep.share_ep_contexts", "1");  // ar64 + ar1, one weight copy
        OrtSession s = env.createSession(f.getAbsolutePath(), so);
        so.close();
        return s;
    }

    // ---- QwenAsr.Model over the EPContext graphs ----

    @Override public float[] encode(float[] mel, float[] mask) throws Exception {
        try (OnnxTensor m = OnnxTensor.createTensor(env, FloatBuffer.wrap(mel), new long[]{1, 128, QwenAsr.BLOCK});
             OnnxTensor k = OnnxTensor.createTensor(env, FloatBuffer.wrap(mask), new long[]{1, 1, 1, QwenAsr.ROWS});
             OrtSession.Result r = NpuStats.run(enc, feed("mel", m, "mask", k), NpuStats.ASR)) {
            FloatBuffer b = ((OnnxTensor) r.get(0)).getFloatBuffer();
            float[] out = new float[b.remaining()];
            b.get(out);
            return out;
        }
    }

    @Override public long[] decode(float[] x, long[] pos, float[] mask, int slot, int valid) throws Exception {
        int ar = pos.length;
        try (OnnxTensor xt = OnnxTensor.createTensor(env, FloatBuffer.wrap(x), new long[]{1, ar, HID});
             OnnxTensor pt = OnnxTensor.createTensor(env, LongBuffer.wrap(pos), new long[]{1, ar});
             OnnxTensor mt = OnnxTensor.createTensor(env, FloatBuffer.wrap(mask), new long[]{ar, CTX + ar})) {
            Map<String, OnnxTensor> f = new HashMap<>(pastFeed);
            f.put("x", xt);
            f.put("pos", pt);
            f.put("mask", mt);
            try (OrtSession.Result r = NpuStats.run(ar == 1 ? ar1 : ar64, f, NpuStats.ASR)) {
                for (int kv = 0; kv < 2; kv++) {   // new_k/new_v [28][8][AR][128] -> slots [slot, slot+valid)
                    ShortBuffer src = ((OnnxTensor) r.get(kv == 0 ? "new_k" : "new_v").get()).getShortBuffer();
                    for (int l = 0; l < L; l++) {
                        for (int h = 0; h < KVH; h++) {
                            ShortBuffer s = src.duplicate();
                            s.position(((l * KVH + h) * ar) * HD).limit(((l * KVH + h) * ar + valid) * HD);
                            ShortBuffer dst = cache[kv * L + l].duplicate();
                            dst.position((h * CTX + slot) * HD);
                            dst.put(s);
                        }
                    }
                }
                long[] next = new long[ar];
                ((OnnxTensor) r.get("next_ids").get()).getLongBuffer().get(next);
                return next;
            }
        }
    }

    private static Map<String, OnnxTensor> feed(String k1, OnnxTensor v1, String k2, OnnxTensor v2) {
        Map<String, OnnxTensor> m = new HashMap<>();   // Map.of needs API 30
        m.put(k1, v1);
        m.put(k2, v2);
        return m;
    }

    // ---- StreamingAsrEngine ----

    /** Synchronous file mode: preserve the model language tag and propagate errors. */
    public String[] transcribeFileBlock(float[] samples) throws Exception {
        asr.reset();
        java.util.List<Integer> ids = asr.transcribe(samples,
                java.util.Collections.emptyList(), java.util.Collections.emptyList(), QwenAsr.CTX - 1);
        return QwenAsr.parse(asr.decode(ids));
    }

    @Override public synchronized void acceptAudio(float[] samples) {
        if (samples == null || ending) return;  // flush padding: Qwen needs no look-ahead
        if (len + samples.length > audio.length) audio = Arrays.copyOf(audio, 2 * (len + samples.length));
        System.arraycopy(samples, 0, audio, len, samples.length);
        len += samples.length;
    }

    @Override public synchronized boolean isReadyToDecode() {
        return ending ? len > finalLen : len - scheduled >= QwenAsr.CHUNK_SAMPLES;
    }

    /** Streaming: start a step on the worker (no-op while one runs). After endOfSpeech: final decode, blocking. */
    @Override public void decodeAvailable() {
        synchronized (this) {
            if (!ending) {
                if (len - scheduled < QwenAsr.CHUNK_SAMPLES || (pending != null && !pending.isDone())) return;
                scheduled = len;
                int n = len, gen = generation, from = cutKnown ? Math.min(asr.start, n) : 0;
                float[] snap = Arrays.copyOfRange(audio, from, n);
                pending = worker.submit(() -> run(snap, from, n, false, gen));
                return;
            }
        }
        decodeFinal();
    }

    @Override public void endOfSpeech() {
        synchronized (this) {
            ending = true;
        }
    }

    private void decodeFinal() {
        Future<?> f;
        synchronized (this) {
            if (len <= finalLen) return;
            finalLen = scheduled = len;
            int n = len, gen = generation, from = cutKnown ? Math.min(asr.start, n) : 0;
            float[] snap = Arrays.copyOfRange(audio, from, n);
            f = worker.submit(() -> run(snap, from, n, true, gen));  // queued behind a running step
        }
        try {
            f.get();
        } catch (Exception e) {
            Log.w(TAG, "final decode failed", e);
        }
    }

    /** Worker thread only (QwenAsr and the cache are single-threaded). */
    private void run(float[] snap, int from, int n, boolean fin, int gen) {
        synchronized (this) {
            if (gen != generation) return;   // utterance already reset
        }
        try {
            if (asrGeneration != gen) {
                asr.reset();
                asrGeneration = gen;
                synchronized (this) {
                    if (gen == generation) cutKnown = true;
                }
            }
            long t0 = System.nanoTime();
            int pre = asr.prefillRuns, dec = asr.decodeRuns;
            String t = asr.step(snap, from, n, fin);
            synchronized (this) {
                if (gen == generation) text = t;
            }
            Log.d(TAG, (fin ? "final" : "step") + " audio=" + n * 1000 / QwenAsr.SR + "ms prefill="
                    + (asr.prefillRuns - pre) + " decode=" + (asr.decodeRuns - dec) + " took="
                    + (System.nanoTime() - t0) / 1_000_000 + "ms text=" + t);
        } catch (Exception e) {
            Log.w(TAG, "Qwen3-ASR step failed", e);
        }
    }

    @Override public PartialResult getPartialResult() {
        return new PartialResult(text, CONFIDENCE, false);
    }

    @Override public PartialResult getFinalResult() {
        endOfSpeech();
        decodeFinal();
        return new PartialResult(text, CONFIDENCE, true);
    }

    @Override public float getConfidence() {
        return CONFIDENCE;
    }

    @Override public AsrLanguage language() {
        return lang;
    }

    /** New utterance: drop audio and text; a step still running is discarded. */
    @Override public synchronized void reset() {
        generation++;
        len = scheduled = 0;
        finalLen = -1;
        ending = cutKnown = false;
        text = "";
    }

    @Override public void close() {
        worker.shutdownNow();
        for (OnnxTensor t : pastFeed.values()) t.close();
        for (OrtSession s : new OrtSession[]{enc, ar64, ar1}) {
            try {
                s.close();
            } catch (Exception ignored) {
            }
        }
    }

    @Override public boolean isFake() {
        return false;
    }
}
