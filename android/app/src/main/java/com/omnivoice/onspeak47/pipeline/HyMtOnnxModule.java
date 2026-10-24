/*
 * OmniVoice — Hy-MT Enhanced INT8 ONNX Module (niedev/OnnxModelsEnhancer)
 *
 * Primary translation backend after the full migration off the 1.25-bit
 * STQ GGUF: a single vanilla-ONNX decoder with KV-cache
 * (model_int8_final.onnx + .onnx.data, ~2.1 GB, weight-only INT8 +
 * dynamic INT8 Gather/MatMul) run by plain ONNX Runtime — no llama.cpp,
 * no STQ kernel, no ORT GenAI. Mirrors backend/translation_hymt_onnx.py:
 *
 *   inputs : input_ids (1, seq), attention_mask (1, past + seq),
 *            past_key_values.{0..31}.key/value (1, 4, past, 128)
 *   outputs: logits (1, seq, 120818), present.{0..31}.key/value
 *   framing: <｜hy_User｜>…<｜hy_Assistant｜> + hand-prepended BOS 120000,
 *            greedy argmax decode, stop on EOS 120020, budget 64–128/256.
 *
 * Asset layout (the 2.1 GB weights CANNOT ship inside the APK — stage with
 * `python optimize/09_stage_hymt_int8.py --adb-push`, which pushes to the
 * app's external files):
 *
 *   getExternalFilesDir("hymt_int8")/model_int8_final.onnx[.data]
 *   getExternalFilesDir("hymt_int8")/tokenizer.json  (else assets copy)
 *
 * Hexagon NPU: when getExternalFilesDir("hymt_npu") holds the EPContext
 * pair from optimize/16_qnn_compile.py hymt for this SoC, HyMtNpuDecoder
 * runs generation on the QNN HTP and the 2.1 GB CPU session is never
 * created (the CPU files are not needed then).
 *
 * Constructed on a background thread (OmniVoiceApp.initializeAll):
 * tokenizer.json parsing (~9 MB) + ORT session creation take seconds.
 */

package com.omnivoice.onspeak47.pipeline;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import com.omnivoice.onspeak47.util.FileUtils;
import com.omnivoice.onspeak47.util.OrtSessionConfig;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class HyMtOnnxModule {

    private static final String TAG = "HyMtOnnxModule";

    public static final String EXT_DIR = "hymt_int8";
    static final String MODEL_FILE = "model_int8_final.onnx";
    static final String MODEL_DATA_FILE = "model_int8_final.onnx.data";
    static final String TOKENIZER_FILE = "tokenizer.json";
    private static final String ASSET_DIR = "hymt_int8";
    /** Sidecar revision; bump when the staged asset bytes change. */
    private static final String ASSET_REV = "1";

    static final String QNN_BACKEND_CPU = "cpu";
    static final String QNN_BACKEND_HTP = "qnn-htp";

    private static final int NUM_LAYERS = 32;
    private static final int NUM_KV_HEADS = 4;
    private static final int HEAD_DIM = 128;
    private static final int N_CTX = 1024;
    private static final int VOCAB_SIZE = 120818;

    private static final int MAX_OUTPUT_TOKENS = 128;
    private static final int MAX_OUTPUT_TOKENS_LONG = 256;

    private OrtEnvironment env;
    private OrtSession session;
    private HyMtNpuDecoder npu;
    private HyMtBpeTokenizer tokenizer;
    private String initError;
    private String backendLabel = QNN_BACKEND_CPU;
    private final Set<String> outputNames = new LinkedHashSet<>();

    public HyMtOnnxModule(Context context) {
        try {
            init(context);
        } catch (Throwable e) {
            // Throwable, not just Exception: a static-init Error (e.g. an
            // Android-unsupported regex flag) must surface as initError, not
            // a silent dead init thread.
            initError = "Hy-MT ONNX init failed (" + e.getClass().getSimpleName()
                    + "): " + e.getMessage();
            Log.e(TAG, initError, e);
        }
    }

    /** Non-null when the backend is unavailable (caller falls back / errors). */
    public String getInitError() {
        return initError;
    }

    public boolean isReady() {
        return initError == null && (session != null || npu != null) && tokenizer != null;
    }

    /** Execution backend of the loaded session: {@code cpu} or {@code qnn-htp}. */
    public String getBackendLabel() {
        return backendLabel;
    }

    // ------------------------------------------------------------------
    // Init
    // ------------------------------------------------------------------

    private void init(Context context) throws Exception {
        File baseDir = context.getExternalFilesDir(null);
        if (baseDir == null) baseDir = context.getFilesDir();
        File npuDir = new File(baseDir, HyMtNpuDecoder.EXT_DIR);
        npu = HyMtNpuDecoder.tryCreate(context, npuDir);
        File modelDir = npu != null ? npuDir : new File(baseDir, EXT_DIR);

        File modelFile = new File(modelDir, MODEL_FILE);
        File dataFile = new File(modelDir, MODEL_DATA_FILE);
        if (npu == null && (!modelFile.isFile() || !dataFile.isFile())) {
            if (!copyModelFromAssets(context, modelDir)) {
                throw new IllegalStateException(
                        "model not staged at " + modelDir
                                + " — run: python optimize/09_stage_hymt_int8.py --adb-push");
            }
        }

        File tokenizerFile = new File(modelDir, TOKENIZER_FILE);
        if (!tokenizerFile.isFile()) {
            copyTokenizerFromAssets(context, modelDir);
            if (!tokenizerFile.isFile()) {
                throw new IllegalStateException("tokenizer.json missing in " + modelDir);
            }
        }

        long t0 = System.currentTimeMillis();
        tokenizer = new HyMtBpeTokenizer(tokenizerFile);
        if (!tokenizer.isReady()) {
            throw new IllegalStateException(tokenizer.getInitError());
        }

        env = OrtEnvironment.getEnvironment();
        if (npu != null) {
            backendLabel = QNN_BACKEND_HTP;
            Log.i(TAG, "HY-MT on Hexagon NPU (" + npuDir + "), CPU session skipped");
            return;
        }
        OrtSession.SessionOptions options = OrtSessionConfig.create(context, false);
        int cores = Runtime.getRuntime().availableProcessors();
        options.setIntraOpNumThreads(Math.max(2, cores - 2));
        options.setInterOpNumThreads(1);
        session = env.createSession(modelFile.getAbsolutePath(), options);
        try {
            options.close();
        } catch (Exception ignored) {
        }
        outputNames.add("logits");
        for (int i = 0; i < NUM_LAYERS; i++) {
            outputNames.add("present." + i + ".key");
            outputNames.add("present." + i + ".value");
        }
        Log.i(TAG, "ONNX session ready in " + (System.currentTimeMillis() - t0)
                + "ms (threads=" + Math.max(2, cores - 2)
                + ", backend=" + backendLabel + ")");
    }

    /** Copy the full model from APK assets when a fat local build has it. */
    private boolean copyModelFromAssets(Context context, File modelDir) {
        try {
            AssetManager am = context.getAssets();
            String[] list = am.list(ASSET_DIR);
            if (list == null) return false;
            boolean hasModel = false, hasData = false, hasTok = false;
            for (String n : list) {
                if (MODEL_FILE.equals(n)) hasModel = true;
                if (MODEL_DATA_FILE.equals(n)) hasData = true;
                if (TOKENIZER_FILE.equals(n)) hasTok = true;
            }
            if (!hasModel || !hasData) return false;
            File revFile = new File(modelDir, ".rev");
            if (!ASSET_REV.equals(FileUtils.readTextFile(revFile))) {
                modelDir.mkdirs();
                copyAsset(am, ASSET_DIR + "/" + MODEL_FILE, new File(modelDir, MODEL_FILE));
                copyAsset(am, ASSET_DIR + "/" + MODEL_DATA_FILE, new File(modelDir, MODEL_DATA_FILE));
                if (hasTok) {
                    copyAsset(am, ASSET_DIR + "/" + TOKENIZER_FILE,
                            new File(modelDir, TOKENIZER_FILE));
                }
                FileUtils.writeTextFile(revFile, ASSET_REV);
            }
            return new File(modelDir, MODEL_FILE).isFile()
                    && new File(modelDir, MODEL_DATA_FILE).isFile();
        } catch (Exception e) {
            Log.w(TAG, "Asset model copy failed", e);
            return false;
        }
    }

    private void copyTokenizerFromAssets(Context context, File modelDir) {
        try {
            AssetManager am = context.getAssets();
            copyAsset(am, ASSET_DIR + "/" + TOKENIZER_FILE,
                    new File(modelDir, TOKENIZER_FILE));
        } catch (Exception e) {
            Log.w(TAG, "Asset tokenizer copy failed", e);
        }
    }

    private static void copyAsset(AssetManager am, String asset, File dst) throws Exception {
        dst.getParentFile().mkdirs();
        try (InputStream is = am.open(asset);
             OutputStream os = new FileOutputStream(dst)) {
            byte[] buf = new byte[256 * 1024];
            int n;
            while ((n = is.read(buf)) > 0) os.write(buf, 0, n);
        }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /**
     * Greedy KV-cache completion of one instruction prompt.
     *
     * @param instruction already-rendered HY-MT instruction (no chat framing)
     * @param budget      max new tokens (mirrors _budget in Python)
     */
    public synchronized String generate(String instruction, int budget) {
        if (!isReady()) return "";
        try {
            int[] promptIds = tokenizer.encode(
                    HyMtBpeTokenizer.USER_MARK + instruction + HyMtBpeTokenizer.ASSISTANT_MARK);
            int[] ids = new int[promptIds.length + 1];
            ids[0] = HyMtBpeTokenizer.BOS_ID;
            System.arraycopy(promptIds, 0, ids, 1, promptIds.length);
            if (npu != null) return tokenizer.decode(npu.generate(ids, budget)).trim();

            int totalLen = ids.length;
            Map<String, OnnxTensor> feed = new HashMap<>();
            feed.put("input_ids", longTensor(new long[][]{toLongArray(ids)}));
            feed.put("attention_mask", onesMask(totalLen));
            addEmptyPast(feed);

            ArrayList<Integer> generated = new ArrayList<>();
            float[][][] pastK = new float[NUM_LAYERS][][];
            float[][][] pastV = new float[NUM_LAYERS][][];
            boolean hasPast = false;

            hasPast = runTokenLoop(feed, ids.length, totalLen, budget,
                    generated, pastK, pastV);
            for (OnnxTensor t : feed.values()) {
                try {
                    t.close();
                } catch (Exception ignored) {
                }
            }
            if (!hasPast && generated.isEmpty()) return "";
            int[] gen = new int[generated.size()];
            for (int i = 0; i < gen.length; i++) gen[i] = generated.get(i);
            return tokenizer.decode(gen).trim();
        } catch (Exception e) {
            Log.e(TAG, "generate failed", e);
            return "";
        }
    }

    /**
     * Shared greedy token loop. Runs at most {@code budget} single-token
     * forwards, appending to {@code generated} and rolling present→past.
     *
     * @param feed       live feed map (owned by caller; prefill tensors in,
     *                   single-token tensors after the first step)
     * @param promptLen  prompt length (argmax window = full prompt on step 0,
     *                   last position after)
     * @param totalLen   prompt + generated length at entry (advanced in place
     *                   via the returned hasPast only — caller tracks length
     *                   across verify/resume boundaries)
     * @return true once at least one present→past roll happened
     */
    private boolean runTokenLoop(Map<String, OnnxTensor> feed, int promptLen, int totalLen,
                                 int budget, ArrayList<Integer> generated,
                                 float[][][] pastK, float[][][] pastV) throws Exception {
        int len = totalLen;
        boolean hasPast = false;
        for (int step = 0; step < budget; step++) {
            OrtSession.Result result = null;
            try {
                result = session.run(feed, outputNames);
                for (OnnxTensor t : feed.values()) {
                    try {
                        t.close();
                    } catch (Exception ignored) {
                    }
                }
                feed.clear();

                OnnxTensor logits = (OnnxTensor) result.get("logits").get();
                int next = argmaxLast(logits, len > promptLen ? 1 : promptLen);
                // NOTE: logits/pk/pv below are owned by `result` — do NOT
                // close them individually. result.close() in finally releases
                // them; explicit closes double-free and spams logcat
                // ("Closing an already closed tensor" ×65 per forward,
                // tripping the log quota and hiding SSBD diagnostics).

                if (next == HyMtBpeTokenizer.EOS_ID) break;
                generated.add(next);
                len++;
                if (len >= N_CTX) break;

                // present → past for the next single-token step.
                for (int i = 0; i < NUM_LAYERS; i++) {
                    OnnxTensor pk = (OnnxTensor) result.get("present." + i + ".key").get();
                    OnnxTensor pv = (OnnxTensor) result.get("present." + i + ".value").get();
                    pastK[i] = toFloat2D(pk);
                    pastV[i] = toFloat2D(pv);
                }
                hasPast = true;

                feed.put("input_ids",
                        longTensor(new long[][]{new long[]{next}}));
                feed.put("attention_mask", onesMask(len));
                for (int i = 0; i < NUM_LAYERS; i++) {
                    feed.put("past_key_values." + i + ".key",
                            kvTensor(pastK[i]));
                    feed.put("past_key_values." + i + ".value",
                            kvTensor(pastV[i]));
                }
            } finally {
                if (result != null) {
                    try {
                        result.close();
                    } catch (Exception ignored) {
                    }
                }
            }
        }
        return hasPast;
    }

    /** Result of an SSBD-assisted generation (Stage A diagnostics). */
    public static class GenerateResult {
        public final String text;
        public final int accepted;
        public final int draftLength;
        public final boolean verified;

        public GenerateResult(String text, int accepted, int draftLength,
                              boolean verified) {
            this.text = text == null ? "" : text;
            this.accepted = accepted;
            this.draftLength = draftLength;
            this.verified = verified;
        }

        public float acceptRate() {
            return draftLength <= 0 ? 0f : (float) accepted / draftLength;
        }
    }

    private static class VerifyResult {
        final int divergence;
        final float[][][] pastK;
        final float[][][] pastV;
        final int totalLen;

        VerifyResult(int divergence, float[][][] pastK, float[][][] pastV,
                     int totalLen) {
            this.divergence = divergence;
            this.pastK = pastK;
            this.pastV = pastV;
            this.totalLen = totalLen;
        }
    }

    /**
     * Verify a draft in one forward pass. Returns the first divergence
     * position d (0..draft.length): tokens [0, d) matched greedy argmax and
     * are reusable with the retained KV cache. d == draft.length means the
     * full draft was accepted. d == -1 signals verify failure (fall back).
     * Stage A is unbiased (exact argmax match only); Stage B adds the β
     * margin in firstDivergence().
     */
    private VerifyResult verifyDraftIds(int[] promptIds, int[] draftIds) throws Exception {
        return verifyDraftIds(promptIds, draftIds, 0f);
    }

    /**
     * Biased variant (paper §3.3 Eq.3): {@code beta} in [0,1] softly favors
     * the draft token. {@code beta<=0} is exactly the unbiased Stage-A path;
     * {@code beta>=1} accepts any in-vocab draft. Callers pass
     * {@code AsrState.SSBD_BETA} (0.2) when SSBD is enabled.
     */
    private VerifyResult verifyDraftIds(int[] promptIds, int[] draftIds, float beta) throws Exception {
        int promptLen = promptIds.length;
        int draftLen = draftIds.length;
        if (promptLen + draftLen > N_CTX) return new VerifyResult(-1, null, null, 0);
        int[] full = new int[promptLen + draftLen];
        System.arraycopy(promptIds, 0, full, 0, promptLen);
        System.arraycopy(draftIds, 0, full, promptLen, draftLen);

        Map<String, OnnxTensor> feed = new HashMap<>();
        feed.put("input_ids", longTensor(new long[][]{toLongArray(full)}));
        feed.put("attention_mask", onesMask(full.length));
        addEmptyPast(feed);

        OrtSession.Result result = null;
        try {
            result = session.run(feed, outputNames);
            OnnxTensor logits = (OnnxTensor) result.get("logits").get();
            int divergence = firstDivergence(logits, promptLen, draftIds, beta);
            float[][][] pastK = new float[NUM_LAYERS][][];
            float[][][] pastV = new float[NUM_LAYERS][][];
            int keepLen = promptLen + divergence;
            for (int i = 0; i < NUM_LAYERS; i++) {
                OnnxTensor pk = (OnnxTensor) result.get("present." + i + ".key").get();
                OnnxTensor pv = (OnnxTensor) result.get("present." + i + ".value").get();
                pastK[i] = sliceHeads(pk, keepLen);
                pastV[i] = sliceHeads(pv, keepLen);
            }
            // Result-owned tensors (logits/pk/pv): released by result.close()
            // below — no individual closes (see runTokenLoop note).
            return new VerifyResult(divergence, pastK, pastV, keepLen);
        } finally {
            for (OnnxTensor t : feed.values()) {
                try { t.close(); } catch (Exception ignored) {}
            }
            feed.clear();
            if (result != null) {
                try { result.close(); } catch (Exception ignored) {}
            }
        }
    }

    /**
     * Greedy argmax-compare of the draft against verify-pass logits.
     * Layout [1, promptLen+draftLen, VOCAB]; position promptLen+i predicts
     * draft[i].
     */
    static int firstDivergence(OnnxTensor logits, int promptLen,
                               int[] draftIds) throws Exception {
        return firstDivergence(logits, promptLen, draftIds, 0f);
    }

    /**
     * SSBD Stage B (paper §3.3 Eq.3): biased draft verification.
     *
     * <p>P'(y) = (1-β)P(y) + β·δ(y, draft). The draft wins iff
     * (1-β)·P[draft] + β &gt; (1-β)·P[best], i.e. P[best]-P[draft] &lt;
     * β/(1-β). β=0.2 → threshold 0.25: keep the draft when confidence is
     * comparable, still allow revision when the model strongly disagrees.
     * β≥0.5 collapses toward Continuous Decoding (paper Table 3) — do not use.
     */
    static int firstDivergence(OnnxTensor logits, int promptLen,
                               int[] draftIds, float beta) throws Exception {
        float b = Math.max(0f, Math.min(1f, beta));
        FloatBuffer buf = logits.getFloatBuffer();
        for (int i = 0; i < draftIds.length; i++) {
            int rowStart = (promptLen + i) * VOCAB_SIZE;
            int draft = draftIds[i];
            if (draft < 0 || draft >= VOCAB_SIZE) return i;
            if (b <= 0f) {
                int best = 0;
                float bestV = Float.NEGATIVE_INFINITY;
                for (int v = 0; v < VOCAB_SIZE; v++) {
                    float f = buf.get(rowStart + v);
                    if (f > bestV) {
                        bestV = f;
                        best = v;
                    }
                }
                if (best != draft) return i;
                continue;
            }
            if (b >= 1f) continue; // accept all in-vocab drafts
            // Pass 1: max + best + draft logit.
            float max = Float.NEGATIVE_INFINITY;
            int best = 0;
            for (int v = 0; v < VOCAB_SIZE; v++) {
                float f = buf.get(rowStart + v);
                if (f > max) {
                    max = f;
                    best = v;
                }
            }
            if (best == draft) continue;
            float logitDraft = buf.get(rowStart + draft);
            // Pass 2: softmax denominator for P[best] vs P[draft].
            double sum = 0.0;
            for (int v = 0; v < VOCAB_SIZE; v++) {
                sum += Math.exp(buf.get(rowStart + v) - max);
            }
            double pBest = 1.0 / sum;
            double pDraft = Math.exp(logitDraft - max) / sum;
            double threshold = b / (1.0 - b);
            if ((pBest - pDraft) >= threshold) return i;
        }
        return draftIds.length;
    }

    /**
     * Pure-Java biased accept check for one vocab row (no ORT needed).
     * Same math as the OnnxTensor path above; used by JVM unit tests.
     */
    static boolean acceptsDraftToken(float[] rowLogits, int draftId, float beta) {
        if (rowLogits == null || draftId < 0 || draftId >= rowLogits.length) return false;
        float b = Math.max(0f, Math.min(1f, beta));
        if (b <= 0f) {
            int best = 0;
            float bestV = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < rowLogits.length; i++) {
                if (rowLogits[i] > bestV) {
                    bestV = rowLogits[i];
                    best = i;
                }
            }
            return best == draftId;
        }
        if (b >= 1f) return true;
        float max = Float.NEGATIVE_INFINITY;
        int best = 0;
        for (int i = 0; i < rowLogits.length; i++) {
            if (rowLogits[i] > max) {
                max = rowLogits[i];
                best = i;
            }
        }
        if (best == draftId) return true;
        double sum = 0.0;
        for (float v : rowLogits) sum += Math.exp(v - max);
        double pBest = 1.0 / sum;
        double pDraft = Math.exp(rowLogits[draftId] - max) / sum;
        return (pBest - pDraft) < (b / (1.0 - b));
    }

    /**
     * Slice a present KV tensor [1, heads, seq, dim] to its first keepLen
     * positions, in compact [heads, keepLen*dim] form.
     */
    private static float[][] sliceHeads(OnnxTensor t, int keepLen) throws Exception {
        long[] shape = t.getInfo().getShape();
        int heads = (int) shape[1];
        int seq = (int) shape[2];
        int dim = (int) shape[3];
        int keep = Math.max(0, Math.min(keepLen, seq));
        FloatBuffer buf = t.getFloatBuffer();
        float[][] out = new float[heads][keep * dim];
        for (int h = 0; h < heads; h++) {
            for (int j = 0; j < keep * dim; j++) {
                out[h][j] = buf.get(h * seq * dim + j);
            }
        }
        return out;
    }

    /**
     * SSBD Stage A+B (paper §3.2 Eq.2 + §3.3 Eq.3): self-speculative decoding
     * with optional biased verification.
     *
     * <p>The previous output for a source prefix is reused as a speculative
     * draft for the grown source: verified in ONE forward pass, generation
     * resumes only from the first divergence with the retained KV cache.
     * Falls back to plain generate() on empty/oversize drafts or verify
     * failure (worst case = today's behavior, never worse output).
     *
     * @param beta 0 = unbiased Stage A; 0.2 = paper Table 3 sweet spot.
     */
    public synchronized GenerateResult generateWithDraft(String instruction, int[] draftIds,
                                                         int budget) {
        return generateWithDraft(instruction, draftIds, budget, 0f);
    }

    public synchronized GenerateResult generateWithDraft(String instruction, int[] draftIds,
                                                         int budget, float beta) {
        if (!isReady()) return new GenerateResult("", 0, 0, false);
        if (npu != null) {
            // Exact greedy verification in the prefill padding (beta ignored: the
            // graphs emit argmax only, and exact acceptance keeps the output unchanged).
            try {
                int[] promptIds = tokenizer.encode(
                        HyMtBpeTokenizer.USER_MARK + instruction + HyMtBpeTokenizer.ASSISTANT_MARK);
                int[] ids = new int[promptIds.length + 1];
                ids[0] = HyMtBpeTokenizer.BOS_ID;
                System.arraycopy(promptIds, 0, ids, 1, promptIds.length);
                String text = tokenizer.decode(npu.generate(ids, budget, draftIds)).trim();
                return new GenerateResult(text, npu.lastAccepted,
                        draftIds == null ? 0 : draftIds.length, true);
            } catch (Exception e) {
                Log.e(TAG, "NPU generateWithDraft failed", e);
                return new GenerateResult("", 0, 0, false);
            }
        }
        try {
            int[] promptIds = tokenizer.encode(
                    HyMtBpeTokenizer.USER_MARK + instruction + HyMtBpeTokenizer.ASSISTANT_MARK);
            int[] ids = new int[promptIds.length + 1];
            ids[0] = HyMtBpeTokenizer.BOS_ID;
            System.arraycopy(promptIds, 0, ids, 1, promptIds.length);

            int[] draft = (draftIds != null) ? draftIds : new int[0];
            if (draft.length == 0 || draft.length > budget) {
                return new GenerateResult(generate(instruction, budget), 0, 0, false);
            }
            VerifyResult vr = verifyDraftIds(ids, draft, beta);
            if (vr.divergence < 0) {
                return new GenerateResult(generate(instruction, budget), 0, 0, false);
            }
            int accepted = vr.divergence;
            ArrayList<Integer> generated = new ArrayList<>();
            for (int i = 0; i < accepted; i++) generated.add(draft[i]);
            Map<String, OnnxTensor> feed = new HashMap<>();
            float[][][] pastK = vr.pastK;
            float[][][] pastV = vr.pastV;
            int resumeTotalLen = vr.totalLen;
            if (accepted > 0) {
                int lastAccepted = draft[accepted - 1];
                feed.put("input_ids", longTensor(new long[][]{new long[]{lastAccepted}}));
                feed.put("attention_mask", onesMask(resumeTotalLen));
                for (int i = 0; i < NUM_LAYERS; i++) {
                    feed.put("past_key_values." + i + ".key", kvTensor(pastK[i]));
                    feed.put("past_key_values." + i + ".value", kvTensor(pastV[i]));
                }
            } else {
                feed.put("input_ids", longTensor(new long[][]{toLongArray(ids)}));
                feed.put("attention_mask", onesMask(ids.length));
                addEmptyPast(feed);
                pastK = new float[NUM_LAYERS][][];
                pastV = new float[NUM_LAYERS][][];
                resumeTotalLen = ids.length;
            }
            boolean hasPast = runTokenLoop(feed, ids.length, resumeTotalLen,
                    Math.max(0, budget - accepted), generated, pastK, pastV);
            for (OnnxTensor t : feed.values()) {
                try { t.close(); } catch (Exception ignored) {}
            }
            if (!hasPast && generated.isEmpty()) {
                return new GenerateResult("", accepted, draft.length, true);
            }
            int[] gen = new int[generated.size()];
            for (int i = 0; i < gen.length; i++) gen[i] = generated.get(i);
            return new GenerateResult(tokenizer.decode(gen).trim(),
                    accepted, draft.length, true);
        } catch (Exception e) {
            Log.e(TAG, "generateWithDraft failed", e);
            return new GenerateResult("", 0, 0, false);
        }
    }

    /** Translate one segment (prompt + budget + cleanup, like Python). */
    public String translateSentence(String text, String srcLang, String tgtLang) {
        return translateSentence(text, srcLang, tgtLang,
                java.util.Collections.emptyList());
    }

    /** Few-shot variant: Tatoeba examples ride inline, task line shape kept. */
    public String translateSentence(String text, String srcLang, String tgtLang,
                                    java.util.List<String[]> examples) {
        int budget = estimateMaxTokens(text);
        String raw = generate(buildInstruction(text, srcLang, tgtLang, examples), budget);
        return clean(raw);
    }

    /**
     * Budget-capped variant for streaming re-translation (same model, same
     * prompt contract — only the generation cap changes).
     */
    public String translateSentenceWithBudget(String text, String srcLang, String tgtLang,
                                             java.util.List<String[]> examples, int budget) {
        int b = budget > 0 ? Math.min(budget, MAX_OUTPUT_TOKENS_LONG) : estimateMaxTokens(text);
        String raw = generate(buildInstruction(text, srcLang, tgtLang, examples), b);
        return clean(raw);
    }

    /**
     * Encode a previous translation to target token ids for SSBD drafts.
     * Returns null when the tokenizer is unavailable.
     */
    public int[] encodeTarget(String text) {
        if (tokenizer == null || !tokenizer.isReady()) return null;
        if (text == null || text.trim().isEmpty()) return new int[0];
        return tokenizer.encode(text.trim());
    }

    /**
     * SSBD-assisted translation (Stage A+B): previous output ids are verified
     * against the grown instruction in one forward pass; generation resumes
     * from the first divergence with retained KV. Falls back to plain
     * budgeted generation when the draft is unusable.
     *
     * @param beta 0 = unbiased; 0.2 = paper sweet spot (Stage B).
     */
    public SsbdResult translateSentenceWithDraft(String text, String srcLang, String tgtLang,
                                                java.util.List<String[]> examples,
                                                int[] draftIds, int budget) {
        return translateSentenceWithDraft(text, srcLang, tgtLang, examples, draftIds, budget, 0f);
    }

    public SsbdResult translateSentenceWithDraft(String text, String srcLang, String tgtLang,
                                                java.util.List<String[]> examples,
                                                int[] draftIds, int budget, float beta) {
        int b = budget > 0 ? Math.min(budget, MAX_OUTPUT_TOKENS_LONG) : estimateMaxTokens(text);
        GenerateResult r = generateWithDraft(
                buildInstruction(text, srcLang, tgtLang, examples), draftIds, b, beta);
        return new SsbdResult(clean(r.text), r.accepted, r.draftLength, r.verified);
    }

    /** SSBD translation result with paper §4 diagnostics (A/D). */
    public static class SsbdResult {
        public final String text;
        public final int accepted;
        public final int draftLength;
        public final boolean verified;

        public SsbdResult(String text, int accepted, int draftLength, boolean verified) {
            this.text = text == null ? "" : text;
            this.accepted = accepted;
            this.draftLength = draftLength;
            this.verified = verified;
        }

        public float acceptRate() {
            return draftLength <= 0 ? 0f : (float) accepted / draftLength;
        }
    }

    // ------------------------------------------------------------------
    // HY-MT prompting (same contract as backend/translation_hymt.py)
    // ------------------------------------------------------------------

    /** Token budget by input length: short ASR → 64–128, long → 256 cap. */
    static int estimateMaxTokens(String text) {
        int len = text == null ? 0 : text.length();
        if (len <= 60) return 64;
        if (len <= 400) return MAX_OUTPUT_TOKENS;
        return MAX_OUTPUT_TOKENS_LONG;
    }

    /** ZH<=>XX → Chinese instruction; XX<=>XX → English instruction (Hy-MT2 README templates). */
    static String buildInstruction(String text, String srcLang, String tgtLang) {
        return buildInstruction(text, srcLang, tgtLang,
                java.util.Collections.emptyList());
    }

    /** Few-shot variant: Tatoeba examples ride inline, task line shape kept. */
    static String buildInstruction(String text, String srcLang, String tgtLang,
                                   java.util.List<String[]> examples) {
        boolean zhInvolved = (srcLang != null && srcLang.startsWith("zh"))
                || (tgtLang != null && tgtLang.startsWith("zh"));
        // This focused app intentionally omits the reference app's optional
        // glossary and translation-memory prompt extensions. The base HY-MT
        // prompt remains the same for the simple ASR -> MT pipeline.
        if (zhInvolved) {
            return "将以下文本翻译为" + zhName(tgtLang) + "，注意只需要输出翻译后的结果，不要额外解释：\n\n" + text;
        }
        // Hy-MT2's documented English default (HY-MT1.5: "Translate the following
        // segment into X, without additional explanation.").
        return "Translate the following text into " + enName(tgtLang)
                + ". Note that you should only output the translated result without any additional explanation:\n\n" + text;
    }

    static String enName(String code) {
        if ("vi".equals(code)) return "Vietnamese";
        if ("zh_hant".equals(code)) return "Traditional Chinese";
        if ("zh".equals(code) || "zh_hans".equals(code)) return "Chinese";
        return "English";
    }

    static String zhName(String code) {
        if ("vi".equals(code)) return "越南语";
        if ("en".equals(code)) return "英语";
        if ("zh_hant".equals(code)) return "繁体中文";
        return "中文";
    }

    static String clean(String raw) {
        if (raw == null) return "";
        int cut = raw.indexOf("<｜hy_");
        if (cut >= 0) raw = raw.substring(0, cut);
        raw = raw.replaceAll("</?(think|answer)>", "");
        // A budget cut can split a UTF-8 char (a trailing U+FFFD in the device log).
        return raw.replace(String.valueOf((char) 0xFFFD), "").trim();
    }

    // ------------------------------------------------------------------
    // Tensor helpers
    // ------------------------------------------------------------------

    private OnnxTensor longTensor(long[][] data) throws Exception {
        long[] flat = new long[data.length * data[0].length];
        int k = 0;
        for (long[] row : data) {
            for (long v : row) flat[k++] = v;
        }
        return OnnxTensor.createTensor(env,
                LongBuffer.wrap(flat), new long[]{data.length, data[0].length});
    }

    private static long[] toLongArray(int[] ids) {
        long[] out = new long[ids.length];
        for (int i = 0; i < ids.length; i++) out[i] = ids[i];
        return out;
    }

    private OnnxTensor onesMask(int totalLen) throws Exception {
        long[] flat = new long[totalLen];
        java.util.Arrays.fill(flat, 1L);
        return OnnxTensor.createTensor(env,
                LongBuffer.wrap(flat), new long[]{1, totalLen});
    }

    private void addEmptyPast(Map<String, OnnxTensor> feed) throws Exception {
        for (int i = 0; i < NUM_LAYERS; i++) {
            feed.put("past_key_values." + i + ".key",
                    OnnxTensor.createTensor(env, FloatBuffer.allocate(0),
                            new long[]{1, NUM_KV_HEADS, 0, HEAD_DIM}));
            feed.put("past_key_values." + i + ".value",
                    OnnxTensor.createTensor(env, FloatBuffer.allocate(0),
                            new long[]{1, NUM_KV_HEADS, 0, HEAD_DIM}));
        }
    }

    /** Argmax over the last sequence position of [1, seq, VOCAB] logits. */
    private static int argmaxLast(OnnxTensor logits, int seqLen) throws Exception {
        FloatBuffer buf = logits.getFloatBuffer();
        int last = (seqLen - 1) * VOCAB_SIZE;
        int best = 0;
        float bestV = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < VOCAB_SIZE; i++) {
            float v = buf.get(last + i);
            if (v > bestV) {
                bestV = v;
                best = i;
            }
        }
        return best;
    }

    /**
     * Copy a present KV tensor [1, heads, seq, dim] into a compact
     * [heads, seq*dim] float array (batch is always 1).
     */
    private static float[][] toFloat2D(OnnxTensor t) throws Exception {
        long[] shape = t.getInfo().getShape();
        int heads = (int) shape[1];
        int seq = (int) shape[2];
        int dim = (int) shape[3];
        FloatBuffer buf = t.getFloatBuffer();
        float[][] out = new float[heads][seq * dim];
        for (int h = 0; h < heads; h++) {
            for (int j = 0; j < seq * dim; j++) {
                out[h][j] = buf.get();
            }
        }
        return out;
    }

    private OnnxTensor kvTensor(float[][] kv) throws Exception {
        int heads = kv.length;
        int flat = kv[0].length;
        int seq = flat / HEAD_DIM;
        float[] buf = new float[heads * flat];
        int k = 0;
        for (float[] h : kv) {
            for (float v : h) buf[k++] = v;
        }
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(buf),
                new long[]{1, heads, seq, HEAD_DIM});
    }

    public void close() {
        if (npu != null) {
            npu.close();
            npu = null;
        }
        if (session != null) {
            try {
                session.close();
            } catch (Exception ignored) {
            }
            session = null;
        }
        // OrtEnvironment is process-wide and is shared by the Qwen runtime.
        // Closing it here would make a later foreground run use a closed global
        // environment after the two model phases have completed once.
        env = null;
    }
}
