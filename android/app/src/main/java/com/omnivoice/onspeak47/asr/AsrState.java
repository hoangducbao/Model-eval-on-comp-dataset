/*
 * OmniVoice — Streaming ASR tuning constants + router states.
 *
 * Spec §6, §9, §11-§12, §14, §19-§20, §21 (MUST 5/6).
 * Pure-Java (no android.* imports) so it runs in local JVM unit tests.
 */
package com.omnivoice.onspeak47.asr;

/** Central tuning constants. Latency/RAM numbers are engineering targets. */
public final class AsrState {

    private AsrState() {}

    // --- Audio frontend (spec §7) ---
    public static final int SAMPLE_RATE = 16000;
    public static final int FRAME_MS = 20;
    public static final int FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS / 1000; // 320

    // --- ASR scheduler (spec §9) ---
    public static final int SCHEDULER_MS = 160;
    public static final int SCHEDULER_SAMPLES = SAMPLE_RATE * SCHEDULER_MS / 1000; // 2560
    public static final int SCHEDULER_FALLBACK_MS = 320;
    /** Zipformer streaming chunk baseline (chunk16; chunk32 fallback, spec §20). */
    public static final int CHUNK_SIZE = 16;
    public static final int CHUNK_SIZE_FALLBACK = 32;
    /**
     * Transducer architecture variant passed as OnlineModelConfig modelType.
     * This is a property of the individual ONNX export, never of the app.
     *
     * Single trilingual model (lmcu000 zh-en-vi stage-2e release):
     * the export metadata carries model_type=zipformer (v1 graph with
     * attention_dims, no query_head_dims), so the value is "zipformer" for
     * every language — passing "zipformer2" aborts natively during
     * initialization (no Java stack, process death).
     */
    public static String transducerModelType(AsrLanguage lang) {
        return "zipformer";
    }

    // --- LID scheduling (spec §11 + VoxLingua pipeline §22) ---
    // ECAPA is a referee, not a realtime decoder (ASR still ticks every
    // 160 ms). Stable → 500–800 ms; uncertain → 300–400 ms; an observed
    // switch candidate is re-polled every 200–300 ms until it passes or
    // the hysteresis persistence window expires.
    public static final int LID_INTERVAL_STABLE_MS = 600;
    public static final int LID_INTERVAL_UNCERTAIN_MS = 400;
    public static final int LID_INTERVAL_CANDIDATE_MS = 250;
    public static final int LID_WINDOW_MS = 480;

    // --- Router scoring (spec §12) ---
    public static final float W_ACOUSTIC = 0.45f;
    public static final float W_TEXT = 0.30f;
    public static final float W_CONFIDENCE = 0.15f;
    public static final float W_HISTORY = 0.10f;

    // --- Bootstrap acoustic LID (VoxLingua pipeline §5, §7-§8, §17) ---
    // Bootstrap answers "which language does this utterance START with?"
    // from AUDIO only — never from the active-ASR transcript (that path is
    // circular: VI decodes EN/ZH into Vietnamese-like text which then votes
    // VI again). Baseline engineering windows for the VoxLingua107 ECAPA
    // classifier (600 ms window / 200 ms hop); 200 ms hard-locks too early
    // (not enough acoustic evidence). Thresholds are starting points for
    // on-device tuning, not absolute optima.
    /** Minimum speech audio before the first bootstrap attempt. */
    public static final int BOOTSTRAP_MIN_MS = 400;
    /**
     * Audio window fed to the acoustic LID classifier. 600 → 1500 ms
     * (2026-09-16 LID-bootstrap fix): measured with the bundled VoxLingua107
     * ECAPA export, a 600 ms window never gives a trustworthy 107-class
     * argmax for English (eu/br/cy win instead) and only ~1 of 3 rolling
     * windows clears the gate at all, while 1500 ms yields en 0.78 and vi
     * 0.96+. See docs/voxlingua_lid_diagnosis.md.
     */
    public static final int BOOTSTRAP_LID_WINDOW_MS = 1500;
    /** Rolling hop between bootstrap attempts (temporal smoothing §16). */
    public static final int BOOTSTRAP_HOP_MS = 200;
    /**
     * Live growing-window cap (2026-09-16 v2: 2500 → 1500). ECAPA cost scales
     * with frames and the slow bootstrap path (relative + 3-hop persistence)
     * commits from 600–1000 ms windows; 1500 ms is plenty live. Endpoints use
     * the wider {@link #BOOTSTRAP_ENDPOINT_WINDOW_MS} (no realtime pressure).
     * Mirrors backend config.
     */
    public static final int BOOTSTRAP_MAX_MS = 1500;
    /** Bootstrap window cap at VAD endpoints (mirrors backend config). */
    public static final int BOOTSTRAP_ENDPOINT_WINDOW_MS = 2500;
    /** Give-up cap: past this much speech without a commit, stop burning
     *  speculative-candidate decodes (2 shadow decodes per pass) and keep
     *  only the hidden provisional stream — avoids a CPU decode storm on
     *  long UNKNOWN spans (unsupported language / noise). Endpoint recovery
     *  still gets one final chance. */
    public static final int BOOTSTRAP_GIVE_UP_MS = 6000;
    /** Cooldown between speculative-candidate passes (each pass = ≤2 decodes). */
    public static final int SPECULATIVE_CANDIDATE_COOLDOWN_MS = 800;
    /** top1 must reach this to commit a bootstrap language. */
    public static final float BOOTSTRAP_THRESHOLD = 0.70f;
    /**
     * Minimum provisional transcript size (LanguageIdEngine.textUnits) before
     * ScriptLid may commit the label off the 2e output's script/case.
     * ponytail: fixed 3-unit bar, not measured on-device; tune with field logs
     * ("script LID picked") if one-word fillers mislabel.
     */
    public static final int SCRIPT_LID_MIN_UNITS = 3;
    /** top1 - top2 must reach this (never force VI on max-probability). */
    public static final float BOOTSTRAP_MARGIN = 0.15f;
    /**
     * Unified bar for the speculative-candidate path (2026-09-16 fix): a
     * one-shot decode used to commit a language at 0.30 (0.65*conf +
     * 0.35*lexicalFit) while the acoustic gate demands 0.70 — so every
     * rejected LID fell through to a low-confidence commit, and because the
     * candidate pair defaulted to [VI, EN] that commit was almost always
     * Vietnamese. Candidates now need the same evidence level as an acoustic
     * commit.
     */
    public static final float CANDIDATE_COMMIT_SCORE = 0.65f;
    /** Winner must beat the runner-up by this (no coin-flip commits). */
    public static final float CANDIDATE_MARGIN = 0.10f;
    /**
     * Minimum supported-relative score before the provisional (hidden)
     * decoder follows uncertain LID evidence away from VI. Below this the
     * pipeline keeps the current provisional audible (avoids silence on
     * noisy starts); when the evidence names a non-resident engine the UI
     * stays quiet instead of decoding on a guess — the old behaviour
     * hard-coded VI here.
     */
    public static final float PROVISIONAL_MIN_SUPPORTED_REL = 0.60f;
    /**
     * Max bootstrap/referee classifies per decision phase (§6.1). The budget
     * is refreshed on every commit, so a weak fallback commit gets its own 6
     * referee passes. The dynamic-window ECAPA pass is up to ~4× the cost of
     * the 480 ms runtime window; after the budget the runtime LID and
     * endpoint verification remain as the correction paths (CPU bound, not a
     * correctness bound).
     */
    public static final int REFEREE_MAX_ATTEMPTS = 6;
    /** bootstrapScore = 0.90 * acoustic + 0.10 * prior (text/history off). */
    public static final float BOOTSTRAP_W_ACOUSTIC = 0.90f;
    public static final float BOOTSTRAP_W_PRIOR = 0.10f;

    // --- Runtime LID weights (VoxLingua pipeline §10, starting point) ---
    // Kept separate from W_* below (which the current tuned pipeline + tests
    // rely on for the heuristic fallback). Once the VoxLingua acoustic scorer
    // is resident, LanguageIdEngine prefers these: acoustic is PRIMARY,
    // text/ASR-confidence are secondary, history only stabilizes.
    // ASR confidence only says the model trusts its own hypothesis, not that
    // the audio is that language — so its weight stays minimal.
    public static final float RUNTIME_W_ACOUSTIC = 0.65f;
    public static final float RUNTIME_W_TEXT = 0.20f;
    public static final float RUNTIME_W_CONFIDENCE = 0.05f;
    public static final float RUNTIME_W_HISTORY = 0.10f;

    public static final float SWITCH_THRESHOLD = 0.72f;
    public static final float SWITCH_MARGIN = 0.20f;
    /** Endpoint (inter-utterance) bar, spec §13.1: boundaries are cheap
     *  switching points — no rollback, no committed-text rewrite. */
    public static final float ENDPOINT_THRESHOLD = 0.60f;
    public static final float ENDPOINT_MARGIN = 0.10f;
    /** Hysteresis persistence: condition must hold this long (spec §12). */
    public static final long SWITCH_PERSIST_MS = 200L;
    public static final long SWITCH_PERSIST_MAX_MS = 300L;

    // --- Rollback (spec §14, MUST 6) ---
    public static final int ROLLBACK_MIN_MS = 640;
    public static final int ROLLBACK_MAX_MS = 960;
    public static final int ROLLBACK_DEFAULT_MS = 640;

    // --- Transcript policy (spec §18, OPTIONAL 8) ---
    public static final int TOKEN_STABLE_UPDATES = 2;
    /** EMA smoothing for confidences (spec OPTIONAL 7). */
    public static final float EMA_ALPHA = 0.4f;

    // --- VAD / endpointing (spec §8) ---
    public static final float VAD_SPEECH_THRESHOLD = 0.5f;
    /**
     * Trailing silence that finalizes an utterance. Raised 600 → 1000 →
     * 2000 ms: natural mid-sentence pauses must not split one sentence into
     * several separately translated fragments. A deliberate Stop flushes
     * immediately, so this only delays hands-free endpointing.
     */
    public static final int ENDPOINT_SILENCE_MS = 2000;
    /**
     * Only this much of the endpoint wait after the last VAD-speech window is
     * fed to the ASR; the rest is held (utterance stays open) but not decoded.
     * Feeding the whole 2 s of room noise made the 2e model invent words.
     */
    public static final int SPEECH_FEED_HANGOVER_MS = 300;
    /**
     * Zero-pad fed at every flush (endpoint / Stop / force) so the streaming
     * transducer emits its held-back look-ahead (2e model card: 32 frames + 7
     * right context ~ 0.66 s; ConfLive TAIL_SECONDS). Rounded up to whole
     * scheduler chunks by the flush loop (5 x 160 ms = 800 ms).
     */
    public static final int TAIL_PAD_SAMPLES = SAMPLE_RATE * 660 / 1000;

    /**
     * Endpoint candidate verification (spec §13.1): replay up to this much
     * recent audio through the non-active models. The text-only router cannot
     * observe other languages (single active decoder, no acoustic LID model),
     * so without this a whole utterance in another language locks to the
     * wrong model forever. Two decodes per utterance — never 3 parallel
     * decoders on the live path (spec §31).
     */
    public static final int ENDPOINT_VERIFY_MS = 6000;
    /**
     * Switch margin applied to the COMBINED endpoint score (confidence +
     * lexical fit), not confidence alone: a cross-lingual decode can be
     * nearly right ("HELLO" via the VI model) with middling confidence, and
     * a pure-confidence margin locks that in forever.
     */
    public static final float ENDPOINT_VERIFY_MARGIN = 0.05f;
    /** Weight of lexical fit inside the combined endpoint score. */
    public static final float ENDPOINT_VERIFY_LEX_W = 0.35f;
    /**
     * Single-token candidates are accepted only at/above this lexical fit
     * (e.g. a lone dictionary word like "HELLO"); otherwise MIN_TOKENS
     * applies. Genuinely ambiguous one-word blips stay put (stability bias).
     */
    public static final float ENDPOINT_VERIFY_STRONG_LEX = 0.45f;
    /** Candidate transcript must carry at least this many text units. */
    public static final int ENDPOINT_VERIFY_MIN_TOKENS = 2;

    // --- Latency targets (spec §19, engineering targets) ---
    public static final int LATENCY_P50_TARGET_MS = 350;
    public static final int LATENCY_P95_TARGET_MS = 500;

    // --- Streaming re-translation (no model change) ---
    // Endpoint-independent segmentation: long continuous speech must commit
    // translations WITHOUT waiting for 2s silence. Mirrors StreamSpeech
    // policy shape (stride/lag) using stable-token counts as the cheap
    // every-frame signal and capped AR budgets as the expensive step.
    //
    // Latency tuning (2026-10-06, GGUF-fallback field log): GGUF decodes run
    // ~2.7-5s (prefill ~1.4-3.2s + gen ~1.3-4.1s), so segments must stay
    // SHORT. 200 chars ~= 15-20s of speech but 5-7s of MT — the gap users
    // feel. 120 chars ~= 8-12s speech, ~2.5-3.5s MT. 8s max age caps the
    // worst case even for slow/paused speech.
    /** Force-commit a live segment after this much continuous speech. */
    public static final int STREAM_MAX_SEGMENT_MS = 8000;
    /** Force-commit a live segment after this many source chars. */
    public static final int STREAM_MAX_SEGMENT_CHARS = 120;
    /** Word cut held this many chars past the limit while an address may be arriving... */
    public static final int EMAIL_HOLD_CHARS = 80;
    /** ...i.e. an email word sits in the newest this many words (a Han char = one word). */
    public static final int EMAIL_HOLD_WORDS = 20;
    /** A sentence end followed by this many words commits at once (punctuated ASR only). */
    public static final int SENTENCE_SETTLE_WORDS = 2;
    /**
     * Hội nghị / Hai người rows: settled sentences gather until the row holds this many
     * chars (~2 short sentences) instead of one row per sentence ("Oh." "Wow." rows read
     * choppy); an endpoint still ends the row. Live keeps one sentence per MT run.
     */
    public static final int CONF_ROW_MIN_CHARS = 80;
    /** Conference forced cuts (Live: STREAM_MAX_SEGMENT_*); 200 chars still fit the 256-slot decode. */
    public static final int CONF_MAX_SEGMENT_CHARS = 200;
    public static final int CONF_MAX_SEGMENT_MS = 15000;
    /** Live-MT cadence: just above one GGUF tail decode (~2.5s). */
    public static final long LIVE_MT_DEBOUNCE_MS = 1200;
    /** Minimum new source text before re-submitting live MT. */
    public static final int LIVE_MT_MIN_DELTA_CHARS = 8;
    /** Minimum new source words before re-submitting live MT. */
    public static final int LIVE_MT_MIN_DELTA_WORDS = 2;
    /** Live tail budget: 1-2 sentences, caps worst-case gen time. */
    public static final int LIVE_MT_MAX_TOKENS = 64;
    /** Commit budget: matches TranslationModule single-call path. */
    public static final int COMMIT_MT_MAX_TOKENS = 128;
    /**
     * Commit MT context (NPU only): the previous commit's source + forced translation
     * prefix is used when it costs at most this many extra 64-token prefill runs.
     * 0 = never slower than without context; 1 = at most one AR64 run.
     * 0 since 2026-10-29: the extra run costs ~65 ms per commit (D:\apk27.txt prefillMs
     * 57-78 one run vs 121-159 two); there 35 of 88 Live commits fit free, 40 needed it.
     */
    public static final int MT_CONTEXT_EXTRA_RUNS = 0;
    /** Display-only mask: hide trailing unstable tokens from UI only. */
    public static final int DISPLAY_MASK_K = 3;
    /** SSBD biased-verification strength (paper Table 3 sweet spot). */
    public static final float SSBD_BETA = 0.2f;
    /**
     * SSBD assist master switch (paper §3.2 + §3.3: unbiased verify + KV
     * resume plus β-biased accept, no auxiliary model). Back OFF after the
     * 2026-10-07 on-device run measured accepted=0/32 across three verifies
     * (A/D 0%, slower than plain: verify-forward + full regen). Prime
     * suspect: Tatoeba few-shot examples change the prompt between the
     * draft-generating call and the verifying call, so position 0 already
     * diverges — drafts are now gated on an examples fingerprint as well.
     * Re-measure before re-enabling (bar: A/D ≥ 55%, speedup ≥ 1.2x).
     * QNN backend stays OFF until CPU path proves out. β≥0.5 converges
     * toward Continuous Decoding and degrades COMET — do not raise without
     * a quality gate.
     */
    public static final boolean SSBD_ENABLED = false;
    /** Cap draft ids offered to the verify pass (paper draft length m). */
    public static final int SSBD_DRAFT_MAX_TOKENS = 128;

    // --- Model asset names (spec §4, §33 + VoxLingua pipeline §12) ---
    /**
     * ONE trilingual model serves VI+EN+ZH
     * (lmcu000 zh-en-vi stage-2e release, read from the repo-root folder
     * `zipformer zh-en-vi onnx phaseB 2e/` and staged into assets/ by
     * optimize/14_fetch_lmcu000_trilingual.py --stage-assets).
     * ONE session decodes all three labels: {@link AsrModelType#TRILINGUAL}
     * is the pool key, so a LID flip between any labels neither loads a
     * model nor restarts a stream. `model_type` for these assets is
     * "zipformer" (v1) — see transducerModelType().
     *
     * FP32 encoder+joiner (not the .int8 twins): under sherpa 1.13.4 /
     * ORT 1.27 the transducer decoded EMPTY on the Snapdragon target (every
     * decode no tokens, confidence stuck at 0.5) in BOTH precisions — while
     * the same files are parity-clean on desktop (see README 9). The app now
     * runs sherpa 1.13.8 (ORT 1.28.2, static-link AAR) carrying the upstream
     * fix (k2-fsa/sherpa-onnx#3935 for #3845); FP32 stays the default until
     * an on-device decode check proves otherwise. The .int8 twins (~2x faster
     * on CPU, 3-6% relative more errors per the 2e model card) are NOT staged
     * any more (APK -125 MB); switching precision = point these names at the
     * int8 encoder/joiner in script 14 after that device check.
     */
    public static final String VNENZH_ENCODER = "zipformer_vnenzh_encoder.onnx";
    public static final String VNENZH_DECODER = "zipformer_vnenzh_decoder.onnx";
    public static final String VNENZH_JOINER = "zipformer_vnenzh_joiner.onnx";
    public static final String VNENZH_TOKENS = "zipformer_vnenzh_tokens.txt";
    public static final String VAD_MODEL = "silero_vad.onnx";

    // --- VoxLingua107 ECAPA acoustic LID (VoxLingua pipeline §11–§13) ---
    // Community ONNX export (beginning-ai/...-onnx, ~85 MB on disk,
    // ~110–150 MB working set). FP32 first: validate accuracy → benchmark
    // latency/RAM → then try INT8 (VI↔EN confusion is the critical pair).
    // Frontend: 16 kHz mono, 60-bin FBank (see VoxLinguaFbankExtractor).
    // Do NOT quantize ECAPA before the FP32 parity check passes.
    public static final String VOXLINGUA_MODEL = "voxlingua_lid_ecapa.onnx";
    public static final String VOXLINGUA_LABELS_JSON = "voxlingua_lid_labels.json";
    public static final int VOXLINGUA_N_MELS = 60;
    /** Absolute global-top floor: below this the window is too flat to trust
     *  even when the supported-relative margin passes (107-way uniform ≈
     *  0.009). Starting point for tuning, not a guarantee.
     *
     *  LEGACY (2026-09-16): the bootstrap gate no longer reads this — the
     *  107-class argmax proved unreliable at the pipeline's window sizes, so
     *  the gate moved to VOXLINGUA_MIN_SUPPORTED_ABS_SCORE. Kept because the
     *  value still documents the "too flat to trust" floor and the Python
     *  reference exports it. */
    public static final float VOXLINGUA_MIN_GLOBAL_SCORE = 0.25f;
    /**
     * Minimum EMA absolute probability of the supported winner (en/vi/zh) for
     * a bootstrap commit (2026-09-16 LID-bootstrap fix). Replaces the old
     * "the 107-class argmax must be en/vi/zh" requirement: on a 400–600 ms
     * window the ECAPA argmax is regularly an unrelated language
     * (eu/br/cy/nn/ja/lo) while the supported-relative evidence is already
     * 0.93–1.00, so the argmax rule rejected every English/Chinese utterance
     * and left the VI-biased fallback in charge.
     *
     * Measured on the bundled samples: abs(en) ≈ 0.12 @600 ms, 0.78 @1500 ms;
     * abs(vi) ≈ 0.88–0.97 @600–1000 ms (hence the 1500 ms window above).
     * The high absolute value protects the §14 policy the argmax rule was
     * written for: Japanese audio scores abs(zh) ≈ 0.05, far below this bar.
     */
    public static final float VOXLINGUA_MIN_SUPPORTED_ABS_SCORE = 0.45f;
    /**
     * Foreign-audio guard for the SLOW bootstrap path (2026-09-16 v2): a
     * STRONG unsupported 107-class argmax means true foreign audio — wait
     * instead of forcing VI/EN/ZH. Weak/flat argmax tops (the normal
     * short-window case on device mic audio, e.g. nn 0.26) do NOT block the
     * slow path. Mirrors backend config.
     */
    public static final float VOXLINGUA_FOREIGN_TOP_REJECT = 0.60f;
    /**
     * EMA coefficient for the smoothed absolute supported posteriors
     * (VoxLinguaTemporalSmoother §16). Lower than EMA_ALPHA would smooth
     * more; 0.4 keeps the referee inside the 200 ms hop budget.
     */
    public static final float LID_EMA_ALPHA = 0.4f;
}
