/*
 * OmniVoice — ASR model identity (the unit the pipeline decodes with).
 *
 * Single trilingual model: ONE streaming transducer
 * (lmcu000 zh-en-vi stage-2e release, assets zipformer_vnenzh_*)
 * decodes Vietnamese, English and Chinese inside one session, so there is
 * exactly one resident engine on every device. {@link AsrLanguage} stays the
 * *language* label — VoxLingua107 scores vi/en/zh separately, the UI names
 * the heard language and translation picks the target from it — while
 * AsrModelType answers "which model decodes this", always TRILINGUAL.
 *
 * Because every language maps to the same model:
 *   - a LID flip between ANY labels mid-utterance is not a model switch: no
 *     rollback, no shadow decode, no stream reset — the shared vocabulary
 *     handles code switching inside the model;
 *   - the resident pool holds 1 native session; trimTo() is a no-op.
 *
 * Pure-Java (no android.* imports) so it runs in local JVM unit tests.
 */
package com.omnivoice.onspeak47.asr;

/** Streaming ASR model identity: one entry per loaded native model. */
public enum AsrModelType {
    /** The single trilingual VI+EN+ZH streaming Zipformer. */
    TRILINGUAL("vnenzh", AsrLanguage.VI),
    /** Undefined / unsupported — never a model slot. */
    UND("und", AsrLanguage.UND);

    /** Stable id for logs/telemetry ("vnenzh", "und"). */
    public final String code;
    /**
     * Nominal language label where a concrete {@link AsrLanguage} is still
     * required (engine construction, asset resolution). The engine itself is
     * language-agnostic; this is informational only, never a second model.
     */
    public final AsrLanguage canonical;

    AsrModelType(String code, AsrLanguage canonical) {
        this.code = code;
        this.canonical = canonical;
    }

    /** Model bucket that serves {@code lang}; null/UND → {@link #UND}. */
    public static AsrModelType of(AsrLanguage lang) {
        if (lang == null) return UND;
        switch (lang) {
            case VI:
            case EN:
            case ZH: return TRILINGUAL;
            default: return UND;
        }
    }

    /** True when this bucket is the model that decodes {@code lang}. */
    public boolean covers(AsrLanguage lang) {
        return of(lang) == this;
    }

    /** Parse a stable id; unknown/null → {@link #UND}. */
    public static AsrModelType fromCode(String code) {
        if (code == null) return UND;
        switch (code) {
            case "vnenzh": return TRILINGUAL;
            default: return UND;
        }
    }
}
