/*
 * OmniVoice — Streaming ASR shared state & tuning constants.
 *
 * Spec: realtime_multilingual_asr_vi_en_zh_android.md §12, §19-§20, §23.
 * Pure-Java (no android.* imports) so it runs in local JVM unit tests.
 */
package com.omnivoice.onspeak47.asr;

/** Languages handled by the streaming router. */
public enum AsrLanguage {
    VI("vi"),
    EN("en"),
    ZH("zh"),
    UND("und");

    public final String code;

    AsrLanguage(String code) {
        this.code = code;
    }

    public static AsrLanguage fromCode(String code) {
        if (code == null) return UND;
        switch (code) {
            case "vi": return VI;
            case "en": return EN;
            case "zh": return ZH;
            default: return UND;
        }
    }

    /**
     * Model bucket that decodes this language (single trilingual model):
     * VI, EN and ZH all map to {@link AsrModelType#TRILINGUAL}. Routing
     * decisions never key on this — the language label is only used for
     * display, LID math and the translation target.
     */
    public AsrModelType modelType() {
        return AsrModelType.of(this);
    }

    /** Canonical language label for a model bucket (for engine construction). */
    public static AsrLanguage canonicalOf(AsrModelType type) {
        return type == null ? UND : type.canonical;
    }
}
