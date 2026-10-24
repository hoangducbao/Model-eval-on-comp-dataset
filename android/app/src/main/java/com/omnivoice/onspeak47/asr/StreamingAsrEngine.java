package com.omnivoice.onspeak47.asr;

/**
 * Small contract retained from the reference app.  larplarpsahur only needs
 * the Qwen NPU implementation for file-based, final-result recognition.
 */
public abstract class StreamingAsrEngine {
    public static final class PartialResult {
        public final String text;
        public final float confidence;
        public final boolean endpoint;

        public PartialResult(String text, float confidence, boolean endpoint) {
            this.text = text == null ? "" : text;
            this.confidence = confidence;
            this.endpoint = endpoint;
        }
    }

    public abstract void acceptAudio(float[] samples);
    public abstract void decodeAvailable();
    public abstract boolean isReadyToDecode();
    public abstract PartialResult getPartialResult();
    public abstract PartialResult getFinalResult();
    public abstract float getConfidence();
    public abstract AsrLanguage language();
    public abstract void reset();
    public abstract void close();
    public abstract boolean isFake();

    public void endOfSpeech() {
    }
}
