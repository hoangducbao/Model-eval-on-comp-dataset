package com.omnivoice.onspeak47.util;

/**
 * Greedy-decode loop guard for the NPU decoders (argmax only, so no sampling or
 * penalty can break a loop). D:\apk23 (1).txt: Qwen3-ASR wrote "là có thể" ~50 times
 * (53 decode runs a step, up to 3.2 s), the 180-char forced cuts of it went to MT twice
 * each and came back as 128 tokens of "It could be possible that ..." (6-13 s a segment).
 * Twin: backend/streaming_asr/qwen3asr_npu.loop_cut.
 */
public final class LoopGuard {

    /** A loop = a phrase of k <= MAX_PHRASE tokens, r >= MIN_REPEATS times in a row, r * k >= MIN_TOKENS. */
    public static final int MIN_REPEATS = 4, MIN_TOKENS = 16, MAX_PHRASE = 16;

    private LoopGuard() {}

    /** Length of t[0, len) without a trailing loop (every copy dropped), or -1 when it ends in none. */
    public static int cut(int[] t, int len) {
        for (int k = 1; k <= MAX_PHRASE && 2 * k <= len; k++) {
            int run = k;   // the tail t[len - run, len) has period k
            while (run < len && t[len - run - 1] == t[len - run - 1 + k]) run++;
            int r = run / k;
            if (r >= MIN_REPEATS && r * k >= MIN_TOKENS) return len - r * k;
        }
        return -1;
    }
}
