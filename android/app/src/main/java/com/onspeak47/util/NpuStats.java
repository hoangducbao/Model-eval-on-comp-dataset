package com.onspeak47.util;

import java.util.Map;
import java.util.concurrent.atomic.AtomicLongArray;

import ai.onnxruntime.OnnxTensorLike;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * App-side Hexagon NPU duty cycle: wall time spent inside QNN HTP graph runs
 * (host I/O included). Android exposes no system-wide NPU load to apps, so
 * the Performance page shows this instead.
 */
public final class NpuStats {

    public static final int ASR = 0, MT = 1;
    private static final AtomicLongArray BUSY_NS = new AtomicLongArray(2);
    private static final AtomicLongArray RUNS = new AtomicLongArray(2);
    /** Set once that engine's HTP contexts have loaded. */
    public static volatile boolean asrOnNpu, mtOnNpu;

    private NpuStats() {}

    public static OrtSession.Result run(OrtSession s, Map<String, ? extends OnnxTensorLike> feed,
                                        int which) throws OrtException {
        return run(s, feed, null, which);
    }

    /** {@code pinned}: outputs written into these tensors (null = ORT allocates them). */
    public static OrtSession.Result run(OrtSession s, Map<String, ? extends OnnxTensorLike> feed,
                                        Map<String, ? extends OnnxValue> pinned, int which) throws OrtException {
        NpuClock clock = OrtSessionConfig.clock;
        if (clock != null) clock.begin();   // long-session mode: high clock vote back before the run
        long t0 = System.nanoTime(), t1 = t0;
        try {
            return pinned == null ? s.run(feed) : s.run(feed, pinned);
        } finally {
            t1 = System.nanoTime();
            if (clock != null) clock.end(t1);
            BUSY_NS.addAndGet(which, t1 - t0);
            RUNS.incrementAndGet(which);
        }
    }

    public static long busyNs(int which) {
        return BUSY_NS.get(which);
    }

    public static long runs(int which) {
        return RUNS.get(which);
    }
}
