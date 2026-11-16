package com.onspeak47.util;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/**
 * Long-session mode: the Hexagon NPU's max-clock vote is held only while graphs run.
 *
 * Every NPU session used to vote sustained_high_performance when it loaded and kept that vote
 * until the app died: DCVS off, max core + bus corner, 40 us sleep latency. In D:\live2.txt the
 * NPU ran graphs 41 % of a 19 min session; the other 59 % (56 % in gaps of 200 ms or more) it
 * idled at that vote, as it does while the app just sits open. Qualcomm's own samples vote down
 * after inference. Here the big models load with no vote at all (the plugin then sets no power
 * config, checked on the x64 HTP emulator) and one tiny session (assets/npu_vote, one Relu)
 * carries the vote as a per-run option (qnn.htp_perf_mode): high before the first run of a
 * burst, {@link #LOW} once nothing ran for {@link #IDLE_MS}. Same graphs, same clock while they
 * run, so the output and the per-run time do not change; a burst pays one ~1 ms vote run first.
 *
 * Hardware check instead of trust: HTP votes should add up across contexts (max wins), but if
 * they did not, vote-less models would crawl. So at start a second tiny vote-less graph
 * (assets/npu_probe) is timed under the high and the low vote. Only if low is at least
 * {@link #MIN_SLOWDOWN} slower does the vote session rule the clock of other contexts; else
 * this returns null and every session votes for itself as before. Logcat `NpuClock: probe`.
 */
public final class NpuClock {

    private static final String TAG = "NpuClock";
    /** Runs closer than this are one burst. Lower = more idle time at the low vote, more vote runs. */
    static final long IDLE_MS = 300;
    /** The idle vote. Any mode below the high one works; this is the plugin's lowest with DCVS on. */
    static final String LOW = "low_power_saver";
    static final float MIN_SLOWDOWN = 1.5f;

    interface Voter {
        void vote(boolean high) throws Exception;
    }

    private final Voter voter;
    private final long idleNs;
    private int running;
    private boolean high = true, broken;
    private long idleSince = Long.MAX_VALUE;   // no down-vote before the first run (models still loading)
    private int ups, downs;

    NpuClock(Voter voter, long idleMs) {
        this.voter = voter;
        this.idleNs = idleMs * 1_000_000;
    }

    /** Before an NPU graph run: back to the high vote if it was released. */
    public synchronized void begin() {
        if (!high) {
            try {
                voter.vote(true);
                high = true;
                ups++;
            } catch (Exception e) {
                Log.w(TAG, "up-vote failed, this run may be slow", e);   // retried by the next run
            }
        }
        running++;
    }

    public synchronized void end(long nowNs) {
        if (--running == 0) idleSince = nowNs;
    }

    /** Releases the vote once no graph ran for the idle time. */
    synchronized void tick(long nowNs) {
        if (!high || broken || running > 0 || nowNs - idleSince < idleNs) return;
        try {
            voter.vote(false);
            high = false;
            downs++;
        } catch (Exception e) {
            broken = true;   // stays at the high vote from here on
            Log.w(TAG, "down-vote failed, clock stays high", e);
        }
    }

    synchronized String stats() {
        return "up=" + ups + " down=" + downs + (broken ? " broken" : "");
    }

    /** @return the running governor, or null: no assets for this SoC, or the probe saw no effect. */
    static NpuClock tryCreate(Context context, String highMode) {
        String soc = Build.VERSION.SDK_INT >= 31 ? Build.SOC_MODEL.toLowerCase() : "";
        OrtSession vote = null, probe = null;
        try {
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            vote = open(context, env, "npu_vote." + soc + ".ctx.onnx", highMode);
            probe = open(context, env, "npu_probe." + soc + ".ctx.onnx", null);
            Map<String, OnnxTensor> in = Collections.singletonMap("x", OnnxTensor.createTensor(env, floats(8), new long[]{1, 8}));
            OrtSession.RunOptions up = new OrtSession.RunOptions(), down = new OrtSession.RunOptions();
            up.addRunConfigEntry("qnn.htp_perf_mode", highMode);
            down.addRunConfigEntry("qnn.htp_perf_mode", LOW);
            final OrtSession v = vote;
            Voter voter = h -> v.run(in, h ? up : down).close();

            float hi, lo, voteMs;
            try (OnnxTensor x = OnnxTensor.createTensor(env, floats(1024), new long[]{1, 1, 1024})) {
                Map<String, OnnxTensor> feed = Collections.singletonMap("x", x);
                hi = time(probe, feed, 2);
                voter.vote(false);
                lo = time(probe, feed, 0);   // no warm-up: DCVS may climb back under load
                long t0 = System.nanoTime();
                voter.vote(true);
                voteMs = (System.nanoTime() - t0) / 1e6f;
            }
            probe.close();
            probe = null;
            boolean rules = lo >= hi * MIN_SLOWDOWN;
            Log.i(TAG, String.format(java.util.Locale.US,
                    "probe high=%.1f ms low=%.1f ms upVote=%.1f ms -> %s (%s / %s, idle %d ms)", hi, lo, voteMs,
                    rules ? "on: models load without a vote" : "off: no effect on other contexts",
                    highMode, LOW, IDLE_MS));
            if (!rules) {
                vote.close();
                return null;
            }
            NpuClock clock = new NpuClock(voter, IDLE_MS);
            Thread t = new Thread(() -> {
                for (long n = 0; ; n++) {
                    try {
                        Thread.sleep(IDLE_MS / 2);
                    } catch (InterruptedException e) {
                        return;
                    }
                    clock.tick(System.nanoTime());
                    if (n % 400 == 399) Log.i(TAG, "votes " + clock.stats());   // once a minute
                }
            }, "NpuClock");
            t.setDaemon(true);
            t.start();
            return clock;
        } catch (Throwable e) {
            Log.i(TAG, "no clock governor (" + e + "), sessions vote for themselves");
            for (OrtSession s : new OrtSession[]{vote, probe}) {
                try {
                    if (s != null) s.close();
                } catch (Exception ignored) {
                }
            }
            return null;
        }
    }

    private static OrtSession open(Context context, OrtEnvironment env, String asset, String perf) throws Exception {
        String path = FileUtils.copyAssetToInternal(context, asset);
        try (OrtSession.SessionOptions so = new OrtSession.SessionOptions()) {
            if (!OrtSessionConfig.addQnnHtp(context, so, perf)) throw new IllegalStateException("no QNN HTP");
            return env.createSession(path, so);
        }
    }

    private static FloatBuffer floats(int n) {
        return ByteBuffer.allocateDirect(4 * n).order(ByteOrder.nativeOrder()).asFloatBuffer();
    }

    /** Mean ms of 3 runs after {@code warm} untimed ones. */
    private static float time(OrtSession s, Map<String, OnnxTensor> feed, int warm) throws Exception {
        for (int i = 0; i < warm; i++) s.run(feed).close();
        long t0 = System.nanoTime();
        for (int i = 0; i < 3; i++) s.run(feed).close();
        return (System.nanoTime() - t0) / 3e6f;
    }
}
