package com.omnivoice.onspeak47.asr;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/** QwenAsr against the Python twin (log-mel goldens) and loop invariants on a fake model. */
public class QwenAsrTest {

    private static final int V_TEXT = 40;

    @Test
    public void heardNameEnd_keepsOnlyNamesInsideHeardAudio() {
        assertEquals("If the Porsche Macan".length(),
                QwenAsr.heardNameEnd("If the Porsche Macan has proven anything.", 2f));
        assertEquals(-1, QwenAsr.heardNameEnd("If the Porsche Macan has proven anything.", 1f));
        assertEquals(-1, QwenAsr.heardNameEnd("I think so. Then I left it.", 3f));
        assertEquals(-1, QwenAsr.heardNameEnd("We saw Paris", 3f));
        assertEquals("anh Hùng".length(), QwenAsr.heardNameEnd("anh Hùng đi làm", 2f));
    }

    /** qasr_assets.bin layout with synthetic filters / embedding (optimize/18_qwen3asr.write_assets). */
    private static QwenAsr.Assets assets() {
        List<Integer> keep = new ArrayList<>();
        for (int i = 0; i < V_TEXT; i++) keep.add(i);
        keep.addAll(Arrays.asList(198, 872, 8948, 77091));   // newline, user, system, assistant (prompt)
        for (int i = 151643; i <= 151704; i++) keep.add(i);
        int v = keep.size();
        List<byte[]> vocab = new ArrayList<>();
        for (int id : keep) {
            String s = id < V_TEXT ? "w" + id + " " : id == 151704 ? "<asr_text>" : "";
            vocab.add(s.getBytes(StandardCharsets.UTF_8));
        }
        int nv = 0;
        for (byte[] b : vocab) nv += b.length;
        int head = 16 + 4 * v + 4 * v + 4 * (v + 1) + 4 * 201 * 128 + nv;
        head += (64 - head % 64) % 64;
        ByteBuffer b = ByteBuffer.allocate(head + v * 1024).order(ByteOrder.LITTLE_ENDIAN);
        b.put("QASR".getBytes(StandardCharsets.US_ASCII)).putInt(1).putInt(v).putInt(nv);
        for (int id : keep) b.putInt(id);
        for (int i = 0; i < v; i++) b.putFloat(0.01f);
        int off = 0;
        b.putInt(0);
        for (byte[] w : vocab) b.putInt(off += w.length);
        for (int k = 0; k < 201; k++) for (int m = 0; m < 128; m++) b.putFloat(((k * 7 + m * 3) % 11) / 10f);
        for (byte[] w : vocab) b.put(w);
        b.position(head);
        Random r = new Random(1);
        for (int i = 0; i < v * 1024; i++) b.put((byte) (r.nextInt(255) - 127));
        b.rewind();
        return new QwenAsr.Assets(b);
    }

    /** Same signal as the golden script: 440 Hz + 3 kHz, silent from sample 6000. */
    private static float[] signal(int n) {
        float[] x = new float[n];
        for (int t = 0; t < Math.min(n, 6000); t++) {
            x[t] = (float) (0.3 * Math.sin(2 * Math.PI * 440 * t / 16000.0) + 0.1 * Math.sin(2 * Math.PI * 3000 * t / 16000.0));
        }
        return x;
    }

    @Test public void logMel_matchesPythonTwin() {
        QwenAsr asr = new QwenAsr(assets(), new Fake());
        float[] x = signal(8000);
        float[][] shortMel = asr.logMel(x, 5000);   // fills the frame cache first
        float[][] m = asr.logMel(x, 8000);
        assertEquals(50, m[0].length);
        float[][] gold = {  // backend/streaming_asr/qwen3asr_npu.log_mel, bins 0 / 37 / 127
                {1.723083f, 1.735767f, 1.68256f}, {1.636165f, 1.674783f, 1.718111f},
                {1.63287f, 1.672549f, 1.719846f}, {1.648992f, 1.680045f, 1.698568f},
                {-0.232029f, -0.232029f, -0.232029f}, {-0.232029f, -0.232029f, -0.232029f}};
        int[] frames = {0, 1, 25, 37, 48, 49};
        int[] bins = {0, 37, 127};
        for (int i = 0; i < frames.length; i++) {
            for (int j = 0; j < 3; j++) assertEquals(gold[i][j], m[bins[j]][frames[i]], 2e-4f);
        }
        assertEquals(31, shortMel[0].length);
        assertEquals(1.63287f, shortMel[0][30], 2e-4f);
    }

    @Test public void parseAndScriptConvention() {
        assertArrayEquals(new String[]{"English", "I would like it."},
                QwenAsr.parse(" language English<asr_text>I would like it. "));
        assertArrayEquals(new String[]{"", "no tag"}, QwenAsr.parse("no tag"));
        assertArrayEquals(new String[]{"", ""}, QwenAsr.parse("language None<asr_text>"));
        assertEquals("I WOULD LIKE TO RENEW MY CARD.", QwenAsr.scriptConvention("English", "I would like to renew my card."));
        assertEquals("这么FANCY，DON'T WORRY吗", QwenAsr.scriptConvention("Chinese", "这么fancy，don't worry吗"));
        assertEquals("Tôi muốn mua iPhone", QwenAsr.scriptConvention("Vietnamese", "Tôi muốn mua iPhone"));
        // Vietnamese syllables inside an English row keep their case (non-ASCII letters)
        assertEquals("CALL Nguyễn NOW", QwenAsr.scriptConvention("English", "call Nguyễn now"));
        assertEquals("en", ScriptLid.label(QwenAsr.scriptConvention("English", "I would like to renew my card.")));
        // A Vietnamese sentence in an English-tagged part stays as written (D:\apk32.txt: `mình XEM`, `người TA`)
        assertEquals("WOW. Nếu mà mình xem một người đó là người thân thì người ta nói.", QwenAsr.scriptConvention(
                "English", "Wow. Nếu mà mình xem một người đó là người thân thì người ta nói."));
        assertEquals("Máu, BLOOD, mồ BOOGERS, ROPE.", QwenAsr.scriptConvention("English", "Máu, blood, mồ boogers, rope."));
    }

    @Test public void contextGoesIntoTheSystemText() throws Exception {
        QwenAsr asr = new QwenAsr(assets(), new Fake());
        int plain = asr.head.length;
        asr.context(Arrays.asList("w3 w12 ", "zz"));   // "zz" has no tokens: skipped, not fatal
        assertEquals(plain + 2, asr.head.length);
        assertEquals("w3 w12 ", asr.decode(Arrays.asList(asr.head[3], asr.head[4])));
        assertEquals(20, asr.transcribe(signal(16000 * 3), new ArrayList<>(), new ArrayList<>(), 20).size());
    }

    @Test public void draftNeverChangesTheOutput() throws Exception {
        float[] x = signal(16000 * 3);
        List<Integer> plain = new QwenAsr(assets(), new Fake()).transcribe(x, new ArrayList<>(), new ArrayList<>(), 20);
        assertEquals(20, plain.size());
        QwenAsr good = new QwenAsr(assets(), new Fake());
        assertEquals(plain, good.transcribe(x, new ArrayList<>(), new ArrayList<>(plain.subList(0, 12)), 20));
        List<Integer> wrong = new ArrayList<>();
        for (int t : plain.subList(0, 12)) wrong.add((t + 1) % V_TEXT);
        QwenAsr bad = new QwenAsr(assets(), new Fake());
        assertEquals(plain, bad.transcribe(x, new ArrayList<>(), wrong, 20));
        assertTrue("accepted draft tokens skip decode runs", good.decodeRuns < bad.decodeRuns);
    }

    @Test public void reusedSlotsGiveTheFreshResult() throws Exception {
        // 10 s: block 0 finishes (cached) and its loudest frame comes first, so the mel floor holds.
        float[] x = new float[16000 * 10];
        Random r = new Random(7);
        for (int i = 0; i < x.length; i++) x[i] = (i < 16000 ? 0.5f : 0.05f) * (float) Math.sin(i * 0.05 + r.nextGaussian());
        List<Integer> prefix = Arrays.asList(3, 5, 7);
        QwenAsr warm = new QwenAsr(assets(), new Fake());
        warm.transcribe(Arrays.copyOf(x, 16000 * 9), new ArrayList<>(), new ArrayList<>(), 10);
        int runs = warm.prefillRuns;
        List<Integer> a = warm.transcribe(x, prefix, Arrays.asList(1, 2), 15);
        List<Integer> b = new QwenAsr(assets(), new Fake()).transcribe(x, prefix, Arrays.asList(1, 2), 15);
        assertEquals(b, a);
        assertTrue("block 0 rows reused, prefill restarts after them", warm.prefillRuns - runs < 3);
    }

    /** The fake never emits EOS: 64 tokens a step outgrow the cache, which must cut, not fail every step. */
    @Test public void cacheOverflowCutsTheUtterance() throws Exception {
        QwenAsr asr = new QwenAsr(assets(), new Fake());
        float[] x = signal(QwenAsr.SR * 18);
        String t = "";
        for (int n = QwenAsr.CHUNK_SAMPLES; n <= x.length; n += QwenAsr.CHUNK_SAMPLES) t = asr.step(x, 0, n, false);
        assertTrue("cut before the 20 s limit", asr.start > 0 && asr.start < QwenAsr.MAX_UTT_SAMPLES);
        assertTrue(t.length() > 0);
    }

    /** The 20 s limit ends the part at the quiet spot, not at sample n, also with a dropped audio head. */
    @Test public void limitCutsAtTheQuietSpot() throws Exception {
        int gap = QwenAsr.SR * 19;
        float[] x = new float[QwenAsr.SR * 20 + 3200];
        for (int t = 0; t < x.length; t++) x[t] = (float) (0.3 * Math.sin(t * 0.17));
        Arrays.fill(x, gap, gap + 3200, 0f);
        int k = QwenAsr.quietCut(x, QwenAsr.SR * 18, QwenAsr.SR * 20);
        assertTrue("in the gap: " + k, k >= gap && k <= gap + 3200);
        QwenAsr asr = new QwenAsr(assets(), new QwenAsr.Model() {
            @Override public float[] encode(float[] mel, float[] mask) {
                return new float[QwenAsr.ROWS * QwenAsr.HID];
            }

            @Override public long[] decode(float[] x, long[] pos, float[] mask, int slot, int valid) {
                long[] next = new long[pos.length];
                Arrays.fill(next, 46);   // EOS
                return next;
            }
        });
        asr.start = QwenAsr.SR / 10;
        int offset = QwenAsr.SR / 20, n = x.length;
        asr.step(Arrays.copyOfRange(x, offset, n), offset, n, false);
        assertTrue("next part starts in the gap: " + asr.start, asr.start >= gap && asr.start <= gap + 3200);
        assertEquals("you guys just", QwenAsr.dropStop("you guys just."));
        assertEquals("转变", QwenAsr.dropStop("转变。"));
        assertEquals("right?", QwenAsr.dropStop("right?"));
    }

    /** D:\apk32.txt: `gia đình Anh ví dụ` -> "families like those in England". */
    @Test public void partJoinDropsTheOpeningCapital() {
        assertEquals("gia đình anh ví dụ mẹ", QwenAsr.joinPart("gia đình", "Anh ví dụ mẹ"));
        assertEquals("rồi. Anh ví dụ", QwenAsr.joinPart("rồi.", "Anh ví dụ"));          // a sentence start
        assertEquals("ở Việt Nam đẹp", QwenAsr.joinPart("ở", "Việt Nam đẹp"));          // two capitals: a name
        assertEquals("gặp Mita rồi Mita nói", QwenAsr.joinPart("gặp Mita rồi", "Mita nói"));  // seen before
        assertEquals("HELLO GUYS THIS IS", QwenAsr.joinPart("HELLO GUYS", "THIS IS"));
        assertEquals("Anh nghĩ", QwenAsr.joinPart("", "Anh nghĩ"));
    }

    /** A greedy loop is dropped and restarts the utterance instead of riding on as the forced prefix. */
    @Test public void repetitionLoopIsCutOut() throws Exception {
        int[] loop = {1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5};   // 6 x 3 tokens
        assertEquals(2, com.omnivoice.onspeak47.util.LoopGuard.cut(loop, loop.length));
        assertEquals(-1, com.omnivoice.onspeak47.util.LoopGuard.cut(loop, 17));      // 5 x 3 < 16 tokens
        int[] thrice = {6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11};  // said 3 times: kept
        assertEquals(-1, com.omnivoice.onspeak47.util.LoopGuard.cut(thrice, thrice.length));
        QwenAsr asr = new QwenAsr(assets(), new QwenAsr.Model() {
            @Override public float[] encode(float[] mel, float[] mask) {
                return new float[QwenAsr.ROWS * QwenAsr.HID];
            }

            @Override public long[] decode(float[] x, long[] pos, float[] mask, int slot, int valid) {
                long[] next = new long[pos.length];
                for (int j = 0; j < next.length; j++) next[j] = 5 + pos[j] % 3;   // w5 w6 w7 w5 ...
                return next;
            }
        });
        int n = QwenAsr.SR * 3;
        assertEquals("", asr.step(signal(n), 0, n, false));
        assertTrue("stopped at the loop, not the 64-token budget", asr.decodeRuns < 20);
        assertEquals("utterance restarts after the loop", n, asr.start);
    }

    @Test public void maskShowsPastThenCausal() {
        float[] m = QwenAsr.mask(3, 2);
        int w = QwenAsr.CTX + 2;
        assertEquals(0f, m[2], 0);
        assertEquals(QwenAsr.MASKED, m[3], 0);
        assertEquals(0f, m[QwenAsr.CTX], 0);
        assertEquals(QwenAsr.MASKED, m[QwenAsr.CTX + 1], 0);
        assertEquals(0f, m[w + QwenAsr.CTX + 1], 0);
    }

    /**
     * Toy model with the graphs' cache contract: each row's key comes from its embedding and
     * position, the next id from every key the mask lets it see (weighted by slot), so a stale,
     * missing or misplaced cache slot changes the output. Never emits EOS.
     */
    private static final class Fake implements QwenAsr.Model {
        private final double[] keys = new double[QwenAsr.CTX];

        @Override public float[] encode(float[] mel, float[] mask) {
            float[] out = new float[QwenAsr.ROWS * QwenAsr.HID];
            for (int r = 0; r < QwenAsr.ROWS; r++) {
                for (int i = 0; i < QwenAsr.HID; i++) {
                    out[r * QwenAsr.HID + i] = mel[(i % 128) * QwenAsr.BLOCK + Math.min(799, r * 7)] * 0.01f + r * 1e-3f;
                }
            }
            return out;
        }

        @Override public long[] decode(float[] x, long[] pos, float[] mask, int slot, int valid) {
            int ar = pos.length, w = QwenAsr.CTX + ar;
            double[] key = new double[ar];
            for (int j = 0; j < ar; j++) {
                double k = pos[j] * 1e-3;
                for (int i = 0; i < QwenAsr.HID; i += 7) k += x[j * QwenAsr.HID + i] * Math.sin(i);
                key[j] = k;
            }
            long[] next = new long[ar];
            for (int r = 0; r < ar; r++) {
                double h = 0;
                for (int s = 0; s < QwenAsr.CTX; s++) if (mask[r * w + s] == 0) h += keys[s] * (s + 1);
                for (int j = 0; j < ar; j++) if (mask[r * w + QwenAsr.CTX + j] == 0) h += key[j] * (pos[j] + 1);
                next[r] = (long) Math.floor(Math.abs(h) * 1e4) % V_TEXT;
            }
            for (int j = 0; j < valid; j++) keys[slot + j] = key[j];
            return next;
        }
    }
}
