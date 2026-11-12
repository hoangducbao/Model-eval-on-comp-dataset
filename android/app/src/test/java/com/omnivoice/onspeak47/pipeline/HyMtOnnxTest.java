package com.omnivoice.onspeak47.pipeline;

import org.junit.Test;

import java.io.File;

import static org.junit.Assert.*;

/**
 * JVM tests for the enhanced HY-MT INT8 ONNX translation path (no device,
 * no 2.1 GB model needed):
 *
 * <ul>
 *   <li>Prompt/budget/cleanup parity with backend/translation_hymt*.py.</li>
 *   <li>BPE golden vectors from the Python AutoTokenizer on the same
 *       tokenizer.json (skipped when the file is absent — it is gitignored
 *       and restored via {@code python optimize/09_stage_hymt_int8.py}).</li>
 * </ul>
 */
public class HyMtOnnxTest {

    // ------------------------------------------------------------------
    // Prompt parity (mirrors backend/translation_hymt.py)
    // ------------------------------------------------------------------

    @Test
    public void buildInstruction_nonZh_usesEnglishTemplate() {
        String out = HyMtOnnxModule.buildInstruction("Hello.", "en", "vi");
        assertEquals("Translate the following text into Vietnamese. Note that you should only"
                + " output the translated result without any additional explanation:\n\nHello.", out);
    }

    @Test
    public void buildInstruction_zhInvolved_usesChineseTemplate() {
        String out = HyMtOnnxModule.buildInstruction("你好。", "zh_hans", "vi");
        assertEquals("将以下文本翻译为越南语，注意只需要输出翻译后的结果，不要额外解释：\n\n你好。",
                out);
    }

    @Test
    public void buildInstruction_targetZhTraditional() {
        String out = HyMtOnnxModule.buildInstruction("Xin chào.", "vi", "zh_hant");
        assertTrue(out.startsWith("将以下文本翻译为繁体中文"));
    }

    @Test
    public void estimateMaxTokens_shortAsrGetsSmallBudget() {
        assertEquals(64, HyMtOnnxModule.estimateMaxTokens("xin chào"));
        assertEquals(128, HyMtOnnxModule.estimateMaxTokens(repeat('a', 200)));
        assertEquals(256, HyMtOnnxModule.estimateMaxTokens(repeat('a', 500)));
    }

    @Test
    public void clean_stripsChatMarkersAndReasoningTags() {
        assertEquals("hello", HyMtOnnxModule.clean("hello<｜hy_foo"));
        // Only the tags are stripped (inner content kept) — mirrors Python _clean.
        assertEquals("xhi there", HyMtOnnxModule.clean("<think>x</think>hi there"));
        assertEquals("", HyMtOnnxModule.clean(null));
    }

    // ------------------------------------------------------------------
    // BPE golden vectors (Python: AutoTokenizer on tokenizer.json)
    // ------------------------------------------------------------------

    private static HyMtBpeTokenizer loadTokenizer() {
        String path = System.getProperty("onspeak.hymtTokenizer");
        org.junit.Assume.assumeTrue("onspeak.hymtTokenizer not set", path != null);
        File f = new File(path);
        org.junit.Assume.assumeTrue("tokenizer.json absent: " + path, f.isFile());
        HyMtBpeTokenizer tok = new HyMtBpeTokenizer(f);
        assertNull(tok.getInitError(), tok.getInitError());
        assertTrue(tok.isReady());
        return tok;
    }

    @Test
    public void bpe_helloWorld_matchesPython() {
        HyMtBpeTokenizer tok = loadTokenizer();
        assertArrayEquals(new int[]{16883, 2385, 13}, tok.encode("Hello world."));
    }

    @Test
    public void bpe_specialMarkersAreSingleIds() {
        HyMtBpeTokenizer tok = loadTokenizer();
        assertArrayEquals(new int[]{120006},
                tok.encode(HyMtBpeTokenizer.USER_MARK));
        assertArrayEquals(new int[]{120007},
                tok.encode(HyMtBpeTokenizer.ASSISTANT_MARK));
        assertArrayEquals(new int[]{120006, 16883, 2385, 13, 120007},
                tok.encode(HyMtBpeTokenizer.USER_MARK + "Hello world."
                        + HyMtBpeTokenizer.ASSISTANT_MARK));
    }

    @Test
    public void bpe_fullViPrompt_matchesPython() {
        HyMtBpeTokenizer tok = loadTokenizer();
        int[] ids = tok.encode(HyMtBpeTokenizer.USER_MARK
                + "Translate the following text into English. Note that you should only"
                + " output the translated result without any additional explanation:\n\n"
                + "Tôi muốn gia hạn căn cước công dân."
                + HyMtBpeTokenizer.ASSISTANT_MARK);
        // HF tokenizers (Hy-MT2 tokenizer.json, byte-identical to HY-MT1.5's)
        assertArrayEquals(new int[]{
                120006, 26403, 252, 1873, 2602, 1079, 4717, 13, 10306, 369,
                444, 1029, 1370, 3602, 252, 17960, 1398, 2553, 1232, 5682,
                12509, 1015, 51, 40509, 17284, 74562, 56849, 307, 29994, 54285,
                77, 258, 34412, 42792, 93590, 13, 120007}, ids);
    }

    @Test
    public void bpe_digitsCjkMixed_matchesPython() {
        HyMtBpeTokenizer tok = loadTokenizer();
        assertArrayEquals(new int[]{
                50, 12935, 206, 7827, 11, 38423, 206, 1717, 19, 0, 11530,
                206, 11015, 5998, 14833, 13},
                tok.encode("Số 123, năm 2024! mixed 中文测试 ABC."));
    }

    @Test
    public void bpe_decodeRoundTrip_skipsSpecials() {
        HyMtBpeTokenizer tok = loadTokenizer();
        assertEquals("Hello world.", tok.decode(new int[]{16883, 2385, 13}));
        assertEquals("Hello world.",
                tok.decode(new int[]{120006, 16883, 2385, 13, 120007}));
        assertEquals("Tôi muốn gia hạn căn cước công dân.",
                tok.decode(new int[]{51, 40509, 17284, 74562, 56849, 307, 29994,
                        54285, 77, 258, 34412, 42792, 93590, 13}));
    }

    // ------------------------------------------------------------------
    // SSBD Stage B biased verification (paper §3.3 Eq.3, no ORT needed)
    // ------------------------------------------------------------------

    @Test
    public void ssbdUnbiased_acceptsOnlyArgmax() {
        float[] row = new float[]{1.0f, 3.0f, 2.0f};
        assertTrue(HyMtOnnxModule.acceptsDraftToken(row, 1, 0f));
        assertTrue(!HyMtOnnxModule.acceptsDraftToken(row, 2, 0f));
        assertTrue(!HyMtOnnxModule.acceptsDraftToken(row, 0, 0f));
    }

    @Test
    public void ssbdBeta02_acceptsCloseRunnerUp_rejectsFarBehind() {
        // best=1 (3.0), close runner-up=2 (2.9): P gap ~ small → biased accept.
        assertTrue(HyMtOnnxModule.acceptsDraftToken(
                new float[]{0f, 3.0f, 2.9f}, 2, 0.2f));
        // best=1 (5.0) vs draft=2 (0.0): P gap ≈ 0.99 > 0.25 → reject.
        assertTrue(!HyMtOnnxModule.acceptsDraftToken(
                new float[]{0f, 5.0f, 0.0f}, 2, 0.2f));
        // Argmax draft always accepted, even with bias.
        assertTrue(HyMtOnnxModule.acceptsDraftToken(
                new float[]{0f, 5.0f, 0.0f}, 1, 0.2f));
    }

    @Test
    public void ssbdBetaBounds_clamped() {
        float[] row = new float[]{0f, 5.0f, 0.0f};
        assertTrue(HyMtOnnxModule.acceptsDraftToken(row, 2, 1.0f));
        assertTrue(!HyMtOnnxModule.acceptsDraftToken(row, 2, -1.0f));
        assertTrue(!HyMtOnnxModule.acceptsDraftToken(row, 99, 0.2f));
    }

    @Test
    public void ssbdExamplesKey_emptyOnlyMatchesEmpty() {
        assertEquals("", TranslationModule.examplesKey(null));
        assertEquals("", TranslationModule.examplesKey(
                java.util.Collections.emptyList()));
        java.util.List<String[]> a = java.util.Collections.singletonList(
                new String[]{"xin chào", "hello"});
        java.util.List<String[]> b = java.util.Collections.singletonList(
                new String[]{"xin chào", "hello"});
        java.util.List<String[]> c = java.util.Collections.singletonList(
                new String[]{"xin chào", "hi"});
        assertEquals(TranslationModule.examplesKey(a), TranslationModule.examplesKey(b));
        assertTrue(!TranslationModule.examplesKey(a).equals(TranslationModule.examplesKey(c)));
        assertTrue(!TranslationModule.examplesKey(a).isEmpty());
    }

    private static String repeat(char c, int n) {
        StringBuilder sb = new StringBuilder(n);
        for (int i = 0; i < n; i++) sb.append(c);
        return sb.toString();
    }
}
