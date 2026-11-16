package com.onspeak47.pipeline;

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.*;

/** LOCAL test (not in OnSpeak47): segmentation of file transcripts for the commit translation path. */
public class SegmentTranslatorTest {
    @Test public void shortTranscriptIsOneSegment() {
        assertEquals(java.util.Arrays.asList("Xin chào. Bạn khỏe không?"),
                SegmentTranslator.segments("  Xin chào.\n Bạn  khỏe không? ", "vi"));
        assertTrue(SegmentTranslator.segments("   ", "vi").isEmpty());
        assertTrue(SegmentTranslator.segments(null, "vi").isEmpty());
    }

    @Test public void longTranscriptIsPackedBySentence() {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < 12; i++) text.append("This is sentence number ").append(i).append(" of the long transcript. ");
        List<String> segments = SegmentTranslator.segments(text.toString(), "en");
        assertTrue(segments.size() > 1);
        StringBuilder joined = new StringBuilder();
        for (String s : segments) {
            assertTrue(s, s.length() <= SegmentTranslator.MAX_SEGMENT_CHARS);
            assertTrue(s, s.endsWith("."));
            joined.append(s).append(' ');
        }
        assertEquals(text.toString().trim(), joined.toString().trim());
    }

    @Test public void oneOverlongSentenceStaysWhole() {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < 60; i++) text.append("word ");
        assertEquals(1, SegmentTranslator.segments(text.toString(), "en").size());
    }

    @Test public void vietnameseSourceGetsTheKinshipPrimer() {
        SegmentTranslator.Prepared p = SegmentTranslator.prepare("Anh còn phiếu bảo hành không?", "vi", "en", null);
        assertFalse(p.copy);
        assertEquals("vi", p.sourceLanguage);
        assertEquals(1, p.context.length);
        assertArrayEquals(HyMtOnnxModule.primer("vi", "en"), p.context[0]);
        // The primer stays the oldest line; the previous segment follows it.
        SegmentTranslator.Prepared next = SegmentTranslator.prepare("Anh mua ở đâu?", "vi", "en",
                new String[]{"vi", "en", "Anh còn phiếu bảo hành không?", "Do you still have the warranty card?"});
        assertEquals(2, next.context.length);
        assertEquals("Do you still have the warranty card?", next.context[1][1]);
    }

    @Test public void chineseHeBecomesSheAfterAFemaleWordInThisOrThePreviousSegment() {
        assertEquals("我姐姐是医生，她每天都很忙。",
                SegmentTranslator.prepare("我姐姐是医生，他每天都很忙。", "zh", "vi", null).input);
        SegmentTranslator.Prepared p = SegmentTranslator.prepare("他的老师很喜欢他。", "zh", "vi",
                new String[]{"zh", "vi", "我女儿今年上小学。", "Con gái tôi năm nay học tiểu học."});
        assertEquals("她的老师很喜欢她。", p.input);
        assertEquals(1, p.context.length);   // no primer for Chinese, only the previous segment
        // Without a female word the ASR's 他 is kept: the audio cannot tell tā apart.
        SegmentTranslator.Prepared alone = SegmentTranslator.prepare("他来了吗？", "zh", "vi", null);
        assertEquals("他来了吗？", alone.input);
        assertEquals(0, alone.context.length);
    }

    @Test public void rowAlreadyInTheTargetLanguageIsCopied() {
        SegmentTranslator.Prepared p = SegmentTranslator.prepare("我们在哪儿学中文？", "vi", "zh", null);
        assertTrue(p.copy);
        assertEquals("zh", p.sourceLanguage);
    }
}
