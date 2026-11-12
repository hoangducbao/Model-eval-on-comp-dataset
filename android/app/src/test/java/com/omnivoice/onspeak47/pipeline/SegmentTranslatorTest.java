package com.omnivoice.onspeak47.pipeline;

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
}
