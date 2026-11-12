package com.omnivoice.onspeak47.pipeline;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class MtContextTest {

    @Test
    public void withContext_prependsPreviousSentenceToTheText() {
        String en = HyMtOnnxModule.buildInstruction("Oh my god, wow, so graphic.", "en", "vi");
        String ctx = HyMtOnnxModule.withContext(en, "The doctor will cut it out.");
        assertEquals(en.replace(":\n\nOh", ":\n\nThe doctor will cut it out.\nOh"), ctx);

        String zh = HyMtOnnxModule.buildInstruction("别迟到。", "zh", "vi");
        assertTrue(HyMtOnnxModule.withContext(zh, "我们九点开会。").endsWith("\n我们九点开会。\n别迟到。"));
    }

    @Test
    public void contextFits_withinExtraRunsAndSmallGraphsOnly() {
        assertTrue(HyMtNpuDecoder.contextFits(33, 64, 128, 0));    // same 64-token chunk
        assertFalse(HyMtNpuDecoder.contextFits(33, 85, 128, 0));   // one more prefill run
        assertTrue(HyMtNpuDecoder.contextFits(33, 85, 128, 1));
        assertFalse(HyMtNpuDecoder.contextFits(33, 129, 128, 1));  // two more runs
        assertFalse(HyMtNpuDecoder.contextFits(100, 128, 128, 1)); // leaves the 256-slot decode graph
    }
}
