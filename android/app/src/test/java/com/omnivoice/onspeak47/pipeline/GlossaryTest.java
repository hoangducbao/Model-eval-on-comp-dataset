package com.omnivoice.onspeak47.pipeline;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class GlossaryTest {

    @After
    public void clear() {
        Glossary.set(Collections.emptyList());
    }

    /** a, b, from, to per pair. */
    private static void use(String... f) {
        List<Glossary.Term> l = new ArrayList<>();
        for (int i = 0; i < f.length; i += 4) l.add(new Glossary.Term(f[i], f[i + 1], f[i + 2], f[i + 3]));
        Glossary.set(l);
    }

    @Test
    public void emptyGlossary_leavesPromptUntouched() {
        String plain = HyMtOnnxModule.buildInstruction("Hello.", "en", "vi");
        use("standup", "họp đầu ngày", "en", "vi");
        assertEquals(plain, HyMtOnnxModule.buildInstruction("Hello.", "en", "vi"));
    }

    @Test
    public void matchingTerm_usesHyMtTerminologyTemplate() {
        use("Standup", "họp đầu ngày", "en", "vi");
        assertEquals("参考下面的翻译：\nStandup 翻译成 họp đầu ngày\n\n"
                        + "将以下文本翻译为越南语，注意只需要输出翻译后的结果，不要额外解释：\n"
                        + "daily standup at nine.",
                HyMtOnnxModule.buildInstruction("daily standup at nine.", "en", "vi"));
        use("站会", "standup", "zh", "en");
        assertEquals("参考下面的翻译：\n站会 翻译成 standup\n\n",
                Glossary.fragment("我们开站会吧", "zh", "en"));   // CJK without spaces
    }

    @Test
    public void term_appliesOnlyToItsLanguagePair() {
        use("standup", "站会", "en", "zh");
        assertEquals("", Glossary.fragment("daily standup", "en", "vi"));
        assertEquals("", Glossary.fragment("我们开站会吧", "zh", "vi"));
        assertEquals("", Glossary.fragment("我们开站会吧", "zh", "en"));   // never reversed
        use("standup", "站会", "", "");   // saved before directions: every pair, term side only
        assertFalse(Glossary.fragment("daily standup", "en", "vi").isEmpty());
        assertEquals("", Glossary.fragment("我们开站会吧", "zh", "en"));
    }

    @Test
    public void capitalizedName_keepsItsCaseInCasedText() {
        use("Thắng", "Thắng", "vi", "en");
        assertFalse(Glossary.fragment("Anh Thắng đến rồi.", "vi", "en").isEmpty());
        assertEquals("", Glossary.fragment("Chúng ta thắng rồi.", "vi", "en"));   // the word, not the name
        assertFalse(Glossary.fragment("anh thắng đến rồi", "vi", "en").isEmpty());   // 2e: no case to read
    }

    @Test
    public void latinTerms_matchWholeWordsOnly() {
        assertFalse(Glossary.occurs("we maintain it", "AI"));
        assertTrue(Glossary.occurs("chip ai mới", "AI"));
        assertTrue(Glossary.occurs("họp đầu ngày lúc 9 giờ", "Họp đầu ngày"));
        assertFalse(Glossary.occurs("standups", "standup"));
    }

    @Test
    public void storage_roundTrips() {
        use("standup", "họp đầu ngày", "en", "vi", "NPU", "神经处理器", "en", "zh");
        String enc = Glossary.encode(Glossary.get());
        assertEquals(enc, Glossary.encode(Glossary.decode(enc)));
        Glossary.Term old = Glossary.decode("standup\thọp đầu ngày\n").get(0);   // pre-direction line
        assertEquals("họp đầu ngày", old.b);
        assertEquals("", old.from);
        assertEquals("a b", Glossary.clean(" a\t b\n"));
    }

    @Test
    public void spokenFormReplacesTargetTermForTtsOnly() {
        Glossary.set(java.util.Arrays.asList(
                new Glossary.Term("Ministry of Home Affairs", "Bộ Nội vụ", "en", "vi", "Bộ Nội Vụ Việt Nam"),
                new Glossary.Term("visa", "thị thực", "en", "vi")));
        assertEquals("Gửi Bộ Nội Vụ Việt Nam và thị thực.", Glossary.spokenForm("Gửi bộ nội vụ và thị thực.", "vi"));
        assertEquals("Bộ Nội vụ", Glossary.spokenForm("Bộ Nội vụ", "en"));
        assertEquals("Bộ Nội vụx", Glossary.spokenForm("Bộ Nội vụx", "vi"));
    }

    @Test
    public void spokenSurvivesRoundTripAndOldLinesDecode() {
        Glossary.Term t = Glossary.decode(Glossary.encode(java.util.Collections.singletonList(
                new Glossary.Term("a", "b", "en", "vi", "c")))).get(0);
        assertEquals("c", t.spoken);
        assertEquals("", Glossary.decode("a	b	en	vi").get(0).spoken);
    }
}
