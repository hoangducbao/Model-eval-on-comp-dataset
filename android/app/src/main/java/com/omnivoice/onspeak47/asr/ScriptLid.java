package com.omnivoice.onspeak47.asr;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Language of a 2e Zipformer transcript, read off the model's own output format
 * (port of ConfLive {@code app/langid.py}; Python twin
 * {@code backend/streaming_asr/script_lid.py}). The zh-en-vi model writes every
 * word in its language's script: Chinese as characters, English in UPPERCASE,
 * Vietnamese in lowercase — so the language is known per word, not guessed, and
 * costs no model (unlike the VoxLingua ECAPA referee).
 */
public final class ScriptLid {
    private ScriptLid() {
    }

    private static final Pattern CJK = Pattern.compile("[㐀-鿿]");
    private static final Pattern CAPS_WORD = Pattern.compile("\\b[A-Z][A-Z']+\\b");

    private static boolean isCjk(char c) {
        return c >= '㐀' && c <= '鿿';
    }

    /** "zh" / "en" / "vi" for one word, null for digits/punctuation. */
    static String wordLang(String w) {
        boolean alpha = false;
        boolean upper = true;
        boolean ascii = true;
        for (int i = 0; i < w.length(); i++) {
            char c = w.charAt(i);
            if (isCjk(c)) return "zh";
            if (c > 127) ascii = false;
            if (Character.isLetter(c)) {
                alpha = true;
                if (!Character.isUpperCase(c)) upper = false;
            }
        }
        if (!alpha) return null;
        return ascii && upper ? "en" : "vi";
    }

    /**
     * Row language, or null = mixed/unsure (keep the router's label, translate).
     * Majority language unless the row also holds words its reader can't read:
     * Chinese in a vi/en row, Vietnamese in a zh row. Latin-script minorities
     * (English terms in vi/zh speech) stay with the majority.
     */
    public static String detect(String text) {
        if (text == null) return null;
        Map<String, Integer> n = new LinkedHashMap<>();
        int total = 0;
        // One unit per CJK char / vi syllable / en word.
        for (String w : CJK.matcher(text).replaceAll(" $0 ").trim().split("\\s+")) {
            String lang = w.isEmpty() ? null : wordLang(w);
            if (lang == null) continue;
            n.merge(lang, 1, Integer::sum);
            total++;
        }
        if (total == 0) return null;
        String dom = null;
        for (Map.Entry<String, Integer> e : n.entrySet()) {
            if (dom == null || e.getValue() > n.get(dom)) dom = e.getKey();
        }
        if (2 * n.get(dom) <= total) return null;
        for (String o : n.keySet()) {
            if (!o.equals(dom) && unreadable(o, dom)) return null;
        }
        return dom;
    }

    /**
     * {@link #detect} for committing the pipeline's language label. Lowercase
     * is the catch-all (any non-CAPS Latin reads as vi), so a vi verdict also
     * needs one diacritic letter: "o a e o" @0.3 stays unlabelled (field-log
     * VI-bias regression). ponytail: diacritic test, not a lexicon;
     * diacritic-free real Vietnamese waits for the acoustic LID instead.
     */
    public static String label(String text) {
        String lang = detect(text);
        if (!"vi".equals(lang)) return lang;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c > 127 && Character.isLetter(c)) return lang;
        }
        return null;
    }

    /**
     * Most frequent word language other than {@code tgt}, or null when every
     * word is already {@code tgt}. A row labelled with the target language is
     * only a copy when this is null: "MAYBE WE CAN HANG UP LATER bạn nghĩ sao"
     * is en by majority but still needs its vi tail translated.
     */
    public static String foreign(String text, String tgt) {
        if (text == null) return null;
        Map<String, Integer> n = new LinkedHashMap<>();
        for (String w : CJK.matcher(text).replaceAll(" $0 ").trim().split("\\s+")) {
            String lang = w.isEmpty() ? null : wordLang(w);
            if (lang != null && !lang.equals(tgt)) n.merge(lang, 1, Integer::sum);
        }
        String best = null;
        for (Map.Entry<String, Integer> e : n.entrySet()) {
            if (best == null || e.getValue() > n.get(best)) best = e.getKey();
        }
        return best;
    }

    /** (minority, row language) pairs whose reader can't read the minority. */
    private static boolean unreadable(String minority, String row) {
        return "zh".equals(minority) || ("vi".equals(minority) && "zh".equals(row));
    }

    /**
     * Lowercase the ASR's ALL-CAPS English words before MT (2e model card: Hy-MT
     * translates CAPS worse; ConfLive measured). Single letters ("I") stay.
     */
    public static String lowerEnglish(String text) {
        if (text == null) return "";
        Matcher m = CAPS_WORD.matcher(text);
        StringBuffer sb = new StringBuffer();
        while (m.find()) {
            m.appendReplacement(sb, Matcher.quoteReplacement(
                    m.group().toLowerCase(java.util.Locale.ROOT)));
        }
        m.appendTail(sb);
        return sb.toString();
    }
}
