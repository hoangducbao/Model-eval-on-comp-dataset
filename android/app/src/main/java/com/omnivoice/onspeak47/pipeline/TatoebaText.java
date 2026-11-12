/*
 * OmniVoice — Tatoeba assist text primitives (pure-Java, no android.* imports).
 *
 * Byte-for-byte mirror of backend/tatoeba_assist.py normalization, tokenizing,
 * scoring and prompt-fragment rendering. The SQLite access lives in
 * {@link TatoebaAssist}; everything here runs in local JVM unit tests so the
 * two platforms can never drift silently.
 */
package com.omnivoice.onspeak47.pipeline;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class TatoebaText {

    private TatoebaText() {}

    public static final int DICT_MAX_WORDS = 1;
    public static final double DICT_MIN_DICE = 0.8;
    public static final int DICT_MIN_COOC = 10;
    public static final int TM_TOP_K = 3;
    public static final double TM_MIN_SIM = 0.35;
    public static final double TM_MIN_SIM_CJK = 0.25;
    public static final int TM_MAX_LEN = 100;
    public static final int TM_CANDIDATES = 60;

    // Must stay identical to backend/tatoeba_assist.py _WORD_RE (order of
    // alternatives matters for token boundaries).
    private static final String VIET =
            "a-z0-9âăđêôơưáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ";
    private static final String CJK =
            "\\u4e00-\\u9fff\\uf900-\\ufaff\\u3400-\\u4dbf"
            + "\\x{20000}-\\x{2a6df}\\x{2a600}-\\x{2b73f}";
    private static final Pattern WORD_RE = Pattern.compile(
            "[" + VIET + "]+|[" + CJK + "]",
            Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE);
    private static final Pattern CJK_CHAR_RE = Pattern.compile("[" + CJK + "]");
    private static final String STRIP_CHARS = " \t\n\r.,!?;:\"'“”‘’()[]";

    private static final Set<String> HANT_MARKERS = markers(
            "語說會們個車門國學漢龍龜鳥魚黃點廣廠親聲時實愛買賣頭麵這邊遠運還過");
    private static final Set<String> HANS_MARKERS = markers(
            "语说会们个车门国学汉龙龟鸟鱼黄点广厂亲声时实爱买卖头这边远运还过");

    private static Set<String> markers(String s) {
        Set<String> out = new HashSet<>();
        for (int i = 0; i < s.length();) {
            int cp = s.codePointAt(i);
            out.add(new String(Character.toChars(cp)));
            i += Character.charCount(cp);
        }
        return out;
    }

    /** Map caller codes (vi/en/zh_hans/zh_hant/eng/vie/cmn/…) to vi/en/zh. */
    public static String baseLang(String code) {
        String c = code == null ? "" : code.toLowerCase(Locale.ROOT);
        if (c.startsWith("zh") || c.equals("cmn") || c.equals("yue")) return "zh";
        if (c.startsWith("en") || c.equals("eng")) return "en";
        if (c.startsWith("vi") || c.equals("vie")) return "vi";
        return c;
    }

    public static String direction(String src, String tgt) {
        return baseLang(src) + "->" + baseLang(tgt);
    }

    /** Lowercase latin, collapse whitespace, strip surrounding punctuation. */
    public static String normalize(String text) {
        if (text == null) return "";
        String t = text.toLowerCase(Locale.ROOT).trim().replaceAll("\\s+", " ");
        int a = 0, b = t.length();
        while (a < b && STRIP_CHARS.indexOf(t.charAt(a)) >= 0) a++;
        while (b > a && STRIP_CHARS.indexOf(t.charAt(b - 1)) >= 0) b--;
        return t.substring(a, b);
    }

    /** Latin words + single CJK chars. */
    public static List<String> tokWords(String text) {
        List<String> out = new ArrayList<>();
        Matcher m = WORD_RE.matcher(normalize(text));
        while (m.find()) out.add(m.group());
        return out;
    }

    public static Set<String> charBigrams(String text) {
        String t = normalize(text).replaceAll("\\s+", "");
        Set<String> out = new HashSet<>();
        if (t.length() < 2) {
            if (!t.isEmpty()) out.add(t);
            return out;
        }
        for (int i = 0; i + 1 < t.length(); i++) out.add(t.substring(i, i + 2));
        return out;
    }

    public static double jaccard(Set<String> a, Set<String> b) {
        if (a.isEmpty() || b.isEmpty()) return 0.0;
        int inter = 0;
        for (String s : a) if (b.contains(s)) inter++;
        return (double) inter / (a.size() + b.size() - inter);
    }

    public static double dice(int cooc, int fSrc, int fTgt) {
        if (fSrc <= 0 || fTgt <= 0) return 0.0;
        return 2.0 * cooc / (fSrc + fTgt);
    }

    /** Best-effort hans/hant/unknown tag for Chinese text. */
    public static String scriptTag(String text) {
        if (text == null) return "unknown";
        boolean hant = false, hans = false;
        for (int i = 0; i < text.length();) {
            int cp = text.codePointAt(i);
            String ch = new String(Character.toChars(cp));
            if (HANT_MARKERS.contains(ch)) hant = true;
            if (HANS_MARKERS.contains(ch)) hans = true;
            i += Character.charCount(cp);
        }
        // Traditional wins ties (mirrors Python: hant checked first).
        if (hant) return "hant";
        if (hans) return "hans";
        return "unknown";
    }

    /** CJK character ratio of the normalized text (0..1). */
    public static double cjkRatio(String text) {
        String t = normalize(text).replaceAll("\\s+", "");
        if (t.isEmpty()) return 0.0;
        int n = 0;
        for (int i = 0; i < t.length();) {
            int cp = t.codePointAt(i);
            if (CJK_CHAR_RE.matcher(new String(Character.toChars(cp))).matches()) n++;
            i += Character.charCount(cp);
        }
        return (double) n / t.codePointCount(0, t.length());
    }

    /**
     * Few-shot fragment shared by every buildInstruction variant, byte-
     * identical to backend/tatoeba_assist usage in translation_hymt.py:
     * {@code " (e.g. \"s\"→\"t\"; …)"} / {@code " (例如\"s\"→\"t\"; …)"},
     * empty string when there are no examples.
     */
    public static String examplesFragment(List<String[]> pairs, boolean chinese) {
        if (pairs == null || pairs.isEmpty()) return "";
        StringBuilder sb = new StringBuilder(chinese ? " (例如" : " (e.g. ");
        boolean first = true;
        int n = 0;
        for (String[] p : pairs) {
            if (n >= TM_TOP_K) break;
            if (!first) sb.append("; ");
            sb.append('"').append(p[0]).append("\"→\"").append(p[1]).append('"');
            first = false;
            n++;
        }
        sb.append(')');
        return sb.toString();
    }

    /** Score helper for rerank parity tests (token-Jaccard path). */
    public static double tokenScore(String query, String candidate) {
        return jaccard(new HashSet<>(tokWords(query)), new HashSet<>(tokWords(candidate)));
    }

    /** Score helper for rerank parity tests (bigram path). */
    public static double bigramScore(String query, String candidate) {
        return jaccard(charBigrams(query), charBigrams(candidate));
    }
}
