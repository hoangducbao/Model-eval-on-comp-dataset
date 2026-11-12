package com.omnivoice.onspeak47.pipeline;

import android.content.Context;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * User glossary of {term → preferred translation} pairs, each for one
 * language direction (ConfLive Glossary port). A pair rides along with a
 * translation only when the MT direction is exactly its own (never reversed)
 * and the sentence contains the term, in HY-MT's terminology template, so
 * an unrelated sentence keeps the exact prompt it had before. Pairs saved
 * before directions existed have from = to = "" and apply to every pair,
 * still term side only.
 * Qwen3-ASR also gets every term as context (QwenAsrNpuEngine); the 2e decoders are greedy,
 * so no hotword biasing there.
 */
public final class Glossary {

    private static final String PREFS = "glossary";
    private static final String KEY = "terms";
    private static volatile List<Term> terms = Collections.emptyList();

    /** {@code a} in language {@code from} → {@code b} in {@code to}. */
    public static final class Term {
        public final String a, b, from, to;
        /** How TTS says {@code b} ("" = as written); the display and the MT prompt keep {@code b}. */
        public final String spoken;

        public Term(String a, String b, String from, String to) {
            this(a, b, from, to, "");
        }

        public Term(String a, String b, String from, String to, String spoken) {
            this.a = a;
            this.b = b;
            this.from = from;
            this.to = to;
            this.spoken = spoken;
        }

        boolean sameKey(Term o) {
            return a.equalsIgnoreCase(o.a) && from.equals(o.from) && to.equals(o.to);
        }
    }

    private Glossary() {}

    public static List<Term> get() {
        return terms;
    }

    /** Add (or replace the pair with the same term and direction). */
    public static void put(Context c, Term t) {
        List<Term> l = new ArrayList<>();
        for (Term o : terms) if (!o.sameKey(t)) l.add(o);
        l.add(t);
        save(c, l);
    }

    public static void remove(Context c, Term t) {
        List<Term> l = new ArrayList<>(terms);
        l.remove(t);
        save(c, l);
    }

    public static void load(Context c) {
        set(decode(c.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY, "")));
    }

    static void save(Context c, List<Term> t) {
        set(t);
        c.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(KEY, encode(t)).apply();
    }

    static void set(List<Term> t) {
        terms = Collections.unmodifiableList(new ArrayList<>(t));
    }

    /** Collapse whitespace so a term never carries the tab/newline separators. */
    public static String clean(String s) {
        return s == null ? "" : s.replaceAll("\\s+", " ").trim();
    }

    /** One {@code a\tb\tfrom\tto\tspoken} line per pair (old builds wrote {@code a\tb} or {@code a\tb\tfrom\tto}). */
    static String encode(List<Term> t) {
        StringBuilder sb = new StringBuilder();
        for (Term e : t) {
            sb.append(e.a).append('\t').append(e.b).append('\t')
                    .append(e.from).append('\t').append(e.to).append('\t').append(e.spoken).append('\n');
        }
        return sb.toString();
    }

    static List<Term> decode(String s) {
        List<Term> l = new ArrayList<>();
        for (String line : s.split("\n")) {
            String[] f = line.split("\t", -1);
            if (f.length < 2 || f[0].isEmpty() || f[1].isEmpty()) continue;
            l.add(new Term(f[0], f[1], f.length > 3 ? f[2] : "", f.length > 3 ? f[3] : "", f.length > 4 ? f[4] : ""));
        }
        return l;
    }

    /**
     * Replaces each translation that has a spoken form with that form, for TTS only: whole word,
     * any case, longest first, for terms whose target language is {@code lang} (or any language).
     */
    public static String spokenForm(String text, String lang) {
        if (text == null || terms.isEmpty()) return text;
        List<Term> t = new ArrayList<>();
        for (Term e : terms) if (!e.spoken.isEmpty() && (e.to.isEmpty() || e.to.equals(lang))) t.add(e);
        if (t.isEmpty()) return text;
        t.sort((x, y) -> y.b.length() - x.b.length());
        String out = Normalizer.normalize(text, Normalizer.Form.NFC);
        for (Term e : t) {
            String b = Normalizer.normalize(e.b, Normalizer.Form.NFC);
            String rx = (isCjk(b.charAt(0)) ? "" : "(?<![\\p{L}\\p{N}])") + java.util.regex.Pattern.quote(b)
                    + (isCjk(b.charAt(b.length() - 1)) ? "" : "(?![\\p{L}\\p{N}])");
            out = java.util.regex.Pattern.compile(rx, java.util.regex.Pattern.CASE_INSENSITIVE | java.util.regex.Pattern.UNICODE_CASE)
                    .matcher(out).replaceAll(java.util.regex.Matcher.quoteReplacement(e.spoken));
        }
        return out;
    }

    /**
     * Terminology lines for {@code text} translated {@code src → tgt} (""
     * when no pair applies), in HY-MT 1.5's documented (Chinese-only)
     * template. A pair applies in its own direction only: an EN→VI term
     * never reaches a VI→EN or EN→ZH prompt. Add the reverse pair for both.
     */
    static String fragment(String text, String src, String tgt) {
        List<Term> t = terms;
        if (t.isEmpty() || text == null) return "";
        String nfc = Normalizer.normalize(text, Normalizer.Form.NFC), hay = nfc.toLowerCase(Locale.ROOT);
        // Cased text (Qwen3-ASR, typed) keeps a capitalized term to its case: the name "Thắng",
        // not the word "thắng". 2e rows (vi lowercase, EN CAPS) carry no case, so they fold.
        boolean cased = !nfc.equals(hay) && !nfc.equals(nfc.toUpperCase(Locale.ROOT));
        StringBuilder lines = new StringBuilder();
        for (Term e : t) {
            String a = Normalizer.normalize(e.a, Normalizer.Form.NFC);
            boolean hit = cased && !a.equals(a.toLowerCase(Locale.ROOT)) ? find(nfc, a) : find(hay, norm(a));
            if ((e.from.isEmpty() || (e.from.equals(src) && e.to.equals(tgt))) && hit) {
                line(lines, e.a, e.b);
            }
        }
        if (lines.length() == 0) return "";
        return "参考下面的翻译：\n" + lines + "\n";
    }

    private static void line(StringBuilder sb, String from, String to) {
        sb.append(from).append(" 翻译成 ").append(to).append('\n');
    }

    private static String norm(String s) {
        return Normalizer.normalize(s, Normalizer.Form.NFC).toLowerCase(Locale.ROOT);
    }

    /** Whole-word match; a CJK edge needs no boundary (no spaces in 中文). */
    static boolean occurs(String hay, String term) {
        return find(hay, norm(term));
    }

    private static boolean find(String hay, String t) {
        if (t.isEmpty()) return false;
        for (int i = hay.indexOf(t); i >= 0; i = hay.indexOf(t, i + 1)) {
            if (edge(hay, i - 1, t.charAt(0)) && edge(hay, i + t.length(), t.charAt(t.length() - 1))) {
                return true;
            }
        }
        return false;
    }

    private static boolean edge(String s, int j, char termEdge) {
        return j < 0 || j >= s.length() || isCjk(termEdge) || !Character.isLetterOrDigit(s.charAt(j));
    }

    private static boolean isCjk(char c) {
        return Character.UnicodeScript.of(c) == Character.UnicodeScript.HAN;
    }
}
