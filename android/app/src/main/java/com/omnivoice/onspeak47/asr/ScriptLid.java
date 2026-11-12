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
    // Word char for the boundaries below; CJK is not one: Qwen glues Latin to Chinese ("寄到ANDERSON").
    private static final String W = "[\\p{L}\\p{N}_&&[^㐀-鿿]]";
    private static final Pattern CAPS_WORD = Pattern.compile("(?<!" + W + ")[A-Z][A-Z']*(?!" + W + ")");
    // Needs the space: the "." of INFO@MANDARINHQ.COM ends no sentence (MT got ".Com", D:\apk34zhmail.txt).
    private static final Pattern SENTENCE_END = Pattern.compile("(^|[.!?]\\s)\\s*$");

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
     * translates CAPS worse; ConfLive measured). The pronoun I and a caps word that
     * opens a sentence are capitalized; a kept capital "A" read as a name ("A rất vui").
     */
    public static String lowerEnglish(String text) {
        if (text == null) return "";
        Matcher m = CAPS_WORD.matcher(text);
        StringBuffer sb = new StringBuffer();
        while (m.find()) {
            String w = m.group().toLowerCase(java.util.Locale.ROOT);
            if (w.equals("i") || w.startsWith("i'")
                    || SENTENCE_END.matcher(text.substring(0, m.start())).find()) {
                w = Character.toUpperCase(w.charAt(0)) + w.substring(1);
            }
            m.appendReplacement(sb, Matcher.quoteReplacement(w));
        }
        m.appendTail(sb);
        return sb.toString();
    }

    // ".点": Qwen writes the dot twice ("RONWENAN.点TECH AT GMAIL.点COM", D:\apk34zhmail.txt).
    private static final String DOT = "(?:\\s+(?:dot|chấm)\\s+|\\s*\\.?\\s*点\\s*)";
    private static final String LABEL = "[A-Za-z0-9_-]+";
    // Local part may be a pinyin address Qwen heard as Chinese words (如安文安点TECH, 服务点科辅AT GMAIL).
    // ponytail: up to 6 Han per label, ended by a stop word; an unlisted word before the name is romanized too.
    private static final String LOCAL = "(?:" + LABEL + "|(?:(?![至到是给向或和及的为在点])[㐀-鿿]){1,6})";
    private static final Pattern SPOKEN_EMAIL = Pattern.compile("(?i)(?<!" + W + ")(?<![.@'-])("
            + LOCAL + "(?:" + DOT + LOCAL + ")*)(?:\\s+at\\s+|(?<=[㐀-鿿])\\s*at\\s+|\\s*艾特\\s*)(" + LABEL + "(?:" + DOT + LABEL
            + ")*" + DOT + "(?:com|net|org|edu|gov|vn|io|co|info|me))(?!" + W + ")");
    private static final Pattern DOT_SEP = Pattern.compile("(?i)" + DOT);
    /** Han to pinyin for addresses. OmniVoiceApp sets it (android.icu, API 29+); without it Han stays. */
    public static java.util.function.UnaryOperator<String> pinyin = s -> s;
    // A second "@" is a spoken dot Qwen misheard: DAVID.MILLER@TECH@GMAIL.COM (D:\apk34enmail.txt).
    private static final Pattern WRITTEN_EMAIL = Pattern.compile(
            "[A-Za-z0-9_.%+-]+(?:@[A-Za-z0-9_-]+(?:\\.[A-Za-z0-9_-]+)*)+");
    // 加/cộng/plus read before a phone number: "加8613912345678" -> "Add 8613…" (D:\apk34zhmail.txt).
    private static final Pattern PHONE_PLUS = Pattern.compile(
            "(?iu)(?:加|(?<![\\p{L}\\p{N}])(?:cộng|plus)\\s+)(?=\\d{9,})");
    // Vietnamese addresses as Qwen really hears them (device 2026-10-30): "@" = a còng/cổng/cộng,
    // glued too ("techacong gmail com", "accomgmail.com", "tacacam gmail.com"); gmail = "gừmêu",
    // "g mươi"; the local part is a name with diacritics. ponytail: up to 4 words per label, ended
    // by VI_STOP or the segment start, so an unlisted word before the name gets glued on; add it to VI_STOP.
    // Lazy, so a row with two addresses ends the first at its own "@".
    private static final String VI_STOP = "(?!(?:qua|là|về|đến|tới|cho|vào|tại|ở|gửi|email|mail|thư|chỉ|hệ"
            + "|và|hoặc|của|tôi|mình|chấm|dot)(?![\\p{L}\\p{N}])(?!\\s+(?:chấm|dot)\\s))"; // "liên hệ chấm khách": a label
    private static final String VI_WORDS = VI_STOP + "[\\p{L}\\p{N}_-]+(?:\\s+" + VI_STOP + "[\\p{L}\\p{N}_-]+){0,3}?";
    private static final String VI_SEP = "(?:\\s+(?:chấm|dot)\\s+|\\s?\\.\\s?)";
    private static final Pattern VI_EMAIL = Pattern.compile("(?i)(?<![\\p{L}\\p{N}_.@'-])("
            + VI_WORDS + "(?:" + VI_SEP + VI_WORDS + ")*?)\\s*(?:a\\s*c+[aoòóỏõọôồốổỗộ][mn](?:g(?!m))?|a\\s*m[oóốô]c)\\s*"
            + "(?:(g[ờừ]?\\s?m(?:ai|â|e|ê|ư)\\p{L}*)(?:" + VI_SEP + "|\\s+)com|(" + LABEL + "(?:" + VI_SEP + LABEL + ")*"
            + VI_SEP + "(?:com|net|org|edu|gov|vn|io|co|info|me)))(?!" + W + ")");
    private static final Pattern VI_SEP_P = Pattern.compile("(?i)" + VI_SEP);
    private static final java.util.List<String> VI_DIGITS = java.util.Arrays.asList(
            "không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín");
    private static final java.util.List<String> EN_DIGITS = java.util.Arrays.asList(
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine");
    private static final String ZH_DIGITS = "零一二三四五六七八九";
    private static final Map<String, Integer> DIGIT_WORD = new java.util.HashMap<>();
    private static final Map<String, Long> VI_NUM = new java.util.HashMap<>();
    private static final Map<Character, Long> ZH_NUM = new java.util.HashMap<>();
    private static final java.util.List<Long> VI_SCALE = java.util.Arrays.asList(1000L, 1000000L, 1000000000L);
    private static final java.util.List<Long> ZH_SCALE = java.util.Arrays.asList(10000L, 100000000L);

    static {
        for (int i = 0; i < 10; i++) {
            DIGIT_WORD.put(VI_DIGITS.get(i), i);
            DIGIT_WORD.put(EN_DIGITS.get(i), i);
            DIGIT_WORD.put(String.valueOf(ZH_DIGITS.charAt(i)), i);
            VI_NUM.put(VI_DIGITS.get(i), (long) i);
            ZH_NUM.put(ZH_DIGITS.charAt(i), (long) i);
        }
        DIGIT_WORD.put("bẩy", 7);
        DIGIT_WORD.put("oh", 0);
        DIGIT_WORD.put("〇", 0);
        DIGIT_WORD.put("幺", 1);
        String[] w = {"bẩy", "lăm", "mốt", "tư", "lẻ", "linh", "mười", "mươi", "trăm", "nghìn", "ngàn", "triệu", "tỷ", "tỉ"};
        long[] v = {7, 5, 1, 4, 0, 0, 10, 10, 100, 1000, 1000, 1000000, 1000000000, 1000000000};
        for (int i = 0; i < w.length; i++) VI_NUM.put(w[i], v[i]);
        ZH_NUM.put('〇', 0L);
        ZH_NUM.put('两', 2L);
        ZH_NUM.put('十', 10L);
        ZH_NUM.put('百', 100L);
        ZH_NUM.put('千', 1000L);
        ZH_NUM.put('万', 10000L);
        ZH_NUM.put('亿', 100000000L);
    }

    private static final String DIGIT = "(?:(?:double|triple)\\s+)?(?:" + String.join("|", VI_DIGITS) + "|"
            + String.join("|", EN_DIGITS) + "|bẩy|oh)";
    // ponytail: 5+ digit words in a row = read digit by digit (phone, code), so "hai ba bốn năm"
    // (2-4 years) stays words; lower to {3,} if spoken 4-digit codes come out wrong.
    private static final Pattern DIGIT_RUN = Pattern.compile(
            "(?i)(?<![\\p{L}\\p{N}_])" + DIGIT + "(?:[\\s,]+" + DIGIT + "){4,}(?![\\p{L}\\p{N}_])");
    private static final Pattern ZH_RUN = Pattern.compile(
            "(?<![点十百千万亿])[零〇一二三四五六七八九幺]{5,}(?![十百千万亿])");   // 幺二三零六 = 12306

    // Numbers are written only in a frame that says what they are (twin: same names). Bare ones
    // stay words: "năm ba", "một vài", 一个, 三心二意, 星期一. Vietnamese: dot thousands, comma decimal.
    private static final String VW = "không|một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín"
            + "|mười|mươi|trăm|nghìn|ngàn|triệu|tỷ|tỉ|lẻ|linh";
    private static final String VN = "(?:" + VW + ")(?:\\s+(?:" + VW + "))*";
    private static final String VI_UNIT = "đồng|đô la|đô(?!\\s*thị)|usd|vnđ|vnd|euro|yên|nhân dân tệ|bảng anh";
    private static final String VI_MONTH = "mười một|mười hai|mười|một|hai|ba|bốn|tư|năm|sáu|bảy|bẩy|tám|chín";
    private static final Pattern VI_AMOUNT = Pattern.compile("(?iu)(?<!" + W + ")(" + VN + ")(?:\\s+phẩy\\s+("
            + VN + "))?\\s+(phần trăm|" + VI_UNIT + ")(?!" + W + ")");
    private static final Pattern VI_DATE = Pattern.compile("(?iu)(?<!" + W + ")ngày\\s+(" + VN + ")\\s+tháng\\s+("
            + VI_MONTH + ")(?!" + W + ")");
    private static final Pattern VI_MONTH_YEAR = Pattern.compile("(?iu)(?<!" + W + ")tháng\\s+(" + VI_MONTH
            + ")(?=\\s+năm\\s+\\d{4})");
    private static final Pattern VI_YEAR = Pattern.compile("(?iu)(?<!" + W + ")năm\\s+(" + VN + ")(?!" + W + ")");
    private static final Pattern VI_TIME = Pattern.compile("(?iu)(?<!" + W + ")(" + VN + ")\\s+giờ\\s+(?:(" + VN
            + ")(\\s+phút)?|rưỡi)(?!" + W + ")");
    private static final String ZD = "零〇一二两三四五六七八九";
    private static final String ZN = "[" + ZD + "十百千万亿]+";
    private static final Pattern ZH_PERCENT = Pattern.compile("百分之(" + ZN + ")(?:点([" + ZD + "]+))?");
    // ponytail: 元 not before 一二三组里 (二元一次, 三元组, 三元里); add a word when another compound shows up.
    private static final Pattern ZH_AMOUNT = Pattern.compile("(" + ZN + ")(?:点([" + ZD + "]+)([万亿]?))?"
            + "(元(?![一二三组里])|块钱|美元|欧元|日元|英镑|港币|韩元)");
    private static final Pattern ZH_YEAR = Pattern.compile("(" + ZN + ")年");
    private static final Pattern ZH_DATE = Pattern.compile("(" + ZN + ")月(" + ZN + ")([日号])");
    private static final Pattern ZH_TIME = Pattern.compile(
            "(上午|下午|早上|晚上|中午|凌晨)?(" + ZN + ")点(?:(" + ZN + ")分|半)");
    private static final Pattern ZH_TEMP = Pattern.compile("摄氏(" + ZN + ")度|(" + ZN + ")摄氏度");

    /** Number from digit (0-9), unit (10, 100…) and scale tokens, -1 if it reads as no number. */
    private static long cardinal(long[] t, java.util.List<Long> scales) {
        long total = 0, sec = 0, num = 0, unit = 0, top = 0;
        boolean have = false, zero = false;
        for (long x : t) {
            if (x == 0) {
                zero = true;
            } else if (x < 10) {
                if (have) return -1;
                num = x;
                have = true;
            } else if (scales.contains(x)) {
                sec += num;
                if (sec == 0 && total == 0) return -1;
                total = total > 0 && x > top ? (total + sec) * x : total + sec * x;
                top = unit = x;
                sec = num = 0;
                have = zero = false;
            } else {
                if (!have && !zero && x != 10) return -1;
                sec += (have || zero ? num : 1) * x;
                unit = x;
                num = 0;
                have = zero = false;
            }
        }
        if (have && unit >= 100 && !zero) num *= unit / 10;   // "ba triệu hai", 一百二
        return total + sec + num;
    }

    private static String digitsOrCardinal(long[] t, java.util.List<Long> scales) {
        StringBuilder sb = new StringBuilder();
        for (long x : t) {
            if (x >= 10) {
                long v = cardinal(t, scales);
                return v < 0 ? null : String.valueOf(v);
            }
            sb.append(x);
        }
        return sb.toString();
    }

    private static String[] split(String s) {
        return s.trim().split("\\s+");
    }

    private static long[] viTokens(String[] w, int from) {
        long[] t = new long[w.length - from];
        for (int i = from; i < w.length; i++) t[i - from] = VI_NUM.get(w[i].toLowerCase(java.util.Locale.ROOT));
        return t;
    }

    /** Index of the number's first word, -1 if none: "năm hai mươi" = 1 (20). */
    private static int viStart(String[] w) {
        for (int i = 0; i < w.length; i++) if (cardinal(viTokens(w, i), VI_SCALE) >= 0) return i;
        return -1;
    }

    private static String withHead(String[] w, int i, String written) {
        return i == 0 ? written : String.join(" ", java.util.Arrays.copyOf(w, i)) + " " + written;
    }

    private static String viAmount(Matcher m) {
        String[] w = split(m.group(1));
        int i = viStart(w);
        if (i < 0) return null;
        long v = cardinal(viTokens(w, i), VI_SCALE);
        String frac = "", scale = "";
        if (m.group(2) != null) {
            String[] f = split(m.group(2));
            int n = f.length;
            if (VI_SCALE.contains(VI_NUM.get(f[n - 1].toLowerCase(java.util.Locale.ROOT)))) scale = " " + f[--n];
            frac = n == 0 ? null : digitsOrCardinal(viTokens(java.util.Arrays.copyOf(f, n), 0), VI_SCALE);
            if (frac == null) return null;
        }
        String num = String.format(java.util.Locale.ROOT, "%,d", v).replace(',', '.')
                + (frac.isEmpty() ? "" : "," + frac) + scale;
        String unit = m.group(3);
        if (unit.equalsIgnoreCase("phần trăm")) return withHead(w, i, num + "%");
        if (m.group(2) == null && v < 100) return null;   // "hai đồng nghiệp", "ba đồng"
        return withHead(w, i, num + " " + unit);
    }

    private static String viDate(Matcher m) {
        String[] w = split(m.group(1));
        long day = viStart(w) == 0 ? cardinal(viTokens(w, 0), VI_SCALE) : -1;
        return day < 1 || day > 31 ? null : split(m.group())[0] + " " + day + " tháng "
                + cardinal(viTokens(split(m.group(2)), 0), VI_SCALE);
    }

    private static String viYear(Matcher m) {
        String y = digitsOrCardinal(viTokens(split(m.group(1)), 0), VI_SCALE);
        if (y == null || y.length() > 4) return null;
        long v = Long.parseLong(y);
        return v < 1000 || v > 2999 ? null : split(m.group())[0] + " " + y;
    }

    private static String viTime(Matcher m) {
        String[] w = split(m.group(1));
        int i = viStart(w);
        long h = i < 0 ? -1 : cardinal(viTokens(w, i), VI_SCALE);
        long min = m.group(2) == null ? 30 : cardinal(viTokens(split(m.group(2)), 0), VI_SCALE);
        // "ba giờ hai người": a small number with no "phút" is not the minutes
        if (h < 0 || h > 24 || min < 0 || min > 59 || (m.group(2) != null && m.group(3) == null && min < 10)) return null;
        return withHead(w, i, h + "h" + (min < 10 ? "0" : "") + min);
    }

    private static long[] zhTokens(String s) {
        long[] t = new long[s.length()];
        for (int i = 0; i < t.length; i++) t[i] = ZH_NUM.get(s.charAt(i));
        return t;
    }

    private static long zh(String s) {
        return cardinal(zhTokens(s), ZH_SCALE);
    }

    private static String zhDigits(String s) {
        return s == null ? "" : digitsOrCardinal(zhTokens(s), ZH_SCALE);
    }

    private static String zhPercent(Matcher m) {
        long v = zh(m.group(1));
        return v < 0 ? null : m.group(2) == null ? v + "%" : v + "." + zhDigits(m.group(2)) + "%";
    }

    private static String zhAmount(Matcher m) {
        String s = m.group(1), frac = zhDigits(m.group(2)), scale = m.group(3) == null ? "" : m.group(3);
        char last = s.charAt(s.length() - 1);
        if ((last == '万' || last == '亿') && frac.isEmpty()) {
            scale = String.valueOf(last);
            s = s.substring(0, s.length() - 1);
        }
        long v = s.isEmpty() ? -1 : zh(s);
        if (v < 0 || (v < 2 && frac.isEmpty() && scale.isEmpty())) return null;   // 一元
        return v + (frac.isEmpty() ? "" : "." + frac) + scale + m.group(4);
    }

    private static String zhYear(Matcher m) {
        String y = digitsOrCardinal(zhTokens(m.group(1)), ZH_SCALE);
        if (y == null || y.length() > 4) return null;
        long v = Long.parseLong(y);
        return v < 1000 || v > 2999 ? null : y + "年";
    }

    private static String zhDate(Matcher m) {
        long mo = zh(m.group(1)), d = zh(m.group(2));
        return mo < 1 || mo > 12 || d < 1 || d > 31 ? null : mo + "月" + d + m.group(3);
    }

    private static String zhTime(Matcher m) {
        String part = m.group(1), minWords = m.group(3);
        long h = zh(m.group(2)), min = minWords == null ? 30 : zh(minWords);
        // 半 needs 下午…; a bare 九点五分 is a score (9.5), 五点零二分 and 十五分 are clock times
        if (h < 0 || h > 24 || min < 0 || min > 59 || (minWords == null && part == null)) return null;
        if (minWords != null && part == null && min < 10 && "零〇".indexOf(minWords.charAt(0)) < 0) return null;
        return (part == null ? "" : part) + h + ":" + (min < 10 ? "0" : "") + min;
    }

    private static String zhTemp(Matcher m) {
        long v = zh(m.group(1) != null ? m.group(1) : m.group(2));
        return v < 0 ? null : v + "°C";
    }

    /** "không chín một hai, ba bốn năm" → "0912 345"; double/triple repeat the next digit. */
    private static String digits(String run) {
        StringBuilder num = new StringBuilder();
        for (String group : run.split("\\s*,[\\s,]*")) {
            if (num.length() > 0) num.append(' ');
            int times = 1;
            for (String word : split(group)) {
                String w = word.toLowerCase(java.util.Locale.ROOT);
                if (w.equals("double")) {
                    times = 2;
                } else if (w.equals("triple")) {
                    times = 3;
                } else {
                    for (; times > 0; times--) num.append(DIGIT_WORD.get(w));
                    times = 1;
                }
            }
        }
        String n = num.toString();
        return n.matches("(?:0\\d{9}){2,}") ? n.replaceAll("(.{10})(?=.)", "$1 ") : n;
    }

    /** Every match of {@code p} replaced by {@code f}; null keeps the match. */
    /** `ba` = dad, not 3: before `mẹ/má`, `của`, or a pronoun that a verb or the end follows. Twin: _VI_DAD. */
    private static final Pattern VI_DAD = Pattern.compile("(?<!\\p{L})(?:([Cc]ó|[Mm]ua|[Nn]hững|[Mm]ấy|[Đđ]ược|[Cc]ả|[Hh]ơn|[Gg]ần"
            + "|[Kk]hoảng|[Mm]ột|[Hh]ai|[Bb]ốn|[Nn]ăm) )?([Bb])a (?=(?:mẹ|má)(?!\\p{L})(?! con(?!\\p{L}))|của(?!\\p{L})|(?:anh|em|chị|tôi|mình)(?:[,.?!]|$| (?:thì|là|nha"
            + "|cũng|vẫn|sẽ|đã|mà|làm|ở|không|đi|năm|nói|bảo|rồi|với|hay|có)(?!\\p{L})))");

    /** A row's first word, plain lowercase letters: not an email, address or `iPhone`. Twin: _VI_OPENER. */
    private static final Pattern VI_OPENER = Pattern.compile("^\\p{Ll}(?=\\p{Ll}*(?:[\\s,?!]|\\.(?!\\p{L})|$))");

    /**
     * Vietnamese MT input only (rows keep the spoken text). Southern `ba` read as a number: `ba anh
     * thì là người thu nhập chính` -> "the three of them", `ba của em` -> "My brother"; skipped after
     * a counting word and in `ba mẹ con`, `ba anh em`, `ba em học sinh` (= three). A row that opens
     * lowercase behind a cut is read as a name: `ai cũng phải ngồi` -> "AI also needs to sit", and
     * capitalized `Ai also has to sit` on the 1.25-bit model, so that opener is said the other way
     * (not `ai cũng được` = anyone will do) (D:\apk32.txt, CPU A/B: 4 wins of 24, the rest same or
     * one-word fragments).
     */
    public static String viMtForm(String text) {
        String s = sub(VI_DAD, text, m -> m.group(1) != null ? null : (m.group(2).equals("B") ? "Bố " : "bố "));
        s = s.replaceFirst("^[Aa]i cũng (?!được)", "Mọi người đều ");
        return sub(VI_OPENER, s, m -> m.group().toUpperCase(java.util.Locale.ROOT));
    }

    private static String sub(Pattern p, String text, java.util.function.Function<Matcher, String> f) {
        Matcher m = p.matcher(text);
        StringBuffer sb = new StringBuffer();
        while (m.find()) {
            String r = f.apply(m);
            m.appendReplacement(sb, Matcher.quoteReplacement(r == null ? m.group() : r));
        }
        m.appendTail(sb);
        return sb.toString();
    }

    /**
     * Spoken emails, framed numbers and digit runs to written form, for the MT and the screen
     * (Qwen3-ASR leaves them spoken in vi, and many in zh; Hy-MT2 CPU 2026-10-29 kept written
     * forms 7/7 but lost every spoken email and read "không chín một…" as "not nine, two…"):
     * "pos a còng gmail chấm com" → "pos@gmail.com", "không chín một hai, ba bốn năm" → "0912 345",
     * "mười hai phẩy năm phần trăm" → "12,5%", 二零二四年三月十五日 → 2024年3月15日. A regex pass over
     * one segment. Back-to-back phone numbers ("0901234567 0987654321", read without a pause) are split.
     */
    public static String writtenForm(String text) {
        if (text == null) return "";
        text = sub(VI_EMAIL, text, m -> {
            String local = VI_SEP_P.matcher(m.group(1)).replaceAll(".").replaceAll("\\s+", "");
            String domain = m.group(2) != null ? "gmail.com" : VI_SEP_P.matcher(m.group(3)).replaceAll(".");
            return ascii(local) + "@" + domain.toLowerCase(java.util.Locale.ROOT);
        });
        text = sub(SPOKEN_EMAIL, text, m -> {
            String address = DOT_SEP.matcher(m.group(1) + "@" + m.group(2)).replaceAll(".");
            if (CJK.matcher(address).find()) address = pinyin.apply(address).replaceAll("\\s+", "");
            return address.toLowerCase(java.util.Locale.ROOT);
        });
        text = sub(WRITTEN_EMAIL, text, m -> {
            String e = m.group().toLowerCase(java.util.Locale.ROOT);
            int at = e.lastIndexOf('@');
            return e.substring(0, at).replace('@', '.') + e.substring(at);
        });
        text = sub(VI_AMOUNT, text, ScriptLid::viAmount);
        text = sub(VI_DATE, text, ScriptLid::viDate);
        text = sub(VI_YEAR, text, ScriptLid::viYear);
        text = sub(VI_MONTH_YEAR, text, m -> split(m.group())[0] + " "
                + cardinal(viTokens(split(m.group(1)), 0), VI_SCALE));
        text = sub(VI_TIME, text, ScriptLid::viTime);
        text = sub(ZH_PERCENT, text, ScriptLid::zhPercent);
        text = sub(ZH_AMOUNT, text, ScriptLid::zhAmount);
        text = sub(ZH_YEAR, text, ScriptLid::zhYear);
        text = sub(ZH_DATE, text, ScriptLid::zhDate);
        text = sub(ZH_TIME, text, ScriptLid::zhTime);
        text = sub(ZH_TEMP, text, ScriptLid::zhTemp);
        text = sub(DIGIT_RUN, text, m -> digits(m.group()));
        text = sub(ZH_RUN, text, m -> digits(m.group().replaceAll("(?<=.)(?=.)", " ")));
        return PHONE_PLUS.matcher(text).replaceAll("+");
    }

    /** A spoken digit, punctuation aside: a forced cut stays out of a digit run. */
    public static boolean digitWord(String w) {
        return DIGIT_WORD.containsKey(w.replaceAll("[.,!?…。！？，]+$", "").toLowerCase(java.util.Locale.ROOT));
    }

    /** A spoken number word (digit, or mươi/trăm/nghìn/...): a forced cut stays out of
     *  "một nghìn chín trăm bốn mươi | sáu" (D:	ts68.txt: "… autumn of 1940." + "Six is called …"). */
    public static boolean numberWord(String w) {
        String k = w.replaceAll("[.,!?…。！？，]+$", "").toLowerCase(java.util.Locale.ROOT);
        return DIGIT_WORD.containsKey(k) || VI_NUM.containsKey(k);
    }

    /** Lowercase without diacritics: an address is ASCII ("nguyễnvănan" → "nguyenvanan"). */
    private static String ascii(String s) {
        return java.text.Normalizer.normalize(s.toLowerCase(java.util.Locale.ROOT), java.text.Normalizer.Form.NFD)
                .replaceAll("\\p{M}+", "").replace('đ', 'd');
    }

    // ASCII only (Qwen glues the address to Chinese: "寄到anderson…"); "gmail .com": the MT may space the dot.
    private static final Pattern EMAIL = Pattern.compile(
            "[A-Za-z0-9_.%+-]+@[A-Za-z0-9_-]+(?:\\s?\\.[A-Za-z0-9_-]+)*");

    /**
     * The source's emails copied over the translation's, in order: Hy-MT2 rewrites
     * addresses ("anderson.one@gmail.com" → "Anderson_one@gmail.com", device 2026-10-30).
     * ponytail: n-th to n-th by position; an address the MT dropped stays dropped.
     */
    public static String keepEmails(String src, String translation) {
        if (src == null || translation == null || src.indexOf('@') < 0) return translation;
        java.util.List<String> emails = new java.util.ArrayList<>();
        Matcher m = EMAIL.matcher(src);
        while (m.find()) emails.add(m.group());
        m = EMAIL.matcher(translation);
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < emails.size() && m.find(); i++) {
            m.appendReplacement(sb, Matcher.quoteReplacement(emails.get(i)));
        }
        m.appendTail(sb);
        return sb.toString();
    }
}
