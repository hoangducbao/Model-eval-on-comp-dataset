package com.larplarpsahur.pipeline;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Spells digits out so hypotheses can be scored against a reference table that was delivered
 * normalized (Round 3 E_to_V: "at five thirty", "năm một nghìn chín trăm chín mươi").
 * Plain cardinals only: the references also use readings no rule can recover from digits
 * ("sixteen forty five", "eight hundred" for 8:00), so numeric rows stay approximate.
 * Input is lowercase NFC text; output keeps the punctuation for the caller to strip.
 */
final class Numbers {
    private Numbers() {}

    private static final Pattern CLOCK = Pattern.compile("(\\d{1,2})[:h](\\d{2})(?!\\d)");
    private static final Pattern NUMBER = Pattern.compile("\\d+(?:[.,]\\d+)*");
    private static final String[] EN_ONES = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
    private static final String[] EN_TENS = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
    private static final String[] EN_SCALES = {"", " thousand", " million", " billion", " trillion"};
    private static final String[] VI_ONES = {"không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"};
    private static final String[] VI_SCALES = {"", " nghìn", " triệu", " tỷ", " nghìn tỷ"};

    static String spell(String text, String lang) {
        boolean vi = "vi".equals(lang);
        // 5:30 / 5h30 are two numbers, not 530.
        String s = CLOCK.matcher(text).replaceAll("$1 $2");
        Matcher m = NUMBER.matcher(s);
        StringBuffer out = new StringBuffer();
        while (m.find()) m.appendReplacement(out, Matcher.quoteReplacement(" " + number(m.group(), vi) + " "));
        m.appendTail(out);
        return out.toString();
    }

    /** One digit token with optional separators: English 1,234.5 and Vietnamese 1.234,5. */
    private static String number(String token, boolean vi) {
        char group = vi ? '.' : ',', decimal = vi ? ',' : '.';
        String whole = token, fraction = "";
        int point = token.lastIndexOf(decimal);
        if (point >= 0 && token.indexOf(decimal) == point) {
            whole = token.substring(0, point);
            fraction = token.substring(point + 1);
        }
        StringBuilder words = new StringBuilder();
        if (whole.matches("\\d{1,3}(?:\\" + group + "\\d{3})+")) {
            // Proper thousands groups are one number; any other separator run is read part by part.
            words.append(integer(whole.replace(String.valueOf(group), ""), vi));
        } else {
            for (String part : whole.split("[.,]")) {
                if (words.length() > 0) words.append(' ');
                words.append(integer(part, vi));
            }
        }
        if (!fraction.isEmpty()) words.append(vi ? " phẩy " : " point ").append(digits(fraction, vi));
        return words.toString();
    }

    private static String integer(String digits, boolean vi) {
        // Codes and phone numbers (leading zero, too long) are read digit by digit.
        if (digits.length() > 15 || (digits.length() > 1 && digits.charAt(0) == '0')) return digits(digits, vi);
        long value = Long.parseLong(digits);
        if (value == 0) return vi ? VI_ONES[0] : EN_ONES[0];
        StringBuilder out = new StringBuilder();
        boolean higher = false;
        for (int scale = 4; scale >= 0; scale--) {
            int group = (int) (value / (long) Math.pow(1000, scale) % 1000);
            if (group == 0) continue;
            if (out.length() > 0) out.append(' ');
            // "one thousand and five" / "một nghìn không trăm linh năm".
            out.append(vi ? viGroup(group, higher) : enGroup(group, higher && scale == 0));
            out.append(vi ? VI_SCALES[scale] : EN_SCALES[scale]);
            higher = true;
        }
        return out.toString();
    }

    private static String digits(String digits, boolean vi) {
        StringBuilder out = new StringBuilder();
        for (char c : digits.toCharArray()) {
            if (out.length() > 0) out.append(' ');
            out.append(vi ? VI_ONES[c - '0'] : EN_ONES[c - '0']);
        }
        return out.toString();
    }

    private static String enGroup(int n, boolean and) {
        StringBuilder out = new StringBuilder();
        int rest = n % 100;
        if (n >= 100) out.append(EN_ONES[n / 100]).append(" hundred");
        if (rest == 0) return out.toString();
        if (n >= 100 || and) out.append(out.length() > 0 ? " and " : "and ");
        if (rest < 20) return out.append(EN_ONES[rest]).toString();
        out.append(EN_TENS[rest / 10]);
        return rest % 10 == 0 ? out.toString() : out.append(' ').append(EN_ONES[rest % 10]).toString();
    }

    private static String viGroup(int n, boolean padded) {
        StringBuilder out = new StringBuilder();
        int hundreds = n / 100, tens = n / 10 % 10, ones = n % 10;
        if (hundreds > 0 || padded) out.append(VI_ONES[hundreds]).append(" trăm");
        if (tens == 0 && ones == 0) return out.toString();
        if (out.length() > 0) out.append(' ');
        if (tens == 0) return out.append(hundreds > 0 || padded ? "linh " : "").append(VI_ONES[ones]).toString();
        out.append(tens == 1 ? "mười" : VI_ONES[tens] + " mươi");
        if (ones == 0) return out.toString();
        return out.append(' ').append(ones == 1 && tens > 1 ? "mốt" : ones == 5 ? "lăm" : VI_ONES[ones]).toString();
    }
}
