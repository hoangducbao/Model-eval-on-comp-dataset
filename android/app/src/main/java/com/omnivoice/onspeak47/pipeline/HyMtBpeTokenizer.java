/*
 * OmniVoice — Hy-MT BPE Tokenizer (pure Java, no native dependencies)
 *
 * Faithful port of the HuggingFace `tokenizers` BPE pipeline described by
 * HY-MT/tokenizer.json (the enhanced HY-MT INT8
 * model from niedev/OnnxModelsEnhancer):
 *
 *   normalizer      : none (empty sequence)
 *   pre_tokenizer   : Split digits (\p{N}{1,3}) → Split CJK → Split GPT-2
 *                     regex → ByteLevel (no prefix space, no regex split)
 *   model           : BPE, vocab 120000 + byte_fallback
 *   post_processor  : ByteLevel (no-op for single-sequence id output)
 *   added_tokens    : 818 entries incl. <｜hy_User｜>/120006,
 *                     <｜hy_Assistant｜>/120007, BOS/120000, EOS/120020
 *
 * Special tokens are matched longest-first and are never split or merged,
 * exactly like the Python AutoTokenizer path in
 * backend/translation_hymt_onnx.py. The encoder does NOT add BOS — the
 * caller (HyMtOnnxModule) prepends it by hand because the staged
 * tokenizer.json carries no chat_template. The decoder skips special ids,
 * mirroring skip_special_tokens=True.
 *
 * Self-contained on purpose: tokenizer.json is parsed by the MiniJson
 * reader below (no org.json — the framework stub cannot run in JVM unit
 * tests) and no android.* APIs are used, so this class is fully testable
 * on the JVM.
 *
 * Golden vectors (backend/translation_hymt_onnx.py, same tokenizer.json):
 *   "Hello world." → [16883, 2385, 13]
 *   "<｜hy_User｜>Hello world.<｜hy_Assistant｜>" → [120006, 16883, 2385, 13, 120007]
 */

package com.omnivoice.onspeak47.pipeline;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class HyMtBpeTokenizer {

    /** BOS <｜hy_begin▁of▁sentence｜> — prepended by hand (no chat template). */
    public static final int BOS_ID = 120000;
    /** EOS <｜hy_place▁holder▁no▁2｜> — stops greedy generation. */
    public static final int EOS_ID = 120020;
    public static final String USER_MARK = "<｜hy_User｜>";
    public static final String ASSISTANT_MARK = "<｜hy_Assistant｜>";

    /** BPE pair separator (never occurs in byte-mapped text). */
    private static final String SEP = " ";

    // Pre-tokenizer split patterns, in pipeline order (tokenizer.json).
    // NOTE: no UNICODE_CHARACTER_CLASS flag — Android's java.util.regex
    // rejects it (IllegalArgumentException → ExceptionInInitializerError,
    // field log 2026-10-07). The flag only affects \s/\S (ASCII vs Unicode
    // whitespace); \p{N}/\p{L}/\p{M}/\p{P}/\p{S} and the explicit CJK ranges
    // are Unicode-aware with or without it. ASR transcripts use ASCII
    // whitespace, so these tokenize identically to the HF `tokenizers`
    // Regex engine on real inputs.
    private static final Pattern SPLIT_DIGITS = Pattern.compile("\\p{N}{1,3}");
    private static final Pattern SPLIT_CJK = Pattern.compile("[一-龥぀-ゟ゠-ヿ]+");
    // GPT-2 style splitter from tokenizer.json (Isolated behavior).
    private static final Pattern SPLIT_GPT2 = Pattern.compile(
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+"
                    + "|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+"
                    + "| ?[\\p{P}\\p{S}]+[\r\n]*"
                    + "|\\s*[\r\n]+"
                    + "|\\s+(?!\\S)"
                    + "|\\s+");

    // GPT-2 byte ↔ unicode maps (tokenizers ByteLevel, use_regex=false).
    private static final char[] BYTE_TO_UNICODE = buildByteToUnicode();
    private static final Map<Character, Byte> UNICODE_TO_BYTE = buildUnicodeToByte();

    private static char[] buildByteToUnicode() {
        List<Integer> bs = new ArrayList<>();
        for (int b = '!'; b <= '~'; b++) bs.add(b);
        for (int b = '¡'; b <= '¬'; b++) bs.add(b);
        for (int b = '®'; b <= 'ÿ'; b++) bs.add(b);
        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n++;
            }
        }
        // Reorder: bs[i] ↔ cs[i] — build inverse index byte→char.
        char[] map = new char[256];
        for (int i = 0; i < bs.size(); i++) {
            map[bs.get(i)] = (char) (int) cs.get(i);
        }
        return map;
    }

    private static Map<Character, Byte> buildUnicodeToByte() {
        Map<Character, Byte> m = new HashMap<>(512);
        for (int b = 0; b < 256; b++) {
            m.put(BYTE_TO_UNICODE[b], (byte) b);
        }
        return m;
    }

    private final Map<String, Integer> vocab = new HashMap<>(130000);
    private final List<String> idToToken = new ArrayList<>();
    private final Map<String, Integer> mergeRank = new HashMap<>(130000);
    private final Map<String, Integer> specialIds = new HashMap<>();
    private final List<String> specialByLength = new ArrayList<>();
    private final Set<Integer> specialIdSet = new HashSet<>();

    private String initError;

    /** Load from a tokenizer.json file (assets copy or external files). */
    public HyMtBpeTokenizer(File tokenizerJson) {
        try {
            byte[] data = readAll(tokenizerJson);
            Map<String, Object> root = MiniJson.parseObject(
                    new String(data, StandardCharsets.UTF_8));

            Map<String, Object> model = MiniJson.obj(root.get("model"));
            Map<String, Object> vocabJson = MiniJson.obj(model.get("vocab"));
            int maxId = -1;
            for (Map.Entry<String, Object> e : vocabJson.entrySet()) {
                String tok = e.getKey();
                int id = ((Number) e.getValue()).intValue();
                vocab.put(tok, id);
                if (id > maxId) maxId = id;
            }
            List<Object> merges = MiniJson.arr(model.get("merges"));
            for (int i = 0; i < merges.size(); i++) {
                List<Object> pair = MiniJson.arr(merges.get(i));
                mergeRank.put((String) pair.get(0) + SEP + (String) pair.get(1), i);
            }
            List<Object> added = MiniJson.arr(root.get("added_tokens"));
            for (int i = 0; i < added.size(); i++) {
                Map<String, Object> t = MiniJson.obj(added.get(i));
                String content = (String) t.get("content");
                int id = ((Number) t.get("id")).intValue();
                specialIds.put(content, id);
                specialIdSet.add(id);
                if (id > maxId) maxId = id;
            }
            for (int i = 0; i <= maxId; i++) idToToken.add(null);
            for (Map.Entry<String, Integer> e : vocab.entrySet()) {
                idToToken.set(e.getValue(), e.getKey());
            }
            for (int i = 0; i < added.size(); i++) {
                Map<String, Object> t = MiniJson.obj(added.get(i));
                idToToken.set(((Number) t.get("id")).intValue(), (String) t.get("content"));
            }
            specialByLength.addAll(specialIds.keySet());
            Collections.sort(specialByLength,
                    (a, b) -> Integer.compare(b.length(), a.length()));
        } catch (Exception e) {
            initError = "Failed to load tokenizer.json (" + e.getClass().getSimpleName()
                    + "): " + e.getMessage();
        }
    }

    /** Returns the init error, or null when ready. */
    public String getInitError() {
        return initError;
    }

    public boolean isReady() {
        return initError == null;
    }

    // ------------------------------------------------------------------
    // Encode
    // ------------------------------------------------------------------

    /**
     * Encode text to token ids. Special markers (e.g. hy_User) become single
     * ids; BOS is NOT added — prepend {@link #BOS_ID} by hand.
     */
    public int[] encode(String text) {
        List<Integer> out = new ArrayList<>();
        if (text == null || text.isEmpty()) return new int[0];
        int pos = 0;
        StringBuilder span = new StringBuilder();
        while (pos < text.length()) {
            String hit = matchSpecial(text, pos);
            if (hit != null) {
                flushSpan(span, out);
                out.add(specialIds.get(hit));
                pos += hit.length();
            } else {
                span.append(text.charAt(pos));
                pos++;
            }
        }
        flushSpan(span, out);
        int[] ids = new int[out.size()];
        for (int i = 0; i < out.size(); i++) ids[i] = out.get(i);
        return ids;
    }

    private String matchSpecial(String text, int pos) {
        for (String s : specialByLength) {
            if (s.isEmpty()) continue;
            if (text.startsWith(s, pos)) return s;
        }
        return null;
    }

    private void flushSpan(StringBuilder span, List<Integer> out) {
        if (span.length() == 0) return;
        String s = span.toString();
        span.setLength(0);
        for (String piece : preTokenize(s)) {
            String mapped = byteLevelEncode(piece);
            for (int id : bpe(mapped)) out.add(id);
        }
    }

    /** Apply the Split×3 pre-tokenizer sequence to one non-special span. */
    private List<String> preTokenize(String span) {
        List<String> cur = new ArrayList<>();
        cur.add(span);
        cur = splitAll(cur, SPLIT_DIGITS);
        cur = splitAll(cur, SPLIT_CJK);
        cur = splitAll(cur, SPLIT_GPT2);
        return cur;
    }

    /** Isolated split: matches become their own pieces, gaps are kept. */
    private List<String> splitAll(List<String> pieces, Pattern pattern) {
        List<String> out = new ArrayList<>();
        for (String piece : pieces) {
            Matcher m = pattern.matcher(piece);
            int last = 0;
            boolean found = false;
            while (m.find()) {
                found = true;
                if (m.start() > last) out.add(piece.substring(last, m.start()));
                out.add(m.group());
                last = m.end();
            }
            if (!found) {
                out.add(piece);
            } else if (last < piece.length()) {
                out.add(piece.substring(last));
            }
        }
        return out;
    }

    /** Map UTF-8 bytes through GPT-2 bytes_to_unicode (no prefix space). */
    private String byteLevelEncode(String piece) {
        byte[] bytes = piece.getBytes(StandardCharsets.UTF_8);
        StringBuilder sb = new StringBuilder(bytes.length);
        for (byte b : bytes) sb.append(BYTE_TO_UNICODE[b & 0xFF]);
        return sb.toString();
    }

    /**
     * Rank-based BPE: each iteration merges every non-overlapping occurrence
     * of the lowest-rank adjacent pair (GPT-2 / HF tokenizers behavior).
     */
    private List<Integer> bpe(String token) {
        List<String> parts = new ArrayList<>(token.length());
        for (int i = 0; i < token.length(); i++) {
            parts.add(String.valueOf(token.charAt(i)));
        }
        if (parts.isEmpty()) return Collections.emptyList();
        while (parts.size() > 1) {
            int bestRank = Integer.MAX_VALUE;
            for (int i = 0; i < parts.size() - 1; i++) {
                Integer r = mergeRank.get(parts.get(i) + SEP + parts.get(i + 1));
                if (r != null && r < bestRank) bestRank = r;
            }
            if (bestRank == Integer.MAX_VALUE) break;
            List<String> merged = new ArrayList<>(parts.size());
            int i = 0;
            while (i < parts.size()) {
                if (i < parts.size() - 1) {
                    Integer r = mergeRank.get(parts.get(i) + SEP + parts.get(i + 1));
                    if (r != null && r == bestRank) {
                        merged.add(parts.get(i) + parts.get(i + 1));
                        i += 2;
                        continue;
                    }
                }
                merged.add(parts.get(i));
                i++;
            }
            parts = merged;
        }
        List<Integer> ids = new ArrayList<>(parts.size());
        for (String p : parts) {
            Integer id = vocab.get(p);
            if (id == null) {
                // byte_fallback: re-emit unknown chars as raw UTF-8 bytes.
                for (byte b : p.getBytes(StandardCharsets.UTF_8)) {
                    String bt = String.valueOf(BYTE_TO_UNICODE[b & 0xFF]);
                    Integer bid = vocab.get(bt);
                    if (bid != null) ids.add(bid);
                }
                continue;
            }
            ids.add(id);
        }
        return ids;
    }

    // ------------------------------------------------------------------
    // Decode
    // ------------------------------------------------------------------

    /**
     * Decode ids to text, skipping special ids
     * (mirrors skip_special_tokens=True).
     */
    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            if (specialIdSet.contains(id)) continue;
            String tok = (id >= 0 && id < idToToken.size()) ? idToToken.get(id) : null;
            if (tok != null) sb.append(tok);
        }
        return byteLevelDecode(sb.toString());
    }

    private String byteLevelDecode(String mapped) {
        byte[] bytes = new byte[mapped.length()];
        int n = 0;
        for (int i = 0; i < mapped.length(); i++) {
            Byte b = UNICODE_TO_BYTE.get(mapped.charAt(i));
            if (b != null) bytes[n++] = b;
        }
        return new String(bytes, 0, n, StandardCharsets.UTF_8);
    }

    private static byte[] readAll(File file) throws IOException {
        try (InputStream is = new FileInputStream(file)) {
            long len = file.length();
            if (len <= 0 || len > 64 * 1024 * 1024) {
                throw new IOException("Unexpected tokenizer.json size: " + len);
            }
            byte[] data = new byte[(int) len];
            int off = 0;
            while (off < data.length) {
                int r = is.read(data, off, data.length - off);
                if (r < 0) break;
                off += r;
            }
            if (off != data.length) throw new IOException("Short read");
            return data;
        }
    }

    // ------------------------------------------------------------------
    // MiniJson — just enough JSON for tokenizer.json (objects, arrays,
    // strings with escapes, integers). No org.json so JVM unit tests run.
    // ------------------------------------------------------------------

    static final class MiniJson {
        private final String s;
        private int p;

        MiniJson(String s) {
            this.s = s;
        }

        static Map<String, Object> parseObject(String text) {
            MiniJson j = new MiniJson(text);
            Object v = j.parseValue();
            return obj(v);
        }

        static Map<String, Object> obj(Object v) {
            @SuppressWarnings("unchecked")
            Map<String, Object> m = (Map<String, Object>) v;
            return m;
        }

        static List<Object> arr(Object v) {
            @SuppressWarnings("unchecked")
            List<Object> l = (List<Object>) v;
            return l;
        }

        Object parseValue() {
            skipWs();
            if (p >= s.length()) throw new IllegalStateException("Unexpected end of JSON");
            char c = s.charAt(p);
            if (c == '{') return parseObj();
            if (c == '[') return parseArr();
            if (c == '"') return parseStr();
            if (c == 't') {
                expect("true");
                return Boolean.TRUE;
            }
            if (c == 'f') {
                expect("false");
                return Boolean.FALSE;
            }
            if (c == 'n') {
                expect("null");
                return null;
            }
            return parseNum();
        }

        private Map<String, Object> parseObj() {
            Map<String, Object> m = new HashMap<>();
            p++; // {
            skipWs();
            if (peek() == '}') {
                p++;
                return m;
            }
            while (true) {
                skipWs();
                String key = parseStr();
                skipWs();
                if (next() != ':') throw new IllegalStateException("Expected ':' in object");
                m.put(key, parseValue());
                skipWs();
                char d = next();
                if (d == '}') break;
                if (d != ',') throw new IllegalStateException("Expected ',' in object");
            }
            return m;
        }

        private List<Object> parseArr() {
            List<Object> l = new ArrayList<>();
            p++; // [
            skipWs();
            if (peek() == ']') {
                p++;
                return l;
            }
            while (true) {
                l.add(parseValue());
                skipWs();
                char d = next();
                if (d == ']') break;
                if (d != ',') throw new IllegalStateException("Expected ',' in array");
            }
            return l;
        }

        private String parseStr() {
            if (next() != '"') throw new IllegalStateException("Expected string");
            StringBuilder sb = new StringBuilder();
            while (true) {
                if (p >= s.length()) throw new IllegalStateException("Unterminated string");
                char c = s.charAt(p++);
                if (c == '"') break;
                if (c == '\\') {
                    if (p >= s.length()) throw new IllegalStateException("Bad escape");
                    char e = s.charAt(p++);
                    switch (e) {
                        case '"': sb.append('"'); break;
                        case '\\': sb.append('\\'); break;
                        case '/': sb.append('/'); break;
                        case 'b': sb.append('\b'); break;
                        case 'f': sb.append('\f'); break;
                        case 'n': sb.append('\n'); break;
                        case 'r': sb.append('\r'); break;
                        case 't': sb.append('\t'); break;
                        case 'u': {
                            if (p + 4 > s.length()) {
                                throw new IllegalStateException("Bad unicode escape");
                            }
                            int hi = Integer.parseInt(s.substring(p, p + 4), 16);
                            p += 4;
                            if (Character.isHighSurrogate((char) hi)
                                    && p + 6 <= s.length()
                                    && s.charAt(p) == '\\' && s.charAt(p + 1) == 'u') {
                                int lo = Integer.parseInt(s.substring(p + 2, p + 6), 16);
                                if (Character.isLowSurrogate((char) lo)) {
                                    sb.append(Character.toChars(
                                            Character.toCodePoint((char) hi, (char) lo)));
                                    p += 6;
                                    break;
                                }
                            }
                            sb.append((char) hi);
                            break;
                        }
                        default: throw new IllegalStateException("Bad escape: \\" + e);
                    }
                } else {
                    sb.append(c);
                }
            }
            return sb.toString();
        }

        private Number parseNum() {
            int start = p;
            while (p < s.length() && "-+0123456789.eE".indexOf(s.charAt(p)) >= 0) p++;
            String num = s.substring(start, p);
            try {
                return Long.parseLong(num);
            } catch (NumberFormatException e) {
                return Double.parseDouble(num);
            }
        }

        private void skipWs() {
            while (p < s.length()) {
                char c = s.charAt(p);
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') p++;
                else break;
            }
        }

        private char peek() {
            if (p >= s.length()) throw new IllegalStateException("Unexpected end of JSON");
            return s.charAt(p);
        }

        private char next() {
            if (p >= s.length()) throw new IllegalStateException("Unexpected end of JSON");
            return s.charAt(p++);
        }

        private void expect(String word) {
            if (!s.startsWith(word, p)) {
                throw new IllegalStateException("Expected " + word);
            }
            p += word.length();
        }
    }
}
