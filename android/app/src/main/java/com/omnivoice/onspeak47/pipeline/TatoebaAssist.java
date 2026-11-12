/*
 * OmniVoice — Tatoeba dictionary + TM assist (Android runtime).
 *
 * Mirror of backend/tatoeba_assist.py over the same SQLite schema built by
 * optimize/15_fetch_tatoeba.py (tatoeba_dict.db + tatoeba_tm_mobile.db,
 * staged into assets/tatoeba/). Text primitives live in pure-Java
 * {@link TatoebaText} (JVM-tested); this class is the thin
 * android.database.sqlite shell. Every query is read-only and guarded:
 * any failure (absent assets, corrupt DB) degrades to no-assist, never to
 * a translation failure. Call off the main thread (translateSentence
 * already runs on the pipeline executor).
 */
package com.omnivoice.onspeak47.pipeline;

import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class TatoebaAssist {

    private static final String TAG = "TatoebaAssist";
    private static final String ASSET_DIR = "tatoeba";
    private static final String DICT_NAME = "tatoeba_dict.db";
    // Mobile TM slice (short pairs only) — see optimize/15_fetch_tatoeba.py.
    private static final String TM_NAME = "tatoeba_tm_mobile.db";

    /** Assist outcome for one piece of text. */
    public static class AssistResult {
        /** Direct dictionary translation (null = fall through to the LLM). */
        public final String direct;
        /** Few-shot TM pairs (possibly empty). Each entry is {src, tgt}. */
        public final List<String[]> examples;
        public final boolean dictHit;

        AssistResult(String direct, List<String[]> examples) {
            this.direct = direct;
            this.examples = examples;
            this.dictHit = direct != null;
        }
    }

    private final Context appContext;
    private SQLiteDatabase dictDb;
    private SQLiteDatabase tmDb;
    private boolean disabled;

    public TatoebaAssist(Context context) {
        this.appContext = context.getApplicationContext();
        try {
            File dir = new File(appContext.getFilesDir(), ASSET_DIR);
            File dictFile = ensureAsset(ASSET_DIR + "/" + DICT_NAME,
                    new File(dir, DICT_NAME));
            File tmFile = ensureAsset(ASSET_DIR + "/" + TM_NAME,
                    new File(dir, TM_NAME));
            if (dictFile != null) {
                dictDb = SQLiteDatabase.openDatabase(dictFile.getAbsolutePath(),
                        null, SQLiteDatabase.OPEN_READONLY);
            }
            if (tmFile != null) {
                tmDb = SQLiteDatabase.openDatabase(tmFile.getAbsolutePath(),
                        null, SQLiteDatabase.OPEN_READONLY);
            }
            Log.i(TAG, "ready dict=" + (dictDb != null)
                    + " tm=" + (tmDb != null));
        } catch (Exception e) {
            Log.w(TAG, "tatoeba assist unavailable — translations continue unaided", e);
            disabled = true;
        }
        if (dictDb == null && tmDb == null) disabled = true;
    }

    /** Copy an asset to internal storage when absent/incomplete. Null when missing. */
    private File ensureAsset(String assetPath, File dest) {
        try {
            appContext.getAssets().open(assetPath).close();
        } catch (Exception e) {
            return null; // not bundled — assist stays off for this DB
        }
        try {
            File parent = dest.getParentFile();
            if (parent != null) parent.mkdirs();
            if (dest.exists() && dest.length() > 0) return dest;
            try (InputStream in = appContext.getAssets().open(assetPath);
                 OutputStream out = new FileOutputStream(dest)) {
                byte[] buf = new byte[1 << 20];
                int n;
                while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
            }
            Log.i(TAG, "copied asset " + assetPath + " (" + dest.length() / 1048576 + " MB)");
            return dest;
        } catch (Exception e) {
            Log.w(TAG, "asset copy failed: " + assetPath, e);
            return null;
        }
    }

    public synchronized boolean isAvailable() {
        return !disabled && (dictDb != null || tmDb != null);
    }

    /**
     * Assist one piece of text: confident dictionary hit first, else TM
     * examples (possibly empty). Mirrors backend/tatoeba_assist lookup()
     * + examples() including thresholds and script filtering.
     */
    public synchronized AssistResult assist(String text, String srcLang, String tgtLang) {
        if (disabled || text == null) {
            return new AssistResult(null, Collections.emptyList());
        }
        // A glossary term must reach the model: a dict hit would bypass it.
        String direct = Glossary.fragment(text, srcLang, tgtLang).isEmpty() ? lookup(text, srcLang, tgtLang) : null;
        if (direct != null) {
            Log.i(TAG, "dict hit [" + srcLang + "->" + tgtLang + "]: \"" + text + "\"");
            return new AssistResult(direct, Collections.emptyList());
        }
        List<String[]> ex = examples(text, srcLang, tgtLang);
        if (!ex.isEmpty()) {
            Log.d(TAG, "tm examples=" + ex.size() + " [" + srcLang + "->" + tgtLang + "]");
        }
        return new AssistResult(null, ex);
    }

    private String lookup(String text, String srcLang, String tgtLang) {
        if (dictDb == null) return null;
        List<String> words = TatoebaText.tokWords(text);
        if (words.size() != TatoebaText.DICT_MAX_WORDS) return null;
        String dir = TatoebaText.direction(srcLang, tgtLang);
        String w = words.get(0);
        try {
            String[] top = top1(dictDb, dir, w);
            if (top == null || !directOk(top)) return null;
            String[] back = top1(dictDb, TatoebaText.direction(tgtLang, srcLang), top[0]);
            if (back == null || !directOk(back) || !back[0].equals(w)) return null;
            return top[0];
        } catch (Exception e) {
            Log.w(TAG, "dict lookup failed", e);
            return null;
        }
    }

    /** {word, dice, cooc} of the top-1 row, or null. */
    private static String[] top1(SQLiteDatabase db, String dir, String word) {
        Cursor c = null;
        try {
            c = db.rawQuery(
                    "SELECT tgt_word, dice, cooc FROM dict WHERE direction=? AND src_word=? "
                    + "ORDER BY dice DESC LIMIT 1",
                    new String[]{dir, word});
            if (c.moveToFirst()) {
                return new String[]{c.getString(0), c.getString(1), c.getString(2)};
            }
            return null;
        } finally {
            if (c != null) c.close();
        }
    }

    private static boolean directOk(String[] row) {
        if (row == null) return false;
        try {
            return Double.parseDouble(row[1]) >= TatoebaText.DICT_MIN_DICE
                    && Long.parseLong(row[2]) >= TatoebaText.DICT_MIN_COOC;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    private List<String[]> examples(String text, String srcLang, String tgtLang) {
        if (tmDb == null) return Collections.emptyList();
        List<String> toks = new ArrayList<>(new HashSet<>(TatoebaText.tokWords(text)));
        // Mirror Python: drop single latin chars, keep single CJK chars.
        List<String> kept = new ArrayList<>();
        for (String t : toks) {
            if (t.length() > 1 || TatoebaText.cjkRatio(t) > 0) kept.add(t);
        }
        if (kept.isEmpty()) return Collections.emptyList();
        String dir = TatoebaText.direction(srcLang, tgtLang);
        Cursor c = null;
        List<String[]> cands = new ArrayList<>();
        try {
            StringBuilder in = new StringBuilder();
            String[] args = new String[kept.size() + 1];
            args[0] = dir;
            for (int i = 0; i < kept.size(); i++) {
                if (i > 0) in.append(',');
                in.append('?');
                args[i + 1] = kept.get(i);
            }
            c = tmDb.rawQuery(
                    "SELECT t.src_text, t.tgt_text, COUNT(*) AS c FROM tm t "
                    + "JOIN tok x ON x.rowid=t.rowid WHERE t.direction=? AND x.token IN ("
                    + in + ") GROUP BY t.rowid ORDER BY c DESC LIMIT "
                    + TatoebaText.TM_CANDIDATES,
                    args);
            while (c.moveToNext()) {
                cands.add(new String[]{c.getString(0), c.getString(1)});
            }
        } catch (Exception e) {
            Log.w(TAG, "tm prefilter failed", e);
            return Collections.emptyList();
        } finally {
            if (c != null) c.close();
        }
        // Python-mirror rerank: CJK-heavy → bigram floor, else token floor.
        boolean useBigram = TatoebaText.cjkRatio(text) > 0.5;
        double floor = useBigram ? TatoebaText.TM_MIN_SIM_CJK : TatoebaText.TM_MIN_SIM;
        String tl = tgtLang == null ? "" : tgtLang.toLowerCase(java.util.Locale.ROOT);
        String wantScript = tl.contains("hant") ? "hant" : (tl.contains("hans") ? "hans" : "");
        List<Scored> scored = new ArrayList<>();
        for (String[] cand : cands) {
            if (cand[0] == null || cand[1] == null
                    || cand[0].isEmpty() || cand[1].isEmpty()) continue;
            if (!wantScript.isEmpty()) {
                String tag = TatoebaText.scriptTag(cand[1]);
                if (!tag.equals("unknown") && !tag.equals(wantScript)) continue;
            }
            double s = useBigram
                    ? TatoebaText.bigramScore(text, cand[0])
                    : TatoebaText.tokenScore(text, cand[0]);
            if (s >= floor) scored.add(new Scored(s, cand[0], cand[1]));
        }
        Collections.sort(scored, (a, b) -> Double.compare(b.score, a.score));
        List<String[]> out = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        for (Scored s : scored) {
            String src = s.src.length() > TatoebaText.TM_MAX_LEN
                    ? s.src.substring(0, TatoebaText.TM_MAX_LEN) : s.src;
            String tgt = s.tgt.length() > TatoebaText.TM_MAX_LEN
                    ? s.tgt.substring(0, TatoebaText.TM_MAX_LEN) : s.tgt;
            if (seen.add(src)) {
                out.add(new String[]{src, tgt});
                if (out.size() >= TatoebaText.TM_TOP_K) break;
            }
        }
        return out;
    }

    private static class Scored {
        final double score;
        final String src;
        final String tgt;

        Scored(double score, String src, String tgt) {
            this.score = score;
            this.src = src;
            this.tgt = tgt;
        }
    }

    public synchronized void close() {
        try {
            if (dictDb != null) dictDb.close();
        } catch (Exception ignored) {
        }
        try {
            if (tmDb != null) tmDb.close();
        } catch (Exception ignored) {
        }
        dictDb = null;
        tmDb = null;
        disabled = true;
    }
}
