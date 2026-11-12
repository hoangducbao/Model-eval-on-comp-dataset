package com.larplarpsahur;

import com.larplarpsahur.dataset.DatasetScanner;
import com.larplarpsahur.dataset.XlsxReader;
import com.larplarpsahur.pipeline.Metrics;
import com.larplarpsahur.pipeline.ResultExporter;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

/** Translation accuracy against the Round 3 translation tables bundled in assets/reference_tables. */
public class TranslationAccuracyTest {
    private static final File TABLES = new File("src/main/assets/reference_tables");
    @Rule public TemporaryFolder temp = new TemporaryFolder();

    private static List<String> column(String file, String header) throws Exception {
        List<String> out = new ArrayList<>();
        for (Map<String, String> row : XlsxReader.read(new File(TABLES, file))) out.add(row.get(header));
        return out;
    }

    /** Same synthetic hypotheses as the sacreBLEU run that produced the golden numbers below. */
    private static List<String> hypotheses(List<String> refs, boolean zh) {
        List<String> out = new ArrayList<>();
        for (int i = 0; i < refs.size(); i++) {
            String t = refs.get(i);
            if (i % 7 == 0) out.add(t);
            else if (i % 4 == 1) out.add(refs.get((i + 1) % refs.size()));
            else if (zh) out.add(t.substring(0, t.length() - 2));
            else out.add(t.substring(0, Math.max(0, t.lastIndexOf(' '))));
        }
        return out;
    }

    private static void golden(String file, String header, String lang, int rows,
                               double bleu, double chrf, double sentBleu, double sentChrf) throws Exception {
        List<String> refs = column(file, header), hyps = hypotheses(refs, "zh".equals(lang));
        assertEquals(rows, refs.size());
        Metrics.Bleu b = new Metrics.Bleu();
        Metrics.Chrf c = new Metrics.Chrf();
        for (int i = 0; i < refs.size(); i++) { b.add(refs.get(i), hyps.get(i), lang); c.add(refs.get(i), hyps.get(i), lang); }
        assertEquals(bleu, b.score(false), 1e-6);
        assertEquals(chrf, c.score(), 1e-6);
        Metrics.Bleu sb = new Metrics.Bleu();
        Metrics.Chrf sc = new Metrics.Chrf();
        sb.add(refs.get(2), hyps.get(2), lang);
        sc.add(refs.get(2), hyps.get(2), lang);
        assertEquals(sentBleu, sb.score(true), 1e-6);
        assertEquals(sentChrf, sc.score(), 1e-6);
    }

    /** Golden values: sacreBLEU 2.x, BLEU(tokenize=13a|zh, lowercase) and CHRF() on the normalized strings. */
    @Test public void bleuAndChrfMatchSacreBleuOnTheRealTables() throws Exception {
        golden("OneVoice_V_to_E_samples_100.xlsx", "tiếng anh - final", "en", 100,
                72.22050500676073, 74.00817650457418, 84.64817248906144, 84.24551840705247);
        golden("OneVoice_V_to_M_samples_180.xlsx", "man_text", "zh", 180,
                74.5600471556829, 74.05644020978484, 89.483931681437, 88.39782765520151);
        golden("OneVoice_V_to_M_samples_180.xlsx", "vie_text", "vi", 180,
                72.7561107190659, 75.24361210193622, 89.483931681437, 91.66692316631874);
        golden("OneVoice_E_to_V_samples_100.xlsx", "tiếng việt (normalized)", "vi", 100,
                75.46735994392287, 78.02547942048363, 94.28731438548752, 92.78607839181036);
    }

    @Test public void bundledTablesGiveEveryTrackItsReference() throws Exception {
        File root = temp.newFolder();
        // Layout of the pinned Hugging Face ZIP: no table that holds an EV row.
        for (String[] f : new String[][]{
                {"Vie-Eng dataset/Eng-Vie/wav/EV-0157.wav"}, {"Vie-Eng dataset/Vie-Eng/wav/VE-0037.wav"},
                {"Vie-Man dataset/Vie_voice/VM-0223.wav"}, {"Vie-Man dataset/Man_voice/VM-0223.wav"},
                {"Vie-Kor dataset/Vie_voice/VK-0143.wav"}}) Io.atomic(new File(root, f[0]), "");
        DatasetScanner.Scan none = DatasetScanner.scan(root, "en", true, null, 0);
        for (DatasetScanner.Sample s : none.samples) { assertNull(s.referenceTranslation); assertEquals("none", s.referenceSource); }

        DatasetScanner.Scan scan = DatasetScanner.scan(root, "en", true, null, 0, TABLES);
        assertEquals(4, scan.samples.size());
        DatasetScanner.Sample ev = find(scan, "EV-en"), ve = find(scan, "VE-vi"), vi = find(scan, "VM-vi"), zh = find(scan, "VM-zh");
        assertEquals("vi", ev.targetLanguage);
        assertTrue(ev.referenceTranscript.startsWith("play only all music released between the year one thousand"));
        assertTrue(ev.referenceTranslation.startsWith("phát chỉ tất cả nhạc được phát hành"));
        assertTrue(ev.referenceNormalized);
        assertEquals("bundled_table", ev.referenceSource);
        assertEquals("Anh giải thích lại chậm hơn được không?", ve.referenceTranscript);
        assertEquals("Could you explain that again more slowly?", ve.referenceTranslation);
        assertFalse(ve.referenceNormalized);
        assertEquals("你带护照了吗？", vi.referenceTranslation);
        assertEquals("Bạn đã mang hộ chiếu chưa?", zh.referenceTranslation);
        assertEquals("你带护照了吗？", zh.referenceTranscript);
        assertEquals(0, scan.report.getJSONArray("unmatched").length());
        assertEquals(1, scan.report.getJSONObject("reference_sources").getInt("EV-en/bundled_table"));
        // The sample survives the run.json round trip used by resume.
        DatasetScanner.Sample again = DatasetScanner.Sample.from(ev.json());
        assertTrue(again.referenceNormalized);
        assertEquals("bundled_table", again.referenceSource);
    }

    @Test public void datasetTableWinsOverTheBundledOne() throws Exception {
        File root = temp.newFolder();
        Io.atomic(new File(root, "Eng-Vie/wav/EV-0157.wav"), "");
        java.nio.file.Files.copy(new File(TABLES, "OneVoice_E_to_V_samples_100.xlsx").toPath(),
                new File(root, "Eng-Vie/OneVoice_E_to_V_samples_100.xlsx").toPath());
        DatasetScanner.Sample ev = DatasetScanner.scan(root, "en", true, null, 0, TABLES).samples.get(0);
        assertEquals("dataset_table", ev.referenceSource);
        assertTrue(ev.referenceNormalized);
        assertNotNull(ev.referenceTranslation);
    }

    @Test public void normalizedReferencesSpellDigitsOut() {
        String vi = Metrics.scoring("vi", true), en = Metrics.scoring("en", true);
        assertEquals("vi+num", vi);
        assertEquals("zh", Metrics.scoring("zh", true));
        assertEquals("vi", Metrics.scoring("vi", false));
        assertEquals("năm một nghìn chín trăm chín mươi và năm hai nghìn", Metrics.normalize("Năm 1990 và năm 2000.", vi));
        assertEquals("lúc năm giờ ba mươi sáng", Metrics.normalize("lúc 5 giờ 30 sáng", vi));
        assertEquals("mười sáu bốn mươi lăm", Metrics.normalize("16:45", vi));
        assertEquals("hai mươi mốt một trăm linh năm mười lăm", Metrics.normalize("21, 105, 15", vi));
        assertEquals("một triệu không trăm linh hai nghìn", Metrics.normalize("1.002.000", vi));
        assertEquals("hai phẩy năm", Metrics.normalize("2,5", vi));
        assertEquals("không chín một hai", Metrics.normalize("0912", vi));
        assertEquals("one thousand nine hundred and ninety", Metrics.normalize("1990", en));
        assertEquals("two thousand three hundred and thirty", Metrics.normalize("2,330", en));
        assertEquals("ten dollars", Metrics.normalize("$10 dollars", en));
        assertEquals("thirty two years old", Metrics.normalize("32 years old", en));
        assertEquals("one thousand and five", Metrics.normalize("1005", en));
        assertEquals("two point five", Metrics.normalize("2.5", en));
        assertEquals("five thirty", Metrics.normalize("5:30", en));
        // Without the normalized flag digits stay digits.
        assertEquals("năm 1990", Metrics.normalize("Năm 1990.", "vi"));
        assertTrue(Metrics.exact("năm một nghìn chín trăm chín mươi", "Năm 1990.", vi));
        assertFalse(Metrics.exact("năm một nghìn chín trăm chín mươi", "Năm 1990.", "vi"));
        assertEquals(0, Metrics.words("lúc năm giờ ba mươi sáng", "Lúc 5 giờ 30 sáng.", vi).edits);
    }

    @Test public void accuracyIsOneMinusCerFlooredAtZero() {
        assertEquals(75.0, Metrics.accuracy(Metrics.chars("abcd", "abcx", "en")), 1e-12);
        assertEquals(0.0, Metrics.accuracy(Metrics.chars("ab", "something much longer", "en")), 0);
        assertNull(Metrics.accuracy(Metrics.chars(null, "anything", "en")));
        Metrics.Chrf none = new Metrics.Chrf();
        none.add(null, "x", "en");
        assertNull(none.score());
        Metrics.Chrf empty = new Metrics.Chrf();
        empty.add("reference", "", "en");
        assertEquals(0.0, empty.score(), 0);
    }

    @Test public void exportReportsTranslationAccuracyPerTrack() throws Exception {
        File root = temp.newFolder();
        JSONArray samples = new JSONArray();
        samples.put(new JSONObject().put("id", "EV-1").put("track", "EV-en").put("audio", "a.wav").put("src_lang", "en").put("tgt_lang", "vi")
                .put("ref_text", "at five thirty").put("ref_trans", "lúc năm giờ ba mươi").put("ref_source", "bundled_table").put("ref_normalized", true));
        samples.put(new JSONObject().put("id", "EV-2").put("track", "EV-en").put("audio", "b.wav").put("src_lang", "en").put("tgt_lang", "vi")
                .put("ref_text", "thank you").put("ref_trans", "cảm ơn bạn").put("ref_source", "bundled_table").put("ref_normalized", true));
        samples.put(new JSONObject().put("id", "X-1").put("track", "manual").put("audio", "c.wav").put("src_lang", "en").put("tgt_lang", "vi")
                .put("ref_text", JSONObject.NULL).put("ref_trans", JSONObject.NULL));
        ResultExporter exporter = new ResultExporter(root, new JSONObject().put("samples", samples).put("mt_mode", "pipeline"));
        exporter.save(0, exporter.record(0).put("asr_done", true).put("mt_done", true).put("status", "ok")
                .put("asr_hyp", "At 5:30.").put("mt_hyp", "Lúc 5 giờ 30."));
        exporter.save(1, exporter.record(1).put("asr_done", true).put("mt_done", true).put("status", "ok")
                .put("asr_hyp", "Thank you.").put("mt_hyp", "Cảm ơn anh."));
        exporter.save(2, exporter.record(2).put("asr_done", true).put("mt_done", true).put("status", "ok")
                .put("asr_hyp", "Hello.").put("mt_hyp", "Xin chào."));
        String text = exporter.export("complete", null);
        assertTrue(text, text.contains("EV-en/pipeline: BLEU"));
        JSONObject summary = new JSONObject(Io.read(new File(root, "eval_summary.json"), 1 << 20));
        JSONObject ev = summary.getJSONObject("mt").getJSONObject("EV-en/pipeline");
        assertEquals(2, ev.getInt("samples"));
        assertEquals(50.0, ev.getDouble("exact_match"), 1e-9);
        // "cảm ơn anh" vs "cảm ơn bạn": 3 edits over 19 + 10 reference characters.
        assertEquals(3, ev.getInt("edits_char"));
        assertEquals(29, ev.getInt("ref_chars"));
        assertEquals(100 * (1 - 3 / 29.0), ev.getDouble("accuracy"), 1e-9);
        assertTrue(ev.getDouble("chrf") > 50 && ev.getDouble("chrf") < 100);
        assertTrue(ev.getString("signature").contains("spell-digits-v1"));
        // ASR is scored with the same number spelling: "At 5:30." is an exact transcript.
        assertEquals(0, summary.getJSONObject("asr").getJSONObject("EV-en").getInt("edits_word"));
        // Samples without a reference are exported but never counted.
        assertEquals(0, summary.getJSONObject("mt").getJSONObject("manual/pipeline").getInt("samples"));
        assertTrue(summary.getJSONObject("mt").getJSONObject("manual/pipeline").isNull("accuracy"));
        String table = Io.read(new File(root, "eval_mt_summary.csv"), 1 << 20);
        assertTrue(table.startsWith("track,mt_input,samples,bleu,chrf,cer,accuracy,exact_match"));
        assertTrue(table.contains("\"EV-en\",\"asr_hyp\",\"2\""));
        assertTrue(Io.read(new File(root, "eval_mt.csv"), 1 << 20).contains("\"bundled_table\""));
    }

    private static DatasetScanner.Sample find(DatasetScanner.Scan s, String track) {
        return s.samples.stream().filter(x -> x.track.equals(track)).findFirst().get();
    }
}
