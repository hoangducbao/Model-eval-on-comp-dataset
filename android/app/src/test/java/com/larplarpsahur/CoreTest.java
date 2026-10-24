package com.larplarpsahur;

import com.larplarpsahur.dataset.*;
import com.larplarpsahur.pipeline.*;
import org.json.*;
import org.junit.*;
import org.junit.rules.TemporaryFolder;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.*;
import static org.junit.Assert.*;

public class CoreTest {
    @Rule public TemporaryFolder temp=new TemporaryFolder();
    @Test public void quadataIsPinnedToTheOnlyZip() {
        assertEquals("fed18c73b4350e07a02124df4f4c716a95e26743",DatasetDownloader.REVISION);
        assertEquals("cde6b58e391094c7578a7bdd37a8202db0735ccd165f91cc18c6f7ded1f8f01e",DatasetDownloader.ZIP_SHA256);
        assertTrue(DatasetDownloader.DATASET_PAGE.endsWith("Echdipi/quadata/tree/main"));
    }
    @Test public void unicodeAndCorpusErrors() {
        assertEquals("tiếng việt",Metrics.normalize("  TIE\u0302\u0301NG\u00a0VIỆT! ","vi"));
        assertEquals("你好",Metrics.normalize("你 好！","zh"));
        assertEquals(1,Metrics.chars("😀a","😀b","en").edits);
        assertEquals(2,Metrics.chars("😀a","😀b","en").length);
        Metrics.Errors corpus=Metrics.words("one two","one bad","en");
        corpus.add(Metrics.words("one two three four five six eight nine","one two three four five six eight nine","en"));
        corpus.add(Metrics.words(null,"unreferenced hypotheses must not count","en"));
        assertEquals(.1,corpus.rate(),1e-12);assertNull(Metrics.words("你好","你","zh").rate());
        assertEquals(0,Metrics.chars(null,"ignored","vi").edits);
    }
    @Test public void bleuGoldenCases() {
        Metrics.Bleu b=new Metrics.Bleu();b.add("a b c d","a b c d","en");
        assertEquals(100,b.score(false),1e-9);
        Metrics.Bleu shortSentence=new Metrics.Bleu();shortSentence.add("a b","a b","en");
        assertEquals(100,shortSentence.score(true),1e-9);assertEquals(0,shortSentence.score(false),1e-9);
        Metrics.Bleu smooth=new Metrics.Bleu();smooth.add("a b c d","a b x y","en");
        assertEquals(31.947155212313625,smooth.score(false),1e-9);
        Metrics.Bleu bp=new Metrics.Bleu();bp.add("a b c d e","a b c d","en");
        assertEquals(100*Math.exp(-.25),bp.score(false),1e-9);
        Metrics.Bleu absent=new Metrics.Bleu();absent.add(null,"a b","en");assertNull(absent.score(false));
        Metrics.Bleu silence=new Metrics.Bleu();silence.add("some words","","en");assertEquals(0,silence.score(true),0);
    }
    @Test public void round3ReferencesAndOverride() throws Exception {
        File root=temp.newFolder();
        file(root,"Vie-Man dataset/Vie_voice/VM-1.wav","");
        file(root,"Vie-Man dataset/Man_voice/VM-1.wav","");
        file(root,"Vie-Man dataset/Vie_text/VM-1.txt","Xin chào");
        file(root,"Vie-Man dataset/Man_text/VM-1.txt","你好");
        file(root,"Vie-Eng/Eng-Vie/wav/EV-1.wav","");
        file(root,"Vie-Eng/Eng-Vie/wav/EV-1.txt","Must not be joined");
        file(root,"Vie-Kor/VK-1.wav","");
        DatasetScanner.Scan scan=DatasetScanner.scan(root,"en",true,null,0);
        assertEquals(3,scan.samples.size());
        DatasetScanner.Sample vi=find(scan,"VM-vi"),zh=find(scan,"VM-zh"),ev=find(scan,"EV-en");
        assertEquals("zh",vi.targetLanguage);assertEquals("你好",vi.referenceTranslation);
        assertEquals("Xin chào",vi.referenceTranscript);assertEquals("vi",zh.targetLanguage);
        assertEquals("你好",zh.referenceTranscript);assertNull(ev.referenceTranscript);assertNull(ev.referenceTranslation);
        DatasetScanner.Scan override=DatasetScanner.scan(root,"en",false,"VM-vi",1);
        assertEquals(1,override.samples.size());assertNull(override.samples.get(0).referenceTranslation);
    }
    @Test public void xlsxUsesNamedSheetSparseColumnsSharedAndInlineStrings() throws Exception {
        File root=temp.newFolder();file(root,"wav/VE-2.wav","");
        File workbook=new File(root,"refs.xlsx");
        Map<String,String> parts=new LinkedHashMap<>();
        parts.put("xl/workbook.xml","<workbook xmlns:r=\"urn:r\"><sheets><sheet name=\"Ignore\" r:id=\"r1\"/><sheet name=\"Final\" r:id=\"r2\"/></sheets></workbook>");
        parts.put("xl/_rels/workbook.xml.rels","<Relationships><Relationship Id=\"r1\" Target=\"worksheets/sheet1.xml\"/><Relationship Id=\"r2\" Target=\"/xl/worksheets/sheet2.xml\"/></Relationships>");
        parts.put("xl/sharedStrings.xml","<sst><si><t>ID</t></si><si><t>Tiếng Việt</t></si><si><t>Tiếng Anh - Final</t></si><si><t>VE-2</t></si></sst>");
        parts.put("xl/worksheets/sheet1.xml","<worksheet><sheetData/></worksheet>");
        parts.put("xl/worksheets/sheet2.xml","<worksheet><sheetData><row><c r=\"A1\" t=\"s\"><v>0</v></c><c r=\"C1\" t=\"s\"><v>1</v></c><c r=\"F1\" t=\"s\"><v>2</v></c></row><row><c r=\"A2\" t=\"s\"><v>3</v></c><c r=\"C2\" t=\"inlineStr\"><is><t>Chào bạn</t></is></c><c r=\"F2\" t=\"inlineStr\"><is><t>Hello friend</t></is></c></row></sheetData></worksheet>");
        zip(workbook,parts);
        DatasetScanner.Sample sample=DatasetScanner.scan(root,"vi",true,null,0).samples.get(0);
        assertEquals("Chào bạn",sample.referenceTranscript);assertEquals("Hello friend",sample.referenceTranslation);
        assertEquals("en",sample.targetLanguage);
    }
    @Test public void rejectsTraversalDuplicatePathsAndManifestEscape() throws Exception {
        File root=temp.newFolder(),archive=temp.newFile("bad.zip");Map<String,String> parts=new LinkedHashMap<>();
        parts.put("../outside.wav","bad");zip(archive,parts);
        assertThrows(IOException.class,()->DatasetDownloader.extract(archive,root,()->false,(p,s)->{}));
        parts.clear();parts.put("a.wav","1");parts.put("nested/../a.wav","2");zip(archive,parts);
        assertThrows(IOException.class,()->DatasetDownloader.extract(archive,root,()->false,(p,s)->{}));
        file(root,"manifest.json","{\"samples\":[{\"audio\":\"../outside.wav\"}]}");
        assertThrows(IOException.class,()->DatasetScanner.scan(root,"en",true,null,0));
    }
    @Test public void exportResumeAndNullReferences() throws Exception {
        File root=temp.newFolder();JSONArray samples=new JSONArray();
        samples.put(new JSONObject().put("id","line,\"quoted\"\n😀").put("track","manual").put("audio","a.wav").put("src_lang","vi").put("tgt_lang","en").put("ref_text",JSONObject.NULL).put("ref_trans",JSONObject.NULL));
        samples.put(new JSONObject(samples.getJSONObject(0).toString()).put("id","second"));
        ResultExporter exporter=new ResultExporter(root,new JSONObject().put("samples",samples).put("mt_mode","pipeline"));
        JSONObject r=exporter.record(0).put("asr_done",true).put("mt_done",true).put("asr_hyp","Xin chào").put("mt_hyp","Hello,\n\"world\"").put("status","ok");
        exporter.save(0,r);exporter.export("cancelled",null);
        assertFalse(new File(root,"DONE").exists());assertTrue(new File(root,"CANCELLED").isFile());
        assertTrue(RunRecovery.hasCheckpoint(root.getAbsolutePath()));
        ResultExporter resumed=new ResultExporter(root,null);assertTrue(resumed.record(0).getBoolean("mt_done"));assertEquals("pending",resumed.record(1).getString("status"));
        resumed.save(1,new JSONObject(r.toString()).put("id","second"));resumed.export("complete",null);
        assertTrue(new File(root,"DONE").isFile());assertFalse(new File(root,"CANCELLED").exists());
        assertFalse(RunRecovery.hasCheckpoint(root.getAbsolutePath()));
        JSONObject report=new JSONObject(Io.read(new File(root,"results.json"),100000));
        assertEquals("Hello,\n\"world\"",report.getJSONArray("samples").getJSONObject(0).getString("mt_hyp"));
        assertTrue(report.getJSONObject("summary").getJSONObject("asr").getJSONObject("ALL").isNull("cer"));
        assertEquals(0,report.getJSONObject("summary").getJSONObject("asr").getJSONObject("ALL").getInt("edits_char"));
        assertTrue(Io.read(new File(root,"results.csv"),100000).contains("\"Hello,\n\"\"world\"\"\""));
        File archive=new File(temp.getRoot(),"results.zip");resumed.zip(new FileOutputStream(archive));
        try(ZipFile zipped=new ZipFile(archive)){assertNotNull(zipped.getEntry("eval_summary.json"));assertNotNull(zipped.getEntry("DONE"));assertNull(zipped.getEntry("run.json"));}
    }
    private static DatasetScanner.Sample find(DatasetScanner.Scan s,String track){return s.samples.stream().filter(x->x.track.equals(track)).findFirst().get();}
    private static void file(File root,String path,String text) throws Exception {Io.atomic(new File(root,path),text);}
    private static void zip(File file,Map<String,String> parts) throws Exception {
        try(ZipOutputStream z=new ZipOutputStream(new FileOutputStream(file))){
            for(Map.Entry<String,String> p:parts.entrySet()){z.putNextEntry(new ZipEntry(p.getKey()));z.write(p.getValue().getBytes(StandardCharsets.UTF_8));z.closeEntry();}
        }
    }
}
