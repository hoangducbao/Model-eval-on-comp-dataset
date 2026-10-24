package com.larplarpsahur.pipeline;

import com.larplarpsahur.Io;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.*;
import org.json.*;

/** Atomic per-sample state is authoritative; JSONL is an append-only recovery/audit trail. */
public final class ResultExporter {
    public final File directory;
    public final JSONObject run;
    public final JSONArray samples;
    private final File state;
    public ResultExporter(File directory,JSONObject config) throws Exception {
        this.directory=directory;Io.mkdir(directory);state=new File(directory,"state");Io.mkdir(state);
        File configFile=new File(directory,"run.json");
        if(config!=null)Io.atomic(configFile,config.toString(2));
        run=new JSONObject(Io.read(configFile,32*1024*1024));samples=run.getJSONArray("samples");
    }
    public JSONObject record(int index) throws Exception {
        File file=new File(state,index+".json");
        if(file.isFile())return new JSONObject(Io.read(file,4*1024*1024));
        JSONObject r=new JSONObject(samples.getJSONObject(index).toString());r.put("status","pending");return r;
    }
    public void save(int index,JSONObject record) throws Exception {
        Io.atomic(new File(state,index+".json"),record.toString());
        try(FileOutputStream out=new FileOutputStream(new File(directory,"results.partial.jsonl"),true)) {
            out.write((record.toString()+"\n").getBytes(StandardCharsets.UTF_8));out.getFD().sync();
        }
    }
    public String export(String status,String failure) throws Exception {
        JSONArray results=new JSONArray(),transcripts=new JSONArray(),translations=new JSONArray(),asrRows=new JSONArray(),mtRows=new JSONArray();
        Map<String,Aggregate> asrTotals=new TreeMap<>(),mtTotals=new TreeMap<>();
        int ok=0,errors=0,pending=0;
        for(int i=0;i<samples.length();i++) {
            JSONObject r=record(i);String track=r.optString("track"),id=r.optString("id");
            String src=r.optString("src_lang","und"),expected=r.optString("reference_language",src),tgt=r.optString("tgt_lang");
            String hyp=r.optString("asr_hyp",""),ref=nullable(r,"ref_text");
            r.put("duration_s",r.optDouble("duration_s",0)).put("asr_ms",r.optLong("asr_ms",0)).put("mt_ms",r.optLong("mt_ms",0));
            r.put("rtf",r.optDouble("duration_s",0)>0?r.optLong("asr_ms",0)/(1000*r.optDouble("duration_s")):JSONObject.NULL);
            if(r.optBoolean("asr_done")) {
                JSONObject a=new JSONObject().put("track",track).put("id",id).put("lang",expected).put("ref_text",nil(ref)).put("asr_hyp",hyp);
                Metrics.Errors ce=Metrics.chars(ref,hyp,expected),we=Metrics.words(ref,hyp,expected);
                a.put("cer",nil(ce.rate())).put("wer",nil(we.rate())).put("ref_chars",ce.length).put("ref_words",we.length).put("edits_char",ce.edits).put("edits_word",we.edits);
                r.put("cer",nil(ce.rate())).put("wer",nil(we.rate()));asrRows.put(a);
                aggregate(asrTotals,track).add(ce,we,null,null,expected);aggregate(asrTotals,"ALL").add(ce,we,null,null,expected);
            } else r.put("cer",JSONObject.NULL).put("wer",JSONObject.NULL);
            transcripts.put(new JSONObject().put("track",track).put("id",id).put("audio_path",r.optString("audio")).put("lang",src).put("asr_hyp",hyp).put("status",r.optString("status")));
            for(String mode:new String[]{"pipeline","oracle"}) {
                String key=mode.equals("pipeline")?"mt_hyp":"mt_oracle_hyp",translated=nullable(r,key),mtRef=nullable(r,"ref_trans");
                if(translated==null)continue;
                String input=mode.equals("pipeline")?"asr_hyp":"ref_text";
                JSONObject m=new JSONObject().put("track",track).put("id",id).put("src_lang",src).put("tgt_lang",tgt)
                    .put("mt_input",input).put("ref_trans",nil(mtRef)).put("mt_hyp",translated);
                Metrics.Errors ce=Metrics.chars(mtRef,translated,tgt);Metrics.Bleu bleu=new Metrics.Bleu();bleu.add(mtRef,translated,tgt);
                m.put("cer",nil(ce.rate())).put("sent_bleu",nil(bleu.score(true))).put("signature",Metrics.signature(tgt,true));
                translations.put(new JSONObject(m.toString()));mtRows.put(m);
                aggregate(mtTotals,track+"/"+mode).add(ce,new Metrics.Errors(0,0),mtRef,translated,tgt);
                aggregate(mtTotals,"ALL/"+mode).add(ce,new Metrics.Errors(0,0),mtRef,translated,tgt);
                if(mode.equals("pipeline"))r.put("mt_cer",nil(ce.rate())).put("sent_bleu",nil(bleu.score(true)));
            }
            if(!r.has("mt_hyp"))r.put("mt_hyp",JSONObject.NULL);
            if(!r.has("sent_bleu"))r.put("sent_bleu",JSONObject.NULL);
            if(!r.has("error"))r.put("error",JSONObject.NULL);
            String rs=r.optString("status");
            if("ok".equals(rs))ok++;else if("error".equals(rs))errors++;else pending++;
            results.put(r);
        }
        JSONObject summary=new JSONObject().put("status",status).put("successful_samples",ok).put("error_samples",errors).put("pending_samples",pending)
            .put("normalization",Metrics.PROFILE).put("wer_zh","N/A").put("failure",nil(failure));
        JSONObject asrSummary=new JSONObject(),mtSummary=new JSONObject();
        for(Map.Entry<String,Aggregate> e:asrTotals.entrySet()) {
            JSONObject total=e.getValue().json().put("track",e.getKey()).put("id","TOTAL");asrSummary.put(e.getKey(),total);asrRows.put(total);
        }
        for(Map.Entry<String,Aggregate> e:mtTotals.entrySet()) {
            JSONObject total=e.getValue().json().put("track",e.getKey()).put("id","TOTAL");mtSummary.put(e.getKey(),total);mtRows.put(total);
        }
        summary.put("asr",asrSummary).put("mt",mtSummary);
        JSONObject meta=new JSONObject(run.toString());meta.remove("samples");meta.put("status",status).put("finished_at_ms",System.currentTimeMillis());
        Io.atomic(new File(directory,"results.json"),new JSONObject().put("schema_version",1).put("run",meta).put("summary",summary).put("samples",results).toString(2));
        csv("results.csv",results,"track,id,audio,duration_s,src_lang,tgt_lang,ref_text,asr_hyp,wer,cer,ref_trans,mt_hyp,mt_oracle_hyp,mt_cer,sent_bleu,asr_ms,mt_ms,rtf,status,error");
        Io.atomic(new File(directory,"transcripts.json"),transcripts.toString(2));
        Io.atomic(new File(directory,"translations.json"),translations.toString(2));
        csv("transcripts.csv",transcripts,"track,id,audio_path,lang,asr_hyp,status");
        csv("translations.csv",translations,"track,id,src_lang,tgt_lang,mt_input,ref_trans,mt_hyp,cer,sent_bleu,signature");
        csv("eval_asr.csv",asrRows,"track,id,lang,ref_text,asr_hyp,wer,cer,ref_words,ref_chars,edits_word,edits_char");
        csv("eval_mt.csv",mtRows,"track,id,src_lang,tgt_lang,mt_input,ref_trans,mt_hyp,cer,sent_bleu,bleu,signature");
        Io.atomic(new File(directory,"eval_summary.json"),summary.toString(2));
        new File(directory,"DONE").delete();new File(directory,"CANCELLED").delete();new File(directory,"FAILED").delete();
        Io.atomic(new File(directory,status.equals("complete")?"DONE":status.equals("cancelled")?"CANCELLED":"FAILED"),summary.toString(2));
        return "OK: "+ok+" · Lỗi: "+errors+" · Chưa xong: "+pending+"\n"+summary.optJSONObject("asr").toString();
    }
    private void csv(String name,JSONArray rows,String columns) throws Exception {
        String[] keys=columns.split(",");StringBuilder out=new StringBuilder(columns).append("\r\n");
        for(int i=0;i<rows.length();i++) { JSONObject row=rows.getJSONObject(i);for(int j=0;j<keys.length;j++){ if(j>0)out.append(',');out.append(Io.csv(row.opt(keys[j]))); }out.append("\r\n"); }
        Io.atomic(new File(directory,name),out.toString());
    }
    private static Object nil(Object v) { return v==null?JSONObject.NULL:v; }
    private static String nullable(JSONObject r,String k) { return r.isNull(k)?null:r.optString(k); }
    private static Aggregate aggregate(Map<String,Aggregate> map,String key) { return map.computeIfAbsent(key,k->new Aggregate()); }
    private static final class Aggregate {
        final Metrics.Errors chars=new Metrics.Errors(0,0),words=new Metrics.Errors(0,0);final Metrics.Bleu bleu=new Metrics.Bleu();final Set<String> languages=new TreeSet<>();
        void add(Metrics.Errors c,Metrics.Errors w,String ref,String hyp,String lang) { chars.add(c);words.add(w);bleu.add(ref,hyp,lang);languages.add(lang); }
        JSONObject json() throws Exception { return new JSONObject().put("cer",nil(chars.rate())).put("wer",nil(words.rate())).put("ref_chars",chars.length).put("ref_words",words.length).put("edits_char",chars.edits).put("edits_word",words.edits).put("bleu",nil(bleu.score(false))).put("signature",languages.size()==1?Metrics.signature(languages.iterator().next(),false):"mixed targets; per-sample tokenization; "+Metrics.PROFILE); }
    }
    public void zip(OutputStream stream) throws Exception {
        try(ZipOutputStream zip=new ZipOutputStream(stream)) {
            File[] files=directory.listFiles();if(files==null)throw new IOException("No results");
            for(File f:files)if(f.isFile()&&!f.getName().endsWith(".tmp")&&!f.getName().equals("run.json")) {
                zip.putNextEntry(new ZipEntry(f.getName()));
                try(InputStream in=new FileInputStream(f)) { byte[] b=new byte[65536];int n;while((n=in.read(b))!=-1)zip.write(b,0,n); }
                zip.closeEntry();
            }
        }
    }
}
