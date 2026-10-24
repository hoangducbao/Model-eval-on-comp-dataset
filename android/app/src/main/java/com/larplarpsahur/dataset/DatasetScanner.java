package com.larplarpsahur.dataset;

import com.larplarpsahur.Io;
import org.json.*;
import java.io.*;
import java.util.*;

/** Round 3 joins are by ID only. EV references are never inferred from the copied VE table. */
public final class DatasetScanner {
    public static final class Sample {
        public String id,audio,track="manual",sourceLanguage="und",targetLanguage;
        public String referenceTranscript,referenceTranslation;
        public Sample(String id,String audio) { this.id=id;this.audio=audio; }
        public JSONObject json() throws Exception {
            return new JSONObject().put("id",id).put("audio",audio).put("track",track)
                .put("src_lang",sourceLanguage).put("tgt_lang",targetLanguage)
                .put("ref_text",referenceTranscript==null?JSONObject.NULL:referenceTranscript)
                .put("ref_trans",referenceTranslation==null?JSONObject.NULL:referenceTranslation);
        }
        public static Sample from(JSONObject o) {
            Sample s=new Sample(o.optString("id"),o.optString("audio"));s.track=o.optString("track","manual");
            s.sourceLanguage=o.optString("src_lang","und");s.targetLanguage=nullable(o,"tgt_lang");
            s.referenceTranscript=nullable(o,"ref_text");s.referenceTranslation=nullable(o,"ref_trans");return s;
        }
    }
    public static final class Scan {
        public final List<Sample> samples=new ArrayList<>();
        public final JSONObject report=new JSONObject();
    }
    public static Scan scan(File root,String target,boolean trackTargets,String tracks,int limit) throws Exception {
        if(!root.isDirectory()) throw new IOException("Dataset directory unavailable");
        List<File> files=new ArrayList<>();collect(root,root,files,0);
        List<File> manifests=new ArrayList<>();
        for(File f:files) if("manifest.json".equalsIgnoreCase(f.getName())) manifests.add(f);
        Scan scan=new Scan();JSONArray warnings=new JSONArray();
        if(manifests.size()>1) throw new IOException("Multiple manifest.json files; select a single dataset root");
        if(manifests.size()==1) {
            File manifest=manifests.get(0);
            JSONArray rows=new JSONObject(Io.read(manifest,8*1024*1024)).getJSONArray("samples");
            for(int i=0;i<rows.length();i++) {
                JSONObject r=rows.getJSONObject(i);File audio=Io.child(manifest.getParentFile(),r.getString("audio"));
                if(!audio.isFile()||!isAudio(audio)) throw new IOException("Missing manifest audio: "+r.getString("audio"));
                Sample s=new Sample(r.optString("id",audio.getName()),audio.getAbsolutePath());
                s.track=r.optString("track","manifest");s.sourceLanguage=r.optString("src_lang",r.optString("source_language","und"));
                s.targetLanguage=trackTargets?r.optString("tgt_lang",r.optString("target_language",target)):target;
                s.referenceTranscript=nullable(r,"ref_text");s.referenceTranslation=nullable(r,"ref_trans");
                if(s.referenceTranslation==null) s.referenceTranslation=nullable(r,"ref_translation");
                scan.samples.add(s);
            }
        } else {
            Map<File,Map<String,Map<String,String>>> tables=new HashMap<>();
            for(File f:files) if(f.getName().toLowerCase(Locale.ROOT).endsWith(".xlsx")) {
                try {
                    Map<String,Map<String,String>> table=tables.computeIfAbsent(f.getParentFile(),x->new HashMap<>());
                    for(Map<String,String> row:XlsxReader.read(f)) {
                        String id=row.getOrDefault("id","").trim().toUpperCase(Locale.ROOT);
                        if(!id.isEmpty()) {
                            if(table.containsKey(id)&&!table.get(id).equals(row)) throw new IOException("Conflicting XLSX reference ID "+id);
                            table.put(id,row);
                        }
                    }
                } catch(Exception e) { throw new IOException("Cannot read references in "+f.getName()+": "+Io.error(e),e); }
            }
            for(File audio:files) if(isAudio(audio)) {
                String id=stem(audio.getName()),upper=id.toUpperCase(Locale.ROOT);
                if(upper.startsWith("VK")) continue;
                String rel=root.toPath().relativize(audio.toPath()).toString().replace('\\','/');
                Sample s=new Sample(id,audio.getAbsolutePath());s.track="custom";s.targetLanguage=target;
                String referenceTarget=target;
                if(upper.startsWith("VM")) {
                    boolean zh=audio.getParentFile().getName().equalsIgnoreCase("Man_voice");
                    s.sourceLanguage=zh?"zh":"vi";s.track=zh?"VM-zh":"VM-vi";
                    referenceTarget=zh?"vi":"zh";if(trackTargets)s.targetLanguage=referenceTarget;
                } else if(upper.startsWith("VE")) { s.track="VE-vi";s.sourceLanguage="vi";referenceTarget="en";if(trackTargets)s.targetLanguage="en"; }
                else if(upper.startsWith("EV")) { s.track="EV-en";s.sourceLanguage="en";referenceTarget="vi";if(trackTargets)s.targetLanguage="vi"; }
                Map<String,String> row=null;
                for(File parent=audio.getParentFile();parent!=null&&inside(root,parent);parent=parent.getParentFile()) {
                    Map<String,Map<String,String>> table=tables.get(parent);
                    if(table!=null&&table.containsKey(upper)) { row=table.get(upper);break; }
                }
                if(!upper.startsWith("EV")) {
                    s.referenceTranscript=sidecar(audio,s.sourceLanguage,id,true);
                    if(s.referenceTranscript==null&&row!=null)s.referenceTranscript=column(row,s.sourceLanguage);
                    if(referenceTarget.equals(s.targetLanguage)) {
                        if(row!=null)s.referenceTranslation=column(row,s.targetLanguage);
                        if(s.referenceTranslation==null)s.referenceTranslation=sidecar(audio,s.targetLanguage,id,false);
                    } else warnings.put(rel+": target override differs from reference language; MT metrics disabled");
                }
                scan.samples.add(s);
            }
        }
        Set<String> selected=new HashSet<>(Arrays.asList(tracks==null||tracks.trim().isEmpty()?new String[0]:tracks.split(",")));
        Map<String,Integer> counts=new TreeMap<>();Set<String> identities=new HashSet<>();JSONArray unmatched=new JSONArray();
        Iterator<Sample> iterator=scan.samples.iterator();
        while(iterator.hasNext()) {
            Sample s=iterator.next();
            if(!Arrays.asList("vi","en","zh").contains(s.targetLanguage))throw new IOException("Unsupported target language: "+s.targetLanguage);
            if(!selected.isEmpty()&&!selected.contains(s.track)) { iterator.remove();continue; }
            int count=counts.getOrDefault(s.track,0);
            if(limit>0&&count>=limit) { iterator.remove();continue; }
            if(!identities.add(s.track+":"+s.id))throw new IOException("Duplicate sample ID within track: "+s.track+"/"+s.id);
            counts.put(s.track,count+1);
            if(s.referenceTranscript==null||s.referenceTranslation==null)unmatched.put(s.track+"/"+s.id);
        }
        if(scan.samples.isEmpty())throw new IOException("No supported audio after filtering");
        scan.report.put("root",root.getAbsolutePath()).put("counts",new JSONObject(counts)).put("unmatched",unmatched).put("warnings",warnings).put("pairing","id_only; no order fallback; EV has no reference");
        return scan;
    }
    private static boolean inside(File root,File f) throws IOException {
        return f.getCanonicalPath().equals(root.getCanonicalPath())||f.getCanonicalPath().startsWith(root.getCanonicalPath()+File.separator);
    }
    private static void collect(File root,File dir,List<File> out,int depth) throws IOException {
        if(depth>32)throw new IOException("Dataset directory nesting too deep");
        File[] list=dir.listFiles();if(list==null)throw new IOException("Cannot read dataset directory");
        Arrays.sort(list,Comparator.comparing(File::getName));
        for(File f:list) {
            String n=f.getName();if(n.startsWith(".")||n.equals("__MACOSX")||n.toLowerCase(Locale.ROOT).contains("vie-kor"))continue;
            if(!inside(root,f))throw new IOException("Dataset symlink escapes root");
            if(f.isDirectory())collect(root,f,out,depth+1);else out.add(f);
            if(out.size()>10000)throw new IOException("Dataset exceeds 10000 files");
        }
    }
    private static String sidecar(File audio,String lang,String id,boolean same) throws IOException {
        List<File> paths=new ArrayList<>();
        if(same)paths.add(new File(audio.getParentFile(),id+".txt"));
        String folder="vi".equals(lang)?"Vie_text":"zh".equals(lang)?"Man_text":"Eng_text";
        File parent=audio.getParentFile().getParentFile();
        if(parent!=null)paths.add(new File(parent,folder+"/"+id+".txt"));
        for(File f:paths)if(f.isFile())return Io.read(f,1024*1024).trim();return null;
    }
    private static String column(Map<String,String> row,String lang) {
        String[] names="vi".equals(lang)?new String[]{"vie_text","tiếng việt"}:"zh".equals(lang)?new String[]{"man_text"}:new String[]{"tiếng anh - final","eng_text"};
        for(String n:names)if(row.containsKey(n)&&!row.get(n).trim().isEmpty())return row.get(n).trim();return null;
    }
    private static String nullable(JSONObject o,String key) { if(o.isNull(key))return null;String s=o.optString(key,"").trim();return s.isEmpty()?null:s; }
    private static String stem(String name) { int dot=name.lastIndexOf('.');return dot<0?name:name.substring(0,dot); }
    public static boolean isAudio(File f) { return f.getName().toLowerCase(Locale.ROOT).matches(".*\\.(wav|mp3|m4a|aac|flac|ogg)$"); }
}
