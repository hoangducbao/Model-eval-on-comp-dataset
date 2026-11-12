package com.larplarpsahur.pipeline;

import android.content.Context;
import android.net.Uri;
import android.os.SystemClock;
import android.util.Log;
import com.larplarpsahur.Io;
import com.omnivoice.onspeak47.asr.QwenFileRecognizer;
import com.omnivoice.onspeak47.pipeline.SegmentTranslator;
import java.util.*;
import org.json.*;

/**
 * Two phases keep the two large NPU model sets out of memory at the same time.
 * Translation runs OnSpeak47's own commit path (SegmentTranslator): script-based source
 * language, copy of rows already in the target language, written-form numbers, 200-char
 * segments with the commit token budget and forced-prefix context between segments.
 */
public final class PipelineRunner {
    public interface Listener extends Io.Progress {
        void active(JSONObject result,String stage);
        void result(JSONObject result);
    }
    private final Context context;
    public PipelineRunner(Context context) { this.context=context.getApplicationContext(); }
    public String run(ResultExporter store,Listener listener,Io.Cancellation cancel) throws Exception {
        int count=store.samples.length();String mode=store.run.optString("mt_mode","pipeline");
        String outcome="complete",failure=null;QwenFileRecognizer asr=null;SegmentTranslator mt=null;
        try {
            for(int i=0;i<count;i++) {
                Io.check(cancel);JSONObject r=store.record(i);
                if(r.optBoolean("asr_done"))continue;
                if(asr==null) {
                    listener.update(0,"Đang nạp Qwen3-ASR…");long t=SystemClock.elapsedRealtime();asr=QwenFileRecognizer.open(context);
                    store.run.put("asr_load_ms",SystemClock.elapsedRealtime()-t);
                }
                listener.active(r,"asr");
                listener.update(i*50/count,"ASR "+(i+1)+"/"+count+": "+r.optString("id"));
                try {
                    String path=r.getString("audio");Uri uri=path.startsWith("content:")?Uri.parse(path):Uri.fromFile(new java.io.File(path));
                    long decodeStart=SystemClock.elapsedRealtime();
                    AudioDecoder.DecodedAudio audio=AudioDecoder.decode(context,uri,cancel);
                    r.put("decode_ms",SystemClock.elapsedRealtime()-decodeStart).put("duration_s",audio.durationSeconds);
                    long start=SystemClock.elapsedRealtime();
                    QwenFileRecognizer.Recognition result=asr.transcribe(audio.pcm16KhzMono,cancel);
                    r.put("asr_ms",SystemClock.elapsedRealtime()-start).put("asr_hyp",result.text).put("asr_routed",result.routedText).put("reference_language",r.optString("src_lang","und"))
                        .put("src_lang",result.sourceLanguage).put("asr_done",true).put("status","asr_done").put("error",JSONObject.NULL);
                } catch(Exception e) { Io.check(cancel);r.put("status","error").put("error",Io.error(e)); }
                store.save(i,r);listener.result(r);
            }
            if(asr!=null) { asr.close();asr=null; }
            for(int i=0;i<count;i++) {
                Io.check(cancel);JSONObject r=store.record(i);
                if(!r.optBoolean("asr_done")||r.optBoolean("mt_done"))continue;
                listener.active(r,"translation");
                listener.update(50+i*50/count,"Dịch "+(i+1)+"/"+count+": "+r.optString("id"));
                try {
                    if(mt==null) {
                        listener.update(50+i*50/count,"Đang nạp HY-MT2…");
                        long t=SystemClock.elapsedRealtime();mt=new SegmentTranslator(context);
                        store.run.put("mt_load_ms",SystemClock.elapsedRealtime()-t);
                    }
                    long start=SystemClock.elapsedRealtime();
                    String src=r.optString("src_lang"),target=r.getString("tgt_lang");
                    if(!"oracle".equals(mode)) {
                        // Runs saved before 1.1.0 have no routed transcript; rebuild it from the tag.
                        String routed=r.has("asr_routed")?r.optString("asr_routed"):QwenFileRecognizer.routed(src,r.optString("asr_hyp"));
                        SegmentTranslator.Result t=translate(mt,routed,src,target,cancel);
                        r.put("mt_hyp",t.text).put("mt_src_lang",t.sourceLanguage).put("mt_segments",t.segments).put("mt_copied",t.copied);
                    }
                    if(!"pipeline".equals(mode)) {
                        if(!r.isNull("ref_text")) {
                            String known=r.optString("reference_language",src);
                            r.put("mt_oracle_hyp",translate(mt,QwenFileRecognizer.routed(known,r.getString("ref_text")),known,target,cancel).text);
                        } else r.put("oracle_note","No reference transcript");
                    }
                    r.put("mt_ms",SystemClock.elapsedRealtime()-start).put("mt_done",true).put("status","ok").put("error",JSONObject.NULL);
                } catch(Exception e) { Io.check(cancel);r.put("status","error").put("error",Io.error(e)); }
                store.save(i,r);listener.result(r);
            }
        } catch(java.io.InterruptedIOException e) { outcome="cancelled";failure=e.getMessage(); }
        catch(Exception e) { outcome="failed";failure=Io.error(e); }
        finally { if(asr!=null)asr.close();if(mt!=null)mt.close(); }
        String summary=store.export(outcome,failure);
        Log.i("BENCH",("complete".equals(outcome)?"BENCH_DONE ":"BENCH_"+outcome.toUpperCase(Locale.ROOT)+" ")+store.directory);
        Log.i("BENCH",summary);
        if("failed".equals(outcome))throw new java.io.IOException(failure);
        return ("cancelled".equals(outcome)?"Đã dừng. Có thể tiếp tục.\n":"Đã hoàn tất.\n")+summary;
    }
    private SegmentTranslator.Result translate(SegmentTranslator mt,String text,String src,String tgt,Io.Cancellation cancel) throws Exception {
        SegmentTranslator.Result result=mt.translate(text,src,tgt,cancel::cancelled);
        // A paused decode returns partial text: never store it as this sample's translation.
        Io.check(cancel);
        if(result.text.trim().isEmpty()&&text!=null&&!text.trim().isEmpty())throw new java.io.IOException("HY-MT returned no translation");
        return result;
    }
}
