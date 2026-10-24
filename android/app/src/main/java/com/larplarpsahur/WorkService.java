package com.larplarpsahur;

import android.app.*;
import android.content.*;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.*;
import android.provider.OpenableColumns;
import com.larplarpsahur.dataset.*;
import com.larplarpsahur.model.ModelRepository;
import com.larplarpsahur.pipeline.*;
import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import org.json.*;

/** Owns one operation independently of the visible Activity. Never persists credentials. */
public final class WorkService extends Service {
    public static final String INIT="models",RUN="audio",DATASET="quadata",LOCAL="local",RESUME="resume",EXPORT="export";
    private final ExecutorService worker=Executors.newSingleThreadExecutor();
    private final AtomicBoolean cancelled=new AtomicBoolean();
    private final Handler main=new Handler(Looper.getMainLooper());
    private final LocalBinder binder=new LocalBinder();
    private ModelRepository models;
    private PowerManager.WakeLock wake;
    private Observer observer;
    private volatile boolean busy;
    private volatile int percent;
    private volatile String message="Sẵn sàng",output="",input="",lastRun="";
    private volatile String asrText="",translationText="",sourceLanguage="und",translationLanguage="en",currentFile="";
    private volatile long asrMillis=-1,translationMillis=-1;
    private volatile String resumeRequest="",operation="",stage="idle";
    private volatile boolean usingTrackTargets;
    public interface Observer { void changed(State state); }
    public static final class State {
        public boolean busy,ready;public int percent;public String message,output,input,lastRun;
        public String asrText,translationText,sourceLanguage,targetLanguage,currentFile;
        public long asrMillis,translationMillis;
        public boolean canPause,canResume,pauseRequested,usingTrackTargets,exportReady;
        public String stage;
    }
    public final class LocalBinder extends Binder { public WorkService service(){return WorkService.this;} }
    @Override public IBinder onBind(Intent intent){return binder;}
    @Override public void onCreate() {
        super.onCreate();models=new ModelRepository(this);
        SharedPreferences p=getSharedPreferences("ui",MODE_PRIVATE);
        lastRun=p.getString("last_run","");output=p.getString("output","");input=p.getString("input","");
        asrText=p.getString("asr_text","");translationText=p.getString("translation_text","");
        sourceLanguage=p.getString("source_language","und");translationLanguage=p.getString("translation_language","en");
        currentFile=p.getString("current_file","");asrMillis=p.getLong("asr_millis",-1);translationMillis=p.getLong("translation_millis",-1);
        stage=p.getString("stage","idle");usingTrackTargets=p.getBoolean("using_track_targets",false);
        resumeRequest=p.getString("resume_request","");
        // Upgrade recovery for runs created before the unified pause control.
        if(resumeRequest.isEmpty()&&RunRecovery.hasCheckpoint(lastRun))
            rememberRequest(new Intent(this,WorkService.class).setAction(RESUME));
        if(!lastRun.isEmpty())message="Có kết quả đã lưu; có thể xuất hoặc tiếp tục lượt chạy gần nhất.";
        getSystemService(NotificationManager.class).createNotificationChannel(
            new NotificationChannel("work","Tải model và xử lý audio",NotificationManager.IMPORTANCE_LOW));
        wake=getSystemService(PowerManager.class).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK,"larplarpsahur:work");
    }
    public void observe(Observer value){observer=value;if(value!=null)value.changed(snapshot());}
    private State snapshot() {
        State s=new State();s.busy=busy;s.ready=models.isReady();s.percent=percent;s.message=message;
        s.output=output;s.input=input;s.lastRun=lastRun;s.asrText=asrText;s.translationText=translationText;
        s.sourceLanguage=sourceLanguage;s.targetLanguage=translationLanguage;s.currentFile=currentFile;
        s.asrMillis=asrMillis;s.translationMillis=translationMillis;s.stage=stage;s.usingTrackTargets=usingTrackTargets;
        s.pauseRequested=busy&&cancelled.get();s.canPause=busy&&!EXPORT.equals(operation)&&!cancelled.get();
        s.canResume=!busy&&!resumeRequest.isEmpty();
        s.exportReady=!lastRun.isEmpty()&&new File(lastRun,"results.json").isFile();return s;
    }
    private void publish(){main.post(()->{if(observer!=null)observer.changed(snapshot());});}
    private Notification notification() {
        PendingIntent open=PendingIntent.getActivity(this,0,new Intent(this,MainActivity.class),PendingIntent.FLAG_IMMUTABLE|PendingIntent.FLAG_UPDATE_CURRENT);
        return new Notification.Builder(this,"work").setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("larplarpsahur").setContentText(message).setContentIntent(open).setOngoing(true)
            .setProgress(100,Math.max(0,percent),percent<0).build();
    }
    public void cancel() {
        if(busy&&!EXPORT.equals(operation)){cancelled.set(true);message="Đang tạm dừng sau bước hiện tại…";publish();}
    }
    @Override public int onStartCommand(Intent intent,int flags,int startId) {
        // startForeground must run even for a duplicate start from a recreated Activity.
        startForeground(1,notification());
        if(intent==null){if(!busy){stopForeground(STOP_FOREGROUND_REMOVE);stopSelf(startId);}return START_NOT_STICKY;}
        if(busy)return START_NOT_STICKY;
        Intent job=intent;
        if(RESUME.equals(intent.getAction())&&!resumeRequest.isEmpty()) {
            try { job=Intent.parseUri(resumeRequest,0).setComponent(new ComponentName(this,WorkService.class)); }
            catch(Exception e){message="Không đọc được lượt tạm dừng: "+Io.error(e);publish();stopForeground(STOP_FOREGROUND_REMOVE);stopSelf(startId);return START_NOT_STICKY;}
        }
        final Intent request=job;
        operation=request.getAction();
        // Keep the input URIs/options if paused while importing or downloading, before a run exists.
        final boolean tracked=!EXPORT.equals(operation)&&(!INIT.equals(operation)||resumeRequest.isEmpty());
        if(tracked)rememberRequest(request);
        if(!INIT.equals(operation)&&!EXPORT.equals(operation)&&!RESUME.equals(operation)) {
            resetLivePanels();translationLanguage=request.getStringExtra("target_language");
            if(translationLanguage==null)translationLanguage="en";
            usingTrackTargets=(DATASET.equals(operation)||LOCAL.equals(operation))&&request.getBooleanExtra("track_targets",true);
        }
        busy=true;cancelled.set(false);percent=-1;message="Đang chuẩn bị…";publish();
        worker.execute(()->{
            try {
                perform(request);
                if(tracked&&!cancelled.get())clearRequest();
                if(cancelled.get())message="Đã tạm dừng. Nhấn Tiếp tục để chạy tiếp.";
            }
            catch(InterruptedIOException e){message="Đã tạm dừng. Nhấn Tiếp tục; nếu đang tải dataset, ZIP sẽ được tải lại.";}
            catch(Exception e){message="Lỗi: "+Io.error(e);android.util.Log.e("BENCH","Work failed",e);}
            finally {
                if(wake.isHeld())wake.release();
                getSharedPreferences("ui",MODE_PRIVATE).edit().putString("output",output).putString("input",input)
                    .putString("asr_text",asrText).putString("translation_text",translationText)
                    .putString("source_language",sourceLanguage).putString("translation_language",translationLanguage)
                    .putString("current_file",currentFile).putLong("asr_millis",asrMillis).putLong("translation_millis",translationMillis)
                    .putString("stage",stage).putBoolean("using_track_targets",usingTrackTargets).apply();
                main.post(()->{busy=false;stopForeground(STOP_FOREGROUND_REMOVE);publish();stopSelf();});
            }
        });
        return START_NOT_STICKY;
    }
    private void progress(int value,String text) {
        percent=value;message=cancelled.get()?"Đang tạm dừng sau bước hiện tại…":text;publish();
        if(Build.VERSION.SDK_INT<33 || checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS)==PackageManager.PERMISSION_GRANTED)
            getSystemService(NotificationManager.class).notify(1,notification());
    }
    private void perform(Intent request) throws Exception {
        wake.acquire(4*60*60*1000L);
        String action=request.getAction();
        if(EXPORT.equals(action)) {
            if(lastRun.isEmpty())throw new IOException("Chưa có kết quả");
            Uri destination=Uri.parse(request.getStringExtra("destination"));
            progress(-1,"Đang lưu ZIP kết quả…");
            try(OutputStream out=getContentResolver().openOutputStream(destination,"wt")){
                if(out==null)throw new IOException("Không mở được nơi lưu");
                new ResultExporter(new File(lastRun),null).zip(out);
            }
            progress(100,"Đã lưu ZIP kết quả.");return;
        }
        models.ensure(this::progress,cancelled::get);
        if(INIT.equals(action)){progress(100,"Model sẵn sàng. Nhấn Upfile để chọn và xử lý audio.");return;}
        if(RESUME.equals(action)) {
            if(lastRun.isEmpty())throw new IOException("Chưa có lượt chạy để tiếp tục");
            execute(new ResultExporter(new File(lastRun),null));return;
        }
        String target=request.getStringExtra("target_language");
        if(target==null)target="en";
        String mode=request.getStringExtra("mt_mode");
        if(mode==null)mode="pipeline";
        if(!Arrays.asList("pipeline","oracle","both").contains(mode))throw new IOException("mt_mode: pipeline, oracle hoặc both");
        DatasetScanner.Scan scan;
        if(DATASET.equals(action)||LOCAL.equals(action)) {
            File root;
            if(DATASET.equals(action)) {
                root=new DatasetDownloader(this).downloadQuadata(this::progress,cancelled::get);
            } else {
                String path=request.getStringExtra("dataset_root");
                if(path==null||path.isEmpty())throw new IOException("Thiếu dataset_root");
                root=new File(path);
            }
            Io.check(cancelled::get);progress(-1,"Đang ghép audio với reference theo ID…");
            scan=DatasetScanner.scan(root,target,request.getBooleanExtra("track_targets",true),request.getStringExtra("tracks"),request.getIntExtra("limit",0));
            input=(DATASET.equals(action)?"Hugging Face: Echdipi/quadata\n":"Dataset: ")+root.getName()+"\n"+scan.samples.size()+" audio";
        } else if(RUN.equals(action)) {
            ArrayList<String> uris=request.getStringArrayListExtra("uris");
            if(uris==null||uris.isEmpty())throw new IOException("Hãy chọn audio trước");
            scan=new DatasetScanner.Scan();StringBuilder names=new StringBuilder();
            File imported=new File(getExternalFilesDir(null),"inputs/"+UUID.randomUUID());Io.mkdir(imported);
            for(int i=0;i<uris.size();i++) {
                Io.check(cancelled::get);Uri uri=Uri.parse(uris.get(i));String name=displayName(uri);
                progress(i*100/uris.size(),"Đang nhập "+name);
                String ext=name.contains(".")?name.substring(name.lastIndexOf('.')).replaceAll("[^a-zA-Z0-9.]",""):"";
                if(ext.length()>12)ext="";
                File audio=new File(imported,i+ext);
                try(InputStream in=getContentResolver().openInputStream(uri);OutputStream out=new FileOutputStream(audio)) {
                    if(in==null)throw new IOException("Không đọc được "+name);
                    byte[] b=new byte[65536];long total=0;int n;
                    while((n=in.read(b))!=-1) {
                        Io.check(cancelled::get);total+=n;
                        if(total>1024L*1024*1024||imported.getUsableSpace()<64L*1024*1024)throw new IOException("Audio quá lớn hoặc hết dung lượng");
                        out.write(b,0,n);
                    }
                }
                DatasetScanner.Sample s=new DatasetScanner.Sample((i+1)+"-"+name,audio.getAbsolutePath());s.targetLanguage=target;scan.samples.add(s);
                if(names.length()>0)names.append('\n');names.append(name);
            }
            input=names.toString();scan.report.put("source","manual").put("samples",scan.samples.size()).put("pairing","none");
        } else throw new IOException("Unknown action");
        JSONArray samples=new JSONArray();for(DatasetScanner.Sample s:scan.samples)samples.put(s.json());
        File results=new File(getExternalFilesDir(null),"results");Io.mkdir(results);
        File directory=new File(results,System.currentTimeMillis()+"-"+UUID.randomUUID().toString().substring(0,8));
        // Explicit output overrides must stay inside app-owned results storage.
        String outDir=request.getStringExtra("out_dir");
        if(outDir!=null&&!outDir.isEmpty()) {
            File chosen=new File(outDir).getCanonicalFile();
            if(!chosen.getPath().startsWith(results.getCanonicalPath()+File.separator)||chosen.exists())
                throw new IOException("out_dir phải là thư mục mới trong "+results);
            directory=chosen;
        }
        JSONObject config=new JSONObject().put("samples",samples).put("mt_mode",mode).put("target_language",target)
            .put("source_kind",action).put("track_targets",usingTrackTargets)
            .put("created_at_ms",System.currentTimeMillis()).put("app_version",BuildConfig.VERSION_NAME)
            .put("device",Build.MODEL).put("soc",Build.VERSION.SDK_INT>=31?Build.SOC_MODEL:"unknown");
        if(DATASET.equals(action)) config.put("dataset_page",DatasetDownloader.DATASET_PAGE).put("dataset_revision",DatasetDownloader.REVISION);
        resetLivePanels();
        ResultExporter store=new ResultExporter(directory,config);
        Io.atomic(new File(directory,"dataset_scan.json"),scan.report.toString(2));
        lastRun=directory.getAbsolutePath();
        getSharedPreferences("ui",MODE_PRIVATE).edit().putString("last_run",lastRun).putString("input",input).commit();
        rememberRequest(new Intent(this,WorkService.class).setAction(RESUME));
        output="";publish();execute(store);
    }
    private void execute(ResultExporter store) throws Exception {
        usingTrackTargets=store.run.optBoolean("track_targets",store.run.has("dataset_page"));
        Io.check(cancelled::get);
        String summary=new PipelineRunner(this).run(store,new PipelineRunner.Listener(){
            public void update(int percent,String message){progress(percent,message);}
            public void active(JSONObject r,String stage) {
                WorkService.this.stage=stage;
                currentFile=displayLabel(r);asrMillis=r.has("asr_ms")?r.optLong("asr_ms",-1):-1;
                translationMillis=r.has("mt_ms")?r.optLong("mt_ms",-1):-1;
                if("asr".equals(stage)) {
                    sourceLanguage="und";asrText="";translationText="";translationLanguage=r.optString("tgt_lang",translationLanguage);
                } else {
                    sourceLanguage=r.optString("src_lang","und");asrText=r.optString("asr_hyp","");
                    translationLanguage=r.optString("tgt_lang",translationLanguage);translationText="";
                }
                publish();
            }
            public void result(JSONObject r) {
                stage=!r.isNull("error")?"error":r.optBoolean("mt_done")?"done":"waiting_mt";
                currentFile=displayLabel(r);sourceLanguage=r.optString("src_lang","und");
                translationLanguage=r.optString("tgt_lang",translationLanguage);asrMillis=r.has("asr_ms")?r.optLong("asr_ms",-1):-1;
                translationMillis=r.has("mt_ms")?r.optLong("mt_ms",-1):-1;asrText=r.optString("asr_hyp","");
                translationText=r.isNull("mt_hyp")?"":r.optString("mt_hyp","");
                if(!r.isNull("error")) {
                    String error=r.optString("error");
                    if(asrText.isEmpty())asrText="Không thể nhận dạng audio.\n"+error;
                    else translationText="Không thể dịch.\n"+error;
                }
                String text="ASR ("+sourceLanguage+"):\n"+asrText+"\n\nDịch ("+translationLanguage+"):\n"+translationText;
                output=text.length()>24000?text.substring(0,24000)+"\n[Kết quả đầy đủ trong ZIP]":text;publish();
            }
        },cancelled::get);
        progress(100,summary+"\n"+store.directory);
    }
    private void resetLivePanels() {
        asrText="";translationText="";sourceLanguage="und";currentFile="";asrMillis=-1;translationMillis=-1;stage="preparing";
    }
    private void rememberRequest(Intent request) {
        resumeRequest=request.toUri(0);
        getSharedPreferences("ui",MODE_PRIVATE).edit().putString("resume_request",resumeRequest).commit();
    }
    private void clearRequest() {
        resumeRequest="";
        getSharedPreferences("ui",MODE_PRIVATE).edit().remove("resume_request").commit();
    }
    private static String displayLabel(JSONObject result) {
        String id=result.optString("id","");
        return id.isEmpty()?"audio":id;
    }
    public String displayName(Uri uri) {
        try(Cursor c=getContentResolver().query(uri,new String[]{OpenableColumns.DISPLAY_NAME},null,null,null)){
            if(c!=null&&c.moveToFirst()&&!c.isNull(0))return c.getString(0);
        } catch(Exception ignored){}
        return uri.getLastPathSegment()==null?"audio":uri.getLastPathSegment();
    }
    @Override public void onDestroy(){observer=null;worker.shutdown();if(wake!=null&&wake.isHeld())wake.release();super.onDestroy();}
}
