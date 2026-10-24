package com.larplarpsahur.model;

import android.app.DownloadManager;
import android.content.*;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import com.larplarpsahur.BuildConfig;
import com.larplarpsahur.Io;
import java.io.*;
import java.util.*;

/** Single worker only; Android retains downloads across process death. */
public final class ModelRepository {
    private final File root;
    private final DownloadManager dm;
    private final SharedPreferences prefs;
    private final Context context;
    private final List<Asset> assets = new ArrayList<>();
    private static final class Asset {
        final String url,path,hash; final long size;
        Asset(String url,String path,long size,String hash) { this.url=url;this.path=path;this.size=size;this.hash=hash; }
    }
    public ModelRepository(Context context) {
        this.context=context.getApplicationContext(); root=context.getExternalFilesDir(null);
        if(root==null) throw new IllegalStateException("App storage unavailable");
        dm=context.getSystemService(DownloadManager.class);
        prefs=context.getSharedPreferences("verified_models_v2",Context.MODE_PRIVATE);
        add(BuildConfig.QWEN_MODEL_BASE,"qwen3asr_npu","qasr_enc.sm8850.ctx.onnx",234574674L,"a414a62d3b2ced4db81a1a6cdbb182249689dc1a1edc326d41f1cb36fbc078d7");
        add(BuildConfig.QWEN_MODEL_BASE,"qwen3asr_npu","qasr_ar64.sm8850.ctx.onnx",3729,"abcced0779aa8938f87a959630ed0670d65e0fa9c36b9562c6bdba40782434f6");
        add(BuildConfig.QWEN_MODEL_BASE,"qwen3asr_npu","qasr_ar1.sm8850.ctx.onnx",3727,"4396c7dcb6ce2510f0fa4f511a1ed5f576f52b20d9f0a308cec5a9da7eb1e7fe");
        add(BuildConfig.QWEN_MODEL_BASE,"qwen3asr_npu","qasr_ar64.sm8850.ctx_qnn.bin",570302464L,"a34d71d320e89639bce54e4a15cef5e9eb10bb975627c8bc34f7c9c6e51b192c");
        add(BuildConfig.QWEN_MODEL_BASE,"qwen3asr_npu","qasr_assets.bin",90964736,"412c19c94232ad166e1c9ea8e2fd1aa85671dee14ddf1489bc10a8c5d871670d");
        add(BuildConfig.HYMT_MODEL_BASE,"hymt_npu","hymt_ar64.sm8850.ctx.onnx",9271,"d9db60e4bb26b8f28630e32b7ef170db12354bd9b0109e69cc1b9018808e58d3");
        add(BuildConfig.HYMT_MODEL_BASE,"hymt_npu","hymt_ar64_256.sm8850.ctx.onnx",9271,"c0692761045725201cc54eec88e348a5b0058b9c4d9f916cc928dbf2bc9a84c0");
        add(BuildConfig.HYMT_MODEL_BASE,"hymt_npu","hymt_ar1.sm8850.ctx.onnx",9271,"116525098c824ce4364e06021bb56650ba784e87191b23f101ad26373deb4244");
        add(BuildConfig.HYMT_MODEL_BASE,"hymt_npu","hymt_ar64.sm8850.ctx_qnn.bin",938835968L,"c3d7bc55145d2ddc61e41e97f3f98abedf1ca3e1ea65159f9dc36dc64b732fc8");
        assets.add(new Asset(BuildConfig.TOKENIZER_URL,"hymt_npu/tokenizer.json",9527287,BuildConfig.TOKENIZER_SHA256));
    }
    private void add(String base,String dir,String name,long size,String hash) {
        assets.add(new Asset(base.replaceAll("/+$","")+"/"+name,dir+"/"+name,size,hash));
    }
    private String stamp(Asset a,File f) { return a.hash+":"+f.length()+":"+f.lastModified(); }
    private static boolean supported() { return Build.VERSION.SDK_INT>=31 && "sm8850".equalsIgnoreCase(Build.SOC_MODEL); }
    public boolean isReady() {
        if(!supported()) return false;
        for(Asset a:assets) {
            File f=new File(root,a.path);
            if(!f.isFile()||f.length()!=a.size||!stamp(a,f).equals(prefs.getString("ok:"+a.path,""))) return false;
        }
        return new File(root,"qwen3asr_npu/REV").isFile()&&new File(root,"hymt_npu/REV").isFile();
    }
    public void ensure(Io.Progress progress,Io.Cancellation cancel) throws Exception {
        if(!supported()) throw new IOException("Published Qwen/HY-MT models require Snapdragon SM8850 (Android 12+). Device: "+(Build.VERSION.SDK_INT>=31?Build.SOC_MODEL:Build.MODEL));
        for(int i=0;i<assets.size();i++) {
            Io.check(cancel); Asset a=assets.get(i); File f=new File(root,a.path);
            if(f.isFile()&&f.length()==a.size&&stamp(a,f).equals(prefs.getString("ok:"+a.path,""))) continue;
            if(f.isFile()&&f.length()==a.size&&a.hash.equals(Io.sha(f,cancel))) {
                prefs.edit().putString("ok:"+a.path,stamp(a,f)).commit();continue;
            }
            Io.mkdir(f.getParentFile()); File part=new File(f+".part"); String key="id:"+a.path;
            long id=prefs.getLong(key,-1);
            if(id!=-1&&!exists(id)) { prefs.edit().remove(key).commit();id=-1; }
            if(id==-1) {
                if(root.getUsableSpace()<a.size+64L*1024*1024) throw new IOException("Insufficient free storage for "+f.getName()+". Models need about 1.85 GB plus dataset space.");
                if(part.exists()&&!part.delete()) throw new IOException("Cannot remove incomplete download");
                if(!a.url.startsWith("https://")) throw new IOException("Model URLs must use HTTPS");
                id=dm.enqueue(new DownloadManager.Request(Uri.parse(a.url)).setTitle("larplarpsahur: "+f.getName())
                    .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                    .setAllowedOverRoaming(false).setDestinationInExternalFilesDir(context,null,a.path+".part"));
                if(!prefs.edit().putLong(key,id).commit()) { dm.remove(id);throw new IOException("Cannot persist download state"); }
            }
            for(;;) {
                Io.check(cancel);
                try(Cursor c=dm.query(new DownloadManager.Query().setFilterById(id))) {
                    if(c==null||!c.moveToFirst()) { prefs.edit().remove(key).commit();throw new IOException("Download removed. Tap Retry."); }
                    int status=c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS));
                    if(status==DownloadManager.STATUS_FAILED) {
                        int reason=c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON));
                        dm.remove(id);prefs.edit().remove(key).commit();
                        throw new IOException("Download failed (Android reason "+reason+"): "+f.getName()+". Retry retains verified files.");
                    }
                    if(status==DownloadManager.STATUS_SUCCESSFUL) break;
                    long done=c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR));
                    int pc=(int)Math.min(100,done*100/a.size);
                    progress.update((i*100+pc)/assets.size(),(status==DownloadManager.STATUS_PAUSED?"Waiting for network: ":"Downloading: ")+f.getName()+" "+pc+"%");
                }
                Thread.sleep(1000);
            }
            progress.update(i*100/assets.size(),"Verifying "+f.getName());
            if(part.length()!=a.size||!a.hash.equals(Io.sha(part,cancel))) {
                dm.remove(id);prefs.edit().remove(key).commit();part.delete();
                throw new IOException("Size or SHA-256 mismatch: "+f.getName()+". Tap Retry.");
            }
            Io.move(part,f);prefs.edit().remove(key).putString("ok:"+a.path,stamp(a,f)).commit();
        }
        Io.atomic(new File(root,"qwen3asr_npu/REV"),"1");Io.atomic(new File(root,"hymt_npu/REV"),"8");
        progress.update(100,"Models verified and ready (offline)");
    }
    private boolean exists(long id) { try(Cursor c=dm.query(new DownloadManager.Query().setFilterById(id))) { return c!=null&&c.moveToFirst(); } }
}
