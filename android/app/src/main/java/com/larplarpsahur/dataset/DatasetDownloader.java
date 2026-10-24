package com.larplarpsahur.dataset;

import android.content.Context;
import com.larplarpsahur.Io;
import java.io.*;
import java.net.*;
import java.util.*;
import java.util.zip.*;

/** Downloads only the pinned public quadata ZIP snapshot. */
public final class DatasetDownloader {
    public static final String DATASET_PAGE="https://huggingface.co/datasets/Echdipi/quadata/tree/main";
    public static final String REVISION="fed18c73b4350e07a02124df4f4c716a95e26743";
    public static final String ZIP_SHA256="cde6b58e391094c7578a7bdd37a8202db0735ccd165f91cc18c6f7ded1f8f01e";
    private static final String ZIP_URL="https://huggingface.co/datasets/Echdipi/quadata/resolve/"+REVISION+
        "/Round%203-B%E1%BB%99%20d%E1%BB%AF%20li%E1%BB%87u%20m%E1%BA%ABu-20261024T014123Z-1-001.zip?download=true";
    private static final long MAX_ZIP=4L*1024*1024*1024,MAX_EXTRACTED=6L*1024*1024*1024;
    private final File base;

    public DatasetDownloader(Context context) {
        File external=context.getExternalFilesDir(null);
        if(external==null)throw new IllegalStateException("App storage unavailable");
        base=new File(external,"datasets");
    }

    /** Fetches exactly the public Hugging Face quadata ZIP and verifies its LFS SHA-256. */
    public File downloadQuadata(Io.Progress progress,Io.Cancellation cancel) throws Exception {
        Io.mkdir(base);
        File part=new File(base,"quadata-"+REVISION+".zip.part");
        File staging=null;
        try {
            for(int attempt=0;;attempt++) {
                try { fetch(part,progress,cancel);break; }
                catch(InterruptedIOException e) { if(cancel.cancelled())throw e;if(attempt==2)throw e; }
                catch(IOException e) {
                    if(attempt==2)throw e;
                    progress.update(0,"Thử lại tải dataset Hugging Face ("+(attempt+2)+"/3): "+Io.error(e));
                }
                Io.check(cancel);Thread.sleep(1000L*(attempt+1));
            }
            try(InputStream in=new FileInputStream(part)) {
                if(in.read()!='P'||in.read()!='K')throw new IOException("Hugging Face trả về dữ liệu không phải ZIP");
            }
            progress.update(95,"Đang kiểm tra SHA-256 dataset Hugging Face…");
            if(!ZIP_SHA256.equals(Io.sha(part,cancel)))throw new IOException("SHA-256 dataset không khớp snapshot Hugging Face");
            staging=new File(base,".incoming-"+UUID.randomUUID());Io.mkdir(staging);
            extract(part,staging,cancel,progress);
            if(!hasInput(staging))throw new IOException("ZIP Hugging Face không có audio hoặc manifest.json");
            File ready=new File(base,"quadata-"+REVISION+"-"+UUID.randomUUID());Io.move(staging,ready);staging=null;
            progress.update(100,"Đã tải và xác thực dataset Hugging Face");return ready;
        } finally {
            part.delete();
            if(staging!=null)removeOwned(staging);
        }
    }

    private static HttpURLConnection open() throws Exception {
        URI current=new URI(ZIP_URL);
        for(int hop=0;hop<8;hop++) {
            if(!"https".equalsIgnoreCase(current.getScheme())||current.getHost()==null||current.getUserInfo()!=null)
                throw new IOException("Từ chối redirect dataset không an toàn");
            HttpURLConnection c=(HttpURLConnection)current.toURL().openConnection();
            c.setInstanceFollowRedirects(false);c.setConnectTimeout(20000);c.setReadTimeout(30000);
            c.setRequestProperty("Accept-Encoding","identity");
            int code=c.getResponseCode();
            if(code>=300&&code<400) {
                String location=c.getHeaderField("Location");c.disconnect();
                if(location==null)throw new IOException("Hugging Face redirect không có đích");
                current=current.resolve(location);continue;
            }
            if(code!=200) {
                c.disconnect();
                throw new IOException("Hugging Face HTTP "+code+(code==401||code==403?" (dataset không còn public)":""));
            }
            return c;
        }
        throw new IOException("Quá nhiều redirect từ Hugging Face");
    }

    private static void fetch(File part,Io.Progress progress,Io.Cancellation cancel) throws Exception {
        Io.check(cancel);HttpURLConnection c=open();
        try {
            String type=c.getContentType();
            if(type!=null&&type.toLowerCase(Locale.ROOT).contains("text/html"))
                throw new IOException("Hugging Face trả về HTML thay vì ZIP");
            long expected=c.getContentLengthLong();
            if(expected>MAX_ZIP)throw new IOException("Dataset ZIP vượt giới hạn 4 GiB");
            long done=0,last=0;
            try(InputStream in=new BufferedInputStream(c.getInputStream());FileOutputStream out=new FileOutputStream(part)) {
                byte[] buf=new byte[65536];int n;
                while((n=in.read(buf))!=-1) {
                    Io.check(cancel);done+=n;
                    if(done>MAX_ZIP||part.getParentFile().getUsableSpace()<1024*1024)
                        throw new IOException("Dataset vượt giới hạn dung lượng lưu trữ");
                    out.write(buf,0,n);
                    if(System.currentTimeMillis()-last>500) {
                        last=System.currentTimeMillis();
                        progress.update(expected>0?(int)(done*94/expected):0,"Đang tải quadata: "+done/1048576+" MiB");
                    }
                }
                out.getFD().sync();
            }
            if(expected>=0&&done!=expected)throw new IOException("Tải dataset không đầy đủ");
        } finally { c.disconnect(); }
    }

    public static void extract(File zip,File root,Io.Cancellation cancel,Io.Progress progress) throws Exception {
        Set<String> names=new HashSet<>();long total=0;int count=0;
        try(ZipInputStream in=new ZipInputStream(new BufferedInputStream(new FileInputStream(zip)))) {
            ZipEntry entry;byte[] buf=new byte[65536];
            while((entry=in.getNextEntry())!=null) {
                Io.check(cancel);String name=entry.getName().replace('\\','/');
                if(name.startsWith("/")||name.matches("^[A-Za-z]:.*"))throw new IOException("Absolute ZIP path rejected");
                File output=Io.child(root,name);
                if(!names.add(output.getCanonicalPath().toLowerCase(Locale.ROOT)))throw new IOException("Duplicate archive path");
                if(++count>10000)throw new IOException("ZIP exceeds 10000 entries");
                if(entry.isDirectory()) { Io.mkdir(output);continue; }
                Io.mkdir(output.getParentFile());
                try(OutputStream out=new BufferedOutputStream(new FileOutputStream(output))) {
                    int n;while((n=in.read(buf))!=-1) {
                        Io.check(cancel);total+=n;
                        if(total>MAX_EXTRACTED||root.getUsableSpace()<1024*1024)
                            throw new IOException("Dataset extraction exceeds storage limit");
                        out.write(buf,0,n);
                    }
                }
                progress.update(97,"Đang giải nén quadata: "+count+" files");
            }
        }
    }
    private static boolean hasInput(File dir) {
        File[] children=dir.listFiles();if(children==null)return false;
        for(File f:children)if(f.isDirectory()?hasInput(f):DatasetScanner.isAudio(f)||f.getName().equals("manifest.json"))return true;
        return false;
    }
    private static void removeOwned(File dir) {
        File[] files=dir.listFiles();if(files!=null)for(File f:files)removeOwned(f);
        dir.delete();
    }
}
