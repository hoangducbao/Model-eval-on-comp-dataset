package com.larplarpsahur;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.Locale;
import org.json.JSONObject;

public final class Io {
    private Io() {}
    public interface Progress { void update(int percent, String message); }
    public interface Cancellation { boolean cancelled(); }
    public static void check(Cancellation cancel) throws InterruptedIOException {
        if (cancel.cancelled() || Thread.currentThread().isInterrupted()) throw new InterruptedIOException("Cancelled");
    }
    public static void mkdir(File dir) throws IOException {
        if (!dir.mkdirs() && !dir.isDirectory()) throw new IOException("Cannot create " + dir);
    }
    public static String read(File file, long maximum) throws IOException {
        if (file.length() > maximum) throw new IOException("File too large: " + file.getName());
        try (InputStream in = new FileInputStream(file)) { return read(in, maximum); }
    }
    public static String read(InputStream in, long maximum) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[8192]; int n;
        while ((n = in.read(buf)) != -1) {
            if ((long)out.size() + n > maximum) throw new IOException("Response too large");
            out.write(buf, 0, n);
        }
        return new String(out.toByteArray(), StandardCharsets.UTF_8);
    }
    public static void atomic(File file, String text) throws IOException {
        mkdir(file.getParentFile());
        File part = new File(file.getPath() + ".tmp");
        try (FileOutputStream out = new FileOutputStream(part)) {
            out.write(text.getBytes(StandardCharsets.UTF_8)); out.getFD().sync();
        }
        move(part, file);
    }
    public static void move(File from, File to) throws IOException {
        try { Files.move(from.toPath(), to.toPath(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE); }
        catch (AtomicMoveNotSupportedException e) { Files.move(from.toPath(), to.toPath(), StandardCopyOption.REPLACE_EXISTING); }
    }
    public static String sha(File file, Cancellation cancel) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new BufferedInputStream(new FileInputStream(file))) {
            byte[] buf = new byte[1024 * 1024]; int n;
            while ((n = in.read(buf)) != -1) { check(cancel); md.update(buf, 0, n); }
        }
        StringBuilder out = new StringBuilder();
        for (byte b : md.digest()) out.append(String.format(Locale.ROOT, "%02x", b & 255));
        return out.toString();
    }
    public static File child(File root, String name) throws IOException {
        File f = new File(root, name).getCanonicalFile();
        if (!f.getPath().startsWith(root.getCanonicalPath() + File.separator)) throw new IOException("Path escapes dataset: " + name);
        return f;
    }
    public static String error(Throwable e) { return e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage(); }
    public static String csv(Object value) { return "\"" + (value == null || value == JSONObject.NULL ? "" : value.toString()).replace("\"", "\"\"") + "\""; }
}
