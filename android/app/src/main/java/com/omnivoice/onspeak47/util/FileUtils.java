/*
 * OmniVoice — File Utilities
 */

package com.omnivoice.onspeak47.util;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;


/**
 * File helper methods for copying assets to internal storage.
 */
public final class FileUtils {

    private static final String TAG = "FileUtils";
    private static final int BUFFER_SIZE = 8192;

    private FileUtils() {}

    /**
     * Copy a file from the app's assets directory to the specified directory.
     * Also automatically copies a matching .onnx_data file if it exists.
     *
     * @param context   Application context
     * @param assetName Filename in the assets directory
     * @param targetDir Target directory
     * @return Path to the copied file
     */
    public static String copyAssetToDir(Context context, String assetName, File targetDir) {
        copySingleAssetToDir(context, assetName, targetDir);

        // If it's an ONNX file, check for external data (.onnx_data)
        if (assetName.endsWith(".onnx")) {
            String dataFile = assetName + "_data";
            try {
                // Check if the data file exists in assets
                context.getAssets().open(dataFile).close();
                copySingleAssetToDir(context, dataFile, targetDir);
            } catch (IOException ignored) {
                // .onnx_data doesn't exist for this model, which is fine
            }
        }

        return new File(targetDir, assetName).getAbsolutePath();
    }

    private static void copySingleAssetToDir(Context context, String assetName, File targetDir) {
        File outFile = new File(targetDir, assetName);

        // FAST PATH: reuse an existing copy — but only if it's complete. An
        // earlier interrupted write leaves a truncated file on disk whose
        // createSession() would then fail (or silently corrupt) forever.
        if (outFile.exists()) {
            long expected = assetSize(context, assetName);
            if (expected >= 0 && outFile.length() != expected) {
                Log.w(TAG, "Discarding incomplete copy of " + assetName
                        + " (on disk " + outFile.length() + " B vs asset " + expected + " B)");
                outFile.delete();
            } else {
                Log.d(TAG, "Asset already exists: " + outFile.getAbsolutePath());
                return;
            }
        }

        // Ensure parent directories exist
        if (!targetDir.exists()) {
            targetDir.mkdirs();
        }

        try (InputStream in = context.getAssets().open(assetName);
             FileOutputStream out = new FileOutputStream(outFile)) {

            byte[] buffer = new byte[BUFFER_SIZE];
            int bytesRead;
            long totalBytes = 0;

            while ((bytesRead = in.read(buffer)) != -1) {
                out.write(buffer, 0, bytesRead);
                totalBytes += bytesRead;
            }

            out.flush();
            Log.i(TAG, String.format("Copied asset '%s' → %s (%.1f MB)",
                    assetName, outFile.getAbsolutePath(), totalBytes / 1e6));

        } catch (IOException e) {
            Log.e(TAG, "Failed to copy asset: " + assetName, e);
            if (outFile.exists()) outFile.delete();
            // Propagate — a caller that then passes the returned path into
            // OrtEnvironment.createSession must not be handed a missing file.
            throw new RuntimeException("Failed to copy asset to storage: " + assetName, e);
        }
    }

    /**
     * Size in bytes of an asset (may exceed 2 GiB; NOT limited by
     * AssetFileDescriptor). Returns -1 if the asset cannot be opened.
     */
    private static long assetSize(Context context, String assetName) {
        try (android.content.res.AssetFileDescriptor afd =
                     context.getAssets().openFd(assetName)) {
            return afd.getLength();
        } catch (IOException e) {
            // openFd requires the asset to be stored uncompressed (it is:
            // build.gradle noCompress covers onnx/json). If it's compressed,
            // fall back to the stream length.
            try (InputStream in = context.getAssets().open(assetName)) {
                return in.available();
            } catch (IOException e2) {
                return -1;
            }
        }
    }

    /**
     * Copy a file from the app's assets directory to internal storage.
     * Deprecated: Use copyAssetToDir instead to avoid filling up internal storage.
     */
    public static String copyAssetToInternal(Context context, String assetName) {
        return copyAssetToDir(context, assetName, context.getFilesDir());
    }

    /**
     * Check if a model file exists in internal storage.
     */
    public static boolean modelExists(Context context, String modelName) {
        return new File(context.getFilesDir(), modelName).exists();
    }

    /**
     * Get the size of a file in megabytes.
     */
    public static float getFileSizeMB(File file) {
        if (!file.exists()) return 0;
        return file.length() / (1024f * 1024f);
    }

    /**
     * Delete a file from internal storage.
     */
    public static boolean deleteInternalFile(Context context, String fileName) {
        File file = new File(context.getFilesDir(), fileName);
        if (file.exists()) {
            return file.delete();
        }
        return false;
    }

    /**
     * Read a small text sidecar file (e.g. an asset ".rev" marker).
     * Returns "" when absent or unreadable.
     */
    public static String readTextFile(File file) {
        try {
            byte[] data = new byte[(int) file.length()];
            try (java.io.FileInputStream fis = new java.io.FileInputStream(file)) {
                int off = 0;
                while (off < data.length) {
                    int n = fis.read(data, off, data.length - off);
                    if (n < 0) break;
                    off += n;
                }
            }
            return new String(data, java.nio.charset.StandardCharsets.UTF_8).trim();
        } catch (Exception e) {
            return "";
        }
    }

    /** Best-effort write of a small text sidecar file. */
    public static void writeTextFile(File file, String content) {
        try {
            try (java.io.FileOutputStream fos = new java.io.FileOutputStream(file)) {
                fos.write(content.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            }
        } catch (Exception e) {
            Log.w(TAG, "Could not write " + file, e);
        }
    }

    /**
     * Rename a downloaded {@code name.part} to {@code name} only if its SHA-256
     * matches the {@code name} line of {@code sums} (sha256sum format,
     * "hex *name"). A mismatching or unlisted file is deleted.
     */
    public static void installVerified(File part, File sums) throws IOException {
        String name = part.getName().replaceFirst("\\.part$", "");
        String want = null;
        for (String line : java.nio.file.Files.readAllLines(sums.toPath())) {
            String[] f = line.trim().split("\\s+");
            if (f.length == 2 && f[1].replaceFirst("^\\*", "").equals(name)) want = f[0].toLowerCase();
        }
        java.security.MessageDigest md;
        try {
            md = java.security.MessageDigest.getInstance("SHA-256");
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IOException(e);
        }
        byte[] buf = new byte[1 << 20];
        try (InputStream in = new java.io.FileInputStream(part)) {
            int n;
            while ((n = in.read(buf)) > 0) md.update(buf, 0, n);
        }
        String got = String.format("%064x", new java.math.BigInteger(1, md.digest()));
        if (!got.equals(want)) {
            part.delete();
            throw new IOException("sha256 mismatch: " + name);
        }
        File dst = new File(part.getParentFile(), name);
        if ((dst.exists() && !dst.delete()) || !part.renameTo(dst)) {
            throw new IOException("rename " + part + " -> " + dst);
        }
    }
}
