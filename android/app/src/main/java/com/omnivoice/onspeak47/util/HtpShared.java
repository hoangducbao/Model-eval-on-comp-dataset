package com.omnivoice.onspeak47.util;

import android.util.Log;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

/**
 * NPU graph tensors in QNN HTP shared memory (rpcmem, src/main/jni/htpshared.c), so the
 * QNN EP binds them by memHandle instead of copying CPU inputs into its own buffers every
 * run (Qwen3-ASR decode: 56 MiB of K/V cache per run; HY-MT AR1: 17 MB per token).
 * ORT 1.29 Java has no device allocator API, so the OrtValue is made natively and
 * wrapped in an OnnxTensor through ORT's package-private constructors. Anything
 * missing (no lib, no allocator, option off) gives a plain direct buffer, as before.
 */
public final class HtpShared {

    private static final String TAG = "HtpShared";
    private static final boolean LOADED = load();
    private static boolean logged;

    private HtpShared() {}

    private static boolean load() {
        try {
            System.loadLibrary("htpshared");
            return true;
        } catch (Throwable t) {   // JVM unit tests, missing .so
            return false;
        }
    }

    private static native long create(long env, long device, long session, long[] shape, int onnxType);

    private static native ByteBuffer buffer(long value);

    /** Zeroed tensor for {@code session}'s graph, in HTP shared memory when available. */
    public static OnnxTensor tensor(OrtSession session, long[] shape, OnnxJavaType type) throws OrtException {
        OnnxTensor t = LOADED && OrtSessionConfig.htpSharedIo ? shared(session, shape, type) : null;
        if (!logged) {
            logged = true;
            Log.i(TAG, "NPU graph I/O in " + (t != null ? "HTP shared memory" : "direct buffers"));
        }
        if (t != null) return t;
        long n = 1;
        for (long d : shape) n *= d;
        ByteBuffer b = ByteBuffer.allocateDirect(Math.toIntExact(n * type.size)).order(ByteOrder.nativeOrder());
        return OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), b, shape, type);
    }

    /** Native-order view of a tensor made by {@link #tensor}. */
    public static ByteBuffer buffer(OnnxTensor t) {
        return ((ByteBuffer) t.getBufferRef().get()).order(ByteOrder.nativeOrder());
    }

    private static OnnxTensor shared(OrtSession session, long[] shape, OnnxJavaType type) {
        int onnx = type == OnnxJavaType.FLOAT ? 1 : type == OnnxJavaType.FLOAT16 ? 10
                : type == OnnxJavaType.INT64 ? 7 : type == OnnxJavaType.INT32 ? 6 : -1;
        if (onnx < 0) return null;
        try {
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            long envH = field(OrtEnvironment.class, "nativeHandle").getLong(env);
            long sessH = field(OrtSession.class, "nativeHandle").getLong(session);
            long devH = 0;
            if (OrtSessionConfig.npuDevice != null) {
                Method m = OrtSessionConfig.npuDevice.getClass().getDeclaredMethod("getNativeHandle");
                m.setAccessible(true);
                devH = (long) m.invoke(OrtSessionConfig.npuDevice);
            }
            long v = create(envH, devH, sessH, shape, onnx);
            if (v == 0) return null;
            ByteBuffer b = buffer(v).order(ByteOrder.nativeOrder());
            while (b.remaining() >= 8) b.putLong(0);   // rpcmem is not zeroed; masked slots must not be NaN
            while (b.hasRemaining()) b.put((byte) 0);
            b.rewind();
            Object alloc = field(OrtEnvironment.class, "defaultAllocator").get(env);
            long allocH = field(alloc.getClass(), "handle").getLong(alloc);
            Constructor<TensorInfo> ic = TensorInfo.class.getDeclaredConstructor(
                    long[].class, OnnxJavaType.class, TensorInfo.OnnxTensorType.class);
            ic.setAccessible(true);
            TensorInfo info = ic.newInstance(shape.clone(), type, TensorInfo.OnnxTensorType.mapFromInt(onnx));
            Constructor<OnnxTensor> tc = OnnxTensor.class.getDeclaredConstructor(
                    long.class, long.class, TensorInfo.class, java.nio.Buffer.class, boolean.class);
            tc.setAccessible(true);
            return tc.newInstance(v, allocH, info, b, false);
        } catch (Throwable e) {
            Log.w(TAG, "HTP shared tensor failed, direct buffer", e);
            return null;
        }
    }

    private static Field field(Class<?> c, String name) throws NoSuchFieldException {
        Field f = c.getDeclaredField(name);
        f.setAccessible(true);
        return f;
    }
}
