package com.omnivoice.onspeak47.util;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

/**
 * One static NPU graph with every input and output tensor allocated once over a
 * direct buffer. A run fills and reads them through {@link #buf(String)} and
 * allocates nothing: ORT writes the outputs into the pinned buffers instead of new
 * tensors that Java then copied to the heap (getShortBuffer / getFloatBuffer copy),
 * which was ~70 MB of large-object garbage per 4 s on the phone (D:\apk21.txt).
 */
public final class NpuGraph implements AutoCloseable {

    private final OrtSession session;
    private final Map<String, OnnxTensor> inputs = new HashMap<>(), outputs = new HashMap<>(), own = new HashMap<>();
    private final Map<String, ByteBuffer> buffers = new HashMap<>();

    /** {@code shared}: inputs the caller owns (K/V caches), by name; the rest are allocated here. */
    public NpuGraph(OrtSession session, Map<String, OnnxTensor> shared) throws OrtException {
        this.session = session;
        inputs.putAll(shared);
        for (Map.Entry<String, NodeInfo> e : session.getInputInfo().entrySet()) {
            if (!inputs.containsKey(e.getKey())) inputs.put(e.getKey(), allocate(e.getKey(), e.getValue()));
        }
        for (Map.Entry<String, NodeInfo> e : session.getOutputInfo().entrySet()) {
            outputs.put(e.getKey(), allocate(e.getKey(), e.getValue()));
        }
    }

    private OnnxTensor allocate(String name, NodeInfo info) throws OrtException {
        TensorInfo t = (TensorInfo) info.getInfo();
        OnnxTensor tensor = HtpShared.tensor(session, t.getShape(), t.type);   // HTP shared memory if available
        buffers.put(name, HtpShared.buffer(tensor));
        own.put(name, tensor);
        return tensor;
    }

    /** A fresh view at offset 0 of an input (fill before {@link #run}) or output (read after). */
    public ByteBuffer buf(String name) {
        return buffers.get(name).duplicate().order(ByteOrder.nativeOrder());
    }

    public void run(int which) throws OrtException {
        NpuStats.run(session, inputs, outputs, which).close();   // pinned outputs are not closed
    }

    /** Closes the tensors allocated here; the session and shared inputs stay with their owner. */
    @Override public void close() {
        for (OnnxTensor t : own.values()) t.close();
    }
}
