// OmniVoice — QNN HTP shared-memory (rpcmem) tensors for ORT Java (util/HtpShared).
//
// ORT 1.29 Java cannot reach a device allocator, so every NPU graph input/output was
// plain CPU memory that the QNN EP copied into its own shared buffers on each run
// (Qwen3-ASR: 56 MiB of K/V cache per decode run). Tensors allocated here come from
// the QNN plugin EP's HtpShared allocator (session option
// enable_htp_shared_memory_allocator=1), which the EP binds by memHandle, no copy.
// Headers: include/ = onnxruntime-android 1.29.0 AAR headers (MIT).
#include <android/log.h>
#include <dlfcn.h>
#include <jni.h>

#include "onnxruntime_c_api.h"

#define TAG "HtpShared"

static const OrtApi *api;
static OrtAllocator *alloc;

static int ok(OrtStatus *s, const char *what) {
    if (!s) return 1;
    __android_log_print(ANDROID_LOG_WARN, TAG, "%s: %s", what, api->GetErrorMessage(s));
    api->ReleaseStatus(s);
    return 0;
}

// The env's shared allocator for the device's host-accessible memory, else the session's
// "QnnHtpShared" allocator; NULL when neither exists (caller keeps direct buffers).
static OrtAllocator *allocator(OrtEnv *env, const OrtEpDevice *dev, const OrtSession *session) {
    if (alloc) return alloc;
    if (!api) {
        void *h = dlopen("libonnxruntime.so", RTLD_NOW);
        const OrtApiBase *(*base)(void) = h ? (const OrtApiBase *(*)(void)) dlsym(h, "OrtGetApiBase") : NULL;
        if (base) api = base()->GetApi(ORT_API_VERSION);
        if (!api) return NULL;
    }
    const OrtMemoryInfo *mi = dev ? api->EpDevice_MemoryInfo(dev, OrtDeviceMemoryType_HOST_ACCESSIBLE) : NULL;
    if (mi && ok(api->GetSharedAllocator(env, mi, &alloc), "shared allocator") && alloc) return alloc;
    alloc = NULL;
    OrtMemoryInfo *named = NULL;
    if (session && ok(api->CreateMemoryInfo("QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeDefault, &named),
                      "memory info")) {
        if (!ok(api->CreateAllocator(session, named, &alloc), "session allocator")) alloc = NULL;
        api->ReleaseMemoryInfo(named);
    }
    __android_log_print(ANDROID_LOG_INFO, TAG, "HTP shared allocator: %s",
                        alloc ? (mi ? "session (device has host-accessible memory)" : "session") : "none");
    return alloc;
}

// New tensor (OrtValue*) of `shape` / ONNX element `type` in HTP shared memory, or 0.
JNIEXPORT jlong JNICALL Java_com_omnivoice_onspeak47_util_HtpShared_create(
        JNIEnv *e, jclass c, jlong env, jlong dev, jlong session, jlongArray shape, jint type) {
    OrtAllocator *a = allocator((OrtEnv *) env, (const OrtEpDevice *) dev, (const OrtSession *) session);
    if (!a) return 0;
    jsize n = (*e)->GetArrayLength(e, shape);
    jlong *dims = (*e)->GetLongArrayElements(e, shape, NULL);
    OrtValue *v = NULL;
    int good = ok(api->CreateTensorAsOrtValue(a, (const int64_t *) dims, (size_t) n,
                                              (ONNXTensorElementDataType) type, &v), "tensor");
    (*e)->ReleaseLongArrayElements(e, shape, dims, JNI_ABORT);
    return good ? (jlong) v : 0;
}

// Direct ByteBuffer over the tensor's data (valid until the OrtValue is released).
JNIEXPORT jobject JNICALL Java_com_omnivoice_onspeak47_util_HtpShared_buffer(JNIEnv *e, jclass c, jlong value) {
    void *data = NULL;
    size_t bytes = 0;
    if (!ok(api->GetTensorMutableData((OrtValue *) value, &data), "data")
        || !ok(api->GetTensorSizeInBytes((const OrtValue *) value, &bytes), "size")) {
        return NULL;
    }
    return (*e)->NewDirectByteBuffer(e, data, (jlong) bytes);
}
