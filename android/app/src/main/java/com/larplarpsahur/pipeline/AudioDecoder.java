package com.larplarpsahur.pipeline;

import android.content.Context;
import android.media.AudioFormat;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;
import android.os.SystemClock;
import com.larplarpsahur.Io;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Decodes an Android-supported audio Uri to 16 kHz mono float PCM for Qwen3-ASR. */
public final class AudioDecoder {
    public static final int TARGET_RATE_HZ = 16_000;
    private static final long TIMEOUT_US = 10_000;
    private static final int MAX_OUTPUT_FRAMES = TARGET_RATE_HZ * 60 * 10;

    private AudioDecoder() {
    }

    public static DecodedAudio decode(Context context, Uri uri, Io.Cancellation cancel) throws IOException {
        MediaExtractor extractor = new MediaExtractor();
        MediaCodec codec = null;
        try {
            extractor.setDataSource(context, uri, null);
            int track = findAudioTrack(extractor);
            if (track < 0) throw new IOException("No audio track found");
            extractor.selectTrack(track);
            MediaFormat inputFormat = extractor.getTrackFormat(track);
            String mime = inputFormat.getString(MediaFormat.KEY_MIME);
            if (mime == null) throw new IOException("Audio MIME type is missing");
            if ("audio/raw".equals(mime)) {
                int rate=inputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE),channels=inputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
                int encoding=inputFormat.containsKey(MediaFormat.KEY_PCM_ENCODING)?inputFormat.getInteger(MediaFormat.KEY_PCM_ENCODING):AudioFormat.ENCODING_PCM_16BIT;
                PcmAccumulator raw=new PcmAccumulator();ByteBuffer buffer=ByteBuffer.allocate(1024*1024).order(ByteOrder.LITTLE_ENDIAN);
                for(;;) { Io.check(cancel);buffer.clear();int n=extractor.readSampleData(buffer,0);if(n<0)break;buffer.position(0);buffer.limit(n);raw.append(buffer,encoding,channels);extractor.advance(); }
                float[] output=resample(raw.toArray(),rate,TARGET_RATE_HZ);
                if(output.length==0||output.length>MAX_OUTPUT_FRAMES)throw new IOException("Audio is empty or exceeds 10 minutes");
                return new DecodedAudio(output,output.length/(double)TARGET_RATE_HZ);
            }
            codec = MediaCodec.createDecoderByType(mime);
            codec.configure(inputFormat, null, null, 0);
            codec.start();

            int outputRate = inputFormat.containsKey(MediaFormat.KEY_SAMPLE_RATE)
                    ? inputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE) : TARGET_RATE_HZ;
            int outputChannels = inputFormat.containsKey(MediaFormat.KEY_CHANNEL_COUNT)
                    ? inputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT) : 1;
            int outputEncoding = AudioFormat.ENCODING_PCM_16BIT;
            PcmAccumulator pcm = new PcmAccumulator();
            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            boolean inputEnded = false;
            boolean outputEnded = false;
            long lastOutput=SystemClock.elapsedRealtime();
            while (!outputEnded) {
                Io.check(cancel);
                if(SystemClock.elapsedRealtime()-lastOutput>60000)throw new IOException("Audio decoder stalled for 60 seconds");
                if (!inputEnded) {
                    int inputIndex = codec.dequeueInputBuffer(TIMEOUT_US);
                    if (inputIndex >= 0) {
                        ByteBuffer input = codec.getInputBuffer(inputIndex);
                        if (input == null) throw new IOException("Decoder input buffer unavailable");
                        input.clear();
                        int size = extractor.readSampleData(input, 0);
                        if (size < 0) {
                            codec.queueInputBuffer(inputIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                            inputEnded = true;
                        } else {
                            codec.queueInputBuffer(inputIndex, 0, size, extractor.getSampleTime(), 0);
                            extractor.advance();
                        }
                    }
                }
                int outputIndex = codec.dequeueOutputBuffer(info, TIMEOUT_US);
                if (outputIndex >= 0) {
                    lastOutput=SystemClock.elapsedRealtime();
                    if (info.size > 0) {
                        ByteBuffer buffer = codec.getOutputBuffer(outputIndex);
                        if (buffer == null) throw new IOException("Decoder output buffer unavailable");
                        buffer.position(info.offset);
                        buffer.limit(info.offset + info.size);
                        pcm.append(buffer.slice().order(ByteOrder.LITTLE_ENDIAN), outputEncoding, outputChannels);
                    }
                    outputEnded = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
                    codec.releaseOutputBuffer(outputIndex, false);
                } else if (outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    MediaFormat output = codec.getOutputFormat();
                    outputRate = output.containsKey(MediaFormat.KEY_SAMPLE_RATE)
                            ? output.getInteger(MediaFormat.KEY_SAMPLE_RATE) : outputRate;
                    outputChannels = output.containsKey(MediaFormat.KEY_CHANNEL_COUNT)
                            ? output.getInteger(MediaFormat.KEY_CHANNEL_COUNT) : outputChannels;
                    outputEncoding = output.containsKey(MediaFormat.KEY_PCM_ENCODING)
                            ? output.getInteger(MediaFormat.KEY_PCM_ENCODING) : AudioFormat.ENCODING_PCM_16BIT;
                }
            }
            float[] mono = pcm.toArray();
            if (mono.length == 0) throw new IOException("Decoded audio is empty");
            float[] target = resample(mono, outputRate, TARGET_RATE_HZ);
            if (target.length > MAX_OUTPUT_FRAMES) {
                throw new IOException("Audio exceeds the 10 minute processing limit");
            }
            return new DecodedAudio(target, target.length / (double) TARGET_RATE_HZ);
        } catch (IllegalArgumentException e) {
            throw new IOException("Cannot decode this audio file", e);
        } finally {
            if (codec != null) {
                try { codec.stop(); } catch (Exception ignored) { }
                codec.release();
            }
            extractor.release();
        }
    }

    private static int findAudioTrack(MediaExtractor extractor) {
        for (int i = 0; i < extractor.getTrackCount(); i++) {
            MediaFormat format = extractor.getTrackFormat(i);
            String mime = format.getString(MediaFormat.KEY_MIME);
            if (mime != null && mime.startsWith("audio/")) return i;
        }
        return -1;
    }

    private static float[] resample(float[] input, int fromRate, int toRate) {
        if (fromRate <= 0 || fromRate == toRate) return input;
        int count = (int) Math.max(1, Math.round(input.length * (double) toRate / fromRate));
        float[] output = new float[count];
        double step = fromRate / (double) toRate;
        for (int i = 0; i < count; i++) {
            double source = i * step;
            int left = Math.min(input.length - 1, (int) source);
            int right = Math.min(input.length - 1, left + 1);
            double fraction = source - left;
            output[i] = (float) (input[left] + (input[right] - input[left]) * fraction);
        }
        return output;
    }

    public static final class DecodedAudio {
        public final float[] pcm16KhzMono;
        public final double durationSeconds;

        DecodedAudio(float[] pcm16KhzMono, double durationSeconds) {
            this.pcm16KhzMono = pcm16KhzMono;
            this.durationSeconds = durationSeconds;
        }
    }

    private static final class PcmAccumulator {
        private float[] samples = new float[16_384];
        private int count;

        void append(ByteBuffer buffer, int encoding, int channels) throws IOException {
            if (channels <= 0) throw new IOException("Invalid audio channel count");
            int bytesPerSample = encoding == AudioFormat.ENCODING_PCM_FLOAT ? 4 : 2;
            int frameBytes = bytesPerSample * channels;
            if (buffer.remaining() % frameBytes != 0) throw new IOException("Invalid PCM frame size");
            int frames = buffer.remaining() / frameBytes;
            grow(frames);
            for (int frame = 0; frame < frames; frame++) {
                float sum = 0f;
                for (int channel = 0; channel < channels; channel++) {
                    float value;
                    if (encoding == AudioFormat.ENCODING_PCM_FLOAT) value = buffer.getFloat();
                    else if (encoding == AudioFormat.ENCODING_PCM_16BIT) value = buffer.getShort() / 32768f;
                    else throw new IOException("Unsupported decoder PCM encoding " + encoding);
                    if(!Float.isFinite(value))throw new IOException("Audio contains invalid PCM samples");
                    sum += value;
                }
                samples[count++] = sum / channels;
            }
        }

        private void grow(int extra) throws IOException {
            if ((long) count + extra > MAX_OUTPUT_FRAMES * 3L) {
                throw new IOException("Audio source exceeds the memory safety limit");
            }
            int required = count + extra;
            if (required <= samples.length) return;
            int size = samples.length;
            while (size < required) size = Math.min(MAX_OUTPUT_FRAMES * 3, size * 2);
            float[] expanded = new float[size];
            System.arraycopy(samples, 0, expanded, 0, count);
            samples = expanded;
        }

        float[] toArray() {
            float[] result = new float[count];
            System.arraycopy(samples, 0, result, 0, count);
            return result;
        }
    }
}
