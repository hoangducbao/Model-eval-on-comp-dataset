# Kế hoạch thiết kế: App Android benchmark ASR + MT (Qwen3-ASR + HY-MT2)

> **Trạng thái:** v0.6 — triển khai larplarpsahur (2026-10-24). §22 chốt phạm vi APK hiện tại: giao diện Input/Output, model tải lần đầu, dataset Hugging Face quadata, export và giới hạn kiểm chứng. Khi khác với đề xuất cũ, §22 được ưu tiên; không triển khai qdc_build_run.py.
> Các mục đánh dấu **[TBD]** là phần chưa rõ, hiện chỉ ghi chung chung và sẽ chi tiết hóa khi có thêm thông tin. Hai thông tin quan trọng nhất còn thiếu: **lệnh nạp dataset** và **định dạng dataset / file kết quả bắt buộc của BTC**.

---

## 1. Mục tiêu và phạm vi

### 1.1 Mục tiêu

- Một app Android **mới, tự chứa (self-contained)**, chỉ cần **một file APK** để cài và chạy trên Qualcomm Device Cloud (QDC).
- Dùng lại hai model từ project `hymt2+qwenasr`:
  - **Qwen3-ASR**: audio → văn bản.
  - **HY-MT2**: văn bản → bản dịch.
- Cho phép **đính kèm (attach) audio** và chạy pipeline ASR → MT trên các audio đó.
- **Đánh giá ngay trong app**: tính **CER** (cho ASR) và **BLEU** (cho MT) so với reference.
- **Xuất kết quả benchmark** ra file **JSON** và **CSV**.
- Nạp dataset bằng lệnh `adb push` + `am start` (§20); app tự dò audio và reference (§17-§18); tạo hai bảng đánh giá: ASR (WER, CER) và dịch (CER, BLEU) (§19).

### 1.2 Ngoài phạm vi (giai đoạn này)

- Ghi âm trực tiếp, streaming/realtime ASR.
- Giao diện đẹp, đa ngôn ngữ giao diện, theme.
- Đồng bộ cloud, tài khoản người dùng.
- Phát hành lên Play Store (chỉ cần APK debug hoặc release ký bằng key tự tạo).

### 1.3 Tiêu chí hoàn thành

1. Một file APK cài được lên thiết bị QDC (SM8850) mà không cần bước thủ công nào khác ngoài việc mở app.
2. Attach được N file audio, chạy hết, không crash, không treo.
3. Có file `results.json` và `results.csv` chứa CER, BLEU ở mức tổng thể và từng mẫu, cùng thời gian chạy.
4. CER/BLEU của app khớp với `jiwer` / `sacrebleu` chạy trên PC cho cùng dữ liệu (sai lệch trong ngưỡng chấp nhận, xem §7.4).
5. Lệnh nạp dataset có thể cắm vào mà không đổi pipeline và metric.

---

## 2. Ràng buộc và hệ quả thiết kế

| Ràng buộc | Hệ quả |
|---|---|
| QDC chỉ cần APK | Không dựa vào PC: không `adb push`, không script Python tính metric. **Metric phải tính trong app.** Model và thư viện native phải nằm trong APK (hoặc app tự tải, xem câu hỏi mở). |
| Thiết bị đích Snapdragon 8 Elite Gen 5 (SM8850), arm64 | Chỉ build ABI `arm64-v8a` để APK nhỏ hơn. |
| Phút thiết bị tốn tiền, phiên có giới hạn thời gian | Chạy tự động được (không cần bấm nhiều), **lưu kết quả từng mẫu ngay khi xong**, có thể resume, có timeout cho từng mẫu. |
| Không chắc thiết bị QDC có internet **[TBD]** | Không phụ thuộc mạng khi chạy. |
| Không chắc cách lấy file kết quả ra khỏi thiết bị **[TBD]** | Hỗ trợ nhiều đường xuất (§9), kể cả in kết quả ra logcat và hiển thị trên màn hình. |
| Model lớn | RAM đỉnh là rủi ro số một. Cân nhắc nạp lần lượt từng model (§6.2). |

**Giả định hiện tại:** audio đưa vào là file (wav/mp3/m4a/flac...), mỗi mẫu có thể có reference transcript và reference translation. Ngôn ngữ nguồn/đích chưa chốt **[TBD]**.

---

## 3. Kiến trúc tổng thể

```
┌───────────────────────────── app (UI, điều hướng) ─────────────────────────────┐
│  HomeScreen · SamplesScreen · RunScreen · ResultsScreen · ExportScreen          │
└───────────────┬────────────────────────────────────────────────┬───────────────┘
                │ gọi                                            │ quan sát state
        ┌───────▼────────┐                               ┌───────▼────────┐
        │ BenchmarkRunner │  (coroutine / foreground svc) │  RunViewModel  │
        └───────┬────────┘                               └────────────────┘
                │
   ┌────────────┼──────────────────────────────┐
   │            │                              │
┌──▼───┐   ┌────▼─────┐   ┌──────────┐   ┌─────▼─────┐   ┌──────────────┐
│Audio │   │AsrEngine │   │MtEngine  │   │Metrics    │   │ResultStore + │
│Loader│   │(Qwen3)   │   │(HY-MT2)  │   │CER / BLEU │   │Exporter      │
└──────┘   └──────────┘   └──────────┘   └───────────┘   └──────────────┘
                ▲
        ┌───────┴─────────┐
        │ DatasetSource   │  ← ManualAttachSource (làm trước)
        └─────────────────┘    CommandDatasetSource (làm sau, TBD)
```

### 3.1 Phân module

| Module | Nội dung | Ghi chú |
|---|---|---|
| `:app` | UI (Jetpack Compose), điều hướng, khởi tạo, quyền | Mỏng, không chứa logic benchmark. |
| `:core` | `Sample`, `DatasetSource`, `BenchmarkRunner`, chuẩn hóa văn bản, **CER, BLEU**, `Exporter`, schema JSON/CSV | **Kotlin thuần, không phụ thuộc Android** để unit test trên JVM. |
| `:engine-asr` | Bọc Qwen3-ASR sau interface `AsrEngine` | Tái sử dụng code từ project cũ. |
| `:engine-mt` | Bọc HY-MT2 sau interface `MtEngine` | Tái sử dụng code từ project cũ. |

### 3.2 Interface chính (phác thảo)

```kotlin
data class Sample(
    val id: String,
    val audio: AudioRef,              // Uri hoặc đường dẫn file
    val srcLang: String? = null,
    val tgtLang: String? = null,
    val refText: String? = null,      // reference transcript (để tính CER)
    val refTrans: String? = null      // reference translation (để tính BLEU)
)

interface DatasetSource {
    suspend fun load(): List<Sample>
}

interface AsrEngine : AutoCloseable {
    suspend fun load(config: AsrConfig)
    suspend fun transcribe(pcm: FloatArray, sampleRate: Int, lang: String?): AsrResult
}

interface MtEngine : AutoCloseable {
    suspend fun load(config: MtConfig)
    suspend fun translate(text: String, src: String?, tgt: String): MtResult
}
```

Nhờ tách interface, đổi backend (CPU ↔ NPU) hoặc đổi cách nạp dataset không ảnh hưởng phần còn lại.

---

## 4. Tái sử dụng từ project `hymt2+qwenasr`

Việc đầu tiên là **kiểm kê** những gì project cũ đang dùng, vì app mới phải tự chứa tất cả. Danh sách cần thu thập:

| Hạng mục | Cần biết | Trạng thái |
|---|---|---|
| Thư viện native / runtime | Project cũ có phụ thuộc AAR `sherpa-onnx-static-link-onnxruntime-1.13.8.aar` đặt trong `app/libs`. Cần xác định AAR này phục vụ model nào, và HY-MT2 chạy bằng runtime nào. | **[TBD]** |
| File model | Tên file, định dạng, dung lượng của Qwen3-ASR và HY-MT2 (kể cả file tokenizer/config/vocab). | **[TBD]** |
| Prompt template của HY-MT2 | Định dạng prompt dịch, tham số sinh (max tokens, temperature/greedy...). | **[TBD]** |
| Cấu hình ASR | Tần số lấy mẫu, cách chunk audio dài, tham số decode, cách chọn ngôn ngữ. | **[TBD]** |
| Backend | CPU hay có dùng NPU (QNN/HTP) trên Snapdragon. | **[TBD]** |
| `minSdk` / `targetSdk` | Theo project cũ và Android của thiết bị QDC. | **[TBD]** |

**Nguyên tắc vendor hóa:** mọi AAR, `.so`, model, tokenizer phải nằm **trong repo của app mới** (hoặc script tải về có kiểm tra). Không được phụ thuộc đường dẫn trên máy cá nhân, để tránh lặp lại lỗi thiếu file AAR khi build.

---

## 5. Chức năng chi tiết

### 5.1 Attach audio

- Chọn nhiều file qua **Storage Access Framework** (`ACTION_OPEN_DOCUMENT` với `EXTRA_ALLOW_MULTIPLE`), không cần xin quyền đọc bộ nhớ rộng.
- Định dạng hỗ trợ dự kiến: wav, mp3, m4a/aac, flac, ogg. Giải mã bằng `MediaExtractor` + `MediaCodec`, riêng wav PCM có thể đọc trực tiếp.
- Mỗi mẫu có thể gắn **reference** theo ba cách (làm dần):
  1. Nhập tay trong app.
  2. File đi kèm cùng tên (ví dụ `abc.wav` + `abc.txt`) **[TBD]**.
  3. Import file manifest CSV/JSON **[TBD]** (định dạng chờ BTC).
- Mẫu không có reference vẫn chạy được, nhưng chỉ có hypothesis và thời gian, metric tương ứng ghi `null`.

### 5.2 Cấu hình lần chạy

- Ngôn ngữ nguồn/đích (hoặc tự động nếu engine hỗ trợ) **[TBD]**.
- Bật/tắt bước MT (để chạy riêng ASR).
- Chế độ MT: `pipeline` (dịch từ kết quả ASR) và/hoặc `oracle` (dịch từ reference transcript, để tách riêng chất lượng MT khỏi lỗi ASR).
- Profile chuẩn hóa văn bản cho CER/BLEU (§7.1).
- Số thread, số mẫu warm-up.

### 5.3 Chạy benchmark

- Chạy tuần tự từng mẫu trong coroutine, bọc trong **foreground service** để không bị hệ thống kill khi màn hình tắt.
- Hiển thị tiến độ, mẫu hiện tại, nút hủy.
- Lưu từng mẫu ngay khi xong vào file tạm (JSON Lines), nên crash giữa chừng vẫn còn dữ liệu và có thể **resume** (bỏ qua mẫu đã xong).
- Lỗi một mẫu không làm hỏng cả lượt chạy: ghi `status = error` kèm thông báo rồi chạy tiếp.

### 5.4 Xem kết quả

- Tóm tắt: CER tổng, BLEU tổng, số mẫu thành công/lỗi, thời gian trung bình, RTF.
- Danh sách mẫu: xem ref vs hyp, CER từng mẫu, sentence-BLEU từng mẫu.

### 5.5 Xuất kết quả

Xem §8 (định dạng) và §9 (cách lấy file ra).

### 5.6 Lệnh nạp dataset

Đã chốt (v0.2): dùng `adb push` qua SSH tunnel của QDC để đẩy dataset vào thư mục của app, rồi `adb shell am start` kèm tham số để app tự dò và chạy. Chi tiết ở §20; cách dò và ghép audio với reference ở §17-§18.

`DatasetSource` vẫn giữ làm interface, với hai implementation:

- `ScannedFolderSource`: quét thư mục `dataset_root` do lệnh truyền vào (đường chính).
- `ManualAttachSource`: attach file thủ công qua giao diện (dự phòng và dùng khi phát triển).

---

## 6. Luồng xử lý (pipeline)

### 6.1 Cho từng mẫu

```
audio file
  → giải mã → PCM mono, 16 kHz (resample nếu cần)           [AudioLoader]
  → (chia chunk nếu audio dài)                              [TBD]
  → ASR (Qwen3-ASR) → asr_hyp, đo asr_ms
  → chuẩn hóa → CER so với ref_text
  → MT (HY-MT2) từ asr_hyp → mt_hyp, đo mt_ms
  → chuẩn hóa/tokenize → sentence-BLEU so với ref_trans
  → ghi 1 dòng kết quả (JSONL) ngay lập tức
```

Các số liệu thu thập cho mỗi mẫu: thời lượng audio, `asr_ms`, `mt_ms`, RTF (`asr_ms / duration`), trạng thái, lỗi (nếu có).

### 6.2 Hai chiến lược nạp model

| Chiến lược | Cách làm | Ưu | Nhược |
|---|---|---|---|
| **A. Hai pha (đề xuất mặc định)** | Nạp ASR → chạy ASR cho **toàn bộ** mẫu → giải phóng → nạp MT → dịch toàn bộ | RAM đỉnh thấp, ổn định | Phải lưu trung gian `asr_hyp`; đo thời gian theo từng pha chứ không phải end-to-end liền mạch. |
| **B. Một pha** | Nạp cả hai model, chạy từng mẫu ASR → MT | Đơn giản, đo end-to-end sát thực tế | Cần đủ RAM cho cả hai model cùng lúc. |

Làm **A** trước vì an toàn; thêm **B** như tùy chọn khi biết rõ mức RAM thực tế **[TBD]**.

### 6.3 Warm-up và đo thời gian

- Chạy 1-2 mẫu warm-up (không tính vào kết quả) để loại ảnh hưởng của lần chạy đầu.
- Dùng `SystemClock.elapsedRealtimeNanos()` để đo.
- Ghi riêng thời gian **nạp model** và thời gian **suy luận**.

---

## 7. Metric

> Lưu ý: cách chuẩn hóa và tokenize quyết định con số cuối cùng. Cần chốt với BTC để khớp cách chấm của họ **[TBD]**. Các lựa chọn dưới đây là **mặc định đề xuất**, và đều cấu hình được.

### 7.1 Chuẩn hóa văn bản (profile cấu hình được)

Mặc định đề xuất:

1. Chuẩn hóa Unicode **NFC** (quan trọng với tiếng Việt, tránh lệch giữa ký tự dựng sẵn và tổ hợp).
2. Hạ chữ thường (chỉ áp dụng cho chữ có phân biệt hoa/thường).
3. Bỏ dấu câu (tùy chọn bật/tắt).
4. Gộp nhiều khoảng trắng thành một, cắt đầu/cuối.
5. Với tiếng Trung: bỏ khoảng trắng khi tính CER.

### 7.2 CER

- Tính theo **ký tự (code point Unicode)**, không dùng độ dài UTF-16 của `String` trong Kotlin.
- `CER_mẫu = edit_distance(ref, hyp) / len(ref)`, trong đó edit distance là Levenshtein (thế, chèn, xóa đều có chi phí 1).
- **CER tổng (corpus-level)** = `Σ edit_distance / Σ len(ref)` trên toàn bộ mẫu. Không lấy trung bình cộng CER từng mẫu, vì sẽ lệch khi độ dài mẫu khác nhau. App ghi **cả hai** (corpus-level là số chính).
- Reference rỗng: bỏ qua mẫu đó và đánh dấu trong kết quả.
- Cài đặt Levenshtein bằng hai hàng DP để tiết kiệm bộ nhớ.

### 7.3 BLEU

- **Corpus BLEU** (không phải trung bình sentence-BLEU), bám theo `sacrebleu`:
  - n-gram từ 1 đến 4, trọng số đều.
  - Brevity penalty tính trên tổng độ dài hyp và ref của toàn corpus.
  - Smoothing `exp` như mặc định của sacreBLEU.
  - Một reference cho mỗi câu (nhiều reference **[TBD]**).
- **Tokenizer theo ngôn ngữ đích:**
  - Ngôn ngữ dùng chữ Latin (Việt, Anh...): tokenizer kiểu `13a`.
  - Tiếng Trung: tokenizer `zh` (tách theo ký tự CJK).
  - Ngôn ngữ khác: **[TBD]**.
- Ghi vào kết quả chuỗi **signature** (tokenizer, smoothing, phiên bản thuật toán) để so sánh công bằng.
- Sentence-BLEU của từng mẫu dùng smoothing để tránh về 0; chỉ để tham khảo.

### 7.4 Kiểm chứng độ đúng

- Dựng một bộ dữ liệu nhỏ (khoảng vài chục cặp ref/hyp, có tiếng Việt và tiếng Trung).
- Chạy `jiwer.cer` và `sacrebleu.corpus_bleu` **một lần trên PC** để lấy số chuẩn, ghi cứng vào unit test của `:core`.
- Tiêu chí: CER khớp tuyệt đối (hoặc sai số rất nhỏ do làm tròn), BLEU lệch không quá ~0.1 điểm.

### 7.5 Chỉ số phụ (ghi trong kết quả)

- Thời gian nạp model, thời gian ASR/MT mỗi mẫu, RTF, trung bình và p50/p95.
- RAM đỉnh (nếu đo được) **[TBD]**.
- Thông tin thiết bị (model, SoC, Android version) và phiên bản app/model.

---

## 8. Định dạng đầu ra

> Nếu BTC quy định schema riêng thì sẽ đổi theo. Đây là bản đề xuất ban đầu **[TBD]**.

### 8.1 `results.json`

```json
{
  "schema_version": 1,
  "run": {
    "run_id": "2026-10-24T10-30-00Z",
    "app_version": "0.1.0",
    "device": { "model": "...", "soc": "SM8850", "android": "..." },
    "models": {
      "asr": { "name": "Qwen3-ASR", "version": "..." },
      "mt":  { "name": "HY-MT2",    "version": "..." }
    },
    "config": {
      "src_lang": "...", "tgt_lang": "...",
      "normalization": "default-v1",
      "mt_mode": "pipeline"
    }
  },
  "summary": {
    "num_samples": 0, "num_ok": 0, "num_error": 0,
    "cer": 0.0,
    "bleu": 0.0,
    "bleu_signature": "tok:13a|smooth:exp|n:4",
    "avg_asr_ms": 0.0, "avg_mt_ms": 0.0, "rtf": 0.0,
    "model_load_ms": { "asr": 0, "mt": 0 }
  },
  "samples": [
    {
      "id": "...", "audio": "...", "duration_s": 0.0,
      "ref_text": "...", "asr_hyp": "...", "cer": 0.0,
      "ref_trans": "...", "mt_hyp": "...", "sent_bleu": 0.0,
      "asr_ms": 0, "mt_ms": 0, "status": "ok", "error": null
    }
  ]
}
```

### 8.2 `results.csv`

Mỗi dòng một mẫu, cột:

```
id,audio,duration_s,src_lang,tgt_lang,ref_text,asr_hyp,cer,ref_trans,mt_hyp,sent_bleu,asr_ms,mt_ms,rtf,status,error
```

- Theo RFC 4180: bao dấu nháy kép các trường chứa dấu phẩy, nháy kép, xuống dòng.
- Mã hóa **UTF-8**. Tùy chọn thêm BOM để Excel đọc đúng tiếng Việt/Trung.
- Thêm một file `summary.csv` (một dòng, các chỉ số tổng) nếu BTC cần **[TBD]**.

### 8.3 Ghi an toàn

- Trong lúc chạy ghi vào `results.partial.jsonl`; khi xong mới dựng `results.json` và `results.csv` từ file này.
- Ghi ra file tạm rồi đổi tên (rename) để tránh file hỏng nếu app bị kill giữa chừng.

---

## 9. Lấy kết quả ra khỏi thiết bị **[TBD]**

Chưa biết chắc cách QDC cho lấy file, nên làm nhiều đường song song, tốn rất ít công:

1. **Lưu vào thư mục riêng của app** (`getExternalFilesDir`). Nếu phiên QDC cho `adb pull`/duyệt file thì lấy được.
2. **Lưu ra thư mục Downloads** bằng `MediaStore` (không cần quyền đặc biệt trên Android 10+).
3. **Nút Share / Save as** (SAF `ACTION_CREATE_DOCUMENT`) để lưu hoặc gửi file.
4. **In ra logcat** khối tóm tắt và toàn bộ JSON với tag cố định, ví dụ:
   ```
   BENCH_RESULT_BEGIN
   {...json...}
   BENCH_RESULT_END
   ```
   Phương án dự phòng cho trường hợp chỉ lấy được log của phiên. Lưu ý logcat có giới hạn độ dài mỗi dòng nên cần chia nhỏ.
5. **Hiển thị trên màn hình** tóm tắt CER/BLEU để chụp ảnh màn hình hoặc xem qua phiên tương tác.

Cần kiểm tra trong lần chạy thử đầu tiên trên QDC xem đường nào dùng được (ảnh phiên QDC có mục `UserCollectedLogs`, chưa rõ nguồn của nó).

---

## 10. Giao diện (tối giản)

| Màn hình | Nội dung |
|---|---|
| **Home** | Trạng thái model (đã nạp/chưa), nút "Attach audio", nút "Run", nút "Run all (auto)". |
| **Samples** | Danh sách mẫu đã attach, sửa reference, xóa mẫu. |
| **Run** | Thanh tiến độ, mẫu đang chạy, thời gian đã trôi qua, nút hủy. |
| **Results** | Tóm tắt CER/BLEU/RTF, danh sách mẫu, xem chi tiết ref vs hyp. |
| **Export** | Các nút lưu/chia sẻ file, vị trí file, nút copy log. |

Ưu tiên chức năng hơn thẩm mỹ: Compose + Material 3 mặc định là đủ.

---

## 11. Cấu trúc project đề xuất

```
asr-mt-benchmark/
├─ settings.gradle.kts
├─ build.gradle.kts
├─ app/
│  ├─ build.gradle.kts
│  ├─ libs/                       # AAR/so cần thiết (vendor hóa)
│  └─ src/main/
│     ├─ AndroidManifest.xml
│     ├─ assets/models/           # file model (nếu đóng gói trong APK)
│     └─ java/.../ui/             # Compose screens, ViewModel
├─ core/
│  └─ src/
│     ├─ main/kotlin/.../
│     │  ├─ model/                # Sample, Result, Config
│     │  ├─ dataset/              # DatasetSource, ManualAttachSource
│     │  ├─ runner/               # BenchmarkRunner
│     │  ├─ text/                 # Normalizer, Tokenizers (13a, zh)
│     │  ├─ metrics/              # Cer.kt, Bleu.kt
│     │  └─ export/               # JsonExporter, CsvExporter
│     └─ test/kotlin/.../         # unit test metric (golden values từ jiwer/sacrebleu)
├─ engine-asr/                    # bọc Qwen3-ASR
├─ engine-mt/                     # bọc HY-MT2
└─ docs/
   └─ plan.md                     # file này
```

---

## 12. Công nghệ và cấu hình build

- **Ngôn ngữ/UI:** Kotlin, Jetpack Compose, Coroutines/Flow.
- **Serialization:** `kotlinx.serialization` cho JSON; CSV tự viết (đơn giản, tránh thêm thư viện).
- **ABI:** `ndk { abiFilters += "arm64-v8a" }`.
- **Thư viện native:** đặt trong `libs/` và khai báo cục bộ; **tránh `flatDir`** vì đã gây lỗi và đang bị khuyến cáo bỏ.
- **Đóng gói model trong APK:**
  - Cấu hình `androidResources { noCompress += listOf("onnx", "gguf", "bin", ...) }` (đuôi file thực tế **[TBD]**) để tránh nén lại file model lớn.
  - Lần chạy đầu có thể phải copy model từ assets ra `filesDir` (tốn thời gian và dung lượng) nếu runtime không đọc trực tiếp từ APK được **[TBD]**.
  - APK có thể rất lớn; cần kiểm tra giới hạn dung lượng upload của QDC **[TBD]**.
- **Ký APK:** debug key hoặc keystore tự tạo là đủ cho QDC.
- **Quyền:** hầu như không cần quyền runtime nhờ SAF; có thể cần `FOREGROUND_SERVICE` cho lúc chạy dài.
- **Build một lệnh:** `./gradlew assembleDebug` (hoặc `assembleRelease`) ra APK duy nhất; script `qdc-runner` hiện có chỉ cần trỏ `--src` sang project mới.

---

## 13. Kế hoạch triển khai theo giai đoạn

| Giai đoạn | Việc chính | Kết quả kiểm tra được |
|---|---|---|
| **P0. Kiểm kê** | Hoàn thành bảng §4: runtime, file model, prompt, cấu hình, backend, minSdk. Build lại được project cũ. | Có danh sách đầy đủ thứ cần copy. |
| **P1. Khung app + ASR** | Tạo project mới, copy engine ASR, attach 1 file audio, hiện transcript. | Transcript đúng trên 1 file mẫu. |
| **P2. Thêm MT + pipeline** | Copy engine MT, nối ASR → MT, đo thời gian. | Dịch được transcript của 1 file. |
| **P3. Metric** | Viết Normalizer, CER, BLEU trong `:core`; unit test với số chuẩn từ `jiwer`/`sacrebleu`. | Test metric xanh. |
| **P4. Batch + lưu bền** | Attach nhiều file, chạy tuần tự, ghi JSONL từng mẫu, resume, xử lý lỗi từng mẫu. | Chạy hết 20-50 mẫu không crash; kill app giữa chừng vẫn resume được. |
| **P5. Export** | Dựng `results.json`/`results.csv`, các đường xuất (§9). | File xuất mở được, số khớp UI. |
| **P6. Chạy thử trên QDC** | Build APK, upload, chạy bộ mẫu nhỏ, kiểm tra đường lấy kết quả. | Có kết quả thật trên SM8850; chốt đường xuất dùng được. |
| **P7. Nạp dataset bằng lệnh** | Cài `CommandDatasetSource` theo thông tin BTC cung cấp; chế độ `auto_run`. | Một lệnh nạp dataset và chạy toàn bộ. |
| **P8. Tối ưu** | Chiến lược nạp model, thread, backend NPU (nếu có), giảm thời gian khởi động. | Giảm thời gian/RAM so với P6. |

Thứ tự nên giữ: **P0 → P3** có thể làm trên điện thoại/emulator thường; chỉ lên QDC ở **P6** để tiết kiệm phút thiết bị.

---

## 14. Kiểm thử

- **Unit test (JVM):** Normalizer, CER, BLEU, CSV escape, JSON schema. Dùng golden values lấy từ `jiwer`/`sacrebleu`.
- **Instrumented test:** giải mã audio các định dạng, chạy ASR/MT trên 2-3 file nhỏ.
- **Test độ bền:** chạy 200+ mẫu liên tục, theo dõi RAM và nhiệt; kill app giữa chừng rồi resume.
- **Test lỗi:** file audio hỏng, audio rỗng, audio rất dài, reference rỗng, ký tự đặc biệt/emoji trong CSV.
- **Smoke test QDC:** một lượt chạy ngắn trước mỗi lần chạy dài.

---

## 15. Rủi ro và giảm thiểu

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| APK quá lớn, khó upload/cài | Cao | Chỉ arm64; không nén model; kiểm tra giới hạn QDC sớm; phương án tải model khi chạy lần đầu nếu có mạng. |
| Hết RAM khi giữ cả hai model | Cao | Chiến lược hai pha (§6.2). |
| Không lấy được file kết quả khỏi thiết bị QDC | Cao | Nhiều đường xuất, kể cả logcat và màn hình (§9); kiểm tra ở P6. |
| Metric lệch so với cách chấm của BTC | Trung bình | Chốt quy tắc chuẩn hóa/tokenizer với BTC; ghi signature trong kết quả; profile cấu hình được. |
| Thiếu thư viện/AAR khi build | Trung bình | Vendor hóa toàn bộ phụ thuộc; build sạch trên máy khác trước khi dùng. |
| Runtime model không chạy trên thiết bị QDC (ABI, phiên bản Android, backend) | Trung bình | Chạy thử sớm ở P6 với mẫu nhỏ; giữ đường CPU làm mặc định. |
| Phiên QDC hết giờ giữa chừng | Trung bình | Lưu từng mẫu, resume, chế độ chạy tự động. |
| Audio dài vượt giới hạn engine | Thấp-Trung bình | Chunk audio **[TBD]**; ghi cảnh báo. |

---

## 16. Câu hỏi mở (cần làm rõ)

1. ~~Lệnh nạp dataset~~: đã chọn `adb push` + `am start` (§20); cần xác nhận thư mục đích và quyền đọc file trên thiết bị QDC thật.
2. **Dataset thật của BTC** có cùng cấu trúc với bộ mẫu Round 3 (§17) không?
3. **Định dạng/schema file kết quả bắt buộc** (tên file, tên cột, đơn vị CER dạng 0-1 hay %)?
4. **Quy tắc tính CER/BLEU của BTC:** chuẩn hóa, tokenizer, corpus-level hay trung bình từng câu?
5. **Ngôn ngữ nguồn/đích** của dataset.
6. **Model đã có sẵn trong project cũ ở dạng nào**, dung lượng bao nhiêu, runtime nào?
7. **QDC có internet không**, và **giới hạn dung lượng APK** khi upload?
8. **Cách lấy file ra** từ phiên QDC (adb pull, UserCollectedLogs, chỉ log...)?
9. Có cần dùng **NPU (QNN/HTP)** hay chỉ CPU là đủ?
10. Có yêu cầu về **thời gian chạy tối đa** hoặc chỉ số hiệu năng khác ngoài CER/BLEU không?
11. ~~Track VM-zh~~: đã xác nhận nằm trong phạm vi chấm.
12. ~~Reference của Eng-Vie~~: đã xác nhận không có; vẫn chạy EV-en để lấy transcript, không tính metric.
13. Một số cặp tham chiếu có vẻ lệch nội dung (§17.2, mục 7): BTC có loại ra khi chấm không?
14. BLEU cho tiếng Trung: BTC dùng tokenizer nào?

---

## 17. Bổ sung v0.2: dataset mẫu Round 3 (đã kiểm tra trên file thật)

### 17.1 Cấu trúc thư mục

```
Round 3-Bộ dữ liệu mẫu/
├─ Vie-Man dataset/
│  ├─ OneVoice_V_to_M_samples_180.xlsx    sheet "Samples": id | vie_text | man_text | vie_voice_path | man_voice_path
│  ├─ Vie_voice/   180 file VM-xxxx.wav   (tiếng Việt)
│  ├─ Vie_text/    180 file VM-xxxx.txt
│  ├─ Man_voice/   180 file VM-xxxx.wav   (tiếng Trung)
│  └─ Man_text/    180 file VM-xxxx.txt
├─ Vie-Kor dataset/                       NGOÀI PHẠM VI: không chạy, không đánh giá
│                                         (cấu trúc giống Vie-Man, tiền tố VK-)
└─ Vie-Eng dataset/
   ├─ Vie-Eng/
   │  ├─ OneVoice_V_to_E_samples_100.xlsx sheet "Final": ID | Tiếng Việt | Tiếng Anh - Final
   │  └─ wav/      100 file VE-xxxx.wav   (tiếng Việt)
   └─ Eng-Vie/
      ├─ OneVoice_V_to_E_samples_100.xlsx bản sao y hệt file bên Vie-Eng
      └─ wav/      100 file EV-xxxx.wav   (tiếng Anh)
```

Ngoài ra có các file rác `.DS_Store`. Cột `*_voice_path` trong xlsx là đường dẫn tương đối tính từ thư mục dataset (ví dụ `Vie_voice/VM-0223.wav`) và đều trỏ đúng file (360/360 ở Vie-Man).

### 17.2 Những phát hiện ảnh hưởng thiết kế

| # | Phát hiện | Hệ quả thiết kế |
|---|---|---|
| 1 | **File đuôi `.wav` thực chất là MP3**: trong phạm vi project là 159/180 file `Man_voice` và 1 file `Vie_voice` của Vie-Man (toàn bộ file zip có 343/920 file như vậy, phần còn lại thuộc Vie-Kor). | Giải mã theo **nội dung** bằng `MediaExtractor`/`MediaCodec`, không dựa vào đuôi file, không tự viết parser WAV. |
| 2 | Định dạng không đồng nhất: `Vie_voice` có 16/22.05/24/44.1/48 kHz và có file stereo; MP3 là 24 kHz mono; Vie-Eng và Eng-Vie đều 16 kHz mono. | Luôn **downmix mono và resample** về tần số model cần. |
| 3 | Audio rất ngắn: trung vị 2.3-5.5 s, dài nhất trong phạm vi khoảng 13 s (Eng-Vie). Tổng thời lượng mỗi thư mục chỉ khoảng 330-630 s. | Chunking chỉ cần cho ngoại lệ; đặt giới hạn độ dài và ghi cảnh báo. Cả bộ chạy trong thời gian ngắn nếu model đủ nhanh. |
| 4 | **Vie-Eng**: xlsx có cột ID khớp 100/100 tên file `VE-xxxx.wav`, sắp xếp tăng dần giống thứ tự file. | Ghép theo **ID** là chính; ghép theo thứ tự trên-xuống chỉ là dự phòng và cho cùng kết quả. Cột "Tiếng Việt" chính là transcript tham chiếu, nên vẫn tính được WER/CER cho Vie-Eng dù không có file `.txt`. |
| 5 | **Eng-Vie**: 100 audio `EV-xxxx` (tiếng Anh) nhưng **không có reference**: xlsx trong thư mục là bản sao của bảng VE, ID không trùng EV nào. Kiểm tra nhanh độ dài audio so với độ dài câu theo thứ tự hàng: Vie-Eng tương quan khoảng 0.81, còn Eng-Vie khoảng -0.1 (không có dấu hiệu khớp). **Đã xác nhận: không có transcript nào khớp với audio tiếng Anh.** | **Không ghép theo thứ tự** cho Eng-Vie. App vẫn chạy ASR + MT và lưu transcript/translation, còn metric ghi `null`. |
| 6 | Vie-Man: nội dung `.txt` trùng 100% với ô tương ứng trong xlsx (180/180 cho cả cột tiếng Việt và cột tiếng đích). | Hai nguồn dự phòng cho nhau. Dùng `.txt` cho bảng transcript và xlsx cho bảng dịch, đúng yêu cầu. |
| 7 | Một số cặp tham chiếu **có vẻ lệch nội dung**. Ví dụ VM-0383: câu Việt chỉ là "Được điểm tối đa." nhưng câu Trung dài hơn nhiều. Tôi mới xem một phần nên chưa biết có bao nhiêu mẫu như vậy. | Lưu điểm **theo từng mẫu** để lọc ngoại lệ về sau; không tinh chỉnh model theo các mẫu này. |
| 8 | Có `.DS_Store`; tên thư mục có dấu và khoảng trắng. | Bỏ qua file ẩn. Khi push lên thiết bị nên dùng thư mục nguồn tên ASCII. |

---

## 18. Dò và ghép dữ liệu tự động

### 18.1 Thuật toán quét (`DatasetScanner`)

1. Nhận `dataset_root`, duyệt đệ quy, bỏ qua file và thư mục ẩn (tên bắt đầu bằng `.`).
2. Thu thập file audio theo đuôi (`.wav .mp3 .m4a .aac .flac .ogg`, không phân biệt hoa thường). Chưa kiểm tra codec ở bước này.
3. Thu thập file `.xlsx` và `.txt`.
4. **Suy ra ngôn ngữ của từng audio**, theo thứ tự ưu tiên:
   - Tên thư mục cha: `Vie_voice` → vi, `Man_voice` → zh.
   - Thư mục `wav` nằm trong thư mục dạng `A-B`: `Vie-Eng/wav` → nguồn vi, `Eng-Vie/wav` → nguồn en.
   - Tiền tố ID (`VM`, `VE`, `EV`) để biết cặp ngôn ngữ của dataset.
   - Bảng ánh xạ cấu hình tay trong app, dùng khi cấu trúc thật khác bộ mẫu.
5. **Đọc xlsx** (§18.4), nhận cột theo tên không phân biệt hoa thường: cột ID (`id`, `ID`), cột văn bản (`vie_text`, `man_text`, `Tiếng Việt`, `Tiếng Anh - Final`...), cột đường dẫn audio (`*_voice_path`, nếu có).
6. **Ghép reference** cho từng audio:

| Loại reference | Ưu tiên 1 | Ưu tiên 2 | Ưu tiên 3 (chỉ khi bật `order_fallback`) |
|---|---|---|---|
| Transcript (ASR) | `<Lang>_text/<id>.txt` trong cùng dataset | cột văn bản nguồn của xlsx theo ID | theo thứ tự hàng |
| Bản dịch (MT) | cột văn bản đích của xlsx theo ID | `<Tgt>_text/<id>.txt` | theo thứ tự hàng |

7. **Ghép theo thứ tự** chỉ áp dụng khi cả ba điều kiện cùng đúng: bảng không có cột ID hoặc ID không khớp tên file; số hàng bằng số file audio trong thư mục; người dùng bật `order_fallback`. Khi đó sắp xếp tên file theo thứ tự số (natural sort) rồi gán lần lượt cho hàng 1, 2, 3... và chạy kiểm tra hợp lý ở §18.3.
8. Ghi `dataset_scan.json`: số audio mỗi thư mục, số mẫu ghép được, audio không có reference, ID trong bảng không có audio, cảnh báo.

### 18.2 Các track suy ra từ bộ mẫu

| Track | Thư mục audio | Ngôn ngữ audio | Ref transcript | Hướng dịch | Ref bản dịch | Số mẫu |
|---|---|---|---|---|---|---|
| VM-vi | Vie-Man/Vie_voice | vi | Vie_text/*.txt | vi → zh | cột `man_text` | 180 |
| VM-zh | Vie-Man/Man_voice | zh | Man_text/*.txt | zh → vi | cột `vie_text` | 180 |
| VE-vi | Vie-Eng/Vie-Eng/wav | vi | cột "Tiếng Việt" | vi → en | cột "Tiếng Anh - Final" | 100 |
| EV-en | Vie-Eng/Eng-Vie/wav | en | không có | en → vi | không có | 100 |

Dataset **Vie-Kor nằm ngoài phạm vi**: `DatasetScanner` bỏ qua mọi thư mục/ID tiền tố `VK` (danh sách dataset được phép cấu hình được, mặc định `VM`, `VE`, `EV`), để nếu BTC đưa lại dữ liệu có Vie-Kor thì app cũng không chạy nhầm.

**Phạm vi đã chốt (v0.4):** cả 4 track đều chạy mặc định.

- **VM-zh** (audio tiếng Trung, dịch về tiếng Việt) nằm trong phạm vi chấm: có đủ Bảng 1 và Bảng 2.
- **EV-en** đã xác nhận không có reference, nhưng vẫn chạy để lấy transcript (và bản dịch sang tiếng Việt): lưu vào `transcripts` và `translations`, còn WER, CER, BLEU ghi `null` và không đưa vào dòng tổng.
- App vẫn cho bật/tắt từng track để chạy thử nhanh (tham số `tracks`, §20.3).

### 18.3 Kiểm tra hợp lý khi ghép theo thứ tự

- So số hàng với số file; so ID nếu bảng có cột ID.
- Tính tương quan giữa độ dài audio và độ dài câu reference (theo ngôn ngữ nguồn). Tham khảo từ bộ mẫu: ghép đúng (Vie-Eng) khoảng 0.8, ghép sai (Eng-Vie với bảng VE) khoảng 0. Nếu thấp bất thường (ngưỡng **[TBD]**), ghi cảnh báo trong `dataset_scan.json` và đánh dấu metric của track đó là `unverified_pairing`.

### 18.4 Đọc xlsx trong app

Không đưa Apache POI vào APK (nặng, nhiều phụ thuộc). Viết trình đọc tối thiểu:

- Mở `.xlsx` như file zip; đọc `xl/workbook.xml` (tên sheet), `xl/sharedStrings.xml` và `xl/worksheets/sheetN.xml` bằng `XmlPullParser`.
- Hỗ trợ ô `t="s"` (shared string), `t="inlineStr"` và ô số; bỏ qua định dạng.
- Lấy sheet đầu tiên, hoặc sheet có tiêu đề khớp (`Samples`, `Final`).
- Trả về `List<Map<String, String>>` theo tên cột. Hàng đầu là tiêu đề.
- Kiểm thử trên các file xlsx mẫu (Vie-Man, Vie-Eng). Kết quả kỳ vọng khi quét bộ mẫu: Vie-Man 180/180 ghép đủ cả hai loại reference; Vie-Eng 100/100 ghép theo ID; Eng-Vie 0/100 (không có reference); Vie-Kor bị bỏ qua.

---

## 19. Hai bảng đánh giá

### 19.1 Bảng 1: chất lượng transcript (ASR)

- So `asr_hyp` với transcript tham chiếu (file `.txt`; với Vie-Eng là cột "Tiếng Việt").
- Cột từng mẫu: `track, id, lang, ref_text, asr_hyp, wer, cer, ref_words, ref_chars, edits_word, edits_char`.
- Dòng tổng theo từng track và toàn bộ: WER và CER dạng corpus-level (Σ lỗi / Σ độ dài tham chiếu).

### 19.2 Bảng 2: chất lượng dịch (MT)

- So `mt_hyp` với bản dịch tham chiếu (cột xlsx).
- Cột từng mẫu: `track, id, src_lang, tgt_lang, mt_input, ref_trans, mt_hyp, cer, sent_bleu`.
- Dòng tổng theo từng track: CER và BLEU dạng corpus-level, kèm signature (§7.3).
- Hai chế độ đầu vào `mt_input`: `asr_hyp` (pipeline, mặc định) và `ref_text` (oracle, tùy chọn) để tách lỗi ASR khỏi lỗi dịch. Dùng `both` để ra cả hai cột.

### 19.3 Quy tắc theo ngôn ngữ

| Ngôn ngữ | WER | CER | Tokenizer BLEU |
|---|---|---|---|
| vi, en | tách theo khoảng trắng sau chuẩn hóa | theo ký tự (code point) | `13a` |
| zh | không có nghĩa nếu không tách từ: ghi `N/A` hoặc coi mỗi ký tự là một từ **[TBD]** | theo ký tự | `zh` |

Chuẩn hóa mặc định: NFC; bỏ dấu câu cả loại toàn góc và CJK (`， 。 ？ ！ 、 ： ；`...); hạ chữ thường; gộp khoảng trắng; với zh bỏ hết khoảng trắng. Chưa chuẩn hóa chữ số so với chữ viết ("10" so với "mười") ở v0.2 **[TBD]**. Chữ "BLEUR" trong yêu cầu được hiểu là BLEU.

### 19.4 File đầu ra bổ sung

| File | Nội dung |
|---|---|
| `transcripts.csv` / `.json` | `track, id, audio_path, lang, asr_hyp` (lưu đủ transcript) |
| `translations.csv` / `.json` | `track, id, src_lang, tgt_lang, mt_input, mt_hyp` (lưu đủ bản dịch) |
| `eval_asr.csv` | Bảng 1 (từng mẫu + dòng tổng theo track) |
| `eval_mt.csv` | Bảng 2 (từng mẫu + dòng tổng theo track) |
| `eval_summary.json` | Tổng hợp theo track và toàn bộ |
| `results.json` / `results.csv` | Như §8, thêm trường `track` |
| `dataset_scan.json` | Kết quả quét (§18.1, bước 8) |

Mẫu không có reference: metric là `null` và nằm trong danh sách `unmatched`. Giao diện có hai tab hiển thị Bảng 1 và Bảng 2, kèm bộ lọc theo track.

---

## 20. Lệnh nạp dataset lên thiết bị QDC

### 20.1 Quy trình

Theo tài liệu QDC, thiết bị di động được truy cập bằng ADB qua SSH tunnel, hỗ trợ `adb install`, `adb push`, `adb pull`, `adb logcat`; ngoài ra có thể tải file lên bằng trình duyệt file trên giao diện web. Quy trình đề xuất (lệnh tunnel chính xác lấy từ hộp thoại **Connect** của phiên, cổng ADB mặc định là 5037; nếu đổi cổng thì thêm `-P <port>` trước mỗi lệnh `adb`):

```powershell
# 0) Mở SSH tunnel của phiên và giữ cửa sổ này chạy suốt phiên
ssh -i <private_key> -L 5037:sa<session_id>.sa.svc.cluster.local:5037 -N sshtunnel@ssh.qdc.qualcomm.com

# 1) Kiểm tra thiết bị
adb devices

# 2) Cài app
adb install -r app\build\outputs\apk\debug\app-debug.apk

# 3) Đẩy dataset (thư mục nguồn nên có tên ASCII, ví dụ D:\data\round3)
adb push D:\data\round3 /sdcard/Android/data/<package>/files/dataset

# 4) Chạy benchmark tự động
adb shell am start -n <package>/.BenchmarkActivity `
  --es dataset_root /sdcard/Android/data/<package>/files/dataset `
  --ez auto_run true

# 5) Theo dõi và lấy kết quả
adb logcat -s BENCH
adb pull /sdcard/Android/data/<package>/files/results D:\data\results
```

### 20.2 Việc cần chỉnh ở `qdc_build_run.py`

- Thêm tham số `--dataset <thư_mục>` (và tùy chọn `--dataset-dest`).
- Sau bước cài APK: `adb push` dataset, `am start` kèm extras, chờ file `DONE` (hoặc dòng `BENCH_DONE` trong logcat), rồi `adb pull` thư mục kết quả.
- Giữ nguyên nguyên tắc build trước để thất bại sớm, trước khi tốn phút thiết bị.
- Tôi chưa thấy nội dung script nên chưa biết nó đang mở tunnel và chạy `adb` như thế nào **[TBD]**; phần này cần xem script để chỉnh cho khớp.

### 20.3 Phía app (Intent extras)

| Extra | Ý nghĩa |
|---|---|
| `dataset_root` | Thư mục gốc dataset (bắt buộc khi chạy tự động) |
| `auto_run` | `true` thì quét và chạy ngay không cần bấm |
| `tracks` | Danh sách track cần chạy, mặc định tất cả |
| `limit` | Số mẫu tối đa mỗi track, dùng để chạy thử nhanh |
| `mt_mode` | `pipeline` / `oracle` / `both` |
| `out_dir` | Thư mục kết quả, mặc định `<files>/results` |

Không có extras thì mở màn hình chọn thư mục thủ công (SAF) hoặc ô nhập đường dẫn.

### 20.4 Quyền đọc file

- Thư mục đích khuyến nghị: `/sdcard/Android/data/<package>/files/dataset`. App đọc được mà không cần quyền đặc biệt; cần cài app trước khi push (hoặc `adb shell mkdir -p` thư mục đó).
- Phương án khác: push vào `/sdcard/<thư_mục>` rồi cấp quyền truy cập toàn bộ file bằng `adb shell appops set --uid <package> MANAGE_EXTERNAL_STORAGE allow` (kèm khai báo `MANAGE_EXTERNAL_STORAGE` trong manifest).
- Không dùng `/data/local/tmp`, vì app thường không đọc được.
- Hành vi phụ thuộc phiên bản Android của thiết bị QDC, nên kiểm tra cả hai ở lần chạy thử đầu tiên **[TBD]**.

---

## 21. Thay đổi so với v0.1 và bước tiếp theo

- §5.6: chốt "lệnh nạp dataset" là `adb push` + `am start`.
- §19.4 bổ sung các file đầu ra; §16 cập nhật câu hỏi mở (thêm câu 11-14).
- v0.3: loại Vie-Kor khỏi phạm vi (còn 4 track: VM-vi, VM-zh, VE-vi, EV-en); xác nhận Eng-Vie không có reference.
- v0.4: chốt VM-zh nằm trong phạm vi chấm; EV-en vẫn chạy để lấy transcript (không có metric).
- Việc tiếp theo, theo thứ tự:
  1. P0 vẫn phải làm: kiểm kê model và runtime của project cũ (§4).
  2. Viết `DatasetScanner` và trình đọc xlsx, kiểm thử trên 3 xlsx mẫu với kết quả kỳ vọng ở §18.4.
  3. Viết hai bảng đánh giá (§19) cùng unit test metric cho vi/zh/en.
  4. Mở rộng `qdc_build_run.py` theo §20.2.

---

## 22. APK larplarpsahur: phạm vi đã triển khai (v0.6)

Phần này ưu tiên khi khác với đề xuất cũ. App ID `com.larplarpsahur`, source ở `android/`, dùng Java một module và chỉ giữ runtime ONNX/QNN cùng các lớp inference cần thiết từ app tham chiếu.

### 22.1 Màn hình và pipeline

- Màn hình chính giữ **Input box** → chọn đích Việt/Anh/Trung → nút **Chạy** → **Output box**. Input liệt kê audio; Output có ngôn ngữ Qwen phát hiện, transcript và bản dịch. Chọn audio không tự chạy; hai box cho phép copy.
- Model và nút tải dataset Hugging Face nằm dưới phần chính. Output hiển thị mẫu gần nhất; ZIP chứa đủ kết quả mọi mẫu.
- `WorkService` foreground có thông báo, wake lock và duy nhất một worker, vì vậy xoay màn hình không chạy trùng. Nếu Android kết thúc tiến trình, người dùng có thể chọn **Tiếp tục lượt chạy gần nhất**; app không giữ hay tự phát lại credential.
- Pipeline chạy ASR cho toàn bộ mẫu, đóng Qwen, rồi mới nạp HY-MT. Nhãn nguồn lấy trực tiếp từ output Qwen: `und` khi không có nhãn hỗ trợ, `mul` khi các đoạn có nhiều nhãn.
- Đọc audio qua MediaExtractor/MediaCodec, chuyển mono float 16 kHz. Giới hạn 10 phút/audio; Qwen chia đoạn 18 giây và HY-MT chia văn bản dài. Cần kiểm chứng chất lượng thực tế vì ranh giới đoạn có thể ảnh hưởng bản dịch.

### 22.2 Model tải ở lần mở đầu

- Graph/trọng số Qwen từ GitHub `DS-PNQ/OnSpeak47-npu-models` release `v2.2.0-npu`; HY-MT từ `v2.1.0-npu`.
- Graph được compile cho **SM8850**; APK arm64 yêu cầu Android 12+, SM8850 và driver QNN tương thích. Không có fallback CPU trong APK này.
- Release GitHub hiện thiếu `tokenizer.json`; app tải tokenizer Tencent HY-MT2-1.8B tại commit `9a341cd1b679d3efd23b46e847b01745a71ed792`, SHA-256 `b475bbef1b0b2fd57dcb865332b546475bd1ede2deb3bb91bafd0c047a8a530a` từ Hugging Face.
- Có thể mirror chính file tokenizer vào GitHub riêng rồi build với `-PtokenizerUrl=<HTTPS URL>`; `qwenModelBase` và `hyMtModelBase` cũng là Gradle properties. Mirror phải trả bytes đúng checksum đã pin. Công việc này không tự tạo/upload release.
- Lần đầu cần khoảng 1.85 GB chưa tính dataset. App kiểm tra dung lượng, kích thước và SHA-256 pin; chỉ đổi `.part` thành file model sau khi hợp lệ. Ghi REV Qwen=1/HY-MT=8 cuối cùng.
- Android DownloadManager giữ tải model qua process death và báo tiến độ/tình trạng mạng. Thử lại giữ file đã xác thực. Dừng tác vụ UI không xóa download model đang dở; lần thử sau tiếp nhận lại.

### 22.3 Hugging Face quadata, dò dataset và export

1. App chỉ tải một ZIP public từ `https://huggingface.co/datasets/Echdipi/quadata/tree/main`: `Round 3-Bộ dữ liệu mẫu-20261024T014123Z-1-001.zip`. Các thư mục/file trùng lặp nằm ngoài ZIP trong repository bị bỏ qua hoàn toàn.
2. URL download pin revision `fed18c73b4350e07a02124df4f4c716a95e26743`; ZIP LFS phải khớp SHA-256 `cde6b58e391094c7578a7bdd37a8202db0735ccd165f91cc18c6f7ded1f8f01e`. Không có ô URL, token hoặc credential dataset.
3. Tải có tiến độ, timeout và tối đa 3 lần thử; dataset lỗi tải lại từ đầu. Download chỉ theo HTTPS redirect, tối đa 8 hop.
4. Giải nén vào staging rồi mới công bố. Chặn path vượt root, duplicate archive path, ZIP >4 GiB, giải nén >6 GiB, >10.000 entries và hết dung lượng.
5. Dò `manifest.json` hoặc Round 3, TXT cùng stem/sibling và XLSX sheet `Samples`/`Final`. Ghép theo ID, **không fallback theo thứ tự hàng**. EV-en không reference; Vie-Kor/VK bị bỏ.
6. Checkbox mặc định dùng 4 hướng VM-vi→zh, VM-zh→vi, VE-vi→en, EV-en→vi. Bỏ checkbox dùng đích chung; reference sai ngôn ngữ đích không tạo MT metric.
7. `dataset_scan.json` ghi counts/unmatched/warnings. Lỗi mỗi mẫu được lưu rồi tiếp tục. State mỗi mẫu atomic; `results.partial.jsonl` là audit, có thể lặp ID; state là dữ liệu chính khi resume.
8. Export: `results.json/csv`, `transcripts.json/csv`, `translations.json/csv`, `eval_asr.csv`, `eval_mt.csv`, `eval_summary.json`, marker `DONE`/`CANCELLED`/`FAILED`. DONE chỉ nghĩa mọi mẫu đã được duyệt, vẫn có thể có sample error — xem `error_samples`.
9. Nút **Lưu kết quả ZIP…** dùng Android document picker. Raw results ở app-owned external storage; logcat tag `BENCH` chỉ giúp tìm trạng thái/đường dẫn.

Manifest tối thiểu (reference có thể bỏ):

```json
{"samples":[{"id":"sample-1","audio":"audio/sample-1.wav","src_lang":"vi","tgt_lang":"en","ref_text":"Xin chào","ref_trans":"Hello"}]}
```

`src_lang` ở manifest là ngôn ngữ reference để tính metric, không ép ASR. Reference vắng mặt cho metric null/N/A và không tính vào tổng.

### 22.4 Cloud device và kiểm chứng

- Có thể dùng UI, hoặc `am start` với `auto_run`, `dataset_root`, `target_language`, `tracks`, `limit` (mỗi track), `mt_mode=pipeline|oracle|both`, `track_targets`. `out_dir` chỉ nhận thư mục mới trong external files/results của app.
- Dataset root phải đọc được bởi app; ưu tiên `/sdcard/Android/data/com.larplarpsahur/files/datasets/...`. Không xin quyền truy cập toàn bộ bộ nhớ.
- Metric Java: NFC/lower/remove punctuation/space; CER code point, WER zh N/A; corpus dùng tổng edits/length. BLEU 13a/zh, exp smoothing, sentence effective order và corpus fixed 4-gram, có signature implementation.
- Dừng được giữa download/decode/chunk; lời gọi QNN native đang chạy phải chờ trả về. Decoder có watchdog 60 giây không output, không hứa timeout cứng khi driver native treo.
- JVM test/build/lint không chứng minh model hoặc driver cloud. Cần smoke test trên SM8850 và ZIP Round 3 từ quadata; không cần credential dataset.
- Không thay đổi `qdc_build_run.py`; cung cấp APK debug đã ký để cài trực tiếp.

## Phụ lục A: Pseudo-code metric

### CER (corpus-level)

```
total_edits = 0; total_ref_chars = 0
for each sample with ref_text:
    r = codepoints(normalize(ref_text))
    h = codepoints(normalize(asr_hyp))
    total_edits     += levenshtein(r, h)
    total_ref_chars += len(r)
CER = total_edits / total_ref_chars
```

### BLEU (corpus-level, tương thích sacreBLEU)

```
for each sample with ref_trans:
    h = tokenize(normalize(mt_hyp), tgt_lang)
    r = tokenize(normalize(ref_trans), tgt_lang)
    hyp_len += len(h); ref_len += len(r)
    for n in 1..4:
        total[n]   += max(len(h) - n + 1, 0)
        matches[n] += Σ over n-gram g in h: min(count_h(g), count_r(g))

precision[n] = matches[n] / total[n]      # smoothing 'exp' nếu matches[n] == 0
BP   = 1 if hyp_len > ref_len else exp(1 - ref_len / hyp_len)
BLEU = 100 * BP * exp( (1/4) * Σ log precision[n] )
```

## Phụ lục B: Checklist trước khi upload lên QDC

### Cập nhật giao diện 1.0.3 (2026-10-24)

- Thanh chọn ngôn ngữ áp dụng cho audio tải bằng Upfile. Dataset quadata mặc định theo hướng dịch của từng track; giao diện ghi rõ để tránh nhầm với lựa chọn audio lẻ.
- Một nút Tạm dừng / Tiếp tục. Khi yêu cầu dừng, chờ bước xử lý an toàn kết thúc; tiếp tục từ checkpoint mẫu đã lưu. Nếu dừng trước khi tạo lượt chạy (nhập audio/tải ZIP), lưu yêu cầu để thử lại bước chuẩn bị.
- Xuất kết quả chỉ bật khi có results.json và không còn tác vụ chạy, chuyển nền xanh sáng. Cho phép xuất cả kết quả một phần sau tạm dừng.
- Không ghi “đang xử lý” cho latency chưa được đo nếu giai đoạn đó chưa chạy. Giữ nguyên thuật toán dịch trong bản sửa giao diện; chẩn đoán hiệu năng tại outputs/TRANSLATION_DIAGNOSIS.md.

- [ ] `./gradlew assembleDebug` chạy sạch trên máy khác (không phụ thuộc đường dẫn cá nhân).
- [ ] APK chỉ chứa `arm64-v8a`, model nằm trong APK hoặc có đường tải rõ ràng.
- [ ] Chạy thử 3-5 mẫu trên điện thoại thật/emulator, ra đủ `results.json` và `results.csv`.
- [ ] CER/BLEU khớp `jiwer`/`sacrebleu` trên bộ mẫu kiểm chứng.
- [ ] Đã có đường lấy kết quả dự phòng (logcat + màn hình).
- [ ] Đã chốt cách nạp dataset hoặc ít nhất có chế độ attach thủ công hoạt động.
