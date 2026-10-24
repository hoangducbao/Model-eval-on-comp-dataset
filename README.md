# Model eval on comp dataset

Ứng dụng Android **larplarpsahur** đánh giá pipeline **Qwen ASR → HY-MT translation** trên audio riêng lẻ hoặc dataset Qualcomm Round 3 (`Echdipi/quadata`). ASR tự nhận diện ngôn ngữ nguồn; audio riêng lẻ cho phép chọn tiếng Việt, English hoặc 中文 làm ngôn ngữ dịch.

## 1. Chuẩn bị

- Windows PowerShell, Git, JDK 21 và Android SDK Platform 34 cùng Build Tools/Platform Tools tương thích Gradle của dự án. Có thể cài SDK bằng Android Studio.
- `java`, `adb` và `ssh` có trong PATH; kiểm tra bằng `java -version`, `adb version`, `ssh -V`.
- APK chỉ chứa ABI `arm64-v8a`, Android tối thiểu API 27. **Chạy inference NPU cần thiết bị SM8850, Android 12+ và driver QNN tương thích**; không đảm bảo chạy trên mọi điện thoại/emulator.
- Internet và dung lượng trống cho model khoảng 1,85 GB, ZIP dataset khoảng 93 MB, dữ liệu giải nén và kết quả.
- Tài khoản Qualcomm Device Cloud (QDC), session còn hoạt động và tùy chọn SSH tunnel nếu cần điều khiển từ máy tính.

Repo có thư viện sherpa-onnx AAR trong `android/app/libs/`; model được tải riêng khi mở app lần đầu. Không cần `qdc_build_run.py`.

## 2. Clone, build và xuất APK

Repo riêng tư: đăng nhập GitHub trước khi clone, ví dụ `gh auth login`.

```powershell
git clone https://github.com/hoangducbao/Model-eval-on-comp-dataset.git
Set-Location .\Model-eval-on-comp-dataset\android

# Sửa theo đường dẫn JDK và Android SDK thực tế trên máy bạn.
$env:JAVA_HOME = 'C:\Program Files\Java\jdk-21'
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
$env:Path = "$env:JAVA_HOME\bin;$env:ANDROID_HOME\platform-tools;$env:Path"

.\gradlew.bat :app:assembleDebug :app:testDebugUnitTest :app:lintDebug --console=plain
if ($LASTEXITCODE -ne 0) { throw 'Build hoặc kiểm thử thất bại.' }

New-Item -ItemType Directory -Path ..\outputs -Force | Out-Null
Copy-Item .\app\build\outputs\apk\debug\app-debug.apk ..\outputs\larplarpsahur-debug.apk
Get-FileHash ..\outputs\larplarpsahur-debug.apk -Algorithm SHA256
```

APK gốc: `android/app/build/outputs/apk/debug/app-debug.apk`. Đây là APK debug để thử nghiệm cloud device, không phải bản ký phát hành lên Play Store. `android/local.properties` là cấu hình riêng của máy, không được commit; nếu file này tồn tại, đảm bảo `sdk.dir` trỏ đúng SDK.

Có thể upload APK qua giao diện cài ứng dụng của QDC, hoặc dùng `adb install` sau khi mở tunnel ở bước tiếp theo.

## 3. SSH tunnel QDC

**Tất cả lệnh trong README chạy ở PowerShell trên máy tính**, không chạy trong ADB Console của trang QDC. ADB Console QDC đã là shell trên điện thoại, không phải nơi chạy SSH hay tải file về Windows.

### 3.1. Tải và bảo vệ private key

Khởi động session QDC có SSH tunnel, tải key PEM được cấp cho session đó. Ví dụ lưu trong `D:\StudioProjects\key`. Dùng đúng tên file vừa tải; không tái sử dụng tên/session cũ.

Đặt đường dẫn đầy đủ của **file PEM**, không phải thư mục:

```powershell
$keyPath = 'D:\StudioProjects\key\TEN_FILE_KEY_HIEN_TAI.pem'
if (-not (Test-Path -LiteralPath $keyPath -PathType Leaf)) {
    throw "Không tìm thấy file key: $keyPath"
}

# Chỉ tài khoản Windows đang chạy lệnh được đọc private key.
$accountSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$keyAcl = Get-Acl -LiteralPath $keyPath
$keyAcl.SetAccessRuleProtection($true, $false)
foreach ($entry in @($keyAcl.Access)) {
    [void]$keyAcl.RemoveAccessRuleSpecific($entry)
}
$keyAcl.AddAccessRule([System.Security.AccessControl.FileSystemAccessRule]::new(
    $accountSid, 'Read', 'Allow'))
Set-Acl -LiteralPath $keyPath -AclObject $keyAcl
icacls $keyPath
ssh-keygen -y -f $keyPath | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'SSH chưa đọc được key.' }
```

Nếu sửa quyền bị `Access denied`, mở PowerShell bằng **Run as administrator** với cùng tài khoản rồi thử lại. Không chia sẻ/commit PEM. `.gitignore` loại trừ PEM, private key, keystore và APK.

### 3.2. Mở tunnel — cửa sổ PowerShell A

Trong QDC, ô **Private Key File Path** chỉ nhập đường dẫn PEM (không nhập `ssh -i`). Chọn cổng local **5038** để tránh xung đột với ADB local thường dùng 5037. Sao chép lệnh QDC tạo cho session hiện tại.

Mẫu tương đương dưới đây dùng placeholder: thay `SESSION_HOST_FROM_QDC` bằng host trong lệnh QDC, ví dụ dạng `saXXXXXX.sa.svc.cluster.local`. Không sao chép host của session đã hết hạn.

```powershell
$keyPath = 'D:\StudioProjects\key\TEN_FILE_KEY_HIEN_TAI.pem'
$qdcHost = 'SESSION_HOST_FROM_QDC'

ssh -i "$keyPath" -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -N -L "127.0.0.1:5038:${qdcHost}:5037" sshtunnel@ssh.qdc.qualcomm.com
```

Lần đầu SSH có thể hỏi xác nhận host key: đối chiếu thông tin máy chủ với hướng dẫn QDC trước khi chấp nhận. **Giữ cửa sổ A mở** suốt khi dùng ADB; lệnh đứng chờ là bình thường. Dòng Welcome riêng lẻ chưa chứng minh xác thực thành công: không được có lỗi `Permission denied` hoặc `bind` sau đó.

### 3.3. Kết nối ADB — cửa sổ PowerShell B

Dùng `-H`/`-P` để chỉ rõ ADB server đi qua tunnel. `5038` là cổng trên máy tính; `5037` trong đích tunnel là ADB server phía QDC. Không dùng `adb connect 127.0.0.1:5038` vì đây là cổng ADB server, không phải cổng adbd của thiết bị.

```powershell
adb -H 127.0.0.1 -P 5038 devices -l

# Thay bằng serial có trạng thái device trong danh sách phía trên.
$serial = 'SERIAL_CLOUD_DEVICE'
adb -H 127.0.0.1 -P 5038 -s $serial shell getprop ro.product.model

# Chạy từ thư mục gốc repo, hoặc thay APK bằng đường dẫn tuyệt đối.
adb -H 127.0.0.1 -P 5038 -s $serial install -r .\outputs\larplarpsahur-debug.apk
adb -H 127.0.0.1 -P 5038 -s $serial shell am start -n com.larplarpsahur/.MainActivity
```

Nếu cài bản build từ máy khác báo `INSTALL_FAILED_UPDATE_INCOMPATIBLE`, chữ ký debug khác nhau. Xuất/lấy kết quả trước khi quyết định gỡ app; gỡ app sẽ xóa dữ liệu riêng, bao gồm model đã tải.

## 4. Sử dụng app

1. Mở app, kết nối Internet và chờ tải/kiểm tra model. Có tiến độ, kiểm tra SHA-256 và thử lại khi lỗi.
2. **Audio riêng lẻ:** chọn ngôn ngữ đích ở thanh trên, nhấn **Upfile**, chọn audio. Pipeline bắt đầu sau khi chọn.
3. **Dataset:** nhấn **Lấy dữ liệu Qualcomm**. App tải duy nhất ZIP Round 3 từ Hugging Face, kiểm tra hash, giải nén, ghép reference theo ID và xử lý. Bỏ qua các file ngoài ZIP.
4. Dataset dùng hướng dịch của từng track, không lấy lựa chọn ngôn ngữ cho Upfile. Ví dụ track VM dịch sang tiếng Việt dù thanh chọn audio lẻ đang là English.
5. Box trên là bản chép lời ASR và ngôn ngữ phát hiện; box dưới là bản dịch. Dòng phía trên hiển thị mẫu hiện tại và latency ASR/MT.
6. **Tạm dừng / Tiếp tục** là một nút. Chờ bước xử lý an toàn kết thúc trước khi tiếp tục; đã lưu checkpoint theo mẫu. Nếu dừng lúc chuẩn bị/tải ZIP, bước đó có thể phải tải hoặc nhập lại.
7. **Xuất kết quả** sáng xanh khi có kết quả và app không bận. Nhấn nút, chọn thư mục **Download** trên điện thoại và lưu tên `larplarpsahur-results.zip`. Có thể xuất kết quả một phần sau tạm dừng.

Dataset chạy ASR cho toàn bộ mẫu rồi mới chạy translation để hạn chế bộ nhớ model. `mt_ms` không bao gồm thời gian nạp model hay ghi kết quả. Bản 1.0.3 vẫn dùng giới hạn sinh 256 token; hiện nhánh decoder nhỏ không được chọn với giới hạn này. Không coi kiểm thử build/JVM là bằng chứng hiệu năng inference trên thiết bị.

Nguồn tải model:

- [Qwen — release v2.2.0-npu](https://github.com/DS-PNQ/OnSpeak47-npu-models/releases/tag/v2.2.0-npu).
- [HY-MT — release v2.1.0-npu](https://github.com/DS-PNQ/OnSpeak47-npu-models/releases/tag/v2.1.0-npu).
- Tokenizer HY-MT lấy từ Tencent/Hy-MT2-1.8B tại revision và SHA-256 cố định trong `android/app/build.gradle`.
- [Dataset Echdipi/quadata](https://huggingface.co/datasets/Echdipi/quadata/tree/main): ZIP `Round 3-Bộ dữ liệu mẫu-20261024T014123Z-1-001.zip`, revision `fed18c73b4350e07a02124df4f4c716a95e26743`.

## 5. Pull ZIP kết quả về máy tính

Sau khi **Xuất kết quả** vào Download trên điện thoại, giữ SSH tunnel ở cửa sổ A; chạy ở PowerShell B:

```powershell
adb -H 127.0.0.1 -P 5038 -s $serial shell ls -lah /sdcard/Download

$resultsDirectory = Join-Path $env:USERPROFILE 'Downloads\qdc-results'
New-Item -ItemType Directory -Path $resultsDirectory -Force | Out-Null

# Sửa tên ZIP nếu Android đã thêm (1), (2) hoặc bạn đặt tên khác.
adb -H 127.0.0.1 -P 5038 -s $serial pull '/sdcard/Download/larplarpsahur-results.zip' "$resultsDirectory\larplarpsahur-results.zip"
if ($LASTEXITCODE -ne 0) { throw 'Chưa lấy được ZIP kết quả.' }

Get-Item -LiteralPath "$resultsDirectory\larplarpsahur-results.zip"
Get-FileHash -LiteralPath "$resultsDirectory\larplarpsahur-results.zip" -Algorithm SHA256
Expand-Archive -LiteralPath "$resultsDirectory\larplarpsahur-results.zip" -DestinationPath "$resultsDirectory\extracted" -Force
```

Kết quả nằm ở `%USERPROFILE%\Downloads\qdc-results`. `adb pull` chỉ sao chép, không xóa bản trên điện thoại. Lấy file về trước khi kết thúc session QDC.

Nếu chưa xuất ZIP, kết quả gốc nằm trong `/sdcard/Android/data/com.larplarpsahur/files/results/<run>/`. Một số thiết bị cho phép lấy trực tiếp qua ADB:

```powershell
adb -H 127.0.0.1 -P 5038 -s $serial shell ls /sdcard/Android/data/com.larplarpsahur/files/results
# Thay RUN_DIRECTORY bằng đúng tên thư mục vừa liệt kê.
adb -H 127.0.0.1 -P 5038 -s $serial pull '/sdcard/Android/data/com.larplarpsahur/files/results/RUN_DIRECTORY' "$resultsDirectory"
```

ZIP gồm `results.json`, `results.csv`, transcripts/translations, `eval_asr.csv`, `eval_mt.csv`, `eval_summary.json` và marker `DONE`, `CANCELLED` hoặc `FAILED`. `DONE` chỉ cho biết lượt chạy đã kết thúc; kiểm tra `error_samples` trong summary để biết mẫu lỗi. Audio/model không được đóng vào ZIP. Không có reference thì các metric tương ứng có thể là null/N/A.

## 6. Lỗi thường gặp

- **Identity file not accessible:** sai đường dẫn/tên file PEM, hoặc file đã bị xóa. Kiểm tra `Test-Path -LiteralPath $keyPath`.
- **UNPROTECTED PRIVATE KEY FILE:** chạy bước siết quyền ở trên; đảm bảo `-i` trỏ vào file PEM, không trỏ vào thư mục.
- **Permission denied (publickey):** key không được máy chủ chấp nhận; kiểm tra key và lệnh tunnel thuộc session hiện tại, session chưa hết hạn. Welcome không phải xác nhận đăng nhập.
- **bind ... Permission denied / Address already in use:** lỗi cổng Windows, không phải lỗi key. Kiểm tra `netstat -ano | findstr :5038`; nếu bị dùng/chặn, chọn cổng local khác trong QDC và đổi đồng bộ `-L` lẫn `adb -P`. Không dừng tiến trình lạ chỉ vì thấy PID.
- **adb devices trống hoặc nhầm điện thoại cá nhân:** luôn dùng đúng `-H 127.0.0.1 -P 5038` và serial cloud. Lệnh `adb devices` không có tham số có thể trỏ ADB local.
- **pull: No such file:** liệt kê Download để lấy đúng tên ZIP; file vẫn nằm trên điện thoại cho đến khi `adb pull` thành công.
- **Tạm dừng chưa phản hồi ngay:** đang chờ bước native inference kết thúc; nút không dừng tức thì giữa mọi token.

## 7. Cấu trúc và kiểm thử

- `android/`: Gradle project, mã Java, UI, pipeline và thư viện AAR.
- `android_asr_mt_benchmark_plan.md`: kế hoạch/định dạng đánh giá.
- `checkpoint.md`: trạng thái các tác vụ triển khai.
- `outputs/`, `work/`, model, kết quả đánh giá và cấu hình SDK cục bộ không được đưa lên Git.

Bản 1.0.3 đã vượt qua assembleDebug, lintDebug và 7 unit test. Chạy kiểm thử thực tế trên SM8850 để xác minh QNN, latency và chất lượng dịch. Các model/dataset và thư viện bên thứ ba tuân theo điều khoản của nguồn cung cấp tương ứng.
