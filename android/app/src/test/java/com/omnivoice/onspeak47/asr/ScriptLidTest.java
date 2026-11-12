package com.omnivoice.onspeak47.asr;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import org.junit.Test;

/** Same vectors as backend/streaming_asr/script_lid.py's self-check (parity). */
public class ScriptLidTest {
    @Test
    public void detect_readsLanguageOffScriptAndCase() {
        assertEquals("vi", ScriptLid.detect("hôm nay chúng ta sẽ xem xét ngân sách"));
        assertEquals("en", ScriptLid.detect("THE SECOND RELEASE FIXES THE LEAK"));
        assertEquals("zh", ScriptLid.detect("我们需要在下个月之前完成"));
        assertNull(ScriptLid.detect("WE NEED TO FINISH THE 测试 BEFORE NEXT MONTH"));
        assertEquals("vi", ScriptLid.detect("em thấy cái MEETING này hơi dài"));
        assertEquals("zh", ScriptLid.detect("这个 PROJECT 的 DEADLINE 是下周"));
        assertNull(ScriptLid.detect("cái này tiếng trung gọi là 预算 đúng không"));
        assertEquals("en", ScriptLid.detect("THE MEETING IS ABOUT ngân sách"));
        assertNull(ScriptLid.detect("THE MEETING IS ABOUT ngân sách quý này"));
        assertNull(ScriptLid.detect("我们 下周 开会 sau đó"));
        assertNull(ScriptLid.detect(""));
        assertNull(ScriptLid.detect("123"));
        assertNull(ScriptLid.detect(null));
    }

    @Test
    public void label_needsDiacriticForVietnamese() {
        assertNull(ScriptLid.label("o a e o"));
        assertEquals("vi", ScriptLid.label("xin chào các bạn"));
        assertEquals("en", ScriptLid.label("LET US START"));
    }

    /** D:\apk32.txt: `ba anh thì là người thu nhập chính` -> "the three of them", `ai cũng` -> "AI". */
    @Test
    public void viMtForm_dadAndRowOpener() {
        assertEquals("Đúng là bố anh thì là người", ScriptLid.viMtForm("Đúng là ba anh thì là người"));
        assertEquals("Bố mẹ anh, bố của em, sợ bố anh, bố em năm nay",
                ScriptLid.viMtForm("Ba mẹ anh, ba của em, sợ ba anh, ba em năm nay"));
        for (String three : new String[]{"Có ba anh em", "Ba em học sinh mới", "Ba mẹ con tôi", "Ba người",
                "Có ba em thì", "Hai ba em là"}) {
            assertEquals(three, ScriptLid.viMtForm(three));
        }
        assertEquals("Mọi người đều phải ngồi", ScriptLid.viMtForm("ai cũng phải ngồi"));
        assertEquals("Ai cũng được", ScriptLid.viMtForm("ai cũng được"));
        assertEquals("Đó.", ScriptLid.viMtForm("đó."));
        for (String keep : new String[]{"an@gmail.com là email", "iPhone mới", "abc.com nhé", "123 người", ""}) {
            assertEquals(keep, ScriptLid.viMtForm(keep));
        }
    }

    @Test
    public void lowerEnglish_lowercasesCapsWordsOnly() {
        assertEquals("I saw the meeting ở Hà Nội",
                ScriptLid.lowerEnglish("I SAW THE MEETING ở Hà Nội"));
    }

    @Test
    public void lowerEnglish_sentenceCaseAndArticleA() {
        assertEquals("Get down from a very happy G2. Okay, I'm sure I'll ask a post-match question.",
                ScriptLid.lowerEnglish(
                        "GET DOWN FROM A VERY HAPPY G2. OKAY, I'M SURE I'LL ASK A POST-MATCH QUESTION."));
    }

    @Test
    public void foreign_findsNonTargetWords() {
        assertEquals("vi", ScriptLid.foreign("MAYBE WE CAN HANG UP LATER bạn nghĩ sao", "en"));
        assertNull(ScriptLid.foreign("LET US START", "en"));
        assertNull(ScriptLid.foreign("", "vi"));
    }

    /** D:\apk34*mail.txt: Qwen's addresses and phones as it really wrote them. */
    @Test
    public void writtenForm_apk34Addresses() {
        assertEquals("我的邮箱是info@mandarinhq.com。", ScriptLid.writtenForm(ScriptLid.lowerEnglish("我的邮箱是INFO@MANDARINHQ.COM。")));
        assertEquals("也可以发送邮件至ronwenan.tech@gmail.com。", ScriptLid.writtenForm("也可以发送邮件至RONWENAN.点TECH AT GMAIL.点COM。"));
        assertEquals("EMAIL AT david.miller.tech@gmail.com.", ScriptLid.writtenForm("EMAIL AT DAVID.MILLER@TECH@GMAIL.COM."));
        assertEquals("support.services@gmail.com, OR contact.customercare@gmail.com.",
                ScriptLid.writtenForm("SUPPORT@SERVICES@GMAIL.COM, OR CONTACT@CUSTOMERCARE@GMAIL.COM."));
        assertEquals("电话+8613800138000，或+8613588889999。", ScriptLid.writtenForm("电话加八六一三八零零一三八零零零，或加八六一三五八八八八九九九九。"));
        assertEquals("gọi +84901234567", ScriptLid.writtenForm("gọi cộng tám bốn chín không một hai ba bốn năm sáu bảy"));
        assertEquals("再加123456", ScriptLid.writtenForm("再加一二三四五六"));   // too short for a phone
        java.util.function.UnaryOperator<String> plain = ScriptLid.pinyin;
        // As ICU writes it: syllables spaced.
        ScriptLid.pinyin = s -> s.replace("如安文安", "ru an wen an").replace("服务", "fu wu").replace("科辅", "ke fu").replace("自勋", "zi xun");
        try {
            assertEquals("发送邮件至ruanwenan.tech@gmail.com，", ScriptLid.writtenForm("发送邮件至如安文安点TECH AT GMAIL点COM，"));
            assertEquals("fuwu.kefu@gmail.com，或kefu.zixun@gmail.com。", ScriptLid.writtenForm("服务点科辅AT GMAIL点COM，或科辅点自勋AT GMAIL点COM。"));
        } finally {
            ScriptLid.pinyin = plain;
        }
    }

    @Test
    public void keepEmails_sourceAddressWins() {
        assertEquals("Đã gửi đến anderson.one@gmail.com. Nếu", ScriptLid.keepEmails(
                "发票已寄到anderson.one@gmail.com，有问题", "Đã gửi đến Anderson_one@gmail .com. Nếu"));
        assertEquals("x@y.z", ScriptLid.keepEmails("không có email", "x@y.z"));
    }

    @Test
    public void writtenForm_spokenEmailsAndDigitRuns() {
        assertEquals("gửi file cho tôi qua pos@gmail.com nhé",
                ScriptLid.writtenForm("gửi file cho tôi qua pos a còng gmail chấm com nhé"));
        assertEquals("Please email me at pos@gmail.com before 5 p.m.",
                ScriptLid.writtenForm("Please email me at pos at gmail dot com before 5 p.m."));
        assertEquals("pos@gmail.com", ScriptLid.writtenForm(ScriptLid.lowerEnglish("POS AT GMAIL DOT COM")));
        assertEquals("john.smith@company.com.vn",
                ScriptLid.writtenForm("john dot smith at company dot com dot vn"));
        assertEquals("Số của tôi là 0912345678.",
                ScriptLid.writtenForm("Số của tôi là không chín một hai ba bốn năm sáu bảy tám."));
        assertEquals("0912 345 678", ScriptLid.writtenForm("Không chín một hai, ba bốn năm, sáu bẩy tám"));
        // Qwen glues Latin to Chinese (device 2026-10-30: "ANDERSON dot one@gmail.com")
        assertEquals("发票已寄到anderson.one@gmail.com，订单号AB1234CD", ScriptLid.writtenForm(
                ScriptLid.lowerEnglish("发票已寄到ANDERSON DOT ONE AT GMAIL DOT COM，订单号AB1234CD")));
        // device 2026-10-30, one spoken address heard five ways
        assertEquals("nguyenvanan.tech@gmail.com rất hân hạnh", ScriptLid.writtenForm(
                "nguyễn văn an chấm techacong gmail com rất hân hạnh"));
        assertEquals("nguyenvanan.tech@gmail.com", ScriptLid.writtenForm("nguyễn văn an chấm tech a cổng gmail com"));
        assertEquals("nguyenvanan.tic@gmail.com cho", ScriptLid.writtenForm("nguyễnvanan.tic accomgmail.com cho"));
        assertEquals("nguyenvanan.tac@gmail.com", ScriptLid.writtenForm("nguyễnvanan.tacacam gmail.com"));
        assertEquals("vanan.tich@gmail.com cho.", ScriptLid.writtenForm("Văn an chấm tích a cộng g mươi chấm com cho."));
        assertEquals("gửi về nguyenvanan.tich@gmail.com", ScriptLid.writtenForm(
                "gửi về Nguyên vanan chấm tích a cổng gừmêu chấm com"));
        assertEquals("nguyenvanan.tich@gmail.com caoduop.dichvu@gmail.com", ScriptLid.writtenForm(
                "Nguyên vanan chấm tích a cổng gừmêu chấm com cao dướp chấm dịch vụ a cổng gừmêu chấm com"));
        assertEquals("liên hệ dichvu@fpt.com.vn", ScriptLid.writtenForm("liên hệ dịch vụ a còng fpt chấm com chấm vn"));
        assertEquals("0901234567 0987654321 hoặc 0918889999", ScriptLid.writtenForm(
                "không chín không một hai ba bốn năm sáu bảy không chín tám bảy sáu năm bốn ba hai một"
                        + " hoặc không chín một tám tám tám chín chín chín chín"));
        assertEquals("nguyenvanantach@gmail.com caoduopdichvu@gmail.com hoặc lienhe.khachtrang@gmail.com",
                ScriptLid.writtenForm("nguyễnvanan tách a còng gmail com cao dướp dịch vụ acong gmail com"
                        + " hoặc liên hệ chấm khách trăng acong gmail com"));
        String[][] written = {
                {"tăng mười hai phẩy năm phần trăm", "tăng 12,5%"}, {"giảm ba mươi phần trăm", "giảm 30%"},
                {"một trăm phần trăm", "100%"}, {"năm trăm nghìn đồng", "500.000 đồng"},
                {"hai triệu ba trăm nghìn đồng", "2.300.000 đồng"}, {"ba triệu hai đồng", "3.200.000 đồng"},
                {"ba phẩy năm triệu đồng", "3,5 triệu đồng"}, {"một trăm lẻ năm usd", "105 usd"},
                {"hai nghìn không trăm hai mươi lăm đồng", "2.025 đồng"},
                {"một tỷ hai trăm triệu đồng", "1.200.000.000 đồng"},
                {"ngày ba tháng tư năm hai nghìn không trăm hai mươi lăm", "ngày 3 tháng 4 năm 2025"},
                {"Ngày mười lăm tháng mười hai năm hai không hai lăm", "Ngày 15 tháng 12 năm 2025"},
                {"tháng ba năm hai nghìn", "tháng 3 năm 2000"}, {"năm hai nghìn lẻ năm", "năm 2005"},
                {"lúc mười bốn giờ ba mươi phút", "lúc 14h30"}, {"hai giờ rưỡi chiều", "2h30 chiều"},
                {"tám giờ chín phút", "8h09"}, {"mười giờ mười lăm", "10h15"},
                {"Call oh nine one two three four five six seven eight", "Call 0912345678"},
                {"double five one two three four", "551234"}, {"triple nine one two three four", "9991234"},
                {"请拨打零九一二三四五六七八", "请拨打0912345678"}, {"可以拨打幺二三零六", "可以拨打12306"},
                {"百分之七十", "70%"}, {"百分之三点五", "3.5%"}, {"一千二百九十九元", "1299元"}, {"三点五万元", "3.5万元"},
                {"五万元", "5万元"}, {"一百二十万元", "120万元"}, {"三万五千元", "35000元"}, {"十三点五美元", "13.5美元"},
                {"二零二四年三月十五日", "2024年3月15日"}, {"二〇二四年", "2024年"}, {"两千零二十四年", "2024年"},
                {"五点零二分", "5:02"}, {"下午三点半", "下午3:30"}, {"十点十五分", "10:15"},
                {"今天气温摄氏二十八度", "今天气温28°C"}, {"三十八摄氏度", "38°C"},
                {"发票寄到anderson艾特gmail点com", "发票寄到anderson@gmail.com"}};
        for (String[] w : written) assertEquals(w[1], ScriptLid.writtenForm(w[0]));
        for (String keep : new String[]{"một vài người", "năm ba người", "mười mấy người", "hai đồng nghiệp",
                "ba đô thị", "ba giờ hai người", "hai năm trước", "năm ngoái", "ba tháng tư năm", "one of the five",
                "a hundred times", "我有一个问题", "看一下", "三心二意", "一年后", "星期一", "十一号五楼", "九点五分",
                "一点", "一块去", "二元一次方程", "二三年开始", "千万别去", "百分百"}) {
            assertEquals(keep, ScriptLid.writtenForm(keep));
        }
        for (String keep : new String[]{"nguyễn văn an chấm dứt hợp đồng", "tôi a cổng trường", "cổng gmail",
                "hai ba bốn năm rồi", "I'm at home dot com", "look at the dot",
                "cuộc họp lúc ba giờ", "mình gặp nhau lúc năm giờ chiều", "pos@gmail.com"}) {
            assertEquals(keep, ScriptLid.writtenForm(keep));
        }
    }
}
