package com.larplarpsahur;

/** Text shared by the workspace's idle, running, and recoverable states. */
public final class WorkspaceUi {
    public static final String[] CODES={"vi","en","zh"};
    private WorkspaceUi() {}
    public static String targetCode(int position) {
        return position>=0&&position<CODES.length?CODES[position]:"en";
    }
    public static String language(String code) {
        if("vi".equals(code))return "Tiếng Việt";
        if("en".equals(code))return "English";
        if("zh".equals(code))return "中文";
        if("und".equals(code))return "chưa nhận diện";
        return code==null||code.isEmpty()?"—":code;
    }
    public static String pauseLabel(boolean busy,boolean requested,boolean canResume) {
        return requested?"Đang tạm dừng…":!busy&&canResume?"Tiếp tục":"Tạm dừng";
    }
    public static String latency(long millis,boolean running) {
        return millis>=0?millis+" ms":running?"đang xử lý":"—";
    }
}
