package com.larplarpsahur;

import java.io.File;

/** Recovery eligibility must not confuse a completed result with paused work. */
public final class RunRecovery {
    private RunRecovery() {}
    public static boolean hasCheckpoint(String path) {
        if(path==null||path.isEmpty())return false;
        File run=new File(path);
        return new File(run,"run.json").isFile()&&!new File(run,"DONE").isFile();
    }
}
