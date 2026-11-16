package com.onspeak47.util;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class NpuClockTest {

    private static final long MS = 1_000_000;

    @Test
    public void highOnlyWhileGraphsRun() {
        List<String> votes = new ArrayList<>();
        NpuClock c = new NpuClock(high -> votes.add(high ? "up" : "down"), 300);

        c.tick(10_000 * MS);                       // models still loading: no run yet, no down-vote
        assertEquals(List.of(), votes);

        c.begin();                                 // already high: no vote run
        c.end(1000 * MS);
        c.tick(1299 * MS);                         // inside the burst gap
        assertEquals(List.of(), votes);
        c.tick(1300 * MS);
        c.tick(1450 * MS);                         // already low: one down-vote only
        assertEquals(List.of("down"), votes);

        c.begin();                                 // ASR step starts: up before its first run
        c.begin();                                 // MT on another thread
        c.end(2000 * MS);
        c.tick(5000 * MS);                         // MT still running
        assertEquals(List.of("down", "up"), votes);
        c.end(5100 * MS);
        c.tick(5400 * MS);
        assertEquals(List.of("down", "up", "down"), votes);
    }

    @Test
    public void failedDownVoteKeepsTheClockHigh() {
        int[] calls = {0};
        NpuClock c = new NpuClock(high -> {
            calls[0]++;
            throw new IllegalStateException("no HTP");
        }, 300);
        c.begin();
        c.end(0);
        c.tick(300 * MS);
        c.tick(600 * MS);                          // not retried
        c.begin();                                 // still high: no up-vote needed
        assertEquals(1, calls[0]);
    }
}
