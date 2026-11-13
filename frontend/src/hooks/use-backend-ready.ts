import { useEffect, useState } from "react";
import { checkBackendReady } from "@/lib/backendWarmup";

const BASE_URL = import.meta.env.VITE_API_BASE_URL as string | undefined;

/** Poll interval while the backend is not ready. */
const POLL_INTERVAL_MS = 3_000;
/** Stop polling after this long; the banner then hides (submit still works). */
const MAX_POLL_MS = 10 * 60_000;

export type BackendReadyStatus = "checking" | "warming" | "ready" | "gave_up" | "disabled";

export interface BackendReadyState {
  status: BackendReadyStatus;
  /** Seconds since polling started (for the banner). */
  elapsedSeconds: number;
}

// Shared across mounts so navigating away and back does not reset the
// timer or re-show the banner once the backend has been seen ready.
let seenReady = false;
let pollStartedAt: number | null = null;

/**
 * Polls GET /ready until the backend reports ready.
 * Purely informational: callers must not block user actions on it.
 */
export function useBackendReady(baseUrl: string | undefined = BASE_URL): BackendReadyState {
  const [status, setStatus] = useState<BackendReadyStatus>(() => {
    if (!baseUrl) return "disabled";
    return seenReady ? "ready" : "checking";
  });
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    if (!baseUrl || seenReady) return;

    if (pollStartedAt === null) pollStartedAt = Date.now();
    const startedAt = pollStartedAt;
    let cancelled = false;
    let pollTimer: ReturnType<typeof setTimeout> | undefined;

    const tick = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1_000);

    const poll = async () => {
      const { ready } = await checkBackendReady(baseUrl);
      if (cancelled) return;
      if (ready) {
        seenReady = true;
        setStatus("ready");
        clearInterval(tick);
        return;
      }
      if (Date.now() - startedAt > MAX_POLL_MS) {
        setStatus("gave_up");
        clearInterval(tick);
        return;
      }
      setStatus("warming");
      pollTimer = setTimeout(poll, POLL_INTERVAL_MS);
    };
    void poll();

    return () => {
      cancelled = true;
      clearInterval(tick);
      if (pollTimer) clearTimeout(pollTimer);
    };
  }, [baseUrl]);

  return { status, elapsedSeconds };
}

/** Test helper: reset the module-level session state. */
export function __resetBackendReadyStateForTests(): void {
  seenReady = false;
  pollStartedAt = null;
}
