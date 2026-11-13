/**
 * Early wake-up for the GPU backend (Hugging Face Space).
 *
 * A sleeping Space only starts booting when it receives a request. Sending a
 * cheap GET /health as soon as the site loads moves most of the cold start
 * behind the time the user spends reading the page and picking an image.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL as string | undefined;
const WAKE_KEY = "dfd:backend-wake-sent-at";

/**
 * Fire-and-forget GET /health, at most once per browser session.
 * Never throws and never blocks rendering.
 */
export function wakeBackendOnce(baseUrl: string | undefined = BASE_URL): void {
  if (!baseUrl || typeof window === "undefined" || typeof fetch === "undefined") return;

  try {
    if (window.sessionStorage.getItem(WAKE_KEY)) return;
    window.sessionStorage.setItem(WAKE_KEY, String(Date.now()));
  } catch {
    // sessionStorage unavailable (privacy mode); still send the ping.
  }

  // Simple GET without custom headers: no CORS preflight. Even if the
  // response is blocked or fails, the request still reaches the Space.
  fetch(`${baseUrl}/health`, { method: "GET", cache: "no-store" }).catch(() => {});
}

export interface ReadyCheckResult {
  ready: boolean;
}

/** One GET /ready call with a timeout. Network errors count as not ready. */
export async function checkBackendReady(
  baseUrl: string,
  timeoutMs = 10_000,
): Promise<ReadyCheckResult> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${baseUrl}/ready`, {
      method: "GET",
      cache: "no-store",
      signal: controller.signal,
    });
    if (!res.ok) return { ready: false };
    const data = (await res.json()) as { status?: string; models_loaded?: boolean };
    return { ready: data.status === "ready" || data.models_loaded === true };
  } catch {
    return { ready: false };
  } finally {
    clearTimeout(timer);
  }
}
