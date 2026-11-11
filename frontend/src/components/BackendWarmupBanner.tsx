import { Loader2 } from "lucide-react";
import { useBackendReady } from "@/hooks/use-backend-ready";

/** Do not flash the banner for a backend that answers quickly. */
const SHOW_AFTER_SECONDS = 2;

/**
 * Small notice shown while the GPU backend is still booting.
 * Informational only: it never disables uploading or submitting.
 */
const BackendWarmupBanner = () => {
  const { status, elapsedSeconds } = useBackendReady();

  const visible =
    status === "warming" || (status === "checking" && elapsedSeconds >= SHOW_AFTER_SECONDS);
  if (!visible) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      className="rounded-lg border border-sky-500/30 bg-sky-500/10 px-4 py-3 flex items-start gap-2.5 text-xs text-muted-foreground"
    >
      <Loader2 className="w-4 h-4 text-sky-500 flex-shrink-0 mt-0.5 animate-spin" aria-hidden="true" />
      <p>
        <span className="font-medium text-foreground">Warming up the GPU server…</span>{" "}
        <span className="tabular-nums">{elapsedSeconds}s</span>. You can still submit an image; the
        first analysis will start as soon as the server is ready.
      </p>
    </div>
  );
};

export default BackendWarmupBanner;
