import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { wakeBackendOnce, checkBackendReady } from "@/lib/backendWarmup";

describe("wakeBackendOnce", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("sends GET /health once per session", () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response("{}"));
    vi.stubGlobal("fetch", fetchMock);

    wakeBackendOnce("https://api.example");
    wakeBackendOnce("https://api.example");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe("https://api.example/health");
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: "GET" });
  });

  it("does nothing without a base URL", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    wakeBackendOnce(undefined);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("swallows network errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    expect(() => wakeBackendOnce("https://api.example")).not.toThrow();
  });
});

describe("checkBackendReady", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("is ready when /ready reports ready", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ status: "ready", models_loaded: true }), { status: 200 }),
      ),
    );
    await expect(checkBackendReady("https://api.example")).resolves.toEqual({ ready: true });
  });

  it("is not ready on not_ready, HTTP errors or network failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ status: "not_ready", models_loaded: false }), { status: 200 }),
      ),
    );
    await expect(checkBackendReady("https://api.example")).resolves.toEqual({ ready: false });

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("", { status: 503 })));
    await expect(checkBackendReady("https://api.example")).resolves.toEqual({ ready: false });

    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    await expect(checkBackendReady("https://api.example")).resolves.toEqual({ ready: false });
  });
});
