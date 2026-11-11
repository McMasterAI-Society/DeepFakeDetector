/**
 * Admin dashboard API client. Uses a bearer token kept in sessionStorage
 * (never cookies), so it is dropped when the tab is closed.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL as string | undefined;
const TOKEN_KEY = "dfd_admin_token";

export class AdminAuthError extends Error {}

export interface SubmodelSummary {
  pred?: "real" | "fake";
  prob_fake?: number;
  contribution_percentage?: number | null;
}

export interface FeedbackInfo {
  user_label: "ai" | "real" | "unsure";
  user_knows_truth: boolean;
  verdict_correct: "yes" | "no" | "unsure";
  source_note?: string | null;
  created_at?: string;
}

export interface AdminPrediction {
  id: string;
  created_at: string;
  image_sha256: string;
  has_image: boolean;
  width?: number | null;
  height?: number | null;
  format?: string | null;
  final_prob: number;
  final_pred: "real" | "fake";
  fusion_used: boolean;
  submodels: Record<string, SubmodelSummary>;
  manifest_hash?: string | null;
  variant: string;
  latency_ms?: number | null;
  feedback: FeedbackInfo | null;
}

export interface ExplainEntry {
  heatmap_base64?: string;
  explainability_type?: "grad_cam" | "attention_rollout";
  focus_summary?: string;
}

export interface LlmInsight {
  key_finding: string;
  what_model_saw: string;
  important_regions: string[];
  confidence_qualifier: string;
}

export interface AdminPredictionDetail extends AdminPrediction {
  explain: Record<string, ExplainEntry>;
  llm: Record<string, LlmInsight>;
  variant_info?: Record<string, unknown> | null;
}

export interface RateRow {
  predictions: number;
  feedback: number;
  labelled: number;
  agreed: number;
  agreement_rate: number | null;
}

export interface AdminStats {
  days: number;
  since: string;
  uncertain_band: [number, number];
  totals: RateRow & { with_image: number; predicted_fake: number; uncertain: number };
  per_day: (RateRow & { date: string })[];
  confusion: { tp: number; fp: number; tn: number; fn: number };
  per_variant: (RateRow & {
    variant: string;
    uncertain: number;
    uncertain_rate: number | null;
    mean_prob_fake: number | null;
  })[];
  calibration: { bin: number; n: number; mean_prob_fake: number; frac_ai: number }[];
}

export interface PredictionFilters {
  disagreement?: boolean;
  uncertain?: boolean;
  has_feedback?: boolean;
  labelled?: boolean;
  has_image?: boolean;
  variant?: string;
  date_from?: string;
  date_to?: string;
}

export const getToken = (): string | null => sessionStorage.getItem(TOKEN_KEY);
export const setToken = (token: string) => sessionStorage.setItem(TOKEN_KEY, token);
export const clearToken = () => sessionStorage.removeItem(TOKEN_KEY);

const requireBase = (): string => {
  if (!BASE_URL) throw new Error("API base URL not configured (VITE_API_BASE_URL).");
  return BASE_URL;
};

const filtersToParams = (filters: PredictionFilters, extra: Record<string, string> = {}) => {
  const params = new URLSearchParams(extra);
  Object.entries(filters).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") return;
    params.set(key, String(value));
  });
  return params;
};

async function adminFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const token = getToken();
  if (!token) throw new AdminAuthError("Not logged in");
  const res = await fetch(`${requireBase()}${path}`, {
    ...init,
    credentials: "omit",
    headers: { ...(init.headers || {}), Authorization: `Bearer ${token}` },
  });
  if (res.status === 401) {
    clearToken();
    throw new AdminAuthError("Session expired. Please log in again.");
  }
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`);
  }
  return res;
}

export async function login(password: string): Promise<void> {
  const res = await fetch(`${requireBase()}/admin/login`, {
    method: "POST",
    credentials: "omit",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password }),
  });
  if (res.status === 401) throw new Error("Wrong password.");
  if (res.status === 429) {
    const retry = res.headers.get("Retry-After");
    throw new Error(`Too many attempts. Try again in ${retry ?? "a few"} seconds.`);
  }
  if (res.status === 503) throw new Error("Admin is not configured on the server.");
  if (!res.ok) throw new Error(`Login failed: ${res.status}`);
  const data = await res.json();
  setToken(data.access_token);
}

export async function checkSession(): Promise<{ expires_at: number; retention_days: number }> {
  return (await adminFetch("/admin/session")).json();
}

export async function listPredictions(
  filters: PredictionFilters,
  cursor?: string | null,
  limit = 50
): Promise<{ items: AdminPrediction[]; next_cursor: string | null }> {
  const params = filtersToParams(filters, { limit: String(limit) });
  if (cursor) params.set("cursor", cursor);
  return (await adminFetch(`/admin/predictions?${params.toString()}`)).json();
}

export async function getPrediction(id: string): Promise<AdminPredictionDetail> {
  return (await adminFetch(`/admin/predictions/${encodeURIComponent(id)}`)).json();
}

export async function getImageObjectUrl(id: string, thumb?: number): Promise<string> {
  const q = thumb ? `?thumb=${thumb}` : "";
  const res = await adminFetch(`/admin/predictions/${encodeURIComponent(id)}/image${q}`);
  return URL.createObjectURL(await res.blob());
}

export async function deletePrediction(id: string): Promise<void> {
  await adminFetch(`/admin/predictions/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export async function getStats(days = 30): Promise<AdminStats> {
  return (await adminFetch(`/admin/stats?days=${days}`)).json();
}

export async function downloadCsv(filters: PredictionFilters): Promise<void> {
  const params = filtersToParams(filters);
  const res = await adminFetch(`/admin/export.csv?${params.toString()}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `predictions-${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
