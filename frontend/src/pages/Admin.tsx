import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Download,
  ImageOff,
  Loader2,
  LogOut,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import HeatmapOverlay from "@/components/HeatmapOverlay";
import {
  AdminAuthError,
  type AdminPrediction,
  type AdminPredictionDetail,
  type AdminStats,
  type PredictionFilters,
  checkSession,
  clearToken,
  deletePrediction,
  downloadCsv,
  getImageObjectUrl,
  getPrediction,
  getStats,
  getToken,
  listPredictions,
  login,
} from "@/lib/adminApi";

const pct = (v: number | null | undefined, digits = 0) =>
  v === null || v === undefined ? "—" : `${(v * 100).toFixed(digits)}%`;

const fmtDate = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
};

const useNoIndex = () => {
  useEffect(() => {
    const meta = document.createElement("meta");
    meta.name = "robots";
    meta.content = "noindex, nofollow";
    document.head.appendChild(meta);
    const prevTitle = document.title;
    document.title = "Admin · DeepFake Detector";
    return () => {
      meta.remove();
      document.title = prevTitle;
    };
  }, []);
};

/** Image from an authenticated endpoint (img tags cannot send bearer tokens). */
const AuthImage = ({
  id,
  thumb,
  className,
  onUrl,
}: {
  id: string;
  thumb?: number;
  className?: string;
  onUrl?: (url: string | null) => void;
}) => {
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let objectUrl: string | null = null;
    let cancelled = false;
    getImageObjectUrl(id, thumb)
      .then((u) => {
        if (cancelled) {
          URL.revokeObjectURL(u);
          return;
        }
        objectUrl = u;
        setUrl(u);
        onUrl?.(u);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      onUrl?.(null);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, thumb]);

  if (failed) return <ImageOff className="w-4 h-4 text-muted-foreground" />;
  if (!url) return <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />;
  return <img src={url} alt="Stored upload" className={className} />;
};

const VerdictBadge = ({ pred }: { pred: "real" | "fake" }) =>
  pred === "fake" ? (
    <Badge variant="destructive" className="gap-1">
      <ShieldAlert className="w-3 h-3" /> AI
    </Badge>
  ) : (
    <Badge variant="secondary" className="gap-1">
      <ShieldCheck className="w-3 h-3" /> Real
    </Badge>
  );

const FeedbackBadge = ({ item }: { item: AdminPrediction }) => {
  const fb = item.feedback;
  if (!fb) return <span className="text-xs text-muted-foreground">—</span>;
  const label = fb.user_label === "ai" ? "AI" : fb.user_label === "real" ? "Real" : "Unsure";
  const tone =
    fb.verdict_correct === "yes"
      ? "text-success"
      : fb.verdict_correct === "no"
      ? "text-destructive"
      : "text-muted-foreground";
  return (
    <span className={`text-xs font-medium ${tone}`}>
      {label}
      {fb.verdict_correct !== "unsure" && ` (${fb.verdict_correct === "yes" ? "agrees" : "disagrees"})`}
    </span>
  );
};

// ------------------------------------------------------------------ login

const LoginForm = ({ onLoggedIn }: { onLoggedIn: () => void }) => {
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(password);
      setPassword("");
      onLoggedIn();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <form onSubmit={submit} className="w-full max-w-sm rounded-xl border border-border bg-card p-6 space-y-4">
        <h1 className="text-lg font-semibold text-foreground">Admin login</h1>
        <Input
          type="password"
          autoComplete="current-password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoFocus
        />
        {error && <p className="text-xs text-destructive">{error}</p>}
        <Button type="submit" className="w-full" disabled={busy || !password}>
          {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : "Log in"}
        </Button>
        <Link to="/" className="block text-center text-xs text-muted-foreground hover:underline">
          Back to site
        </Link>
      </form>
    </div>
  );
};

// ------------------------------------------------------------------ stats

const StatTile = ({ label, value, hint }: { label: string; value: string | number; hint?: string }) => (
  <div className="rounded-lg border border-border bg-card p-3">
    <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</div>
    <div className="text-xl font-semibold text-foreground tabular-nums">{value}</div>
    {hint && <div className="text-[11px] text-muted-foreground">{hint}</div>}
  </div>
);

const StatsHeader = ({ stats }: { stats: AdminStats }) => {
  const t = stats.totals;
  const c = stats.confusion;
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <StatTile label={`Predictions (${stats.days}d)`} value={t.predictions} />
        <StatTile label="Images stored" value={t.with_image} hint="with consent" />
        <StatTile label="Feedback" value={t.feedback} />
        <StatTile label="Labelled" value={t.labelled} hint="user knew the answer" />
        <StatTile label="Agreement" value={pct(t.agreement_rate, 1)} hint="with user labels" />
        <StatTile
          label="Uncertain"
          value={t.predictions ? pct(t.uncertain / t.predictions) : "—"}
          hint={`p in ${stats.uncertain_band[0]}–${stats.uncertain_band[1]}`}
        />
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-lg border border-border bg-card p-3">
          <div className="text-xs font-medium text-foreground mb-2">Confusion matrix (labelled items)</div>
          <table className="text-xs w-full tabular-nums">
            <thead>
              <tr className="text-muted-foreground">
                <th className="text-left font-normal" />
                <th className="text-right font-normal">Predicted AI</th>
                <th className="text-right font-normal">Predicted real</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="text-muted-foreground">User: AI</td>
                <td className="text-right">{c.tp}</td>
                <td className="text-right">{c.fn}</td>
              </tr>
              <tr>
                <td className="text-muted-foreground">User: real</td>
                <td className="text-right">{c.fp}</td>
                <td className="text-right">{c.tn}</td>
              </tr>
            </tbody>
          </table>
          <p className="mt-2 text-[11px] text-muted-foreground">
            Labels come from users who chose to answer, so treat this as a sanity signal, not a benchmark.
          </p>
        </div>
        <div className="rounded-lg border border-border bg-card p-3">
          <div className="text-xs font-medium text-foreground mb-2">Per variant</div>
          <table className="text-xs w-full tabular-nums">
            <thead>
              <tr className="text-muted-foreground">
                <th className="text-left font-normal">Variant</th>
                <th className="text-right font-normal">Predictions</th>
                <th className="text-right font-normal">Labelled</th>
                <th className="text-right font-normal">Agreement</th>
                <th className="text-right font-normal">Uncertain</th>
              </tr>
            </thead>
            <tbody>
              {stats.per_variant.map((v) => (
                <tr key={v.variant}>
                  <td>{v.variant}</td>
                  <td className="text-right">{v.predictions}</td>
                  <td className="text-right">{v.labelled}</td>
                  <td className="text-right">{pct(v.agreement_rate, 1)}</td>
                  <td className="text-right">{pct(v.uncertain_rate)}</td>
                </tr>
              ))}
              {stats.per_variant.length === 0 && (
                <tr>
                  <td colSpan={5} className="text-muted-foreground">No data yet</td>
                </tr>
              )}
            </tbody>
          </table>
          {stats.per_day.length > 0 && (
            <div className="mt-3">
              <div className="text-[11px] text-muted-foreground mb-1">Daily volume</div>
              <div className="flex items-end gap-px h-10" aria-label="Predictions per day">
                {stats.per_day.map((d) => {
                  const max = Math.max(...stats.per_day.map((x) => x.predictions), 1);
                  return (
                    <div
                      key={d.date}
                      title={`${d.date}: ${d.predictions} predictions, ${d.feedback} feedback`}
                      className="flex-1 bg-primary/60 rounded-sm min-w-[3px]"
                      style={{ height: `${Math.max(8, (d.predictions / max) * 100)}%` }}
                    />
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// ------------------------------------------------------------------ filters

type TriState = "any" | "yes" | "no";
const triToBool = (v: TriState): boolean | undefined => (v === "any" ? undefined : v === "yes");

const TriSelect = ({
  label,
  value,
  onChange,
}: {
  label: string;
  value: TriState;
  onChange: (v: TriState) => void;
}) => (
  <label className="flex flex-col gap-1 text-[11px] text-muted-foreground">
    {label}
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as TriState)}
      className="h-9 rounded-md border border-input bg-background px-2 text-sm text-foreground"
    >
      <option value="any">Any</option>
      <option value="yes">Yes</option>
      <option value="no">No</option>
    </select>
  </label>
);

interface FilterState {
  disagreement: TriState;
  uncertain: TriState;
  has_feedback: TriState;
  labelled: TriState;
  has_image: TriState;
  variant: string;
  date_from: string;
  date_to: string;
}

const EMPTY_FILTERS: FilterState = {
  disagreement: "any",
  uncertain: "any",
  has_feedback: "any",
  labelled: "any",
  has_image: "any",
  variant: "",
  date_from: "",
  date_to: "",
};

const toApiFilters = (f: FilterState): PredictionFilters => ({
  disagreement: triToBool(f.disagreement),
  uncertain: triToBool(f.uncertain),
  has_feedback: triToBool(f.has_feedback),
  labelled: triToBool(f.labelled),
  has_image: triToBool(f.has_image),
  variant: f.variant.trim() || undefined,
  date_from: f.date_from || undefined,
  date_to: f.date_to || undefined,
});

// ------------------------------------------------------------------ detail

const DetailDrawer = ({
  id,
  onClose,
  onDelete,
  onAuthError,
}: {
  id: string | null;
  onClose: () => void;
  onDelete: (id: string) => void;
  onAuthError: (e: unknown) => boolean;
}) => {
  const [detail, setDetail] = useState<AdminPredictionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);

  useEffect(() => {
    setDetail(null);
    setError(null);
    if (!id) return;
    let cancelled = false;
    getPrediction(id)
      .then((d) => !cancelled && setDetail(d))
      .catch((e) => {
        if (!cancelled && !onAuthError(e)) setError(e instanceof Error ? e.message : "Failed to load");
      });
    return () => {
      cancelled = true;
    };
  }, [id, onAuthError]);

  return (
    <Sheet open={!!id} onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="w-full sm:max-w-xl overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Prediction detail</SheetTitle>
          <SheetDescription className="font-mono text-[11px] break-all">{id}</SheetDescription>
        </SheetHeader>

        {error && <p className="text-sm text-destructive mt-4">{error}</p>}
        {!detail && !error && <Loader2 className="w-5 h-5 animate-spin mt-6" />}

        {detail && (
          <div className="space-y-5 mt-4">
            <div className="flex items-center gap-3">
              <VerdictBadge pred={detail.final_pred} />
              <span className="text-sm tabular-nums">p(AI) = {detail.final_prob.toFixed(3)}</span>
              <Badge variant="outline">{detail.variant}</Badge>
            </div>

            {detail.has_image ? (
              <AuthImage id={detail.id} className="w-full rounded-md border border-border" onUrl={setImageUrl} />
            ) : (
              <p className="text-xs text-muted-foreground">Image not stored (no consent).</p>
            )}

            <div className="space-y-1 text-xs">
              <div className="font-medium text-foreground">User feedback</div>
              {detail.feedback ? (
                <div className="text-muted-foreground space-y-0.5">
                  <div>
                    Label: <FeedbackBadge item={detail} />
                  </div>
                  {detail.feedback.source_note && <div>Source: {detail.feedback.source_note}</div>}
                  {detail.feedback.created_at && <div>At: {fmtDate(detail.feedback.created_at)}</div>}
                </div>
              ) : (
                <div className="text-muted-foreground">None</div>
              )}
            </div>

            <div className="space-y-4">
              <div className="text-xs font-medium text-foreground">Submodels</div>
              {Object.entries(detail.submodels).map(([name, sub]) => {
                const ex = detail.explain[name] || {};
                const insight = detail.llm[name];
                return (
                  <div key={name} className="rounded-lg border border-border p-3 space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium">{name}</span>
                      <span className="tabular-nums text-muted-foreground">
                        p(AI) {sub.prob_fake?.toFixed(3) ?? "—"}
                        {sub.contribution_percentage != null && ` · ${sub.contribution_percentage.toFixed(1)}% of fusion`}
                      </span>
                    </div>
                    {ex.heatmap_base64 && (
                      <HeatmapOverlay
                        heatmapBase64={ex.heatmap_base64}
                        originalImageUrl={imageUrl ?? undefined}
                        explainabilityType={ex.explainability_type}
                        modelName={name}
                      />
                    )}
                    {ex.focus_summary && <p className="text-xs text-muted-foreground">Focus: {ex.focus_summary}</p>}
                    {insight && (
                      <div className="text-xs text-muted-foreground bg-muted/30 rounded-md p-2 space-y-1">
                        <div className="font-medium text-foreground">{insight.key_finding}</div>
                        <div>{insight.what_model_saw}</div>
                        <div className="italic">{insight.confidence_qualifier}</div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            <dl className="grid grid-cols-[auto,1fr] gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
              <dt>Created</dt>
              <dd>{fmtDate(detail.created_at)}</dd>
              <dt>Size</dt>
              <dd>
                {detail.width ?? "?"}×{detail.height ?? "?"} {detail.format ?? ""}
              </dd>
              <dt>Latency</dt>
              <dd>{detail.latency_ms != null ? `${detail.latency_ms} ms` : "—"}</dd>
              <dt>Models</dt>
              <dd className="break-all">{detail.manifest_hash ?? "—"}</dd>
              <dt>SHA-256</dt>
              <dd className="font-mono break-all">{detail.image_sha256}</dd>
            </dl>

            <Button variant="destructive" size="sm" className="gap-1" onClick={() => onDelete(detail.id)}>
              <Trash2 className="w-3 h-3" /> Delete record{detail.has_image ? " and image" : ""}
            </Button>
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
};

// ------------------------------------------------------------------ page

const Dashboard = ({ onLogout }: { onLogout: () => void }) => {
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [items, setItems] = useState<AdminPrediction[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [filters, setFilters] = useState<FilterState>(EMPTY_FILTERS);
  const [applied, setApplied] = useState<FilterState>(EMPTY_FILTERS);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [retentionDays, setRetentionDays] = useState<number | null>(null);

  const handleAuthError = useCallback(
    (e: unknown) => {
      if (e instanceof AdminAuthError) {
        onLogout();
        return true;
      }
      return false;
    },
    [onLogout]
  );

  const load = useCallback(
    async (f: FilterState, after: string | null) => {
      setLoading(true);
      setError(null);
      try {
        const page = await listPredictions(toApiFilters(f), after);
        setItems((prev) => (after ? [...prev, ...page.items] : page.items));
        setCursor(page.next_cursor);
      } catch (e) {
        if (!handleAuthError(e)) setError(e instanceof Error ? e.message : "Failed to load");
      } finally {
        setLoading(false);
      }
    },
    [handleAuthError]
  );

  const loadStats = useCallback(async () => {
    try {
      setStats(await getStats(30));
    } catch (e) {
      handleAuthError(e);
    }
  }, [handleAuthError]);

  useEffect(() => {
    checkSession()
      .then((s) => setRetentionDays(s.retention_days))
      .catch(handleAuthError);
    loadStats();
    load(EMPTY_FILTERS, null);
  }, [load, loadStats, handleAuthError]);

  const apply = () => {
    setApplied(filters);
    load(filters, null);
  };

  const refresh = () => {
    loadStats();
    load(applied, null);
  };

  const remove = async (id: string) => {
    if (!window.confirm("Delete this record, its feedback and any stored image? This cannot be undone.")) return;
    try {
      await deletePrediction(id);
      setItems((prev) => prev.filter((i) => i.id !== id));
      setSelected(null);
      loadStats();
    } catch (e) {
      if (!handleAuthError(e)) setError(e instanceof Error ? e.message : "Delete failed");
    }
  };

  const exportCsv = async () => {
    try {
      await downloadCsv(toApiFilters(applied));
    } catch (e) {
      if (!handleAuthError(e)) setError(e instanceof Error ? e.message : "Export failed");
    }
  };

  const set = <K extends keyof FilterState>(key: K) => (value: FilterState[K]) =>
    setFilters((prev) => ({ ...prev, [key]: value }));

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between gap-2">
          <div>
            <h1 className="text-base font-semibold text-foreground">DeepFake Detector · Admin</h1>
            {retentionDays !== null && (
              <p className="text-[11px] text-muted-foreground">Retention: {retentionDays} days</p>
            )}
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" className="gap-1" onClick={refresh} disabled={loading}>
              <RefreshCw className="w-3 h-3" /> Refresh
            </Button>
            <Button variant="outline" size="sm" className="gap-1" onClick={exportCsv}>
              <Download className="w-3 h-3" /> Export CSV
            </Button>
            <Button variant="ghost" size="sm" className="gap-1" onClick={onLogout}>
              <LogOut className="w-3 h-3" /> Log out
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">
        {stats ? <StatsHeader stats={stats} /> : <Loader2 className="w-5 h-5 animate-spin" />}

        <section className="rounded-lg border border-border bg-card p-3">
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-9 gap-3 items-end">
            <TriSelect label="Disagreement" value={filters.disagreement} onChange={set("disagreement")} />
            <TriSelect label="Uncertain band" value={filters.uncertain} onChange={set("uncertain")} />
            <TriSelect label="Has feedback" value={filters.has_feedback} onChange={set("has_feedback")} />
            <TriSelect label="Labelled" value={filters.labelled} onChange={set("labelled")} />
            <TriSelect label="Has image" value={filters.has_image} onChange={set("has_image")} />
            <label className="flex flex-col gap-1 text-[11px] text-muted-foreground">
              Variant
              <Input value={filters.variant} onChange={(e) => set("variant")(e.target.value)} className="h-9" />
            </label>
            <label className="flex flex-col gap-1 text-[11px] text-muted-foreground">
              From
              <Input type="date" value={filters.date_from} onChange={(e) => set("date_from")(e.target.value)} className="h-9" />
            </label>
            <label className="flex flex-col gap-1 text-[11px] text-muted-foreground">
              To
              <Input type="date" value={filters.date_to} onChange={(e) => set("date_to")(e.target.value)} className="h-9" />
            </label>
            <div className="flex gap-2">
              <Button size="sm" onClick={apply} disabled={loading}>
                Apply
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  setFilters(EMPTY_FILTERS);
                  setApplied(EMPTY_FILTERS);
                  load(EMPTY_FILTERS, null);
                }}
              >
                Reset
              </Button>
            </div>
          </div>
        </section>

        {error && <p className="text-sm text-destructive">{error}</p>}

        <section className="rounded-lg border border-border bg-card overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-16">Image</TableHead>
                <TableHead>Date</TableHead>
                <TableHead>Verdict</TableHead>
                <TableHead className="text-right">p(AI)</TableHead>
                <TableHead>Submodels</TableHead>
                <TableHead>User label</TableHead>
                <TableHead>Variant</TableHead>
                <TableHead />
              </TableRow>
            </TableHeader>
            <TableBody>
              {items.map((item) => (
                <TableRow key={item.id} className="cursor-pointer" onClick={() => setSelected(item.id)}>
                  <TableCell>
                    <div className="w-12 h-12 flex items-center justify-center rounded bg-muted/40 overflow-hidden">
                      {item.has_image ? (
                        <AuthImage id={item.id} thumb={96} className="w-12 h-12 object-cover" />
                      ) : (
                        <ImageOff className="w-4 h-4 text-muted-foreground" aria-label="No stored image" />
                      )}
                    </div>
                  </TableCell>
                  <TableCell className="text-xs whitespace-nowrap">{fmtDate(item.created_at)}</TableCell>
                  <TableCell>
                    <VerdictBadge pred={item.final_pred} />
                  </TableCell>
                  <TableCell className="text-right tabular-nums text-xs">{item.final_prob.toFixed(3)}</TableCell>
                  <TableCell className="text-[11px] text-muted-foreground tabular-nums">
                    {Object.entries(item.submodels).map(([name, sub]) => (
                      <div key={name} className="whitespace-nowrap">
                        {name}: {sub.prob_fake?.toFixed(2) ?? "—"}
                      </div>
                    ))}
                  </TableCell>
                  <TableCell>
                    <FeedbackBadge item={item} />
                  </TableCell>
                  <TableCell className="text-xs">{item.variant}</TableCell>
                  <TableCell>
                    <Button
                      variant="ghost"
                      size="icon"
                      aria-label="Delete record"
                      onClick={(e) => {
                        e.stopPropagation();
                        remove(item.id);
                      }}
                    >
                      <Trash2 className="w-4 h-4 text-muted-foreground" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
              {!loading && items.length === 0 && (
                <TableRow>
                  <TableCell colSpan={8} className="text-center text-sm text-muted-foreground py-8">
                    No predictions match these filters.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </section>

        <div className="flex justify-center">
          {loading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            cursor && (
              <Button variant="outline" size="sm" onClick={() => load(applied, cursor)}>
                Load more
              </Button>
            )
          )}
        </div>
      </main>

      <DetailDrawer id={selected} onClose={() => setSelected(null)} onDelete={remove} onAuthError={handleAuthError} />
    </div>
  );
};

const Admin = () => {
  useNoIndex();
  const [loggedIn, setLoggedIn] = useState(() => !!getToken());

  const logout = useCallback(() => {
    clearToken();
    setLoggedIn(false);
  }, []);

  return loggedIn ? <Dashboard onLogout={logout} /> : <LoginForm onLoggedIn={() => setLoggedIn(true)} />;
};

export default Admin;
