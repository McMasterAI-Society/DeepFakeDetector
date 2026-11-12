import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import { ShieldCheck, ShieldAlert, ShieldQuestion, Clock, Flame, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import ReasoningPanel from "@/components/ReasoningPanel";
import type { ModelDisplayInfo, SingleModelInsight } from "@/components/ModelTab";

interface FusionMeta {
  submodel_weights: Record<string, number>;
  weighted_contributions: Record<string, number>;
  contribution_percentages: Record<string, number>;
  /** Foundation probe only: weighted_contributions + logit_intercept = raw logit */
  logit_intercept?: number | null;
}

export type Verdict = "ai" | "inconclusive" | "real";

interface SubmodelResult {
  pred: "real" | "fake";
  pred_int: number;
  prob_fake: number;
  heatmap_base64?: string;
  explainability_type?: "grad_cam" | "attention_rollout" | "input_gradient";
  focus_summary?: string;
  contribution_percentage?: number;
}

interface PredictionResult {
  final: { 
    pred: "real" | "fake"; 
    pred_int: number; 
    prob_fake: number;
    heatmap_base64?: string;
    explainability_type?: "grad_cam" | "attention_rollout" | "input_gradient";
    focus_summary?: string;
  };
  fusion_used: boolean;
  submodels: Record<string, SubmodelResult> | null;
  timing_ms: { total: number; inference?: number; fusion?: number };
  fusion_meta?: FusionMeta | null;
  model_display_info?: Record<string, ModelDisplayInfo> | null;
  /** Three-way verdict (newer backends); falls back to final.pred when absent */
  verdict?: Verdict | null;
  verdict_bands?: number[] | null;
  model_version?: string | null;
}

interface ResultsPanelProps {
  result: PredictionResult;
  showSubmodels: boolean;
  originalFile?: File | null;
  onRequestInsight: (
    modelName: string,
    probFake: number,
    heatmapBase64?: string,
    focusSummary?: string,
    contributionPercentage?: number
  ) => Promise<SingleModelInsight | null>;
  /** Fetch heatmaps on demand (the foundation probe computes them off the hot path) */
  onRequestHeatmaps?: () => Promise<void>;
}

const VERDICT_STYLE: Record<Verdict, { title: string; text: string; box: string; color: string }> = {
  ai: {
    title: "AI-Generated",
    text: "This image shows signs of AI generation",
    box: "bg-destructive/10 border border-destructive/30 glow-destructive",
    color: "text-destructive",
  },
  inconclusive: {
    title: "Inconclusive",
    text: "The detector cannot tell confidently. High-quality real photos sometimes look generated to it, so treat this result as uncertain.",
    box: "bg-warning/10 border border-warning/30",
    color: "text-warning",
  },
  real: {
    title: "Likely Real",
    text: "This image appears to be authentic",
    box: "bg-success/10 border border-success/30 glow-success",
    color: "text-success",
  },
};

const ConfidenceBar = ({ value, label }: { value: number; label: string }) => (
  <div className="space-y-1">
    <div className="flex justify-between text-xs text-muted-foreground">
      <span>{label}</span>
      <span>{(value * 100).toFixed(1)}%</span>
    </div>
    <div className="h-2 rounded-full bg-muted overflow-hidden">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${value * 100}%` }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className={`h-full rounded-full ${
          label.includes("AI") ? "bg-destructive" : "bg-success"
        }`}
      />
    </div>
  </div>
);

const ResultsPanel = ({ result, showSubmodels, originalFile, onRequestInsight, onRequestHeatmaps }: ResultsPanelProps) => {
  const verdict: Verdict = result.verdict ?? (result.final.pred === "fake" ? "ai" : "real");
  const style = VERDICT_STYLE[verdict];
  const probFake = result.final.prob_fake;
  const probReal = 1 - probFake;
  const [heatmapsLoading, setHeatmapsLoading] = useState(false);
  const [heatmapsError, setHeatmapsError] = useState<string | null>(null);
  const hasHeatmaps = Object.values(result.submodels ?? {}).some((s) => !!s.heatmap_base64);
  const canRequestHeatmaps =
    !!onRequestHeatmaps && showSubmodels && !!result.submodels && !!result.verdict && !hasHeatmaps;

  const requestHeatmaps = async () => {
    if (!onRequestHeatmaps) return;
    setHeatmapsLoading(true);
    setHeatmapsError(null);
    try {
      await onRequestHeatmaps();
    } catch (err) {
      setHeatmapsError(err instanceof Error ? err.message : "Failed to load heatmaps");
    } finally {
      setHeatmapsLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      aria-live="polite"
      aria-label="Analysis results"
      className="space-y-4"
    >
      {/* Verdict */}
      <motion.div
        initial={{ scale: 0.95 }}
        animate={{ scale: 1 }}
        className={`rounded-lg p-5 text-center ${style.box}`}
      >
        <div className="flex items-center justify-center gap-2 mb-2">
          {verdict === "ai" && <ShieldAlert className={`w-6 h-6 ${style.color}`} />}
          {verdict === "inconclusive" && <ShieldQuestion className={`w-6 h-6 ${style.color}`} />}
          {verdict === "real" && <ShieldCheck className={`w-6 h-6 ${style.color}`} />}
          <span className={`text-lg font-bold ${style.color}`}>{style.title}</span>
        </div>
        <p className="text-xs text-muted-foreground">{style.text}</p>
      </motion.div>

      {/* Confidence bars */}
      <div className="rounded-lg border border-border bg-card p-4 space-y-3">
        <h3 className="text-sm font-medium text-foreground">Confidence</h3>
        <ConfidenceBar value={probFake} label="AI-Generated probability" />
        <ConfidenceBar value={probReal} label="Likely Real probability" />
      </div>

      {/* On-demand heatmaps (foundation probe) */}
      {canRequestHeatmaps && (
        <div className="flex flex-col items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            onClick={requestHeatmaps}
            disabled={heatmapsLoading}
            className="gap-2 text-xs"
          >
            {heatmapsLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : <Flame className="w-3 h-3" />}
            {heatmapsLoading ? "Computing heatmaps..." : "Show where the model looked"}
          </Button>
          {heatmapsError && <p className="text-xs text-destructive">{heatmapsError}</p>}
        </div>
      )}

      {/* Model Reasoning Panel */}
      {showSubmodels && result.submodels && (
        <ReasoningPanel
          submodels={result.submodels}
          modelDisplayInfo={result.model_display_info || {}}
          originalImageFile={originalFile}
          fusionMeta={result.fusion_meta}
          onRequestInsight={onRequestInsight}
          finalPrediction={result.final.pred}
        />
      )}

      {/* Timing */}
      <div className="flex items-center gap-1 text-xs text-muted-foreground">
        <Clock className="w-3 h-3" />
        <span>
          Total: {result.timing_ms.total.toFixed(0)}ms
          {result.timing_ms.inference != null && ` · Inference: ${result.timing_ms.inference.toFixed(0)}ms`}
          {result.timing_ms.fusion != null && ` · Fusion: ${result.timing_ms.fusion.toFixed(0)}ms`}
          {result.model_version && ` · Model: ${result.model_version}`}
        </span>
      </div>
    </motion.div>
  );
};

export default ResultsPanel;
