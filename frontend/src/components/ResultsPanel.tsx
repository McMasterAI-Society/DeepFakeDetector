import { motion, AnimatePresence } from "framer-motion";
import { ShieldCheck, ShieldAlert, Clock } from "lucide-react";
import ReasoningPanel from "@/components/ReasoningPanel";
import type { ModelDisplayInfo, SingleModelInsight } from "@/components/ModelTab";

interface FusionMeta {
  submodel_weights: Record<string, number>;
  weighted_contributions: Record<string, number>;
  contribution_percentages: Record<string, number>;
}

interface SubmodelResult {
  pred: "real" | "fake";
  pred_int: number;
  prob_fake: number;
  heatmap_base64?: string;
  explainability_type?: "grad_cam" | "attention_rollout";
  focus_summary?: string;
  contribution_percentage?: number;
}

interface PredictionResult {
  final: { 
    pred: "real" | "fake"; 
    pred_int: number; 
    prob_fake: number;
    heatmap_base64?: string;
    explainability_type?: "grad_cam" | "attention_rollout";
    focus_summary?: string;
  };
  fusion_used: boolean;
  submodels: Record<string, SubmodelResult> | null;
  timing_ms: { total: number; inference?: number; fusion?: number };
  fusion_meta?: FusionMeta | null;
  model_display_info?: Record<string, ModelDisplayInfo> | null;
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
}

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

const ResultsPanel = ({ result, showSubmodels, originalFile, onRequestInsight }: ResultsPanelProps) => {
  const isFake = result.final.pred === "fake";
  const probFake = result.final.prob_fake;
  const probReal = 1 - probFake;

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
        className={`rounded-lg p-5 text-center ${
          isFake
            ? "bg-destructive/10 border border-destructive/30 glow-destructive"
            : "bg-success/10 border border-success/30 glow-success"
        }`}
      >
        <div className="flex items-center justify-center gap-2 mb-2">
          {isFake ? (
            <ShieldAlert className="w-6 h-6 text-destructive" />
          ) : (
            <ShieldCheck className="w-6 h-6 text-success" />
          )}
          <span
            className={`text-lg font-bold ${
              isFake ? "text-destructive" : "text-success"
            }`}
          >
            {isFake ? "AI-Generated" : "Likely Real"}
          </span>
        </div>
        <p className="text-xs text-muted-foreground">
          {isFake
            ? "This image shows signs of AI generation"
            : "This image appears to be authentic"}
        </p>
      </motion.div>

      {/* Confidence bars */}
      <div className="rounded-lg border border-border bg-card p-4 space-y-3">
        <h3 className="text-sm font-medium text-foreground">Confidence</h3>
        <ConfidenceBar value={probFake} label="AI-Generated probability" />
        <ConfidenceBar value={probReal} label="Likely Real probability" />
      </div>

      {/* Model Reasoning Panel */}
      {showSubmodels && result.submodels && (
        <ReasoningPanel
          submodels={result.submodels}
          modelDisplayInfo={result.model_display_info || {}}
          originalImageFile={originalFile}
          fusionMeta={result.fusion_meta}
          onRequestInsight={onRequestInsight}
        />
      )}

      {/* Timing */}
      <div className="flex items-center gap-1 text-xs text-muted-foreground">
        <Clock className="w-3 h-3" />
        <span>
          Total: {result.timing_ms.total.toFixed(0)}ms
          {result.timing_ms.inference != null && ` · Inference: ${result.timing_ms.inference.toFixed(0)}ms`}
          {result.timing_ms.fusion != null && ` · Fusion: ${result.timing_ms.fusion.toFixed(0)}ms`}
        </span>
      </div>
    </motion.div>
  );
};

export default ResultsPanel;
