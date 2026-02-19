import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ShieldCheck, ShieldAlert, Clock, Bot, CheckCircle2, ChevronDown } from "lucide-react";
import HeatmapOverlay from "@/components/HeatmapOverlay";

interface SubmodelResult {
  pred: "real" | "fake";
  pred_int: number;
  prob_fake: number;
  heatmap_base64?: string;
  explainability_type?: "grad_cam" | "attention_rollout";
}

interface PredictionResult {
  final: { 
    pred: "real" | "fake"; 
    pred_int: number; 
    prob_fake: number;
    heatmap_base64?: string;
    explainability_type?: "grad_cam" | "attention_rollout";
  };
  fusion_used: boolean;
  submodels: Record<string, SubmodelResult> | null;
  timing_ms: { total: number; inference?: number; fusion?: number };
}

interface ResultsPanelProps {
  result: PredictionResult;
  showSubmodels: boolean;
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

const ResultsPanel = ({ result, showSubmodels }: ResultsPanelProps) => {
  const [expandedModel, setExpandedModel] = useState<string | null>(null);
  
  const isFake = result.final.pred === "fake";
  const probFake = result.final.prob_fake;
  const probReal = 1 - probFake;
  
  // Check if any submodel has heatmaps
  const hasHeatmaps = result.submodels && Object.values(result.submodels).some(sub => sub.heatmap_base64);

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

      {/* Submodels */}
      {showSubmodels && result.submodels && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="rounded-lg border border-border bg-card p-4"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-foreground">Submodel Results</h3>
            {hasHeatmaps && (
              <span className="text-xs text-muted-foreground">Click to view heatmaps</span>
            )}
          </div>
          <div className="space-y-2">
            {Object.entries(result.submodels).map(([name, sub]) => (
              <div key={name}>
                <button
                  onClick={() => sub.heatmap_base64 && setExpandedModel(expandedModel === name ? null : name)}
                  className={`w-full flex items-center justify-between py-2 px-3 rounded-md bg-muted/30 text-sm transition-colors ${
                    sub.heatmap_base64 ? "hover:bg-muted/50 cursor-pointer" : "cursor-default"
                  }`}
                  aria-expanded={expandedModel === name}
                  disabled={!sub.heatmap_base64}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    {sub.pred === "fake" ? (
                      <Bot className="w-4 h-4 text-destructive flex-shrink-0" aria-hidden="true" />
                    ) : (
                      <CheckCircle2 className="w-4 h-4 text-success flex-shrink-0" aria-hidden="true" />
                    )}
                    <span className="text-foreground truncate">{name}</span>
                  </div>
                  <div className="flex items-center gap-3 flex-shrink-0">
                    <span
                      className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                        sub.pred === "fake"
                          ? "bg-destructive/15 text-destructive"
                          : "bg-success/15 text-success"
                      }`}
                    >
                      {sub.pred === "fake" ? "Fake" : "Real"}
                    </span>
                    <span className="text-xs text-muted-foreground w-14 text-right">
                      {(sub.prob_fake * 100).toFixed(1)}%
                    </span>
                    {sub.heatmap_base64 && (
                      <motion.span
                        animate={{ rotate: expandedModel === name ? 180 : 0 }}
                        transition={{ duration: 0.2 }}
                      >
                        <ChevronDown className="w-4 h-4 text-muted-foreground" />
                      </motion.span>
                    )}
                  </div>
                </button>
                
                {/* Expandable heatmap section */}
                <AnimatePresence>
                  {expandedModel === name && sub.heatmap_base64 && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden"
                    >
                      <div className="pt-3 pl-3">
                        <HeatmapOverlay
                          heatmapBase64={sub.heatmap_base64}
                          explainabilityType={sub.explainability_type}
                          modelName={name}
                        />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            ))}
          </div>
        </motion.div>
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
