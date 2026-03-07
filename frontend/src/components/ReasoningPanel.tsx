import { useState, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import { BrainCircuit } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ModelTab, { type ModelDisplayInfo, type SingleModelInsight } from "@/components/ModelTab";

interface SubmodelResult {
  pred: "real" | "fake";
  pred_int: number;
  prob_fake: number;
  heatmap_base64?: string;
  explainability_type?: "grad_cam" | "attention_rollout";
  focus_summary?: string;
  contribution_percentage?: number;
}

interface FusionMeta {
  submodel_weights: Record<string, number>;
  weighted_contributions: Record<string, number>;
  contribution_percentages: Record<string, number>;
}

interface ReasoningPanelProps {
  /** Submodel results */
  submodels: Record<string, SubmodelResult>;
  /** Model display info for each model */
  modelDisplayInfo: Record<string, ModelDisplayInfo>;
  /** Original image file for the slider */
  originalImageFile?: File | null;
  /** Fusion metadata (if fusion was used) */
  fusionMeta?: FusionMeta | null;
  /** Final overall prediction to filter agreeing models */
  finalPrediction: "real" | "fake";
  /** Callback to request AI insight for a model */
  onRequestInsight: (
    modelName: string,
    probFake: number,
    heatmapBase64?: string,
    focusSummary?: string,
    contributionPercentage?: number
  ) => Promise<SingleModelInsight | null>;
}

/**
 * Tabbed panel showing reasoning for each model's prediction.
 * Dynamically creates tabs based on actual submodels returned.
 */
const ReasoningPanel = ({
  submodels,
  modelDisplayInfo,
  originalImageFile,
  fusionMeta,
  finalPrediction,
  onRequestInsight,
}: ReasoningPanelProps) => {
  const [originalImageUrl, setOriginalImageUrl] = useState<string | null>(null);
  // Store insights per model so they persist across tab switches
  const [modelInsights, setModelInsights] = useState<Record<string, SingleModelInsight>>({});
  
  // Create object URL for original image
  useEffect(() => {
    if (originalImageFile) {
      const url = URL.createObjectURL(originalImageFile);
      setOriginalImageUrl(url);
      return () => URL.revokeObjectURL(url);
    }
    return undefined;
  }, [originalImageFile]);

  // Filter to only models that agree with the final prediction
  const modelNames = Object.keys(submodels).filter(
    (name) => submodels[name].pred === finalPrediction
  );
  const sortedModelNames = fusionMeta?.contribution_percentages
    ? [...modelNames].sort(
        (a, b) =>
          (fusionMeta.contribution_percentages[b] || 0) -
          (fusionMeta.contribution_percentages[a] || 0)
      )
    : modelNames;

  // Create insight request handler for a specific model
  const createInsightHandler = useCallback(
    (modelName: string) => async (): Promise<SingleModelInsight | null> => {
      const sub = submodels[modelName];
      const contributionPct = fusionMeta?.contribution_percentages?.[modelName];
      const result = await onRequestInsight(
        modelName,
        sub.prob_fake,
        sub.heatmap_base64,
        sub.focus_summary,
        contributionPct
      );
      // Store the insight if successful
      if (result) {
        setModelInsights(prev => ({ ...prev, [modelName]: result }));
      }
      return result;
    },
    [submodels, fusionMeta, onRequestInsight]
  );

  // Get display info with fallback
  const getDisplayInfo = (modelName: string): ModelDisplayInfo => {
    return (
      modelDisplayInfo[modelName] || {
        display_name: modelName,
        short_name: modelName.slice(0, 8),
        method_name: "Analysis",
        method_description: "Model analysis method",
        educational_text: "This model analyzes the image for signs of AI generation.",
        what_it_looks_for: ["Image patterns", "Artifacts", "Inconsistencies"],
      }
    );
  };

  // Custom tab labels mapping
  const tabLabelMap: Record<string, string> = {
    "cnn-transfer": "Texture Analysis",
    "cnn_transfer": "Texture Analysis",
    "vit-patch": "Patch Consistency",
    "vit_patch": "Patch Consistency",
    "vit": "Patch Consistency",
    "deit-distilled": "Global Structure",
    "deit_distilled": "Global Structure",
    "deit": "Global Structure",
    "gradient-field": "Edge Coherence",
    "gradient_field": "Edge Coherence",
    "gradient-field-cnn": "Edge Coherence",
    "gradient_field_cnn": "Edge Coherence",
    "gradfield-cnn": "Edge Coherence",
    "gradfield_cnn": "Edge Coherence",
  };

  const getTabLabel = (modelName: string): string => {
    const lower = modelName.toLowerCase();
    if (tabLabelMap[lower]) return tabLabelMap[lower];
    if (lower.includes("gradient")) return "Edge Coherence";
    if (lower.includes("cnn")) return "Texture Analysis";
    if (lower.includes("vit")) return "Patch Consistency";
    if (lower.includes("deit")) return "Global Structure";
    return getDisplayInfo(modelName).method_name;
  };

  // Default to first model
  const defaultTab = sortedModelNames[0] || "";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="rounded-lg border border-border bg-card overflow-hidden"
    >
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border bg-muted/30">
        <BrainCircuit className="w-4 h-4 text-primary" />
        <h3 className="text-sm font-semibold text-foreground">Model Reasoning</h3>
        <span className="text-xs text-muted-foreground ml-auto">
          {sortedModelNames.length} model{sortedModelNames.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Tabs */}
      <Tabs defaultValue={defaultTab} className="w-full">
        <div className="border-b border-border px-2 overflow-x-auto">
          <TabsList className="h-10 bg-transparent gap-1">
            {sortedModelNames.map((name) => {
              const info = getDisplayInfo(name);
              const sub = submodels[name];
              const contributionPct = fusionMeta?.contribution_percentages?.[name];
              
              return (
                <TabsTrigger
                  key={name}
                  value={name}
                  className="relative data-[state=active]:bg-primary/10 data-[state=active]:text-primary rounded-md px-3 py-1.5 text-xs font-medium transition-colors"
                >
                  <span className="flex items-center gap-2">
                    <span
                      className={`w-2 h-2 rounded-full ${
                        sub.pred === "fake" ? "bg-destructive" : "bg-success"
                      }`}
                    />
                    {getTabLabel(name)}
                    {contributionPct !== undefined && (
                      <span className="text-[10px] text-muted-foreground">
                        {contributionPct.toFixed(0)}%
                      </span>
                    )}
                  </span>
                </TabsTrigger>
              );
            })}
          </TabsList>
        </div>

        {sortedModelNames.map((name) => {
          const info = getDisplayInfo(name);
          const sub = submodels[name];
          const contributionPct = fusionMeta?.contribution_percentages?.[name];

          return (
            <TabsContent key={name} value={name} className="p-4 mt-0">
              <ModelTab
                modelName={name}
                displayInfo={info}
                prediction={{
                  pred: sub.pred,
                  prob_fake: sub.prob_fake,
                  heatmap_base64: sub.heatmap_base64,
                  focus_summary: sub.focus_summary,
                }}
                contributionPercentage={contributionPct}
                originalImageUrl={originalImageUrl || undefined}
                onRequestInsight={createInsightHandler(name)}
                existingInsight={modelInsights[name]}
              />
            </TabsContent>
          );
        })}
      </Tabs>
    </motion.div>
  );
};

export default ReasoningPanel;
