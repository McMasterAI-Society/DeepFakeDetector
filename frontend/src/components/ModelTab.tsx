import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Sparkles,
  Loader2,
  BookOpen,
  Target,
  ChevronDown,
  AlertTriangle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import ImageSlider from "@/components/ImageSlider";

export interface ModelDisplayInfo {
  display_name: string;
  short_name: string;
  method_name: string;
  method_description: string;
  educational_text: string;
  what_it_looks_for: string[];
}

export interface SingleModelInsight {
  key_finding: string;
  what_model_saw: string;
  important_regions: string[];
  confidence_qualifier: string;
}

interface ModelTabProps {
  /** Internal model name (e.g., "cnn-transfer") */
  modelName: string;
  /** Display info for this model */
  displayInfo: ModelDisplayInfo;
  /** Prediction result for this model */
  prediction: {
    pred: "real" | "fake";
    prob_fake: number;
    heatmap_base64?: string;
    focus_summary?: string;
  };
  /** Contribution percentage from logistic regression (if fusion used) */
  contributionPercentage?: number;
  /** Original image URL for slider */
  originalImageUrl?: string;
  /** Callback to request AI explanation */
  onRequestInsight: () => Promise<SingleModelInsight | null>;
  /** Pre-existing insight (persisted from ReasoningPanel) */
  existingInsight?: SingleModelInsight;
}

const ModelTab = ({
  modelName,
  displayInfo,
  prediction,
  contributionPercentage,
  originalImageUrl,
  onRequestInsight,
  existingInsight,
}: ModelTabProps) => {
  const [isLoadingInsight, setIsLoadingInsight] = useState(false);
  const [insightError, setInsightError] = useState<string | null>(null);
  const [showEducational, setShowEducational] = useState(false);

  // Use existing insight from parent if available
  const insight = existingInsight || null;

  const handleGetInsight = async () => {
    setIsLoadingInsight(true);
    setInsightError(null);
    try {
      const result = await onRequestInsight();
      if (!result) {
        setInsightError("Could not generate insight. Please try again.");
      }
    } catch (err) {
      setInsightError(err instanceof Error ? err.message : "Failed to get insight");
    } finally {
      setIsLoadingInsight(false);
    }
  };

  const isFake = prediction.pred === "fake";

  return (
    <div className="space-y-4">
      {/* Model header with contribution */}
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <h4 className="text-sm font-semibold text-foreground">
            {displayInfo.display_name}
          </h4>
          <p className="text-xs text-muted-foreground">
            Method: {displayInfo.method_name}
          </p>
        </div>
        {contributionPercentage !== undefined && (
          <div className="text-right flex-shrink-0">
            <div className="text-lg font-bold text-foreground">
              {contributionPercentage.toFixed(1)}%
            </div>
            <div className="text-[10px] text-muted-foreground">contribution</div>
          </div>
        )}
      </div>

      {/* Prediction result */}
      <div
        className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${
          isFake
            ? "bg-destructive/15 text-destructive"
            : "bg-success/15 text-success"
        }`}
      >
        <Target className="w-3 h-3" />
        {isFake ? "AI-Generated" : "Likely Real"}: {(prediction.prob_fake * 100).toFixed(1)}%
        {" "}fake probability
      </div>

      {/* Image slider (if heatmap available) */}
      {prediction.heatmap_base64 && originalImageUrl && (
        <ImageSlider
          originalImageUrl={originalImageUrl}
          heatmapBase64={prediction.heatmap_base64}
          modelName={modelName}
        />
      )}

      {/* Focus summary from model */}
      {prediction.focus_summary && (
        <div className="text-xs text-muted-foreground bg-muted/30 rounded-md p-2">
          <span className="font-medium">Model focus:</span> {prediction.focus_summary}
        </div>
      )}

      {/* Educational section */}
      <div className="border border-border rounded-lg overflow-hidden">
        <button
          onClick={() => setShowEducational(!showEducational)}
          className="w-full flex items-center justify-between p-3 text-sm hover:bg-muted/30 transition-colors"
        >
          <div className="flex items-center gap-2 text-foreground">
            <BookOpen className="w-4 h-4 text-primary" />
            <span className="font-medium">How this works</span>
          </div>
          <motion.span
            animate={{ rotate: showEducational ? 180 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <ChevronDown className="w-4 h-4 text-muted-foreground" />
          </motion.span>
        </button>
        <AnimatePresence>
          {showEducational && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div className="px-3 pb-3 pt-1 space-y-3">
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {displayInfo.educational_text}
                </p>
                <div>
                  <p className="text-xs font-medium text-foreground mb-1">
                    What it looks for:
                  </p>
                  <ul className="text-xs text-muted-foreground space-y-0.5 pl-3">
                    {displayInfo.what_it_looks_for.map((item, i) => (
                      <li key={i} className="list-disc">
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
                <p className="text-[10px] text-muted-foreground/70">
                  <span className="font-medium">Method:</span>{" "}
                  {displayInfo.method_description}
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* AI Insight section */}
      <div className="space-y-2">
        {!insight ? (
          <Button
            onClick={handleGetInsight}
            disabled={isLoadingInsight}
            variant="outline"
            className="w-full gap-2"
          >
            {isLoadingInsight ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Generating insight...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                Get AI Insight
              </>
            )}
          </Button>
        ) : (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-primary/5 border border-primary/20 rounded-lg p-3 space-y-3"
          >
            <div className="flex items-center gap-2 text-xs font-medium text-primary">
              <Sparkles className="w-3 h-3" />
              AI Interpretation
            </div>
            
            <div className="space-y-2">
              <p className="text-sm font-medium text-foreground">
                {insight.key_finding}
              </p>
              <p className="text-xs text-muted-foreground">
                {insight.what_model_saw}
              </p>
            </div>

            {insight.important_regions.length > 0 && (
              <div>
                <p className="text-xs font-medium text-foreground mb-1">
                  Key regions identified:
                </p>
                <ul className="text-xs text-muted-foreground space-y-0.5 pl-3">
                  {insight.important_regions.map((region, i) => (
                    <li key={i} className="list-disc">
                      {region}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="flex items-start gap-1.5 pt-2 border-t border-primary/20">
              <AlertTriangle className="w-3 h-3 text-muted-foreground/70 mt-0.5 flex-shrink-0" />
              <p className="text-[10px] text-muted-foreground/70 italic">
                {insight.confidence_qualifier}
              </p>
            </div>
          </motion.div>
        )}

        {insightError && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-xs text-destructive text-center"
          >
            {insightError}
          </motion.p>
        )}
      </div>
    </div>
  );
};

export default ModelTab;
