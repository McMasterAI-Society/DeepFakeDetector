import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Eye, EyeOff, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface HeatmapOverlayProps {
  /** Base64-encoded PNG heatmap */
  heatmapBase64: string;
  /** Original image (optional, for overlay mode) */
  originalImageUrl?: string;
  /** Type of explainability method used */
  explainabilityType?: "grad_cam" | "attention_rollout";
  /** Model name for display */
  modelName: string;
}

const HeatmapOverlay = ({
  heatmapBase64,
  originalImageUrl,
  explainabilityType,
  modelName,
}: HeatmapOverlayProps) => {
  const [showOverlay, setShowOverlay] = useState(true);
  const [opacity, setOpacity] = useState(0.6);

  const heatmapSrc = `data:image/png;base64,${heatmapBase64}`;
  
  const methodLabel = explainabilityType === "grad_cam" 
    ? "Grad-CAM" 
    : explainabilityType === "attention_rollout" 
    ? "Attention Rollout" 
    : "Heatmap";

  const methodDescription = explainabilityType === "grad_cam"
    ? "Highlights CNN regions most influential for the prediction"
    : explainabilityType === "attention_rollout"
    ? "Shows aggregated transformer attention across layers"
    : "Visualization of model focus areas";

  return (
    <div className="space-y-2">
      {/* Header with controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-muted-foreground">{methodLabel}</span>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="w-3 h-3 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-xs">
                <p className="text-xs">{methodDescription}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        
        {originalImageUrl && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowOverlay(!showOverlay)}
            className="h-6 px-2 text-xs gap-1"
          >
            {showOverlay ? (
              <>
                <EyeOff className="w-3 h-3" />
                Hide overlay
              </>
            ) : (
              <>
                <Eye className="w-3 h-3" />
                Show overlay
              </>
            )}
          </Button>
        )}
      </div>

      {/* Heatmap visualization */}
      <div className="relative rounded-md overflow-hidden bg-muted aspect-square max-w-[200px]">
        {/* Original image (if provided) */}
        {originalImageUrl && (
          <img
            src={originalImageUrl}
            alt={`Original for ${modelName}`}
            className="absolute inset-0 w-full h-full object-cover"
          />
        )}
        
        {/* Heatmap overlay or standalone */}
        <AnimatePresence>
          {(showOverlay || !originalImageUrl) && (
            <motion.img
              initial={{ opacity: 0 }}
              animate={{ opacity: originalImageUrl ? opacity : 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              src={heatmapSrc}
              alt={`${methodLabel} heatmap for ${modelName}`}
              className="absolute inset-0 w-full h-full object-cover mix-blend-multiply"
              style={originalImageUrl ? { opacity } : undefined}
            />
          )}
        </AnimatePresence>

        {/* Color scale legend */}
        <div className="absolute bottom-1 right-1 flex items-center gap-1 bg-background/80 rounded px-1 py-0.5">
          <div className="w-12 h-2 rounded-sm" style={{
            background: "linear-gradient(to right, #30123b, #28459f, #1fa087, #8ed85c, #f7e225)"
          }} />
          <span className="text-[8px] text-foreground">Low → High</span>
        </div>
      </div>

      {/* Opacity slider (when overlay mode) */}
      {originalImageUrl && showOverlay && (
        <div className="flex items-center gap-2 text-xs">
          <span className="text-muted-foreground">Opacity:</span>
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.1"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="flex-1 h-1 accent-primary"
          />
          <span className="text-muted-foreground w-8">{Math.round(opacity * 100)}%</span>
        </div>
      )}
    </div>
  );
};

export default HeatmapOverlay;
