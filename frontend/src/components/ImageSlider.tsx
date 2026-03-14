import { useState } from "react";
import { Eye, Layers } from "lucide-react";
import { Slider } from "@/components/ui/slider";

interface ImageSliderProps {
  /** Original image URL (object URL or base64) */
  originalImageUrl: string;
  /** Heatmap image as base64 string */
  heatmapBase64: string;
  /** Alt text for images */
  modelName: string;
}

/**
 * Image with horizontal opacity slider bar at the bottom.
 * Left (0%) = pure original image, Right (100%) = full heatmap overlay.
 */
const ImageSlider = ({
  originalImageUrl,
  heatmapBase64,
  modelName,
}: ImageSliderProps) => {
  const [heatmapOpacity, setHeatmapOpacity] = useState(50);

  const heatmapSrc = `data:image/png;base64,${heatmapBase64}`;

  return (
    <div className="space-y-3">
      {/* Image container with stacked layers */}
      <div className="relative w-full aspect-square rounded-lg overflow-hidden bg-muted">
        {/* Original image (fades out as slider moves right) */}
        <img
          src={originalImageUrl}
          alt={`Original image for ${modelName}`}
          className="absolute inset-0 w-full h-full object-cover"
          style={{ opacity: 1 - heatmapOpacity / 100 }}
          draggable={false}
        />
        
        {/* Heatmap (fades in as slider moves right) */}
        <img
          src={heatmapSrc}
          alt={`Heatmap for ${modelName}`}
          className="absolute inset-0 w-full h-full object-cover"
          style={{ opacity: heatmapOpacity / 100 }}
          draggable={false}
        />

        {/* Color scale legend */}
        <div className="absolute bottom-2 right-2 flex items-center gap-1 bg-background/90 backdrop-blur-sm rounded px-2 py-1 pointer-events-none">
          <div
            className="w-12 h-2 rounded-sm"
            style={{
              background:
                "linear-gradient(to right, #30123b, #28459f, #1fa087, #8ed85c, #f7e225)",
            }}
          />
          <span className="text-[9px] font-medium text-foreground">Low → High</span>
        </div>
      </div>

      {/* Horizontal opacity slider bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <Eye className="w-3 h-3" />
            <span>Original</span>
          </div>
          <span className="text-[10px] font-medium">{heatmapOpacity}%</span>
          <div className="flex items-center gap-1">
            <Layers className="w-3 h-3" />
            <span>Heatmap</span>
          </div>
        </div>
        
        <Slider
          value={[heatmapOpacity]}
          onValueChange={(values) => setHeatmapOpacity(values[0])}
          min={0}
          max={100}
          step={1}
          className="w-full"
          aria-label={`Heatmap opacity for ${modelName}`}
        />
      </div>
    </div>
  );
};

export default ImageSlider;
