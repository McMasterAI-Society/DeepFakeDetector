import { useState } from "react";
import { motion } from "framer-motion";
import { Sparkles, ImageIcon, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

// Gemini Nano Banana Pro images
import geminiD001 from "@/assets/demo/gemini-d001.jpg";
import geminiD002 from "@/assets/demo/gemini-d002.jpg";
import geminiD003 from "@/assets/demo/gemini-d003.jpg";
import geminiD004 from "@/assets/demo/gemini-d004.jpg";
import geminiD005 from "@/assets/demo/gemini-d005.jpg";

// GPT-4o images
import gpt4oD001 from "@/assets/demo/gpt4o-d001.png";
import gpt4oD002 from "@/assets/demo/gpt4o-d002.png";
import gpt4oD003 from "@/assets/demo/gpt4o-d003.png";
import gpt4oD004 from "@/assets/demo/gpt4o-d004.png";
import gpt4oD005 from "@/assets/demo/gpt4o-d005.png";

interface DemoPrompt {
  id: string;
  shortLabel: string;
  prompt: string;
  geminiImage?: string;
  gpt4oImage?: string;
  realImage?: string;
}

const demoPrompts: DemoPrompt[] = [
  {
    id: "d001",
    shortLabel: "Rainy Train Platform",
    prompt: "Create a highly photorealistic candid image of a commuter standing on a rainy train platform at night, captured on a full-frame DSLR with an 85mm f/1.8 lens. Wet pavement reflections, subtle motion blur from passing train lights, realistic skin texture with slight blemishes, damp hair strands sticking together, mixed cool fluorescent and warm sodium lighting, mild chromatic aberration near edges, natural sensor grain in darker areas.",
    geminiImage: geminiD001,
    gpt4oImage: gpt4oD001,
  },
  {
    id: "d002",
    shortLabel: "Cracked Smartphone",
    prompt: "Create a highly photorealistic macro image of a cracked smartphone screen lying on a wooden desk, shot with a 100mm macro lens. Realistic micro-scratches, dust particles on glass surface, natural light from a nearby window, shallow depth of field falloff, slight lens distortion, authentic reflections and imperfect fingerprint smudges.",
    geminiImage: geminiD002,
    gpt4oImage: gpt4oD002,
  },
  {
    id: "d003",
    shortLabel: "Mountain Valley Sunrise",
    prompt: "Create a highly photorealistic landscape image of a fog-covered mountain valley at sunrise, captured with a 70-200mm telephoto lens. Atmospheric haze layering, gradual light gradient in sky, realistic tree density variation, subtle shadow detail in valley floor, accurate depth compression and natural color grading.",
    geminiImage: geminiD003,
    gpt4oImage: gpt4oD003,
  },
  {
    id: "d004",
    shortLabel: "Commercial Kitchen",
    prompt: "Create a highly photorealistic image of a busy commercial kitchen prep station, stainless steel surfaces with visible fingerprints and smudges, steam rising from hot pan, overhead fluorescent lighting casting realistic shadows, minor grease splatter on counter, natural clutter and asymmetry.",
    geminiImage: geminiD004,
    gpt4oImage: gpt4oD004,
  },
  {
    id: "d005",
    shortLabel: "Snowy Street at Dusk",
    prompt: "Create a highly photorealistic image of a snow-covered residential street during light snowfall at dusk, shot with a 35mm lens. Falling snowflakes captured at varying motion blur lengths, cold blue ambient tones mixed with warm window lights, tire tracks in fresh snow, realistic atmospheric diffusion.",
    geminiImage: geminiD005,
    gpt4oImage: gpt4oD005,
  },
  {
    id: "d006",
    shortLabel: "Deer in Forest",
    prompt: "Create a highly photorealistic wildlife image of a deer standing near the edge of a forest clearing, captured with a 300mm telephoto lens. Natural fur texture with uneven strands, subtle breath condensation in cool air, realistic depth separation from blurred background foliage, accurate shadow falloff.",
  },
  {
    id: "d007",
    shortLabel: "Garage Workshop",
    prompt: "Create a highly photorealistic interior image of a cluttered garage workshop, concrete floor with oil stains, tools hanging slightly uneven on pegboard, single overhead bulb casting harsh directional shadows, realistic dust in air illuminated by light beam, natural color temperature imbalance.",
  },
  {
    id: "d008",
    shortLabel: "Ocean Cliff Waves",
    prompt: "Create a highly photorealistic image of ocean waves crashing against a rocky cliff during overcast weather, captured with a 70mm lens. Fine sea spray misting the air, motion blur in water while rocks remain sharp, realistic foam texture variation, muted natural color palette.",
  },
  {
    id: "d009",
    shortLabel: "Rain Through Windshield",
    prompt: "Create a highly photorealistic image of a city street viewed through a rain-covered car windshield at night, shallow focus on water droplets, distorted bokeh from streetlights, realistic interior dashboard reflection, natural sensor noise in darker areas.",
  },
  {
    id: "d010",
    shortLabel: "Coffee Mug Close-up",
    prompt: "Create a highly photorealistic close-up of a ceramic coffee mug with visible chips and minor glaze imperfections, sitting on a textured wooden table in morning light. Subtle steam rising with natural diffusion, realistic wood grain detail, accurate light bounce and shadow softness.",
  },
];

type SourceFilter = "all" | "gemini" | "gpt4o" | "real";

interface DemoImagesProps {
  onSelectImage: (imageUrl: string, fileName: string) => void;
}

const DemoImages = ({ onSelectImage }: DemoImagesProps) => {
  const [expanded, setExpanded] = useState(false);
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>("all");
  const [expandedPrompt, setExpandedPrompt] = useState<string | null>(null);

  const handleImageClick = async (imageUrl: string, label: string, source: string) => {
    const response = await fetch(imageUrl);
    const blob = await response.blob();
    const ext = imageUrl.endsWith(".png") ? "png" : "jpg";
    const file = new File([blob], `${source}-${label}.${ext}`, { type: blob.type });
    onSelectImage(URL.createObjectURL(file), file.name);

    // We need to pass the File object, so let's use a different approach
    const event = new CustomEvent("demo-image-selected", { detail: file });
    window.dispatchEvent(event);
  };

  const getVisibleImages = (prompt: DemoPrompt) => {
    const images: { src: string; source: string; label: string }[] = [];
    if ((sourceFilter === "all" || sourceFilter === "gemini") && prompt.geminiImage) {
      images.push({ src: prompt.geminiImage, source: "Gemini", label: prompt.id });
    }
    if ((sourceFilter === "all" || sourceFilter === "gpt4o") && prompt.gpt4oImage) {
      images.push({ src: prompt.gpt4oImage, source: "GPT-4o", label: prompt.id });
    }
    if ((sourceFilter === "all" || sourceFilter === "real") && prompt.realImage) {
      images.push({ src: prompt.realImage, source: "Real", label: prompt.id });
    }
    return images;
  };

  const hasAnyImage = (prompt: DemoPrompt) => getVisibleImages(prompt).length > 0;

  const filters: { value: SourceFilter; label: string; count: number }[] = [
    { value: "all", label: "All", count: demoPrompts.filter((p) => p.geminiImage || p.gpt4oImage || p.realImage).length },
    { value: "gemini", label: "Gemini", count: demoPrompts.filter((p) => p.geminiImage).length },
    { value: "gpt4o", label: "GPT-4o", count: demoPrompts.filter((p) => p.gpt4oImage).length },
    { value: "real", label: "Real", count: demoPrompts.filter((p) => p.realImage).length },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
      className="rounded-xl border border-border bg-card shadow-sm overflow-hidden"
    >
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between p-4 hover:bg-muted/30 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-primary" />
          <span className="text-sm font-semibold text-foreground">Demo Images</span>
          <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
            {demoPrompts.filter((p) => p.geminiImage || p.gpt4oImage).length} available
          </Badge>
        </div>
        {expanded ? (
          <ChevronUp className="w-4 h-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="w-4 h-4 text-muted-foreground" />
        )}
      </button>

      {expanded && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          transition={{ duration: 0.2 }}
          className="border-t border-border"
        >
          {/* Source filter tabs */}
          <div className="flex gap-1 p-3 pb-2 flex-wrap">
            {filters.map((f) => (
              <Button
                key={f.value}
                variant={sourceFilter === f.value ? "default" : "outline"}
                size="sm"
                onClick={() => setSourceFilter(f.value)}
                className="h-7 text-xs gap-1 px-2.5"
              >
                {f.label}
                <span className="opacity-60">({f.count})</span>
              </Button>
            ))}
          </div>

          <p className="px-3 text-[11px] text-muted-foreground mb-2">
            Click any image to load it into the analyzer. These AI-generated images were created with identical prompts across models.
          </p>

          {/* Prompts grid */}
          <div className="px-3 pb-3 space-y-2 max-h-[500px] overflow-y-auto">
            {demoPrompts.map((prompt) => {
              const images = getVisibleImages(prompt);
              const isExpanded = expandedPrompt === prompt.id;
              const hasImages = hasAnyImage(prompt);

              return (
                <div
                  key={prompt.id}
                  className="rounded-lg border border-border/60 bg-muted/20 overflow-hidden"
                >
                  {/* Prompt header */}
                  <button
                    onClick={() => setExpandedPrompt(isExpanded ? null : prompt.id)}
                    className="w-full flex items-center gap-2 p-2.5 hover:bg-muted/40 transition-colors text-left"
                  >
                    <span className="text-xs font-mono text-muted-foreground w-8 flex-shrink-0">
                      #{prompt.id.replace("d", "")}
                    </span>
                    <span className="text-xs font-medium text-foreground flex-1 truncate">
                      {prompt.shortLabel}
                    </span>
                    {!hasImages && (
                      <Badge variant="outline" className="text-[9px] px-1 py-0 text-muted-foreground">
                        Coming soon
                      </Badge>
                    )}
                    {hasImages && (
                      <div className="flex gap-1">
                        {prompt.geminiImage && (
                          <Badge variant="secondary" className="text-[9px] px-1 py-0">G</Badge>
                        )}
                        {prompt.gpt4oImage && (
                          <Badge variant="secondary" className="text-[9px] px-1 py-0">4o</Badge>
                        )}
                        {prompt.realImage && (
                          <Badge className="text-[9px] px-1 py-0 bg-success/20 text-success border-success/30">R</Badge>
                        )}
                      </div>
                    )}
                    {isExpanded ? (
                      <ChevronUp className="w-3 h-3 text-muted-foreground flex-shrink-0" />
                    ) : (
                      <ChevronDown className="w-3 h-3 text-muted-foreground flex-shrink-0" />
                    )}
                  </button>

                  {isExpanded && (
                    <div className="border-t border-border/40 p-2.5 space-y-2.5">
                      {/* Prompt text */}
                      <p className="text-[10px] text-muted-foreground leading-relaxed bg-muted/40 rounded p-2 font-mono">
                        {prompt.prompt}
                      </p>

                      {/* Images row */}
                      {images.length > 0 ? (
                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                          {images.map((img) => (
                            <button
                              key={`${img.source}-${img.label}`}
                              onClick={() => handleImageClick(img.src, img.label, img.source)}
                              className="group relative rounded-md overflow-hidden border border-border/40 hover:border-primary/60 transition-all hover:shadow-md aspect-[4/3]"
                            >
                              <img
                                src={img.src}
                                alt={`${img.source} generated: ${prompt.shortLabel}`}
                                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                                loading="lazy"
                              />
                              <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                              <Badge
                                className={`absolute top-1.5 left-1.5 text-[9px] px-1.5 py-0 ${
                                  img.source === "Real"
                                    ? "bg-success/90 text-success-foreground"
                                    : "bg-background/90 text-foreground"
                                }`}
                              >
                                {img.source}
                              </Badge>
                              <div className="absolute bottom-1.5 right-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                                <Badge variant="secondary" className="text-[9px] px-1.5 py-0 bg-primary text-primary-foreground">
                                  Use this
                                </Badge>
                              </div>
                            </button>
                          ))}

                          {/* Real image placeholder */}
                          {!prompt.realImage && sourceFilter !== "gemini" && sourceFilter !== "gpt4o" && (
                            <div className="rounded-md border border-dashed border-border/60 flex flex-col items-center justify-center gap-1 aspect-[4/3] bg-muted/30">
                              <ImageIcon className="w-4 h-4 text-muted-foreground/50" />
                              <span className="text-[9px] text-muted-foreground/60">Real image</span>
                              <span className="text-[8px] text-muted-foreground/40">Coming soon</span>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="text-center py-4 text-xs text-muted-foreground">
                          No images available for this filter
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};

export default DemoImages;
