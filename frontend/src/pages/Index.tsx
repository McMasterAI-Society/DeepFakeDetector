import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ScanSearch, AlertCircle, RotateCcw } from "lucide-react";
import AppLayout from "@/components/AppLayout";
import UploadDropzone from "@/components/UploadDropzone";
import AdvancedOptions, { type AdvancedOptionsState } from "@/components/AdvancedOptions";
import ResultsPanel from "@/components/ResultsPanel";
import ResultsSkeleton from "@/components/ResultsSkeleton";
import AnalyzingOverlay from "@/components/AnalyzingOverlay";
import { Button } from "@/components/ui/button";

type AppState = "idle" | "ready" | "loading" | "success" | "error";

interface PredictionResult {
  final: { pred: "real" | "fake"; pred_int: number; prob_fake: number };
  fusion_used: boolean;
  submodels: Record<string, { pred: "real" | "fake"; pred_int: number; prob_fake: number }> | null;
  timing_ms: { total: number; inference?: number; fusion?: number };
}

const BASE_URL = import.meta.env.VITE_API_BASE_URL;

const Index = () => {
  const [state, setState] = useState<AppState>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string>("");
  const [options, setOptions] = useState<AdvancedOptionsState>({
    useFusion: true,
    returnSubmodels: true,
    model: "test-random-a",
  });

  const handleFileSelect = useCallback((f: File | null) => {
    setFile(f);
    setState(f ? "ready" : "idle");
    setResult(null);
    setError("");
  }, []);

  const analyze = useCallback(async () => {
    if (!file) return;

    if (!BASE_URL) {
      setError("API base URL not configured. Set VITE_API_BASE_URL in your .env file.");
      setState("error");
      return;
    }

    setState("loading");
    setResult(null);
    setError("");

    const params = new URLSearchParams();
    params.set("use_fusion", String(options.useFusion));
    params.set("return_submodels", String(options.returnSubmodels));
    if (!options.useFusion) {
      params.set("model", options.model);
    }

    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await fetch(`${BASE_URL}/predict?${params.toString()}`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status} ${res.statusText}`);
      }

      const data: PredictionResult = await res.json();
      setResult(data);
      setState("success");
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred.");
      setState("error");
    }
  }, [file, options]);

  const isLoading = state === "loading";

  return (
    <AppLayout>
      <div className="max-w-xl mx-auto space-y-6">
        {/* Upload card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-xl border border-border bg-card p-6 space-y-5 shadow-lg shadow-background/50"
        >
          <UploadDropzone file={file} onFileSelect={handleFileSelect} disabled={isLoading} />

          <AdvancedOptions options={options} onChange={setOptions} disabled={isLoading} />

          <Button
            onClick={analyze}
            disabled={state === "idle" || isLoading}
            aria-label="Analyze uploaded image for deepfake detection"
            className="w-full gap-2 h-11 text-sm font-semibold"
          >
            <ScanSearch className="w-4 h-4" />
            {isLoading ? "Analyzing…" : "Analyze Image"}
          </Button>
        </motion.div>

        {/* Loading state */}
        <AnimatePresence>
          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <AnalyzingOverlay />
              <ResultsSkeleton />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error state */}
        <AnimatePresence>
          {state === "error" && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              role="alert"
              className="rounded-lg border border-destructive/30 bg-destructive/10 p-4 flex items-start gap-3"
            >
              <AlertCircle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
              <div className="flex-1 min-w-0">
                <p className="text-sm text-foreground font-medium">Analysis failed</p>
                <p className="text-xs text-muted-foreground mt-1">{error}</p>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={analyze}
                aria-label="Retry analysis"
                className="flex-shrink-0 gap-1 text-destructive hover:text-destructive"
              >
                <RotateCcw className="w-3 h-3" />
                Retry
              </Button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results */}
        <AnimatePresence>
          {state === "success" && result && (
            <ResultsPanel result={result} showSubmodels={options.returnSubmodels} />
          )}
        </AnimatePresence>
      </div>
    </AppLayout>
  );
};

export default Index;
