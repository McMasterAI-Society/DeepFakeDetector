import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { useState } from "react";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export interface AdvancedOptionsState {
  useFusion: boolean;
  returnSubmodels: boolean;
  model: string;
  explain: boolean;
}

interface AdvancedOptionsProps {
  options: AdvancedOptionsState;
  onChange: (options: AdvancedOptionsState) => void;
  disabled?: boolean;
}

const MODELS = ["test-random-a", "test-random-b", "test-random-c"];

const AdvancedOptions = ({ options, onChange, disabled }: AdvancedOptionsProps) => {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-lg border border-border">
      <button
        onClick={() => setOpen(!open)}
        disabled={disabled}
        aria-expanded={open}
        aria-controls="advanced-options-panel"
        className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background rounded-lg"
      >
        Advanced Options
        <motion.span animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.2 }}>
          <ChevronDown className="w-4 h-4" />
        </motion.span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            id="advanced-options-panel"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 space-y-4 border-t border-border pt-4">
              <div className="flex items-center justify-between">
                <Label htmlFor="fusion-toggle" className="text-sm text-foreground cursor-pointer">
                  Use Fusion (ensemble)
                </Label>
                <Switch
                  id="fusion-toggle"
                  checked={options.useFusion}
                  onCheckedChange={(v) => onChange({ ...options, useFusion: v })}
                  disabled={disabled}
                  aria-label="Toggle fusion ensemble mode"
                />
              </div>

              <div className="flex items-center justify-between">
                <Label htmlFor="submodel-toggle" className="text-sm text-foreground cursor-pointer">
                  Show Submodel Results
                </Label>
                <Switch
                  id="submodel-toggle"
                  checked={options.returnSubmodels}
                  onCheckedChange={(v) => onChange({ ...options, returnSubmodels: v })}
                  disabled={disabled}
                  aria-label="Toggle submodel results display"
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="explain-toggle" className="text-sm text-foreground cursor-pointer">
                    Show Explainability Heatmaps
                  </Label>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    Visualize model attention regions
                  </p>
                </div>
                <Switch
                  id="explain-toggle"
                  checked={options.explain}
                  onCheckedChange={(v) => onChange({ ...options, explain: v })}
                  disabled={disabled}
                  aria-label="Toggle explainability heatmaps"
                />
              </div>

              <AnimatePresence>
                {!options.useFusion && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="pt-1">
                      <Label htmlFor="model-select" className="text-sm text-foreground mb-2 block">
                        Select Model
                      </Label>
                      <Select
                        value={options.model}
                        onValueChange={(v) => onChange({ ...options, model: v })}
                        disabled={disabled}
                      >
                        <SelectTrigger id="model-select" className="bg-input border-border" aria-label="Select analysis model">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {MODELS.map((m) => (
                            <SelectItem key={m} value={m}>
                              {m}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default AdvancedOptions;
