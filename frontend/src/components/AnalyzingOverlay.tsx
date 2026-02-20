import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";

const AnalyzingOverlay = () => (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    exit={{ opacity: 0 }}
    className="flex items-center justify-center gap-3 py-4"
  >
    <motion.div
      animate={{ rotate: 360 }}
      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
    >
      <Loader2 className="w-5 h-5 text-primary" />
    </motion.div>
    <span className="text-sm text-muted-foreground">Analyzing image…</span>
  </motion.div>
);

export default AnalyzingOverlay;
