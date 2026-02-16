import { motion } from "framer-motion";

const ResultsSkeleton = () => (
  <div className="space-y-4" aria-label="Loading results" role="status">
    {/* Verdict skeleton */}
    <div className="rounded-lg border border-border bg-muted/20 p-5">
      <div className="flex items-center justify-center gap-2 mb-2">
        <div className="w-6 h-6 rounded bg-muted animate-pulse" />
        <div className="w-32 h-5 rounded bg-muted animate-pulse" />
      </div>
      <div className="w-48 h-3 rounded bg-muted animate-pulse mx-auto mt-2" />
    </div>

    {/* Confidence skeleton */}
    <div className="rounded-lg border border-border bg-card p-4 space-y-3">
      <div className="w-20 h-4 rounded bg-muted animate-pulse" />
      {[1, 2].map((i) => (
        <div key={i} className="space-y-1">
          <div className="flex justify-between">
            <div className="w-36 h-3 rounded bg-muted animate-pulse" />
            <div className="w-10 h-3 rounded bg-muted animate-pulse" />
          </div>
          <div className="h-2 rounded-full bg-muted overflow-hidden">
            <motion.div
              className="h-full bg-muted-foreground/20 rounded-full"
              initial={{ width: "0%" }}
              animate={{ width: "60%" }}
              transition={{ duration: 1.5, repeat: Infinity, repeatType: "reverse" }}
            />
          </div>
        </div>
      ))}
    </div>

    <span className="sr-only">Analyzing image, please wait…</span>
  </div>
);

export default ResultsSkeleton;
