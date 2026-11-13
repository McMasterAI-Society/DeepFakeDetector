import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageSquareHeart, X, Check, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

type UserLabel = "ai" | "real" | "unsure";
type Verdict = "yes" | "no" | "unsure";

interface FeedbackCardProps {
  predictionId: string;
  baseUrl: string;
}

const ANSWERED_KEY = "dfd_feedback_answered";
const NOTE_MAX = 500;

const readAnswered = (): string[] => {
  try {
    return JSON.parse(sessionStorage.getItem(ANSWERED_KEY) || "[]");
  } catch {
    return [];
  }
};

const markAnswered = (id: string) => {
  const ids = readAnswered().filter((x) => x !== id);
  ids.push(id);
  sessionStorage.setItem(ANSWERED_KEY, JSON.stringify(ids.slice(-50)));
};

const FeedbackCard = ({ predictionId, baseUrl }: FeedbackCardProps) => {
  const [hidden, setHidden] = useState(() => readAnswered().includes(predictionId));
  const [label, setLabel] = useState<UserLabel | null>(null);
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const [note, setNote] = useState("");
  const [noteSent, setNoteSent] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const send = async (userLabel: UserLabel, sourceNote?: string): Promise<Verdict | null> => {
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch(`${baseUrl}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prediction_id: predictionId,
          user_knows_truth: userLabel !== "unsure",
          user_label: userLabel,
          source_note: sourceNote?.trim() || null,
        }),
      });
      if (res.status === 429) throw new Error("Too many requests, please try again shortly.");
      if (!res.ok) throw new Error("Could not send feedback.");
      const data = await res.json();
      markAnswered(predictionId);
      return data.verdict_correct as Verdict;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not send feedback.");
      return null;
    } finally {
      setSubmitting(false);
    }
  };

  const choose = async (userLabel: UserLabel) => {
    const result = await send(userLabel);
    if (result) {
      setLabel(userLabel);
      setVerdict(result);
    }
  };

  const sendNote = async () => {
    if (!label || !note.trim()) return;
    const result = await send(label, note);
    if (result) setNoteSent(true);
  };

  if (hidden) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0 }}
        className="rounded-lg border border-border bg-card p-4 space-y-3"
        aria-label="Feedback on this result"
      >
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            <MessageSquareHeart className="w-4 h-4 text-primary flex-shrink-0" aria-hidden="true" />
            <h3 className="text-sm font-medium text-foreground">
              {label ? "Thanks for your feedback!" : "Do you know whether this image is AI-generated?"}
            </h3>
          </div>
          <button
            type="button"
            onClick={() => setHidden(true)}
            aria-label="Dismiss feedback question"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {!label && (
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" disabled={submitting} onClick={() => choose("ai")}>
              Yes, it's AI
            </Button>
            <Button size="sm" variant="outline" disabled={submitting} onClick={() => choose("real")}>
              Yes, it's real
            </Button>
            <Button size="sm" variant="ghost" disabled={submitting} onClick={() => choose("unsure")}>
              I don't know
            </Button>
            {submitting && <Loader2 className="w-4 h-4 animate-spin text-muted-foreground self-center" />}
          </div>
        )}

        {label && verdict && verdict !== "unsure" && (
          <p className="text-xs text-muted-foreground">
            {verdict === "yes"
              ? "Thanks — we got it right."
              : "Thanks — we got it wrong. Your answer helps us improve the detector."}
          </p>
        )}
        {label === "unsure" && (
          <p className="text-xs text-muted-foreground">No problem — thanks for letting us know.</p>
        )}

        {label && label !== "unsure" && !noteSent && (
          <div className="space-y-2">
            <label htmlFor="feedback-source" className="text-xs text-muted-foreground">
              Where is it from? (optional — e.g. the generator, camera or website)
            </label>
            <div className="flex gap-2">
              <Input
                id="feedback-source"
                value={note}
                maxLength={NOTE_MAX}
                onChange={(e) => setNote(e.target.value)}
                placeholder="e.g. Midjourney v6, iPhone 15, a news site"
                className="h-9 text-sm"
              />
              <Button size="sm" disabled={submitting || !note.trim()} onClick={sendNote}>
                Send
              </Button>
            </div>
          </div>
        )}
        {noteSent && (
          <p className="text-xs text-muted-foreground flex items-center gap-1">
            <Check className="w-3 h-3" /> Source noted.
          </p>
        )}

        {error && <p className="text-xs text-destructive">{error}</p>}
      </motion.div>
    </AnimatePresence>
  );
};

export default FeedbackCard;
