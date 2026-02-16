import { useCallback, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, X, Image as ImageIcon, FileType } from "lucide-react";
import { Button } from "@/components/ui/button";

interface UploadDropzoneProps {
  file: File | null;
  onFileSelect: (file: File | null) => void;
  disabled?: boolean;
}

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp"];
const MAX_SIZE_MB = 10;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const UploadDropzone = ({ file, onFileSelect, disabled }: UploadDropzoneProps) => {
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const validateAndSet = useCallback(
    (f: File) => {
      setError(null);
      if (!ACCEPTED_TYPES.includes(f.type)) {
        setError("Unsupported file type. Use JPEG, PNG, GIF, WebP, or BMP.");
        return;
      }
      if (f.size > MAX_SIZE_BYTES) {
        setError(`File too large. Max ${MAX_SIZE_MB}MB.`);
        return;
      }
      const url = URL.createObjectURL(f);
      setPreview(url);
      onFileSelect(f);
    },
    [onFileSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (disabled) return;
      const f = e.dataTransfer.files?.[0];
      if (f) validateAndSet(f);
    },
    [disabled, validateAndSet]
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) validateAndSet(f);
    if (inputRef.current) inputRef.current.value = "";
  };

  const removeFile = () => {
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setError(null);
    onFileSelect(null);
  };

  return (
    <div className="space-y-3">
      <AnimatePresence mode="wait">
        {!file ? (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            onDragOver={(e) => {
              e.preventDefault();
              if (!disabled) setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => !disabled && inputRef.current?.click()}
            onKeyDown={(e) => {
              if ((e.key === "Enter" || e.key === " ") && !disabled) {
                e.preventDefault();
                inputRef.current?.click();
              }
            }}
            role="button"
            tabIndex={0}
            aria-label="Upload image for analysis. Drag and drop or click to browse."
            className={`
              relative cursor-pointer rounded-lg border-2 border-dashed p-8 text-center transition-all duration-200
              focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background
              ${disabled ? "opacity-50 cursor-not-allowed" : ""}
              ${dragOver ? "border-primary bg-primary/5 glow-primary" : "border-border hover:border-primary/50 hover:bg-muted/30"}
            `}
          >
            <div className="flex flex-col items-center gap-3">
              <motion.div
                animate={dragOver ? { scale: 1.1, y: -4 } : { scale: 1, y: 0 }}
                className="rounded-full bg-muted p-3"
              >
                <Upload className="w-6 h-6 text-muted-foreground" />
              </motion.div>
              <div>
                <p className="text-sm font-medium text-foreground">
                  Drop your image here
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  or click to browse · JPEG, PNG, GIF, WebP, BMP · Max {MAX_SIZE_MB}MB
                </p>
              </div>
            </div>
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED_TYPES.join(",")}
              onChange={handleChange}
              className="hidden"
              aria-hidden="true"
              disabled={disabled}
            />
          </motion.div>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="relative rounded-lg border border-border bg-muted/20 p-4"
          >
            <div className="flex items-start gap-4">
              {preview && (
                <div className="w-20 h-20 rounded-md overflow-hidden flex-shrink-0 bg-muted">
                  <img
                    src={preview}
                    alt="Preview of uploaded image"
                    className="w-full h-full object-contain"
                  />
                </div>
              )}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-foreground truncate">
                  {file.name}
                </p>
                <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <FileType className="w-3 h-3" />
                    {file.type.split("/")[1].toUpperCase()}
                  </span>
                  <span className="flex items-center gap-1">
                    <ImageIcon className="w-3 h-3" />
                    {formatSize(file.size)}
                  </span>
                </div>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={removeFile}
                disabled={disabled}
                aria-label="Remove selected file"
                className="flex-shrink-0 h-8 w-8 text-muted-foreground hover:text-foreground"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {error && (
        <motion.p
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-xs text-destructive"
          role="alert"
        >
          {error}
        </motion.p>
      )}
    </div>
  );
};

export default UploadDropzone;
