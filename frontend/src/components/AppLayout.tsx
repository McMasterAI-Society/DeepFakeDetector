import { ReactNode } from "react";
import { motion } from "framer-motion";
import { Shield } from "lucide-react";

interface AppLayoutProps {
  children: ReactNode;
}

const AppLayout = ({ children }: AppLayoutProps) => {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-border/50 backdrop-blur-sm sticky top-0 z-50 bg-background/80">
        <div className="container max-w-4xl mx-auto px-4 py-4 flex items-center gap-3">
          <motion.div
            initial={{ rotate: -10, scale: 0.9 }}
            animate={{ rotate: 0, scale: 1 }}
            transition={{ type: "spring", stiffness: 200 }}
          >
            <Shield className="w-7 h-7 text-primary" />
          </motion.div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-foreground">
              DeepFake Detector
            </h1>
            <p className="text-xs text-muted-foreground">
              AI-powered image authenticity analysis
            </p>
          </div>
        </div>
      </header>

      <main className="flex-1 container max-w-4xl mx-auto px-4 py-8">
        {children}
      </main>

      <footer className="border-t border-border/50 py-4">
        <div className="container max-w-4xl mx-auto px-4">
          <p className="text-xs text-muted-foreground text-center">
            DeepFake Detector — For research and educational purposes
          </p>
        </div>
      </footer>
    </div>
  );
};

export default AppLayout;
