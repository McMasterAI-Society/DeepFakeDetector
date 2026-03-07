import { ReactNode } from "react";
import { motion } from "framer-motion";
import { Link, useLocation } from "react-router-dom";
import logo from "@/assets/logo.ico";
import { cn } from "@/lib/utils";

interface AppLayoutProps {
  children: ReactNode;
}

const navItems = [
  { to: "/", label: "Home" },
  { to: "/analyze", label: "Analyze" },
  { to: "/about", label: "About" },
];

const AppLayout = ({ children }: AppLayoutProps) => {
  const location = useLocation();

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-border/50 backdrop-blur-sm sticky top-0 z-50 bg-background/80">
        <div className="container max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3 group" aria-label="Home">
            <motion.img
              src={logo}
              alt="DeepFake Detector logo"
              className="w-8 h-8"
              initial={{ rotate: -10, scale: 0.9 }}
              animate={{ rotate: 0, scale: 1 }}
              transition={{ type: "spring", stiffness: 200 }}
            />
            <div>
              <h1 className="text-lg font-bold tracking-tight text-foreground">
                DeepFake Detector
              </h1>
            </div>
          </Link>

          <nav className="flex items-center gap-1" aria-label="Main navigation">
            {navItems.map((item) => (
              <Link
                key={item.to}
                to={item.to}
                className={cn(
                  "px-3 py-1.5 text-sm rounded-md transition-colors",
                  location.pathname === item.to
                    ? "bg-primary text-primary-foreground font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent"
                )}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
      </header>

      <main className="flex-1 container max-w-5xl mx-auto px-4 py-8">
        {children}
      </main>

      <footer className="border-t border-border/50 py-4">
        <div className="container max-w-5xl mx-auto px-4">
          <p className="text-xs text-muted-foreground text-center">
            DeepFake Detector · MACAI Society · CUCAI 2026
          </p>
        </div>
      </footer>
    </div>
  );
};

export default AppLayout;
