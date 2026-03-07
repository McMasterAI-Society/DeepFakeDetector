import { ReactNode } from "react";
import { motion } from "framer-motion";
import { Link, useLocation } from "react-router-dom";
import logo from "@/assets/logo.png";
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
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-[100] focus:bg-primary focus:text-primary-foreground focus:px-4 focus:py-2 focus:rounded-md focus:text-sm"
      >
        Skip to main content
      </a>

      <header className="border-b border-border/50 backdrop-blur-sm sticky top-0 z-50 bg-background/80">
        <div className="container max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2 sm:gap-3 group" aria-label="DeepFake Detector Home">
            <motion.img
              src={logo}
              alt=""
              aria-hidden="true"
              className="w-7 h-7 sm:w-8 sm:h-8"
              initial={{ rotate: -10, scale: 0.9 }}
              animate={{ rotate: 0, scale: 1 }}
              transition={{ type: "spring", stiffness: 200 }}
            />
            <div>
              <h1 className="text-base sm:text-lg font-bold tracking-tight text-foreground">
                DeepFake Detector
              </h1>
            </div>
          </Link>

          <nav className="flex items-center gap-0.5 sm:gap-1" aria-label="Main navigation">
            {navItems.map((item) => (
              <Link
                key={item.to}
                to={item.to}
                aria-current={location.pathname === item.to ? "page" : undefined}
                className={cn(
                  "px-2.5 sm:px-3 py-1.5 text-sm rounded-md transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
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

      <main id="main-content" className="flex-1 container max-w-5xl mx-auto px-4 py-6 sm:py-8">
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
