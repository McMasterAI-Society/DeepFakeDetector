import { createRoot } from "react-dom/client";
import { Analytics } from "@vercel/analytics/react";
import { SpeedInsights } from "@vercel/speed-insights/react";
import App from "./App.tsx";
import { wakeBackendOnce } from "./lib/backendWarmup";
import "./index.css";

// Start waking the GPU backend as soon as the site loads (non-blocking).
wakeBackendOnce();

createRoot(document.getElementById("root")!).render(
	<>
		<App />
		<Analytics />
		<SpeedInsights />
	</>
);
