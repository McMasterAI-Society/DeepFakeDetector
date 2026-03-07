import { motion } from "framer-motion";
import { TypeAnimation } from "react-type-animation";
import { ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import AppLayout from "@/components/AppLayout";
import logo from "@/assets/logo.ico";
import midjourneyLogo from "@/assets/midjourney-logo.png";
import bingLogo from "@/assets/bing-logo.png";
import geminiLogo from "@/assets/gemini-logo.png";
import stableDiffusionLogo from "@/assets/stable-diffusion-logo.png";

const modelLogos = [
  { name: "GPT-4o", icon: "https://cdn.worldvectorlogo.com/logos/openai-2.svg" },
  { name: "Midjourney", icon: midjourneyLogo },
  { name: "Bing Image Creator", icon: bingLogo },
  { name: "Gemini Nano Banana Pro", icon: geminiLogo },
  { name: "Stable Diffusion", icon: stableDiffusionLogo },
];

const Home = () => {
  return (
    <AppLayout>
      <div className="flex flex-col items-center justify-center min-h-[70vh] text-center space-y-12 px-2">
        {/* Hero */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="space-y-6 max-w-2xl"
        >
          <motion.img
            src={logo}
            alt="DeepFake Detector logo"
            className="w-20 h-20 mx-auto"
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 150, delay: 0.2 }}
          />

          <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold tracking-tight text-foreground">
            Identify{" "}
            <TypeAnimation
              sequence={[
                "AI-Generated Images",
                2000,
                "Deepfakes",
                2000,
                "Synthetic Media",
                2000,
                "Fake Photos",
                2000,
              ]}
              wrapper="span"
              cursor={true}
              repeat={Infinity}
              className="text-muted-foreground"
            />
          </h2>

          <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
            Multi-model ensemble detection for spotting AI-generated and
            deepfake images across leading generative models.
          </p>

          <Link to="/analyze">
            <Button size="lg" className="gap-2 mt-4">
              Start Analysis
              <ArrowRight className="w-4 h-4" />
            </Button>
          </Link>
        </motion.div>

        {/* Trained on */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="space-y-4 w-full max-w-xl"
          aria-label="Models trained to detect"
        >
          <p className="text-xs uppercase tracking-widest text-muted-foreground font-medium">
            Trained to detect outputs from
          </p>

          <div className="flex flex-wrap items-start justify-center gap-6 sm:gap-8">
            {modelLogos.map((model) => (
              <motion.div
                key={model.name}
                whileHover={{ scale: 1.05 }}
                className="flex flex-col items-center gap-2 w-20 sm:w-24"
              >
                <div className="w-12 h-12 rounded-lg bg-accent flex items-center justify-center p-2">
                  <img
                    src={model.icon}
                    alt={`${model.name} logo`}
                    className="w-8 h-8 object-contain grayscale"
                    loading="lazy"
                  />
                </div>
                <span className="text-[10px] sm:text-xs text-muted-foreground text-center">{model.name}</span>
              </motion.div>
            ))}
          </div>

          <p className="text-xs text-muted-foreground">
            GANs and other top generative models
          </p>

          <p className="text-[10px] text-muted-foreground/50 max-w-lg mt-4 leading-relaxed">
            The OpenAI, MidJourney, Bing Image Creator, Gemini Nano Banana Pro, and Stable Diffusion trademarks and logos are the property of their respective trademark holders. They are not affiliated with, endorsed by, or sponsored by DeepFake Detector.
          </p>
        </motion.div>
      </div>
    </AppLayout>
  );
};

export default Home;
