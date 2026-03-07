import { motion } from "framer-motion";
import { ExternalLink, Github, BookOpen, Linkedin, Globe } from "lucide-react";
import AppLayout from "@/components/AppLayout";
import { Button } from "@/components/ui/button";

import lukhsaanImg from "@/assets/team/lukhsaan.png";
import zuhairImg from "@/assets/team/zuhair.png";
import krishImg from "@/assets/team/krish.png";
import orianaImg from "@/assets/team/oriana.png";
import andrewImg from "@/assets/team/andrew.png";

const teamMembers = [
  {
    name: "Lukhsaan Elankumaran",
    role: "Project Lead & Web Interface",
    image: lukhsaanImg,
    linkedin: "https://linkedin.com/in/lukhsaan",
    github: "https://github.com/lukhsaankumar",
    portfolio: "https://lukhsaankumar.com",
  },
  {
    name: "Md Nafieu Hossain Alif",
    role: "ML Engineer",
    image: null,
    linkedin: "https://www.linkedin.com/in/alifhossain86/",
    github: "https://github.com/coolguy-stack",
  },
  {
    name: "Zuhair Qureshi",
    role: "ML Engineer",
    image: zuhairImg,
    linkedin: "https://www.linkedin.com/in/zuhair-qureshi/",
    github: "https://github.com/ZuhairQureshi",
    portfolio: "https://zuhairqureshi.netlify.app/",
  },
  {
    name: "Krish Bhagirath",
    role: "ML Engineer",
    image: krishImg,
    linkedin: "https://www.linkedin.com/in/krish-bhagirath/",
    github: "https://github.com/krishbhagirath",
    portfolio: "https://krish-bhagirath.vercel.app",
  },
  {
    name: "Oriana Rueckert",
    role: "Data Engineer",
    image: orianaImg,
    linkedin: "https://www.linkedin.com/in/oriana-rueckert/",
    github: "https://github.com/orianarueckert",
  },
  {
    name: "Vihaan Singhal",
    role: "Data & Preprocessing Engineer",
    image: null,
    linkedin: "https://www.linkedin.com/in/vihaan-singhal-21baa6379/",
    github: "https://github.com/Vihaan-Singhal1",
  },
  {
    name: "Andrew Wu",
    role: "ML Engineer",
    image: andrewImg,
    linkedin: "https://www.linkedin.com/in/andrew-wu13/",
    github: "https://github.com/andrewwu13",
    portfolio: "https://www.andrewwu.ca/",
  },
];

const links = [
  {
    label: "GitHub Repository",
    url: "https://github.com/McMasterAI-Society/DeepFakeDetector",
    icon: Github,
  },
  {
    label: "HuggingFace Models",
    url: "https://huggingface.co/DeepFakeDetector",
    icon: ExternalLink,
  },
  {
    label: "CUCAI 2026 Paper",
    url: "https://cucai.ca/",
    icon: BookOpen,
  },
  {
    label: "Post-CUCAI Paper (Coming Soon)",
    url: "#",
    icon: BookOpen,
  },
];

const fadeUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
};

const About = () => {
  return (
    <AppLayout>
      <div className="max-w-3xl mx-auto space-y-12 py-4">
        {/* Header */}
        <motion.div {...fadeUp} transition={{ duration: 0.5 }} className="space-y-4">
          <h2 className="text-3xl font-bold tracking-tight text-foreground">About</h2>
          <p className="text-muted-foreground leading-relaxed">
            <span className="font-semibold text-foreground">DeepFakeDetector</span> is a multi-branch fusion framework for cross-generator detection of AI-generated images, developed by members of the{" "}
            <a href="https://www.mcmasterai.ca/" target="_blank" rel="noopener noreferrer" className="font-semibold text-primary hover:underline">
              McMaster Artificial Intelligence Society (MacAI)
            </a>.
          </p>
          <p className="text-muted-foreground leading-relaxed">
            AI-generated imagery from diffusion and GAN-based models is now photorealistic, enabling misinformation, fraud, and non-consensual synthetic media. A key challenge is <span className="italic">cross-generator generalization</span>: detectors that overfit to generators seen during training often fail on new architectures.
          </p>
        </motion.div>

        {/* Approach */}
        <motion.div {...fadeUp} transition={{ delay: 0.1, duration: 0.5 }} className="space-y-4">
          <h3 className="text-xl font-semibold text-foreground">Our Approach</h3>
          <p className="text-muted-foreground leading-relaxed">
            DeepFakeDetector combines complementary forensic cues through four branches:
          </p>
          <ul className="list-disc list-inside text-muted-foreground space-y-2 ml-2">
            <li>A fine-tuned <span className="font-medium text-foreground">Vision Transformer (ViT)</span> for global semantic consistency</li>
            <li>A distilled <span className="font-medium text-foreground">DeiT</span> model for efficient inference</li>
            <li>A transfer-based <span className="font-medium text-foreground">EfficientNet-B0</span> CNN baseline for robust convolutional features</li>
            <li>A <span className="font-medium text-foreground">Gradient Field CNN</span> that exploits structure-tensor coherence in image gradients</li>
          </ul>
          <p className="text-muted-foreground leading-relaxed">
            Individual branches range from 72.69% (frozen ViT) to 87.90% (EfficientNet-B0) accuracy on an OpenFake subset. Fusion is performed via logistic regression stacking and a meta-classifier to produce a single calibrated probability.
          </p>
        </motion.div>

        {/* Team */}
        <motion.div
          {...fadeUp}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="space-y-4"
        >
          <h3 className="text-xl font-semibold text-foreground">The Team</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {teamMembers.map((member, i) => (
              <motion.div
                key={member.name}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.25 + i * 0.08 }}
                className="rounded-lg border border-border bg-card p-4 space-y-3"
              >
                <div className="flex items-center gap-3">
                  {member.image ? (
                    <img
                      src={member.image}
                      alt={member.name}
                      className="w-12 h-12 rounded-full object-cover"
                    />
                  ) : (
                    <div className="w-12 h-12 rounded-full bg-accent flex items-center justify-center text-sm font-bold text-accent-foreground">
                      {member.name.split(" ").map((n) => n[0]).join("")}
                    </div>
                  )}
                  <div>
                    <p className="text-sm font-medium text-foreground">{member.name}</p>
                    <p className="text-xs text-muted-foreground">{member.role}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {member.linkedin && (
                    <a href={member.linkedin} target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-foreground transition-colors">
                      <Linkedin className="w-4 h-4" />
                    </a>
                  )}
                  {member.github && (
                    <a href={member.github} target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-foreground transition-colors">
                      <Github className="w-4 h-4" />
                    </a>
                  )}
                  {"portfolio" in member && member.portfolio && (
                    <a href={member.portfolio} target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-foreground transition-colors">
                      <Globe className="w-4 h-4" />
                    </a>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* CUCAI */}
        <motion.div
          {...fadeUp}
          transition={{ delay: 0.3, duration: 0.5 }}
          className="rounded-lg border border-border bg-card p-6 space-y-3"
        >
          <h3 className="text-xl font-semibold text-foreground">
            <a href="https://cucai.ca/" target="_blank" rel="noopener noreferrer" className="hover:text-primary transition-colors">
              CUCAI 2026
            </a>
          </h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Submitted to the Canadian Undergraduate Conference on Artificial Intelligence 2026. Our work presents a multi-branch fusion framework combining Vision Transformers, distilled DeiT, EfficientNet-B0, and a novel Gradient Field CNN for robust cross-generator deepfake detection. The system includes explainability tools like Grad-CAM and attention heatmaps, and is evaluated against the AI-GenBench benchmark.
          </p>
        </motion.div>

        {/* Links */}
        <motion.div
          {...fadeUp}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="space-y-4"
        >
          <h3 className="text-xl font-semibold text-foreground">Resources</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {links.map((link) => (
              <a
                key={link.label}
                href={link.url}
                target="_blank"
                rel="noopener noreferrer"
                className={`group ${link.url === "#" ? "pointer-events-none opacity-60" : ""}`}
              >
                <Button
                  variant="outline"
                  className="w-full justify-start gap-3 h-auto py-3 text-left"
                >
                  <link.icon className="w-4 h-4 flex-shrink-0 text-muted-foreground group-hover:text-foreground transition-colors" />
                  <span className="text-sm">{link.label}</span>
                </Button>
              </a>
            ))}
          </div>
        </motion.div>
      </div>
    </AppLayout>
  );
};

export default About;
