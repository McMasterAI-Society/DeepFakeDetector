import { motion } from "framer-motion";
import { ExternalLink, Github, BookOpen, Linkedin, Globe } from "lucide-react";
import AppLayout from "@/components/AppLayout";
import { Button } from "@/components/ui/button";

import lukhsaanImg from "@/assets/team/lukhsaan.png";
import zuhairImg from "@/assets/team/zuhair.png";
import krishImg from "@/assets/team/krish.png";
import orianaImg from "@/assets/team/oriana.png";
import andrewImg from "@/assets/team/andrew.png";
import vihaanImg from "@/assets/team/vihaan.png";
import alifImg from "@/assets/team/alif.png";

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
    image: alifImg,
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
    image: vihaanImg,
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
      <div className="max-w-3xl mx-auto space-y-14 py-4 px-1 sm:px-0">
        {/* Header */}
        <motion.section {...fadeUp} transition={{ duration: 0.5 }} className="space-y-5" aria-labelledby="about-heading">
          <h2 id="about-heading" className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground">About</h2>
          <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
            <span className="font-semibold text-foreground">DeepFakeDetector</span> is a multi-branch fusion framework for cross-generator detection of AI-generated images, developed by members of the{" "}
            <a href="https://www.mcmasterai.ca/" target="_blank" rel="noopener noreferrer" className="font-semibold text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background rounded-sm">
              McMaster Artificial Intelligence Society (MacAI)
            </a>.
          </p>
          <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
            AI-generated imagery from diffusion and GAN-based models is now photorealistic, enabling misinformation, fraud, and non-consensual synthetic media. A key challenge is <span className="italic">cross-generator generalization</span>: detectors that overfit to generators seen during training often fail on new architectures.
          </p>
        </motion.section>

        {/* Approach */}
        <motion.section {...fadeUp} transition={{ delay: 0.1, duration: 0.5 }} className="space-y-4" aria-labelledby="approach-heading">
          <h3 id="approach-heading" className="text-2xl font-semibold text-foreground">Our Approach</h3>
          <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
            DeepFakeDetector combines complementary forensic cues through four branches:
          </p>
          <ul className="list-disc list-inside text-base sm:text-lg text-muted-foreground space-y-2 ml-2">
            <li>A fine-tuned <span className="font-medium text-foreground">Vision Transformer (ViT)</span> for global semantic consistency</li>
            <li>A distilled <span className="font-medium text-foreground">DeiT</span> model for efficient inference</li>
            <li>A transfer-based <span className="font-medium text-foreground">EfficientNet-B0</span> CNN baseline for robust convolutional features</li>
            <li>A <span className="font-medium text-foreground">Gradient Field CNN</span> that exploits structure-tensor coherence in image gradients</li>
          </ul>
          <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
            Individual branches range from 72.69% (frozen ViT) to 87.90% (EfficientNet-B0) accuracy on an OpenFake subset. Fusion is performed via logistic regression stacking and a meta-classifier to produce a single calibrated probability.
          </p>
        </motion.section>

        {/* Team */}
        <motion.section
          {...fadeUp}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="space-y-5"
          aria-labelledby="team-heading"
        >
          <h3 id="team-heading" className="text-2xl font-semibold text-foreground">The Team</h3>
          <div className="flex flex-wrap justify-center gap-4 [&>*]:w-full [&>*]:sm:w-[calc(50%-0.5rem)] [&>*]:lg:w-[calc(33.333%-0.75rem)]">
            {teamMembers.map((member, i) => (
              <motion.article
                key={member.name}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.25 + i * 0.08 }}
                className="rounded-lg border border-border bg-card p-4 space-y-3"
                aria-label={`${member.name}, ${member.role}`}
              >
                <div className="flex items-center gap-3">
                  {member.image ? (
                    <img
                      src={member.image}
                      alt={`${member.name}'s photo`}
                      className="w-14 h-14 rounded-full object-cover"
                    />
                  ) : (
                    <div className="w-14 h-14 rounded-full bg-accent flex items-center justify-center text-base font-bold text-accent-foreground" aria-hidden="true">
                      {member.name.split(" ").map((n) => n[0]).join("")}
                    </div>
                  )}
                  <div>
                    <p className="text-base font-medium text-foreground">{member.name}</p>
                    <p className="text-sm text-muted-foreground">{member.role}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  {member.linkedin && (
                    <a href={member.linkedin} target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-foreground transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-sm" aria-label={`${member.name} LinkedIn profile`}>
                      <Linkedin className="w-5 h-5" />
                    </a>
                  )}
                  {member.github && (
                    <a href={member.github} target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-foreground transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-sm" aria-label={`${member.name} GitHub profile`}>
                      <Github className="w-5 h-5" />
                    </a>
                  )}
                  {"portfolio" in member && member.portfolio && (
                    <a href={member.portfolio} target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-foreground transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-sm" aria-label={`${member.name} portfolio`}>
                      <Globe className="w-5 h-5" />
                    </a>
                  )}
                </div>
              </motion.article>
            ))}
          </div>
        </motion.section>

        {/* CUCAI */}
        <motion.section
          {...fadeUp}
          transition={{ delay: 0.3, duration: 0.5 }}
          className="rounded-lg border border-border bg-card p-6 space-y-3"
          aria-labelledby="cucai-heading"
        >
          <h3 id="cucai-heading" className="text-2xl font-semibold text-foreground">
            <a href="https://cucai.ca/" target="_blank" rel="noopener noreferrer" className="hover:text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-sm">
              CUCAI 2026
            </a>
          </h3>
          <p className="text-base text-muted-foreground leading-relaxed">
            Submitted to the Canadian Undergraduate Conference on Artificial Intelligence 2026. Our work presents a multi-branch fusion framework combining Vision Transformers, distilled DeiT, EfficientNet-B0, and a novel Gradient Field CNN for robust cross-generator deepfake detection. The system includes explainability tools like Grad-CAM and attention heatmaps, and is evaluated against the AI-GenBench benchmark.
          </p>
        </motion.section>

        {/* Links */}
        <motion.section
          {...fadeUp}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="space-y-4"
          aria-labelledby="resources-heading"
        >
          <h3 id="resources-heading" className="text-2xl font-semibold text-foreground">Resources</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {links.map((link) => (
              <a
                key={link.label}
                href={link.url}
                target="_blank"
                rel="noopener noreferrer"
                className={`group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-md ${link.url === "#" ? "pointer-events-none opacity-60" : ""}`}
                aria-disabled={link.url === "#" ? "true" : undefined}
              >
                <Button
                  variant="outline"
                  className="w-full justify-start gap-3 h-auto py-3 text-left"
                  tabIndex={-1}
                >
                  <link.icon className="w-5 h-5 flex-shrink-0 text-muted-foreground group-hover:text-foreground transition-colors" />
                  <span className="text-base">{link.label}</span>
                </Button>
              </a>
            ))}
          </div>
        </motion.section>
      </div>
    </AppLayout>
  );
};

export default About;
