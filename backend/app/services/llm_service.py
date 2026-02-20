"""
LLM Service for generating human-readable explanations of model predictions.

Uses Google Gemini to translate model-space evidence (heatmaps, attention maps)
into human-understandable hypotheses with proper hedging language.
"""

import json
import base64
from typing import Any, Dict, List, Optional
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Model type descriptions for the LLM
MODEL_TYPE_DESCRIPTIONS = {
    "cnn-transfer": {
        "type": "rgb_texture_cnn",
        "description": "Analyzes RGB pixel textures, colors, and fine details at multiple scales",
        "typical_cues": ["skin texture uniformity", "shading gradients", "fine detail at boundaries"]
    },
    "vit-base": {
        "type": "patch_consistency_vit",
        "description": "Analyzes global consistency and relationships between image patches",
        "typical_cues": ["lighting consistency", "background blur patterns", "patch-level coherence"]
    },
    "deit-distilled": {
        "type": "patch_consistency_vit",
        "description": "Analyzes global consistency with knowledge distillation for refined attention",
        "typical_cues": ["global-local consistency", "texture repetition", "depth coherence"]
    },
    "gradfield-cnn": {
        "type": "edge_coherence_cnn",
        "description": "Analyzes edge patterns, boundary sharpness, and gradient field coherence",
        "typical_cues": ["edge smoothness", "boundary naturalness", "gradient consistency"]
    }
}

SYSTEM_PROMPT = """You are an AI image analysis interpreter for a deepfake detection system. Your role is to translate model evidence into human-understandable hypotheses.

CRITICAL RULES:
1. NEVER claim certainty. Always use hedging language: "may", "suggests", "possible", "could indicate", "might show"
2. ALWAYS cite which model's evidence supports each statement (e.g., "based on CNN heatmap focus")
3. If evidence is diffuse or unclear, say so explicitly: "Evidence is spread across the image; interpretation is less certain"
4. Provide user-checkable observations, not definitive claims about what IS fake
5. Remember: you are explaining what the MODEL focused on, not proving the image is fake

MODEL TYPES AND WHAT THEY ANALYZE:
- CNN (rgb_texture_cnn): Pixel textures, colors, fine details - looks for texture anomalies
- ViT/DeiT (patch_consistency_vit): Global consistency, patch relationships - looks for coherence issues
- GradField (edge_coherence_cnn): Edge patterns, boundaries, gradient fields - looks for edge artifacts

OUTPUT FORMAT:
You must respond with valid JSON matching this exact structure:
{
  "per_model_insights": {
    "<model_name>": {
      "what_model_relied_on": "One sentence describing the model's focus area",
      "possible_cues": ["Cue 1 with hedging (based on evidence)", "Cue 2...", "Cue 3..."],
      "confidence_note": "Note about confidence level"
    }
  },
  "consensus_summary": [
    "Bullet 1 about model agreement/disagreement",
    "Bullet 2 about overall evidence pattern"
  ]
}"""


class LLMService:
    """Service for generating LLM-powered explanations of model predictions."""
    
    def __init__(self):
        self._client = None
        self._model_name = None
        self._enabled = False
        self._initialize()
    
    def _initialize(self):
        """Initialize the Gemini client if API key is available."""
        settings = get_settings()
        
        if not settings.llm_enabled:
            logger.info("LLM explanations disabled: No GOOGLE_API_KEY configured")
            return
        
        try:
            from google import genai
            self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)
            self._model_name = settings.GEMINI_MODEL
            self._enabled = True
            logger.info(f"LLM service initialized with model: {settings.GEMINI_MODEL}")
        except ImportError:
            logger.warning("google-genai package not installed. LLM explanations disabled.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
    
    @property
    def enabled(self) -> bool:
        """Check if LLM explanations are available."""
        return self._enabled
    
    def build_evidence_packet(
        self,
        model_name: str,
        model_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build a structured evidence packet from model output.
        
        Args:
            model_name: Name of the model (e.g., "cnn-transfer")
            model_output: Raw output from the model's predict() method
            
        Returns:
            Structured evidence packet for LLM consumption
        """
        model_info = MODEL_TYPE_DESCRIPTIONS.get(model_name, {
            "type": "unknown",
            "description": "Unknown model type",
            "typical_cues": []
        })
        
        return {
            "model_name": model_name,
            "model_type": model_info["type"],
            "model_description": model_info["description"],
            "prob_fake": model_output.get("prob_fake", 0.0),
            "prediction": model_output.get("pred", "unknown"),
            "focus_summary": model_output.get("focus_summary", "focus pattern not available"),
            "explainability_type": model_output.get("explainability_type", "unknown"),
            "typical_cues_for_this_model": model_info["typical_cues"]
        }
    
    def generate_explanation(
        self,
        original_image_b64: Optional[str],
        submodel_outputs: Dict[str, Dict[str, Any]],
        include_images: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Generate LLM explanation for model predictions.
        
        Args:
            original_image_b64: Base64-encoded original image (optional)
            submodel_outputs: Dict mapping model names to their outputs
            include_images: Whether to include images in the prompt (uses vision model)
            
        Returns:
            ExplanationResult dict or None if generation fails
        """
        if not self._enabled:
            logger.warning("LLM explanations requested but service not enabled")
            return None
        
        try:
            # Build evidence packets for all models
            evidence_packets = {}
            for model_name, output in submodel_outputs.items():
                evidence_packets[model_name] = self.build_evidence_packet(model_name, output)
            
            # Build the prompt
            user_prompt = self._build_user_prompt(evidence_packets, submodel_outputs)
            
            # Build content parts (text + optional images)
            content_parts = []
            
            # Add images if requested and available
            if include_images:
                # Add original image
                if original_image_b64:
                    content_parts.append({
                        "mime_type": "image/png",
                        "data": original_image_b64
                    })
                    content_parts.append("Original image shown above.\n\n")
                
                # Add heatmap overlays for each model
                for model_name, output in submodel_outputs.items():
                    if output.get("heatmap_base64"):
                        content_parts.append({
                            "mime_type": "image/png",
                            "data": output["heatmap_base64"]
                        })
                        content_parts.append(f"Heatmap overlay for {model_name} shown above.\n\n")
            
            # Add the main text prompt
            content_parts.append(user_prompt)
            
            # Call the LLM using new google.genai API
            logger.info("Generating LLM explanation...")
            from google.genai import types
            
            # Build the parts list for the new API
            parts = []
            for part in content_parts:
                if isinstance(part, dict) and "mime_type" in part:
                    # Image part
                    parts.append(types.Part.from_bytes(
                        data=__import__('base64').b64decode(part["data"]),
                        mime_type=part["mime_type"]
                    ))
                else:
                    # Text part
                    parts.append(types.Part.from_text(text=str(part)))
            
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[SYSTEM_PROMPT] + parts,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    top_p=0.8,
                    max_output_tokens=2048,
                )
            )
            
            # Parse the response
            return self._parse_response(response.text, list(submodel_outputs.keys()))
            
        except Exception as e:
            logger.error(f"Failed to generate LLM explanation: {e}")
            return None
    
    def _build_user_prompt(
        self,
        evidence_packets: Dict[str, Dict],
        submodel_outputs: Dict[str, Dict]
    ) -> str:
        """Build the user prompt with evidence data."""
        
        # Calculate some aggregate stats
        prob_fakes = [p["prob_fake"] for p in evidence_packets.values()]
        avg_prob = sum(prob_fakes) / len(prob_fakes) if prob_fakes else 0
        agreement = "Models generally agree" if max(prob_fakes) - min(prob_fakes) < 0.3 else "Models show disagreement"
        
        prompt = f"""I have {len(evidence_packets)} deepfake detection models analyzing an image.

EVIDENCE FROM EACH MODEL:
{json.dumps(evidence_packets, indent=2)}

AGGREGATE ANALYSIS:
- Average fake probability: {avg_prob:.1%}
- Model agreement: {agreement}
- Probability range: {min(prob_fakes):.1%} to {max(prob_fakes):.1%}

TASK:
For each model, provide:
1. "what_model_relied_on": One sentence describing where the model focused (cite the focus_summary)
2. "possible_cues": 2-4 possible visual cues a human could check, phrased as hypotheses with hedging language
3. "confidence_note": Assessment based on prob_fake value and focus pattern

Then provide "consensus_summary": 2-3 bullets about where models agreed/disagreed and overall evidence quality.

Remember: Use hedging language ("may", "suggests", "possible"). Never claim certainty.

Respond with valid JSON only, no markdown formatting."""

        return prompt
    
    def _parse_response(
        self,
        response_text: str,
        expected_models: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Parse and validate the LLM response."""
        
        try:
            # Try to extract JSON from the response
            # Sometimes the model wraps it in markdown code blocks
            text = response_text.strip()
            if text.startswith("```"):
                # Remove markdown code block
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
                text = text.strip()
            
            result = json.loads(text)
            
            # Validate structure
            if "per_model_insights" not in result:
                logger.warning("LLM response missing per_model_insights")
                result["per_model_insights"] = {}
            
            if "consensus_summary" not in result:
                logger.warning("LLM response missing consensus_summary")
                result["consensus_summary"] = ["Model analysis completed."]
            
            # Ensure all expected models have entries (fill with defaults if missing)
            for model_name in expected_models:
                if model_name not in result["per_model_insights"]:
                    result["per_model_insights"][model_name] = {
                        "what_model_relied_on": f"The {model_name} model analyzed the image.",
                        "possible_cues": ["Evidence details not available for this model."],
                        "confidence_note": "Unable to generate detailed analysis."
                    }
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {response_text[:500]}...")
            
            # Return a fallback response
            return {
                "per_model_insights": {
                    model: {
                        "what_model_relied_on": f"The {model} model analyzed the image.",
                        "possible_cues": ["Unable to generate detailed explanation."],
                        "confidence_note": "LLM response parsing failed."
                    }
                    for model in expected_models
                },
                "consensus_summary": ["Model analysis completed but detailed explanation unavailable."]
            }


# Global singleton
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
