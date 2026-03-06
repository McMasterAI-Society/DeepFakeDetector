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

# User-facing display information for each model (used in frontend)
MODEL_DISPLAY_INFO = {
    "cnn-transfer": {
        "display_name": "Texture Analysis",
        "short_name": "CNN",
        "method_name": "Grad-CAM",
        "method_description": "Gradient-weighted Class Activation Mapping",
        "educational_text": (
            "This model examines fine-grained texture patterns and pixel-level details. "
            "The heatmap highlights regions where texture anomalies were detected. "
            "AI-generated images often have subtle texture inconsistencies - overly smooth skin, "
            "unnatural fabric patterns, or repetitive background textures that this model can detect."
        ),
        "what_it_looks_for": [
            "Skin texture uniformity vs natural variation",
            "Fine detail preservation at edges and boundaries",
            "Color gradient smoothness and shading realism"
        ]
    },
    "vit-base": {
        "display_name": "Patch Consistency",
        "short_name": "ViT",
        "method_name": "Attention Rollout",
        "method_description": "Aggregated attention across all transformer layers",
        "educational_text": (
            "This model analyzes how different parts of the image relate to each other. "
            "The heatmap shows which image patches drew the most attention. "
            "AI-generated images may have inconsistencies between regions - "
            "mismatched lighting, perspective errors, or elements that don't quite fit together."
        ),
        "what_it_looks_for": [
            "Consistency of lighting across the image",
            "Spatial relationships between objects",
            "Background-foreground coherence"
        ]
    },
    "deit-distilled": {
        "display_name": "Global Structure",
        "short_name": "DeiT",
        "method_name": "Attention Rollout",
        "method_description": "Distilled attention patterns from teacher model",
        "educational_text": (
            "This model uses knowledge distillation to detect global structural anomalies. "
            "The heatmap reveals areas where the overall image structure seems inconsistent. "
            "AI-generated images sometimes have subtle global issues - "
            "like depth inconsistencies or anatomical improbabilities."
        ),
        "what_it_looks_for": [
            "Global-to-local consistency",
            "Depth and perspective coherence",
            "Structural plausibility of objects"
        ]
    },
    "gradfield-cnn": {
        "display_name": "Edge Coherence",
        "short_name": "GradField",
        "method_name": "Gradient Field Analysis",
        "method_description": "Analysis of image gradient patterns and edge transitions",
        "educational_text": (
            "This model analyzes edge patterns and how colors transition across boundaries. "
            "The heatmap highlights areas with unusual edge characteristics. "
            "AI-generated images often have telltale edge artifacts - "
            "unnaturally sharp or blurry boundaries, inconsistent edge directions, or gradient anomalies."
        ),
        "what_it_looks_for": [
            "Edge sharpness consistency",
            "Natural boundary transitions",
            "Gradient flow coherence"
        ]
    }
}

def get_model_display_info(model_name: str) -> Dict[str, Any]:
    """Get display info for a model, with fallback for unknown models."""
    return MODEL_DISPLAY_INFO.get(model_name, {
        "display_name": model_name.replace("-", " ").title(),
        "short_name": model_name[:3].upper(),
        "method_name": "Analysis",
        "method_description": "Model-specific analysis",
        "educational_text": f"This model ({model_name}) analyzes the image for signs of AI generation.",
        "what_it_looks_for": ["Image anomalies", "Generation artifacts"]
    })

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
    
    def generate_single_model_explanation(
        self,
        model_name: str,
        prob_fake: float,
        original_image_b64: Optional[str] = None,
        heatmap_b64: Optional[str] = None,
        focus_summary: Optional[str] = None,
        contribution_percentage: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate LLM explanation for a single model's prediction.
        
        This is more token-efficient than generating all explanations at once,
        and allows users to request explanations on-demand per model.
        
        Args:
            model_name: Name of the model (e.g., "cnn-transfer")
            prob_fake: The model's fake probability
            original_image_b64: Base64-encoded original image
            heatmap_b64: Base64-encoded heatmap overlay
            focus_summary: Text summary of where model focused
            contribution_percentage: How much this model contributed to fusion decision
            
        Returns:
            Dict with insight for this model or None if generation fails
        """
        if not self._enabled:
            logger.warning("LLM explanations requested but service not enabled")
            return None
        
        try:
            # Get display info for this model
            display_info = get_model_display_info(model_name)
            model_type_info = MODEL_TYPE_DESCRIPTIONS.get(model_name, {
                "type": "unknown",
                "description": "Unknown model type",
                "typical_cues": []
            })
            
            # Build focused prompt for single model
            prompt = f"""You are analyzing a single model's output from a deepfake detection system.

MODEL INFORMATION:
- Display Name: {display_info['display_name']}
- Analysis Method: {display_info['method_name']} ({display_info['method_description']})
- What It Analyzes: {model_type_info['description']}
- Typical Cues It Detects: {', '.join(model_type_info['typical_cues'])}

DETECTION RESULTS:
- Fake Probability: {prob_fake:.1%}
- Prediction: {"Likely AI-Generated" if prob_fake >= 0.5 else "Likely Real"}
- Focus Summary: {focus_summary or "Not available"}
{f"- Contribution to Final Decision: {contribution_percentage:.1f}%" if contribution_percentage else ""}

The heatmap shows where this model focused its attention. Brighter/warmer colors indicate higher attention.

TASK:
Analyze the image and heatmap to explain what this specific model detected. Provide:
1. A clear explanation of what the model focused on and why it might indicate AI generation (or authenticity)
2. 2-4 specific visual cues a human could verify, phrased as hypotheses with hedging language
3. A confidence assessment based on the probability and focus pattern

CRITICAL: Use hedging language - "may", "suggests", "possible", "could indicate". Never claim certainty.

Respond with valid JSON matching this exact structure:
{{
  "key_finding": "One sentence main finding about what the model detected",
  "what_model_saw": "2-3 sentences explaining what the model detected and why it matters",
  "important_regions": ["Region 1 with hedging language", "Region 2...", "Region 3..."],
  "confidence_qualifier": "Assessment of reliability with appropriate hedging"
}}

Respond with valid JSON only, no markdown formatting."""
            
            # Build content parts
            content_parts = []
            
            if original_image_b64:
                from google.genai import types
                content_parts.append(types.Part.from_bytes(
                    data=base64.b64decode(original_image_b64),
                    mime_type="image/png"
                ))
                content_parts.append(types.Part.from_text(text="Original image shown above.\n\n"))
            
            if heatmap_b64:
                from google.genai import types
                content_parts.append(types.Part.from_bytes(
                    data=base64.b64decode(heatmap_b64),
                    mime_type="image/png"
                ))
                content_parts.append(types.Part.from_text(text=f"{display_info['method_name']} heatmap shown above.\n\n"))
            
            from google.genai import types
            content_parts.append(types.Part.from_text(text=prompt))
            
            # Call the LLM with JSON response mode
            logger.info(f"Generating LLM explanation for {model_name}...")
            
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=content_parts,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    top_p=0.8,
                    max_output_tokens=2048,  # Increased to avoid truncation
                    response_mime_type="application/json",
                )
            )
            
            # Parse response - even with JSON mode, sometimes there are issues
            text = response.text.strip()
            
            try:
                result = json.loads(text)
            except json.JSONDecodeError as parse_err:
                # Log the problematic text for debugging
                logger.warning(f"Initial JSON parse failed: {parse_err}")
                logger.warning(f"Raw text (first 500 chars): {repr(text[:500])}")
                
                # Try to fix common issues: newlines inside strings
                # Replace literal newlines with escaped ones, but only inside quoted strings
                import re
                
                # More robust approach: find all string values and escape newlines
                def escape_newlines_in_strings(s):
                    result = []
                    in_string = False
                    escape_next = False
                    for i, c in enumerate(s):
                        if escape_next:
                            result.append(c)
                            escape_next = False
                            continue
                        if c == '\\':
                            escape_next = True
                            result.append(c)
                            continue
                        if c == '"' and not escape_next:
                            in_string = not in_string
                            result.append(c)
                            continue
                        if in_string and c == '\n':
                            result.append('\\n')
                        elif in_string and c == '\r':
                            result.append('\\r')
                        else:
                            result.append(c)
                    return ''.join(result)
                
                fixed_text = escape_newlines_in_strings(text)
                result = json.loads(fixed_text)
            
            # Add model metadata to result
            result["model_name"] = model_name
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse single model LLM response: {e}")
            return {
                "model_name": model_name,
                "key_finding": f"The {display_info['display_name']} detected potential signs of manipulation.",
                "what_model_saw": f"The model analyzed the image but detailed analysis could not be parsed. The fake probability was {prob_fake:.1%}.",
                "important_regions": ["Unable to identify specific regions."],
                "confidence_qualifier": "Analysis completed but detailed explanation unavailable due to parsing error."
            }
        except Exception as e:
            logger.error(f"Failed to generate single model explanation: {e}")
            return None


# Global singleton
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
