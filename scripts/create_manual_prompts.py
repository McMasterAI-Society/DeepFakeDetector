"""
Script to create the manual-gen-images/prompts directory structure
and populate it with AI image generation prompts.
"""

import os
from pathlib import Path

# Define prompts
PROMPTS = {
    "p001": "Generate a photorealistic candid photo of an adult waiting at a city bus stop on a rainy evening, wet pavement reflections, natural lighting, 35mm lens, slight motion blur.",
    "p002": "Generate a candid photo of an adult walking down a sidewalk in the early morning, wearing casual clothes, soft sunlight, realistic skin texture, shallow depth of field.",
    "p003": "Generate a photo of an adult sitting alone at a café table, looking at their phone, indoor lighting, realistic shadows, documentary style.",
    "p004": "Generate a natural photo of two adults talking while walking through a park, trees in background, overcast lighting, casual candid moment.",
    "p005": "Generate a photo of an adult standing at a crosswalk waiting for the light to change, urban street setting, realistic reflections and shadows.",
    "p006": "Generate a candid photo of an adult sitting on a park bench reading a book, afternoon sunlight filtering through trees, realistic colors.",
    "p007": "Generate a photo of an adult grocery shopping in a supermarket aisle, shelves full of products, fluorescent lighting, slight motion blur.",
    "p008": "Generate a photo of an adult cooking in a small kitchen, vegetables on the counter, warm indoor lighting, natural clutter.",
    "p009": "Generate a candid photo of an adult tying their shoes while sitting on apartment building steps, midday light, realistic textures.",
    "p010": "Generate a photo of an adult waiting in line at a coffee shop, menu board in background, indoor lighting, candid framing.",
    "p011": "Generate a natural photo of an adult standing at a train platform, blurred train tracks in background, cool lighting.",
    "p012": "Generate a photo of an adult walking a dog on a residential street, late afternoon sun, realistic shadows.",
    "p013": "Generate a candid photo of an adult sitting at a desk working on a laptop, papers and notebooks scattered, soft window light.",
    "p014": "Generate a photo of an adult standing near a window inside an apartment, daylight coming in, natural posture.",
    "p015": "Generate a photo of an adult walking through a parking lot carrying grocery bags, cloudy sky, realistic color tones.",
    "p016": "Generate a candid photo of an adult sitting on public transit, holding onto a pole, fluorescent lighting, realistic noise.",
    "p017": "Generate a photo of an adult standing in a hallway of an apartment building, neutral walls, overhead lighting.",
    "p018": "Generate a photo of an adult leaning against a wall checking their phone, urban setting, shallow depth of field.",
    "p019": "Generate a natural photo of an adult standing in a doorway of a small shop, indoor lighting spilling outside.",
    "p020": "Generate a photo of an adult waiting at a pedestrian crossing at night, streetlights, slight grain.",
    "p021": "Generate a photo of an adult carrying a backpack walking down a quiet street, early evening light.",
    "p022": "Generate a candid photo of an adult sitting at a bus shelter, cloudy weather, realistic reflections on glass.",
    "p023": "Generate a photo of an adult entering a convenience store, automatic doors opening, indoor lighting visible.",
    "p024": "Generate a photo of an adult sitting at a dining table eating a meal, warm indoor lighting, realistic clutter.",
    "p025": "Generate a candid photo of an adult standing in line at a checkout counter, store interior, natural posture.",
    "p026": "Generate a photo of a messy bedroom desk with a laptop, notebooks, cables, and a coffee mug, daylight from window.",
    "p027": "Generate a photo of a small kitchen sink filled with dishes, sponge on the side, warm indoor lighting.",
    "p028": "Generate a photo of a living room with a couch, coffee table, and scattered magazines, natural daylight.",
    "p029": "Generate a photo of a hallway inside an apartment building, neutral walls, overhead lighting, realistic perspective.",
    "p030": "Generate a photo of a bathroom mirror with water spots, sink visible, soft indoor lighting.",
    "p031": "Generate a photo of a dining table with plates, cups, and leftovers, evening indoor lighting.",
    "p032": "Generate a photo of a home office with a desk, monitor, keyboard, and sticky notes, daylight through blinds.",
    "p033": "Generate a photo of a refrigerator interior with food containers and bottles, interior fridge light.",
    "p034": "Generate a photo of a cluttered entryway with shoes, jackets, and a doormat, natural lighting.",
    "p035": "Generate a photo of a bookshelf filled with books and small objects, indoor lighting, shallow depth of field.",
    "p036": "Generate a photo of a laptop on a café table, coffee cup beside it, blurred background.",
    "p037": "Generate a photo of a kitchen counter with groceries laid out, fruits and vegetables, indoor lighting.",
    "p038": "Generate a photo of a television on a stand in a living room, soft ambient lighting.",
    "p039": "Generate a photo of a bed with wrinkled sheets and pillows, morning light.",
    "p040": "Generate a photo of a washing machine in a laundry room, overhead lighting.",
    "p041": "Generate a photo of a dining chair pushed slightly away from a table, natural indoor shadows.",
    "p042": "Generate a photo of a window with light curtains blowing slightly, daylight.",
    "p043": "Generate a photo of a coffee maker on a kitchen counter, mug underneath, indoor lighting.",
    "p044": "Generate a photo of a microwave on a kitchen counter, digital clock visible.",
    "p045": "Generate a photo of a hallway closet with coats hanging inside, soft indoor lighting.",
    "p046": "Generate a photo of a desk drawer open with office supplies inside.",
    "p047": "Generate a photo of a sink faucet dripping water, close-up, realistic reflections.",
    "p048": "Generate a photo of a dining table set for one person, natural indoor lighting.",
    "p049": "Generate a photo of a lamp turned on in a dim living room, warm glow.",
    "p050": "Generate a photo of a door slightly ajar with light coming from inside the room.",
    "p051": "Generate a photo of a quiet residential street with parked cars, overcast sky.",
    "p052": "Generate a photo of a city sidewalk with pedestrians in the distance, natural daylight.",
    "p053": "Generate a photo of a bicycle locked to a metal rack on a sidewalk, urban setting.",
    "p054": "Generate a photo of a small neighborhood park with benches and trees, afternoon light.",
    "p055": "Generate a photo of a street corner with traffic lights and crosswalk markings.",
    "p056": "Generate a photo of a building entrance with glass doors reflecting the street.",
    "p057": "Generate a photo of a parking lot with a few cars, cloudy sky.",
    "p058": "Generate a photo of a bus stop sign beside the road, urban background.",
    "p059": "Generate a photo of a street with wet pavement after rain, reflections visible.",
    "p060": "Generate a photo of an alleyway between buildings, muted lighting.",
    "p061": "Generate a photo of a storefront window with items displayed inside.",
    "p062": "Generate a photo of a residential building facade with balconies.",
    "p063": "Generate a photo of a sidewalk café seating area with empty chairs.",
    "p064": "Generate a photo of a pedestrian bridge with railings, overcast weather.",
    "p065": "Generate a photo of a city street at dusk, streetlights beginning to turn on.",
    "p066": "Generate a photo of a small public square with benches and trash cans.",
    "p067": "Generate a photo of a parking garage entrance, concrete structure.",
    "p068": "Generate a photo of a residential driveway with a parked car.",
    "p069": "Generate a photo of a street sign on a pole, clear background.",
    "p070": "Generate a photo of a bike lane painted on the road.",
    "p071": "Generate a photo of a sidewalk covered in fallen leaves, autumn light.",
    "p072": "Generate a photo of a crosswalk at an intersection, daytime lighting.",
    "p073": "Generate a photo of a bus pulling away from a stop, slight motion blur.",
    "p074": "Generate a photo of a city block with mixed residential and commercial buildings.",
    "p075": "Generate a photo of a quiet street at night with illuminated windows.",
    "p076": "Generate a close-up photo of a hand holding a smartphone, natural lighting, realistic skin texture.",
    "p077": "Generate a photo of a smartphone lying on a wooden table, soft indoor lighting.",
    "p078": "Generate a photo of a pair of shoes placed near a doorway.",
    "p079": "Generate a photo of a backpack resting against a wall indoors.",
    "p080": "Generate a photo of a reusable water bottle on a desk.",
    "p081": "Generate a photo of a coffee cup on a table, shallow depth of field.",
    "p082": "Generate a photo of a notebook open on a desk with handwritten notes.",
    "p083": "Generate a photo of a pen and keys on a kitchen counter.",
    "p084": "Generate a photo of a wallet placed next to a phone on a table.",
    "p085": "Generate a photo of a pair of headphones resting on a desk.",
    "p086": "Generate a photo of a pair of glasses folded on a book.",
    "p087": "Generate a photo of a watch lying on a bedside table.",
    "p088": "Generate a photo of a shopping bag on the floor near a chair.",
    "p089": "Generate a photo of a laptop charger plugged into a wall outlet.",
    "p090": "Generate a photo of a chair beside a window with daylight coming in.",
    "p091": "Generate a photo of a plate with leftover food on a table.",
    "p092": "Generate a photo of a set of keys hanging on a hook.",
    "p093": "Generate a photo of a notebook and coffee cup on a café table.",
    "p094": "Generate a photo of a folded jacket placed on a chair.",
    "p095": "Generate a photo of a trash bin near a sidewalk.",
    "p096": "Generate a photo of a grocery receipt on a kitchen counter.",
    "p097": "Generate a photo of a door handle in close-up, indoor lighting.",
    "p098": "Generate a photo of a light switch on a wall.",
    "p099": "Generate a photo of a window view showing nearby buildings.",
    "p100": "Generate a photo of a table with scattered papers and a pen, natural daylight.",
}


def main():
    # Get the project root (parent of scripts/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Define paths
    datasets_dir = project_root / "datasets"
    manual_gen_dir = datasets_dir / "manual-gen-images"
    prompts_dir = manual_gen_dir / "prompts"
    images_dir = manual_gen_dir / "images"
    
    # Create directories if they don't exist
    prompts_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure at: {manual_gen_dir}")
    
    # Create prompt files
    created_count = 0
    for prompt_id, prompt_text in PROMPTS.items():
        prompt_file = prompts_dir / f"{prompt_id}.txt"
        prompt_file.write_text(prompt_text, encoding="utf-8")
        created_count += 1
    
    print(f"Created {created_count} prompt files in: {prompts_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
