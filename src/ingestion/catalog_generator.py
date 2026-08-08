"""Deterministic, staged catalog generator (Part 7/9 of the RAG evolution
spec).

Rather than hand-typing hundreds of ad hoc products into one JSON blob,
products are generated from a small set of *families* — a base product
concept (e.g. "Noise Cancelling Headphones") expanded into 2-4 *variants*
that differ on concrete, meaningful attributes (price, battery, weight,
brand, rating, features). This is what makes the catalog useful for
retrieval evaluation: a query like "long battery life headphones" should
surface the right variant, not any headphone.

Determinism: given the same `--seed`, this script produces byte-identical
output every time (the only randomness — stock/review_count jitter and
which "filler" families are included to hit an exact --target — is drawn
from a seeded `random.Random`, never the global `random` module).

Usage:
    python -m src.ingestion.catalog_generator --target 100 --seed 42 \\
        --output data/raw/products_100.json

The original 39-product baseline (data/raw/sample_products.json) is always
included unchanged and keeps its original IDs — ground_truth.json's
expected_product_ids stay valid against any generated catalog stage.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from src.ingestion.schemas import validate_catalog

BASELINE_PATH = Path("data/raw/sample_products.json")
NEW_ID_START = 200


# ── Family definitions ─────────────────────────────────────
# Each family expands into len(variants) products. `desc` is a template
# filled with the variant's own fields plus the shared `unit` fields, so
# every variant reads as a distinct, realistic product rather than a
# find-and-replace clone.
FAMILIES: list[dict[str, Any]] = [
    # ── Electronics ─────────────────────────────────────
    {
        "category": "Electronics",
        "name": "Noise Cancelling Headphones",
        "desc": "{brand} over-ear headphones with {anc} active noise cancellation, "
                "{battery}h battery life, and {driver}mm dynamic drivers. {extra}",
        "variants": [
            {"suffix": "Pro", "brand": "Aurelia", "price": 199.99, "anc": "adaptive", "battery": 40, "driver": 40,
             "weight_g": 250, "extra": "Multipoint Bluetooth 5.3 pairing with two devices at once.",
             "rating": 4.6, "features": ["adaptive ANC", "multipoint pairing", "40h battery"]},
            {"suffix": "Lite", "brand": "Aurelia", "price": 89.99, "anc": "hybrid", "battery": 30, "driver": 35,
             "weight_g": 220, "extra": "Foldable design with a hard travel case included.",
             "rating": 4.2, "features": ["hybrid ANC", "foldable", "travel case"]},
            {"suffix": "Sport", "brand": "Kinetic", "price": 129.99, "anc": "hybrid", "battery": 22, "driver": 38,
             "weight_g": 210, "extra": "IPX4 sweat resistance rated for workouts.",
             "rating": 4.3, "features": ["IPX4", "workout fit", "hybrid ANC"]},
        ],
    },
    {
        "category": "Electronics",
        "name": "True Wireless Earbuds",
        "desc": "{brand} true wireless earbuds with {battery}h battery ({case_battery}h with case), "
                "{anc} noise cancellation, and touch controls. {extra}",
        "variants": [
            {"suffix": "Air", "brand": "Kinetic", "price": 79.99, "anc": "passive", "battery": 6, "case_battery": 24,
             "weight_g": 4, "extra": "Wireless charging case, IPX4 rated.",
             "rating": 4.1, "features": ["wireless charging", "IPX4", "touch controls"]},
            {"suffix": "Max", "brand": "Aurelia", "price": 149.99, "anc": "active", "battery": 8, "case_battery": 32,
             "weight_g": 5, "extra": "Custom EQ via companion app and transparency mode.",
             "rating": 4.5, "features": ["active ANC", "transparency mode", "app EQ"]},
        ],
    },
    {
        "category": "Electronics",
        "name": "Bluetooth Speaker",
        "desc": "{brand} portable Bluetooth speaker rated {waterproof}, {battery}h playtime, "
                "{watts}W output. {extra}",
        "variants": [
            {"suffix": "Mini", "brand": "Wavefront", "price": 39.99, "waterproof": "IPX5", "battery": 10, "watts": 10,
             "extra": "Pocket-sized with a carabiner clip.",
             "rating": 4.0, "features": ["IPX5", "carabiner clip", "compact"]},
            {"suffix": "Boom", "brand": "Wavefront", "price": 99.99, "waterproof": "IP67", "battery": 20, "watts": 30,
             "extra": "Pair two units together for stereo sound.",
             "rating": 4.4, "features": ["IP67", "stereo pairing", "30W output"]},
        ],
    },
    {
        "category": "Electronics",
        "name": "Smart Watch",
        "desc": "{brand} smart watch with {display} display, {battery}-day battery life, "
                "GPS, and heart-rate + SpO2 monitoring. {extra}",
        "variants": [
            {"suffix": "SE", "brand": "Pulseline", "price": 149.99, "display": "1.3\" AMOLED", "battery": 5,
             "weight_g": 32, "extra": "50m water resistance, 20+ workout modes.",
             "rating": 4.2, "features": ["AMOLED", "GPS", "50m water resistance"]},
            {"suffix": "Ultra", "brand": "Pulseline", "price": 249.99, "display": "1.5\" AMOLED", "battery": 9,
             "weight_g": 38, "extra": "Titanium case with sapphire crystal glass.",
             "rating": 4.6, "features": ["titanium case", "sapphire glass", "9-day battery"]},
        ],
    },
    {
        "category": "Electronics",
        "name": "Portable SSD",
        "desc": "{brand} {capacity}TB portable SSD, USB-C {speed}Gbps transfer speed, "
                "shock-resistant aluminium housing. {extra}",
        "variants": [
            {"suffix": "1TB", "brand": "Databrick", "price": 89.99, "capacity": 1, "speed": 10,
             "extra": "IP55 dust and water resistance.",
             "rating": 4.5, "features": ["IP55", "10Gbps", "aluminium housing"]},
            {"suffix": "2TB", "brand": "Databrick", "price": 149.99, "capacity": 2, "speed": 20,
             "extra": "USB4 compatible for next-gen laptop transfer speeds.",
             "rating": 4.6, "features": ["IP55", "20Gbps", "USB4 compatible"]},
        ],
    },
    {
        "category": "Electronics",
        "name": "Wireless Gaming Mouse",
        "desc": "{brand} wireless gaming mouse with {dpi} DPI optical sensor, {battery}h battery, "
                "and low-latency 2.4GHz receiver. {extra}",
        "variants": [
            {"suffix": "", "brand": "Kinetic", "price": 59.99, "dpi": "26,000", "battery": 70,
             "extra": "Six programmable buttons with onboard memory.",
             "rating": 4.4, "features": ["26000 DPI", "programmable buttons", "70h battery"]},
        ],
    },
    {
        "category": "Electronics",
        "name": "27-Inch 4K Monitor",
        "desc": "{brand} 27-inch 4K IPS monitor, {hz}Hz refresh rate, 98% DCI-P3 color gamut, "
                "USB-C 65W power delivery. {extra}",
        "variants": [
            {"suffix": "", "brand": "Visiora", "price": 379.99, "hz": 60,
             "extra": "Height-adjustable stand with 90-degree pivot.",
             "rating": 4.5, "features": ["4K IPS", "USB-C PD", "pivot stand"]},
        ],
    },
    {
        "category": "Electronics",
        "name": "USB Condenser Microphone",
        "desc": "{brand} USB condenser microphone with cardioid pickup pattern, {sample_rate}kHz sample rate, "
                "built-in pop filter and shock mount. {extra}",
        "variants": [
            {"suffix": "", "brand": "Wavefront", "price": 69.99, "sample_rate": 48,
             "extra": "Zero-latency headphone monitoring jack.",
             "rating": 4.3, "features": ["cardioid", "48kHz", "shock mount"]},
        ],
    },
    # ── Outdoor Gear ─────────────────────────────────────
    {
        "category": "Outdoor Gear",
        "name": "Hiking Backpack",
        "desc": "{brand} {capacity}L hiking backpack with {waterproof}, adjustable torso fit, "
                "and hydration bladder compartment. {extra}",
        "variants": [
            {"suffix": "35L", "brand": "Summit Ridge", "price": 79.99, "capacity": 35, "waterproof": "water-resistant coating",
             "weight_g": 1100, "extra": "Ideal for day hikes and overnight trips.",
             "rating": 4.3, "features": ["35L", "hydration compatible", "water-resistant"]},
            {"suffix": "55L", "brand": "Summit Ridge", "price": 139.99, "capacity": 55, "waterproof": "fully waterproof rainfly included",
             "weight_g": 1800, "extra": "Internal frame for multi-day backcountry trips.",
             "rating": 4.5, "features": ["55L", "internal frame", "rainfly included"]},
        ],
    },
    {
        "category": "Outdoor Gear",
        "name": "4-Person Camping Tent",
        "desc": "{brand} 4-person dome tent, {season}-season rated, aluminium pole frame, "
                "double-layer rainfly. {extra}",
        "variants": [
            {"suffix": "", "brand": "Trailhead", "price": 189.99, "season": 3,
             "weight_g": 4200, "extra": "Two doors, two vestibules for gear storage.",
             "rating": 4.4, "features": ["4-person", "3-season", "two vestibules"]},
        ],
    },
    {
        "category": "Outdoor Gear",
        "name": "Portable Camping Stove",
        "desc": "{brand} compact butane camping stove with piezo ignition, {output}kW output, "
                "wind-resistant burner head. {extra}",
        "variants": [
            {"suffix": "", "brand": "Trailhead", "price": 34.99, "output": 2.7,
             "extra": "Folds flat into included carry case.",
             "rating": 4.1, "features": ["piezo ignition", "wind-resistant", "compact"]},
        ],
    },
    {
        "category": "Outdoor Gear",
        "name": "Camping Hammock with Straps",
        "desc": "{brand} parachute nylon camping hammock rated to {capacity}kg, includes tree-friendly "
                "straps and carabiners. {extra}",
        "variants": [
            {"suffix": "", "brand": "Trailhead", "price": 44.99, "capacity": 200,
             "extra": "Packs down to the size of a water bottle.",
             "rating": 4.2, "features": ["parachute nylon", "tree straps included", "packable"]},
        ],
    },
    # ── Footwear ─────────────────────────────────────────
    {
        "category": "Footwear",
        "name": "Road Running Shoes",
        "desc": "{brand} road running shoes with {cushion} cushioning, breathable mesh upper, "
                "and {drop}mm heel-to-toe drop. {extra}",
        "variants": [
            {"suffix": "Cloud", "brand": "Stridewell", "price": 119.99, "cushion": "max", "drop": 8,
             "weight_g": 280, "extra": "Designed for long-distance comfort.",
             "rating": 4.5, "features": ["max cushioning", "breathable mesh", "8mm drop"]},
            {"suffix": "Race", "brand": "Stridewell", "price": 149.99, "cushion": "responsive", "drop": 4,
             "weight_g": 220, "extra": "Carbon-infused plate for a propulsive toe-off.",
             "rating": 4.6, "features": ["carbon plate", "responsive cushioning", "lightweight"]},
        ],
    },
    {
        "category": "Footwear",
        "name": "Hiking Sandals",
        "desc": "{brand} hiking sandals with contoured EVA footbed, adjustable quick-lace straps, "
                "and grippy rubber outsole. {extra}",
        "variants": [
            {"suffix": "", "brand": "Summit Ridge", "price": 54.99,
             "extra": "Quick-drying straps suited to river crossings.",
             "rating": 4.2, "features": ["quick-dry", "EVA footbed", "grippy outsole"]},
        ],
    },
    {
        "category": "Footwear",
        "name": "Insulated Winter Snow Boots",
        "desc": "{brand} insulated snow boots rated to {temp}, waterproof upper, "
                "and thermal-reflective lining. {extra}",
        "variants": [
            {"suffix": "", "brand": "Summit Ridge", "price": 99.99, "temp": "-32C",
             "extra": "High-traction lugged outsole for icy surfaces.",
             "rating": 4.4, "features": ["waterproof", "thermal lining", "-32C rated"]},
        ],
    },
    # ── Sports & Fitness ─────────────────────────────────
    {
        "category": "Sports & Fitness",
        "name": "Cast Iron Kettlebell Set",
        "desc": "{brand} cast iron kettlebell set ({weights}), wide handle for two-hand grip, "
                "flat base for stable storage. {extra}",
        "variants": [
            {"suffix": "", "brand": "Ironcore", "price": 89.99, "weights": "5kg/10kg/15kg",
             "extra": "Powder-coated finish resists chipping.",
             "rating": 4.5, "features": ["cast iron", "3-piece set", "powder coated"]},
        ],
    },
    {
        "category": "Sports & Fitness",
        "name": "Doorway Pull-Up Bar",
        "desc": "{brand} no-screw doorway pull-up bar, supports up to {capacity}kg, "
                "multi-grip design for wide/narrow/neutral grip. {extra}",
        "variants": [
            {"suffix": "", "brand": "Ironcore", "price": 29.99, "capacity": 135,
             "extra": "Fits doorframes 65-80cm wide, no drilling required.",
             "rating": 4.1, "features": ["no-drill", "multi-grip", "135kg capacity"]},
        ],
    },
    {
        "category": "Sports & Fitness",
        "name": "Folding Exercise Bike",
        "desc": "{brand} magnetic resistance exercise bike with {levels} resistance levels, "
                "foldable frame, and LCD tracking display. {extra}",
        "variants": [
            {"suffix": "", "brand": "Ironcore", "price": 219.99, "levels": 8,
             "extra": "Tracks time, speed, distance, and calories burned.",
             "rating": 4.0, "features": ["foldable", "8 resistance levels", "LCD display"]},
        ],
    },
    # ── Clothing ─────────────────────────────────────────
    {
        "category": "Clothing",
        "name": "Packable Rain Jacket",
        "desc": "{brand} packable rain jacket, {waterproof} rating, taped seams, "
                "packs into its own chest pocket. {extra}",
        "variants": [
            {"suffix": "Men's", "brand": "Northfield", "price": 79.99, "waterproof": "10,000mm",
             "extra": "Adjustable hood and elasticated cuffs.",
             "rating": 4.2, "features": ["packable", "taped seams", "10000mm rating"]},
            {"suffix": "Women's", "brand": "Northfield", "price": 79.99, "waterproof": "10,000mm",
             "extra": "Tailored fit with adjustable waist drawcord.",
             "rating": 4.3, "features": ["packable", "tailored fit", "10000mm rating"]},
        ],
    },
    {
        "category": "Clothing",
        "name": "Running Shorts with Liner",
        "desc": "{brand} lightweight running shorts with built-in brief liner, {inseam}cm inseam, "
                "reflective trim for low-light visibility. {extra}",
        "variants": [
            {"suffix": "", "brand": "Stridewell", "price": 34.99, "inseam": 13,
             "extra": "Side zip pocket fits a phone securely.",
             "rating": 4.2, "features": ["reflective trim", "zip pocket", "built-in liner"]},
        ],
    },
    {
        "category": "Clothing",
        "name": "Touchscreen Thermal Gloves",
        "desc": "{brand} thermal fleece-lined gloves with touchscreen-compatible fingertips, "
                "windproof shell, rated to {temp}. {extra}",
        "variants": [
            {"suffix": "", "brand": "Northfield", "price": 24.99, "temp": "-15C",
             "extra": "Adjustable wrist strap for a secure fit.",
             "rating": 4.0, "features": ["touchscreen compatible", "windproof", "-15C rated"]},
        ],
    },
    # ── Home Appliances ───────────────────────────────────
    {
        "category": "Home Appliances",
        "name": "Drip Coffee Maker",
        "desc": "{brand} {cups}-cup programmable drip coffee maker with reusable filter, "
                "auto shut-off, and keep-warm plate. {extra}",
        "variants": [
            {"suffix": "", "brand": "Homehearth", "price": 49.99, "cups": 12,
             "extra": "24-hour programmable brew delay.",
             "rating": 4.1, "features": ["programmable", "reusable filter", "keep-warm plate"]},
        ],
    },
    {
        "category": "Home Appliances",
        "name": "Digital Air Fryer",
        "desc": "{brand} {capacity}L digital air fryer, {watts}W, 8 preset cooking modes, "
                "dishwasher-safe non-stick basket. {extra}",
        "variants": [
            {"suffix": "", "brand": "Homehearth", "price": 89.99, "capacity": 5.5, "watts": 1700,
             "extra": "Rapid air circulation cuts oil use by up to 85%.",
             "rating": 4.5, "features": ["5.5L capacity", "8 presets", "dishwasher-safe basket"]},
        ],
    },
    {
        "category": "Home Appliances",
        "name": "Cool Mist Humidifier",
        "desc": "{brand} {capacity}L ultrasonic cool mist humidifier, covers up to {coverage} sq ft, "
                "whisper-quiet operation. {extra}",
        "variants": [
            {"suffix": "", "brand": "Homehearth", "price": 44.99, "capacity": 4, "coverage": 400,
             "extra": "Built-in night light with adjustable brightness.",
             "rating": 4.2, "features": ["ultrasonic", "night light", "400 sq ft coverage"]},
        ],
    },
    # ── Home Office ────────────────────────────────────────
    {
        "category": "Home Office",
        "name": "Ergonomic Mesh Office Chair",
        "desc": "{brand} ergonomic mesh office chair with adjustable lumbar support, "
                "4D armrests, and {recline} recline. {extra}",
        "variants": [
            {"suffix": "", "brand": "Deskform", "price": 229.99, "recline": "135-degree",
             "extra": "Breathable mesh back reduces heat buildup during long sits.",
             "rating": 4.4, "features": ["4D armrests", "lumbar support", "135-degree recline"]},
        ],
    },
    {
        "category": "Home Office",
        "name": "Gas-Spring Monitor Arm",
        "desc": "{brand} single-monitor gas-spring arm, fits {size}-inch displays, "
                "full articulation (tilt/swivel/rotate). {extra}",
        "variants": [
            {"suffix": "", "brand": "Deskform", "price": 44.99, "size": "13-32",
             "extra": "Clamp and grommet desk mounts both included.",
             "rating": 4.3, "features": ["gas-spring", "full articulation", "13-32in displays"]},
        ],
    },
    {
        "category": "Home Office",
        "name": "Bamboo Desk Organizer",
        "desc": "{brand} bamboo desk organizer with {compartments} compartments, "
                "phone stand slot, and pen/pencil tray. {extra}",
        "variants": [
            {"suffix": "", "brand": "Deskform", "price": 27.99, "compartments": 6,
             "extra": "Sustainably sourced bamboo with a natural finish.",
             "rating": 4.1, "features": ["bamboo", "6 compartments", "phone stand"]},
        ],
    },
    {
        "category": "Home Office",
        "name": "Adjustable LED Desk Lamp",
        "desc": "{brand} adjustable LED desk lamp with {modes} lighting modes, "
                "USB charging port, and touch dimmer. {extra}",
        "variants": [
            {"suffix": "", "brand": "Deskform", "price": 32.99, "modes": 5,
             "extra": "Flicker-free, eye-care certified light.",
             "rating": 4.3, "features": ["5 lighting modes", "USB port", "touch dimmer"]},
        ],
    },
    # ── Nutrition ────────────────────────────────────────
    {
        "category": "Nutrition",
        "name": "Daily Multivitamin",
        "desc": "{brand} daily multivitamin, {count} tablets, {vitamins} essential vitamins and minerals. {extra}",
        "variants": [
            {"suffix": "", "brand": "Vitalcore", "price": 14.99, "count": 90, "vitamins": 23,
             "extra": "Non-GMO, gluten-free, one tablet per day.",
             "rating": 4.2, "features": ["non-GMO", "gluten-free", "90 tablets"]},
        ],
    },
    {
        "category": "Nutrition",
        "name": "Protein Bars Box",
        "desc": "{brand} protein bars, {protein}g protein per bar, box of {count}, {flavor} flavor. {extra}",
        "variants": [
            {"suffix": "", "brand": "Vitalcore", "price": 24.99, "protein": 20, "count": 12, "flavor": "peanut butter",
             "extra": "Low sugar, no artificial sweeteners.",
             "rating": 4.3, "features": ["20g protein", "low sugar", "12-pack"]},
        ],
    },
    {
        "category": "Nutrition",
        "name": "Pre-Workout Powder",
        "desc": "{brand} pre-workout powder, {caffeine}mg caffeine per serving, {servings} servings, "
                "{flavor} flavor. {extra}",
        "variants": [
            {"suffix": "", "brand": "Vitalcore", "price": 29.99, "caffeine": 200, "servings": 30, "flavor": "blue raspberry",
             "extra": "Includes beta-alanine and L-citrulline for endurance.",
             "rating": 4.1, "features": ["200mg caffeine", "beta-alanine", "30 servings"]},
        ],
    },
    # ── Travel ───────────────────────────────────────────
    {
        "category": "Travel",
        "name": "Hardside Carry-On Luggage",
        "desc": "{brand} {liters}L hardside carry-on with 360-degree spinner wheels, "
                "TSA-approved combination lock, and {material} shell. {extra}",
        "variants": [
            {"suffix": "Standard", "brand": "Farbound", "price": 99.99, "liters": 38, "material": "polycarbonate",
             "extra": "Meets most airline carry-on size limits.",
             "rating": 4.3, "features": ["spinner wheels", "TSA lock", "polycarbonate shell"]},
            {"suffix": "Expandable", "brand": "Farbound", "price": 129.99, "liters": 42, "material": "polycarbonate",
             "extra": "Zip expansion adds an extra 15% packing space.",
             "rating": 4.4, "features": ["expandable", "spinner wheels", "TSA lock"]},
        ],
    },
    {
        "category": "Travel",
        "name": "Compression Packing Cubes Set",
        "desc": "{brand} packing cubes set ({count} pieces), water-resistant ripstop fabric, "
                "double zip compression. {extra}",
        "variants": [
            {"suffix": "", "brand": "Farbound", "price": 34.99, "count": 6,
             "extra": "Mesh top panel for at-a-glance packing.",
             "rating": 4.4, "features": ["6-piece set", "compression zip", "ripstop fabric"]},
        ],
    },
    {
        "category": "Travel",
        "name": "Memory Foam Travel Pillow",
        "desc": "{brand} memory foam neck travel pillow with {closure} closure, "
                "machine-washable velour cover. {extra}",
        "variants": [
            {"suffix": "", "brand": "Farbound", "price": 22.99, "closure": "magnetic",
             "extra": "Compresses into an included drawstring pouch.",
             "rating": 4.0, "features": ["memory foam", "magnetic closure", "washable cover"]},
        ],
    },
    {
        "category": "Travel",
        "name": "Universal Travel Adapter",
        "desc": "{brand} universal travel adapter for {countries}+ countries, {usb_ports} USB-A "
                "and 1 USB-C port, built-in fuse protection. {extra}",
        "variants": [
            {"suffix": "", "brand": "Farbound", "price": 24.99, "countries": 150, "usb_ports": 2,
             "extra": "Compact all-in-one design, no interchangeable parts to lose.",
             "rating": 4.3, "features": ["150+ countries", "USB-C port", "fuse protection"]},
        ],
    },
    {
        "category": "Travel",
        "name": "Anti-Theft Neck Wallet",
        "desc": "{brand} RFID-blocking neck wallet with {compartments} compartments, "
                "moisture-wicking strap. {extra}",
        "variants": [
            {"suffix": "", "brand": "Farbound", "price": 16.99, "compartments": 3,
             "extra": "Slim enough to wear discreetly under clothing.",
             "rating": 4.0, "features": ["RFID-blocking", "3 compartments", "moisture-wicking"]},
        ],
    },
    # ── Kitchen ──────────────────────────────────────────
    {
        "category": "Kitchen",
        "name": "Chef Knife Set",
        "desc": "{brand} {count}-piece chef knife set, high-carbon stainless steel blades, "
                "ergonomic handles, includes wooden block. {extra}",
        "variants": [
            {"suffix": "", "brand": "Hearthstone", "price": 79.99, "count": 6,
             "extra": "Full-tang construction for balance and durability.",
             "rating": 4.5, "features": ["high-carbon steel", "6-piece set", "wooden block"]},
        ],
    },
    {
        "category": "Kitchen",
        "name": "Non-Stick Cookware Set",
        "desc": "{brand} {count}-piece non-stick cookware set, PFOA-free coating, "
                "tempered glass lids, oven-safe to {temp}. {extra}",
        "variants": [
            {"suffix": "", "brand": "Hearthstone", "price": 119.99, "count": 10, "temp": "260C",
             "extra": "Compatible with all stovetops including induction.",
             "rating": 4.3, "features": ["PFOA-free", "10-piece set", "induction compatible"]},
        ],
    },
    {
        "category": "Kitchen",
        "name": "Bamboo Cutting Board Set",
        "desc": "{brand} {count}-piece bamboo cutting board set with juice grooves, "
                "non-slip feet, various sizes. {extra}",
        "variants": [
            {"suffix": "", "brand": "Hearthstone", "price": 29.99, "count": 3,
             "extra": "Naturally antimicrobial bamboo surface.",
             "rating": 4.2, "features": ["bamboo", "3-piece set", "non-slip feet"]},
        ],
    },
    {
        "category": "Kitchen",
        "name": "Glass Food Storage Containers",
        "desc": "{brand} {count}-piece glass food storage set with airtight snap lids, "
                "microwave/freezer/dishwasher safe. {extra}",
        "variants": [
            {"suffix": "", "brand": "Hearthstone", "price": 39.99, "count": 10,
             "extra": "BPA-free lids with a silicone seal.",
             "rating": 4.4, "features": ["airtight", "10-piece set", "BPA-free"]},
        ],
    },
    {
        "category": "Kitchen",
        "name": "Digital Kitchen Scale",
        "desc": "{brand} digital kitchen scale, {capacity}kg capacity, 1g precision, "
                "tare function, tempered glass platform. {extra}",
        "variants": [
            {"suffix": "", "brand": "Hearthstone", "price": 17.99, "capacity": 5,
             "extra": "Auto-off after 2 minutes of inactivity to save battery.",
             "rating": 4.1, "features": ["1g precision", "tare function", "tempered glass"]},
        ],
    },
    # ── Personal Care ─────────────────────────────────────
    {
        "category": "Personal Care",
        "name": "Sonic Electric Toothbrush",
        "desc": "{brand} sonic electric toothbrush, {modes} cleaning modes, "
                "{battery}-day battery life, includes {heads} brush heads. {extra}",
        "variants": [
            {"suffix": "", "brand": "Purelume", "price": 44.99, "modes": 5, "battery": 30, "heads": 4,
             "extra": "Built-in 2-minute smart timer with quadrant alerts.",
             "rating": 4.4, "features": ["5 modes", "30-day battery", "smart timer"]},
        ],
    },
    {
        "category": "Personal Care",
        "name": "Ionic Hair Dryer",
        "desc": "{brand} ionic hair dryer, {watts}W motor, {settings} heat/speed settings, "
                "cool-shot button. {extra}",
        "variants": [
            {"suffix": "", "brand": "Purelume", "price": 54.99, "watts": 1875, "settings": 6,
             "extra": "Ionic technology reduces frizz and drying time.",
             "rating": 4.2, "features": ["ionic", "1875W", "6 settings"]},
        ],
    },
    {
        "category": "Personal Care",
        "name": "Facial Cleansing Brush",
        "desc": "{brand} silicone facial cleansing brush, {speeds} vibration speeds, "
                "waterproof for shower use. {extra}",
        "variants": [
            {"suffix": "", "brand": "Purelume", "price": 29.99, "speeds": 3,
             "extra": "USB rechargeable with 90-minute battery life.",
             "rating": 4.1, "features": ["silicone", "waterproof", "USB rechargeable"]},
        ],
    },
    {
        "category": "Personal Care",
        "name": "Precision Beard Trimmer",
        "desc": "{brand} precision beard trimmer, {settings} length settings, "
                "self-sharpening steel blades, {battery}min runtime. {extra}",
        "variants": [
            {"suffix": "", "brand": "Purelume", "price": 34.99, "settings": 20, "battery": 90,
             "extra": "Fully washable head for easy cleaning.",
             "rating": 4.3, "features": ["20 length settings", "self-sharpening blades", "washable head"]},
        ],
    },
    # ── Gaming ───────────────────────────────────────────
    {
        "category": "Gaming",
        "name": "Wireless Gaming Headset",
        "desc": "{brand} wireless gaming headset with {surround} surround sound, "
                "{battery}h battery, detachable noise-cancelling mic. {extra}",
        "variants": [
            {"suffix": "Core", "brand": "Kinetic", "price": 69.99, "surround": "stereo", "battery": 20,
             "extra": "Memory foam ear cushions for extended sessions.",
             "rating": 4.2, "features": ["stereo", "20h battery", "detachable mic"]},
            {"suffix": "Elite", "brand": "Kinetic", "price": 129.99, "surround": "7.1", "battery": 30,
             "extra": "Low-latency 2.4GHz wireless with USB-C fast charging.",
             "rating": 4.5, "features": ["7.1 surround", "30h battery", "USB-C fast charge"]},
        ],
    },
    {
        "category": "Gaming",
        "name": "Ergonomic Gaming Chair",
        "desc": "{brand} ergonomic gaming chair with 4D armrests, {recline} recline, "
                "and removable lumbar + neck pillows. {extra}",
        "variants": [
            {"suffix": "", "brand": "Deskform", "price": 249.99, "recline": "165-degree",
             "extra": "PU leather upholstery rated for up to 150kg.",
             "rating": 4.3, "features": ["165-degree recline", "4D armrests", "150kg rated"]},
        ],
    },
    {
        "category": "Gaming",
        "name": "Wireless Game Controller",
        "desc": "{brand} wireless game controller with {battery}h battery, "
                "programmable back paddles, and hall-effect analog sticks. {extra}",
        "variants": [
            {"suffix": "Standard", "brand": "Kinetic", "price": 54.99, "battery": 25,
             "extra": "Compatible with PC, console, and mobile via Bluetooth.",
             "rating": 4.4, "features": ["hall-effect sticks", "back paddles", "25h battery"]},
            {"suffix": "Pro", "brand": "Kinetic", "price": 79.99, "battery": 35,
             "extra": "Interchangeable magnetic thumbsticks and adjustable trigger stops.",
             "rating": 4.6, "features": ["magnetic thumbsticks", "adjustable triggers", "35h battery"]},
        ],
    },
    {
        "category": "Gaming",
        "name": "RGB Gaming Mouse Pad",
        "desc": "{brand} extended RGB gaming mouse pad, {size}, stitched anti-fray edges, "
                "non-slip rubber base. {extra}",
        "variants": [
            {"suffix": "", "brand": "Kinetic", "price": 24.99, "size": "900x400mm",
             "extra": "16.8M color RGB lighting synced via companion app.",
             "rating": 4.0, "features": ["RGB lighting", "900x400mm", "anti-fray edges"]},
        ],
    },
]


def _fill_description(template: str, variant: dict[str, Any]) -> str:
    return template.format(**variant)


def _build_title(family_name: str, suffix: str) -> str:
    return f"{family_name} {suffix}".strip()


def _stock_and_reviews(rng: random.Random) -> tuple[int, int]:
    stock = rng.choice([0, 5, 12, 18, 25, 40, 60, 100, 150])
    review_count = rng.randint(3, 2400)
    return stock, review_count


def generate_new_products(seed: int = 42) -> list[dict[str, Any]]:
    """Deterministically expands FAMILIES into flat product dicts. Same
    seed -> same output, every time (no dependence on dict/set iteration
    order beyond what's fixed by FAMILIES' own list order)."""
    rng = random.Random(seed)
    products: list[dict[str, Any]] = []
    next_id = NEW_ID_START

    for family in FAMILIES:
        for variant in family["variants"]:
            stock, review_count = _stock_and_reviews(rng)
            description = _fill_description(family["desc"], variant)
            title = _build_title(family["name"], variant["suffix"])
            product = {
                "id": str(next_id),
                "title": title,
                "description": description,
                "price": variant["price"],
                "category": family["category"],
                "brand": variant["brand"],
                "rating": variant["rating"],
                "review_count": review_count,
                "stock": stock,
                "features": variant["features"],
                "tags": [family["category"].lower().replace(" & ", "_").replace(" ", "_")] + [
                    f.replace(" ", "_") for f in variant["features"][:2]
                ],
                "availability": "in_stock" if stock > 0 else "out_of_stock",
                "metadata": {"family": family["name"]},
            }
            products.append(product)
            next_id += 1

    return products


def generate_catalog(target: int, seed: int = 42) -> list[dict[str, Any]]:
    """baseline (39, unchanged IDs) + as many generated family variants as
    needed to reach `target`, in deterministic family order. Raises if
    `target` exceeds baseline + all available generated variants, rather
    than silently duplicating products to pad the count."""
    baseline = json.loads(BASELINE_PATH.read_text())
    generated = generate_new_products(seed=seed)

    max_possible = len(baseline) + len(generated)
    if target > max_possible:
        raise ValueError(
            f"Requested target={target} exceeds available baseline+generated "
            f"products ({max_possible}). Add more families to FAMILIES."
        )
    if target < len(baseline):
        raise ValueError(f"target={target} is smaller than the baseline catalog ({len(baseline)}).")

    needed_new = target - len(baseline)
    return baseline + generated[:needed_new]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic catalog stage.")
    parser.add_argument("--target", type=int, required=True, help="Total product count (e.g. 100, 250, 500, 1000).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    catalog = generate_catalog(target=args.target, seed=args.seed)

    valid_records, report = validate_catalog(catalog, source_file=str(args.output))
    if not report.is_ingestible():
        raise SystemExit(
            f"Generated catalog failed validation: "
            f"{report.invalid_records} invalid, {len(report.duplicate_ids)} duplicate IDs."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, indent=2))
    print(f"Wrote {len(catalog)} products to {args.output}")
    print(f"Categories: {report.categories}")
    print(f"Unknown categories: {report.unknown_categories}")


if __name__ == "__main__":
    main()
