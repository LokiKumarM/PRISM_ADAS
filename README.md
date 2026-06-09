# PRISM-ADAS

> A Cognitive Reasoning Layer for Advanced Driver-Assistance Systems — one that takes perception outputs and *explains* the driving decisions it makes.

**[▶ Try the live demo on Hugging Face Spaces](https://huggingface.co/spaces/Lokikumar/PRISM_ADAS)**

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/Lokikumar/PRISM_ADAS)

---

## Objective

This project sets out to build an **AI model that serves as a reasoning engine for ADAS** — a model that takes perception outputs (the objects around a vehicle) and reasons over them to produce driving decisions together with a transparent, human-understandable explanation of *why*.

The central thesis is that the next step for ADAS — especially for deployment on Indian roads — is **not merely better perception**, but a learned reasoning layer that can:
- deliberate over a complex, unstructured scene,
- weigh competing considerations,
- anticipate intent, and
- justify its conclusions.

To validate this thesis quickly and concretely, we first built a **rule-based reasoning prototype**. This prototype is *not* the end goal; it is a scaffold. It:

1. Demonstrates the value of explainable reasoning for ADAS functions.
2. Defines the **interfaces and output format** a reasoning engine should produce.
3. Generates the **structured reasoning traces** used to train the AI model that is the project's true objective.

---

## What's in this repo today

A self-contained, deterministic, fully-offline rule-based reasoning core built around the **nuScenes Mini** dataset (JSON annotations only — no sensor files, no devkit, no network access).

Every frame produces a **Reasoned Alert from the PRISM layer**:

```
CONTEXT   What is happening in Road
          Vulnerable road user crossing the protected path

RISK      Level of Risk
          HIGH

ACTION    Action taken by ADAS
          Auto-Hold engaged

REASON    Reasoning behind the Action
          A pedestrian is 4.2 m from you, inside our 8.0 m stop zone and
          entering your path — we're holding position until they're clear.
```

The same structured output drives both:
- a **Streamlit demo** with a bird's-eye-view (BEV) plot of the scene next to the Reasoned Alert panel; and
- the **training data** for the downstream ML reasoning model.

---

## Why a reasoning layer?

Today's production ADAS functions (AEB, ACC, BSW, LKA) are *threshold trips* over individual signals: "object inside this distance → fire this actuator." They cannot:

- categorise the **driving situation** in human-meaningful terms,
- explain **why one action beats another** in this frame,
- carry forward the **intent** behind a manoeuvre, or
- produce a **trace** that a driver trusts or an engineer can debug.

The Reasoned Alert format above directly addresses each of these gaps. CONTEXT names the semantic situation (a "vulnerable road user safety event", a "lane-change opportunity blocked by adjacent traffic", "sustained car-following"). REASON is a plain-language driver-facing sentence that quotes the actual thresholds the rule used. ACTION names the active vehicle function (AEB, Auto-Hold, ACC, LCI) rather than a recommendation phrased *to* the driver. RISK is a deterministic hybrid score combining rule priority and trigger urgency.

This is the **interface contract** the learned model will be trained to match.

---

## Architecture

```
nuScenes Mini JSON (13 tables)
        │   src/store.py          custom token-keyed loader (no devkit)
        ▼
NuMiniStore (in-memory relational store)
        │   src/object_list.py    global → ego-frame transform, taxonomy mapping
        ▼
ObjectList[DetectedObject]        per-frame perception output (ego frame)
        │   src/velocity.py       cross-frame Δpos/Δt by instance_token
        │   src/reasoning/relations.py     zone, gap, closing speed, TTC
        ▼
Relations
        │   src/reasoning/rules.py         R1 BRAKE · R2 STOP/YIELD ·
        │                                  R3 FOLLOW · R4 INHIBIT_LANE_CHANGE ·
        │                                  R5 CRUISE (fallback)
        ▼
Findings[]
        │   src/reasoning/arbiter.py       pick the winner by priority
        ▼
Decision (action + full justification trace)
        │   src/reasoning/narrate.py       compose the Reasoned Alert
        ▼
Narration → SignatureOutput { context, risk, action, reason }
        │   src/viz.py            BEV plot · app.py: Streamlit UI
        ▼
Reasoned Alert from PRISM layer
```

Every layer is a small, pure function module. Thresholds live in `config/rules.yaml`; the class/state taxonomy lives in `config/taxonomy.yaml`. **No magic numbers in code.**

---

## Reasoned Alert — the canonical output

The 4-field schema is fixed across all decisions:

| Field   | Question                                | What it captures |
|---------|-----------------------------------------|------------------|
| CONTEXT | *What is happening in Road?*            | Semantic interpretation of the driving environment that traditional ADAS does not explicitly produce (VRU safety event, lane-change intent conflict, cooperative car-following, behavioural threats). |
| RISK    | *Level of Risk*                         | Deterministic hybrid score → `HIGH` / `MEDIUM` / `LOW`, computed from rule priority + how deep we are inside the trigger zone. |
| ACTION  | *Action taken by ADAS*                  | Active vehicle function being engaged right now (AEB, Auto-Hold, ACC, Lane-Change Inhibit, Cruise Control). |
| REASON  | *Reasoning behind the Action*           | Plain-language sentence written to the driver, quoting the actual threshold the rule used. |

Six action classes are supported end-to-end: **BRAKE, STOP, YIELD, FOLLOW, INHIBIT_LANE_CHANGE, CRUISE**.

---

## Repository layout

```
PRISM_ADAS/
├── README.md
├── requirements.txt               streamlit, numpy, matplotlib, pyquaternion, pyyaml, pandas
├── app.py                         Streamlit demo (BEV + Reasoned Alert)
├── config/
│   ├── rules.yaml                 all numeric thresholds
│   └── taxonomy.yaml              23 categories + 8 attributes → simplified classes/states
├── src/
│   ├── store.py                   JSON-only nuScenes loader (NuMiniStore)
│   ├── object_list.py             DetectedObject + global-to-ego transform
│   ├── velocity.py                cross-frame velocity by instance_token
│   ├── viz.py                     BEV matplotlib renderer (dark/light themes)
│   ├── pipeline.py                end-to-end per-frame orchestration
│   ├── config.py                  YAML loaders
│   └── reasoning/
│       ├── relations.py           Layer A: spatial facts (zone, gap, TTC, …)
│       ├── rules.py               Layer B: R1..R5 situation rules
│       ├── arbiter.py             Layer C: highest-priority Finding wins
│       └── narrate.py             Reasoned Alert + structured narration
├── tests/                         96 unit tests, no dataset dependency for rule tests
│   ├── test_store.py
│   ├── test_object_list.py
│   ├── test_rules.py
│   ├── test_arbiter.py
│   └── test_narrate.py
└── nuscenes-mini-JSON/            the 13 JSON tables (you supply this)
```

---

## Quick start

### Try it without installing anything

The app is hosted on Hugging Face Spaces:

**→ [huggingface.co/spaces/Lokikumar/PRISM_ADAS](https://huggingface.co/spaces/Lokikumar/PRISM_ADAS)**

Pick a scene from the dropdown and the BEV plot + Reasoned Alert panel render against the bundled nuScenes Mini JSON. Nothing to install locally.

### Or run it locally

#### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

#### 2. Provide the nuScenes Mini JSON tables

Place the 13 JSON files (`attribute.json`, `calibrated_sensor.json`, `category.json`, `ego_pose.json`, `instance.json`, `log.json`, `map.json`, `sample.json`, `sample_annotation.json`, `sample_data.json`, `scene.json`, `sensor.json`, `visibility.json`) under either `./nuscenes-mini-JSON/` or `./nuscenes-mini-JSON/v1.0-mini/`. The loader auto-detects either layout.

> **No sensor files are required.** This project reads only the 13 JSONs. There are no `.jpg`, `.pcd`, or `.bin` reads anywhere in the codebase.

#### 3. Run the Streamlit demo

```bash
streamlit run app.py
```

Pick a scene from the dropdown. The page shows:
- a dark BEV plot of the scene (ego at origin, objects coloured by class, ego/adjacent-lane bands shaded, pedestrian caution/stop radii dashed); and
- the **Reasoned Alert from PRISM layer** panel with CONTEXT / RISK / ACTION / REASON locked to the same height.

#### 4. Inspect a scene from the CLI

```bash
python -m src.pipeline ./nuscenes-mini-JSON 0 6
```

Prints the detailed per-frame justification trace (perception, step-by-step reasoning, alternatives considered, confidence) for the first 6 frames of scene 0.

#### 5. Run the tests

```bash
python -m pytest tests/ -q
```

96 tests cover the loader, the perception layer, the rules (R1–R5) on synthetic inputs, the arbiter, and the Reasoned Alert composition — including a contract test that pins the semantic CONTEXT phrasing.

---

## Data contracts (the interface for the ML model)

These are the structured types the rule-based prototype already produces and the AI reasoning model will be trained to match:

```python
@dataclass
class EgoState:
    x: float; y: float; yaw: float        # global frame
    speed: float                          # m/s
    timestamp: int                        # μs

@dataclass
class DetectedObject:                     # ego frame, +x forward, +y left
    id: str                               # instance_token (stable across frames)
    raw_category: str                     # nuScenes category verbatim
    cls: str                              # VEHICLE / LARGE_VEHICLE / PEDESTRIAN / CYCLIST / STATIC
    state: str                            # MOVING / STOPPED / PARKED / STANDING
    x: float; y: float; distance: float; yaw: float
    size: tuple[float, float, float]
    vx: float | None; vy: float | None    # cross-frame velocity
    num_lidar_pts: int; visibility: int

@dataclass
class Finding:
    action: str; priority: int; reason: str
    object_id: str | None; rule: str

@dataclass
class Decision:
    action: str; priority: int; primary_reason: str
    supporting_findings: list[Finding]    # full priority-ordered trace
    frame_token: str; num_objects: int

@dataclass
class SignatureOutput:                    # the Reasoned Alert
    context: str                          # semantic environment interpretation
    risk: str                             # HIGH / MEDIUM / LOW
    action: str                           # active vehicle function
    reason: str                           # plain-language driver-facing prose
```

Every frame in every scene of the dataset can be exported as `(perception inputs) → (Reasoned Alert)` — exactly the training pair needed by the downstream reasoning model.

---

## Design constraints (kept throughout)

- **Fully offline, JSON-only.** No `nuscenes-devkit` default loader, no `.jpg` / `.pcd` / `.bin` reads, no network calls.
- **Deterministic.** Same input → identical output. No randomness, no LLM in the rule layer.
- **Explainable.** Every Decision carries a full priority-ordered list of Findings — a justification trace.
- **Configurable.** Every threshold is in `config/rules.yaml`; every class/state mapping is in `config/taxonomy.yaml`. Switching the prototype to the full nuScenes JSON requires only pointing the path at the larger folder.

---

## Roadmap

This repository delivers Phase 1 — the rule-based scaffold and the canonical Reasoned Alert format.

The follow-on phases (the real objective of the project):

1. **Trace export.** A batch job that runs the rule-based engine across the full nuScenes (and India-specific) datasets and emits `(perception, Reasoned Alert)` JSONL.
2. **Reasoning model training.** A learned model trained on those traces — initial target is a small instruction-tuned transformer that takes the per-frame `ObjectList` as structured input and produces the four Reasoned Alert fields directly.
3. **Indian-road grounding.** Extending the taxonomy and the situation classes (auto-rickshaws, two-wheelers weaving, mixed-flow intersections, unmarked lanes) so the reasoning layer reads situations that aren't expressible in nuScenes-only data.
4. **Comparative evaluation.** Holding the learned model to the same explainability + determinism contract as the rule-based scaffold, with the rule engine acting as one of the baselines.

The interface between the perception stack, the reasoning layer, and the HMI is already fixed by the data contracts above — the rule-based engine is just the first implementation behind that interface.

---

## Acknowledgements

- **nuScenes** (Aptiv / Motional) for the Mini dataset annotations used as the validation corpus.
- The rule definitions and threshold defaults are spelled out in `config/rules.yaml`; everything else is built on top of `numpy`, `matplotlib`, `pyquaternion`, `pyyaml`, `streamlit`, and `pandas` only.
