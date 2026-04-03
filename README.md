# Activation Engineering for Personality Alignment in LLMs

**Accepted at EACL 2025 (Main Conference)**

This repository contains code and resources for our work on **activation engineering for controllable behavioural alignment in Large Language Models (LLMs)**.

The project studies how internal representations of LLMs can be used to steer behavioural tendencies such as personality traits in a **stable, interpretable, and efficient way**, without modifying model weights.

---

## Overview

Modern LLMs exhibit behavioural tendencies that resemble psychological constructs such as personality traits, tone, and interaction style. However, controlling these behaviours reliably remains challenging.

Traditional approaches such as fine-tuning or reinforcement learning require retraining the model and often introduce instability or interference across behaviours.

Activation engineering provides an alternative approach by modifying the model **at inference time** through small, structured adjustments to internal representations.

Our work focuses on understanding:

- how behavioural traits are represented inside LLM activations
- how these representations can be controlled reliably
- how multiple traits interact in shared representational space
- how to steer behaviour without degrading model capability

The work is grounded in psychological frameworks such as the **Big Five personality traits (OCEAN)** and explores how behavioural characteristics can be encoded and manipulated in neural representations. :contentReference[oaicite:0]{index=0}

---

## Problem

Existing activation steering methods often rely on selecting a fixed layer (e.g. middle layer) for intervention. However:

- different models respond differently across layers
- different traits may appear in different parts of the network
- the same layer may not work consistently across prompts
- fixed layer selection leads to unstable or weak steering effects

As a result, steering may appear inconsistent or unreliable across tasks and architectures. :contentReference[oaicite:1]{index=1}

This creates a challenge similar to psychology:

behaviour is not controlled by a single factor, but emerges from multiple interacting processes.

---

## Core Idea

We model behavioural traits as **directions in activation space** and identify a shared structure that captures how traits are represented internally.

The approach learns trait representations from examples labelled with high and low levels of each behavioural characteristic.

These representations are then used to gently influence the model’s internal state during generation.

Importantly, this allows behaviour to be adjusted **without retraining the model**, preserving its general knowledge and reasoning ability.

---

## Hybrid Layer Selection (Conceptual Explanation)

A key contribution of this work is a **hybrid strategy for selecting where to intervene inside the model**.

Rather than assuming one fixed layer works for all situations, we combine two complementary signals:

### 1. Offline verification (global reliability)

We first analyse model activations across many examples to identify layers that consistently respond to behavioural differences.

These layers represent locations where behavioural information is clearly encoded.

This step provides a stable prior indicating where steering is most likely to be effective.

---

### 2. Dynamic prompt sensitivity (local adaptivity)

Different prompts activate different parts of the network.

At runtime, we measure which layers respond most strongly to the current prompt.

This allows the intervention to adapt to the specific context.

---

### 3. Combined hybrid selection

The final steering location is selected by combining:

- globally reliable layers (stable behavioural signal)
- locally responsive layers (prompt-specific signal)

This balance improves robustness across:

- different models
- different prompts
- different behavioural traits

The hybrid method produces stronger and more stable behavioural shifts compared to using either static or dynamic selection alone. :contentReference[oaicite:2]{index=2}

Conceptually, this is similar to psychology:

stable traits interact with situational context to produce behaviour.

---

## Method Summary

The pipeline consists of four main stages:

### 1. Extract behavioural directions
Using datasets labelled with behavioural attributes, we identify activation patterns associated with high and low trait levels.

### 2. Learn shared representation structure
Trait directions are organised into a low-dimensional structure capturing relationships between behavioural tendencies.

This helps reduce noise and improves interpretability.

### 3. Hybrid layer selection
We identify intervention points using a combination of:

- offline diagnostic analysis
- runtime responsiveness signals

### 4. Inference-time steering
Behaviour is adjusted by injecting small structured perturbations into the model’s residual stream using forward hooks.

This allows controllable behaviour without modifying model parameters. :contentReference[oaicite:3]{index=3}

---

## Benchmarks and Evaluation

We evaluate behavioural control across multiple scenarios:

- personality questionnaires
- open-ended dialogue
- discourse-based evaluation
- reasoning benchmarks (capability preservation)

Key evaluation dimensions include:

- strength of behavioural control
- preservation of fluency
- stability across prompts
- consistency across models

Results show improved trait separation and stable behaviour across architectures using the hybrid approach. :contentReference[oaicite:4]{index=4}

---

## Motivation from Psychology

Human behaviour is often modeled as interaction between:

- stable latent traits
- contextual influences
- cognitive constraints

Similarly, LLM behaviour emerges from structured internal representations that can be analysed and influenced.

This work explores how psychological concepts such as behavioural consistency, trait interaction, and latent structure can inform more interpretable AI alignment methods.

---

## Research Direction

This project contributes toward:

- interpretable alignment methods
- controllable behavioural generation
- personality-consistent agents
- activation-level alignment techniques
- efficient alternatives to full fine-tuning

Potential applications include:

- personalised AI assistants
- consistent conversational agents
- human-aligned interaction design
- controllable simulation environments

---

