# HARMONY

**HARMONY is a personal financial enforcement layer that sits between user intent and execution.**  
It prevents bad financial mistakes by simulating future commitments and enforcing user-defined constraints before any action proceeds.

This is **not** a payment app, chatbot, or budgeting tool.  
HARMONY stages, evaluates, negotiates, and logs decisions before execution.

---

## What Problem Does HARMONY Solve?

Payments are easy.  
Financial mistakes happen because decisions are made under impulse, stress, or incomplete context.

Existing tools either:
- execute blindly (autopay), or
- warn without authority (alerts and dashboards).

HARMONY enforces what users already know they should do — at the exact moment a decision is made.

---

## Core Principles

- Intent before execution  
- Simulation before commitment  
- User-owned constraints  
- No silent actions  
- Full auditability and reversibility  

---

## How HARMONY Works

1. User speaks or types an intent (e.g., *“pay my credit card bill”*)
2. HARMONY reads:
   - current balance  
   - transaction history  
   - future commitments (subscriptions, EMIs, bills)  
   - user non-negotiables  
3. Multiple future scenarios are simulated in real time  
4. Outcomes are evaluated against constraints  
5. A decision is returned:
   - **ALLOW**  
   - **SUGGEST ALTERNATIVE**  
   - **REFUSE**  
6. The action is staged as **PENDING**  
7. Execution happens only after confirmation or override  
8. Every step is logged

---
## Architecture Diagram 
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/34d90a57-65d6-4d4f-8b4c-3e2457af5bbe" />


## What Makes HARMONY Different

- **Intent-triggered future simulation**  
- **User-owned financial veto power**  
- **Enforcement with consent and auditability**  
- **Decisions are staged, not executed blindly**

HARMONY does not decide *for* the user — it enforces what the user has explicitly declared as important.

---

## Behavioral Learning

HARMONY adapts confidence based on:
- overrides  
- repeated decisions  
- regret signals  
- stress patterns  

Learning influences *how strongly* decisions are enforced — never removes user control.

---

## Architecture Overview

- Voice & Text Interface  
- Veto Engine (core authority)  
- Simulation Engine  
- Decision Log  
- Transaction History  
- Risk & Alerts Layer  

The Veto Engine is the **single source of truth**.

---

## Technology Stack

- Python  
- FastAPI  
- LiveKit (real-time voice)  
- Deepgram (speech-to-text)  
- LLMs (language understanding only)  
- Firebase (state, logs, history)  

LLMs **never** make financial decisions.

---

## Roadmap

- Emergency and context modes  
- Trustee-based delegation  
- Agent-to-agent enforced payments  
- Institutional and platform licensing  

---

## Philosophy

Autopay executes transactions.  
Finance apps warn after the fact.

**HARMONY enforces user intent at the moment it matters.**
