# Voice-First Finance Agent - System Architecture

## Overview
This document describes the complete system architecture of the Voice-First Finance Agent, a multi-layered application for voice-based financial decision-making.

## System Architecture Diagram

![Voice-First Finance Agent Architecture](docs/architecture-diagram.png)

## Architecture Layers

### Layer 1: Client
**Browser (React + Vite Frontend)**
- UI: Harmony Slash + Radix UI
- Real-time state synchronization with Firebase Auth
- Handles JWT session management
- Connection: `onAuthStateChanged` events and JWT tokens

### Layer 2: Real-Time Transport
**FastAPI Server (Server.py)**
- REST API endpoints
- WebSocket bridge for bidirectional communication
- Endpoints:
  - `GET /token` - Issues JWT tokens for LiveKit access
  - `POST /room_join` - Handles room joining, publishing, subscribing
  - `WS /ws` - WebSocket relay for real-time messages

**LiveKit Room**
- Real-time audio and data channel communication
- Audio tracks (UDP/TCP + SRTP/DTLS)
- Data channel for text/metadata (JSON, API calls)

### Layer 3: Voice Processing Pipeline
**Status: IN PROGRESS / PARTIALLY WIRED**

#### 3a. Silero VAD (Voice Activity Detection)
- **Purpose**: Detects speech and silence
- **Integration**: Local, detect_speech dependency present
- **Status**: Not yet wired into AgentSession

#### 3b. Deepgram STT (Speech-to-Text)
- **Purpose**: Converts audio to text
- **API**: Cloud API for accurate transcription
- **Dependencies**: Not yet fully wired into AgentSession
- **Status**: In progress

#### 3c. Python Agent (main.py)
- **Purpose**: Main orchestrator for voice pipeline
- **Framework**: LiveKit Agents SDK
- **Current Functionality**: 
  - Handles TEXT input via data_received events
  - Does NOT yet have full audio pipeline wired
  - Communicates with Layer 4 (LLM) and returns responses

### Layer 4: Language Understanding
**LLM (OpenAI GPT-4o-mini or OpenRouter via raw HTTP)**
- **Function**: 
  - Maintains conversation history with `chat_ctx`
  - Understands user intent (does NOT make financial decisions by design)
  - ROLE: Understands intent only
- **Architecture Principle**: 
  - **LLM = Language**
  - **Veto Engine = Authority**

### Layer 5: State & Data
**Firebase (Real-time Database + Cloud Storage)**
- **users/{uid}** document:
  - Commitments
  - Balance, settings
- **users/{uid}/commitments/{tid}** subcollection:
  - title, amount, dueDate, urgency level
- **Real-time Listener (onSnapshot)**:
  - Syncs commitments to frontend via WebSocket/Firebase

**Realtime Listener (onSnapshot)**
- Watches Firestore for changes
- Syncs updates to frontend UI via Firebase WebSocket

### Layer 6: Decision Engine
**Deterministic (No LLM)**

#### 6a. Simulation Engine (simulation-engine/src/engine.js)
- **Function**: Financial simulation
- **Strategies**: 
  - `pay_credit_card` - Payment optimization for credit cards
  - `pay_earliest_emi` - Earliest EMI strategy
  - `pay_emergency` - Emergency fund preservation
- **Parallel Processing**: ALL strategies run in parallel per request
- **Output**: JSON projections (month-by-month breakdown)

#### 6b. Veto Engine (veto-engine/src/engine.js)
- **Function**: Rule-based authority for financial decisions
- **Input**: 
  - `total_penalties`
  - `cash_buffer_breach_month`
  - `emi_miss_month`
  - `balance` after each strategy
  - Cross-strategy rule evaluation
- **Rules**:
  - Valid: FALSE if balance < min_cash_buffer OR breach_month exists
  - OR breach_month exists AND strategy has no EMI miss
  - Invalid strategies excluded from recommendations
- **Output**: `recommended_strategy` (lowest cost among valid, tiered by highest post-action balance)

#### Decision Result (Layer 6 Output)
- **Format**: JSON with structure:
  ```json
  {
    "ALLOW": { "strategy": "...", "reason": "..." },
    "SUGGEST_ALTERNATIVE": { "strategy": "...", "reason": "..." },
    "REFUSE": { "reason": "..." }
  }
  ```

### Layer 7: Response Path

#### 7a. Browser (React + Vite)
- Receives decision results via data channel
- Displays recommendations to user

#### 7b. Python Agent (main.py)
- Publishes decision results
- Reliable data channel delivery

#### 7c. LiveKit Room
- Real-time audio/data delivery
- Syncs with onSnapshot (Firebase)

## Main Flow Summary (Order of Operations)

| Step | Component | Action |
|------|-----------|--------|
| 1 | JWT | Client authenticates via Firebase, receives JWT |
| 2 | LiveKit | Client joins room via FastAPI, Server SDK establishes audio/data channel |
| 3 | WebSocket | FastAPI relays messages between browser and Python Agent |
| 4 | Voice Input | Audio captured (VAD/STT in progress) OR text input via data channel |
| 5 | Data Channel | Text payload sent to Python Agent |
| 6 | LLM | Python Agent queries LLM, maintains conversation context |
| 7 | Simulation | Simulation Engine runs ALL strategies in parallel |
| 8 | Veto | Veto Engine validates/filters strategies |
| 9 | Response | Decision Result sent via data channel to Browser |
| 10 | Firestore | Real-time Listener syncs commitments update to frontend |
| 11 | Veto Rules | Applied post-action, ensures no invalid strategies |

## Key Architectural Principles

1. **LLM = Language Understanding Only**
   - LLM understands intent but does NOT make financial decisions
   - Decision authority delegated to Veto Engine

2. **Veto Engine = Authority**
   - Rule-based constraints ensure financial safety
   - No LLM involvement in final decision logic

3. **Real-Time Synchronization**
   - Firebase onSnapshot for state consistency
   - WebSocket for command/response flow

4. **Parallel Processing**
   - All financial strategies evaluated simultaneously
   - Veto Engine filters based on constraints

## Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Browser (UI) | ✅ Complete | React + Vite with Harmony Slash + Radix UI |
| FastAPI Server | ✅ Complete | REST + WebSocket endpoints functional |
| Firebase Integration | ✅ Complete | Auth, Firestore, Real-time Listener |
| Silero VAD | ⏳ In Progress | Dependency present, not wired into AgentSession |
| Deepgram STT | ⏳ In Progress | Not yet fully integrated |
| Python Agent | ⏳ Partial | TEXT pipeline working, audio pipeline in progress |
| LLM Integration | ✅ Complete | OpenAI + OpenRouter support |
| Simulation Engine | ✅ Complete | All strategies implemented |
| Veto Engine | ✅ Complete | Rule validation logic implemented |
| Decision Result | ✅ Complete | JSON response format finalized |

## Environment Configuration

Secrets loaded via environment variables:
- `FIREBASE_API_KEY`
- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`
- `DEEPGRAM_API_KEY`

See `.env.example` for full configuration.

---

**Last Updated**: 2024
**Project Status**: Active Development
