# 🌍 GeoAI

An Akinator-style AI that guesses countries through yes/no questions using a Bayesian inference engine.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Backend](https://img.shields.io/badge/Backend-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/rafsan1711/geoai-backend)
[![CI](https://github.com/rafsan1711/geoai/actions/workflows/sync-backend.yml/badge.svg)](https://github.com/rafsan1711/geoai/actions)

## How It Works

1. User thinks of a country
2. AI asks yes/no/probably questions (continent, population, landlocked, etc.)
3. Bayesian engine narrows down candidates using information gain
4. AI guesses the country — usually within 10–25 questions

## Repository Structure

```
.
├── .github
│   └── workflows
│       └── update-readme-structure.yml
├── Chessmate Data Prep.ipynb.md
├── Chessmate_v2_DataPrep.ipynb.md
├── Chessmate_v2_ResNet_Training.ipynb.md
├── ENDGAME ORACLE V2.ipynb.md
├── ENDGAME_ORACLE_V2.ipynb
├── G-ROADMAP.md
├── GambitFlow_Opening_Architect.ipynb.md
├── GambitFlow_Opening_Architect_v2.ipynb
├── GambitFlow_Synapse_Base_Training.ipynb.md
├── GambitFlow_Tactical_Forge.ipynb.md
├── Match_Data_Curator.ipynb
├── Nexus-Core-container.txt
├── Plan.md
├── Prompt.md
├── README.md
├── Synapse_Edge_01_DataPrep.ipynb.md
├── Synapse_Edge_DataSplitter.ipynb.md
├── TACTICAL_FORGE_V2.ipynb
├── a.md
├── auth-screen-div.html
├── big_g_relay.py
├── build.txt
├── index.html
├── official_training.ipynb.md
├── secrets
│   └── Service.json
└── vortexAlpha.md
```

## Setup

### Backend (HuggingFace Docker Space)

1. Create a [HuggingFace Docker Space](https://huggingface.co/new-space?sdk=docker)
2. Add GitHub secret `HF_TOKEN` (your HF write token)
3. Push to `main` — GitHub Actions auto-syncs the `backend/` folder

### Frontend

Static files — deploy anywhere (GitHub Pages, Cloudflare Pages, etc.).  
Update `frontend/js/config.js` with your HF Space URL.

### Environment Variables (HF Space Secrets)

| Variable | Description |
|----------|-------------|
| `FIREBASE_DATABASE_URL` | Firebase RTDB URL |
| `FIREBASE_API_KEY` | Firebase API key |

## License

[GNU General Public License v3.0](LICENSE)