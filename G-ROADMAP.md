# GuessMyPlace - Prompt For Claude ai

Amra ekta Akinator Type Country, City, Historic Place guessing Game banabo. akinator Zevabe kaj kore shevabei amra kajj korbo.
to tumi ekta Roadmap.md banaba zetate lekha thakbe, amra ki banacchi. and onekgulo milestone and steps likhba za dhore dhore amra agabo. ekta md file dekhiye alada alada ai ke diyeo kaj korano zay step by step, emon korba. 
1.   to ami shob code Github e rakhbo. eta ekdom Professionally manage korbo professional softwares er moto.
2. to github e shob code rakhar por site amra host korbo vercel e and backend Host korbo Huggingface docker space e . to amra ekdom modular structure e banabo , onek onek file onek onek folder, onek onek advanced languages er file thakbe shekhane. .
3. to Github Repo er Ekta Folder er sathe Huggingface docker space repo sync korbo. https://huggingface.co/docs/hub/spaces-github-actions eta Read kore bujhe niyo.
4. ar frontend ar backend amra advanced api diye connect korbo. akinator zemon onek powerfull amader tao onek powerfull hobe.
5. amra onek onek advanced tech stack use korbo. shob theke Best tech stack shob jaygay. ar ekhane etake zodi improve korar kono way thake Database and cache diye tahole korba, zodi 100% free way hoy, like firebase RTDB 1GB.
6. amra backend e c++ ,python , flask shoho aro onek onek advanced jinish, languages, techs, algorithms use korbo karon eta hobe open source er Continuosly improve hote thakbe bochorer por bochor. license tomar deya lagbe na ami nije banabo GPL v3.
7. ami github.dev ar VSCode O use korte parbo.
8. tumi CI CD shoho aro onek onek advanced jinish use korba jeno ami etake fully complete korar por contributors der kache dite pari. onek onekgulo languages O use korba like Docker ar Kubernetes. aro onek onek advanced jinish za company ba open source projects ra kore. amra site keo onek onek shundor and professional korbo.
10. ar Important kotha. Kono Code improve ba new feature add korte gele Zeno full code jana na lage. kono ekta Feature ba Code Improve korte Shudhu Oi part er Code janlei zeno hoy. Zemon Claude ai ke ami koyek bochor por Full code dite parbo na karon Context full hoye zabe. kintu ami kono specific part er code diye zeno take diye code korate pari, eta zodi kora na zay tahole optional dhore nao.
11. are ekhane ki Machine Learning er dorkar ache? tahole use Amra kortam. but mone rakhba ze Space er Data kintu sleep korlei hariye zay, shob docs pore niba.
12. Data Shomoyer sathe sathe Boro korte thakbo.
ar steps er modhye testing O rakhba debug korte shohoj zeno hoy . Space e 2 vCPU and 16GB RAM. apatoto eta ami eka and friends rai Use korbe. amar etar Name Hobe GuessMyPlace. ar Data manually Github thekei Edit kore Data barano hobe and enhance kora hobe. ar dataset onekvabei generate kora zay,like ai diye aro bivinno vabe, ar data O Modular hobe. to Ok start, ROADMAP.md amake ekta Artifact e diba. abari bolchi, artifact.

# GuessMyPlace - Development Roadmap

## Project Overview
**GuessMyPlace** is an Akinator-style guessing game for Countries, Cities, and Historic Places. The system uses intelligent question algorithms to guess what the user is thinking through strategic yes/no questions.

### Tech Stack
- **Frontend**: React + TypeScript + Tailwind CSS + Vite
- **Backend**: Python (Flask) + C++ (Performance-critical algorithms)
- **Database**: Firebase Realtime Database (1GB free tier)
- **Cache**: Redis (via Upstash free tier)
- **Hosting**: 
  - Frontend: Vercel
  - Backend: Hugging Face Docker Space (2 vCPU, 16GB RAM)
- **Container**: Docker + Docker Compose
- **CI/CD**: GitHub Actions
- **Version Control**: Git + GitHub
- **License**: GPL v3

---

## Repository Structure

```
GuessMyPlace/
├── .github/
│   └── workflows/
│       ├── frontend-deploy.yml
│       ├── backend-deploy.yml
│       ├── tests.yml
│       └── sync-huggingface.yml
├── frontend/
│   ├── src/
│   ├── public/
│   ├── tests/
│   ├── package.json
│   ├── tsconfig.json
│   └── Dockerfile
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   ├── services/
│   │   ├── models/
│   │   └── utils/
│   ├── algorithms/
│   │   ├── cpp/
│   │   │   ├── decision_tree.cpp
│   │   │   ├── probability_engine.cpp
│   │   │   └── CMakeLists.txt
│   │   └── python/
│   │       ├── question_selector.py
│   │       └── answer_analyzer.py
│   ├── tests/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   ├── places/
│   │   ├── countries.json
│   │   ├── cities.json
│   │   └── historic_places.json
│   ├── questions/
│   │   └── question_bank.json
│   ├── schema/
│   │   └── data_schema.json
│   └── scripts/
│       ├── generate_data.py
│       ├── validate_data.py
│       └── enhance_data.py
├── docs/
│   ├── API.md
│   ├── ARCHITECTURE.md
│   ├── CONTRIBUTING.md
│   └── DATA_FORMAT.md
├── scripts/
│   ├── setup.sh
│   ├── deploy.sh
│   └── test.sh
├── .env.example
├── .gitignore
├── docker-compose.yml
├── LICENSE
└── README.md
```

---

## Milestone 1: Project Foundation & Setup
**Goal**: Set up infrastructure, repository, and basic project skeleton

### Step 1.1: Repository & Infrastructure Setup
- [ ] Create GitHub repository with GPL v3 license
- [ ] Set up branch protection rules (main, develop)
- [ ] Create `.gitignore` for Python, Node, C++
- [ ] Set up GitHub Projects board for task tracking
- [ ] Create initial `README.md` with project description
- [ ] Add `.env.example` file with all environment variables

### Step 1.2: Documentation Foundation
- [ ] Create `docs/ARCHITECTURE.md` - System architecture overview
- [ ] Create `docs/API.md` - API endpoint documentation
- [ ] Create `docs/CONTRIBUTING.md` - Contribution guidelines
- [ ] Create `docs/DATA_FORMAT.md` - Data structure specification

### Step 1.3: Development Environment
- [ ] Set up Docker configuration
- [ ] Create `docker-compose.yml` for local development
- [ ] Write setup scripts (`scripts/setup.sh`)
- [ ] Test environment in VSCode and github.dev

**Deliverables**: 
- Organized repository structure
- Complete documentation templates
- Working local development environment

---

## Milestone 2: Data Layer & Schema Design
**Goal**: Design and implement modular, scalable data structures

### Step 2.1: Data Schema Design
- [ ] Design JSON schema for places (countries, cities, historic places)
- [ ] Design question bank schema with metadata
- [ ] Create relationship mapping (place → characteristics)
- [ ] Define data validation rules
- [ ] Document schema in `data/schema/data_schema.json`

### Step 2.2: Initial Dataset Creation
- [ ] Create 50+ countries with characteristics
- [ ] Create 100+ cities with characteristics  
- [ ] Create 50+ historic places with characteristics
- [ ] Create 200+ questions with discriminating power
- [ ] Validate all data against schema

### Step 2.3: Data Management Tools
- [ ] Build `generate_data.py` - AI-assisted data generation
- [ ] Build `validate_data.py` - Schema validation tool
- [ ] Build `enhance_data.py` - Add missing characteristics
- [ ] Create data versioning strategy
- [ ] Add data statistics generator

**Deliverables**:
- Complete data schema
- Initial dataset (200+ places, 200+ questions)
- Data management tools

---

## Milestone 3: Backend Core - Algorithm Implementation
**Goal**: Build the intelligent question selection and guessing algorithms

### Step 3.1: C++ Performance Engine
- [ ] Set up C++ build system (CMake)
- [ ] Implement decision tree algorithm
- [ ] Implement information gain calculator
- [ ] Implement probability scoring engine
- [ ] Create Python bindings (pybind11)
- [ ] Write unit tests for C++ modules

### Step 3.2: Python Algorithm Layer
- [ ] Implement question selection strategy
- [ ] Build answer history tracker
- [ ] Create place probability calculator
- [ ] Implement "Did I guess correctly?" logic
- [ ] Build backtracking for wrong guesses
- [ ] Write unit tests (pytest)

### Step 3.3: Game Logic Service
- [ ] Session management system
- [ ] Game state persistence
- [ ] Question history tracking
- [ ] Guess validation logic
- [ ] Learn from wrong guesses feature

**Deliverables**:
- Working C++ performance modules
- Complete game logic implementation
- Comprehensive test suite

---

## Milestone 4: Backend API Development
**Goal**: Create RESTful API with Flask

### Step 4.1: Flask Application Structure
- [ ] Set up Flask app with blueprints
- [ ] Configure CORS for frontend
- [ ] Add request/response logging
- [ ] Implement error handling middleware
- [ ] Set up rate limiting

### Step 4.2: Core API Endpoints
- [ ] `POST /api/game/start` - Start new game session
- [ ] `POST /api/game/answer` - Submit answer to question
- [ ] `GET /api/game/state/:sessionId` - Get current game state
- [ ] `POST /api/game/guess` - Validate final guess
- [ ] `POST /api/game/feedback` - Learn from incorrect guess
- [ ] `GET /api/stats` - Get game statistics

### Step 4.3: Database Integration
- [ ] Set up Firebase Realtime Database
- [ ] Implement session storage
- [ ] Add game statistics tracking
- [ ] Build cache layer (Redis/Upstash)
- [ ] Implement cache invalidation strategy

### Step 4.4: Testing & Documentation
- [ ] Write API integration tests
- [ ] Create Postman collection
- [ ] Generate OpenAPI/Swagger docs
- [ ] Add request/response examples

**Deliverables**:
- Complete REST API
- Database integration
- API documentation

---

## Milestone 5: Frontend Development
**Goal**: Build beautiful, responsive UI with React + TypeScript

### Step 5.1: Project Setup
- [ ] Initialize Vite + React + TypeScript
- [ ] Configure Tailwind CSS
- [ ] Set up React Router
- [ ] Configure Axios for API calls
- [ ] Add state management (Zustand/Context)

### Step 5.2: Core Components
- [ ] Landing page with game start
- [ ] Question display component
- [ ] Answer buttons (Yes/No/Don't Know/Probably/Probably Not)
- [ ] Progress indicator
- [ ] Guess reveal animation
- [ ] Feedback form for wrong guesses

### Step 5.3: Advanced Features
- [ ] Game history viewer
- [ ] Statistics dashboard
- [ ] Multi-language support (EN, BN)
- [ ] Dark/Light mode toggle
- [ ] Responsive design (mobile-first)
- [ ] Loading states & error handling

### Step 5.4: UI/UX Polish
- [ ] Animations with Framer Motion
- [ ] Sound effects (optional)
- [ ] Accessibility (ARIA labels, keyboard nav)
- [ ] Performance optimization
- [ ] PWA capabilities

**Deliverables**:
- Complete frontend application
- Responsive, accessible UI
- Smooth animations

---

## Milestone 6: Docker & Containerization
**Goal**: Containerize backend for Hugging Face Space deployment

### Step 6.1: Backend Dockerfile
- [ ] Create multi-stage Dockerfile
- [ ] Optimize image size
- [ ] Add health check endpoint
- [ ] Configure environment variables
- [ ] Test local container build

### Step 6.2: Docker Compose Setup
- [ ] Configure services (Flask, Redis)
- [ ] Set up networking
- [ ] Add volume mounts for development
- [ ] Create production compose file
- [ ] Test full stack locally

### Step 6.3: Hugging Face Space Configuration
- [ ] Create Space repository
- [ ] Configure Space settings (Docker, CPU, RAM)
- [ ] Add secrets/environment variables
- [ ] Test deployment

**Deliverables**:
- Production-ready Docker images
- Working Hugging Face Space

---

## Milestone 7: CI/CD Pipeline
**Goal**: Automate testing, building, and deployment

### Step 7.1: GitHub Actions - Testing
- [ ] Frontend test workflow (Jest, Vitest)
- [ ] Backend test workflow (pytest)
- [ ] C++ test workflow
- [ ] Linting workflow (ESLint, Black, clang-format)
- [ ] Coverage reporting

### Step 7.2: GitHub Actions - Deployment
- [ ] Frontend deploy to Vercel workflow
- [ ] Backend sync to Hugging Face workflow
- [ ] Automated versioning (semantic-release)
- [ ] Release notes generation

### Step 7.3: Monitoring & Alerts
- [ ] Health check monitoring
- [ ] Error alerting (optional: Sentry)
- [ ] Performance monitoring
- [ ] Uptime tracking

**Deliverables**:
- Fully automated CI/CD
- Continuous deployment pipeline

---

## Milestone 8: Advanced Features & Optimization
**Goal**: Add ML, caching, and performance improvements

### Step 8.1: Machine Learning Enhancement
- [ ] Train decision tree on game data
- [ ] Implement adaptive question ordering
- [ ] Add confidence scoring
- [ ] Build feedback learning system
- [ ] Save/load model checkpoints

### Step 8.2: Caching Strategy
- [ ] Implement Redis caching for frequent queries
- [ ] Cache question selection results
- [ ] Cache probability calculations
- [ ] Add cache warming on startup
- [ ] Monitor cache hit rates

### Step 8.3: Performance Optimization
- [ ] Profile C++ algorithms
- [ ] Optimize database queries
- [ ] Implement lazy loading
- [ ] Add CDN for static assets
- [ ] Bundle size optimization

**Deliverables**:
- ML-powered question selection
- High-performance caching
- Optimized application

---

## Milestone 9: Data Expansion & Community
**Goal**: Scale dataset and prepare for open-source contributions

### Step 9.1: Dataset Scaling
- [ ] Expand to 500+ places
- [ ] Add 500+ questions
- [ ] Improve characteristic coverage
- [ ] Add regional diversity
- [ ] Create data contribution guide

### Step 9.2: Contributor Tools
- [ ] Create data submission template
- [ ] Build automated data validation
- [ ] Add contributor guidelines
- [ ] Create issue templates
- [ ] Set up discussion forums

### Step 9.3: Documentation
- [ ] Write architecture deep-dive
- [ ] Create video tutorials
- [ ] Document all algorithms
- [ ] Add code examples
- [ ] Create troubleshooting guide

**Deliverables**:
- Large-scale dataset
- Contributor-friendly project
- Comprehensive documentation

---

## Milestone 10: Testing, Debugging & Launch
**Goal**: Ensure production readiness and launch

### Step 10.1: Comprehensive Testing
- [ ] End-to-end testing (Playwright/Cypress)
- [ ] Load testing (Locust/k6)
- [ ] Security testing (OWASP)
- [ ] Accessibility testing
- [ ] Cross-browser testing

### Step 10.2: Debugging & Monitoring
- [ ] Add extensive logging
- [ ] Set up error tracking
- [ ] Create debug mode
- [ ] Build admin dashboard
- [ ] Add game replay feature

### Step 10.3: Launch Preparation
- [ ] Create landing page with features
- [ ] Write launch announcement
- [ ] Prepare social media content
- [ ] Set up analytics
- [ ] Create feedback mechanism

### Step 10.4: Go Live
- [ ] Final production deployment
- [ ] Monitor initial traffic
- [ ] Gather user feedback
- [ ] Fix critical bugs
- [ ] Plan next iterations

**Deliverables**:
- Production-ready application
- Public launch
- Active monitoring

---

## Technology Deep Dive

### Backend Tech Stack Justification

**Python + Flask**
- Rapid API development
- Excellent ML/data processing libraries
- Easy integration with C++

**C++ Modules**
- High-performance algorithms (decision tree, probability)
- Compiled for speed (10-100x faster than Python)
- Efficient memory usage

**Firebase Realtime Database**
- 1GB free tier
- Real-time sync capabilities
- No server maintenance
- JSON-based (matches our data structure)

**Redis/Upstash Cache**
- Free tier available
- Extremely fast read/write
- Reduces database load
- TTL support for session management

### Frontend Tech Stack Justification

**React + TypeScript**
- Type safety reduces bugs
- Large ecosystem
- Excellent developer experience
- Easy to maintain

**Tailwind CSS**
- Rapid UI development
- Small bundle size
- Consistent design system
- Great documentation

**Vite**
- Lightning-fast dev server
- Optimized production builds
- Modern ES modules

---

## Modular Development Strategy

Each component is designed to be **independently developable**:

### For AI-Assisted Development
When working with Claude or other AI:
- Provide only the specific module code
- Include the interface/API contract
- Share relevant type definitions
- Include test cases

**Example**: To improve question selection algorithm
```
Files needed:
- backend/algorithms/python/question_selector.py
- backend/app/models/question.py
- backend/tests/test_question_selector.py
```

### Module Independence
- **Frontend**: Communicates via REST API only
- **Backend Routes**: Call services, no business logic
- **Services**: Pure business logic, no HTTP concerns
- **Algorithms**: Pure functions, no external dependencies
- **Data**: Static JSON, versioned separately

---

## Development Workflow

### For Individual Contributors
1. Pick a specific milestone step
2. Create feature branch
3. Work on that module only
4. Run module-specific tests
5. Submit PR with clear scope
6. CI/CD runs full test suite
7. Merge after review

### For You (Project Lead)
1. Work through milestones sequentially
2. Use github.dev or VSCode
3. Test locally with Docker Compose
4. Push to GitHub
5. Automatic deployment via CI/CD
6. Monitor Space/Vercel dashboards

---

## Important Notes

### Hugging Face Space Persistence
⚠️ **Space sleep = data loss**
- Use Firebase for ALL persistent data
- Session data must sync to Firebase
- Game statistics stored externally
- ML models versioned in Git LFS

### Data Management
- All data changes via Git commits
- Manual editing in `data/` folder
- Version control tracks all changes
- Automated validation on PR

### Scaling Strategy
- Start simple, iterate
- Add ML only when needed
- Optimize after measuring
- Keep modules independent

---

## Success Metrics

- [ ] 95%+ uptime
- [ ] <500ms average response time
- [ ] 80%+ guess accuracy
- [ ] 10+ contributors
- [ ] 1000+ places in database
- [ ] Mobile-responsive
- [ ] Accessible (WCAG AA)

---

## Future Enhancements (Post-Launch)

- Multiplayer mode
- Custom place categories
- User-submitted places
- Mobile apps (React Native)
- API for third-party integration
- Advanced statistics & analytics
- Achievement system
- Social sharing

---

## Getting Started

### First Steps
1. Complete Milestone 1 (Setup)
2. Read all documentation
3. Set up local environment
4. Start with data layer (Milestone 2)
5. Build incrementally

### Daily Development
```bash
# Pull latest
git pull origin main

# Create feature branch
git checkout -b feature/your-feature

# Run locally
docker-compose up

# Run tests
./scripts/test.sh

# Commit & push
git add .
git commit -m "feat: description"
git push origin feature/your-feature
```

---

## Resources

- **Akinator Algorithm**: Information theory, decision trees
- **Firebase Docs**: https://firebase.google.com/docs/database
- **Hugging Face Spaces**: https://huggingface.co/docs/hub/spaces
- **Docker Best Practices**: https://docs.docker.com/develop/dev-best-practices/
- **GitHub Actions**: https://docs.github.com/en/actions

---

**License**: GPL v3  
**Maintainer**: You  
**Status**: In Active Development  
**Last Updated**: February 2026

Let's build something amazing! 🚀
