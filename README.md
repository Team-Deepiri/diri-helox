# Deepiri - AI-Powered Digital Productivity Playground

Welcome to Deepiri, your AI-powered digital productivity playground that gamifies your tasks, study material, coding projects, and creative work into engaging mini-games, challenges, and interactive experiences.

## Core Concept

Gamify your productivity by turning tasks, study material, coding projects, or creative work into mini-games, challenges, and interactive experiences. Rewards, progress tracking, and adaptive difficulty make boring tasks fun. AI adapts challenges to your workflow and style, with optional multiplayer/competitive features.

## Target Users

- Students, knowledge workers, creatives, developers
- Anyone looking to increase productivity in a fun, engaging way
- Users who enjoy gamification, streaks, badges, and challenges

## Key Features

### Task Gamification
- Input tasks manually, via integrations (Trello, Notion, GitHub, Google Docs), or via AI scan of documents/code
- Convert tasks into mini-games (e.g., puzzle, quiz, coding challenge, timed completion challenge)

### Adaptive AI Challenges
- AI analyzes user behavior, performance, and preferences
- Generates adaptive challenges, e.g., "You typically code slowly in the morning — here's a short 15-min coding sprint."

### Rewards & Progression
- Points, badges, streaks, leaderboards
- Optional peer/competitor comparison

### Content/Task Integration
- Supports different media types: text documents, spreadsheets, PDFs, code repos, notes
- Can ingest multiple task types at once and turn them into daily "missions."

### Analytics & Insights
- Tracks efficiency, time management, and improvement trends
- Suggests optimized schedules or break timing

### Optional Multiplayer
- Challenges friends or coworkers to productivity duels
- Shared missions and collaborative mini-games

## Architecture

Deepiri follows a modern microservices architecture:

### Backend Services

- **User Service**: Authentication, profiles, progress, preferences
- **Task Service**: CRUD tasks, fetch tasks from integrations, store metadata
- **AI Challenge Service**: Generates challenges from tasks using NLP + RL models
- **Gamification Service**: Points, badges, leaderboards, streaks
- **Analytics Service**: Tracks performance, creates insights, suggests optimizations
- **Notification Service**: Sends reminders, daily missions, streak updates
- **Integration Service**: Connects to Notion, Trello, Google Docs, GitHub for task ingestion

### Python Agent (FastAPI)
- **AI Challenge Generation**: NLP models for task parsing and challenge creation
- **Reinforcement Learning**: Adaptive difficulty optimization
- **Task Understanding**: Classifies and categorizes tasks

### Frontend (React + Vite)
- **Modern React 18**: Hooks, Context API, and modern patterns
- **Responsive Design**: Mobile-first design with Tailwind CSS
- **Real-time Updates**: Socket.IO integration for live progress
- **Progressive Web App**: Offline support and mobile optimization

### Database & Infrastructure
- **MongoDB**: Primary database for all application data
- **Redis**: Caching layer and leaderboard storage
- **Docker**: Containerized deployment
- **NGINX**: Load balancing and reverse proxy

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for local development)
- Git

### Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd deepiri
   ```

2. **Run the setup script**
   ```bash
   ./scripts/setup.sh
   ```

3. **Configure environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Start the application**
   ```bash
   # Normal start (uses existing images, no rebuild)
   docker compose -f docker-compose.dev.yml up -d
   
   # To rebuild (only when needed - removes old images, rebuilds fresh)
   ./rebuild.sh  # Linux/Mac
   .\rebuild.ps1 # Windows PowerShell
   ```

5. **Access the application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:5000
   - Python Agent: http://localhost:8000
   - MLflow: http://localhost:5500
   - API Documentation: http://localhost:5000/api-docs

### Rebuilding Containers (Only When Needed)

**Normal operation:** `docker compose up` uses existing images - no rebuild.

**When to rebuild:** Only when code changes or you want fresh images:

```bash
# Use the clean rebuild scripts (removes old images first)
./rebuild.sh              # Linux/Mac
.\rebuild.ps1             # Windows PowerShell

# Or manually rebuild specific service
docker compose -f docker-compose.dev.yml build --no-cache pyagent
docker compose -f docker-compose.dev.yml up -d pyagent

# Or rebuild all (manual)
docker compose -f docker-compose.dev.yml down --rmi local
docker builder prune -af
docker compose -f docker-compose.dev.yml build --no-cache
docker compose -f docker-compose.dev.yml up -d
```

**Why use rebuild scripts?** They prevent Docker from accumulating 50GB+ of old images. See `DOCKER-IMAGE-CLEANSING-COMMANDS.md` for details.

### Local Development

1. **Install dependencies**
   ```bash
   npm run setup
   ```

2. **Start MongoDB and Redis**
   ```bash
   docker-compose up -d mongodb redis
   ```

3. **Start the development servers**
   ```bash
   npm run dev
   ```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure the following:

#### Required API Keys
```bash
# OpenAI for AI-powered challenge generation
OPENAI_API_KEY=your_openai_api_key

# Integration APIs (optional for Phase 1)
NOTION_API_KEY=your_notion_api_key
TRELLO_API_KEY=your_trello_api_key
GITHUB_TOKEN=your_github_token
```

#### Firebase Configuration
```bash
# Firebase for authentication and notifications
VITE_FIREBASE_API_KEY=your_firebase_api_key
VITE_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your-project-id
# ... other Firebase config
```

#### Database Configuration
```bash
# MongoDB
MONGO_ROOT_USER=admin
MONGO_ROOT_PASSWORD=your_secure_password
MONGO_DB=deepiri

# Redis
REDIS_PASSWORD=your_redis_password

# JWT Secret
JWT_SECRET=your_jwt_secret_key
```

## API Documentation

The API is fully documented with Swagger/OpenAPI. Once the application is running, visit:
- **Swagger UI**: http://localhost:5000/api-docs

### Main API Endpoints

#### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user

#### Tasks
- `POST /api/tasks` - Create new task
- `GET /api/tasks` - Get user's tasks
- `GET /api/tasks/:id` - Get task details
- `PATCH /api/tasks/:id` - Update task
- `DELETE /api/tasks/:id` - Delete task

#### Challenges
- `POST /api/challenges/generate` - Generate challenge from task
- `GET /api/challenges` - Get user's challenges
- `POST /api/challenges/:id/complete` - Complete challenge

#### Gamification
- `GET /api/gamification/profile` - Get user progress
- `GET /api/gamification/leaderboard` - Get leaderboard
- `GET /api/gamification/badges` - Get user badges

#### Integrations
- `POST /api/integrations/connect` - Connect external service
- `GET /api/integrations/sync` - Sync tasks from integration

## Testing

### Backend Tests (Node)
```bash
cd api-server
npm test
```

### Python Agent Tests
```bash
cd python_backend
pytest -q
```

### Frontend Tests
```bash
cd frontend
npm test
```

## Deployment

### Production Deployment with Docker

1. **Set production environment**
   ```bash
   export NODE_ENV=production
   ```

2. **Build and start with production profile**
   ```bash
   docker-compose --profile production up -d
   ```

3. **Enable NGINX reverse proxy**
   The production setup includes NGINX for:
   - SSL termination
   - Load balancing
   - Static file serving
   - API proxying

### Cloud Deployment

The application is designed to be cloud-native and can be deployed on:
- **AWS**: ECS, EKS, or EC2 with Docker
- **Google Cloud**: GKE or Cloud Run
- **Azure**: AKS or Container Instances
- **DigitalOcean**: App Platform or Kubernetes

## Development

### Adding New Features

1. **Backend Feature**
   - Add model in `api-server/models/`
   - Create service in `api-server/services/`
   - Add routes in `api-server/routes/`
   - Update controllers in `api-server/controllers/`

2. **Frontend Feature**
   - Create components in `frontend/src/components/`
   - Add pages in `frontend/src/pages/`
   - Update API layer in `frontend/src/api/`
   - Add routing to `frontend/src/App.jsx`

### Code Style

- **Backend**: ESLint + Prettier
- **Frontend**: ESLint + Prettier
- **Git Hooks**: Husky for pre-commit checks

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `npm test`
5. Commit your changes: `git commit -m "Add feature"`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Documentation

### Complete Documentation Index
See **[DOCUMENTATION-INDEX.md](DOCUMENTATION-INDEX.md)** for a complete guide to all 63+ markdown files in this project, organized by category.

### Quick Navigation
- **New to project?** → `GETTING_STARTED.md`
- **Setting up?** → `START_EVERYTHING.md`
- **Team member?** → `FIND_YOUR_TASKS.md`
- **Rebuilding?** → `rebuild.sh` / `rebuild.ps1` or `DOCKER-IMAGE-CLEANSING-COMMANDS.md`
- **Having issues?** → `docs/TROUBLESHOOTING.md`
- **Architecture?** → `docs/SYSTEM_ARCHITECTURE.md`
- **Contributing?** → `CONTRIBUTING.md`

### Team-Specific Guides
- **AI Team** → `docs/AI_TEAM_ONBOARDING.md`
- **Backend Team** → `docs/BACKEND_TEAM_ONBOARDING.md`
- **Frontend Team** → `docs/FRONTEND_TEAM_ONBOARDING.md`
- **ML Team** → `docs/ML_ENGINEER_COMPLETE_GUIDE.md`
- **ML Ops** → `docs/MLOPS_TEAM_ONBOARDING.md`
- **Platform Team** → `docs/PLATFORM_TEAM_ONBOARDING.md`
- **QA Team** → `docs/SECURITY_QA_TEAM_ONBOARDING.md`

## Support

### Common Issues

**Port already in use**
```bash
# Stop existing containers
docker-compose -f docker-compose.dev.yml down
# Or change ports in docker-compose.dev.yml
```

**API key errors**
- Ensure all required API keys are set in `.env`
- Check API key permissions and quotas

**Database connection issues**
- Ensure MongoDB is running: `docker-compose -f docker-compose.dev.yml up -d mongodb`
- Check database credentials in `.env`

**Docker storage bloat (50GB+ images)**
```bash
# Use clean rebuild script
./rebuild.sh  # Linux/Mac
.\rebuild.ps1 # Windows

# Or manually clean
docker-compose -f docker-compose.dev.yml down --rmi local
docker builder prune -af
```

See `DOCKER-IMAGE-CLEANSING-COMMANDS.md` for complete cleanup guide.

## Acknowledgments

- OpenAI for GPT-4 integration
- React and Node.js communities
- Hugging Face for transformer models

---
