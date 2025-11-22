# Core API Status and Frontend Team Access

## Quick Answers

### 1. Do we need to give the frontend team the core-api?

**NO** ❌ - The frontend team does **NOT** need access to `deepiri-core-api`.

**Why:**
- Frontend connects to the **API Gateway** at `http://localhost:5000/api`
- The API Gateway routes all requests to the new microservices
- Core-api is deprecated and being replaced by microservices
- Frontend only needs to know the API Gateway endpoints

### 2. Is the core-api just old or is it still important?

**It's DEPRECATED** ⚠️ - Core-api is a legacy monolith being actively migrated to microservices.

**Status:**
- ✅ **Current Architecture:** Microservices in `platform-services/backend/`
- ❌ **Legacy:** `deepiri-core-api/` is the old monolith
- 🔄 **Migration:** Services are being extracted from core-api to microservices
- 📅 **Timeline:** Core-api will be removed once migration is complete

---

## Architecture Overview

### Current (Microservices) ✅
```
Frontend (5173)
    ↓
API Gateway (5000)
    ↓
┌─────────────────────────────────────┐
│  Microservices (platform-services)  │
├─────────────────────────────────────┤
│  • auth-service (5001)             │
│  • task-orchestrator (5002)         │
│  • engagement-service (5003)        │
│  • platform-analytics (5004)        │
│  • notification-service (5005)       │
│  • external-bridge (5006)           │
│  • challenge-service (5007)          │
│  • realtime-gateway (5008)          │
│  • cyrex (8000)                      │
└─────────────────────────────────────┘
```

### Legacy (Deprecated) ❌
```
Frontend
    ↓
deepiri-core-api (monolith)
    ↓
All services bundled together
```

---

## Frontend Team Requirements

### ✅ What Frontend Team Needs:

1. **API Gateway Access** (Required)
   - Service: `platform-services/backend/deepiri-api-gateway`
   - URL: `http://localhost:5000/api`
   - Purpose: All API calls go through this gateway

2. **Frontend Codebase** (Required)
   - Service: `deepiri-web-frontend`
   - Purpose: Their main work area

3. **API Documentation** (Recommended)
   - Service endpoint documentation
   - API Gateway routing rules
   - Request/response schemas

### ❌ What Frontend Team Does NOT Need:

1. **Core-API** - Deprecated, not used
2. **Individual Microservices** - They go through API Gateway
3. **Backend Implementation Details** - They only need API contracts

---

## Core-API Migration Status

### Services Already Migrated ✅

| Service | Old Location (core-api) | New Location (microservices) |
|---------|------------------------|------------------------------|
| Auth | `src/services/userService.ts` | `platform-services/backend/deepiri-auth-service/` |
| Tasks | `src/services/taskService.ts` | `platform-services/backend/deepiri-task-orchestrator/` |
| Challenges | `src/services/challengeService.ts` | `platform-services/backend/deepiri-challenge-service/` |
| Gamification | `src/services/gamificationService.ts` | `platform-services/backend/deepiri-engagement-service/` |
| Analytics | `src/services/analyticsService.ts` | `platform-services/backend/deepiri-platform-analytics-service/` |
| Integrations | `src/services/integrationService.ts` | `platform-services/backend/deepiri-external-bridge-service/` |
| Notifications | `src/services/notificationService.ts` | `platform-services/backend/deepiri-notification-service/` |
| WebSocket | `src/server.ts` (Socket.IO) | `platform-services/backend/deepiri-realtime-gateway/` |

### Still in Core-API (if any) ⚠️

- Any remaining legacy code
- Migration utilities
- Old database migrations
- Legacy test files

---

## Who Needs Core-API Access?

### ✅ Teams That May Need It:

1. **Backend Team** (Migration work)
   - Extracting services from monolith
   - Understanding legacy patterns
   - Migration planning

2. **Infrastructure Team** (Reference)
   - Understanding old deployment patterns
   - Migration tooling
   - Legacy system support

3. **Platform Engineers** (Reference)
   - Migration standards
   - Platform tooling
   - Architecture documentation

### ❌ Teams That Do NOT Need It:

1. **Frontend Team** ❌
   - Uses API Gateway only
   - No direct backend access needed

2. **AI/ML Team** ❌
   - Works with `diri-cyrex` service
   - No core-api dependencies

3. **QA Team** ❌
   - Tests against API Gateway
   - No need for legacy code

---

## Frontend API Integration

### Current Frontend Setup:

```typescript
// Frontend uses API Gateway
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

// All API calls go through gateway
axios.get(`${API_BASE_URL}/auth/login`)
axios.get(`${API_BASE_URL}/tasks`)
axios.get(`${API_BASE_URL}/challenges`)
```

### API Gateway Routes:

The API Gateway (`platform-services/backend/deepiri-api-gateway`) handles:
- `/api/auth/*` → `auth-service:5001`
- `/api/tasks/*` → `task-orchestrator:5002`
- `/api/engagement/*` → `engagement-service:5003`
- `/api/analytics/*` → `platform-analytics-service:5004`
- `/api/notifications/*` → `notification-service:5005`
- `/api/integrations/*` → `external-bridge-service:5006`
- `/api/challenges/*` → `challenge-service:5007`
- `/api/realtime/*` → `realtime-gateway:5008`
- `/api/cyrex/*` → `cyrex:8000`

---

## Recommendations

### For Frontend Team:

1. **Use API Gateway Only**
   - All requests to `http://localhost:5000/api`
   - Don't call microservices directly
   - Gateway handles routing, auth, rate limiting

2. **API Documentation**
   - Request API endpoint documentation
   - Get OpenAPI/Swagger specs if available
   - Understand request/response formats

3. **No Core-API Access Needed**
   - Core-api is internal implementation detail
   - Frontend doesn't need to see backend code
   - API Gateway provides clean interface

### For Project Management:

1. **Complete Migration**
   - Finish extracting all services from core-api
   - Remove core-api once migration is done
   - Update documentation to remove core-api references

2. **Clear API Contracts**
   - Document all API Gateway endpoints
   - Provide OpenAPI/Swagger specs
   - Maintain backward compatibility during migration

---

## Summary

| Question | Answer |
|----------|--------|
| **Frontend needs core-api?** | ❌ NO - They use API Gateway only |
| **Core-api still important?** | ⚠️ DEPRECATED - Being migrated to microservices |
| **What frontend needs?** | ✅ API Gateway access + API documentation |
| **Who needs core-api?** | Backend team (migration), Infrastructure (reference) |

---

**Last Updated:** 2025-11-22

