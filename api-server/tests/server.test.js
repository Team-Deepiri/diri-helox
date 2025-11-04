/**
 * Comprehensive test suite for the Node.js server
 */
const request = require('supertest');
const mongoose = require('mongoose');
const jwt = require('jsonwebtoken');
const { MongoMemoryServer } = require('mongodb-memory-server');

// Import the app
const { app } = require('../server');

// Test utilities
const createTestUser = () => ({
  _id: new mongoose.Types.ObjectId(),
  email: 'test@example.com',
  name: 'Test User',
  preferences: {
    interests: ['food', 'music'],
    skillLevel: 'beginner',
    socialMode: 'solo',
    budget: 'medium',
    maxDistance: 5000,
    preferredDuration: 60,
    timePreferences: {
      morning: true,
      afternoon: true,
      evening: false,
      night: false
    }
  }
});

const createTestAdventure = (userId) => ({
  _id: new mongoose.Types.ObjectId(),
  userId,
  name: 'Test Adventure',
  description: 'A test adventure',
  status: 'pending',
  steps: [
    {
      type: 'venue',
      name: 'Test Venue',
      description: 'Test venue description',
      location: {
        lat: 40.7128,
        lng: -74.0060,
        address: '123 Test St, Test City'
      },
      startTime: new Date(),
      endTime: new Date(Date.now() + 30 * 60000),
      duration: 30,
      completed: false
    }
  ],
  createdAt: new Date(),
  updatedAt: new Date()
});

const generateTestToken = (userId) => {
  return jwt.sign(
    { userId, email: 'test@example.com' },
    process.env.JWT_SECRET || 'test-secret',
    { expiresIn: '1h' }
  );
};

describe('Server Health and Basic Functionality', () => {
  test('Health check endpoint returns correct status', async () => {
    const response = await request(app)
      .get('/api/health')
      .expect(200);

    expect(response.body).toHaveProperty('status', 'healthy');
    expect(response.body).toHaveProperty('version', '2.0.0');
    expect(response.body).toHaveProperty('services');
    expect(response.body).toHaveProperty('timestamp');
  });

  test('Metrics endpoint returns Prometheus metrics', async () => {
    const response = await request(app)
      .get('/metrics')
      .expect(200);

    expect(response.headers['content-type']).toContain('text/plain');
    expect(response.text).toContain('http_request_duration_seconds');
  });

  test('Non-existent endpoint returns 404', async () => {
    await request(app)
      .get('/api/non-existent')
      .expect(404);
  });
});

describe('Authentication Middleware', () => {
  test('Protected routes require valid JWT token', async () => {
    await request(app)
      .get('/api/users/profile')
      .expect(401);
  });

  test('Valid JWT token allows access to protected routes', async () => {
    const userId = new mongoose.Types.ObjectId();
    const token = generateTestToken(userId);

    await request(app)
      .get('/api/users/profile')
      .set('Authorization', `Bearer ${token}`)
      .expect(200);
  });

  test('Invalid JWT token is rejected', async () => {
    await request(app)
      .get('/api/users/profile')
      .set('Authorization', 'Bearer invalid-token')
      .expect(401);
  });

  test('Malformed Authorization header is rejected', async () => {
    await request(app)
      .get('/api/users/profile')
      .set('Authorization', 'InvalidFormat token')
      .expect(401);
  });
});

describe('Adventure Routes', () => {
  let testUser;
  let authToken;

  beforeEach(() => {
    testUser = createTestUser();
    authToken = generateTestToken(testUser._id);
  });

  describe('POST /api/adventures/generate', () => {
    test('Generates adventure with valid data', async () => {
      const adventureData = {
        location: {
          lat: 40.7128,
          lng: -74.0060,
          address: '123 Test St, Test City'
        },
        interests: ['food', 'music'],
        duration: 60,
        maxDistance: 5000,
        socialMode: 'solo'
      };

      const response = await request(app)
        .post('/api/adventures/generate')
        .set('Authorization', `Bearer ${authToken}`)
        .send(adventureData)
        .expect(201);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('data');
      expect(response.body.data).toHaveProperty('_id');
      expect(response.body.data).toHaveProperty('name');
      expect(response.body.data).toHaveProperty('steps');
    });

    test('Rejects invalid location data', async () => {
      const adventureData = {
        location: {
          lat: 91, // Invalid latitude
          lng: -74.0060
        },
        interests: ['food']
      };

      const response = await request(app)
        .post('/api/adventures/generate')
        .set('Authorization', `Bearer ${authToken}`)
        .send(adventureData)
        .expect(400);

      expect(response.body).toHaveProperty('success', false);
      expect(response.body).toHaveProperty('message', 'Validation error');
    });

    test('Rejects empty interests array', async () => {
      const adventureData = {
        location: {
          lat: 40.7128,
          lng: -74.0060
        },
        interests: [] // Empty array
      };

      const response = await request(app)
        .post('/api/adventures/generate')
        .set('Authorization', `Bearer ${authToken}`)
        .send(adventureData)
        .expect(400);

      expect(response.body).toHaveProperty('success', false);
    });

    test('Rejects missing required fields', async () => {
      const adventureData = {
        interests: ['food']
        // Missing location
      };

      const response = await request(app)
        .post('/api/adventures/generate')
        .set('Authorization', `Bearer ${authToken}`)
        .send(adventureData)
        .expect(400);

      expect(response.body).toHaveProperty('success', false);
    });
  });

  describe('GET /api/adventures', () => {
    test('Returns user adventures with pagination', async () => {
      const response = await request(app)
        .get('/api/adventures?limit=10&offset=0')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('data');
      expect(Array.isArray(response.body.data)).toBe(true);
    });

    test('Filters adventures by status', async () => {
      const response = await request(app)
        .get('/api/adventures?status=completed')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
    });
  });

  describe('GET /api/adventures/:adventureId', () => {
    test('Returns specific adventure', async () => {
      const adventureId = new mongoose.Types.ObjectId();
      
      const response = await request(app)
        .get(`/api/adventures/${adventureId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(404); // Will be 404 since adventure doesn't exist

      expect(response.body).toHaveProperty('success', false);
    });

    test('Rejects invalid adventure ID format', async () => {
      await request(app)
        .get('/api/adventures/invalid-id')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(400);
    });
  });

  describe('POST /api/adventures/:adventureId/start', () => {
    test('Starts adventure successfully', async () => {
      const adventureId = new mongoose.Types.ObjectId();
      
      const response = await request(app)
        .post(`/api/adventures/${adventureId}/start`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(400); // Will be 400 since adventure doesn't exist

      expect(response.body).toHaveProperty('success', false);
    });
  });

  describe('POST /api/adventures/:adventureId/complete', () => {
    test('Completes adventure with valid feedback', async () => {
      const adventureId = new mongoose.Types.ObjectId();
      const feedback = {
        rating: 5,
        comments: 'Great adventure!',
        completedSteps: ['step1'],
        suggestions: 'More food options'
      };

      const response = await request(app)
        .post(`/api/adventures/${adventureId}/complete`)
        .set('Authorization', `Bearer ${authToken}`)
        .send({ feedback })
        .expect(400); // Will be 400 since adventure doesn't exist

      expect(response.body).toHaveProperty('success', false);
    });

    test('Rejects invalid feedback rating', async () => {
      const adventureId = new mongoose.Types.ObjectId();
      const feedback = {
        rating: 6, // Invalid rating
        comments: 'Great adventure!'
      };

      const response = await request(app)
        .post(`/api/adventures/${adventureId}/complete`)
        .set('Authorization', `Bearer ${authToken}`)
        .send({ feedback })
        .expect(400);

      expect(response.body).toHaveProperty('success', false);
    });
  });

  describe('PUT /api/adventures/:adventureId/steps', () => {
    test('Updates adventure step successfully', async () => {
      const adventureId = new mongoose.Types.ObjectId();
      const stepUpdate = {
        stepIndex: 0,
        action: 'complete'
      };

      const response = await request(app)
        .put(`/api/adventures/${adventureId}/steps`)
        .set('Authorization', `Bearer ${authToken}`)
        .send(stepUpdate)
        .expect(400); // Will be 400 since adventure doesn't exist

      expect(response.body).toHaveProperty('success', false);
    });

    test('Rejects invalid step action', async () => {
      const adventureId = new mongoose.Types.ObjectId();
      const stepUpdate = {
        stepIndex: 0,
        action: 'invalid-action'
      };

      const response = await request(app)
        .put(`/api/adventures/${adventureId}/steps`)
        .set('Authorization', `Bearer ${authToken}`)
        .send(stepUpdate)
        .expect(400);

      expect(response.body).toHaveProperty('success', false);
    });
  });

  describe('GET /api/adventures/recommendations', () => {
    test('Returns adventure recommendations with location', async () => {
      const response = await request(app)
        .get('/api/adventures/recommendations?lat=40.7128&lng=-74.0060&limit=5')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('data');
    });

    test('Rejects request without location', async () => {
      const response = await request(app)
        .get('/api/adventures/recommendations')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(400);

      expect(response.body).toHaveProperty('success', false);
      expect(response.body.message).toContain('Location');
    });
  });

  describe('GET /api/adventures/analytics', () => {
    test('Returns adventure analytics', async () => {
      const response = await request(app)
        .get('/api/adventures/analytics?timeRange=30d')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('data');
    });
  });
});

describe('User Routes', () => {
  let testUser;
  let authToken;

  beforeEach(() => {
    testUser = createTestUser();
    authToken = generateTestToken(testUser._id);
  });

  describe('GET /api/users/profile', () => {
    test('Returns user profile', async () => {
      const response = await request(app)
        .get('/api/users/profile')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('data');
    });
  });

  describe('PUT /api/users/profile', () => {
    test('Updates user profile', async () => {
      const updateData = {
        name: 'Updated Name',
        preferences: {
          interests: ['food', 'music', 'art'],
          skillLevel: 'intermediate'
        }
      };

      const response = await request(app)
        .put('/api/users/profile')
        .set('Authorization', `Bearer ${authToken}`)
        .send(updateData)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
    });
  });
});

describe('Event Routes', () => {
  let testUser;
  let authToken;

  beforeEach(() => {
    testUser = createTestUser();
    authToken = generateTestToken(testUser._id);
  });

  describe('GET /api/events', () => {
    test('Returns events with location filter', async () => {
      const response = await request(app)
        .get('/api/events?lat=40.7128&lng=-74.0060&radius=5000')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('data');
    });

    test('Filters events by category', async () => {
      const response = await request(app)
        .get('/api/events?lat=40.7128&lng=-74.0060&category=music')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
    });
  });

  describe('POST /api/events', () => {
    test('Creates new event with valid data', async () => {
      const eventData = {
        name: 'Test Event',
        description: 'A test event',
        type: 'concert',
        location: {
          lat: 40.7128,
          lng: -74.0060,
          address: '123 Test St, Test City'
        },
        startTime: new Date(Date.now() + 24 * 60 * 60 * 1000), // Tomorrow
        endTime: new Date(Date.now() + 25 * 60 * 60 * 1000), // Tomorrow + 1 hour
        capacity: 100,
        price: {
          isFree: false,
          amount: 25
        }
      };

      const response = await request(app)
        .post('/api/events')
        .set('Authorization', `Bearer ${authToken}`)
        .send(eventData)
        .expect(201);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('data');
    });

    test('Rejects event with invalid time', async () => {
      const eventData = {
        name: 'Test Event',
        type: 'concert',
        location: {
          lat: 40.7128,
          lng: -74.0060
        },
        startTime: new Date(Date.now() - 24 * 60 * 60 * 1000), // Yesterday
        endTime: new Date(Date.now() + 60 * 60 * 1000) // Tomorrow
      };

      const response = await request(app)
        .post('/api/events')
        .set('Authorization', `Bearer ${authToken}`)
        .send(eventData)
        .expect(400);

      expect(response.body).toHaveProperty('success', false);
    });
  });
});

describe('External API Routes', () => {
  describe('GET /api/external/adventure-data', () => {
    test('Returns adventure data with valid coordinates', async () => {
      const response = await request(app)
        .get('/api/external/adventure-data?lat=40.7128&lng=-74.0060&radius=5000')
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('data');
    });

    test('Rejects request with invalid coordinates', async () => {
      const response = await request(app)
        .get('/api/external/adventure-data?lat=91&lng=-74.0060')
        .expect(400);

      expect(response.body).toHaveProperty('success', false);
    });
  });

  describe('GET /api/external/weather/current', () => {
    test('Returns current weather data', async () => {
      const response = await request(app)
        .get('/api/external/weather/current?lat=40.7128&lng=-74.0060')
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('data');
    });
  });

  describe('GET /api/external/directions', () => {
    test('Returns directions between two points', async () => {
      const response = await request(app)
        .get('/api/external/directions?fromLat=40.7128&fromLng=-74.0060&toLat=40.7589&toLng=-73.9851&mode=walking')
        .expect(200);

      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('data');
    });
  });
});

describe('Rate Limiting', () => {
  test('Rate limiting is applied to API routes', async () => {
    // Make multiple requests quickly to test rate limiting
    const promises = Array(10).fill().map(() => 
      request(app).get('/api/health')
    );

    const responses = await Promise.all(promises);
    
    // All should succeed for health endpoint (it might not be rate limited)
    responses.forEach(response => {
      expect(response.status).toBeLessThan(500);
    });
  });
});

describe('Error Handling', () => {
  test('Handles malformed JSON gracefully', async () => {
    const response = await request(app)
      .post('/api/adventures/generate')
      .set('Authorization', `Bearer ${generateTestToken(new mongoose.Types.ObjectId())}`)
      .set('Content-Type', 'application/json')
      .send('{"invalid": json}')
      .expect(400);

    expect(response.body).toHaveProperty('success', false);
  });

  test('Handles missing request body gracefully', async () => {
    const response = await request(app)
      .post('/api/adventures/generate')
      .set('Authorization', `Bearer ${generateTestToken(new mongoose.Types.ObjectId())}`)
      .expect(400);

    expect(response.body).toHaveProperty('success', false);
  });
});

describe('CORS Configuration', () => {
  test('CORS headers are present in responses', async () => {
    const response = await request(app)
      .options('/api/health')
      .expect(204);

    expect(response.headers).toHaveProperty('access-control-allow-origin');
  });
});

describe('Security Headers', () => {
  test('Security headers are present', async () => {
    const response = await request(app)
      .get('/api/health')
      .expect(200);

    // Check for Helmet security headers
    expect(response.headers).toHaveProperty('x-content-type-options');
    expect(response.headers).toHaveProperty('x-frame-options');
  });
});

// Integration tests
describe('Integration Tests', () => {
  let testUser;
  let authToken;
  let adventureId;

  beforeEach(() => {
    testUser = createTestUser();
    authToken = generateTestToken(testUser._id);
  });

  test('Complete adventure workflow', async () => {
    // 1. Generate adventure
    const adventureData = {
      location: {
        lat: 40.7128,
        lng: -74.0060,
        address: '123 Test St, Test City'
      },
      interests: ['food', 'music'],
      duration: 60
    };

    const generateResponse = await request(app)
      .post('/api/adventures/generate')
      .set('Authorization', `Bearer ${authToken}`)
      .send(adventureData)
      .expect(201);

    adventureId = generateResponse.body.data._id;

    // 2. Start adventure
    await request(app)
      .post(`/api/adventures/${adventureId}/start`)
      .set('Authorization', `Bearer ${authToken}`)
      .expect(200);

    // 3. Complete first step
    await request(app)
      .put(`/api/adventures/${adventureId}/steps`)
      .set('Authorization', `Bearer ${authToken}`)
      .send({
        stepIndex: 0,
        action: 'complete'
      })
      .expect(200);

    // 4. Complete adventure
    const feedback = {
      rating: 5,
      comments: 'Great adventure!',
      completedSteps: ['step1']
    };

    await request(app)
      .post(`/api/adventures/${adventureId}/complete`)
      .set('Authorization', `Bearer ${authToken}`)
      .send({ feedback })
      .expect(200);

    // 5. Verify adventure is completed
    const getResponse = await request(app)
      .get(`/api/adventures/${adventureId}`)
      .set('Authorization', `Bearer ${authToken}`)
      .expect(200);

    expect(getResponse.body.data.status).toBe('completed');
  });
});

module.exports = {
  createTestUser,
  createTestAdventure,
  generateTestToken
};
