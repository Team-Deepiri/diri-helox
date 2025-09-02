import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

export const eventApi = {
  createEvent: async (eventData) => {
    try {
      const response = await api.post('/events', eventData);
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  getEvent: async (eventId) => {
    try {
      const response = await api.get(`/events/${eventId}`);
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  getNearbyEvents: async (location, filters = {}) => {
    try {
      const params = {
        lat: location.lat,
        lng: location.lng,
        ...filters
      };
      const response = await api.get('/events/nearby', { params });
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  updateEvent: async (eventId, eventData) => {
    try {
      const response = await api.put(`/events/${eventId}`, eventData);
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  joinEvent: async (eventId) => {
    try {
      const response = await api.post(`/events/${eventId}/join`);
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  leaveEvent: async (eventId) => {
    try {
      const response = await api.post(`/events/${eventId}/leave`);
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  reviewEvent: async (eventId, rating, comment) => {
    try {
      const response = await api.post(`/events/${eventId}/review`, {
        rating,
        comment
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  cancelEvent: async (eventId) => {
    try {
      const response = await api.post(`/events/${eventId}/cancel`);
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  getUserEvents: async (type = 'all', limit = 20, offset = 0) => {
    try {
      const response = await api.get('/events/user/events', {
        params: { type, limit, offset }
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  getEventCategories: async () => {
    try {
      const response = await api.get('/events/categories');
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  getEventAnalytics: async (eventId) => {
    try {
      const response = await api.get(`/events/${eventId}/analytics`);
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  getPopularEvents: async (location, limit = 10) => {
    try {
      const response = await api.get('/events/popular', {
        params: {
          lat: location.lat,
          lng: location.lng,
          limit
        }
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  getTrendingEvents: async (location, timeRange = '7d', limit = 10) => {
    try {
      const response = await api.get('/events/trending', {
        params: {
          lat: location.lat,
          lng: location.lng,
          timeRange,
          limit
        }
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  searchEvents: async (query, location, radius = 5000, limit = 20) => {
    try {
      const response = await api.get('/events/search', {
        params: {
          query,
          lat: location.lat,
          lng: location.lng,
          radius,
          limit
        }
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  }
};
