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

export const notificationApi = {
  getNotifications: async (limit = 50, offset = 0, unreadOnly = false) => {
    try {
      const response = await api.get('/notifications', {
        params: { limit, offset, unreadOnly }
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  markAsRead: async (notificationId) => {
    try {
      const response = await api.put(`/notifications/${notificationId}/read`);
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  markAllAsRead: async () => {
    try {
      const response = await api.put('/notifications/read-all');
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  deleteNotification: async (notificationId) => {
    try {
      const response = await api.delete(`/notifications/${notificationId}`);
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  getSettings: async () => {
    try {
      const response = await api.get('/notifications/settings');
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  updateSettings: async (settings) => {
    try {
      const response = await api.put('/notifications/settings', settings);
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  getUnreadCount: async () => {
    try {
      const response = await api.get('/notifications/unread/count');
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  }
};
