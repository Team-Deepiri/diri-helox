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

export const externalApi = {
  getNearbyPlaces: async (location, radius = 5000, type = 'establishment', keyword = null) => {
    try {
      const params = {
        lat: location.lat,
        lng: location.lng,
        radius,
        type
      };
      if (keyword) params.keyword = keyword;
      
      const response = await api.get('/external/places/nearby', { params });
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  getPlaceDetails: async (placeId) => {
    try {
      const response = await api.get(`/external/places/${placeId}`);
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  getDirections: async (origin, destination, mode = 'walking') => {
    try {
      const response = await api.get('/external/directions', {
        params: {
          fromLat: origin.lat,
          fromLng: origin.lng,
          toLat: destination.lat,
          toLng: destination.lng,
          mode
        }
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  getCurrentWeather: async (location) => {
    try {
      const response = await api.get('/external/weather/current', {
        params: {
          lat: location.lat,
          lng: location.lng
        }
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  getWeatherForecast: async (location, days = 5) => {
    try {
      const response = await api.get('/external/weather/forecast', {
        params: {
          lat: location.lat,
          lng: location.lng,
          days
        }
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  getNearbyEvents: async (location, radius = 5000, category = null) => {
    try {
      const params = {
        lat: location.lat,
        lng: location.lng,
        radius
      };
      if (category) params.category = category;
      
      const response = await api.get('/external/events/nearby', { params });
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  getNearbyBusinesses: async (location, radius = 5000, category = null, limit = 20) => {
    try {
      const params = {
        lat: location.lat,
        lng: location.lng,
        radius,
        limit
      };
      if (category) params.category = category;
      
      const response = await api.get('/external/businesses/nearby', { params });
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  geocodeAddress: async (address) => {
    try {
      const response = await api.get('/external/geocode', {
        params: { address }
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  reverseGeocode: async (location) => {
    try {
      const response = await api.get('/external/reverse-geocode', {
        params: {
          lat: location.lat,
          lng: location.lng
        }
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  },

  getAdventureData: async (location, radius = 5000, interests = []) => {
    try {
      const response = await api.get('/external/adventure-data', {
        params: {
          lat: location.lat,
          lng: location.lng,
          radius,
          interests: interests.join(',')
        }
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || error;
    }
  }
};
