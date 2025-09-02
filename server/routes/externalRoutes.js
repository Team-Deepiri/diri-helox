const express = require('express');
const externalApiService = require('../services/externalApiService');
const logger = require('../utils/logger');

const router = express.Router();

// Get nearby places
router.get('/places/nearby', async (req, res) => {
  try {
    const { lat, lng, radius = 5000, type = 'establishment', keyword } = req.query;

    if (!lat || !lng) {
      return res.status(400).json({
        success: false,
        message: 'Latitude and longitude are required'
      });
    }

    const location = { lat: parseFloat(lat), lng: parseFloat(lng) };
    const places = await externalApiService.getNearbyPlaces(
      location,
      parseInt(radius),
      type,
      keyword
    );

    res.json({
      success: true,
      data: places
    });

  } catch (error) {
    logger.error('Failed to get nearby places:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get place details
router.get('/places/:placeId', async (req, res) => {
  try {
    const { placeId } = req.params;
    const details = await externalApiService.getPlaceDetails(placeId);

    res.json({
      success: true,
      data: details
    });

  } catch (error) {
    logger.error('Failed to get place details:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get directions
router.get('/directions', async (req, res) => {
  try {
    const { fromLat, fromLng, toLat, toLng, mode = 'walking' } = req.query;

    if (!fromLat || !fromLng || !toLat || !toLng) {
      return res.status(400).json({
        success: false,
        message: 'Origin and destination coordinates are required'
      });
    }

    const origin = { lat: parseFloat(fromLat), lng: parseFloat(fromLng) };
    const destination = { lat: parseFloat(toLat), lng: parseFloat(toLng) };

    const directions = await externalApiService.getDirections(origin, destination, mode);

    res.json({
      success: true,
      data: directions
    });

  } catch (error) {
    logger.error('Failed to get directions:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get current weather
router.get('/weather/current', async (req, res) => {
  try {
    const { lat, lng } = req.query;

    if (!lat || !lng) {
      return res.status(400).json({
        success: false,
        message: 'Latitude and longitude are required'
      });
    }

    const location = { lat: parseFloat(lat), lng: parseFloat(lng) };
    const weather = await externalApiService.getCurrentWeather(location);

    res.json({
      success: true,
      data: weather
    });

  } catch (error) {
    logger.error('Failed to get current weather:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get weather forecast
router.get('/weather/forecast', async (req, res) => {
  try {
    const { lat, lng, days = 5 } = req.query;

    if (!lat || !lng) {
      return res.status(400).json({
        success: false,
        message: 'Latitude and longitude are required'
      });
    }

    const location = { lat: parseFloat(lat), lng: parseFloat(lng) };
    const forecast = await externalApiService.getWeatherForecast(location, parseInt(days));

    res.json({
      success: true,
      data: forecast
    });

  } catch (error) {
    logger.error('Failed to get weather forecast:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get nearby events
router.get('/events/nearby', async (req, res) => {
  try {
    const { lat, lng, radius = 5000, category } = req.query;

    if (!lat || !lng) {
      return res.status(400).json({
        success: false,
        message: 'Latitude and longitude are required'
      });
    }

    const location = { lat: parseFloat(lat), lng: parseFloat(lng) };
    const events = await externalApiService.getNearbyEvents(
      location,
      parseInt(radius),
      category
    );

    res.json({
      success: true,
      data: events
    });

  } catch (error) {
    logger.error('Failed to get nearby events:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get nearby businesses
router.get('/businesses/nearby', async (req, res) => {
  try {
    const { lat, lng, radius = 5000, category, limit = 20 } = req.query;

    if (!lat || !lng) {
      return res.status(400).json({
        success: false,
        message: 'Latitude and longitude are required'
      });
    }

    const location = { lat: parseFloat(lat), lng: parseFloat(lng) };
    const businesses = await externalApiService.getNearbyBusinesses(
      location,
      parseInt(radius),
      category,
      parseInt(limit)
    );

    res.json({
      success: true,
      data: businesses
    });

  } catch (error) {
    logger.error('Failed to get nearby businesses:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Geocode address
router.get('/geocode', async (req, res) => {
  try {
    const { address } = req.query;

    if (!address) {
      return res.status(400).json({
        success: false,
        message: 'Address is required'
      });
    }

    const location = await externalApiService.geocodeAddress(address);

    res.json({
      success: true,
      data: location
    });

  } catch (error) {
    logger.error('Failed to geocode address:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Reverse geocode
router.get('/reverse-geocode', async (req, res) => {
  try {
    const { lat, lng } = req.query;

    if (!lat || !lng) {
      return res.status(400).json({
        success: false,
        message: 'Latitude and longitude are required'
      });
    }

    const location = { lat: parseFloat(lat), lng: parseFloat(lng) };
    const address = await externalApiService.reverseGeocode(location);

    res.json({
      success: true,
      data: address
    });

  } catch (error) {
    logger.error('Failed to reverse geocode:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get combined adventure data
router.get('/adventure-data', async (req, res) => {
  try {
    const { lat, lng, radius = 5000, interests } = req.query;

    if (!lat || !lng) {
      return res.status(400).json({
        success: false,
        message: 'Latitude and longitude are required'
      });
    }

    const location = { lat: parseFloat(lat), lng: parseFloat(lng) };
    const interestsArray = interests ? interests.split(',') : [];
    
    const data = await externalApiService.getAdventureData(
      location,
      parseInt(radius),
      interestsArray
    );

    res.json({
      success: true,
      data: data
    });

  } catch (error) {
    logger.error('Failed to get adventure data:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

module.exports = router;
