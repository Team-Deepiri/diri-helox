const express = require('express');
const router = express.Router();

const timeSeriesAnalytics = require('./timeSeriesAnalytics');
const behavioralClustering = require('./behavioralClustering');
const predictiveModeling = require('./predictiveModeling');

const timeSeries = timeSeriesAnalytics.TimeSeriesAnalytics
  ? new timeSeriesAnalytics.TimeSeriesAnalytics()
  : timeSeriesAnalytics;
const clustering = behavioralClustering.BehavioralClustering
  ? new behavioralClustering.BehavioralClustering()
  : behavioralClustering;
const predictive = predictiveModeling.PredictiveModeling
  ? new predictiveModeling.PredictiveModeling()
  : predictiveModeling;

// Time series routes
router.post('/time-series/record', (req, res) => timeSeries.recordData(req, res));
router.get('/time-series/:userId', (req, res) => timeSeries.getAnalytics(req, res));

// Clustering routes
router.post('/clustering/analyze', (req, res) => clustering.analyze(req, res));
router.get('/clustering/:userId/group', (req, res) => clustering.getUserGroup(req, res));

// Predictive routes
router.post('/predictive/forecast', (req, res) => predictive.forecast(req, res));
router.get('/predictive/:userId/recommendations', (req, res) => predictive.getRecommendations(req, res));

module.exports = router;

