const express = require('express');
const router = express.Router();

const multiCurrencyService = require('./multiCurrencyService');
const badgeSystemService = require('./badgeSystemService');
const eloLeaderboardService = require('./eloLeaderboardService');

const currency = multiCurrencyService.MultiCurrencyService 
  ? new multiCurrencyService.MultiCurrencyService() 
  : multiCurrencyService;
const badges = badgeSystemService.BadgeSystemService
  ? new badgeSystemService.BadgeSystemService()
  : badgeSystemService;
const leaderboard = eloLeaderboardService.ELOLeaderboardService
  ? new eloLeaderboardService.ELOLeaderboardService()
  : eloLeaderboardService;

// Currency routes
router.post('/currency/award', (req, res) => currency.awardPoints(req, res));
router.get('/currency/:userId', (req, res) => currency.getBalance(req, res));

// Badge routes
router.get('/badges/:userId', (req, res) => badges.getBadges(req, res));
router.post('/badges/award', (req, res) => badges.awardBadge(req, res));

// Leaderboard routes
router.get('/leaderboard', (req, res) => leaderboard.getLeaderboard(req, res));
router.post('/leaderboard/update', (req, res) => leaderboard.updateRating(req, res));

module.exports = router;

