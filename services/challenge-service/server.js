const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const helmet = require('helmet');
const dotenv = require('dotenv');
const winston = require('winston');
const axios = require('axios');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5007;

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  transports: [new winston.transports.Console({ format: winston.format.simple() })]
});

app.use(helmet());
app.use(cors());
app.use(express.json());

const MONGO_URI = process.env.MONGO_URI || 'mongodb://mongodb:27017/deepiri';
mongoose.connect(MONGO_URI)
  .then(() => logger.info('Challenge Service: Connected to MongoDB'))
  .catch(err => logger.error('Challenge Service: MongoDB connection error', err));

const CYREX_URL = process.env.CYREX_URL || 'http://cyrex:8000';

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', service: 'challenge-service', timestamp: new Date().toISOString() });
});

// Challenge generation (calls AI service)
app.post('/generate', async (req, res) => {
  try {
    const response = await axios.post(`${CYREX_URL}/agent/challenge/generate`, req.body);
    res.json(response.data);
  } catch (error) {
    logger.error('Challenge generation error:', error);
    res.status(500).json({ error: 'Failed to generate challenge' });
  }
});

app.use((err, req, res, next) => {
  logger.error('Challenge Service error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

app.listen(PORT, () => {
  logger.info(`Challenge Service running on port ${PORT}`);
});

module.exports = app;

