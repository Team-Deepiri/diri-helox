const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const helmet = require('helmet');
const dotenv = require('dotenv');
const winston = require('winston');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5006;

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
  .then(() => logger.info('Integration Service: Connected to MongoDB'))
  .catch(err => logger.error('Integration Service: MongoDB connection error', err));

const routes = require('./src/index');

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', service: 'integration-service', timestamp: new Date().toISOString() });
});

app.use('/', routes);

app.use((err, req, res, next) => {
  logger.error('Integration Service error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

app.listen(PORT, () => {
  logger.info(`Integration Service running on port ${PORT}`);
});

module.exports = app;

