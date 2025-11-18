/**
 * Shared Utilities for Deepiri Microservices
 * Export all shared utilities from this module
 */
const { createLogger } = require('./logger');

module.exports = {
  createLogger,
  logger: createLogger('shared-utils') // Default logger instance
};

