const express = require('express');
const fs = require('fs');
const path = require('path');
const authenticateJWT = require('../middleware/authenticateJWT');
const logger = require('../utils/logger');
const router = express.Router();

// Determine logs directory (same logic as logger.js)
const isDocker = process.env.NODE_ENV === 'production' || process.env.DOCKER === 'true' || fs.existsSync('/.dockerenv');
const defaultLogsDir = isDocker ? '/app/logs' : path.join(process.cwd(), 'logs');
const envLogsDir = process.env.LOG_DIR;
let logsDir = defaultLogsDir;

if (envLogsDir && typeof envLogsDir === 'string') {
  if (path.isAbsolute(envLogsDir)) {
    if (isDocker && envLogsDir.startsWith('/app/')) {
      logsDir = envLogsDir;
    } else if (!isDocker) {
      logsDir = envLogsDir;
    }
  } else {
    logsDir = path.join(defaultLogsDir, envLogsDir);
  }
}

// Helper function to read log file safely
const readLogFile = (filePath, limit = 1000) => {
  try {
    if (!fs.existsSync(filePath)) {
      return { error: 'Log file not found' };
    }

    const stats = fs.statSync(filePath);
    const fileSize = stats.size;
    
    // If file is too large, read from the end
    let content = '';
    if (fileSize > 1024 * 1024) { // 1MB
      const fd = fs.openSync(filePath, 'r');
      const buffer = Buffer.alloc(1024 * 1024); // 1MB buffer
      const bytesRead = fs.readSync(fd, buffer, 0, buffer.length, fileSize - buffer.length);
      content = buffer.slice(0, bytesRead).toString('utf8');
      fs.closeSync(fd);
    } else {
      content = fs.readFileSync(filePath, 'utf8');
    }

    // Split into lines and limit
    const lines = content.split('\n').filter(line => line.trim());
    const limitedLines = lines.slice(-limit);
    
    return {
      lines: limitedLines,
      totalLines: lines.length,
      fileSize: fileSize,
      lastModified: stats.mtime
    };
  } catch (error) {
    logger.error('Error reading log file:', error);
    return { error: 'Failed to read log file' };
  }
};

// GET /api/logs - Get recent logs (still requires auth for reading)
router.get('/', authenticateJWT, (req, res) => {
  try {
    const { type = 'combined', limit = 100 } = req.query;
    
    // Validate log type
    const validTypes = ['combined', 'error'];
    if (!validTypes.includes(type)) {
      return res.status(400).json({
        success: false,
        message: 'Invalid log type. Must be "combined" or "error"',
        requestId: req.requestId,
        timestamp: new Date().toISOString()
      });
    }

    // Validate limit
    const parsedLimit = parseInt(limit);
    if (isNaN(parsedLimit) || parsedLimit < 1 || parsedLimit > 10000) {
      return res.status(400).json({
        success: false,
        message: 'Invalid limit. Must be between 1 and 10000',
        requestId: req.requestId,
        timestamp: new Date().toISOString()
      });
    }

    const logFile = path.join(logsDir, `${type}.log`);
    const result = readLogFile(logFile, parsedLimit);

    if (result.error) {
      return res.status(404).json({
        success: false,
        message: result.error,
        requestId: req.requestId,
        timestamp: new Date().toISOString()
      });
    }

    res.json({
      success: true,
      data: {
        type: type,
        lines: result.lines,
        totalLines: result.totalLines,
        fileSize: result.fileSize,
        lastModified: result.lastModified,
        limit: parsedLimit
      },
      requestId: req.requestId,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Error in logs route:', error);
    res.status(500).json({
      success: false,
      message: 'Internal server error',
      requestId: req.requestId,
      timestamp: new Date().toISOString()
    });
  }
});

// GET /api/logs/files - List available log files
router.get('/files', authenticateJWT, (req, res) => {
  try {
    const files = [];
    
    if (fs.existsSync(logsDir)) {
      const logFiles = fs.readdirSync(logsDir).filter(file => file.endsWith('.log'));
      
      for (const file of logFiles) {
        const filePath = path.join(logsDir, file);
        const stats = fs.statSync(filePath);
        
        files.push({
          name: file,
          size: stats.size,
          lastModified: stats.mtime,
          type: file.replace('.log', '')
        });
      }
    }

    res.json({
      success: true,
      data: {
        files: files,
        logsDirectory: logsDir
      },
      requestId: req.requestId,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Error listing log files:', error);
    res.status(500).json({
      success: false,
      message: 'Internal server error',
      requestId: req.requestId,
      timestamp: new Date().toISOString()
    });
  }
});

// GET /api/logs/search - Search logs by keyword
router.get('/search', authenticateJWT, (req, res) => {
  try {
    const { q: query, type = 'combined', limit = 100 } = req.query;
    
    if (!query || query.trim().length === 0) {
      return res.status(400).json({
        success: false,
        message: 'Search query is required',
        requestId: req.requestId,
        timestamp: new Date().toISOString()
      });
    }

    const validTypes = ['combined', 'error'];
    if (!validTypes.includes(type)) {
      return res.status(400).json({
        success: false,
        message: 'Invalid log type. Must be "combined" or "error"',
        requestId: req.requestId,
        timestamp: new Date().toISOString()
      });
    }

    const parsedLimit = parseInt(limit);
    if (isNaN(parsedLimit) || parsedLimit < 1 || parsedLimit > 1000) {
      return res.status(400).json({
        success: false,
        message: 'Invalid limit. Must be between 1 and 1000',
        requestId: req.requestId,
        timestamp: new Date().toISOString()
      });
    }

    const logFile = path.join(logsDir, `${type}.log`);
    
    if (!fs.existsSync(logFile)) {
      return res.status(404).json({
        success: false,
        message: 'Log file not found',
        requestId: req.requestId,
        timestamp: new Date().toISOString()
      });
    }

    const content = fs.readFileSync(logFile, 'utf8');
    const lines = content.split('\n').filter(line => line.trim());
    const matchingLines = lines.filter(line => 
      line.toLowerCase().includes(query.toLowerCase())
    ).slice(-parsedLimit);

    res.json({
      success: true,
      data: {
        query: query,
        type: type,
        matches: matchingLines,
        totalMatches: matchingLines.length,
        limit: parsedLimit
      },
      requestId: req.requestId,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Error searching logs:', error);
    res.status(500).json({
      success: false,
      message: 'Internal server error',
      requestId: req.requestId,
      timestamp: new Date().toISOString()
    });
  }
});

// POST /api/logs - Allow unauthenticated log sink (no-op storage), to avoid client errors
router.post('/', (req, res) => {
  // Accept payload, but do not persist to avoid abuse. Respond 204 to indicate accepted.
  res.status(204).end();
});

module.exports = router;
