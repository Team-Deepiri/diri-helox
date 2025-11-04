/**
 * Comprehensive logging system for the React frontend
 */
class Logger {
  constructor(service = 'frontend') {
    this.service = service;
    this.logLevel = process.env.NODE_ENV === 'production' ? 'warn' : 'debug';
    this.levels = {
      error: 0,
      warn: 1,
      info: 2,
      debug: 3
    };
  }

  shouldLog(level) {
    return this.levels[level] <= this.levels[this.logLevel];
  }

  formatMessage(level, message, context = {}) {
    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      level,
      service: this.service,
      message,
      context: {
        userAgent: navigator.userAgent,
        url: window.location.href,
        ...context
      }
    };

    // Add user context if available
    const user = this.getUserContext();
    if (user) {
      logEntry.context.user = user;
    }

    return logEntry;
  }

  getUserContext() {
    try {
      const token = localStorage.getItem('token');
      if (token) {
        // Decode JWT token to get user info (without verification for logging)
        const payload = JSON.parse(atob(token.split('.')[1]));
        return {
          userId: payload.userId,
          email: payload.email
        };
      }
    } catch (error) {
      // Ignore token parsing errors
    }
    return null;
  }

  log(level, message, context = {}) {
    if (!this.shouldLog(level)) return;

    const logEntry = this.formatMessage(level, message, context);

    // Console logging
    if (process.env.NODE_ENV !== 'production') {
      const consoleMethod = level === 'error' ? 'error' : 
                           level === 'warn' ? 'warn' : 
                           level === 'info' ? 'info' : 'log';
      console[consoleMethod](`[${level.toUpperCase()}] ${message}`, logEntry);
    }

    // Send to logging service in production
    if (process.env.NODE_ENV === 'production') {
      this.sendToLoggingService(logEntry);
    }

    // Store critical errors in localStorage for debugging
    if (level === 'error' && process.env.NODE_ENV === 'development') {
      this.storeErrorLog(logEntry);
    }
  }

  async sendToLoggingService(logEntry) {
    try {
      const raw = localStorage.getItem('token');
      const token = raw && raw !== 'null' && raw !== 'undefined' ? raw : null;
      if (!token) {
        return; // skip sending logs without auth to avoid 401 noise
      }
      // Send to backend logging endpoint
      await fetch('/api/logs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(logEntry)
      });
    } catch (error) {
      // Fallback to console if logging service fails
      console.error('Failed to send log to service:', error);
    }
  }

  storeErrorLog(logEntry) {
    try {
      const errorLogs = JSON.parse(localStorage.getItem('errorLogs') || '[]');
      errorLogs.push(logEntry);
      
      // Keep only last 10 error logs
      if (errorLogs.length > 10) {
        errorLogs.splice(0, errorLogs.length - 10);
      }
      
      localStorage.setItem('errorLogs', JSON.stringify(errorLogs));
    } catch (error) {
      console.error('Failed to store error log:', error);
    }
  }

  error(message, context = {}) {
    this.log('error', message, context);
  }

  warn(message, context = {}) {
    this.log('warn', message, context);
  }

  info(message, context = {}) {
    this.log('info', message, context);
  }

  debug(message, context = {}) {
    this.log('debug', message, context);
  }
}

// Specialized loggers for different components
export class AuthLogger extends Logger {
  constructor() {
    super('auth');
  }

  logLoginAttempt(email, success = false, error = null) {
    this.info('Login attempt', {
      email,
      success,
      error: error?.message,
      timestamp: new Date().toISOString()
    });
  }

  logLogout(userId) {
    this.info('User logout', { userId });
  }

  logTokenExpiry(userId) {
    this.warn('Token expired', { userId });
  }

  logAuthError(error, context = {}) {
    this.error('Authentication error', {
      error: error.message,
      stack: error.stack,
      ...context
    });
  }
}

export class AdventureLogger extends Logger {
  constructor() {
    super('adventure');
  }

  logAdventureGeneration(userId, preferences, success = false, error = null) {
    this.info('Adventure generation', {
      userId,
      preferences,
      success,
      error: error?.message,
      duration: preferences?.duration,
      interests: preferences?.interests
    });
  }

  logAdventureStart(adventureId, userId) {
    this.info('Adventure started', { adventureId, userId });
  }

  logAdventureComplete(adventureId, userId, feedback = null) {
    this.info('Adventure completed', {
      adventureId,
      userId,
      rating: feedback?.rating,
      completedSteps: feedback?.completedSteps?.length
    });
  }

  logStepUpdate(adventureId, stepIndex, action, userId) {
    this.debug('Adventure step updated', {
      adventureId,
      stepIndex,
      action,
      userId
    });
  }
}

export class EventLogger extends Logger {
  constructor() {
    super('events');
  }

  logEventView(eventId, userId) {
    this.debug('Event viewed', { eventId, userId });
  }

  logEventJoin(eventId, userId) {
    this.info('Event joined', { eventId, userId });
  }

  logEventLeave(eventId, userId) {
    this.info('Event left', { eventId, userId });
  }

  logEventCreate(eventData, userId, success = false, error = null) {
    this.info('Event created', {
      eventId: eventData.id,
      userId,
      success,
      error: error?.message,
      eventType: eventData.type,
      capacity: eventData.capacity
    });
  }
}

export class APILogger extends Logger {
  constructor() {
    super('api');
  }

  logAPIRequest(method, url, requestId, userId = null) {
    this.debug('API request', {
      method,
      url,
      requestId,
      userId,
      timestamp: new Date().toISOString()
    });
  }

  logAPIResponse(method, url, status, duration, requestId, userId = null) {
    const level = status >= 400 ? 'error' : status >= 300 ? 'warn' : 'info';
    this.log(level, 'API response', {
      method,
      url,
      status,
      duration,
      requestId,
      userId
    });
  }

  logAPIError(error, method, url, requestId, userId = null) {
    this.error('API error', {
      method,
      url,
      requestId,
      userId,
      error: error.message,
      stack: error.stack
    });
  }
}

export class PerformanceLogger extends Logger {
  constructor() {
    super('performance');
  }

  logPageLoad(page, loadTime) {
    this.info('Page load', {
      page,
      loadTime,
      timestamp: new Date().toISOString()
    });
  }

  logComponentRender(component, renderTime) {
    this.debug('Component render', {
      component,
      renderTime
    });
  }

  logAPIResponseTime(endpoint, responseTime) {
    this.debug('API response time', {
      endpoint,
      responseTime
    });
  }

  logMemoryUsage() {
    if (performance.memory) {
      this.debug('Memory usage', {
        used: performance.memory.usedJSHeapSize,
        total: performance.memory.totalJSHeapSize,
        limit: performance.memory.jsHeapSizeLimit
      });
    }
  }
}

export class ErrorLogger extends Logger {
  constructor() {
    super('errors');
  }

  logUnhandledError(error, errorInfo = null) {
    this.error('Unhandled error', {
      error: error.message,
      stack: error.stack,
      componentStack: errorInfo?.componentStack,
      timestamp: new Date().toISOString()
    });
  }

  logPromiseRejection(reason, promise) {
    this.error('Unhandled promise rejection', {
      reason: reason?.message || reason,
      stack: reason?.stack,
      timestamp: new Date().toISOString()
    });
  }

  logNetworkError(error, url, method) {
    this.error('Network error', {
      error: error.message,
      url,
      method,
      timestamp: new Date().toISOString()
    });
  }
}

// Create singleton instances
export const authLogger = new AuthLogger();
export const adventureLogger = new AdventureLogger();
export const eventLogger = new EventLogger();
export const apiLogger = new APILogger();
export const performanceLogger = new PerformanceLogger();
export const errorLogger = new ErrorLogger();

// Global error handlers
export const setupGlobalErrorHandling = () => {
  // Unhandled errors
  window.addEventListener('error', (event) => {
    errorLogger.logUnhandledError(event.error, {
      componentStack: event.error?.componentStack
    });
  });

  // Unhandled promise rejections
  window.addEventListener('unhandledrejection', (event) => {
    errorLogger.logPromiseRejection(event.reason, event.promise);
  });

  // Network errors
  window.addEventListener('error', (event) => {
    if (event.target !== window && event.target.tagName === 'IMG') {
      errorLogger.logNetworkError(
        new Error(`Failed to load image: ${event.target.src}`),
        event.target.src,
        'GET'
      );
    }
  });
};

// Performance monitoring
export const setupPerformanceMonitoring = () => {
  // Monitor page load times
  window.addEventListener('load', () => {
    const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
    performanceLogger.logPageLoad(window.location.pathname, loadTime);
  });

  // Monitor memory usage periodically
  setInterval(() => {
    performanceLogger.logMemoryUsage();
  }, 60000); // Every minute
};

// React Error Boundary logger
export const logErrorBoundary = (error, errorInfo) => {
  errorLogger.logUnhandledError(error, errorInfo);
};

// Default logger instance
export default new Logger();
