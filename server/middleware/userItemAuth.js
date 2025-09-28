const UserItem = require('../models/UserItem');
const logger = require('../utils/logger');

/**
 * Middleware to verify user owns the item they're trying to access
 * Should be used after authenticateJWT middleware
 */
const verifyItemOwnership = async (req, res, next) => {
  try {
    const { itemId } = req.params;
    const userId = req.user.userId;

    if (!itemId) {
      return res.status(400).json({
        success: false,
        message: 'Item ID is required'
      });
    }

    // Check if item exists and belongs to the authenticated user
    const item = await UserItem.findOne({ 
      _id: itemId, 
      userId: userId,
      status: { $ne: 'deleted' }
    });

    if (!item) {
      return res.status(404).json({
        success: false,
        message: 'Item not found or access denied'
      });
    }

    // Add item to request for use in route handler
    req.item = item;
    next();

  } catch (error) {
    logger.error('Item ownership verification error:', error);
    return res.status(500).json({
      success: false,
      message: 'Error verifying item ownership'
    });
  }
};

/**
 * Middleware to verify user can access shared items
 */
const verifySharedItemAccess = async (req, res, next) => {
  try {
    const { itemId } = req.params;
    const userId = req.user.userId;

    if (!itemId) {
      return res.status(400).json({
        success: false,
        message: 'Item ID is required'
      });
    }

    // Check if item exists and user has access (owner or shared with them)
    const item = await UserItem.findOne({
      _id: itemId,
      status: { $ne: 'deleted' },
      $or: [
        { userId: userId }, // User owns the item
        { 'sharing.sharedWith.userId': userId }, // Item is shared with user
        { 'metadata.isPublic': true } // Item is public
      ]
    });

    if (!item) {
      return res.status(404).json({
        success: false,
        message: 'Item not found or access denied'
      });
    }

    // Add item and access level to request
    req.item = item;
    req.isOwner = item.userId.toString() === userId;
    req.isShared = item.sharing.sharedWith.some(share => 
      share.userId.toString() === userId
    );
    req.isPublic = item.metadata.isPublic;

    next();

  } catch (error) {
    logger.error('Shared item access verification error:', error);
    return res.status(500).json({
      success: false,
      message: 'Error verifying item access'
    });
  }
};

/**
 * Middleware to check if user has edit permissions for an item
 */
const verifyEditPermission = (req, res, next) => {
  // Only owners can edit items
  if (!req.isOwner) {
    return res.status(403).json({
      success: false,
      message: 'Permission denied. Only item owner can edit.'
    });
  }
  next();
};

/**
 * Middleware to validate user ID format
 */
const validateUserId = (req, res, next) => {
  const userId = req.user?.userId;
  
  if (!userId) {
    return res.status(401).json({
      success: false,
      message: 'User ID not found in token'
    });
  }

  // Basic MongoDB ObjectId validation
  if (!/^[0-9a-fA-F]{24}$/.test(userId)) {
    return res.status(400).json({
      success: false,
      message: 'Invalid user ID format'
    });
  }

  next();
};

/**
 * Middleware to check rate limits for item operations
 */
const itemRateLimit = (maxRequests = 100, windowMs = 15 * 60 * 1000) => {
  const requests = new Map();

  return (req, res, next) => {
    const userId = req.user.userId;
    const now = Date.now();
    const windowStart = now - windowMs;

    // Clean old requests
    if (requests.has(userId)) {
      const userRequests = requests.get(userId).filter(time => time > windowStart);
      requests.set(userId, userRequests);
    }

    // Check current request count
    const currentRequests = requests.get(userId) || [];
    
    if (currentRequests.length >= maxRequests) {
      return res.status(429).json({
        success: false,
        message: 'Too many requests. Please try again later.',
        retryAfter: Math.ceil(windowMs / 1000)
      });
    }

    // Add current request
    currentRequests.push(now);
    requests.set(userId, currentRequests);

    next();
  };
};

/**
 * Middleware to log user item operations for audit
 */
const auditItemOperation = (operation) => {
  return (req, res, next) => {
    const originalSend = res.send;
    
    res.send = function(data) {
      // Log the operation after response is sent
      const logData = {
        operation,
        userId: req.user.userId,
        itemId: req.params.itemId,
        ip: req.ip,
        userAgent: req.get('User-Agent'),
        timestamp: new Date(),
        success: res.statusCode < 400
      };

      logger.info('Item operation audit:', logData);
      
      // Call original send
      originalSend.call(this, data);
    };

    next();
  };
};

module.exports = {
  verifyItemOwnership,
  verifySharedItemAccess,
  verifyEditPermission,
  validateUserId,
  itemRateLimit,
  auditItemOperation
};
