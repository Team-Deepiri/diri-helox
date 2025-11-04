const express = require('express');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const Joi = require('joi');
const userService = require('../services/userService');
const logger = require('../utils/logger');
const crypto = require('crypto');
const cookieParser = require('cookie-parser');
const AuditLog = require('../models/AuditLog');

const router = express.Router();
router.use(cookieParser());

// Validation schemas
const registerSchema = Joi.object({
  name: Joi.string().min(2).max(100).required(),
  email: Joi.string().email().required(),
  password: Joi.string().min(6).required(),
  preferences: Joi.object({
    interests: Joi.array().items(Joi.string()).optional(),
    skillLevel: Joi.string().valid('beginner', 'intermediate', 'advanced').optional(),
    maxDistance: Joi.number().min(1000).max(20000).optional(),
    preferredDuration: Joi.number().min(30).max(90).optional(),
    socialMode: Joi.string().valid('solo', 'friends', 'meet_new_people').optional(),
    budget: Joi.string().valid('low', 'medium', 'high').optional(),
    timePreferences: Joi.object({
      morning: Joi.boolean().optional(),
      afternoon: Joi.boolean().optional(),
      evening: Joi.boolean().optional(),
      night: Joi.boolean().optional()
    }).optional()
  }).optional()
});

const loginSchema = Joi.object({
  email: Joi.string().email().required(),
  password: Joi.string().required()
});

// Register new user
router.post('/register', async (req, res) => {
  try {
    // Validate request data
    const { error, value } = registerSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: error.details.map(detail => detail.message)
      });
    }

    // Create user
    const user = await userService.createUser(value);

    // Generate JWT token
    const token = jwt.sign(
      { 
        userId: user._id,
        email: user.email,
        roles: user.roles
      },
      process.env.JWT_SECRET,
      { expiresIn: process.env.JWT_EXPIRES_IN || '7d' }
    );

    // Issue refresh token
    const refreshToken = crypto.randomBytes(32).toString('hex');
    const rtTtlDays = parseInt(process.env.REFRESH_TOKEN_TTL_DAYS || '30');
    const expiresAt = new Date(Date.now() + rtTtlDays * 24 * 60 * 60 * 1000);
    user.refreshTokens.push({ token: refreshToken, expiresAt });
    await user.save();

    res.cookie('refresh_token', refreshToken, {
      httpOnly: true,
      sameSite: 'lax',
      secure: process.env.NODE_ENV === 'production',
      expires: expiresAt
    });

    logger.info(`User registered: ${user.email}`);
    try { await AuditLog.create({ userId: user._id, action: 'register', ip: req.ip, userAgent: req.get('User-Agent') }); } catch {}

    res.status(201).json({
      success: true,
      message: 'User registered successfully',
      data: {
        user: user.getPublicProfile(),
        token
      }
    });

  } catch (error) {
    logger.error('Registration failed:', error);
    res.status(400).json({
      success: false,
      message: error.message
    });
  }
});

// Login user
router.post('/login', async (req, res) => {
  try {
    // Validate request data
    const { error, value } = loginSchema.validate(req.body);
    if (error) {
      return res.status(400).json({
        success: false,
        message: 'Validation error',
        errors: error.details.map(detail => detail.message)
      });
    }

    // Get user by email
    const user = await userService.getUserByEmail(value.email);

    // Check password
    const isPasswordValid = await user.comparePassword(value.password);
    if (!isPasswordValid) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials'
      });
    }

    // Generate JWT token
    const token = jwt.sign(
      { 
        userId: user._id,
        email: user.email 
      },
      process.env.JWT_SECRET,
      { expiresIn: process.env.JWT_EXPIRES_IN || '7d' }
    );

    logger.info(`User logged in: ${user.email}`);
    try { await AuditLog.create({ userId: user._id, action: 'login', ip: req.ip, userAgent: req.get('User-Agent') }); } catch {}

    res.json({
      success: true,
      message: 'Login successful',
      data: {
        user: user.getPublicProfile(),
        token
      }
    });

  } catch (error) {
    logger.error('Login failed:', error);
    res.status(401).json({
      success: false,
      message: 'Invalid credentials'
    });
  }
});

// Verify token
router.get('/verify', async (req, res) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'No token provided'
      });
    }

    // Verify token
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    
    // Get user data
    const user = await userService.getUserById(decoded.userId);

    res.json({
      success: true,
      message: 'Token is valid',
      data: {
        user: user.getPublicProfile()
      }
    });

  } catch (error) {
    logger.error('Token verification failed:', error);
    res.status(401).json({
      success: false,
      message: 'Invalid token'
    });
  }
});

// Refresh token
router.post('/refresh', async (req, res) => {
  try {
    const token = req.cookies?.refresh_token;
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'No refresh token'
      });
    }
    const user = await userService.findByRefreshToken(token);
    if (!user) {
      return res.status(401).json({ success: false, message: 'Invalid refresh token' });
    }

    // Generate new token
    const newToken = jwt.sign(
      { 
        userId: user._id,
        email: user.email,
        roles: user.roles
      },
      process.env.JWT_SECRET,
      { expiresIn: process.env.JWT_EXPIRES_IN || '7d' }
    );

    res.json({
      success: true,
      message: 'Token refreshed successfully',
      data: {
        user: user.getPublicProfile(),
        token: newToken
      }
    });

  } catch (error) {
    logger.error('Token refresh failed:', error);
    res.status(401).json({
      success: false,
      message: 'Invalid token'
    });
  }
});

// Forgot password (placeholder - would need email service)
router.post('/forgot-password', async (req, res) => {
  try {
    const { email } = req.body;
    
    if (!email) {
      return res.status(400).json({
        success: false,
        message: 'Email is required'
      });
    }

    // Check if user exists
    try {
      await userService.getUserByEmail(email);
      
      // TODO: Send password reset email
      logger.info(`Password reset requested for: ${email}`);
      
      res.json({
        success: true,
        message: 'Password reset email sent (if user exists)'
      });
    } catch (error) {
      // Don't reveal if user exists or not
      res.json({
        success: true,
        message: 'Password reset email sent (if user exists)'
      });
    }

  } catch (error) {
    logger.error('Forgot password failed:', error);
    res.status(500).json({
      success: false,
      message: 'Internal server error'
    });
  }
});

// Reset password (placeholder - would need email service)
router.post('/reset-password', async (req, res) => {
  try {
    const { token, newPassword } = req.body;
    
    if (!token || !newPassword) {
      return res.status(400).json({
        success: false,
        message: 'Token and new password are required'
      });
    }

    // TODO: Verify reset token and update password
    logger.info('Password reset attempted');
    
    res.json({
      success: true,
      message: 'Password reset successfully'
    });

  } catch (error) {
    logger.error('Password reset failed:', error);
    res.status(400).json({
      success: false,
      message: 'Invalid or expired reset token'
    });
  }
});

// Logout (client-side token removal)
router.post('/logout', async (req, res) => {
  const token = req.cookies?.refresh_token;
  if (token) await userService.revokeRefreshToken(token);
  res.clearCookie('refresh_token');
  res.json({ success: true, message: 'Logged out successfully' });
  try { await AuditLog.create({ action: 'logout', ip: req.ip, userAgent: req.get('User-Agent') }); } catch {}
});

module.exports = router;
