const express = require('express');
const Notification = require('../models/Notification');
const logger = require('../utils/logger');

const router = express.Router();

// Get user notifications
router.get('/', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { limit = 50, offset = 0, unreadOnly = false } = req.query;

    let query = { userId };
    if (unreadOnly === 'true') {
      query.status = { $in: ['pending', 'sent', 'delivered'] };
    }

    const notifications = await Notification.findForUser(userId, parseInt(limit));

    res.json({
      success: true,
      data: notifications
    });

  } catch (error) {
    logger.error('Failed to get notifications:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Mark notification as read
router.put('/:notificationId/read', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { notificationId } = req.params;

    const notification = await Notification.findOne({
      _id: notificationId,
      userId: userId
    });

    if (!notification) {
      return res.status(404).json({
        success: false,
        message: 'Notification not found'
      });
    }

    await notification.markAsRead();

    res.json({
      success: true,
      message: 'Notification marked as read',
      data: notification
    });

  } catch (error) {
    logger.error('Failed to mark notification as read:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Mark all notifications as read
router.put('/read-all', async (req, res) => {
  try {
    const userId = req.user.userId;

    await Notification.updateMany(
      { userId: userId, status: { $in: ['pending', 'sent', 'delivered'] } },
      { status: 'read', readAt: new Date() }
    );

    res.json({
      success: true,
      message: 'All notifications marked as read'
    });

  } catch (error) {
    logger.error('Failed to mark all notifications as read:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Delete notification
router.delete('/:notificationId', async (req, res) => {
  try {
    const userId = req.user.userId;
    const { notificationId } = req.params;

    const notification = await Notification.findOneAndDelete({
      _id: notificationId,
      userId: userId
    });

    if (!notification) {
      return res.status(404).json({
        success: false,
        message: 'Notification not found'
      });
    }

    res.json({
      success: true,
      message: 'Notification deleted successfully'
    });

  } catch (error) {
    logger.error('Failed to delete notification:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get notification settings (placeholder)
router.get('/settings', async (req, res) => {
  try {
    const userId = req.user.userId;

    // TODO: Implement notification settings
    const settings = {
      push: true,
      email: true,
      sms: false,
      adventureReminders: true,
      eventReminders: true,
      friendActivity: true,
      systemAnnouncements: true
    };

    res.json({
      success: true,
      data: settings
    });

  } catch (error) {
    logger.error('Failed to get notification settings:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Update notification settings (placeholder)
router.put('/settings', async (req, res) => {
  try {
    const userId = req.user.userId;
    const settings = req.body;

    // TODO: Implement notification settings update
    logger.info(`User ${userId} updated notification settings:`, settings);

    res.json({
      success: true,
      message: 'Notification settings updated successfully',
      data: settings
    });

  } catch (error) {
    logger.error('Failed to update notification settings:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get unread notification count
router.get('/unread/count', async (req, res) => {
  try {
    const userId = req.user.userId;

    const count = await Notification.countDocuments({
      userId: userId,
      status: { $in: ['pending', 'sent', 'delivered'] }
    });

    res.json({
      success: true,
      data: { count }
    });

  } catch (error) {
    logger.error('Failed to get unread notification count:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

module.exports = router;
