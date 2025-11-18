/**
 * Push Notification Service
 * FCM/APNS for mobile push notifications
 */
const admin = require('firebase-admin');
const logger = require('../utils/logger');

class PushNotificationService {
  constructor() {
    this.fcmInitialized = false;
    this._initializeFCM();
  }

  _initializeFCM() {
    try {
      if (process.env.FIREBASE_SERVICE_ACCOUNT) {
        const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
        admin.initializeApp({
          credential: admin.credential.cert(serviceAccount)
        });
        this.fcmInitialized = true;
        logger.info('FCM initialized');
      } else {
        logger.warn('FCM not configured - FIREBASE_SERVICE_ACCOUNT not set');
      }
    } catch (error) {
      logger.error('FCM initialization failed:', error);
    }
  }

  /**
   * Send push notification
   */
  async sendPushNotification(userId, deviceToken, notification) {
    try {
      if (!this.fcmInitialized) {
        logger.warn('FCM not initialized, skipping push notification');
        return { success: false, reason: 'FCM not initialized' };
      }

      const message = {
        token: deviceToken,
        notification: {
          title: notification.title,
          body: notification.body
        },
        data: notification.data || {},
        android: {
          priority: 'high',
          notification: {
            sound: 'default',
            channelId: 'deepiri_notifications'
          }
        },
        apns: {
          payload: {
            aps: {
              sound: 'default',
              badge: notification.badge || 0
            }
          }
        }
      };

      const response = await admin.messaging().send(message);
      logger.info('Push notification sent', { userId, messageId: response });
      
      return { success: true, messageId: response };
    } catch (error) {
      logger.error('Error sending push notification:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Send to multiple devices
   */
  async sendToMultipleDevices(deviceTokens, notification) {
    try {
      if (!this.fcmInitialized) {
        return { success: false, reason: 'FCM not initialized' };
      }

      const message = {
        notification: {
          title: notification.title,
          body: notification.body
        },
        data: notification.data || {},
        tokens: deviceTokens
      };

      const response = await admin.messaging().sendMulticast(message);
      logger.info('Multicast push notification sent', { 
        successCount: response.successCount,
        failureCount: response.failureCount 
      });

      return {
        success: true,
        successCount: response.successCount,
        failureCount: response.failureCount
      };
    } catch (error) {
      logger.error('Error sending multicast push:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Subscribe to topic
   */
  async subscribeToTopic(deviceToken, topic) {
    try {
      if (!this.fcmInitialized) {
        return { success: false };
      }

      await admin.messaging().subscribeToTopic([deviceToken], topic);
      logger.info('Device subscribed to topic', { deviceToken, topic });
      return { success: true };
    } catch (error) {
      logger.error('Error subscribing to topic:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Send to topic
   */
  async sendToTopic(topic, notification) {
    try {
      if (!this.fcmInitialized) {
        return { success: false };
      }

      const message = {
        topic,
        notification: {
          title: notification.title,
          body: notification.body
        },
        data: notification.data || {}
      };

      const response = await admin.messaging().send(message);
      logger.info('Topic notification sent', { topic, messageId: response });
      return { success: true, messageId: response };
    } catch (error) {
      logger.error('Error sending topic notification:', error);
      return { success: false, error: error.message };
    }
  }
}

module.exports = new PushNotificationService();

