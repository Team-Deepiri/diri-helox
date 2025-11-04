const Integration = require('../models/Integration');
const Task = require('../models/Task');
const taskService = require('./taskService');
const logger = require('../utils/logger');
const axios = require('axios');

const integrationService = {
  async connectIntegration(userId, service, credentials) {
    try {
      // Check if integration already exists
      let integration = await Integration.findOne({ userId, service });

      if (integration) {
        integration.status = 'connected';
        integration.credentials = credentials;
        integration.lastSync = new Date();
      } else {
        integration = new Integration({
          userId,
          service,
          credentials,
          status: 'connected',
          lastSync: new Date()
        });
      }

      await integration.save();

      // Perform initial sync
      await this.syncIntegration(userId, service);

      logger.info(`Integration connected: ${service} for user: ${userId}`);
      return integration;
    } catch (error) {
      logger.error('Error connecting integration:', error);
      throw error;
    }
  },

  async disconnectIntegration(userId, service) {
    try {
      const integration = await Integration.findOne({ userId, service });
      if (!integration) {
        throw new Error('Integration not found');
      }

      integration.status = 'disconnected';
      integration.credentials = {};
      await integration.save();

      logger.info(`Integration disconnected: ${service} for user: ${userId}`);
      return integration;
    } catch (error) {
      logger.error('Error disconnecting integration:', error);
      throw error;
    }
  },

  async getUserIntegrations(userId) {
    try {
      const integrations = await Integration.find({ userId });
      return integrations;
    } catch (error) {
      logger.error('Error fetching integrations:', error);
      throw error;
    }
  },

  async syncIntegration(userId, service) {
    try {
      const integration = await Integration.findOne({ userId, service });
      if (!integration || integration.status !== 'connected') {
        throw new Error('Integration not found or not connected');
      }

      integration.status = 'syncing';
      await integration.save();

      const tasks = await this.fetchTasksFromService(integration);
      
      // Create tasks from fetched data
      const createdTasks = [];
      for (const taskData of tasks) {
        try {
          const task = await taskService.createTask(userId, {
            ...taskData,
            type: service,
            metadata: {
              sourceId: taskData.id,
              sourceUrl: taskData.url,
              sourceData: taskData
            }
          });
          createdTasks.push(task);
        } catch (error) {
          logger.error('Error creating task from integration:', error);
        }
      }

      integration.status = 'connected';
      integration.lastSync = new Date();
      integration.syncStats.totalTasksSynced += createdTasks.length;
      integration.syncStats.lastSyncSuccess = true;
      await integration.save();

      logger.info(`Synced ${createdTasks.length} tasks from ${service} for user: ${userId}`);
      return { tasks: createdTasks, count: createdTasks.length };
    } catch (error) {
      logger.error('Error syncing integration:', error);
      
      // Update integration status
      const integration = await Integration.findOne({ userId, service });
      if (integration) {
        integration.status = 'error';
        integration.syncStats.lastSyncSuccess = false;
        integration.syncStats.lastSyncError = error.message;
        await integration.save();
      }
      
      throw error;
    }
  },

  async fetchTasksFromService(integration) {
    try {
      switch (integration.service) {
        case 'notion':
          return await this.fetchNotionTasks(integration);
        case 'trello':
          return await this.fetchTrelloTasks(integration);
        case 'github':
          return await this.fetchGithubTasks(integration);
        case 'google_docs':
          return await this.fetchGoogleDocsTasks(integration);
        default:
          throw new Error(`Unsupported service: ${integration.service}`);
      }
    } catch (error) {
      logger.error(`Error fetching tasks from ${integration.service}:`, error);
      throw error;
    }
  },

  async fetchNotionTasks(integration) {
    // Placeholder - implement Notion API integration
    // This would use the Notion API to fetch pages/databases
    logger.info('Notion integration not yet implemented');
    return [];
  },

  async fetchTrelloTasks(integration) {
    // Placeholder - implement Trello API integration
    // This would use the Trello API to fetch cards
    logger.info('Trello integration not yet implemented');
    return [];
  },

  async fetchGithubTasks(integration) {
    // Placeholder - implement GitHub API integration
    // This would use the GitHub API to fetch issues/pull requests
    logger.info('GitHub integration not yet implemented');
    return [];
  },

  async fetchGoogleDocsTasks(integration) {
    // Placeholder - implement Google Docs API integration
    logger.info('Google Docs integration not yet implemented');
    return [];
  },

  async syncAllIntegrations(userId) {
    try {
      const integrations = await Integration.find({ 
        userId, 
        status: 'connected',
        'configuration.autoSync': true
      });

      const results = [];
      for (const integration of integrations) {
        try {
          const result = await this.syncIntegration(userId, integration.service);
          results.push({ service: integration.service, ...result });
        } catch (error) {
          results.push({ 
            service: integration.service, 
            error: error.message 
          });
        }
      }

      return results;
    } catch (error) {
      logger.error('Error syncing all integrations:', error);
      throw error;
    }
  }
};

module.exports = integrationService;

