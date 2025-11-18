/**
 * Task Versioning Service
 * Manages task history and versioning
 */
const mongoose = require('mongoose');
const logger = require('../utils/logger');

const TaskVersionSchema = new mongoose.Schema({
  taskId: { type: mongoose.Schema.Types.ObjectId, required: true, index: true },
  version: { type: Number, required: true },
  userId: { type: mongoose.Schema.Types.ObjectId, required: true, index: true },
  changes: {
    title: { type: String },
    description: { type: String },
    status: { type: String },
    priority: { type: String },
    dueDate: { type: Date },
    tags: [String],
    metadata: mongoose.Schema.Types.Mixed
  },
  changeType: {
    type: String,
    enum: ['create', 'update', 'delete', 'restore'],
    required: true
  },
  changedBy: { type: mongoose.Schema.Types.ObjectId, required: true },
  changeReason: { type: String },
  snapshot: mongoose.Schema.Types.Mixed, // Full task snapshot
  createdAt: { type: Date, default: Date.now }
}, {
  timestamps: false
});

TaskVersionSchema.index({ taskId: 1, version: 1 }, { unique: true });

const TaskVersion = mongoose.model('TaskVersion', TaskVersionSchema);

class TaskVersioningService {
  /**
   * Create initial version
   */
  async createInitialVersion(taskId, userId, taskData) {
    try {
      const version = new TaskVersion({
        taskId,
        version: 1,
        userId,
        changes: taskData,
        changeType: 'create',
        changedBy: userId,
        snapshot: taskData
      });

      await version.save();
      logger.info('Initial task version created', { taskId, version: 1 });
      return version;
    } catch (error) {
      logger.error('Error creating initial version:', error);
      throw error;
    }
  }

  /**
   * Create new version from changes
   */
  async createVersion(taskId, userId, changes, changeReason = null) {
    try {
      // Get current version
      const currentVersion = await TaskVersion.findOne({ taskId })
        .sort({ version: -1 })
        .limit(1);

      const newVersionNumber = currentVersion ? currentVersion.version + 1 : 1;

      // Get current snapshot
      const currentSnapshot = currentVersion?.snapshot || {};

      // Merge changes into snapshot
      const newSnapshot = { ...currentSnapshot, ...changes };

      const version = new TaskVersion({
        taskId,
        version: newVersionNumber,
        userId,
        changes,
        changeType: 'update',
        changedBy: userId,
        changeReason,
        snapshot: newSnapshot
      });

      await version.save();
      logger.info('Task version created', { taskId, version: newVersionNumber });
      return version;
    } catch (error) {
      logger.error('Error creating version:', error);
      throw error;
    }
  }

  /**
   * Get version history
   */
  async getVersionHistory(taskId, limit = 50) {
    try {
      const versions = await TaskVersion.find({ taskId })
        .sort({ version: -1 })
        .limit(limit)
        .populate('changedBy', 'name email')
        .select('version changes changeType changedBy changeReason createdAt snapshot');

      return versions;
    } catch (error) {
      logger.error('Error getting version history:', error);
      throw error;
    }
  }

  /**
   * Get specific version
   */
  async getVersion(taskId, versionNumber) {
    try {
      const version = await TaskVersion.findOne({ taskId, version: versionNumber })
        .populate('changedBy', 'name email');

      if (!version) {
        throw new Error(`Version ${versionNumber} not found for task ${taskId}`);
      }

      return version;
    } catch (error) {
      logger.error('Error getting version:', error);
      throw error;
    }
  }

  /**
   * Restore to specific version
   */
  async restoreToVersion(taskId, versionNumber, userId) {
    try {
      const version = await this.getVersion(taskId, versionNumber);
      
      // Create new version with restore change type
      const restoreVersion = new TaskVersion({
        taskId,
        version: (await this.getCurrentVersion(taskId)) + 1,
        userId,
        changes: version.snapshot,
        changeType: 'restore',
        changedBy: userId,
        changeReason: `Restored to version ${versionNumber}`,
        snapshot: version.snapshot
      });

      await restoreVersion.save();
      logger.info('Task restored to version', { taskId, version: versionNumber });
      return restoreVersion;
    } catch (error) {
      logger.error('Error restoring version:', error);
      throw error;
    }
  }

  /**
   * Get current version number
   */
  async getCurrentVersion(taskId) {
    try {
      const latest = await TaskVersion.findOne({ taskId })
        .sort({ version: -1 })
        .limit(1)
        .select('version');

      return latest ? latest.version : 0;
    } catch (error) {
      logger.error('Error getting current version:', error);
      throw error;
    }
  }

  /**
   * Compare two versions
   */
  async compareVersions(taskId, version1, version2) {
    try {
      const v1 = await this.getVersion(taskId, version1);
      const v2 = await this.getVersion(taskId, version2);

      const diff = {
        added: {},
        removed: {},
        modified: {}
      };

      const snapshot1 = v1.snapshot || {};
      const snapshot2 = v2.snapshot || {};

      // Find added and modified fields
      Object.keys(snapshot2).forEach(key => {
        if (!(key in snapshot1)) {
          diff.added[key] = snapshot2[key];
        } else if (JSON.stringify(snapshot1[key]) !== JSON.stringify(snapshot2[key])) {
          diff.modified[key] = {
            old: snapshot1[key],
            new: snapshot2[key]
          };
        }
      });

      // Find removed fields
      Object.keys(snapshot1).forEach(key => {
        if (!(key in snapshot2)) {
          diff.removed[key] = snapshot1[key];
        }
      });

      return {
        version1: v1.version,
        version2: v2.version,
        diff,
        timeDiff: v2.createdAt - v1.createdAt
      };
    } catch (error) {
      logger.error('Error comparing versions:', error);
      throw error;
    }
  }

  /**
   * Get version statistics
   */
  async getVersionStats(taskId) {
    try {
      const stats = await TaskVersion.aggregate([
        { $match: { taskId: new mongoose.Types.ObjectId(taskId) } },
        {
          $group: {
            _id: null,
            totalVersions: { $sum: 1 },
            creates: { $sum: { $cond: [{ $eq: ['$changeType', 'create'] }, 1, 0] } },
            updates: { $sum: { $cond: [{ $eq: ['$changeType', 'update'] }, 1, 0] } },
            restores: { $sum: { $cond: [{ $eq: ['$changeType', 'restore'] }, 1, 0] } },
            firstVersion: { $min: '$version' },
            lastVersion: { $max: '$version' },
            firstChange: { $min: '$createdAt' },
            lastChange: { $max: '$createdAt' }
          }
        }
      ]);

      return stats[0] || {
        totalVersions: 0,
        creates: 0,
        updates: 0,
        restores: 0
      };
    } catch (error) {
      logger.error('Error getting version stats:', error);
      throw error;
    }
  }
}

module.exports = new TaskVersioningService();

