/**
 * Task Dependency Graph Service
 * Manages task dependencies and dependency graphs
 */
const mongoose = require('mongoose');
const logger = require('../../utils/logger');

const TaskDependencySchema = new mongoose.Schema({
  taskId: { type: mongoose.Schema.Types.ObjectId, required: true, index: true },
  dependsOn: { type: mongoose.Schema.Types.ObjectId, required: true, index: true },
  dependencyType: {
    type: String,
    enum: ['blocks', 'precedes', 'related', 'subtask'],
    default: 'blocks'
  },
  userId: { type: mongoose.Schema.Types.ObjectId, required: true, index: true },
  createdAt: { type: Date, default: Date.now }
}, {
  timestamps: false
});

TaskDependencySchema.index({ taskId: 1, dependsOn: 1 }, { unique: true });

const TaskDependency = mongoose.model('TaskDependency', TaskDependencySchema);

class DependencyGraphService {
  /**
   * Add dependency
   */
  async addDependency(taskId, dependsOn, dependencyType = 'blocks', userId) {
    try {
      // Check for circular dependencies
      if (await this._wouldCreateCycle(taskId, dependsOn)) {
        throw new Error('Circular dependency detected');
      }

      // Check if dependency already exists
      const existing = await TaskDependency.findOne({ taskId, dependsOn });
      if (existing) {
        return existing;
      }

      const dependency = new TaskDependency({
        taskId,
        dependsOn,
        dependencyType,
        userId
      });

      await dependency.save();
      logger.info('Task dependency added', { taskId, dependsOn, dependencyType });
      return dependency;
    } catch (error) {
      logger.error('Error adding dependency:', error);
      throw error;
    }
  }

  /**
   * Remove dependency
   */
  async removeDependency(taskId, dependsOn) {
    try {
      const result = await TaskDependency.findOneAndDelete({ taskId, dependsOn });
      if (result) {
        logger.info('Task dependency removed', { taskId, dependsOn });
      }
      return result;
    } catch (error) {
      logger.error('Error removing dependency:', error);
      throw error;
    }
  }

  /**
   * Get dependencies for a task
   */
  async getDependencies(taskId) {
    try {
      const dependencies = await TaskDependency.find({ taskId })
        .populate('dependsOn', 'title status priority dueDate')
        .select('dependsOn dependencyType createdAt');

      return dependencies.map(dep => ({
        task: dep.dependsOn,
        type: dep.dependencyType,
        createdAt: dep.createdAt
      }));
    } catch (error) {
      logger.error('Error getting dependencies:', error);
      throw error;
    }
  }

  /**
   * Get dependents (tasks that depend on this task)
   */
  async getDependents(taskId) {
    try {
      const dependents = await TaskDependency.find({ dependsOn: taskId })
        .populate('taskId', 'title status priority dueDate')
        .select('taskId dependencyType createdAt');

      return dependents.map(dep => ({
        task: dep.taskId,
        type: dep.dependencyType,
        createdAt: dep.createdAt
      }));
    } catch (error) {
      logger.error('Error getting dependents:', error);
      throw error;
    }
  }

  /**
   * Get full dependency graph
   */
  async getDependencyGraph(taskIds) {
    try {
      const graph = {
        nodes: [],
        edges: []
      };

      const taskIds = taskIds.map(id => new mongoose.Types.ObjectId(id));
      const dependencies = await TaskDependency.find({
        $or: [
          { taskId: { $in: taskIds } },
          { dependsOn: { $in: taskIds } }
        ]
      })
        .populate('taskId', 'title status')
        .populate('dependsOn', 'title status');

      const nodeSet = new Set();
      
      dependencies.forEach(dep => {
        const taskId = dep.taskId._id.toString();
        const dependsOnId = dep.dependsOn._id.toString();

        if (!nodeSet.has(taskId)) {
          graph.nodes.push({
            id: taskId,
            title: dep.taskId.title,
            status: dep.taskId.status
          });
          nodeSet.add(taskId);
        }

        if (!nodeSet.has(dependsOnId)) {
          graph.nodes.push({
            id: dependsOnId,
            title: dep.dependsOn.title,
            status: dep.dependsOn.status
          });
          nodeSet.add(dependsOnId);
        }

        graph.edges.push({
          from: dependsOnId,
          to: taskId,
          type: dep.dependencyType
        });
      });

      return graph;
    } catch (error) {
      logger.error('Error getting dependency graph:', error);
      throw error;
    }
  }

  /**
   * Get execution order (topological sort)
   */
  async getExecutionOrder(taskIds) {
    try {
      const dependencies = await TaskDependency.find({
        taskId: { $in: taskIds.map(id => new mongoose.Types.ObjectId(id)) }
      });

      // Build adjacency list
      const graph = {};
      const inDegree = {};
      
      taskIds.forEach(id => {
        graph[id] = [];
        inDegree[id] = 0;
      });

      dependencies.forEach(dep => {
        const taskId = dep.taskId.toString();
        const dependsOn = dep.dependsOn.toString();
        
        if (taskIds.includes(dependsOn)) {
          graph[dependsOn].push(taskId);
          inDegree[taskId]++;
        }
      });

      // Topological sort
      const queue = [];
      const result = [];

      Object.keys(inDegree).forEach(id => {
        if (inDegree[id] === 0) {
          queue.push(id);
        }
      });

      while (queue.length > 0) {
        const current = queue.shift();
        result.push(current);

        graph[current].forEach(neighbor => {
          inDegree[neighbor]--;
          if (inDegree[neighbor] === 0) {
            queue.push(neighbor);
          }
        });
      }

      // Check for cycles
      if (result.length !== taskIds.length) {
        throw new Error('Circular dependency detected in task set');
      }

      return result;
    } catch (error) {
      logger.error('Error getting execution order:', error);
      throw error;
    }
  }

  /**
   * Check if adding dependency would create cycle
   */
  async _wouldCreateCycle(taskId, dependsOn) {
    try {
      // If dependsOn depends on taskId, it would create a cycle
      const reverseDependency = await TaskDependency.findOne({
        taskId: dependsOn,
        dependsOn: taskId
      });

      if (reverseDependency) {
        return true;
      }

      // Check transitive dependencies
      const visited = new Set();
      const stack = [dependsOn];

      while (stack.length > 0) {
        const current = stack.pop();
        
        if (current.toString() === taskId.toString()) {
          return true; // Cycle detected
        }

        if (visited.has(current.toString())) {
          continue;
        }

        visited.add(current.toString());

        const deps = await TaskDependency.find({ taskId: current })
          .select('dependsOn');
        
        deps.forEach(dep => {
          stack.push(dep.dependsOn);
        });
      }

      return false;
    } catch (error) {
      logger.error('Error checking for cycles:', error);
      return true; // Assume cycle to be safe
    }
  }

  /**
   * Get blocking tasks (tasks that block this task)
   */
  async getBlockingTasks(taskId) {
    try {
      const blocking = await TaskDependency.find({
        taskId,
        dependencyType: 'blocks'
      })
        .populate('dependsOn', 'title status priority dueDate completedAt')
        .select('dependsOn');

      return blocking
        .filter(dep => dep.dependsOn.status !== 'completed')
        .map(dep => dep.dependsOn);
    } catch (error) {
      logger.error('Error getting blocking tasks:', error);
      throw error;
    }
  }

  /**
   * Check if task can be started (all dependencies completed)
   */
  async canStartTask(taskId) {
    try {
      const blocking = await this.getBlockingTasks(taskId);
      return blocking.length === 0;
    } catch (error) {
      logger.error('Error checking if task can start:', error);
      return false;
    }
  }
}

module.exports = new DependencyGraphService();

