const express = require('express');
const router = express.Router();
const taskService = require('../services/taskService');
const authenticateJWT = require('../middleware/authenticateJWT');
const logger = require('../utils/logger');

// Get all tasks for user
router.get('/', authenticateJWT, async (req, res, next) => {
  try {
    const tasks = await taskService.getUserTasks(req.user.id, req.query);
    res.json({ success: true, data: tasks });
  } catch (error) {
    logger.error('Error fetching tasks:', error);
    next(error);
  }
});

// Get single task
router.get('/:id', authenticateJWT, async (req, res, next) => {
  try {
    const task = await taskService.getTaskById(req.params.id, req.user.id);
    res.json({ success: true, data: task });
  } catch (error) {
    logger.error('Error fetching task:', error);
    next(error);
  }
});

// Create new task
router.post('/', authenticateJWT, async (req, res, next) => {
  try {
    const task = await taskService.createTask(req.user.id, req.body);
    res.status(201).json({ success: true, data: task });
  } catch (error) {
    logger.error('Error creating task:', error);
    next(error);
  }
});

// Update task
router.patch('/:id', authenticateJWT, async (req, res, next) => {
  try {
    const task = await taskService.updateTask(req.params.id, req.user.id, req.body);
    res.json({ success: true, data: task });
  } catch (error) {
    logger.error('Error updating task:', error);
    next(error);
  }
});

// Delete task
router.delete('/:id', authenticateJWT, async (req, res, next) => {
  try {
    await taskService.deleteTask(req.params.id, req.user.id);
    res.json({ success: true, message: 'Task deleted successfully' });
  } catch (error) {
    logger.error('Error deleting task:', error);
    next(error);
  }
});

// Complete task
router.post('/:id/complete', authenticateJWT, async (req, res, next) => {
  try {
    const task = await taskService.completeTask(req.params.id, req.user.id, req.body);
    res.json({ success: true, data: task });
  } catch (error) {
    logger.error('Error completing task:', error);
    next(error);
  }
});

module.exports = router;

