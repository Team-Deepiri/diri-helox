const express = require('express');
const router = express.Router();

const taskVersioningService = require('./taskVersioningService');
const dependencyGraphService = require('./dependencyGraphService');

// Check if services export instances or classes
const taskVersioning = taskVersioningService.TaskVersioningService 
  ? new taskVersioningService.TaskVersioningService() 
  : taskVersioningService;
const dependencyGraph = dependencyGraphService.DependencyGraphService
  ? new dependencyGraphService.DependencyGraphService()
  : dependencyGraphService;

// Task routes
router.get('/tasks', (req, res) => taskVersioning.getTasks(req, res));
router.post('/tasks', (req, res) => taskVersioning.createTask(req, res));
router.put('/tasks/:id', (req, res) => taskVersioning.updateTask(req, res));
router.get('/tasks/:id/versions', (req, res) => taskVersioning.getVersions(req, res));

// Dependency routes
router.get('/dependencies/:taskId', (req, res) => dependencyGraph.getDependencies(req, res));
router.post('/dependencies', (req, res) => dependencyGraph.addDependency(req, res));

module.exports = router;

