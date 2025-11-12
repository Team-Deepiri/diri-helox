/**
 * Behavioral Clustering Service
 * K-means + DBSCAN for user behavior clustering
 */
const logger = require('../utils/logger');

class BehavioralClusteringService {
  constructor() {
    this.clusters = new Map();
  }

  /**
   * K-means clustering
   */
  async kMeansClustering(dataPoints, k = 5) {
    try {
      // Simple K-means implementation
      const centroids = this._initializeCentroids(dataPoints, k);
      let clusters = [];
      let changed = true;
      let iterations = 0;
      const maxIterations = 100;

      while (changed && iterations < maxIterations) {
        clusters = this._assignToClusters(dataPoints, centroids);
        const newCentroids = this._calculateCentroids(clusters);
        
        changed = this._centroidsChanged(centroids, newCentroids);
        centroids.splice(0, centroids.length, ...newCentroids);
        iterations++;
      }

      return {
        clusters,
        centroids,
        iterations
      };
    } catch (error) {
      logger.error('Error in K-means clustering:', error);
      throw error;
    }
  }

  /**
   * DBSCAN clustering
   */
  async dbscanClustering(dataPoints, eps = 0.5, minPoints = 3) {
    try {
      const visited = new Set();
      const clusters = [];
      let clusterId = 0;

      for (let i = 0; i < dataPoints.length; i++) {
        if (visited.has(i)) continue;

        visited.add(i);
        const neighbors = this._getNeighbors(dataPoints, i, eps);

        if (neighbors.length < minPoints) {
          // Noise point
          continue;
        }

        // New cluster
        const cluster = [dataPoints[i]];
        clusterId++;

        // Expand cluster
        for (let j = 0; j < neighbors.length; j++) {
          const neighborIdx = neighbors[j];
          if (!visited.has(neighborIdx)) {
            visited.add(neighborIdx);
            const neighborNeighbors = this._getNeighbors(dataPoints, neighborIdx, eps);
            if (neighborNeighbors.length >= minPoints) {
              neighbors.push(...neighborNeighbors);
            }
          }
          cluster.push(dataPoints[neighborIdx]);
        }

        clusters.push({
          id: clusterId,
          points: cluster
        });
      }

      return {
        clusters,
        noisePoints: dataPoints.filter((_, i) => !visited.has(i))
      };
    } catch (error) {
      logger.error('Error in DBSCAN clustering:', error);
      throw error;
    }
  }

  _initializeCentroids(dataPoints, k) {
    const centroids = [];
    const indices = new Set();
    
    while (centroids.length < k && indices.size < dataPoints.length) {
      const idx = Math.floor(Math.random() * dataPoints.length);
      if (!indices.has(idx)) {
        indices.add(idx);
        centroids.push([...dataPoints[idx]]);
      }
    }
    
    return centroids;
  }

  _assignToClusters(dataPoints, centroids) {
    const clusters = centroids.map(() => []);
    
    dataPoints.forEach(point => {
      let minDist = Infinity;
      let clusterIdx = 0;
      
      centroids.forEach((centroid, idx) => {
        const dist = this._euclideanDistance(point, centroid);
        if (dist < minDist) {
          minDist = dist;
          clusterIdx = idx;
        }
      });
      
      clusters[clusterIdx].push(point);
    });
    
    return clusters;
  }

  _calculateCentroids(clusters) {
    return clusters.map(cluster => {
      if (cluster.length === 0) return null;
      
      const dimensions = cluster[0].length;
      const centroid = new Array(dimensions).fill(0);
      
      cluster.forEach(point => {
        point.forEach((val, idx) => {
          centroid[idx] += val;
        });
      });
      
      return centroid.map(sum => sum / cluster.length);
    }).filter(c => c !== null);
  }

  _centroidsChanged(oldCentroids, newCentroids) {
    if (oldCentroids.length !== newCentroids.length) return true;
    
    for (let i = 0; i < oldCentroids.length; i++) {
      const dist = this._euclideanDistance(oldCentroids[i], newCentroids[i]);
      if (dist > 0.001) return true;
    }
    
    return false;
  }

  _getNeighbors(dataPoints, pointIdx, eps) {
    const neighbors = [];
    const point = dataPoints[pointIdx];
    
    dataPoints.forEach((otherPoint, idx) => {
      if (idx !== pointIdx) {
        const dist = this._euclideanDistance(point, otherPoint);
        if (dist <= eps) {
          neighbors.push(idx);
        }
      }
    });
    
    return neighbors;
  }

  _euclideanDistance(point1, point2) {
    let sum = 0;
    for (let i = 0; i < point1.length; i++) {
      sum += Math.pow(point1[i] - point2[i], 2);
    }
    return Math.sqrt(sum);
  }
}

module.exports = new BehavioralClusteringService();

