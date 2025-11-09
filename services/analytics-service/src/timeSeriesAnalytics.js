/**
 * Time-Series Analytics Service
 * Uses InfluxDB for time-series data collection
 */
const { InfluxDB, Point } = require('@influxdata/influxdb-client');
const logger = require('../../utils/logger');

class TimeSeriesAnalyticsService {
  constructor() {
    this.client = null;
    this.writeApi = null;
    this.queryApi = null;
    this._initialize();
  }

  _initialize() {
    try {
      const url = process.env.INFLUXDB_URL || 'http://localhost:8086';
      const token = process.env.INFLUXDB_TOKEN || '';
      const org = process.env.INFLUXDB_ORG || 'deepiri';
      const bucket = process.env.INFLUXDB_BUCKET || 'analytics';

      this.client = new InfluxDB({ url, token });
      this.writeApi = this.client.getWriteApi(org, bucket, 'ns');
      this.queryApi = this.client.getQueryApi(org);
      this.bucket = bucket;
      this.org = org;

      logger.info('InfluxDB initialized');
    } catch (error) {
      logger.error('InfluxDB initialization failed:', error);
    }
  }

  async recordMetric(userId, metric, value, tags = {}) {
    try {
      const point = new Point(metric)
        .tag('userId', userId.toString())
        .floatField('value', value);

      Object.keys(tags).forEach(key => {
        point.tag(key, tags[key].toString());
      });

      this.writeApi.writePoint(point);
      await this.writeApi.flush();

      logger.debug('Metric recorded', { userId, metric, value });
    } catch (error) {
      logger.error('Error recording metric:', error);
      throw error;
    }
  }

  async queryMetrics(userId, metric, startTime, endTime) {
    try {
      const query = `
        from(bucket: "${this.bucket}")
          |> range(start: ${startTime.toISOString()}, stop: ${endTime.toISOString()})
          |> filter(fn: (r) => r._measurement == "${metric}")
          |> filter(fn: (r) => r.userId == "${userId}")
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
      `;

      const results = [];
      await this.queryApi.collectRows(query, (row, tableMeta) => {
        results.push({
          time: row._time,
          value: row._value,
          field: row._field
        });
      });

      return results;
    } catch (error) {
      logger.error('Error querying metrics:', error);
      throw error;
    }
  }
}

module.exports = new TimeSeriesAnalyticsService();

